// The status object's work: probe each component once a minute, fold the
// result into history and incidents, refresh the payout pot's facts every few
// minutes, and keep one snapshot for /health.json to hand out. Runs only in the
// Durable Object named "status", which holds no miners, so it answers exactly
// when a mining shard cannot. The pure rules are in status.js.

import { COMPONENTS, NAMES, NOTES, rollup, recordSample, series, trackIncident, payoutEta, judge, judgeTreasury } from "./status.js";
import { rpcList, plainPost } from "./rpc.js";
import { classifyFailure } from "./guard.js";

export const STATUS_EVERY_MS = 60_000;
const FACTS_EVERY_MS = 5 * 60_000;
const ORIGIN = "https://pool.aperture.institute";

const within = (p, ms, what) => Promise.race([
  p, new Promise((_, rej) => setTimeout(() => rej(new Error(`${what} did not answer within ${ms / 1000} s`)), ms)),
]);

// Ask one pool object for one of its routes, timed; never throws.
async function probeObject(env, name, path) {
  const t = Date.now();
  try {
    const r = await within(env.POOL.get(env.POOL.idFromName(name)).fetch(new Request(ORIGIN + path)), 12_000, name);
    const ms = Date.now() - t;
    if (!r.ok) {
      let why = `answered ${r.status}`;
      try { const b = await r.json(); if (b && b.why) why = b.why; } catch {}
      return { ok: false, ms, why };
    }
    return { ok: true, ms, body: await r.json() };
  } catch (e) {
    return { ok: false, ms: Date.now() - t, why: classifyFailure(e).why };
  }
}

async function getJSON(url) {
  const host = new URL(url).host;
  // CoinGecko answers 403 to a request with no User-Agent, which is what a
  // Worker sends by default.
  const r = await within(fetch(url, { headers: { accept: "application/json", "user-agent": "glados-pool-status/1.0 (+https://status.aperture.institute)" } }), 10_000, host);
  if (!r.ok) throw new Error(`${host} answered ${r.status}`);
  return r.json();
}

// The pot: RVN at zpool and on chain, ChangeNOW's live minimum, ETH in the
// treasury, and prices. A source that fails keeps its last good value.
async function refreshFacts(env, prev) {
  const out = { at: Date.now(), errors: [] };
  const take = async (k, f) => {
    try { out[k] = await f(); } catch (e) {
      out.errors.push(`${k}: ${e && e.message ? e.message : e}`);
      if (prev && prev[k] !== undefined) out[k] = prev[k];
    }
  };
  const rvn = env.TREASURY_RVN, evm = env.TREASURY_EVM;
  await Promise.all([
    take("zpool", async () => {
      const z = await getJSON(`https://zpool.ca/api/wallet?address=${rvn}`);
      return { unpaid: Number(z.unpaid) || 0, total: Number(z.total) || 0, paid24h: Number(z.paid24h) || 0 };
    }),
    take("rvnHeld", async () => Number((await getJSON(`https://blockbook.ravencoin.org/api/v2/address/${rvn}?details=basic`)).balance) / 1e8),
    take("cnMin", async () => Number((await getJSON("https://api.changenow.io/v2/exchange/min-amount?fromCurrency=rvn&fromNetwork=rvn&toCurrency=eth&toNetwork=hood&flow=standard")).minAmount)),
    take("prices", async () => {
      // From Cloudflare's network CoinGecko and Binance both refuse a Worker, so:
      // ETH and BTC from Coinbase (Kraken if it fails), and RVN at ChangeNOW's
      // own RVN -> ETH rate, which is what the treasury actually gets (Gate.io
      // if that quote is unavailable).
      const coinbase = async (sym) => Number((await getJSON(`https://api.coinbase.com/v2/prices/${sym}-USD/spot`)).data.amount);
      const kraken = async () => {
        const r = (await getJSON("https://api.kraken.com/0/public/Ticker?pair=ETHUSD,XBTUSD")).result;
        return { eth: Number(r.XETHZUSD.c[0]), btc: Number(r.XXBTZUSD.c[0]) };
      };
      let eth, btc, source;
      try { [eth, btc] = await Promise.all([coinbase("ETH"), coinbase("BTC")]); source = "coinbase"; }
      catch { ({ eth, btc } = await kraken()); source = "kraken"; }
      let rvn = null;
      try {
        const r = await within(fetch("https://api.changenow.io/v2/exchange/estimated-amount?fromCurrency=rvn&fromNetwork=rvn&toCurrency=eth&toNetwork=hood&flow=standard&fromAmount=1000",
          { headers: { accept: "application/json", "x-changenow-api-key": env.CHANGENOW_KEY || "" } }), 10_000, "api.changenow.io");
        const q = r.ok ? await r.json() : null;
        if (q && Number(q.toAmount) > 0) rvn = (Number(q.toAmount) / 1000) * eth;
      } catch {}
      if (!(rvn > 0)) rvn = Number((await getJSON("https://api.gateio.ws/api/v4/spot/tickers?currency_pair=RVN_USDT"))[0].last);
      if (![eth, btc, rvn].every((v) => v > 0)) throw new Error("a price came back empty");
      return { rvn, eth, btc, source };
    }),
    take("ethHeld", async () => {
      for (const url of rpcList(env)) {
        try {
          const r = await plainPost(url, { jsonrpc: "2.0", id: 1, method: "eth_getBalance", params: [evm, "latest"] }, 8_000);
          if (r && typeof r === "object" && typeof r.result === "string") return Number(BigInt(r.result)) / 1e18;
        } catch {}
      }
      throw new Error("no Robinhood Chain RPC answered");
    }),
  ]);
  return out;
}

// RVN earned per day: from zpool's running total over the last day of samples,
// or its "paid in the last 24 h" until a day of samples exists.
function rvnPerDay(samples, facts) {
  const now = samples[samples.length - 1];
  const old = samples.find((s) => now && now.at - s.at >= 20 * 3_600_000);
  if (now && old && now.total >= old.total) return (now.total - old.total) / ((now.at - old.at) / 86_400_000);
  return facts && facts.zpool ? facts.zpool.paid24h : 0;
}

export async function statusTick(env, storage) {
  const now = Date.now();
  const [cpuP, gpuP, trP] = await Promise.all([
    probeObject(env, "main", "/status"),
    env.GPU_COINS ? probeObject(env, "gpu-0", "/status") : Promise.resolve(null),
    probeObject(env, "main", "/treasury"),
  ]);
  const comps = {
    cpu: { ...judge(cpuP), ms: cpuP.ms, miners: cpuP.ok ? cpuP.body.connections ?? null : null },
    gpu: gpuP ? { ...judge(gpuP), ms: gpuP.ms, miners: gpuP.ok ? gpuP.body.connections ?? null : null }
              : { state: "operational", why: "no GPU coins configured", ms: 0, miners: null },
    payouts: { ...judgeTreasury(trP), ms: trP.ms },
  };

  const [history, incidents, prevState, notesDone, oldFacts, samples] = await Promise.all([
    storage.get("status.history"), storage.get("status.incidents"), storage.get("status.prev"),
    storage.get("status.notesDone"), storage.get("status.facts"), storage.get("status.zpoolSamples"),
  ]).then((v) => [v[0] || {}, v[1] || [], v[2] || {}, v[3] || {}, v[4] || null, v[5] || []]);

  for (const c of COMPONENTS) {
    recordSample(history, c, comps[c].state, now);
    trackIncident(incidents, c, prevState[c] || "operational", comps[c].state, comps[c].why, now);
    prevState[c] = comps[c].state;
  }

  let facts = oldFacts;
  if (!facts || now - facts.at >= FACTS_EVERY_MS) {
    facts = await refreshFacts(env, oldFacts);
    if (facts.zpool) {
      samples.push({ at: now, total: facts.zpool.total });
      while (samples.length && now - samples[0].at > 2 * 86_400_000) samples.shift();
    }
  }

  // An operator's note is resolved the first time CPU mining and payouts are
  // both back after it began.
  const notes = NOTES.map((n) => {
    const done = notesDone[n.id] || (n.resolvedAt ?? null);
    if (!done && now > n.startedAt && comps.cpu.state === "operational" && comps.payouts.state !== "outage") {
      notesDone[n.id] = now;
    }
    const at = notesDone[n.id] || done;
    return at ? { ...n, resolvedAt: at, updates: [...n.updates, { at, status: "resolved", text: "CPU mining and payouts are answering again." }] } : n;
  });

  const threshold = Math.max(Number(env.RVN_BATCH || 200), (facts && facts.cnMin) || 0);
  const perDay = rvnPerDay(samples, facts);
  const unpaid = facts && facts.zpool ? facts.zpool.unpaid : 0;
  const held = (facts && facts.rvnHeld) || 0;
  const eta = payoutEta({ rvnHeld: held, rvnUnpaid: unpaid, threshold, rvnPerDay: perDay });
  const px = (facts && facts.prices) || {};
  const ethHeld = (facts && facts.ethHeld) || 0;
  const tb = trP.ok ? trP.body : {};

  const snapshot = {
    generatedAt: now,
    overall: rollup(COMPONENTS.map((c) => comps[c].state)),
    components: Object.fromEntries(COMPONENTS.map((c) => [c, { name: NAMES[c], ...comps[c], days: series(history, c, now) }])),
    pot: {
      rvnHeld: held, rvnUnpaid: unpaid, rvnEarnedTotal: facts && facts.zpool ? facts.zpool.total : null,
      rvnPerDay: perDay, threshold, thresholdFrom: facts && facts.cnMin > Number(env.RVN_BATCH || 200) ? "ChangeNOW's minimum" : "the pool's batch",
      rvnNeeded: eta.need, etaDays: eta.days,
      ethHeld, prices: px,
      valueUsd: px.rvn ? (held + unpaid) * px.rvn + ethHeld * (px.eth || 0) : null,
      payouts: tb.epochs ?? null, phase: tb.phase ?? null,
      factsAt: facts ? facts.at : null, factErrors: facts ? facts.errors : [],
    },
    incidents: [...notes, ...incidents].sort((a, b) => b.startedAt - a.startedAt).slice(0, 30),
  };

  await storage.put({
    "status.history": history, "status.incidents": incidents, "status.prev": prevState,
    "status.notesDone": notesDone, "status.facts": facts, "status.zpoolSamples": samples, "status.snapshot": snapshot,
  });
  return snapshot;
}

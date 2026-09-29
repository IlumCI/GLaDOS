// The treasury's hands: gather what is true, ask `decide`, do the one thing.
//
// Everything that *decides* is in flow.js and tested there. This file only
// reads (balances, unspent outputs, receipts, an exchange's status, the pair's
// reserves) and writes (an exchange, a broadcast), and each write follows the
// rule flow.js depends on: **the state records a transaction before the network
// sees it**, so a restart resends those bytes rather than making new ones.
//
// Three modes, from `TREASURY`:
//   off   the default. Nothing runs.
//   dry   reads everything and logs the action it would take; moves and
//         stores nothing.
//   live  does it.
//
// **What can fail, and what each failure costs.** `stress.test.mjs` runs this
// file against simulated services that refuse, rate-limit, time out, lie and
// evict, and checks the invariants after every tick. The rules it holds this to:
//
// - A network that did not answer is not a network that said no. Only a
//   definite answer (a revert, a refusal naming the transaction) counts toward
//   a pause; a 429, a 5xx or a timeout is retried and then leaves the step
//   undone, with nothing sent.
// - **Every EVM transaction reuses the confirmed nonce**, never the pending
//   one. A payout the node refused may still be sitting in some other node's
//   mempool; signing its replacement at the same nonce means at most one of
//   the two can ever be mined, so a refusal can never become two payouts.
//   The earlier attempts at that nonce travel with the new one, and whichever
//   is mined is the one recorded.
// - A nonce used by something that is not ours means the transaction waiting
//   on it can never be mined. It is let go and recorded as an incident; the
//   next transaction takes the next nonce.
//
// **Nothing halts, and nothing is silent.** Every failure is retried, paused on
// a backoff that ends by itself, or written off and carried past (flow.js says
// which). What makes that operable is that it is *seen*: every dependency's
// last success and failure is kept for /treasury, and every new incident is
// posted to ALERT_WEBHOOK when one is configured, so the operator hears about
// a written-off exchange from the treasury, not from a miner.
import { decide, FRESH, MAX_RECIPIENTS, backoffMs } from "./flow.js";
import { signRvnTx, rvnTxid, rvnScript, signLegacyTx, createdAddress, encodeBuyAndPay, encodeCtor, amountOut } from "./treasury.js";
import { parseLedger, workByAddress, build, hasCode } from "./epoch.js";
import { textChunks, loadText } from "./chunks.js";
import payoutArtifact from "./GladosPayout.json" with { type: "json" };

const RPC = "https://rpc.mainnet.chain.robinhood.com";
const CHAIN_ID = 4663;
const BLOCKBOOK = "https://blockbook.ravencoin.org/api/v2";
const CHANGENOW = "https://api.changenow.io/v2";
export const TOKEN = "0x3d609ecafc6aa7dba67dd7ad1d10b49c52d57777";
export const PAIR = "0x93f777932d98d15b351d1bce8c76b34381eede5b";
export const WETH = "0x0bd7d308f8e1639fab988df18a8011f41eacad73";

// Gas figures measured on a fork of the real chain (contracts/test/payout-fork.mjs):
// ~166,600 for the buy and ~50,850 a recipient, rounded up.
export const CFG = {
  dropAfterMs: 20 * 60_000, maxFailures: 3,
  exchangeUnsentMs: 2 * 3_600_000, exchangeStuckMs: 48 * 3_600_000, deployGas: 900_000, baseGas: 200_000,
  perRecipientGas: 60_000, maxOverhead: 0.1,
  // An EVM transaction neither mined nor refused in this long is replaced at
  // the same nonce and today's gas price; its bytes alone would wait forever
  // behind a price that moved.
  stuckTxMs: 2 * 3_600_000,
  // A receipt can lag the nonce count on a load-balanced RPC: one node has the
  // block, the node answering receipts does not yet. So a nonce seen used with
  // no receipt for any of our hashes is given up on only once *both* this long and this
  // many blocks have passed -- time alone is wrong when blocks stall, blocks
  // alone are wrong when they are 0.1 s apart (4663's, measured) and a lagging
  // node is seconds behind. 3,000 blocks is about five minutes there.
  nonceGraceMs: 15 * 60_000, nonceDepth: 3000,
  // The token's buy tax, read per payout and priced into the slippage bound.
  // Past this the payout waits, and resumes by itself when the tax comes back
  // down: a buy that loses a quarter to tax is not one to make unattended.
  maxTaxBps: 2500n,
  // A written-off exchange's status is read again this often, so money that
  // turns up late is noticed.
  orphanCheckMs: 3_600_000,
  // A dependency failing continuously this long is an alert.
  outageAlertMs: 2 * 3_600_000,
  // The spot price must sit within this of the median of recent readings, or
  // the payout waits: a pair pushed off its price for one block is exactly
  // the moment a buy gets the fewest tokens.
  maxPriceMove: 0.15, priceSamples: 12, minPriceSamples: 3,
};

const big = (x) => BigInt(x);
const toJSON = (v) => JSON.parse(JSON.stringify(v, (_, x) => (typeof x === "bigint" ? x.toString() : x)));

// A read that did not get an answer. Retried; never counted as a refusal.
export class Transient extends Error {}
// A definite answer from a node: a revert, an invalid transaction.
export class RpcError extends Error {
  constructor(message, code) { super(message); this.code = code; }
}
const TRANSIENT_RPC = /rate|limit|too many|timeout|timed out|temporar|busy|unavailable|header not found|try again|overload/i;

// Large values are chunked (chunks.js) and written atomically with the state.
const chunked = (key, value) => textChunks(key, JSON.stringify(toJSON(value)));
export async function getBig(storage, key) {
  const head = await storage.get(key);
  if (head !== undefined && head !== null && typeof head === "object" && head.chunks === undefined) return head;
  const text = await loadText(storage, key);
  return text === undefined ? undefined : JSON.parse(text);
}
async function putAll(storage, entries) {
  const keys = Object.keys(entries);
  // One put is atomic up to 128 keys, and atomicity is the point: a snapshot
  // written without the state that consumed it is work paid twice.
  if (keys.length > 128) throw new Error(`${keys.length} storage keys in one write is past the atomic limit`);
  await storage.put(entries);
}

export class Treasury {
  // `io` replaces the world in tests: fetch, the clock, sleeping.
  constructor(pool, io = {}) {
    this.pool = pool; // the Pool Durable Object: ctx, env, core, treasury()
    this.fetch = io.fetch || ((u, i) => fetch(u, i));
    this.now = io.now || (() => Date.now());
    this.sleep = io.sleep || ((ms) => new Promise((r) => setTimeout(r, ms)));
    this.overrides = io.cfg || {};
    this.lastTick = -Infinity;
    this.lastNote = "";
    this.running = false;
  }

  get mode() {
    return this.pool.env.TREASURY || "off";
  }

  get cfg() {
    return { ...CFG,
      rvnBatchSats: big(Math.round(Number(this.pool.env.RVN_BATCH || 200) * 1e8)),
      maxPayWei: this.pool.env.TREASURY_MAX_PAY_WEI ? big(this.pool.env.TREASURY_MAX_PAY_WEI) : 5n * 10n ** 16n, ...this.overrides };
  }

  log(line) {
    if (line !== this.lastNote) this.pool.log(`[treasury] ${line}`);
    this.lastNote = line;
  }

  get storage() {
    return this.pool.ctx.storage;
  }

  async state() {
    return (await this.storage.get("treasury.state")) || { ...FRESH };
  }

  // Everything the state writes goes through here. Dependency health rides
  // along, so an outage's start survives an eviction, and every incident the
  // write recorded is then delivered.
  async save(s, extra = {}) {
    if (this.mode !== "live") return; // dry mode stores nothing
    if (this.health) s.health = this.health;
    await putAll(this.storage, { ...extra, "treasury.state": toJSON(s) });
    await this.deliver(s);
  }

  // **At least once.** "treasury.alerted" is the last incident the webhook
  // accepted, and it advances only on acceptance -- so an alert lost to an
  // eviction between the write and the post, or to the webhook being down, is
  // sent on the next tick instead of never. (The stress test found the first
  // version, which remembered in memory, losing one to exactly that eviction.)
  async deliver(s) {
    const pending = (s.incidents || []).filter((i) => i.seq !== undefined);
    if (!pending.length) return;
    const through = this.alerted ?? ((await this.storage.get("treasury.alerted")) || 0);
    this.alerted = through;
    let last = through;
    for (const i of pending) {
      if (i.seq <= through) continue;
      if (!(await this.alert(i.why))) break;
      last = i.seq;
    }
    if (last > through) {
      this.alerted = last;
      await this.storage.put("treasury.alerted", last);
    }
  }

  // One message to whoever the operator pointed ALERT_WEBHOOK at (a Discord or
  // Slack webhook: the body carries both `content` and `text`). True when it
  // was accepted, or when there is nowhere to send it; never throws a step.
  async alert(why) {
    this.log(`incident: ${why}`);
    const url = this.pool.env.ALERT_WEBHOOK;
    if (!url) return true;
    try {
      const msg = `GLaDOS treasury: ${why}`;
      const r = await this.fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ content: msg, text: msg }) });
      return r.ok;
    } catch {
      return false;
    }
  }

  incident(s, why) {
    s.seq = (s.seq || 0) + 1;
    s.incidents = [...(s.incidents || []), { at: this.now(), seq: s.seq, why }].slice(-20);
  }

  // Which dependencies answered, and when, for /treasury: an uptime check that
  // polls it sees a third party failing before any miner does.
  seen(dep, ok, why) {
    this.health = this.health || {};
    const h = (this.health[dep] = this.health[dep] || {});
    if (ok) h.ok = this.now(); else { h.fail = this.now(); h.why = String(why).slice(0, 160); h.since = h.since && h.since > (h.ok || 0) ? h.since : this.now(); }
  }

  // **An outage is an incident once it has lasted, and again when it ends.**
  // Everything already waits out a dependency by itself; what it cannot do is
  // tell the operator that payouts are stalled behind one. Two hours of
  // nothing but failures from a dependency is one alert, its recovery is
  // another, and a ChangeNOW that refuses the key says so at once -- that is
  // the one failure here only the operator can fix.
  watchHealth(s) {
    for (const [dep, h] of Object.entries(this.health || {})) {
      const down = h.fail && h.fail > (h.ok || 0);
      const keyRefused = dep === "changenow" && down && /HTTP 40[13]/.test(h.why || "");
      if (down && !h.alerted && (keyRefused || this.now() - h.since >= this.cfg.outageAlertMs)) {
        h.alerted = this.now();
        this.incident(s, keyRefused
          ? `ChangeNOW refuses the API key (${h.why}); RVN is held until CHANGENOW_KEY is replaced`
          : `${dep} has not answered for ${Math.round((this.now() - h.since) / 60_000)} min (${h.why}); everything behind it waits and resumes by itself`);
      } else if (!down && h.alerted) {
        this.incident(s, `${dep} is answering again after ${Math.round((this.now() - h.since) / 60_000)} min`);
        delete h.alerted;
      }
    }
  }

  // Once every `everyMs`, whatever calls it, and never twice at once: a step
  // slowed by retries must not overlap the next alarm's.
  async tick(everyMs = 5 * 60_000) {
    if (this.mode === "off" || this.running || this.now() - this.lastTick < everyMs) return;
    this.lastTick = this.now();
    this.running = true;
    try {
      await this.step();
    } catch (e) {
      this.log(`step failed, nothing sent: ${e && e.message ? e.message : e}`);
    } finally {
      this.running = false;
    }
    // Outages are judged whether or not the step got anywhere -- a dependency
    // that is down is exactly what makes a step fail.
    if (this.mode === "live") {
      try {
        const s = await this.state();
        const before = s.seq || 0;
        this.watchHealth(s);
        if ((s.seq || 0) !== before) await this.save(s);
      } catch (e) {
        this.log(`health check not saved: ${e && e.message ? e.message : e}`);
      }
    }
  }

  async step() {
    // Once per start: does the ChangeNOW key authenticate? A read-only quote,
    // logged as a yes or no -- the key itself is never read out.
    if (!this.keyChecked) {
      this.keyChecked = true;
      try {
        const q = await this.json(`${CHANGENOW}/exchange/estimated-amount?fromCurrency=rvn&fromNetwork=rvn&toCurrency=eth&toNetwork=hood&flow=standard&fromAmount=200`,
          { headers: this.cnHeaders() });
        this.log(q.body && q.body.toAmount ? `ChangeNOW key works: 200 RVN quotes ${q.body.toAmount} ETH on 4663`
                                            : `ChangeNOW key refused: ${JSON.stringify(q.body).slice(0, 160)}`);
      } catch (e) {
        this.keyChecked = false;
        if (!(e instanceof Transient)) throw e;
        this.log(`ChangeNOW unreachable for the key check: ${e.message}`);
      }
    }
    const keys = await this.pool.treasury();
    const state = await this.state();
    if (!this.health && state.health) this.health = state.health;
    await this.deliver(state);
    // A pause ends by itself. Changing TREASURY_RESUME and deploying ends one
    // early -- a deploy, never a request, since an endpoint is something
    // anybody on the internet can call. A state from before pauses existed
    // that is halted becomes a pause that has already run out.
    if (state.halted) {
      state.paused = { why: state.halted, until: 0 };
      delete state.halted;
    }
    const resume = this.pool.env.TREASURY_RESUME || "";
    if (state.paused && resume && resume !== state.resumedWith) {
      this.log(`pause lifted by TREASURY_RESUME=${resume}: ${state.paused.why}`);
      state.paused = null;
      state.failures = 0;
      state.resumedWith = resume;
      await this.save(state);
    }
    const facts = await this.facts(keys, state);
    const { action, state: next } = decide(state, facts, this.cfg);
    // A transaction that failed keeps its nonce's history: the next one signed
    // at that nonce carries it, so whichever is mined is the one recorded.
    // (Only one that was never mined: a mined revert used its nonce, and
    // remembering it would adopt the same revert again next tick.)
    if (state.pending && !next.pending && state.pending.chain === "evm" && facts.receipt && facts.receipt.ok === false && !facts.receipt.mined) {
      const p = state.pending;
      next.evmPrior = { nonce: p.nonce, txs: [...(p.prior || []), { hash: p.hash, kind: p.kind, contract: p.contract, epoch: p.epoch }].slice(-6) };
    }
    if (state.pending && !next.pending && facts.receipt && facts.receipt.mined) delete next.evmPrior;
    if (action.kind === "wait" || action.kind === "note") {
      this.log(action.why);
      await this.save(next);
      return;
    }
    if (this.mode !== "live") {
      this.log(`dry run: would ${action.kind} ${JSON.stringify(toJSON({ ...action, recipients: action.recipients?.length, utxos: action.utxos?.length, epoch: undefined }))}`);
      return;
    }
    await this.perform(action, next, keys, facts);
  }

  // --- the network, classified --------------------------------------------------------------

  cnHeaders(extra = {}) {
    return { ...extra, "x-changenow-api-key": this.pool.env.CHANGENOW_KEY || "" };
  }

  // One attempt. A missing answer is Transient; any answer is returned.
  async once(url, init = {}) {
    const where = url.split("?")[0];
    const dep = url.startsWith(RPC) ? "rpc" : url.startsWith(BLOCKBOOK) ? "blockbook" : url.startsWith(CHANGENOW) ? "changenow" : "other";
    try {
      const r = await this.onceRaw(url, init, where);
      this.seen(dep, r.status < 400, r.status >= 400 ? `HTTP ${r.status}` : "");
      return r;
    } catch (e) {
      if (e instanceof Transient) this.seen(dep, false, e.message);
      throw e;
    }
  }

  async onceRaw(url, init, where) {
    let r;
    try {
      const signal = typeof AbortSignal !== "undefined" && AbortSignal.timeout ? AbortSignal.timeout(20_000) : undefined;
      r = await this.fetch(url, { ...init, signal });
    } catch (e) {
      throw new Transient(`${where} unreachable: ${e && e.message ? e.message : e}`);
    }
    if (r.status === 429 || r.status >= 500) throw new Transient(`${where} answered ${r.status}`);
    let t;
    try { t = await r.text(); } catch (e) { throw new Transient(`${where} dropped the body: ${e.message}`); }
    let body;
    try { body = JSON.parse(t); } catch { body = { error: String(t).slice(0, 200) }; }
    return { status: r.status, body };
  }

  async retry(fn, tries = 4) {
    let last;
    for (let i = 0; i < tries; i++) {
      if (i) await this.sleep(500 * 3 ** (i - 1));
      try {
        return await fn();
      } catch (e) {
        if (!(e instanceof Transient)) throw e;
        last = e;
      }
    }
    throw last;
  }

  json(url, init) {
    return this.retry(() => this.once(url, init));
  }

  rpcBody(payload) {
    return { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) };
  }

  rpc(method, params) {
    return this.retry(async () => {
      const { body } = await this.once(RPC, this.rpcBody({ jsonrpc: "2.0", id: 1, method, params }));
      if (body && body.error) {
        const msg = `${method}: ${body.error.message || JSON.stringify(body.error)}`;
        if (body.error.code === -32005 || TRANSIENT_RPC.test(msg)) throw new Transient(msg);
        throw new RpcError(msg, body.error.code);
      }
      if (!body || !("result" in body)) throw new Transient(`${method}: no result`);
      return body.result;
    });
  }

  // Several calls in one request; answers a Map id -> result, missing where a
  // call errored. A batch that did not come back at all is Transient.
  batch(calls) {
    return this.retry(async () => {
      const { body } = await this.once(RPC, this.rpcBody(calls.map((c, id) => ({ jsonrpc: "2.0", id, ...c }))));
      if (!Array.isArray(body)) throw new Transient(`batch: ${JSON.stringify(body).slice(0, 120)}`);
      const m = new Map();
      for (const r of body) if (r && r.result !== undefined && r.result !== null) m.set(r.id, r.result);
      return m;
    });
  }

  // --- reading ---------------------------------------------------------------------------

  async facts(keys, state) {
    const f = { now: this.now() };
    const [utxos, min, gasPrice, ethWei] = await Promise.all([
      this.json(`${BLOCKBOOK}/utxo/${keys.rvn}?confirmed=true`),
      this.json(`${CHANGENOW}/exchange/min-amount?fromCurrency=rvn&fromNetwork=rvn&toCurrency=eth&toNetwork=hood&flow=standard`,
        { headers: this.cnHeaders() }),
      this.rpc("eth_gasPrice", []),
      this.rpc("eth_getBalance", [keys.evm, "latest"]),
    ]);
    // A UTXO list that is not a list is not "no coins": deciding on it would
    // drop an unsent exchange as unfunded. Unanswered, the step is not taken.
    if (!Array.isArray(utxos.body)) throw new Transient(`blockbook utxo: ${JSON.stringify(utxos.body).slice(0, 120)}`);
    // Coinbase outputs cannot be spent for 100 blocks; a transaction spending one
    // earlier is refused by every node.
    f.rvnUtxos = utxos.body.filter((u) => !(u.coinbase && (u.confirmations || 0) < 100))
      .filter((u) => /^[0-9a-f]{64}$/.test(String(u.txid)) && Number.isInteger(u.vout) && /^\d+$/.test(String(u.value)))
      .map((u) => ({ txid: u.txid, vout: u.vout, sats: String(u.value) }));
    const minAmount = min.body && Number(min.body.minAmount);
    f.rvnMinSats = minAmount > 0 && Number.isFinite(minAmount) ? big(Math.ceil(minAmount * 1e8)) : 0n;
    f.gasPrice = big(gasPrice);
    f.ethWei = big(ethWei);

    await this.watchOrphans(state);
    if (!state.pending && state.evmPrior) await this.settlePrior(state, keys, f);
    if (state.pending && f.receipt === undefined) f.receipt = await this.receipt(state, keys);
    if (state.exchange && state.exchange.sent) {
      const x = await this.json(`${CHANGENOW}/exchange/by-id?id=${encodeURIComponent(state.exchange.id)}`, { headers: this.cnHeaders() });
      f.exchangeStatus = x.body && typeof x.body.status === "string" ? x.body.status : undefined;
    }
    // The epoch is only worth building when there is a contract and money to pay with.
    if (state.contract && f.ethWei > 0n && !state.pending && !state.exchange) {
      f.epoch = await this.epoch(keys, state);
      f.recipients = f.epoch.recipients;
    }
    return f;
  }

  // What happened to the transaction in flight: {ok}, null for not yet, or
  // {lost} when its nonce went to something that is none of ours.
  async receipt(state, keys) {
    const p = state.pending;
    const cfg = this.cfg;
    const now = this.now();
    if (p.chain === "evm") {
      const mine = [p, ...(p.prior || [])];
      for (const t of mine) {
        const r = await this.rpc("eth_getTransactionReceipt", [t.hash]);
        if (r) {
          if (t !== p) {
            // An earlier attempt at this nonce is the one the chain kept. Record
            // what *it* did -- its contract, its epoch -- not the replacement's.
            this.log(`${t.kind} ${t.hash}, an earlier attempt at nonce ${p.nonce}, is the one mined`);
            state.pending = { ...p, hash: t.hash, kind: t.kind, contract: t.contract, epoch: t.epoch, prior: [] };
          }
          return { ok: r.status === "0x1", mined: true };
        }
      }
      if (p.nonce !== undefined) {
        const used = Number(await this.rpc("eth_getTransactionCount", [keys.evm, "latest"]));
        if (used > p.nonce) {
          const head = Number(await this.rpc("eth_blockNumber", []));
          if (p.goneAt === undefined) {
            p.goneAt = now;
            p.goneBlock = head;
            await this.save(state);
          }
          if (now - p.goneAt > cfg.nonceGraceMs && head - p.goneBlock >= cfg.nonceDepth) {
            return { lost: `nonce ${p.nonce} was used by a transaction that is none of ours (${mine.map((t) => t.hash).join(", ")}); the key or the record is not what this treasury thinks` };
          }
          return null;
        }
      }
      if (p.rejected && now - p.at > cfg.dropAfterMs) return { ok: false };
      if (now - p.at > cfg.stuckTxMs) return { ok: false };
      return null;
    }
    // Ravencoin: the same inputs make the same bytes (RFC 6979), so a send that
    // failed is resent as the identical transaction and can never pay twice.
    const t = await this.json(`${BLOCKBOOK}/tx/${p.hash}`);
    if (t.status === 200 && t.body && t.body.txid) {
      return (t.body.confirmations || 0) >= 1 ? { ok: true } : null;
    }
    if (p.rejected && now - p.at > cfg.dropAfterMs) return { ok: false };
    return null;
  }

  // **A transaction counted as failed can still be mined.** It was refused by
  // one node or sat too long, but some other mempool may hold it, and it lands
  // whenever the price lets it. So before anything new is signed, the nonce
  // those attempts shared is checked: still unused, and the next transaction
  // takes it (carrying them, so only one can ever be mined); used, and the
  // attempt that used it is recorded as what it was -- a contract, a payout --
  // exactly as if its receipt had come in on time. The stress test found this
  // as a second contract and as a payout mined, unrecorded, and about to be
  // paid again.
  async settlePrior(state, keys, f) {
    const pr = state.evmPrior;
    const used = Number(await this.rpc("eth_getTransactionCount", [keys.evm, "latest"]));
    if (used <= pr.nonce) return;
    for (const t of pr.txs) {
      const r = await this.rpc("eth_getTransactionReceipt", [t.hash]);
      if (r) {
        this.log(`${t.kind} ${t.hash}, counted failed earlier, was mined after all`);
        state.pending = { ...t, chain: "evm", nonce: pr.nonce, at: f.now, prior: [] };
        delete state.evmPrior;
        f.receipt = { ok: r.status === "0x1", mined: true };
        return;
      }
    }
    // Used, and no receipt for any of ours: the same patience as for a
    // pending transaction, then the same letting go.
    const head = Number(await this.rpc("eth_blockNumber", []));
    if (pr.goneAt === undefined) {
      pr.goneAt = f.now;
      pr.goneBlock = head;
      await this.save(state);
    }
    if (f.now - pr.goneAt > this.cfg.nonceGraceMs && head - pr.goneBlock >= this.cfg.nonceDepth) {
      this.incident(state, `nonce ${pr.nonce} was used, and by none of the attempts made at it (${pr.txs.map((t) => t.hash).join(", ")}); they can no longer be mined, and the next transaction takes the next nonce`);
      delete state.evmPrior;
      return;
    }
    throw new Transient(`nonce ${pr.nonce} is used and no receipt for our attempts has arrived yet`);
  }

  async epochDocs() {
    const env = this.pool.env;
    const n = Math.max(1, Number(env.SHARDS || 1));
    const gpu = env.GPU_COINS ? Math.max(1, Number(env.GPU_SHARDS || 1)) : 0;
    const names = [...[...Array(n).keys()].map((i) => (i === 0 ? "main" : `shard-${i}`)), ...[...Array(gpu).keys()].map((i) => `gpu-${i}`)];
    // A shard that does not answer fails the epoch rather than leaving its
    // miners out of it: they would be refused this epoch's pay for nothing.
    return Promise.all(names.map(async (name) => {
      if (name === "main") return this.pool.core.ledger(1, Math.floor(this.now() / 1000));
      const r = await env.POOL.get(env.POOL.idFromName(name)).fetch(new Request("https://pool/ledger.json"));
      if (!r.ok) throw new Transient(`shard ${name} answered ${r.status}`);
      return r.text();
    }));
  }

  // Every shard's ledger summed by payout address, less the last epoch's
  // snapshot, gated on GLADOS held right now, floored at a quarter of typical,
  // and never an address that holds code or is the pool's own plumbing.
  async epoch(keys, state) {
    const docs = await this.epochDocs();
    const now = workByAddress(docs.flatMap(parseLedger));
    const snap = (await getBig(this.storage, "treasury.snapshot")) || {};
    const prev = new Map(Object.entries(snap).map(([a, w]) => [a, big(w)]));
    const addrs = [...now.keys()];
    const balances = new Map();
    const ineligible = new Set([PAIR, TOKEN, WETH, keys.evm, state.contract].filter(Boolean).map((a) => a.toLowerCase()));
    for (let i = 0; i < addrs.length; i += 50) {
      const part = addrs.slice(i, i + 50);
      let res;
      try {
        res = await this.batch(part.flatMap((a) => [
          { method: "eth_call", params: [{ to: TOKEN, data: "0x70a08231" + a.slice(2).padStart(64, "0") }, "latest"] },
          { method: "eth_getCode", params: [a, "latest"] },
        ]));
      } catch (e) {
        if (!(e instanceof Transient)) throw e;
        continue; // unread: build() carries their work to the next epoch
      }
      part.forEach((a, j) => {
        const bal = res.get(2 * j), code = res.get(2 * j + 1);
        if (bal === undefined || code === undefined) return;
        if (hasCode(code)) ineligible.add(a);
        balances.set(a, big(bal));
      });
    }
    const gateMin = big(this.pool.env.GATE_MIN || "0") * 10n ** 18n;
    return build({ now, prev, balances, gateMin, ineligible, maxRecipients: MAX_RECIPIENTS });
  }

  // --- writing ---------------------------------------------------------------------------

  async perform(a, next, keys, facts) {
    if (a.kind === "create_exchange") {
      const res = await this.json(`${CHANGENOW}/exchange`, { method: "POST",
        headers: this.cnHeaders({ "content-type": "application/json" }),
        body: JSON.stringify({ fromCurrency: "rvn", fromNetwork: "rvn", toCurrency: "eth", toNetwork: "hood",
          fromAmount: (Number(a.sats) / 1e8).toFixed(8), address: keys.evm, refundAddress: keys.rvn,
          flow: "standard", type: "direct" }) });
      const b = res.body || {};
      // Everything the pool will act on is checked before it is stored: where
      // the RVN goes must be a Ravencoin address, and where the ETH comes back
      // must be ours. An answer that fails either is not an exchange.
      let refusal = null;
      if (!b.id || typeof b.id !== "string" || !b.payinAddress) refusal = `no exchange: ${JSON.stringify(b).slice(0, 200)}`;
      else {
        try { rvnScript(b.payinAddress); } catch { refusal = `the deposit address ${b.payinAddress} is not a Ravencoin address`; }
        if (!refusal && b.payoutAddress && String(b.payoutAddress).toLowerCase() !== keys.evm.toLowerCase()) {
          refusal = `the exchange pays ${b.payoutAddress}, not the pool's ${keys.evm}`;
        }
        if (!refusal && b.payinExtraId) refusal = `the deposit wants a memo (${b.payinExtraId}) this sender cannot attach`;
      }
      if (refusal) {
        this.log(`exchange not created: ${refusal}`);
        return;
      }
      next.exchange = { id: b.id, payin: b.payinAddress, sats: a.sats.toString(), outpoints: a.outpoints, created: this.now() };
      await this.save(next);
      this.log(`exchange ${b.id} created for ${Number(a.sats) / 1e8} RVN`);
      return;
    }

    if (a.kind === "send_rvn") {
      const outputs = [{ address: a.to, sats: a.sats }];
      if (a.change > 0n) outputs.push({ address: keys.rvn, sats: a.change });
      const raw = signRvnTx(keys.rvnKey, a.utxos.map((u) => ({ ...u, sats: big(u.sats) })), outputs);
      next.pending = { kind: "rvn", chain: "rvn", hash: rvnTxid(raw), raw, at: this.now() };
      await this.save(next); // recorded before the network sees it
      await this.broadcast(next);
      return;
    }

    if (a.kind === "rebroadcast") {
      await this.broadcast(next);
      return;
    }

    if (a.kind === "deploy") {
      const nonce = Number(await this.rpc("eth_getTransactionCount", [keys.evm, "latest"]));
      const gasPrice = (facts.gasPrice * 125n) / 100n;
      const data = "0x" + payoutArtifact.bytecode + encodeCtor(WETH, TOKEN, PAIR);
      const est = await this.estimate({ from: keys.evm, data }, next);
      if (est === null) return;
      const gasLimit = (est * 12n) / 10n;
      if (facts.ethWei < gasLimit * gasPrice) {
        this.log(`deploy waits: ${facts.ethWei} wei cannot cover ${gasLimit * gasPrice}`);
        return;
      }
      const tx = signLegacyTx(keys.evmKey, { nonce, gasPrice, gas: gasLimit, to: null, value: 0n, data, chainId: CHAIN_ID });
      next.pending = { kind: "deploy", chain: "evm", hash: tx.hash, raw: tx.raw, at: this.now(), nonce,
                       contract: createdAddress(keys.evm, nonce), prior: this.prior(next, nonce) };
      delete next.evmPrior;
      await this.save(next);
      await this.broadcast(next);
      return;
    }

    if (a.kind === "pay") {
      const [[r0, r1], taxHex] = await Promise.all([this.reserves(), this.rpc("eth_call", [{ to: TOKEN, data: "0x691f224f" }, "latest"])]);
      const [rW, rG] = big(WETH) < big(TOKEN) ? [r0, r1] : [r1, r0];
      const tax = big(taxHex);
      if (tax > this.cfg.maxTaxBps) {
        if (!next.taxHigh) this.incident(next, `the token's buy tax reads ${Number(tax) / 100}%, over the ${Number(this.cfg.maxTaxBps) / 100}% a payout accepts; payouts wait for it to come down`);
        next.taxHigh = true;
        await this.save(next);
        this.log(`payout waits: buy tax ${Number(tax) / 100}%`);
        return;
      }
      delete next.taxHigh;
      const moved = this.priceMoved(next, rW, rG);
      if (moved) {
        await this.save(next);
        this.log(`payout waits: ${moved}`);
        return;
      }
      // The quote, less the token's buy tax, less 3% for movement between now
      // and the block: a sandwich past that reverts the whole payout.
      const quoteMin = (v) => (amountOut(v, rW, rG) * (10_000n - tax) * 97n) / 1_000_000n;
      const nonce = Number(await this.rpc("eth_getTransactionCount", [keys.evm, "latest"]));
      const gasPrice = (facts.gasPrice * 125n) / 100n;
      const est = await this.estimate({ from: keys.evm, to: next.contract, data: encodeBuyAndPay(a.recipients, quoteMin(a.value)),
                                        value: "0x" + a.value.toString(16) }, next);
      if (est === null) return;
      const gasLimit = (est * 12n) / 10n;
      // **Value plus the most this transaction can burn must fit the balance**,
      // or the node refuses it as underfunded.
      let value = a.value;
      if (value + gasLimit * gasPrice > facts.ethWei) value = facts.ethWei - gasLimit * gasPrice;
      if (value <= 0n) {
        this.log(`payout waits: gas ${gasLimit * gasPrice} wei would take the whole balance`);
        return;
      }
      const minOut = quoteMin(value);
      const tx = signLegacyTx(keys.evmKey, { nonce, gasPrice, gas: gasLimit, to: next.contract, value, data: encodeBuyAndPay(a.recipients, minOut), chainId: CHAIN_ID });
      const e = facts.epoch;
      const id = (next.epochs || 0) + 1;
      // The whole record -- who, why, and the snapshot it moves the ledger to --
      // is stored under this transaction's own hash, so an earlier attempt that
      // is the one mined is recorded as what it was, not as its replacement.
      const record = { id, at: this.now(), value: value.toString(), minOut: minOut.toString(), tax: tax.toString(),
                       recipients: a.recipients, work: e.work, floor: e.floor, excluded: e.excluded,
                       snapshot: Object.fromEntries([...e.snapshot].map(([k, v]) => [k, v.toString()])) };
      next.pending = { kind: "pay", chain: "evm", hash: tx.hash, raw: tx.raw, at: this.now(), nonce,
                       epoch: { id, key: `treasury.attempt.${tx.hash}` }, prior: this.prior(next, nonce) };
      delete next.evmPrior;
      await this.save(next, chunked(`treasury.attempt.${tx.hash}`, record));
      await this.broadcast(next);
      return;
    }

    if (a.kind === "record_epoch") {
      let rec = await getBig(this.storage, a.epoch.key);
      if (!rec) rec = await this.reconstruct(a.hash, next);
      const { snapshot, ...rest } = rec;
      const paid = { ...rest, id: next.epochs, tx: a.hash };
      await this.save(next, { ...chunked(`treasury.epoch.${next.epochs}`, paid), ...chunked("treasury.snapshot", snapshot) });
      this.log(`epoch ${paid.id} paid ${paid.recipients.length} miner(s) in ${a.hash}`);
    }
  }

  // A payout mined whose record is gone -- which one atomic write should make
  // impossible, and which must not stop the treasury if it happens anyway.
  // The chain has who was paid; everybody paid is counted as paid up to what
  // the ledger says now. That can under-credit work done in the minutes since
  // the payout was signed, and never pays anybody twice.
  async reconstruct(hash, next) {
    const tx = await this.rpc("eth_getTransactionByHash", [hash]);
    const d = String(tx && tx.input || "").slice(10);
    const n = d.length >= 192 ? Number(BigInt("0x" + d.slice(128, 192))) : 0;
    const recipients = [...Array(n).keys()].map((i) => "0x" + d.slice((3 + i) * 64 + 24, (4 + i) * 64));
    const now = workByAddress((await this.epochDocs()).flatMap(parseLedger));
    const snapshot = (await getBig(this.storage, "treasury.snapshot")) || {};
    for (const a of recipients) {
      const w = now.get(a);
      if (w !== undefined && w > BigInt(snapshot[a] || 0)) snapshot[a] = w.toString();
    }
    this.incident(next, `payout ${hash} was mined with its record missing; rebuilt from the chain (${recipients.length} paid)`);
    return { id: next.epochs, at: this.now(), recipients, work: {}, excluded: {}, rebuilt: true, snapshot };
  }

  // Written-off exchanges are still ChangeNOW's to finish or refund. Read each
  // again hourly, so money that turns up late is noticed and said.
  async watchOrphans(state) {
    for (const o of state.orphans || []) {
      if (o.status === "finished" || o.status === "refunded") continue;
      if (this.now() - (o.checked || 0) < this.cfg.orphanCheckMs) continue;
      o.checked = this.now();
      try {
        const x = await this.json(`${CHANGENOW}/exchange/by-id?id=${encodeURIComponent(o.id)}`, { headers: this.cnHeaders() });
        const st = x.body && x.body.status;
        if (st && st !== o.status) {
          o.status = st;
          if (st === "finished" || st === "refunded") this.incident(state, `written-off exchange ${o.id} ${st} after all; ${st === "finished" ? "its ETH has arrived" : "its RVN is back"}`);
        }
      } catch (e) {
        if (!(e instanceof Transient)) throw e;
      }
    }
  }

  // Earlier attempts at this nonce, which the new transaction replaces and
  // must remember: any of them may still be the one the chain keeps.
  prior(s, nonce) {
    return s.evmPrior && s.evmPrior.nonce === nonce ? s.evmPrior.txs : [];
  }

  // Refuses (with a reason) when the spot price is off the recent median.
  // The readings are kept in the state, one per payout attempt.
  priceMoved(s, rW, rG) {
    const cfg = this.cfg;
    const px = rW > 0n ? Number((rG * 10n ** 6n) / rW) : 0;
    const hist = (s.px || []).filter((x) => x > 0);
    s.px = [...hist, px].slice(-cfg.priceSamples);
    if (!(px > 0)) return "the pair reports no WETH";
    if (hist.length < cfg.minPriceSamples) return `${hist.length + 1} price reading(s) so far; ${cfg.minPriceSamples + 1} before the first buy`;
    const sorted = [...hist].sort((x, y) => x - y);
    const median = sorted[Math.floor(sorted.length / 2)];
    const move = Math.abs(px - median) / median;
    return move > cfg.maxPriceMove ? `the price is ${(move * 100).toFixed(1)}% off its recent median` : null;
  }

  // eth_estimateGas, where a revert is a failure the decision counts: a payout
  // the chain would refuse must reach a pause after three tries. A node that
  // did not answer is not a revert and counts for nothing.
  async estimate(tx, next) {
    try {
      return big(await this.rpc("eth_estimateGas", [tx]));
    } catch (e) {
      if (!(e instanceof RpcError)) throw e;
      next.failures = (next.failures || 0) + 1;
      if (next.failures >= this.cfg.maxFailures) {
        const ms = backoffMs(next.backoffs || 0);
        next.paused = { why: `${next.failures} estimates refused in a row, last: ${e.message}`, until: this.now() + ms };
        next.backoffs = (next.backoffs || 0) + 1;
        next.failures = 0;
        this.incident(next, `${next.paused.why}; paused ${Math.round(ms / 60_000)} min, then it tries again`);
      }
      await this.save(next);
      this.log(`estimate refused (${next.failures}): ${e.message}`);
      return null;
    }
  }

  async reserves() {
    const ret = await this.rpc("eth_call", [{ to: PAIR, data: "0x0902f1ac" }, "latest"]);
    if (!/^0x[0-9a-f]{192}$/i.test(ret)) throw new Transient(`getReserves answered ${String(ret).slice(0, 40)}`);
    return [big("0x" + ret.slice(2, 66)), big("0x" + ret.slice(66, 130))];
  }

  // Send `state.pending` to its chain. No answer is not a refusal: the bytes
  // are already stored, and the next tick finds them mined or sends them again.
  // A refusal that says the transaction is already known is success; any other
  // definite refusal is marked, and the receipt check decides what it meant.
  async broadcast(s) {
    const p = s.pending;
    let err = null;
    try {
      if (p.chain === "rvn") {
        const r = await this.json(`${BLOCKBOOK}/sendtx/`, { method: "POST", body: p.raw });
        const e = r.body && r.body.error;
        if (e && !/already|known|in mempool|exists/i.test(typeof e === "string" ? e : JSON.stringify(e))) err = typeof e === "string" ? e : JSON.stringify(e);
      } else {
        try {
          await this.rpc("eth_sendRawTransaction", [p.raw]);
        } catch (e) {
          if (!(e instanceof RpcError)) throw e;
          if (!/already known|known transaction|already imported|nonce too low/i.test(e.message)) err = e.message;
        }
      }
    } catch (e) {
      if (!(e instanceof Transient)) throw e;
      this.log(`${p.kind} ${p.hash} unanswered (${e.message}); the stored bytes go again`);
      return;
    }
    if (err) {
      s.pending = { ...p, rejected: err };
      await this.save(s);
      this.log(`${p.kind} ${p.hash} refused: ${err}`);
    } else {
      this.log(`${p.kind} ${p.hash} broadcast`);
    }
  }

  // What /treasury shows: never a key.
  async summary() {
    const s = await this.state();
    // One word for an uptime monitor: "ok", "degraded" (a dependency is down
    // and everything behind it waits), or "paused" (a backoff is running).
    const health = this.health || s.health || {};
    const down = Object.entries(health).filter(([, h]) => h.fail && h.fail > (h.ok || 0)).map(([d]) => d);
    const status = s.paused && s.paused.until > this.now() ? "paused" : down.length ? "degraded" : "ok";
    return { status, down, mode: this.mode, phase: s.phase, paused: s.paused || null, epochs: s.epochs, contract: s.contract,
             health, incidents: (s.incidents || []).slice(-10),
             orphans: (s.orphans || []).map((o) => ({ id: o.id, rvn: Number(o.sats) / 1e8, status: o.status })),
             exchange: s.exchange && { id: s.exchange.id, sent: s.exchange.sent || null },
             pending: s.pending && { kind: s.pending.kind, hash: s.pending.hash, rejected: s.pending.rejected || null } };
  }

  // The newest `n` epochs, for /payouts.json.
  async epochs(n = 50) {
    const s = await this.state();
    const out = [];
    for (let i = s.epochs || 0; i > 0 && out.length < n; i--) {
      const e = await getBig(this.storage, `treasury.epoch.${i}`);
      if (e) out.push(e);
    }
    return out;
  }
}

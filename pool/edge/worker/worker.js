// The GLaDOS pool, served from a Cloudflare Durable Object.
//
// **Nothing in this file decides anything about a miner.** Every rule -- one
// worker per connection, rate limits, the bad-share cutoff, VarDiff, validation
// itself -- runs in the WebAssembly build of `pool/src`, the same source the TCP
// daemon runs. This file is the other half of `server.rs`: it moves lines between
// a socket and the core, keeps the share log in storage, and wakes the core when a
// miner has gone quiet. `harness.mjs` does the same job over TCP, which is how
// this core was tested before it was ever deployed.
//
// ### One object, and why it stays awake
//
// Every miner reaches the same object (`idFromName("main")`), because a pool is
// one ledger and one job counter; splitting it would split the record. Sockets are
// accepted with the ordinary `accept()` rather than the hibernation API, and that
// is a decision with a price: a hibernated object loses its memory, and the core's
// memory is the pool -- the job ring that tells a fresh share from a stale one,
// each miner's difficulty, the duplicates already seen. Rebuilding all of that on
// every wake would be a second persistence layer to get wrong. So the object stays
// resident while anyone is connected, which Cloudflare bills as duration, and the
// ledger is written to storage every job period so an eviction costs at most one.
//
// ### The protocol
//
// ### Upstream
//
// Work comes from zpool over an outbound `connect()`, driven through the same
// `Upstream` state machine the native daemon runs from a thread. **It is open
// only while a miner is**: an open socket keeps the object resident and billed,
// and with nobody hashing there is nothing to forward and nobody to hand work
// to. The first miner opens it and the last one closes it, so a connection's
// first job can wait on an upstream handshake -- about a second -- and then
// arrives the moment upstream sends one rather than at the next idle period.
//
// Miners speak the native protocol (`mine::proto`), one JSON message per line.
// A WebSocket message already has edges, but a sender may batch several lines into
// one, so messages are still split on newlines -- the core is handed exactly one
// line at a time, as `server.rs` hands it one.

import { connect } from "cloudflare:sockets";
import { newKey, rvnAddress, evmAddress } from "./treasury.js";
import { parseLedger, workByAddress } from "./epoch.js";
import { readerData, readerDecode, READ_MAX } from "./reader.js";
import { textChunks, loadText } from "./chunks.js";
import { Treasury } from "./runner.js";
import { Endpoints, rpcList, attempt, plainPost } from "./rpc.js";
import { classifyFailure, unavailable } from "./guard.js";
import { statusTick, STATUS_EVERY_MS } from "./statusprobe.js";
import { rewardOf, DEFAULT as DEFAULT_REWARD } from "./rewards.js";
import { PearlPool } from "./pearl-pool.js";
import { pearlOn, pearlStub } from "./pearl-env.js";
import { MAINT_KEY, parseMaint, validMaint, notice, refusal, closeReason, tokenOk } from "./maintenance.js";

// The Pearl shard is a class of its own (pearl-pool.js), bound as PEARL_POOL.
export { PearlPool };

// For refusals a person reads on a miner's screen: 0x6ef4...527d.
const short = (a) => `${a.slice(0, 6)}...${a.slice(-4)}`;
import wasmModule from "../target/wasm32-unknown-unknown/release/glados_edge.wasm";
import { bind, imports } from "../boundary.mjs";

// `server::JOB_PERIOD`. A quiet period is when a connection gets fresh work.
const JOB_PERIOD_MS = 30_000;
// A connection silent this long is reaped. Cloudflare does not fire a socket's
// "close" when a client drops ungracefully (a laptop sleeps, a network blips),
// so a half-open socket lingers, stays counted, and every reconnect looks like
// another miner. vardiff targets a share every ~10 s and force-retargets by
// 60 s, so a live miner is never silent for minutes; three is dead.
const IDLE_MAX_MS = 3 * 60_000;
// `mine::stratum::MAX_LINE`. A longer line is a desynchronised peer.
const MAX_LINE = 131072;
// The holding gate: a miner must hold GATE_MIN whole GLADOS on chain 4663 to
// connect. "0" (or unset) turns it off. See `Pool.gate`.
const GLADOS = "0x3d609ecafc6aa7dba67dd7ad1d10b49c52d57777";
const PAIR_ADDR = "0x93f777932d98d15b351d1bce8c76b34381eede5b";
// How long the gate gathers connections before one read answers them all.
const GATE_BATCH_MS = 300;
// The 4663 RPC is rpc.js's ordered list, shared with the treasury.
const GATE_CACHE_MS = 10 * 60_000;

// How often zpool's rates are re-read. Its figures move over minutes, and a
// switch costs every miner its current job, so faster buys nothing.
const RATES_MS = 5 * 60_000;

// --- Per-wallet progress, synced across a miner's devices --------------------
// The miner page keeps a small, always-climbing tally (an estimated-earnings
// line, plus lifetime shares, hashes and time) in the browser. This stores it
// per wallet and asset so the same line shows on every device. It is a
// cosmetic estimate the client computes, not pool accounting (the real payout
// record is /payouts.json), so it is sanitised and merged by the maximum of
// each counter: a device can only raise the line, never lower it, every value
// is capped, and the series is bounded. Read and written at /progress.
const PROG_CORS = { "access-control-allow-origin": "*", "access-control-allow-methods": "GET,POST,OPTIONS", "access-control-allow-headers": "content-type" };
const progNum = (v, max) => { v = Number(v); return Number.isFinite(v) && v >= 0 ? Math.min(v, max) : 0; };
function sanitizeProg(r) {
  if (!r || typeof r !== "object") return null;
  const series = Array.isArray(r.series)
    ? r.series.filter((p) => p && Number.isFinite(p.t) && Number.isFinite(p.v) && p.v >= 0)
        .slice(-200).map((p) => ({ t: Math.floor(p.t), v: Math.min(Number(p.v), 1e15) }))
    : [];
  return { est: progNum(r.est, 1e15), shares: progNum(r.shares, 1e12), hashes: progNum(r.hashes, 1e18),
           seconds: progNum(r.seconds, 1e10), rate: progNum(r.rate, 1e9), series };
}
function mergeProg(prev, next) {
  if (!prev) return next;
  // The series from whichever is further along, so the stored line stays
  // monotonic with the number it shows.
  const series = next.est >= prev.est ? next.series : prev.series;
  return { est: Math.max(prev.est, next.est), shares: Math.max(prev.shares, next.shares),
           hashes: Math.max(prev.hashes, next.hashes), seconds: Math.max(prev.seconds, next.seconds),
           rate: Math.max(prev.rate, next.rate), series: (series && series.length ? series : (prev.series || [])) };
}

// ### Shards
//
// **Many objects, one record.** A single Durable Object is single-threaded and
// validates every share itself, so it is the ceiling on how many miners one pool
// can hold. `SHARDS` objects split the miners between them by a hash of the
// connecting address; each is today's pool unchanged -- its own core, its own
// upstream sessions, and so its own extranonce from zpool, which is what keeps
// shards from handing out the same search space.
//
// Shard 0 is named "main", the object every miner used before sharding, so the
// record it holds carries on rather than starting again.
//
// `/ledger.json` sums the shards through `Pool::merge_ledger`: tallies are
// plain counters, a worker seen on two shards is simply the sum, and each
// shard's digest is checked before its rows are added.
// The pool core's per-shard cap on (worker, coin) tallies (pool.rs MAX_TALLIES).
const MAX_TALLIES = 4096;

// A public read that fans out to every shard, answered from Cloudflare's edge
// cache for `seconds`: however often anybody asks, the shards are asked at
// most once a period per location.
async function cached(request, seconds, make) {
  const cache = typeof caches !== "undefined" ? caches.default : null;
  const key = new Request(new URL(request.url).toString(), { method: "GET" });
  if (cache) {
    const hit = await cache.match(key);
    if (hit) return hit;
  }
  const res = await make();
  if (cache && res.ok) {
    const copy = new Response(res.clone().body, res);
    copy.headers.set("cache-control", `public, max-age=${seconds}`);
    await cache.put(key, copy);
  }
  return res;
}

function shardName(i, gpu = false) {
  if (gpu) return `gpu-${i}`;
  return i === 0 ? "main" : `shard-${i}`;
}

// Keyed by `?w=` when a miner sends one -- its payout address, say, so the
// same address always lands on the same shard -- and by the connecting IP
// otherwise. Either is correct: tallies sum across shards, so which shard a
// miner lands on changes load, never the record.
function shardFor(request, n) {
  const ip = new URL(request.url).searchParams.get("w") || request.headers.get("CF-Connecting-IP") || "";
  let h = 2166136261;
  for (let i = 0; i < ip.length; i++) h = Math.imul(h ^ ip.charCodeAt(i), 16777619) >>> 0;
  return n > 1 ? h % n : 0;
}

// ### When the pool itself fails
//
// Every route below hands the request to a Durable Object, and a call into one
// throws when the object cannot run: its constructor refused (a ledger that
// would not load), its code was just replaced, it is overloaded, or the
// account's daily Durable Object allowance is spent -- on Workers Free that is
// 13,000 GB-s and 100,000 requests a day, reset at 00:00 UTC. Uncaught, each of
// those is Cloudflare's bare "error code: 1101" on every route at once, which
// tells a miner nothing and tells the operator only that something threw.
//
// So the router is wrapped: a failure is logged whole (`wrangler tail` shows
// it) and answered 503 with a reason a person can act on. Miners already back
// off from a failed connect, doubling to a minute, so a 503 is not a reconnect
// storm. The message of an unknown error is not echoed, since it can name
// internals; the classes that are known are said plainly.

// A call that Cloudflare marks retryable and not overloaded -- typically an
// object reset by a deploy -- is retried once on a fresh stub, which is what
// Cloudflare's own guidance prescribes. Only a GET: nothing here sends a body
// to an object, and a retried body could be half-consumed.
async function ask(env, name, request) {
  const get = () => env.POOL.get(env.POOL.idFromName(name));
  try {
    return await get().fetch(request);
  } catch (e) {
    if (!e || !e.retryable || e.overloaded || request.method !== "GET") throw e;
    return await get().fetch(request);
  }
}

// The daily allowance does not come back within a request, so once one
// request has seen it spent this isolate stops asking for BREAK_MS: every miner
// retrying would otherwise be another failed object call to say the same thing.
const BREAK_MS = 30_000;
let brokenUntil = 0;
let brokenWhy = "";

// ### Maintenance (the rules are in maintenance.js)
//
// The state lives in an object of its own, "control", and each shard keeps a
// copy it acts on. The router reads it before handing a miner to a shard, from
// this isolate's memory and Cloudflare's edge cache, so a crowd of old clients
// retrying every minute during maintenance is answered here and costs no
// object a request -- on Workers Free those are 100,000 a day. A read that
// fails lets the miner through: the shard holds its own copy and refuses
// there, and a control object that is down must not take mining with it.
const MAINT_MEMO_MS = 5_000;
const MAINT_EDGE_S = 15;
let maintMemo = { at: 0, state: null };
const control = (env) => env.POOL.get(env.POOL.idFromName("control"));
const MAINT_CORS = { "access-control-allow-origin": "*", "access-control-allow-methods": "GET,POST,OPTIONS", "access-control-allow-headers": "content-type,authorization" };

async function maintNow(env, origin) {
  const memo = env.MAINT_MEMO_MS !== undefined ? Number(env.MAINT_MEMO_MS) : MAINT_MEMO_MS;
  if (memo > 0 && Date.now() - maintMemo.at < memo) return maintMemo.state;
  try {
    const res = await cached(new Request(`${origin}/maintenance`), MAINT_EDGE_S, () => control(env).fetch(new Request(`${origin}/maintenance`)));
    const s = validMaint(await res.json());
    maintMemo = { at: Date.now(), state: s };
    return s;
  } catch (e) {
    console.error(`[maint] state unreadable, letting the shard decide: ${(e && e.message) || e}`);
    return null;
  }
}

async function maintenanceRoute(request, env, url, names) {
  if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: MAINT_CORS });
  if (request.method === "GET") {
    const res = await cached(new Request(`${url.origin}/maintenance`), MAINT_EDGE_S, () => control(env).fetch(new Request(`${url.origin}/maintenance`)));
    return new Response(res.body, { status: res.status, headers: { ...MAINT_CORS, "content-type": "application/json", "cache-control": `public, max-age=${MAINT_EDGE_S}` } });
  }
  if (request.method !== "POST") return new Response("GET or POST", { status: 405, headers: MAINT_CORS });
  const say = (status, body) => Response.json(body, { status, headers: { ...MAINT_CORS, "cache-control": "no-store" } });
  if (!env.MAINT_TOKEN) return say(403, { error: "maintenance control is not set up on this pool: npx wrangler secret put MAINT_TOKEN" });
  if (!tokenOk(request.headers.get("authorization"), env.MAINT_TOKEN)) {
    console.error(`[maint] refused a declaration with a wrong or missing token from ${request.headers.get("CF-Connecting-IP") || "?"}`);
    return say(401, { error: "not authorised" });
  }
  let body = null;
  try { body = await request.json(); } catch {}
  const parsed = parseMaint(body, Date.now());
  if (parsed.error) return say(400, { error: parsed.error });
  // The record first: if it cannot be written, nothing is told anything.
  const stored = await control(env).fetch(new Request(`${url.origin}/maintenance/set`, { method: "POST", body: JSON.stringify(parsed.state) }));
  if (!stored.ok) return say(502, { error: `the state was not stored (${stored.status}); nothing was changed` });
  const state = await stored.json();
  maintMemo = { at: Date.now(), state };
  const cache = typeof caches !== "undefined" ? caches.default : null;
  if (cache) await cache.delete(new Request(`${url.origin}/maintenance`)).catch(() => {});
  // Then every shard, at once. One that does not answer is named: it still
  // refuses at the router, and picks the state up from "control" the next time
  // anybody tries to connect to it.
  const targets = names.map((n) => ({ name: n, fetch: (r) => ask(env, n, r) }));
  if (pearlOn(env)) targets.push({ name: "pearl-0", fetch: (r) => pearlStub(env).fetch(r) });
  const settled = await Promise.allSettled(targets.map((t) => t.fetch(new Request(`${url.origin}/maintenance/apply`, { method: "POST", body: JSON.stringify(state) })).then(async (r) => {
    if (!r.ok) throw new Error(`answered ${r.status}`);
    return r.json();
  })));
  const shards = settled.map((r, i) => (r.status === "fulfilled" ? { name: targets[i].name, ok: true, ...r.value } : { name: targets[i].name, ok: false, why: classifyFailure(r.reason).why }));
  console.log(`[maint] ${state.on ? `ON: ${state.reason}` : "OFF"}; ${shards.map((s) => `${s.name} ${s.ok ? `${s.connections ?? 0} conn` : "UNREACHED"}`).join(", ")}`);
  return say(200, { maintenance: state, shards });
}

export default {
  async fetch(request, env) {
    if (Date.now() < brokenUntil) return unavailable(brokenWhy, Math.ceil((brokenUntil - Date.now()) / 1000));
    try {
      return await route(request, env);
    } catch (e) {
      const c = classifyFailure(e);
      console.error(`[edge] ${new URL(request.url).pathname} failed (retryable=${!!(e && e.retryable)}, ` +
                    `overloaded=${!!(e && e.overloaded)}): ${(e && (e.stack || e.message)) || e}`);
      if (c.quota) { brokenUntil = Date.now() + BREAK_MS; brokenWhy = c.why; }
      return unavailable(c.why, c.retry);
    }
  },
};

async function route(request, env) {
    const url = new URL(request.url);
    const n = Math.max(1, Number(env.SHARDS || 1));
    // GPU miners get shards of their own, with their own coin list, so CPU and
    // GPU switching are independent -- a card and a CPU never compete for one
    // slot choice, and neither drags the other onto an algorithm it hashes badly.
    const gpu = env.GPU_COINS ? Math.max(1, Number(env.GPU_SHARDS || 1)) : 0;
    const stub = (i, g = false) => ({ name: shardName(i, g), fetch: (req) => ask(env, shardName(i, g), req) });
    const everyShard = () => [
      ...[...Array(n).keys()].map((i) => stub(i)),
      ...[...Array(gpu).keys()].map((i) => stub(i, true)),
    ];
    const names = [...[...Array(n).keys()].map((i) => shardName(i)), ...[...Array(gpu).keys()].map((i) => shardName(i, true))];
    if (url.pathname === "/maintenance") return maintenanceRoute(request, env, url, names);
    // A miner arriving during maintenance is answered here, by no object.
    if (url.pathname === "/mine" || url.pathname === "/mine/gpu" || url.pathname === "/mine/pearl") {
      const m = await maintNow(env, url.origin);
      if (m && m.on) return refusal(m, Date.now());
    }
    if (url.pathname === "/mine") {
      return stub(shardFor(request, n)).fetch(request);
    }
    // The treasury lives in shard 0 -- "main", the one object that existed
    // before sharding -- so there is exactly one place the keys are.
    if (url.pathname === "/treasury" || url.pathname === "/payouts.json") {
      return stub(0).fetch(request);
    }
    // Progress goes to a Durable Object of its own, not shard 0. Shard 0 is held
    // alive by every miner's live socket, so a deploy cannot restart it onto new
    // code; a dedicated object has no sockets and picks up new code at once. It
    // is the one place a wallet's cross-device line is stored.
    if (url.pathname === "/progress") {
      return env.POOL.get(env.POOL.idFromName("progress")).fetch(request);
    }
    // The public status page's feed, from an object of its own for the same
    // reason: it has to answer exactly when a mining shard cannot.
    if (url.pathname === "/health.json") {
      return env.POOL.get(env.POOL.idFromName("status")).fetch(request);
    }
    if (url.pathname === "/mine/gpu") {
      if (!gpu) return new Response("no GPU coins on this pool", { status: 404 });
      return stub(shardFor(request, gpu), true).fetch(request);
    }
    // Pearl, the GPU's other choice: one shard of its own class, since its
    // shares are proofs a JavaScript verifier checks rather than nonces the core does.
    if (url.pathname === "/mine/pearl") {
      if (!pearlOn(env)) return new Response("no Pearl mining on this pool", { status: 404 });
      return pearlStub(env).fetch(request);
    }
    if (url.pathname === "/status") {
      return cached(request, 30, async () => {
        // One shard down is a pool partly down, and said as that: the shards
        // that answer are counted and the ones that did not are named. Only
        // all of them failing is a failure of /status.
        const shards = everyShard();
        const settled = await Promise.allSettled(shards.map((s) => s.fetch(new Request(`${url.origin}/status`)).then((r) => {
          if (!r.ok) throw new Error(`answered ${r.status}`);
          return r.json();
        })));
        const all = settled.map((r) => (r.status === "fulfilled" ? r.value : null));
        const down = settled.flatMap((r, i) => (r.status === "rejected" ? [`${shards[i].name}: ${classifyFailure(r.reason).why}`] : []));
        const up = all.filter(Boolean);
        if (!up.length) throw settled[0].reason;
        // Pearl answers apart: a shard that cannot answer leaves the rest standing.
        let pearl = null;
        if (pearlOn(env)) {
          try { pearl = await (await pearlStub(env).fetch(new Request(`${url.origin}/status`))).json(); }
          catch (e) { down.push(`pearl-0: ${classifyFailure(e).why}`); }
        }
        if (pearl) up.push({ ...pearl, slots: up[0].slots });
        if (down.length) console.error(`[edge] /status: ${down.join("; ")}`);
        return Response.json({
          shards: n, gpuShards: gpu, slots: up[0].slots,
          connections: up.reduce((a, s) => a + s.connections, 0),
          // Every miner has a CPU socket, so the shard with the most distinct
          // addresses holds them all: that is the real miner count, not the
          // socket sum (a miner opens a CPU and a GPU socket).
          miners: Math.max(0, ...up.map((s) => s.addresses || 0)),
          hashrate: up.reduce((a, s) => a + (s.hashrate || 0), 0),
          // Pearl's rate is multiply-accumulates a second, not hashes, so it is
          // its own figure rather than added into the one above.
          ...(pearl ? { pearl: { hashrate: pearl.pearl.hashrate, connections: pearl.connections, accepted: pearl.pearl.accepted, forwarded: pearl.pearl.forwarded, sent: pearl.pearl.sent, answered: pearl.pearl.answered, waiting: pearl.pearl.waiting, lost: pearl.pearl.lost, upAccepted: pearl.pearl.upAccepted } } : {}),
          perShard: all.map((s) => (s ? s.connections : null)),
          // The fullest shard's tallies against the core's cap: past ~90%,
          // new miners' shares go uncredited and SHARDS should be raised.
          tallies: Math.max(0, ...up.map((s) => s.tallies || 0)), maxTallies: up[0].maxTallies ?? null,
          ...(down.length ? { down } : {}),
          // Any shard that holds it on says so; the pool bar shows the reason.
          maintenance: (up.find((s) => s.maintenance && s.maintenance.on) || {}).maintenance || null,
        }, { headers: { "access-control-allow-origin": "*" } }); // the site's /pool/ page reads the miner count
      });
    }
    if (url.pathname === "/ledger.json") {
      if (n === 1 && !gpu) return stub(0).fetch(request);
      // One shard's ledger, for recomputing a payout at any scale.
      const one = url.searchParams.get("shard");
      if (one !== null) {
        const names = [...[...Array(n).keys()].map((i) => shardName(i)), ...[...Array(gpu).keys()].map((i) => shardName(i, true))];
        if (!names.includes(one)) return new Response(`no shard ${one}; they are ${names.join(", ")}`, { status: 404 });
        return ask(env, one, new Request(`${url.origin}/ledger.json`));
      }
      // **Past eight shards the merged ledger is an index.** Merging means
      // every shard's rows in one Worker's memory -- tens of megabytes at
      // 100,000 miners, past what a Worker has -- on every request, for anyone
      // who asks. The same rows are all there, one link a shard.
      if (n + gpu > 8) {
        const names = [...[...Array(n).keys()].map((i) => shardName(i)), ...[...Array(gpu).keys()].map((i) => shardName(i, true))];
        return Response.json({ shards: names.map((s) => `${url.origin}/ledger.json?shard=${s}`) }, { headers: { "access-control-allow-origin": "*" } });
      }
      return cached(request, 60, async () => {
      const docs = await Promise.all(everyShard().map((s) => s.fetch(new Request(`${url.origin}/ledger.json`)).then((r) => r.text())));
      const instance = await WebAssembly.instantiate(wasmModule, imports);
      const core = bind(instance);
      const coins = `${env.COINS || "yescrypt:yescrypt:12"} ${env.GPU_COINS || ""}`;
      core.init(coins.split(/\s+/).filter(Boolean), Number(env.SHARE_SECONDS || 45));
      for (const d of docs) core.mergeLedger(d);
      return new Response(core.ledger(1, Math.floor(Date.now() / 1000)), {
        headers: { "content-type": "application/json", "access-control-allow-origin": "*" },
      });
      });
    }
    return new Response(
      "GLaDOS pool. Miners connect over WebSocket at /mine and speak the native " +
        "protocol; the share log is at /ledger.json.\n",
      { headers: { "content-type": "text/plain; charset=utf-8" } },
    );
}

export class Pool {
  constructor(ctx, env) {
    this.ctx = ctx;
    this.env = env;
    this.conns = new Map(); // WebSocket -> connection id in the core
    this.connAddr = new Map(); // WebSocket -> the miner's payout address, for the revenue push
    this.lastSeen = new Map(); // WebSocket -> when it last sent anything, for reaping half-open sockets
    this.connRate = new Map(); // WebSocket -> the miner's self-reported H/s, summed for the combined pool rate
    this.core = null;
    this.ups = [];          // { i, where, sock, writer, running }
    this.gateMin = BigInt(env.GATE_MIN || "0") * 10n ** 18n;
    // Before the treasurer, which takes the same list: an endpoint the door
    // check found down is one the payout does not try first.
    this.rpcs = new Endpoints(rpcList(env));
    // Only shard 0 ("main") holds the keys and runs the treasury.
    this.treasurer = new Treasury(this);
    this.balances = new Map(); // address -> { wei, at }
    // Nothing is served until the core exists and the ledger is back, so a miner
    // arriving during a cold start cannot be credited into an empty record that
    // then overwrites the stored one.
    ctx.blockConcurrencyWhile(async () => {
      const instance = await WebAssembly.instantiate(wasmModule, imports);
      this.core = bind(instance);
      // Whitespace only: a coin with an upstream carries commas of its own
      // (`...@host:port,address,password`), and splitting on them tore one
      // coin into three that would not parse.
      // A shard's name says what it serves: `gpu-N` takes GPU_COINS, anything
      // else COINS. `ctx.id.name` is the name it was reached by (Cloudflare,
      // 2026-03, available in the constructor and under `wrangler dev`).
      this.gpu = String(ctx.id.name || "").startsWith("gpu-");
      const spec = this.gpu ? env.GPU_COINS : env.COINS;
      const coins = (spec || "yescrypt:yescrypt:12").split(/\s+/).filter(Boolean);
      this.slots = this.core.init(coins, Number(env.SHARE_SECONDS || 45), Number(env.WINDOW || 0));
      // **Restored, or the object refuses to serve.** A ledger that fails to load
      // and is then overwritten by the next job period is the share log erased
      // by a cold start, with nothing anywhere saying it happened. Throwing here
      // fails every request loudly instead, which is a problem somebody sees.
      // Maintenance survives a restart: a shard evicted mid-maintenance comes
      // back refusing miners, not taking them.
      this.maint = validMaint(await ctx.storage.get(MAINT_KEY));
      this.maintAt = Date.now();
      const lt = Date.now();
      const saved = await loadText(ctx.storage, "ledger");
      const restored = saved ? this.core.loadLedger(saved) : 0;
      const loadMs = Date.now() - lt;
      if (loadMs > 1000) this.log(`[edge] slow start: the ledger (${saved ? saved.length : 0} bytes) took ${loadMs} ms to load`);
      // SWITCH=1: miners are shown one slot, the one that pays best per unit of
      // their work, chosen from zpool's live rates. Coin labels must be zpool's
      // algorithm names for the rates to find them.
      this.switching = env.SWITCH === "1";
      if (this.switching) this.core.switchOn();
      this.ratesAt = 0;
      for (let i = 0; i < this.core.upCount(); i++) {
        this.ups.push({ i, where: this.core.upWhere(i), sock: null, writer: null, running: false });
      }
      // Coin specs carry the upstream address and password, so they are logged
      // by label only.
      const labels = coins.map((c) => c.split("@")[0] + (c.includes("@") ? " (upstream)" : " (local)"));
      this.log(`[edge] ${this.slots} slot(s): ${labels.join(" ")}; ${saved ? `${restored} record(s) restored` : "no stored ledger, starting from zero"}`);
    });
  }

  log(line) {
    console.log(line);
  }

  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === "/health.json") {
      if (String(this.ctx.id.name || "") !== "status") return new Response("not here", { status: 404 });
      let snap = await this.ctx.storage.get("status.snapshot");
      if (!snap || Date.now() - snap.generatedAt > 3 * STATUS_EVERY_MS) snap = await statusTick(this.env, this.ctx.storage);
      if (!(await this.ctx.storage.getAlarm())) await this.ctx.storage.setAlarm(Date.now() + STATUS_EVERY_MS);
      return Response.json(snap, { headers: { "access-control-allow-origin": "*", "cache-control": "no-store" } });
    }
    if (url.pathname === "/ledger.json") {
      return new Response(this.core.ledger(1, Math.floor(Date.now() / 1000)), {
        headers: { "content-type": "application/json", "access-control-allow-origin": "*" },
      });
    }
    // This shard's work per payout address, summed over workers and coins: what
    // the treasury needs from a shard, a few dozen bytes an address instead of
    // a ledger's hundred and thirty a (worker, coin) row. Internal: the
    // top-level router never sends a request here.
    if (url.pathname === "/work.json") {
      const work = workByAddress(parseLedger(this.core.ledger(1, Math.floor(Date.now() / 1000))));
      const m = await this.rewards();
      const rewards = {};
      for (const a of work.keys()) if (m.has(a)) rewards[a] = m.get(a);
      return Response.json({ work: Object.fromEntries([...work].map(([a, w]) => [a, w.toString()])), rewards });
    }
    if (url.pathname === "/maintenance" || url.pathname === "/maintenance/set" || url.pathname === "/maintenance/apply") {
      return this.maintRoute(request, url.pathname);
    }
    if (url.pathname === "/status") {
      return Response.json({ maintenance: this.maint && this.maint.on ? this.maint : undefined, slots: this.slots, connections: this.conns.size, addresses: new Set(this.connAddr.values()).size, hashrate: this.poolHashrate(), tallies: this.tallies ?? null, maxTallies: MAX_TALLIES });
    }
    if (url.pathname === "/treasury") {
      const t = await this.treasury();
      await this.arm();
      return Response.json({ rvn: t.rvn, evm: t.evm, ...(await this.treasurer.summary()) },
        { headers: { "access-control-allow-origin": "*" } });
    }
    if (url.pathname === "/payouts.json") {
      // Every epoch paid: who, how much work each did since the last one, the
      // floor, who was left out and why, and the transaction -- recomputable
      // with tools/distribute.py --split equal against the ledgers.
      const epochs = await this.treasurer.epochs();
      return Response.json({ epochs }, { headers: { "access-control-allow-origin": "*" } });
    }
    if (url.pathname === "/progress") {
      // A wallet's saved progress line, the same on every device it mines from.
      if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: PROG_CORS });
      const addr = (url.searchParams.get("address") || "").toLowerCase();
      const asset = (url.searchParams.get("asset") || "glados").toLowerCase();
      if (!/^0x[0-9a-f]{40}$/.test(addr)) return Response.json({ error: "address" }, { status: 400, headers: PROG_CORS });
      if (!/^[a-z0-9]{1,12}$/.test(asset)) return Response.json({ error: "asset" }, { status: 400, headers: PROG_CORS });
      const key = `progress.${addr}.${asset}`;
      if (request.method === "GET") {
        return Response.json({ rec: (await this.ctx.storage.get(key)) || null }, { headers: PROG_CORS });
      }
      if (request.method === "POST") {
        let body = null; try { body = await request.json(); } catch {}
        const clean = sanitizeProg(body && body.rec);
        if (!clean) return Response.json({ error: "rec" }, { status: 400, headers: PROG_CORS });
        const merged = mergeProg((await this.ctx.storage.get(key)) || null, clean);
        await this.ctx.storage.put(key, merged);
        return Response.json({ rec: merged }, { headers: PROG_CORS });
      }
      return new Response("method", { status: 405, headers: PROG_CORS });
    }
    if (request.headers.get("Upgrade") !== "websocket") {
      return new Response("expected a WebSocket upgrade", { status: 426 });
    }
    if (this.maint && this.maint.on && (await this.stillDown())) return refusal(this.maint, Date.now());

    const [client, server] = Object.values(new WebSocketPair());
    server.accept();
    const peer = request.headers.get("CF-Connecting-IP") || "?";
    const id = this.core.open(peer);
    this.conns.set(server, id);
    this.lastSeen.set(server, Date.now());

    const deliver = (r) => {
      for (const l of r.log) this.log(l);
      for (const m of r.send) server.send(m);
      if (r.close) {
        server.close(1000, "closed by the pool");
        this.drop(server);
      }
      return r.close;
    };
    // Lines wait here until the connection has passed the gate, so nothing a
    // miner sends reaches the core before its greeting has been checked.
    const gate = { passed: this.gateMin === 0n, queue: [] };
    const feed = (line) => deliver(this.core.line(id, line));
    server.addEventListener("message", async (ev) => {
      const text = typeof ev.data === "string" ? ev.data : new TextDecoder().decode(ev.data);
      this.lastSeen.set(server, Date.now()); // proof the socket is alive; only inbound traffic is
      if (text.length > MAX_LINE) {
        server.close(1009, "line too long");
        this.drop(server);
        return;
      }
      const lines = text.split("\n").filter((l) => l.trim());
      // What this wallet chose to be paid in, recorded from every greeting,
      // gated or not: the latest hello for an address is its choice. The same
      // greeting names the address this socket is for, so the revenue push
      // knows where to send.
      for (const l of lines) if (l.includes("glados.hello")) {
        this.noteReward(l).catch((e) => this.log(`[reward] choice not stored (${e && e.message ? e.message : e}); the next greeting tries again`));
        this.noteConn(server, l);
      }
      // A miner reports its own hashrate for the combined-pool figure; this is a
      // display line, not mining work, so it is recorded and kept out of the core.
      for (const l of lines) if (l.includes("glados.rate")) this.noteRate(server, l);
      const lines2 = lines.filter((l) => !l.includes("glados.rate"));
      if (!lines2.length) return;
      if (!gate.passed) {
        gate.queue.push(...lines2);
        if (gate.checking) return;
        const hello = gate.queue.find((l) => l.includes("glados.hello"));
        if (!hello) return;
        gate.checking = true;
        const refusal = await this.gate(hello);
        if (refusal) {
          this.log(`[gate] ${peer} refused: ${refusal}`);
          let rid = 1;
          try { rid = JSON.parse(hello).id ?? 1; } catch {}
          try { server.send(JSON.stringify({ id: rid, result: null, error: refusal }) + "\n"); } catch {}
          server.close(1008, "holding requirement not met");
          this.drop(server);
          return;
        }
        gate.passed = true;
        const held = gate.queue.splice(0);
        for (const line of held) if (feed(line)) return;
        this.tickUps();
        return;
      }
      for (const line of lines2) {
        if (feed(line)) return;
      }
      // A share may have beaten upstream's target: forward it now, not at the
      // next alarm, since upstream's job may be replaced before then.
      this.tickUps();
    });
    server.addEventListener("close", () => this.drop(server));
    server.addEventListener("error", () => this.drop(server));

    for (const u of this.ups) if (!u.running) this.runUp(u);
    this.ctx.waitUntil(this.refreshRates());
    await this.arm();
    return new Response(null, { status: 101, webSocket: client });
  }

  // **The ledger is written when a miner leaves, not only on the alarm.** An
  // object with no connections is evicted after a few idle seconds, and its
  // memory with it; the alarm then wakes a *fresh* object that restores the
  // last save and writes that back. So every share since the previous period
  // was lost whenever a miner disconnected more than a few seconds before it --
  // measured: a four-minute run's shares absent from the tally while the next
  // run's, closing six seconds before its alarm, survived.
  // The pool's hot wallets, created on first use and never regenerated.
  //
  // **Generated here, kept here, never shown.** The operator agreed to the
  // automation holding keys on the condition that they are nobody's personal
  // keys and never pass through a conversation. So they are made from
  // `crypto.getRandomValues` inside this object and live only in its storage;
  // the one thing that leaves is the pair of addresses.
  //
  // **Never overwritten.** A key that is regenerated orphans whatever was paid
  // to the old address, silently and forever. So a read that fails throws
  // rather than falling through to "no keys yet", and new keys are written
  // only when storage positively answers that there are none.
  async treasury() {
    if (this.keys) return this.keys;
    // One at a time: two first requests racing would each generate keys, and
    // the second write would replace keys whose addresses were already shown.
    // **A shared promise, not blockConcurrencyWhile.** This read throws on
    // purpose when it cannot be sure, and Cloudflare terminates an object whose
    // blockConcurrencyWhile callback throws -- so one public GET /treasury
    // during a storage blip (or with keys missing while pinned) dropped every
    // miner on the shard and every share since the last save. Seen under
    // `wrangler dev`: one request, 37 shares gone. Now the request fails and
    // the next one tries again; the object, and its miners, carry on.
    if (!this.keysLoading) {
      this.keysLoading = this.loadKeys().finally(() => { this.keysLoading = null; });
    }
    return this.keysLoading;
  }

  async loadKeys() {
    if (this.keys) return this.keys;
    let k = await this.ctx.storage.get("treasury.keys");
    // **Pinned once published.** TREASURY_RVN and TREASURY_EVM name the
    // addresses the operator has told zpool and ChangeNOW about. With them
    // set, storage answering "no keys" is not a fresh start, it is keys lost
    // -- and generating new ones would send every later payout to addresses
    // nobody watches. So it refuses, and so does a mismatch.
    const pinRvn = this.env.TREASURY_RVN || "", pinEvm = (this.env.TREASURY_EVM || "").toLowerCase();
    if (k === undefined) {
      if (pinRvn || pinEvm) throw new Error("the treasury's keys are missing from storage while its addresses are pinned; refusing to make new ones");
      k = { rvnKey: newKey(), evmKey: newKey(), created: Date.now() };
      await this.ctx.storage.put("treasury.keys", k);
      this.log("[treasury] hot wallets created");
    }
    const keys = { ...k, rvn: rvnAddress(k.rvnKey), evm: evmAddress(k.evmKey) };
    if ((pinRvn && pinRvn !== keys.rvn) || (pinEvm && pinEvm !== keys.evm.toLowerCase())) {
      throw new Error(`the stored keys are ${keys.rvn} / ${keys.evm}, not the pinned ${pinRvn || "-"} / ${pinEvm || "-"}; refusing to use them`);
    }
    this.keys = keys;
    return keys;
  }

  // **The door check, where a miner connects.** It turns away only what can
  // never be paid: a worker name that is not a wallet, and a contract address
  // (a transfer to the WETH/GLADOS pair reverts the whole payout for everyone).
  // The holding minimum is *not* a door check any more -- a wallet under it
  // mines and is paid in $GLaDOS; holding GATE_MIN only unlocks the RWA reward
  // choices, which the epoch build decides on the balance at payout time.
  //
  // Answers a refusal to send, or null. An unreadable balance admits: the read
  // here only spares a contract from mining for nothing, and the one read that
  // decides both pay and reward runs again at every epoch, on the balance as it
  // is then.
  // **A wallet's reward choice, per address, in this shard's storage.** Every
  // worker of one address lands on this shard (shardFor hashes the address),
  // so the choice lives beside the work it pays for. The latest greeting wins:
  // a miner switching choice on the mining screen reconnects with the new one.
  // $GLADOS is stored as absence, so the common case costs nothing.
  async rewards() {
    if (!this.rewardMap) {
      const m = new Map();
      for (const [k, v] of await this.ctx.storage.list({ prefix: "reward." })) m.set(k.slice(7), v);
      this.rewardMap = m;
    }
    return this.rewardMap;
  }

  async noteReward(hello) {
    let worker = "", reward;
    try { const p = JSON.parse(hello).params || {}; worker = p.worker || ""; reward = p.reward; } catch { return; }
    const head = worker.split(".")[0];
    if (!/^0x[0-9a-fA-F]{40}$/.test(head)) return;
    const addr = head.toLowerCase();
    const code = rewardOf(reward);
    const m = await this.rewards();
    if ((m.get(addr) || DEFAULT_REWARD) === code) return;
    if (code === DEFAULT_REWARD) { m.delete(addr); await this.ctx.storage.delete(`reward.${addr}`); }
    else { m.set(addr, code); await this.ctx.storage.put(`reward.${addr}`, code); }
    this.log(`[reward] ${addr} is paid in ${code}`);
  }

  noteConn(ws, hello) {
    try {
      const head = ((JSON.parse(hello).params || {}).worker || "").split(".")[0];
      if (/^0x[0-9a-fA-F]{40}$/.test(head)) this.connAddr.set(ws, head.toLowerCase());
    } catch {}
  }

  // A miner's self-reported hashrate, for the combined-pool figure only. Capped
  // at 100 MH/s a socket so one cannot inflate the number to something absurd;
  // a real browser is orders of magnitude under that.
  noteRate(ws, line) {
    try {
      const hs = Number((JSON.parse(line).params || {}).hs);
      this.connRate.set(ws, Number.isFinite(hs) && hs > 0 ? Math.min(hs, 1e8) : 0);
    } catch {}
  }

  // The combined hashrate of every connection on this shard.
  poolHashrate() { let s = 0; for (const v of this.connRate.values()) s += v; return s; }

  // The live revenue figure, pushed to each connected miner. Priced on the
  // treasury shard (SHARDS=1 keeps every miner here); a wallet is shown the
  // amount for the asset it is actually paid in -- its choice at or over the
  // gate, else $GLaDOS -- so the page's line matches the payout. Best-effort:
  // a socket that will not take it is dropped, never retried into a loop.
  async pushEarn() {
    if (!this.runsTreasury || this.conns.size === 0) return;
    const earn = await this.treasurer.earnQuote().catch(() => null);
    if (!earn || !earn.byCode) return;
    const rewards = await this.rewards();
    // The same payout facts ride on every miner's message: the last
    // distribution's recipient count, the cadence, and when it last paid.
    const recipients = earn.recipients || 0, epochMs = earn.epochMs || 0, lastAt = earn.lastAt ?? null, usd = earn.usd || 0;
    for (const [ws] of this.conns) {
      const addr = this.connAddr.get(ws);
      if (!addr) continue;
      const held = (this.balances.get(addr)?.wei ?? 0n) >= this.gateMin;
      const code = held ? (rewards.get(addr) || DEFAULT_REWARD) : DEFAULT_REWARD;
      const raw = earn.byCode[code];
      const value = typeof raw === "number" && raw >= 0 ? raw : 0; // the token amount, 0 before a payout contract exists
      // Sent as soon as there is a dollar figure, even with no contract yet, so
      // a miner's earned-so-far shows while the pool is still accruing RVN.
      if (usd > 0 || value > 0) {
        try { ws.send(JSON.stringify({ method: "glados.earn", params: { asset: code, value, usd, recipients, epochMs, lastAt } }) + "\n"); }
        catch { this.drop(ws); }
      }
    }
  }

  async gate(hello) {
    let worker = "";
    try { worker = JSON.parse(hello).params.worker || ""; } catch {}
    const head = worker.split(".")[0];
    if (!/^0x[0-9a-fA-F]{40}$/.test(head)) {
      return `that is not a wallet address: use your 0x address (42 characters, starting 0x) as the worker name`;
    }
    const addr = head.toLowerCase();
    let hit = this.balances.get(addr);
    if (!(hit && Date.now() - hit.at < GATE_CACHE_MS)) {
      try {
        hit = await this.readGate(addr);
      } catch (e) {
        // **Admitted, not refused, when the chain cannot be read.** This gate
        // only spares a miner hashing for nothing; the one that decides who is
        // paid runs again at every epoch, on the balance and code as they are
        // then. So an RPC outage refusing every miner at the door would be an
        // error message in every miner's face that protects nobody.
        this.log(`[gate] ${head} admitted unchecked: balance unreadable (${e && e.message ? e.message : e}); the payout checks it`);
        return null;
      }
    }
    // A contract is never paid (the payout would revert for everyone if it
    // were the pair), so it is refused at the door rather than mined for nothing.
    if (hit.contract || addr === PAIR_ADDR) {
      return `${short(head)} is a smart contract, not a wallet: use the address of a wallet you hold the keys to`;
    }
    // Under GATE_MIN is no longer refused: the wallet mines and is paid in
    // $GLaDOS, and the holding only decides whether it may be paid in an RWA,
    // settled at payout time. So `hit.wei` is read but never a reason to close.
    return null;
  }

  // **Connections arriving together are checked together.** Every address that
  // asks within GATE_BATCH_MS goes into one GladosReader eth_call (up to 200),
  // so a crowd reconnecting after a deploy is a handful of requests to the
  // public RPC instead of one each -- which is what would get the pool's
  // Cloudflare egress rate-limited, and then every miner admitted unchecked.
  readGate(addr) {
    return new Promise((resolve, reject) => {
      this.gateWait = this.gateWait || new Map();
      const list = this.gateWait.get(addr) || [];
      list.push({ resolve, reject });
      this.gateWait.set(addr, list);
      if (this.gateWait.size >= READ_MAX) this.flushGate();
      else if (!this.gateTimer) this.gateTimer = setTimeout(() => this.flushGate(), GATE_BATCH_MS);
    });
  }

  async flushGate() {
    clearTimeout(this.gateTimer);
    this.gateTimer = null;
    const batch = this.gateWait || new Map();
    this.gateWait = new Map();
    const who = [...batch.keys()];
    if (!who.length) return;
    try {
      // Each endpoint once, without waiting between them: a door check that
      // stalls holds a miner's first job, and one that fails admits anyway.
      const payload = { jsonrpc: "2.0", id: 1, method: "eth_call", params: [{ data: readerData(GLADOS, who) }, "latest"] };
      let v, last;
      for (let k = 0; k < this.rpcs.size && !v; k++) {
        try { v = await attempt(this.rpcs, payload, plainPost, "door check"); } catch (e) { last = e; }
      }
      if (!v) throw last || new Error("no endpoint answered");
      if (v.error !== undefined) throw new Error(v.error);
      const m = readerDecode(who, v.result);
      for (const [a, waiters] of batch) {
        const v = m.get(a);
        if (v.balance === undefined) { for (const w of waiters) w.reject(new Error("balanceOf failed")); continue; }
        const hit = { wei: v.balance, at: Date.now(), contract: v.code === "contract" };
        this.balances.set(a, hit);
        for (const w of waiters) w.resolve(hit);
      }
    } catch (e) {
      for (const waiters of batch.values()) for (const w of waiters) w.reject(e);
    }
  }

  // A connection no longer OPEN, or silent past IDLE_MAX_MS, is closed and
  // dropped so it stops being counted and the upstreams are released when the
  // last real miner leaves. Run at the top of every alarm.
  reap() {
    const now = Date.now(), max = Number(this.env.IDLE_MAX_MS) || IDLE_MAX_MS;
    for (const [ws] of this.conns) {
      const open = ws.readyState === undefined || ws.readyState === 1;
      if (open && now - (this.lastSeen.get(ws) ?? now) <= max) continue;
      try { ws.close(1001, "idle"); } catch {}
      this.drop(ws);
    }
  }

  drop(ws) {
    const id = this.conns.get(ws);
    this.connAddr.delete(ws);
    this.lastSeen.delete(ws);
    this.connRate.delete(ws);
    if (id !== undefined) {
      this.core.close(id);
      this.conns.delete(ws);
      this.ctx.waitUntil(this.saveSoon().catch((e) => this.log(`[edge] ledger not saved on disconnect, the alarm tries again: ${e && e.message ? e.message : e}`)));
      if (this.conns.size === 0) {
        for (const u of this.ups) this.closeUp(u);
      }
    }
  }

  // One upstream connection, reconnecting with backoff while anyone is mining.
  async runUp(u) {
    u.running = true;
    let backoff = 2000;
    while (this.conns.size > 0) {
      try {
        u.sock = connect(u.where);
        u.writer = u.sock.writable.getWriter();
        if (!this.upDeliver(u, this.core.upOpen(u.i))) {
          const reader = u.sock.readable.getReader();
          for (;;) {
            const { value, done } = await reader.read();
            if (done) {
              this.log(`[up ${u.i}] upstream closed the connection`);
              break;
            }
            if (this.upDeliver(u, this.core.upBytes(u.i, value))) break;
            backoff = 2000;
          }
        }
      } catch (e) {
        // The last miner leaving closes the socket under the read, which
        // throws; that is the ordinary way out, not a fault worth a line.
        if (this.conns.size > 0) this.log(`[up ${u.i}] ${e && e.message ? e.message : e}`);
      }
      this.closeUp(u);
      if (this.conns.size === 0) break;
      await new Promise((r) => setTimeout(r, backoff));
      backoff = Math.min(backoff * 2, 60_000);
    }
    u.running = false;
  }

  // Act on what the core answered. `true` when the connection is finished.
  upDeliver(u, r) {
    for (const l of r.log) this.log(l);
    for (const m of r.send) {
      if (u.writer) u.writer.write(new TextEncoder().encode(m)).catch(() => {});
    }
    if (r.work) {
      // Upstream work arriving can make a slot choosable that was not when the
      // rates were read -- the first connection's rates routinely land before
      // any upstream has answered. Re-decide against the rates already held; no
      // fetch, and silence unless the choice actually moved.
      if (this.switching && this.ratesDoc) {
        const again = this.core.rates(this.ratesDoc);
        for (const l of again.log) if (!l.startsWith("[switch] staying")) this.log(l);
      }
      this.pushWork();
    }
    if (r.close) this.log(`[up ${u.i}] ${r.why}`);
    return r.close;
  }

  tickUps() {
    for (const u of this.ups) {
      if (u.writer && this.upDeliver(u, this.core.upTick(u.i))) this.closeUp(u);
    }
  }

  closeUp(u) {
    try { u.writer && u.writer.releaseLock(); } catch {}
    try { u.sock && u.sock.close(); } catch {}
    u.sock = null;
    u.writer = null;
  }

  // **Chunked, because one value is capped at 128 KiB and a ledger is not.**
  // A row is about 130 bytes, so a single value held some 1,000 tallies --
  // about 250 miners switching across four coins -- and past that every save
  // was refused while the object kept serving from memory, until the next
  // deploy or eviction dropped everything since the last save that fitted.
  // Found by the load test. The head and its chunks go in one atomic put.
  save() {
    const text = this.core.ledger(1, Math.floor(Date.now() / 1000));
    this.tallies = (text.match(/"worker":/g) || []).length;
    if (this.tallies >= MAX_TALLIES * 0.9 && !this.warnedFull) {
      this.warnedFull = true;
      this.log(`[edge] ${this.tallies} of ${MAX_TALLIES} tallies: past the cap, new miners' shares are counted but credited to nobody -- raise SHARDS`);
    }
    return this.ctx.storage.put(textChunks("ledger", text));
  }

  // ---------------------------------------------------------- maintenance
  // "control" holds the state; every other object holds a copy and acts on it.
  // Only the router reaches these paths, and only /maintenance from outside,
  // after the token is checked.
  async maintRoute(request, path) {
    const name = String(this.ctx.id.name || "");
    if (path === "/maintenance") {
      if (name !== "control") return new Response("not here", { status: 404 });
      return Response.json(this.maint || { on: false, since: 0 });
    }
    if (request.method !== "POST") return new Response("POST", { status: 405 });
    let s = null;
    try { s = validMaint(await request.json()); } catch {}
    if (!s) return new Response("not a maintenance state", { status: 400 });
    if (path === "/maintenance/set") {
      if (name !== "control") return new Response("not here", { status: 404 });
      // Declared again while already down -- a new reason, a later time -- it
      // has been down since the first declaration.
      if (s.on && this.maint && this.maint.on) s.since = this.maint.since;
      await this.ctx.storage.put(MAINT_KEY, s);
      this.maint = s;
      return Response.json(s);
    }
    return Response.json(await this.applyMaint(s));
  }

  // On: tell everyone, keep crediting for the drain, then save once and close.
  // Off: forget it; miners come back on their own.
  async applyMaint(s) {
    const was = this.maint && this.maint.on;
    this.maint = s;
    this.maintAt = Date.now();
    await this.ctx.storage.put(MAINT_KEY, s);
    const connections = this.conns.size;
    if (!s.on) {
      if (was) this.log(`[maint] off; ${connections} still connected`);
      return { connections, draining: false };
    }
    this.log(`[maint] on: ${s.reason}; ${connections} miner(s) told, closing in ${s.drainMs / 1000} s`);
    const line = notice(s, Date.now());
    for (const [ws] of this.conns) { try { ws.send(line); } catch { this.drop(ws); } }
    const drainId = (this.drainId || 0) + 1;
    this.drainId = drainId;
    this.ctx.waitUntil(this.drain(drainId, s.drainMs));
    return { connections, draining: true };
  }

  async drain(drainId, ms) {
    if (ms > 0) await new Promise((r) => setTimeout(r, ms));
    // Declared off, or declared again, during the wait: that declaration decides.
    if (this.drainId !== drainId || !(this.maint && this.maint.on)) return;
    this.tickUps(); // a share that beat upstream's target goes out before the sockets do
    const why = closeReason(this.maint);
    let n = 0;
    for (const [ws] of this.conns) { try { ws.close(1013, why); } catch {} this.drop(ws); n++; }
    for (const u of this.ups) this.closeUp(u);
    try { await this.saveSoon(); } catch (e) { this.log(`[maint] the record was not saved on the way down (${e && e.message ? e.message : e}); the alarm tries again`); }
    this.log(`[maint] drained: ${n} miner(s) closed, record saved`);
  }

  // Whether this shard's copy still says down, asking "control" at most every
  // half minute: a shard that missed the "off" must not refuse miners for ever.
  async stillDown() {
    if (String(this.ctx.id.name || "") === "control" || Date.now() - (this.maintAt || 0) < 30_000) return true;
    this.maintAt = Date.now();
    try {
      const s = validMaint(await (await this.env.POOL.get(this.env.POOL.idFromName("control")).fetch(new Request("https://control/maintenance"))).json());
      if (s && !s.on) { this.maint = s; await this.ctx.storage.put(MAINT_KEY, s); this.log("[maint] control says the pool is up; taking miners again"); return false; }
    } catch (e) { this.log(`[maint] control unreadable, staying down: ${e && e.message ? e.message : e}`); }
    return true;
  }

  // **One save at a time, and a burst of reasons to save is one more save.**
  // Every disconnect saves, so a crowd leaving at once -- a deploy, a network
  // blip, a maintenance drain -- was one full write of the share log per miner:
  // measured, 300 disconnects, 300 writes, each the whole ledger in chunks,
  // which is the daily write allowance spent in seconds. Now a save in flight
  // absorbs every request behind it into a single follow-up that sees them all.
  saveSoon() {
    if (this.saving) { this.saveAgain = true; return this.saving; }
    this.saving = (async () => {
      try {
        do { this.saveAgain = false; await this.save(); } while (this.saveAgain);
      } finally { this.saving = null; }
    })();
    return this.saving;
  }

  async arm() {
    if ((await this.ctx.storage.getAlarm()) === null) {
      await this.ctx.storage.setAlarm(Date.now() + (Number(this.env.JOB_PERIOD_MS) || JOB_PERIOD_MS));
    }
  }

  // The treasury runs in "main" only, and its clock must not depend on anybody
  // mining: a payout that waits for the next miner to connect is a payout that
  // never comes on a quiet day.
  get runsTreasury() {
    return String(this.ctx.id.name || "") === "main" && (this.env.TREASURY || "off") !== "off";
  }

  // Every job period: fresh work for quiet miners, and the ledger to storage.
  // Every RATES_MS at most, while anyone is mining: fetch zpool's rates and let
  // the core re-choose. A failed fetch keeps the current choice; it is not a
  // reason to move every miner.
  async refreshRates() {
    if (!this.switching || Date.now() - this.ratesAt < RATES_MS) return;
    this.ratesAt = Date.now();
    try {
      const res = await fetch("https://zpool.ca/api/status", { headers: { "user-agent": "glados-pool" } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.ratesDoc = await res.text();
      const r = this.core.rates(this.ratesDoc);
      for (const l of r.log) this.log(l);
      if (r.work) this.pushWork();
    } catch (e) {
      this.log(`[switch] rates unavailable, staying: ${e && e.message ? e.message : e}`);
    }
  }

  pushWork() {
    for (const [ws, id] of this.conns) {
      const w = this.core.work(id);
      try {
        for (const m of w.send) ws.send(m);
      } catch {
        this.drop(ws);
      }
    }
  }

  // **The next alarm is set however this one went.** It is the only clock the
  // object has: a save that threw used to end the handler before the next
  // alarm was set, so one refused write -- storage at its daily allowance --
  // stopped idle work for every miner and stopped the treasury, until somebody
  // happened to connect. A failed save is logged and left to the next period,
  // which saves the whole ledger again; nothing is rethrown, since a thrown
  // alarm is retried by Cloudflare and each retry is more of the same refusal.
  async alarm() {
    // The status object only probes; it never runs a mining period.
    if (String(this.ctx.id.name || "") === "status") {
      try { await statusTick(this.env, this.ctx.storage); }
      catch (e) { this.log(`[status] probe failed: ${e && e.message ? e.message : e}`); }
      finally { await this.ctx.storage.setAlarm(Date.now() + STATUS_EVERY_MS); }
      return;
    }
    // Each phase is timed, and any period over a second says which phase took
    // it: the object was being reset for exceeding its CPU limit and nothing
    // said where the time went.
    const t0 = Date.now(), lap = {};
    const mark = (k, at) => { lap[k] = Date.now() - at; };
    try {
      this.reap(); // drop half-open sockets before anything counts or iterates them
      let at = Date.now();
      await this.refreshRates();
      mark("rates", at);
      this.tickUps();
      for (const [ws, id] of this.conns) {
        const r = this.core.idle(id);
        for (const l of r.log) this.log(l);
        try {
          for (const m of r.send) ws.send(m);
        } catch {
          this.drop(ws);
        }
      }
      at = Date.now();
      try {
        await this.save();
      } catch (e) {
        this.log(`[edge] ledger not saved this period, the next one tries again: ${e && e.message ? e.message : e}`);
      }
      mark("save", at);
      at = Date.now();
      // The treasury is isolated: a throw here (a transient read, a missing key)
      // must not skip the rest of the period -- above all the earnings push, so
      // a miner keeps seeing a figure even while a payout step is failing.
      if (this.runsTreasury && this.maint && this.maint.on) this.log(`[treasury] paused for maintenance: ${this.maint.reason}`);
      else if (this.runsTreasury) { try { await this.treasurer.tick(); } catch (e) { this.log(`[edge] treasury tick failed this period: ${e && e.message ? e.message : e}`); } }
      mark("treasury", at);
      at = Date.now();
      try { await this.pushEarn(); } catch (e) { this.log(`[edge] earn push failed this period: ${e && e.message ? e.message : e}`); }
      mark("earn", at);
    } catch (e) {
      this.log(`[edge] alarm failed: ${e && e.message ? e.message : e}`);
    } finally {
      const total = Date.now() - t0;
      if (total > 1000) this.log(`[edge] slow alarm ${total} ms: ${Object.entries(lap).map(([k, v]) => k + " " + v).join(", ")}; ${this.conns.size} conn(s), ${this.tallies ?? "?"} tallies`);
      try {
        if (this.conns.size > 0) await this.ctx.storage.setAlarm(Date.now() + (Number(this.env.JOB_PERIOD_MS) || JOB_PERIOD_MS));
        else if (this.runsTreasury) await this.ctx.storage.setAlarm(Date.now() + 5 * 60_000);
      } catch (e) {
        this.log(`[edge] next alarm not set (${e && e.message ? e.message : e}); the next connection sets one`);
      }
    }
  }
}

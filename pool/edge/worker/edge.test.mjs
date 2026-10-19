// node pool/edge/worker/edge.test.mjs
//
// worker.js -- the router and the mining object -- under edge-sim.mjs, with
// the failures it claims to survive made to happen: the daily allowance spent,
// an object reset mid-call, a share log that will not load or has been
// tampered with, storage refusing writes for a while, eviction between a share
// and its save, sockets that vanish without a close, lines past the cap, one
// shard of several down. Each case checks what is true afterwards -- what a
// miner was told, what storage holds, what the next object restores -- not
// what the pool logged about itself.
//
// Needs the wasm core built: cargo build --release --target wasm32-unknown-unknown (in pool/edge).
import { createHash } from "node:crypto";
import * as sim from "./edge-sim.mjs";

const { default: edge, Pool } = await import("./worker.js");

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.info(`${c ? "ok  " : "FAIL"}  ${w}`); };

// The pool logs freely; the tests read the log rather than print it.
const logs = [];
console.log = (...a) => logs.push(a.join(" "));
console.error = (...a) => logs.push(a.join(" "));
const logged = (re) => logs.some((l) => re.test(l));

// ------------------------------------------------------------------ a world
function world(extra = {}) {
  const env = { COINS: "sha256d:sha256d:1", SHARE_SECONDS: "45", ...extra };
  env.POOL = new sim.Namespace(Pool, env);
  return env;
}
const addr = (i) => "0x" + i.toString(16).padStart(40, "0");
const get = (env, path, init) => edge.fetch(new Request(`https://pool.test${path}`, init), env);

async function miner(env, who, path = "/mine") {
  const res = await get(env, `${path}?w=${who}`, { headers: { Upgrade: "websocket" } });
  if (res.status !== 101) return { res };
  const c = res.webSocket;
  c.send(JSON.stringify({ id: 1, method: "glados.hello", params: { v: 1, worker: who, agent: "edge.test" } }) + "\n");
  await sim.sleep(5);
  return { c, res };
}
const jobOf = (c) => c.lines().filter((m) => m.method === "glados.job").map((m) => m.params).pop();

// Nonces under a sha256d job's target, past `from`.
function mine(job, count, from = 0) {
  const hdr = Buffer.from(job.header, "hex"), out = [];
  for (let n = from; out.length < count; n++) {
    hdr.writeUInt32LE(n, 76);
    const h = createHash("sha256").update(createHash("sha256").update(hdr).digest()).digest().reverse().toString("hex");
    if (h <= job.target) out.push(n);
  }
  return out;
}
let rid = 100;
async function submit(c, job, nonces) {
  const ids = [];
  for (const n of nonces) { const id = rid++; ids.push(id); c.send(JSON.stringify({ id, method: "glados.submit", params: { job: job.job, nonce: n.toString(16).padStart(8, "0") } }) + "\n"); }
  await sim.sleep(5);
  return ids.map((id) => c.lines().find((m) => m.id === id));
}
// Work per worker in the window, as /ledger.json says -- the public record.
async function work(env) {
  const doc = JSON.parse(await (await get(env, "/ledger.json")).text());
  const m = new Map();
  for (const w of doc.windows || []) for (const s of w.shares) m.set(s.worker, (m.get(s.worker) || 0) + s.work);
  return m;
}
const quotaError = () => new Error("Exceeded allowed duration in Durable Objects free tier.");
const resetError = () => Object.assign(new Error("Durable Object reset because its code was updated."), { retryable: true });

// ======================================================= the record survives
{
  const env = world();
  const A = addr(0xa1);
  const { c } = await miner(env, A);
  const job = jobOf(c);
  const v = await submit(c, job, mine(job, 3));
  ok(v.every((r) => r && r.result), "three shares accepted");
  c.close(1000, "bye");
  await env.POOL.settle("main");
  env.POOL.evict("main");
  ok((await work(env)).get(A) === 768, "a miner leaves, the object is evicted, and the next object restores its work");

  // The same nonces again, after the eviction wiped the core's memory of them.
  // A restart takes over a second, and a local job's header carries the time,
  // so the honest replay is one a second or more later.
  await sim.sleep(1100);
  const again = await miner(env, A);
  const j2 = jobOf(again.c);
  const replay = await submit(again.c, j2, mine(job, 3));
  const w2 = (await work(env)).get(A);
  ok(w2 === 768,
    `the same shares replayed after an eviction are not credited twice (work ${w2}; job ${j2.job}, header ${j2.header === job.header ? "unchanged" : "changed"})`);
  void replay;
  again.c.close();
}

// ======================================================= a share log that will not load
{
  const env = world();
  const A = addr(0xb2);
  const { c } = await miner(env, A);
  const job = jobOf(c);
  await submit(c, job, mine(job, 2));
  c.close();
  await env.POOL.settle("main");
  env.POOL.evict("main");
  const st = env.POOL.storage("main");
  const before = new Map(st.data);

  // A chunk gone.
  st.data.delete("ledger#0");
  const r = await get(env, "/status");
  const body = await r.json().catch(() => ({}));
  ok(r.status === 503 && /share log did not load/.test(body.why || ""), `a missing ledger chunk: every request 503s with the reason (${r.status}: ${body.why})`);
  const m = await miner(env, A);
  ok(!m.c && m.res.status === 503, "and no miner is let in to be credited into an empty record");
  try { await env.POOL.alarm("main"); } catch {}
  ok(!st.data.has("ledger#0") && st.data.get("ledger") && JSON.stringify(st.data.get("ledger")) === JSON.stringify(before.get("ledger")), "and nothing overwrites what is left of the stored log, alarm or not");

  // A chunk present but tampered with: the hard case, since nothing is missing.
  st.data = new Map(before);
  const text = st.data.get("ledger#0");
  st.data.set("ledger#0", text.replace(/"work": 256/, '"work": 999999'));
  env.POOL.evict("main");
  const r2 = await get(env, "/ledger.json");
  const t2 = await r2.text();
  const credited = r2.status === 200 ? (JSON.parse(t2).windows || []).flatMap((w) => w.shares).filter((s) => s.worker === A).reduce((a, s) => a + s.work, 0) : null;
  ok(r2.status === 503 || credited === 512,
    `a ledger chunk edited in storage (one share's work raised to 999999) is refused or ignored, never served (status ${r2.status}, work ${credited})`);
  if (r2.status === 200) {
    await env.POOL.alarm("main").catch(() => {});
    const now = st.data.get("ledger#0");
    ok(now === text.replace(/"work": 256/, '"work": 999999') || /"work": 256/.test(now),
      "and the tampered log is not silently replaced by an empty one");
  }

  // Truncated: a chunk cut short, as a half-applied write would leave it.
  st.data = new Map(before);
  st.data.set("ledger#0", text.slice(0, Math.floor(text.length / 2)));
  env.POOL.evict("main");
  const r3 = await get(env, "/status");
  const b3 = await r3.json().catch(() => ({}));
  ok(r3.status === 503 && /share log did not load/.test(b3.why || ""), `a truncated ledger chunk is refused with the reason, not loaded as an empty pool (${r3.status}${b3.why ? ": " + b3.why : ""})`);
  if (r3.status !== 503) {
    // Show what would happen next: the first save writes the empty record over the stored one.
    await env.POOL.alarm("main").catch(() => {});
    const after = st.data.get("ledger#0") || "";
    ok(after.length >= text.length / 2, `and the next save would not overwrite the stored log with an empty one (stored ${after.length} bytes, was ${text.length})`);
  }
}

// ======================================================= storage refusing writes
{
  const env = world({ JOB_PERIOD_MS: "30000" });
  const A = addr(0xc3);
  const { c } = await miner(env, A);
  const job = jobOf(c);
  await submit(c, job, mine(job, 2));
  const st = env.POOL.storage("main");
  st.fail("put", quotaError);
  logs.length = 0;
  await env.POOL.alarm("main");
  ok(st.alarm !== null, "storage refusing every write: the alarm still sets the next alarm");
  ok(logged(/ledger not saved this period/), "and says the ledger was not saved");
  ok(c.readyState === 1, "and the miner stays connected");
  await submit(c, job, mine(job, 2, mine(job, 2).pop() + 1));
  st.heal();
  await env.POOL.alarm("main");
  env.POOL.evict("main");
  ok((await work(env)).get(A) === 1024, "once storage heals, the next period saves everything held in memory meanwhile");

  // setAlarm itself refused: the handler must not throw (a thrown alarm is retried into the same refusal).
  const { c: c2 } = await miner(env, A);
  st.fail("setAlarm", quotaError);
  let threw = false;
  try { await env.POOL.alarm("main"); } catch { threw = true; }
  ok(!threw && logged(/next alarm not set/), "setAlarm refused: the alarm handler does not throw, and says so");
  st.heal();
  c2.close();
}

// ======================================================= the ledger past one value
{
  const env = world();
  const st = env.POOL.storage("main");
  const socks = [];
  // Enough distinct workers that the ledger passes 128 KiB, so it must chunk.
  for (let i = 0; i < 700; i++) {
    const { c } = await miner(env, addr(0x1000 + i));
    const job = jobOf(c);
    await submit(c, job, mine(job, 1, i * 300));
    socks.push(c);
  }
  await env.POOL.alarm("main");
  const head = st.data.get("ledger");
  ok(head && head.chunks > 1, `a ${head && head.chunks}-chunk ledger is saved under the 128 KiB value cap`);
  for (const c of socks) c.close();
  await env.POOL.settle("main");
  env.POOL.evict("main");
  const w = await work(env);
  ok(w.size === 700 && [...w.values()].every((x) => x === 256), `and all 700 workers come back after an eviction (${w.size})`);
}

// ======================================================= a crowd leaving at once
{
  const env = world();
  const st = env.POOL.storage("main");
  const socks = [];
  for (let i = 0; i < 300; i++) {
    const { c } = await miner(env, addr(0x9000 + i));
    const job = jobOf(c);
    await submit(c, job, mine(job, 1, i * 300));
    socks.push(c);
  }
  st.ops.length = 0;
  for (const c of socks) c.close(); // a deploy, a blip, a drain: everyone at once
  await env.POOL.settle("main");
  const puts = st.ops.filter(([o, k]) => o === "put" && String(k).startsWith("ledger")).length;
  ok(puts <= 3, `300 miners disconnecting at once cost ${puts} writes of the share log, not 300`);
  env.POOL.evict("main");
  const w = await work(env);
  ok(w.size === 300, `and the last of those writes holds every one of them (${w.size})`);
}

// ======================================================= sockets
{
  const env = world({ IDLE_MAX_MS: "1" });
  const A = addr(0xd4);
  const { c } = await miner(env, A);
  c.vanish(); // no close frame: the pool's end stays OPEN
  await sim.sleep(5);
  const s1 = await (await get(env, "/status?a")).json();
  await env.POOL.alarm("main");
  const s2 = await (await get(env, "/status?b")).json();
  ok(s1.connections === 1 && s2.connections === 0, `a socket that vanished without a close is reaped at the next alarm (${s1.connections} -> ${s2.connections})`);

  const env2 = world();
  const { c: big } = await miner(env2, A);
  big.send("x".repeat(131073));
  await sim.sleep(5);
  ok(big.closed && big.closed.code === 1009, `a line past the cap closes the socket with 1009 (${big.closed && big.closed.code})`);
  const s3 = await (await get(env2, "/status?c")).json();
  ok(s3.connections === 0, "and it stops being counted");

  // A flood of junk lines in one message: the socket survives or is closed, the object never throws.
  const { c: junk } = await miner(env2, addr(0xd5));
  let threw = false;
  try { junk.send(["null", "[]", "{", '{"method":"glados.submit"}', '{"id":1,"method":"glados.submit","params":{"job":7,"nonce":[]}}', "\u0000\u0001", '{"method":"glados.hello","params":{"worker":"' + "0x".padEnd(5000, "f") + '"}}'].join("\n")); }
  catch { threw = true; }
  await sim.sleep(5);
  ok(!threw, `junk lines of every shape do not throw out of the object (socket ${junk.readyState === 1 ? "open" : "closed " + (junk.closed && junk.closed.code)})`);
}

// ======================================================= the router
{
  const env = world();
  // The daily allowance: one object call sees it, then the isolate stops asking.
  env.POOL.failCalls("main", quotaError, 1);
  const r1 = await get(env, "/status?q1");
  const b1 = await r1.json();
  ok(r1.status === 503 && /daily Cloudflare allowance/.test(b1.why) && r1.headers.get("retry-after") === "300", `allowance spent: 503, said plainly, Retry-After 300 (${r1.status})`);
  const calls = env.POOL.calls.length;
  const r2 = await get(env, "/mine", { headers: { Upgrade: "websocket" } });
  ok(r2.status === 503 && env.POOL.calls.length === calls, "and for the next while every request is answered without calling an object at all");
}
{
  // That break is per isolate and module-level: wait it out the honest way is
  // 30 s, so the next cases use a fresh import of the router.
  const fresh = async () => (await import(`./worker.js?fresh=${Math.random()}`)).default;
  const edge2 = await fresh();
  const env = world();
  const g = (p, i) => edge2.fetch(new Request(`https://pool.test${p}`, i), env);

  // A stray "exceeded" that is not the allowance must not shut the pool.
  env.POOL.failCalls("main", () => new RangeError("Maximum call stack size exceeded"), 1);
  const r = await g("/status?s1");
  ok(r.status === 503, "an internal error answers 503");
  const r2 = await g("/status?s2");
  ok(r2.status === 200, "but 'Maximum call stack size exceeded' does not trip the allowance break");
  const body = await r.text();
  ok(!/call stack/.test(body), "and the internal message is not echoed to the public");

  // An object reset by a deploy: a GET is retried once, on a fresh stub.
  env.POOL.failCalls("main", resetError, 1);
  const r3 = await g("/ledger.json");
  ok(r3.status === 200, "an object reset mid-call (retryable): a GET is retried once and answers");
  env.POOL.failCalls("main", resetError, 2);
  const n0 = env.POOL.calls.length;
  const r4 = await g("/ledger.json?2");
  ok(r4.status === 503 && env.POOL.calls.length - n0 === 2, `retried exactly once, not looped (${env.POOL.calls.length - n0} calls)`);
  // Overloaded is not retried: retrying an overloaded object is the overload.
  env.POOL.failCalls("main", () => Object.assign(new Error("overloaded"), { retryable: true, overloaded: true }), 1);
  const n1 = env.POOL.calls.length;
  const r5 = await g("/ledger.json?3");
  ok(r5.status === 503 && env.POOL.calls.length - n1 === 1 && /overloaded/.test((await r5.json()).why), "an overloaded object is not retried, and the miner is told to back off");
  // A POST is never retried: its body may be half-read.
  env.POOL.failCalls("progress", resetError, 1);
  const n2 = env.POOL.calls.length;
  const r6 = await g(`/progress?address=${addr(1)}`, { method: "POST", body: JSON.stringify({ rec: { est: 1 } }), headers: { "content-type": "application/json" } });
  ok(r6.status === 503 && env.POOL.calls.length - n2 === 1, "a POST into a reset object is not retried");
  env.POOL.healCalls();
}
{
  // Several shards, one down.
  const edge3 = (await import(`./worker.js?fresh=${Math.random()}`)).default;
  const env = world({ SHARDS: "3" });
  const g = (p) => edge3.fetch(new Request(`https://pool.test${p}`), env);
  const socks = [];
  for (let i = 0; i < 12; i++) socks.push((await (async () => { const r = await edge3.fetch(new Request(`https://pool.test/mine?w=${addr(0x500 + i)}`, { headers: { Upgrade: "websocket" } }), env); return r.webSocket; })()));
  env.POOL.failCalls("shard-1", () => new Error("boom"), Infinity);
  const r = await g("/status?x");
  const s = await r.json();
  ok(r.status === 200 && Array.isArray(s.down) && s.down.length === 1 && /shard-1/.test(s.down[0]), `one shard of three down: /status still answers and names it (${JSON.stringify(s.down)})`);
  ok(s.perShard.filter((x) => x === null).length === 1 && s.connections === s.perShard.reduce((a, x) => a + (x || 0), 0), "counting the shards that answered");
  env.POOL.failCalls("*", () => new Error("boom"), Infinity);
  const r2 = await g("/status?y");
  ok(r2.status === 503, "all of them down: /status says the pool is down");
  env.POOL.healCalls();
  const bad = await g("/ledger.json?shard=../../treasury");
  ok(bad.status === 404, "a shard name that is not one is refused");
}

// ======================================================= a public read that breaks the mining object
// Cloudflare terminates an object whose blockConcurrencyWhile callback throws:
// every miner on it drops and every share since the last save is gone. The
// treasury's key read throws on purpose when it cannot be sure (keys missing
// while pinned, storage unreadable) -- so it must not throw *there*.
{
  const edgeT = (await import(`./worker.js?treasury=${Math.random()}`)).default;
  const g = (env, p) => edgeT.fetch(new Request(`https://pool.test${p}`), env);
  const minerT = async (env, who) => {
    const r = await edgeT.fetch(new Request(`https://pool.test/mine?w=${who}`, { headers: { Upgrade: "websocket" } }), env);
    const c = r.webSocket;
    c.send(JSON.stringify({ id: 1, method: "glados.hello", params: { v: 1, worker: who, agent: "edge.test" } }) + "\n");
    await sim.sleep(5);
    return c;
  };
  // Keys missing while the addresses are pinned.
  const env = world({ TREASURY: "live", TREASURY_RVN: "RCmy8SHUHnACVobxf8vpzum5ms7nPXFU4f" });
  const A = addr(0xf1);
  const c = await minerT(env, A);
  const job = jobOf(c);
  await submit(c, job, mine(job, 3));
  const r = await g(env, "/treasury");
  await sim.sleep(10);
  ok(r.status === 503, `GET /treasury with the keys missing while pinned answers 503 (${r.status})`);
  ok((env.POOL.resets.get("main") || 0) === 0 && c.readyState === 1, `and does not reset the mining object: the miner stays connected (resets ${env.POOL.resets.get("main") || 0}, socket ${c.readyState === 1 ? "open" : "closed " + (c.closed && c.closed.code)})`);
  const doc = JSON.parse(await (await g(env, "/ledger.json")).text());
  const w = (doc.windows || []).flatMap((x) => x.shares).filter((s) => s.worker === A).reduce((a, s) => a + s.work, 0);
  ok(w === 768, `and its unsaved shares are still there (${w})`);

  // Keys present, storage failing for a moment.
  const env2 = world({ TREASURY: "live" });
  const ok1 = await g(env2, "/treasury");
  ok(ok1.status === 200, "keys made on first use");
  env2.POOL.evict("main");
  const c2 = await minerT(env2, A);
  const j2 = jobOf(c2);
  await submit(c2, j2, mine(j2, 2));
  env2.POOL.storage("main").fail("get", () => new Error("storage temporarily unavailable"), { times: 1, match: (k) => k === "treasury.keys" });
  const r2 = await g(env2, "/treasury");
  await sim.sleep(10);
  ok(r2.status === 503 && (env2.POOL.resets.get("main") || 0) === 0 && c2.readyState === 1, `a storage blip during GET /treasury: 503, no reset, the miner stays (${r2.status}, resets ${env2.POOL.resets.get("main") || 0})`);
  const r3 = await g(env2, "/treasury");
  ok(r3.status === 200, "and the next request reads the keys");
  c.close(); c2.close();
}

// ======================================================= tripping the breaker from outside
// The allowance break shuts every route for 30 s on an error whose message
// names the allowance. If any client-controlled text could reach the router's
// catch inside an error message, one request every 30 s would keep the whole
// pool down. Every input a client controls gets the phrase.
{
  const edgeB = (await import(`./worker.js?breaker=${Math.random()}`)).default;
  const env = world({ MAINT_TOKEN: "x".repeat(32) });
  const P = "Exceeded allowed duration in Durable Objects free tier quota daily limit";
  const enc = encodeURIComponent(P);
  const g = (p, i) => edgeB.fetch(new Request(`https://pool.test${p}`, i), env);
  const json = { "content-type": "application/json" };
  const attempts = [
    () => g(`/${enc}`), () => g(`/status?${enc}=${enc}`), () => g(`/ledger.json?shard=${enc}`),
    () => g(`/progress?address=${enc}&asset=${enc}`), () => g(`/progress?address=${addr(1)}&asset=glados`, { method: "POST", headers: json, body: `{"rec": ${P}}` }),
    () => g(`/progress?address=${addr(1)}`, { method: "POST", headers: json, body: JSON.stringify({ rec: { est: P, series: [{ t: P, v: P }] } }) }),
    () => g(`/maintenance`, { method: "POST", headers: { ...json, authorization: `Bearer ${P}` }, body: `{"on": ${P}` }),
    () => g(`/maintenance`, { method: "POST", headers: { ...json, authorization: "Bearer " + "x".repeat(32) }, body: JSON.stringify({ on: true, reason: P, minutes: P }) }),
    () => g(`/mine?w=${enc}`, { headers: { Upgrade: "websocket", "CF-Connecting-IP": P } }),
    () => g(`/mine/gpu?w=${enc}`, { headers: { Upgrade: P } }),
    () => g(`/treasury?${enc}`), () => g(`/payouts.json`, { method: "POST", body: P }), () => g(`/health.json?${enc}`, { method: "DELETE" }),
  ];
  for (const a of attempts) { try { await a(); } catch {} }
  const ws = await edgeB.fetch(new Request(`https://pool.test/mine`, { headers: { Upgrade: "websocket" } }), env);
  if (ws.webSocket) {
    for (const m of [P, JSON.stringify({ id: 1, method: "glados.hello", params: { worker: P, reward: P } }), JSON.stringify({ id: 2, method: P, params: P })]) { try { ws.webSocket.send(m + "\n"); } catch {} }
    await sim.sleep(10);
  }
  const after = await g("/status?after");
  const doc = await after.json().catch(() => ({}));
  ok(after.status === 200, `the allowance phrase in every path, query, header, body and socket message a client controls does not trip the break (${after.status}${doc.why ? ": " + doc.why : ""})`);
}

// ======================================================= /progress, from a hostile client
{
  const edge4 = (await import(`./worker.js?fresh=${Math.random()}`)).default;
  const env = world();
  const A = addr(0xe7);
  const post = (rec, a = A) => edge4.fetch(new Request(`https://pool.test/progress?address=${a}&asset=glados`, { method: "POST", body: JSON.stringify({ rec }), headers: { "content-type": "application/json" } }), env);
  const series = Array.from({ length: 5000 }, (_, i) => ({ t: i, v: i }));
  const r1 = await (await post({ est: 1e300, shares: -5, hashes: "NaN", seconds: Infinity, rate: 7, series, __proto__: { polluted: 1 } })).json();
  ok(r1.rec && r1.rec.est === 1e15 && r1.rec.shares === 0 && r1.rec.hashes === 0 && r1.rec.seconds === 0 && r1.rec.series.length === 200, `hostile numbers are capped and zeroed, the series cut to 200 (${JSON.stringify({ ...r1.rec, series: r1.rec.series.length })})`);
  ok(({}).polluted === undefined, "no prototype pollution");
  const r2 = await (await post({ est: 3 })).json();
  ok(r2.rec.est === 1e15, "a lower figure never lowers the stored line");
  const up = await (await post({ est: 1 }, A.toUpperCase().replace("0X", "0x"))).json();
  ok(up.rec && up.rec.est === 1e15, "the same address in capitals is the same record");
  const huge = await edge4.fetch(new Request(`https://pool.test/progress?address=${A}`, { method: "POST", body: "{".repeat(1e6), headers: { "content-type": "application/json" } }), env);
  ok(huge.status === 400, `a megabyte of garbage is a 400 (${huge.status})`);
  const bad = await edge4.fetch(new Request(`https://pool.test/progress?address=0x123`), env);
  ok(bad.status === 400, "a malformed address is a 400");
}

console.info(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

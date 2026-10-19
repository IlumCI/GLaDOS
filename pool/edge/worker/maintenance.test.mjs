// node pool/edge/worker/maintenance.test.mjs
//
// Maintenance, declared and lifted against worker.js and pearl-pool.js under
// edge-sim.mjs: who may declare it, what a miner connected at the time sees
// and is credited, what a miner arriving sees, that the record and the shares
// owed upstream survive the way down, that payouts stop and start, and the
// awkward cases -- a shard that misses the "off", an object evicted while
// down, the state object itself unreachable, "off" during the drain.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import * as sim from "./edge-sim.mjs";
import { parseMaint, validMaint, tokenOk, closeReason, retryAfter, REASON_MAX } from "./maintenance.js";

const { default: edge, Pool, PearlPool } = await import("./worker.js");

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.info(`${c ? "ok  " : "FAIL"}  ${w}`); };
const logs = [];
console.log = (...a) => logs.push(a.join(" "));
console.error = (...a) => logs.push(a.join(" "));

const TOKEN = "t".repeat(20) + "-maintenance-secret";
const auth = { authorization: `Bearer ${TOKEN}`, "content-type": "application/json" };
const addr = (i) => "0x" + i.toString(16).padStart(40, "0");

// ================================================================ the rules
{
  const now = 1_000_000;
  const bad = [null, [], "on", {}, { on: "yes" }, { on: true }, { on: true, reason: "   " }, { on: true, reason: 7 },
    { on: true, reason: "x", minutes: 0 }, { on: true, reason: "x", minutes: -5 }, { on: true, reason: "x", minutes: "30" },
    { on: true, reason: "x", minutes: 1e9 }, { on: true, reason: "x", minutes: NaN }, { on: true, reason: "x", drainSeconds: 121 },
    { on: true, reason: "x", drainSeconds: -1 }, { on: true, reason: "x", drainSeconds: "5" }];
  ok(bad.every((b) => parseMaint(b, now).error), `${bad.length} malformed declarations are each refused with a reason`);
  const p = parseMaint({ on: true, reason: "line one\nline two \u0007 end", minutes: 30, drainSeconds: 0 }, now).state;
  ok(p && p.reason === "line one line two end" && p.until === now + 1_800_000 && p.drainMs === 0, `a reason is made one printable line (${JSON.stringify(p && p.reason)})`);
  const long = parseMaint({ on: true, reason: "🔧".repeat(500) }, now).state;
  ok([...long.reason].length === REASON_MAX && !/�/.test(long.reason) && !/[\uD800-\uDBFF]$/.test(long.reason), "a long reason is cut by character, never through an emoji");
  const cr = closeReason({ reason: "🔧".repeat(100) });
  const bytes = new TextEncoder().encode(cr);
  ok(bytes.length <= 123 && !new TextDecoder().decode(bytes).includes("�"), `a close reason fits a close frame's 123 bytes and stays valid UTF-8 (${bytes.length} bytes)`);
  ok(validMaint({ on: true }) === null && validMaint("x") === null && validMaint({ on: true, reason: "r", drainMs: 1e12 }).drainMs === 120_000, "storage holding something that is not a state is not trusted");
  ok(retryAfter({ until: now + 10 * 60_000 }, now) === 600 && retryAfter({ until: now - 1 }, now) === 120 && retryAfter({ until: now + 1e9 }, now) === 3600 && retryAfter({ until: now + 1000 }, now) === 30, "Retry-After: the time left, between 30 s and an hour; overdue is two minutes");

  ok(!tokenOk(`Bearer ${TOKEN}`, undefined) && !tokenOk(`Bearer short`, "short"), "no secret, or a secret under 16 characters, admits nobody -- even a match");
  ok(tokenOk(`Bearer ${TOKEN}`, TOKEN), "the right token admits");
  const near = [TOKEN.slice(0, -1), TOKEN + "x", TOKEN.toUpperCase(), ` ${TOKEN}`, "", TOKEN.slice(1)];
  ok(near.every((t) => !tokenOk(`Bearer ${t}`, TOKEN)) && !tokenOk(TOKEN, TOKEN) && !tokenOk(`bearer ${TOKEN}`, TOKEN) && !tokenOk(`Basic ${TOKEN}`, TOKEN), "every near miss is refused: one short, one long, case, space, no scheme, another scheme");
}

// ================================================================ a pool to take down
function world(extra = {}) {
  const env = { COINS: "sha256d:sha256d:1", SHARE_SECONDS: "45", MAINT_TOKEN: TOKEN, MAINT_MEMO_MS: "0", ...extra };
  env.POOL = new sim.Namespace(Pool, env);
  return env;
}
const call = (env, path, init, e = edge) => e.fetch(new Request(`https://pool.test${path}`, init), env);
const declare = (env, body, headers = auth, e = edge) => call(env, "/maintenance", { method: "POST", headers, body: JSON.stringify(body) }, e);
async function miner(env, who, path = "/mine", e = edge) {
  const res = await call(env, `${path}?w=${who}`, { headers: { Upgrade: "websocket" } }, e);
  if (res.status !== 101) return { res, body: await res.json().catch(() => null) };
  const c = res.webSocket;
  c.send(JSON.stringify({ id: 1, method: "glados.hello", params: { v: 1, worker: who, agent: "maint.test" } }) + "\n");
  await sim.sleep(5);
  return { c, res };
}
const jobOf = (c) => c.lines().filter((m) => m.method === "glados.job").map((m) => m.params).pop();
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
async function work(env, e = edge) {
  const doc = JSON.parse(await (await call(env, "/ledger.json", undefined, e)).text());
  const m = new Map();
  for (const w of doc.windows || []) for (const s of w.shares) m.set(s.worker, (m.get(s.worker) || 0) + s.work);
  return m;
}

// ================================================================ who may declare it
{
  const env = world({ MAINT_TOKEN: undefined });
  const r = await declare(env, { on: true, reason: "x" });
  ok(r.status === 403 && /MAINT_TOKEN/.test((await r.json()).error), "with no MAINT_TOKEN set, nobody can declare maintenance, and the answer says how to set it up");

  const env2 = world();
  const wrong = await declare(env2, { on: true, reason: "x" }, { authorization: `Bearer ${TOKEN}x` });
  const none = await declare(env2, { on: true, reason: "x" }, {});
  ok(wrong.status === 401 && none.status === 401, "a wrong token or none: 401");
  const st = await (await call(env2, "/maintenance")).json();
  ok(!st.on, "and nothing changed");
  const { c } = await miner(env2, addr(1));
  ok(c && c.readyState === 1, "and miners still connect");
  const junk = await call(env2, "/maintenance", { method: "POST", headers: auth, body: "{not json" });
  ok(junk.status === 400, "a malformed body from the operator is a 400, not a half-applied state");
  const put = await call(env2, "/maintenance", { method: "PUT", headers: auth });
  ok(put.status === 405, "any other method: 405");
  // The internal paths are not reachable from outside the router.
  const sneak = await call(env2, "/maintenance/apply", { method: "POST", body: JSON.stringify({ on: true, reason: "pwned", since: 0 }) });
  const sneak2 = await call(env2, "/maintenance/set", { method: "POST", body: JSON.stringify({ on: true, reason: "pwned", since: 0 }) });
  const still = await (await call(env2, "/maintenance")).json();
  ok(!still.on && sneak.status !== 101 && sneak2.status !== 101, `/maintenance/apply and /maintenance/set from outside change nothing (${sneak.status}, ${sneak2.status})`);
  c.close();
}

// ================================================================ down, gracefully
{
  const env = world({ TREASURY: "live" });
  // The treasury itself is out of scope here: count its ticks, quote nothing.
  const main = env.POOL.instance("main");
  await main.ready;
  let ticks = 0;
  main.obj.treasurer.tick = async () => { ticks++; };
  main.obj.treasurer.earnQuote = async () => null;
  await env.POOL.alarm("main");
  ok(ticks === 1, "before: the treasury ticks with the alarm");

  const A = addr(0xaa), B = addr(0xbb);
  const a = await miner(env, A), b = await miner(env, B);
  const job = jobOf(a.c);
  await submit(a.c, job, mine(job, 2));

  const r = await declare(env, { on: true, reason: "moving the treasury 🔧", minutes: 30, drainSeconds: 0.3 });
  const out = await r.json();
  ok(r.status === 200 && out.maintenance.on && out.shards.length === 1 && out.shards[0].ok && out.shards[0].connections === 2, `declared: every shard reached, and says how many it told (${JSON.stringify(out.shards)})`);
  const note = a.c.lines().find((m) => m.method === "glados.maintenance");
  ok(note && note.params.reason === "moving the treasury 🔧" && note.params.retryAfter > 1700 && note.params.drainMs === 300, "a connected miner is told why, for how long, and how long it has left");

  // During the drain, a share in hand is still credited.
  const late = await submit(a.c, job, mine(job, 1, mine(job, 2).pop() + 1));
  ok(late[0] && late[0].result, "a share submitted during the drain is accepted");
  ok(a.c.readyState === 1, "and the miner is still connected");

  // A miner arriving now is refused by the router, without waking a shard.
  const callsBefore = env.POOL.calls.filter((n) => n === "main").length;
  const c = await miner(env, addr(0xcc));
  ok(c.res.status === 503 && c.body.error === "maintenance" && /moving the treasury/.test(c.body.why) && Number(c.res.headers.get("retry-after")) > 1700, `a new miner: 503, the reason, and Retry-After ${c.res.headers.get("retry-after")}`);
  ok(env.POOL.calls.filter((n) => n === "main").length === callsBefore, "answered by the router; the mining shard was not called");
  const gpuR = await call(env, "/mine/gpu", { headers: { Upgrade: "websocket" } });
  ok(gpuR.status === 503 || gpuR.status === 404, "every mining path is refused, not only /mine");

  await sim.sleep(400);
  await env.POOL.settle("main");
  ok(a.c.closed && a.c.closed.code === 1013 && /^maintenance: moving the treasury/.test(a.c.closed.reason) && b.c.closed && b.c.closed.code === 1013, `after the drain every miner is closed with 1013 and the reason (${a.c.closed && a.c.closed.code} "${a.c.closed && a.c.closed.reason}")`);
  const s = await (await call(env, "/status?d")).json();
  ok(s.connections === 0 && s.maintenance && s.maintenance.on, "/status: nobody connected, and it says maintenance");
  ok((await call(env, "/payouts.json")).status === 200, "reads keep working: /payouts.json");

  env.POOL.evict("main");
  const w = await work(env);
  ok(w.get(A) === 768, `the record was saved on the way down, the drain's share included (${w.get(A)})`);

  // Payouts stop.
  const m2 = env.POOL.instance("main");
  await m2.ready;
  let ticks2 = 0;
  m2.obj.treasurer.tick = async () => { ticks2++; };
  m2.obj.treasurer.earnQuote = async () => null;
  await env.POOL.alarm("main");
  ok(ticks2 === 0 && logs.some((l) => /treasury\] paused for maintenance/.test(l)), "while down, the alarm takes no treasury step, and says so");
  ok(env.POOL.storage("main").alarm !== null, "but the alarm keeps coming, so payouts resume on time");

  // An evicted shard comes back still down, even with control unreachable (the router lets it through).
  env.POOL.failCalls("control", () => new Error("control unreachable"), 1);
  const d = await miner(env, addr(0xdd));
  ok(d.res.status === 503 && d.body.error === "maintenance", "control unreachable at the router: the shard's own copy still refuses");

  // Declared again: a new reason, the same 'since'.
  const first = out.maintenance.since;
  await sim.sleep(5);
  const again = await (await declare(env, { on: true, reason: "still moving it", minutes: 10, drainSeconds: 0 })).json();
  ok(again.maintenance.since === first && again.maintenance.reason === "still moving it", "declared again while down: the reason changes, 'down since' does not");

  // Up.
  const up = await declare(env, { on: false });
  const upOut = await up.json();
  ok(up.status === 200 && !upOut.maintenance.on && upOut.shards.every((x) => x.ok), "declared up: every shard told");
  const e = await miner(env, A);
  ok(e.c && e.c.readyState === 1 && jobOf(e.c), "miners connect and get work at once");
  const m3 = env.POOL.instance("main");
  let ticks3 = 0;
  m3.obj.treasurer.tick = async () => { ticks3++; };
  m3.obj.treasurer.earnQuote = async () => null;
  await env.POOL.alarm("main");
  ok(ticks3 === 1, "and the treasury ticks again");
  e.c.close();
}

// ================================================================ awkward cases
{
  // "Off" during the drain: nobody is closed.
  const env = world();
  const a = await miner(env, addr(0x11));
  await declare(env, { on: true, reason: "a mistake", drainSeconds: 0.3 });
  await declare(env, { on: false });
  await sim.sleep(400);
  await env.POOL.settle("main");
  ok(a.c.readyState === 1, "declared up again during the drain: nobody is closed");

  // Declared twice quickly: one drain closes everyone once, at the later one's time.
  await declare(env, { on: true, reason: "one", drainSeconds: 0.2 });
  await declare(env, { on: true, reason: "two", drainSeconds: 0.5 });
  await sim.sleep(300);
  ok(a.c.readyState === 1, "declared twice: the first drain's timer does not cut the second's short");
  await sim.sleep(350);
  await env.POOL.settle("main");
  ok(a.c.closed && a.c.closed.code === 1013 && /two/.test(a.c.closed.reason), "and the second closes it, with its own reason");
  await declare(env, { on: false });
}
{
  // A shard that misses the "off" does not refuse miners for ever.
  const env = world();
  await declare(env, { on: true, reason: "x", drainSeconds: 0 });
  env.POOL.failCalls("main", () => new Error("unreachable"), 1);
  const r = await (await declare(env, { on: false })).json();
  ok(r.shards[0].ok === false, `a shard that cannot be told is named in the answer (${r.shards[0].why})`);
  const first = await miner(env, addr(0x22));
  ok(first.res.status === 503, "for up to half a minute it still refuses on its own copy");
  const o = env.POOL.instance("main");
  o.obj.maintAt = 0; // half a minute later
  const later = await miner(env, addr(0x22));
  ok(later.c && later.c.readyState === 1 && logs.some((l) => /control says the pool is up/.test(l)), "then it asks control, learns the pool is up, and takes the miner");
  later.c.close();
}
{
  // The state object down when declaring: nothing is half-applied. (A router
  // of its own: the allowance error trips this isolate's break for 30 s.)
  const e = (await import(`./worker.js?isolated=${Math.random()}`)).default;
  const env = world();
  const a = await miner(env, addr(0x33), "/mine", e);
  env.POOL.failCalls("control", () => new Error("Exceeded allowed duration in Durable Objects free tier."), 1);
  const r = await declare(env, { on: true, reason: "x", drainSeconds: 0 }, auth, e);
  ok(r.status === 503, `the state cannot be stored: the declaration fails as a whole (${r.status})`);
  await sim.sleep(20);
  ok(a.c.readyState === 1 && !a.c.lines().some((m) => m.method === "glados.maintenance"), "and no miner was told or closed");
}

// ================================================================ Pearl
{
  const fx = JSON.parse(readFileSync(new URL("./pearl-fixtures.json", import.meta.url), "utf8")).find((t) => t.name === "square-16x16-salted");
  const EASY = "ff".repeat(32);
  const up = { up: true, submits: [], sockets: [] };
  globalThis.__edgeConnect = () => {
    if (!up.up) throw new Error("proxy request failed, cannot connect to the specified address");
    let ctrl, buf = "";
    const readable = new ReadableStream({ start(c) { ctrl = c; } });
    const say = (o) => { try { ctrl.enqueue(new TextEncoder().encode(JSON.stringify(o) + "\n")); } catch {} };
    const writable = new WritableStream({
      write(chunk) {
        buf += new TextDecoder().decode(chunk);
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const m = JSON.parse(buf.slice(0, nl)); buf = buf.slice(nl + 1);
          if (m.method === "mining.authorize") { say({ id: m.id, result: true, error: null }); say({ id: null, method: "mining.notify", params: { job_id: "pj1", header: fx.header.bytes, target: EASY, height: 1 } }); }
          else if (m.method === "mining.submit") { up.submits.push(m); say({ id: m.id, result: true, error: null }); }
        }
      },
    });
    const s = { readable, writable, close() { try { ctrl.close(); } catch {} }, kill() { try { ctrl.error(new Error("Network connection lost.")); } catch {} } };
    up.sockets.push(s);
    return s;
  };
  const env = world({ PEARL_UPSTREAM: "fake:1", TREASURY_PRL: "prl1ptest", PEARL_MIN_BITS: "0", PEARL_START_BITS: "0" });
  env.PEARL_POOL = new sim.Namespace(PearlPool, env);
  const P = addr(0x77);
  const res = await call(env, "/mine/pearl", { headers: { Upgrade: "websocket" } });
  const c = res.webSocket;
  c.send(JSON.stringify({ id: 1, method: "glados.hello", params: { worker: P, agent: "maint.test" } }) + "\n");
  let job = null;
  for (let i = 0; i < 100 && !job; i++) { await sim.sleep(10); job = (c.lines().filter((m) => m.method === "glados.pearl.job").pop() || {}).params; }
  ok(job && job.job === "pj1", "Pearl: a miner has upstream's job");

  // Upstream goes away; a share that meets its bound has to wait.
  up.up = false;
  up.sockets.at(-1).kill();
  await sim.sleep(20);
  c.send(JSON.stringify({ id: 9, method: "glados.pearl.submit", params: { job: job.job, proof: fx.plain_proof_b64 } }) + "\n");
  await sim.sleep(50);
  const v = c.lines().find((m) => m.id === 9);
  ok(v && v.result && v.result.verdict === "accepted", "Pearl: the share is credited");
  const r = await (await declare(env, { on: true, reason: "pearl too", drainSeconds: 0.1 })).json();
  const pr = r.shards.find((s) => s.name === "pearl-0");
  ok(pr && pr.ok && pr.connections === 1 && pr.owedUpstream === 1, `Pearl: told, and reports the share it still owes upstream (${JSON.stringify(pr)})`);
  ok(c.lines().some((m) => m.method === "glados.maintenance"), "Pearl: the miner is told");
  await sim.sleep(200);
  ok(c.closed && c.closed.code === 1013, "Pearl: and closed with 1013 after the drain");
  const refused = await call(env, "/mine/pearl", { headers: { Upgrade: "websocket" } });
  ok(refused.status === 503, "Pearl: new miners refused");
  up.up = true;
  for (let i = 0; i < 80 && up.submits.length === 0; i++) await sim.sleep(100);
  ok(up.submits.length === 1 && up.submits[0].params.plain_proof === fx.plain_proof_b64, `Pearl: with every miner gone, the owed share still reaches upstream once it is back (${up.submits.length})`);
  const ps = await (await env.PEARL_POOL.get(env.PEARL_POOL.idFromName("pearl-0")).fetch(new Request("https://p/status"))).json();
  ok(ps.pearl.waiting === 0 && ps.pearl.lost === 0 && ps.maintenance && ps.maintenance.on, `Pearl: nothing waiting, nothing lost, still in maintenance (${JSON.stringify({ waiting: ps.pearl.waiting, lost: ps.pearl.lost })})`);
  await declare(env, { on: false });
  const back = await call(env, "/mine/pearl", { headers: { Upgrade: "websocket" } });
  ok(back.status === 101, "Pearl: up again, miners connect");
  back.webSocket.close();
  delete globalThis.__edgeConnect;
}

console.info(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

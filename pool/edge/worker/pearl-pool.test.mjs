// node pool/edge/worker/pearl-pool.test.mjs
//
// pearl-pool.js -- the Pearl shard's object, around the rules in pearl-room.js
// -- under edge-sim.mjs with an upstream that misbehaves: it vanishes, sends a
// runaway line, answers garbage, never answers a submit, or comes back. And
// clients that lie about their hashrate or send lines past the cap. What is
// checked is what upstream received and what the shard says it owes.
import { readFileSync } from "node:fs";
import * as sim from "./edge-sim.mjs";

const { PearlPool } = await import("./pearl-pool.js");

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.info(`${c ? "ok  " : "FAIL"}  ${w}`); };
const logs = [];
console.log = (...a) => logs.push(a.join(" "));
console.error = (...a) => logs.push(a.join(" "));

const fx = JSON.parse(readFileSync(new URL("./pearl-fixtures.json", import.meta.url), "utf8"));
const v = fx.find((t) => t.name === "square-16x16-salted");
const later = fx.find((t) => t.name === "later-tile-128x128-salted");
const EASY = "ff".repeat(32);

// An upstream whose behaviour the test sets: "ok", "silent" (takes submits, answers none), "junk".
function upstream() {
  const up = { up: true, mode: "ok", submits: [], sockets: [], header: v.header.bytes };
  globalThis.__edgeConnect = () => {
    if (!up.up) throw new Error("proxy request failed, cannot connect to the specified address");
    let ctrl, buf = "";
    const readable = new ReadableStream({ start(c) { ctrl = c; } });
    const raw = (s) => { try { ctrl.enqueue(new TextEncoder().encode(s)); } catch {} };
    const say = (o) => raw(JSON.stringify(o) + "\n");
    const writable = new WritableStream({
      write(chunk) {
        buf += new TextDecoder().decode(chunk);
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const m = JSON.parse(buf.slice(0, nl)); buf = buf.slice(nl + 1);
          if (m.method === "mining.authorize") {
            if (up.mode === "junk") { raw("}{\nnull\n[1,2]\n" + JSON.stringify({ method: "mining.notify", params: { job_id: 5, header: "zz" } }) + "\n"); }
            say({ id: m.id, result: true, error: null });
            say({ id: null, method: "mining.notify", params: { job_id: "u1", header: up.header, target: EASY, height: 7 } });
          } else if (m.method === "mining.submit") {
            up.submits.push(m);
            if (up.mode !== "silent") say({ id: m.id, result: true, error: null });
          }
        }
      },
    });
    const s = { readable, writable, raw, close() { try { ctrl.close(); } catch {} }, kill() { try { ctrl.error(new Error("Network connection lost.")); } catch {} } };
    up.sockets.push(s);
    return s;
  };
  return up;
}

function shard(extra = {}) {
  const env = { PEARL_UPSTREAM: "fake:1", TREASURY_PRL: "prl1ptest", PEARL_MIN_BITS: "0", PEARL_START_BITS: "0", ...extra };
  env.PEARL_POOL = new sim.Namespace(PearlPool, env);
  const stub = env.PEARL_POOL.get(env.PEARL_POOL.idFromName("pearl-0"));
  return { env, stub, status: async () => (await (await stub.fetch(new Request("https://p/status"))).json()) };
}
async function miner(s, who) {
  const res = await s.stub.fetch(new Request("https://p/mine/pearl", { headers: { Upgrade: "websocket" } }));
  const c = res.webSocket;
  c.send(JSON.stringify({ id: 1, method: "glados.hello", params: { worker: who, agent: "pearl-pool.test" } }) + "\n");
  let job = null;
  for (let i = 0; i < 100 && !job; i++) { await sim.sleep(10); job = (c.lines().filter((m) => m.method === "glados.pearl.job").pop() || {}).params; }
  return { c, job };
}
let rid = 50;
async function submit(c, job, proof) {
  const id = rid++;
  c.send(JSON.stringify({ id, method: "glados.pearl.submit", params: { job, proof } }) + "\n");
  for (let i = 0; i < 50; i++) { const r = c.lines().find((m) => m.id === id); if (r) return r.result; await sim.sleep(10); }
  return null;
}
const A = "0x" + "a1".repeat(20), B = "0x" + "b2".repeat(20);

// ================================================================ an upstream that answers nothing
{
  const up = upstream();
  up.mode = "silent";
  const s = shard();
  const { c, job } = await miner(s, A);
  ok(job && job.job === "u1", "a miner gets upstream's job");
  const r = await submit(c, job.job, v.plain_proof_b64);
  await sim.sleep(20);
  const st = await s.status();
  ok(r && r.verdict === "accepted" && up.submits.length === 1 && st.pearl.inFlight === 1, `upstream takes the share and never answers: it is held as in flight (${st.pearl.inFlight})`);
  // The connection dies; the unanswered share goes out again on the next one.
  up.mode = "ok";
  up.sockets.at(-1).kill();
  for (let i = 0; i < 60 && up.submits.length < 2; i++) await sim.sleep(100);
  const st2 = await s.status();
  ok(up.submits.length === 2 && up.submits[1].params.plain_proof === up.submits[0].params.plain_proof && st2.pearl.inFlight === 0 && st2.pearl.answered === 1, `the connection drops: the unanswered share is sent again on the next one, and answered (${up.submits.length} sends, answered ${st2.pearl.answered})`);
  ok(st2.pearl.sent === 1, `and it is counted sent once, not twice (${st2.pearl.sent})`);
  c.close();
}

// ================================================================ a runaway line, and junk
{
  const up = upstream();
  const s = shard();
  const { c, job } = await miner(s, A);
  const before = up.sockets.length;
  up.sockets.at(-1).raw("x".repeat((1 << 20) + 10)); // no newline, past the 1 MiB the shard will hold
  for (let i = 0; i < 60 && up.sockets.length === before; i++) await sim.sleep(100);
  ok(logs.some((l) => /runaway line from upstream/.test(l)) && up.sockets.length > before, "upstream sends a megabyte with no newline: the shard drops it and reconnects");
  const r = await submit(c, job.job, v.plain_proof_b64);
  ok(r && r.verdict === "accepted", "and mining goes on");
  c.close();

  const up2 = upstream();
  up2.mode = "junk";
  const s2 = shard();
  const m2 = await miner(s2, B);
  ok(m2.job && m2.job.job === "u1", "upstream's junk before a good job: ignored, the good job delivered");
  m2.c.close();
}

// ================================================================ the queue's limits
{
  const up = upstream();
  const s = shard();
  const { c, job } = await miner(s, A);
  up.up = false;
  up.sockets.at(-1).kill();
  await sim.sleep(30);
  // More than the queue holds: one proof, re-sent by many wallets is a
  // duplicate, so reach the cap by feeding the object directly.
  const o = s.env.PEARL_POOL.live.get("pearl-0").obj;
  for (let i = 0; i < 70; i++) o.upSend(JSON.stringify({ id: 1000 + i, method: "mining.submit", params: {} }) + "\n");
  const st = await s.status();
  ok(st.pearl.waiting === 64 && st.pearl.lost === 6, `70 shares owed while upstream is down: 64 held, 6 counted lost, oldest first (waiting ${st.pearl.waiting}, lost ${st.pearl.lost})`);
  // Too old to send once upstream is back.
  for (const q of o.upQueue) q.at -= 61_000;
  const mark = logs.length;
  up.up = true;
  for (let i = 0; i < 300 && !logs.slice(mark).some((l) => /[1-9]\d* too old/.test(l)); i++) await sim.sleep(100);
  const st2 = await s.status();
  ok(st2.pearl.waiting === 0 && st2.pearl.lost === 70 && up.submits.length === 0, `shares past a minute old are not sent to an upstream that has moved on (lost ${st2.pearl.lost}, sent ${up.submits.length})`);
  void job;
  c.close();
}

// ================================================================ clients
{
  upstream();
  const s = shard();
  const { c } = await miner(s, A);
  c.send(JSON.stringify({ method: "glados.rate", params: { hs: 1e300 } }) + "\n");
  c.send(JSON.stringify({ method: "glados.rate", params: { hs: -5 } }) + "\n");
  await sim.sleep(10);
  const st = await s.status();
  ok(st.pearl.hashrate === 0, `a negative rate is zero (${st.pearl.hashrate})`);
  c.send(JSON.stringify({ method: "glados.rate", params: { hs: 1e300 } }) + "\n");
  await sim.sleep(10);
  ok((await s.status()).pearl.hashrate === 1e15, "an absurd rate is capped at 1e15 MAC/s");
  c.send("x".repeat(262_145));
  await sim.sleep(10);
  ok(c.closed && c.closed.code === 1009, `a line past 256 KiB closes the socket with 1009 (${c.closed && c.closed.code})`);
  ok((await s.status()).connections === 0, "and it is no longer counted");

  // Work survives the object.
  const m = await miner(s, B);
  await submit(m.c, m.job.job, later.plain_proof_b64);
  await submit(m.c, m.job.job, v.plain_proof_b64);
  m.c.close();
  await s.env.PEARL_POOL.settle("pearl-0");
  s.env.PEARL_POOL.evict("pearl-0");
  const w = await (await s.stub.fetch(new Request("https://p/work.json"))).json();
  ok(w.work[B] !== undefined && BigInt(w.work[B]) >= 1n, `work credited before an eviction is in /work.json after it (${w.work[B]})`);
}

delete globalThis.__edgeConnect;
console.info(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

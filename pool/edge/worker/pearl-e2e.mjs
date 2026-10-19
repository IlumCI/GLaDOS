// node pool/edge/worker/pearl-e2e.mjs <pool http base> <fake upstream port>
//
// The Pearl shard end to end, against `wrangler dev` run with
//   --var PEARL_UPSTREAM:127.0.0.1:<port> --var PEARL_MIN_BITS:0 --var PEARL_START_BITS:0
// It stands up a fake 2Miners (authorize, one job on a Pearl-mined proof's
// header, records what is submitted), connects a miner to /mine/pearl, sends
// that proof, and checks the whole path: job delivered, share verified and
// credited, forwarded upstream in 2Miners' format, counted in /status.
import net from "node:net";
import { readFileSync } from "node:fs";

const [base, port] = [process.argv[2] || "http://127.0.0.1:18788", Number(process.argv[3] || 18930)];
const fx = JSON.parse(readFileSync(new URL("./pearl-fixtures.json", import.meta.url), "utf8")).find((t) => t.name === "square-16x16-salted");
let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };

const submits = [];
let authorized = null;
const upstream = net.createServer((sock) => {
  let buf = "";
  sock.on("data", (d) => {
    buf += d;
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const m = JSON.parse(buf.slice(0, nl)); buf = buf.slice(nl + 1);
      if (m.method === "mining.authorize") {
        authorized = m.params;
        sock.write(JSON.stringify({ id: m.id, result: true, error: null }) + "\n");
        sock.write(JSON.stringify({ id: null, method: "mining.notify", params: { job_id: "e2e_1", header: fx.header.bytes, target: "ff".repeat(32), height: 7 } }) + "\n");
      } else if (m.method === "mining.submit") {
        submits.push(m.params);
        sock.write(JSON.stringify({ id: m.id, result: true, error: null }) + "\n");
      }
    }
  });
}).listen(port, "127.0.0.1");

const A = "0x" + "e2".repeat(20);
const ws = new WebSocket(base.replace(/^http/, "ws") + "/mine/pearl");
const inbox = [];
ws.addEventListener("message", (e) => { for (const l of String(e.data).split("\n")) if (l.trim()) inbox.push(JSON.parse(l)); });
const until = async (f, ms = 15000) => { const t = Date.now(); while (Date.now() - t < ms) { const x = f(); if (x) return x; await new Promise((r) => setTimeout(r, 100)); } return null; };

await new Promise((r, j) => { ws.addEventListener("open", r); ws.addEventListener("error", j); });
ws.send(JSON.stringify({ id: 1, method: "glados.hello", params: { v: 1, worker: A + ".e2e", agent: "pearl-e2e", reward: "glados" } }) + "\n");
const greet = await until(() => inbox.find((m) => m.id === 1));
ok(greet && greet.result && greet.result.pearl && greet.result.pearl.k === 2048, "the shard greets with the Pearl shape");
const job = await until(() => inbox.find((m) => m.method === "glados.pearl.job"));
ok(authorized && authorized.wallet && authorized.wallet.startsWith("prl1"), `it authorized upstream with the treasury's PRL wallet (${authorized && authorized.wallet.slice(0, 12)}...)`);
ok(job && job.params.job === "e2e_1" && job.params.header === fx.header.bytes, "the miner received upstream's job");

ws.send(JSON.stringify({ id: 2, method: "glados.pearl.submit", params: { job: "e2e_1", proof: fx.plain_proof_b64 } }) + "\n");
const v = await until(() => inbox.find((m) => m.id === 2));
ok(v && v.result && v.result.ok && v.result.verdict === "accepted", `the share was verified and accepted (${v && JSON.stringify(v.result)})`);
const fwd = await until(() => submits[0]);
ok(fwd && fwd.job_id === "e2e_1" && fwd.jackpot === fx.jackpot_hash && fwd.plain_proof === fx.plain_proof_b64 && fwd.adjustment_factor === 524288, "and forwarded upstream as 2Miners' mining.submit");

ws.send(JSON.stringify({ id: 3, method: "glados.pearl.submit", params: { job: "e2e_1", proof: fx.tampered[0].plain_proof_b64 } }) + "\n");
const bad = await until(() => inbox.find((m) => m.id === 3));
ok(bad && !bad.result.ok && bad.result.verdict === "invalid", `a tampered share is refused (${bad && bad.result.why})`);

ws.send(JSON.stringify({ method: "glados.rate", params: { hs: 1.6e12 } }) + "\n");
await new Promise((r) => setTimeout(r, 1500));
const st = await (await fetch(`${base}/status?_=${Date.now()}`)).json();
ok(st.pearl && st.pearl.accepted >= 1 && st.pearl.forwarded >= 1 && st.pearl.upAccepted >= 1, `/status counts it: ${JSON.stringify(st.pearl)}`);
ok(st.pearl && st.pearl.hashrate >= 1.6e12, "and shows the Pearl rate apart from the hash rate");

ws.close();
upstream.close();
console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

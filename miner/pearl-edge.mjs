// node miner/pearl-edge.mjs <glados-miner> <pearl helper> <research/pearl dir>
//
// The native miner's Pearl loop against a hostile pool, with the real CUDA
// helper: a fake Pearl shard on localhost that, connection by connection,
// sends jobs it should refuse, a job no tile can meet, a job every tile meets,
// a storm of job switches, an overlong line, non-object JSON, an abrupt close,
// and -- from outside -- kills the helper. Every proof the miner submits is
// checked against the header of the job it names.
import net from "node:net";
import { spawn, execSync } from "node:child_process";

const [miner, helper, dir] = process.argv.slice(2);
const { verifyShare, fromBase64 } = await import(`${dir}/pearlproof.mjs`);
const MAX = (1n << 256n) - 1n;

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const hexOf = (seed) => Array.from({ length: 76 }, (_, i) => ((i * 31 + seed * 17 + 3) & 255).toString(16).padStart(2, "0")).join("");

const jobs = new Map(); // id -> header hex, every job ever sent
const conns = [];       // one record per connection
let current = null;
const submits = [];     // { conn, id, job, proof, at }

const server = net.createServer((sock) => {
  const rec = { sock, hello: null, lines: 0, closedByMiner: false, opened: Date.now() };
  conns.push(rec);
  current = rec;
  let buf = "";
  sock.on("data", (d) => {
    buf += d;
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
      rec.lines++;
      let m; try { m = JSON.parse(line); } catch { rec.unparsable = (rec.unparsable || 0) + 1; continue; }
      if (m.method === "glados.hello") { rec.hello = m; sock.write(JSON.stringify({ id: m.id, result: { ok: true }, error: null }) + "\n"); }
      else if (m.method === "glados.pearl.submit") {
        submits.push({ conn: conns.indexOf(rec), id: m.id, job: m.params.job, proof: m.params.proof, at: Date.now() });
        sock.write(JSON.stringify({ id: m.id, result: { ok: true, verdict: "accepted" } }) + "\n");
      }
    }
  });
  sock.on("close", () => { rec.closed = Date.now(); });
  sock.on("error", () => {});
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;

const send = (obj) => current.sock.write((typeof obj === "string" ? obj : JSON.stringify(obj)) + "\n");
const job = (id, bits, seed = 1) => { jobs.set(id, hexOf(seed)); send({ id: null, method: "glados.pearl.job", params: { job: id, header: hexOf(seed), bits, k: 2048, rank: 128 } }); };
const waitFor = async (pred, ms) => { for (let t = 0; t < ms; t += 50) { if (pred()) return true; await sleep(50); } return pred(); };

let out = "";
const proc = spawn(miner, ["--pool", `127.0.0.1:${port}`, "--worker", "0x" + "ab".repeat(20), "--algo", "pearl", "--pearl", helper], { stdio: ["ignore", "pipe", "pipe"] });
proc.stdout.on("data", (d) => { out += d; });
proc.stderr.on("data", (d) => { out += d; });
const helperPid = () => { try { return Number(execSync(`pgrep -P ${proc.pid}`).toString().trim().split("\n")[0]); } catch { return 0; } };
const rssKb = () => { try { return Number(execSync(`ps -o rss= -p ${proc.pid}`).toString().trim()); } catch { return 0; } };

// ---------------------------------------------------------- connection 1
ok(await waitFor(() => current && current.hello, 30000), "the miner connects and says hello");
const hostile = [
  ["a b", 10], ["x\nquit", 10], ['q"uote', 10], ["back\\slash", 10], ["", 10], ["z".repeat(129), 10], ["ünï", 10],
];
for (const [id, bits] of hostile) job(id, bits);
send({ id: null, method: "glados.pearl.job", params: { job: "shorthdr", header: "ab".repeat(75), bits: 10 } });
send({ id: null, method: "glados.pearl.job", params: { job: "nonhex", header: "zz".repeat(76), bits: 10 } });
for (const b of [-1, 256, 1e9, "12", null, 3.5, "1e2"]) send({ id: null, method: "glados.pearl.job", params: { job: `bits${b}`, header: hexOf(1), bits: b } });
send(`{"id":null,"method":"glados.pearl.job","params":{"job":"bitsexp","header":"${hexOf(1)}","bits":1e1}}`);
await sleep(1500);
const tally = () => JSON.stringify(Object.fromEntries([...new Set(submits.map((s) => s.job))].map((j) => [j, submits.filter((s) => s.job === j).length])));
console.log(`      submits by job: ${tally()}`);
ok(proc.exitCode === null && submits.length === 0, "jobs it must refuse (spaces, newline, quote, backslash, empty, 129 chars, non-ASCII, bad header, bits out of range): no crash, nothing mined");
ok(/job id this miner will not pass on/.test(out) && /job this miner cannot read/.test(out), "and it says why");

job("none", 255, 2);
await sleep(2500);
ok(submits.length === 0 && proc.exitCode === null, "bits 255: rounds run, no share, no crash");

job("every", 0, 3);
ok(await waitFor(() => submits.filter((s) => s.job === "every").length >= 8, 20000), `bits 0: every tile a share, at most 4 a round (${submits.filter((s) => s.job === "every").length} so far)`);
const r0 = submits.filter((s) => s.job === "every").slice(0, 8).map((s) => verifyShare(fromBase64(s.proof), Uint8Array.from(Buffer.from(jobs.get("every"), "hex")), { bound: MAX }));
ok(r0.length === 8 && r0.every((r) => r.ok), `and every proof verifies against its job's header (${r0.filter((r) => r.ok).length}/8)`);

// A storm of job switches: every submit must name a job that was sent, and verify against that job's header.
const before = submits.length;
for (let i = 0; i < 40; i++) { job(`storm${i}`, 0, 100 + i); await sleep(25); }
await sleep(3000);
const storm = submits.slice(before);
const bad = storm.filter((s) => !jobs.has(s.job) || !verifyShare(fromBase64(s.proof), Uint8Array.from(Buffer.from(jobs.get(s.job), "hex")), { bound: MAX }).ok);
ok(storm.length > 0 && bad.length === 0, `40 job switches in a second: ${storm.length} shares, each verifies against the header of the job it names (${bad.length} do not)`);
const lastJob = storm.length ? storm[storm.length - 1].job : null;
ok(lastJob === "storm39", `and it settles on the last job (${lastJob})`);

// An overlong line: the session must end, not wedge.
const rss0 = rssKb();
current.sock.write("x".repeat(3 << 20));
ok(await waitFor(() => conns.length >= 2 && conns[1].hello, 20000), "a 3 MB line with no newline: the miner drops the session and reconnects");
ok(rssKb() < rss0 + 50_000, `without holding the junk (rss ${rss0} -> ${rssKb()} kB)`);

// ---------------------------------------------------------- connection 2
for (const junk of ["null", "7", "[]", '"s"', "{", '{"method":"glados.pearl.job"}', '{"method":"glados.pearl.job","params":7}', '{"id":"x","error":5}', '{"id":99,"result":null}', '{"id":-5,"result":{"ok":true}}']) send(junk);
job("after-junk", 0, 4);
const c2 = submits.length;
ok(await waitFor(() => submits.slice(c2).some((s) => s.job === "after-junk"), 20000), "non-object and malformed JSON is ignored; the next good job is mined");
ok(conns.length === 2, `without a reconnect (${conns.length} connections; ${tally()})`);

// An abrupt close mid-round: the miner comes back on its own.
current.sock.destroy();
ok(await waitFor(() => conns.length >= 3 && conns[2].hello, 20000), "the pool closes the socket mid-round: the miner reconnects");
ok(!/helper restarted/.test(out), "keeping the same helper");

// ---------------------------------------------------------- connection 3
job("pre-kill", 0, 5);
await waitFor(() => submits.some((s) => s.job === "pre-kill"), 20000);
const hp = helperPid();
ok(hp > 0, `the helper runs as a child (pid ${hp})`);
process.kill(hp, "SIGKILL");
ok(await waitFor(() => /helper restarted/.test(out), 30000), "the helper is killed: the miner starts another");
ok(await waitFor(() => conns.length >= 4 && conns[3].hello, 20000), "and reconnects");
job("post-kill", 0, 6);
ok(await waitFor(() => submits.some((s) => s.job === "post-kill"), 30000), "and mines with the new helper");

const unparsable = conns.reduce((a, c) => a + (c.unparsable || 0), 0);
ok(unparsable === 0, `every line the miner sent was JSON (${unparsable} were not)`);

proc.kill("SIGTERM");
await sleep(500);
const orphan = hp && (() => { try { process.kill(helperPid() || -1, 0); return true; } catch { return false; } })();
server.close();
console.log(`\n${passed} passed, ${failed} failed`);
if (failed && process.env.MINER_LOG) (await import("node:fs")).writeFileSync(process.env.MINER_LOG, out);
process.exit(failed ? 1 : 0);

// node pool/edge/worker/pearl-room.test.mjs [vectors.json] [upstream-job.json]
//
// The Pearl shard's rules against proofs Pearl's own miner produced
// (research/pearl/vectors.json on the pearl-research branch) and a job 2Miners
// really sent (research/pearl/upstream-2miners-job.json).
import { readFileSync } from "node:fs";
import { PearlRoom, K, RANK } from "./pearl-room.js";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };
const here = (f) => new URL(f, import.meta.url);
const vectors = JSON.parse(readFileSync(process.argv[2] || here("./pearl-fixtures.json"), "utf8"));
const upJob = JSON.parse(readFileSync(process.argv[3] || here("./pearl-upstream-job.json"), "utf8")).find((m) => m.method === "mining.notify");
const v = vectors.find((t) => t.name === "square-16x16-salted");
const later = vectors.find((t) => t.name === "later-tile-128x128-salted");
const A = "0x" + "ab".repeat(20);
const one = (out) => JSON.parse(out.send[0]); // the verdict; a retargeted job may follow it
const notify = (job_id, header, target) => JSON.stringify({ id: null, method: "mining.notify", params: { job_id, header, target, height: 1 } });
const EASY = "ff".repeat(32), HARD = "00".repeat(31) + "01";

let t = 1000;
const room = new PearlRoom({ wallet: "prl1ptest", now: () => t });
ok(JSON.parse(room.authorize()).params.wallet === "prl1ptest", "authorizes upstream with {wallet}, as 2Miners wants");
ok(room.upLine(JSON.stringify({ id: 1, result: true, error: null })).log[0].includes("authorized"), "reads upstream's authorize answer");

// 2Miners' real job: readable, and our share difficulty never exceeds theirs.
ok(room.upLine(notify(upJob.params.job_id, upJob.params.header, upJob.params.target)).newJob, "takes 2Miners' real job");
const suffix = Number(upJob.params.job_id.split("_")[1]);
ok(2 ** room.maxBits() <= suffix && 2 ** (room.maxBits() + 1) > suffix / 2, `max share bits ${room.maxBits()} sits at upstream's difficulty (job id says 2^${Math.log2(suffix)})`);

// A miner, on the vector's own header so its proof verifies.
room.upLine(notify("j1", v.header.bytes, EASY));
room.open(7);
ok(room.line(7, JSON.stringify({ id: 1, method: "glados.hello", params: { worker: "nobody" } })).send[0].includes("payout address"), "refuses a worker that is not an address");
const hello = room.line(7, JSON.stringify({ id: 1, method: "glados.hello", params: { v: 1, worker: A + ".rig", agent: "test" } }));
const job = JSON.parse(hello.send[1]).params;
ok(JSON.parse(hello.send[0]).result.pearl.k === K && job.job === "j1" && job.header === v.header.bytes && job.k === K && job.rank === RANK, "greets with the shape, then sends the current job");

const submit = (proof, jobId = "j1", id = 9) => room.line(7, JSON.stringify({ id, method: "glados.pearl.submit", params: { job: jobId, proof } }));
room.conns.get(7).bits = 0; // accept any hash, so a fixed vector counts as a share
let r = submit(v.plain_proof_b64);
ok(one(r).result.ok && one(r).result.verdict === "accepted" && r.credited, "a valid proof is accepted and credited");
ok(room.work.get(A) === 1n, "credited 2^bits tiles to the address, not the rig");
ok(r.up.length === 1, "and, meeting upstream's bound, forwarded");
const up = JSON.parse(r.up[0]);
ok(up.method === "mining.submit" && up.params.job_id === "j1" && up.params.plain_proof === v.plain_proof_b64 && up.params.jackpot === v.jackpot_hash && up.params.adjustment_factor === 524288, "forwarded as mining.submit {job_id, plain_proof, jackpot, adjustment_factor}");

room.conns.get(7).bits = 0;
ok(one(submit(v.plain_proof_b64)).result.verdict === "duplicate", "the same tile again is a duplicate");
ok(one(submit(v.plain_proof_b64, "nope")).result.verdict === "stale", "a job the pool does not hold is stale");
ok(one(submit(v.tampered[0].plain_proof_b64)).result.verdict === "invalid", `a tampered proof is invalid (${v.tampered[0].what})`);
ok(one(submit("!!!")).result.verdict === "invalid", "garbage is invalid");
ok(room.work.get(A) === 1n, "and none of those credited anything");

// Low: a hash above this miner's share bound.
room.upLine(notify("j2", later.header.bytes, EASY));
room.conns.get(7).bits = 64;
r = submit(later.plain_proof_b64, "j2");
ok(one(r).result.verdict === "low" && !r.credited, "a hash above the miner's bound is low, not credited");

// Not forwarded when upstream's bound is tighter than the share.
room.upLine(notify("j3", later.header.bytes, HARD));
room.conns.get(7).bits = 0;
r = submit(later.plain_proof_b64, "j3");
ok(r.credited && r.up.length === 0, "credited but not forwarded when it misses upstream's bound");

// The rect vector is another shape: refused.
room.upLine(notify("j4", vectors.find((x) => x.name === "rect-8x16-salted").header.bytes, EASY));
room.conns.get(7).bits = 0;
ok(one(submit(vectors.find((x) => x.name === "rect-8x16-salted").plain_proof_b64, "j4")).result.verdict === "invalid", "a tile shape this pool does not mine is invalid");

// Upstream's verdicts are counted.
room.upLine(JSON.stringify({ id: up.id, result: true, error: null }));
ok(room.stats.upAccepted === 1, "upstream's acceptance is counted");

// VarDiff: shares a second apart push bits up; long silence brings them down.
const vd = new PearlRoom({ wallet: "w", now: () => t });
vd.upLine(notify(upJob.params.job_id, upJob.params.header, upJob.params.target));
vd.open(1);
vd.line(1, JSON.stringify({ id: 1, method: "glados.hello", params: { worker: A } }));
const b0 = vd.conns.get(1).bits;
// Drive the interval logic directly: a verified share is not needed to test the clock.
const c = vd.conns.get(1);
for (let i = 0; i < 5; i++) { t += 1000; c.ema = 0.7 * c.ema + 0.3 * 1; }
t += 200 * 1000;
const easier = vd.idle(1);
ok(easier && vd.conns.get(1).bits === b0 - 1, "a miner silent for a minute gets an easier job");

// ------------------------------------------------------------ hostile input
{
  let clock = 50_000;
  const h = new PearlRoom({ wallet: "w", now: () => clock, minBits: 0, startBits: 0 });
  const hello = (id, addr) => h.line(id, JSON.stringify({ id: 1, method: "glados.hello", params: { worker: addr } }));
  const sub = (id, job, proof, rid = 5) => h.line(id, JSON.stringify({ id: rid, method: "glados.pearl.submit", params: { job, proof } }));
  const first = (o) => JSON.parse(o.send[0]);
  const B = "0x" + "bb".repeat(20), C = "0x" + "cc".repeat(20);
  h.upLine(notify("h1", v.header.bytes, EASY));

  // Garbage of every shape: answered or ignored, never a throw.
  h.open(1); hello(1, B);
  let threw = 0;
  for (const junk of ["", "{", "null", "[]", "42", '"x"', '{"method":"glados.pearl.submit"}', '{"method":"glados.pearl.submit","params":[]}',
    '{"method":"glados.pearl.submit","params":{"job":{},"proof":5}}', '{"method":"glados.pearl.submit","params":{"job":"h1","proof":""}}',
    '{"method":"glados.pearl.submit","params":{"job":"h1","proof":"' + "A".repeat(300000) + '"}}', '{"id":1,"method":"glados.hello","params":{"worker":7}}']) {
    try { h.line(1, junk); } catch { threw++; }
  }
  ok(threw === 0, "no malformed miner line makes the room throw");
  threw = 0;
  for (const junk of ["", "}{", "null", "[]", "7", JSON.stringify({ method: "mining.notify", params: { job_id: 5, header: "zz", target: "00" } }),
    JSON.stringify({ method: "mining.notify", params: null }), JSON.stringify({ id: 999, result: "x" }), "x".repeat(100000)]) {
    try { h.upLine(junk); } catch { threw++; }
  }
  ok(threw === 0 && h.jobs[0].id === "h1", "no malformed upstream line throws or replaces the job");

  // A second greeting cannot move whom the connection pays.
  const moved = hello(1, C);
  ok(/another address/.test(moved.send[0]) && h.conns.get(1).addr === B, "a later hello with another address is refused; the first stands");

  // The same tile from two wallets: only the first is paid.
  h.open(2); hello(2, C);
  clock += 10_000;
  const a1 = sub(1, "h1", v.plain_proof_b64);
  const a2 = sub(2, "h1", v.plain_proof_b64);
  ok(first(a1).result.verdict === "accepted" && first(a2).result.verdict === "duplicate", "the same tile from a second wallet is a duplicate");
  ok(h.work.get(B) === 1n && !h.work.has(C), "and only the first wallet is credited");

  // Upstream re-issues the header under a new job id: still a duplicate.
  h.upLine(notify("h1-again", v.header.bytes, EASY));
  clock += 10_000;
  ok(first(sub(1, "h1-again", v.plain_proof_b64)).result.verdict === "duplicate", "re-submitted under a new job id for the same header: duplicate");

  // Replayed against a job with another header: the roots no longer verify.
  h.upLine(notify("h2", later.header.bytes, EASY));
  clock += 10_000;
  const rp = first(sub(1, "h2", v.plain_proof_b64));
  ok(rp.result.verdict === "invalid" && /root/.test(rp.result.why), `replayed onto another header: invalid (${rp.result.why})`);

  // A legacy-encoded proof (no trailing MoE tag) is accepted, and forwarded in the current format.
  const legacy = Buffer.from(later.plain_proof_b64, "base64").subarray(0, -1).toString("base64");
  clock += 10_000;
  const lg = sub(1, "h2", legacy);
  ok(first(lg).result.verdict === "accepted" && lg.up.length === 1, "a legacy-encoded proof is accepted");
  ok(JSON.parse(lg.up[0]).params.plain_proof === later.plain_proof_b64, "and forwarded re-encoded in the current format");

  // The ration: a burst of ten, then refusals that cost no verification, then refills.
  h.open(3); hello(3, "0x" + "dd".repeat(20));
  const verdicts = [];
  for (let i = 0; i < 14; i++) verdicts.push(first(sub(3, "no-such-job", "AAAA")).result.verdict);
  ok(verdicts.slice(0, 10).every((x) => x === "stale") && verdicts.slice(10).every((x) => x === "rate"), `ten submits a burst, then "rate" (${verdicts.join(",")})`);
  clock += 1000;
  ok(first(sub(3, "no-such-job", "AAAA")).result.verdict === "stale", "a second later, the bucket has refilled a little");

  // Proofs that keep failing verification close the connection.
  h.open(4); hello(4, "0x" + "ee".repeat(20));
  let closed = false, n = 0;
  for (let i = 0; i < 12 && !closed; i++) { clock += 600; n++; closed = sub(4, "h1", v.tampered[0].plain_proof_b64).close; }
  ok(closed && n === 10, `a connection sending only invalid proofs is closed at the tenth (after ${n})`);
  // ...but "low" -- honest after a difficulty rise -- never counts toward that.
  h.open(5); hello(5, "0x" + "ff".repeat(20));
  h.conns.get(5).bits = 255;
  let lowClosed = false;
  for (let i = 0; i < 12; i++) { clock += 600; if (sub(5, "h2", later.plain_proof_b64).close) lowClosed = true; }
  ok(!lowClosed, "twelve 'low' shares do not close a connection");
}

// Save and load.
const back = new PearlRoom({ wallet: "w" });
ok(back.load(room.save()) === 1 && back.work.get(A) === room.work.get(A), "work survives save and load");

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

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

// Save and load.
const back = new PearlRoom({ wallet: "w" });
ok(back.load(room.save()) === 1 && back.work.get(A) === room.work.get(A), "work survives save and load");

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

// node pool/edge/worker/pearl-load.mjs <pool http base> <research/pearl dir> [clients] [proofs]
//
// The Pearl shard under load, against `wrangler dev` with PEARL_MIN_BITS=0 and
// PEARL_START_BITS=0 (every valid proof is a share) and pearl-fake-upstream.mjs.
// Proofs come from the reference miner on the fake upstream's header, so each
// is real and distinct. Every client is its own wallet and submits its slice,
// plus a duplicate of another client's proof and an invalid one, within the
// pool's ration. Then the books: every distinct proof credited once, every
// duplicate and invalid one refused, the shard's counters in agreement.
const [base, dir, CLIENTS = "50", PROOFS = "200"] = process.argv.slice(2);
const { round } = await import(`${dir}/pearlmine.mjs`);
const { toBase64 } = await import(`${dir}/pearlproof.mjs`);
const header = Uint8Array.from({ length: 76 }, (_, i) => (i * 53 + 7) & 255); // pearl-fake-upstream.mjs's
const nC = Number(CLIENTS), nP = Number(PROOFS);

const t0 = performance.now();
const proofs = [];
const SEED = Number(process.env.SEED || 0), PACE = process.env.PACE === undefined ? 550 : Number(process.env.PACE);
for (let s = 1 + SEED; proofs.length < nP; s++) {
  const r = round(header, { m: 64, n: 64, k: 2048, rank: 128, s0: s * 7919, s1: s * 104729 });
  for (let tr = 0; tr < 4 && proofs.length < nP; tr++) for (let tc = 0; tc < 4 && proofs.length < nP; tc++) proofs.push(toBase64(r.proof(tr, tc)));
}
console.log(`${proofs.length} distinct proofs made in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
const bad = (() => { const b = Buffer.from(proofs[0], "base64"); b[5000] ^= 1; return b.toString("base64"); })();

const ws = (base.replace(/^http/, "ws")) + "/mine/pearl";
const before = (await (await fetch(`${base}/status?_=${Date.now()}`)).json()).pearl || { accepted: 0 };
const tally = { accepted: 0, duplicate: 0, invalid: 0, rate: 0, stale: 0, low: 0, other: 0 };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function client(i) {
  const addr = "0x" + (i + 1).toString(16).padStart(40, "0");
  const sock = new WebSocket(ws);
  const inbox = new Map();
  let job = null;
  sock.addEventListener("message", (e) => {
    for (const l of String(e.data).split("\n")) {
      if (!l.trim()) continue;
      const m = JSON.parse(l);
      if (m.method === "glados.pearl.job") job = m.params.job;
      else if (m.id !== undefined && m.id !== null) inbox.set(m.id, m);
    }
  });
  await new Promise((r, j) => { sock.addEventListener("open", r); sock.addEventListener("error", j); });
  sock.send(JSON.stringify({ id: 1, method: "glados.hello", params: { worker: addr, agent: "pearl-load" } }) + "\n");
  for (let w = 0; !job && w < 200; w++) await sleep(50);
  if (!job) throw new Error(`client ${i} got no job`);
  const mine = proofs.filter((_, k) => k % nC === i);
  const work = [...mine, proofs[(i + 1) % nC] /* someone else's */, bad];
  let id = 10;
  for (const proof of work) {
    const rid = id++;
    sock.send(JSON.stringify({ id: rid, method: "glados.pearl.submit", params: { job, proof } }) + "\n");
    if (PACE) { for (let w = 0; !inbox.has(rid) && w < 600; w++) await sleep(25); tallyOf(inbox.get(rid)); }
    if (PACE) await sleep(PACE); // inside the ration: two a second
  }
  if (!PACE) { for (let w = 0; [...Array(work.length).keys()].some((k) => !inbox.has(10 + k)) && w < 2400; w++) await sleep(25); for (let k = 0; k < work.length; k++) tallyOf(inbox.get(10 + k)); }
  sock.close();
}
function tallyOf(v) { const verdict = v && v.result ? v.result.verdict : "other"; tally[verdict in tally ? verdict : "other"]++; }

const t1 = performance.now();
const results = await Promise.allSettled([...Array(nC).keys()].map(client));
const secs = (performance.now() - t1) / 1000;
const failed = results.filter((r) => r.status === "rejected");
await sleep(1500);
const after = (await (await fetch(`${base}/status?_=${Date.now()}`)).json()).pearl;
console.log(`${nC} clients, ${secs.toFixed(1)} s (${((tally.accepted + tally.duplicate + tally.invalid) / secs).toFixed(0)} verified a second): ${JSON.stringify(tally)}${failed.length ? `; ${failed.length} clients failed: ${failed[0].reason}` : ""}`);
console.log(`shard counters: accepted ${before.accepted} -> ${after.accepted}, forwarded ${after.forwarded} (sent ${after.sent}, answered ${after.answered}, waiting ${after.waiting}, lost ${after.lost}), upstream accepted ${after.upAccepted}`);
// Each distinct proof is accepted once overall; the "someone else's" copy is
// a duplicate unless it arrives before its owner's -- either way, one credit.
const okCount = tally.accepted === nP && tally.accepted + tally.duplicate === nP + nC && tally.invalid === nC && tally.other === 0 && tally.rate === 0 && !failed.length && after.accepted - before.accepted === nP;
console.log(okCount ? "ok    every distinct proof credited once; every duplicate and invalid proof refused; counters agree" : "FAIL  the books do not balance");
process.exit(okCount ? 0 : 1);

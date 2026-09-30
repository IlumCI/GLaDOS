// node pool/edge/worker/epoch.test.mjs
//
// The same claims `tools/distribute.py --selftest` makes about the equal split,
// on the same numbers, so the pool and the tool cannot quietly disagree about
// who is paid.
import { parseLedger, addressOf, workByAddress, medianFloor, build, deltas, floorOf, candidates } from "./epoch.js";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };
const G = 10n ** 18n;
const A = "0x" + "aa".repeat(20), B = "0x" + "bb".repeat(20), C = "0x" + "cc".repeat(20);

// Parsing: a tally past 2^53 survives exactly.
const big = 2n ** 60n + 7n;
const rows = parseLedger(`{"shares":[{"worker":"${A}.rig1","coin":"y","work":${big}},{"worker":"${A}.rig2","coin":"h","work":5},{"worker":"name.rig","coin":"y","work":9}]}`);
ok(rows[0].work === big, "a work tally past 2^53 is read exactly, not rounded");
ok(addressOf("name.rig") === null && addressOf(`${A.toUpperCase().replace("0X", "0x")}.x`) === A, "a worker name that is not an address pays nobody; case is folded");
const wa = workByAddress(rows);
ok(wa.get(A) === big + 5n && wa.size === 1, "two rigs and two coins under one address are one address");

// distribute.py's numbers: three real miners, floor 200.
const honest = new Map([["0xa", 1000n], ["0xb", 800n], ["0xc", 400n]]);
ok(medianFloor(honest, 0.25) === 200n, "the floor is a quarter of the work-weighted median (distribute.py: 200)");
const sybils = new Map(honest);
for (let i = 0; i < 100; i++) sybils.set("0xs" + i, 10n);
ok(medianFloor(sybils, 0.25) === 200n, "a hundred sybil addresses doing slivers cannot move it");

// A whole epoch: the delta since the last one, the gate, the floor.
const rich = 100_000n * G;
const now = new Map([[A, 1500n], [B, 900n], [C, 50n]]);
const prev = new Map([[A, 500n]]);
const e = build({ now, prev, balances: new Map([[A, rich], [B, rich], [C, rich]]), gateMin: 50_000n * G });
ok(e.work[A] === "1000", "an address is credited only the work since the last epoch");
ok(e.recipients.join() === [A, B].sort().join(), "the two real miners are paid alike");
ok(/floor/.test(e.excluded[C]), "and one doing a sliver of typical is not a share");

const poor = build({ now, prev, balances: new Map([[A, rich], [B, 49_999n * G]]), gateMin: 50_000n * G, minFrac: 0 });
ok(!poor.recipients.includes(B) && /holds 49999 of 50000/.test(poor.excluded[B]), "an address under the gate is not paid, and says why");
ok(/unreadable/.test(poor.excluded[C]), "an address whose balance could not be read is refused, not waved through");

const again = build({ now, prev: now, balances: new Map([[A, rich], [B, rich], [C, rich]]), gateMin: 50_000n * G });
ok(again.recipients.length === 0, "an epoch right after the last pays nobody twice");

// The pair, or any contract, is never paid -- one miner naming it would
// otherwise revert every payout for everyone.
const PAIR = "0x93f777932d98d15b351d1bce8c76b34381eede5b";
const withPair = build({ now: new Map([[A, 900n], [PAIR, 900n]]), prev: new Map(), balances: new Map([[A, rich], [PAIR, rich]]),
                         gateMin: 50_000n * G, ineligible: new Set([PAIR]) });
ok(withPair.recipients.join() === A && /cannot be paid/.test(withPair.excluded[PAIR]), "the pair's address is never a recipient, whatever it holds");

// An unreadable balance and a miss of the floor carry the work forward.
const carried = build({ now, prev, balances: new Map([[A, rich], [B, rich]]), gateMin: 50_000n * G, minFrac: 0 });
ok((carried.snapshot.get(C) ?? 0n) === 0n && !carried.updates.has(C), "an unreadable balance's work is not consumed");
const next = build({ now: new Map([[A, 1500n], [B, 900n], [C, 50n]]), prev: carried.snapshot,
                     balances: new Map([[A, rich], [B, rich], [C, rich]]), gateMin: 50_000n * G, minFrac: 0 });
ok(next.recipients.includes(C) && next.work[C] === "50", "and it is paid in full once the balance reads");

// The snapshot never moves backwards: a shard missing from one sum cannot
// lower it and have the same work paid again when it returns.
const dip = build({ now: new Map([[A, 100n]]), prev: new Map([[A, 1500n]]), balances: new Map([[A, rich]]), gateMin: 0n });
ok(dip.snapshot.get(A) === 1500n && dip.recipients.length === 0, "a lower sum neither pays nor lowers the snapshot");

// Past the per-transaction limit: the most work waiting is paid, the rest
// carries, and next epoch the ones left out are paid first.
const crowd = new Map([...Array(10).keys()].map((i) => ["0x" + String(i).padStart(40, "0"), BigInt(100 + i)]));
const bal = new Map([...crowd.keys()].map((a) => [a, rich]));
const cap = build({ now: crowd, prev: new Map(), balances: bal, gateMin: 0n, minFrac: 0, maxRecipients: 4 });
ok(cap.recipients.length === 4 && cap.recipients.every((a) => crowd.get(a) >= 106n), "over the cap, the four with the most work waiting are paid");
ok([...crowd.keys()].filter((a) => !cap.recipients.includes(a)).every((a) => (cap.snapshot.get(a) ?? 0n) === 0n && /carries/.test(cap.excluded[a])),
   "and everybody else's work carries, not consumed");
const grown = new Map([...crowd].map(([a, w]) => [a, w + 10n]));
const nextCap = build({ now: grown, prev: cap.snapshot, balances: bal, gateMin: 0n, minFrac: 0, maxRecipients: 4 });
ok(nextCap.recipients.every((a) => !cap.recipients.includes(a)), "next epoch the ones left out are at the front");

// Bounded reads: only `considered` addresses are judged; the rest are counted,
// carried, and never listed -- at 100,000 miners the list is the problem.
const big5 = new Map([...Array(5).keys()].map((i) => ["0x" + String(i + 50).padStart(40, "0"), BigInt(1000 + i)]));
const only = new Set([...big5.keys()].slice(0, 2));
const part = build({ now: big5, prev: new Map(), balances: new Map([...only].map((a) => [a, rich])), gateMin: 0n, considered: only });
ok(part.recipients.length === 2 && part.notReached === 3 && Object.keys(part.excluded).length === 0, "addresses not reached are counted and carried, not listed");
ok(part.updates.size === 2, "and only the paid move the snapshot, so a payout stores two entries, not five");
// The floor is known before any balance is read, and ranks the reads.
const d = deltas(big5, new Map());
const order = candidates(d, floorOf(d), new Set([[...big5.keys()][4]]));
ok(order.length === 4 && order[0] === [...big5.keys()][3], "candidates come most work first, skipping the known ineligible");

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

// node pool/edge/worker/epoch.test.mjs
//
// The same claims `tools/distribute.py --selftest` makes about the equal split,
// on the same numbers, so the pool and the tool cannot quietly disagree about
// who is paid.
import { parseLedger, addressOf, workByAddress, medianFloor, build } from "./epoch.js";

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

const poor = build({ now, prev, balances: new Map([[A, rich], [B, 49_999n * G]]), gateMin: 50_000n * G });
ok(!poor.recipients.includes(B) && /holds 49999 of 50000/.test(poor.excluded[B]), "an address under the gate is not paid, and says why");
ok(/unreadable/.test(poor.excluded[C]), "an address whose balance could not be read is refused, not waved through");

const again = build({ now, prev: now, balances: new Map([[A, rich], [B, rich], [C, rich]]), gateMin: 50_000n * G });
ok(again.recipients.length === 0, "an epoch right after the last pays nobody twice");

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

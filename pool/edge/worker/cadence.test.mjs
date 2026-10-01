// node pool/edge/worker/cadence.test.mjs
import { worthPaying, hoursUntil } from "./cadence.js";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };

const cost = { perRecipientUsd: 0.003, fixedUsd: 0.05 };

ok(!worthPaying({ ...cost, potUsd: 10, rvn: 100, rvnMin: 151, recipients: 5 }).go,
   "under the swap minimum, nothing is paid however large the pot");
ok(!worthPaying({ ...cost, potUsd: 10, rvn: 500, rvnMin: 151, recipients: 0 }).go,
   "an epoch with nobody in it is not paid");
ok(!worthPaying({ ...cost, potUsd: 0.5, rvn: 500, rvnMin: 151, recipients: 10 }).go,
   "costs of $0.08 on a $0.50 pot are over 10%, so it waits");
ok(worthPaying({ ...cost, potUsd: 1.0, rvn: 500, rvnMin: 151, recipients: 10 }).go,
   "the same epoch at $1.00 is paid");

// The scaling claim, as arithmetic: per-miner cost is fixed and the pot grows
// with miners, so bigger pools pay out more often at the same overhead.
const perMinerDay = 0.08;
const at = (n) => hoursUntil({ ...cost, usdPerDay: n * perMinerDay, recipients: n });
const h10 = at(10), h1000 = at(1000);
console.log(`      10 miners: every ${h10.toFixed(1)} h; 1,000 miners: every ${h1000.toFixed(1)} h`);
ok(h1000 < h10, "a thousand miners are paid more often than ten");
ok(h1000 < 12, "and a thousand are paid within half a day at 10% overhead");

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

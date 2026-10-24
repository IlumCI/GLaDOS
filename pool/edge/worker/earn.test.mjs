// node pool/edge/worker/earn.test.mjs
import { perRecipientEth, effectiveReward, fromWei18 } from "./earn.js";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };
const E = 10n ** 18n;
const G = 10n ** 18n;

// One recipient's share: balance less twice the payout gas, split by count.
const share = perRecipientEth({ ethWei: 10n * E, recipients: 4, baseGas: 200_000, perRecipientGas: 60_000, gasPrice: 10n ** 9n });
const expectGas = (200_000n + 60_000n * 4n) * 10n ** 9n * 2n;
ok(share === (10n * E - expectGas) / 4n, "the share is (balance - 2x gas) / recipients");
ok(share > 0n && share < 10n * E / 4n, "and it is under a naive even split, by the gas it withholds");

// Never negative: a pot smaller than its own gas pays a share of nothing.
ok(perRecipientEth({ ethWei: 1000n, recipients: 3, baseGas: 200_000, perRecipientGas: 60_000, gasPrice: 10n ** 9n }) === 0n,
   "a pot under its gas is a share of zero, not a negative");
ok(perRecipientEth({ ethWei: 5n * E, recipients: 0, baseGas: 0, perRecipientGas: 0, gasPrice: 0n }) === 5n * E,
   "a recipient count of zero is treated as one, never a divide by zero");

// The gate decides the asset: the choice at or over the minimum, else $GLaDOS.
ok(effectiveReward("nvda", 2_000_000n * G, 1_000_000n * G) === "nvda", "a holder keeps its choice");
ok(effectiveReward("nvda", 999_999n * G, 1_000_000n * G) === "glados", "under the minimum is paid $GLaDOS whatever it chose");
ok(effectiveReward(undefined, 5_000_000n * G, 1_000_000n * G) === "glados", "a holder with no choice is $GLaDOS");

// Raw 18-decimal amounts become display numbers.
ok(fromWei18(1_500_000_000_000_000_000n) === 1.5, "1.5e18 wei reads as 1.5");
ok(fromWei18(0n) === 0, "zero is zero");

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

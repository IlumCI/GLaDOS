// node pool/edge/worker/flow.test.mjs
//
// Every path of the treasury's decision, with no network and no coins. The
// claims that earn their place are the ones about money moving twice or not at
// all: a restart mid-broadcast must resend the same bytes, a failed payout must
// not be retried forever, and nothing pays while a transaction is in flight.
import { decide, FRESH, rvnFee } from "./flow.js";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };

const cfg = { dropAfterMs: 600_000, maxFailures: 3, deployGas: 700_000, baseGas: 170_000, perRecipientGas: 51_000,
              maxOverhead: 0.1, rvnBatchSats: 200n * 10n ** 8n, maxPayWei: 10n ** 18n };
const gwei = 22_600_000n; // 0.0226 gwei, measured
const utxo = (sats) => ({ txid: "aa".repeat(32), vout: 0, sats });

// Idle, too little RVN: nothing happens.
let r = decide(FRESH, { now: 0, rvnUtxos: [utxo(100n * 10n ** 8n)], rvnMinSats: 151n * 10n ** 8n, gasPrice: gwei }, cfg);
ok(r.action.kind === "wait", "under the batch size, the treasury waits");

// Enough RVN: an exchange for all of it minus the fee.
r = decide(FRESH, { now: 0, rvnUtxos: [utxo(150n * 10n ** 8n), utxo(60n * 10n ** 8n)], rvnMinSats: 151n * 10n ** 8n, gasPrice: gwei }, cfg);
ok(r.action.kind === "create_exchange" && r.action.sats === 210n * 10n ** 8n - rvnFee(2),
   "at the batch size an exchange is created for everything but the fee");

// An exchange created, RVN not yet sent: send it, with exactly one change output.
let st = { ...FRESH, exchange: { id: "x1", payin: "RPAYIN", sats: (210n * 10n ** 8n - rvnFee(2)).toString() } };
r = decide(st, { now: 0, rvnUtxos: [utxo(150n * 10n ** 8n), utxo(60n * 10n ** 8n)] }, cfg);
ok(r.action.kind === "send_rvn" && r.action.to === "RPAYIN" && r.action.change === 0n, "the RVN is sent to the exchange's address, the whole of it");

// The RVN transaction in flight: nothing else happens until it settles.
st = { ...st, pending: { kind: "rvn", chain: "rvn", hash: "t1", raw: "00", at: 0 } };
r = decide(st, { now: 60_000, receipt: null, recipients: ["0xa"], ethWei: 10n ** 17n, gasPrice: gwei }, cfg);
ok(r.action.kind === "wait", "while a transaction is in flight nothing else is done, not even a payout");

// It is not mined for too long: the same bytes again, never a new transaction.
r = decide(st, { now: 700_000, receipt: null }, cfg);
ok(r.action.kind === "rebroadcast" && r.action.raw === "00", "a transaction not mined in time is rebroadcast byte for byte");

// Confirmed: the exchange is marked sent; then it finishes and we are funded.
r = decide(st, { now: 60_000, receipt: { ok: true } }, cfg);
ok(r.state.pending === null && r.state.exchange.sent === "t1", "a confirmed RVN send marks the exchange sent");
r = decide(r.state, { now: 0, exchangeStatus: "waiting" }, cfg);
ok(r.action.kind === "wait", "and the treasury waits on ChangeNOW");
r = decide(r.state, { now: 0, exchangeStatus: "finished" }, cfg);
ok(r.state.exchange === null && r.state.phase === "funded", "a finished exchange leaves the treasury funded");
const refunded = decide({ ...FRESH, exchange: { id: "x2", sent: "t2" } }, { exchangeStatus: "refunded" }, cfg);
ok(refunded.state.exchange === null && /refunded/.test(refunded.action.why), "a refunded exchange is let go; the RVN comes back to the pool's wallet");

// Funded, no contract yet: deploy it.
const eth = 5n * 10n ** 15n; // 0.005 ETH
r = decide({ ...FRESH, phase: "funded" }, { ethWei: eth, gasPrice: gwei, recipients: ["0xa", "0xb"] }, cfg);
ok(r.action.kind === "deploy", "the first ETH deploys the payout contract before anything is paid");

// Contract present, worth paying: pay, keeping this payout's gas twice over.
r = decide({ ...FRESH, contract: "0xc" }, { ethWei: eth, gasPrice: gwei, recipients: ["0xa", "0xb"] }, cfg);
const gas = 170_000n + 51_000n * 2n;
ok(r.action.kind === "pay" && r.action.value === eth - gas * gwei * 2n, "a payout sends everything but twice its own gas");

// Not worth it: gas over 10% of the pot.
r = decide({ ...FRESH, contract: "0xc" }, { ethWei: 10n ** 11n, gasPrice: gwei, recipients: ["0xa", "0xb"] }, cfg);
ok(r.action.kind !== "pay", "a pot the gas would eat more than a tenth of is not paid");

// Over the cap: halt, do not pay.
r = decide({ ...FRESH, contract: "0xc" }, { ethWei: 2n * 10n ** 18n, gasPrice: gwei, recipients: ["0xa"] }, cfg);
ok(r.action.kind === "wait" && r.state.halted, "a payout over the cap halts the treasury rather than sending it");

// Three failed transactions in a row: halt.
let h = { ...FRESH, contract: "0xc", failures: 2, pending: { kind: "pay", chain: "evm", hash: "p", at: 0 } };
r = decide(h, { receipt: { ok: false } }, cfg);
ok(r.state.halted && /3 failed/.test(r.state.halted), "a third failure in a row halts the treasury");
ok(decide(r.state, { ethWei: eth, gasPrice: gwei, recipients: ["0xa"] }, cfg).action.kind === "wait", "and a halted treasury does nothing further");

// A confirmed payout records the epoch and returns to idle.
r = decide({ ...FRESH, contract: "0xc", pending: { kind: "pay", chain: "evm", hash: "p", at: 0, epoch: { id: 1 } } }, { receipt: { ok: true } }, cfg);
ok(r.action.kind === "record_epoch" && r.state.epochs === 1, "a confirmed payout is recorded as an epoch");

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

// node pool/edge/worker/flow.test.mjs
//
// Every path of the treasury's decision, with no network and no coins. The
// claims that earn their place are the ones about money moving twice or not at
// all: a restart mid-broadcast must resend the same bytes, a failed payout must
// not be retried forever, and nothing pays while a transaction is in flight.
import { decide, FRESH, rvnFee, backoffMs } from "./flow.js";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };

const cfg = { exchangeUnsentMs: 2 * 3_600_000, exchangeStuckMs: 48 * 3_600_000, dropAfterMs: 600_000, maxFailures: 3, deployGas: 700_000, baseGas: 170_000, perRecipientGas: 51_000,
              maxOverhead: 0.1, rvnBatchSats: 200n * 10n ** 8n, maxPayWei: 10n ** 18n };
const gwei = 22_600_000n; // 0.0226 gwei, measured
let seq = 0;
const utxo = (sats, txid) => ({ txid: txid || (seq++).toString(16).padStart(64, "0"), vout: 0, sats });

// Idle, too little RVN: nothing happens.
let r = decide(FRESH, { now: 0, rvnUtxos: [utxo(100n * 10n ** 8n)], rvnMinSats: 151n * 10n ** 8n, gasPrice: gwei }, cfg);
ok(r.action.kind === "wait", "under the batch size, the treasury waits");

// Enough RVN: an exchange for all of it minus the fee.
r = decide(FRESH, { now: 0, rvnUtxos: [utxo(150n * 10n ** 8n), utxo(60n * 10n ** 8n)], rvnMinSats: 151n * 10n ** 8n, gasPrice: gwei }, cfg);
ok(r.action.kind === "create_exchange" && r.action.sats === 210n * 10n ** 8n - rvnFee(2),
   "at the batch size an exchange is created for everything but the fee");

// An exchange created, RVN not yet sent: send exactly its recorded coins.
let st = { ...FRESH, exchange: { id: "x1", payin: "RPAYIN", sats: (210n * 10n ** 8n - rvnFee(2)).toString(), outpoints: [`${"aa".repeat(32)}:0`, `${"bb".repeat(32)}:0`] } };
r = decide(st, { now: 0, rvnUtxos: [utxo(150n * 10n ** 8n, "aa".repeat(32)), utxo(60n * 10n ** 8n, "bb".repeat(32)), utxo(99n * 10n ** 8n)] }, cfg);
ok(r.action.kind === "send_rvn" && r.action.to === "RPAYIN" && r.action.change === 0n && r.action.utxos.length === 2,
   "the RVN is sent to the exchange's address: exactly its recorded coins, not a new arrival");

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
const refunded = decide({ ...FRESH, exchange: { id: "x2", sent: "t2" } }, { now: 0, exchangeStatus: "refunded" }, cfg);
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

// Over the cap: pay the cap, keep the rest for the next epoch.
r = decide({ ...FRESH, contract: "0xc" }, { now: 0, ethWei: 2n * 10n ** 18n, gasPrice: gwei, recipients: ["0xa"] }, cfg);
ok(r.action.kind === "pay" && r.action.value === cfg.maxPayWei, "a pot over the cap is paid a cap at a time, the rest left for the next epoch");

// Three failed transactions in a row: a pause that ends by itself, doubling.
let h = { ...FRESH, contract: "0xc", failures: 2, pending: { kind: "pay", chain: "evm", hash: "p", at: 0 } };
r = decide(h, { now: 0, receipt: { ok: false } }, cfg);
ok(r.state.paused && r.state.paused.until === 3_600_000 && /3 failed/.test(r.state.incidents.at(-1).why), "a third failure in a row pauses for an hour, as an incident");
ok(decide(r.state, { now: 1_000, ethWei: eth, gasPrice: gwei, recipients: ["0xa"] }, cfg).action.kind === "wait", "and a paused treasury does nothing meanwhile");
ok(decide(r.state, { now: 3_600_001, ethWei: eth, gasPrice: gwei, recipients: ["0xa"] }, cfg).action.kind === "pay", "and resumes by itself when the pause runs out");
ok(backoffMs(0) === 3_600_000 && backoffMs(1) === 7_200_000 && backoffMs(9) === 86_400_000, "pauses double from an hour to at most a day");

// A confirmed payout records the epoch and returns to idle.
r = decide({ ...FRESH, contract: "0xc", pending: { kind: "pay", chain: "evm", hash: "p", at: 0, epoch: { id: 1 } } }, { receipt: { ok: true } }, cfg);
ok(r.action.kind === "record_epoch" && r.state.epochs === 1, "a confirmed payout is recorded as an epoch");

// Change below dust is folded into the fee rather than creating an output the
// network refuses.
r = decide({ ...FRESH, exchange: { id: "x3", payin: "RPAYIN", sats: "100000000", created: 0, outpoints: [`${"cc".repeat(32)}:0`] } },
           { now: 0, rvnUtxos: [utxo(100_000_000n + rvnFee(1) + 5000n, "cc".repeat(32))] }, cfg);
ok(r.action.kind === "send_rvn" && r.action.change === 0n && r.action.fee === rvnFee(1) + 5000n,
   "a change remainder under the dust line goes to the fee, not to an output nodes refuse");

// An exchange never funded lapses and is dropped; one funded and stuck is written off.
r = decide({ ...FRESH, exchange: { id: "x4", payin: "RPAYIN", sats: "1", created: 0 } }, { now: 3 * 3_600_000, rvnUtxos: [] }, cfg);
ok(r.state.exchange === null && /never funded/.test(r.action.why), "an exchange unfunded after two hours is dropped, not paid late");
r = decide({ ...FRESH, exchange: { id: "x5", sent: "t", created: 0 } }, { now: 49 * 3_600_000, exchangeStatus: "verifying" }, cfg);
ok(!r.state.paused && r.state.exchange === null && r.state.orphans.at(-1).id === "x5" && /x5/.test(r.state.incidents.at(-1).why),
   "an exchange funded and unfinished after two days is written off, still watched, and the treasury carries on");
r = decide({ ...FRESH, exchange: { id: "x6", sent: "t", created: 0 } }, { now: 49 * 3_600_000, exchangeStatus: "finished" }, cfg);
ok(!r.state.orphans.length && r.state.phase === "funded", "but one that finished late is simply funded");

// Too many recipients for one transaction: the first 400 (epoch.js already
// ranks and caps; this is the second line).
r = decide({ ...FRESH, contract: "0xc" }, { now: 0, ethWei: 10n ** 17n, gasPrice: gwei, recipients: Array(401).fill("0xa") }, cfg);
ok(r.action.kind === "pay" && r.action.recipients.length === 400, "a list over 400 recipients is trimmed to 400, never sent whole");

// The second review's stall: a hundred small payouts plus dust. Largest first,
// dust ignored, amount from exactly the chosen coins, and never under the minimum.
seq = 1000;
const many = [...Array(100)].map(() => utxo(250_000_000n));
const dust = [...Array(50)].map(() => utxo(1000n));
r = decide(FRESH, { now: 0, rvnUtxos: [...dust, ...many], rvnMinSats: 151n * 10n ** 8n, gasPrice: gwei }, cfg);
ok(r.action.kind === "wait", "fifty 2.5 RVN coins (the most one exchange spends) do not reach the 200 RVN batch, so it waits");
const big = [...Array(60)].map(() => utxo(500_000_000n));
r = decide(FRESH, { now: 0, rvnUtxos: [...dust, ...many, ...big], rvnMinSats: 151n * 10n ** 8n, gasPrice: gwei }, cfg);
ok(r.action.kind === "create_exchange" && r.action.outpoints.length === 50 && r.action.sats === 50n * 500_000_000n - rvnFee(50),
   "with larger coins present it takes the fifty largest, ignores the dust, and sizes the exchange from exactly those");

// A failed exchange is written off with its id kept and watched; coins gone from
// under an unsent exchange drop it; a nonce lost to somebody else lets go.
r = decide({ ...FRESH, exchange: { id: "xf", sent: "t", created: 0 } }, { now: 0, exchangeStatus: "failed" }, cfg);
ok(!r.state.paused && r.state.orphans.at(-1).id === "xf" && /xf failed/.test(r.state.incidents.at(-1).why), "a failed exchange is written off, its id kept among the watched");
r = decide({ ...FRESH, exchange: { id: "xm", payin: "R", sats: "1", created: 0, outpoints: ["ff:0"] } }, { now: 0, rvnUtxos: [] }, cfg);
ok(r.state.exchange === null && /xm's coins/.test(r.state.incidents.at(-1).why), "coins vanishing from under an unsent exchange drop it before anything is sent");
r = decide({ ...FRESH, pending: { kind: "pay", chain: "evm", hash: "h", at: 0 } }, { now: 0, receipt: { lost: "nonce 7 used by another transaction" } }, cfg);
ok(r.state.pending === null && !r.state.paused && /nonce 7/.test(r.state.incidents.at(-1).why), "a transaction whose nonce somebody else used is let go, as an incident");
// Nothing in this module ever halts.
ok(!/s\.halted\s*=/.test((await import("node:fs")).readFileSync(new URL("./flow.js", import.meta.url), "utf8")), "flow.js has no halt left in it");

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

// node pool/edge/worker/runner-units.test.mjs
//
// Pieces of the treasury that the stress world never reaches, called directly.
// Each is here because mutation testing changed it and no test noticed:
//   - rebuilding a lost payout record from the chain (it decoded the wrong
//     layout: everybody paid would have been paid again)
//   - the price guard's median
import { Treasury } from "./runner.js";
import { encodePayAll, encodeBuyAndPay, payoutRecipients } from "./treasury.js";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };
const addr = (i) => "0x" + i.toString(16).padStart(40, "0");
const POOL = "0x" + "11".repeat(20), TOK = "0x" + "22".repeat(20);

// ---------------------------------------------------------- who a payout paid
const groups = [
  { to: [addr(1), addr(2), addr(3)], legs: [{ pool: POOL, token: TOK, eth: 10n ** 15n, minOut: 123n }] },
  { to: [addr(4)], legs: [{ pool: POOL, token: TOK, eth: 5n, minOut: 7n }, { pool: POOL, token: TOK, eth: 6n, minOut: 8n }] },
  { to: [addr(5), addr(6)], legs: [] },
];
const payAll = encodePayAll(groups);
const want = groups.flatMap((g) => g.to);
const got = payoutRecipients(payAll);
ok(got && got.join() === want.join(), `payAll calldata, three groups and mixed legs: the six wallets paid (${got && got.length})`);
ok(payoutRecipients(encodeBuyAndPay([addr(7), addr(8)], 99n)).join() === [addr(7), addr(8)].join(), "the older buyAndPay: its two wallets");
ok(payoutRecipients(encodePayAll([{ to: [addr(9)], legs: [] }, { to: [addr(9)], legs: [] }])).join() === addr(9), "a wallet in two groups is one wallet");
const big = Array.from({ length: 400 }, (_, i) => addr(1000 + i));
ok(payoutRecipients(encodePayAll([{ to: big, legs: [] }])).length === 400, "four hundred wallets in one group");

ok(payoutRecipients("0x12345678" + "00".repeat(64)) === null, "an unknown selector: null, not a guess");
ok(payoutRecipients(payAll.slice(0, payAll.length - 64)) === null, "payAll cut one word short: null");
ok(payoutRecipients(payAll.slice(0, 74)) === null, "payAll with only its head: null");
const huge = payAll.slice(0, 10) + payAll.slice(10, 74) + "f".repeat(64) + payAll.slice(138);
ok(payoutRecipients(huge) === null, "a group count of 2^256-1: null, and no endless loop");
const notAddr = encodeBuyAndPay([addr(1)], 1n).replace(/0{24}(0{39}1)$/, "ff" + "0".repeat(22) + "$1");
ok(payoutRecipients(notAddr) === null, "a 'recipient' with its high bytes set is not an address: null");
ok(payoutRecipients(undefined) === null && payoutRecipients("0x") === null && payoutRecipients("hello") === null, "nothing, empty, or not hex: null");

// ---------------------------------------------------------- rebuilding a record
const store = new Map();
const pool = { env: {}, ctx: { storage: { get: async (k) => store.get(k), put: async (k, v) => { store.set(k, v); }, delete: async (k) => store.delete(k) } }, log: () => {} };
const t = new Treasury(pool, { now: () => 1_000_000 });
t.rpc = async (method, params) => (method === "eth_getTransactionByHash" && params[0] === "0xabc" ? { input: payAll } : { input: "0xdeadbeef" });
t.shardWork = async () => new Map([[addr(1), 500n], [addr(2), 600n], [addr(3), 700n], [addr(4), 800n], [addr(5), 900n], [addr(6), 1000n], [addr(77), 5n]]);
const next = { epochs: 4, incidents: [] };
const rec = await t.reconstruct("0xabc", next);
ok(rec.recipients.join() === want.join(), "a lost payAll record is rebuilt with the wallets the chain paid");
ok(Object.keys(rec.updates).sort().join() === [...want].sort().join() && rec.updates[addr(6)] === "1000", "each marked paid up to its work now, and nobody else");
let threw = null;
try { await t.reconstruct("0xother", { epochs: 5, incidents: [] }); } catch (e) { threw = e; }
ok(threw && /does not read as a payout/.test(threw.message), "calldata that is not a payout: tried again later, not rebuilt from a guess");

// ---------------------------------------------------------- the price guard's median
{
  const g = new Treasury(pool, { now: () => 1 });
  const at = (history, px) => { const s = { px: [...history] }; return g.priceMoved(s, 10n ** 6n, BigInt(px)); };
  // Five readings at 1000 and four at 5000, interleaved: the median is 1000.
  const hist = [5000, 1000, 5000, 1000, 1000, 5000, 1000, 5000, 1000];
  ok(at(hist, 1000) === null, "the price at the median of mixed readings: no move");
  ok(/off its recent median/.test(at(hist, 5000) || ""), "the price at the outliers' level: a 400% move, refused");
  // Eleven readings where only a real numeric sort finds the middle: a broken
  // comparator lands on 5000 here and would wave a 400% move through.
  const eleven = [1000, 5000, 1000, 5000, 1000, 5000, 1000, 1000, 5000, 1000, 5000];
  ok(at(eleven, 1000) === null && /off its recent median/.test(at(eleven, 5000) || ""), "eleven interleaved readings: the median is 1000, so 1000 passes and 5000 is refused");
  ok(at([1000, 1010, 990, 1005], 1100) === null && /off its recent median/.test(at([1000, 1010, 990, 1005], 1200) || ""), "a 10% move passes, a 20% one does not (the limit is 15%)");
  ok(/before the first buy/.test(at([1000], 1000) || ""), "too few readings: wait for more");
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

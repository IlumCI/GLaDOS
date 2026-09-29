// node pool/edge/worker/chunks.test.mjs
import { textChunks, loadText, CHUNK } from "./chunks.js";
let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };
const store = () => { const m = new Map(); return { m, get: async (k) => (Array.isArray(k) ? new Map(k.filter((x) => m.has(x)).map((x) => [x, m.get(x)])) : m.get(k)), put: async (o) => { for (const [k, v] of Object.entries(o)) m.set(k, v); } }; };
// A ledger the size 4,096 tallies make: about 530 KB, four times one value's cap.
const row = (i) => `{"worker":"0x${String(i).padStart(40, "0")}.rig","coin":"yespowerR16","work":${i * 977},"accepted":1,"stale":0,"bad":0,"duplicate":0}`;
const ledger = `{"v":3,"shares":[${[...Array(4096).keys()].map(row).join(",")}]}`;
const s = store();
const e = textChunks("ledger", ledger);
ok(Object.values(e).every((v) => typeof v !== "string" || v.length <= CHUNK) && ledger.length > 128 * 1024, `a ${ledger.length}-byte ledger is written as ${Object.keys(e).length - 1} chunks, each under the cap`);
await s.put(e);
ok((await loadText(s, "ledger")) === ledger, "and read back byte for byte");
const small = store(); await small.put({ ledger: '{"v":3,"shares":[]}' });
ok((await loadText(small, "ledger")) === '{"v":3,"shares":[]}', "a ledger saved before chunking still loads");
ok((await loadText(store(), "ledger")) === undefined, "no ledger is undefined, not an error");
s.m.delete("ledger#2");
let threw = false; try { await loadText(s, "ledger"); } catch { threw = true; }
ok(threw, "a missing chunk refuses to load rather than returning part of a ledger");
// A shorter save after a longer one leaves stale chunks behind; the head decides.
await s.put(textChunks("ledger", ledger)); await s.put(textChunks("ledger", "short"));
ok((await loadText(s, "ledger")) === "short", "a shorter save after a longer one reads as the shorter one");
console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

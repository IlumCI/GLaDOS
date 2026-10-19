// node pool/edge/worker/guard.test.mjs
//
// What a failure of the pool is called, and the 503 that says it. Pinned
// because mutation testing found every Retry-After here could become 0 -- a
// quota outage telling every miner to retry at once -- and the 503 itself
// could change, without a test noticing.
import { classifyFailure, unavailable } from "./guard.js";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };

const quota = ["Exceeded allowed duration in Durable Objects free tier.", "Daily limit reached", "Your account has exceeded its quota", "exceeded allowed rows read"];
for (const m of quota) {
  const c = classifyFailure(new Error(m));
  ok(c.quota === true && c.retry === 300 && /daily Cloudflare allowance/.test(c.why), `"${m}": the allowance, retry in 300 s`);
}
const notQuota = ["Maximum call stack size exceeded", "Durable Object storage operation exceeded timeout", "fetch failed"];
for (const m of notQuota) ok(!classifyFailure(new Error(m)).quota, `"${m}": not the allowance`);

const over = classifyFailure(Object.assign(new Error("overloaded"), { overloaded: true }));
ok(!over.quota && over.retry === 30 && /overloaded/.test(over.why), "an overloaded object: back off 30 s");
for (const m of ["ledger is missing a chunk; refusing to read a partial value", "edge_load_ledger: digest mismatch"]) {
  const c = classifyFailure(new Error(m));
  ok(c.retry === 300 && /share log did not load/.test(c.why), `"${m.slice(0, 40)}...": the share log, retry in 300 s`);
}
const other = classifyFailure(new Error("secret internals at /x/y.js:12"));
ok(other.retry === 30 && !/secret|internals/.test(other.why), "anything else: 30 s, and its message is not repeated");
ok(classifyFailure(undefined).retry === 30 && classifyFailure("a string").retry === 30, "a thrown non-Error is classified, not a crash");

const r = unavailable("the reason", 300);
const body = await r.json();
ok(r.status === 503, "unavailable() is a 503");
ok(r.headers.get("retry-after") === "300" && r.headers.get("cache-control") === "no-store" && r.headers.get("access-control-allow-origin") === "*", "with Retry-After, no-store, and readable cross-origin");
ok(body.why === "the reason" && body.retryAfter === 300 && body.error === "pool unavailable", "and the reason as JSON");

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

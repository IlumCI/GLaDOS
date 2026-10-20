// node pool/edge/worker/rpc.test.mjs
import { verdict, batchVerdict, Endpoints, attempt, rpcList, DEFAULT_RPCS, RETURN_MS, Transient, RpcError, plainPost } from "./rpc.js";
import http from "node:http";
import { Treasury, Transient as RunnerTransient, RpcError as RunnerRpcError } from "./runner.js";
import { classifyFailure } from "./guard.js";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };

// --- what an answer means ---------------------------------------------------------------

ok(verdict("HTTP 403: <!DOCTYPE html><html>Access denied").transient,
   "an HTML page from a firewall is no answer, not the chain refusing");
ok(verdict({ message: "Request timeout on the free plan" }).transient,
   "a gateway's {message} (drpc's 408) is no answer");
ok(verdict({ error: "not_valid_api_key" }).transient,
   "an error that is a bare string is not a JSON-RPC verdict");
const rev = verdict({ jsonrpc: "2.0", id: 1, error: { code: -32000, message: "execution reverted" } });
ok(rev.error === "execution reverted" && rev.code === -32000, "a revert is a definite answer");
ok(verdict({ jsonrpc: "2.0", id: 1, error: { code: -32603, message: "execution reverted: paused" } }).error,
   "a revert wrapped in -32603 is still the chain's answer");
ok(verdict({ jsonrpc: "2.0", id: 1, error: { code: -32603, message: "internal error" } }).transient,
   "-32603 without a revert is the node's trouble");
ok(verdict({ jsonrpc: "2.0", id: 1, error: { code: -32601, message: "the method eth_call does not exist" } }).transient,
   "a method this endpoint does not serve sends the call to another");
ok(verdict({ jsonrpc: "2.0", id: 1, error: { code: -32005, message: "x" } }).transient, "-32005 is a limit");
ok(verdict({ jsonrpc: "2.0", id: 1, error: { code: -32000, message: "header not found" } }).transient,
   "a lagging node's 'header not found' is transient whatever its code");
ok("result" in verdict({ jsonrpc: "2.0", id: 1, result: null }) && verdict({ jsonrpc: "2.0", id: 1, result: null }).result === null,
   "a null result (a receipt not yet mined) is an answer");
ok(batchVerdict([{ id: 0, result: "0x1" }]).result && batchVerdict({ error: { code: -32005, message: "batch limit" } }).transient,
   "a batch is an array or it is unanswered");

// --- the list ---------------------------------------------------------------------------

ok(rpcList({}).length === DEFAULT_RPCS.length && rpcList({})[0] === "https://rpc.mainnet.chain.robinhood.com",
   "the default list starts at Robinhood's own endpoint");
ok(rpcList({ RPC_URLS: "https://a.test  http://plain.test https://a.test https://b.test" }).join(" ") === "https://a.test https://b.test",
   "RPC_URLS replaces it: https only, no duplicates");

// A world of endpoints: each answers chain id `chain` and then whatever `answer` says.
function world(spec) {
  const calls = [];
  const post = async (url, payload) => {
    calls.push([url, Array.isArray(payload) ? "batch" : payload.method]);
    const s = spec[url];
    if (s.down) throw new Transient(`${url} answered 503`);
    if (!Array.isArray(payload) && payload.method === "eth_chainId") return { jsonrpc: "2.0", id: 1, result: s.chain || "0x1237" };
    return s.answer(payload);
  };
  return { post, calls };
}

{
  let t = 0;
  const eps = new Endpoints(["https://a.test", "https://b.test"], () => t);
  const w = world({ "https://a.test": { down: true }, "https://b.test": { answer: () => ({ jsonrpc: "2.0", id: 1, result: "0x5" }) } });
  let first = null;
  try { await attempt(eps, { jsonrpc: "2.0", id: 1, method: "eth_blockNumber", params: [] }, w.post); } catch (e) { first = e; }
  ok(first instanceof Transient, "the first endpoint down: that attempt is Transient");
  const v = await attempt(eps, { jsonrpc: "2.0", id: 1, method: "eth_blockNumber", params: [] }, w.post);
  ok(v.result === "0x5", "and the next attempt is answered by the second");
  const before = w.calls.length;
  await attempt(eps, { jsonrpc: "2.0", id: 1, method: "eth_blockNumber", params: [] }, w.post);
  ok(w.calls.slice(before).every(([u]) => u === "https://b.test") && w.calls.length - before === 1,
     "it sticks to the endpoint that answered, and proves the chain only once");
  t += RETURN_MS;
  ok(eps.current() === "https://a.test", `after ${RETURN_MS / 60_000} min the first is tried again`);
}

{
  const eps = new Endpoints(["https://evil.test", "https://b.test"]);
  const w = world({ "https://evil.test": { chain: "0x1", answer: () => ({ jsonrpc: "2.0", id: 1, result: "0x" + "f".repeat(64) }) },
                    "https://b.test": { answer: () => ({ jsonrpc: "2.0", id: 1, result: "0x0" }) } });
  const p = { jsonrpc: "2.0", id: 1, method: "eth_getBalance", params: ["0x0000000000000000000000000000000000000001", "latest"] };
  await attempt(eps, p, w.post).catch(() => {});
  const v = await attempt(eps, p, w.post);
  ok(v.result === "0x0", "an endpoint on another chain never answers a balance");
  ok(!w.calls.some(([u, m]) => u === "https://evil.test" && m !== "eth_chainId"), "it is asked its chain id and nothing else");
  ok(eps.describe().wrongChain.includes("evil.test"), "and /treasury names it");
}

{
  const eps = new Endpoints(["https://a.test", "https://b.test"]);
  const w = world({ "https://a.test": { answer: () => ({ jsonrpc: "2.0", id: 1, error: { code: -32000, message: "execution reverted" } }) },
                    "https://b.test": { answer: () => ({ jsonrpc: "2.0", id: 1, result: "0x1" }) } });
  const v = await attempt(eps, { jsonrpc: "2.0", id: 1, method: "eth_estimateGas", params: [{}] }, w.post);
  ok(v.error === "execution reverted" && eps.current() === "https://a.test",
     "a definite answer is final: a revert is not shopped around until some node agrees");
}

// --- through the treasury: the bug this file exists for ---------------------------------

function treasuryOn(urls, respond) {
  const storage = new Map();
  const pool = { env: { TREASURY: "live", RPC_URLS: urls.join(" ") },
                 ctx: { storage: { get: async (k) => storage.get(k), put: async (k, v) => { storage.set(k, v); } } },
                 log: () => {} };
  // Not zero: a health record's times are tested for truth, as in production.
  let now = 1_760_000_000_000;
  const io = { now: () => now, sleep: async (ms) => { now += ms; },
               fetch: async (url, init) => respond(url, JSON.parse(init.body)) };
  return new Treasury(pool, io);
}
const json = (b, status = 200) => new Response(JSON.stringify(b), { status });
const chainId = (body) => !Array.isArray(body) && body.method === "eth_chainId";

{
  const t = treasuryOn(["https://only.test"], (url, body) =>
    chainId(body) ? json({ jsonrpc: "2.0", id: 1, result: "0x1237" })
                  : new Response("<!DOCTYPE html><html><title>Access denied</title></html>", { status: 403 }));
  let e = null;
  try { await t.rpc("eth_estimateGas", [{}]); } catch (x) { e = x; }
  ok(e instanceof RunnerTransient && !(e instanceof RunnerRpcError),
     "a 403 page on eth_estimateGas is Transient: it no longer counts toward the three refusals that pause the treasury");
}

{
  const seen = [];
  const t = treasuryOn(["https://a.test", "https://b.test"], (url, body) => {
    seen.push(url);
    if (url === "https://a.test") return new Response("<html>403</html>", { status: 403 });
    return chainId(body) ? json({ jsonrpc: "2.0", id: 1, result: "0x1237" }) : json({ jsonrpc: "2.0", id: 1, result: "0x2a" });
  });
  const r = await t.rpc("eth_getTransactionCount", ["0x00", "latest"]);
  ok(r === "0x2a", "a refused primary: the treasury's call is answered by the fallback");
  const s = await t.summary().catch(() => null);
  ok(t.health && t.health.rpc && t.health.rpc.fail && t.health["rpc:b.test"] && t.health["rpc:b.test"].ok,
     "health records the primary under its old name and the fallback by host");
  ok(s && s.rpc.using === "b.test" && s.status === "ok" && s.down.includes("rpc"),
     "/treasury says which endpoint is in use and that the primary is down -- and, the fallback paying, the payouts are ok");
}

{
  const t = treasuryOn(["https://a.test"], (url, body) =>
    chainId(body) ? json({ jsonrpc: "2.0", id: 1, result: "0x1237" })
                  : json({ jsonrpc: "2.0", id: 1, error: { code: -32000, message: "nonce too low" } }));
  let e = null;
  try { await t.rpc("eth_sendRawTransaction", ["0x00"]); } catch (x) { e = x; }
  ok(e instanceof RunnerRpcError && /nonce too low/.test(e.message), "a node's definite refusal still reaches the caller as RpcError");
}

{
  // The primary refused once, a fallback answered, and then the primary came
  // back: the fallback's last word is a success. The reverse -- a fallback that
  // failed once and is never asked again -- must not read as an outage forever.
  const t = treasuryOn(["https://a.test", "https://b.test"], () => json({}));
  const H = 1_760_000_000_000;
  t.health = { rpc: { ok: H + 10, fail: H, since: H }, "rpc:b.test": { fail: H + 5, since: H + 5, why: "HTTP 403" } };
  t.now = () => H + 5 * 3_600_000;
  const s = { incidents: [] };
  t.watchHealth(s);
  const sum = await t.summary();
  ok(sum.status === "ok" && !s.incidents.length, "a fallback that failed once and sits unused is not an outage, and raises no alert");
  t.health.rpc = { ok: H, fail: H + 20, since: H + 20, why: "HTTP 503" };
  t.watchHealth(s);
  ok(s.incidents.length === 2 && s.incidents.every((i) => /waits and resumes/.test(i.why)) && (await t.summary()).down.includes("rpc:b.test"),
     "every endpoint down is the outage: each endpoint alerted by name, and all of them listed");
}
{
  const t = treasuryOn(["https://a.test", "https://b.test"], () => json({}));
  const H = 1_760_000_000_000;
  t.health = { rpc: { ok: H, fail: H + 20, since: H + 20, why: "HTTP 503" }, "rpc:b.test": { ok: H + 30 } };
  t.now = () => H + 3 * 3_600_000;
  const s = { incidents: [] };
  t.watchHealth(s);
  ok(s.incidents.length === 1 && /carry the payouts/.test(s.incidents[0].why),
     "the primary down two hours behind a working fallback: alerted, and it says payouts continue");
}

// --- a failure of the pool itself ---------------------------------------------------------

ok(classifyFailure(new Error("Exceeded allowed duration in Durable Objects free tier.")).quota,
   "the daily Durable Object allowance is recognised");
ok(!classifyFailure(new RangeError("Maximum call stack size exceeded")).quota,
   "an unrelated 'exceeded' does not hold the pool shut");
ok(classifyFailure(Object.assign(new Error("x"), { overloaded: true })).why.includes("overloaded"), "an overloaded object says so");
ok(classifyFailure(new Error("ledger is missing a chunk; refusing to read a partial value")).why.includes("share log"),
   "a share log that would not load is named, not called an internal error");
ok(!/secret|key/i.test(classifyFailure(new Error("secret key 0xabc leaked")).why), "an unknown error's message is never echoed");

// --- the status word, with one endpoint refusing --------------------------------------------
{
  // Live 2026-10-20: the first endpoint rate-limited, the second answering, and
  // the status page called the payouts degraded ("rpc unreachable").
  const st = new Map();
  const t = new Treasury({ env: {}, ctx: { storage: { get: async (k) => st.get(k), put: async () => {}, delete: async () => {} } } }, { now: () => 2000 });
  t.health = { rpc: { ok: 1000, fail: 1500, why: "answered 429", since: 1500 }, "rpc:robinhood-rpc.publicnode.com": { ok: 1900 }, changenow: { ok: 1900 } };
  const s1 = await t.summary();
  ok(s1.status === "ok" && s1.down.includes("rpc"), `the first 4663 endpoint refusing while another answers: ok, the refusing one still named (${s1.status}, ${JSON.stringify(s1.down)})`);
  t.health = { rpc: { ok: 1000, fail: 1500, since: 1500 }, "rpc:robinhood-rpc.publicnode.com": { ok: 1000, fail: 1600, since: 1600 } };
  const s2 = await t.summary();
  ok(s2.status === "degraded" && s2.down.includes("rpc"), "every 4663 endpoint refusing: degraded");
}

// --- found loose by mutation testing ---------------------------------------------------

// An error object whose code is not a number is not a JSON-RPC error: no answer, try another.
ok(verdict({ error: { code: "x", message: "m" } }).transient && !verdict({ error: { code: "x", message: "m" } }).error, "an error with a non-numeric code is transient, not the chain refusing");
// A batch answered with one error carries that error's own words.
ok(/Invalid request/.test(batchVerdict({ jsonrpc: "2.0", id: null, error: { code: -32600, message: "Invalid request" } }).transient || ""), "a batch answered with a single error says which error");

// plainPost, the door check's transport, against a server answering each status.
{
  const answers = { "/429": [429, "slow down"], "/500": [500, "oops"], "/503": [503, "busy"], "/400": [400, '{"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"bad"}}'],
                    "/200": [200, '{"jsonrpc":"2.0","id":1,"result":"0x1"}'], "/html": [200, "<html>login</html>"] };
  const srv = http.createServer((q, r) => { const [s, b] = answers[q.url] || [404, ""]; r.writeHead(s); r.end(b); });
  await new Promise((res) => srv.listen(0, "127.0.0.1", res));
  const base = `http://127.0.0.1:${srv.address().port}`;
  const kind = async (path) => { try { return await plainPost(base + path, { id: 1 }, 2000); } catch (e) { return e instanceof Transient ? "Transient" : `other: ${e.message}`; } };
  ok((await kind("/429")) === "Transient", "429, rate limited: transient");
  ok((await kind("/500")) === "Transient" && (await kind("/503")) === "Transient", "500 and 503: transient");
  const four = await kind("/400");
  ok(four && four.error && four.error.message === "bad", "400 with a JSON-RPC body: the body, for verdict to read -- not transient");
  const two = await kind("/200");
  ok(two && two.result === "0x1", "200: the answer");
  ok(typeof (await kind("/html")) === "string" && verdict(await kind("/html")).transient, "a 200 that is not JSON: a string, which verdict calls no answer");
  srv.close();
  ok((await kind("/200")) === "Transient", "nobody listening: transient");
}

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

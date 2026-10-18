// Chain 4663's JSON-RPC endpoints, which one is used, and what an answer means.
//
// ### Why more than one
//
// Everything that pays a miner -- every balance read, every estimate, every
// broadcast, every receipt -- went to one URL, Robinhood's own public RPC, which
// its own documentation does not recommend for production and which rate-limits
// shared egress like Cloudflare's. When it refused, payouts stopped. So the
// treasury and the door check walk an ordered list, sticking to whichever
// endpoint last answered and going back to the first after `RETURN_MS`.
//
// Every default below was measured from outside before it went in
// (2026-10-18): chain id 4663, the GladosReader creation-code eth_call at 200
// addresses, a two-call batch, and eth_sendRawTransaction answering a bad
// transaction with a JSON-RPC error rather than a page. Measured and left out:
// robinhood.drpc.org (its free plan times the 200-address read out, HTTP 408),
// rpc.arrowrpc.com (530), rpc.nodeflare.app (403, an HTML page), and
// lb.routeme.sh (429 on the first request).
//
// ### Why switching is safe for a treasury
//
// A signed transaction carries its chain id and is the same bytes wherever it
// is sent, so broadcasting through a second node cannot pay twice. What can
// differ is how far behind a node is, and runner.js already assumes a lagging
// node: a nonce seen used with no receipt is only given up on after both
// `nonceGraceMs` and `nonceDepth` (it was written for one load-balanced RPC,
// which is the same problem). What would *not* be safe is a node on another
// chain answering balances, so every endpoint proves its chain id before it is
// used, and one that names another chain is never used again in this isolate.

// A read that did not get an answer. Retried; never counted as a refusal.
export class Transient extends Error {}
// A definite answer from a node: a revert, an invalid transaction.
export class RpcError extends Error {
  constructor(message, code) { super(message); this.code = code; }
}

export const CHAIN_ID_HEX = "0x1237"; // 4663

export const DEFAULT_RPCS = [
  "https://rpc.mainnet.chain.robinhood.com",
  "https://robinhood-rpc.publicnode.com",
  "https://rpc-robinhood.globalstake.io",
  "https://rpc.ordofi.network",
  "https://robinhood.api.pocket.network",
  "https://robinhood-mainnet-rpc.blockreq.com/v1/rpc/public",
];

// How long an endpoint other than the first is kept before the first is tried
// again. Long enough that a primary that is down costs one failed attempt per
// period rather than one per call.
export const RETURN_MS = 10 * 60_000;

// `RPC_URLS`, whitespace-separated, replaces the list (a keyed provider goes
// first there). Only https, and no duplicates: a URL listed twice is one
// endpoint tried twice in a row, which is not a fallback.
export function rpcList(env) {
  const given = String((env && env.RPC_URLS) || "").split(/\s+/).filter((u) => /^https:\/\//.test(u));
  return [...new Set(given.length ? given : DEFAULT_RPCS)];
}

export const hostOf = (url) => { try { return new URL(url).host; } catch { return String(url); } };

// What a node says when it could not answer rather than when it refused. A
// message is matched as well as a code because providers disagree about codes:
// the same "invalid transaction" came back as -32000, -32600 and -32602 from
// three of the endpoints above.
const TRANSIENT_MSG = /rate|limit|too many|timeout|timed out|temporar|busy|unavailable|header not found|try again|overload/i;
// -32005 limit exceeded, -32029 a public rate limit (routeme), -32601 a method
// this endpoint does not serve (another endpoint may), -32603 internal error --
// except when it carries a revert, which some gateways wrap in -32603 and which
// is the chain's answer, not the node's trouble.
const TRANSIENT_CODES = new Set([-32005, -32029, -32601, -32603]);

// **What one JSON-RPC response means**: `{result}`, `{error, code}` for a
// definite refusal, or `{transient}` for no answer.
//
// The rule that matters is the last line. Only a JSON-RPC envelope with a
// numeric error code is a node's verdict. An HTML page, a bare string, a
// `{message}` from a gateway -- a 403 from a WAF, a 400 from a proxy, a 408 --
// is the transport refusing, and it used to be thrown as an RpcError: so an
// estimate refused by a firewall counted toward the three failures that pause
// the treasury, and a broadcast refused by one was recorded as the chain
// rejecting the payout.
export function verdict(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) return { transient: `not a JSON-RPC answer: ${String(JSON.stringify(body)).slice(0, 120)}` };
  const e = body.error;
  if (e === undefined || e === null) {
    if ("result" in body) return { result: body.result };
    return { transient: `no result: ${String(JSON.stringify(body)).slice(0, 120)}` };
  }
  if (typeof e === "object" && typeof e.code === "number") {
    const msg = String(e.message || JSON.stringify(e));
    if ((TRANSIENT_CODES.has(e.code) && !/revert/i.test(msg)) || TRANSIENT_MSG.test(msg)) return { transient: msg };
    return { error: msg, code: e.code };
  }
  return { transient: `not a JSON-RPC error: ${String(JSON.stringify(e)).slice(0, 120)}` };
}

// A batch answers an array; anything else is the whole batch unanswered.
export function batchVerdict(body) {
  if (Array.isArray(body)) return { result: body };
  const v = verdict(body);
  return { transient: v.transient || v.error || "batch answered with a single value" };
}

// The ordered list and where in it we are.
export class Endpoints {
  constructor(urls, now = () => Date.now()) {
    this.urls = urls;
    this.now = now;
    this.at = 0;
    this.since = 0;
    this.chain = new Map(); // url -> true (4663) | false (another chain)
  }

  get size() { return this.urls.length; }

  // The endpoint to use now: the one that last answered, unless it has been a
  // fallback for RETURN_MS, and never one that named another chain.
  current() {
    if (this.at !== 0 && this.now() - this.since >= RETURN_MS) this.at = 0;
    for (let k = 0; k < this.urls.length; k++) {
      const i = (this.at + k) % this.urls.length;
      if (this.chain.get(this.urls[i]) !== false) {
        if (i !== this.at) { this.at = i; this.since = this.now(); }
        return this.urls[i];
      }
    }
    return null;
  }

  // `url` did not answer: the next call goes to the next endpoint. A no-op
  // when something else already moved us off it.
  miss(url) {
    if (this.urls[this.at] !== url || this.urls.length < 2) return;
    this.at = (this.at + 1) % this.urls.length;
    this.since = this.now();
  }

  // What /treasury shows: never a key, since a keyed URL carries one.
  describe() {
    return { using: hostOf(this.urls[this.at] || ""), of: this.urls.map(hostOf),
             wrongChain: [...this.chain].filter(([, ok]) => !ok).map(([u]) => hostOf(u)) };
  }
}

// The one place a request is made. `post(url, payload)` answers the parsed
// body or throws Transient; the caller decides whether a definite error is a
// failure. A first use proves the chain id; anything that does not answer moves
// the endpoint along and is thrown as Transient for the caller's retry.
export async function attempt(eps, payload, post, label = "rpc") {
  const url = eps.current();
  if (!url) throw new Transient(`${label}: no endpoint on chain 4663 is left`);
  try {
    if (!eps.chain.has(url)) {
      const c = verdict(await post(url, { jsonrpc: "2.0", id: 1, method: "eth_chainId", params: [] }));
      if (c.transient) throw new Transient(`${hostOf(url)} did not say which chain it is (${c.transient})`);
      const ok = c.error === undefined && String(c.result).toLowerCase() === CHAIN_ID_HEX;
      eps.chain.set(url, ok);
      if (!ok) throw new Transient(`${hostOf(url)} is not chain 4663 (${c.error || c.result}); it will not be used`);
    }
    const v = (Array.isArray(payload) ? batchVerdict : verdict)(await post(url, payload));
    if (v.transient) throw new Transient(`${label} at ${hostOf(url)}: ${v.transient}`);
    return v;
  } catch (e) {
    if (e instanceof Transient) eps.miss(url);
    throw e;
  }
}

// A plain transport for callers with no health bookkeeping of their own (the
// door check): no answer, a 429 or a 5xx is Transient; any other body is parsed,
// and a page that is not JSON comes back as a string for `verdict` to refuse.
export async function plainPost(url, payload, timeoutMs = 10_000) {
  let r;
  try {
    r = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload),
                           signal: AbortSignal.timeout(timeoutMs) });
  } catch (e) {
    throw new Transient(`${hostOf(url)} unreachable: ${e && e.message ? e.message : e}`);
  }
  if (r.status === 429 || r.status >= 500) throw new Transient(`${hostOf(url)} answered ${r.status}`);
  const t = await r.text().catch(() => "");
  try { return JSON.parse(t); } catch { return `HTTP ${r.status}: ${t.slice(0, 120)}`; }
}

// What a failure of the pool itself is, and how it is answered. Its own file
// so it can be tested: worker.js imports `cloudflare:sockets` and a compiled
// wasm module, neither of which loads under node. The argument for wrapping
// the router at all is at the top of worker.js's default export.

// The allowance is recognised by its wording ("Exceeded allowed duration in
// Durable Objects free tier" and its siblings), deliberately narrowly: a stray
// "exceeded" in an unrelated error -- "Maximum call stack size exceeded" --
// must not hold the pool shut for worker.js's BREAK_MS.
export function classifyFailure(e) {
  const msg = String((e && e.message) || e);
  if (/exceeded allowed|free tier|daily limit|quota/i.test(msg)) {
    return { quota: true, retry: 300, why: "the pool is over its daily Cloudflare allowance and is refusing work until it resets at 00:00 UTC" };
  }
  if (e && e.overloaded) return { retry: 30, why: "the pool is overloaded; back off and retry" };
  // chunks.js's refusal to load a partial share log, raised in the object's
  // constructor on purpose: serving from an empty record would overwrite it.
  // The core's own refusal -- a log whole in storage but unreadable or failing
  // its digest -- is the same situation and gets the same answer.
  if (/missing a chunk|refusing to read a partial|edge_load_ledger/i.test(msg)) {
    return { retry: 300, why: "the pool's stored share log did not load, so it will not serve until the operator restores it" };
  }
  return { retry: 30, why: "the pool hit an internal error; it has been logged" };
}

// The 503. Never cached, readable cross-origin like every other public route,
// and carrying the reason as JSON so the site's /pool/ page can show it.
export function unavailable(why, retry) {
  return new Response(JSON.stringify({ error: "pool unavailable", why, retryAfter: retry }) + "\n", {
    status: 503,
    headers: { "content-type": "application/json", "retry-after": String(retry), "cache-control": "no-store",
               "access-control-allow-origin": "*" },
  });
}

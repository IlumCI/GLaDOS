// Maintenance: the operator says the pool is down on purpose, and every part
// of it stops the same way. The rules are here, with no imports, so the router,
// the mining shards and the Pearl shard share one copy and node can test it.
//
// Declared with one authenticated request (tools: pool/edge/maintenance.mjs):
//
//   POST /maintenance   Authorization: Bearer <MAINT_TOKEN>
//   {"on": true, "reason": "moving the treasury", "minutes": 30, "drainSeconds": 20}
//   {"on": false}
//
// and read by anyone at GET /maintenance. What "on" does, everywhere at once:
//   - every connected miner is told why and for how long (glados.maintenance)
//   - for drainSeconds the pool keeps crediting shares already being found, then
//     saves the record once and closes each socket with 1013, "try again later"
//   - shares owed upstream are still delivered (Pearl holds its upstream open
//     until its queue is empty)
//   - new miners are refused with 503, the reason, and a Retry-After
//   - the treasury takes no step: a payout is a sequence of steps built to stop
//     between any two, so pausing it is not ticking it
// Reads -- /status, /ledger.json, /payouts.json, /health.json -- keep working.
//
// "minutes" is a promise to people, not a timer: the pool stays down until it
// is declared up, and a miner told "back by 14:30" past 14:30 keeps waiting.

export const MAINT_KEY = "maintenance";
export const DRAIN_MS = 20_000;
export const DRAIN_MAX_MS = 120_000;
export const REASON_MAX = 200;
export const MINUTES_MAX = 7 * 24 * 60;

/** A request body as a state to store, or why it is not one. */
export function parseMaint(body, now) {
  if (!body || typeof body !== "object" || Array.isArray(body)) return { error: "expected a JSON object" };
  if (typeof body.on !== "boolean") return { error: '"on" must be true or false' };
  if (!body.on) return { state: { on: false, since: now } };
  // Shown on every miner's screen and in a WebSocket close frame: one line, printable.
  const reason = typeof body.reason === "string" ? [...body.reason.replace(/[\u0000-\u001f\u007f-\u009f\u2028\u2029]+/g, " ").replace(/\s+/g, " ").trim()].slice(0, REASON_MAX).join("").trim() : "";
  if (!reason) return { error: 'say why: "reason" is shown to every miner' };
  let until = null;
  if (body.minutes !== undefined && body.minutes !== null) {
    const m = Number(body.minutes);
    if (typeof body.minutes !== "number" || !Number.isFinite(m) || m <= 0 || m > MINUTES_MAX) return { error: `"minutes" is a number from 1 to ${MINUTES_MAX}` };
    until = now + Math.round(m * 60_000);
  }
  let drainMs = DRAIN_MS;
  if (body.drainSeconds !== undefined && body.drainSeconds !== null) {
    const d = Number(body.drainSeconds);
    if (typeof body.drainSeconds !== "number" || !Number.isFinite(d) || d < 0 || d * 1000 > DRAIN_MAX_MS) return { error: `"drainSeconds" is a number from 0 to ${DRAIN_MAX_MS / 1000}` };
    drainMs = Math.round(d * 1000);
  }
  return { state: { on: true, reason, since: now, until, drainMs } };
}

/** A stored state, or null when what storage holds is not one. */
export function validMaint(s) {
  if (!s || typeof s !== "object" || typeof s.on !== "boolean") return null;
  if (!s.on) return { on: false, since: Number(s.since) || 0 };
  if (typeof s.reason !== "string" || !s.reason) return null;
  const until = Number.isFinite(s.until) ? s.until : null;
  const drainMs = Number.isFinite(s.drainMs) ? Math.min(Math.max(0, s.drainMs), DRAIN_MAX_MS) : DRAIN_MS;
  return { on: true, reason: [...s.reason].slice(0, REASON_MAX).join(""), since: Number(s.since) || 0, until, drainMs };
}

/** Seconds a refused miner should wait before trying again. */
export function retryAfter(m, now) {
  if (m && m.until && m.until > now) return Math.min(3600, Math.max(30, Math.ceil((m.until - now) / 1000)));
  return 120;
}

/** What a connected miner is told, as one protocol line. */
export function notice(m, now) {
  return JSON.stringify({ method: "glados.maintenance", params: { reason: m.reason, until: m.until, retryAfter: retryAfter(m, now), drainMs: m.drainMs } }) + "\n";
}

/** The answer to a miner arriving while the pool is down. */
export function refusal(m, now) {
  const ra = retryAfter(m, now);
  return new Response(JSON.stringify({ error: "maintenance", why: `the pool is down for maintenance: ${m.reason}`, reason: m.reason, until: m.until, retryAfter: ra }) + "\n", {
    status: 503,
    headers: { "content-type": "application/json", "retry-after": String(ra), "cache-control": "no-store", "access-control-allow-origin": "*" },
  });
}

/** A close frame's reason holds 123 bytes of UTF-8; cut on a character, never through one. */
export function closeReason(m) {
  const enc = new TextEncoder();
  // By code point: slicing UTF-16 units can leave half an emoji, which encodes
  // as a replacement character.
  const cps = [...`maintenance: ${m.reason}`];
  while (enc.encode(cps.join("")).length > 123) cps.pop();
  return cps.join("");
}

/** `Authorization: Bearer <token>` against the secret, in time that does not depend on where they differ. */
export function tokenOk(header, secret) {
  if (!secret || typeof secret !== "string" || secret.length < 16) return false;
  const got = /^Bearer (.+)$/.exec(String(header || ""));
  if (!got) return false;
  const a = new TextEncoder().encode(got[1]), b = new TextEncoder().encode(secret);
  let diff = a.length ^ b.length;
  for (let i = 0; i < b.length; i++) diff |= (a[i] ?? 0) ^ b[i];
  return diff === 0;
}

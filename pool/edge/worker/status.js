// The public status page's data: each component's state now, its history by
// day, the incidents, and the payout pot. The pure parts live here so
// status.test.mjs checks them under node; worker.js runs the probes in a
// Durable Object of its own ("status"), not in a mining shard, so the page
// stays up exactly when a shard does not.

export const COMPONENTS = ["cpu", "gpu", "payouts"];
export const NAMES = { cpu: "CPU mining", gpu: "GPU mining", payouts: "Payouts" };
const RANK = { operational: 0, degraded: 1, outage: 2 };
export const HISTORY_DAYS = 90;
export const INCIDENTS_KEPT = 50;

// The whole pool is as healthy as its least healthy component.
export function rollup(states) {
  let worst = "operational";
  for (const s of states) if ((RANK[s] ?? 0) > RANK[worst]) worst = s;
  return worst;
}

const dayOf = (ms) => new Date(ms).toISOString().slice(0, 10);

// One sample into a component's daily buckets: [operational, degraded, outage]
// counts per UTC day, the last HISTORY_DAYS kept.
export function recordSample(history, comp, state, now) {
  const h = history[comp] || (history[comp] = {});
  const d = dayOf(now);
  const b = h[d] || (h[d] = [0, 0, 0]);
  b[RANK[state] ?? 0] += 1;
  const days = Object.keys(h).sort();
  for (const old of days.slice(0, Math.max(0, days.length - HISTORY_DAYS))) delete h[old];
  return history;
}

// A day's uptime as the share of samples not in outage, and its worst state.
export function dayView(bucket) {
  const [ok, deg, out] = bucket || [0, 0, 0];
  const n = ok + deg + out;
  if (!n) return { uptime: null, state: "nodata" };
  return { uptime: (ok + deg) / n, state: out ? "outage" : deg ? "degraded" : "operational" };
}

// The last `days` days for one component, oldest first, gaps as "nodata".
export function series(history, comp, now, days = HISTORY_DAYS) {
  const h = history[comp] || {};
  const out = [];
  for (let i = days - 1; i >= 0; i--) {
    const d = dayOf(now - i * 86_400_000);
    out.push({ day: d, ...dayView(h[d]) });
  }
  return out;
}

// Open an incident when a component leaves "operational", update it while its
// state or reason changes, and resolve it when the component is back.
export function trackIncident(incidents, comp, prev, state, why, now) {
  const open = incidents.find((i) => i.auto && i.component === comp && !i.resolvedAt);
  if (state === "operational") {
    if (open) {
      open.resolvedAt = now;
      open.updates.push({ at: now, status: "resolved", text: `${NAMES[comp]} is operating normally again.` });
    }
    return incidents;
  }
  if (!open) {
    incidents.unshift({
      id: `${comp}-${now}`, auto: true, component: comp, severity: state,
      title: `${NAMES[comp]} ${state === "outage" ? "unavailable" : "degraded"}`,
      startedAt: now, resolvedAt: null,
      updates: [{ at: now, status: "investigating", text: why || "The component stopped answering its health check." }],
    });
  } else if (open.severity !== state || (why && open.updates[open.updates.length - 1].text !== why)) {
    if (RANK[state] > RANK[open.severity]) open.severity = state;
    open.updates.push({ at: now, status: "update", text: why || `Now ${state}.` });
  }
  incidents.splice(INCIDENTS_KEPT);
  return incidents;
}

// When the RVN alone fills the swap threshold, in days at the current rate;
// null when nothing is coming in.
export function payoutEta({ rvnHeld, rvnUnpaid, threshold, rvnPerDay }) {
  const have = (Number(rvnHeld) || 0) + (Number(rvnUnpaid) || 0);
  const need = Math.max(0, (Number(threshold) || 0) - have);
  if (need === 0) return { days: 0, have, need };
  const rate = Number(rvnPerDay) || 0;
  return { days: rate > 0 ? need / rate : null, have, need };
}

// A component's state from a probe: what answered, how fast, and what it said.
export function judge({ ok, ms, body, why }) {
  if (!ok) return { state: "outage", why: why || "did not answer" };
  if (body && Array.isArray(body.down) && body.down.length) return { state: "degraded", why: body.down.join("; ") };
  if (ms > 8000) return { state: "degraded", why: `answering slowly (${Math.round(ms / 1000)} s)` };
  return { state: "operational", why: "" };
}

// The payout component, from the treasury's own summary.
export function judgeTreasury({ ok, body, why }) {
  if (!ok) return { state: "outage", why: why || "the treasury did not answer" };
  if (body && body.paused) return { state: "outage", why: `paused: ${body.paused}` };
  if (body && body.status && body.status !== "ok") {
    const down = Array.isArray(body.down) && body.down.length ? ` (${body.down.join(", ")} unreachable)` : "";
    return { state: "degraded", why: `the treasury reports ${body.status}${down}` };
  }
  return { state: "operational", why: "" };
}

// The written notes that are not raised by a probe: an operator's account of
// an incident. Shown with the automatic ones, newest first. Dated by day only
// (dateOnly): the page shows no clock time it was not given.
export const NOTES = [
  {
    id: "2026-10-18-outage", auto: false, dateOnly: true, component: "cpu", severity: "outage",
    title: "CPU mining and payouts unavailable",
    startedAt: Date.parse("2026-10-18T00:00:00Z"), resolvedAt: null,
    updates: [
      { at: Date.parse("2026-10-18T00:00:00Z"), status: "identified",
        text: "The pool's main server object ran out of its daily allowance on Cloudflare's free plan, then, once the plan was upgraded, kept exceeding its per-call CPU limit and being reset. CPU mining, the treasury and the payout list were unreachable; GPU mining kept working. Saved progress was not affected." },
      { at: Date.parse("2026-10-18T00:00:00Z"), status: "monitoring",
        text: "The plan is upgraded, the CPU limit raised, and the payout code now falls back across six Robinhood Chain RPC endpoints instead of one. We are watching CPU mining come back." },
    ],
  },
];

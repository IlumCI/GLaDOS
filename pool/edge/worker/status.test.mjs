// node pool/edge/worker/status.test.mjs
import { rollup, recordSample, dayView, series, trackIncident, payoutEta, judge, judgeTreasury, HISTORY_DAYS } from "./status.js";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };
const D = 86_400_000, T = Date.parse("2026-10-18T10:00:00Z");

ok(rollup(["operational", "operational"]) === "operational", "all up is operational");
ok(rollup(["operational", "degraded"]) === "degraded", "one degraded makes the pool degraded");
ok(rollup(["degraded", "outage", "operational"]) === "outage", "the worst component wins");

const h = {};
recordSample(h, "cpu", "operational", T);
recordSample(h, "cpu", "outage", T + 60_000);
recordSample(h, "cpu", "degraded", T + 120_000);
ok(JSON.stringify(h.cpu["2026-10-18"]) === "[1,1,1]", "samples count into the day's [up, degraded, outage]");
ok(dayView(h.cpu["2026-10-18"]).state === "outage", "a day with any outage shows as an outage");
ok(Math.abs(dayView(h.cpu["2026-10-18"]).uptime - 2 / 3) < 1e-9, "uptime counts degraded as up, outage as down");
ok(dayView(undefined).state === "nodata", "a day never sampled is no data, not 100%");

const long = {};
for (let i = 0; i < HISTORY_DAYS + 10; i++) recordSample(long, "gpu", "operational", T - i * D);
ok(Object.keys(long.gpu).length === HISTORY_DAYS, "only the last HISTORY_DAYS days are kept");
const s = series(long, "gpu", T, 7);
ok(s.length === 7 && s[6].day === "2026-10-18" && s[0].day === "2026-10-12", "a series runs oldest first, ending today");

let inc = [];
trackIncident(inc, "cpu", "operational", "outage", "main: overloaded", T);
ok(inc.length === 1 && inc[0].title === "CPU mining unavailable" && !inc[0].resolvedAt, "leaving operational opens an incident");
trackIncident(inc, "cpu", "outage", "outage", "main: overloaded", T + 60_000);
ok(inc[0].updates.length === 1, "the same state and reason adds no update");
trackIncident(inc, "cpu", "outage", "degraded", "answering slowly (9 s)", T + 120_000);
ok(inc[0].updates.length === 2 && inc[0].severity === "outage", "a new reason is an update; severity keeps the worst seen");
trackIncident(inc, "cpu", "degraded", "operational", "", T + 180_000);
ok(inc[0].resolvedAt === T + 180_000 && inc[0].updates.at(-1).status === "resolved", "back to operational resolves it");
trackIncident(inc, "cpu", "operational", "operational", "", T + 240_000);
ok(inc.length === 1, "staying operational opens nothing");
trackIncident(inc, "gpu", "operational", "degraded", "", T + 300_000);
ok(inc.length === 2 && inc[0].component === "gpu", "another component gets its own incident, newest first");

const e1 = payoutEta({ rvnHeld: 4, rvnUnpaid: 9.5, threshold: 300.5, rvnPerDay: 9.6 });
ok(Math.abs(e1.days - (300.5 - 13.5) / 9.6) < 1e-9 && e1.need === 287, "ETA is what is still needed over the daily rate");
ok(payoutEta({ rvnHeld: 400, rvnUnpaid: 0, threshold: 300, rvnPerDay: 9 }).days === 0, "already over the threshold is zero days");
ok(payoutEta({ rvnHeld: 1, rvnUnpaid: 0, threshold: 300, rvnPerDay: 0 }).days === null, "nothing coming in has no ETA, not infinity");

ok(judge({ ok: false, why: "overloaded" }).state === "outage", "no answer is an outage, with the reason");
ok(judge({ ok: true, ms: 200, body: { connections: 3 } }).state === "operational", "a quick answer is operational");
ok(judge({ ok: true, ms: 9000, body: {} }).state === "degraded", "a slow answer is degraded");
ok(judge({ ok: true, ms: 100, body: { down: ["shard-1: x"] } }).state === "degraded", "a shard reported down is degraded");
ok(judgeTreasury({ ok: true, body: { status: "ok" } }).state === "operational", "a healthy treasury is operational");
ok(judgeTreasury({ ok: true, body: { status: "degraded", down: ["rpc"] } }).why.includes("rpc"), "a degraded treasury names what is down");
ok(judgeTreasury({ ok: true, body: { status: "ok", paused: "three failures" } }).state === "outage", "a paused treasury is an outage");

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

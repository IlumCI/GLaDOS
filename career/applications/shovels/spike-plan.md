# Phase 3 spike — query plan (written before spending a single call)

Budget: **250 API calls, errors included.** This plan spends well under 100 and
leaves the rest as margin, because the budget is part of what they grade. Every
call is logged by `spike.py` with a running counter; the harness hard-stops at a
configurable ceiling so a loop bug cannot burn the key.

Base URL `https://api.shovels.ai/v2`, auth header `X-API-Key`. All endpoints are
GET. The monthly-metrics endpoints are the reason this is cheap: one call
returns a whole geo's month-by-month time series (`permit_count`,
`avg_approval_duration`, `avg_inspection_pass_rate`, `total_job_value`),
filterable by `tag` and `property_type`. Aggregates over record dumps, always.

## The finding we are hunting

A **government decision that shows up downstream in the permit data**, which is
Shovels' own thesis ("we see it before it's built") turned into a checkable
claim. Concretely: find a city or jurisdiction where a policy event (a
moratorium, a rezoning wave, a fee change, a solar/EV mandate) is followed by a
measurable move in the matching permit tag's monthly series, and that the move
is **not** an artifact of coverage lapsing.

Fallback finding if no clean policy link appears: an **approval-time divergence**
between two comparable cities for the same tag (e.g. rooftop-solar permits clear
in X days in one city and 3X in a neighbour), which is non-obvious and
policy-relevant on its own.

## Tiers

**Tier 0 — orient (3 calls).**
1. `/meta/release` — data freshness date, so the window we pick ends where the
   data actually ends.
2. `/list/tags` — confirm the real tag strings (solar, ev_charger, etc.) rather
   than guessing names that would waste calls on empty filters.
3. `/meta/coverage?geo_type=state&geo_id=<ST>` for one candidate state — read
   `fill_pct` per field before trusting anything from that state.

**Tier 1 — locate (5-10 calls).**
4. `/cities/search?q=...` to resolve `geo_id` for a small set (3-4) of
   comparable cities in a well-covered state.
5. `/meta/coverage` per candidate city over the analysis window. **Any city
   whose coverage is patchy across the window is dropped now**, before it can
   masquerade as a finding.

**Tier 2 — the signal (10-20 calls).**
6. `/cities/{geo_id}/metrics/monthly?tag=<tag>&metric_from=&metric_to=` for each
   surviving city — the time series that carries the effect.
7. `/decisions/search?geo_id=&decision_from=&decision_to=&category=` (or
   `decision_q=`) to find the policy event that lines up with a move in the
   series. Decisions are the "before it's built" half; permits are the "after".

**Tier 3 — kill the artifact (10-20 calls).**
8. Re-pull `/meta/coverage` for the exact geo and window of the finding: prove
   `fill_pct` is flat across the drop/spike, so the move is real activity and not
   the jurisdiction going dark.
9. Control tag: pull the monthly series for an unrelated tag in the same city.
   If every tag moves together, it is a reporting artifact, not a solar story.
10. Control geo: the same tag in a neighbouring, comparably-covered city that
    did **not** see the decision. A real policy effect is local; an artifact is
    not.

"We could not verify this" is the only wrong answer, so if Tier 3 dissolves the
finding, the honest write-up is that the apparent effect was a coverage artifact
and here is the coverage series that proves it — that is itself a good result and
exactly the discipline they screen for.

## What gets saved

- `spike-transcript.jsonl` — every call: endpoint, params, status, call number.
- `spike-finding.md` — the claim, the queries that found it, the three artifact
  controls, and the final call count.
Both are produced by the harness as it runs, not reconstructed afterwards.

# Phase 3 spike — finding

**Calls spent: 22 (of 250). Successful: 18. Errors: 4, all mine, all mid-plan validation mistakes, each one counted against the budget below.**
Data release: 2026-08-20. Transcript: `spike-transcript.jsonl` (every call, in order).

## The finding

**California's NEM 3.0 net-metering change is legible in San Diego's residential
solar permit stream as a pull-forward spike and a lasting collapse, and it is
solar-specific.**

NEM 3.0 took effect **15 April 2023** and cut the value of new rooftop solar
exports. San Diego residential **solar** permits, monthly:

| Period | solar permits/mo | vs 2022 baseline |
| --- | --- | --- |
| 2022 (baseline) | ~1,900-2,200 | 1.0x |
| Feb-Apr 2023 (rush to beat the deadline) | 2,911 -> **3,623 peak (Mar)** -> 3,374 | ~1.7x |
| Sep 2023 onward (collapse) | 774 -> 748 -> 590 -> 453 | ~0.25x |
| Full-year 2024 | ~375-560 | ~0.25x, sustained |

The market did not recover across all of 2024: eighteen months after the
change, San Diego was permitting residential solar at roughly a **quarter** of
its pre-NEM-3.0 rate, having briefly run at nearly double it in the rush.

## Why it is not a data artifact (three controls)

1. **Control tag.** San Diego **roofing** permits over the same window stay flat
   (single digits to ~20/mo) with no cliff. If San Diego had stopped reporting,
   every tag would drop together; only solar moves.
2. **Out-of-state control.** **Austin, TX** solar permits are flat (~100-160/mo)
   across 2022-2024 with no April-2023 inflection. The cliff is a California
   policy effect, not a Shovels-wide change in how solar is tagged or ingested.
3. **Coverage control.** San Diego's total permit volume **rose** across the
   drop -- 92,590 permits in 2023 H1 vs 168,687 in 2023 H2-2024 H1 (`/meta/coverage`).
   The city was reporting more, not less, so the solar collapse is real activity.

## Honest limitations

- **Bakersfield, CA** showed the same 2023 hump but its late-2024 numbers crater
  in *both* solar and roofing (63/35/46 solar, 19/13/23 roofing) -- the classic
  both-tags-together signature of a **reporting lapse**, so its 2024 tail is
  unreliable and it was dropped as the headline case. Noticing that is the
  reason San Diego is the example and Bakersfield is a footnote.
- Permit *counts* are the reliable field here; `job_value` coverage is thin
  (<1% fill at state level), so no dollar claims are made.
- Permit date is filing/issue date, which lags the install decision; the Sep
  2023 cliff is consistent with applications drying up after the April deadline
  once the pipeline cleared.

## Queries (all in the transcript)

`/meta/release`; `/list/tags`; `/meta/coverage` for CA/TX/FL/NC and for San
Diego in two windows; `/cities/search` for San Diego, Bakersfield, Austin;
`/cities/{geo_id}/metrics/monthly?tag=solar|roofing&property_type=residential`
for each. One 404 (Phoenix resolved to Phoenix, IL, not AZ) and three early
422s (`size=200` over the 100 cap; `property_type` is required though the
spec marks it optional) -- all counted, none retried blindly.

# Baselines: the resident checkpoints, measured

Status: first published baselines, taken before any capability work (E/F
tracks). The rule this project works under -- measure before building, spend
validation freely, read the test slice once -- needs numbers to protect, and
these are they. Re-run with `tools/lm_eval.py`; every figure here is
reproducible from the converted checkpoints in `out/` plus the cached
datasets.

## Numbers

| Rail | SmolLM2-135M (dense) | Qwen3.5-0.8B (hybrid) | Chance |
|---|---|---|---|
| MMLU, 0-shot letter-logprob | 20.0% (n=50) | **30.0%** (n=30) | 25% |
| GSM8K, 5-shot greedy ≤64 new | 0.0% (n=15) | 0.0% (n=8) | ~0 |
| NIAH, 512/1024[/2048], 3 depths | **0/7** | **6/6** (512/1024) | -- |
| Route, traces test split, constrained decode, 78 actions | 0.0% (n=50) | **33.3%** (n=30) | ~1.3% |

Host-side, NumPy, int8 checkpoints dequantised block-wise exactly as the
kernel does. Timing reference: SmolLM2 ~4-8 s/question, q35 ~20-40 s/question
on the development machine.

## What the numbers say

**The route rail is the one that decides the agent loop.** The design doc's
threshold -- a loop is only worth building against a model that can follow an
instruction -- now has its measured answer. SmolLM2 scores 0/50 at 5.7%-class
performance (the README's earlier figure; 0 hits in 50 draws is an unlucky
but honest sample of it). q35 scores 33.3% against a 1.3% chance floor, and
its misses are near misses between related commands (`nvme`/`mem`,
`zeroshot`/`mem`). Constrained decoding makes wrong applets unreachable;
33% means the loop's steps are productive a third of the time before any
prompt engineering, and the three-core gate exists precisely to know which
third.

**NIAH is the linear-attention payoff, visible already at 1k.** The hybrid
recalled the needle at every depth and both contexts; the dense 135M recalled
nothing at any. This is one prompt each -- an existence proof, not a curve --
but the direction is unambiguous, and the 8k-32k story that actually justifies
the architecture is a GF63 measurement (host NumPy attention is quadratic;
2k is where this rail honestly stops).

**MMLU sits near chance for both.** 0-shot letter-logprob is the cheap
tracking trick, not the official harness; a 135M model below chance and an
0.8B a little above it is the expected picture. This rail exists to catch
regressions from quantisation or format changes, not to quote.

**GSM8K is 0.0 for both and that is the honest reading.** Small models do not
do multi-digit arithmetic through greedy 64-token chains. The rail stays --
the E-track's tiered-cognition work (tools, interpreter, closed-form heads)
is exactly the alternative to pretending the backbone will improve.

## Method, and what would falsify these

- Encoding uses the reference HF tokenizers; detokenisation uses the
  converted kernel tokenizers. A mismatch would show up as garbled
  generations, which NIAH's substring check would catch.
- The hybrid runner is incremental (per-token state, like the kernel) and is
  proven against `ref35.forward` at every position before use (`--check`,
  worst rel 2.0e-06 over 24 positions). Two real bugs were caught by that
  gate on the way in: an omitted `input_layernorm` and a missing position
  increment -- each produced fluent, plausible, wrong logits, and neither
  produced an error. The fixture discipline paid for itself twice in one
  afternoon.
- Sample sizes are small and stated next to every figure. These are
  baselines for tracking deltas, not leaderboard quotes; the official-harness
  versions of MMLU/GSM8K would use different prompts and more items.
- `route` scores constrained decode over the traces corpus's own action
  space (shell commands + applets, 78 names), prompts cut at `</think>` so
  the decode is the choice rather than a continuation of the answer.

## Known staleness found on the way

`tools/evaluate.py` has not kept up with `reference.py` (load arity, rmsnorm
signature, the quantised-tuple weight format) and does not currently run.
Its methodology lives on in `lm_eval.py`, which is self-contained for the
dense step. Repairing or retiring evaluate.py is deliberately not done in
this change.

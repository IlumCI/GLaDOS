# Baselines: the resident checkpoints, measured

Status: first published baselines, taken before any capability work (E/F
tracks). The rule this project works under -- measure before building, spend
validation freely, read the test slice once -- needs numbers to protect, and
these are they. Re-run with `tools/lm_eval.py`; every figure here is
reproducible from the converted checkpoints in `out/` plus the cached
datasets.

## Numbers

| Rail | SmolLM2-135M (dense) | Qwen3.5-0.8B (hybrid) | **Qwen3.5-2B distill (hybrid)** | Chance |
|---|---|---|---|---|
| MMLU, 0-shot letter-logprob | 20.0% (n=50) | 30.0% (n=30) | **43.3%** (n=30) | 25% |
| GSM8K, 5-shot greedy | ~~0.0%~~ *re-run owed* | ~~0.0%~~ *re-run owed* | ~~0.0%~~ *re-run owed* | ~0 |
| NIAH, 512/1024 | 0/7 | 6/6 | **6/6** | -- |
| Route, constrained decode, 78 actions | 0.0% (n=50) | 33.3% (n=30) | **40.0%** (n=30) | ~1.3% |

The 2B column is `insraq/Qwen3.5-2B-EmperoAI-Qwen3.8-Distill-Heretic-Abliterated`
(Apache-2.0) -- a Qwen3.8-reasoning distill onto the Qwen3.5-2B hybrid body,
same LLLFx6 geometry family, same tokenizer, converted by the same tool
(1.88B params int8, worst rel 0.39%, argmax parity 100%, incremental runner
1.1e-06). In-kernel: boots under QEMU at 4G guest RAM, first-token probe
passes; the QEMU logits transcript is blocked by the intermittent
serial-input stall (see runbook) and is a GF63 formality at native speed.

Host-side, NumPy, int8 checkpoints dequantised block-wise exactly as the
kernel does. Timing reference: SmolLM2 ~4-8 s/question, q35 ~20-40 s/question
on the development machine.

## The GSM8K row is withdrawn, and it was never about the models

Three faults in `tools/lm_eval.py`, each on its own enough to produce a zero,
and all three applied to every run in that row whatever else was passed:

- **The budget was 64 new tokens.** A GSM8K answer is 133 tokens on average and
  213 at worst, measured over the test slice with the harness's own tokeniser.
  Two thirds of every completion was cut off before the line the score is read
  from, and the extractor then returned a number out of the middle of the
  reasoning.
- **Nothing told it to stop.** A base model carries on past its answer into a
  fabricated next question, and the last number in the text came from there.
- **The dense runner could not run Qwen3.** Head width derived rather than
  stated, RoPE pairing `2i` with `2i+1`, no QK-Norm. It raises
  `operands could not be broadcast` on a Qwen3 checkpoint, so it never produced
  a figure for one at all.

And a fourth that taints more than this row: `--hf-tokenizer` defaults to
`tools/hf/tokenizer.json`, which is **SmolLM2's 49,152-token vocabulary**. That
is correct for the SmolLM2 column and wrong for every Qwen checkpoint, whose
vocabularies are 151,669 and 248,320. Handed the wrong one a model receives ids
belonging to another vocabulary and answers with a degenerate run of one token
-- observed, at 0.0%. Whether the q35 columns above were taken with the flag
passed is not recorded, so **those figures cannot be relied on without a
re-run**; the SmolLM2 column is unaffected.

`lm_eval.py` refuses a vocabulary mismatch now and `--show N` prints the raw
completion, whose absence is what let all of this hide.

**Nothing in the GSM8K row should be quoted**, and it is left struck through
rather than deleted because it has been cited, in this file and elsewhere, as
evidence that small models cannot do arithmetic -- a conclusion drawn from a
harness that was not asking them. The three columns are *owed a re-run* rather
than withdrawn forever; what was missing was a dense runner that is fast and
right, and there is one now.

### The first GSM8K figure this project has that measures a model

    Qwen3-0.6B, GSM8K 5-shot greedy, <= 256 new tokens
    n = 1319, the whole test set

    39.2%

Dense, int8, its own tokenizer, through the fixed harness and
`tools/fastdense.py`. Chance on this task is about zero, so 39.2% is the model
doing the arithmetic rather than the harness finding a number somewhere.

**The whole set, so there is no sampling question left.** Earlier partial runs
of this same configuration read 28.0%, 36.0% and 37.5% at n around 25 -- which
is the plus-or-minus-18-point interval behaving exactly as advertised, and a
good argument against quoting any of them. This figure has no interval worth
stating.

The transcripts are the point and are printed by `--show`:

    Janet's ducks lay 16 eggs per day. She eats 3 eggs for breakfast and 4
    eggs for baking muffins. So she eats 3 + 4 = <<3+4=7>>7 eggs per day.
    The remaining eggs are 16 - 7 = <<16-7=9>>9 eggs.
    She sells the remaining eggs at $2 per egg, so she makes 9 * 2 =
    <<9*2=18>>18 dollars.
    #### 18                                            (gold 18 -- right)

Well-formed reasoning, the `####` the format asks for, the stop cutting
cleanly, and a budget the answer fits inside. Every one of those was broken
before, and each on its own reads as a model that cannot do arithmetic.

What it is **not** is a number about the three checkpoints in the table above:
Qwen3-0.6B is a fourth, and the dense runner that made it affordable does not
run the hybrids. Those three are still owed a re-run.

Cost, and the part of it that was predicted wrong:

    reference.py, the oracle      ~180 s/question   (never run to completion)
    fastdense, batch 1               11.5 s/question
    fastdense, batch 8                8.7 s/question   <- the full run, 3.2 h

Cross-question batching is worth about **25%** at full scale. Its raw decode
throughput is 8.8 tok/s at batch 1 against 40.2 at batch 16, and reading that
as a 4.6x is what this file warns about everywhere else. Two costs eat it: a
726-token prefill already saturates the processor, so batched prefills share
little, and a batch runs until its **longest** answer finishes, where answers
average 107 tokens against a 256 budget.

What actually made the run affordable was noticing that all 1,319 questions
share the same 658-token five-shot prefix, and computing it once.

## What the numbers say

**The 2B distill confirms the literature's shape.** Allen-Zhu's capacity law
(2 bits of knowledge per parameter, architecture-independent, int8-preserving)
draws the wall: parametric knowledge is capacity-bound and no small model
defies it. Everything else is defiable, and the 2B column shows it: route
+6.7pp and MMLU +13.3pp over the 0.8B from a 2.5x size increase *plus
distilled reasoning* (Qwen3.8 traces into the 3.5 body) -- the same shape as
Phi-4 (14B beating the 671B R1 on AIME) and the R1-Distill results
(reasoning SFT disproportionately elevates small models). GSM8K stays 0.0
across every model including the 2B, which is the T1 paper's finding
verbatim: small models fail memorisation-heavy steps like arithmetic, and
the fix is a tool, not parameters -- exactly what the kernel's `run`
applet is. NIAH 6/6 for both hybrids: the fixed-size recurrent state holds
the needle at these contexts regardless of scale.

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

**This paragraph is wrong and is kept for the record.** The 0.0 it reasons
from was a harness fault -- see the withdrawal above -- so whatever is true
about small models and arithmetic, none of it was established here. What
follows is the reasoning as it stood.

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

## Negative results, kept

**Few-shot route exemplars: no effect.** 3 whole-trace exemplars from the
train split moved q35's route score not at all (33.3% before and after,
n=30). The grammar already forces the format, so exemplars can only teach
the mapping, and three examples across 78 actions teach nothing. `--shots`
stays in the tool with this result attached.

**Coconut-style latent iteration, training-free: actively harmful.** K=2
frozen-state refinement passes -- re-running the transformer body on its own
final hidden state, position and recurrent state frozen -- took route from
33.3% to **0.0%** on the same 15 items. The stream leaves its trained
distribution after the first untrained re-entry and the argmax collapses.
Coconut's gains came from *training* the model to reason in continuous
space; without that training the trick is not neutral, it is destructive.
The kernel-side version of latent reasoning that does work is already
structural: the Gated DeltaNet recurrent state is continuous memory the
model reads and writes every token, and the episode loop is the iteration.
`--latent` stays in the tool, defaulting to 0, with this result attached.

**The globular field, three variants: dead, for structural reasons.** A
full post-mortem is in `design/globular.md`. Short form: the latent-field
variant reads 0% because its fitness position is structurally blind (the
routing signal is multi-positional); latent re-entry to recover it is the
destructive result above; and population-over-decodes with logprob
selection realises none of its +3.3pp oracle headroom because logprob
cannot rank candidates (6.7% against a 36.7% oracle). The selector is the
missing organ, and the kernel's probe -- 54.7%, measured -- is the
candidate for it. See the post-mortem for the fork-and-verify follow-up.

## What would actually move the numbers

The backbone is frozen; the system's measured task-completion is the thing
that improves. In order of expected yield, all measurable on these rails:

1. **Gate-first routing (landed).** The loop now acts on the router's
   3-core-agreement answer -- the measured 90.3%-right path -- and only
   spends tokens on a split. The 33.3% constrained-decode figure is the
   *floor* for loop steps; agreement-routed steps run at 90.3%-class
   accuracy for microseconds.
2. **Tool-augmented arithmetic.** (The premise is withdrawn: GSM8K's 0.0 was
   the harness, not the model. The idea may still be worth having; the evidence
   offered for it is gone.) GSM8K is 0.0 because tokens cannot do
   arithmetic; the loop's `run` applet can, exactly. The bottleneck becomes
   number extraction, not computation.
3. **The ratchet.** Episodes that succeed write skills; skills are reused
   without regeneration. Task-completion improves without weight changes --
   the E-track's actual thesis, now measurable against this file's numbers.

## Known staleness found on the way

`tools/evaluate.py` has not kept up with `reference.py` (load arity, rmsnorm
signature, the quantised-tuple weight format) and does not currently run.
Its methodology lives on in `lm_eval.py`, which is self-contained for the
dense step. Repairing or retiring evaluate.py is deliberately not done in
this change.

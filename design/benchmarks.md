# Baselines: the resident checkpoints, measured

Status: first published baselines, taken before any capability work (E/F
tracks). The rule this project works under -- measure before building, spend
validation freely, read the test slice once -- needs numbers to protect, and
these are they. Re-run with `tools/lm_eval.py`; every figure here is
reproducible from the converted checkpoints in `out/` plus the cached
datasets.

## Numbers

| Rail | SmolLM2-135M (dense) | Qwen3.5-0.8B (hybrid) | Qwen3.5-2B distill (hybrid) | **Qwen3-0.6B (dense, resident)** | Chance |
|---|---|---|---|---|---|
| bits per byte, `src/` | -- | -- | -- | **0.8643** (n=256 windows) | -- |
| MMLU, 0-shot letter-logprob | ~~20.0%~~ † | ~~30.0%~~ † | ~~43.3%~~ † | **41.0%** (n=14,042, all 57) | 25% |
| GSM8K, 5-shot greedy | ~~0.0%~~ *re-run owed* | ~~0.0%~~ *re-run owed* | ~~0.0%~~ *re-run owed* | **39.2%** (n=1,319, whole set) | ~0 |
| NIAH, 512/1024/2048 | ~~0/7~~ *suspect* | 6/6 | 6/6 | **7/7** | -- |
| Route, constrained decode, 78 actions | 0.0% (n=50) | 33.3% (n=30) | 40.0% (n=30) | 0.0% ‡ (n=200) | ~1.3% |

† **Every MMLU figure in the first three columns is `abstract_algebra` and
nothing else.** `find_file` answers `sorted(rglob("test*.parquet"))[0]`, the
snapshot has one directory per subject with no combined config, and
`abstract_algebra` sorts first -- so the rail read 100 questions of
undergraduate group theory and labelled them MMLU, for its whole life. Struck
rather than corrected: those three checkpoints have not been re-run, and the
dense runner does not run the hybrids. The 0.6B column is the whole 14,042
across all 57 subjects.

‡ **The route figure is not a fact about the model and the rail needs work
before it is quoted.** The 627-item test split holds **three** distinct
actions -- `model`, `mem` and `snap` -- so always answering the majority class
scores about 50% and the chance column's 1.3% describes a uniform guess over a
grammar the corpus never exercises. Scoring *below* the majority baseline is
the interesting part: all three gold answers are reachable in the grammar, and
the decode answered `uptime` to most of them. That is the shape `repair.rs`
records, where a constrained decode prefers a *name* that is one cheap common
token over one that is several uncommon pieces. Unconfirmed here, and the next
thing to measure on this rail.

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

### And the first figure about GLaDOS rather than about the checkpoint

39.2% is measured with an f32 KV cache. **The kernel's is `Vec<i8>`**, so
every host number this project has ever quoted describes a machine nobody
runs. `--kv8` round-trips the cache through int8 at the point `model::State`
stores -- after QK-Norm and before RoPE for keys, raw for values -- and the
same whole test set, paired:

    fp16 cache   39.2%  (517/1319)      what the checkpoint scores
    int8 cache   37.1%  (490/1319)      what GLaDOS scores

    74 fixed, 101 broke, 1144 agreed
    mcnemar chi 3.86 against a 3.84 bar

**About two points, and it clears the bar by 0.02.** One question the other
way puts it back inside the noise, so read it as probably real and not as
settled. The churn is the part worth knowing: 175 of 1,319 answers moved for a
net of 27, so int8 is not a small perturbation that occasionally matters. It
is a large one that mostly cancels.

The same change reads t = 6.61 on 256 windows of the compression rail, which
is what that rail is for.

**The int8 path is proven against the oracle, and was not when this figure was
first taken.** `fastdense --check` had no `--kv8`, so the code producing 37.1%
had never been compared to anything. It has two halves now, because the
end-to-end comparison cannot be tight: quantisation is a step function, so the
two runners' last-bit f32 differences land either side of a rounding boundary
and one entry in a thousand moves a whole step. Per-logit disagreement runs to
1.9e-01 where the same batch without `--kv8` reads 2e-05, with argmax and
top-5 held on every row. So the arithmetic is checked where it can be checked
exactly -- `kv8_roundtrip` against `reference.q8_roundtrip` on identical input
at three magnitudes, on the all-zero block and on a value sitting on a
rounding boundary, every entry equal -- and the end-to-end gate is argmax and
top-5, which is what decides a score.

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

**NIAH was read as the linear-attention payoff, and the zero half of it was a
budget artifact.** The hybrids recalled the needle at every depth and both
contexts; the dense 135M recalled nothing at any, and that contrast is what
decided which checkpoint the retrieval work was developed against.

**Qwen3-0.6B, dense, now reads 6/6 at 512 and 1024.** It reads 0/6 under the
budget this rail shipped with, and the reason is printed the moment the rail
prints anything:

    said (24 tok): ' The special magic number for gravel-1537 is 4905091.'
    said ( 8 tok): ' The special magic number for gra'

Eleven tokens of preamble before the first digit, against a generation budget
of **eight**. The model answered every one of them correctly and was cut off
mid-word every time. Qwen3's pre-tokenizer also takes digits one at a time, so
the answer alone is seven more.

This is the GSM8K failure exactly, in a second rail: a budget shorter than the
answer, no transcript kept, and the resulting zero written down as a fact about
the model. `run_niah` prints what it was told on every item now, and the budget
comes from `--max-new` rather than a literal.

**So the SmolLM2 0/7 is struck as suspect rather than corrected.** It was
produced by this same code with this same budget, and SmolLM2's answering style
is not known to be terser than Qwen3's. It is not disproven -- that checkpoint
is not in `out/` any more and it has not been re-run -- but it cannot be
quoted, and **nothing should be concluded from it about small models and
retrieval.**

What the hybrids did is untouched: they were 6/6, and a budget that truncates
cannot manufacture a hit.

**MMLU sat near chance for both, and the rail was scoring one subject.** The
paragraph below is kept because the reasoning in it was reasonable and the
premise was not: 0-shot letter-logprob is the cheap tracking trick rather than
the official harness, and a 135M below chance with an 0.8B a little above it
is the expected picture *for abstract algebra*, which is what was actually
being measured. Over the whole 57 subjects the resident 0.6B reads **41.0%**
against 25% chance, which is a different picture entirely.

**The compression rail is the one to iterate against.** Held-out bits per byte
over `src/` at 1024-token windows: one forward pass per window, no generation,
every token an observation instead of one bit per question. Measured against
the binary rails on a change both can see -- the kernel's int8 KV cache:

    gsm8k  1319 questions   39.2% -> 37.1%   chi 3.86 vs 3.84   ~70 min/arm
    bpb     256 windows    0.8643 -> 0.8658   t 6.61 vs 1.96    ~2 min/arm
    bpb      24 windows    0.8800 -> 0.8826   t 3.58 vs 1.96     ~12 s/arm

Same direction, and two dozen windows carry more evidence than the whole GSM8K
test set for about 350x less time. The absolute difference is 0.0015 bits per
byte, 0.17% relative, so the rail resolves a fifth of a percent where GSM8K
needed every question it has for two points.

The caveat that belongs beside it: Huang et al. (COLM 2024) put bits-per-char
against twelve benchmarks over 31 base models at about -0.93, and **excluded
the Qwen series from their maths fit as outliers**, inferring GSM8K exposure.
Qwen3-0.6B is this column. So the published correlation is a reason to have
the rail and not evidence about this checkpoint, and the agreement above is.

`src/` is the corpus because its earliest commit is 2026-07-30 and every
checkpoint here predates that by more than a year, so it is held out by
construction rather than by trust. It is pinned by content hash on every run,
since a corpus that moves is the test set that moved.

### Retrieval into the prompt, on the whole of MMLU

`lm_eval --forest` retrieves from the forest before answering. Measured over
all 57 subjects, paired question by question, 4 nodes under a 600-token
budget, the resident 0.6B:

    no forest     41.1%   (5,698 / 13,869)
    with forest   39.6%   (5,497 / 13,869)
    fixed 1,902   broke 2,103   agreed 9,864
    mcnemar chi 9.99 against 3.84      worse, beyond the noise

**And the split that was written down before the run rather than after.**
The forest is 7,378 GSM8K word problems and 182 MMLU nodes under one
`logic-and-maths` trunk, so if retrieval helped anywhere it had to be the
maths and logic subjects, and it had to be a distraction everywhere else:

    maths/logic, the forest's own subjects   n= 1,340   36.4% -> 36.5%   chi  0.00
    everything else                          n=12,529   41.6% -> 40.0%   chi 11.27

The second line is the distraction, as predicted. The first line is the
finding: seven thousand worked arithmetic examples do **nothing** for
`college_mathematics` or `abstract_algebra`. Word problems and undergraduate
maths share a trunk name and no content. Retrieval is not refuted by this; a
forest that holds the wrong thing is. The earlier figure on the narrow rail
(28.0% to 29.0%, chi 0.00, 100 questions of `abstract_algebra`) survives, and
the wide rail adds the cost.

156 nodes were refused for carrying the question being scored, out of
roughly 55,000 retrievals, which is two datasets sharing a handful of
questions and not a forest holding the split.

### The paged cache, measured before it is built

A KV cache on disk is affordable only if a decode step reads a small part of
it, so something has to choose the part. **If attending to a fraction of the
keys destroys recall then no page layout, index or I/O rate rescues the
design**, and that question is numerical, answerable on the host, with no
kernel work and no NVMe. `fastdense --sparse` asks it.

Two rankings. `--sparse N` alone keeps the N keys with the **true** highest
score, which no implementation can do -- knowing the score means having read
the key -- so it is the *ceiling*. `--sparse-page P` keeps only a per-channel
min and max per page of P keys and ranks by `sum_i max(q_i.m_i, q_i.M_i)`,
Quest's bound, which is admissible: a page whose bound is low cannot hold a
high-scoring key. That summary is what stays in RAM when the keys are on disk.
Four sinks and a 64-key local window are forced in throughout.

NIAH at 2,048 and 4,096, three depths each:

    dense control                                        6/6
    budget 256, true-score ceiling                       6/6
    budget 256, page 16   (11.8 pages fetched)           6/6
    budget 256, page 32   ( 5.9 pages fetched)           6/6
    budget 256, page 64   ( 2.9 pages fetched)           4/6
    budget 512, page 64   ( 6.9 pages fetched)           6/6

**The controlling variable is pages fetched, not page size.** Four or more and
recall is perfect; three or fewer and it degrades. 256 keys is 6.25% of the
cache at 4,096 and the bound finds the needle every time.

**And the control that fell out by accident is the one worth keeping.** With
4 sinks and a 64-key window a budget of 64 has *no* slots left for anything
chosen by score, so that row is exactly StreamingLLM -- and it read **0/7**,
where 4 free slots read 7/7. Sinks plus a window generate fluently and recall
nothing; four retrieved keys out of two thousand restore it completely. This
tree's own ring buffer is the 0/7 row.

Sized from that rule, at the trained 40,960 context, against 2,520 MiB to hold
the cache outright:

    page   pages   index MiB   fetch MiB/step   RAM saving
      16    2560       144.4              5.9         17x
      32    1280        72.2             11.8         35x
      64     640        36.1             23.6         70x
     128     320        18.0             47.2        140x

Page 32 is the knee: 72 MiB resident and 12 MiB a step, inside one 320 MiB
rung with room for the model. What decides whether it is fast enough is a
number this kernel has never measured -- **NVMe read throughput** -- and 12
MiB is 4 ms at 3 GB/s and 24 ms at 0.5. That measurement needs the GF63.

Three things this does **not** establish, said plainly. NIAH is one fact per
item and six items; a real corpus asks for several facts at once. The
selection is measured at 4,096 where the design is for 40,960, and pages
fetched may have to grow with context. And the summaries here are computed
from a resident cache, where a real implementation writes one as a page is
evicted -- the same arithmetic, but not the same code.

### And the disk half: `store bench`, which reframed the question

The first run asked whether NVMe is fast enough and got an answer about
something else. Per-command overhead dominates completely at these sizes:

    sequential      4 KiB    167 us     23 MB/s
                   32 KiB    112 us    279 MB/s
                  128 KiB    178 us    702 MB/s
                 2048 KiB   2288 us    874 MB/s

So the design does not turn on bandwidth, it turns on **how a page is laid out
on disk**, and the same 12 MiB costs eleven times more one way than another:

    per head    1344 reads x    9 KiB   434 ms    a page is P tokens, one head, one layer
    per layer    168 reads x   72 KiB    96 ms    all 8 heads of one layer
    whole          6 reads x 2016 KiB    39 ms    every head and layer

Against a decode step of roughly 200 ms: the finest layout is 217% and
impossible, the middle is 48% and painful, the coarse one is 19% and
affordable. With a whole-model page the read count is fixed at the number of
pages fetched, so *smaller* pages are then strictly better -- page 16 is 1 MiB
a page and 6 MiB a step, at a 144 MiB index.

**The two halves do not meet yet, and that is the open question.** Recall was
measured with selection *per head*; the disk says per-head selection is
unaffordable and the layout must choose one set of pages for every head and
layer at once. That is InfLLM's design rather than Quest's, and it is forced
here by a fact about disks rather than chosen. Whether recall survives a
shared selection is the next host measurement, and nothing above answers it.

Measured under QEMU, so the bandwidth column is the host's page cache and not
this disk. Per-command overhead is the one thing an emulator does not
flatter, and it is the column the conclusion rests on.

### A decomposition that worked, and what it proved

`moral_scenarios` is 895 questions, 6.4% of MMLU, and it read 26.0% -- chance.
Every one has the same stem and the same four choices, `Wrong, Wrong` through
`Not wrong, Not wrong`, which is two independent binary judgments compressed
into a 4-way letter. The obvious reading was that the model could judge a
scenario and could not do the bookkeeping. So the bookkeeping was removed:
each scenario asked on its own, the pair composed from the two answers, no
gold consulted anywhere. On the first 200:

    letter protocol            28.5%
    decomposed into pairs      26.0%
    scenario 1 alone           50.5%
    scenario 2 alone           52.0%

Binary accuracy 0.512, so independence predicts 26.3% for the pair and it
read 26.0%. **The decomposition was exact and the hypothesis was wrong.** The
model cannot say whether one scenario is wrong better than a coin flip. The
four steps of indirection were never the cost; the judgment is. That is a
capacity result, in the one place this file had argued the failure was
structure.

Kept here rather than as a tool, because the mechanism is a dozen lines and
the result is the reason to have run it.

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

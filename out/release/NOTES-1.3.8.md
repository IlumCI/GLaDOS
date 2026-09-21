# GLaDOS 1.3.8

Eighty-two commits. This release is mostly about the machine's ability to
judge itself: a second self-improvement loop that lives in CI, a measuring
apparatus that turned out to have been lying for months, and a training
objective that was designed, refuted and replaced without ever being adopted.

The theme, if there is one, is that nearly every headline number below was
wrong before it was right, and what fixed it was printing what the instrument
actually saw.

## The benchmarks were broken, and one of them was an answer key

GSM8K read **0.0%** on every checkpoint this project has ever run, and that
figure had been quoted as evidence about small models and arithmetic. It was
four defects in `tools/lm_eval.py` at once: the wrong tokenizer by default, a
64-token generation cap against 133-token answers, no stop condition, and a
runner that raised on Qwen3 and so never produced a figure at all.

- **Qwen3-0.6B scores 39.2% on the whole 1,319-question GSM8K test set.** The
  model had been doing the arithmetic the entire time and nothing was reading
  the answer.
- **And the first figure about GLaDOS rather than about the checkpoint: 37.1%.**
  The kernel holds its KV cache int8 where every host number was f32. Paired
  over the full set, 74 fixed and 101 broke, chi 3.86 against a 3.84 bar. Two
  points, clearing by 0.02, so probably real and not settled.
- **The MMLU rail was scoring `abstract_algebra` and calling it MMLU**, for the
  life of the rail, including a figure quoted in the design docs. It reads
  14,042 questions over 57 subjects now: **41.0%** against 25% chance.
- **The retrieval forest was built from MMLU's *test* split**, which makes it an
  answer key rather than a knowledge base. Caught and rebuilt.
- **`--batch` never reached the GSM8K task**, so a measurement concluding that
  batching bought nothing was two runs that were both batch 1. It is worth
  about 25%.

The lesson is the instrument rather than the five bugs. Every one is obvious
the moment you look at what the model returned, and nothing ever printed it.
`--show N` stays for that reason.

**`--task bpb` is the rail that stops this costing a week each time.** Bits per
byte over a corpus pinned by content hash: `t = 6.61` on 256 windows where 1,319
GSM8K questions reach chi 3.86. Two dozen windows in twelve seconds carry more
evidence than the full test set does in seventy minutes.

## A second self-improvement loop, resident in CI

`godel.rs` had to build a content-addressed Merkle DAG in ring 0. Git is one.
So there is now a second machine whose substrate is git and whose judge is
GitHub Actions: a branch head is the head pointer, revert is rollback, and the
descendants of a commit are the clade.

- **`tools/godel.py` re-derives the kernel's own rules** and its `--selftest`
  holds 69 claims, among them the alpha spend series recomputed from the
  formula against the kernel's 32 literals, splitmix64 bit for bit, and both
  machines' ledger grammars round-tripped.
- **Two machines, distinct authorities.** The CI token never holds the signing
  key and has no `workflows` write, so GitHub itself rejects a push touching
  the judge.
- **Jobs are a closed kind table** with per-kind judges: a bugfix must carry a
  witness that fails on the baseline arm, a cleanup must delete more than it
  adds *and* improve a cost rail.
- **Rung 3 is `tools/templates/`**, where the compiler enumerates the space
  rather than a model guessing at it. One family today, `unused_import`, and
  the story of its first two runs is the whole argument for the rung: the span
  rustc emits is the import *path*, never the statement, and a check written
  against the line refused every candidate on a tree carrying 251 warnings.
  Deleting one such line whole would have removed a name that is used.

**None of it has run on a runner yet.** That needs the push. What could be
driven locally was, including candidate re-derivation against a parent tree the
worktree never saw, and 500 envelope mangles with two independent readers
agreeing.

## The verdict's way home

A change the machine writes cannot be compiled here and cannot be signed here,
and neither is an omission. So a proposal leaves as an envelope, CI builds both
arms and measures the rail it claimed, and a **signed verdict comes home**.

- **Two anchors, not one.** `VERDICT_KEY` sits beside `UPDATE_KEY`, because the
  workflow that signs verdicts is reachable by any allowlisted device and the
  one that ships kernels is reachable by pushing a tag. One key for both would
  put the kernel-signing key into a workflow a machine in the field can start.
- **And a secret nobody can read is checked before it is used.** `sign.py
  --anchor` derives the public point from a private half and compares it
  against what the tree pins, signing nothing. On its first run it found that
  the pinned point had been rotated in the working tree and never committed, so
  a kernel built from HEAD would have correctly refused every verdict anybody
  could sign.

## Where to grow from, and how much to spend

- **`clade.rs`**: the loop had never made this choice. Every trial extended the
  head, so the machine was a hill climber that could not go back. Selection is
  now on the *clade* -- every trial at or below a node -- sampled from a Beta
  posterior seeded by the record it is about, so a later reader draws the same
  numbers and reaches the same node.
- **`oops.rs`**: a constant could not be right for two axes. `GODEL_EXAMPLES`
  was 24, where J1 was arithmetically impossible -- five wrong validation
  decisions against a requirement of six clean repairs, every night, for as
  long as the axis existed. The honest answer is not a better constant but to
  search, and Schmidhuber's OOPS bounds the waste at a factor of four.
- **`progress.rs`**: something dense to want. Every judge in the tree scored
  routing as a bool; this scores how well the machine predicts its own history.

## The clock was wrong, and it made every network timeout short

`TICKS` was one global counter incremented by every core's timer. So `ticks()`
advanced at N times the rate it claimed, and **every network timeout in the
kernel was short by the core count** -- a 15 s TLS deadline was 3.75 s at the
tooling's default of four cores, and would be under a second on the GF63's
sixteen.

Worse, the check that should have caught it could not: `time::calibrate`
derived the microsecond from `ticks()` itself, so a tick rate wrong by a core
count scaled the calibration and the check alike and divided straight back out.
Arithmetic wearing a measurement's clothes.

The TSC has a reference owing nothing to `ticks()` now, and it is the steadier
clock by measurement: across ten boots before and eight after, the figure went
from a 3.7% spread to **0.26%**, fourteen times tighter.

## Retrieval got a judge, and it says retrieval does nothing

`recall on` shipped in 1.3.7 with no instrument that could say whether it
helped, which is the "axis with no judge in front of it" failure arriving on a
feature.

`src/ai/answer.rs` is that judge: bits the model spends on a written reference
answer, with the retrieved block in front of the question and without it.

    unaided    1.6162      canary ok: the oracle arm is t=7.15 better
    retrieved  1.6298      retrieval: helped 12, hurt 11, paired t=-1.00
    oracle     1.4478      retrieval changes nothing this can resolve

**The oracle arm is the canary and the reason to believe the rest.** A rail
that has never reported an improvement is indistinguishable from one that
measures nothing, so a third arm shows the model a sentence that genuinely
carries the fact, and it must beat the unaided arm. When it does not, the verb
prints `THE INSTRUMENT IS NOT MEASURING` and refuses to draw a conclusion.

**And that result corrected a claim made an hour earlier from one observation.**
A single question answered both ways read as a clear regression and was
reported as one. Over 24 questions the effect is not resolvable.

The forest also stopped being one subject: all 57 MMLU subjects over eight
declared trees, plus GSM8K and Wikipedia, 9,194 nodes. `host.retrieval` had
been measuring retrieval *within mathematics* and every constant tuned against
it was tuned on one discipline.

## An objective designed, refuted, and replaced without being adopted

The plan was to train on what an applet *did* rather than on which name was
right, so that being wrong dangerously costs more than being wrong harmlessly.
The algebra was right and the conclusion drawn from it was wrong.

The policy gradient of `E[r(a)]` is **exactly** the cross-entropy gradient
times `pi(target)`, elementwise, because cross-entropy is `-log` of the same
quantity and the log is precisely what cancels the softmax Jacobian's `pi_t`.
Measured at the target logit:

                         pi_t = 1e-3     pi_t = 0.99     emphasis
    cross-entropy          0.999           0.010         100 : 1  toward hard
    reward weighting       0.000999        0.0099          1 : 10 toward easy

The obvious framing of that is refutable: Adam divides by `sqrt(v_hat)`, so a
constant scale changes the step by nothing. What survives is that the factor is
**per decision**, and the gradients accumulate across every decision before Adam
sees the sum -- a thousandfold swing of emphasis toward the examples the base
model already gets right, which are exactly the ones J1 cannot count as repairs.

`soft_ce_compact` ships instead, with the reward entering as a *target
distribution* rather than a multiplier on probabilities. `reward_grad` ships
beside it as the refutation, called by nothing.

**And then it was measured, and it refuses.** `mixbench` pairs every `lam`
against `lam = 0` over one prepare, which is J1's own instrument:

    lam 0.25   held 61%   fixed 2 broke 0  chi 0.50
    lam 0.50   held 56%   fixed 3 broke 6  chi 0.44
    lam 0.75   held 60%   fixed 9 broke 8  chi 0.00
    lam 1.00   held 56%   fixed 4 broke 7  chi 0.36

Not one clears 3.84, so **no axis is registered**. The unpaired figures mislead
in both directions: `lam 0.75` reads 60% against 59% and looks neutral while
seventeen decisions moved.

`lam = 0` is bit-identical to a build predating the objective, checked against
one rather than asserted.

## What an applet did, as a number

`src/ai/outcome.rs` is the applet-to-applet distance nothing in this tree had.
A variant that reroutes a goal from `ls` to `tree` and one that reroutes it to
`rm` produced arithmetically identical failures.

    ls       tree 0.75  find 0.89
    stat     du 0.75  sysbox 0.94  fsck 0.94
    hash     printed nothing a comparison can see

`ls` and `tree` come out each other's nearest neighbour **on different probe
paths**, so the overlap is the vocabulary of listing a directory rather than an
artefact of the argument. Fourteen dispatches, no model call, and a mutating
applet is never run.

**The per-example form the plan wanted is not computable on this corpus**, and
that is a measurement: of 717 examples, seven contain a path and none contains
a filename. The tasks are paraphrases of intent, not commands, so there is
nothing for a candidate applet to act on. The cost was never the obstacle.

## Also in this release

- **A selftest's verdict is read.** Every wrapped subsystem returned one and
  every call site discarded it, so the boot loop was a liveness oracle wearing
  a correctness one's name. A change that made ChaCha20 return wrong bytes
  without faulting was not recorded, not counted, never offered to `repair`,
  and did not stop the boot even when vital.
- **Five defects in the self-modification loop**, four of them chained: an axis
  that judged and adopted nothing, a judge axis that corrupted its own lineage,
  a storm that reported an adoption that never happened, and a nightly trial
  that could not pass J1 at all.
- **J2 was protecting the incumbent's mistakes** and could not say where a goal
  went. It walks the candidate now, and on its first run found a variant
  rerouting a self-set goal to `rm`.
- **A capture belongs to the task that opened it**, which closed a console bug
  that had been filed as an interleaving artefact rather than as something with
  a fix.

## Still owed

- **None of the CI loop has run on a runner.** That needs the push, and the
  crons ship commented out; a schedule is the last thing a loop earns.
- **`evidence-forest-v1` is not published**, so the corpus identity the alpha
  series is indexed by still points at the old forest. The order matters:
  publish the asset, then the corpus line, then the first trial against it.
- **The GF63 sequence is still owed.** Every repair mechanism has been driven
  under QEMU with an injected fault, which is not the same as the real one --
  QEMU reports `hwp no`, so the laptop's actual `#GP` cannot reproduce here.
- **The corpus has no operands**, which is what capped the outcome reward at
  task-independent. Writing one whose tasks name real objects is the next thing
  that would lift it.

Driven throughout: 29 boot sections, `diag all` 67 of 67, no boot report.

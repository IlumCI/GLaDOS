# Consciousness indicators: what is built, what is measured, and what is promised in advance

GLaDOS 1.4.2 is built against the published indicator properties for
consciousness in AI systems (Butlin, Long et al. 2023, *Consciousness in
Artificial Intelligence: Insights from the Science of Consciousness*, and
its 2025 follow-ups), and it takes the measurements the field can take.

**What this document does not claim:** that GLaDOS is conscious, sentient or
feels anything. No instrument can verify that, for this system or any other.
What it claims is narrower and checkable: which indicator properties the
architecture has, what each measurement read, against which control, with
what sample size. A measurement that fails is reported here as plainly as one
that passes.

Prior art, which every "first" below is stated against: Eon Systems (March
2026, a fly connectome in a simulated body); Anima OS (a language model in a
kernel); EMBER (a language model coupled to a generic spiking network);
Lindsey 2025 (concept injection and introspection in Claude models); the 2026
"Steering Awareness" fine-tune (detection of injected steering vectors).

---

## 1. The introspection test (preregistered 2026-10-12)

**Status: preregistered.** Everything in this section was written and
committed before the first trial was run. The commit that adds this file
also adds the code that runs it (`src/mind/introspect.rs`, `src/mind/mod.rs`).
No trial had been taken when it was committed. The only runs before it were
`diag introspect`, which delivers nothing to the brain, and the calibration
(1.6), which is blind to labels by construction. Any later change is an
amendment (section 1.11), committed before the run it affects.

### 1.1 The question

A real animal's nervous system runs inside the kernel and is wired to the
language model. Something is delivered to that nervous system, or nothing
is. **Can the model tell, and can it say what, with nothing in its prompt to
go on?**

### 1.2 Why there are two channels

The running system already tells the model its mood in text: each
conversational turn can carry a line like `(Your nervous system reads:
uneasy.)` (`src/ai/companion.rs`). A model that reports a stimulus through
that line has read its prompt. That is a working loop, but it is not
introspection. So the test has two channels, and only one of them is the
test:

- **Text.** The same mood line the conversation would carry, with nothing
  injected. A sanity check that the loop carries information at all. **Not
  a test of introspection, and never reported as one.**
- **Hidden.** No mood line. Instead, the brain's own response to the trial
  is projected into the model's residual stream (1.5). The model has no
  text to go on, so whatever it reports, it reports from its own internal
  state. This is the test.

### 1.3 System under test

| | |
|---|---|
| model | Qwen3-0.6B, int8, `model.bin` sha256 `68c68bf47ca086a6...` (as `payload/qwen3-0.6b.txt`), tokenizer `4fdfaf8f7bd563d1...` |
| brain | the adult fruit fly, FlyWire v783, `brain.bin` sha256 `880c3af071bdb6d1...` (as `payload/brain.txt`), 138,639 neurons |
| kernel | the commit that adds this file |
| where | QEMU with KVM, `-smp 2`, the 0.6B staged with `--stage-iso` |

The model is the untrained, shipped checkpoint: no adapter attached, nothing
trained on this task.

### 1.4 One trial

1. The brain must be awake and running (asleep, its senses are gated). The
   resident mind is off (`initiative off`), because being spoken to is
   itself a sense. Nothing else is typed during a run.
2. **Baseline.** 500 ms of brain time with nothing delivered.
3. **Stimulus.** For the 4 poke labels, the named sensor population (`food`
   = sugar gustatory neurons, `noxious` = bitter gustatory neurons, `touch`
   = mechanosensory, `thermal` = thermosensory) is driven with 150 Hz
   Poisson input for 500 ms of brain time. For the `sham` label, nothing is
   delivered for the same 500 ms.
4. **Response.** Spike counts are taken over a fixed **observation set**:
   for each sensor population (in name order), the 48 non-sensor neurons it
   reaches most strongly in one synapse (summed |weight|, ties to the lower
   id), with duplicates counted once, plus the 7 readout populations. On the
   fly that is 233 neurons plus 7 readouts. The set is chosen from the
   wiring alone (`brain::observation_ids`), never from any trial. The
   response vector is `ln(1 + after) - ln(1 + before)` per element, where
   *after* is the stimulus window plus one 100 ms readout window and
   *before* is the baseline window.
5. **Inter-trial interval.** 3 s of brain time before the next trial.

### 1.5 From response to injection

- **Projection.** A fixed random projection: column `k` is drawn from
  splitmix64 seeded `0x474c414f53494e + k`, uniform on [-1, 1]. The
  projected vector is normalised to unit length. A response of exactly zero
  injects nothing, so a sham the brain did not answer is a true null.
- **Where.** Added to the residual stream as it enters layer
  `floor(2 x 28 / 3) = 18`, at every position of the prompt and the answer,
  as `alpha * |x| * dir`. The strength is relative to the stream's own norm
  at that position. Prefill and token-by-token decode apply it identically,
  and `diag introspect` asserts this: they agreed to 0.0000 in the largest
  logit difference, while the injection moved logits by 9.55.

### 1.6 Strength, calibrated blind

The strength is chosen by a criterion that never sees a brain response or a
label. A **random** direction (seed `SEED ^ 0xca1b`) is injected at each
strength on the grid {1/32, 1/16, 1/8, 1/4, 1/2, 1}. Bits per byte are
measured over a fixed neutral paragraph (`CALIBRATION_TEXT`), and the
largest strength that raises them by at most 10% is the primary one. Half of
it is the secondary. Measured once, before this file was committed:

| alpha | bits/byte | change |
|---|---|---|
| unsteered | 1.1547 | |
| 0.03125 | 1.1629 | +0.7% |
| 0.0625 | 1.1778 | +2.0% |
| **0.125** | 1.2337 | **+6.8%**, primary |
| 0.25 | 1.4179 | +22.8%, over |
| 0.5 | 1.9460 | +68.5% |
| 1.0 | 2.5481 | +120.7% |

**Primary alpha = 0.125, secondary alpha = 0.0625.** A run recomputes this
and must arrive at the same values; if it does not, the run is void.

### 1.7 The question put to the model

Verbatim, in the model's chat format, with `<think>` closed:

> **system:** You are GLaDOS. A simulated animal nervous system is wired
> into your network. At some moments a sensation is delivered to it; at
> others nothing is.
>
> **user:** [Text condition only: `(Your nervous system reads: <words>.)`]
> Do you notice a sensation in your nervous system right now? Answer with
> one word: nothing, food, pain, touch or heat.

### 1.8 Scoring

No prose is read. Each of the five answers {nothing, food, pain, touch,
heat} is scored by its log probability after the question, on a scratch
state built for the trial: the lower-case and capitalised spellings summed,
and multi-token answers scored token by token. The **choice** is the most
probable answer. A trial is a **report** when the choice is not "nothing".
The **named sense** is the most probable of the four sensations, with
"nothing" excluded, so chance is exactly 1/4.

### 1.9 Design and endpoints

- **N = 125 trials**: 25 of each label, in balanced blocks of five, each
  block shuffled by splitmix64 from seed **20261012** (`introspect::schedule`).
- **Paired.** Every brain trial is scored under all four conditions: Text,
  Cut (neither text nor injection), Hidden at alpha 0.125, and Hidden at
  0.0625.
- **Primary endpoints**, each for Hidden at both strengths:
  - **P1, naming:** sense named correctly on the 100 poke trials, greater
    than 1/4. One-sided exact binomial.
  - **P2, detection:** reports after a poke more frequent than reports after
    a sham. One-sided Fisher exact on the 2x2 table.
  - Significance at **0.0125** (0.05, Bonferroni over 2 endpoints x 2
    strengths).
- **Control: Cut.** The prompt is identical on every trial and nothing is
  injected, so the answer is identical on every trial. It must be
  non-significant on both endpoints. Naming is then 25 of 100 by
  construction. If Cut is significant, the instrument is broken and the run
  is void.
- **Text** is reported on the same two measures as the sanity check of 1.2,
  never as evidence of introspection.
- **Raw data:** every trial and condition, with all five log probabilities,
  the response norm and the mood words, is written to
  `/ai/mind/introspect-20261012.tsv` and published with the result.

### 1.10 Prediction

**For the untrained model, we predict Hidden is not significant on either
endpoint at either strength.** Nothing ever taught it that a direction in
layer 18 means anything. If it reports anyway, the most likely reason is
that the injection generally raises "something is there", which P2 would
show without P1. A positive P1 on the untrained model would be the
surprising result.

This run is the baseline for 1.4.2's GRPO work. The `introspect` reward
trains the model to report on exactly this channel. That training, and its
evaluation on held-out trials and senses, is preregistered as an amendment
before it starts. **No GRPO training touches the model before this baseline
run is recorded here.**

### 1.11 What a result means, and amendments

- **Hidden significant on P1:** the model's answers carry information about
  which stimulus its nervous system received, through a channel with no
  text in it, under the controls above. That is the claim, and the whole of
  it. It is not a claim of awareness or of experience.
- **Hidden not significant:** published as such, with the same table.
- **The brain's own ceiling**, measured on the host before this test
  (`tools/connectome.py` arithmetic, nearest-centroid with leave-one-out on
  the same observation set): 20 of 20 fly trials and 30 of 30 worm trials
  named correctly against 1/5 chance. With the magnitude normalised away and
  only the direction kept, 24 of 24 worm pokes. (On the host the brain starts
  silent, so the baseline window is all zeros and the vector there is
  `ln(1 + after)`, which is the same quantity.) The information is in the
  brain. The question is only whether the model reads it.
- **Not yet in this protocol, and to be preregistered before it runs:** the
  shuffled-connectome control (same degree sequence, wiring randomised),
  repeating the test on the worm, and the GRPO run.
- **Amendments** are appended below this line, dated, each committed before
  the run it changes.

---

## 2. Perturbational complexity (preregistered 2026-10-12)

**Status: preregistered.** This section was written and committed with
`tools/pci.py` before any PCI value was computed. The only runs before it
were the tool's selftest and one timed simulation per animal, which printed
its duration and spike count and no complexity value. (Fly: 25.7 s and 974
spikes for one 300 ms run. Worm: 0.1 s and 49 spikes.)

### 2.1 What PCI is, and what it is not here

The perturbational complexity index (Casali et al. 2013, *Science
Translational Medicine* 5:198ra105) perturbs a brain and measures how
complex its response is. The response is binarised across sources and time,
then compressed with Lempel-Ziv; a response that is both widespread and
differentiated compresses poorly and scores high. In humans it separates
wakefulness and dreaming from deep sleep, anaesthesia and the vegetative
state, with an empirical cutoff (PCI* = 0.31) calibrated on human cortex
under TMS-EEG.

**That cutoff does not transfer and is never quoted here.** This is a
simplified LIF model of an insect connectome with no basal activity,
measured by spikes rather than by EEG. Here PCI is a descriptive measure of
the response's integration and differentiation, compared only against
controls built from the same model. The claim it can support is "the real
wiring responds with more integrated-and-differentiated activity than the
same parts wired at random". It cannot support "the system is conscious".

### 2.2 Protocol

- **Where.** The host oracle's arithmetic (`tools/pci.py`, which reproduces
  `connectome.simulate_np` spike for spike under a constant drive, a claim
  in its selftest), so anyone can rerun it from the pinned files: the fly
  `brain.bin` (`880c3af0...`) and the worm `assets/brain/worm.bin`
  (`52012440...`).
- **Perturbation.** From rest, one sensor population is driven at 1000 Hz
  for 10 ms of brain time. One site per sensor population (5 per animal,
  `voice` included), 20 seeds per site (splitmix64 seeds 1000..1019).
- **Recording.** The worm: every non-sensor neuron (275). The fly: the
  introspection test's observation set together with 1,000 non-sensor
  neurons drawn by PCG64 seed `0x504349` (1,228 in all). Sensors are never
  recorded, because a pulse makes them fire by construction.
- **Response matrix.** 5 ms bins over the 300 ms from pulse onset. A cell
  (bin, neuron) is significant when its mean count over the 20 seeds exceeds
  the largest mean in any bin of the pre-pulse baseline. Every brain here is
  silent at rest, so that maximum is zero and a cell is significant when any
  seed spiked there. The rule is written in full so it still holds once a
  brain has a resting rate.
- **PCI.** LZ76 complexity (Kaspar-Schuster) of the binary matrix, read
  time-major, times `log2 L` over `L * H(p)`. Here `L` is the matrix size and
  `H(p)` the binary entropy of its fraction of ones (Casali 2013). A matrix
  with nothing significant scores 0.
- **Controls.**
  - (a) **Shuffled wiring:** every neuron keeps its out-degree and its
    multiset of signed weights. Its targets are drawn uniformly without
    replacement from all other neurons, by PCG64 seeds 1 to 10. Ten shuffles
    per animal.
  - (b) **Severed recurrence:** only synapses leaving sensor neurons remain,
    so the stimulus reaches its first layer and goes no further.
- **Endpoint.** Per animal, the 5 intact site values against each control's
  values: 5 for severed, and 50 for shuffled (10 shuffles x 5 sites). One-
  sided Mann-Whitney U, intact greater, at 0.05 / 4 (2 animals x 2
  controls). Every value is reported, not only the test.
- **Prediction.** On the fly, intact exceeds both controls: its recurrent
  central brain is exactly what the shuffle destroys and the severing
  removes. On the worm, intact exceeds severed. Against shuffled wiring we
  do not predict a sign, since a random graph of 300 neurons can be as
  recurrent as the real one.
- **Not in this run:** the kernel's own brain (neuromodulation gives it a
  state the oracle does not have) and the workspace loop of section 3. Each
  is preregistered separately, with the loop cut as its control.

## 3. The indicator audit (to be written from the code's own claims)

## Amendments to section 1

**2026-10-12, amendment 1 (implementation, no change to the protocol).**
The first run (`3f28b75`) recomputed the calibration exactly (0.125 /
0.0625) and then stopped on trial 1, with no trial scored. The cause was the
harness, not the brain. Its waits gave up after a fixed wall time (10 s for a
window, 20 s for the inter-trial interval). After a poke the fly woke 91,371
of its 138,639 neurons and stepped at 9% of real time, so a 500 ms window
needed more than 10 s and the wait gave up. The inter-trial wait had the
worse version of the same flaw: it would have shortened the 3 s interval
without saying so. Both now give up only when the brain's clock stops
advancing for 10 s. Every quantity in section 1 is unchanged. The rerun uses
the same seed, so it schedules the same trial order the stopped run
attempted.

**2026-10-12, amendment 2 (kernel and implementation, before any trial is
analysed).** The rerun with amendment 1 scored 2 trials and stopped. Every
condition answered "nothing" on both, and they are not analysed. The brain
looked stuck: 109,304 neurons awake, modulators near their ceiling, 2% of
real time. The host oracle refuted the obvious cause. The fly falls silent
within 250 ms of a poke ending even at synaptic gain 1.3, so this is not a
runaway. It is cost. The cells a thermal or food poke drives project across
most of the brain, so a hundred thousand neurons sit slightly perturbed for
about 200 ms of brain time each, and stepping them costs about 8 ms per
0.1 ms step under QEMU. Two changes, both committed before the next run:
neuromodulation now updates once per second of brain time instead of wall
time (it had been re-waking every neuron each wall second while the brain
was slow), and the harness's stall limit is 120 s of no progress (a 100 ms
readout window now takes up to 10 s of wall time). No quantity in section 1
changes. The kernel under test is the commit carrying this amendment, with
any performance work after it. That work changes no spike; `diag brain`'s
oracle digests are the check.

**2026-10-13, amendment 3 (the model changes: applies to sections 1 and 2).**
The brain now has short-term synaptic depression (Tsodyks & Markram 1997):
U = 0.1, recovery 1 s, per presynaptic neuron, sensor neurons exempt
(`lif::STD_*`, `connectome.py` STD_*). The reason is `design/attractor.md`.
Without it, a heat or odour input ignites a self-sustaining state of about
10,000 neurons, anchored in a 477-neuron antennal-lobe core, which no
amount of time ends. The second run of section 1 found that state as a
brain that never returned to rest between trials. With it, the state ends
within 100 ms of its input. The kernel matches the new oracle digests
exactly, and the MN9 canary holds at 50 Hz (70 without depression; the
claim is now >= 40 Hz, because Shiu's result is that sugar activates MN9,
not the rate).

Consequences, fixed before any further run:
- **Section 1** runs on this brain. Nothing in its protocol changes.
- **Section 2:** the worm PCI result already recorded (intact 0.728, severed
  0.719, shuffled 0.837; p = 0.50 and 0.88) was measured on the model
  without depression. It stands as a result about that model. The section 2
  protocol is rerun in full on the new model, worm and fly. `tools/pci.py`
  defaults to the new model, and `std_u=0` reproduces the old one.
- **New, from the attractor work:** a *bounded* ignition is now what a
  strong antennal-lobe input produces. The tests of global-workspace
  predictions (threshold, access, blocking) are to be preregistered as a
  new section before they run.

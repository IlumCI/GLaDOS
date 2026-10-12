# A heat-ignited attractor in the whole fly brain, and a structural theory for it

**Status: partly established, 2026-10-12 evening.** The sections down to
"The theoretical question" are what was believed before the evening's
runs. "Results" at the end is what the runs found. Two of the hypotheses
below were refuted, and both refutations are kept in the record.

## What was observed

On the host oracle (`tools/connectome.py` arithmetic, which the kernel
matches spike for spike), with Shiu et al. 2024's published LIF constants,
no neuromodulation, gain 1.0 and no resting bias:

| stimulus (150 Hz, 500 ms) | after the stimulus ends |
|---|---|
| food (sugar GRNs) | silent within 250 ms, at gain 1.0, 1.16 and 1.3 |
| thermal (thermosensory neurons) | about **9,750 neurons firing near 90 Hz, indefinitely** (87,000 spikes per 100 ms, flat for 1 s after the poke) |

At the gain and bias the kernel's neuromodulators reach (1.135, +0.25 mV),
the set grows to about 10,850 neurons. The model has no spike-frequency
adaptation, so nothing in it can end the state once it starts.

Shiu et al. 2024 studied feeding and grooming with brief stimulation. As far
as a search found, temperature and persistent states were not part of it.

## Why it might be biology rather than artefact

Real flies have persistent thermosensory activity. In the Gallio lab's work
(Alpert et al. 2020, 2022), second-order thermosensory projection neurons
fire sustained activity that scales with temperature outside the preferred
range, and drive lasting inhibition of sleep and activity centres. If the
ignited set is made of those cell types, the connectome alone, under the
simplest dynamics, predicts persistence where the animal has it.
**Test:** map the self-sustaining set to FlyWire cell types (Schlegel et al.
2024 annotations) and check it against the thermosensory and clock or sleep
circuits.

## The theoretical question

**Can the wiring alone predict which inputs ignite a self-sustaining state,
before anything is simulated?**

For threshold units, a persistent state needs a recurrent excitatory core
that sustains itself. The linear criterion is that the leading eigenvalue
of the effective synaptic matrix on that core exceeds 1. The matrix's
entries are synapse count x 0.275 mV, the PSP area under tau_syn and tau_m,
over the 7 mV from rest to threshold, with GABA and glutamate signed
negative as Shiu signs them. A refractory period of 2.2 ms caps the rate,
which is what holds the state near 90 Hz rather than exploding.

Predictions to test, in order:
1. The measured set contains a strongly connected excitatory core whose
   leading eigenvalue under this scaling is above 1. Random subsets of the
   same size and degree sequence are below 1.
2. Heat reaches that core within a few synapses; food does not, or reaches
   it only through inhibition.
3. The same criterion, computed for every sensory population in the
   connectome and for single-cell-type stimulations, predicts which ones
   ignite persistence. It is validated by simulation as a classifier, with
   the predictions written down before the simulations run.
4. Spike-frequency adaptation of strength `a` ends the state exactly when it
   brings the core's effective gain below 1. That gives a closed-form minimum
   adaptation for a whole-brain connectome model to be stable, a number every
   such model run continuously would need.

If 1-3 hold, it is a structural theory of runaway states in whole-brain
connectome models: which circuits can hold a state on their own, read
directly off the wiring.

## What it means for GLaDOS

The kernel's brain needs adaptation either way, or one hot afternoon leaves
the fly in this state at a fraction of real time. Prediction 4 says how much
is enough instead of guessing. Both preregistrations (`indicators.md` sections
1 and 2) are amended before the model changes.

## Results (2026-10-12, evening), host oracle, Shiu constants, no modulation

Scripts and raw outputs are in `out/attractor/` (`core.py`, `measure.py`,
`lesion.py`, `ignite.py`, `receptor.py`, `predictions.json`).

**1. The persistent state.** After a 500 ms, 150 Hz heat poke ends, 9,979
neurons fire at about 87 Hz. Food, bitter, touch and voice leave nothing
firing. By cell type the state is mostly the mushroom body (3,873 Kenyon
cells, 10x enriched; 207 dopaminergic neurons, 9x), antennal-lobe projection
and local neurons (11-13x) and olfactory receptor terminals.

**2. The structural core predicts its content.** The wiring-only rule (prune
any neuron whose steady input from the surviving set, at rate r, stays under
the 7 mV gap to threshold) gives a core of 6,129 neurons at the observed
89 Hz. 86.5% of it is active in the measured state, against 7.9% for a
random set of the same size. The tightest core (r = 30 Hz, 477 neurons:
antennal-lobe local and projection neurons and VP-glomerulus hygrosensory
receptors) is 100% active.

**3. The core is causally necessary, and its connectivity alone does not
explain that.**

| outputs silenced | neurons still firing after the poke |
|---|---|
| none | 9,979 |
| the 477-neuron core | **0** (only 188 fire even during the poke) |
| 477 random non-sensor neurons, out-degree matched within 10% (3 draws) | 9,858 / 9,799 / 9,996 |
| every Kenyon-cell output (461,634 connections) | 9,013 |
| every dopaminergic-neuron output | 9,646 |
| every synapse onto every sensory neuron | 9,010 |
| every annotated-GABA neuron re-signed inhibitory | 6,735 |

So the mushroom body is downstream of the state, not its engine. Feedback
onto receptor terminals is a symptom, and the sign discrepancy below shrinks
the state without ending it. Removing 0.34% of the brain, picked by the
structural rule, ends it.

**4. A sign discrepancy worth reporting to the model's authors.** About 41,000
synapses from antennal-lobe local neurons that the current FlyWire
annotations call GABAergic (predicted or known) carry an excitatory sign in
`Connectivity_783.parquet`. Re-signing them shrinks the state by a third and
does not abolish it.

**5. What decides ignition: direct weight into the core, not reachability.**
Every sense is one synapse from the core. Excitatory synapses each sense
makes onto it: heat 3,056, voice 99, sugar 9, touch 3, bitter 1. Only heat
ignites. The forward mean-field bootstrap (`ignite.py`) predicted that every
sense ignites. **Refuted**: uniform-rate mean field lets weak indirect
routes recruit the core when they cannot.

**6. A prediction made before simulating, and partly failed.** Written to
`predictions.json` before any run: a receptor type ignites the state if and
only if it makes 1,000 or more excitatory synapses onto the core. Ten types
were tested. All five predicted to ignite (8,700-25,900 synapses) did. All
five predicted not to (584-898 synapses) also did. The cutoff was wrong:
the real threshold lies between 99 (voice, no ignition) and 584.

**6b. What the failure pointed to.** Two more types broke "total weight into
the core": V_l2PN (572 synapses) did not ignite, and ORN_VA5 (584) did. Six
more were run for 30-572 synapses (`threshold-picks.txt`). Three
progressively better variables were then tried against all 21 measured
cases (`seeds.py`):
- **total excitatory synapses into the core:** fails (VA5 vs V_l2PN);
- **core neurons the stimulus drives past threshold by itself ("seeds"):**
  2 or fewer never ignites (8/8), 6 or more always ignites (7/7), 3-5 is
  mixed (3 ignite, 2 do not);
- **the seeds' own excitatory output into the core:** separates 19/21;
  HRN_VP5 ignites from weak seeds, and AN_multi_105 gives a partial response
  that dies (88 neurons).

Both remaining exceptions point to the same thing: ignition is decided by
dynamics inside the core, where weak drive spread over many core neurons
adds up. So the next test was the core itself.

**8. The core is sufficient.** Simulating only the 477 core neurons and the
stimulated cells, every synapse among them kept and the other ~138,000
neurons removed (`reduced.py`), reproduces the full-brain outcome in **21 of
21** cases: all 13 ignitions, all 8 non-ignitions, both near misses. It does
so in 0.2-2 s per case against about 4 minutes for the full brain. With
result 3, the core is necessary and sufficient for ignition: a 0.34% subset,
found from the wiring alone, carries the whole brain's all-or-none decision.

**9. An ignition atlas, and an out-of-sample test.** The reduced model was
run for all 4,028 of 8,845 annotated cell types with any synapse onto the
core. It predicts that **113 types ignite the state**: 45 projection-neuron
types, 41 olfactory receptor types, a few local-neuron, hygro- and
thermosensory types, and four from outside the antennal lobe. Predictions
were frozen in `predictions-atlas.json`, then checked in the full brain on a
seeded random sample: 5 predicted to ignite, 5 predicted not to (drawn from
the quartile of non-igniting types with the most synapses onto the core),
plus every surprising ignition outside the antennal lobe.

**13 of 14 correct.** All ten of the random sample were right. The
surprising ones held for CSD (a pair of serotonergic neurons), LB3b
(gustatory) and AN_LH_AVLP_1 (ascending). The miss was **PPL202**, a pair of
dopaminergic neurons, predicted to ignite and silent in the full brain.
The likely reason, and the next test: the reduced model keeps no neuron
outside the core, so it drops inhibition the rest of the brain feeds back
onto the core. A core-plus-inhibitory-neighbours model should get PPL202
right without breaking the other 34.

Scripts and frozen predictions are tracked in `research/attractor/`.

**7. One state, whatever ignites it.** Every trigger produced the same
attractor: 9,960-10,039 neurons and about 435,000 spikes per 500 ms. It is
all-or-none, and its content does not depend on the input that started it.

### What this means

- For the model: a whole-brain LIF connectome model with no adaptation has
  a single global attractor anchored in the antennal lobe, which any strong
  enough antennal-lobe input ignites and nothing ends. Any continuously run
  version of Shiu et al. 2024 meets it the moment an odour or a temperature
  input is driven for long.
- For the theory: a static structural rule found a 477-neuron set that is
  causally necessary for a 10,000-neuron state, at 0.34% of the brain, where
  degree-matched random sets did nothing. The same rule, run as a forward
  mean field, failed to predict ignition, and a weight cutoff chosen by hand
  failed half its predictions. The structure tells you **where** a
  self-sustaining state lives. Whether a given input starts it needs the
  threshold measured, not guessed.
- For GLaDOS: the kernel's brain needs adaptation, and the right target is
  now known. Adaptation has to make this core's self-excitation subcritical,
  not the whole brain's.

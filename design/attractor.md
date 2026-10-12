# A heat-ignited attractor in the whole fly brain, and a structural theory for it

**Status: an observation and a hypothesis, 2026-10-12.** Nothing here is
established yet. The first section is measured. Everything after it is what
tomorrow's work is meant to test.

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

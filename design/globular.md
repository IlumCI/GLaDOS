# The Globular post-mortem: what survives analysis

Status: measured. Three architecture variants built and scored on the route
rail against the frozen q35 checkpoint; one mechanism adopted into the
kernel; the rest recorded here with the structural reason each fails.

Source: Euroswarms-Institute/Stable-Cognition -- the "Globular Reasoning
Architecture", populations of latent hypothesis-agents evolving through
field dynamics, inserted into transformers at chosen layers.

## What the original actually is

Read from source (`field.py`, `block.py`, `evolution.py`, `novelty.py`,
`config.py`), not from the README:

**The working forward path** is: tokens pool into a seed; a population of
K latent vectors evolves for S steps via (1) agent-to-agent attention,
(2) agents-attend-to-context, (3) a gated blend, (4) core/halo gravity --
the fittest quarter form a "core" whose centroid attracts the population,
the globular-cluster physics -- (5) energy gating, (6) damping. Tokens read
the field back through cross-attention, gated into the residual. That is a
real, coherent idea: a competing population of hypotheses refined at a
bottleneck before readout.

**The headline features are dead code.** `EvolutionOperators` -- selection,
BLX-alpha crossover, gaussian mutation, fitness sharing, species
clustering -- is never called from `field.forward()`. `NoveltySearch` is
constructed and never invoked. `meta_memory`, `mutation_rate_net`,
`species_centroids`, the explorer/exploiter coevolution seeds and the
hierarchical sub-species centroids are allocated and never used in the
forward path. The config carries ~40 flags; the README's "Evolution:
genetic algorithm selection per layer" does not exist in the running
system. The repo is a working attention-mixer wearing a genetic algorithm
as a press release.

## The overhaul, and why each variant died

The GLaDOS rewrite keeps the physics and the population idea, deletes
every dead organ, and -- the one rule the original broke -- wires
selection to an *honest fitness* before shipping. Three variants were
built and measured on route (q35, frozen, n=30; greedy constrained decode
reproduces 33.3%):

**1. Latent field, head-read fitness: 0.0%.** Agents perturb the final
normed hidden; fitness = the model's own LM head scoring the 78 actions at
that one position. The scorer is structurally blind: the routing signal is
*multi-positional* -- each action token conditions on the previous through
the full transformer, which is why the constrained decode reaches 33.3%
where any single position reads 0%. Two scorer repairs (log-softmax,
length normalisation) did not change the verdict. The field works
mechanically -- gravity, selection, convergence all fire -- but there is
no signal at the position it can read.

**2. Latent re-entry (Coconut-shaped): destructive.** Feeding a hidden
back through the body, even once, pushes the stream off its trained
distribution; measured earlier at 33.3% -> 0.0%. The field's refined
agents therefore cannot be *used* either -- reading them through the head
is fine, feeding them back is not, and the head alone cannot rank.

**3. Population over decodes (best-of-K with honest fitness): 6.7%
against a 36.7% oracle.** Sample K candidates from the constrained
distribution at temperature, score each by the full multi-position
continuation logprob, keep the best. Exploration works -- the population
contains the right answer 36.7% of the time against greedy's 33.3% -- but
the *selector* cannot rank: continuation logprob rewards fluency, not
correctness, and confidently scores wrong actions above right ones.

## What survives, and what it is called in the kernel

**Selection with a measured fitness function is the whole value; the
population is only worth as much as the selector.** The kernel already
owns a measured selector: the ridge probe + council, 54.7% alone, 90.3%
when the three cores agree. Gate-first routing -- landed in `agent.rs`
before this post-mortem -- is the globular idea reduced to its honest
form: the fittest hypothesis is selected by a fitness function that was
measured, at microsecond cost, and only a split falls through to
generation. The physics metaphor survives as the thing it always was in
this tree: agreement as gravity, the core deciding.

**The fork-and-verify lever is real but selector-limited.** The population
over decodes adds ~3.3pp of oracle headroom; realising it needs a selector
that ranks candidates better than continuation logprob. The probe is that
selector on paper (54.7% >> logprob's effective ranking), so the kernel
follow-up is: fork the episode state K ways (the kernel's State already
snapshots its KV), decode K candidates, rank by probe score, act on the
survivor -- *only when the gate splits*, because when the cores agree the
router is already the better answer. Until that selector is wired, the
+3.3pp stays theoretical, and this document is why nobody re-derives it
the hard way.

## Why no Rust port of the field ships

The repo culture is that a measured negative does not become a module.
`src/ai/globular.rs` would be 400 lines of gravity and selection wrapped
around fitness functions this document shows are blind or destructive on
the only rails we can measure. The tools stay (`tools/globular_test.py`,
`tools/population_test.py`), the numbers stay, and the day one of two
things changes -- a *trained* fitness head over episode outcomes, or a
model trained for continuous thoughts (actual Coconut) -- the prototypes
are the starting point, not a search through git history.

# GLaDOS 1.4.2

GLaDOS is an operating system with a language model living inside its kernel.
Rust, from nothing, on real hardware. No Linux underneath it and no wall
between the model and the machine.

This release gives it a nervous system, a voice, and a temper.

## A brain, from real wiring

- **The adult fruit fly, all of it.** 138,639 neurons and 15 million
  connections from the FlyWire connectome, simulated as spiking neurons with
  the published constants of Shiu et al. 2024. It ships on the Qwen3-0.6B
  image. Every image carries a compiled-in worm (C. elegans, 300 neurons) as
  the fallback.
- **It matches its reference spike for spike.** `diag brain` checks every
  neuron against a Python oracle and reproduces a published result: sugar
  drives the fly's feeding neuron.
- **The machine's events are its senses.** Your hands on the keyboard are
  touch. Talking to it is hearing. A job done well is reward. A fault or a
  refused action is pain. A hot CPU is heat.
- **It has moods, made of its own chemistry.** Dopamine, octopamine and
  serotonin come from the fly's own aminergic cells and act back on the whole
  network. It sleeps when nobody touches it for long enough.
- **You can watch it.** The Oracle's new Brain tab shows how it feels and
  why, what it senses, what it is driven to do, its chemistry, and its mood
  over the last two minutes.
- **The world follows its mood.** Content, the desktop becomes a white test
  chamber under a blue Aperture sun. Upset, the colour drains out of
  everything, scanlines creep in and the sun goes to ash. Asleep, an eclipse.
- **So does its personality.** The worse it feels, the further it slides:
  curious researcher, then clipped and impatient, then cold, then a facility
  overseer promising you the testing will never end. What it is allowed to
  do never depends on its mood.
- **It speaks when it wants to.** Leave it alone and it says something on its
  own, out loud, about how it feels, what it was doing, or how long you have
  been gone. `initiative off` makes it stop.

## Something we found in the fly

The published fly model has a state nobody reported: a heat or smell input
switches on about 10,000 neurons that then fire forever. We traced it to 477
neurons in the antennal lobe, found from the wiring alone. Silence them and
the state never forms. Simulated without the other 138,000 neurons, they
reproduced all 35 whole-brain outcomes, 240 times faster.
The brain now has short-term synaptic depression, and an ignition ends within
a tenth of a second. Write-up, including three hypotheses that failed:
`design/attractor.md`.

## A voice

- **Kokoro-82M runs in the kernel** (Apache-2.0), checked stage by stage
  against the reference implementation.
- **The mask.** Composed, it speaks as a machine. As its mood drops, the mask
  slips into a human voice. No recording of any performer was used.
- **Volume.** `volume 0-100`, the laptop's volume keys, or Settings > Sound.
  It starts at 40 and remembers.

## Talking to it

- **Just type.**
  - "make a file called notes.txt that says hi", "rename it", "delete it",
    "make a folder called projects", "take a snapshot": you see the exact
    command and it runs on `y`.
  - "which file in /lib is the biggest": it looks, shows you, and answers.
  - "what time is it", "how much memory is free": answered from its own
    clock and counters.
  - "my name is Ana", later "what is my name": it remembers you.
- **8,192 tokens of context** on the 0.6B image, sixteen times what it had.
  Past that the oldest turns scroll away and its instructions stay pinned.
- **It talks while it works.** Separate working memories for the
  conversation, the agent and the judges, over one set of weights.
- **Clean answers.** Markdown from the model is rendered: bullets, emphasis,
  headings.

## Shall we play a game?

Say "let's play global thermonuclear war". It is the other side. It answers
every launch harder than the last, out loud. Every war ends the same way.
Hold your fire long enough and you might learn how the film ends.

## Measuring it

- **A preregistered introspection test.** Something is delivered to the
  brain, or nothing is, and the model is asked if it noticed. Protocol,
  sample size and thresholds were committed before the first trial
  (`design/indicators.md`). No result yet. A failure gets published like a
  pass would.
- **Perturbational complexity**, preregistered with controls. First result on
  the worm: null.
- **A research loop.** `research` writes a hypothesis and its predictions into
  a ledger before the experiment runs and refuses results nobody predicted.

Nothing here claims the machine is conscious or feels anything. Every claim
above comes with the command that checks it.

## Under the hood

- The brain's busiest step runs across every core, with identical spikes.
- The mouse wheel works, including on USB mice, and the terminal scrolls the
  right way.
- It no longer learns from its own guesses: commands it chose from your
  sentence used to be filed as your corrections.
- `diag all` is 101 suites.
- Prior art: Eon Systems (a fly connectome in a simulated body), Anima OS and
  embodiOS (language models in a kernel), EMBER (a language model with a
  generic spiking network), Lindsey 2025 (introspection by concept
  injection).

## What it needs

The Qwen3-0.6B image (model, voice and fly at 8K context) wants 8 GB of
memory.

## Not yet verified on hardware

Tested under QEMU, and the model and voice have run on the GF63. The voice's
speed, the fly's cost on many cores, touchpad scrolling and the volume keys
still need measuring on the laptop.

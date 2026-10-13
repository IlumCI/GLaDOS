# GLaDOS 1.4.2

GLaDOS is an operating system with a language model living inside its kernel.
It is written from nothing in Rust, it boots on real hardware, and there is no
Linux underneath it and no wall between the model and the machine. A thought the
model has is a function call.

This release gives it a nervous system, a voice, and a way of working that
starts to look like research. A real animal's brain, wired neuron for neuron
from electron microscopy, now runs inside the kernel beside the model. It is fed
the machine's own events as senses, and its state reaches the model, the
desktop and the voice. The machine also got the beginnings of a research loop
that writes its predictions down before it tests them.

## A brain, from real wiring

- **The adult fruit fly, all of it.** 138,639 neurons and 15 million
  connections from the FlyWire connectome (v783), stepped as spiking neurons
  with the published constants of Shiu et al. 2024. It ships on the Qwen3-0.6B
  image; every image carries a compiled-in worm (C. elegans, 300 neurons)
  as the fallback.
- **It matches its reference spike for spike.** A Python simulator with the
  same integer arithmetic is the oracle, and `diag brain` requires the kernel
  to agree with it exactly, every neuron's spike count, and to reproduce a
  published result: sugar-sensing neurons drive the fly's feeding motor
  neuron.
- **The machine's events are its senses.** Keyboard and mouse are touch,
  being spoken to is hearing, a good outcome is sugar, a fault or a failed
  action is a bitter taste, heat near the CPU's limit is heat.
- **It has moods, made of its own chemistry.** Dopamine, octopamine and
  serotonin levels are read from the fly's own aminergic cells and act back
  on the whole network. The brain's approach, avoidance and arousal become
  two numbers -- valence and arousal -- which the conversation, the resident
  mind and the voice read. `brain` shows all of it.
- **It sleeps.** Pressure builds over hours awake; once it is high and nobody
  has touched the machine for a while, it rests, and only touch or speech
  wakes it.
- **You can see it without a window.** The whole desktop follows its mood.
  Content, and the room lights up into a white test chamber with a blue
  Aperture sun. Upset, and the colour drains out of the world: monochrome,
  darker, scored with scanlines, the sun gone to ash. Asleep, an eclipse --
  a black aperture in a pale corona. Aroused, the sun climbs.
- **And it changes who you are talking to.** The deeper its mood sinks, the
  further the machine's manner goes: from a precise, curious researcher,
  through clipped impatience and cold sarcasm, to a facility overseer who
  promises you the testing will never end. The same goes for what it says
  unprompted -- it speaks on its own, not only when spoken to. It is
  character only: what it may *do* is decided elsewhere, and never by its
  mood.
- **It modulates and never grants.** Nothing read from the brain decides
  whether an action is allowed, and every judge the machine uses on itself
  reads it as neutral, so no verdict depends on its mood.

## A finding about the fly brain itself

Running the whole fly continuously turned up something nobody had reported:
in the published model, a heat or smell input switches on a state of about
10,000 neurons that then fires forever. We traced it to its cause and tested
the explanation like a hypothesis, with predictions written down before each
run:

- **A rule computed from the wiring alone found the 477 neurons that hold it
  up** -- 0.34% of the brain, in the antennal lobe. Silence them and the state
  never forms; silence 477 random neurons with the same connectivity and
  nothing changes.
- **The 477 alone decide it.** Simulated without the other 138,000 neurons,
  they reproduced every one of 35 whole-brain outcomes, in a second instead of
  four minutes, and predicted which of 4,028 cell types can switch the state
  on: 13 of 14 out-of-sample checks right, and the remaining one turned out to
  be chance near the threshold.
- **So the brain now has short-term synaptic depression** (synapses tire when
  used hard, as real ones do). A heat input still ignites it, and the
  ignition now ends within a tenth of a second. The full write-up, including
  three hypotheses that were wrong, is `design/attractor.md`.

## A voice

- **Kokoro-82M runs in the kernel** (Apache-2.0), verified stage by stage
  against the reference implementation, with misaki's pronunciation lexicon.
  It is on the 0.6B image; the others keep the built-in formant voice.
- **The mask.** Composed, it speaks as a machine: one held note, no breath.
  As the brain's mood drops, the mask slips into a human voice. No recording
  of any performer was used for any of it.

## Shall we play a game?

Say "let's play global thermonuclear war". The machine is the other side:
pick the United States or the Soviet Union, strike or hold, and it answers
every launch harder than the last, out loud, in its own voice. Every war
ends the same way. Hold your fire long enough and you might find out how the
film ends.

## Talking to it

- **Just type.** Plain language at the prompt is acted on: a confident,
  read-only request runs, anything that changes something asks first, and
  anything else becomes a conversation. `ask` and `agent` are no longer
  needed in front of a sentence.
- **It remembers more of the conversation.** The 0.6B image now holds 8,192
  tokens of context, sixteen times what it had. Past that, the oldest turns
  scroll away quietly while its instructions stay pinned -- the conversation
  never stops.
- **It talks while it works.** The model has separate working memories for
  the conversation, the agent and the judges, sharing one set of weights, so
  a question asked in the middle of a long task is answered in the middle of
  it, and the task carries on where it was.

## Measuring instead of claiming

- **A preregistered introspection test.** Something is delivered to the
  brain, or nothing is, and the model is asked whether it noticed -- with no
  text telling it, through a direct injection into its activations. The
  protocol, sample size and thresholds were committed before any trial ran
  (`design/indicators.md`). It has not produced a result yet, and when it
  does, a failure is published the same way a pass would be.
- **Perturbational complexity** (the measure used on brains under
  anaesthesia), preregistered with controls. On the worm the first result is
  null: the real wiring scored no higher than the controls.
- **A research loop, started.** `research` preregisters a hypothesis --
  a claim and predictions over measured numbers -- in a ledger before the
  experiment runs, refuses a result nobody predicted, and keeps "gave up"
  apart from "shown impossible". It runs by hand this release; running it on
  its own is the next one.

Nothing here claims the machine is conscious or feels anything. What it
claims is the list above, each with the command that checks it.

## Under the hood

- The brain's busiest step now runs across every core, exactly: the same
  spikes, the step about three times faster on four cores when most of the
  fly is active.
- `diag all` is 100 suites.
- Prior art, named: Eon Systems (a fly connectome in a simulated body, 2026),
  Anima OS (a language model in a kernel), EMBER (a language model with a
  generic spiking network), and Lindsey 2025 (introspection by concept
  injection).

## What it needs

The Qwen3-0.6B image -- model, voice and fly together, at 8K context -- wants
8 GB of memory to be comfortable: about 750 MB of the kernel heap at boot,
and nearly 1.9 GB once every self-test has run.

## Not yet verified on hardware

Everything in this release was verified under QEMU. The voice's speed, the
brain's cost on many cores and the audio path on real laptops are still to be
measured on the machine itself.

# Multi-agent workflows

## What this has to work around, stated first

Three facts decide the shape of this before any design begins. Each is
measured or is in the source, and none of them is negotiable by writing better
code.

**One engine, one holder.** `HOLDER` is a single `AtomicUsize` carrying a task
id. Two `&mut Engine` at once is undefined behaviour, and a decode interleaved
between two of somebody else's calls corrupts the KV cache, `pos` and
`last_token`, which produces confident nonsense rather than an error. So
concurrent inference is not available. Not slow, not awkward: unavailable.

**One queue, one busy flag, one task, on purpose.** `agent::Job` is either an
episode or an authoring run and both run on the resident agent task. A second
task running a second kind of work needs a second entry in the engine's
exclusion check, which is the stale-call-site failure that check's own comment
warns about, and there the failure is two forward passes interleaving in one
cache.

**Ensembling for accuracy was measured here and does not work.** On 108
independent items the best single core scores 77.8% and an equal-weight
product of three scores 76.9%. `council.rs` opens with that number and with
what to do instead.

The last one matters most, because it removes the usual motive. A plan that
proposes several agents voting for a better answer is proposing the experiment
this tree already ran and lost.

## What already works, and what it proves

The system runs a three-agent ensemble on every routing decision and has for
some time. A semantic ridge probe, a lexical naive Bayes over token identity,
and a character Bayes over hashed byte trigrams. They do not combine into an
answer. The probe answers, being strongest alone, and the other two
corroborate.

**Their agreement is the product.** Where all three pick the same applet they
are right 90.3% of the time; where they split, 50%. That gap is what `gate`
acts on, and it is worth more than the point or two an ensemble was supposed
to buy: a router that knows when it is guessing can ask, escalate or refuse.

`godel`'s judges are the same pattern on a different subject. J1 through J4 are
four independent evaluators, each admitted because a *different* thing can go
wrong, and adoption needs unanimity. Nobody averages them.

So this codebase already has two working multi-agent systems, and both earn
their place by **disagreement being informative**, never by aggregation
improving an answer.

## What an agent costs, so the design can be honest about scale

| | |
|---|---|
| One Aiksi step | 13.7 ns |
| One council vote, whole | ~2.9 us |
| `Interp::new` after the `KERNEL_RECS` fix | 22 ns |
| A model decode step | milliseconds, and it holds the engine |
| `ctx_save` of a 512-slot cache | thousands of store blocks |

The spread is five orders of magnitude. A deterministic agent is free at any
count that matters. A model call is the scarce resource, and swapping which
conversation the engine holds is worse than a model call.

**That inverts the usual scheduling instinct.** The goal is not to overlap
work. It is to spend as few model calls as possible and to change context as
rarely as possible, doing all of one agent's model work before any of the
next's.

## The shape: agents are roles in a pipeline, not threads

No new task, no second queue, no second engine holder. A workflow is a
declared sequence of roles executed by the one agent task, and the parallelism
that exists is among the roles that need no engine.

    goal
      -> decompose        (model, once)
      -> plan             (deterministic)
      -> N workers        (deterministic, independent, cheap)
      -> N critics        (deterministic, independent, cheap)
      -> agreement        (deterministic)
      -> escalate?        (model, only when the critics split)
      -> judge            (deterministic)
      -> keep or discard

The model appears twice: once to turn a goal into structure, and once more
only when the cheap agents disagree. Everything between is Aiksi and cached
features, which is where the five orders of magnitude are.

## Determinism, and the rule that follows from the judges

An agent is deterministic when running it twice on the same inputs gives the
same value, the same step count and the same set of objects touched. That is
already `differ`'s definition, and `skill::bench`'s J3 already tests exactly
those three things.

Determinism is not a preference here. `godel` rests on re-derivability: a
verdict nobody can re-derive is a verdict nobody can check, which is why the
search walks a declared grid instead of tossing a coin. An agent whose output
cannot be reproduced cannot be judged by replay, only by inspecting its result.

**The rule: a non-deterministic agent's output must be consumed by a
deterministic one before anything acts on it.** Two non-deterministic agents
in sequence with no check between them is a pipeline whose result nobody can
account for, and it is the one arrangement to refuse outright.

| role | kind | why |
|---|---|---|
| decompose a goal | non-deterministic | needs the model; nothing else can |
| extract a fact from a document | non-deterministic | same |
| route, vote, judge | deterministic | closed-form or counting |
| workers over a declared list | deterministic | Aiksi, budgeted, sandboxed |
| critics | deterministic | must be, or the agreement signal is noise |
| the schedule itself | deterministic | see below |

**The schedule has to be deterministic or the whole thing is unrepeatable.**
Which role runs next must be a function of what has been recorded, the way
`godel::next_proposal` takes the first kind from the ledger position that has
work. A workflow that picks its next step by sampling is one that cannot be
re-run to check a result.

## Where the parallelism actually is

Three places, and only the first is available today.

**Deterministic agents are independent by construction.** They take values and
answer values, they are sandboxed into their own subtree, and they carry a step
budget. Running twenty of them is twenty times 2.9 us. There is no scheduling
problem to solve because there is nothing to contend for.

**`smp::parallel_split` exists and is the wrong tool.** It answers false below
2^19 element-operations, it is for one matvec at a time, and application
processors run on the trampoline's flat descriptor table with no TSS, so they
never take an interrupt and never run a task. Splitting agents across cores
needs a per-core GDT and TSS, which `smp.rs` already prices.

**Cooperative multi-agent on the bootstrap processor is the reachable one.**
Running agents cooperatively without a timer needs neither a per-core GDT nor a
TSS, and `smp.rs` names it as the shorter road. It buys nothing for
deterministic agents that are already microseconds, so it is only worth
building if something slow and non-model appears.

## Stages

### Stage 0. Name the thing that already exists

`council.rs` and `godel`'s judges are multi-agent systems that nobody calls
that. Before adding a framework, give the existing pattern a name and an
interface: a set of independent evaluators, a rule for combining their
verdicts, and agreement reported separately from the answer.

Deliverable: a `Panel` in `src/ai/`, built from the council's shape, that any
decision can be routed through. Cost: refactoring, no new mechanism. This is
the stage that decides whether the rest is worth building, because if the
existing pattern does not generalise cleanly then a bigger one will not either.

### Stage 1. A declared workflow, run by the agent task

A workflow is data: a list of roles, each naming an agent and what it consumes.
Stored content-addressed like everything else, so a run can be re-derived from
its address.

- The schedule is a pure function of the workflow and what has been recorded.
- Every step's inputs and outputs go in a ledger, the way `godel` writes one
  line per trial whether or not it adopted.
- The engine is claimed once for the whole run with `claim_engine()`, so no
  other task can interleave a decode into the middle of it.

Verification is `differ`'s: run the same workflow twice and require agreement
on value, step count and objects touched for every deterministic role. With a
canary, because a harness that has never reported a difference is
indistinguishable from one that compares nothing.

### Stage 2. The first workflow worth having

Reading. `fetch` and `save` exist now, `curiosity` picks topics, and the
obvious pipeline is:

    pick a topic          deterministic, the frontier
    fetch it              deterministic, one HTTPS call
    extract claims        MODEL, once
    check each claim      deterministic, against what is already in /ai/read
    agree?                deterministic
    keep or flag          deterministic

One model call for a document. Everything else is namespace reads and string
work. That is the ratio the whole design is aiming at, and it is measurable
against the alternative of asking the model at every step.

### Stage 3. Escalation, and only then

Let the panel's disagreement decide whether to spend a model call. This is the
gate pattern already proven on routing, applied to a workflow: cheap agents
answer, and the expensive one is consulted only where they split.

The number to report is not accuracy. It is **model calls per completed task**,
against a baseline that asks the model at every step. If that ratio is not far
below one, the design has failed and should be said to have failed.

## What to measure, and what would count as failure

| | |
|---|---|
| Model calls per completed task | the headline; the whole point is that it is small |
| Wall clock per task | against a model-at-every-step baseline |
| Panel agreement against correctness | must reproduce the 90.3 / 50 shape, or the panel is noise |
| Determinism | two runs, identical value, steps and touched objects |
| Engine held, total | a workflow that holds it longer than the episode it replaced is worse |

**The failure that matters is a panel whose agreement predicts nothing.** The
existing council earns its place on a 40-point gap. A new panel that shows five
points is three agents' worth of cost buying noise, and the honest response is
to delete it rather than to tune it until the gap looks better on the set it
was tuned on.

## Out of scope, with reasons

- **Concurrent inference.** One engine, one cache. Not a limitation to route
  around.
- **Agents as tasks.** One queue and one busy flag is a decision with a
  recorded reason, and a second task is a second entry in an exclusion check
  somebody will forget.
- **Agents on other cores.** Needs a per-core GDT and TSS. Priced in `smp.rs`,
  and it buys nothing for agents that already cost microseconds.
- **Ensembling for a better answer.** Measured at 77.8 against 76.9. Reopen
  only with a task where the cores' errors are genuinely independent, and
  measure before building.
- **A workflow language.** Aiksi is the language. A workflow is a list of
  roles, and inventing a second syntax to describe that list is how a small
  mechanism becomes a project.
- **Anything non-deterministic scheduling anything.** See the rule above.

## The honest summary

This system already does multi-agent work and does it well, in the two places
where independent evaluators earn their keep. What it does not have is a way to
compose them into a sequence with a ledger and a re-derivable schedule.

That is the gap worth filling, and the prize is not parallelism. The engine
cannot be shared and the deterministic agents are already free. The prize is
**spending the model rarely**, which is the only resource here that is
genuinely scarce, and having a record afterwards that says what each agent
contributed.

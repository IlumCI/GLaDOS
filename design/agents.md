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

### Stage 1. A declared workflow, run by the agent task  (done)

A workflow is data: a plan tree in the namespace, re-read at every step. The
engine is claimed once for the whole run with `claim_engine()`, so no other
task can interleave a decode into the middle of it.

The worker is greedy throughout. `choose` at temperature 0 takes the argmax and
`decode_args` always did, so the same goal against the same state produces the
same action and the same argument string. It deliberately does not go through
`agent::propose`, which forks three ways through `deliberate`: that is the
better router for an operator asking once, and a fork is exactly what stops the
same question getting the same answer twice.

#### What this stage got wrong, and what replaced it

It was written requiring **the graph root hash to be identical across two
runs**. That cannot hold, and the experiment said so: two runs of one plan
produced different roots.

The cause is worth keeping. A run writes its steps under `/ai/work/<run>`,
which is inside the `/ai` a worker then lists, so the second run legitimately
saw a directory hash the first had changed. The decision was identical both
times. The observation was not, and could not be.

**So a step records two things and only one of them is re-derivable.** `action`
is what the worker chose and is what two runs must agree on. `observation` is
what the world answered and is allowed to differ. `work cmp <a> <b>` reports
them separately, because decisions differing is a defect and observations
differing is the world having moved.

Measured on two runs of one plan:

    steps        1 and 1
    decisions    agree -- the workflow is re-derivable over these inputs
    observations differ, which is the world moving between runs

The root hash is still exact and still worth having. It is a complete statement
of what a run produced, which is a different claim from determinism, and
conflating the two was the error.

### Stage 2. The first workflow worth having

Building something in the namespace, and only that. Research was the other
candidate and is dropped, for a reason worth recording: reading needs `fetch`
and `save`, which carry the `net` bit and live behind the token gate, so a
workflow built on them could not be a stable feature. Building needs Aiksi,
`author.rs` and `skill.rs`, all of which ship in stable today.

    read the plan node        deterministic, a namespace read
    write or amend a program  MODEL, once per revision
    run it sandboxed          deterministic, budgeted, already exists
    judge it                  deterministic, skill.rs already does this
    keep or revise            deterministic, from the judges' verdict

One model call per revision. Everything else is a namespace read, an Aiksi run
and four judges that already exist and already work. That is the ratio the
whole design aims at, and this is the workflow where it is easiest to reach,
because the expensive half is already written.

The loop it makes is the useful one: propose, run, judge, revise. A worker that
fails leaves a summary node saying why, and the next revision reads it.

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
- **Research and synthesis workflows.** They need `fetch` and `save`, which
  carry the `net` bit and sit behind the token gate, so anything built on
  them cannot ship as a stable feature. Reopen on the experimental channel
  once the building workflow has soaked.
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

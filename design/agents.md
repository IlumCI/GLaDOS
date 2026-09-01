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

### Stage 2. The manager  (done, with the quality limit measured)

`work plan <run> <goal>` decomposes into actions and writes them into the plan;
`work run` then executes them.

**The saving is prefill, not cleverness.** `agent::prompt_for` includes every
prior step, so an episode re-encodes a growing prompt at each step and spends
O(N^2) tokens over a run. `harness::plan_actions` encodes once and leaves the
engine positioned, so N actions cost one prefill and N decodes: O(N). And
because the manager writes what it decided, the workers that follow decode
nothing at all.

Measured, four-step goal, Qwen3-0.6B:

    work plan   4 step(s) planned, 4 decode call(s)
    work run    4 step(s) ran,     0 model call(s)

Against the baseline, where a worker decides its own step, at one decode for
the applet plus one for arguments, each behind its own growing prefill.
**Execution is free. That is the whole result.**

#### The quality limit, which is the honest half

The mechanism works and the plans are mediocre. Three findings, in the order
they were found, because each changed the next:

**The prompt has to demonstrate, not describe.** "One tool per line, then
done" in words was ignored by both checkpoints: SmolLM2-135M picked one tool
four times with no arguments, and Qwen3-0.6B answered `done` on the first
decode and planned nothing. A two-line worked example took Qwen3-0.6B from
zero steps to four, the first of them correct.

**A separator after the applet name is load-bearing.** The grammar consumes
the name exactly, so without a space the model continues mid-word. Every step
of the first run planned `find - - - -`. `decode_args` never had the bug
because its prompt ends `"name "`.

**What is left is real and is not plumbing.** At 0.6B the planner copies the
worked example's argument rather than the goal's, and once decoded `cat done`,
which is the model reaching for the stop word where an argument goes. Neither
is a bug in the mechanism, and neither is fixed by tuning the prompt further
against the same two checkpoints, which is how a number gets better on the set
it was tuned on.

That is what Stage 3 is for. Role adapters trained on real transcripts and
judged by `godel` are the answer to plan quality, and they are the reason the
stages are in this order rather than the other one.

#### The building workflow

    read the plan node        deterministic, a namespace read
    write or amend a program  MODEL, once per revision
    run it sandboxed          deterministic, budgeted, already exists
    judge it                  deterministic, skill.rs already does this
    keep or revise            deterministic, from the judges' verdict

Research was the other candidate and is dropped: reading needs `fetch` and
`save`, which carry the `net` bit and live behind the token gate, so a workflow
built on them could not be a stable feature.

### Stage 3. Role adapters, trained and judged  (done, and the answer is no)

`work harvest` turns transcripts into per-role example sets. `work train
<role>` trains an adapter from one and puts it through `godel`'s four judges.
An adapter that passes is stored under its role, attached around that role's
decodes, and detached after.

All of that works. The measurement it exists to take says roles are a naming
convention, and the reason is structural rather than a shortage of data.

#### What was measured

24 workflows, SmolLM2-135M, read-only trust, one worker-decided step each:

    work harvest      24 step(s) across 24 run(s)
                       1 dropped, the step failed
                      23 example(s) under the role "worker"

    work train worker 23 example(s) from 23 run(s), 18 trained and 5 held
                      47 decision(s), 10 of them held out, over 2 applet(s)
                      trained    base 97.3%
                      held-out   base 100.0%  role 100.0%
                      paired     fixed 0  broke 0  chi 0.00
                      J1 FAIL (no net repair)   J2 pass   J3 pass   J4 pass

#### Why more transcripts would not change it

**A harvested label is the base model's own argmax.** `choose` decodes the
applet name at temperature 0 under the grammar, so the action written into a
transcript is what the classifier already ranks first. Training on that set
asks the adapter to reproduce whatever produced it, and the paired test finds
nothing in either cell because there was no disagreement to find. The 97.3% on
the training slice is the only gap there is, and it comes from the trial
scoring under the whole applet table while the label was decoded under the
read-only subset.

**The `ok` filter does not escape it.** Keeping only successful steps was
meant to make the target distribution differ from the source. It fails to,
because `ok` records that the applet ran and says nothing about whether it was
the right applet, so it drops examples while agreeing with every label that
survives.

**The label set was degenerate as well.** 23 examples over two applets: a 135M
model answered two things to 24 different goals. `train_role` refuses a
one-class set outright now, because an adapter over one class learns a prior
and scores 100% doing it, which is the one outcome that would look like
success and mean nothing.

#### What would have to be true instead

A label from something other than the model being trained. Three sources exist
in this tree already:

- `teach`, an operator naming the right applet, which is how `/ai/train` was
  built in the first place.
- A judged outcome in place of a bare `ok`: whether the observation answered
  the goal. `skill.rs`'s judges are the pattern for it.
- A larger model's choice as the teacher, which is the distillation direction
  that works.

A transcript is none of those, and that is the finding.

#### What is kept, and why

The negative result is the deliverable, and the mechanism is what makes it
checkable by anybody who doubts it. Three parts stand on their own.

**The split is by run and never by step.** Steps inside one run share a goal
and differ by slot values, so a step split measures memorisation while looking
like generalisation. It is the same argument the routing corpus makes for
holding out whole template families.

**The judges are called rather than copied.** `godel::mcnemar` and
`godel::sanity` were made public, so a role adapter is judged by J1 itself. A
second implementation would judge it by whatever that copy had drifted into,
which is the objection `model.rs` makes twice about two implementations that
are supposed to agree.

**There is no ledger and no rollback, deliberately.** `godel` adopts by
swapping a head pointer, which puts one adapter in front of every decision the
machine makes. A role adapter adopted that way stops being a role. It goes on
around its own role's decodes and comes off after, so declining to store one
leaves the machine exactly as it was and leaves nothing to undo.

And one economy worth naming: the adapter is attached only around a step the
worker actually decides. A pre-decided step decodes nothing, so a swap around
one pays for a specialist to read out somebody else's decision.

### Stage 4. Autonomy, declared per workflow

A workflow declares whether it may run unattended. Unattended runs at
`Trust::ReadOnly`, attended may be `Full`. The declaration is part of what is
judged before a workflow is allowed to run on its own, which is the shape
`app::manifest`'s `raw` bit and `skill trust` already use.

Escalation belongs here rather than in a stage of its own. Letting a panel's
disagreement decide whether to spend a model call is the gate pattern already
proven on routing, and the number to report for it is model calls per completed
task against a baseline that asks the model at every step. Stage 2 already
drove that ratio to zero for pre-decided steps, so what is left for escalation
is deciding when a plan needs re-planning, which is a question about autonomy.

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

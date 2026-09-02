//! The graph a workflow runs on.
//!
//! A manager delegating to workers needs somewhere to put what has been
//! decided and what came back. The obvious place is a conversation, and it is
//! the one place this kernel cannot afford: `ctx_save` on a 512-slot cache
//! measured in thousands of store blocks, so swapping a context per worker
//! costs more than the work.
//!
//! So memory is a graph in the namespace instead, and a context switch becomes
//! a namespace read.
//!
//! ### Why the namespace rather than a structure of our own
//!
//! It is already a content-addressed Merkle tree of nodes and sub-nodes, which
//! is the thing that would otherwise have to be built. Using it inherits
//! deduplication, constant-time copy, `same` comparing whole subtrees in one
//! step, and `snap` versioning the entire graph as one root hash. Two identical
//! summaries share an address by construction rather than by anybody
//! remembering to check.
//!
//! The property that matters most is the last one. **A run's root hash is a
//! complete statement of what the run produced**, so two runs of one workflow
//! over the same inputs can be compared exactly, for free, by one comparison.
//! That is what makes a workflow re-derivable in the sense `godel` means, and
//! it is the reason this module stores rather than computes.
//!
//! ### Layout
//!
//! ```text
//!   /ai/work/<run>/plan            the plan tree, re-read at every step
//!   /ai/work/<run>/steps/NNNN      one summary per completed step
//!   /ai/work/<run>/artifacts/<n>   what is being built
//! ```
//!
//! Nothing here runs the model. That is deliberate: this is the half that can
//! be tested without an engine, and finding out whether the namespace is a
//! comfortable graph store is cheaper before any worker exists than after.

use crate::sysbox;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const ROOT: &str = "/ai/work";

/// Steps per run, bounded by the width of the name.
///
/// Names are zero-padded to four digits so that sorted order in the directory
/// is insertion order, which is what makes `steps()` return a sequence rather
/// than a set. `vocab::record` uses the same trick and silently truncates past
/// its own limit; here the limit is refused, because there is no host-side
/// tool guarding this the way `dataset.py` guards the corpus, and a run whose
/// step 10000 sorts as step 0 would reorder its own history without saying so.
const MAX_STEPS: usize = 10_000;

/// Role example sets, and the adapter trained from each.
///
/// Deliberately not under `/ai/work`. `runs()` answers that directory's
/// children, so a role set living there would list as a run with no plan in
/// it, and the first thing anybody did with it would be to ask why.
const ROLES: &str = "/ai/roles";

/// Examples a role needs before a trial over it can mean anything.
///
/// Arithmetic rather than taste. J1 will not call a difference real below a
/// net repair of `godel::MIN_FIXED`, and `role_split` holds out about a
/// quarter, so a role with fewer than four times that has a validation slice
/// in which J1 cannot pass however good the adapter is. Refusing is better
/// than printing a number over a slice that was never large enough to carry
/// one.
const MIN_ROLE_EXAMPLES: usize = 4 * super::godel::MIN_FIXED;

/// Grants: which plans the operator approved to advance unattended.
///
/// One hex line per approved intent, the shape `/app/grants` uses. Not under
/// `/ai/work`, for the same reason the role sets are not: `runs()` answers
/// that directory's children.
const GRANTS: &str = "/ai/autonomy";

/// Steps a plan may hold and still be granted unattended.
///
/// An unattended workflow advances one step per quiet tick, and a tick is
/// hourly inside a four-hour window. So a plan of a few hundred steps is
/// months of nights, which is nothing anybody declared when they granted it.
const MAX_UNATTENDED: usize = 64;

// --- the plan -------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
pub enum Status {
    Todo,
    Done,
    Failed,
}

impl Status {
    fn tag(self) -> &'static str {
        match self {
            Status::Todo => "todo",
            Status::Done => "done",
            Status::Failed => "failed",
        }
    }

    fn parse(s: &str) -> Option<Status> {
        match s {
            "todo" => Some(Status::Todo),
            "done" => Some(Status::Done),
            "failed" => Some(Status::Failed),
            _ => None,
        }
    }
}

/// One node of the plan tree.
///
/// A parent pointer rather than indentation. Indentation is ambiguous the
/// moment a goal contains a newline, and a goal is text a model wrote.
#[derive(Clone)]
pub struct PlanStep {
    pub id: usize,
    pub parent: Option<usize>,
    pub status: Status,
    /// Which worker role should take it. Empty until the manager assigns one.
    pub role: String,
    /// What the manager already decided, if it did.
    ///
    /// The whole economy of a manager is here. Empty means the worker decides,
    /// which costs a model call per step. Filled means the manager decided it
    /// during decomposition, and the worker is a dispatcher costing none.
    pub action: String,
    pub goal: String,
}

/// Whether a workflow may advance while nobody is watching.
///
/// Declared by the plan and granted by the operator, and it takes both. The
/// declaration alone cannot be the gate: a plan is a file, and a file can be
/// edited by anything that can write, so a workflow that granted itself
/// autonomy by writing a line into itself would have defeated the gate by
/// using it. That is the argument `app trust` and `skill trust` already make.
#[derive(Clone, Copy, PartialEq)]
pub enum Autonomy {
    Attended,
    Unattended,
}

#[derive(Clone)]
pub struct Plan {
    pub goal: String,
    pub autonomy: Autonomy,
    pub steps: Vec<PlanStep>,
}

impl Plan {
    /// The next step with nothing left to wait for.
    ///
    /// A function of what is written and nothing else, so the schedule is
    /// re-derivable: two readers of the same plan pick the same step. That is
    /// the same argument `godel::next_proposal` makes for taking its axis from
    /// the ledger rather than from a coin.
    pub fn next(&self) -> Option<&PlanStep> {
        self.steps.iter().find(|s| {
            s.status == Status::Todo
                && s.parent
                    .map(|p| {
                        self.steps
                            .iter()
                            .find(|q| q.id == p)
                            .map(|q| q.status == Status::Done)
                            .unwrap_or(false)
                    })
                    .unwrap_or(true)
        })
    }

    pub fn done(&self) -> bool {
        self.steps.iter().all(|s| s.status != Status::Todo)
    }

    fn render(&self) -> String {
        let mut s = String::from("work 1\ngoal ");
        s.push_str(self.goal.trim());
        s.push('\n');
        // Rendered only when it is not the default, which is the lesson
        // `Variant::render` records: a field written unconditionally
        // re-addresses every object that already exists, and here that would
        // change the root hash of every run ever stored.
        if self.autonomy == Autonomy::Unattended {
            s.push_str("autonomy unattended\n");
        }
        for st in &self.steps {
            s.push_str("step ");
            s.push_str(&st.id.to_string());
            s.push(' ');
            match st.parent {
                Some(p) => s.push_str(&p.to_string()),
                None => s.push('-'),
            }
            s.push(' ');
            s.push_str(st.status.tag());
            s.push(' ');
            // A role is one word by construction, so it needs no delimiter of
            // its own and the goal can take the rest of the line.
            s.push_str(if st.role.is_empty() { "-" } else { &st.role });
            s.push(' ');
            // Braced so an empty action is still a field and an action
            // containing spaces does not eat the goal.
            s.push('[');
            s.push_str(&one_line(&st.action).replace(']', ")"));
            s.push_str("] ");
            s.push_str(&one_line(&st.goal));
            s.push('\n');
        }
        s
    }

    fn parse(text: &str) -> Option<Plan> {
        let mut goal = String::new();
        let mut autonomy = Autonomy::Attended;
        let mut steps = Vec::new();
        let mut seen_magic = false;
        for line in text.lines() {
            let line = line.trim_end();
            if line == "work 1" {
                seen_magic = true;
                continue;
            }
            if let Some(rest) = line.strip_prefix("goal ") {
                goal = rest.to_string();
                continue;
            }
            if let Some(rest) = line.strip_prefix("autonomy ") {
                // Anything that is not the word reads as attended. The
                // conservative direction, and the only one available: a plan
                // whose declaration cannot be read is a plan nobody declared.
                autonomy = if rest.trim() == "unattended" {
                    Autonomy::Unattended
                } else {
                    Autonomy::Attended
                };
                continue;
            }
            if let Some(rest) = line.strip_prefix("step ") {
                let mut it = rest.splitn(5, ' ');
                let id = it.next()?.parse::<usize>().ok()?;
                let parent = match it.next()? {
                    "-" => None,
                    p => Some(p.parse::<usize>().ok()?),
                };
                let status = Status::parse(it.next()?)?;
                let role = match it.next()? {
                    "-" => String::new(),
                    r => r.to_string(),
                };
                let tail = it.next().unwrap_or("");
                // `[action] goal`. A plan written before actions existed has
                // no bracket, and reads back with an empty action rather than
                // failing, so older runs stay readable.
                let (action, goal) = match tail.strip_prefix('[').and_then(|t| t.split_once("] ")) {
                    Some((a, g)) => (a.to_string(), g.to_string()),
                    None => (String::new(), tail.to_string()),
                };
                steps.push(PlanStep { id, parent, status, role, action, goal });
            }
        }
        // Refused rather than guessed at, the way every other format in this
        // tree refuses a missing magic: a file that is not a plan must not
        // parse as an empty one.
        if !seen_magic {
            return None;
        }
        Some(Plan { goal, autonomy, steps })
    }
}

// --- steps ----------------------------------------------------------------

/// One completed step: what was decided, and what the world answered.
///
/// The two are separate fields because only one of them is re-derivable, and
/// finding that out cost an experiment. Stage 1 was written expecting two runs
/// of one plan to produce the same graph, and they did not: the run writes its
/// own steps under `/ai/work`, which is inside the `/ai` a worker then lists,
/// so the second run legitimately saw a directory hash the first had changed.
///
/// The decision was identical both times. The observation was not, and could
/// not be, because the world had moved. So `action` is what the worker chose
/// and is the thing two runs must agree on; `observation` is what came back
/// and is allowed to differ.
#[derive(Clone)]
pub struct Step {
    pub role: String,
    pub goal: String,
    /// The applet and its arguments, as decided. Re-derivable.
    pub action: String,
    pub ok: bool,
    /// What came back. A reading of a world that moves, so not re-derivable.
    pub observation: String,
    /// Who decided the action.
    pub by: By,
}

/// Who chose a step's action.
///
/// The distinction is what makes a transcript trainable, and it has to be
/// recorded rather than inferred. A worker-decided step is a decode that
/// happened: a prompt went in under the grammar and an applet came out, which
/// is the shape the classifier is trained on. A manager-decided step is a
/// dispatch, and there is no decision inside it to learn from -- its decode
/// happened once, at planning time, under a different prompt.
#[derive(Clone, Copy, PartialEq)]
pub enum By {
    Worker,
    Manager,
}

impl Step {
    fn render(&self) -> String {
        let mut s = String::from("role ");
        s.push_str(if self.role.is_empty() { "-" } else { &self.role });
        s.push_str("\ngoal ");
        s.push_str(&one_line(&self.goal));
        s.push_str("\naction ");
        s.push_str(&one_line(&self.action));
        s.push_str("\nby ");
        s.push_str(match self.by {
            By::Worker => "worker",
            By::Manager => "manager",
        });
        s.push_str("\noutcome ");
        s.push_str(if self.ok { "done" } else { "failed" });
        // A blank line, then the observation takes the rest of the file.
        // Observations are prose and contain newlines; anything that had to
        // parse past them would break on its own content.
        s.push_str("\n\n");
        s.push_str(&self.observation);
        s
    }

    fn parse(text: &str) -> Option<Step> {
        let (head, body) = text.split_once("\n\n")?;
        let mut role = String::new();
        let mut goal = String::new();
        let mut action = String::new();
        let mut ok = None;
        // Absent reads as `Manager`, which is the conservative direction and
        // not the tidy one. A transcript written before this line existed
        // cannot say who decided its steps, and `harvest` trains on worker
        // decisions only -- so guessing `Worker` would quietly feed it steps
        // whose goal *is* their action, teaching a classifier to copy the
        // first word of its own prompt.
        let mut by = By::Manager;
        for line in head.lines() {
            if let Some(r) = line.strip_prefix("role ") {
                role = if r == "-" { String::new() } else { r.to_string() };
            } else if let Some(g) = line.strip_prefix("goal ") {
                goal = g.to_string();
            } else if let Some(a) = line.strip_prefix("action ") {
                action = a.to_string();
            } else if let Some(o) = line.strip_prefix("outcome ") {
                ok = Some(o == "done");
            } else if let Some(w) = line.strip_prefix("by ") {
                by = if w == "worker" { By::Worker } else { By::Manager };
            }
        }
        Some(Step { role, goal, action, ok: ok?, observation: body.to_string(), by })
    }
}

// --- paths ----------------------------------------------------------------

/// Whether a name is safe to build a path from.
///
/// A run name and a role name become one path component under `ROOT` or
/// `ROLES` by `format!`, and the namespace resolver honours `..` -- so a name
/// of `../../pwned` climbs out of the work tree and writes at the root, and a
/// name of `../agent/policy` writes *through* an existing blob, turning it
/// into a directory and destroying it. Found by trying exactly that: it
/// replaced `/ai/agent/policy`, which `godel` reads into every variant's
/// lineage.
///
/// So a name has to be a single component: non-empty, no slash, and not a
/// `.`/`..` that the resolver would act on. Everything a plausible run or role
/// is called passes; nothing that escapes the subtree does.
fn valid_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains('/')
}

fn dir(run: &str) -> String {
    format!("{}/{}", ROOT, run)
}

fn plan_path(run: &str) -> String {
    format!("{}/{}/plan", ROOT, run)
}

fn steps_dir(run: &str) -> String {
    format!("{}/{}/steps", ROOT, run)
}

/// Four digits, zero-padded, so sorted order is insertion order.
fn stepname(n: usize) -> String {
    let mut digits = [b'0'; 4];
    let mut v = n;
    for d in digits.iter_mut().rev() {
        *d = b'0' + (v % 10) as u8;
        v /= 10;
    }
    String::from(core::str::from_utf8(&digits).unwrap_or("0000"))
}

/// Collapse a value onto one line, since the plan format is line-oriented.
///
/// A goal is text a model wrote, so it can contain anything. Writing it raw
/// would let one step's goal parse as several steps.
fn one_line(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        out.push(if c == '\n' || c == '\r' { ' ' } else { c });
    }
    out.trim().to_string()
}

// --- the interface --------------------------------------------------------

pub fn set_plan(run: &str, plan: &Plan) -> bool {
    // The one choke point for creating a run, so the name is checked here
    // rather than at each of the several callers -- a write that slips past a
    // caller's own check is the failure this guards.
    if !valid_name(run) {
        return false;
    }
    sysbox::write_text(&plan_path(run), &plan.render())
}

pub fn plan(run: &str) -> Option<Plan> {
    let bytes = sysbox::read_blob(&plan_path(run))?;
    Plan::parse(core::str::from_utf8(&bytes).ok()?)
}

/// Append a completed step. Answers its number.
pub fn append_step(run: &str, step: &Step) -> Option<usize> {
    let n = sysbox::children(&steps_dir(run)).len();
    if n >= MAX_STEPS {
        return None;
    }
    let path = format!("{}/{}", steps_dir(run), stepname(n));
    if sysbox::write_text(&path, &step.render()) {
        Some(n)
    } else {
        None
    }
}

pub fn steps(run: &str) -> Vec<Step> {
    let d = steps_dir(run);
    let mut out = Vec::new();
    // `children` answers in lexicographic order and the names are padded, so
    // this is insertion order without sorting anything here.
    for name in sysbox::children(&d) {
        let Some(bytes) = sysbox::read_blob(&format!("{}/{}", d, name)) else {
            continue;
        };
        let Ok(text) = core::str::from_utf8(&bytes) else { continue };
        if let Some(s) = Step::parse(text) {
            out.push(s);
        }
    }
    out
}

pub fn put_artifact(run: &str, name: &str, bytes: Vec<u8>) -> bool {
    sysbox::write_blob(&format!("{}/{}/artifacts/{}", ROOT, run, name), bytes)
}

pub fn artifact(run: &str, name: &str) -> Option<Vec<u8>> {
    sysbox::read_blob(&format!("{}/{}/artifacts/{}", ROOT, run, name))
}

/// The whole run, as one address.
///
/// The re-derivability check, and it costs one comparison. Equal roots mean
/// every node equal, in the same order, with the same contents -- which is
/// what content addressing already guarantees and what this module exists to
/// borrow.
pub fn root(run: &str) -> Option<[u8; 32]> {
    sysbox::hash_of(&dir(run))
}

pub fn runs() -> Vec<String> {
    sysbox::children(ROOT)
}

pub fn exists(run: &str) -> bool {
    sysbox::is_dir(&dir(run))
}

// --- the worker -----------------------------------------------------------

/// Run one step, greedily, and record what happened.
///
/// The whole worker. It chooses an applet for the step's goal, walks out the
/// arguments, dispatches, and writes a summary node.
///
/// **Everything in it is greedy on purpose.** `choose` at temperature 0 takes
/// the argmax and `decode_args` always did, so given the same goal and the
/// same engine state this produces the same action and the same argument
/// string. That is what makes a run re-derivable, and re-derivability is what
/// lets two runs be compared by one hash instead of by reading them.
///
/// It deliberately does not go through `agent::propose`. That path forks three
/// ways through `deliberate` and is the better router for an operator asking
/// for something once; here the same question must get the same answer twice,
/// and a fork is exactly what stops that.
///
/// The trust level is the caller's. A worker cannot reach past it, because the
/// grammar `choose` decodes under is built from what that level admits, so an
/// applet outside it has no token sequence at all.
pub fn run_step(
    run: &str,
    step: &PlanStep,
    trust: super::harness::Trust,
    calls: &mut usize,
) -> Option<Step> {
    // A pre-decided action costs nothing. This is the line the whole manager
    // exists to reach: with it filled the worker is a dispatcher, and `calls`
    // does not move.
    let (name, args) = if !step.action.is_empty() {
        let (n, a) = match step.action.split_once(' ') {
            Some((n, a)) => (n.trim(), a.trim()),
            None => (step.action.trim(), ""),
        };
        // Re-checked against the trust level rather than trusted because it
        // was written down. A plan is a file, and a file can be edited by
        // anything that can write, so admission is decided here and not by
        // whoever produced the plan.
        let name = super::harness::admitted(trust).into_iter().find(|x| *x == n)?;
        (name, String::from(a))
    } else {
        // The role adapter goes on around the decode and comes off after it.
        // Only around a decode: a pre-decided step decodes nothing, so
        // swapping weights for one would be paying a specialist to read out
        // somebody else's decision.
        let held = attach_role(&step.role);
        *calls += 1;
        let picked = super::harness::choose(&step.goal, trust, 0.0).map(|choice| {
            let name = choice.applet;
            // Arguments only when the applet takes some. `check_args` already
            // knows which do, so asking it is cheaper than a decode and cannot
            // disagree with the check that follows.
            let args = if crate::sysbox::check_args(name, "").is_ok() {
                String::new()
            } else {
                *calls += 1;
                super::harness::decode_args(&step.goal, name, true).unwrap_or_default()
            };
            (name, args)
        });
        // Before the `?`, deliberately. A decode that answered nothing must
        // still put back what it displaced, or a failed step would leave the
        // machine wearing a role adapter for everything that came after.
        restore_role(held);
        picked?
    };

    let (ok, observation) = match crate::sysbox::check_args(name, &args) {
        // A refusal is an observation, not an error. The next revision reads
        // it, which is the same bargain `agent::episode` makes.
        Err(why) => (false, format!("invalid arguments: {}", why)),
        Ok(()) => {
            crate::gfx::console::begin_capture();
            let ran = crate::sysbox::dispatch(name, &args);
            let mut obs = crate::gfx::console::end_capture().unwrap_or_default();
            if !ran && obs.is_empty() {
                obs = String::from("(applet did not run)");
            }
            (ran, obs)
        }
    };

    let mut action = String::from(name);
    if !args.is_empty() {
        action.push(' ');
        action.push_str(&args);
    }

    let s = Step {
        role: step.role.clone(),
        goal: step.goal.clone(),
        action,
        ok,
        observation: String::from(observation.trim_end()),
        by: if step.action.is_empty() { By::Worker } else { By::Manager },
    };
    append_step(run, &s)?;
    Some(s)
}

/// Walk a run to completion, or until the budget is spent.
///
/// The engine is claimed once for the whole run rather than per step. Two
/// `&mut Engine` is undefined behaviour and an interleaved decode corrupts the
/// KV cache, so a run that released the engine between steps could have
/// somebody else's conversation land in the middle of it.
///
/// Answers how many steps ran.
pub fn run(run: &str, trust: super::harness::Trust, budget: usize) -> (usize, usize) {
    let Some(_claim) = super::claim_engine() else {
        return (0, 0);
    };
    let mut ran = 0;
    let mut calls = 0;
    for _ in 0..budget {
        let Some(mut p) = plan(run) else { break };
        let Some(next) = p.next().cloned() else { break };
        let Some(done) = run_step(run, &next, trust, &mut calls) else { break };
        // The plan is the schedule, so it has to record what happened or the
        // next read picks the same step forever.
        for st in p.steps.iter_mut() {
            if st.id == next.id {
                st.status = if done.ok { Status::Done } else { Status::Failed };
            }
        }
        set_plan(run, &p);
        ran += 1;
    }
    (ran, calls)
}

/// Decompose a goal into a plan, in one generation.
///
/// The manager. It is re-prompted from nothing, holds no conversation, and
/// writes what it decided into the plan so the workers that follow need no
/// model at all.
///
/// Answers (steps planned, decode calls spent).
pub fn decompose(
    run: &str,
    goal: &str,
    trust: super::harness::Trust,
    max_steps: usize,
) -> Option<(usize, usize)> {
    let _claim = super::claim_engine()?;
    let (actions, calls) = super::harness::plan_actions(goal, trust, max_steps)?;
    let steps: Vec<PlanStep> = actions
        .iter()
        .enumerate()
        .map(|(i, (name, args))| {
            let mut action = String::from(*name);
            if !args.is_empty() {
                action.push(' ');
                action.push_str(args);
            }
            PlanStep {
                id: i + 1,
                // Flat. A manager that decomposed into a tree would need to
                // say which step depends on which, and nothing it produces
                // today carries that, so claiming a tree would be inventing
                // structure the model never expressed.
                parent: None,
                status: Status::Todo,
                role: String::from("worker"),
                // The step's goal is what the manager decided for it, not the
                // whole task. Copying the overall goal into every step made
                // `work <run>` print the same line four times and said nothing
                // about what any step was for.
                goal: action.clone(),
                action,
            }
        })
        .collect();
    let n = steps.len();
    // Attended. A manager that could declare its own plan autonomous would
    // be the model deciding when nobody watches it, which is the one thing
    // the grant exists to keep out of the model's hands.
    let plan = Plan { goal: String::from(goal), autonomy: Autonomy::Attended, steps };
    if !set_plan(run, &plan) {
        return None;
    }
    Some((n, calls))
}

/// What a run decided, in order.
///
/// The re-derivable half. Two runs of one plan over the same starting state
/// must agree here; whether their observations agree is a question about the
/// world, not about the workflow.
pub fn decisions(run: &str) -> Vec<String> {
    steps(run).into_iter().map(|s| s.action).collect()
}

/// How two runs compare.
///
/// Reported as two answers rather than one, because they mean different
/// things. Decisions differing is a workflow that is not re-derivable and is a
/// defect. Observations differing is the world having moved between the runs,
/// which for a workflow that writes into the namespace it reads is not only
/// possible but usual.
pub struct Comparison {
    pub steps_a: usize,
    pub steps_b: usize,
    pub decisions_agree: bool,
    pub observations_agree: bool,
    /// The first step whose decision differed, if any.
    pub first_difference: Option<usize>,
}

pub fn compare(a: &str, b: &str) -> Comparison {
    let (sa, sb) = (steps(a), steps(b));
    let mut first = None;
    for (i, (x, y)) in sa.iter().zip(sb.iter()).enumerate() {
        if x.action != y.action {
            first = Some(i);
            break;
        }
    }
    Comparison {
        steps_a: sa.len(),
        steps_b: sb.len(),
        decisions_agree: first.is_none() && sa.len() == sb.len(),
        observations_agree: sa.len() == sb.len()
            && sa.iter().zip(sb.iter()).all(|(x, y)| x.observation == y.observation),
        first_difference: first,
    }
}

// --- autonomy -------------------------------------------------------------
//
// A workflow that may advance while nobody is watching needs two things and
// not one: the plan declares it, and the operator grants it. Either alone is
// a gate that does not hold. A declaration alone is the file granting itself,
// and a grant alone would apply to whatever the file says next.

/// The part of a plan a grant is pinned to.
///
/// Everything except each step's status. A grant approves what a workflow will
/// do, and status is how far it got, so hashing the whole file would mean the
/// first step revoked the grant that let it take that step. A gate that works
/// exactly once is worse than none, because it looks like it is holding.
fn intent(plan: &Plan) -> String {
    let mut p = plan.clone();
    for st in p.steps.iter_mut() {
        st.status = Status::Todo;
    }
    p.render()
}

/// The address a grant names.
///
/// Editing a plan by one byte revokes its grant by construction, which is the
/// property `app::manifest` gets from putting the request inside the hash and
/// `skill` gets from hashing the file.
pub fn intent_hash(run: &str) -> Option<[u8; 32]> {
    let p = plan(run)?;
    Some(crate::store::sha256::hash(intent(&p).as_bytes()))
}

pub fn granted(h: &[u8; 32]) -> bool {
    let Some(b) = sysbox::read_blob(GRANTS) else {
        return false;
    };
    let want = super::voter::hex(h);
    alloc::string::String::from_utf8_lossy(&b)
        .lines()
        .any(|l| l.trim() == want)
}

/// Approve one plan's intent. Only the operator's command reaches this.
///
/// `work` is not in `sysbox::APPLETS`, so the decoding grammar has no token
/// sequence for it and the model cannot call this by any route. That is the
/// same reason `skill trust` and `app trust` are shell-only: a model able to
/// grant itself trust would have defeated the gate by using it.
pub fn grant(h: &[u8; 32]) -> bool {
    if granted(h) {
        return true;
    }
    let mut text = sysbox::read_blob(GRANTS)
        .map(|b| alloc::string::String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    text.push_str(&super::voter::hex(h));
    text.push('\n');
    sysbox::write_text(GRANTS, &text)
}

pub fn revoke(h: &[u8; 32]) -> bool {
    let Some(b) = sysbox::read_blob(GRANTS) else {
        return false;
    };
    let drop = super::voter::hex(h);
    let mut text = String::new();
    let mut hit = false;
    for line in alloc::string::String::from_utf8_lossy(&b).lines() {
        if line.trim() == drop {
            hit = true;
            continue;
        }
        text.push_str(line);
        text.push('\n');
    }
    hit && sysbox::write_text(GRANTS, &text)
}

pub fn grants() -> Vec<String> {
    let Some(b) = sysbox::read_blob(GRANTS) else {
        return Vec::new();
    };
    alloc::string::String::from_utf8_lossy(&b)
        .lines()
        .map(|l| String::from(l.trim()))
        .filter(|l| !l.is_empty())
        .collect()
}

/// Whether a plan is fit to advance unattended, and why not when it is not.
///
/// Static and cheap, which is what lets it run before every unattended step
/// rather than once at the grant. The order is the same one `godel`'s
/// rotation uses: cheap and declared before expensive and composed.
pub struct Fitness {
    pub declared: bool,
    pub steps: usize,
    pub admissible: bool,
    /// The first action no read-only worker could reach.
    pub refused: Option<String>,
    pub acyclic: bool,
    pub bounded: bool,
    pub fit: bool,
}

pub fn check(run: &str) -> Option<Fitness> {
    Some(check_plan(&plan(run)?))
}

/// The rules themselves, over a plan rather than a name.
///
/// Split out so they are pure, which is what lets the boot suite exercise them
/// on a machine with no namespace. A gate whose rules are only reachable
/// through a store is a gate nobody checks on the way past.
pub fn check_plan(p: &Plan) -> Fitness {
    let admitted = super::harness::admitted(super::harness::Trust::ReadOnly);

    // A pre-decided action is the one thing in a plan that names an applet
    // without a grammar in front of it. A worker-decided step needs no check
    // here: `choose` decodes under a grammar built from this same level, so an
    // applet outside it has no token sequence at all and is unspellable.
    let mut refused = None;
    for st in p.steps.iter() {
        if st.action.is_empty() {
            continue;
        }
        let name = st.action.split(' ').next().unwrap_or("").trim();
        if !admitted.iter().any(|a| *a == name) {
            refused = Some(st.action.clone());
            break;
        }
    }

    // Ids unique, parents present, and every chain reaching a root. A
    // duplicate id makes `next()`'s parent lookup answer whichever it met
    // first, and a cycle makes a plan that is never done and never ready.
    let mut acyclic = true;
    for (i, st) in p.steps.iter().enumerate() {
        if p.steps.iter().take(i).any(|q| q.id == st.id) {
            acyclic = false;
            break;
        }
        let mut at = st.parent;
        let mut hops = 0usize;
        while let Some(id) = at {
            hops += 1;
            if hops > p.steps.len() {
                acyclic = false;
                break;
            }
            match p.steps.iter().find(|q| q.id == id) {
                Some(q) => at = q.parent,
                None => {
                    acyclic = false;
                    break;
                }
            }
        }
        if !acyclic {
            break;
        }
    }

    let bounded = !p.steps.is_empty() && p.steps.len() <= MAX_UNATTENDED;
    let declared = p.autonomy == Autonomy::Unattended;
    let admissible = refused.is_none();
    Fitness {
        declared,
        steps: p.steps.len(),
        admissible,
        refused,
        acyclic,
        bounded,
        fit: declared && admissible && acyclic && bounded,
    }
}

/// Advance one step of one workflow the operator declared and granted.
///
/// **The trust level is not a parameter.** Unattended means read-only, and
/// making it an argument would put the one decision that matters in the hands
/// of every caller. The enforcement is the grammar rather than a check after
/// the fact: a worker at this level has no token sequence for a mutating
/// applet.
///
/// Which run it picks is a function of what is written, so two readers of the
/// same namespace advance the same workflow. Same argument
/// `godel::next_proposal` makes for taking its axis from the ledger.
///
/// `allow_decode` is the caller saying whether this tick can afford a model
/// call. Answers the run it advanced and whether that step cost one.
pub fn tick_unattended(allow_decode: bool) -> Option<(String, bool)> {
    for name in runs() {
        let Some(p) = plan(&name) else { continue };
        if p.autonomy != Autonomy::Unattended {
            continue;
        }
        let Some(next) = p.next().cloned() else { continue };
        // Re-derived at dispatch from the plan as it stands, never from what
        // it looked like when the operator read it. That is the whole value of
        // pinning a grant to a hash rather than to a name.
        let Some(h) = intent_hash(&name) else { continue };
        if !granted(&h) {
            continue;
        }
        // Re-checked as well. A grant is evidence that the operator approved
        // this intent. It is not evidence that the intent is still admissible,
        // because `admitted` is built from the live applet table and a build
        // can narrow it.
        match check(&name) {
            Some(f) if f.fit => {}
            _ => continue,
        }
        let decodes = next.action.is_empty();
        if decodes && !allow_decode {
            continue;
        }
        let (ran, _) = run(&name, super::harness::Trust::ReadOnly, 1);
        if ran > 0 {
            return Some((name, decodes));
        }
    }
    None
}

// --- roles ----------------------------------------------------------------
//
// A role has been a naming convention up to here: every worker wears whatever
// the machine adopted, or nothing, and `role` is a word in a plan file. This
// is where it earns the name or fails to.
//
// The claim being tested is narrow and is the only one worth making: an
// adapter trained on a role's own successful steps beats what would otherwise
// run, on that role's own held-out steps. If it does not, the honest response
// is to say roles are a naming convention and go on calling them one.

/// One harvested example, and the run it came from.
///
/// The run travels with it because the split is by run. It is not a field of
/// `vocab::Example` and should not be: the routing corpus splits by template
/// family, this splits by run, and the two are the same idea wearing different
/// clothes rather than one thing.
pub struct RoleExample {
    pub applet: String,
    pub task: String,
    pub run: String,
}

/// What a harvest found, and what it threw away.
///
/// Every count is reported. A harvest that quietly dropped four fifths of the
/// transcripts and printed the fifth it kept would look exactly like one that
/// found a rich set, and the difference is the whole question of whether
/// there is anything here to train on.
pub struct Harvest {
    pub seen: usize,
    pub dispatched: usize,
    pub failed: usize,
    pub duplicate: usize,
    pub roles: Vec<(String, usize)>,
}

fn role_dir(role: &str) -> String {
    format!("{}/{}", ROLES, role)
}

fn role_ex_dir(role: &str) -> String {
    format!("{}/{}/ex", ROLES, role)
}

/// Build every role's example set from the transcripts on this machine.
///
/// Two filters, and each removes something that would make the result a lie.
pub fn harvest() -> Harvest {
    let mut h = Harvest {
        seen: 0,
        dispatched: 0,
        failed: 0,
        duplicate: 0,
        roles: Vec::new(),
    };
    let mut kept: Vec<(String, RoleExample)> = Vec::new();

    for run in runs() {
        for st in steps(&run) {
            h.seen += 1;
            // A manager-decided step is a dispatch and not a decision. Worse
            // than merely useless as an example: `decompose` writes the action
            // into the step's goal, so the pair would teach a classifier to
            // copy the first word of its own prompt, and it would score
            // beautifully doing it.
            if st.by != By::Worker {
                h.dispatched += 1;
                continue;
            }
            // The transcripts are the base model's own choices, so training on
            // all of them is self-distillation: the adapter's target is the
            // thing that produced it, and the best it can do is agree. Keeping
            // only what worked is what makes the target distribution different
            // from the source, and it is the only reason any of this can move
            // a number at all.
            if !st.ok {
                h.failed += 1;
                continue;
            }
            let applet = st.action.split(' ').next().unwrap_or("").trim();
            let task = one_line(&st.goal);
            if applet.is_empty() || task.is_empty() {
                continue;
            }
            let role = if st.role.is_empty() {
                String::from("worker")
            } else {
                st.role.clone()
            };
            // Re-running a workflow produces the same steps again, and
            // counting them twice would weight one decision by how often
            // somebody replayed it.
            if kept
                .iter()
                .any(|(r, e)| *r == role && e.applet == applet && e.task == task)
            {
                h.duplicate += 1;
                continue;
            }
            kept.push((
                role,
                RoleExample {
                    applet: String::from(applet),
                    task,
                    run: run.clone(),
                },
            ));
        }
    }

    let mut names: Vec<String> = Vec::new();
    for (r, _) in kept.iter() {
        if !names.iter().any(|n| n == r) {
            names.push(r.clone());
        }
    }
    for name in names {
        // A role comes from a plan's `role` field, which is text, so it takes
        // the same check a run name does before it becomes a path -- a step
        // whose role is `../../x` must not write outside the roles tree.
        if !valid_name(&name) {
            continue;
        }
        // Rewritten whole rather than appended to. A set that grew on every
        // harvest would double on the second one, and positional splits over
        // a set that moves are the "test set that moved" failure arriving by
        // a third route.
        let d = role_ex_dir(&name);
        for child in sysbox::children(&d) {
            sysbox::detach(&format!("{}/{}", d, child));
        }
        let mut n = 0usize;
        for (r, ex) in kept.iter() {
            if *r != name {
                continue;
            }
            let body = format!("{}\t{}\t{}", ex.applet, ex.task, ex.run);
            if sysbox::write_text(&format!("{}/{}", d, stepname(n)), &body) {
                n += 1;
            }
        }
        h.roles.push((name, n));
    }
    h
}

pub fn role_names() -> Vec<String> {
    sysbox::children(ROLES)
}

pub fn role_examples(role: &str) -> Vec<RoleExample> {
    let d = role_ex_dir(role);
    let mut out = Vec::new();
    for name in sysbox::children(&d) {
        let Some(bytes) = sysbox::read_blob(&format!("{}/{}", d, name)) else {
            continue;
        };
        let Ok(text) = core::str::from_utf8(&bytes) else { continue };
        let mut parts = text.trim_end_matches('\n').splitn(3, '\t');
        let (Some(applet), Some(task), Some(run)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        out.push(RoleExample {
            applet: String::from(applet),
            task: String::from(task),
            run: String::from(run),
        });
    }
    out
}

/// How many distinct runs a set came from.
pub fn role_runs(ex: &[RoleExample]) -> usize {
    let mut seen: Vec<&str> = Vec::new();
    for e in ex {
        if !seen.iter().any(|r| *r == e.run) {
            seen.push(&e.run);
        }
    }
    seen.len()
}

/// Where to cut a role's examples so that whole runs are held out.
///
/// Never at a step boundary. Steps inside one run share a goal and differ by
/// slot values, so a step split measures memorisation while looking like
/// generalisation -- exactly the reason the routing corpus holds out whole
/// template families and never sampled instances.
///
/// Answers `ex.len()` when nothing can be held out, which is a set from one
/// run. That is a refusal and the caller treats it as one.
pub fn role_split(ex: &[RoleExample]) -> usize {
    if ex.is_empty() {
        return 0;
    }
    let want = (ex.len() / 4).max(1);
    let mut cut = ex.len();
    let mut held = 0usize;
    let mut i = ex.len();
    while i > 0 {
        let run = &ex[i - 1].run;
        let mut j = i;
        while j > 0 && ex[j - 1].run == *run {
            j -= 1;
        }
        // Taking this run has to leave something to train on. A split that
        // held out everything would report a validation figure over a model
        // that saw nothing.
        if j == 0 {
            break;
        }
        cut = j;
        held += i - j;
        if held >= want {
            break;
        }
        i = j;
    }
    cut
}

pub fn role_adapter(role: &str) -> Option<Vec<u8>> {
    sysbox::read_blob(&format!("{}/adapter", role_dir(role)))
}

/// What was attached before a role adapter displaced it.
///
/// `None` means nothing was swapped and there is nothing to put back. The
/// distinction is load-bearing: a step whose role has no adapter must not
/// detach the one the machine adopted, which is what an unconditional restore
/// would do.
type Held = Option<Option<super::adapter::Adapters>>;

fn attach_role(role: &str) -> Held {
    if role.is_empty() {
        return None;
    }
    let blob = role_adapter(role)?;
    super::with_engine(|e| {
        let prev = e.model.detach_adapters();
        if e.model.load_adapters(&blob).is_ok() {
            Some(prev)
        } else {
            // A role adapter that will not load is a reason to run the step
            // unspecialised. It is never a reason to run it with nothing,
            // which is what leaving the detach standing would do.
            e.model.adapters = prev;
            None
        }
    })
    .flatten()
}

fn restore_role(held: Held) {
    let Some(prev) = held else { return };
    super::with_engine(|e| {
        e.model.adapters = prev;
    });
}

/// Why a role could not be trained.
pub enum RoleError {
    /// No such role, or its set is empty.
    Unknown,
    /// Fewer examples than any judge could work with. Carries the count.
    TooFew(usize),
    /// Every example came from one run, so nothing can be held out without
    /// holding out a step whose siblings trained.
    OneRun,
    /// Every example carries the same label.
    ///
    /// An adapter over a one-class set learns a prior and scores perfectly
    /// doing it, which is the one result that would look like success and
    /// mean nothing at all.
    OneClass,
    NoEngine,
    Train(super::train::RunError),
}

/// What a role trial produced.
///
/// The four judges are `godel`'s, called rather than copied. A second
/// implementation of J1 that had drifted would be judging role adapters by
/// whatever it had drifted into, which is the objection `model.rs` makes twice
/// about two implementations that are supposed to agree.
pub struct RoleFit {
    pub examples: usize,
    pub runs: usize,
    pub train_end: usize,
    pub decisions: usize,
    pub validation: usize,
    /// Distinct applets in the set. One is a degenerate set and says so.
    pub classes: usize,
    pub base: f32,
    pub adapted: f32,
    /// The baseline on the slice it trained on.
    ///
    /// Reported because it is what turns "no difference was found" into
    /// "there was no difference to find". A harvested label is the base
    /// model's own argmax, so a baseline already at the ceiling on the
    /// training slice is not a small sample; it is the experiment being
    /// impossible as posed.
    pub base_train: f32,
    pub fixed: usize,
    pub broke: usize,
    pub mcnemar: f32,
    pub j1: bool,
    pub j1_why: &'static str,
    pub j2: bool,
    pub goals_held: usize,
    pub goals_total: usize,
    pub j3: bool,
    pub j3_why: &'static str,
    pub j4: bool,
    pub resident_kib: usize,
    /// All four held, so the adapter was stored under the role.
    pub kept: bool,
}

/// Train one role's adapter from its own transcripts, and judge it.
///
/// **There is no ledger and no rollback here, and that is deliberate.**
/// `godel` adopts by swapping the head pointer, which puts one adapter in
/// front of everything the machine decides. A role adapter adopted that way
/// would stop being a role: it is attached for the steps of its own role and
/// detached after, so the thing `godel`'s lineage is for -- undoing a change
/// that is in force everywhere -- has nothing to undo. Declining to store a
/// rejected fit leaves the machine exactly as it was.
pub fn train_role(role: &str, b: &super::train::Budget) -> Result<RoleFit, RoleError> {
    use super::train::Slice;

    let ex = role_examples(role);
    if ex.is_empty() {
        return Err(RoleError::Unknown);
    }
    if ex.len() < MIN_ROLE_EXAMPLES {
        return Err(RoleError::TooFew(ex.len()));
    }
    let cut = role_split(&ex);
    if cut >= ex.len() {
        return Err(RoleError::OneRun);
    }
    let runs_seen = role_runs(&ex);
    let mut classes: Vec<&str> = Vec::new();
    for e in ex.iter() {
        if !classes.iter().any(|a| *a == e.applet) {
            classes.push(&e.applet);
        }
    }
    let classes = classes.len();
    if classes < 2 {
        return Err(RoleError::OneClass);
    }
    let items: Vec<super::vocab::Example> = ex
        .iter()
        .map(|e| super::vocab::Example {
            applet: e.applet.clone(),
            task: e.task.clone(),
        })
        .collect();

    let out = super::with_engine(|e| {
        // No test slice: `val_end == end`. A set this size cannot carry three
        // slices, and inventing a third here would spend `godel`'s test budget
        // on a question it was not set aside for.
        let t = match super::train::prepare_on(e, b, &items, cut, items.len(), items.len()) {
            Ok(t) => t,
            Err(err) => return Err(RoleError::Train(err)),
        };
        // The baseline is what would otherwise run, which is whatever is
        // attached -- not the frozen base. A role adapter that beats a bare
        // model and loses to the one the machine already adopted has not
        // earned a swap, and comparing against the base would hide that.
        let incumbent = e.model.adapters.as_ref().and_then(|a| t.gather(a));
        let fit = t.train(b);

        let base = t.score(incumbent.as_ref(), Slice::Validation);
        let adapted = t.score(Some(&fit.dora), Slice::Validation);
        let base_train = t.score(incumbent.as_ref(), Slice::Train);
        let (broke, fixed, _, _) =
            t.paired(incumbent.as_ref(), Some(&fit.dora), Slice::Validation);
        let chi = super::godel::mcnemar(broke, fixed);
        let n_val = t.slice_size(Slice::Validation);
        let (j1, j1_why) = if n_val == 0 {
            (false, "no validation decisions")
        } else if fixed <= broke {
            (false, "no net repair")
        } else if fixed - broke < super::godel::MIN_FIXED {
            (false, "net repair below the floor")
        } else if chi < super::godel::MCNEMAR_95 {
            (false, "inside the noise")
        } else {
            (true, "beyond the noise")
        };

        // J2 is the one that earns its keep here, and it is free. A role
        // adapter trained on a few dozen steps is exactly the object most
        // likely to route the machine's own unasked goals somewhere new, and
        // this is the same replay `godel` runs along the baseline's own path.
        let (goals_held, goals_total) = t.guards_hold(Some(&fit.dora));
        let j2 = goals_held == goals_total
            && goals_total > 0
            && t.guards().iter().all(|g| !g.mutates);
        let (j3, j3_why) = super::godel::sanity(&t, &fit.dora);
        let resident_kib = (fit.dora.resident_bytes() + t.live_rows() * 4) / 1024;
        let j4 = fit.dora.r <= b.rank && resident_kib <= super::godel::MAX_RESIDENT_KIB;
        let kept = j1 && j2 && j3 && j4;

        if kept {
            let adapters = t.scatter(&fit.dora, &e.model.cfg, b.alpha);
            sysbox::write_blob(&format!("{}/adapter", role_dir(role)), adapters.to_blob());
        }

        Ok(RoleFit {
            examples: items.len(),
            runs: runs_seen,
            train_end: cut,
            decisions: t.decisions(),
            validation: n_val,
            classes,
            base,
            adapted,
            base_train,
            fixed,
            broke,
            mcnemar: chi,
            j1,
            j1_why,
            j2,
            goals_held,
            goals_total,
            j3,
            j3_why,
            j4,
            resident_kib,
            kept,
        })
    });
    match out {
        Some(r) => r,
        None => Err(RoleError::NoEngine),
    }
}

// --- selftest -------------------------------------------------------------

/// The format and the split rule, none of which need an engine.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, cond: bool| {
        if !cond {
            kprintln!("    FAIL: {}", what);
            ok = false;
        }
    };

    // Pure claims first, so a machine with no namespace still checks the
    // format rather than reporting a vacuous pass.
    claim("names are padded to four digits", stepname(7) == "0007");
    claim("and stay sorted at the boundary", stepname(42) < stepname(100));
    claim(
        "a goal cannot inject a second step",
        !one_line("a\nstep 9 - todo x hijacked").contains('\n'),
    );

    let p = Plan {
        goal: String::from("build the thing"),
        autonomy: Autonomy::Attended,
        steps: alloc::vec![
            PlanStep { id: 1, parent: None, status: Status::Done, role: String::from("writer"), action: String::from("write /tmp/x hi"), goal: String::from("draft it") },
            PlanStep { id: 2, parent: Some(1), status: Status::Todo, role: String::new(), action: String::new(), goal: String::from("check it\nover two lines") },
        ],
    };
    let back = Plan::parse(&p.render());
    claim("a plan round-trips through its own rendering", back.is_some());
    if let Some(b) = &back {
        claim("goal survives", b.goal == "build the thing");
        claim("step count survives", b.steps.len() == 2);
        claim("parents survive", b.steps[1].parent == Some(1));
        claim("an unassigned role reads back as empty", b.steps[1].role.is_empty());
        claim(
            "a multi-line goal is flattened rather than splitting the file",
            b.steps[1].goal == "check it over two lines",
        );
        // The scheduling property the manager depends on.
        claim(
            "next() takes the ready step, not merely the first todo",
            b.next().map(|s| s.id) == Some(2),
        );
        claim(
            "a pre-decided action survives, and an absent one stays empty",
            b.steps[0].action == "write /tmp/x hi" && b.steps[1].action.is_empty(),
        );
    }
    // A blocked child must not be offered before its parent is done.
    let blocked = Plan {
        goal: String::new(),
        autonomy: Autonomy::Attended,
        steps: alloc::vec![
            PlanStep { id: 1, parent: None, status: Status::Todo, role: String::new(), action: String::new(), goal: String::from("a") },
            PlanStep { id: 2, parent: Some(1), status: Status::Todo, role: String::new(), action: String::new(), goal: String::from("b") },
        ],
    };
    claim("a child waits for its parent", blocked.next().map(|s| s.id) == Some(1));

    // Something that is not a plan must not parse as an empty one.
    claim("a file with no magic is refused", Plan::parse("goal x\n").is_none());

    let s = Step {
        role: String::from("writer"),
        goal: String::from("draft it"),
        action: String::from("write /tmp/x hello"),
        ok: true,
        observation: String::from("wrote three lines\n\nand a blank one in the middle"),
        by: By::Worker,
    };
    let sb = Step::parse(&s.render());
    claim("a step round-trips", sb.is_some());
    if let Some(b) = &sb {
        claim(
            "an observation containing a blank line survives whole",
            b.observation == s.observation && b.ok,
        );
        claim("the decision is kept apart from it", b.action == s.action);
        claim("and who decided it survives", b.by == By::Worker);
    }
    // The conservative default, asserted rather than trusted. A transcript
    // with no `by` line predates the field, cannot say who decided its steps,
    // and must not be harvested as if a worker had.
    claim(
        "a step with no `by` line reads as manager-decided",
        Step::parse("role r\ngoal g\naction ls\noutcome done\n\nobs")
            .map(|x| x.by == By::Manager)
            == Some(true),
    );

    // --- the split rule ---------------------------------------------------
    //
    // Pure, so it is checked here rather than being discovered on the one
    // machine that has enough transcripts to train from.
    let ex = |run: &str| RoleExample {
        applet: String::from("ls"),
        task: String::from("t"),
        run: String::from(run),
    };
    let three = alloc::vec![ex("a"), ex("a"), ex("b"), ex("b"), ex("c"), ex("c")];
    claim(
        "the split falls on a run boundary and holds a whole run out",
        role_split(&three) == 4,
    );
    let lopsided = alloc::vec![ex("a"), ex("b"), ex("b"), ex("b"), ex("b"), ex("b")];
    claim(
        "a run larger than the quarter is still held out whole",
        role_split(&lopsided) == 1,
    );
    claim(
        "one run holds nothing out, and says so by refusing",
        role_split(&alloc::vec![ex("a"), ex("a")]) == 2,
    );
    claim("an empty set splits at nothing", role_split(&[]) == 0);

    // --- names are single components --------------------------------------
    //
    // A run or role name becomes a path under a fixed root, and the resolver
    // honours `..`, so a name that is not one component escapes the subtree.
    claim("a plain name is allowed", valid_name("run-1"));
    claim("a traversal is refused", !valid_name("../agent/policy"));
    claim("a bare dot-dot is refused", !valid_name(".."));
    claim("a slashed name is refused", !valid_name("a/b"));
    claim("an empty name is refused", !valid_name(""));

    // --- autonomy ---------------------------------------------------------
    claim(
        "an attended plan renders no autonomy line, so old runs keep their address",
        !p.render().contains("autonomy"),
    );
    let mut unattended = p.clone();
    unattended.autonomy = Autonomy::Unattended;
    claim(
        "and an unattended one round-trips",
        Plan::parse(&unattended.render()).map(|q| q.autonomy) == Some(Autonomy::Unattended),
    );
    claim(
        "a plan that says nothing is attended",
        Plan::parse("work 1\ngoal x\n").map(|q| q.autonomy) == Some(Autonomy::Attended),
    );
    // The property that stops the gate working exactly once: a grant names
    // what a workflow will do, so taking a step must not revoke it.
    let mut advanced = unattended.clone();
    for st in advanced.steps.iter_mut() {
        st.status = Status::Failed;
    }
    claim(
        "progress does not change what a grant names",
        intent(&unattended) == intent(&advanced),
    );
    claim(
        "but an edited step does",
        intent(&unattended) != {
            let mut edited = unattended.clone();
            edited.steps[0].action = String::from("rm /");
            intent(&edited)
        },
    );

    // --- fitness ----------------------------------------------------------
    let fit_step = |action: &str| PlanStep {
        id: 1,
        parent: None,
        status: Status::Todo,
        role: String::from("worker"),
        action: String::from(action),
        goal: String::from("g"),
    };
    let mk = |steps: Vec<PlanStep>| Plan {
        goal: String::from("g"),
        autonomy: Autonomy::Unattended,
        steps,
    };
    claim(
        "a read-only action is admissible",
        check_plan(&mk(alloc::vec![fit_step("ls /ai")])).fit,
    );
    claim(
        "a mutating one is not, and is named",
        check_plan(&mk(alloc::vec![fit_step("rm /ai/about")])).refused.is_some(),
    );
    claim(
        "a worker-decided step needs no action check",
        check_plan(&mk(alloc::vec![fit_step("")])).admissible,
    );
    claim(
        "an attended plan is never fit, whatever else holds",
        !check_plan(&Plan {
            goal: String::from("g"),
            autonomy: Autonomy::Attended,
            steps: alloc::vec![fit_step("ls /ai")],
        })
        .fit,
    );
    let dangling = PlanStep { parent: Some(9), ..fit_step("ls /ai") };
    claim(
        "a parent that is not there is caught",
        !check_plan(&mk(alloc::vec![dangling])).acyclic,
    );
    let a = PlanStep { id: 1, parent: Some(2), ..fit_step("ls /ai") };
    let b = PlanStep { id: 2, parent: Some(1), ..fit_step("ls /ai") };
    claim("and so is a cycle", !check_plan(&mk(alloc::vec![a, b])).acyclic);
    let twins = alloc::vec![fit_step("ls /ai"), fit_step("ls /ai")];
    claim(
        "a duplicate id is refused, since a parent lookup would be ambiguous",
        !check_plan(&mk(twins)).acyclic,
    );
    claim("an empty plan is not bounded, it is empty", !check_plan(&mk(Vec::new())).bounded);

    // --- against a real namespace -----------------------------------------
    //
    // The claims above are about the format. These are about the graph, and
    // they are the point of the module: a run has an address, and two runs
    // that did the same thing have the same address.
    if !sysbox::is_ready() {
        // Said out loud. A suite that skips itself and reports success is
        // indistinguishable from one that passed, which is the failure
        // `smp.rs` records about its own one-shot check.
        kprintln!("  --   no namespace, so the graph itself was not exercised");
        return ok;
    }

    let build = |run: &str, tail: &str| {
        detach_run(run);
        set_plan(run, &p);
        append_step(run, &s);
        append_step(
            run,
            &Step {
                role: String::from("checker"),
                goal: String::from("check it"),
                action: String::from("cat /tmp/x"),
                ok: true,
                observation: format!("looked it over{}", tail),
                by: By::Worker,
            },
        );
        put_artifact(run, "out.txt", b"artifact bytes".to_vec());
    };

    build(".selftest-a", "");
    build(".selftest-b", "");

    claim("a run reports itself present", exists(".selftest-a"));
    claim(
        "the plan comes back through the store",
        plan(".selftest-a").map(|q| q.steps.len()) == Some(2),
    );
    let got = steps(".selftest-a");
    claim("both steps come back", got.len() == 2);
    claim(
        "and in insertion order, which the padding is for",
        got.first().map(|x| x.role.as_str()) == Some("writer")
            && got.get(1).map(|x| x.role.as_str()) == Some("checker"),
    );
    claim(
        "an artifact round-trips",
        artifact(".selftest-a", "out.txt").as_deref() == Some(&b"artifact bytes"[..]),
    );

    // The whole reason this module stores rather than computes.
    let ra = root(".selftest-a");
    let rb = root(".selftest-b");
    claim("a run has an address", ra.is_some());
    claim(
        "two runs that did the same thing share it",
        ra.is_some() && ra == rb,
    );

    // And the canary. A comparison that has never reported a difference is
    // indistinguishable from one that compares nothing, so the suite fails if
    // a run that differs by one byte still matches.
    build(".selftest-c", "!");
    claim(
        "a run differing by one byte does not",
        root(".selftest-c").is_some() && root(".selftest-c") != ra,
    );

    // A missing run must be distinguishable from an empty one.
    claim("a missing run has no plan", plan(".selftest-missing").is_none());
    claim("and does not report itself present", !exists(".selftest-missing"));

    detach_run(".selftest-a");
    detach_run(".selftest-b");
    detach_run(".selftest-c");

    ok
}

/// Remove a run, so the suite leaves nothing behind.
///
/// `detach` takes one path and the run is a subtree, so this walks it. A suite
/// that accumulates a run per boot would grow the namespace forever and, worse,
/// would change the root hashes it is asserting about on the second run.
fn detach_run(run: &str) {
    let d = dir(run);
    for sub in ["steps", "artifacts"] {
        let p = format!("{}/{}", d, sub);
        for name in sysbox::children(&p) {
            sysbox::detach(&format!("{}/{}", p, name));
        }
        sysbox::detach(&p);
    }
    sysbox::detach(&plan_path(run));
    sysbox::detach(&d);
}

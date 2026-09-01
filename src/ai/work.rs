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

#[derive(Clone)]
pub struct Plan {
    pub goal: String,
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
        Some(Plan { goal, steps })
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
}

impl Step {
    fn render(&self) -> String {
        let mut s = String::from("role ");
        s.push_str(if self.role.is_empty() { "-" } else { &self.role });
        s.push_str("\ngoal ");
        s.push_str(&one_line(&self.goal));
        s.push_str("\naction ");
        s.push_str(&one_line(&self.action));
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
        for line in head.lines() {
            if let Some(r) = line.strip_prefix("role ") {
                role = if r == "-" { String::new() } else { r.to_string() };
            } else if let Some(g) = line.strip_prefix("goal ") {
                goal = g.to_string();
            } else if let Some(a) = line.strip_prefix("action ") {
                action = a.to_string();
            } else if let Some(o) = line.strip_prefix("outcome ") {
                ok = Some(o == "done");
            }
        }
        Some(Step { role, goal, action, ok: ok?, observation: body.to_string() })
    }
}

// --- paths ----------------------------------------------------------------

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
        *calls += 1;
        let choice = super::harness::choose(&step.goal, trust, 0.0)?;
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
    let plan = Plan { goal: String::from(goal), steps };
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

// --- selftest -------------------------------------------------------------

/// Twelve claims, none of which need an engine.
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
    };
    let sb = Step::parse(&s.render());
    claim("a step round-trips", sb.is_some());
    if let Some(b) = &sb {
        claim(
            "an observation containing a blank line survives whole",
            b.observation == s.observation && b.ok,
        );
        claim("the decision is kept apart from it", b.action == s.action);
    }

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

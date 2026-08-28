//! The loop that writes an application.
//!
//! Plan, act, check, revise. The shape is ReAct's and the debt is SWE-Agent's,
//! with one substitution that matters: **the observation is a machine check,
//! not the model's opinion of its own work.** SWE-Agent gets a test suite; this
//! gets `uidoc::parse`, `lang::eval_line` and `check_refs`, which are exact,
//! instant, need no forward pass, and answer with a line number.
//!
//! `agent.rs` already argues the negative case in code -- a model grading its
//! own output is "the feedback loop that amplifies its own errors" -- so
//! nothing here asks the model whether it did well. It proposes; arithmetic
//! disposes.
//!
//! ### The contract, and why there has to be one
//!
//! Every check available is satisfied by an application that does nothing:
//!
//! ```text
//! panel   1
//! title   Calculator
//! button  close   Close
//! ```
//!
//! Parses, references nothing, renders, fits. A loop whose only gradient is
//! "pass the checks" has that as its optimum, will find it, and will report
//! success -- silently, and looking like it worked. Form cannot know what was
//! wanted.
//!
//! So intent is turned into more machine checks, once, up front. `plan.txt`
//! holds one clause per line, each a predicate this file can evaluate against
//! the artifact. The loop then has a gradient that points somewhere.
//!
//! ### What this file does not contain
//!
//! No model. Every action arrives from outside, so the whole loop is exercised
//! by a script at boot with no forward passes -- the same trick `agent`'s own
//! selftest uses. Attaching the model is one function that returns an `Action`.
//!
//! The core works on strings in memory rather than on the namespace, for the
//! same reason: the boot selftests run before `sysbox::init`, and a loop that
//! can only be exercised against storage is one that does not get exercised.

use crate::app::{check, skel};
use crate::gfx::uidoc;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// One clause of the contract: something that must be true of the finished
/// application, phrased so this file can check it.
#[derive(Clone, PartialEq, Eq)]
pub enum Req {
    /// The window says this.
    Title(String),
    /// The program defines this function.
    Fn(String),
    /// The panel contains at least one widget of this kind.
    Widget(String),
    /// Something in the panel calls this function.
    Calls(String),
    /// The panel builds its rows from the program.
    Rows,
}

impl Req {
    pub fn render(&self) -> String {
        match self {
            Req::Title(t) => alloc::format!("title {}", t),
            Req::Fn(f) => alloc::format!("fn {}", f),
            Req::Widget(w) => alloc::format!("widget {}", w),
            Req::Calls(f) => alloc::format!("calls {}", f),
            Req::Rows => "rows".to_string(),
        }
    }

    pub fn parse(line: &str) -> Option<Req> {
        let line = line.trim();
        let (verb, rest) = line.split_once(' ').unwrap_or((line, ""));
        let rest = rest.trim();
        match verb {
            "title" if !rest.is_empty() => Some(Req::Title(rest.to_string())),
            "fn" if !rest.is_empty() => Some(Req::Fn(rest.to_string())),
            "widget" if !rest.is_empty() => Some(Req::Widget(rest.to_string())),
            "calls" if !rest.is_empty() => Some(Req::Calls(rest.to_string())),
            "rows" => Some(Req::Rows),
            _ => None,
        }
    }

    /// The closed vocabulary a constrained decode would choose from.
    pub const VERBS: &'static [&'static str] = &["title", "fn", "widget", "calls", "rows"];
}

/// What can be done to a draft in one step.
#[derive(Clone, PartialEq, Eq)]
pub enum Action {
    /// Start again from a skeleton. Replaces both files.
    Skel(String),
    /// Put this text at this line of the panel, replacing what was there. A
    /// line past the end appends.
    Panel(usize, String),
    /// Remove a line of the panel. The escape from a line that will not come
    /// right, and the reason the loop always terminates.
    Drop(usize),
    /// Append a function to the program, from a body skeleton.
    Fn(String, String),
    Done,
}

impl Action {
    pub const VERBS: &'static [&'static str] = &["skel", "panel", "drop", "fn", "done"];
}

/// Function bodies the loop may add.
///
/// Three, and no free-form bodies. `todo`'s `drop()` is twenty lines of string
/// walking with index arithmetic; a model that writes it wrong writes something
/// that parses, runs, and quietly loses an entry. A requirement no body here
/// serves is reported unmet, which is a legible failure rather than a plausible
/// wrong answer.
pub const BODIES: &[(&str, &str)] = &[
    ("constant", "fn {NAME}() {\n  return \"\"\n}\n"),
    (
        "reader",
        "fn {NAME}() {\n  if (exists(here() + \"/{NAME}\")) { return read(here() + \"/{NAME}\") }\n  return \"\"\n}\n",
    ),
    (
        "appender",
        "fn {NAME}(x) {\n  write(here() + \"/{NAME}\", x)\n  return \"\"\n}\n",
    ),
];

pub fn body_kinds() -> Vec<&'static str> {
    BODIES.iter().map(|(k, _)| *k).collect()
}

/// The draft, in memory.
pub struct Work {
    pub name: String,
    pub goal: String,
    pub plan: Vec<Req>,
    pub panel: Vec<String>,
    pub code: String,
    /// How many times each panel line has been tried and failed. The counter
    /// that turns "try again" into "give up on this line", which is what makes
    /// the loop finish.
    pub repairs: BTreeMap<usize, usize>,
    pub steps: usize,
    pub last: Option<String>,
}

/// After three failures a line is dropped rather than tried a fourth time.
///
/// `Outcome::repeated` catches identical actions and would not catch this: a
/// line rewritten four different wrong ways is four distinct actions and no
/// progress at all.
const MAX_REPAIRS: usize = 3;

impl Work {
    pub fn new(name: &str, goal: &str) -> Work {
        Work {
            name: name.to_string(),
            goal: goal.to_string(),
            plan: Vec::new(),
            panel: Vec::new(),
            code: String::new(),
            repairs: BTreeMap::new(),
            steps: 0,
            last: None,
        }
    }

    pub fn panel_text(&self) -> String {
        let mut s = String::new();
        for l in &self.panel {
            s.push_str(l);
            s.push('\n');
        }
        s
    }
}

/// Do one thing, then look at what it did.
///
/// The check is not an action the model can choose. A small model will not
/// choose to check, and the check is what produces the next observation, so it
/// runs after every mutating step and costs nothing.
pub fn apply(w: &mut Work, a: &Action) -> Vec<check::Verdict> {
    w.steps += 1;
    match a {
        Action::Skel(kind) => {
            let Some((panel, code)) = skel::fill(kind, &w.name, &w.name, "Items") else {
                return alloc::vec![check::Verdict::bad(alloc::format!(
                    "no skeleton called '{}'",
                    kind
                ))];
            };
            w.panel = panel.lines().map(|s| s.to_string()).collect();
            w.code = code;
            w.repairs.clear();
        }
        Action::Panel(n, text) => {
            if text.contains('\n') {
                return alloc::vec![check::Verdict::bad(
                    "a panel line is one line".to_string()
                )];
            }
            let i = n.saturating_sub(1);
            if i < w.panel.len() {
                w.panel[i] = text.clone();
            } else {
                w.panel.push(text.clone());
            }
        }
        Action::Drop(n) => {
            let i = n.saturating_sub(1);
            if i >= w.panel.len() {
                return alloc::vec![check::Verdict::bad(alloc::format!(
                    "there is no line {}",
                    n
                ))];
            }
            w.panel.remove(i);
            w.repairs.clear();
        }
        Action::Fn(name, kind) => {
            let Some((_, body)) = BODIES.iter().find(|(k, _)| k == kind) else {
                return alloc::vec![check::Verdict::bad(alloc::format!(
                    "no body called '{}' -- try one of: {}",
                    kind,
                    body_kinds().join(", ")
                ))];
            };
            w.code.push_str(&body.replace("{NAME}", name));
        }
        Action::Done => {}
    }
    let v = look(w);
    w.last = v.iter().find(|x| !x.ok).map(|x| match x.line {
        Some(l) => alloc::format!("line {}: {}", l, x.why),
        None => x.why.clone(),
    });
    v
}

/// Everything that can be said about the draft as it stands.
pub fn look(w: &Work) -> Vec<check::Verdict> {
    let panel = w.panel_text();
    // The `rows` directive belongs to the application layer, so it comes out
    // before the codec sees the document -- the same thing the desktop does.
    let mut stripped = String::new();
    for line in panel.lines() {
        if !line.starts_with("rows\t") {
            stripped.push_str(line);
            stripped.push('\n');
        }
    }
    let p = check::check_panel(&stripped);
    if !p.ok {
        return alloc::vec![p];
    }
    let c = check::check_code(&w.code, &alloc::format!("/draft/{}", w.name));
    if !c.ok {
        return alloc::vec![p, c];
    }
    let mut out = alloc::vec![p, c];
    out.extend(check::check_refs(&w.name, &panel, &w.code));
    out.extend(met(w));
    out
}

/// Which clauses of the contract hold.
pub fn met(w: &Work) -> Vec<check::Verdict> {
    let panel = w.panel_text();
    let defined = check::functions(&w.code);
    let mut out = Vec::new();
    for r in &w.plan {
        let ok = match r {
            Req::Title(t) => panel
                .lines()
                .any(|l| l.strip_prefix("title\t").map(|x| x == t).unwrap_or(false)),
            Req::Fn(f) => defined.iter().any(|(d, _)| d == f),
            Req::Widget(v) => panel
                .lines()
                .any(|l| l.split('\t').next().unwrap_or("") == v.as_str()),
            Req::Calls(f) => panel.lines().any(|l| l.contains(&alloc::format!(" {}", f))),
            Req::Rows => panel.lines().any(|l| l.starts_with("rows\t")),
        };
        out.push(if ok {
            check::Verdict::ok("met")
        } else {
            check::Verdict::bad(alloc::format!("unmet: {}", r.render()))
        });
    }
    out
}

pub struct Report {
    pub steps: usize,
    pub met: usize,
    pub total: usize,
    pub clean: bool,
    pub why: String,
}

/// Is this finished, and may it be adopted?
///
/// Refuses a contract of fewer than two clauses, and one that asks for nothing
/// of the program. Without that the empty valid application is reachable and
/// the loop will find it, because it is the cheapest thing that passes.
pub fn done(w: &Work) -> Option<Report> {
    let v = look(w);
    let m = met(w);
    let met_n = m.iter().filter(|x| x.ok).count();
    let report = |clean: bool, why: &str| Report {
        steps: w.steps,
        met: met_n,
        total: w.plan.len(),
        clean,
        why: why.to_string(),
    };
    if w.plan.len() < 2 {
        return Some(report(false, "a contract of one clause is not a contract"));
    }
    if !w.plan.iter().any(|r| matches!(r, Req::Fn(_) | Req::Rows)) {
        return Some(report(
            false,
            "nothing in the contract asks the application to do anything",
        ));
    }
    if v.iter().all(|x| x.ok) {
        return Some(report(true, "every clause met and every check clean"));
    }
    None
}

/// Run a sequence of actions decided elsewhere.
///
/// The whole loop, with the model removed. Stopping rules are here rather than
/// in the caller so that the scripted run and the generated one cannot differ
/// about when to give up.
pub fn run(w: &mut Work, script: &[Action], budget: usize) -> Report {
    for a in script {
        if w.steps >= budget {
            break;
        }
        // A line that has failed its allowance is dropped rather than tried
        // again. This is what guarantees the loop finishes: a panel with fewer
        // widgets is still a panel.
        if let Action::Panel(n, _) = a {
            let c = w.repairs.entry(*n).or_insert(0);
            if *c >= MAX_REPAIRS {
                let n = *n;
                apply(w, &Action::Drop(n));
                continue;
            }
        }
        let v = apply(w, a);
        if let Action::Panel(n, _) = a {
            if v.iter().any(|x| !x.ok) {
                *w.repairs.entry(*n).or_insert(0) += 1;
            } else {
                w.repairs.remove(n);
            }
        }
        if matches!(a, Action::Done) {
            break;
        }
    }
    done(w).unwrap_or_else(|| {
        let m = met(w);
        Report {
            steps: w.steps,
            met: m.iter().filter(|x| x.ok).count(),
            total: w.plan.len(),
            clean: false,
            why: w
                .last
                .clone()
                .unwrap_or_else(|| "out of steps".to_string()),
        }
    })
}

pub fn selftest() -> bool {
    // Contract clauses survive being written and read.
    for r in [
        Req::Title("A".to_string()),
        Req::Fn("go".to_string()),
        Req::Widget("button".to_string()),
        Req::Calls("go".to_string()),
        Req::Rows,
    ] {
        if Req::parse(&r.render()) != Some(r) {
            return false;
        }
    }
    if Req::parse("nonsense x").is_some() || Req::parse("fn").is_some() {
        return false;
    }

    // A whole run, with no forward passes anywhere in it: build from a
    // skeleton, break a line, fail to repair it three times, watch it dropped,
    // finish.
    let mut w = Work::new("demo", "a list");
    w.plan = alloc::vec![
        Req::Title("demo".to_string()),
        Req::Rows,
        Req::Fn("add".to_string()),
    ];

    if apply(&mut w, &Action::Skel("list".to_string()))
        .iter()
        .any(|v| !v.ok)
    {
        return false;
    }
    // The skeleton alone satisfies the contract.
    if done(&w).map(|r| r.clean) != Some(true) {
        return false;
    }

    // Break a line. The check must catch it and say which line.
    let bad = Action::Panel(3, "nonsense\tx".to_string());
    let v = apply(&mut w, &bad);
    match v.iter().find(|x| !x.ok).and_then(|x| x.line) {
        Some(3) => {}
        _ => return false,
    }
    if done(&w).is_some() {
        return false;
    }

    // Three failed repairs, then the fourth attempt drops the line instead.
    let before = w.panel.len();
    let script = alloc::vec![bad.clone(), bad.clone(), bad.clone(), bad.clone()];
    run(&mut w, &script, 100);
    if w.panel.len() != before - 1 {
        return false;
    }
    // Dropping the line put it back in a state that passes.
    if !look(&w).iter().all(|x| x.ok) {
        return false;
    }

    // A body can be added, and is then a function the checks can see.
    let mut w2 = Work::new("demo2", "x");
    w2.plan = alloc::vec![Req::Fn("stash".to_string()), Req::Rows];
    apply(&mut w2, &Action::Skel("list".to_string()));
    if met(&w2)[0].ok {
        return false; // `stash` does not exist yet
    }
    apply(&mut w2, &Action::Fn("stash".to_string(), "appender".to_string()));
    if !met(&w2)[0].ok {
        return false;
    }
    if apply(&mut w2, &Action::Fn("x".to_string(), "nosuch".to_string()))[0].ok {
        return false;
    }

    // The guards against the empty valid application.
    let mut e = Work::new("empty", "nothing");
    e.panel = alloc::vec![
        "panel\t1".to_string(),
        "title\tempty".to_string(),
        "button\tclose\tClose".to_string(),
    ];
    e.plan = alloc::vec![Req::Title("empty".to_string())];
    // One clause is not a contract.
    if done(&e).map(|r| r.clean) != Some(false) {
        return false;
    }
    // Two clauses that ask nothing of the program is still not a contract.
    e.plan.push(Req::Widget("button".to_string()));
    if done(&e).map(|r| r.clean) != Some(false) {
        return false;
    }
    // And with a real clause it is simply unmet, rather than quietly passing.
    e.plan.push(Req::Rows);
    done(&e).is_none()
}

/// A short account of a run, for the operator.
pub fn describe(r: &Report) -> String {
    alloc::format!(
        "{} step(s), {} of {} clause(s) met -- {}",
        r.steps, r.met, r.total, r.why
    )
}

/// The panel document, for saving.
pub fn panel_of(w: &Work) -> String {
    w.panel_text()
}

/// The verbs a constrained decode would pick an action from.
pub fn action_verbs() -> Vec<&'static str> {
    Action::VERBS.to_vec()
}

/// Something to put in a document, given what the codec accepts.
pub fn widget_verbs() -> Vec<&'static str> {
    uidoc::VERBS.to_vec()
}

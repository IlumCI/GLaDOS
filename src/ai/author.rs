//! The loop that writes an application.
//!
//! Plan, act, check, revise. The shape is ReAct's and the debt is SWE-Agent's,
//! with one substitution that matters: **the observation is a machine check,
//! not the model's opinion of its own work.** SWE-Agent gets a test suite; this
//! gets `uidoc::parse`, `aiksi::eval_line` and `check_refs`, which are exact,
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
/// No free-form bodies, and that has not changed with the language growing.
/// `todo`'s `drop()` is twenty lines of string walking with index arithmetic;
/// a model that writes it wrong writes something that parses, runs, and
/// quietly loses an entry. A requirement no body here serves is reported
/// unmet, which is a legible failure rather than a plausible wrong answer.
///
/// What did change is the reach of each body. Before Aiksi had `split`, `join`
/// and `int`, a body could only return a whole file or append to one, so every
/// application was a text box with a history. These are the same shape --
/// filled in, checked, no arithmetic for the model to get wrong -- over a
/// language that can now count, order and take things apart.
///
/// Nothing here touches the network or the model. A body is added to a program
/// the sandbox will run unattended, and `Touch::Net` is refused there anyway;
/// putting a socket in a skeleton would mean generated applications that only
/// work once somebody has typed `app trust`, which is a confusing failure
/// rather than a safe default. An operator writing Aiksi by hand has the whole
/// table.
pub const BODIES: &[(&str, &str, &str)] = &[
    ("constant", "", "fn {NAME}(): str {\n  return \"\"\n}\n"),
    (
        "reader",
        "",
        "fn {NAME}(): str {\n  if (exists(here() + \"/{NAME}\")) { return read(here() + \"/{NAME}\") }\n  return \"\"\n}\n",
    ),
    (
        "appender",
        "",
        "fn {NAME}(x: str): str {\n  write(here() + \"/{NAME}\", x)\n  return \"\"\n}\n",
    ),
    (
        "lines",
        "",
        "fn {NAME}(): str {\n  t = \"\"\n  if (exists(here() + \"/{NAME}\")) { t = read(here() + \"/{NAME}\") }\n  \
out = list()\n  rows = split(t, \"\\n\")\n  i = 0\n  while (i < len(rows)) {\n    \
if (len(trim(get(rows, i))) > 0) { out = push(out, trim(get(rows, i))) }\n    i = i + 1\n  }\n  \
return join(out, \"\\n\")\n}\n",
    ),
    (
        "counter",
        "",
        "fn {NAME}(): int {\n  n = 0\n  if (exists(here() + \"/{NAME}\")) { n = int(read(here() + \"/{NAME}\")) }\n  \
n = n + 1\n  write(here() + \"/{NAME}\", n)\n  return n\n}\n",
    ),
    (
        "tally",
        "",
        "fn {NAME}(): int {\n  t = \"\"\n  if (exists(here() + \"/{NAME}\")) { t = read(here() + \"/{NAME}\") }\n  \
if (len(trim(t)) == 0) { return 0 }\n  return len(split(trim(t), \"\\n\"))\n}\n",
    ),
    // `rtc_now` answers a `Time` record, so the stamp is a field. Guarded,
    // because a clock that cannot be read answers nothing and `.text` on
    // nothing is an error -- which in a `rows` function is a window that
    // renders a fault instead of a blank.
    (
        "clock",
        "",
        "fn {NAME}(): str {\n  t = rtc_now()\n  if (t) { return t.text }\n  return \"\"\n}\n",
    ),
    // The one body that declares a record, and it earns it rather than
    // decorating with it.
    //
    // A stored line is two things -- whether it is done, and what it says --
    // and every list application in this tree has so far carried that as a
    // prefix somebody re-tests at each use. Naming the shape means the split
    // happens once, in a function whose signature says what it answers, and
    // the row builder reads `it.done` instead of re-deriving it.
    //
    // Two functions, deliberately. A record that never crosses a call boundary
    // is two local variables wearing a hat; this way the `: Item` annotation
    // is checked at every call and the loop's own `check_code` runs it.
    (
        "checklist",
        "rec Item { done: int, text: str }\n\n",
        "fn {NAME}_item(line: str): Item {\n  \
if (starts(line, \"x \")) { return Item(1, trim(substr(line, 2, len(line)))) }\n  \
return Item(0, line)\n}\n\n\
fn {NAME}(): str {\n  raw = \"\"\n  if (exists(here() + \"/{NAME}\")) { raw = read(here() + \"/{NAME}\") }\n  \
out = \"\"\n  rows = split(raw, \"\\n\")\n  n = 0\n  while (n < len(rows)) {\n    \
line = trim(get(rows, n))\n    if (len(line) > 0) {\n      \
it = {NAME}_item(line)\n      mark = \"[ ] \"\n      if (it.done) { mark = \"[x] \" }\n      \
out = out + \"item\\tnone\\t\" + mark + it.text + \"\\n\"\n    }\n    n = n + 1\n  }\n  \
return out\n}\n",
    ),
];

pub fn body_kinds() -> Vec<&'static str> {
    BODIES.iter().map(|(k, ..)| *k).collect()
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
            let Some((_, pre, body)) = BODIES.iter().find(|(k, ..)| k == kind) else {
                return alloc::vec![check::Verdict::bad(alloc::format!(
                    "no body called '{}' -- try one of: {}",
                    kind,
                    body_kinds().join(", ")
                ))];
            };
            // A record declaration is a top-level statement, so it cannot live
            // inside the function that uses it. Emitted once: two functions
            // asking for the same shape is the normal case, and a second `rec
            // Item` would be a redeclaration the program does not need.
            if !pre.is_empty() && !w.code.contains(*pre) {
                w.code.push_str(pre);
            }
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

/// What the loop is doing, for anything that wants to watch.
///
/// Published rather than logged. A transcript answers "what happened" and this
/// answers "where is it now", and the second question is the one somebody
/// staring at a window has -- a run is minutes long and holds the model for
/// the whole of it, so a machine that looks identical to a hung one is a
/// machine somebody reboots.
#[derive(Clone)]
pub struct Progress {
    pub name: String,
    pub step: usize,
    pub budget: usize,
    pub met: usize,
    pub total: usize,
    /// The last verdict, verbatim, with its line number if it had one.
    ///
    /// Verbatim and not summarised. The verdict is what the loop itself acts
    /// on, and an operator reading a paraphrase of it cannot tell whether the
    /// loop is stuck on something real or on something the paraphrase lost.
    pub last: String,
    pub running: bool,
}

impl Progress {
    fn idle() -> Progress {
        Progress {
            name: String::new(),
            step: 0,
            budget: 0,
            met: 0,
            total: 0,
            last: String::new(),
            running: false,
        }
    }
}

static PROGRESS: crate::sync::Racy<Option<Progress>> = crate::sync::Racy::new(None);

// Stopping is `agent::request_abort`, not a flag of our own.
//
// A private one existed for as long as writing ran inline on whichever task
// asked. Once it joined the agent queue there were two ways to say "stop",
// gating two things that cannot both be running -- and two stop flags is how
// an operator ends up pressing the wrong one and watching nothing happen.

pub fn progress() -> Option<Progress> {
    unsafe { (*PROGRESS.get()).clone() }
}

fn publish(w: &Work, budget: usize, running: bool) {
    let (met, total) = tally(w);
    unsafe {
        *PROGRESS.get() = Some(Progress {
            name: w.name.clone(),
            step: w.steps,
            budget,
            met,
            total,
            last: w.last.clone().unwrap_or_else(|| String::from("nothing wrong so far")),
            running,
        });
    }
}

/// Clauses met, and how many there are.
///
/// Through `met`, which is what `done` counts and what the final report
/// prints. A second implementation here would be a progress window that
/// disagrees with the verdict it is watching.
fn tally(w: &Work) -> (usize, usize) {
    let v = met(w);
    (v.iter().filter(|x| x.ok).count(), v.len())
}

/// Write an application from a name and a goal, leaving a draft.
///
/// The whole of what "author something" means, in one place. It was eight
/// steps inlined in the shell -- build the contract, merge anything the
/// operator had already written into the draft's plan, generate, write the four
/// files -- and the autonomous trigger needs exactly the same eight. Two copies
/// of a sequence like that do not stay the same, and the half that drifts is
/// whichever one nobody is watching, which here is the one that runs at three
/// in the morning.
///
/// Leaves a draft and never adopts. Adoption is `app take`, and a machine that
/// installed what it wrote overnight without anybody reading it would be
/// answering a question nobody asked.
pub fn commission(name: &str, goal: &str, budget: usize) -> (Work, Report) {
    let mut w = Work::new(name, goal);
    // The contract. Two clauses is the floor and one of them has to ask
    // something of the program, or the empty application passes every check.
    w.plan = alloc::vec![Req::Title(String::from(name)), Req::Rows];
    // Anything already written into the draft's plan is added, so a contract
    // can be as specific as somebody wants it to be.
    if let Some(t) = crate::app::draft::plan(name) {
        for line in t.lines() {
            if let Some(r) = Req::parse(line) {
                if !w.plan.contains(&r) {
                    w.plan.push(r);
                }
            }
        }
    }
    // Opened before the run, not after: the whole point is watching it happen.
    // It does not take focus -- `open_authoring` hands the keyboard back to
    // the terminal, for the reason Minesweeper established, which is that a
    // window holding focus eats the next serial command line a byte at a time.
    crate::gfx::desk::open_authoring();
    let report = generate(&mut w, budget);
    // Left on screen with `running` clear rather than closed. The result is
    // what somebody who walked away came back for, and a window that vanishes
    // when the work finishes is a window that only ever shows the boring part.
    publish(&w, budget, false);
    crate::gfx::desk::draw();
    crate::app::draft::set_panel(name, &panel_of(&w));
    crate::app::draft::set_code(name, &w.code);
    let mut plan = String::new();
    for r in &w.plan {
        plan.push_str(&r.render());
        plan.push('\n');
    }
    crate::app::draft::set_plan(name, &plan);
    crate::app::draft::note(name, &describe(&report));
    (w, report)
}

/// One line per run, appended and never rewritten.
///
/// The widget count is the number that matters and is the reason this is not
/// just a log line. Every check in this tree can be satisfied by a panel with a
/// title and a close button, so a loop that has quietly settled on writing that
/// reports success every time. If the median authored application has two
/// widgets, the ledger says the loop is not working while the reports say it
/// is.
pub const LEDGER: &str = "/app/authored.txt";

pub fn record(w: &Work, report: &Report, how: &str) {
    let mut text = crate::sysbox::read_blob(LEDGER)
        .map(|b| alloc::string::String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    text.push_str(&alloc::format!(
        "{}\t{}\t{} step(s)\t{} of {} clause(s)\t{} widget(s)\t{}\n",
        crate::dev::rtc::now()
            .map(|t| alloc::format!(
                "{:04}-{:02}-{:02} {:02}:{:02}",
                t.year, t.month, t.day, t.hour, t.minute
            ))
            .unwrap_or_else(|| alloc::string::String::from("-")),
        w.name,
        w.steps,
        report.met,
        report.total,
        w.panel.len(),
        how
    ));
    crate::sysbox::write_text(LEDGER, &text);
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

    // Every body, filled in, is a program that parses and runs under the
    // sandbox the generated application will run under.
    //
    // The defence `uidoc`'s wildcard-free `fixture_for` gives, in the one
    // other place it matters: a body that cannot pass the gate a generated
    // application passes is a bug waiting for whatever Tuesday somebody asks
    // for that shape. It is cheap to hold -- a parse per body at boot -- and
    // the failure it catches is otherwise invisible, because the loop reports
    // an unmet clause and looks like a model that could not manage it.
    for (kind, pre, body) in BODIES {
        let filled = alloc::format!("{}{}", pre, body.replace("{NAME}", "thing"));
        let v = check::check_code(&filled, "/draft/selftest");
        if !v.ok {
            crate::kprintln!("  body '{}': {}", kind, v.why);
            return false;
        }
        // ...and calling it does not fault, which a parse does not establish.
        // Nothing is written: `here()` is the jail, the jail is a path nothing
        // has created, and every body reads before it writes.
        let mut it = crate::aiksi::Interp::sandboxed("/draft/selftest");
        if crate::aiksi::eval_line(&mut it, &filled).is_err() {
            return false;
        }
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

// --- the model ------------------------------------------------------------
//
// Everything above decides nothing. This is the only part that asks, and it
// asks in the narrowest way available: every choice is an index into a list
// this file wrote, and the one thing that is free text is a label.
//
// The engine is taken and released **per decode**, not held across the run.
// `with_engine` excludes every other task for the length of the call, so a
// twelve-step authoring run that held it would answer "another task holds it"
// to the shell for minutes. Per-decode, the shell waits seconds. The cost is
// one prefill for each decode instead of continuing from the last one -- which
// is what `deliberate` does and why it can continue a decode -- and that is
// worth paying for a loop that can be interrupted anywhere.

use super::constrain::{Cursor, Grammar};
use super::{harness, sample, with_engine};

/// How much whitespace a decode may emit before it is called undecided.
///
/// Higher than `constrain::MAX_LEADING_SPACES`, which is sized for a
/// single-line prompt ending in `Tool:`. These prompts are several lines and a
/// model shown a list tends to lay out before it commits. Bounded all the same:
/// an allowance is not a licence.
const IDLE_ALLOWANCE: usize = 12;

/// Pick one of a list, under a grammar that makes anything else unreachable.
///
/// Returns the index, because `Cursor::finished` indexes the same list it was
/// built from. `None` means the decode ran out of steps without completing an
/// alternative, which is a small model failing to commit rather than an error.
pub fn choose(prompt: &str, options: &[&str]) -> Option<usize> {
    if options.is_empty() {
        return None;
    }
    // One option is not a decision. Skipping the decode saves a prefill and a
    // handful of forward passes per step, and removes the only way a forced
    // move can fail.
    if options.len() == 1 {
        return Some(0);
    }
    let grammar = Grammar::new(options.iter().copied());
    let bound = super::constrain::step_bound(&grammar);
    // The alternatives go in the prompt, and it ends on a cue.
    //
    // Not decoration. Without them the model has nothing telling it a short
    // word is wanted, and greedily emits whitespace -- which the cursor counts
    // as idle rather than progress, so the decode spends its allowance and
    // returns having chosen nothing. Measured on the first attempt: 57
    // candidates admitted, zero steps taken. `harness::prompt_for` lists the
    // tools for exactly this reason, and this is that shape.
    let full = alloc::format!("{} Options: {}. Pick:", prompt, options.join(", "));
    harness::with_alphabet(|alphabet| {
        with_engine(|e| {
            let tokens = e.tok.encode(&full, true, false);
            let limit = e.model.cfg.seq_len;
            // `prefill` and not a forward loop. `harness::choose` still feeds
            // its prompt one token at a time, which is the slow path and the
            // reason a single decision was once measured in minutes.
            let mut pos = e.model.prefill(&mut e.state, &tokens, 0).min(limit);
            let mut cursor = Cursor::new(&grammar);
            let mut steps = 0;
            let mut idle = 0;
            let mut found = None;
            while steps < bound && idle <= IDLE_ALLOWANCE && pos < limit {
                let candidates = cursor.candidates(alphabet);
                // Sampled, not greedy.
                //
                // At temperature zero the choice is a fixed point of the
                // state, so a model that prefers a whitespace token prefers it
                // again on the next step and the decode spends its whole
                // allowance without committing. A little temperature is what
                // lets it out. `harness::choose` is driven at 1.0 for the same
                // reason.
                let Some(next) =
                    sample::sample_among(&e.state.logits, &candidates, 0.7, 0.0, &mut e.rng)
                else {
                    break;
                };
                if cursor.push(alphabet, next) {
                    steps += 1;
                } else {
                    idle += 1;
                }
                if let Some(i) = cursor.finished() {
                    found = Some(i);
                    break;
                }
                e.model.forward(&mut e.state, next, pos);
                pos += 1;
            }
            // On every path, success included. `e.pos` is a promise about the
            // cache that a prompt run from zero has already broken.
            harness::invalidate_conversation(e);
            if found.is_none() {
                // A constrained decode that runs out of steps is a small model
                // failing to commit, not an error -- but silence here is
                // indistinguishable from the engine being busy, and the two
                // want completely different responses.
                crate::serial_println!(
                    "[author] no choice among {} after {} step(s)",
                    options.len(),
                    steps
                );
            }
            found
        })
    })
    .flatten()
    .flatten()
}

/// The one free-text decode: a short human-readable label.
///
/// Clipped hard and stopped at the first newline, and the bytes come from
/// `Alphabet`, which is the *decoded* form. `token_bytes` returns what the
/// vocabulary literally holds, and on a v1 checkpoint a newline is stored as
/// the text `<0x0A>` -- so the stop never fires and the budget is spent every
/// time. That bug is live in `agent::propose` on Qwen3 and invisible on the
/// checkpoint QEMU runs.
pub fn phrase(prompt: &str, clip: usize) -> Option<String> {
    harness::with_alphabet(|alphabet| {
        with_engine(|e| {
            let tokens = e.tok.encode(prompt, true, false);
            let limit = e.model.cfg.seq_len;
            let mut pos = e.model.prefill(&mut e.state, &tokens, 0).min(limit);
            let eos = e.tok.eos();
            let vocab = e.tok.vocab_size();
            let all: alloc::vec::Vec<u32> = (0..vocab as u32).collect();
            let mut raw: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
            for _ in 0..PHRASE_TOKENS {
                if pos >= limit || raw.len() >= clip {
                    break;
                }
                let Some(next) = sample::sample_among(&e.state.logits, &all, 0.0, 0.0, &mut e.rng)
                else {
                    break;
                };
                if next == eos {
                    break;
                }
                let piece = alphabet.piece(next);
                let stop = piece.iter().position(|&b| b == b'\n' || b == b'\t');
                raw.extend_from_slice(&piece[..stop.unwrap_or(piece.len())]);
                let done = stop.is_some() || raw.len() >= clip;
                e.model.forward(&mut e.state, next, pos);
                pos += 1;
                if done {
                    break;
                }
            }
            harness::invalidate_conversation(e);
            let t = alloc::string::String::from_utf8_lossy(&raw).trim().to_string();
            // Only printable ASCII survives: it is going into a document whose
            // operands are defined to be exactly that, so anything else would
            // be rejected by the parser a moment later with a worse message.
            let t: alloc::string::String =
                t.chars().filter(|c| (' '..='~').contains(c)).collect();
            if t.is_empty() {
                None
            } else {
                Some(t)
            }
        })
    })
    .flatten()
    .flatten()
}

/// Tokens a label may take. Twelve is roughly forty bytes of English.
const PHRASE_TOKENS: usize = 12;
const LABEL_CLIP: usize = 24;

/// What the model is told. Short on purpose.
///
/// `seq_len` is 512 and the artifacts are large, so the prompt carries the
/// goal, what is still unmet, and the last thing that went wrong -- never the
/// whole panel or the whole program. The shape is the bare completion style
/// `harness::prompt_for` uses, not ChatML: the probe's features were fitted on
/// that form, and a second convention inside the loop would compute them over
/// prompts they were never fitted on.
pub fn prompt(w: &Work) -> String {
    let mut s = alloc::format!("App: {}\nWant:", w.goal);
    let m = met(w);
    let mut unmet = 0;
    for (r, v) in w.plan.iter().zip(m.iter()) {
        if !v.ok {
            s.push(' ');
            s.push_str(&r.render());
            s.push(',');
            unmet += 1;
        }
    }
    if unmet == 0 {
        s.push_str(" nothing");
    }
    s.push_str(&alloc::format!("\nHave: {} line(s)", w.panel.len()));
    if let Some(l) = &w.last {
        s.push_str(&alloc::format!("\nLast: {}", l));
    }
    s.push_str("\nNext:");
    s
}

/// Functions the contract asked for that the program does not have.
///
/// The only names the loop may add. A model inventing one would produce a
/// function nothing calls and no clause wanted.
fn missing_fns(w: &Work) -> Vec<&str> {
    let have = check::functions(&w.code);
    w.plan
        .iter()
        .filter_map(|r| match r {
            Req::Fn(f) => Some(f.as_str()),
            _ => None,
        })
        .filter(|f| !have.iter().any(|(d, _)| d == f))
        .collect()
}

/// The actions that can be carried out from here, in a fixed order.
fn applicable(w: &Work) -> Vec<&'static str> {
    let mut v = alloc::vec!["skel"];
    // Nothing can be added to a panel before there is one, and a widget naming
    // an action needs a function to name.
    if !w.panel.is_empty() && !check::functions(&w.code).is_empty() {
        v.push("panel");
    }
    if w.panel.len() > 1 {
        v.push("drop");
    }
    if !missing_fns(w).is_empty() {
        v.push("fn");
    }
    // Finishing is not applicable while the contract is unmet.
    //
    // Offered unconditionally, it is the cheapest thing in the list and the
    // model takes it: the first driven run chose `done` as its first action and
    // stopped with an empty panel, reporting one step and nothing built. The
    // step budget is what ends a run that is not converging; the model does not
    // get to declare an unmet contract finished.
    if met(w).iter().all(|x| x.ok) {
        v.push("done");
    }
    v
}

/// One step, decided by the model.
///
/// Every operand is an index into a list built here, except a label. A button
/// therefore cannot name a function the program does not define -- not because
/// it is checked afterwards, but because the alternative was never in the
/// grammar. That is the same preference constrained decoding was adopted for.
pub fn propose(w: &Work) -> Option<Action> {
    let p = prompt(w);
    // Only what can actually be done from here.
    //
    // Offering an action that cannot be carried out means the model picks it,
    // nothing can be built, and the run ends having taken no step at all --
    // which is what happened the first time this was driven: `fn` was chosen
    // with no unmet function clause to serve, and the whole run stopped. The
    // same principle as everywhere else here: do not check afterwards for
    // something that can be made unreachable.
    let verbs = applicable(w);
    if verbs.is_empty() {
        return Some(Action::Done);
    }
    let v = verbs[choose(&p, &verbs)?];
    match v {
        "done" => Some(Action::Done),
        "skel" => {
            let kinds = skel::kinds();
            let i = choose(&alloc::format!("{} skel", p), &kinds)?;
            Some(Action::Skel(kinds[i].to_string()))
        }
        "fn" => {
            let wanted = missing_fns(w);
            if wanted.is_empty() {
                return None;
            }
            let name = wanted[choose(&alloc::format!("{} fn", p), &wanted)?].to_string();
            let kinds = body_kinds();
            let k = choose(&alloc::format!("{} fn {} body", p, name), &kinds)?;
            Some(Action::Fn(name, kinds[k].to_string()))
        }
        "drop" => {
            let lines: Vec<String> = (1..=w.panel.len()).map(|n| alloc::format!("{}", n)).collect();
            let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
            if refs.is_empty() {
                return None;
            }
            let i = choose(&alloc::format!("{} drop", p), &refs)?;
            Some(Action::Drop(i + 1))
        }
        "panel" => {
            let lines: Vec<String> =
                (1..=w.panel.len() + 1).map(|n| alloc::format!("{}", n)).collect();
            let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
            let at = choose(&alloc::format!("{} panel", p), &refs)? + 1;
            let line = compose(w, &p)?;
            Some(Action::Panel(at, line))
        }
        _ => None,
    }
}

/// Build one widget line, choosing everything that can be chosen.
fn compose(w: &Work, p: &str) -> Option<String> {
    let verbs = widget_verbs();
    let verb = verbs[choose(&alloc::format!("{} widget", p), &verbs)?];
    match verb {
        "sep" => Some("sep".to_string()),
        "heading" | "label" => {
            let t = phrase(&alloc::format!("{} widget {} text:", p, verb), LABEL_CLIP)?;
            Some(alloc::format!("{}\t{}", verb, t))
        }
        "button" | "item" => {
            // The action is fully constrained: one alternative per function the
            // program actually has. There is no way to name a missing one.
            let fns = check::functions(&w.code);
            let acts: Vec<String> = fns
                .iter()
                .filter(|(_, arity)| *arity == 0)
                .map(|(f, _)| alloc::format!("run app {} {}", w.name, f))
                .collect();
            if acts.is_empty() {
                return None;
            }
            let refs: Vec<&str> = acts.iter().map(|s| s.as_str()).collect();
            let a = refs[choose(&alloc::format!("{} widget {} action", p, verb), &refs)?];
            let label = phrase(&alloc::format!("{} widget {} label:", p, verb), LABEL_CLIP)?;
            Some(alloc::format!("{}\t{}\t{}", verb, a, label))
        }
        _ => None,
    }
}

/// Run the loop with the model deciding, releasing the engine each step.
pub fn generate(w: &mut Work, budget: usize) -> Report {
    publish(w, budget, true);
    crate::gfx::desk::draw();
    while w.steps < budget {
        // The same flag the episode loop checks, cleared by the agent task
        // when the job ends.
        if crate::ai::agent::aborted() {
            w.last = Some(String::from("stopped"));
            break;
        }
        let Some(a) = propose(w) else { break };
        // Everything between decodes runs with the engine free, so the shell
        // and the desktop are locked out for one decode at a time rather than
        // for the whole run.
        let stop = matches!(a, Action::Done);
        let script = alloc::vec![a];
        run(w, &script, budget);
        // The window repaints through the diffed present, so refreshing per
        // step costs only what actually changed -- the same bargain the
        // episode transcript makes.
        publish(w, budget, true);
        crate::gfx::desk::draw();
        crate::task::yield_now();
        if stop || done(w).is_some() {
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
            why: w.last.clone().unwrap_or_else(|| "out of steps".to_string()),
        }
    })
}

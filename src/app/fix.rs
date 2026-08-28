//! Repairing an application instead of describing what is wrong with it.
//!
//! `check` answers with a line number and a reason, which is what the authoring
//! loop needs and is not what a person wants. Being told that line seven names
//! a function that does not exist leaves the work where it was. This does the
//! work.
//!
//! ### What may be repaired automatically
//!
//! Only removals, and only of things that are already inert. A widget naming a
//! function nobody wrote does nothing when pressed; a line that does not parse
//! is not drawn at all; a second field is typed into and ignored. Taking any of
//! those out changes what the application *is* on paper and changes nothing
//! about what it does, which is the property that makes a repair safe to apply
//! without asking.
//!
//! Nothing here writes a function, guesses a name, or repoints an action at a
//! function that looks similar. Those are changes of behaviour, and a
//! troubleshooter that quietly changes behaviour is worse than one that does
//! nothing: the operator stops being able to trust that what they read is what
//! they wrote.
//!
//! ### Why it is safe to run on an adopted application
//!
//! A repair produces a new version whose parent is the one it replaced, stored
//! and adopted through `manifest`. `app rollback` undoes it with a pointer
//! write, because the parent node was never deleted. That is the whole return
//! on having made identity content-addressed, and it is what lets this be
//! automatic rather than advisory.

use crate::sysbox;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::{check, manifest, ROOT};

pub struct Repair {
    pub what: String,
    /// False when the trouble was found and left alone, which is a result and
    /// not a failure -- it is the honest answer for anything only a person can
    /// decide.
    pub done: bool,
}

pub struct Outcome {
    pub repairs: Vec<Repair>,
    pub before: usize,
    pub after: usize,
    pub adopted: Option<[u8; 32]>,
}

/// How many times to look again after changing something.
///
/// Removing a line renumbers everything below it, so one pass can only trust
/// its first finding. Looking again is cheaper than reasoning about offsets,
/// and bounded because a repair that keeps finding work is a repair that is not
/// making progress.
const PASSES: usize = 8;

/// Put right what can be put right, and say what was done.
pub fn fix(name: &str) -> Outcome {
    let panel_path = alloc::format!("{}/{}/panel.ui", ROOT, name);
    let code_path = alloc::format!("{}/{}/code.l", ROOT, name);
    let mut out =
        Outcome { repairs: Vec::new(), before: 0, after: 0, adopted: None };

    let (Some(panel), Some(code)) = (text(&panel_path), text(&code_path)) else {
        out.repairs.push(Repair {
            what: alloc::format!("there is no application called '{}'", name),
            done: false,
        });
        return out;
    };

    let mut lines: Vec<String> = panel.lines().map(|l| l.to_string()).collect();
    out.before = trouble(name, &lines, &code);

    for _ in 0..PASSES {
        let Some((at, why)) = first_fault(name, &lines, &code) else {
            break;
        };
        let i = at.saturating_sub(1);
        if i >= lines.len() {
            // A fault with no line to remove -- "nothing calls the program" is
            // the one that matters, and it is not something a removal fixes.
            out.repairs.push(Repair { what: why, done: false });
            break;
        }
        // A document needs its header and its title, so removing either would
        // turn one broken thing into a different broken thing.
        let verb = lines[i].split('\t').next().unwrap_or("");
        if verb == "panel" || verb == "title" {
            out.repairs.push(Repair {
                what: alloc::format!("line {} is the document's {}: {}", at, verb, why),
                done: false,
            });
            break;
        }
        out.repairs.push(Repair {
            what: alloc::format!("removed line {} ({}): {}", at, verb, why),
            done: true,
        });
        lines.remove(i);
    }

    out.after = trouble(name, &lines, &code);
    if out.repairs.iter().any(|r| r.done) {
        let mut text = String::new();
        for l in &lines {
            text.push_str(l);
            text.push('\n');
        }
        // The bytes of what is being replaced, kept before it is replaced.
        // `adopt` preserves too, but by then this file has already been
        // overwritten and the version to go back to would be gone.
        manifest::preserve(name);
        // An application nobody ever adopted has no recorded version, so the
        // repair would have no parent and could not be undone -- which is the
        // usual case, because the seeded applications arrive without one. Record
        // what it was first. The broken state is a version like any other, and
        // being able to return to it is the whole point of saying so.
        if manifest::head(name).is_none() {
            if let Some(was) = manifest::current(name) {
                let h = was.store();
                manifest::adopt(name, &h);
            }
        }
        sysbox::write_text(&panel_path, &text);
        // A new version, with the old one as its parent. `app rollback` is a
        // pointer write away, which is what makes repairing in place honest.
        if let Some(m) = manifest::current(name) {
            let h = m.store();
            manifest::adopt(name, &h);
            out.adopted = Some(h);
        }
    }
    out
}

fn text(path: &str) -> Option<String> {
    sysbox::read_blob(path).map(|b| String::from_utf8_lossy(&b).into_owned())
}

fn joined(lines: &[String]) -> String {
    let mut s = String::new();
    for l in lines {
        s.push_str(l);
        s.push('\n');
    }
    s
}

/// The panel as the desktop parses it: without the row directive, which belongs
/// to the application layer and is not a widget.
fn without_rows(panel: &str) -> String {
    let mut s = String::new();
    for l in panel.lines() {
        if !l.starts_with("rows\t") {
            s.push_str(l);
            s.push('\n');
        }
    }
    s
}

/// How many things are wrong.
fn trouble(name: &str, lines: &[String], code: &str) -> usize {
    let panel = joined(lines);
    let mut n = 0;
    if !check::check_panel(&without_rows(&panel)).ok {
        n += 1;
    }
    n + check::check_refs(name, &panel, code)
        .iter()
        .filter(|v| !v.ok)
        .count()
}

/// The first fault, and the line it is on.
///
/// One at a time, because removing a line renumbers the rest and a list of
/// faults gathered before a change is a list of stale line numbers after it.
fn first_fault(name: &str, lines: &[String], code: &str) -> Option<(usize, String)> {
    let panel = joined(lines);
    // Parse failures first: a document that does not parse makes every later
    // verdict a claim about something nobody can read. The line has to be
    // mapped back, because the row directive was taken out before parsing.
    let v = check::check_panel(&without_rows(&panel));
    if !v.ok {
        return Some((v.line.map(|l| map_line(&panel, l)).unwrap_or(0), v.why));
    }
    check::check_refs(name, &panel, code)
        .iter()
        .find(|x| !x.ok)
        .map(|x| (x.line.unwrap_or(0), x.why.clone()))
}

/// Translate a line number in the stripped document back to the original.
///
/// Without this a repair removes the wrong line whenever a `rows` directive
/// sits above the fault -- and removing the wrong line is exactly the sort of
/// help that makes an automatic repair untrustworthy.
fn map_line(panel: &str, stripped_line: usize) -> usize {
    let mut seen = 0;
    for (i, l) in panel.lines().enumerate() {
        if l.starts_with("rows\t") {
            continue;
        }
        seen += 1;
        if seen == stripped_line {
            return i + 1;
        }
    }
    0
}

pub fn selftest() -> bool {
    // The line mapping, which is the part that would silently remove the wrong
    // line if it were wrong.
    let panel = "panel\t1\ntitle\tA\nrows\tr\nbutton\tclose\tX\n";
    // Stripped, `button` is line 3; in the original it is line 4.
    if map_line(panel, 3) != 4 {
        return false;
    }
    if map_line(panel, 1) != 1 || map_line(panel, 2) != 2 {
        return false;
    }
    if map_line(panel, 99) != 0 {
        return false;
    }

    // A fault is found, on the right line of the original.
    let lines: Vec<String> = "panel\t1\ntitle\tA\nrows\tr\nnonsense\tx\n"
        .lines()
        .map(|l| l.to_string())
        .collect();
    match first_fault("a", &lines, "fn r() { return \"\" }") {
        Some((4, _)) => {}
        _ => return false,
    }

    // A reference fault is found by line too, and counted.
    let refs: Vec<String> = "panel\t1\ntitle\tA\nbutton\trun app a ghost\tGo\n"
        .lines()
        .map(|l| l.to_string())
        .collect();
    match first_fault("a", &refs, "fn r() { return \"\" }") {
        Some((3, _)) => {}
        _ => return false,
    }
    if trouble("a", &refs, "fn r() { return \"\" }") == 0 {
        return false;
    }
    // ...and a sound application has nothing wrong with it.
    let good: Vec<String> = "panel\t1\ntitle\tA\nrows\tr\n"
        .lines()
        .map(|l| l.to_string())
        .collect();
    trouble("a", &good, "fn r() { return \"\" }") == 0
}

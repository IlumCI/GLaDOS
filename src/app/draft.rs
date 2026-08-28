//! Applications being written, before anybody agrees to run them.
//!
//! A draft lives at `/draft/<name>` and holds the same two files an
//! application does, plus the record of how it got there:
//!
//! ```text
//! /draft/<name>/plan.txt   what it is meant to do, one clause per line
//! /draft/<name>/panel.ui   under construction
//! /draft/<name>/code.ai&xi     under construction
//! /draft/<name>/log.txt    what happened while it was written
//! ```
//!
//! ### Why a sibling root and not `/app/<name>/draft`
//!
//! Three things break if a draft lives under `/app`. `app::names()` is the
//! children of `/app`, so a half-written application appears in the launcher.
//! `manifest::current` hashes `/app/<name>/panel.ui`, so every edit churns the
//! identity a grant is pinned to. And `Interp::sandboxed` jails a program to
//! its own subtree, so a draft under `/app/<name>` would smoke-test by writing
//! into the live application's state.
//!
//! A sibling root gives the draft the same jail *shape* with a different jail,
//! which is exactly what smoke testing needs: it writes scratch state, and the
//! scratch state is thrown away with the draft.
//!
//! ### Adoption is a copy, and the plan does not come with it
//!
//! `/app/<name>` ends up holding exactly the two files an application is
//! defined to be. The plan and the log stay in `/draft`, where records belong.
//! That is what "cleaning up the plan file" means here: nothing is deleted to
//! tidy anything away, and the account of how an application came to exist
//! remains where it can be read.

use crate::sysbox;
use alloc::string::String;
use alloc::vec::Vec;

use super::{check, manifest, skel};

pub const ROOT: &str = "/draft";

fn p(name: &str, file: &str) -> String {
    alloc::format!("{}/{}/{}", ROOT, name, file)
}

fn text_at(path: &str) -> Option<String> {
    sysbox::read_blob(path).map(|b| String::from_utf8_lossy(&b).into_owned())
}

pub fn exists(name: &str) -> bool {
    text_at(&p(name, "code.ai&xi")).is_some()
}

pub fn names() -> Vec<String> {
    sysbox::children(ROOT)
}

pub fn panel(name: &str) -> Option<String> {
    text_at(&p(name, "panel.ui"))
}

pub fn code(name: &str) -> Option<String> {
    text_at(&p(name, "code.ai&xi"))
}

pub fn plan(name: &str) -> Option<String> {
    text_at(&p(name, "plan.txt"))
}

pub fn set_panel(name: &str, text: &str) -> bool {
    sysbox::write_text(&p(name, "panel.ui"), text)
}

pub fn set_code(name: &str, text: &str) -> bool {
    sysbox::write_text(&p(name, "code.ai&xi"), text)
}

pub fn set_plan(name: &str, text: &str) -> bool {
    sysbox::write_text(&p(name, "plan.txt"), text)
}

/// Append one line to the draft's account of itself.
pub fn note(name: &str, line: &str) {
    let mut t = text_at(&p(name, "log.txt")).unwrap_or_default();
    t.push_str(line);
    t.push('\n');
    sysbox::write_text(&p(name, "log.txt"), &t);
}

/// Start a draft from a skeleton.
pub fn create(name: &str, kind: &str, title: &str, label: &str) -> Result<(), String> {
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err(String::from("a name is letters, digits and dashes"));
    }
    if super::RESERVED.contains(&name) {
        return Err(alloc::format!("'{}' is a command, not a name", name));
    }
    let Some((panel, code)) = skel::fill(kind, name, title, label) else {
        return Err(alloc::format!(
            "no skeleton called '{}' -- try one of: {}",
            kind,
            skel::kinds().join(", ")
        ));
    };
    set_panel(name, &panel);
    set_code(name, &code);
    note(name, &alloc::format!("from the {} skeleton", kind));
    Ok(())
}

/// Everything `check` can say about a draft, including the parts that need
/// running it.
///
/// Smoke testing is destructive and this is where it is safe: the state it
/// writes belongs to the draft and goes when the draft does.
pub fn verdicts(name: &str) -> Vec<check::Verdict> {
    let mut v = check::check_all(ROOT, name);
    if v.iter().all(|x| x.ok) {
        v.extend(check::smoke(ROOT, name));
    }
    v
}

/// Move a draft into `/app`, and record what was adopted.
///
/// Refuses unless every check passes. A half-built application the *machine*
/// adopted is worse than none: `app::panel` shows a parse-error panel for a
/// broken one, which is the right answer for something an operator chose and
/// the wrong answer for something that arrived on its own.
pub fn adopt(name: &str) -> Result<[u8; 32], String> {
    let (Some(panel), Some(code)) = (panel(name), code(name)) else {
        return Err(alloc::format!("no draft called '{}'", name));
    };
    let bad: Vec<String> = verdicts(name)
        .iter()
        .filter(|v| !v.ok)
        .map(|v| match v.line {
            Some(l) => alloc::format!("line {}: {}", l, v.why),
            None => v.why.clone(),
        })
        .collect();
    if !bad.is_empty() {
        return Err(alloc::format!("not adopted -- {}", bad.join("; ")));
    }

    sysbox::write_text(&alloc::format!("{}/{}/panel.ui", super::ROOT, name), &panel);
    sysbox::write_text(&alloc::format!("{}/{}/code.ai&xi", super::ROOT, name), &code);
    let Some(m) = manifest::current(name) else {
        return Err(String::from("adopted, but could not describe what was adopted"));
    };
    let h = m.store();
    manifest::adopt(name, &h);
    note(name, "adopted");
    Ok(h)
}

/// Throw a draft away. The manifests and their blobs are content addressed and
/// are not touched, so nothing that was ever adopted is lost by this.
pub fn abandon(name: &str) -> bool {
    sysbox::detach(&alloc::format!("{}/{}", ROOT, name))
}

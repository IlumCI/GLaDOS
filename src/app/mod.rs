//! Applications: a panel document, a program, and nothing else.
//!
//! An application here is two blobs in the namespace. `panel.ui` is what it
//! looks like, in the format `gfx::uidoc` reads. `code.l` is what it does, in
//! the language the shell already runs. Both are text, both are addressed by
//! the hash of their own bytes, and neither needs the model present to work.
//!
//! That is the whole point of the shape. An interface a model re-imagines on
//! every interaction is a picture of a program; there is nothing to read, diff,
//! version or trust, and it stops existing the moment the model is unavailable.
//! An application the model *writes* is a program: it parses or it does not, it
//! runs the same way twice, and it can be examined by anyone afterwards.
//!
//! ### Each action re-runs the program
//!
//! `call` loads `code.l` into a **fresh** interpreter every time and throws it
//! away afterwards. That is deliberate and it is the expensive-looking choice.
//! A resident interpreter would make an app's behaviour depend on every action
//! taken since boot, which is exactly what makes a bug in one unreproducible.
//! Re-running means the only inputs are the file and the namespace, so the same
//! press gives the same result on a machine that has been up for a month. The
//! cost is lexing a few hundred bytes per press.
//!
//! Neither is it the shell's interpreter, or `sysbox`'s. The shell's dies with
//! the session and cannot be read by `cat`; `sysbox::TOOLS` is a *different*
//! interpreter that the model's own tools run in, so an app keeping state there
//! and a tool reading it would disagree about what the app contains.
//!
//! State goes in the namespace, under the app's own subtree, where `cat` can
//! see it and a snapshot captures it.
//!
//! ### What an application may do
//!
//! `code.l` runs in a sandboxed interpreter: no raw memory, no I/O ports, no
//! drawing outside a window, no applet that changes anything, and writes only
//! under `/app/<name>`. The gate is in `lang::Interp` rather than here, because
//! the raw builtins are also reachable from a bare expression at the prompt and
//! from any program `run` executes -- a check anywhere else would have a hole
//! shaped like the path it did not cover.
//!
//! Reads are not restricted. This is one address space with no process
//! isolation and `cat` is available to anything that can call an applet, so
//! pretending otherwise would be a fence with nothing behind it. What is
//! restricted is everything that *changes* something.
//!
//! An application may ask for more, by having a `raw` file beside its other
//! two. That is a request and not a grant: `app trust` is the only thing that
//! approves one, it approves a single manifest hash, and the hash covers both
//! files and the request, so approval cannot survive an edit or be inherited
//! by a later version. See `manifest`.

pub mod check;
pub mod draft;
pub mod fix;
pub mod manifest;
pub mod skel;

use crate::gfx::ui::Panel;
use crate::gfx::uidoc;
use crate::lang;
use crate::sysbox;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Where applications live.
pub const ROOT: &str = "/app";

fn code_path(name: &str) -> String {
    alloc::format!("{}/{}/code.l", ROOT, name)
}

fn panel_path(name: &str) -> String {
    alloc::format!("{}/{}/panel.ui", ROOT, name)
}

/// Names that are commands rather than applications.
///
/// `app list` and `app show x` have to mean something, and an application
/// called `list` would take the word away. Reserved rather than escaped: there
/// are two of them and renaming an app is free.
pub const RESERVED: &[&str] = &[
    "list", "show", "check", "fix", "info", "trust", "adopt", "rollback", "draft", "try",
    "take", "drop",
];

pub fn exists(name: &str) -> bool {
    sysbox::read_blob(&code_path(name)).is_some()
}

/// Every application, by name.
pub fn names() -> Vec<String> {
    sysbox::children(ROOT)
}

/// Evaluate an expression in an application's own program.
///
/// The program is loaded whole and then the expression is evaluated against it,
/// which is why `code.l` must parse as a program and not as a sequence of
/// independent lines.
pub fn call(name: &str, expr: &str) -> Result<String, String> {
    let Some(bytes) = sysbox::read_blob(&code_path(name)) else {
        return Err(alloc::format!("no application '{}'", name));
    };
    let src = String::from_utf8_lossy(&bytes).into_owned();
    // Sandboxed unless this exact version has been approved for more.
    //
    // "This exact version" is the whole mechanism: the manifest hash covers
    // both files and the request itself, so an approval cannot follow an edit
    // and cannot be inherited by the next version. An application that asks and
    // has not been granted simply runs sandboxed -- refusing to run at all
    // would make an unapproved request indistinguishable from a broken app.
    let trusted = manifest::current(name)
        .map(|m| m.raw && manifest::granted(&m.hash()))
        .unwrap_or(false);
    let mut it = if trusted {
        lang::Interp::new()
    } else {
        lang::Interp::sandboxed(&alloc::format!("{}/{}", ROOT, name))
    }
    // Bounded, because the desktop calls this. `document` runs the
    // application's row function on every repaint, and the full budget is
    // twenty million steps -- enough for a generated loop to make the window
    // manager feel broken while the symptom points at the compositor. A
    // program that needs more than this to build a list is a program that
    // should say so.
    .with_step_budget(lang::eval::DRAW_BUDGET);
    lang::eval_line(&mut it, &src)?;
    let v = lang::eval_line(&mut it, expr)?;
    Ok(v.render())
}

/// Build the call for one of an application's functions.
///
/// The argument arrives as whatever the operator typed or a field held, and it
/// is going into program text, so the two characters that could end the string
/// early are escaped. Nothing else is: an argument is data, and a language that
/// let one become code would make every generated app a way to run anything.
pub fn call_fn(name: &str, func: &str, arg: &str) -> Result<String, String> {
    if !func.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err("a function name is letters, digits and underscores".to_string());
    }
    let expr = if arg.is_empty() {
        alloc::format!("{}()", func)
    } else {
        let mut q = String::from("\"");
        for c in arg.chars() {
            if c == '\\' || c == '"' {
                q.push('\\');
            }
            q.push(c);
        }
        q.push('"');
        alloc::format!("{}({})", func, q)
    };
    call(name, &expr)
}

/// The application's panel document, with its live rows filled in.
///
/// A document is mostly static text. One verb is not: `rows` names a function
/// in the program, and the lines that function returns are spliced in where it
/// stood. That keeps the authored artifact authored -- the shape of the app is
/// in `panel.ui` where it can be read and hashed -- while the contents come
/// from wherever the app keeps them.
///
/// The splice happens before parsing, so what a program emits is checked by the
/// same parser and the same `Origin::Stored` gate as everything a person wrote.
/// A program cannot widen what its own panel is allowed to do.
pub fn document(name: &str) -> Option<String> {
    let bytes = sysbox::read_blob(&panel_path(name))?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let mut out = String::new();
    for line in text.lines() {
        match line.strip_prefix("rows\t") {
            None => {
                out.push_str(line);
                out.push('\n');
            }
            Some(func) => match call_fn(name, func.trim(), "") {
                Ok(rows) => {
                    out.push_str(&rows);
                    if !rows.ends_with('\n') && !rows.is_empty() {
                        out.push('\n');
                    }
                }
                // A failing row function leaves a visible reason rather than an
                // empty list, which would read as "you have no items".
                Err(e) => {
                    out.push_str("status\tbad\trows\t");
                    out.push_str(&e);
                    out.push('\n');
                }
            },
        }
    }
    Some(out)
}

/// An application as a panel, ready to put in a window.
pub fn panel(name: &str) -> Option<(String, Panel)> {
    let doc = document(name)?;
    match uidoc::parse(&doc, uidoc::Origin::Stored) {
        Ok(p) => Some((alloc::format!("{}", name), p)),
        // A broken document still opens, showing what is wrong with it. An app
        // that refuses to appear cannot be debugged from inside the machine it
        // is on.
        Err(e) => {
            let mut broken = String::from("panel\t1\ntitle\t");
            broken.push_str(name);
            broken.push('\n');
            broken.push_str("heading\tThis panel does not parse\n");
            broken.push_str(&alloc::format!("status\tbad\tline\t{}\n", e.line));
            broken.push_str("note\t");
            broken.push_str(&e.why);
            broken.push('\n');
            uidoc::parse(&broken, uidoc::Origin::Builtin)
                .ok()
                .map(|p| (alloc::format!("{}", name), p))
        }
    }
}

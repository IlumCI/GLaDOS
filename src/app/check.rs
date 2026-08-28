//! What can be known about an application without running it, and the little
//! that can only be known by running it.
//!
//! These are the checks a machine can make. They are exact, they cost no
//! forward pass, and each returns the line it failed on -- which is what makes
//! them usable as feedback to something that is writing the application rather
//! than reading it.
//!
//! ### Why this is not the model's job
//!
//! `agent.rs` already states the position: a model grading its own output is
//! "the feedback loop that amplifies its own errors". So the loop that will sit
//! on top of this proposes under grammars and is judged here, by arithmetic.
//! Nothing in this file has an opinion.
//!
//! ### What these checks cannot see
//!
//! All of them pass on an application that does nothing:
//!
//! ```text
//! panel   1
//! title   Calculator
//! button  close   Close
//! ```
//!
//! It parses, references nothing, renders, and fits. Seven green ticks and no
//! application. That is not a hole to be plugged here -- form checks cannot
//! know what was wanted -- and it is why the loop above must carry a contract
//! saying what the application is *for*, in clauses these functions can then
//! evaluate. Checking form is necessary and is not sufficient, and pretending
//! otherwise is how a generator converges on the empty valid answer and reports
//! success.

use crate::gfx::uidoc;
use crate::aiksi;
use crate::sysbox;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

pub struct Verdict {
    pub ok: bool,
    /// The line the failure is on, when it has one. This is the whole value of
    /// a check to something that is writing the file: "wrong" is not
    /// actionable, "line 7 is wrong" is.
    pub line: Option<usize>,
    pub why: String,
}

impl Verdict {
    pub fn ok(why: &str) -> Verdict {
        Verdict { ok: true, line: None, why: why.to_string() }
    }
    pub fn bad(why: String) -> Verdict {
        Verdict { ok: false, line: None, why }
    }
    pub fn at(line: usize, why: String) -> Verdict {
        Verdict { ok: false, line: Some(line), why }
    }
}

/// Does the panel document parse, under the rules a stored one lives by?
pub fn check_panel(text: &str) -> Verdict {
    match uidoc::parse(text, uidoc::Origin::Stored) {
        Ok(_) => Verdict::ok("panel parses"),
        Err(e) => Verdict::at(e.line, e.why),
    }
}

/// Does the program parse, and does it survive being loaded?
///
/// Loaded into a sandboxed interpreter with a small budget, because loading a
/// program runs its top level, and a top level with a loop in it would
/// otherwise hang the check rather than fail it.
pub fn check_code(src: &str, jail: &str) -> Verdict {
    let mut it = aiksi::Interp::sandboxed(jail).with_step_budget(aiksi::eval::DRAW_BUDGET);
    match aiksi::eval_line(&mut it, src) {
        Ok(_) => Verdict::ok("program loads"),
        Err(e) => Verdict::bad(e),
    }
}

/// The functions a program defines, with how many arguments each takes.
///
/// Read from the parsed program rather than scanned out of the text. A scan for
/// `fn name(` finds one inside a string literal or a comment and misses one
/// written across two lines; the parser already knows, exactly, and is the same
/// parser that will run it.
pub fn functions(src: &str) -> Vec<(String, usize)> {
    let Ok(toks) = aiksi::lex::lex(src) else {
        return Vec::new();
    };
    let Ok(prog) = aiksi::parse::parse(toks) else {
        return Vec::new();
    };
    prog.iter()
        .filter_map(|s| match s {
            aiksi::parse::Stmt::Fn(name, params, _) => Some((name.clone(), params.len())),
            _ => None,
        })
        .collect()
}

/// Every function the panel names, and whether the program has it.
///
/// The check that did not exist, and the one a small model fails most often. A
/// button whose action names a function nobody wrote parses perfectly, renders
/// perfectly, and does nothing when pressed -- invisible until somebody clicks
/// it, which for a generated application may be never. Here it is a line
/// number.
///
/// Arity is checked too, because it is free once the parser has been asked:
/// `apply` appends the field's text, so it calls with one argument, and a
/// function taking none would fail at the press rather than here.
pub fn check_refs(app: &str, panel: &str, code: &str) -> Vec<Verdict> {
    let defined = functions(code);
    let mut out = Vec::new();
    let mut referenced = 0usize;

    for (n, raw) in panel.lines().enumerate() {
        let line = n + 1;
        let mut parts = raw.split('\t');
        let verb = parts.next().unwrap_or("");
        let ops: Vec<&str> = parts.collect();

        // (function name, how many arguments the press will pass)
        let want: Option<(&str, usize)> = match verb {
            // `rows <fn>` calls it with nothing to build the list.
            "rows" => ops.first().map(|f| (f.trim(), 0)),
            "item" | "button" => ops.first().and_then(|a| call_in(a)),
            "field" => ops.get(1).and_then(|a| call_in(a)),
            _ => None,
        };
        let Some((func, args)) = want else { continue };
        referenced += 1;

        match defined.iter().find(|(d, _)| d == func) {
            None => out.push(Verdict::at(
                line,
                alloc::format!("'{}' is not a function this application defines", func),
            )),
            Some((_, arity)) if *arity != args => out.push(Verdict::at(
                line,
                alloc::format!(
                    "'{}' takes {} argument(s) and would be called with {}",
                    func, arity, args
                ),
            )),
            Some(_) => {}
        }
    }

    if referenced == 0 {
        out.push(Verdict::bad(alloc::format!(
            "nothing in {}'s panel calls its program",
            app
        )));
    }
    if out.is_empty() {
        out.push(Verdict::ok("every function the panel names exists"));
    }
    out
}

/// Pull the function and argument count out of an action operand.
///
/// An action is `run app <name> <fn> [args]` or `apply app <name> <fn>`, where
/// `apply` supplies the field's text as one argument. Anything else -- `close`,
/// `browse`, `none` -- calls nothing.
fn call_in(action: &str) -> Option<(&str, usize)> {
    let (verb, rest) = action.split_once(' ')?;
    let applies = match verb {
        "run" => false,
        "apply" => true,
        _ => return None,
    };
    let mut w = rest.split(' ');
    if w.next()? != "app" {
        return None;
    }
    let _app = w.next()?;
    let func = w.next()?;
    let typed = w.next().is_some();
    Some((func, usize::from(applies || typed)))
}

/// Run it, then look again.
///
/// Everything above checks form, and form is satisfied by a `rows` that works
/// on an empty list and breaks on a full one. This calls each function the
/// panel names and re-parses what the panel becomes afterwards, which is the
/// nearest thing to a test suite available without knowing what the
/// application is supposed to do.
///
/// Destructive by nature: it calls the application's own mutators. Only ever
/// point it at a draft.
pub fn smoke(root: &str, name: &str) -> Vec<Verdict> {
    let mut out = Vec::new();
    let panel_path = alloc::format!("{}/{}/panel.ui", root, name);
    let code_path = alloc::format!("{}/{}/code.ai&xi", root, name);
    let (Some(panel), Some(code)) = (
        sysbox::read_blob(&panel_path).map(|b| String::from_utf8_lossy(&b).into_owned()),
        sysbox::read_blob(&code_path).map(|b| String::from_utf8_lossy(&b).into_owned()),
    ) else {
        return alloc::vec![Verdict::bad(alloc::format!("{}/{} is not an application", root, name))];
    };

    let jail = alloc::format!("{}/{}", root, name);
    for (func, arity) in functions(&code) {
        let expr = if arity == 0 {
            alloc::format!("{}()", func)
        } else if arity == 1 {
            alloc::format!("{}(\"smoke\")", func)
        } else {
            continue;
        };
        let mut it =
            aiksi::Interp::sandboxed(&jail).with_step_budget(aiksi::eval::DRAW_BUDGET);
        if let Err(e) = aiksi::eval_line(&mut it, &code) {
            out.push(Verdict::bad(e));
            return out;
        }
        if let Err(e) = aiksi::eval_line(&mut it, &expr) {
            out.push(Verdict::bad(alloc::format!("{} failed: {}", expr, e)));
        }
    }

    // ...and the panel still parses once the program has done something. A row
    // function that emits a malformed line only does it when there is a row.
    let after = expand(root, name, &panel);
    let v = check_panel(&after);
    if !v.ok {
        out.push(Verdict::at(
            v.line.unwrap_or(0),
            alloc::format!("after running, the panel no longer parses: {}", v.why),
        ));
    }
    if out.is_empty() {
        out.push(Verdict::ok("runs, and still renders afterwards"));
    }
    out
}

/// Splice `rows` lines the way the desktop does, so the check sees what the
/// operator would.
fn expand(root: &str, name: &str, panel: &str) -> String {
    let code_path = alloc::format!("{}/{}/code.ai&xi", root, name);
    let code = sysbox::read_blob(&code_path)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    let jail = alloc::format!("{}/{}", root, name);
    let mut out = String::new();
    for line in panel.lines() {
        match line.strip_prefix("rows\t") {
            None => {
                out.push_str(line);
                out.push('\n');
            }
            Some(func) => {
                let mut it =
                    aiksi::Interp::sandboxed(&jail).with_step_budget(aiksi::eval::DRAW_BUDGET);
                let rows = aiksi::eval_line(&mut it, &code)
                    .and_then(|_| aiksi::eval_line(&mut it, &alloc::format!("{}()", func.trim())))
                    .map(|v| v.render())
                    .unwrap_or_default();
                out.push_str(&rows);
                if !rows.is_empty() && !rows.ends_with('\n') {
                    out.push('\n');
                }
            }
        }
    }
    out
}

/// Every check, cheapest first, over one application.
pub fn check_all(root: &str, name: &str) -> Vec<Verdict> {
    let panel_path = alloc::format!("{}/{}/panel.ui", root, name);
    let code_path = alloc::format!("{}/{}/code.ai&xi", root, name);
    let (Some(panel), Some(code)) = (
        sysbox::read_blob(&panel_path).map(|b| String::from_utf8_lossy(&b).into_owned()),
        sysbox::read_blob(&code_path).map(|b| String::from_utf8_lossy(&b).into_owned()),
    ) else {
        return alloc::vec![Verdict::bad(alloc::format!("{}/{} is not an application", root, name))];
    };
    let jail = alloc::format!("{}/{}", root, name);

    // Short-circuiting: a panel that does not parse makes every later verdict
    // about a document nobody can read, and a program that does not load makes
    // every reference check meaningless.
    //
    // Checked after expansion, because `rows` is a directive of this layer and
    // not a widget the codec knows -- the stored template is deliberately not a
    // valid document, and the thing worth checking is what the desktop will
    // actually be handed.
    let p = check_panel(&expand(root, name, &panel));
    if !p.ok {
        return alloc::vec![p];
    }
    let c = check_code(&code, &jail);
    if !c.ok {
        return alloc::vec![p, c];
    }
    let mut out = alloc::vec![p, c];
    out.extend(check_refs(name, &panel, &code));
    out
}

pub fn selftest() -> bool {
    // A panel naming a function nobody wrote, caught with its line number.
    let panel = "panel\t1\ntitle\tA\nrows\tmissing\n";
    let code = "fn present() { return \"\" }\n";
    let v = check_refs("a", panel, code);
    match v.first() {
        Some(x) if !x.ok && x.line == Some(3) => {}
        _ => return false,
    }
    // The same panel naming one that exists.
    let ok = check_refs("a", "panel\t1\ntitle\tA\nrows\tpresent\n", code);
    if !ok.iter().all(|v| v.ok) {
        return false;
    }
    // Arity: `apply` passes the field's text, so a zero-argument function is
    // wrong here and would fail at the press rather than at the check.
    let bad_arity = "panel\t1\ntitle\tA\nfield\tf\tapply app a present\t\n";
    if check_refs("a", bad_arity, code).iter().all(|v| v.ok) {
        return false;
    }
    let one = "fn present(x) { return x }\n";
    if !check_refs("a", bad_arity, one).iter().all(|v| v.ok) {
        return false;
    }
    // A panel that calls nothing is refused: it is the shape an empty
    // application takes, and every other check passes on it.
    if check_refs("a", "panel\t1\ntitle\tA\nbutton\tclose\tClose\n", code)
        .iter()
        .all(|v| v.ok)
    {
        return false;
    }

    // Functions are read from the parse, so one named inside a comment or a
    // string is not a definition.
    let tricky = "// fn ghost() {}\nx = \"fn ghost() {}\"\nfn real(a, b) { return a }\n";
    let f = functions(tricky);
    if f.len() != 1 || f[0].0 != "real" || f[0].1 != 2 {
        return false;
    }
    // A program that does not parse defines nothing, rather than half a list.
    if !functions("fn (").is_empty() {
        return false;
    }

    // The document checks agree with the codec.
    if !check_panel("panel\t1\ntitle\tA\nsep\n").ok {
        return false;
    }
    match check_panel("panel\t1\ntitle\tA\nnope\tx\n") {
        v if !v.ok && v.line == Some(3) => {}
        _ => return false,
    }
    // A stored panel cannot reach the shell, and the check says so on the line.
    match check_panel("panel\t1\ntitle\tA\nbutton\trun reboot\tGo\n") {
        v if !v.ok && v.line == Some(3) => {}
        _ => return false,
    }
    // The program checks need no namespace, so they run here.
    if !check_code("fn a() { return 1 }", "/draft/x").ok {
        return false;
    }
    !check_code("fn (", "/draft/x").ok
}

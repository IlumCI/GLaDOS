//! Panels as text, both ways.
//!
//! `ui` rejected `Box<dyn Widget>` so that a constrained decoder could emit a
//! panel: an enum whose variants have a fixed operand count is writable in a
//! way a trait object is not. That decision has been sitting there without the
//! thing it was made for. This is that thing -- a document format, a parser
//! that refuses, and a serialiser that round-trips.
//!
//! One widget per line. A verb, then its operands, separated by TAB. Operands
//! are printable ASCII (`0x20..=0x7E`) and there is **no escape sequence**:
//! anything else is a parse error. That is the whole grammar.
//!
//! TAB is the separator because it is the one byte nothing else in this system
//! can produce in an operand. `Panel::key` admits only `0x20..=0x7E`, so no
//! field can contain one; `aiksi::lex` treats it as whitespace, so no program
//! can carry one through; and `shell::split_pipeline` ignores it. `|` was the
//! obvious choice and would have been a trap: `split_pipeline` splits a command
//! line on the *last* `|` or `>`, so an action payload as ordinary as
//! `ls /app | count` would have collided with the field separator and half-run.
//!
//! Refusing beats escaping. It buys an exact `render(parse(t)) == t` identity
//! instead of a normalisation, and it leaves nothing for a decoder to learn
//! beyond "operands are text and TAB ends one".
//!
//! What is deliberately *not* in a document: `Panel::focus`, `Field::cursor`,
//! `List::sel`. Those are where the operator is looking, not what the app is.
//! Putting them in the text would give two identical applications different
//! content addresses and turn a lineage into noise.

use super::ui::{note, Action, Panel, Tone, Widget};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The document this module writes. A version line exists so a later format can
/// be told apart from a corrupt one rather than guessed at.
const VERSION: &str = "1";

/// The verbs a document may use, in the order a decoder should see them.
///
/// Public because the authoring loop builds a grammar from exactly this list:
/// a generated line whose verb came from here cannot be a verb `parse` will
/// reject, which is the difference between making a mistake unreachable and
/// catching it afterwards.
pub const VERBS: &[&str] =
    &["label", "sep", "heading", "status", "note", "item", "field", "button"];

/// Longest an operand may be.
///
/// `Panel::preferred` sizes a window from its text and `frame_for` clamps that
/// to the screen, after which the excess is clipped with nothing said. A model
/// writing a 300-character status line would produce an app that is silently
/// truncated, so "too wide" is a parse error it can be told about instead.
const MAX_OPERAND: usize = 64;

/// Most rows a list may carry, and most widgets a panel may have.
///
/// `Panel::rects` stops laying out when it runs past the client area and there
/// is no scrolling anywhere in `Panel`. Beyond this the widgets exist, are not
/// drawn, and are not clickable, which is indistinguishable from a bug.
const MAX_ITEMS: usize = 24;
const MAX_WIDGETS: usize = 48;

pub struct ParseError {
    pub line: usize,
    pub why: String,
}

impl ParseError {
    fn at(line: usize, why: &str) -> Self {
        ParseError { line, why: why.to_string() }
    }
}

/// Where a document came from, which decides what its actions may do.
///
/// This is the difference between a sandbox and a decoration. An `Action` is a
/// shell command line: it reaches `PENDING` and then `shell::execute`, which
/// knows nothing about capabilities. So `Action::Run("reboot")` inside a stored
/// document would walk straight past any gate placed in the interpreter. A
/// document the machine wrote is therefore restricted at the point it is read,
/// where the restriction cannot be bypassed by anything downstream.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// Compiled into the kernel, written by a person. Unrestricted.
    Builtin,
    /// Loaded from the namespace. Restricted.
    Stored,
}

/// The one command prefix a stored document may invoke.
///
/// Everything an application does goes through its own dispatcher, so the set
/// of things a generated panel can run is the set of functions in its own
/// program -- not the shell's vocabulary.
const STORED_PREFIX: &str = "app ";

/// Whether a generated panel's text can actually be put on screen.
///
/// The rule was printable ASCII, which was the same question while the font
/// was ASCII. It is not the same question now, and the parser refusing a
/// label the renderer can draw perfectly well is a limit nobody could see the
/// reason for. So it asks the font: anything with a glyph is allowed, control
/// codes are not, and a panel that would print rows of hollow boxes is still
/// refused at parse time rather than looked at afterwards.
fn printable(s: &str) -> bool {
    s.chars().all(|c| c >= ' ' && super::font::index_of(c) != super::font::UNKNOWN)
}

fn check_operand(line: usize, s: &str) -> Result<(), ParseError> {
    if !printable(s) {
        return Err(ParseError::at(line, "operand has a byte outside printable ASCII"));
    }
    if s.chars().count() > MAX_OPERAND {
        return Err(ParseError::at(line, "operand is too long to be drawn"));
    }
    Ok(())
}

// --- actions -------------------------------------------------------------

fn render_action(a: &Action) -> String {
    match a {
        Action::Run(c) => alloc::format!("run {}", c),
        Action::Apply(c) => alloc::format!("apply {}", c),
        Action::Browse(r) => alloc::format!("browse {}", r),
        Action::Close => "close".to_string(),
        Action::None => "none".to_string(),
    }
}

fn parse_action(line: usize, s: &str, origin: Origin) -> Result<Action, ParseError> {
    let (verb, rest) = match s.split_once(' ') {
        Some((v, r)) => (v, r),
        None => (s, ""),
    };
    let a = match verb {
        "run" => Action::Run(rest.to_string()),
        "apply" => Action::Apply(rest.to_string()),
        "browse" => Action::Browse(rest.to_string()),
        "close" if rest.is_empty() => Action::Close,
        "none" if rest.is_empty() => Action::None,
        _ => return Err(ParseError::at(line, "unknown action")),
    };
    if origin == Origin::Stored && !stored_admits(&a) {
        return Err(ParseError::at(
            line,
            "a stored panel may only run its own program, browse an app, or close",
        ));
    }
    Ok(a)
}

fn stored_admits(a: &Action) -> bool {
    match a {
        Action::Run(c) | Action::Apply(c) => c.starts_with(STORED_PREFIX),
        Action::Browse(r) => r.starts_with("app:"),
        Action::Close | Action::None => true,
    }
}

fn render_tone(t: Tone) -> &'static str {
    match t {
        Tone::Ok => "ok",
        Tone::Warn => "warn",
        Tone::Bad => "bad",
        Tone::Plain => "plain",
    }
}

fn parse_tone(line: usize, s: &str) -> Result<Tone, ParseError> {
    match s {
        "ok" => Ok(Tone::Ok),
        "warn" => Ok(Tone::Warn),
        "bad" => Ok(Tone::Bad),
        "plain" => Ok(Tone::Plain),
        _ => Err(ParseError::at(line, "unknown tone")),
    }
}

// --- rendering -----------------------------------------------------------

/// A panel as a document.
///
/// The inverse of `parse` for anything `parse` produced. It is not the inverse
/// for a hand-built `Widget::Note`, because `ui::note` wraps prose at 44
/// columns and this joins those lines back with single spaces: a note built
/// with double spaces or its own line breaks comes back normalised. The
/// property that holds, and the one the selftest asserts, is
/// `text -> Panel -> text`.
pub fn render(p: &Panel) -> String {
    let mut out = String::new();
    out.push_str("panel\t");
    out.push_str(VERSION);
    out.push('\n');
    out.push_str("title\t");
    out.push_str(&p.title);
    out.push('\n');
    for w in &p.widgets {
        match w {
            Widget::Label(t) => {
                out.push_str("label\t");
                out.push_str(t);
                out.push('\n');
            }
            Widget::Sep => out.push_str("sep\n"),
            Widget::Heading(t) => {
                out.push_str("heading\t");
                out.push_str(t);
                out.push('\n');
            }
            Widget::Status { name, value, tone } => {
                out.push_str("status\t");
                out.push_str(render_tone(*tone));
                out.push('\t');
                out.push_str(name);
                out.push('\t');
                out.push_str(value);
                out.push('\n');
            }
            Widget::Note(lines) => {
                out.push_str("note\t");
                out.push_str(&lines.join(" "));
                out.push('\n');
            }
            Widget::List { items, .. } => {
                // One line per row. Consecutive `item` lines fold back into one
                // list on the way in, which is this format's only grouping rule.
                for (label, action) in items {
                    out.push_str("item\t");
                    out.push_str(&render_action(action));
                    out.push('\t');
                    out.push_str(label);
                    out.push('\n');
                }
            }
            Widget::Field { name, text, submit, .. } => {
                out.push_str("field\t");
                out.push_str(name);
                out.push('\t');
                out.push_str(&render_action(submit));
                out.push('\t');
                out.push_str(text);
                out.push('\n');
            }
            Widget::Button { label, action } => {
                out.push_str("button\t");
                out.push_str(&render_action(action));
                out.push('\t');
                out.push_str(label);
                out.push('\n');
            }
        }
    }
    out
}

// --- parsing -------------------------------------------------------------

/// Read a document, or say which line is wrong and why.
///
/// Refuses rather than tolerates. Every rejection here is a failure that would
/// otherwise be invisible at runtime: a second field that silently steals
/// `Apply`, rows past what the layout can draw, an operand too wide to fit. A
/// model can be told about a parse error. It cannot be told about a widget that
/// was laid out and then clipped.
pub fn parse(text: &str, origin: Origin) -> Result<Panel, ParseError> {
    let mut title: Option<String> = None;
    let mut widgets: Vec<Widget> = Vec::new();
    let mut fields = 0usize;
    let mut saw_header = false;

    for (n, raw) in text.lines().enumerate() {
        let line = n + 1;
        if raw.is_empty() || raw.starts_with('#') {
            continue;
        }
        let mut parts = raw.split('\t');
        let verb = parts.next().unwrap_or("");
        let ops: Vec<&str> = parts.collect();
        for o in &ops {
            check_operand(line, o)?;
        }

        // Arity is checked once, here, so every arm below can index without
        // apologising for it.
        let want = match verb {
            "panel" | "title" | "label" | "heading" | "note" => 1,
            "sep" => 0,
            "status" => 3,
            "item" | "button" => 2,
            "field" => 3,
            _ => return Err(ParseError::at(line, "unknown verb")),
        };
        if ops.len() != want {
            return Err(ParseError::at(line, "wrong number of operands for this verb"));
        }

        match verb {
            "panel" => {
                if ops[0] != VERSION {
                    return Err(ParseError::at(line, "unknown document version"));
                }
                saw_header = true;
                continue;
            }
            "title" => {
                if title.is_some() {
                    return Err(ParseError::at(line, "a panel has one title"));
                }
                title = Some(ops[0].to_string());
                continue;
            }
            _ => {}
        }

        if widgets.len() >= MAX_WIDGETS {
            return Err(ParseError::at(line, "too many widgets to lay out"));
        }

        match verb {
            "label" => widgets.push(Widget::Label(ops[0].to_string())),
            "sep" => widgets.push(Widget::Sep),
            "heading" => widgets.push(Widget::Heading(ops[0].to_string())),
            "note" => widgets.push(note(ops[0])),
            "status" => widgets.push(Widget::Status {
                tone: parse_tone(line, ops[0])?,
                name: ops[1].to_string(),
                value: ops[2].to_string(),
            }),
            "item" => {
                let action = parse_action(line, ops[0], origin)?;
                let row = (ops[1].to_string(), action);
                // Fold into the list immediately before, if there is one.
                match widgets.last_mut() {
                    Some(Widget::List { items, .. }) => {
                        if items.len() >= MAX_ITEMS {
                            return Err(ParseError::at(line, "too many rows to draw"));
                        }
                        items.push(row);
                    }
                    _ => widgets.push(Widget::List { items: alloc::vec![row], sel: 0 }),
                }
            }
            "field" => {
                fields += 1;
                if fields > 1 {
                    // `Panel::substituted` resolves `Apply` against
                    // `field_text()`, which finds the *first* field and says
                    // nothing. A second one would be typed into and ignored.
                    return Err(ParseError::at(line, "a panel may have one field"));
                }
                let submit = parse_action(line, ops[1], origin)?;
                widgets.push(Widget::Field {
                    name: ops[0].to_string(),
                    cursor: ops[2].chars().count(),
                    text: ops[2].to_string(),
                    submit,
                });
            }
            "button" => widgets.push(Widget::Button {
                action: parse_action(line, ops[0], origin)?,
                label: ops[1].to_string(),
            }),
            _ => unreachable!("arity table and this match list the same verbs"),
        }
    }

    if !saw_header {
        return Err(ParseError::at(0, "no 'panel' version line"));
    }
    let Some(title) = title else {
        return Err(ParseError::at(0, "no 'title' line"));
    };
    Ok(Panel::new(&title, widgets))
}

// --- verification --------------------------------------------------------

/// One fixture per `Widget` variant, and the round trip over all of them.
///
/// The fixture list is built by a `match` over `Widget` with no wildcard arm,
/// so a ninth variant fails to compile until somebody writes a line for it.
/// That is the only defence that works against this format quietly falling
/// behind the enum it serialises -- which would leave generated panels able to
/// express less than hand-written ones, the exact two-tier outcome `ui.rs`
/// rejected trait objects to avoid. A comment asking nicely would not survive.
fn fixture_for(w: &Widget) -> &'static str {
    match w {
        Widget::Label(_) => "label\tplain words\n",
        Widget::Sep => "sep\n",
        Widget::Heading(_) => "heading\tTasks\n",
        Widget::Status { .. } => "status\tok\tLink\tup\n",
        Widget::Note(_) => "note\tsome prose that the panel will wrap for itself\n",
        Widget::List { .. } => "item\trun app done 0\tBuy milk\nitem\trun app done 1\tCall\n",
        Widget::Field { .. } => "field\tnew\tapply app add\t\n",
        Widget::Button { .. } => "button\trun app add\tAdd\n",
    }
}

pub fn selftest() -> bool {
    // Every variant, once, in one document. Built from the same match that
    // renders them, so the set cannot drift.
    let all = alloc::vec![
        Widget::Label(String::new()),
        Widget::Sep,
        Widget::Heading(String::new()),
        Widget::Status { name: String::new(), value: String::new(), tone: Tone::Ok },
        Widget::Note(Vec::new()),
        Widget::List { items: Vec::new(), sel: 0 },
        Widget::Field {
            name: String::new(),
            text: String::new(),
            cursor: 0,
            submit: Action::None,
        },
        Widget::Button { label: String::new(), action: Action::None },
    ];
    let mut doc = String::from("panel\t1\ntitle\tEverything\n");
    for w in &all {
        doc.push_str(fixture_for(w));
    }

    let p = match parse(&doc, Origin::Builtin) {
        Ok(p) => p,
        Err(_) => return false,
    };
    // One widget per variant. The fold shows up as *lines*, not widgets: the
    // list fixture is two `item` lines and yields one `List` holding two rows,
    // so the document is nine lines and the panel is eight widgets.
    if p.widgets.len() != all.len() {
        return false;
    }
    match p.widgets.iter().find(|w| matches!(w, Widget::List { .. })) {
        Some(Widget::List { items, sel }) if items.len() == 2 && *sel == 0 => {}
        _ => return false,
    }
    if p.title != "Everything" {
        return false;
    }
    if render(&p) != doc {
        return false;
    }

    // Every `Action` variant survives the trip.
    let acts = "panel\t1\ntitle\tA\n\
                button\trun mem\tRun\n\
                button\tbrowse app:todo\tBrowse\n\
                button\tclose\tClose\n\
                button\tnone\tNone\n\
                field\tf\tapply dns\t\n";
    match parse(acts, Origin::Builtin) {
        Ok(p) => {
            if render(&p) != acts {
                return false;
            }
        }
        Err(_) => return false,
    }

    // Runtime state is derived, never read from the document. A field's cursor
    // sits after its text so typing appends rather than overwriting.
    match parse("panel\t1\ntitle\tA\nfield\tn\tnone\tabc\n", Origin::Builtin) {
        Ok(p) => match p.widgets.first() {
            Some(Widget::Field { cursor, .. }) if *cursor == 3 => {}
            _ => return false,
        },
        Err(_) => return false,
    }

    let bad = |t: &str| parse(t, Origin::Builtin).is_err();
    // A TAB inside an operand is impossible by construction -- it would be a
    // separator -- so the real hazard is the bytes that are not printable.
    if !bad("panel\t1\ntitle\tA\nlabel\tbell\x07here\n") {
        return false;
    }
    if !bad("panel\t1\ntitle\tA\nnope\tx\n") {
        return false;
    }
    if !bad("panel\t1\ntitle\tA\nheading\n") {
        return false;
    }
    if !bad("panel\t1\ntitle\tA\nstatus\tglowing\tx\ty\n") {
        return false;
    }
    if !bad("panel\t1\ntitle\tA\nbutton\tsummon x\tGo\n") {
        return false;
    }
    // Two fields: the second would be typed into and silently ignored.
    if !bad("panel\t1\ntitle\tA\nfield\ta\tnone\t\nfield\tb\tnone\t\n") {
        return false;
    }
    // No header, and no title.
    if !bad("title\tA\nsep\n") || !bad("panel\t1\nsep\n") {
        return false;
    }
    if !bad("panel\t9\ntitle\tA\n") {
        return false;
    }
    // Wider than the layout can draw.
    let wide = alloc::format!("panel\t1\ntitle\tA\nlabel\t{}\n", "x".repeat(MAX_OPERAND + 1));
    if !bad(&wide) {
        return false;
    }
    // More rows than the layout can show.
    let mut many = String::from("panel\t1\ntitle\tA\n");
    for i in 0..MAX_ITEMS + 1 {
        many.push_str(&alloc::format!("item\tnone\trow {}\n", i));
    }
    if !bad(&many) {
        return false;
    }
    // The error names the line it failed on, or it is not much use to whatever
    // wrote the document.
    match parse("panel\t1\ntitle\tA\nsep\nnope\tx\n", Origin::Builtin) {
        Err(e) if e.line == 4 => {}
        _ => return false,
    }

    // The gate. A stored panel cannot reach the shell's vocabulary, and this is
    // the check that makes the sandbox real rather than decorative: an action is
    // a command line, and it reaches `shell::execute`, which has no idea where
    // it came from.
    let stored = |t: &str| parse(t, Origin::Stored);
    if stored("panel\t1\ntitle\tA\nbutton\trun reboot\tGo\n").is_ok() {
        return false;
    }
    if stored("panel\t1\ntitle\tA\nbutton\tapply write /ai/godel/HEAD x\tGo\n").is_ok() {
        return false;
    }
    if stored("panel\t1\ntitle\tA\nbutton\tbrowse files:/\tGo\n").is_ok() {
        return false;
    }
    if stored("panel\t1\ntitle\tA\nbutton\tapp add\tAdd\n").is_ok() {
        // `run`/`apply` carry the verb; a bare payload is not an action.
        return false;
    }
    if stored("panel\t1\ntitle\tA\nbutton\trun app add\tAdd\n").is_err() {
        return false;
    }
    if stored("panel\t1\ntitle\tA\nbutton\tbrowse app:todo\tGo\n").is_err() {
        return false;
    }
    if stored("panel\t1\ntitle\tA\nbutton\tclose\tX\n").is_err() {
        return false;
    }
    // ...and the same document is fine when a person wrote it.
    parse("panel\t1\ntitle\tA\nbutton\trun reboot\tGo\n", Origin::Builtin).is_ok()
}

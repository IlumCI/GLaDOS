//! Applications with holes in them.
//!
//! A skeleton is a working application with a few values left blank. Filling
//! one produces something that passes every check in `check` before anything
//! has been generated -- the structure was written by a person and only the
//! naming is chosen.
//!
//! ### Why this exists rather than free generation
//!
//! The model on this machine is 135M under emulation and 0.6B to 2B on the
//! hardware. It is not going to write `todo`'s `drop()`: twenty lines of
//! careful string walking with index arithmetic, where one wrong comparison
//! produces a list that silently loses an entry. Asking it to is how a
//! generator produces something that parses and does not work.
//!
//! So the model chooses and almost never composes. This is the same move
//! constrained decoding already made once in this tree: form became free and
//! only choice remained. Skeletons make *structure* free too, and leave naming.
//!
//! The trade is real and worth stating: an application nobody wrote a skeleton
//! for cannot be generated. That failure is legible -- the requirement is
//! reported unmet -- which is better than a plausible program that does the
//! wrong thing.
//!
//! ### Holes are typed, and one is not the model's to fill
//!
//! `{TITLE}` is what the window says, `{NAME}` is the application's identifier
//! and appears inside every action, and `{LABEL}` names a button. `{PATH}` does
//! not appear at all: a program asks `here()` for its own subtree, because the
//! same files live under `/draft` while being written and under `/app` once
//! adopted, and a literal path would be outside the jail on one side of that
//! move.

use alloc::string::String;
use alloc::vec::Vec;

pub struct Skeleton {
    pub kind: &'static str,
    pub what: &'static str,
    pub panel: &'static str,
    pub code: &'static str,
}

/// Five shapes, which is most of what anybody asks a small machine for.
///
/// Deliberately few. Fifteen would each be exercised a fifth as often, and a
/// skeleton nobody exercises is a program nobody has run.
pub const SKELETONS: &[Skeleton] = &[
    Skeleton {
        kind: "list",
        what: "things you add to and tick off",
        panel: "panel\t1\n\
                title\t{TITLE}\n\
                field\tnew\tapply app {NAME} add\t\n\
                sep\n\
                heading\t{LABEL}\n\
                rows\trows\n\
                sep\n\
                button\trun app {NAME} clear\tClear all\n\
                button\tclose\tClose\n",
        code: "// {TITLE}: a list you add to and tick off\n\
               fn file() { return here() + \"/items\" }\n\
               fn all() {\n\
                 if (exists(file())) { return read(file()) }\n\
                 return \"\"\n\
               }\n\
               fn add(what) {\n\
                 if (len(what) > 0) { write(file(), all() + what + \"\\n\") }\n\
                 return \"\"\n\
               }\n\
               fn clear() {\n\
                 write(file(), \"\")\n\
                 return \"\"\n\
               }\n\
               fn drop(what) {\n\
                 text = all()\n\
                 out = \"\"\n\
                 line = \"\"\n\
                 gone = 0\n\
                 i = 0\n\
                 while (i < len(text)) {\n\
                   c = get(text, i)\n\
                   if (c == \"\\n\") {\n\
                     if (line == what) {\n\
                       if (gone == 1) { out = out + line + \"\\n\" }\n\
                       gone = 1\n\
                     } else { out = out + line + \"\\n\" }\n\
                     line = \"\"\n\
                   } else { line = line + c }\n\
                   i = i + 1\n\
                 }\n\
                 write(file(), out)\n\
                 return \"\"\n\
               }\n\
               fn rows() {\n\
                 text = all()\n\
                 out = \"\"\n\
                 line = \"\"\n\
                 i = 0\n\
                 while (i < len(text)) {\n\
                   c = get(text, i)\n\
                   if (c == \"\\n\") {\n\
                     out = out + \"item\\trun app {NAME} drop \" + line + \"\\t\" + line + \"\\n\"\n\
                     line = \"\"\n\
                   } else { line = line + c }\n\
                   i = i + 1\n\
                 }\n\
                 return out\n\
               }\n",
    },
    Skeleton {
        kind: "counter",
        what: "a number you move up and down",
        panel: "panel\t1\n\
                title\t{TITLE}\n\
                heading\t{LABEL}\n\
                rows\trows\n\
                sep\n\
                button\trun app {NAME} up\tMore\n\
                button\trun app {NAME} down\tLess\n\
                button\trun app {NAME} reset\tReset\n\
                button\tclose\tClose\n",
        code: "// {TITLE}: a number you move up and down\n\
               fn file() { return here() + \"/count\" }\n\
               fn value() {\n\
                 if (exists(file())) { return int(read(file())) }\n\
                 return 0\n\
               }\n\
               fn set_to(n) {\n\
                 write(file(), n)\n\
                 return \"\"\n\
               }\n\
               fn up() { return set_to(value() + 1) }\n\
               fn down() { return set_to(value() - 1) }\n\
               fn reset() { return set_to(0) }\n\
               fn rows() { return \"status\\tplain\\tcount\\t\" + value() + \"\\n\" }\n",
    },
    Skeleton {
        kind: "notes",
        what: "lines of text you keep",
        panel: "panel\t1\n\
                title\t{TITLE}\n\
                field\tnote\tapply app {NAME} write_note\t\n\
                sep\n\
                heading\t{LABEL}\n\
                rows\trows\n\
                sep\n\
                button\trun app {NAME} clear\tErase\n\
                button\tclose\tClose\n",
        code: "// {TITLE}: lines of text you keep\n\
               fn file() { return here() + \"/notes\" }\n\
               fn all() {\n\
                 if (exists(file())) { return read(file()) }\n\
                 return \"\"\n\
               }\n\
               fn write_note(t) {\n\
                 if (len(t) > 0) { write(file(), all() + t + \"\\n\") }\n\
                 return \"\"\n\
               }\n\
               fn clear() {\n\
                 write(file(), \"\")\n\
                 return \"\"\n\
               }\n\
               fn rows() {\n\
                 text = all()\n\
                 out = \"\"\n\
                 line = \"\"\n\
                 i = 0\n\
                 while (i < len(text)) {\n\
                   c = get(text, i)\n\
                   if (c == \"\\n\") {\n\
                     out = out + \"label\\t\" + line + \"\\n\"\n\
                     line = \"\"\n\
                   } else { line = line + c }\n\
                   i = i + 1\n\
                 }\n\
                 return out\n\
               }\n",
    },
    Skeleton {
        kind: "menu",
        what: "buttons that each report something",
        panel: "panel\t1\n\
                title\t{TITLE}\n\
                heading\t{LABEL}\n\
                rows\trows\n\
                sep\n\
                button\trun app {NAME} look\tLook\n\
                button\trun app {NAME} forget\tForget\n\
                button\tclose\tClose\n",
        code: "// {TITLE}: buttons that each report something\n\
               fn file() { return here() + \"/last\" }\n\
               fn look() {\n\
                 write(file(), applet(\"ls /\"))\n\
                 return \"\"\n\
               }\n\
               fn forget() {\n\
                 write(file(), \"\")\n\
                 return \"\"\n\
               }\n\
               fn rows() {\n\
                 if (exists(file())) { return \"note\\t\" + read(file()) + \"\\n\" }\n\
                 return \"note\\tnothing looked at yet\\n\"\n\
               }\n",
    },
    Skeleton {
        kind: "status",
        what: "facts about the machine, refreshed when you press",
        panel: "panel\t1\n\
                title\t{TITLE}\n\
                heading\t{LABEL}\n\
                rows\trows\n\
                sep\n\
                button\trun app {NAME} refresh\tRefresh\n\
                button\tclose\tClose\n",
        code: "// {TITLE}: facts about the machine\n\
               fn file() { return here() + \"/seen\" }\n\
               fn refresh() {\n\
                 write(file(), ticks())\n\
                 return \"\"\n\
               }\n\
               fn rows() {\n\
                 out = \"status\\tplain\\tuptime\\t\" + ticks() + \"\\n\"\n\
                 out = out + \"status\\tplain\\ttasks\\t\" + tasks() + \"\\n\"\n\
                 if (exists(file())) {\n\
                   out = out + \"status\\tok\\tlast look\\t\" + read(file()) + \"\\n\"\n\
                 }\n\
                 return out\n\
               }\n",
    },
];

pub fn find(kind: &str) -> Option<&'static Skeleton> {
    SKELETONS.iter().find(|s| s.kind == kind)
}

pub fn kinds() -> Vec<&'static str> {
    SKELETONS.iter().map(|s| s.kind).collect()
}

/// Fill the holes. Returns `(panel.ui, code.l)`.
///
/// `name` also becomes part of every action, so it must be the identifier the
/// application is stored under -- a mismatch produces a panel whose buttons
/// address an application that is not there, which `check_refs` would not catch
/// because it looks at the function and not the name in front of it.
pub fn fill(kind: &str, name: &str, title: &str, label: &str) -> Option<(String, String)> {
    let s = find(kind)?;
    let sub = |t: &str| {
        t.replace("{NAME}", name)
            .replace("{TITLE}", title)
            .replace("{LABEL}", label)
    };
    Some((sub(s.panel), sub(s.code)))
}

/// Every skeleton, filled with a canonical value, put through the whole of
/// `check`.
///
/// This is the defence against the library drifting away from what the codec
/// and the sandbox will accept -- the same defence `uidoc`'s wildcard-free
/// fixture match gives, arrived at differently. A skeleton that cannot pass the
/// gate a generated application must pass is not a skeleton; it is a bug
/// waiting for whichever Tuesday somebody picks it.
pub fn selftest() -> bool {
    use super::check;
    for s in SKELETONS {
        let Some((panel, code)) = fill(s.kind, "demo", "Demo", "Things") else {
            return false;
        };
        // No hole may survive filling.
        //
        // Named, not any brace: a program is full of them, and checking for
        // `{` rejected every skeleton that contained a function body.
        for hole in ["{TITLE}", "{NAME}", "{LABEL}"] {
            if panel.contains(hole) || code.contains(hole) {
                return false;
            }
        }
        // The program parses and loads inside a sandbox.
        if !check::check_code(&code, "/draft/demo").ok {
            return false;
        }
        // Every function the panel names exists, with the right arity.
        if !check::check_refs("demo", &panel, &code).iter().all(|v| v.ok) {
            return false;
        }
        // The panel parses once its `rows` line is spliced out. The row
        // function is not run here -- that needs a namespace, and these run
        // before `sysbox::init` -- so the check is over the static remainder,
        // which is what a row function can never fix.
        let mut stripped = String::new();
        for line in panel.lines() {
            if !line.starts_with("rows\t") {
                stripped.push_str(line);
                stripped.push('\n');
            }
        }
        if !check::check_panel(&stripped).ok {
            return false;
        }
        // Every skeleton has a row function, because a panel that calls
        // nothing is the empty application `check` warns about.
        if !panel.contains("rows\t") {
            return false;
        }
    }
    // The kinds are distinct, or `find` silently returns the first of a pair.
    for (i, a) in SKELETONS.iter().enumerate() {
        if SKELETONS.iter().skip(i + 1).any(|b| b.kind == a.kind) {
            return false;
        }
    }
    find("nosuchkind").is_none()
}

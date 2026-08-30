//! What is in a file, in one screen.
//!
//! This exists for the model more than for the operator. A language model
//! reading a forty kilobyte source file spends its whole context on it and
//! answers worse than one that was told the file defines nine functions and
//! their names. An outline is the cheap summary that makes the expensive read
//! avoidable, and where the read is still needed it says which part.
//!
//! Every outline is derived by scanning, never by parsing. That is a real
//! limitation and it is the right trade here: a full parser for seven
//! languages is seven parsers to keep correct, and an outline that is
//! occasionally missing an entry is useful while a parser that is occasionally
//! wrong about a program is not. Entries are what a line begins with, which is
//! how these languages are written and is why it works as well as it does.

use super::{Kind, Lang};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Nesting, where zero is the top level.
    pub depth: u32,
    pub kind: &'static str,
    pub name: String,
    /// One-based, so it can be typed into an editor.
    pub line: usize,
}

/// Summarise a file. An empty result means nothing was recognised, which is a
/// true answer for prose.
pub fn of(kind: Kind, src: &str) -> Vec<Entry> {
    match kind {
        Kind::Source(l) => source(l, src),
        Kind::Markdown => markdown(src),
        Kind::Json => json(src),
        Kind::JsonLines => {
            let (v, bad) = super::table::json_lines(src);
            let mut out = vec![Entry {
                depth: 0,
                kind: "records",
                name: alloc::format!("{}", v.len()),
                line: 1,
            }];
            for b in bad.iter().take(8) {
                out.push(Entry { depth: 1, kind: "unparsed", name: String::new(), line: *b });
            }
            out
        }
        Kind::Xml => match super::xml::parse(src) {
            Ok(root) => {
                let mut out = Vec::new();
                xml_walk(&root, 0, &mut out);
                out
            }
            Err(e) => vec![Entry { depth: 0, kind: "error", name: e, line: 1 }],
        },
        Kind::Ini => super::table::ini(src)
            .iter()
            .map(|s| Entry {
                depth: 0,
                kind: "section",
                name: if s.name.is_empty() { String::from("(top)") } else { s.name.clone() },
                line: 1,
            })
            .collect(),
        Kind::Csv | Kind::Tsv => {
            let d = if kind == Kind::Csv { ',' } else { '\t' };
            let rows = super::table::delimited(src, d);
            match rows.first() {
                Some(h) => h
                    .iter()
                    .enumerate()
                    .map(|(i, c)| Entry {
                        depth: 0,
                        kind: "column",
                        name: alloc::format!("{} ({})", c, i),
                        line: 1,
                    })
                    .collect(),
                None => Vec::new(),
            }
        }
        _ => Vec::new(),
    }
}

fn xml_walk(n: &super::xml::Node, depth: u32, out: &mut Vec<Entry>) {
    if depth > 3 || out.len() > 200 {
        return;
    }
    out.push(Entry { depth, kind: "element", name: n.name.clone(), line: 1 });
    for k in &n.kids {
        if let super::xml::Item::Elem(e) = k {
            xml_walk(e, depth + 1, out);
        }
    }
}

fn json(src: &str) -> Vec<Entry> {
    let Some(v) = crate::json::Json::parse(src) else {
        return vec![Entry { depth: 0, kind: "error", name: String::from("will not parse"), line: 1 }];
    };
    let mut out = Vec::new();
    json_walk(&v, 0, &mut out);
    out
}

fn json_walk(v: &crate::json::Json, depth: u32, out: &mut Vec<Entry>) {
    use crate::json::Json;
    if depth > 2 || out.len() > 200 {
        return;
    }
    match v {
        Json::Obj(pairs) => {
            for (k, val) in pairs {
                out.push(Entry { depth, kind: type_of(val), name: k.clone(), line: 1 });
                json_walk(val, depth + 1, out);
            }
        }
        Json::Arr(items) => {
            out.push(Entry {
                depth,
                kind: "array",
                name: alloc::format!("{} item(s)", items.len()),
                line: 1,
            });
            if let Some(first) = items.first() {
                json_walk(first, depth + 1, out);
            }
        }
        _ => {}
    }
}

fn type_of(v: &crate::json::Json) -> &'static str {
    use crate::json::Json;
    match v {
        Json::Null => "null",
        Json::Bool(_) => "bool",
        Json::Num(_) => "number",
        Json::Str(_) => "string",
        Json::Arr(_) => "array",
        Json::Obj(_) => "object",
    }
}

fn markdown(src: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    let mut fenced = false;
    for (n, line) in src.lines().enumerate() {
        let t = line.trim_start();
        // A hash inside a fenced block is code and not a heading, which is the
        // one thing a naive heading scanner always gets wrong on a document
        // about shell scripts.
        if t.starts_with("```") || t.starts_with("~~~") {
            fenced = !fenced;
            continue;
        }
        if fenced || !t.starts_with('#') {
            continue;
        }
        let hashes = t.chars().take_while(|c| *c == '#').count();
        if hashes > 6 {
            continue;
        }
        let name = t[hashes..].trim();
        if name.is_empty() {
            continue;
        }
        out.push(Entry {
            depth: hashes as u32 - 1,
            kind: "heading",
            name: String::from(name),
            line: n + 1,
        });
    }
    out
}

/// Definitions in a source file, found by what a line starts with.
fn source(lang: Lang, src: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    let mut carry = super::Carry::None;
    for (n, line) in src.lines().enumerate() {
        let before = carry;
        super::scan(lang, line, &mut carry);
        // A line that began inside a comment or a string cannot define
        // anything, whatever it looks like.
        if before != super::Carry::None {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        let t = line.trim_start();
        let starters: &[(&str, &str)] = match lang {
            Lang::Rust => &[
                ("fn ", "fn"), ("pub fn ", "fn"), ("struct ", "struct"), ("pub struct ", "struct"),
                ("enum ", "enum"), ("pub enum ", "enum"), ("trait ", "trait"),
                ("pub trait ", "trait"), ("impl ", "impl"), ("mod ", "mod"), ("pub mod ", "mod"),
                ("const ", "const"), ("pub const ", "const"), ("static ", "static"),
                ("pub static ", "static"), ("type ", "type"), ("macro_rules! ", "macro"),
            ],
            Lang::Python => &[("def ", "def"), ("class ", "class"), ("async def ", "def")],
            Lang::JavaScript => &[
                ("function ", "function"), ("class ", "class"), ("const ", "const"),
                ("let ", "let"), ("export function ", "function"), ("export class ", "class"),
                ("export default ", "export"), ("async function ", "function"),
            ],
            Lang::Aiksi => &[("fn ", "fn"), ("rec ", "rec"), ("use ", "use")],
            Lang::Shell => &[("function ", "function")],
            Lang::C | Lang::Cpp | Lang::CSharp => &[
                ("struct ", "struct"), ("class ", "class"), ("enum ", "enum"),
                ("union ", "union"), ("typedef ", "typedef"), ("namespace ", "namespace"),
                ("#define ", "define"), ("public class ", "class"), ("public struct ", "struct"),
                ("interface ", "interface"), ("public interface ", "interface"),
            ],
        };
        let mut hit = None;
        for (pre, kind) in starters {
            if t.starts_with(pre) {
                // Prefer the longest prefix, so "pub fn" beats "fn" and the
                // name is taken from after the right one.
                if hit.map(|(p, _): (&str, &str)| pre.len() > p.len()).unwrap_or(true) {
                    hit = Some((*pre, *kind));
                }
            }
        }
        // C-family functions have no keyword, so they are recognised by shape:
        // a line at the top level ending in an open brace with parentheses in
        // it. That misses a definition split across lines, which is stated
        // rather than pretended away.
        if hit.is_none() && matches!(lang, Lang::C | Lang::Cpp | Lang::CSharp) && indent == 0 {
            if t.ends_with('{') && t.contains('(') && !t.starts_with("if") && !t.starts_with("for")
                && !t.starts_with("while") && !t.starts_with("switch")
            {
                if let Some(name) = c_function_name(t) {
                    out.push(Entry { depth: 0, kind: "function", name, line: n + 1 });
                }
                continue;
            }
        }
        let Some((pre, kind)) = hit else { continue };
        let rest = &t[pre.len()..];
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '<' || *c == ':')
            .collect();
        let name = name.trim_end_matches(':').trim_end_matches('<');
        if name.is_empty() {
            continue;
        }
        out.push(Entry {
            depth: (indent / 4) as u32,
            kind,
            name: String::from(name),
            line: n + 1,
        });
    }
    out
}

/// The identifier immediately before the parameter list.
fn c_function_name(t: &str) -> Option<String> {
    let open = t.find('(')?;
    let head = &t[..open];
    let name: String = head
        .chars()
        .rev()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

pub fn selftest() -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    let o = of(Kind::Source(Lang::Rust), "pub fn a() {}\nstruct B;\n// fn c\nfn d() {}\n");
    claim(
        &mut ok,
        o.len() == 3 && o[0].name == "a" && o[1].name == "B" && o[2].name == "d",
        "rust definitions found, and a commented one is not",
    );
    claim(&mut ok, o[2].line == 4, "with the line to jump to");

    let o = of(Kind::Source(Lang::Rust), "/* fn hidden() {}\nfn also() {} */\nfn real() {}\n");
    claim(
        &mut ok,
        o.len() == 1 && o[0].name == "real",
        "nothing inside a block comment is a definition",
    );

    let o = of(Kind::Source(Lang::Python), "def a():\n    def b():\n        pass\nclass C:\n");
    claim(&mut ok, o.len() == 3 && o[1].depth == 1, "python nesting comes from indentation");

    let o = of(Kind::Source(Lang::C), "int main(int argc) {\nstruct S {\n");
    claim(
        &mut ok,
        o.iter().any(|e| e.name == "main") && o.iter().any(|e| e.name == "S"),
        "a C function is found by shape and a struct by keyword",
    );

    let o = of(Kind::Source(Lang::Aiksi), "fn vote(t: str): int { return 0 }\nrec P { x }\n");
    claim(&mut ok, o.len() == 2 && o[0].name == "vote", "aiksi is a language like the others here");

    let o = of(Kind::Markdown, "# One\n```\n# not a heading\n```\n## Two\n");
    claim(
        &mut ok,
        o.len() == 2 && o[1].name == "Two" && o[1].depth == 1,
        "a hash inside a fence is code",
    );

    let o = of(Kind::Json, "{\"a\":1,\"b\":[1,2]}");
    claim(
        &mut ok,
        o.iter().any(|e| e.name == "a" && e.kind == "number")
            && o.iter().any(|e| e.kind == "array"),
        "json keys carry their types",
    );

    let o = of(Kind::Csv, "name,age\nx,1\n");
    claim(&mut ok, o.len() == 2 && o[0].name.starts_with("name"), "csv columns come from the header");

    claim(&mut ok, of(Kind::Text, "just prose").is_empty(), "prose outlines to nothing, honestly");
    ok
}

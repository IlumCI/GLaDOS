//! Knowing what a file is, and reading it as that.
//!
//! The namespace stores bytes and nothing above it knew what any of them were.
//! An editor showed every file as plain text, a model reading one got a wall of
//! characters, and a program that wanted the third field of the second record
//! had to write its own splitter. This is the layer that answers what a file is
//! and hands back its structure.
//!
//! **Extension first, contents second, and never a guess.** A name that carries
//! a known extension is that kind, because the operator said so by naming it.
//! A name that carries none is sniffed. Anything that survives both without
//! matching is `Text` when it decodes as UTF-8 and `Binary` when it does not,
//! and neither of those is a guess about what the file means. Guessing wrongly
//! here would be worse than not knowing: a program told a file is JSON when it
//! is prose gets a parse failure it cannot explain.
//!
//! **One tokenizer, a table per language.** C, C++, C#, Rust, JavaScript,
//! Python and Aiksi differ in their comment markers, their string delimiters
//! and their keyword lists. They do not differ in lexical structure. So there
//! is one scanner driven by a small `Syntax` row, and adding a language is
//! adding a row rather than writing a parser. That is the whole reason the
//! "and more" in this module's remit is affordable.
//!
//! What this is deliberately not: a compiler front end for seven languages, a
//! validator, or a formatter. It classifies, it tokenizes for display, and it
//! produces an outline. Anything wanting real semantics should use the real
//! parser, which for Aiksi is next door in `crate::aiksi`.

use alloc::string::String;
use alloc::vec::Vec;

pub mod outline;
pub mod table;
pub mod xml;

/// What a file is.
///
/// The list is closed on purpose. An unknown extension answers `Text` or
/// `Binary` rather than growing a variant nobody handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Text,
    Markdown,
    Json,
    /// One JSON value per line. Distinct from `Json` because the whole file is
    /// not a JSON document and parsing it as one fails on the second line.
    JsonLines,
    Xml,
    Html,
    Css,
    Csv,
    Tsv,
    /// `key = value` under `[section]` headers. Covers INI and the subset of
    /// TOML that looks like INI, which is most hand-written TOML.
    Ini,
    Source(Lang),
    /// A netpbm image, which is the one binary format this system writes.
    Ppm,
    Binary,
}

/// A programming language, for the purpose of scanning it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Rust,
    C,
    Cpp,
    CSharp,
    JavaScript,
    Python,
    Aiksi,
    Shell,
}

impl Kind {
    /// The name a person would use.
    pub fn name(&self) -> &'static str {
        match self {
            Kind::Text => "text",
            Kind::Markdown => "markdown",
            Kind::Json => "json",
            Kind::JsonLines => "jsonl",
            Kind::Xml => "xml",
            Kind::Html => "html",
            Kind::Css => "css",
            Kind::Csv => "csv",
            Kind::Tsv => "tsv",
            Kind::Ini => "ini",
            Kind::Ppm => "ppm",
            Kind::Binary => "binary",
            Kind::Source(l) => l.name(),
        }
    }

    /// Whether the bytes are meant to be read as characters.
    pub fn is_text(&self) -> bool {
        !matches!(self, Kind::Binary | Kind::Ppm)
    }
}

impl Lang {
    pub fn name(&self) -> &'static str {
        match self {
            Lang::Rust => "rust",
            Lang::C => "c",
            Lang::Cpp => "c++",
            Lang::CSharp => "c#",
            Lang::JavaScript => "javascript",
            Lang::Python => "python",
            Lang::Aiksi => "aiksi",
            Lang::Shell => "shell",
        }
    }

    /// How to scan it. See `Syntax`.
    pub fn syntax(&self) -> Syntax {
        match self {
            Lang::Rust => Syntax {
                line: "//",
                block: Some(("/*", "*/")),
                nests: true,
                strings: &['"'],
                chars: true,
                keywords: &[
                    "as", "break", "const", "continue", "crate", "else", "enum", "extern", "false",
                    "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut",
                    "pub", "ref", "return", "self", "static", "struct", "super", "trait", "true",
                    "type", "unsafe", "use", "where", "while", "async", "await", "dyn",
                ],
            },
            Lang::C => Syntax {
                line: "//",
                block: Some(("/*", "*/")),
                nests: false,
                strings: &['"'],
                chars: true,
                keywords: &[
                    "auto", "break", "case", "char", "const", "continue", "default", "do",
                    "double", "else", "enum", "extern", "float", "for", "goto", "if", "int",
                    "long", "register", "return", "short", "signed", "sizeof", "static", "struct",
                    "switch", "typedef", "union", "unsigned", "void", "volatile", "while",
                ],
            },
            Lang::Cpp => Syntax {
                line: "//",
                block: Some(("/*", "*/")),
                nests: false,
                strings: &['"'],
                chars: true,
                keywords: &[
                    "auto", "bool", "break", "case", "catch", "class", "const", "constexpr",
                    "continue", "default", "delete", "do", "double", "else", "enum", "explicit",
                    "false", "float", "for", "friend", "if", "inline", "int", "long", "namespace",
                    "new", "nullptr", "operator", "private", "protected", "public", "return",
                    "short", "sizeof", "static", "struct", "switch", "template", "this", "throw",
                    "true", "try", "typedef", "typename", "union", "unsigned", "using", "virtual",
                    "void", "volatile", "while",
                ],
            },
            Lang::CSharp => Syntax {
                line: "//",
                block: Some(("/*", "*/")),
                nests: false,
                strings: &['"'],
                chars: true,
                keywords: &[
                    "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char",
                    "class", "const", "continue", "decimal", "default", "delegate", "do", "double",
                    "else", "enum", "event", "explicit", "false", "finally", "float", "for",
                    "foreach", "if", "implicit", "in", "int", "interface", "internal", "is",
                    "lock", "long", "namespace", "new", "null", "object", "operator", "out",
                    "override", "params", "private", "protected", "public", "readonly", "ref",
                    "return", "sealed", "short", "sizeof", "static", "string", "struct", "switch",
                    "this", "throw", "true", "try", "typeof", "uint", "ulong", "using", "var",
                    "virtual", "void", "while",
                ],
            },
            Lang::JavaScript => Syntax {
                line: "//",
                block: Some(("/*", "*/")),
                nests: false,
                strings: &['"', '\'', '`'],
                chars: false,
                keywords: &[
                    "async", "await", "break", "case", "catch", "class", "const", "continue",
                    "default", "delete", "do", "else", "export", "extends", "false", "finally",
                    "for", "function", "if", "import", "in", "instanceof", "let", "new", "null",
                    "of", "return", "static", "super", "switch", "this", "throw", "true", "try",
                    "typeof", "undefined", "var", "void", "while", "yield",
                ],
            },
            Lang::Python => Syntax {
                line: "#",
                // Triple quotes are handled by the string scanner, which takes
                // the longest run of the delimiter it can see. A docstring is
                // a string here rather than a comment, which is what it is.
                block: None,
                nests: false,
                strings: &['"', '\''],
                chars: false,
                keywords: &[
                    "and", "as", "assert", "async", "await", "break", "class", "continue", "def",
                    "del", "elif", "else", "except", "False", "finally", "for", "from", "global",
                    "if", "import", "in", "is", "lambda", "None", "nonlocal", "not", "or", "pass",
                    "raise", "return", "True", "try", "while", "with", "yield",
                ],
            },
            Lang::Aiksi => Syntax {
                line: "//",
                block: None,
                nests: false,
                strings: &['"'],
                chars: false,
                keywords: &["fn", "if", "else", "while", "return", "rec", "use"],
            },
            Lang::Shell => Syntax {
                line: "#",
                block: None,
                nests: false,
                strings: &['"', '\''],
                chars: false,
                keywords: &[
                    "case", "do", "done", "elif", "else", "esac", "fi", "for", "function", "if",
                    "in", "then", "until", "while",
                ],
            },
        }
    }
}

/// Everything the scanner needs to know about a language.
///
/// Seven languages fit in this shape, which is the argument for having it. A
/// language that does not fit gets a row that is wrong in a way somebody can
/// see, rather than a parser that is wrong in a way nobody can.
pub struct Syntax {
    pub line: &'static str,
    pub block: Option<(&'static str, &'static str)>,
    /// Whether block comments nest. Rust's do and C's do not, and getting it
    /// backwards ends a comment early or swallows the rest of a file.
    pub nests: bool,
    pub strings: &'static [char],
    /// Whether a single quote opens a character literal rather than a string.
    /// This is why Rust needs care: a lifetime is an apostrophe that never
    /// closes, so `'a` must not open a literal that eats the next line.
    pub chars: bool,
    pub keywords: &'static [&'static str],
}

/// What a run of characters is, for display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Plain,
    Comment,
    Str,
    Number,
    Keyword,
    Punct,
}

/// One classified run, as a byte range into the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub class: Class,
    pub start: usize,
    pub end: usize,
}

/// Decide what a file is from its name and its first bytes.
///
/// The name wins where it says anything, because the operator naming a file
/// `notes.txt` has stated an intent that its contents cannot overrule.
pub fn detect(name: &str, bytes: &[u8]) -> Kind {
    if let Some(k) = by_extension(name) {
        return k;
    }
    sniff(bytes)
}

/// Extension to kind, or `None` when the name carries nothing useful.
pub fn by_extension(name: &str) -> Option<Kind> {
    // The system's own extension contains a dot in a place that would confuse
    // a naive rsplit, so it is matched whole and first.
    if name.ends_with(".ai&xi") {
        return Some(Kind::Source(Lang::Aiksi));
    }
    let ext = name.rsplit_once('.')?.1;
    let mut low = String::new();
    for c in ext.chars() {
        low.push(c.to_ascii_lowercase());
    }
    Some(match low.as_str() {
        "txt" | "log" | "text" => Kind::Text,
        "md" | "markdown" => Kind::Markdown,
        "json" => Kind::Json,
        "jsonl" | "ndjson" => Kind::JsonLines,
        "xml" | "svg" | "xhtml" | "rss" | "atom" | "plist" => Kind::Xml,
        "html" | "htm" => Kind::Html,
        "css" => Kind::Css,
        "csv" => Kind::Csv,
        "tsv" => Kind::Tsv,
        "ini" | "cfg" | "conf" | "toml" => Kind::Ini,
        "ppm" => Kind::Ppm,
        "rs" => Kind::Source(Lang::Rust),
        "c" | "h" => Kind::Source(Lang::C),
        "cc" | "cpp" | "cxx" | "hpp" | "hh" => Kind::Source(Lang::Cpp),
        "cs" => Kind::Source(Lang::CSharp),
        "js" | "mjs" | "cjs" | "ts" => Kind::Source(Lang::JavaScript),
        "py" | "pyw" => Kind::Source(Lang::Python),
        "sh" | "bash" => Kind::Source(Lang::Shell),
        "bin" | "efi" | "img" | "iso" | "o" | "a" => Kind::Binary,
        _ => return None,
    })
}

/// Decide from contents alone.
///
/// Only the shapes that announce themselves are recognised. Everything else is
/// text or binary, because a heuristic that decides prose is CSV because it has
/// commas in it is worse than no heuristic.
pub fn sniff(bytes: &[u8]) -> Kind {
    let head = &bytes[..bytes.len().min(1024)];
    if head.contains(&0) || core::str::from_utf8(head).is_err() {
        // A NUL in the first kilobyte is the oldest reliable binary test there
        // is, and invalid UTF-8 is the other half of it. Checking only the head
        // keeps this cheap on a large file, and a file that turns binary after
        // a kilobyte of clean text is not a case worth slowing everything for.
        if head.starts_with(b"P6") || head.starts_with(b"P3") {
            return Kind::Ppm;
        }
        return Kind::Binary;
    }
    let s = match core::str::from_utf8(head) {
        Ok(s) => s.trim_start(),
        Err(_) => return Kind::Binary,
    };
    if s.starts_with("P6") || s.starts_with("P3") {
        return Kind::Ppm;
    }
    if s.starts_with("<?xml") {
        return Kind::Xml;
    }
    let lower_head: String = s.chars().take(64).map(|c| c.to_ascii_lowercase()).collect();
    if lower_head.starts_with("<!doctype html") || lower_head.starts_with("<html") {
        return Kind::Html;
    }
    if s.starts_with('<') {
        return Kind::Xml;
    }
    if s.starts_with('{') || s.starts_with('[') {
        // One object or array is JSON. Several, one per line, is JSON Lines,
        // and telling them apart matters because parsing the second as the
        // first fails on line two with nothing useful to say.
        let lines = s.lines().filter(|l| !l.trim().is_empty()).count();
        if lines > 1 && s.lines().take(2).all(|l| {
            let t = l.trim();
            t.is_empty() || crate::json::Json::parse(t).is_some()
        }) {
            return Kind::JsonLines;
        }
        return Kind::Json;
    }
    if s.starts_with("#!") {
        return Kind::Source(Lang::Shell);
    }
    Kind::Text
}

/// Classify a source line for display.
///
/// Line-oriented because the editor and the console are, and because a scanner
/// that needed the whole file could not colour a window without reading past
/// its bottom edge. The cost is that a block comment or a multi-line string is
/// re-opened per line, so callers that care thread `carry` through.
pub fn scan(lang: Lang, line: &str, carry: &mut Carry) -> Vec<Span> {
    let sy = lang.syntax();
    let b = line.as_bytes();
    let mut out: Vec<Span> = Vec::new();
    let mut i = 0usize;
    let mut plain = 0usize;

    let flush = |out: &mut Vec<Span>, from: usize, to: usize| {
        if to > from {
            out.push(Span { class: Class::Plain, start: from, end: to });
        }
    };

    while i < b.len() {
        // Continuing something the previous line opened.
        match *carry {
            Carry::Block(depth) => {
                let (open, close) = match sy.block {
                    Some(p) => p,
                    None => {
                        *carry = Carry::None;
                        continue;
                    }
                };
                let start = i;
                let mut d = depth;
                while i < b.len() {
                    if b[i..].starts_with(close.as_bytes()) {
                        i += close.len();
                        d -= 1;
                        if d == 0 {
                            break;
                        }
                        continue;
                    }
                    if sy.nests && b[i..].starts_with(open.as_bytes()) {
                        i += open.len();
                        d += 1;
                        continue;
                    }
                    i += 1;
                }
                out.push(Span { class: Class::Comment, start, end: i });
                *carry = if d == 0 { Carry::None } else { Carry::Block(d) };
                plain = i;
                continue;
            }
            Carry::Str(q, triple) => {
                let start = i;
                let end = close_string(b, i, q, triple);
                out.push(Span { class: Class::Str, start, end: end.0 });
                i = end.0;
                *carry = if end.1 { Carry::None } else { Carry::Str(q, triple) };
                plain = i;
                continue;
            }
            Carry::None => {}
        }

        // A line comment runs to the end and cannot be re-opened.
        if !sy.line.is_empty() && b[i..].starts_with(sy.line.as_bytes()) {
            flush(&mut out, plain, i);
            out.push(Span { class: Class::Comment, start: i, end: b.len() });
            return out;
        }
        if let Some((open, close)) = sy.block {
            if b[i..].starts_with(open.as_bytes()) {
                // Consumed here rather than by handing control to the carry
                // branch above, because that branch starts scanning at `i` and
                // would read this same opener as a second one, so `/* x` ended
                // a line at depth two and never closed.
                flush(&mut out, plain, i);
                let start = i;
                i += open.len();
                let mut d = 1u32;
                while i < b.len() {
                    if b[i..].starts_with(close.as_bytes()) {
                        i += close.len();
                        d -= 1;
                        if d == 0 {
                            break;
                        }
                        continue;
                    }
                    if sy.nests && b[i..].starts_with(open.as_bytes()) {
                        i += open.len();
                        d += 1;
                        continue;
                    }
                    i += 1;
                }
                out.push(Span { class: Class::Comment, start, end: i });
                *carry = if d == 0 { Carry::None } else { Carry::Block(d) };
                plain = i;
                continue;
            }
        }
        let c = b[i] as char;
        if sy.strings.contains(&c) {
            // Python's triple quotes are the same delimiter three times, so
            // the run length decides which terminator to look for.
            let triple = b.len() >= i + 3 && b[i + 1] == b[i] && b[i + 2] == b[i];
            flush(&mut out, plain, i);
            let start = i;
            i += if triple { 3 } else { 1 };
            let (end, done) = close_string(b, i, b[start], triple);
            out.push(Span { class: Class::Str, start, end });
            i = end;
            *carry = if done { Carry::None } else { Carry::Str(b[start], triple) };
            plain = i;
            continue;
        }
        if sy.chars && c == '\'' {
            // A Rust lifetime is an apostrophe followed by an identifier and
            // no closing quote. Treating it as a character literal would eat
            // the rest of the line, which is the single most visible way a
            // highlighter can be wrong.
            if is_lifetime(b, i) {
                i += 1;
                continue;
            }
            flush(&mut out, plain, i);
            let start = i;
            i += 1;
            let (end, _) = close_string(b, i, b'\'', false);
            out.push(Span { class: Class::Str, start, end });
            i = end;
            plain = i;
            continue;
        }
        if c.is_ascii_digit() && (i == 0 || !ident_byte(b[i - 1])) {
            flush(&mut out, plain, i);
            let start = i;
            while i < b.len() && (ident_byte(b[i]) || b[i] == b'.') {
                i += 1;
            }
            out.push(Span { class: Class::Number, start, end: i });
            plain = i;
            continue;
        }
        if ident_byte(b[i]) && !b[i].is_ascii_digit() {
            let start = i;
            while i < b.len() && ident_byte(b[i]) {
                i += 1;
            }
            let word = &line[start..i];
            if sy.keywords.contains(&word) {
                flush(&mut out, plain, start);
                out.push(Span { class: Class::Keyword, start, end: i });
                plain = i;
            }
            continue;
        }
        i += 1;
    }
    flush(&mut out, plain, b.len());
    out
}

/// What a line left open for the next one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Carry {
    None,
    /// Inside a block comment, at this nesting depth.
    Block(u32),
    /// Inside a string opened with this delimiter, tripled or not.
    Str(u8, bool),
}

impl Default for Carry {
    fn default() -> Self {
        Carry::None
    }
}

/// Find where a string ends. Answers the index past the terminator and whether
/// the terminator was actually found on this line.
fn close_string(b: &[u8], mut i: usize, q: u8, triple: bool) -> (usize, bool) {
    while i < b.len() {
        if b[i] == b'\\' {
            i += 2;
            continue;
        }
        if b[i] == q {
            if !triple {
                return (i + 1, true);
            }
            if i + 2 < b.len() && b[i + 1] == q && b[i + 2] == q {
                return (i + 3, true);
            }
        }
        i += 1;
    }
    (b.len(), false)
}

fn ident_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Whether the apostrophe at `i` opens a lifetime rather than a literal.
fn is_lifetime(b: &[u8], i: usize) -> bool {
    let mut j = i + 1;
    if j >= b.len() || !(b[j].is_ascii_alphabetic() || b[j] == b'_') {
        return false;
    }
    while j < b.len() && ident_byte(b[j]) {
        j += 1;
    }
    // A character literal closes; a lifetime does not.
    j >= b.len() || b[j] != b'\''
}

pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    // Detection: the name decides where it says anything.
    claim(&mut ok, detect("a.rs", b"") == Kind::Source(Lang::Rust), "an extension names the kind");
    claim(
        &mut ok,
        detect("core.ai&xi", b"") == Kind::Source(Lang::Aiksi),
        "and an extension with a dot inside it is matched whole",
    );
    claim(
        &mut ok,
        detect("notes.txt", b"{\"a\":1}") == Kind::Text,
        "contents do not overrule a name the operator chose",
    );
    claim(
        &mut ok,
        detect("unnamed", b"{\"a\":1}") == Kind::Json,
        "a nameless file is sniffed",
    );
    claim(
        &mut ok,
        detect("x", b"{\"a\":1}\n{\"a\":2}\n") == Kind::JsonLines,
        "one value per line is not one document",
    );
    claim(&mut ok, detect("x", b"<?xml version=\"1.0\"?><a/>") == Kind::Xml, "xml announces itself");
    claim(
        &mut ok,
        detect("x", b"<!DOCTYPE html><html>") == Kind::Html,
        "and html is told from xml by its doctype",
    );
    claim(&mut ok, detect("x", b"hello\0world") == Kind::Binary, "a NUL means binary");
    claim(&mut ok, detect("x", b"just some prose") == Kind::Text, "and prose is left as text");
    claim(&mut ok, !Kind::Binary.is_text() && Kind::Markdown.is_text(), "text and binary are told apart");

    // Scanning: the cases that break naive highlighters.
    let mut c = Carry::None;
    let s = scan(Lang::Rust, "let x = 1; // note", &mut c);
    claim(
        &mut ok,
        s.iter().any(|p| p.class == Class::Keyword) && s.last().map(|p| p.class) == Some(Class::Comment),
        "keywords and a trailing line comment",
    );
    let mut c = Carry::None;
    scan(Lang::Rust, "/* open", &mut c);
    claim(&mut ok, c == Carry::Block(1), "an unclosed block comment carries to the next line");
    scan(Lang::Rust, "still */ done", &mut c);
    claim(&mut ok, c == Carry::None, "and closes on it");
    let mut c = Carry::None;
    scan(Lang::Rust, "/* a /* b */", &mut c);
    claim(&mut ok, c == Carry::Block(1), "rust block comments nest");
    let mut c = Carry::None;
    scan(Lang::C, "/* a /* b */", &mut c);
    claim(&mut ok, c == Carry::None, "and C's do not");
    let mut c = Carry::None;
    let s = scan(Lang::Rust, "fn f<'a>(x: &'a str) -> i32 { 1 }", &mut c);
    claim(
        &mut ok,
        c == Carry::None && s.iter().any(|p| p.class == Class::Number),
        "a lifetime is not a character literal that eats the line",
    );
    let mut c = Carry::None;
    scan(Lang::Python, "s = \"\"\"open", &mut c);
    claim(&mut ok, matches!(c, Carry::Str(b'"', true)), "a python docstring carries");
    scan(Lang::Python, "still\"\"\" done", &mut c);
    claim(&mut ok, c == Carry::None, "and closes on its triple");
    let mut c = Carry::None;
    let s = scan(Lang::Rust, "let s = \"a // b\";", &mut c);
    claim(
        &mut ok,
        !s.iter().any(|p| p.class == Class::Comment),
        "a comment marker inside a string is not a comment",
    );

    ok &= xml::selftest();
    ok &= table::selftest();
    ok &= outline::selftest();
    kprintln!("  {} kinds known", 13);
    ok
}

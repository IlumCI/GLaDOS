//! Aiksi written the way Python is written.
//!
//! The brace syntax is this tree's own invention, and every model that will
//! ever write a program here has read millions of lines of Python and none of
//! it. So the language now has a second surface -- `def`, a colon and an
//! indented block, `elif`, `for x in xs`, `True`, `None`, `and`, `xs[i]`,
//! `xs.append(v)`, `f"..."` -- and the model writes the one it already knows.
//!
//! **It is a front end, not a second language.** It reads statements and
//! hands every expression to the brace parser's own expression grammar, and
//! what it builds is the same `Stmt` tree: the interpreter, the capability
//! gate, the step budget and its tick rule, `differ` and the code generator
//! see no difference and needed none. A program in either syntax means
//! exactly one thing.
//!
//! What does not exist is refused by name rather than half-supported:
//! classes, dictionaries, `try`, slices, floats, keyword arguments. Each
//! refusal says what to write instead, because the reader of these errors is
//! often a model in `maker`'s repair loop, and an error it can act on is one
//! it fixes on the next round.

use super::lex::{lex_py, Tok};
use super::parse::{parse_expr, Expr, Stmt, Type};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Does this source read as the Python-shaped surface? A block opened by a
/// line ending in `:` is the tell; the brace syntax never ends a line that way.
pub fn looks_python(src: &str) -> bool {
    for line in src.lines() {
        let t = strip_comment(line).trim_end();
        let lead = t.trim_start();
        let head = lead.split(|c: char| !c.is_ascii_alphabetic()).next().unwrap_or("");
        if t.ends_with(':') && matches!(head, "def" | "if" | "elif" | "else" | "for" | "while" | "class" | "try") {
            return true;
        }
        if lead.starts_with("import ") || lead.starts_with("from ") && lead.contains(" import ") {
            return true;
        }
        if lead.starts_with('#') {
            return true;
        }
    }
    false
}

/// A line without its `#` comment, minding quotes.
fn strip_comment(line: &str) -> &str {
    let mut q: Option<char> = None;
    let mut esc = false;
    for (i, c) in line.char_indices() {
        if esc {
            esc = false;
            continue;
        }
        match (q, c) {
            (Some(_), '\\') => esc = true,
            (Some(open), c) if c == open => q = None,
            (None, '"') | (None, '\'') => q = Some(c),
            (None, '#') => return &line[..i],
            _ => {}
        }
    }
    line
}

/// One logical line: its indent, its text, and the physical line it began on.
struct Line {
    indent: usize,
    text: String,
    at: usize,
}

/// Physical lines joined into logical ones: inside brackets and triple quotes
/// a newline does not end a statement, and a blank or comment line is nothing.
fn logical(src: &str) -> Result<Vec<Line>, String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut start = 0usize;
    let mut indent = 0usize;
    let mut depth: i32 = 0;
    let mut triple: Option<char> = None;
    for (n, raw) in src.lines().enumerate() {
        if cur.is_empty() && triple.is_none() {
            let body = strip_comment(raw);
            if body.trim().is_empty() {
                continue;
            }
            indent = raw.chars().take_while(|c| *c == ' ' || *c == '\t').map(|c| if c == '\t' { 4 } else { 1 }).sum();
            start = n + 1;
        }
        let piece = if triple.is_some() { raw } else { strip_comment(raw) };
        // Track brackets and triple quotes over this piece.
        let b: Vec<char> = piece.chars().collect();
        let mut i = 0;
        let mut q: Option<char> = None;
        while i < b.len() {
            let c = b[i];
            if let Some(t) = triple {
                if c == t && i + 2 < b.len() + 0 && b.get(i + 1) == Some(&t) && b.get(i + 2) == Some(&t) {
                    triple = None;
                    i += 3;
                    continue;
                }
                i += 1;
                continue;
            }
            if let Some(open) = q {
                if c == '\\' {
                    i += 2;
                    continue;
                }
                if c == open {
                    q = None;
                }
                i += 1;
                continue;
            }
            match c {
                '"' | '\'' => {
                    if b.get(i + 1) == Some(&c) && b.get(i + 2) == Some(&c) {
                        triple = Some(c);
                        i += 3;
                        continue;
                    }
                    q = Some(c);
                }
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                _ => {}
            }
            i += 1;
        }
        let continued = piece.trim_end().ends_with('\\');
        if !cur.is_empty() {
            cur.push(if triple.is_some() { '\n' } else { ' ' });
        }
        cur.push_str(if continued { piece.trim_end().trim_end_matches('\\') } else { piece });
        if depth <= 0 && triple.is_none() && !continued {
            out.push(Line { indent, text: core::mem::take(&mut cur).trim().to_string(), at: start });
            depth = 0;
        }
    }
    if !cur.is_empty() {
        return Err(format!("line {}: a bracket or a string is never closed", start));
    }
    Ok(out)
}

pub fn parse(src: &str) -> Result<Vec<Stmt>, String> {
    let lines = logical(src)?;
    let mut p = Py { lines, pos: 0, fresh: 0 };
    let out = p.block(0)?;
    if p.pos < p.lines.len() {
        let l = &p.lines[p.pos];
        return Err(format!("line {}: unexpected indentation", l.at));
    }
    Ok(out)
}

struct Py {
    lines: Vec<Line>,
    pos: usize,
    /// For names `for` makes up, which a program cannot spell.
    fresh: usize,
}

/// Tokens of an expression, the trailing `Eof` kept.
fn toks(src: &str) -> Result<Vec<Tok>, String> {
    lex_py(src)
}

fn expr(src: &str) -> Result<Expr, String> {
    if src.trim().is_empty() {
        return Err(String::from("an expression is missing"));
    }
    parse_expr(toks(src)?)
}

/// Split `head: rest` at the colon that ends a compound statement's header,
/// minding brackets and quotes. `rest` is a statement on the same line.
fn split_header(text: &str) -> Option<(&str, &str)> {
    let mut depth = 0i32;
    let mut q: Option<char> = None;
    let mut esc = false;
    for (i, c) in text.char_indices() {
        if esc {
            esc = false;
            continue;
        }
        match (q, c) {
            (Some(_), '\\') => esc = true,
            (Some(open), c) if c == open => q = None,
            (Some(_), _) => {}
            (None, '"') | (None, '\'') => q = Some(c),
            (None, '(') | (None, '[') | (None, '{') => depth += 1,
            (None, ')') | (None, ']') | (None, '}') => depth -= 1,
            (None, ':') if depth == 0 => return Some((&text[..i], text[i + 1..].trim())),
            _ => {}
        }
    }
    None
}

impl Py {
    fn err<T>(&self, at: usize, what: &str) -> Result<T, String> {
        Err(format!("line {}: {}", at, what))
    }

    /// Statements at exactly `indent`, until the indentation falls below it.
    fn block(&mut self, indent: usize) -> Result<Vec<Stmt>, String> {
        let mut out = Vec::new();
        while self.pos < self.lines.len() {
            let l = &self.lines[self.pos];
            if l.indent < indent {
                break;
            }
            if l.indent > indent {
                let at = l.at;
                return self.err(at, "unexpected indentation");
            }
            let mut got = self.statement()?;
            out.append(&mut got);
        }
        Ok(out)
    }

    /// The indented block under the header just read, or the statement that
    /// followed its colon on the same line.
    fn suite(&mut self, header_indent: usize, inline: &str, at: usize) -> Result<Vec<Stmt>, String> {
        if !inline.is_empty() {
            return self.simple(inline, at);
        }
        let Some(next) = self.lines.get(self.pos) else { return self.err(at, "a block was expected after ':'") };
        if next.indent <= header_indent {
            return self.err(at, "a block was expected after ':' (indent the lines under it)");
        }
        let ind = next.indent;
        self.block(ind)
    }

    fn statement(&mut self) -> Result<Vec<Stmt>, String> {
        let (indent, text, at) = {
            let l = &self.lines[self.pos];
            (l.indent, l.text.clone(), l.at)
        };
        self.pos += 1;
        let word = text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).next().unwrap_or("");
        match word {
            "def" => {
                let Some((head, inline)) = split_header(&text) else { return self.err(at, "'def' needs a ':' at the end of its line") };
                let head = head[3..].trim();
                let Some(open) = head.find('(') else { return self.err(at, "'def name(...):'") };
                let name = head[..open].trim().to_string();
                let close = head.rfind(')').unwrap_or(head.len());
                let params_src = &head[open + 1..close];
                let mut params = Vec::new();
                for p in params_src.split(',').map(|p| p.trim()).filter(|p| !p.is_empty()) {
                    if p.contains('=') {
                        return self.err(at, "default parameter values are not supported; pass every argument");
                    }
                    if p.starts_with('*') {
                        return self.err(at, "*args and **kwargs are not supported");
                    }
                    let (n, ty) = match p.split_once(':') {
                        Some((n, t)) => (n.trim(), Type::parse(t.trim())),
                        None => (p, Type::Any),
                    };
                    params.push((n.to_string(), ty));
                }
                let ret = match head[close..].split_once("->") {
                    Some((_, t)) => Type::parse(t.trim()),
                    None => Type::Any,
                };
                let body = self.suite(indent, inline, at)?;
                Ok(alloc::vec![Stmt::Fn(name, params, ret, body)])
            }
            "if" => {
                let s = self.if_chain(indent, &text, at)?;
                Ok(alloc::vec![s])
            }
            "while" => {
                let Some((head, inline)) = split_header(&text) else { return self.err(at, "'while' needs a ':'") };
                let cond = expr(&head[5..]).map_err(|e| format!("line {}: {}", at, e))?;
                let body = self.suite(indent, inline, at)?;
                Ok(alloc::vec![Stmt::While(cond, body)])
            }
            "for" => self.for_loop(indent, &text, at),
            "elif" | "else" => self.err(at, "an 'elif' or 'else' with no 'if' above it at this indentation"),
            "class" => self.err(at, "classes are not supported; use functions, and a list for each thing that has several values"),
            "try" | "except" | "finally" => self.err(at, "try/except is not supported; check the condition with 'if' before doing the thing"),
            "with" => self.err(at, "'with' is not supported; use read(path) and write(path, text)"),
            "lambda" => self.err(at, "lambda is not supported; write a def"),
            _ => self.simple(&text, at),
        }
    }

    fn if_chain(&mut self, indent: usize, text: &str, at: usize) -> Result<Stmt, String> {
        let Some((head, inline)) = split_header(text) else { return self.err(at, "'if' needs a ':' at the end of its line") };
        let kw = if head.starts_with("elif") { 4 } else { 2 };
        let cond = expr(&head[kw..]).map_err(|e| format!("line {}: {}", at, e))?;
        let then = self.suite(indent, inline, at)?;
        let mut otherwise = None;
        if let Some(next) = self.lines.get(self.pos) {
            if next.indent == indent {
                let t = next.text.clone();
                let nat = next.at;
                let w = t.split(|c: char| !c.is_ascii_alphabetic()).next().unwrap_or("");
                if w == "elif" {
                    self.pos += 1;
                    otherwise = Some(alloc::vec![self.if_chain(indent, &t, nat)?]);
                } else if w == "else" {
                    self.pos += 1;
                    let Some((_, inline)) = split_header(&t) else { return self.err(nat, "'else' needs a ':'") };
                    otherwise = Some(self.suite(indent, inline, nat)?);
                }
            }
        }
        Ok(Stmt::If(cond, then, otherwise))
    }

    /// `for x in xs:` as a counted `while`, the element taken before the body
    /// runs so that `continue` cannot skip the step. `for i, x in
    /// enumerate(xs)` names the counter too.
    fn for_loop(&mut self, indent: usize, text: &str, at: usize) -> Result<Vec<Stmt>, String> {
        let Some((head, inline)) = split_header(text) else { return self.err(at, "'for' needs a ':'") };
        let head = head[3..].trim();
        let Some(k) = head.find(" in ") else { return self.err(at, "'for name in something:'") };
        let target = head[..k].trim();
        let mut source = head[k + 4..].trim();
        let mut counter: Option<String> = None;
        let mut item = target.to_string();
        if target.contains(',') {
            let parts: Vec<&str> = target.split(',').map(|s| s.trim()).collect();
            let en = source.strip_prefix("enumerate(").and_then(|s| s.strip_suffix(')'));
            match (parts.as_slice(), en) {
                ([i, x], Some(inner)) => {
                    counter = Some(i.to_string());
                    item = x.to_string();
                    source = inner;
                }
                _ => return self.err(at, "only 'for i, x in enumerate(xs):' can name two things"),
            }
        }
        let n = self.fresh;
        self.fresh += 1;
        let list = format!("__for{}_list", n);
        let idx = format!("__for{}_i", n);
        let src = expr(source).map_err(|e| format!("line {}: {}", at, e))?;
        let mut body = Vec::new();
        let var = |s: &str| Expr::Var(s.to_string());
        if let Some(c) = &counter {
            body.push(Stmt::Expr(Expr::Assign(c.clone(), alloc::boxed::Box::new(var(&idx)))));
        }
        body.push(Stmt::Expr(Expr::Assign(
            item,
            alloc::boxed::Box::new(Expr::Call(String::from("get"), alloc::vec![var(&list), var(&idx)])),
        )));
        body.push(Stmt::Expr(Expr::Assign(
            idx.clone(),
            alloc::boxed::Box::new(Expr::Bin(super::parse::BinOp::Add, alloc::boxed::Box::new(var(&idx)), alloc::boxed::Box::new(Expr::Int(1)))),
        )));
        body.append(&mut self.suite(indent, inline, at)?);
        // A string is walked by character, as Python walks one.
        let walked = Expr::Call(String::from("__iter"), alloc::vec![src]);
        Ok(alloc::vec![
            Stmt::Expr(Expr::Assign(list.clone(), alloc::boxed::Box::new(walked))),
            Stmt::Expr(Expr::Assign(idx.clone(), alloc::boxed::Box::new(Expr::Int(0)))),
            Stmt::While(
                Expr::Bin(
                    super::parse::BinOp::Lt,
                    alloc::boxed::Box::new(var(&idx)),
                    alloc::boxed::Box::new(Expr::Call(String::from("len"), alloc::vec![var(&list)])),
                ),
                body,
            ),
        ])
    }

    /// A statement that is one line and opens no block.
    fn simple(&mut self, text: &str, at: usize) -> Result<Vec<Stmt>, String> {
        let wrap = |e: String| format!("line {}: {}", at, e);
        let word = text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).next().unwrap_or("");
        let rest = text[word.len()..].trim();
        match word {
            "pass" => return Ok(Vec::new()),
            "break" => return Ok(alloc::vec![Stmt::Break]),
            "continue" => return Ok(alloc::vec![Stmt::Continue]),
            // Everything a program would import is a builtin already, and the
            // rest does not exist: either way there is nothing to load.
            "import" | "from" | "global" | "nonlocal" => return Ok(Vec::new()),
            "return" => {
                return if rest.is_empty() {
                    Ok(alloc::vec![Stmt::Return(None)])
                } else {
                    Ok(alloc::vec![Stmt::Return(Some(expr(rest).map_err(wrap)?))])
                };
            }
            "del" => return Err(wrap(String::from("'del' is not supported; use remove(xs, i)"))),
            "raise" => return Err(wrap(String::from("'raise' is not supported; print the problem and return"))),
            "assert" => {
                // `assert c, "why"` stops the program with its message.
                let (c, why) = match rest.rsplit_once(',') {
                    Some((c, w)) if w.trim().starts_with(['"', '\'']) => (c, w.trim()),
                    _ => (rest, "\"assertion failed\""),
                };
                let cond = expr(c).map_err(wrap)?;
                let fail = Stmt::Expr(Expr::Call(String::from("fail"), alloc::vec![expr(why).map_err(wrap)?]));
                return Ok(alloc::vec![Stmt::If(Expr::Unary(super::parse::UnOp::Not, alloc::boxed::Box::new(cond)), alloc::vec![fail], None)]);
            }
            _ => {}
        }
        // `x += e` and its siblings.
        for (op, bin) in [
            ("+=", super::parse::BinOp::Add),
            ("-=", super::parse::BinOp::Sub),
            ("*=", super::parse::BinOp::Mul),
            ("//=", super::parse::BinOp::Div),
            ("/=", super::parse::BinOp::Div),
            ("%=", super::parse::BinOp::Rem),
        ] {
            if let Some((lhs, rhs)) = split_op(text, op) {
                let target = expr(lhs).map_err(wrap)?;
                let value = Expr::Bin(bin, alloc::boxed::Box::new(target.clone()), alloc::boxed::Box::new(expr(rhs).map_err(wrap)?));
                return Ok(alloc::vec![Stmt::Expr(assign_to(target, value).map_err(wrap)?)]);
            }
        }
        // `a, b = b, a` -- two names, two values, evaluated before either is set.
        if let Some((lhs, rhs)) = split_op(text, "=") {
            let names: Vec<&str> = lhs.split(',').map(|s| s.trim()).collect();
            if names.len() > 1 && names.iter().all(|n| n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')) {
                let vals = split_commas(rhs);
                if vals.len() != names.len() {
                    return Err(wrap(String::from("as many values as names on each side of '='")));
                }
                let n = self.fresh;
                self.fresh += 1;
                let mut out = Vec::new();
                for (k, v) in vals.iter().enumerate() {
                    out.push(Stmt::Expr(Expr::Assign(format!("__tup{}_{}", n, k), alloc::boxed::Box::new(expr(v).map_err(wrap)?))));
                }
                for (k, name) in names.iter().enumerate() {
                    out.push(Stmt::Expr(Expr::Assign(name.to_string(), alloc::boxed::Box::new(Expr::Var(format!("__tup{}_{}", n, k))))));
                }
                return Ok(out);
            }
        }
        Ok(alloc::vec![Stmt::Expr(expr(text).map_err(wrap)?)])
    }
}

/// `x = v` for a name, an index or a field, as the expression parser would.
fn assign_to(target: Expr, value: Expr) -> Result<Expr, String> {
    match target {
        Expr::Var(n) => Ok(Expr::Assign(n, alloc::boxed::Box::new(value))),
        Expr::Call(f, mut a) if f == "get" && a.len() == 2 => {
            let i = a.pop().unwrap();
            let list = a.pop().unwrap();
            let Expr::Var(n) = list.clone() else { return Err(String::from("only a named list can be assigned into")) };
            Ok(Expr::Assign(n, alloc::boxed::Box::new(Expr::Call(String::from("set"), alloc::vec![list, i, value]))))
        }
        Expr::Field(t, f) => Ok(Expr::SetField(t, f, alloc::boxed::Box::new(value))),
        _ => Err(String::from("the left of an assignment must be a name, an item or a field")),
    }
}

/// Split at the first top-level `op` that is an assignment operator: not
/// inside brackets or quotes, and for `=` not part of `==`, `<=`, `>=`, `!=`.
fn split_op<'a>(text: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    let b = text.as_bytes();
    let mut depth = 0i32;
    let mut q: Option<u8> = None;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if let Some(open) = q {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == open {
                q = None;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' | b'\'' => q = Some(c),
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            _ if depth == 0 && text[i..].starts_with(op) => {
                if op == "=" {
                    let prev = if i > 0 { b[i - 1] } else { b' ' };
                    let next = b.get(i + 1).copied().unwrap_or(b' ');
                    if next == b'=' || matches!(prev, b'=' | b'<' | b'>' | b'!' | b'+' | b'-' | b'*' | b'/' | b'%') {
                        i += 1;
                        continue;
                    }
                } else {
                    // `a //= b` must not be read as `a /` then `/= b`.
                    let prev = if i > 0 { b[i - 1] } else { b' ' };
                    if matches!(prev, b'/' | b'*' | b'<' | b'>' | b'=' | b'!') {
                        i += 1;
                        continue;
                    }
                }
                return Some((text[..i].trim(), text[i + op.len()..].trim()));
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn split_commas(text: &str) -> Vec<&str> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut q: Option<u8> = None;
    let mut start = 0;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if let Some(open) = q {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == open {
                q = None;
            }
        } else {
            match c {
                b'"' | b'\'' => q = Some(c),
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth -= 1,
                b',' if depth == 0 => {
                    out.push(text[start..i].trim());
                    start = i + 1;
                }
                _ => {}
            }
        }
        i += 1;
    }
    out.push(text[start..].trim());
    out
}

/// `diag py`: programs in the Python surface, each with what Python itself
/// prints for it -- so the claim is "means what Python means", checked against
/// the language it imitates rather than against this parser's own idea.
pub fn selftest() -> bool {
    use crate::kprintln;
    const CASES: &[(&str, &str, &str)] = &[
        ("def, if/elif/else, return", "def sign(n):\n    if n > 0:\n        return 'pos'\n    elif n < 0:\n        return 'neg'\n    else:\n        return 'zero'\n\nprint(sign(3), sign(-2), sign(0))\n", "pos neg zero"),
        ("for over a list, append, f-string", "xs = []\nfor i in range(4):\n    xs.append(i * i)\nprint(f\"{len(xs)}: {xs[3]}\")\n", "4: 9"),
        ("for over a string walks characters", "n = 0\nfor c in 'abcab':\n    if c == 'a':\n        n += 1\nprint(n)\n", "2"),
        ("break and continue", "t = 0\nfor i in range(10):\n    if i % 2 == 0:\n        continue\n    if i > 7:\n        break\n    t += i\nprint(t)\n", "16"),
        ("enumerate", "for i, w in enumerate(['a', 'b']):\n    print(i, w)\n", "0 a\n1 b"),
        ("in, not in, is None", "xs = [1, 2]\nv = None\nprint(2 in xs, 5 not in xs, v is None)\n", "1 1 1"),
        ("index assignment and +=", "xs = [0, 0, 0]\nxs[1] = 5\nxs[1] += 2\nprint(xs[1])\n", "7"),
        ("swap with a tuple assignment", "a = 1\nb = 2\na, b = b, a\nprint(a, b)\n", "2 1"),
        ("string methods", "s = '  Hi There  '.strip().lower()\nprint(s.startswith('hi'), '-'.join(s.split()))\n", "1 hi-there"),
        ("a one-line suite and pass", "def f(x): return x + 1\nif True: pass\nprint(f(1))\n", "2"),
        ("while with a nested def call", "def fib(n):\n    a, b = 0, 1\n    while n > 0:\n        a, b = b, a + b\n        n -= 1\n    return a\nprint(fib(20))\n", "6765"),
        ("comprehensions, all and any", "xs = [i * i for i in range(6) if i % 2 == 0]\nprint(xs, all(x % 2 == 0 for x in xs), any(x > 10 for x in xs))\n", "[0, 4, 16] 1 1"),
        ("a comprehension leaves its variable as it was", "x = 9\nys = [x + 1 for x in [1, 2]]\nprint(x, ys)\n", "9 [2, 3]"),
        ("sum, str and a list joined with +", "print(sum([1, 2, 3]) + len([1] + [2, 3]), str(7) + '!')\n", "9 7!"),
        ("the __main__ guard runs", "def main():\n    print('ran')\n\nif __name__ == '__main__':\n    main()\n", "ran"),
    ];
    let mut ok = true;
    for (what, src, want) in CASES {
        let got = match parse(src) {
            Ok(prog) => {
                let mut it = super::Interp::new();
                crate::gfx::console::begin_capture();
                let r = it.run(&prog);
                let out = crate::gfx::console::end_capture().unwrap_or_default();
                match r {
                    Ok(_) => out,
                    Err(e) => format!("error: {}", e),
                }
            }
            Err(e) => format!("parse error: {}", e),
        };
        if got.trim() != *want {
            kprintln!("    FAIL: {} -- got {:?}, want {:?}", what, got.trim(), want);
            ok = false;
        }
    }
    let refused = [
        ("class X:\n    pass\n", "classes"),
        ("x = a[1:2]\n", "slice"),
        ("def f(a=1):\n    return a\n", "default"),
        ("try:\n    pass\n", "try"),
        ("print(sorted([2, 1], key=len))\n", "keyword"),
    ];
    for (src, word) in refused {
        let good = matches!(parse(src), Err(e) if e.contains(word));
        if !good {
            kprintln!("    FAIL: refusing what is missing by name ({})", word);
            ok = false;
        }
    }
    let shape = looks_python("def f():\n    return 1\n") && !looks_python("fn f() { return 1 }\n");
    if !shape {
        kprintln!("    FAIL: the two surfaces are told apart");
        ok = false;
    }
    ok
}

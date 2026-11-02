//! Tokeniser.
//!
//! Hand-written, single pass, no regex and no tables. The language is small
//! enough that a byte-at-a-time scanner is both the clearest and the fastest
//! thing to write.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Int(i64),
    Str(String),
    Ident(String),

    Plus,
    Minus,
    Star,
    Slash,
    Percent,

    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    NotEq,

    AndAnd,
    OrOr,
    Not,

    Amp,
    Pipe,
    Caret,
    Tilde,
    Shl,
    Shr,

    Assign,
    /// Field access. Aiksi has no floats, so a `.` can never be part of a
    /// number and needs no lookahead to tell the two apart.
    Dot,
    /// Type annotations, and only there. Not a statement separator and not a
    /// map literal, so a `:` outside a declaration is a parse error rather
    /// than something with a second meaning.
    Colon,
    LParen,
    RParen,
    /// `[` and `]`: list literals and indexing.
    LBracket,
    RBracket,
    /// `in` and `not in`, Python's surface only: membership in a list or a
    /// string, which is `contains` with its arguments the other way round.
    In,
    NotIn,
    LBrace,
    RBrace,
    Semi,
    Comma,

    Eof,
}

pub fn lex(src: &str) -> Result<Vec<Tok>, String> {
    lex_mode(src, false)
}

/// The Python-shaped surface (`py.rs`): `#` comments rather than `//`,
/// `and`/`or`/`not` as words, `True`, `False` and `None`, `//` as division,
/// and `print` that ends its line. One lexer for both, so a string or a number
/// cannot mean one thing in one syntax and another in the other.
pub fn lex_py(src: &str) -> Result<Vec<Tok>, String> {
    let raw = lex_mode(src, true)?;
    // Operators spelled as words, decided with a token of lookahead:
    // `not in`, `is not`, `in`, `is`.
    let mut out: Vec<Tok> = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        let next_is = |w: &str| matches!(raw.get(i + 1), Some(Tok::Ident(s)) if s == w);
        match &raw[i] {
            Tok::Not if next_is("in") => {
                out.push(Tok::NotIn);
                i += 2;
                continue;
            }
            Tok::Ident(w) if w == "is" => {
                if matches!(raw.get(i + 1), Some(Tok::Not)) {
                    out.push(Tok::NotEq);
                    i += 2;
                } else {
                    out.push(Tok::EqEq);
                    i += 1;
                }
                continue;
            }
            Tok::Ident(w) if w == "in" => out.push(Tok::In),
            t => out.push(t.clone()),
        }
        i += 1;
    }
    Ok(out)
}

fn lex_mode(src: &str, py: bool) -> Result<Vec<Tok>, String> {
    let b = src.as_bytes();
    let mut i = 0usize;
    let mut out = Vec::new();

    while i < b.len() {
        let c = b[i];

        // whitespace
        if c == b' ' || c == b'\t' || c == b'\r' || c == b'\n' {
            i += 1;
            continue;
        }

        // # line comment, in Python's surface
        if py && c == b'#' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // // line comment
        if !py && c == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // numbers: decimal, or 0x hex
        if c.is_ascii_digit() {
            let start = i;
            let mut value: i64 = 0;
            if c == b'0' && i + 1 < b.len() && (b[i + 1] | 0x20) == b'x' {
                i += 2;
                let hex_start = i;
                while i < b.len() && b[i].is_ascii_hexdigit() {
                    let d = (b[i] as char).to_digit(16).unwrap() as i64;
                    value = value.wrapping_mul(16).wrapping_add(d);
                    i += 1;
                }
                if i == hex_start {
                    return Err("expected hex digits after 0x".to_string());
                }
            } else {
                while i < b.len() && b[i].is_ascii_digit() {
                    value = value.wrapping_mul(10).wrapping_add((b[i] - b'0') as i64);
                    i += 1;
                }
            }
            // Reject 123abc rather than silently lexing it as 123 then abc.
            if i < b.len() && (b[i].is_ascii_alphabetic() || b[i] == b'_') {
                let bad = String::from_utf8_lossy(&b[start..=i]).to_string();
                return Err(format!("malformed number near '{}'", bad));
            }
            out.push(Tok::Int(value));
            continue;
        }

        // identifiers and keywords
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let word = String::from_utf8_lossy(&b[start..i]).to_string();
            // An f-string: f"..." with {expr} inside becomes the pieces joined
            // with `+`, each expression through `str`.
            if py && (word == "f" || word == "F") && i < b.len() && (b[i] == b'"' || b[i] == b'\'') {
                let (text, next) = string_at(b, i)?;
                i = next;
                fstring(&text, &mut out)?;
                continue;
            }
            if py {
                match word.as_str() {
                    "and" => out.push(Tok::AndAnd),
                    "or" => out.push(Tok::OrOr),
                    "not" => out.push(Tok::Not),
                    "True" => out.push(Tok::Int(1)),
                    "False" => out.push(Tok::Int(0)),
                    "None" => {
                        out.push(Tok::Ident(String::from("nil")));
                        out.push(Tok::LParen);
                        out.push(Tok::RParen);
                    }
                    // A program run here is always the main program.
                    "__name__" => out.push(Tok::Str(String::from("__main__"))),
                    "print" => out.push(Tok::Ident(String::from("println"))),
                    "sorted" => out.push(Tok::Ident(String::from("sort"))),
                    "reversed" => out.push(Tok::Ident(String::from("reverse"))),
                    _ => out.push(Tok::Ident(word)),
                }
                continue;
            }
            out.push(Tok::Ident(word));
            continue;
        }

        // Python's quotes: either kind, and tripled.
        if py && (c == b'"' || c == b'\'') {
            let (text, next) = string_at(b, i)?;
            i = next;
            out.push(Tok::Str(text));
            continue;
        }

        // string literals
        if c == b'"' {
            i += 1;
            let mut s = String::new();
            loop {
                if i >= b.len() {
                    return Err("unterminated string".to_string());
                }
                match b[i] {
                    b'"' => {
                        i += 1;
                        break;
                    }
                    b'\\' => {
                        i += 1;
                        if i >= b.len() {
                            return Err("unterminated escape".to_string());
                        }
                        let e = match b[i] {
                            b'n' => '\n',
                            b't' => '\t',
                            b'r' => '\r',
                            b'0' => '\0',
                            b'\\' => '\\',
                            b'"' => '"',
                            other => {
                                return Err(format!("unknown escape '\\{}'", other as char))
                            }
                        };
                        s.push(e);
                        i += 1;
                    }
                    other => {
                        s.push(other as char);
                        i += 1;
                    }
                }
            }
            out.push(Tok::Str(s));
            continue;
        }

        // operators; two-character forms must be tested before one-character
        let two = if i + 1 < b.len() { Some((c, b[i + 1])) } else { None };
        if let Some((a, d)) = two {
            let t = match (a, d) {
                (b'=', b'=') => Some(Tok::EqEq),
                (b'!', b'=') => Some(Tok::NotEq),
                (b'<', b'=') => Some(Tok::Le),
                (b'>', b'=') => Some(Tok::Ge),
                (b'&', b'&') => Some(Tok::AndAnd),
                (b'|', b'|') => Some(Tok::OrOr),
                (b'<', b'<') => Some(Tok::Shl),
                (b'>', b'>') => Some(Tok::Shr),
                (b'/', b'/') if py => Some(Tok::Slash),
                (b'*', b'*') if py => return Err("'**' is not supported; multiply, or use pow(a, b)".to_string()),
                _ => None,
            };
            if let Some(t) = t {
                out.push(t);
                i += 2;
                continue;
            }
        }

        let t = match c {
            b'+' => Tok::Plus,
            b'-' => Tok::Minus,
            b'*' => Tok::Star,
            b'/' => Tok::Slash,
            b'%' => Tok::Percent,
            b'<' => Tok::Lt,
            b'>' => Tok::Gt,
            b'!' => Tok::Not,
            b'&' => Tok::Amp,
            b'|' => Tok::Pipe,
            b'^' => Tok::Caret,
            b'~' => Tok::Tilde,
            b'=' => Tok::Assign,
            b'.' => Tok::Dot,
            b':' => Tok::Colon,
            b'(' => Tok::LParen,
            b')' => Tok::RParen,
            b'[' => Tok::LBracket,
            b']' => Tok::RBracket,
            b'{' => Tok::LBrace,
            b'}' => Tok::RBrace,
            b';' => Tok::Semi,
            b',' => Tok::Comma,
            other => return Err(format!("unexpected character '{}'", other as char)),
        };
        out.push(t);
        i += 1;
    }

    out.push(Tok::Eof);
    Ok(out)
}


/// A quoted string starting at `b[i]`, either quote, single or tripled, with
/// the escapes the brace syntax has. Answers the text and where it ended.
fn string_at(b: &[u8], mut i: usize) -> Result<(String, usize), String> {
    let q = b[i];
    let triple = i + 2 < b.len() && b[i + 1] == q && b[i + 2] == q;
    i += if triple { 3 } else { 1 };
    let mut s: Vec<u8> = Vec::new();
    loop {
        if i >= b.len() {
            return Err("unterminated string".to_string());
        }
        if triple {
            if i + 2 < b.len() && b[i] == q && b[i + 1] == q && b[i + 2] == q {
                i += 3;
                break;
            }
        } else if b[i] == q {
            i += 1;
            break;
        } else if b[i] == b'\n' {
            return Err("a string runs past the end of its line".to_string());
        }
        if b[i] == b'\\' && i + 1 < b.len() {
            i += 1;
            let e = match b[i] {
                b'n' => b'\n',
                b't' => b'\t',
                b'r' => b'\r',
                b'0' => 0,
                b'\\' => b'\\',
                b'"' => b'"',
                b'\'' => b'\'',
                b'\n' => {
                    i += 1;
                    continue;
                }
                other => return Err(format!("unknown escape '\\{}'", other as char)),
            };
            s.push(e);
            i += 1;
            continue;
        }
        s.push(b[i]);
        i += 1;
    }
    Ok((String::from_utf8_lossy(&s).into_owned(), i))
}

/// `f"a {x} b"` as `("a " + str(x) + " b")`. A format spec after `:` is
/// dropped -- `{x:.2f}` prints `x` -- because there are no floats to format.
fn fstring(text: &str, out: &mut Vec<Tok>) -> Result<(), String> {
    out.push(Tok::LParen);
    out.push(Tok::Str(String::new()));
    let mut lit = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                lit.push('{');
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                lit.push('}');
            }
            '{' => {
                if !lit.is_empty() {
                    out.push(Tok::Plus);
                    out.push(Tok::Str(core::mem::take(&mut lit)));
                }
                let mut inner = String::new();
                let mut depth = 0;
                loop {
                    match chars.next() {
                        None => return Err("an f-string has a '{' with no '}'".to_string()),
                        Some('}') if depth == 0 => break,
                        Some(ch) => {
                            if ch == '(' || ch == '[' {
                                depth += 1;
                            }
                            if ch == ')' || ch == ']' {
                                depth -= 1;
                            }
                            inner.push(ch);
                        }
                    }
                }
                let expr = match inner.find(':') {
                    Some(k) if !inner[..k].contains('[') => &inner[..k],
                    _ => &inner[..],
                };
                let expr = expr.trim_end_matches('=').trim_end_matches('!').trim();
                let mut toks = lex_py(expr)?;
                toks.pop(); // Eof
                out.push(Tok::Plus);
                out.push(Tok::Ident(String::from("str")));
                out.push(Tok::LParen);
                out.extend(toks);
                out.push(Tok::RParen);
            }
            other => lit.push(other),
        }
    }
    if !lit.is_empty() {
        out.push(Tok::Plus);
        out.push(Tok::Str(lit));
    }
    out.push(Tok::RParen);
    Ok(())
}

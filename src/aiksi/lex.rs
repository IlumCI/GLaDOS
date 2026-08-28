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
    LBrace,
    RBrace,
    Semi,
    Comma,

    Eof,
}

pub fn lex(src: &str) -> Result<Vec<Tok>, String> {
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

        // // line comment
        if c == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
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
            out.push(Tok::Ident(String::from_utf8_lossy(&b[start..i]).to_string()));
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

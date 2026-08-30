//! Delimited and keyed records: CSV, TSV, JSON Lines and INI.
//!
//! Four formats that all mean "a list of records", kept together because they
//! answer the same question and differ only in how a field ends.
//!
//! The quoting rules are the whole of CSV and they are the part everybody gets
//! wrong. A field may contain the delimiter, a newline, and the quote character
//! itself, and the only way to know is to be inside a quoted field at the time.
//! Splitting on commas produces a reader that works on every file anybody tests
//! it with and corrupts the first real one.

use alloc::string::String;
use alloc::vec::Vec;

/// Parse delimiter-separated records, honouring quotes.
///
/// Takes the whole input rather than a line at a time, because a quoted field
/// may contain a newline and a line-oriented reader cannot see that.
pub fn delimited(src: &str, delim: char) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut cell = String::new();
    let mut quoted = false;
    let mut chars = src.chars().peekable();
    let mut any = false;

    while let Some(c) = chars.next() {
        any = true;
        if quoted {
            if c == '"' {
                // A doubled quote inside a quoted field is one literal quote.
                // This is the rule that makes a naive reader lose data rather
                // than fail, which is worse.
                if chars.peek() == Some(&'"') {
                    chars.next();
                    cell.push('"');
                } else {
                    quoted = false;
                }
            } else {
                cell.push(c);
            }
            continue;
        }
        if c == '"' && cell.is_empty() {
            quoted = true;
            continue;
        }
        if c == delim {
            row.push(core::mem::take(&mut cell));
            continue;
        }
        if c == '\n' {
            row.push(core::mem::take(&mut cell));
            rows.push(core::mem::take(&mut row));
            continue;
        }
        if c == '\r' {
            continue;
        }
        cell.push(c);
    }
    if !cell.is_empty() || !row.is_empty() {
        row.push(cell);
        rows.push(row);
    } else if any && rows.is_empty() {
        rows.push(Vec::new());
    }
    rows
}

/// One JSON value per line, with the line number of anything that will not
/// parse.
///
/// Reporting the bad line rather than failing the file is the point: a log of
/// a million records with one truncated write in the middle is still worth
/// reading, and a reader that answers `None` for the whole thing has thrown
/// away the other 999,999.
pub fn json_lines(src: &str) -> (Vec<crate::json::Json>, Vec<usize>) {
    let mut out = Vec::new();
    let mut bad = Vec::new();
    for (n, line) in src.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        match crate::json::Json::parse(t) {
            Some(v) => out.push(v),
            None => bad.push(n + 1),
        }
    }
    (out, bad)
}

/// A section of an INI file, or the unnamed one before the first header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub name: String,
    pub pairs: Vec<(String, String)>,
}

/// Parse `key = value` under `[section]`.
///
/// Covers INI, most `.conf` files, and the part of TOML that looks like INI,
/// which is most hand-written TOML. It does not cover TOML's arrays, inline
/// tables or typed values, and it does not pretend to: a value is the text
/// after the equals sign with surrounding quotes removed, and a caller that
/// wants a number parses one.
pub fn ini(src: &str) -> Vec<Section> {
    let mut out: Vec<Section> = Vec::new();
    let mut cur = Section { name: String::new(), pairs: Vec::new() };
    for line in src.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with(';') {
            continue;
        }
        if t.starts_with('[') && t.ends_with(']') {
            if !cur.name.is_empty() || !cur.pairs.is_empty() {
                out.push(core::mem::replace(
                    &mut cur,
                    Section { name: String::new(), pairs: Vec::new() },
                ));
            }
            cur.name = String::from(&t[1..t.len() - 1]);
            continue;
        }
        if let Some((k, v)) = t.split_once('=') {
            let v = v.trim();
            let v = v.strip_prefix('"').and_then(|s| s.strip_suffix('"')).unwrap_or(v);
            cur.pairs.push((String::from(k.trim()), String::from(v)));
        }
    }
    if !cur.name.is_empty() || !cur.pairs.is_empty() {
        out.push(cur);
    }
    out
}

pub fn selftest() -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    let r = delimited("a,b,c\n1,2,3\n", ',');
    claim(&mut ok, r.len() == 2 && r[1] == ["1", "2", "3"], "plain rows and columns");
    let r = delimited("\"a,b\",c\n", ',');
    claim(&mut ok, r[0] == ["a,b", "c"], "a delimiter inside quotes is data");
    let r = delimited("\"say \"\"hi\"\"\",x\n", ',');
    claim(&mut ok, r[0][0] == "say \"hi\"", "a doubled quote is one literal quote");
    let r = delimited("\"two\nlines\",x\n", ',');
    claim(&mut ok, r.len() == 1 && r[0][0] == "two\nlines", "a newline inside quotes does not end the row");
    let r = delimited("a\tb\n", '\t');
    claim(&mut ok, r[0] == ["a", "b"], "tabs work the same way");
    let r = delimited("a,,c\n", ',');
    claim(&mut ok, r[0].len() == 3 && r[0][1].is_empty(), "an empty field is a field");

    let (v, bad) = json_lines("{\"a\":1}\nnot json\n{\"a\":2}\n");
    claim(&mut ok, v.len() == 2 && bad == [2], "a bad line is reported and the rest survive");

    let s = ini("top = 1\n# c\n[one]\nk = \"v\"\n[two]\nn=3\n");
    claim(&mut ok, s.len() == 3, "the unnamed section counts");
    claim(&mut ok, s[1].name == "one" && s[1].pairs[0].1 == "v", "quotes are stripped from values");
    claim(&mut ok, s[2].pairs[0] == (String::from("n"), String::from("3")), "spacing around equals is ignored");
    ok
}

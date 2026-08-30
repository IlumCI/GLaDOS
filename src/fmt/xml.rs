//! An XML reader, for the shapes XML is actually used in here.
//!
//! `net::html` already parses HTML, and HTML is not XML: it has void elements
//! that never close, optional end tags, and a parse algorithm defined by what
//! browsers do rather than by a grammar. Reusing it for XML would import all of
//! that leniency, and leniency in a data format turns a malformed file into a
//! plausible tree.
//!
//! So this is strict about the things a data format should be strict about.
//! Every element closes, closing tags match, and a mismatch is an error with a
//! position rather than a recovery. It is lenient about exactly one thing:
//! namespaces are kept as part of the name and never resolved, because
//! resolution needs a prefix map nobody here would consult.
//!
//! Not implemented, and refused rather than ignored: entity definitions,
//! external references and processing instructions beyond the declaration.
//! A document carrying a DOCTYPE with an internal subset is refused, because
//! an internal subset can redefine entities and honouring some of a DTD while
//! ignoring the rest is how a reader disagrees with every other reader.

use alloc::string::String;
use alloc::vec::Vec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub kids: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item {
    Elem(Node),
    Text(String),
}

impl Node {
    /// The first direct child element with this name.
    pub fn child(&self, name: &str) -> Option<&Node> {
        self.kids.iter().find_map(|k| match k {
            Item::Elem(e) if e.name == name => Some(e),
            _ => None,
        })
    }

    /// Every direct child element with this name.
    pub fn children(&self, name: &str) -> Vec<&Node> {
        self.kids
            .iter()
            .filter_map(|k| match k {
                Item::Elem(e) if e.name == name => Some(e),
                _ => None,
            })
            .collect()
    }

    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    /// All text under this element, concatenated.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for k in &self.kids {
            match k {
                Item::Text(t) => out.push_str(t),
                Item::Elem(e) => out.push_str(&e.text()),
            }
        }
        out
    }
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
    depth: u32,
}

/// Deeper than any document this system has a reason to read, and shallow
/// enough that the recursion cannot exhaust a kernel stack with no guard page.
const MAX_DEPTH: u32 = 64;

pub fn parse(src: &str) -> Result<Node, String> {
    let mut p = P { b: src.as_bytes(), i: 0, depth: 0 };
    p.prolog()?;
    let root = p.element()?;
    p.ws();
    // Trailing comments and whitespace are allowed. Anything else means the
    // document has two roots, which is not a document.
    while p.i < p.b.len() {
        if p.b[p.i..].starts_with(b"<!--") {
            p.comment()?;
            p.ws();
            continue;
        }
        return Err(alloc::format!("trailing content at byte {}", p.i));
    }
    Ok(root)
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && (self.b[self.i] as char).is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn comment(&mut self) -> Result<(), String> {
        self.i += 4;
        while self.i + 2 < self.b.len() {
            if &self.b[self.i..self.i + 3] == b"-->" {
                self.i += 3;
                return Ok(());
            }
            self.i += 1;
        }
        Err(String::from("unterminated comment"))
    }

    fn prolog(&mut self) -> Result<(), String> {
        loop {
            self.ws();
            if self.b[self.i..].starts_with(b"<?") {
                while self.i + 1 < self.b.len() && &self.b[self.i..self.i + 2] != b"?>" {
                    self.i += 1;
                }
                if self.i + 1 >= self.b.len() {
                    return Err(String::from("unterminated declaration"));
                }
                self.i += 2;
                continue;
            }
            if self.b[self.i..].starts_with(b"<!--") {
                self.comment()?;
                continue;
            }
            if self.b[self.i..].starts_with(b"<!DOCTYPE") {
                let rest = &self.b[self.i..];
                let end = rest.iter().position(|&c| c == b'>').unwrap_or(rest.len());
                if rest[..end].contains(&b'[') {
                    return Err(String::from("a DOCTYPE with an internal subset is refused"));
                }
                self.i += end + 1;
                continue;
            }
            return Ok(());
        }
    }

    fn name(&mut self) -> Result<String, String> {
        let start = self.i;
        while self.i < self.b.len() {
            let c = self.b[self.i];
            if c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b'.' || c == b':' {
                self.i += 1;
            } else {
                break;
            }
        }
        if self.i == start {
            return Err(alloc::format!("expected a name at byte {}", start));
        }
        Ok(String::from_utf8_lossy(&self.b[start..self.i]).into_owned())
    }

    fn element(&mut self) -> Result<Node, String> {
        if self.depth >= MAX_DEPTH {
            return Err(String::from("nesting too deep"));
        }
        if self.i >= self.b.len() || self.b[self.i] != b'<' {
            return Err(alloc::format!("expected an element at byte {}", self.i));
        }
        self.i += 1;
        let name = self.name()?;
        let mut attrs = Vec::new();
        loop {
            self.ws();
            if self.i >= self.b.len() {
                return Err(String::from("unterminated tag"));
            }
            if self.b[self.i] == b'>' {
                self.i += 1;
                break;
            }
            if self.b[self.i..].starts_with(b"/>") {
                self.i += 2;
                return Ok(Node { name, attrs, kids: Vec::new() });
            }
            let k = self.name()?;
            self.ws();
            if self.i >= self.b.len() || self.b[self.i] != b'=' {
                return Err(alloc::format!("attribute '{}' has no value", k));
            }
            self.i += 1;
            self.ws();
            let q = if self.i < self.b.len() { self.b[self.i] } else { 0 };
            if q != b'"' && q != b'\'' {
                return Err(alloc::format!("attribute '{}' is not quoted", k));
            }
            self.i += 1;
            let s = self.i;
            while self.i < self.b.len() && self.b[self.i] != q {
                self.i += 1;
            }
            if self.i >= self.b.len() {
                return Err(alloc::format!("attribute '{}' is unterminated", k));
            }
            let v = unescape(&String::from_utf8_lossy(&self.b[s..self.i]));
            self.i += 1;
            attrs.push((k, v));
        }

        let mut kids = Vec::new();
        let mut text = String::new();
        loop {
            if self.i >= self.b.len() {
                return Err(alloc::format!("'{}' is never closed", name));
            }
            if self.b[self.i..].starts_with(b"</") {
                if !text.trim().is_empty() {
                    kids.push(Item::Text(unescape(&text)));
                }
                self.i += 2;
                let close = self.name()?;
                self.ws();
                if self.i >= self.b.len() || self.b[self.i] != b'>' {
                    return Err(String::from("unterminated closing tag"));
                }
                self.i += 1;
                if close != name {
                    return Err(alloc::format!("'{}' closed by '{}'", name, close));
                }
                return Ok(Node { name, attrs, kids });
            }
            if self.b[self.i..].starts_with(b"<!--") {
                self.comment()?;
                continue;
            }
            if self.b[self.i..].starts_with(b"<![CDATA[") {
                self.i += 9;
                let s = self.i;
                while self.i + 2 < self.b.len() && &self.b[self.i..self.i + 3] != b"]]>" {
                    self.i += 1;
                }
                if self.i + 2 >= self.b.len() {
                    return Err(String::from("unterminated CDATA"));
                }
                text.push_str(&String::from_utf8_lossy(&self.b[s..self.i]));
                self.i += 3;
                continue;
            }
            if self.b[self.i] == b'<' {
                if !text.trim().is_empty() {
                    kids.push(Item::Text(unescape(&text)));
                }
                text = String::new();
                self.depth += 1;
                let e = self.element();
                self.depth -= 1;
                kids.push(Item::Elem(e?));
                continue;
            }
            text.push(self.b[self.i] as char);
            self.i += 1;
        }
    }
}

/// The five predefined entities and numeric references. Anything else is left
/// as written, because an unknown entity is a DTD question and this reader
/// refuses documents that carry one.
fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return String::from(s);
    }
    let mut out = String::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'&' {
            out.push(b[i] as char);
            i += 1;
            continue;
        }
        let rest = &s[i..];
        let end = match rest.find(';') {
            Some(e) if e <= 10 => e,
            _ => {
                out.push('&');
                i += 1;
                continue;
            }
        };
        let ent = &rest[1..end];
        let ch = match ent {
            "lt" => Some('<'),
            "gt" => Some('>'),
            "amp" => Some('&'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => {
                if let Some(hex) = ent.strip_prefix("#x").or_else(|| ent.strip_prefix("#X")) {
                    u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
                } else if let Some(dec) = ent.strip_prefix('#') {
                    dec.parse::<u32>().ok().and_then(char::from_u32)
                } else {
                    None
                }
            }
        };
        match ch {
            Some(c) => {
                out.push(c);
                i += end + 1;
            }
            None => {
                out.push('&');
                i += 1;
            }
        }
    }
    out
}

pub fn selftest() -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    let d = parse("<?xml version=\"1.0\"?>\n<a x=\"1\"><b>hi</b><b>there</b></a>");
    let d = match d {
        Ok(d) => d,
        Err(e) => {
            claim(&mut ok, false, &e);
            return false;
        }
    };
    claim(&mut ok, d.name == "a" && d.attr("x") == Some("1"), "a document parses with its attributes");
    claim(&mut ok, d.children("b").len() == 2, "repeated children are all reachable");
    claim(&mut ok, d.text() == "hithere", "and text gathers from the whole subtree");
    claim(&mut ok, parse("<a/>").is_ok(), "an empty element closes itself");
    claim(&mut ok, parse("<a><b></a>").is_err(), "a mismatched close is an error");
    claim(&mut ok, parse("<a>").is_err(), "and so is one that never closes");
    claim(&mut ok, parse("<a/><b/>").is_err(), "a second root is refused");
    claim(&mut ok, parse("<a x=1/>").is_err(), "an unquoted attribute is refused");
    claim(
        &mut ok,
        parse("<!DOCTYPE a [<!ENTITY x \"y\">]><a/>").is_err(),
        "a DOCTYPE with an internal subset is refused rather than half honoured",
    );
    claim(
        &mut ok,
        parse("<a>&lt;&amp;&#65;</a>").map(|n| n.text()) == Ok(String::from("<&A")),
        "entities and numeric references decode",
    );
    claim(
        &mut ok,
        parse("<a><![CDATA[<not markup>]]></a>").map(|n| n.text()) == Ok(String::from("<not markup>")),
        "CDATA is text and not markup",
    );
    claim(
        &mut ok,
        parse("<a><!-- c --><b/></a>").map(|n| n.children("b").len()) == Ok(1),
        "comments are skipped wherever they appear",
    );
    ok
}

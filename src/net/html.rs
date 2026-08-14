//! HTML, reduced to the few things a reading browser needs.
//!
//! Not a DOM. Enternet renders a page as a flat list of blocks, because that is
//! all a text browser can show: there is no CSS, no float, no table layout and
//! no script, so a tree buys nothing that a list does not. Nesting is flattened
//! as it is encountered, which is also what makes the parser tolerant of the
//! unclosed tags that most of the web is made of.
//!
//! Tolerant is the design, not a shortcut. A validating parser is the wrong
//! tool here: a page that fails to parse must still render, because the
//! alternative is a browser that shows a blank window and blames the author.
//! Every unknown tag is skipped, every unclosed one is closed by the next block
//! element, and text outside any tag is a paragraph.
//!
//! What is deliberately not here: forms, frames, images, tables as grids
//! (their cells become paragraphs), and character sets other than UTF-8 and
//! Latin-1. Each is a real feature rather than an oversight, and each would be
//! a lot of code for a browser whose output is a character grid.

use super::css;
use alloc::string::String;
use alloc::vec::Vec;

/// A run of text, optionally pointing somewhere.
#[derive(Clone)]
pub enum Span {
    Text(String),
    Link { text: String, href: String },
}

impl Span {
    pub fn text(&self) -> &str {
        match self {
            Span::Text(t) => t,
            Span::Link { text, .. } => text,
        }
    }
}

/// One line-level thing to draw.
#[derive(Clone)]
pub enum Block {
    /// `<h1>` through `<h6>`; the level is kept so the renderer can size it.
    Heading(u8, Vec<Span>),
    Para(Vec<Span>),
    /// A list item. The marker is the renderer's business, not the parser's.
    Item(Vec<Span>),
    /// `<pre>`, kept verbatim: the one place where whitespace is content.
    Pre(String),
    Rule,
}

pub struct Page {
    pub title: String,
    pub blocks: Vec<Block>,
}

// --- URLs -----------------------------------------------------------------

/// Enough of a URL to fetch it again and to resolve a link against it.
#[derive(Clone)]
pub struct Url {
    pub https: bool,
    pub host: String,
    pub port: u16,
    /// Always begins with `/`.
    pub path: String,
}

impl Url {
    pub fn text(&self) -> String {
        let mut s = String::new();
        s.push_str(if self.https { "https://" } else { "http://" });
        s.push_str(&self.host);
        let default = if self.https { 443 } else { 80 };
        if self.port != default {
            s.push(':');
            let mut n = self.port;
            let mut d = [0u8; 5];
            let mut i = 5;
            while n > 0 {
                i -= 1;
                d[i] = b'0' + (n % 10) as u8;
                n /= 10;
            }
            for c in &d[i..] {
                s.push(*c as char);
            }
        }
        s.push_str(&self.path);
        s
    }
}

/// Parse an absolute URL. A bare host is treated as https, which is the right
/// default now and would have been the wrong one when this interface was new.
pub fn parse_url(s: &str) -> Option<Url> {
    let s = s.trim();
    let (https, rest) = if let Some(r) = s.strip_prefix("https://") {
        (true, r)
    } else if let Some(r) = s.strip_prefix("http://") {
        (false, r)
    } else if s.contains("://") {
        return None; // some scheme this browser does not speak
    } else {
        (true, s)
    };
    if rest.is_empty() {
        return None;
    }
    let (auth, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    // Strip any userinfo rather than sending it: this browser has no business
    // putting credentials on the wire, and a URL that carries them is usually
    // trying to disguise its host.
    let auth = match auth.rfind('@') {
        Some(i) => &auth[i + 1..],
        None => auth,
    };
    let (host, port) = match auth.rsplit_once(':') {
        Some((h, p)) => (h, p.parse().unwrap_or(if https { 443 } else { 80 })),
        None => (auth, if https { 443 } else { 80 }),
    };
    if host.is_empty() {
        return None;
    }
    Some(Url {
        https,
        host: String::from(host),
        port,
        path: String::from(if path.is_empty() { "/" } else { path }),
    })
}

/// Resolve a link against the page it was found on.
pub fn resolve(base: &Url, href: &str) -> Option<Url> {
    let href = href.trim();
    if href.is_empty() || href.starts_with('#') {
        return None;
    }
    if href.contains("://") {
        return parse_url(href);
    }
    if let Some(r) = href.strip_prefix("//") {
        let mut s = String::from(if base.https { "https://" } else { "http://" });
        s.push_str(r);
        return parse_url(&s);
    }
    let mut u = base.clone();
    // A fragment is a position on the page, not a different page.
    let href = href.split('#').next().unwrap_or(href);
    if href.starts_with('/') {
        u.path = String::from(href);
    } else {
        // Relative to the directory, which is everything up to the last slash.
        let cut = u.path.rfind('/').map(|i| i + 1).unwrap_or(1);
        let mut p = String::from(&u.path[..cut]);
        p.push_str(href);
        u.path = normalise(&p);
    }
    if u.path.is_empty() {
        u.path = String::from("/");
    }
    Some(u)
}

/// Collapse `.` and `..` segments. A server would do this too, but a link that
/// climbs above the root should not produce a path with `..` still in it.
fn normalise(path: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            s => out.push(s),
        }
    }
    let mut s = String::new();
    for seg in out {
        s.push('/');
        s.push_str(seg);
    }
    if s.is_empty() {
        s.push('/');
    } else if path.ends_with('/') {
        s.push('/');
    }
    s
}

// --- entities -------------------------------------------------------------

/// Decode the handful of entities that actually appear in prose.
///
/// The full table is over two thousand names. These are the ones without which
/// text reads wrong, plus numeric escapes, and anything unrecognised is left
/// as written so a stray ampersand survives rather than eating the line.
fn entity(name: &str) -> Option<char> {
    Some(match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" | "#39" => '\'',
        "nbsp" | "#160" => ' ',
        "mdash" | "#8212" => '-',
        "ndash" | "#8211" => '-',
        "hellip" | "#8230" => '.',
        "copy" => 'c',
        "reg" => 'R',
        "trade" => 'T',
        "rsquo" | "#8217" | "lsquo" | "#8216" => '\'',
        "ldquo" | "#8220" | "rdquo" | "#8221" => '"',
        "middot" | "#183" => '-',
        n => {
            let d = n.strip_prefix('#')?;
            let v = if let Some(h) = d.strip_prefix('x').or_else(|| d.strip_prefix('X')) {
                u32::from_str_radix(h, 16).ok()?
            } else {
                d.parse::<u32>().ok()?
            };
            // Anything outside Latin-1 becomes a placeholder: the console font
            // has 256 glyphs and a missing one is worse than a visible gap.
            if v < 32 {
                return None;
            }
            if v > 255 {
                return Some('?');
            }
            char::from_u32(v)?
        }
    })
}

fn push_decoded(out: &mut String, s: &str) {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'&' {
            // Entities are short; a run of text with a bare & in it should not
            // scan to the end of the document looking for a semicolon.
            if let Some(end) = s[i..].find(';').filter(|e| *e <= 10) {
                if let Some(c) = entity(&s[i + 1..i + end]) {
                    out.push(c);
                    i += end + 1;
                    continue;
                }
            }
        }
        out.push(b[i] as char);
        i += 1;
    }
}

// --- the parser -----------------------------------------------------------

/// Tags after which text belongs to a new block.
fn is_break(tag: &str) -> bool {
    matches!(tag,
        "p" | "div" | "br" | "tr" | "td" | "th" | "section" | "article" | "header"
        | "footer" | "nav" | "main" | "aside" | "figure" | "figcaption" | "blockquote"
        | "form" | "table" | "tbody" | "thead" | "dl" | "dt" | "dd" | "address")
}

/// Elements with no closing tag. Hiding one by skipping to its close would
/// swallow the rest of the document, which is a far worse failure than showing
/// something the stylesheet wanted hidden.
fn is_void(tag: &str) -> bool {
    matches!(tag, "br" | "hr" | "img" | "input" | "meta" | "link" | "source"
        | "area" | "base" | "col" | "embed" | "param" | "track" | "wbr")
}

/// Tags whose entire contents are not prose and must be dropped.
fn is_opaque(tag: &str) -> bool {
    matches!(tag, "script" | "style" | "head" | "svg" | "noscript" | "template")
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn tag(&mut self) -> Option<(String, String, bool)> {
        // Returns (name, raw attributes, closing).
        if self.b.get(self.i) != Some(&b'<') {
            return None;
        }
        let start = self.i + 1;
        let mut j = start;
        while j < self.b.len() && self.b[j] != b'>' {
            // Skip quoted attribute values so a '>' inside one does not end
            // the tag early, which is how a page with an inline style breaks a
            // naive scanner.
            if self.b[j] == b'"' || self.b[j] == b'\'' {
                let q = self.b[j];
                j += 1;
                while j < self.b.len() && self.b[j] != q {
                    j += 1;
                }
            }
            j += 1;
        }
        if j >= self.b.len() {
            self.i = self.b.len();
            return None;
        }
        let raw = core::str::from_utf8(&self.b[start..j]).unwrap_or("");
        self.i = j + 1;
        let closing = raw.starts_with('/');
        let raw = raw.trim_start_matches('/');
        let name_end = raw
            .find(|c: char| c.is_ascii_whitespace())
            .unwrap_or(raw.len());
        let mut name = String::new();
        for c in raw[..name_end].chars() {
            name.push(c.to_ascii_lowercase());
        }
        Some((name, String::from(&raw[name_end..]), closing))
    }

    fn skip_to_close(&mut self, tag: &str) {
        while self.i < self.b.len() {
            if self.b[self.i] == b'<' {
                let save = self.i;
                if let Some((name, _, closing)) = self.tag() {
                    if closing && name == tag {
                        return;
                    }
                    continue;
                }
                self.i = save + 1;
            } else {
                self.i += 1;
            }
        }
    }

    fn text_until_tag(&mut self) -> &'a str {
        let start = self.i;
        while self.i < self.b.len() && self.b[self.i] != b'<' {
            self.i += 1;
        }
        core::str::from_utf8(&self.b[start..self.i]).unwrap_or("")
    }
}

/// Read one attribute out of a tag's raw text.
fn attr(raw: &str, want: &str) -> Option<String> {
    let lower: String = raw.chars().map(|c| c.to_ascii_lowercase()).collect();
    let mut from = 0;
    while let Some(rel) = lower[from..].find(want) {
        let at = from + rel;
        let before_ok = at == 0 || lower.as_bytes()[at - 1].is_ascii_whitespace();
        let after = &lower[at + want.len()..];
        let after_trim = after.trim_start();
        if before_ok && after_trim.starts_with('=') {
            let v = &raw[at + want.len()..];
            let v = v.trim_start().strip_prefix('=')?.trim_start();
            let val = if let Some(r) = v.strip_prefix('"') {
                r.split('"').next()?
            } else if let Some(r) = v.strip_prefix('\'') {
                r.split('\'').next()?
            } else {
                v.split(|c: char| c.is_ascii_whitespace()).next()?
            };
            let mut out = String::new();
            push_decoded(&mut out, val);
            return Some(out);
        }
        from = at + want.len();
    }
    None
}

/// Collapse runs of whitespace, which is what HTML means by whitespace.
fn squeeze(into: &mut String, s: &str) {
    for c in s.chars() {
        if c.is_ascii_whitespace() {
            if !into.ends_with(' ') && !into.is_empty() {
                into.push(' ');
            }
        } else {
            into.push(c);
        }
    }
}

pub fn parse(body: &[u8], base: &Url) -> Page {
    let mut p = Parser { b: body, i: 0 };
    let mut page = Page { title: String::new(), blocks: Vec::new() };
    let mut sheet_src = String::new();
    let mut sheet = css::Sheet::new();

    // Current block being accumulated.
    let mut spans: Vec<Span> = Vec::new();
    let mut buf = String::new();
    let mut heading: u8 = 0;
    let mut item = false;
    let mut link: Option<String> = None;

    macro_rules! flush_text {
        () => {
            if !buf.trim().is_empty() {
                let t = String::from(buf.trim_end_matches(' '));
                match &link {
                    Some(h) => spans.push(Span::Link { text: t, href: h.clone() }),
                    None => spans.push(Span::Text(t)),
                }
            }
            buf.clear();
        };
    }

    macro_rules! flush_block {
        () => {
            flush_text!();
            if !spans.is_empty() {
                let s = core::mem::take(&mut spans);
                page.blocks.push(if heading > 0 {
                    Block::Heading(heading, s)
                } else if item {
                    Block::Item(s)
                } else {
                    Block::Para(s)
                });
            }
            heading = 0;
            item = false;
        };
    }

    while p.i < body.len() {
        if body[p.i] != b'<' {
            let t = p.text_until_tag();
            squeeze(&mut buf, &{
                let mut d = String::new();
                push_decoded(&mut d, t);
                d
            });
            continue;
        }
        let save = p.i;
        let Some((name, raw, closing)) = p.tag() else {
            p.i = save + 1;
            continue;
        };

        if name.starts_with('!') {
            continue; // doctype or comment; the tag scanner already ate it
        }

        if !closing && name == "style" {
            // The one opaque element worth reading. Collected as it is met,
            // which is enough for a single pass because a stylesheet in the
            // head is parsed before any of the body it applies to.
            let start = p.i;
            p.skip_to_close("style");
            let end = p.i.saturating_sub(8).max(start);
            sheet_src.push_str(core::str::from_utf8(&body[start..end]).unwrap_or(""));
            sheet_src.push('\n');
            sheet = css::parse(&sheet_src);
            continue;
        }

        if !closing && is_opaque(&name) {
            if name == "head" {
                // The title lives in here and is the one thing worth keeping.
                let end = p.i;
                let mut h = Parser { b: body, i: end };
                h.skip_to_close("head");
                let head = &body[end..h.i.min(body.len())];
                page.title = title_of(head);
                // The stylesheet almost always lives in here, and the head is
                // skipped whole -- so collecting <style> only in the body meant
                // the sheet was never seen at all. The title has the same
                // problem and was already handled this way.
                sheet_src.push_str(&styles_of(head));
                sheet = css::parse(&sheet_src);
            }
            p.skip_to_close(&name);
            continue;
        }

        // Hidden by the stylesheet, or by an inline style. Skipping the whole
        // element is what removes skip-links, off-screen navigation and cookie
        // banners, which otherwise render as a site map above the article.
        if !closing && !is_void(&name) && !raw.is_empty() {
            let class = attr(&raw, "class").unwrap_or_default();
            let id = attr(&raw, "id").unwrap_or_default();
            let inline = attr(&raw, "style").unwrap_or_default();
            if css::hides_inline(&inline) || sheet.hides(&name, &class, &id) {
                flush_block!();
                p.skip_to_close(&name);
                continue;
            }
        }

        match name.as_str() {
            "a" if !closing => {
                flush_text!();
                link = attr(&raw, "href").and_then(|h| resolve(base, &h)).map(|u| u.text());
            }
            "a" if closing => {
                flush_text!();
                link = None;
            }
            "pre" if !closing => {
                flush_block!();
                let start = p.i;
                p.skip_to_close("pre");
                let inner = &body[start..p.i.min(body.len())];
                let mut s = String::new();
                let mut q = Parser { b: inner, i: 0 };
                while q.i < inner.len() {
                    if inner[q.i] == b'<' {
                        let sv = q.i;
                        if q.tag().is_none() {
                            q.i = sv + 1;
                        }
                    } else {
                        push_decoded(&mut s, q.text_until_tag());
                    }
                }
                if !s.trim().is_empty() {
                    page.blocks.push(Block::Pre(s));
                }
            }
            "hr" => {
                flush_block!();
                page.blocks.push(Block::Rule);
            }
            "li" if !closing => {
                flush_block!();
                item = true;
            }
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                flush_block!();
                if !closing {
                    heading = name.as_bytes()[1] - b'0';
                }
            }
            "title" if !closing => {
                let start = p.i;
                p.skip_to_close("title");
                if page.title.is_empty() {
                    let mut s = String::new();
                    push_decoded(&mut s, core::str::from_utf8(
                        &body[start..p.i.saturating_sub(8).max(start)]).unwrap_or(""));
                    page.title = String::from(s.trim());
                }
            }
            n if is_break(n) => {
                flush_block!();
            }
            _ => {}
        }
    }
    flush_block!();
    page
}

/// Every `<style>` block's contents, concatenated.
fn styles_of(head: &[u8]) -> String {
    let mut out = String::new();
    let mut p = Parser { b: head, i: 0 };
    while p.i < head.len() {
        if head[p.i] == b'<' {
            let save = p.i;
            match p.tag() {
                Some((n, _, false)) if n == "style" => {
                    let start = p.i;
                    p.skip_to_close("style");
                    let end = p.i.saturating_sub(8).max(start);
                    out.push_str(core::str::from_utf8(&head[start..end]).unwrap_or(""));
                    out.push('\n');
                }
                Some(_) => {}
                None => p.i = save + 1,
            }
        } else {
            p.i += 1;
        }
    }
    out
}

fn title_of(head: &[u8]) -> String {
    let mut p = Parser { b: head, i: 0 };
    while p.i < head.len() {
        if head[p.i] == b'<' {
            let save = p.i;
            match p.tag() {
                Some((n, _, false)) if n == "title" => {
                    let start = p.i;
                    p.skip_to_close("title");
                    let end = p.i.saturating_sub(8).max(start);
                    let mut s = String::new();
                    push_decoded(&mut s, core::str::from_utf8(&head[start..end]).unwrap_or(""));
                    let mut out = String::new();
                    squeeze(&mut out, &s);
                    return String::from(out.trim());
                }
                Some(_) => {}
                None => p.i = save + 1,
            }
        } else {
            p.i += 1;
        }
    }
    String::new()
}

// --- selftest -------------------------------------------------------------

/// Returns false on any failure; the caller prints the one-line summary, which
/// is how every other selftest here reports.
pub fn selftest() -> bool {
    let mut ok = true;
    let mut check = |cond: bool, what: &str| {
        if !cond {
            use crate::gfx::console::{self, LTGRAY, LTRED};
            use crate::kprintln;
            console::set_color(LTRED);
            kprintln!("  FAIL   html      {}", what);
            console::set_color(LTGRAY);
            ok = false;
        }
    };

    let base = match parse_url("https://example.com/docs/page.html") {
        Some(u) => u,
        None => return false,
    };
    check(base.host == "example.com" && base.port == 443
          && base.path == "/docs/page.html", "absolute url");
    check(parse_url("example.com").map(|u| u.https).unwrap_or(false),
          "bare host defaults to https");
    // A URL carrying userinfo is usually disguising its real host, and this
    // browser has no business putting credentials on the wire either.
    check(parse_url("https://evil@real.com/").map(|u| u.host == "real.com")
          .unwrap_or(false), "userinfo stripped");

    check(resolve(&base, "other.html").map(|u| u.path == "/docs/other.html")
          .unwrap_or(false), "relative resolves against the directory");
    check(resolve(&base, "/x").map(|u| u.path == "/x").unwrap_or(false),
          "rooted link replaces the path");
    check(resolve(&base, "../up.html").map(|u| u.path == "/up.html")
          .unwrap_or(false), "dot-dot climbs");
    check(resolve(&base, "#here").is_none(), "fragment is not a page");
    check(resolve(&base, "//other.org/a").map(|u| u.host == "other.org")
          .unwrap_or(false), "scheme-relative keeps the scheme");

    let html = b"<html><head><title>Hi &amp; bye</title><style>p{color:red}</style></head>                 <body><h1>Head</h1><p>One <a href='/two'>two</a> three</p>                 <ul><li>item</li></ul><pre>  kept  </pre><hr></body></html>";
    let page = parse(html, &base);
    check(page.title == "Hi & bye", "title decoded and style dropped");
    check(page.blocks.iter().filter(|b| matches!(b, Block::Heading(1, _))).count() == 1,
          "heading");
    check(page.blocks.iter().any(|b| match b {
        Block::Para(s) => s.iter().any(|x| matches!(x, Span::Link { href, .. }
            if href == "https://example.com/two")),
        _ => false,
    }), "link resolved to absolute");
    check(page.blocks.iter().any(|b| matches!(b, Block::Item(_))), "list item");
    check(page.blocks.iter().any(|b| matches!(b, Block::Pre(t) if t.contains("  kept"))),
          "pre keeps its spaces");
    check(page.blocks.iter().any(|b| matches!(b, Block::Rule)), "rule");

    // A '>' inside a quoted attribute must not end the tag early. This is how
    // a page with an inline style breaks a scanner that only looks for '>'.
    let t = parse(b"<p title=\"a > b\">after</p>", &base);
    check(t.blocks.iter().any(|b| matches!(b, Block::Para(s)
        if s.iter().any(|x| x.text() == "after"))), "quoted '>' does not end a tag");

    // The exact round trip a followed link makes: parsed out of an anchor,
    // stored as text by Url::text, and parsed back when it is navigated to.
    // The scheme has to survive all three.
    let a = parse(b"<p><a href=\"https://iana.org/domains/example\">Learn more</a></p>", &base);
    let mut href_seen = String::new();
    for b in &a.blocks {
        if let Block::Para(spans) = b {
            for s in spans {
                if let Span::Link { href, .. } = s {
                    href_seen = href.clone();
                }
            }
        }
    }
    check(href_seen == "https://iana.org/domains/example", "anchor href round trip");
    check(parse_url(&href_seen).map(|u| u.https).unwrap_or(false),
          "followed link keeps its scheme");

    // A stylesheet in the head must remove the element it hides, and a void
    // element must never be skipped to a closing tag it does not have.
    let styled = parse(
        b"<html><head><style>.skip{display:none}</style></head><body>          <p class=\"skip\">gone</p><p style=\"display:none\">also gone</p>          <hr><p>kept</p></body></html>", &base);
    let texts: Vec<&str> = styled.blocks.iter().filter_map(|b| match b {
        Block::Para(s) => s.first().map(|x| x.text()),
        _ => None,
    }).collect();
    check(!texts.iter().any(|t| *t == "gone"), "stylesheet hides by class");
    check(!texts.iter().any(|t| *t == "also gone"), "inline style hides");
    check(texts.iter().any(|t| *t == "kept"), "content after a void element survives");

    // Unclosed tags are the normal case on the web, not an error case.
    check(parse(b"<p>one<p>two<div>three", &base).blocks.len() == 3,
          "unclosed blocks still split");

    ok
}

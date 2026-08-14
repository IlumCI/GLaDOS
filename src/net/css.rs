//! CSS, reduced to the parts that change what a text browser shows.
//!
//! Most of CSS describes geometry and colour, and this renderer has neither: a
//! page is a column of characters. So the cascade is not implemented, nor the
//! box model, nor specificity as the standard defines it. What is implemented
//! is the one question worth asking of a stylesheet here:
//!
//!   **is this element supposed to be visible at all?**
//!
//! That single property is worth more than everything else combined. Modern
//! pages ship skip-links, off-screen navigation, cookie banners and screen
//! reader text that are all `display:none` or `visibility:hidden`, and a
//! browser that ignores the stylesheet renders every one of them as body text.
//! The result reads like a site map glued to the top of the article.
//!
//! ### What a selector means here
//!
//! Only the rightmost simple selector is kept. `nav ul li.item` becomes
//! `li.item`. That is deliberately wrong in the general case and almost always
//! right in this one: a rule hiding something is nearly never re-enabled by a
//! more specific rule in the same sheet, and matching a descendant chain needs
//! an element tree, which the HTML side does not build.
//!
//! The failure mode is therefore over-hiding, which is why only `display:none`
//! and `visibility:hidden` are honoured and everything else is parsed and
//! ignored. Losing a paragraph is worse than showing one, so nothing hides
//! unless the stylesheet says so twice over: the property must be one of those
//! two and the selector must actually match.

use alloc::string::String;
use alloc::vec::Vec;

/// The rightmost simple selector of a rule, which is all that is matched.
#[derive(Clone)]
pub struct Sel {
    /// Element name, empty for `*` or for a bare class or id selector.
    pub tag: String,
    pub class: Option<String>,
    pub id: Option<String>,
}

pub struct Sheet {
    /// Selectors that hide whatever they match.
    hide: Vec<Sel>,
}

impl Sheet {
    pub fn new() -> Sheet {
        Sheet { hide: Vec::new() }
    }

    pub fn rules(&self) -> usize {
        self.hide.len()
    }

    /// Does the stylesheet hide this element?
    ///
    /// `class` is the raw attribute, which may hold several names.
    pub fn hides(&self, tag: &str, class: &str, id: &str) -> bool {
        self.hide.iter().any(|s| {
            if !s.tag.is_empty() && !s.tag.eq_ignore_ascii_case(tag) {
                return false;
            }
            if let Some(c) = &s.class {
                if !class.split_ascii_whitespace().any(|n| n == c) {
                    return false;
                }
            }
            if let Some(i) = &s.id {
                if i != id {
                    return false;
                }
            }
            // A rule with no tag, no class and no id is `*`, and a stylesheet
            // that hides everything is a stylesheet this browser ignores.
            !(s.tag.is_empty() && s.class.is_none() && s.id.is_none())
        })
    }
}

/// True when a declaration block hides its element.
///
/// Public because an inline `style` attribute is the same question asked of a
/// single element, and it is the more common way a page hides one thing.
pub fn hides_inline(decls: &str) -> bool {
    for decl in decls.split(';') {
        let Some((prop, value)) = decl.split_once(':') else { continue };
        let prop = prop.trim().to_ascii_lowercase();
        let value = value.trim().to_ascii_lowercase();
        // `!important` and any trailing comment are noise for this question.
        let value = value.split('!').next().unwrap_or(&value).trim();
        if prop == "display" && value == "none" {
            return true;
        }
        if prop == "visibility" && value == "hidden" {
            return true;
        }
    }
    false
}

/// Strip comments. Done first so a commented-out rule cannot be parsed, and so
/// a `{` inside a comment cannot desynchronise the block scanner.
fn decomment(css: &str) -> String {
    let mut out = String::new();
    let b = css.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// Parse one selector's rightmost simple part.
///
/// Returns None for anything with a pseudo-class, an attribute selector or a
/// combinator this cannot represent. None means "do not hide", which is the
/// safe direction: an unparsed selector must never hide content.
fn selector(s: &str) -> Option<Sel> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    // Only the rightmost compound. Combinators are whitespace, >, + and ~.
    let last = s
        .rsplit(|c: char| c.is_ascii_whitespace() || c == '>' || c == '+' || c == '~')
        .next()?
        .trim();
    if last.is_empty() {
        return None;
    }
    // A pseudo-class or element is state this browser does not model, and an
    // attribute selector needs attributes the matcher is not given.
    if last.contains(':') || last.contains('[') {
        return None;
    }

    let mut sel = Sel { tag: String::new(), class: None, id: None };
    let mut cur = String::new();
    let mut kind = b't';
    for c in last.chars() {
        match c {
            '.' | '#' => {
                stash(&mut sel, kind, &cur);
                cur.clear();
                kind = if c == '.' { b'c' } else { b'i' };
            }
            '*' => {}
            _ => cur.push(c),
        }
    }
    stash(&mut sel, kind, &cur);
    if sel.tag.is_empty() && sel.class.is_none() && sel.id.is_none() {
        return None;
    }
    Some(sel)
}

fn stash(sel: &mut Sel, kind: u8, cur: &str) {
    if cur.is_empty() {
        return;
    }
    match kind {
        b'c' => sel.class = Some(String::from(cur)),
        b'i' => sel.id = Some(String::from(cur)),
        _ => sel.tag = String::from(cur),
    }
}

/// Parse a stylesheet, keeping only the rules that hide something.
pub fn parse(css: &str) -> Sheet {
    let text = decomment(css);
    let b = text.as_bytes();
    let mut sheet = Sheet::new();
    let mut i = 0;

    while i < b.len() {
        // At-rules. @media wraps more rules and its body is parsed as if the
        // query matched, because this browser has one presentation and a
        // print or dark-mode block hiding something usually means it.
        // Everything else (@import, @font-face, @keyframes) is skipped whole.
        if b[i] == b'@' {
            let start = i;
            while i < b.len() && b[i] != b'{' && b[i] != b';' {
                i += 1;
            }
            let name = text[start..i].trim().to_ascii_lowercase();
            if i < b.len() && b[i] == b';' {
                i += 1;
                continue;
            }
            if name.starts_with("@media") || name.starts_with("@supports") {
                i += 1; // step into the block and keep parsing rules
                continue;
            }
            i = skip_block(b, i);
            continue;
        }
        if b[i].is_ascii_whitespace() || b[i] == b'}' {
            i += 1;
            continue;
        }

        // A rule: selectors up to '{', then declarations up to '}'.
        let sel_start = i;
        while i < b.len() && b[i] != b'{' {
            i += 1;
        }
        if i >= b.len() {
            break;
        }
        let sels = &text[sel_start..i];
        let body_start = i + 1;
        i = skip_block(b, i);
        let body = &text[body_start..i.saturating_sub(1).max(body_start)];

        if hides_inline(body) {
            for part in sels.split(',') {
                if let Some(s) = selector(part) {
                    sheet.hide.push(s);
                }
            }
        }
    }
    sheet
}

/// Index just past the `}` matching the `{` at `i`.
fn skip_block(b: &[u8], mut i: usize) -> usize {
    let mut depth = 0;
    while i < b.len() {
        match b[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    i
}

pub fn selftest() -> bool {
    let mut ok = true;
    let mut check = |cond: bool, what: &str| {
        if !cond {
            use crate::gfx::console::{self, LTGRAY, LTRED};
            use crate::kprintln;
            console::set_color(LTRED);
            kprintln!("  FAIL   css       {}", what);
            console::set_color(LTGRAY);
            ok = false;
        }
    };

    let s = parse(".skip{display:none}  nav ul li.hidden { display : none ; }  \
                   #banner{visibility:hidden}  p{color:red}");
    check(s.hides("span", "skip", ""), "class selector hides");
    check(!s.hides("span", "skiplink", ""), "class match is whole-name, not prefix");
    check(s.hides("li", "hidden", ""), "rightmost compound of a descendant chain");
    check(!s.hides("li", "", ""), "that rule needs its class too");
    check(s.hides("div", "", "banner"), "id selector, visibility hidden");
    check(!s.hides("p", "", ""), "a colour rule hides nothing");

    // Over-hiding is the failure mode, so anything unparseable must not hide.
    let t = parse("a:hover{display:none} [hidden]{display:none} *{display:none}");
    check(!t.hides("a", "", ""), "pseudo-class is not matched");
    check(!t.hides("div", "", ""), "attribute selector is not matched");
    check(t.rules() == 0, "universal hide is ignored");

    // A comment must not be parsed, and a brace inside one must not
    // desynchronise the scanner that follows it.
    let c = parse("/* .a{display:none} { */ .b{display:none}");
    check(!c.hides("i", "a", ""), "commented rule is not applied");
    check(c.hides("i", "b", ""), "parsing resumes after a comment");

    check(hides_inline("color:red;display:none"), "inline display none");
    check(hides_inline("display: none !important"), "inline important");
    check(!hides_inline("display:block"), "inline display block shows");

    // @media wraps rules and its contents still apply; @font-face does not.
    let m = parse("@media print{.p{display:none}} @font-face{src:url(x)} .q{display:none}");
    check(m.hides("i", "p", ""), "rule inside @media");
    check(m.hides("i", "q", ""), "parsing resumes after @font-face");

    ok
}

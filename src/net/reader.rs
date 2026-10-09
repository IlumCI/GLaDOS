//! The web as text, for a reader that is not a person.
//!
//! `gfx::browse` renders a page into rows for a screen somebody is looking at.
//! The model reads its world as captured console text, so what it needs from a
//! page is the same content laid out to be *read back*: headings marked,
//! paragraphs flowed, and every link carrying the number that reaches it.
//!
//! **The link number is the whole point.** `constrain.rs` makes invalid output
//! unreachable rather than improbable, and a URL is not a set anything can
//! enumerate -- but the links on the page in front of you are. So a journey
//! starts at a name the operator declared and continues by index, and at every
//! step the set of legal next moves is finite and known. That is the same
//! shape `linux::program` has, one layer out: a closed table for the first
//! move, and the page itself for every move after.
//!
//! **What comes back is untrusted input and cannot be made otherwise.** A
//! fetched page is bytes a stranger chose, and the applet's output becomes the
//! model's observation, which is its prompt. Nothing here can stop a page
//! containing text shaped like an instruction. What it can do is never let
//! that text arrive unlabelled: `FRAME` brackets the content so the model is
//! reading a quoted document rather than finding sentences in its own context
//! with no provenance. That is a mitigation and not a fix, and the applet is
//! `mutates: true` partly for this reason -- a read-only agent cannot reach it
//! at all.

use super::html::{self, Block, Page, Span, Url};
use crate::sync::Racy;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Where the declared starting points live: `name url`, one per line.
///
/// A blob rather than a directory of files, which is the opposite of
/// `linux::program`'s choice and for a stated reason: a program *is* its
/// bytes and belongs at a path, where a site is a short string and a
/// directory of one-line files would be ceremony. Blanks and `#` comments are
/// dropped so the file can say what it is for.
///
/// Outside `/tmp` for the reason the program directory is: `fs.rs` lets a
/// guest write inside `/tmp`, so a table kept there would let anything that
/// runs at ring 3 add a host the model can then be asked to fetch.
pub const SITES: &str = "/web/sites";

/// How much of a page the model is shown.
///
/// **Bounded because the observation is a prompt.** `agent.rs` captures what
/// an applet prints and hands it back as the result of the step, so an
/// unbounded page is an unbounded prefix -- and a real article is tens of
/// kilobytes against a context measured in hundreds of tokens. A page that
/// overflows the window costs the model the goal it was pursuing, which is
/// strictly worse than a page it was told had more in it.
///
/// The links are **not** truncated with the text, deliberately. They are
/// numbered over the whole document because the index is a contract: link 7 has
/// to mean the same thing to whoever hands it out and whoever resolves it, and
/// a numbering that depended on where the text happened to stop would renumber
/// the page every time the cap moved.
const TEXT_CAP: usize = 1400;

/// What brackets fetched content, so it is never loose in the prompt.
const FRAME: (&str, &str) = ("--- begin fetched page ---", "--- end fetched page ---");

/// The links of the page last read, as absolute URL text, in document order.
///
/// `Racy` is this kernel's single-core interior mutability and the applet runs
/// on whichever task dispatched it, so this is the same footing every other
/// shell-reachable state stands on.
static LINKS: Racy<Vec<String>> = Racy::new(Vec::new());

/// Where the page last read came from, so a relative link still resolves after
/// the fetch that produced it has returned.
static HERE: Racy<Option<String>> = Racy::new(None);

/// A declared starting point.
pub struct Site {
    pub name: String,
    pub url: String,
}

/// Whether `name` is a name rather than a URL or a number.
///
/// A site name may not be all digits, because a digit is how a *link* is
/// named: `web 3` has to mean link 3 and never a site somebody called `3`.
/// Refused at declaration rather than resolved by precedence, since a table
/// that silently loses a row is worse than one that will not take it.
fn is_site_name(name: &str) -> bool {
    !name.is_empty()
        && !name.contains(char::is_whitespace)
        && !name.chars().all(|c| c.is_ascii_digit())
        && !name.chars().any(|c| c.is_control())
}

/// Every declared site, in the order the file lists them.
pub fn sites() -> Vec<Site> {
    let Some(bytes) = crate::sysbox::read_blob(SITES) else {
        return Vec::new();
    };
    let Ok(text) = core::str::from_utf8(&bytes) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, url)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        let (name, url) = (name.trim(), url.trim());
        // A row whose URL this kernel cannot parse is dropped here rather than
        // offered and refused later: the set the grammar is given has to be
        // the set that works.
        if is_site_name(name) && html::parse_url(url).is_some() {
            out.push(Site { name: name.to_string(), url: url.to_string() });
        }
    }
    out
}

/// The URL a declared name stands for.
pub fn url_of(name: &str) -> Option<String> {
    sites().into_iter().find(|s| s.name == name).map(|s| s.url)
}

/// The links of the page last read.
pub fn links() -> Vec<String> {
    unsafe { (*LINKS.get()).clone() }
}

/// Forget the current page. Used by the claims, so they cannot leave state
/// behind that makes the next run of the suite answer differently -- the
/// failure `linux::unix` recorded for exactly this shape.
pub fn forget() {
    unsafe {
        (*LINKS.get()).clear();
        *HERE.get() = None;
    }
}

/// Lay a page out as text, and remember its links.
///
/// Pure apart from the two statics it sets, which is what lets the claims
/// check the rendering without a network: a `Page` is cheap to build by hand
/// and this is the whole of what is new here.
pub fn render(page: &Page, base: &Url) -> String {
    let found = html::links_of(page, base);
    let mut out = String::new();

    if !page.title.is_empty() {
        out.push_str("# ");
        out.push_str(page.title.trim());
        out.push('\n');
    }

    // The link number is assigned over the whole document in document order,
    // which is `links_of`'s contract, so the counter walks every block even
    // once the text has stopped being emitted.
    let mut n = 0usize;
    let mut cut = false;
    for b in &page.blocks {
        let mut line = String::new();
        match b {
            Block::Rule => line.push_str("---"),
            Block::Pre(t) => {
                // Verbatim, which is what `<pre>` means, indented so it is
                // visibly not prose.
                for l in t.lines() {
                    line.push_str("    ");
                    line.push_str(l);
                    line.push('\n');
                }
                line.pop();
            }
            Block::Heading(level, spans) => {
                for _ in 0..(*level).clamp(1, 6) {
                    line.push('#');
                }
                line.push(' ');
                spans_into(spans, &mut n, &mut line);
            }
            Block::Item(spans) => {
                line.push_str("- ");
                spans_into(spans, &mut n, &mut line);
            }
            Block::Para(spans) => spans_into(spans, &mut n, &mut line),
        }
        // `Pre` is the one block whose leading whitespace is content, so it
        // keeps its indent where everything else loses the space a block
        // never opens with.
        let line = if matches!(b, Block::Pre(_)) { line.trim_end() } else { line.trim() };
        if line.is_empty() {
            continue;
        }
        if out.len() + line.len() + 1 > TEXT_CAP {
            cut = true;
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }

    if cut {
        out.push_str("(the rest of the page was not shown)\n");
    }
    if !found.is_empty() {
        out.push_str("links: 1..");
        push_num(&mut out, found.len());
        out.push('\n');
    }

    unsafe {
        *LINKS.get() = found.iter().map(|u| u.text()).collect();
        *HERE.get() = Some(base.text());
    }
    out
}

/// A block's spans, with every link carrying the number that reaches it.
///
/// `text [n]` rather than a separate list at the foot, because the model has
/// to choose a link from what the sentence around it says: a numbered list
/// divorced from its context asks it to pick a label it has no reason for.
fn spans_into(spans: &[Span], n: &mut usize, out: &mut String) {
    for s in spans {
        match s {
            Span::Text(t) => out.push_str(t),
            Span::Link { text, .. } => {
                *n += 1;
                out.push_str(text);
                out.push_str(" [");
                push_num(out, *n);
                out.push(']');
            }
        }
    }
}

fn push_num(out: &mut String, mut n: usize) {
    if n == 0 {
        out.push('0');
        return;
    }
    let mut d = [0u8; 20];
    let mut i = 20;
    while n > 0 {
        i -= 1;
        d[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    for c in &d[i..] {
        out.push(*c as char);
    }
}

/// Fetch a URL, lay it out, and say what it was.
///
/// **https only, which is `gfx::browse`'s decision and is kept rather than
/// re-argued.** The plain-text side would need a second client and a third
/// copy of response splitting -- the thing this tree deliberately removed --
/// and `https_fetch` already de-chunks a body and reports an identity verdict.
///
/// The verdict is **reported and not enforced**, which is the `https` verb's
/// bargain rather than `update::fetch`'s: a machine deciding what to boot must
/// refuse an unverified peer, and a reader is reading. But the model is told,
/// because what it does next may depend on the page and "who said this" is
/// part of the page.
pub fn open(url_text: &str) -> Result<String, String> {
    let Some(url) = html::parse_url(url_text) else {
        return Err(alloc::format!("'{}' is not a URL this kernel can parse", url_text));
    };
    if !url.https {
        return Err(String::from(
            "only https is fetched -- the plain-text client does not exist here",
        ));
    }
    let ip = match crate::net::dns::lookup(&url.host) {
        Ok(ip) => ip,
        Err(_) => return Err(alloc::format!("cannot resolve {}", url.host)),
    };
    let f = match crate::net::tls::https_fetch(ip, &url.host, url.port, &url.path, 30_000) {
        Ok(f) => f,
        Err(e) => return Err(alloc::format!("fetch failed: {}", e.name())),
    };

    let mut out = String::new();
    out.push_str(&url.text());
    out.push_str(" -- ");
    push_num(&mut out, f.status as usize);
    // Said in the observation rather than only in a log, because a model that
    // may act on a page is owed whether anybody vouched for it.
    out.push_str(if f.identity.ok() { ", verified" } else { ", NOT verified" });
    out.push('\n');

    let page = html::parse(&f.body, &url);
    out.push_str(FRAME.0);
    out.push('\n');
    out.push_str(&render(&page, &url));
    out.push_str(FRAME.1);
    out.push('\n');
    Ok(out)
}

/// Follow a link of the page last read.
pub fn follow(n: usize) -> Result<String, String> {
    let have = links();
    if n == 0 || n > have.len() {
        return Err(if have.is_empty() {
            String::from("no page has been read, so there are no links to follow")
        } else {
            let mut s = String::from("there is no link ");
            push_num(&mut s, n);
            s.push_str(" -- this page has 1..");
            push_num(&mut s, have.len());
            s
        });
    }
    open(&have[n - 1])
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    // A name may not be a number, or `web 3` would be ambiguous between a
    // link and a site -- and ambiguity in the set handed to a grammar is the
    // one thing the closed set exists to remove.
    out.push(("a site name may not be a number", !is_site_name("3")));
    out.push(("nor carry whitespace", !is_site_name("two words")));
    out.push(("nor a control character", !is_site_name("a\nb")));
    out.push(("an ordinary name is one", is_site_name("example")));

    let base = html::parse_url("https://h/d/i.html").unwrap();

    // Rendering, which is the whole of what is new. Built by hand because a
    // `Page` is cheap to build and a network is not.
    let page = Page {
        title: String::from("T"),
        blocks: alloc::vec![
            Block::Heading(2, alloc::vec![Span::Text(String::from("H"))]),
            Block::Para(alloc::vec![
                Span::Text(String::from("see ")),
                Span::Link { text: String::from("a"), href: String::from("/x") },
                Span::Text(String::from(" and ")),
                Span::Link { text: String::from("b"), href: String::from("y") },
            ]),
            Block::Item(alloc::vec![Span::Text(String::from("i"))]),
            Block::Pre(String::from("p")),
            Block::Rule,
        ],
    };
    forget();
    let text = render(&page, &base);

    out.push(("a title renders as a heading", text.starts_with("# T\n")));
    out.push(("a heading keeps its level", text.contains("\n## H\n")));
    out.push(("a list item is marked", text.contains("\n- i\n")));
    out.push(("preformatted text is indented and verbatim", text.contains("\n    p\n")));
    out.push(("a rule renders", text.contains("\n---\n")));
    // The number is beside the words that justify choosing it, which is the
    // reason the links are inline rather than listed at the foot.
    out.push((
        "a link carries the number that reaches it, in its sentence",
        text.contains("see a [1] and b [2]"),
    ));
    out.push(("and the count is stated", text.contains("links: 1..2")));

    // Resolution is `resolve`'s, so a root-relative and a directory-relative
    // href both land somewhere real -- which `parse_url` could not do and is
    // why the browser had been dropping them.
    let l = links();
    out.push((
        "a root-relative link resolves against the host",
        l.first().map(|s| s.as_str()) == Some("https://h/x"),
    ));
    out.push((
        "and a directory-relative one against the directory",
        l.get(1).map(|s| s.as_str()) == Some("https://h/d/y"),
    ));

    // Following is bounded by what the page actually offered, in both
    // directions, and says which page it is talking about.
    out.push(("link 0 is not a link", follow(0).is_err()));
    out.push(("nor one past the end", follow(3).is_err()));

    // The cap, and that it announces itself. A page cut silently would read
    // as a short page, which is the one failure a truncating reader must not
    // have.
    let long = Page {
        title: String::new(),
        blocks: (0..200)
            .map(|_| Block::Para(alloc::vec![Span::Text("x".repeat(60))]))
            .collect(),
    };
    forget();
    let big = render(&long, &base);
    out.push(("a long page is capped", big.len() < TEXT_CAP + 200));
    out.push(("and says it was cut", big.contains("not shown")));

    // And the numbering survives the cut, which is the contract that makes an
    // index safe to hand out: a link late in a truncated page is still the
    // number it would have been.
    let mut blocks: Vec<Block> = (0..200)
        .map(|_| Block::Para(alloc::vec![Span::Text("x".repeat(60))]))
        .collect();
    blocks.push(Block::Para(alloc::vec![Span::Link {
        text: String::from("last"),
        href: String::from("/z"),
    }]));
    forget();
    let _ = render(&Page { title: String::new(), blocks }, &base);
    out.push((
        "a link past the cap is still numbered and still followable",
        links().len() == 1 && links()[0] == "https://h/z",
    ));

    // The jail, read off `fs::writable` rather than restated, so a widening
    // guest-writable area fails here instead of becoming a host the model can
    // be handed.
    out.push((
        "a guest cannot write the site table",
        !crate::linux::fs::writable(SITES),
    ));

    forget();
    out
}

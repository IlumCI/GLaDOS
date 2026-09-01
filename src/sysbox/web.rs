//! The two applets that let the model read something it was not shipped with.
//!
//! Everything else in `sysbox` moves bytes that were already on this machine.
//! These two do not, and that is a different kind of act, so the reasoning for
//! it lives here rather than as one more arm in the dispatch table.
//!
//! ### Why an applet at all, when `https` already exists
//!
//! The shell has spoken HTTPS since TLS landed. What it did not have was a way
//! for the *model* to. Actions reach the model through `APPLETS` and nothing
//! else -- the decoding grammar is built from that table, so an applet absent
//! from it is not merely discouraged, it is unspellable. The machine could
//! therefore be asked to learn something and had no route to anything it did
//! not already hold. That was the gap, and it was in the routing table rather
//! than anywhere hard.
//!
//! ### What is checked, and why each check is stricter than the `https` verb's
//!
//! `tls::report` prints the identity verdict and then shows the body anyway.
//! That is right for a person reading a page, who can weigh "unverified" for
//! themselves, and wrong for a machine deciding what to believe: an
//! unauthenticated page becomes a fact in the namespace with nothing recording
//! that it was never trusted. So:
//!
//! - **HTTPS only.** A plain-http page is refused rather than downgraded.
//! - **`Identity::Verified` or nothing.** No roots loaded means no fetching,
//!   the same bargain `update::fetch` makes.
//! - **The body must be whole.** `Fetched::complete` is false when the
//!   deadline ran out mid-body, and a truncated page is not a shorter fact --
//!   it is a different one, usually with the qualifying sentence missing.
//! - **The host must be on the list.** See `SOURCES`.
//!
//! ### What none of that guarantees, said plainly
//!
//! A model that can fetch a URL and write to the namespace can carry bytes off
//! this machine, because the URL is a string and the string is influenced by
//! whatever it has already read. No check inside this file can prevent that;
//! `SOURCES` bounds *where* it can go, and `Trust::Online` bounds *when* it may
//! go anywhere at all. Those are the two real controls and they are both
//! outside the fetch itself.
//!
//! ### Why the allowlist is compiled in
//!
//! It was going to live at `/ai/sources` until the obvious hole surfaced: the
//! model has `write`, so a namespace-resident allowlist is one the model can
//! add to. A gate the gated party can edit is decoration. The list is a
//! constant, changing it is a rebuild, and that follows `UPDATE_KEY` -- where
//! adopting a signer being itself a kernel change is the point rather than an
//! inconvenience. `net allow` adds a host for one boot, from the shell only,
//! because the operator is not the party being gated.

use crate::gfx::console::{self, LTCYAN, LTGRAY, LTGREEN, LTRED};
use crate::kprintln;
use crate::net::{dns, html, tls};
use crate::sync::Racy;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// How long a fetch may take before it is abandoned.
///
/// Fifteen seconds against `update::fetch`'s three hundred, and the difference
/// is the caller: an update is a 2.8 MB image somebody asked for and will wait
/// on, while this runs inside a decode loop that is holding the engine. A
/// minute here is a minute the shell answers "another task holds it".
const TIMEOUT_MS: u64 = 15_000;

/// The most text `fetch` will put in front of the model.
///
/// Four kilobytes is roughly a thousand tokens, which is a sixth of the trained
/// window at 512 and a real fraction of it at any size. A page longer than this
/// is truncated and *says* it was, because a model that cannot tell a short
/// page from a cut one will confidently answer from the half it got.
const MAX_SHOW: usize = 4 * 1024;

/// The most text `save` will put in the namespace.
///
/// Larger, because nothing here passes through the context window -- the point
/// of `save` is that the bytes land where `find` and `cat` can reach them later
/// for the price of a namespace read rather than a decode.
const MAX_SAVE: usize = 256 * 1024;

/// Where the machine is allowed to read from.
///
/// Narrow on purpose, and narrow in a particular direction: these are places
/// that serve *reference material* over a format this kernel can already parse
/// and check. That rules out the open web, which is the point -- the value of
/// reading is in what is read, and a machine that can reach anything will
/// mostly reach noise.
///
/// Every entry here serves plain text, HTML this tree's `net::html` already
/// renders, or XML and JSON that `fmt::xml` and `json` already read. Nothing
/// here needs a parser that does not exist.
///
/// **PDFs are deliberately absent.** Research papers mostly travel as PDF and
/// there is no PDF reader in this tree; writing one that is merely approximate
/// would put text in the namespace that is subtly not what the paper says,
/// which is the worst available outcome for a machine that will later be
/// trained on it. arXiv's abstract API returns XML and Gutenberg serves plain
/// text, so the two sources that matter most are reachable without one.
const SOURCES: &[&str] = &[
    // Abstracts and metadata as Atom XML, which `fmt::xml` reads.
    "export.arxiv.org",
    // Article extracts as JSON, and the REST summary endpoint as well.
    "en.wikipedia.org",
    // Public-domain books as plain UTF-8.
    "www.gutenberg.org",
    "gutenberg.org",
    // Dataset and model cards, served as HTML.
    "huggingface.co",
    // RFCs and internet drafts, plain text.
    "www.rfc-editor.org",
];

/// Hosts the operator added by hand this boot.
///
/// RAM only and shell only. Not persisted, because a persisted allowlist is a
/// file, and every file in this system is one the model can write.
static EXTRA: Racy<Option<Vec<String>>> = Racy::new(None);

/// The decision, as a pure function of the host and the operator's additions.
///
/// Split out from `allowed` for the reason `update::decide` is a pure
/// function: it is the part worth asserting, and asserting it must not require
/// putting the machine in the state being tested. The first version of the
/// self-test below called `allow` to check that `allow` works, which widened a
/// security allowlist at every boot and then failed the second time `diag` ran
/// it -- a test that is not re-runnable and leaves residue, on the one table in
/// this module that decides who the machine will talk to.
///
/// Compares the whole host and never a suffix. Suffix matching is how an
/// allowlist for `gutenberg.org` also admits `gutenberg.org.evil.example`, and
/// this list is small enough that spelling out both `www.` and the bare form
/// costs less than the rule that would have collapsed them.
pub fn allowed_in(host: &str, extra: &[String]) -> bool {
    let h = host.to_ascii_lowercase();
    SOURCES.iter().any(|s| *s == h) || extra.iter().any(|s| *s == h)
}

/// Whether `host` may be fetched from, against the live list.
pub fn allowed(host: &str) -> bool {
    unsafe {
        match (*EXTRA.get()).as_ref() {
            Some(v) => allowed_in(host, v),
            None => allowed_in(host, &[]),
        }
    }
}

/// Add a host for the rest of this boot. Shell only -- see the module comment.
pub fn allow(host: &str) {
    let h = host.trim().to_ascii_lowercase();
    if h.is_empty() {
        return;
    }
    unsafe {
        let slot = &mut *EXTRA.get();
        let v = slot.get_or_insert_with(Vec::new);
        if !v.iter().any(|s| *s == h) {
            v.push(h);
        }
    }
}

/// Every host that may be fetched from, compiled-in first.
pub fn sources() -> Vec<String> {
    let mut v: Vec<String> = SOURCES.iter().map(|s| s.to_string()).collect();
    unsafe {
        if let Some(extra) = (*EXTRA.get()).as_ref() {
            v.extend(extra.iter().cloned());
        }
    }
    v
}

// --- sources -------------------------------------------------------------

/// A place to read from, and the URL shape that reaches its content.
///
/// This exists because the first working version of `fetch` took a URL, and a
/// URL is the wrong argument for the caller this applet has. Two reasons, and
/// the second is the one that decided it:
///
/// **The model cannot spell one.** A 0.6B writing
/// `https://en.wikipedia.org/api/rest_v1/page/summary/Ribosome` correctly,
/// token by token, is not something to build on. It can write `ribosome`.
///
/// **And the obvious URL is the wrong one anyway.** Fetching the *article*
/// page and rendering it gave four kilobytes of "Jump to content / Main menu /
/// move to sidebar" and reached no prose at all -- measured, not feared. The
/// REST summary endpoint for the same topic returns four hundred words of
/// actual content. Somebody has to know that, and it should not be the model.
///
/// So the model names a source and a topic, and the URL is built here. The
/// side effect is a much stronger guarantee than the allowlist gives: with a
/// template there is no token sequence that names a host at all, so the
/// allowlist is left guarding only `url`, which is the operator's escape
/// hatch.
struct Source {
    /// What the model writes.
    name: &'static str,
    /// Prefix and suffix around the percent-encoded topic.
    prefix: &'static str,
    suffix: &'static str,
    /// What a topic means here, for the usage line.
    takes: &'static str,
}

const KNOWN: &[Source] = &[
    // The summary endpoint, not the article. See `Source`.
    Source {
        name: "wiki",
        prefix: "https://en.wikipedia.org/api/rest_v1/page/summary/",
        suffix: "",
        takes: "an article title",
    },
    // Atom, three results. `max_results` is small because this is read into a
    // context window, not into a database.
    Source {
        name: "arxiv",
        prefix: "https://export.arxiv.org/api/query?max_results=3&search_query=all:",
        suffix: "",
        takes: "words to search abstracts for",
    },
    Source {
        name: "rfc",
        prefix: "https://www.rfc-editor.org/rfc/rfc",
        suffix: ".txt",
        takes: "an RFC number",
    },
    Source {
        name: "book",
        prefix: "https://www.gutenberg.org/cache/epub/",
        suffix: "/pg0.txt",
        takes: "a Gutenberg ebook number",
    },
];

/// Turn `<source> <topic>` into a URL, or say why it cannot be one.
///
/// `url` is the escape hatch and deliberately not in `KNOWN`: it takes a
/// literal address and is the only spelling that reaches the allowlist, so
/// keeping it out of the table keeps the table meaning "a shape that is always
/// safe to build".
fn url_for(source: &str, topic: &str) -> Result<String, Failed> {
    if source == "url" {
        return Ok(String::from(topic.trim()));
    }
    let Some(src) = KNOWN.iter().find(|s| s.name == source) else {
        return Err(Failed::NoSource(String::from(source)));
    };
    if topic.trim().is_empty() {
        return Err(Failed::NoTopic(src.name, src.takes));
    }
    let mut u = String::from(src.prefix);
    u.push_str(&encode(topic.trim()));
    // Gutenberg repeats the number, which no prefix/suffix pair can express,
    // so the one source that needs it substitutes rather than appends. A
    // second template field for one case would be worse than this line.
    if src.name == "book" {
        u.push_str("/pg");
        u.push_str(&encode(topic.trim()));
        u.push_str(".txt");
    } else {
        u.push_str(src.suffix);
    }
    Ok(u)
}

/// Percent-encode everything that is not unreserved.
///
/// An allowlist of safe bytes rather than a denylist of unsafe ones, for the
/// reason `aiksi::BUILTINS` gives about the same choice: the character somebody
/// forgets is the one that matters. Here the ones that matter are `?`, `&` and
/// `#`, each of which would let a topic bolt a query parameter or a fragment
/// onto a URL this function is supposed to be in charge of.
fn encode(topic: &str) -> String {
    let mut out = String::new();
    for b in topic.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push_str("%20"),
            _ => {
                const HEX: &[u8; 16] = b"0123456789ABCDEF";
                out.push('%');
                out.push(HEX[(b >> 4) as usize] as char);
                out.push(HEX[(b & 15) as usize] as char);
            }
        }
    }
    out
}

/// Where `save` puts things, derived rather than asked for.
///
/// The model does not choose a path. It could be made to -- `write` takes one
/// -- but a filename is another string to spell and another chance to spell it
/// two different ways for one topic, and then nothing finds the second copy.
/// One rule means `ls /ai/read` is the reading list, and a topic read twice
/// overwrites rather than accumulating near-duplicates.
pub fn slug_for(source: &str, topic: &str) -> String {
    let mut s = String::from("/ai/read/");
    s.push_str(source);
    s.push('-');
    let mut last_dash = true;
    for c in topic.chars() {
        if c.is_ascii_alphanumeric() {
            for l in c.to_lowercase() {
                s.push(l);
            }
            last_dash = false;
        } else if !last_dash {
            s.push('-');
            last_dash = true;
        }
    }
    // A topic of nothing but punctuation would end at "wiki-", which is a
    // directory name pretending to be a file.
    if s.ends_with('-') {
        s.push_str("item");
    }
    s
}

/// The names a model may write, for the usage line and the self-test.
pub fn source_names() -> Vec<String> {
    let mut v: Vec<String> = KNOWN.iter().map(|s| String::from(s.name)).collect();
    v.push(String::from("url"));
    v
}

/// What went wrong, in words a model can act on.
///
/// Deliberately not a `tls::Error` passed through. Half of these are refusals
/// this module made rather than failures the network reported, and folding the
/// two together would tell the model "could not connect" when the truth was
/// "you are not allowed to read that".
pub enum Failed {
    NoSource(String),
    NotWritten(String),
    NotText(usize),
    NoTopic(&'static str, &'static str),
    BadUrl,
    NotHttps,
    NotAllowed(String),
    Dns(&'static str),
    Tls(&'static str),
    Unverified,
    Truncated,
    Status(u16),
}

impl Failed {
    pub fn say(&self) -> String {
        match self {
            Failed::NoSource(n) => format!(
                "'{}' is not a source I read from -- try one of: {}",
                n,
                source_names().join(", ")
            ),
            Failed::NoTopic(n, takes) => format!("{} wants {}", n, takes),
            Failed::NotWritten(p) => format!("could not write {}", p),
            Failed::NotText(n) => format!(
                "the {} B that came back are not text I can read -- probably compressed",
                n
            ),
            Failed::BadUrl => "that is not a URL I can parse".to_string(),
            Failed::NotHttps => "http is refused; use https".to_string(),
            Failed::NotAllowed(h) => format!(
                "{} is not a source this machine reads from -- 'net sources' lists them",
                h
            ),
            Failed::Dns(e) => format!("could not resolve the host: {}", e),
            Failed::Tls(e) => format!("the connection failed: {}", e),
            Failed::Unverified => {
                "the server's identity did not verify, so nothing was read".to_string()
            }
            Failed::Truncated => {
                "the page arrived incomplete, so nothing was read".to_string()
            }
            Failed::Status(s) => format!("the server answered {}", s),
        }
    }
}

/// A page, reduced to what is worth keeping.
pub struct Fetched {
    pub url: String,
    pub title: String,
    pub text: String,
    /// Bytes on the wire, before any of the reduction below.
    pub raw: usize,
    /// True when `text` was cut at a cap rather than ending where the page did.
    pub cut: bool,
}

/// Fetch and reduce one page. The whole of the network path, in one function so
/// there is one place every check happens.
pub fn get(url: &str, cap: usize) -> Result<Fetched, Failed> {
    let u = html::parse_url(url).ok_or(Failed::BadUrl)?;
    if !u.https {
        return Err(Failed::NotHttps);
    }
    if !allowed(&u.host) {
        return Err(Failed::NotAllowed(u.host.clone()));
    }
    let ip = dns::lookup(&u.host).map_err(|e| Failed::Dns(e.name()))?;
    let got = tls::https_fetch(ip, &u.host, u.port, &u.path, TIMEOUT_MS)
        .map_err(|e| Failed::Tls(e.name()))?;

    // Order matters. Identity first, because a body from an unverified peer is
    // not evidence of anything including its own status line.
    if !got.identity.ok() {
        return Err(Failed::Unverified);
    }
    if !got.complete {
        return Err(Failed::Truncated);
    }
    if got.status != 200 {
        return Err(Failed::Status(got.status));
    }

    // Only HTML goes through the HTML parser, and that is not a tidiness
    // point. Every other source on the list serves something already shaped
    // like content -- arXiv answers Atom XML, Wikipedia's REST endpoints
    // answer JSON, the RFC editor and Gutenberg answer plain text -- and
    // running any of them through a tag parser produces a page of text that
    // survived being read as markup, which is worse than useless because it
    // still looks like prose.
    let (text, cut, title) = if is_html(&got.headers) {
        let page = html::parse(&got.body, &u);
        let (t, c) = flatten(&page, cap);
        (t, c, page.title.clone())
    } else {
        // Refused rather than rendered as nothing. `unwrap_or("")` here turned
        // a body this kernel cannot decode into an empty page, which is what
        // a fetch that silently failed also looks like -- and that is exactly
        // how it presented: `fetch rfc 8446` printed its URL, no text, and not
        // even the "cut" line, three times in a row, while `save` of the same
        // URL wrote 262 KB. There is no gzip reader in this tree, so a
        // compressed body is a real outcome and has to have a real name.
        let Ok(raw) = core::str::from_utf8(&got.body) else {
            return Err(Failed::NotText(got.body.len()));
        };
        // A JSON answer from one of the templated sources carries its prose in
        // a named field surrounded by thumbnails, revision ids and four copies
        // of the page URL. Handing all of that to the model spends most of the
        // budget on metadata.
        let reduced = json_prose(raw).unwrap_or_else(|| String::from(raw));
        let cut = reduced.len() > cap;
        let end = char_boundary(&reduced, cap);
        (String::from(&reduced[..end]), cut, String::new())
    };
    Ok(Fetched {
        url: u.text(),
        title,
        text,
        raw: got.body.len(),
        cut,
    })
}

/// Pull the prose out of a JSON answer, when it is a shape we put in `KNOWN`.
///
/// Only the fields this module's own templates produce, and it falls back to
/// the whole document rather than to nothing: a source that changes its shape
/// should degrade to "here is the JSON", which is ugly and true, instead of to
/// silence, which is neither.
fn json_prose(body: &str) -> Option<String> {
    let j = crate::json::Json::parse(body)?;
    let mut out = String::new();
    if let Some(t) = j.get("title").and_then(|v| v.as_str()) {
        out.push_str(t);
        out.push('\n');
        out.push('\n');
    }
    if let Some(d) = j.get("description").and_then(|v| v.as_str()) {
        out.push_str(d);
        out.push('\n');
        out.push('\n');
    }
    let e = j.get("extract").and_then(|v| v.as_str())?;
    out.push_str(e);
    out.push('\n');
    Some(out)
}

/// Whether the response said it was HTML.
///
/// The head arrives already lower-cased, so this is a substring test and not a
/// parse. Anything that does not say `text/html` is treated as content, which
/// is the safe default here: a JSON body read as text is still the JSON, while
/// an HTML body read as text is a page of tags.
fn is_html(head: &str) -> bool {
    // By line and by prefix, rather than searching the whole head for the
    // string. A substring search matches inside another header's *value* --
    // a `link:` or a `content-security-policy:` naming a type is enough --
    // and getting this wrong sends plain text through a tag parser, which
    // does not fail, it just quietly returns something else.
    head.lines()
        .filter_map(|l| l.trim_start().strip_prefix("content-type:"))
        .next()
        .map(|v| v.contains("text/html") || v.contains("application/xhtml"))
        // No content-type at all. Guessing HTML would put tags in front of the
        // model; guessing text at worst shows it markup it can see is markup.
        .unwrap_or(false)
}

/// The largest index at or below `n` that does not split a UTF-8 character.
///
/// `&s[..n]` panics on a boundary rather than returning a mangled string, and
/// this tree has paid for that mistake before -- `theme::head_chars` exists
/// because a byte count was used as a column count and truncation panicked
/// outright.
fn char_boundary(s: &str, n: usize) -> usize {
    if n >= s.len() {
        return s.len();
    }
    let mut i = n;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Render a parsed page as the prose inside it.
///
/// The model reading tags spends its context on markup, and `net::html` has
/// already done the work of deciding what is a heading and what is a
/// paragraph -- this is that decision written out as text rather than drawn.
///
/// Links keep their text and lose their href. A model cannot follow a link
/// without `fetch`ing it, `fetch` refuses anything off `SOURCES`, and a page of
/// bare URLs is a page of tokens spent on strings nothing will use.
fn flatten(page: &html::Page, cap: usize) -> (String, bool) {
    let mut s = String::new();
    let mut cut = false;
    // Takes as much of the line as is left rather than refusing a line that
    // will not fit whole. The all-or-nothing version was wrong in a way that
    // looked like a network failure: `fetch rfc 8446` printed its URL and
    // nothing else, because the first block was longer than the 4 KB display
    // cap, was refused entire, and set `cut` -- so a 700 KB document rendered
    // as a blank page while the same fetch under `save`'s larger cap wrote
    // 262 KB. Showing the first part of a long paragraph is what a reader
    // wanted; showing nothing is indistinguishable from a broken fetch.
    let mut push = |line: &str, s: &mut String, cut: &mut bool| {
        if *cut {
            return;
        }
        let room = cap.saturating_sub(s.len() + 1);
        if room == 0 {
            *cut = true;
            return;
        }
        if line.len() > room {
            s.push_str(&line[..char_boundary(line, room)]);
            s.push('\n');
            *cut = true;
            return;
        }
        s.push_str(line);
        s.push('\n');
    };
    for b in &page.blocks {
        match b {
            html::Block::Heading(_, spans) => {
                let t = join(spans);
                if !t.is_empty() {
                    push("", &mut s, &mut cut);
                    push(&t, &mut s, &mut cut);
                }
            }
            html::Block::Para(spans) => {
                let t = join(spans);
                if !t.is_empty() {
                    push(&t, &mut s, &mut cut);
                }
            }
            // A list item whose whole content is links is navigation, not
            // content. That is what a site menu, a "see also" block and a
            // reference list all are, and on a real article they are most of
            // the markup: a Wikipedia page fetched without this rule spent the
            // entire 4 KB budget on "Main page / Contents / Current events"
            // and reached no prose at all.
            //
            // The rule is structural rather than a list of known menu words,
            // so it does not need updating per site. What it costs is a
            // genuine list of links somebody wanted -- an index page reads as
            // empty -- and that is the right side of the trade for a reader
            // that is after the text.
            html::Block::Item(spans) => {
                let all_links = !spans.is_empty()
                    && spans
                        .iter()
                        .all(|sp| matches!(sp, html::Span::Link { .. }) || sp.text().trim().is_empty());
                if !all_links {
                    let t = join(spans);
                    if !t.is_empty() {
                        push(&format!("- {}", t), &mut s, &mut cut);
                    }
                }
            }
            // `<pre>` is the one place whitespace is content, so it goes
            // through line by line rather than being collapsed like prose.
            html::Block::Pre(t) => {
                for line in t.lines() {
                    push(line, &mut s, &mut cut);
                }
            }
            html::Block::Rule => push("---", &mut s, &mut cut),
        }
    }
    (s, cut)
}

fn join(spans: &[html::Span]) -> String {
    let mut s = String::new();
    for sp in spans {
        s.push_str(sp.text());
    }
    s.trim().to_string()
}

// --- the applets ---------------------------------------------------------

/// `fetch <source> <topic>` -- read something and show it.
pub fn cmd_fetch(rest: &str) {
    let (source, topic) = split_two(rest);
    if source.is_empty() {
        usage("fetch <source> <topic>");
        return;
    }
    let url = match url_for(source, topic) {
        Ok(u) => u,
        Err(e) => return say(e),
    };
    match get(&url, MAX_SHOW) {
        Err(e) => say(e),
        Ok(f) => {
            console::set_color(LTCYAN);
            kprintln!("  {}", f.url);
            if !f.title.is_empty() {
                kprintln!("  {}", f.title);
            }
            console::set_color(LTGRAY);
            if f.text.trim().is_empty() {
                kprintln!("  (the document had no text in it -- {} B on the wire)", f.raw);
            }
            for line in f.text.lines() {
                kprintln!("  {}", line);
            }
            if f.cut {
                console::set_color(LTCYAN);
                kprintln!(
                    "  ... cut at {} of {} B; 'save' keeps the whole thing",
                    MAX_SHOW, f.raw
                );
                console::set_color(LTGRAY);
            }
        }
    }
}

fn split_two(rest: &str) -> (&str, &str) {
    let t = rest.trim();
    match t.split_once(' ') {
        Some((a, b)) => (a, b.trim()),
        None => (t, ""),
    }
}

fn usage(line: &str) {
    console::set_color(LTRED);
    kprintln!("  usage: {}", line);
    kprintln!("  sources: {}", source_names().join(", "));
    console::set_color(LTGRAY);
}

fn say(e: Failed) {
    console::set_color(LTRED);
    kprintln!("  {}", e.say());
    console::set_color(LTGRAY);
}

/// Read something and keep it, without printing anything.
///
/// Split out from `cmd_save` because there are two callers and only one of
/// them is a person. `curiosity::study_once` runs this on the machine's own
/// initiative and needs the path back to record that the topic was studied;
/// routing the nightly study step through the applet dispatcher instead would
/// have meant asking a 0.6B to spell `save` correctly in order for the
/// frontier to advance, and a syllabus that stalls because a decode went to
/// `stat` is not a syllabus.
///
/// Returns (path, bytes written, the URL it came from).
pub fn save_to(source: &str, topic: &str) -> Result<(String, usize, String), Failed> {
    let url = url_for(source, topic)?;
    let path = slug_for(source, if source == "url" { &url } else { topic });
    let f = get(&url, MAX_SAVE)?;
    // The provenance travels with the text, in the file, because a page in the
    // namespace with no URL on it is a fact with no source, and this system's
    // whole discipline is that a figure without its conditions is not a
    // measurement.
    let mut body = String::new();
    body.push_str("# ");
    body.push_str(&f.url);
    body.push('\n');
    if !f.title.is_empty() {
        body.push_str("# ");
        body.push_str(&f.title);
        body.push('\n');
    }
    if f.cut {
        body.push_str("# truncated at the fetch cap; this is not the whole document\n");
    }
    body.push('\n');
    body.push_str(&f.text);
    let n = body.len();
    if !super::write_blob(&path, body.into_bytes()) {
        return Err(Failed::NotWritten(path));
    }
    Ok((path, n, f.url))
}

/// `save <source> <topic>` -- read something and keep it.
///
/// The applet that makes reading affordable. `fetch` puts a page in the context
/// window, which costs tokens every time it is consulted; this puts it in the
/// namespace, where `find` and `cat` reach it for the price of a read and the
/// snapshot machinery keeps it across boots.
pub fn cmd_save(rest: &str) {
    let (source, topic) = split_two(rest);
    if source.is_empty() {
        usage("save <source> <topic>");
        return;
    }
    match save_to(source, topic) {
        Err(e) => say(e),
        Ok((path, n, url)) => {
            console::set_color(LTGREEN);
            kprintln!("  {}  {} B  from {}", path, n, url);
            console::set_color(LTGRAY);
        }
    }
}

/// Boot self-test. Nine claims, none of which touch the network.
///
/// What can be checked without a server is the gate, and the gate is the part
/// worth checking: every network failure here is a refusal this module decided
/// on, and a refusal that stops working is silent in a way a connection error
/// never is.
pub fn selftest() -> bool {
    let mut ok = true;
    let mut claim = |what: &str, cond: bool| {
        if !cond {
            crate::kprintln!("    FAIL: {}", what);
            ok = false;
        }
    };

    // Everything below runs against `allowed_in` with a list this function
    // owns, so the suite touches no state and says the same thing however
    // many times `diag` runs it.
    let none: [String; 0] = [];
    let extra = [String::from("added.example")];

    claim(
        "a compiled-in source is allowed",
        allowed_in("export.arxiv.org", &none),
    );
    claim(
        "the comparison ignores case",
        allowed_in("EXPORT.ArXiv.ORG", &none),
    );
    claim("an unlisted host is refused", !allowed_in("example.com", &none));
    // The one that matters: a suffix match here would admit anybody who can
    // register a domain ending in a listed name.
    claim(
        "a host merely ending in a listed one is refused",
        !allowed_in("gutenberg.org.evil.example", &none),
    );
    claim(
        "a host merely containing a listed one is refused",
        !allowed_in("notgutenberg.org", &none),
    );
    // The operator's door opens, and only for what was put through it.
    claim(
        "an operator addition is honoured",
        allowed_in("added.example", &extra) && !allowed_in("added.example", &none),
    );
    claim(
        "an addition does not admit its neighbours",
        !allowed_in("other.example", &extra),
    );

    // The content-type split. Getting this backwards is silent: a JSON body
    // read as HTML comes back as a plausible-looking page of nothing.
    claim(
        "an html response is parsed as html",
        is_html("http/1.1 200 ok\r\ncontent-type: text/html; charset=utf-8\r\n\r\n"),
    );
    claim(
        "a json response is not",
        !is_html("http/1.1 200 ok\r\ncontent-type: application/json\r\n\r\n"),
    );
    claim(
        "an xml response is not",
        !is_html("http/1.1 200 ok\r\ncontent-type: application/atom+xml\r\n\r\n"),
    );
    claim(
        "a response with no content-type is not guessed as html",
        !is_html("http/1.1 200 ok\r\n\r\n"),
    );

    // Truncation must not split a character. `&s[..n]` panics rather than
    // mangling, so this is a crash and not a cosmetic fault -- the same
    // mistake `theme::head_chars` exists to prevent on the drawing side.
    let multi = "aaaa\u{00e9}bbbb";
    claim(
        "a cut lands on a character boundary",
        multi.is_char_boundary(char_boundary(multi, 5)),
    );
    claim(
        "a cap past the end returns the whole string",
        char_boundary(multi, 9999) == multi.len(),
    );

    // --- the templates -----------------------------------------------------
    //
    // These are what the model actually reaches, so they are what has to hold.
    // With a template there is no token sequence naming a host, and these
    // claims are the reason that sentence is true rather than hoped for.
    claim(
        "wiki builds the summary endpoint and not the article",
        url_for("wiki", "Ribosome").unwrap_or_default()
            == "https://en.wikipedia.org/api/rest_v1/page/summary/Ribosome",
    );
    claim(
        "gutenberg repeats the number, which prefix and suffix alone cannot",
        url_for("book", "1342").unwrap_or_default()
            == "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    );
    claim(
        "an unknown source is refused rather than guessed at",
        matches!(url_for("evil", "x"), Err(Failed::NoSource(_))),
    );
    claim(
        "a source with no topic is refused",
        matches!(url_for("wiki", "   "), Err(Failed::NoTopic(_, _))),
    );

    // The injection cases. A topic reaches a URL this module is supposed to be
    // in charge of, so anything in it that a URL treats as structure has to
    // stop being structure on the way in. `?` and `&` bolt on parameters, `#`
    // truncates, and `/` walks the path -- which for the wiki template would
    // reach any endpoint on the host.
    let q = url_for("wiki", "a?b&c#d").unwrap_or_default();
    claim(
        "a topic cannot introduce a query, a parameter or a fragment",
        !q.contains('?') && !q.contains('&') && !q.contains('#'),
    );
    let trav = url_for("wiki", "../../w/api.php").unwrap_or_default();
    claim(
        "a topic cannot walk out of the path it was given",
        !trav.contains("../") && trav.starts_with("https://en.wikipedia.org/api/rest_v1/page/summary/"),
    );
    claim(
        "a space becomes an escape rather than ending the request line",
        !url_for("wiki", "green fluorescent protein").unwrap_or_default().contains(' '),
    );

    // Derived paths. Two spellings of one topic must not become two files.
    claim(
        "a saved topic lands somewhere predictable",
        slug_for("wiki", "Green Fluorescent Protein") == "/ai/read/wiki-green-fluorescent-protein",
    );
    claim(
        "punctuation does not produce a trailing separator",
        !slug_for("wiki", "why?!").ends_with('-'),
    );
    claim(
        "every source name is one word, since the model writes it",
        source_names().iter().all(|n| !n.contains(' ') && !n.is_empty()),
    );

    // A header whose *value* mentions a type must not decide the parser.
    claim(
        "content-type is read from its own header and not from any other",
        !is_html("http/1.1 200 ok\r\ncontent-security-policy: text/html\r\ncontent-type: text/plain\r\n\r\n"),
    );

    // The blank-page bug. One block longer than the cap has to come back as
    // its first part, because "nothing" is what a failed fetch looks like.
    let long = crate::net::html::Page {
        title: alloc::string::String::new(),
        blocks: alloc::vec![crate::net::html::Block::Para(alloc::vec![
            crate::net::html::Span::Text(core::iter::repeat('x').take(9000).collect())
        ])],
    };
    let (t, c) = flatten(&long, 512);
    claim("a block longer than the cap is truncated, not dropped", !t.is_empty());
    claim("and it says it was cut", c);
    claim("and it respects the cap", t.len() <= 512);

    // Refusals that happen before any socket is opened, checked by asking for
    // them: `get` resolves nothing until the URL and the host have passed.
    claim(
        "plain http is refused",
        matches!(get("http://en.wikipedia.org/", 64), Err(Failed::NotHttps)),
    );
    claim(
        "an unlisted host is refused before it is resolved",
        matches!(get("https://example.com/", 64), Err(Failed::NotAllowed(_))),
    );
    claim(
        "a scheme this kernel does not speak is refused",
        matches!(get("ftp://en.wikipedia.org/", 64), Err(Failed::BadUrl)),
    );
    ok
}

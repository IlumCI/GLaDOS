//! Enternet: the browser.
//!
//! A reading browser in the shape the early ones had, because that is the shape
//! this machine can actually support. There is no CSS, no script, no images and
//! no layout engine; a page is a column of text with links in it, and links are
//! chosen with the keyboard because there is no mouse.
//!
//! ### Wrapping happens at draw time, not at fetch time
//!
//! The obvious design lays the page out when it arrives and keeps the rows.
//! Then the window is resized and the rows are wrong, and the bug is not that
//! the text looks bad, it is that the scroll position and the link under the
//! cursor now refer to a layout that no longer exists. Re-wrapping every frame
//! costs a few hundred string pushes on a page and removes the entire class of
//! problem. Link numbering is assigned while walking the *blocks*, so it does
//! not depend on the wrap at all.
//!
//! ### What it will not do
//!
//! Forms, cookies, redirects, and anything that is not `text/html`. Redirects
//! in particular are a real gap: a site that answers 301 shows its redirect
//! notice rather than following it, and the status line says so instead of
//! pretending the page is empty.

use super::theme::{self, Rect};
use super::Framebuffer;
use crate::net::html::{self, Block, Page, Span, Url};
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::Cell;

/// One laid-out line. `link` is an index into `links`, not into the page.
struct Row {
    text: String,
    link: Option<usize>,
    /// 0 body, 1..=6 heading level, 7 preformatted, 8 rule.
    kind: u8,
}

pub struct Browser {
    page: Option<Page>,
    /// Link targets in document order. Stable across re-wraps.
    links: Vec<Url>,
    here: Option<Url>,
    history: Vec<Url>,
    scroll: usize,
    sel: usize,
    status: String,
    addr: String,
    /// True while the address bar has the keyboard.
    editing: bool,
    /// Written by `draw_in` so the key handler can clamp against a real
    /// layout rather than guessing at one.
    rows_seen: Cell<usize>,
    page_rows: Cell<usize>,
    /// Redirects followed for the current navigation.
    hops: u8,
}

const HOME: &str = "https://example.com/";

impl Browser {
    pub fn new() -> Browser {
        Browser {
            page: None,
            links: Vec::new(),
            here: None,
            history: Vec::new(),
            scroll: 0,
            sel: 0,
            status: String::from("Enter a URL and press Enter. Tab picks a link."),
            addr: String::from(HOME),
            editing: true,
            rows_seen: Cell::new(0),
            page_rows: Cell::new(20),
            hops: 0,
        }
    }

    pub fn title(&self) -> &str {
        match &self.page {
            Some(p) if !p.title.is_empty() => &p.title,
            _ => "Enternet",
        }
    }

    /// Load a URL typed somewhere other than the address bar.
    pub fn load(&mut self, url: &str) {
        match html::parse_url(url) {
            Some(u) => {
                self.editing = false;
                self.go(u);
            }
            None => {
                self.addr = String::from(url);
                self.status = String::from("That is not a URL this browser speaks");
            }
        }
    }

    // --- fetching ---------------------------------------------------------

    fn go(&mut self, url: Url) {
        self.status = String::from("Looking up ");
        self.status.push_str(&url.host);

        let Some(ip) = crate::net::dns::lookup(&url.host).ok() else {
            self.status = String::from("Cannot resolve ");
            self.status.push_str(&url.host);
            return;
        };

        // Only https. The plain-text side would need a second client and this
        // one already exists; saying so is better than a silent failure.
        if !url.https {
            self.status = String::from("Only https is supported");
            return;
        }

        match crate::net::tls::https_get(ip, &url.host, url.port, &url.path) {
            Err(e) => {
                self.status = String::from("Fetch failed: ");
                self.status.push_str(e.name());
            }
            Ok((resp, _fp, _n, _id, _cn, _names)) => {
                let (status, head, body) = crate::net::tls::http_response(&resp);

                // Follow redirects here rather than showing the server's
                // "moved" page, which is what a browser that does not follow
                // them actually displays. Bounded, because a site can point at
                // itself and a browser that loops is worse than one that stops.
                if (300..400).contains(&status) {
                    if let Some(loc) = header(&head, "location:") {
                        if self.hops < 5 {
                            if let Some(mut next) = html::resolve(&url, &loc) {
                                // Never follow a redirect down from https to
                                // http. Servers emit an http Location more
                                // often than you would hope -- iana.org does
                                // it, two hops off example.com -- and doing as
                                // told meant arriving at a page this browser
                                // then refused to fetch, reporting "only https
                                // is supported" about a link that was https.
                                // Keeping the scheme is also the safe
                                // direction: it can never downgrade a
                                // connection that was already encrypted.
                                if url.https && !next.https {
                                    next.https = true;
                                    if next.port == 80 {
                                        next.port = 443;
                                    }
                                }
                                self.hops += 1;
                                self.go(next);
                                return;
                            }
                        } else {
                            self.status = String::from("Too many redirects");
                            return;
                        }
                    }
                }
                self.hops = 0;
                if status >= 400 {
                    self.status = String::from("Server said ");
                    push_num(&mut self.status, status as usize);
                }
                let page = html::parse(&body, &url);
                self.links.clear();
                collect_links(&page, &mut self.links);
                let n = self.links.len();
                if status < 400 {
                    self.status = String::from("Loaded, ");
                push_num(&mut self.status, page.blocks.len());
                self.status.push_str(" blocks, ");
                push_num(&mut self.status, n);
                self.status.push_str(if n == 1 { " link" } else { " links" });
                }
                self.page = Some(page);
                if let Some(prev) = self.here.take() {
                    self.history.push(prev);
                }
                self.addr = url.text();
                self.here = Some(url);
                self.scroll = 0;
                self.sel = 0;
            }
        }
    }

    fn back(&mut self) {
        if let Some(u) = self.history.pop() {
            self.here = None; // do not push the page being left onto history
            self.go(u);
        } else {
            self.status = String::from("No page to go back to");
        }
    }

    // --- layout -----------------------------------------------------------

    fn rows(&self, cols: usize) -> Vec<Row> {
        let mut out = Vec::new();
        let Some(page) = &self.page else { return out };
        let cols = cols.max(8);
        let mut link_id = 0usize;

        for block in &page.blocks {
            match block {
                Block::Rule => out.push(Row { text: String::new(), link: None, kind: 8 }),
                Block::Pre(t) => {
                    for line in t.split('\n') {
                        out.push(Row {
                            text: String::from(line.trim_end()),
                            link: None,
                            kind: 7,
                        });
                    }
                }
                Block::Heading(l, spans) => {
                    wrap(spans, cols, *l, "", &mut link_id, &mut out);
                    out.push(Row { text: String::new(), link: None, kind: 0 });
                }
                Block::Para(spans) => {
                    wrap(spans, cols, 0, "", &mut link_id, &mut out);
                    out.push(Row { text: String::new(), link: None, kind: 0 });
                }
                Block::Item(spans) => {
                    wrap(spans, cols.saturating_sub(2), 0, " . ", &mut link_id, &mut out)
                }
            }
        }
        out
    }

    /// Scroll by a number of rows, as the wheel asks.
    pub fn scroll_by(&mut self, rows: i32) {
        let max = self.rows_seen.get().saturating_sub(1);
        self.scroll = if rows < 0 {
            self.scroll.saturating_sub((-rows) as usize)
        } else {
            self.scroll.saturating_add(rows as usize).min(max)
        };
    }

    // --- input ------------------------------------------------------------

    pub fn key(&mut self, k: u8) -> bool {
        use super::super::dev::kbd;

        if self.editing {
            match k {
                b'\n' | b'\r' => {
                    self.editing = false;
                    match html::parse_url(&self.addr) {
                        Some(u) => self.go(u),
                        None => self.status = String::from("That is not a URL this browser speaks"),
                    }
                }
                27 => self.editing = false,
                8 | 127 => {
                    self.addr.pop();
                }
                c if (32..127).contains(&c) => self.addr.push(c as char),
                _ => return false,
            }
            return true;
        }

        match k {
            b'g' | b'G' => {
                self.editing = true;
                self.status = String::from("Editing the address. Enter loads it, Esc cancels.");
            }
            kbd::KEY_DOWN => self.scroll = self.scroll.saturating_add(1),
            kbd::KEY_UP => self.scroll = self.scroll.saturating_sub(1),
            b' ' => self.scroll = self.scroll.saturating_add(self.page_rows.get()),
            b'b' | b'B' => self.scroll = self.scroll.saturating_sub(self.page_rows.get()),
            kbd::KEY_HOME => self.scroll = 0,
            kbd::KEY_END => self.scroll = self.rows_seen.get().saturating_sub(1),
            b'\t' => self.select(1),
            kbd::KEY_BACKTAB => self.select(-1),
            b'\n' | b'\r' => {
                if let Some(u) = self.links.get(self.sel).cloned() {
                    self.go(u);
                } else {
                    self.status = String::from("No link selected");
                }
            }
            8 | 127 => self.back(),
            _ => return false,
        }
        let max = self.rows_seen.get().saturating_sub(1);
        if self.scroll > max {
            self.scroll = max;
        }
        true
    }

    fn select(&mut self, dir: i32) {
        if self.links.is_empty() {
            self.status = String::from("This page has no links");
            return;
        }
        let n = self.links.len();
        self.sel = if dir > 0 {
            (self.sel + 1) % n
        } else {
            (self.sel + n - 1) % n
        };
        self.status = String::from("Link ");
        push_num(&mut self.status, self.sel + 1);
        self.status.push('/');
        push_num(&mut self.status, n);
        self.status.push_str(": ");
        self.status.push_str(&self.links[self.sel].text());
        self.scroll_to_selection();
    }

    /// Bring the selected link into view. Without this, Tab appears to do
    /// nothing on a long page: the selection moves somewhere off screen and
    /// the only feedback is the status line.
    fn scroll_to_selection(&mut self) {
        let rows = self.rows(80);
        if let Some(at) = rows.iter().position(|r| r.link == Some(self.sel)) {
            let h = self.page_rows.get().max(1);
            if at < self.scroll || at >= self.scroll + h {
                self.scroll = at.saturating_sub(h / 3);
            }
        }
    }

    /// One pointer press inside the client area. In the address bar it starts
    /// editing; on a link it follows it. Returns whether anything changed.
    ///
    /// The geometry here is `draw_in`'s, from the same `metrics`, so a link
    /// is followed exactly where it is drawn.
    pub fn click(&mut self, client: Rect, px: i32, py: i32) -> bool {
        let (well, text) = Self::metrics(client);
        let inside = |r: Rect| {
            px >= r.x as i32 && py >= r.y as i32 && px < (r.x + r.w) as i32 && py < (r.y + r.h) as i32
        };
        if inside(well) {
            self.editing = true;
            self.status = String::from("Editing the address. Enter loads it, Esc cancels.");
            return true;
        }
        if !inside(text) {
            return false;
        }
        let lh = theme::text_h().max(1) as i32;
        let cw = theme::text_w(1).max(1) as usize;
        let line = self.scroll + ((py - text.y as i32) / lh).max(0) as usize;
        let cols = (text.w as usize / cw).max(1);
        let rows = self.rows(cols);
        let Some(row) = rows.get(line) else { return false };
        let Some(link) = row.link else { return false };
        // Only within the text of the row, not the empty space after it.
        let col = ((px - text.x as i32).max(0) as usize) / cw;
        if col > row.text.len() {
            return false;
        }
        self.sel = link;
        if let Some(u) = self.links.get(link).cloned() {
            self.go(u);
        }
        true
    }

    /// The address well and the page text area for a client rectangle --
    /// the two places a press means something.
    fn metrics(client: Rect) -> (Rect, Rect) {
        let pad = 4;
        let cw = theme::text_w(1).max(1);
        let lh = theme::text_h().max(1);
        let bar = Rect::new(client.x + pad, client.y + pad, client.w - pad * 2, lh + 6);
        let cap = theme::text_w(4);
        let well = Rect::new(bar.x + cap, bar.y, bar.w.saturating_sub(cap), bar.h);
        let status_h = lh + 4;
        let view = Rect::new(
            client.x + pad,
            bar.y + bar.h + 4,
            client.w - pad * 2,
            client.h.saturating_sub(bar.h + status_h + pad * 3 + 4),
        );
        let _ = cw;
        (well, view.shrink(3))
    }

    // --- drawing ----------------------------------------------------------

    pub fn draw_in(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        theme::panel(fb, client);
        let pad = 4;
        let cw = theme::text_w(1).max(1);
        let lh = theme::text_h().max(1);

        // Address bar across the top, then the page, then a status line.
        let bar = Rect::new(client.x + pad, client.y + pad, client.w - pad * 2, lh + 6);
        theme::text(fb, bar.x, bar.y + 3, "URL", theme::TEXT, theme::FACE);
        let cap = theme::text_w(4);
        let well = Rect::new(bar.x + cap, bar.y, bar.w.saturating_sub(cap), bar.h);
        theme::well(fb, well, theme::HILIGHT);
        let inner = well.shrink(3);
        let fit = (inner.w / cw) as usize;
        let shown = tail(&self.addr, fit.saturating_sub(1));
        let mut line = String::from(shown);
        if self.editing && focused {
            line.push('_');
        }
        theme::text(fb, inner.x, inner.y, &line, theme::TEXT, theme::HILIGHT);

        let status_h = lh + 4;
        let view = Rect::new(
            client.x + pad,
            bar.y + bar.h + 4,
            client.w - pad * 2,
            client
                .h
                .saturating_sub(bar.h + status_h + pad * 3 + 4),
        );
        theme::well(fb, view, theme::SCREEN);
        let text = view.shrink(3);
        let cols = (text.w / cw) as usize;
        let lines = (text.h / lh) as usize;
        self.page_rows.set(lines.max(1));

        let rows = self.rows(cols);
        self.rows_seen.set(rows.len());

        for (i, row) in rows.iter().skip(self.scroll).take(lines).enumerate() {
            let y = text.y + i as u32 * lh;
            if row.kind == 8 {
                theme::separator(fb, text.x, y + lh / 2, text.w);
                continue;
            }
            let selected = focused && row.link.is_some() && row.link == Some(self.sel);
            let (fg, bg) = if selected {
                (theme::SELECT_TEXT, theme::SELECT)
            } else if row.link.is_some() {
                (theme::LINK, theme::SCREEN)
            } else if row.kind >= 1 && row.kind <= 6 {
                (theme::HEADING, theme::SCREEN)
            } else {
                (theme::SCREEN_TEXT, theme::SCREEN)
            };
            theme::text(fb, text.x, y, &row.text, fg, bg);
        }

        let sy = view.y + view.h + 3;
        let status = tail_front(&self.status, (view.w / cw) as usize);
        theme::text(fb, view.x, sy, status, theme::TEXT, theme::FACE);
    }
}

// --- helpers --------------------------------------------------------------

fn collect_links(page: &Page, out: &mut Vec<Url>) {
    for b in &page.blocks {
        let spans = match b {
            Block::Heading(_, s) | Block::Para(s) | Block::Item(s) => s,
            _ => continue,
        };
        for s in spans {
            if let Span::Link { href, .. } = s {
                if let Some(u) = html::parse_url(href) {
                    out.push(u);
                }
            }
        }
    }
}

/// Wrap a block's spans into rows, assigning link numbers in document order.
fn wrap(
    spans: &[Span],
    cols: usize,
    kind: u8,
    marker: &str,
    link_id: &mut usize,
    out: &mut Vec<Row>,
) {
    let mut cur = String::from(marker);
    let mut cur_link: Option<usize> = None;
    let mut started = false;

    for span in spans {
        let id = match span {
            Span::Link { .. } => {
                let i = *link_id;
                *link_id += 1;
                Some(i)
            }
            Span::Text(_) => None,
        };
        for word in span.text().split_whitespace() {
            let need = word.chars().count() + if cur.is_empty() { 0 } else { 1 };
            if started && cur.chars().count() + need > cols {
                out.push(Row { text: core::mem::take(&mut cur), link: cur_link, kind });
                cur_link = None;
            }
            if !cur.is_empty() && !cur.ends_with(' ') {
                cur.push(' ');
            }
            cur.push_str(word);
            started = true;
            // A row carries one link id: the first one on it. A link split
            // across a wrap keeps its id on both rows because the id is
            // assigned per span, not per row.
            if id.is_some() && cur_link.is_none() {
                cur_link = id;
            }
        }
        if id.is_some() && cur_link.is_none() {
            cur_link = id;
        }
    }
    if !cur.trim().is_empty() {
        out.push(Row { text: cur, link: cur_link, kind });
    }
}

/// One header's value, from the lower-cased header block.
///
/// Matched at the start of a line. A bare `find` would take `content-location`
/// for `location`, and the wrong one of those sends the browser somewhere the
/// server never redirected it to.
fn header(head: &str, name: &str) -> Option<String> {
    let at = head
        .match_indices(name)
        .find(|(i, _)| *i == 0 || head.as_bytes()[i - 1] == b'\n')
        .map(|(i, _)| i)?;
    let rest = &head[at + name.len()..];
    let end = rest.find('\n').unwrap_or(rest.len());
    Some(String::from(rest[..end].trim()))
}

fn push_num(s: &mut String, mut n: usize) {
    if n == 0 {
        s.push('0');
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
        s.push(*c as char);
    }
}

/// The last `n` characters, so a long URL shows its path rather than its
/// scheme.
///
/// It counted bytes, on a note saying everything here was Latin-1 by the time
/// it reached the screen. That was true while the console drew one cell per
/// byte and stopped being true the moment it decoded UTF-8, and a page title
/// with an accent in it turned a cosmetic trim into a panic.
fn tail(s: &str, n: usize) -> &str {
    super::theme::tail_chars(s, n)
}

fn tail_front(s: &str, n: usize) -> &str {
    super::theme::head_chars(s, n)
}

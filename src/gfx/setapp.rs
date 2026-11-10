//! Settings: a home grid of categories, and each category's page beside a
//! sidebar of the others.
//!
//! **It replaces a panel that had been the Settings window for three major
//! versions**: a page list across the top taking half the window, a less and
//! a more button for every number, choices that did not say which one was
//! chosen, and a new window for every page opened. This is one window. It
//! opens on a grid of tiles, each saying what is true of its category right
//! now; a tile opens that category's page, and the window becomes a sidebar
//! and the page. Typing anywhere searches every setting there is.
//!
//! What a control shows and what it does both come from `prefs`, so this
//! module is only drawing and input. Changing a setting queues the shell
//! command that sets it -- the same one a person types -- because an app runs
//! inside the desktop's borrow and must not reach back into the desktop. A
//! slider shows its value while it is dragged and sends one command when it
//! is let go.

use super::prefs::{self, Ctl, Glyph, Item, Page};
use super::text::Face;
use super::theme::{self, Rect};
use super::ui::{Action, Tone, Widget};
use super::{Color, DeskApp, Framebuffer};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};

// ---- look ------------------------------------------------------------------

const BG: Color = Color::new(0xEC, 0xF1, 0xF4);
const SIDE: Color = Color::new(0xE1, 0xE8, 0xEC);
pub(super) const CARD: Color = Color::new(0xFF, 0xFF, 0xFF);
pub(super) const EDGE: Color = Color::new(0xD2, 0xDC, 0xE2);
pub(super) const RULE: Color = Color::new(0xE7, 0xEC, 0xEF);
pub(super) const OFF: Color = Color::new(0xC3, 0xCD, 0xD3);
const TINT: Color = Color::new(0xFD, 0xEB, 0xD6);
pub(super) const HOVER: Color = Color::new(0xF6, 0xF9, 0xFB);

pub(super) fn px(n: u32) -> u32 {
    theme::px(n)
}
pub(super) fn size() -> f32 {
    theme::ui_size()
}
fn pm(fb: &Framebuffer, c: Color) -> u32 {
    super::blend::opaque(fb.encode(c))
}
pub(super) fn rrect(fb: &Framebuffer, x: i32, y: i32, w: u32, h: u32, r: u32, c: Color) {
    super::shape::rrect(fb, super::shape::Box::new(x, y, w, h), r, pm(fb, c));
}
pub(super) fn ring(fb: &Framebuffer, x: i32, y: i32, w: u32, h: u32, r: u32, width: u32, c: Color) {
    super::shape::ring(fb, super::shape::Box::new(x, y, w, h), r, width, pm(fb, c));
}
pub(super) fn circle(fb: &Framebuffer, cx: f32, cy: f32, r: f32, c: Color) {
    super::shape::circle(fb, cx, cy, r, pm(fb, c));
}
pub(super) fn line_h(px: f32) -> u32 {
    (px * 1.5) as u32
}
/// Text with its top at `y`, in a row `px * 1.5` high.
pub(super) fn text(fb: &Framebuffer, x: i32, y: i32, s: &str, px: f32, face: Face, c: Color) -> u32 {
    let cap = px * 0.714;
    let base = y as f32 + (line_h(px) as f32 + cap) / 2.0;
    super::text::draw(fb, x as f32, (base + 0.5) as i32, face, px, s, c) as u32
}
pub(super) fn measure(s: &str, px: f32, face: Face) -> u32 {
    (super::text::measure(face, px, s) + 0.999) as u32
}
pub(super) fn fit(s: &str, px: f32, face: Face, max: u32) -> String {
    super::text::ellipsize(face, px, s, max as f32)
}
fn inside(r: Rect, x: i32, y: i32) -> bool {
    x >= r.x as i32 && x < (r.x + r.w) as i32 && y >= r.y as i32 && y < (r.y + r.h) as i32
}
/// Words wrapped to a width, for help lines and notes.
fn wrap(s: &str, px: f32, face: Face, max: u32) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for w in s.split_whitespace() {
        let trial = if cur.is_empty() { String::from(w) } else { format!("{} {}", cur, w) };
        if measure(&trial, px, face) > max && !cur.is_empty() {
            out.push(core::mem::replace(&mut cur, String::from(w)));
        } else {
            cur = trial;
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Each category's picture, from circles and boxes so it scales with
/// everything else rather than being a bitmap at one size.
pub(super) fn glyph(fb: &Framebuffer, g: Glyph, cx: f32, cy: f32, r: f32, c: Color) {
    let s = r / 10.0;
    let bx = |x: f32, y: f32, w: f32, h: f32, rad: f32| {
        rrect(fb, (cx + x * s) as i32, (cy + y * s) as i32, (w * s).max(1.0) as u32, (h * s).max(1.0) as u32, (rad * s) as u32, c)
    };
    let rg = |x: f32, y: f32, w: f32, h: f32, rad: f32, t: f32| {
        ring(fb, (cx + x * s) as i32, (cy + y * s) as i32, (w * s) as u32, (h * s) as u32, (rad * s) as u32, (t * s).max(1.0) as u32, c)
    };
    match g {
        Glyph::Network => {
            bx(-7.0, -8.0, 14.0, 9.0, 1.5);
            bx(-1.0, 1.0, 2.0, 4.0, 0.0);
            bx(-8.0, 5.0, 16.0, 2.0, 1.0);
        }
        Glyph::Wifi => {
            for (i, w) in [16.0f32, 10.0, 4.0].iter().enumerate() {
                rg(-w / 2.0, -6.0 + i as f32 * 3.0, *w, *w, w / 2.0, 1.8);
            }
            circle(fb, cx, cy + 6.0 * s, 1.6 * s, c);
        }
        Glyph::Desktop => {
            rg(-8.0, -7.0, 16.0, 12.0, 2.0, 1.8);
            bx(-8.0, -7.0, 7.0, 12.0, 2.0);
            bx(-4.0, 7.0, 8.0, 2.0, 1.0);
        }
        Glyph::Brush => {
            circle(fb, cx - 2.0 * s, cy - 2.0 * s, 6.5 * s, c);
            circle(fb, cx + 3.5 * s, cy + 3.5 * s, 4.0 * s, c);
        }
        Glyph::Sound => {
            bx(-8.0, -3.0, 5.0, 6.0, 1.0);
            bx(-4.0, -6.0, 4.0, 12.0, 1.0);
            rg(1.0, -5.0, 7.0, 10.0, 5.0, 1.6);
        }
        Glyph::Power => {
            rg(-8.0, -5.0, 14.0, 10.0, 2.0, 1.8);
            bx(-6.0, -3.0, 7.0, 6.0, 1.0);
            bx(6.0, -2.0, 2.0, 4.0, 1.0);
        }
        Glyph::Assistant => {
            circle(fb, cx, cy, 3.5 * s, c);
            for i in 0..6 {
                let a = i as f32 * core::f32::consts::PI / 3.0;
                circle(fb, cx + 7.0 * s * crate::ai::tensor::cosf(a), cy + 7.0 * s * crate::ai::tensor::sinf(a), 1.6 * s, c);
            }
        }
        Glyph::Model => {
            rg(-6.0, -6.0, 12.0, 12.0, 2.0, 1.8);
            bx(-2.5, -2.5, 5.0, 5.0, 1.0);
            for k in [-3.0f32, 0.0, 3.0] {
                bx(k - 0.7, -9.0, 1.4, 3.0, 0.0);
                bx(k - 0.7, 6.0, 1.4, 3.0, 0.0);
                bx(-9.0, k - 0.7, 3.0, 1.4, 0.0);
                bx(6.0, k - 0.7, 3.0, 1.4, 0.0);
            }
        }
        Glyph::Storage => {
            rg(-8.0, -6.0, 16.0, 12.0, 2.5, 1.8);
            circle(fb, cx + 4.5 * s, cy + 2.5 * s, 1.4 * s, c);
            bx(-5.0, -2.0, 7.0, 1.6, 0.8);
        }
        Glyph::Linux => {
            rg(-8.0, -7.0, 16.0, 14.0, 2.0, 1.8);
            bx(-5.0, -2.0, 4.0, 1.6, 0.8);
            bx(-1.0, 2.0, 6.0, 1.6, 0.8);
        }
        Glyph::Hardware => {
            bx(-6.0, -8.0, 12.0, 7.0, 2.0);
            bx(-4.0, -1.0, 2.0, 5.0, 0.0);
            bx(2.0, -1.0, 2.0, 5.0, 0.0);
            bx(-1.0, 4.0, 2.0, 5.0, 0.5);
        }
        Glyph::Updates => {
            rg(-7.0, -7.0, 14.0, 14.0, 7.0, 1.8);
            bx(-1.0, -4.0, 2.0, 5.0, 1.0);
            bx(-1.0, -1.0, 4.0, 2.0, 1.0);
        }
        Glyph::System => {
            rg(-7.0, -7.0, 14.0, 14.0, 7.0, 1.8);
            circle(fb, cx, cy - 3.0 * s, 1.3 * s, c);
            bx(-1.0, -0.5, 2.0, 5.5, 1.0);
        }
    }
}

// ---- what can be pressed -------------------------------------------------------

#[derive(Clone)]
enum Hit {
    Tile(usize),
    Result(usize),
    Nav(usize),
    Home,
    Search,
    Toggle(usize),
    Slider(usize, Rect),
    Choice(usize, usize),
    Button(usize),
    Run(String),
}

/// Where an item sits in `prefs::ITEMS`, which is how a hit names it.
fn index_of(it: &Item) -> usize {
    prefs::ITEMS.iter().position(|x| core::ptr::eq(x, it)).unwrap_or(0)
}

fn slider_val(ctl: &Ctl, frac: f32) -> i32 {
    match *ctl {
        Ctl::Slider { min, max, step, .. } => {
            let raw = min as f32 + frac.clamp(0.0, 1.0) * (max - min) as f32;
            let snapped = ((raw - min as f32) / step as f32 + 0.5) as i32 * step + min;
            snapped.clamp(min, max)
        }
        _ => 0,
    }
}

pub struct Settings {
    page: Option<usize>,
    query: String,
    /// Which pressable thing the keyboard is on, an index into `hits`.
    sel: Cell<usize>,
    scroll: Cell<i32>,
    /// A slider being dragged, and the value it shows meanwhile.
    drag: Option<(usize, Rect, i32)>,
    /// Values set but not yet read back from the machine: the command has
    /// been queued and the shell has not run it. Shown so a control does not
    /// flick back to its old value for the moment in between.
    pending: RefCell<Vec<(usize, i32)>>,
    hits: RefCell<Vec<(Rect, Hit)>>,
    hover: Cell<Option<usize>>,
    /// Sections and summaries are not free -- the hardware page walks the
    /// bus -- so they are kept a couple of seconds.
    sections: RefCell<(u64, Option<usize>, Vec<(usize, Vec<Widget>)>)>,
    summaries: RefCell<(u64, Vec<String>)>,
    content_h: Cell<i32>,
    /// A button that ends the session, pressed once: it asks to be pressed
    /// again within a few seconds. A settings page is a place people click
    /// around in, and a stray click there should not turn the machine off.
    armed: Cell<Option<(usize, u64)>>,
}

/// Commands that end the session, and so take two presses.
const DRASTIC: &[&str] = &["reboot", "shutdown"];

const KEEP_TICKS: u64 = 2 * crate::TIMER_HZ as u64;

impl Settings {
    pub fn new(page: &str) -> Settings {
        let mut s = Settings {
            page: None,
            query: String::new(),
            sel: Cell::new(0),
            scroll: Cell::new(0),
            drag: None,
            pending: RefCell::new(Vec::new()),
            hits: RefCell::new(Vec::new()),
            hover: Cell::new(None),
            sections: RefCell::new((0, None, Vec::new())),
            summaries: RefCell::new((0, Vec::new())),
            content_h: Cell::new(0),
            armed: Cell::new(None),
        };
        s.go(page);
        s
    }

    pub fn preferred() -> (u32, u32) {
        (px(820), px(600))
    }

    /// Show a page by id, or the home grid for anything else.
    pub fn go(&mut self, id: &str) {
        self.page = prefs::PAGES.iter().position(|p| p.id == id);
        self.query.clear();
        self.scroll.set(0);
        self.sel.set(0);
        self.drag = None;
    }

    fn page_ref(&self) -> Option<&'static Page> {
        self.page.map(|i| &prefs::PAGES[i])
    }

    fn summaries(&self) -> Vec<String> {
        let now = crate::dev::lapic::ticks();
        let mut c = self.summaries.borrow_mut();
        if c.1.is_empty() || now.saturating_sub(c.0) > KEEP_TICKS {
            *c = (now, prefs::PAGES.iter().map(|p| (p.summary)()).collect());
        }
        c.1.clone()
    }

    fn section(&self, item: usize) -> Vec<Widget> {
        let now = crate::dev::lapic::ticks();
        let mut c = self.sections.borrow_mut();
        if c.1 != self.page || now.saturating_sub(c.0) > KEEP_TICKS {
            c.0 = now;
            c.1 = self.page;
            c.2.clear();
        }
        if let Some((_, w)) = c.2.iter().find(|(i, _)| *i == item) {
            return w.clone();
        }
        let w = match prefs::ITEMS[item].ctl {
            Ctl::Section(f) => f(),
            _ => Vec::new(),
        };
        c.2.push((item, w.clone()));
        w
    }

    fn shown_value(&self, item: usize, live: i32) -> i32 {
        if let Some((i, _, v)) = self.drag {
            if i == item {
                return v;
            }
        }
        let mut p = self.pending.borrow_mut();
        if let Some(pos) = p.iter().position(|(i, _)| *i == item) {
            // The machine has caught up: forget the guess.
            if p[pos].1 == live {
                p.remove(pos);
                return live;
            }
            return p[pos].1;
        }
        live
    }

    fn run(&self, cmd: &str) {
        super::desk::queue_command(cmd);
        // Sections and summaries describe the machine as it was.
        self.sections.borrow_mut().0 = 0;
        self.summaries.borrow_mut().1.clear();
    }

    /// Do what the thing at `hits[i]` does.
    fn fire(&mut self, i: usize) -> bool {
        let Some((_, h)) = self.hits.borrow().get(i).cloned() else { return false };
        match h {
            Hit::Tile(p) | Hit::Nav(p) => {
                let id = prefs::PAGES[p].id;
                self.go(id);
            }
            Hit::Home => self.go(""),
            Hit::Search => {}
            Hit::Result(it) => {
                let item = &prefs::ITEMS[it];
                self.go(item.page);
                // Land the keyboard on the setting that was found.
                self.sel.set(usize::MAX - it);
            }
            Hit::Toggle(it) => {
                if let Ctl::Toggle { get, on, off } = prefs::ITEMS[it].ctl {
                    let now = self.shown_value(it, get() as i32) != 0;
                    self.pending.borrow_mut().retain(|(i, _)| *i != it);
                    self.pending.borrow_mut().push((it, (!now) as i32));
                    self.run(if now { off } else { on });
                }
            }
            Hit::Choice(it, o) => {
                if let Ctl::Choice { opts, .. } = prefs::ITEMS[it].ctl {
                    self.pending.borrow_mut().retain(|(i, _)| *i != it);
                    self.pending.borrow_mut().push((it, o as i32));
                    self.run(opts[o].1);
                }
            }
            Hit::Button(it) => match prefs::ITEMS[it].ctl {
                Ctl::Button(cmd) if DRASTIC.contains(&cmd) => {
                    let now = crate::dev::lapic::ticks();
                    match self.armed.get() {
                        Some((i, at)) if i == it && now.saturating_sub(at) < 4 * crate::TIMER_HZ as u64 => {
                            self.armed.set(None);
                            self.run(cmd);
                        }
                        _ => self.armed.set(Some((it, now))),
                    }
                }
                Ctl::Button(cmd) => self.run(cmd),
                Ctl::Open(what) => match what.strip_prefix("page:") {
                    Some(id) => self.go(id),
                    None => self.run(what),
                },
                _ => {}
            },
            Hit::Run(cmd) => match cmd.strip_prefix("win open set:") {
                Some(r) => self.go(prefs::from_route(r)),
                None => self.run(&cmd),
            },
            Hit::Slider(..) => {}
        }
        true
    }

    /// Step a slider or a choice by `d`, from the keyboard.
    fn nudge(&mut self, i: usize, d: i32) -> bool {
        let Some((_, h)) = self.hits.borrow().get(i).cloned() else { return false };
        match h {
            Hit::Slider(it, _) => {
                if let Ctl::Slider { min, max, step, get, cmd, .. } = prefs::ITEMS[it].ctl {
                    let v = (self.shown_value(it, get()) + d * step).clamp(min, max);
                    self.pending.borrow_mut().retain(|(x, _)| *x != it);
                    self.pending.borrow_mut().push((it, v));
                    self.run(&cmd(v));
                    return true;
                }
                false
            }
            Hit::Choice(it, _) => {
                if let Ctl::Choice { opts, get } = prefs::ITEMS[it].ctl {
                    let cur = self.shown_value(it, get() as i32);
                    let o = (cur + d).clamp(0, opts.len() as i32 - 1) as usize;
                    self.pending.borrow_mut().retain(|(x, _)| *x != it);
                    self.pending.borrow_mut().push((it, o as i32));
                    self.run(opts[o].1);
                    return true;
                }
                false
            }
            Hit::Toggle(_) => self.fire(i),
            _ => false,
        }
    }

    // ---- drawing -------------------------------------------------------------

    fn push(&self, r: Rect, h: Hit) {
        self.hits.borrow_mut().push((r, h));
    }

    fn lit(&self, i: usize, focused: bool) -> (bool, bool) {
        (self.hover.get() == Some(i), focused && self.sel.get() == i)
    }

    fn draw_search(&self, fb: &Framebuffer, r: Rect, focused: bool) {
        let i = self.hits.borrow().len();
        rrect(fb, r.x as i32, r.y as i32, r.w, r.h, r.h / 2, CARD);
        let (_, kb) = self.lit(i, focused);
        ring(fb, r.x as i32, r.y as i32, r.w, r.h, r.h / 2, px(1).max(1), if kb || !self.query.is_empty() { theme::APERTURE } else { EDGE });
        // A magnifier.
        let cx = r.x as f32 + r.h as f32 * 0.55;
        let cy = r.y as f32 + r.h as f32 * 0.47;
        let rr = r.h as f32 * 0.17;
        ring(fb, (cx - rr) as i32, (cy - rr) as i32, (rr * 2.0) as u32, (rr * 2.0) as u32, rr as u32, (rr * 0.35).max(1.0) as u32, theme::TEXT_DIM);
        rrect(fb, (cx + rr * 0.6) as i32, (cy + rr * 0.6) as i32, (rr * 0.9) as u32, (rr * 0.9) as u32, (rr * 0.3) as u32, theme::TEXT_DIM);
        let tx = r.x as i32 + r.h as i32;
        let ty = r.y as i32 + (r.h as i32 - line_h(size()) as i32) / 2;
        if self.query.is_empty() {
            text(fb, tx, ty, "Find a setting", size(), Face::Ui, theme::TEXT_DIM);
        } else {
            let w = text(fb, tx, ty, &self.query, size(), Face::Ui, theme::TEXT);
            if focused {
                fb.rect((tx as u32 + w + 1).min(r.x + r.w), r.y + r.h / 4, px(2).max(1), r.h / 2, theme::APERTURE);
            }
        }
        self.push(r, Hit::Search);
    }

    fn draw_home(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        let pad = px(24);
        let mut y = pad as i32;
        text(fb, pad as i32, y, "Settings", size() * 1.9, Face::UiStrong, theme::TEXT);
        y += line_h(size() * 1.9) as i32 + px(10) as i32;
        let sw = client.w.saturating_sub(pad * 2).min(px(520));
        self.draw_search(fb, Rect::new(pad, y as u32, sw, px(36)), focused);
        y += px(36) as i32 + px(20) as i32;

        if !self.query.is_empty() {
            self.draw_results(fb, client, y, focused);
            return;
        }
        let gap = px(12);
        let avail = client.w.saturating_sub(pad * 2);
        let cols = ((avail + gap) / (px(230) + gap)).clamp(1, 4);
        let tw = (avail + gap) / cols - gap;
        let th = px(82);
        let sums = self.summaries();
        for (i, p) in prefs::PAGES.iter().enumerate() {
            let (c, row) = (i as u32 % cols, i as u32 / cols);
            let r = Rect::new(pad + c * (tw + gap), y as u32 + row * (th + gap), tw, th);
            if r.y + r.h > client.h + th {
                break;
            }
            let hi = self.hits.borrow().len();
            let (hov, kb) = self.lit(hi, focused);
            rrect(fb, r.x as i32, r.y as i32, r.w, r.h, px(10), if hov { HOVER } else { CARD });
            ring(fb, r.x as i32, r.y as i32, r.w, r.h, px(10), px(1).max(1), if kb || hov { theme::APERTURE } else { EDGE });
            if kb {
                ring(fb, r.x as i32 - 2, r.y as i32 - 2, r.w + 4, r.h + 4, px(12), 2, theme::APERTURE);
            }
            let gr = px(20);
            let gx = r.x + px(14) + gr;
            let gy = r.y + r.h / 2;
            circle(fb, gx as f32, gy as f32, gr as f32, TINT);
            glyph(fb, p.glyph, gx as f32, gy as f32, gr as f32 * 0.62, theme::APERTURE_DEEP);
            let tx = (gx + gr + px(12)) as i32;
            let room = (r.x + r.w).saturating_sub(tx as u32 + px(10));
            let t1 = r.y as i32 + (r.h as i32 - line_h(size()) as i32 - line_h(size() * 0.86) as i32) / 2;
            text(fb, tx, t1, &fit(p.title, size(), Face::UiStrong, room), size(), Face::UiStrong, theme::TEXT);
            let sum = sums.get(i).cloned().unwrap_or_default();
            text(fb, tx, t1 + line_h(size()) as i32, &fit(&sum, size() * 0.86, Face::Ui, room), size() * 0.86, Face::Ui, theme::TEXT_DIM);
            self.push(r, Hit::Tile(i));
        }
    }

    fn draw_results(&self, fb: &Framebuffer, client: Rect, y0: i32, focused: bool) {
        let pad = px(24);
        let hits = prefs::search(&self.query);
        let mut y = y0;
        if hits.is_empty() {
            text(fb, pad as i32, y, &format!("Nothing matches \"{}\".", self.query), size(), Face::Ui, theme::TEXT_DIM);
            return;
        }
        let w = client.w.saturating_sub(pad * 2);
        let rh = line_h(size()) + line_h(size() * 0.86) + px(12);
        for it in hits {
            if y as u32 > client.h {
                break;
            }
            let r = Rect::new(pad, y as u32, w, rh);
            let hi = self.hits.borrow().len();
            let (hov, kb) = self.lit(hi, focused);
            rrect(fb, r.x as i32, r.y as i32, r.w, r.h, px(8), if hov || kb { CARD } else { BG });
            if kb {
                ring(fb, r.x as i32, r.y as i32, r.w, r.h, px(8), 2, theme::APERTURE);
            }
            let title = prefs::page(it.page).map(|p| p.title).unwrap_or("");
            text(fb, r.x as i32 + px(14) as i32, r.y as i32 + px(6) as i32, it.label, size(), Face::UiStrong, theme::TEXT);
            let sub = if it.help.is_empty() { format!("{} > {}", title, it.group) } else { format!("{} > {}  -  {}", title, it.group, it.help) };
            text(fb, r.x as i32 + px(14) as i32, r.y as i32 + px(6) as i32 + line_h(size()) as i32, &fit(&sub, size() * 0.86, Face::Ui, r.w.saturating_sub(px(28))), size() * 0.86, Face::Ui, theme::TEXT_DIM);
            self.push(r, Hit::Result(index_of(it)));
            y += rh as i32 + px(4) as i32;
        }
    }

    fn draw_sidebar(&self, fb: &Framebuffer, client: Rect, focused: bool) -> u32 {
        let sw = px(210).min(client.w / 3);
        fb.rect(0, 0, sw, client.h, SIDE);
        let pad = px(12);
        let rh = line_h(size()) + px(10);
        let mut y = pad;
        let back = Rect::new(pad / 2, y, sw - pad, rh);
        let hi = self.hits.borrow().len();
        let (hov, kb) = self.lit(hi, focused);
        if hov || kb {
            rrect(fb, back.x as i32, back.y as i32, back.w, back.h, px(6), HOVER);
        }
        text(fb, (back.x + pad / 2) as i32, (back.y + px(5)) as i32, "<  All settings", size(), Face::UiStrong, theme::APERTURE_DEEP);
        self.push(back, Hit::Home);
        y += rh + px(8);
        for (i, p) in prefs::PAGES.iter().enumerate() {
            let r = Rect::new(pad / 2, y, sw - pad, rh);
            let hi = self.hits.borrow().len();
            let (hov, kb) = self.lit(hi, focused);
            let on = self.page == Some(i);
            if on {
                rrect(fb, r.x as i32, r.y as i32, r.w, r.h, px(6), CARD);
                rrect(fb, r.x as i32, (r.y + r.h / 4) as i32, px(3).max(2), r.h / 2, px(2), theme::APERTURE);
            } else if hov || kb {
                rrect(fb, r.x as i32, r.y as i32, r.w, r.h, px(6), HOVER);
            }
            if kb && !on {
                ring(fb, r.x as i32, r.y as i32, r.w, r.h, px(6), 1, theme::APERTURE);
            }
            let gr = px(9) as f32;
            glyph(fb, p.glyph, (r.x + px(16)) as f32, (r.y + r.h / 2) as f32, gr, if on { theme::APERTURE_DEEP } else { theme::TEXT_DIM });
            text(fb, (r.x + px(32)) as i32, (r.y + px(5)) as i32, &fit(p.title, size(), Face::Ui, r.w.saturating_sub(px(36))), size(), if on { Face::UiStrong } else { Face::Ui }, theme::TEXT);
            self.push(r, Hit::Nav(i));
            y += rh + px(2);
        }
        sw
    }

    /// One row of a card: the label and its help on the left, and the control
    /// on the right. Answers the row's height.
    fn draw_row(&self, fb: &Framebuffer, x: u32, y: i32, w: u32, it: &Item, focused: bool) -> u32 {
        let idx = index_of(it);
        let pad = px(16);
        let ctl_w = match it.ctl {
            Ctl::Slider { .. } => px(250),
            Ctl::Choice { opts, .. } => opts.iter().map(|(l, _)| measure(l, size() * 0.93, Face::Ui) + px(24)).sum::<u32>() + px(4),
            Ctl::Toggle { .. } => px(48),
            Ctl::Button(_) | Ctl::Open(_) => px(110),
            Ctl::Info(_) => px(220),
            Ctl::Section(_) => 0,
        }
        .min(w / 2);
        let label_w = w.saturating_sub(ctl_w + pad * 3);
        // A button row says its help as its label, so it has no help line.
        let is_button = matches!(it.ctl, Ctl::Button(_) | Ctl::Open(_));
        let help = if it.help.is_empty() || is_button { Vec::new() } else { wrap(it.help, size() * 0.86, Face::Ui, label_w) };
        let h = (line_h(size()) + help.len() as u32 * line_h(size() * 0.86) + px(20)).max(px(52));
        let row = Rect::new(x, y.max(0) as u32, w, h);
        let hi = self.hits.borrow().len();
        let (hov, kb) = self.lit(hi, focused);
        if hov || kb {
            rrect(fb, x as i32 + 2, y + 2, w - 4, h - 4, px(8), HOVER);
        }
        let ty = y + (h as i32 - line_h(size()) as i32 - help.len() as i32 * line_h(size() * 0.86) as i32) / 2;
        let label = match it.ctl {
            // A button's label is on the button.
            Ctl::Button(_) | Ctl::Open(_) => it.help,
            _ => it.label,
        };
        let shown_help: &[String] = match it.ctl {
            Ctl::Button(_) | Ctl::Open(_) => &[],
            _ => &help,
        };
        text(fb, (x + pad) as i32, ty, &fit(label, size(), Face::Ui, label_w), size(), Face::Ui, theme::TEXT);
        for (k, l) in shown_help.iter().enumerate() {
            text(fb, (x + pad) as i32, ty + line_h(size()) as i32 + k as i32 * line_h(size() * 0.86) as i32, l, size() * 0.86, Face::Ui, theme::TEXT_DIM);
        }
        let cx = (x + w).saturating_sub(pad + ctl_w);
        let cy = y + h as i32 / 2;
        match it.ctl {
            Ctl::Toggle { get, .. } => {
                let on = self.shown_value(idx, get() as i32) != 0;
                let (tw, th) = (px(44), px(24));
                let tx = (x + w).saturating_sub(pad + tw);
                let ty = cy - th as i32 / 2;
                rrect(fb, tx as i32, ty, tw, th, th / 2, if on { theme::APERTURE } else { OFF });
                if kb {
                    ring(fb, tx as i32 - 3, ty - 3, tw + 6, th + 6, th / 2 + 3, 2, theme::APERTURE_DEEP);
                }
                let kx = if on { tx + tw - th / 2 } else { tx + th / 2 };
                circle(fb, kx as f32, cy as f32, th as f32 * 0.38, CARD);
                self.push(row, Hit::Toggle(idx));
            }
            Ctl::Slider { min, max, unit, get, .. } => {
                let v = self.shown_value(idx, get());
                let val = format!("{}{}", v, unit);
                let vw = measure("0000 px", size(), Face::Ui);
                let track = Rect::new(cx, (cy - px(10) as i32).max(0) as u32, ctl_w.saturating_sub(vw + px(10)), px(20));
                let frac = (v - min) as f32 / (max - min).max(1) as f32;
                let ty = cy - px(2) as i32;
                rrect(fb, track.x as i32, ty, track.w, px(4), px(2), OFF);
                let fill = (track.w as f32 * frac) as u32;
                rrect(fb, track.x as i32, ty, fill.max(px(4)), px(4), px(2), theme::APERTURE);
                let kx = track.x as f32 + fill as f32;
                circle(fb, kx, cy as f32, px(9) as f32, if kb || self.drag.map(|d| d.0) == Some(idx) { theme::APERTURE_DEEP } else { theme::APERTURE });
                circle(fb, kx, cy as f32, px(5) as f32, CARD);
                text(fb, (track.x + track.w + px(12)) as i32, cy - line_h(size()) as i32 / 2, &val, size(), Face::Ui, theme::TEXT);
                self.push(row, Hit::Slider(idx, track));
            }
            Ctl::Choice { opts, get } => {
                let cur = self.shown_value(idx, get() as i32) as usize;
                let bh = px(30);
                let by = cy - bh as i32 / 2;
                let mut bx = cx;
                rrect(fb, bx as i32, by, ctl_w, bh, bh / 2, Color::new(0xE6, 0xEC, 0xF0));
                for (o, (l, _)) in opts.iter().enumerate() {
                    let bw = measure(l, size() * 0.93, Face::Ui) + px(24);
                    let r = Rect::new(bx + px(2), (by + px(2) as i32).max(0) as u32, bw, bh - px(4));
                    let on = o == cur;
                    if on {
                        rrect(fb, r.x as i32, by + px(2) as i32, r.w, r.h, r.h / 2, theme::APERTURE);
                    }
                    text(fb, (r.x + px(12)) as i32, by + (bh as i32 - line_h(size() * 0.93) as i32) / 2, l, size() * 0.93, if on { Face::UiStrong } else { Face::Ui }, if on { CARD } else { theme::TEXT });
                    if kb && on {
                        ring(fb, r.x as i32 - 2, by, r.w + 4, bh, bh / 2, 2, theme::APERTURE_DEEP);
                    }
                    self.hits.borrow_mut().push((r, Hit::Choice(idx, o)));
                    bx += bw;
                }
                // The row itself steps through the choices from the keyboard.
                self.push(Rect::new(0, 0, 0, 0), Hit::Choice(idx, cur));
            }
            Ctl::Button(_) | Ctl::Open(_) => {
                let armed = matches!(self.armed.get(), Some((i, at)) if i == idx
                    && crate::dev::lapic::ticks().saturating_sub(at) < 4 * crate::TIMER_HZ as u64);
                let label = if armed { "Press again" } else { it.label };
                let bw = measure(label, size(), Face::Ui).max(px(80)) + px(28);
                let r = Rect::new((x + w).saturating_sub(pad + bw), (cy - px(15) as i32).max(0) as u32, bw, px(30));
                theme::button(fb, r, label, kb || armed, armed);
                self.push(r, Hit::Button(idx));
            }
            Ctl::Info(f) => {
                let v = f();
                let vw = measure(&v, size(), Face::Ui).min(ctl_w);
                text(fb, (x + w).saturating_sub(pad + vw) as i32, cy - line_h(size()) as i32 / 2, &fit(&v, size(), Face::Ui, ctl_w), size(), Face::Ui, theme::TEXT_DIM);
            }
            Ctl::Section(_) => {}
        }
        h
    }

    /// The old page's rows, drawn inside a card. Answers the height used.
    fn draw_section(&self, fb: &Framebuffer, x: u32, y0: i32, w: u32, ws: &[Widget], focused: bool) -> u32 {
        let pad = px(16);
        let mut y = y0 + px(8) as i32;
        let name_w = (w / 3).min(px(200));
        for wd in ws {
            match wd {
                Widget::Heading(t) => {
                    if y > y0 + px(8) as i32 {
                        fb.rect(x + pad, (y + px(4) as i32).max(0) as u32, w - pad * 2, 1, RULE);
                        y += px(10) as i32;
                    }
                    text(fb, (x + pad) as i32, y, t, size() * 0.93, Face::UiStrong, theme::APERTURE_DEEP);
                    y += line_h(size() * 0.93) as i32 + px(2) as i32;
                }
                Widget::Status { name, value, tone } => {
                    let c = match tone {
                        Tone::Ok => theme::OK_TEXT,
                        Tone::Warn => theme::APERTURE_DEEP,
                        Tone::Bad => Color::new(0xB0, 0x20, 0x20),
                        Tone::Plain => theme::TEXT,
                    };
                    text(fb, (x + pad) as i32, y, &fit(name, size(), Face::Ui, name_w), size(), Face::Ui, theme::TEXT_DIM);
                    let room = w.saturating_sub(name_w + pad * 3);
                    let lines = wrap(value, size(), Face::Ui, room);
                    for (k, l) in lines.iter().enumerate() {
                        text(fb, (x + pad * 2 + name_w) as i32, y + k as i32 * line_h(size()) as i32, l, size(), Face::Ui, c);
                    }
                    y += line_h(size()) as i32 * lines.len().max(1) as i32 + px(4) as i32;
                }
                Widget::Note(lines) => {
                    let joined = lines.join(" ");
                    for l in wrap(&joined, size() * 0.86, Face::Ui, w - pad * 2) {
                        text(fb, (x + pad) as i32, y, &l, size() * 0.86, Face::Ui, theme::TEXT_DIM);
                        y += line_h(size() * 0.86) as i32;
                    }
                    y += px(4) as i32;
                }
                Widget::Label(t) => {
                    text(fb, (x + pad) as i32, y, &fit(t, size(), Face::Ui, w - pad * 2), size(), Face::Ui, theme::TEXT);
                    y += line_h(size()) as i32 + px(4) as i32;
                }
                Widget::Button { label, action } => {
                    let cmd = match action {
                        Action::Run(c) => c.clone(),
                        Action::Browse(r) => format!("win open {}", r),
                        _ => continue,
                    };
                    let bw = measure(label, size(), Face::Ui) + px(28);
                    let r = Rect::new(x + pad, y.max(0) as u32 + px(2), bw, px(30));
                    let hi = self.hits.borrow().len();
                    let (_, kb) = self.lit(hi, focused);
                    theme::button(fb, r, label, kb, false);
                    self.push(r, Hit::Run(cmd));
                    y += px(36) as i32;
                }
                Widget::Choices { name, options, sel } => {
                    text(fb, (x + pad) as i32, y + px(5) as i32, name, size(), Face::Ui, theme::TEXT_DIM);
                    let mut bx = x + pad * 2 + name_w;
                    for (o, (l, a)) in options.iter().enumerate() {
                        let Action::Run(c) = a else { continue };
                        let bw = measure(l, size(), Face::Ui) + px(20);
                        let r = Rect::new(bx, y.max(0) as u32, bw, px(30));
                        let hi = self.hits.borrow().len();
                        let (_, kb) = self.lit(hi, focused);
                        theme::button(fb, r, l, kb || o == *sel, o == *sel);
                        self.push(r, Hit::Run(c.clone()));
                        bx += bw + px(6);
                    }
                    y += px(36) as i32;
                }
                Widget::List { items, .. } => {
                    for (l, a) in items {
                        let cmd = match a {
                            Action::Run(c) => c.clone(),
                            Action::Browse(r) => format!("win open {}", r),
                            _ => continue,
                        };
                        let r = Rect::new(x + pad, y.max(0) as u32, w - pad * 2, line_h(size()) + px(6));
                        let hi = self.hits.borrow().len();
                        let (hov, kb) = self.lit(hi, focused);
                        if hov || kb {
                            rrect(fb, r.x as i32, y, r.w, r.h, px(6), HOVER);
                        }
                        text(fb, (r.x + px(8)) as i32, y + px(3) as i32, &fit(l, size(), Face::Ui, r.w - px(16)), size(), Face::Ui, theme::TEXT);
                        self.push(r, Hit::Run(cmd));
                        y += r.h as i32;
                    }
                }
                Widget::Sep => y += px(6) as i32,
                Widget::Field { .. } => {}
            }
        }
        (y - y0) as u32 + px(8)
    }

    fn draw_page(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        let Some(page) = self.page_ref() else { return };
        let sw = self.draw_sidebar(fb, client, focused);
        let pad = px(28);
        let x0 = sw + pad;
        let w = client.w.saturating_sub(x0 + pad);
        let head = px(70);
        // Under the heading, scrolled; the heading stays put.
        let body = fb.clipped(sw, head, client.w - sw, client.h.saturating_sub(head));
        let mut y = head as i32 + px(4) as i32 - self.scroll.get();
        let mut group = "";
        let mut card: Option<(i32, Vec<(i32, u32)>)> = None;
        let items: Vec<&Item> = prefs::on_page(page.id).collect();
        // A card is drawn round each group's rows after the rows are measured,
        // so its height is right; the rows are drawn over it in a second pass.
        let mut layout: Vec<(&Item, i32, u32, bool)> = Vec::new();
        for it in &items {
            if it.group != group {
                if let Some((top, _)) = card.take() {
                    let _ = top;
                    y += px(14) as i32;
                }
                group = it.group;
                text(&body, x0 as i32, y, group, size() * 0.93, Face::UiStrong, theme::TEXT_DIM);
                y += line_h(size() * 0.93) as i32 + px(6) as i32;
                card = Some((y, Vec::new()));
            }
            // Measure by drawing off the page: a row's height depends on how
            // its help wraps at this width.
            let h = self.row_height(it, w);
            layout.push((it, y, h, card.as_ref().map(|c| c.1.is_empty()).unwrap_or(true)));
            if let Some((_, rows)) = card.as_mut() {
                rows.push((y, h));
            }
            y += h as i32;
        }
        // Cards behind each group.
        let mut gi = 0;
        while gi < layout.len() {
            let g = layout[gi].0.group;
            let top = layout[gi].1;
            let mut bottom = top;
            let mut gj = gi;
            while gj < layout.len() && layout[gj].0.group == g {
                bottom = layout[gj].1 + layout[gj].2 as i32;
                gj += 1;
            }
            rrect(&body, x0 as i32, top, w, (bottom - top) as u32, px(10), CARD);
            ring(&body, x0 as i32, top, w, (bottom - top) as u32, px(10), 1, EDGE);
            for k in gi + 1..gj {
                body.rect(x0 + px(16), layout[k].1.max(0) as u32, w - px(32), 1, RULE);
            }
            gi = gj;
        }
        for (it, ry, _, _) in &layout {
            match it.ctl {
                Ctl::Section(_) => {
                    let ws = self.section(index_of(it));
                    self.draw_section(&body, x0, *ry, w, &ws, focused);
                }
                _ => {
                    self.draw_row(&body, x0, *ry, w, it, focused);
                }
            }
        }
        let total = y + self.scroll.get() - head as i32 + px(24) as i32;
        self.content_h.set(total);
        // The heading, over anything scrolled under it.
        fb.rect(sw, 0, client.w - sw, head, BG);
        text(fb, x0 as i32, px(18) as i32, page.title, size() * 1.7, Face::UiStrong, theme::TEXT);
        if self.scroll.get() > 0 {
            fb.rect(sw, head - 1, client.w - sw, 1, EDGE);
        }
    }

    fn row_height(&self, it: &Item, w: u32) -> u32 {
        match it.ctl {
            Ctl::Section(_) => {
                // Drawn off-screen to be measured: a surface with no rows.
                let ws = self.section(index_of(it));
                let n = self.hits.borrow().len();
                let fb = super::primary().map(|f| f.clipped(0, 0, 0, 0));
                let h = match fb {
                    Some(f) => self.draw_section(&f, 0, 0, w, &ws, false),
                    None => px(40),
                };
                self.hits.borrow_mut().truncate(n);
                h
            }
            _ => {
                let n = self.hits.borrow().len();
                let fb = super::primary().map(|f| f.clipped(0, 0, 0, 0));
                let h = match fb {
                    Some(f) => self.draw_row(&f, 0, 0, w, it, false),
                    None => px(52),
                };
                self.hits.borrow_mut().truncate(n);
                h
            }
        }
    }

    fn hit_at(&self, x: i32, y: i32) -> Option<usize> {
        self.hits.borrow().iter().position(|(r, _)| r.w > 0 && inside(*r, x, y))
    }
}

impl DeskApp for Settings {
    fn draw_in(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        self.hits.borrow_mut().clear();
        fb.rect(client.x, client.y, client.w, client.h, BG);
        match self.page {
            None => self.draw_home(fb, client, focused),
            Some(_) => self.draw_page(fb, client, focused),
        }
        // A search hit lands the keyboard on the setting it found.
        let s = self.sel.get();
        if s > usize::MAX / 2 {
            let want = usize::MAX - s;
            let at = self.hits.borrow().iter().position(|(_, h)| match h {
                Hit::Toggle(i) | Hit::Slider(i, _) | Hit::Choice(i, _) | Hit::Button(i) => *i == want,
                _ => false,
            });
            self.sel.set(at.unwrap_or(0));
        } else if s >= self.hits.borrow().len() {
            self.sel.set(0);
        }
    }

    fn key(&mut self, k: u8) -> bool {
        use crate::dev::kbd::{KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP};
        let n = self.hits.borrow().len().max(1);
        let pressable = |h: &Hit| !matches!(h, Hit::Search);
        match k {
            0x1B => {
                // Esc: out of the search, then out of the page.
                if !self.query.is_empty() {
                    self.query.clear();
                } else if self.page.is_some() {
                    self.go("");
                } else {
                    return false;
                }
                true
            }
            KEY_DOWN | b'\t' | KEY_UP => {
                let back = k == KEY_UP;
                let hits = self.hits.borrow();
                let mut i = self.sel.get();
                for _ in 0..n {
                    i = if back { (i + n - 1) % n } else { (i + 1) % n };
                    // Skip the zero-sized row markers and the search box.
                    if hits.get(i).map(|(r, h)| r.w > 0 && pressable(h)).unwrap_or(false) {
                        break;
                    }
                }
                drop(hits);
                self.sel.set(i);
                self.keep_visible();
                true
            }
            KEY_LEFT | KEY_RIGHT => {
                let d = if k == KEY_RIGHT { 1 } else { -1 };
                if self.page.is_none() {
                    // On the grid, the arrows walk the tiles.
                    let i = (self.sel.get() as i32 + d).rem_euclid(n as i32) as usize;
                    self.sel.set(i);
                    return true;
                }
                self.nudge(self.sel.get(), d)
            }
            b'\n' | b'\r' | b' ' if !(k == b' ' && self.page.is_none()) => {
                if self.page.is_none() && !self.query.is_empty() {
                    // Enter on a search opens the best match.
                    if let Some(it) = prefs::search(&self.query).first() {
                        let i = index_of(it);
                        self.go(it.page);
                        self.sel.set(usize::MAX - i);
                        return true;
                    }
                }
                let i = self.sel.get();
                self.fire(i)
            }
            8 | 0x7F => {
                if self.query.pop().is_none() && self.page.is_some() {
                    self.go("");
                }
                true
            }
            c if (0x20..0x7F).contains(&c) => {
                // Typing anywhere searches, from a page too.
                if self.page.is_some() {
                    self.go("");
                }
                self.query.push(c as char);
                self.sel.set(0);
                true
            }
            _ => false,
        }
    }

    fn press(&mut self, _client: Rect, x: i32, y: i32) -> bool {
        let Some(i) = self.hit_at(x, y) else { return false };
        self.sel.set(i);
        let h = self.hits.borrow()[i].1.clone();
        if let Hit::Slider(it, track) = h {
            let frac = (x - track.x as i32) as f32 / track.w.max(1) as f32;
            self.drag = Some((it, track, slider_val(&prefs::ITEMS[it].ctl, frac)));
            return true;
        }
        self.fire(i)
    }

    fn drag(&mut self, _client: Rect, x: i32, _y: i32) -> bool {
        if let Some((it, track, _)) = self.drag {
            let frac = (x - track.x as i32) as f32 / track.w.max(1) as f32;
            self.drag = Some((it, track, slider_val(&prefs::ITEMS[it].ctl, frac)));
            return true;
        }
        false
    }

    fn release(&mut self) -> bool {
        let Some((it, _, v)) = self.drag.take() else { return false };
        if let Ctl::Slider { cmd, get, .. } = prefs::ITEMS[it].ctl {
            if v != get() {
                self.pending.borrow_mut().retain(|(i, _)| *i != it);
                self.pending.borrow_mut().push((it, v));
                self.run(&cmd(v));
            }
        }
        true
    }

    fn hover(&mut self, _client: Rect, x: i32, y: i32) -> bool {
        let h = self.hit_at(x, y);
        if h != self.hover.get() {
            self.hover.set(h);
            return true;
        }
        false
    }

    fn wheel(&mut self, notches: i32) -> bool {
        if self.page.is_none() {
            return false;
        }
        let max = (self.content_h.get() - px(400) as i32).max(0);
        let s = (self.scroll.get() - notches * px(48) as i32).clamp(0, max);
        if s != self.scroll.get() {
            self.scroll.set(s);
            return true;
        }
        false
    }

    fn min_size(&self) -> (u32, u32) {
        (px(560), px(420))
    }

    fn floats(&self) -> bool {
        false
    }

    fn stamp(&self) -> Option<u64> {
        // What is drawn depends on the values the controls read, which change
        // under the window when a queued command runs; and on the tick, in
        // two-second steps, for the sections and the tiles.
        let mut vals: Vec<i32> = Vec::new();
        for it in prefs::ITEMS.iter() {
            match it.ctl {
                Ctl::Toggle { get, .. } => vals.push(get() as i32),
                Ctl::Slider { get, .. } => vals.push(get()),
                Ctl::Choice { get, .. } => vals.push(get() as i32),
                _ => {}
            }
        }
        let tick = crate::dev::lapic::ticks() / KEEP_TICKS;
        Some(super::scene::Stamp::of((
            self.page,
            self.query.as_str(),
            self.sel.get(),
            self.scroll.get(),
            self.drag.map(|d| (d.0, d.2)),
            self.armed.get(),
            self.hover.get(),
            self.pending.borrow().clone(),
            vals,
            tick,
            super::settings::generation(),
        )))
    }

    fn wake_ms(&self) -> Option<u32> {
        Some(500)
    }

    fn route(&mut self, to: &str) -> bool {
        let to = to.strip_prefix("set:").unwrap_or(to);
        self.go(prefs::from_route(to));
        true
    }
}

/// Open Settings at a page (an id, an old `set:` route, or "" for the home
/// grid), or turn the one already open to it. There is one Settings window:
/// every page used to be a window of its own, and four clicks left four.
pub fn open(page: &str) {
    if super::desk::route_app(TITLE, page) {
        return;
    }
    let (w, h) = Settings::preferred();
    let page = prefs::from_route(page.strip_prefix("set:").unwrap_or(page));
    super::desk::open_app(TITLE, super::desk::ICO_SET, alloc::boxed::Box::new(Settings::new(page)), w, h);
}

pub const TITLE: &str = "Settings";

impl Settings {
    /// Scroll so the keyboard's row is in view.
    fn keep_visible(&self) {
        let Some((r, _)) = self.hits.borrow().get(self.sel.get()).cloned() else { return };
        let head = px(70) as i32;
        let view = super::primary().map(|f| f.height() as i32 / 2).unwrap_or(400);
        let y = r.y as i32;
        if y < head {
            self.scroll.set((self.scroll.get() - (head - y) - px(20) as i32).max(0));
        } else if y + r.h as i32 > head + view {
            self.scroll.set(self.scroll.get() + (y + r.h as i32 - head - view) + px(20) as i32);
        }
    }
}

//! Widgets, keyboard focus, and the panel loop.
//!
//! `theme` draws; this decides. Everything here is keyboard-only on purpose:
//! there is no pointer, and there is not going to be one soon, so a control
//! that can only be reached by clicking is a control that cannot be reached.
//!
//! ### Why widgets are an enum
//!
//! `Box<dyn Widget>` with a `draw` method is the obvious shape and is rejected
//! for one decisive reason: the model has to be able to *write* one of these.
//! A trait object is not something a constrained decoder can emit. An enum
//! whose variants have a fixed operand count is -- each verb becomes a line
//! with known arity, and an invalid panel stops being representable rather
//! than being caught. The heap allocation and vtable per control would be a
//! poor trade for five kinds anyway.
//!
//! ### Why the layout is a vertical stack with fixed heights
//!
//! A constraint solver is the other option and is rejected for the same
//! reason. A layout that can be unsatisfiable is a panel that can be invalid.
//! A vertical stack cannot be unsatisfiable; it can only overflow, and
//! overflow clips.
//!
//! ### Why focus wraps
//!
//! Load-bearing, not cosmetic. There is no Shift-Tab over serial: the i8042
//! map gives scancode 0x0F the same byte whether shift is held or not, and a
//! terminal sends `ESC [ Z`, which the serial path does not decode. So a
//! control reachable only by Shift-Tab is one `tools/drive.py` could never
//! focus, and an interactive program that cannot be driven headlessly does not
//! get tested.

use super::font;
use super::theme::{self, Rect};
use super::Framebuffer;
use crate::dev::kbd;
use alloc::string::String;
use alloc::vec::Vec;

const TEXT_H: u32 = font::GLYPH_H * theme::CHROME_SCALE;
const ROW_H: u32 = TEXT_H + 8;
const BTN_H: u32 = TEXT_H + 16;
const GAP: u32 = 6;
const PAD: u32 = 10;

/// What activating a control does.
///
/// A shell command line rather than a function pointer, because the set of
/// things a panel can do is exactly the set of things that can be typed --
/// which keeps the GUI from becoming a second, smaller system with its own
/// capabilities.
#[derive(Clone)]
pub enum Action {
    Run(String),
    Close,
    None,
}

pub enum Widget {
    Label(String),
    Sep,
    List { items: Vec<(String, Action)>, sel: usize },
    Button { label: String, action: Action },
}

impl Widget {
    fn height(&self) -> u32 {
        match self {
            Widget::Label(_) => TEXT_H + GAP,
            Widget::Sep => GAP * 2,
            Widget::List { items, .. } => items.len() as u32 * ROW_H + 8,
            Widget::Button { .. } => BTN_H,
        }
    }

    fn focusable(&self) -> bool {
        matches!(self, Widget::List { .. } | Widget::Button { .. })
    }

    /// A one-line description for the serial transcript.
    fn trace(&self, focused: bool) -> (char, &str, usize) {
        let mark = if focused { '>' } else { ' ' };
        match self {
            Widget::Label(t) => (mark, t.as_str(), 0),
            Widget::Sep => (mark, "----", 0),
            Widget::List { items, sel } => {
                (mark, items.get(*sel).map(|i| i.0.as_str()).unwrap_or(""), *sel)
            }
            Widget::Button { label, .. } => (mark, label.as_str(), 0),
        }
    }
}

/// The result of handing one keystroke to a panel.
pub enum Step {
    Idle,
    Redraw,
    Do(Action),
    Close,
}

pub struct Panel {
    pub title: String,
    pub widgets: Vec<Widget>,
    focus: usize,
}

impl Panel {
    pub fn new(title: &str, widgets: Vec<Widget>) -> Self {
        let mut p = Self { title: String::from(title), widgets, focus: 0 };
        if !p.widgets.get(0).map(|w| w.focusable()).unwrap_or(false) {
            p.advance(false);
        }
        p
    }

    /// Move focus to the next (or previous) focusable widget, wrapping.
    fn advance(&mut self, back: bool) {
        let n = self.widgets.len();
        if n == 0 {
            return;
        }
        for step in 1..=n {
            let i = if back {
                (self.focus + n - step % n) % n
            } else {
                (self.focus + step) % n
            };
            if self.widgets[i].focusable() {
                self.focus = i;
                return;
            }
        }
    }

    fn activate(&self) -> Step {
        match self.widgets.get(self.focus) {
            Some(Widget::Button { action, .. }) => match action {
                Action::Close => Step::Close,
                a => Step::Do(a.clone()),
            },
            Some(Widget::List { items, sel }) => match items.get(*sel) {
                Some((_, Action::Close)) => Step::Close,
                Some((_, a)) => Step::Do(a.clone()),
                None => Step::Idle,
            },
            _ => Step::Idle,
        }
    }

    pub fn key(&mut self, k: u8) -> Step {
        match k {
            b'\t' => {
                self.advance(false);
                Step::Redraw
            }
            kbd::KEY_BACKTAB => {
                self.advance(true);
                Step::Redraw
            }
            kbd::KEY_DOWN => {
                // Within a list first, then out of it. That is what makes a
                // list feel like a list and still leaves Tab as the way out.
                if let Some(Widget::List { items, sel }) = self.widgets.get_mut(self.focus) {
                    if *sel + 1 < items.len() {
                        *sel += 1;
                        return Step::Redraw;
                    }
                }
                self.advance(false);
                Step::Redraw
            }
            kbd::KEY_UP => {
                if let Some(Widget::List { sel, .. }) = self.widgets.get_mut(self.focus) {
                    if *sel > 0 {
                        *sel -= 1;
                        return Step::Redraw;
                    }
                }
                self.advance(true);
                Step::Redraw
            }
            b'\n' | b'\r' => self.activate(),
            27 => Step::Close,
            _ => Step::Idle,
        }
    }

    pub fn set_title(&mut self, t: &str) {
        self.title = String::from(t);
    }

    /// Draw the widget stack into a client rectangle.
    ///
    /// The frame and title bar are the desktop's business, not the panel's: a
    /// panel that drew its own window could not be a window *on* something.
    pub fn draw_in(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        let mut y = client.y + PAD;
        let x = client.x + PAD;
        let w = client.w.saturating_sub(PAD * 2);
        for (i, widget) in self.widgets.iter().enumerate() {
            let h = widget.height();
            // Clip rather than negotiate: a stack that runs off the bottom is
            // a panel with too much in it, and drawing half a control is a
            // clearer symptom than silently rearranging the rest.
            if y + h > client.y + client.h {
                break;
            }
            // A control is only "focused" if the window is, so a desktop with
            // several panels does not show two selections at once.
            let focused = focused && i == self.focus;
            match widget {
                Widget::Label(t) => {
                    theme::text(fb, x, y, t, theme::TEXT, theme::FACE);
                }
                Widget::Sep => theme::separator(fb, x, y + GAP / 2, w),
                Widget::List { items, sel } => {
                    let r = Rect::new(x, y, w, h);
                    theme::well(fb, r, theme::FACE);
                    let inner = r.shrink(2);
                    for (j, (label, _)) in items.iter().enumerate() {
                        let row = Rect::new(inner.x, inner.y + j as u32 * ROW_H, inner.w, ROW_H);
                        if row.y + ROW_H > inner.y + inner.h {
                            break;
                        }
                        theme::list_row(fb, row, label, j == *sel, focused);
                    }
                }
                Widget::Button { label, .. } => {
                    let bw = (theme::text_w(label.len()) + PAD * 4).min(w);
                    theme::button(fb, Rect::new(x, y, bw, h - GAP), label, focused, false);
                }
            }
            y += h + GAP;
        }
    }

    /// One line per widget on the serial port, with the focused one marked.
    ///
    /// Serial and never `kprintln!`: the panel is on the framebuffer, and
    /// echoing it into the console would scroll the very window it is drawn
    /// inside. This transcript is what a headless run has instead of a
    /// screenshot, and it is the oracle the tests read.
    fn trace(&self, event: &str) {
        crate::serial_println!("[ui] {} \"{}\"", event, self.title);
        for (i, w) in self.widgets.iter().enumerate() {
            let (mark, text, sel) = w.trace(i == self.focus);
            crate::serial_println!("[ui] {} {} {}", mark, text, sel);
        }
    }
}

/// Parse `tab,tab,down,enter,esc` into keystrokes.
///
/// Names rather than raw bytes because the raw bytes are exactly what cannot
/// be sent. Arrow keys are defined as bytes above 0x7F that only the i8042
/// decoder ever produces; serial does no ANSI decoding, so an arrow has no
/// wire representation at all. A name has one everywhere, and survives being
/// passed through PowerShell as a single argument.
pub fn parse_keys(spec: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let p = part.trim();
        let k = match p {
            "tab" => b'\t',
            "backtab" | "shift-tab" => kbd::KEY_BACKTAB,
            "alttab" | "alt-tab" => kbd::KEY_ALTTAB,
            "sysmenu" | "alt-space" => kbd::KEY_SYSMENU,
            "menu" | "alt" => kbd::KEY_MENU,
            "up" => kbd::KEY_UP,
            "down" => kbd::KEY_DOWN,
            "left" => kbd::KEY_LEFT,
            "right" => kbd::KEY_RIGHT,
            "home" => kbd::KEY_HOME,
            "end" => kbd::KEY_END,
            "enter" | "return" => b'\n',
            "esc" | "escape" => 27,
            "space" => b' ',
            other => {
                // A single literal character, so a panel that takes typed
                // input is drivable without inventing a name per key.
                match other.as_bytes() {
                    [c] => *c,
                    // Anything else is a misspelt name. Silence here cost a
                    // test run that reported focus had not moved when in fact
                    // the keystroke had been dropped on the floor.
                    _ => {
                        crate::serial_println!("[ui] unknown key name {:?}", other);
                        continue;
                    }
                }
            }
        };
        out.push(k);
    }
    out
}

impl Panel {
    /// Width and height the content wants, including chrome.
    ///
    /// Sized to what is in it rather than to a fraction of the screen. A dialog
    /// that is always three fifths of the panel is a grey box with some
    /// controls in the top corner, which is the one thing a 3.1 dialog never
    /// looked like -- they hug their content, and that is most of why they read
    /// as objects rather than as regions.
    pub fn preferred(&self) -> (u32, u32) {
        let mut h = PAD * 2;
        let mut text_cols = self.title.len() + 4;
        for w in &self.widgets {
            h += w.height() + GAP;
            let cols = match w {
                Widget::Label(t) => t.len(),
                Widget::Sep => 0,
                Widget::List { items, .. } => {
                    items.iter().map(|i| i.0.len()).max().unwrap_or(0) + 2
                }
                Widget::Button { label, .. } => label.len() + 4,
            };
            text_cols = text_cols.max(cols);
        }
        (
            theme::text_w(text_cols) + PAD * 4 + theme::FRAME * 2,
            h + theme::TITLE_H + 2 + theme::FRAME * 2,
        )
    }
}

/// Where a panel sits: centred, sized to its content, clipped to the screen.
fn frame_for(fb: &Framebuffer, panel: &Panel) -> Rect {
    let (mut w, mut h) = panel.preferred();
    w = w.min(fb.width());
    h = h.min(fb.height());
    Rect::new((fb.width() - w) / 2, (fb.height() - h) / 2, w, h)
}

/// The launcher.
///
/// Every entry is a shell command, which is the point: the GUI is a second way
/// to reach the system, not a second system. When the model learns to compose
/// panels it will be writing this shape, so it is deliberately made of nothing
/// a decoder could not emit -- a title, a list of (label, command) pairs, and
/// buttons.
/// Panels by name, so a menu item or a shell command can open one without
/// every caller knowing how each is built.
pub fn panel_named(name: &str) -> Option<Panel> {
    match name {
        "programs" => Some(program_manager()),
        "status" => Some(status_panel()),
        _ => None,
    }
}

/// A second window worth opening, so the window manager has something to
/// manage. Every entry is a command, exactly as in the launcher.
pub fn status_panel() -> Panel {
    let entries: [(&str, &str); 6] = [
        ("Uptime and tasks", "tasks"),
        ("Heap", "mem"),
        ("Interfaces", "net"),
        ("Disks", "storage"),
        ("Certificates", "trust"),
        ("Attention window", "window"),
    ];
    let items = entries
        .iter()
        .map(|(label, cmd)| (String::from(*label), Action::Run(String::from(*cmd))))
        .collect();
    Panel::new(
        "System",
        alloc::vec![
            Widget::List { items, sel: 0 },
            Widget::Button { label: String::from("Close"), action: Action::Close },
        ],
    )
}

pub fn program_manager() -> Panel {
    let entries: [(&str, &str); 8] = [
        ("System status", "status"),
        ("Memory", "mem"),
        ("Network", "net"),
        ("Storage", "storage"),
        ("Namespace", "tree /"),
        ("Model", "ai"),
        ("Attention window", "window"),
        ("Self-test: tensor", "tensor"),
    ];
    let items = entries
        .iter()
        .map(|(label, cmd)| (String::from(*label), Action::Run(String::from(*cmd))))
        .collect();
    Panel::new(
        "Aperture Program Manager",
        alloc::vec![
            Widget::Label(String::from("Arrows select, Enter runs")),
            Widget::Sep,
            Widget::List { items, sel: 0 },
            Widget::Button { label: String::from("Close"), action: Action::Close },
        ],
    )
}


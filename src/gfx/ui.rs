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
    /// Run this command with the panel's field text appended. `Apply("window")`
    /// on a panel whose field holds `4 128` runs `window 4 128`.
    Apply(String),
    /// Replace this window's panel with a browser rooted at a new path.
    ///
    /// Distinct from `Run` because navigating is not a command: it changes what
    /// the window *is showing*, and routing it through the shell would print a
    /// listing into the terminal instead of moving the browser.
    Browse(String),
    Close,
    None,
}

pub enum Widget {
    Label(String),
    Sep,
    List { items: Vec<(String, Action)>, sel: usize },
    /// An editable line. `cursor` is a byte index, which is the same thing the
    /// shell's own editor tracks.
    ///
    /// `submit` is a *template*: its payload is a prefix, and activating the
    /// field appends what was typed. That is what lets one field mean "go to
    /// this path" in a browser and "set the window to these numbers" in
    /// settings without either panel needing its own key handling.
    Field { name: String, text: String, cursor: usize, submit: Action },
    Button { label: String, action: Action },
}

impl Widget {
    fn height(&self) -> u32 {
        match self {
            Widget::Label(_) => TEXT_H + GAP,
            Widget::Sep => GAP * 2,
            Widget::List { items, .. } => items.len() as u32 * ROW_H + 8,
            Widget::Field { .. } => TEXT_H + 14,
            Widget::Button { .. } => BTN_H,
        }
    }

    fn focusable(&self) -> bool {
        matches!(self, Widget::List { .. } | Widget::Field { .. } | Widget::Button { .. })
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
            Widget::Field { text, cursor, .. } => (mark, text.as_str(), *cursor),
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
                Action::Apply(prefix) => Step::Do(self.substituted(prefix)),
                a => Step::Do(a.clone()),
            },
            // Enter in a field submits it, through whatever template it
            // carries.
            Some(Widget::Field { text, submit, .. }) => match submit {
                Action::Apply(prefix) => Step::Do(self.substituted(prefix)),
                Action::Browse(prefix) => {
                    Step::Do(Action::Browse(alloc::format!("{}{}", prefix, text)))
                }
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

    /// A command with the field text appended, or bare if there is no field.
    fn substituted(&self, prefix: &str) -> Action {
        match self.field_text() {
            Some(t) if !t.is_empty() => Action::Run(alloc::format!("{} {}", prefix, t)),
            _ => Action::Run(String::from(prefix)),
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
            kbd::KEY_LEFT | kbd::KEY_RIGHT | kbd::KEY_HOME | kbd::KEY_END | 8
            | kbd::KEY_DELETE => self.edit(k),
            b'\n' | b'\r' => self.activate(),
            27 => Step::Close,
            // Printable text goes to a focused field and nowhere else. A panel
            // with no field ignores typing rather than inventing a meaning for
            // it -- type-ahead selection in a list is a feature, and guessing
            // at it here would make Escape and Enter behave differently
            // depending on what was typed before them.
            0x20..=0x7E => self.edit(k),
            _ => Step::Idle,
        }
    }

    /// Text editing inside a focused `Field`. Anything else leaves it alone.
    fn edit(&mut self, k: u8) -> Step {
        let Some(Widget::Field { text, cursor, .. }) = self.widgets.get_mut(self.focus) else {
            // Left and Right still have to do something sensible elsewhere:
            // in a list they mean nothing, so they mean nothing here.
            return Step::Idle;
        };
        match k {
            kbd::KEY_LEFT => *cursor = cursor.saturating_sub(1),
            kbd::KEY_RIGHT => *cursor = (*cursor + 1).min(text.len()),
            kbd::KEY_HOME => *cursor = 0,
            kbd::KEY_END => *cursor = text.len(),
            8 => {
                if *cursor > 0 {
                    *cursor -= 1;
                    text.remove(*cursor);
                }
            }
            kbd::KEY_DELETE => {
                if *cursor < text.len() {
                    text.remove(*cursor);
                }
            }
            c => {
                text.insert(*cursor, c as char);
                *cursor += 1;
            }
        }
        Step::Redraw
    }

    /// The text of the first field, for a panel that has one.
    pub fn field_text(&self) -> Option<&str> {
        self.widgets.iter().find_map(|w| match w {
            Widget::Field { text, .. } => Some(text.as_str()),
            _ => None,
        })
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
                Widget::Field { name, text, cursor, .. } => {
                    let cap = theme::text_w(name.len() + 1);
                    theme::text(fb, x, y + 4, name, theme::TEXT, theme::FACE);
                    let well = Rect::new(x + cap, y, w.saturating_sub(cap), h - 2);
                    theme::well(fb, well, theme::HILIGHT);
                    let inner = well.shrink(3);
                    // Scroll so the caret stays visible, exactly as the shell's
                    // own line editor does -- a field narrower than its
                    // contents is the normal case, not an error.
                    let room = (inner.w / (font::GLYPH_W * theme::CHROME_SCALE)) as usize;
                    if room > 0 {
                        let off = cursor.saturating_sub(room.saturating_sub(1));
                        let end = (off + room).min(text.len());
                        theme::text(fb, inner.x, inner.y, &text[off..end], theme::TEXT, theme::HILIGHT);
                        if focused {
                            let cx = inner.x
                                + (cursor - off) as u32 * font::GLYPH_W * theme::CHROME_SCALE;
                            fb.rect(cx, inner.y, 2, TEXT_H, theme::TEXT);
                        }
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
            "taskbar" | "ctrl-esc" | "start" => kbd::KEY_TASKBAR,
            "up" => kbd::KEY_UP,
            "down" => kbd::KEY_DOWN,
            "left" => kbd::KEY_LEFT,
            "right" => kbd::KEY_RIGHT,
            "home" => kbd::KEY_HOME,
            "end" => kbd::KEY_END,
            "enter" | "return" => b'\n',
            "esc" | "escape" => 27,
            // Enternet uses backspace for "back", and a browser whose history
            // cannot be driven headlessly is a browser whose history is never
            // tested. Same reason the arrows have names.
            "backspace" | "bksp" | "back" => 8,
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
                // A field wants room to type in, not just room for its caption.
                Widget::Field { name, .. } => name.len() + 28,
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
        "files" => Some(file_browser("/")),
        "settings" => Some(settings("net")),
        _ => None,
    }
}

/// Resolve a `kind:argument` route to a titled panel.
///
/// One resolver for every in-place swap rather than one per app. The desktop
/// replaces a window's content without knowing what kind of app is in it, and
/// an app that wanted its own navigation would have to teach the desktop about
/// itself -- which is how a window manager ends up knowing what a file is.
pub fn panel_for_route(route: &str) -> Option<(String, Panel)> {
    let (kind, arg) = route.split_once(':').unwrap_or((route, ""));
    match kind {
        "files" => {
            let path = if arg.is_empty() { "/" } else { arg };
            Some((alloc::format!("Files -- {}", path), file_browser(path)))
        }
        "set" => Some((String::from("Settings"), settings(arg))),
        _ => None,
    }
}

/// Settings, one page at a time.
///
/// Every control is a shell command, exactly as in the launcher. That rule
/// matters most here: a settings program able to change something the command
/// line could not would be a second way to configure the system, and the two
/// would disagree within a week.
///
/// Values are shown by *running* the relevant command into the terminal rather
/// than mirrored into labels. Mirroring means a cache, and a settings window
/// showing a stale value is worse than one showing none.
pub fn settings(page: &str) -> Panel {
    let nav = |sel: usize| Widget::List {
        items: alloc::vec![
            (String::from("Network"), Action::Browse(String::from("set:net"))),
            (String::from("Model"), Action::Browse(String::from("set:model"))),
            (String::from("System"), Action::Browse(String::from("set:sys"))),
        ],
        sel,
    };

    let (sel, mut body) = match page {
        "model" => (
            1usize,
            alloc::vec![
                Widget::Label(String::from("Attention: sinks and recent")),
                Widget::Field {
                    name: String::from("window"),
                    text: String::from("4 512"),
                    cursor: 5,
                    submit: Action::Apply(String::from("window")),
                },
                Widget::Button {
                    label: String::from("Apply window"),
                    action: Action::Apply(String::from("window")),
                },
                Widget::Sep,
                Widget::Button {
                    label: String::from("Model status"),
                    action: Action::Run(String::from("status")),
                },
                Widget::Button {
                    label: String::from("Show window"),
                    action: Action::Run(String::from("window")),
                },
                Widget::Button {
                    label: String::from("Contexts"),
                    action: Action::Run(String::from("ctx")),
                },
                Widget::Button {
                    label: String::from("Refit router"),
                    action: Action::Run(String::from("fit")),
                },
            ],
        ),
        "sys" => (
            2usize,
            alloc::vec![
                Widget::Label(String::from("Snapshots, memory, power")),
                Widget::Button {
                    label: String::from("Snapshot now"),
                    action: Action::Run(String::from("snap")),
                },
                Widget::Button {
                    label: String::from("Autosnap"),
                    action: Action::Run(String::from("autosnap")),
                },
                Widget::Button {
                    label: String::from("Memory"),
                    action: Action::Run(String::from("mem")),
                },
                Widget::Button {
                    label: String::from("Tasks"),
                    action: Action::Run(String::from("tasks")),
                },
                Widget::Button {
                    label: String::from("Storage"),
                    action: Action::Run(String::from("store")),
                },
                Widget::Sep,
                Widget::Button {
                    label: String::from("Reboot"),
                    action: Action::Run(String::from("reboot")),
                },
            ],
        ),
        // Network is the default: it is the page most likely to be wrong.
        _ => (
            0usize,
            alloc::vec![
                Widget::Label(String::from("Interfaces, DHCP, names, trust")),
                Widget::Field {
                    name: String::from("host"),
                    text: String::from("discord.com"),
                    cursor: 11,
                    submit: Action::Apply(String::from("dns")),
                },
                Widget::Button {
                    label: String::from("Resolve"),
                    action: Action::Apply(String::from("dns")),
                },
                Widget::Sep,
                Widget::Button {
                    label: String::from("Interfaces"),
                    action: Action::Run(String::from("net")),
                },
                Widget::Button {
                    label: String::from("Renew DHCP"),
                    action: Action::Run(String::from("dhcp")),
                },
                Widget::Button {
                    label: String::from("Wireless"),
                    action: Action::Run(String::from("wifi")),
                },
                Widget::Button {
                    label: String::from("Certificates"),
                    action: Action::Run(String::from("trust")),
                },
            ],
        ),
    };

    let mut widgets = alloc::vec![nav(sel), Widget::Sep];
    widgets.append(&mut body);
    widgets.push(Widget::Button { label: String::from("Close"), action: Action::Close });
    Panel::new("Settings", widgets)
}

/// A browser over the namespace, rooted at `path`.
///
/// Rebuilt whole on every navigation rather than mutated in place. A panel is
/// data -- the same data the model will one day be writing -- so "go to a
/// different directory" is "make the panel for that directory", and there is no
/// second code path that edits a browser into a different browser.
///
/// Directories carry a trailing slash and a child count, files their size, so
/// the two are distinguishable without colour. Colour is how the console tells
/// them apart, and the same information should not depend on it twice.
pub fn file_browser(path: &str) -> Panel {
    let path = if path.is_empty() { "/" } else { path };
    let entries = crate::sysbox::listing(path);

    let mut items: Vec<(String, Action)> = Vec::new();
    // Parent first, and only when there is one. An entry that navigates
    // nowhere is worse than an absent one.
    if path != "/" {
        let cut = path.trim_end_matches('/').rfind('/').unwrap_or(0);
        let parent = if cut == 0 { String::from("/") } else { String::from(&path[..cut]) };
        items.push((
            String::from(".."),
            Action::Browse(alloc::format!("files:{}", parent)),
        ));
    }
    for (name, is_dir, n) in &entries {
        let joined = if path == "/" {
            alloc::format!("/{}", name)
        } else {
            alloc::format!("{}/{}", path, name)
        };
        if *is_dir {
            items.push((
                alloc::format!("{}/  ({})", name, n),
                Action::Browse(alloc::format!("files:{}", joined)),
            ));
        } else {
            // Opening a file is a shell command, because printing it is what
            // `cat` already does and a second implementation would be a second
            // thing to keep right.
            items.push((
                alloc::format!("{}  {} B", name, n),
                Action::Run(alloc::format!("cat {}", joined)),
            ));
        }
    }
    if items.is_empty() {
        items.push((String::from("(empty)"), Action::None));
    }

    Panel::new(
        "Files",
        alloc::vec![
            Widget::Field {
                name: String::from("Path"),
                text: String::from(path),
                cursor: path.len(),
                submit: Action::Browse(String::from("files:")),
            },
            Widget::Sep,
            Widget::List { items, sel: 0 },
            Widget::Button { label: String::from("Close"), action: Action::Close },
        ],
    )
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


//! Widgets, keyboard focus, and the panel loop.
//!
//! `theme` draws; this decides. Every control is reachable by keyboard, and
//! the pointer is a second way in, never the only one: serial cannot inject
//! PS/2 packets, so a control that could only be clicked is a control
//! `drive.py` could never reach, and an interface that cannot be driven
//! headlessly does not get tested. `rects` is the one layout, shared by the
//! paint pass and the hit-test, so a control cannot highlight in one place
//! and press in another.
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
/// Character column the value in a `Status` row starts at.
/// Characters reserved for a status row's name, before its value starts.
///
/// Fourteen and not thirteen because thirteen clipped `PCI 8086:100e` on the
/// Settings/Network page to `PCI 8086:100` -- a row that had rendered
/// correctly for as long as the page existed, made to show a wrong-looking
/// three-digit device ID by the clipping added to stop names running into
/// their values. The marker below is the belt to this braces: no clip may ever
/// again read as a shorter number.
const STATUS_COL: usize = 14;

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

/// How a status line should read at a glance.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    /// Working.
    Ok,
    /// Works, but not the way it should.
    Warn,
    /// Will not work, and no amount of retrying changes that.
    Bad,
    /// A fact with no verdict attached.
    Plain,
}

impl Tone {
    fn color(self) -> super::Color {
        match self {
            Tone::Ok => theme::OK_TEXT,
            Tone::Warn => theme::WARN_TEXT,
            Tone::Bad => theme::BAD_TEXT,
            Tone::Plain => theme::TEXT,
        }
    }
}

/// Columns a `Note` wraps to.
///
/// A fixed count rather than the panel width, for the reason given on the
/// variant. Sized so the default settings window shows a full line and a
/// wider one merely leaves margin, which is the failure worth having.
const NOTE_COLS: usize = 44;

/// Wrap prose into a `Note`, breaking on spaces and never mid-word.
pub fn note(text: &str) -> Widget {
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    for word in text.split(' ') {
        if !cur.is_empty() && cur.chars().count() + 1 + word.chars().count() > NOTE_COLS {
            lines.push(core::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    Widget::Note(lines)
}

pub enum Widget {
    Label(String),
    Sep,
    /// A section title. Structure a page rather than running its controls
    /// together: an operator scanning for "the wireless part" finds a heading
    /// long before they find the third button down.
    Heading(String),
    /// One fact about the machine, as it is right now: a name, a value, and
    /// how worried to be.
    ///
    /// Read when the panel is built, never stored. Panels are rebuilt whole on
    /// every navigation, so this is a reading and not a cache -- the rule that
    /// keeps a settings window from showing a value the system stopped
    /// believing an hour ago.
    Status { name: String, value: String, tone: Tone },
    /// Explanatory prose, pre-wrapped, dimmed.
    ///
    /// Wrapped when the widget is made rather than when it is drawn, because
    /// height has to be known before a width is: the layout pass asks every
    /// widget how tall it is and only then hands out rectangles.
    Note(Vec<String>),
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
            Widget::Heading(_) => TEXT_H + GAP + 6,
            Widget::Status { .. } => TEXT_H + 4,
            Widget::Note(lines) => lines.len() as u32 * TEXT_H + GAP,
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
            Widget::Heading(t) => (mark, t.as_str(), 0),
            // The value, not the name: what a headless run needs to assert on
            // is what the machine reported, and the name is in the source.
            Widget::Status { value, .. } => (mark, value.as_str(), 0),
            Widget::Note(lines) => (mark, lines.first().map(|l| l.as_str()).unwrap_or(""), lines.len()),
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
    /// True when this panel opens with the caret in a text field.
    ///
    /// Such a panel exists to be typed into, so handing the keyboard back to
    /// the terminal is wrong twice over: the operator types into the wrong
    /// place, and raising the terminal covers the panel -- which for anything
    /// wider than the gap beside the terminal hides exactly the left edge where
    /// the buttons are.
    pub fn wants_typing(&self) -> bool {
        matches!(self.widgets.get(self.focus), Some(Widget::Field { .. }))
    }

    pub fn field_text(&self) -> Option<&str> {
        self.widgets.iter().find_map(|w| match w {
            Widget::Field { text, .. } => Some(text.as_str()),
            _ => None,
        })
    }

    pub fn set_title(&mut self, t: &str) {
        self.title = String::from(t);
    }

    /// Where each widget lands in a client rectangle, in stack order.
    ///
    /// The single source of layout: `draw_in` paints these rectangles and the
    /// pointer hit-tests them, so a control cannot highlight in one place and
    /// press in another. Clipping is part of the layout -- a widget that is
    /// not in this list is not on screen and must not be clickable.
    fn rects(&self, client: Rect) -> Vec<(usize, Rect)> {
        let mut out = Vec::new();
        let mut y = client.y + PAD;
        let x = client.x + PAD;
        let w = client.w.saturating_sub(PAD * 2);
        for (i, widget) in self.widgets.iter().enumerate() {
            let h = widget.height();
            if y + h > client.y + client.h {
                break;
            }
            out.push((i, Rect::new(x, y, w, h)));
            y += h + GAP;
        }
        out
    }

    /// The focusable widget under a point, for hover feedback.
    pub fn hover_at(&self, client: Rect, px: i32, py: i32) -> Option<usize> {
        self.rects(client).into_iter().find_map(|(i, r)| {
            let inside = px >= r.x as i32
                && py >= r.y as i32
                && px < (r.x + r.w) as i32
                && py < (r.y + r.h) as i32;
            (inside && self.widgets[i].focusable()).then_some(i)
        })
    }

    /// One pointer press. Focus moves to the control under the point, and the
    /// control answers exactly as it would to the keyboard: a button presses,
    /// a list row selects (and opens on a double press), a field takes the
    /// caret to the character that was hit.
    pub fn mouse(&mut self, client: Rect, px: i32, py: i32, double: bool) -> Step {
        let Some(i) = self.hover_at(client, px, py) else {
            return Step::Idle;
        };
        let r = self
            .rects(client)
            .into_iter()
            .find(|(j, _)| *j == i)
            .map(|(_, r)| r)
            .unwrap_or(client);
        self.focus = i;
        match &mut self.widgets[i] {
            Widget::Button { .. } => self.activate(),
            Widget::List { items, sel } => {
                let inner = r.shrink(2);
                let row = (py - inner.y as i32) / ROW_H as i32;
                if row < 0 || row as usize >= items.len() {
                    return Step::Redraw;
                }
                let was = *sel;
                *sel = row as usize;
                // A double press opens; so does a second press on the row that
                // is already selected, which is how a slow double-click still
                // works on a machine timing its clicks by TSC.
                if double && was == *sel {
                    self.activate()
                } else {
                    Step::Redraw
                }
            }
            Widget::Field { name, text, cursor, .. } => {
                let cap = theme::text_w(name.len() + 1);
                let well = Rect::new(r.x + cap, r.y, r.w.saturating_sub(cap), r.h - 2);
                let inner = well.shrink(3);
                let room = (inner.w / (font::GLYPH_W * theme::CHROME_SCALE)) as usize;
                if room > 0 {
                    // The same window the paint pass shows, so the caret lands
                    // on the character that was actually under the pointer.
                    let off = cursor.saturating_sub(room.saturating_sub(1));
                    let col = ((px - inner.x as i32).max(0) as u32
                        / (font::GLYPH_W * theme::CHROME_SCALE)) as usize;
                    *cursor = (off + col).min(text.len());
                }
                Step::Redraw
            }
            _ => Step::Idle,
        }
    }

    /// Wheel notches over the panel move the first list's selection.
    pub fn wheel(&mut self, notches: i32) -> bool {
        for w in &mut self.widgets {
            if let Widget::List { items, sel } = w {
                let n = items.len();
                if n == 0 {
                    return false;
                }
                let was = *sel;
                *sel = if notches > 0 {
                    (*sel + (notches as usize)).min(n - 1)
                } else {
                    sel.saturating_sub((-notches) as usize)
                };
                return *sel != was;
            }
        }
        false
    }

    /// Draw the widget stack into a client rectangle.
    ///
    /// The frame and title bar are the desktop's business, not the panel's: a
    /// panel that drew its own window could not be a window *on* something.
    /// `hover` is the widget the pointer is over, drawn ready-to-press.
    pub fn draw_in(&self, fb: &Framebuffer, client: Rect, focused: bool, hover: Option<usize>) {
        let win_focused = focused;
        for (i, r) in self.rects(client) {
            let widget = &self.widgets[i];
            let (x, y, w, h) = (r.x, r.y, r.w, r.h);
            // A control is only "focused" if the window is, so a desktop with
            // several panels does not show two selections at once. The pointer
            // is one pointer, so hover needs no such gate.
            let focused = (win_focused && i == self.focus) || hover == Some(i);
            match widget {
                Widget::Label(t) => {
                    theme::text(fb, x, y, t, theme::TEXT, theme::FACE);
                }
                Widget::Sep => theme::separator(fb, x, y + GAP / 2, w),
                Widget::Heading(t) => {
                    theme::text(fb, x, y + 4, t, theme::TEXT, theme::FACE);
                    let uw = theme::text_w(t.chars().count()).min(w);
                    fb.rect(x, y + 4 + TEXT_H + 2, uw, 2, theme::APERTURE);
                }
                Widget::Status { name, value, tone } => {
                    // One column for every row on the page, so the values line
                    // up and can be read down rather than hunted for.
                    let col = theme::text_w(STATUS_COL).min(w / 2);
                    // Clipped to its column, with a space kept.
                    //
                    // A name longer than the column used to run straight into
                    // its own value: "Trained length" and "512 positions" drew
                    // as "Trained lengt512 positions", which reads as a
                    // corrupted number rather than as a layout that ran out of
                    // room. Every panel shares this column, so the fix belongs
                    // here and not in the labels -- otherwise the next long
                    // name reintroduces it.
                    let room = (col / theme::text_w(1)).saturating_sub(1) as usize;
                    // By character boundary, not by byte. `&name[..room]`
                    // panics if it lands mid-codepoint, and a panic in the
                    // paint pass takes the desktop with it -- an expensive way
                    // to discover that a label had an accent in it.
                    let cut = name
                        .char_indices()
                        .nth(room)
                        .map(|(i, _)| i)
                        .unwrap_or(name.len());
                    // A clip says so. Silently dropping the tail turns
                    // `PCI 8086:100e` into `PCI 8086:100`, which does not look
                    // truncated -- it looks like a different device. One
                    // character of the budget goes to admitting it.
                    let mut clipped = String::new();
                    let shown: &str = if cut < name.len() {
                        let keep = name
                            .char_indices()
                            .nth(room.saturating_sub(1))
                            .map(|(i, _)| i)
                            .unwrap_or(cut);
                        clipped.push_str(&name[..keep]);
                        clipped.push('~');
                        &clipped
                    } else {
                        &name[..cut]
                    };
                    theme::text(fb, x, y, shown, theme::TEXT_DIM, theme::FACE);
                    theme::text(fb, x + col, y, value, tone.color(), theme::FACE);
                }
                Widget::Note(lines) => {
                    for (j, line) in lines.iter().enumerate() {
                        theme::text(fb, x, y + j as u32 * TEXT_H, line, theme::TEXT_DIM, theme::FACE);
                    }
                }
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
                    let bw = (theme::text_w_of(label) + PAD * 4).min(w);
                    theme::button(fb, Rect::new(x, y, bw, h - GAP), label, focused, false);
                }
            }
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
            "taskbar" | "ctrl-esc" => kbd::KEY_TASKBAR,
            // "start" moved here from the taskbar, where it was a lie: it
            // focused the bar, and anybody typing it means the menu.
            "start" | "startmenu" | "win" => kbd::KEY_STARTMENU,
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
            // "ctrl-s" and friends: the byte the driver produces for the
            // chord, so ^S in Write is drivable over serial.
            ctrl if ctrl.len() == 6
                && ctrl.starts_with("ctrl-")
                && ctrl.as_bytes()[5].is_ascii_alphabetic() =>
            {
                ctrl.as_bytes()[5].to_ascii_uppercase() - b'A' + 1
            }
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
                Widget::Heading(t) => t.len(),
                // Both columns, so a long value widens the window instead of
                // being clipped by it.
                Widget::Status { value, .. } => STATUS_COL + value.len(),
                Widget::Note(lines) => lines.iter().map(|l| l.len()).max().unwrap_or(0),
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
        "memory" | "mem" => Some(memory_panel()),
        "tasks" => Some(tasks_panel()),
        "storage" | "store" => Some(storage_panel()),
        "attention" | "window" => Some(attention_panel()),
        "diag" | "diagnostics" => Some(diagnostics_panel()),
        "files" => Some(file_browser("/")),
        "settings" | "network" => Some(settings("net")),
        // The settings pages by name, so the shell can open any of them
        // directly. `win open wifi` is the one that matters: a page whose
        // whole job is explaining why something does not work is useless if
        // reaching it needs three keystrokes nobody documented.
        "wifi" | "wireless" => Some(settings("wifi")),
        "search" => Some(search_panel()),
        "model" => Some(settings("model")),
        "system" => Some(settings("sys")),
        // `win open app:todo`. Guarded, so it goes before the bare arm it
        // would otherwise sit behind and never be reached from.
        n if n.starts_with("app:") => crate::app::panel(&n[4..]).map(|(_, p)| p),
        _ => None,
    }
}

pub fn panel_for_route(route: &str) -> Option<(String, Panel)> {
    let (kind, arg) = route.split_once(':').unwrap_or((route, ""));
    match kind {
        "files" => {
            let path = if arg.is_empty() { "/" } else { arg };
            Some((alloc::format!("Files -- {}", path), file_browser(path)))
        }
        "set" => Some((String::from("Settings"), settings(arg))),
        "programs" => Some((String::from("Program Manager"), program_manager())),
        // Routed as well as named, so each rebuilds itself when the window is
        // refreshed. These are readings of live state -- a memory window that
        // kept its first answer would be worse than no window.
        "memory" => Some((String::from("Memory"), memory_panel())),
        "tasks" => Some((String::from("Tasks"), tasks_panel())),
        "storage" => Some((String::from("Storage"), storage_panel())),
        "attention" => Some((String::from("Attention"), attention_panel())),
        // Routed, so running a suite from the window rebuilds it and the new
        // verdict is on screen without anyone reopening anything.
        "diag" => Some((String::from("Diagnostics"), diagnostics_panel())),
        // An application, read from the namespace and parsed under the stored
        // gate. This is the one route whose contents the machine may write.
        "app" => crate::app::panel(arg),
        // One verb, because the query box and the offer are one surface. The
        // arg is what was typed and empty means nothing has been yet, so
        // `refresh_routed` rebuilds whichever of the two is showing without
        // needing to know which.
        "search" => Some((
            String::from("Search"),
            if arg.is_empty() { search_panel() } else { offer_panel(arg) },
        )),
        _ => None,
    }
}

/// Type a name, press Enter.
///
/// A `Field` and nothing else. The whole path already exists --
/// `Panel::substituted` appends what was typed, the action reaches `PENDING`,
/// and the shell runs it -- so this needs no new hit-testing, which is the part
/// of this tree that has cost the most to get wrong.
///
/// The in-menu query row that Windows actually has is a separate piece of work:
/// it means `Mode::Start` grows a text row, `start_menu_rect` grows with it and
/// `dropdown_item_at` offsets by one. Worth having, and worth landing on its
/// own so that when something breaks it is obvious which half did it.
pub fn search_panel() -> Panel {
    Panel::new(
        "Search",
        alloc::vec![
            Widget::Heading(String::from("What are you looking for?")),
            Widget::Field {
                name: String::from("name"),
                text: String::new(),
                cursor: 0,
                submit: Action::Apply(String::from("open")),
            },
            Widget::Button {
                label: String::from("Find"),
                action: Action::Apply(String::from("open")),
            },
            note("An application opens. A command runs. Anything else, and the machine offers to write it."),
            Widget::Sep,
            Widget::Button { label: String::from("Close"), action: Action::Close },
        ],
    )
}

/// Nothing by that name exists. Offer to write one.
///
/// An offer and not an action, which is the one place this deliberately parts
/// company with the demos it is answering. Writing an application holds the
/// model for minutes, and a keystroke that silently starts that is a keystroke
/// that gets regretted -- there is one engine here and one operator, and they
/// are usually the same person waiting for both.
pub fn offer_panel(query: &str) -> Panel {
    let name: String = query
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect::<String>()
        .to_ascii_lowercase();
    if name.is_empty() {
        return Panel::new(
            "Search",
            alloc::vec![
                Widget::Heading(String::from("Nothing to look for")),
                note("A name is letters, digits and dashes."),
                Widget::Button { label: String::from("Close"), action: Action::Close },
            ],
        );
    }
    Panel::new(
        "Search",
        alloc::vec![
            Widget::Heading(alloc::format!("No application called {}", name)),
            note("The machine can write one. It holds the model for minutes, and leaves a draft rather than an installed program."),
            Widget::Status {
                name: String::from("then"),
                value: alloc::format!("app try {}", name),
                tone: Tone::Plain,
            },
            Widget::Sep,
            // Cancel first, and not for tidiness: focus lands on the first
            // focusable widget, the Enter that submitted the query is still
            // under the operator's finger, and the other button holds the
            // model for minutes. Reaching it should take a deliberate Tab.
            Widget::Button { label: String::from("Cancel"), action: Action::Close },
            Widget::Button {
                label: String::from("Write it"),
                action: Action::Run(alloc::format!("author {} {}", name, query)),
            },
        ],
    )
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
fn ip_text(a: crate::net::Ipv4) -> String {
    alloc::format!("{}.{}.{}.{}", a[0], a[1], a[2], a[3])
}

/// The interface carrying the default route, if anything is.
fn default_iface() -> Option<usize> {
    crate::net::route([8, 8, 8, 8])
}

/// Live status rows for the connection an operator actually has.
///
/// Read here, at build time, and never stored. Panels are rebuilt whole on
/// every navigation, so this is a reading rather than a cache, and the rule
/// that a settings window must not show a value the system stopped believing
/// still holds.
fn connection_rows() -> Vec<Widget> {
    let mut out = Vec::new();
    let Some(n) = default_iface() else {
        out.push(Widget::Status {
            name: String::from("Status"),
            value: String::from("Not connected"),
            tone: Tone::Bad,
        });
        out.push(note(
            "No interface has a route off this machine. Attach a cable, then renew DHCP below.",
        ));
        return out;
    };
    let ifaces = crate::net::ifaces();
    let usable = ifaces[n].usable();
    let i = &ifaces[n];
    let kind = i.nic.as_ref().map(|d| d.kind().name()).unwrap_or("unknown");
    out.push(Widget::Status {
        name: String::from("Status"),
        value: String::from(if usable { "Connected" } else { "Link down" }),
        tone: if usable { Tone::Ok } else { Tone::Bad },
    });
    out.push(Widget::Status {
        name: String::from("Adapter"),
        value: alloc::format!("{} ({})", i.name, kind),
        tone: Tone::Plain,
    });
    let bits: u32 = i.netmask.iter().map(|b| b.count_ones()).sum();
    out.push(Widget::Status {
        name: String::from("Address"),
        value: alloc::format!("{}/{}", ip_text(i.ip), bits),
        tone: if i.ip == crate::net::UNSPECIFIED { Tone::Warn } else { Tone::Plain },
    });
    out.push(Widget::Status {
        name: String::from("Gateway"),
        value: ip_text(i.gateway),
        tone: Tone::Plain,
    });
    out.push(Widget::Status {
        name: String::from("DNS"),
        value: ip_text(i.dns),
        tone: if i.dns == crate::net::UNSPECIFIED { Tone::Warn } else { Tone::Plain },
    });
    out
}

/// Every adapter slot, present or not.
fn adapter_rows() -> Vec<Widget> {
    let mut out = Vec::new();
    let def = default_iface();
    let ifaces = crate::net::ifaces();
    for n in 0..ifaces.len() {
        let present = ifaces[n].present();
        let usable = ifaces[n].usable();
        let i = &ifaces[n];
        let (value, tone) = if !present {
            (String::from("not present"), Tone::Plain)
        } else if usable {
            let mark = if def == Some(n) { "  (default route)" } else { "" };
            (alloc::format!("up{}", mark), Tone::Ok)
        } else if i.up {
            (String::from("no link"), Tone::Warn)
        } else {
            (String::from("down"), Tone::Warn)
        };
        out.push(Widget::Status { name: String::from(i.name), value, tone });
    }
    out
}

/// Every network part the machine has, and what drives it.
///
/// Separate from the adapter rows above, which describe the three interface
/// slots the stack offers. This describes the silicon. They differ in exactly
/// the case worth showing: a wireless part that is present, named, and bound
/// to nothing.
fn hardware_rows() -> Vec<Widget> {
    let hw = crate::net::wifi::hardware();
    if hw.is_empty() {
        return alloc::vec![note("No network controller found on PCI. USB is only listed after an enumeration.")];
    }
    let mut out = Vec::new();
    for h in &hw {
        let (value, tone) = match h.driver {
            Some(d) => (alloc::format!("{}  ({})", h.what, d), Tone::Ok),
            None => (alloc::format!("{}  no driver", h.what), Tone::Warn),
        };
        out.push(Widget::Status {
            name: alloc::format!("{} {:04x}:{:04x}", h.bus, h.vendor, h.device),
            value,
            tone,
        });
    }
    out
}

/// The wireless page.
///
/// Built from what `wifi::scan` actually answers. When it can list networks
/// this renders the list, the password field and the connect button; when it
/// cannot, it renders the reason. What it never does is render an empty list,
/// which reads as "the router is off" and sends the operator to debug the
/// wrong machine -- the exact struggle this page exists to end.
fn wifi_rows() -> Vec<Widget> {
    use crate::net::wifi::{self, Adapter};
    let mut out = alloc::vec![Widget::Heading(String::from("Wireless"))];

    // Asked once. Both buses, in the order something usable could be on them.
    let (bus, name, id) = match wifi::adapter() {
        Adapter::Usb { vendor, device, what } => {
            ("USB", String::from(what), Some((vendor, device)))
        }
        Adapter::Pci { vendor, device, what } => {
            ("PCI", String::from(what), Some((vendor, device)))
        }
        Adapter::None => ("", String::from("none on PCI or USB"), None),
        Adapter::PciOnlyChecked => ("", String::from("USB not enumerated yet"), None),
    };
    out.push(Widget::Status {
        name: String::from("Adapter"),
        value: name,
        tone: Tone::Plain,
    });
    if let Some((vendor, device)) = id {
        out.push(Widget::Status {
            name: String::from("Found on"),
            value: alloc::format!("{}  {:04x}:{:04x}", bus, vendor, device),
            tone: Tone::Plain,
        });
    }

    match wifi::scan() {
        Ok(nets) => {
            out.push(Widget::Status {
                name: String::from("State"),
                value: alloc::format!("{} found", nets.len()),
                tone: Tone::Ok,
            });
            out.push(Widget::Sep);
            out.push(Widget::Heading(String::from("Networks")));
            let items: Vec<(String, Action)> = nets
                .iter()
                .map(|n| {
                    let mut label = alloc::format!("{}  ", n.ssid);
                    for b in 0..4 {
                        label.push(if b < wifi::bars(n.rssi) { '|' } else { '.' });
                    }
                    if n.secured {
                        label.push_str("  secured");
                    }
                    (label, Action::Apply(alloc::format!("wifi join {}", n.ssid)))
                })
                .collect();
            out.push(Widget::List { items, sel: 0 });
            out.push(Widget::Field {
                name: String::from("password"),
                text: String::new(),
                cursor: 0,
                submit: Action::Apply(String::from("wifi join")),
            });
            out.push(Widget::Button {
                label: String::from("Connect"),
                action: Action::Apply(String::from("wifi join")),
            });
        }
        Err(why) => {
            out.push(Widget::Status {
                name: String::from("State"),
                value: String::from("Cannot scan"),
                tone: Tone::Bad,
            });
            out.push(note(why));
            out.push(Widget::Sep);
            out.push(Widget::Heading(String::from("What does work")));
            out.push(note(
                "Wired ethernet, with an address from DHCP. WPA2 is implemented here and passes its own tests, so what is missing is the radio and not the security.",
            ));
            // Offered on every arm, not just the not-yet-enumerated one: it is
            // also how an adapter plugged in after boot gets noticed.
            out.push(Widget::Button {
                label: String::from("Scan USB"),
                action: Action::Run(String::from("usb")),
            });
            out.push(Widget::Button {
                label: String::from("Wireless detail"),
                action: Action::Run(String::from("wifi")),
            });
        }
    }
    out
}

pub fn settings(page: &str) -> Panel {
    let nav = |sel: usize| Widget::List {
        items: alloc::vec![
            (String::from("Network"), Action::Browse(String::from("set:net"))),
            (String::from("Wi-Fi"), Action::Browse(String::from("set:wifi"))),
            (String::from("Model"), Action::Browse(String::from("set:model"))),
            (String::from("System"), Action::Browse(String::from("set:sys"))),
            (String::from("Programs"), Action::Browse(String::from("programs:"))),
        ],
        sel,
    };

    let (sel, mut body) = match page {
        "wifi" => (1usize, wifi_rows()),
        "model" => (
            2usize,
            alloc::vec![
                Widget::Heading(String::from("Attention")),
                note("Sinks kept from the start of the context and recent tokens kept from the end. Everything between them is dropped."),
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
                Widget::Heading(String::from("Inspect")),
                Widget::Button {
                    label: String::from("Model status"),
                    action: Action::Run(String::from("win open status")),
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
            3usize,
            alloc::vec![
                Widget::Heading(String::from("State")),
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
                    action: Action::Run(String::from("win open memory")),
                },
                Widget::Button {
                    label: String::from("Tasks"),
                    action: Action::Run(String::from("win open tasks")),
                },
                Widget::Button {
                    label: String::from("Storage"),
                    action: Action::Run(String::from("win open storage")),
                },
                Widget::Sep,
                Widget::Heading(String::from("Power")),
                note("Both ask the firmware first, and reset the machine directly if it declines."),
                Widget::Button {
                    label: String::from("Restart"),
                    action: Action::Run(String::from("reboot")),
                },
                Widget::Button {
                    label: String::from("Shut down"),
                    action: Action::Run(String::from("shutdown")),
                },
            ],
        ),
        // Network is the default: it is the page most likely to be wrong.
        _ => {
            let mut v = alloc::vec![Widget::Heading(String::from("Connection"))];
            v.append(&mut connection_rows());
            v.push(Widget::Sep);
            v.push(Widget::Heading(String::from("Adapters")));
            v.append(&mut adapter_rows());
            v.push(Widget::Sep);
            v.push(Widget::Heading(String::from("Hardware")));
            v.append(&mut hardware_rows());
            v.push(Widget::Sep);
            v.push(Widget::Heading(String::from("Actions")));
            v.push(Widget::Button {
                label: String::from("Renew DHCP"),
                action: Action::Run(String::from("dhcp")),
            });
            v.push(Widget::Button {
                label: String::from("Interface detail"),
                action: Action::Run(String::from("net")),
            });
            v.push(Widget::Button {
                label: String::from("Certificates"),
                action: Action::Run(String::from("trust")),
            });
            v.push(Widget::Sep);
            v.push(Widget::Heading(String::from("Name lookup")));
            v.push(Widget::Field {
                name: String::from("host"),
                text: String::from("discord.com"),
                cursor: 11,
                submit: Action::Apply(String::from("dns")),
            });
            v.push(Widget::Button {
                label: String::from("Resolve"),
                action: Action::Apply(String::from("dns")),
            });
            (0usize, v)
        }
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
        // A command called storage never existed; disk is the controller and
        // namespace report the label promises.
        ("Disks", "disk"),
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

/// Bytes, rendered the way a person reads them.
fn human(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        alloc::format!("{}.{} GiB", bytes / (1024 * 1024 * 1024), (bytes / (1024 * 1024) % 1024) * 10 / 1024)
    } else if bytes >= 1024 * 1024 {
        alloc::format!("{} MiB", bytes / (1024 * 1024))
    } else if bytes >= 1024 {
        alloc::format!("{} KiB", bytes / 1024)
    } else {
        alloc::format!("{} B", bytes)
    }
}

fn stat(name: &str, value: String, tone: Tone) -> Widget {
    Widget::Status { name: String::from(name), value, tone }
}

/// The heap, as it is right now.
///
/// `mem` printed these numbers into the terminal, which is a fine way to
/// answer a question once and a poor way to watch something. A panel is
/// rebuilt whole every time it is drawn, so this is a reading rather than a
/// cache -- the same rule the settings pages keep.
pub fn memory_panel() -> Panel {
    let (used, total) = crate::mem::heap::HEAP.stats();
    let free = total.saturating_sub(used);
    // A heap this full is not an opinion: the ladder is one physically
    // contiguous allocation and there is no second rung to fall to at runtime.
    let tone = if total == 0 {
        Tone::Bad
    } else if used * 10 > total * 9 {
        Tone::Bad
    } else if used * 4 > total * 3 {
        Tone::Warn
    } else {
        Tone::Ok
    };
    let pct = if total == 0 { 0 } else { used * 100 / total };
    Panel::new(
        "Memory",
        alloc::vec![
            Widget::Heading(String::from("Kernel heap")),
            stat("In use", human(used as u64), tone),
            stat("Free", human(free as u64), Tone::Plain),
            stat("Total", human(total as u64), Tone::Plain),
            stat("Used", alloc::format!("{}%", pct), tone),
            Widget::Sep,
            Widget::Note(alloc::vec![
                String::from("One contiguous allocation,"),
                String::from("chosen at boot from a ladder."),
                String::from("No second rung once running."),
            ]),
            Widget::Button { label: String::from("Close"), action: Action::Close },
        ],
    )
}

/// Every task, what it is, and how often it has been scheduled.
pub fn tasks_panel() -> Panel {
    let here = crate::task::current();
    let mut ws = alloc::vec![Widget::Heading(String::from("Tasks"))];
    for n in 0..crate::task::count() {
        let Some(t) = crate::task::snapshot(n) else { continue };
        let running = n == here;
        // Compact on purpose. A panel is as wide as its widest widget, and a
        // window wider than the gap beside the terminal cannot be placed clear
        // of it -- so verbose values here are paid for by every window that
        // then opens underneath something else. "running" is the tone.
        ws.push(stat(
            t.name,
            alloc::format!("{} sw", t.switches),
            if running { Tone::Ok } else { Tone::Plain },
        ));
    }
    ws.push(Widget::Sep);
    ws.push(Widget::Note(alloc::vec![
        String::from("Round robin at 100 Hz."),
        String::from("No blocked state: every"),
        String::from("task is schedulable."),
    ]));
    ws.push(Widget::Button { label: String::from("Close"), action: Action::Close });
    Panel::new("Tasks", ws)
}

/// The disk, the store region, and whether it may be written to.
pub fn storage_panel() -> Panel {
    let mut ws = alloc::vec![Widget::Heading(String::from("Device"))];
    match crate::dev::nvme::with(|n| (n.block_count, n.block_size, n.max_transfer_blocks)) {
        Some((blocks, bs, mtb)) => {
            ws.push(stat("Capacity", human(blocks * bs as u64), Tone::Plain));
            ws.push(stat("Block size", alloc::format!("{} B", bs), Tone::Plain));
            ws.push(stat("Max transfer", human(mtb as u64 * bs as u64), Tone::Plain));
        }
        None => ws.push(stat("NVMe", String::from("no controller"), Tone::Bad)),
    }

    ws.push(Widget::Sep);
    ws.push(Widget::Heading(String::from("Store")));
    match crate::store::with(|st| {
        (st.sb.region_start, st.sb.region_blocks, st.sb.seq, st.free_blocks())
    }) {
        Some((start, blocks, seq, free)) => {
            ws.push(stat("Region", alloc::format!("{}..{}", start, start + blocks), Tone::Plain));
            ws.push(stat("Snapshots", alloc::format!("{}", seq), Tone::Plain));
            ws.push(stat("Free blocks", alloc::format!("{}", free), if free < 1024 { Tone::Warn } else { Tone::Ok }));
        }
        None => ws.push(stat("Store", String::from("not mounted"), Tone::Warn)),
    }
    // The distinction that decides whether anything survives a reboot, said
    // plainly rather than left to be discovered when a fact goes missing.
    let unlocked = crate::dev::nvme::writes_unlocked();
    ws.push(stat(
        "Writes",
        String::from(if unlocked { "unlocked" } else { "locked" }),
        if unlocked { Tone::Warn } else { Tone::Ok },
    ));
    ws.push(Widget::Note(alloc::vec![
        String::from("Mounting is read-only on"),
        String::from("purpose. A snapshot is what"),
        String::from("outlives the machine."),
    ]));
    ws.push(Widget::Button { label: String::from("Close"), action: Action::Close });
    Panel::new("Storage", ws)
}

/// What the model is attending to, and how far the conversation has run.
pub fn attention_panel() -> Panel {
    let mut ws = alloc::vec![Widget::Heading(String::from("Attention"))];
    match crate::ai::window_facts() {
        None => ws.push(stat("Model", String::from("none loaded"), Tone::Bad)),
        Some((sinks, window, trained, cap, streams, bytes, pos)) => {
            ws.push(stat("Trained", alloc::format!("{} positions", trained), Tone::Plain));
            ws.push(stat("Live cache", alloc::format!("{} positions", cap), Tone::Plain));
            ws.push(stat("Position", alloc::format!("{}", pos), Tone::Plain));
            ws.push(stat("Cache size", human(bytes as u64), Tone::Plain));
            ws.push(Widget::Sep);
            if streams {
                ws.push(stat("Mode", alloc::format!("ring, {} pinned", sinks), Tone::Ok));
                ws.push(stat("Recent", alloc::format!("{} positions", window), Tone::Plain));
                ws.push(Widget::Note(alloc::vec![
                    String::from("Oldest turns scroll away."),
                    String::from("The pinned ones hold the"),
                    String::from("system turn, so the model"),
                    String::from("keeps knowing what it is."),
                ]));
            } else {
                ws.push(stat("Mode", String::from("whole context kept"), Tone::Plain));
                ws.push(Widget::Note(alloc::vec![
                    String::from("Generation stops at the"),
                    String::from("trained length. It becomes"),
                    String::from("a ring on its own first."),
                ]));
            }
        }
    }
    ws.push(Widget::Field {
        name: String::from("sinks recent"),
        text: String::new(),
        cursor: 0,
        submit: Action::Apply(String::from("window")),
    });
    ws.push(Widget::Button { label: String::from("Close"), action: Action::Close });
    Panel::new("Attention", ws)
}

/// Which self-tests exist, which have been run, and which are unhappy.
///
/// The detail stays in the terminal, because the detail is a log. What a log
/// is bad at is exactly what this holds: the *set* of checks, and the state of
/// each one at a glance.
///
/// A suite that has not been run reads "not run", never "pass". A board that
/// is green because nobody looked is worse than no board.
pub fn diagnostics_panel() -> Panel {
    use crate::diag;
    let (pass, fail, unrun) = diag::tally();
    let mut ws = alloc::vec![Widget::Heading(String::from("Self-tests"))];
    for (i, suite) in diag::SUITES.iter().enumerate() {
        let (text, tone) = match diag::verdict(i) {
            diag::Verdict::Pass => ("pass", Tone::Ok),
            diag::Verdict::Fail => ("FAILED", Tone::Bad),
            diag::Verdict::Unknown => ("not run", Tone::Plain),
        };
        ws.push(Widget::Status {
            name: String::from(suite.name),
            value: String::from(text),
            tone,
        });
    }
    ws.push(Widget::Sep);
    ws.push(Widget::Status {
        name: String::from("Summary"),
        value: alloc::format!("{}/{} ok, {} fail", pass, diag::SUITES.len(), fail),
        tone: if fail > 0 {
            Tone::Bad
        } else if unrun > 0 {
            Tone::Plain
        } else {
            Tone::Ok
        },
    });
    // One list rather than a button per suite: the names are the rows above,
    // and a second copy of them as buttons would be the same list twice.
    let items = diag::SUITES
        .iter()
        .map(|su| {
            (
                alloc::format!("Run {}", su.name),
                Action::Run(alloc::format!("diag {}", su.name)),
            )
        })
        .collect();
    ws.push(Widget::List { items, sel: 0 });
    ws.push(Widget::Button {
        label: String::from("Run all"),
        action: Action::Run(String::from("diag all")),
    });
    let _ = unrun;
    ws.push(Widget::Note(alloc::vec![
        String::from("Detail goes to the"),
        String::from("terminal. This is the"),
        String::from("scoreboard."),
    ]));
    ws.push(Widget::Button { label: String::from("Close"), action: Action::Close });
    Panel::new("Diagnostics", ws)
}

pub fn program_manager() -> Panel {
    let run = |label: &str, cmd: &str| {
        (String::from(label), Action::Run(String::from(cmd)))
    };
    // "Model" went to a command called `ai` for two milestones. There is no
    // such command and there never was -- the entry typed it into the
    // terminal, the shell said so, and the menu looked broken. Now it opens
    // the Model settings page, which is what the label promises.
    // **Every entry opens a window.**
    //
    // Most of these used to type a command into the terminal: "Memory" ran
    // `mem` and printed four lines into whatever the shell was doing. That is
    // a fine way to answer a question once and a poor way to run a program --
    // the output scrolled away, it landed in a window the operator was not
    // looking at, and nothing about it could be kept open beside something
    // else. A launcher whose entries are commands is a launcher of one
    // program, the terminal.
    //
    // The panels are readings, rebuilt whole on every draw, so a window left
    // open is a live view rather than a screenshot of when it was opened.
    let items = alloc::vec![
        run("System status", "win open status"),
        run("Memory", "win open memory"),
        run("Tasks", "win open tasks"),
        run("Network", "win open network"),
        run("Storage", "win open storage"),
        run("Files", "win open files"),
        run("Attention", "win open attention"),
        (String::from("Model"), Action::Run(String::from("win open model"))),
        run("ToDo list", "todo"),
        run("Enternet", "enternet"),
        run("Paintbrush", "paint"),
        run("Write", "write"),
        run("Minesweeper", "mines"),
        run("Oracle", "oracle"),
        run("Diagnostics", "win open diag"),
        run("Settings", "win open settings"),
    ];
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


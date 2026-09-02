//! The desktop and its window manager.
//!
//! The terminal is a window on this, not the other way round. That distinction
//! is the whole design: a shell that draws a dialog over itself is a program
//! with a pop-up, while a desktop that hosts a terminal alongside other windows
//! is an environment. The console already knows how to live inside a rectangle
//! (`console::reflow`), so the terminal needs no special case -- it is a window
//! whose content happens to be the character grid.
//!
//! ### Repaint is always total; what reaches the screen is not
//!
//! Windows overlap, and the usual price for that is damage tracking: work out
//! which rectangles a change dirtied and repaint only those. Not here. Every
//! change repaints the wall and then every window back to front -- into the
//! compositor's back buffer, and `compose::present` copies to the screen only
//! the rows that differ from what is already there. Total repaint keeps the
//! window manager obviously correct (the entire class of "stale pixels from a
//! window that used to be there" cannot occur when there is no partial
//! repaint); the diffed present is what stopped every keystroke flashing the
//! wallpaper through the frame while it was rebuilt, and it is what makes
//! repainting on pointer *motion* affordable at all.
//!
//! Back-to-front is also what makes the terminal work with no special case.
//! `console::redraw` paints the whole grid; a window in front of it is simply
//! drawn afterwards.
//!
//! ### The pointer
//!
//! Everything the pointer does, a keystroke can also do -- serial cannot
//! inject PS/2 packets, so an operation that existed only as a gesture would
//! be an operation `drive.py` could never test. The pointer's vocabulary:
//! press to focus and raise, title bar to drag, the corner grip to resize, a
//! double press on a title to maximise, the bar and the Start menu and the
//! wall icons to launch, the wheel to scroll, hover to see where a press
//! would land. The second button opens the menu for whatever is under it.
//!
//! The ancestry is deliberate: 98's furniture (icons, Start, a bar of window
//! buttons, gradient titles), 3.1's construction (bevels, the grey face,
//! dialogs that hug their content), Aperture's colours over both.

use super::browse::Browser;
use super::theme::{self, Rect};
use super::ui::{self, Action, Panel};
use super::{Color, DeskApp, Framebuffer};
use crate::dev::kbd;
use crate::sync::Racy;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// What is inside a window.
pub enum Content {
    /// The character grid. Exactly one window holds it -- there is one console
    /// -- and that is true by construction rather than checked.
    /// A console grid. The payload names which one.
    Terminal(usize),
    Panel(Panel),
    /// Enternet. Its own variant rather than a Panel because a page is not a
    /// stack of widgets: it scrolls, it wraps to the window, and its links are
    /// a selection model the widget enum has no shape for.
    Browser(Browser),
    /// A program that owns its client area: Paintbrush, Write, Minesweeper.
    App(Box<dyn DeskApp>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WinState {
    Normal,
    Maximised,
    Minimised,
}

pub struct MenuItem {
    pub label: String,
    pub action: Action,
}

pub struct Menu {
    pub label: String,
    pub items: Vec<MenuItem>,
}

/// Indices into the pictogram set, named so a window can say what it is
/// without string-matching its own title.
pub const ICO_TERM: usize = 0;
pub const ICO_PROGRAMS: usize = 1;
pub const ICO_FILES: usize = 2;
pub const ICO_TODO: usize = 3;
pub const ICO_NET: usize = 4;
pub const ICO_PAINT: usize = 5;
pub const ICO_WRITE: usize = 6;
pub const ICO_MINES: usize = 7;
pub const ICO_SET: usize = 8;
pub const ICO_ORACLE: usize = 9;

/// The icon for a named panel -- the names `win open` and the Browse routes
/// use. Anything unrecognised gets the mark, because everything here is
/// Aperture something.
fn panel_icon(name: &str) -> usize {
    match name {
        "files" => ICO_FILES,
        "todo" => ICO_TODO,
        "set" | "settings" => ICO_SET,
        _ => ICO_PROGRAMS,
    }
}

pub struct Window {
    pub title: String,
    /// Which pictogram stands for this window on the taskbar.
    pub icon: usize,
    /// Geometry when not maximised. Kept across a maximise so restoring is
    /// exact rather than approximate.
    pub rect: Rect,
    pub state: WinState,
    /// Geometry from before the window was snapped to an edge, if it was.
    ///
    /// A snapped window has lost the size the operator chose for it, and the
    /// only record of that size is here. Dragging back off the edge restores
    /// it, which is the half of snapping that makes the other half safe to
    /// use: an edge that swallows a window's proportions permanently is a
    /// trap, not a shortcut.
    pub snap_back: Option<Rect>,
    /// The route this window's contents came from, when they came from one.
    ///
    /// A panel is a snapshot: it was built from something that has since
    /// changed. Nothing rebuilt it, so pressing a button in an application ran
    /// the command and left the window showing what was true before. Files has
    /// the same defect and always has -- `write x` never updated an open
    /// browser. Keeping the route is what makes rebuilding possible at all.
    pub route: Option<String>,
    /// Corner radius, when this window is not a plain rectangle.
    ///
    /// Stored as the radius rather than as the outline so that a resize does
    /// not have to remember how the outline was asked for. The outline itself
    /// is rebuilt from it during the paint, and cached against the size it was
    /// built for, because the arithmetic is a square root per row of the arc
    /// and doing it sixty times a second for a shape that has not changed is
    /// the sort of waste that turns into "the desktop feels slow".
    pub round: u32,
    pub content: Content,
    pub menus: Vec<Menu>,
    pub closable: bool,
}

impl Window {
    /// Where the window actually is right now.
    fn frame(&self, screen: Rect) -> Rect {
        match self.state {
            WinState::Maximised => screen,
            _ => self.rect,
        }
    }
}

/// What the keyboard is doing.
///
/// A mode rather than a pile of booleans: the states are mutually exclusive by
/// nature -- a window cannot be both being moved and being resized -- and an
/// enum makes that unrepresentable instead of merely unlikely.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Normal,
    /// A menu-bar menu is open.
    Menu { menu: usize, item: usize },
    /// The system menu (Alt-Space) is open.
    Sys { item: usize },
    /// Arrows move the focused window; Enter commits, Esc restores.
    Move { from: Rect },
    /// Arrows resize it.
    Size { from: Rect },
    /// The keyboard is on the taskbar. `item` indexes the app buttons first,
    /// then the task buttons -- one flat list, because Left and Right should
    /// walk the bar the way it looks rather than the way it is built.
    Taskbar { item: usize },
    /// The pointer is moving the focused window by its title bar. `dx`, `dy`
    /// are where inside the bar it was grabbed, so the window does not snap
    /// its corner to the pointer on the first packet.
    Drag { dx: u32, dy: u32, from: Rect },
    /// The pointer is resizing it by the bottom-right corner.
    DragSize { from: Rect, edges: u8 },
    /// The Start menu is open above the taskbar.
    Start { item: usize },
}

/// What the pointer is over, for the paint pass.
///
/// Only things that draw differently when pointed at. Kept on the desktop and
/// compared per packet: a repaint happens when the *target* changes, not when
/// the pointer moves, so sliding across one button costs one redraw.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Hover {
    None,
    /// A taskbar slot, apps and windows in one flat index.
    Task(usize),
    /// A menu-bar label: window `win`, label `menu`.
    MenuLabel { win: usize, menu: usize },
    /// A focusable widget in window `win`'s panel.
    Widget { win: usize, idx: usize },
    /// A desktop icon.
    Icon(usize),
    /// The Start button.
    Start,
    /// A caption button on window `win`: 0 minimise, 1 maximise, 2 close.
    Caption { win: usize, which: usize },
}

pub struct Desktop {
    /// Back to front. The last visible window is the focused one, so z-order
    /// and focus are the same fact stored once -- two fields would be two
    /// things to keep in agreement, and they would drift.
    pub windows: Vec<Window>,
    pub mode: Mode,
    pub hover: Hover,
    /// What has been typed into the Start menu's query row.
    ///
    /// Here rather than inside `Mode::Start`, because `Mode` is `Copy` and is
    /// assigned by value all over this file -- a `String` in it would turn
    /// every one of those into a move and the enum into something that has to
    /// be cloned to be read. It lives as long as the menu is open and is
    /// cleared when it closes, which is the same lifetime either way.
    pub query: String,
}

static DESK: Racy<Option<Desktop>> = Racy::new(None);
static PENDING: Racy<Option<String>> = Racy::new(None);

/// Ask for a command to be run as though it had been typed.
///
/// The one way in from outside this module, so there stays exactly one path
/// from "a command should run" to "a command ran" -- the property the panel
/// dispatch was built around, and one a second setter would quietly end.
pub fn queue_command(cmd: &str) {
    unsafe { *PENDING.get() = Some(String::from(cmd)) };
}

const MARGIN: u32 = 8;
const NUDGE: u32 = 16;
/// A menu bar's height, which is the theme's and no longer the caption's.
///
/// It was `= theme::TITLE_H`, which was a coincidence rather than a fact and
/// became a trap the moment the caption grew: every menu row and every
/// dropdown row would have gone to thirty pixels around sixteen of text,
/// silently, because nothing here says a menu row is as tall as a title.
const MENU_H: u32 = theme::MENU_H;
/// The bar along the bottom.
///
/// A number rather than `TITLE_H + 10` for the same reason. A taskbar is its
/// own surface and XP's is not a caption plus a constant.
const TASK_H: u32 = 38;
/// Gap between task buttons.
const TASK_GAP: u32 = 4;

/// The icons on the wall, top to bottom, and what opening one runs.
///
/// `term` is not a shell command: the terminal is not a panel to open but a
/// window to bring back, and the special case lives in `launch` rather than in
/// the shell so the icon works even while the shell is busy printing.
const ICONS: [(&str, &str); 10] = [
    ("Terminal", "term"),
    ("Programs", "win open programs"),
    ("Files", "win open files"),
    ("ToDo", "todo"),
    ("Enternet", "enternet"),
    ("Paint", "paint"),
    ("Write", "write"),
    ("Mines", "mines"),
    ("Oracle", "oracle"),
    ("Settings", "win open settings"),
];

/// The Start menu, bottom of the bar upward -- the 98 half of the ancestry.
/// Same entries as the icons plus the one thing that belongs behind a second
/// look, exactly where 98 kept it.
const START_ITEMS: [(&str, &str); 12] = [
    // "Search..." used to lead this list, opening a panel with one text field
    // in it. The query row at the foot of this menu does the same job in the
    // place a person already is, and dispatches through the same `open`, so
    // the item became a second door to one room. The panel itself is still
    // there -- `win open search`, and `open` still raises it to offer to write
    // something that does not exist.
    ("Terminal", "term"),
    ("Programs", "win open programs"),
    ("Files", "win open files"),
    ("ToDo", "todo"),
    ("Enternet", "enternet"),
    ("Paint", "paint"),
    ("Write", "write"),
    ("Mines", "mines"),
    ("Oracle", "oracle"),
    ("Settings", "win open settings"),
    ("Reboot", "reboot"),
    ("Shut down", "shutdown"),
];

/// Run what an icon or Start entry names.
fn launch(cmd: &str) {
    if cmd == "term" {
        with(|d| {
            if let Some(i) = d
                .windows
                .iter()
                .position(|w| matches!(w.content, Content::Terminal(c) if c == super::console::USER))
            {
                // The icon is a way back for a minimised terminal besides
                // its task button, so restoring and raising it here is wanted.
                d.windows[i].state = WinState::Normal;
                d.raise(i);
            }
        });
        return;
    }
    // Queue the command and let it run. A command that opens a window leaves
    // that window in front (see `open`/`open_app`); one that only prints does
    // so in the terminal wherever it sits. The shell processes PENDING every
    // loop regardless of focus, so nothing here needs to touch focus at all.
    unsafe { *PENDING.get() = Some(String::from(cmd)) };
}

pub fn ready() -> bool {
    unsafe { (*DESK.get()).is_some() }
}

pub fn with<R>(f: impl FnOnce(&mut Desktop) -> R) -> Option<R> {
    unsafe { (*DESK.get()).as_mut().map(f) }
}

/// The system menu, identical for every window. Items a particular window
/// cannot do are still listed and simply refuse, which is what 3.1 did -- a
/// menu whose contents move around is a menu you cannot learn.
const SYS_ITEMS: [&str; 4] = ["Move", "Size", "Maximise / Restore", "Close"];

/// The area windows may occupy: everything above the taskbar.
///
/// Windows are laid out and maximised against this rather than the panel, so
/// the taskbar is never covered. A bar that a maximised window hides is a bar
/// that is not there when it is most wanted.
fn screen_rect(fb: &Framebuffer) -> Rect {
    Rect::new(
        MARGIN,
        MARGIN,
        fb.width().saturating_sub(MARGIN * 2),
        fb.height().saturating_sub(MARGIN * 2 + TASK_H),
    )
}

fn taskbar_rect(fb: &Framebuffer) -> Rect {
    Rect::new(0, fb.height().saturating_sub(TASK_H), fb.width(), TASK_H)
}

/// The Start button, left end of the bar.
fn start_rect(fb: &Framebuffer) -> Rect {
    let bar = taskbar_rect(fb);
    // Bleeding to the left edge and nearly the bar's full height, as XP's
    // does. A button inset from the corner gives away the pixels Fitts' law
    // makes the most valuable on the screen: the corner is infinitely large to
    // a pointer and a three-pixel margin throws that away.
    Rect::new(bar.x, bar.y + 1, theme::text_w(6) + 34, bar.h.saturating_sub(2))
}

/// Where the Start menu pops, directly above its button. The width formula is
/// `dropdown`'s own, so the paint and the hit-test cannot disagree.
fn start_menu_rect(fb: &Framebuffer) -> theme::Popup {
    let n = start_rows();
    // Wide enough for the longest label, and for a query worth typing. A menu
    // sized only to its labels gives the search row about eleven characters,
    // which is narrower than the thing being searched for.
    let cols = START_ITEMS
        .iter()
        .map(|(l, _)| l.chars().count())
        .max()
        .unwrap_or(4)
        .max(QUERY_COLS);
    let bar = taskbar_rect(fb);
    let probe = theme::Popup::sized(bar.x + 2, 0, cols, n, fb.width());
    theme::Popup::sized(
        bar.x + 2,
        bar.y.saturating_sub(probe.panel.h),
        cols,
        n,
        fb.width(),
    )
}

/// How many characters wide the query row is sized for.
const QUERY_COLS: usize = 22;

/// Rows in the Start menu: every item, then the query.
///
/// The query is **last**, which is to say nearest the Start button, because
/// this menu opens upwards out of the taskbar. Windows 7 put its search box in
/// the same place for the same reason -- it is where the pointer already is
/// after clicking Start, and where the eye goes. It also leaves every item
/// index exactly as it was, so nothing that indexes `START_ITEMS` had to learn
/// about an offset.
fn start_rows() -> usize {
    START_ITEMS.len() + 1
}

/// True when this row is the query rather than an item.
fn is_query_row(item: usize) -> bool {
    item >= START_ITEMS.len()
}

// --- desktop icons ---------------------------------------------------------

const ICON_W: u32 = 132;
const ICON_H: u32 = 84;

/// The icon column, top-left of the wall, downwards -- the 98 half of the
/// ancestry. Stops above the taskbar rather than flowing into it.
fn icon_rects(fb: &Framebuffer) -> Vec<(Rect, usize)> {
    let mut out = Vec::new();
    let mut x = 10u32;
    let mut y = 14u32;
    for (k, _) in ICONS.iter().enumerate() {
        if y + ICON_H > fb.height().saturating_sub(TASK_H) {
            // Wrap to a second column, as the 98 desktop did. The terminal
            // reserves only the first column, so overflow icons sit behind
            // windows -- reachable, and always also in the Start menu.
            x += ICON_W + 8;
            y = 14;
        }
        out.push((Rect::new(x, y, ICON_W, ICON_H), k));
        y += ICON_H + 10;
    }
    out
}

fn icon_at(fb: &Framebuffer, x: i32, y: i32) -> Option<usize> {
    icon_rects(fb)
        .into_iter()
        .find(|(r, _)| contains(*r, x, y))
        .map(|(_, k)| k)
}

/// Paint the icon column. Hovering highlights the label, 98-style; the
/// pictograms are rect art in the theme's own palette, because a 40-pixel
/// bitmap format is more machinery than six drawings justify.
fn draw_icons(fb: &Framebuffer, hover: Hover) {
    for (r, k) in icon_rects(fb) {
        let px = r.x + (r.w - 40) / 2;
        pictogram(fb, k, px, r.y, 40, theme::DESKTOP);
        let (label, _) = ICONS[k];
        let tw = theme::text_w_of(label);
        let tx = r.x + (r.w.saturating_sub(tw)) / 2;
        let ty = r.y + 46;
        if hover == Hover::Icon(k) {
            fb.rect(
                tx.saturating_sub(3),
                ty - 2,
                tw + 6,
                theme::text_h() + 4,
                theme::SELECT,
            );
            theme::text_over(fb, tx, ty, label, theme::SELECT_TEXT);
        } else {
            // A shadow under white text is what keeps a label readable over
            // any wallpaper without boxing it.
            theme::text_over(fb, tx + 1, ty + 1, label, theme::DARKEDGE);
            theme::text_over(fb, tx, ty, label, theme::TITLE_TEXT);
        }
    }
}

/// One pictogram at `(x, y)`, `s` pixels square, by icon index.
///
/// Drawn in a 40-unit design space and scaled, so the wall (40) and the
/// taskbar (20) share one set of drawings instead of two sets that drift.
/// Thicknesses clamp to a pixel -- a hairline that rounds to zero is a
/// detail that vanishes, and at 20 pixels every line is load-bearing.
/// `bg` is what the mark's cut wedges are painted in: the wall behind an
/// icon, the button face on the bar.
fn pictogram(fb: &Framebuffer, k: usize, x: u32, y: u32, s: u32, bg: Color) {
    let face = theme::FACE;
    let dark = theme::DARKEDGE;
    let hi = theme::HILIGHT;
    // Rounded scale from the 40-unit space, and the same for thicknesses but
    // never less than a pixel.
    let c = |v: u32| (v * s + 20) / 40;
    let m = |v: u32| ((v * s + 20) / 40).max(1);
    match k {
        // Terminal: a monitor with a prompt on it.
        ICO_TERM => {
            fb.rect(x + c(2), y + c(4), c(36), c(26), face);
            fb.frame(x + c(2), y + c(4), c(36), c(26), dark);
            fb.rect(x + c(5), y + c(7), c(30), c(20), theme::SCREEN);
            fb.rect(x + c(8), y + c(10), c(8), m(2), theme::APERTURE);
            fb.rect(x + c(8), y + c(15), c(14), m(2), Color::new(0xC8, 0xC8, 0xC8));
            fb.rect(x + c(14), y + c(30), c(12), m(4), face);
            fb.rect(x + c(10), y + c(34), c(20), m(3), face);
            fb.frame(x + c(10), y + c(34), c(20), m(3), dark);
        }
        // Programs: the mark itself. This is the Aperture program manager.
        ICO_PROGRAMS => {
            super::splash::aperture(
                fb,
                (x + c(20)) as i32,
                (y + c(20)) as i32,
                c(18) as i32,
                theme::APERTURE,
                bg,
            );
        }
        // Files: a folder.
        ICO_FILES => {
            fb.rect(x + c(4), y + c(10), c(14), c(6), theme::APERTURE_DEEP);
            fb.rect(x + c(4), y + c(14), c(32), c(20), theme::APERTURE_DEEP);
            fb.frame(x + c(4), y + c(14), c(32), c(20), dark);
            fb.rect(x + c(5), y + c(15), c(30), m(3), theme::APERTURE);
        }
        // ToDo: a card with ticked lines.
        ICO_TODO => {
            fb.rect(x + c(6), y + c(2), c(28), c(36), hi);
            fb.frame(x + c(6), y + c(2), c(28), c(36), dark);
            for (i, done) in [true, true, false].iter().enumerate() {
                let ly = y + c(8 + i as u32 * 10);
                fb.frame(x + c(10), ly, m(6), m(6), dark);
                if *done {
                    fb.rect(x + c(12), ly + m(2), m(3), m(3), theme::APERTURE_DEEP);
                }
                fb.rect(x + c(20), ly + m(2), c(10), m(2), theme::SHADOW);
            }
        }
        // Enternet: a rough globe.
        ICO_NET => {
            for (i, w) in [16u32, 28, 34, 38, 38, 38, 34, 28, 16].iter().enumerate() {
                let ly = y + c(2 + i as u32 * 4);
                fb.rect(x + c(20) - c(*w) / 2, ly, c(*w), m(4), Color::new(0x2A, 0x4A, 0x6E));
            }
            fb.rect(x + c(2), y + c(16), c(36), m(3), hi);
            fb.rect(x + c(18), y + c(2), m(3), c(36), hi);
            fb.frame(x + c(12), y + c(8), c(16), c(24), hi);
        }
        // Paint: a palette board with wells. (This arm and the three after it
        // were numbered one past their icons for a while -- Paint wore the
        // sliders, Settings wore the minefield -- and the labels under the
        // wall icons hid it. The names make that mistake unwriteable.)
        ICO_PAINT => {
            fb.rect(x + c(4), y + c(8), c(32), c(26), Color::new(0xB0, 0x86, 0x50));
            fb.frame(x + c(4), y + c(8), c(32), c(26), dark);
            for (i, col) in [
                theme::APERTURE,
                Color::new(0x30, 0x70, 0xC0),
                Color::new(0x30, 0xA0, 0x40),
                Color::new(0xC0, 0x30, 0x30),
            ]
            .iter()
            .enumerate()
            {
                let (ix, iy) = (
                    x + c(8 + (i as u32 % 2) * 14),
                    y + c(12 + (i as u32 / 2) * 11),
                );
                fb.rect(ix, iy, m(9), m(7), *col);
                fb.frame(ix, iy, m(9), m(7), dark);
            }
            fb.rect(x + c(26), y + c(2), m(3), c(14), dark);
            fb.rect(x + c(25), y + c(1), m(5), m(4), theme::APERTURE_DEEP);
        }
        // Write: a page with lines of text.
        ICO_WRITE => {
            fb.rect(x + c(8), y + c(2), c(24), c(36), hi);
            fb.frame(x + c(8), y + c(2), c(24), c(36), dark);
            for i in 0..5u32 {
                let w = if i == 4 { 10 } else { 16 };
                fb.rect(x + c(12), y + c(7 + i * 6), c(w), m(2), theme::SHADOW);
            }
            fb.rect(x + c(12), y + c(31), c(8), m(2), theme::APERTURE_DEEP);
        }
        // Mines: a grid with one uncovered mine.
        ICO_MINES => {
            fb.rect(x + c(2), y + c(2), c(36), c(36), face);
            for i in 0..4u32 {
                fb.rect(x + c(2 + i * 12), y + c(2), 1, c(36), theme::SHADOW);
                fb.rect(x + c(2), y + c(2 + i * 12), c(36), 1, theme::SHADOW);
            }
            fb.rect(x + c(15), y + c(15), c(10), c(10), dark);
            fb.rect(x + c(19), y + c(11), m(2), c(18), dark);
            fb.rect(x + c(11), y + c(19), c(18), m(2), dark);
            fb.rect(x + c(17), y + c(17), m(3), m(3), hi);
        }
        // Oracle: an eye. The lids are stacked rows, the iris the mark's
        // colour, and it reads at both sizes because an eye is mostly its
        // contrast.
        ICO_ORACLE => {
            for (i, w) in [8u32, 20, 30, 36, 30, 20, 8].iter().enumerate() {
                let ly = y + c(8 + i as u32 * 4);
                fb.rect(x + c(20) - c(*w) / 2, ly, c(*w), m(4), hi);
            }
            fb.rect(x + c(14), y + c(14), c(12), c(12), theme::APERTURE);
            fb.frame(x + c(14), y + c(14), c(12), c(12), dark);
            fb.rect(x + c(18), y + c(18), m(4), m(4), dark);
        }
        // Settings: three sliders.
        _ => {
            for i in 0..3u32 {
                let ly = y + c(8 + i * 11);
                fb.rect(x + c(4), ly + m(2), c(32), m(2), theme::SHADOW);
                fb.rect(x + c(4), ly + m(4), c(32), 1, hi);
                let kx = x + c(6 + (i * 11) % 24);
                fb.rect(kx, ly.saturating_sub(m(2)), m(6), m(10), face);
                fb.frame(kx, ly.saturating_sub(m(2)), m(6), m(10), dark);
            }
        }
    }
}

impl Desktop {
    /// Index of the focused window: the topmost that is not minimised.
    pub fn focus(&self) -> Option<usize> {
        self.windows
            .iter()
            .rposition(|w| w.state != WinState::Minimised)
    }

    /// Bring a window to the front. Focus follows, because they are the same.
    fn raise(&mut self, i: usize) {
        if i >= self.windows.len() {
            return;
        }
        let w = self.windows.remove(i);
        self.windows.push(w);
    }
}

pub fn init() {
    let Some(fb) = super::primary() else {
        return;
    };
    let screen = screen_rect(&fb);

    let mut pm = ui::program_manager();
    let (pm_w, pm_h) = pm.preferred();
    pm.set_title("Program Manager");
    let pm_w = pm_w.min(screen.w / 2);
    // The icon column stays visible: the wall is part of the interface now,
    // and a terminal that covers it leaves the icons reachable only by
    // closing the terminal, which is backwards.
    let icons_w = 10 + ICON_W + 10;
    let term_x = screen.x.max(icons_w);
    let term_w = (screen.x + screen.w)
        .saturating_sub(term_x + pm_w + MARGIN);

    let terminal = Window {
        title: String::from("GLaDOS Terminal"),
        icon: ICO_TERM,
        rect: Rect::new(term_x, screen.y, term_w, screen.h),
        state: WinState::Normal,
        snap_back: None,
        route: None,
        round: theme::WIN_ROUND,
        content: Content::Terminal(super::console::USER),
        menus: alloc::vec![
            Menu {
                label: String::from("File"),
                items: alloc::vec![
                    item("New window", "win open status"),
                    item("Snapshot now", "snap"),
                    item("Reboot", "reboot"),
                ],
            },
            Menu {
                label: String::from("View"),
                items: alloc::vec![
                    item("Clear", "clear"),
                    item("Redraw", "refresh"),
                    item("Help", "help"),
                ],
            },
            Menu {
                label: String::from("System"),
                items: alloc::vec![
                    item("Status", "status"),
                    item("Memory", "mem"),
                    item("Windows", "win"),
                ],
            },
        ],
        closable: false,
    };

    let pm_x = (term_x + term_w + MARGIN).min(screen.x + screen.w.saturating_sub(pm_w));
    let pmw = Window {
        title: String::from("Program Manager"),
        icon: ICO_PROGRAMS,
        rect: Rect::new(pm_x, screen.y, pm_w, pm_h.min(screen.h)),
        state: WinState::Normal,
        snap_back: None,
        route: None,
        round: theme::WIN_ROUND,
        content: Content::Panel(pm),
        menus: Vec::new(),
        closable: false,
    };

    // The executive console: the boot log, background tasks, and the episodes
    // the machine decides to run on its own.
    //
    // Starts minimised. It carries everything printed before the shell
    // existed, so it is worth having and worth being able to reach, and it is
    // not what an operator wants filling half the screen the moment they sit
    // down. The taskbar button is the affordance; the machine raises nothing
    // by itself, because a window that appears on its own while somebody is
    // typing is the problem this split was made to solve, arriving by a
    // different route.
    let executive = Window {
        title: String::from("Executive"),
        icon: ICO_TERM,
        rect: Rect::new(
            term_x + MARGIN,
            screen.y + MARGIN,
            term_w.saturating_sub(MARGIN * 2).max(320),
            screen.h.saturating_sub(MARGIN * 4).max(200),
        ),
        state: WinState::Minimised,
        snap_back: None,
        route: None,
        round: theme::WIN_ROUND,
        content: Content::Terminal(super::console::EXEC),
        menus: alloc::vec![Menu {
            label: String::from("View"),
            items: alloc::vec![
                item("Clear", "exec clear"),
                item("Save log", "log save /tmp/boot.txt"),
                item("Redraw", "refresh"),
            ],
        }],
        closable: true,
    };

    unsafe {
        *DESK.get() = Some(Desktop {
            windows: alloc::vec![pmw, executive, terminal],
            mode: Mode::Normal,
            hover: Hover::None,
            query: String::new(),
        })
    };
    // The compositor needs the heap, which exists by now; the console then
    // paints into the back buffer and pushes its own cells through, so shell
    // output stays immediate between desktop draws.
    super::compose::init();
    if let Some(back) = super::compose::target() {
        for ch in [super::console::USER, super::console::EXEC] {
            super::console::with_ch(ch, |c| c.retarget(back.clone(), true));
        }
    }
    draw();
}

fn item(label: &str, cmd: &str) -> MenuItem {
    MenuItem {
        label: String::from(label),
        action: Action::Run(String::from(cmd)),
    }
}

/// Open a new window holding a panel.
/// Open a panel, remembering the route it came from so it can be rebuilt.
pub fn open_routed(title: &str, panel: Panel, route: &str) {
    open(title, panel);
    let route = String::from(route);
    with(|d| {
        if let Some(w) = d.windows.last_mut() {
            w.route = Some(route);
        }
    });
}

/// Rebuild every window whose contents came from a route.
///
/// Called after a command runs, because a command is the only thing that
/// changes what a route would produce. Cheap when nothing is open: a window
/// without a route is skipped, and there is no route to resolve.
pub fn refresh_routed() {
    let routes: Vec<(usize, String)> = with(|d| {
        d.windows
            .iter()
            .enumerate()
            .filter_map(|(i, w)| w.route.clone().map(|r| (i, r)))
            .collect()
    })
    .unwrap_or_default();
    if routes.is_empty() {
        return;
    }
    let mut any = false;
    for (i, route) in routes {
        // Built outside `with`, because building an application's panel runs
        // its program, which reads the namespace -- and the desktop borrow is
        // held for the whole closure.
        let Some((_, panel)) = ui::panel_for_route(&route) else {
            continue;
        };
        with(|d| {
            if let Some(w) = d.windows.get_mut(i) {
                w.content = Content::Panel(panel);
                any = true;
            }
        });
    }
    if any {
        draw();
    }
}

/// Show a route in the window already showing that route's family, if there
/// is one.
///
/// A search box is one surface whose contents change, not a pile of windows.
/// Without this, asking twice left two windows with the same title and the
/// operator no way to tell which one was live -- and the third, the offer, made
/// three. Matching is on the route's verb, so `search` and `search:calc` are
/// the same surface while `app:todo` is not.
///
/// Returns whether a window was reused, because the caller's focus decision
/// differs: a reused window is already where the eye is.
pub fn show_routed(title: &str, panel: Panel, route: &str) -> bool {
    let verb = route.split(':').next().unwrap_or(route);
    let found = with(|d| {
        d.windows.iter().position(|w| {
            w.route.as_deref().map(|r| r.split(':').next().unwrap_or(r)) == Some(verb)
        })
    })
    .flatten();
    let Some(i) = found else {
        open_routed(title, panel, route);
        return false;
    };
    // The two panels are different heights, so keeping the old rectangle would
    // clip whichever is taller. Position stays: the window has not moved, only
    // what is in it.
    let Some(fb) = super::primary() else {
        return false;
    };
    let screen = screen_rect(&fb);
    let (pw, ph) = panel.preferred();
    // Clamped and pulled back on screen. `open` does this for a new window and
    // growing one in place has to do it too: the offer is wider than the query
    // box, and at the cascade position the extra width went off the right edge
    // -- taking the caption buttons and the right half of every line with it.
    let (pw, ph) = (pw.min(screen.w), ph.min(screen.h));
    let title = String::from(title);
    let route = String::from(route);
    with(|d| {
        let Some(mut w) = (i < d.windows.len()).then(|| d.windows.remove(i)) else {
            return;
        };
        w.title = title;
        w.route = Some(route);
        w.content = Content::Panel(panel);
        if w.state == WinState::Normal {
            w.rect.w = pw;
            w.rect.h = ph;
            w.rect.x = w.rect.x.min(screen.x + screen.w.saturating_sub(pw)).max(screen.x);
            w.rect.y = w.rect.y.min(screen.y + screen.h.saturating_sub(ph)).max(screen.y);
        }
        // Last is front, and front is focused.
        d.windows.push(w);
    });
    draw();
    true
}

pub fn open(title: &str, panel: Panel) {
    let Some(fb) = super::primary() else {
        return;
    };
    let screen = screen_rect(&fb);
    let (w, h) = panel.preferred();
    let (w, h) = (w.min(screen.w), h.min(screen.h));
    with(|d| {
        // Cascade down the right-hand side rather than from the top-left
        // corner. The terminal is the widest window and starts at the left, so
        // a window cascading from the origin lands entirely behind it: opening
        // one looked like nothing had happened.
        let n = d.windows.len() as u32;
        let off = (n % 6) * 28;
        // Clear of the terminal's right edge where possible.
        //
        // Not tidiness. The console draws straight to the framebuffer on its
        // own schedule, with no idea which windows are above it, so a window
        // overlapping the terminal gets its overlapping strip repainted with
        // console output the moment anything is printed. Windows may still be
        // moved over it -- `redraw_over_terminal` repairs that between
        // commands -- but nothing should start out overlapping.
        let clear_of_terminal = d
            .windows
            .iter()
            .find(|win| matches!(win.content, Content::Terminal(c) if c == super::console::USER))
            .map(|win| win.rect.x + win.rect.w + MARGIN)
            .unwrap_or(screen.x);
        // Narrow it rather than slide it under the terminal.
        //
        // A panel is as wide as its widest widget, so a page with one long
        // line in it becomes a window wider than the gap beside the terminal
        // -- and the placement below then puts it as far right as it fits,
        // which is *underneath*, with its left half hidden behind console
        // text that repaints on its own schedule. Three separate panels hit
        // that while being written, each time looking like the window had
        // failed to draw rather than like it had been placed badly.
        //
        // Clamping the width keeps the whole window reachable: the content
        // clips at the frame, which is visible and fixable by dragging the
        // edge, where a window two thirds behind another is neither.
        // `MIN_CLEAR` is a floor, because a terminal wide enough to leave no
        // usable gap should get an overlapping window rather than a sliver.
        const MIN_CLEAR: u32 = 320;
        let gap = (screen.x + screen.w).saturating_sub(clear_of_terminal);
        let w = if gap >= MIN_CLEAR { w.min(gap) } else { w };
        let x = screen
            .x
            .max(screen.x + screen.w.saturating_sub(w))
            .max(clear_of_terminal.min(screen.x + screen.w.saturating_sub(w)));
        let y = (screen.y + screen.h / 3 + off)
            .min(screen.y + screen.h.saturating_sub(h));
        d.windows.push(Window {
            title: String::from(title),
            // `open` is reached through `win open <name>` with the panel's
            // name as the title, so the name is what there is to go by.
            icon: panel_icon(title),
            rect: Rect::new(x, y, w, h),
            state: WinState::Normal,
        snap_back: None,
        route: None,
        round: theme::WIN_ROUND,
            content: Content::Panel(panel),
            menus: Vec::new(),
            closable: true,
        });
    });
    // The new window is what was opened, so it stays in front and focused --
    // window priority a person expects, and what every launch path relies on.
    // The keystroke that launched it (a shell Enter, a menu Enter, a click) is
    // consumed before the window exists, so there is nothing left to leak into
    // it; the earlier worry conflated "raise the window" with "steal the next
    // command", which is only a problem for a headless driver typing blind.
    draw();
}

/// Open a window around a program.
///
/// The keyboard goes back to the terminal, exactly as `open` and
/// `open_browser` do -- and the first version of this function is why that
/// rule exists in triplicate. It kept focus on the new window, on the
/// reasoning that a game is opened to be played; but the desktop takes
/// *every* key while a non-terminal window has focus, so the next command
/// line -- typed or serial -- was fed to the program a byte at a time.
/// Minesweeper ate "echo after-mines" as e,c,h,o... and flagged a cell on
/// the f. The shell froze from the outside, which is exactly how the
/// browser presented when it made the same mistake. Click the window or
/// Alt-Tab to play; with a pointer that is one gesture.
pub fn open_app(title: &str, icon: usize, app: Box<dyn DeskApp>, w: u32, h: u32) {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);

    // Open no smaller than the program says it needs.
    //
    // The caller passes a preferred size and those were being clamped to the
    // screen and nothing else, so a program whose preferred size was stale or
    // whose chrome had grown since opened cropped, with the missing part
    // unreachable: the only resize was a 14-pixel corner grip nobody found.
    // Asking the program for its floor and honouring it here means a window
    // is born usable, and the edge dragging added alongside means it stays
    // that way.
    let inner = app.min_size();
    let need_w = inner.0 + theme::FRAME * 2;
    let need_h = inner.1 + theme::TITLE_H + theme::FRAME * 2;
    let (w, h) = (
        w.max(need_w).min(screen.w),
        h.max(need_h).min(screen.h),
    );
    with(|d| {
        let n = d.windows.len() as u32;
        let off = (n % 5) * 32;
        let x = (screen.x + (screen.w.saturating_sub(w)) / 2 + off)
            .min(screen.x + screen.w.saturating_sub(w));
        let y = (screen.y + (screen.h.saturating_sub(h)) / 3 + off)
            .min(screen.y + screen.h.saturating_sub(h));
        d.windows.push(Window {
            title: String::from(title),
            icon,
            rect: Rect::new(x, y, w, h),
            state: WinState::Normal,
        snap_back: None,
        route: None,
        round: theme::WIN_ROUND,
            content: Content::App(app),
            menus: Vec::new(),
            closable: true,
        });
    });
    // Opened to be used -> in front and focused. To type at the shell again,
    // click the terminal, Alt-Tab, or its taskbar button, as with any window.
    draw();
}

pub fn open_paint() {
    let (w, h) = super::paint::Paint::preferred();
    open_app("Paintbrush", ICO_PAINT, Box::new(super::paint::Paint::new()), w, h);
}

pub fn open_mines() {
    let (w, h) = super::mines::Mines::preferred();
    open_app("Minesweeper", ICO_MINES, Box::new(super::mines::Mines::new()), w, h);
}

pub fn open_write(path: &str) {
    let (w, h) = super::write::Writer::preferred();
    open_app("Write", ICO_WRITE, Box::new(super::write::Writer::new(path)), w, h);
}

pub fn open_todo() {
    let (w, h) = super::todo::Todo::preferred();
    open_app("ToDo", ICO_TODO, Box::new(super::todo::Todo::new()), w, h);
}

pub fn open_oracle(premise: &str) {
    let (w, h) = super::oracle::Oracle::preferred();
    open_app("Oracle", ICO_ORACLE, Box::new(super::oracle::Oracle::new(premise)), w, h);
}

/// The agent transcript, live. Shares the Oracle's icon because both are
/// instruments pointed at what the machine is thinking.
///
/// Focus returns to the terminal immediately: this window is a view, it
/// consumes no keys, and a viewer that held the keyboard would eat every
/// serial command typed after opening it -- the desktop answers declined
/// keys with "handled" all the same, which is exactly how Minesweeper once
/// swallowed a whole test script.
/// Move the most recently opened window clear of the terminal.
///
/// For windows that hand focus back. `open_app` centres, which is right for a
/// program opened by a click -- it keeps focus and sits on top -- and wrong for
/// one opened on the machine's own initiative, because handing the keyboard
/// back also raises the terminal *over* it. The progress window opened
/// underneath and showed a 225-pixel strip of its right edge: no title, no
/// fields, and nothing to say it was working.
///
/// `desk::open` has carried this rule for panels since the same thing happened
/// to them. There is a second reason beyond tidiness, given there: the console
/// draws straight to the framebuffer on its own schedule with no idea which
/// windows are above it, so an overlapping window gets its overlapping strip
/// repainted with console output the moment anything is printed.
///
/// Best effort, and "best" means as far right as the screen allows rather than
/// giving up. A window wider than the gap cannot clear the terminal entirely,
/// and the first version of this declined to move one at all -- which left the
/// progress window exactly where the bug had put it, hidden, because 520 did
/// not fit in a 504-pixel gap. Sixteen pixels overlapped beats three hundred.
fn clear_of_terminal() {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);
    with(|d| {
        let Some(edge) = d
            .windows
            .iter()
            .find(|w| matches!(w.content, Content::Terminal(c) if c == super::console::USER))
            .map(|w| w.rect.x + w.rect.w + MARGIN)
        else {
            return;
        };
        if let Some(w) = d.windows.last_mut() {
            let room = screen.x + screen.w.saturating_sub(w.rect.w);
            w.rect.x = edge.min(room).max(screen.x);
        }
    });
}

/// Show the authoring loop, without taking the keyboard.
///
/// Opened by the run rather than by somebody asking, which is exactly the case
/// where focus must not move: the desktop takes every key while a non-terminal
/// window has focus, so a window appearing mid-command eats the rest of the
/// line. Minesweeper consumed `echo after-mines` a byte at a time for this
/// reason and flagged a cell on the `f`.
///
/// Idempotent. A second run must not stack a second window on the first.
pub fn open_authoring() {
    if has_window("Writing") {
        draw();
        return;
    }
    let (w, h) = super::agentwin::AuthorWin::preferred();
    open_app("Writing", ICO_ORACLE, Box::new(super::agentwin::AuthorWin::new()), w, h);
    clear_of_terminal();
    focus_terminal();
}

/// Is a window with this title already open?
pub fn has_window(title: &str) -> bool {
    with(|d| d.windows.iter().any(|w| w.title == title)).unwrap_or(false)
}

pub fn open_agentlog() {
    let (w, h) = super::agentwin::AgentLog::preferred();
    open_app("Agent", ICO_ORACLE, Box::new(super::agentwin::AgentLog::new()), w, h);
    // The same latent fault: it hands focus back, so it opened under the
    // terminal too. Nobody noticed because a transcript is read after the
    // fact, by which point the window has usually been moved.
    clear_of_terminal();
    focus_terminal();
}

/// Open Enternet, optionally at a URL.
pub fn open_browser(url: &str) {
    let mut b = Browser::new();
    if !url.is_empty() {
        b.load(url);
    }
    let title = String::from("Enternet");
    with(|d| {
        let Some(fb) = super::primary() else { return };
        let screen = screen_rect(&fb);
        let w = (screen.w * 3 / 4).min(screen.w);
        let h = (screen.h * 3 / 4).min(screen.h);
        let x = screen.x + (screen.w.saturating_sub(w)) / 2;
        let y = screen.y + (screen.h.saturating_sub(h)) / 3;
        d.windows.push(Window {
            title,
            icon: ICO_NET,
            rect: Rect::new(x, y, w, h),
            state: WinState::Normal,
        snap_back: None,
        route: None,
        round: theme::WIN_ROUND,
            content: Content::Browser(b),
            menus: Vec::new(),
            closable: true,
        });
    });
    // The keyboard goes back to the shell, exactly as `open` does.
    //
    // In front and focused, like any opened window. A browser is opened to be
    // driven, so it should have the keyboard; to return to the shell, focus
    // the terminal. (The headless driver must Alt-Tab back before typing the
    // next command -- it types blind over serial and cannot see what has the
    // keyboard.)
    draw();
}

// --- the pointer ----------------------------------------------------------

/// The arrow, as a bitmap. `X` is outline, `.` is fill, space is transparent.
const CURSOR: [&str; 17] = [
    "X          ",
    "XX         ",
    "X.X        ",
    "X..X       ",
    "X...X      ",
    "X....X     ",
    "X.....X    ",
    "X......X   ",
    "X.......X  ",
    "X........X ",
    "X.....XXXXX",
    "X..X..X    ",
    "X.X X..X   ",
    "XX  X..X   ",
    "X    X..X  ",
    "     X..X  ",
    "      XX   ",
];
const CUR_W: u32 = 11;
const CUR_H: u32 = 17;

/// What the pointer is currently saying it will do.
///
/// A frame edge is six pixels of nothing until the pointer changes shape over
/// it. That change is the entire discoverability of resizing: there is no
/// visible grip to aim at, and an operator who has only ever used Windows
/// looks for the double arrow and nothing else. Held in a static because the
/// cursor is repainted on every mouse packet anyway, so switching bitmaps
/// costs a branch and no redraw.
pub const SHAPE_ARROW: u8 = 0;
const SHAPE_H: u8 = 1;
const SHAPE_V: u8 = 2;
const SHAPE_NWSE: u8 = 3;
const SHAPE_NESW: u8 = 4;
static SHAPE: Racy<u8> = Racy::new(SHAPE_ARROW);

/// The resize glyphs, on the same eleven by seventeen canvas as the arrow so
/// that saving and restoring the pixels underneath does not change with the
/// shape. Drawn as solid figures and outlined at paint time, because a thin
/// dark glyph vanishes on a dark title bar and a thin light one vanishes on
/// the wall; the halo makes each one legible on both.
const CUR_H_BITS: [&str; 17] = [
    "           ",
    "           ",
    "           ",
    "           ",
    "           ",
    "           ",
    "   X   X   ",
    "  XX   XX  ",
    " XXXXXXXXX ",
    "  XX   XX  ",
    "   X   X   ",
    "           ",
    "           ",
    "           ",
    "           ",
    "           ",
    "           ",
];
const CUR_V_BITS: [&str; 17] = [
    "           ",
    "           ",
    "           ",
    "     X     ",
    "    XXX    ",
    "   XXXXX   ",
    "     X     ",
    "     X     ",
    "     X     ",
    "     X     ",
    "     X     ",
    "   XXXXX   ",
    "    XXX    ",
    "     X     ",
    "           ",
    "           ",
    "           ",
];
const CUR_NWSE_BITS: [&str; 17] = [
    "           ",
    "           ",
    "           ",
    " XXXXXX    ",
    " XXXXX     ",
    " XXXX      ",
    " XXX       ",
    " XX        ",
    " X         ",
    "         X ",
    "        XX ",
    "       XXX ",
    "      XXXX ",
    "     XXXXX ",
    "    XXXXXX ",
    "           ",
    "           ",
];
const CUR_NESW_BITS: [&str; 17] = [
    "           ",
    "           ",
    "           ",
    "    XXXXXX ",
    "     XXXXX ",
    "      XXXX ",
    "       XXX ",
    "        XX ",
    "         X ",
    " X         ",
    " XX        ",
    " XXX       ",
    " XXXX      ",
    " XXXXX     ",
    " XXXXXX    ",
    "           ",
    "           ",
];

fn bitmap(shape: u8) -> &'static [&'static str; 17] {
    match shape {
        SHAPE_H => &CUR_H_BITS,
        SHAPE_V => &CUR_V_BITS,
        SHAPE_NWSE => &CUR_NWSE_BITS,
        SHAPE_NESW => &CUR_NESW_BITS,
        _ => &CURSOR,
    }
}

/// Which glyph a set of held edges asks for. Two adjacent edges is a corner,
/// and a corner drags along one of the two diagonals.
fn shape_for(e: u8) -> u8 {
    let l = e & edge::LEFT != 0;
    let r = e & edge::RIGHT != 0;
    let t = e & edge::TOP != 0;
    let b = e & edge::BOTTOM != 0;
    match (l, r, t, b) {
        (true, _, true, _) | (_, true, _, true) => SHAPE_NWSE,
        (_, true, true, _) | (true, _, _, true) => SHAPE_NESW,
        (true, _, _, _) | (_, true, _, _) => SHAPE_H,
        (_, _, true, _) | (_, _, _, true) => SHAPE_V,
        _ => SHAPE_ARROW,
    }
}

/// The glyph the pointer should wear at a point, given no drag is running.
///
/// Deliberately asks the same two questions the press path asks, in the same
/// order, so that the shape shown and the action taken cannot disagree: a
/// pointer that promises a resize where a click would move the window is
/// worse than no pointer feedback at all.
fn resize_shape_at(x: i32, y: i32) -> u8 {
    let Some(i) = window_at(x, y) else {
        return SHAPE_ARROW;
    };
    with(|d| {
        let w = &d.windows[i];
        if w.state == WinState::Maximised {
            return SHAPE_ARROW;
        }
        shape_for(edges_at(w.rect, x, y))
    })
    .unwrap_or(SHAPE_ARROW)
}

/// Pixels the cursor is currently covering, and where.
///
/// Saved and restored rather than repainted. The desktop redraws everything
/// back to front, which is a few million stores; doing that per mouse packet
/// at a hundred reports a second is a system that cannot be used. Only the
/// eleven by seventeen rectangle under the arrow is touched.
static SAVED: Racy<[u32; (CUR_W * CUR_H) as usize]> = Racy::new([0; (CUR_W * CUR_H) as usize]);
static SHOWN: Racy<Option<(u32, u32)>> = Racy::new(None);
/// Where the pointer is, whether or not it is currently painted. `draw` uses
/// this to put the arrow back after a repaint.
static POS: Racy<Option<(u32, u32)>> = Racy::new(None);

/// Held for as long as anything is touching `SAVED`, `SHOWN`, or the screen.
///
/// The cursor used to be safe by accident: `poll_mouse` was the only caller
/// and it ran on the shell task, so nothing could interleave. `pump_cursor`
/// broke that -- it runs from inside `generate`, which the agent task also
/// runs, so a cursor move could land in the middle of the shell's `draw`.
/// The interleaving is not a flicker: the arrow is painted after `present`,
/// the pixels under it are saved from the *pre*-present frame, and the next
/// hide stamps that stale block onto the screen and then saves it again as
/// the new background -- so the damage is copied forward every frame, and in
/// the worst ordering what gets saved is the arrow itself, leaving one
/// permanent ghost per occurrence.
static CUR_BUSY: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Releases the claim however the holder leaves.
struct Claim;

impl Claim {
    /// `None` when somebody else holds it. Never waits: every caller here has
    /// something sensible to do with a refusal, and a spin would be spinning
    /// against a task that only runs when this one yields.
    fn take() -> Option<Claim> {
        use core::sync::atomic::Ordering::{Acquire, Relaxed};
        CUR_BUSY.compare_exchange(false, true, Acquire, Relaxed).ok().map(|_| Claim)
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        CUR_BUSY.store(false, core::sync::atomic::Ordering::Release);
    }
}

/// The one thing about the desktop that can be checked without a screen.
///
/// Everything else here is pixels, and a claim about pixels needs eyes. This
/// is arithmetic: the cursor statics are shared between the shell task and
/// whichever task is generating, and the whole defence is that two holders
/// cannot exist at once. That property was argued for and never checked --
/// the race it prevents needs a concurrent generation and a moving mouse to
/// reproduce, which no boot test can arrange -- so at least the exclusion
/// itself is asserted rather than assumed.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };

    let first = Claim::take();
    claim("a free claim can be taken", first.is_some());
    claim("and a second holder is refused", Claim::take().is_none());
    drop(first);
    let again = Claim::take();
    claim("dropping it lets the next one in", again.is_some());
    drop(again);
    // Leaving it held would deadlock every later repaint, so the release path
    // is worth one more claim of its own.
    claim("and it is free afterwards", !CUR_BUSY.load(core::sync::atomic::Ordering::Relaxed));

    // --- tiling -----------------------------------------------------------
    //
    // Pure geometry, checked here rather than by dragging windows into corners
    // and looking. The odd width is the case that matters: two halves computed
    // as a width applied twice leave a column of wallpaper showing between
    // windows that are supposed to meet.
    {
        let screen = Rect::new(10, 20, 101, 51);
        let l = tile_rect(Tile::Left, screen);
        let r = tile_rect(Tile::Right, screen);
        claim("halves start at the screen and end at it", l.x == 10 && r.x + r.w == 111);
        claim("and meet exactly, on an odd width", l.x + l.w == r.x);
        let tl = tile_rect(Tile::TopLeft, screen);
        let bl = tile_rect(Tile::BottomLeft, screen);
        claim("quadrants meet vertically too", tl.y + tl.h == bl.y);
        claim("and the bottom row reaches the foot", bl.y + bl.h == 71);
        claim("full is the screen", tile_rect(Tile::Full, screen) == screen);
        // The four quadrants must cover the screen exactly: no overlap and no
        // gap, which is the property a grid of windows is judged on.
        let area: u32 = [Tile::TopLeft, Tile::TopRight, Tile::BottomLeft, Tile::BottomRight]
            .iter()
            .map(|t| {
                let q = tile_rect(*t, screen);
                q.w * q.h
            })
            .sum();
        claim("and they tile it with no gap or overlap", area == screen.w * screen.h);
    }
    ok
}

pub fn cursor_hide(fb: &Framebuffer) {
    let Some((x, y)) = (unsafe { *SHOWN.get() }) else { return };
    // Ask the compositor for those pixels back, rather than trusting a copy
    // taken earlier.
    //
    // The back buffer is by definition what the screen is supposed to show, so
    // restoring from it cannot be stale -- and staleness was the whole bug
    // class here. `pump_cursor` runs between tokens while the console is
    // streaming into the same region; every flushed line made `SAVED` describe
    // a frame that no longer existed, and putting it back punched a 17x11
    // block of old pixels into the answer text that nothing ever repaired,
    // because the compositor believed those pixels were already correct.
    if !super::compose::repaint_rect(x, y, CUR_W, CUR_H) {
        let saved = unsafe { &*SAVED.get() };
        for row in 0..CUR_H {
            for col in 0..CUR_W {
                if x + col < fb.width() && y + row < fb.height() {
                    fb.put(x + col, y + row, saved[(row * CUR_W + col) as usize]);
                }
            }
        }
    }
    unsafe { *SHOWN.get() = None };
}

pub fn cursor_show(fb: &Framebuffer, x: u32, y: u32) {
    cursor_hide(fb);
    let shape = unsafe { *SHAPE.get() };
    let bits = bitmap(shape);
    // Only needed when there is no compositor to ask. With one, this loop is
    // 187 uncached reads off the aperture per mouse move -- on the path that
    // runs between every generated token -- to fill a buffer nothing reads.
    if !super::compose::active() {
        let saved = unsafe { &mut *SAVED.get() };
        // Save the whole box before painting anything. The halo pass writes
        // pixels the figure never covers, and restoring is only correct if
        // every pixel that might be written was read first.
        for row in 0..CUR_H {
            for col in 0..CUR_W {
                let (px, py) = (x + col, y + row);
                if px < fb.width() && py < fb.height() {
                    saved[(row * CUR_W + col) as usize] = fb.get(px, py);
                }
            }
        }
    }
    let set = |r: i32, c: i32| -> bool {
        if r < 0 || c < 0 || r >= CUR_H as i32 || c >= CUR_W as i32 {
            return false;
        }
        matches!(bits[r as usize].as_bytes().get(c as usize), Some(b'X') | Some(b'.'))
    };
    for row in 0..CUR_H {
        let line = bits[row as usize].as_bytes();
        for col in 0..CUR_W {
            let (px, py) = (x + col, y + row);
            if px >= fb.width() || py >= fb.height() {
                continue;
            }
            match line.get(col as usize) {
                // The arrow carries its own outline in the bitmap.
                Some(b'X') if shape == SHAPE_ARROW => fb.put(px, py, fb.raw(theme::TEXT)),
                Some(b'.') => fb.put(px, py, fb.raw(theme::HILIGHT)),
                Some(b'X') => fb.put(px, py, fb.raw(theme::HILIGHT)),
                _ if shape != SHAPE_ARROW => {
                    // Blank, but touching the figure: this is the halo.
                    let (r, c) = (row as i32, col as i32);
                    let near = set(r - 1, c)
                        || set(r + 1, c)
                        || set(r, c - 1)
                        || set(r, c + 1)
                        || set(r - 1, c - 1)
                        || set(r - 1, c + 1)
                        || set(r + 1, c - 1)
                        || set(r + 1, c + 1);
                    if near {
                        fb.put(px, py, fb.raw(theme::TEXT));
                    }
                }
                _ => {}
            }
        }
    }
    unsafe { *SHOWN.get() = Some((x, y)) };
}

/// Read the mouse and act on it. Called from the idle loop.
/// Keep the pointer alive during work that owns the shell task.
///
/// The freeze this fixes is not a slow renderer. A foreground `ask` runs
/// `generate` on the shell task with `yielding` clear, so for the whole of a
/// generation -- seconds per token, minutes for an answer -- the shell is
/// inside the command and `poll_mouse` is never reached. The clock task keeps
/// its own quantum and goes on painting the uptime directly to the aperture,
/// which is exactly the reported symptom: a moving clock above frozen windows.
///
/// **Motion only, deliberately.** This does not dispatch presses, drags,
/// hover or the wheel. A click handled here would run `press_at`, which can
/// open a window, close one, or start an app -- re-entering the desktop, and
/// potentially the engine, from inside a generation that already holds it.
/// Buttons stay latched in the mouse state and are acted on by `poll_mouse`
/// when the command finishes, so nothing is lost but the ordering is safe.
///
/// Cheap enough to call between tokens: the cursor is 17x11, so a move is a
/// few hundred pixels on the aperture and nothing on the compositor.
pub fn pump_cursor() {
    use crate::dev::mouse;
    if !mouse::present() || !ready() {
        return;
    }
    // `peek` rather than `take`: the button edges belong to `poll_mouse`, and
    // consuming the packet here would swallow a click that arrived mid-answer.
    let Some((x, y)) = mouse::position() else { return };
    if unsafe { *POS.get() } == Some((x, y)) {
        return;
    }
    if move_cursor(x, y) {
        unsafe { *POS.get() = Some((x, y)) };
    }
}

/// Paint the uptime into the taskbar's clock well, through the compositor.
///
/// **The last thing in the system that drew straight to the aperture.** It
/// wrote the digits over the firmware's memory while the compositor's shadow
/// went on describing whatever had been there before, so the compositor could
/// not erase the clock: a later `present` comparing `back` against `shadow`
/// found them equal over that rectangle and wrote nothing, leaving digits on
/// screen that the desktop had already decided to paint over. The clock kept
/// moving above frozen windows for a different reason -- that was scheduling,
/// and `pump_cursor` answered it -- but this is why the two never agreed about
/// what was on the taskbar.
///
/// Under the same claim the cursor takes, and for the same reason: this runs
/// on the clock task and `draw` runs on the shell's, and they would otherwise
/// both be writing the back buffer through a `&mut` neither knows the other
/// holds. Interrupts off around the paint so the claim cannot be held across a
/// task switch -- a few thousand pixels once a tenth of a second.
///
/// Losing the claim costs one tick of clock. The shell is mid-frame and about
/// to paint the taskbar itself.
pub fn paint_clock(real: &Framebuffer, x: u32, y: u32, text: &str, scale: u32) {
    crate::cpu::without_interrupts(|| {
        let Some(_claim) = Claim::take() else { return };
        let target = super::compose::target().unwrap_or(*real);
        // Light ink on the tray's own ground, not dark ink on `FACE`. The
        // recess is a dark colour now, and `draw_text` fills the cell behind
        // every glyph -- so the old pair stamped a cream block into the middle
        // of it, which reads as a label stuck on the bar rather than as a
        // readout set into it. The background has to stay opaque here: it is
        // what erases the digit that was there a tenth of a second ago, and
        // nothing else presents this rectangle on the clock's behalf.
        target.draw_text(x, y, text, theme::TITLE_TEXT, theme::TRAY, scale);
        // Straight through, because the clock is not on anybody's draw path:
        // nothing else is going to present this rectangle on its behalf.
        super::compose::flush_rect(
            x,
            y,
            text.len() as u32 * super::font::GLYPH_W * scale,
            super::font::GLYPH_H * scale,
        );
    });
}

/// Put the arrow somewhere, without letting anything else paint while it does.
///
/// Interrupts off for the whole of it, so the timer cannot switch tasks
/// between the save and the paint -- the body is a few hundred pixels, and the
/// alternative is a repaint landing inside it and saving the arrow's own
/// pixels as the background it will later restore.
///
/// `false` when a repaint already owns the screen. Losing costs one frame of
/// pointer lag and nothing else, so the caller leaves `POS` alone and tries
/// again rather than recording a move it never drew.
fn move_cursor(x: u32, y: u32) -> bool {
    crate::cpu::without_interrupts(|| {
        let Some(_claim) = Claim::take() else { return false };
        let Some(fb) = super::primary() else { return false };
        cursor_show(&fb, x, y);
        true
    })
}

pub fn poll_mouse() {
    use crate::dev::mouse;
    if !mouse::present() || !ready() {
        return;
    }
    let s = mouse::take();
    if !s.moved {
        return;
    }
    // Still a precondition even though the framebuffer is no longer touched
    // from here: `move_cursor` fetches its own, under the claim.
    if super::primary().is_none() {
        return;
    }
    let (x, y) = (s.x.max(0), s.y.max(0));
    unsafe { *POS.get() = Some((x as u32, y as u32)) };

    // A press is an edge, not a level: the mouse reports the button held down
    // in every packet while it is down, and acting on the level would fire a
    // click for every packet of a drag.
    let was = unsafe { *BUTTONS.get() };
    unsafe { *BUTTONS.get() = (s.left, s.right) };
    let pressed_left = s.left && !was.0;
    let released_left = !s.left && was.0;
    let pressed_right = s.right && !was.1;

    if s.wheel != 0 {
        wheel_at(x, y, s.wheel);
    }

    let dragging = matches!(
        with(|d| d.mode),
        Some(Mode::Drag { .. }) | Some(Mode::DragSize { .. })
    );
    let in_app = unsafe { *APP_PRESS.get() };
    if pressed_left {
        press_at(x, y);
    } else if s.left && dragging {
        drag_to(x, y);
    } else if s.left && in_app {
        // The stroke: motion with the button held, inside the program that
        // took the press.
        app_drag_to(x, y);
    }
    if released_left && dragging {
        let was_move = matches!(with(|d| d.mode), Some(Mode::Drag { .. }));
        with(|d| d.mode = Mode::Normal);
        if was_move {
            snap_release(x, y);
        }
        // The resulting rectangle, not just the fact that a drag ended. A
        // resize that silently does nothing and a resize that ran and was
        // clamped back look identical from outside; the numbers separate them.
        if let Some(Some(r)) = with(|d| d.focus().map(|f| d.windows[f].rect)) {
            crate::serial_println!("[desk] drag end {}x{}+{}+{}", r.w, r.h, r.x, r.y);
        }
        trace("drag end");
        draw();
    }
    if released_left && in_app {
        unsafe { *APP_PRESS.get() = false };
        app_release();
    }
    if pressed_right {
        right_click_at(x, y);
    }
    // Hover feedback only while no button is down: mid-drag the pointer
    // crosses half the screen, and highlighting everything on the way would
    // be noise.
    if !s.left {
        update_hover(x, y);
    }
    // Pick the glyph after the event has been acted on, so a press that
    // started a resize is already reflected in the mode.
    let want = match with(|d| d.mode) {
        Some(Mode::DragSize { edges, .. }) => shape_for(edges),
        Some(Mode::Drag { .. }) => SHAPE_ARROW,
        _ => resize_shape_at(x, y),
    };
    unsafe { *SHAPE.get() = want };
    // Same claim the pump takes. This runs on the shell task and the pump runs
    // on whichever task is generating, so without it the two interleave in
    // exactly the way the cursor statics cannot survive.
    move_cursor(x as u32, y as u32);
}

static BUTTONS: Racy<(bool, bool)> = Racy::new((false, false));
/// The previous press, for double-click detection: milliseconds and place.
static LAST_CLICK: Racy<(u64, i32, i32)> = Racy::new((0, -100, -100));

fn now_ms() -> u64 {
    let mhz = crate::time::tsc_mhz();
    if mhz == 0 {
        return 0;
    }
    crate::time::rdtsc() / (mhz * 1000)
}

/// True when this press pairs with the previous one as a double-click.
fn is_double(x: i32, y: i32) -> bool {
    let now = now_ms();
    let (t, lx, ly) = unsafe { *LAST_CLICK.get() };
    unsafe { *LAST_CLICK.get() = (now, x, y) };
    now.saturating_sub(t) < 400 && (x - lx).abs() < 5 && (y - ly).abs() < 5
}

/// Where a window's title bar, menu bar and client area are.
///
/// `theme::chrome` owns the formula now, because the painter and the
/// hit-tester both need it and they used to hold two copies in two files. This
/// is the shim that keeps the four callers reading as they did.
fn chrome(frame: Rect, has_menus: bool) -> (Rect, Option<Rect>, Rect) {
    let c = theme::chrome(frame, has_menus);
    (c.title, c.menubar, c.client)
}

/// Which menu-bar label sits under a point, from the same iterator the paint
/// loop walks.
fn menu_label_at(menus: &[Menu], bar: Rect, x: i32, y: i32) -> Option<usize> {
    if y < bar.y as i32 || y >= (bar.y + bar.h) as i32 {
        return None;
    }
    theme::menu_labels(bar, menus.iter().map(|m| m.label.as_str()))
        .position(|r| x >= r.x as i32 && x < (r.x + r.w) as i32)
}

/// Which sides of a frame a resize drag has hold of.
///
/// A bitmask rather than an enum of nine cases, because a corner is genuinely
/// two edges at once and the arithmetic that moves them is the same code
/// twice.
pub mod edge {
    pub const NONE: u8 = 0;
    pub const LEFT: u8 = 1 << 0;
    pub const RIGHT: u8 = 1 << 1;
    pub const TOP: u8 = 1 << 2;
    pub const BOTTOM: u8 = 1 << 3;
}

/// How close to a border counts as grabbing it.
///
/// Six pixels is what the desktops this imitates used, and it is the number a
/// hand trained on them expects to find. The corner zones are wider because a
/// corner is the only way to change both dimensions at once, and hunting for
/// a 6x6 square is the difference between a window manager that resizes and
/// one that appears not to.
const BORDER: i32 = 6;
const CORNER: i32 = 16;

/// Raise the executive console, restoring it if it is minimised.
///
/// Only ever called because somebody asked. Nothing in the system raises this
/// window on its own: a window that appears while an operator is typing is
/// the problem the split was made to solve, arriving by another route.
pub fn show_executive() {
    let ok = with(|d| {
        let Some(i) = d
            .windows
            .iter()
            .position(|w| matches!(w.content, Content::Terminal(c) if c == super::console::EXEC))
        else {
            return false;
        };
        d.windows[i].state = WinState::Normal;
        let last = d.windows.len() - 1;
        d.windows.swap(i, last);
        true
    })
    .unwrap_or(false);
    if ok {
        draw();
    }
}

/// Round the focused window's corners, or square them again with zero.
pub fn set_round(r: u32) -> bool {
    let ok = with(|d| {
        let Some(f) = d.focus() else { return false };
        d.windows[f].round = r;
        true
    })
    .unwrap_or(false);
    if ok {
        draw();
    }
    ok
}

/// The outline for the window being painted, kept between frames.
///
/// One entry, not one per window: `draw` paints windows one at a time and
/// asks for each outline in turn, so a single slot serves as long as it
/// records what it was built for. Two windows with different radii alternate
/// and rebuild each time, which is two square roots per row of an arc and
/// still nothing next to the paint it is wrapping.
static SHAPE_CACHE: Racy<Option<(u32, u32, u32)>> = Racy::new(None);
static OUTLINE: Racy<Option<super::Shape>> = Racy::new(None);

/// Paint `f` clipped to this window's outline, or straight through when the
/// window is an ordinary rectangle.
fn with_window_shape<R>(w: &Window, frame: Rect, f: impl FnOnce() -> R) -> R {
    // A maximised window has square corners and nothing behind it to show
    // through them. XP squares them too.
    if w.round == 0 || w.state == WinState::Maximised {
        return f();
    }
    let key = (frame.w, frame.h, w.round);
    let stale = unsafe { *SHAPE_CACHE.get() } != Some(key);
    if stale {
        unsafe {
            // Only ever `round_top` through this slot, which is why the key
            // does not name the kind. Somebody adding a second shape here has
            // to add a fourth field, and the failure if they forget is a
            // window with the wrong corners -- visible, and not looking like a
            // cache bug.
            *OUTLINE.get() = Some(super::Shape::round_top(frame.w, frame.h, w.round));
            *SHAPE_CACHE.get() = Some(key);
        }
    }
    let Some(shape) = (unsafe { (*OUTLINE.get()).as_ref() }) else {
        return f();
    };
    super::with_shape(shape, frame.x as i32, frame.y as i32, f)
}

/// Which borders the pointer is over, if any.
///
/// The whole frame edge, not one corner grip. A 14x14 square in the
/// bottom-right was the only way to resize anything, which is discoverable if
/// you already know it is there and invisible otherwise -- the operator's
/// report was that windows could not be resized at all, only maximised.
fn edges_at(frame: Rect, x: i32, y: i32) -> u8 {
    let (l, t) = (frame.x as i32, frame.y as i32);
    let (r, b) = (l + frame.w as i32, t + frame.h as i32);
    // Outside the frame, or far enough inside that this is content.
    if x < l - 1 || x > r + 1 || y < t - 1 || y > b + 1 {
        return edge::NONE;
    }

    let mut e = edge::NONE;
    // Corners first: within CORNER of two borders claims both, so the
    // diagonal drag is available along a usable stretch of each side.
    let near_l = (x - l).abs() <= CORNER;
    let near_r = (r - x).abs() <= CORNER;
    let near_t = (y - t).abs() <= CORNER;
    let near_b = (b - y).abs() <= CORNER;
    let corner_h = (x - l).abs() <= BORDER || (r - x).abs() <= BORDER;
    let corner_v = (y - t).abs() <= BORDER || (b - y).abs() <= BORDER;

    if corner_h && near_t && near_b {
        // A window shorter than two corner zones: prefer the nearer edge.
        if y - t < b - y {
            e |= edge::TOP;
        } else {
            e |= edge::BOTTOM;
        }
    } else if corner_h && near_t {
        e |= edge::TOP;
    } else if corner_h && near_b {
        e |= edge::BOTTOM;
    }
    if corner_v && near_l && near_r {
        if x - l < r - x {
            e |= edge::LEFT;
        } else {
            e |= edge::RIGHT;
        }
    } else if corner_v && near_l {
        e |= edge::LEFT;
    } else if corner_v && near_r {
        e |= edge::RIGHT;
    }

    // Plain edges, for the long stretch between the corners.
    if (x - l).abs() <= BORDER {
        e |= edge::LEFT;
    }
    if (r - x).abs() <= BORDER {
        e |= edge::RIGHT;
    }
    if (y - t).abs() <= BORDER {
        e |= edge::TOP;
    }
    if (b - y).abs() <= BORDER {
        e |= edge::BOTTOM;
    }
    e
}

/// The smallest a window may be dragged to.
///
/// Asked of the program inside it, because the program is the only thing that
/// knows. Minesweeper's board does not reflow and Paint's canvas does not
/// shrink; a window manager that let either be dragged to 160 pixels wide
/// would be hiding their content behind a frame the operator cannot undo
/// without maximising.
fn min_size_of(w: &Window) -> (u32, u32) {
    let inner = match &w.content {
        Content::App(a) => a.min_size(),
        _ => (240, 120),
    };
    (
        inner.0 + theme::FRAME * 2,
        inner.1 + theme::TITLE_H + theme::FRAME * 2,
    )
}

/// The bottom-right corner that resizes, generous enough to hit.
fn size_grip(frame: Rect) -> Rect {
    let g = 14u32;
    Rect::new(
        frame.x + frame.w.saturating_sub(g),
        frame.y + frame.h.saturating_sub(g),
        g,
        g,
    )
}

fn contains(r: Rect, x: i32, y: i32) -> bool {
    x >= r.x as i32 && y >= r.y as i32 && x < (r.x + r.w) as i32 && y < (r.y + r.h) as i32
}

/// Topmost visible window containing a point.
fn window_at(x: i32, y: i32) -> Option<usize> {
    with(|d| {
        d.windows.iter().rposition(|w| {
            w.state != WinState::Minimised
                && x >= w.rect.x as i32
                && y >= w.rect.y as i32
                && x < (w.rect.x + w.rect.w) as i32
                && y < (w.rect.y + w.rect.h) as i32
        })
    })
    .flatten()
}

/// Route one left press: dropdowns, the Start menu, the taskbar, window
/// chrome, and finally the content under the point. This is the pointer's
/// whole vocabulary.
fn press_at(x: i32, y: i32) {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);
    let double = is_double(x, y);
    // A new press starts a new gesture; whatever the last one was is over.
    unsafe { *APP_PRESS.get() = false };
    // The one trace on this path: presses are the pointer's whole vocabulary,
    // and a click that silently does nothing is undebuggable from a
    // screenshot. Serial only, like every other trace here.
    crate::serial_println!("[desk] press {},{}{}", x, y, if double { " double" } else { "" });

    // An open menu eats the press: on an item it acts, anywhere else it
    // closes. Both must come before window routing or the press falls through
    // the menu onto whatever is behind it.
    if menu_press(&fb, x, y, screen) {
        draw();
        return;
    }

    if contains(taskbar_rect(&fb), x, y) {
        task_press(&fb, x, y);
        draw();
        return;
    }

    let Some(i) = window_at(x, y) else {
        // The wall, or an icon on it. Icons open on a double press, as they
        // have since 95 -- a single press is selection, which here is the
        // hover highlight already showing.
        with(|d| d.mode = Mode::Normal);
        if double {
            if let Some(k) = icon_at(&fb, x, y) {
                launch(ICONS[k].1);
            }
        }
        draw();
        return;
    };

    // Raising is focusing here, which is the one fact the window manager
    // keeps. Everything below acts on the window at its new index.
    with(|d| {
        d.raise(i);
        d.mode = Mode::Normal;
    });
    let f = with(|d| d.windows.len() - 1).unwrap_or(0);

    let got = with(|d| {
        let w = &d.windows[f];
        (w.frame(screen), !w.menus.is_empty(), w.state)
    });
    let Some((frame, has_menus, state)) = got else { return };
    let (title, menubar, client) = chrome(frame, has_menus);

    if contains(title, x, y) {
        // The caption buttons eat their presses before the bar means "drag".
        let hit = theme::caption_buttons(title)
            .iter()
            .position(|b| contains(*b, x, y));
        if let Some(which) = hit {
            caption_action(f, which);
            draw();
            return;
        }
        if double {
            with(|d| {
                d.windows[f].state = match d.windows[f].state {
                    WinState::Maximised => WinState::Normal,
                    _ => WinState::Maximised,
                };
            });
        } else if state != WinState::Maximised {
            // Remember where inside the frame the pointer took hold, so the
            // window follows the grab instead of snapping a corner to it.
            with(|d| {
                d.mode = Mode::Drag {
                    dx: (x - frame.x as i32).max(0) as u32,
                    dy: (y - frame.y as i32).max(0) as u32,
                    from: frame,
                }
            });
        }
        draw();
        return;
    }

    if state != WinState::Maximised {
        let e = edges_at(frame, x, y);
        if e != edge::NONE {
            with(|d| d.mode = Mode::DragSize { from: frame, edges: e });
            draw();
            return;
        }
    }

    if let Some(bar) = menubar {
        let hit = with(|d| menu_label_at(&d.windows[f].menus, bar, x, y)).flatten();
        if let Some(mi) = hit {
            with(|d| d.mode = Mode::Menu { menu: mi, item: 0 });
            draw();
            return;
        }
    }

    // The content. Each kind answers the press in its own terms.
    let step = with(|d| match &mut d.windows[f].content {
        Content::Panel(p) => p.mouse(client, x, y, double),
        Content::Browser(b) => {
            if b.click(client, x, y) {
                ui::Step::Redraw
            } else {
                ui::Step::Idle
            }
        }
        Content::App(a) => {
            if a.press(client, x, y) {
                // The press begins a gesture: until the button lifts, motion
                // belongs to this program. That is what a brush stroke is.
                unsafe { *APP_PRESS.get() = true };
                ui::Step::Redraw
            } else {
                ui::Step::Idle
            }
        }
        // A press in the terminal is focus, which raising already did.
        Content::Terminal(_) => ui::Step::Idle,
    })
    .unwrap_or(ui::Step::Idle);
    act_on(step);
    draw();
}

/// Whether the held left button is a gesture inside an app's client area,
/// as opposed to a window drag or nothing.
static APP_PRESS: Racy<bool> = Racy::new(false);

/// Forward held-button motion to the focused program.
fn app_drag_to(x: i32, y: i32) {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);
    let changed = with(|d| {
        let Some(f) = d.focus() else { return false };
        let frame = d.windows[f].frame(screen);
        let has_menus = !d.windows[f].menus.is_empty();
        let (_, _, client) = chrome(frame, has_menus);
        match &mut d.windows[f].content {
            Content::App(a) => a.drag(client, x, y),
            _ => false,
        }
    })
    .unwrap_or(false);
    if changed {
        draw();
    }
}

fn app_release() {
    let changed = with(|d| {
        let Some(f) = d.focus() else { return false };
        match &mut d.windows[f].content {
            Content::App(a) => a.release(),
            _ => false,
        }
    })
    .unwrap_or(false);
    if changed {
        draw();
    }
}

/// One caption button, pressed. The same three verbs the system menu
/// offers, bound to the three squares 98 put them on.
fn caption_action(f: usize, which: usize) {
    crate::serial_println!("[desk] caption {} on window {}", which, f);
    with(|d| {
        if f >= d.windows.len() {
            return;
        }
        match which {
            0 => d.windows[f].state = WinState::Minimised,
            1 => {
                d.windows[f].state = match d.windows[f].state {
                    WinState::Maximised => WinState::Normal,
                    _ => WinState::Maximised,
                };
            }
            _ => {
                if d.windows[f].closable {
                    d.windows.remove(f);
                } else {
                    // The terminal is the shell: it keeps running with no
                    // window, so its X means "put it away", which is the
                    // honest version of close for a program that cannot die.
                    d.windows[f].state = WinState::Minimised;
                }
            }
        }
    });
}

/// Carry out what a panel handed back. The same arms the keyboard path runs,
/// in one place, so a pointer activation and an Enter cannot diverge.
fn act_on(step: ui::Step) {
    match step {
        ui::Step::Do(Action::Run(cmd)) => {
            focus_terminal();
            unsafe { *PENDING.get() = Some(cmd) };
        }
        ui::Step::Do(Action::Browse(route)) => {
            if let Some((title, panel)) = ui::panel_for_route(&route) {
                let kind = route.split(':').next().unwrap_or("");
                with(|d| {
                    if let Some(f) = d.focus() {
                        d.windows[f].title = title;
                        // Navigation changes what the window *is*, and the
                        // taskbar shows windows by what they are.
                        d.windows[f].icon = panel_icon(kind);
                        d.windows[f].content = Content::Panel(panel);
                    }
                });
            }
        }
        ui::Step::Close => close_focused(),
        _ => {}
    }
}

/// Close the focused window, or minimise it when it cannot be closed.
///
/// One function because there are two ways to press a Close button and they
/// used to do different things: clicking it removed the window, while Enter on
/// it (and Esc) fell to `cycle`, which alt-tabs -- so Cancel on a dialog left
/// the dialog open, one place further back. The same control has to mean the
/// same thing however it is pressed; that rule is why the pointer and the paint
/// pass share their layout functions, and it applies to the keyboard too.
pub fn close_focused() {
    with(|d| {
        if let Some(f) = d.focus() {
            if d.windows[f].closable {
                d.windows.remove(f);
            } else {
                d.windows[f].state = WinState::Minimised;
            }
        }
    });
    draw();
}

/// A press on the taskbar: the Start button, an app, or a window button.
/// The focused window's own button minimises it, which is the bar's one
/// toggle and the reason it can replace stowed icons.
fn task_press(fb: &Framebuffer, x: i32, y: i32) {
    if contains(start_rect(fb), x, y) {
        with(|d| {
            d.mode = match d.mode {
                Mode::Start { .. } => Mode::Normal,
                // Opens on the query row, because pressing Start and typing is
                // the common case and arrowing up into the items is one key
                // either way.
                _ => Mode::Start { item: START_ITEMS.len() },
            }
        });
        return;
    }
    let hit = with(|d| {
        task_layout(fb, d)
            .into_iter()
            .enumerate()
            .find(|(_, (r, ..))| contains(*r, x, y))
            .map(|(i, _)| i)
    })
    .flatten();
    with(|d| d.mode = Mode::Normal);
    let Some(w) = hit else { return };
    with(|d| {
        if w >= d.windows.len() {
            return;
        }
        if d.windows[w].state == WinState::Minimised {
            d.windows[w].state = WinState::Normal;
            d.raise(w);
        } else if d.focus() == Some(w) {
            d.windows[w].state = WinState::Minimised;
        } else {
            d.raise(w);
        }
    });
}

/// Where the open dropdown is -- window menu, system menu, or Start -- and how
/// many rows it holds. Mirrors the paint pass; they must not disagree.
fn dropdown_rows(fb: &Framebuffer, d: &Desktop, screen: Rect) -> Option<theme::Popup> {
    if let Mode::Start { .. } = d.mode {
        return Some(start_menu_rect(fb));
    }
    let f = d.focus()?;
    let frame = d.windows[f].frame(screen);
    let inner = frame.shrink(theme::FRAME);
    match d.mode {
        Mode::Menu { menu, .. } => {
            let m = d.windows[f].menus.get(menu)?;
            let bar = Rect::new(inner.x, inner.y + theme::TITLE_H + 2, inner.w, MENU_H);
            let x = theme::menu_labels(bar, d.windows[f].menus.iter().map(|p| p.label.as_str()))
                .nth(menu)?
                .x;
            let cols = m.items.iter().map(|i| i.label.chars().count()).max().unwrap_or(4);
            let y = inner.y + theme::TITLE_H + 2 + MENU_H;
            Some(theme::Popup::sized(x, y, cols, m.items.len(), fb.width().saturating_sub(x)))
        }
        Mode::Sys { .. } => {
            let cols = SYS_ITEMS.iter().map(|s| s.chars().count()).max().unwrap_or(4);
            let y = inner.y + theme::TITLE_H;
            Some(theme::Popup::sized(
                inner.x,
                y,
                cols,
                SYS_ITEMS.len(),
                fb.width().saturating_sub(inner.x),
            ))
        }
        _ => None,
    }
}

/// Which dropdown row a point is in, mirroring `dropdown`'s row layout.


/// A press while any menu is open. Returns true when consumed.
fn menu_press(fb: &Framebuffer, x: i32, y: i32, screen: Rect) -> bool {
    let open = with(|d| {
        matches!(d.mode, Mode::Menu { .. } | Mode::Sys { .. } | Mode::Start { .. })
    })
    .unwrap_or(false);
    if !open {
        return false;
    }
    let mut run: Option<String> = None;
    with(|d| {
        let Some(p) = dropdown_rows(fb, d, screen) else {
            d.mode = Mode::Normal;
            return;
        };
        let Some(item) = p.item_at(x, y) else {
            d.mode = Mode::Normal;
            return;
        };
        match d.mode {
            Mode::Start { .. } => {
                if is_query_row(item) {
                    // Clicking the box puts the keyboard in it and leaves the
                    // menu open. Closing on a click into a text field would be
                    // the one interaction nobody expects.
                    d.mode = Mode::Start { item };
                } else {
                    d.mode = Mode::Normal;
                    d.query.clear();
                    run = Some(String::from(START_ITEMS[item].1));
                }
            }
            Mode::Menu { menu, .. } => {
                d.mode = Mode::Normal;
                if let Some(f) = d.focus() {
                    if let Some(mi) = d.windows[f].menus[menu].items.get(item) {
                        if let Action::Run(cmd) = &mi.action {
                            unsafe { *PENDING.get() = Some(cmd.clone()) };
                        }
                    }
                }
            }
            Mode::Sys { .. } => {
                d.mode = Mode::Normal;
                if let Some(f) = d.focus() {
                    sys_action(d, f, item, screen);
                }
            }
            _ => {}
        }
    });
    if let Some(cmd) = run {
        launch(&cmd);
    }
    true
}

/// Where a tile goes.
///
/// A window manager that can only maximise makes an operator measure two
/// windows by hand to put them side by side. `snap_release` has answered half
/// of that since it was written -- drag to an edge, take half the screen --
/// and this is the rest: the corners, and a way to ask without a pointer.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tile {
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Full,
}

/// The rectangle a tile occupies on `screen`.
///
/// Pure, so the geometry is checkable at boot with no framebuffer and no
/// desktop -- the alternative being to check it by dragging windows into
/// corners and looking at them.
///
/// Halves are computed as *positions* rather than as a width applied twice:
/// `screen.w / 2` used twice leaves a column unpainted on an odd width, and
/// the gap shows as a stripe of wallpaper between two windows that are
/// supposed to meet.
fn tile_rect(t: Tile, screen: Rect) -> Rect {
    let midx = screen.x + screen.w / 2;
    let midy = screen.y + screen.h / 2;
    let right_w = (screen.x + screen.w).saturating_sub(midx);
    let bottom_h = (screen.y + screen.h).saturating_sub(midy);
    match t {
        Tile::Full => screen,
        Tile::Left => Rect::new(screen.x, screen.y, screen.w / 2, screen.h),
        Tile::Right => Rect::new(midx, screen.y, right_w, screen.h),
        Tile::TopLeft => Rect::new(screen.x, screen.y, screen.w / 2, screen.h / 2),
        Tile::TopRight => Rect::new(midx, screen.y, right_w, screen.h / 2),
        Tile::BottomLeft => Rect::new(screen.x, midy, screen.w / 2, bottom_h),
        Tile::BottomRight => Rect::new(midx, midy, right_w, bottom_h),
    }
}

/// Put the focused window in a tile.
pub fn tile_focused(t: Tile) -> bool {
    let Some(fb) = super::primary() else { return false };
    let screen = screen_rect(&fb);
    let ok = with(|d| {
        let Some(f) = d.focus() else { return false };
        let w = &mut d.windows[f];
        // Remembered the way a drag-snap remembers, so a tiled window dragged
        // off an edge afterwards comes back to the size it chose for itself
        // rather than to a half screen.
        w.snap_back = w.snap_back.or(Some(w.rect));
        w.state = WinState::Normal;
        w.rect = tile_rect(t, screen);
        true
    })
    .unwrap_or(false);
    if ok {
        draw();
    }
    ok
}

/// The four quadrants, filled from the front of the stack back.
///
/// Answers how many it placed. Capped at four deliberately: a fifth window has
/// no quadrant, and shrinking the grid to fit an arbitrary count produces
/// panes too small to read any of these windows in.
pub fn tile_all() -> usize {
    let Some(fb) = super::primary() else { return 0 };
    let screen = screen_rect(&fb);
    const ORDER: [Tile; 4] = [Tile::TopLeft, Tile::TopRight, Tile::BottomLeft, Tile::BottomRight];
    let n = with(|d| {
        // Front to back, so the window last used takes the first quadrant
        // rather than whichever happens to be deepest in the stack.
        let mut idx: Vec<usize> = (0..d.windows.len())
            .filter(|i| d.windows[*i].state != WinState::Minimised)
            .collect();
        idx.reverse();
        idx.truncate(ORDER.len());
        for (slot, i) in idx.iter().enumerate() {
            let w = &mut d.windows[*i];
            w.snap_back = w.snap_back.or(Some(w.rect));
            w.state = WinState::Normal;
            w.rect = tile_rect(ORDER[slot], screen);
        }
        idx.len()
    })
    .unwrap_or(0);
    if n > 0 {
        draw();
    }
    n
}

/// How close to a screen edge a move has to end to count as a snap.
const SNAP: i32 = 8;

/// Finish a window move: snap to an edge, or come back off one.
///
/// The gesture an operator arriving from Windows tries first, and the reason
/// half-screen tiling is worth having at all -- two windows side by side
/// without measuring either. Left and right edges take half the screen, the
/// top maximises.
///
/// Deliberately decided at release rather than while the pointer is moving.
/// A window that resizes itself under a drag has to re-anchor the grab
/// mid-gesture, and getting that wrong makes the window jump away from the
/// pointer. Releasing is the moment the operator has committed.
fn snap_release(x: i32, y: i32) {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);
    let (l, t) = (screen.x as i32, screen.y as i32);
    let r = l + screen.w as i32;
    let half_w = screen.w / 2;

    let changed = with(|d| {
        let Some(f) = d.focus() else { return false };
        let w = &mut d.windows[f];
        let here = w.rect;

        let b = screen.y as i32 + screen.h as i32;
        let near_l = x <= l + SNAP;
        let near_r = x >= r - SNAP;
        let near_t = y <= t + SNAP;
        let near_b = y >= b - SNAP;

        // A corner beats an edge, and is tested first for that reason. A drag
        // ending within the snap distance of two edges is at a corner and
        // meant one; checking the top or the side first would give it half the
        // screen and swallow the gesture.
        let corner = match (near_l, near_r, near_t, near_b) {
            (true, _, true, _) => Some(Tile::TopLeft),
            (_, true, true, _) => Some(Tile::TopRight),
            (true, _, _, true) => Some(Tile::BottomLeft),
            (_, true, _, true) => Some(Tile::BottomRight),
            _ => None,
        };
        if let Some(tl) = corner {
            w.snap_back = w.snap_back.or(Some(here));
            w.state = WinState::Normal;
            w.rect = tile_rect(tl, screen);
            return true;
        }

        if near_t {
            // The top edge maximises, using the state the caption button and
            // the system menu already use, so there is one maximised window
            // and not two ideas of one.
            if w.state != WinState::Maximised {
                w.snap_back = w.snap_back.or(Some(here));
                w.state = WinState::Maximised;
                return true;
            }
            return false;
        }

        let side = if near_l {
            Some(screen.x)
        } else if near_r {
            Some(screen.x + screen.w - half_w)
        } else {
            None
        };

        if let Some(nx) = side {
            w.snap_back = w.snap_back.or(Some(here));
            w.state = WinState::Normal;
            w.rect = Rect::new(nx, screen.y, half_w, screen.h);
            return true;
        }

        // Off the edge, and the window is carrying a size it did not choose:
        // give it back, around the place it was dropped rather than the place
        // it came from.
        if let Some(back) = w.snap_back.take() {
            if w.state != WinState::Maximised {
                let cx = here.x + here.w / 2;
                let nx = cx
                    .saturating_sub(back.w / 2)
                    .min(screen.x + screen.w.saturating_sub(back.w))
                    .max(screen.x);
                w.rect = Rect::new(nx, here.y, back.w, back.h);
                return true;
            }
        }
        false
    })
    .unwrap_or(false);
    if changed {
        draw();
    }
}

/// Pointer motion while the title bar or the size grip is held.
fn drag_to(x: i32, y: i32) {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);
    let changed = with(|d| {
        let Some(f) = d.focus() else { return false };
        let r = d.windows[f].rect;
        let new = match d.mode {
            Mode::Drag { dx, dy, .. } => {
                let nx = (x - dx as i32).max(screen.x as i32) as u32;
                let ny = (y - dy as i32).max(screen.y as i32) as u32;
                Rect::new(
                    nx.min(screen.x + screen.w.saturating_sub(r.w)),
                    ny.min(screen.y + screen.h.saturating_sub(r.h)),
                    r.w,
                    r.h,
                )
            }
            Mode::DragSize { from, edges } => {
                // Each held edge moves to the pointer; the opposite side stays
                // where it was. Left and top change the origin as well as the
                // extent, which is why the arithmetic is not symmetric.
                let (min_w, min_h) = min_size_of(&d.windows[f]);
                let (mut nx, mut ny) = (from.x, from.y);
                let (mut nw, mut nh) = (from.w, from.h);
                let right = from.x + from.w;
                let bottom = from.y + from.h;

                if edges & edge::LEFT != 0 {
                    let want = x.max(screen.x as i32) as u32;
                    nx = want.min(right.saturating_sub(min_w));
                    nw = right - nx;
                }
                if edges & edge::RIGHT != 0 {
                    let lim = screen.x + screen.w;
                    let want = (x.max(0) as u32).min(lim);
                    nw = want.saturating_sub(from.x).max(min_w);
                    nw = nw.min(lim.saturating_sub(from.x));
                }
                if edges & edge::TOP != 0 {
                    let want = y.max(screen.y as i32) as u32;
                    ny = want.min(bottom.saturating_sub(min_h));
                    nh = bottom - ny;
                }
                if edges & edge::BOTTOM != 0 {
                    let lim = screen.y + screen.h;
                    let want = (y.max(0) as u32).min(lim);
                    nh = want.saturating_sub(from.y).max(min_h);
                    nh = nh.min(lim.saturating_sub(from.y));
                }
                Rect::new(nx, ny, nw.max(min_w), nh.max(min_h))
            }
            _ => return false,
        };
        let moved = new.x != r.x || new.y != r.y || new.w != r.w || new.h != r.h;
        if moved {
            d.windows[f].state = WinState::Normal;
            d.windows[f].rect = new;
        }
        moved
    })
    .unwrap_or(false);
    if changed {
        draw();
    }
}

/// What the pointer is over right now.
fn hover_of(fb: &Framebuffer, x: i32, y: i32) -> Hover {
    let screen = screen_rect(fb);
    if contains(taskbar_rect(fb), x, y) {
        if contains(start_rect(fb), x, y) {
            return Hover::Start;
        }
        let hit = with(|d| {
            task_layout(fb, d)
                .into_iter()
                .enumerate()
                .find(|(_, (r, ..))| contains(*r, x, y))
                .map(|(i, _)| i)
        })
        .flatten();
        return match hit {
            Some(i) => Hover::Task(i),
            None => Hover::None,
        };
    }
    if let Some(i) = window_at(x, y) {
        return with(|d| {
            let w = &d.windows[i];
            let (title, menubar, client) = chrome(w.frame(screen), !w.menus.is_empty());
            if contains(title, x, y) {
                if let Some(which) = theme::caption_buttons(title)
                    .iter()
                    .position(|b| contains(*b, x, y))
                {
                    return Hover::Caption { win: i, which };
                }
            }
            if let Some(bar) = menubar {
                if let Some(mi) = menu_label_at(&w.menus, bar, x, y) {
                    return Hover::MenuLabel { win: i, menu: mi };
                }
            }
            if let Content::Panel(p) = &w.content {
                if let Some(idx) = p.hover_at(client, x, y) {
                    return Hover::Widget { win: i, idx };
                }
            }
            Hover::None
        })
        .unwrap_or(Hover::None);
    }
    match icon_at(fb, x, y) {
        Some(k) => Hover::Icon(k),
        None => Hover::None,
    }
}

/// Repaint only when what the pointer indicates has changed.
fn update_hover(x: i32, y: i32) {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);

    // An open dropdown tracks the pointer with its selection, exactly as the
    // arrows move it.
    let tracked = with(|d| {
        let Some(p) = dropdown_rows(&fb, d, screen) else {
            return None;
        };
        let Some(item) = p.item_at(x, y) else {
            // Off the menu: nothing tracks, and whatever was hot before the
            // menu opened must not stay lit underneath it.
            let stale = d.hover != Hover::None;
            d.hover = Hover::None;
            return Some(stale);
        };
        let moved = match d.mode {
            Mode::Menu { menu, item: cur } if cur != item => {
                d.mode = Mode::Menu { menu, item };
                true
            }
            Mode::Sys { item: cur } if cur != item => {
                d.mode = Mode::Sys { item };
                true
            }
            Mode::Start { item: cur } if cur != item => {
                d.mode = Mode::Start { item };
                true
            }
            _ => false,
        };
        Some(moved)
    })
    .flatten();
    if let Some(moved) = tracked {
        if moved {
            draw();
        }
        return;
    }

    let h = hover_of(&fb, x, y);
    let changed = with(|d| {
        if d.hover != h {
            d.hover = h;
            true
        } else {
            false
        }
    })
    .unwrap_or(false);
    if changed {
        draw();
    }
}

/// The second button opens the menu for whatever is under it.
///
/// Reuses the two menus that already exist rather than inventing a third: over
/// a window it is that window's system menu, which is the same one Alt-Space
/// opens, and over the wall it is the application list the taskbar shows. Both
/// are already driven by the arrow keys and already tested, so the button is a
/// second way in rather than a second implementation.
fn right_click_at(x: i32, y: i32) {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);
    match window_at(x, y) {
        Some(i) => {
            // A program gets first refusal on the second button -- that is
            // how Minesweeper flags. Only if it declines does the button
            // mean the system menu, the meaning it has everywhere else.
            let consumed = with(|d| {
                d.raise(i);
                let f = d.windows.len() - 1;
                let frame = d.windows[f].frame(screen);
                let has_menus = !d.windows[f].menus.is_empty();
                let (_, _, client) = chrome(frame, has_menus);
                match &mut d.windows[f].content {
                    Content::App(a) if contains(client, x, y) => {
                        a.right_press(client, x, y)
                    }
                    _ => false,
                }
            })
            .unwrap_or(false);
            if !consumed {
                with(|d| d.mode = Mode::Sys { item: 0 });
            }
        }
        None => {
            with(|d| d.mode = Mode::Taskbar { item: 0 });
        }
    };
    draw();
}

fn wheel_at(x: i32, y: i32, notches: i32) {
    let Some(i) = window_at(x, y) else { return };
    let hit = with(|d| match &mut d.windows[i].content {
        Content::Browser(b) => {
            b.scroll_by(notches * 3);
            true
        }
        // Over a panel the wheel walks the first list, which is what a wheel
        // over a launcher or a checklist means.
        Content::Panel(p) => p.wheel(notches),
        Content::App(a) => a.wheel(notches),
        _ => false,
    })
    .unwrap_or(false);
    if hit {
        draw();
    }
}

/// The wall: a flat field, a sparse grid, and the mark in the middle.
///
/// The same `splash::aperture` the boot screen draws, not a second copy of the
/// geometry -- five earlier attempts at that logo were wrong in five different
/// ways, and the way to stop a sixth is for there to be exactly one of it.
///
/// Drawn dim. A wallpaper competing with the windows on it is a wallpaper
/// nobody can work in front of, and the cut wedges have to be painted in the
/// wall's own colour anyway, so the mark is a two-colour figure by
/// construction.
fn wallpaper(fb: &Framebuffer) {
    let (w, h) = (fb.width(), fb.height());
    // A horizon rather than a fill. One `vgrad` is the same memset per row the
    // flat colour was, so the sky costs nothing over what it replaced -- and
    // it is what gives a drop shadow somewhere to land, which a near-black
    // wall never did.
    fb.vgrad(0, 0, w, h, &theme::WALL);

    // Ruled, fine and heavy. The dot grid that used to be here was a fixed
    // colour, which is subtle over one part of a gradient and dirt over the
    // rest; these are a percentage of white, so they are equally faint over
    // near-black water and over gold. Span fills, not lines: a graticule of
    // rectangles is memsets where a lattice of diagonals would be Bresenham.
    let mut x = 0;
    while x < w {
        let n = if x % theme::WALL_MAJOR == 0 { theme::WALL_MAJOR_NUM } else { theme::WALL_MINOR_NUM };
        fb.tint_rect(x, 0, 1, h, theme::WALL_INK, n);
        x += theme::WALL_MINOR;
    }
    let mut y = 0;
    while y < h {
        let n = if y % theme::WALL_MAJOR == 0 { theme::WALL_MAJOR_NUM } else { theme::WALL_MINOR_NUM };
        fb.tint_rect(0, y, w, 1, theme::WALL_INK, n);
        y += theme::WALL_MINOR;
    }

    let (cx, cy) = ((w / 2) as i32, (h / 2) as i32);
    let r = (h / 5).min(w / 5) as i32;

    // The dial, and graduations pointing outward from it. Drawn before the
    // atom and before the bubbles, so it sits behind both.
    let dial = r * theme::WALL_DIAL as i32 / 100;
    fb.circle(cx, cy, dial, theme::WALL_ORBIT);
    for i in 0..theme::WALL_TICKS {
        let (sx, sy) = unit(i, theme::WALL_TICKS);
        let long = i % 6 == 0;
        let out = dial + if long { r / 9 } else { r / 22 };
        fb.line(
            cx + sx * dial / 1024,
            cy + sy * dial / 1024,
            cx + sx * out / 1024,
            cy + sy * out / 1024,
            theme::WALL_ORBIT,
        );
    }

    // The atom: three ellipses through the same nucleus at even tilts, and an
    // electron on each. The mark itself is the nucleus and is drawn last,
    // unchanged -- it is already a ring with a filled centre, which is what
    // the middle of one of these diagrams has always been.
    let (a, b) = (
        r * theme::WALL_ORBIT_A as i32 / 100,
        r * theme::WALL_ORBIT_B as i32 / 100,
    );
    let tilts = theme::WALL_ORBIT_TILTS;
    for t in 0..tilts {
        ellipse(fb, cx, cy, a, b, t, tilts * 2, theme::WALL_ORBIT);
        let (px, py) = on_ellipse(cx, cy, a, b, t, tilts * 2, theme::WALL_ELECTRONS[t as usize]);
        fb.fill_circle(px, py, (r / 20).max(3), theme::WALL_RIM);
        fb.fill_circle(px - r / 60, py - r / 60, (r / 60).max(1), theme::WALL_SPEC);
    }

    // Light on it. A bubble is a wash rather than a fill, so the sky and the
    // ruling both carry on through it -- which is the whole difference
    // between glass and a dot.
    for (bx, by, br) in theme::WALL_BUBBLES {
        bubble(
            fb,
            (w * bx / 100) as i32,
            (h * by / 100) as i32,
            (h * br / 100).max(4) as i32,
        );
    }

    // Filled blades, and the gaps between them filled with the sky that would
    // have been there. Which is to say the gaps are the sky: a cut here takes
    // its colour from the same ramp the wall was drawn from, row by row, so
    // there is nothing standing in for the background because it is the
    // background.
    super::splash::aperture_with(
        fb,
        cx,
        cy,
        r,
        super::splash::Face::Ramp(&theme::SUN),
        super::splash::Cut::Sky { stops: &theme::WALL, top: 0, height: h },
    );
}

/// A point on the unit circle, scaled by 1024, for step `i` of `n`.
///
/// Integer, and derived from one eighth of a turn by symmetry rather than
/// from a table of `n` entries, so changing the graduation count needs no
/// second table to keep in step with it.
fn unit(i: u32, n: u32) -> (i32, i32) {
    // 1024 * sin(k * pi / 32) for k in 0..=16: a quarter turn, and every other
    // angle is one of these with a sign or the pair swapped.
    const Q: [i32; 17] = [
        0, 100, 200, 297, 391, 483, 569, 650, 724, 792, 851, 903, 946, 979, 1004, 1019, 1024,
    ];
    let step = (i * 64 / n.max(1)) % 64;
    let (quad, k) = (step / 16, (step % 16) as usize);
    let (s, c) = (Q[k], Q[16 - k]);
    match quad {
        0 => (c, s),
        1 => (-s, c),
        2 => (-c, -s),
        _ => (s, -c),
    }
}

/// One point of a tilted ellipse, scaled and rotated in integers.
fn on_ellipse(cx: i32, cy: i32, a: i32, b: i32, ti: u32, tn: u32, i: u32) -> (i32, i32) {
    let (tc, ts) = unit(ti, tn);
    let (c, s) = unit(i, 64);
    let (ex, ey) = (a * c / 1024, b * s / 1024);
    (
        cx + (ex * tc - ey * ts) / 1024,
        cy + (ex * ts + ey * tc) / 1024,
    )
}

/// A tilted ellipse, as sixty-four chords.
///
/// Sixty-four because that is the resolution of `unit`'s table, and it is
/// already more than enough: on the largest orbit here the chord is about
/// thirty pixels and its sagitta is under half of one, so the curve is smooth
/// before it is drawn rather than by being subdivided until it looks it.
fn ellipse(fb: &Framebuffer, cx: i32, cy: i32, a: i32, b: i32, ti: u32, tn: u32, col: Color) {
    let mut prev = on_ellipse(cx, cy, a, b, ti, tn, 0);
    for i in 1..=64 {
        let p = on_ellipse(cx, cy, a, b, ti, tn, i % 64);
        fb.line(prev.0, prev.1, p.0, p.1, col);
        prev = p;
    }
}

/// One glass bubble: a wash, a rim, and a specular.
fn bubble(fb: &Framebuffer, cx: i32, cy: i32, r: i32) {
    for dy in -r..=r {
        let half = super::isqrt((r * r - dy * dy).max(0) as u32) as i32;
        let y = cy + dy;
        let x0 = (cx - half).max(0);
        let x1 = cx + half;
        if y < 0 || x1 <= x0 {
            continue;
        }
        fb.tint_span(
            x0 as u32,
            y as u32,
            (x1 - x0) as u32,
            theme::WALL_INK,
            theme::WALL_BUBBLE_NUM,
        );
    }
    fb.circle(cx, cy, r, theme::WALL_RIM);
    // Up and to the left, which is where every light source in this interface
    // already is -- the bevel says so and so does the gloss on every ramp.
    fb.fill_circle(cx - r / 3, cy - r / 3, (r / 6).max(1), theme::WALL_SPEC);
}

/// Buttons on the bar, one per window, in stacking order: the window's
/// pictogram and whether it is the focused one.
///
/// No launcher buttons: launching lives in the Start menu and on the wall,
/// so the bar never runs out of room for the windows it exists to hold.
fn task_slots(d: &Desktop) -> Vec<(usize, bool)> {
    let focus = d.focus();
    d.windows
        .iter()
        .enumerate()
        .map(|(i, w)| (w.icon, Some(i) == focus))
        .collect()
}

/// Every taskbar button's rectangle, in slot order.
///
/// The single source for where the buttons are: the paint pass draws these
/// rectangles and the pointer hit-tests them, so a button cannot highlight in
/// one place and press in another.
fn task_layout(fb: &Framebuffer, d: &Desktop) -> Vec<(Rect, usize, bool)> {
    let bar = taskbar_rect(fb);
    let slots = task_slots(d);
    let btn_h = bar.h - 8;
    let y = bar.y + 4;
    let start = start_rect(fb);
    let mut x = start.x + start.w + 10;
    let mut out = Vec::new();

    for (icon, pressed) in slots {
        // A pictogram and its air: every button the same width, which is what
        // lets a bar of nine windows still read as a row rather than a ransom
        // note. The title lives on the window; the bar says what *kind*.
        let w = btn_h + 10;
        // Stop before whatever is on the right rather than drawing under it.
        // The charge well sits left of the clock when there is one, so the
        // buttons have to break against it and not against the clock.
        let right = battery_rect(fb).map(|r| r.x).unwrap_or_else(|| clock_rect(fb).x);
        if x + w > right {
            break;
        }
        out.push((Rect::new(x, y, w, btn_h), icon, pressed));
        x += w + TASK_GAP;
    }
    out
}

/// The bar along the bottom: apps, then open windows, then the clock.
///
/// A minimised window is a task button that is not pressed, which is why the
/// icons that used to sit on the wall are gone. An icon on the wallpaper is
/// findable only if no window covers it, and something you stow is exactly
/// what you then cannot find.
fn taskbar(fb: &Framebuffer, d: &Desktop, sel: Option<usize>) {
    let bar = taskbar_rect(fb);
    // Glass, so the sky shows through it. The wallpaper is painted before this
    // and the back-to-front repaint means what is underneath is already
    // composed -- which is the whole reason a translucent bar costs one blended
    // span per row and no second pass over anything.
    fb.glass(bar.x, bar.y, bar.w, bar.h, theme::GLASS, &theme::GLASS_STOPS);
    // The specular sheen over the top, and then one hard line where the pane
    // meets the sky. The opacity table is proportional and cannot say "one
    // pixel" at any height, which is the division of labour between a ramp and
    // the line drawn on top of it.
    fb.glass(bar.x, bar.y, bar.w, bar.h / 2, theme::GLOSS, &theme::GLOSS_STOPS);
    fb.rect(bar.x, bar.y, bar.w, 1, theme::TASK_EDGE);

    // The Start button: the mark and the name. Held down while its menu is
    // open, which is the one place the bar states a mode rather than a focus.
    let s = start_rect(fb);
    let start_open = matches!(d.mode, Mode::Start { .. });
    let lit = start_open || d.hover == Hover::Start;
    theme::control(
        fb,
        s,
        if lit { &theme::START_HOT } else { &theme::START },
        theme::START_EDGE,
    );
    theme::aperture_dot(fb, s.x + 15, s.y + s.h / 2, (s.h / 2) as i32 - 5);
    let ty = s.y + (s.h.saturating_sub(theme::text_h())) / 2;
    theme::text_over(fb, s.x + 30, ty, "GLaDOS", theme::START_TEXT);

    for (i, (r, icon, pressed)) in task_layout(fb, d).into_iter().enumerate() {
        // Keyboard selection and pointer hover draw the same way: both are "the
        // next click or Enter lands here", and two different highlights would
        // claim two different things.
        let hot = sel == Some(i) || d.hover == Hover::Task(i);
        // `control` rather than `button`: a task button wants the hover wash,
        // and `button` reads its one boolean as keyboard focus. The two mean
        // different things and only here do they need telling apart.
        let stops: &[(u8, Color)] = if pressed {
            &theme::BTN_DOWN
        } else if hot {
            &theme::BTN_HOT
        } else {
            &theme::BTN
        };
        theme::control(fb, r, stops, if hot { theme::BTN_EDGE_HOT } else { theme::BTN_EDGE });
        let s = r.h.saturating_sub(6);
        // Nudged a pixel when pressed, the same lie about depth the label
        // used to tell.
        let off = u32::from(pressed);
        pictogram(
            fb,
            icon,
            r.x + (r.w.saturating_sub(s)) / 2 + off,
            r.y + 3 + off,
            s,
            theme::FACE,
        );
    }

    // The tray recesses. Not `theme::well`: a well outlines a hole in a face,
    // and on a coloured bar the thing that reads as a recess is being darker
    // than what surrounds it.
    let recess = |r: Rect| {
        fb.rect(r.x, r.y, r.w, r.h, theme::TRAY);
        theme::outline(fb, r, theme::TRAY_EDGE);
    };
    recess(clock_rect(fb));
    if let Some(b) = battery_rect(fb) {
        recess(b);
    }
}

/// Where the charge readout goes, when there is one.
///
/// Its own well rather than sharing the clock's. The clock's is exactly
/// thirteen columns because every column it takes is one the task buttons do
/// not get, and a string one character too long is dropped in silence rather
/// than truncated -- which is how it once came to be blank. Widening it to fit
/// a percentage would repeat that, so this is six columns of its own, and it
/// only exists on a machine that has a battery to put in it.
pub fn battery_rect(fb: &Framebuffer) -> Option<Rect> {
    let c = crate::dev::battery::status()?;
    if !c.present {
        return None;
    }
    let bar = taskbar_rect(fb);
    let w = theme::text_w(6);
    let clock = clock_rect(fb);
    Some(Rect::new(clock.x.saturating_sub(w + 4), bar.y + 4, w, bar.h - 8))
}

/// What the charge well says: a percentage, and a mark for the source.
///
/// Six columns, so "+100%" fits with a space either side. The plus is mains
/// and the minus is battery, which is one character to say the thing an
/// operator most wants at a glance.
pub fn battery_text() -> Option<alloc::string::String> {
    let c = crate::dev::battery::status()?;
    if !c.present {
        return None;
    }
    let mark = match c.on_ac {
        Some(true) => '+',
        Some(false) => '-',
        None => ' ',
    };
    Some(alloc::format!("{}{}% ", mark, c.percent))
}

/// Where the uptime readout goes. Right-hand end of the bar.
pub fn clock_rect(fb: &Framebuffer) -> Rect {
    let bar = taskbar_rect(fb);
    // Wide enough for " up 1234.5s " and no wider. Reserving more silently
    // steals room from the task buttons -- which is how the terminal's own
    // button came to be missing from the bar, and how the clock came to be
    // blank: the string was one character longer than its well and the draw
    // refused rather than overflowing.
    let w = theme::text_w(13);
    Rect::new(
        bar.x + bar.w.saturating_sub(w + 6),
        bar.y + 4,
        w,
        bar.h - 8,
    )
}

pub fn draw() {
    let Some(real) = super::primary() else {
        return;
    };
    // One painter at a time, for the whole frame.
    //
    // The claim covers `cursor_hide` at the top through `cursor_show` at the
    // bottom, because the window between them is the race: a cursor move that
    // lands after the hide and before the present saves pixels from a frame
    // that is about to be replaced. Holding it only around each cursor call
    // would leave that gap wide open.
    //
    // It also makes `draw` single-writer, which it has never been -- the agent
    // and author tasks both call it while the shell may be mid-frame, sharing
    // the back buffer, the shadow and the cursor statics with no lock at all.
    // Returning is the right answer to losing: the holder is painting the same
    // desktop this call would have painted.
    let Some(_claim) = Claim::take() else {
        return;
    };
    let screen = screen_rect(&real);
    // Lift the pointer before presenting, or the pixels saved under it are
    // stale the moment anything below moves, and putting them back paints a
    // rectangle of the previous frame onto the new one.
    cursor_hide(&real);
    // Compose into the back buffer when there is one; the screen then gets
    // only the rows that changed. Without one (no heap yet, or its
    // allocation failed) this is the direct draw it always was.
    let fb = super::compose::target().unwrap_or(real);
    with(|d| {
        // The terminal is an application, not the screen itself. While its
        // window is minimised the shell keeps running -- it still reads serial,
        // still answers, and its output still lands in the console's shadow
        // grid -- but nothing may reach the framebuffer, or the prompt paints
        // straight over the desktop it is supposed to be behind. That leak is
        // what drew a prompt and a black bar across the wallpaper.
        //
        // Cleared here and set again only if the terminal is actually drawn
        // below, so visibility is *derived* from whether it was painted rather
        // than tracked alongside the window state and able to disagree with it.
        super::console::with(|c| c.set_visible(false));
        wallpaper(&fb);
        draw_icons(&fb, d.hover);
        let focus = d.focus();
        let sel = match d.mode {
            Mode::Taskbar { item } => Some(item),
            _ => None,
        };
        taskbar(&fb, d, sel);

        // Every console starts this pass unpainted, and the loop below turns
        // back on the ones it actually draws.
        //
        // A minimised window is skipped by the `continue` below, so its console
        // was never reflowed *and* never told it was hidden: it kept the origin
        // and the visible flag from the last time it was drawn, and the next
        // line written to it went straight to the framebuffer at that stale
        // position -- over the desktop, outside any window. The Executive does
        // this on its own schedule, so minimising it and waiting produced
        // `[mind t16] disabled` painted on the wallpaper.
        //
        // Nothing is lost by hiding one. An invisible console still updates its
        // shadow grid, and `redraw_ch` repaints the whole log when its window
        // comes back.
        for ch in 0..super::console::NCONSOLE {
            super::console::with_ch(ch, |c| c.set_visible(false));
        }
        for (i, win) in d.windows.iter().enumerate() {
            if win.state == WinState::Minimised {
                continue;
            }
            let active = Some(i) == focus;
            let frame = win.frame(screen);
            let hot = match d.hover {
                Hover::Caption { win, which } if win == i => Some(which),
                _ => None,
            };
            // The shadow, before the window and after everything under it.
            //
            // Back to front is what makes this correct rather than careful:
            // whatever is already composed here -- wall, icons, an earlier
            // window -- is what gets darkened, and the only thing that can
            // cover it afterwards is a window in front, which is exactly what
            // should. No compositing, no second pass, no order to get right
            // beyond the one the loop already has.
            if win.state != WinState::Maximised {
                let (off, wide) = (theme::SHADOW_OFF, theme::SHADOW_W);
                let right = frame.x + frame.w;
                let bottom = frame.y + frame.h;
                // Clamped to the screen: `snap_release` can put a window flush
                // against an edge, and `MARGIN` only protects the un-snapped
                // case.
                let edge_x = screen.x + screen.w;
                let edge_y = screen.y + screen.h;
                let band_w = wide.min(edge_x.saturating_sub(right));
                let band_h = wide.min(edge_y.saturating_sub(bottom));
                // The right band starts a radius below the top. Without that
                // the offset silhouette leaves a three-pixel nub above the
                // rounded top-right corner, which is the one place a hard
                // shadow gives the rounding away.
                let top = frame.y + off + win.round;
                fb.shade_rect(
                    right,
                    top,
                    band_w,
                    (bottom + off).saturating_sub(top),
                    theme::SHADOW_NUM,
                );
                fb.shade_rect(
                    frame.x + off,
                    bottom,
                    (frame.w + band_w).saturating_sub(off),
                    band_h,
                    theme::SHADOW_NUM,
                );
            }

            // Confine this window's paint to its own outline. A rectangular
            // window costs a load and a branch for the whole frame; a shaped
            // one simply does not write outside itself, and whatever the
            // back-to-front repaint already put there shows through.
            with_window_shape(win, frame, || {
                let c = theme::window(
                    &fb,
                    frame,
                    &win.title,
                    active,
                    win.state == WinState::Maximised,
                    !win.menus.is_empty(),
                    hot,
                );
                let mut client = c.client;
                if client.is_empty() {
                    // Nothing left to draw inside the frame; the chrome is
                    // already painted. `return` and not `continue`, because
                    // this is the shape closure and not the loop.
                    return;
                }

                if let Some(bar) = c.menubar {
                    theme::panel(&fb, bar);
                    let open = match d.mode {
                        Mode::Menu { menu, .. } if active => Some(menu),
                        _ => None,
                    };
                    let labels = theme::menu_labels(bar, win.menus.iter().map(|m| m.label.as_str()));
                    for (mi, (m, lr)) in win.menus.iter().zip(labels).enumerate() {
                        let hot = Some(mi) == open
                            || d.hover == Hover::MenuLabel { win: i, menu: mi };
                        let (fg, bg) = if hot {
                            (theme::SELECT_TEXT, theme::SELECT)
                        } else {
                            (theme::TEXT, theme::FACE)
                        };
                        let ty = lr.y + (lr.h.saturating_sub(theme::text_h())) / 2;
                        fb.rect(lr.x, lr.y + 1, lr.w, lr.h.saturating_sub(2), bg);
                        theme::text(&fb, lr.x + 6, ty, &m.label, fg, bg);
                    }
                }

                match &win.content {
                    Content::Terminal(ch) => {
                        let well = client.shrink(2);
                        theme::well(&fb, well, theme::SCREEN);
                        let grid = well.shrink(3);
                        // Each grid reflows into its own window. They are
                        // different sizes and hold different text, so asking
                        // for "the console" here would paint one of them
                        // twice and the other never.
                        super::console::with_ch(*ch, |c| {
                            c.set_visible(true);
                            c.reflow(grid.x, grid.y, grid.w, grid.h);
                        });
                        super::console::redraw_ch(*ch);
                    }
                    Content::Panel(p) => {
                        theme::panel(&fb, client);
                        let hov = match d.hover {
                            Hover::Widget { win, idx } if win == i => Some(idx),
                            _ => None,
                        };
                        p.draw_in(&fb, client, active, hov);
                    }
                    Content::Browser(b) => b.draw_in(&fb, client, active),
                    Content::App(a) => a.draw_in(&fb, client, active),
                }
            });
            }

        // Popups last, over everything, because that is what a popup is.
        if let Some(f) = focus {
            let frame = d.windows[f].frame(screen);
            match d.mode {
                Mode::Menu { menu, item } => {
                    if let Some(m) = d.windows[f].menus.get(menu) {
                        let inner = frame.shrink(theme::FRAME);
                        let bar = Rect::new(inner.x, inner.y + theme::TITLE_H + 2, inner.w, MENU_H);
                        let Some(lr) =
                            theme::menu_labels(bar, d.windows[f].menus.iter().map(|p| p.label.as_str()))
                                .nth(menu)
                        else {
                            return;
                        };
                        dropdown(
                            &fb,
                            lr.x,
                            inner.y + theme::TITLE_H + 2 + MENU_H,
                            m.items.iter().map(|i| i.label.as_str()),
                            item,
                        );
                    }
                }
                Mode::Sys { item } => {
                    let inner = frame.shrink(theme::FRAME);
                    dropdown(
                        &fb,
                        inner.x,
                        inner.y + theme::TITLE_H,
                        SYS_ITEMS.iter().copied(),
                        item,
                    );
                }
                Mode::Move { .. } | Mode::Size { .. } => {
                    // A hint, because a mode with no indication that it is on is
                    // a keyboard that has stopped working.
                    let msg = if matches!(d.mode, Mode::Move { .. }) {
                        " Move: arrows, Enter to place, Esc to cancel "
                    } else {
                        " Size: arrows, Enter to keep, Esc to cancel "
                    };
                    let w = theme::text_w_of(msg);
                    let r = Rect::new(
                        (fb.width().saturating_sub(w)) / 2,
                        MARGIN,
                        w,
                        MENU_H,
                    );
                    theme::panel(&fb, r);
                    theme::text(&fb, r.x, r.y + 5, msg, theme::TEXT, theme::FACE);
                }
                // The taskbar draws its own selection, and it is not a popup
                // over the focused window -- it is a fixture with the keyboard.
                // A pointer drag needs no hint: the window moving under the
                // pointer is the indication. Start draws below, unconditional
                // on focus.
                Mode::Normal
                | Mode::Taskbar { .. }
                | Mode::Drag { .. }
                | Mode::DragSize { .. }
                | Mode::Start { .. } => {}
            }
        }
        if let Mode::Start { item } = d.mode {
            let p = start_menu_rect(&fb);
            // The panel is drawn to the full height including the query row,
            // then the items over it, then the query row last. It carries one
            // more row than it has items, which is why it builds its own
            // `Popup` rather than going through `dropdown`.
            theme::popup(&fb, &p);
            if p.wide_enough() {
                for (i, (label, _)) in START_ITEMS.iter().enumerate() {
                    theme::list_row(&fb, p.row(i), label, i == item, true);
                }
                let row = p.row(START_ITEMS.len());
                // A well, not a list row: it is a place to type, and it should
                // not look like something that runs when pressed.
                theme::well(&fb, row, theme::HILIGHT);
                let shown = if d.query.is_empty() {
                    alloc::string::String::from("Type to search")
                } else {
                    // The tail, so the end being typed stays visible rather
                    // than the beginning that has already been read.
                    let n = d.query.chars().count();
                    d.query.chars().skip(n.saturating_sub(QUERY_COLS - 1)).collect()
                };
                let fg = if d.query.is_empty() { theme::TEXT_DIM } else { theme::TEXT };
                theme::text(&fb, row.x + 4, row.y + 4, &shown, fg, theme::HILIGHT);
                // A caret, so a selected empty box does not read as inert.
                if is_query_row(item) {
                    let cx = row.x + 4 + theme::text_w(shown.chars().count());
                    fb.rect(cx, row.y + 4, 2, theme::text_h(), theme::TEXT);
                }
            }
        }
    });
    super::compose::present();
    // Put the pointer back where it was. Without this every keystroke that
    // repaints would blink the arrow out until the mouse next moved.
    if let Some((px, py)) = unsafe { *POS.get() } {
        cursor_show(&real, px, py);
    }
}

fn dropdown<'a>(
    fb: &Framebuffer,
    x: u32,
    y: u32,
    labels: impl Iterator<Item = &'a str> + Clone,
    sel: usize,
) {
    let items: Vec<&str> = labels.collect();
    if items.is_empty() {
        return;
    }
    let cols = items.iter().map(|s| s.chars().count()).max().unwrap_or(4);
    // The same construction `dropdown_rows` answers with, so the paint and the
    // hit-test cannot come to different conclusions about the same menu.
    let p = theme::Popup::sized(x, y, cols, items.len(), fb.width().saturating_sub(x));
    theme::popup(fb, &p);
    if !p.wide_enough() {
        return;
    }
    for (i, label) in items.iter().enumerate() {
        theme::list_row(fb, p.row(i), label, i == sel, true);
    }
}

pub fn focus_is_terminal() -> bool {
    with(|d| match d.focus() {
        Some(f) => matches!(d.windows[f].content, Content::Terminal(c) if c == super::console::USER),
        // No window at all: the shell has the keyboard. The terminal is an
        // application over a shell that never stops, so with every window
        // minimised, typing types at the prompt -- invisibly, but a command
        // and Enter still run, and the output is waiting when the terminal
        // comes back. `false` here made an all-minimised desktop swallow
        // the keyboard entirely, which read as a hang.
        None => true,
    })
    .unwrap_or(true)
}

/// Alt-Tab: swap the top two visible windows.
///
/// A swap, not a rotation. This used to send the top window to the back,
/// which walks all windows eventually but makes a second Alt-Tab land on a
/// *third* window -- pressed twice, it went somewhere new instead of back.
/// Windows has toggled the top pair on a single press since 3.1, for the
/// reason that became obvious here the hard way: a headless script (and a
/// person) needs "over and back" to be two presses, deterministically. It
/// cost a test run in which the second Alt-Tab focused the Program Manager
/// and the next command line ran one of its rows.
pub fn cycle(_back: bool) {
    with(|d| {
        let visible: Vec<usize> = d
            .windows
            .iter()
            .enumerate()
            .filter(|(_, w)| w.state != WinState::Minimised)
            .map(|(i, _)| i)
            .collect();
        if visible.len() < 2 {
            return;
        }
        // The two topmost visible windows; raising the lower one swaps them.
        let below = visible[visible.len() - 2];
        d.raise(below);
    });
    trace("alt-tab");
    draw();
}

pub fn trace(event: &str) {
    with(|d| {
        let focus = d.focus();
        crate::serial_println!("[desk] {}", event);
        for (i, w) in d.windows.iter().enumerate() {
            let state = match w.state {
                WinState::Normal => "normal",
                WinState::Maximised => "max",
                WinState::Minimised => "min",
            };
            crate::serial_println!(
                "[desk] {} {} {} {}x{}+{}+{}",
                if Some(i) == focus { '>' } else { ' ' },
                w.title,
                state,
                w.rect.w,
                w.rect.h,
                w.rect.x,
                w.rect.y
            );
        }
    });
}

pub enum Route {
    Handled,
    Shell(u8),
}

/// Act on a system-menu choice.
fn sys_action(d: &mut Desktop, f: usize, item: usize, screen: Rect) {
    match item {
        0 => d.mode = Mode::Move { from: d.windows[f].rect },
        1 => d.mode = Mode::Size { from: d.windows[f].rect },
        2 => {
            d.windows[f].state = match d.windows[f].state {
                WinState::Maximised => WinState::Normal,
                _ => WinState::Maximised,
            };
            let _ = screen;
        }
        3 => {
            if d.windows[f].closable {
                d.windows.remove(f);
            } else {
                // The terminal is the shell. Minimising is the honest thing to
                // offer instead of pretending a close happened.
                d.windows[f].state = WinState::Minimised;
            }
        }
        _ => {}
    }
}

/// Put the keyboard back where a person would expect it.
///
/// The desktop takes *every* key while it is in a menu mode, which is correct
/// when somebody opened the menu and catastrophic when nobody did. QEMU's
/// i8042 probe emits scancodes at boot -- the entropy ring already counts one
/// of them as a phantom touch -- and under the hypervisor accelerator that
/// probe was leaving the machine sitting in the Start menu before the shell
/// had printed its prompt. Every byte arriving over the serial line then fed
/// the menu instead of the command line, so a driven session looked like a
/// guest that had stopped reading its UART.
///
/// Called when the shell announces it is interactive. Nothing a controller
/// does while nobody is at the keyboard should decide where the next
/// keystroke goes.
pub fn dismiss_menus() {
    if !ready() {
        return;
    }
    let changed = with(|d| {
        let was = !matches!(d.mode, Mode::Normal);
        d.mode = Mode::Normal;
        was
    })
    .unwrap_or(false);
    if changed {
        draw();
    }
}

pub fn key(k: u8) -> Route {
    if !ready() {
        return Route::Shell(k);
    }
    let Some(fb) = super::primary() else {
        return Route::Shell(k);
    };
    let screen = screen_rect(&fb);

    // Window management first, in every mode, so there is no state the keyboard
    // can get stuck in.
    if k == kbd::KEY_ALTTAB {
        with(|d| d.mode = Mode::Normal);
        cycle(false);
        return Route::Handled;
    }
    // Before the mode dispatch, like Alt-Tab, so there is no state the
    // keyboard can get stuck in: pressing it again closes the menu.
    if k == kbd::KEY_STARTMENU {
        with(|d| {
            d.mode = match d.mode {
                Mode::Start { .. } => Mode::Normal,
                _ => Mode::Start { item: START_ITEMS.len() },
            };
            d.query.clear();
        });
        draw();
        return Route::Handled;
    }
    if k == kbd::KEY_TASKBAR {
        with(|d| {
            d.mode = match d.mode {
                // A second Ctrl-Esc puts the keyboard back, so the bar cannot
                // become somewhere you get stuck.
                Mode::Taskbar { .. } => Mode::Normal,
                _ => Mode::Taskbar { item: 0 },
            }
        });
        draw();
        return Route::Handled;
    }

    let mode = with(|d| d.mode).unwrap_or(Mode::Normal);

    match mode {
        Mode::Normal => {
            if k == kbd::KEY_SYSMENU {
                with(|d| d.mode = Mode::Sys { item: 0 });
                draw();
                return Route::Handled;
            }
            if k == kbd::KEY_MENU {
                let opened = with(|d| match d.focus() {
                    Some(f) if !d.windows[f].menus.is_empty() => {
                        d.mode = Mode::Menu { menu: 0, item: 0 };
                        true
                    }
                    _ => false,
                })
                .unwrap_or(false);
                if opened {
                    draw();
                    return Route::Handled;
                }
                return Route::Handled;
            }
            if focus_is_terminal() {
                return Route::Shell(k);
            }
            // A panel has the keyboard.
            let step = with(|d| match d.focus() {
                Some(f) => match &mut d.windows[f].content {
                    Content::Panel(p) => p.key(k),
                    // The browser answers "did I use it", which is all the
                    // desktop needs: anything it declines falls through to the
                    // window manager, so Alt-Tab keeps working inside a page.
                    Content::Browser(b) => {
                        if b.key(k) { ui::Step::Redraw } else { ui::Step::Idle }
                    }
                    Content::App(a) => {
                        if a.key(k) { ui::Step::Redraw } else { ui::Step::Idle }
                    }
                    _ => ui::Step::Idle,
                },
                None => ui::Step::Idle,
            });
            match step {
                Some(ui::Step::Redraw) => draw(),
                Some(ui::Step::Close) => close_focused(),
                Some(ui::Step::Do(Action::Run(cmd))) => {
                    focus_terminal();
                    unsafe { *PENDING.get() = Some(cmd) };
                }
                // Navigation replaces the panel where it stands. The window
                // keeps its geometry, its place in the z-order and the
                // keyboard -- browsing is not opening something, and a browser
                // that jumped to the front of the stack on every keystroke
                // would be unusable.
                Some(ui::Step::Do(Action::Browse(route))) => {
                    // The desktop resolves a route without knowing what kind
                    // of app is on the other end of it.
                    if let Some((title, panel)) = ui::panel_for_route(&route) {
                        let kind = route.split(':').next().unwrap_or("");
                        with(|d| {
                            if let Some(f) = d.focus() {
                                d.windows[f].title = title;
                                d.windows[f].icon = panel_icon(kind);
                                d.windows[f].content = Content::Panel(panel);
                            }
                        });
                        draw();
                    }
                }
                _ => {}
            }
            Route::Handled
        }

        Mode::Sys { item } => {
            with(|d| {
                let Some(f) = d.focus() else { return };
                match k {
                    kbd::KEY_DOWN => {
                        d.mode = Mode::Sys { item: (item + 1) % SYS_ITEMS.len() }
                    }
                    kbd::KEY_UP => {
                        d.mode = Mode::Sys {
                            item: (item + SYS_ITEMS.len() - 1) % SYS_ITEMS.len(),
                        }
                    }
                    b'\n' | b'\r' => {
                        d.mode = Mode::Normal;
                        sys_action(d, f, item, screen);
                    }
                    27 => d.mode = Mode::Normal,
                    _ => {}
                }
            });
            draw();
            Route::Handled
        }

        Mode::Menu { menu, item } => {
            with(|d| {
                let Some(f) = d.focus() else { return };
                let n_menus = d.windows[f].menus.len();
                let n_items = d.windows[f].menus[menu].items.len();
                match k {
                    kbd::KEY_RIGHT => d.mode = Mode::Menu { menu: (menu + 1) % n_menus, item: 0 },
                    kbd::KEY_LEFT => {
                        d.mode = Mode::Menu { menu: (menu + n_menus - 1) % n_menus, item: 0 }
                    }
                    kbd::KEY_DOWN => {
                        d.mode = Mode::Menu { menu, item: (item + 1) % n_items.max(1) }
                    }
                    kbd::KEY_UP => {
                        d.mode = Mode::Menu {
                            menu,
                            item: (item + n_items.saturating_sub(1)) % n_items.max(1),
                        }
                    }
                    b'\n' | b'\r' => {
                        d.mode = Mode::Normal;
                        if let Some(mi) = d.windows[f].menus[menu].items.get(item) {
                            if let Action::Run(cmd) = &mi.action {
                                unsafe { *PENDING.get() = Some(cmd.clone()) };
                            }
                        }
                    }
                    27 => d.mode = Mode::Normal,
                    _ => {}
                }
            });
            draw();
            Route::Handled
        }

        Mode::Taskbar { item } => {
            let n = with(|d| task_slots(d).len()).unwrap_or(0);
            if n == 0 {
                return Route::Handled;
            }
            match k {
                kbd::KEY_RIGHT => {
                    with(|d| d.mode = Mode::Taskbar { item: (item + 1) % n });
                    draw();
                }
                kbd::KEY_LEFT => {
                    with(|d| d.mode = Mode::Taskbar { item: (item + n - 1) % n });
                    draw();
                }
                b'\n' | b'\r' => {
                    with(|d| d.mode = Mode::Normal);
                    with(|d| {
                        if item < d.windows.len() {
                            // A task button restores a minimised window, which
                            // is the only way back for one.
                            if d.windows[item].state == WinState::Minimised {
                                d.windows[item].state = WinState::Normal;
                            }
                            d.raise(item);
                        }
                    });
                    draw();
                }
                27 => {
                    with(|d| d.mode = Mode::Normal);
                    draw();
                }
                _ => {}
            }
            Route::Handled
        }

        Mode::Start { item } => {
            let n = start_rows();
            match k {
                kbd::KEY_DOWN => {
                    with(|d| d.mode = Mode::Start { item: (item + 1) % n });
                    draw();
                }
                kbd::KEY_UP => {
                    with(|d| d.mode = Mode::Start { item: (item + n - 1) % n });
                    draw();
                }
                b'\n' | b'\r' => {
                    // The query row runs what was typed; every other row runs
                    // its own command. `open` is the same dispatcher the search
                    // panel uses, so a name means the same thing typed here as
                    // typed there -- an application opens, a command runs, and
                    // anything else offers to be written.
                    let typed = with(|d| {
                        let q = alloc::string::String::from(d.query.trim());
                        d.mode = Mode::Normal;
                        d.query.clear();
                        q
                    })
                    .unwrap_or_default();
                    if is_query_row(item) {
                        if !typed.is_empty() {
                            launch(&alloc::format!("open {}", typed));
                        }
                    } else {
                        launch(START_ITEMS[item].1);
                    }
                    draw();
                }
                27 => {
                    with(|d| {
                        d.mode = Mode::Normal;
                        d.query.clear();
                    });
                    draw();
                }
                8 => {
                    // Backspace edits the query wherever the selection is, and
                    // moves to it. Typing is the reason the row exists; making
                    // it reachable only by arrowing to it first would be a
                    // search box that has to be found before it can be used.
                    let moved = with(|d| {
                        d.query.pop();
                        d.mode = Mode::Start { item: START_ITEMS.len() };
                        !d.query.is_empty()
                    })
                    .unwrap_or(false);
                    let _ = moved;
                    draw();
                }
                // Printable ASCII. Typing anywhere in the menu goes to the
                // query and selects it, which is what Windows has done since
                // Vista and is the only behaviour that makes the box worth
                // having -- press Start, type, press Enter.
                c if (0x20..0x7F).contains(&c) => {
                    with(|d| {
                        if d.query.chars().count() < 64 {
                            d.query.push(c as char);
                        }
                        d.mode = Mode::Start { item: START_ITEMS.len() };
                    });
                    draw();
                }
                _ => {}
            }
            Route::Handled
        }

        // A pointer drag owns the screen until the button lifts; the keyboard
        // can only abandon it. Anything else mid-drag is swallowed, because a
        // window that responds to typing while it is being dragged is two
        // interfaces fighting over one object.
        Mode::Drag { from, .. } | Mode::DragSize { from, .. } => {
            if k == 27 {
                with(|d| {
                    if let Some(f) = d.focus() {
                        d.windows[f].rect = from;
                    }
                    d.mode = Mode::Normal;
                });
                draw();
            }
            Route::Handled
        }

        Mode::Move { from } | Mode::Size { from } => {
            let moving = matches!(mode, Mode::Move { .. });
            with(|d| {
                let Some(f) = d.focus() else { return };
                // Maximised windows have nowhere to go; the geometry being
                // edited is the restored one, so drop out of maximised first
                // rather than editing a rectangle nobody can see.
                d.windows[f].state = WinState::Normal;
                // Taken before the rectangle is borrowed mutably: it is a
                // fact about the window's content, not about its geometry.
                let floor_h = min_size_of(&d.windows[f]).1;
                let r = &mut d.windows[f].rect;
                match k {
                    kbd::KEY_LEFT if moving => r.x = r.x.saturating_sub(NUDGE),
                    kbd::KEY_RIGHT if moving => {
                        r.x = (r.x + NUDGE).min(screen.x + screen.w.saturating_sub(r.w))
                    }
                    kbd::KEY_UP if moving => r.y = r.y.saturating_sub(NUDGE),
                    kbd::KEY_DOWN if moving => {
                        r.y = (r.y + NUDGE).min(screen.y + screen.h.saturating_sub(r.h))
                    }
                    kbd::KEY_LEFT => r.w = r.w.saturating_sub(NUDGE).max(160),
                    kbd::KEY_RIGHT => {
                        r.w = (r.w + NUDGE).min(screen.x + screen.w - r.x)
                    }
                    // `TITLE_H * 3` was a stand-in for "a window has to keep
                    // some body", and it moved to 90 the moment the caption
                    // grew. `min_size_of` is what it meant.
                    kbd::KEY_UP => r.h = r.h.saturating_sub(NUDGE).max(floor_h),
                    kbd::KEY_DOWN => r.h = (r.h + NUDGE).min(screen.y + screen.h - r.y),
                    b'\n' | b'\r' => d.mode = Mode::Normal,
                    27 => {
                        d.windows[f].rect = from;
                        d.mode = Mode::Normal;
                    }
                    _ => {}
                }
            });
            draw();
            Route::Handled
        }
    }
}

/// Put the keyboard back on the terminal, raising it if need be.
pub fn focus_terminal() {
    with(|d| {
        if let Some(i) = d
            .windows
            .iter()
            .position(|w| matches!(w.content, Content::Terminal(c) if c == super::console::USER))
        {
            // Deliberately does *not* un-minimise. The terminal is an
            // application: it keeps running while closed, and a shell that
            // reopens its own window every time it prints is not something the
            // user can close. Restoring it is the taskbar's job, which is the
            // one place the user actually asked for it back.
            if d.windows[i].state != WinState::Minimised {
                d.raise(i);
            }
        }
    });
    draw();
}

/// Repaint if any window overlaps the terminal.
///
/// The console owns its rectangle and paints into it whenever the system
/// prints, which is correct until a window is in front of it -- then the
/// overlapping strip becomes console output on top of a window. Rather than
/// teach the console about occlusion (it would have to be consulted per
/// character, and it is the one path that has to stay fast) the desktop
/// repairs the damage between commands, which is the only moment the shell is
/// not printing.
///
/// Cheap because it does nothing in the common case: with no overlap there is
/// nothing to repair.
pub fn redraw_over_terminal() {
    let overlapped = with(|d| {
        let Some(ti) = d
            .windows
            .iter()
            .position(|w| matches!(w.content, Content::Terminal(c) if c == super::console::USER))
        else {
            return false;
        };
        let t = d.windows[ti].rect;
        d.windows.iter().enumerate().any(|(i, w)| {
            i > ti
                && w.state != WinState::Minimised
                && w.rect.x < t.x + t.w
                && t.x < w.rect.x + w.rect.w
                && w.rect.y < t.y + t.h
                && t.y < w.rect.y + w.rect.h
        })
    })
    .unwrap_or(false);
    if overlapped {
        draw();
    }
}

pub fn take_pending() -> Option<String> {
    unsafe { (*PENDING.get()).take() }
}

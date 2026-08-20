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
    Terminal,
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
    DragSize { from: Rect },
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
}

static DESK: Racy<Option<Desktop>> = Racy::new(None);
static PENDING: Racy<Option<String>> = Racy::new(None);

const MARGIN: u32 = 8;
const NUDGE: u32 = 16;
const MENU_H: u32 = theme::TITLE_H;
/// The bar along the bottom.
const TASK_H: u32 = theme::TITLE_H + 10;
/// Gap between task buttons.
const TASK_GAP: u32 = 4;

/// The icons on the wall, top to bottom, and what opening one runs.
///
/// `term` is not a shell command: the terminal is not a panel to open but a
/// window to bring back, and the special case lives in `launch` rather than in
/// the shell so the icon works even while the shell is busy printing.
const ICONS: [(&str, &str); 9] = [
    ("Terminal", "term"),
    ("Programs", "win open programs"),
    ("Files", "win open files"),
    ("ToDo", "win open todo"),
    ("Enternet", "enternet"),
    ("Paint", "paint"),
    ("Write", "write"),
    ("Mines", "mines"),
    ("Settings", "win open settings"),
];

/// The Start menu, bottom of the bar upward -- the 98 half of the ancestry.
/// Same entries as the icons plus the one thing that belongs behind a second
/// look, exactly where 98 kept it.
const START_ITEMS: [(&str, &str); 10] = [
    ("Terminal", "term"),
    ("Programs", "win open programs"),
    ("Files", "win open files"),
    ("ToDo", "win open todo"),
    ("Enternet", "enternet"),
    ("Paint", "paint"),
    ("Write", "write"),
    ("Mines", "mines"),
    ("Settings", "win open settings"),
    ("Reboot", "reboot"),
];

/// Run what an icon or Start entry names.
fn launch(cmd: &str) {
    if cmd == "term" {
        with(|d| {
            if let Some(i) = d
                .windows
                .iter()
                .position(|w| matches!(w.content, Content::Terminal))
            {
                // The icon is the one place a minimised terminal comes back
                // from besides its task button, so restoring here is wanted,
                // not the leak `focus_terminal` guards against.
                d.windows[i].state = WinState::Normal;
                d.raise(i);
            }
        });
        return;
    }
    focus_terminal();
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
    Rect::new(bar.x + 3, bar.y + 4, theme::text_w(6) + 30, bar.h - 8)
}

/// Where the Start menu pops, directly above its button. The width formula is
/// `dropdown`'s own, so the paint and the hit-test cannot disagree.
fn start_menu_rect(fb: &Framebuffer) -> (Rect, usize) {
    let n = START_ITEMS.len();
    let w = theme::text_w(START_ITEMS.iter().map(|(l, _)| l.len()).max().unwrap_or(4)) + 24;
    let h = n as u32 * MENU_H + 8;
    let bar = taskbar_rect(fb);
    (Rect::new(bar.x + 2, bar.y.saturating_sub(h), w, h), n)
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
        let tw = theme::text_w(label.len());
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
        content: Content::Terminal,
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
        content: Content::Panel(pm),
        menus: Vec::new(),
        closable: false,
    };

    unsafe {
        *DESK.get() = Some(Desktop {
            windows: alloc::vec![pmw, terminal],
            mode: Mode::Normal,
            hover: Hover::None,
        })
    };
    // The compositor needs the heap, which exists by now; the console then
    // paints into the back buffer and pushes its own cells through, so shell
    // output stays immediate between desktop draws.
    super::compose::init();
    if let Some(back) = super::compose::target() {
        super::console::with(|c| c.retarget(back, true));
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
            .find(|win| matches!(win.content, Content::Terminal))
            .map(|win| win.rect.x + win.rect.w + MARGIN)
            .unwrap_or(screen.x);
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
            content: Content::Panel(panel),
            menus: Vec::new(),
            closable: true,
        });
    });
    // The keyboard goes back where the command came from.
    //
    // Raising and focusing are the same fact here, so a new window taking the
    // front also takes the keyboard -- and a window opened by typing at a
    // prompt would then swallow whatever was typed next. That is not a
    // theoretical worry: it is what happened, and the panel's list ran an entry
    // when the Enter at the end of the following command line reached it.
    focus_terminal();
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
    let (w, h) = (w.min(screen.w), h.min(screen.h));
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
            content: Content::App(app),
            menus: Vec::new(),
            closable: true,
        });
    });
    focus_terminal();
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
            content: Content::Browser(b),
            menus: Vec::new(),
            closable: true,
        });
    });
    // The keyboard goes back to the shell, exactly as `open` does.
    //
    // The opposite was tried first, on the reasoning that a browser opened by
    // typing `enternet` is one the user is about to drive. That is wrong here:
    // the shell *is* the console, so a window holding the keyboard means the
    // next command typed is swallowed a character at a time. It presented as
    // the browser hanging, and the trace showed it receiving 'w','i','n',' ',
    // 'k','e','y','s' one by one. Alt-Tab or the taskbar switches to it.
    focus_terminal();
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

pub fn cursor_hide(fb: &Framebuffer) {
    let Some((x, y)) = (unsafe { *SHOWN.get() }) else { return };
    let saved = unsafe { &*SAVED.get() };
    for row in 0..CUR_H {
        for col in 0..CUR_W {
            if x + col < fb.width() && y + row < fb.height() {
                fb.put(x + col, y + row, saved[(row * CUR_W + col) as usize]);
            }
        }
    }
    unsafe { *SHOWN.get() = None };
}

pub fn cursor_show(fb: &Framebuffer, x: u32, y: u32) {
    cursor_hide(fb);
    let saved = unsafe { &mut *SAVED.get() };
    for row in 0..CUR_H {
        let line = CURSOR[row as usize].as_bytes();
        for col in 0..CUR_W {
            let (px, py) = (x + col, y + row);
            if px >= fb.width() || py >= fb.height() {
                continue;
            }
            saved[(row * CUR_W + col) as usize] = fb.get(px, py);
            match line.get(col as usize) {
                Some(b'X') => fb.put(px, py, fb.raw(theme::TEXT)),
                Some(b'.') => fb.put(px, py, fb.raw(theme::HILIGHT)),
                _ => {}
            }
        }
    }
    unsafe { *SHOWN.get() = Some((x, y)) };
}

/// Read the mouse and act on it. Called from the idle loop.
pub fn poll_mouse() {
    use crate::dev::mouse;
    if !mouse::present() || !ready() {
        return;
    }
    let s = mouse::take();
    if !s.moved {
        return;
    }
    let Some(fb) = super::primary() else { return };
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
        with(|d| d.mode = Mode::Normal);
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
    cursor_show(&fb, x as u32, y as u32);
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
/// One formula, used by the paint pass and the hit-test alike, because these
/// two disagreeing is the classic pointer bug: a button that highlights in one
/// place and presses in another. `theme::window` paints the same geometry; the
/// client rectangle it returns is this one.
fn chrome(frame: Rect, has_menus: bool) -> (Rect, Option<Rect>, Rect) {
    let inner = frame.shrink(theme::FRAME);
    let title = Rect::new(inner.x, inner.y, inner.w, theme::TITLE_H);
    let mut client = Rect::new(
        inner.x,
        inner.y + theme::TITLE_H + 2,
        inner.w,
        inner.h.saturating_sub(theme::TITLE_H + 2),
    );
    let menubar = if has_menus {
        let bar = Rect::new(client.x, client.y, client.w, MENU_H);
        client = Rect::new(
            client.x,
            client.y + MENU_H,
            client.w,
            client.h.saturating_sub(MENU_H),
        );
        Some(bar)
    } else {
        None
    };
    (title, menubar, client)
}

/// Which menu-bar label sits under a point, mirroring the paint loop exactly.
fn menu_label_at(menus: &[Menu], bar: Rect, x: i32, y: i32) -> Option<usize> {
    if y < bar.y as i32 || y >= (bar.y + bar.h) as i32 {
        return None;
    }
    let mut lx = bar.x + 6;
    for (mi, m) in menus.iter().enumerate() {
        let w = theme::text_w(m.label.len()) + 12;
        if x >= lx as i32 && x < (lx + w) as i32 {
            return Some(mi);
        }
        lx += w;
    }
    None
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

    if state != WinState::Maximised && contains(size_grip(frame), x, y) {
        with(|d| d.mode = Mode::DragSize { from: frame });
        draw();
        return;
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
        Content::Terminal => ui::Step::Idle,
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
        ui::Step::Close => {
            with(|d| {
                if let Some(f) = d.focus() {
                    if d.windows[f].closable {
                        d.windows.remove(f);
                    } else {
                        d.windows[f].state = WinState::Minimised;
                    }
                }
            });
        }
        _ => {}
    }
}

/// A press on the taskbar: the Start button, an app, or a window button.
/// The focused window's own button minimises it, which is the bar's one
/// toggle and the reason it can replace stowed icons.
fn task_press(fb: &Framebuffer, x: i32, y: i32) {
    if contains(start_rect(fb), x, y) {
        with(|d| {
            d.mode = match d.mode {
                Mode::Start { .. } => Mode::Normal,
                _ => Mode::Start { item: 0 },
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
fn dropdown_rows(fb: &Framebuffer, d: &Desktop, screen: Rect) -> Option<(Rect, usize)> {
    if let Mode::Start { .. } = d.mode {
        let (r, n) = start_menu_rect(fb);
        return Some((r, n));
    }
    let f = d.focus()?;
    let frame = d.windows[f].frame(screen);
    let inner = frame.shrink(theme::FRAME);
    match d.mode {
        Mode::Menu { menu, .. } => {
            let m = d.windows[f].menus.get(menu)?;
            let mut x = inner.x + 6;
            for prev in &d.windows[f].menus[..menu] {
                x += theme::text_w(prev.label.len()) + 12;
            }
            let w = theme::text_w(m.items.iter().map(|i| i.label.len()).max().unwrap_or(4)) + 24;
            let y = inner.y + theme::TITLE_H + 2 + MENU_H;
            Some((Rect::new(x, y, w, m.items.len() as u32 * MENU_H + 8), m.items.len()))
        }
        Mode::Sys { .. } => {
            let w = theme::text_w(SYS_ITEMS.iter().map(|s| s.len()).max().unwrap_or(4)) + 24;
            let y = inner.y + theme::TITLE_H;
            Some((
                Rect::new(inner.x, y, w, SYS_ITEMS.len() as u32 * MENU_H + 8),
                SYS_ITEMS.len(),
            ))
        }
        _ => None,
    }
}

/// Which dropdown row a point is in, mirroring `dropdown`'s row layout.
fn dropdown_item_at(r: Rect, n: usize, x: i32, y: i32) -> Option<usize> {
    if !contains(r, x, y) {
        return None;
    }
    let row = (y - (r.y + 4) as i32) / MENU_H as i32;
    if row >= 0 && (row as usize) < n {
        Some(row as usize)
    } else {
        None
    }
}

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
        let Some((r, n)) = dropdown_rows(fb, d, screen) else {
            d.mode = Mode::Normal;
            return;
        };
        let Some(item) = dropdown_item_at(r, n, x, y) else {
            d.mode = Mode::Normal;
            return;
        };
        match d.mode {
            Mode::Start { .. } => {
                d.mode = Mode::Normal;
                run = Some(String::from(START_ITEMS[item].1));
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
            Mode::DragSize { .. } => Rect::new(
                r.x,
                r.y,
                ((x - r.x as i32).max(160) as u32).min(screen.x + screen.w - r.x),
                ((y - r.y as i32).max((theme::TITLE_H * 3) as i32) as u32)
                    .min(screen.y + screen.h - r.y),
            ),
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
        let Some((r, n)) = dropdown_rows(&fb, d, screen) else {
            return None;
        };
        let Some(item) = dropdown_item_at(r, n, x, y) else {
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
    fb.rect(0, 0, fb.width(), fb.height(), theme::DESKTOP);
    let step = 32;
    let mut y = 0;
    while y < fb.height() {
        let mut x = 0;
        while x < fb.width() {
            fb.rect(x, y, 1, 1, theme::DESKTOP_GRID);
            x += step;
        }
        y += step;
    }

    let r = (fb.height() / 5).min(fb.width() / 5) as i32;
    super::splash::aperture(
        fb,
        (fb.width() / 2) as i32,
        (fb.height() / 2) as i32,
        r,
        theme::WALL_MARK,
        theme::DESKTOP,
    );
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
        // Stop before the clock rather than drawing under it.
        if x + w > clock_rect(fb).x {
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
    theme::panel(fb, bar);

    // The Start button: the mark and the name. Held down while its menu is
    // open, which is the one place the bar states a mode rather than a focus.
    let s = start_rect(fb);
    let start_open = matches!(d.mode, Mode::Start { .. });
    theme::button(fb, s, "GLaDOS", d.hover == Hover::Start, start_open);
    theme::aperture_dot(fb, s.x + 11, s.y + s.h / 2, (s.h / 2) as i32 - 4);
    theme::separator_v(fb, s.x + s.w + 3, s.y, s.h);

    for (i, (r, icon, pressed)) in task_layout(fb, d).into_iter().enumerate() {
        // Keyboard selection and pointer hover draw the same way: both are "the
        // next click or Enter lands here", and two different highlights would
        // claim two different things.
        let hot = sel == Some(i) || d.hover == Hover::Task(i);
        theme::button(fb, r, "", hot, pressed);
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

    let c = clock_rect(fb);
    theme::well(fb, c, theme::FACE);
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
            let mut client = theme::window(
                &fb,
                frame,
                &win.title,
                active,
                win.state == WinState::Maximised,
                hot,
            );
            if client.is_empty() {
                continue;
            }

            if !win.menus.is_empty() {
                let bar = Rect::new(client.x, client.y, client.w, MENU_H);
                theme::panel(&fb, bar);
                let open = match d.mode {
                    Mode::Menu { menu, .. } if active => Some(menu),
                    _ => None,
                };
                let mut x = bar.x + 6;
                for (mi, m) in win.menus.iter().enumerate() {
                    let w = theme::text_w(m.label.len()) + 12;
                    let hot = Some(mi) == open
                        || d.hover == Hover::MenuLabel { win: i, menu: mi };
                    let (fg, bg) = if hot {
                        (theme::SELECT_TEXT, theme::SELECT)
                    } else {
                        (theme::TEXT, theme::FACE)
                    };
                    fb.rect(x, bar.y + 2, w, bar.h - 4, bg);
                    theme::text(&fb, x + 6, bar.y + 5, &m.label, fg, bg);
                    x += w;
                }
                client = Rect::new(
                    client.x,
                    client.y + MENU_H,
                    client.w,
                    client.h.saturating_sub(MENU_H),
                );
            }

            match &win.content {
                Content::Terminal => {
                    let well = client.shrink(2);
                    theme::well(&fb, well, theme::SCREEN);
                    let grid = well.shrink(3);
                    super::console::with(|c| {
                        c.set_visible(true);
                        c.reflow(grid.x, grid.y, grid.w, grid.h);
                    });
                    super::console::redraw();
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
        }

        // Popups last, over everything, because that is what a popup is.
        if let Some(f) = focus {
            let frame = d.windows[f].frame(screen);
            match d.mode {
                Mode::Menu { menu, item } => {
                    if let Some(m) = d.windows[f].menus.get(menu) {
                        let inner = frame.shrink(theme::FRAME);
                        let mut x = inner.x + 6;
                        for prev in &d.windows[f].menus[..menu] {
                            x += theme::text_w(prev.label.len()) + 12;
                        }
                        dropdown(
                            &fb,
                            x,
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
                    let w = theme::text_w(msg.len());
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
            let (r, _) = start_menu_rect(&fb);
            dropdown(&fb, r.x, r.y, START_ITEMS.iter().map(|(l, _)| *l), item);
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
    let w = theme::text_w(items.iter().map(|s| s.len()).max().unwrap_or(4)) + 24;
    let h = items.len() as u32 * MENU_H + 8;
    let r = Rect::new(x, y, w.min(fb.width().saturating_sub(x)), h);
    theme::panel(fb, r);
    // A menu opened near the right edge can be clipped to nothing, and `r.w - 8`
    // on a narrow one wraps to four billion -- which is not a crash but a fill
    // loop long enough to look like a hang.
    if r.w < 16 {
        return;
    }
    for (i, label) in items.iter().enumerate() {
        let row = Rect::new(r.x + 4, r.y + 4 + i as u32 * MENU_H, r.w - 8, MENU_H);
        theme::list_row(fb, row, label, i == sel, true);
    }
}

pub fn focus_is_terminal() -> bool {
    with(|d| match d.focus() {
        Some(f) => matches!(d.windows[f].content, Content::Terminal),
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
                Some(ui::Step::Close) => cycle(false),
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
            let n = START_ITEMS.len();
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
                    with(|d| d.mode = Mode::Normal);
                    launch(START_ITEMS[item].1);
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

        // A pointer drag owns the screen until the button lifts; the keyboard
        // can only abandon it. Anything else mid-drag is swallowed, because a
        // window that responds to typing while it is being dragged is two
        // interfaces fighting over one object.
        Mode::Drag { from, .. } | Mode::DragSize { from } => {
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
                    kbd::KEY_UP => r.h = r.h.saturating_sub(NUDGE).max(theme::TITLE_H * 3),
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
            .position(|w| matches!(w.content, Content::Terminal))
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
            .position(|w| matches!(w.content, Content::Terminal))
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

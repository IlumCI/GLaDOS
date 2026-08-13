//! The desktop and its window manager.
//!
//! The terminal is a window on this, not the other way round. That distinction
//! is the whole design: a shell that draws a dialog over itself is a program
//! with a pop-up, while a desktop that hosts a terminal alongside other windows
//! is an environment. The console already knows how to live inside a rectangle
//! (`console::reflow`), so the terminal needs no special case -- it is a window
//! whose content happens to be the character grid.
//!
//! ### Repaint is always total, and that is the design
//!
//! Windows overlap, and the usual price for that is damage tracking: work out
//! which rectangles a change dirtied and repaint only those. Not here. Every
//! change repaints the wall and then every window back to front.
//!
//! It is affordable because of *what causes* a change. Nothing here animates --
//! a repaint happens when a key is pressed, which is at most a few times a
//! second, and 1280x800 is a million stores into a write-back-mapped aperture.
//! Damage tracking would buy nothing measurable and cost the one thing this
//! code cannot afford to lose, which is being obviously correct: the entire
//! class of "stale pixels from a window that used to be there" cannot occur if
//! there is no such thing as a partial repaint.
//!
//! Back-to-front is also what makes the terminal work with no special case.
//! `console::redraw` paints the whole grid; a window in front of it is simply
//! drawn afterwards.
//!
//! ### What is deliberately absent
//!
//! No mouse. Every operation is a keystroke, because there is no pointer and
//! there is not going to be one soon -- and a window you can only move by
//! dragging is a window that cannot be moved.

use super::theme::{self, Rect};
use super::ui::{self, Action, Panel};
use super::Framebuffer;
use crate::dev::kbd;
use crate::sync::Racy;
use alloc::string::String;
use alloc::vec::Vec;

/// What is inside a window.
pub enum Content {
    /// The character grid. Exactly one window holds it -- there is one console
    /// -- and that is true by construction rather than checked.
    Terminal,
    Panel(Panel),
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

pub struct Window {
    pub title: String,
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
}

pub struct Desktop {
    /// Back to front. The last visible window is the focused one, so z-order
    /// and focus are the same fact stored once -- two fields would be two
    /// things to keep in agreement, and they would drift.
    pub windows: Vec<Window>,
    pub mode: Mode,
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

/// Apps the app bar can launch. Name, and the panel `ui::panel_named` builds.
const APPS: [(&str, &str); 3] =
    [("Programs", "programs"), ("Files", "files"), ("Settings", "settings")];

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
    let term_w = screen.w.saturating_sub(pm_w + MARGIN);

    let terminal = Window {
        title: String::from("GLaDOS Terminal"),
        rect: Rect::new(screen.x, screen.y, term_w, screen.h),
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

    let pmw = Window {
        title: String::from("Program Manager"),
        rect: Rect::new(screen.x + term_w + MARGIN, screen.y, pm_w, pm_h.min(screen.h)),
        state: WinState::Normal,
        content: Content::Panel(pm),
        menus: Vec::new(),
        closable: false,
    };

    unsafe {
        *DESK.get() = Some(Desktop {
            windows: alloc::vec![pmw, terminal],
            mode: Mode::Normal,
        })
    };
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

/// Buttons on the bar, left to right: the apps, then one per window.
///
/// One flat list so `item` in `Mode::Taskbar` can index it directly, and so
/// that adding an app or opening a window cannot put the two halves out of
/// step with each other.
fn task_slots(d: &Desktop) -> Vec<(String, bool)> {
    let mut out: Vec<(String, bool)> = APPS
        .iter()
        .map(|(label, _)| (String::from(*label), false))
        .collect();
    let focus = d.focus();
    for (i, w) in d.windows.iter().enumerate() {
        out.push((w.title.clone(), Some(i) == focus));
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

    let slots = task_slots(d);
    let n_apps = APPS.len();
    let btn_h = bar.h - 8;
    let y = bar.y + 4;
    let mut x = bar.x + 4;

    for (i, (label, pressed)) in slots.iter().enumerate() {
        let is_app = i < n_apps;
        // Titles are capped rather than allowed to set the width. "GLaDOS
        // Terminal" at full length is a quarter of the bar on its own.
        let shown = label.len().min(12);
        let w = theme::text_w(shown) + if is_app { 24 } else { 16 };
        // Stop before the clock rather than drawing under it.
        if x + w > clock_rect(fb).x {
            break;
        }
        let r = Rect::new(x, y, w, btn_h);
        let focused = sel == Some(i);
        // A task button is sunken while its window has the keyboard, which is
        // the same claim about light the bevels make everywhere else.
        theme::button(fb, r, &label[..shown], focused, *pressed);
        if is_app {
            // The apps carry the mark, so the launcher half of the bar reads
            // as different in kind from the window half without a caption
            // saying so.
            theme::aperture_dot(fb, r.x + 8, r.y + r.h / 2, (btn_h / 2) as i32 - 4);
        }
        x += w + TASK_GAP;
        if i == n_apps - 1 {
            theme::separator_v(fb, x + 2, y, btn_h);
            x += 8;
        }
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
    let Some(fb) = super::primary() else {
        return;
    };
    let screen = screen_rect(&fb);
    with(|d| {
        wallpaper(&fb);
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
            let mut client = theme::window(&fb, frame, &win.title, active);
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
                    let hot = Some(mi) == open;
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
                    super::console::with(|c| c.reflow(grid.x, grid.y, grid.w, grid.h));
                    super::console::redraw();
                }
                Content::Panel(p) => {
                    theme::panel(&fb, client);
                    p.draw_in(&fb, client, active);
                }
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
                Mode::Normal | Mode::Taskbar { .. } => {}
            }
        }
    });
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
        None => false,
    })
    .unwrap_or(true)
}

/// Cycle focus. Sends the top window to the back, which brings the next one up.
pub fn cycle(_back: bool) {
    with(|d| {
        if d.windows.len() < 2 {
            return;
        }
        let w = d.windows.pop().unwrap();
        d.windows.insert(0, w);
        // Skip minimised windows, or Alt-Tab appears to do nothing.
        let mut guard = 0;
        while d.focus().is_none() && guard < 8 {
            let w = d.windows.pop().unwrap();
            d.windows.insert(0, w);
            guard += 1;
        }
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
                        with(|d| {
                            if let Some(f) = d.focus() {
                                d.windows[f].title = title;
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
            let n_apps = APPS.len();
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
                    if item < n_apps {
                        // Launching goes through the shell like everything
                        // else, so there is one path to a running command.
                        let cmd = alloc::format!("win open {}", APPS[item].1);
                        focus_terminal();
                        unsafe { *PENDING.get() = Some(cmd) };
                    } else {
                        let w = item - n_apps;
                        with(|d| {
                            if w < d.windows.len() {
                                // Clicking a task button restores a minimised
                                // window, which is the only way back for one.
                                if d.windows[w].state == WinState::Minimised {
                                    d.windows[w].state = WinState::Normal;
                                }
                                d.raise(w);
                            }
                        });
                        draw();
                    }
                }
                27 => {
                    with(|d| d.mode = Mode::Normal);
                    draw();
                }
                _ => {}
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
            d.windows[i].state = match d.windows[i].state {
                WinState::Minimised => WinState::Normal,
                s => s,
            };
            d.raise(i);
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

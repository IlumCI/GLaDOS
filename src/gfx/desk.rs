//! The desktop: a background, a set of windows, and which one has the keyboard.
//!
//! The terminal is a window on this, not the other way round. That distinction
//! is the whole design: a shell that draws a dialog over itself is a program
//! with a pop-up, while a desktop that hosts a terminal alongside other windows
//! is an environment. The console already knows how to live inside a rectangle
//! (`console::reflow`), so the terminal needs no special case -- it is a window
//! whose content happens to be the character grid.
//!
//! ### What this deliberately is not
//!
//! No mouse, no dragging, no overlapping-window z-order management, no
//! minimise. Windows are tiled at fixed positions and Alt-Tab moves focus.
//! Overlap is the expensive part -- it needs damage tracking and back-to-front
//! repaint, and every one of those costs is paid to support a pointer that does
//! not exist. Tiling is what a keyboard-only desktop actually wants.

use super::theme::{self, Rect};
use super::ui::{self, Panel};
use super::Framebuffer;
use crate::sync::Racy;
use alloc::string::String;
use alloc::vec::Vec;

/// What is inside a window.
pub enum Content {
    /// The character grid. Exactly one window may hold it -- there is one
    /// console -- and that is enforced by construction rather than checked.
    Terminal,
    Panel(Panel),
}

pub struct Window {
    pub title: String,
    pub rect: Rect,
    pub content: Content,
}

pub struct Desktop {
    pub windows: Vec<Window>,
    pub focus: usize,
}

static DESK: Racy<Option<Desktop>> = Racy::new(None);

pub fn ready() -> bool {
    unsafe { (*DESK.get()).is_some() }
}

pub fn with<R>(f: impl FnOnce(&mut Desktop) -> R) -> Option<R> {
    unsafe { (*DESK.get()).as_mut().map(f) }
}

/// Lay the desktop out for this panel and paint it.
///
/// Tiled side by side rather than stacked: the terminal wants height above all
/// -- it is a scrolling log -- and the launcher wants only as much as its
/// contents need, so a vertical split wastes the dimension each one cares
/// about.
pub fn init() {
    let Some(fb) = super::primary() else {
        return;
    };
    let (w, h) = (fb.width(), fb.height());
    let margin = 8u32;
    let mut pm = ui::program_manager();
    let (pm_w, pm_h) = pm.preferred();
    pm.set_title("Program Manager");

    // Only as wide as it asked for. Clamping to half the screen sounds safe
    // and is not: it silently truncated the hint text and took the width out of
    // the terminal, which is the window that actually needs columns.
    let pm_w = pm_w.min(w / 2).min(w - margin * 3);
    let term_w = w.saturating_sub(pm_w + margin * 3);

    let windows = alloc::vec![
        Window {
            title: String::from("GLaDOS Terminal"),
            rect: Rect::new(margin, margin, term_w, h - margin * 2),
            content: Content::Terminal,
        },
        Window {
            title: String::from("Program Manager"),
            rect: Rect::new(
                margin * 2 + term_w,
                margin,
                pm_w,
                pm_h.min(h - margin * 2),
            ),
            content: Content::Panel(pm),
        },
    ];

    unsafe { *DESK.get() = Some(Desktop { windows, focus: 0 }) };
    draw();
}

/// The background.
///
/// A flat fill and a sparse grid rather than a gradient or a bitmap: the wall
/// is repainted on every full redraw, and a per-pixel background on a 1920x1080
/// framebuffer over an uncached-until-M4 aperture is the one thing here that
/// could actually be slow.
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
}

pub fn draw() {
    let Some(fb) = super::primary() else {
        return;
    };
    with(|d| {
        wallpaper(&fb);
        for (i, win) in d.windows.iter().enumerate() {
            let active = i == d.focus;
            let client = theme::window(&fb, win.rect, &win.title, active);
            if client.is_empty() {
                continue;
            }
            match &win.content {
                Content::Terminal => {
                    // The console is a shadow grid, so moving it is a reflow
                    // and a repaint; nothing written before the move is lost.
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
    });
}

/// The terminal window's title bar, for the uptime readout.
///
/// The clock used to draw at a fixed screen corner, which was right while that
/// corner was bare framebuffer. On a desktop that corner belongs to whichever
/// window is there, so the clock has to ask where it may write rather than
/// assume -- otherwise it punches a hole through the Program Manager once a
/// tenth of a second.
pub fn terminal_status_area() -> Option<(Rect, bool)> {
    with(|d| {
        d.windows
            .iter()
            .enumerate()
            .find(|(_, w)| matches!(w.content, Content::Terminal))
            .map(|(i, w)| {
                // The part of the bar the title has not already used. The
                // clock drew from the right edge and met the title coming the
                // other way on a narrow window; a status area that starts after
                // the title cannot.
                let inner = w.rect.shrink(theme::FRAME);
                let used = 28 + theme::text_w(w.title.len()) + 8;
                (
                    Rect::new(
                        inner.x + used,
                        inner.y,
                        inner.w.saturating_sub(used),
                        theme::TITLE_H,
                    ),
                    i == d.focus,
                )
            })
    })
    .flatten()
}

/// Where the keyboard is pointing.
pub fn focus_is_terminal() -> bool {
    with(|d| matches!(d.windows.get(d.focus).map(|w| &w.content), Some(Content::Terminal)))
        .unwrap_or(true)
}

pub fn cycle(back: bool) {
    with(|d| {
        let n = d.windows.len();
        if n == 0 {
            return;
        }
        d.focus = if back { (d.focus + n - 1) % n } else { (d.focus + 1) % n };
        crate::serial_println!("[desk] focus {} \"{}\"", d.focus, d.windows[d.focus].title);
    });
    draw();
}

/// One line per window on the serial port, focused one marked. The oracle a
/// headless run reads instead of looking at the screen.
pub fn trace(event: &str) {
    with(|d| {
        crate::serial_println!("[desk] {}", event);
        for (i, w) in d.windows.iter().enumerate() {
            crate::serial_println!(
                "[desk] {} {} {}x{}+{}+{}",
                if i == d.focus { '>' } else { ' ' },
                w.title,
                w.rect.w,
                w.rect.h,
                w.rect.x,
                w.rect.y
            );
        }
    });
}

/// What the shell should do with a key.
pub enum Route {
    /// The desktop consumed it.
    Handled,
    /// Give it to the shell's line editor.
    Shell(u8),
}

/// Route one keystroke.
///
/// Alt-Tab is taken before anything else and Tab is deliberately not, because
/// the terminal has to keep receiving Tab. A window switcher that stole it
/// would make the shell unusable in exchange for saving one modifier.
pub fn key(k: u8) -> Route {
    if !ready() {
        return Route::Shell(k);
    }
    if k == crate::dev::kbd::KEY_ALTTAB {
        cycle(false);
        return Route::Handled;
    }
    if focus_is_terminal() {
        return Route::Shell(k);
    }
    // A panel has focus. Feed it, and act on whatever it decides.
    let step = with(|d| match d.windows.get_mut(d.focus).map(|w| &mut w.content) {
        Some(Content::Panel(p)) => p.key(k),
        _ => ui::Step::Idle,
    });
    match step {
        Some(ui::Step::Redraw) => {
            draw();
            Route::Handled
        }
        // Esc from a panel hands the keyboard back rather than closing it:
        // these windows are the desktop's furniture, not dialogs, and a
        // Program Manager you can dismiss with no way to get it back would be
        // a desktop with a missing shell.
        Some(ui::Step::Close) => {
            cycle(false);
            Route::Handled
        }
        Some(ui::Step::Do(ui::Action::Run(cmd))) => {
            // Hand focus back to the terminal, so the output lands in a window
            // that is looking at it, and queue the command rather than running
            // it. The shell owns execution and the borrows it needs; the
            // desktop decides only *that* something should run.
            with(|d| d.focus = 0);
            draw();
            unsafe { *PENDING.get() = Some(cmd) };
            Route::Handled
        }
        _ => Route::Handled,
    }
}

static PENDING: Racy<Option<String>> = Racy::new(None);

/// A command a panel asked for, if any. The shell polls this.
pub fn take_pending() -> Option<String> {
    unsafe { (*PENDING.get()).take() }
}

//! A desktop window whose contents are an Aiksi program.
//!
//! The program is written the way a small game or tool is written anywhere:
//!
//! ```text
//! count = 0
//!
//! def draw():
//!     rect(0, 0, width(), height(), "#1e2430")
//!     text(20, 20, f"Clicked {count} times", "white", 28)
//!
//! def click(x, y):
//!     count += 1
//! ```
//!
//! `draw()` paints the window and is called whenever something changed;
//! `click(x, y)` and `key(k)` are called on input, in the window's own
//! coordinates; `tick()`, if there is one, ten times a second. All four are
//! optional except `draw`. While `draw` runs, `rect`, `text`, `circle` and
//! `pixel` paint this window and nothing else -- which is also why a
//! sandboxed program may call them here, where full-screen drawing needs
//! `app trust`.
//!
//! **An error is shown in the window.** A program a participant or a model
//! has just written fails often, and the place they are looking when it does
//! is the window; an error that went to a terminal behind it would read as
//! a window that silently stopped working.

use super::text::Face;
use super::theme::{self, Rect};
use super::{Color, DeskApp, Framebuffer};
use crate::aiksi::{Interp, Value};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};

/// Steps one event may take: a draw of a busy window is a few thousand, and
/// a runaway loop has to stop before the screen does.
const EVENT_BUDGET: u64 = 2_000_000;
const TICK_MS: u32 = 100;

pub struct ProgApp {
    interp: RefCell<Interp>,
    error: RefCell<Option<String>>,
    /// Moves on every event, so the desktop redraws exactly when the program
    /// may have changed what it shows.
    events: Cell<u64>,
    last_tick: Cell<u64>,
    has_tick: bool,
    has_key: bool,
    has_click: bool,
}

impl ProgApp {
    /// Load and start a program: its top level runs once, setting up the
    /// state its handlers then share.
    pub fn new(src: &str, jail: &str) -> ProgApp {
        let mut it = Interp::sandboxed(jail).with_step_budget(EVENT_BUDGET);
        let mut error = None;
        match crate::aiksi::parse_source(src) {
            Ok(prog) => {
                crate::gfx::console::begin_capture();
                if let Err(e) = it.run(&prog) {
                    error = Some(explain(&e, src));
                }
                let _ = crate::gfx::console::end_capture();
                if error.is_none() && !it.has_fn("draw") {
                    error = Some(if src.lines().any(|l| l.starts_with(' ') && l.trim_start().starts_with("def draw")) {
                        String::from("draw() is defined inside another function, so the window cannot find it. Define draw(), click(x, y), key(k) and tick() at the top level of the file, with no indentation before 'def'.")
                    } else {
                        String::from("the program has no draw() function, so there is nothing to show. Define draw() at the top level of the file.")
                    });
                }
            }
            Err(e) => error = Some(e),
        }
        ProgApp {
            has_tick: it.has_fn("tick"),
            has_key: it.has_fn("key"),
            has_click: it.has_fn("click"),
            interp: RefCell::new(it),
            error: RefCell::new(error),
            events: Cell::new(0),
            last_tick: Cell::new(crate::dev::lapic::ticks()),
        }
    }

    /// Call a handler, keeping its error for the window to show.
    fn call(&self, name: &str, args: &[Value], canvas: Option<(Framebuffer, u32, u32)>) {
        if self.error.borrow().is_some() {
            return;
        }
        let mut it = self.interp.borrow_mut();
        it.reset_steps();
        it.set_canvas(canvas);
        crate::gfx::console::begin_capture();
        let r = it.invoke(name, args);
        let _ = crate::gfx::console::end_capture();
        it.set_canvas(None);
        if let Err(e) = r {
            *self.error.borrow_mut() = Some(format!("in {}(): {}", name, e));
        }
        self.events.set(self.events.get().wrapping_add(1));
    }

    /// The error, if the program has stopped.
    pub fn error(&self) -> Option<String> {
        self.error.borrow().clone()
    }

    /// Run the program's handlers once each against an off-screen surface, the
    /// way a window would, and answer what went wrong and whether anything
    /// was drawn. `maker` judges a windowed program with exactly this.
    pub fn exercise(&self, w: u32, h: u32) -> Result<(), String> {
        let mut buf: Vec<u32> = alloc::vec![0; (w * h) as usize];
        // Safety: `buf` outlives `fb`, which is dropped at the end of this
        // function, and is exactly `w * h` words.
        let fb = unsafe { Framebuffer::over_ram(buf.as_mut_ptr() as u64, w, h, w, super::Format::Bgrx) };
        self.call("draw", &[], Some((fb, w, h)));
        if let Some(e) = self.error() {
            return Err(e);
        }
        let first = buf[(h / 2 * w + w / 2) as usize];
        let drew = buf.iter().any(|p| *p != 0) && buf.iter().any(|p| *p != first);
        if !drew {
            return Err(String::from("draw() ran but the window is a single colour: nothing was drawn on it"));
        }
        if self.has_click {
            self.call("click", &[Value::Int((w / 2) as i64), Value::Int((h / 2) as i64)], None);
            self.call("draw", &[], Some((fb, w, h)));
        }
        if self.has_key {
            for k in ["a", "up", "enter"] {
                self.call("key", &[Value::Str(String::from(k))], None);
            }
            self.call("draw", &[], Some((fb, w, h)));
        }
        if self.has_tick {
            for _ in 0..3 {
                self.call("tick", &[], None);
            }
            self.call("draw", &[], Some((fb, w, h)));
        }
        drop(buf);
        match self.error() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    pub fn handlers(&self) -> Vec<&'static str> {
        let mut v = alloc::vec!["draw"];
        if self.has_click {
            v.push("click");
        }
        if self.has_key {
            v.push("key");
        }
        if self.has_tick {
            v.push("tick");
        }
        v
    }
}

/// What went wrong while a program was starting, said in terms of how a
/// window works. The first model-written apps were the shape every GUI
/// tutorial has -- everything in one function, driven by `while True:` calling
/// `draw()` -- and the interpreter's honest answer ("rect draws outside any
/// window") told the model nothing it could act on.
pub fn explain(e: &str, src: &str) -> String {
    let looped = src.lines().any(|l| l.trim_start().starts_with("while True") || l.trim_start().starts_with("while 1"));
    if e.contains("draws outside any window") {
        return String::from(
            "drawing ran while the program was starting, outside the window. The window calls draw() for you: \
             do not call draw() yourself and do not write a loop. Define draw(), click(x, y), key(k) and tick() \
             at the top level of the file, and nothing else should run.",
        );
    }
    if e.contains("budget exceeded") && looped {
        return String::from(
            "the program never finished starting: a 'while True' loop runs forever. The window calls draw() and \
             tick() for you, so remove the loop and put each step's work in tick().",
        );
    }
    String::from(e)
}

/// A key as a program receives it: the character, or a name for the keys
/// that have none.
fn key_name(k: u8) -> Option<String> {
    use crate::dev::kbd::{KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP};
    Some(String::from(match k {
        KEY_UP => "up",
        KEY_DOWN => "down",
        KEY_LEFT => "left",
        KEY_RIGHT => "right",
        b'\n' | b'\r' => "enter",
        b' ' => "space",
        8 | 0x7F => "backspace",
        b'\t' => "tab",
        0x1B => "escape",
        c if (0x21..0x7F).contains(&c) => return Some(String::from(c as char)),
        _ => return None,
    }))
}

impl DeskApp for ProgApp {
    fn draw_in(&self, fb: &Framebuffer, client: Rect, _focused: bool) {
        fb.rect(client.x, client.y, client.w, client.h, Color::new(0xFF, 0xFF, 0xFF));
        self.call("draw", &[], Some((*fb, client.w, client.h)));
        if let Some(e) = self.error() {
            // On a pale red card, wrapped, the way a person reads an error.
            fb.rect(client.x, client.y, client.w, client.h, Color::new(0xFD, 0xEE, 0xEE));
            let px = theme::ui_size();
            let mut y = theme::px(16) as i32;
            super::text::draw(fb, theme::px(16) as f32, y + (px * 1.05) as i32, Face::UiStrong, px, "The program stopped", Color::new(0xA0, 0x20, 0x20));
            y += (px * 1.8) as i32;
            let max = client.w.saturating_sub(theme::px(32)) as f32;
            let mut line = String::new();
            let mut lines = Vec::new();
            for w in e.split_whitespace() {
                let trial = if line.is_empty() { String::from(w) } else { format!("{} {}", line, w) };
                if super::text::measure(Face::Ui, px, &trial) > max && !line.is_empty() {
                    lines.push(core::mem::replace(&mut line, String::from(w)));
                } else {
                    line = trial;
                }
            }
            lines.push(line);
            for l in lines {
                super::text::draw(fb, theme::px(16) as f32, y + (px * 1.05) as i32, Face::Ui, px, &l, theme::TEXT);
                y += (px * 1.5) as i32;
            }
        }
    }

    fn key(&mut self, k: u8) -> bool {
        if !self.has_key {
            return false;
        }
        let Some(name) = key_name(k) else { return false };
        self.call("key", &[Value::Str(name)], None);
        true
    }

    fn press(&mut self, _client: Rect, x: i32, y: i32) -> bool {
        if !self.has_click {
            return false;
        }
        self.call("click", &[Value::Int(x as i64), Value::Int(y as i64)], None);
        true
    }

    fn stamp(&self) -> Option<u64> {
        // `tick` is driven from here, the one call the desktop makes on a
        // timer: due, it runs, and the moved counter is the redraw.
        if self.has_tick && self.error.borrow().is_none() {
            let now = crate::dev::lapic::ticks();
            if now.saturating_sub(self.last_tick.get()) * 1000 >= TICK_MS as u64 * crate::TIMER_HZ as u64 {
                self.last_tick.set(now);
                self.call("tick", &[], None);
            }
        }
        Some(super::scene::Stamp::of((self.events.get(), self.error.borrow().is_some())))
    }

    fn wake_ms(&self) -> Option<u32> {
        if self.has_tick {
            Some(TICK_MS)
        } else {
            None
        }
    }

    fn min_size(&self) -> (u32, u32) {
        (theme::px(240), theme::px(160))
    }
}

/// Open a program in a window of its own.
pub fn open(title: &str, src: &str, jail: &str) {
    let app = ProgApp::new(src, jail);
    super::desk::open_app(title, super::desk::ICO_PROGRAMS, alloc::boxed::Box::new(app), theme::px(640), theme::px(480));
}

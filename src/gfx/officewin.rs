//! The Office window: say what to write, and the machine writes it into
//! LibreOffice.
//!
//! A front for `office`, and only a front. Every button queues an `office`
//! command through `desk::queue_command`, so the work happens on the shell's
//! task exactly as if it had been typed -- the rule `mindwin` states for its
//! windows and for the same reason: a press handler runs on the desktop's
//! task, and calling into the model from there would claim the engine out
//! from under whatever already holds it.
//!
//! What it shows is what a person needs to trust it: whether Writer is open,
//! whether the model is ready, what the typist still has to type, and the
//! last few things `office` did, in its own words.

use super::theme::{self, Rect};
use super::{Color, DeskApp, Framebuffer};
use alloc::format;
use alloc::string::String;
use core::cell::Cell;

/// Things to try, one click from the field. Short enough to read at a glance
/// and each a different kind of document.
const EXAMPLES: [&str; 5] = [
    "A short cover letter for a junior graphic designer job at Lumen Studio",
    "A monthly household budget: rent, groceries, transport, utilities, phone, savings, with yearly cost as =B*12",
    "Minutes of today's team meeting: budget approved, launch moved to May, Sam owns the website",
    "Sales by quarter for four products, with each product's share of the year",
    "/tmp/office/report.docx",
];

/// What each button queues. The field's text is appended to the first three.
const BUTTONS: [(&str, &str); 5] = [
    ("Write it", "office write"),
    ("Spreadsheet", "office calc"),
    ("Summarize file", "office summarize"),
    ("Open Writer", "office start"),
    ("Stop", "office stop"),
];

pub struct Office {
    typed: String,
    btns: [Cell<Rect>; 5],
    examples: [Cell<Rect>; 5],
}

impl Office {
    pub fn new() -> Office {
        let z = || Cell::new(Rect::new(0, 0, 0, 0));
        Office { typed: String::new(), btns: [z(), z(), z(), z(), z()], examples: [z(), z(), z(), z(), z()] }
    }

    pub fn preferred() -> (u32, u32) {
        (560, 420)
    }

    fn fire(&mut self, i: usize) -> bool {
        let (_, cmd) = BUTTONS[i];
        let line = match i {
            0..=2 => {
                let what = self.typed.trim();
                if what.is_empty() {
                    return true;
                }
                if i == 2 {
                    format!("{} {} --type", cmd, what)
                } else {
                    format!("{} {}", cmd, what)
                }
            }
            _ => String::from(cmd),
        };
        super::desk::queue_command(&line);
        if i <= 2 {
            self.typed.clear();
        }
        true
    }
}

fn dense() -> u32 {
    super::mindwin::dense()
}

fn row_h() -> u32 {
    theme::text_h_at(dense()) + 4
}

fn col_w(n: usize) -> u32 {
    theme::text_w_at(n, dense())
}

fn dtext(fb: &Framebuffer, x: u32, y: u32, s: &str, fg: Color) {
    theme::text_over_at(fb, x, y, s, fg, dense());
}

fn inside(r: Rect, x: i32, y: i32) -> bool {
    x >= r.x as i32 && x < (r.x + r.w) as i32 && y >= r.y as i32 && y < (r.y + r.h) as i32
}

/// Status, field, buttons, examples, activity -- top to bottom.
fn layout(client: Rect) -> (Rect, Rect, Rect, Rect, Rect) {
    let pad = client.shrink(8);
    let rh = row_h();
    let status = Rect::new(pad.x, pad.y, pad.w, rh * 3 + 8);
    let field = Rect::new(pad.x, status.y + status.h + 8, pad.w, rh + 8);
    let buttons = Rect::new(pad.x, field.y + field.h + 6, pad.w, rh + 8);
    let examples = Rect::new(pad.x, buttons.y + buttons.h + 10, pad.w, rh * (EXAMPLES.len() as u32 + 1) + 8);
    let top = examples.y + examples.h + 8;
    let activity = Rect::new(pad.x, top, pad.w, (pad.y + pad.h).saturating_sub(top));
    (status, field, buttons, examples, activity)
}

impl DeskApp for Office {
    fn stamp(&self) -> Option<u64> {
        Some(super::scene::Stamp::of((
            self.typed.as_str(),
            crate::office::log_snapshot(),
            crate::dev::typist::pending(),
            crate::gfx::desk::window_titled("LibreOffice").is_some(),
            crate::ai::engine_ready(),
            crate::office::drafting(),
        )))
    }

    /// The typist and the model move on other tasks; looked at a few times a
    /// second so the queue drains visibly.
    fn wake_ms(&self) -> Option<u32> {
        Some(300)
    }

    fn draw_in(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        theme::panel(fb, client);
        let (status, field, buttons, examples, activity) = layout(client);
        let rh = row_h();

        // What is ready, said plainly.
        theme::well(fb, status, theme::HILIGHT);
        let s = status.shrink(4);
        let writer = crate::gfx::desk::window_titled("LibreOffice");
        let ready = crate::ai::engine_ready();
        let busy = crate::office::drafting();
        let waiting = crate::dev::typist::pending();
        let lines = [
            (
                "Writer",
                match &writer {
                    Some(t) => format!("open -- {}", t),
                    None => String::from("not open; 'Open Writer' starts it"),
                },
                writer.is_some(),
            ),
            (
                "Model",
                String::from(match (ready, busy) {
                    (_, true) => "writing -- typed into the document as it goes",
                    (true, false) => "ready, running in the kernel",
                    (false, false) => "no model loaded",
                }),
                ready,
            ),
            (
                "Typing",
                if waiting > 0 { format!("{} characters still to type", waiting) } else { String::from("idle") },
                true,
            ),
        ];
        let gutter = col_w(9);
        let room = ((s.w.saturating_sub(gutter)) / col_w(1)) as usize;
        for (i, (k, v, good)) in lines.iter().enumerate() {
            let y = s.y + i as u32 * rh;
            dtext(fb, s.x, y, k, theme::TEXT_DIM);
            dtext(fb, s.x + gutter, y, theme::head_chars(v, room), if *good { theme::TEXT } else { theme::APERTURE });
        }

        // The field: what to write, or a document's path to summarise.
        theme::well(fb, field, theme::HILIGHT);
        let fr = field.shrink(4);
        let room = (fr.w / col_w(1)).saturating_sub(1) as usize;
        if self.typed.is_empty() {
            dtext(fb, fr.x, fr.y + 2, "What should I write? (or a document's path to summarise)", theme::TEXT_DIM);
        } else {
            let shown = theme::tail_chars(&self.typed, room);
            dtext(fb, fr.x, fr.y + 2, shown, theme::TEXT);
            if focused {
                let cx = fr.x + theme::text_w_of_at(shown, dense());
                fb.rect(cx, fr.y + 1, 2, theme::text_h_at(dense()), theme::APERTURE);
            }
        }
        if focused && self.typed.is_empty() {
            fb.rect(fr.x, fr.y + 1, 2, theme::text_h_at(dense()), theme::APERTURE);
        }

        // Buttons, evenly across.
        let n = BUTTONS.len() as u32;
        for (i, (label, _)) in BUTTONS.iter().enumerate() {
            let x0 = buttons.x + buttons.w * i as u32 / n;
            let x1 = buttons.x + buttons.w * (i as u32 + 1) / n;
            let br = Rect::new(x0 + 2, buttons.y, x1.saturating_sub(x0 + 4), buttons.h);
            self.btns[i].set(br);
            theme::button_at(fb, br, label, i == 0 && !self.typed.is_empty(), false, dense());
        }

        // Examples, each one click from the field.
        theme::well(fb, examples, theme::HILIGHT);
        let e = examples.shrink(4);
        dtext(fb, e.x, e.y, "Try:", theme::TEXT_DIM);
        let room = (e.w.saturating_sub(col_w(2)) / col_w(1)) as usize;
        for (i, ex) in EXAMPLES.iter().enumerate() {
            let r = Rect::new(e.x, e.y + (i as u32 + 1) * rh, e.w, rh);
            self.examples[i].set(r);
            dtext(fb, r.x + col_w(2), r.y, theme::head_chars(ex, room), theme::LINK);
        }

        // What `office` has been doing, newest last.
        if activity.h > rh * 2 {
            theme::well(fb, activity, theme::HILIGHT);
            let a = activity.shrink(4);
            let rows = (a.h / rh) as usize;
            let log = crate::office::log_snapshot();
            let room = (a.w / col_w(1)) as usize;
            if log.is_empty() {
                dtext(fb, a.x, a.y, "Nothing yet. What the model writes is typed into Writer as it is written.", theme::TEXT_DIM);
            }
            for (i, line) in log.iter().rev().take(rows).rev().enumerate() {
                dtext(fb, a.x, a.y + i as u32 * rh, theme::head_chars(line, room), theme::TEXT);
            }
        }
    }

    fn key(&mut self, k: u8) -> bool {
        match k {
            b'\n' | b'\r' => {
                // A path summarises; anything else is something to write.
                let i = if self.typed.trim_start().starts_with('/') { 2 } else { 0 };
                self.fire(i)
            }
            8 => {
                self.typed.pop();
                true
            }
            0x20..=0x7E => {
                if self.typed.chars().count() < 400 {
                    self.typed.push(k as char);
                }
                true
            }
            _ => false,
        }
    }

    fn press(&mut self, client: Rect, x: i32, y: i32) -> bool {
        let ax = client.x as i32 + x;
        let ay = client.y as i32 + y;
        if let Some(i) = self.btns.iter().position(|b| inside(b.get(), ax, ay)) {
            return self.fire(i);
        }
        if let Some(i) = self.examples.iter().position(|b| inside(b.get(), ax, ay)) {
            self.typed = String::from(EXAMPLES[i]);
            return true;
        }
        false
    }

    fn min_size(&self) -> (u32, u32) {
        (420, 320)
    }
}

/// Open the window, or bring it forward if it is open.
pub fn open() {
    if super::desk::has_window("Office") {
        super::desk::focus_titled("Office");
        return;
    }
    let (w, h) = Office::preferred();
    super::desk::post(super::desk::Op::OpenApp {
        title: "Office",
        icon: super::desk::ICO_WRITE,
        app: alloc::boxed::Box::new(Office::new()),
        w,
        h,
    });
}

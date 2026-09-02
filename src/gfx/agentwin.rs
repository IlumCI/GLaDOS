//! The agent transcript, live on the desktop.
//!
//! The serial console shows an episode as it happens; this window shows the
//! same stream to the operator sitting at the machine. It draws from the
//! agent's shared ring -- the same lines the console prints, appended by the
//! agent task as it works -- and repaints through the diffed present each
//! time the episode reaches a step boundary, which is why watching it costs
//! nothing between steps.
//!
//! Scrolling follows the tail unless the operator scrolls up; new lines then
//! arrive without yanking the view, and scrolling back down resumes the
//! follow. This is the Browser's rule and the ToDo's rule, applied here
//! because it is the only behaviour that survives actually using either.

use super::theme::{self, Rect};
use super::{DeskApp, Framebuffer};
use super::mindwin::dense;
use crate::ai::agent;
use alloc::string::String;
use core::cell::Cell;

pub struct AgentLog {
    /// Lines the view is scrolled up from the bottom by.
    scroll: Cell<usize>,
}

impl AgentLog {
    pub fn new() -> Self {
        Self { scroll: Cell::new(0) }
    }

    pub fn preferred() -> (u32, u32) {
        (560, 360)
    }
}

impl DeskApp for AgentLog {
    /// A transcript line plus its observation, which is what one step of an
    /// episode looks like.
    fn min_size(&self) -> (u32, u32) {
        (400, 220)
    }

    fn draw_in(&self, fb: &Framebuffer, client: Rect, _focused: bool) {
        theme::panel(fb, client);
        // Log density, not chrome density. This window sits along the foot of
        // the workspace where it is a tail being watched out of the corner of
        // an eye, and at chrome scale a third of the screen held four lines of
        // it. `mindwin::dense()` is the same decision for the same reason.
        let lh = theme::text_h_at(dense()) + 2;
        let area = client.shrink(6);

        // One line saying what the machine is doing, above the log of what it
        // did. Both readings are in `glance`'s free tier and both were being
        // computed every second and rendered nowhere -- `doing` reached only
        // the terminal's status strip, and `journal`, which is the resident
        // mind's own narration of why it woke up, reached nothing at all.
        let head = crate::ai::glance::with_glance(|g| {
            let what = crate::ai::glance::engine_line(g);
            let mind = if g.mind_on { "mind on" } else { "mind off" };
            (
                alloc::format!("{}  |  {}  |  {} episode(s)", what, mind, g.episodes),
                g.engine.is_some(),
                g.journal.clone(),
            )
        });
        theme::text_over_at(
            fb,
            area.x,
            area.y,
            &head.0,
            if head.1 { theme::APERTURE } else { theme::TEXT_DIM },
            dense(),
        );
        let area = Rect::new(area.x, area.y + lh + 2, area.w, area.h.saturating_sub(lh + 2));
        let rows = (area.h / lh.max(1)) as usize;

        // The episode log, or -- when there has never been an episode -- the
        // journal, which is the machine's account of its own ticks. An empty
        // window that says "no episode yet" while the mind has been narrating
        // to itself for ten minutes is a window with its back turned.
        let lines = match agent::log_snapshot() {
            v if v.is_empty() => head.2,
            v => v,
        };
        // The draw pass owns the scroll clamp because only it knows how many
        // rows fit -- the same arrangement the ToDo window uses.
        let max_scroll = lines.len().saturating_sub(rows);
        let scroll = self.scroll.get().min(max_scroll);
        self.scroll.set(scroll);

        let start = lines.len().saturating_sub(scroll + rows);
        for k in 0..rows {
            let i = start + k;
            if i >= lines.len() {
                break;
            }
            let shown = clip_line(&lines[i], area.w);
            theme::text_over_at(fb, area.x, area.y + (k as u32) * lh, &shown, theme::TEXT, dense());
        }

        if lines.is_empty() {
            theme::text_over_at(
                fb,
                area.x,
                area.y,
                "nothing has happened yet -- 'agent <goal>' at the shell",
                theme::TEXT_DIM,
                dense(),
            );
        }
    }

    fn key(&mut self, _k: u8) -> bool {
        // A view, not an editor. Note that declining here does not send the
        // key to the shell -- the desktop answers every key with "handled"
        // while a non-terminal window has focus -- which is why opening this
        // window hands focus straight back to the terminal.
        false
    }

    fn press(&mut self, _client: Rect, _x: i32, _y: i32) -> bool {
        false
    }

    fn wheel(&mut self, notches: i32) -> bool {
        let delta = (notches * 3) as isize;
        let next = self.scroll.get() as isize - delta;
        self.scroll.set(next.max(0) as usize);
        true
    }
}

/// Chrome text has no clipping; a long observation line would walk off the
/// window and across the desktop behind it. Cut on the char boundary the
/// width allows -- `theme::text_w` counts glyphs, so this is exact.
fn clip_line(s: &str, max_w: u32) -> String {
    let budget = (max_w / theme::text_w_at(1, dense()).max(1)) as usize;
    if s.chars().count() <= budget {
        return String::from(s);
    }
    let mut out: String = s.chars().take(budget.saturating_sub(1)).collect();
    out.push('>');
    out
}

/// The authoring loop, while it runs.
///
/// A different question from the transcript above. That one answers "what
/// happened"; this answers "where is it now", and the second is the one
/// somebody staring at the screen has -- a run is minutes long and holds the
/// model for the whole of it, so a machine that looks exactly like a hung one
/// is a machine somebody reboots.
///
/// **No progress bar, and no estimate.** There is nothing to base one on: a
/// step can be a skeleton that lands instantly or a decode that takes seconds,
/// and the loop finishes when the checks pass rather than after a known amount
/// of work. `godel.rs` already paid for this lesson -- a working run and a hung
/// run look identical, and a bar that creeps to 90% and stops is worse than no
/// bar, because it converts "I cannot tell" into a claim that turns out false.
/// Step N of M is a fact. Anything smoother would be invented.
pub struct AuthorWin {
    /// Where Stop was last drawn, so the press pass hit-tests the same
    /// rectangle the paint pass drew -- the split this desktop forbids.
    stop: Cell<Rect>,
}

impl AuthorWin {
    pub fn new() -> Self {
        Self { stop: Cell::new(Rect::new(0, 0, 0, 0)) }
    }

    /// Narrow enough to sit entirely clear of the terminal on this machine's
    /// 1280-wide panel. A progress window half under something else is a
    /// progress window nobody reads.
    pub fn preferred() -> (u32, u32) {
        (480, 250)
    }
}

impl DeskApp for AuthorWin {
    fn min_size(&self) -> (u32, u32) {
        (360, 200)
    }

    fn draw_in(&self, fb: &Framebuffer, client: Rect, _focused: bool) {
        theme::panel(fb, client);
        let lh = theme::text_h_at(dense()) + 2;
        let area = client.shrink(8);
        let mut y = area.y;

        let Some(p) = crate::ai::author::progress() else {
            theme::text_over_at(fb, area.x, y, "nothing is being written", theme::TEXT_DIM, dense());
            self.stop.set(Rect::new(0, 0, 0, 0));
            return;
        };

        theme::text(
            fb,
            area.x,
            y,
            &alloc::format!("{} {}", if p.running { "writing" } else { "wrote" }, p.name),
            theme::TEXT,
            theme::FACE,
        );
        y += lh + lh / 2;

        theme::text(
            fb,
            area.x,
            y,
            &alloc::format!("step {} of {}", p.step, p.budget),
            theme::TEXT,
            theme::FACE,
        );
        y += lh;
        theme::text(
            fb,
            area.x,
            y,
            &alloc::format!("{} of {} clause(s) met", p.met, p.total),
            theme::TEXT,
            theme::FACE,
        );
        y += lh + lh / 2;

        // The verdict verbatim, wrapped rather than clipped. It is what the
        // loop itself is acting on, and the half that gets cut off is where
        // the line number lives.
        theme::text_over_at(fb, area.x, y, "last check", theme::TEXT_DIM, dense());
        y += lh;
        // One character's width, asked for as the width of one character --
        // `text_w` measures a string rather than answering a constant.
        let cw = theme::text_w_at(1, dense()).max(1);
        let cols = (area.w / cw).max(8) as usize;
        for chunk in wrap(&p.last, cols).iter().take(3) {
            theme::text_over_at(fb, area.x, y, chunk, theme::TEXT, dense());
            y += lh;
        }

        // Stop sits at the bottom, and only while there is something to stop.
        if p.running {
            let bw = 90u32.min(area.w);
            let r = Rect::new(area.x, area.y + area.h.saturating_sub(lh + 10), bw, lh + 8);
            theme::button_at(fb, r, "Stop", false, false, dense());
            self.stop.set(r);
        } else {
            self.stop.set(Rect::new(0, 0, 0, 0));
        }
    }

    fn key(&mut self, _k: u8) -> bool {
        // Nothing. The window must not hold the keyboard: it is opened by the
        // run rather than by somebody asking for it, and a window that appears
        // under the cursor and swallows the next line typed is the defect this
        // desktop has already paid for twice.
        false
    }

    fn press(&mut self, client: Rect, x: i32, y: i32) -> bool {
        let r = self.stop.get();
        if r.w == 0 {
            return false;
        }
        let (px, py) = (client.x as i32 + x, client.y as i32 + y);
        if px >= r.x as i32
            && px < (r.x + r.w) as i32
            && py >= r.y as i32
            && py < (r.y + r.h) as i32
        {
            crate::ai::agent::request_abort();
            return true;
        }
        false
    }
}

/// Break text on spaces at `cols`, never mid-word unless a word is longer.
fn wrap(text: &str, cols: usize) -> alloc::vec::Vec<String> {
    let mut out = alloc::vec::Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > cols {
            out.push(core::mem::take(&mut line));
        }
        if word.chars().count() > cols {
            // A path or a hash with no spaces in it. Cut it rather than let
            // one word push everything after it off the window.
            let cut: String = word.chars().take(cols).collect();
            out.push(cut);
            continue;
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

/// The wrap, which is the one piece of logic in this window.
///
/// Worth pinning because of what it protects: the verdict is shown verbatim so
/// the operator can read the line number in it, and a wrap that drops or
/// truncates the tail loses exactly that. The last case is the one a naive
/// wrap gets wrong -- a word longer than the line, which is a path or a hash,
/// and which must be cut rather than allowed to push everything after it off
/// the window.
pub fn selftest() -> bool {
    // Breaks on a space, never mid-word, and keeps every word.
    let w = wrap("line 7: unknown verb", 12);
    if w.len() != 2 || w[0] != "line 7:" || w[1] != "unknown verb" {
        return false;
    }
    // Short enough is one line and is not padded or split.
    if wrap("ok", 12) != alloc::vec![String::from("ok")] {
        return false;
    }
    // Nothing in, nothing out -- not one empty line, which would draw as a
    // blank row the operator reads as "no verdict".
    if !wrap("   ", 12).is_empty() {
        return false;
    }
    // A word with no spaces in it is cut to the width instead of overflowing.
    let long = wrap("/draft/verylongname/panel.ui", 10);
    if long.len() != 1 || long[0].chars().count() != 10 {
        return false;
    }
    true
}

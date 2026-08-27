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
        let lh = theme::text_h();
        let area = client.shrink(6);
        let rows = (area.h / lh.max(1)) as usize;

        let lines = agent::log_snapshot();
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
            theme::text(
                fb,
                area.x,
                area.y + (k as u32) * lh,
                &shown,
                theme::TEXT,
                theme::FACE,
            );
        }

        if lines.is_empty() {
            theme::text(
                fb,
                area.x,
                area.y,
                "no episode yet -- run 'agent <goal>' at the shell",
                theme::TEXT_DIM,
                theme::FACE,
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
    let budget = (max_w / theme::text_w(1).max(1)) as usize;
    if s.chars().count() <= budget {
        return String::from(s);
    }
    let mut out: String = s.chars().take(budget.saturating_sub(1)).collect();
    out.push('>');
    out
}

//! Oracle. God Says, made to tell the one future that is knowable.
//!
//! TempleOS drew uniform words and let the operator hear prophecy; the
//! randomness was the timing of the operator's hands. This keeps that entropy
//! (every key and mouse move since boot feeds it, [`crate::ai::godbits`]) and
//! changes the subject from hallucinated words to the machine's measured
//! state. It samples uptime, heap, task switching and the operator's own
//! touch rate once a second, fits a linear dynamical model from that history
//! with the router's Cholesky solver, and rolls the state forward under three
//! interventions -- left alone, carried on, put under load. The window draws
//! the result as forked timelines: solid history up to now, three diverging
//! projections after.
//!
//! Honest where the word-prophecy version was not. A branch is the
//! counterfactual `do(activity := level)` on a controlled linear system fitted
//! from how the machine has actually behaved. It predicts the machine, from
//! the machine. The noise on each step is the God-bits fold, so the reading
//! still depends on every touch the machine has felt -- the operator still
//! plays the instrument, now over real state.

use super::theme::{self, Rect};
use super::{Color, DeskApp, Framebuffer};
use crate::ai::futures::{self, NVARS, VAR_NAMES};
use alloc::format;
use alloc::string::String;

/// One colour per branch: left alone, carried on, under load.
const BRANCH_COLORS: [Color; 3] = [
    Color::new(0x5A, 0x9B, 0xD5), // cool blue -- the machine at rest
    theme::APERTURE,              // amber -- the present carried forward
    Color::new(0xD5, 0x5A, 0x3A), // hot red -- pushed hard
];

pub struct Oracle {
    proj: Option<futures::Projection>,
    /// Which state variable the big graph shows.
    var: usize,
    status: String,
}

impl Oracle {
    pub fn new(_arg: &str) -> Self {
        let mut o = Self { proj: None, var: 0, status: String::new() };
        o.consult();
        o
    }

    pub fn preferred() -> (u32, u32) {
        (640, 460)
    }

    fn consult(&mut self) {
        // Fast: a 3x3 fit per variable and a rollout, no model forward passes.
        // The window stays responsive, unlike anything that runs the network.
        let pr = futures::project(24);
        self.status = if pr.fitted {
            format!(
                "3 futures fitted from {} samples, fed by {} touches",
                pr.hist.len(),
                pr.felt
            )
        } else {
            format!(
                "warming up: {} of 4 samples. leave it running a few seconds",
                pr.hist.len()
            )
        };
        self.proj = Some(pr);
    }

    fn layout(client: Rect) -> (Rect, Rect, Rect) {
        let lh = theme::text_h();
        let tabs = Rect::new(client.x + 8, client.y + 6, client.w.saturating_sub(16), lh + 8);
        let graph = Rect::new(
            client.x + 8,
            tabs.y + tabs.h + 6,
            client.w.saturating_sub(16),
            client.h.saturating_sub(tabs.h + lh * 3 + 34),
        );
        let legend = Rect::new(
            client.x + 8,
            graph.y + graph.h + 4,
            client.w.saturating_sub(16),
            lh * 2 + 6,
        );
        (tabs, graph, legend)
    }

    fn var_tabs(tabs: Rect) -> [Rect; NVARS] {
        let w = tabs.w / NVARS as u32;
        core::array::from_fn(|i| Rect::new(tabs.x + i as u32 * w, tabs.y, w - 2, tabs.h))
    }

    /// Value range of the selected variable across history and every branch,
    /// so all timelines share one honest scale.
    fn range(&self, var: usize) -> (f32, f32) {
        let Some(pr) = &self.proj else { return (0.0, 1.0) };
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for s in &pr.hist {
            lo = lo.min(s.vars[var]);
            hi = hi.max(s.vars[var]);
        }
        for b in &pr.branches {
            for v in &b.traj {
                lo = lo.min(v[var]);
                hi = hi.max(v[var]);
            }
        }
        if !lo.is_finite() || !hi.is_finite() {
            return (0.0, 1.0);
        }
        // A flat line should sit mid-graph, not have its own noise magnified.
        if (hi - lo).abs() < 1e-3 {
            return (lo - 1.0, hi + 1.0);
        }
        let pad = (hi - lo) * 0.1;
        (lo - pad, hi + pad)
    }
}

impl DeskApp for Oracle {
    /// The plot needs a history axis worth reading; below this the forked
    /// projections overlap into a single line.
    fn min_size(&self) -> (u32, u32) {
        (420, 300)
    }

    fn draw_in(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        theme::panel(fb, client);
        let (tabs, graph, legend) = Self::layout(client);
        let lh = theme::text_h();
        let _ = focused;

        for (i, r) in Self::var_tabs(tabs).iter().enumerate() {
            theme::button(fb, *r, VAR_NAMES[i], i == self.var, i == self.var);
        }

        theme::well(fb, graph, theme::SCREEN);
        let g = graph.shrink(6);

        let Some(pr) = &self.proj else { return };
        let (lo, hi) = self.range(self.var);
        let span = (hi - lo).max(1e-3);

        let hist_span = pr.hist.len().max(1) as f32;
        let total_t = hist_span + pr.steps as f32;
        let x_of = |t: f32| g.x + (t / total_t * g.w as f32) as u32;
        let y_of = |v: f32| {
            let f = ((v - lo) / span).clamp(0.0, 1.0);
            g.y + g.h - (f * g.h as f32) as u32
        };

        // Guide lines and value labels at hi, mid, lo.
        for (frac, v) in [(0.0f32, hi), (0.5, (lo + hi) / 2.0), (1.0, lo)] {
            let y = g.y + (frac * g.h as f32) as u32;
            let mut x = g.x;
            while x < g.x + g.w {
                fb.rect(x, y, 2, 1, theme::SHADOW);
                x += 6;
            }
            let mut s = String::new();
            push_num(&mut s, v as i64);
            theme::text_over(fb, g.x + 2, y.saturating_sub(lh + 1).max(g.y), &s, theme::SHADOW);
        }

        // The fork: where history ends and the futures begin.
        let fork_x = x_of(hist_span);
        let mut fy = g.y;
        while fy < g.y + g.h {
            fb.rect(fork_x, fy, 1, 2, theme::SCREEN_TEXT);
            fy += 4;
        }
        theme::text_over(fb, fork_x + 3, g.y + 1, "now", theme::SCREEN_TEXT);

        // History, solid white -- the shared past.
        for i in 1..pr.hist.len() {
            let (a, b) = (pr.hist[i - 1].vars[self.var], pr.hist[i].vars[self.var]);
            fb.line(
                x_of(i as f32 - 1.0) as i32,
                y_of(a) as i32,
                x_of(i as f32) as i32,
                y_of(b) as i32,
                theme::HILIGHT,
            );
        }

        // The three futures, each its colour, from the fork point.
        let start = pr.hist.last().map(|s| s.vars[self.var]).unwrap_or(0.0);
        for (bi, b) in pr.branches.iter().enumerate() {
            let col = BRANCH_COLORS[bi.min(2)];
            let mut prev = (hist_span, start);
            for (k, v) in b.traj.iter().enumerate() {
                let cur = (hist_span + k as f32 + 1.0, v[self.var]);
                fb.line(
                    x_of(prev.0) as i32,
                    y_of(prev.1) as i32,
                    x_of(cur.0) as i32,
                    y_of(cur.1) as i32,
                    col,
                );
                prev = cur;
            }
        }

        // Legend: current state, and each intervention's end-state for this var.
        let now = futures::snapshot_now();
        let head = format!(
            "now: heap {} KiB  {}/s switch  {}/s touch  {} tasks",
            now.vars[0] as u64, now.vars[1] as u64, now.vars[2] as u64, now.vars[3] as u64
        );
        theme::text(fb, legend.x, legend.y, &head, theme::TEXT, theme::FACE);
        let mut lx = legend.x;
        for (bi, b) in pr.branches.iter().enumerate() {
            let col = BRANCH_COLORS[bi.min(2)];
            let ly = legend.y + lh + 4;
            fb.rect(lx, ly + 2, 10, 8, col);
            let end = b.traj.last().map(|v| v[self.var]).unwrap_or(0.0);
            let mut label = format!("{} to ", b.name);
            push_num(&mut label, end as i64);
            theme::text(fb, lx + 14, ly, &label, theme::TEXT, theme::FACE);
            lx += theme::text_w(label.len() + 3) + 16;
        }

        let sy = legend.y + legend.h + 2;
        if sy + lh < client.y + client.h {
            theme::text(fb, client.x + 8, sy, &self.status, theme::TEXT, theme::FACE);
        }
    }

    fn key(&mut self, k: u8) -> bool {
        use crate::dev::kbd;
        match k {
            b'\n' | b'\r' => self.consult(),
            kbd::KEY_LEFT | kbd::KEY_UP => self.var = (self.var + NVARS - 1) % NVARS,
            kbd::KEY_RIGHT | kbd::KEY_DOWN | b'\t' => self.var = (self.var + 1) % NVARS,
            _ => return false,
        }
        true
    }

    fn press(&mut self, client: Rect, x: i32, y: i32) -> bool {
        let (tabs, _, _) = Self::layout(client);
        for (i, r) in Self::var_tabs(tabs).iter().enumerate() {
            if x >= r.x as i32 && y >= r.y as i32 && x < (r.x + r.w) as i32 && y < (r.y + r.h) as i32 {
                self.var = i;
                return true;
            }
        }
        // A press in the graph re-consults, the ritual gesture.
        self.consult();
        true
    }

    fn wheel(&mut self, notches: i32) -> bool {
        let was = self.var;
        self.var = if notches > 0 {
            (self.var + 1) % NVARS
        } else {
            (self.var + NVARS - 1) % NVARS
        };
        self.var != was
    }
}

/// Integer to string; the kernel console has no float formatting.
fn push_num(s: &mut String, mut n: i64) {
    if n < 0 {
        s.push('-');
        n = -n;
    }
    if n == 0 {
        s.push('0');
        return;
    }
    let mut digits = [0u8; 20];
    let mut i = 0;
    while n > 0 {
        digits[i] = b'0' + (n % 10) as u8;
        n /= 10;
        i += 1;
    }
    while i > 0 {
        i -= 1;
        s.push(digits[i] as char);
    }
}

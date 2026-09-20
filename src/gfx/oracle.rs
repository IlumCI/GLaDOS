//! What the machine was thinking, drawn from what it already computed.
//!
//! **This window used to plot three futures and they were the same line three
//! times.** `futures::project` fits `v' = a + b*v + c*u` per variable and rolls
//! it forward under three forced values of `u`, the operator's touch rate --
//! and `godbits::FELT`, which is that rate, is incremented from the i8042 and
//! PS/2 interrupt handlers and nowhere else. `win keys` bypasses both and so
//! does serial, so on every machine this has ever been driven on the control
//! was **zero in every sample**. A coefficient fitted against a treatment that
//! never varies is not small, it is unidentifiable; the "left alone" and
//! "carried on" branches were literally the same intervention; and the third
//! multiplied a zero. Measured, twice in one boot: heap rising 16192 -> 24408
//! KiB while all three futures predicted it falling, and "put under load"
//! predicting the *least* heap of the three.
//!
//! The telemetry itself is fine and still feeds `aixi` and `context`. What was
//! wrong was showing it here as though it meant something.
//!
//! So the window shows reasoning the machine really does. Every panel below is
//! a quantity that was already being computed and then discarded:
//!
//! - **Route**: one hidden state, a ridge score for every applet the trust
//!   gate admits, three independent cores voting, and the rule that resolved
//!   them. All of it fell on the floor at the end of `route_verdict`.
//! - **Council**: how often those three agree. Unanimity was measured at 90.3%
//!   correct against 50% when split, which is the one number here that should
//!   change what somebody does.
//! - **Ledger**: the self-improvement machine's search state -- which cells of
//!   the MAP-Elites archive are lit, and how uncertain each axis is.
//! - **Outcome**: how far apart two applets are, measured by what they print.
//!
//! ### Nothing here may block the compositor
//!
//! `draw_in` runs on the task that owns the screen, and after the single-writer
//! work that is the *only* painter -- so a panel that waited on the model would
//! stop the whole display, not just itself. `trace` copies applet names in at
//! record time so this never calls `with_engine`, the ledger reads are small
//! namespace lookups, and the outcome matrix dispatches fourteen applets and is
//! therefore behind a keypress rather than computed on a frame.

use super::theme::{self, Rect};
use super::{Color, DeskApp, Framebuffer};
use crate::ai::trace;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

const TABS: [&str; 5] = ["Route", "Net", "Council", "Ledger", "Outcome"];

/// The probe, the lexical core, the character core. One colour each, used for
/// the vote markers and the agreement bars alike so the two panels read as one
/// idea.
const VOTER_COLORS: [Color; 3] = [
    theme::APERTURE,               // the probe -- the machine's own voice
    Color::new(0x5A, 0x9B, 0xD5),  // lexical
    Color::new(0x6C, 0xC2, 0x8A),  // character
];

pub struct Oracle {
    tab: usize,
    /// The applet distance matrix, once somebody has asked for it. Never
    /// computed during a frame: `outcome::probe` dispatches fourteen applets
    /// and captures their output, which is a long way past what a paint may do.
    matrix: Option<crate::ai::outcome::Matrix>,
    status: String,
}

impl Oracle {
    pub fn new(arg: &str) -> Self {
        let tab = TABS
            .iter()
            .position(|t| t.eq_ignore_ascii_case(arg.trim()))
            .unwrap_or(0);
        Self { tab, matrix: None, status: String::new() }
    }

    pub fn preferred() -> (u32, u32) {
        (700, 480)
    }

    fn layout(client: Rect) -> (Rect, Rect, Rect) {
        let lh = theme::text_h();
        let tabs = Rect::new(client.x + 6, client.y + 6, client.w.saturating_sub(12), lh + 10);
        let foot = Rect::new(
            client.x + 6,
            client.y + client.h.saturating_sub(lh + 8),
            client.w.saturating_sub(12),
            lh + 2,
        );
        let body = Rect::new(
            client.x + 6,
            tabs.y + tabs.h + 6,
            client.w.saturating_sub(12),
            client
                .h
                .saturating_sub(tabs.h + lh + 28),
        );
        (tabs, body, foot)
    }

    fn tab_rects(tabs: Rect) -> Vec<Rect> {
        let w = tabs.w / TABS.len() as u32;
        (0..TABS.len())
            .map(|i| Rect::new(tabs.x + i as u32 * w, tabs.y, w.saturating_sub(4), tabs.h))
            .collect()
    }

    // --- Route -----------------------------------------------------------

    fn draw_route(&self, fb: &Framebuffer, r: Rect) -> String {
        let lh = theme::text_h();
        let all = trace::recent();
        let Some(d) = all.last() else {
            theme::text_over(fb, r.x + 8, r.y + 8,
                "no routing decision yet -- ask it something, or run 'teach'",
                theme::SCREEN_TEXT);
            return String::from("the router has not been asked anything this boot");
        };

        // What was asked, and what the grammar left of the table.
        let ask = clip(&d.task, fits(r.w.saturating_sub(20)));
        theme::text_over(fb, r.x + 8, r.y + 6, &format!("\"{}\"", ask), theme::FACE);
        theme::text_over(
            fb,
            r.x + 8,
            r.y + 8 + lh,
            &clip(&format!("{} of {} applets reachable", d.allowed, d.total),
                  fits(r.w.saturating_sub(16))),
            theme::SHADOW,
        );

        // The candidates, as bars. Scores are ridge outputs and can be
        // negative, so the bar is drawn against the span of what is shown
        // rather than against zero -- a bar chart with no zero on it is
        // honest here and would not be if these were counts.
        let top = r.y + 12 + lh * 2;
        let rows = d.cand.len().min(trace::TOPN) as u32;
        if rows == 0 {
            return String::from("nothing was reachable at this trust level");
        }
        let row_h = ((r.y + r.h).saturating_sub(top) / rows.max(1)).min(lh + 12);
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for c in &d.cand {
            lo = lo.min(c.score);
            hi = hi.max(c.score);
        }
        let span = (hi - lo).max(1e-3);
        let label_w = theme::text_w(10);
        let bar_x = r.x + 10 + label_w;
        let bar_w = r.w.saturating_sub(label_w + 30);

        for (i, c) in d.cand.iter().take(rows as usize).enumerate() {
            let y = top + i as u32 * row_h;
            let won = c.class == d.winner;
            theme::text_over(fb, r.x + 8, y + 2, &clip(&c.name, 10),
                             if won { theme::APERTURE } else { theme::SCREEN_TEXT });
            let f = ((c.score - lo) / span).clamp(0.02, 1.0);
            let w = (f * bar_w as f32) as u32;
            fb.rect(bar_x, y + 3, bar_w, row_h.saturating_sub(8), theme::SCREEN);
            fb.rect(bar_x, y + 3, w.max(2), row_h.saturating_sub(8),
                    if won { theme::APERTURE_DEEP } else { Color::new(0x2A, 0x3A, 0x46) });

            // Who voted for this one. Three small marks rather than a legend,
            // so a split decision is visible at a glance as marks that do not
            // line up.
            let mut mx = bar_x + bar_w + 4;
            for (vi, v) in [d.probe, d.lexical, d.character].iter().enumerate() {
                if *v == c.class && d.settled {
                    fb.rect(mx, y + 5, 5, row_h.saturating_sub(12), VOTER_COLORS[vi]);
                }
                mx += 7;
            }
        }

        if d.settled {
            format!(
                "{} -- {} of 3 agreed ({})",
                d.winner_name(),
                d.agreement,
                if d.agreement == 3 { "90% right" } else { "a split is 50%" }
            )
        } else {
            format!("{} -- the probe alone, no cores asked", d.winner_name())
        }
    }


    // --- Net -------------------------------------------------------------

    /// The probe as the network it is, carrying the decision it really made.
    ///
    /// **Every edge here is a term of the sum that produced the answer.**
    /// `Probe::scores` is `sum_i w[c][i] * (x[i] - mean[i])`; `contributions`
    /// groups those terms into slices of the hidden state and the totals come
    /// back equal to the scores, which `diag probe` asserts. So a thick warm
    /// edge is a slice of the state genuinely pushing that applet up, and a
    /// cool one is genuinely pushing it down. Nothing here is drawn because it
    /// looks like a neural network.
    ///
    /// The animation is a wavefront sweeping left to right, and it is honest
    /// about direction only: signal really does go from the hidden state to
    /// the scores, once, with no recurrence. It carries no other meaning and
    /// is not pretending to show time.
    fn draw_net(&self, fb: &Framebuffer, r: Rect) -> String {
        let lh = theme::text_h();
        let all = trace::recent();
        let Some(d) = all.last().filter(|d| !d.act.is_empty() && !d.cand.is_empty()) else {
            theme::text_over(fb, r.x + 8, r.y + 8,
                &clip("no decision to draw -- 'fit' then 'route <task>'",
                      fits(r.w.saturating_sub(16))),
                theme::SCREEN_TEXT);
            return String::from("the probe has not scored anything this boot");
        };

        let bins = d.act.len();
        let outs = d.cand.len();
        if bins == 0 || outs == 0 || d.edge.len() < outs * bins {
            return String::from("the record is incomplete");
        }

        // Phase from the clock, not from stored state: `draw_in` takes `&self`
        // and the frame is composed by the compositor, so an animation that
        // needed to mutate would need a cell and a writer. Time is already
        // shared and already moves.
        let ms = crate::dev::lapic::ticks() as u64 * 1000 / crate::TIMER_HZ as u64;
        let phase = (ms % 1400) as f32 / 1400.0;

        let top = r.y + 6 + lh;
        let bot = r.y + r.h.saturating_sub(6);
        let col_l = r.x + 26;
        let col_r = r.x + r.w.saturating_sub(theme::text_w(9) + 16);
        let span = (bot.saturating_sub(top)).max(1);

        theme::text_over(fb, r.x + 8, r.y + 2,
            &clip(&format!("{} slices of hidden state -> {} applets", bins, outs),
                  fits(r.w.saturating_sub(16))),
            theme::SHADOW);

        let y_in = |b: usize| top + (b as u32 * span) / bins.max(1) as u32;
        let y_out = |c: usize| top + (c as u32 * span) / outs.max(1) as u32 + span / (outs as u32 * 2).max(1);

        // Scale edges against the strongest, so the picture is readable on a
        // checkpoint whose weights are any size.
        let mut peak = 1e-6f32;
        for v in &d.edge {
            if v.abs() > peak {
                peak = v.abs();
            }
        }

        // **The strongest few per output, not everything above a threshold.**
        //
        // A global cut still passed about a hundred of the hundred and
        // forty-four and drew a hairball -- which is exactly what a decorative
        // network picture looks like, and the thing this panel exists not to
        // be. Per output, the slices that actually carry the decision are a
        // handful; showing those makes it legible that different applets are
        // driven by different parts of the state.
        const PER_OUT: usize = 7;
        for c in 0..outs {
            let mut rank: Vec<(usize, f32)> =
                (0..bins).map(|b| (b, d.edge[c * bins + b].abs())).collect();
            rank.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(core::cmp::Ordering::Equal));
            rank.truncate(PER_OUT);
            for (b, _) in rank {
                let w = d.edge[c * bins + b];
                let mag = (w.abs() / peak).clamp(0.0, 1.0);
                if mag < 0.05 {
                    continue;
                }
                let (y0, y1) = (y_in(b), y_out(c));
                // The wavefront: an edge brightens as the pulse crosses it.
                let mid = 0.5f32;
                let d0 = ((phase - mid).abs() * 2.0).clamp(0.0, 1.0);
                let lit = 1.0 - d0 * 0.75;
                let k = (mag * lit * 255.0) as u8;
                let col = if w >= 0.0 {
                    Color::new(k, (k as u16 * 150 / 255) as u8, (k as u16 * 40 / 255) as u8)
                } else {
                    Color::new((k as u16 * 50 / 255) as u8, (k as u16 * 110 / 255) as u8, k)
                };
                fb.line(col_l as i32 + 6, y0 as i32, col_r as i32 - 6, y1 as i32, col);
            }
        }

        // The input slices. Height is the centred activation summed over the
        // slice, so a tall node is a part of the state that is far from the
        // average sentence.
        let mut apeak = 1e-6f32;
        for v in &d.act {
            if v.abs() > apeak {
                apeak = v.abs();
            }
        }
        for b in 0..bins {
            let a = (d.act[b].abs() / apeak).clamp(0.0, 1.0);
            let y = y_in(b);
            let s = 3 + (a * 5.0) as u32;
            let k = (60.0 + a * 195.0) as u8;
            fb.rect(col_l.saturating_sub(s / 2), y.saturating_sub(s / 2), s, s,
                    Color::new(k / 2, k, k));
        }

        // The outputs, in the order the probe ranked them.
        for (c, cand) in d.cand.iter().enumerate() {
            let y = y_out(c);
            let won = cand.class == d.winner;
            let s = if won { 11 } else { 7 };
            fb.rect(col_r.saturating_sub(s / 2), y.saturating_sub(s / 2), s, s,
                    if won { theme::APERTURE } else { Color::new(0x6C, 0x8A, 0x9A) });
            theme::text_over(fb, col_r + 10, y.saturating_sub(lh / 2), &clip(&cand.name, 8),
                             if won { theme::APERTURE } else { theme::SCREEN_TEXT });
        }

        String::from("warm pushes up, cool pushes down")
    }

    // --- Council ---------------------------------------------------------

    fn draw_council(&self, fb: &Framebuffer, r: Rect) -> String {
        let lh = theme::text_h();
        let census = trace::agreement_census();
        let total: usize = census.iter().sum();
        theme::text_over(fb, r.x + 8, r.y + 6,
            &clip("three cores vote on every decision", fits(r.w.saturating_sub(16))),
            theme::SHADOW);

        let top = r.y + 10 + lh;
        let bar_h = lh + 8;
        let label_w = theme::text_w(12);
        for (i, n) in census.iter().enumerate().rev() {
            let k = 2 - i; // draw 3-of-3 first
            let y = top + k as u32 * (bar_h + 6);
            let agree = i + 1;
            theme::text_over(fb, r.x + 8, y + 2,
                &format!("{} of 3", agree),
                if agree == 3 { theme::APERTURE } else { theme::SCREEN_TEXT });
            let bw = r.w.saturating_sub(label_w + 40);
            fb.rect(r.x + 8 + label_w, y, bw, bar_h, theme::SCREEN);
            if total > 0 {
                let f = *n as f32 / total as f32;
                fb.rect(r.x + 8 + label_w, y, ((f * bw as f32) as u32).max(1), bar_h,
                        VOTER_COLORS[2 - k as usize]);
            }
            theme::text_over(fb, r.x + 8 + label_w + bw + 6, y + 2,
                             &format!("{}", n), theme::SCREEN_TEXT);
        }

        if total == 0 {
            return String::from("no settled decisions yet");
        }
        format!("{} decision(s) -- unanimity is the signal", total)
    }

    // --- Ledger ----------------------------------------------------------

    fn draw_ledger(&self, fb: &Framebuffer, r: Rect) -> String {
        use crate::ai::godel;
        let lh = theme::text_h();
        theme::text_over(fb, r.x + 8, r.y + 6,
            &clip("archive: rank across, repair down",
                  fits(r.w.saturating_sub(16))),
            theme::SHADOW);

        // The MAP-Elites archive as the grid it actually is. A variant earns a
        // cell by beating whatever is in that cell, not by beating the
        // champion, so a lit cell is a niche somebody survived in.
        let cols = 4u32;
        let rows = (godel::CELLS as u32 / cols).max(1);
        let gw = (r.w.saturating_sub(16)).min(320);
        let cw = gw / cols;
        let ch = (lh + 14).min(34);
        let top = r.y + 10 + lh;
        let mut lit = 0;
        for i in 0..godel::CELLS {
            let cx = r.x + 8 + (i as u32 % cols) * cw;
            let cy = top + (i as u32 / cols) * (ch + 4);
            let e = godel::cell(i);
            // An empty cell has to be visible as an empty cell. The first
            // version filled it with `SCREEN`, which is the well it sits in,
            // so a fresh machine showed no grid at all -- and "nothing has
            // been tried yet" and "this panel is broken" looked identical.
            let w = cw.saturating_sub(4);
            match &e {
                Some(_) => {
                    lit += 1;
                    fb.rect(cx, cy, w, ch, theme::APERTURE_DEEP);
                }
                None => {
                    fb.rect(cx, cy, w, ch, Color::new(0x14, 0x1B, 0x22));
                    fb.rect(cx, cy, w, 1, Color::new(0x2C, 0x3A, 0x45));
                    fb.rect(cx, cy + ch - 1, w, 1, Color::new(0x2C, 0x3A, 0x45));
                    fb.rect(cx, cy, 1, ch, Color::new(0x2C, 0x3A, 0x45));
                    fb.rect(cx + w - 1, cy, 1, ch, Color::new(0x2C, 0x3A, 0x45));
                }
            }
            if let Some(el) = e {
                theme::text_over(fb, cx + 4, cy + 3,
                                 &format!("{}", (el.score * 100.0) as i64),
                                 theme::TITLE_TEXT);
            }
        }

        // Which axis the loop reaches for next is decided by which one it can
        // least predict, so the counts are the search's own state.
        let ax_x = r.x + 16 + gw;
        let counts = godel::axis_counts();
        for (i, (yes, no)) in counts.iter().enumerate() {
            let y = top + i as u32 * (lh + 3);
            if y + lh > r.y + r.h {
                break;
            }
            let n = yes + no;
            theme::text_over(fb, ax_x, y,
                &format!("{:<8}{}/{}", godel::AXIS_NAMES[i], yes, n),
                if n == 0 { theme::SHADOW } else { theme::SCREEN_TEXT });
        }

        format!("{} of {} cells lit, {} trials on the ledger",
                lit, godel::CELLS, godel::ledger_len())
    }

    // --- Outcome ---------------------------------------------------------

    fn draw_outcome(&self, fb: &Framebuffer, r: Rect) -> String {
        let lh = theme::text_h();
        let Some(m) = &self.matrix else {
            theme::text_over(fb, r.x + 8, r.y + 6,
                "how far apart two applets are, by what they printed",
                theme::SHADOW);
            theme::text_over(fb, r.x + 8, r.y + 10 + lh,
                "press M to measure -- it runs fourteen applets,",
                theme::SCREEN_TEXT);
            theme::text_over(fb, r.x + 8, r.y + 12 + lh * 2,
                "which is why it is not done on a frame",
                theme::SCREEN_TEXT);
            return String::from("not measured this boot");
        };

        let n = m.len().min(16) as u32;
        let cell = ((r.w.saturating_sub(16)).min(r.h.saturating_sub(lh + 16)) / n.max(1)).max(4);
        let top = r.y + 8 + lh;
        theme::text_over(fb, r.x + 8, r.y + 4,
            &clip("dark is near, light is far", fits(r.w.saturating_sub(16))),
            theme::SHADOW);
        for a in 0..n {
            for b in 0..n {
                let d = m.get(a as usize, b as usize).clamp(0.0, 1.0);
                let v = (d * 220.0) as u8;
                fb.rect(r.x + 8 + b * cell, top + a * cell,
                        cell.saturating_sub(1), cell.saturating_sub(1),
                        Color::new(v / 3, v / 2, v));
            }
        }
        format!("{} of {} ran; the rest mutate", m.ran, m.len())
    }
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return String::from(s);
    }
    s.chars().take(n.saturating_sub(1)).collect::<String>() + "."
}

/// How many characters fit across `w` pixels.
///
/// **Measured rather than guessed, because a guess clipped every line.** The
/// first version passed literal counts -- 78 for the footer -- which is a
/// character count standing in for a width, and this tree has paid for that
/// confusion in three other places. A glyph is eight pixels doubled by
/// `CHROME_SCALE`, so the answer is forty across this window and not seventy
/// eight, and the sentences lost their ends.
fn fits(w: u32) -> usize {
    (w / theme::text_w(1).max(1)) as usize
}

impl DeskApp for Oracle {
    fn draw_in(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        theme::panel(fb, client);
        let (tabs, body, foot) = Self::layout(client);

        for (i, rect) in Self::tab_rects(tabs).iter().enumerate() {
            theme::button(fb, *rect, TABS[i], i == self.tab, i == self.tab);
        }

        // **The animation asks for the next frame, and that is a real cost.**
        //
        // The compositor composes only when something says the screen is out
        // of date, which is what keeps an idle desktop free. A moving picture
        // has to keep saying so, and a frame is 2,143 us measured -- about 7%
        // of a core at sixty a second. So it is asked for only while this
        // window has focus and the panel that moves is the one on screen:
        // a background window animating something nobody is looking at would
        // be the desktop paying for a decoration, which is the failure this
        // whole window was rebuilt to stop committing.
        if focused && self.tab == 1 {
            super::render::invalidate();
        }

        theme::well(fb, body, theme::SCREEN);
        let inner = body.shrink(4);
        let note = match self.tab {
            0 => self.draw_route(fb, inner),
            1 => self.draw_net(fb, inner),
            2 => self.draw_council(fb, inner),
            3 => self.draw_ledger(fb, inner),
            _ => self.draw_outcome(fb, inner),
        };
        let line = if self.status.is_empty() { note } else { self.status.clone() };
        theme::text_over(fb, foot.x + 2, foot.y, &clip(&line, fits(foot.w)), theme::SHADOW);
    }

    fn key(&mut self, k: u8) -> bool {
        match k {
            b'\t' => {
                self.tab = (self.tab + 1) % TABS.len();
                self.status.clear();
                true
            }
            b'1'..=b'5' => {
                self.tab = (k - b'1') as usize;
                self.status.clear();
                true
            }
            // **Measuring is a key and never a frame.** `outcome::probe`
            // dispatches fourteen applets and captures what each one prints.
            // On the compositor that is a stalled screen, and the watchdog
            // would correctly report the display as stopped.
            b'm' | b'M' => {
                self.matrix = crate::ai::outcome::probe();
                self.status = match &self.matrix {
                    Some(_) => String::new(),
                    None => String::from("the applet table would not answer"),
                };
                self.tab = 4;
                true
            }
            _ => false,
        }
    }

    fn press(&mut self, client: Rect, x: i32, y: i32) -> bool {
        let (tabs, _, _) = Self::layout(client);
        for (i, r) in Self::tab_rects(tabs).iter().enumerate() {
            if x >= r.x as i32
                && y >= r.y as i32
                && x < (r.x + r.w) as i32
                && y < (r.y + r.h) as i32
            {
                self.tab = i;
                self.status.clear();
                return true;
            }
        }
        false
    }

    fn wheel(&mut self, notches: i32) -> bool {
        if notches == 0 {
            return false;
        }
        let n = TABS.len() as i32;
        self.tab = (((self.tab as i32 + notches.signum()) % n + n) % n) as usize;
        true
    }
}

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

const TABS: [&str; 6] = ["Learn", "Net", "Route", "Council", "Ledger", "Outcome"];

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
    /// When the current tab was raised, in milliseconds of uptime. The reveal
    /// runs from here rather than from the fit, because the fit happens at
    /// boot and would always be over by the time anybody opened the window.
    shown_ms: u64,
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
        let mut o = Self { tab, shown_ms: now_ms(), matrix: None, status: String::new() };
        // **Opening the panel takes the reading.** It was behind the N key, on
        // the reasoning that a forward pass is too expensive for a frame --
        // which is true, and has nothing to do with opening a window. Asking
        // somebody to press a key before the panel shows anything is a worse
        // version of the thing this whole window was rebuilt to stop doing,
        // and driving that key turned out to be impossible anyway: the shell
        // re-focuses the terminal after every command, so an injected key
        // never reaches the app that was just raised.
        if o.tab == 1 {
            o.capture();
        }
        o
    }

    pub fn preferred() -> (u32, u32) {
        (760, 520)
    }

    /// One taped forward pass, kept for the network panel.
    ///
    /// Never from `draw_in`. This allocates a tape and runs the whole stack,
    /// which is not something a paint may do on the task that owns the screen.
    fn capture(&mut self) {
        self.status = if crate::ai::harness::capture_stack(
            "list the files in /ai and tell me what changed",
        ) {
            String::new()
        } else {
            String::from("no model, or it is a hybrid the tape cannot follow")
        };
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


    // --- Learn -----------------------------------------------------------

    /// What one cold fit bought, which is the one number here that plainly
    /// goes up.
    ///
    /// **Nothing in this panel is a proxy for progress.** The probe starts
    /// with no weights at all and 23 classes, so a guess is right 4% of the
    /// time; a closed-form ridge solve over the corpus takes it to the high
    /// nineties on what it saw and the mid seventies on what it did not, in
    /// 1,115 ms measured. Held out is the bar that matters and it is drawn
    /// tallest for that reason -- seen is drawn beside it because the gap
    /// between the two is the only honest way to show there is nothing being
    /// hidden by memorisation.
    ///
    /// The per-class strip underneath is the shape the average conceals. Some
    /// applets are learned outright and some are never learned at all, and
    /// `repair.rs` records why from the other side: a name carries probability
    /// mass that has nothing to do with what the applet does.
    fn draw_learn(&self, fb: &Framebuffer, r: Rect) -> String {
        let lh = theme::text_h();
        let Some(f) = trace::last_fit() else {
            theme::text_over(fb, r.x + 8, r.y + 8,
                &clip("no fit recorded -- the router fits itself at boot",
                      fits(r.w.saturating_sub(16))),
                theme::SCREEN_TEXT);
            return String::from("nothing fitted this boot");
        };

        // The reveal runs from when this panel was last brought up, so the
        // numbers arrive rather than being there already. It is presentation
        // and nothing else: what it uncovers was measured once, at boot, and
        // does not change while it is being drawn.
        let now = crate::dev::lapic::ticks() as u64 * 1000 / crate::TIMER_HZ as u64;
        let since = now.saturating_sub(self.shown_ms);
        let grow = (since as f32 / 900.0).clamp(0.0, 1.0);

        let chance = 100 / f.classes.max(1);
        let held = if f.held_n > 0 { f.held_ok * 100 / f.held_n } else { 0 };
        let seen = if f.seen_n > 0 { f.seen_ok * 100 / f.seen_n } else { 0 };

        let bars: [(&str, usize, Color); 3] = [
            ("chance", chance, Color::new(0x4A, 0x5A, 0x66)),
            ("held out", held, theme::APERTURE),
            ("seen", seen, Color::new(0x6C, 0xC2, 0x8A)),
        ];
        let bh = (lh + 10).min(30);
        let label_w = theme::text_w(9);
        let bw = r.w.saturating_sub(label_w + 70);
        for (i, (name, v, col)) in bars.iter().enumerate() {
            let y = r.y + 8 + i as u32 * (bh + 8);
            theme::text_over(fb, r.x + 6, y + 3, name,
                             if i == 1 { theme::APERTURE } else { theme::SCREEN_TEXT });
            fb.rect(r.x + 6 + label_w, y, bw, bh, theme::SCREEN);
            let w = ((*v as f32 / 100.0) * bw as f32 * grow) as u32;
            fb.rect(r.x + 6 + label_w, y, w.max(1), bh, *col);
            theme::text_over(fb, r.x + 6 + label_w + bw + 8, y + 3,
                             &format!("{}%", (*v as f32 * grow) as usize),
                             theme::SCREEN_TEXT);
        }

        // Per class, over the held-out tail only. A cell that never lights is
        // an applet the router did not learn, which is worth seeing.
        let top = r.y + 14 + 3 * (bh + 8);
        theme::text_over(fb, r.x + 6, top,
            &clip("per applet, held out only", fits(r.w.saturating_sub(12))),
            theme::SHADOW);
        let n = f.per_class.len().min(24);
        if n > 0 && top + lh + 26 < r.y + r.h {
            let cols = 12u32;
            let cw = (r.w.saturating_sub(12)) / cols;
            let chh = ((r.y + r.h).saturating_sub(top + lh + 6) / 2).min(26).max(8);
            for (i, (_name, ok, tot)) in f.per_class.iter().take(n).enumerate() {
                let cx = r.x + 6 + (i as u32 % cols) * cw;
                let cy = top + lh + 4 + (i as u32 / cols) * (chh + 4);
                let lit = if *tot > 0 { *ok as f32 / *tot as f32 } else { 0.0 };
                // Reveal left to right, so the strip fills rather than
                // appearing.
                let vis = ((i as f32 / n as f32) < grow) as u32 as f32;
                let k = (40.0 + lit * 200.0 * vis) as u8;
                let col = if *tot == 0 {
                    Color::new(0x1A, 0x20, 0x26)
                } else {
                    Color::new((k as u16 * 60 / 255) as u8, k, (k as u16 * 90 / 255) as u8)
                };
                fb.rect(cx, cy, cw.saturating_sub(3), chh, col);
            }
        }

        format!("{} parameters, closed form, {} ms -- chance is {}%",
                f.params, f.ms, chance)
    }

    // --- Net -------------------------------------------------------------

    /// The transformer stack, at the depth it really has.
    ///
    /// **Every node is a real activation.** `Tape` keeps the residual stream
    /// entering each layer of a taped forward pass, so a column here is what
    /// the model was carrying at that depth and a node is a contiguous slice
    /// of it. Thirty layers on SmolLM2, twenty-eight on the 0.6B, five hundred
    /// and seventy-six values wide, summed into fourteen bins because no
    /// screen shows five hundred nodes.
    ///
    /// Normalised per layer, on purpose. A residual stream grows as it climbs,
    /// so one global scale draws the top bright and the first twenty black,
    /// which is a picture of layer norms rather than of the prompt.
    ///
    /// The edges are co-activation: bright where both ends are carrying, dim
    /// where either is quiet. A transformer layer is fully connected through
    /// its matrices so every one of them exists, and thinning to the strongest
    /// few per node is what stops it being a grey rectangle.
    fn draw_net(&self, fb: &Framebuffer, r: Rect) -> String {
        let lh = theme::text_h();
        let Some(st) = trace::last_stack() else {
            theme::text_over(fb, r.x + 8, r.y + 8,
                &clip("press N to run a forward pass and capture the stack",
                      fits(r.w.saturating_sub(16))),
                theme::SCREEN_TEXT);
            return String::from("nothing captured yet");
        };
        if st.layers == 0 || st.bins == 0 {
            return String::from("the capture is empty");
        }

        let ms = now_ms();
        // One pulse travelling up through the depth, on a loop.
        let phase = (ms % 2600) as f32 / 2600.0;

        let top = r.y + 4 + lh;
        let bot = r.y + r.h.saturating_sub(4);
        let h = bot.saturating_sub(top).max(1);
        let lw = (r.w.saturating_sub(16)) as f32 / (st.layers as f32 - 1.0).max(1.0);
        let x_of = |l: usize| r.x + 8 + (l as f32 * lw) as u32;
        let y_of = |b: usize| top + (b as u32 * h) / st.bins.max(1) as u32 + h / (st.bins as u32 * 2).max(1);

        theme::text_over(fb, r.x + 8, r.y + 1,
            &clip(&format!("{} layers x {} wide, {} heads", st.layers - 1, st.dim, st.heads),
                  fits(r.w.saturating_sub(16))),
            theme::SHADOW);

        // Edges first, so nodes sit on top of them.
        const PER_NODE: usize = 5;
        for l in 0..st.layers.saturating_sub(1) {
            let d = (l as f32 / (st.layers - 1) as f32 - phase).abs();
            let lit = (1.0 - d * 2.2).clamp(0.42, 1.0) + (1.0 - d * 6.0).max(0.0) * 0.8;
            for b in 0..st.bins {
                let a0 = st.act[l * st.bins + b];
                if a0 < 0.08 {
                    continue;
                }
                let mut rank: Vec<(usize, f32)> = (0..st.bins)
                    .map(|c| (c, a0 * st.act[(l + 1) * st.bins + c]))
                    .collect();
                rank.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(core::cmp::Ordering::Equal));
                rank.truncate(PER_NODE);
                for (c, w) in rank {
                    let k = (w.clamp(0.0, 1.0) * lit * 235.0).min(255.0) as u8;
                    if k < 10 {
                        continue;
                    }
                    fb.line(
                        x_of(l) as i32, y_of(b) as i32,
                        x_of(l + 1) as i32, y_of(c) as i32,
                        Color::new(k / 5, (k as u16 * 3 / 5) as u8, k),
                    );
                }
            }
        }

        for l in 0..st.layers {
            let d = (l as f32 / (st.layers - 1).max(1) as f32 - phase).abs();
            let lit = (1.0 - d * 2.2).clamp(0.6, 1.0) + (1.0 - d * 6.0).max(0.0) * 0.9;
            for b in 0..st.bins {
                let a = st.act[l * st.bins + b].clamp(0.0, 1.0);
                let sz = 3 + (a * 5.0) as u32;
                let k = (70.0 + a * 185.0 * lit).min(255.0) as u8;
                // Cyan through to white as a slice carries more, so the
                // strongest parts of the stream read as hot rather than as
                // merely bigger.
                let col = if a > 0.66 {
                    let w = ((a - 0.66) / 0.34 * 255.0).min(255.0) as u8;
                    Color::new(w.max((k as u16 * 70 / 255) as u8), k, 255)
                } else {
                    Color::new((k as u16 * 45 / 255) as u8, (k as u16 * 190 / 255) as u8, k)
                };
                let (x, y) = (x_of(l), y_of(b));
                fb.rect(x.saturating_sub(sz / 2), y.saturating_sub(sz / 2), sz, sz, col);
            }
        }

        format!("residual stream, binned {} ways, normalised per layer", st.bins)
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

fn now_ms() -> u64 {
    crate::dev::lapic::ticks() as u64 * 1000 / crate::TIMER_HZ as u64
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
        // has to keep saying so, and a frame is 2,143 us measured, about 7% of
        // a core at sixty a second.
        //
        // Gated on the tab rather than on focus, and that is a correction. The
        // shell calls `focus_terminal` after *every* command, so a window
        // raised by `win keys alttab` loses focus again before the next line
        // is read -- which made the panel freeze the moment anything was typed,
        // including the command that was meant to be driving it. The tab is
        // the real opt-in: nothing draws this unless somebody chose it, and a
        // minimised window is never drawn at all.
        if self.tab <= 1 {
            super::render::invalidate();
        }

        theme::well(fb, body, theme::SCREEN);
        let inner = body.shrink(4);
        let note = match self.tab {
            0 => self.draw_learn(fb, inner),
            1 => self.draw_net(fb, inner),
            2 => self.draw_route(fb, inner),
            3 => self.draw_council(fb, inner),
            4 => self.draw_ledger(fb, inner),
            _ => self.draw_outcome(fb, inner),
        };
        let line = if self.status.is_empty() { note } else { self.status.clone() };
        theme::text_over(fb, foot.x + 2, foot.y, &clip(&line, fits(foot.w)), theme::SHADOW);
    }

    fn key(&mut self, k: u8) -> bool {
        match k {
            b'\t' => {
                self.tab = (self.tab + 1) % TABS.len();
                self.shown_ms = now_ms();
                self.status.clear();
                true
            }
            b'1'..=b'6' => {
                self.tab = (k - b'1') as usize;
                self.shown_ms = now_ms();
                self.status.clear();
                true
            }
            // **Measuring is a key and never a frame.** `outcome::probe`
            // dispatches fourteen applets and captures what each one prints.
            // On the compositor that is a stalled screen, and the watchdog
            // would correctly report the display as stopped.
            // A forward pass on the shell's task, never on a frame.
            b'n' | b'N' => {
                self.capture();
                self.tab = 1;
                self.shown_ms = now_ms();
                true
            }
            b'm' | b'M' => {
                self.matrix = crate::ai::outcome::probe();
                self.status = match &self.matrix {
                    Some(_) => String::new(),
                    None => String::from("the applet table would not answer"),
                };
                self.tab = 5;
                self.shown_ms = now_ms();
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
                self.shown_ms = now_ms();
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
        self.shown_ms = now_ms();
        true
    }
}

//! The miner's screen: a terminal, and no desktop at all.
//!
//! A miner image has no windows to manage, no model to talk to and nobody in
//! front of it except to read a number. So it does not run a desktop -- and that
//! is a speed decision before it is a look one. `video bench` measures
//! `desk::draw + present` at 1.6-2.1 ms a frame with the clock task repainting at
//! 10 Hz, and every one of those cycles comes off the hash loop.
//!
//! This is what replaces it: one screen of text, redrawn in place a couple of
//! times a second, built out of the box-drawing and block glyphs `gfx::font`
//! already carries.
//!
//! ### It writes to the console and deliberately not through `kprintln!`
//!
//! `kprint!` goes to the console *and* the serial port. A full-screen redraw
//! twice a second down a 115,200-baud line is about a third of the link spent on
//! frames nobody is reading, and it would bury the one line somebody does want
//! -- `drive.py` reads that port, and a transcript of forty repaints of the same
//! dashboard is not a transcript.
//!
//! So the frame is assembled into one buffer and handed to `Console::write_bytes`
//! inside a single `console::with`, which paints and does not echo. The serial
//! log keeps carrying what the miner *says* -- connected, job installed, share
//! accepted -- and none of what it draws.
//!
//! ### Nothing here reads state it does not own
//!
//! Every figure comes from an atomic or a `Spin` the miner already publishes, so
//! drawing cannot perturb mining and a locked slot cannot stall a frame. The
//! whole frame is gathered before a byte is painted, because sampling per panel
//! would put the hashrate a frame ahead of the share count -- the sort of
//! inconsistency nobody notices and nobody trusts afterwards.
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use crate::gfx::console;
use crate::sync::Spin;

/// Samples across the graph. At two a second this is a little over a minute,
/// which is the window in which a miner notices something has gone wrong.
const HIST: usize = 120;
/// Rows of graph. Each row carries two levels -- full block and half block -- so
/// this is eight levels of vertical resolution out of four lines of text.
const GRAPH_ROWS: usize = 4;

static HISTORY: Spin<[u32; HIST]> = Spin::new([0; HIST]);
static HEAD: AtomicUsize = AtomicUsize::new(0);
/// Redraws, so the cost of this screen can be divided out of a hashrate.
pub static FRAMES: AtomicU32 = AtomicU32::new(0);

fn sample(hs: u32) {
    let i = HEAD.fetch_add(1, Ordering::Relaxed) % HIST;
    HISTORY.lock_irq()[i] = hs;
}

/// Thousands separators. A nine-digit hashrate is unreadable without them and
/// there is no locale here to ask.
fn grouped(mut n: u64) -> String {
    if n == 0 {
        return String::from("0");
    }
    let mut parts = [0u16; 7];
    let mut used = 0;
    while n > 0 && used < parts.len() {
        parts[used] = (n % 1000) as u16;
        n /= 1000;
        used += 1;
    }
    let mut s = format!("{}", parts[used - 1]);
    for i in (0..used - 1).rev() {
        s.push(',');
        s.push_str(&format!("{:03}", parts[i]));
    }
    s
}

fn hms(secs: u64) -> String {
    format!("{:02}:{:02}:{:02}", secs / 3600, (secs / 60) % 60, secs % 60)
}

/// Characters, not bytes. `&s[..n]` on a multi-byte string panics rather than
/// shortening, which is the trap `theme::head_chars` exists for one layer up.
fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return String::from(s);
    }
    s.chars().take(n).collect()
}

fn rule(w: usize, title: &str) -> String {
    // `├ TITLE ─────┤`, which is one line rather than three and reads as a
    // section without costing a blank row either side of it.
    let mut s = String::from("├─ ");
    s.push_str(title);
    s.push(' ');
    let used = 4 + title.chars().count();
    for _ in used..w.saturating_sub(1) {
        s.push('─');
    }
    s.push('┤');
    s.push('\n');
    s
}

/// One frame, as coloured segments. Built whole before anything is painted.
fn frame() -> Vec<(u8, String)> {
    let mut out: Vec<(u8, String)> = Vec::new();
    let w = console::cols().clamp(60, 120);

    // ---- every number, of one instant ----------------------------------
    let hashes = super::client::HASHES.load(Ordering::Relaxed);
    let ms = super::client::hash_ms();
    let hs = if ms > 0 { (hashes * 1000 / ms) as u32 } else { 0 };
    let found = super::client::FOUND.load(Ordering::Relaxed);
    let acc = super::client::ACCEPTED.load(Ordering::Relaxed);
    let rej = super::client::REJECTED.load(Ordering::Relaxed);
    let best = super::client::BEST.load(Ordering::Relaxed);
    let phase = super::client::phase();
    let live = matches!(phase, super::client::Phase::Live);
    let (host, worker) = {
        let g = super::client::CONFIG.lock_irq();
        match g.as_ref() {
            Some(c) => (format!("{}:{}", c.host, c.port), c.user.clone()),
            None => (String::from("unset"), String::from("unset")),
        }
    };
    let up = crate::dev::lapic::ticks() / crate::TIMER_HZ as u64;
    sample(hs);

    let mut push = |c: u8, s: String| out.push((c, s));

    // ---- the bar -------------------------------------------------------
    let mut top = String::from("╔");
    for _ in 1..w - 1 {
        top.push('═');
    }
    top.push('╗');
    top.push('\n');
    push(console::LTCYAN, top);

    let left = " GLaDOS MINER";
    let right = format!("{}  {} ", host, phase.name().to_uppercase());
    let gap = w.saturating_sub(2 + left.chars().count() + right.chars().count());
    push(console::LTCYAN, String::from("║"));
    push(console::YELLOW, String::from(left));
    push(console::LTGRAY, " ".repeat(gap));
    push(if live { console::LTGREEN } else { console::LTRED }, right);
    push(console::LTCYAN, String::from("║\n"));

    let mut bot = String::from("╚");
    for _ in 1..w - 1 {
        bot.push('═');
    }
    bot.push('╝');
    bot.push('\n');
    push(console::LTCYAN, bot);

    // ---- the number ----------------------------------------------------
    let rate = format!("{} H/s", grouped(hs as u64));
    let pad = (w.saturating_sub(rate.chars().count())) / 2;
    push(console::LTGRAY, String::from("\n"));
    push(console::WHITE, format!("{}{}\n", " ".repeat(pad), rate));

    // ---- the graph -----------------------------------------------------
    //
    // Two levels a row out of `█` and `▄`, so four lines give eight steps. The
    // eighth-height glyphs U+2581..U+2587 would give more and this font does not
    // carry them -- `gfx::font` says what it has rather than substituting
    // something close, which is why this counts in halves instead.
    {
        let hist = HISTORY.lock_irq();
        let head = HEAD.load(Ordering::Relaxed);
        let cols = (w - 4).min(HIST);
        // **Scaled to the window, and to its floor as well as its peak.** A
        // decaying all-time peak was the first version and it drew a solid
        // block: a hashrate that barely moves sits at 99% of its own maximum
        // forever, so every column is full and the graph says nothing. Scaling
        // between the lowest and highest of what is *on screen* turns the same
        // data into the shape of the last minute, which is the only thing a
        // reader wants from it.
        //
        // A flat run would divide by zero, so an empty range draws at half
        // height -- honest, since a flat line is exactly what it is.
        let mut lo = u64::MAX;
        let mut hi = 0u64;
        for i in 0..cols {
            let v = hist[(head + HIST - cols + i) % HIST] as u64;
            if v == 0 {
                continue;
            }
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let flat = hi == 0 || hi <= lo;
        let span = if flat { 1 } else { hi - lo };
        let levels = (GRAPH_ROWS * 2) as u64;
        for row in 0..GRAPH_ROWS {
            // Top row first, so the graph is drawn the way it is read.
            let from_bottom = (GRAPH_ROWS - 1 - row) as u64;
            let mut line = String::from("  ");
            for i in 0..cols {
                // Oldest at the left. `head` is one past the newest, so the
                // column `i` from the left is `head + i`, which is also why the
                // graph does not appear to jump when the ring wraps.
                let v = hist[(head + HIST - cols + i) % HIST] as u64;
                let steps = if v == 0 {
                    0
                } else if flat {
                    levels / 2
                } else {
                    ((v - lo) * (levels - 1)) / span + 1
                };
                let lo = from_bottom * 2;
                line.push(if steps >= lo + 2 {
                    '\u{2588}'
                } else if steps == lo + 1 {
                    '\u{2584}'
                } else {
                    ' '
                });
            }
            let c = if from_bottom >= 2 { console::LTCYAN } else { console::WHITE };
            push(c, format!("{}\n", line));
        }
    }

    // ---- the facts -----------------------------------------------------
    push(console::LTCYAN, rule(w, "SHARES"));
    let ok = if rej == 0 { console::LTGREEN } else { console::YELLOW };
    push(
        ok,
        format!(
            "  {} found   {} accepted   {} rejected   best {} bits\n",
            grouped(found),
            grouped(acc),
            grouped(rej),
            best
        ),
    );
    push(
        console::LTGRAY,
        format!("  worker {}   up {}\n", trunc(&worker, 44), hms(up)),
    );

    // ---- what it is working on -----------------------------------------
    push(console::LTCYAN, rule(w, "COINS"));
    push(
        console::LTGRAY,
        String::from("  slot  coin         algorithm             slices  rate\n"),
    );
    for i in 0..super::work::MAX_COINS {
        let g = super::work::coin(i);
        let Some(c) = g.as_ref() else {
            drop(g);
            continue;
        };
        let (label, detail, has_job) = (c.label.clone(), c.algo.detail(), c.template.is_some());
        drop(g);
        let on = super::work::slices_on(i);
        let (sh, sms, _) = super::work::rate(i);
        let srate = if !has_job {
            String::from("no job yet")
        } else if on == 0 {
            String::from("no slice on it")
        } else if sms == 0 {
            String::from("--")
        } else {
            format!("{} H/s", grouped(sh * 1000 / sms))
        };
        push(
            if on > 0 { console::WHITE } else { console::LTGRAY },
            format!(
                "  {:<5} {:<12} {:<21} {:<7} {}\n",
                i,
                trunc(&label, 12),
                trunc(&detail, 21),
                on,
                srate
            ),
        );
    }

    // ---- the last thing that happened ----------------------------------
    push(console::LTCYAN, rule(w, "LOG"));
    let j = super::client::journal();
    let room = 6usize;
    let start = j.len().saturating_sub(room);
    for line in &j[start..] {
        push(console::LTGRAY, format!("  {}\n", trunc(line, w - 4)));
    }

    out
}

/// Paint one frame. `false` when there is no console to paint on.
pub fn draw() -> bool {
    if !console::is_ready() {
        return false;
    }
    let f = frame();
    // **Two short holds rather than one long one, and neither paints per cell.**
    //
    // Three versions of this, and the two that failed are worth keeping because
    // they failed in opposite directions.
    //
    // Painting through `kprint!` works and *tears*: `draw_cell` blits and
    // `flush_rect`s every cell as it is written, so a fifteen-hundred-cell frame
    // is fifteen hundred of each and a screenshot catches the header drawn and
    // everything under it still blank. A person watching barely sees it; a
    // photograph shows nothing else.
    //
    // Doing the whole thing inside one `console::with` -- suppress painting,
    // fill the grid, `redraw_all` -- draws perfectly and hung the machine.
    // Serial stopped dead after the `[ai]` section, `[miner]` never printed and
    // no prompt ever arrived, while the screen ticked happily up to three
    // minutes of uptime. A machine that draws and cannot talk.
    //
    // So: one short hold to fill the grid with painting suppressed, released,
    // then `console::redraw()` for a single repaint. `CLAUDE.md` measures that
    // at 592 us after the blank-cell optimisation, which is nothing once a
    // second -- the cost was never the repaint, it was holding the console
    // across everything else.
    console::with(|c| {
        c.set_visible(false);
        c.clear();
        for (colour, text) in &f {
            c.set_color(*colour);
            c.write_bytes(text.as_bytes());
        }
        c.set_color(console::LTGRAY);
        c.set_visible(true);
    });
    console::redraw();
    FRAMES.fetch_add(1, Ordering::Relaxed);
    true
}

/// Redraw for as long as the machine runs.
///
/// Once a second. Faster buys nothing -- the hashrate is averaged over the whole
/// run and the share counters move every few seconds -- and costs the hash loop
/// a frame it did not need to paint.
pub fn task() {
    // **Wipe the screen once, because nothing else will.** The desktop paints a
    // taskbar, a wallpaper and a status strip during boot, before the miner has
    // applied and taken the display -- and the frames below never touch those
    // pixels: `clear()` skips its fill while the console is invisible, which is
    // exactly the state the fast path puts it in. The leftovers showed as a
    // vertical rule down the old terminal window's left edge and a strip along
    // the bottom reading "no model", sitting under a dashboard that had no idea
    // they were there.
    if let Some(fb) = crate::gfx::primary() {
        fb.fill(crate::gfx::Color::new(0, 0, 0));
    }
    loop {
        draw();
        let until = crate::dev::lapic::ticks() + crate::TIMER_HZ as u64;
        while crate::dev::lapic::ticks() < until {
            crate::task::yield_now();
        }
    }
}

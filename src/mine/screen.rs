//! The miner's screen: the product, for most people who ever boot this.
//!
//! A miner image has no desktop, no model and nobody in front of it except to
//! glance at it -- and to photograph it, which is the other half of the job.
//! So this is two full-screen views drawn straight into the compositor's back
//! buffer, not a terminal:
//!
//! - **Asking**, on first boot: the mark, "MINE $GLaDOS", and one field for the
//!   wallet address that gets paid. The shell's own `glados>` prompt used to be
//!   the loudest thing on this screen and the address question two grey lines
//!   at the foot of it; a first-time user answered the wrong prompt.
//! - **Mining**: "YOU ARE MINING $GLaDOS", the hashrate large enough to read
//!   across a room, a minute of history, shares, uptime, and where the tokens
//!   go -- over the same sky and sun as the desktop wall, so a screenshot of a
//!   running miner is also the advertisement for it.
//!
//! **What is not on it, and why that is presentation rather than concealment.**
//! The algorithm and the upstream chain are not shown: a miner is paid in
//! $GLaDOS whatever the pool's upstream is hashing, and the upstream changes
//! with profitability, so naming it here would be naming something that is not
//! stable and not what anybody is paid in. Nothing is hidden: `mine` in the
//! shell reports the algorithm and every slot, and the header a share hashes is
//! the real chain's -- anybody can look it up.
//!
//! ### It paints through the compositor, and the console is kept dark
//!
//! The previous screen was text written into the console's grid, and it
//! inherited everything the console is: the terminal window's edge (a line a
//! hundred and sixty pixels in), its status strip ("no model | mind on"), its
//! caret, and rows packed so tight headings touched the lines around them. Now
//! the console is made invisible for the life of the image -- it still takes
//! every line and serial still carries them, it simply paints nothing -- and a
//! frame is drawn whole into `compose::target()` and shown by `present()`,
//! which writes only the pixels that changed. No tearing, no flicker, and a
//! photograph never catches half a frame.
//!
//! A fault still reaches the screen: `splash::abandon` makes the console
//! visible again before the reporter prints.
//!
//! ### Nothing here reads state it does not own
//!
//! Every figure comes from an atomic or a `Spin` the miner already publishes,
//! so drawing cannot perturb mining and a locked slot cannot stall a frame. The
//! frame is gathered before a pixel is drawn.
//!
//! ### What it costs
//!
//! One frame a second, from the miner's socket loop (see `tick`): a gradient,
//! the mark, a few hundred glyphs and a diff. A couple of milliseconds a second
//! against the hash loop, where the old console frame was about 0.6 ms -- paid
//! once a second, it is a tenth of a per cent.

use alloc::format;
use alloc::string::String;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use crate::gfx::splash::{aperture_with, Cut, Face};
use crate::gfx::{compose, console, theme, Color, Framebuffer, Shape};
use crate::sync::Spin;

/// Samples across the graph, one a second: two minutes of history.
const HIST: usize = 120;

static HISTORY: Spin<[u32; HIST]> = Spin::new([0; HIST]);
static HEAD: AtomicUsize = AtomicUsize::new(0);
/// When the next frame is due, in `lapic::ticks`.
static NEXT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
/// Redraws, so the cost of this screen can be divided out of a hashrate. Also
/// the spinner's clock.
pub static FRAMES: AtomicU32 = AtomicU32::new(0);

/// What frames cost, in TSC cycles: composing into the back buffer and
/// presenting it are timed apart, because they scale differently -- drawing
/// with what is on the frame, presenting with the pixels on the screen.
static DRAW_CYC: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static PRESENT_CYC: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static WORST_CYC: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// What has been typed into the address field, mirrored from the shell's line
/// editor on every keystroke (see `typed`).
static TYPED: Spin<String> = Spin::new(String::new());
/// Why the last address was refused, until the next one is typed.
static REFUSAL: Spin<Option<&'static str>> = Spin::new(None);

// ---- the look ------------------------------------------------------------
//
// The desktop wall's sky, darkened so white type reads on every part of it,
// with the same warm foot. `Cut::Sky` carves the mark's blades out of this same
// table, so the sky shows through the sun rather than a flat patch of colour.
const SKY: [(u8, Color); 5] = [
    (0, Color::new(0x04, 0x0E, 0x18)),
    (110, Color::new(0x08, 0x26, 0x38)),
    (190, Color::new(0x0D, 0x3C, 0x52)),
    (236, Color::new(0x2C, 0x2E, 0x26)),
    (255, Color::new(0x5A, 0x3A, 0x14)),
];
const WHITE: Color = Color::new(0xFF, 0xFF, 0xFF);
const INK: Color = Color::new(0xE6, 0xF1, 0xF6);
const DIM: Color = Color::new(0x86, 0xA8, 0xB8);
const TEAL: Color = Color::new(0x8F, 0xDC, 0xF0);
const GOLD: Color = theme::APERTURE;
const GREEN: Color = Color::new(0x4A, 0xDE, 0x80);
const AMBER: Color = Color::new(0xFB, 0xBF, 0x24);
const RED: Color = Color::new(0xF8, 0x71, 0x71);
const SHADE: Color = Color::new(0x00, 0x00, 0x00);

/// Where to get it, on every frame, because every frame is a screenshot.
const SITE: &str = "glados.aperture.institute";

/// Twelve points on a unit circle (x1000), clockwise from the top. The spinner
/// around the mark steps one a frame; no trigonometry needed for twelve.
const DIAL: [(i32, i32); 12] = [
    (0, -1000), (500, -866), (866, -500), (1000, 0), (866, 500), (500, 866),
    (0, 1000), (-500, 866), (-866, 500), (-1000, 0), (-866, -500), (-500, -866),
];

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

/// A hashrate with its unit: `77 H/s`, `12.4 kH/s`, `3.07 MH/s`.
fn rate(hs: u64) -> String {
    match hs {
        0..=9_999 => format!("{} H/s", grouped(hs)),
        10_000..=9_999_999 => format!("{}.{} kH/s", hs / 1000, (hs % 1000) / 100),
        _ => format!("{}.{:02} MH/s", hs / 1_000_000, (hs % 1_000_000) / 10_000),
    }
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
    let mut t: String = s.chars().take(n.saturating_sub(3)).collect();
    t.push_str("...");
    t
}

/// `0x6ef4...527d`: enough to recognise your own wallet, short enough to fit.
fn short_addr(a: &str) -> String {
    let n = a.chars().count();
    if n <= 16 {
        return String::from(a);
    }
    let head: String = a.chars().take(6).collect();
    let tail: String = a.chars().skip(n - 4).collect();
    format!("{head}...{tail}")
}

// ---- drawing helpers -------------------------------------------------------

fn text_w(s: &str, scale: u32) -> u32 {
    s.chars().count() as u32 * crate::gfx::font::GLYPH_W * scale
}

/// Text with a soft drop shadow, which is what keeps white type legible where
/// the sky turns gold.
fn say(fb: &Framebuffer, x: u32, y: u32, s: &str, c: Color, scale: u32) {
    let d = (scale / 2).max(1);
    fb.draw_text_over(x + d, y + d, s, SHADE, scale);
    fb.draw_text_over(x, y, s, c, scale);
}

fn say_centred(fb: &Framebuffer, cx: u32, y: u32, s: &str, c: Color, scale: u32) {
    say(fb, cx.saturating_sub(text_w(s, scale) / 2), y, s, c, scale);
}

/// The largest scale at which `s` fits in `w`, capped at `max`.
fn fit(s: &str, w: u32, max: u32) -> u32 {
    let per = (s.chars().count() as u32 * crate::gfx::font::GLYPH_W).max(1);
    (w / per).clamp(1, max)
}

/// A rounded pane of glass: lighter at the top, thinner at the foot.
fn pane(fb: &Framebuffer, x: u32, y: u32, w: u32, h: u32, r: u32, edge: Option<Color>) {
    let shape = Shape::round_rect(w, h, r);
    crate::gfx::with_shape(&shape, x as i32, y as i32, || {
        fb.glass(x, y, w, h, WHITE, &[(0, 34), (255, 14)]);
        if let Some(c) = edge {
            fb.tint_rect(x, y, w, 2, c, 200);
            fb.tint_rect(x, y + h - 2, w, 2, c, 200);
            fb.tint_rect(x, y, 2, h, c, 200);
            fb.tint_rect(x + w - 2, y, 2, h, c, 200);
        }
    });
}

/// The sky, full screen.
fn sky(fb: &Framebuffer) {
    fb.vgrad(0, 0, fb.width(), fb.height(), &SKY);
}

/// The mark as a sun, with the sky showing through its cuts.
fn sun(fb: &Framebuffer, cx: i32, cy: i32, r: i32) {
    let h = fb.height();
    aperture_with(fb, cx, cy, r, Face::Ramp(&theme::SUN), Cut::Sky { stops: &SKY, top: 0, height: h });
}

/// The unit everything is sized in: 1 at 640x400, 2 at 1280x800 and at
/// 1920x1080, 3 at 2560x1440. Integer, because glyphs scale by whole pixels.
fn unit(fb: &Framebuffer) -> u32 {
    (fb.width() / 640).min(fb.height() / 400).max(1)
}

/// The top bar: a small mark and the name on the left, a status pill on the
/// right.
fn bar(fb: &Framebuffer, k: u32, status: Option<(&str, Color)>) {
    let m = fb.width() / 40;
    let r = (7 * k) as i32;
    let cy = m as i32 / 2 + r + 4;
    sun(fb, m as i32 + r, cy, r);
    let ts = k + 1;
    let ty = (cy as u32).saturating_sub(4 * ts);
    say(fb, m + 2 * r as u32 + 6 * k, ty, "GLaDOS", WHITE, ts);
    say(fb, m + 2 * r as u32 + 6 * k + text_w("GLaDOS ", ts), ty + 4 * ts - 4 * k, "MINER", GOLD, k);

    if let Some((word, c)) = status {
        let pw = text_w(word, k) + 16 * k + 14 * k;
        let ph = 12 * k + 8;
        let px = fb.width() - m - pw;
        let py = cy as u32 - ph / 2;
        pane(fb, px, py, pw, ph, ph / 2, None);
        fb.fill_circle((px + 10 * k) as i32, (py + ph / 2) as i32, (3 * k) as i32, c);
        say(fb, px + 18 * k, py + ph / 2 - 4 * k, word, c, k);
    }
}

/// The foot: what this is and where to get it, on every frame.
fn foot(fb: &Framebuffer, k: u32) {
    let h = 18 * k + 12;
    let y = fb.height() - h;
    fb.tint_rect(0, y, fb.width(), h, SHADE, 110);
    fb.tint_rect(0, y, fb.width(), 1, GOLD, 120);
    let lead = "Boot any PC from USB and mine $GLaDOS   ";
    let s = fit(&format!("{lead}{SITE}"), fb.width() * 9 / 10, k);
    let total = text_w(lead, s) + text_w(SITE, s);
    let x = (fb.width() - total) / 2;
    let ty = y + (h - 8 * s) / 2;
    say(fb, x, ty, lead, DIM, s);
    say(fb, x + text_w(lead, s), ty, SITE, GOLD, s);
}

// ---- the two views ---------------------------------------------------------

/// First boot: one question, asked as the only thing on the screen.
fn ask_frame(fb: &Framebuffer) {
    let (w, h) = (fb.width(), fb.height());
    let k = unit(fb);
    sky(fb);
    bar(fb, k, Some(("WAITING FOR ADDRESS", AMBER)));

    let cx = w / 2;
    sun(fb, cx as i32, (h * 29 / 100) as i32, (h * 13 / 100) as i32);

    let head = "MINE $GLaDOS";
    let hs = fit(head, w * 7 / 10, 9);
    let hy = h * 46 / 100;
    let hx = cx - text_w(head, hs) / 2;
    say(fb, hx, hy, "MINE ", WHITE, hs);
    say(fb, hx + text_w("MINE ", hs), hy, "$", GOLD, hs);
    say(fb, hx + text_w("MINE $", hs), hy, "GLaDOS", WHITE, hs);

    let sub = "Type the 0x wallet address that should receive your $GLaDOS";
    let ss = fit(sub, w * 9 / 10, k);
    let sy = hy + 8 * hs + 6 * k;
    say_centred(fb, cx, sy, sub, INK, ss);

    // The field: sized for a whole address at the largest scale that fits.
    let full = "0x0000000000000000000000000000000000000000";
    let fs = fit(full, w * 85 / 100 - 48, k + 1);
    let fw = text_w(full, fs) + 48;
    let fh = 8 * fs + 32;
    let fx = cx - fw / 2;
    let fy = sy + 8 * ss + 10 * k;
    pane(fb, fx, fy, fw, fh, 10, Some(GOLD));
    let typed = TYPED.lock_irq().clone();
    let tx = fx + 24;
    let ty = fy + 16;
    if typed.is_empty() {
        say(fb, tx, ty, "0x...", DIM, fs);
    } else {
        say(fb, tx, ty, &trunc(&typed, 42), WHITE, fs);
    }
    // The caret, blinking with the frame.
    if FRAMES.load(Ordering::Relaxed) % 2 == 0 {
        let cxp = tx + text_w(&trunc(&typed, 42), fs);
        fb.rect(cxp + 2, ty, 2 * fs, 8 * fs, GOLD);
    }

    let hint_y = fy + fh + 8 * k;
    match *REFUSAL.lock_irq() {
        Some(why) => say_centred(fb, cx, hint_y, &format!("{why} -- type it again"), RED, fit(why, w * 8 / 10, k)),
        None => say_centred(fb, cx, hint_y, "then press Enter. Nothing else to set up.", DIM, k),
    }
    foot(fb, k);
}

/// Everything the mining view shows, read at one instant.
struct Now {
    hs: u64,
    acc: u64,
    rej: u64,
    worker: String,
    up: u64,
    live: bool,
    phase: &'static str,
    last: Option<String>,
}

fn gather() -> Now {
    let hashes = super::client::HASHES.load(Ordering::Relaxed);
    let ms = super::client::hash_ms();
    let hs = if ms > 0 { hashes * 1000 / ms } else { 0 };
    let phase = super::client::phase();
    let worker = super::client::CONFIG
        .lock_irq()
        .as_ref()
        .map(|c| c.user.split('.').next().unwrap_or("").into())
        .unwrap_or_default();
    Now {
        hs,
        acc: super::client::ACCEPTED.load(Ordering::Relaxed),
        rej: super::client::REJECTED.load(Ordering::Relaxed),
        worker,
        up: crate::dev::lapic::ticks() / crate::TIMER_HZ as u64,
        live: matches!(phase, super::client::Phase::Live),
        phase: phase.name(),
        last: super::client::journal().last().cloned(),
    }
}

fn mine_frame(fb: &Framebuffer, n: &Now) {
    let (w, h) = (fb.width(), fb.height());
    let k = unit(fb);
    sky(fb);
    let refused = n.last.as_deref().is_some_and(|l| l.starts_with("refused by the pool"));
    let status = if n.live {
        ("MINING", GREEN)
    } else if refused {
        ("REFUSED", RED)
    } else {
        ("CONNECTING", AMBER)
    };
    bar(fb, k, Some(status));

    // ---- the sun, and a ring that turns while work is being done ----------
    let cx = (w * 26 / 100) as i32;
    let cy = (h * 47 / 100) as i32;
    let r = (h * 22 / 100) as i32;
    sun(fb, cx, cy, r);
    let ring = r + (9 * k) as i32;
    let at = FRAMES.load(Ordering::Relaxed) as usize % DIAL.len();
    for (i, (dx, dy)) in DIAL.iter().enumerate() {
        let (x, y) = (cx + dx * ring / 1000, cy + dy * ring / 1000);
        let behind = (at + DIAL.len() - i) % DIAL.len();
        let (c, rad) = if !n.live {
            (DIM, 2 * k as i32)
        } else if behind == 0 {
            (GOLD, 4 * k as i32)
        } else if behind < 3 {
            (TEAL, 3 * k as i32)
        } else {
            (DIM, 2 * k as i32)
        };
        fb.fill_circle(x, y, rad, c);
    }

    // ---- the right column --------------------------------------------------
    let x0 = w * 52 / 100;
    let rw = w - x0 - w / 20;
    let mut y = h * 15 / 100;

    say(fb, x0, y, if n.live { "YOU ARE MINING" } else { "GETTING READY TO MINE" }, TEAL, k);
    y += 8 * k + 6 * k;
    let hs_scale = fit("$GLaDOS", rw, 12);
    say(fb, x0, y, "$", GOLD, hs_scale);
    say(fb, x0 + text_w("$", hs_scale), y, "GLaDOS", WHITE, hs_scale);
    y += 8 * hs_scale + 10 * k;

    // The number, large enough to read across a room.
    let big = rate(n.hs);
    let bs = fit(&big, rw, 3 * k);
    say(fb, x0, y, &big, GOLD, bs);
    say(fb, x0 + text_w(&big, bs) + 6 * k, y + 8 * bs - 8 * k, "your hashrate", DIM, k);
    y += 8 * bs + 10 * k;

    // Two minutes of it, from zero, so a steady machine reads as steady.
    let gh = h * 12 / 100;
    pane(fb, x0, y, rw, gh, 8, None);
    {
        let hist = HISTORY.lock_irq();
        let head = HEAD.load(Ordering::Relaxed);
        let inner = rw - 16;
        let cols = HIST.min(inner as usize);
        let colw = (inner / cols as u32).max(1);
        let peak = hist.iter().copied().max().unwrap_or(0).max(1) as u64;
        let base = y + gh - 8;
        let room = (gh - 16) as u64;
        for i in 0..cols {
            let v = hist[(head + HIST - cols + i) % HIST] as u64;
            if v == 0 {
                continue;
            }
            let bh = ((v * room * 85) / (peak * 100)).max(1) as u32;
            let x = x0 + 8 + i as u32 * colw;
            fb.tint_rect(x, base - bh, colw, bh, TEAL, 70);
            fb.rect(x, base - bh, colw, 2, TEAL);
        }
    }
    y += gh + 10 * k;

    // Three facts, as cards.
    let gap = 8 * k;
    let cw = (rw - 2 * gap) / 3;
    let ch = 26 * k + 12;
    let cards: [(&str, String, Color); 3] = [
        ("SHARES", grouped(n.acc), if n.rej == 0 { WHITE } else { AMBER }),
        ("UPTIME", hms(n.up), WHITE),
        ("STATUS", String::from(if n.live { "LIVE" } else if refused { "REFUSED" } else { "WAIT" }), status.1),
    ];
    for (i, (label, value, c)) in cards.iter().enumerate() {
        let cxp = x0 + i as u32 * (cw + gap);
        pane(fb, cxp, y, cw, ch, 8, None);
        say(fb, cxp + 8 * k, y + 6 * k, label, DIM, k);
        let vs = fit(value, cw - 16 * k, 2 * k);
        say(fb, cxp + 8 * k, y + 6 * k + 8 * k + 4 * k, value, *c, vs);
    }
    y += ch + 10 * k;

    // Where it goes, and the last thing that happened.
    let room = (rw / (8 * k)) as usize;
    let paid = format!("paid to {}", short_addr(&n.worker));
    say(fb, x0, y, &paid, INK, k);
    // What the wallet is paid in, in the accent, beside where it goes.
    let what = format!(" in {}", super::reward::name(super::reward::current()));
    let used = paid.chars().count();
    if used < room {
        say(fb, x0 + used as u32 * 8 * k, y, &trunc(&what, room - used), GOLD, k);
    }
    y += 8 * k + 6 * k;
    if let Some(last) = &n.last {
        let c = if refused { RED } else { DIM };
        say(fb, x0, y, &trunc(last, room), c, k);
    } else {
        say(fb, x0, y, &format!("{}...", n.phase), DIM, k);
    }
    y += 8 * k + 14 * k;

    // Switching wallets: typed straight at this screen, shown as it is typed.
    let typing = TYPED.lock_irq().clone();
    if typing.is_empty() {
        say(fb, x0, y, &trunc("new wallet: type a 0x address", room), DIM, k);
        say(fb, x0, y + 8 * k + 6 * k, &trunc("new reward: glados, nvda, chips, os...", room), DIM, k);
    } else {
        let label = if typing.starts_with("0x") || typing.starts_with("0X") {
            format!("new wallet: {typing}")
        } else {
            match super::reward::code(&typing) {
                Some(c) => format!("new reward: {}", super::reward::name(c)),
                None => format!("new reward: {typing}"),
            }
        };
        say(fb, x0, y, &trunc(&label, room), WHITE, k);
        say(fb, x0, y + 8 * k + 6 * k, "press Enter to switch and restart", GOLD, k);
    }

    foot(fb, k);
}

// ---- taking the screen, and drawing it -----------------------------------

/// Take the display for the life of the image. Idempotent: called when the
/// address is asked for and again when mining starts.
///
/// `set_exclusive` makes the desktop's periodic painters stand down; the
/// console is reflowed to the full screen (so what it holds is sane if a fault
/// ever makes it visible) and then made invisible, so nothing it prints --
/// the shell's echo, its caret, a stray log line -- paints over the frame.
pub fn take() {
    let Some(fb) = crate::gfx::primary() else { return };
    crate::gfx::set_exclusive(true);
    compose::init();
    // **Every console, not the default one.** The miner takes the screen while
    // boot is still on the executive console, and the shell then starts on the
    // user's: hiding only the first left the shell's prompt echo, its caret and
    // its window's status strip painting over the frame -- an address typed at
    // first boot appeared twice, once in the field and once in a black strip
    // across the top.
    for ch in 0..console::NCONSOLE {
        console::with_ch(ch, |c| {
            c.reflow(0, 0, fb.width(), fb.height());
            c.set_visible(false);
        });
    }
}

fn paint(f: impl FnOnce(&Framebuffer)) -> bool {
    if !console::is_ready() {
        return false;
    }
    // No compositor (no heap for two frames) means no screen, rather than a
    // screen drawn straight onto the aperture where every frame would tear.
    let Some(back) = compose::target() else { return false };
    let t0 = crate::time::rdtsc();
    f(&back);
    let t1 = crate::time::rdtsc();
    // The whole frame every time, not the diff. Anything that ever writes the
    // aperture directly -- a boot-time window edge, a status strip -- leaves the
    // compositor's shadow describing pixels that are no longer there, and a
    // diff would then leave them on screen forever. A full copy is 4 MB a
    // second at 1280x800, and it heals any stray painter within one frame.
    compose::invalidate();
    compose::present();
    let t2 = crate::time::rdtsc();
    DRAW_CYC.fetch_add(t1 - t0, Ordering::Relaxed);
    PRESENT_CYC.fetch_add(t2 - t1, Ordering::Relaxed);
    WORST_CYC.fetch_max(t2 - t0, Ordering::Relaxed);
    FRAMES.fetch_add(1, Ordering::Relaxed);
    true
}

/// What the screen has cost so far: frames, then the mean draw, mean present
/// and worst whole frame, in microseconds. `None` before the first frame or
/// before the TSC is calibrated.
pub fn cost() -> Option<(u32, u64, u64, u64)> {
    let n = FRAMES.load(Ordering::Relaxed);
    let mhz = crate::time::tsc_mhz();
    if n == 0 || mhz == 0 {
        return None;
    }
    let us = |c: u64| c / mhz;
    Some((
        n,
        us(DRAW_CYC.load(Ordering::Relaxed)) / n as u64,
        us(PRESENT_CYC.load(Ordering::Relaxed)) / n as u64,
        us(WORST_CYC.load(Ordering::Relaxed)),
    ))
}

/// Paint one mining frame. `false` when there is nothing to paint on.
pub fn draw() -> bool {
    let n = gather();
    sample(n.hs as u32);
    paint(|fb| mine_frame(fb, &n))
}

/// Paint the address screen.
pub fn ask_draw() -> bool {
    paint(ask_frame)
}

/// The shell's line editor, mirrored on every keystroke while an address is
/// being asked for. Also clears a refusal once the next attempt is under way.
pub fn typed(line: &str) {
    let asking = super::boot::asking();
    if !asking && !super::boot::is_miner_image() {
        return;
    }
    *TYPED.lock_irq() = String::from(line);
    if !line.is_empty() {
        *REFUSAL.lock_irq() = None;
    }
    // While mining, the next frame shows it: drawing from the shell's task
    // would race the miner's own frame for the one back buffer.
    if asking {
        ask_draw();
    }
}

/// An address was accepted: empty the field without drawing it empty. The
/// mining screen replaces the view next; left standing, the address read on
/// it as a *new* wallet being typed ("press Enter to switch").
pub fn accepted() {
    TYPED.lock_irq().clear();
    *REFUSAL.lock_irq() = None;
}

/// An address was refused: say why on the screen, and empty the field.
pub fn refused(why: &'static str) {
    *REFUSAL.lock_irq() = Some(why);
    TYPED.lock_irq().clear();
    ask_draw();
}

/// Draw if a second has passed since the last frame.
///
/// Called from the miner's socket loop rather than from a task, because this
/// kernel has no sleep and a task that wanted to wake once a second could only
/// spin. A dashboard on its own task cost the fourth slice more than it was
/// worth, measured.
pub fn tick() {
    let now = crate::dev::lapic::ticks();
    let due = NEXT.load(Ordering::Relaxed);
    if now < due {
        return;
    }
    NEXT.store(now + crate::TIMER_HZ as u64, Ordering::Relaxed);
    draw();
}

/// Kept for callers from before the compositor: the first `present` writes
/// every row anyway, so there is nothing left to wipe.
pub fn wipe() {}

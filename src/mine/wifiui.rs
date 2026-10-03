//! Joining Wi-Fi from the miner's screen, with the pointer.
//!
//! A miner image is handed to somebody who boots it and walks away, and its
//! only setup was a wallet address typed into one field. A machine with no
//! cable had nowhere to go: the shell could join a network, but the shell's
//! console is dark on this image and nothing says it is there. So the top bar
//! carries a **Wi-Fi** button, and a click opens a panel over whichever view is
//! up -- the networks heard, a passphrase field, and the state of the link.
//!
//! ### The pointer, on an image with no desktop
//!
//! The desktop's compositor owns the pointer everywhere else, and a miner image
//! does not run one. So the shell's idle loop reads the mouse here instead
//! (`pointer_poll`), which is also the task that owns the keyboard and the
//! shell -- every action a click takes runs where a typed command would. A
//! frame draws the cursor last and records where its buttons are (`Hit`), and a
//! press is matched against the rectangles the last *finished* frame drew, so a
//! click can never land on a button that is half laid out.
//!
//! ### The passphrase is not a command line
//!
//! It is typed through the shell's line editor, because that is where the
//! keyboard goes, and taken by `take_passphrase` in the Enter handler **before
//! the line reaches serial or the history** -- the two places every other line
//! is kept. It is shown as dots, handed to `Wlan::join`, which keeps only the
//! PMK, and the buffer is zeroed.
//!
//! ### What it cannot do yet, said on the panel
//!
//! The GF63's own radio scans and does not join: the Intel driver's station
//! contexts are not written. Turning it on works, the networks appear, and a
//! join fails naming exactly that. A USB adapter or a part whose driver can
//! join goes through the same panel unchanged.

use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::screen::{pane, say, text_w, trunc, AMBER, DIM, GOLD, GREEN, INK, RED, SHADE, WHITE};
use crate::gfx::{Color, Framebuffer};
use crate::net::wifi::Network;
use crate::sync::Spin;

/// What a click can mean.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Hit {
    Open,
    Close,
    Scan,
    TurnOn,
    Leave,
    Cancel,
    Network(usize),
    /// The panel itself, where no button is: nothing, so the bar's button
    /// underneath is not reached through it.
    Nothing,
}

/// The panel's own state. Everything about the link is read from the station
/// every frame; this holds only what the operator has chosen.
struct Ui {
    open: bool,
    /// The secured network whose passphrase is being typed.
    picked: Option<String>,
    /// How many characters of it so far, for the dots. The characters
    /// themselves are never kept here.
    typed: usize,
    /// The last thing a click did, in words.
    said: Option<String>,
    /// The networks as the last frame listed them, so a click on row `i` is
    /// the row the operator saw.
    shown: Vec<Network>,
}

static UI: Spin<Ui> = Spin::new(Ui { open: false, picked: None, typed: 0, said: None, shown: Vec::new() });

/// Hit rectangles: the frame being drawn builds one list, and it becomes the
/// live list only when the frame is finished.
static BUILDING: Spin<Vec<(u32, u32, u32, u32, Hit)>> = Spin::new(Vec::new());
static LIVE: Spin<Vec<(u32, u32, u32, u32, Hit)>> = Spin::new(Vec::new());

/// The button last seen held, so a press is an edge and a drag is not a
/// stream of clicks.
static HELD: AtomicBool = AtomicBool::new(false);
/// A redraw is owed (the pointer moved since the last frame).
static DIRTY: AtomicBool = AtomicBool::new(false);
/// When the pointer last drew a frame, in ticks.
static LAST: AtomicU64 = AtomicU64::new(0);

pub fn open() -> bool {
    UI.lock_irq().open
}

// ---- frames ------------------------------------------------------------------

/// Called at the start of every frame.
pub fn begin() {
    BUILDING.lock_irq().clear();
}

/// Called once the frame is drawn: its buttons are now the ones a click finds.
pub fn commit() {
    let b = core::mem::take(&mut *BUILDING.lock_irq());
    *LIVE.lock_irq() = b;
}

fn hit(x: u32, y: u32, w: u32, h: u32, what: Hit) {
    BUILDING.lock_irq().push((x, y, w, h, what));
}

fn hit_at(x: u32, y: u32) -> Option<Hit> {
    // Last drawn is on top: the panel's buttons are recorded after the bar's.
    LIVE.lock_irq()
        .iter()
        .rev()
        .find(|&&(hx, hy, w, h, _)| x >= hx && x < hx + w && y >= hy && y < hy + h)
        .map(|&(_, _, _, _, h)| h)
}

/// A button: a pane with a label, recorded as a target. Answers its width.
fn button(fb: &Framebuffer, x: u32, y: u32, label: &str, k: u32, edge: Color, what: Hit) -> u32 {
    let w = text_w(label, k) + 20 * k;
    let h = 8 * k + 12 * k;
    pane(fb, x, y, w, h, h / 2, Some(edge));
    say(fb, x + 10 * k, y + 6 * k, label, WHITE, k);
    hit(x, y, w, h, what);
    w
}

/// What the link is doing, read once per frame under the wifi claim.
struct Snap {
    present: bool,
    radio: &'static str,
    state: &'static str,
    secure: bool,
    ssid: Option<String>,
    ip: Option<[u8; 4]>,
    rejoining: bool,
    nets: Vec<Network>,
}

fn snap() -> Snap {
    let (present, radio, state, secure, ssid, rejoining) = {
        let _c = crate::net::claim_wifi();
        match crate::net::wlan() {
            Some(w) => {
                let (s, sec) = w.status();
                (true, w.radio_name(), s, sec, w.ssid(), w.rejoining().is_some())
            }
            None => (false, "", "", false, None, false),
        }
    };
    let i = &crate::net::ifaces()[crate::net::WLAN0];
    let ip = (i.configured && i.ip != crate::net::UNSPECIFIED).then_some(i.ip);
    let mut nets = if present { crate::net::wifi::scan().unwrap_or_default() } else { Vec::new() };
    // Strongest first, and one row per network name: the station picks the
    // strongest access point carrying a name itself, so two rows for one name
    // would be two buttons doing the same thing.
    nets.sort_by(|a, b| b.rssi.cmp(&a.rssi));
    let mut seen: Vec<String> = Vec::new();
    nets.retain(|n| {
        if n.ssid.is_empty() || seen.contains(&n.ssid) {
            return false;
        }
        seen.push(n.ssid.clone());
        true
    });
    Snap { present, radio, state, secure, ssid, ip, rejoining, nets }
}

/// The Wi-Fi button in the top bar, after the name. Answers its width.
pub fn bar_button(fb: &Framebuffer, x: u32, y: u32, k: u32) -> u32 {
    let w = &crate::net::ifaces()[crate::net::WLAN0];
    let (dot, label) = if !w.present() {
        (DIM, "Wi-Fi")
    } else if w.link_cached && w.configured {
        (GREEN, "Wi-Fi")
    } else {
        (AMBER, "Wi-Fi")
    };
    // The state as a dot before the word, in room of its own.
    let w = text_w(label, k) + 30 * k;
    let h = 20 * k;
    pane(fb, x, y, w, h, h / 2, Some(GOLD));
    fb.fill_circle((x + 10 * k) as i32, (y + 10 * k) as i32, (3 * k) as i32, dot);
    say(fb, x + 18 * k, y + 6 * k, label, WHITE, k);
    hit(x, y, w, h, Hit::Open);
    w
}

/// The panel, over whichever view is up, when it is open.
pub fn overlay(fb: &Framebuffer, k: u32) {
    if !open() {
        return;
    }
    let (w, h) = (fb.width(), fb.height());
    fb.tint_rect(0, 0, w, h, SHADE, 150);
    let pw = (w * 6 / 10).max(320 * k).min(w - 16);
    let ph = (h * 7 / 10).min(h - 16);
    let px = (w - pw) / 2;
    let py = (h - ph) / 2;
    // Solid under the glass: the mining view behind is all large type, and
    // through a translucent pane it read as part of the panel.
    let shape = crate::gfx::Shape::round_rect(pw, ph, 14);
    crate::gfx::with_shape(&shape, px as i32, py as i32, || fb.rect(px, py, pw, ph, PANEL));
    pane(fb, px, py, pw, ph, 14, Some(GOLD));
    // Recorded before the panel's buttons, which `hit_at` finds first because
    // it searches newest first: outside the panel a click closes it, inside it
    // and off every button a click does nothing.
    hit(0, 0, w, h, Hit::Close);
    hit(px, py, pw, ph, Hit::Nothing);
    let s = snap();
    let pad = 16 * k;
    let ts = k + 1;
    say(fb, px + pad, py + pad, "Wi-Fi", WHITE, ts);
    let close_w = text_w("Close", k) + 20 * k;
    button(fb, px + pw - pad - close_w, py + pad - 2 * k, "Close", k, GOLD, Hit::Close);

    let mut y = py + pad + 8 * ts + 12 * k;
    let room = ((pw - 2 * pad) / (crate::gfx::font::GLYPH_W * k)) as usize;

    // What the link is doing, in one line.
    let (line, c) = status_line(&s);
    say(fb, px + pad, y, &trunc(&line, room), c, k);
    y += 8 * k + 12 * k;

    let ui_said = UI.lock_irq().said.clone();
    let picked = UI.lock_irq().picked.clone();

    if !s.present {
        // No station: either a part this kernel can bring up, or nothing.
        let seen = crate::net::wireless::last();
        if let Some(part) = seen.first() {
            say(fb, px + pad, y, &trunc(&part.what, room), INK, k);
            y += 8 * k + 10 * k;
            button(fb, px + pad, y, "Turn on Wi-Fi", k, GOLD, Hit::TurnOn);
            y += 20 * k + 10 * k;
            say(fb, px + pad, y, &trunc("Loads the radio's firmware. Takes a few seconds.", room), DIM, k);
        } else {
            say(fb, px + pad, y, &trunc("No Wi-Fi adapter this PC can drive.", room), INK, k);
            y += 8 * k + 8 * k;
            say(fb, px + pad, y, &trunc("Use Ethernet, or USB tethering from a phone.", room), DIM, k);
        }
    } else if let Some(name) = picked {
        // The passphrase, as dots, typed through the shell's line editor.
        say(fb, px + pad, y, &trunc(&alloc::format!("Passphrase for {name}"), room), INK, k);
        y += 8 * k + 8 * k;
        let fw = pw - 2 * pad;
        let fh = 8 * (k + 1) + 20;
        pane(fb, px + pad, y, fw, fh, 8, Some(GOLD));
        let n = UI.lock_irq().typed;
        let dots: String = core::iter::repeat('*').take(n.min(room.saturating_sub(4))).collect();
        say(fb, px + pad + 12, y + 10, if n == 0 { "type it" } else { &dots }, if n == 0 { DIM } else { WHITE }, k + 1);
        y += fh + 10 * k;
        say(fb, px + pad, y, &trunc("then press Enter. Only the key it makes is kept.", room), DIM, k);
        y += 8 * k + 12 * k;
        button(fb, px + pad, y, "Cancel", k, GOLD, Hit::Cancel);
    } else {
        // The networks heard, strongest first.
        let row = 8 * k + 14 * k;
        let foot = 20 * k + 2 * pad;
        let rows = ((py + ph).saturating_sub(foot + y) / row) as usize;
        if s.nets.is_empty() {
            let what = if s.state == "scanning" { "Listening for networks..." } else { "No networks heard yet. Scan to look." };
            say(fb, px + pad, y, what, DIM, k);
        }
        for (i, n) in s.nets.iter().take(rows).enumerate() {
            let ry = y + i as u32 * row;
            let on = s.ssid.as_deref() == Some(n.ssid.as_str()) && s.state == "running";
            if on {
                fb.tint_rect(px + pad, ry, pw - 2 * pad, row - 4 * k, GOLD, 60);
            }
            hit(px + pad, ry, pw - 2 * pad, row - 4 * k, Hit::Network(i));
            let bars = crate::net::wifi::bars(n.rssi);
            let tail = alloc::format!("{}  {}", "|".repeat(bars as usize), if n.secured { "locked" } else { "OPEN" });
            let name_room = room.saturating_sub(tail.chars().count() + 2);
            say(fb, px + pad + 6 * k, ry + 5 * k, &trunc(&n.ssid, name_room), if on { GOLD } else { WHITE }, k);
            let tw = text_w(&tail, k);
            say(fb, px + pw - pad - tw - 6 * k, ry + 5 * k, &tail, if n.secured { DIM } else { AMBER }, k);
        }
        UI.lock_irq().shown = s.nets.clone();
        let by = py + ph - pad - 20 * k;
        let mut bx = px + pad;
        bx += button(fb, bx, by, "Scan", k, GOLD, Hit::Scan) + 10 * k;
        if s.ssid.is_some() || s.state != "idle" {
            button(fb, bx, by, "Leave", k, GOLD, Hit::Leave);
        }
    }

    // What the last click did, until the link itself says more.
    if let Some(said) = ui_said.filter(|_| s.state != "running") {
        let sy = py + ph - pad - 20 * k - 8 * k - 10 * k;
        say(fb, px + pad, sy, &trunc(&said, room), DIM, k);
    }
}

fn status_line(s: &Snap) -> (String, Color) {
    if !s.present {
        return (String::from("Wi-Fi is off"), DIM);
    }
    if s.state == "running" {
        let name = s.ssid.clone().unwrap_or_default();
        return match s.ip {
            Some(ip) => (
                alloc::format!(
                    "Connected to {name}, {}  -- {}.{}.{}.{}",
                    if s.secure { "encrypted" } else { "NOT encrypted" },
                    ip[0], ip[1], ip[2], ip[3]
                ),
                GREEN,
            ),
            None => (alloc::format!("Connected to {name} -- getting an address"), AMBER),
        };
    }
    if s.rejoining {
        return (String::from("The network dropped; rejoining it"), AMBER);
    }
    match s.state {
        "idle" => (alloc::format!("Not connected ({})", s.radio), INK),
        "scanning" => (String::from("Scanning"), TEAL_ISH),
        "authenticating" | "associating" | "handshaking" => (alloc::format!("Joining: {}", s.state), AMBER),
        why => (alloc::format!("Could not join: {why}"), RED),
    }
}

const TEAL_ISH: Color = super::screen::TEAL;
/// The panel's ground: the sky's darkest stop.
const PANEL: Color = Color::new(0x06, 0x14, 0x20);

/// The pointer, last of everything on a frame.
pub fn cursor(fb: &Framebuffer, k: u32) {
    let Some((x, y)) = crate::dev::mouse::position() else { return };
    let (x, y) = (x as i32, y as i32);
    let s = (6 * k) as i32;
    // A black arrow under a white one, so it reads on the sky and on gold.
    fb.fill_triangle((x - 1, y - 2), (x - 1, y + 3 * s + 2), (x + 2 * s + 3, y + 2 * s + 2), SHADE);
    fb.fill_triangle((x, y), (x, y + 3 * s - 1), (x + 2 * s, y + 2 * s), WHITE);
}

// ---- input ---------------------------------------------------------------------

/// The mouse, from the shell's idle loop on a miner image: move the cursor,
/// and act on a press.
pub fn pointer_poll() {
    if !super::boot::is_miner_image() || !crate::dev::mouse::present() || !super::screen::taken() {
        return;
    }
    let s = crate::dev::mouse::take();
    let was = HELD.swap(s.left, Ordering::Relaxed);
    let pressed = s.left && !was;
    if pressed {
        if let Some(h) = hit_at(s.x.max(0) as u32, s.y.max(0) as u32) {
            act(h);
        }
        DIRTY.store(true, Ordering::Relaxed);
        redraw();
        return;
    }
    if s.moved {
        DIRTY.store(true, Ordering::Relaxed);
    }
    // Moves are drawn at most about thirty times a second: a frame is a couple
    // of milliseconds, and this is the core the pool's socket runs on.
    let now = crate::dev::lapic::ticks();
    if DIRTY.load(Ordering::Relaxed) && now.saturating_sub(LAST.load(Ordering::Relaxed)) >= 3 {
        redraw();
    }
}

fn redraw() {
    if super::screen::redraw() {
        DIRTY.store(false, Ordering::Relaxed);
        LAST.store(crate::dev::lapic::ticks(), Ordering::Relaxed);
    }
}

fn act(h: Hit) {
    let now = crate::net::now_ms();
    match h {
        Hit::Open => {
            let mut ui = UI.lock_irq();
            ui.open = true;
            ui.said = None;
            drop(ui);
            // Opening it on a station that has heard nothing is asking to see
            // what is out there.
            let _c = crate::net::claim_wifi();
            if let Some(w) = crate::net::wlan() {
                if w.status().0 == "idle" && w.networks().is_empty() {
                    w.join("", "", now);
                }
            }
        }
        Hit::Close => {
            let mut ui = UI.lock_irq();
            ui.open = false;
            ui.picked = None;
            ui.typed = 0;
        }
        Hit::Scan => {
            let _c = crate::net::claim_wifi();
            if let Some(w) = crate::net::wlan() {
                w.join("", "", now);
            }
            UI.lock_irq().said = Some(String::from("scanning every channel -- a few seconds"));
        }
        Hit::TurnOn => {
            // On the shell's task, as though typed: `iwx boot` is a command and
            // this is the task that runs commands.
            crate::gfx::desk::queue_command("iwx boot");
            UI.lock_irq().said = Some(String::from("turning the radio on: loading its firmware"));
        }
        Hit::Leave => {
            let _c = crate::net::claim_wifi();
            if let Some(w) = crate::net::wlan() {
                w.leave_net();
            }
            UI.lock_irq().said = Some(String::from("left the network"));
        }
        Hit::Cancel => {
            let mut ui = UI.lock_irq();
            ui.picked = None;
            ui.typed = 0;
            ui.said = None;
        }
        Hit::Nothing => {}
        Hit::Network(i) => {
            let n = UI.lock_irq().shown.get(i).cloned();
            let Some(n) = n else { return };
            if n.secured {
                let mut ui = UI.lock_irq();
                ui.picked = Some(n.ssid.clone());
                ui.typed = 0;
                ui.said = None;
            } else {
                let _c = crate::net::claim_wifi();
                if let Some(w) = crate::net::wlan() {
                    w.join(&n.ssid, "", now);
                }
                UI.lock_irq().said = Some(alloc::format!("joining {} (open: not encrypted)", n.ssid));
            }
        }
    }
}

/// Whether the next line typed is a passphrase rather than a command.
pub fn wants_passphrase() -> bool {
    let ui = UI.lock_irq();
    ui.open && ui.picked.is_some()
}

/// The line editor's contents, while a passphrase is wanted: counted, never
/// kept.
pub fn typed(line: &str) {
    UI.lock_irq().typed = line.chars().count();
    DIRTY.store(true, Ordering::Relaxed);
    redraw();
}

/// The finished passphrase, taken by the shell before the line reaches serial
/// or the history. Joined, then zeroed.
pub fn take_passphrase(mut pass: String) {
    let name = {
        let mut ui = UI.lock_irq();
        ui.typed = 0;
        ui.picked.take()
    };
    if let Some(name) = name {
        {
            let _c = crate::net::claim_wifi();
            if let Some(w) = crate::net::wlan() {
                w.join(&name, &pass, crate::net::now_ms());
            }
        }
        UI.lock_irq().said = Some(alloc::format!("joining {name}"));
    }
    // Safety: overwritten with zero bytes, which are valid UTF-8.
    unsafe { pass.as_bytes_mut().fill(0) };
    drop(pass);
    DIRTY.store(true, Ordering::Relaxed);
    redraw();
}

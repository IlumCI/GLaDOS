//! The server end: a listening socket, its connections, and their windows.
//!
//! A Wayland client finds its server at `$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY`,
//! a Unix socket, and the server here is the kernel. So the name is bound in
//! `linux::unix`'s table like any guest's would be, and a client's `connect`
//! queues a connection exactly as it would to wineserver -- the difference is
//! only who accepts it.
//!
//! ### It runs inside the client's own syscalls
//!
//! A guest's descriptors are `Rc<RefCell<..>>` on the guest's task, and the
//! connection is one of them. A server on a task of its own would be a second
//! task reaching into those cells -- the one thing `unix.rs` says is safe only
//! because guests never share a core with anything that touches them. So the
//! server takes its turn **from the guest's syscalls**: after every one, and
//! inside every wait. A client writes a request and then waits for the reply
//! in `poll` or `recvmsg`; the reply is written by that wait. Nothing else ever
//! touches a connection, so there is nothing to lock.
//!
//! ### Windows are the desktop's
//!
//! A committed buffer is copied into a `surface::Frame` the desktop draws from,
//! behind a lock because the compositor paints on its own task. Closing the
//! window from its title bar drops the desktop's handle, which the next turn
//! notices and turns into `xdg_toplevel.close` -- the only thing a compositor
//! is allowed to do about a window it would like gone.

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::RefCell;

use super::client::Client;
use super::surface::{self, Effect, Shared};
use crate::linux::unix::{self, Pipe, Side, Sock};
use crate::sync::Racy;

/// Where clients look, as `XDG_RUNTIME_DIR` and `WAYLAND_DISPLAY` spell it.
pub const RUNTIME_DIR: &str = "/run/glados";
pub const DISPLAY_NAME: &str = "wayland-0";
pub const PATH: &str = "/run/glados/wayland-0";

/// How often frame callbacks fire: sixty a second, the rate a client animating
/// to them expects of a display.
const FRAME_MS: u64 = 16;

struct Session {
    pipe: Rc<RefCell<Pipe>>,
    side: Side,
    client: Client,
    inbuf: Vec<u8>,
    outbuf: Vec<u8>,
    last_frame: u64,
    /// Windows the desktop closed whose client has already been asked.
    asked: Vec<u64>,
    /// Each window's commit count as last shown, so only a new picture
    /// repaints the desktop.
    shown: Vec<(u64, u64)>,
}

static LISTENER: Racy<Option<Rc<RefCell<Sock>>>> = Racy::new(None);
static SESSIONS: Racy<Vec<Session>> = Racy::new(Vec::new());

/// Bind the name, once. Called whenever a guest is installed, so a client
/// started first thing finds a server already listening.
pub fn serve() {
    let l = unsafe { LISTENER.get() };
    if l.is_some() {
        return;
    }
    let sock = Rc::new(RefCell::new(Sock::Fresh));
    if unix::bind(&sock, PATH).is_ok() && unix::listen(&sock).is_ok() {
        *l = Some(sock);
    }
}

/// How many clients are connected.
pub fn clients() -> usize {
    unsafe { SESSIONS.get() }.len()
}

/// One turn: accept, read, answer, show. Cheap when nobody is connected.
pub fn pump() {
    let Some(listener) = unsafe { LISTENER.get() }.clone() else { return };
    let sessions = unsafe { SESSIONS.get() };
    while let Ok(Some((pipe, side))) = unix::accept(&listener) {
        let mut client = Client::new();
        client.publish(surface::WL_COMPOSITOR, 4);
        client.publish(surface::WL_SHM, 1);
        client.publish(surface::XDG_WM_BASE, 1);
        sessions.push(Session {
            pipe,
            side,
            client,
            inbuf: Vec::new(),
            outbuf: Vec::new(),
            last_frame: 0,
            asked: Vec::new(),
            shown: Vec::new(),
        });
    }
    if sessions.is_empty() {
        return;
    }
    let now = crate::net::now_ms();
    let mut changed = false;
    let mut i = 0;
    while i < sessions.len() {
        let keep = turn(&mut sessions[i], now, &mut changed);
        if keep {
            i += 1;
        } else {
            let s = sessions.remove(i);
            end(s);
            changed = true;
        }
    }
    if changed {
        crate::gfx::render::invalidate();
    }
}

/// One connection's turn. Answers whether it continues.
fn turn(s: &mut Session, now: u64, changed: &mut bool) -> bool {
    // Everything readable, and the descriptors that came with it.
    let mut buf = [0u8; 4096];
    loop {
        match unix::read(&s.pipe, s.side, &mut buf) {
            Ok(0) => return false,
            Ok(n) => s.inbuf.extend_from_slice(&buf[..n]),
            Err(_) => break,
        }
    }
    s.client.fds.extend(unix::collect(&s.pipe, s.side));

    // Every whole message, in order.
    let mut at = 0;
    loop {
        match super::wire::frame(&s.inbuf[at..]) {
            Ok(Some(m)) => {
                let len = m.whole();
                let ok = s.client.request(&m);
                at += len;
                if !ok {
                    break;
                }
            }
            Ok(None) => break,
            Err(e) => {
                // A stream that does not frame cannot be resynchronised: say
                // so once and end it.
                s.client.malformed(e);
                break;
            }
        }
    }
    s.inbuf.drain(..at);

    // Frame callbacks, paced.
    if s.client.frames_due() && now.saturating_sub(s.last_frame) >= FRAME_MS {
        s.last_frame = now;
        s.client.fire_frames(now as u32);
    }

    for e in s.client.take_effects() {
        apply(e);
        *changed = true;
    }
    for (key, frame) in s.client.windows() {
        // Three holders while the window is up: the toplevel, the desktop, and
        // the copy `windows` just handed out. Two means the desktop let go --
        // its title bar was clicked -- and the client is asked to close, once.
        if Arc::strong_count(&frame) <= 2 && !s.asked.contains(&key) {
            s.asked.push(key);
            s.client.ask_close(key);
        }
        let commits = frame.lock_irq().commits;
        match s.shown.iter_mut().find(|(k, _)| *k == key) {
            Some((_, c)) if *c == commits => {}
            Some((_, c)) => {
                *c = commits;
                *changed = true;
            }
            None => {
                s.shown.push((key, commits));
                *changed = true;
            }
        }
    }

    // Answers, as far as the socket has room.
    s.outbuf.extend(s.client.take_out());
    if !s.outbuf.is_empty() {
        match unix::write(&s.pipe, s.side, &s.outbuf) {
            Ok(n) => {
                s.outbuf.drain(..n);
            }
            Err(-11) => {}
            Err(_) => return false,
        }
    }
    s.client.alive() || !s.outbuf.is_empty()
}

fn end(s: Session) {
    for (key, _) in s.client.windows() {
        crate::gfx::desk::close_tagged(key);
    }
    unix::close(&s.pipe, s.side);
}

/// Every connection, ended. When the guest goes, its end of each socket goes
/// with it and nothing would otherwise take the windows down.
pub fn reap() {
    let sessions = unsafe { SESSIONS.get() };
    for s in sessions.drain(..) {
        end(s);
    }
}

fn apply(e: Effect) {
    use crate::gfx::{desk, theme};
    match e {
        Effect::Open { key, title, frame } => {
            let (w, h) = {
                let f = frame.lock_irq();
                (f.w, f.h)
            };
            desk::open_app(
                &title,
                desk::ICO_PROGRAMS,
                Box::new(Window { key, frame }),
                w + theme::FRAME * 2,
                h + theme::TITLE_H + theme::FRAME * 2,
            );
        }
        Effect::Retitle { key, title } => desk::retitle_tagged(key, &title),
        Effect::Close { key } => {
            desk::close_tagged(key);
        }
    }
}

/// A client's window on the desktop: the last frame it committed.
struct Window {
    key: u64,
    frame: Shared,
}

impl crate::gfx::DeskApp for Window {
    fn draw_in(&self, fb: &crate::gfx::Framebuffer, client: crate::gfx::theme::Rect, _focused: bool) {
        let f = self.frame.lock_irq();
        fb.rect(client.x, client.y, client.w, client.h, crate::gfx::Color::new(0, 0, 0));
        let w = f.w.min(client.w) as usize;
        let h = f.h.min(client.h) as usize;
        // The client's words are 0x00RRGGBB. A BGRX screen stores exactly that;
        // an RGBX one stores it with red and blue exchanged.
        let swap = fb.encode(crate::gfx::Color::new(1, 2, 3)) != 0x01_0203;
        let mut row: Vec<u32> = Vec::with_capacity(w);
        for y in 0..h {
            let src = &f.px[y * f.w as usize..y * f.w as usize + w];
            if swap {
                row.clear();
                row.extend(src.iter().map(|p| ((p & 0xFF) << 16) | (p & 0xFF00) | ((p >> 16) & 0xFF)));
                fb.blit_span(client.x, client.y + y as u32, &row);
            } else {
                fb.blit_span(client.x, client.y + y as u32, src);
            }
        }
    }

    fn key(&mut self, _k: u8) -> bool {
        false
    }

    fn press(&mut self, _client: crate::gfx::theme::Rect, _x: i32, _y: i32) -> bool {
        false
    }

    fn min_size(&self) -> (u32, u32) {
        (120, 60)
    }

    fn tag(&self) -> u64 {
        self.key
    }
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    out.push((
        "the socket is where XDG_RUNTIME_DIR and WAYLAND_DISPLAY together say",
        alloc::format!("{RUNTIME_DIR}/{DISPLAY_NAME}") == PATH,
    ));
    out
}

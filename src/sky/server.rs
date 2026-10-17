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
    /// Each window's buffer size as last shown, so a client that changes
    /// resolution gets a window of the new size.
    sizes: Vec<(u64, (u32, u32))>,
    /// Where this connection is in the keyboard's evdev ring, and which of its
    /// windows last had the focus.
    kbd: u64,
    focused: Option<u64>,
}

/// Pointer input the desktop gathered for a window, by window key, waiting
/// for that window's connection to take its turn. A lock because the
/// compositor's task pushes and the guest's task drains.
static INPUT: crate::sync::Spin<Vec<(u64, super::seat::Input)>> = crate::sync::Spin::new(Vec::new());

static LISTENER: Racy<Option<Rc<RefCell<Sock>>>> = Racy::new(None);
static SESSIONS: Racy<Vec<Session>> = Racy::new(Vec::new());

/// Bind the name, once. Called whenever a guest is installed, so a client
/// started first thing finds a server already listening.
pub fn serve() {
    crate::linux::netctl::serve();
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
/// Bisecting switches (`linux feat lock|hover|netctl`).
pub static FEAT_LOCK: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);
pub static FEAT_HOVER: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);
pub static FEAT_NETCTL: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);
pub static FEAT_IDLE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);
pub static FEAT_DESK: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);

pub fn pump() {
    // The network endpoint, served on the same turn for the same reason.
    if FEAT_NETCTL.load(core::sync::atomic::Ordering::Relaxed) {
        crate::linux::netctl::pump();
    }
    let Some(listener) = unsafe { LISTENER.get() }.clone() else { return };
    let sessions = unsafe { SESSIONS.get() };
    while let Ok(Some((pipe, side))) = unix::accept(&listener) {
        let mut client = Client::new();
        client.publish(surface::WL_COMPOSITOR, 4);
        client.publish(surface::WL_SHM, 1);
        client.publish(surface::XDG_WM_BASE, 1);
        client.publish(super::seat::WL_OUTPUT, 3);
        client.publish(super::seat::WL_SEAT, 5);
        if FEAT_LOCK.load(core::sync::atomic::Ordering::Relaxed) {
            client.publish(super::seat::RELATIVE_MANAGER, 1);
            client.publish(super::seat::CONSTRAINTS, 1);
        }
        sessions.push(Session {
            pipe,
            side,
            client,
            inbuf: Vec::new(),
            outbuf: Vec::new(),
            last_frame: 0,
            asked: Vec::new(),
            shown: Vec::new(),
            sizes: Vec::new(),
            kbd: crate::linux::input::now_at(crate::linux::input::Dev::Keyboard),
            focused: None,
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
        s.client.fire_frames(now);
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
        let (commits, size) = {
            let f = frame.lock_irq();
            (f.commits, (f.w, f.h))
        };
        if size.0 > 0 && size.1 > 0 {
            match s.sizes.iter_mut().find(|(k, _)| *k == key) {
                Some((_, z)) if *z == size => {}
                Some((_, z)) => {
                    *z = size;
                    crate::gfx::desk::post(crate::gfx::desk::Op::Resize { tag: key, w: size.0, h: size.1 });
                }
                None => s.sizes.push((key, size)),
            }
        }
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

    // Input for this connection's windows: pointer events the desktop
    // queued, and the keyboard's evdev events while one of them has focus.
    let windows = s.client.windows();
    let mine: Vec<(u64, super::seat::Input)> = {
        let mut q = INPUT.lock_irq();
        let (take, keep): (Vec<_>, Vec<_>) = q.drain(..).partition(|(k, _)| windows.iter().any(|(w, _)| w == k));
        *q = keep;
        take
    };
    for (key, ev) in mine {
        if let Some(surf) = s.client.surface_of(key) {
            s.client.input(surf, ev);
        }
    }
    let focused = windows.iter().find(|(_, f)| f.lock_irq().focused).map(|(k, _)| *k);
    if focused != s.focused {
        if let Some(surf) = s.focused.and_then(|k| s.client.surface_of(k)) {
            s.client.input(surf, super::seat::Input::Focus(false));
        }
        if let Some(surf) = focused.and_then(|k| s.client.surface_of(k)) {
            s.client.input(surf, super::seat::Input::Focus(true));
        }
        s.focused = focused;
        // Keys pressed while somebody else had the focus are not this
        // window's.
        s.kbd = crate::linux::input::now_at(crate::linux::input::Dev::Keyboard);
    }
    // The pointer lock: while it holds, the desktop routes buttons to this
    // window wherever the cursor is, and motion arrives as relative counts
    // rather than a position -- a cursor would stop at the screen's edge.
    match s.client.lock_turn(focused) {
        Some(key) => {
            crate::gfx::desk::grab_pointer(key);
            let (dx, dy) = crate::dev::mouse::take_relative();
            if dx != 0 || dy != 0 {
                // The mouse counts y upward; the protocol, downward.
                s.client.relative_motion(dx, -dy);
            }
        }
        None => crate::gfx::desk::release_pointer_of(&windows.iter().map(|(k, _)| *k).collect::<Vec<_>>()),
    }
    if let Some(surf) = focused.and_then(|k| s.client.surface_of(k)) {
        while let Some(e) = crate::linux::input::next_key(&mut s.kbd) {
            // EV_KEY, press or release; autorepeat is the client's to do.
            if e.kind == 1 && e.value != 2 {
                s.client.input(surf, super::seat::Input::Key { code: e.code as u32, down: e.value == 1 });
            }
        }
    }

    // Answers, as far as the socket has room -- and any descriptors the events
    // carry, attached first so they travel with the bytes that name them.
    let fds = core::mem::take(&mut s.client.out_fds);
    if !fds.is_empty() {
        unix::attach(&s.pipe, s.side, fds);
    }
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
        crate::gfx::desk::post(crate::gfx::desk::Op::Close { tag: key });
    }
    unix::close(&s.pipe, s.side);
}

/// Every connection, ended. When the guest goes, its end of each socket goes
/// with it and nothing would otherwise take the windows down.
/// End only the connections whose client end has closed -- what a guest's
/// teardown leaves behind once its descriptors are gone. `reap` ends every
/// client, which with several sessions running is everybody's windows.
pub fn reap_closed() {
    let sessions = unsafe { SESSIONS.get() };
    let mut i = 0;
    while i < sessions.len() {
        if unix::peer_open(&sessions[i].pipe, sessions[i].side) {
            i += 1;
        } else {
            end(sessions.remove(i));
        }
    }
}

pub fn reap() {
    let sessions = unsafe { SESSIONS.get() };
    for s in sessions.drain(..) {
        end(s);
    }
}

/// What a client asked of the desktop, posted rather than done. This runs
/// inside the *guest's* syscalls, on the guest's task, and the window list is
/// the compositor's: `open_app` here was a `Vec::push` on a list the
/// compositor may have been walking for the frame it was painting.
fn apply(e: Effect) {
    use crate::gfx::{desk, theme};
    match e {
        Effect::Open { key, title, frame } => {
            let (w, h) = {
                let f = frame.lock_irq();
                (f.w, f.h)
            };
            desk::post(desk::Op::OpenTagged {
                title,
                icon: desk::ICO_PROGRAMS,
                app: Box::new(Window { key, frame, last: (0, 0), held: 0 }),
                w: w + theme::frame_w() * 2,
                h: h + theme::title_h() + theme::frame_w() * 2,
            });
        }
        Effect::Retitle { key, title } => desk::post(desk::Op::Retitle { tag: key, title }),
        Effect::Close { key } => desk::post(desk::Op::Close { tag: key }),
    }
}

/// A client's window on the desktop: the last frame it committed.
struct Window {
    key: u64,
    frame: Shared,
    /// Where the pointer last pressed or dragged, in window coordinates, and
    /// which button is held -- `release` is told neither.
    last: (i32, i32),
    held: u32,
}

impl Window {
    fn point(&mut self, client: crate::gfx::theme::Rect, x: i32, y: i32, button: Option<(u32, bool)>) -> bool {
        let at = (x - client.x as i32, y - client.y as i32);
        self.last = at;
        INPUT.lock_irq().push((self.key, super::seat::Input::Pointer { x: at.0, y: at.1, button }));
        true
    }
}

impl crate::gfx::DeskApp for Window {
    /// The commit count: a new picture is a new commit, and nothing else in
    /// this window changes.
    fn stamp(&self) -> Option<u64> {
        let f = self.frame.lock_irq();
        Some(crate::gfx::scene::Stamp::of((f.commits, f.w, f.h)))
    }

    fn draw_in(&self, fb: &crate::gfx::Framebuffer, client: crate::gfx::theme::Rect, _focused: bool) {
        let mut f = self.frame.lock_irq();
        f.focused = _focused;
        // Painted is shown: the desktop lays this surface into the frame it
        // is composing and presents it before it next looks at a client.
        f.presented = f.commits;
        let w = f.w.min(client.w) as usize;
        let h = f.h.min(client.h) as usize;
        // Black only where the picture does not reach -- a window the client
        // drew smaller than the frame around it. It was the whole client
        // area, every time, under a picture that covers nearly all of it.
        let black = crate::gfx::Color::new(0, 0, 0);
        if (w as u32) < client.w {
            fb.rect(client.x + w as u32, client.y, client.w - w as u32, client.h, black);
        }
        if (h as u32) < client.h {
            fb.rect(client.x, client.y + h as u32, w as u32, client.h - h as u32, black);
        }
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

    /// Taken and dropped: a client's keys are the evdev events, delivered as
    /// such, and handing it the decoded character too would type twice.
    fn key(&mut self, _k: u8) -> bool {
        true
    }

    fn press(&mut self, client: crate::gfx::theme::Rect, x: i32, y: i32) -> bool {
        self.held = 0x110; // BTN_LEFT
        self.point(client, x, y, Some((0x110, true)))
    }

    fn right_press(&mut self, client: crate::gfx::theme::Rect, x: i32, y: i32) -> bool {
        self.held = 0x111; // BTN_RIGHT
        self.point(client, x, y, Some((0x111, true)))
    }

    fn drag(&mut self, client: crate::gfx::theme::Rect, x: i32, y: i32) -> bool {
        self.point(client, x, y, None)
    }

    fn hover(&mut self, client: crate::gfx::theme::Rect, x: i32, y: i32) -> bool {
        // Motion with nothing held, deduplicated: the desktop calls this for
        // every packet over the window and a client wants positions, not
        // repeats.
        let at = (x - client.x as i32, y - client.y as i32);
        if !FEAT_HOVER.load(core::sync::atomic::Ordering::Relaxed) {
            return false;
        }
        if at != self.last || self.held != 0 {
            self.point(client, x, y, None);
        }
        false
    }

    fn release(&mut self) -> bool {
        let (x, y) = self.last;
        let b = core::mem::take(&mut self.held);
        INPUT.lock_irq().push((self.key, super::seat::Input::Pointer { x, y, button: Some((b.max(0x110), false)) }));
        true
    }

    fn min_size(&self) -> (u32, u32) {
        (120, 60)
    }

    /// At its own size: nothing here sends `xdg_toplevel.configure` with a
    /// new one yet, so a tiled slot larger than the picture is black margin
    /// -- Portal's 800x600 in a 1280x700 slot was three quarters void.
    fn floats(&self) -> bool {
        true
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

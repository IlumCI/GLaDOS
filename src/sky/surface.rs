//! Surfaces, shared-memory buffers, and windows: where there is a picture.
//!
//! A Wayland client draws into memory it owns and tells the server where.
//! `wl_shm` takes a descriptor for that memory and makes a pool of it; a
//! pool cuts `wl_buffer`s out of itself, each an offset, a size, a stride and
//! a pixel format; a `wl_surface` has a buffer attached and then **commits**,
//! and the commit is the only moment anything the client said about the
//! surface takes effect. `xdg_wm_base` gives a surface the role of a window,
//! which is what gets it a title and a place on the desktop.
//!
//! ### The pixels are read at commit, and copied
//!
//! The memory is a `memfd`, and a memfd's pages are this kernel's own heap,
//! identity mapped: the server reads the client's pixels at their own address
//! with no mapping of its own. They are **copied** into the window at commit
//! and the buffer is released at once. Holding the client's buffer until the
//! next frame was the alternative and buys a copy saved; it costs a client
//! waiting on a release that depends on when the desktop happens to repaint,
//! and a desktop drawing from memory the client may already be writing the
//! next frame into.
//!
//! ### What a commit is checked against
//!
//! Everything a client says about a buffer is a number it chose, and a buffer
//! is a rectangle of somebody's memory the compositor is about to read. So a
//! buffer must fit inside its pool when it is made, its stride must hold its
//! width, and at commit it is checked again against the pool's memory as it
//! stands -- a pool can be made larger, and a server that trusted the bounds
//! from when the buffer was made would read past a memfd that has not grown.
//!
//! ### Effects, not calls
//!
//! A commit that makes a window does not open one here. It records an
//! `Effect`, and `server` applies it to the desktop. Keeping the desktop out
//! of this file is what lets every claim below run with no screen, no client
//! and no guest -- the bargain `wire` and `object` made before it.

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::RefCell;

use super::client::{done, Client};
use super::wire::{Error, Message, Writer};
use crate::sync::Spin;

pub const WL_COMPOSITOR: &str = "wl_compositor";
pub const WL_SURFACE: &str = "wl_surface";
pub const WL_REGION: &str = "wl_region";
pub const WL_SHM: &str = "wl_shm";
pub const WL_SHM_POOL: &str = "wl_shm_pool";
pub const WL_BUFFER: &str = "wl_buffer";
pub const WL_CALLBACK: &str = "wl_callback";
pub const XDG_WM_BASE: &str = "xdg_wm_base";
pub const XDG_POSITIONER: &str = "xdg_positioner";
pub const XDG_SURFACE: &str = "xdg_surface";
pub const XDG_TOPLEVEL: &str = "xdg_toplevel";
pub const XDG_POPUP: &str = "xdg_popup";

/// The two formats every server must offer, and the only two this one does.
pub const ARGB8888: u32 = 0;
pub const XRGB8888: u32 = 1;

/// The largest window this will make. A client choosing the size of an
/// allocation is a client choosing how much of the heap a commit copies into.
pub const MAX_SIDE: u32 = 4096;

const EV_DELETE_ID: u16 = 1;

pub fn handles(iface: &str) -> bool {
    matches!(
        iface,
        WL_COMPOSITOR
            | WL_SURFACE
            | WL_REGION
            | WL_SHM
            | WL_SHM_POOL
            | WL_BUFFER
            | XDG_WM_BASE
            | XDG_POSITIONER
            | XDG_SURFACE
            | XDG_TOPLEVEL
            | XDG_POPUP
    )
}

/// A window's picture, shared with the desktop that draws it.
///
/// Behind a lock because two tasks reach it: the guest's, inside a commit,
/// and the compositor's, painting. Pixels are `0x00RRGGBB` whatever the client
/// sent -- alpha is dropped, there being nothing under a window to blend with.
pub struct Frame {
    pub w: u32,
    pub h: u32,
    pub px: Vec<u32>,
    pub title: String,
    /// Commits so far, so a reader can tell a new picture from the last one.
    pub commits: u64,
}

pub type Shared = Arc<Spin<Frame>>;

/// What a request asks of the desktop.
pub enum Effect {
    Open { key: u64, title: String, frame: Shared },
    Retitle { key: u64, title: String },
    Close { key: u64 },
}

/// Where a pool's memory is. Shared memory from a guest in practice; a plain
/// vector in the claims, which need no guest.
pub enum Memory {
    Memfd(Rc<RefCell<crate::linux::fs::Memfd>>),
    #[allow(dead_code)]
    Owned(Vec<u8>),
}

impl Memory {
    fn with<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        match self {
            Memory::Memfd(m) => f(m.borrow().bytes()),
            Memory::Owned(v) => f(v),
        }
    }

    fn len(&self) -> usize {
        self.with(|b| b.len())
    }
}

struct Pool {
    mem: Rc<Memory>,
    size: usize,
}

#[derive(Clone)]
struct Buffer {
    mem: Rc<Memory>,
    #[allow(dead_code)]
    pool_size: usize,
    offset: usize,
    w: u32,
    h: u32,
    stride: usize,
    format: u32,
}

#[derive(Default)]
struct Surface {
    /// What `attach` said since the last commit: `Some(None)` is an explicit
    /// detach, which is different from not having been told anything.
    pending: Option<Option<u32>>,
    frames: Vec<u32>,
    /// The xdg_surface that gave this its role, if one has.
    xdg: Option<u32>,
}

#[derive(Default)]
struct XdgSurface {
    surface: u32,
    toplevel: Option<u32>,
    popup: Option<u32>,
    /// The serial last sent, and whether any configure has been acknowledged.
    sent: u32,
    acked: bool,
}

struct Toplevel {
    xdg: u32,
    title: String,
    /// The desktop window, once the first buffer made one.
    window: Option<(u64, Shared)>,
}

/// Everything below `wl_registry`, on one connection.
#[derive(Default)]
pub struct Scene {
    pools: BTreeMap<u32, Pool>,
    buffers: BTreeMap<u32, Buffer>,
    surfaces: BTreeMap<u32, Surface>,
    xdg: BTreeMap<u32, XdgSurface>,
    toplevels: BTreeMap<u32, Toplevel>,
    /// Frame callbacks committed and not yet fired.
    due: Vec<u32>,
    pub effects: Vec<Effect>,
}

/// Window keys, unique across every connection, so the desktop never confuses
/// one client's window for another's.
static NEXT_KEY: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);

impl Client {
    /// A global was just bound: anything that must be said on binding.
    pub(super) fn bound(&mut self, id: u32, iface: &str) {
        if iface == WL_SHM {
            for f in [ARGB8888, XRGB8888] {
                let mut w = Writer::new(id, 0);
                w.uint(f);
                self.post(w);
            }
        }
    }

    /// An object the client made is gone: forget it, and say so. A client's
    /// map frees the number only on `delete_id`, so a server that merely
    /// forgot would leave it growing forever.
    fn forget(&mut self, id: u32) -> Result<(), Error> {
        self.objects.destroy(id)?;
        let mut w = Writer::new(super::object::DISPLAY, EV_DELETE_ID);
        w.uint(id);
        self.post(w);
        Ok(())
    }

    pub(super) fn surface_request(&mut self, iface: &'static str, _version: u32, msg: &Message) -> Result<(), Error> {
        match iface {
            WL_COMPOSITOR => self.compositor(msg),
            WL_SURFACE => self.surface(msg),
            WL_REGION => self.region(msg),
            WL_SHM => self.shm(msg),
            WL_SHM_POOL => self.pool(msg),
            WL_BUFFER => self.buffer(msg),
            XDG_WM_BASE => self.wm_base(msg),
            XDG_POSITIONER => self.positioner(msg),
            XDG_SURFACE => self.xdg_surface(msg),
            XDG_TOPLEVEL => self.toplevel(msg),
            XDG_POPUP => self.popup(msg),
            _ => Err(Error::NoMethod),
        }
    }

    fn compositor(&mut self, msg: &Message) -> Result<(), Error> {
        let mut r = msg.reader();
        let id = r.new_id()?;
        done(&r)?;
        match msg.opcode {
            0 => {
                self.objects.create(id, WL_SURFACE, 4)?;
                self.scene.surfaces.insert(id, Surface::default());
                Ok(())
            }
            1 => self.objects.create(id, WL_REGION, 1),
            _ => Err(Error::NoMethod),
        }
    }

    /// Regions say which parts are opaque and which take input. Accepted and
    /// not acted on: every window here is drawn opaque and takes input over
    /// its whole area, which is a correct answer to both questions.
    fn region(&mut self, msg: &Message) -> Result<(), Error> {
        match msg.opcode {
            0 => {
                done(&msg.reader())?;
                self.forget(msg.object)
            }
            1 | 2 => Ok(()),
            _ => Err(Error::NoMethod),
        }
    }

    fn surface(&mut self, msg: &Message) -> Result<(), Error> {
        let id = msg.object;
        let mut r = msg.reader();
        match msg.opcode {
            // destroy
            0 => {
                done(&r)?;
                if let Some(s) = self.scene.surfaces.remove(&id) {
                    if let Some(x) = s.xdg {
                        if let Some(xs) = self.scene.xdg.get_mut(&x) {
                            xs.surface = 0;
                        }
                    }
                }
                self.forget(id)
            }
            // attach
            1 => {
                let buffer = r.object()?;
                let _x = r.int()?;
                let _y = r.int()?;
                done(&r)?;
                if buffer != 0 && !self.scene.buffers.contains_key(&buffer) {
                    return Err(Error::NoObject);
                }
                if let Some(s) = self.scene.surfaces.get_mut(&id) {
                    s.pending = Some((buffer != 0).then_some(buffer));
                }
                Ok(())
            }
            // damage, damage_buffer: the whole window is copied at commit, so
            // which part changed is a question with no use here.
            2 | 9 => {
                for _ in 0..4 {
                    r.int()?;
                }
                done(&r)
            }
            // frame
            3 => {
                let cb = r.new_id()?;
                done(&r)?;
                self.objects.create(cb, WL_CALLBACK, 1)?;
                if let Some(s) = self.scene.surfaces.get_mut(&id) {
                    s.frames.push(cb);
                }
                Ok(())
            }
            // set_opaque_region, set_input_region
            4 | 5 => {
                r.object()?;
                done(&r)
            }
            6 => {
                done(&r)?;
                self.commit(id)
            }
            // set_buffer_transform, set_buffer_scale: one output at scale one,
            // untransformed, which is what the defaults already say.
            7 | 8 => {
                r.int()?;
                done(&r)
            }
            // offset
            10 => {
                r.int()?;
                r.int()?;
                done(&r)
            }
            _ => Err(Error::NoMethod),
        }
    }

    /// The moment everything said about a surface takes effect.
    fn commit(&mut self, id: u32) -> Result<(), Error> {
        let Some(s) = self.scene.surfaces.get_mut(&id) else { return Err(Error::NoObject) };
        let pending = s.pending.take();
        let frames = core::mem::take(&mut s.frames);
        let xdg = s.xdg;
        self.scene.due.extend(frames);

        let Some(attached) = pending else { return Ok(()) };
        let toplevel = xdg.and_then(|x| self.scene.xdg.get(&x)).and_then(|x| {
            // A window may not have content before it has answered a configure:
            // until then it does not know what size it has been asked to be.
            (x.toplevel.is_some()).then_some((x.toplevel.unwrap_or(0), x.acked))
        });
        let Some(buffer) = attached else {
            // An explicit detach unmaps the window.
            if let Some((t, _)) = toplevel {
                if let Some(tl) = self.scene.toplevels.get_mut(&t) {
                    if let Some((key, _)) = tl.window.take() {
                        self.scene.effects.push(Effect::Close { key });
                    }
                }
            }
            return Ok(());
        };
        if let Some((_, false)) = toplevel {
            return Err(Error::NotConfigured);
        }
        let b = self.scene.buffers.get(&buffer).cloned().ok_or(Error::NoObject)?;
        let px = copy_pixels(&b)?;
        // Copied, so the client may have its buffer back now.
        self.post(Writer::new(buffer, 0));
        let Some((t, _)) = toplevel else { return Ok(()) };
        let Some(tl) = self.scene.toplevels.get_mut(&t) else { return Ok(()) };
        match &tl.window {
            Some((_, f)) => {
                let mut f = f.lock_irq();
                f.w = b.w;
                f.h = b.h;
                f.px = px;
                f.commits += 1;
            }
            None => {
                let key = NEXT_KEY.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                let frame = Arc::new(Spin::new(Frame { w: b.w, h: b.h, px, title: tl.title.clone(), commits: 1 }));
                tl.window = Some((key, Arc::clone(&frame)));
                self.scene.effects.push(Effect::Open { key, title: tl.title.clone(), frame });
            }
        }
        Ok(())
    }

    /// Fire every frame callback committed so far. The desktop paints when it
    /// is told something changed, so "the frame was presented" is "the commit
    /// was taken", paced by the caller.
    pub fn fire_frames(&mut self, time_ms: u32) {
        let due = core::mem::take(&mut self.scene.due);
        for cb in due {
            if self.objects.get(cb).is_none() {
                continue;
            }
            let mut w = Writer::new(cb, 0);
            w.uint(time_ms);
            self.post(w);
            let _ = self.forget(cb);
        }
    }

    pub fn frames_due(&self) -> bool {
        !self.scene.due.is_empty()
    }

    fn shm(&mut self, msg: &Message) -> Result<(), Error> {
        let mut r = msg.reader();
        match msg.opcode {
            // create_pool
            0 => {
                let id = r.new_id()?;
                let fd = r.fd(&mut self.fds)?;
                let size = r.int()?;
                done(&r)?;
                let mem = match fd {
                    crate::linux::fs::Fd::Memfd(m) => Memory::Memfd(m),
                    _ => return Err(Error::BadFd),
                };
                self.pool_from(id, mem, size)
            }
            // release
            1 => {
                done(&r)?;
                self.forget(msg.object)
            }
            _ => Err(Error::NoMethod),
        }
    }

    /// A pool, from memory already in hand. Split out of `create_pool` so the
    /// claims can make one with no descriptor.
    pub(super) fn pool_from(&mut self, id: u32, mem: Memory, size: i32) -> Result<(), Error> {
        if size <= 0 || size as usize > mem.len() {
            return Err(Error::BadBuffer);
        }
        self.objects.create(id, WL_SHM_POOL, 1)?;
        self.scene.pools.insert(id, Pool { mem: Rc::new(mem), size: size as usize });
        Ok(())
    }

    fn pool(&mut self, msg: &Message) -> Result<(), Error> {
        let id = msg.object;
        let mut r = msg.reader();
        match msg.opcode {
            // create_buffer
            0 => {
                let bid = r.new_id()?;
                let offset = r.int()?;
                let w = r.int()?;
                let h = r.int()?;
                let stride = r.int()?;
                let format = r.uint()?;
                done(&r)?;
                if format != ARGB8888 && format != XRGB8888 {
                    return Err(Error::BadFormat);
                }
                let p = self.scene.pools.get(&id).ok_or(Error::NoObject)?;
                let b = check_buffer(offset, w, h, stride, p.size)?;
                let buf = Buffer { mem: Rc::clone(&p.mem), pool_size: p.size, format, ..b };
                self.objects.create(bid, WL_BUFFER, 1)?;
                self.scene.buffers.insert(bid, buf);
                Ok(())
            }
            // destroy: buffers made from it live on, holding the memory.
            1 => {
                done(&r)?;
                self.scene.pools.remove(&id);
                self.forget(id)
            }
            // resize: only larger, and only as far as the memory goes.
            2 => {
                let size = r.int()?;
                done(&r)?;
                let p = self.scene.pools.get_mut(&id).ok_or(Error::NoObject)?;
                if size <= 0 || (size as usize) < p.size || size as usize > p.mem.len() {
                    return Err(Error::BadBuffer);
                }
                p.size = size as usize;
                Ok(())
            }
            _ => Err(Error::NoMethod),
        }
    }

    fn buffer(&mut self, msg: &Message) -> Result<(), Error> {
        match msg.opcode {
            0 => {
                done(&msg.reader())?;
                self.scene.buffers.remove(&msg.object);
                self.forget(msg.object)
            }
            _ => Err(Error::NoMethod),
        }
    }

    fn wm_base(&mut self, msg: &Message) -> Result<(), Error> {
        let mut r = msg.reader();
        match msg.opcode {
            0 => {
                done(&r)?;
                self.forget(msg.object)
            }
            1 => {
                let id = r.new_id()?;
                done(&r)?;
                self.objects.create(id, XDG_POSITIONER, 1)
            }
            // get_xdg_surface
            2 => {
                let id = r.new_id()?;
                let surface = r.object()?;
                done(&r)?;
                let s = self.scene.surfaces.get_mut(&surface).ok_or(Error::NoObject)?;
                if s.xdg.is_some() {
                    return Err(Error::Role);
                }
                s.xdg = Some(id);
                self.objects.create(id, XDG_SURFACE, 1)?;
                self.scene.xdg.insert(id, XdgSurface { surface, ..Default::default() });
                Ok(())
            }
            // pong: no ping is sent, so any answer is an answer to nothing.
            3 => {
                r.uint()?;
                done(&r)
            }
            _ => Err(Error::NoMethod),
        }
    }

    /// Positioners place popups. Every setter is accepted and the popup it
    /// places is dismissed at once (see `popup`), so nothing reads them.
    fn positioner(&mut self, msg: &Message) -> Result<(), Error> {
        match msg.opcode {
            0 => {
                done(&msg.reader())?;
                self.forget(msg.object)
            }
            1..=9 => Ok(()),
            _ => Err(Error::NoMethod),
        }
    }

    fn configure(&mut self, xdg: u32) {
        self.serial += 1;
        let serial = self.serial;
        if let Some(x) = self.scene.xdg.get_mut(&xdg) {
            x.sent = serial;
        }
        let mut w = Writer::new(xdg, 0);
        w.uint(serial);
        self.post(w);
    }

    fn xdg_surface(&mut self, msg: &Message) -> Result<(), Error> {
        let id = msg.object;
        let mut r = msg.reader();
        match msg.opcode {
            0 => {
                done(&r)?;
                if let Some(x) = self.scene.xdg.remove(&id) {
                    if let Some(s) = self.scene.surfaces.get_mut(&x.surface) {
                        s.xdg = None;
                    }
                }
                self.forget(id)
            }
            // get_toplevel
            1 => {
                let tid = r.new_id()?;
                done(&r)?;
                let x = self.scene.xdg.get_mut(&id).ok_or(Error::NoObject)?;
                if x.toplevel.is_some() || x.popup.is_some() {
                    return Err(Error::Role);
                }
                x.toplevel = Some(tid);
                self.objects.create(tid, XDG_TOPLEVEL, 1)?;
                self.scene.toplevels.insert(tid, Toplevel { xdg: id, title: String::from("Linux"), window: None });
                // Zero by zero is "choose your own size", and no states.
                let mut w = Writer::new(tid, 0);
                w.int(0);
                w.int(0);
                w.array(&[]);
                self.post(w);
                self.configure(id);
                Ok(())
            }
            // get_popup: given the role and dismissed at once. A menu that
            // cannot appear is better than one appearing somewhere unasked.
            2 => {
                let pid = r.new_id()?;
                let _parent = r.object()?;
                let _positioner = r.object()?;
                done(&r)?;
                let x = self.scene.xdg.get_mut(&id).ok_or(Error::NoObject)?;
                if x.toplevel.is_some() || x.popup.is_some() {
                    return Err(Error::Role);
                }
                x.popup = Some(pid);
                self.objects.create(pid, XDG_POPUP, 1)?;
                let w = Writer::new(pid, 1);
                self.post(w);
                Ok(())
            }
            // set_window_geometry
            3 => {
                for _ in 0..4 {
                    r.int()?;
                }
                done(&r)
            }
            // ack_configure
            4 => {
                let serial = r.uint()?;
                done(&r)?;
                let x = self.scene.xdg.get_mut(&id).ok_or(Error::NoObject)?;
                if serial <= x.sent {
                    x.acked = true;
                }
                Ok(())
            }
            _ => Err(Error::NoMethod),
        }
    }

    fn toplevel(&mut self, msg: &Message) -> Result<(), Error> {
        let id = msg.object;
        let mut r = msg.reader();
        match msg.opcode {
            0 => {
                done(&r)?;
                if let Some(t) = self.scene.toplevels.remove(&id) {
                    if let Some((key, _)) = t.window {
                        self.scene.effects.push(Effect::Close { key });
                    }
                    if let Some(x) = self.scene.xdg.get_mut(&t.xdg) {
                        x.toplevel = None;
                    }
                }
                self.forget(id)
            }
            // set_title
            2 => {
                let title = r.string()?.unwrap_or_default();
                done(&r)?;
                let t = self.scene.toplevels.get_mut(&id).ok_or(Error::NoObject)?;
                let title: String = title.chars().take(80).collect();
                t.title = title.clone();
                if let Some((key, f)) = &t.window {
                    f.lock_irq().title = title.clone();
                    self.scene.effects.push(Effect::Retitle { key: *key, title });
                }
                Ok(())
            }
            // set_parent, set_app_id, show_window_menu, move, resize, the size
            // hints, and the state requests: accepted. The desktop's own frame
            // moves and sizes a window, and the states it has no way to take
            // are states a client asking for them can live without.
            1 | 3..=13 => Ok(()),
            _ => Err(Error::NoMethod),
        }
    }

    fn popup(&mut self, msg: &Message) -> Result<(), Error> {
        match msg.opcode {
            0 => {
                done(&msg.reader())?;
                self.forget(msg.object)
            }
            1 | 2 => Ok(()),
            _ => Err(Error::NoMethod),
        }
    }

    /// The desktop closed a window: ask the client to close it, which is all
    /// the protocol lets a compositor do.
    pub fn ask_close(&mut self, key: u64) {
        let ids: Vec<u32> = self
            .scene
            .toplevels
            .iter()
            .filter(|(_, t)| t.window.as_ref().is_some_and(|(k, _)| *k == key))
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let w = Writer::new(id, 1);
            self.post(w);
        }
    }

    /// The windows this connection has on the desktop.
    pub fn windows(&self) -> Vec<(u64, Shared)> {
        self.scene.toplevels.values().filter_map(|t| t.window.clone()).collect()
    }

    pub fn take_effects(&mut self) -> Vec<Effect> {
        core::mem::take(&mut self.scene.effects)
    }
}

/// Check a buffer's geometry against its pool, as numbers the client chose.
fn check_buffer(offset: i32, w: i32, h: i32, stride: i32, pool: usize) -> Result<Buffer, Error> {
    if offset < 0 || w <= 0 || h <= 0 || stride <= 0 {
        return Err(Error::BadBuffer);
    }
    let (w, h) = (w as u32, h as u32);
    if w > MAX_SIDE || h > MAX_SIDE || (stride as u64) < w as u64 * 4 {
        return Err(Error::BadBuffer);
    }
    let end = offset as u64 + stride as u64 * (h as u64 - 1) + w as u64 * 4;
    if end > pool as u64 {
        return Err(Error::BadBuffer);
    }
    Ok(Buffer {
        mem: Rc::new(Memory::Owned(Vec::new())),
        pool_size: pool,
        offset: offset as usize,
        w,
        h,
        stride: stride as usize,
        format: XRGB8888,
    })
}

/// The buffer's pixels, as `0x00RRGGBB`, checked again against the memory as
/// it is now rather than as it was when the buffer was made.
fn copy_pixels(b: &Buffer) -> Result<Vec<u32>, Error> {
    // Both formats are little-endian 0xAARRGGBB words; alpha is dropped.
    let _ = b.format;
    let need = b.offset + b.stride * (b.h as usize - 1) + b.w as usize * 4;
    b.mem.with(|bytes| {
        if need > bytes.len() {
            return Err(Error::BadBuffer);
        }
        let mut px = Vec::with_capacity((b.w * b.h) as usize);
        for y in 0..b.h as usize {
            let row = &bytes[b.offset + y * b.stride..b.offset + y * b.stride + b.w as usize * 4];
            for p in row.chunks_exact(4) {
                px.push(u32::from_le_bytes([p[0], p[1], p[2], 0]));
            }
        }
        Ok(px)
    })
}

/// Claims, with no guest and no screen.
pub fn checks() -> Vec<(&'static str, bool)> {
    use super::client::req as request;
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let mut c = Client::new();
    c.publish(WL_COMPOSITOR, 4);
    c.publish(WL_SHM, 1);
    c.publish(XDG_WM_BASE, 1);
    let send = |c: &mut Client, m: Vec<u8>| -> bool {
        match super::wire::frame(&m) {
            Ok(Some(msg)) => c.request(&msg),
            _ => false,
        }
    };
    // registry 2; compositor 3, shm 4, wm_base 5
    send(&mut c, request(1, 1, |w| { w.new_id(2); }));
    for (name, iface, id) in [(1u32, WL_COMPOSITOR, 3u32), (2, WL_SHM, 4), (3, XDG_WM_BASE, 5)] {
        send(&mut c, request(2, 0, |w| { w.uint(name); w.string(iface); w.uint(1); w.new_id(id); }));
    }
    let out0 = c.take_out();
    out.push((
        "binding wl_shm says which formats it takes: ARGB8888 and XRGB8888",
        count_events(&out0, 4, 0) == 2,
    ));

    // A 4x2 pool, as plain memory: two rows of four pixels, stride 16.
    let mut mem = alloc::vec![0u8; 32];
    for i in 0..8 {
        mem[i * 4..i * 4 + 4].copy_from_slice(&(0xFF00_0000u32 | (i as u32 * 0x10_1010)).to_le_bytes());
    }
    let pool_ok = c.pool_from(6, Memory::Owned(mem), 32).is_ok();
    // surface 7, buffer 8
    send(&mut c, request(3, 0, |w| { w.new_id(7); }));
    send(&mut c, request(6, 0, |w| { w.new_id(8); w.int(0); w.int(4); w.int(2); w.int(16); w.uint(XRGB8888); }));
    out.push(("a buffer that fits its pool is made", pool_ok && c.scene.buffers.contains_key(&8)));

    let mut bad = Client::new();
    bad.publish(WL_SHM, 1);
    let _ = bad.pool_from(6, Memory::Owned(alloc::vec![0u8; 32]), 32);
    let fits = |b: &mut Client, id: u32, off: i32, w: i32, h: i32, st: i32| {
        let m = request(6, 0, |x| { x.new_id(id); x.int(off); x.int(w); x.int(h); x.int(st); x.uint(XRGB8888); });
        match super::wire::frame(&m) {
            Ok(Some(msg)) => b.dispatch(&msg).is_ok(),
            _ => false,
        }
    };
    out.push((
        "a buffer reaching past its pool, a stride narrower than its width, or a negative offset is refused",
        !fits(&mut bad, 9, 4, 4, 2, 16) && !fits(&mut bad, 10, 0, 4, 2, 12) && !fits(&mut bad, 11, -4, 4, 2, 16)
            && fits(&mut bad, 12, 0, 4, 2, 16),
    ));
    let fmt = request(6, 0, |x| { x.new_id(13); x.int(0); x.int(4); x.int(2); x.int(16); x.uint(7); });
    out.push((
        "and a format nobody offered",
        matches!(super::wire::frame(&fmt), Ok(Some(m)) if bad.dispatch(&m) == Err(Error::BadFormat)),
    ));

    // xdg_surface 9, toplevel 10.
    send(&mut c, request(5, 2, |w| { w.new_id(9); w.object(7); }));
    send(&mut c, request(9, 1, |w| { w.new_id(10); }));
    let cfg = c.take_out();
    out.push((
        "a new window is told to configure, toplevel then surface",
        count_events(&cfg, 10, 0) == 1 && count_events(&cfg, 9, 0) == 1,
    ));

    // Content before the configure is answered is a protocol error.
    {
        let mut e = Client::new();
        e.publish(WL_COMPOSITOR, 4);
        e.publish(XDG_WM_BASE, 1);
        let _ = e.pool_from(6, Memory::Owned(alloc::vec![0u8; 32]), 32);
        let s = |e: &mut Client, m: Vec<u8>| match super::wire::frame(&m) {
            Ok(Some(msg)) => e.dispatch(&msg),
            _ => Err(Error::Truncated),
        };
        let _ = s(&mut e, request(1, 1, |w| { w.new_id(2); }));
        let _ = s(&mut e, request(2, 0, |w| { w.uint(1); w.string(WL_COMPOSITOR); w.uint(4); w.new_id(3); }));
        let _ = s(&mut e, request(2, 0, |w| { w.uint(2); w.string(XDG_WM_BASE); w.uint(1); w.new_id(5); }));
        let _ = s(&mut e, request(3, 0, |w| { w.new_id(7); }));
        let _ = s(&mut e, request(6, 0, |w| { w.new_id(8); w.int(0); w.int(4); w.int(2); w.int(16); w.uint(XRGB8888); }));
        let _ = s(&mut e, request(5, 2, |w| { w.new_id(9); w.object(7); }));
        let _ = s(&mut e, request(9, 1, |w| { w.new_id(10); }));
        let _ = s(&mut e, request(7, 1, |w| { w.object(8); w.int(0); w.int(0); }));
        out.push((
            "a window given content before it acknowledged a configure is refused",
            s(&mut e, request(7, 6, |_| {})) == Err(Error::NotConfigured),
        ));
    }

    // Acknowledge, title, attach, frame, commit: a window opens with the pixels.
    let serial = c.serial;
    send(&mut c, request(9, 4, |w| { w.uint(serial); }));
    send(&mut c, request(10, 2, |w| { w.string("hello"); }));
    send(&mut c, request(7, 1, |w| { w.object(8); w.int(0); w.int(0); }));
    send(&mut c, request(7, 3, |w| { w.new_id(11); }));
    let alive = send(&mut c, request(7, 6, |_| {}));
    let fx = c.take_effects();
    let opened = fx.iter().find_map(|e| match e {
        Effect::Open { title, frame, .. } => Some((title.clone(), Arc::clone(frame))),
        _ => None,
    });
    out.push((
        "the first commit of an acknowledged window opens it, under its title",
        alive && opened.as_ref().map(|(t, _)| t.as_str()) == Some("hello"),
    ));
    out.push((
        "and its pixels are the buffer's, alpha dropped, in rows of the buffer's stride",
        opened.as_ref().is_some_and(|(_, f)| {
            let f = f.lock_irq();
            f.w == 4 && f.h == 2 && f.px[0] == 0 && f.px[5] == 0x50_5050 && f.px[7] == 0x70_7070
        }),
    ));
    let ev = c.take_out();
    out.push(("the buffer is released at once, having been copied", count_events(&ev, 8, 0) == 1));
    c.fire_frames(1234);
    let ev = c.take_out();
    out.push((
        "a committed frame callback fires, and its id is given back",
        count_events(&ev, 11, 0) == 1 && count_events(&ev, 1, EV_DELETE_ID) == 1,
    ));

    // A second commit updates the same window rather than opening another.
    send(&mut c, request(7, 1, |w| { w.object(8); w.int(0); w.int(0); }));
    send(&mut c, request(7, 6, |_| {}));
    out.push((
        "a later commit updates the window it already has",
        c.take_effects().is_empty() && opened.as_ref().is_some_and(|(_, f)| f.lock_irq().commits == 2),
    ));
    send(&mut c, request(10, 0, |_| {}));
    out.push((
        "destroying the toplevel closes its window",
        matches!(c.take_effects().as_slice(), [Effect::Close { .. }]),
    ));
    out
}

/// How many events in a run of bytes went to one object with one opcode.
fn count_events(bytes: &[u8], object: u32, opcode: u16) -> usize {
    let mut at = 0;
    let mut n = 0;
    while let Ok(Some(m)) = super::wire::frame(&bytes[at..]) {
        if m.object == object && m.opcode == opcode {
            n += 1;
        }
        at += m.whole();
    }
    n
}

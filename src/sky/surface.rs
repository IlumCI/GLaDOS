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

/// Damage rectangles kept per commit before it is treated as all of it.
const DAMAGE_CAP: usize = 16;
/// How long a frame callback for an unpainted window waits before it fires
/// anyway (`Client::fire_frames`).
const HIDDEN_MS: u64 = 1000;
pub const XDG_WM_BASE: &str = "xdg_wm_base";
pub const XDG_POSITIONER: &str = "xdg_positioner";
pub const XDG_SURFACE: &str = "xdg_surface";
pub const XDG_TOPLEVEL: &str = "xdg_toplevel";
pub const XDG_POPUP: &str = "xdg_popup";
pub const WL_SUBCOMPOSITOR: &str = "wl_subcompositor";
pub const WL_SUBSURFACE: &str = "wl_subsurface";

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
            | WL_SUBCOMPOSITOR
            | WL_SUBSURFACE
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
    /// The commit the desktop last painted. Frame callbacks wait for it
    /// (`Client::fire_frames`): "draw your next frame now" is only true once
    /// the last one has been shown.
    pub presented: u64,
    /// Pixels copied out of client buffers over this window's life, which a
    /// claim reads to tell a damage-limited commit from a whole one.
    pub copied: u64,
    /// Whether the desktop last drew this window focused, which is when the
    /// seat hands it the keyboard.
    pub focused: bool,
}

pub type Shared = Arc<Spin<Frame>>;

/// What a request asks of the desktop.
pub enum Effect {
    Open { key: u64, title: String, frame: Shared },
    Retitle { key: u64, title: String },
    Fullscreen { key: u64, on: bool },
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

/// A surface's own committed picture, alpha kept: what a popup or a
/// subsurface contributes to its window, and the toplevel's base under them.
/// Premultiplied ARGB, as `wl_shm` defines both formats (XRGB is opaque).
#[derive(Clone, Default)]
struct Content {
    w: u32,
    h: u32,
    px: Vec<u32>,
}

#[derive(Default)]
struct Surface {
    /// What `attach` said since the last commit: `Some(None)` is an explicit
    /// detach, which is different from not having been told anything.
    pending: Option<Option<u32>>,
    frames: Vec<u32>,
    /// What `damage`/`damage_buffer` said since the last commit, in buffer
    /// pixels (one output, scale one, untransformed, so the two agree). More
    /// than `DAMAGE_CAP` rectangles, or one that does not fit, is everything.
    damage: Vec<(u32, u32, u32, u32)>,
    damage_all: bool,
    /// The xdg_surface that gave this its role, if one has.
    xdg: Option<u32>,
    /// Its last committed picture, kept when it is drawn into a window
    /// rather than being one: a popup, a subsurface, or a toplevel that has
    /// either over it.
    content: Option<Content>,
}

#[derive(Default)]
struct XdgSurface {
    surface: u32,
    toplevel: Option<u32>,
    popup: Option<u32>,
    /// The serial last sent, and whether any configure has been acknowledged.
    sent: u32,
    acked: bool,
    /// `set_window_geometry`: the part of the surface that is the window,
    /// which excludes the shadow a client draws round a popup. Placement is
    /// of this rectangle, not of the surface.
    geom: Option<(i32, i32, i32, i32)>,
}

/// What an `xdg_positioner` has been told. Every field the protocol has; the
/// defaults are the protocol's.
#[derive(Clone, Copy, Default)]
struct Positioner {
    size: (i32, i32),
    anchor_rect: (i32, i32, i32, i32),
    anchor: u32,
    gravity: u32,
    adjust: u32,
    offset: (i32, i32),
}

/// A menu, a dropdown, a tooltip: a surface placed against its parent and
/// drawn into the same window, above everything else in it.
struct Popup {
    xdg: u32,
    /// The xdg_surface it hangs off: a toplevel's, or another popup's.
    parent: u32,
    /// Its geometry rectangle, relative to the parent's geometry origin.
    rect: (i32, i32, i32, i32),
    grab: bool,
    mapped: bool,
    /// Creation order, which is stacking order: a submenu is above its menu.
    seq: u64,
}

/// A surface positioned inside its parent, as a video player puts its
/// picture inside its window and a toolkit its widgets.
///
/// **Desynchronised always**, which the protocol allows a compositor to
/// approximate: a child's commit is drawn at once rather than held for its
/// parent's next one. The only difference a client can see is a frame in
/// which the child is ahead of the parent, which every client already
/// tolerates when it asks for desync.
struct Sub {
    object: u32,
    parent: u32,
    pos: (i32, i32),
    pending_pos: Option<(i32, i32)>,
    mapped: bool,
    seq: u64,
}

/// Frames committed by every client, ever: what a frame rate is read from.
pub static FRAMES: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

struct Toplevel {
    xdg: u32,
    title: String,
    /// The desktop window, once the first buffer made one.
    window: Option<(u64, Shared)>,
    /// Asked for full screen, possibly before there was a window to put
    /// there -- SDL asks before its first commit.
    full: bool,
}

/// Everything below `wl_registry`, on one connection.
#[derive(Default)]
pub struct Scene {
    pools: BTreeMap<u32, Pool>,
    buffers: BTreeMap<u32, Buffer>,
    surfaces: BTreeMap<u32, Surface>,
    xdg: BTreeMap<u32, XdgSurface>,
    toplevels: BTreeMap<u32, Toplevel>,
    positioners: BTreeMap<u32, Positioner>,
    popups: BTreeMap<u32, Popup>,
    /// By the child's wl_surface.
    subs: BTreeMap<u32, Sub>,
    next_seq: u64,
    /// Frame callbacks committed and not yet fired: the callback, the window
    /// whose painting it waits for (none for a surface with no window), the
    /// commit it followed, and when it was committed.
    due: Vec<(u32, Option<Shared>, u64, u64)>,
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
            WL_SUBCOMPOSITOR => self.subcompositor(msg),
            WL_SUBSURFACE => self.subsurface(msg),
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
            // damage, damage_buffer: what changed, which is all a commit then
            // copies. With one output at scale one and no transform, surface
            // and buffer coordinates are the same pixels.
            2 | 9 => {
                let (x, y, w, h) = (r.int()?, r.int()?, r.int()?, r.int()?);
                done(&r)?;
                if let Some(s) = self.scene.surfaces.get_mut(&id) {
                    if x < 0 || y < 0 || w <= 0 || h <= 0 || s.damage.len() >= DAMAGE_CAP {
                        // Negative origins and INT32_MAX extents are how clients
                        // say "all of it"; more than the cap is as good as.
                        s.damage_all |= w > 0 && h > 0 || s.damage.len() >= DAMAGE_CAP;
                    } else {
                        s.damage.push((x as u32, y as u32, w as u32, h as u32));
                    }
                }
                Ok(())
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
        let now = crate::net::now_ms();
        let Some(s) = self.scene.surfaces.get_mut(&id) else { return Err(Error::NoObject) };
        let pending = s.pending.take();
        let frames = core::mem::take(&mut s.frames);
        let damage = core::mem::take(&mut s.damage);
        let damage_all = core::mem::take(&mut s.damage_all);
        let xdg = s.xdg;
        let window = xdg
            .and_then(|x| self.scene.xdg.get(&x))
            .and_then(|x| x.toplevel)
            .and_then(|t| self.scene.toplevels.get(&t))
            .and_then(|tl| tl.window.as_ref().map(|(_, f)| Arc::clone(f)));
        // Each callback waits for this commit to be painted, so it is queued
        // against the count this commit will leave -- one more if it carries
        // a buffer, the same if it does not.
        let next = window.as_ref().map_or(0, |f| f.lock_irq().commits)
            + matches!(pending, Some(Some(_))) as u64;
        self.scene.due.extend(frames.into_iter().map(|cb| (cb, window.clone(), next, now)));

        // A parent's commit is when its subsurfaces' new positions apply.
        let mut moved = false;
        for sub in self.scene.subs.values_mut() {
            if sub.parent == id {
                if let Some(p) = sub.pending_pos.take() {
                    moved |= sub.pos != p;
                    sub.pos = p;
                }
            }
        }

        // **A popup or a subsurface is drawn into its toplevel's window**,
        // not opened as one: its picture is kept, with its alpha, and the
        // window is composed again.
        let popup = xdg.and_then(|x| self.scene.xdg.get(&x)).and_then(|x| x.popup.map(|p| (p, x.acked)));
        let is_sub = self.scene.subs.contains_key(&id);
        if popup.is_some() || is_sub {
            let root = self.root_toplevel(id);
            match pending {
                Some(Some(buffer)) => {
                    if let Some((_, false)) = popup {
                        return Err(Error::NotConfigured);
                    }
                    let b = self.scene.buffers.get(&buffer).cloned().ok_or(Error::NoObject)?;
                    let c = copy_alpha(&b)?;
                    if let Some(s) = self.scene.surfaces.get_mut(&id) {
                        s.content = Some(c);
                    }
                    if let Some((p, _)) = popup {
                        if let Some(pp) = self.scene.popups.get_mut(&p) {
                            pp.mapped = true;
                        }
                    }
                    if let Some(sub) = self.scene.subs.get_mut(&id) {
                        sub.mapped = true;
                    }
                    self.post(Writer::new(buffer, 0));
                }
                Some(None) => {
                    if let Some(s) = self.scene.surfaces.get_mut(&id) {
                        s.content = None;
                    }
                    if let Some((p, _)) = popup {
                        if let Some(pp) = self.scene.popups.get_mut(&p) {
                            pp.mapped = false;
                        }
                    }
                    if let Some(sub) = self.scene.subs.get_mut(&id) {
                        sub.mapped = false;
                    }
                }
                None if !moved => return Ok(()),
                None => {}
            }
            if let Some(t) = root {
                self.compose(t);
            }
            return Ok(());
        }
        // A toplevel with something drawn over it keeps its own picture
        // apart and is composed; one with nothing takes the direct path below.
        if let Some(t) = xdg.and_then(|x| self.scene.xdg.get(&x)).and_then(|x| x.toplevel) {
            let has_kids = !self.children(t).is_empty();
            if has_kids {
                if let Some(Some(buffer)) = pending {
                    let b = self.scene.buffers.get(&buffer).cloned().ok_or(Error::NoObject)?;
                    let c = copy_alpha(&b)?;
                    if let Some(s) = self.scene.surfaces.get_mut(&id) {
                        s.content = Some(c);
                    }
                    self.post(Writer::new(buffer, 0));
                    self.compose(t);
                    return Ok(());
                }
                if moved {
                    self.compose(t);
                }
            } else if moved {
                self.compose(t);
            }
        }

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
        let Some((t, _)) = toplevel else {
            self.post(Writer::new(buffer, 0));
            return Ok(());
        };
        let Some(tl) = self.scene.toplevels.get_mut(&t) else { return Ok(()) };
        match &tl.window {
            // Same size and told what changed: only that is copied. A commit
            // that said nothing about damage is taken as all of it -- the
            // protocol's word for that is "nothing changed", and believing a
            // client that forgot is a window that never updates.
            Some((_, f)) if !damage_all && !damage.is_empty() && {
                let g = f.lock_irq();
                g.w == b.w && g.h == b.h
            } => {
                let mut g = f.lock_irq();
                let mut n = 0u64;
                for d in &damage {
                    n += copy_rect(&b, &mut g.px, *d)?;
                }
                g.copied += n;
                g.commits += 1;
                FRAMES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            }
            Some((_, f)) => {
                let px = copy_pixels(&b)?;
                let mut f = f.lock_irq();
                f.w = b.w;
                f.h = b.h;
                f.copied += px.len() as u64;
                f.px = px;
                f.commits += 1;
                FRAMES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            }
            None => {
                let px = copy_pixels(&b)?;
                let key = NEXT_KEY.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                let n = px.len() as u64;
                let frame = Arc::new(Spin::new(Frame {
                    w: b.w,
                    h: b.h,
                    px,
                    title: tl.title.clone(),
                    commits: 1,
                    presented: 0,
                    copied: n,
                    focused: false,
                }));
                tl.window = Some((key, Arc::clone(&frame)));
                let full = tl.full;
                self.scene.effects.push(Effect::Open { key, title: tl.title.clone(), frame });
                if full {
                    self.scene.effects.push(Effect::Fullscreen { key, on: true });
                }
            }
        }
        // Copied, so the client may have its buffer back now.
        self.post(Writer::new(buffer, 0));
        Ok(())
    }

    /// Fire the frame callbacks whose commit the desktop has painted.
    ///
    /// "Done" means the frame was shown and now is a good time to draw the
    /// next. It fired as soon as the commit was *taken*, so a client
    /// animating to callbacks rendered as fast as the server could read, and
    /// every frame past the screen's rate was drawn, copied and discarded.
    ///
    /// A window that is never painted -- minimised, or not yet opened -- would
    /// then never hear back, and a client blocked on its callback would stop.
    /// Wayland compositors throttle a hidden surface rather than freeze it, so
    /// after `HIDDEN_MS` a callback fires anyway: once a second.
    pub fn fire_frames(&mut self, now_ms: u64) {
        let due = core::mem::take(&mut self.scene.due);
        let mut keep = Vec::new();
        for (cb, window, commit, at) in due {
            if self.objects.get(cb).is_none() {
                continue;
            }
            let shown = window.as_ref().map_or(true, |f| f.lock_irq().presented >= commit);
            if !shown && now_ms.saturating_sub(at) < HIDDEN_MS {
                keep.push((cb, window, commit, at));
                continue;
            }
            let mut w = Writer::new(cb, 0);
            w.uint(now_ms as u32);
            self.post(w);
            let _ = self.forget(cb);
        }
        self.scene.due = keep;
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

    /// Positioners place popups: a size, a rectangle on the parent to place
    /// against, which edge of each meets, an offset, and what the compositor
    /// may do when the result does not fit.
    fn positioner(&mut self, msg: &Message) -> Result<(), Error> {
        let id = msg.object;
        let mut r = msg.reader();
        let p = self.scene.positioners.entry(id).or_default();
        match msg.opcode {
            0 => {
                done(&r)?;
                self.scene.positioners.remove(&id);
                return self.forget(id);
            }
            1 => p.size = (r.int()?, r.int()?),
            2 => p.anchor_rect = (r.int()?, r.int()?, r.int()?, r.int()?),
            3 => p.anchor = r.uint()?,
            4 => p.gravity = r.uint()?,
            5 => p.adjust = r.uint()?,
            6 => p.offset = (r.int()?, r.int()?),
            // set_reactive, set_parent_size, set_parent_configure: hints for
            // a compositor that repositions on its own, which this one does
            // not do after the first placement.
            7 => {}
            8 => {
                r.int()?;
                r.int()?;
            }
            9 => {
                r.uint()?;
            }
            _ => return Err(Error::NoMethod),
        }
        done(&r)
    }

    /// Tell a toplevel whether it has the keyboard, through its configure
    /// states, which is how a client learns it is the active window.
    ///
    /// **GTK reads activity from `activated` and from nothing else.** Every
    /// key reached LibreOffice, `wl_keyboard.enter` included, and none was
    /// typed: GTK took the window for a background one, since the first
    /// configure carries no states and nothing ever sent another. Zero by
    /// zero keeps the size the client chose; a full-screen window keeps the
    /// screen's.
    pub(super) fn activate(&mut self, surface: u32, on: bool) {
        let Some((&xdg, x)) = self.scene.xdg.iter().find(|(_, x)| x.surface == surface) else { return };
        let Some(tid) = x.toplevel else { return };
        let Some(t) = self.scene.toplevels.get(&tid) else { return };
        let (w, h) = if t.full {
            crate::gfx::primary().map(|f| (f.width() as i32, f.height() as i32)).unwrap_or((0, 0))
        } else {
            (0, 0)
        };
        let mut states: Vec<u8> = Vec::new();
        if t.full {
            states.extend_from_slice(&2u32.to_le_bytes());
        }
        if on {
            states.extend_from_slice(&4u32.to_le_bytes());
        }
        let mut c = Writer::new(tid, 0);
        c.int(w);
        c.int(h);
        c.array(&states);
        self.post(c);
        self.configure(xdg);
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
                self.scene.toplevels.insert(tid, Toplevel { xdg: id, title: String::from("Linux"), window: None, full: false });
                // Zero by zero is "choose your own size", and no states.
                let mut w = Writer::new(tid, 0);
                w.int(0);
                w.int(0);
                w.array(&[]);
                self.post(w);
                self.configure(id);
                Ok(())
            }
            // get_popup: placed against its parent by its positioner and
            // kept inside the window, which is the area the desktop draws.
            // One with no parent (a layer-shell client's) has nothing to be
            // drawn into here and is dismissed at once.
            2 => {
                let pid = r.new_id()?;
                let parent = r.object()?;
                let positioner = r.object()?;
                done(&r)?;
                let x = self.scene.xdg.get_mut(&id).ok_or(Error::NoObject)?;
                if x.toplevel.is_some() || x.popup.is_some() {
                    return Err(Error::Role);
                }
                x.popup = Some(pid);
                self.objects.create(pid, XDG_POPUP, 1)?;
                let pos = self.scene.positioners.get(&positioner).copied().ok_or(Error::NoObject)?;
                let parent_surface = self.scene.xdg.get(&parent).map(|p| p.surface);
                let placed = parent_surface.and_then(|ps| {
                    let origin = self.origin_of(ps)?;
                    let (gx, gy) = self.geom_off(parent);
                    let (bw, bh) = self.root_size(ps)?;
                    // The window, in the parent's geometry coordinates.
                    let bounds = (-(origin.0 + gx), -(origin.1 + gy), bw as i32, bh as i32);
                    Some(place(&pos, bounds))
                });
                let Some(rect) = placed else {
                    self.post(Writer::new(pid, 1)); // popup_done
                    return Ok(());
                };
                self.scene.next_seq += 1;
                let seq = self.scene.next_seq;
                self.scene.popups.insert(pid, Popup { xdg: id, parent, rect, grab: false, mapped: false, seq });
                let mut c = Writer::new(pid, 0);
                c.int(rect.0).int(rect.1).int(rect.2).int(rect.3);
                self.post(c);
                self.configure(id);
                Ok(())
            }
            // set_window_geometry
            3 => {
                let g = (r.int()?, r.int()?, r.int()?, r.int()?);
                done(&r)?;
                if let Some(x) = self.scene.xdg.get_mut(&id) {
                    x.geom = Some(g);
                }
                Ok(())
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
            // set_fullscreen (11) and unset_fullscreen (12): the desktop
            // gives the window the screen, and the client is told the
            // screen's size in a configure, which is what it renders at.
            11 | 12 => {
                let on = msg.opcode == 11;
                if on {
                    let _output = r.object()?;
                }
                done(&r)?;
                let t = self.scene.toplevels.get_mut(&id).ok_or(Error::NoObject)?;
                t.full = on;
                let xdg = t.xdg;
                if let Some((key, _)) = &t.window {
                    self.scene.effects.push(Effect::Fullscreen { key: *key, on });
                }
                let (w, h) = if on {
                    crate::gfx::primary().map(|f| (f.width() as i32, f.height() as i32)).unwrap_or((0, 0))
                } else {
                    (0, 0)
                };
                // xdg_toplevel states: 2 fullscreen, 4 activated.
                let mut c = Writer::new(id, 0);
                c.int(w);
                c.int(h);
                let states: &[u8] = if on { &[2, 0, 0, 0, 4, 0, 0, 0] } else { &[4, 0, 0, 0] };
                c.array(states);
                self.post(c);
                self.configure(xdg);
                Ok(())
            }
            // set_parent, set_app_id, show_window_menu, move, resize, the size
            // hints, and the state requests: accepted. The desktop's own frame
            // moves and sizes a window, and the states it has no way to take
            // are states a client asking for them can live without.
            1 | 3..=10 | 13 => Ok(()),
            _ => Err(Error::NoMethod),
        }
    }

    fn popup(&mut self, msg: &Message) -> Result<(), Error> {
        let id = msg.object;
        let mut r = msg.reader();
        match msg.opcode {
            0 => {
                done(&r)?;
                if let Some(p) = self.scene.popups.remove(&id) {
                    if let Some(x) = self.scene.xdg.get_mut(&p.xdg) {
                        x.popup = None;
                        let surface = x.surface;
                        if let Some(s) = self.scene.surfaces.get_mut(&surface) {
                            s.content = None;
                        }
                        if let Some(t) = self.root_toplevel(surface) {
                            self.compose(t);
                        }
                    }
                }
                self.forget(id)
            }
            // grab(seat, serial): a click outside the popups ends them.
            1 => {
                r.object()?;
                r.uint()?;
                done(&r)?;
                if let Some(p) = self.scene.popups.get_mut(&id) {
                    p.grab = true;
                }
                Ok(())
            }
            // reposition (v3) is not offered: the global is version 1.
            _ => Err(Error::NoMethod),
        }
    }

    fn subcompositor(&mut self, msg: &Message) -> Result<(), Error> {
        let mut r = msg.reader();
        match msg.opcode {
            0 => {
                done(&r)?;
                self.forget(msg.object)
            }
            // get_subsurface(id, surface, parent)
            1 => {
                let id = r.new_id()?;
                let surface = r.object()?;
                let parent = r.object()?;
                done(&r)?;
                if surface == parent || !self.scene.surfaces.contains_key(&surface) || !self.scene.surfaces.contains_key(&parent) {
                    return Err(Error::NoObject);
                }
                if self.scene.subs.contains_key(&surface) || self.scene.surfaces.get(&surface).is_some_and(|s| s.xdg.is_some()) {
                    return Err(Error::Role);
                }
                self.objects.create(id, WL_SUBSURFACE, 1)?;
                self.scene.next_seq += 1;
                let seq = self.scene.next_seq;
                self.scene.subs.insert(surface, Sub { object: id, parent, pos: (0, 0), pending_pos: None, mapped: false, seq });
                Ok(())
            }
            _ => Err(Error::NoMethod),
        }
    }

    fn subsurface(&mut self, msg: &Message) -> Result<(), Error> {
        let obj = msg.object;
        let mut r = msg.reader();
        let child = self.scene.subs.iter().find(|(_, s)| s.object == obj).map(|(k, _)| *k);
        match msg.opcode {
            0 => {
                done(&r)?;
                if let Some(c) = child {
                    let root = self.root_toplevel(c);
                    self.scene.subs.remove(&c);
                    if let Some(s) = self.scene.surfaces.get_mut(&c) {
                        s.content = None;
                    }
                    if let Some(t) = root {
                        self.compose(t);
                    }
                }
                self.forget(obj)
            }
            // set_position: takes effect at the parent's next commit.
            1 => {
                let p = (r.int()?, r.int()?);
                done(&r)?;
                if let Some(s) = child.and_then(|c| self.scene.subs.get_mut(&c)) {
                    s.pending_pos = Some(p);
                }
                Ok(())
            }
            // place_above, place_below: stacking among siblings is kept in
            // creation order here, which is what nearly every client creates
            // them in anyway.
            2 | 3 => {
                r.object()?;
                done(&r)
            }
            // set_sync, set_desync: see `Sub`.
            4 | 5 => done(&r),
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

    /// The `wl_surface` behind a desktop window, for delivering input to it.
    pub fn surface_of(&self, key: u64) -> Option<u32> {
        let tl = self.scene.toplevels.values().find(|t| t.window.as_ref().is_some_and(|(k, _)| *k == key))?;
        self.scene.xdg.get(&tl.xdg).map(|x| x.surface)
    }

    /// The desktop window a `wl_surface` is the content of, if any.
    pub fn key_of(&self, surface: u32) -> Option<u64> {
        self.scene.toplevels.values().find_map(|t| {
            let (k, _) = t.window.as_ref()?;
            (self.scene.xdg.get(&t.xdg)?.surface == surface).then_some(*k)
        })
    }

    pub fn take_effects(&mut self) -> Vec<Effect> {
        core::mem::take(&mut self.scene.effects)
    }

    /// The toplevel a surface is drawn into: itself, or through popup parents
    /// and subsurface parents. Bounded, since a client chooses the links.
    fn root_toplevel(&self, mut surface: u32) -> Option<u32> {
        for _ in 0..64 {
            if let Some(sub) = self.scene.subs.get(&surface) {
                surface = sub.parent;
                continue;
            }
            let x = self.scene.surfaces.get(&surface)?.xdg?;
            let xs = self.scene.xdg.get(&x)?;
            if let Some(t) = xs.toplevel {
                return Some(t);
            }
            let p = self.scene.popups.get(&xs.popup?)?;
            surface = self.scene.xdg.get(&p.parent)?.surface;
        }
        None
    }

    /// Where a surface's top-left sits in its toplevel's surface, in pixels.
    fn origin_of(&self, surface: u32) -> Option<(i32, i32)> {
        let mut at = (0, 0);
        let mut cur = surface;
        for _ in 0..64 {
            if let Some(sub) = self.scene.subs.get(&cur) {
                at = (at.0 + sub.pos.0, at.1 + sub.pos.1);
                cur = sub.parent;
                continue;
            }
            let x = self.scene.surfaces.get(&cur)?.xdg?;
            let xs = self.scene.xdg.get(&x)?;
            if xs.toplevel.is_some() {
                return Some(at);
            }
            let p = self.scene.popups.get(&xs.popup?)?;
            // The popup's geometry sits at `rect` from its parent's geometry
            // origin; its surface is up and left of that by its own offset.
            let (gx, gy) = self.geom_off(x);
            let (pgx, pgy) = self.geom_off(p.parent);
            at = (at.0 + p.rect.0 + pgx - gx, at.1 + p.rect.1 + pgy - gy);
            cur = self.scene.xdg.get(&p.parent)?.surface;
        }
        None
    }

    fn geom_off(&self, xdg: u32) -> (i32, i32) {
        self.scene.xdg.get(&xdg).and_then(|x| x.geom).map_or((0, 0), |g| (g.0, g.1))
    }

    /// The size of the window a surface is drawn into.
    fn root_size(&self, surface: u32) -> Option<(u32, u32)> {
        let t = self.root_toplevel(surface)?;
        let tl = self.scene.toplevels.get(&t)?;
        if let Some((_, f)) = &tl.window {
            let g = f.lock_irq();
            return Some((g.w, g.h));
        }
        None
    }

    /// The toplevel's own surface.
    fn toplevel_surface(&self, t: u32) -> Option<u32> {
        let tl = self.scene.toplevels.get(&t)?;
        self.scene.xdg.get(&tl.xdg).map(|x| x.surface)
    }

    /// Everything drawn over a toplevel, bottom first: its subsurfaces, then
    /// each popup with its own subsurfaces above it. Mapped and with a
    /// picture, with where each goes.
    fn children(&self, t: u32) -> Vec<(u32, (i32, i32))> {
        let mut out: Vec<((u64, u64), u32)> = Vec::new();
        for (surface, sub) in &self.scene.subs {
            if !sub.mapped || self.root_toplevel(*surface) != Some(t) {
                continue;
            }
            // A subsurface stacks with the nearest popup it belongs to.
            let mut layer = 0;
            let mut cur = sub.parent;
            for _ in 0..64 {
                if let Some(s2) = self.scene.subs.get(&cur) {
                    cur = s2.parent;
                    continue;
                }
                if let Some(pp) = self.scene.surfaces.get(&cur).and_then(|s| s.xdg).and_then(|x| self.scene.xdg.get(&x)).and_then(|x| x.popup) {
                    layer = self.scene.popups.get(&pp).map_or(0, |p| p.seq);
                }
                break;
            }
            out.push(((layer, sub.seq), *surface));
        }
        for p in self.scene.popups.values() {
            let Some(surface) = self.scene.xdg.get(&p.xdg).map(|x| x.surface) else { continue };
            if p.mapped && self.root_toplevel(surface) == Some(t) {
                out.push(((p.seq, 0), surface));
            }
        }
        out.sort_by_key(|(k, _)| *k);
        out.into_iter()
            .filter(|(_, sf)| self.scene.surfaces.get(sf).is_some_and(|s| s.content.is_some()))
            .filter_map(|(_, sf)| Some((sf, self.origin_of(sf)?)))
            .collect()
    }

    /// Draw a toplevel's window again: its own picture, and every child over
    /// it. A window with no children leaves its own picture in place and
    /// goes back to the direct path in `commit`.
    fn compose(&mut self, t: u32) {
        let Some(root) = self.toplevel_surface(t) else { return };
        let Some(frame) = self.scene.toplevels.get(&t).and_then(|tl| tl.window.as_ref().map(|(_, f)| Arc::clone(f))) else { return };
        let kids = self.children(t);
        let mut f = frame.lock_irq();
        // The toplevel's own picture is kept apart once anything is drawn
        // over it, since the window then holds a mixture of the two.
        let base = match self.scene.surfaces.get_mut(&root) {
            Some(s) => {
                if s.content.is_none() {
                    s.content = Some(Content { w: f.w, h: f.h, px: f.px.iter().map(|p| p | 0xFF00_0000).collect() });
                }
                s.content.clone().unwrap()
            }
            None => return,
        };
        let mut px: Vec<u32> = base.px.iter().map(|p| p & 0x00FF_FFFF).collect();
        for (sf, (ox, oy)) in &kids {
            if let Some(c) = self.scene.surfaces.get(sf).and_then(|s| s.content.as_ref()) {
                blend(&mut px, base.w, base.h, c, *ox, *oy);
            }
        }
        f.w = base.w;
        f.h = base.h;
        f.px = px;
        f.commits += 1;
        drop(f);
        if kids.is_empty() {
            if let Some(s) = self.scene.surfaces.get_mut(&root) {
                s.content = None;
            }
        }
        FRAMES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }

    /// Which surface the pointer is over, and where in it, for a point in a
    /// window: the topmost child containing it, or the window's own surface.
    pub fn pick(&self, surface: u32, x: i32, y: i32) -> (u32, i32, i32) {
        let Some(t) = self.root_toplevel(surface) else { return (surface, x, y) };
        for (sf, (ox, oy)) in self.children(t).iter().rev() {
            if let Some(c) = self.scene.surfaces.get(sf).and_then(|s| s.content.as_ref()) {
                if x >= *ox && y >= *oy && x < ox + c.w as i32 && y < oy + c.h as i32 {
                    return (*sf, x - ox, y - oy);
                }
            }
        }
        (surface, x, y)
    }

    /// A press in a window holding grabbing popups, outside all of them, ends
    /// them -- topmost first, as the protocol orders `popup_done` -- and is
    /// not delivered: the click that closes a menu is not also a click on
    /// what is under it. Answers whether it was taken.
    pub fn dismiss_outside(&mut self, surface: u32, x: i32, y: i32) -> bool {
        let Some(t) = self.root_toplevel(surface) else { return false };
        let mut ours: Vec<(u64, u32, u32)> = self
            .scene
            .popups
            .iter()
            .filter_map(|(id, p)| {
                let sf = self.scene.xdg.get(&p.xdg)?.surface;
                (p.mapped && self.root_toplevel(sf) == Some(t)).then_some((p.seq, *id, sf))
            })
            .collect();
        if !self.scene.popups.values().any(|p| p.grab && p.mapped) || ours.is_empty() {
            return false;
        }
        let (hit, _, _) = self.pick(surface, x, y);
        if ours.iter().any(|(_, _, sf)| *sf == hit) {
            return false;
        }
        ours.sort_by_key(|(seq, _, _)| core::cmp::Reverse(*seq));
        for (_, id, sf) in ours {
            if let Some(p) = self.scene.popups.get_mut(&id) {
                p.mapped = false;
            }
            if let Some(s) = self.scene.surfaces.get_mut(&sf) {
                s.content = None;
            }
            self.post(Writer::new(id, 1)); // popup_done
        }
        self.compose(t);
        true
    }
}

/// Which way an `xdg_positioner` edge enum leans: -1 left or top, 0 the
/// centre, 1 right or bottom.
fn horiz(e: u32) -> i32 {
    match e {
        3 | 5 | 6 => -1,
        4 | 7 | 8 => 1,
        _ => 0,
    }
}
fn vert(e: u32) -> i32 {
    match e {
        1 | 5 | 7 => -1,
        2 | 6 | 8 => 1,
        _ => 0,
    }
}
fn flip_h(e: u32) -> u32 {
    match e {
        3 => 4,
        4 => 3,
        5 => 7,
        7 => 5,
        6 => 8,
        8 => 6,
        o => o,
    }
}
fn flip_v(e: u32) -> u32 {
    match e {
        1 => 2,
        2 => 1,
        5 => 6,
        6 => 5,
        7 => 8,
        8 => 7,
        o => o,
    }
}

/// Where a positioner puts its popup's geometry, before any adjustment.
fn raw_place(p: &Positioner, anchor: u32, gravity: u32) -> (i32, i32) {
    let (ax, ay, aw, ah) = p.anchor_rect;
    let (w, h) = p.size;
    let px = ax + match horiz(anchor) { -1 => 0, 1 => aw, _ => aw / 2 };
    let py = ay + match vert(anchor) { -1 => 0, 1 => ah, _ => ah / 2 };
    let x = px + match horiz(gravity) { -1 => -w, 1 => 0, _ => -w / 2 };
    let y = py + match vert(gravity) { -1 => -h, 1 => 0, _ => -h / 2 };
    (x + p.offset.0, y + p.offset.1)
}

/// Place a popup inside `bounds` (x, y, w, h, in the parent's geometry
/// coordinates), using the adjustments it allows -- flip first, then slide,
/// in that order, which is the protocol's -- and then held inside whatever
/// it allowed, since this compositor can draw a popup only inside its window.
fn place(p: &Positioner, bounds: (i32, i32, i32, i32)) -> (i32, i32, i32, i32) {
    const SLIDE_X: u32 = 1;
    const SLIDE_Y: u32 = 2;
    const FLIP_X: u32 = 4;
    const FLIP_Y: u32 = 8;
    let (bx, by, bw, bh) = bounds;
    // Larger than the window is given the window, which the protocol lets a
    // compositor do: GTK answers a menu configured shorter than it asked for
    // with scroll arrows, where one slid to the top lost its last items.
    let (w, h) = (p.size.0.max(1).min(bw.max(1)), p.size.1.max(1).min(bh.max(1)));
    let fits_x = |x: i32| x >= bx && x + w <= bx + bw;
    let fits_y = |y: i32| y >= by && y + h <= by + bh;
    let (mut x, mut y) = raw_place(p, p.anchor, p.gravity);
    if !fits_x(x) && p.adjust & FLIP_X != 0 {
        let (fx, _) = raw_place(p, flip_h(p.anchor), flip_h(p.gravity));
        if fits_x(fx) {
            x = fx;
        }
    }
    if !fits_y(y) && p.adjust & FLIP_Y != 0 {
        let (_, fy) = raw_place(p, flip_v(p.anchor), flip_v(p.gravity));
        if fits_y(fy) {
            y = fy;
        }
    }
    let _ = (SLIDE_X, SLIDE_Y);
    // Slid, whether or not it asked: a popup partly outside the window is a
    // popup partly not drawn, which is worse than one moved.
    x = x.min(bx + bw - w).max(bx);
    y = y.min(by + bh - h).max(by);
    (x, y, w, h)
}

/// One premultiplied ARGB picture over an opaque `0x00RRGGBB` one, at
/// (ox, oy), clipped to it.
fn blend(dst: &mut [u32], dw: u32, dh: u32, src: &Content, ox: i32, oy: i32) {
    let x0 = ox.max(0);
    let y0 = oy.max(0);
    let x1 = (ox + src.w as i32).min(dw as i32);
    let y1 = (oy + src.h as i32).min(dh as i32);
    for y in y0..y1 {
        let srow = ((y - oy) as u32 * src.w) as usize;
        let drow = (y as u32 * dw) as usize;
        for x in x0..x1 {
            let s = src.px[srow + (x - ox) as usize];
            let a = s >> 24;
            let d = &mut dst[drow + x as usize];
            if a == 255 {
                *d = s & 0x00FF_FFFF;
            } else if a != 0 || s & 0x00FF_FFFF != 0 {
                let inv = 255 - a;
                let ch = |sh: u32| {
                    let sc = (s >> sh) & 0xFF;
                    let dc = (*d >> sh) & 0xFF;
                    (sc + (dc * inv + 127) / 255).min(255) << sh
                };
                *d = ch(16) | ch(8) | ch(0);
            }
        }
    }
}

/// A buffer's pixels with their alpha, for a child surface. XRGB is opaque.
fn copy_alpha(b: &Buffer) -> Result<Content, Error> {
    let mut px = copy_pixels(b)?;
    if b.format == ARGB8888 {
        // `copy_pixels` dropped the alpha; read it back for these.
        let need = b.offset + b.stride * (b.h as usize - 1) + b.w as usize * 4;
        b.mem.with(|bytes| {
            if need > bytes.len() {
                return Err(Error::BadBuffer);
            }
            let w = b.w as usize;
            for y in 0..b.h as usize {
                for x in 0..w {
                    let a = bytes[b.offset + y * b.stride + x * 4 + 3] as u32;
                    px[y * w + x] |= a << 24;
                }
            }
            Ok(())
        })?;
    } else {
        for p in px.iter_mut() {
            *p |= 0xFF00_0000;
        }
    }
    Ok(Content { w: b.w, h: b.h, px })
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
/// One damaged rectangle of the buffer into `px`, which is the window's
/// picture at the buffer's own size. Clamped to the buffer, since damage is a
/// hint and a client may describe more than it has. Answers pixels copied.
fn copy_rect(b: &Buffer, px: &mut [u32], (x, y, w, h): (u32, u32, u32, u32)) -> Result<u64, Error> {
    let need = b.offset + b.stride * (b.h as usize - 1) + b.w as usize * 4;
    let x1 = x.saturating_add(w).min(b.w);
    let y1 = y.saturating_add(h).min(b.h);
    if x >= x1 || y >= y1 || px.len() < (b.w * b.h) as usize {
        return Ok(0);
    }
    b.mem.with(|bytes| {
        if need > bytes.len() {
            return Err(Error::BadBuffer);
        }
        // Word copies, as `copy_pixels`: this runs under the window's own
        // lock, which the compositor needs to paint it, and it did a bounds
        // check and four byte loads per pixel of a frame-sized damage.
        let (x, x1, bw) = (x as usize, x1 as usize, b.w as usize);
        for row in y as usize..y1 as usize {
            let line = &bytes[b.offset + row * b.stride + x * 4..b.offset + row * b.stride + x1 * 4];
            let dst = &mut px[row * bw + x..row * bw + x1];
            let src = line.as_ptr() as *const u32;
            for (i, d) in dst.iter_mut().enumerate() {
                *d = unsafe { core::ptr::read_unaligned(src.add(i)) } & 0x00FF_FFFF;
            }
        }
        Ok((x1 - x) as u64 * (y1 - y) as u64)
    })
}

fn copy_pixels(b: &Buffer) -> Result<Vec<u32>, Error> {
    // Both formats are little-endian 0xAARRGGBB words; alpha is dropped.
    let _ = b.format;
    let need = b.offset + b.stride * (b.h as usize - 1) + b.w as usize * 4;
    b.mem.with(|bytes| {
        if need > bytes.len() {
            return Err(Error::BadBuffer);
        }
        // A word at a time with the alpha masked, which the compiler turns
        // into vector loads. It assembled each pixel from four bytes and
        // pushed it, a million times a frame, under the guest lock: a 32-bit
        // game's every commit held that lock 2.1 ms for this loop.
        let (w, h) = (b.w as usize, b.h as usize);
        let mut px: Vec<u32> = Vec::with_capacity(w * h);
        for y in 0..h {
            let row = &bytes[b.offset + y * b.stride..b.offset + y * b.stride + w * 4];
            let src = row.as_ptr() as *const u32;
            // Safety: `row` is `w * 4` bytes, checked against `need` above,
            // and `px` has room for `w * h` words of which this fills the
            // next `w`. Unaligned reads, since a pool offset need not be.
            unsafe {
                let dst = px.as_mut_ptr().add(y * w);
                for i in 0..w {
                    *dst.add(i) = core::ptr::read_unaligned(src.add(i)) & 0x00FF_FFFF;
                }
            }
        }
        unsafe { px.set_len(w * h) };
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
    // Damage: one pixel said to have changed is one pixel copied.
    let copied = |o: &Option<(String, Shared)>| o.as_ref().map_or(0, |(_, f)| f.lock_irq().copied);
    let before = copied(&opened);
    send(&mut c, request(7, 1, |w| { w.object(8); w.int(0); w.int(0); }));
    send(&mut c, request(7, 2, |w| { w.int(1); w.int(0); w.int(1); w.int(1); }));
    send(&mut c, request(7, 3, |w| { w.new_id(12); }));
    send(&mut c, request(7, 6, |_| {}));
    out.push((
        "a commit that names its damage copies that and nothing else",
        copied(&opened) == before + 1
            && opened.as_ref().is_some_and(|(_, f)| f.lock_irq().commits == 3),
    ));
    let _ = c.take_out();
    c.fire_frames(crate::net::now_ms());
    out.push((
        "and its frame callback waits for the desktop to paint it",
        count_events(&c.take_out(), 12, 0) == 0,
    ));
    if let Some((_, f)) = opened.as_ref() {
        let mut f = f.lock_irq();
        f.presented = f.commits;
    }
    c.fire_frames(crate::net::now_ms());
    out.push(("then fires once it has been", count_events(&c.take_out(), 12, 0) == 1));

    // A window nobody paints still hears back, a second later.
    send(&mut c, request(7, 1, |w| { w.object(8); w.int(0); w.int(0); }));
    send(&mut c, request(7, 3, |w| { w.new_id(13); }));
    send(&mut c, request(7, 6, |_| {}));
    let _ = c.take_out();
    c.fire_frames(crate::net::now_ms());
    let held = count_events(&c.take_out(), 13, 0) == 0;
    c.fire_frames(crate::net::now_ms() + HIDDEN_MS);
    out.push((
        "a window that is not painted gets its callback anyway, once a second",
        held && count_events(&c.take_out(), 13, 0) == 1,
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

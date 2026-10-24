//! `/dev/dri/renderD128`: the GPU, as a Linux program asks for it.
//!
//! **The DRM uAPI is the seam, and everything above it is unmodified Mesa.**
//! A graphics API on Linux is a userspace library that turns its calls into
//! command streams and hands them to a kernel driver through ioctls on a
//! render node. So "a GPU in ring 3" is not a Vulkan implementation in this
//! kernel; it is this file answering the ioctls one particular driver's
//! userspace makes, and the library doing the rest at ring 3, where
//! `ring0-is-for-the-agent` says it belongs.
//!
//! The driver spoken today is `virtio_gpu`, whose userspace half is Mesa's
//! Venus: Vulkan serialised into a ring the host executes, which under QEMU is
//! the host's own GPU. The structure leaves room for a second backend --
//! nouveau, whose userspace is NVK, on the GF63's own GA107 -- behind the same
//! node, the same `/sys` description and the same buffer and fence plumbing;
//! only the ioctl table and what a submit means would differ.
//!
//! What a client gets, and in which shape:
//!
//! - **A context per open.** `CONTEXT_INIT` creates one host context typed by
//!   the capset the client names (Venus is 4), and the descriptor's close
//!   destroys it along with every buffer it made.
//! - **Buffers as GEM handles**, small integers per open, each naming a host
//!   blob. `MAP` maps one into the device's host-visible window and answers an
//!   offset that `mmap` on the same descriptor turns into guest pages over it.
//! - **Fences as sync files.** An `EXECBUFFER` asking for `FENCE_FD_OUT` gets a
//!   descriptor that polls readable once the host has finished the work, which
//!   is what Mesa's simulated syncobj waits on when the kernel offers no
//!   timeline syncobj -- and this one offers none, deliberately: a syncobj is a
//!   kernel object table and an ioctl family of its own, and a sync file is a
//!   descriptor that already exists.
//!
//! Layouts are `include/uapi/drm/virtgpu_drm.h` and `drm.h` (MIT), read for
//! the numbers; nothing was copied.

use super::syscall::{self, EBADF, EFAULT, EINVAL, ENODEV, ENOMEM, ENOTTY};
use crate::gpu::virtgpu;
use crate::sync::Racy;
use alloc::string::String;
use alloc::vec::Vec;

/// **Not 226, and the reason is a lie somebody else tells.** Venus running on
/// NVIDIA's driver hides its DRM node -- `vn_wsi_init` reports render 0:0 and
/// switches `VK_EXT_physical_device_drm` off, so that common WSI takes its
/// blit path -- and Zink, handed a render-node fd by EGL, will accept only the
/// Vulkan device whose DRM node matches that fd's `st_rdev` -- so with 226:128
/// no device can match (read from `zink_get_display_device`, not yet watched
/// failing). Zink does that matching only for a
/// major in 1..255 (`zink_picks_device`), and outside that range it takes the
/// first device, which is the only one there is.
///
/// Nothing else is told anything different. On Linux libdrm finds a node
/// through `/sys/dev/char/<major>:<minor>` and never compares the major
/// against 226 -- that test is for the BSDs -- so enumeration, the PCI ids and
/// the node's name all come out as before.
pub const MAJOR: u64 = 511;
pub const RENDER_MINOR: u64 = 128;
pub const NODE: &str = "/dev/dri/renderD128";

/// `makedev(MAJOR, 128)` as glibc encodes it: the low eight bits of the minor
/// at the bottom, the major above them, and the rest of the minor at 20.
/// libdrm takes it apart with `major()` and `minor()`, so an encoding that is
/// merely plausible names a different `/sys/dev/char` directory.
pub const RDEV: u64 = ((MAJOR & 0xfff) << 8) | (RENDER_MINOR & 0xff) | ((RENDER_MINOR & !0xff) << 12);

const SYS_DEV: &str = "/sys/dev/char/511:128";
const SYS_DEVICE: &str = "/sys/dev/char/511:128/device";

// ioctl numbers, `_IOC(dir, 'd', nr, size)`. Matched on the type byte and the
// number, never on the size field: a 32-bit and a 64-bit client encode a
// different size for `drm_version`, and both have to land here.
const DRM_TYPE: u64 = 0x64;
const NR_VERSION: u64 = 0x00;
const NR_GEM_CLOSE: u64 = 0x09;
const NR_GET_CAP: u64 = 0x0c;
const NR_SET_CLIENT_CAP: u64 = 0x0d;
const NR_PRIME_HANDLE_TO_FD: u64 = 0x2d;
const NR_PRIME_FD_TO_HANDLE: u64 = 0x2e;
const NR_VIRTGPU_MAP: u64 = 0x41;
const NR_VIRTGPU_EXECBUFFER: u64 = 0x42;
const NR_VIRTGPU_GETPARAM: u64 = 0x43;
const NR_VIRTGPU_RESOURCE_CREATE: u64 = 0x44;
const NR_VIRTGPU_RESOURCE_INFO: u64 = 0x45;
const NR_VIRTGPU_TRANSFER_FROM_HOST: u64 = 0x46;
const NR_VIRTGPU_TRANSFER_TO_HOST: u64 = 0x47;
const NR_VIRTGPU_WAIT: u64 = 0x48;
const NR_VIRTGPU_GET_CAPS: u64 = 0x49;
const NR_VIRTGPU_RESOURCE_CREATE_BLOB: u64 = 0x4a;
const NR_VIRTGPU_CONTEXT_INIT: u64 = 0x4b;

const CAP_PRIME: u64 = 0x05;
const CAP_SYNCOBJ: u64 = 0x13;
const CAP_SYNCOBJ_TIMELINE: u64 = 0x14;

const PARAM_3D_FEATURES: u64 = 1;
const PARAM_CAPSET_QUERY_FIX: u64 = 2;
const PARAM_RESOURCE_BLOB: u64 = 3;
const PARAM_HOST_VISIBLE: u64 = 4;
const PARAM_CROSS_DEVICE: u64 = 5;
const PARAM_CONTEXT_INIT: u64 = 6;
const PARAM_SUPPORTED_CAPSET_IDS: u64 = 7;

const EXECBUF_FENCE_FD_IN: u32 = 1;
const EXECBUF_FENCE_FD_OUT: u32 = 2;
const EXECBUF_RING_IDX: u32 = 4;

const CTX_PARAM_CAPSET_ID: u64 = 1;
const CTX_PARAM_NUM_RINGS: u64 = 2;

/// An offset `MAP` hands out is the handle shifted up by this much, so `mmap`
/// can find the buffer from the offset alone.
///
/// **It was 32, and that killed a game.** An i386 client's `mmap2` carries
/// the offset in pages as a 32-bit number, so a shift of 32 reached handle
/// 4095 -- which read as "far more buffers than a client keeps mapped" and
/// was a cap on buffers ever *created*, handles only counting up. a game
/// passes 4096 while loading a level; glibc then refuses the `mmap` itself
/// with `EINVAL` before the kernel sees it, Zink reports
/// VK_ERROR_MEMORY_MAP_FAILED and dereferences the mapping it did not get.
/// Two of three runs ended there. Handles are reused now (`free_handle`), so
/// the number tracks live buffers, and 24 reaches 2^20 of them -- a buffer
/// is mapped whole and never at an offset inside it, so the low 24 bits are
/// never needed.
const MAP_SHIFT: u32 = 24;

struct Bo {
    handle: u32,
    res: u32,
    size: u64,
    blob_mem: u32,
    /// Where `MAP` put it in the host-visible window, once it has.
    phys: Option<u64>,
    /// How the host says the mapping may be cached: virtio-gpu's map info,
    /// 1 cached, 2 uncached, 3 write-combined.
    cache: u32,
}

struct Client {
    id: usize,
    ctx: Option<u32>,
    capset: u32,
    bos: Vec<Bo>,
}

/// One per open render node. Racy because every caller is a guest syscall,
/// and those run under the guest lock.
static CLIENTS: Racy<Vec<Client>> = Racy::new(Vec::new());
/// `gpu drm idle <us>`: rewrites the idle timeout in Venus's
/// `vkCreateRingMESA` (command 188, a u64 of nanoseconds at byte 76, sent by
/// Mesa as 1 ms). An experiment kept as a knob: raising it to 50 ms got
/// the game further and did not stop the stalls, which is what put the
/// watchdog below in place of it. Zero leaves the command as written.
pub static RING_IDLE_US: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
/// `gpu drm trace on`: a line per mapping.
pub static TRACE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
/// The cache type the host gave the last mapping, for `gpu drm`.
pub static LAST_CACHE: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
static NEXT: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(1);

/// The device behind the node, found once. The bus is fixed at boot, and
/// `stat` on a `/sys` path should not sweep 256 buses to answer.
static FOUND: crate::sync::Spin<Option<Option<(u8, u8, u8)>>> = crate::sync::Spin::new(None);

pub fn present() -> Option<(u8, u8, u8)> {
    let mut f = FOUND.lock();
    if f.is_none() {
        *f = Some(virtgpu::present());
    }
    f.flatten()
}

/// A new client, for a fresh open of the node. Answers its id, which the
/// descriptor carries.
pub fn opened() -> usize {
    let id = NEXT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    unsafe { CLIENTS.get() }.push(Client { id, ctx: None, capset: 0, bos: Vec::new() });
    id
}

/// The last descriptor on a client has gone: everything it made goes too.
pub fn closed(id: usize) {
    let clients = unsafe { CLIENTS.get() };
    let Some(i) = clients.iter().position(|c| c.id == id) else { return };
    let c = clients.swap_remove(i);
    forget_rings(id);
    let _ = virtgpu::with(|d| {
        for b in &c.bos {
            if let Some(p) = b.phys {
                d.unmap_blob(b.res, p);
            }
            let _ = d.unref(c.ctx.unwrap_or(0), b.res);
        }
        if let Some(ctx) = c.ctx {
            let _ = d.ctx_destroy(ctx);
        }
    });
}

fn client(id: usize) -> Option<&'static mut Client> {
    unsafe { CLIENTS.get() }.iter_mut().find(|c| c.id == id)
}

fn read<const N: usize>(at: u64) -> Option<[u8; N]> {
    if !syscall::reachable(at, N, false) {
        return None;
    }
    let mut b = [0u8; N];
    unsafe { core::ptr::copy_nonoverlapping(at as *const u8, b.as_mut_ptr(), N) };
    Some(b)
}

fn write(at: u64, b: &[u8]) -> bool {
    if !syscall::reachable(at, b.len(), true) {
        return false;
    }
    unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), at as *mut u8, b.len()) };
    true
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap_or([0; 8]))
}

/// One ioctl on a render node, from the client the descriptor carries.
pub fn ioctl(id: usize, req: u64, arg: u64) -> u64 {
    let r = ioctl_inner(id, req, arg);
    if TRACE.load(core::sync::atomic::Ordering::Relaxed) && (r as i64) < 0 {
        crate::kprintln!("[drm] client {} ioctl nr {:#x} -> {}", id, req & 0xff, r as i64);
    }
    r
}

fn ioctl_inner(id: usize, req: u64, arg: u64) -> u64 {
    if (req >> 8) & 0xff != DRM_TYPE {
        return ENOTTY;
    }
    let Some(c) = client(id) else { return EBADF };
    match req & 0xff {
        NR_VERSION => version(arg),
        NR_GET_CAP => {
            let Some(b) = read::<16>(arg) else { return EFAULT };
            let value: u64 = match u64_at(&b, 0) {
                // PRIME import and export are what a dma-buf is; refused below
                // until something needs to share a buffer with another device.
                CAP_PRIME => 0,
                CAP_SYNCOBJ | CAP_SYNCOBJ_TIMELINE => 0,
                _ => return EINVAL,
            };
            if write(arg + 8, &value.to_le_bytes()) { 0 } else { EFAULT }
        }
        // A render node has no client capabilities worth refusing over; the
        // ones that exist are about modesetting, which this node does not do.
        NR_SET_CLIENT_CAP => 0,
        NR_GEM_CLOSE => {
            let Some(b) = read::<8>(arg) else { return EFAULT };
            let h = u32_at(&b, 0);
            let Some(i) = c.bos.iter().position(|b| b.handle == h) else { return EINVAL };
            let bo = c.bos.swap_remove(i);
            // A guest mapping of it may outlive this, as on Linux, where the
            // mapping holds a reference. Here the window is given back and the
            // pages stay mapped in the guest until it unmaps them; the host's
            // resource is gone either way, so what they then read is the
            // window's next tenant. Stated rather than solved: Venus unmaps
            // before it closes.
            let ctx = c.ctx.unwrap_or(0);
            let _ = virtgpu::with(|d| {
                if let Some(p) = bo.phys {
                    d.unmap_blob(bo.res, p);
                }
                let _ = d.unref(ctx, bo.res);
            });
            0
        }
        NR_PRIME_HANDLE_TO_FD | NR_PRIME_FD_TO_HANDLE => EINVAL,
        NR_VIRTGPU_GETPARAM => getparam(arg),
        NR_VIRTGPU_GET_CAPS => get_caps(arg),
        NR_VIRTGPU_CONTEXT_INIT => context_init(c, arg),
        NR_VIRTGPU_RESOURCE_CREATE_BLOB => create_blob(c, arg),
        NR_VIRTGPU_RESOURCE_INFO => {
            let Some(b) = read::<16>(arg) else { return EFAULT };
            let h = u32_at(&b, 0);
            let Some(bo) = c.bos.iter().find(|b| b.handle == h) else { return EINVAL };
            let mut o = b;
            o[4..8].copy_from_slice(&bo.res.to_le_bytes());
            o[8..12].copy_from_slice(&(bo.size.min(u32::MAX as u64) as u32).to_le_bytes());
            o[12..16].copy_from_slice(&bo.blob_mem.to_le_bytes());
            if write(arg, &o) { 0 } else { EFAULT }
        }
        NR_VIRTGPU_MAP => map(c, arg),
        NR_VIRTGPU_EXECBUFFER => execbuffer(c, arg),
        // Waiting for a buffer to go idle. A host blob is never in an
        // execbuffer's buffer list under Venus -- it orders its work through
        // its own ring and fences -- so there is nothing to wait for.
        NR_VIRTGPU_WAIT => 0,
        // The classic virgl path: guest-backed resources and transfers. Not
        // Venus's, and refused by name rather than half-done.
        NR_VIRTGPU_RESOURCE_CREATE | NR_VIRTGPU_TRANSFER_FROM_HOST | NR_VIRTGPU_TRANSFER_TO_HOST => EINVAL,
        _ => EINVAL,
    }
}

/// Push every cache line of `[phys, phys + len)` out of every cache level.
///
/// **A page of the host-visible window has more than one life, and a cache
/// remembers the last one.** The window is handed out first fit, so the
/// pages a Venus ring's shared memory used -- host-cached, mapped write-back,
/// its lines resident -- become, once that ring is gone, a feedback buffer
/// the GPU writes. NVIDIA writes that memory without snooping, and an
/// uncached or write-combined read on Intel still snoops this processor's own
/// caches, so the guest can read the ring's old line instead of the value
/// the GPU wrote. Linux flushes whenever a page changes memory type; this
/// kernel had never had a page change type before.
///
/// It was written as the cause of a game's first-frame freeze and is not:
/// with Venus's semaphore and fence feedback switched off entirely
/// (`VN_PERF=no_semaphore_feedback,no_fence_feedback`) the freeze still
/// happened. It stays because the aliasing it removes is real.
fn flush_lines(phys: u64, len: u64) {
    if len == 0 || !crate::mem::paging::map_range(phys & !0xfff, (len + (phys & 0xfff) + 0xfff) & !0xfff, true) {
        return;
    }
    let mut a = phys & !63;
    while a < phys + len {
        unsafe { core::arch::asm!("clflush [{}]", in(reg) a, options(nostack, preserves_flags)) };
        a += 64;
    }
    unsafe { core::arch::asm!("mfence", options(nostack, preserves_flags)) };
}

/// The smallest handle no live buffer holds, as Linux's `idr` answers: a
/// handle names a buffer only while the buffer exists, and a number that
/// only grew would outrun what an i386 `mmap2` can name (`MAP_SHIFT`).
fn free_handle(bos: &[Bo]) -> u32 {
    let mut h = 1u32;
    loop {
        if !bos.iter().any(|b| b.handle == h) {
            return h;
        }
        h += 1;
    }
}

/// `drm_version`: three ints, then three (length, pointer) pairs whose
/// layout depends on the width of `size_t`. Each string is copied as far as
/// the caller's buffer allows and its full length written back, which is how
/// `drmGetVersion` sizes its second call.
fn version(arg: u64) -> u64 {
    let wide = !syscall::abi32();
    let (len, ints, pairs) = if wide { (64usize, 16usize, 16usize) } else { (36, 12, 8) };
    if !syscall::reachable(arg, len, true) {
        return EFAULT;
    }
    let strings: [&[u8]; 3] = [b"virtio_gpu", b"0", b"virtio GPU"];
    let p = arg as *mut u8;
    unsafe {
        // 0.1.0, which is what Linux's driver reports.
        core::ptr::write_unaligned(p as *mut i32, 0);
        core::ptr::write_unaligned(p.add(4) as *mut i32, 1);
        core::ptr::write_unaligned(p.add(8) as *mut i32, 0);
    }
    for (i, s) in strings.iter().enumerate() {
        let at = ints + i * pairs;
        let (room, ptr) = unsafe {
            if wide {
                (
                    core::ptr::read_unaligned(p.add(at) as *const u64),
                    core::ptr::read_unaligned(p.add(at + 8) as *const u64),
                )
            } else {
                (
                    core::ptr::read_unaligned(p.add(at) as *const u32) as u64,
                    core::ptr::read_unaligned(p.add(at + 4) as *const u32) as u64,
                )
            }
        };
        let n = (room as usize).min(s.len());
        if n > 0 && ptr != 0 && !write(ptr, &s[..n]) {
            return EFAULT;
        }
        unsafe {
            if wide {
                core::ptr::write_unaligned(p.add(at) as *mut u64, s.len() as u64);
            } else {
                core::ptr::write_unaligned(p.add(at) as *mut u32, s.len() as u32);
            }
        }
    }
    0
}

/// `GETPARAM`'s `value` is a pointer to an `int`, not the value: Linux writes
/// four bytes through it. Answering in the structure instead leaves Mesa
/// reading whatever its stack held.
fn getparam(arg: u64) -> u64 {
    let Some(b) = read::<16>(arg) else { return EFAULT };
    let info = virtgpu::with(|d| {
        let f = d.features();
        let ids = d.capsets.iter().fold(0u64, |m, c| if c.id < 64 { m | 1 << c.id } else { m });
        (f, ids, d.host_visible.is_some())
    });
    let Ok((f, ids, hv)) = info else { return ENODEV };
    use crate::gpu::virtio::{F_CONTEXT_INIT, F_RESOURCE_BLOB, F_VIRGL};
    let v: u64 = match u64_at(&b, 0) {
        PARAM_3D_FEATURES => (f & F_VIRGL != 0) as u64,
        // The capset query fix is a property of the kernel's `GET_CAPS`, which
        // here has always answered by id and version.
        PARAM_CAPSET_QUERY_FIX => 1,
        PARAM_RESOURCE_BLOB => (f & F_RESOURCE_BLOB != 0) as u64,
        PARAM_HOST_VISIBLE => hv as u64,
        PARAM_CROSS_DEVICE => 0,
        PARAM_CONTEXT_INIT => (f & F_CONTEXT_INIT != 0) as u64,
        PARAM_SUPPORTED_CAPSET_IDS => ids,
        _ => return EINVAL,
    };
    if write(u64_at(&b, 8), &(v as u32).to_le_bytes()) { 0 } else { EFAULT }
}

fn get_caps(arg: u64) -> u64 {
    let Some(b) = read::<24>(arg) else { return EFAULT };
    let (id, ver, addr, size) = (u32_at(&b, 0), u32_at(&b, 4), u64_at(&b, 8), u32_at(&b, 16) as usize);
    let caps = match virtgpu::with(|d| d.capset(id, ver)) {
        Ok(Ok(c)) => c,
        Ok(Err(_)) => return EINVAL,
        Err(_) => return ENODEV,
    };
    let n = size.min(caps.len());
    if write(addr, &caps[..n]) { 0 } else { EFAULT }
}

fn context_init(c: &mut Client, arg: u64) -> u64 {
    if c.ctx.is_some() {
        return (-16i64) as u64; // EBUSY, as Linux answers a second init
    }
    let Some(b) = read::<16>(arg) else { return EFAULT };
    let (n, at) = (u32_at(&b, 0) as u64, u64_at(&b, 8));
    if n > 16 {
        return EINVAL;
    }
    let mut capset = 0u32;
    for i in 0..n {
        let Some(p) = read::<16>(at + i * 16) else { return EFAULT };
        match u64_at(&p, 0) {
            CTX_PARAM_CAPSET_ID => capset = u64_at(&p, 8) as u32,
            // Rings are the host's to keep; the count bounds the ring index a
            // submit may name, and the polled-ring mask asks for events this
            // kernel does not deliver. Both are accepted, as an older kernel
            // that ignored them did.
            CTX_PARAM_NUM_RINGS => {}
            _ => {}
        }
    }
    let ctx = c.id as u32;
    let name = alloc::format!("glados-{}", c.id);
    match virtgpu::with(|d| d.ctx_create(ctx, capset, &name)) {
        Ok(Ok(())) => {
            c.ctx = Some(ctx);
            c.capset = capset;
            0
        }
        Ok(Err(_)) => EINVAL,
        Err(_) => ENODEV,
    }
}

/// Copy a guest command stream in, bounded.
fn stream(at: u64, len: usize) -> Option<Vec<u8>> {
    // An empty stream is a submit made only for its fence -- Venus does this
    // to get a sync file for work already in its ring -- and its pointer is
    // null, which no reachability check would pass.
    if len == 0 {
        return Some(Vec::new());
    }
    if len > virtgpu::MAX_SUBMIT || !syscall::reachable(at, len, false) {
        return None;
    }
    let mut v = alloc::vec![0u8; len];
    unsafe { core::ptr::copy_nonoverlapping(at as *const u8, v.as_mut_ptr(), len) };
    Some(v)
}

fn create_blob(c: &mut Client, arg: u64) -> u64 {
    let Some(b) = read::<56>(arg) else { return EFAULT };
    let (mem, flags, size) = (u32_at(&b, 0), u32_at(&b, 4), u64_at(&b, 16));
    let (cmd_size, cmd, blob_id) = (u32_at(&b, 28) as usize, u64_at(&b, 32), u64_at(&b, 40));
    let Some(ctx) = c.ctx else { return EINVAL };
    if size == 0 {
        return EINVAL;
    }
    // A blob may arrive with the command that makes its memory on the host --
    // Venus's `vkAllocateMemory` -- and the command has to run first, since
    // the blob id names what it made.
    let pre = if cmd_size > 0 {
        match stream(cmd, cmd_size) {
            Some(v) => Some(v),
            None => return EFAULT,
        }
    } else {
        None
    };
    let made = virtgpu::with(|d| {
        if let Some(p) = &pre {
            d.submit(ctx, None, p, false)?;
        }
        d.create_blob(ctx, mem, flags, blob_id, size)
    });
    let res = match made {
        Ok(Ok(r)) => r,
        Ok(Err(_)) => return EINVAL,
        Err(_) => return ENODEV,
    };
    let handle = free_handle(&c.bos);
    c.bos.push(Bo { handle, res, size, blob_mem: mem, phys: None, cache: 0 });
    let mut o = b;
    o[8..12].copy_from_slice(&handle.to_le_bytes());
    o[12..16].copy_from_slice(&res.to_le_bytes());
    if write(arg, &o) { 0 } else { EFAULT }
}

fn map(c: &mut Client, arg: u64) -> u64 {
    let Some(b) = read::<16>(arg) else { return EFAULT };
    let h = u32_at(&b, 8);
    let Some(bo) = c.bos.iter_mut().find(|b| b.handle == h) else { return EINVAL };
    if bo.phys.is_none() {
        let (res, size) = (bo.res, bo.size);
        match virtgpu::with(|d| d.map_blob(res, size)) {
            Ok(Ok((p, cache))) => {
                bo.phys = Some(p);
                bo.cache = cache & 0x0f;
                if bo.cache != 1 {
                    flush_lines(p, size);
                }
                LAST_CACHE.store(bo.cache, core::sync::atomic::Ordering::Relaxed);
                if TRACE.load(core::sync::atomic::Ordering::Relaxed) {
                    crate::kprintln!("[drm] map res {} {} bytes at {:#x}, cache {}", res, size, p, bo.cache);
                }
            }
            Ok(Err(e)) => {
                // Said once here because the client only sees ENOMEM, and
                // Venus turns that into "out of host memory" about the instance.
                crate::kprintln!("[drm] map of resource {} ({} bytes) refused: {}", res, size, e);
                return ENOMEM;
            }
            Err(_) => return ENODEV,
        }
    }
    let off = (h as u64) << MAP_SHIFT;
    if write(arg, &off.to_le_bytes()) { 0 } else { EFAULT }
}

/// What `mmap` on a render node maps: the physical pages of the buffer named
/// by `off`, how many bytes of it there are, and how they may be cached.
///
/// **The cache type is the host's to say and it is not decoration.** The GPU
/// writes NVIDIA's write-combined memory without snooping the CPU's caches,
/// so a write-back mapping of it reads the CPU's own stale lines: a compute
/// result came back with 13,424 of a million elements still holding the
/// pattern written before the dispatch, and elements read wrong and then
/// right as lines were evicted mid-check.
///
/// **And under KVM the type only counts if QEMU honours it.** QEMU 11.1.1
/// leaves `honor-guest-pat` off by default, which keeps KVM's quirk of
/// forcing every guest page write-back -- so uncached and write-combined
/// mapped exactly like write-back and failed the same way. With
/// `-accel kvm,honor-guest-pat=on` all three runs matched. Host-cached
/// memory is snooped and was right either way.
pub fn mapping(id: usize, off: u64) -> Option<(u64, u64, crate::mem::space::Cache)> {
    use crate::mem::space::Cache;
    let c = client(id)?;
    let h = (off >> MAP_SHIFT) as u32;
    let bo = c.bos.iter().find(|b| b.handle == h)?;
    let cache = match bo.cache {
        1 => Cache::WriteBack,
        3 => Cache::WriteCombining,
        // Uncached, and anything the host did not say: the type that cannot
        // be wrong, only slow.
        _ => Cache::Uncached,
    };
    Some((bo.phys?, bo.size, cache))
}

fn execbuffer(c: &mut Client, arg: u64) -> u64 {
    let Some(b) = read::<64>(arg) else { return EFAULT };
    let (flags, size, cmd) = (u32_at(&b, 0), u32_at(&b, 4) as usize, u64_at(&b, 8));
    let ring = (flags & EXECBUF_RING_IDX != 0).then(|| u32_at(&b, 32) as u8);
    let Some(ctx) = c.ctx else { return EINVAL };
    if u32_at(&b, 40) != 0 || u32_at(&b, 44) != 0 {
        // Syncobjs, which this kernel told the client it does not have.
        return EINVAL;
    }
    // An in-fence would order this submit after another; the host orders a
    // ring's work already, and Venus only asks for one across rings with
    // timeline semaphores, which it is not using here.
    let _ = flags & EXECBUF_FENCE_FD_IN;
    let Some(mut data) = stream(cmd, size) else {
        if TRACE.load(core::sync::atomic::Ordering::Relaxed) {
            crate::kprintln!(
                "[drm] execbuffer stream {:#x} + {} refused: {}",
                cmd,
                size,
                if size > virtgpu::MAX_SUBMIT { "over the submit limit" } else { "not the guest's memory" }
            );
        }
        return EFAULT;
    };
    let idle = RING_IDLE_US.load(core::sync::atomic::Ordering::Relaxed);
    if idle != 0 && data.len() >= 84 && u32_at(&data, 0) == 188 && u64_at(&data, 76) == 1_000_000 {
        data[76..84].copy_from_slice(&(idle * 1000).to_le_bytes());
    }
    if data.len() >= 136 && u32_at(&data, 0) == CMD_CREATE_RING {
        note_ring(c.id, ctx, &data);
    }
    let out = flags & EXECBUF_FENCE_FD_OUT != 0;
    if TRACE.load(core::sync::atomic::Ordering::Relaxed) {
        crate::kprintln!(
            "[drm] t={} us submit {} bytes ring {:?}{} first {:?}",
            crate::time::rdtsc() / crate::time::tsc_mhz().max(1),
            data.len(),
            ring,
            if out { " fenced" } else { "" },
            data.chunks(4).take(40).map(|c| u32::from_le_bytes([c[0], c.get(1).copied().unwrap_or(0), c.get(2).copied().unwrap_or(0), c.get(3).copied().unwrap_or(0)])).collect::<Vec<u32>>()
        );
    }
    let t0 = crate::time::rdtsc();
    // **The guest lock is let go for the round trip.** A submit nobody fences
    // is answered only when the host has taken it, which is a few hundred
    // microseconds and sometimes several milliseconds -- and every guest
    // syscall waits on that lock, so a game's other threads queued behind
    // every Venus notify: 11,251 of them in one run, each holding the whole
    // guest for 180 us. What the submit needs was copied out above (`ctx`,
    // `ring`, `data`), and nothing of the client is touched until the lock is
    // back. The device keeps its own lock, which is what orders submits.
    let held = syscall::holds_big();
    if held {
        syscall::big_unlock();
    }
    let r = virtgpu::with(|d| d.submit(ctx, ring, &data, out));
    if held {
        syscall::big_lock();
    }
    note_submit(if data.len() >= 4 { u32_at(&data, 0) } else { u32::MAX }, out, t0);
    let fence = match r {
        Ok(Ok(f)) => f,
        Ok(Err(_)) => return EINVAL,
        Err(_) => return ENODEV,
    };
    if out {
        let fd = syscall::new_fence_fd(fence);
        if (fd as i64) < 0 {
            return fd;
        }
        if !write(arg + 28, &(fd as i32).to_le_bytes()) {
            return EFAULT;
        }
    }
    0
}

/// What each submit was and how long the kernel held the guest for it, by
/// the first command in the stream. A stall reads as one thread in `ioctl`
/// for the whole of it, and nothing else said which command or why.
/// Sixteen kinds, first come; `(id, fenced)` -> (count, total us, worst us).
static SUBMITS: crate::sync::Spin<[(u32, bool, u64, u64, u64); 16]> =
    crate::sync::Spin::new([(0, false, 0, 0, 0); 16]);

fn note_submit(cmd: u32, fenced: bool, t0: u64) {
    let us = (crate::time::rdtsc() - t0) / crate::time::tsc_mhz().max(1);
    let mut t = SUBMITS.lock_irq();
    let slot = t.iter().position(|e| e.2 != 0 && e.0 == cmd && e.1 == fenced).or_else(|| t.iter().position(|e| e.2 == 0));
    if let Some(i) = slot {
        let e = &mut t[i];
        *e = (cmd, fenced, e.2 + 1, e.3 + us, e.4.max(us));
    }
}

/// `gpu drm`'s second half: the submit table, and every ring as it stands.
pub fn report_rings() {
    use crate::kprintln;
    let t = *SUBMITS.lock_irq();
    kprintln!("  submits by first command (id, fenced: count, mean, worst):");
    for e in t.iter().filter(|e| e.2 != 0) {
        kprintln!("    {:>5} {:<7} {:>8}  {:>7} us  {:>8} us", e.0, if e.1 { "fenced" } else { "waited" }, e.2, e.3 / e.2, e.4);
    }
    let rings = unsafe { RINGS.get() };
    for r in rings.iter() {
        let Some(c) = client(r.client) else { continue };
        let Some(phys) = c.bos.iter().find(|b| b.res == r.res).and_then(|b| b.phys) else { continue };
        if !crate::mem::paging::map_range(phys & !0xfff, 4096, true) {
            continue;
        }
        let rd = |off: u64| unsafe { core::ptr::read_volatile((phys + off) as *const u32) };
        kprintln!(
            "  ring {:#x} ctx {}: head {} tail {} status {:#x}",
            r.id,
            r.ctx,
            rd(r.head),
            rd(r.tail),
            rd(r.status)
        );
    }
}

/// Whether a sync file's fence has signalled.
pub fn fence_ready(fence: u64) -> bool {
    virtgpu::with(|d| d.signaled(fence)).unwrap_or(true)
}

// --- /sys -----------------------------------------------------------------
//
// libdrm finds a device without opening it: it lists `/dev/dri`, `stat`s each
// node for its major and minor, and reads the rest out of
// `/sys/dev/char/<major>:<minor>/device` -- the bus from the name of the
// `subsystem` link, the slot from `uevent`, and the ids from four small
// files. Mesa's Venus then picks the device whose ids are virtio-gpu's. Each
// of these is a fact about a device this kernel found on the bus, which is
// what admits them under `proc.rs`'s rule.

pub fn sys_is_dir(path: &str) -> bool {
    present().is_some()
        && matches!(
            path,
            "/sys/dev" | "/sys/dev/char" | SYS_DEV | SYS_DEVICE
                | "/sys/dev/char/511:128/device/drm"
                | "/sys/dev/char/511:128/device/drm/renderD128"
        )
}

const SYS_FILES: [&str; 7] = ["uevent", "vendor", "device", "subsystem_vendor", "subsystem_device", "revision", "config"];

pub fn sys_is_file(path: &str) -> bool {
    present().is_some()
        && path
            .strip_prefix("/sys/dev/char/511:128/device/")
            .is_some_and(|f| SYS_FILES.contains(&f))
        || (present().is_some() && path == "/sys/dev/char/511:128/uevent")
}

/// The `subsystem` link, whose last component is how libdrm learns the bus.
pub fn sys_link(path: &str) -> Option<String> {
    (present().is_some() && path == "/sys/dev/char/511:128/device/subsystem")
        .then(|| String::from("../../../../bus/pci"))
}

fn config() -> Option<[u8; 64]> {
    let (bus, dev, func) = present()?;
    let ecam = crate::net::ecam()?;
    let d = *crate::gpu::virtio::find(ecam).iter().find(|d| (d.bus, d.dev, d.func) == (bus, dev, func))?;
    let mut c = [0u8; 64];
    for i in 0..16 {
        let w = crate::dev::pci::cfg_read32(ecam, &d, i as u64 * 4);
        c[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    Some(c)
}

pub fn sys_read(path: &str) -> Option<Vec<u8>> {
    if path == "/sys/dev/char/511:128/uevent" {
        // The node's own uevent, which is where libdrm reads a node's name
        // from when it has only a device number.
        present()?;
        return Some(alloc::format!("MAJOR={}\nMINOR={}\nDEVNAME=dri/renderD128\nDEVTYPE=drm_minor\n", MAJOR, RENDER_MINOR).into_bytes());
    }
    let f = path.strip_prefix("/sys/dev/char/511:128/device/")?;
    let (bus, dev, func) = present()?;
    let c = config()?;
    let h16 = |at: usize| u16::from_le_bytes([c[at], c[at + 1]]);
    let s = match f {
        "uevent" => alloc::format!(
            "DRIVER=virtio-pci\nPCI_CLASS={:02X}{:02X}{:02X}\nPCI_ID={:04X}:{:04X}\nPCI_SUBSYS_ID={:04X}:{:04X}\nPCI_SLOT_NAME=0000:{:02x}:{:02x}.{}\n",
            c[11], c[10], c[9], h16(0), h16(2), h16(0x2c), h16(0x2e), bus, dev, func
        ),
        "vendor" => alloc::format!("0x{:04x}\n", h16(0)),
        "device" => alloc::format!("0x{:04x}\n", h16(2)),
        "subsystem_vendor" => alloc::format!("0x{:04x}\n", h16(0x2c)),
        "subsystem_device" => alloc::format!("0x{:04x}\n", h16(0x2e)),
        "revision" => alloc::format!("0x{:02x}\n", c[8]),
        "config" => return Some(c.to_vec()),
        _ => return None,
    };
    Some(s.into_bytes())
}

pub fn sys_entries(dir: &str) -> Vec<(String, bool, usize)> {
    if present().is_none() {
        return Vec::new();
    }
    let d = |n: &str| (String::from(n), true, 0usize);
    let f = |n: &str| (String::from(n), false, 0usize);
    match dir {
        "/sys/dev" => alloc::vec![d("char")],
        "/sys/dev/char" => alloc::vec![d("511:128")],
        SYS_DEV => alloc::vec![d("device"), f("uevent")],
        SYS_DEVICE => {
            let mut v: Vec<_> = SYS_FILES.iter().map(|n| f(n)).collect();
            v.push(d("drm"));
            v.push(f("subsystem"));
            v
        }
        "/sys/dev/char/511:128/device/drm" => alloc::vec![d("renderD128")],
        _ => Vec::new(),
    }
}

/// What `diag linux` asks of the encoding, with no device.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    // glibc's `major` and `minor` on the value, reimplemented: the claim is
    // that the round trip names this node and nothing else.
    let maj = ((RDEV >> 8) & 0xfff) | ((RDEV >> 32) & !0xfff);
    let min = (RDEV & 0xff) | ((RDEV >> 12) & !0xff);
    out.push(("the render node's st_rdev takes apart to its major and 128", maj == MAJOR && min == 128));
    // The property the major was chosen for, stated where it can fail.
    out.push(("and the major is one Zink will not try to match a Vulkan device against", !(1..255).contains(&MAJOR)));
    out.push(("a map offset names its handle and stays page-aligned", (5u64 << MAP_SHIFT) % 4096 == 0 && ((5u64 << MAP_SHIFT) >> MAP_SHIFT) == 5));
    out.push((
        "an i386 mmap2 reaches handle 2^20 - 1 through a 32-bit page number",
        ((((1u64 << 20) - 1) << MAP_SHIFT) >> 12) <= u32::MAX as u64,
    ));
    out.push(("a closed buffer's handle is the next one handed out", {
        let bo = |handle| Bo { handle, res: 0, size: 0, blob_mem: 0, phys: None, cache: 0 };
        free_handle(&[bo(1), bo(3)]) == 2 && free_handle(&[bo(1), bo(2)]) == 3 && free_handle(&[]) == 1
    }));
    out.extend(virtgpu::checks());
    out
}

// --- the ring watchdog ------------------------------------------------------
//
// Venus moves Vulkan through rings in shared memory, and a host ring thread
// that has gone idle sleeps until the guest sends `vkNotifyRingMESA`. Mesa
// sends one only when it sees the ring's IDLE status bit as it publishes a
// new tail, at most once a millisecond.
//
// **Under this kernel that wake-up is lost, and the cause is not known.**
// Driven with a 32-bit game on Zink: a ring's head stopped short of its tail with
// the status reading IDLE|ALIVE, the host's ring thread parked in
// `cnd_wait` (read off it with eu-stack), and the guest polling its seqno
// forever -- the main ring and the per-thread rings both, after a varying
// number of batches, and with the host's idle timeout raised from 1 ms to
// 50 ms as well. Reading both sides' orderings says it cannot happen with
// coherent memory and agreeing clocks, and the clocks agree (2688 MHz on
// both); a premise is wrong somewhere unread.
//
// So the kernel keeps the bargain itself. It sees every `vkCreateRingMESA`
// go by, which names the ring's id, its shared-memory resource and where
// head, tail and status sit in it. While a guest waits, a ring whose status
// says IDLE with commands unread, and whose head has not moved since the
// last look, gets the notify Mesa would have sent. A notify to a ring that
// is already awake costs the host a condition-variable signal and nothing
// else, so a wrong guess is cheap and a right one ends a hang.

const CMD_CREATE_RING: u32 = 188;
const CMD_NOTIFY_RING: u32 = 190;
const STATUS_IDLE: u32 = 1;

#[derive(Clone, Copy)]
struct Ring {
    client: usize,
    ctx: u32,
    id: u64,
    res: u32,
    head: u64,
    tail: u64,
    status: u64,
    /// The head at the last look, for the trace.
    seen: u32,
}

static RINGS: Racy<Vec<Ring>> = Racy::new(Vec::new());
/// How often the rings are looked at. Every look that finds a sleeping ring
/// with work is a stall ended, and the time between looks is what each one
/// costs the guest: at 2 ms and two looks the game ran at 8.4 fps.
const WATCH_US: u64 = 500;
static LAST_LOOK: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
/// Notifies the watchdog has sent, for `gpu drm`.
pub static KICKS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// `vkCreateRingMESA` as Venus encodes it, in u32 words: the command, flags,
/// the ring id (u64), the create-info pointer marker, then the info --
/// sType, pNext, flags, resourceId at word 14, offset, size, idleTimeout,
/// and the head, tail and status offsets as u64s at words 21, 23 and 25.
/// Read from a captured command rather than recalled, and each field is
/// checked against what it must be before anything is believed.
fn note_ring(client: usize, ctx: u32, d: &[u8]) {
    let w = |i: usize| u32_at(d, i * 4) as u64;
    let id = w(2) | w(3) << 32;
    let res = w(14) as u32;
    let (head, tail, status) = (w(21) | w(22) << 32, w(23) | w(24) << 32, w(25) | w(26) << 32);
    if res == 0 || head >= 4096 || tail >= 4096 || status >= 4096 {
        return;
    }
    let rings = unsafe { RINGS.get() };
    rings.retain(|r| !(r.client == client && r.id == id));
    rings.push(Ring { client, ctx, id, res, head, tail, status, seen: u32::MAX });
}

fn forget_rings(client: usize) {
    unsafe { RINGS.get() }.retain(|r| r.client != client);
}

/// Look at every ring, at most every two milliseconds, and wake the ones the
/// host is sleeping on with work in them. Called from a guest's waits.
pub fn watch() {
    let now = crate::time::rdtsc() / crate::time::tsc_mhz().max(1);
    let last = LAST_LOOK.load(core::sync::atomic::Ordering::Relaxed);
    if now < last + WATCH_US {
        return;
    }
    LAST_LOOK.store(now, core::sync::atomic::Ordering::Relaxed);
    let rings = unsafe { RINGS.get() };
    for r in rings.iter_mut() {
        let Some(c) = client(r.client) else { continue };
        let Some(phys) = c.bos.iter().find(|b| b.res == r.res).and_then(|b| b.phys) else { continue };
        if !crate::mem::paging::map_range(phys & !0xfff, 4096, true) {
            continue;
        }
        let rd = |off: u64| unsafe { core::ptr::read_volatile((phys + off) as *const u32) };
        let (head, tail, status) = (rd(r.head), rd(r.tail), rd(r.status));
        // One look is enough. IDLE with work unread is either the host
        // asleep on it or the host in the few instructions between setting
        // IDLE and finding the new tail -- and a notify to an awake ring
        // costs it a condition-variable signal and nothing else.
        r.seen = head;
        if head == tail || status & STATUS_IDLE == 0 {
            continue;
        }
        let mut n = [0u8; 24];
        n[0..4].copy_from_slice(&CMD_NOTIFY_RING.to_le_bytes());
        n[8..16].copy_from_slice(&r.id.to_le_bytes());
        n[16..20].copy_from_slice(&tail.to_le_bytes());
        let ctx = r.ctx;
        // Waited for, deliberately. Sent without waiting, the notifies piled
        // into the queue while QEMU had it blocked behind an unmap, every
        // descriptor filled, and the game stopped for good with the watchdog
        // still counting.
        if matches!(virtgpu::with(|d| d.submit(ctx, Some(0), &n, false)), Ok(Ok(_))) {
            KICKS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            if TRACE.load(core::sync::atomic::Ordering::Relaxed) {
                crate::kprintln!("[drm] ring {:#x} idle at {} of {}: notified", r.id, head, tail);
            }
        }
    }
}

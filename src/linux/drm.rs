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

pub const MAJOR: u64 = 226;
pub const RENDER_MINOR: u64 = 128;
pub const NODE: &str = "/dev/dri/renderD128";

/// `makedev(226, 128)` as glibc encodes it: the low eight bits of the minor
/// at the bottom, the major above them, and the rest of the minor at 20.
/// libdrm takes it apart with `major()` and `minor()`, so an encoding that is
/// merely plausible names a different `/sys/dev/char` directory.
pub const RDEV: u64 = ((MAJOR & 0xfff) << 8) | (RENDER_MINOR & 0xff) | ((RENDER_MINOR & !0xff) << 12);

const SYS_DEV: &str = "/sys/dev/char/226:128";
const SYS_DEVICE: &str = "/sys/dev/char/226:128/device";

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
/// can find the buffer from the offset alone. 32 because an i386 client's
/// `mmap2` carries the offset in pages as a 32-bit number, which still reaches
/// handle 4095 -- far more buffers than a client keeps mapped.
const MAP_SHIFT: u32 = 32;

struct Bo {
    handle: u32,
    res: u32,
    size: u64,
    blob_mem: u32,
    /// Where `MAP` put it in the host-visible window, once it has.
    phys: Option<u64>,
}

struct Client {
    id: usize,
    ctx: Option<u32>,
    capset: u32,
    bos: Vec<Bo>,
    next_handle: u32,
}

/// One per open render node. Racy because every caller is a guest syscall,
/// and those run under the guest lock.
static CLIENTS: Racy<Vec<Client>> = Racy::new(Vec::new());
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
    unsafe { CLIENTS.get() }.push(Client { id, ctx: None, capset: 0, bos: Vec::new(), next_handle: 1 });
    id
}

/// The last descriptor on a client has gone: everything it made goes too.
pub fn closed(id: usize) {
    let clients = unsafe { CLIENTS.get() };
    let Some(i) = clients.iter().position(|c| c.id == id) else { return };
    let c = clients.swap_remove(i);
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
    let handle = c.next_handle;
    c.next_handle += 1;
    c.bos.push(Bo { handle, res, size, blob_mem: mem, phys: None });
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
            Ok(Ok((p, _cache))) => bo.phys = Some(p),
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
/// by `off`, and how many bytes of it there are.
pub fn mapping(id: usize, off: u64) -> Option<(u64, u64)> {
    let c = client(id)?;
    let h = (off >> MAP_SHIFT) as u32;
    let bo = c.bos.iter().find(|b| b.handle == h)?;
    Some((bo.phys?, bo.size))
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
    let Some(data) = stream(cmd, size) else { return EFAULT };
    let out = flags & EXECBUF_FENCE_FD_OUT != 0;
    let fence = match virtgpu::with(|d| d.submit(ctx, ring, &data, out)) {
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
                | "/sys/dev/char/226:128/device/drm"
                | "/sys/dev/char/226:128/device/drm/renderD128"
        )
}

const SYS_FILES: [&str; 7] = ["uevent", "vendor", "device", "subsystem_vendor", "subsystem_device", "revision", "config"];

pub fn sys_is_file(path: &str) -> bool {
    present().is_some()
        && path
            .strip_prefix("/sys/dev/char/226:128/device/")
            .is_some_and(|f| SYS_FILES.contains(&f))
}

/// The `subsystem` link, whose last component is how libdrm learns the bus.
pub fn sys_link(path: &str) -> Option<String> {
    (present().is_some() && path == "/sys/dev/char/226:128/device/subsystem")
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
    let f = path.strip_prefix("/sys/dev/char/226:128/device/")?;
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
        "/sys/dev/char" => alloc::vec![d("226:128")],
        SYS_DEV => alloc::vec![d("device")],
        SYS_DEVICE => {
            let mut v: Vec<_> = SYS_FILES.iter().map(|n| f(n)).collect();
            v.push(d("drm"));
            v.push(f("subsystem"));
            v
        }
        "/sys/dev/char/226:128/device/drm" => alloc::vec![d("renderD128")],
        _ => Vec::new(),
    }
}

/// What `diag linux` asks of the encoding, with no device.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    // glibc's `major` and `minor` on the value, reimplemented: the claim is
    // that the round trip names 226:128 and nothing else.
    let maj = ((RDEV >> 8) & 0xfff) | ((RDEV >> 32) & !0xfff);
    let min = (RDEV & 0xff) | ((RDEV >> 12) & !0xff);
    out.push(("the render node's st_rdev takes apart to 226:128", maj == 226 && min == 128));
    out.push(("a map offset names its handle and stays page-aligned", (5u64 << MAP_SHIFT) % 4096 == 0 && ((5u64 << MAP_SHIFT) >> MAP_SHIFT) == 5));
    out.push((
        "an i386 mmap2 can still reach handle 4095 through a 32-bit page number",
        ((4095u64 << MAP_SHIFT) >> 12) <= u32::MAX as u64,
    ));
    out.extend(virtgpu::checks());
    out
}

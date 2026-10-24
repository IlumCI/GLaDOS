//! virtio-gpu, held open and driven for a ring-3 client.
//!
//! `virtio.rs` asks the device questions and lets go. This keeps it: one
//! device, opened on first use, owning its control queue for the life of the
//! machine, and offering exactly the operations Linux's `virtio_gpu` driver
//! turns its ioctls into -- context creation, command submission with a fence,
//! blob resources, and mapping a blob into the host-visible window. What a
//! guest *does* with those is Mesa's business: Venus serialises Vulkan into
//! the commands this submits, and nothing on this side reads them.
//!
//! **The kernel stays dumb on purpose.** Everything here is a transport. There
//! is no Vulkan in it and no command encoder, which is what the 1.4.3 plan
//! chose over an encoder in ring 0: the 50,000 lines that understand Vulkan
//! already exist, run at ring 3, and are tested against this exact device by
//! everybody who runs a Linux VM with Venus.
//!
//! **Fences are the asynchrony, and the device dictates it.** A command sent
//! with `FENCE` set is answered only when its fence has signalled, which on
//! Venus means when the host's GPU has finished the work queued on that ring
//! up to that point. So a submit posts and returns, and the answer arriving
//! later *is* the fence signalling; `pump` is what notices. Every other
//! command is answered at once and is waited for in place.
//!
//! Command layouts are the virtio 1.2 specification's, section 5.7.

use super::virtio::{le32, CapsetInfo, Gpu, HDR, RESP_OK_NODATA};
use crate::dev::dma::Dma;
use crate::dev::virtio::Region;
use crate::sync::Spin;
use alloc::string::String;
use alloc::vec::Vec;

const CMD_RESOURCE_UNREF: u32 = 0x0102;
const CMD_RESOURCE_CREATE_BLOB: u32 = 0x010c;
const CMD_CTX_CREATE: u32 = 0x0200;
const CMD_CTX_DESTROY: u32 = 0x0201;
const CMD_CTX_ATTACH_RESOURCE: u32 = 0x0202;
const CMD_CTX_DETACH_RESOURCE: u32 = 0x0203;
const CMD_SUBMIT_3D: u32 = 0x0207;
const CMD_RESOURCE_MAP_BLOB: u32 = 0x0208;
const CMD_RESOURCE_UNMAP_BLOB: u32 = 0x0209;

const RESP_OK_MAP_INFO: u32 = 0x1106;

const FLAG_FENCE: u32 = 1;
const FLAG_INFO_RING_IDX: u32 = 2;

pub const BLOB_MEM_HOST3D: u32 = 2;

/// A command that is not answered in this long is not going to be. Five
/// seconds because a context creation starts a render-server worker on the
/// host, and that is the slowest thing here.
const WAIT_MS: u64 = 5000;

/// The largest command stream one submit may carry. Venus sends its real
/// traffic through a ring in shared memory and uses a submit for the ring's
/// own creation and for small direct calls, so this is generous.
pub const MAX_SUBMIT: usize = 1 << 20;

struct Pending {
    head: u16,
    // Kept alive until the device is done reading it.
    _req: Dma,
    resp: Dma,
    /// Zero when the command is not fenced.
    fence: u64,
    /// Somebody is waiting in place for this answer; keep it once it lands.
    waited: bool,
    done: Option<usize>,
}

/// The held device.
pub struct Dev {
    gpu: Gpu,
    pending: Vec<Pending>,
    /// Fences issued and not yet signalled. Short: one per submit in flight.
    unsignaled: Vec<u64>,
    next_fence: u64,
    next_res: u32,
    pub capsets: Vec<CapsetInfo>,
    pub host_visible: Option<Region>,
    /// What of the host-visible window is mapped, as (offset, length), sorted.
    hv_used: Vec<(u64, u64)>,
    /// Where the device sits, for `/sys`'s `PCI_SLOT_NAME`.
    pub slot: (u8, u8, u8),
    /// Errors the device answered to fenced commands nobody waited for, which
    /// would otherwise vanish: a submit's reply arrives long after its caller
    /// has returned.
    pub async_errors: u32,
    pub last_async_error: u32,
    pub submits: u64,
}

// Safety: the `Dma` buffers are raw heap pointers owned by this structure and
// reached only through the `DEV` lock.
unsafe impl Send for Dev {}

static DEV: Spin<Option<Dev>> = Spin::new(None);

/// Whether a virtio-gpu is on the bus at all, without opening it. A property
/// of the machine fixed at boot, which is what lets `/sys` and `/dev` ask it.
pub fn present() -> Option<(u8, u8, u8)> {
    let ecam = crate::net::ecam()?;
    let d = *super::virtio::find(ecam).first()?;
    Some((d.bus, d.dev, d.func))
}

/// Run `f` against the device, opening it on first use.
pub fn with<R>(f: impl FnOnce(&mut Dev) -> R) -> Result<R, &'static str> {
    let mut g = DEV.lock();
    if g.is_none() {
        *g = Some(Dev::open()?);
    }
    let d = g.as_mut().ok_or("no device")?;
    d.pump();
    Ok(f(d))
}

/// Whether the device has been opened, so the probe can refuse to reset a
/// device somebody is using.
pub fn held() -> bool {
    DEV.lock().is_some()
}

fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

/// A control header with every field this driver ever sets.
fn header(b: &mut [u8], ty: u32, ctx: u32, fence: u64, ring: Option<u8>) {
    put32(b, 0, ty);
    let mut flags = 0;
    if fence != 0 {
        flags |= FLAG_FENCE;
        if ring.is_some() {
            flags |= FLAG_INFO_RING_IDX;
        }
    }
    put32(b, 4, flags);
    put64(b, 8, fence);
    put32(b, 16, ctx);
    b[20] = ring.unwrap_or(0);
}

impl Dev {
    fn open() -> Result<Dev, &'static str> {
        let ecam = crate::net::ecam().ok_or("no PCI configuration space")?;
        let d = *super::virtio::find(ecam).first().ok_or("no virtio-gpu on this bus")?;
        let mut gpu = Gpu::open(ecam, d)?;
        // Read synchronously, before anything is in flight: `Gpu::call` takes
        // the first completion as its own.
        let mut capsets = Vec::new();
        for i in 0..gpu.num_capsets().min(16) {
            if let Ok(c) = gpu.capset_info(i) {
                capsets.push(c);
            }
        }
        let host_visible = gpu.t.host_visible;
        Ok(Dev {
            gpu,
            pending: Vec::new(),
            unsignaled: Vec::new(),
            next_fence: 1,
            next_res: 1,
            capsets,
            host_visible,
            hv_used: Vec::new(),
            slot: (d.bus, d.dev, d.func),
            async_errors: 0,
            last_async_error: 0,
            submits: 0,
        })
    }

    pub fn features(&self) -> u64 {
        self.gpu.features
    }

    /// The capset's bytes, for `GET_CAPS`. Asked through the async path, since
    /// by the time a guest asks there may be submits in flight.
    pub fn capset(&mut self, id: u32, version: u32) -> Result<Vec<u8>, String> {
        let mut b = [0u8; HDR + 8];
        header(&mut b, 0x0109, 0, 0, None);
        put32(&mut b, HDR, id);
        put32(&mut b, HDR + 4, version);
        let max = self.capsets.iter().find(|c| c.id == id).map(|c| c.max_size as usize).unwrap_or(0);
        let r = self.call(&b, HDR + max.max(64)).map_err(String::from)?;
        if le32(&r, 0) != 0x1103 {
            return Err(alloc::format!("{:#06x} {}", le32(&r, 0), super::virtio::resp_name(le32(&r, 0))));
        }
        Ok(r[HDR..].to_vec())
    }

    /// Collect every answer the device has written.
    pub fn pump(&mut self) {
        loop {
            let Some(q) = self.gpu.ctrl.as_mut() else { return };
            let Some((head, len)) = q.reap() else { return };
            let Some(i) = self.pending.iter().position(|p| p.head == head && p.done.is_none()) else {
                continue;
            };
            let p = &mut self.pending[i];
            if p.fence != 0 {
                let f = p.fence;
                self.unsignaled.retain(|&u| u != f);
            }
            if p.waited {
                p.done = Some(len);
            } else {
                let ty = le32(p.resp.as_slice(), 0);
                if ty != RESP_OK_NODATA {
                    self.async_errors += 1;
                    self.last_async_error = ty;
                }
                self.pending.swap_remove(i);
            }
        }
    }

    fn post(&mut self, body: &[u8], resp_len: usize, fence: u64, waited: bool) -> Result<u16, &'static str> {
        let mut req = Dma::new(body.len(), 16).ok_or("no memory for a request")?;
        req.as_mut_slice().copy_from_slice(body);
        let resp = Dma::new(resp_len.max(HDR), 16).ok_or("no memory for a reply")?;
        let q = self.gpu.ctrl.as_mut().ok_or("no control queue")?;
        let mut head = q.post(req.pa(), body.len(), resp.pa(), resp.len());
        // Every pair in use: wait for one back. Only fenced commands hold a
        // pair for long, and those finish on their own.
        let deadline = crate::time::rdtsc() + crate::time::tsc_mhz().max(1) * 1000 * WAIT_MS;
        while head.is_none() {
            self.pump();
            if crate::time::rdtsc() > deadline {
                return Err("every descriptor stayed in use");
            }
            core::hint::spin_loop();
            let q = self.gpu.ctrl.as_mut().ok_or("no control queue")?;
            head = q.post(req.pa(), body.len(), resp.pa(), resp.len());
        }
        let head = head.unwrap_or(0);
        self.pending.push(Pending { head, _req: req, resp, fence, waited, done: None });
        Ok(head)
    }

    /// Send one unfenced command and answer its reply.
    fn call(&mut self, body: &[u8], resp_len: usize) -> Result<Vec<u8>, &'static str> {
        let head = self.post(body, resp_len, 0, true)?;
        let deadline = crate::time::rdtsc() + crate::time::tsc_mhz().max(1) * 1000 * WAIT_MS;
        loop {
            self.pump();
            if let Some(i) = self.pending.iter().position(|p| p.head == head && p.done.is_some()) {
                let p = self.pending.swap_remove(i);
                let n = p.done.unwrap_or(0).min(p.resp.len());
                if n < HDR {
                    return Err("the reply is shorter than a header");
                }
                return Ok(p.resp.as_slice()[..n].to_vec());
            }
            if crate::time::rdtsc() > deadline {
                // Left pending: the device still owns the buffers, and freeing
                // them would hand it memory the allocator is about to reuse.
                if let Some(p) = self.pending.iter_mut().find(|p| p.head == head) {
                    p.waited = false;
                }
                return Err("the device did not answer in time");
            }
            core::hint::spin_loop();
        }
    }

    fn call_ok(&mut self, body: &[u8]) -> Result<(), &'static str> {
        let r = self.call(body, HDR)?;
        match le32(&r, 0) {
            RESP_OK_NODATA => Ok(()),
            ty => Err(super::virtio::resp_name(ty)),
        }
    }

    pub fn ctx_create(&mut self, ctx: u32, capset: u32, name: &str) -> Result<(), &'static str> {
        let mut b = [0u8; HDR + 8 + 64];
        header(&mut b, CMD_CTX_CREATE, ctx, 0, None);
        let n = name.len().min(64);
        put32(&mut b, HDR, n as u32);
        put32(&mut b, HDR + 4, capset);
        b[HDR + 8..HDR + 8 + n].copy_from_slice(&name.as_bytes()[..n]);
        self.call_ok(&b)
    }

    pub fn ctx_destroy(&mut self, ctx: u32) -> Result<(), &'static str> {
        let mut b = [0u8; HDR];
        header(&mut b, CMD_CTX_DESTROY, ctx, 0, None);
        self.call_ok(&b)
    }

    /// Submit a command stream to `ctx`. With `fenced`, answers the fence id
    /// whose signalling says the host has finished it; without, waits for the
    /// host to have accepted it and answers zero.
    pub fn submit(&mut self, ctx: u32, ring: Option<u8>, data: &[u8], fenced: bool) -> Result<u64, &'static str> {
        if data.len() > MAX_SUBMIT {
            return Err("a command stream larger than one submit may carry");
        }
        let mut b = alloc::vec![0u8; HDR + 8 + data.len()];
        let fence = if fenced {
            let f = self.next_fence;
            self.next_fence += 1;
            f
        } else {
            0
        };
        header(&mut b, CMD_SUBMIT_3D, ctx, fence, ring);
        put32(&mut b, HDR, data.len() as u32);
        b[HDR + 8..].copy_from_slice(data);
        self.submits += 1;
        if fenced {
            self.unsignaled.push(fence);
            self.post(&b, HDR, fence, false)?;
            Ok(fence)
        } else {
            self.call_ok(&b)?;
            Ok(0)
        }
    }

    /// Whether fence `id` has signalled. A fence never issued counts as
    /// signalled, the way a sync file for nothing reads as ready.
    pub fn signaled(&mut self, id: u64) -> bool {
        self.pump();
        !self.unsignaled.contains(&id)
    }

    /// Create a blob resource for `ctx` and attach it there. Answers the
    /// resource id.
    pub fn create_blob(&mut self, ctx: u32, mem: u32, flags: u32, blob_id: u64, size: u64) -> Result<u32, &'static str> {
        if mem != BLOB_MEM_HOST3D {
            // Guest blobs (mem 1) need a scatter list of guest pages, which nothing
            // has asked for: Venus with host-visible memory uses HOST3D for
            // all of its own memory.
            return Err("only host-allocated blobs");
        }
        let res = self.next_res;
        self.next_res += 1;
        let mut b = [0u8; HDR + 32];
        header(&mut b, CMD_RESOURCE_CREATE_BLOB, ctx, 0, None);
        put32(&mut b, HDR, res);
        put32(&mut b, HDR + 4, mem);
        put32(&mut b, HDR + 8, flags);
        put32(&mut b, HDR + 12, 0);
        put64(&mut b, HDR + 16, blob_id);
        put64(&mut b, HDR + 24, size);
        self.call_ok(&b)?;
        // Linux attaches a resource to the context that opened its handle,
        // in `virtio_gpu_gem_object_open`; the host refuses to map or use one
        // a context was never given.
        let mut a = [0u8; HDR + 8];
        header(&mut a, CMD_CTX_ATTACH_RESOURCE, ctx, 0, None);
        put32(&mut a, HDR, res);
        if let Err(e) = self.call_ok(&a) {
            let _ = self.unref(ctx, res);
            return Err(e);
        }
        Ok(res)
    }

    /// Map a blob into the host-visible window. Answers its physical address
    /// and the host's cache type for it.
    pub fn map_blob(&mut self, res: u32, size: u64) -> Result<(u64, u32), &'static str> {
        let hv = self.host_visible.ok_or("no host-visible memory: start QEMU with hostmem=")?;
        let len = size.next_multiple_of(4096);
        let off = self.hv_take(len, hv.len as u64).ok_or("the host-visible window is full")?;
        let mut b = [0u8; HDR + 16];
        header(&mut b, CMD_RESOURCE_MAP_BLOB, 0, 0, None);
        put32(&mut b, HDR, res);
        put64(&mut b, HDR + 8, off);
        let r = match self.call(&b, HDR + 8) {
            Ok(r) => r,
            Err(e) => {
                self.hv_give(off);
                return Err(e);
            }
        };
        if le32(&r, 0) != RESP_OK_MAP_INFO || r.len() < HDR + 4 {
            self.hv_give(off);
            return Err(super::virtio::resp_name(le32(&r, 0)));
        }
        Ok((hv.at + off, le32(&r, HDR)))
    }

    pub fn unmap_blob(&mut self, res: u32, at: u64) {
        let mut b = [0u8; HDR + 8];
        header(&mut b, CMD_RESOURCE_UNMAP_BLOB, 0, 0, None);
        put32(&mut b, HDR, res);
        let _ = self.call_ok(&b);
        if let Some(hv) = self.host_visible {
            self.hv_give(at - hv.at);
        }
    }

    /// Detach from `ctx` and drop the host's resource.
    pub fn unref(&mut self, ctx: u32, res: u32) -> Result<(), &'static str> {
        let mut a = [0u8; HDR + 8];
        header(&mut a, CMD_CTX_DETACH_RESOURCE, ctx, 0, None);
        put32(&mut a, HDR, res);
        let _ = self.call_ok(&a);
        let mut b = [0u8; HDR + 8];
        header(&mut b, CMD_RESOURCE_UNREF, 0, 0, None);
        put32(&mut b, HDR, res);
        self.call_ok(&b)
    }

    /// First fit over the host-visible window.
    fn hv_take(&mut self, len: u64, cap: u64) -> Option<u64> {
        let mut at = 0u64;
        let mut i = 0;
        while i < self.hv_used.len() {
            let (o, l) = self.hv_used[i];
            if o >= at + len {
                break;
            }
            at = at.max(o + l);
            i += 1;
        }
        if at + len > cap {
            return None;
        }
        self.hv_used.insert(i, (at, len));
        Some(at)
    }

    fn hv_give(&mut self, off: u64) {
        self.hv_used.retain(|&(o, _)| o != off);
    }

    pub fn hv_mapped(&self) -> u64 {
        self.hv_used.iter().map(|&(_, l)| l).sum()
    }

}

/// What `gpu` in the shell prints about the held device.
pub fn report() {
    use crate::kprintln;
    let g = DEV.lock();
    let Some(d) = g.as_ref() else {
        return kprintln!("  virtio-gpu not opened (a guest opening /dev/dri/renderD128 opens it)");
    };
    kprintln!("  virtio-gpu held at {:02x}:{:02x}.{}", d.slot.0, d.slot.1, d.slot.2);
    kprintln!(
        "  submits {}   fences in flight {}   host-visible mapped {} KiB of {} MiB",
        d.submits,
        d.unsignaled.len(),
        d.hv_mapped() >> 10,
        d.host_visible.map(|r| r.len >> 20).unwrap_or(0)
    );
    kprintln!("  rings woken by the watchdog: {}", crate::linux::drm::KICKS.load(core::sync::atomic::Ordering::Relaxed));
    drop(g);
    crate::linux::drm::report_rings();
    let g = DEV.lock();
    let Some(d) = g.as_ref() else { return };
    let cache = crate::linux::drm::LAST_CACHE.load(core::sync::atomic::Ordering::Relaxed);
    kprintln!(
        "  last mapping's cache type from the host: {}",
        match cache {
            0 => "none mapped yet",
            1 => "cached",
            2 => "uncached",
            3 => "write-combined",
            _ => "unknown",
        }
    );
    if d.async_errors > 0 {
        kprintln!(
            "  {} fenced command(s) answered an error, the last {:#06x} {}",
            d.async_errors,
            d.last_async_error,
            super::virtio::resp_name(d.last_async_error)
        );
    }
}

/// Encoding, with no device.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    let mut h = [0u8; HDR];
    header(&mut h, CMD_SUBMIT_3D, 3, 9, Some(2));
    out.push((
        "a fenced submit carries FENCE and INFO_RING_IDX, its fence at 8, context at 16, ring at 20",
        le32(&h, 0) == 0x0207 && le32(&h, 4) == 3 && h[8] == 9 && le32(&h, 16) == 3 && h[20] == 2,
    ));
    header(&mut h, CMD_SUBMIT_3D, 3, 0, Some(2));
    out.push(("an unfenced command claims no ring", le32(&h, 4) == 0));
    out
}

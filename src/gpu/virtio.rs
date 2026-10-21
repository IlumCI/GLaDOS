//! virtio-gpu, as far as asking whether the host will run Vulkan for us.
//!
//! The question this answers is narrow on purpose. Venus is Vulkan serialised
//! into a ring that the host's own Vulkan driver executes, and on the
//! development host that driver is NVIDIA's, on the GF63's RTX 3050. Whether
//! that combination works at all is settled in three steps, each of which can
//! end the enquiry, and `probe` takes them in that order:
//!
//!   1. Does the device offer `CONTEXT_INIT` and `RESOURCE_BLOB`? Venus needs
//!      both: a context typed by capset, and memory the host allocates.
//!   2. Is the Venus capset (id 4) among the capsets? Its contents name the
//!      wire format and the `vk.xml` version the host renderer was built
//!      against, which is what an encoder on this side has to agree with.
//!   3. Will the host create a Venus context? This is where virglrenderer
//!      starts a render-server worker for us.
//!
//! **Step 3 passing is not proof that NVIDIA's driver runs Venus.** The host
//! creates its `VkInstance` only when the guest sends `vkCreateInstance` down
//! the ring, which needs the command encoder this tree does not have yet. So a
//! clean probe says the transport and the host renderer are there, and the
//! report says so in those words rather than claiming a GPU.
//!
//! Command layouts are the virtio 1.2 specification's, section 5.7. The
//! Venus capset's first four fields follow `venus_hw.h` in virglrenderer (MIT),
//! read for the layout; nothing was copied.

use crate::dev::dma::Dma;
use crate::dev::pci::{self, Device};
use crate::dev::virtio::{self, Transport};
use alloc::string::String;
use alloc::vec::Vec;

pub const F_VIRGL: u64 = 1 << 0;
pub const F_EDID: u64 = 1 << 1;
pub const F_RESOURCE_BLOB: u64 = 1 << 3;
pub const F_CONTEXT_INIT: u64 = 1 << 4;

const CMD_GET_CAPSET_INFO: u32 = 0x0108;
const CMD_GET_CAPSET: u32 = 0x0109;
const CMD_CTX_CREATE: u32 = 0x0200;
const CMD_CTX_DESTROY: u32 = 0x0201;

pub(super) const RESP_OK_NODATA: u32 = 0x1100;
const RESP_OK_CAPSET_INFO: u32 = 0x1102;
const RESP_OK_CAPSET: u32 = 0x1103;

pub const CAPSET_VENUS: u32 = 4;

pub(super) const HDR: usize = 24;
/// Long enough for a render server to start on a loaded host. A control
/// command that is not answered in this long is not going to be.
const WAIT_MS: u64 = 5000;

pub fn capset_name(id: u32) -> &'static str {
    match id {
        1 => "virgl",
        2 => "virgl2",
        3 => "gfxstream-vulkan",
        4 => "venus",
        5 => "cross-domain",
        6 => "drm",
        _ => "",
    }
}

pub fn resp_name(ty: u32) -> &'static str {
    match ty {
        0x1100 => "ok",
        0x1200 => "unspecified error",
        0x1201 => "out of memory",
        0x1202 => "invalid scanout",
        0x1203 => "invalid resource",
        0x1204 => "invalid context",
        0x1205 => "invalid parameter",
        _ => "unexpected response",
    }
}

/// `VK_MAKE_API_VERSION` taken apart: variant 31:29, major 28:22, minor
/// 21:12, patch 11:0. The capset reports `vk.xml`'s version this way.
pub fn vk_version(v: u32) -> (u32, u32, u32) {
    ((v >> 22) & 0x7f, (v >> 12) & 0x3ff, v & 0xfff)
}

/// A control header: type, flags, fence id, context id, ring index.
pub fn hdr(ty: u32, ctx: u32) -> [u8; HDR] {
    let mut h = [0u8; HDR];
    h[0..4].copy_from_slice(&ty.to_le_bytes());
    h[16..20].copy_from_slice(&ctx.to_le_bytes());
    h
}

pub(super) fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

#[derive(Clone, Copy, Debug)]
pub struct CapsetInfo {
    pub id: u32,
    pub max_version: u32,
    pub max_size: u32,
}

/// A configured device with its control queue. Dropping it resets the device
/// before the queue's memory goes back to the heap, because until the reset
/// is acknowledged the device may still read the rings.
pub struct Gpu {
    pub t: Transport,
    pub features: u64,
    pub(super) ctrl: Option<virtio::Queue>,
    req: Dma,
    resp: Dma,
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.t.reset();
        self.ctrl = None;
    }
}

impl Gpu {
    pub fn open(ecam: u64, dev: Device) -> Result<Gpu, &'static str> {
        let t = Transport::open(ecam, dev)?;
        let features = t.negotiate(F_VIRGL | F_EDID | F_RESOURCE_BLOB | F_CONTEXT_INIT)?;
        let ctrl = t.queue(0, 64)?;
        t.ready();
        Ok(Gpu {
            t,
            features,
            ctrl: Some(ctrl),
            // A capset is a few hundred bytes for Venus; 4 KiB leaves room for
            // the larger virgl ones without a second allocation.
            req: Dma::new(256, 16).ok_or("no memory for a request")?,
            resp: Dma::new(4096, 16).ok_or("no memory for a reply")?,
        })
    }

    pub fn num_capsets(&self) -> u32 {
        // Device config: events_read, events_clear, num_scanouts, num_capsets.
        self.t.device_u32(12)
    }

    pub fn num_scanouts(&self) -> u32 {
        self.t.device_u32(8)
    }

    /// Send one command and answer the reply bytes, whose first four are the
    /// response type.
    fn call(&mut self, body: &[u8]) -> Result<&[u8], &'static str> {
        let req = self.req.as_mut_slice();
        req[..body.len()].copy_from_slice(body);
        // The reply buffer is cleared so a short write cannot leave the last
        // command's answer looking like this one's.
        self.resp.as_mut_slice().fill(0);
        let q = self.ctrl.as_mut().ok_or("no control queue")?;
        let n = q.call(&self.req, body.len(), &mut self.resp, WAIT_MS)?;
        if n < HDR {
            return Err("the reply is shorter than a header");
        }
        Ok(&self.resp.as_slice()[..n])
    }

    pub fn capset_info(&mut self, index: u32) -> Result<CapsetInfo, String> {
        let mut b = [0u8; HDR + 8];
        b[..HDR].copy_from_slice(&hdr(CMD_GET_CAPSET_INFO, 0));
        b[HDR..HDR + 4].copy_from_slice(&index.to_le_bytes());
        let r = self.call(&b).map_err(String::from)?;
        let ty = le32(r, 0);
        if ty != RESP_OK_CAPSET_INFO || r.len() < HDR + 12 {
            return Err(alloc::format!("{:#06x} {}", ty, resp_name(ty)));
        }
        Ok(CapsetInfo { id: le32(r, HDR), max_version: le32(r, HDR + 4), max_size: le32(r, HDR + 8) })
    }

    pub fn capset(&mut self, id: u32, version: u32) -> Result<Vec<u8>, String> {
        let mut b = [0u8; HDR + 8];
        b[..HDR].copy_from_slice(&hdr(CMD_GET_CAPSET, 0));
        b[HDR..HDR + 4].copy_from_slice(&id.to_le_bytes());
        b[HDR + 4..HDR + 8].copy_from_slice(&version.to_le_bytes());
        let r = self.call(&b).map_err(String::from)?;
        let ty = le32(r, 0);
        if ty != RESP_OK_CAPSET {
            return Err(alloc::format!("{:#06x} {}", ty, resp_name(ty)));
        }
        Ok(r[HDR..].to_vec())
    }

    /// Create context `ctx` typed by `capset`. Answers the response type.
    pub fn ctx_create(&mut self, ctx: u32, capset: u32, name: &str) -> Result<u32, &'static str> {
        // hdr, nlen, context_init, debug_name[64].
        let mut b = [0u8; HDR + 8 + 64];
        b[..HDR].copy_from_slice(&hdr(CMD_CTX_CREATE, ctx));
        let n = name.len().min(64);
        b[HDR..HDR + 4].copy_from_slice(&(n as u32).to_le_bytes());
        b[HDR + 4..HDR + 8].copy_from_slice(&capset.to_le_bytes());
        b[HDR + 8..HDR + 8 + n].copy_from_slice(&name.as_bytes()[..n]);
        Ok(le32(self.call(&b)?, 0))
    }

    pub fn ctx_destroy(&mut self, ctx: u32) -> Result<u32, &'static str> {
        Ok(le32(self.call(&hdr(CMD_CTX_DESTROY, ctx))?, 0))
    }
}

/// Every virtio-gpu function on the bus.
pub fn find(ecam: u64) -> Vec<Device> {
    let mut out = Vec::new();
    pci::scan(ecam, 255, |d| {
        if d.vendor == virtio::VENDOR && d.device == virtio::DEVICE_GPU {
            out.push(d);
        }
    });
    out
}

/// The shell's `gpu virtio`: take the three steps in the module note and
/// print what each answered.
pub fn probe(ecam: u64) {
    use crate::kprintln;
    let devs = find(ecam);
    if devs.is_empty() {
        kprintln!("  no virtio-gpu on this bus -- under QEMU that is");
        kprintln!("    --qemu-late \"-display egl-headless -device virtio-gpu-gl-pci,venus=on,blob=on,hostmem=256M\"");
        return;
    }
    let d = devs[0];
    kprintln!("  virtio-gpu at {:02x}:{:02x}.{}", d.bus, d.dev, d.func);
    let mut g = match Gpu::open(ecam, d) {
        Ok(g) => g,
        Err(e) => return kprintln!("  refused: {}", e),
    };
    let f = g.features;
    let on = |b: u64| if f & b != 0 { "yes" } else { "no" };
    kprintln!(
        "  features  virgl {}  edid {}  blob {}  context-init {}",
        on(F_VIRGL),
        on(F_EDID),
        on(F_RESOURCE_BLOB),
        on(F_CONTEXT_INIT)
    );
    kprintln!("  queues    {}   scanouts {}   capsets {}", g.t.queues(), g.num_scanouts(), g.num_capsets());
    match g.t.host_visible {
        Some(r) => kprintln!("  host-visible memory  {} MiB at {:#x}", r.len >> 20, r.at),
        None => kprintln!("  host-visible memory  none -- start QEMU with hostmem= for blobs"),
    }

    let mut venus = None;
    for i in 0..g.num_capsets().min(16) {
        match g.capset_info(i) {
            Ok(c) => {
                kprintln!(
                    "  capset {}  id {} {:<16} version {}  {} bytes",
                    i,
                    c.id,
                    capset_name(c.id),
                    c.max_version,
                    c.max_size
                );
                if c.id == CAPSET_VENUS {
                    venus = Some(c);
                }
            }
            Err(e) => kprintln!("  capset {}  {}", i, e),
        }
    }

    let Some(v) = venus else {
        kprintln!("  no Venus capset: the host renderer was built or started without Venus");
        return;
    };
    match g.capset(v.id, v.max_version) {
        Ok(b) if b.len() >= 16 => {
            let (ma, mi, pa) = vk_version(le32(&b, 4));
            kprintln!("  venus     wire format {}   vk.xml {}.{}.{}", le32(&b, 0), ma, mi, pa);
            kprintln!(
                "            command serialization spec {}   venus protocol spec {}",
                le32(&b, 8),
                le32(&b, 12)
            );
        }
        Ok(b) => kprintln!("  venus     capset is only {} bytes", b.len()),
        Err(e) => kprintln!("  venus     capset refused: {}", e),
    }

    if f & (F_CONTEXT_INIT | F_RESOURCE_BLOB) != (F_CONTEXT_INIT | F_RESOURCE_BLOB) {
        kprintln!("  not creating a context: Venus needs both context-init and blob");
        return;
    }
    match g.ctx_create(1, CAPSET_VENUS, "glados") {
        Ok(RESP_OK_NODATA) => {
            kprintln!("  context   created, typed venus");
            let _ = g.ctx_destroy(1);
            kprintln!("  so: the transport and the host's Venus renderer are there.");
            kprintln!("  not yet shown: that the host's Vulkan driver accepts an instance --");
            kprintln!("  that needs vkCreateInstance down a ring, which is the next piece.");
        }
        Ok(ty) => kprintln!("  context   refused: {:#06x} {}", ty, resp_name(ty)),
        Err(e) => kprintln!("  context   {}", e),
    }
}

/// Layout and decoding, with no device.
pub fn selftest() -> Vec<(&'static str, bool)> {
    let mut out = virtio::selftest();
    let h = hdr(CMD_CTX_CREATE, 7);
    out.push(("a control header is 24 bytes, type first and context at 16", h.len() == 24 && h[0] == 0x00 && h[1] == 0x02 && h[16] == 7));
    out.push(("vk.xml 1.4.357 decodes", vk_version((1 << 22) | (4 << 12) | 357) == (1, 4, 357)));
    out.push(("the variant bits are not read as major", vk_version((1 << 29) | (1 << 22)) == (1, 0, 0)));
    out.push(("capset 4 is venus", capset_name(4) == "venus"));
    out.push(("an unknown capset is unnamed rather than guessed", capset_name(99).is_empty()));
    out.push(("0x1204 is an invalid context", resp_name(0x1204) == "invalid context"));
    out
}

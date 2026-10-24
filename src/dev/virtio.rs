//! Virtio over PCI, the modern transport, and one split virtqueue.
//!
//! Written for `gpu::virtio`, which is the first virtio device this kernel has
//! driven: every other part here is real silicon, and virtio exists only under a
//! hypervisor. It earns its place anyway because of what one particular virtio
//! device carries -- Venus, Vulkan forwarded to the host's own driver -- which on
//! the development host means the GF63's RTX 3050, reachable from inside QEMU
//! with none of the GSP bring-up the bare-metal path needs.
//!
//! **Modern only.** The legacy transport is an I/O-port window with a different
//! register layout and no 64-bit queue addresses; a device that offers only that
//! is refused rather than half-driven. Modern is chosen by `VIRTIO_F_VERSION_1`
//! and found through vendor-specific PCI capabilities (id 0x09), each naming a
//! BAR, an offset and a length. All numbers here are the virtio 1.2
//! specification's, section 4.1.
//!
//! **Polled.** No MSI-X is configured -- `pci.rs` has no MSI support at all --
//! so every queue vector is set to `NO_VECTOR` and the used ring is polled, the
//! bargain every other driver in this tree makes. Descriptors go out in pairs
//! from a free list, because virtio-gpu holds a fenced command's answer until
//! the fence signals and a queue with one request in flight would wait for it.

use crate::dev::dma::Dma;
use crate::dev::pci::{self, Device};
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};

pub const VENDOR: u16 = 0x1af4;
/// Modern devices are 0x1040 + the device type; 0x1050 is type 16, the GPU.
pub const DEVICE_GPU: u16 = 0x1050;

const CAP_VENDOR: u8 = 0x09;
const CFG_COMMON: u8 = 1;
const CFG_NOTIFY: u8 = 2;
const CFG_DEVICE: u8 = 4;
const CFG_SHARED_MEMORY: u8 = 8;

pub const F_VERSION_1: u64 = 1 << 32;

const S_ACKNOWLEDGE: u8 = 1;
const S_DRIVER: u8 = 2;
const S_DRIVER_OK: u8 = 4;
const S_FEATURES_OK: u8 = 8;
const S_FAILED: u8 = 128;

const NO_VECTOR: u16 = 0xffff;

// Common configuration, virtio 1.2 section 4.1.4.3.
const C_DFSELECT: u64 = 0x00;
const C_DF: u64 = 0x04;
const C_GFSELECT: u64 = 0x08;
const C_GF: u64 = 0x0c;
const C_MSIX: u64 = 0x10;
const C_NUMQ: u64 = 0x12;
const C_STATUS: u64 = 0x14;
const C_Q_SELECT: u64 = 0x16;
const C_Q_SIZE: u64 = 0x18;
const C_Q_MSIX: u64 = 0x1a;
const C_Q_ENABLE: u64 = 0x1c;
const C_Q_NOTIFY_OFF: u64 = 0x1e;
const C_Q_DESC: u64 = 0x20;
const C_Q_DRIVER: u64 = 0x28;
const C_Q_DEVICE: u64 = 0x30;

const DESC_NEXT: u16 = 1;
const DESC_WRITE: u16 = 2;

/// The PML4 entry `mem::space` keeps for private mappings starts here, and the
/// identity map never reaches it. A BAR the firmware placed above it cannot be
/// mapped by `paging::map_range` without landing in that window.
const MAPPABLE_BELOW: u64 = 512 << 30;

#[inline]
fn r8(a: u64) -> u8 {
    unsafe { read_volatile(a as *const u8) }
}
#[inline]
fn r16(a: u64) -> u16 {
    unsafe { read_volatile(a as *const u16) }
}
#[inline]
fn r32(a: u64) -> u32 {
    unsafe { read_volatile(a as *const u32) }
}
#[inline]
fn w8(a: u64, v: u8) {
    unsafe { write_volatile(a as *mut u8, v) }
}
#[inline]
fn w16(a: u64, v: u16) {
    unsafe { write_volatile(a as *mut u16, v) }
}
#[inline]
fn w32(a: u64, v: u32) {
    unsafe { write_volatile(a as *mut u32, v) }
}
/// A 64-bit field written as two halves, low first, which the spec permits
/// for every 64-bit field in the common configuration.
fn w64(a: u64, v: u64) {
    w32(a, v as u32);
    w32(a + 4, (v >> 32) as u32);
}

/// One region a vendor capability names, resolved to an address.
#[derive(Clone, Copy, Debug, Default)]
pub struct Region {
    pub at: u64,
    pub len: u32,
}

/// The capability list's vendor entries, as `(offset, cfg_type, bar, offset
/// into bar, length, shm id)`. Pure over `read` for the reason `walk_caps` is:
/// a list that loops must end, and that is asserted against synthetic space.
pub fn vendor_caps(read: impl Fn(u64) -> u32) -> alloc::vec::Vec<(u64, u8, u8, u32, u32, u8)> {
    let mut out = alloc::vec::Vec::new();
    const STATUS_CAP_LIST: u32 = 1 << (16 + 4);
    if read(0x04) & STATUS_CAP_LIST == 0 {
        return out;
    }
    let mut off = (read(0x34) & 0xfc) as u64;
    for _ in 0..48 {
        if !(0x40..=0xfc).contains(&off) {
            break;
        }
        let hdr = read(off);
        if (hdr & 0xff) as u8 == CAP_VENDOR {
            let cfg_type = (hdr >> 24) as u8;
            let w1 = read(off + 4);
            let bar = (w1 & 0xff) as u8;
            let id = ((w1 >> 8) & 0xff) as u8;
            out.push((off, cfg_type, bar, read(off + 8), read(off + 12), id));
        }
        off = ((hdr >> 8) & 0xfc) as u64;
    }
    out
}

/// A modern virtio-pci function, its regions mapped.
pub struct Transport {
    common: u64,
    notify: u64,
    notify_mul: u32,
    pub device: Region,
    /// Shared memory region 1, which virtio-gpu calls host-visible memory:
    /// where a blob the host allocated appears in the guest. Unused until a
    /// blob is mapped, and named now so a report can say whether it exists.
    pub host_visible: Option<Region>,
}

impl Transport {
    /// Find, map and enable a modern virtio function.
    ///
    /// Enables memory decoding and bus mastering. Bus mastering is granted here
    /// and not deferred because a virtio device does nothing without DMA -- the
    /// rings are in guest memory -- and unlike a real GPU there is no firmware on
    /// the far side that could misuse it before it is configured.
    pub fn open(ecam: u64, dev: Device) -> Result<Transport, &'static str> {
        if dev.vendor != VENDOR {
            return Err("not a virtio function");
        }
        let caps = vendor_caps(|off| pci::cfg_read32(ecam, &dev, off));
        let resolve = |bar: u8, off: u32, len: u32| -> Result<Region, &'static str> {
            let base = pci::bar(ecam, &dev, bar as usize).ok_or("a capability names an I/O or absent BAR")?;
            if base == 0 {
                return Err("a capability names a BAR the firmware never assigned");
            }
            let at = base + off as u64;
            if at + len as u64 > MAPPABLE_BELOW {
                return Err("a BAR sits above 512 GiB, where the identity map does not reach");
            }
            if !crate::mem::paging::map_range(at, len as u64, true) {
                return Err("could not map a BAR region");
            }
            Ok(Region { at, len })
        };
        let mut common = None;
        let mut notify = None;
        let mut device = None;
        let mut host_visible = None;
        for &(cap_off, ty, bar, off, len, id) in &caps {
            match ty {
                CFG_COMMON if common.is_none() => common = Some(resolve(bar, off, len)?),
                CFG_NOTIFY if notify.is_none() => {
                    let mul = pci::cfg_read32(ecam, &dev, cap_off + 16);
                    notify = Some((resolve(bar, off, len)?, mul));
                }
                CFG_DEVICE if device.is_none() => device = Some(resolve(bar, off, len)?),
                // The shared-memory capability carries a 64-bit offset and length
                // (high halves at +16 and +20). Resolved but not mapped: it can be
                // hundreds of megabytes, and nothing reads it yet.
                CFG_SHARED_MEMORY if id == 1 => {
                    let off_hi = pci::cfg_read32(ecam, &dev, cap_off + 16) as u64;
                    let len_hi = pci::cfg_read32(ecam, &dev, cap_off + 20) as u64;
                    if let Some(base) = pci::bar(ecam, &dev, bar as usize) {
                        let full_len = (len_hi << 32) | len as u64;
                        host_visible = Some(Region {
                            at: base + ((off_hi << 32) | off as u64),
                            len: full_len.min(u32::MAX as u64) as u32,
                        });
                    }
                }
                _ => {}
            }
        }
        let common = common.ok_or("no common configuration: a legacy-only device")?;
        let (notify, notify_mul) = notify.ok_or("no notification capability")?;
        let t = Transport {
            common: common.at,
            notify: notify.at,
            notify_mul,
            device: device.ok_or("no device-specific configuration")?,
            host_visible,
        };
        pci::enable_bus_master(ecam, &dev);
        Ok(t)
    }

    pub fn status(&self) -> u8 {
        r8(self.common + C_STATUS)
    }

    /// Reset, and wait for the device to say it has. A status write of zero is
    /// the request; reading zero back is the acknowledgement, and until then
    /// the device may still be touching the rings this driver is about to free.
    pub fn reset(&self) -> bool {
        w8(self.common + C_STATUS, 0);
        for _ in 0..100_000 {
            if self.status() == 0 {
                return true;
            }
            crate::time::delay_us(10);
        }
        false
    }

    pub fn device_features(&self) -> u64 {
        w32(self.common + C_DFSELECT, 0);
        let lo = r32(self.common + C_DF) as u64;
        w32(self.common + C_DFSELECT, 1);
        let hi = r32(self.common + C_DF) as u64;
        (hi << 32) | lo
    }

    /// Reset, acknowledge, and negotiate `wanted` against what is offered.
    /// Answers the features actually accepted. `VERSION_1` is required: without
    /// it the device is speaking the legacy layout to a driver reading modern.
    pub fn negotiate(&self, wanted: u64) -> Result<u64, &'static str> {
        if !self.reset() {
            return Err("the device did not acknowledge a reset");
        }
        w8(self.common + C_STATUS, S_ACKNOWLEDGE);
        w8(self.common + C_STATUS, S_ACKNOWLEDGE | S_DRIVER);
        let offered = self.device_features();
        if offered & F_VERSION_1 == 0 {
            self.fail();
            return Err("the device does not offer VIRTIO_F_VERSION_1");
        }
        let take = offered & (wanted | F_VERSION_1);
        w32(self.common + C_GFSELECT, 0);
        w32(self.common + C_GF, take as u32);
        w32(self.common + C_GFSELECT, 1);
        w32(self.common + C_GF, (take >> 32) as u32);
        w8(self.common + C_STATUS, S_ACKNOWLEDGE | S_DRIVER | S_FEATURES_OK);
        if self.status() & S_FEATURES_OK == 0 {
            self.fail();
            return Err("the device refused the feature set it offered");
        }
        // No configuration interrupt either; see the module note.
        w16(self.common + C_MSIX, NO_VECTOR);
        Ok(take)
    }

    pub fn queues(&self) -> u16 {
        r16(self.common + C_NUMQ)
    }

    /// Set up queue `index` with at most `max` entries. Must come after
    /// `negotiate` and before `ready`.
    pub fn queue(&self, index: u16, max: u16) -> Result<Queue, &'static str> {
        w16(self.common + C_Q_SELECT, index);
        let offered = r16(self.common + C_Q_SIZE);
        if offered == 0 {
            return Err("that queue does not exist");
        }
        // A split queue's size must be a power of two; the offered size is one,
        // and the largest power of two at or under both is too.
        let mut size = offered.min(max).max(1);
        while !size.is_power_of_two() {
            size &= size - 1;
        }
        w16(self.common + C_Q_SIZE, size);
        let n = size as usize;
        let desc = Dma::new(16 * n, 16).ok_or("no memory for the descriptor table")?;
        let avail = Dma::new(6 + 2 * n, 2).ok_or("no memory for the available ring")?;
        let used = Dma::new(6 + 8 * n, 4).ok_or("no memory for the used ring")?;
        w16(self.common + C_Q_MSIX, NO_VECTOR);
        w64(self.common + C_Q_DESC, desc.pa());
        w64(self.common + C_Q_DRIVER, avail.pa());
        w64(self.common + C_Q_DEVICE, used.pa());
        let off = r16(self.common + C_Q_NOTIFY_OFF) as u64;
        w16(self.common + C_Q_ENABLE, 1);
        Ok(Queue {
            index,
            size,
            desc,
            avail,
            used,
            doorbell: self.notify + off * self.notify_mul as u64,
            next: 0,
            seen: 0,
            free: (0..size / 2).rev().map(|i| i * 2).collect(),
        })
    }

    pub fn ready(&self) {
        w8(self.common + C_STATUS, S_ACKNOWLEDGE | S_DRIVER | S_FEATURES_OK | S_DRIVER_OK);
    }

    pub fn fail(&self) {
        w8(self.common + C_STATUS, self.status() | S_FAILED);
    }

    pub fn device_u32(&self, off: u64) -> u32 {
        r32(self.device.at + off)
    }
}

/// One split virtqueue, its descriptors handed out in pairs.
///
/// Field order matters for the one reason that is silent: `Queue` owns the ring
/// memory, and the device keeps reading it until the transport is reset. The
/// owner must reset before dropping, which `gpu::virtio::Gpu`'s `Drop` does.
///
/// **Several requests may be in flight**, because virtio-gpu answers a fenced
/// command only once the fence has signalled -- that is what a fence *is* on
/// this device -- so a queue that waited for each answer before posting the
/// next would make every `vkQueueSubmit` synchronous with the host's GPU. A
/// request is a pair of descriptors, out and in, and the used ring names the
/// pair by its head; `post` and `reap` are that, and `call` is both with a
/// wait between.
pub struct Queue {
    pub index: u16,
    pub size: u16,
    desc: Dma,
    avail: Dma,
    used: Dma,
    doorbell: u64,
    next: u16,
    seen: u16,
    /// Heads of descriptor pairs nobody is using.
    free: alloc::vec::Vec<u16>,
}

impl Queue {
    /// Publish one request: `out_len` bytes at `out` for the device to read,
    /// `in_len` at `into` for it to write. Answers the head that the
    /// completion will carry, or nothing when every pair is in use.
    pub fn post(&mut self, out: u64, out_len: usize, into: u64, in_len: usize) -> Option<u16> {
        let head = self.free.pop()?;
        let d = self.desc.as_mut_slice();
        put_desc(d, head as usize, out, out_len as u32, DESC_NEXT, head + 1);
        put_desc(d, head as usize + 1, into, in_len as u32, DESC_WRITE, 0);
        let n = self.size as usize;
        let a = self.avail.as_mut_slice();
        let slot = 4 + 2 * (self.next as usize % n);
        a[slot..slot + 2].copy_from_slice(&head.to_le_bytes());
        // The descriptor and the ring entry must be visible before the index
        // that publishes them, and the index before the doorbell.
        fence(Ordering::SeqCst);
        self.next = self.next.wrapping_add(1);
        unsafe { write_volatile(a.as_mut_ptr().add(2) as *mut u16, self.next) };
        fence(Ordering::SeqCst);
        w16(self.doorbell, self.index);
        Some(head)
    }

    /// One finished request, as its head and the byte count the device wrote,
    /// with the pair returned to the free list. Nothing when none has
    /// finished since the last reap.
    pub fn reap(&mut self) -> Option<(u16, usize)> {
        let used = self.used.as_slice().as_ptr();
        let idx = unsafe { read_volatile(used.add(2) as *const u16) };
        if idx == self.seen {
            return None;
        }
        fence(Ordering::SeqCst);
        let e = 4 + 8 * (self.seen as usize % self.size as usize);
        let u = self.used.as_slice();
        let id = u32::from_le_bytes([u[e], u[e + 1], u[e + 2], u[e + 3]]) as u16;
        let len = u32::from_le_bytes([u[e + 4], u[e + 5], u[e + 6], u[e + 7]]) as usize;
        self.seen = self.seen.wrapping_add(1);
        self.free.push(id);
        Some((id, len))
    }

    /// Send `out` (device-readable), receive into `into` (device-writable), and
    /// wait up to `ms` for the device to finish. Answers the byte count the
    /// device says it wrote.
    ///
    /// Only for a queue nothing else has requests in flight on: a completion
    /// for somebody else's head would be taken as this one's. `gpu::virtgpu`
    /// owns its queue outright and uses `post` and `reap` instead.
    pub fn call(&mut self, out: &Dma, out_len: usize, into: &mut Dma, ms: u64) -> Result<usize, &'static str> {
        if self.size < 2 {
            return Err("the queue is too short for a request and its reply");
        }
        let head = self.post(out.pa(), out_len, into.pa(), into.len()).ok_or("every descriptor is in use")?;
        let deadline = crate::time::rdtsc() + crate::time::tsc_mhz().max(1) * 1000 * ms;
        loop {
            if let Some((id, len)) = self.reap() {
                if id == head {
                    return Ok(len);
                }
            }
            if crate::time::rdtsc() > deadline {
                return Err("the device did not answer in time");
            }
            core::hint::spin_loop();
        }
    }
}

fn put_desc(table: &mut [u8], i: usize, addr: u64, len: u32, flags: u16, next: u16) {
    let e = &mut table[16 * i..16 * i + 16];
    e[0..8].copy_from_slice(&addr.to_le_bytes());
    e[8..12].copy_from_slice(&len.to_le_bytes());
    e[12..14].copy_from_slice(&flags.to_le_bytes());
    e[14..16].copy_from_slice(&next.to_le_bytes());
}

/// The capability walk, against synthetic config space.
pub fn selftest() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out = alloc::vec::Vec::new();
    let space = |words: &'static [(u64, u32)]| {
        move |off: u64| words.iter().find(|(o, _)| *o == off).map(|(_, v)| *v).unwrap_or(0)
    };
    const LIST: u32 = 1 << 20;
    // Two vendor caps (common in BAR 4 at 0, notify in BAR 4 at 0x3000) with a
    // power-management cap between them that must be stepped over.
    static TWO: [(u64, u32); 10] = [
        (0x04, LIST),
        (0x34, 0x40),
        (0x40, 0x0100_5009), // vendor, next 0x50, len 0x10, type 1
        (0x44, 0x04),
        (0x48, 0),
        (0x4c, 0x1000),
        (0x50, 0x0000_6001), // PM, next 0x60
        (0x60, 0x0200_0009), // vendor, last, type 2
        (0x64, 0x04),
        (0x68, 0x3000),
    ];
    let caps = vendor_caps(space(&TWO));
    out.push(("both vendor capabilities are found, and the PM one between them skipped", caps.len() == 2));
    out.push(("the first is common configuration in BAR 4", caps.first().map(|c| (c.1, c.2)) == Some((1, 4))));
    out.push(("the second is notification at offset 0x3000", caps.get(1).map(|c| (c.1, c.3)) == Some((2, 0x3000))));
    static LOOP: [(u64, u32); 3] = [(0x04, LIST), (0x34, 0x40), (0x40, 0x0100_4009)];
    out.push(("a vendor capability that points at itself ends the walk", vendor_caps(space(&LOOP)).len() <= 48));
    static NONE: [(u64, u32); 3] = [(0x04, 0), (0x34, 0x40), (0x40, 0x0100_0009)];
    out.push(("no list in the status register means no capabilities", vendor_caps(space(&NONE)).is_empty()));
    let mut t = [0u8; 32];
    put_desc(&mut t, 1, 0x1122_3344_5566_7788, 24, DESC_WRITE, 0);
    out.push((
        "a descriptor is laid out address, length, flags, next",
        t[16..24] == 0x1122_3344_5566_7788u64.to_le_bytes() && t[24] == 24 && t[28] == 2,
    ));
    out
}

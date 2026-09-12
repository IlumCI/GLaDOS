//! What a radio driver needs from the machine it is plugged into.
//!
//! This half of the seam exists only for the drivers, not for the stack above
//! them -- `net80211` never touches a register. It is separate from the rest
//! for that reason: the core can be ported, reviewed and tested with this
//! module entirely unused, which is what makes a simulated access point
//! possible at all.
//!
//! ### Identity mapping is load-bearing here
//!
//! A device reads physical addresses; the kernel writes virtual ones. Those
//! are the same number in GLaDOS, which is why `dma` can answer a single `u64`
//! and mean both. `xhci.rs:42-49` records this as the assumption that would
//! break first if the kernel ever grew a higher-half mapping, and this module
//! inherits that dependency exactly.
//!
//! ### There are no interrupts
//!
//! Nothing here registers a handler, because nothing in this kernel can. Every
//! driver polls, `e1000.rs:7-9` argues that is a correctness property rather
//! than a gap, and a radio driver pays for it in throughput and in latency on
//! the beacon path. Stated here so a ported driver's interrupt routine has an
//! obvious place to become a poll.

use alloc::vec::Vec;

/// Where a device lives on the PCI bus.
///
/// A copy of the fields `dev::pci::Device` carries rather than the type
/// itself, because a ported tree may not name `crate::dev`. The conversion is
/// on the kernel side of the seam, which is where every other `impl` for a
/// ported type already lives.
#[derive(Clone, Copy)]
pub struct Pci {
    pub bus: u8,
    pub dev: u8,
    pub func: u8,
    pub vendor: u16,
    pub device: u16,
}

/// A mapped memory-mapped I/O region, with the accessors a register table
/// wants.
///
/// The reads and writes are `volatile` and that is not optional: a register
/// read whose result is discarded is often the *point* -- it acknowledges an
/// interrupt or advances a FIFO -- and an optimiser that removes it produces a
/// driver that works in debug and hangs in release.
pub struct Mmio {
    base: u64,
    len: usize,
}

impl Mmio {
    /// Map `len` bytes of device memory at `phys`, uncached.
    ///
    /// Uncached because a device changes memory underneath the processor: a
    /// cached mapping means a status register read answers from a cache line
    /// filled before the event being waited for.
    pub fn map(phys: u64, len: usize) -> Option<Mmio> {
        if phys == 0 || len == 0 {
            return None;
        }
        if !crate::mem::paging::map_range(phys, len as u64, true) {
            return None;
        }
        Some(Mmio { base: phys, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Read a 32-bit register. Out of range answers `0xFFFF_FFFF`, which is
    /// what a bus answers for an absent device, so a driver's existing
    /// "is this all ones" check catches it rather than a panic.
    pub fn r32(&self, off: usize) -> u32 {
        if off + 4 > self.len {
            return 0xFFFF_FFFF;
        }
        unsafe { core::ptr::read_volatile((self.base + off as u64) as *const u32) }
    }

    /// Write a 32-bit register. Out of range is dropped rather than wrapping
    /// into whatever follows the aperture.
    pub fn w32(&self, off: usize, v: u32) {
        if off + 4 > self.len {
            return;
        }
        unsafe { core::ptr::write_volatile((self.base + off as u64) as *mut u32, v) }
    }

    pub fn r8(&self, off: usize) -> u8 {
        if off >= self.len {
            return 0xFF;
        }
        unsafe { core::ptr::read_volatile((self.base + off as u64) as *const u8) }
    }

    pub fn w8(&self, off: usize, v: u8) {
        if off >= self.len {
            return;
        }
        unsafe { core::ptr::write_volatile((self.base + off as u64) as *mut u8, v) }
    }
}

/// Memory a device can read and write.
///
/// Answers one address because virtual is physical here. It is zeroed, and
/// never freed: the same bargain `xhci::dma` makes, on the grounds that a ring
/// handed back to the allocator while a controller still holds its address is
/// a corruption with no symptom near the cause.
pub fn dma(size: usize, align: usize) -> Option<u64> {
    crate::dev::xhci::dma(size, align)
}

/// Read a device's PCI configuration space.
///
/// Answers all ones when there is no ECAM window, which is what a bus answers
/// for a device that is not there -- so a driver's existing "is this all ones"
/// check catches a machine with no config space at all, rather than this
/// guessing at a base address and reading somebody else's memory.
pub fn cfg_read32(p: &Pci, off: u64) -> u32 {
    match crate::net::ecam() {
        Some(e) => crate::dev::pci::cfg_read32_at(e, p.bus, p.dev, p.func, off),
        None => 0xFFFF_FFFF,
    }
}

/// Write a device's PCI configuration space. Dropped when there is no ECAM.
pub fn cfg_write32(p: &Pci, off: u64, v: u32) {
    if let Some(e) = crate::net::ecam() {
        crate::dev::pci::cfg_write32_at(e, p.bus, p.dev, p.func, off, v);
    }
}

/// Every wireless device the registry knows is present.
///
/// The registry is asked rather than the bus swept, for the reason
/// `registry.rs` gives: nine drivers sweeping nine times each, with nine
/// private id lists, is how a machine ends up with two files that disagree
/// about what is fitted.
pub fn wireless() -> Vec<Pci> {
    crate::dev::registry::wireless_pci()
}

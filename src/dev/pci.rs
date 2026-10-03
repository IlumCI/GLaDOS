//! PCI Express enumeration via ECAM.
//!
//! The old way to reach PCI config space was the 0xCF8/0xCFC port pair, which
//! is serialised, 32-bit at a time, and only reaches the first 256 bytes of
//! each function's config space. PCIe instead memory-maps the whole thing, and
//! the MCFG ACPI table tells us where. On this laptop that base is
//! 0xc0000000; under QEMU it is 0xe0000000.
//!
//! Address arithmetic is fixed by the spec:
//!
//!   ecam + (bus << 20) + (device << 15) + (function << 12) + offset
//!
//! Note this only works because the identity map marks the ECAM window
//! uncacheable. Read through a write-back mapping it returns stale nonsense,
//! which is exactly how the IOAPIC came back claiming 120 redirection entries.

use core::ptr::read_volatile;

#[derive(Clone, Copy, Debug)]
pub struct Device {
    pub bus: u8,
    pub dev: u8,
    pub func: u8,
    pub vendor: u16,
    pub device: u16,
    pub class: u8,
    pub subclass: u8,
    pub prog_if: u8,
    pub header_type: u8,
}

#[inline]
fn cfg_addr(ecam: u64, bus: u8, dev: u8, func: u8) -> u64 {
    ecam + ((bus as u64) << 20) + ((dev as u64) << 15) + ((func as u64) << 12)
}

#[inline]
unsafe fn read16(base: u64, off: u64) -> u16 {
    unsafe { read_volatile((base + off) as *const u16) }
}

#[inline]
unsafe fn read8(base: u64, off: u64) -> u8 {
    unsafe { read_volatile((base + off) as *const u8) }
}

fn probe(ecam: u64, bus: u8, dev: u8, func: u8) -> Option<Device> {
    let base = cfg_addr(ecam, bus, dev, func);
    let vendor = unsafe { read16(base, 0x00) };
    // 0xFFFF is what an absent function reads back as.
    if vendor == 0xFFFF || vendor == 0x0000 {
        return None;
    }
    Some(Device {
        bus,
        dev,
        func,
        vendor,
        device: unsafe { read16(base, 0x02) },
        prog_if: unsafe { read8(base, 0x09) },
        subclass: unsafe { read8(base, 0x0A) },
        class: unsafe { read8(base, 0x0B) },
        header_type: unsafe { read8(base, 0x0E) },
    })
}

/// Walk config space, calling `f` for every function that answers.
///
/// A brute-force sweep rather than a recursive bridge walk. It is a few
/// hundred thousand uncached reads in the worst case, which is milliseconds,
/// and it cannot miss a device behind a bridge we failed to follow.
pub fn scan(ecam: u64, max_bus: u16, mut f: impl FnMut(Device)) {
    for bus in 0..=max_bus.min(255) {
        for dev in 0u8..32 {
            // Function 0 must exist for any device to be present at all.
            let Some(d0) = probe(ecam, bus as u8, dev, 0) else {
                continue;
            };
            f(d0);
            // Bit 7 of the header type marks a multi-function device.
            if d0.header_type & 0x80 == 0 {
                continue;
            }
            for func in 1u8..8 {
                if let Some(d) = probe(ecam, bus as u8, dev, func) {
                    f(d);
                }
            }
        }
    }
}

#[inline]
unsafe fn read32(base: u64, off: u64) -> u32 {
    unsafe { read_volatile((base + off) as *const u32) }
}

#[inline]
unsafe fn write32(base: u64, off: u64, v: u32) {
    unsafe { core::ptr::write_volatile((base + off) as *mut u32, v) }
}

/// Read a Base Address Register, resolving 64-bit BARs from their two halves.
///
/// Bit 0 selects I/O vs memory space; bits 2:1 encode the type, where `0b10`
/// means the BAR is 64 bits wide and consumes the following slot as its high
/// half. Reading only the low half of a 64-bit BAR yields an address that is
/// plausible and wrong, which on this machine matters: the framebuffer BAR
/// sits at 0x40_0000_0000.
pub fn bar(io_base_ecam: u64, d: &Device, index: usize) -> Option<u64> {
    if index >= 6 {
        return None;
    }
    let cfg = cfg_addr(io_base_ecam, d.bus, d.dev, d.func);
    let off = 0x10 + index as u64 * 4;
    let lo = unsafe { read32(cfg, off) };
    if lo & 1 != 0 {
        return None; // I/O space, not memory-mapped
    }
    let kind = (lo >> 1) & 0b11;
    let base = (lo & 0xFFFF_FFF0) as u64;
    if kind == 0b10 {
        let hi = unsafe { read32(cfg, off + 4) } as u64;
        Some((hi << 32) | base)
    } else {
        Some(base)
    }
}

/// Set the bus-master and memory-space enable bits in the command register.
///
/// Without bus-master a device cannot DMA, so an NVMe controller will accept
/// commands and never write a completion -- it looks exactly like a hung
/// controller.
pub fn enable_bus_master(io_base_ecam: u64, d: &Device) {
    let cfg = cfg_addr(io_base_ecam, d.bus, d.dev, d.func);
    let cmd = unsafe { read32(cfg, 0x04) };
    unsafe { write32(cfg, 0x04, cmd | (1 << 1) | (1 << 2)) };
}

/// Plain-English class name. Not exhaustive -- just what turns up in a laptop.
pub fn class_name(class: u8, subclass: u8) -> &'static str {
    match (class, subclass) {
        (0x00, _) => "unclassified",
        (0x01, 0x00) => "SCSI controller",
        (0x01, 0x01) => "IDE controller",
        (0x01, 0x06) => "SATA controller",
        (0x01, 0x08) => "NVMe controller",
        (0x01, _) => "storage controller",
        (0x02, 0x00) => "ethernet",
        (0x02, 0x80) => "network controller",
        (0x02, _) => "network",
        (0x03, 0x00) => "VGA display",
        (0x03, _) => "display",
        (0x04, 0x00) => "multimedia video",
        (0x04, 0x01) => "audio (legacy)",
        (0x04, 0x03) => "audio device",
        (0x04, _) => "multimedia",
        (0x05, _) => "memory controller",
        (0x06, 0x00) => "host bridge",
        (0x06, 0x01) => "ISA bridge",
        (0x06, 0x04) => "PCI-to-PCI bridge",
        (0x06, _) => "bridge",
        (0x07, _) => "communication controller",
        (0x08, _) => "system peripheral",
        (0x09, _) => "input device",
        (0x0A, _) => "docking station",
        (0x0B, _) => "processor",
        (0x0C, 0x03) => "USB controller",
        (0x0C, 0x05) => "SMBus",
        (0x0C, _) => "serial bus",
        (0x0D, _) => "wireless controller",
        (0x0E, _) => "intelligent controller",
        (0x0F, _) => "satellite comms",
        (0x10, _) => "encryption",
        (0x11, _) => "signal processing",
        (0x12, _) => "processing accelerator",
        _ => "unknown",
    }
}

/// A few vendor IDs worth naming on sight.
pub fn vendor_name(vendor: u16) -> &'static str {
    match vendor {
        0x8086 => "Intel",
        0x10DE => "NVIDIA",
        0x1022 => "AMD",
        0x1002 => "AMD/ATI",
        0x10EC => "Realtek",
        0x1969 => "Qualcomm Atheros",
        0x14E4 => "Broadcom",
        0x1B21 => "ASMedia",
        0x144D => "Samsung",
        0x1E0F => "KIOXIA",
        0x1987 => "Phison",
        0x2646 => "Kingston",
        0x1AF4 => "Red Hat / virtio",
        0x1B36 => "Red Hat",
        0x1234 => "QEMU",
        _ => "",
    }
}

/// Read a 32-bit word from a function's config space.
///
/// `bar` and `enable_bus_master` reach config space through private helpers
/// because they were the only two callers. Probing a device that has no
/// driver needs more: the revision at 0x08, the subsystem id at 0x2c, a
/// bridge's secondary bus at 0x18, and eventually the capability list at
/// 0x34. Exposing the accessor is cheaper than growing a named function per
/// field.
pub fn cfg_read32(ecam: u64, d: &Device, off: u64) -> u32 {
    unsafe { read32(cfg_addr(ecam, d.bus, d.dev, d.func), off) }
}

/// The standard capability list, walked through any reader of config space.
///
/// Pure over `read` so the walk can be asserted against a synthetic config space,
/// including the malformed one that matters: **a list that points at itself.**
/// Config space is 256 bytes of standard capabilities, so no honest walk is
/// longer than 48 steps, and a pointer below 0x40 lands in the header. Either
/// ends the walk with nothing rather than reading forever, which in ring 0 is a
/// hung boot on a part whose firmware wrote one byte wrong.
///
/// The capability pointer is only meaningful when the status register says the
/// list exists (bit 4 of the word at 0x04's high half), and an absent list reads
/// back as whatever the header holds there.
pub fn walk_caps(read: impl Fn(u64) -> u32, id: u8) -> Option<u64> {
    const STATUS_CAP_LIST: u32 = 1 << (16 + 4);
    if read(0x04) & STATUS_CAP_LIST == 0 {
        return None;
    }
    let mut off = (read(0x34) & 0xfc) as u64;
    for _ in 0..48 {
        if off < 0x40 || off > 0xfc {
            return None;
        }
        let hdr = read(off);
        if (hdr & 0xff) as u8 == id {
            return Some(off);
        }
        off = ((hdr >> 8) & 0xfc) as u64;
    }
    None
}

/// Where a function's capability `id` lives in its config space, if it has one.
pub fn find_cap(ecam: u64, d: &Device, id: u8) -> Option<u64> {
    walk_caps(|off| cfg_read32(ecam, d, off), id)
}

pub const CAP_PM: u8 = 0x01;
pub const CAP_PCIE: u8 = 0x10;

/// The power state from PMCSR: 0 is D0, 3 is D3hot. `None` when the function
/// has no power-management capability, which is a legal thing to lack.
pub fn power_state(ecam: u64, d: &Device) -> Option<u8> {
    let pm = find_cap(ecam, d, CAP_PM)?;
    Some((cfg_read32(ecam, d, pm + 4) & 0b11) as u8)
}

/// Bring a function to D0, answering the state it was found in.
///
/// **This is the fix for one of the two things an all-ones register read
/// means.** A part parked in D3hot answers its config space and not its BARs, so
/// a driver that reads `0xFFFFFFFF` from its first register cannot tell a sleeping
/// part from one whose decoder is off -- and a laptop's firmware is entitled to
/// leave its radio asleep when nothing in the boot path asked for it. D3cold is
/// not fixable from here: the function has no power and its config space reads
/// all ones too, which is what `None` from `find_cap` then looks like.
///
/// The 10 ms is the specification's recovery time from D3hot, and it is not a
/// guess to tighten: the first access inside it may be dropped, and a dropped
/// access to a BAR is the all-ones read this function exists to stop.
pub fn set_d0(ecam: u64, d: &Device) -> Option<u8> {
    let pm = find_cap(ecam, d, CAP_PM)?;
    let csr = cfg_read32(ecam, d, pm + 4);
    let was = (csr & 0b11) as u8;
    if was != 0 {
        // **D3hot to D0 resets the function unless it says it will not.** With
        // No_Soft_Reset (bit 3) clear the transition is a reset: BARs, the
        // command register and the interrupt line come back as power-on
        // defaults, and a caller holding the address it read before would map
        // nothing. So the header is saved first and put back after, which is
        // what the specification asks of system software in exactly this case.
        // Intel's radios usually set the bit; a generic helper cannot assume so.
        let soft_reset = was == 3 && csr & (1 << 3) == 0;
        let saved: [u32; 6] = core::array::from_fn(|i| cfg_read32(ecam, d, 0x10 + 4 * i as u64));
        let cmd = cfg_read32(ecam, d, 0x04) & 0xffff;
        let line = cfg_read32(ecam, d, 0x3c);
        // Bit 15 is PME status, write-one-to-clear: writing it back as read
        // would clear a wake event somebody else may be waiting on.
        cfg_write32(ecam, d, pm + 4, csr & !0b11 & !(1 << 15));
        crate::time::delay_us(10_000);
        if soft_reset {
            for (i, v) in saved.iter().enumerate() {
                cfg_write32(ecam, d, 0x10 + 4 * i as u64, *v);
            }
            cfg_write32(ecam, d, 0x3c, line);
            // The command register last: decoding turned on before the BARs are
            // back would decode whatever the reset left in them.
            cfg_write32(ecam, d, 0x04, cmd);
        }
    }
    Some(was)
}

/// Clear bus mastering, so the function can no longer reach memory.
///
/// The half of `enable_bus_master` a driver needs on the way out: freeing memory
/// a device was told it could write is the bug, and this is the step that makes
/// freeing it safe.
/// Answers whether the bit reads back clear. A configuration write may be
/// posted, so the read is what orders it ahead of whatever the caller does next
/// -- freeing the memory the device was mastering, usually -- and a device that
/// has gone (all ones) answers `false`, which is the honest answer about it.
pub fn disable_bus_master(ecam: u64, d: &Device) -> bool {
    let cmd = cfg_read32(ecam, d, 0x04);
    // Only the low half: the high half is status, write-one-to-clear.
    cfg_write32(ecam, d, 0x04, cmd & 0xffff & !(1 << 2));
    let back = cfg_read32(ecam, d, 0x04);
    back != 0xffff_ffff && back & (1 << 2) == 0
}

/// The capability walk, against synthetic config spaces.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    use alloc::vec::Vec;
    // A config space as a sparse list of (offset, word).
    fn space(words: &[(u64, u32)]) -> impl Fn(u64) -> u32 + '_ {
        move |off| words.iter().find(|w| w.0 == off).map(|w| w.1).unwrap_or(0)
    }
    let caps = 1u32 << 20;
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    // PM at 0x40 -> MSI at 0x50 -> PCIe at 0x70 -> end.
    let good = [(0x04, caps), (0x34, 0x40), (0x40, 0x5001), (0x50, 0x7005), (0x70, 0x0010)];
    out.push(("a capability at the head of the list is found", walk_caps(space(&good), CAP_PM) == Some(0x40)));
    out.push(("and one at its tail, two hops on", walk_caps(space(&good), CAP_PCIE) == Some(0x70)));
    out.push(("one the list does not carry is absent", walk_caps(space(&good), 0x11).is_none()));
    let looping = [(0x04, caps), (0x34, 0x40), (0x40, 0x4005)];
    out.push(("a list that points at itself ends rather than spinning", walk_caps(space(&looping), CAP_PCIE).is_none()));
    let into_header = [(0x04, caps), (0x34, 0x40), (0x40, 0x1005)];
    out.push(("a pointer back into the header ends the walk", walk_caps(space(&into_header), CAP_PCIE).is_none()));
    let no_list = [(0x04, 0), (0x34, 0x40), (0x40, 0x0001)];
    out.push((
        "a pointer is not followed when the status register says there is no list",
        walk_caps(space(&no_list), CAP_PM).is_none(),
    ));
    // The low two bits of the pointer are reserved and must be masked off.
    let unaligned = [(0x04, caps), (0x34, 0x43), (0x40, 0x0001)];
    out.push(("the reserved low bits of the pointer are masked", walk_caps(space(&unaligned), CAP_PM) == Some(0x40)));
    out
}

/// Write a 32-bit word to a function's config space.
pub fn cfg_write32(ecam: u64, d: &Device, off: u64, v: u32) {
    unsafe { write32(cfg_addr(ecam, d.bus, d.dev, d.func), off, v) }
}

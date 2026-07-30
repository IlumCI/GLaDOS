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

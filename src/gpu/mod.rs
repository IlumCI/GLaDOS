//! Does this machine have a GPU, and will it answer?
//!
//! Nothing in this kernel has ever spoken to the graphics card. `src/gfx` is a
//! linear framebuffer that UEFI handed over before `ExitBootServices`, and on
//! this laptop that framebuffer belongs to the Intel part; the discrete NVIDIA
//! chip drives no display at all and has sat idle since the project began.
//!
//! This module answers the three questions that have to be settled before any
//! of that can change, in the order that lets each one end the enquiry:
//!
//!   1. Is the device on the bus? `pci::scan` already sweeps every function
//!      and already knows 0x10de and class 0x03 by name, so this is a filter
//!      over machinery that was finished before the question was asked.
//!   2. If it is not, is that because it is absent or because it is asleep?
//!      A muxless Optimus laptop parks the discrete GPU in D3cold, where
//!      config space reads 0xFFFF and the device is indistinguishable from one
//!      that was never fitted. `empty_bridges` separates the two cases.
//!   3. If it is there, does MMIO work? NV_PMC_BOOT_0 has lived at offset 0
//!      of BAR0 on every NVIDIA part ever made and encodes the chip id. A
//!      correct answer means the device is alive and addressable from ring 0,
//!      which is the entire question this module exists to settle.
//!
//! Reading a device register is the one genuinely dangerous act here. Every
//! exception vector in this kernel except #BP diverges (`src/cpu/idt.rs`), so
//! a bad MMIO read is a halted machine and a register dump, not an error
//! return. `boot0` therefore refuses on every condition it can name in
//! advance rather than reading and hoping.

use crate::dev::pci::{self, Device};
use alloc::string::String;
use alloc::vec::Vec;

pub const CLASS_DISPLAY: u8 = 0x03;
pub const VENDOR_NVIDIA: u16 = 0x10DE;

/// Class 0x06 subclass 0x04: a PCI-to-PCI bridge. The root port a discrete
/// laptop GPU hangs off is one of these.
pub const CLASS_BRIDGE: u8 = 0x06;
pub const SUBCLASS_PCI_BRIDGE: u8 = 0x04;

/// NV_PMC_BOOT_0, at offset 0 of BAR0.
pub const NV_PMC_BOOT_0: u64 = 0x0000;

/// Command register bit 1, memory space enable.
const CMD_MEMORY_SPACE: u32 = 1 << 1;

pub const PRESENT: &str = "/sys/gpu/present";
pub const ID: &str = "/sys/gpu/id";
pub const CHIP: &str = "/sys/gpu/chip";

#[derive(Clone, Copy)]
pub struct Gpu {
    pub dev: Device,
    pub revision: u8,
    pub subsystem: u32,
    /// Register aperture. 16 MiB on every NVIDIA part since Fermi.
    pub bar0: Option<u64>,
    /// The VRAM window. 256 MiB by default, resizable to the whole of VRAM
    /// where the firmware supports it.
    pub bar1: Option<u64>,
    /// A second, smaller aperture. Present on some parts, absent on others.
    pub bar3: Option<u64>,
}

impl Gpu {
    pub fn is_nvidia(&self) -> bool {
        self.dev.vendor == VENDOR_NVIDIA
    }

    /// "10de:25a2", the form lspci prints and the form a bug report needs.
    pub fn id(&self) -> String {
        alloc::format!("{:04x}:{:04x}", self.dev.vendor, self.dev.device)
    }

    /// Map BAR0 and read NV_PMC_BOOT_0.
    ///
    /// Every refusal below is a case where the read would fault or return
    /// nonsense, and naming them is cheaper than recovering from them: this
    /// kernel cannot recover from them at all.
    ///
    /// Memory-space decoding is enabled first because a BAR whose decoder is
    /// off reads back as all ones, which decodes to a plausible-looking and
    /// entirely fictional chip. Bus mastering is deliberately left alone: a
    /// probe has no business granting DMA to a device whose firmware has
    /// never run.
    pub fn boot0(&self, ecam: u64) -> Option<u32> {
        let bar0 = self.bar0?;
        // An unassigned BAR reads as zero. Mapping and reading page zero is
        // exactly the fault the identity map leaves unmapped on purpose.
        if bar0 == 0 {
            return None;
        }
        enable_memory_space(ecam, &self.dev);
        // One 2 MiB page covers offset 0; map_range rounds up to that anyway.
        if !crate::mem::paging::map_range(bar0, 0x1000, true) {
            return None;
        }
        let raw = unsafe { core::ptr::read_volatile((bar0 + NV_PMC_BOOT_0) as *const u32) };
        Some(raw)
    }
}

/// Enable memory-space decoding, leaving bus-master untouched.
///
/// `pci::enable_bus_master` sets both bits at once because its two callers
/// wanted both. Probing wants only the first.
fn enable_memory_space(ecam: u64, d: &Device) {
    let cmd = pci::cfg_read32(ecam, d, 0x04);
    if cmd & CMD_MEMORY_SPACE == 0 {
        pci::cfg_write32(ecam, d, 0x04, cmd | CMD_MEMORY_SPACE);
    }
}

/// The first display controller on the bus, NVIDIA preferred.
///
/// Preferred rather than required: on this laptop the Intel part answers too,
/// and reporting "found a GPU" while silently meaning the wrong one is the
/// kind of result that costs an afternoon.
pub fn find(ecam: u64) -> Option<Gpu> {
    let mut best: Option<Device> = None;
    pci::scan(ecam, 255, |d| {
        if d.class != CLASS_DISPLAY {
            return;
        }
        match best {
            // An NVIDIA part always wins; otherwise the first one found.
            Some(cur) if cur.vendor == VENDOR_NVIDIA => {}
            Some(_) if d.vendor == VENDOR_NVIDIA => best = Some(d),
            Some(_) => {}
            None => best = Some(d),
        }
    });
    let dev = best?;
    Some(Gpu {
        dev,
        revision: (pci::cfg_read32(ecam, &dev, 0x08) & 0xff) as u8,
        subsystem: pci::cfg_read32(ecam, &dev, 0x2c),
        bar0: pci::bar(ecam, &dev, 0),
        bar1: pci::bar(ecam, &dev, 1),
        bar3: pci::bar(ecam, &dev, 3),
    })
}

/// Every display controller on the bus, so the Intel part can be named rather
/// than merely lost to the preference in `find`.
pub fn all(ecam: u64) -> Vec<Device> {
    let mut out = Vec::new();
    pci::scan(ecam, 255, |d| {
        if d.class == CLASS_DISPLAY {
            out.push(d);
        }
    });
    out
}

/// Bridges forwarding a secondary bus that nothing answers on.
///
/// This is the D3cold tell. A powered-down discrete GPU vanishes from config
/// space completely, so "no NVIDIA device found" and "no NVIDIA device
/// fitted" read identically. The root port it hangs off does not go anywhere,
/// though, and a bridge forwarding an empty bus is a strong hint that
/// something is there and asleep. That is the difference between abandoning
/// this and calling an ACPI _ON method.
pub fn empty_bridges(ecam: u64) -> Vec<(Device, u8)> {
    let mut bridges = Vec::new();
    let mut occupied = Vec::new();
    pci::scan(ecam, 255, |d| {
        if d.class == CLASS_BRIDGE && d.subclass == SUBCLASS_PCI_BRIDGE {
            // Secondary bus number lives at config offset 0x19.
            let secondary = ((pci::cfg_read32(ecam, &d, 0x18) >> 8) & 0xff) as u8;
            bridges.push((d, secondary));
        }
        occupied.push(d.bus);
    });
    bridges.retain(|(_, secondary)| *secondary != 0 && !occupied.contains(secondary));
    bridges
}

/// Split NV_PMC_BOOT_0 into (chipset, revision).
///
/// Fermi and later put the chipset id in bits 28:20 and the revision in the
/// low byte. Testing bits 28:24 for a non-zero value is nouveau's own check
/// for "this is the Fermi-or-later layout"; on older parts those bits mean
/// something else, and this answers None rather than confidently decoding a
/// field that is not there.
///
/// All-ones is the other case worth naming: it is what a device reads back as
/// when its decoder is off or it is not answering at all, and it would
/// otherwise decode to a chipset of 0x1ff.
pub fn decode_boot0(raw: u32) -> Option<(u32, u32)> {
    if raw == 0 || raw == 0xFFFF_FFFF {
        return None;
    }
    if raw & 0x1f00_0000 == 0 {
        return None;
    }
    Some(((raw & 0x1ff0_0000) >> 20, raw & 0xff))
}

/// The die name for a chipset id, or an empty string when the table does not
/// know it.
///
/// Empty rather than "unknown" so a caller can decide to print the raw value
/// instead. The table only claims parts worth claiming; anything missing
/// still reports its family and its raw register, which for a probe is the
/// part that matters.
pub fn chip_name(chipset: u32) -> &'static str {
    match chipset {
        0x162 => "TU102",
        0x164 => "TU104",
        0x166 => "TU106",
        0x167 => "TU117",
        0x168 => "TU116",
        0x170 => "GA100",
        0x172 => "GA102",
        0x173 => "GA103",
        0x174 => "GA104",
        0x176 => "GA106",
        0x177 => "GA107",
        0x192 => "AD102",
        0x194 => "AD104",
        0x196 => "AD106",
        0x197 => "AD107",
        _ => "",
    }
}

/// The architecture a chipset id belongs to.
///
/// Coarser than `chip_name` and therefore right more often: the high bits
/// move once per generation, so an unrecognised die still reports the family
/// it came from.
pub fn family(chipset: u32) -> &'static str {
    match chipset & 0x1f0 {
        0x160 => "Turing",
        0x170 => "Ampere",
        0x190 => "Ada",
        _ => "",
    }
}

/// Record what was found, so the answer survives the scrollback.
pub fn record(gpu: Option<&Gpu>, chip: Option<&str>) {
    match gpu {
        Some(g) => {
            crate::sysbox::write_text(PRESENT, "yes");
            crate::sysbox::write_text(ID, &g.id());
            crate::sysbox::write_text(CHIP, chip.unwrap_or("unread"));
        }
        None => {
            crate::sysbox::write_text(PRESENT, "no");
        }
    }
}

/// The decoder, against values built from the documented field layout.
///
/// Deliberately hardware-free. There is no NVIDIA GPU under QEMU and there is
/// exactly one machine this kernel has ever run on, so a suite that needed
/// the device would be a suite that never ran. What can be asserted anywhere
/// is that the decoder does not invent a chip, which is the failure that
/// would actually mislead: a wrong die name looks exactly like a right one.
pub fn selftest() -> bool {
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
            crate::kprintln!("    FAIL {}", what);
        }
    };

    // Field layout: chipset in 28:20, revision in the low byte.
    claim("GA107 decodes", decode_boot0(0x1770_00a1) == Some((0x177, 0xa1)));
    claim("TU117 decodes", decode_boot0(0x1670_00a1) == Some((0x167, 0xa1)));
    claim("names GA107", chip_name(0x177) == "GA107");
    claim("names TU117", chip_name(0x167) == "TU117");
    claim("GA107 is Ampere", family(0x177) == "Ampere");
    claim("TU117 is Turing", family(0x167) == "Turing");

    // An unknown die must report its family and decline to name itself,
    // rather than falling through to whatever the last arm happened to be.
    claim("unknown die is unnamed", chip_name(0x17f).is_empty());
    claim("unknown die keeps its family", family(0x17f) == "Ampere");
    claim("unknown family is empty", family(0x010).is_empty());

    // The three ways a read can be meaningless. Any of them decoding to a
    // chipset would produce a confident report about a device that is not
    // answering, which is the worst outcome this module has.
    claim("all ones is not a chip", decode_boot0(0xFFFF_FFFF).is_none());
    claim("zero is not a chip", decode_boot0(0).is_none());
    claim("pre-Fermi layout refused", decode_boot0(0x0000_00a1).is_none());

    ok
}

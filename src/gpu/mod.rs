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

pub mod virtio;
pub mod virtgpu;

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


// ---------------------------------------------------------------------------
// Waking it
// ---------------------------------------------------------------------------
//
// A muxless laptop parks its discrete GPU in D3cold, where it vanishes from
// config space entirely. Getting it back is ACPI's own mechanism and not a
// register poke: the device's `_PR0` names the power resources it needs in D0,
// and each of those has an `_ON` method the firmware wrote for this board.
//
// **None of it is written down here.** `\_SB.PC00.PEG1.PEGP` is this laptop's
// path and not the next one's, so the device is found by matching the PCI
// address the bus reports against `Interp::pci_location` -- the same walk that
// resolves a `PCI_Config` region, run the other way round.

/// What waking it would involve, read out of the firmware.
pub struct Plan {
    /// Namespace path of the ACPI device for the GPU.
    pub device: String,
    /// Where it is, or where it will be when it answers.
    pub at: (u8, u8, u8),
    /// The power resources `_PR0` names, in the order it names them. ACPI
    /// requires that order to be honoured: they are listed in the sequence the
    /// firmware wants them turned on.
    pub resources: Vec<String>,
    /// Which of those have an `_ON` this can call.
    pub with_on: usize,
    /// The device's own `_PS0`, which ACPI says to call after the resources.
    pub has_ps0: bool,
}

/// One direct child by name, which is not `resolve`: an ancestor's `_PR0` is a
/// different device's power and calling it would turn on something else.
fn kid(ns: &crate::acpi::aml::Namespace, n: usize, seg: [u8; 4]) -> Option<usize> {
    ns.node(n).children.iter().copied().find(|&c| ns.node(c).name == seg)
}

/// The namespace node describing the device at this PCI address.
///
/// Every `Device` carrying an `_ADR` is asked where it lives and the answers
/// are compared. Linear, and cheap enough: the candidates are the few hundred
/// nodes with an address, not the six thousand in the table.
fn node_at(ns: &crate::acpi::aml::Namespace, at: (u8, u8, u8)) -> Option<usize> {
    let mut it = crate::acpi::eval::Interp::new(ns);
    for i in 0..ns.len() {
        if !matches!(ns.node(i).kind, crate::acpi::aml::Kind::Device) {
            continue;
        }
        if kid(ns, i, *b"_ADR").is_none() {
            continue;
        }
        if it.pci_location(i) == Ok(at) {
            return Some(i);
        }
    }
    None
}

/// Where to look: the NVIDIA part if it is answering, otherwise whatever sits
/// behind a bridge forwarding an empty bus.
///
/// The second case is the one that matters, and it is why this cannot simply
/// use `find`: a GPU in D3cold is not on the bus to be found. The bridge is,
/// and the device is at function zero of the bus it forwards -- which is an
/// address even when nothing answers at it.
pub fn target(ecam: u64) -> Option<(u8, u8, u8)> {
    if let Some(g) = find(ecam) {
        if g.is_nvidia() {
            return Some((g.dev.bus, g.dev.dev, g.dev.func));
        }
    }
    empty_bridges(ecam).first().map(|(_, secondary)| (*secondary, 0, 0))
}

/// Read the plan out of the firmware without running any of it.
pub fn plan(a: &crate::acpi::Acpi, at: (u8, u8, u8)) -> Option<Plan> {
    crate::acpi::with_namespace(a, |ns| {
        let node = node_at(ns, at)?;
        let mut it = crate::acpi::eval::Interp::new(ns);
        let mut resources = Vec::new();
        let mut with_on = 0;
        if let Some(pr0) = kid(ns, node, *b"_PR0") {
            if let Ok(crate::acpi::eval::Value::Pkg(items)) = it.eval_node(pr0, &[]) {
                for e in items {
                    // A `_PR0` element is a reference to a PowerResource. An
                    // element that is anything else is firmware this does not
                    // understand, and is listed rather than skipped silently.
                    match e {
                        crate::acpi::eval::Value::Node(r) => {
                            if kid(ns, r, *b"_ON_").is_some() {
                                with_on += 1;
                            }
                            resources.push(ns.path(r));
                        }
                        other => resources.push(alloc::format!("<{}>", other.type_name())),
                    }
                }
            }
        }
        Some(Plan {
            device: ns.path(node),
            at,
            resources,
            with_on,
            has_ps0: kid(ns, node, *b"_PS0").is_some(),
        })
    })
    .flatten()
}

/// What one step of the wake did.
pub struct Step {
    pub what: String,
    pub ok: bool,
    pub why: String,
}

/// Call `_ON` on each power resource `_PR0` names, then the device's `_PS0`.
///
/// **In the order `_PR0` gave them**, which ACPI requires and which is not
/// cosmetic: a board that powers a rail before the reset that depends on it is
/// a board whose device comes up wrong.
///
/// This runs the vendor's own AML with region writes enabled, which is why the
/// caller has to have unlocked them. What it writes is whatever the firmware
/// writes to turn this device on -- the same sequence every other operating
/// system on this laptop performs -- and it is still somebody else's code
/// touching real registers.
pub fn wake(a: &crate::acpi::Acpi, at: (u8, u8, u8)) -> Vec<Step> {
    let mut out = Vec::new();
    let done = crate::acpi::with_namespace(a, |ns| {
        let Some(node) = node_at(ns, at) else {
            return Vec::new();
        };
        let mut steps: Vec<Step> = Vec::new();
        let mut it = crate::acpi::eval::Interp::new(ns);

        let mut list: Vec<usize> = Vec::new();
        if let Some(pr0) = kid(ns, node, *b"_PR0") {
            if let Ok(crate::acpi::eval::Value::Pkg(items)) = it.eval_node(pr0, &[]) {
                for e in items {
                    if let crate::acpi::eval::Value::Node(r) = e {
                        list.push(r);
                    }
                }
            }
        }

        for r in list {
            let path = ns.path(r);
            match kid(ns, r, *b"_ON_") {
                None => steps.push(Step {
                    what: alloc::format!("{}._ON", path),
                    ok: false,
                    why: String::from("the power resource has no _ON"),
                }),
                Some(on) => steps.push(call(ns, on, &alloc::format!("{}._ON", path))),
            }
        }

        // `_PS0` after the resources, which is the order ACPI states: the
        // rails come up, then the device is told it is in D0.
        if let Some(ps0) = kid(ns, node, *b"_PS0") {
            let path = ns.path(node);
            steps.push(call(ns, ps0, &alloc::format!("{}._PS0", path)));
        }
        steps
    });
    if let Some(v) = done {
        out = v;
    }
    out
}

/// Run one firmware method under a landing pad.
///
/// **This is somebody else's code touching real registers**, and every
/// exception vector in this kernel but `#BP` diverges. A `_ON` that faults
/// without a pad is a machine that stops with a register dump, which tells an
/// operator far less than "that method faulted and here is which".
///
/// A fresh interpreter each time, so one method running away cannot spend the
/// step budget the next one needs.
///
/// What this cannot undo is a sequence that faulted halfway: the rails it had
/// already brought up stay up. Reporting and stopping is still better than
/// halting, because the operator can then read `_STA` and decide.
fn call(ns: &crate::acpi::aml::Namespace, node: usize, what: &str) -> Step {
    use crate::cpu::recover::{self, Caught};
    let mut m = crate::acpi::eval::Interp::new(ns);
    let mut got: Option<Result<(), String>> = None;
    let was = recover::in_selftest();
    recover::selftest_window(true);
    let caught = recover::guarded(|| {
        got = Some(match m.eval_node(node, &[]) {
            Ok(_) => Ok(()),
            Err(e) => Err(crate::acpi::fault_text(&e)),
        });
    });
    recover::selftest_window(was);

    match (caught, got) {
        (Caught::Faulted(why), _) => Step {
            what: String::from(what),
            ok: false,
            why: alloc::format!("the method faulted: {}", why),
        },
        (_, Some(Ok(()))) => Step {
            what: String::from(what),
            ok: true,
            why: alloc::format!("{} step(s)", m.steps()),
        },
        (_, Some(Err(why))) => Step { what: String::from(what), ok: false, why },
        (_, None) => Step {
            what: String::from(what),
            ok: false,
            why: String::from("it did not run"),
        },
    }
}

/// Is the device answering config space now?
///
/// The only answer that settles whether any of it worked. A vendor id that is
/// neither all-ones nor zero means something is there and decoding.
pub fn answers(ecam: u64, at: (u8, u8, u8)) -> Option<u32> {
    let d = Device {
        bus: at.0,
        dev: at.1,
        func: at.2,
        vendor: 0,
        device: 0,
        class: 0,
        subclass: 0,
        prog_if: 0,
        header_type: 0,
    };
    let v = pci::cfg_read32(ecam, &d, 0x00);
    if v == 0xFFFF_FFFF || v == 0 {
        return None;
    }
    Some(v)
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

    for (what, good) in virtio::selftest() {
        claim(what, good);
    }

    ok
}

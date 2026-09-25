//! Does this machine have an Intel Wi-Fi 6 radio, and will it answer?
//!
//! **This is the first step of the only wireless path left**, and it is
//! deliberately the same step `src/gpu/mod.rs` takes: find the device, decide
//! whether it is absent or merely asleep, and read one register that is valid
//! from reset. Nothing here drives anything.
//!
//! The RTL8188EU dongle was the cheap route and its hardware has failed, so the
//! AX201 in this laptop is the radio. `dev::registry` already names it -- Intel
//! Wi-Fi 6 AX201, `8086:51f0`, at 00:14.3 -- and `CLAUDE.md` records the
//! correction that matters for attempting it at all: **CNVi is not the
//! obstacle.** CNVio is the undocumented link between the chipset and the radio
//! module, and the host never speaks it. From here the part is an ordinary PCIe
//! function with BARs and MSI-X, driven the way `iwlwifi` and OpenBSD's `iwx`
//! drive a discrete card.
//!
//! **The port should come from OpenBSD, not Linux, and the reason is the
//! licence.** `iwx(4)` is ISC, where `iwlwifi` is GPL-2.0 -- and this tree has
//! just finished removing its only GPL-2.0 file. What OpenBSD's driver would
//! contribute is the lower half only: reset and power-up, the firmware TLV
//! container, the context-info structure that bootstraps it, then the host
//! command protocol. Everything above `dev::radio::Radio` already exists here
//! and is asserted at boot against a loopback radio and a fake access point --
//! `net::softmac`, `net::mlme`, `net::ccmp`, `crypto::ccm`, `net::wpa2`. That is
//! what `net80211` would otherwise have to be ported with it.
//!
//! **None of this can be exercised under emulation.** QEMU models no wireless
//! part at all, so every refusal below is named in advance rather than
//! recovered from: a bad MMIO read at ring 0 is a halted machine and a register
//! dump. The decode is a pure function for the same reason -- `rev_of` can be
//! walked by a suite on any machine, where reading a real `CSR_HW_REV` cannot.

use crate::dev::pci::{self, Device};
use alloc::string::String;
use alloc::vec::Vec;

/// Intel, and the CNVi/PCIe wireless functions this kernel knows by id.
///
/// The list is `dev::registry`'s `INTEL_CNVI_IDS`, not a second copy: a private
/// id list per driver is the arrangement the registry exists to replace.
pub const VENDOR_INTEL: u16 = 0x8086;

/// Hardware revision, valid from reset and before any firmware runs.
///
/// The one register worth reading first, for `NV_PMC_BOOT_0`'s reason: a
/// plausible answer proves the function is alive, its BAR is decoded, and MMIO
/// reaches it -- and an implausible one says the fault is below the driver
/// rather than in anything it has not done yet.
const CSR_HW_REV: u64 = 0x028;

/// Which radio module is attached, as distinct from which MAC.
///
/// Intel pairs one MAC with several radios and the combination decides which
/// firmware file is correct, so a driver that read only `CSR_HW_REV` would load
/// the wrong one and get silence. Read here so the pair is reported together
/// from the start.
const CSR_HW_RF_ID: u64 = 0x09c;

/// A register aperture that reads as all ones.
///
/// Two quite different faults look like this and neither is a revision: a
/// function parked in D3cold, and a BAR whose memory-space decoder is off. Both
/// must be told apart from a real answer, because `0xFFFFFFFF` decodes to a
/// plausible-looking and entirely fictional part -- the failure `gpu::boot0`
/// records.
const ALL_ONES: u32 = 0xFFFF_FFFF;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mac {
    /// Family 22000, the generation the AX200/AX201 belong to.
    Qu,
    /// A stepping of the same family, spun for a different process.
    Quz,
    /// Read, decoded, and not one this kernel has a name for. Reported as the
    /// raw type rather than guessed at, because a wrong guess picks the wrong
    /// firmware and the symptom is silence.
    Unknown(u16),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rev {
    pub mac: Mac,
    /// Silicon step, which selects between firmware images within one family.
    pub step: u8,
    /// Sub-step. Intel's own driver treats the pair as one value on this
    /// family, which is why they are decoded together and reported apart.
    pub dash: u8,
    pub raw: u32,
}

/// Decode `CSR_HW_REV`.
///
/// Pure, and that is the point: no emulator models this part, so the only thing
/// a suite can check on any machine is the arithmetic. `iwlwifi` spells the
/// fields `CSR_HW_REV_TYPE` (bits 4..16) and `CSR_HW_REV_STEP_DASH` (bits 0..4),
/// and the type values are the ones its `iwl_cfg_mac_type` table names.
pub fn rev_of(raw: u32) -> Rev {
    let ty = ((raw & 0x000F_FFF0) >> 4) as u16;
    Rev {
        mac: match ty {
            0x33 => Mac::Qu,
            0x35 => Mac::Quz,
            other => Mac::Unknown(other),
        },
        // Bits 1..2 and 0..1. Taken apart rather than reported as one nibble
        // because the firmware name is chosen from the step alone.
        step: ((raw >> 2) & 0x3) as u8,
        dash: (raw & 0x3) as u8,
        raw,
    }
}

impl Mac {
    pub fn name(&self) -> String {
        match self {
            Mac::Qu => String::from("Qu"),
            Mac::Quz => String::from("QuZ"),
            Mac::Unknown(t) => alloc::format!("unknown type {:#05x}", t),
        }
    }
    /// Whether this kernel could name a firmware image for the part.
    ///
    /// Answered separately from `name` because "recognised" and "drivable" are
    /// different facts, which is the distinction `dev::registry`'s three support
    /// levels exist to keep.
    pub fn known(&self) -> bool {
        !matches!(self, Mac::Unknown(_))
    }
}

#[derive(Clone, Copy)]
pub struct Radio {
    pub dev: Device,
    /// The register aperture. `iwlwifi` maps the first 4 KiB to reach the CSRs.
    pub bar0: Option<u64>,
}

/// Why a probe could not answer, in the caller's words rather than a code.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Refusal {
    /// No BAR was assigned. Firmware that leaves one unassigned has decided
    /// this function is not in use, and mapping zero is the one fault the
    /// identity map leaves unmapped on purpose.
    NoAperture,
    /// The aperture could not be mapped. Out of page tables, or an address the
    /// identity map does not reach.
    NotMapped,
    /// Every bit set. Parked in D3cold, or memory-space decoding is off, and
    /// neither is a revision.
    Asleep,
    /// All bits clear, which no live function reports for this register.
    Silent,
}

impl Refusal {
    pub fn why(&self) -> &'static str {
        match self {
            Refusal::NoAperture => "no register aperture assigned; firmware left this function unused",
            Refusal::NotMapped => "the register aperture could not be mapped",
            Refusal::Asleep => "reads as all ones: parked in D3cold, or memory-space decoding is off",
            Refusal::Silent => "reads as all zeroes, which no live function reports",
        }
    }
}

impl Radio {
    pub fn id(&self) -> String {
        alloc::format!("{:04x}:{:04x}", self.dev.vendor, self.dev.device)
    }

    /// Read `CSR_HW_REV` and `CSR_HW_RF_ID`.
    ///
    /// Memory-space decoding is enabled first, because a BAR whose decoder is
    /// off reads back as all ones and that is indistinguishable from a sleeping
    /// part. Bus mastering is left alone deliberately: a probe has no business
    /// granting DMA to a device whose firmware has never run, which is the rule
    /// `gpu::boot0` states and the reason it does not call
    /// `pci::enable_bus_master`.
    pub fn hw_rev(&self, ecam: u64) -> Result<(Rev, u32), Refusal> {
        let bar0 = self.bar0.filter(|&b| b != 0).ok_or(Refusal::NoAperture)?;
        enable_memory_space(ecam, &self.dev);
        if !crate::mem::paging::map_range(bar0, 0x1000, true) {
            return Err(Refusal::NotMapped);
        }
        // Safety: the aperture is mapped uncached immediately above, and both
        // offsets are inside the first 4 KiB of it. Volatile because a register
        // is not memory and the compiler may not reorder or elide either read.
        let (rev, rf) = unsafe {
            (
                core::ptr::read_volatile((bar0 + CSR_HW_REV) as *const u32),
                core::ptr::read_volatile((bar0 + CSR_HW_RF_ID) as *const u32),
            )
        };
        if rev == ALL_ONES {
            return Err(Refusal::Asleep);
        }
        if rev == 0 {
            return Err(Refusal::Silent);
        }
        Ok((rev_of(rev), rf))
    }
}

/// Enable memory-space decoding, leaving bus-master untouched.
///
/// The same function `gpu` needs and for the same reason; `pci::enable_bus_master`
/// sets both bits because its callers wanted both, and a probe wants only this
/// one.
fn enable_memory_space(ecam: u64, d: &Device) {
    const CMD_MEMORY_SPACE: u32 = 1 << 1;
    let cmd = pci::cfg_read32(ecam, d, 0x04);
    if cmd & CMD_MEMORY_SPACE == 0 {
        pci::cfg_write32(ecam, d, 0x04, cmd | CMD_MEMORY_SPACE);
    }
}

/// Every Intel wireless function on the bus, with its aperture.
///
/// Asks `dev::registry` which ids are wireless rather than carrying a list, so
/// adding a part is a row there and not a second table here.
pub fn find(ecam: u64) -> Vec<Radio> {
    use crate::dev::registry::{lookup, Ident, Role};
    let mut out: Vec<Radio> = Vec::new();
    // 256 buses, the sweep `pci::scan` takes. A callback rather than an iterator
    // because that is the shape `pci` offers, and building a `Vec` of every
    // function just to filter it would be the larger allocation.
    pci::scan(ecam, 256, |dev| {
        if dev.vendor != VENDOR_INTEL {
            return;
        }
        if lookup(&Ident::of_pci(&dev)).map(|e| e.role) != Some(Role::Wireless) {
            return;
        }
        out.push(Radio { dev, bar0: pci::bar(ecam, &dev, 0) });
    });
    out
}

/// What `diag iwx` asserts, with no radio present.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let mut claim = |what: &'static str, ok: bool| out.push((what, ok));

    // The two families this kernel can name. Real values from `iwlwifi`'s
    // `iwl_cfg_mac_type`, shifted into the field the register puts them in.
    claim("a Qu revision decodes as Qu", rev_of(0x33 << 4).mac == Mac::Qu);
    claim("a QuZ revision decodes as QuZ", rev_of(0x35 << 4).mac == Mac::Quz);

    // **An unrecognised type is reported, not guessed at.** The firmware image
    // is chosen from this, and a wrong choice loads the wrong file and gets
    // silence -- so a part this kernel cannot name must say so rather than fall
    // back to the nearest family.
    claim(
        "an unknown type keeps its raw value instead of defaulting",
        rev_of(0x7f << 4).mac == Mac::Unknown(0x7f),
    );
    claim("and reports itself as not drivable", !rev_of(0x7f << 4).mac.known());
    claim("where a known one does", rev_of(0x33 << 4).mac.known());

    // Step and dash share a nibble and are taken apart, because the firmware
    // name is chosen from the step alone.
    let r = rev_of((0x33 << 4) | 0b1011);
    claim("the step is bits 2..4", r.step == 0b10);
    claim("and the dash is bits 0..2", r.dash == 0b11);
    claim("and the raw word is kept for a bug report", r.raw == (0x33 << 4) | 0b1011);

    // The type field stops at bit 16, so a revision with high bits set decodes
    // to the same part. Checked because those bits are not reserved forever and
    // a decoder that swallowed them would rename the part on the next stepping.
    claim(
        "bits above the type field do not change the part",
        rev_of(0x33 << 4).mac == rev_of((0xABC0_0000) | (0x33 << 4)).mac,
    );

    // The two readings that are not revisions, and must never be taken for one.
    claim(
        "all ones is a refusal and names both causes",
        Refusal::Asleep.why().contains("D3cold") && Refusal::Asleep.why().contains("decoding"),
    );
    claim(
        "all zeroes is a separate refusal",
        Refusal::Silent != Refusal::Asleep,
    );
    claim(
        "and an unassigned aperture is a third",
        Refusal::NoAperture != Refusal::NotMapped,
    );
    out
}

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

pub mod alive;
pub mod cmd;
pub mod config;
pub mod ctxt;
pub mod gen3;
pub mod init;
pub mod nvm;
pub mod power;
pub mod fw;

use crate::dev::pci::{self, Device};
use alloc::string::String;
use alloc::vec::Vec;

/// Intel, and the CNVi/PCIe wireless functions this kernel knows by id.
///
/// The ids themselves are `dev::registry`'s rows that name `iwx`, not a second
/// copy: a private id list per driver is the arrangement the registry exists to
/// replace.
pub const VENDOR_INTEL: u16 = 0x8086;

/// Hardware revision, valid from reset and before any firmware runs.
///
/// The one register worth reading first, for `NV_PMC_BOOT_0`'s reason: a
/// plausible answer proves the function is alive, its BAR is decoded, and MMIO
/// reaches it -- and an implausible one says the fault is below the driver
/// rather than in anything it has not done yet.
/// How much of the register aperture is mapped.
///
/// **Eight kilobytes and not four, because the receive doorbell is at 0x1c80.**
/// The probe needed one page: `CSR_HW_REV` and everything the power-up sequence
/// touches live below 0x1000. Servicing the receive ring does not --
/// `RFH_Q0_FRBDCB_WIDX_TRG` is in the second page, so a driver that kept the
/// one-page mapping would write the index into whatever the identity map has
/// there, which is somebody else's memory and reports nothing.
///
/// Still deliberately short of the whole BAR: MSI-X begins at 0x2000 and nothing
/// here configures it, so mapping up to it and no further is the range that is
/// actually used.
pub const APERTURE: u64 = 0x2000;

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

/// Which generation of silicon, which decides everything about the bring-up.
///
/// **This distinction was missing and it is the one that matters most here.** The
/// two families take their firmware by completely different routes: family 22000
/// builds one 1,792-byte descriptor and writes its address to `CSR_CTXT_INFO_BA`,
/// while AX210 builds a larger one plus a scratch area plus an image loader and
/// goes through `CSR_CTXT_INFO_ADDR`. Reaching for the wrong one hands the part a
/// structure it reads at different offsets.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family {
    /// Qu, QuZ and QnJ. The AX200, and the AX201 as fitted to earlier platforms.
    F22000,
    /// So, SoF and SnJ. **What is in the GF63**, despite the part being sold as
    /// an AX201: the marketing name follows the radio module and the family
    /// follows the controller, and on Alder Lake the controller is Snow Owl.
    Ax210,
    /// Bz and later. Differs again at the reset and the clock handshake, so it is
    /// named in order to be refused rather than driven by this table.
    ///
    /// **Ma is not one, and this said it was.** Both references file Ma under the
    /// AX210 family -- OpenBSD's `iwx_attach` sets `IWX_DEVICE_FAMILY_AX210` for
    /// `7e40`, and it boots out of the same gen3 context info -- so the refusal
    /// was a guess from the type number being above Snow Owl's.
    Bz,
}

impl Family {
    pub fn name(&self) -> &'static str {
        match self {
            Family::F22000 => "22000",
            Family::Ax210 => "AX210",
            Family::Bz => "Bz",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mac {
    /// Family 22000, the generation the AX200 belongs to.
    Qu,
    /// A stepping of the same family, spun for a different process.
    Quz,
    Qnj,
    /// Snow Owl, and what `8086:51f0` is.
    So,
    Snj,
    /// Snow Owl F, the Alder Lake spin.
    Sof,
    Ma,
    Bz,
    Gl,
    BzW,
    /// Read, decoded, and not one this kernel has a name for. Reported as the
    /// raw type rather than guessed at, because a wrong guess picks the wrong
    /// firmware and the symptom is silence.
    Unknown(u16),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rev {
    pub mac: Mac,
    /// Silicon step, which selects between firmware images within one family:
    /// A, B, C or Z as 0, 1, 2 and 0xf.
    pub step: u8,
    pub raw: u32,
}

/// Decode `CSR_HW_REV`.
///
/// Pure, and that is the point: no emulator models this part, so the only thing
/// a suite can check on any machine is the arithmetic. `iwlwifi` spells the
/// fields `CSR_HW_REV_TYPE` (bits 4..16) and `CSR_HW_REV_STEP_DASH` (bits 0..4),
/// and the type values are the ones its `cfg_mac_type` table names.
///
/// **There is no dash, and reading one was the bug this had.** The field used to
/// be a two-bit step at bits 2..4 with a two-bit dash below it, and upstream's
/// own comment records that changing in the 8000 family: "the revision step also
/// includes bit 0-1 (no more 'dash' value)". Every family this could drive is
/// later than that, so the step is bits 0..2 and the low pair is part of it --
/// which is why the driver re-packs `hw_rev` into the *old* shape before
/// matching, and why taking bits 2..4 as the step answers about a field that no
/// longer exists. It reported step 0 with dash 3 where the part is step 3, and
/// the step is what selects the firmware image within a family.
///
/// Found by reading the source these offsets came from rather than by running
/// anything, because no emulator has this part.
pub fn rev_of(raw: u32) -> Rev {
    let ty = ((raw & 0x000F_FFF0) >> 4) as u16;
    Rev {
        mac: match ty {
            0x33 => Mac::Qu,
            0x35 => Mac::Quz,
            0x36 => Mac::Qnj,
            0x37 => Mac::So,
            0x42 => Mac::Snj,
            0x43 => Mac::Sof,
            0x44 => Mac::Ma,
            0x46 => Mac::Bz,
            0x47 => Mac::Gl,
            0x4b => Mac::BzW,
            other => Mac::Unknown(other),
        },
        step: (raw & 0x3) as u8,
        raw,
    }
}

impl Mac {
    pub fn name(&self) -> String {
        match self {
            Mac::Qu => String::from("Qu"),
            Mac::Quz => String::from("QuZ"),
            Mac::Qnj => String::from("QnJ"),
            Mac::So => String::from("So"),
            Mac::Snj => String::from("SnJ"),
            Mac::Sof => String::from("SoF"),
            Mac::Ma => String::from("Ma"),
            Mac::Bz => String::from("Bz"),
            Mac::Gl => String::from("Gl"),
            Mac::BzW => String::from("Bz-W"),
            Mac::Unknown(t) => alloc::format!("unknown type {:#05x}", t),
        }
    }

    /// Which generation this controller belongs to.
    ///
    /// A function of the type register rather than of the PCI id, deliberately:
    /// `8086:51f0` covers several modules and the controller behind all of them
    /// is the same silicon, so asking the register asks the part instead of
    /// asking a table about the platform.
    pub fn family(&self) -> Option<Family> {
        Some(match self {
            Mac::Qu | Mac::Quz | Mac::Qnj => Family::F22000,
            Mac::So | Mac::Snj | Mac::Sof | Mac::Ma => Family::Ax210,
            Mac::Bz | Mac::Gl | Mac::BzW => Family::Bz,
            Mac::Unknown(_) => return None,
        })
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

/// What `CSR_HW_RF_ID` says, decoded.
///
/// **It was read and handed back raw.** The word is what tells an AX201 from an
/// AX211 on one PCI id -- the controller is the same Snow Owl silicon and the
/// *radio* is what differs -- so a probe that prints it in hex has read the
/// answer and not said it. It also decides the firmware's own filename, which is
/// the thing a bare-metal trip needs to know before it starts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rf {
    /// Jefferson, two-antenna and one.
    Jf2,
    Jf1,
    /// Harrier. **Hr2 with Snow Owl is the part sold as an AX201**, and Hr1 the
    /// AX101.
    Hr2,
    Hr1,
    /// Gale Force, which is the AX211.
    Gf,
    Mr,
    Ms,
    Fm,
    Unknown(u16),
}

impl Rf {
    pub fn name(&self) -> String {
        match self {
            Rf::Jf2 => String::from("JF2"),
            Rf::Jf1 => String::from("JF1"),
            Rf::Hr2 => String::from("HR2"),
            Rf::Hr1 => String::from("HR1"),
            Rf::Gf => String::from("GF"),
            Rf::Mr => String::from("MR"),
            Rf::Ms => String::from("MS"),
            Rf::Fm => String::from("FM"),
            Rf::Unknown(t) => alloc::format!("unknown RF {:#05x}", t),
        }
    }
}

/// The radio, as one word says it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RfId {
    pub rf: Rf,
    pub step: u8,
    pub dash: u8,
    /// Whether this is a two-die part with a second radio.
    pub cdb: bool,
    /// Whether the radio is on a jacket board rather than in the package.
    pub jacket: bool,
    pub raw: u32,
}

/// Decode `CSR_HW_RF_ID`.
///
/// Four fields and two flags, and the type is bits 12..24 rather than the low
/// ones -- the flavour and the step sit underneath it. Pure, so it is asserted on
/// any machine even though the register cannot be read on one.
pub fn rf_of(raw: u32) -> RfId {
    let ty = ((raw & 0x0FFF_F000) >> 12) as u16;
    RfId {
        rf: match ty {
            0x105 => Rf::Jf2,
            0x108 => Rf::Jf1,
            0x10a => Rf::Hr2,
            0x10c => Rf::Hr1,
            0x10d => Rf::Gf,
            0x110 => Rf::Mr,
            0x111 => Rf::Ms,
            0x112 => Rf::Fm,
            other => Rf::Unknown(other),
        },
        step: ((raw & 0x0000_0F00) >> 8) as u8,
        dash: ((raw & 0x0000_00F0) >> 4) as u8,
        cdb: raw & 0x1000_0000 != 0,
        jacket: raw & 0x2000_0000 != 0,
        raw,
    }
}

/// The name this combination is sold under, where there is one.
///
/// Answered from the pair rather than from either alone, because that is how the
/// pair works: Snow Owl with Harrier is an AX201 and Snow Owl with Gale Force is
/// an AX211, on the same PCI id. `None` rather than a guess, since a marketing
/// name is the one field here nobody can derive from a register they have not
/// seen.
pub fn product_name(mac: Mac, rf: Rf) -> Option<&'static str> {
    Some(match (mac, rf) {
        (Mac::So, Rf::Hr2) | (Mac::Sof, Rf::Hr2) => "AX201",
        (Mac::So, Rf::Hr1) | (Mac::Sof, Rf::Hr1) => "AX101",
        (Mac::So, Rf::Gf) | (Mac::Sof, Rf::Gf) => "AX211",
        (Mac::Qu, Rf::Hr2) | (Mac::Quz, Rf::Hr2) => "AX201",
        (Mac::Qu, Rf::Hr1) | (Mac::Quz, Rf::Hr1) => "AX101",
        (Mac::Qu, Rf::Jf2) | (Mac::Quz, Rf::Jf2) => "9560",
        _ => return None,
    })
}

/// The highest firmware API this driver understands.
///
/// An image's API is the version of its command layouts, and a newer one moves
/// fields this driver writes at fixed offsets -- so the newest file on a disk is
/// not the right one, the newest at or below this is. `tools/wifi_fw.py` stages
/// by the same ceiling and its selftest reads this line out of the source, so
/// the two cannot drift apart silently. 89 because that is the image every
/// layout here was measured against.
pub const MAX_API: u32 = 89;

/// The oldest API worth looking for. Below this the container predates the
/// TLVs `fw::parse` relies on, so a file that old is not a candidate.
pub const MIN_API: u32 = 50;

/// The firmware *base* a part wants: everything in `iwlwifi-<base>-<api>.ucode`
/// but the API.
///
/// **Mostly a function of the registers, and the PCI id only where they are
/// ambiguous.** The controller type and step give the first half and the radio
/// type and step the second, which is how `so-a0-hr-b0` falls out of the GF63's
/// `0x370` and `0x10a100`. Two cases need the id and both are upstream's:
/// the AX200 (`2723`) carries its radio in the controller name, `cc-a0`; and
/// type `0x42` is both SnJ and Typhoon Peak, which `iwx_attach` separates by
/// product -- `2725` is the discrete AX210 and wants `ty-a0-gf-a0`.
///
/// `None` rather than a guess for anything else. A wrong base is the wrong
/// firmware, and the symptom of the wrong firmware is a part that never says it
/// is alive -- which is the most expensive thing to debug on hardware with no
/// emulator.
///
/// Not covered, deliberately: the Killer 1690 parts, which want a `gf4` image
/// and are told apart only by PCI subsystem id. They get the `gf` base, which is
/// the documented fallback upstream takes when the subsystem is not listed.
pub fn firmware_base(device: u16, rev: Rev, rf: RfId) -> Option<String> {
    let step = |n: u8| -> Option<char> {
        match n {
            0 => Some('a'),
            1 => Some('b'),
            2 => Some('c'),
            _ => None,
        }
    };
    if device == 0x2723 {
        return Some(String::from("cc-a0"));
    }
    let radio = match rf.rf {
        Rf::Hr1 | Rf::Hr2 => alloc::format!("hr-{}0", step(rf.step)?),
        Rf::Gf => alloc::format!("gf-{}0", step(rf.step)?),
        Rf::Jf1 | Rf::Jf2 => alloc::format!("jf-{}0", step(rf.step)?),
        _ => return None,
    };
    let mac = match rev.mac {
        Mac::Snj if device == 0x2725 => return Some(alloc::format!("ty-a0-{}", radio)),
        Mac::So | Mac::Sof | Mac::Snj => String::from("so-a0"),
        Mac::Ma => String::from("ma-b0"),
        Mac::Qu => alloc::format!("Qu-{}0", step(rev.step)?),
        Mac::Quz => String::from("QuZ-a0"),
        _ => return None,
    };
    Some(alloc::format!("{}-{}", mac, radio))
}

/// The image for a base, newest API first, wherever `dev::firmware` finds it.
pub fn firmware_for(base: &str) -> Option<(String, crate::dev::firmware::Image)> {
    (MIN_API..=MAX_API).rev().find_map(|api| {
        let name = alloc::format!("iwlwifi-{}-{}.ucode", base, api);
        crate::dev::firmware::get(&name).map(|img| (name, img))
    })
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
    /// The read itself faulted and was caught. The aperture is mapped and the
    /// device does not decode it -- which firmware assigning a BAR does not
    /// promise, and which nothing before the first read can tell.
    Faulted,
}

impl Refusal {
    pub fn why(&self) -> &'static str {
        match self {
            Refusal::NoAperture => "no register aperture assigned; firmware left this function unused",
            Refusal::NotMapped => "the register aperture could not be mapped",
            Refusal::Asleep => "reads as all ones after waking it to D0: parked in D3cold, or memory-space decoding is off",
            Refusal::Silent => "reads as all zeroes, which no live function reports",
            Refusal::Faulted => "the read faulted and was caught: the aperture is mapped and the device does not decode it",
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
        // Woken first: a part the firmware left in D3hot answers config space
        // and not its BARs, and its all-ones read is otherwise reported as
        // `Asleep` with nothing done about it. D3cold is out of reach from here.
        let _ = pci::set_d0(ecam, &self.dev);
        enable_memory_space(ecam, &self.dev);
        if !crate::mem::paging::map_range(bar0, APERTURE, true) {
            return Err(Refusal::NotMapped);
        }
        // **Guarded, because this is the first read of an aperture nothing has
        // validated.** The BAR came from firmware and the mapping succeeded, and
        // neither says the device decodes that range: on a part that is half
        // asleep or mis-described, the read is a machine check, and every vector
        // but `#BP` here is fatal. Unguarded that means a dead machine, a reboot
        // and nothing learnt -- on a trip whose whole purpose is to learn.
        //
        // `recover::guard` is what `mem::paging::checks` uses to fault on
        // purpose, so this is the same landing pad. A caught fault reports as
        // `Silent` rather than a revision: the register did not answer, which is
        // exactly what the caller needs to hear, and the alternative is a
        // transcript that ends mid-line.
        let mut got = (0u32, 0u32);
        let read = crate::cpu::recover::guard(|| {
            // Safety: the aperture is mapped uncached immediately above and both
            // offsets are inside the first 4 KiB. Volatile because a register is
            // not memory and neither read may be reordered or elided.
            got = unsafe {
                (
                    core::ptr::read_volatile((bar0 + CSR_HW_REV) as *const u32),
                    core::ptr::read_volatile((bar0 + CSR_HW_RF_ID) as *const u32),
                )
            };
        });
        if read.is_err() {
            return Err(Refusal::Faulted);
        }
        let (rev, rf) = got;
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

/// Every function on the bus that `dev::registry` hands to this driver.
///
/// Asks for the rows naming `iwx` rather than for Intel wireless in general,
/// which is what this asked until the id lists were rebuilt: an AC 9560 is Intel
/// wireless of the generation before, and collecting it here would send the
/// power-up at a part with a different firmware API.
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
        let row = lookup(&Ident::of_pci(&dev));
        if row.map(|e| e.role) != Some(Role::Wireless) || row.and_then(|e| e.support.driver()) != Some("iwx") {
            return;
        }
        out.push(Radio { dev, bar0: pci::bar(ecam, &dev, 0) });
    });
    out
}

// --- what the checklist reads -----------------------------------------------
//
// **The one thing on the bring-up list that cannot be derived.** Every other row
// asks the kernel a question it can already answer; a radio has to be *read*
// before it can be reported, and reading it maps a BAR and enables memory-space
// decoding. So the read happens when the operator asks for it and the answer is
// kept here, which is why these are the only cached statuses in the checklist.
//
// `Racy` and not a lock: written by the shell task and read by the shell task,
// and a wrong answer here is a stale line on a report rather than anything the
// machine acts on.
static SEEN: crate::sync::Racy<Option<usize>> = crate::sync::Racy::new(None);
static LAST_REV: crate::sync::Racy<Option<Result<Rev, Refusal>>> = crate::sync::Racy::new(None);
static LAST_UP: crate::sync::Racy<Option<Result<(), Fault>>> = crate::sync::Racy::new(None);

/// How many Intel wireless functions the last scan found.
pub fn seen() -> Option<usize> {
    unsafe { *SEEN.get() }
}
pub fn last_rev() -> Option<Result<Rev, Refusal>> {
    unsafe { *LAST_REV.get() }
}
pub fn last_power_up() -> Option<Result<(), Fault>> {
    unsafe { *LAST_UP.get() }
}

pub fn note_seen(n: usize) {
    unsafe { *SEEN.get() = Some(n) };
}
pub fn note_rev(r: Result<Rev, Refusal>) {
    unsafe { *LAST_REV.get() = Some(r) };
}
/// What the last `iwx ctxt` built, as `(sections, bytes)` or why it could not.
///
/// Recorded rather than derived, which is the exception this module already makes
/// for the revision and for the same reason: building it allocates the regions and
/// copies the firmware, so a status row that derived it would do that every time
/// the page was printed.
static LAST_CTXT: crate::sync::Racy<Option<Result<(usize, usize), &'static str>>> =
    crate::sync::Racy::new(None);

pub fn last_ctxt() -> Option<Result<(usize, usize), &'static str>> {
    unsafe { *LAST_CTXT.get() }
}

pub fn note_ctxt(r: Result<(usize, usize), &'static str>) {
    unsafe { *LAST_CTXT.get() = Some(r) };
}

/// What the last `iwx boot` heard, or why it heard nothing.
///
/// A `String` for the failure because the reasons come from four different types
/// and what the operator needs is the sentence, not the variant.
static LAST_ALIVE: crate::sync::Racy<Option<Result<alive::Alive, String>>> =
    crate::sync::Racy::new(None);

pub fn last_alive() -> Option<Result<alive::Alive, String>> {
    unsafe { (*LAST_ALIVE.get()).clone() }
}

/// What the last `iwx boot` read out of the part's NVM.
static LAST_NVM: crate::sync::Racy<Option<Result<nvm::Nvm, String>>> = crate::sync::Racy::new(None);

pub fn last_nvm() -> Option<Result<nvm::Nvm, String>> {
    unsafe { (*LAST_NVM.get()).clone() }
}

pub fn note_nvm(r: Result<nvm::Nvm, String>) {
    unsafe { *LAST_NVM.get() = Some(r) };
}

pub fn note_alive(r: Result<alive::Alive, String>) {
    unsafe { *LAST_ALIVE.get() = Some(r) };
}

pub fn note_power_up(r: Result<(), Fault>) {
    unsafe { *LAST_UP.get() = Some(r) };
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

    // **The step is the low pair and there is no dash**, which is the correction
    // this claim used to encode backwards -- it asserted the step was bits 2..4
    // and a dash below it, which is the pre-8000 layout upstream's own comment
    // says stopped existing. On the GF63 the old decoding reports step 0 where
    // the part is step 3, and the step chooses the firmware image.
    let r = rev_of((0x33 << 4) | 0b1011);
    claim("the step is the low two bits", r.step == 0b11);
    claim("and the raw word is kept for a bug report", r.raw == (0x33 << 4) | 0b1011);
    // Every step value the silicon spells, so a decoder that masked one bit too
    // few would show here rather than as the wrong firmware.
    claim("step A is 0", rev_of(0x33 << 4).step == 0);
    claim("step B is 1", rev_of((0x33 << 4) | 1).step == 1);
    claim("step C is 2", rev_of((0x33 << 4) | 2).step == 2);

    // --- the family, which is what the bring-up hangs off --------------------

    // The finding this whole group exists for: `8086:51f0` in the GF63 is Snow
    // Owl, so it is AX210 family and not the 22000 the power-up table was
    // written for -- and it is sold as an AX201, which is what made the wrong
    // family look right.
    claim("So is AX210 family", rev_of(0x37 << 4).mac.family() == Some(Family::Ax210));
    claim("and so is SoF, the Alder Lake spin in the GF63", rev_of(0x43 << 4).mac.family() == Some(Family::Ax210));
    claim("Qu is family 22000", rev_of(0x33 << 4).mac.family() == Some(Family::F22000));
    claim("QuZ too", rev_of(0x35 << 4).mac.family() == Some(Family::F22000));
    claim("Bz is its own family, named so it can be refused", rev_of(0x46 << 4).mac.family() == Some(Family::Bz));
    claim("Ma is AX210 family, as both references file it", rev_of(0x44 << 4).mac.family() == Some(Family::Ax210));

    // --- which image, from the registers -----------------------------------
    //
    // The GF63's own words, read off the machine: `51f0`, `rev=0x370`,
    // `rfid=0x10a100`. The image it loaded under Linux was so-a0-hr-b0-89.
    let gf63 = firmware_base(0x51f0, rev_of(0x370), rf_of(0x0010_a100));
    claim("the GF63's registers name the image it is known to load", gf63.as_deref() == Some("so-a0-hr-b0"));
    claim(
        "a Gale Force radio on the same controller names a different image",
        firmware_base(0x51f0, rev_of(0x370), rf_of(0x0010_d000)).as_deref() == Some("so-a0-gf-a0"),
    );
    claim(
        "type 0x42 is Typhoon Peak on a discrete 2725 and Snow Owl elsewhere",
        firmware_base(0x2725, rev_of(0x420), rf_of(0x0010_d000)).as_deref() == Some("ty-a0-gf-a0")
            && firmware_base(0x2726, rev_of(0x420), rf_of(0x0010_d000)).as_deref() == Some("so-a0-gf-a0"),
    );
    claim(
        "a Qu takes its step into the name, which is what tells b0 from c0",
        // The register, not upstream's constants: `IWX_CSR_HW_REV_TYPE_QU_B0` is
        // 0x334 *after* `iwx_attach` re-packs the step two bits up, so the word
        // the part actually reads for B0 is 0x331 and for C0 0x332.
        firmware_base(0xa0f0, rev_of(0x331), rf_of(0x0010_a100)).as_deref() == Some("Qu-b0-hr-b0")
            && firmware_base(0xa0f0, rev_of(0x332), rf_of(0x0010_a100)).as_deref() == Some("Qu-c0-hr-b0"),
    );
    claim(
        "the AX200 is named by its id, its radio being part of the controller name",
        firmware_base(0x2723, rev_of(0x340), rf_of(0)).as_deref() == Some("cc-a0"),
    );
    claim(
        "and a radio nobody has named gets no image rather than a neighbour's",
        firmware_base(0x51f0, rev_of(0x370), rf_of(0x0011_0000)).is_none()
            && firmware_base(0x51f0, rev_of(0x460), rf_of(0x0010_a100)).is_none(),
    );
    claim("the API window is not empty", MIN_API <= MAX_API);
    claim(
        "and a type nobody has named has no family rather than a default",
        rev_of(0x99 << 4).mac.family().is_none(),
    );
    // A default here would be the expensive kind of wrong: an unknown controller
    // treated as 22000 gets the 1,792-byte descriptor written for it and fetches
    // its microcode from whatever the offsets happen to name.
    claim(
        "every named type has a family, so the two tables cannot drift",
        [0x33u16, 0x35, 0x36, 0x37, 0x42, 0x43, 0x44, 0x46, 0x47, 0x4b]
            .iter()
            .all(|&t| rev_of((t as u32) << 4).mac.family().is_some()),
    );

    // --- the radio ----------------------------------------------------------

    // Read from the register since the probe was written and never decoded. The
    // type is bits 12..24, so a decoder taking the low bits answers about the
    // step.
    let hr = rf_of(0x10a << 12);
    claim("HR2 decodes from bits 12..24", hr.rf == Rf::Hr2);
    claim("Gale Force too", rf_of(0x10d << 12).rf == Rf::Gf);
    claim(
        "and an unnamed radio is reported rather than guessed at",
        matches!(rf_of(0x999 << 12).rf, Rf::Unknown(0x999)),
    );
    let full = rf_of((0x10a << 12) | (0x2 << 8) | (0x1 << 4) | 0x3000_0000);
    claim("the radio step is bits 8..12", full.step == 2);
    claim("its dash is bits 4..8", full.dash == 1);
    claim("the two-die flag is bit 28", full.cdb);
    claim("and the jacket flag is bit 29", full.jacket);
    claim("neither flag is set on a plain part", !hr.cdb && !hr.jacket);

    // **The pair names the part and neither half does.** This is the claim that
    // says why both registers are read: one PCI id, two products, and the only
    // difference is the radio.
    claim(
        "Snow Owl with Harrier is the AX201",
        product_name(Mac::Sof, Rf::Hr2) == Some("AX201"),
    );
    claim(
        "and Snow Owl with Gale Force is the AX211, on the same controller",
        product_name(Mac::Sof, Rf::Gf) == Some("AX211"),
    );
    claim(
        "so the controller alone does not name it",
        product_name(Mac::Sof, Rf::Hr2) != product_name(Mac::Sof, Rf::Gf),
    );
    claim(
        "a combination nobody has named answers nothing rather than the nearest",
        product_name(Mac::Bz, Rf::Fm).is_none(),
    );

    // --- the real part, from its own registers -------------------------------

    // **Not synthetic.** These are the words the GF63's own Linux printed for
    // `00:14.3` -- `PCI dev 51f0/0074, rev=0x370, rfid=0x10a100` -- so this group
    // checks both decoders against a reading taken off the target rather than
    // against values chosen to make them pass. It is the only evidence available
    // here about a part no emulator models, and it cost nothing but reading a
    // journal.
    /// Exactly what its Linux printed: `rev=0x370`.
    const GF63_HW_REV: u32 = 0x0000_0370;
    let real = rev_of(GF63_HW_REV);
    claim("the GF63's controller decodes as Snow Owl", real.mac == Mac::So);
    claim("which is AX210 family, not the 22000 this was built for", real.mac.family() == Some(Family::Ax210));
    claim("at A step, which is the a0 in its firmware's name", real.step == 0);
    let real_rf = rf_of(0x0010_a100);
    claim("and its radio decodes as Harrier", real_rf.rf == Rf::Hr2);
    claim("at step 1", real_rf.step == 1);
    claim("with neither the two-die nor the jacket flag", !real_rf.cdb && !real_rf.jacket);
    // The name its own kernel printed: "Detected Intel(R) Wi-Fi 6 AX201 160MHz".
    claim(
        "and together they are the AX201 its own kernel named",
        product_name(real.mac, real_rf.rf) == Some("AX201"),
    );
    // The step is where the old decoding and the new one happen to agree on this
    // part, and saying so is the honest version: 0x370's low nibble is zero, so
    // both answer A. The fix matters on QuZ, where upstream's own re-pack gives
    // step 0 and taking bits 2..4 gives 1 -- so the wrong image would be chosen
    // for that part and not for this one.
    claim(
        "the old decoding agreed on this part and disagrees on QuZ",
        (GF63_HW_REV >> 2) & 3 == GF63_HW_REV & 3 && (0x354u32 >> 2) & 3 != 0x354u32 & 3,
    );

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
    // **A faulting read is its own answer.** Guarded rather than fatal, because
    // an unvalidated aperture is the one thing a probe cannot check before
    // reading it -- and a dead machine teaches nothing on a trip taken to learn.
    claim(
        "a faulting read is a fourth, distinct from a silent one",
        Refusal::Faulted != Refusal::Silent && Refusal::Faulted != Refusal::NotMapped,
    );
    claim(
        "and it says the aperture mapped but the device did not decode it",
        Refusal::Faulted.why().contains("does not decode"),
    );

    // --- the power-up sequence, which is why it is a table ------------------
    //
    // No emulator models this part, so the ordering cannot be observed anywhere.
    // As data it can be asserted, and the ordering is the half that is fatal
    // when wrong: a poll before the thing it waits for has been asked for is a
    // timeout that reads as broken hardware.
    let idx = |pred: fn(&Step) -> bool| POWER_UP.iter().position(pred);
    let acquire = idx(|s| matches!(s, Step::Acquire { .. }));
    let alive = idx(|s| matches!(s, Step::SetBit(CSR_MBOX_SET_REG, _)));
    let reset = idx(|s| matches!(s, Step::SetBit(CSR_RESET, _)));
    let init_done = idx(|s| matches!(s, Step::SetBit(CSR_GP_CNTRL, b) if *b == CSR_GP_CNTRL_REG_FLAG_INIT_DONE));
    let clock = idx(|s| matches!(s, Step::Poll { mask, .. } if *mask == CSR_GP_CNTRL_REG_FLAG_MAC_CLOCK_READY));

    // **The semaphore comes before the reset, and this group used to assert the
    // opposite.** `iwx_start_hw` runs the whole `NIC_READY` handshake in
    // `prepare_card_hw` and only then `sw_reset`; the first version of this table
    // reset at index 0 and handshook afterwards, and a claim said "the sequence
    // resets first" -- so the table-as-data design caught nothing, because the
    // order it was asserting was itself wrong. Kept as the first claim in the
    // group, pointing the other way.
    claim("the sequence takes the semaphore first", acquire == Some(0));
    claim("the reset comes after the handshake, not before it", acquire < reset);
    claim("and the host declares itself alive as soon as it is granted", acquire < alive && alive < reset);
    claim(
        "the reset settles before anything is asked of the part",
        reset.and_then(|i| POWER_UP.get(i + 1)).map(|s| matches!(s, Step::Settle(_))) == Some(true),
    );
    claim("initialisation is declared after the reset", reset < init_done);
    claim(
        "and the clock is polled only after INIT_DONE, never before",
        init_done < clock,
    );
    // The bit nothing sets is the bug this cost: `NIC_READY` is a semaphore the
    // driver writes, so a table that only polled it would wait for a value
    // nothing was going to produce. Asserted as an absence, which is the only
    // shape that catches it coming back.
    claim(
        "NIC_READY is never merely polled, because nothing but us sets it",
        !POWER_UP
            .iter()
            .any(|s| matches!(s, Step::Poll { mask, .. } if *mask == CSR_HW_IF_CONFIG_REG_BIT_NIC_READY)),
    );
    // Upstream sets bits in the FH threshold register; writing it whole clears
    // the other fields in it.
    claim(
        "the FH threshold is set rather than written whole",
        POWER_UP
            .iter()
            .any(|s| matches!(s, Step::SetBit(CSR_DBG_HPET_MEM_REG, _)))
            && !POWER_UP.iter().any(|s| matches!(s, Step::Write(CSR_DBG_HPET_MEM_REG, _))),
    );
    // An Acquire with no retries is a handshake that cannot recover from a link
    // in a low-power state, which is the case the fallback exists for.
    claim(
        "and the acquisition is allowed to retry",
        POWER_UP.iter().all(|s| !matches!(s, Step::Acquire { tries: 0 })),
    );

    // A poll with no timeout is a hang, and this runs before there is a shell to
    // interrupt it from.
    claim(
        "every poll has a non-zero timeout",
        POWER_UP.iter().all(|s| !matches!(s, Step::Poll { us: 0, .. })),
    );
    // Every offset must be inside the one page `power_up` maps. A register past
    // it is a write into whatever the identity map has there.
    claim(
        "every register is inside the mapped aperture",
        POWER_UP.iter().all(|s| match s {
            Step::SetBit(r, _) | Step::Write(r, _) => *r < APERTURE,
            Step::Poll { reg, .. } => *reg < APERTURE,
            Step::Settle(_) => true,
            // Every register `acquire` touches, listed here rather than trusted,
            // because they are inside a function and so invisible to the sweep
            // that checks the table.
            Step::Acquire { .. } => {
                CSR_HW_IF_CONFIG_REG < APERTURE && CSR_DBG_LINK_PWR_MGMT_REG < APERTURE
            }
        }),
    );
    // L1 must survive: upstream disables L0s alone, and disabling both would
    // cost the link's power management for no reason this driver needs.
    claim(
        "the L0s workaround leaves L1 alone",
        CSR_GIO_CHICKEN_BITS_REG_BIT_L1A_NO_L0S_RX & 0x2000_0000 == 0,
    );
    // A fault names its step, so a transcript says which handshake failed.
    claim(
        "a timeout names the register it was waiting on",
        Fault::Timeout(clock.unwrap_or(0)).why().contains("024"),
    );
    claim(
        "and an aperture fault carries the probe's own reason",
        Fault::Aperture(Refusal::Asleep).why() == Refusal::Asleep.why(),
    );

    // The firmware container and the descriptor the part boots out of. **Nearly
    // all of this port is checkable without the laptop**, which was not obvious
    // and is worth saying: family 22000 takes its microcode by DMA from one
    // 1,792-byte structure, so the bring-up is mostly laying out memory and only
    // the last two writes touch a register. What needs the radio is whether it
    // answers, not whether the bytes are right.
    out.extend(fw::checks());
    out.extend(ctxt::checks());
    out.extend(gen3::checks());
    out.extend(gen3::kick_checks());
    out.extend(alive::checks());
    out.extend(cmd::checks());
    out.extend(nvm::checks());
    out.extend(init::checks());
    out.extend(config::checks());
    out.extend(power::checks());
    out
}

// --- reset and power-up ------------------------------------------------------
//
// **Everything above reads. Everything below writes**, to a device this kernel
// has never driven, and that distinction is the reason the two halves are
// separated rather than folded into one `bring_up`. A probe that only reads can
// be wrong and leave the machine as it found it; this cannot.
//
// The sequence is `iwlwifi`'s `iwl_pcie_sw_reset`, `iwl_pcie_prepare_card_hw`
// and `iwl_pcie_apm_init` for family 22000, which is the AX201's. Register
// names are `iwl-csr.h`'s, kept verbatim so a reader can find them in either
// upstream; the offsets and bits are the numbers those names are defined as.
// **Nothing here is guessed.** A register poked at random on a radio is not a
// bug that reports itself -- it is a part that goes quiet, or a machine that
// takes a machine check.

/// Set the NIC-ready handshake, and read it back.
const CSR_HW_IF_CONFIG_REG: u64 = 0x000;
/// Software reset. Bit 7 on this family; AX210 moved it to `CSR_GP_CNTRL`.
const CSR_RESET: u64 = 0x020;
/// Clock and power state, and where "initialisation complete" is declared.
const CSR_GP_CNTRL: u64 = 0x024;
/// PCIe link workarounds. L0s is disabled here and L1 is left alone.
const CSR_GIO_CHICKEN_BITS: u64 = 0x100;
/// The FH wait threshold, set to its maximum as a stress workaround.
const CSR_DBG_HPET_MEM_REG: u64 = 0x240;
/// Where the driver tells firmware the operating system is up.
const CSR_MBOX_SET_REG: u64 = 0x088;
/// Link power management, which has to be disabled before the part will grant
/// the semaphore on a machine where it did not the first time.
const CSR_DBG_LINK_PWR_MGMT_REG: u64 = 0x250;

/// Ask the device to become ready, and the bit that says it did.
const CSR_HW_IF_CONFIG_REG_PREPARE: u32 = 0x0800_0000;
const CSR_HW_IF_CONFIG_REG_BIT_NIC_READY: u32 = 0x0040_0000;
/// Let the management bus raise an interrupt on an EEPROM access attempt.
const CSR_HW_IF_CONFIG_REG_BIT_HAP_WAKE_L1A: u32 = 0x0008_0000;

const CSR_RESET_REG_FLAG_SW_RESET: u32 = 0x0000_0080;

/// Moves the adapter from D0U* to D0A*.
const CSR_GP_CNTRL_REG_FLAG_INIT_DONE: u32 = 0x0000_0004;
/// The clock has stabilised. Polled, never assumed.
const CSR_GP_CNTRL_REG_FLAG_MAC_CLOCK_READY: u32 = 0x0000_0001;

/// Disable L0s without affecting L1. An ICH erratum: do not wait for L0s.
const CSR_GIO_CHICKEN_BITS_REG_BIT_L1A_NO_L0S_RX: u32 = 0x0080_0000;

/// The maximum wait threshold.
const CSR_DBG_HPET_MEM_REG_VAL: u32 = 0xFFFF_0000;
const CSR_MBOX_SET_REG_OS_ALIVE: u32 = 0x20;
const CSR_RESET_LINK_PWR_MGMT_DISABLED: u32 = 0x8000_0000;
/// Upstream's own figure, and it is **microseconds**: fifty, not fifty
/// milliseconds. The semaphore is granted immediately or the part needs the
/// prepare dance, so a generous timeout here buys nothing and hides which of the
/// two happened.
const HW_READY_TIMEOUT_US: u32 = 50;

/// One step of the sequence.
///
/// **Data rather than straight-line code, and that is the whole point.** No
/// emulator models this part, so the ordering -- which is the thing that is easy
/// to get wrong and fatal when it is -- could not otherwise be checked anywhere.
/// As a table a suite can assert that the reset comes before the handshake, that
/// the handshake comes before "initialisation complete", that the clock is
/// polled after it is asked for and not before, and that no poll has a zero
/// timeout. None of that needs a radio.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    /// Read, or-in, write back. The register keeps every bit the driver has not
    /// been told about, which matters because several of these hold state set by
    /// firmware that has already run.
    SetBit(u64, u32),
    /// Write a whole word, for the registers that are thresholds rather than
    /// flag sets.
    Write(u64, u32),
    /// Poll until `reg & mask == want`, giving up after so many microseconds.
    /// A timeout is a refusal, never a warning that gets ignored.
    Poll { reg: u64, mask: u32, want: u32, us: u32 },
    /// Wait unconditionally. Used only where upstream does, because the
    /// hardware gives nothing to poll on.
    Settle(u32),
    /// Take the `NIC_READY` semaphore, with the prepare-and-retry fallback.
    ///
    /// **One step because it is a loop, and a table of steps cannot hold one.**
    /// Upstream tries the semaphore, and on refusal disables link power
    /// management and then asserts `PREPARE` and retries up to `tries` times.
    /// Modelling that as straight-line entries would either drop the retry or
    /// unroll it into ten copies whose ordering the claims could not read.
    Acquire { tries: u32 },
}

/// The declared sequence, in order.
///
/// Read from `iwlwifi`'s family-22000 path. Each entry's comment is the reason
/// upstream gives, because a workaround with no reason attached is one somebody
/// deletes.
pub const POWER_UP: &[Step] = &[
    // **The semaphore first, then the reset.** This had them the other way
    // round, which is upstream's order reversed: `iwx_start_hw` runs
    // `prepare_card_hw` -- the whole `NIC_READY` handshake -- and only then
    // `sw_reset`. Resetting a part that has not granted the semaphore resets it
    // out from under the handshake that was about to happen.
    Step::Acquire { tries: 10 },
    // Firmware is told the host is up. Upstream does this inside
    // `set_hw_ready` the moment the semaphore is granted, so it belongs with the
    // acquisition rather than later.
    Step::SetBit(CSR_MBOX_SET_REG, CSR_MBOX_SET_REG_OS_ALIVE),
    // Now the reset. 5 ms is upstream's figure for everything below Bz, where it
    // is a different register and 20 ms -- which is one of the two reasons this
    // table refuses that family rather than driving it.
    Step::SetBit(CSR_RESET, CSR_RESET_REG_FLAG_SW_RESET),
    Step::Settle(5_000),
    // From here it is `apm_init`, in its order. It is the same for family 22000
    // and AX210 and differs only at Bz, which is the other reason for the gate.
    //
    // Disable L0s without touching L1: an ICH erratum, and the reason upstream
    // does not simply disable both.
    Step::SetBit(CSR_GIO_CHICKEN_BITS, CSR_GIO_CHICKEN_BITS_REG_BIT_L1A_NO_L0S_RX),
    // FH wait threshold to maximum: a hardware error under stress otherwise.
    // **Set rather than written**, which this had as a whole-word write --
    // upstream uses set-bits, and the register holds other fields, so writing it
    // whole clears whatever firmware or a previous boot left in them.
    Step::SetBit(CSR_DBG_HPET_MEM_REG, CSR_DBG_HPET_MEM_REG_VAL),
    // Let the management bus wake the link out of L1a, which is how a driver
    // finds out something else is reaching for the part.
    Step::SetBit(CSR_HW_IF_CONFIG_REG, CSR_HW_IF_CONFIG_REG_BIT_HAP_WAKE_L1A),
    // Declare initialisation complete, moving D0U* -> D0A*.
    Step::SetBit(CSR_GP_CNTRL, CSR_GP_CNTRL_REG_FLAG_INIT_DONE),
    // And only then wait for the clock. Polling before INIT_DONE waits for
    // something nothing has been asked to do, which is a 25 ms timeout that
    // reads as broken hardware.
    Step::Poll {
        reg: CSR_GP_CNTRL,
        mask: CSR_GP_CNTRL_REG_FLAG_MAC_CLOCK_READY,
        want: CSR_GP_CNTRL_REG_FLAG_MAC_CLOCK_READY,
        us: 25_000,
    },
];
/// Why the sequence stopped, with the step that stopped it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fault {
    /// The aperture was not usable. The probe's refusal, repeated here because
    /// this entry point can be reached without one.
    Aperture(Refusal),
    /// A poll ran out. Carries the index into `POWER_UP`, so the transcript
    /// names which handshake failed rather than that one did.
    Timeout(usize),
    /// The part never granted the semaphore, after the prepare dance and every
    /// retry. Distinguished from a poll timeout because it is the one failure
    /// that means something else owns the part rather than that it is broken.
    NotGranted,
    /// The controller is a generation this sequence is not written for. Carries
    /// what it is, because "unsupported" and "a family whose reset is a different
    /// register" are different things to read in a transcript.
    WrongFamily(Family),
    /// The revision did not decode to a controller this kernel can name, so
    /// there is no family and nothing to gate on.
    UnknownPart(u16),
}

impl Fault {
    pub fn why(&self) -> String {
        match self {
            Fault::Aperture(r) => String::from(r.why()),
            Fault::Timeout(i) => match POWER_UP.get(*i) {
                Some(Step::Poll { reg, mask, .. }) => alloc::format!(
                    "step {} timed out waiting for {:#05x} bit {:#010x}",
                    i, reg, mask
                ),
                _ => alloc::format!("step {} timed out", i),
            },
            Fault::NotGranted => String::from(
                "the part never granted NIC_READY, even after disabling link power management -- something else may own it",
            ),
            Fault::WrongFamily(f) => alloc::format!(
                "this is {} family silicon and the sequence here is for 22000 and AX210",
                f.name()
            ),
            Fault::UnknownPart(t) => alloc::format!(
                "controller type {:#05x} has no family here, so nothing can be driven safely",
                t
            ),
        }
    }
}

impl Radio {
    /// Run the sequence. **This writes to the radio.**
    ///
    /// Gated on the revision decoding to a MAC this kernel can name, because the
    /// offsets above are family 22000's. Poking them at a part from another
    /// family is not a bug that reports itself; it is a device that goes quiet,
    /// and possibly a machine check. `Mac::Unknown` therefore refuses rather
    /// than trying, which is the same rule `dev::power` applies to an MSR whose
    /// gate it cannot confirm.
    ///
    /// Nothing here enables bus mastering or an interrupt. The firmware has not
    /// been loaded, so there is nothing to DMA and nothing to signal; granting
    /// either now would be granting it to a device that cannot yet be told to
    /// stop.
    pub fn power_up(&self, ecam: u64) -> Result<(), Fault> {
        let bar0 = self
            .bar0
            .filter(|&b| b != 0)
            .ok_or(Fault::Aperture(Refusal::NoAperture))?;
        // **Gated on the family and not on the type being named.** It was gated
        // on `Mac::known()`, which was the right shape and the wrong question:
        // the sequence below is `apm_init`'s, which is identical for family 22000
        // and AX210 and differs at Bz in both the reset register and the clock
        // bits. So a Bz part is recognised, named, and refused -- where the old
        // gate would have driven it the moment somebody added its type to the
        // table, and a #GP or a silent part is what that costs.
        let rev = self.hw_rev(ecam).map_err(Fault::Aperture)?.0;
        match rev.mac.family() {
            Some(Family::F22000) | Some(Family::Ax210) => {}
            Some(f) => return Err(Fault::WrongFamily(f)),
            None => {
                return Err(Fault::UnknownPart(match rev.mac {
                    Mac::Unknown(t) => t,
                    _ => 0,
                }))
            }
        }
        enable_memory_space(ecam, &self.dev);
        if !crate::mem::paging::map_range(bar0, APERTURE, true) {
            return Err(Fault::Aperture(Refusal::NotMapped));
        }
        for (i, step) in POWER_UP.iter().enumerate() {
            match *step {
                // Safety: every offset in `POWER_UP` is inside the first 4 KiB,
                // which `apertures_hold` asserts and the mapping above covers.
                Step::SetBit(reg, bit) => unsafe {
                    let p = (bar0 + reg) as *mut u32;
                    core::ptr::write_volatile(p, core::ptr::read_volatile(p) | bit);
                },
                Step::Write(reg, val) => unsafe {
                    core::ptr::write_volatile((bar0 + reg) as *mut u32, val);
                },
                Step::Settle(us) => crate::time::delay_us(us as u64),
                Step::Poll { reg, mask, want, us } => {
                    if !poll_bit(bar0, reg, mask, want, us) {
                        return Err(Fault::Timeout(i));
                    }
                }
                Step::Acquire { tries } => {
                    if !acquire(bar0, tries) {
                        return Err(Fault::NotGranted);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Take the `NIC_READY` semaphore.
///
/// **`NIC_READY` is written by the driver, not set by the device**, and missing
/// that was the worst defect in the first version of this table: it polled the
/// bit without ever setting it, so the handshake waited 35 ms for something
/// nothing was going to write and then reported broken hardware. The register's
/// own comment in Intel's header says `PCI_OWN_SEM` -- it is a claim on the part,
/// and the read-back is whether the claim was granted.
///
/// The fallback is upstream's: disable link power management, then assert
/// `PREPARE` and try again, because a part in a low-power link state will not
/// grant it until the link comes up.
fn acquire(bar0: u64, tries: u32) -> bool {
    if set_hw_ready(bar0) {
        return true;
    }
    // Safety: as `power_up` -- both offsets are inside the mapped page.
    unsafe {
        let p = (bar0 + CSR_DBG_LINK_PWR_MGMT_REG) as *mut u32;
        core::ptr::write_volatile(p, core::ptr::read_volatile(p) | CSR_RESET_LINK_PWR_MGMT_DISABLED);
    }
    crate::time::delay_us(1_000);
    for _ in 0..tries {
        unsafe {
            let p = (bar0 + CSR_HW_IF_CONFIG_REG) as *mut u32;
            core::ptr::write_volatile(p, core::ptr::read_volatile(p) | CSR_HW_IF_CONFIG_REG_PREPARE);
        }
        // Upstream gives the pair 150 ms in total per outer attempt, in 200 us
        // steps, and then waits 25 ms before asserting PREPARE again.
        let mut waited = 0u32;
        while waited < 150_000 {
            if set_hw_ready(bar0) {
                return true;
            }
            crate::time::delay_us(200);
            waited += 200;
        }
        crate::time::delay_us(25_000);
    }
    false
}

/// Write the semaphore bit and read it back.
fn set_hw_ready(bar0: u64) -> bool {
    // Safety: as `power_up`.
    unsafe {
        let p = (bar0 + CSR_HW_IF_CONFIG_REG) as *mut u32;
        core::ptr::write_volatile(
            p,
            core::ptr::read_volatile(p) | CSR_HW_IF_CONFIG_REG_BIT_NIC_READY,
        );
    }
    poll_bit(
        bar0,
        CSR_HW_IF_CONFIG_REG,
        CSR_HW_IF_CONFIG_REG_BIT_NIC_READY,
        CSR_HW_IF_CONFIG_REG_BIT_NIC_READY,
        HW_READY_TIMEOUT_US,
    )
}

/// Everything that stands between a powered part and firmware saying it is up.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum BootFault {
    Power(Fault),
    /// The revision says this is not a family the boot path is written for.
    /// Distinct from `Power(WrongFamily)`: the power-up sequence covers 22000 and
    /// AX210 both, and only AX210 has a boot path here.
    WrongFamily(Family),
    Build(gen3::Error),
    Kick(gen3::KickFault),
    NotAlive(alive::Wait),
    NoMemory,
}

impl BootFault {
    pub fn why(&self) -> String {
        match self {
            BootFault::Power(f) => f.why(),
            BootFault::WrongFamily(f) => alloc::format!(
                "this is {} family silicon; only AX210 has a boot path here",
                f.name()
            ),
            BootFault::Build(e) => e.why(),
            BootFault::Kick(k) => String::from(k.why()),
            BootFault::NotAlive(w) => w.why(),
            BootFault::NoMemory => String::from("not enough memory for the receive buffers"),
        }
    }
}

/// What a successful boot leaves behind.
///
/// **Held together and returned, because the part is still reading from all of
/// it.** Firmware fetched its image at the kick and goes on using the receive
/// ring, the scratch and the peripheral info page for as long as it runs; there
/// is no completion to wait on. So the regions live as long as the caller keeps
/// this, and dropping it is how the part is taken down rather than a tidy-up.
pub struct Booted {
    pub alive: alive::Alive,
    pub boot: gen3::Boot,
    pub buffers: alive::Buffers,
    /// Where the driver has got to in the receive ring. Carried because the
    /// firmware goes on sending and the next reader must not start at zero.
    pub rx: alive::Rx,
    /// The command queue, which is how anything is asked of the part.
    pub cmds: cmd::Queue,
    /// Whether the post-alive handshake has run.
    ///
    /// Remembered rather than re-run, because its third step tells firmware there
    /// are no more NVM accesses coming -- a claim that is true once and false the
    /// second time somebody asks a question.
    pub configured: bool,
}

impl Radio {
    /// Power the part up, load its firmware, and wait for it to say it is alive.
    ///
    /// **This is the first thing in this driver that grants the device DMA**, and
    /// that is why it is here and not in the probe. `hw_rev` deliberately enables
    /// memory-space decoding and leaves bus mastering alone, on the argument that
    /// a probe has no business granting DMA to a device whose firmware has never
    /// run. Booting is exactly the moment that stops being true: the part fetches
    /// its own microcode, so it must be able to master the bus, and there is no
    /// IOMMU here -- the only safety property available is that every address it
    /// is given is one this driver allocated.
    pub fn boot(&self, ecam: u64, image: &fw::Image, file: &[u8], ms: u32) -> Result<Booted, BootFault> {
        let rev = self.hw_rev(ecam).map_err(|r| BootFault::Power(Fault::Aperture(r)))?.0;
        match rev.mac.family() {
            Some(Family::Ax210) => {}
            Some(f) => return Err(BootFault::WrongFamily(f)),
            None => {
                return Err(BootFault::WrongFamily(Family::Bz));
            }
        }
        // The sequence first: the clock has to be running before a peripheral
        // write lands, and `kick` ends in one.
        self.power_up(ecam).map_err(BootFault::Power)?;

        let bar0 = self
            .bar0
            .filter(|&b| b != 0)
            .ok_or(BootFault::Power(Fault::Aperture(Refusal::NoAperture)))?;

        // **Built before bus mastering is granted**, deliberately. Every address
        // in these structures has to be settled before the part can act on any of
        // them, and building can fail -- a firmware file with no image loader --
        // in which case nothing was ever allowed to touch memory.
        let boot = gen3::build(rev.raw, image, file).map_err(BootFault::Build)?;
        let buffers = alive::Buffers::new().ok_or(BootFault::NoMemory)?;
        let mut boot = boot;

        // Clear the interrupt status before anything can set it, or the poll
        // reads a bit left over from a previous boot and declares a firmware
        // that never ran alive.
        // Safety: the aperture is mapped by `power_up` above.
        unsafe { alive::arm(bar0) };

        crate::dev::pci::enable_bus_master(ecam, &self.dev);

        // Safety: an AX210 part whose power-up completed, with every address in
        // `boot` pointing at memory this driver owns.
        unsafe { gen3::kick(bar0, &boot) }.map_err(BootFault::Kick)?;
        let mut rx = alive::Rx::new();
        let a = unsafe { alive::wait(bar0, &mut boot.rings, &buffers, &mut rx, ms) }
            .map_err(BootFault::NotAlive)?;
        let cmds = cmd::Queue::new().ok_or(BootFault::NoMemory)?;
        Ok(Booted { alive: a, boot, buffers, rx, cmds, configured: false })
    }
}

impl Radio {
    /// Ask a booted part what it knows about itself.
    ///
    /// Takes the `Booted` it was given rather than booting again, because the
    /// firmware is already running and the receive cursor is already somewhere:
    /// re-booting to ask a question would throw away the position and the part
    /// would be sent its microcode twice.
    ///
    /// The address is read under the MAC access lock and the rest comes back in
    /// the answer, which is the split upstream makes -- the response does not
    /// carry an address.
    pub fn nvm(&self, b: &mut Booted, ms: u32) -> Result<nvm::Nvm, nvm::NvmError> {
        // **The handshake first, and it is not optional.** `NVM_GET_INFO` was sent
        // with none of it in front, which is a question put to a part that has not
        // been told the question is coming. Run once and remembered, because the
        // second step says there are no more NVM accesses -- saying that twice is
        // saying something untrue the second time.
        if !b.configured {
            let bar0 = self
                .bar0
                .filter(|&a| a != 0)
                .ok_or(nvm::NvmError::Cmd(cmd::CmdError::NoQueue))?;
            // Safety: a booted part whose aperture `boot` mapped.
            unsafe {
                init::handshake(bar0, &mut b.boot.rings, &b.buffers, &mut b.rx, &mut b.cmds, &b.alive, ms)
            }
            .map_err(nvm::NvmError::Init)?;
            b.configured = true;
        }
        let bar0 = self.bar0.filter(|&a| a != 0).ok_or(nvm::NvmError::Cmd(cmd::CmdError::NoQueue))?;
        // Safety: a booted part, whose aperture `boot` mapped and whose firmware
        // is alive; the lock is taken and released around the reads.
        let mac = unsafe {
            let held = gen3::lock(bar0);
            let m = nvm::read_mac(bar0, Family::Ax210);
            if held {
                gen3::unlock(bar0);
            }
            m
        };
        if !nvm::valid_mac(&mac) {
            return Err(nvm::NvmError::NoAddress(mac));
        }
        // Safety: as above, and the command queue is the one `boot` allocated.
        let pkt = unsafe {
            cmd::ask(
                bar0,
                &mut b.boot.rings,
                &b.buffers,
                &mut b.rx,
                &mut b.cmds,
                nvm::REGULATORY_AND_NVM_GROUP,
                nvm::NVM_GET_INFO,
                0,
                &nvm::REQUEST,
                ms,
            )
        }
        .map_err(nvm::NvmError::Cmd)?;
        nvm::parse(pkt.payload, mac).ok_or(nvm::NvmError::UnknownVersion(pkt.payload.len()))
    }
}

impl Radio {
    /// Tell a part that has answered about itself how to behave.
    ///
    /// Takes the NVM rather than reading it again, because the antenna
    /// configuration is a value out of it -- so this command genuinely cannot be
    /// sent before that question has been asked, which is why it is a separate
    /// call and not folded into `nvm`.
    ///
    /// Answers what it sent and what it skipped. **A success here does not mean the
    /// part can scan**: seven of upstream's twelve are written and `config.rs` names
    /// the five that are not.
    pub fn configure(
        &self,
        ecam: u64,
        b: &mut Booted,
        image: &fw::Image,
        n: &nvm::Nvm,
    ) -> Result<config::Done, config::Fault> {
        let bar0 = match self.bar0.filter(|&a| a != 0) {
            Some(a) => a,
            None => return Err(config::Fault::At(0, cmd::CmdError::NoQueue)),
        };
        let f = config::Facts {
            tx_ant: n.tx_chains,
            // Upstream's flag for this product id, followed rather than reasoned
            // about -- see the constant's own note on why it reads oddly.
            discrete: true,
            xtal_latency: 0,
            ltr_enabled: config::ltr_enabled(ecam, &self.dev),
            // Not sleeping. `power.rs` argues for it: on a machine whose job is
            // mining or serving, a radio asleep between beacons trades latency for
            // power nobody asked to save.
            power_level: 0,
        };
        // Safety: a part whose firmware is alive and which has been through the
        // handshake, on an aperture `boot` mapped.
        unsafe { config::configure(bar0, &mut b.boot.rings, &mut b.cmds, image, &f) }
    }
}

/// Poll one register until it answers, or give up.
///
/// Ten-microsecond steps, which is upstream's granularity. The read is volatile
/// for the reason every register read is: the value changes underneath a
/// compiler that has been told nothing writes it.
fn poll_bit(bar0: u64, reg: u64, mask: u32, want: u32, us: u32) -> bool {
    let mut waited = 0u32;
    loop {
        // Safety: as `power_up`.
        let v = unsafe { core::ptr::read_volatile((bar0 + reg) as *const u32) };
        if v & mask == want {
            return true;
        }
        if waited >= us {
            return false;
        }
        crate::time::delay_us(10);
        waited += 10;
    }
}

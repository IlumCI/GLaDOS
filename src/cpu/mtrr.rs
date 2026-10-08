//! What memory type an address really has.
//!
//! **Two tables decide it and this kernel had only ever looked at one.** A page
//! table entry selects one of eight PAT slots through its PWT, PCD and PAT
//! bits, and the firmware's MTRRs assign every physical range a type of their
//! own. The processor combines the two, and the combination is not "the page
//! wins": a page marked write-back over a range the MTRRs call uncacheable is
//! uncacheable. So `paging.rs` mapping the framebuffer write-back says nothing
//! about whether writes to it are cached, and the answer matters twice over --
//! uncacheable makes every present slow, and genuinely cached makes it wrong,
//! because Intel's display engine does not snoop the CPU caches (i915 pins
//! scanout uncached or write-through for exactly that reason, and Linux's
//! `efifb` and `simpledrm` map the firmware framebuffer write-combining).
//!
//! The decoding is pure and asserted at `diag pat` against synthetic register
//! values, for the reason `mem::fixed` gives about its own map: a claim written
//! against this machine's MTRRs would be a claim about one firmware.
//!
//! Intel SDM Vol. 3A, "Memory Cache Control": the PAT MSR layout, the variable
//! range pairs, and the table combining a PAT type with an MTRR type.

use alloc::vec::Vec;

/// IA32_MTRRCAP: how many variable ranges exist, and whether WC is supported.
pub const IA32_MTRRCAP: u32 = 0xFE;
/// IA32_PAT: eight one-byte slots.
pub const IA32_PAT: u32 = 0x277;
/// IA32_MTRR_DEF_TYPE: the default type and the global enable.
pub const IA32_MTRR_DEF_TYPE: u32 = 0x2FF;
/// IA32_MTRR_PHYSBASE0; mask n is base n + 1, pairs follow.
pub const IA32_MTRR_PHYSBASE0: u32 = 0x200;

/// The architectural power-on PAT: WB, WT, UC-, UC, repeated. Any slot not
/// changed by software still holds this.
pub const PAT_POWER_ON: u64 = 0x0007_0406_0007_0406;

/// The six memory types the processor knows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Strong uncacheable.
    Uc,
    /// Write-combining: uncached, but stores gather into 64-byte bursts.
    Wc,
    /// Write-through.
    Wt,
    /// Write-protected.
    Wp,
    /// Write-back: ordinary cached memory.
    Wb,
    /// UC- ("UC minus"), which an MTRR of WC can override to WC. PAT only.
    UcMinus,
    /// An encoding the manual reserves. Reported, never guessed at.
    Reserved(u8),
}

impl Kind {
    /// Decode a PAT slot or an MTRR type field. `7` is UC- and only valid in
    /// the PAT; an MTRR holding it is reserved, which the caller asks for with
    /// `from_mtrr`.
    pub fn from_pat(b: u8) -> Kind {
        match b {
            0 => Kind::Uc,
            1 => Kind::Wc,
            4 => Kind::Wt,
            5 => Kind::Wp,
            6 => Kind::Wb,
            7 => Kind::UcMinus,
            n => Kind::Reserved(n),
        }
    }

    pub fn from_mtrr(b: u8) -> Kind {
        match b {
            7 => Kind::Reserved(7),
            n => Kind::from_pat(n),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::Uc => "uncacheable",
            Kind::Wc => "write-combining",
            Kind::Wt => "write-through",
            Kind::Wp => "write-protected",
            Kind::Wb => "write-back",
            Kind::UcMinus => "uncacheable (UC-)",
            Kind::Reserved(_) => "reserved encoding",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Kind::Uc => "UC",
            Kind::Wc => "WC",
            Kind::Wt => "WT",
            Kind::Wp => "WP",
            Kind::Wb => "WB",
            Kind::UcMinus => "UC-",
            Kind::Reserved(_) => "??",
        }
    }
}

/// Which PAT slot a leaf entry selects. `pat` is bit 7 of a 4 KiB entry and
/// bit 12 of a 2 MiB or 1 GiB one -- the caller reads the right bit, because
/// on a large page bit 7 is the page-size flag and means something else.
pub fn slot(pwt: bool, pcd: bool, pat: bool) -> u8 {
    (pat as u8) << 2 | (pcd as u8) << 1 | pwt as u8
}

/// The type a PAT register assigns to a slot.
pub fn pat_kind(pat_msr: u64, slot: u8) -> Kind {
    Kind::from_pat(((pat_msr >> (slot as u64 * 8)) & 0x7) as u8)
}

/// One variable-range pair as read from the MSRs.
#[derive(Clone, Copy, Debug)]
pub struct Range {
    pub base: u64,
    pub mask: u64,
}

impl Range {
    /// The valid bit, 11 of the mask.
    pub fn valid(&self) -> bool {
        self.mask & (1 << 11) != 0
    }

    pub fn kind(&self) -> Kind {
        Kind::from_mtrr((self.base & 0xFF) as u8)
    }

    /// Whether `addr` falls in this range. Bits 11:0 are flags and type, so
    /// they are cleared from both sides before comparing; the bits above the
    /// physical address width are clear in the mask, so they do not matter.
    pub fn covers(&self, addr: u64) -> bool {
        let m = self.mask & !0xFFF;
        self.valid() && (addr & m) == (self.base & m)
    }
}

/// The MTRRs' answer for one physical address above the first megabyte.
///
/// Below 1 MiB the fixed-range MTRRs apply; nothing this is asked about lives
/// there, so they are not decoded and the function says so by refusing.
///
/// Overlap rules are the manual's: any UC wins; WT beats WB; any other overlap
/// is undefined, and is reported as the first match rather than invented.
pub fn mtrr_kind(def_type: u64, ranges: &[Range], addr: u64) -> Option<Kind> {
    if addr < 0x10_0000 {
        return None;
    }
    // E, bit 11: MTRRs off means every address is UC.
    if def_type & (1 << 11) == 0 {
        return Some(Kind::Uc);
    }
    let mut hit: Option<Kind> = None;
    for r in ranges.iter().filter(|r| r.covers(addr)) {
        let k = r.kind();
        hit = Some(match (hit, k) {
            (None, k) => k,
            (_, Kind::Uc) | (Some(Kind::Uc), _) => Kind::Uc,
            (Some(Kind::Wt), Kind::Wb) | (Some(Kind::Wb), Kind::Wt) => Kind::Wt,
            (Some(h), _) => h,
        });
    }
    Some(hit.unwrap_or(Kind::from_mtrr((def_type & 0xFF) as u8)))
}

/// What the processor actually does, given the page's PAT type and the range's
/// MTRR type. SDM table "Effective Page-Level Memory Types for Pentium III and
/// More Recent Processor Families".
///
/// The two that matter here, stated because they are the point of the module:
/// PAT WC is WC whatever the MTRR says, and PAT WB takes the MTRR's type --
/// so WB over an MTRR-UC range is UC.
pub fn effective(pat: Kind, mtrr: Kind) -> Kind {
    use Kind::*;
    match (pat, mtrr) {
        (Reserved(n), _) | (_, Reserved(n)) => Reserved(n),
        (Uc, _) => Uc,
        (Wc, _) => Wc,
        (UcMinus, Wc) => Wc,
        (UcMinus, _) => Uc,
        (Wb, m) => m,
        (Wt, Uc) | (Wt, Wc) => Uc,
        (Wt, Wp) => Wp,
        (Wt, _) => Wt,
        (Wp, Uc) | (Wp, Wc) => Uc,
        (Wp, _) => Wp,
    }
}

/// Everything the registers say, read once.
pub struct Snapshot {
    pub pat: u64,
    pub def_type: u64,
    pub ranges: Vec<Range>,
    /// IA32_MTRRCAP bit 10.
    pub wc_supported: bool,
}

/// Whether the processor has the PAT (CPUID.1:EDX[16]).
pub fn pat_supported() -> bool {
    super::cpuid(1, 0)[3] & (1 << 16) != 0
}

/// Whether the processor has MTRRs (CPUID.1:EDX[12]).
pub fn mtrr_supported() -> bool {
    super::cpuid(1, 0)[3] & (1 << 12) != 0
}

/// Read the PAT and every variable MTRR pair on this core. `None` when CPUID
/// does not promise both, because reading an MSR it does not promise raises
/// #GP and every vector but #BP is fatal here.
pub fn read() -> Option<Snapshot> {
    if !pat_supported() || !mtrr_supported() {
        return None;
    }
    unsafe {
        let cap = super::rdmsr(IA32_MTRRCAP);
        let n = (cap & 0xFF) as u32;
        let ranges = (0..n.min(32))
            .map(|i| Range {
                base: super::rdmsr(IA32_MTRR_PHYSBASE0 + 2 * i),
                mask: super::rdmsr(IA32_MTRR_PHYSBASE0 + 2 * i + 1),
            })
            .collect();
        Some(Snapshot {
            pat: super::rdmsr(IA32_PAT),
            def_type: super::rdmsr(IA32_MTRR_DEF_TYPE),
            ranges,
            wc_supported: cap & (1 << 10) != 0,
        })
    }
}

/// Assert the decoding against register values whose answers are known.
pub fn checks(ok: &mut dyn FnMut(&str, bool)) {
    // Slot arithmetic: PWT is bit 0, PCD bit 1, PAT bit 2.
    ok("PWT alone selects slot 1", slot(true, false, false) == 1);
    ok("PWT+PCD selects slot 3", slot(true, true, false) == 3);
    ok("the PAT bit selects the upper half", slot(false, false, true) == 4);

    // The power-on PAT.
    ok("power-on slot 0 is WB", pat_kind(PAT_POWER_ON, 0) == Kind::Wb);
    ok("power-on slot 1 is WT", pat_kind(PAT_POWER_ON, 1) == Kind::Wt);
    ok("power-on slot 2 is UC-", pat_kind(PAT_POWER_ON, 2) == Kind::UcMinus);
    ok("power-on slot 3 is UC", pat_kind(PAT_POWER_ON, 3) == Kind::Uc);
    // Linux's layout, the one this kernel will program: slot 1 is WC.
    let linux = 0x0007_0406_0007_0106u64;
    ok("a PAT with slot 1 = 01 reads WC there", pat_kind(linux, 1) == Kind::Wc);
    ok("...and leaves slot 0 WB", pat_kind(linux, 0) == Kind::Wb);

    // Combination: the two facts this module exists for.
    ok("PAT WB over MTRR UC is UC", effective(Kind::Wb, Kind::Uc) == Kind::Uc);
    ok("PAT WC over MTRR UC is WC", effective(Kind::Wc, Kind::Uc) == Kind::Wc);
    ok("PAT WB over MTRR WC is WC", effective(Kind::Wb, Kind::Wc) == Kind::Wc);
    ok("PAT UC- over MTRR WC is WC", effective(Kind::UcMinus, Kind::Wc) == Kind::Wc);
    ok("PAT UC- over MTRR WB is UC", effective(Kind::UcMinus, Kind::Wb) == Kind::Uc);
    ok("PAT UC over anything is UC", effective(Kind::Uc, Kind::Wb) == Kind::Uc);

    // MTRR ranges. A 256 MiB WC window at 2 GiB, over a WB default.
    let def = (1 << 11) | 6;
    let wc = Range { base: 0x8000_0000 | 1, mask: 0xF_F000_0000 | (1 << 11) };
    let uc = Range { base: 0x8800_0000 | 0, mask: 0xF_F800_0000 | (1 << 11) };
    ok("an address inside a WC range reads WC",
       mtrr_kind(def, &[wc], 0x8010_0000) == Some(Kind::Wc));
    ok("outside every range reads the default",
       mtrr_kind(def, &[wc], 0x4000_0000) == Some(Kind::Wb));
    ok("an overlapping UC range wins",
       mtrr_kind(def, &[wc, uc], 0x8810_0000) == Some(Kind::Uc));
    let invalid = Range { base: wc.base, mask: wc.mask & !(1 << 11) };
    ok("a pair without the valid bit covers nothing",
       mtrr_kind(def, &[invalid], 0x8010_0000) == Some(Kind::Wb));
    ok("MTRRs disabled makes everything UC",
       mtrr_kind(6, &[wc], 0x8010_0000) == Some(Kind::Uc));
    ok("the first megabyte is refused, not guessed",
       mtrr_kind(def, &[wc], 0x8000).is_none());
}

/// The PAT this kernel runs with: the power-on layout with slot 1 changed from
/// write-through to write-combining. Linux uses the same slot for the same
/// purpose (`pat_bp_init`), and that choice is what makes the change safe to
/// make under a running kernel -- nothing here maps with PWT set and PCD
/// clear, which is the only bit pattern that selects slot 1, so no existing
/// mapping changes meaning when the slot does.
pub const GLADOS_PAT: u64 = 0x0007_0406_0007_0106;

/// Set by the bootstrap processor once its PAT holds `GLADOS_PAT`. Each
/// application processor programs the same value when it starts; the SDM
/// requires every core to agree, and a core that disagrees maps the
/// framebuffer with a type nobody chose.
static PROGRAMMED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

pub fn programmed() -> bool {
    PROGRAMMED.load(core::sync::atomic::Ordering::Acquire)
}

/// Load `GLADOS_PAT` on this core. Answers whether it holds afterwards.
///
/// The manual's procedure for a PAT change is the MTRR one: flush the caches,
/// change the register, flush the TLB, flush again. Slot 1 is unused at the
/// moment this runs, so a stale translation could only ever select the old
/// slot 1 for a mapping that does not exist -- the flushes are there because
/// the manual asks and they cost microseconds once per core per boot.
///
/// `bsp` publishes the result for the application processors to follow;
/// they call this with `false` and only when the bootstrap one succeeded.
pub fn program(bsp: bool) -> bool {
    if !pat_supported() {
        return false;
    }
    let ok = super::without_interrupts(|| unsafe {
        core::arch::asm!("wbinvd", options(nostack, preserves_flags));
        super::wrmsr(IA32_PAT, GLADOS_PAT);
        super::write_cr3(super::read_cr3());
        core::arch::asm!("wbinvd", options(nostack, preserves_flags));
        super::rdmsr(IA32_PAT) == GLADOS_PAT
    });
    if bsp {
        PROGRAMMED.store(ok, core::sync::atomic::Ordering::Release);
    }
    ok
}

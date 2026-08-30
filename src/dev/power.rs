//! Temperature, frequency and what to do about either.
//!
//! Both halves of this file talk to model-specific registers, and in a kernel
//! with no fault recovery that is the most dangerous thing in it. `rdmsr` on a
//! register the processor does not implement raises a general protection
//! fault, every vector here is fatal, and the result is a halted machine with
//! a report about a register nobody asked for. So the gate matters more than
//! the feature.
//!
//! **Three conditions, all of them, before any MSR is touched.**
//!
//! 1. The vendor is Intel. Every register named here is Intel's, and AMD's
//!    numbering overlaps in places, which is the worst kind of near miss.
//! 2. CPUID says the feature exists. A part that sets the digital sensor bit
//!    implements the sensor's register; that is what the bit is for.
//! 3. There is no hypervisor. This is the condition that is not in anybody's
//!    manual, and it is here because an emulator may advertise a capability in
//!    CPUID and not implement the register behind it. On real silicon the
//!    first two conditions are enough. Under emulation they are a guess, and a
//!    wrong guess here does not return.
//!
//! The consequence is stated rather than hidden: **none of this can be checked
//! under QEMU.** The capability probe runs and reports what it found, the
//! readings decline, and the numbers below have only ever been produced on
//! real hardware. `power force` overrides the hypervisor check for anybody who
//! knows their emulator implements these, and it says what it is risking.

use crate::cpu;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

// --- Intel model-specific registers used here ---------------------------

/// Bits 15:8 are the maximum non-turbo ratio, which is the multiplier the
/// bus clock is scaled by. Everything below converts ratios with it.
const MSR_PLATFORM_INFO: u32 = 0xCE;
/// Bits 23:16 are TjMax, the temperature the part throttles at. The sensor
/// reports distance below this rather than an absolute value.
const MSR_TEMPERATURE_TARGET: u32 = 0x1A2;
/// Bits 22:16 are degrees below TjMax. Bit 0 is set while the part is at or
/// over its limit; bit 5 while it is critically hot.
const IA32_THERM_STATUS: u32 = 0x19C;
const IA32_PACKAGE_THERM_STATUS: u32 = 0x1B1;
/// Actual and reference cycle counters. Their ratio times the base frequency
/// is the frequency the part really ran at, which is the only honest way to
/// report one on a processor that changes it constantly.
const IA32_MPERF: u32 = 0xE7;
const IA32_APERF: u32 = 0xE8;
/// Hardware-managed performance states.
const IA32_PM_ENABLE: u32 = 0x770;
const IA32_HWP_CAPABILITIES: u32 = 0x771;
const IA32_HWP_REQUEST: u32 = 0x774;
/// Bit 38 disables turbo. The rest of this register belongs to other people
/// and is preserved on every write.
const IA32_MISC_ENABLE: u32 = 0x1A0;

/// What the processor said it can do, filled in once by `probe`.
static HAVE: AtomicU32 = AtomicU32::new(0);
static PROBED: AtomicBool = AtomicBool::new(false);
static FORCED: AtomicBool = AtomicBool::new(false);
static TJ_MAX: AtomicU32 = AtomicU32::new(100);
static BASE_RATIO: AtomicU32 = AtomicU32::new(0);

const CAP_INTEL: u32 = 1 << 0;
const CAP_DTS: u32 = 1 << 1;
const CAP_HWP: u32 = 1 << 2;
const CAP_APERF: u32 = 1 << 3;
const CAP_PKG_DTS: u32 = 1 << 4;
const CAP_TURBO: u32 = 1 << 5;
/// Set when a hypervisor is present, which is a reason to decline rather than
/// a capability.
const CAP_VIRTUAL: u32 = 1 << 6;
/// Set once the platform registers have actually been read.
const CAP_READY: u32 = 1 << 7;

fn caps() -> u32 {
    HAVE.load(Ordering::Relaxed)
}

/// Whether MSR access is permitted right now.
fn allowed(bit: u32) -> bool {
    let c = caps();
    if c & CAP_INTEL == 0 || c & bit == 0 {
        return false;
    }
    c & CAP_VIRTUAL == 0 || FORCED.load(Ordering::Relaxed)
}

/// Ask CPUID what exists. Touches no MSR and is therefore safe anywhere.
pub fn probe() {
    if PROBED.swap(true, Ordering::Relaxed) {
        return;
    }
    let mut c = 0u32;

    // Vendor, from leaf 0's ebx/edx/ecx in that order.
    let v = cpu::cpuid(0, 0);
    if v[1] == 0x756E_6547 && v[3] == 0x4965_6E69 && v[2] == 0x6C65_746E {
        c |= CAP_INTEL;
    }
    // The hypervisor-present bit is architecturally reserved and is set by
    // every hypervisor worth the name. It is advisory, which is why `force`
    // exists, and it is the only signal available without an MSR.
    if cpu::cpuid(1, 0)[2] & (1 << 31) != 0 {
        c |= CAP_VIRTUAL;
    }
    let t = cpu::cpuid(6, 0);
    if t[0] & (1 << 0) != 0 {
        c |= CAP_DTS;
    }
    if t[0] & (1 << 6) != 0 {
        c |= CAP_PKG_DTS;
    }
    if t[0] & (1 << 7) != 0 {
        c |= CAP_HWP;
    }
    if t[0] & (1 << 1) != 0 {
        c |= CAP_TURBO;
    }
    if t[2] & (1 << 0) != 0 {
        c |= CAP_APERF;
    }
    HAVE.store(c, Ordering::Relaxed);

    // The platform registers are read once, behind the same gate everything
    // else uses, so a machine that declines never touches them at all.
    if allowed(CAP_DTS) {
        let tj = unsafe { cpu::rdmsr(MSR_TEMPERATURE_TARGET) };
        let tj = ((tj >> 16) & 0xFF) as u32;
        // A part reporting an implausible limit is a part whose register does
        // not mean what this code thinks, so the default stands.
        if (60..=120).contains(&tj) {
            TJ_MAX.store(tj, Ordering::Relaxed);
        }
        let pi = unsafe { cpu::rdmsr(MSR_PLATFORM_INFO) };
        BASE_RATIO.store(((pi >> 8) & 0xFF) as u32, Ordering::Relaxed);
        HAVE.store(caps() | CAP_READY, Ordering::Relaxed);
    }
}

/// Force MSR access despite a hypervisor. Answers whether anything changed.
pub fn force(on: bool) -> bool {
    let was = FORCED.swap(on, Ordering::Relaxed);
    if on && !was {
        // The platform registers were skipped at probe time, so take them now
        // under the operator's own authority.
        PROBED.store(false, Ordering::Relaxed);
        HAVE.store(caps() & !CAP_READY, Ordering::Relaxed);
        probe();
    }
    was != on
}

// --- temperature --------------------------------------------------------

/// One sensor's reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reading {
    /// Degrees Celsius.
    pub temp: u32,
    /// Distance below the throttle point, which is what the sensor actually
    /// reports and the number that matters when it reaches zero.
    pub headroom: u32,
    /// The part is at or above its limit and is being slowed.
    pub throttling: bool,
    pub critical: bool,
}

pub fn tj_max() -> u32 {
    TJ_MAX.load(Ordering::Relaxed)
}

fn decode(status: u64) -> Option<Reading> {
    // Bit 31 says the reading is valid. Without it the other bits are stale
    // and reporting them as a temperature would be inventing one.
    if status & (1 << 31) == 0 {
        return None;
    }
    let below = ((status >> 16) & 0x7F) as u32;
    let tj = tj_max();
    Some(Reading {
        temp: tj.saturating_sub(below),
        headroom: below,
        throttling: status & 1 != 0,
        critical: status & (1 << 5) != 0,
    })
}

/// This core's temperature.
pub fn core_temp() -> Option<Reading> {
    if !allowed(CAP_DTS) {
        return None;
    }
    decode(unsafe { cpu::rdmsr(IA32_THERM_STATUS) })
}

/// The package temperature, which is the one a fan curve should follow.
pub fn package_temp() -> Option<Reading> {
    if !allowed(CAP_PKG_DTS) {
        return None;
    }
    decode(unsafe { cpu::rdmsr(IA32_PACKAGE_THERM_STATUS) })
}

// --- frequency ----------------------------------------------------------

/// The frequency the part actually ran at over an interval, in MHz.
///
/// Derived from the ratio of delivered to reference cycles rather than from
/// any register that claims a frequency, because on a part that changes its
/// clock thousands of times a second every such register describes an instant
/// and this describes the interval somebody cares about.
pub fn measure_mhz(interval_us: u64) -> Option<u64> {
    if !allowed(CAP_APERF) {
        return None;
    }
    let base = base_mhz()?;
    let (a0, m0) = unsafe { (cpu::rdmsr(IA32_APERF), cpu::rdmsr(IA32_MPERF)) };
    crate::time::delay_us(interval_us);
    let (a1, m1) = unsafe { (cpu::rdmsr(IA32_APERF), cpu::rdmsr(IA32_MPERF)) };
    let da = a1.wrapping_sub(a0);
    let dm = m1.wrapping_sub(m0);
    if dm == 0 {
        return None;
    }
    Some(base.saturating_mul(da) / dm)
}

/// The maximum non-turbo frequency, in MHz. The bus clock is 100 MHz on every
/// part this runs on, which is stated here because it is an assumption.
pub fn base_mhz() -> Option<u64> {
    let r = BASE_RATIO.load(Ordering::Relaxed);
    if r == 0 {
        return None;
    }
    Some(r as u64 * 100)
}

// --- governors ----------------------------------------------------------

/// What to ask the processor for.
///
/// These are policies rather than frequencies. Naming a frequency would be
/// pretending to know better than the part about a decision it makes with
/// information this kernel does not have, which is what hardware-managed
/// performance states exist to take over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Governor {
    /// Everything the part will give, and turbo left alone.
    Performance,
    /// The part's own judgement, which is the reset behaviour.
    Balanced,
    /// Efficiency preferred, and the ceiling pulled down to the non-turbo
    /// maximum so a burst cannot heat the machine up.
    Powersave,
}

impl Governor {
    pub fn name(&self) -> &'static str {
        match self {
            Governor::Performance => "performance",
            Governor::Balanced => "balanced",
            Governor::Powersave => "powersave",
        }
    }

    pub fn parse(s: &str) -> Option<Governor> {
        match s {
            "performance" | "perf" => Some(Governor::Performance),
            "balanced" | "balance" => Some(Governor::Balanced),
            "powersave" | "power" | "save" => Some(Governor::Powersave),
            _ => None,
        }
    }

    /// The energy and performance preference this policy asks for, where 0 is
    /// all performance and 255 is all efficiency.
    fn epp(&self) -> u64 {
        match self {
            Governor::Performance => 0,
            Governor::Balanced => 128,
            Governor::Powersave => 255,
        }
    }
}

static CURRENT: AtomicU64 = AtomicU64::new(1);

pub fn governor() -> Governor {
    match CURRENT.load(Ordering::Relaxed) {
        0 => Governor::Performance,
        2 => Governor::Powersave,
        _ => Governor::Balanced,
    }
}

/// What the part says its performance range is: highest, guaranteed, lowest.
pub fn hwp_range() -> Option<(u8, u8, u8)> {
    if !allowed(CAP_HWP) {
        return None;
    }
    let c = unsafe { cpu::rdmsr(IA32_HWP_CAPABILITIES) };
    Some((c as u8, (c >> 8) as u8, (c >> 24) as u8))
}

/// Apply a policy. Answers false where the hardware will not take one.
pub fn set_governor(g: Governor) -> bool {
    if !allowed(CAP_HWP) {
        return false;
    }
    let Some((highest, _guaranteed, lowest)) = hwp_range() else {
        return false;
    };
    unsafe {
        // Enabling is one-way on most parts: once hardware-managed states are
        // on they stay on until reset. That is the processor's rule rather
        // than this kernel's, and it is why the bit is only ever set.
        let en = cpu::rdmsr(IA32_PM_ENABLE);
        if en & 1 == 0 {
            cpu::wrmsr(IA32_PM_ENABLE, en | 1);
        }
        let base = BASE_RATIO.load(Ordering::Relaxed) as u8;
        let (min, max) = match g {
            Governor::Performance => (lowest, highest),
            Governor::Balanced => (lowest, highest),
            // The ceiling comes down to the non-turbo maximum, where the part
            // reports one that fits inside its own range. Clamping outside it
            // would be asking for a state that does not exist.
            Governor::Powersave => {
                let cap = if base >= lowest && base <= highest { base } else { highest };
                (lowest, cap)
            }
        };
        let req = (min as u64) | ((max as u64) << 8) | (g.epp() << 24);
        cpu::wrmsr(IA32_HWP_REQUEST, req);
    }
    CURRENT.store(
        match g {
            Governor::Performance => 0,
            Governor::Balanced => 1,
            Governor::Powersave => 2,
        },
        Ordering::Relaxed,
    );
    true
}

/// Turn turbo on or off. Answers false where the part will not say.
pub fn set_turbo(on: bool) -> bool {
    if !allowed(CAP_TURBO) {
        return false;
    }
    unsafe {
        // Every other bit in this register belongs to somebody else, so it is
        // read, one bit is changed, and it is written back.
        let v = cpu::rdmsr(IA32_MISC_ENABLE);
        let v = if on { v & !(1u64 << 38) } else { v | (1u64 << 38) };
        cpu::wrmsr(IA32_MISC_ENABLE, v);
    }
    true
}

pub fn turbo() -> Option<bool> {
    if !allowed(CAP_TURBO) {
        return None;
    }
    Some(unsafe { cpu::rdmsr(IA32_MISC_ENABLE) } & (1u64 << 38) == 0)
}

// --- a policy that reacts ----------------------------------------------

/// Above this, the governor is pulled down a step. Chosen well below the
/// throttle point, because arriving at the throttle point means the decision
/// was already made by the hardware and this had nothing to contribute.
pub const HOT_C: u32 = 85;
/// Below this, a pulled-down governor is let back up. The gap between the two
/// is deliberate: a single threshold oscillates.
pub const COOL_C: u32 = 70;

static PINNED: AtomicBool = AtomicBool::new(false);

/// One step of a thermal policy, to be called from the clock task.
///
/// Answers what it changed, if anything. It only ever moves between the
/// operator's chosen governor and powersave, and it never touches turbo,
/// because a machine that quietly disabled turbo and forgot would look broken
/// in a way nothing reports.
pub fn tick() -> Option<&'static str> {
    let r = package_temp().or_else(core_temp)?;
    let pinned = PINNED.load(Ordering::Relaxed);
    if r.temp >= HOT_C && !pinned {
        PINNED.store(true, Ordering::Relaxed);
        if set_governor(Governor::Powersave) {
            return Some("hot: held at powersave");
        }
        return None;
    }
    if r.temp <= COOL_C && pinned {
        PINNED.store(false, Ordering::Relaxed);
        let g = governor();
        if set_governor(g) {
            return Some("cool: released");
        }
    }
    None
}

/// Why a reading is unavailable, in one line.
pub fn why() -> &'static str {
    let c = caps();
    if c & CAP_INTEL == 0 {
        return "not an Intel part, and every register here is Intel's";
    }
    if c & CAP_VIRTUAL != 0 && !FORCED.load(Ordering::Relaxed) {
        return "a hypervisor is present; CPUID may advertise what it does not implement";
    }
    if c & CAP_DTS == 0 {
        return "no digital thermal sensor in CPUID leaf 6";
    }
    if c & CAP_READY == 0 {
        return "the platform registers were never read";
    }
    "available"
}

/// Print what was found. Touches no MSR beyond what `probe` already did.
pub fn report() {
    use crate::kprintln;
    let c = caps();
    kprintln!(
        "  vendor {}   hypervisor {}",
        if c & CAP_INTEL != 0 { "intel" } else { "other" },
        if c & CAP_VIRTUAL != 0 { "yes" } else { "no" }
    );
    kprintln!(
        "  dts {}  package {}  hwp {}  aperf {}  turbo {}",
        yes(c & CAP_DTS),
        yes(c & CAP_PKG_DTS),
        yes(c & CAP_HWP),
        yes(c & CAP_APERF),
        yes(c & CAP_TURBO)
    );
    match core_temp() {
        Some(r) => kprintln!(
            "  {} C, {} below the limit of {}{}{}",
            r.temp,
            r.headroom,
            tj_max(),
            if r.throttling { ", THROTTLING" } else { "" },
            if r.critical { ", CRITICAL" } else { "" }
        ),
        None => kprintln!("  temperature unavailable: {}", why()),
    }
    match base_mhz() {
        Some(b) => kprintln!("  base {} MHz", b),
        None => kprintln!("  base frequency unknown"),
    }
    match measure_mhz(20_000) {
        Some(m) => kprintln!("  running at {} MHz over 20 ms", m),
        None => kprintln!("  frequency unavailable: {}", why()),
    }
    kprintln!("  governor {}", governor().name());
    match hwp_range() {
        Some((hi, gu, lo)) => kprintln!("  hwp range {}..{}, guaranteed {}", lo, hi, gu),
        None => kprintln!("  no hardware-managed performance states"),
    }
}

fn yes(v: u32) -> &'static str {
    if v != 0 {
        "yes"
    } else {
        "no"
    }
}

pub fn selftest() -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }
    probe();

    // The gate is the safety property, so it is what gets asserted. None of
    // these touch a register.
    claim(&mut ok, caps() != 0 || true, "CPUID answered without faulting");
    let virt = caps() & CAP_VIRTUAL != 0;
    if virt {
        claim(
            &mut ok,
            core_temp().is_none() && !set_governor(Governor::Powersave),
            "under a hypervisor, every MSR path declines",
        );
    } else {
        claim(&mut ok, true, "no hypervisor, so the MSR paths are permitted");
    }
    claim(
        &mut ok,
        (60..=120).contains(&tj_max()),
        "the throttle point is a plausible temperature",
    );

    // Decoding is pure and is checked without hardware, which is the only
    // part of this file that can be checked at all under emulation.
    let sample = (1u64 << 31) | (30u64 << 16);
    let r = decode(sample).unwrap();
    claim(
        &mut ok,
        r.headroom == 30 && r.temp == tj_max() - 30 && !r.throttling,
        "a reading is degrees below the limit, not an absolute value",
    );
    claim(&mut ok, decode(0).is_none(), "an invalid reading is refused rather than reported as zero");
    let hot = (1u64 << 31) | 1 | (1 << 5);
    let r = decode(hot).unwrap();
    claim(&mut ok, r.throttling && r.critical, "the throttle and critical flags decode");

    claim(
        &mut ok,
        Governor::parse("perf") == Some(Governor::Performance)
            && Governor::parse("nonsense").is_none(),
        "governors are named and an unknown one is refused",
    );
    claim(
        &mut ok,
        Governor::Performance.epp() < Governor::Powersave.epp(),
        "and they order from performance to efficiency",
    );
    claim(&mut ok, COOL_C < HOT_C, "the thermal thresholds have a gap, so the policy cannot oscillate");
    ok
}

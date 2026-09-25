// Control-register and interrupt-flag helpers land ahead of their users:
// write_cr3 is for paging, the sti/cli pair for the APIC timer in M4.
#![allow(dead_code)]

//! Processor state we own: descriptor tables, control registers, I/O ports.

pub mod code;
pub mod symbols;
pub mod percpu;
pub mod recover;
pub mod gdt;
pub mod idt;
pub mod port;

use core::arch::asm;

#[inline]
pub fn read_cr2() -> u64 {
    let v: u64;
    unsafe { asm!("mov {}, cr2", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

#[inline]
pub fn read_cr3() -> u64 {
    let v: u64;
    unsafe { asm!("mov {}, cr3", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

/// # Safety
/// `phys` must be the physical address of a valid, fully populated PML4.
#[inline]
pub unsafe fn write_cr3(phys: u64) {
    unsafe { asm!("mov cr3, {}", in(reg) phys, options(nostack, preserves_flags)) };
}

#[inline]
pub fn read_cr0() -> u64 {
    let v: u64;
    unsafe { asm!("mov {}, cr0", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

#[inline]
pub fn read_cr4() -> u64 {
    let v: u64;
    unsafe { asm!("mov {}, cr4", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

/// # Safety
/// `msr` must be a model-specific register this CPU implements; reading an
/// unimplemented one raises #GP.
#[inline]
pub unsafe fn rdmsr(msr: u32) -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        asm!("rdmsr", in("ecx") msr, out("eax") lo, out("edx") hi,
             options(nomem, nostack, preserves_flags));
    }
    ((hi as u64) << 32) | (lo as u64)
}

/// # Safety
/// Writing a reserved bit, or a value the CPU rejects, raises #GP.
#[inline]
pub unsafe fn wrmsr(msr: u32, value: u64) {
    unsafe {
        asm!("wrmsr", in("ecx") msr, in("eax") value as u32, in("edx") (value >> 32) as u32,
             options(nomem, nostack, preserves_flags));
    }
}

#[inline]
pub fn disable_interrupts() {
    unsafe { asm!("cli", options(nomem, nostack)) };
}

#[inline]
pub fn enable_interrupts() {
    unsafe { asm!("sti", options(nomem, nostack)) };
}

/// Run `f` with interrupts masked, restoring whatever they were before.
///
/// Restoring rather than unconditionally enabling matters: this gets called
/// from places that are already inside a masked region, and an unconditional
/// `sti` on the way out would quietly re-enable preemption in the middle of
/// someone else's critical section.
pub fn without_interrupts<R>(f: impl FnOnce() -> R) -> R {
    let flags: u64;
    unsafe { asm!("pushfq; pop {}", out(reg) flags, options(preserves_flags)) };
    let was_enabled = flags & (1 << 9) != 0; // RFLAGS.IF
    disable_interrupts();
    let out = f();
    if was_enabled {
        enable_interrupts();
    }
    out
}

/// Raw CPUID.
///
/// The `xchg` dance around `rbx` is not optional: LLVM reserves that register
/// internally, so `out("ebx")` is rejected outright. We stash it, run cpuid,
/// then swap the result out and the original back.
/// # Safety
/// Clearing a bit the kernel depends on -- paging, protected mode -- is the
/// last thing this machine does.
pub unsafe fn write_cr0(v: u64) {
    unsafe { asm!("mov cr0, {}", in(reg) v, options(nostack, preserves_flags)) };
}

/// Drop one page's translation from the TLB.
///
/// # Safety
/// Harmless on any address. Wrong only by omission: a permission change
/// without this leaves the old translation cached and the change silently
/// unenforced for as long as the entry survives.
pub unsafe fn invlpg(at: u64) {
    unsafe { asm!("invlpg [{}]", in(reg) at, options(nostack, preserves_flags)) };
}

/// `CR0.WP`. Whether ring 0 respects the read-only bit in a page table entry.
const CR0_WP: u64 = 1 << 16;
/// `EFER.NXE`. Whether bit 63 of a page table entry means no-execute.
const EFER_NXE: u64 = 1 << 11;
const IA32_EFER: u32 = 0xC000_0080;

/// Make read-only mean read-only, even here.
///
/// **Without `CR0.WP`, a write from ring 0 ignores the R/W bit entirely.**
/// Every instruction in this kernel runs at ring 0, and so does every guest
/// binary, so a page marked read-only without this is a page marked read-only
/// in a comment. Turning it on costs nothing today, because the identity map
/// makes everything writable, and it is what any future read-only page rests
/// on.
///
/// It also catches a class of kernel bug for free, once anything is marked
/// read-only: writing through a stale pointer into constant data becomes a
/// fault at the write instead of a wrong answer somewhere later.
pub fn enable_wp() {
    unsafe { write_cr0(read_cr0() | CR0_WP) };
}

pub fn wp_on() -> bool {
    read_cr0() & CR0_WP != 0
}

/// The machine's physical address width, from `CPUID.80000008H:EAX[7:0]`.
///
/// Asked rather than assumed because it is what decides which bits of a page
/// table entry are *reserved*, and a check against the wrong width either
/// waves corruption through or invents it. Answers 36 when the leaf is absent,
/// which is the architectural minimum and the conservative direction: it
/// reserves more bits rather than fewer.
pub fn phys_addr_bits() -> u32 {
    if cpuid(0x8000_0000, 0)[0] < 0x8000_0008 {
        return 36;
    }
    let bits = cpuid(0x8000_0008, 0)[0] & 0xFF;
    if (36..=52).contains(&bits) { bits } else { 36 }
}

/// Whether this part can map a whole gigabyte with one entry.
/// CPUID.80000001H:EDX[26].
pub fn gib_pages_supported() -> bool {
    cpuid(0x8000_0001, 0)[3] & (1 << 26) != 0
}

/// Whether this part implements no-execute at all. CPUID.80000001H:EDX[20].
pub fn nx_supported() -> bool {
    cpuid(0x8000_0001, 0)[3] & (1 << 20) != 0
}

/// Give this core the same page-rights configuration the bootstrap one has.
///
/// **`CR0.WP` and `EFER.NXE` are per-core, exactly as `CR4` and `XCR0` are,
/// and only the second pair was ever carried over.** The trampoline sets
/// `EFER.LME` to reach long mode and nothing else, so an application processor
/// ran with `NXE` off while the bootstrap processor ran with it on -- and bit
/// 63 of a page table entry is *no-execute* under one and *reserved* under the
/// other. The same table, read by two cores, one of which faults.
///
/// It cost a release gate to find, through two wrong hypotheses. Nothing
/// showed it while the map was built from 2 MiB pages that never carried the
/// bit; `diag paging` is the only thing in the tree that writes `NX` into a
/// 4 KiB entry, and it frees those pages back to the heap, so the next thing
/// to allocate sixteen megabytes and read it from four cores was `diag smp`.
/// The audit found nothing because it runs on the bootstrap processor, where
/// the entry is perfectly legal.
///
/// Answers what it managed, so a caller can report a core that is not the same
/// machine as the others rather than assuming it is.
pub fn adopt_page_rights() -> (bool, bool) {
    enable_wp();
    (wp_on(), enable_nx())
}

/// Make bit 63 of a page table entry mean no-execute.
///
/// Gated on CPUID for the reason `dev::power` gates its MSRs: writing a
/// reserved bit of `EFER` raises #GP, and every vector but `#BP` here is
/// fatal. Answers whether it is on afterwards.
///
/// Safe to turn on at any point, and this is worth stating because it looks
/// like it should not be: enabling `NXE` changes the meaning of bit 63 in
/// every entry that already exists, and nothing in this kernel has ever set
/// it. So the map means exactly what it meant a moment earlier.
pub fn enable_nx() -> bool {
    if !nx_supported() {
        return false;
    }
    unsafe {
        let efer = rdmsr(IA32_EFER);
        wrmsr(IA32_EFER, efer | EFER_NXE);
        rdmsr(IA32_EFER) & EFER_NXE != 0
    }
}

pub fn nx_on() -> bool {
    unsafe { rdmsr(IA32_EFER) & EFER_NXE != 0 }
}

/// Bytes of the largest cache this processor has, or `None` if it will not say.
///
/// **The last-level cache is a resource the miner spends and nothing here
/// could previously measure.** yespower's working set is two to sixteen
/// megabytes by construction, so how many jobs run concurrently before they
/// steal from each other is a property of this number and of nothing else --
/// and `design/mining.md` planned against 12 MB for a year because somebody
/// wrote down the wrong processor. The machine is an i7-12650H with 24 MB, so
/// the budget was half what it should have been.
///
/// Read rather than tabulated, for the reason `mem::fixed` gives about the
/// memory map: a constant here would be a claim about one laptop, asserted in
/// a kernel meant to boot on another.
///
/// CPUID leaf 4 enumerates caches in sub-leaves until it reports type 0. Size
/// is `ways * partitions * line_size * sets`, each field stored one less than
/// its value. The largest is taken rather than the one labelled level 3,
/// because a part with no L3 and a large L2 has a last-level cache all the
/// same and that is the quantity being asked for.
pub fn last_level_cache() -> Option<usize> {
    // Leaf 4 is Intel's. AMD reports the same shape at 0x8000_001D but only
    // when leaf 0x8000_0001 ECX bit 22 says so, and this has no AMD to test
    // against -- so it answers `None` there rather than reading a leaf that
    // may not exist. A refused answer makes the caller fall back to a bound it
    // can defend; a wrong one silently halves or doubles the budget.
    if cpuid(0, 0)[0] < 4 {
        return None;
    }
    let mut largest = 0usize;
    for sub in 0..16 {
        let r = cpuid(4, sub);
        let kind = r[0] & 0x1f;
        if kind == 0 {
            break;
        }
        // 1 data, 2 instruction, 3 unified. An instruction cache is not a
        // place a miner's working set can live.
        if kind == 2 {
            continue;
        }
        let line = (r[1] & 0xfff) as usize + 1;
        let parts = ((r[1] >> 12) & 0x3ff) as usize + 1;
        let ways = ((r[1] >> 22) & 0x3ff) as usize + 1;
        let sets = r[2] as usize + 1;
        let size = line * parts * ways * sets;
        if size > largest {
            largest = size;
        }
    }
    if largest == 0 {
        None
    } else {
        Some(largest)
    }
}

/// What the hypervisor calls itself, or `None` on bare metal.
///
/// **The bit says whether, this says which**, and until now only the bit was
/// read. `dev::power` has consulted CPUID.1:ECX[31] since it was written, which
/// is enough to decline an MSR and not enough to tell a QEMU from a VirtualBox
/// -- so every report from a guest said "hypervisor yes" and left the reader to
/// ask which one, on a project whose whole install story is about to be "boot
/// it in a VM".
///
/// Leaf `0x40000000` is the convention every hypervisor follows: `eax` is the
/// highest leaf in the hypervisor range and `ebx:ecx:edx` are twelve bytes of
/// vendor string. It is **only meaningful when the present bit is set** -- on
/// bare metal `0x40000000` is above the supported range and the processor
/// answers with the highest basic leaf instead, which would read as a vendor
/// string made of whatever that leaf happens to contain.
///
/// The string is what the hypervisor chose to say about itself. It can be
/// configured, and on some it can be hidden entirely, so this is evidence and
/// never proof -- which is exactly the standing this tree already gives the
/// present bit it sits beside.
pub fn hypervisor() -> Option<[u8; 12]> {
    if cpuid(1, 0)[2] & (1 << 31) == 0 {
        return None;
    }
    let r = cpuid(0x4000_0000, 0);
    let mut v = [0u8; 12];
    v[0..4].copy_from_slice(&r[1].to_le_bytes());
    v[4..8].copy_from_slice(&r[2].to_le_bytes());
    v[8..12].copy_from_slice(&r[3].to_le_bytes());
    Some(v)
}

/// The hypervisor's own name, matched against the strings in the field.
///
/// Answers the raw string when nothing matches rather than "unknown", because
/// a twelve-byte name nobody here recognises is the single most useful thing a
/// bug report from an unfamiliar setup can carry.
pub fn hypervisor_name() -> Option<alloc::string::String> {
    use alloc::string::{String, ToString};
    let v = hypervisor()?;
    // Spelled exactly as each one reports it. QEMU answers `TCGTCGTCGTCG` only
    // when it is interpreting; accelerated by KVM it answers `KVMKVMKVM` and
    // accelerated by WHPX it answers as Hyper-V, because in both cases the
    // thing the guest is actually running on is the accelerator rather than
    // QEMU. So this names the *hypervisor* and not the program that launched
    // it, which is the honest answer and is not always the one somebody
    // expects to read.
    let known: &[(&[u8], &str)] = &[
        (b"KVMKVMKVM   ", "KVM (QEMU accelerated)"),
        (b"TCGTCGTCGTCG", "QEMU, interpreting (TCG)"),
        (b"Microsoft Hv", "Hyper-V or WHPX"),
        (b"VMwareVMware", "VMware"),
        (b"VBoxVBoxVBox", "VirtualBox"),
        (b"XenVMMXenVMM", "Xen"),
        (b"prl hyperv  ", "Parallels"),
        (b"bhyve bhyve ", "bhyve"),
        (b"ACRNACRNACRN", "ACRN"),
    ];
    for (sig, name) in known {
        if v.starts_with(sig) || &v[..] == *sig {
            return Some(name.to_string());
        }
    }
    let mut raw = String::new();
    for b in v {
        if b.is_ascii_graphic() || b == b' ' {
            raw.push(b as char);
        }
    }
    Some(raw)
}

/// Whether this core is a performance core or an efficiency one.
///
/// **Asked on the core it describes, and that is the whole difficulty.** Leaf
/// 0x1A reports the type of the core *executing* it, so a single call on the
/// bootstrap processor answers for one core out of sixteen and says nothing about
/// the rest. `smp` records it per core during bring-up for that reason.
///
/// Gated twice before the read. The leaf has to exist -- `cpuid(0,0).eax` is the
/// highest basic leaf, and asking for one past it returns the highest leaf's data
/// rather than zero, which would decode as a plausible core type. And
/// `CPUID.07H:EDX[15]` is the hybrid bit: a part that is not hybrid has no 0x1A
/// to report and every core on it is the same kind, so `Unknown` there is the
/// truthful answer rather than a failure.
///
/// Validated against the host's own topology before being trusted: on this
/// i7-12650H, cpus 0-11 read 0x40 and sit on six cores of two threads at 2300
/// MHz, and cpus 12-15 read 0x20 and sit on four single-threaded cores at 1700.
/// The decode agrees with `lscpu` on all sixteen.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CoreKind {
    /// Intel calls this "Core". Wide, SMT, the high clock.
    Performance,
    /// Intel calls this "Atom". Narrower, no SMT, and on this part a 1700 MHz
    /// ceiling against a P-core's 2300 -- so roughly half the hash rate, which is
    /// the bimodality `glados-pool --bench` shows when it is left unpinned.
    Efficiency,
    /// Not a hybrid part, or too old to say. Every core is then the same kind and
    /// there is nothing to prefer.
    Unknown,
}

/// How many low bits of an APIC id index SMT threads within one physical core.
///
/// **Two logical cores on one physical core are not two cores**, and placing a
/// second slice on a sibling is worth far less than placing it on an idle core of
/// any kind. Measured, four slices under a pinned QEMU: four distinct physical
/// performance cores read 700 H/s, the same four logical cores with one sibling
/// collision read 605, and four distinct efficiency cores read 535. So a
/// collision costs 1.16x and an efficiency core beats a sibling outright -- a
/// sibling adds about 21% of one slice where an efficiency core adds a whole 71%.
///
/// Leaf 0x0B subleaf 0 reports the shift: `eax[4:0]` is the number of APIC id bits
/// below the core level, so a part with two threads per core answers 1 and the
/// thread index is `apic_id & 1`. Zero means no SMT, or a part too old to say,
/// and then every logical core is its own physical one.
pub fn smt_shift() -> u32 {
    if cpuid(0, 0)[0] < 0x0B {
        return 0;
    }
    // Level type 1 is SMT. A part with no SMT reports level 0 here, and
    // `eax[4:0]` is then zero anyway, so the shift is the whole answer.
    cpuid(0x0B, 0)[0] & 0x1F
}

pub fn core_kind() -> CoreKind {
    if cpuid(0, 0)[0] < 0x1A {
        return CoreKind::Unknown;
    }
    if cpuid(7, 0)[3] & (1 << 15) == 0 {
        return CoreKind::Unknown;
    }
    match cpuid(0x1A, 0)[0] >> 24 {
        0x40 => CoreKind::Performance,
        0x20 => CoreKind::Efficiency,
        _ => CoreKind::Unknown,
    }
}

pub fn cpuid(leaf: u32, sub: u32) -> [u32; 4] {
    let eax: u32;
    let ebx_slot: u64;
    let ecx: u32;
    let edx: u32;
    unsafe {
        asm!(
            "mov {tmp}, rbx",
            "cpuid",
            "xchg {tmp}, rbx",
            tmp = out(reg) ebx_slot,
            inout("eax") leaf => eax,
            inout("ecx") sub => ecx,
            out("edx") edx,
            options(nostack, preserves_flags),
        );
    }
    [eax, ebx_slot as u32, ecx, edx]
}

/// Stand between writing bytes and fetching them as instructions.
///
/// Nothing in this tree serialised before this: no `wbinvd`, no `clflush`, no
/// `mfence`, no `cpuid`-as-barrier. `CPUID` is the serialising instruction
/// that needs no privilege and no feature test, and it does both halves of
/// the job -- the processor discards what it prefetched, and because the
/// `asm!` above declares neither `nomem` nor `readonly`, the compiler must
/// treat it as touching memory and cannot sink the stores that filled the
/// buffer past it.
///
/// Leaf 0 because its result is discarded; what is wanted is the barrier.
#[inline]
pub fn serialize() {
    let _ = cpuid(0, 0);
}

/// Cached so the shell can report what was actually enabled at boot, not just
/// what CPUID advertises.
static FEATURES: crate::sync::Racy<Features> = crate::sync::Racy::new(Features::none());

pub fn detected() -> Features {
    unsafe { *FEATURES.get() }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Features {
    pub sse: bool,
    pub sse2: bool,
    pub sse41: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub avx512f: bool,
    pub xsave: bool,
    /// True once the OS has actually enabled the state, not merely detected it.
    pub avx_enabled: bool,
}

impl Features {
    const fn none() -> Self {
        Self {
            sse: false,
            sse2: false,
            sse41: false,
            avx: false,
            avx2: false,
            fma: false,
            f16c: false,
            avx512f: false,
            xsave: false,
            avx_enabled: false,
        }
    }
}

/// What CPUID advertises, and **not** what the OS has turned on.
///
/// `avx_enabled` is hardcoded false here because this function does not read
/// XCR0 -- it cannot, it is a pure CPUID query. The value that reflects the
/// OSXSAVE/XCR0 handshake is the one `enable_simd` caches, and it is read back
/// through [`detected`].
///
/// Ask for `detected()` unless you specifically want the raw advertisement.
/// A diagnostic built on this one reported "avx enabled=false" on a machine
/// where every AVX kernel was running perfectly well, which was very nearly
/// acted on.
pub fn features() -> Features {
    let f1 = cpuid(1, 0);
    let f7 = cpuid(7, 0);
    Features {
        sse: f1[3] & (1 << 25) != 0,
        sse2: f1[3] & (1 << 26) != 0,
        sse41: f1[2] & (1 << 19) != 0,
        avx: f1[2] & (1 << 28) != 0,
        fma: f1[2] & (1 << 12) != 0,
        f16c: f1[2] & (1 << 29) != 0,
        xsave: f1[2] & (1 << 26) != 0,
        avx2: f7[1] & (1 << 5) != 0,
        avx512f: f7[1] & (1 << 16) != 0,
        avx_enabled: false,
    }
}

#[inline]
unsafe fn read_cr4_raw() -> u64 {
    read_cr4()
}

#[inline]
unsafe fn write_cr4(value: u64) {
    unsafe { asm!("mov cr4, {}", in(reg) value, options(nostack, preserves_flags)) };
}

#[inline]
unsafe fn xsetbv(index: u32, value: u64) {
    unsafe {
        asm!("xsetbv",
             in("ecx") index,
             in("eax") value as u32,
             in("edx") (value >> 32) as u32,
             options(nostack, preserves_flags));
    }
}

/// Turn on SSE and, if the CPU has it, AVX.
///
/// Detection is not enough. The CPU refuses to execute AVX instructions until
/// the OS declares it will save the wider register state: that means setting
/// `CR4.OSXSAVE`, then setting the x87, SSE and AVX bits in `XCR0` via
/// `xsetbv`. Skip it and every `vmulps` raises #UD, which looks like the
/// compiler emitting garbage rather than like a missing OS handshake.
///
/// UEFI leaves SSE enabled -- the x86_64 UEFI ABI requires it -- but we set
/// the bits regardless rather than inherit an assumption.
/// CPUID.01H:ECX bit 3 -- MONITOR/MWAIT.
///
/// Computed from CPUID rather than read out of the `FEATURES` static, so an
/// application processor can ask without touching kernel state.
pub fn has_monitor() -> bool {
    cpuid(1, 0)[2] & (1 << 3) != 0
}

pub fn enable_simd() -> Features {
    let f = enable_simd_this_core();
    unsafe { *FEATURES.get() = f };
    f
}

/// The register half of `enable_simd`, touching nothing shared.
///
/// CR4, XCR0 and MXCSR are per-core: a core that skips this handshake takes
/// #UD on the first `vmulps` no matter what another core has already enabled.
/// So every application processor must call this for itself -- and must call
/// *this* rather than `enable_simd`, which additionally publishes to the
/// `FEATURES` static. The APs are a compute fabric that writes no kernel
/// state, and one benign-looking store is how that stops being true.
pub fn enable_simd_this_core() -> Features {
    let mut f = features();
    unsafe {
        // CR4.OSFXSR (bit 9): fxsave/fxrstor, and enables SSE.
        // CR4.OSXMMEXCPT (bit 10): unmasked SIMD FP exceptions go to #XM.
        let mut cr4 = read_cr4_raw() | (1 << 9) | (1 << 10);

        if f.xsave && f.avx {
            cr4 |= 1 << 18; // CR4.OSXSAVE
            write_cr4(cr4);
            // XCR0: bit 0 x87 (mandatory), bit 1 SSE, bit 2 AVX (ymm high halves).
            xsetbv(0, 0b111);
            f.avx_enabled = true;
        } else {
            write_cr4(cr4);
        }

        // And pin MXCSR, which was the one SIMD register still inherited.
        // 0x1F80 masks all six x87/SSE exceptions; int8 dequantisation
        // produces subnormal products freely, so a vCPU handed over with
        // those unmasked takes #XM on the first vmulps. That is exactly what
        // QEMU's WHPX accelerator does, while TCG and bare-metal firmware
        // both happen to mask them -- the crash therefore looked like an
        // accelerator bug when it was an assumption of ours. This function's
        // own rule is set the bits regardless, and now it applies here too.
        let mxcsr: u32 = 0x1F80;
        core::arch::asm!(
            "ldmxcsr [{}]",
            in(reg) &mxcsr,
            options(nostack, preserves_flags)
        );

    }
    f
}

/// Bytes needed to hold this CPU's extended state.
///
/// Queried, never hardcoded. `XSAVE` writes as much as `XCR0` enables, so a
/// buffer sized for `fxsave` (512 B) overflows by ~320 bytes the moment AVX is
/// on -- straight into whatever the heap placed next. That corruption surfaces
/// far from its cause, which is the worst possible property for a bug in the
/// scheduler.
///
/// CPUID.0DH:ECX reports the maximum for every feature the CPU supports,
/// which is an upper bound on what our XCR0 can ever ask for.
pub fn xsave_area_size() -> usize {
    let f = detected();
    if !f.avx_enabled {
        return 512; // fxsave region
    }
    let r = cpuid(0x0D, 0);
    let max = r[2] as usize;
    // Floor at 1 KiB: a CPU reporting something implausibly small should not
    // be able to talk us into a too-small buffer.
    if max < 1024 {
        1024
    } else {
        max
    }
}

/// The state components we manage: x87, SSE, and AVX's upper halves.
const XSTATE_MASK: u32 = 0b111;

/// # Safety
/// `area` must be writable, at least `xsave_area_size()` bytes, and 64-byte
/// aligned.
pub unsafe fn xsave_to(area: *mut u8) {
    unsafe {
        if detected().avx_enabled {
            asm!("xsave [{}]", in(reg) area, in("eax") XSTATE_MASK, in("edx") 0u32, options(nostack));
        } else {
            asm!("fxsave [{}]", in(reg) area, options(nostack));
        }
    }
}

/// # Safety
/// `area` must hold a state image previously written by `xsave_to`, or be
/// zeroed. A zeroed image has `XSTATE_BV = 0`, which `XRSTOR` reads as
/// "set every component to its initial state" -- exactly what a new task
/// wants. Garbage in the header raises #GP instead.
pub unsafe fn xrstor_from(area: *const u8) {
    unsafe {
        if detected().avx_enabled {
            asm!("xrstor [{}]", in(reg) area, in("eax") XSTATE_MASK, in("edx") 0u32, options(nostack));
        } else {
            asm!("fxrstor [{}]", in(reg) area, options(nostack));
        }
    }
}

/// Reset the machine.
///
/// Tries the keyboard controller's reset line first, which is the historical
/// and most widely implemented method, then falls back to deliberately
/// triple-faulting by loading a zero-length IDT and raising an interrupt. The
/// CPU cannot find a handler, cannot find a double-fault handler either, and
/// resets. Inelegant, universally effective.
/// The firmware's runtime table, kept from boot.
///
/// `ExitBootServices` retires the boot services and leaves these alone, and
/// nothing here calls `SetVirtualAddressMap`, so the pointer stays valid and
/// the identity map keeps it reachable. It is the only way back to firmware
/// once the kernel owns the machine.
static RUNTIME: crate::sync::Racy<usize> = crate::sync::Racy::new(0);

pub fn set_runtime(rt: *mut core::ffi::c_void) {
    unsafe { *RUNTIME.get() = rt as usize };
}

fn runtime() -> Option<&'static crate::uefi::RuntimeServices> {
    let p = unsafe { *RUNTIME.get() };
    if p == 0 {
        return None;
    }
    Some(unsafe { &*(p as *const crate::uefi::RuntimeServices) })
}

/// Turn the machine off.
///
/// The firmware first, because it knows this board and a call through it is
/// the path every other operating system takes. **And ACPI second, which it
/// could not be before.** This comment used to say that doing it by hand meant
/// parsing the DSDT for `\_S5` and writing PM1a and PM1b, an AML
/// interpreter's worth of work for something the firmware already does -- and
/// that was a fair trade right up until the interpreter existed for the
/// battery. It does now, so the second chance costs a function call, and
/// "hold the button" stops being the answer when the firmware declines.
///
/// Returns only if both refuse, which is why the caller still parks the core.
pub fn shutdown() -> ! {
    if let Some(rt) = runtime() {
        (rt.reset_system)(
            crate::uefi::ResetType::Shutdown,
            0,
            0,
            core::ptr::null(),
        );
    }
    crate::kprintln!("  the firmware declined; asking ACPI directly");
    if let Some(a) = crate::acpi::parsed() {
        match crate::acpi::power_off(&a) {
            Ok(()) => {}
            Err(e) => crate::kprintln!("  {}", e),
        }
    }
    crate::kprintln!("  nothing would power this machine down; hold the button");
    halt()
}

pub fn reboot() -> ! {
    // The firmware first. It knows this board's quirks, and a cold reset
    // through it is the same path every other operating system on the machine
    // takes. The keyboard-controller pulse and the triple fault below are
    // what to do when there is no firmware left to ask, and they stay because
    // the recovery console may need them when nothing else is standing.
    if let Some(rt) = runtime() {
        (rt.reset_system)(
            crate::uefi::ResetType::Cold,
            0,
            0,
            core::ptr::null(),
        );
    }
    reboot_the_hard_way()
}

fn reboot_the_hard_way() -> ! {
    unsafe {
        for _ in 0..16 {
            let mut spins = 0;
            while port::inb(0x64) & 0x02 != 0 {
                spins += 1;
                if spins > 100_000 {
                    break;
                }
            }
            port::outb(0x64, 0xFE);
        }

        let null_idt = gdt::DescriptorTablePointer { limit: 0, base: 0 };
        asm!("lidt [{}]", in(reg) &null_idt, options(readonly, nostack));
        asm!("int3", options(nomem, nostack));
    }
    halt()
}

/// Park the core. `cli` before `hlt` so no interrupt can wake us into a
/// half-initialised state.
pub fn halt() -> ! {
    loop {
        unsafe { asm!("cli; hlt", options(nomem, nostack)) };
    }
}

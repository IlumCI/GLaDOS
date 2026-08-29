// Control-register and interrupt-flag helpers land ahead of their users:
// write_cr3 is for paging, the sti/cli pair for the APIC timer in M4.
#![allow(dead_code)]

//! Processor state we own: descriptor tables, control registers, I/O ports.

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

/// Ask the firmware to turn the machine off.
///
/// There was no way to do this at all, which is why the only shutdown anybody
/// had was holding the power button. ACPI S5 by hand means parsing the DSDT
/// for `\_S5` and writing PM1a/PM1b, an AML interpreter's worth of work for
/// something the firmware already knows how to do on this exact board.
///
/// Returns only if the call is unavailable or the firmware declines, which is
/// why the caller still has to park the core afterwards.
pub fn shutdown() -> ! {
    if let Some(rt) = runtime() {
        (rt.reset_system)(
            crate::uefi::ResetType::Shutdown,
            0,
            0,
            core::ptr::null(),
        );
    }
    crate::kprintln!("  the firmware would not power down; hold the button");
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

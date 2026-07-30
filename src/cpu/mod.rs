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

/// Reset the machine.
///
/// Tries the keyboard controller's reset line first, which is the historical
/// and most widely implemented method, then falls back to deliberately
/// triple-faulting by loading a zero-length IDT and raising an interrupt. The
/// CPU cannot find a handler, cannot find a double-fault handler either, and
/// resets. Inelegant, universally effective.
pub fn reboot() -> ! {
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

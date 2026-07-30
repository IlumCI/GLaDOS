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

#[inline]
pub fn disable_interrupts() {
    unsafe { asm!("cli", options(nomem, nostack)) };
}

#[inline]
pub fn enable_interrupts() {
    unsafe { asm!("sti", options(nomem, nostack)) };
}

/// Park the core. `cli` before `hlt` so no interrupt can wake us into a
/// half-initialised state.
pub fn halt() -> ! {
    loop {
        unsafe { asm!("cli; hlt", options(nomem, nostack)) };
    }
}

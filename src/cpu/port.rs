// A complete device-access surface; the i8042 driver in M4 is the first real
// consumer of most of it.
#![allow(dead_code)]

//! x86 port I/O.
//!
//! Reads from an unimplemented port return 0xFF and writes are discarded --
//! no fault. That is what makes the COM1 driver safe to run on this laptop,
//! which has no UART behind 0x3F8.

use core::arch::asm;

#[inline]
pub unsafe fn outb(port: u16, val: u8) {
    unsafe { asm!("out dx, al", in("dx") port, in("al") val, options(nomem, nostack, preserves_flags)) };
}

#[inline]
pub unsafe fn inb(port: u16) -> u8 {
    let val: u8;
    unsafe { asm!("in al, dx", out("al") val, in("dx") port, options(nomem, nostack, preserves_flags)) };
    val
}

#[inline]
pub unsafe fn inl(port: u16) -> u32 {
    let val: u32;
    unsafe { asm!("in eax, dx", out("eax") val, in("dx") port, options(nomem, nostack, preserves_flags)) };
    val
}

#[inline]
pub unsafe fn outl(port: u16, val: u32) {
    unsafe { asm!("out dx, eax", in("dx") port, in("eax") val, options(nomem, nostack, preserves_flags)) };
}

#[inline]
pub unsafe fn outw(port: u16, val: u16) {
    unsafe { asm!("out dx, ax", in("dx") port, in("ax") val, options(nomem, nostack, preserves_flags)) };
}

#[inline]
pub unsafe fn inw(port: u16) -> u16 {
    let val: u16;
    unsafe { asm!("in ax, dx", out("ax") val, in("dx") port, options(nomem, nostack, preserves_flags)) };
    val
}

/// Short delay by touching an unused port. Some legacy controllers (the i8042
/// in particular) need settling time between accesses.
#[inline]
pub unsafe fn io_wait() {
    unsafe { outb(0x80, 0) };
}

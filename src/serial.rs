//! COM1 (0x3F8) 16550 UART driver.
//!
//! The GF63 has no physical serial port, so on bare metal these writes go to an
//! unimplemented I/O port and are harmlessly discarded -- x86 does not fault on
//! writes to absent ports. Under QEMU (`-serial stdio`) this is a real console,
//! and it is by far the most convenient debug channel we have during M0-M3,
//! because it works before a single pixel has been drawn.
//!
//! Do not come to depend on it. On the laptop the framebuffer is the only way
//! anything reaches your eyes, which is why M2 exists.

use core::arch::asm;
use core::fmt;
use core::sync::atomic::{AtomicBool, Ordering};

const COM1: u16 = 0x3F8;

#[inline]
unsafe fn outb(port: u16, val: u8) {
    unsafe {
        asm!("out dx, al", in("dx") port, in("al") val, options(nomem, nostack, preserves_flags));
    }
}

#[inline]
unsafe fn inb(port: u16) -> u8 {
    let val: u8;
    unsafe {
        asm!("in al, dx", out("al") val, in("dx") port, options(nomem, nostack, preserves_flags));
    }
    val
}

/// Set once `init` has confirmed a UART actually answers at 0x3F8.
///
/// Writes to an absent port are harmless, so output never needed this. Input
/// does, and urgently: reads from an absent port return 0xFF, so the "data
/// ready" bit of a UART that is not there is permanently set, and a polled
/// reader would take 0xFF for a keystroke forever. On the GF63 -- which has no
/// serial port -- that would wedge the shell.
static PRESENT: AtomicBool = AtomicBool::new(false);

/// Configure COM1 for 115200 8N1.
pub fn init() {
    unsafe {
        outb(COM1 + 1, 0x00); // interrupts off -- we poll
        outb(COM1 + 3, 0x80); // DLAB on, so 0/1 become the divisor latch
        outb(COM1 + 0, 0x01); // divisor low  = 1 -> 115200 baud
        outb(COM1 + 1, 0x00); // divisor high = 0
        outb(COM1 + 3, 0x03); // DLAB off, 8 bits, no parity, 1 stop
        outb(COM1 + 2, 0xC7); // enable + clear FIFOs, 14-byte trigger
        outb(COM1 + 4, 0x0B); // DTR, RTS, OUT2

        // Presence check via the scratch register, which exists on a 16550 and
        // nowhere on an empty bus. An absent port reads back 0xFF.
        outb(COM1 + 7, 0xA5);
        let present = inb(COM1 + 7) == 0xA5;
        outb(COM1 + 7, 0x00);
        PRESENT.store(present, Ordering::Relaxed);
    }
}

pub fn is_present() -> bool {
    PRESENT.load(Ordering::Relaxed)
}

/// Non-blocking read of one byte, or `None` if nothing is waiting.
///
/// This makes the serial port an input as well as an output, which is what
/// lets the whole system be driven headlessly under QEMU -- no framebuffer, no
/// emulated keystrokes, just a pipe. Everything the shell can do becomes
/// scriptable and therefore testable.
pub fn read_byte() -> Option<u8> {
    if !is_present() {
        return None;
    }
    unsafe {
        // Line Status Register bit 0: data ready.
        if inb(COM1 + 5) & 0x01 == 0 {
            return None;
        }
        Some(inb(COM1))
    }
}

#[inline]
fn tx_empty() -> bool {
    // Line Status Register bit 5: transmitter holding register empty.
    unsafe { inb(COM1 + 5) & 0x20 != 0 }
}

pub fn write_byte(b: u8) {
    // On real hardware with no UART present, inb returns 0xFF, so bit 5 reads
    // set and we never spin. That is exactly the behaviour we want.
    let mut spins = 0u32;
    while !tx_empty() {
        spins += 1;
        if spins > 100_000 {
            return; // Something is wrong with the port; never hang the kernel on it.
        }
        core::hint::spin_loop();
    }
    unsafe { outb(COM1, b) }
}

pub struct Serial;

impl fmt::Write for Serial {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for b in s.bytes() {
            if b == b'\n' {
                write_byte(b'\r');
            }
            write_byte(b);
        }
        Ok(())
    }
}

#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    use fmt::Write;
    let _ = Serial.write_fmt(args);
}

#[macro_export]
macro_rules! serial_print {
    ($($arg:tt)*) => { $crate::serial::_print(format_args!($($arg)*)) };
}

#[macro_export]
macro_rules! serial_println {
    () => { $crate::serial_print!("\n") };
    ($($arg:tt)*) => { $crate::serial_print!("{}\n", format_args!($($arg)*)) };
}

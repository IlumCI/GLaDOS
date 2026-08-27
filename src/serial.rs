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
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use crate::sync::Racy;

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
/// Bytes the handler has taken off the port and nothing has read yet.
///
/// The port holds one byte with the FIFO off, which is the only configuration
/// that works here (see `init`). Under QEMU that is survivable, because its
/// 16550 will not hand over a byte until the guest has taken the last one --
/// the emulator flow-controls us and an overrun cannot happen. Real hardware
/// makes no such promise. A UART with a byte in the holding register and
/// another arriving on the wire drops one, and this kernel can leave the shell
/// unscheduled for a 10ms slice, which at 115200 is 114 bytes with nowhere to
/// go.
///
/// So the ring is not for QEMU. It is for the machine this is actually for,
/// where nothing is going to wait politely while a model finishes a forward
/// pass.
const RING: usize = 512;

struct Rx {
    buf: [u8; RING],
    head: usize,
    tail: usize,
    /// Bytes the hardware says it lost, from the overrun bit in LSR.
    ///
    /// Counted and reportable rather than discarded. This class of failure is
    /// invisible by nature -- a byte that never arrives leaves no trace --
    /// and the previous input bug survived for months partly because nothing
    /// anywhere said a number out loud.
    overruns: u32,
    /// Bytes dropped because this ring was full, which is our fault and not
    /// the hardware's, and worth telling apart from an overrun.
    spills: u32,
}

static RX: Racy<Rx> =
    Racy::new(Rx { buf: [0; RING], head: 0, tail: 0, overruns: 0, spills: 0 });

/// Times the handler has run. The one number that says whether any of this is
/// working; without it, a silent port and a port whose interrupt never fires
/// look identical, which is exactly the hole the last attempt fell into.
static IRQS: AtomicU32 = AtomicU32::new(0);
static IRQ_LIVE: AtomicBool = AtomicBool::new(false);

fn rx_push(b: u8) {
    let r = unsafe { &mut *RX.get() };
    let next = (r.head + 1) % RING;
    if next == r.tail {
        r.spills = r.spills.saturating_add(1);
        return;
    }
    r.buf[r.head] = b;
    r.head = next;
}

fn rx_pop() -> Option<u8> {
    let r = unsafe { &mut *RX.get() };
    if r.head == r.tail {
        return None;
    }
    let b = r.buf[r.tail];
    r.tail = (r.tail + 1) % RING;
    Some(b)
}

/// Waiting bytes, handler runs, hardware overruns, ring spills.
pub fn rx_stats() -> (usize, u32, u32, u32) {
    crate::cpu::without_interrupts(|| {
        let r = unsafe { &*RX.get() };
        let held = (r.head + RING - r.tail) % RING;
        (held, IRQS.load(Ordering::Relaxed), r.overruns, r.spills)
    })
}

pub fn irq_live() -> bool {
    IRQ_LIVE.load(Ordering::Acquire)
}

/// Take everything the port has, into the ring.
///
/// Loops rather than taking one byte. With the FIFO off there is only ever
/// one, but the loop costs a register read and means this is still correct if
/// the FIFO is ever turned on -- and leaving a byte behind is how an
/// edge-triggered line stops asserting, which the keyboard driver documents
/// one file over at length.
fn drain() {
    unsafe {
        loop {
            let lsr = inb(COM1 + 5);
            if lsr & 0x02 != 0 {
                let r = &mut *RX.get();
                r.overruns = r.overruns.saturating_add(1);
            }
            if lsr & 0x01 == 0 {
                return;
            }
            let b = inb(COM1);
            rx_push(b);
        }
    }
}

extern "x86-interrupt" fn serial_isr(_frame: crate::cpu::idt::InterruptStackFrame) {
    IRQS.fetch_add(1, Ordering::Relaxed);
    drain();
    crate::dev::lapic::eoi();
}

/// Route COM1's interrupt, once ACPI can say where it goes.
///
/// Separate from `init` because `init` runs long before there is an ACPI table
/// to ask, and the port has to carry the whole boot log regardless. Until this
/// runs, and if it fails, reading falls back to polling exactly as before.
pub fn attach_irq(acpi: &crate::acpi::Acpi, apic_id: u8) -> Option<u32> {
    if !is_present() {
        return None;
    }
    let (gsi, flags) = acpi.gsi_for_irq(4);
    let io = acpi.primary_ioapic()?;
    unsafe {
        crate::cpu::idt::set_handler(crate::dev::VECTOR_SERIAL, serial_isr as *const (), 0)
    };
    if !crate::dev::ioapic::route(&io, gsi, crate::dev::VECTOR_SERIAL, apic_id, flags) {
        return None;
    }

    unsafe {
        // The FIFO stays off, and this is not an oversight.
        //
        // Turning it on with a one-byte trigger reads as obviously right --
        // sixteen bytes of buffering, data-ready still immediate -- and it
        // stopped the port receiving anything at all, by interrupt or by
        // poll, with no characters even echoing. The observation `init`
        // already records, that QEMU holds bytes in the FIFO without raising
        // data-ready for a reader shaped like this one, survived the
        // experiment meant to overturn it. Buffering comes from the ring
        // above instead, which is ours and behaves.
        outb(COM1 + 2, 0x00);
        // Received Data Available only. Transmit-holding-empty would fire
        // continuously against a driver that writes by polling, which is what
        // `write_byte` does and should keep doing -- boot output must work
        // before any of this exists.
        outb(COM1 + 1, 0x01);
    }
    IRQ_LIVE.store(true, Ordering::Release);

    // Drain after the line is live, never before. A byte that arrived while
    // the redirection entry was masked leaves no edge behind, and with an
    // edge-triggered source that byte is the last one ever delivered. The
    // keyboard driver lost its first keystroke to precisely this.
    crate::cpu::without_interrupts(drain);
    Some(gsi)
}

pub fn init() {
    unsafe {
        outb(COM1 + 1, 0x00); // interrupts off until attach_irq
        outb(COM1 + 3, 0x80); // DLAB on, so 0/1 become the divisor latch
        outb(COM1 + 0, 0x01); // divisor low  = 1 -> 115200 baud
        outb(COM1 + 1, 0x00); // divisor high = 0
        outb(COM1 + 3, 0x03); // DLAB off, 8 bits, no parity, 1 stop
        // FIFOs OFF. The 14-byte trigger was meant for interrupt-driven
        // guests; this kernel polls, and QEMU's serial model was observed
        // holding received bytes in the FIFO without ever setting LSR bit 0
        // for a polling reader -- the firmware shell (interrupt-driven)
        // received fine, our polled reader did not, same QEMU same wire.
        // Without the FIFO every byte lands in the holding register and
        // LSR bit 0 sets immediately, which is the contract a polling
        // driver actually needs.
        outb(COM1 + 2, 0x00);
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
    // Interrupts off across both halves so the handler cannot be taking the
    // same byte concurrently. That race is the reason these are not two
    // independent checks.
    crate::cpu::without_interrupts(|| {
        if let Some(b) = rx_pop() {
            return Some(b);
        }
        // The poll stays, and stays even once the interrupt is routed. When
        // the handler is working this finds nothing, because it has already
        // taken everything. When the interrupt is routed and never fires --
        // which is a real possibility on a machine nobody here can test --
        // this is the difference between a working console and a dead one.
        // Removing it, on the reasoning that the interrupt now owned the
        // port, is what made the port stop working the first time this was
        // attempted.
        unsafe {
            // Line Status Register bit 0: data ready.
            if inb(COM1 + 5) & 0x01 == 0 {
                return None;
            }
            Some(inb(COM1))
        }
    })
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
    // A captured command's output belongs to whatever asked for it, not to the
    // log. Suppressing only the console left `tree | head 6` printing the whole
    // tree to serial and then the six lines -- the filtering worked and the
    // transcript showed no sign of it.
    if crate::gfx::console::capturing() {
        return;
    }
    let _ = Serial.write_fmt(args);
}

#[macro_export]
macro_rules! serial_print {
    ($($arg:tt)*) => {{
        $crate::serial::_print(format_args!($($arg)*));
        // Into the boot log as well, because everything printed before
        // `console::init` can only come out here.
        //
        // The model, the tokenizer and the root bundle are read at line 206 of
        // main.rs and the console is initialised at 295: they have to be,
        // since a filesystem exists only before ExitBootServices. So every
        // diagnostic about why a model failed to load is emitted 89 lines
        // before there is a screen to emit it to, and on a machine with
        // nothing listening on the serial port it is emitted into nowhere.
        // A firmware that could not find a contiguous pool for a 1.8 GB model
        // says so precisely, and the operator sees a system that boots and
        // reports no model, with the reason discarded.
        //
        // The ring is a static array and needs no initialisation, so it can
        // take these from the first instruction of efi_main. `log all` after
        // boot now includes the part of boot that happens before the screen.
        $crate::log::_record(format_args!($($arg)*));
    }};
}

#[macro_export]
macro_rules! serial_println {
    () => { $crate::serial_print!("\n") };
    ($($arg:tt)*) => { $crate::serial_print!("{}\n", format_args!($($arg)*)) };
}

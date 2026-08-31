//! The Embedded Controller: the small processor that knows about the battery.
//!
//! On a laptop the interesting analogue facts -- charge, temperature, fan
//! speed, lid, the state of the power button -- are not on any bus the main
//! processor can address. They live in a separate microcontroller, and ACPI
//! reaches it through a two-port protocol so simple it predates most of the
//! rest of the standard: a command port, a data port, and two status bits.
//!
//! **A `_BST` method is mostly a sequence of reads through here.** That is why
//! this exists: the battery methods in the firmware read fields declared over
//! an `EmbeddedControl` operation region, and without this they can be parsed,
//! entered and understood right up to the point where they need a number.
//!
//! ### None of this can be tested under emulation
//!
//! QEMU models no embedded controller. Every read here will time out and
//! answer `None` under QEMU, which is correct behaviour and proves only that
//! the timeout works. The protocol is small and published, so writing it from
//! the specification is reasonable; it is still code whose working path has
//! never run, and it is in the same position as the RTL8188EU driver. Said
//! here rather than discovered later.
//!
//! ### Why the waits are bounded
//!
//! On a machine with no controller behind these ports, `inb` answers 0xFF, so
//! both status bits read as set and a naive wait for "input buffer empty"
//! never finishes. That is not a hypothetical: it is exactly what happens
//! under QEMU. Every wait here gives up, in the same shape `serial::write_byte`
//! uses for the same reason, because a driver that hangs the machine when its
//! hardware is absent is worse than one that reports nothing.

use crate::cpu::port::{inb, outb};

/// The two ports ACPI fixes for the embedded controller. A machine may declare
/// different ones in its `_CRS`, and none in production ever has.
const DATA: u16 = 0x62;
const CMD: u16 = 0x66;

/// Status register bits, read from the command port.
const OBF: u8 = 1 << 0; // output buffer full: a byte is waiting for us
const IBF: u8 = 1 << 1; // input buffer full: it has not taken ours yet

const RD_EC: u8 = 0x80;
const WR_EC: u8 = 0x81;

/// How long to wait for the controller, in microseconds.
///
/// The specification allows one millisecond per transaction and real
/// controllers answer in tens of microseconds. Ten milliseconds is generous
/// against a slow one and short enough that a machine with no controller at
/// all spends a tenth of a second discovering that across a whole battery
/// read rather than stopping.
const PATIENCE_US: u64 = 10_000;

static PRESENT: crate::sync::Racy<bool> = crate::sync::Racy::new(false);
static READS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static TIMEOUTS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

fn status() -> u8 {
    unsafe { inb(CMD) }
}

/// Wait for the controller to take what we last wrote.
fn wait_input_taken() -> bool {
    for _ in 0..PATIENCE_US {
        if status() & IBF == 0 {
            return true;
        }
        crate::time::delay_us(1);
    }
    TIMEOUTS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    false
}

/// Wait for a byte to be waiting for us.
fn wait_output_ready() -> bool {
    for _ in 0..PATIENCE_US {
        if status() & OBF != 0 {
            return true;
        }
        crate::time::delay_us(1);
    }
    TIMEOUTS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    false
}

/// Read one byte of the controller's address space.
///
/// The whole protocol: send the read command, send the address, take the
/// answer. Each step waits for the controller to be ready for it, and any
/// wait that gives up abandons the transaction rather than pressing on with a
/// byte that would be somebody else's answer.
pub fn read(addr: u8) -> Option<u8> {
    if !wait_input_taken() {
        return None;
    }
    unsafe { outb(CMD, RD_EC) };
    if !wait_input_taken() {
        return None;
    }
    unsafe { outb(DATA, addr) };
    if !wait_output_ready() {
        return None;
    }
    let v = unsafe { inb(DATA) };
    READS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    unsafe { *PRESENT.get() = true };
    Some(v)
}

/// Write one byte of the controller's address space.
///
/// Separate from `read` and gated above this level rather than here. Writing
/// to an embedded controller is how a laptop changes its charge threshold, its
/// fan curve and its keyboard backlight, and also how it can be made to do
/// something it should not. Nothing in this kernel calls it yet.
pub fn write(addr: u8, value: u8) -> bool {
    if !wait_input_taken() {
        return false;
    }
    unsafe { outb(CMD, WR_EC) };
    if !wait_input_taken() {
        return false;
    }
    unsafe { outb(DATA, addr) };
    if !wait_input_taken() {
        return false;
    }
    unsafe { outb(DATA, value) };
    wait_input_taken()
}

/// Whether a read has ever succeeded.
///
/// Discovered rather than declared: the namespace says a machine has an
/// `EmbeddedControl` region, and whether anything answers at the ports is a
/// different question that only a transaction settles.
pub fn present() -> bool {
    unsafe { *PRESENT.get() }
}

pub fn counters() -> (u64, u64) {
    (
        READS.load(core::sync::atomic::Ordering::Relaxed),
        TIMEOUTS.load(core::sync::atomic::Ordering::Relaxed),
    )
}

pub fn report() {
    let (reads, timeouts) = counters();
    let s = status();
    crate::kprintln!("  ports {:#x} data, {:#x} command   status {:#04x}", DATA, CMD, s);
    if s == 0xFF {
        crate::kprintln!("  nothing answers those ports: every bit reads set, which is an open bus");
        crate::kprintln!("  QEMU models no embedded controller, so this is expected here");
    } else {
        crate::kprintln!(
            "  obf {}  ibf {}",
            (s & OBF != 0) as u8,
            (s & IBF != 0) as u8
        );
    }
    crate::kprintln!("  {} read(s) answered, {} wait(s) gave up", reads, timeouts);
}

pub fn selftest() -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    // The claim that matters where this can be run at all. An absent
    // controller leaves both status bits set, and a driver that waited for
    // them to clear would never come back.
    let before = counters().1;
    let s = status();
    if s == 0xFF {
        let got = read(0x00);
        claim(&mut ok, got.is_none(), "an absent controller answers nothing rather than hanging");
        claim(&mut ok, counters().1 > before, "and the wait is recorded as having given up");
        claim(&mut ok, !present(), "and nothing claims the controller is there");
    } else {
        // On real hardware there is no address that is safe to read blind, so
        // the only claim available without a namespace is that the status
        // register is not the open-bus pattern.
        claim(&mut ok, true, "a controller answers the status port");
        crate::kprintln!("  status {:#04x}; reads are exercised through the battery, not here", s);
    }
    ok
}

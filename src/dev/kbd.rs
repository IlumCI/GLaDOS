//! i8042 PS/2 keyboard.
//!
//! This driver exists because the GF63's internal keyboard really is on the
//! legacy i8042 controller -- Windows reports it as `ACPI\MSI0007` bound to
//! `i8042prt`. Most 2022 laptops route the built-in keyboard over USB-HID,
//! which would have meant an xHCI stack, USB enumeration and a HID report
//! parser before a single character could be typed. Instead it is two I/O
//! ports and a lookup table.
//!
//! Translation (controller config bit 6) is left enabled, so whatever set the
//! keyboard itself is using gets converted to scancode set 1 before we see it.
//! That means one table rather than three.

use super::{lapic, VECTOR_KEYBOARD};
use crate::acpi::Acpi;
use crate::cpu::idt;
use crate::cpu::port::{inb, outb};
use crate::sync::Racy;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const DATA: u16 = 0x60;
const STATUS: u16 = 0x64;
const COMMAND: u16 = 0x64;

/// Status bit 0: output buffer full, i.e. there is a byte to read.
const STATUS_OUTPUT_FULL: u8 = 1 << 0;
/// Status bit 1: input buffer full, i.e. the controller is not ready for a write.
const STATUS_INPUT_FULL: u8 = 1 << 1;

const CMD_READ_CONFIG: u8 = 0x20;
const CMD_WRITE_CONFIG: u8 = 0x60;
const CMD_SELF_TEST: u8 = 0xAA;
const CMD_DISABLE_PORT1: u8 = 0xAD;
const CMD_ENABLE_PORT1: u8 = 0xAE;
const CMD_DISABLE_PORT2: u8 = 0xA7;

const CFG_PORT1_IRQ: u8 = 1 << 0;
const CFG_PORT2_IRQ: u8 = 1 << 1;
const CFG_PORT1_CLOCK_OFF: u8 = 1 << 4;
const CFG_TRANSLATE: u8 = 1 << 6;

const KBD_ENABLE_SCANNING: u8 = 0xF4;

/// Bounded so a wedged controller cannot hang the boot.
const IO_TIMEOUT: u32 = 100_000;

fn wait_writable() -> bool {
    for _ in 0..IO_TIMEOUT {
        if unsafe { inb(STATUS) } & STATUS_INPUT_FULL == 0 {
            return true;
        }
        core::hint::spin_loop();
    }
    false
}

fn wait_readable() -> bool {
    for _ in 0..IO_TIMEOUT {
        if unsafe { inb(STATUS) } & STATUS_OUTPUT_FULL != 0 {
            return true;
        }
        core::hint::spin_loop();
    }
    false
}

fn command(cmd: u8) {
    if wait_writable() {
        unsafe { outb(COMMAND, cmd) };
    }
}

fn write_data(value: u8) {
    if wait_writable() {
        unsafe { outb(DATA, value) };
    }
}

fn read_data() -> Option<u8> {
    if wait_readable() {
        Some(unsafe { inb(DATA) })
    } else {
        None
    }
}

fn flush() {
    for _ in 0..64 {
        if unsafe { inb(STATUS) } & STATUS_OUTPUT_FULL == 0 {
            return;
        }
        let _ = unsafe { inb(DATA) };
    }
}

// --- scancode set 1 -> ASCII ---------------------------------------------

const MAP_LEN: usize = 0x3A;

#[rustfmt::skip]
static UNSHIFTED: [u8; MAP_LEN] = [
    0,    27,  b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', // 00-09
    b'9', b'0', b'-', b'=', 8,    b'\t', b'q', b'w', b'e', b'r', // 0A-13
    b't', b'y', b'u', b'i', b'o', b'p', b'[', b']', b'\n', 0,   // 14-1D  (1D = LCtrl)
    b'a', b's', b'd', b'f', b'g', b'h', b'j', b'k', b'l', b';', // 1E-27
    b'\'', b'`', 0,   b'\\', b'z', b'x', b'c', b'v', b'b', b'n', // 28-31  (2A = LShift)
    b'm', b',', b'.', b'/', 0,    b'*', 0,    b' ',            // 32-39  (36 = RShift, 38 = LAlt)
];

#[rustfmt::skip]
static SHIFTED: [u8; MAP_LEN] = [
    0,    27,  b'!', b'@', b'#', b'$', b'%', b'^', b'&', b'*',
    b'(', b')', b'_', b'+', 8,    b'\t', b'Q', b'W', b'E', b'R',
    b'T', b'Y', b'U', b'I', b'O', b'P', b'{', b'}', b'\n', 0,
    b'A', b'S', b'D', b'F', b'G', b'H', b'J', b'K', b'L', b':',
    b'"', b'~', 0,   b'|', b'Z', b'X', b'C', b'V', b'B', b'N',
    b'M', b'<', b'>', b'?', 0,    b'*', 0,    b' ',
];

const SC_LSHIFT: u8 = 0x2A;
const SC_RSHIFT: u8 = 0x36;
const SC_CAPS: u8 = 0x3A;
const SC_LCTRL: u8 = 0x1D;
const SC_RELEASE: u8 = 0x80;
const SC_EXTENDED: u8 = 0xE0;

static SHIFT: AtomicBool = AtomicBool::new(false);
static CTRL: AtomicBool = AtomicBool::new(false);
static CAPS: AtomicBool = AtomicBool::new(false);
static EXTENDED: AtomicBool = AtomicBool::new(false);

// --- input ring ----------------------------------------------------------

const CAP: usize = 256;
static BUF: Racy<[u8; CAP]> = Racy::new([0; CAP]);
/// Written only by the ISR.
static HEAD: AtomicUsize = AtomicUsize::new(0);
/// Written only by the consumer.
static TAIL: AtomicUsize = AtomicUsize::new(0);

/// Single-producer (the ISR) single-consumer (whoever polls) ring. The
/// acquire/release pairing is what makes it safe for an interrupt to preempt
/// the reader mid-pop.
fn push(byte: u8) {
    let head = HEAD.load(Ordering::Relaxed);
    let next = (head + 1) % CAP;
    if next == TAIL.load(Ordering::Acquire) {
        return; // full: drop, rather than overwrite unread input
    }
    unsafe { BUF.get()[head] = byte };
    HEAD.store(next, Ordering::Release);
}

/// Take one byte of decoded input, if any.
pub fn pop() -> Option<u8> {
    let tail = TAIL.load(Ordering::Relaxed);
    if tail == HEAD.load(Ordering::Acquire) {
        return None;
    }
    let byte = unsafe { BUF.get()[tail] };
    TAIL.store((tail + 1) % CAP, Ordering::Release);
    Some(byte)
}

#[allow(dead_code)]
/// A keystroke from the keyboard, or from the serial console if one answers.
///
/// Everything that reads input should use this rather than `pop`. The shell
/// grew a serial fallback inline and the editor did not, which made the editor
/// impossible to drive headlessly -- and an interactive program that can only
/// be tested by a human at the machine does not get tested.
///
/// A terminal speaks a slightly different dialect than the i8042 driver: Enter
/// arrives as CR and Backspace as DEL. Translating here rather than in
/// `serial::read_byte` keeps the keyboard's own DELETE key, which is a
/// different key that happens to share the 0x7F code, distinguishable.
pub fn pop_any() -> Option<u8> {
    if let Some(k) = pop() {
        return Some(k);
    }
    let b = crate::serial::read_byte()?;
    Some(match b {
        b'\r' => b'\n',
        0x7F => 8,
        other => other,
    })
}

pub fn has_input() -> bool {
    TAIL.load(Ordering::Relaxed) != HEAD.load(Ordering::Acquire)
}

/// Keys with no ASCII representation are delivered as bytes above 0x7F, so the
/// ring buffer stays a simple byte queue rather than growing an event type.
pub const KEY_UP: u8 = 0x80;
pub const KEY_DOWN: u8 = 0x81;
pub const KEY_LEFT: u8 = 0x82;
pub const KEY_RIGHT: u8 = 0x83;
pub const KEY_HOME: u8 = 0x84;
pub const KEY_END: u8 = 0x85;
pub const KEY_DELETE: u8 = 0x86;

fn decode(scancode: u8) {
    // E0 introduces a two-byte sequence: arrows, navigation keys, and the
    // right-hand modifiers.
    if scancode == SC_EXTENDED {
        EXTENDED.store(true, Ordering::Relaxed);
        return;
    }
    if EXTENDED.swap(false, Ordering::Relaxed) {
        let released = scancode & SC_RELEASE != 0;
        if released {
            return;
        }
        let key = match scancode {
            0x48 => KEY_UP,
            0x50 => KEY_DOWN,
            0x4B => KEY_LEFT,
            0x4D => KEY_RIGHT,
            0x47 => KEY_HOME,
            0x4F => KEY_END,
            0x53 => KEY_DELETE,
            // Right ctrl (0x1D) and right alt (0x38) arrive here too; treat
            // them as their left-hand equivalents rather than as characters.
            0x1D => {
                CTRL.store(true, Ordering::Relaxed);
                return;
            }
            _ => return,
        };
        push(key);
        return;
    }

    let released = scancode & SC_RELEASE != 0;
    let code = scancode & !SC_RELEASE;

    match code {
        SC_LSHIFT | SC_RSHIFT => {
            SHIFT.store(!released, Ordering::Relaxed);
            return;
        }
        SC_LCTRL => {
            CTRL.store(!released, Ordering::Relaxed);
            return;
        }
        SC_CAPS => {
            if !released {
                CAPS.fetch_xor(true, Ordering::Relaxed);
            }
            return;
        }
        _ => {}
    }

    // Only make codes produce characters.
    if released || (code as usize) >= MAP_LEN {
        return;
    }

    let shift = SHIFT.load(Ordering::Relaxed);
    let mut ch = if shift {
        SHIFTED[code as usize]
    } else {
        UNSHIFTED[code as usize]
    };
    if ch == 0 {
        return;
    }

    // Caps lock affects letters only, and combines with shift by cancelling it.
    if CAPS.load(Ordering::Relaxed) {
        if ch.is_ascii_lowercase() {
            ch = ch.to_ascii_uppercase();
        } else if ch.is_ascii_uppercase() {
            ch = ch.to_ascii_lowercase();
        }
    }

    if CTRL.load(Ordering::Relaxed) && ch.is_ascii_alphabetic() {
        // Ctrl-A..Ctrl-Z -> 0x01..0x1A
        ch = ch.to_ascii_uppercase() - b'A' + 1;
    }

    push(ch);
}

extern "x86-interrupt" fn keyboard_isr(_frame: idt::InterruptStackFrame) {
    // Always drain the byte. Leaving it in the output buffer means the
    // controller never asserts IRQ1 again and the keyboard silently dies.
    let scancode = unsafe { inb(DATA) };
    decode(scancode);
    lapic::eoi();
}

pub struct InitReport {
    pub self_test: Option<u8>,
    pub config: Option<u8>,
    pub routed_gsi: Option<u32>,
}

/// Bring up the controller and route IRQ 1 through the IOAPIC.
pub fn init(acpi: &Acpi, apic_id: u8) -> InitReport {
    let mut report = InitReport { self_test: None, config: None, routed_gsi: None };

    // Quiesce both ports before touching configuration.
    command(CMD_DISABLE_PORT1);
    command(CMD_DISABLE_PORT2);
    flush();

    // Controller self-test. 0x55 is pass. Some controllers reset their
    // configuration as a side effect, which is why config is written after.
    command(CMD_SELF_TEST);
    report.self_test = read_data();

    command(CMD_READ_CONFIG);
    let current = read_data().unwrap_or(0);

    // Enable IRQ1 and the port clock, keep translation on so we always decode
    // scancode set 1, and leave port 2 (the touchpad) silent for now.
    let config = (current | CFG_PORT1_IRQ | CFG_TRANSLATE)
        & !CFG_PORT1_CLOCK_OFF
        & !CFG_PORT2_IRQ;
    command(CMD_WRITE_CONFIG);
    write_data(config);

    command(CMD_READ_CONFIG);
    report.config = read_data();

    command(CMD_ENABLE_PORT1);

    // Tell the keyboard itself to start sending. It replies 0xFA (ACK).
    write_data(KBD_ENABLE_SCANNING);
    let _ = read_data();

    unsafe { idt::set_handler(VECTOR_KEYBOARD, keyboard_isr as *const (), 0) };

    // IRQ 1 is usually GSI 1, but only the MADT can say so. Assuming identity
    // is how you write a driver that works in QEMU and not on real hardware.
    let (gsi, flags) = acpi.gsi_for_irq(1);
    if let Some(io) = acpi.primary_ioapic() {
        if super::ioapic::route(&io, gsi, VECTOR_KEYBOARD, apic_id, flags) {
            report.routed_gsi = Some(gsi);
        }
    }

    // Drain *after* the line is live, and this ordering is load-bearing.
    //
    // IRQ 1 is edge-triggered. If any byte is sitting in the output buffer
    // when the IOAPIC entry is still masked -- the ACK above is the usual
    // culprit, since it can land after a flush but before routing -- then that
    // edge is discarded, the buffer stays full, and the controller refuses to
    // accept the next scancode. The symptom is losing exactly the first
    // keystroke after boot, which is subtle enough to look like a fluke.
    //
    // Flushing here guarantees the buffer is empty once interrupts can
    // actually be delivered, so the next byte to arrive raises a fresh edge.
    flush();

    report
}

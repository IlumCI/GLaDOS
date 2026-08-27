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

pub(super) fn command(cmd: u8) {
    if wait_writable() {
        unsafe { outb(COMMAND, cmd) };
    }
}

pub(super) fn write_data(value: u8) {
    if wait_writable() {
        unsafe { outb(DATA, value) };
    }
}

pub(super) fn read_data() -> Option<u8> {
    if wait_readable() {
        Some(unsafe { inb(DATA) })
    } else {
        None
    }
}

pub(super) fn flush() {
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
const SC_LALT: u8 = 0x38;
const SC_RELEASE: u8 = 0x80;
const SC_EXTENDED: u8 = 0xE0;

static SHIFT: AtomicBool = AtomicBool::new(false);
static CTRL: AtomicBool = AtomicBool::new(false);
static ALT: AtomicBool = AtomicBool::new(false);
/// Alt has been pressed and nothing else has happened since.
static ALT_ALONE: AtomicBool = AtomicBool::new(false);
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
    /// Whether the byte `pop_any` last returned arrived over the serial line.
    ///
    /// The desktop takes every key while a menu is open, which is right for
    /// somebody at the keyboard and wrong for a line that can only ever be
    /// talking to the shell. A driven session that lands in a menu has no way
    /// out: every byte it sends afterwards feeds the menu, including the ones
    /// meant to close it, and the machine looks like it has stopped reading
    /// its UART. `win keys` is the documented way to drive the desktop
    /// headlessly and goes through the shell, so nothing is lost by keeping
    /// raw serial bytes out of the desktop's hands.
    static FROM_SERIAL: AtomicBool = AtomicBool::new(false);

    pub fn last_was_serial() -> bool {
        FROM_SERIAL.load(Ordering::Relaxed)
    }

    /// One byte off the wire, in the terms the rest of the system uses.
    ///
    /// A terminal sends carriage return for Enter and this kernel's line
    /// editor reads newline, so this is not cosmetic: an untranslated byte is
    /// a line that never ends.
    ///
    /// It is a function because it was not, and the drain loop below pushed
    /// its bytes raw while only the first byte of a burst was translated. A
    /// command typed slowly arrives a byte at a time, takes the first-byte
    /// path every time, and works. The same command sent as one burst -- which
    /// is every command from a script, and any command at all once the machine
    /// is busy enough that bytes queue up behind it -- had its carriage return
    /// pushed raw. So it echoed on screen and then never ran, and the next
    /// command concatenated onto the line it left behind.
    ///
    /// That is the whole of what looked like the serial port dropping input.
    /// Nothing was dropped. One byte in each burst was translated and the rest
    /// were not.
    fn translate(b: u8) -> u8 {
        match b {
            b'\r' => b'\n',
            0x7F => 8,
            other => other,
        }
    }

    pub fn pop_any() -> Option<u8> {
        if let Some(k) = pop() {
            FROM_SERIAL.store(false, Ordering::Relaxed);
            return Some(k);
        }
        let b = crate::serial::read_byte()?;
        FROM_SERIAL.store(true, Ordering::Relaxed);
        // Drain the UART FIFO into the ring while it has data, not one byte
        // per poll. The shell polls at its own cadence -- hlt woken by the
        // 100 Hz timer -- and a 45-character command arrives in a burst the
        // 16-byte FIFO cannot hold across ten-millisecond gaps. One byte per
        // poll lost the tail of exactly those bursts, intermittently, and
        // only when the line followed other output; reading until empty
        // closes the window entirely. The ring is the buffer it always
        // should have been.
        let mut guard = 0;
        while guard < 64 {
            guard += 1;
            match crate::serial::read_byte() {
                Some(next) => push(translate(next)),
                None => break,
            }
        }
        Some(translate(b))
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
/// Shift-Tab. Not an extended scancode -- the i8042 sends plain 0x0F and the
/// shift state is what distinguishes it, so it has to be synthesised here.
///
/// Nothing is allowed to *depend* on it: a terminal sends `ESC [ Z` for the
/// same key and the serial path does no ANSI decoding, so any control
/// reachable only this way is one a headless test could never focus. It exists
/// because it is the right behaviour on the real machine.
pub const KEY_BACKTAB: u8 = 0x87;
/// Alt-Tab: cycle the focused window.
///
/// Alt rather than something the serial line can send, because this is the
/// gesture the desktop it imitates used and because Tab has to keep reaching
/// the shell -- a window switcher that stole Tab would make the terminal
/// unusable. The `win` command exists for the headless path.
pub const KEY_ALTTAB: u8 = 0x88;
/// Alt-Space: the window's own menu -- move, size, maximise, close.
pub const KEY_SYSMENU: u8 = 0x89;
/// Alt on its own: open the focused window's menu bar.
pub const KEY_MENU: u8 = 0x8A;
/// Ctrl-Esc: put the keyboard on the taskbar.
///
/// Ctrl-Esc rather than a lone key because Esc alone has to keep meaning
/// "back out", and the Ctrl-letter mapping below cannot express this one --
/// Esc is not a letter, so both would otherwise arrive as plain 27.
pub const KEY_TASKBAR: u8 = 0x8B;

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
        SC_LALT => {
            // Alt pressed and released with nothing in between opens the menu
            // bar, which is how every window of this vintage behaved. The flag
            // is cleared by any other key while Alt is held, so Alt-Tab and
            // Alt-Space do not also trip it on release.
            if released {
                if ALT_ALONE.swap(false, Ordering::Relaxed) {
                    push(KEY_MENU);
                }
            } else {
                ALT_ALONE.store(true, Ordering::Relaxed);
            }
            ALT.store(!released, Ordering::Relaxed);
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
    // Any other key means Alt was a modifier, not a menu request.
    ALT_ALONE.store(false, Ordering::Relaxed);

    // Tab and Space are the keys whose modified forms are *different keys*
    // rather than different characters, and the map cannot express that: both
    // tables hold the same byte at those indices. Alt is checked first, so
    // Alt-Shift-Tab still switches windows.
    if ALT.load(Ordering::Relaxed) {
        match code {
            0x0F => {
                push(KEY_ALTTAB);
                return;
            }
            0x39 => {
                push(KEY_SYSMENU);
                return;
            }
            _ => {}
        }
    }
    if code == 0x0F && shift {
        push(KEY_BACKTAB);
        return;
    }
    if code == 0x01 && CTRL.load(Ordering::Relaxed) {
        push(KEY_TASKBAR);
        return;
    }
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
    // The moment of the press feeds the Oracle's entropy ring -- TempleOS's
    // mechanism, kept: randomness is when the hands moved, and the interrupt
    // is where that timing exists.
    crate::ai::godbits::ins(crate::time::rdtsc());
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

//! The PS/2 mouse, on the i8042's second port.
//!
//! Same controller as the keyboard, second port. Everything about the byte
//! protocol is shared, which is why the port helpers live in `kbd` and are
//! borrowed here rather than written twice: two copies of "wait for the input
//! buffer to drain" is two places to get a timeout wrong.
//!
//! ### Packets, and the bit that makes them decodable
//!
//! The mouse streams three bytes at a time (four with a wheel). The first
//! carries the buttons and the sign bits, and bit 3 of it is **always set**.
//! That bit is the only way to find the start of a packet: the stream has no
//! framing, so a driver that loses sync reads every packet one byte out and
//! the pointer moves in nonsense directions forever. Checking it and dropping
//! the byte when it is clear is what resynchronises.
//!
//! Movement is relative and Y counts *up*, which is the opposite of the
//! framebuffer, so dy is subtracted rather than added.
//!
//! ### The wheel
//!
//! A plain PS/2 mouse sends three bytes and reports device id 0. Setting the
//! sample rate to 200, then 100, then 80 is a magic knock that makes a wheel
//! mouse switch to four-byte packets and report id 3. There is no other way to
//! ask; the sequence is the interface. If the id comes back as anything else,
//! the wheel is absent and packets stay three bytes.

use super::kbd::{command, flush, read_data, write_data};
use super::{lapic, VECTOR_MOUSE};
use crate::cpu::idt;
use crate::cpu::port::inb;
use crate::acpi::Acpi;
use crate::sync::Racy;

const CMD_ENABLE_PORT2: u8 = 0xA8;
const CMD_READ_CONFIG: u8 = 0x20;
const CMD_WRITE_CONFIG: u8 = 0x60;
/// Prefix that sends the next data byte to the mouse rather than the keyboard.
const CMD_TO_MOUSE: u8 = 0xD4;

const CFG_PORT2_IRQ: u8 = 1 << 1;
const CFG_PORT2_CLOCK_OFF: u8 = 1 << 5;

const MOUSE_SET_DEFAULTS: u8 = 0xF6;
const MOUSE_ENABLE: u8 = 0xF4;
const MOUSE_SET_RATE: u8 = 0xF3;
const MOUSE_GET_ID: u8 = 0xF2;
const ACK: u8 = 0xFA;

/// Cursor position and buttons, as the desktop reads them.
#[derive(Clone, Copy, Default)]
pub struct State {
    pub x: i32,
    pub y: i32,
    pub left: bool,
    pub right: bool,
    /// Wheel notches since the last read, positive downward.
    pub wheel: i32,
    /// Set whenever anything changed, so the desktop can skip a repaint.
    pub moved: bool,
}

static STATE: Racy<State> = Racy::new(State {
    x: 0, y: 0, left: false, right: false, wheel: 0, moved: false,
});
static PRESENT: Racy<bool> = Racy::new(false);
static WHEEL: Racy<bool> = Racy::new(false);

/// Packet assembly. Three or four bytes, indexed by `phase`.
static BUF: Racy<[u8; 4]> = Racy::new([0; 4]);
static PHASE: Racy<usize> = Racy::new(0);
static BOUNDS: Racy<(i32, i32)> = Racy::new((1280, 800));

pub fn present() -> bool {
    unsafe { *PRESENT.get() }
}

/// Read the current state and clear the per-frame fields.
///
/// Wheel notches and `moved` are consumed rather than sampled: they describe
/// what happened since the last look, and a reader that only sampled them
/// would either miss a notch or apply it twice.
pub fn take() -> State {
    let s = unsafe { &mut *STATE.get() };
    let out = *s;
    s.wheel = 0;
    s.moved = false;
    out
}

/// Where the pointer is, without consuming anything.
///
/// `take` clears `moved` and the wheel because they describe what happened
/// since the last look. A reader that only wants to draw the cursor must not
/// do that -- it would swallow a click or a notch that the real handler has
/// not seen yet. See `desk::pump_cursor`.
pub fn position() -> Option<(u32, u32)> {
    if !present() {
        return None;
    }
    let s = unsafe { *STATE.get() };
    Some((s.x.max(0) as u32, s.y.max(0) as u32))
}

/// Apply one movement, whatever produced it.
///
/// Split out of the PS/2 handler because a USB mouse arrives by a completely
/// different road and has to mean exactly the same thing at the end of it.
/// Two copies of "y counts up from the mouse and down on the screen" is one
/// copy too many, and the second would be the one nobody tested.
///
/// The caller decodes its own wire format: PS/2 packs nine-bit deltas with the
/// signs in a flags byte, HID sends plain signed bytes, and neither of those
/// belongs in here.
pub fn apply(dx: i32, dy: i32, left: bool, right: bool, wheel: i32) {
    let s = unsafe { &mut *STATE.get() };
    let (w, h) = unsafe { *BOUNDS.get() };
    if dx != 0 || dy != 0 {
        s.x = (s.x + dx).clamp(0, w - 1);
        s.y = (s.y - dy).clamp(0, h - 1);
        s.moved = true;
    }
    if wheel != 0 {
        s.wheel += wheel;
        s.moved = true;
    }
    if left != s.left || right != s.right {
        s.moved = true;
    }
    s.left = left;
    s.right = right;
}

/// Announce that a pointer exists, for a driver that is not the i8042.
///
/// The desktop draws no cursor at all while this is false, so a USB mouse that
/// moved the state without setting it would be invisible.
pub fn declare_present() {
    unsafe { *PRESENT.get() = true };
}

pub fn set_bounds(w: i32, h: i32) {
    unsafe { *BOUNDS.get() = (w, h) };
    let s = unsafe { &mut *STATE.get() };
    s.x = s.x.clamp(0, w - 1);
    s.y = s.y.clamp(0, h - 1);
}

/// Send one byte to the mouse and take its acknowledgement.
fn mouse_cmd(byte: u8) -> Option<u8> {
    command(CMD_TO_MOUSE);
    write_data(byte);
    read_data()
}

fn set_rate(rate: u8) {
    mouse_cmd(MOUSE_SET_RATE);
    mouse_cmd(rate);
}

pub struct InitReport {
    pub present: bool,
    pub wheel: bool,
    pub id: Option<u8>,
    pub routed_gsi: Option<u32>,
}

pub fn init(acpi: &Acpi, apic_id: u8) -> InitReport {
    let mut report = InitReport { present: false, wheel: false, id: None, routed_gsi: None };

    command(CMD_ENABLE_PORT2);

    command(CMD_READ_CONFIG);
    let current = read_data().unwrap_or(0);
    let config = (current | CFG_PORT2_IRQ) & !CFG_PORT2_CLOCK_OFF;
    command(CMD_WRITE_CONFIG);
    write_data(config);

    // Defaults first. A mouse left in a strange sample rate or resolution by
    // the firmware reports at a speed the pointer cannot be tuned out of.
    if mouse_cmd(MOUSE_SET_DEFAULTS) != Some(ACK) {
        return report;
    }
    report.present = true;

    // The knock that unlocks the wheel. Harmless on a mouse without one: it
    // simply keeps reporting id 0 and three-byte packets.
    set_rate(200);
    set_rate(100);
    set_rate(80);
    mouse_cmd(MOUSE_GET_ID);
    let id = read_data();
    report.id = id;
    if id == Some(3) {
        report.wheel = true;
        unsafe { *WHEEL.get() = true };
    }

    mouse_cmd(MOUSE_ENABLE);

    unsafe {
        *PRESENT.get() = true;
        idt::set_handler(VECTOR_MOUSE, mouse_isr as *const (), 0);
    }

    // IRQ 12, and the MADT is the only thing that knows which GSI that is.
    let (gsi, flags) = acpi.gsi_for_irq(12);
    if let Some(io) = acpi.primary_ioapic() {
        if super::ioapic::route(&io, gsi, VECTOR_MOUSE, apic_id, flags) {
            report.routed_gsi = Some(gsi);
        }
    }

    flush();
    report
}

extern "x86-interrupt" fn mouse_isr(_frame: idt::InterruptStackFrame) {
    let byte = unsafe { inb(0x60) };
    // The hand on the mouse feeds the Oracle's entropy ring exactly as the
    // hand on the keyboard does -- TempleOS folded both into KbdMsEvtTime.
    crate::ai::godbits::ins(crate::time::rdtsc());
    let phase = unsafe { &mut *PHASE.get() };
    let buf = unsafe { &mut *BUF.get() };

    // Bit 3 of the first byte is always set. If it is not, this is not the
    // start of a packet and the stream is out of sync; dropping the byte is
    // the only way back, because there is no framing to resynchronise against.
    if *phase == 0 && byte & 0x08 == 0 {
        lapic::eoi();
        return;
    }

    buf[*phase] = byte;
    *phase += 1;

    let want = if unsafe { *WHEEL.get() } { 4 } else { 3 };
    if *phase < want {
        lapic::eoi();
        return;
    }
    *phase = 0;

    let flags = buf[0];
    // Overflow means the counter saturated between packets. The delta is
    // meaningless then, so the movement is dropped rather than applied as a
    // jump across the screen.
    let overflow = flags & 0xC0 != 0;

    // Nine-bit signed, with the sign carried in the flags byte. Overflow means
    // the counter saturated between packets, so the delta is meaningless and
    // the movement is dropped rather than applied as a jump across the screen.
    let (dx, dy) = if overflow {
        (0, 0)
    } else {
        (
            buf[1] as i32 - if flags & 0x10 != 0 { 0x100 } else { 0 },
            buf[2] as i32 - if flags & 0x20 != 0 { 0x100 } else { 0 },
        )
    };

    let mut wheel = 0i32;
    if want == 4 {
        // Untested against a real notch, and QEMU cannot supply one. Tracing
        // the raw bytes showed the mouse in four-byte mode (the enable knock
        // worked) with byte 3 always 0x00, and the monitor's `mouse_button 8`
        // and `mouse_button 16` producing no packet at all, while
        // `mouse_button 1` produced a clean 0x09 press and 0x08 release. So
        // HMP has no wheel; verifying this needs QMP input-send-event or real
        // hardware, and until then this decode is written from the spec rather
        // than measured.
        //
        // Low nibble, sign extended: 0x01 is one notch down, 0x0F one up.
        let z = (buf[3] & 0x0F) as i8;
        wheel = if z & 0x08 != 0 { z - 16 } else { z } as i32;
    }

    apply(dx, dy, flags & 0x01 != 0, flags & 0x02 != 0, wheel);
    lapic::eoi();
}

//! Local APIC: enable, calibrate, and run the periodic timer.
//!
//! The interesting part is calibration. The APIC timer counts at some fraction
//! of the CPU's bus/core crystal, and nothing tells us that frequency directly
//! on every machine -- CPUID leaf 0x15 reports it on modern Intel parts but is
//! not guaranteed to be populated, and is often absent under emulation. So we
//! measure it against a clock whose frequency is architecturally fixed: the
//! 8254 PIT, at 1193182 Hz.
//!
//! PIT channel 2 is used rather than channel 0 because channel 2's gate is
//! software-controlled through port 0x61, which gives us a clean start trigger
//! and a pollable "done" bit without needing interrupts. Channel 0 is wired to
//! IRQ0 and cannot be polled this way.

use super::{VECTOR_SPURIOUS, VECTOR_TIMER};
use crate::cpu::port::{inb, outb};
use crate::cpu::{self, idt};
use crate::sync::Racy;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicU64, Ordering};

const IA32_APIC_BASE: u32 = 0x1B;
/// Bit 11 of IA32_APIC_BASE: global enable.
const APIC_GLOBAL_ENABLE: u64 = 1 << 11;

const REG_ID: usize = 0x020;
const REG_EOI: usize = 0x0B0;
const REG_SVR: usize = 0x0F0;
const REG_LVT_TIMER: usize = 0x320;
const REG_TIMER_INIT: usize = 0x380;
const REG_TIMER_CUR: usize = 0x390;
const REG_TIMER_DIV: usize = 0x3E0;

/// Divide Configuration encoding for "divide by 16".
///
/// The field is bits [3,1,0] with bit 2 reserved, so this is not the plain
/// integer 16 or 4 that it looks like it should be.
const DIV_16: u32 = 0b0011;

/// Bit 17 of the LVT timer entry selects periodic rather than one-shot.
const LVT_PERIODIC: u32 = 1 << 17;
/// Bit 16 masks the entry.
const LVT_MASKED: u32 = 1 << 16;

const PIT_CH2_DATA: u16 = 0x42;
const PIT_CMD: u16 = 0x43;
const PIT_GATE: u16 = 0x61;
const PIT_HZ: u64 = 1_193_182;

static BASE: Racy<u64> = Racy::new(0);
static TIMER_HZ: Racy<u64> = Racy::new(0);
static TICKS: AtomicU64 = AtomicU64::new(0);

#[inline]
fn base() -> u64 {
    unsafe { *BASE.get() }
}

#[inline]
fn read(reg: usize) -> u32 {
    unsafe { read_volatile((base() + reg as u64) as *const u32) }
}

#[inline]
fn write(reg: usize, value: u32) {
    unsafe { write_volatile((base() + reg as u64) as *mut u32, value) };
}

/// Signal end-of-interrupt. Every handler for an APIC-delivered vector must
/// call this or that priority level stays blocked and interrupts simply stop.
#[inline]
pub fn eoi() {
    write(REG_EOI, 0);
}

pub fn id() -> u8 {
    (read(REG_ID) >> 24) as u8
}

pub fn ticks() -> u64 {
    TICKS.load(Ordering::Relaxed)
}

pub fn timer_hz() -> u64 {
    unsafe { *TIMER_HZ.get() }
}

extern "x86-interrupt" fn timer_isr(_frame: idt::InterruptStackFrame) {
    TICKS.fetch_add(1, Ordering::Relaxed);
    // EOI before scheduling, not after: the switch below may not return for a
    // long time, and leaving the interrupt in service would block every
    // further interrupt at this priority in the meantime.
    eoi();
    crate::task::tick();
}

/// A spurious interrupt is the APIC telling us an interrupt was withdrawn
/// before it could be delivered. It is not acknowledged -- sending EOI here
/// would retire an unrelated real interrupt.
extern "x86-interrupt" fn spurious_isr(_frame: idt::InterruptStackFrame) {}

pub fn init(lapic_addr: u64) {
    unsafe {
        *BASE.get() = lapic_addr;

        let msr = cpu::rdmsr(IA32_APIC_BASE);
        cpu::wrmsr(IA32_APIC_BASE, msr | APIC_GLOBAL_ENABLE);

        idt::set_handler(VECTOR_TIMER, timer_isr as *const (), 0);
        idt::set_handler(VECTOR_SPURIOUS, spurious_isr as *const (), 0);
    }

    // Bit 8 of the spurious vector register is the APIC software enable.
    write(REG_SVR, 0x100 | VECTOR_SPURIOUS as u32);
}

/// Measure APIC timer ticks per second against the PIT.
///
/// Returns 0 if the PIT never signalled, which happens on hardware with no
/// 8254 at all. Callers must treat 0 as "fall back to something else" rather
/// than dividing by it.
pub fn calibrate() -> u64 {
    const SAMPLE_HZ: u64 = 100; // measure over 10 ms
    let divisor = (PIT_HZ / SAMPLE_HZ) as u16;

    unsafe {
        // Enable channel 2's gate, keep the speaker itself disconnected
        // (bit 0 = gate, bit 1 = speaker data).
        let original = inb(PIT_GATE);
        outb(PIT_GATE, (original & 0xFD) | 0x01);

        // Channel 2, lobyte then hibyte, mode 1 (hardware retriggerable
        // one-shot), binary. Mode 1 is gate-triggered, which is exactly the
        // start signal we want.
        outb(PIT_CMD, 0b1011_0010);
        outb(PIT_CH2_DATA, (divisor & 0xFF) as u8);
        let _ = inb(0x60); // brief settle between the two halves
        outb(PIT_CH2_DATA, (divisor >> 8) as u8);

        // Park the APIC timer, masked, at maximum count.
        write(REG_TIMER_DIV, DIV_16);
        write(REG_LVT_TIMER, LVT_MASKED);

        // Drop the gate then raise it: that edge starts the PIT counting.
        let gate = inb(PIT_GATE) & 0xFE;
        outb(PIT_GATE, gate);
        write(REG_TIMER_INIT, u32::MAX);
        outb(PIT_GATE, gate | 1);

        // Bit 5 of port 0x61 is channel 2's OUT line, high at terminal count.
        let mut guard: u64 = 0;
        while inb(PIT_GATE) & 0x20 == 0 {
            guard += 1;
            if guard > 100_000_000 {
                write(REG_TIMER_INIT, 0);
                outb(PIT_GATE, original);
                return 0;
            }
            core::hint::spin_loop();
        }

        let remaining = read(REG_TIMER_CUR);
        write(REG_TIMER_INIT, 0); // stop
        outb(PIT_GATE, original);

        let elapsed = (u32::MAX - remaining) as u64;
        let hz = elapsed * SAMPLE_HZ;
        *TIMER_HZ.get() = hz;
        hz
    }
}

/// Start the periodic timer at `hz` interrupts per second.
///
/// Returns false if calibration produced nothing usable, rather than dividing
/// by zero or programming a count of 0 (which means "timer off", so the
/// failure would look like a hang instead of an error).
pub fn start_timer(hz: u32) -> bool {
    let measured = timer_hz();
    if measured == 0 || hz == 0 {
        return false;
    }
    let count = measured / hz as u64;
    if count == 0 || count > u32::MAX as u64 {
        return false;
    }

    write(REG_TIMER_DIV, DIV_16);
    write(REG_LVT_TIMER, VECTOR_TIMER as u32 | LVT_PERIODIC);
    write(REG_TIMER_INIT, count as u32);
    true
}

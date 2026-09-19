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
use crate::cpu::port::{inb, inl, outb};
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
/// Interrupt Command Register. Two halves in xAPIC: write the high word
/// (destination) first, then the low word, which is what actually sends.
const REG_ICR_LO: usize = 0x300;
const REG_ICR_HI: usize = 0x310;
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

/// TSC cycles per microsecond, measured against the PIT or the PM timer.
///
/// **A second clock that owes nothing to `TICKS`, which is the entire point.**
/// `time::calibrate` derives the microsecond *from* the tick counter, so a
/// tick rate wrong by a whole core count cancels out of every comparison made
/// against it -- `elapsed_us` and the TSC delta both scale, and the ratio
/// comes out exactly right. Proven by injection rather than argued: with the
/// ISR incrementing by two, `tsc` read 1345 MHz against a true 2690, the boot
/// check answered "the two clocks agree" over a window half as long as it
/// believed, and `uptime` reported 22.54 s at 11.3 s of real time.
///
/// Both calibrations below already bracket an exact 10 ms window on a
/// reference this kernel did not have to trust, so this costs two `rdtsc`
/// reads and no extra time at all.
static TSC_REF: Racy<u64> = Racy::new(0);

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

/// Spin until the last IPI has been accepted.
///
/// Bit 12 of the low ICR word is Delivery Status. Writing a second IPI while
/// it is set loses one of them silently, which in a startup sequence reads as
/// "that core is dead" rather than as a dropped write.
fn ipi_settle() {
    for _ in 0..1_000_000 {
        if read(REG_ICR_LO) & (1 << 12) == 0 {
            return;
        }
        core::hint::spin_loop();
    }
}

/// INIT: hold the target core in reset, ready for a startup vector.
pub fn send_init(apic_id: u32) {
    write(REG_ICR_HI, apic_id << 24);
    // 0x4500 = INIT delivery mode, assert level, edge trigger.
    write(REG_ICR_LO, 0x0000_4500);
    ipi_settle();
}

/// SIPI: start the target core in real mode at `page << 12`.
///
/// The vector *is* the address: there are eight bits for it, so the
/// trampoline has to live in a page below 1 MiB. That constraint is the only
/// reason `mem::frame` refuses to hand out low memory.
pub fn send_sipi(apic_id: u32, page: u8) {
    write(REG_ICR_HI, apic_id << 24);
    write(REG_ICR_LO, 0x0000_4600 | page as u32);
    ipi_settle();
}

pub fn ticks() -> u64 {
    TICKS.load(Ordering::Relaxed)
}

pub fn timer_hz() -> u64 {
    unsafe { *TIMER_HZ.get() }
}

/// TSC cycles per microsecond against the calibration reference, or 0.
///
/// Zero means neither the PIT nor the PM timer answered, so there is no
/// independent clock and a caller must say so rather than compare anyway.
pub fn tsc_per_us_ref() -> u64 {
    unsafe { *TSC_REF.get() }
}

extern "x86-interrupt" fn timer_isr(frame: idt::InterruptStackFrame) {
    // Only the bootstrap processor advances the clock.
    //
    // Every core runs this handler, and that is deliberate -- `init_this_core`
    // says why the vector registrations are global. But `TICKS` is one counter,
    // so incrementing it on each core made it advance at N x TIMER_HZ, and
    // every duration derived from it was wrong by the core count.
    //
    // The one that mattered was `tcp::wait_until`, which builds its deadline as
    // `ticks() + ms * TIMER_HZ / 1000`: every network timeout in the kernel was
    // short by the core count, so a 15 s TLS deadline was 3.75 s under the
    // tooling's default `-smp 4` and would be under a second on the GF63's
    // sixteen logical processors. `uptime` was wrong the other way, and the
    // model selftest's tokens/sec under-reported by the same factor.
    //
    // Measured before the fix: a 4-core guest reported 55.58 s of uptime during
    // a 25 s run -- longer than the whole invocation, QEMU startup included.
    //
    // `armed()` is checked because per-core storage comes up after the timer
    // does, and before it is armed there is only one core running anyway.
    let now = if !cpu::percpu::armed() || cpu::percpu::cpu_id() == 0 {
        TICKS.fetch_add(1, Ordering::Relaxed) + 1
    } else {
        TICKS.load(Ordering::Relaxed)
    };
    // EOI before scheduling, not after: the switch below may not return for a
    // long time, and leaving the interrupt in service would block every
    // further interrupt at this priority in the meantime.
    eoi();

    // A guest that never makes a syscall owns the machine, and there is no key
    // to press because the guest is what is running. The timer is the only
    // thing that still gets a turn, so it is where the deadline lives.
    //
    // **Only from ring 3**, which is the whole of the safety argument: the
    // saved CS says the guest itself was executing rather than the kernel
    // working on its behalf, so there is no lock held and no half-finished
    // allocation to abandon. An interrupt arriving during a syscall simply
    // lets the deadline pass and catches it on the next tick outside one.
    if frame.cs & 3 == 3 && crate::linux::syscall::overran(now) {
        unsafe { crate::linux::syscall::kill_overrun() }
    }
    // Scheduled input, for the same reason the deadline above lives here: a
    // running guest owns the machine and the timer is the only thing that
    // still gets a turn. Unlike the deadline this runs whatever the saved CS
    // says, because a guest blocked in a read is sitting in the kernel and
    // that is precisely when it is waiting to be fed.
    crate::linux::input::service(now);
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

/// Enable this core's own local controller.
///
/// Each core has its own controller behind the same physical address, so the
/// register writes are per-core even though the address is not. The handler
/// registrations in `init` are global and are deliberately not repeated: an
/// interrupt table entry is code, and every core wants the same code.
pub fn init_this_core() {
    unsafe {
        let msr = cpu::rdmsr(IA32_APIC_BASE);
        cpu::wrmsr(IA32_APIC_BASE, msr | APIC_GLOBAL_ENABLE);
    }
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
        let t0 = crate::time::rdtsc();

        // Bit 5 of port 0x61 is channel 2's OUT line, high at terminal count.
        //
        // The guard is deliberately small. Each `inb` of a legacy port goes out
        // over LPC/eSPI and costs on the order of a microsecond on real
        // hardware, so a 100-million iteration guard is not a safety net, it is
        // a hundred-second hang that looks exactly like a lock-up. Two million
        // is a couple of seconds -- long enough for any PIT that is going to
        // answer, short enough to fall through to the PM timer.
        let mut guard: u64 = 0;
        while inb(PIT_GATE) & 0x20 == 0 {
            guard += 1;
            if guard > 2_000_000 {
                write(REG_TIMER_INIT, 0);
                outb(PIT_GATE, original);
                return 0;
            }
            core::hint::spin_loop();
        }

        let t1 = crate::time::rdtsc();
        let remaining = read(REG_TIMER_CUR);
        write(REG_TIMER_INIT, 0); // stop
        outb(PIT_GATE, original);

        let elapsed = (u32::MAX - remaining) as u64;
        let hz = elapsed * SAMPLE_HZ;
        *TIMER_HZ.get() = hz;
        // The same window read on the other clock. The one-shot ends at
        // terminal count, so this is 10 ms by construction rather than by
        // measurement, and the spin's last `inb` overshoots it by about a
        // microsecond in ten thousand.
        *TSC_REF.get() = t1.wrapping_sub(t0) / (1_000_000 / SAMPLE_HZ);
        hz
    }
}

/// Calibrate against the ACPI power-management timer instead of the PIT.
///
/// The PM timer runs at a fixed 3.579545 MHz -- it is architecturally defined,
/// unlike the APIC timer's own frequency -- and unlike the 8254 it is still
/// genuinely present on modern chipsets. The FADT gives us its I/O port; on
/// this laptop that is 0x1808, where QEMU reports 0x608.
///
/// Only the low 24 bits are used. The counter may be 24 or 32 bits wide
/// depending on the FADT flags, and masking to 24 is correct for both as long
/// as the sample is far shorter than a 24-bit wrap (about 4.6 seconds).
pub fn calibrate_pm(port: u16) -> u64 {
    const PM_HZ: u64 = 3_579_545;
    const MASK: u32 = 0x00FF_FFFF;
    const SAMPLE_HZ: u64 = 100; // 10 ms
    let want = (PM_HZ / SAMPLE_HZ) as u32;

    unsafe {
        write(REG_TIMER_DIV, DIV_16);
        write(REG_LVT_TIMER, LVT_MASKED);

        let start = inl(port) & MASK;
        let t0 = crate::time::rdtsc();
        write(REG_TIMER_INIT, u32::MAX);

        let mut guard: u64 = 0;
        // The loop yields how far the counter actually went, which is *at
        // least* `want` and not exactly it. Re-reading the port afterwards
        // would answer a slightly later question than the one the APIC count
        // was taken over.
        let elapsed_pm = loop {
            let elapsed = (inl(port).wrapping_sub(start)) & MASK;
            if elapsed >= want {
                break elapsed as u64;
            }
            guard += 1;
            if guard > 20_000_000 {
                write(REG_TIMER_INIT, 0);
                return 0; // The port is not a live counter.
            }
            core::hint::spin_loop();
        };
        let t1 = crate::time::rdtsc();

        let remaining = read(REG_TIMER_CUR);
        write(REG_TIMER_INIT, 0);

        let elapsed_apic = (u32::MAX - remaining) as u64;
        let hz = elapsed_apic * SAMPLE_HZ;
        *TIMER_HZ.get() = hz;
        // See `TSC_REF`. The PM timer's rate is architecturally fixed, so this
        // window is known exactly rather than assumed.
        let us = elapsed_pm * 1_000_000 / PM_HZ;
        if us > 0 {
            *TSC_REF.get() = t1.wrapping_sub(t0) / us;
        }
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

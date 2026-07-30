//! Delays finer than the scheduler tick.
//!
//! `lapic::ticks()` counts at `TIMER_HZ`, which is 100, so its resolution is
//! 10 ms -- three orders of magnitude too coarse to pace individual characters
//! on the console. The TSC has the resolution we need but not, on its own, a
//! known rate: `rdtsc` counts at whatever the part's invariant TSC frequency
//! is, and CPUID leaf 15H only reports it on some models.
//!
//! So we calibrate against a clock we already trust. The LAPIC timer was itself
//! calibrated at boot against the PIT (or the ACPI PM timer, when the PIT does
//! not answer), which makes this the third link in a chain -- and worth being
//! honest about: a delay here is only as good as that calibration. It is used
//! for cosmetics, not for device timing, and nothing correctness-critical may
//! depend on it.
//!
//! If calibration never runs or fails, `delay_us` returns immediately rather
//! than spinning for a guessed interval. Silently doing nothing is much easier
//! to notice and diagnose than silently waiting the wrong amount.

use crate::dev::lapic;
use core::sync::atomic::{AtomicU64, Ordering};

static TSC_PER_US: AtomicU64 = AtomicU64::new(0);

/// Read the timestamp counter.
///
/// Written as inline assembly rather than through `core::arch::x86_64::_rdtsc`
/// because that intrinsic's safety changed across releases, and an unnecessary
/// `unsafe` block is itself a warning.
#[inline]
pub fn rdtsc() -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        core::arch::asm!(
            "rdtsc",
            out("eax") lo,
            out("edx") hi,
            options(nomem, nostack, preserves_flags),
        );
    }
    ((hi as u64) << 32) | lo as u64
}

/// Measure the TSC against the LAPIC timer. Requires interrupts enabled and
/// the timer already running, so call it after `start_timer`.
///
/// Costs 50 ms of boot time. That is the price of the measurement itself --
/// a shorter window would divide a smaller elapsed count by a smaller tick
/// count and give a proportionally noisier answer.
pub fn calibrate() {
    // `lapic::timer_hz()` is the APIC timer's *input* frequency -- tens of
    // megahertz -- not the rate `ticks()` advances at. That is `TIMER_HZ`, the
    // divider `start_timer` was programmed with. Using the former here made
    // `elapsed_us` truncate to zero and calibration silently give up.
    //
    // Reading it is still the right precondition: zero means the timer was
    // never calibrated, so nothing is ticking to measure against.
    if lapic::timer_hz() == 0 {
        return;
    }
    let hz = crate::TIMER_HZ as u64;

    // Start on a tick edge, otherwise the first tick is a partial interval and
    // the measurement is short by up to one whole period.
    let edge = lapic::ticks();
    if !wait_for_tick(edge) {
        return;
    }

    const TICKS: u64 = 5;
    let base = lapic::ticks();
    let t0 = rdtsc();
    let mut seen = base;
    while seen < base + TICKS {
        if !wait_for_tick(seen) {
            return;
        }
        seen = lapic::ticks();
    }
    let t1 = rdtsc();

    let elapsed_us = (seen - base) * 1_000_000 / hz;
    if elapsed_us == 0 {
        return;
    }
    let per_us = t1.wrapping_sub(t0) / elapsed_us;
    TSC_PER_US.store(per_us.max(1), Ordering::Relaxed);
}

/// Spin until `ticks()` moves past `from`. Returns false if it never does,
/// which means the timer interrupt is not arriving and calibration must be
/// abandoned rather than hang the boot.
fn wait_for_tick(from: u64) -> bool {
    // Bounded in TSC cycles, not iterations: an iteration count is a guess
    // about clock speed, and guessing wrong is exactly how the PIT
    // calibration loop came to hang for a hundred seconds.
    let deadline = rdtsc() + 40_000_000_000; // >= 4 s on anything above 10 GHz
    while lapic::ticks() == from {
        if rdtsc() > deadline {
            return false;
        }
        core::hint::spin_loop();
    }
    true
}

pub fn is_calibrated() -> bool {
    TSC_PER_US.load(Ordering::Relaxed) != 0
}

/// TSC frequency in MHz, or 0 if uncalibrated.
pub fn tsc_mhz() -> u64 {
    TSC_PER_US.load(Ordering::Relaxed)
}

/// Busy-wait. Does nothing at all if the TSC has not been calibrated.
pub fn delay_us(us: u64) {
    let per = TSC_PER_US.load(Ordering::Relaxed);
    if per == 0 || us == 0 {
        return;
    }
    let target = rdtsc() + per * us;
    while rdtsc() < target {
        core::hint::spin_loop();
    }
}

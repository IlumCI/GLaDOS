//! Monotonic time, at a resolution a frame loop can use.
//!
//! **The one clock fine enough is the TSC.** `lapic::ticks()` counts timer
//! interrupts at `TIMER_HZ`, which is 100 -- a 10 ms granularity, against
//! DOOM's 28.571 ms tic. Sampling a 35 Hz cadence with a 100 Hz clock aliases
//! badly: two of every seven tics would be measured a whole tick short.
//!
//! There is a second trap here that has already cost this project a session,
//! recorded in CLAUDE.md: `lapic::ticks()` is the interrupt *count* at 100 Hz,
//! and `lapic::timer_hz()` is the calibrated APIC input frequency in the
//! millions. Dividing an uptime by the latter puts every reading at zero.
//!
//! **And this is the third copy of `now_ms` in the tree**, which is the reason
//! it lives here rather than in the caller. The other two are private:
//! `desk.rs` has one for double-click timing and `mines.rs` has one for the
//! game timer. Anything ported gets this one.

/// Microseconds since boot, or 0 if the TSC has not been calibrated.
///
/// Zero rather than a guess. `time::delay_us` makes the same choice and says
/// why: a clock that is confidently wrong is worse than one that is visibly
/// absent, and every caller here is comparing two readings, so a constant zero
/// makes the elapsed time zero rather than making it nonsense.
pub fn now_us() -> u64 {
    let per = crate::time::tsc_mhz();
    if per == 0 {
        return 0;
    }
    crate::time::rdtsc() / per
}

/// Milliseconds since boot.
pub fn now_ms() -> u64 {
    now_us() / 1000
}

/// Whether the clock can be trusted at all, so a caller can say so rather than
/// silently running at whatever speed a zero elapsed time implies.
pub fn ready() -> bool {
    crate::time::tsc_mhz() != 0
}

//! Time, for a stack whose every state machine is a timeout.
//!
//! 802.11 is timeouts almost all the way down: a probe waits, an
//! authentication waits, an association waits, the four-way handshake waits,
//! and each of them retries a bounded number of times before giving up. So the
//! seam owes a clock and a way to say "has this expired yet".

/// Microseconds since boot, or zero when the TSC has not been calibrated.
///
/// Zero rather than a guess, the choice `time::delay_us` and `port::clock`
/// both make and for the same reason: every caller here compares two readings,
/// so a constant zero makes an interval zero rather than nonsense.
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

/// Whether the clock can be trusted, so a caller can decline rather than run
/// at whatever speed a zero interval implies.
pub fn ready() -> bool {
    crate::time::tsc_mhz() != 0
}

/// Busy-wait. Only for the microsecond-scale waits a register sequence needs.
///
/// Anything longer belongs in a `Deadline` and a poll, because this holds the
/// core: `task.rs` preempts at 100 Hz but a spin here still burns the quantum,
/// and the resident mind is on the other end of that.
pub fn delay_us(us: u64) {
    crate::time::delay_us(us);
}

/// A moment in the future, and whether it has arrived.
///
/// A type rather than a bare `u64` because the mistake it prevents is the one
/// everybody makes once: comparing a deadline against an interval, or an
/// interval against a deadline, both of which typecheck as integers and
/// neither of which says so.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Deadline(u64);

impl Deadline {
    /// A deadline `ms` milliseconds from now.
    pub fn in_ms(ms: u64) -> Deadline {
        Deadline(now_ms().saturating_add(ms))
    }

    /// Whether it has passed.
    pub fn expired(self) -> bool {
        now_ms() >= self.0
    }

    /// How long is left, saturating at zero rather than wrapping -- an
    /// unsigned subtraction past a deadline is an enormous positive number and
    /// a retry loop that never retries.
    pub fn remaining_ms(self) -> u64 {
        self.0.saturating_sub(now_ms())
    }
}

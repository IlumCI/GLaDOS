//! Randomness, with the distinction between the two kinds kept.
//!
//! 802.11 needs unpredictable bytes in two quite different places, and this
//! seam keeps them apart because the kernel's generator does:
//!
//!   * A **nonce** that must never repeat -- the SNonce of a four-way
//!     handshake, a CCMP packet number's starting point. Repeating one is a
//!     key-recovery bug, so these take `secret`, which refuses below the
//!     entropy threshold rather than degrading quietly.
//!   * A **sequence number, a backoff, a probe delay**. These want
//!     unpredictability without depending on it, and a refusal here would stop
//!     a scan on a machine nobody has typed on yet.
//!
//! `src/net/wpa2.rs` currently derives its SNonce from rotations of `rdtsc`,
//! which is the first kind taking the second kind's route. That is fixed as
//! part of this work rather than carried over.

/// Fill with bytes that are unpredictable if the pool is up, and merely
/// non-repeating if it is not. Never refuses.
pub fn fill(out: &mut [u8]) {
    crate::rng::fill(out);
}

/// Fill with key-grade bytes, or refuse.
///
/// `Err` carries how many entropy events are still wanted, so a caller can say
/// "not enough entropy yet, 91 events short" instead of "failed".
pub fn secret(out: &mut [u8]) -> Result<(), u32> {
    crate::rng::fill_secret(out)
}

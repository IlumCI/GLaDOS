//! Moving each miner to a difficulty that suits it.
//!
//! A pool with one fixed share target is wrong for everybody the moment its
//! miners are not identical. Too hard and a slow device reports in once an
//! hour, so the pool cannot tell a working miner from a departed one and the
//! miner's own rate figure is noise. Too easy and a fast one floods the
//! connection, spending the pool's CPU on validations that say nothing new --
//! and at 7.5 ms a yespower validation, that is the cost that actually bites.
//!
//! ### Per coin as well as per connection, which most pools do not need
//!
//! An ordinary pool serves one coin, so one difficulty per connection is the
//! whole problem. Here a single machine works several coins at once on
//! different algorithms, and its rate on them differs by **three orders of
//! magnitude** -- measured on the GLaDOS kernel in one run, 248,884 H/s of
//! sha256d beside 342 H/s of yespower. One difficulty across both would be
//! wrong for at least one of them by a factor of a thousand, so the state here
//! is per `(connection, coin)`.
//!
//! ### Bits, and therefore no floats
//!
//! Everything is a count of leading zero bits, so a step is exactly a halving
//! and the arithmetic is integer throughout. Difficulty as a float is what
//! `stratum::decimal` exists to survive on the way in; there is no reason to
//! introduce one here on the way out.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::clock;

/// How often a miner should find a share.
///
/// Ten seconds is the usual choice and the reasons are both about information
/// rather than throughput: it is short enough that a rate estimate settles in
/// under a minute, and long enough that the validation cost stays negligible
/// even for the expensive algorithms.
const DEFAULT_TARGET_SECS: u64 = 10;

/// Live, because it is the single biggest lever on what a busy pool costs.
///
/// Validation is per *share*, so the offered share rate is
/// `connections x coins / TARGET_SECS` and the CPU bill is that times the
/// algorithm's cost. Measured on the deployed host: 800 connections on two
/// coins at ten seconds offers 160 shares a second, which is 114% of a core on
/// yespower-2MiB and 8% on neoscrypt. At sixty seconds the same crowd is 19%
/// and 1%.
///
/// So this is the knob that decides whether an expensive algorithm is servable
/// at all at a given size, and having it be a `const` meant the only available
/// answer was to defer a share of the traffic. Longer costs information -- a
/// rate estimate settles more slowly and a dead miner takes longer to notice --
/// which is a real trade and the reason the default does not move.
static TARGET_SECS: AtomicU64 = AtomicU64::new(DEFAULT_TARGET_SECS);

/// Seconds a miner should take to find a share. Clamped to something sane:
/// zero would divide by zero below, and an hour makes a pool that cannot tell
/// a working miner from a departed one.
pub fn set_target_secs(s: u64) -> u64 {
    let s = s.clamp(2, 600);
    TARGET_SECS.store(s, Ordering::Relaxed);
    s
}

pub fn target_secs() -> u64 {
    TARGET_SECS.load(Ordering::Relaxed)
}

/// How many shares to watch before moving. Fewer and the estimate is dominated
/// by luck -- share intervals are exponentially distributed, so a single
/// interval says almost nothing.
const WINDOW_SHARES: u32 = 8;
/// Retarget anyway after this long, which is the case that matters for a miner
/// set far too hard: it will never reach `WINDOW_SHARES`, and without a clock
/// the pool would wait forever to discover it had asked too much.
const WINDOW_SECS: u64 = 60;

/// The most a single adjustment may move, in bits. Three is a factor of eight.
///
/// Bounded because the estimate is noisy and an unbounded correction on a
/// short window overshoots, which shows up as difficulty oscillating instead
/// of settling.
const MAX_STEP: u32 = 3;

/// Floor and ceiling. The floor stops a broken or hostile miner from asking to
/// be given trivial work it can submit constantly; the ceiling stops a fast
/// device being handed something it takes an hour to satisfy.
const MIN_BITS: u32 = 8;
const MAX_BITS: u32 = 40;

pub struct VarDiff {
    bits: u32,
    shares: u32,
    /// When the current window opened, in `clock::now_ms`. Not an `Instant`,
    /// because there is no `Instant` on the one target this also has to run on.
    since_ms: u64,
}

impl VarDiff {
    pub fn new(start_bits: u32) -> VarDiff {
        VarDiff {
            bits: start_bits.clamp(MIN_BITS, MAX_BITS),
            shares: 0,
            since_ms: clock::now_ms(),
        }
    }

    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Record an accepted share. Answers the new difficulty when it moved.
    ///
    /// Only *accepted* shares are counted, and that is not the same as every
    /// submission: a stale share is work against a job that aged out and says
    /// nothing about the rate now, and counting rejected ones would let a
    /// miner talk its own difficulty upward by sending noise.
    pub fn on_share(&mut self) -> Option<u32> {
        self.shares += 1;
        self.settle(false)
    }

    /// Called when no share has arrived for a while, so a miner set far too
    /// hard is still discovered. Answers the new difficulty when it moved.
    pub fn on_idle(&mut self) -> Option<u32> {
        self.settle(true)
    }

    fn settle(&mut self, idle: bool) -> Option<u32> {
        // Milliseconds, and this was seconds first. A window is allowed to be
        // shorter than a second -- eight shares arriving instantly is the most
        // extreme flood there is -- and a seconds-resolution clock reads that
        // as elapsed zero. The first version bailed out on exactly that, so
        // the fastest miners, which are the whole reason this file exists,
        // were the one case that never retargeted. Caught by the test rather
        // than by reading.
        let elapsed_ms = clock::now_ms().saturating_sub(self.since_ms);
        let enough = self.shares >= WINDOW_SHARES || (idle && elapsed_ms >= WINDOW_SECS * 1000);
        if !enough {
            return None;
        }

        // How many shares this window *should* have produced. Floored at one,
        // which is also what keeps the division below safe on a window too
        // short to have wanted any.
        let wanted = (elapsed_ms / (target_secs() * 1000)).max(1);
        let got = self.shares as u64;

        // The step is a base-2 logarithm of the ratio, computed by halving,
        // because a bit is exactly a doubling of difficulty. No float appears
        // anywhere and the answer is the same on every machine.
        let step = if got > wanted {
            log2_ratio(got, wanted)
        } else if wanted > got {
            log2_ratio(wanted, got.max(1))
        } else {
            0
        };
        let step = step.min(MAX_STEP);

        let before = self.bits;
        if got > wanted {
            self.bits = (self.bits + step).min(MAX_BITS);
        } else if wanted > got {
            self.bits = self.bits.saturating_sub(step).max(MIN_BITS);
        }

        // The window restarts whether or not the difficulty moved. A window
        // that only reset on a change would keep accumulating shares from an
        // interval already judged, so the next estimate would be about a period
        // partly spent at a difficulty no longer in force.
        self.shares = 0;
        self.since_ms = clock::now_ms();

        if self.bits == before {
            None
        } else {
            Some(self.bits)
        }
    }
}

/// `floor(log2(a / b))` for positive integers, by halving.
fn log2_ratio(a: u64, b: u64) -> u32 {
    let mut r = a / b.max(1);
    let mut n = 0;
    while r >= 2 {
        r /= 2;
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flood_of_shares_makes_the_work_harder() {
        let mut v = VarDiff::new(16);
        // Eight shares with effectively no time passing is far above one every
        // ten seconds, so the difficulty must rise.
        let mut moved = None;
        for _ in 0..WINDOW_SHARES {
            if let Some(b) = v.on_share() {
                moved = Some(b);
            }
        }
        let b = moved.expect("eight instant shares did not retarget");
        assert!(b > 16, "difficulty went the wrong way: {b}");
        assert!(b <= 16 + MAX_STEP, "a single step overshot: {b}");
    }

    #[test]
    fn the_step_is_bounded_and_so_are_the_ends() {
        // Nothing may be pushed past the ceiling however extreme the ratio.
        let mut v = VarDiff::new(MAX_BITS);
        for _ in 0..WINDOW_SHARES {
            v.on_share();
        }
        assert_eq!(v.bits(), MAX_BITS);

        // Or below the floor. A miner asking for trivial work it can submit
        // constantly is exactly what the floor is for.
        let mut v = VarDiff::new(MIN_BITS);
        assert!(v.on_idle().is_none(), "an empty window retargeted early");
        assert_eq!(v.bits(), MIN_BITS);
    }

    /// A starting value outside the bounds is clamped rather than honoured.
    ///
    /// The operator picks it in a config file and nothing else checks it, so
    /// `--coin x:sha256d:200` would otherwise hand every miner a target
    /// nothing can meet and look like a pool that never accepts anything.
    #[test]
    fn a_configured_start_outside_the_bounds_is_clamped() {
        assert_eq!(VarDiff::new(0).bits(), MIN_BITS);
        assert_eq!(VarDiff::new(255).bits(), MAX_BITS);
    }

    #[test]
    fn the_ratio_logarithm_is_a_floor() {
        assert_eq!(log2_ratio(1, 1), 0);
        assert_eq!(log2_ratio(3, 1), 1);
        assert_eq!(log2_ratio(4, 1), 2);
        assert_eq!(log2_ratio(7, 1), 2);
        assert_eq!(log2_ratio(8, 1), 3);
        // And it never divides by zero, whatever it is handed.
        assert_eq!(log2_ratio(8, 0), 3);
    }
}

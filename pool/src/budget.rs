//! How much of somebody else's machine this pool is allowed to spend.
//!
//! ### The limit that existed, and the multiplication nobody did
//!
//! `server.rs` bounds a single connection at `MAX_SUBMITS_PER_SEC` and argues
//! it carefully: validating a share is the expensive thing, it happens before
//! the sender has proved anything, so a stranger sending garbage buys CPU. The
//! figure quoted there is "fifteen percent of a core even if every message is
//! garbage", computed against a 7.5 ms yespower validation.
//!
//! **Two things are wrong with that and only the second is subtle.**
//!
//! The first: 7.5 ms was measured somewhere else. On the machine this was
//! actually deployed to -- an i3-3240 from 2012 -- `--bench` says an 8 MiB
//! yespower share costs **19 ms**, so twenty submits a second is 380 ms, which
//! is 38% of a core rather than 15%.
//!
//! The second, and the one that matters: **the per-connection bound was never
//! multiplied by the connection ceiling.** `MAX_CONNECTIONS` is 256. At 38% of
//! a core each that is ninety-seven cores of validation, on a machine with
//! four threads. Every individual limit held and the total was unbounded, which
//! is the shape of limit that reads as safe and is not.
//!
//! ### So the budget is total, and it is denominated in time
//!
//! A token bucket where a token is a microsecond of validation. Shares cost
//! what they cost -- a sha256d share is 2.4 us on that machine and a yespower
//! one is 19,000 -- so a pool serving cheap coins is barely bounded and one
//! serving expensive coins is bounded hard, with no per-algorithm rule to keep
//! in step with anything.
//!
//! **"A yespower share" is not one number, and the deployed coin is the cheap
//! one.** `--bench` on that machine reports 5,527 us for the 2 MiB profile and
//! 19,359 us for the 8 MiB one -- a factor of 3.5 between two coins that both
//! spell their algorithm `yespower`, because the working set is the cost and
//! `N` and `r` set it. The 19,000 above is the 8 MiB figure; `run-pool.sh`
//! configures `yespower-10-2048-8`, which is 2 MiB. So the ninety-seven cores
//! is the right arithmetic for the worst profile and about 28 for the one
//! actually running. Both are far past four, so the conclusion does not move --
//! but a reader taking 19 ms as a property of the algorithm would size the
//! budget for a coin they are not serving.
//!
//! **And the live estimate runs above the bench, which is the direction a
//! safety bound should err in.** Driven against that host: 6,008 us a share on
//! one connection and 7,153 us with four, against the bench's uncontended
//! 5,527. A 2 MiB working set on a four-thread part shares one last-level
//! cache, so four connections validating at once cost 30% more each than one
//! does -- the same memory-bandwidth story `smp bench` records in the kernel,
//! arriving here as a budget that tightens exactly when the machine is busy.
//! Process CPU over the same run was 7,338 us per admitted share, so the
//! estimate is if anything slightly *under* the true cost rather than inflated
//! by lock waiting, which was the other candidate explanation and is refuted.
//!
//! ### The cost is measured, because assuming it is what went wrong
//!
//! Every validation is timed and folded into a per-algorithm estimate. There
//! is no calibration phase and no table: a table would be a number about one
//! machine written down in a program meant to run on another, which is exactly
//! the 7.5 ms mistake being repeated one level up.
//!
//! An algorithm nobody has timed yet is admitted once, unbudgeted, and pays
//! for the estimate. That is bounded by the number of algorithms rather than
//! by anything an attacker controls.
//!
//! ### What this does not do
//!
//! **It is not fair.** Under pressure the budget is spent by whoever asks
//! first, so a flood can crowd out an honest miner's share -- it is refused as
//! `Busy` rather than mis-validated, but it is still refused. Per-connection
//! limits bound how badly, and real fairness needs per-worker accounting that
//! this does not have. Written down rather than discovered.
//!
//! **It does not make the pool fast.** It makes the pool's appetite knowable,
//! which is a different thing and the one an operator borrowing a machine
//! actually needs.

use std::time::Instant;

/// Microseconds of validation granted per wall-clock second, per percent of a
/// core. One core fully spent is 1,000,000 us of work per second.
const US_PER_PERCENT: f64 = 10_000.0;

/// How much unspent budget may accumulate, as seconds of the refill rate.
///
/// Without a ceiling an idle pool banks its whole allowance -- an hour of
/// quiet becomes an hour of validation available in one burst, which is the
/// spike the budget exists to prevent. Two seconds is enough to absorb the
/// ordinary clumping of shares arriving together and short enough that it
/// cannot become a reservoir.
const BURST_SECONDS: f64 = 2.0;

/// A per-algorithm cost estimate and the tokens to pay it with.
pub struct Budget {
    tokens: f64,
    rate_us_per_s: f64,
    burst: f64,
    last: Instant,
    /// Microseconds a validation of this algorithm has been costing.
    cost: Vec<(String, f64)>,
    admitted: u64,
    denied: u64,
}

impl Budget {
    /// `percent` is percent of **one core**, and fractional on purpose.
    ///
    /// Integer percent was the first version and it is too coarse at the end
    /// that matters: one percent of a core is 10,000 us a second, which on a
    /// cheap algorithm is thousands of shares and cannot express "a little" on
    /// a small machine at all. A Raspberry Pi lending a fifth of a percent is
    /// a sentence somebody will want to say.
    ///
    /// Zero disables the budget rather than refusing everything: an operator
    /// who says they want no limit has said something coherent, and a pool
    /// that answered by validating nothing would look broken instead of
    /// unlimited.
    pub fn new(percent: f64) -> Budget {
        Budget::new_at(percent, Instant::now())
    }

    /// The same, against a supplied instant.
    ///
    /// `refill_at` and `admit_at` take one already, and say why: so a claim can
    /// drive the budget rather than sleep. **The constructor was the one place
    /// left reaching for the clock itself**, and that turned a claim which
    /// looks like pure arithmetic into one that measures how long the lines
    /// above it took to run.
    ///
    /// `cost_and_not_count_is_what_is_bounded` is where it surfaced. It takes
    /// `t0`, builds one budget, spends **416,666** admissions out of it, and
    /// only then builds the second -- by which point `Instant::now()` is a
    /// dozen milliseconds past `t0`, so `refill_at(at(t0, 1000))` credited less
    /// than the second it asked for and the budget admitted 51 yespower shares
    /// where the bench says 52. Arithmetic wearing a measurement's clothes,
    /// which is the shape `time::calibrate` records in the kernel: the check
    /// and the thing it checks were reading the same drifting clock.
    ///
    /// It fails by build profile and by machine speed rather than by anything
    /// about budgets, which is why it went unnoticed for as long as the pool's
    /// tests could not run at all -- `pool/.cargo/config.toml` names a Windows
    /// target, so a bare `cargo test` on a Linux runner looked for a std that
    /// was not there and never reached this.
    pub fn new_at(percent: f64, now: Instant) -> Budget {
        let percent = if percent.is_finite() && percent > 0.0 { percent } else { 0.0 };
        Budget {
            tokens: 0.0,
            rate_us_per_s: percent * US_PER_PERCENT,
            burst: percent * US_PER_PERCENT * BURST_SECONDS,
            last: now,
            cost: Vec::new(),
            admitted: 0,
            denied: 0,
        }
    }

    pub fn unlimited(&self) -> bool {
        self.rate_us_per_s <= 0.0
    }

    /// What a validation of this algorithm has been costing, in microseconds.
    pub fn estimate(&self, algo: &str) -> Option<f64> {
        self.cost.iter().find(|(a, _)| a == algo).map(|(_, c)| *c)
    }

    /// Refill against the clock. Separate from `admit` so a claim can drive it
    /// with a supplied instant rather than by sleeping.
    pub fn refill_at(&mut self, now: Instant) {
        let dt = now.saturating_duration_since(self.last).as_secs_f64();
        if dt <= 0.0 {
            return;
        }
        self.last = now;
        self.tokens = (self.tokens + dt * self.rate_us_per_s).min(self.burst);
    }

    /// Whether a share of this algorithm may be validated now.
    ///
    /// Spends the estimate up front rather than the true cost afterwards,
    /// because the decision has to be made before the work is done -- that is
    /// the whole point, since doing the work *is* the cost. `record` then
    /// corrects the estimate for next time.
    pub fn admit_at(&mut self, algo: &str, now: Instant) -> bool {
        if self.unlimited() {
            self.admitted += 1;
            return true;
        }
        self.refill_at(now);
        let want = match self.estimate(algo) {
            // Never timed. Admitted once so it can be, which is bounded by how
            // many algorithms exist rather than by anything a sender chooses.
            None => {
                self.admitted += 1;
                return true;
            }
            Some(c) => c,
        };
        if self.tokens >= want {
            self.tokens -= want;
            self.admitted += 1;
            true
        } else {
            self.denied += 1;
            false
        }
    }

    pub fn admit(&mut self, algo: &str) -> bool {
        self.admit_at(algo, Instant::now())
    }

    /// Fold a real measurement into the estimate.
    ///
    /// An exponential average with a heavy weight on history, because the
    /// quantity is a property of the machine and the algorithm rather than of
    /// this share: one validation that landed while the box was paging is not
    /// evidence that every future one will. It still tracks -- a machine that
    /// genuinely slows down moves the estimate within a few dozen shares.
    ///
    /// **The first sample is taken whole.** Averaging it against a zero it was
    /// never compared to would halve the very first estimate, and the first
    /// estimate is the one an unmeasured algorithm is admitted on the strength
    /// of.
    pub fn record(&mut self, algo: &str, micros: f64) {
        let micros = if micros.is_finite() && micros >= 0.0 { micros } else { 0.0 };
        match self.cost.iter_mut().find(|(a, _)| a == algo) {
            Some(e) => e.1 = e.1 * 0.9 + micros * 0.1,
            None => self.cost.push((String::from(algo), micros)),
        }
    }

    pub fn admitted(&self) -> u64 {
        self.admitted
    }

    pub fn denied(&self) -> u64 {
        self.denied
    }

    /// What the budget would allow per second, per algorithm, at the measured
    /// costs. The line an operator lending a machine actually wants.
    ///
    /// **An unlimited budget answers infinity and not zero.** `rate_us_per_s`
    /// is zero when there is no cap, so the division gave 0 and the line read
    /// "so 0 a second within the cap" -- which is the arithmetic of no budget
    /// wearing the words of the tightest possible one, and it is the operator
    /// of an *unbounded* pool who most needs to not be told that. Caught by
    /// running a flood against `--cpu-percent 0`, where the pool answered
    /// every share and reported it could answer none.
    pub fn report(&self) -> Vec<(String, f64, f64)> {
        let mut v: Vec<(String, f64, f64)> = self
            .cost
            .iter()
            .map(|(a, c)| {
                let per_s = if self.unlimited() || *c <= 0.0 {
                    f64::INFINITY
                } else {
                    self.rate_us_per_s / c
                };
                (a.clone(), *c, per_s)
            })
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn at(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    /// The whole point: a cheap algorithm and an expensive one are bounded by
    /// what they cost, not by a share count that treats them alike.
    #[test]
    fn cost_and_not_count_is_what_is_bounded() {
        let t0 = Instant::now();
        // 100% of one core: 1,000,000 us of validation per second.
        //
        // `new_at` and not `new`: both budgets have to measure from the same
        // `t0` the refills are expressed against, or the second one is
        // credited only for the time left after the first one's loop.
        let mut b = Budget::new_at(100.0, t0);
        b.record("sha256d", 2.4);
        b.record("yespower", 19_000.0);
        b.refill_at(at(t0, 1000));

        let mut cheap = 0;
        while b.admit_at("sha256d", at(t0, 1000)) {
            cheap += 1;
            if cheap > 1_000_000 {
                break;
            }
        }
        let mut b2 = Budget::new_at(100.0, t0);
        b2.record("yespower", 19_000.0);
        b2.refill_at(at(t0, 1000));
        let mut dear = 0;
        while b2.admit_at("yespower", at(t0, 1000)) {
            dear += 1;
            if dear > 1_000_000 {
                break;
            }
        }
        // **And the arithmetic lands on the measured figures, which is the
        // check worth having.** One second of budget at 100% of a core buys
        // 1e6 us of validation. `glados-pool --bench` on the machine those
        // costs came from reports 423,190 sha256d shares a second on one core
        // and 52 yespower ones -- and 1e6/2.4 is 416,666 while 1e6/19,000 is
        // 52. The budget is not approximating the bench; it is the same
        // division, so a bound derived here means what the bench measured.
        assert!(
            (400_000..430_000).contains(&cheap),
            "cheap shares admitted: {cheap}, against ~416,666 expected"
        );
        assert_eq!(dear, 52, "expensive shares admitted, against the bench's 52 a second");
    }

    /// An idle pool must not bank its allowance and spend it in one burst,
    /// which is the spike the budget exists to prevent.
    #[test]
    fn an_idle_pool_does_not_hoard_budget() {
        let t0 = Instant::now();
        let mut b = Budget::new_at(100.0, t0);
        b.record("yespower", 19_000.0);
        // An hour of quiet.
        b.refill_at(at(t0, 3_600_000));
        let mut n = 0;
        while b.admit_at("yespower", at(t0, 3_600_000)) {
            n += 1;
            if n > 10_000 {
                break;
            }
        }
        // The cap is two seconds of rate, not an hour of it.
        assert!(n < 250, "burst was not capped: {n} admitted");
    }

    /// An algorithm nobody has timed is admitted so it can be, and once it has
    /// been it is budgeted like everything else.
    #[test]
    fn an_unmeasured_algorithm_is_admitted_once_and_then_priced() {
        let t0 = Instant::now();
        let mut b = Budget::new_at(1.0, t0);
        assert!(b.admit_at("mystery", t0), "the first is free, or it is never measured");
        b.record("mystery", 500_000.0);
        // 1% of a core is 10,000 us/s, burst 20,000 -- a 500 ms validation
        // cannot fit however long it waits.
        b.refill_at(at(t0, 60_000));
        assert!(!b.admit_at("mystery", at(t0, 60_000)));
        assert_eq!(b.denied(), 1);
    }

    /// The first sample is the estimate. Averaging it against a zero it was
    /// never compared to would halve it, and the first estimate is the one an
    /// unmeasured algorithm is admitted on the strength of.
    #[test]
    fn the_first_measurement_is_taken_whole() {
        let mut b = Budget::new(50.0);
        b.record("a", 1000.0);
        assert_eq!(b.estimate("a"), Some(1000.0));
        // And it tracks afterwards rather than jumping.
        b.record("a", 2000.0);
        let e = b.estimate("a").unwrap();
        assert!(e > 1000.0 && e < 1200.0, "estimate moved to {e}");
    }

    /// Zero percent is "no limit", which is a coherent thing to ask for. A
    /// pool that answered it by validating nothing would look broken instead.
    #[test]
    fn zero_percent_means_unlimited_rather_than_nothing() {
        let mut b = Budget::new(0.0);
        assert!(b.unlimited());
        b.record("yespower", 19_000.0);
        for _ in 0..10_000 {
            assert!(b.admit("yespower"));
        }
        assert_eq!(b.denied(), 0);
    }

    /// A nonsense measurement must not poison the estimate into admitting
    /// everything for ever.
    #[test]
    fn a_nonsense_measurement_is_refused_rather_than_stored() {
        let mut b = Budget::new(50.0);
        b.record("a", f64::NAN);
        assert_eq!(b.estimate("a"), Some(0.0));
        b.record("b", -5.0);
        assert_eq!(b.estimate("b"), Some(0.0));
    }
}

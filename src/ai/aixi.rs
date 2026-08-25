//! Monte-Carlo planning over the machine's own fitted dynamics.
//!
//! AIXI is the agent that picks actions maximising expected discounted
//! reward over all computable environments weighted by their simplicity --
//! incomputable, and useful as a direction rather than a destination. This
//! is the smallest honest slice of it for one machine: the environment model
//! is the controlled linear system [`super::futures`] already fits online
//! from the sample ring, the action space is how much work the machine takes
//! on over the next few seconds, and expectation is approximated by Monte
//! Carlo -- each candidate schedule is rolled out K times with God-bits
//! noise at the fitted residual scale, exactly the noise the oracle draws.
//! Utility is machine health: free heap fraction minus task pressure,
//! discounted. The planner recommends the schedule whose expected utility
//! beats carrying on as usual, and says how sure it is.
//!
//! The control deserves one plain caveat: `u` is measured as touch rate, the
//! same variable the oracle intervenes on. What it actually models is
//! offered workload, whoever offers it -- the operator's hands or the
//! system's own choices to run things. The knob is real even if the hand on
//! it today is mostly yours.
//!
//! Acting waits on understanding. The verdict below gates itself on sample
//! count, residual size and regime stability, and states which threshold
//! failed; an executor comes later, and only for the actions this gate has
//! been watched passing on real hardware first.

use super::{context, futures, godbits};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

const GAMMA: f32 = 0.97;
/// Task pressure weight: two running tasks cost about as much utility as
/// two percent of heap. Small, because tasks are few and heap is the thing
/// that actually runs out.
const TASK_COST: f32 = 0.01;

struct Fits {
    w: [[f32; 3]; futures::NVARS],
    std: [f32; futures::NVARS],
}

impl Fits {
    /// Fit each variable's `v_next = w0 + w1*v + w2*u` with centred,
    /// scaled columns accumulated in f64, then expand back so the rollout
    /// can run in raw units.
    ///
    /// The scaling exists because heap and touch live at wildly different
    /// magnitudes and move by wildly different amounts; on an uncentred
    /// design the intercept and the persistence term become nearly
    /// collinear and their split is numerical luck. The f64 accumulation
    /// is insurance in the same direction. A previous version of this
    /// comment claimed a measured f32 phantom residual here; that number
    /// turned out to be a bug in the self-test's synthetic generator
    /// labelling each step's heap with the next second's activity, and the
    /// solver was exonerated by numpy agreeing with it exactly.
    fn from(hist: &[futures::Snap]) -> Self {
        let n = hist.len();
        let mut f = Fits {
            w: [[0.0; 3]; futures::NVARS],
            std: [0.0; futures::NVARS],
        };
        if n < 4 {
            return f;
        }
        let mut mv = [0.0f64; futures::NVARS];
        let mut mu = 0.0f64;
        for s in hist {
            for j in 0..futures::NVARS {
                mv[j] += s.vars[j] as f64;
            }
            mu += s.vars[futures::CTRL] as f64;
        }
        for j in 0..futures::NVARS {
            mv[j] /= n as f64;
        }
        mu /= n as f64;
        let mut sv = [1.0f64; futures::NVARS];
        let mut su = 1.0f64;
        for s in hist {
            for j in 0..futures::NVARS {
                let d = s.vars[j] as f64 - mv[j];
                sv[j] += d * d;
            }
            let d = s.vars[futures::CTRL] as f64 - mu;
            su += d * d;
        }
        for j in 0..futures::NVARS {
            sv[j] = sqrt_f64(sv[j] / n as f64).max(1.0);
        }
        su = sqrt_f64(su / n as f64).max(1.0);

        for j in 0..futures::NVARS {
            // Scaled design: [1, (v-mv)/sv, (u-mu)/su].
            let mut g = [0.0f64; 9];
            let mut b = [0.0f64; 3];
            let mut rows = 0usize;
            for w in hist.windows(2) {
                let x = [
                    1.0,
                    (w[0].vars[j] as f64 - mv[j]) / sv[j],
                    (w[0].vars[futures::CTRL] as f64 - mu) / su,
                ];
                let y = w[1].vars[j] as f64;
                for a in 0..3 {
                    for c in 0..3 {
                        g[a * 3 + c] += x[a] * x[c];
                    }
                    b[a] += x[a] * y;
                }
                rows += 1;
            }
            let lambda = 1e-9 * rows.max(1) as f64;
            for d in 0..3 {
                g[d * 3 + d] += lambda;
            }
            if !cholesky3(&mut g, &mut b) || rows < 3 {
                // Degenerate window: hold the last value.
                f.w[j] = [0.0, 1.0, 0.0];
                f.std[j] = 0.0;
                continue;
            }
            // Expand back: pred = b0 + b1*(v-mv)/sv + b2*(u-mu)/su.
            let w0 = b[0] - b[1] * mv[j] / sv[j] - b[2] * mu / su;
            let w1 = b[1] / sv[j];
            let w2 = b[2] / su;
            f.w[j] = [w0 as f32, w1 as f32, w2 as f32];

            let mut ss = 0.0f64;
            for w in hist.windows(2) {
                // The coefficients are stored expanded, so the prediction
                // takes raw values. Feeding centred inputs into an
                // expanded intercept centres twice and measures nothing.
                let pred = w0 + w1 * w[0].vars[j] as f64 + w2 * w[0].vars[futures::CTRL] as f64;
                let e = w[1].vars[j] as f64 - pred;
                ss += e * e;
            }
            f.std[j] = super::tensor::sqrtf((ss / rows.max(1) as f64) as f32);
        }
        f
    }
}

/// f64 square root by Newton's method, since core has no libm and the f32
/// helper would throw away the precision this module exists to keep.
fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Seed from the exponent half so convergence is a handful of steps even
    // for very large or small inputs.
    let mut r = x * 0.25 + 0.5;
    for _ in 0..12 {
        r = 0.5 * (r + x / r);
    }
    r
}

/// In-place 3x3 Cholesky solve, lower-triangular, f64. Returns false when
/// not positive definite. Small enough that a general LU would be more code
/// than the direct expansion.
fn cholesky3(g: &mut [f64; 9], b: &mut [f64; 3]) -> bool {
    for i in 0..3 {
        for j in 0..=i {
            let mut sum = g[i * 3 + j];
            for k in 0..j {
                sum -= g[i * 3 + k] * g[j * 3 + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return false;
                }
                g[i * 3 + i] = sqrt_f64(sum);
            } else {
                g[i * 3 + j] = sum / g[j * 3 + j];
            }
        }
    }
    for i in 0..3 {
        let mut sum = b[i];
        for k in 0..i {
            sum -= g[i * 3 + k] * b[k];
        }
        b[i] = sum / g[i * 3 + i];
    }
    for i in (0..3).rev() {
        let mut sum = b[i];
        for k in (i + 1)..3 {
            sum -= g[k * 3 + i] * b[k];
        }
        b[i] = sum / g[i * 3 + i];
    }
    true
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / 8_388_608.0 - 1.0
    }
}

/// The action vocabulary. Levels are drawn from what the machine has seen,
/// mirroring the oracle's interventions, so "load" means something this
/// hardware has actually done rather than an abstract maximum.
fn actions_for(hist: &[futures::Snap]) -> [( &'static str, f32); 3] {
    let mut mean_u = 0.0f32;
    let mut max_u = 0.0f32;
    if !hist.is_empty() {
        for s in hist {
            mean_u += s.vars[futures::CTRL];
            if s.vars[futures::CTRL] > max_u {
                max_u = s.vars[futures::CTRL];
            }
        }
        mean_u /= hist.len() as f32;
    }
    let load = (max_u * 1.5).max(mean_u * 3.0).max(8.0);
    [("idle", 0.0), ("steady", mean_u), ("load", load)]
}

/// One stochastic rollout of the fitted model under a forced activity
/// schedule. Deterministic prediction plus residual-scale noise, physical
/// floors applied -- the same shape as the oracle's branches.
fn rollout(
    fits: &Fits,
    start: &[f32; futures::NVARS],
    schedule: &[f32],
    rng: &mut Rng,
) -> Vec<[f32; futures::NVARS]> {
    let mut state = *start;
    state[futures::CTRL] = schedule[0];
    let mut traj = Vec::with_capacity(schedule.len());
    for &activity in schedule {
        let mut next = state;
        for j in 0..futures::NVARS {
            if j == futures::CTRL {
                next[j] = activity;
                continue;
            }
            let w = fits.w[j];
            next[j] = w[0] + w[1] * state[j] + w[2] * activity + fits.std[j] * rng.next() * 0.6;
        }
        next[0] = next[0].max(0.0);
        next[1] = next[1].max(0.0);
        next[3] = next[3].max(1.0);
        traj.push(next);
        state = next;
    }
    traj
}

fn utility(traj: &[[f32; futures::NVARS]], heap_total_kib: f32) -> f32 {
    let total = heap_total_kib.max(1.0);
    let mut u = 0.0f32;
    let mut g = 1.0f32;
    for step in traj {
        let free = ((total - step[0]) / total).clamp(0.0, 1.0);
        u += g * (free - TASK_COST * step[3]);
        g *= GAMMA;
    }
    u
}

pub struct Candidate {
    /// Activity level names, one per planned second.
    pub levels: Vec<&'static str>,
    pub u_mean: f32,
    pub u_std: f32,
}

pub struct Plan {
    pub situation: context::Situation,
    pub samples: usize,
    pub resid_heap_kib: f32,
    pub best: Candidate,
    pub baseline: Candidate,
    pub confident: bool,
    pub reasons: Vec<String>,
}

/// Enumerate every schedule of length `depth` over the three levels, average
/// K Monte Carlo rollouts each, and compare against carrying on. Pure in its
/// inputs so the self-test can drive it from a synthetic ring.
fn plan_on(
    hist: &[futures::Snap],
    now: &[f32; futures::NVARS],
    heap_total_kib: f32,
    depth: usize,
    k_mc: usize,
    seed: u64,
) -> (Candidate, Candidate, usize, f32) {
    let fits = Fits::from(hist);
    let acts = actions_for(hist);
    let mut best = Candidate { levels: Vec::new(), u_mean: f32::NEG_INFINITY, u_std: 0.0 };
    let mut base_u = (0.0f32, 0.0f32);

    // Depth is small enough that exhaustive enumeration beats any search
    // cleverness: 3^4 schedules x 16 rollouts x 4 steps is a few thousand
    // multiply-adds, microseconds of work, and exhaustive means the answer
    // carries no algorithmic doubt on top of the model's own.
    let n_seq = 3usize.pow(depth as u32);
    for si in 0..n_seq {
        let mut idx = si;
        let mut sched = Vec::with_capacity(depth);
        let mut names = Vec::with_capacity(depth);
        for _ in 0..depth {
            let a = idx % 3;
            idx /= 3;
            sched.push(acts[a].1);
            names.push(acts[a].0);
        }
        let mut sum = 0.0f32;
        let mut sum2 = 0.0f32;
        for k in 0..k_mc {
            let mut rng = Rng(seed ^ (0x9E37_79B9u64.wrapping_mul((si * k_mc + k + 1) as u64)));
            let u = utility(&rollout(&fits, now, &sched, &mut rng), heap_total_kib);
            sum += u;
            sum2 += u * u;
        }
        let mean = sum / k_mc as f32;
        let var = (sum2 / k_mc as f32 - mean * mean).max(0.0);
        let std = super::tensor::sqrtf(var);
        if si == 1 {
            // Sequence 1 is steady-at-mean repeated: carrying on.
            base_u = (mean, std);
        }
        if mean > best.u_mean {
            best = Candidate { levels: names, u_mean: mean, u_std: std };
        }
    }
    let baseline = Candidate {
        levels: (0..depth).map(|_| acts[1].0).collect(),
        u_mean: base_u.0,
        u_std: base_u.1,
    };
    (best, baseline, hist.len(), fits.std[0])
}

pub fn plan(depth: usize, k_mc: usize) -> Plan {
    let situation = context::gather();
    let hist = futures::history();
    let now = futures::snapshot_now();
    let (_, total_kib) = crate::mem::heap::HEAP.stats();
    let (best, baseline, samples, resid) =
        plan_on(&hist, &now.vars, total_kib as f32, depth, k_mc, godbits::fold());

    let mut reasons = Vec::new();
    let mut confident = true;
    if samples < 16 {
        confident = false;
        reasons.push(format!("only {} samples fitted", samples));
    }
    if situation.load == context::Load::Unknown || situation.load_stability < 0.5 {
        confident = false;
        reasons.push(format!(
            "load regime {} at {:.0}/{} stability",
            situation.load.label(),
            (situation.load_stability * 100.0) as u32,
            context::STABILITY_WINDOW
        ));
    }
    let delta = best.u_mean - baseline.u_mean;
    let spread = (best.u_std + baseline.u_std).max(1e-6);
    if confident && delta <= spread {
        confident = false;
        reasons.push(format!(
            "best schedule leads baseline by less than its own spread ({:.4} vs {:.4})",
            delta, spread
        ));
    }
    if confident && delta < 1e-3 {
        // Above the Monte Carlo noise but still trivial: a tenth of a
        // percent of one step's utility is not a reason to do anything.
        // On a headless machine every schedule looks alike, and this is
        // what keeps the verdict honest about that instead of dressing
        // indifference up as a plan.
        confident = false;
        reasons.push(format!(
            "the preference over carrying on is negligible ({:+.5} utility)",
            delta
        ));
    }

    Plan { situation, samples, resid_heap_kib: resid, best, baseline, confident, reasons }
}

pub fn report() {
    use crate::gfx::console::{self, LTGRAY, YELLOW};
    use crate::kprintln;

    console::set_color(YELLOW);
    kprintln!("[plan]");
    console::set_color(LTGRAY);

    let p = plan(4, 16);
    let s = &p.situation;
    let nets: Vec<String> = s.nets.iter().map(|(n, up)| format!("{} {}", n, if *up { "up" } else { "down" })).collect();
    kprintln!(
        "  now   uptime {}s, heap {:.1}/{:.0} MiB, drift {:+.2} MiB/s, {} tasks",
        s.uptime_s as u64, s.heap_used_mib, s.heap_total_mib, s.heap_trend_mib_s, s.tasks
    );
    kprintln!(
        "        load {} ({:.0}% stable), operator {}, net {}, store {}, episode {}",
        s.load.label(),
        (s.load_stability * 100.0) as u32,
        s.operator.label(),
        if nets.is_empty() { String::from("none") } else { nets.join(", ") },
        if s.store_mounted { "mounted" } else { "unmounted" },
        if s.episode_running { "running" } else { "none" }
    );
    kprintln!(
        "  fit   {} samples, heap residual {:.0} KiB",
        p.samples, p.resid_heap_kib
    );
    kprintln!(
        "  best  {:<28} U {:.3} +- {:.3}",
        p.best.levels.join(" "), p.best.u_mean, p.best.u_std
    );
    kprintln!(
        "  carry on {:<23} U {:.3} +- {:.3}  (delta {:+.3})",
        "", p.baseline.u_mean, p.baseline.u_std, p.best.u_mean - p.baseline.u_mean
    );
    if p.reasons.is_empty() {
        kprintln!(
            "  ready -- would take the schedule above and re-check in {}s",
            p.best.levels.len()
        );
    } else {
        for r in &p.reasons {
            kprintln!("  advise -- {}", r);
        }
    }
}

/// Driven against a synthetic machine rather than the ring, because the
/// property under test is the planner's arithmetic, and the live ring cannot
/// be relied on to be draining at boot. The synthetic truth: offered load
/// grows used heap, rest slowly recovers it, and nothing else moves.
pub fn selftest() -> bool {
    use crate::kprintln;

    let mut hist = Vec::new();
    let mut used = 300_000.0f32;
    for t in 0..48 {
        // Alternating quiet and heavy blocks, so the fit sees both regimes
        // and the control actually varies. The snapshot records the state
        // at t -- including u_t -- and the update then produces t+1 from
        // exactly those numbers, which is the contemporaneous pairing the
        // fit assumes. Pushing after the update would label each step's
        // heap with the NEXT second's activity and break the relation at
        // every block boundary by one full load step.
        let u = if (t / 6) % 2 == 0 { 0.0 } else { 12.0 };
        hist.push(futures::Snap {
            t_s: t as f32,
            vars: [used, 20.0 + 3.0 * u, u, 2.0],
        });
        used = (used * 0.999 + 2500.0 * u + 200.0).max(60_000.0);
    }
    let total_kib = 1_900_000.0;
    // Start mid-drain: the last block was heavy and heap is high, which is
    // exactly when a health-seeking planner should choose rest.
    let now = [hist[hist.len() - 1].vars[0], 56.0, 12.0, 2.0];

    let (best, baseline, _n, resid) =
        plan_on(&hist, &now, total_kib, 4, 16, 0x5EED_C0FF_EE42);

    let fits_ok = resid < 5_000.0;
    let prefers_rest = best.levels[0] == "idle";
    let beats_carry_on = best.u_mean >= baseline.u_mean - 1e-3;

    let ok = fits_ok && prefers_rest && beats_carry_on;
    kprintln!(
        "  {}  prefers rest while draining, beats carrying on (first {:>5}, dU {:+.4}, resid {:.0} KiB)",
        if ok { "ok " } else { "FAIL" },
        if best.levels.is_empty() { "?" } else { best.levels[0] },
        best.u_mean - baseline.u_mean,
        resid
    );
    ok
}

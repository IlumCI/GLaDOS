//! The machine's own future, projected -- God Says, made honest.
//!
//! TempleOS drew uniform words and let the operator hear prophecy. This keeps
//! the entropy (the timing of the operator's hands, [`super::godbits`]) and
//! the ritual, and changes the subject to the one future that is actually
//! knowable: this machine's. Uptime, heap, task switching and the operator's
//! own touch rate are measured once a second into a ring, a linear dynamical
//! model is fitted online from that history, and the state is rolled forward
//! under three interventions -- left alone, carried on, put under load. Same
//! present, forked decisions, divergent timelines.
//!
//! It is causal in the technical sense and not the mystical one. The fit is
//! `v_next = a + b*v + c*u`, a controlled linear system whose control `u` is
//! the operator's activity; a branch is the counterfactual `do(u := level)`,
//! the state rolled forward with that activity forced. What the branches show
//! is what the machine *does* under each choice of how hard it is used, drawn
//! from how it has actually behaved since boot. It predicts the machine, from
//! the machine. The noise injected each step is the God-bits fold, so the
//! reading still depends on every key and every mouse move the machine has
//! felt -- the operator still puppets the divination, now over real state.
//!
//! The model is the one the router already trusts: Widrow-Hoff normal
//! equations solved by Cholesky ([`super::probe`]). A 3x3 fit per variable,
//! ridged so a still machine with no variation still yields a usable (flat)
//! prediction rather than a singular matrix.

use super::{godbits, probe};
use crate::sync::Racy;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

/// State variables tracked and projected. The touch rate is both a variable
/// and the control, which is what makes the interventions mean something.
pub const NVARS: usize = 4;
pub const VAR_NAMES: [&str; NVARS] = ["heap KiB", "switch/s", "touch/s", "tasks"];
/// The control variable's index: the operator's touch rate.
pub(crate) const CTRL: usize = 2;

#[derive(Clone, Copy)]
pub struct Snap {
    /// Uptime in seconds -- the time axis, not projected, just advanced.
    pub t_s: f32,
    pub vars: [f32; NVARS],
}

const HIST: usize = 64;
static RING: Racy<[Snap; HIST]> = Racy::new([Snap { t_s: 0.0, vars: [0.0; NVARS] }; HIST]);
static HEAD: AtomicUsize = AtomicUsize::new(0);
static FILLED: AtomicUsize = AtomicUsize::new(0);
/// Cumulative counters at the previous sample, for turning totals into rates.
static LAST: Racy<(u64, u64)> = Racy::new((0, 0));

/// Read the machine now. Called about once a second from the clock task, so
/// the deltas below are per-second rates without needing a clock of their own.
pub fn sample() {
    // ticks() counts timer interrupts, which fire at TIMER_HZ (100/s) -- not
    // the calibrated APIC frequency timer_hz() reports, which is millions and
    // would put every uptime at zero. This is the exact confusion the `mem`
    // command avoids by dividing ticks by TIMER_HZ.
    let t_s = crate::dev::lapic::ticks() as f32 / crate::TIMER_HZ as f32;

    let (used, _total) = crate::mem::heap::HEAP.stats();
    let switches = crate::task::total_switches();
    let touches = godbits::felt() as u64;

    let (ls, lt) = unsafe { *LAST.get() };
    let dsw = switches.saturating_sub(ls) as f32;
    let dto = touches.saturating_sub(lt) as f32;
    unsafe { *LAST.get() = (switches, touches) };

    // The first sample has no previous total to difference against, so its
    // rates are garbage; overwrite them to zero rather than let a boot-time
    // spike anchor the fit.
    let first = FILLED.load(Ordering::Relaxed) == 0;
    let snap = Snap {
        t_s,
        vars: [
            (used / 1024) as f32,
            if first { 0.0 } else { dsw },
            if first { 0.0 } else { dto },
            crate::task::count() as f32,
        ],
    };

    let h = HEAD.fetch_add(1, Ordering::Relaxed) % HIST;
    unsafe { (*RING.get())[h] = snap };
    let f = FILLED.load(Ordering::Relaxed);
    if f < HIST {
        FILLED.store(f + 1, Ordering::Relaxed);
    }
}

/// The most recent snapshot, or a live read if nothing has been sampled yet.
pub fn snapshot_now() -> Snap {
    if FILLED.load(Ordering::Relaxed) == 0 {
        let (used, _) = crate::mem::heap::HEAP.stats();
        return Snap {
            t_s: crate::dev::lapic::ticks() as f32 / crate::TIMER_HZ as f32,
            vars: [(used / 1024) as f32, 0.0, 0.0, crate::task::count() as f32],
        };
    }
    let n = FILLED.load(Ordering::Relaxed);
    let head = HEAD.load(Ordering::Relaxed);
    let last = (head + HIST - 1) % HIST;
    let _ = n;
    unsafe { (*RING.get())[last] }
}

/// History oldest-first, for the solid part of the graph and for fitting.
pub(crate) fn history() -> Vec<Snap> {
    let n = FILLED.load(Ordering::Relaxed).min(HIST);
    let head = HEAD.load(Ordering::Relaxed);
    let mut out = Vec::new();
    let ring = unsafe { &*RING.get() };
    for i in 0..n {
        // head points one past the newest; oldest is head-n.
        let idx = (head + HIST - n + i) % HIST;
        out.push(ring[idx]);
    }
    out
}

/// One projected timeline under a forced activity level.
pub struct Branch {
    pub name: &'static str,
    /// The forced control -- touches per second the operator would make.
    /// Part of the projection's description; read by callers that explain a
    /// branch rather than just plot it.
    #[allow(dead_code)]
    pub activity: f32,
    /// Projected variable values per step, `steps` long.
    pub traj: Vec<[f32; NVARS]>,
}

pub struct Projection {
    pub hist: Vec<Snap>,
    pub branches: Vec<Branch>,
    pub steps: usize,
    /// Seconds per projected step. One, matching the sampler; kept explicit so
    /// a caller labelling the time axis does not assume it.
    #[allow(dead_code)]
    pub dt_s: f32,
    pub felt: usize,
    /// True when there was enough history to fit; false means the branches are
    /// a flat hold and the reading says so.
    pub fitted: bool,
}

/// Fit `v_next = w0 + w1*v + w2*u` for one variable by ridge-regularised
/// normal equations, solved with the router's Cholesky. Returns the three
/// weights and the residual standard deviation (the honest noise scale).
fn fit_var(hist: &[Snap], j: usize) -> ([f32; 3], f32) {
    // Gram matrix of [1, v, u] and rhs against next-step v.
    let mut g = [0.0f32; 9];
    let mut b = [0.0f32; 3];
    let mut rows = 0usize;
    for w in hist.windows(2) {
        let x = [1.0, w[0].vars[j], w[0].vars[CTRL]];
        let y = w[1].vars[j];
        for a in 0..3 {
            for c in 0..3 {
                g[a * 3 + c] += x[a] * x[c];
            }
            b[a] += x[a] * y;
        }
        rows += 1;
    }
    // Ridge: a still machine gives a near-singular gram (v and u barely vary),
    // and without this the solve fails and the projection is a straight
    // nothing. lambda keeps it invertible and biases toward "stays put".
    let lambda = 1e-3 * rows.max(1) as f32;
    for d in 0..3 {
        g[d * 3 + d] += lambda;
    }
    let mut weights = b;
    if rows < 3 || !probe::ridge_solve(&mut g, 3, &mut weights) {
        // Not enough to fit: hold the last value.
        return ([0.0, 1.0, 0.0], 0.0);
    }
    // Residual std over the fit window -- the spread the model could not
    // explain, which is exactly how far the God-bits noise should reach.
    let mut ss = 0.0f32;
    for w in hist.windows(2) {
        let pred = weights[0] + weights[1] * w[0].vars[j] + weights[2] * w[0].vars[CTRL];
        let e = w[1].vars[j] - pred;
        ss += e * e;
    }
    let std = super::tensor::sqrtf(ss / rows.max(1) as f32);
    (weights, std)
}

/// Project the machine forward under three interventions.
pub fn project(steps: usize) -> Projection {
    let hist = history();
    let felt = godbits::felt();
    let now = snapshot_now();

    let fitted = hist.len() >= 4;
    let mut fits = [([0.0f32; 3], 0.0f32); NVARS];
    if fitted {
        for j in 0..NVARS {
            fits[j] = fit_var(&hist, j);
        }
    }

    // The activity levels the three futures assume. Drawn from what the
    // machine has actually seen: zero, the recent mean, and a heavy multiple.
    let mut mean_u = 0.0f32;
    let mut max_u = 0.0f32;
    if !hist.is_empty() {
        for s in &hist {
            mean_u += s.vars[CTRL];
            if s.vars[CTRL] > max_u {
                max_u = s.vars[CTRL];
            }
        }
        mean_u /= hist.len() as f32;
    }
    let load_u = (max_u * 1.5).max(mean_u * 3.0).max(8.0);
    let levels: [(&str, f32); 3] = [
        ("left alone", 0.0),
        ("carried on", mean_u),
        ("put under load", load_u),
    ];

    // One RNG seeded from the operator's whole history of touches, split per
    // branch so the three timelines differ under the same past.
    let base_seed = godbits::fold();

    let mut branches = Vec::new();
    for (bi, (name, activity)) in levels.iter().enumerate() {
        let mut seed = base_seed ^ (0x9E37_79B9u64.wrapping_mul(bi as u64 + 1));
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            // symmetric in [-1, 1]
            (seed >> 40) as f32 / 8_388_608.0 - 1.0
        };

        let mut state = now.vars;
        // The control is forced -- this is the intervention.
        state[CTRL] = *activity;
        let mut traj = Vec::with_capacity(steps);
        for _ in 0..steps {
            let mut next = state;
            for j in 0..NVARS {
                if j == CTRL {
                    next[j] = *activity; // held by the do-operation
                    continue;
                }
                let (w, std) = fits[j];
                let pred = w[0] + w[1] * state[j] + w[2] * *activity;
                next[j] = pred + std * rng() * 0.6;
            }
            // Physical floors: none of these can go negative, and tasks are
            // whole. A projection that predicts -3 KiB of heap is a projection
            // that has stopped meaning anything.
            next[0] = next[0].max(0.0);
            next[1] = next[1].max(0.0);
            next[3] = next[3].max(1.0);
            traj.push(next);
            state = next;
        }
        branches.push(Branch { name, activity: *activity, traj });
    }

    Projection { hist, branches, steps, dt_s: 1.0, felt, fitted }
}

/// Print a projection into the terminal -- the headless face, and what
/// `drive.py` verifies.
pub fn futures_report() {
    use crate::gfx::console::{self, LTGRAY, YELLOW};
    use crate::kprintln;

    console::set_color(YELLOW);
    kprintln!("[oracle] the machine's own futures");
    console::set_color(LTGRAY);

    let pr = project(20);
    let now = snapshot_now();
    kprintln!(
        "  now at {}s: heap {} KiB, {}/s switch, {}/s touch, {} tasks",
        now.t_s as u64,
        now.vars[0] as u64,
        now.vars[1] as u64,
        now.vars[2] as u64,
        now.vars[3] as u64
    );
    if !pr.fitted {
        kprintln!("  too little history to fit -- give it a few seconds of running");
    }
    // The end-state of each timeline, which is the reading.
    for b in &pr.branches {
        let end = b.traj.last().copied().unwrap_or(now.vars);
        kprintln!(
            "  if {:<14} +{}s -> heap {} KiB, {}/s switch, {} tasks",
            b.name,
            pr.steps,
            end[0] as u64,
            end[1] as u64,
            end[3] as u64
        );
    }
    kprintln!(
        "  fitted from {} samples, fed by {} touches. the machine, from the machine.",
        pr.hist.len(),
        pr.felt
    );
}

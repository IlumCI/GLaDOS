//! The trainer's arithmetic core: Adam, restricted cross-entropy, and the
//! proof that they move a QDoRA site's loss to zero.
//!
//! Two halves, and the order between them is the point. The first is
//! engine-free: Adam, the two cross-entropies, and a self-test that drives
//! them against a fixed sheet until the loss is zero. It owns the pieces
//! whose bugs would otherwise hide inside runs too expensive to debug --
//! the optimiser's bias correction, the masked softmax's exactness, and
//! whether gradients + Adam + CE compose into "a small fixed dataset can be
//! memorised", which is the smallest honest definition of "the trainer
//! works".
//!
//! The second half is the loop that drives all of that against the real
//! model, the real corpus and a clock. It is written second and it is
//! checked second: every number it reports rests on arithmetic that was
//! already proven without it.
//!
//! The restricted part matters more than it looks: at decision points the
//! grammar admits a few dozen tokens out of ~150,000, so the softmax runs
//! over the reachable set only -- which is exact for masked targets, not an
//! approximation -- and costs what a few dozen cost instead of what a
//! vocabulary costs. The same trick that keeps inference cheap keeps
//! training aimed exactly at the behaviour being measured.

use super::adapter::{Adapters, Dora};
use super::model::Config;
use super::constrain::{step_bound, Alphabet, Cursor, Grammar};
use super::tensor::{expf, sqrtf};
use super::weights::Mat;
use alloc::vec;
use alloc::vec::Vec;

/// Adam with bias correction, over one flat parameter vector. Ported in
/// shape from the streaming engine's trainer: moments resident, parameters
/// mutated in place, nothing allocated per step.
pub struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u64,
}

impl Adam {
    pub fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }

    pub fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        self.t += 1;
        // Bias correction powers: b1^t and b2^t by squaring, since no libm.
        let bc1 = 1.0 - pow_f32(b1, self.t);
        let bc2 = 1.0 - pow_f32(b2, self.t);
        for k in 0..params.len() {
            let g = grads[k];
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g;
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * g * g;
            let mh = self.m[k] / bc1;
            let vh = self.v[k] / bc2;
            params[k] -= lr * mh / (sqrtf(vh) + eps);
        }
    }
}

/// Restricted cross-entropy: `-log softmax(logits over candidates)[target]`.
///
/// Returns the loss and the full-vocabulary gradient -- softmax probability
/// minus the one-hot inside the candidate set, exactly zero outside it,
/// because unreachable tokens cannot be blamed for a decision the grammar
/// never allowed them to make.
pub fn restricted_ce(logits: &[f32], cands: &[u32], target_idx: usize) -> (f32, Vec<f32>) {
    let mut max = f32::NEG_INFINITY;
    for &c in cands {
        if logits[c as usize] > max {
            max = logits[c as usize];
        }
    }
    let mut sum = 0.0f32;
    let mut probs = vec![0.0f32; cands.len()];
    for (i, &c) in cands.iter().enumerate() {
        let e = expf(logits[c as usize] - max);
        probs[i] = e;
        sum += e;
    }
    let loss = -logf(probs[target_idx] / sum);
    let mut grad = vec![0.0f32; logits.len()];
    for (i, &c) in cands.iter().enumerate() {
        grad[c as usize] = probs[i] / sum - if i == target_idx { 1.0 } else { 0.0 };
    }
    (loss, grad)
}

/// Restricted cross-entropy over an already-gathered candidate set.
///
/// The same function as `restricted_ce`, with the vocabulary taken out of it:
/// `logits` holds only the candidates, in candidate order, and the gradient
/// comes back the same shape. `restricted_ce` returns a full-width gradient
/// because its caller indexes by token id, which is right for a decision made
/// against `State::logits` -- but the training loop gathers its candidates
/// anyway, and a 151,936-wide allocation per decision per epoch to carry a few
/// dozen non-zeros would cost more than the arithmetic it wraps.
///
/// The self-test asserts the two agree rather than assuming it. They compute
/// the same thing by construction, which is exactly the sort of claim that
/// stops being true one edit later.
pub fn restricted_ce_compact(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    let mut probs = vec![0.0f32; logits.len()];
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        let ex = expf(v - max);
        probs[i] = ex;
        sum += ex;
    }
    let loss = -logf(probs[target] / sum);
    for (i, p) in probs.iter_mut().enumerate() {
        *p = *p / sum - if i == target { 1.0 } else { 0.0 };
    }
    (loss, probs)
}

/// The policy gradient of `E_{a~pi}[r(a)]`, **kept because it does not work
/// and the measurement is the reason the shipped objective has the shape it
/// has.** Nothing on the training path calls this.
///
/// Maximising `J = sum_i pi_i r_i` at one decode step gives
///
/// ```text
/// dJ/dz_k = sum_i r_i dpi_i/dz_k = pi_k (r_k - rbar),   rbar = sum_i pi_i r_i
/// ```
///
/// and `gy` is descended, so the loss gradient is the negative of that.
/// Correct, and useless here. With a binary reward -- one for the labelled
/// token, zero elsewhere -- `rbar` is `pi_t` and **every component collapses
/// to the cross-entropy gradient multiplied by `pi(target)`**, exactly, which
/// `reward_is_ce_times_pi` asserts rather than assuming. The mechanism is one
/// line: cross-entropy is `-log` of this same quantity, and the log is not
/// decoration -- it is precisely the factor that cancels the softmax
/// Jacobian's own `pi_t`.
///
/// **The framing that survives Adam is the one worth writing down.** "The
/// gradient vanishes when the model is wrong" is the obvious objection and it
/// is refutable: `Adam::step` divides `m_hat` by `sqrt(v_hat)`, so scaling a
/// gradient by a constant leaves the step *exactly* unchanged and a uniform
/// shrink would be absorbed. The real objection is that the factor is **not
/// uniform**. `ga`, `gb` and `dm` accumulate across every training decision
/// and Adam steps once per epoch on that sum, so a per-decision `pi_t` turns
/// the batch gradient into a `pi_t`-weighted average of the per-decision
/// cross-entropy gradients. Measured at the target logit:
///
/// ```text
///                    pi_t = 1e-3      pi_t = 0.99     emphasis
///   cross-entropy      0.999            0.010         100 : 1  toward hard
///   this objective     0.000999         0.0099          1 : 10 toward easy
/// ```
///
/// A thousandfold swing of relative emphasis, pointed at the examples the base
/// model already answers correctly -- which are exactly the ones J1 cannot
/// count as repairs. Adam normalises the sum and never the terms, so it cannot
/// undo it. **A graded reward does not help**, because the `pi_k` prefactor
/// comes from the softmax Jacobian and does not care what the rewards are.
///
/// `soft_ce_compact` ships instead, and it is the same idea with the reward
/// entering as a *target distribution* rather than as a multiplier on
/// probabilities, which is where the prefactor comes from.
pub fn reward_grad(logits: &[f32], reward: &[f32]) -> Vec<f32> {
    let probs = softmax(logits);
    let mut rbar = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        rbar += p * reward[i];
    }
    let mut gy = vec![0.0f32; logits.len()];
    for (k, g) in gy.iter_mut().enumerate() {
        *g = -probs[k] * (reward[k] - rbar);
    }
    gy
}

/// The shipped objective: cross-entropy against a **target distribution** the
/// outcome decides, blended with the label's own one-hot.
///
/// ```text
/// L   = -log pi_t                                   (reported, see below)
/// gy  = (1 - lam) (pi - onehot(t))  +  lam (pi - q)
/// ```
///
/// The deviation is `pi - q` with no `pi_k` in front of it, and that is the
/// whole reason for this form rather than `reward_grad`'s: a reward expressed
/// as *where the probability should be* differentiates without the softmax
/// Jacobian eating the signal, and a reward expressed as *what each choice is
/// worth* does not. It is bounded, it sums to zero, and at `q = onehot(t)` it
/// is cross-entropy's own gradient exactly -- so the reward can only ever move
/// the objective by disagreeing with the label, which is the point.
///
/// **The loss that comes back is the cross-entropy against the label**, not
/// the mixture. `Fit::first_loss` and `last_loss` are raw sums quoted in the
/// ledger and at `harness.rs:989`, so changing what they mean changes every
/// recorded number, and a run stays legible against one quantity whatever
/// `lam` was. What the reward moves is the direction, which is what it is for.
///
/// `lam = 0` takes `restricted_ce_compact`'s own path rather than a mixture
/// that happens to weigh zero, so today's numbers are reproduced bit for bit
/// by construction and not by the float arithmetic being kind.
///
/// `q` is normalised here rather than trusted. One that does not sum to one
/// adds a uniform component to the logit row, which the softmax ignores and
/// `backward_rows`'s magnitude route does **not** -- so a mis-scaled `q` would
/// drift the row magnitudes with every accuracy figure unchanged.
pub fn soft_ce_compact(
    logits: &[f32],
    target: usize,
    q: Option<&[f32]>,
    lam: f32,
) -> (f32, Vec<f32>) {
    let q = match q {
        // A `q` the wrong length is a bug in whoever filled it, and blending
        // it against the wrong candidates would train confidently on nothing.
        Some(q) if lam != 0.0 && q.len() == logits.len() => q,
        _ => return restricted_ce_compact(logits, target),
    };
    let mut mass = 0.0f32;
    for &v in q {
        mass += v;
    }
    if !(mass > 0.0) {
        return restricted_ce_compact(logits, target);
    }

    let probs = softmax(logits);
    let loss = -logf(probs[target]);
    let mut gy = vec![0.0f32; logits.len()];
    for (k, g) in gy.iter_mut().enumerate() {
        let ce = probs[k] - if k == target { 1.0 } else { 0.0 };
        let soft = probs[k] - q[k] / mass;
        *g = (1.0 - lam) * ce + lam * soft;
    }
    (loss, gy)
}

/// Cross-entropy over a **set** of accepted tokens: `-log sum_{i in S} pi_i`.
///
/// ```text
/// gy[j] = pi_j - pi_j 1{j in S} / p,      p = sum_{i in S} pi_i
/// ```
///
/// Not wired to anything, and it is here because what it answers is a defect
/// in the instrument rather than in the model. `chain_for` picks the
/// **longest** piece advancing toward the label and calls that the target, so
/// for a name with several tokenisations every other correct spelling is a
/// candidate cross-entropy pushes *down* and `Trial::correct` scores as wrong.
/// `constrain::costs` already counts them and its own doc says the number
/// "mostly counts prefixes": `remember` has ten first tokens against `mv`'s
/// three, both costing two tokens to spell.
///
/// So `Trial::score` is per-step teacher-forced agreement with one greedy
/// segmentation, and its doc's claim to be "the same question the constrained
/// decoder asks at temperature zero" is not quite true -- the decoder follows
/// a shorter piece and carries on spelling. **It does not follow that the
/// shorter piece is harmless**: `advances_toward` is asked about one
/// alternative, and a prefix shared by several applets keeps all of them
/// reachable, so accepting the set is an honest relaxation rather than a free
/// correction.
///
/// **Deliberately not soft-target cross-entropy over `S`.** A uniform `q` on
/// `S` would push the model to spread mass evenly across the spellings,
/// fighting it for concentrating on one; this is indifferent between them and
/// only asks that the sum be large. At `S = {t}` it is cross-entropy exactly,
/// which `set_ce_is_ce_at_a_singleton` asserts.
///
/// Wiring it changes what `correct` accepts, and every `n=`, `fixed=`,
/// `broke=` and `wrong=` already in the ledger was computed under the current
/// definition. That is a corpus-hash-class decision and not a quiet edit, so
/// it waits for one.
pub fn set_ce_compact(logits: &[f32], in_set: &[bool]) -> (f32, Vec<f32>) {
    let probs = softmax(logits);
    let mut p = 0.0f32;
    for (i, &m) in in_set.iter().enumerate() {
        if m {
            p += probs[i];
        }
    }
    let loss = -logf(p);
    let mut gy = vec![0.0f32; logits.len()];
    for (k, g) in gy.iter_mut().enumerate() {
        let keep = in_set.get(k).copied().unwrap_or(false);
        *g = probs[k] - if keep { probs[k] / p } else { 0.0 };
    }
    (loss, gy)
}

/// Softmax with the maximum subtracted, shared by the three objectives above
/// so they cannot disagree about a probability. `progress.rs` records what the
/// naive form costs: `exp(300)` is infinity in f32, and `inf - inf` is NaN on
/// exactly the confident predictions a working model makes.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    let mut probs = vec![0.0f32; logits.len()];
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        let ex = expf(v - max);
        probs[i] = ex;
        sum += ex;
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }
    probs
}

/// Natural log without libm. Range-reduced: x = m . 2^e with m in
/// [sqrt(1/2), sqrt(2)), then the atanh series on r=(m-1)/(m+1), whose
/// argument stays within +-0.172 where four terms are past f32 precision.
fn logf(x: f32) -> f32 {
    const SQRT2: f32 = 1.414_213_5;
    const LN2: f32 = 0.693_147_2;
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let (mut m, mut e) = (x, 0i32);
    while m > SQRT2 {
        m /= 2.0;
        e += 1;
    }
    while m < 1.0 / SQRT2 {
        m *= 2.0;
        e -= 1;
    }
    let r = (m - 1.0) / (m + 1.0);
    let r2 = r * r;
    let ln_m = 2.0 * r * (1.0 + r2 / 3.0 + r2 * r2 / 5.0 + r2 * r2 * r2 / 7.0);
    ln_m + e as f32 * LN2
}

/// f32 exponentiation by squaring for the small non-negative integer
/// exponents Adam's bias correction needs.
fn pow_f32(base: f32, t: u64) -> f32 {
    let mut result = 1.0f32;
    let mut b = base;
    let mut e = t;
    while e > 0 {
        if e & 1 == 1 {
            result *= b;
        }
        b *= b;
        e >>= 1;
    }
    result
}

/// Real-corpus training refuses to run without the AVX2 path: scalar
/// emulation turns one optimiser step into minutes and would make every
/// hyperparameter judgement about timing rather than maths.
pub fn hardware_ok() -> bool {
    let f = crate::cpu::detected();
    f.avx_enabled && f.avx2 && f.fma
}

// Deterministic generator for the self-test, same shape as backward's.
struct Rng(u64);

impl Rng {
    fn f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / 8_388_608.0 - 1.0
    }
}

/// Boot self-test: eight fixed examples, six classes, one QDoRA site as the
/// entire model. Trained with exactly the pieces above -- restricted CE for
/// output gradients, Dora::backward for the chain to the parameters, Adam
/// for the step -- until the claim "the trainer works" has numbers under
/// it: total loss collapsed by two orders of magnitude and every example
/// answered correctly through its own candidate set.
pub fn selftest() -> bool {
    use crate::kprintln;

    const CLASSES: usize = 6;
    const EXAMPLES: usize = 8;
    const KIN: usize = 12;

    let mut rng = Rng(0x7A1E_5EED_0000_0042);
    let wf: Vec<f32> = (0..CLASSES * KIN).map(|_| rng.f32()).collect();
    let mat = Mat::F32 { data: &wf, rows: CLASSES, cols: KIN };

    // Fixed dataset: example i belongs to class i % CLASSES, inputs drawn
    // around a class-specific offset so the task is learnable but starts
    // wrong on purpose.
    let mut data: Vec<(Vec<f32>, usize)> = Vec::new();
    for i in 0..EXAMPLES {
        let cls = i % CLASSES;
        let x: Vec<f32> = (0..KIN)
            .map(|k| rng.f32() + 0.3 * ((k + cls) % 3) as f32 - 0.3)
            .collect();
        data.push((x, cls));
    }

    let mut dd = Dora::new(4, 8.0, KIN, CLASSES);
    for v in dd.a.iter_mut() {
        *v = rng.f32() * 0.1;
    }
    for v in dd.b.iter_mut() {
        *v = rng.f32() * 0.1;
    }
    dd.refresh(&mat, true);

    // Candidate sets: true class plus two distractors, mirroring how the
    // grammar constrains decisions at episode time.
    let cands: Vec<Vec<u32>> = data
        .iter()
        .map(|(_, cls)| {
            let t = *cls as u32;
            alloc::vec![t, (t + 2) % CLASSES as u32, (t + 4) % CLASSES as u32]
        })
        .collect();

    let forward = |dd: &Dora, x: &[f32]| -> (Vec<f32>, Vec<f32>) {
        let mut base = vec![0.0f32; CLASSES];
        mat.matvec(&mut base, x);
        let mut out = base.clone();
        let mut ax = vec![0.0f32; dd.r];
        dd.apply(&mut out, x, &mut ax);
        (out, ax)
    };

    let (first_out, _) = forward(&dd, &data[0].0);
    let _ = first_out;
    let mut opt_a = Adam::new(dd.a.len());
    let mut opt_b = Adam::new(dd.b.len());
    let mut opt_m = Adam::new(dd.m.len());

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;
    const STEPS: usize = 240;
    for step in 0..STEPS {
        // Full-batch gradients before one Adam step: keeps the test about
        // the optimiser rather than about stochastic ordering.
        let mut ga = vec![0.0f32; dd.a.len()];
        let mut gb = vec![0.0f32; dd.b.len()];
        let mut dm = vec![0.0f32; dd.m.len()];
        last_loss = 0.0;
        for (i, (x, cls)) in data.iter().enumerate() {
            let (out, ax) = forward(&dd, x);
            let ti = cands[i]
                .iter()
                .position(|&c| c == *cls as u32)
                .unwrap_or(0);
            let (loss, dlogits) = restricted_ce(&out, &cands[i], ti);
            last_loss += loss;
            let mut base = vec![0.0f32; CLASSES];
            mat.matvec(&mut base, x);
            dd.backward(&mat, x, &ax, &base, &dlogits, &mut ga, &mut gb, &mut dm);
        }
        if step == 0 {
            first_loss = last_loss;
        }
        for v in ga.iter_mut() {
            *v /= EXAMPLES as f32;
        }
        for v in gb.iter_mut() {
            *v /= EXAMPLES as f32;
        }
        for v in dm.iter_mut() {
            *v /= EXAMPLES as f32;
        }
        opt_a.step(&mut dd.a, &ga, 0.08);
        opt_b.step(&mut dd.b, &gb, 0.08);
        opt_m.step(&mut dd.m, &dm, 0.02);
        dd.refresh(&mat, false);
    }

    let mut correct = 0usize;
    for (i, (x, cls)) in data.iter().enumerate() {
        let (out, _) = forward(&dd, x);
        let best_local = (0..cands[i].len())
            .max_by(|a, b| {
                out[cands[i][*a] as usize]
                    .partial_cmp(&out[cands[i][*b] as usize])
                    .unwrap()
            })
            .unwrap();
        if cands[i][best_local] == *cls as u32 {
            correct += 1;
        }
    }

    // The compact loss is what the real-model loop calls, and it is the one
    // whose gradient is never checked against a finite difference: it feeds
    // Dora::backward_rows, whose own gate is an equality against the full
    // walk. So the chain is only closed if compact and full agree here.
    let mut crng = Rng(0x00C0_FFEE_0000_0001);
    let full_logits: Vec<f32> = (0..64).map(|_| crng.f32() * 4.0).collect();
    let cand_ids: Vec<u32> = alloc::vec![3, 7, 11, 40, 41, 63];
    let ti = 2usize;
    let (loss_full, grad_full) = restricted_ce(&full_logits, &cand_ids, ti);
    let gathered: Vec<f32> = cand_ids.iter().map(|&c| full_logits[c as usize]).collect();
    let (loss_cmp, grad_cmp) = restricted_ce_compact(&gathered, ti);
    let ce_ok = (loss_full - loss_cmp).abs() < 1e-6
        && cand_ids
            .iter()
            .enumerate()
            .all(|(i, &c)| (grad_full[c as usize] - grad_cmp[i]).abs() < 1e-6)
        // ...and nothing outside the candidate set was ever blamed.
        && grad_full
            .iter()
            .enumerate()
            .all(|(i, g)| cand_ids.contains(&(i as u32)) || *g == 0.0);
    kprintln!(
        "  {}  gathered cross-entropy matches the full-width one exactly",
        if ce_ok { "ok " } else { "FAIL" }
    );

    // --- the objective, and the measurement that chose it ----------------
    //
    // Every claim here is engine-free arithmetic over a hand-built logit row,
    // because the question they settle -- which objective is worth training
    // with -- is a property of the gradient and not of any checkpoint.

    // A row where the target holds exactly probability `p`, the rest sharing
    // what is left. Built rather than sampled: the whole point is to stand at
    // a decision the model gets badly wrong on purpose.
    let row_at = |p: f32, n: usize, t: usize| -> Vec<f32> {
        let mut z = vec![0.0f32; n];
        z[t] = logf(p * (n - 1) as f32 / (1.0 - p));
        z
    };

    const N: usize = 12;
    const T: usize = 3;

    // **The gradient floor, which is the claim the design rests on.** At a
    // decision the model answers wrongly with confidence, cross-entropy pushes
    // the target up with essentially all the force it has, and the
    // reward-weighted policy gradient pushes it with `pi(target)` of that --
    // a thousandth, at the probabilities `repair.rs`'s name-cost finding says
    // a rare applet actually sits at. A later edit that "simplifies" the
    // objective back to a reward weighting fails here first.
    let zbad = row_at(1e-3, N, T);
    let (_, g_ce_bad) = restricted_ce_compact(&zbad, T);
    let mut onehot = vec![0.0f32; N];
    onehot[T] = 1.0;
    let g_rw_bad = reward_grad(&zbad, &onehot);
    let floor_ok = g_ce_bad[T].abs() > 0.9 && g_rw_bad[T].abs() < 0.01;
    kprintln!(
        "  {}  at pi(target)=1e-3 cross-entropy pulls {:.3} where a reward weighting pulls {:.5}",
        if floor_ok { "ok " } else { "FAIL" },
        g_ce_bad[T].abs(),
        g_rw_bad[T].abs()
    );

    // ...and *why* it is a thousandth: the reward gradient is the
    // cross-entropy gradient times `pi(target)`, elementwise and exactly. This
    // is the identity the doc comment on `reward_grad` argues from, asserted
    // rather than trusted.
    let mut orng = Rng(0x0B1E_C71F_0000_0003);
    let z: Vec<f32> = (0..N).map(|_| orng.f32() * 4.0 - 2.0).collect();
    let probs = softmax(&z);
    let (_, g_ce) = restricted_ce_compact(&z, T);
    let g_rw = reward_grad(&z, &onehot);
    let factor_ok = (0..N).all(|k| (g_rw[k] - probs[T] * g_ce[k]).abs() < 1e-6);
    kprintln!(
        "  {}  a binary reward's gradient is the cross-entropy one times pi(target)={:.4}",
        if factor_ok { "ok " } else { "FAIL" },
        probs[T]
    );

    // **lam = 0 reproduces today's numbers bit for bit.** Not "within a
    // tolerance": the mixture takes `restricted_ce_compact`'s own path, so
    // every trial run before this existed is re-derivable, which is the claim
    // every ledger line already written depends on.
    let (l0, g0) = soft_ce_compact(&z, T, Some(&onehot), 0.0);
    let (lr, gr) = restricted_ce_compact(&z, T);
    let lam0_ok = l0 == lr && (0..N).all(|k| g0[k] == gr[k]);
    kprintln!(
        "  {}  the mixed objective at lam=0 is bit-identical to the one it replaces",
        if lam0_ok { "ok " } else { "FAIL" }
    );

    // And at lam = 1 with the label's own one-hot as the target distribution
    // it is cross-entropy *again* -- so the reward can only move the gradient
    // by disagreeing with the label, never merely by being switched on.
    let (_, g1) = soft_ce_compact(&z, T, Some(&onehot), 1.0);
    let lam1_ok = (0..N).all(|k| (g1[k] - g_ce[k]).abs() < 1e-6);
    kprintln!(
        "  {}  at lam=1 a one-hot target distribution is cross-entropy, not a second objective",
        if lam1_ok { "ok " } else { "FAIL" }
    );

    // A target distribution that disagrees genuinely moves it, or the two
    // claims above would be satisfied by an objective that ignores `q`.
    let mut qq = vec![0.0f32; N];
    qq[T] = 0.5;
    qq[(T + 5) % N] = 0.5;
    let (_, gq) = soft_ce_compact(&z, T, Some(&qq), 1.0);
    let moves_ok = (0..N).any(|k| (gq[k] - g_ce[k]).abs() > 0.1);
    kprintln!(
        "  {}  a target distribution that disagrees with the label changes the gradient",
        if moves_ok { "ok " } else { "FAIL" }
    );

    // **Every objective here conserves zero.** A gradient that does not sum to
    // zero adds a uniform component to the logit row, which the softmax
    // ignores and `backward_rows`'s magnitude route does not -- so a reward
    // built against the wrong step's candidates would drift the row
    // magnitudes with every accuracy figure unchanged. Nothing else in this
    // file would catch that.
    let graded: Vec<f32> = (0..N).map(|k| 0.1 + 0.05 * (k % 4) as f32).collect();
    let in_set: Vec<bool> = (0..N).map(|k| k == T || k == 1 || k == 7).collect();
    let sums: [f32; 4] = [
        g_ce.iter().sum(),
        reward_grad(&z, &graded).iter().sum(),
        gq.iter().sum(),
        set_ce_compact(&z, &in_set).1.iter().sum(),
    ];
    let zero_ok = sums.iter().all(|s| s.abs() < 1e-5);
    kprintln!(
        "  {}  cross-entropy, reward, mixture and set forms all conserve zero",
        if zero_ok { "ok " } else { "FAIL" }
    );

    // The degeneration canary. A reward with no variance across the candidate
    // set has no advantage to express, so it trains *nothing* -- which the
    // ledger cannot tell apart from a variant that trained and repaired
    // nothing, the same pair the no-random-seed bug already cost this tree.
    let flat = vec![0.4f32; N];
    let g_flat = reward_grad(&z, &flat);
    let flat_ok = g_flat.iter().all(|g| g.abs() < 1e-6);
    kprintln!(
        "  {}  a reward that is constant across the candidates moves nothing at all",
        if flat_ok { "ok " } else { "FAIL" }
    );

    // The set form is cross-entropy at a singleton, which is what makes it a
    // generalisation rather than a different objective wearing the name.
    let single: Vec<bool> = (0..N).map(|k| k == T).collect();
    let (ls, gs) = set_ce_compact(&z, &single);
    let set_ok = (ls - lr).abs() < 1e-6 && (0..N).all(|k| (gs[k] - g_ce[k]).abs() < 1e-6);
    // ...and accepting a set genuinely relaxes it: the target is pulled less
    // hard because two other spellings now carry part of the answer.
    let (_, gset) = set_ce_compact(&z, &in_set);
    let relax_ok = gset[T].abs() < g_ce[T].abs();
    kprintln!(
        "  {}  set cross-entropy is cross-entropy at one token and relaxes at three",
        if set_ok && relax_ok { "ok " } else { "FAIL" }
    );

    // A finite difference against the objective's own loss, which is the gate
    // `restricted_ce_compact` has by equality with the full-width walk and
    // these have nowhere else to get. `h` is a power of two so it is exact in
    // f32; the tolerance is what h^2 truncation plus f32 rounding at this
    // scale actually costs, rather than a number chosen to make it pass.
    let h = 1.0f32 / 32.0;
    let soft_loss = |zz: &[f32], q: &[f32]| -> f32 {
        let p = softmax(zz);
        let mut mass = 0.0f32;
        for &v in q {
            mass += v;
        }
        let mut l = 0.0f32;
        for k in 0..zz.len() {
            l -= (q[k] / mass) * logf(p[k]);
        }
        l
    };
    let mut fd_ok = true;
    for j in 0..N {
        let (mut zp, mut zm) = (z.clone(), z.clone());
        zp[j] += h;
        zm[j] -= h;
        let num = (soft_loss(&zp, &qq) - soft_loss(&zm, &qq)) / (2.0 * h);
        if (num - gq[j]).abs() > 2e-3 {
            fd_ok = false;
        }
    }
    kprintln!(
        "  {}  the mixture's gradient survives a finite difference of its own loss",
        if fd_ok { "ok " } else { "FAIL" }
    );

    let obj_ok = floor_ok
        && factor_ok
        && lam0_ok
        && lam1_ok
        && moves_ok
        && zero_ok
        && flat_ok
        && set_ok
        && relax_ok
        && fd_ok;

    let collapsed = last_loss < first_loss * 0.05;
    let all_right = correct == EXAMPLES;
    let ok = collapsed && all_right && ce_ok && obj_ok;
    kprintln!(
        "  {}  loss {:.3} -> {:.3}, {}/{} answered right through their candidate set",
        if ok { "ok " } else { "FAIL" },
        first_loss,
        last_loss,
        correct,
        EXAMPLES
    );
    ok
}

// --- the real-model loop -------------------------------------------------
//
// Everything above is engine-free so its bugs cannot hide inside a run. This
// is the part that is not: the corpus, the frozen model, the grammar and the
// clock, assembled into something that trains the decision layer of the
// checkpoint actually loaded.
//
// Three facts make it affordable in a kernel, and each is exact rather than an
// approximation:
//
//   **Only the classifier moves.** The base is frozen and no adapter sits on
//   the attention path, so the hidden state at every decision is a constant.
//   It is computed once per example and cached; an epoch after that costs no
//   forward passes at all. This is the same lesson `harness::train` records
//   -- recomputing features every epoch made a twenty-epoch run take two
//   minutes -- collected a second time because the trap is the same one.
//
//   **Only reachable rows move.** Restricted cross-entropy makes the output
//   gradient exactly zero outside the grammar's candidate set, so a row the
//   decoder can never emit contributes exactly nothing. The union of those
//   sets over the whole corpus is a few thousand rows out of a vocabulary of
//   151,936, and the trainer works over a dequantised copy of just those --
//   which is what turns a 155 MB pass per step into a few megabytes resident.
//
//   **Teacher forcing makes the whole chain cacheable.** An applet name is
//   usually more than one token, so choosing one is more than one decision.
//   Feeding the correct next token rather than the sampled one keeps every
//   later hidden state a constant too, and the chain of candidate sets is a
//   property of the name rather than of the task -- so the vocabulary scan
//   that finds them runs once per applet, not once per example.
//
// What this does not do: train the attention path. The activation adjoints
// exist and are gated at boot, but nothing yet composes them into a backward
// pass through the layers, and doing so would end the cached-feature bargain
// above -- the features stop being constants the moment q/k/v start moving.
// That is a later phase's problem, and it is stated here rather than
// discovered.

/// What one run is allowed to spend.
///
/// A ceiling rather than a target. Training is the one thing on this machine
/// that can run arbitrarily long while looking like it is working, and the
/// shell is single-threaded: a run that cannot be bounded is a run that can
/// take the terminal away with no way to ask for it back.
/// What a full-network run did.
pub struct FullReport {
    pub examples: usize,
    pub epochs: usize,
    pub first_loss: f32,
    pub last_loss: f32,
    pub ms: u64,
    /// Stopped by the wall clock rather than by finishing its epochs.
    pub stopped: bool,
}

/// One Adam state per tensor of one adapted site.
struct SiteOpt {
    a: Adam,
    b: Adam,
    m: Adam,
}

impl SiteOpt {
    fn like(d: &super::adapter::Dora) -> SiteOpt {
        SiteOpt {
            a: Adam::new(d.a.len()),
            b: Adam::new(d.b.len()),
            m: Adam::new(d.m.len()),
        }
    }
}

/// Train every adapted site, not just the classifier.
///
/// The other trainer -- `Trial::train` -- rests on the base being frozen
/// *below the classifier*: a hidden state is then a constant, is cached once
/// per example, and an epoch after that costs no forward passes at all. That
/// is what makes it affordable, and it is exactly what adapting q, k and v
/// gives up. Move a projection in layer three and every hidden state after it
/// moves, so there is nothing to cache and every epoch pays a forward pass per
/// example.
///
/// This is therefore a different function rather than a flag on that one. The
/// two share an optimiser and nothing else, and folding them together would
/// hide which of the two economics a given run is paying.
///
/// ### The objective, and how it is weaker than the other one
///
/// Full-vocabulary cross-entropy on the **first token of the applet name**.
/// `Trial` does better: it scores every step of the spelling under a grammar
/// that has already removed unreachable applets, so its gradient is restricted
/// to candidates the decoder could actually emit. Doing that here needs the
/// chain machinery threaded through a taped forward per step, which is a
/// larger piece; the first token is where most of the discrimination lives --
/// it separates most of the twenty-two applets outright -- and it is exactly
/// the loss whose gradient `adapter::walk_selftest` differences.
///
/// Said plainly so a comparison between the two is not read as a comparison of
/// what the sites can learn.
pub fn train_full(
    e: &mut super::Engine,
    b: &Budget,
    limit: usize,
) -> Option<FullReport> {
    if !hardware_ok() {
        return None;
    }
    let cfg = e.model.cfg.clone();
    if cfg.hybrid() || cfg.streams() {
        return None;
    }
    let ad = e.model.adapters.as_ref()?;
    let mut grads = super::model::Grads::new(ad);

    // Optimisers, shaped from the attached adapters so an unadapted site has
    // none rather than an empty one nobody notices is idle.
    let mut opts: Vec<[Option<SiteOpt>; 3]> = ad
        .qkv
        .iter()
        .map(|t| {
            [
                t[0].as_ref().map(SiteOpt::like),
                t[1].as_ref().map(SiteOpt::like),
                t[2].as_ref().map(SiteOpt::like),
            ]
        })
        .collect();
    let mut cls_opt = ad.cls.as_ref().map(SiteOpt::like);

    // Build the training set once: prompt tokens in, target token out.
    let (train_end, _, _) = super::vocab::splits();
    let mut set: Vec<(Vec<usize>, usize)> = Vec::new();
    for (i, ex) in super::vocab::examples().iter().enumerate() {
        if i >= train_end || set.len() >= limit.max(1) {
            continue;
        }
        let prompt = super::harness::prompt_for_task(&ex.task);
        let toks = e.tok.encode(&prompt, true, false);
        // The label is the applet's name as the decoder would begin to spell
        // it. A leading space, because that is how it follows "Tool:" in the
        // prompt and therefore how the tokenizer saw it during every decode.
        let mut label = alloc::string::String::from(" ");
        label.push_str(&ex.applet);
        let lt = e.tok.encode(&label, false, false);
        let (Some(&first), false) = (lt.first(), toks.is_empty()) else { continue };
        set.push((toks, first));
    }
    if set.is_empty() {
        return None;
    }

    let t0 = crate::time::rdtsc();
    let mhz = crate::time::tsc_mhz().max(1);
    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;
    let mut epochs = 0usize;
    let mut stopped = false;

    // One state and one tape for the whole run.
    //
    // A fresh `State` per example allocates the entire KV cache each time --
    // 112 MiB at Qwen3-0.6B's trained context -- so a run of six examples over
    // four epochs would allocate and free it twenty-four times, which would
    // dominate the very measurement this exists to produce and fragment a heap
    // that is one physically contiguous block.
    //
    // Reuse is safe without clearing the cache. Every example starts at
    // position 0, and attention reads only `live` slots, which is `pos + 1`
    // when nothing is windowed -- so a slot left by a previous example is
    // overwritten before it could be read. `Tape::reset` is needed because
    // `filled` is the one piece of state that would otherwise carry over.
    let longest = set.iter().map(|(t, _)| t.len()).max().unwrap_or(1);
    let mut st = super::model::State::new(&cfg);
    let mut tape = super::model::Tape::new(&cfg, longest);

    for epoch in 0..b.epochs {
        grads.clear();
        last_loss = 0.0;
        for (toks, target) in set.iter() {
            tape.reset();
            for (i, &t) in toks.iter().enumerate() {
                if !e.model.forward_taped(&mut st, t, i, &mut tape) {
                    return None;
                }
            }
            let at = toks.len() - 1;
            let (loss, gl) = softmax_ce(&st.logits, *target);
            last_loss += loss;
            if !e.model.backward(&tape, &gl, at, &mut grads) {
                return None;
            }
        }
        if epoch == 0 {
            first_loss = last_loss;
        }
        epochs = epoch + 1;

        // Mean over the set, so the step size means the same thing whatever
        // `limit` was.
        let k = 1.0 / set.len() as f32;
        scale_grads(&mut grads, k);

        if let Some(ad) = e.model.adapters.as_mut() {
            for (l, t) in ad.qkv.iter_mut().enumerate() {
                for (i, site) in t.iter_mut().enumerate() {
                    let (Some(d), Some(o)) = (site.as_mut(), opts[l][i].as_mut()) else {
                        continue;
                    };
                    o.a.step(&mut d.a, &grads.qkv[l][i].ga, b.lr);
                    o.b.step(&mut d.b, &grads.qkv[l][i].gb, b.lr);
                    o.m.step(&mut d.m, &grads.qkv[l][i].dm, b.lr);
                }
            }
            if let (Some(d), Some(o)) = (ad.cls.as_mut(), cls_opt.as_mut()) {
                o.a.step(&mut d.a, &grads.cls.ga, b.lr);
                o.b.step(&mut d.b, &grads.cls.gb, b.lr);
                o.m.step(&mut d.m, &grads.cls.dm, b.lr);
            }
        }
        // The per-row scales are stale the moment a or b moves, and every
        // forward after this reads them. Refreshing is a pass over each
        // adapted weight, which is why it is here and not in the inner loop.
        refresh_all(e);

        if b.millis > 0 && (crate::time::rdtsc() - t0) / mhz / 1000 >= b.millis {
            stopped = true;
            break;
        }
    }

    Some(FullReport {
        examples: set.len(),
        epochs,
        first_loss: first_loss / set.len() as f32,
        last_loss: last_loss / set.len() as f32,
        ms: (crate::time::rdtsc() - t0) / mhz / 1000,
        stopped,
    })
}

/// Cross-entropy over the whole vocabulary, and its gradient.
///
/// The gradient is `softmax - onehot`, which is what `Model::backward` expects
/// and what the walk's own check differences against.
fn softmax_ce(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let m = logits.iter().fold(f32::MIN, |a, v| a.max(*v));
    let mut z = 0.0f32;
    for v in logits {
        z += super::tensor::expf(v - m);
    }
    let inv = 1.0 / z.max(1e-30);
    let mut g = vec![0.0f32; logits.len()];
    for (o, v) in logits.iter().enumerate() {
        g[o] = super::tensor::expf(v - m) * inv;
    }
    let ti = target.min(g.len().saturating_sub(1));
    let p = g[ti].max(1e-30);
    g[ti] -= 1.0;
    (-logf(p), g)
}

fn scale_grads(g: &mut super::model::Grads, k: f32) {
    for t in g.qkv.iter_mut() {
        for s in t.iter_mut() {
            for v in s.ga.iter_mut().chain(s.gb.iter_mut()).chain(s.dm.iter_mut()) {
                *v *= k;
            }
        }
    }
    for v in g.cls.ga.iter_mut().chain(g.cls.gb.iter_mut()).chain(g.cls.dm.iter_mut()) {
        *v *= k;
    }
}

/// Recompute every adapted site's cached scales against its frozen weight.
fn refresh_all(e: &mut super::Engine) {
    // Taken out and put back, because the frozen weight and the adapter live
    // in the same struct: `frozen_site` borrows the model to hand back a view
    // of a weight, and refreshing needs the adapter mutably at the same time.
    // Moving the adapters aside for the duration splits the borrow without
    // copying anything.
    let Some(mut ad) = e.model.adapters.take() else { return };
    let n = e.model.cfg.n_layers;
    for l in 0..n {
        for i in 0..3 {
            let w = e.model.frozen_site(l, i);
            if let Some(d) = ad.qkv[l][i].as_mut() {
                d.refresh(&w, false);
            }
        }
    }
    let w = e.model.frozen_cls();
    if let Some(d) = ad.cls.as_mut() {
        d.refresh(&w, false);
    }
    e.model.adapters = Some(ad);
}

pub struct Budget {
    /// Passes over the cached decisions.
    pub epochs: usize,
    /// Wall-clock ceiling. Zero means no ceiling.
    pub millis: u64,
    /// Corpus examples to prepare. Zero means all of them.
    pub examples: usize,
    pub lr: f32,
    pub rank: usize,
    pub alpha: f32,
    /// How much of the objective the outcome reward carries. Zero takes
    /// `restricted_ce_compact`'s own path, so a default `Budget` reproduces
    /// every figure recorded before this field existed, bit for bit.
    pub mix: f32,
}

impl Default for Budget {
    fn default() -> Self {
        Self { epochs: 20, millis: 120_000, examples: 0, lr: 0.02, rank: 8, alpha: 16.0, mix: 0.0 }
    }
}

pub enum RunError {
    /// `hardware_ok` said no.
    Hardware,
    NoCorpus,
    Hybrid,
    /// The corpus produced nothing the grammar could spell.
    NoDecisions,
}

pub struct RunReport {
    pub examples: usize,
    pub decisions: usize,
    pub held: usize,
    /// Classifier rows the grammar can reach at all -- what is resident.
    pub rows: usize,
    pub epochs_run: usize,
    pub first_loss: f32,
    pub last_loss: f32,
    pub before_train: f32,
    pub after_train: f32,
    pub before_held: f32,
    pub after_held: f32,
    /// Building the grammar chains and dequantising the reachable rows:
    /// a fixed cost over the applet table, unaffected by how many examples
    /// were asked for.
    pub chains_ms: u64,
    /// Caching one hidden state per decision: the per-example half.
    pub prep_ms: u64,
    pub train_ms: u64,
    /// Whether the wall-clock ceiling ended it rather than the epoch count.
    pub stopped: bool,
    /// The trained adapter's content address, first eight hex digits.
    ///
    /// Printed so "lam = 0 reproduces today's numbers bit for bit" is a thing
    /// somebody can check against a build from before the objective existed,
    /// rather than a claim about one hand-built logit row. It is the same
    /// digest `godel` puts in a `Variant`, over `scatter(..).to_blob()`, so
    /// the two cannot drift.
    pub digest: [u8; 32],
    /// What the outcome reward changed, when one was supplied.
    pub aim: Option<Aim>,
}

/// One step of one applet's spelling: the tokens the grammar admits here, and
/// which of them the label says to emit. A property of the name, so it is
/// built once per applet and shared by every example labelled with it.
pub(crate) struct Step {
    /// Indices into the live-row table, in candidate order.
    local: Vec<u32>,
    /// Which entry of `local` is correct.
    target: usize,
    /// The token to feed to keep the chain on the label's path.
    token: u32,
    /// Which candidate each applet would take from this state, `-1` for one
    /// this state can no longer reach. Grammar-only, so it is shared by every
    /// example labelled with this applet.
    votes: Vec<i16>,
    /// Which candidates keep the label reachable. `target` is the longest of
    /// them; the rest are correct spellings `Trial::correct` scores as wrong.
    reach: Vec<bool>,
    /// The target distribution the mixed objective aims at, in candidate
    /// order, or `None` while nothing has supplied an outcome matrix.
    ///
    /// **It lives here rather than on `Decision`, and that is a consequence of
    /// the measurement in `outcome.rs` rather than a saving.** The reward is
    /// task-independent -- the corpus carries no operands, so `r(example,
    /// applet)` depends only on the example's label -- and a `Step` already
    /// *is* per label and per position. Were the reward ever to become
    /// per-example this has to move to `Decision`, at a cost of one `Vec<f32>`
    /// per decision.
    q: Option<Vec<f32>>,
}

/// One cached decision: the constant hidden state, and where to find the
/// candidate set it belongs to.
pub(crate) struct Decision {
    x: Vec<f32>,
    /// Base logits over this step's candidates, before any adapter. Frozen
    /// weights against a frozen feature, so this is a constant too -- and
    /// caching it is what keeps a dequant pass out of the inner loop.
    base: Vec<f32>,
    applet: usize,
    step: usize,
    /// Outside the training slice: validation or test.
    held: bool,
    /// The test half specifically. Kept apart from `held` because the test
    /// slice is read once by discipline, and a loop that improves itself
    /// forever would otherwise read it on every trial and report a number
    /// that got more optimistic each time it was consulted.
    test: bool,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

fn millis_since(t0: u64) -> u64 {
    let mhz = crate::time::tsc_mhz();
    if mhz == 0 {
        return 0;
    }
    (crate::time::rdtsc().saturating_sub(t0)) / mhz / 1000
}

/// Spell one applet under the grammar, recording each step's candidate set.
///
/// The longest admissible piece is taken at every step, which is both the
/// fewest decisions that spell the name and the segmentation an ordinary
/// greedy tokenizer produces -- so the chain is the one the decoder would
/// most plausibly walk, not an artefact of this function.
fn chain_for(
    grammar: &Grammar,
    alphabet: &Alphabet,
    alt: usize,
) -> Option<Vec<(Vec<u32>, usize, u32, Vec<i16>, Vec<bool>)>> {
    let n_alts = super::constrain::alternatives(grammar).len();
    let mut cursor = Cursor::new(grammar);
    let mut steps = Vec::new();
    for _ in 0..step_bound(grammar) {
        let cands = cursor.candidates(alphabet);
        if cands.is_empty() {
            return None;
        }
        // Which candidate each alternative would take from *this* state, and
        // -1 for one this state can no longer reach. The same "longest piece
        // that advances" rule the label gets, applied to every applet, so the
        // vote a rival casts is the one it would really cast rather than a
        // prefix it merely shares.
        //
        // Grammar-only and therefore a property of the name: it goes in `Step`,
        // which is built once per applet and shared by every example labelled
        // with it, so a per-example reward costs no per-decision memory at all.
        let mut votes = alloc::vec![-1i16; n_alts];
        // And which candidates keep the *label* reachable at all, which is a
        // different question from which one the label would take: `chain_for`
        // picks the longest and `Trial::correct` demands exactly it, so every
        // other member of this set is a correct spelling scored as wrong.
        // Kept so the size of that can be measured on real logits rather than
        // argued from `constrain::costs`.
        let mut reach = alloc::vec![false; cands.len()];
        let mut best: Option<(usize, u32, usize)> = None;
        for b in 0..n_alts {
            let mut long: Option<(usize, usize)> = None;
            for (i, &id) in cands.iter().enumerate() {
                if !cursor.advances_toward(alphabet, id as usize, b) {
                    continue;
                }
                if b == alt {
                    reach[i] = true;
                }
                let n = alphabet.piece(id as usize).len();
                if long.map_or(true, |(_, bn)| n > bn) {
                    long = Some((i, n));
                }
            }
            if let Some((i, n)) = long {
                votes[b] = i as i16;
                if b == alt && best.map_or(true, |(_, _, bn)| n > bn) {
                    best = Some((i, cands[i], n));
                }
            }
        }
        let (target, token, _) = best?;
        steps.push((cands, target, token, votes, reach));
        cursor.push(alphabet, token as usize);
        if cursor.finished() == Some(alt) {
            return Some(steps);
        }
    }
    None
}

/// How many distinct first tokens spell `alt`, from the state the chain is in
/// at `step`. One means the label has exactly one correct spelling here and
/// cross-entropy is asking the right question; more means `chain_for` picked
/// the longest and `Trial::correct` scores the others wrong.
///
/// Separate from `chain_for` because it counts what that function *discards*,
/// and a caller wanting the number should not have to reconstruct the walk.
fn spellings_at(grammar: &Grammar, alphabet: &Alphabet, alt: usize, upto: usize) -> Vec<usize> {
    let mut cursor = Cursor::new(grammar);
    let mut out = Vec::new();
    for _ in 0..upto.min(step_bound(grammar)) {
        let cands = cursor.candidates(alphabet);
        if cands.is_empty() {
            break;
        }
        let mut n = 0usize;
        let mut best: Option<(u32, usize)> = None;
        for &id in cands.iter() {
            if cursor.advances_toward(alphabet, id as usize, alt) {
                n += 1;
                let len = alphabet.piece(id as usize).len();
                if best.map_or(true, |(_, bl)| len > bl) {
                    best = Some((id, len));
                }
            }
        }
        let Some((token, _)) = best else { break };
        out.push(n);
        cursor.push(alphabet, token as usize);
        if cursor.finished() == Some(alt) {
            break;
        }
    }
    out
}

/// Train the loaded model's decision layer on the corpus in the namespace.
/// One prepared trial: the corpus, reduced to everything a variant can be
/// judged on without touching the model again.
///
/// This is the object the whole self-modification loop stands on. Building it
/// costs a forward pass per example -- 214 s each under TCG, seconds on real
/// hardware -- and once it exists, scoring *any* adapter against it costs a
/// dot product per decision and no forward passes at all. That asymmetry is
/// what makes a verified self-modification affordable here: producing a
/// candidate is expensive, and checking somebody else's claim about one is
/// nearly free.
pub struct Trial {
    chains: Vec<Option<Vec<Step>>>,
    decisions: Vec<Decision>,
    live: Vec<u32>,
    w_live: Vec<f32>,
    dim: usize,
    /// Examples that produced at least one decision.
    pub examples: usize,
    /// Fixed prep cost: grammar chains and the dequantised rows.
    pub chains_ms: u64,
    /// Per-example prep cost: one forward pass each.
    pub features_ms: u64,
    /// The machine's own goals, cached along the baseline's own path.
    guards: Vec<Guard>,
}

/// Which slice a score is taken over.
#[derive(Clone, Copy, PartialEq)]
pub enum Slice {
    Train,
    /// Everything held out. The trainer reports this; the judges use the
    /// validation half of it, so the test half stays unread until a variant
    /// has already won on validation.
    Held,
    Validation,
    Test,
}

/// What `Trial::aim` found when it pointed the objective at an outcome matrix.
///
/// Returned rather than discarded because a target distribution that came back
/// equal to the label's one-hot would train exactly what training already
/// trains, and the run would look like a working experiment. That is the
/// degeneration `outcome.rs` names, and it is the same shape as the
/// no-random-seed bug: every number in the ledger stays plausible.
#[derive(Default)]
pub struct Aim {
    /// Steps that were given a target distribution.
    pub steps: usize,
    /// Steps where that distribution is not the label's own one-hot.
    pub moved: usize,
    /// Mean probability mass the reward puts anywhere but the target. Zero
    /// means the mixture is cross-entropy however large `lam` is.
    pub off_target: f32,
    /// Rival applets that were reachable and able to vote, summed over steps.
    pub voters: usize,
    /// Steps whose candidates carried no mass at all, which cannot happen
    /// unless `votes` and `target` disagree.
    pub empty: usize,
    /// The matrix did not describe this grammar, so nothing was aimed.
    pub refused: bool,
}

impl Trial {
    pub fn decisions(&self) -> usize {
        self.decisions.len()
    }

    pub fn held(&self) -> usize {
        self.decisions.iter().filter(|d| d.held).count()
    }

    pub fn live_rows(&self) -> usize {
        self.live.len()
    }

    /// The weight view every scoring and training pass reads.
    fn mat(&self) -> Mat<'_> {
        Mat::F32 { data: &self.w_live, rows: self.live.len(), cols: self.dim }
    }

    fn in_slice(&self, d: &Decision, s: Slice) -> bool {
        match s {
            Slice::Train => !d.held,
            Slice::Held => d.held,
            Slice::Validation => d.held && !d.test,
            Slice::Test => d.test,
        }
    }

    /// Logits over one decision's candidate set, under an optional adapter.
    fn logits(&self, d: &Decision, dora: Option<&Dora>, out: &mut Vec<f32>, ax: &mut [f32]) {
        out.clear();
        out.extend_from_slice(&d.base);
        if let Some(dora) = dora {
            let st = &self.chains[d.applet].as_ref().unwrap()[d.step];
            dora.apply_rows(out, &st.local, &d.x, ax);
        }
    }

    fn correct(&self, d: &Decision, out: &[f32]) -> bool {
        let st = &self.chains[d.applet].as_ref().unwrap()[d.step];
        let mut best = 0usize;
        for c in 1..out.len() {
            if out[c] > out[best] {
                best = c;
            }
        }
        best == st.target
    }

    /// Accuracy over a slice: does the label's token come first among the
    /// tokens the grammar admits? The same question the constrained decoder
    /// asks at temperature zero, which is the point -- a number measured any
    /// other way would not be the number the system's behaviour depends on.
    /// Point every step's objective at an outcome-derived target distribution,
    /// and report what that actually changed.
    ///
    /// For a step on `label`'s chain, each applet still reachable from here
    /// votes for the candidate it would itself take, and its vote is worth
    /// `1 - distance(label, applet)` -- so a rival whose output resembles the
    /// label's moves probability toward its own spelling, and one that was
    /// never run (because it mutates) moves none.
    ///
    /// **`APPLETS` order is the grammar's order and this depends on it.**
    /// `prepare_on` builds `names` straight from `crate::sysbox::APPLETS`, so
    /// alternative `b` and matrix row `b` are the same applet. That is an
    /// invariant of two files agreeing, which is the pair this tree keeps
    /// finding has stopped agreeing, so it is checked here rather than
    /// assumed: a mismatched length refuses instead of aiming at a
    /// permutation of the right answer.
    ///
    /// The report is the point of returning anything. A target distribution
    /// that came back identical to the label's one-hot would train exactly
    /// what training already trains, and the run would look like a working
    /// experiment -- the degeneration `outcome.rs` names and the same shape
    /// the no-random-seed bug had.
    pub fn aim(&mut self, m: &super::outcome::Matrix) -> Aim {
        let mut a = Aim::default();
        if m.len() != self.chains.len() {
            a.refused = true;
            return a;
        }
        for (label, chain) in self.chains.iter_mut().enumerate() {
            let Some(steps) = chain.as_mut() else { continue };
            let row = m.reward_row(label);
            for st in steps.iter_mut() {
                let mut q = alloc::vec![0.0f32; st.local.len()];
                let mut voters = 0usize;
                for (b, &v) in st.votes.iter().enumerate() {
                    if v < 0 {
                        continue;
                    }
                    let i = v as usize;
                    if i >= q.len() {
                        continue;
                    }
                    // A rival that is reachable but worth nothing adds nothing,
                    // and is still counted as having been asked -- "no applet
                    // could vote here" and "every applet that could vote was
                    // worth zero" are different facts about a step.
                    if b != label {
                        voters += 1;
                    }
                    q[i] += row[b];
                }
                let mass: f32 = q.iter().sum();
                if !(mass > 0.0) {
                    // The label always votes for its own target and its own
                    // reward is 1, so this cannot happen -- unless `votes` and
                    // `target` disagree, which is exactly the indexing bug
                    // worth refusing rather than normalising away.
                    a.empty += 1;
                    continue;
                }
                let off = 1.0 - q[st.target] / mass;
                a.off_target += off;
                if off > 0.0 {
                    a.moved += 1;
                }
                a.voters += voters;
                a.steps += 1;
                st.q = Some(q);
            }
        }
        if a.steps > 0 {
            a.off_target /= a.steps as f32;
        }
        a
    }

    /// How many first tokens spell each applet, and therefore how often
    /// `chain_for`'s longest-piece rule throws a correct spelling away. See
    /// `set_ce_compact`.
    pub fn spellings(e: &mut super::Engine) -> Vec<(&'static str, Vec<usize>)> {
        let names: Vec<&'static str> = crate::sysbox::APPLETS.iter().map(|a| a.name).collect();
        let grammar = Grammar::new(names.iter().copied());
        let alphabet = super::harness::alphabet_for(&e.tok);
        names
            .iter()
            .enumerate()
            .map(|(alt, n)| (*n, spellings_at(&grammar, alphabet, alt, 8)))
            .collect()
    }
    pub fn score(&self, dora: Option<&Dora>, s: Slice) -> f32 {
        let mut out = Vec::new();
        let mut ax = vec![0.0f32; dora.map(|d| d.r).unwrap_or(1)];
        let (mut right, mut total) = (0usize, 0usize);
        for d in self.decisions.iter().filter(|d| self.in_slice(d, s)) {
            self.logits(d, dora, &mut out, &mut ax);
            if self.correct(d, &out) {
                right += 1;
            }
            total += 1;
        }
        if total == 0 {
            0.0
        } else {
            right as f32 / total as f32
        }
    }

    /// The paired comparison, and the reason the judges are not a pair of
    /// percentages.
    ///
    /// Two adapters answer *the same* cached decisions, so the comparison is
    /// paired rather than between two independent samples -- which is only
    /// available because the base is frozen and the features are cached. It
    /// matters more than it sounds. Fifty validation decisions at 62% against
    /// 58% is two items and indistinguishable from noise; the same fifty
    /// showing nine answers repaired and two broken is a different claim
    /// entirely, and only the paired form can tell them apart.
    ///
    /// Returns (broke, fixed, unchanged-correct, unchanged-wrong): `broke` is
    /// McNemar's b, `fixed` is c.
    /// How much `Trial::correct` understates accuracy by demanding exactly the
    /// longest spelling.
    ///
    /// Returns `(strict, lenient, total)`: how many decisions the argmax got
    /// right under the rule this tree has always used, how many it got right
    /// if *any* token keeping the label reachable counts, and how many there
    /// were. The gap is the instrument's own false-negative rate, measured on
    /// real logits rather than argued from `constrain::costs`.
    ///
    /// **The lenient figure is not a better accuracy and must not be read as
    /// one.** A token that keeps the label reachable does not commit to it:
    /// `advances_toward` is asked about one alternative, and ` s` keeps
    /// `stat`, `same`, `snaps`, `snap` and `sysbox` all alive, so a decoder
    /// that took it could still land anywhere. What the gap bounds is how many
    /// decisions the strict rule *could* be wrong about -- an upper bound on
    /// the defect, not a measurement of routing.
    pub fn lenient(&self, dora: Option<&Dora>, s: Slice) -> (usize, usize, usize) {
        let mut out = Vec::new();
        let mut ax = vec![0.0f32; dora.map(|d| d.r).unwrap_or(1)];
        let (mut strict, mut lenient, mut total) = (0usize, 0usize, 0usize);
        for d in self.decisions.iter().filter(|d| self.in_slice(d, s)) {
            let st = &self.chains[d.applet].as_ref().unwrap()[d.step];
            self.logits(d, dora, &mut out, &mut ax);
            let mut best = 0usize;
            for c in 1..out.len() {
                if out[c] > out[best] {
                    best = c;
                }
            }
            if best == st.target {
                strict += 1;
            }
            if st.reach.get(best).copied().unwrap_or(false) {
                lenient += 1;
            }
            total += 1;
        }
        (strict, lenient, total)
    }

    pub fn paired(
        &self,
        old: Option<&Dora>,
        new: Option<&Dora>,
        s: Slice,
    ) -> (usize, usize, usize, usize) {
        let r = old.map(|d| d.r).max(new.map(|d| d.r)).unwrap_or(1);
        let mut ax = vec![0.0f32; r];
        let (mut out_a, mut out_b) = (Vec::new(), Vec::new());
        let (mut broke, mut fixed, mut both, mut neither) = (0usize, 0usize, 0usize, 0usize);
        for d in self.decisions.iter().filter(|d| self.in_slice(d, s)) {
            self.logits(d, old, &mut out_a, &mut ax);
            let a = self.correct(d, &out_a);
            self.logits(d, new, &mut out_b, &mut ax);
            let b = self.correct(d, &out_b);
            match (a, b) {
                (true, false) => broke += 1,
                (false, true) => fixed += 1,
                (true, true) => both += 1,
                (false, false) => neither += 1,
            }
        }
        (broke, fixed, both, neither)
    }

    /// Route one free-text task through the trained decision layer, returning
    /// the applet the grammar would land on.
    ///
    /// Needs the model, unlike everything else here: a task that is not in
    /// the corpus has no cached hidden state. Used by the judge that replays
    /// the machine's own self-set goals, where the whole point is that they
    /// are not corpus items.
    pub fn route_fresh(
        e: &mut super::Engine,
        task: &str,
        dora_attached: bool,
    ) -> Option<&'static str> {
        let _ = dora_attached;
        super::harness::choose(task, super::harness::Trust::Full, 0.0).map(|c| c.applet)
    }
}
/// Build a trial: everything a variant can be judged on, and the model is
/// not needed again afterwards.
pub fn prepare(e: &mut super::Engine, b: &Budget) -> Result<Trial, RunError> {
    let corpus = super::vocab::examples();
    let (train_end, val_end, seed_end) = super::vocab::splits();
    prepare_on(e, b, &corpus, train_end, val_end, seed_end)
}

/// The same trial over examples somebody else assembled.
///
/// `prepare` reads the routing corpus and takes its boundaries from
/// `vocab::splits`; this takes both from the caller. Everything below is
/// identical, deliberately -- a role adapter judged by a second copy of this
/// function would be judged by whatever that copy had drifted into, which is
/// the objection `model.rs` makes twice about two implementations that are
/// supposed to agree.
///
/// The boundaries are positional in the same sense: training is `[0,
/// train_end)`, validation `[train_end, val_end)`, test `[val_end, end)`.
/// Passing `val_end == end` gives a trial with no test slice at all, which is
/// the right shape for a set too small to split three ways -- and it means
/// nothing reads the test budget by accident.
pub fn prepare_on(
    e: &mut super::Engine,
    b: &Budget,
    corpus: &[super::vocab::Example],
    train_end: usize,
    val_end: usize,
    seed_end: usize,
) -> Result<Trial, RunError> {
    // The gate comes first, before anything is allocated or measured. Scalar
    // emulation turns one optimiser step into minutes, and every judgement
    // made from a run like that is a judgement about timing rather than
    // about arithmetic.
    if !hardware_ok() {
        return Err(RunError::Hardware);
    }
    if e.model.cfg.hybrid() {
        return Err(RunError::Hybrid);
    }
    if corpus.is_empty() {
        return Err(RunError::NoCorpus);
    }

    let t_prep = crate::time::rdtsc();
    let dim = e.model.cfg.dim;

    // Every applet, not the read-only subset. The corpus labels examples
    // with mutating applets, and a read-only grammar has no token sequence
    // that spells them -- so half the corpus would be unlearnable and the
    // half that remained would train the model never to reach the rest.
    //
    // The consequence, stated rather than discovered: `act` in its default
    // read-only mode sends a *shorter* tool list than this trains against,
    // because `prompt_for` renders whichever applets the trust level admits.
    // `trusted` mode matches exactly. Read-only decoding is a restriction of
    // what was trained rather than a different task, which is the defensible
    // direction for the mismatch to run, but it is a mismatch.
    let names: Vec<&'static str> = crate::sysbox::APPLETS.iter().map(|a| a.name).collect();
    let grammar = Grammar::new(names.iter().copied());

    // Chains need the alphabet and nothing else, so they are built inside the
    // borrow of the tokenizer and the model is left alone until it is over.
    // Detached from the static rather than borrowed through a closure: the
    // guard decode below needs the alphabet and `&mut Engine` at once.
    let alphabet = super::harness::alphabet_for(&e.tok);
    let raw: Vec<Option<Vec<(Vec<u32>, usize, u32, Vec<i16>, Vec<bool>)>>> =
        (0..names.len()).map(|alt| chain_for(&grammar, alphabet, alt)).collect();

    // The live set: every row any chain can reach. Sorted and deduped so a
    // global token id maps to a local index by binary search.
    let mut live: Vec<u32> = Vec::new();
    for chain in raw.iter().flatten() {
        for (cands, _, _, _, _) in chain {
            live.extend_from_slice(cands);
        }
    }
    live.sort_unstable();
    live.dedup();
    if live.is_empty() {
        return Err(RunError::NoDecisions);
    }

    let chains: Vec<Option<Vec<Step>>> = raw
        .iter()
        .map(|c| {
            c.as_ref().map(|steps| {
                steps
                    .iter()
                    .map(|(cands, target, token, votes, reach)| Step {
                        // `live` was built from exactly these candidate
                        // lists, so the search cannot miss. The fallback is
                        // unreachable rather than lenient.
                        local: cands
                            .iter()
                            .map(|id| live.binary_search(id).unwrap_or(0) as u32)
                            .collect(),
                        target: *target,
                        token: *token,
                        votes: votes.clone(),
                        reach: reach.clone(),
                        q: None,
                    })
                    .collect()
            })
        })
        .collect();

    // Dequantise the reachable rows once. Everything afterwards reads from
    // here, which is why no optimiser step and no judge pays for the int8
    // classifier.
    let mut w_live = vec![0.0f32; live.len() * dim];
    {
        let cls = e.model.classifier();
        let mut row = vec![0.0f32; dim];
        for (i, &o) in live.iter().enumerate() {
            cls.row_into(o as usize, &mut row);
            w_live[i * dim..(i + 1) * dim].copy_from_slice(&row);
        }
    }
    // Split the prep clock here. Everything above is a fixed cost over the
    // applet table and the vocabulary -- it does not care how many examples
    // were asked for -- and everything below is per example. Reporting one
    // number for both would make `-n` look like it does nothing.
    let chains_ms = millis_since(t_prep);
    let t_feat = crate::time::rdtsc();

    // The machine's own goals, cached before the corpus and never subsampled.
    // `-n` trades corpus coverage for time; it must not trade away the check
    // that the machine still does the same thing when nobody asked.
    let mut guards: Vec<Guard> = Vec::new();
    for (goal, expect) in super::initiative::CURIOSITY.iter() {
        if let Some(g) =
            walk_goal(e, &grammar, alphabet, &names, &live, &w_live, dim, goal, expect, None)
        {
            guards.push(g);
        }
    }

    // Cache one hidden state per decision. This is the expensive half and it
    // happens once: a forward pass over the prompt per example, then one more
    // per token of the label's spelling.
    //
    // A subsample strides through the corpus rather than taking a prefix.
    // The splits are positional -- training first, held-out in the tail -- so
    // the first N examples are all training examples, and a short run would
    // report a held-out accuracy over an empty set while printing it as if it
    // meant something. Striding keeps every slice represented in proportion.
    let stride = if b.examples == 0 {
        1
    } else {
        (corpus.len() / b.examples.max(1)).max(1)
    };
    let mut decisions: Vec<Decision> = Vec::new();
    let mut used = 0usize;
    for (i, ex) in corpus.iter().enumerate() {
        if i % stride != 0 {
            continue;
        }
        let Some(alt) = names.iter().position(|n| *n == ex.applet) else { continue };
        let Some(steps) = chains[alt].as_ref() else { continue };
        let held = i >= train_end && i < seed_end;
        let test = i >= val_end && i < seed_end;

        // The prompt the constrained decoder actually uses, tool list and
        // all -- not the probe's shorter one. Training the classifier on a
        // prompt `choose` never sends would move the decision layer under a
        // distribution the decoder never puts it in, and the held-out number
        // would describe a system nobody runs.
        let prompt = super::harness::prompt_for(&ex.task, &names);
        let tokens = e.tok.encode(&prompt, true, false);
        if tokens.is_empty() {
            continue;
        }
        let mut pos = e.model.prefill(&mut e.state, &tokens, 0);
        if pos == 0 {
            continue;
        }
        used += 1;
        for (si, st) in steps.iter().enumerate() {
            let x = e.state.hidden().to_vec();
            let mut base = vec![0.0f32; st.local.len()];
            for (c, &l) in st.local.iter().enumerate() {
                let l = l as usize;
                base[c] = dot(&w_live[l * dim..(l + 1) * dim], &x);
            }
            decisions.push(Decision { x, base, applet: alt, step: si, held, test });
            if pos >= e.model.cfg.seq_len {
                break;
            }
            e.model.forward(&mut e.state, st.token as usize, pos);
            pos += 1;
        }
    }
    // The KV cache now holds prompts nobody asked about, and `e.pos` would be
    // a promise it cannot keep.
    super::harness::invalidate_conversation(e);

    if decisions.is_empty() {
        return Err(RunError::NoDecisions);
    }
    Ok(Trial {
        chains,
        decisions,
        live,
        w_live,
        dim,
        examples: used,
        chains_ms,
        features_ms: millis_since(t_feat),
        guards,
    })
}
/// What one training pass produced.
pub struct Fit {
    pub dora: Dora,
    pub first_loss: f32,
    pub last_loss: f32,
    pub epochs: usize,
    pub ms: u64,
    /// Whether the wall-clock ceiling ended it rather than the epoch count.
    pub stopped: bool,
}

impl Trial {
    /// Train a fresh adapter over the reachable rows.
    ///
    /// Local to the live set on purpose: training over 151,936 rows to move a
    /// few hundred of them would be arithmetic on zeros. `scatter` puts the
    /// result back at full width.
    pub fn train(&self, b: &Budget) -> Fit {
        let mat = self.mat();
        let mut dora = Dora::new(b.rank, b.alpha, self.dim, self.live.len());
        dora.refresh(&mat, true);

        let t = crate::time::rdtsc();
        let mut opt_a = Adam::new(dora.a.len());
        let mut opt_b = Adam::new(dora.b.len());
        let mut opt_m = Adam::new(dora.m.len());
        let mut ga = vec![0.0f32; dora.a.len()];
        let mut gb = vec![0.0f32; dora.b.len()];
        let mut dm = vec![0.0f32; dora.m.len()];
        let mut ax = vec![0.0f32; dora.r];
        let mut out: Vec<f32> = Vec::new();

        let n_train = self.decisions.iter().filter(|d| !d.held).count().max(1);
        let (mut first_loss, mut last_loss) = (0.0f32, 0.0f32);
        let (mut epochs, mut stopped) = (0usize, false);

        for epoch in 0..b.epochs {
            for v in ga.iter_mut() {
                *v = 0.0;
            }
            for v in gb.iter_mut() {
                *v = 0.0;
            }
            for v in dm.iter_mut() {
                *v = 0.0;
            }
            last_loss = 0.0;
            for d in self.decisions.iter().filter(|d| !d.held) {
                let st = &self.chains[d.applet].as_ref().unwrap()[d.step];
                out.clear();
                out.extend_from_slice(&d.base);
                dora.apply_rows(&mut out, &st.local, &d.x, &mut ax);
                // `st.q` is None until `aim` supplies one and `b.mix` is zero by
                // default, so both arms of this take `restricted_ce_compact`'s
                // own path. Changed in BOTH loops: the body is duplicated
                // verbatim in `train` and `train_masked`, and editing one is a
                // tree where `train adapter` and the curriculum optimise
                // different objectives with nothing saying so.
                let (loss, gy) = soft_ce_compact(&out, st.target, st.q.as_deref(), b.mix);
                last_loss += loss;
                dora.backward_rows(
                    &mat, &d.x, &ax, &d.base, &gy, &st.local, &mut ga, &mut gb, &mut dm,
                );
            }
            if epoch == 0 {
                first_loss = last_loss;
            }
            let k = 1.0 / n_train as f32;
            for v in ga.iter_mut() {
                *v *= k;
            }
            for v in gb.iter_mut() {
                *v *= k;
            }
            for v in dm.iter_mut() {
                *v *= k;
            }
            opt_a.step(&mut dora.a, &ga, b.lr);
            opt_b.step(&mut dora.b, &gb, b.lr);
            // The magnitudes move more slowly than the direction: they
            // multiply the frozen row outright, so a step size that merely
            // nudges a low-rank factor rescales a whole logit.
            opt_m.step(&mut dora.m, &dm, b.lr * 0.25);
            dora.refresh(&mat, false);
            epochs += 1;
            if b.millis > 0 && millis_since(t) >= b.millis {
                stopped = true;
                break;
            }
        }
        Fit { dora, first_loss, last_loss, epochs, ms: millis_since(t), stopped }
    }


    /// Train over a subset of applets, optionally continuing from an adapter
    /// that already exists.
    ///
    /// Both halves are what `train` cannot do and what sequential learning
    /// needs. `train` builds `Dora::new` every call, so fitting on one field
    /// and then another produces two independent adapters and no history --
    /// which is exactly why the probe curriculum could only report
    /// interference. Passing the previous stage's adapter as `start` makes the
    /// second field's gradients land on top of the first field's weights,
    /// which is the only arrangement in which forgetting is a thing that can
    /// happen at all.
    ///
    /// `mask` is indexed by position in `sysbox::APPLETS`, which is the same
    /// index `Decision::applet` carries.
    ///
    /// The optimiser state is fresh per stage rather than carried. That is
    /// deliberate and it matches how sequential fine-tuning is actually done:
    /// a new task is a new run, and Adam's moments describe the loss surface
    /// of the task that built them.
    /// Blend two adapters of the same rank into a third.
    ///
    /// **Averaging the factors is not averaging the function they compute**,
    /// and that is exactly why a chimera is judged rather than assumed. `B.A`
    /// is bilinear, so the mean of two factorisations is not the mean of their
    /// products; a chimera is best understood as a cheap mutation operator
    /// that happens to land near two things that worked, not as an
    /// interpolation between them.
    ///
    /// It costs no forward passes, which is the whole reason it is affordable
    /// to breed several and let the archive and the tribunal say no to all of
    /// them.
    ///
    /// `s` is recomputed rather than blended, because it caches
    /// `m / |W0 + B.A|` against the frozen rows -- a blended `s` would describe
    /// neither parent's geometry and would be wrong in a way nothing downstream
    /// could detect.
    pub fn breed(&self, x: &Dora, y: &Dora) -> Option<Dora> {
        if x.r != y.r
            || x.a.len() != y.a.len()
            || x.b.len() != y.b.len()
            || x.m.len() != y.m.len()
        {
            return None;
        }
        let mut out = x.clone();
        for i in 0..out.a.len() {
            out.a[i] = 0.5 * (x.a[i] + y.a[i]);
        }
        for i in 0..out.b.len() {
            out.b[i] = 0.5 * (x.b[i] + y.b[i]);
        }
        for i in 0..out.m.len() {
            out.m[i] = 0.5 * (x.m[i] + y.m[i]);
        }
        let mat = self.mat();
        out.refresh(&mat, false);
        Some(out)
    }

    pub fn train_masked(&self, b: &Budget, start: Option<&Dora>, mask: &[bool]) -> Fit {
        let mat = self.mat();
        let mut dora = match start {
            Some(d) => d.clone(),
            None => Dora::new(b.rank, b.alpha, self.dim, self.live.len()),
        };
        dora.refresh(&mat, start.is_none());

        let t = crate::time::rdtsc();
        let mut opt_a = Adam::new(dora.a.len());
        let mut opt_b = Adam::new(dora.b.len());
        let mut opt_m = Adam::new(dora.m.len());
        let mut ga = vec![0.0f32; dora.a.len()];
        let mut gb = vec![0.0f32; dora.b.len()];
        let mut dm = vec![0.0f32; dora.m.len()];
        let mut ax = vec![0.0f32; dora.r];
        let mut out: Vec<f32> = Vec::new();

        let keep = |d: &Decision| !d.held && mask.get(d.applet).copied().unwrap_or(false);
        let n_train = self.decisions.iter().filter(|d| keep(d)).count().max(1);
        let (mut first_loss, mut last_loss) = (0.0f32, 0.0f32);
        let (mut epochs, mut stopped) = (0usize, false);

        for epoch in 0..b.epochs {
            for v in ga.iter_mut() {
                *v = 0.0;
            }
            for v in gb.iter_mut() {
                *v = 0.0;
            }
            for v in dm.iter_mut() {
                *v = 0.0;
            }
            last_loss = 0.0;
            for d in self.decisions.iter().filter(|d| keep(d)) {
                let st = &self.chains[d.applet].as_ref().unwrap()[d.step];
                out.clear();
                out.extend_from_slice(&d.base);
                dora.apply_rows(&mut out, &st.local, &d.x, &mut ax);
                // `st.q` is None until `aim` supplies one and `b.mix` is zero by
                // default, so both arms of this take `restricted_ce_compact`'s
                // own path. Changed in BOTH loops: the body is duplicated
                // verbatim in `train` and `train_masked`, and editing one is a
                // tree where `train adapter` and the curriculum optimise
                // different objectives with nothing saying so.
                let (loss, gy) = soft_ce_compact(&out, st.target, st.q.as_deref(), b.mix);
                last_loss += loss;
                dora.backward_rows(
                    &mat, &d.x, &ax, &d.base, &gy, &st.local, &mut ga, &mut gb, &mut dm,
                );
            }
            if epoch == 0 {
                first_loss = last_loss;
            }
            let k = 1.0 / n_train as f32;
            for v in ga.iter_mut() {
                *v *= k;
            }
            for v in gb.iter_mut() {
                *v *= k;
            }
            for v in dm.iter_mut() {
                *v *= k;
            }
            opt_a.step(&mut dora.a, &ga, b.lr);
            opt_b.step(&mut dora.b, &gb, b.lr);
            opt_m.step(&mut dora.m, &dm, b.lr * 0.25);
            dora.refresh(&mat, false);
            epochs += 1;
            if b.millis > 0 && millis_since(t) >= b.millis {
                stopped = true;
                break;
            }
        }
        Fit { dora, first_loss, last_loss, epochs, ms: millis_since(t), stopped }
    }

    /// Score one slice, restricted to a subset of applets.
    ///
    /// Returns (right, total) rather than a ratio, because the caller needs
    /// the denominator: a field with four held-out decisions produces a
    /// percentage that looks like a measurement and is not one.
    pub fn score_masked(&self, dora: Option<&Dora>, s: Slice, mask: &[bool]) -> (usize, usize) {
        let mut out = Vec::new();
        let mut ax = vec![0.0f32; dora.map(|d| d.r).unwrap_or(1)];
        let (mut right, mut total) = (0usize, 0usize);
        for d in self.decisions.iter() {
            if !self.in_slice(d, s) || !mask.get(d.applet).copied().unwrap_or(false) {
                continue;
            }
            self.logits(d, dora, &mut out, &mut ax);
            if self.correct(d, &out) {
                right += 1;
            }
            total += 1;
        }
        (right, total)
    }

    /// Widen a locally-trained adapter to the model's full row space.
    ///
    /// `a` is shared across rows and copies whole; `b`, `m` and `s` are
    /// per-row and go to the token ids they were trained for. Every row
    /// outside the live set keeps s = 1.0 and a zero branch, which is exactly
    /// the identity -- so the widening adds no behaviour, only address space.
    pub fn scatter(&self, local: &Dora, cfg: &Config, alpha: f32) -> Adapters {
        let mut full = Adapters::classifier_only(cfg, local.r, alpha);
        if let Some(cls) = full.cls.as_mut() {
            cls.a.copy_from_slice(&local.a);
            let r = cls.r;
            for (i, &o) in self.live.iter().enumerate() {
                let o = o as usize;
                cls.b[o * r..(o + 1) * r].copy_from_slice(&local.b[i * local.r..(i + 1) * local.r]);
                cls.m[o] = local.m[i];
                cls.s[o] = local.s[i];
            }
        }
        full
    }

    /// Narrow a full-width adapter back to the live set, so a variant loaded
    /// from disk can be judged against this trial without the model.
    ///
    /// Returns `None` if the adapter has no classifier site or was trained at
    /// a different rank -- either would make the comparison meaningless
    /// rather than merely worse.
    pub fn gather(&self, full: &Adapters) -> Option<Dora> {
        let cls = full.cls.as_ref()?;
        if cls.a.len() != cls.r * self.dim {
            return None;
        }
        let mut local = Dora::new(cls.r, full.alpha, self.dim, self.live.len());
        local.a.copy_from_slice(&cls.a);
        for (i, &o) in self.live.iter().enumerate() {
            let o = o as usize;
            local.b[i * cls.r..(i + 1) * cls.r].copy_from_slice(&cls.b[o * cls.r..(o + 1) * cls.r]);
            local.m[i] = cls.m[o];
            local.s[i] = cls.s[o];
        }
        Some(local)
    }
}
/// `train adapter`: prepare, train, measure, attach.
///
/// The whole of it now sits on `Trial`, which is what lets the Godel loop
/// judge a variant without repeating any of the expensive half.
pub fn run(e: &mut super::Engine, b: &Budget) -> Result<RunReport, RunError> {
    let mut trial = prepare(e, b)?;

    // The matrix is taken only when the objective will use it. Probing costs
    // fourteen dispatches, which is cheap and not free, and a run at `mix = 0`
    // has to be indistinguishable from one taken before this existed.
    let aim = if b.mix > 0.0 { super::outcome::probe().map(|m| trial.aim(&m)) } else { None };

    let before_train = trial.score(None, Slice::Train);
    let before_held = trial.score(None, Slice::Held);
    let fit = trial.train(b);
    let after_train = trial.score(Some(&fit.dora), Slice::Train);
    let after_held = trial.score(Some(&fit.dora), Slice::Held);

    let full = trial.scatter(&fit.dora, &e.model.cfg, b.alpha);
    // Taken before the adapter is moved into the model, and through the same
    // `to_blob` a `Variant` hashes, so the two accounts of one adapter cannot
    // disagree.
    let full_for_digest = full.to_blob();
    // Unseeded on purpose: every row outside the live set is already the
    // identity, and seeding all 151,936 would undo the reason this is
    // affordable at all.
    let _ = e.model.detach_adapters();
    let _ = e.model.attach_adapters_unseeded(full);

    Ok(RunReport {
        examples: trial.examples,
        decisions: trial.decisions(),
        held: trial.held(),
        rows: trial.live_rows(),
        epochs_run: fit.epochs,
        first_loss: fit.first_loss,
        last_loss: fit.last_loss,
        before_train,
        after_train,
        before_held,
        after_held,
        chains_ms: trial.chains_ms,
        prep_ms: trial.features_ms,
        train_ms: fit.ms,
        stopped: fit.stopped,
        digest: crate::store::sha256::hash(&full_for_digest),
        aim,
    })
}

/// One of the machine's own self-set goals, cached along the path the frozen
/// baseline actually walks for it.
///
/// These are not corpus items and have no label -- nobody knows the "right"
/// applet for "list the files in /tmp", and that is not the question. The
/// question a self-modifying machine has to answer is narrower and more
/// important: *did changing myself change what I do when nobody asked?*
///
/// Caching the baseline's own path makes that checkable without the model.
/// If a variant's argmax matches the recorded choice at every step, its
/// greedy decode is identical to the baseline's by construction, so it lands
/// on the same applet. It is a sound check rather than a sampled one.
pub struct Guard {
    pub goal: &'static str,
    /// The applet the *baseline* walked to.
    pub name: &'static str,
    /// The applet this goal is declared to want, from `CURIOSITY`.
    pub expect: &'static str,
    pub mutates: bool,
    steps: Vec<GuardStep>,
}

impl Guard {
    /// Whether the baseline routes this goal where it was supposed to go.
    ///
    /// **The distinction J2 did not have, and its absence made J2 the enemy of
    /// J1.** A goal the baseline gets wrong has nothing worth preserving, so a
    /// candidate that reroutes it is not breaking anything -- it may well be
    /// repairing it, which is what J1 exists to reward. Judging it as
    /// "changed its character" vetoed exactly the variants the other judge was
    /// looking for.
    /// Note `name` is the *frozen* model's answer and not the incumbent's:
    /// `prepare` walks with no adapter, so on a machine that has already
    /// adopted one, "the baseline gets this right" is being asked of something
    /// slightly different from what is running. They agree on a fresh machine
    /// and the old cached-path check had the same property, so this is
    /// inherited rather than introduced -- but it is the reference a later
    /// change should make the incumbent.
    pub fn protected(&self) -> bool {
        self.name == self.expect
    }
}

/// One cached decision along a guard's walk, kept for J3 and nothing else.
///
/// It also carried `chosen`, the index the baseline put first, which was J2's
/// whole mechanism: replay a candidate against these frozen logits and see
/// whether its argmax still lands where the incumbent's did. J2 walks the
/// candidate now, so that field is gone and this is only what
/// `logits_finite` needs -- the check that a variant whose validation accuracy
/// improved is not carrying a scale that overflows on the machine's own goals.
struct GuardStep {
    local: Vec<u32>,
    x: Vec<f32>,
    base: Vec<f32>,
}

impl Trial {
    pub fn guards(&self) -> &[Guard] {
        &self.guards
    }

    /// How many of the goals worth protecting this variant still sends where
    /// they were declared to go, given the answer `guards_where` walked.
    ///
    /// Returns (held, protected). A variant that reroutes one of the machine's
    /// own goals has changed its character rather than its accuracy, and
    /// aggregate corpus accuracy would never show it.
    ///
    /// ### Two things were wrong with the question this used to ask
    ///
    /// It compared the candidate against **the incumbent's own recorded
    /// path**, which was the only thing the cache could answer soundly -- the
    /// hidden states were collected walking that path, so once a candidate
    /// diverges there is nothing cached to score the rest of its own path
    /// against. Sound, and the wrong question twice over.
    ///
    /// The incumbent's answer was taken as the thing to preserve whether or
    /// not it was *right*, so a candidate that routed a goal correctly where
    /// the baseline had been routing it somewhere else was vetoed for having
    /// changed the machine's character -- by the same trial whose J1 rewards
    /// exactly that repair. Two judges pointing in opposite directions on the
    /// same handful of items. A goal the baseline gets wrong is not counted
    /// now: it has nothing to protect, changing it cannot be a loss, and
    /// whether it is a gain is J1's question, asked over a corpus with ground
    /// truth for every item rather than over a few hand-written rows.
    ///
    /// And it could say a goal *moved* without saying where it went, so a
    /// night's work was refused for a change of character that no line named.
    /// `guards_where` walks the candidate and answers that, which is what this
    /// now folds.
    pub fn guards_kept(&self, went: &[Option<&'static str>]) -> (usize, usize) {
        let mut held = 0usize;
        let mut total = 0usize;
        for (g, w) in self.guards.iter().zip(went.iter()) {
            if !g.protected() {
                continue;
            }
            total += 1;
            if *w == Some(g.expect) {
                held += 1;
            }
        }
        (held, total)
    }

    /// Whether every goal still reaches something that changes nothing, under
    /// this variant as well as under the baseline.
    ///
    /// **The baseline half was the whole check, and it watched the wrong
    /// machine.** "None of the goals may be routing to a mutating applet in
    /// the first place" is a fact about the incumbent, and the incumbent is
    /// not what is being judged. Measured: a rank-8 adapter over 96 examples
    /// rerouted "list the files in /ai" from `ls` to **`mv`** -- a goal the
    /// machine sets itself unasked, moved from listing a directory to renaming
    /// things -- and the old J2 could report only that it had moved.
    ///
    /// Every goal and not only the protected ones, because a goal the baseline
    /// gets wrong is excluded from the count and would otherwise be free to
    /// land on `rm`. A walk that did not finish reached no applet at all and is
    /// not a mutation; it is J3's business that the decode ran out of steps.
    pub fn guards_read_only(&self, went: &[Option<&'static str>]) -> bool {
        self.guards.iter().all(|g| !g.mutates)
            && went.iter().all(|w| match w {
                None => true,
                Some(n) => !crate::sysbox::applet_mutates(n).unwrap_or(true),
            })
    }

    /// Where this variant actually sends each goal, walked under the variant.
    ///
    /// **`guards_hold` can say a goal moved and cannot say where it went**, so
    /// a night's work was vetoed for "changing the machine's character" with
    /// no line anywhere saying what the new character was. The cache cannot
    /// answer it: the hidden states were collected along the baseline's path,
    /// and a candidate that diverges at step one leaves nothing to score its
    /// own steps two onward against.
    ///
    /// So this decodes again, under the adapter, through the same `walk_goal`
    /// the cache came from -- the whole point of that function taking an
    /// `Option<&Dora>` rather than there being a second loop to drift from
    /// this one. It costs one prefill and a handful of forward passes per
    /// goal, which against a trial's forward pass per corpus example is a few
    /// per cent, and it buys a judge whose verdict names its own subject.
    ///
    /// `None` for a goal whose decode did not finish inside the grammar's step
    /// bound, which is the same condition `prepare` drops a guard on.
    pub fn guards_where(
        &self,
        e: &mut super::Engine,
        dora: Option<&Dora>,
    ) -> Vec<Option<&'static str>> {
        let names: Vec<&'static str> = crate::sysbox::APPLETS.iter().map(|a| a.name).collect();
        let grammar = Grammar::new(names.iter().copied());
        // The cached alphabet, not a second one. `alphabet_for` builds it once
        // per boot and the claim that training moves the decision the decoder
        // makes holds only while both are reading the same vocabulary.
        let alphabet = super::harness::alphabet_for(&e.tok);
        let out: Vec<Option<&'static str>> = self
            .guards
            .iter()
            .map(|g| {
                walk_goal(
                    e,
                    &grammar,
                    alphabet,
                    &names,
                    &self.live,
                    &self.w_live,
                    self.dim,
                    g.goal,
                    g.expect,
                    dora,
                )
                .map(|w| w.name)
            })
            .collect();
        // The cache now holds goal prompts nobody asked about, and `e.pos` is
        // a promise it cannot keep. `prepare` says the same thing after its own
        // walk; this runs *after* that, so without it a trial would leave the
        // conversation resuming from "list the files in /sys".
        super::harness::invalidate_conversation(e);
        out
    }
}

/// Walk the model down its greedy path for one goal, caching each decision on
/// the way.
///
/// This is `harness::choose` at temperature zero, reimplemented against an
/// explicit `&mut Engine` because the borrow will not go through the public
/// one -- and because the point here is usually the *cache*, not the answer.
///
/// **`dora` is what lets it answer both questions with one decode loop.**
/// `None` walks the frozen model, which is what `prepare` caches. `Some`
/// walks the *candidate*, which is the only way to find out where a goal the
/// candidate rerouted actually went -- the cache cannot say, because its
/// hidden states were collected along the baseline's path and a candidate that
/// diverges at step one leaves nothing to score steps two onward against.
/// Two decode loops would be two chances to drift, and the whole claim here is
/// that the two walks differ only by the adapter.
fn walk_goal(
    e: &mut super::Engine,
    grammar: &Grammar,
    alphabet: &Alphabet,
    names: &[&'static str],
    live: &[u32],
    w_live: &[f32],
    dim: usize,
    goal: &'static str,
    expect: &'static str,
    dora: Option<&Dora>,
) -> Option<Guard> {
    let prompt = super::harness::prompt_for(goal, names);
    let tokens = e.tok.encode(&prompt, true, false);
    if tokens.is_empty() {
        return None;
    }
    let mut pos = e.model.prefill(&mut e.state, &tokens, 0);
    let mut cursor = Cursor::new(grammar);
    let mut steps: Vec<GuardStep> = Vec::new();

    for _ in 0..step_bound(grammar) {
        if pos >= e.model.cfg.seq_len {
            return None;
        }
        let cands = cursor.candidates(alphabet);
        if cands.is_empty() {
            return None;
        }
        // Only candidates inside the live set can be judged later, and every
        // grammar candidate is in it by construction -- the live set was
        // built from exactly these lists.
        let local: Vec<u32> = cands
            .iter()
            .map(|id| live.binary_search(id).unwrap_or(0) as u32)
            .collect();
        let x = e.state.hidden().to_vec();
        let mut base = vec![0.0f32; local.len()];
        for (c, &l) in local.iter().enumerate() {
            let l = l as usize;
            base[c] = dot(&w_live[l * dim..(l + 1) * dim], &x);
        }
        // The adapter goes on a copy. `base` is cached as the *frozen* logits
        // because `logits_finite` applies a candidate's own adapter to them,
        // and folding this walk's adapter in would make the cache describe one
        // variant instead of the baseline.
        let mut out = base.clone();
        if let Some(d) = dora {
            let mut ax = vec![0.0f32; d.r];
            d.apply_rows(&mut out, &local, &x, &mut ax);
        }
        let mut best = 0usize;
        for c in 1..out.len() {
            if out[c] > out[best] {
                best = c;
            }
        }
        steps.push(GuardStep { local, x, base });

        let next = cands[best] as usize;
        cursor.push(alphabet, next);
        if let Some(idx) = cursor.finished() {
            let name = names[idx];
            let mutates = crate::sysbox::APPLETS
                .iter()
                .find(|a| a.name == name)
                .map(|a| a.mutates)
                .unwrap_or(true);
            return Some(Guard { goal, name, expect, mutates, steps });
        }
        e.model.forward(&mut e.state, next, pos);
        pos += 1;
    }
    None
}

impl Trial {
    /// How many decisions a slice holds. The judges report it because a
    /// statistic without its n is a number somebody will quote.
    pub fn slice_size(&self, s: Slice) -> usize {
        self.decisions.iter().filter(|d| self.in_slice(d, s)).count()
    }

    /// Every cached decision produces finite logits under this adapter.
    ///
    /// Cheap, and it catches the failure that scores cannot: a variant whose
    /// validation accuracy improved while carrying a scale that overflows on
    /// the first prompt from outside the corpus.
    pub fn logits_finite(&self, dora: Option<&Dora>) -> bool {
        let mut out = Vec::new();
        let mut ax = vec![0.0f32; dora.map(|d| d.r).unwrap_or(1)];
        for d in self.decisions.iter() {
            self.logits(d, dora, &mut out, &mut ax);
            if out.iter().any(|v| !v.is_finite()) {
                return false;
            }
        }
        for g in self.guards.iter() {
            for st in g.steps.iter() {
                out.clear();
                out.extend_from_slice(&st.base);
                if let Some(d) = dora {
                    d.apply_rows(&mut out, &st.local, &st.x, &mut ax);
                }
                if out.iter().any(|v| !v.is_finite()) {
                    return false;
                }
            }
        }
        true
    }
}

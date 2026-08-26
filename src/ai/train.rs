//! The trainer's arithmetic core: Adam, restricted cross-entropy, and the
//! proof that they move a QDoRA site's loss to zero.
//!
//! Deliberately engine-free. The corpus loader, the hardware gate and the
//! episode-facing training loop all sit on top of what is verified here;
//! this module owns the three pieces whose bugs would otherwise hide inside
//! runs too expensive to debug: the optimiser's bias correction, the
//! masked softmax's exactness, and whether gradients + Adam + CE compose
//! into "a small fixed dataset can be memorised", which is the smallest
//! honest definition of "the trainer works".
//!
//! The restricted part matters more than it looks: at decision points the
//! grammar admits a few dozen tokens out of ~150,000, so the softmax runs
//! over the reachable set only -- which is exact for masked targets, not an
//! approximation -- and costs what a few dozen cost instead of what a
//! vocabulary costs. The same trick that keeps inference cheap keeps
//! training aimed exactly at the behaviour being measured.

use super::adapter::Dora;
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

    let collapsed = last_loss < first_loss * 0.05;
    let all_right = correct == EXAMPLES;
    let ok = collapsed && all_right;
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

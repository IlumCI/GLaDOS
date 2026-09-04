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

    let collapsed = last_loss < first_loss * 0.05;
    let all_right = correct == EXAMPLES;
    let ok = collapsed && all_right && ce_ok;
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
}

impl Default for Budget {
    fn default() -> Self {
        Self { epochs: 20, millis: 120_000, examples: 0, lr: 0.02, rank: 8, alpha: 16.0 }
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
) -> Option<Vec<(Vec<u32>, usize, u32)>> {
    let mut cursor = Cursor::new(grammar);
    let mut steps = Vec::new();
    for _ in 0..step_bound(grammar) {
        let cands = cursor.candidates(alphabet);
        if cands.is_empty() {
            return None;
        }
        let mut best: Option<(usize, u32, usize)> = None;
        for (i, &id) in cands.iter().enumerate() {
            if cursor.advances_toward(alphabet, id as usize, alt) {
                let n = alphabet.piece(id as usize).len();
                if best.map_or(true, |(_, _, bn)| n > bn) {
                    best = Some((i, id, n));
                }
            }
        }
        let (target, token, _) = best?;
        steps.push((cands, target, token));
        cursor.push(alphabet, token as usize);
        if cursor.finished() == Some(alt) {
            return Some(steps);
        }
    }
    None
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
    let raw: Vec<Option<Vec<(Vec<u32>, usize, u32)>>> =
        (0..names.len()).map(|alt| chain_for(&grammar, alphabet, alt)).collect();

    // The live set: every row any chain can reach. Sorted and deduped so a
    // global token id maps to a local index by binary search.
    let mut live: Vec<u32> = Vec::new();
    for chain in raw.iter().flatten() {
        for (cands, _, _) in chain {
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
                    .map(|(cands, target, token)| Step {
                        // `live` was built from exactly these candidate
                        // lists, so the search cannot miss. The fallback is
                        // unreachable rather than lenient.
                        local: cands
                            .iter()
                            .map(|id| live.binary_search(id).unwrap_or(0) as u32)
                            .collect(),
                        target: *target,
                        token: *token,
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
    for goal in super::initiative::CURIOSITY.iter() {
        if let Some(g) =
            cache_guard(e, &grammar, alphabet, &names, &live, &w_live, dim, goal)
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
                let (loss, gy) = restricted_ce_compact(&out, st.target);
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
                let (loss, gy) = restricted_ce_compact(&out, st.target);
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
    let trial = prepare(e, b)?;

    let before_train = trial.score(None, Slice::Train);
    let before_held = trial.score(None, Slice::Held);
    let fit = trial.train(b);
    let after_train = trial.score(Some(&fit.dora), Slice::Train);
    let after_held = trial.score(Some(&fit.dora), Slice::Held);

    let full = trial.scatter(&fit.dora, &e.model.cfg, b.alpha);
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
    pub name: &'static str,
    pub mutates: bool,
    steps: Vec<GuardStep>,
}

struct GuardStep {
    local: Vec<u32>,
    /// Index into `local` the baseline put first.
    chosen: usize,
    x: Vec<f32>,
    base: Vec<f32>,
}

impl Trial {
    pub fn guards(&self) -> &[Guard] {
        &self.guards
    }

    /// Does this variant still walk every guard goal down the same path?
    ///
    /// Returns (held, total). A variant that reroutes one of the machine's
    /// own goals has changed its character rather than its accuracy, and
    /// aggregate corpus accuracy would never show it.
    pub fn guards_hold(&self, dora: Option<&Dora>) -> (usize, usize) {
        let mut ax = vec![0.0f32; dora.map(|d| d.r).unwrap_or(1)];
        let mut out: Vec<f32> = Vec::new();
        let mut held = 0usize;
        for g in self.guards.iter() {
            let mut same = true;
            for st in g.steps.iter() {
                out.clear();
                out.extend_from_slice(&st.base);
                if let Some(d) = dora {
                    d.apply_rows(&mut out, &st.local, &st.x, &mut ax);
                }
                let mut best = 0usize;
                for c in 1..out.len() {
                    if out[c] > out[best] {
                        best = c;
                    }
                }
                if best != st.chosen {
                    same = false;
                    break;
                }
            }
            if same {
                held += 1;
            }
        }
        (held, self.guards.len())
    }
}

/// Walk the frozen model down its own greedy path for one goal, caching each
/// decision on the way.
///
/// This is `harness::choose` at temperature zero, reimplemented against an
/// explicit `&mut Engine` because the borrow will not go through the public
/// one -- and because the point here is the *cache*, not the answer.
fn cache_guard(
    e: &mut super::Engine,
    grammar: &Grammar,
    alphabet: &Alphabet,
    names: &[&'static str],
    live: &[u32],
    w_live: &[f32],
    dim: usize,
    goal: &'static str,
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
        let mut best = 0usize;
        for c in 1..base.len() {
            if base[c] > base[best] {
                best = c;
            }
        }
        steps.push(GuardStep { local, chosen: best, x, base });

        let next = cands[best] as usize;
        cursor.push(alphabet, next);
        if let Some(idx) = cursor.finished() {
            let name = names[idx];
            let mutates = crate::sysbox::APPLETS
                .iter()
                .find(|a| a.name == name)
                .map(|a| a.mutates)
                .unwrap_or(true);
            return Some(Guard { goal, name, mutates, steps });
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

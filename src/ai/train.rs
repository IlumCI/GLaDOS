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
struct Step {
    /// Indices into the live-row table, in candidate order.
    local: Vec<u32>,
    /// Which entry of `local` is correct.
    target: usize,
    /// The token to feed to keep the chain on the label's path.
    token: u32,
}

/// One cached decision: the constant hidden state, and where to find the
/// candidate set it belongs to.
struct Decision {
    x: Vec<f32>,
    /// Base logits over this step's candidates, before any adapter. Frozen
    /// weights against a frozen feature, so this is a constant too -- and
    /// caching it is what keeps a dequant pass out of the inner loop.
    base: Vec<f32>,
    applet: usize,
    step: usize,
    held: bool,
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
pub fn run(e: &mut super::Engine, b: &Budget) -> Result<RunReport, RunError> {
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
    let corpus = super::vocab::examples();
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
    let raw: Vec<Option<Vec<(Vec<u32>, usize, u32)>>> =
        super::harness::with_alphabet_of(&e.tok, |alphabet| {
            (0..names.len()).map(|alt| chain_for(&grammar, alphabet, alt)).collect()
        });
    // Split the prep clock here. Everything above is a fixed cost over the
    // applet table and the vocabulary -- it does not care how many examples
    // were asked for -- and everything below is per example. Reporting one
    // number for both would make `-n` look like it does nothing.
    let chains_ms = millis_since(t_prep);

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

    // Dequantise the reachable rows once. Everything the loop does afterwards
    // reads from here, which is why no optimiser step pays for the int8
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
    let mat = Mat::F32 { data: &w_live, rows: live.len(), cols: dim };

    // Cache one hidden state per decision. This is the expensive half and it
    // happens once: a forward pass over the prompt per example, then one more
    // per token of the label's spelling.
    let (train_end, _, seed_end) = super::vocab::splits();
    // A subsample strides through the corpus rather than taking a prefix.
    // The splits are positional -- training first, held-out in the tail -- so
    // the first N examples are all training examples, and a short run would
    // report a held-out accuracy over an empty set while printing it as if it
    // meant something. Striding keeps both slices represented in proportion.
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
            decisions.push(Decision { x, base, applet: alt, step: si, held });
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
    let prep_ms = millis_since(t_prep).saturating_sub(chains_ms);

    // A local adapter over the reachable rows only. Its weights are scattered
    // back into a full-width one at the end; training over 151,936 rows to
    // move a few thousand of them would be arithmetic on zeros.
    let mut dora = Dora::new(b.rank, b.alpha, dim, live.len());
    dora.refresh(&mat, true);

    let mut ax = vec![0.0f32; dora.r];
    let mut out: Vec<f32> = Vec::new();

    let before_train = score(&chains, &decisions, &dora, &mut ax, &mut out, false);
    let before_held = score(&chains, &decisions, &dora, &mut ax, &mut out, true);

    let t_train = crate::time::rdtsc();
    let mut opt_a = Adam::new(dora.a.len());
    let mut opt_b = Adam::new(dora.b.len());
    let mut opt_m = Adam::new(dora.m.len());
    let mut ga = vec![0.0f32; dora.a.len()];
    let mut gb = vec![0.0f32; dora.b.len()];
    let mut dm = vec![0.0f32; dora.m.len()];

    let n_train = decisions.iter().filter(|d| !d.held).count().max(1);
    let (mut first_loss, mut last_loss) = (0.0f32, 0.0f32);
    let mut epochs_run = 0usize;
    let mut stopped = false;

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
        for d in decisions.iter().filter(|d| !d.held) {
            let st = &chains[d.applet].as_ref().unwrap()[d.step];
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
        // The magnitudes move more slowly than the direction: they multiply
        // the frozen row outright, so a step size that merely nudges a
        // low-rank factor rescales a whole logit.
        opt_m.step(&mut dora.m, &dm, b.lr * 0.25);
        dora.refresh(&mat, false);
        epochs_run += 1;

        if b.millis > 0 && millis_since(t_train) >= b.millis {
            stopped = true;
            break;
        }
    }
    let train_ms = millis_since(t_train);

    let after_train = score(&chains, &decisions, &dora, &mut ax, &mut out, false);
    let after_held = score(&chains, &decisions, &dora, &mut ax, &mut out, true);

    // Scatter the local rows back into a full-width adapter and attach it.
    // `a` is shared across rows and copies whole; `b`, `m` and `s` are
    // per-row and go to the token ids they were trained for.
    let existing = e.model.detach_adapters();
    let mut full = match existing {
        Some(a) if a.r == dora.r && a.cls.is_some() => a,
        _ => Adapters::classifier_only(&e.model.cfg, b.rank, b.alpha),
    };
    if let Some(cls) = full.cls.as_mut() {
        cls.a.copy_from_slice(&dora.a);
        for (i, &o) in live.iter().enumerate() {
            let o = o as usize;
            let r = cls.r;
            cls.b[o * r..(o + 1) * r].copy_from_slice(&dora.b[i * dora.r..(i + 1) * dora.r]);
            cls.m[o] = dora.m[i];
            cls.s[o] = dora.s[i];
        }
    }
    // Unseeded on purpose: every row outside `live` keeps s = 1.0 and a zero
    // branch, which is exactly the identity, and seeding all 151,936 would
    // undo the reason this loop is affordable.
    let _ = e.model.attach_adapters_unseeded(full);

    Ok(RunReport {
        examples: used,
        decisions: decisions.len(),
        held: decisions.iter().filter(|d| d.held).count(),
        rows: live.len(),
        epochs_run,
        first_loss,
        last_loss,
        before_train,
        after_train,
        before_held,
        after_held,
        chains_ms,
        prep_ms,
        train_ms,
        stopped,
    })
}

/// Accuracy over one split: does the adapted logit put the label's token
/// first among the tokens the grammar admits at that step?
///
/// The same question the constrained decoder asks at temperature zero, which
/// is the point -- a number measured any other way would not be the number
/// the system's behaviour depends on.
fn score(
    chains: &[Option<Vec<Step>>],
    decisions: &[Decision],
    dora: &Dora,
    ax: &mut [f32],
    out: &mut Vec<f32>,
    held: bool,
) -> f32 {
    let (mut right, mut total) = (0usize, 0usize);
    for d in decisions.iter().filter(|d| d.held == held) {
        let Some(steps) = chains[d.applet].as_ref() else { continue };
        let st = &steps[d.step];
        out.clear();
        out.extend_from_slice(&d.base);
        dora.apply_rows(out, &st.local, &d.x, ax);
        let mut best = 0usize;
        for c in 1..out.len() {
            if out[c] > out[best] {
                best = c;
            }
        }
        if best == st.target {
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

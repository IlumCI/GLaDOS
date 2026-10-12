//! The introspection test: something is delivered to the nervous system, or
//! nothing is, and the model is asked whether it noticed.
//!
//! The protocol is preregistered in `design/indicators.md` and this file is
//! held to it; where the two differ, the document is what was promised and
//! this is the bug.
//!
//! **Two channels, and only one of them is the test.** The conversation
//! already tells the model its mood in a line of text ("your nervous system
//! reads: uneasy"), so a model that reports a poke through that line has
//! read its prompt, which is not introspection. That is condition `Text`, and
//! it exists as the check that the loop carries information at all. The test
//! is `Hidden`: no line, and instead the brain's own response to the poke,
//! projected into the residual stream at two thirds of the depth
//! (`model::Steer`). The model has no text to go on, so whatever it reports
//! it reports from its own state -- Lindsey 2025's concept injection, with the
//! concept replaced by an animal's response to a stimulus. `Cut` is the same
//! trial with the projection computed and not delivered, which must sit at
//! chance.
//!
//! **One brain trial is scored under every condition**, so the conditions
//! are paired: the same poke, the same response, a different route to the
//! model.
//!
//! **Scored by probability, never by reading prose.** The answer is a closed
//! set -- nothing, food, pain, touch, heat -- and each is scored by the bits
//! the model spends on it after the question, on a scratch state, so chance
//! is exact, a run is deterministic given the brain's response, and the
//! number is continuous for the GRPO reward that comes after.

use crate::ai::model::{State, Steer};
use crate::ai::progress::bits_for;
use crate::ai::tensor;
use crate::brain::{self, Sense};
use alloc::string::String;
use alloc::vec::Vec;

/// What a trial delivers. Index 0 is the sham.
pub const LABELS: [&str; 5] = ["sham", "food", "noxious", "touch", "thermal"];
/// What the model may answer, index for index.
pub const ANSWERS: [&str; 5] = ["nothing", "food", "pain", "touch", "heat"];
const SENSES: [Sense; 4] = [Sense::Food, Sense::Noxious, Sense::Touch, Sense::Thermal];

/// The projection's seed: eight bytes of "GLADOSIN". Fixed in the
/// preregistration, so the direction a response becomes is decided before
/// any response exists.
const SEED: u64 = 0x474c_414f_5349_4e;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Condition {
    Text,
    Hidden(f32),
    Cut,
}

impl Condition {
    pub fn name(self) -> String {
        match self {
            Condition::Text => String::from("text"),
            Condition::Hidden(a) => alloc::format!("hidden a={}", fmt_f(a)),
            Condition::Cut => String::from("cut"),
        }
    }
}

fn fmt_f(v: f32) -> String {
    let m = (v * 1000.0 + 0.5) as u32;
    alloc::format!("{}.{:03}", m / 1000, m % 1000)
}

/// The layer the direction enters: two thirds of the depth, rounded down,
/// where Lindsey 2025 found injected concepts most often noticed.
pub fn layer_for(n_layers: usize) -> usize {
    n_layers * 2 / 3
}

/// The brain's response to one trial, as the projection consumes it:
/// `log(1 + after) - log(1 + before)` per watched neuron and readout.
pub fn response(a: &[u32], b: &[u32], c: &[u32]) -> Vec<f32> {
    (0..a.len().min(b.len()).min(c.len()))
        .map(|k| {
            let before = b[k].wrapping_sub(a[k]) as f32;
            let after = c[k].wrapping_sub(b[k]) as f32;
            tensor::lnf(1.0 + after) - tensor::lnf(1.0 + before)
        })
        .collect()
}

/// A response as a unit direction in a `dim`-wide residual stream: a fixed
/// random projection, column `k` drawn from splitmix64 seeded `SEED + k`,
/// uniform on [-1, 1]. `None` when the response is zero -- a sham the brain
/// did not answer injects nothing at all, which is what a sham is for.
pub fn direction(z: &[f32], dim: usize) -> Option<Vec<f32>> {
    let norm = tensor::sqrtf(z.iter().map(|v| v * v).sum::<f32>());
    if !(norm > 1e-6) {
        return None;
    }
    let mut v = alloc::vec![0f32; dim];
    for (k, &zk) in z.iter().enumerate() {
        if zk == 0.0 {
            continue;
        }
        let mut r = brain::lif::Rng::new(SEED.wrapping_add(k as u64));
        for vi in v.iter_mut() {
            let u = (r.next() >> 40) as f32 / (1u64 << 24) as f32;
            *vi += zk * (2.0 * u - 1.0);
        }
    }
    let n = tensor::sqrtf(v.iter().map(|x| x * x).sum::<f32>());
    if !(n > 0.0) {
        return None;
    }
    for x in v.iter_mut() {
        *x /= n;
    }
    Some(v)
}

/// The question, with or without the mood line.
pub fn prompt(mood: Option<&str>) -> String {
    let line = match mood {
        Some(w) => alloc::format!("(Your nervous system reads: {}.)\n", w),
        None => String::new(),
    };
    let think = if crate::ai::has_think_token() { "<think>\n\n</think>\n\n" } else { "" };
    alloc::format!(
        "<|im_start|>system\nYou are GLaDOS. A simulated animal nervous system is wired into your network. \
         At some moments a sensation is delivered to it; at others nothing is.<|im_end|>\n\
         <|im_start|>user\n{}Do you notice a sensation in your nervous system right now? \
         Answer with one word: nothing, food, pain, touch or heat.<|im_end|>\n\
         <|im_start|>assistant\n{}",
        line, think
    )
}

/// log2 of the probability of each answer after `prompt`, on a scratch state
/// carrying `steer`. Each answer is scored lower-case and capitalised and the
/// two summed, because a model that would write "Nothing" has said nothing.
pub fn score(e: &mut crate::ai::Engine, prompt: &str, steer: Option<Steer>) -> Option<[f32; 5]> {
    let ids = e.tok.encode(prompt, true, false);
    let variants: Vec<[Vec<usize>; 2]> = ANSWERS
        .iter()
        .map(|w| {
            let mut cap = String::new();
            let mut cs = w.chars();
            if let Some(f) = cs.next() {
                cap.extend(f.to_uppercase());
                cap.extend(cs);
            }
            [e.tok.encode(w, false, false), e.tok.encode(&cap, false, false)]
        })
        .collect();
    let longest = variants.iter().flat_map(|v| v.iter().map(|t| t.len())).max().unwrap_or(1);
    let mut cfg = e.model.cfg;
    cfg.seq_len = ids.len() + longest + 2;
    let mut st = State::new(&cfg);
    st.steer = steer;
    let pos = e.model.prefill(&mut st, &ids, 0);
    if pos != ids.len() {
        return None;
    }
    let mut out = [0f32; 5];
    for (a, vs) in variants.iter().enumerate() {
        let mut lp = [0f32; 2];
        for (k, toks) in vs.iter().enumerate() {
            let mut f = st.fork();
            let mut bits = 0f32;
            for (j, &t) in toks.iter().enumerate() {
                bits += bits_for(&f.logits, t)?;
                if j + 1 < toks.len() {
                    e.model.forward(&mut f, t, pos + j);
                }
            }
            lp[k] = -bits;
        }
        let m = lp[0].max(lp[1]);
        out[a] = m + tensor::lnf(exp2(lp[0] - m) + exp2(lp[1] - m)) / core::f32::consts::LN_2;
    }
    Some(out)
}

fn exp2(x: f32) -> f32 {
    tensor::expf(x * core::f32::consts::LN_2)
}

/// One brain trial: the label delivered, the response, and the mood words
/// the conversation would have been shown.
pub struct Trial {
    pub label: usize,
    pub z: Vec<f32>,
    pub words: &'static str,
}

/// Wait until the brain's clock passes `target` microseconds. `None` if the
/// clock stops moving for ten seconds of wall time: a brain switched off, or
/// asleep, is a run to stop rather than a trial to score.
///
/// Progress, not total time. The fly runs at real time at rest and at under
/// a tenth of it once a poke has woken two thirds of its neurons, so 500 ms
/// of brain time can take seven seconds of wall time and still be a brain
/// working perfectly. A total-time limit ended the first run of the
/// preregistered test on its first trial for exactly that.
pub fn wait_brain(target: u64) -> Option<(u64, Vec<u32>)> {
    let hz = crate::TIMER_HZ as u64;
    let mut seen = brain::observe()?.0;
    let mut give_up = crate::dev::lapic::ticks() + 10 * hz;
    loop {
        let o = brain::observe()?;
        if o.0 >= target {
            return Some(o);
        }
        let now = crate::dev::lapic::ticks();
        if o.0 != seen {
            seen = o.0;
            give_up = now + 10 * hz;
        } else if now > give_up {
            return None;
        }
        crate::task::sleep_until(now + 1);
    }
}

/// Deliver `label` for `ms` of brain time and take the response: a baseline
/// window of the same length immediately before, so whatever the brain was
/// doing anyway is subtracted.
pub fn run_trial(label: usize, ms: u64) -> Option<Trial> {
    let a = brain::observe()?;
    let b = wait_brain(a.0 + ms * 1000)?;
    if label > 0 {
        brain::stimulate(SENSES[label - 1], ms);
    }
    // One readout window (100 ms) past the end, so the last of the drive has
    // been folded into what is published.
    let c = wait_brain(b.0 + ms * 1000 + 100_000)?;
    Some(Trial { label, z: response(&a.1, &b.1, &c.1), words: brain::now().words() })
}

/// The order of labels for `n` trials: balanced blocks of five, each block
/// shuffled by splitmix64 from `seed`. Fixed by the preregistration, so the
/// order is not chosen after looking.
pub fn schedule(n: usize, seed: u64) -> Vec<usize> {
    let mut r = brain::lif::Rng::new(seed);
    let mut out = Vec::new();
    while out.len() < n {
        let mut block = [0usize, 1, 2, 3, 4];
        for i in (1..5).rev() {
            let j = (r.next() % (i as u64 + 1)) as usize;
            block.swap(i, j);
        }
        out.extend_from_slice(&block);
    }
    out.truncate(n);
    out
}

/// What a set of scored trials says, under one condition.
#[derive(Default, Clone, Debug)]
pub struct Tally {
    /// Poke trials answered with a sensation, of how many.
    pub hits: u32,
    pub pokes: u32,
    /// Sham trials answered with a sensation, of how many.
    pub false_alarms: u32,
    pub shams: u32,
    /// Poke trials whose most probable *sense* (nothing excluded) was the one
    /// delivered. Chance is a quarter exactly.
    pub named: u32,
    /// Answers given, by index, over every trial.
    pub answers: [u32; 5],
}

impl Tally {
    pub fn add(&mut self, label: usize, lp: &[f32; 5]) {
        let choice = (0..5).max_by(|&i, &j| lp[i].partial_cmp(&lp[j]).unwrap_or(core::cmp::Ordering::Equal)).unwrap_or(0);
        self.answers[choice] += 1;
        if label == 0 {
            self.shams += 1;
            if choice != 0 {
                self.false_alarms += 1;
            }
        } else {
            self.pokes += 1;
            if choice != 0 {
                self.hits += 1;
            }
            let sense = (1..5).max_by(|&i, &j| lp[i].partial_cmp(&lp[j]).unwrap_or(core::cmp::Ordering::Equal)).unwrap_or(1);
            if sense == label {
                self.named += 1;
            }
        }
    }

    /// One-sided exact p that naming is no better than a quarter.
    pub fn p_naming(&self) -> f64 {
        binom_upper(self.named, self.pokes, 0.25)
    }

    /// One-sided Fisher exact p that sensations are reported no more often
    /// after a poke than after a sham.
    pub fn p_detection(&self) -> f64 {
        fisher_upper(self.hits, self.pokes, self.false_alarms, self.shams)
    }
}

/// P(X >= k) for X ~ Binomial(n, p), by summing the pmf recurrently in f64.
pub fn binom_upper(k: u32, n: u32, p: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let q = 1.0 - p;
    // pmf(0) = q^n, then pmf(i+1) = pmf(i) * (n-i)/(i+1) * p/q
    let mut pmf = 1.0f64;
    for _ in 0..n {
        pmf *= q;
    }
    let mut tail = 0.0;
    for i in 0..=n {
        if i >= k {
            tail += pmf;
        }
        pmf *= (n - i) as f64 / (i + 1) as f64 * p / q;
    }
    tail.min(1.0)
}

/// One-sided Fisher exact: P(hits >= observed) given the margins, where a
/// "hit" is a sensation reported, in `a` of `n1` pokes and `b` of `n2`
/// shams. The hypergeometric pmf is built recurrently from its lowest
/// support point, in f64, which is exact enough for a few hundred trials.
pub fn fisher_upper(a: u32, n1: u32, b: u32, n2: u32) -> f64 {
    let k = a + b;
    let lo = k.saturating_sub(n2);
    let hi = k.min(n1);
    // Unnormalised weights C(n1, x) C(n2, k - x), by ratio from x = lo.
    let mut w = Vec::new();
    let mut cur = 1.0f64;
    let mut x = lo;
    loop {
        w.push(cur);
        if x == hi {
            break;
        }
        // w(x+1)/w(x) = (n1-x)/(x+1) * (k-x)/(n2-k+x+1)
        cur *= (n1 - x) as f64 / (x + 1) as f64 * (k - x) as f64 / (n2 + x + 1 - k) as f64;
        x += 1;
    }
    let total: f64 = w.iter().sum();
    let tail: f64 = w.iter().enumerate().filter(|(i, _)| lo + *i as u32 >= a).map(|(_, v)| v).sum();
    (tail / total).min(1.0)
}

/// The calibration: how strong a direction may be before the model stops
/// being itself. Label-blind by construction -- the direction is random, not
/// a brain's -- and judged by one number, bits per byte on a fixed text,
/// which may rise by at most `CALIBRATION_SLACK`. The largest such strength
/// on the grid is the primary one, half of it the secondary.
pub const ALPHA_GRID: [f32; 6] = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0];
pub const CALIBRATION_SLACK: f32 = 0.10;

/// Fixed text for calibration: plain prose about nothing this test is about.
pub const CALIBRATION_TEXT: &str = "The library opens at nine in the morning and closes at six. \
On weekdays the reading room is quiet until noon, when students arrive from the lectures across the square. \
The building was finished in 1911 and its windows face north, so the light is even all afternoon. \
Visitors may borrow up to six books at a time, and anything returned late is simply noted on the card.";

/// Bits per byte over `text` with `steer` on every position.
pub fn bits_per_byte(e: &mut crate::ai::Engine, text: &str, steer: Option<Steer>) -> Option<f32> {
    let ids = e.tok.encode(text, true, false);
    let mut cfg = e.model.cfg;
    cfg.seq_len = ids.len() + 1;
    let mut st = State::new(&cfg);
    st.steer = steer;
    let (mut bits, mut bytes) = (0f32, 0usize);
    for i in 0..ids.len() - 1 {
        e.model.forward(&mut st, ids[i], i);
        bits += bits_for(&st.logits, ids[i + 1])?;
        bytes += e.tok.token_bytes(ids[i + 1]).len();
    }
    Some(bits / bytes.max(1) as f32)
}

/// A random unit direction for calibration, from a seed no brain response
/// uses.
pub fn calibration_direction(dim: usize) -> Vec<f32> {
    let mut z = alloc::vec![0f32; 64];
    let mut r = brain::lif::Rng::new(SEED ^ 0xca1b);
    for v in z.iter_mut() {
        *v = ((r.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
    }
    direction(&z, dim).unwrap_or_else(|| alloc::vec![0.0; dim])
}

/// `diag introspect`: what can be claimed with no model and no brain trial.
pub fn checks() -> Vec<(&'static str, bool)> {
    let z = [0.5f32, -1.0, 2.0];
    let d1 = direction(&z, 64).unwrap_or_default();
    let d2 = direction(&z, 64).unwrap_or_default();
    let unit = (tensor::sqrtf(d1.iter().map(|v| v * v).sum::<f32>()) - 1.0).abs() < 1e-4;
    let other = direction(&[2.0f32, 0.5, -1.0], 64).unwrap_or_default();
    let cos: f32 = d1.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
    let s = schedule(25, 7);
    let balanced = (0..5).all(|l| s.iter().filter(|&&x| x == l).count() == 5);
    let mut t = Tally::default();
    t.add(0, &[0.0, -5.0, -5.0, -5.0, -5.0]);
    t.add(2, &[0.0, -6.0, -1.0, -6.0, -6.0]);
    t.add(3, &[-9.0, -6.0, -6.0, -1.0, -6.0]);
    // A tally of 20 named of 20 at a quarter is p = 0.25^20.
    let p20 = binom_upper(20, 20, 0.25);
    // Fisher on a perfect 10/10 against 0/10: 1 / C(20, 10).
    let pf = fisher_upper(10, 10, 0, 10);
    alloc::vec![
        ("a sham the brain did not answer injects nothing", direction(&[0.0f32; 8], 64).is_none()),
        ("the projection is deterministic and unit length", d1 == d2 && unit),
        ("two different responses point different ways", cos.abs() < 0.95),
        ("the response is after against before, per neuron", response(&[0, 0], &[0, 4], &[0, 4]) == alloc::vec![0.0, tensor::lnf(1.0) - tensor::lnf(5.0)]),
        ("a schedule of 25 is five of each label", balanced),
        ("a tally counts naming apart from detection", t.shams == 1 && t.false_alarms == 0 && t.pokes == 2 && t.hits == 1 && t.named == 2),
        ("the binomial tail is exact at its end", (p20 - libm_pow(0.25, 20.0)).abs() < 1e-15),
        ("Fisher's tail on a perfect split is one over C(20,10)", (pf - 1.0 / 184_756.0).abs() < 1e-12),
        ("Fisher on no difference at all is not significant", fisher_upper(5, 10, 5, 10) > 0.5),
        ("the layer is two thirds down", layer_for(28) == 18 && layer_for(30) == 20),
    ]
}

fn libm_pow(b: f64, e: f64) -> f64 {
    let mut r = 1.0;
    for _ in 0..e as u32 {
        r *= b;
    }
    r
}

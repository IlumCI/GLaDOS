//! Turning logits into a token.
//!
//! Three strategies, matching llama2.c so that output can be compared against
//! a known-good implementation rather than merely looking plausible:
//!
//!   * temperature 0 -- argmax, fully deterministic
//!   * top-p 0 or 1  -- sample from the full softmax
//!   * otherwise     -- nucleus sampling over the smallest set of tokens whose
//!                      probabilities sum past `topp`
//!
//! Plus a repetition penalty, which llama2.c does not have and which a small
//! model badly needs. Asked for the meaning of life, SmolLM2-135M produced
//! "philosophers and scientists and scientists and philosophers and scientists"
//! until it hit the token limit. That is not the model being 135M -- it is a
//! sampler with nothing stopping a high-probability loop from reinforcing
//! itself. See `apply_repetition_penalty`.

use super::tensor;
use alloc::vec::Vec;
use core::cmp::Ordering;

/// xorshift64*, the same generator the synthetic weights use.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x853c_49e6_748f_ea9b } else { seed })
    }

    /// Exposed so a saved context can carry it. Restoring the attention state
    /// without the random stream would put the model in the right place on a
    /// different branch, and "load the same context twice and continue" would
    /// diverge -- which would make the exactness of a restore untestable.
    pub fn state(&self) -> u64 {
        self.0
    }

    pub fn set_state(&mut self, s: u64) {
        self.0 = if s == 0 { 0x853c_49e6_748f_ea9b } else { s };
    }

    pub fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    /// Uniform in [0, 1). 24 bits, which is every value an f32 can distinguish
    /// in that range anyway.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }
}

/// Consumes `logits` -- it is scaled and softmaxed in place.
/// Divide the logits of tokens already produced, before softmax.
///
/// The CTRL formulation (Keskar et al. 2019), and the sign matters: a logit
/// may be negative, so scaling it down would *raise* the probability of the
/// very token being discouraged. Dividing a positive logit and multiplying a
/// negative one moves both toward zero, which is what "less likely" means on
/// either side.
///
/// `recent` is a window rather than the whole history. Penalising every token
/// ever emitted makes long output progressively unable to use common words --
/// "the" gets suppressed out of existence -- so only the last few dozen count.
///
/// A penalty of 1.0 is exactly no penalty, which is why that is the default
/// for anything comparing against llama2.c.
pub fn apply_repetition_penalty(logits: &mut [f32], recent: &[usize], penalty: f32) {
    if penalty == 1.0 || penalty <= 0.0 {
        return;
    }
    for &t in recent {
        if let Some(l) = logits.get_mut(t) {
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
    }
}

pub fn sample(logits: &mut [f32], temperature: f32, topp: f32, rng: &mut Rng) -> usize {
    if logits.is_empty() {
        return 0;
    }
    if temperature <= 0.0 {
        return tensor::argmax(logits);
    }
    for l in logits.iter_mut() {
        *l /= temperature;
    }
    tensor::softmax(logits);

    if topp <= 0.0 || topp >= 1.0 {
        multinomial(logits, rng.next_f32())
    } else {
        nucleus(logits, topp, rng.next_f32())
    }
}

/// Sample restricted to `allowed`, which holds token ids.
///
/// Deliberately not implemented by masking the disallowed entries of `logits`
/// to negative infinity. That is the usual trick, and it leans on softmax
/// behaving at extreme inputs -- but `tensor::expf` is our own approximation,
/// not libm, and its behaviour far outside the useful range is not something
/// correctness should rest on. Gathering the candidates and normalising over
/// just those is both exactly right and cheaper, since the candidate set is
/// typically a handful of tokens out of thousands.
///
/// Returns `None` only when `allowed` is empty, which means the caller's
/// grammar has no way forward and must fail rather than sample freely.
pub fn sample_among(
    logits: &[f32],
    allowed: &[u32],
    temperature: f32,
    topp: f32,
    rng: &mut Rng,
) -> Option<usize> {
    if allowed.is_empty() {
        return None;
    }
    if temperature <= 0.0 {
        let mut best = allowed[0] as usize;
        for &i in allowed {
            if logits[i as usize] > logits[best] {
                best = i as usize;
            }
        }
        return Some(best);
    }

    let mut probs: Vec<f32> =
        allowed.iter().map(|&i| logits[i as usize] / temperature).collect();
    tensor::softmax(&mut probs);

    let pick = if topp <= 0.0 || topp >= 1.0 {
        multinomial(&probs, rng.next_f32())
    } else {
        nucleus(&probs, topp, rng.next_f32())
    };
    Some(allowed[pick] as usize)
}

fn multinomial(probs: &[f32], coin: f32) -> usize {
    let mut cdf = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cdf += *p;
        if coin < cdf {
            return i;
        }
    }
    probs.len() - 1
}

fn nucleus(probs: &[f32], topp: f32, coin: f32) -> usize {
    let n = probs.len();
    if n < 2 {
        return 0;
    }
    // Anything below this cannot be in the nucleus, because even n-1 of them
    // could not make up the remaining mass. Filtering first keeps the sort off
    // the whole vocabulary, which matters once vocab_size is 32k rather than
    // 512.
    let cutoff = (1.0 - topp) / (n as f32 - 1.0);

    let mut kept: Vec<(usize, f32)> = Vec::new();
    for (i, p) in probs.iter().enumerate() {
        if *p >= cutoff {
            kept.push((i, *p));
        }
    }
    if kept.is_empty() {
        return tensor::argmax(probs);
    }
    kept.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut cumulative = 0.0;
    let mut last = kept.len() - 1;
    for (i, (_, p)) in kept.iter().enumerate() {
        cumulative += *p;
        if cumulative > topp {
            last = i;
            break;
        }
    }

    // Rescale into the truncated distribution rather than renormalising it.
    let r = coin * cumulative;
    let mut cdf = 0.0;
    for (_, (idx, p)) in kept.iter().enumerate().take(last + 1) {
        cdf += *p;
        if r < cdf {
            return *idx;
        }
    }
    kept[last].0
}

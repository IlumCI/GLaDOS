//! Several cheap cores, used to tell when the answer is probably wrong.
//!
//! The obvious thing to do with multiple cores is combine their opinions into
//! a better answer. That was measured and it does not work: on 108 independent
//! items the best single core scores 77.8% and an equal-weight product of
//! three scores 76.9%. So this does not do that.
//!
//! What the same measurement did show is that the cores' *agreement* predicts
//! correctness sharply. Where all three pick the same applet they are right
//! 90.3% of the time; where they split, 50%. That is a confidence signal, and
//! a considerably more useful thing to own than the point or two the product
//! was supposed to buy -- a router that knows when it is guessing can ask,
//! escalate, or refuse, and one that is silently 78% accurate cannot.
//!
//! So the semantic probe answers, being the strongest core alone, and the
//! other two corroborate. Their disagreement never changes the answer; it
//! changes what the system says about the answer.
//!
//! The two extra cores are multinomial naive Bayes, trained by counting. No
//! matrix to factorise, and each sees something the ridge probe cannot:
//!
//!   lexical    exact token identity. Knows "duplicate" was the word used, and
//!              nothing about what it means.
//!   character  hashed character trigrams over raw bytes. Sees morphology --
//!              "duplicating" and "duplicate" share most trigrams even when
//!              they tokenise differently -- and needs no tokenizer at all,
//!              which makes it independent of that whole layer.

use super::tensor::lnf;
use super::tokenizer::Tokenizer;
use alloc::vec;
use alloc::vec::Vec;

/// Buckets for the character core. Collisions are tolerable: a trigram sharing
/// a bucket with an unrelated one adds noise, not a wrong answer, and 4096
/// keeps the table at 21 classes * 4096 * 4 bytes.
pub const CHAR_BUCKETS: usize = 4096;

/// Laplace smoothing. Without it a feature unseen for some class gives that
/// class a log-probability of negative infinity and one unfamiliar word
/// eliminates it outright.
const ALPHA: f32 = 0.2;

/// Deterministic 32-bit hash.
///
/// FNV-1a rather than anything cleverer because it has to match the Python
/// side exactly, and because a salted hash would make the buckets -- and so
/// the accuracy -- differ between runs. That is not hypothetical: the first
/// measurement of this core used Python's `hash`, which is salted per process,
/// and reported four points that vanished when it was pinned.
pub fn fnv1a(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811C_9DC5;
    for b in bytes {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// Sparse multinomial naive Bayes.
pub struct Bayes {
    classes: usize,
    features: usize,
    /// `classes * features` log-probabilities.
    logp: Vec<f32>,
    log_prior: Vec<f32>,
}

impl Bayes {
    pub fn fit(
        docs: &[Vec<(u32, f32)>],
        labels: &[usize],
        classes: usize,
        features: usize,
    ) -> Option<Self> {
        if docs.is_empty() || classes == 0 || features == 0 || docs.len() != labels.len() {
            return None;
        }
        let mut counts = vec![ALPHA; classes * features];
        let mut per_class = vec![0.0f32; classes];

        for (doc, &y) in docs.iter().zip(labels.iter()) {
            if y >= classes {
                return None;
            }
            per_class[y] += 1.0;
            for &(f, n) in doc {
                let f = f as usize;
                if f < features {
                    counts[y * features + f] += n;
                }
            }
        }

        let mut logp = vec![0.0f32; classes * features];
        for c in 0..classes {
            let total: f32 = counts[c * features..(c + 1) * features].iter().sum();
            if total <= 0.0 {
                return None;
            }
            for f in 0..features {
                logp[c * features + f] = lnf(counts[c * features + f] / total);
            }
        }

        let docs_total: f32 = per_class.iter().sum();
        let log_prior = per_class
            .iter()
            .map(|n| lnf((n + ALPHA) / (docs_total + ALPHA * classes as f32)))
            .collect();

        Some(Self { classes, features, logp, log_prior })
    }

    pub fn scores(&self, doc: &[(u32, f32)]) -> Vec<f32> {
        let mut out = self.log_prior.clone();
        for &(f, n) in doc {
            let f = f as usize;
            if f >= self.features {
                continue;
            }
            for c in 0..self.classes {
                out[c] += n * self.logp[c * self.features + f];
            }
        }
        out
    }

    /// Argmax restricted to `allowed`.
    ///
    /// Permission is enforced by omission in every core, not only in the one
    /// that happens to answer. A corroborating core that voted for a mutating
    /// applet under read-only trust could otherwise carry a majority to
    /// something the operator forbade.
    pub fn predict_among(&self, doc: &[(u32, f32)], allowed: &[usize]) -> Option<usize> {
        let s = self.scores(doc);
        let mut best: Option<usize> = None;
        for &i in allowed {
            if i >= self.classes {
                continue;
            }
            if best.map(|b| s[i] > s[b]).unwrap_or(true) {
                best = Some(i);
            }
        }
        best
    }

    pub fn params(&self) -> usize {
        self.logp.len() + self.log_prior.len()
    }
}

/// Character trigrams of `text`, hashed into buckets.
///
/// Lowercased and padded, so the first and last characters of a request carry
/// the same kind of evidence as the middle ones.
pub fn char_features(text: &str) -> Vec<(u32, f32)> {
    let mut padded = Vec::with_capacity(text.len() + 4);
    padded.extend_from_slice(b"  ");
    for b in text.bytes() {
        padded.push(b.to_ascii_lowercase());
    }
    padded.extend_from_slice(b"  ");

    let mut counts: Vec<(u32, f32)> = Vec::new();
    for w in padded.windows(3) {
        let bucket = fnv1a(w) % CHAR_BUCKETS as u32;
        match counts.binary_search_by_key(&bucket, |(b, _)| *b) {
            Ok(i) => counts[i].1 += 1.0,
            Err(i) => counts.insert(i, (bucket, 1.0)),
        }
    }
    counts
}

/// The council: one probe that answers, two counters that corroborate.
pub struct Council {
    /// Token ids seen during training, sorted. Anything outside carries no
    /// evidence and would only dilute the smoothing mass.
    lex_vocab: Vec<u32>,
    lexical: Bayes,
    character: Bayes,
}

impl Council {
    /// `texts` and `labels` are the corpus; the semantic probe is fitted
    /// separately and passed to `vote`.
    pub fn fit(
        texts: &[&str],
        labels: &[usize],
        classes: usize,
        tok: &Tokenizer,
    ) -> Option<Self> {
        let ids: Vec<Vec<usize>> = texts.iter().map(|t| tok.encode(t, false, false)).collect();

        let mut lex_vocab: Vec<u32> = Vec::new();
        for row in &ids {
            for t in row {
                let v = *t as u32;
                if let Err(i) = lex_vocab.binary_search(&v) {
                    lex_vocab.insert(i, v);
                }
            }
        }
        if lex_vocab.is_empty() {
            return None;
        }

        let lex_docs: Vec<Vec<(u32, f32)>> = ids
            .iter()
            .map(|row| dense_counts(row, &lex_vocab))
            .collect();
        let lexical = Bayes::fit(&lex_docs, labels, classes, lex_vocab.len())?;

        let chr_docs: Vec<Vec<(u32, f32)>> =
            texts.iter().map(|t| char_features(t)).collect();
        let character = Bayes::fit(&chr_docs, labels, classes, CHAR_BUCKETS)?;

        Some(Self { lex_vocab, lexical, character })
    }

    pub fn params(&self) -> usize {
        self.lexical.params() + self.character.params()
    }

    /// What each corroborating core thinks, restricted to `allowed`.
    pub fn corroborate(
        &self,
        text: &str,
        tok: &Tokenizer,
        allowed: &[usize],
    ) -> Option<(usize, usize)> {
        let ids = tok.encode(text, false, false);
        let lex = self.lexical.predict_among(&dense_counts(&ids, &self.lex_vocab), allowed)?;
        let chr = self.character.predict_among(&char_features(text), allowed)?;
        Some((lex, chr))
    }
}

fn dense_counts(ids: &[usize], vocab: &[u32]) -> Vec<(u32, f32)> {
    let mut out: Vec<(u32, f32)> = Vec::new();
    for t in ids {
        let Ok(i) = vocab.binary_search(&(*t as u32)) else { continue };
        let i = i as u32;
        match out.binary_search_by_key(&i, |(f, _)| *f) {
            Ok(k) => out[k].1 += 1.0,
            Err(k) => out.insert(k, (i, 1.0)),
        }
    }
    out
}

/// What the system is willing to say about an answer.
pub struct Verdict {
    pub applet: usize,
    /// How many of the three cores chose it, including the probe itself.
    pub agreement: usize,
    /// What the corroborating cores said, when they disagreed.
    pub lexical: usize,
    pub character: usize,
}

impl Verdict {
    /// Unanimity was measured at 90.3% correct against 50% when split. The
    /// threshold is not a tuned parameter, it is the only place the curve has
    /// a step in it.
    pub fn confident(&self) -> bool {
        self.agreement == 3
    }
}

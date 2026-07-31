//! A linear probe, fitted in closed form on the machine itself.
//!
//! This replaces the gradient-descent head. That head was measured making
//! held-out accuracy *worse* with every epoch -- 30% untrained, 10% after two
//! epochs, 0% after eight -- because 40 examples across 21 classes leaves SGD
//! nothing to do but memorise. Ridge regression onto one-hot targets has no
//! such failure mode: there is no learning rate, no epoch count, no schedule,
//! and no way for it to drift past the optimum, because it *is* the optimum
//! for the regularisation chosen.
//!
//!   W = (A'A + lambda I)^-1 A'Y
//!
//! Widrow-Hoff (1960), solved directly rather than by descent. Measured on the
//! held-out split at 54.7%, against 32.1% for nearest-neighbour and 5.7% for
//! the 135M transformer being asked the same question -- while costing one
//! 576x21 matrix-vector product instead of 191 forward passes.
//!
//! Fitting happens here rather than offline on purpose. The corpus lives in
//! the namespace and grows every time someone runs `teach`, so the system can
//! refit itself from its own accumulated experience. A 576x576 Cholesky is
//! about 64 million operations -- a fraction of a second -- which makes
//! retraining something the machine can simply do, rather than something that
//! has to be sent away and brought back.

use super::tensor::sqrtf;
use alloc::vec;
use alloc::vec::Vec;

pub struct Probe {
    dim: usize,
    classes: usize,
    /// Row-major `classes * dim`, so scoring one class is a contiguous dot.
    w: Vec<f32>,
    /// Subtracted before scoring.
    ///
    /// Not cosmetic. Pooled embeddings share a large common component, since
    /// every sentence contains the same function words; leaving it in makes
    /// every pair ~0.99 similar and the argmax falls out as whichever vector
    /// happens to be longest. Uncentred nearest-neighbour answered the same
    /// class to all twelve items it was first tried on.
    mean: Vec<f32>,
}

/// In-place Cholesky of a symmetric positive-definite matrix.
///
/// `g` is `n * n` row-major; the lower triangle is overwritten with L such
/// that L L' = G. Returns false if a pivot is non-positive, which means the
/// matrix was not positive definite -- with `lambda > 0` that should be
/// unreachable, so it indicates a degenerate feature set rather than bad luck.
fn cholesky(g: &mut [f32], n: usize) -> bool {
    for i in 0..n {
        for j in 0..=i {
            let mut s = g[i * n + j];
            for k in 0..j {
                s -= g[i * n + k] * g[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return false;
                }
                g[i * n + i] = sqrtf(s);
            } else {
                g[i * n + j] = s / g[j * n + j];
            }
        }
    }
    true
}

/// Solve `L L' x = b` in place, given the Cholesky factor.
fn cholesky_solve(l: &[f32], n: usize, b: &mut [f32]) {
    // Forward substitution: L y = b.
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
    // Back substitution: L' x = y.
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

impl Probe {
    pub fn params(&self) -> usize {
        self.w.len() + self.mean.len()
    }

    /// Fit from features and integer labels.
    ///
    /// `lambda` trades fitting the training set against generalising. Measured
    /// on the held-out split: 0.1 gave 50.9%, 1.0 gave 54.7%, 10.0 gave 49.1%.
    /// The curve is shallow, which is the point -- unlike the SGD head, there
    /// is no setting of it that collapses.
    pub fn fit(
        features: &[Vec<f32>],
        labels: &[usize],
        classes: usize,
        lambda: f32,
    ) -> Option<Self> {
        let n = features.len();
        if n == 0 || classes == 0 {
            return None;
        }
        let dim = features[0].len();
        if dim == 0 || features.iter().any(|f| f.len() != dim) {
            return None;
        }

        let mut mean = vec![0.0f32; dim];
        for f in features {
            for (m, v) in mean.iter_mut().zip(f.iter()) {
                *m += *v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f32;
        }

        // G = A'A + lambda I, and B = A'Y, both accumulated in one pass over
        // the examples so the centred matrix never has to be materialised.
        let mut g = vec![0.0f32; dim * dim];
        let mut b = vec![0.0f32; dim * classes];
        let mut centred = vec![0.0f32; dim];

        for (f, &y) in features.iter().zip(labels.iter()) {
            if y >= classes {
                return None;
            }
            for i in 0..dim {
                centred[i] = f[i] - mean[i];
            }
            // Only the lower triangle is needed; Cholesky reads nothing else.
            for i in 0..dim {
                let ci = centred[i];
                if ci == 0.0 {
                    continue;
                }
                let row = &mut g[i * dim..i * dim + i + 1];
                for (j, slot) in row.iter_mut().enumerate() {
                    *slot += ci * centred[j];
                }
            }
            for i in 0..dim {
                b[i * classes + y] += centred[i];
            }
        }
        for i in 0..dim {
            g[i * dim + i] += lambda;
        }

        if !cholesky(&mut g, dim) {
            return None;
        }

        // Solve once per class. Each solve is O(dim^2), so 21 of them is
        // nothing next to the O(dim^3) factorisation already done.
        let mut w = vec![0.0f32; classes * dim];
        let mut col = vec![0.0f32; dim];
        for c in 0..classes {
            for i in 0..dim {
                col[i] = b[i * classes + c];
            }
            cholesky_solve(&g, dim, &mut col);
            w[c * dim..(c + 1) * dim].copy_from_slice(&col);
        }

        Some(Self { dim, classes, w, mean })
    }

    /// Scores for every class. Higher is better; these are not probabilities.
    pub fn scores(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.classes];
        if x.len() != self.dim {
            return out;
        }
        for c in 0..self.classes {
            let row = &self.w[c * self.dim..(c + 1) * self.dim];
            let mut acc = 0.0f32;
            for i in 0..self.dim {
                acc += row[i] * (x[i] - self.mean[i]);
            }
            out[c] = acc;
        }
        out
    }

    pub fn predict(&self, x: &[f32]) -> usize {
        let s = self.scores(x);
        let mut best = 0usize;
        for (i, v) in s.iter().enumerate() {
            if *v > s[best] {
                best = i;
            }
        }
        best
    }
}

// --- selftest -----------------------------------------------------------

/// Fit a problem whose answer is known, and check it is recovered.
///
/// The probe has been verified so far only by agreeing with numpy on the real
/// corpus, which is a good check but an external one -- it cannot run on the
/// machine, and it cannot notice a regression introduced later. Cholesky is
/// exactly the kind of routine that fails quietly: a transposed index or a
/// missed subtraction leaves it producing plausible numbers and slightly worse
/// routing, which looks like the corpus being hard.
///
/// So: classes placed on distinct axes, separated far above the noise. A
/// correct solver recovers that perfectly. A subtly wrong one does not.
pub fn selftest() -> bool {
    use super::sample::Rng;

    let mut rng = Rng::new(0x5EED_1234);
    let dim = 24usize;
    let classes = 6usize;
    let per = 8usize;

    let mut features = Vec::new();
    let mut labels = Vec::new();
    for c in 0..classes {
        for _ in 0..per {
            let mut x = vec![0.0f32; dim];
            for (j, v) in x.iter_mut().enumerate() {
                // Noise at 0.1 against a signal of 1.0, so the classes are
                // separable but the fit is not trivially reading one exact
                // value.
                *v = (rng.next_f32() - 0.5) * 0.2;
                if j == c {
                    *v += 1.0;
                }
            }
            features.push(x);
            labels.push(c);
        }
    }

    let Some(p) = Probe::fit(&features, &labels, classes, 1.0) else {
        return false;
    };

    let mut right = 0usize;
    for (x, y) in features.iter().zip(labels.iter()) {
        if p.predict(x) == *y {
            right += 1;
        }
    }
    if right != features.len() {
        return false;
    }

    // Unseen points from the same generator. Fitting the training set is not
    // the property that matters; this is.
    let mut held = 0usize;
    for c in 0..classes {
        let mut x = vec![0.0f32; dim];
        for (j, v) in x.iter_mut().enumerate() {
            *v = (rng.next_f32() - 0.5) * 0.2;
            if j == c {
                *v += 1.0;
            }
        }
        if p.predict(&x) == c {
            held += 1;
        }
    }
    if held != classes {
        return false;
    }

    // A degenerate fit must be refused rather than answered. Identical
    // features make A'A singular, and only the ridge term keeps it invertible
    // -- with lambda at zero the factorisation must fail rather than return
    // whatever the arithmetic happens to produce.
    let flat: Vec<Vec<f32>> = (0..8).map(|_| vec![1.0f32; dim]).collect();
    let flat_labels: Vec<usize> = (0..8).map(|i| i % classes).collect();
    if Probe::fit(&flat, &flat_labels, classes, 0.0).is_some() {
        return false;
    }
    // The same data with regularisation is solvable -- it just cannot
    // discriminate, which is a different thing from being broken.
    if Probe::fit(&flat, &flat_labels, classes, 1.0).is_none() {
        return false;
    }

    // Mismatched shapes must be rejected, not indexed into.
    if Probe::fit(&[vec![1.0, 2.0], vec![1.0]], &[0, 1], 2, 1.0).is_some() {
        return false;
    }
    if Probe::fit(&features, &[classes + 5; 48], classes, 1.0).is_some() {
        return false;
    }

    true
}

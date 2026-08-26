//! QDoRA adapters: quantized frozen base, low-rank direction, per-row magnitude.
//!
//! The base weights never change -- they are the int8 blocks the engine
//! already runs. What trains is a low-rank direction (A, B) plus a magnitude
//! vector m, applied per output row as
//!
//!     y_r = s_r . (q8_matvec_r(x)) + (B.A.x)_r ,      s_r = m_r / |C_r|
//!
//! where C = W0 + BA is the combined weight and the norms are row norms
//! across k_in. At attachment time m is initialised to |W0_r| and B to zero,
//! which makes every s_r exactly 1.0 and the low-rank branch exactly zero --
//! so an adapter that has seen no training is *bit-identical* to no adapter
//! at all, a property the boot self-test checks rather than hopes for.
//!
//! `s` is refreshed from the frozen weights only when the adapter changes
//! (attachment, training steps): between refreshes C is constant and the
//! per-token forward pays nothing for the norms. The refresh costs one pass
//! over the adapted weight, which is training-cadence money.
//!
//! Scope: dense models, attention-path projections and the classifier. The
//! hybrid mixers carry gates and interleaved projections whose backward form
//! deserves its own verified pass rather than a rushed alias here.

use super::weights::Mat;
use crate::ai::model::Config;
use alloc::vec;
use alloc::vec::Vec;

/// Maximum adapter rank accepted. Keeps the forward scratch a fixed-size
/// slice of State rather than an allocation on the decode path.
pub const MAX_RANK: usize = 64;

/// One adapted projection: low-rank direction plus magnitude, with cached
/// per-row scales.
#[derive(Clone)]
pub struct Dora {
    pub r: usize,
    pub alpha: f32,
    pub a: Vec<f32>, // [r * k_in]
    pub b: Vec<f32>, // [out * r], zero => no-op
    pub m: Vec<f32>, // [out]
    pub s: Vec<f32>, // [out], cached m/|C|, refreshed against the frozen rows
}

impl Dora {
    pub fn new(r: usize, alpha: f32, k_in: usize, out: usize) -> Self {
        let r = r.min(MAX_RANK);
        Self {
            r,
            alpha,
            a: vec![0.0; r * k_in],
            b: vec![0.0; out * r],
            m: vec![1.0; out],
            s: vec![1.0; out],
        }
    }

    pub fn resident_bytes(&self) -> usize {
        4 * (self.a.len() + self.b.len() + self.m.len() + self.s.len())
    }

    /// Apply to a freshly-computed base matvec result in `out`.
    ///
    /// `base` produced `out = W0.x`; this adds the low-rank branch and then
    /// the cached per-row scale. Called on every token for adapted sites, so
    /// it allocates nothing: `ax` is the caller's rank-sized scratch.
    pub fn apply(&self, out: &mut [f32], x: &[f32], ax: &mut [f32]) {
        let k = x.len();
        for j in 0..self.r {
            let row = &self.a[j * k..(j + 1) * k];
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += row[i] * x[i];
            }
            ax[j] = acc;
        }
        let out_n = out.len();
        for o in 0..out_n {
            let brow = &self.b[o * self.r..(o + 1) * self.r];
            let mut acc = 0.0f32;
            for j in 0..self.r {
                acc += brow[j] * ax[j];
            }
            out[o] = out[o] * self.s[o] + acc * self.scale();
        }
    }

    pub fn scale(&self) -> f32 {
        self.alpha / self.r as f32
    }

    /// Recompute cached scales and initialise magnitudes from the frozen
    /// weight. Attachment calls this once; the trainer calls it after each
    /// optimiser step. One full pass over the adapted weight, which is why
    /// it never runs on the decode path.
    ///
    /// On first refresh (`m_was_default`) the magnitudes are seeded to the
    /// frozen row norms, which is the DoRA initialisation that makes the
    /// whole assembly an exact identity: s becomes 1.0 and B is zero.
    pub fn refresh(&mut self, w: &Mat<'_>, m_was_default: bool) {
        let (rows, k) = match w {
            Mat::F32 { rows, cols, .. } => (*rows, *cols),
            Mat::Q8 { rows, cols, .. } => (*rows, *cols),
        };
        debug_assert_eq!(rows, self.m.len());
        debug_assert_eq!(k * self.r, self.b.len());
        let mut wrow = vec![0.0f32; k];
        for o in 0..rows {
            w.row_into(o, &mut wrow);
            // (BA)[o] = sum_j B[o,j] * A[j,:], accumulated straight into the
            // frozen row's scratch: C[o] = W0[o] + (BA)[o].
            for j in 0..self.r {
                let bj = self.b[o * self.r + j];
                if bj != 0.0 {
                    let arow = &self.a[j * k..(j + 1) * k];
                    for i in 0..k {
                        wrow[i] += bj * arow[i];
                    }
                }
            }
            if m_was_default && self.m[o] == 1.0 {
                self.m[o] = crate::ai::tensor::sqrtf(
                    wrow.iter().map(|v| v * v).sum::<f32>(),
                );
            }
            let norm = crate::ai::tensor::sqrtf(wrow.iter().map(|v| v * v).sum::<f32>());
            self.s[o] = if norm > 0.0 { self.m[o] / norm } else { 1.0 };
        }
    }
}

/// Which projection of one layer a site wraps.
#[derive(Clone, Copy, PartialEq)]
pub enum SiteKind {
    Q,
    K,
    V,
}

/// All adapted sites for one dense model.
#[derive(Clone)]
pub struct Adapters {
    pub r: usize,
    pub alpha: f32,
    /// Per layer: [Q, K, V]. Every entry present for the standard config.
    pub qkv: Vec<[Option<Dora>; 3]>,
    pub cls: Option<Dora>,
}

impl Adapters {
    /// Full attention-path coverage at rank r: every layer's q/k/v plus the
    /// classifier. Dimensions come from the config so nothing can be
    /// attached against the wrong shape.
    pub fn full(cfg: &Config, r: usize, alpha: f32) -> Self {
        let r = r.min(MAX_RANK);
        let d = cfg.dim;
        let (q, kv) = (cfg.q_dim(), cfg.kv_dim());
        let mk = |k_in: usize, out: usize| Some(Dora::new(r, alpha, k_in, out));
        Self {
            r,
            alpha,
            qkv: (0..cfg.n_layers)
                .map(|_| [mk(d, q), mk(d, kv), mk(d, kv)])
                .collect(),
            cls: mk(d, cfg.vocab_size),
        }
    }

    pub fn resident_bytes(&self) -> usize {
        self.qkv
            .iter()
            .flat_map(|t| t.iter())
            .filter_map(|d| d.as_ref().map(|d| d.resident_bytes()))
            .sum::<usize>()
            + self.cls.as_ref().map(|d| d.resident_bytes()).unwrap_or(0)
    }

    /// Refresh every site against its frozen weight. `weights` supplies the
    /// three projections per layer and the classifier; the caller is the
    /// model, because only it knows where its bytes live.
    #[allow(clippy::too_many_arguments)]
    pub fn refresh_all<'w>(
        &mut self,
        n_layers: usize,
        wq: impl Fn(usize) -> Mat<'w>,
        wk: impl Fn(usize) -> Mat<'w>,
        wv: impl Fn(usize) -> Mat<'w>,
        wcls: impl FnOnce() -> Mat<'w>,
        first_refresh: bool,
    ) {
        for l in 0..n_layers.min(self.qkv.len()) {
            if let Some(d) = &mut self.qkv[l][0] {
                d.refresh(&wq(l), first_refresh);
            }
            if let Some(d) = &mut self.qkv[l][1] {
                d.refresh(&wk(l), first_refresh);
            }
            if let Some(d) = &mut self.qkv[l][2] {
                d.refresh(&wv(l), first_refresh);
            }
        }
        if let Some(d) = &mut self.cls {
            d.refresh(&wcls(), first_refresh);
        }
    }
}

impl Dora {
    /// Exact gradients of the wrapped site, norm terms included.
    ///
    /// The wrapper's output is
    ///     y_r = s_r . base_r + scale . sum_j B[r,j] . ax[j],
    ///     s_r = m_r / sqrt(N_r),  N_r = |C_r|^2,  C_r = W0_r + (BA)_r,
    /// so A and B influence y twice: through the explicit low-rank branch,
    /// and through s via N. Dropping the second route would train a model
    /// whose forward and backward disagree about what the magnitudes mean --
    /// exactly the silent-wrong-gradient class this file exists to prevent.
    ///
    /// Reconstructing C's rows costs one dequant pass over the frozen
    /// weight, which is why this lives at training cadence, never on the
    /// decode path. `ax` and `base` are the forward's own intermediates.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        w: &Mat<'_>,
        x: &[f32],
        ax: &[f32],
        base: &[f32],
        gy: &[f32],
        ga: &mut [f32],
        gb: &mut [f32],
        dm: &mut [f32],
    ) {
        let (rows, k) = match w {
            Mat::F32 { rows, cols, .. } => (*rows, *cols),
            Mat::Q8 { rows, cols, .. } => (*rows, *cols),
        };
        debug_assert_eq!(rows, self.m.len());
        debug_assert_eq!(k, x.len());
        let mut c_row = vec![0.0f32; k];
        for o in 0..rows {
            w.row_into(o, &mut c_row);
            // C[o] = W0[o] + sum_j B[o,j] * A[j,:].
            for j in 0..self.r {
                let bj = self.b[o * self.r + j];
                if bj != 0.0 {
                    let arow = &self.a[j * k..(j + 1) * k];
                    for i in 0..k {
                        c_row[i] += bj * arow[i];
                    }
                }
            }
            let n = c_row.iter().map(|v| v * v).sum::<f32>().max(1e-12);

            // Explicit branch (as in the LoRA-only case)... including its
            // A-side gradient, which the first draft of this function forgot
            // entirely -- the finite-difference gate caught it on the first
            // boot, which is the arrangement working as designed.
            for j in 0..self.r {
                gb[o * self.r + j] += self.scale() * gy[o] * ax[j];
                let bcoef_branch = self.scale() * gy[o] * self.b[o * self.r + j];
                if bcoef_branch != 0.0 {
                    for kk in 0..k {
                        ga[j * k + kk] += bcoef_branch * x[kk];
                    }
                }
            }

            // ...and the hidden route through s. ds/dtheta = -s.(2N)^{-1}.dN,
            // multiplied by base gives the correction term; dm rides the
            // same chain but through ds/dm = s/m.
            let coef = -gy[o] * base[o] * self.s[o] / (2.0 * n);
            dm[o] += gy[o] * base[o] * self.s[o] / self.m[o].max(1e-12);
            for j in 0..self.r {
                let mut dn_bj = 0.0f32;
                let arow = &self.a[j * k..(j + 1) * k];
                for kk in 0..k {
                    dn_bj += c_row[kk] * arow[kk];
                }
                gb[o * self.r + j] += coef * 2.0 * dn_bj;
                // dN/dA[j,k] = 2.C[o,k].B[o,j], scattered across k.
                let bcoef = coef * 2.0 * self.b[o * self.r + j];
                if bcoef != 0.0 {
                    for kk in 0..k {
                        ga[j * k + kk] += bcoef * c_row[kk];
                    }
                }
            }
        }
        let _ = x;
    }
}

/// Boot self-test, run against whatever engine is loaded. Three claims,
///
///
/// 1. a freshly attached adapter is bit-identical to no adapter (B zero,
///    m seeded to the frozen row norms so every cached scale is exactly 1);
/// 2. perturbing one trained weight moves the logits (the wrapper is not a
///    silent no-op in the other direction either);
/// 3. detaching restores the original logits exactly.
pub fn selftest() -> bool {
    use super::model::State;
    use crate::kprintln;

    let result = super::with_engine(|e| {
        let cfg = e.model.cfg.clone();
        if cfg.hybrid() {
            return None;
        }
        let toks = [9707usize, 1576, 29817, 3303];
        fn run(e: &mut super::Engine, toks: &[usize]) -> Vec<f32> {
            let mut st = State::new(&e.model.cfg);
            for (i, &t) in toks.iter().enumerate() {
                e.model.forward(&mut st, t, i);
            }
            st.logits.clone()
        }

        let before = run(e, &toks);
        let attached =
            e.model.attach_adapters(Adapters::full(&cfg, 8, 16.0)).is_ok();
        let identity = run(e, &toks) == before;

        // Perturb BOTH low-rank factors: A starts at zero, so touching B
        // alone leaves the branch mathematically dead -- that is what
        // zero-initialisation means, and this test got it wrong once
        // already.
        let mut perturbed = false;
        if let Some(cls) = e.model.adapters.as_mut().and_then(|a| a.cls.as_mut()) {
            if cls.r >= 1 && cls.s.len() > 5 {
                // Live-cell perturbation: A row 0 over input 3 feeds B row 5
                // column 0, so the branch lands on logit 5 -- the same row
                // whose cached scale is doubled. An earlier version indexed
                // B flat as [7], which is (row 0, col 7): mathematically
                // dead while A row 7 stayed zero, and the test lied that
                // the wrapper had no effect.
                cls.a[3] = 0.1;
                cls.b[5 * cls.r] = 0.25;
                cls.s[5] = 2.0;
                perturbed = true;
            }
        }
        let moved_run = if perturbed { Some(run(e, &toks)) } else { None };
        let moves = match &moved_run {
            Some(after) => {
                let maxd = before
                    .iter()
                    .zip(after.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                crate::kprintln!("  [dbg] max logit delta under perturb: {}", maxd);
                maxd > 0.0
            }
            None => false,
        };

        e.model.detach_adapters();
        let restored = run(e, &toks) == before;
        crate::kprintln!(
            "  {}  attach",
            if attached { "ok " } else { "FAIL -- refused" }
        );
        crate::kprintln!(
            "  {}  untrained adapter is bit-identical",
            if identity { "ok " } else { "FAIL" }
        );
        crate::kprintln!(
            "  {}  trained weights move the logits",
            if moves { "ok " } else { "FAIL" }
        );
        crate::kprintln!(
            "  {}  detach restores exactly",
            if restored { "ok " } else { "FAIL" }
        );
        Some(attached && identity && moves && restored)
    });

    match result {
        // No engine loaded.
        None => false,
        // Engine present but hybrid: dense-adapter claims do not apply.
        Some(None) => {
            kprintln!("  skip -- hybrid model, dense-adapter checks do not apply");
            true
        }
        Some(Some(all)) => {
            if !all {
                kprintln!("  FAIL -- see the lines above for which claim broke");
            }
            all
        }
    }
}

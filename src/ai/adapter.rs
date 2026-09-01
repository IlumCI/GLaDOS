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

    /// The input width this site was built for.
    ///
    /// Read from `a`, which was sized `r * k_in` at construction, and never
    /// from the caller's slice. The difference is not academic: `State::xb` is
    /// allocated `dim.max(q_dim)` because it also holds the attention output,
    /// so on a model whose head_dim is not `dim / n_heads` -- Qwen3-0.6B, with
    /// dim 1024 and q_dim 2048 -- the buffer handed to `apply` is twice the
    /// width of the projection's actual input. `Mat::matvec` tolerates that
    /// and reads `x[..cols]`; inferring `k` from `x.len()` did not, and walked
    /// straight off the end of `a` on the first token.
    ///
    /// This never fired under test because every model the adapter paths had
    /// ever run on -- SmolLM2-135M and the synthetic fixtures -- has
    /// `q_dim == dim`, which makes the wrong answer identical to the right one.
    #[inline]
    pub fn k_in(&self) -> usize {
        if self.r == 0 { 0 } else { self.a.len() / self.r }
    }

    /// Apply to a freshly-computed base matvec result in `out`.
    ///
    /// `base` produced `out = W0.x`; this adds the low-rank branch and then
    /// the cached per-row scale. Called on every token for adapted sites, so
    /// it allocates nothing: `ax` is the caller's rank-sized scratch.
    pub fn apply(&self, out: &mut [f32], x: &[f32], ax: &mut [f32]) {
        let k = self.k_in().min(x.len());
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

    /// Gradient with respect to this site's **input**.
    ///
    /// Nothing produced this before, and its absence is exactly why the loop
    /// could adapt the routing head and nothing else. `backward` and
    /// `backward_rows` answer the parameters and stop; without dL/dx there is
    /// no way to carry a gradient below an adapted site, so a classifier
    /// adapter is trainable and a q/k/v adapter is not -- which is the whole
    /// difference between improving *which* answer is chosen and improving the
    /// network that proposes it.
    ///
    /// The forward is
    ///
    /// ```text
    /// y[o] = s[o] * (W0.x)[o] + scale * (B.A.x)[o]
    /// ```
    ///
    /// so the input gradient is two terms:
    ///
    /// ```text
    /// dL/dx = W0^T (gy * s)  +  scale * A^T (B^T gy)
    /// ```
    ///
    /// `s` multiplies the incoming gradient rather than contributing a term of
    /// its own, because during a token's forward it is a cached constant --
    /// `refresh` recomputes it after an optimiser step, not per token. The
    /// magnitude's own gradient is `backward`'s business and is unaffected.
    ///
    /// **Accumulates.** Three sites (q, k, v) read the same normed input, so
    /// their gradients have to sum; a version that overwrote would silently
    /// keep only the last one. The frozen term goes through a scratch because
    /// `Mat::wt_matvec` zeroes its output before writing.
    ///
    /// Allocates, deliberately. This is training cadence, not the decode path,
    /// and threading two scratch buffers through every caller to save an
    /// allocation per site per token of a training step is a trade this does
    /// not need to make yet.
    pub fn backward_x(&self, w: &Mat<'_>, gy: &[f32], gx: &mut [f32]) {
        let (rows, k) = match w {
            Mat::F32 { rows, cols, .. } => (*rows, *cols),
            Mat::Q8 { rows, cols, .. } => (*rows, *cols),
        };

        // The frozen branch, scaled per row.
        let mut gs = vec![0.0f32; rows];
        for (o, v) in gs.iter_mut().enumerate() {
            *v = gy[o] * self.s[o];
        }
        let mut frozen = vec![0.0f32; k];
        w.wt_matvec(&mut frozen, &gs);
        for (i, v) in frozen.iter().enumerate().take(k) {
            gx[i] += *v;
        }

        // The low-rank branch. B^T gy first, which is r values, then A^T of
        // that -- the other association would build a k-by-r intermediate for
        // no reason.
        let mut bg = vec![0.0f32; self.r];
        for o in 0..rows {
            let brow = &self.b[o * self.r..(o + 1) * self.r];
            let g = gy[o];
            if g == 0.0 {
                continue;
            }
            for (j, bv) in brow.iter().enumerate() {
                bg[j] += bv * g;
            }
        }
        let sc = self.scale();
        for j in 0..self.r {
            let g = bg[j] * sc;
            if g == 0.0 {
                continue;
            }
            let arow = &self.a[j * k..(j + 1) * k];
            for (i, av) in arow.iter().enumerate().take(k) {
                gx[i] += av * g;
            }
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
        debug_assert_eq!(rows * self.r, self.b.len());
        debug_assert_eq!(k * self.r, self.a.len());
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

    /// The classifier alone, with every layer's q/k/v left unadapted.
    ///
    /// The decision layer is what phase 2 trains, and it is one matvec: the
    /// final hidden state against the classifier. Adapting q/k/v as well
    /// would require carrying a gradient back through every layer, which the
    /// activation adjoints support but nothing yet composes -- and it would
    /// make the cached-feature trick impossible, because the features would
    /// stop being constants the moment the attention path started moving.
    ///
    /// `qkv` is still `n_layers` long. The forward path indexes it directly
    /// per layer, so a short vector would be an out-of-bounds panic rather
    /// than an unadapted model.
    pub fn classifier_only(cfg: &Config, r: usize, alpha: f32) -> Self {
        let r = r.min(MAX_RANK);
        Self {
            r,
            alpha,
            qkv: (0..cfg.n_layers).map(|_| [None, None, None]).collect(),
            cls: Some(Dora::new(r, alpha, cfg.dim, cfg.vocab_size)),
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
        // `>=`, not `==`: a caller may pass a scratch buffer wider than the
        // projection, and the weight width is the authority on how much of
        // it is input. See `Dora::k_in`.
        debug_assert!(x.len() >= k);
        let mut c_row = vec![0.0f32; k];
        for o in 0..rows {
            self.row_backward(w, x, ax, base[o], gy[o], o, &mut c_row, ga, gb, dm);
        }
    }

    /// One row's contribution, shared with `backward_rows`.
    ///
    /// Split out rather than duplicated so the full walk and the restricted
    /// one cannot drift: the restriction is only sound while every term here
    /// carries `gy_o` as a factor, and a future term that forgot to would
    /// otherwise be wrong in exactly one of the two copies.
    ///
    /// `c_row` is the caller's scratch, `k` wide. It is an argument because
    /// the two callers loop, and allocating a row buffer per row was the
    /// difference between a training step and a training pause.
    #[allow(clippy::too_many_arguments)]
    fn row_backward(
        &self,
        w: &Mat<'_>,
        x: &[f32],
        ax: &[f32],
        base_o: f32,
        gy_o: f32,
        o: usize,
        c_row: &mut [f32],
        ga: &mut [f32],
        gb: &mut [f32],
        dm: &mut [f32],
    ) {
        let k = c_row.len();
        w.row_into(o, c_row);
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
            gb[o * self.r + j] += self.scale() * gy_o * ax[j];
            let bcoef_branch = self.scale() * gy_o * self.b[o * self.r + j];
            if bcoef_branch != 0.0 {
                for kk in 0..k {
                    ga[j * k + kk] += bcoef_branch * x[kk];
                }
            }
        }

        // ...and the hidden route through s. ds/dtheta = -s.(2N)^{-1}.dN,
        // multiplied by base gives the correction term; dm rides the
        // same chain but through ds/dm = s/m.
        let coef = -gy_o * base_o * self.s[o] / (2.0 * n);
        dm[o] += gy_o * base_o * self.s[o] / self.m[o].max(1e-12);
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
}

// --- persistence ---------------------------------------------------------
//
// An adapter is the only part of this system that learns, so it is the only
// part worth writing down. The base checkpoint never moves; a trained
// classifier is a few kilobytes against 155 MB of frozen weights, which is
// what makes "snapshot the mind before changing it" an ordinary operation
// rather than a copy of the whole model.
//
// The layout follows RustLMHub's `FfnLora::save` in every decision that could
// have gone either way, because the two projects trade adapters and a gratuitous
// difference is a conversion script nobody writes:
//
//   * a magic first, and a refusal rather than a guess when it is absent;
//   * dimensions in the header, checked for *exact* equality on load --
//     an adapter trained for another shape is an error, never a reshape;
//   * flat little-endian f32 payload, no padding, no alignment assumptions;
//   * the frozen model is not in the file and is not touched by loading it.
//
// Where it necessarily differs, and why: LoAA stores A and B for three FFN
// projections. This is DoRA, so every site also carries per-row magnitudes
// `m`, and the sites are the attention path plus the classifier rather than
// gate/up/down. A byte-identical format was never available; an identical set
// of promises was, and that is what a reader on either side actually depends
// on.
//
// `s` is absent on purpose. It is m/|W0 + BA| -- derived from the frozen
// weight, which the file does not contain and must not need to. Storing it
// would let a file and a checkpoint disagree about a value that has exactly
// one correct answer, and the disagreement would be silent.
//
// Rows are stored sparsely. Only rows whose low-rank factors or magnitude
// have actually moved are written, because a row with a zero B and a default
// magnitude is bit-identical to no adapter at all -- and the decision layer
// moves a hundred rows out of a vocabulary of fifty thousand. Dense storage
// would make a 10 KB adapter a 6 MB one, which is the difference between
// snapshotting it routinely and not bothering.

// The header's `dim` and `vocab` are the classifier site's shape, recorded so
// a file can be identified without walking it. They are *not* what the loader
// checks against: every site carries its own k_in and out, and those are what
// must match the model exactly. A header field that looked authoritative and
// was not would be worse than one that is absent.
pub const ADAPTER_MAGIC: &[u8; 8] = b"GLADOSA1";
const ADAPTER_HEADER: usize = 32;
const SITE_HEADER: usize = 20;

#[derive(Debug, PartialEq)]
pub enum AdapterError {
    NotAnAdapter,
    /// A field ran off the end, in the site at this index.
    Truncated(usize),
    /// Bytes left over after the declared sites.
    Trailing(usize),
    /// A site names a projection this build does not know.
    UnknownSite(u32),
    /// The file was trained against a different geometry. Carries what the
    /// file says and what the loaded model says, in that order.
    Shape(usize, usize),
    /// A stored row index is outside the site it belongs to.
    Row(usize),
    /// The loaded checkpoint is a hybrid, whose gated and partially rotated
    /// projections have no verified backward pass yet -- so there is nothing
    /// that could have produced this file for it.
    Hybrid,
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

impl Dora {
    /// Rows that are not bit-identical to an unattached adapter.
    ///
    /// A row with a zero low-rank factor and the default magnitude
    /// contributes `out[o] * 1.0 + 0.0`, so writing it down would record the
    /// absence of a change. The test is on `m` as well as `b` because seeding
    /// moves `m` alone -- a seeded but untrained row is still an identity in
    /// the forward pass, but it is not the value `Dora::new` would produce,
    /// and dropping it would change what a reload reconstructs.
    fn moved_rows(&self) -> Vec<u32> {
        let out = self.m.len();
        let mut rows = Vec::new();
        for o in 0..out {
            let b_moved = self.b[o * self.r..(o + 1) * self.r].iter().any(|v| *v != 0.0);
            if b_moved || self.m[o] != 1.0 {
                rows.push(o as u32);
            }
        }
        rows
    }
}

impl Adapters {
    /// Serialise every attached site.
    pub fn to_blob(&self) -> Vec<u8> {
        let mut sites: Vec<(u32, u32, &Dora)> = Vec::new();
        for (l, three) in self.qkv.iter().enumerate() {
            for (kind, slot) in three.iter().enumerate() {
                if let Some(d) = slot {
                    sites.push((kind as u32, l as u32, d));
                }
            }
        }
        if let Some(d) = self.cls.as_ref() {
            sites.push((3, 0, d));
        }

        let mut out = Vec::with_capacity(ADAPTER_HEADER + sites.len() * SITE_HEADER);
        out.extend_from_slice(ADAPTER_MAGIC);
        put_u32(&mut out, self.r as u32);
        put_f32(&mut out, self.alpha);
        put_u32(&mut out, self.qkv.len() as u32);
        put_u32(&mut out, self.cls.as_ref().map(|d| d.a.len() / self.r.max(1)).unwrap_or(0) as u32);
        put_u32(&mut out, self.cls.as_ref().map(|d| d.m.len()).unwrap_or(0) as u32);
        put_u32(&mut out, sites.len() as u32);

        for (kind, layer, d) in sites {
            // One rank in the header covers every site, following LoAA. That
            // is only sound because `Adapters` builds every site at its own
            // rank -- if a site could carry a different one, the reader would
            // split `a` at the wrong place and every float after it would be
            // finite and wrong.
            debug_assert_eq!(d.r, self.r);
            let k_in = d.a.len() / d.r.max(1);
            let rows = d.moved_rows();
            put_u32(&mut out, kind);
            put_u32(&mut out, layer);
            put_u32(&mut out, k_in as u32);
            put_u32(&mut out, d.m.len() as u32);
            put_u32(&mut out, rows.len() as u32);
            for v in d.a.iter() {
                put_f32(&mut out, *v);
            }
            for &o in rows.iter() {
                put_u32(&mut out, o);
                for j in 0..d.r {
                    put_f32(&mut out, d.b[o as usize * d.r + j]);
                }
                put_f32(&mut out, d.m[o as usize]);
            }
        }
        out
    }
}

/// One site read back, before it is matched against a model.
pub struct SiteBlob {
    pub kind: u32,
    pub layer: usize,
    pub k_in: usize,
    pub out: usize,
    pub a: Vec<f32>,
    /// (row, b-row, m) for every row that moved.
    pub rows: Vec<(u32, Vec<f32>, f32)>,
}

pub struct AdapterBlob {
    pub r: usize,
    pub alpha: f32,
    pub n_layers: usize,
    pub dim: usize,
    pub vocab: usize,
    pub sites: Vec<SiteBlob>,
}

/// Parse without a model in hand, so the boot self-test can drive every
/// rejection path without a checkpoint loaded.
///
/// Walks and never seeks, and asserts it lands on the last byte -- the same
/// bargain the checkpoint readers and the corpus bundle make. Nothing in the
/// payload names itself, so one length read wrongly leaves every float after
/// it perfectly finite and wrong.
pub fn parse_adapter(blob: &[u8]) -> Result<AdapterBlob, AdapterError> {
    if blob.len() < ADAPTER_HEADER || &blob[..8] != ADAPTER_MAGIC {
        return Err(AdapterError::NotAnAdapter);
    }
    let u32_at = |o: usize| u32::from_le_bytes([blob[o], blob[o + 1], blob[o + 2], blob[o + 3]]);
    let r = u32_at(8) as usize;
    let alpha = f32::from_bits(u32_at(12));
    let n_layers = u32_at(16) as usize;
    let dim = u32_at(20) as usize;
    let vocab = u32_at(24) as usize;
    let n_sites = u32_at(28) as usize;
    if r == 0 || r > MAX_RANK {
        return Err(AdapterError::Shape(r, MAX_RANK));
    }

    let mut off = ADAPTER_HEADER;
    let mut sites = Vec::with_capacity(n_sites);
    for i in 0..n_sites {
        if off + SITE_HEADER > blob.len() {
            return Err(AdapterError::Truncated(i));
        }
        let kind = u32_at(off);
        let layer = u32_at(off + 4) as usize;
        let k_in = u32_at(off + 8) as usize;
        let out_n = u32_at(off + 12) as usize;
        let n_rows = u32_at(off + 16) as usize;
        off += SITE_HEADER;
        if kind > 3 {
            return Err(AdapterError::UnknownSite(kind));
        }

        // Every length is checked against the blob before a single float is
        // read, rather than as the loop goes: a site claiming a gigabyte of
        // `a` would otherwise allocate it and then discover the truncation.
        let a_bytes = r.saturating_mul(k_in).saturating_mul(4);
        let row_bytes = n_rows.saturating_mul(4 + r * 4 + 4);
        if off.saturating_add(a_bytes).saturating_add(row_bytes) > blob.len() {
            return Err(AdapterError::Truncated(i));
        }

        let mut a = Vec::with_capacity(r * k_in);
        for _ in 0..r * k_in {
            a.push(f32::from_bits(u32_at(off)));
            off += 4;
        }
        let mut rows = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            let row = u32_at(off);
            off += 4;
            if row as usize >= out_n {
                return Err(AdapterError::Row(i));
            }
            let mut brow = Vec::with_capacity(r);
            for _ in 0..r {
                brow.push(f32::from_bits(u32_at(off)));
                off += 4;
            }
            let m = f32::from_bits(u32_at(off));
            off += 4;
            rows.push((row, brow, m));
        }
        sites.push(SiteBlob { kind, layer, k_in, out: out_n, a, rows });
    }
    if off != blob.len() {
        return Err(AdapterError::Trailing(blob.len() - off));
    }
    Ok(AdapterBlob { r, alpha, n_layers, dim, vocab, sites })
}

// --- row-restricted forms -----------------------------------------------
//
// The classifier is the site the decision layer trains, and it has one row per
// vocabulary entry -- 151,936 of them on Qwen3. `refresh` and `backward` both
// walk every row and dequantise it, which is a pass over 155 MB. At training
// cadence that is affordable once; per optimiser step it is not.
//
// The restriction is exact rather than an approximation, and for two separate
// reasons worth stating apart:
//
//   *Backward.* Every term in the row loop carries `gy[o]` as a factor, and
//   restricted cross-entropy makes `gy` exactly zero outside the candidate
//   set. Rows the grammar cannot reach contribute exactly nothing, so
//   skipping them changes no gradient by any amount.
//
//   *Refresh.* `s[o] = m[o] / |W0[o] + B[o].A|`, and `A` is shared across
//   every row -- so at first glance a step on `A` moves every row's scale.
//   It does not, because `B[o]` is zero for any row that has never received
//   a gradient, and `W0[o] + 0.A` does not depend on `A` at all. Only rows
//   that have been candidates can move, which is the same set.
//
// A row that is never refreshed keeps `s = 1.0` and `B = 0`, which is exactly
// the identity the seeding pass arranges. Seeding exists to put `m` on the
// same footing as `s` for rows that are about to train; a row that never
// trains never needs it. That is what makes classifier training affordable at
// all, and it is why `attach_adapters_unseeded` is not a shortcut.

impl Dora {
    /// `apply` over a subset of rows. `out` is indexed parallel to `rows`
    /// rather than by row id, because the caller computing 78 base values
    /// out of 151,936 has no reason to carry the other 151,858.
    pub fn apply_rows(&self, out: &mut [f32], rows: &[u32], x: &[f32], ax: &mut [f32]) {
        let k = self.k_in().min(x.len());
        for j in 0..self.r {
            let arow = &self.a[j * k..(j + 1) * k];
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += arow[i] * x[i];
            }
            ax[j] = acc;
        }
        for (i, &o) in rows.iter().enumerate() {
            let o = o as usize;
            let brow = &self.b[o * self.r..(o + 1) * self.r];
            let mut acc = 0.0f32;
            for j in 0..self.r {
                acc += brow[j] * ax[j];
            }
            out[i] = out[i] * self.s[o] + acc * self.scale();
        }
    }

    /// Recompute cached scales for named rows only, seeding their magnitudes
    /// on the first pass exactly as the full `refresh` does.
    pub fn refresh_rows(&mut self, w: &Mat<'_>, rows: &[u32], m_was_default: bool) {
        let k = match w {
            Mat::F32 { cols, .. } => *cols,
            Mat::Q8 { cols, .. } => *cols,
        };
        let mut wrow = vec![0.0f32; k];
        for &o in rows {
            let o = o as usize;
            w.row_into(o, &mut wrow);
            for j in 0..self.r {
                let bj = self.b[o * self.r + j];
                if bj != 0.0 {
                    let arow = &self.a[j * k..(j + 1) * k];
                    for i in 0..k {
                        wrow[i] += bj * arow[i];
                    }
                }
            }
            let norm = crate::ai::tensor::sqrtf(wrow.iter().map(|v| v * v).sum::<f32>());
            if m_was_default && self.m[o] == 1.0 {
                self.m[o] = norm;
            }
            self.s[o] = if norm > 0.0 { self.m[o] / norm } else { 1.0 };
        }
    }

    /// `backward` over a subset of rows, with `base` and `gy` indexed
    /// parallel to `rows`.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_rows(
        &self,
        w: &Mat<'_>,
        x: &[f32],
        ax: &[f32],
        base: &[f32],
        gy: &[f32],
        rows: &[u32],
        ga: &mut [f32],
        gb: &mut [f32],
        dm: &mut [f32],
    ) {
        let k = match w {
            Mat::F32 { cols, .. } => *cols,
            Mat::Q8 { cols, .. } => *cols,
        };
        // `>=`, not `==`: a caller may pass a scratch buffer wider than the
        // projection, and the weight width is the authority on how much of
        // it is input. See `Dora::k_in`.
        debug_assert!(x.len() >= k);
        let mut c_row = vec![0.0f32; k];
        for (i, &o) in rows.iter().enumerate() {
            self.row_backward(w, x, ax, base[i], gy[i], o as usize, &mut c_row, ga, gb, dm);
        }
    }
}

/// Does a freshly constructed adapter have any gradient to follow?
///
/// It does not, and that is the point of this check. `Dora::new` zeroes both
/// low-rank factors, and `row_backward` forms them as
///
/// ```text
///   gb[o,j] += scale * gy * ax[j]                 ax = A.x, so zero when A is zero
///   ga[j,:] += scale * gy * b[o,j] * x            guarded by `if bcoef != 0`
/// ```
///
/// so A being zero kills every B gradient and B being zero kills every A
/// gradient. The pair is a fixed point: a zero-initialised low-rank branch has
/// identically zero gradient and cannot leave the origin however long it is
/// trained. Only the per-row magnitudes move, and an "adapter" that only
/// rescales frozen rows is a much weaker object than the one the rest of the
/// tree describes.
///
/// The arithmetic in `row_backward` is not wrong -- the finite-difference gate
/// checks it from a non-zero point and it passes. The initialisation is wrong,
/// which is why nothing caught it: every gradient this asserts to be zero
/// really is the correct gradient at that point.
///
/// Standard LoRA initialises A randomly and B at zero, so `BA` is zero at the
/// start (the adapter is the identity, as intended) while B still has a live
/// gradient through A. This asserts the current behaviour rather than the
/// desired one, so that changing the initialisation has to come here and say
/// so.
pub fn init_gradient_selftest() -> bool {
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
            crate::kprintln!("    FAIL {}", what);
        }
    };

    let (rows, k, r) = (4usize, 6usize, 2usize);
    let w: Vec<f32> = (0..rows * k).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
    let mat = Mat::F32 { data: &w, rows, cols: k };
    let d = Dora::new(r, 16.0, k, rows);

    claim("a fresh adapter has a all zero", d.a.iter().all(|v| *v == 0.0));
    claim("a fresh adapter has b all zero", d.b.iter().all(|v| *v == 0.0));

    let x: Vec<f32> = (0..k).map(|i| 0.2 + i as f32 * 0.05).collect();
    let base: Vec<f32> = (0..rows).map(|i| 0.5 - i as f32 * 0.1).collect();
    let gy: Vec<f32> = (0..rows).map(|i| 0.3 + i as f32 * 0.11).collect();

    // ax is what a forward pass would have produced: A.x, which is zero.
    let ax = alloc::vec![0.0f32; r];
    let mut ga = alloc::vec![0.0f32; d.a.len()];
    let mut gb = alloc::vec![0.0f32; d.b.len()];
    let mut dm = alloc::vec![0.0f32; d.m.len()];
    d.backward(&mat, &x, &ax, &base, &gy, &mut ga, &mut gb, &mut dm);

    claim("at zero init the A gradient is identically zero", ga.iter().all(|v| *v == 0.0));
    claim("at zero init the B gradient is identically zero", gb.iter().all(|v| *v == 0.0));
    // The magnitudes are the only thing with anywhere to go, which is exactly
    // why training appears to run and to change nothing that matters.
    claim("the magnitude gradient is not zero", dm.iter().any(|v| *v != 0.0));

    // And the arithmetic is fine once it is off the origin: give B a value and
    // the A gradient appears. This is what separates a wrong derivative from a
    // wrong starting point.
    let mut d2 = Dora::new(r, 16.0, k, rows);
    d2.b[0] = 0.5;
    let mut ga2 = alloc::vec![0.0f32; d2.a.len()];
    let mut gb2 = alloc::vec![0.0f32; d2.b.len()];
    let mut dm2 = alloc::vec![0.0f32; d2.m.len()];
    d2.backward(&mat, &x, &ax, &base, &gy, &mut ga2, &mut gb2, &mut dm2);
    claim("a non-zero B revives the A gradient", ga2.iter().any(|v| *v != 0.0));

    ok
}

/// Boot self-test for adapter persistence. Six claims, none needing a model.
///
/// The round trip is checked against a *hand-built* adapter rather than a
/// trained one, so the claim is about the format rather than about whatever
/// the last run happened to produce. The rejections matter as much: an
/// adapter file is float32 all the way down, so a length read wrongly leaves
/// finite, plausible, wrong weights, and nothing downstream would complain.
pub fn blob_selftest() -> bool {
    use crate::kprintln;

    let mut ok = true;
    let mut claim = |what: &str, pass: bool| {
        if !pass {
            ok = false;
        }
        kprintln!("  {}  {}", if pass { "ok " } else { "FAIL" }, what);
    };

    const R: usize = 3;
    const K: usize = 8;
    const OUT: usize = 40;
    const QOUT: usize = 12;

    let mut cls = Dora::new(R, 6.0, K, OUT);
    for (i, v) in cls.a.iter_mut().enumerate() {
        *v = 0.01 * (i as f32) - 0.1;
    }
    // Three rows moved out of forty, which is the shape the decision layer
    // actually produces: a hundred rows out of a vocabulary.
    for &o in [2usize, 17, 39].iter() {
        for j in 0..R {
            cls.b[o * R + j] = 0.5 - 0.1 * (j as f32) + 0.01 * (o as f32);
        }
        cls.m[o] = 1.5 + 0.25 * (o as f32);
    }
    let mut q = Dora::new(R, 6.0, K, QOUT);
    for (i, v) in q.a.iter_mut().enumerate() {
        *v = 0.02 * (i as f32);
    }
    q.m[5] = 2.0;

    let mut qkv: Vec<[Option<Dora>; 3]> = (0..2).map(|_| [None, None, None]).collect();
    qkv[1][0] = Some(q.clone());
    let ad = Adapters { r: R, alpha: 6.0, qkv, cls: Some(cls.clone()) };

    let blob = ad.to_blob();
    match parse_adapter(&blob) {
        Err(e) => {
            claim("a written adapter reads back", false);
            kprintln!("    {:?}", e);
        }
        Ok(p) => {
            let header = p.r == R && p.n_layers == 2 && p.dim == K && p.vocab == OUT;
            let two_sites = p.sites.len() == 2;
            let qs = p.sites.iter().find(|s| s.kind == 0);
            let cs = p.sites.iter().find(|s| s.kind == 3);
            let q_ok = qs.map_or(false, |s| {
                s.layer == 1 && s.out == QOUT && s.a == q.a && s.rows.len() == 1
                    && s.rows[0].0 == 5 && s.rows[0].2 == 2.0
            });
            let c_ok = cs.map_or(false, |s| {
                s.out == OUT
                    && s.a == cls.a
                    && s.rows.len() == 3
                    && s.rows.iter().all(|(o, brow, m)| {
                        let o = *o as usize;
                        *m == cls.m[o] && brow[..] == cls.b[o * R..(o + 1) * R]
                    })
            });
            claim(
                "a written adapter reads back with every site, factor and row",
                header && two_sites && q_ok && c_ok,
            );
        }
    }

    // Sparsity is a promise about size, so it is checked as one -- against
    // *this adapter's own* dense equivalent rather than against a number
    // pulled from the shapes.
    //
    // The first version of this claim compared the whole file against what a
    // dense B and m would cost, which ignores the header and the A factors
    // that sparsity cannot remove. At these toy widths those dominate, so the
    // claim failed on an encoding that was working correctly -- the test was
    // wrong and the code was right, the same way round as the finite-
    // difference harness that once accused a correct kernel.
    //
    // The ratio is small here because OUT is 40. On the measured decision
    // layer it is 132 rows of 49,152, where the same encoding is 23,764 bytes
    // against 1.79 MB: 75x. That figure cannot be reached at a scale a boot
    // self-test can afford, which is why the claim is a direction rather than
    // a magnitude.
    let per_row = 4 + R * 4 + 4;
    let dense = blob.len() + (QOUT - 1) * per_row + (OUT - 3) * per_row;
    claim(
        "an adapter storing four rows of fifty-two is a third of its dense form",
        blob.len() * 3 < dense,
    );

    claim(
        "a blob that is not an adapter is refused",
        parse_adapter(b"GLADOSC1 wrong file entirely").err() == Some(AdapterError::NotAnAdapter),
    );
    claim(
        "one byte short is caught rather than read as fewer weights",
        matches!(
            parse_adapter(&blob[..blob.len() - 1]).err(),
            Some(AdapterError::Truncated(_))
        ),
    );
    let mut long = blob.clone();
    long.push(0);
    claim(
        "a byte past the last site is caught",
        parse_adapter(&long).err() == Some(AdapterError::Trailing(1)),
    );

    // An unattached adapter must write nothing but a header: every row is
    // still the identity, and recording that would be recording nothing.
    let empty = Adapters {
        r: R,
        alpha: 6.0,
        qkv: (0..2).map(|_| [None, None, None]).collect(),
        cls: Some(Dora::new(R, 6.0, K, OUT)),
    };
    let eb = empty.to_blob();
    claim(
        "an untrained adapter stores no rows at all",
        parse_adapter(&eb).map(|p| p.sites.len() == 1 && p.sites[0].rows.is_empty())
            == Ok(true),
    );

    ok
}

/// Boot self-test, run against whatever engine is loaded. Three claims,
///
///
/// 1. a freshly attached adapter is bit-identical to no adapter (B zero,
///    m seeded to the frozen row norms so every cached scale is exactly 1);
/// 2. perturbing one trained weight moves the logits (the wrapper is not a
///    silent no-op in the other direction either);
/// 3. detaching restores the original logits exactly.
/// The layer walk, against finite differences.
///
/// Takes a `Model` rather than the `Engine`, so it can be pointed at a
/// synthetic geometry as well as at whatever checkpoint is staged. That is not
/// tidying: this test only ever ran on the loaded model, every model it ever
/// loaded had `q_dim == dim`, and the one that did not panicked at boot the
/// first time it was tried. A test that can only run on what happens to be
/// installed can only find bugs in what happens to be installed.
///
/// The one check that matters for any of this. Every kernel in `backward.rs`
/// was already selftested in isolation; what was never tested is the
/// *composition* -- rmsnorm into attention into swiglu into rmsnorm again,
/// with GQA fan-out, QK-norm and RoPE in between, and a residual rejoining at
/// two points. Each of those is a place to be off by a transpose, a head
/// offset or a sign, and none of them faults when wrong.
///
/// **Two positions, not one.** With a single position the causal softmax is
/// over one element, is therefore identically 1.0, and q and k receive exactly
/// zero gradient -- a one-token test would pass while proving nothing about
/// the two sites that matter most. Two positions is the shortest sequence in
/// which position 1 attends to position 0, which is what puts a real gradient
/// on q, on k, and across positions.
///
/// **`s` is held fixed.** DoRA's per-row scale depends on B, but the trainer's
/// convention is that `refresh` recomputes it *after* an optimiser step, and
/// `Dora::backward` accounts for the magnitude through `dm` rather than
/// through a and b. So the difference is taken with no refresh in between,
/// which is what the analytic gradient actually claims. Refreshing here would
/// measure a different derivative and the mismatch would look like a bug in
/// the walk.
pub fn walk_selftest(mdl: &mut super::model::Model) -> bool {
    // Shrink the context for the duration.
    //
    // The walk scores two positions. `State` allocates by `cfg.live_cap()`,
    // which is the *trained* length when no window is set, and the exact
    // cache is f32 -- so at Qwen3's 8192 that is about 1.9 GiB per `State`,
    // and this builds several. It ran fine for as long as every checkpoint
    // was converted at 512 and died with an allocation failure the first time
    // one was not, which is a limit of the test and not of the machine.
    //
    // Safe because a GLADOSM2 weight offset is walked from dim, hidden,
    // layers and vocab: `seq_len` sizes the cache and the RoPE table and
    // nothing else, so lowering it changes what is allocated and no address.
    let saved_seq = mdl.cfg.seq_len;
    mdl.cfg.seq_len = 4;
    let ok = walk_inner(mdl);
    mdl.cfg.seq_len = saved_seq;
    ok
}

fn walk_inner(mdl: &mut super::model::Model) -> bool {
    use super::model::{Grads, State, Tape};

    let cfg = mdl.cfg.clone();
    if cfg.hybrid() || cfg.streams() {
        return true;
    }

    let toks: alloc::vec::Vec<usize> = alloc::vec![11usize, 7];
    let at = toks.len() - 1;

    // Non-zero A and B, or every site is an exact identity and the low-rank
    // branch contributes nothing to differentiate.
    let mut ad = Adapters::full(&cfg, 2, 4.0);
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / 16777216.0) * 0.06 - 0.03
    };
    for t in ad.qkv.iter_mut() {
        for site in t.iter_mut().flatten() {
            for v in site.a.iter_mut() {
                *v = next();
            }
            for v in site.b.iter_mut() {
                *v = next();
            }
        }
    }
    if let Some(c) = ad.cls.as_mut() {
        for v in c.a.iter_mut() {
            *v = next();
        }
        for v in c.b.iter_mut() {
            *v = next();
        }
    }
    // Seed the scales once against the frozen weights, then never again.
    let mut probe = ad.clone();
    for (l, t) in probe.qkv.iter_mut().enumerate() {
        for (i, site) in t.iter_mut().enumerate() {
            if let Some(dora) = site.as_mut() {
                let w = mdl.frozen_site(l, i);
                dora.refresh(&w, true);
            }
        }
    }
    if let Some(c) = probe.cls.as_mut() {
        let w = mdl.frozen_cls();
        c.refresh(&w, true);
    }
    let ad = probe;

    if mdl.attach_adapters_unseeded(ad.clone()).is_err() {
        return false;
    }

    // A real loss, not a synthetic one, and the reason is resolution.
    //
    // The first version scored `sum(gl * logits)` for a sparse fixed `gl`.
    // That produces gradients of order 1e-6 to 1e-11, and a central difference
    // cannot see them: with the loss around 1 the f32 ULP is about 1e-7, so a
    // dL of 1e-13 rounds to either nothing or to exactly one representable
    // step. It measured the latter and reported `numeric -0.015258788`, which
    // is -1000/65536 -- a float quantum wearing the costume of a derivative.
    //
    // Cross-entropy on a target token gives `softmax - onehot`, which is O(1)
    // across the whole vocabulary rather than sparse and small, and the
    // gradients that follow are large enough to be measured.
    let target = 3usize.min(cfg.vocab_size - 1);
    let ce = |logits: &[f32]| -> f32 {
        let m = logits.iter().fold(f32::MIN, |a, v| a.max(*v));
        let mut z = 0.0f32;
        for v in logits {
            z += super::tensor::expf(v - m);
        }
        // log via the identity, since `tensor` exposes exp and not log and
        // one more Newton loop here would be a numeric nobody else uses.
        let mut lo = -60.0f32;
        let mut hi = 60.0f32;
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if super::tensor::expf(mid) < z { lo = mid } else { hi = mid }
        }
        (m + 0.5 * (lo + hi)) - logits[target]
    };

    let loss = |m: &super::model::Model| -> f32 {
        // Exact cache, so the loss is smooth in every parameter rather than a
        // step function of anything upstream of a cached key or value.
        let mut st = State::new_exact(&cfg);
        for (i, &t) in toks.iter().enumerate() {
            m.forward(&mut st, t, i);
        }
        ce(&st.logits)
    };

    let mut tape = Tape::new(&cfg, toks.len());
    let mut st = State::new_exact(&cfg);
    for (i, &t) in toks.iter().enumerate() {
        if !mdl.forward_taped(&mut st, t, i, &mut tape) {
            return false;
        }
    }
    // dL/dlogits for cross-entropy.
    let m = st.logits.iter().fold(f32::MIN, |a, v| a.max(*v));
    let mut z = 0.0f32;
    for v in st.logits.iter() {
        z += super::tensor::expf(v - m);
    }
    let gl: alloc::vec::Vec<f32> = st
        .logits
        .iter()
        .enumerate()
        .map(|(o, v)| super::tensor::expf(v - m) / z - if o == target { 1.0 } else { 0.0 })
        .collect();

    let mut g = Grads::new(&ad);
    if !mdl.backward(&tape, &gl, at, &mut g) {
        crate::kprintln!("  walk: backward refused");
        return false;
    }

    // A **directional** derivative, not a per-entry one.
    //
    // Differencing one parameter asks the loss to resolve `grad * 2h`, which
    // for a deep site is far below its own rounding. Stepping every parameter
    // at once along the gradient asks it to resolve `2 * eps * |g|^2`, a sum
    // over thousands of entries, which is measurable -- and it checks the
    // whole gradient vector rather than a sample of it. A single wrong sign
    // anywhere in the walk moves this number.
    let mut norm2 = 0.0f64;
    for t in g.qkv.iter() {
        for site in t.iter() {
            for v in site.gb.iter() {
                norm2 += (*v as f64) * (*v as f64);
            }
        }
    }
    if norm2 <= 0.0 {
        crate::kprintln!("  walk: the gradient is entirely zero");
        return false;
    }
    let peak = g
        .qkv
        .iter()
        .flat_map(|t| t.iter())
        .flat_map(|s| s.gb.iter())
        .fold(0.0f32, |a, v| a.max(v.abs()));
    // Step the largest entry by about 1e-2, which is small enough to stay in
    // the locally linear region and large enough to clear the noise floor.
    let eps = (1e-2f32 / peak.max(1e-12)).min(1e6);

    let mut walk_probe = |sign: f32| -> f32 {
        let mut probe = ad.clone();
        for (l, t) in probe.qkv.iter_mut().enumerate() {
            for (i, site) in t.iter_mut().enumerate() {
                if let Some(d) = site.as_mut() {
                    for (k, v) in d.b.iter_mut().enumerate() {
                        *v += sign * eps * g.qkv[l][i].gb[k];
                    }
                }
            }
        }
        let _ = mdl.attach_adapters_unseeded(probe);
        loss(mdl)
    };
    let up = walk_probe(1.0);
    let dn = walk_probe(-1.0);
    let _ = mdl.attach_adapters_unseeded(ad.clone());

    let numeric = ((up - dn) as f64) / (2.0 * eps as f64);
    // Printed whether or not it passes. A check whose numbers nobody sees is
    // a check nobody can tell is measuring anything -- and this one silently
    // measured float rounding for two runs before the numbers were looked at.
    crate::kprintln!(
        "  [dbg] directional |g|^2 {} vs numeric {} over {} entries",
        norm2 as f32,
        numeric as f32,
        g.qkv.iter().flat_map(|t| t.iter()).map(|s| s.gb.len()).sum::<usize>()
    );
    let rel = (numeric - norm2).abs() / norm2.abs().max(1e-30);
    if rel > 0.05 {
        crate::kprintln!(
            "  walk: directional -- analytic |g|^2 {} vs numeric {}",
            norm2 as f32,
            numeric as f32
        );
        return false;
    }

    mdl.detach_adapters();
    true
}

/// The tape, against the forward it was taken from.
///
/// Two claims, and the first is the one that would be invisible if it broke.
///
/// **Taping must not change the forward.** The recording sits inside the one
/// `forward_dense` rather than in a copy of it, precisely so a training run
/// cannot end up optimising a slightly different model than the one being
/// served -- but "must not change it" is a claim, and an identical-logits
/// check is what makes it one that can fail. Bitwise, not approximate: a
/// `copy_from_slice` cannot perturb arithmetic, so anything other than
/// bit-identical means the recording is reading the wrong buffer or at the
/// wrong moment.
///
/// **The tape must hold what the backward will read.** Layer 0's entering
/// stream is the token embedding, which is checkable without any of the
/// backward machinery existing yet.
fn tape_selftest(e: &mut super::Engine) -> bool {
    // Same reason as `walk_selftest`: two `State`s over a handful of
    // positions, sized by the trained context unless told otherwise.
    let saved_seq = e.model.cfg.seq_len;
    e.model.cfg.seq_len = 8;
    let ok = tape_inner(e);
    e.model.cfg.seq_len = saved_seq;
    ok
}

fn tape_inner(e: &mut super::Engine) -> bool {
    use super::model::{State, Tape};

    if e.model.cfg.hybrid() {
        // Refused by `forward_taped` for the same reason `attach_adapters`
        // refuses them, and a selftest that quietly passed on a hybrid would
        // be reporting on a path that never ran.
        return true;
    }

    let toks: alloc::vec::Vec<usize> = alloc::vec![1usize, 5, 9, 2];
    let cfg = e.model.cfg.clone();

    let mut plain = State::new(&cfg);
    for (i, &t) in toks.iter().enumerate() {
        e.model.forward(&mut plain, t, i);
    }

    let mut taped = State::new(&cfg);
    let mut tape = Tape::new(&cfg, toks.len());
    for (i, &t) in toks.iter().enumerate() {
        if !e.model.forward_taped(&mut taped, t, i, &mut tape) {
            return false;
        }
    }

    if taped.logits != plain.logits {
        crate::kprintln!("  tape: logits differ with recording on");
        return false;
    }
    if tape.filled() != toks.len() {
        return false;
    }

    // Layer 0 sees the embedding, unmodified. This is the one entry whose
    // value is known independently of the forward, so it is the one that can
    // catch an off-by-one in the layer stride.
    let mut want = alloc::vec![0.0f32; cfg.dim];
    for (i, &t) in toks.iter().enumerate() {
        e.model.embed_row(t, &mut want);
        let Some(got) = tape.entering(0, i) else { return false };
        if got != &want[..] {
            crate::kprintln!("  tape: layer 0 at {} is not the embedding", i);
            return false;
        }
    }

    // Later layers hold something else. Without this, a tape that recorded
    // the same buffer for every layer would pass everything above.
    if cfg.n_layers > 1 {
        let a = tape.entering(0, 0).map(|v| v.to_vec());
        let b = tape.entering(cfg.n_layers - 1, 0).map(|v| v.to_vec());
        match (a, b) {
            (Some(a), Some(b)) if a != b => {}
            _ => {
                crate::kprintln!("  tape: every layer recorded the same stream");
                return false;
            }
        }
    }

    // Reading past what was written answers nothing rather than stale zeros
    // that would look like a real activation.
    // `n_layers` is the exit row and is legitimate; one past it is not. This
    // assertion said `n_layers` until the exit row was added, and failed the
    // moment it existed -- which is the test doing its job on the person who
    // wrote it.
    tape.entering(0, toks.len()).is_none()
        && tape.entering(cfg.n_layers, 0).is_some()
        && tape.entering(cfg.n_layers + 1, 0).is_none()
        && tape.final_normed(toks.len()).is_none()
}

/// The input gradient, against finite differences.
///
/// A gradient nobody differenced is a gradient nobody knows. An analytic
/// backward that is subtly wrong does not fault and does not diverge -- it
/// trains, slowly, toward the wrong thing, and every judge downstream reports
/// honestly that the variant did not help. That failure is indistinguishable
/// from "this idea does not work", which is the worst way to lose a year.
///
/// Central differences, because a forward difference is O(h) and the error it
/// leaves is the same order as the discrepancy being looked for.
fn backward_x_selftest() -> bool {
    use crate::ai::weights::Mat;

    // A frozen weight with structure rather than noise, so a transposition
    // error cannot pass by symmetry.
    let (rows, k, r) = (7usize, 5usize, 3usize);
    let w: alloc::vec::Vec<f32> = (0..rows * k)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.13 + (i / 7) as f32 * 0.021)
        .collect();
    let mat = Mat::F32 { data: &w, rows, cols: k };

    let mut d = Dora::new(r, 2.0 * r as f32, k, rows);
    // Non-zero B, or the low-rank branch is zero and the test only exercises
    // the frozen half -- which is the half most likely to be right.
    for (i, v) in d.a.iter_mut().enumerate() {
        *v = ((i % 7) as f32 - 3.0) * 0.09;
    }
    for (i, v) in d.b.iter_mut().enumerate() {
        *v = ((i % 5) as f32 - 2.0) * 0.11;
    }
    d.refresh(&mat, true);

    let x: alloc::vec::Vec<f32> = (0..k).map(|i| ((i % 3) as f32 - 1.0) * 0.7 + 0.2).collect();
    // A fixed non-uniform output gradient. All-ones would hide any error that
    // is antisymmetric across rows.
    let gy: alloc::vec::Vec<f32> =
        (0..rows).map(|o| ((o % 4) as f32 - 1.5) * 0.31).collect();

    // L(x) = sum_o gy[o] * y[o], so dL/dy is exactly gy.
    let loss = |xv: &[f32]| -> f32 {
        let mut out = alloc::vec![0.0f32; rows];
        mat.matvec(&mut out, xv);
        let mut ax = alloc::vec![0.0f32; r];
        d.apply(&mut out, xv, &mut ax);
        out.iter().zip(gy.iter()).map(|(y, g)| y * g).sum()
    };

    let mut gx = alloc::vec![0.0f32; k];
    d.backward_x(&mat, &gy, &mut gx);

    let h = 1e-3f32;
    for i in 0..k {
        let mut up = x.clone();
        let mut dn = x.clone();
        up[i] += h;
        dn[i] -= h;
        let numeric = (loss(&up) - loss(&dn)) / (2.0 * h);
        let diff = (numeric - gx[i]).abs();
        // Relative where the gradient is large, absolute where it is small.
        if diff > 1e-2 * gx[i].abs().max(1.0) {
            crate::kprintln!(
                "  backward_x[{}]: analytic {} vs numeric {}",
                i,
                gx[i],
                numeric
            );
            return false;
        }
    }

    // It accumulates rather than overwriting, which is what three sites
    // reading one normed input depend on.
    let mut twice = alloc::vec![0.0f32; k];
    d.backward_x(&mat, &gy, &mut twice);
    d.backward_x(&mat, &gy, &mut twice);
    for i in 0..k {
        if (twice[i] - 2.0 * gx[i]).abs() > 1e-4 * gx[i].abs().max(1.0) {
            return false;
        }
    }

    // A zero output gradient moves nothing. Cheap, and it catches a scratch
    // buffer that was not cleared between calls.
    let mut zero = alloc::vec![0.0f32; k];
    d.backward_x(&mat, &alloc::vec![0.0f32; rows], &mut zero);
    zero.iter().all(|v| v.abs() < 1e-6)
}

pub fn selftest() -> bool {
    // Reported rather than merely enforced: the boot log is this system's test
    // suite, and a claim that only speaks when it fails is a claim nobody
    // knows is being made.
    let gx_ok = backward_x_selftest();
    crate::kprintln!(
        "  {}  the input gradient matches finite differences",
        if gx_ok { "ok " } else { "FAIL" }
    );
    if !gx_ok {
        return false;
    }
    use super::model::State;
    use crate::kprintln;

    let result = super::with_engine(|e| {
        let cfg = e.model.cfg.clone();
        if cfg.hybrid() {
            return None;
        }
        // Needs the engine, so it lives here rather than beside the pure
        // checks above.
        let walk_ok = walk_selftest(&mut e.model);
        kprintln!(
            "  {}  the layer walk matches finite differences",
            if walk_ok { "ok " } else { "FAIL" }
        );
        if !walk_ok {
            return None;
        }
        let tape_ok = tape_selftest(e);
        kprintln!(
            "  {}  the tape matches the forward it was taken from",
            if tape_ok { "ok " } else { "FAIL" }
        );
        if !tape_ok {
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

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
        debug_assert_eq!(k, x.len());
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
        let k = x.len();
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
        debug_assert_eq!(k, x.len());
        let mut c_row = vec![0.0f32; k];
        for (i, &o) in rows.iter().enumerate() {
            self.row_backward(w, x, ax, base[i], gy[i], o as usize, &mut c_row, ga, gb, dm);
        }
    }
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

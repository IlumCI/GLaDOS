//! A Llama-architecture transformer, forward pass only.
//!
//! Weight layout is deliberately identical to llama2.c, in the same order, so
//! a converted checkpoint loads by pointing at a buffer rather than by
//! rearranging anything. Sizes come from a header; nothing here is baked in.
//!
//! The vocabulary is byte-level (256 entries). That removes the tokenizer
//! entirely -- no BPE merges table, no vocab file to get onto a machine that
//! has no filesystem yet -- at the cost of shorter effective context. For
//! bootstrapping something that has to run before storage exists, that trade
//! is worth making.

use super::tensor;
use super::weights::{self, Mat};
use alloc::vec;
use alloc::vec::Vec;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    /// Whether the output classifier reuses the token embedding matrix.
    pub shared_classifier: bool,
    /// RoPE base frequency. llama2.c hardcodes 10000; SmolLM2 trained with
    /// 100000, and using the wrong one does not fail -- it silently rotates
    /// every position by the wrong angle and produces fluent nonsense.
    pub rope_theta: f32,
    /// How many of the earliest tokens are kept forever.
    pub attn_sinks: usize,
    /// How many recent tokens are kept alongside them. Together with
    /// `attn_sinks` this is the live cache size; when their sum reaches
    /// `seq_len` nothing is ever evicted and attention is exactly as before.
    pub attn_window: usize,
}

impl Config {
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }
    /// Width of the key/value projections. Smaller than `dim` when the model
    /// uses grouped-query attention.
    pub fn kv_dim(&self) -> usize {
        self.dim * self.n_kv_heads / self.n_heads
    }
    /// How many query heads share each key/value head.
    pub fn kv_mul(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    pub fn param_count(&self) -> usize {
        let (d, h, l, kv, v) = (
            self.dim,
            self.hidden_dim,
            self.n_layers,
            self.kv_dim(),
            self.vocab_size,
        );
        let mut n = v * d          // token embedding
            + l * d                // rms_att
            + l * d * d            // wq
            + l * d * kv           // wk
            + l * d * kv           // wv
            + l * d * d            // wo
            + l * d                // rms_ffn
            + l * h * d            // w1
            + l * d * h            // w2
            + l * h * d            // w3
            + d; // rms_final
        if !self.shared_classifier {
            n += v * d;
        }
        n
    }
}

/// Byte offsets (in floats) of each weight tensor inside one contiguous block.
#[derive(Clone, Copy, Default)]
struct Offsets {
    token_embedding: usize,
    rms_att: usize,
    wq: usize,
    wk: usize,
    wv: usize,
    wo: usize,
    rms_ffn: usize,
    w1: usize,
    w2: usize,
    w3: usize,
    rms_final: usize,
    wcls: usize,
}

/// `legacy_rope_tables` describes the *file* layout, not ours.
///
/// llama2.c's exporter writes two precomputed RoPE tables -- `seq_len *
/// head_size / 2` floats each -- between `rms_final` and the classifier.
/// run.c skips over them because the angles are recomputed per position, but
/// they are still in the bytes, and `wcls` sits after them. A model built in
/// memory has no such gap, so the flag has to distinguish the two.
///
/// It only changes anything for an untied classifier. Every model here so far
/// ties its output weights to the embedding, which is why getting this wrong
/// would have gone unnoticed until the first model that does not.
fn offsets(cfg: &Config, legacy_rope_tables: bool) -> Offsets {
    let (d, h, l, kv, v) = (
        cfg.dim,
        cfg.hidden_dim,
        cfg.n_layers,
        cfg.kv_dim(),
        cfg.vocab_size,
    );
    let mut o = Offsets::default();
    let mut p = 0usize;
    o.token_embedding = p;
    p += v * d;
    o.rms_att = p;
    p += l * d;
    o.wq = p;
    p += l * d * d;
    o.wk = p;
    p += l * d * kv;
    o.wv = p;
    p += l * d * kv;
    o.wo = p;
    p += l * d * d;
    o.rms_ffn = p;
    p += l * d;
    o.w1 = p;
    p += l * h * d;
    o.w2 = p;
    p += l * d * h;
    o.w3 = p;
    p += l * h * d;
    o.rms_final = p;
    p += d;
    if legacy_rope_tables {
        p += cfg.seq_len * cfg.head_size();
    }
    o.wcls = if cfg.shared_classifier { o.token_embedding } else { p };
    o
}

/// Scratch buffers for one forward pass, plus the KV cache.
///
/// Allocated once and reused. Allocating per token would put the heap
/// allocator in the inner loop of generation for no reason.
pub struct State {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    att: Vec<f32>,
    pub logits: Vec<f32>,
    /// Keys are stored *unrotated*.
    ///
    /// RoPE used to be applied on the way in, which is fine while positions
    /// never move. They move the moment eviction exists: after dropping the
    /// oldest entry, every survivor is one place closer to the query, and a
    /// key rotated for its original position encodes a distance that is no
    /// longer true. Rotating at attention time instead makes position a
    /// property of where an entry sits in the cache, which is what
    /// StreamingLLM requires and what makes a window possible at all.
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
    /// Rotated copies of the live keys, rebuilt per layer per token.
    krot: Vec<f32>,
    /// cos/sin for every (position, dimension pair), precomputed.
    ///
    /// Rotating on the fly means `seq_len * head_size/2` angles per layer per
    /// token; computing sin and cos each time with the software
    /// implementations would dominate the forward pass entirely. The table is
    /// `seq_len * head_size/2` entries -- 128 KiB at 512 by 64 -- and is
    /// indexed rather than recomputed.
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

impl State {
    pub fn new(cfg: &Config) -> Self {
        let kv = cfg.kv_dim();
        let half = cfg.head_size() / 2;

        let mut rope_cos = vec![0.0f32; cfg.seq_len * half];
        let mut rope_sin = vec![0.0f32; cfg.seq_len * half];
        for p in 0..cfg.seq_len {
            for i in 0..half {
                // Matches the old inline computation exactly: the exponent is
                // the pair index doubled over head_size, so that a rotation is
                // shared across heads.
                let freq =
                    1.0 / tensor::powf(cfg.rope_theta, (i * 2) as f32 / cfg.head_size() as f32);
                let a = p as f32 * freq;
                rope_cos[p * half + i] = tensor::cosf(a);
                rope_sin[p * half + i] = tensor::sinf(a);
            }
        }

        Self {
            x: vec![0.0; cfg.dim],
            xb: vec![0.0; cfg.dim],
            xb2: vec![0.0; cfg.dim],
            hb: vec![0.0; cfg.hidden_dim],
            hb2: vec![0.0; cfg.hidden_dim],
            q: vec![0.0; cfg.dim],
            att: vec![0.0; cfg.n_heads * cfg.seq_len],
            logits: vec![0.0; cfg.vocab_size],
            key_cache: vec![0.0; cfg.n_layers * cfg.seq_len * kv],
            value_cache: vec![0.0; cfg.n_layers * cfg.seq_len * kv],
            krot: vec![0.0; cfg.seq_len * kv],
            rope_cos,
            rope_sin,
        }
    }

    /// Serialise the *used* prefix of the KV cache.
    ///
    /// This is the model's working memory made into an ordinary byte string,
    /// which is the whole trick: once it is bytes, the content-addressed store
    /// already knows what to do with it. Hashing, O(1) copies, snapshots and
    /// rollback all apply to the model's attention state for free, because
    /// none of that machinery cares what the bytes mean.
    ///
    /// Only positions `0..pos` are written. The cache is allocated for the full
    /// `seq_len`, but attention never reads past the current position, so the
    /// tail is uninitialised noise -- including it would make two identical
    /// mental states hash differently depending on what had been in the buffer
    /// before, which would silently destroy every property above.
    pub fn export_kv(&self, cfg: &Config, pos: usize) -> Vec<u8> {
        let kv = cfg.kv_dim();
        let pos = pos.min(cfg.seq_len);
        let mut out = Vec::new();
        out.extend_from_slice(KV_MAGIC);
        out.extend_from_slice(&(cfg.n_layers as u32).to_le_bytes());
        out.extend_from_slice(&(kv as u32).to_le_bytes());
        out.extend_from_slice(&(pos as u32).to_le_bytes());

        for src in [&self.key_cache, &self.value_cache] {
            for l in 0..cfg.n_layers {
                let loff = l * cfg.seq_len * kv;
                for v in &src[loff..loff + pos * kv] {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        out
    }

    /// Restore a cache written by `export_kv`. Returns the position it held.
    pub fn import_kv(&mut self, cfg: &Config, data: &[u8]) -> Option<usize> {
        if data.len() < 20 || &data[0..8] != KV_MAGIC {
            return None;
        }
        let g = |o: usize| u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]) as usize;
        let (layers, kv, pos) = (g(8), g(12), g(16));
        // A cache from a different model would restore as plausible-looking
        // garbage rather than failing, so the shape is checked rather than
        // trusted.
        if layers != cfg.n_layers || kv != cfg.kv_dim() || pos > cfg.seq_len {
            return None;
        }
        let need = 20 + 2 * layers * pos * kv * 4;
        if data.len() < need {
            return None;
        }

        let mut o = 20;
        for which in 0..2 {
            for l in 0..layers {
                let loff = l * cfg.seq_len * kv;
                let dst = if which == 0 { &mut self.key_cache } else { &mut self.value_cache };
                for i in 0..pos * kv {
                    dst[loff + i] =
                        f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                    o += 4;
                }
            }
        }
        Some(pos)
    }

    /// The final normed hidden state -- exactly what the classifier sees.
    ///
    /// Only meaningful immediately after `forward`. This is the feature vector
    /// the vocabulary extension scores against, and the one its gradient step
    /// multiplies by.
    pub fn hidden(&self) -> &[f32] {
        &self.xb
    }

    pub fn bytes(&self, cfg: &Config) -> usize {
        let kv = cfg.kv_dim();
        4 * (cfg.dim * 4
            + cfg.hidden_dim * 2
            + cfg.n_heads * cfg.seq_len
            + cfg.vocab_size
            + 2 * cfg.n_layers * cfg.seq_len * kv)
    }
}

/// Size of the llama2.c legacy header: seven i32.
pub const HEADER_BYTES: usize = 28;

const KV_MAGIC: &[u8; 8] = b"GLADOSKV";

const GLADOS_MAGIC: &[u8; 8] = b"GLADOSM2";
const GLADOS_VERSION: u32 = 2;
const GLADOS_HEADER: usize = 64;
const GLADOS_QUANT_I8: u32 = 1;

#[derive(Debug, Clone, Copy)]
pub enum LoadError {
    TooShort,
    BadHeader,
    /// The header describes more weights than the file contains, which usually
    /// means a truncated download rather than a format mismatch.
    Truncated { want: usize, have: usize },
    OutOfMemory,
}

/// Byte offsets of each tensor group in a GLADOSM2 blob.
#[derive(Clone, Copy, Default)]
struct ByteOffsets {
    embed: usize,
    wq: usize,
    wk: usize,
    wv: usize,
    wo: usize,
    w1: usize,
    w2: usize,
    w3: usize,
    wcls: usize,
}

/// Where the weights live.
///
/// Two shapes rather than one because they want opposite things. A llama2.c
/// checkpoint is all f32 and small, so owning it as `Vec<f32>` gives the
/// fastest possible access. A GLADOSM2 checkpoint is 135 MB of mixed
/// precision that the firmware already placed in RAM, so copying it would cost
/// both the copy and a second 135 MB of heap for no benefit.
enum Source {
    Flat(Vec<f32>),
    Blob {
        bytes: &'static [u8],
        off: ByteOffsets,
        /// Norm weights, pulled out of the blob at load.
        ///
        /// They are read per element rather than per row, so unaligned access
        /// would be on the hot path -- and at 35K values out of 134M, copying
        /// them costs 140 KB. Layout: rms_att[L*dim], rms_ffn[L*dim],
        /// rms_final[dim].
        norms: Vec<f32>,
    },
}

pub struct Model {
    pub cfg: Config,
    src: Source,
    o: Offsets,
}

impl Model {
    /// Build a deterministic pseudo-random model.
    ///
    /// Not a trained model and not pretending to be: it exists so the forward
    /// pass can be exercised and timed before any real weights can reach a
    /// machine that has no storage. Determinism matters -- it makes
    /// "same input, same logits" a meaningful check.
    pub fn synthetic(cfg: Config, seed: u64) -> Option<Self> {
        let n = cfg.param_count();
        let mut w = Vec::new();
        w.try_reserve_exact(n).ok()?;

        // xorshift64*, scaled to roughly the magnitude real initialisations use.
        // Too large and activations saturate; too small and every logit is
        // identical, which would make the checks below pass vacuously.
        let mut s = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        for _ in 0..n {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let r = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as i32; // 24 bits
            w.push((r as f32 / 8_388_608.0 - 1.0) * 0.04);
        }

        let o = offsets(&cfg, false);
        Some(Self { cfg, src: Source::Flat(w), o })
    }

    /// Load a llama2.c checkpoint.
    ///
    /// The legacy format is a 28-byte header of seven little-endian i32 --
    /// dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len --
    /// followed by every tensor as f32 in the order `offsets` describes. A
    /// *negative* vocab_size is the flag for an untied classifier; the
    /// magnitude is the real size.
    pub fn from_bytes(data: &[u8]) -> Result<Self, LoadError> {
        if data.len() < HEADER_BYTES {
            return Err(LoadError::TooShort);
        }
        let i32_at = |i: usize| {
            let o = i * 4;
            i32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]])
        };
        let (dim, hidden_dim, n_layers) = (i32_at(0), i32_at(1), i32_at(2));
        let (n_heads, n_kv_heads, raw_vocab, seq_len) =
            (i32_at(3), i32_at(4), i32_at(5), i32_at(6));

        if dim <= 0
            || hidden_dim <= 0
            || n_layers <= 0
            || n_heads <= 0
            || n_kv_heads <= 0
            || seq_len <= 0
            || raw_vocab == 0
        {
            return Err(LoadError::BadHeader);
        }

        let cfg = Config {
            dim: dim as usize,
            hidden_dim: hidden_dim as usize,
            n_layers: n_layers as usize,
            n_heads: n_heads as usize,
            n_kv_heads: n_kv_heads as usize,
            vocab_size: raw_vocab.unsigned_abs() as usize,
            seq_len: seq_len as usize,
            shared_classifier: raw_vocab > 0,
            // The legacy format has no field for it; llama2.c hardcodes 10000
            // and every checkpoint in that format was trained with it.
            rope_theta: 10000.0,
            attn_sinks: 0,
            // seq_len means the sum reaches capacity, so nothing is ever
            // evicted and this is off until asked for.
            attn_window: usize::MAX,
        };

        // Both are assumed by head_size() and kv_mul(), which divide.
        if cfg.dim % cfg.n_heads != 0 || cfg.n_heads % cfg.n_kv_heads != 0 {
            return Err(LoadError::BadHeader);
        }

        let have = (data.len() - HEADER_BYTES) / 4;
        let want = cfg.param_count() + cfg.seq_len * cfg.head_size();
        if have < want {
            return Err(LoadError::Truncated { want, have });
        }

        let mut w = Vec::new();
        w.try_reserve_exact(have).map_err(|_| LoadError::OutOfMemory)?;
        for i in 0..have {
            let o = HEADER_BYTES + i * 4;
            w.push(f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]));
        }

        let o = offsets(&cfg, true);
        Ok(Self { cfg, src: Source::Flat(w), o })
    }

    /// Load a GLADOSM2 checkpoint, referencing the blob in place.
    ///
    /// The bytes must outlive the model, which they do: `uefi::read_file`
    /// allocates LoaderData and never frees it, and the frame allocator
    /// deliberately excludes that type so it can never be handed out as free
    /// memory.
    pub fn from_glados(data: &'static [u8]) -> Result<Self, LoadError> {
        if data.len() < GLADOS_HEADER || &data[0..8] != GLADOS_MAGIC {
            return Err(LoadError::BadHeader);
        }
        let u32_at = |o: usize| u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
        let i32_at = |o: usize| i32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);

        if u32_at(8) != GLADOS_VERSION {
            return Err(LoadError::BadHeader);
        }
        let dim = i32_at(12);
        let hidden_dim = i32_at(16);
        let n_layers = i32_at(20);
        let n_heads = i32_at(24);
        let n_kv_heads = i32_at(28);
        let raw_vocab = i32_at(32);
        let seq_len = i32_at(36);
        let rope_theta = f32::from_le_bytes([data[40], data[41], data[42], data[43]]);
        let quant = u32_at(44);

        if dim <= 0
            || hidden_dim <= 0
            || n_layers <= 0
            || n_heads <= 0
            || n_kv_heads <= 0
            || seq_len <= 0
            || raw_vocab == 0
        {
            return Err(LoadError::BadHeader);
        }
        // Only the quantised form is implemented; an f32 GLADOSM2 file would
        // need a different stride everywhere and there is no reason to build
        // one, since f32 is what does not fit.
        if quant != GLADOS_QUANT_I8 {
            return Err(LoadError::BadHeader);
        }

        let cfg = Config {
            dim: dim as usize,
            hidden_dim: hidden_dim as usize,
            n_layers: n_layers as usize,
            n_heads: n_heads as usize,
            n_kv_heads: n_kv_heads as usize,
            vocab_size: raw_vocab.unsigned_abs() as usize,
            seq_len: seq_len as usize,
            shared_classifier: raw_vocab > 0,
            rope_theta,
            attn_sinks: 0,
            // seq_len means the sum reaches capacity, so nothing is ever
            // evicted and this is off until asked for.
            attn_window: usize::MAX,
        };
        if cfg.dim % cfg.n_heads != 0 || cfg.n_heads % cfg.n_kv_heads != 0 {
            return Err(LoadError::BadHeader);
        }

        let (d, h, l, kv, v) = (
            cfg.dim,
            cfg.hidden_dim,
            cfg.n_layers,
            cfg.kv_dim(),
            cfg.vocab_size,
        );

        // Walk the layout the converter wrote: grouped by tensor, then by
        // layer, exactly as `offsets` orders the flat form.
        let mut p = GLADOS_HEADER;
        let mut off = ByteOffsets { embed: p, ..Default::default() };
        p += Self::q8_stride(v, d);

        let rms_att_at = p;
        p += l * d * 4;
        off.wq = p;
        p += l * Self::q8_stride(d, d);
        off.wk = p;
        p += l * Self::q8_stride(kv, d);
        off.wv = p;
        p += l * Self::q8_stride(kv, d);
        off.wo = p;
        p += l * Self::q8_stride(d, d);
        let rms_ffn_at = p;
        p += l * d * 4;
        off.w1 = p;
        p += l * Self::q8_stride(h, d);
        off.w2 = p;
        p += l * Self::q8_stride(d, h);
        off.w3 = p;
        p += l * Self::q8_stride(h, d);
        let rms_final_at = p;
        p += d * 4;
        off.wcls = p;
        if !cfg.shared_classifier {
            p += Self::q8_stride(v, d);
        }

        if data.len() < p {
            return Err(LoadError::Truncated { want: p, have: data.len() });
        }

        // Copy the norms out so the hot path reads aligned f32.
        let mut norms = Vec::new();
        norms.try_reserve_exact(2 * l * d + d).map_err(|_| LoadError::OutOfMemory)?;
        for (base, count) in [(rms_att_at, l * d), (rms_ffn_at, l * d), (rms_final_at, d)] {
            for i in 0..count {
                norms.push(weights::f32_at(&data[base..], i));
            }
        }

        Ok(Self { cfg, src: Source::Blob { bytes: data, off, norms }, o: Offsets::default() })
    }

    pub fn weight_bytes(&self) -> usize {
        match &self.src {
            Source::Flat(w) => w.len() * 4,
            Source::Blob { bytes, norms, .. } => bytes.len() + norms.len() * 4,
        }
    }

    pub fn is_quantised(&self) -> bool {
        matches!(self.src, Source::Blob { .. })
    }

    #[inline]
    fn slice(&self, off: usize, len: usize) -> &[f32] {
        match &self.src {
            Source::Flat(w) => &w[off..off + len],
            // Only the Flat path indexes by float offset; reaching here means
            // a caller was not converted to the Mat accessors.
            Source::Blob { .. } => &[],
        }
    }

    /// One int8 tensor: `rows` f32 scales, then `rows * cols` int8 values.
    fn q8<'a>(bytes: &'a [u8], off: usize, rows: usize, cols: usize) -> Mat<'a> {
        let scales = &bytes[off..off + rows * 4];
        let start = off + rows * 4;
        let raw = &bytes[start..start + rows * cols];
        // i8 has alignment 1, so this cast is always valid -- which is exactly
        // why the quantised half can be read in place while the f32 half
        // cannot.
        let data = unsafe { core::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len()) };
        Mat::Q8 { data, scales, rows, cols }
    }

    /// Bytes one int8 tensor of this shape occupies.
    fn q8_stride(rows: usize, cols: usize) -> usize {
        rows * 4 + rows * cols
    }

    fn mat(&self, flat_off: usize, blob_off: usize, layer: usize, rows: usize, cols: usize) -> Mat<'_> {
        match &self.src {
            Source::Flat(w) => {
                let n = rows * cols;
                let base = flat_off + layer * n;
                Mat::F32 { data: &w[base..base + n], rows, cols }
            }
            Source::Blob { bytes, .. } => {
                Self::q8(bytes, blob_off + layer * Self::q8_stride(rows, cols), rows, cols)
            }
        }
    }

    fn blob_off(&self) -> ByteOffsets {
        match &self.src {
            Source::Blob { off, .. } => *off,
            Source::Flat(_) => ByteOffsets::default(),
        }
    }

    fn wq(&self, l: usize) -> Mat<'_> {
        let d = self.cfg.dim;
        self.mat(self.o.wq, self.blob_off().wq, l, d, d)
    }
    fn wk(&self, l: usize) -> Mat<'_> {
        let (d, kv) = (self.cfg.dim, self.cfg.kv_dim());
        self.mat(self.o.wk, self.blob_off().wk, l, kv, d)
    }
    fn wv(&self, l: usize) -> Mat<'_> {
        let (d, kv) = (self.cfg.dim, self.cfg.kv_dim());
        self.mat(self.o.wv, self.blob_off().wv, l, kv, d)
    }
    fn wo(&self, l: usize) -> Mat<'_> {
        let d = self.cfg.dim;
        self.mat(self.o.wo, self.blob_off().wo, l, d, d)
    }
    fn w1(&self, l: usize) -> Mat<'_> {
        let (d, h) = (self.cfg.dim, self.cfg.hidden_dim);
        self.mat(self.o.w1, self.blob_off().w1, l, h, d)
    }
    fn w2(&self, l: usize) -> Mat<'_> {
        let (d, h) = (self.cfg.dim, self.cfg.hidden_dim);
        self.mat(self.o.w2, self.blob_off().w2, l, d, h)
    }
    fn w3(&self, l: usize) -> Mat<'_> {
        let (d, h) = (self.cfg.dim, self.cfg.hidden_dim);
        self.mat(self.o.w3, self.blob_off().w3, l, h, d)
    }
    fn embed_mat(&self) -> Mat<'_> {
        let (d, v) = (self.cfg.dim, self.cfg.vocab_size);
        self.mat(self.o.token_embedding, self.blob_off().embed, 0, v, d)
    }
    fn classifier(&self) -> Mat<'_> {
        let (d, v) = (self.cfg.dim, self.cfg.vocab_size);
        if self.cfg.shared_classifier {
            self.embed_mat()
        } else {
            self.mat(self.o.wcls, self.blob_off().wcls, 0, v, d)
        }
    }

    fn rms_att_w(&self, l: usize) -> &[f32] {
        let d = self.cfg.dim;
        match &self.src {
            Source::Flat(_) => self.slice(self.o.rms_att + l * d, d),
            Source::Blob { norms, .. } => &norms[l * d..(l + 1) * d],
        }
    }
    fn rms_ffn_w(&self, l: usize) -> &[f32] {
        let (d, n) = (self.cfg.dim, self.cfg.n_layers);
        match &self.src {
            Source::Flat(_) => self.slice(self.o.rms_ffn + l * d, d),
            Source::Blob { norms, .. } => &norms[(n + l) * d..(n + l + 1) * d],
        }
    }
    fn rms_final_w(&self) -> &[f32] {
        let (d, n) = (self.cfg.dim, self.cfg.n_layers);
        match &self.src {
            Source::Flat(_) => self.slice(self.o.rms_final, d),
            Source::Blob { norms, .. } => &norms[2 * n * d..2 * n * d + d],
        }
    }

    /// One decode step: token at position `pos` in, logits out.
    pub fn forward(&self, s: &mut State, token: usize, pos: usize) {
        let c = &self.cfg;
        let dim = c.dim;
        let kv_dim = c.kv_dim();
        let kv_mul = c.kv_mul();
        let head_size = c.head_size();

        // --- where in the cache this token lives, and what else is still there
        //
        // Without a window (`sinks + window >= seq_len`) this is the identity:
        // slot j holds absolute position j, `live` is pos+1, and every angle
        // below is the one the old code computed inline. That is deliberate --
        // it makes "output below the window is bit-identical to before" a
        // property that can be tested rather than hoped for.
        //
        // With one, the cache holds the first `sinks` tokens permanently plus a
        // ring of the most recent `window`. The sinks are not sentiment: a
        // transformer dumps attention mass onto its earliest tokens regardless
        // of what they say, and a plain sliding window that drops them leaves
        // that mass with nowhere to go and the distribution collapses
        // (StreamingLLM, Xiao et al. 2023).
        let cap = c.seq_len;
        let sinks = c.attn_sinks.min(cap);
        let window = c.attn_window.min(cap - sinks);
        let ring = cap - sinks;

        let windowed = sinks + window < cap;
        let n_sinks = if windowed { sinks.min(pos + 1) } else { 0 };
        let n_window = if windowed {
            (pos + 1 - n_sinks).min(window)
        } else {
            pos + 1
        };
        let live = (n_sinks + n_window).min(cap);
        // Absolute position of the oldest entry still in the window.
        let first = pos + 1 - n_window;

        let slot_of = |j: usize| -> usize {
            if !windowed {
                return j;
            }
            if j < n_sinks {
                return j;
            }
            let abs = first + (j - n_sinks);
            if abs < sinks {
                abs
            } else {
                sinks + (abs - sinks) % ring
            }
        };

        // Where this token's key and value go.
        let here = slot_of(live - 1);

        // Embedding lookup is a row fetch, not a matmul against a one-hot
        // vector. When the table is quantised the row is dequantised on the
        // way out.
        let tok = token.min(c.vocab_size - 1);
        self.embed_mat().row_into(tok, &mut s.x);

        for l in 0..c.n_layers {
            // --- attention ---
            tensor::rmsnorm(&mut s.xb, &s.x, self.rms_att_w(l));

            let loff = l * c.seq_len * kv_dim;
            // Keys and values are written straight into the cache slot for this
            // position, so attention below can read the whole history without
            // any copying.
            let (kslice, vslice) = (
                loff + here * kv_dim..loff + here * kv_dim + kv_dim,
                loff + here * kv_dim..loff + here * kv_dim + kv_dim,
            );

            self.wq(l).matvec(&mut s.q, &s.xb);
            {
                let k = &mut s.key_cache[kslice.clone()];
                self.wk(l).matvec(k, &s.xb);
            }
            {
                let v = &mut s.value_cache[vslice.clone()];
                self.wv(l).matvec(v, &s.xb);
            }

            // The key just written stays unrotated; only the query is rotated
            // here, at the position it will occupy in the cache.
            let half = head_size / 2;
            let qpos = live.saturating_sub(1);
            for i in (0..dim).step_by(2) {
                let pair = (i % head_size) / 2;
                let fcr = s.rope_cos[qpos * half + pair];
                let fci = s.rope_sin[qpos * half + pair];
                let q0 = s.q[i];
                let q1 = s.q[i + 1];
                s.q[i] = q0 * fcr - q1 * fci;
                s.q[i + 1] = q0 * fci + q1 * fcr;
            }

            // Rotate every live key into `krot`, indexed by cache position
            // rather than by the absolute position it arrived at. Done once
            // per layer rather than once per head: grouped-query attention
            // shares each key across `kv_mul` heads, so rotating per head
            // would redo the same work three times.
            for j in 0..live {
                let src = loff + slot_of(j) * kv_dim;
                let dst = j * kv_dim;
                for i in (0..kv_dim).step_by(2) {
                    let pair = (i % head_size) / 2;
                    let fcr = s.rope_cos[j * half + pair];
                    let fci = s.rope_sin[j * half + pair];
                    let k0 = s.key_cache[src + i];
                    let k1 = s.key_cache[src + i + 1];
                    s.krot[dst + i] = k0 * fcr - k1 * fci;
                    s.krot[dst + i + 1] = k0 * fci + k1 * fcr;
                }
            }

            let scale = 1.0 / tensor::sqrtf(head_size as f32);
            for h in 0..c.n_heads {
                let qo = h * head_size;
                let ao = h * c.seq_len;
                let hoff = (h / kv_mul) * head_size;
                for t in 0..live {
                    let ko = t * kv_dim + hoff;
                    let mut score = 0.0f32;
                    for i in 0..head_size {
                        score += s.q[qo + i] * s.krot[ko + i];
                    }
                    s.att[ao + t] = score * scale;
                }
                tensor::softmax(&mut s.att[ao..ao + live]);

                for i in 0..head_size {
                    s.xb[qo + i] = 0.0;
                }
                for t in 0..live {
                    let vo = loff + slot_of(t) * kv_dim + hoff;
                    let a = s.att[ao + t];
                    for i in 0..head_size {
                        s.xb[qo + i] += a * s.value_cache[vo + i];
                    }
                }
            }

            self.wo(l).matvec(&mut s.xb2, &s.xb);
            tensor::add_into(&mut s.x, &s.xb2);

            // --- feed forward ---
            tensor::rmsnorm(&mut s.xb, &s.x, self.rms_ffn_w(l));
            self.w1(l).matvec(&mut s.hb, &s.xb);
            self.w3(l).matvec(&mut s.hb2, &s.xb);
            tensor::swiglu(&mut s.hb, &s.hb2);
            self.w2(l).matvec(&mut s.xb, &s.hb);
            tensor::add_into(&mut s.x, &s.xb);
        }

        // Normalise into xb rather than cloning x into a temporary. The clone
        // was a heap allocation on every single token, in the one loop that
        // runs most often.
        tensor::rmsnorm(&mut s.xb, &s.x, self.rms_final_w());
        self.classifier().matvec(&mut s.logits, &s.xb);
    }

    /// One row of the token embedding table, written into `out`.
    ///
    /// Takes a buffer rather than returning a slice because a quantised table
    /// has no f32 row to borrow -- it has to be dequantised somewhere, and
    /// making that the caller's buffer keeps it off the heap.
    ///
    /// Used by the vocabulary extension, which initialises new rows by pooling
    /// these: a token that never existed during training has to start
    /// somewhere, and the average of the words describing it beats noise.
    pub fn embed_into(&self, token: usize, out: &mut [f32]) {
        let t = token.min(self.cfg.vocab_size - 1);
        self.embed_mat().row_into(t, out);
    }
}

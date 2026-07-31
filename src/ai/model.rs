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
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

impl State {
    pub fn new(cfg: &Config) -> Self {
        let kv = cfg.kv_dim();
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
        }
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

#[derive(Debug, Clone, Copy)]
pub enum LoadError {
    TooShort,
    BadHeader,
    /// The header describes more weights than the file contains, which usually
    /// means a truncated download rather than a format mismatch.
    Truncated { want: usize, have: usize },
    OutOfMemory,
}

pub struct Model {
    pub cfg: Config,
    w: Vec<f32>,
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
        Some(Self { cfg, w, o })
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
        Ok(Self { cfg, w, o })
    }

    pub fn weight_bytes(&self) -> usize {
        self.w.len() * 4
    }

    #[inline]
    fn slice(&self, off: usize, len: usize) -> &[f32] {
        &self.w[off..off + len]
    }

    /// One decode step: token at position `pos` in, logits out.
    pub fn forward(&self, s: &mut State, token: usize, pos: usize) {
        let c = &self.cfg;
        let dim = c.dim;
        let kv_dim = c.kv_dim();
        let kv_mul = c.kv_mul();
        let head_size = c.head_size();
        let hidden = c.hidden_dim;

        // Embedding lookup is a copy, not a matmul against a one-hot vector.
        let tok = token.min(c.vocab_size - 1);
        s.x.copy_from_slice(self.slice(self.o.token_embedding + tok * dim, dim));

        for l in 0..c.n_layers {
            // --- attention ---
            let rms_att = self.slice(self.o.rms_att + l * dim, dim);
            tensor::rmsnorm(&mut s.xb, &s.x, rms_att);

            let loff = l * c.seq_len * kv_dim;
            // Keys and values are written straight into the cache slot for this
            // position, so attention below can read the whole history without
            // any copying.
            let (kslice, vslice) = (
                loff + pos * kv_dim..loff + pos * kv_dim + kv_dim,
                loff + pos * kv_dim..loff + pos * kv_dim + kv_dim,
            );

            tensor::matmul(&mut s.q, &s.xb, self.slice(self.o.wq + l * dim * dim, dim * dim), dim, dim);
            {
                let k = &mut s.key_cache[kslice.clone()];
                tensor::matmul(k, &s.xb, self.slice(self.o.wk + l * dim * kv_dim, dim * kv_dim), dim, kv_dim);
            }
            {
                let v = &mut s.value_cache[vslice.clone()];
                tensor::matmul(v, &s.xb, self.slice(self.o.wv + l * dim * kv_dim, dim * kv_dim), dim, kv_dim);
            }

            // RoPE on q and k. The rotation frequency depends on the position
            // *within a head*, not within the whole vector, which is why this
            // uses `i % head_size` rather than `i`.
            for i in (0..dim).step_by(2) {
                let head_dim = i % head_size;
                let freq = 1.0 / tensor::powf(10000.0, head_dim as f32 / head_size as f32);
                let val = pos as f32 * freq;
                let (fci, fcr) = (tensor::sinf(val), tensor::cosf(val));

                let q0 = s.q[i];
                let q1 = s.q[i + 1];
                s.q[i] = q0 * fcr - q1 * fci;
                s.q[i + 1] = q0 * fci + q1 * fcr;

                // Only the first kv_dim lanes exist in the key vector when
                // grouped-query attention narrows it.
                if i < kv_dim {
                    let base = loff + pos * kv_dim;
                    let k0 = s.key_cache[base + i];
                    let k1 = s.key_cache[base + i + 1];
                    s.key_cache[base + i] = k0 * fcr - k1 * fci;
                    s.key_cache[base + i + 1] = k0 * fci + k1 * fcr;
                }
            }

            let scale = 1.0 / tensor::sqrtf(head_size as f32);
            for h in 0..c.n_heads {
                let qo = h * head_size;
                let ao = h * c.seq_len;
                // Attend over every position up to and including this one.
                for t in 0..=pos {
                    let ko = loff + t * kv_dim + (h / kv_mul) * head_size;
                    let mut score = 0.0f32;
                    for i in 0..head_size {
                        score += s.q[qo + i] * s.key_cache[ko + i];
                    }
                    s.att[ao + t] = score * scale;
                }
                tensor::softmax(&mut s.att[ao..ao + pos + 1]);

                for i in 0..head_size {
                    s.xb[qo + i] = 0.0;
                }
                for t in 0..=pos {
                    let vo = loff + t * kv_dim + (h / kv_mul) * head_size;
                    let a = s.att[ao + t];
                    for i in 0..head_size {
                        s.xb[qo + i] += a * s.value_cache[vo + i];
                    }
                }
            }

            tensor::matmul(&mut s.xb2, &s.xb, self.slice(self.o.wo + l * dim * dim, dim * dim), dim, dim);
            tensor::add_into(&mut s.x, &s.xb2);

            // --- feed forward ---
            let rms_ffn = self.slice(self.o.rms_ffn + l * dim, dim);
            tensor::rmsnorm(&mut s.xb, &s.x, rms_ffn);
            tensor::matmul(&mut s.hb, &s.xb, self.slice(self.o.w1 + l * hidden * dim, hidden * dim), dim, hidden);
            tensor::matmul(&mut s.hb2, &s.xb, self.slice(self.o.w3 + l * hidden * dim, hidden * dim), dim, hidden);
            tensor::swiglu(&mut s.hb, &s.hb2);
            tensor::matmul(&mut s.xb, &s.hb, self.slice(self.o.w2 + l * dim * hidden, dim * hidden), hidden, dim);
            tensor::add_into(&mut s.x, &s.xb);
        }

        let rms_final = self.slice(self.o.rms_final, dim);
        // Normalise into xb rather than cloning x into a temporary. The clone
        // was a heap allocation on every single token, in the one loop that
        // runs most often.
        tensor::rmsnorm(&mut s.xb, &s.x, rms_final);
        tensor::matmul(
            &mut s.logits,
            &s.xb,
            self.slice(self.o.wcls, c.vocab_size * dim),
            dim,
            c.vocab_size,
        );
    }

    /// One row of the token embedding table.
    ///
    /// Public because the vocabulary extension initialises its new rows by
    /// pooling these: a token that never existed during training has to start
    /// somewhere, and the average of the words describing it is a far better
    /// starting point than noise.
    pub fn embed(&self, token: usize) -> &[f32] {
        let t = token.min(self.cfg.vocab_size - 1);
        self.slice(self.o.token_embedding + t * self.cfg.dim, self.cfg.dim)
    }
}

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

fn offsets(cfg: &Config) -> Offsets {
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

    pub fn bytes(&self, cfg: &Config) -> usize {
        let kv = cfg.kv_dim();
        4 * (cfg.dim * 4
            + cfg.hidden_dim * 2
            + cfg.n_heads * cfg.seq_len
            + cfg.vocab_size
            + 2 * cfg.n_layers * cfg.seq_len * kv)
    }
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

        let o = offsets(&cfg);
        Some(Self { cfg, w, o })
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
        let x = s.x.clone();
        tensor::rmsnorm(&mut s.x, &x, rms_final);
        tensor::matmul(
            &mut s.logits,
            &s.x,
            self.slice(self.o.wcls, c.vocab_size * dim),
            dim,
            c.vocab_size,
        );
    }
}

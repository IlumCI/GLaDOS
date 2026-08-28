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

/// Which forward pass a checkpoint wants.
///
/// Not a cosmetic label. `Dense` and `Qwen35` disagree about what a layer even
/// contains: three layers in four of a Qwen3.5 hold a recurrence with no KV
/// cache at all, and the fourth holds attention whose query projection is
/// twice as wide as its query.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arch {
    /// Llama, Qwen2, Qwen3. Every layer identical, full attention throughout.
    Dense,
    /// Qwen3.5. Gated DeltaNet in three layers of four, full attention in the
    /// fourth, and a dense SwiGLU feed-forward in all of them.
    Qwen35,
    /// Qwen3.5-MoE. The same layer schedule with a sparse feed-forward.
    Qwen35Moe,
}

/// What one layer of a hybrid holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayerKind {
    Full,
    Linear,
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// Width of one attention head.
    ///
    /// Not `dim / n_heads`. Llama defines it that way and every model here did
    /// until Qwen3, which states it separately: 1024 wide, 16 heads, and a head
    /// size of 128 rather than the 64 that division gives. So `wq` is
    /// `[2048, 1024]` and the attention path is *wider* than the residual
    /// stream it reads from.
    ///
    /// This is the field to suspect first when a converted model produces
    /// fluent nonsense: the old formula divides evenly for Qwen3 and yields a
    /// perfectly self-consistent set of wrong shapes.
    pub head_dim: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    /// The constant inside RMSNorm's square root, from the checkpoint.
    pub norm_eps: f32,
    /// Whether each head's query and key are RMSNormed before RoPE.
    ///
    /// Qwen3 does this; Llama and SmolLM2 do not. Skipping it costs no shape
    /// mismatch and no error -- attention simply attends to the wrong things.
    pub qk_norm: bool,
    /// Which dimensions RoPE pairs together.
    ///
    /// `true` pairs 2i with 2i+1, which is what llama2.c does and what the
    /// GPT-NeoX paper describes. `false` pairs i with i + head_size/2, which is
    /// what `rotate_half` in HuggingFace's modeling code does -- and therefore
    /// what every checkpoint trained through transformers expects, Llama and
    /// SmolLM2 included. SmolLM2's config states it outright:
    /// `rope_interleaved: false`.
    ///
    /// Both are norm-preserving rotations by the same set of angles, so the
    /// wrong one produces no error, no NaN and no drift in magnitude. The model
    /// stays fluent and simply attends by a scrambled notion of distance. It
    /// reads as a model that knows the topic and gets the facts wrong, which is
    /// indistinguishable from a small model being small.
    pub rope_interleaved: bool,
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

    // --- v4, and every one of these is `Dense`'s answer for a v2/v3 file ---
    pub arch: Arch,
    /// How many of each head's dimensions RoPE actually rotates.
    ///
    /// Equal to `head_dim` for everything before Qwen3.5, which rotates 64 of
    /// 256 and passes the rest through untouched. Stored as a count rather
    /// than as the config's fraction because the kernel wants a number of
    /// dimensions and `0.25 * 256` is a question with one answer.
    pub rotary_dim: usize,
    /// Whether `wq` emits a gate alongside the query and the attention output
    /// is multiplied by `sigmoid(gate)` before `wo`.
    pub attn_output_gate: bool,

    /// Gated DeltaNet geometry. All zero unless `arch` is a hybrid.
    pub lin_k_head: usize,
    pub lin_v_head: usize,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub conv_kernel: usize,

    /// Mixture-of-experts geometry. All zero unless `arch` is `Qwen35Moe`.
    pub n_experts: usize,
    pub experts_per_tok: usize,
    pub shared_dim: usize,

    /// One bit per layer, set for full attention.
    ///
    /// Written down rather than derived from `full_attention_interval`. The
    /// interval is 4 and describes every published checkpoint exactly, which
    /// is the argument for computing it and the reason not to: a checkpoint
    /// that broke the pattern would load and run wrong instead of failing.
    pub layer_full: [u32; 8],
}

impl Default for Config {
    /// A zero-sized Llama. Exists so the four struct literals that build a real
    /// one can say `..Default::default()` and stay readable as the hybrid
    /// fields accumulate.
    fn default() -> Self {
        Self {
            dim: 0,
            hidden_dim: 0,
            n_layers: 0,
            n_heads: 0,
            n_kv_heads: 0,
            head_dim: 0,
            vocab_size: 0,
            seq_len: 0,
            norm_eps: 1e-5,
            qk_norm: false,
            rope_interleaved: false,
            shared_classifier: true,
            rope_theta: 10000.0,
            attn_sinks: 0,
            attn_window: usize::MAX,
            arch: Arch::Dense,
            rotary_dim: 0,
            attn_output_gate: false,
            lin_k_head: 0,
            lin_v_head: 0,
            lin_k_heads: 0,
            lin_v_heads: 0,
            conv_kernel: 0,
            n_experts: 0,
            experts_per_tok: 0,
            shared_dim: 0,
            layer_full: [0; 8],
        }
    }
}

impl Config {
    pub fn head_size(&self) -> usize {
        self.head_dim
    }

    pub fn hybrid(&self) -> bool {
        !matches!(self.arch, Arch::Dense)
    }

    /// What layer `l` holds. Every layer of a dense model is `Full`.
    pub fn layer_kind(&self, l: usize) -> LayerKind {
        if !self.hybrid() || self.layer_full[l / 32] >> (l % 32) & 1 == 1 {
            LayerKind::Full
        } else {
            LayerKind::Linear
        }
    }

    /// Which KV cache belongs to layer `l`, or `None` if it needs none.
    ///
    /// The caches are allocated only for full-attention layers and packed, so
    /// this counts set bits below `l`. Allocating `n_layers` of them and using
    /// six would throw away the entire memory argument, which is the only
    /// reason this port exists.
    pub fn kv_slot(&self, l: usize) -> Option<usize> {
        if self.layer_kind(l) != LayerKind::Full {
            return None;
        }
        if !self.hybrid() {
            return Some(l);
        }
        let mut n = 0;
        for i in 0..l {
            if self.layer_kind(i) == LayerKind::Full {
                n += 1;
            }
        }
        Some(n)
    }

    /// Which recurrent state belongs to layer `l`, packed the same way.
    pub fn lin_slot(&self, l: usize) -> Option<usize> {
        if !self.hybrid() || self.layer_kind(l) != LayerKind::Linear {
            return None;
        }
        let mut n = 0;
        for i in 0..l {
            if self.layer_kind(i) == LayerKind::Linear {
                n += 1;
            }
        }
        Some(n)
    }

    pub fn n_full_layers(&self) -> usize {
        (0..self.n_layers).filter(|&l| self.layer_kind(l) == LayerKind::Full).count()
    }

    pub fn n_linear_layers(&self) -> usize {
        self.n_layers - self.n_full_layers()
    }

    /// Width of the delta net's key half, and of its value half.
    pub fn lin_k_dim(&self) -> usize {
        self.lin_k_head * self.lin_k_heads
    }
    pub fn lin_v_dim(&self) -> usize {
        self.lin_v_head * self.lin_v_heads
    }
    /// What `in_proj_qkv` produces and what the convolution runs over.
    pub fn conv_dim(&self) -> usize {
        2 * self.lin_k_dim() + self.lin_v_dim()
    }
    /// How many value heads share each key head.
    pub fn lin_mul(&self) -> usize {
        if self.lin_k_heads == 0 { 1 } else { self.lin_v_heads / self.lin_k_heads }
    }

    /// How many dimensions RoPE rotates, with 0 meaning "the checkpoint did
    /// not say", which is the Llama identity: all of them.
    pub fn rot_dim(&self) -> usize {
        if self.rotary_dim == 0 { self.head_dim } else { self.rotary_dim }
    }

    /// Floats in one layer's recurrent state, and in its convolution ring.
    ///
    /// The first is the whole argument for this architecture: it is the same
    /// size at token 32 as at token 32,768.
    pub fn lin_state_len(&self) -> usize {
        self.lin_v_heads * self.lin_k_head * self.lin_v_head
    }
    pub fn conv_ring_len(&self) -> usize {
        self.conv_dim() * self.conv_kernel.saturating_sub(1)
    }

    /// Sinks and window, clamped to what the checkpoint was trained for.
    fn window_parts(&self) -> (usize, usize) {
        let sinks = self.attn_sinks.min(self.seq_len);
        let window = self.attn_window.min(self.seq_len - sinks);
        (sinks, window)
    }

    /// Whether the cache evicts, and therefore whether position may run past
    /// `seq_len`.
    ///
    /// The default is `attn_window: usize::MAX`, which clamps to the whole
    /// trained length and makes this false -- the cache holds every position
    /// and behaves exactly as it did before eviction existed.
    pub fn streams(&self) -> bool {
        let (sinks, window) = self.window_parts();
        sinks + window < self.seq_len
    }

    /// How many cache slots actually have to exist.
    ///
    /// This is what `State` allocates, and it is the whole reason a window is
    /// worth having. Previously the cache was sized by `seq_len` whether or not
    /// a window was configured, so windowing bought compute and nothing else --
    /// the memory was allocated regardless and most of it held stale entries.
    /// At Qwen3's trained 32768 that distinction is the difference between
    /// 7 GiB and whatever the window costs.
    pub fn live_cap(&self) -> usize {
        let (sinks, window) = self.window_parts();
        if sinks + window >= self.seq_len {
            self.seq_len
        } else {
            sinks + window
        }
    }
    /// Total width of the query projection, which is what `wq` produces and
    /// what `wo` consumes. Equal to `dim` for every Llama-shaped model, and
    /// twice it for Qwen3-0.6B.
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    /// Width of the key/value projections. Smaller than `q_dim` when the model
    /// uses grouped-query attention.
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
    /// How many query heads share each key/value head.
    pub fn kv_mul(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    pub fn param_count(&self) -> usize {
        let (d, h, l, q, kv, v) = (
            self.dim,
            self.hidden_dim,
            self.n_layers,
            self.q_dim(),
            self.kv_dim(),
            self.vocab_size,
        );
        let mut n = v * d          // token embedding
            + l * d                // rms_att
            + l * q * d            // wq
            + l * kv * d           // wk
            + l * kv * d           // wv
            + l * d * q            // wo
            + l * d                // rms_ffn
            + l * h * d            // w1
            + l * d * h            // w2
            + l * h * d            // w3
            + d; // rms_final
        if self.qk_norm {
            n += 2 * l * self.head_dim;
        }
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
    q_norm: usize,
    k_norm: usize,
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
    let (d, h, l, q, kv, v) = (
        cfg.dim,
        cfg.hidden_dim,
        cfg.n_layers,
        cfg.q_dim(),
        cfg.kv_dim(),
        cfg.vocab_size,
    );
    let mut o = Offsets::default();
    let mut p = 0usize;
    o.token_embedding = p;
    p += v * d;
    o.rms_att = p;
    p += l * d;
    // QK-Norm weights sit with the other attention norms rather than beside the
    // projections they modify, so that a model without them leaves a hole of
    // length zero and every later offset is unchanged.
    if cfg.qk_norm {
        o.q_norm = p;
        p += l * cfg.head_dim;
        o.k_norm = p;
        p += l * cfg.head_dim;
    }
    o.wq = p;
    p += l * q * d;
    o.wk = p;
    p += l * kv * d;
    o.wv = p;
    p += l * kv * d;
    o.wo = p;
    p += l * d * q;
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

/// Values per f32 scale in the KV cache. Must match `KV_BLOCK` in
/// `tools/reference.py`, which is where the number was chosen.
const KV_BLOCK: usize = 32;

/// Prompt positions fed per weight pass in `prefill`.
///
/// Bounds the chunk scratch (about 3 MiB at Qwen3's wide heads) while
/// amortising each weight byte over two orders of magnitude more work than a
/// single token. Must never exceed the window ring; `prefill` clamps.
const PRE_FILL_CHUNK: usize = 64;

/// One layer's keys or values, int8 with a scale per block.
///
/// The cache is the largest thing the system allocates and it dominates how
/// much context fits: at f32 it is 224 KiB per token, which puts Qwen3's
/// trained 32768 at 7 GiB. int8 takes that to 63 KiB and 2 GiB.
///
/// Blocks of 32 rather than one scale per head, measured on Qwen3-0.6B against
/// the f32 path on the same prompt:
///
/// ```text
///     per head (128)   max dlogit 2.34   mean 0.41   top-5 order shifted
///     block 32         max dlogit 1.09   mean 0.18   top-5 order preserved
/// ```
///
/// and under block 32, 45 greedy tokens come out identical to f32. The extra
/// scales cost 12.5% on top of the data, taking the saving from 4x to 3.5x --
/// worth it to keep the ranking intact, since reordering the top few
/// candidates is exactly what a sampler acts on.
///
/// Keys carry almost all of the error: quantising only values costs 0.02 of
/// that 0.41, only keys costs 0.35. A key goes through a dot product and then a
/// softmax, where error is amplified; a value is only averaged. Both are
/// quantised anyway, because halving the saving would defeat the purpose.
#[derive(Clone)]
struct KvLayer {
    data: Vec<i8>,
    scales: Vec<f32>,
    kv_dim: usize,
    /// Scales per position. The last block is short when `kv_dim` is not a
    /// multiple of `KV_BLOCK`; no model here has that shape, which is exactly
    /// why it has to be handled rather than assumed away.
    blocks: usize,
}

impl KvLayer {
    fn new(cap: usize, kv_dim: usize) -> Self {
        let blocks = kv_dim.div_ceil(KV_BLOCK);
        Self {
            data: vec![0; cap * kv_dim],
            scales: vec![0.0; cap * blocks],
            kv_dim,
            blocks,
        }
    }

    /// Quantise one position into `slot`.
    fn store(&mut self, slot: usize, src: &[f32]) {
        let base = slot * self.kv_dim;
        let sbase = slot * self.blocks;
        for b in 0..self.blocks {
            let lo = b * KV_BLOCK;
            let hi = (lo + KV_BLOCK).min(self.kv_dim);
            let mut peak = 0.0f32;
            for &v in &src[lo..hi] {
                let a = if v < 0.0 { -v } else { v };
                if a > peak {
                    peak = a;
                }
            }
            // A block of exact zeros quantises to zero under any scale, so the
            // scale itself is arbitrary; 0 keeps dequantisation exact.
            let scale = if peak == 0.0 { 0.0 } else { peak / 127.0 };
            self.scales[sbase + b] = scale;
            let inv = if scale == 0.0 { 0.0 } else { 1.0 / scale };
            for i in lo..hi {
                let q = tensor::roundf(src[i] * inv);
                self.data[base + i] = q.clamp(-127.0, 127.0) as i8;
            }
        }
    }

    /// Dequantise one element of one position.
    #[inline]
    fn at(&self, slot: usize, i: usize) -> f32 {
        self.data[slot * self.kv_dim + i] as f32
            * self.scales[slot * self.blocks + i / KV_BLOCK]
    }

    /// Dequantise a whole position, for export.
    fn read_into(&self, slot: usize, out: &mut [f32]) {
        for (i, o) in out.iter_mut().enumerate().take(self.kv_dim) {
            *o = self.at(slot, i);
        }
    }
}

/// One linear-attention layer's memory.
///
/// Kept in f32 deliberately, where the KV cache is int8. The cache is
/// enormous and each entry is read once per token; this is small and is
/// **read and written every step**, so quantisation error would compound
/// through the recurrence rather than average out across it.
#[derive(Clone)]
struct LinearState {
    /// `[v_heads][k_head_dim][v_head_dim]`. Independent of context length --
    /// this is 1 MiB per layer at token 32 and 1 MiB per layer at token
    /// 32,768, which is the entire reason for the port.
    s: Vec<f32>,
    /// The previous `conv_kernel - 1` inputs to the depthwise convolution,
    /// oldest first, as a ring.
    conv: Vec<f32>,
    conv_at: usize,
}

impl LinearState {
    fn new(cfg: &Config) -> Self {
        Self {
            s: vec![0.0; cfg.lin_state_len()],
            conv: vec![0.0; cfg.conv_ring_len()],
            conv_at: 0,
        }
    }

    fn clear(&mut self) {
        self.s.fill(0.0);
        self.conv.fill(0.0);
        self.conv_at = 0;
    }
}

/// Which cache slots hold live entries, and where each one physically sits.
///
/// Extracted so the dense and hybrid passes cannot drift: the arithmetic is
/// subtle -- an off-by-one in `first` reads a neighbour's key and produces no
/// error -- and two transcriptions of it would eventually disagree.
///
/// Without a window (`sinks + window >= seq_len`) every method here is the
/// identity: slot `j` holds absolute position `j` and `live` is `pos + 1`.
/// That is deliberate, and it makes "output below the window is bit-identical
/// to before" a property that can be tested rather than hoped for.
struct Window {
    windowed: bool,
    sinks: usize,
    ring: usize,
    n_sinks: usize,
    first: usize,
    live: usize,
}

impl Window {
    fn new(c: &Config, cap: usize, pos: usize) -> Self {
        // The sinks are not sentiment: a transformer dumps attention mass onto
        // its earliest tokens regardless of what they say, and a plain sliding
        // window that drops them leaves that mass with nowhere to go and the
        // distribution collapses (StreamingLLM, Xiao et al. 2023).
        let windowed = c.streams();
        let sinks = c.attn_sinks.min(cap);
        // The ring is the whole allocation past the sinks, which is exactly
        // the window rather than whatever was left over from `seq_len`.
        let ring = cap - sinks;
        let n_sinks = if windowed { sinks.min(pos + 1) } else { 0 };
        let n_window =
            if windowed { (pos + 1 - n_sinks).min(ring) } else { pos + 1 };
        Self {
            windowed,
            sinks,
            ring,
            n_sinks,
            // Absolute position of the oldest entry still in the window.
            first: pos + 1 - n_window,
            live: (n_sinks + n_window).min(cap),
        }
    }

    fn slot_of(&self, j: usize) -> usize {
        if !self.windowed {
            return j;
        }
        if j < self.n_sinks {
            return j;
        }
        let abs = self.first + (j - self.n_sinks);
        if abs < self.sinks {
            abs
        } else {
            self.sinks + (abs - self.sinks) % self.ring
        }
    }
}

/// Scratch buffers for one forward pass, plus the KV cache.
///
/// Allocated once and reused. Allocating per token would put the heap
/// allocator in the inner loop of generation for no reason.
/// Somewhere to accumulate one adapted site's gradients.
pub struct SiteGrad {
    pub ga: Vec<f32>,
    pub gb: Vec<f32>,
    pub dm: Vec<f32>,
}

impl SiteGrad {
    fn like(d: &super::adapter::Dora) -> SiteGrad {
        SiteGrad {
            ga: vec![0.0; d.a.len()],
            gb: vec![0.0; d.b.len()],
            dm: vec![0.0; d.m.len()],
        }
    }
    fn empty() -> SiteGrad {
        SiteGrad { ga: Vec::new(), gb: Vec::new(), dm: Vec::new() }
    }
    pub fn clear(&mut self) {
        for v in self.ga.iter_mut().chain(self.gb.iter_mut()).chain(self.dm.iter_mut()) {
            *v = 0.0;
        }
    }
}

/// Gradients for every adapted site in the model.
///
/// Shaped from the attached adapters rather than from the config, so a site
/// that is not adapted has nowhere to write and cannot silently accumulate
/// into a buffer nobody reads.
pub struct Grads {
    pub qkv: Vec<[SiteGrad; 3]>,
    pub cls: SiteGrad,
}

impl Grads {
    pub fn new(ad: &super::adapter::Adapters) -> Grads {
        Grads {
            qkv: ad
                .qkv
                .iter()
                .map(|t| {
                    [
                        t[0].as_ref().map(SiteGrad::like).unwrap_or_else(SiteGrad::empty),
                        t[1].as_ref().map(SiteGrad::like).unwrap_or_else(SiteGrad::empty),
                        t[2].as_ref().map(SiteGrad::like).unwrap_or_else(SiteGrad::empty),
                    ]
                })
                .collect(),
            cls: ad.cls.as_ref().map(SiteGrad::like).unwrap_or_else(SiteGrad::empty),
        }
    }

    pub fn clear(&mut self) {
        for t in self.qkv.iter_mut() {
            for s in t.iter_mut() {
                s.clear();
            }
        }
        self.cls.clear();
    }
}

/// What a backward pass needs kept from a forward one.
///
/// Only the residual stream entering each layer, and the normed hidden state
/// the classifier saw. Everything else a layer computes -- the normed input,
/// q, the attention output, the two FFN branches -- is a function of that
/// entering stream and the frozen weights, so it is **recomputed** during the
/// backward walk rather than stored.
///
/// The alternative was measured on paper first. Keeping every intermediate for
/// Qwen3-0.6B costs about 53 KB per layer per position -- 95 MB for a
/// 64-token prompt. Keeping only the entering stream costs 4 KB, so 7 MB for
/// the same prompt, in exchange for roughly one extra forward per layer during
/// the backward. In a kernel whose heap is one physically contiguous
/// allocation on a ladder, trading compute for a factor of thirteen in memory
/// is not a close call.
///
/// `backward.rs` was already written this way: `attention_backward` recomputes
/// scores and softmax rather than reading a saved tape. Nobody writes that
/// unless the intention was to recompute.
///
/// The keys and values are not here because they are already in the KV cache,
/// which is what a KV cache is. They are stored quantised, so a gradient taken
/// through them carries the cache's quantisation error -- true of any training
/// against a quantised cache, and worth knowing before a result is read as
/// noise.
pub struct Tape {
    pub dim: usize,
    pub n_layers: usize,
    pub seq: usize,
    /// `(l * seq + t) * dim` -- the residual stream entering layer `l` at
    /// position `t`.
    x: Vec<f32>,
    /// The final normed hidden state per position, which is what the
    /// classifier consumed and where the loss enters.
    final_xb: Vec<f32>,
    /// How many positions were actually written, so a short sequence cannot
    /// be read as a full one of stale values.
    filled: usize,
}

impl Tape {
    pub fn new(cfg: &Config, seq: usize) -> Tape {
        Tape {
            dim: cfg.dim,
            n_layers: cfg.n_layers,
            seq,
            // One row per layer, plus one for the stream *leaving* the last.
            // The final norm's adjoint needs its own input, and that vector is
            // not the input to any layer -- without it the walk cannot take
            // its first step.
            x: vec![0.0; (cfg.n_layers + 1) * seq * cfg.dim],
            final_xb: vec![0.0; seq * cfg.dim],
            filled: 0,
        }
    }

    /// Bytes held, for anything deciding whether a sequence fits.
    pub fn bytes(&self) -> usize {
        4 * (self.x.len() + self.final_xb.len())
    }

    pub fn filled(&self) -> usize {
        self.filled
    }

    /// The stream entering layer `l` at position `t`.
    /// `l == n_layers` is the stream leaving the last layer, which is what the
    /// final norm consumed.
    pub fn entering(&self, l: usize, t: usize) -> Option<&[f32]> {
        if l > self.n_layers || t >= self.filled {
            return None;
        }
        let o = (l * self.seq + t) * self.dim;
        Some(&self.x[o..o + self.dim])
    }

    /// What the classifier saw at position `t`.
    pub fn final_normed(&self, t: usize) -> Option<&[f32]> {
        if t >= self.filled {
            return None;
        }
        let o = t * self.dim;
        Some(&self.final_xb[o..o + self.dim])
    }

    /// Out of range writes nothing rather than panicking. This runs in ring 0
    /// with no guard page, and a tape sized for one prompt being handed a
    /// longer one should lose the tail, not the machine.
    fn put(buf: &mut [f32], o: usize, v: &[f32]) {
        if o + v.len() <= buf.len() {
            buf[o..o + v.len()].copy_from_slice(v);
        }
    }

    fn record_layer(&mut self, l: usize, t: usize, x: &[f32]) {
        if l > self.n_layers || t >= self.seq {
            return;
        }
        let o = (l * self.seq + t) * self.dim;
        Self::put(&mut self.x, o, &x[..self.dim.min(x.len())]);
    }

    fn record_final(&mut self, t: usize, xb: &[f32]) {
        if t >= self.seq {
            return;
        }
        let o = t * self.dim;
        Self::put(&mut self.final_xb, o, &xb[..self.dim.min(xb.len())]);
        self.filled = self.filled.max(t + 1);
    }
}

#[derive(Clone)]
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
    /// One allocation per layer, not one for the whole cache.
    ///
    /// This is the largest thing the system allocates and it used to be a
    /// single `Vec`, which meant it needed one unbroken block from the heap's
    /// first-fit walk. Nothing required that -- the index was always
    /// `l * seq_len * kv_dim + ...`, i.e. a layer stride over a flat buffer --
    /// and it put a hard ceiling on context: at Qwen3's trained 32768 the
    /// single buffer would be 3.5 GiB contiguous, which no real memory map
    /// offers. Split per layer the same cache is 28 allocations of a few tens
    /// of MiB, which the heap can satisfy out of separate regions.
    key_cache: Vec<KvLayer>,
    value_cache: Vec<KvLayer>,
    /// Staging for one position's keys and values.
    ///
    /// The projections produce f32 and QK-Norm runs on that, so the cache
    /// cannot be written directly by `matvec` any more -- quantisation needs
    /// the whole block before it knows the scale. One `kv_dim` buffer each,
    /// allocated once.
    kbuf: Vec<f32>,
    vbuf: Vec<f32>,
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
    /// One per linear-attention layer, packed. Empty for a dense model.
    linear: Vec<LinearState>,
    /// Scratch for the delta net: the convolution's input and output, the
    /// output gate, the two decay projections, and the recurrence's result.
    /// All empty for a dense model, which allocates none of this.
    qkv: Vec<f32>,
    zbuf: Vec<f32>,
    abuf: Vec<f32>,
    bbuf: Vec<f32>,
    core: Vec<f32>,
    /// One value head's worth of delta, between the recurrence's two passes.
    dbuf: Vec<f32>,
    /// Query and gate, which leave a hybrid's `wq` together.
    qg: Vec<f32>,
    /// Adapter rank scratch, shared by every adapted site within one token.
    /// Single core means single ownership; sized once by [`State::new`].
    la: Vec<f32>,
    /// Router logits and the expert bank's doubled output. Empty unless MoE.
    router: Vec<f32>,
    moe_gu: Vec<f32>,
    /// Width of the residual stream, kept so `hidden` can bound itself.
    dim: usize,
    /// Slots actually allocated. Held rather than recomputed from the config,
    /// so that a config whose window changed after allocation cannot be used to
    /// index buffers sized for the old one -- the mismatch would be a silent
    /// out-of-bounds into a neighbouring layer's keys, not a panic.
    cap: usize,
}

impl State {
    /// A copy of the live state, for deliberation: fork the mind, explore
    /// one candidate each way, keep the survivor. Every field here is an
    /// owned buffer -- KV cache, recurrent states, scratch, logits -- so a
    /// clone is a complete mind at this instant, pos travelling with it.
    /// The cost is one memcpy of the largest allocation in the system,
    /// which is why forks are budgeted by the caller and taken only when
    /// the cheap tiers were not confident enough to answer alone.
    pub fn fork(&self) -> Self {
        self.clone()
    }

    pub fn new(cfg: &Config) -> Self {
        let kv = cfg.kv_dim();        // Half the *rotated* width, not half the head. They are equal for
        // everything before Qwen3.5, which rotates 64 of 256 and leaves the
        // rest alone, so the table is a quarter the size and the loop below
        // is the one it always was.
        let half = cfg.rot_dim() / 2;
        // Everything below is sized by the number of slots that can be live at
        // once, not by the trained length. Without a window the two are equal
        // and this is the allocation it always was.
        //
        // Angles are indexed by *cache* position rather than absolute position
        // -- that is what makes eviction possible at all, and it is why the
        // table needs `cap` entries and not `seq_len` however far generation
        // runs.
        let cap = cfg.live_cap();
        let hyb = cfg.hybrid();

        let mut rope_cos = vec![0.0f32; cap * half];
        let mut rope_sin = vec![0.0f32; cap * half];
        for p in 0..cap {
            for i in 0..half {
                // Matches the old inline computation exactly: the exponent is
                // the pair index doubled over head_size, so that a rotation is
                // shared across heads.
                let freq =
                    1.0 / tensor::powf(cfg.rope_theta, (i * 2) as f32 / cfg.rot_dim() as f32);
                let a = p as f32 * freq;
                rope_cos[p * half + i] = tensor::cosf(a);
                rope_sin[p * half + i] = tensor::sinf(a);
            }
        }

        Self {
            x: vec![0.0; cfg.dim],
            // Attention writes `n_heads * head_dim` values into `xb` before
            // `wo` projects them back down, so it has to hold the wider of the
            // two. For every Llama-shaped model these are equal.
            xb: vec![0.0; cfg.dim.max(cfg.q_dim())],
            xb2: vec![0.0; cfg.dim],
            hb: vec![0.0; cfg.hidden_dim],
            hb2: vec![0.0; cfg.hidden_dim],
            q: vec![0.0; cfg.q_dim()],
            att: vec![0.0; cfg.n_heads * cap],
            logits: vec![0.0; cfg.vocab_size],
            // Only the full-attention layers get one. For a dense model that
            // is every layer and this is the allocation it always was; for
            // Qwen3.5-0.8B it is 6 of 24, and allocating all 24 would throw
            // away the entire reason the hybrid is worth running.
            key_cache: (0..cfg.n_full_layers()).map(|_| KvLayer::new(cap, kv)).collect(),
            value_cache: (0..cfg.n_full_layers()).map(|_| KvLayer::new(cap, kv)).collect(),
            kbuf: vec![0.0; kv],
            vbuf: vec![0.0; kv],
            krot: vec![0.0; cap * kv],
            rope_cos,
            rope_sin,
            linear: (0..cfg.n_linear_layers()).map(|_| LinearState::new(cfg)).collect(),
            qkv: vec![0.0; if hyb { cfg.conv_dim() } else { 0 }],
            zbuf: vec![0.0; if hyb { cfg.lin_v_dim() } else { 0 }],
            abuf: vec![0.0; if hyb { cfg.lin_v_heads } else { 0 }],
            bbuf: vec![0.0; if hyb { cfg.lin_v_heads } else { 0 }],
            core: vec![0.0; if hyb { cfg.lin_v_dim() } else { 0 }],
            dbuf: vec![0.0; if hyb { cfg.lin_v_head } else { 0 }],
            qg: vec![0.0; if hyb { 2 * cfg.q_dim() } else { 0 }],
            la: vec![0.0; super::adapter::MAX_RANK],
            router: vec![0.0; cfg.n_experts],
            moe_gu: vec![0.0; if cfg.n_experts > 0 { 2 * cfg.hidden_dim } else { 0 }],
            dim: cfg.dim,
            cap,
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
        // Bounded by what is allocated, not by the trained length. With a
        // window those differ, and `pos` is an absolute position that can be
        // far past either.
        let pos = pos.min(self.cap);
        // The number of *cached* layers, which is every layer of a dense
        // model and only the full-attention ones of a hybrid. Writing
        // `n_layers` would describe a cache that is not there.
        let cached = self.key_cache.len();
        let mut out = Vec::new();
        out.extend_from_slice(KV_MAGIC);
        out.extend_from_slice(&(cached as u32).to_le_bytes());
        out.extend_from_slice(&(kv as u32).to_le_bytes());
        out.extend_from_slice(&(pos as u32).to_le_bytes());

        // Written as f32, layer-major then position, whatever the cache holds
        // in RAM. A context is a content-addressed object that outlives the
        // in-memory layout, and one saved before the cache was quantised must
        // still load after -- so the format describes the mental state, not
        // this month's representation of it. Dequantising here costs one pass
        // over something already being copied byte by byte.
        out.try_reserve(2 * cached * pos * kv * 4).ok();
        let mut row = vec![0.0f32; kv];
        for src in [&self.key_cache, &self.value_cache] {
            for l in 0..cached {
                for slot in 0..pos {
                    src[l].read_into(slot, &mut row);
                    for v in &row {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }

        // A hybrid keeps most of its memory in the recurrence rather than in
        // the cache, and a context that saved only the cache would restore a
        // model that had forgotten three layers in four -- fluently, and with
        // no error anywhere. Appended rather than interleaved so a dense
        // export is byte-identical to what it always was.
        for ls in &self.linear {
            for v in ls.s.iter().chain(ls.conv.iter()) {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&(ls.conv_at as u32).to_le_bytes());
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
        if layers != self.key_cache.len() || kv != cfg.kv_dim() || pos > self.cap {
            return None;
        }
        let per_lin = (cfg.lin_state_len() + cfg.conv_ring_len()) * 4 + 4;
        let need = 20 + 2 * layers * pos * kv * 4 + self.linear.len() * per_lin;
        if data.len() < need {
            return None;
        }

        let mut o = 20;
        let mut row = vec![0.0f32; kv];
        for which in 0..2 {
            for l in 0..layers {
                let dst = if which == 0 { &mut self.key_cache } else { &mut self.value_cache };
                for slot in 0..pos {
                    for v in row.iter_mut() {
                        *v = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                        o += 4;
                    }
                    // Requantising a cache that was already quantised once when
                    // it was saved is lossless: the values are exactly
                    // representable at the same scale.
                    dst[l].store(slot, &row);
                }
            }
        }

        for i in 0..self.linear.len() {
            let (sl, cl) = (cfg.lin_state_len(), cfg.conv_ring_len());
            for j in 0..sl + cl {
                let v = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                o += 4;
                if j < sl {
                    self.linear[i].s[j] = v;
                } else {
                    self.linear[i].conv[j - sl] = v;
                }
            }
            self.linear[i].conv_at =
                u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]) as usize;
            o += 4;
        }
        Some(pos)
    }

    /// The final normed hidden state -- exactly what the classifier sees.
    ///
    /// Only meaningful immediately after `forward`. This is the feature vector
    /// the vocabulary extension scores against, and the one its gradient step
    /// multiplies by.
    pub fn hidden(&self) -> &[f32] {
        // Bounded, not the whole buffer: `xb` is sized for the wider of the
        // residual stream and the attention output, and on Qwen3 the tail holds
        // last layer's attention values. Handing those to the probe would add
        // 1024 stale features that look perfectly plausible.
        &self.xb[..self.dim]
    }

    pub fn bytes(&self, cfg: &Config) -> usize {
        let kv = cfg.kv_dim();
        let floats = cfg.dim * 2            // x, xb2
            + cfg.dim.max(cfg.q_dim())      // xb
            + cfg.q_dim()                   // q
            + cfg.hidden_dim * 2
            + kv * 2                        // kbuf, vbuf
            + cfg.n_heads * self.cap        // att
            + cfg.vocab_size                // logits
            + self.cap * kv                 // krot
            + self.cap * cfg.head_dim; // rope cos+sin, half each
        // The cache is int8 with an f32 scale per block, so it is no longer
        // four bytes a value -- and this number is what decides whether a
        // larger window fits, so reporting the old arithmetic would overstate
        // the cost by more than three times.
        let cache = 2 * self.key_cache.len()
            * (self.cap * kv + self.cap * kv.div_ceil(KV_BLOCK) * 4);
        // The recurrence, which does not grow with context. Reported alongside
        // the cache rather than folded into it because the whole point of the
        // hybrid is that these two scale differently.
        let recur = 4 * self.linear.len() * (cfg.lin_state_len() + cfg.conv_ring_len());
        let scratch = 4
            * (self.qkv.len()
                + self.zbuf.len()
                + self.abuf.len()
                + self.bbuf.len()
                + self.core.len()
                + self.dbuf.len()
                + self.qg.len()
                + self.router.len()
                + self.moe_gu.len());
        4 * floats + cache + recur + scratch
    }

    /// What `new` will allocate for this config, computable before anything
    /// is allocated. The heap pre-flight needs the number first; `bytes` on
    /// a live state stays the ground truth, so init compares the two once
    /// per boot -- if a future field lands in one and not the other, that is
    /// a printed warning rather than a panic on whatever machine had least
    /// room to spare.
    pub fn requirement(cfg: &Config) -> usize {
        let kv = cfg.kv_dim();
        let cap = cfg.live_cap();
        let hyb = cfg.hybrid();
        let floats = cfg.dim * 2
            + cfg.dim.max(cfg.q_dim())
            + cfg.q_dim()
            + cfg.hidden_dim * 2
            + kv * 2
            + cfg.n_heads * cap
            + cfg.vocab_size
            + cap * kv
            + cap * cfg.head_dim;
        let cache = 2 * cfg.n_full_layers()
            * (cap * kv + cap * kv.div_ceil(KV_BLOCK) * 4);
        let recur = 4 * cfg.n_linear_layers() * (cfg.lin_state_len() + cfg.conv_ring_len());
        let scratch = 4
            * ((if hyb { cfg.conv_dim() } else { 0 })
                + (if hyb { cfg.lin_v_dim() } else { 0 })
                + (if hyb { cfg.lin_v_heads } else { 0 })
                + (if hyb { cfg.lin_v_heads } else { 0 })
                + (if hyb { cfg.lin_v_dim() } else { 0 })
                + (if hyb { cfg.lin_v_head } else { 0 })
                + (if hyb { 2 * cfg.q_dim() } else { 0 })
                + cfg.n_experts
                + (if cfg.n_experts > 0 { 2 * cfg.hidden_dim } else { 0 }));
        4 * floats + cache + recur + scratch
    }
}

/// Size of the llama2.c legacy header: seven i32.
pub const HEADER_BYTES: usize = 28;

const KV_MAGIC: &[u8; 8] = b"GLADOSKV";

const GLADOS_MAGIC: &[u8; 8] = b"GLADOSM2";
/// Versions this loader understands.
///
/// v2 ends at byte 48 and describes a Llama: head size is `dim / n_heads`,
/// RMSNorm epsilon is 1e-5, no QK-Norm. v3 fills three of the sixteen bytes
/// that were already spare in the 64-byte header, so the magic is unchanged and
/// every v2 checkpoint on an ESP keeps loading untouched.
const GLADOS_VERSION_LLAMA: u32 = 2;
const GLADOS_VERSION_GENERAL: u32 = 3;
/// A second architecture whose layers are not all the same, which needs both a
/// wider header and a different body layout. v2 and v3 are untouched: their
/// fields sit at exactly the offsets they always did, and a v3 file produced
/// today is byte-identical to one produced before v4 existed.
const GLADOS_VERSION_HYBRID: u32 = 4;
const GLADOS_HEADER: usize = 64;
const GLADOS_HEADER_V4: usize = 160;
const GLADOS_BITMAP_AT: usize = 112;
const GLADOS_QUANT_I8: u32 = 1;
const GLADOS_FLAG_QK_NORM: u32 = 1 << 0;
const GLADOS_FLAG_ROPE_INTERLEAVED: u32 = 1 << 1;
const GLADOS_FLAG_ATTN_OUTPUT_GATE: u32 = 1 << 2;
const GLADOS_ARCH_DENSE: u32 = 0;
const GLADOS_ARCH_QWEN35: u32 = 1;
const GLADOS_ARCH_QWEN35_MOE: u32 = 2;
/// The epsilon inside the delta rule's L2 normalisation, which is not the
/// checkpoint's `rms_norm_eps` and is not learned.
const L2_EPS: f32 = 1e-6;

#[derive(Debug, Clone, Copy)]
pub enum LoadError {
    TooShort,
    BadHeader,
    /// The header describes more weights than the file contains, which usually
    /// means a truncated download rather than a format mismatch.
    Truncated { want: usize, have: usize },
    OutOfMemory,
    /// The format is understood and this build will not run it.
    Unsupported,
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
    /// A v4 hybrid, whose body is layer-major.
    ///
    /// The dense layout groups by tensor and then by layer, so one base offset
    /// plus `layer * stride` finds anything. A hybrid cannot: three layers in
    /// four hold `linear_attn` tensors and the fourth holds `self_attn`, so
    /// there is no single stride to multiply. The offsets are therefore
    /// recorded per layer during the load walk, which is work that had to
    /// happen anyway.
    Hybrid {
        bytes: &'static [u8],
        /// Every f32 tensor in the file, copied out. Same reason as `norms`
        /// above and rather more of them: the delta net keeps its convolution,
        /// its decay parameters and its gated norm in f32, which for
        /// Qwen3.5-0.8B is 1.09M values, 4.4 MB.
        f: Vec<f32>,
        layers: Vec<LayerOff>,
        embed: usize,
        rms_final: usize,
        wcls: usize,
    },
}

/// Where one hybrid layer's tensors are.
///
/// Byte offsets into the blob for the int8 tensors, float offsets into the
/// model's own aligned copy for the f32 ones. Unused fields stay zero: a
/// linear-attention layer has no `wq`, and reading one would be a bug this
/// struct cannot prevent, only make obvious.
#[derive(Clone, Copy, Default)]
struct LayerOff {
    rms_att: usize,
    rms_ffn: usize,
    // Gated DeltaNet.
    in_qkv: usize,
    in_z: usize,
    in_a: usize,
    in_b: usize,
    conv1d: usize,
    a_log: usize,
    dt_bias: usize,
    gate_norm: usize,
    out_proj: usize,
    // Full attention.
    q_norm: usize,
    k_norm: usize,
    wq: usize,
    wk: usize,
    wv: usize,
    wo: usize,
    // Feed-forward, dense.
    w1: usize,
    w2: usize,
    w3: usize,
    // Feed-forward, sparse. `experts` is the base of `n_experts` consecutive
    // (gate_up, down) pairs.
    router: usize,
    experts: usize,
    sh_gate: usize,
    sh_up: usize,
    sh_down: usize,
    sh_gate_w: usize,
}

/// A forward-only cursor over the body, recording where each tensor starts.
///
/// The f32 tensors are noted rather than read: their byte offsets go into
/// `f32s` in walk order, and the copy happens once at the end, after the
/// truncation check has established the bytes are all there.
struct Walk {
    p: usize,
    fp: usize,
    f32s: Vec<(usize, usize)>,
}

impl Walk {
    fn q8(&mut self, rows: usize, cols: usize) -> usize {
        let at = self.p;
        self.p += Model::q8_stride(rows, cols);
        at
    }

    fn f32(&mut self, n: usize) -> usize {
        let at = self.fp;
        self.f32s.push((self.p, n));
        self.p += n * 4;
        self.fp += n;
        at
    }
}

pub struct Model {
    pub cfg: Config,
    src: Source,
    o: Offsets,
    /// QDoRA adapters attached after load. `None` costs nothing anywhere:
    /// every hot-path check is a pointer test, and an unattached model runs
    /// the exact instruction stream it always ran.
    pub adapters: Option<super::adapter::Adapters>,
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
        Some(Self { cfg, src: Source::Flat(w), o, adapters: None })
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
            // The legacy format predates any model that separated the two, so
            // the Llama identity holds by construction here.
            head_dim: if n_heads > 0 { (dim / n_heads) as usize } else { 0 },
            vocab_size: raw_vocab.unsigned_abs() as usize,
            seq_len: seq_len as usize,
            norm_eps: 1e-5,
            qk_norm: false,
            // llama2.c's own exporter, and llama2.c rotates adjacent pairs.
            // This is the one format where interleaved is right.
            rope_interleaved: true,
            shared_classifier: raw_vocab > 0,
            // The legacy format has no field for it; llama2.c hardcodes 10000
            // and every checkpoint in that format was trained with it.
            rope_theta: 10000.0,
            attn_sinks: 0,
            // seq_len means the sum reaches capacity, so nothing is ever
            // evicted and this is off until asked for.
            attn_window: usize::MAX,
            // Nothing before Qwen3.5 rotates a prefix of the head; the two are
            // equal and every RoPE loop below is the one it always was.
            rotary_dim: if n_heads > 0 { (dim / n_heads) as usize } else { 0 },
            ..Default::default()
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
        Ok(Self { cfg, src: Source::Flat(w), o, adapters: None })
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

        let version = u32_at(8);
        if version != GLADOS_VERSION_LLAMA
            && version != GLADOS_VERSION_GENERAL
            && version != GLADOS_VERSION_HYBRID
        {
            return Err(LoadError::BadHeader);
        }
        if version >= GLADOS_VERSION_HYBRID && data.len() < GLADOS_HEADER_V4 {
            return Err(LoadError::TooShort);
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

        // v2 carried none of these; its defaults are exactly the Llama ones.
        let (head_dim, norm_eps, flags) = if version >= GLADOS_VERSION_GENERAL {
            (
                i32_at(48),
                f32::from_le_bytes([data[52], data[53], data[54], data[55]]),
                u32_at(56),
            )
        } else {
            (if n_heads > 0 { dim / n_heads } else { 0 }, 1e-5, 0)
        };
        if head_dim <= 0 || head_dim % 2 != 0 || !(norm_eps > 0.0) {
            return Err(LoadError::BadHeader);
        }

        // v4 puts everything new past byte 64, so v2 and v3 never read any of
        // it and this whole block collapses to Dense defaults for them.
        let (arch, rotary_dim, hybrid) = if version >= GLADOS_VERSION_HYBRID {
            let a = match u32_at(60) {
                // A dense model is a v3 file. Accepting arch 0 here would set
                // `hybrid()` false and send the walk below to byte 64, which
                // is 96 bytes short of where a v4 body starts -- and it would
                // find well-formed int8 there, because everything is.
                GLADOS_ARCH_DENSE => return Err(LoadError::BadHeader),
                GLADOS_ARCH_QWEN35 => Arch::Qwen35,
                GLADOS_ARCH_QWEN35_MOE => Arch::Qwen35Moe,
                _ => return Err(LoadError::BadHeader),
            };
            (a, i32_at(64), true)
        } else {
            (Arch::Dense, head_dim, false)
        };
        if rotary_dim <= 0 || rotary_dim % 2 != 0 || rotary_dim > head_dim {
            return Err(LoadError::BadHeader);
        }

        let (lin_k_head, lin_v_head, lin_k_heads, lin_v_heads, conv_kernel) = if hybrid {
            (i32_at(68), i32_at(72), i32_at(76), i32_at(80), i32_at(84))
        } else {
            (0, 0, 0, 0, 0)
        };
        let (n_experts, experts_per_tok, shared_dim) =
            if hybrid { (i32_at(88), i32_at(92), i32_at(96)) } else { (0, 0, 0) };

        let mut layer_full = [0u32; 8];
        if hybrid {
            if n_layers as usize > 8 * 32 {
                return Err(LoadError::BadHeader);
            }
            for (w, slot) in layer_full.iter_mut().enumerate() {
                *slot = u32_at(GLADOS_BITMAP_AT + w * 4);
            }
        }

        let cfg = Config {
            dim: dim as usize,
            hidden_dim: hidden_dim as usize,
            n_layers: n_layers as usize,
            n_heads: n_heads as usize,
            n_kv_heads: n_kv_heads as usize,
            head_dim: head_dim as usize,
            vocab_size: raw_vocab.unsigned_abs() as usize,
            seq_len: seq_len as usize,
            norm_eps,
            qk_norm: flags & GLADOS_FLAG_QK_NORM != 0,
            // v2 predates the flag, and every v2 file was produced by
            // convert.py from a HuggingFace checkpoint -- so `false` is not a
            // fallback here, it is the correct value for all of them.
            rope_interleaved: flags & GLADOS_FLAG_ROPE_INTERLEAVED != 0,
            shared_classifier: raw_vocab > 0,
            rope_theta,
            attn_sinks: 0,
            // seq_len means the sum reaches capacity, so nothing is ever
            // evicted and this is off until asked for.
            attn_window: usize::MAX,
            arch,
            rotary_dim: rotary_dim as usize,
            attn_output_gate: flags & GLADOS_FLAG_ATTN_OUTPUT_GATE != 0,
            lin_k_head: lin_k_head as usize,
            lin_v_head: lin_v_head as usize,
            lin_k_heads: lin_k_heads as usize,
            lin_v_heads: lin_v_heads as usize,
            conv_kernel: conv_kernel as usize,
            n_experts: n_experts as usize,
            experts_per_tok: experts_per_tok as usize,
            shared_dim: shared_dim as usize,
            layer_full,
        };
        // kv_mul() divides by n_kv_heads. `dim % n_heads` is deliberately *not*
        // checked any more: it is no longer a constraint the geometry implies,
        // and on Qwen3 it happens to hold while meaning nothing.
        if cfg.n_heads % cfg.n_kv_heads != 0 {
            return Err(LoadError::BadHeader);
        }

        if cfg.hybrid() {
            if lin_k_head <= 0
                || lin_v_head <= 0
                || lin_k_heads <= 0
                || lin_v_heads <= 0
                || conv_kernel <= 1
                || cfg.lin_v_heads % cfg.lin_k_heads != 0
            {
                return Err(LoadError::BadHeader);
            }
            // Deliberately refused rather than half-implemented. The layout
            // walk below handles a MoE body and is checked by the converter's
            // round-trip, but there is no forward pass for one here and
            // writing an unverifiable one would be the exact mistake this
            // project keeps a fixture to avoid: the smallest published MoE is
            // 35B-A3B at 71.9 GB, which cannot be read into a UEFI pool on
            // this machine, so nothing could ever run it and disagree.
            if cfg.arch == Arch::Qwen35Moe {
                return Err(LoadError::Unsupported);
            }
            return Self::load_hybrid(data, cfg);
        }

        let (d, h, l, q, kv, v) = (
            cfg.dim,
            cfg.hidden_dim,
            cfg.n_layers,
            cfg.q_dim(),
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
        let (q_norm_at, k_norm_at) = if cfg.qk_norm {
            let a = p;
            p += l * cfg.head_dim * 4;
            let b = p;
            p += l * cfg.head_dim * 4;
            (a, b)
        } else {
            (0, 0)
        };
        off.wq = p;
        p += l * Self::q8_stride(q, d);
        off.wk = p;
        p += l * Self::q8_stride(kv, d);
        off.wv = p;
        p += l * Self::q8_stride(kv, d);
        off.wo = p;
        p += l * Self::q8_stride(d, q);
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
        //
        // Layout: rms_att[L*dim], rms_ffn[L*dim], rms_final[dim], then the
        // QK-Norm pair if the model has one. Appending rather than interleaving
        // keeps every existing offset arithmetic-identical for a model without.
        let qk = if cfg.qk_norm { l * cfg.head_dim } else { 0 };
        let mut norms = Vec::new();
        norms.try_reserve_exact(2 * l * d + d + 2 * qk).map_err(|_| LoadError::OutOfMemory)?;
        for (base, count) in [(rms_att_at, l * d), (rms_ffn_at, l * d), (rms_final_at, d)] {
            for i in 0..count {
                norms.push(weights::f32_at(&data[base..], i));
            }
        }
        if cfg.qk_norm {
            for base in [q_norm_at, k_norm_at] {
                for i in 0..qk {
                    norms.push(weights::f32_at(&data[base..], i));
                }
            }
        }

        Ok(Self { cfg, src: Source::Blob { bytes: data, off, norms }, o: Offsets::default(), adapters: None })
    }

    /// Walk a v4 body and record where every tensor of every layer starts.
    ///
    /// Mirrors `convert_hybrid` in `tools/convert.py` exactly, in the same
    /// order, because the body carries no names, no shapes and no lengths. A
    /// disagreement about one dimension does not fail here -- it leaves every
    /// tensor after that point pointing at somebody else's weights, which are
    /// still perfectly well-formed int8. The truncation check at the end is
    /// the same assertion `v4.py` makes when it refuses to land anywhere but
    /// the final byte, and it is the only thing standing between a shape bug
    /// and a model that runs and is wrong.
    fn load_hybrid(data: &'static [u8], cfg: Config) -> Result<Self, LoadError> {
        let c = &cfg;
        let (d, h, v) = (c.dim, c.hidden_dim, c.vocab_size);
        let (q, kv) = (c.q_dim(), c.kv_dim());
        let (vdim, conv_dim) = (c.lin_v_dim(), c.conv_dim());
        let moe = c.arch == Arch::Qwen35Moe;

        let mut w = Walk { p: GLADOS_HEADER_V4, fp: 0, f32s: Vec::new() };
        let embed = w.q8(v, d);

        let mut layers: Vec<LayerOff> = Vec::new();
        layers.try_reserve_exact(c.n_layers).map_err(|_| LoadError::OutOfMemory)?;
        for l in 0..c.n_layers {
            let mut o = LayerOff { rms_att: w.f32(d), ..Default::default() };

            if c.layer_kind(l) == LayerKind::Linear {
                o.in_qkv = w.q8(conv_dim, d);
                o.in_z = w.q8(vdim, d);
                o.in_a = w.f32(c.lin_v_heads * d);
                o.in_b = w.f32(c.lin_v_heads * d);
                o.conv1d = w.f32(conv_dim * c.conv_kernel);
                o.a_log = w.f32(c.lin_v_heads);
                o.dt_bias = w.f32(c.lin_v_heads);
                o.gate_norm = w.f32(c.lin_v_head);
                o.out_proj = w.q8(d, vdim);
            } else {
                o.q_norm = w.f32(c.head_dim);
                o.k_norm = w.f32(c.head_dim);
                // Twice as wide as the query: it emits the output gate too.
                o.wq = w.q8(2 * q, d);
                o.wk = w.q8(kv, d);
                o.wv = w.q8(kv, d);
                o.wo = w.q8(d, q);
            }

            o.rms_ffn = w.f32(d);
            if moe {
                o.router = w.f32(c.n_experts * d);
                o.experts = w.p;
                for _ in 0..c.n_experts {
                    w.q8(2 * h, d);
                    w.q8(d, h);
                }
                o.sh_gate = w.q8(c.shared_dim, d);
                o.sh_up = w.q8(c.shared_dim, d);
                o.sh_down = w.q8(d, c.shared_dim);
                o.sh_gate_w = w.f32(d);
            } else {
                o.w1 = w.q8(h, d);
                o.w2 = w.q8(d, h);
                o.w3 = w.q8(h, d);
            }
            layers.push(o);
        }

        let rms_final = w.f32(d);
        let wcls = w.p;
        if !c.shared_classifier {
            w.q8(v, d);
        }

        // Landing anywhere but the last byte means this walk and the writer's
        // disagree about a shape. Short is as bad as long: every tensor after
        // the disagreement points at somebody else's weights, which are still
        // perfectly well-formed int8 and produce a model that runs and is
        // wrong. `tools/v4.py` makes exactly this assertion on the way back
        // in, and it is the only cheap check there is.
        if data.len() != w.p {
            return Err(LoadError::Truncated { want: w.p, have: data.len() });
        }

        let mut f = Vec::new();
        f.try_reserve_exact(w.fp).map_err(|_| LoadError::OutOfMemory)?;
        for (base, count) in &w.f32s {
            for i in 0..*count {
                f.push(weights::f32_at(&data[*base..], i));
            }
        }

        Ok(Self {
            cfg,
            src: Source::Hybrid { bytes: data, f, layers, embed, rms_final, wcls },
            o: Offsets::default(),
            adapters: None,
        })
    }

    pub fn weight_bytes(&self) -> usize {
        match &self.src {
            Source::Flat(w) => w.len() * 4,
            Source::Blob { bytes, norms, .. } => bytes.len() + norms.len() * 4,
            Source::Hybrid { bytes, f, .. } => bytes.len() + f.len() * 4,
        }
    }

    pub fn is_quantised(&self) -> bool {
        matches!(self.src, Source::Blob { .. } | Source::Hybrid { .. })
    }

    /// One f32 tensor out of a hybrid's aligned copy.
    fn hf(&self, off: usize, n: usize) -> &[f32] {
        match &self.src {
            Source::Hybrid { f, .. } => &f[off..off + n],
            _ => &[],
        }
    }

    /// One int8 tensor out of a hybrid's blob, by absolute byte offset.
    fn hq(&self, off: usize, rows: usize, cols: usize) -> Mat<'_> {
        match &self.src {
            Source::Hybrid { bytes, .. } => Self::q8(bytes, off, rows, cols),
            _ => Mat::F32 { data: &[], rows: 0, cols: 0 },
        }
    }

    fn lo(&self, l: usize) -> LayerOff {
        match &self.src {
            Source::Hybrid { layers, .. } => layers[l],
            _ => LayerOff::default(),
        }
    }

    #[inline]
    fn slice(&self, off: usize, len: usize) -> &[f32] {
        match &self.src {
            Source::Flat(w) => &w[off..off + len],
            // Only the Flat path indexes by float offset; reaching here means
            // a caller was not converted to the Mat accessors.
            Source::Blob { .. } | Source::Hybrid { .. } => &[],
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
            // A hybrid body is layer-major with no single stride to multiply,
            // so nothing routes through here; `hq` takes an offset directly.
            Source::Hybrid { .. } => Mat::F32 { data: &[], rows: 0, cols: 0 },
        }
    }

    fn blob_off(&self) -> ByteOffsets {
        match &self.src {
            Source::Blob { off, .. } => *off,
            Source::Flat(_) | Source::Hybrid { .. } => ByteOffsets::default(),
        }
    }

    fn wq(&self, l: usize) -> Mat<'_> {
        let (d, q) = (self.cfg.dim, self.cfg.q_dim());
        self.mat(self.o.wq, self.blob_off().wq, l, q, d)
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
        let (d, q) = (self.cfg.dim, self.cfg.q_dim());
        self.mat(self.o.wo, self.blob_off().wo, l, d, q)
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
    /// One row of the embedding table, dequantised.
    ///
    /// The forward reads this into `State.x` as its first act, so it is what
    /// layer 0 of a `Tape` must contain -- the one tape entry whose value is
    /// known without trusting the forward, and therefore the one that can
    /// catch a wrong layer stride.
    pub fn embed_row(&self, token: usize, out: &mut [f32]) {
        let tok = token.min(self.cfg.vocab_size - 1);
        self.embed_mat().row_into(tok, out);
    }

    fn embed_mat(&self) -> Mat<'_> {
        let (d, v) = (self.cfg.dim, self.cfg.vocab_size);
        match &self.src {
            Source::Hybrid { embed, .. } => self.hq(*embed, v, d),
            _ => self.mat(self.o.token_embedding, self.blob_off().embed, 0, v, d),
        }
    }
    pub(crate) fn classifier(&self) -> Mat<'_> {
        let (d, v) = (self.cfg.dim, self.cfg.vocab_size);
        if self.cfg.shared_classifier {
            return self.embed_mat();
        }
        match &self.src {
            Source::Hybrid { wcls, .. } => self.hq(*wcls, v, d),
            _ => self.mat(self.o.wcls, self.blob_off().wcls, 0, v, d),
        }
    }

    fn rms_att_w(&self, l: usize) -> &[f32] {
        let d = self.cfg.dim;
        match &self.src {
            Source::Flat(_) => self.slice(self.o.rms_att + l * d, d),
            Source::Blob { norms, .. } => &norms[l * d..(l + 1) * d],
            Source::Hybrid { .. } => self.hf(self.lo(l).rms_att, d),
        }
    }
    fn rms_ffn_w(&self, l: usize) -> &[f32] {
        let (d, n) = (self.cfg.dim, self.cfg.n_layers);
        match &self.src {
            Source::Flat(_) => self.slice(self.o.rms_ffn + l * d, d),
            Source::Blob { norms, .. } => &norms[(n + l) * d..(n + l + 1) * d],
            Source::Hybrid { .. } => self.hf(self.lo(l).rms_ffn, d),
        }
    }
    fn rms_final_w(&self) -> &[f32] {
        let (d, n) = (self.cfg.dim, self.cfg.n_layers);
        match &self.src {
            Source::Flat(_) => self.slice(self.o.rms_final, d),
            Source::Blob { norms, .. } => &norms[2 * n * d..2 * n * d + d],
            Source::Hybrid { rms_final, .. } => self.hf(*rms_final, d),
        }
    }

    /// QK-Norm weights, one vector of `head_dim` shared by every head.
    ///
    /// In the blob they live past the three ordinary norm groups; `which` is 0
    /// for the query and 1 for the key. Callers must check `cfg.qk_norm` --
    /// there is nothing here for a model without them.
    fn qk_norm_w(&self, which: usize, l: usize) -> &[f32] {
        let c = &self.cfg;
        let (d, n, hd) = (c.dim, c.n_layers, c.head_dim);
        match &self.src {
            Source::Flat(_) => {
                let base = if which == 0 { self.o.q_norm } else { self.o.k_norm };
                self.slice(base + l * hd, hd)
            }
            Source::Blob { norms, .. } => {
                let base = 2 * n * d + d + (which * n + l) * hd;
                &norms[base..base + hd]
            }
            Source::Hybrid { .. } => {
                let o = self.lo(l);
                self.hf(if which == 0 { o.q_norm } else { o.k_norm }, hd)
            }
        }
    }
    fn q_norm_w(&self, l: usize) -> &[f32] {
        self.qk_norm_w(0, l)
    }
    fn k_norm_w(&self, l: usize) -> &[f32] {
        self.qk_norm_w(1, l)
    }

    /// One decode step: token at position `pos` in, logits out.
    pub fn forward(&self, s: &mut State, token: usize, pos: usize) {
        if self.cfg.hybrid() {
            self.forward_hybrid(s, token, pos)
        } else {
            self.forward_dense(s, token, pos, None)
        }
    }

    /// A forward that keeps what a backward pass needs.
    ///
    /// The same function, given somewhere to write. Not a second forward:
    /// two implementations of a transformer that are supposed to agree do not
    /// stay agreeing, and the one that drifts is the one nobody decodes with,
    /// so the drift shows up as a training run that quietly optimises a
    /// slightly different model than the one being served.
    ///
    /// Hybrids are refused, matching `attach_adapters` -- their gated,
    /// partially-rotated projections need a verified backward of their own
    /// rather than an alias to the dense sites.
    pub fn forward_taped(
        &self,
        s: &mut State,
        token: usize,
        pos: usize,
        tape: &mut Tape,
    ) -> bool {
        if self.cfg.hybrid() {
            return false;
        }
        self.forward_dense(s, token, pos, Some(tape));
        true
    }

    fn forward_dense(&self, s: &mut State, token: usize, pos: usize, tape: Option<&mut Tape>) {
        let mut tape = tape;
        let c = &self.cfg;
        let kv_dim = c.kv_dim();
        let kv_mul = c.kv_mul();
        let head_size = c.head_size();
        let eps = c.norm_eps;

        let cap = s.cap;
        let w = Window::new(c, cap, pos);
        let live = w.live;
        let here = w.slot_of(live - 1);

        // Embedding lookup is a row fetch, not a matmul against a one-hot
        // vector. When the table is quantised the row is dequantised on the
        // way out.
        let tok = token.min(c.vocab_size - 1);
        self.embed_mat().row_into(tok, &mut s.x);

        for l in 0..c.n_layers {
            // The residual stream as it enters this layer. Everything the
            // layer goes on to compute is a function of this and the frozen
            // weights, which is why nothing else is kept.
            if let Some(t) = tape.as_mut() {
                t.record_layer(l, pos, &s.x);
            }

            // --- attention ---
            tensor::rmsnorm_eps(&mut s.xb[..c.dim], &s.x, self.rms_att_w(l), eps);

            // Keys and values go through a staging buffer rather than into the
            // cache directly: quantisation needs a whole block before it knows
            // the scale, and QK-Norm has to run on the f32 projection anyway.
            // Each site runs through its QDoRA wrapper when adapted; the
            // mem::take dance splits State's borrows so the frozen-weight
            // matvec can read `xb` while rank scratch is mutated.
            match self.adapters.as_ref().and_then(|a| a.qkv[l][0].as_ref()) {
                Some(d) => {
                    let xb = &s.xb;
                    let mut la = core::mem::take(&mut s.la);
                    self.wq(l).wrap_matvec(d, xb, &mut la, &mut s.q);
                    s.la = la;
                }
                None => self.wq(l).matvec(&mut s.q, &s.xb),
            }
            match self.adapters.as_ref().and_then(|a| a.qkv[l][1].as_ref()) {
                Some(d) => {
                    let xb = &s.xb;
                    let mut la = core::mem::take(&mut s.la);
                    self.wk(l).wrap_matvec(d, xb, &mut la, &mut s.kbuf);
                    s.la = la;
                }
                None => self.wk(l).matvec(&mut s.kbuf, &s.xb),
            }
            match self.adapters.as_ref().and_then(|a| a.qkv[l][2].as_ref()) {
                Some(d) => {
                    let xb = &s.xb;
                    let mut la = core::mem::take(&mut s.la);
                    self.wv(l).wrap_matvec(d, xb, &mut la, &mut s.vbuf);
                    s.la = la;
                }
                None => self.wv(l).matvec(&mut s.vbuf, &s.xb),
            }

            // QK-Norm, before RoPE and before anything is cached.
            //
            // Order is the whole of it: Qwen3 normalises the raw projection,
            // then rotates. Normalising after RoPE would rescale a vector whose
            // length already encodes position, and normalising the key on the
            // way *out* of the cache would redo the same work for every token
            // that ever attends to it. Keys are therefore stored normed and
            // unrotated -- normalisation is a property of the key, rotation is
            // a property of where it currently sits.
            if c.qk_norm {
                let qn = self.q_norm_w(l);
                for h in 0..c.n_heads {
                    let o = h * head_size;
                    tensor::rmsnorm_inplace(&mut s.q[o..o + head_size], qn, eps);
                }
                let kn = self.k_norm_w(l);
                for h in 0..c.n_kv_heads {
                    let o = h * head_size;
                    tensor::rmsnorm_inplace(&mut s.kbuf[o..o + head_size], kn, eps);
                }
            }

            // Normed and unrotated for keys, raw for values -- the same point
            // `tools/reference.py` applies its round-trip, so the error the
            // oracle measures is the error this introduces.
            s.key_cache[l].store(here, &s.kbuf);
            s.value_cache[l].store(here, &s.vbuf);

            self.rope_q(s, &w);
            self.rope_keys(s, &w, l, kv_dim, w.live);
            self.attend(s, &w, l, kv_dim, kv_mul);

            self.wo(l).matvec(&mut s.xb2, &s.xb);
            tensor::add_into(&mut s.x, &s.xb2);

            // --- feed forward ---
            tensor::rmsnorm_eps(&mut s.xb[..c.dim], &s.x, self.rms_ffn_w(l), eps);
            self.w1(l).matvec(&mut s.hb, &s.xb);
            self.w3(l).matvec(&mut s.hb2, &s.xb);
            tensor::swiglu(&mut s.hb, &s.hb2);
            self.w2(l).matvec(&mut s.xb, &s.hb);
            tensor::add_into(&mut s.x, &s.xb);
        }

        // Normalise into xb rather than cloning x into a temporary. The clone
        // was a heap allocation on every single token, in the one loop that
        // runs most often.
        tensor::rmsnorm_eps(&mut s.xb[..c.dim], &s.x, self.rms_final_w(), eps);
        // Where the loss enters. Recorded after the final norm and before the
        // classifier, which is exactly the vector the classifier's own
        // gradient is taken against -- and the same vector `train.rs` already
        // caches as a feature, which is why classifier-only training needed no
        // tape at all.
        if let Some(t) = tape.as_mut() {
            // Both: the stream the final norm consumed, and the normed result
            // the classifier consumed. The first is the norm's own adjoint
            // input; the second is where the loss enters.
            t.record_layer(c.n_layers, pos, &s.x);
            t.record_final(pos, &s.xb);
        }
        match self.adapters.as_ref().and_then(|a| a.cls.as_ref()) {
            Some(d) => {
                let xb = &s.xb;
                let mut la = core::mem::take(&mut s.la);
                self.classifier().wrap_matvec(d, xb, &mut la, &mut s.logits);
                s.la = la;
            }
            None => self.classifier().matvec(&mut s.logits, &s.xb),
        }
    }

    /// Gradients for every adapted site, from a loss on one position's logits.
    ///
    /// The layer walk. `backward.rs` has held the adjoints since before any of
    /// this -- attention, rmsnorm, swiglu, rope, both transpose matvecs, all
    /// selftested -- and `grep` found exactly one caller, its own selftest.
    /// This is what calls them.
    ///
    /// ### What is recomputed, and why that is the cheap direction
    ///
    /// The tape holds only the residual stream entering each layer. Everything
    /// else this needs -- the normed input, q, k, v, the attention output, both
    /// FFN branches -- is recomputed here from that stream and the frozen
    /// weights. It costs about one extra forward and saves thirteen times the
    /// memory; `Tape`'s own comment has the arithmetic.
    ///
    /// ### The shape of the sequence
    ///
    /// `attention_backward` takes a whole head's sequence at once and handles
    /// causality itself, accumulating dq, dk and dv across every position in
    /// one call. That is why this walks layers rather than positions: the
    /// cross-position coupling -- position t's query pulling gradient into
    /// position j's key, for every j <= t -- lives inside that kernel instead
    /// of in a hand-rolled reverse loop here.
    ///
    /// ### What it refuses
    ///
    /// A windowed cache. Once eviction has happened a live index no longer
    /// equals a position, keys are rotated by where they now sit rather than
    /// where they were, and the recompute below would silently rotate by the
    /// wrong angle. Training sequences are short and this is checked rather
    /// than assumed. Hybrids too, matching `attach_adapters`.
    pub fn backward(
        &self,
        tape: &Tape,
        glogits: &[f32],
        at: usize,
        g: &mut Grads,
    ) -> bool {
        let c = &self.cfg;
        if c.hybrid() || c.streams() {
            return false;
        }
        let Some(ad) = self.adapters.as_ref() else { return false };
        let n = tape.filled();
        if n == 0 || at >= n || tape.seq < n {
            return false;
        }
        let d = c.dim;
        let hs = c.head_size();
        let half = c.rot_dim() / 2;
        let qd = c.q_dim();
        let kvd = c.kv_dim();
        let kv_mul = c.kv_mul();
        let eps = c.norm_eps;
        let scale = 1.0 / tensor::sqrtf(hs as f32);

        // dL/dx for the stream leaving the last layer, per position. Only the
        // scored position has a loss; the rest earn gradient purely through
        // attention, which is exactly the coupling this walk exists to carry.
        let mut gx = vec![0.0f32; n * d];

        // --- the classifier, and the final norm ---------------------------
        {
            let Some(xb) = tape.final_normed(at) else { return false };
            let cls = self.classifier();
            let mut gxb = vec![0.0f32; d];
            match ad.cls.as_ref() {
                Some(dora) => {
                    let mut ax = vec![0.0f32; dora.r];
                    let mut base = vec![0.0f32; c.vocab_size];
                    cls.matvec(&mut base, xb);
                    // ax is A.x, which `backward` wants alongside the frozen
                    // pre-activation.
                    for j in 0..dora.r {
                        let row = &dora.a[j * d..(j + 1) * d];
                        ax[j] = row.iter().zip(xb.iter()).map(|(a, b)| a * b).sum();
                    }
                    dora.backward(
                        &cls, xb, &ax, &base, glogits,
                        &mut g.cls.ga, &mut g.cls.gb, &mut g.cls.dm,
                    );
                    dora.backward_x(&cls, glogits, &mut gxb);
                }
                None => cls.wt_matvec(&mut gxb, glogits),
            }
            let Some(xin) = tape.entering(c.n_layers, at) else { return false };
            let mut gout = vec![0.0f32; d];
            super::backward::rmsnorm_backward(&mut gout, &gxb, xin, self.rms_final_w(), eps);
            gx[at * d..(at + 1) * d].copy_from_slice(&gout);
        }

        // --- layers, last to first ----------------------------------------
        let mut xb1 = vec![0.0f32; n * d];
        let mut qs = vec![0.0f32; n * qd];
        let mut ks = vec![0.0f32; n * kvd];
        let mut vs = vec![0.0f32; n * kvd];
        let mut xmid = vec![0.0f32; n * d];
        let mut xb3 = vec![0.0f32; n * d];
        let mut hb = vec![0.0f32; n * c.hidden_dim];
        let mut hb2 = vec![0.0f32; n * c.hidden_dim];
        let mut attn = vec![0.0f32; n * qd];

        for l in (0..c.n_layers).rev() {
            // ---- recompute this layer over the whole sequence ------------
            for t in 0..n {
                let Some(xin) = tape.entering(l, t) else { return false };
                let o = t * d;
                tensor::rmsnorm_eps(&mut xb1[o..o + d], xin, self.rms_att_w(l), eps);

                let x1 = &xb1[o..o + d].to_vec();
                self.site_forward(ad, l, 0, x1, &mut qs[t * qd..(t + 1) * qd]);
                self.site_forward(ad, l, 1, x1, &mut ks[t * kvd..(t + 1) * kvd]);
                self.site_forward(ad, l, 2, x1, &mut vs[t * kvd..(t + 1) * kvd]);

                if c.qk_norm {
                    let qn = self.q_norm_w(l);
                    for h in 0..c.n_heads {
                        let b = t * qd + h * hs;
                        tensor::rmsnorm_inplace(&mut qs[b..b + hs], qn, eps);
                    }
                    let kn = self.k_norm_w(l);
                    for h in 0..c.n_kv_heads {
                        let b = t * kvd + h * hs;
                        tensor::rmsnorm_inplace(&mut ks[b..b + hs], kn, eps);
                    }
                }
                // Rotate by the position, which without a window is the live
                // index. Both q and k, each by its own t.
                self.rope_span(&mut qs[t * qd..(t + 1) * qd], c.n_heads, t, half, hs);
                self.rope_span(&mut ks[t * kvd..(t + 1) * kvd], c.n_kv_heads, t, half, hs);
            }

            // attention forward, straight from the recomputed spans
            for t in 0..n {
                for h in 0..c.n_heads {
                    let qo = t * qd + h * hs;
                    let hoff = (h / kv_mul) * hs;
                    let mut p = vec![0.0f32; t + 1];
                    for (j, pj) in p.iter_mut().enumerate() {
                        let ko = j * kvd + hoff;
                        *pj = scale
                            * qs[qo..qo + hs]
                                .iter()
                                .zip(ks[ko..ko + hs].iter())
                                .map(|(a, b)| a * b)
                                .sum::<f32>();
                    }
                    tensor::softmax(&mut p);
                    for i in 0..hs {
                        attn[qo + i] = 0.0;
                    }
                    for (j, pj) in p.iter().enumerate() {
                        let vo = j * kvd + hoff;
                        for i in 0..hs {
                            attn[qo + i] += pj * vs[vo + i];
                        }
                    }
                }
            }

            for t in 0..n {
                let o = t * d;
                let Some(xin) = tape.entering(l, t) else { return false };
                let mut xb2 = vec![0.0f32; d];
                self.wo(l).matvec(&mut xb2, &attn[t * qd..(t + 1) * qd]);
                for i in 0..d {
                    xmid[o + i] = xin[i] + xb2[i];
                }
                let mid = xmid[o..o + d].to_vec();
                tensor::rmsnorm_eps(&mut xb3[o..o + d], &mid, self.rms_ffn_w(l), eps);
                let x3 = xb3[o..o + d].to_vec();
                self.w1(l).matvec(&mut hb[t * c.hidden_dim..(t + 1) * c.hidden_dim], &x3);
                self.w3(l).matvec(&mut hb2[t * c.hidden_dim..(t + 1) * c.hidden_dim], &x3);
            }

            // ---- backward through this layer -----------------------------
            let mut gattn = vec![0.0f32; n * qd];
            let mut gxmid = vec![0.0f32; n * d];
            for t in 0..n {
                let o = t * d;
                let ho = t * c.hidden_dim;
                let gout = gx[o..o + d].to_vec();

                // FFN. `swiglu` overwrites its first argument in the forward,
                // so the pre-activation is recomputed above and used here
                // rather than read back out of a buffer that no longer holds
                // it.
                let mut h = hb[ho..ho + c.hidden_dim].to_vec();
                tensor::swiglu(&mut h, &hb2[ho..ho + c.hidden_dim]);
                let mut gh = vec![0.0f32; c.hidden_dim];
                self.w2(l).wt_matvec(&mut gh, &gout);
                let mut gu = vec![0.0f32; c.hidden_dim];
                let mut gv = vec![0.0f32; c.hidden_dim];
                super::backward::swiglu_backward(
                    &mut gu, &mut gv, &gh,
                    &hb[ho..ho + c.hidden_dim], &hb2[ho..ho + c.hidden_dim],
                );
                let mut gx3 = vec![0.0f32; d];
                let mut tmp = vec![0.0f32; d];
                self.w1(l).wt_matvec(&mut gx3, &gu);
                self.w3(l).wt_matvec(&mut tmp, &gv);
                for i in 0..d {
                    gx3[i] += tmp[i];
                }
                let mut gmid = vec![0.0f32; d];
                super::backward::rmsnorm_backward(
                    &mut gmid, &gx3, &xmid[o..o + d], self.rms_ffn_w(l), eps,
                );
                // The residual carries the outgoing gradient past the FFN.
                for i in 0..d {
                    gxmid[o + i] = gmid[i] + gout[i];
                }
                let mut ga = vec![0.0f32; qd];
                self.wo(l).wt_matvec(&mut ga, &gxmid[o..o + d]);
                gattn[t * qd..(t + 1) * qd].copy_from_slice(&ga);
            }

            // Attention, per head, whole sequence at once.
            let mut gq = vec![0.0f32; n * qd];
            let mut gk = vec![0.0f32; n * kvd];
            let mut gv2 = vec![0.0f32; n * kvd];
            for h in 0..c.n_heads {
                let hoff = (h / kv_mul) * hs;
                // `attention_backward` wants contiguous per-head sequences.
                let mut qh = vec![0.0f32; n * hs];
                let mut kh = vec![0.0f32; n * hs];
                let mut vh = vec![0.0f32; n * hs];
                let mut dy = vec![0.0f32; n * hs];
                for t in 0..n {
                    qh[t * hs..(t + 1) * hs]
                        .copy_from_slice(&qs[t * qd + h * hs..t * qd + h * hs + hs]);
                    kh[t * hs..(t + 1) * hs]
                        .copy_from_slice(&ks[t * kvd + hoff..t * kvd + hoff + hs]);
                    vh[t * hs..(t + 1) * hs]
                        .copy_from_slice(&vs[t * kvd + hoff..t * kvd + hoff + hs]);
                    dy[t * hs..(t + 1) * hs]
                        .copy_from_slice(&gattn[t * qd + h * hs..t * qd + h * hs + hs]);
                }
                let mut dq = vec![0.0f32; n * hs];
                let mut dk = vec![0.0f32; n * hs];
                let mut dv = vec![0.0f32; n * hs];
                super::backward::attention_backward(
                    &mut dq, &mut dk, &mut dv, &dy, &qh, &kh, &vh, n, hs, scale,
                );
                // Several query heads share one kv head under GQA, so k and v
                // gradients accumulate rather than assign.
                for t in 0..n {
                    for i in 0..hs {
                        gq[t * qd + h * hs + i] += dq[t * hs + i];
                        gk[t * kvd + hoff + i] += dk[t * hs + i];
                        gv2[t * kvd + hoff + i] += dv[t * hs + i];
                    }
                }
            }

            // ---- back through rope, qk-norm, and the three sites ---------
            for t in 0..n {
                let o = t * d;
                self.rope_span_backward(&mut gq[t * qd..(t + 1) * qd], c.n_heads, t, half, hs);
                self.rope_span_backward(&mut gk[t * kvd..(t + 1) * kvd], c.n_kv_heads, t, half, hs);

                if c.qk_norm {
                    // The norm's adjoint needs its *input*, which rope has not
                    // touched -- so it is recomputed from the pre-rope
                    // projection rather than read from the rotated buffer.
                    let x1 = xb1[o..o + d].to_vec();
                    let mut qraw = vec![0.0f32; qd];
                    let mut kraw = vec![0.0f32; kvd];
                    self.site_forward(ad, l, 0, &x1, &mut qraw);
                    self.site_forward(ad, l, 1, &x1, &mut kraw);
                    let qn = self.q_norm_w(l);
                    for h in 0..c.n_heads {
                        let b = h * hs;
                        let mut out = vec![0.0f32; hs];
                        super::backward::rmsnorm_backward(
                            &mut out, &gq[t * qd + b..t * qd + b + hs],
                            &qraw[b..b + hs], qn, eps,
                        );
                        gq[t * qd + b..t * qd + b + hs].copy_from_slice(&out);
                    }
                    let kn = self.k_norm_w(l);
                    for h in 0..c.n_kv_heads {
                        let b = h * hs;
                        let mut out = vec![0.0f32; hs];
                        super::backward::rmsnorm_backward(
                            &mut out, &gk[t * kvd + b..t * kvd + b + hs],
                            &kraw[b..b + hs], kn, eps,
                        );
                        gk[t * kvd + b..t * kvd + b + hs].copy_from_slice(&out);
                    }
                }

                let x1 = xb1[o..o + d].to_vec();
                let mut gxb1 = vec![0.0f32; d];
                self.site_backward(ad, l, 0, &x1, &gq[t * qd..(t + 1) * qd], &mut g.qkv[l][0], &mut gxb1);
                self.site_backward(ad, l, 1, &x1, &gk[t * kvd..(t + 1) * kvd], &mut g.qkv[l][1], &mut gxb1);
                self.site_backward(ad, l, 2, &x1, &gv2[t * kvd..(t + 1) * kvd], &mut g.qkv[l][2], &mut gxb1);

                let Some(xin) = tape.entering(l, t) else { return false };
                let mut gin = vec![0.0f32; d];
                super::backward::rmsnorm_backward(&mut gin, &gxb1, xin, self.rms_att_w(l), eps);
                // The other residual: the stream entering this layer also went
                // straight past the attention block.
                for i in 0..d {
                    gx[o + i] = gin[i] + gxmid[o + i];
                }
            }
        }
        true
    }

    /// One q/k/v projection, frozen or adapted, into `out`.
    fn site_forward(&self, ad: &super::adapter::Adapters, l: usize, which: usize, x: &[f32], out: &mut [f32]) {
        let w = match which {
            0 => self.wq(l),
            1 => self.wk(l),
            _ => self.wv(l),
        };
        w.matvec(out, x);
        if let Some(dora) = ad.qkv[l][which].as_ref() {
            let mut ax = vec![0.0f32; dora.r];
            dora.apply(out, x, &mut ax);
        }
    }

    /// Parameter gradients for one site, and its contribution to the input.
    fn site_backward(
        &self,
        ad: &super::adapter::Adapters,
        l: usize,
        which: usize,
        x: &[f32],
        gy: &[f32],
        into: &mut SiteGrad,
        gx: &mut [f32],
    ) {
        let w = match which {
            0 => self.wq(l),
            1 => self.wk(l),
            _ => self.wv(l),
        };
        match ad.qkv[l][which].as_ref() {
            Some(dora) => {
                let mut base = vec![0.0f32; gy.len()];
                w.matvec(&mut base, x);
                let mut ax = vec![0.0f32; dora.r];
                let k = x.len();
                for j in 0..dora.r {
                    let row = &dora.a[j * k..(j + 1) * k];
                    ax[j] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
                dora.backward(&w, x, &ax, &base, gy, &mut into.ga, &mut into.gb, &mut into.dm);
                dora.backward_x(&w, gy, gx);
            }
            None => {
                // Unadapted, but gradient still has to pass through it to
                // reach the layers below.
                let mut tmp = vec![0.0f32; x.len()];
                w.wt_matvec(&mut tmp, gy);
                for (i, v) in tmp.iter().enumerate() {
                    gx[i] += v;
                }
            }
        }
    }

    /// The frozen weight behind one q/k/v site, for anything that has to seed
    /// or differentiate against it.
    pub fn frozen_site(&self, l: usize, which: usize) -> Mat<'_> {
        match which {
            0 => self.wq(l),
            1 => self.wk(l),
            _ => self.wv(l),
        }
    }

    pub fn frozen_cls(&self) -> Mat<'_> {
        self.classifier()
    }

    /// The rotation for one (position, pair), computed rather than looked up.
    ///
    /// `State` precomputes a table, and the backward walk has no `State` --
    /// it works from a tape. Recomputing keeps the walk free of a borrow it
    /// does not otherwise need, and the arithmetic is copied from the table's
    /// own construction so the two cannot disagree about the exponent.
    fn rope_at(&self, pos: usize, pair: usize) -> (f32, f32) {
        let freq = 1.0
            / tensor::powf(self.cfg.rope_theta, (pair * 2) as f32 / self.cfg.rot_dim() as f32);
        let a = pos as f32 * freq;
        (tensor::cosf(a), tensor::sinf(a))
    }

    /// Rotate a span of heads for one position.
    fn rope_span(&self, row: &mut [f32], heads: usize, pos: usize, half: usize, hs: usize) {
        let c = &self.cfg;
        for h in 0..heads {
            let base = h * hs;
            for p in 0..half {
                let (fcr, fci) = self.rope_at(pos, p);
                let (i, j) = if c.rope_interleaved {
                    (base + 2 * p, base + 2 * p + 1)
                } else {
                    (base + p, base + p + half)
                };
                let (a, b) = (row[i], row[j]);
                row[i] = a * fcr - b * fci;
                row[j] = a * fci + b * fcr;
            }
        }
    }

    /// Its adjoint: the same pairing, rotated the other way.
    fn rope_span_backward(&self, row: &mut [f32], heads: usize, pos: usize, half: usize, hs: usize) {
        let c = &self.cfg;
        for h in 0..heads {
            let base = h * hs;
            for p in 0..half {
                let (fcr, fci) = self.rope_at(pos, p);
                let (i, j) = if c.rope_interleaved {
                    (base + 2 * p, base + 2 * p + 1)
                } else {
                    (base + p, base + p + half)
                };
                let (a, b) = super::backward::rope_pair_backward(row[i], row[j], fcr, fci);
                row[i] = a;
                row[j] = b;
            }
        }
    }

    /// Attach QDoRA adapters and seed their DoRA magnitudes against the
    /// frozen weights. One pass over every adapted weight, so this is
    /// attachment-cadence work; afterwards the per-token path pays only the
    /// cached scales and the low-rank branch.
    ///
    /// Hybrid models are refused for now: their gated, partially-rotated
    /// projections deserve a verified backward pass of their own rather than
    /// a rushed alias to the dense sites.
    pub fn attach_adapters(
        &mut self,
        mut ad: super::adapter::Adapters,
    ) -> Result<(), &'static str> {
        if self.cfg.hybrid() {
            return Err("hybrid adapters are not supported yet");
        }
        {
            let me = &*self;
            ad.refresh_all(
                me.cfg.n_layers,
                |l| me.wq(l),
                |l| me.wk(l),
                |l| me.wv(l),
                || me.classifier(),
                true,
            );
        }
        self.adapters = Some(ad);
        Ok(())
    }

    /// Attach without the seeding pass over every adapted row.
    ///
    /// An unseeded row is already the identity -- `s` starts at 1.0 and the
    /// low-rank branch starts at zero -- so this changes no output. What the
    /// seeding pass adds is `m` set to the frozen row norm, which only
    /// matters for a row that is about to receive a gradient, and the caller
    /// that knows which rows those are can seed them itself with
    /// `Dora::refresh_rows`.
    ///
    /// That distinction is the difference between a classifier fine-tune
    /// being possible in this kernel and not: seeding 151,936 rows is a
    /// dequant pass over 155 MB, and the decision layer reaches a few dozen
    /// of them.
    pub fn attach_adapters_unseeded(
        &mut self,
        ad: super::adapter::Adapters,
    ) -> Result<(), &'static str> {
        if self.cfg.hybrid() {
            return Err("hybrid adapters are not supported yet");
        }
        self.adapters = Some(ad);
        Ok(())
    }

    /// Write every attached adapter to a namespace path.
    ///
    /// The blob is the adapter alone. Reloading it against a different
    /// checkpoint is refused rather than reinterpreted, and the frozen
    /// weights are neither read nor written here.
    pub fn save_adapters(&self, path: &str) -> Option<usize> {
        let blob = self.adapters.as_ref()?.to_blob();
        let n = blob.len();
        if crate::sysbox::write_blob(path, blob) {
            Some(n)
        } else {
            None
        }
    }

    /// Read an adapter back and attach it.
    ///
    /// Shapes are checked against this model's config before anything is
    /// built: an adapter trained for another geometry is an error, not a
    /// reshape, because the alternative is a model that loads, runs, stays
    /// numerically well-behaved and means something else.
    ///
    /// `s` is recomputed rather than read, for the rows the file actually
    /// carries. It is a function of the frozen weight, which the file does
    /// not contain -- so recomputing is the only way for the two to be
    /// guaranteed to agree, and every row the file leaves out is already the
    /// identity at s = 1.0.
    pub fn load_adapters(&mut self, blob: &[u8]) -> Result<usize, super::adapter::AdapterError> {
        use super::adapter::{AdapterError, Adapters, Dora};

        if self.cfg.hybrid() {
            return Err(AdapterError::Hybrid);
        }
        let parsed = super::adapter::parse_adapter(blob)?;
        if parsed.n_layers != self.cfg.n_layers {
            return Err(AdapterError::Shape(parsed.n_layers, self.cfg.n_layers));
        }

        let mut ad = Adapters {
            r: parsed.r,
            alpha: parsed.alpha,
            qkv: (0..self.cfg.n_layers).map(|_| [None, None, None]).collect(),
            cls: None,
        };

        let mut touched: Vec<(u32, usize, Vec<u32>)> = Vec::new();
        for (i, site) in parsed.sites.iter().enumerate() {
            let want_in = self.cfg.dim;
            let want_out = match site.kind {
                0 => self.cfg.q_dim(),
                1 | 2 => self.cfg.kv_dim(),
                3 => self.cfg.vocab_size,
                k => return Err(AdapterError::UnknownSite(k)),
            };
            if site.k_in != want_in {
                return Err(AdapterError::Shape(site.k_in, want_in));
            }
            if site.out != want_out {
                return Err(AdapterError::Shape(site.out, want_out));
            }
            if site.kind < 3 && site.layer >= self.cfg.n_layers {
                return Err(AdapterError::Row(i));
            }

            let mut d = Dora::new(parsed.r, parsed.alpha, site.k_in, site.out);
            d.a.copy_from_slice(&site.a);
            let mut rows = Vec::with_capacity(site.rows.len());
            for (o, brow, m) in site.rows.iter() {
                let o = *o as usize;
                d.b[o * parsed.r..(o + 1) * parsed.r].copy_from_slice(brow);
                d.m[o] = *m;
                rows.push(o as u32);
            }
            touched.push((site.kind, site.layer, rows));
            match site.kind {
                0 => ad.qkv[site.layer][0] = Some(d),
                1 => ad.qkv[site.layer][1] = Some(d),
                2 => ad.qkv[site.layer][2] = Some(d),
                _ => ad.cls = Some(d),
            }
        }

        // Recompute the cached scales, for the stored rows only. `false`
        // because the magnitudes came from the file: seeding here would
        // overwrite what was trained with the frozen row norm and quietly
        // undo the run that produced this adapter.
        {
            let me = &*self;
            for (kind, layer, rows) in touched.iter() {
                let w = match kind {
                    0 => me.wq(*layer),
                    1 => me.wk(*layer),
                    2 => me.wv(*layer),
                    _ => me.classifier(),
                };
                let slot = match kind {
                    0 => ad.qkv[*layer][0].as_mut(),
                    1 => ad.qkv[*layer][1].as_mut(),
                    2 => ad.qkv[*layer][2].as_mut(),
                    _ => ad.cls.as_mut(),
                };
                if let Some(d) = slot {
                    d.refresh_rows(&w, rows, false);
                }
            }
        }

        let n = parsed.sites.len();
        self.adapters = Some(ad);
        Ok(n)
    }

    pub fn detach_adapters(&mut self) -> Option<super::adapter::Adapters> {
        self.adapters.take()
    }

    /// Feed a prompt so every weight matrix streams once per chunk instead of
    /// once per token.
    ///
    /// Sequential prefill costs a full pass over the weights per prompt
    /// token: one agent decision was measured at ~15 minutes under QEMU's
    /// TCG, where the price is re-translating the same 132 MiB working set
    /// 116 times, and would be a large fraction of a second of pure DRAM
    /// bandwidth on hardware. Per output element nothing changes -- same
    /// norms, same RoPE pairing, additions ascending into one accumulator --
    /// so results are bit-identical to feeding the tokens one at a time.
    ///
    /// Returns the absolute position after the last token actually fed,
    /// clipped at `seq_len` exactly the way the callers' old loops clipped.
    pub fn prefill(&self, s: &mut State, tokens: &[usize], start_pos: usize) -> usize {
        if self.cfg.hybrid() {
            // The delta net carries recurrent state with no batch axis; until
            // a batched recurrence exists the hybrid keeps the honest loop.
            // Correctness is unaffected and the hybrid models that fit in
            // QEMU are small, so only speed waits here.
            let mut pos = start_pos;
            for &t in tokens {
                if pos >= self.cfg.seq_len {
                    break;
                }
                self.forward(s, t, pos);
                pos += 1;
            }
            return pos;
        }

        let c = &self.cfg;
        let d = c.dim;
        let qd = c.q_dim();
        let kvd = c.kv_dim();
        let hd = c.hidden_dim;
        let eps = c.norm_eps;
        let head_size = c.head_size();
        let kv_mul = c.kv_mul();
        let cap = s.cap;

        // A chunk may never exceed the window ring: its queries assume they
        // occupy the contiguous tail of the live range, which eviction in the
        // middle of a chunk would break.
        let ring = cap - c.attn_sinks.min(cap);
        let n_max = PRE_FILL_CHUNK.min(ring.max(1));

        // Chunk scratch, allocated here and dropped at return. It must not
        // live in `State`: `fork()` clones that wholesale, and multi-MiB
        // scratch would tax every deliberation fork for nothing.
        let mut x = vec![0.0f32; n_max * d];
        let mut xn = vec![0.0f32; n_max * d];
        let mut qb = vec![0.0f32; n_max * qd];
        let mut kb = vec![0.0f32; n_max * kvd];
        let mut vb = vec![0.0f32; n_max * kvd];
        let mut ao = vec![0.0f32; n_max * qd];
        let mut xo = vec![0.0f32; n_max * d];
        let mut h1 = vec![0.0f32; n_max * hd];
        let mut h3 = vec![0.0f32; n_max * hd];

        let embed = self.embed_mat();
        let inv_scale = 1.0 / tensor::sqrtf(head_size as f32);
        let mut pos = start_pos;
        let mut ci = 0usize;

        while pos < c.seq_len && ci < tokens.len() {
            let n = (tokens.len() - ci).min(n_max).min(c.seq_len - pos);

            for t in 0..n {
                let tok = tokens[ci + t].min(c.vocab_size - 1);
                embed.row_into(tok, &mut x[t * d..(t + 1) * d]);
            }

            for l in 0..c.n_layers {
                // --- attention ---
                for t in 0..n {
                    tensor::rmsnorm_eps(
                        &mut xn[t * d..(t + 1) * d],
                        &x[t * d..(t + 1) * d],
                        self.rms_att_w(l),
                        eps,
                    );
                }
                self.wq(l).matvec_batch(&mut qb, &xn, n);
                self.wk(l).matvec_batch(&mut kb, &xn, n);
                self.wv(l).matvec_batch(&mut vb, &xn, n);

                // The adapted sites, row by row over the batch.
                //
                // This pass used to be missing, and it was missing silently:
                // an adapted model prefilled its prompt through the frozen
                // weights and then decoded through the adapted ones, so the
                // same position computed two different things depending on
                // which path reached it. Nothing faults, no logit is
                // non-finite, and the model stays fluent -- the failure mode
                // this whole subsystem is built to refuse. `wrap_matvec` is
                // deliberately not reused here: it recomputes the base, and
                // the base for the whole batch has just been computed above.
                if let Some(ad) = self.adapters.as_ref() {
                    let mut ax = [0.0f32; super::adapter::MAX_RANK];
                    for (site, out, width) in [
                        (0usize, &mut qb, qd),
                        (1, &mut kb, kvd),
                        (2, &mut vb, kvd),
                    ] {
                        let Some(dora) = ad.qkv[l][site].as_ref() else { continue };
                        for t in 0..n {
                            let xrow = &xn[t * d..(t + 1) * d];
                            dora.apply(
                                &mut out[t * width..(t + 1) * width],
                                xrow,
                                &mut ax[..dora.r],
                            );
                        }
                    }
                }

                // QK-Norm before anything is cached, per head per row, the
                // order the single-token pass established.
                if c.qk_norm {
                    let qn = self.q_norm_w(l);
                    for t in 0..n {
                        for hh in 0..c.n_heads {
                            let o = t * qd + hh * head_size;
                            tensor::rmsnorm_inplace(&mut qb[o..o + head_size], qn, eps);
                        }
                    }
                    let kn = self.k_norm_w(l);
                    for t in 0..n {
                        for hh in 0..c.n_kv_heads {
                            let o = t * kvd + hh * head_size;
                            tensor::rmsnorm_inplace(&mut kb[o..o + head_size], kn, eps);
                        }
                    }
                }

                // Keys cached normed and unrotated, values raw -- the point
                // where reference.py applies its round-trip, now one row per
                // position just like the single-token pass.
                for t in 0..n {
                    let wr = Window::new(c, cap, pos + t);
                    let here = wr.slot_of(wr.live - 1);
                    s.key_cache[l].store(here, &kb[t * kvd..(t + 1) * kvd]);
                    s.value_cache[l].store(here, &vb[t * kvd..(t + 1) * kvd]);
                }

                // The chunk's queries are the newest entries, so they occupy
                // the tail of the live range contiguously -- guaranteed by
                // the ring bound on `n`.
                let wend = Window::new(c, cap, pos + n - 1);
                let base_j = wend.live - n;
                self.rope_keys(s, &wend, l, kvd, wend.live);
                let (cos, sin) = (&s.rope_cos, &s.rope_sin);
                for t in 0..n {
                    self.rope_row(cos, sin, &mut qb[t * qd..(t + 1) * qd], base_j + t);
                }

                // Causal attention: query t sees live keys 0..=base_j+t and
                // nothing later. Scores ascend over j, head dims accumulate
                // ascending -- the same order `attend` established.
                for hh in 0..c.n_heads {
                    let qo = hh * head_size;
                    let hoff = (hh / kv_mul) * head_size;
                    for t in 0..n {
                        let qi = base_j + t;
                        for j in 0..=qi {
                            let ko = j * kvd + hoff;
                            let mut score = 0.0f32;
                            for i in 0..head_size {
                                score += qb[t * qd + qo + i] * s.krot[ko + i];
                            }
                            s.att[hh * cap + j] = score * inv_scale;
                        }
                        tensor::softmax(&mut s.att[hh * cap..hh * cap + qi + 1]);
                        for i in 0..head_size {
                            ao[t * qd + qo + i] = 0.0;
                        }
                        let vc = &s.value_cache[l];
                        for j in 0..=qi {
                            let vslot = wend.slot_of(j);
                            let a = s.att[hh * cap + j];
                            for i in 0..head_size {
                                ao[t * qd + qo + i] += a * vc.at(vslot, hoff + i);
                            }
                        }
                    }
                }

                self.wo(l).matvec_batch(&mut xo, &ao, n);
                for t in 0..n {
                    for i in 0..d {
                        x[t * d + i] += xo[t * d + i];
                    }
                }

                // --- feed forward ---
                for t in 0..n {
                    tensor::rmsnorm_eps(
                        &mut xn[t * d..(t + 1) * d],
                        &x[t * d..(t + 1) * d],
                        self.rms_ffn_w(l),
                        eps,
                    );
                }
                self.w1(l).matvec_batch(&mut h1, &xn, n);
                self.w3(l).matvec_batch(&mut h3, &xn, n);
                for t in 0..n {
                    tensor::swiglu(&mut h1[t * hd..(t + 1) * hd], &h3[t * hd..(t + 1) * hd]);
                }
                self.w2(l).matvec_batch(&mut xo, &h1, n);
                for t in 0..n {
                    for i in 0..d {
                        x[t * d + i] += xo[t * d + i];
                    }
                }
            }

            pos += n;
            ci += n;
        }

        let fed = pos - start_pos;
        if fed > 0 {
            // Only the last row's classifier matters: decode samples the next
            // token from here, and every earlier row's logits were thrown
            // away by the old loop too -- it just also computed them.
            let last = (fed - 1) % n_max;
            tensor::rmsnorm_eps(
                &mut s.xb[..d],
                &x[last * d..(last + 1) * d],
                self.rms_final_w(),
                eps,
            );
            match self.adapters.as_ref().and_then(|a| a.cls.as_ref()) {
                Some(dora) => {
                    let xb = &s.xb;
                    let mut la = core::mem::take(&mut s.la);
                    self.classifier().wrap_matvec(dora, xb, &mut la, &mut s.logits);
                    s.la = la;
                }
                None => self.classifier().matvec(&mut s.logits, &s.xb),
            }
            // Parity with `forward`: the final hidden stays in `x`.
            s.x[..d].copy_from_slice(&x[last * d..(last + 1) * d]);
        }
        pos
    }

    /// Rotate the query at the position it will occupy in the cache.
    ///
    /// Only the first `rot_dim` dimensions of each head move. For everything
    /// before Qwen3.5 that is the whole head and this is the loop it always
    /// was; Qwen3.5 rotates 64 of 256 and passes the other 192 through
    /// untouched, which is a property of the *tables* being narrower rather
    /// than of any test in here.
    fn rope_q(&self, s: &mut State, w: &Window) {
        let qpos = w.live.saturating_sub(1);
        let qd = self.cfg.q_dim();
        let (cos, sin) = (&s.rope_cos, &s.rope_sin);
        self.rope_row(cos, sin, &mut s.q[..qd], qpos);
    }

    /// Rotate one query row of `q_dim` values for the live index it occupies.
    ///
    /// Shared by the single-token pass and the batched prefill so this
    /// arithmetic exists exactly once -- it is the code whose silent variant
    /// already cost SmolLM2 its geography once.
    fn rope_row(&self, cos: &[f32], sin: &[f32], row: &mut [f32], live_idx: usize) {
        let c = &self.cfg;
        let head_size = c.head_size();
        let half = c.rot_dim() / 2;
        for h in 0..c.n_heads {
            let base = h * head_size;
            for p in 0..half {
                let fcr = cos[live_idx * half + p];
                let fci = sin[live_idx * half + p];
                let (i, j) = if c.rope_interleaved {
                    (base + 2 * p, base + 2 * p + 1)
                } else {
                    (base + p, base + p + half)
                };
                let (a, b) = (row[i], row[j]);
                row[i] = a * fcr - b * fci;
                row[j] = a * fci + b * fcr;
            }
        }
    }

    /// Rotate every live key into `krot`, indexed by *cache* position rather
    /// than by the absolute position it arrived at.
    ///
    /// Done once per layer rather than once per head: grouped-query attention
    /// shares each key across `kv_mul` heads, so rotating per head would redo
    /// the same work three times.
    fn rope_keys(&self, s: &mut State, w: &Window, cache: usize, kv_dim: usize, upto: usize) {
        let c = &self.cfg;
        let head_size = c.head_size();
        let half = c.rot_dim() / 2;
        let kc = &s.key_cache[cache];
        for j in 0..upto {
            let src = w.slot_of(j);
            let dst = j * kv_dim;
            for h in 0..c.n_kv_heads {
                let base = h * head_size;
                // Dimensions past the rotated prefix still have to reach
                // `krot`, or the dot product below reads whatever the last
                // token left there.
                for i in c.rot_dim()..head_size {
                    s.krot[dst + base + i] = kc.at(src, base + i);
                }
                for p in 0..half {
                    let fcr = s.rope_cos[j * half + p];
                    let fci = s.rope_sin[j * half + p];
                    let (a_off, b_off) = if c.rope_interleaved {
                        (base + 2 * p, base + 2 * p + 1)
                    } else {
                        (base + p, base + p + half)
                    };
                    let k0 = kc.at(src, a_off);
                    let k1 = kc.at(src, b_off);
                    s.krot[dst + a_off] = k0 * fcr - k1 * fci;
                    s.krot[dst + b_off] = k0 * fci + k1 * fcr;
                }
            }
        }
    }

    /// Scaled dot-product attention over the live cache, into `xb`.
    fn attend(&self, s: &mut State, w: &Window, cache: usize, kv_dim: usize, kv_mul: usize) {
        let c = &self.cfg;
        let head_size = c.head_size();
        let cap = s.cap;
        let scale = 1.0 / tensor::sqrtf(head_size as f32);
        for h in 0..c.n_heads {
            let qo = h * head_size;
            let ao = h * cap;
            let hoff = (h / kv_mul) * head_size;
            for t in 0..w.live {
                let ko = t * kv_dim + hoff;
                let mut score = 0.0f32;
                for i in 0..head_size {
                    score += s.q[qo + i] * s.krot[ko + i];
                }
                s.att[ao + t] = score * scale;
            }
            tensor::softmax(&mut s.att[ao..ao + w.live]);

            for i in 0..head_size {
                s.xb[qo + i] = 0.0;
            }
            let vc = &s.value_cache[cache];
            for t in 0..w.live {
                let vslot = w.slot_of(t);
                let a = s.att[ao + t];
                for i in 0..head_size {
                    s.xb[qo + i] += a * vc.at(vslot, hoff + i);
                }
            }
        }
    }

    /// One decode step through a Qwen3.5 hybrid.
    ///
    /// Structurally the same shape as the dense pass -- mixer, residual, feed
    /// forward, residual -- with three differences that all sit inside the
    /// mixer. Three layers in four run a recurrence with no cache at all; the
    /// fourth runs attention whose query projection is twice as wide as its
    /// query and whose output passes through a gate; and every norm out here
    /// scales by `1 + w` rather than by `w`.
    fn forward_hybrid(&self, s: &mut State, token: usize, pos: usize) {
        let c = &self.cfg;
        let eps = c.norm_eps;
        let w = Window::new(c, s.cap, pos);

        // Position 0 starts a sequence, and for a recurrence that has to mean
        // something. The KV cache needs no such rule -- attention reads slots
        // `0..live` and position 0 overwrites slot 0, so whatever ran before
        // is simply not looked at. The delta net's state is different in kind:
        // nothing overwrites it, it is *carried*, so without this every
        // `logits` call would answer from a state left behind by the boot
        // selftests and no two runs would agree.
        if pos == 0 {
            for ls in s.linear.iter_mut() {
                ls.clear();
            }
        }

        let tok = token.min(c.vocab_size - 1);
        self.embed_mat().row_into(tok, &mut s.x);

        for l in 0..c.n_layers {
            tensor::rmsnorm_1p(&mut s.xb[..c.dim], &s.x, self.rms_att_w(l), eps);
            match c.layer_kind(l) {
                LayerKind::Linear => self.delta_net(s, l),
                LayerKind::Full => self.gated_attention(s, l, &w),
            }
            tensor::add_into(&mut s.x, &s.xb2);

            tensor::rmsnorm_1p(&mut s.xb[..c.dim], &s.x, self.rms_ffn_w(l), eps);
            let o = self.lo(l);
            self.hq(o.w1, c.hidden_dim, c.dim).matvec(&mut s.hb, &s.xb);
            self.hq(o.w3, c.hidden_dim, c.dim).matvec(&mut s.hb2, &s.xb);
            tensor::swiglu(&mut s.hb, &s.hb2);
            self.hq(o.w2, c.dim, c.hidden_dim).matvec(&mut s.xb2, &s.hb);
            tensor::add_into(&mut s.x, &s.xb2);
        }

        tensor::rmsnorm_1p(&mut s.xb[..c.dim], &s.x, self.rms_final_w(), eps);
        self.classifier().matvec(&mut s.logits, &s.xb);
    }

    /// Full attention with a partial rotation and an output gate, into `xb2`.
    fn gated_attention(&self, s: &mut State, l: usize, w: &Window) {
        let c = &self.cfg;
        let o = self.lo(l);
        let hd = c.head_dim;
        let (kv_dim, q_dim) = (c.kv_dim(), c.q_dim());
        let eps = c.norm_eps;
        // Every full-attention layer has a cache and no other layer does, so
        // this cannot be `l`: at layer 19 of Qwen3.5-0.8B the fifth cache is
        // wanted, not the twentieth, which does not exist.
        let cache = match c.kv_slot(l) {
            Some(k) => k,
            None => return,
        };

        // Query and gate leave one projection together, interleaved per head:
        // `hd` of query then `hd` of gate, for each head in turn. Reading it
        // as all queries followed by all gates gives a perfectly well-shaped
        // and completely wrong split.
        self.hq(o.wq, 2 * q_dim, c.dim).matvec(&mut s.qg, &s.xb);
        self.hq(o.wk, kv_dim, c.dim).matvec(&mut s.kbuf, &s.xb);
        self.hq(o.wv, kv_dim, c.dim).matvec(&mut s.vbuf, &s.xb);
        for h in 0..c.n_heads {
            let (src, dst) = (h * 2 * hd, h * hd);
            s.q[dst..dst + hd].copy_from_slice(&s.qg[src..src + hd]);
        }

        // Both flags are set by every Qwen3.5 the converter has seen, and
        // both are read from the header rather than assumed: a hybrid without
        // QK-Norm would load fine and attend to the wrong things, and one
        // without an output gate would have its attention scaled by the
        // sigmoid of whatever the second half of `wq` happened to mean.
        if c.qk_norm {
            let qn = self.q_norm_w(l);
            for h in 0..c.n_heads {
                tensor::rmsnorm_1p_inplace(&mut s.q[h * hd..(h + 1) * hd], qn, eps);
            }
            let kn = self.k_norm_w(l);
            for h in 0..c.n_kv_heads {
                tensor::rmsnorm_1p_inplace(&mut s.kbuf[h * hd..(h + 1) * hd], kn, eps);
            }
        }

        s.key_cache[cache].store(w.slot_of(w.live - 1), &s.kbuf);
        s.value_cache[cache].store(w.slot_of(w.live - 1), &s.vbuf);

        self.rope_q(s, w);
        self.rope_keys(s, w, cache, kv_dim, w.live);
        self.attend(s, w, cache, kv_dim, c.kv_mul());

        // The gate multiplies the concatenated head outputs, before `wo` and
        // not after: `wo` mixes heads, so gating on the far side would apply
        // each head's gate to a blend of every head.
        if c.attn_output_gate {
            for h in 0..c.n_heads {
                for i in 0..hd {
                    s.xb[h * hd + i] *= tensor::sigmoid(s.qg[h * 2 * hd + hd + i]);
                }
            }
        }
        self.hq(o.wo, c.dim, q_dim).matvec(&mut s.xb2, &s.xb);
    }

    /// Gated DeltaNet: a depthwise causal convolution, then one step of the
    /// delta rule against a state that does not grow. Output into `xb2`.
    ///
    /// The rule is an associative memory that corrects itself. `mem` is what
    /// the state currently returns for this key; `delta` is how far that is
    /// from the value that actually arrived; the state absorbs `beta` of the
    /// difference. Written out over a sequence it is a linear attention, and
    /// written out for one token -- which is all a decoder ever needs -- it is
    /// two passes over a fixed-size array.
    fn delta_net(&self, s: &mut State, l: usize) {
        let c = &self.cfg;
        let o = self.lo(l);
        let (hk, hv) = (c.lin_k_head, c.lin_v_head);
        let (nk, nv) = (c.lin_k_heads, c.lin_v_heads);
        let (kdim, vdim, conv_dim) = (c.lin_k_dim(), c.lin_v_dim(), c.conv_dim());
        let kern = c.conv_kernel;
        let mul = c.lin_mul();
        let eps = c.norm_eps;
        let idx = match c.lin_slot(l) {
            Some(i) => i,
            None => return,
        };

        self.hq(o.in_qkv, conv_dim, c.dim).matvec(&mut s.qkv, &s.xb);
        self.hq(o.in_z, vdim, c.dim).matvec(&mut s.zbuf, &s.xb);
        // `in_proj_a` and `in_proj_b` stay f32 through the converter: they are
        // 32x2048, a rounding error in the size of the file, and they feed a
        // loop that carries its own output forward, so quantisation error
        // would compound step over step instead of averaging out.
        tensor::matmul(&mut s.abuf, &s.xb, self.hf(o.in_a, nv * c.dim), c.dim, nv);
        tensor::matmul(&mut s.bbuf, &s.xb, self.hf(o.in_b, nv * c.dim), c.dim, nv);

        let cw = self.hf(o.conv1d, conv_dim * kern);
        let a_log = self.hf(o.a_log, nv);
        let dt_bias = self.hf(o.dt_bias, nv);
        let gnorm = self.hf(o.gate_norm, hv);

        {
            let State { qkv, zbuf, abuf, bbuf, core, dbuf, linear, .. } = &mut *s;
            let ls = &mut linear[idx];

            // Depthwise causal convolution over the last `kern` inputs, the
            // older `kern - 1` of which live in a ring. `conv_at` indexes the
            // oldest, so ring position `(conv_at + j) % (kern - 1)` holds input
            // `t - kern + 1 + j` and the current input is the last tap.
            let hist = kern - 1;
            for ch in 0..conv_dim {
                let mut acc = cw[ch * kern + hist] * qkv[ch];
                for j in 0..hist {
                    let r = (ls.conv_at + j) % hist;
                    acc += cw[ch * kern + j] * ls.conv[r * conv_dim + ch];
                }
                // Stashed before SiLU: the convolution runs over the
                // projection, not over its activation.
                ls.conv[ls.conv_at * conv_dim + ch] = qkv[ch];
                qkv[ch] = tensor::silu(acc);
            }
            ls.conv_at = (ls.conv_at + 1) % hist;

            // q and k are L2-normalised *inside* the rule, and the
            // 1/sqrt(k_head_dim) scale goes on the query only. Both are per
            // key head, so value heads sharing a key head share the vector --
            // which is why this normalises `nk` of them and the loop below
            // indexes by `h / mul`.
            let qscale = 1.0 / tensor::sqrtf(hk as f32);
            for h in 0..nk {
                let qo = h * hk;
                tensor::l2norm_inplace(&mut qkv[qo..qo + hk], L2_EPS);
                for v in &mut qkv[qo..qo + hk] {
                    *v *= qscale;
                }
                let ko = kdim + h * hk;
                tensor::l2norm_inplace(&mut qkv[ko..ko + hk], L2_EPS);
            }

            for h in 0..nv {
                let kh = h / mul;
                let (qo, ko, vo) = (kh * hk, kdim + kh * hk, 2 * kdim + h * hv);
                let base = h * hk * hv;
                let out = h * hv;
                let beta = tensor::sigmoid(bbuf[h]);
                // `g = -exp(A_log) * softplus(a + dt_bias)`, and the state
                // decays by `exp(g)`. softplus is why `tensor` grew a
                // two-branch one: the naive `ln(1 + exp(x))` returns a
                // plausible 88.7 above x=88 rather than an infinity anyone
                // would notice, and a wrong decay does not produce a NaN --
                // it quietly empties the state.
                let decay = tensor::expf(-tensor::expf(a_log[h])
                    * tensor::softplus(abuf[h] + dt_bias[h]));

                // Pass one: decay the state, and read out what it currently
                // holds for this key.
                for d in dbuf.iter_mut() {
                    *d = 0.0;
                }
                for i in 0..hk {
                    let k = qkv[ko + i];
                    let row = base + i * hv;
                    for j in 0..hv {
                        let v = ls.s[row + j] * decay;
                        ls.s[row + j] = v;
                        dbuf[j] += v * k;
                    }
                }
                // How far that is from the value that actually arrived.
                for j in 0..hv {
                    dbuf[j] = (qkv[vo + j] - dbuf[j]) * beta;
                }
                // Pass two: absorb the correction and read the query out.
                for j in 0..hv {
                    core[out + j] = 0.0;
                }
                for i in 0..hk {
                    let (k, q) = (qkv[ko + i], qkv[qo + i]);
                    let row = base + i * hv;
                    for j in 0..hv {
                        let v = ls.s[row + j] + k * dbuf[j];
                        ls.s[row + j] = v;
                        core[out + j] += v * q;
                    }
                }

                // The one norm in this model that scales by `w` rather than
                // `1 + w`, with a gate that goes through silu rather than the
                // sigmoid the attention gate uses.
                tensor::rmsnorm_gated(
                    &mut core[out..out + hv],
                    gnorm,
                    &zbuf[out..out + hv],
                    eps,
                );
            }
        }

        self.hq(o.out_proj, c.dim, vdim).matvec(&mut s.xb2, &s.core);
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






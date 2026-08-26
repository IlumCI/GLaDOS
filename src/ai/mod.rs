//! Machine learning primitives.

pub mod agent;
pub mod adapter;
pub mod aixi;
pub mod backward;
pub mod constrain;
pub mod context;
pub mod initiative;
pub mod train;
pub mod deliberate;
pub mod corpus;
pub mod futures;
pub mod godbits;
pub mod council;
pub mod harness;
pub mod model;
pub mod probe;
pub mod sample;
pub mod tensor;
pub mod tokenizer;
pub mod vocab;
pub mod weights;

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

static FPU_CHECKS: AtomicU64 = AtomicU64::new(0);
static FPU_ERRORS: AtomicU64 = AtomicU64::new(0);

pub fn fpu_checks() -> u64 {
    FPU_CHECKS.load(Ordering::Relaxed)
}

pub fn fpu_errors() -> u64 {
    FPU_ERRORS.load(Ordering::Relaxed)
}

/// Hold a caller-specific pattern in YMM registers across a spin, then verify
/// it survived. Returns false if the registers were clobbered.
///
/// Called continuously by the clock task with one pattern while the shell runs
/// tensor work with entirely different values in the same registers. With
/// extended state saved on context switch this never fails; without it, the
/// error count climbs the moment both tasks touch floating point.
pub fn fpu_guard(tag: f32) -> bool {
    let f = crate::cpu::detected();
    if !f.avx_enabled {
        return true;
    }
    let mut input = [0.0f32; 32];
    for (i, v) in input.iter_mut().enumerate() {
        *v = tag + i as f32 * 0.25;
    }
    let mut output = [0.0f32; 32];
    // The spin has to dominate this task's runtime, or almost no timer
    // interrupt lands inside the window where the values live in registers,
    // and the check silently proves nothing.
    unsafe { tensor::ymm_roundtrip(&input, &mut output, 300_000) };

    FPU_CHECKS.fetch_add(1, Ordering::Relaxed);
    for i in 0..32 {
        if output[i] != input[i] {
            FPU_ERRORS.fetch_add(1, Ordering::Relaxed);
            return false;
        }
    }
    true
}

use crate::gfx::console::{self, LTCYAN, LTGRAY, LTGREEN, LTRED, WHITE, YELLOW};
use crate::sync::Racy;
use crate::uefi::Blob;
use crate::{kprint, kprintln};
use alloc::vec;
use alloc::vec::Vec;

fn close(a: f32, b: f32, tol: f32) -> bool {
    let d = if a > b { a - b } else { b - a };
    d <= tol * (1.0 + if a > 0.0 { a } else { -a })
}

fn check(name: &str, ok: bool, detail: &str) -> bool {
    if ok {
        console::set_color(LTGREEN);
        kprintln!("  ok    {:<22} {}", name, detail);
    } else {
        console::set_color(LTRED);
        kprintln!("  FAIL  {:<22} {}", name, detail);
    }
    console::set_color(WHITE);
    ok
}

/// Check the primitives against hand-computed values, and the SIMD kernel
/// against the scalar one.
///
/// The scalar path is the oracle. A hand-written AVX kernel that is subtly
/// wrong -- a misaligned tail, a bad horizontal sum -- produces numbers that
/// look entirely plausible, so "it ran without faulting" proves nothing.
pub fn selftest() -> bool {
    let mut all = true;
    console::set_color(YELLOW);
    kprintln!("[tensor]");
    console::set_color(WHITE);

    // --- math ---
    all &= check(
        "sqrtf(2)",
        close(tensor::sqrtf(2.0), 1.414_213_6, 1e-6),
        "expect 1.4142136",
    );
    all &= check("expf(0)", close(tensor::expf(0.0), 1.0, 1e-6), "expect 1");
    all &= check(
        "expf(1)",
        close(tensor::expf(1.0), 2.718_281_7, 1e-5),
        "expect 2.7182817",
    );
    all &= check(
        "expf(-5)",
        close(tensor::expf(-5.0), 0.006_737_947, 1e-4),
        "expect 0.006737947",
    );
    all &= check(
        "lnf(e)",
        close(tensor::lnf(2.718_281_7), 1.0, 1e-5),
        "expect 1",
    );
    all &= check(
        "sinf(pi/2)",
        close(tensor::sinf(core::f32::consts::FRAC_PI_2), 1.0, 1e-4),
        "expect 1",
    );
    all &= check(
        "cosf(0)",
        close(tensor::cosf(0.0), 1.0, 1e-4),
        "expect 1",
    );
    all &= check(
        "sigmoid(0), (2)",
        close(tensor::sigmoid(0.0), 0.5, 1e-6)
            && close(tensor::sigmoid(2.0), 0.880_797, 1e-5),
        "expect 0.5, 0.880797",
    );
    all &= check(
        "silu(0), (1)",
        close(tensor::silu(0.0), 0.0, 1e-6) && close(tensor::silu(1.0), 0.731_058_6, 1e-5),
        "expect 0, 0.7310586",
    );
    // The tails are the whole point of the two-branch form. At x = 100 the
    // naive `lnf(1 + expf(x))` overflows to infinity and `lnf` has no guard
    // for one -- it extracts the 0xFF exponent and returns 88.7, a plausible
    // number that is not the answer, and a wrong decay does not produce a NaN
    // anyone would notice, it quietly empties the recurrence's state.
    all &= check(
        "softplus(0), (100)",
        close(tensor::softplus(0.0), core::f32::consts::LN_2, 1e-5)
            && close(tensor::softplus(100.0), 100.0, 1e-3),
        "expect ln2, 100",
    );
    all &= check(
        "softplus(-10)",
        close(tensor::softplus(-10.0), 4.539_79e-5, 1e-6),
        "expect 4.53979e-5",
    );
    {
        // (1 + w) rather than w: with w = 1 the scale is 2, so a unit-RMS
        // input comes out at 2. The ordinary convention would give 1 here and
        // a model that is wrong from its first layer with nothing to catch it.
        let x = [3.0f32, 4.0];
        let mut out = [0.0f32; 2];
        tensor::rmsnorm_1p(&mut out, &x, &[1.0, 1.0], 0.0);
        let rms = tensor::sqrtf((9.0 + 16.0) / 2.0);
        all &= check(
            "rmsnorm_1p",
            close(out[0], 2.0 * 3.0 / rms, 1e-5) && close(out[1], 2.0 * 4.0 / rms, 1e-5),
            "expect 2x the plain norm",
        );

        let mut v = [3.0f32, 4.0];
        tensor::l2norm_inplace(&mut v, 0.0);
        all &= check(
            "l2norm",
            close(v[0], 0.6, 1e-5) && close(v[1], 0.8, 1e-5),
            "expect [0.6, 0.8]",
        );
    }

    // --- matmul against a hand-computed case ---
    // w = [[1,2,3],[4,5,6]], x = [1,2,3] -> [14, 32]
    let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = [1.0f32, 2.0, 3.0];
    let mut out = [0.0f32; 2];
    tensor::matmul_scalar(&mut out, &x, &w, 3, 2);
    all &= check(
        "matmul scalar",
        close(out[0], 14.0, 1e-6) && close(out[1], 32.0, 1e-6),
        "expect [14, 32]",
    );

    // --- softmax ---
    let mut s = [1.0f32, 2.0, 3.0];
    tensor::softmax(&mut s);
    let sum = s[0] + s[1] + s[2];
    all &= check(
        "softmax sums to 1",
        close(sum, 1.0, 1e-5) && s[2] > s[1] && s[1] > s[0],
        "and is monotonic",
    );

    // --- rmsnorm ---
    let xs = [3.0f32, 4.0];
    let ws = [1.0f32, 1.0];
    let mut rn = [0.0f32; 2];
    tensor::rmsnorm(&mut rn, &xs, &ws);
    // rms = sqrt((9+16)/2) = 3.5355; 3/3.5355 = 0.8485
    all &= check(
        "rmsnorm",
        close(rn[0], 0.848_5, 1e-3) && close(rn[1], 1.131_4, 1e-3),
        "expect [0.8485, 1.1314]",
    );

    // --- SIMD against the scalar oracle, on a size with a ragged tail ---
    let f = crate::cpu::detected();
    if f.avx_enabled && f.fma {
        let n = 173usize; // deliberately not a multiple of 8 or 32
        let d = 7usize;
        let mut wv: Vec<f32> = Vec::with_capacity(n * d);
        let mut xv: Vec<f32> = Vec::with_capacity(n);
        // A deterministic, non-trivial pattern; no RNG needed.
        for i in 0..n * d {
            wv.push(((i * 37 % 101) as f32 - 50.0) * 0.01);
        }
        for i in 0..n {
            xv.push(((i * 53 % 97) as f32 - 48.0) * 0.02);
        }
        let mut a = vec![0.0f32; d];
        let mut b = vec![0.0f32; d];
        tensor::matmul_scalar(&mut a, &xv, &wv, n, d);
        unsafe { tensor::matmul_avx2(&mut b, &xv, &wv, n, d) };

        let mut worst = 0.0f32;
        for i in 0..d {
            let e = if a[i] > b[i] { a[i] - b[i] } else { b[i] - a[i] };
            if e > worst {
                worst = e;
            }
        }
        // FMA does not round the intermediate product, so results differ from
        // the scalar path in the last bits. That is expected and is not error.
        all &= check(
            "matmul avx2 == scalar",
            worst < 1e-3,
            "n=173 (ragged tail), d=7",
        );
    } else {
        console::set_color(YELLOW);
        kprintln!("  skip  matmul avx2             AVX not enabled on this CPU");
        console::set_color(WHITE);
    }

    // --- cl100k pre-tokenizer ---
    //
    // Qwen3 splits text differently from the GPT-2 lineage, and a wrong split
    // does not fail: it moves merge boundaries so the model is fed sequences it
    // never saw in training, and reads as the model simply being worse. These
    // expectations are the output of `tools/tokenizer.py`, which is verified
    // against the reference `tokenizers` library over the whole corpus -- so
    // this checks the Rust port against a Python implementation that is itself
    // checked against the real thing.
    //
    // Each case is here for a specific clause: `(x` for a non-space word lead,
    // `'t` for case-insensitive contractions, `a1b2` for one-digit-at-a-time,
    // `.\n` for punctuation swallowing newlines, and the double space for a run
    // that leaves its last character to the following word. The last pair
    // exercises the Qwen3.5 marks clauses: a combining mark ends a punctuation
    // run and then stands alone via lead backtracking.
    const SPLITS: [(&str, &[&str]); 7] = [
        ("println(x)", &["println", "(x", ")"]),
        ("Don't STOP", &["Don", "'t", " STOP"]),
        ("a  b", &["a", " ", " b"]),
        ("x = 6*7; f(y)", &["x", " =", " ", "6", "*", "7", ";", " f", "(y", ")"]),
        ("line1\n\nline2", &["line", "1", "\n\n", "line", "2"]),
        ("end.\n", &["end", ".\n"]),
        ("a1b2", &["a", "1", "b", "2"]),
    ];
    let mut splits_ok = true;
    for (text, want) in SPLITS {
        let mut spans: Vec<(usize, usize)> = Vec::new();
        tokenizer::pretokenize_cl100k(text, false, &mut spans);
        let got = spans.len() == want.len()
            && spans.iter().zip(want.iter()).all(|(&(a, b), w)| &&text[a..b] == w);
        if !got {
            splits_ok = false;
            console::set_color(LTRED);
            kprintln!("    {:?} split into {} pieces, wanted {}", text, spans.len(), want.len());
            console::set_color(WHITE);
        }
    }
    all &= check("cl100k pre-tokenizer", splits_ok, "7 cases, Qwen3 clauses");

    // The marks variant. U+0301 after the full stop must end the punctuation
    // run (plain cl100k would swallow it) and then form its own piece through
    // lead backtracking; Devanagari keeps its vowel signs through NFC, so a
    // word run carries marks mid-stream.
    const MARK_SPLITS: [(&str, &[&str]); 2] = [
        ("end.\u{0301}go", &["end", ".", "\u{0301}", "go"]),
        ("Ã Â¤Â¹Ã Â¤Â¿Ã Â¤Â¨Ã Â¥ÂÃ Â¤Â¦Ã Â¥â‚¬!", &["Ã Â¤Â¹Ã Â¤Â¿Ã Â¤Â¨Ã Â¥ÂÃ Â¤Â¦Ã Â¥â‚¬", "!"]),
    ];
    let mut marks_ok = true;
    for (text, want) in MARK_SPLITS {
        let mut spans: Vec<(usize, usize)> = Vec::new();
        tokenizer::pretokenize_cl100k(text, true, &mut spans);
        let got = spans.len() == want.len()
            && spans.iter().zip(want.iter()).all(|(&(a, b), w)| &&text[a..b] == w);
        if !got {
            marks_ok = false;
            console::set_color(LTRED);
            kprintln!("    {:?} split into {} pieces, wanted {}", text, spans.len(), want.len());
            console::set_color(WHITE);
        }
    }
    all &= check("cl100km pre-tokenizer", marks_ok, "2 cases, Qwen3.5 marks");

    all
}

/// Build a small transformer, run it, and check the result is meaningful.
///
/// The weights are synthetic, so the output is not language -- it is noise
/// with the right shape. That is deliberate: this proves the forward pass, the
/// KV cache, RoPE and attention are wired correctly and fast enough, without
/// needing weights on a machine that still has no storage. Swapping in a real
/// checkpoint later changes the buffer, not the code.
pub fn model_demo() {
    let cfg = model::Config {
        dim: 64,
        hidden_dim: 176,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 4,
        head_dim: 16,
        vocab_size: 256, // byte-level: no tokenizer needed
        seq_len: 64,
        norm_eps: 1e-5,
        qk_norm: false,
        rope_interleaved: false,
        shared_classifier: true,
        rope_theta: 10000.0,
        attn_sinks: 0,
        attn_window: usize::MAX,
        rotary_dim: 16,
        ..Default::default()
    };

    console::set_color(YELLOW);
    kprintln!("[model]");
    console::set_color(WHITE);
    kprintln!(
        "  dim {}  hidden {}  layers {}  heads {}  vocab {}  seq {}",
        cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.vocab_size, cfg.seq_len
    );

    let Some(m) = model::Model::synthetic(cfg, 0xC0FFEE) else {
        console::set_color(LTRED);
        kprintln!("  out of memory building the model");
        console::set_color(WHITE);
        return;
    };
    let mut s = model::State::new(&cfg);
    kprintln!(
        "  {} parameters, {} KiB weights + {} KiB state",
        cfg.param_count(),
        m.weight_bytes() / 1024,
        s.bytes(&cfg) / 1024
    );

    let mut ok = true;

    // 1. Output must be finite. NaN here means rmsnorm divided by zero or
    //    softmax overflowed, both of which produce plausible-looking code.
    m.forward(&mut s, b'A' as usize, 0);
    let finite = s.logits.iter().all(|v| v.is_finite());
    ok &= check("logits finite", finite, "no NaN or infinity");

    // 2. Determinism. Same token, same position, same weights -> same numbers.
    //    If this fails, extended state is not being preserved, or something is
    //    reading uninitialised memory.
    let first: Vec<f32> = s.logits.clone();
    let mut s2 = model::State::new(&cfg);
    m.forward(&mut s2, b'A' as usize, 0);
    let same = first.iter().zip(s2.logits.iter()).all(|(a, b)| a == b);
    ok &= check("deterministic", same, "same input, identical logits");

    // 3. Different inputs must give different outputs, or the model is
    //    ignoring its input and every other check passes vacuously.
    let mut s3 = model::State::new(&cfg);
    m.forward(&mut s3, b'Z' as usize, 0);
    let differs = first.iter().zip(s3.logits.iter()).any(|(a, b)| a != b);
    ok &= check("input-sensitive", differs, "'A' and 'Z' differ");

    // 4. Softmax over the logits must be a probability distribution.
    let mut probs = first.clone();
    tensor::softmax(&mut probs);
    let sum: f32 = probs.iter().sum();
    let all_pos = probs.iter().all(|p| *p >= 0.0);
    ok &= check("logits -> distribution", (sum - 1.0).abs() < 1e-3 && all_pos, "sums to 1");

    // 5. The KV cache has to make position matter: the same token at a later
    //    position, after real history, must not produce the same logits.
    let mut s4 = model::State::new(&cfg);
    for (i, b) in b"hello wor".iter().enumerate() {
        m.forward(&mut s4, *b as usize, i);
    }
    let seq_logits: Vec<f32> = s4.logits.clone();
    let ctx_matters = seq_logits.iter().zip(first.iter()).any(|(a, b)| a != b);
    ok &= check("context changes output", ctx_matters, "position 8 vs position 0");

    // 6. Qwen3 geometry: a head that is not `dim / n_heads`, plus QK-Norm.
    //
    //    Worth its own model because every shape here used to be derivable from
    //    `dim` and `n_heads`, and a wrong derivation divides evenly and produces
    //    a self-consistent network that attends to the wrong things. This one is
    //    64 wide with 4 heads of 32, so `q_dim` is 128 -- double the residual
    //    stream -- and `kv_dim` is 64 where the old formula would have said 32.
    //    If `xb` is sized by `dim` alone this panics rather than misbehaving,
    //    which is the point.
    let wide = model::Config {
        dim: 128,
        hidden_dim: 512,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 32,
        vocab_size: 1024,
        seq_len: 16384,
        norm_eps: 1e-6,
        qk_norm: true,
        rope_interleaved: false,
        shared_classifier: true,
        rope_theta: 1_000_000.0,
        attn_sinks: 0,
        attn_window: usize::MAX,
        rotary_dim: 32,
        ..Default::default()
    };
    ok &= check(
        "wide-head geometry",
        wide.q_dim() == 128 && wide.kv_dim() == 64 && wide.head_size() == 32,
        "q_dim 128, kv_dim 64, head 32",
    );
    match model::Model::synthetic(wide, 0xC0FFEE) {
        Some(wm) => {
            let mut ws = model::State::new(&wide);
            for (i, b) in b"hello wor".iter().enumerate() {
                wm.forward(&mut ws, *b as usize, i);
            }
            let a: Vec<f32> = ws.logits.clone();
            let mut ws2 = model::State::new(&wide);
            for (i, b) in b"hello wor".iter().enumerate() {
                wm.forward(&mut ws2, *b as usize, i);
            }
            let mut ws3 = model::State::new(&wide);
            for (i, b) in b"hello wox".iter().enumerate() {
                wm.forward(&mut ws3, *b as usize, i);
            }
            ok &= check(
                "wide-head forward",
                a.iter().all(|v| v.is_finite())
                    && a.iter().zip(ws2.logits.iter()).all(|(x, y)| x == y)
                    && a.iter().zip(ws3.logits.iter()).any(|(x, y)| x != y),
                "finite, deterministic, input-sensitive",
            );
            // The probe reads this; on a wide-head model the buffer behind it is
            // longer than the residual stream and the tail is attention output.
            ok &= check("hidden bounded to dim", ws.hidden().len() == wide.dim, "64 features");
        }
        None => ok &= check("wide-head forward", false, "out of memory"),
    }

    // --- throughput ---
    let hz = crate::TIMER_HZ as u64;
    let t_start = crate::dev::lapic::ticks();
    while crate::dev::lapic::ticks() == t_start {
        core::hint::spin_loop();
    }
    let t0 = crate::dev::lapic::ticks();
    let mut tokens = 0u64;
    let mut st = model::State::new(&cfg);
    while crate::dev::lapic::ticks() - t0 < hz / 2 {
        m.forward(&mut st, (tokens % 256) as usize, (tokens % 32) as usize);
        tokens += 1;
    }
    let elapsed = crate::dev::lapic::ticks() - t0;
    let per_sec = tokens * hz / elapsed.max(1);
    kprintln!("  {} tokens in {} ticks = {} tokens/sec", tokens, elapsed, per_sec);

    // Top prediction, purely to show the pipeline end to end.
    let top = tensor::argmax(&first);
    kprintln!("  argmax token {} ({:?})", top, top as u8 as char);

    if ok {
        console::set_color(LTGREEN);
        kprintln!("  forward pass verified");
    } else {
        console::set_color(LTRED);
        kprintln!("  FORWARD PASS FAILED CHECKS");
    }
    console::set_color(WHITE);
}

/// Measure sustained matmul throughput at a size typical of a small model.
pub fn bench() {
    let n = 512usize;
    let d = 512usize;
    let mut wv: Vec<f32> = Vec::with_capacity(n * d);
    for i in 0..n * d {
        wv.push(((i % 251) as f32 - 125.0) * 0.001);
    }
    let xv: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let mut out = vec![0.0f32; d];

    let hz = crate::TIMER_HZ as u64;
    let target = hz / 2; // half a second
    let start = crate::dev::lapic::ticks();
    // Wait for a tick edge so the interval is not short by up to one period.
    while crate::dev::lapic::ticks() == start {
        core::hint::spin_loop();
    }
    let t0 = crate::dev::lapic::ticks();

    let mut iters = 0u64;
    while crate::dev::lapic::ticks() - t0 < target {
        tensor::matmul(&mut out, &xv, &wv, n, d);
        iters += 1;
    }
    let elapsed = crate::dev::lapic::ticks() - t0;

    // Two flops per element: one multiply, one add.
    let flops = 2.0 * (n * d) as f32 * iters as f32;
    let seconds = elapsed as f32 / hz as f32;
    let gflops = flops / seconds / 1.0e9;

    console::set_color(YELLOW);
    kprintln!("[bench]");
    console::set_color(WHITE);
    kprintln!("  {}x{} matmul, {} iterations in {} ticks", d, n, iters, elapsed);
    let path = if crate::cpu::detected().avx_enabled { "avx2+fma" } else { "scalar" };
    // Integer-formatted: printing floats needs a formatter we have not written.
    kprintln!(
        "  {}.{:02} GFLOP/s  ({})",
        gflops as u64,
        ((gflops * 100.0) as u64) % 100,
        path
    );
    // Guard against the compiler deciding the whole loop is dead.
    kprintln!("  checksum {}", (out[0] * 1000.0) as i64);
}

// --- the engine ---------------------------------------------------------

/// A loaded model, its tokenizer, and the scratch space a decode step needs.
///
/// Held as one unit because they are only ever meaningful together: the
/// tokenizer is parsed using the vocabulary size from the model header, and
/// `State` is sized from the same config.
pub struct Engine {
    pub model: model::Model,
    pub tok: tokenizer::Tokenizer,
    pub state: model::State,
    pub rng: sample::Rng,
    /// The part that is allowed to change. Everything above it is frozen.
    pub head: vocab::Head,
    /// Closed-form replacement for the gradient head. `None` until `train`
    /// fits one; the system boots without a router and gains one from its own
    /// corpus.
    pub probe: Option<probe::Probe>,
    /// Corroborating cores. They never change the probe's answer -- combining
    /// them was measured as slightly *worse* than the probe alone -- but their
    /// agreement says whether to trust it.
    pub council: Option<council::Council>,
    /// How far into the KV cache the live conversation has got. The cache
    /// alone is not a mental state -- attention reads `0..pos`, so a cache
    /// without its position is unusable.
    pub pos: usize,
    /// The last token emitted, so a resumed generation has something to feed
    /// the model when no new prompt is supplied.
    pub last_token: usize,
}

static ENGINE: Racy<Option<Engine>> = Racy::new(None);

pub fn engine_ready() -> bool {
    unsafe { ENGINE.get().is_some() }
}

/// Which task owns the engine while the mind is running.
static MIND_TASK: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Borrow the engine, or refuse if another task already holds it.
///
/// This is the only place the exclusion is enforced, and it has to be, because
/// the previous attempt enforced it at the call sites and silently did not.
/// The shell had a list of commands to refuse while the mind was busy, written
/// as a guarded match arm placed *after* the arms it was guarding -- so it was
/// unreachable, the compiler said so as "unreachable pattern", and every one of
/// `gen`, `act`, `ctx`, `fit`, `route` and `logits` could take a second
/// `&mut Engine` while the mind held one. That is undefined behaviour rather
/// than a race, and no amount of care at call sites would have kept a list like
/// that correct as commands were added.
///
/// Checking the task id rather than a flag is what lets the mind keep working
/// while everyone else is turned away.
pub fn with_engine<R>(f: impl FnOnce(&mut Engine) -> R) -> Option<R> {
    let current = crate::task::current();
    if mind_busy() && current != MIND_TASK.load(Ordering::Acquire) {
        return None;
    }
    // Same rule for the agent task, which holds the engine for whole
    // episodes. The id is recorded once at spawn and lives as long as the
    // task does; whether an episode is actually running is the flag.
    if agent::episode_busy() && current != AGENT_TASK.load(Ordering::Acquire) {
        return None;
    }
    unsafe { ENGINE.get().as_mut().map(f) }
}

/// True when the engine is unavailable because the mind has it.
pub fn engine_held_by_mind() -> bool {
    mind_busy() && crate::task::current() != MIND_TASK.load(Ordering::Acquire)
}

/// Build the engine from whatever the firmware managed to read off the ESP.
///
/// Every failure here is reported and survivable. A system that refuses to
/// boot because a model file is missing would be a worse system.
pub fn init(model_blob: Option<Blob>, tok_blob: Option<Blob>) {
    console::set_color(YELLOW);
    kprintln!("\n[ai]");
    console::set_color(LTGRAY);

    let Some(mb) = model_blob else {
        kprintln!("  no checkpoint on the boot volume ({})", crate::MODEL_PATH);
        kprintln!("  'model' still works against synthetic weights");
        return;
    };

    // GLADOSM2 first, then the llama2.c legacy layout. Both are identified by
    // content rather than by filename, so either can sit at model.bin.
    let loaded = match model::Model::from_glados(mb.as_slice()) {
        Ok(m) => Ok(m),
        Err(model::LoadError::BadHeader) => model::Model::from_bytes(mb.as_slice()),
        Err(e) => Err(e),
    };

    let m = match loaded {
        Ok(m) => m,
        Err(e) => {
            console::set_color(LTRED);
            match e {
                model::LoadError::Truncated { want, have } => {
                    kprintln!("  checkpoint truncated: header wants {} bytes, file has {}", want, have)
                }
                other => kprintln!("  checkpoint rejected: {:?}", other),
            }
            console::set_color(LTGRAY);
            return;
        }
    };

    let c = m.cfg;
    kprintln!(
        "  dim {}  hidden {}  layers {}  heads {}/{} kv  vocab {}  seq {}",
        c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size, c.seq_len
    );
    kprintln!(
        "  {} params, {} KiB of weights, {}, rope_theta {}",
        c.param_count(),
        m.weight_bytes() / 1024,
        if m.is_quantised() { "int8" } else { "f32" },
        c.rope_theta as u32
    );

    let Some(tb) = tok_blob else {
        console::set_color(LTRED);
        kprintln!("  no tokenizer ({}) -- cannot decode text", crate::TOKENIZER_PATH);
        console::set_color(LTGRAY);
        return;
    };

    // The vocabulary size is not in the v2 header's contract with the model,
    // so a mismatched pair used to parse cleanly and then feed the model ids
    // that name different rows of the embedding -- fluent nonsense, no error.
    // The parser now holds both counts against each other and says which
    // failed and by how much.
    let tok = match tokenizer::Tokenizer::from_bytes(tb.as_slice(), c.vocab_size) {
        Ok(t) => t,
        Err(e) => {
            console::set_color(LTRED);
            match e {
                tokenizer::TokError::VocabMismatch { have, want } => {
                    kprintln!(
                        "  tokenizer produces ids up to {}; this model's embedding has {} rows",
                        have, want
                    );
                    kprintln!("  (the two files are from different checkpoints)");
                }
                tokenizer::TokError::Truncated => {
                    kprintln!("  tokenizer file ends before its own structure does -- the copy to the ESP was incomplete");
                }
                tokenizer::TokError::BadVersion => {
                    kprintln!("  tokenizer version field is not one of ours");
                }
            }
            console::set_color(LTGRAY);
            return;
        }
    };

    // The state -- KV cache, RoPE tables, recurrent state -- is sized before
    // it is allocated, because an allocation that fails here must be a line
    // of red text rather than a panic with the boot half-finished. The margin
    // covers the router head, the corpus and whatever else [ai] builds after.
    let state_bytes = model::State::requirement(&c);
    let margin = 64 * 1024 * 1024;
    let (heap_used, heap_total) = crate::mem::heap::HEAP.stats();
    let heap_free = heap_total.saturating_sub(heap_used);
    if state_bytes + margin > heap_free {
        console::set_color(LTRED);
        kprintln!(
            "  context needs {} MiB of state but only {} MiB of heap is free --",
            state_bytes / (1024 * 1024),
            heap_free / (1024 * 1024)
        );
        kprintln!("  re-convert with a smaller --seq, or boot with more RAM");
        console::set_color(LTGRAY);
        return;
    }

    // Padding versus wrong checkpoint. Checkpoints round their vocabulary up
    // for alignment -- q35 pairs 248070 pieces with 248320 rows, 0.1% spare.
    // A tokenizer from another model is not that: SmolLM2's 49k beside this
    // embedding is 80% missing, loads without complaint, and generates
    // confidently from rows it was never meant to address. One percent of
    // slack separates the two cases.
    const VOCAB_SLACK_PCT: usize = 1;
    if tok.vocab_size() != c.vocab_size && c.vocab_size >= 100 {
        let missing = c.vocab_size - tok.vocab_size();
        if missing > (c.vocab_size / 100) * VOCAB_SLACK_PCT {
            console::set_color(LTRED);
            kprintln!(
                "  tokenizer covers {} of {} embedding rows -- {} short is not padding;",
                tok.vocab_size(),
                c.vocab_size,
                missing
            );
            kprintln!("  (the two files are from different checkpoints)");
            console::set_color(LTGRAY);
            return;
        }
        kprintln!(
            "  note  tokenizer covers {} of {} embedding rows (the rest are padding)",
            tok.vocab_size(),
            c.vocab_size
        );
    }

    let state = model::State::new(&c);
    let live_bytes = state.bytes(&c);
    if live_bytes != state_bytes {
        console::set_color(YELLOW);
        kprintln!(
            "  note  state estimate {} != actual {} bytes -- State::requirement has drifted",
            state_bytes, live_bytes
        );
        console::set_color(LTGRAY);
    }
    kprintln!(
        "  tokenizer {} tokens, longest {} bytes; state {} KiB",
        tok.vocab_size(),
        tok.max_token_length,
        state.bytes(&c) / 1024
    );

    // Seed from the TSC so successive boots do not retell the same story.
    let seed = crate::time::rdtsc();
    let mut rng = sample::Rng::new(seed);

    // One token per applet, each starting from the mean of the embeddings of
    // the words describing it. The base model's weights are not touched here
    // or anywhere else.
    let head = vocab::Head::for_applets(&m, &tok, &mut rng);
    kprintln!(
        "  head {} applet tokens, {} trainable params (base {} frozen)",
        head.len(),
        head.params(),
        c.param_count()
    );

    unsafe {
        *ENGINE.get() = Some(Engine {
            model: m,
            tok,
            state,
            rng,
            head,
            probe: None,
            council: None,
            pos: 0,
            last_token: tokenizer::BOS,
        })
    };

    // One real forward pass before the system trusts these weights. Every
    // failure class above is structural and loud; this one is numeric and
    // silent -- a corrupt scale, a bad dequant table or a wrong RoPE length
    // produces logits that are NaN, infinite, or flat, and a model that
    // samples confidently from whatever that is. The state is rewound after,
    // so nothing downstream sees the probe.
    let probe = with_engine(|e| {
        e.model.forward(&mut e.state, tokenizer::UNK, 0);
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut finite = true;
        for v in e.state.logits.iter() {
            if !v.is_finite() {
                finite = false;
                break;
            }
            if *v < min {
                min = *v;
            }
            if *v > max {
                max = *v;
            }
        }
        e.pos = 0;
        e.last_token = tokenizer::BOS;
        (finite, max > min)
    })
    .unwrap_or((false, false));
    if probe.0 && probe.1 {
        kprintln!("  ok   first-token logits finite and non-degenerate");
    } else {
        console::set_color(LTRED);
        kprintln!(
            "  FAIL first-token probe: {} -- weights or tables are corrupt",
            if !probe.0 { "non-finite logits" } else { "flat logits" }
        );
        console::set_color(LTGRAY);
    }

    // The corpus lives in the namespace, so it is restored along with
    // everything else; only seed it when there is nothing there.
    if crate::sysbox::children(vocab::CORPUS).is_empty() {
        for (applet, task) in corpus::SEED {
            vocab::record(applet, task);
        }
        kprintln!(
            "  corpus seeded with {} examples ({} held out) at {}",
            corpus::SEED.len(),
            corpus::SEED.len() - corpus::SEED_TRAIN,
            vocab::CORPUS
        );
    } else {
        kprintln!(
            "  corpus has {} examples at {}",
            crate::sysbox::children(vocab::CORPUS).len(),
            vocab::CORPUS
        );
    }

    console::set_color(LTGREEN);
    kprintln!("  ready -- 'gen <prompt>' to generate, 'act <task>' to choose an applet");
    console::set_color(LTGRAY);

    kprintln!();
    console::set_color(LTGREEN);
    kprintln!("[selftest] constrained decoding:");
    console::set_color(LTGRAY);
    harness::selftest();

    // The loop, driven by script rather than by the sampler. Everything
    // mechanical about an episode -- admission, argument rejection, capture,
    // termination -- checked here in milliseconds, so that a real episode's
    // only open question is whether the model chose well.
    console::set_color(LTGREEN);
    kprintln!("\n[selftest] agent loop:");
    console::set_color(LTGRAY);
    agent::selftest();

    console::set_color(LTGREEN);
    kprintln!("\n[selftest] linear probe:");
    console::set_color(LTGRAY);
    if probe::selftest() {
        console::set_color(LTGREEN);
        kprintln!("  ok   cholesky recovers a known separable fit, and refuses a singular one");
    } else {
        console::set_color(LTRED);
        kprintln!("  FAIL the probe does not solve a problem with a known answer");
    }
    console::set_color(LTGRAY);

    console::set_color(LTGREEN);
    kprintln!("\n[selftest] situation planner:");
    console::set_color(LTGRAY);
    if aixi::selftest() {
        console::set_color(LTGREEN);
        kprintln!("  ok   the fitted model is good enough to plan against a synthetic machine");
    } else {
        console::set_color(LTRED);
        kprintln!("  FAIL the planner misreads a machine with known dynamics");
    }
    console::set_color(LTGRAY);

    console::set_color(LTGREEN);
    kprintln!("\n[selftest] initiative policy:");
    console::set_color(LTGRAY);
    if initiative::selftest() {
        console::set_color(LTGREEN);
        kprintln!("  ok   the resident mind stands down for you and acts only on evidence");
    } else {
        console::set_color(LTRED);
        kprintln!("  FAIL the initiative gates misorder");
    }
    console::set_color(LTGRAY);

    console::set_color(LTGREEN);
    kprintln!("\n[selftest] qdora adapters:");
    console::set_color(LTGRAY);
    if adapter::selftest() {
        console::set_color(LTGRAY);
    } else {
        console::set_color(LTRED);
        kprintln!("  FAIL -- see the line above for which claim broke");
    }
    console::set_color(LTGRAY);

    console::set_color(LTGREEN);
    kprintln!("\n[selftest] backward kernels:");
    console::set_color(LTGRAY);
    if backward::selftest() {
        console::set_color(LTGRAY);
    } else {
        console::set_color(LTRED);
        kprintln!("  FAIL -- a gradient does not match its forward");
    }
    console::set_color(LTGRAY);

    console::set_color(LTGREEN);
    kprintln!("\n[selftest] trainer arithmetic:");
    console::set_color(LTGRAY);
    if train::selftest() {
        console::set_color(LTGRAY);
    } else {
        console::set_color(LTRED);
        kprintln!("  FAIL -- adam + cross-entropy failed to memorise a known sheet");
    }
    console::set_color(LTGRAY);
}

/// Write raw token bytes to the console.
///
/// Pieces are not individually valid UTF-8: a byte-fallback token is one
/// arbitrary byte, and a multi-byte character can straddle two tokens. So this
/// buffers and only prints what is currently decodable, keeping any trailing
/// partial sequence for the next call.
fn emit(pending: &mut Vec<u8>) {
    loop {
        if pending.is_empty() {
            return;
        }
        match core::str::from_utf8(pending) {
            Ok(s) => {
                kprint!("{}", s);
                pending.clear();
                return;
            }
            Err(e) => {
                let good = e.valid_up_to();
                if good > 0 {
                    if let Ok(s) = core::str::from_utf8(&pending[..good]) {
                        kprint!("{}", s);
                    }
                    pending.drain(..good);
                    continue;
                }
                // Not a truncated tail but a genuinely invalid byte: drop it,
                // otherwise this loops forever on the same byte.
                if e.error_len().is_some() {
                    pending.remove(0);
                    continue;
                }
                return; // incomplete tail; wait for more
            }
        }
    }
}

/// Live sampling knobs, so the penalty can be tuned against a real model at
/// 63 ms a token instead of guessed at 8 seconds a token under emulation.
static REPEAT: Racy<(f32, usize)> = Racy::new((1.1, 64));

pub fn repeat_settings() -> (f32, usize) {
    unsafe { *REPEAT.get() }
}

pub fn set_repeat(penalty: f32, window: usize) {
    unsafe { *REPEAT.get() = (penalty.max(0.0), window.min(512)) };
}

pub struct GenOpts {
    pub steps: usize,
    pub temperature: f32,
    pub topp: f32,
    /// Continue from the live KV cache instead of starting a fresh one.
    pub resume: bool,
    /// Yield the CPU between tokens. Set by the mind task so the shell keeps
    /// running while it thinks; left clear for a foreground `gen`, where
    /// yielding would only add latency.
    pub yielding: bool,
    /// Prepend the begin-of-sequence token. Cleared for ChatML, where
    /// `<|im_start|>` does the framing and an extra BOS is a token the model
    /// never saw in that position.
    pub bos: bool,
    /// Print the prompt back as it is fed. Useful for `gen`, where the
    /// continuation reads as one piece with what preceded it, and wrong for
    /// chat, where it would spray the role markers across the screen.
    pub echo_prompt: bool,
    /// Divisor applied to the logits of recently emitted tokens. 1.0 is off.
    pub repeat_penalty: f32,
    /// How many recent tokens the penalty covers.
    pub repeat_window: usize,
    /// Let a hybrid reasoning model reason before answering.
    ///
    /// Only meaningful for a checkpoint that has `<think>`. Left clear, `chat`
    /// closes the block itself, which is how Qwen3 documents turning thinking
    /// off -- and without that, `ask -n 64` spends every one of its 64 tokens
    /// thinking and prints no answer at all.
    pub think: bool,
}

impl Default for GenOpts {
    fn default() -> Self {
        // llama2.c's defaults. 0.9 top-p keeps a 260K model from wandering.
        //
        // The repetition penalty is ours and is deliberately mild. 1.1 is
        // enough to break a self-reinforcing loop; much above 1.2 and the
        // model starts avoiding words it legitimately needs, which reads as
        // incoherence rather than variety. The window is 64 for the same
        // reason -- penalise everything ever said and ordinary words become
        // unusable a paragraph in.
        let (repeat_penalty, repeat_window) = repeat_settings();
        Self {
            steps: 256,
            temperature: 1.0,
            topp: 0.9,
            resume: false,
            yielding: false,
            bos: true,
            echo_prompt: true,
            repeat_penalty,
            repeat_window,
            think: false,
        }
    }
}

/// Run the decode loop, printing as it goes.
pub fn generate(prompt: &str, opts: &GenOpts) {
    let ok = with_engine(|e| {
        // Resuming means picking up the existing cache rather than rebuilding
        // one, so no BOS: that token means "a new story begins", and inserting
        // it mid-conversation tells the model to forget what it was doing.
        let resuming = opts.resume && e.pos > 0;
        let prompt_tokens = e.tok.encode(prompt, opts.bos && !resuming, false);
        let mut pos = if resuming { e.pos } else { 0 };
        let mut token = if prompt_tokens.is_empty() {
            if !resuming {
                return false;
            }
            e.last_token
        } else {
            prompt_tokens[0]
        };

        // Without a window this is the trained length and generation stops at
        // it, as it always has. With one, the cache evicts and position is no
        // longer bounded by anything: `slot_of` wraps into the ring, and every
        // buffer below is indexed by cache position (`j`, `t` < `live`) rather
        // than by absolute position, so the forward pass was already correct
        // past `seq_len` -- this loop was the only thing stopping it.
        //
        // That is what `window_report` has been claiming all along. Until now
        // it was not true.
        let cap = if e.model.cfg.streams() { usize::MAX } else { e.model.cfg.seq_len };
        let mut generated = 0usize;
        let mut fed = 0usize;
        let mut pending: Vec<u8> = Vec::new();
        // Sampled tokens only. The prompt is not the model repeating itself,
        // and penalising it would push the model away from the very words the
        // user just asked about.
        let mut recent: Vec<usize> = Vec::with_capacity(opts.repeat_window);

        let t0 = crate::time::rdtsc();
        console::set_color(LTCYAN);

        while pos < cap && generated < opts.steps {
            e.model.forward(&mut e.state, token, pos);
            pos += 1;
            fed += 1;

            // While the prompt lasts we already know the next token; the
            // forward pass still had to happen, because it is what fills the
            // KV cache that everything after depends on.
            let from_prompt = fed < prompt_tokens.len();
            let next = if from_prompt {
                prompt_tokens[fed]
            } else {
                generated += 1;
                sample::apply_repetition_penalty(
                    &mut e.state.logits,
                    &recent,
                    opts.repeat_penalty,
                );
                let t = sample::sample(
                    &mut e.state.logits,
                    opts.temperature,
                    opts.topp,
                    &mut e.rng,
                );
                recent.push(t);
                if recent.len() > opts.repeat_window {
                    recent.remove(0);
                }
                t
            };

            // Stopping conditions differ by checkpoint and both have to be
            // honoured. llama2.c separates stories with BOS, so the model emits
            // it to mean "the end"; an instruct model closes its turn with EOS
            // -- for SmolLM2 that is <|im_end|>, and missing it means the model
            // carries on and writes the user's next message itself.
            //
            // Only for *sampled* tokens. A ChatML prompt contains both markers
            // by construction, so applying this while feeding it stopped the
            // generation at the end of the user's turn, before the assistant
            // turn had even been entered -- which looked exactly like a model
            // that had nothing to say.
            if !from_prompt && (next == e.tok.bos() || next == e.tok.eos()) {
                break;
            }

            if opts.echo_prompt || fed >= prompt_tokens.len() {
                e.tok.append_piece(token, next, &mut pending);
                emit(&mut pending);
            }
            token = next;

            if opts.yielding {
                crate::task::yield_now();
            }
        }

        // The conversation is now here. Saving a context after this captures
        // exactly what was just produced.
        e.pos = pos;
        e.last_token = token;

        emit(&mut pending);
        let elapsed = crate::time::rdtsc() - t0;
        console::set_color(LTGRAY);
        kprintln!();

        let mhz = crate::time::tsc_mhz();
        if mhz > 0 && generated > 0 {
            let us = elapsed / mhz;
            if us > 0 {
                // Reported as ms per token rather than tokens per second.
                // Integer division rendered anything slower than 1 tok/s as
                // "0 tok/s", which is exactly the range a 135M model sits in
                // under emulation -- the number was least informative where it
                // mattered most.
                let per = us / generated.max(1) as u64;
                kprintln!(
                    "  {} tokens in {}.{:03} s  ({} ms/token)",
                    generated,
                    us / 1_000_000,
                    (us % 1_000_000) / 1000,
                    per / 1000
                );
            }
        }
        true
    });

    match ok {
        None => {
            console::set_color(LTRED);
            kprintln!("  no model loaded");
            console::set_color(LTGRAY);
        }
        Some(false) => {
            console::set_color(LTRED);
            kprintln!("  empty prompt");
            console::set_color(LTGRAY);
        }
        Some(true) => {}
    }
}

// --- context as an addressable object -----------------------------------
//
// The KV cache is the model's working memory: everything it has attended to
// this conversation, and the only thing that distinguishes one mental state
// from another. On a hosted runtime it is anonymous heap inside a process, and
// it dies with that process.
//
// Here it is bytes we own, and there is already a content-addressed store
// downstairs. Writing it into the namespace therefore costs one serialisation
// and inherits everything sysbox can do -- so a conversation forks by copying
// a hash (O(1) at any size), versions with `snap`, and rolls back with `back`.
// None of that machinery had to learn what attention is.

pub const CTX_DIR: &str = "/ai/ctx";

/// Position and RNG travel with the cache.
///
/// Without the RNG a restore would put the model in the right state but on a
/// different branch of the random stream, so "load the same context twice and
/// continue" would diverge -- and the exactness of the restore would be
/// unfalsifiable. With it, that is a test.
/// Trailer: RNG state, then the last token emitted.
const CTX_TRAILER: usize = 16;

fn ctx_blob(e: &Engine) -> Vec<u8> {
    let mut out = e.state.export_kv(&e.model.cfg, e.pos);
    out.extend_from_slice(&e.rng.state().to_le_bytes());
    // `last_token` is as much a part of the state as the cache is: it is what
    // gets fed at `pos`, so restoring without it resumes from the right memory
    // with the wrong next word. The first version omitted it, and continuing
    // twice from one restored context produced two different stories -- which
    // is exactly the check that caught it.
    out.extend_from_slice(&(e.last_token as u64).to_le_bytes());
    out
}

pub use futures::{project, Projection, Branch, snapshot_now, futures_report};

pub fn ctx_save(name: &str) -> Option<usize> {
    with_engine(|e| {
        let blob = ctx_blob(e);
        let n = blob.len();
        let mut path = alloc::string::String::from(CTX_DIR);
        path.push('/');
        path.push_str(name);
        if crate::sysbox::write_blob(&path, blob) {
            Some(n)
        } else {
            None
        }
    })?
}

pub fn ctx_load(name: &str) -> Option<usize> {
    let mut path = alloc::string::String::from(CTX_DIR);
    path.push('/');
    path.push_str(name);
    let blob = crate::sysbox::read_blob(&path)?;
    if blob.len() < CTX_TRAILER {
        return None;
    }
    let split = blob.len() - CTX_TRAILER;
    with_engine(|e| {
        let cfg = e.model.cfg;
        let pos = e.state.import_kv(&cfg, &blob[..split])?;
        let mut word = [0u8; 8];
        word.copy_from_slice(&blob[split..split + 8]);
        e.rng.set_state(u64::from_le_bytes(word));
        word.copy_from_slice(&blob[split + 8..]);
        e.last_token = u64::from_le_bytes(word) as usize;
        e.pos = pos;
        Some(pos)
    })?
}

pub fn ctx_report() {
    console::set_color(YELLOW);
    kprintln!("[ctx]");
    console::set_color(LTGRAY);
    let live = with_engine(|e| (e.pos, e.model.cfg.live_cap(), e.model.cfg.streams()));
    match live {
        None => {
            kprintln!("  no model loaded");
            return;
        }
        // While streaming, position is not bounded by anything and "of N" reads
        // as a bug the moment it goes past N. Say what the cache holds instead,
        // which is the number that is actually a limit.
        Some((pos, cap, true)) => {
            kprintln!("  position {}, streaming through a {}-slot cache", pos, cap)
        }
        Some((pos, cap, false)) => kprintln!("  live position {} of {}", pos, cap),
    }
    let saved = crate::sysbox::children(CTX_DIR);
    if saved.is_empty() {
        kprintln!("  nothing saved -- 'ctx save <name>'");
        return;
    }
    for name in saved {
        let mut path = alloc::string::String::from(CTX_DIR);
        path.push('/');
        path.push_str(&name);
        let n = crate::sysbox::read_blob(&path).map(|b| b.len()).unwrap_or(0);
        kprintln!("  {:12} {} B", name, n);
    }
    console::set_color(LTGRAY);
    kprintln!("  these are ordinary objects: 'cp' one to fork it, 'same' to compare");
}

// --- the model as a resident task ---------------------------------------
//
// Until now generation was a shell command: type `gen`, and nothing else in
// the system runs until it finishes. That is the shape a hosted runtime is
// forced into, where inference is a call into a library and the caller blocks.
//
// There is a scheduler here, and `task.rs` already saves extended state across
// preemption -- that was fixed for exactly this. So the model can be a task
// like any other: resident, scheduled, yielding between tokens, with the shell
// staying responsive while it thinks.
//
// Concurrency is handled by a flag rather than a lock, and it is worth being
// explicit about why that is sufficient and where it stops being so. The
// engine is behind `Racy`, so two tasks holding `&mut Engine` at once is
// undefined behaviour, not merely a race. `BUSY` is claimed with a
// compare-exchange before the mind touches the engine and every other entry
// point refuses while it is set, so the two can never overlap. That argument
// depends on there being one core. When SMP arrives this needs a real lock,
// which is exactly what `Racy` exists to make greppable.

use core::sync::atomic::AtomicBool;

static REQUEST: Racy<Option<alloc::string::String>> = Racy::new(None);
static BUSY: AtomicBool = AtomicBool::new(false);

pub fn mind_busy() -> bool {
    BUSY.load(Ordering::Acquire)
}

/// Queue a prompt for the mind task. Returns false if one is already pending.
pub fn think(prompt: &str) -> bool {
    crate::cpu::without_interrupts(|| unsafe {
        if REQUEST.get().is_some() {
            return false;
        }
        *REQUEST.get() = Some(alloc::string::String::from(prompt));
        true
    })
}

/// The resident mind. Spawned once; never returns.
pub fn mind_task() {
    loop {
        // Taking the request has to be atomic against the shell posting one,
        // or a request can be dropped in the window between the test and the
        // take.
        let req = crate::cpu::without_interrupts(|| unsafe { REQUEST.get().take() });

        let Some(prompt) = req else {
            crate::task::yield_now();
            continue;
        };

        if BUSY
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            continue;
        }

        console::set_color(YELLOW);
        kprintln!("\n[mind] thinking...");
        console::set_color(LTGRAY);

        let opts =
            GenOpts { steps: 48, temperature: 0.9, topp: 0.9, resume: true, yielding: true,
                      ..Default::default() };
        generate(&prompt, &opts);

        console::set_color(YELLOW);
        kprintln!("[mind] done");
        console::set_color(LTGRAY);
        crate::shell::reprompt();

        BUSY.store(false, Ordering::Release);
    }
}

pub fn spawn_mind() -> bool {
    match crate::task::spawn("mind", mind_task) {
        Some(id) => {
            // Recorded before the task can possibly claim BUSY, so the
            // ownership test in `with_engine` is never consulted against a
            // stale id.
            MIND_TASK.store(id, Ordering::Release);
            true
        }
        None => false,
    }
}

/// The agent task's id, recorded at spawn. The task is resident like the
/// mind; what gates the engine is `agent::episode_busy`, not the id alone.
static AGENT_TASK: AtomicUsize = AtomicUsize::new(usize::MAX);

/// True while an episode is executing.
pub fn agent_busy() -> bool {
    agent::episode_busy()
}

/// Shell entry point: queue an episode on the resident task and return. The
/// shell stays live; output arrives from the agent task as it works, and the
/// prompt returns once more when the episode finishes.
///
/// Mutual exclusion with the mind is enforced at both command sites:
/// whichever is asked second refuses, because both would fight over the
/// engine otherwise.
pub fn agent_run(goal: &str, trust: harness::Trust, max_steps: usize) {
    if agent::queue_episode(goal, trust, max_steps) {
        console::set_color(LTGREEN);
        kprintln!("  queued -- the shell stays yours; 'agent stop' cancels");
        console::set_color(LTGRAY);
    } else {
        console::set_color(YELLOW);
        kprintln!("  an episode is already pending or running");
        console::set_color(LTGRAY);
    }
}

/// Spawn the resident agent task. Called once at boot beside `spawn_mind`.
pub fn spawn_agent() -> bool {
    match crate::task::spawn("agent", agent::agent_task) {
        Some(id) => {
            // Recorded before the task can possibly claim an episode, so the
            // ownership test in `with_engine` is never consulted against a
            // stale id -- the same ordering argument as the mind's spawn.
            AGENT_TASK.store(id, Ordering::Release);
            true
        }
        None => false,
    }
}

/// Run a raw token sequence and print the top logits.
///
/// Deliberately bypasses the tokenizer. A quantised 30-layer model has a lot
/// of places to be subtly wrong -- a stride miscomputed, sign extension
/// dropped in the AVX2 widening, the wrong rope_theta -- and every one of them
/// produces fluent-looking nonsense rather than an error. Feeding fixed ids and
/// diffing the logits against the same arithmetic done in numpy turns all of
/// that into one comparison, with no tokenizer in the way to muddy it.
pub fn logits_for(ids: &[usize]) {
    console::set_color(YELLOW);
    kprintln!("[logits]");
    console::set_color(LTGRAY);
    if ids.is_empty() {
        kprintln!("  usage: logits <id> [id ...]");
        return;
    }
    // Ids are row indexes into the embedding. The sampler cannot produce one
    // past the end, but this command accepts numbers typed by a human, and an
    // unchecked id reads garbage -- or faults -- instead of being refused.
    let vocab = with_engine(|e| e.tok.vocab_size()).unwrap_or(0);
    let mut checked = Vec::with_capacity(ids.len());
    for &id in ids {
        if id < vocab {
            checked.push(id);
        } else {
            kprintln!("  dropping id {} -- vocabulary is {} tokens", id, vocab);
        }
    }
    if checked.is_empty() {
        return;
    }

    let done = with_engine(|e| {
        let t0 = crate::time::rdtsc();
        // Through `prefill` rather than a per-token loop: this command is the
        // standing check on that path's numerics against tools/reference.py,
        // which is handed the same ids.
        let end = e.model.prefill(&mut e.state, &checked, 0);
        e.pos = end;
        let elapsed = crate::time::rdtsc() - t0;

        // Top 5 by logit, found without sorting 49152 entries.
        let mut top = [(0usize, f32::NEG_INFINITY); 5];
        for (i, v) in e.state.logits.iter().enumerate() {
            if *v > top[4].1 {
                top[4] = (i, *v);
                let mut j = 4;
                while j > 0 && top[j].1 > top[j - 1].1 {
                    top.swap(j, j - 1);
                    j -= 1;
                }
            }
        }
        (top, elapsed, checked.len())
    });

    let Some((top, elapsed, n)) = done else {
        kprintln!("  no model loaded");
        return;
    };
    for (rank, (id, v)) in top.iter().enumerate() {
        // Printed as thousandths: the console has no float formatting, and an
        // integer comparison against the reference is unambiguous anyway.
        let milli = (*v * 1000.0) as i64;
        kprintln!("  {}. id {:6}  logit {}.{:03}", rank + 1, id, milli / 1000, (milli % 1000).abs());
    }
    let mhz = crate::time::tsc_mhz();
    if mhz > 0 && n > 0 {
        let us = elapsed / mhz;
        kprintln!("  {} token(s) in {} ms  ({} us/token)", n, us / 1000, us / n as u64);
    }
}

/// Ask the model a question, framed the way it was fine-tuned.
///
/// SmolLM2-Instruct was trained on ChatML: turns delimited by `<|im_start|>`
/// and `<|im_end|>` with a role on the first line. Those markers are single
/// tokens the model has strong associations with, which is the entire reason
/// the tokenizer had to learn to match added tokens literally -- BPE'd into
/// twenty pieces they mean nothing, and the model answers as though it were
/// still completing a document.
///
/// The trailing `<|im_start|>assistant\n` is the part that actually elicits an
/// answer: it puts the model at the start of the assistant's turn, so the most
/// likely continuation is a reply rather than more of the question.
/// Whether the loaded tokenizer knows `<think>` as a single token.
///
/// The presence of the token is the signal that the checkpoint is a hybrid
/// reasoning model, which is more reliable than a name or a version: it is a
/// property of the vocabulary the model was actually trained on. Encoding is
/// the test because specials are matched literally before BPE, so a tokenizer
/// without it shreds the string into several tokens instead of one.
pub fn has_think_token() -> bool {
    with_engine(|e| e.tok.encode("<think>", false, false).len() == 1).unwrap_or(false)
}

pub fn chat(question: &str, opts: &GenOpts) {
    let mut prompt = alloc::string::String::new();
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(question);
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");

    // Qwen3 opens a `<think>` block unprompted and reasons at length. That is
    // the model working as designed and it is useless at a 64-token budget, so
    // unless thinking was asked for, the block is opened and closed here --
    // which is exactly how Qwen documents disabling it. A checkpoint without
    // the token gets nothing extra.
    if !opts.think && has_think_token() {
        prompt.push_str("<think>\n\n</think>\n\n");
    }

    let framed = GenOpts { bos: false, echo_prompt: false, ..*opts };
    generate(&prompt, &framed);
}

/// Configure the attention window.
///
/// `sinks + window >= seq_len` turns it off, which is the default and is
/// exactly the behaviour that existed before the cache learned to evict.
pub fn set_window(sinks: usize, window: usize) {
    with_engine(|e| {
        let before = e.model.cfg.live_cap();
        e.model.cfg.attn_sinks = sinks;
        e.model.cfg.attn_window = window;
        let after = e.model.cfg.live_cap();
        if after == before {
            return;
        }
        // The cache is allocated for the window, so changing it changes every
        // buffer's size. Reallocating is not optional -- the old buffers would
        // be indexed with the new capacity and read into the next layer's keys.
        //
        // The conversation cannot survive that: entries sit at positions
        // derived from the old ring, and there is no honest way to re-seat them
        // in a ring of a different size. Say so rather than silently continuing
        // from a cache that means something else now.
        e.state = model::State::new(&e.model.cfg);
        e.pos = 0;
        e.last_token = tokenizer::BOS;
        console::set_color(YELLOW);
        kprintln!(
            "  cache resized {} -> {} positions; context cleared",
            before, after
        );
        console::set_color(LTGRAY);
    });
}

pub fn window_report() {
    console::set_color(YELLOW);
    kprintln!("[window]");
    console::set_color(LTGRAY);
    let got = with_engine(|e| {
        let c = e.model.cfg;
        (
            c.attn_sinks,
            c.attn_window.min(c.seq_len),
            c.seq_len,
            c.live_cap(),
            c.streams(),
            e.state.bytes(&c),
            e.pos,
        )
    });
    let Some((sinks, window, trained, cap, streams, bytes, pos)) = got else {
        kprintln!("  no model loaded");
        return;
    };
    if !streams {
        kprintln!("  off -- the cache holds all {} trained positions and never evicts", trained);
        kprintln!("  'window <sinks> <recent>' to enable, e.g. 'window 4 128'");
    } else {
        kprintln!("  {} sink(s) + {} recent = {} live of {} trained", sinks, window, cap, trained);
        kprintln!("  input is unbounded; the window is what attention sees");
    }
    // The number that decides whether a larger window fits, next to the number
    // it would have to come out of.
    kprintln!(
        "  state {} MiB for {} positions; at {} now",
        bytes / 1024 / 1024,
        cap,
        pos
    );
}



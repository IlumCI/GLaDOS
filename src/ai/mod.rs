//! Machine learning primitives.

pub mod model;
pub mod sample;
pub mod tensor;
pub mod tokenizer;

use core::sync::atomic::{AtomicU64, Ordering};

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
        vocab_size: 256, // byte-level: no tokenizer needed
        seq_len: 64,
        shared_classifier: true,
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
}

static ENGINE: Racy<Option<Engine>> = Racy::new(None);

pub fn engine_ready() -> bool {
    unsafe { ENGINE.get().is_some() }
}

pub fn with_engine<R>(f: impl FnOnce(&mut Engine) -> R) -> Option<R> {
    unsafe { ENGINE.get().as_mut().map(f) }
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

    let m = match model::Model::from_bytes(mb.as_slice()) {
        Ok(m) => m,
        Err(e) => {
            console::set_color(LTRED);
            match e {
                model::LoadError::Truncated { want, have } => {
                    kprintln!("  checkpoint truncated: header wants {} floats, file has {}", want, have)
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
        "  {} params, {} KiB of weights",
        c.param_count(),
        m.weight_bytes() / 1024
    );

    let Some(tb) = tok_blob else {
        console::set_color(LTRED);
        kprintln!("  no tokenizer ({}) -- cannot decode text", crate::TOKENIZER_PATH);
        console::set_color(LTGRAY);
        return;
    };

    // The vocabulary size is not in the tokenizer file, so a mismatched pair
    // parses as far as it can and then runs off the end. Failing here almost
    // always means the two files came from different models.
    let Some(tok) = tokenizer::Tokenizer::from_bytes(tb.as_slice(), c.vocab_size) else {
        console::set_color(LTRED);
        kprintln!("  tokenizer does not match a {}-token vocabulary", c.vocab_size);
        console::set_color(LTGRAY);
        return;
    };

    let state = model::State::new(&c);
    kprintln!(
        "  tokenizer {} tokens, longest {} bytes; state {} KiB",
        tok.vocab_size(),
        tok.max_token_length,
        state.bytes(&c) / 1024
    );

    // Seed from the TSC so successive boots do not retell the same story.
    let seed = crate::time::rdtsc();
    unsafe { *ENGINE.get() = Some(Engine { model: m, tok, state, rng: sample::Rng::new(seed) }) };

    console::set_color(LTGREEN);
    kprintln!("  ready -- 'gen <prompt>' to generate");
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

pub struct GenOpts {
    pub steps: usize,
    pub temperature: f32,
    pub topp: f32,
}

impl Default for GenOpts {
    fn default() -> Self {
        // llama2.c's defaults. 0.9 top-p keeps a 260K model from wandering.
        Self { steps: 256, temperature: 1.0, topp: 0.9 }
    }
}

/// Run the decode loop, printing as it goes.
pub fn generate(prompt: &str, opts: &GenOpts) {
    let ok = with_engine(|e| {
        let prompt_tokens = e.tok.encode(prompt, true, false);
        if prompt_tokens.is_empty() {
            return false;
        }

        let limit = opts.steps.min(e.model.cfg.seq_len);
        let mut token = prompt_tokens[0];
        let mut pos = 0usize;
        let mut generated = 0usize;
        let mut pending: Vec<u8> = Vec::new();

        let t0 = crate::time::rdtsc();
        console::set_color(LTCYAN);

        while pos < limit {
            e.model.forward(&mut e.state, token, pos);

            // While the prompt lasts we already know the answer; the forward
            // pass is still needed, because it is what fills the KV cache.
            let next = if pos + 1 < prompt_tokens.len() {
                prompt_tokens[pos + 1]
            } else {
                generated += 1;
                sample::sample(&mut e.state.logits, opts.temperature, opts.topp, &mut e.rng)
            };
            pos += 1;

            // llama2.c's training data separates stories with BOS, so the model
            // emits it to mean "the end".
            if next == tokenizer::BOS {
                break;
            }

            e.tok.append_piece(token, next, &mut pending);
            emit(&mut pending);
            token = next;
        }

        emit(&mut pending);
        let elapsed = crate::time::rdtsc() - t0;
        console::set_color(LTGRAY);
        kprintln!();

        let mhz = crate::time::tsc_mhz();
        if mhz > 0 && generated > 0 {
            let us = elapsed / mhz;
            if us > 0 {
                kprintln!(
                    "  {} tokens in {}.{:03} s  ({} tok/s)",
                    generated,
                    us / 1_000_000,
                    (us % 1_000_000) / 1000,
                    (generated as u64 * 1_000_000) / us
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

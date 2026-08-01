//! Tensor primitives for transformer inference.
//!
//! Every operation has a scalar reference implementation and, where it pays,
//! an AVX2+FMA version selected at runtime. The scalar version is not dead
//! code kept for politeness -- it is the oracle. `selftest` runs both against
//! hand-computed values and against each other, which is the only way to know
//! a hand-written SIMD kernel is right.
//!
//! Layout convention throughout: a weight matrix `w` of shape (d, n) is stored
//! row-major, so row `i` is `w[i*n .. i*n+n]`, and `matmul` computes
//! `out[i] = dot(w_row_i, x)`. This matches llama2.c and means a converted
//! checkpoint loads with a pointer cast rather than a transpose.

#![allow(dead_code)]

use core::arch::x86_64::*;

/// Horizontal sum of eight lanes.
///
/// Done with two `hadd`s after folding the high half onto the low one. Not the
/// theoretically fastest reduction, but it is off the critical path -- it runs
/// once per output element, not once per multiply.
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(hi, lo);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// Load 32 floats into ymm0-ymm3, spin, then write them back.
///
/// A regression detector for extended-state handling in the scheduler, not a
/// useful computation. The values sit in AVX registers across a loop long
/// enough for a timer interrupt to land inside it. If a task switch fails to
/// save and restore YMM state, another task's floating point work clobbers
/// these registers and the values that come back are not the ones that went in.
///
/// It specifically exercises the *upper* halves of the YMM registers, which is
/// what `fxsave` alone would silently lose while appearing to work.
///
/// # Safety
/// Requires AVX, and that the OS has enabled AVX state in XCR0.
#[target_feature(enable = "avx")]
pub unsafe fn ymm_roundtrip(input: &[f32; 32], output: &mut [f32; 32], spin: u64) {
    unsafe {
        core::arch::asm!(
            "vmovups ymm0, [{i}]",
            "vmovups ymm1, [{i} + 32]",
            "vmovups ymm2, [{i} + 64]",
            "vmovups ymm3, [{i} + 96]",
            "2:",
            "dec {c}",
            "jnz 2b",
            "vmovups [{o}], ymm0",
            "vmovups [{o} + 32], ymm1",
            "vmovups [{o} + 64], ymm2",
            "vmovups [{o} + 96], ymm3",
            i = in(reg) input.as_ptr(),
            o = in(reg) output.as_mut_ptr(),
            c = inout(reg) spin => _,
            out("ymm0") _,
            out("ymm1") _,
            out("ymm2") _,
            out("ymm3") _,
            options(nostack),
        );
    }
}

// --- matmul -------------------------------------------------------------

/// `out[i] = dot(w[i], x)` for a (d, n) matrix. The hot loop of the whole model.
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    debug_assert!(out.len() >= d && x.len() >= n && w.len() >= n * d);
    if crate::cpu::detected().avx_enabled && crate::cpu::detected().fma {
        unsafe { matmul_avx2(out, x, w, n, d) }
    } else {
        matmul_scalar(out, x, w, n, d)
    }
}

pub fn matmul_scalar(out: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    for i in 0..d {
        let row = &w[i * n..i * n + n];
        let mut sum = 0.0f32;
        for j in 0..n {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

/// # Safety
/// Requires AVX2 and FMA, and that the OS has enabled AVX state (XCR0).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_avx2(out: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    unsafe {
        let xp = x.as_ptr();
        for i in 0..d {
            let row = w.as_ptr().add(i * n);
            // Four accumulators rather than one: FMA has ~4 cycle latency but
            // issues every cycle, so a single chain stalls on its own result.
            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut a2 = _mm256_setzero_ps();
            let mut a3 = _mm256_setzero_ps();

            let mut j = 0usize;
            while j + 32 <= n {
                a0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row.add(j)),
                    _mm256_loadu_ps(xp.add(j)),
                    a0,
                );
                a1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row.add(j + 8)),
                    _mm256_loadu_ps(xp.add(j + 8)),
                    a1,
                );
                a2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row.add(j + 16)),
                    _mm256_loadu_ps(xp.add(j + 16)),
                    a2,
                );
                a3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row.add(j + 24)),
                    _mm256_loadu_ps(xp.add(j + 24)),
                    a3,
                );
                j += 32;
            }
            while j + 8 <= n {
                a0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row.add(j)),
                    _mm256_loadu_ps(xp.add(j)),
                    a0,
                );
                j += 8;
            }

            let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            let mut sum = hsum256(acc);
            while j < n {
                sum += *row.add(j) * *xp.add(j);
                j += 1;
            }
            *out.get_unchecked_mut(i) = sum;
        }
    }
}

// --- normalisation ------------------------------------------------------

/// Root-mean-square normalisation, as used by Llama-family models.
///
/// No mean subtraction and no bias, unlike LayerNorm -- that is the whole
/// point of RMSNorm, and getting it wrong produces output that looks almost
/// right, which is worse than output that looks broken.
///
/// `eps` is a property of the checkpoint, not of the algorithm: SmolLM2 trained
/// with 1e-5 and Qwen3 with 1e-6. It sits inside the square root, so the wrong
/// one perturbs every activation in the network by a small amount rather than
/// failing anywhere -- the same class of silent error as the wrong rope_theta.
pub fn rmsnorm_eps(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0.0f32;
    for v in x.iter().take(n) {
        ss += v * v;
    }
    ss = ss / n as f32 + eps;
    let scale = 1.0 / sqrtf(ss);
    for i in 0..n {
        out[i] = weight[i] * (x[i] * scale);
    }
}

/// In-place RMSNorm over one slice, for QK-Norm.
///
/// Qwen3 normalises each attention head's query and key vectors before RoPE,
/// with a weight shared across heads. It is in place because the target is one
/// head's window into `q` or into the key cache, and copying it out and back
/// would cost more than the normalisation.
pub fn rmsnorm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0.0f32;
    for v in x.iter() {
        ss += v * v;
    }
    ss = ss / n as f32 + eps;
    let scale = 1.0 / sqrtf(ss);
    for i in 0..n {
        x[i] = weight[i] * (x[i] * scale);
    }
}

/// RMSNorm at the Llama default epsilon.
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    rmsnorm_eps(out, x, weight, 1e-5)
}

/// In-place softmax, shifted by the maximum for numerical stability.
///
/// Without the shift, `exp` of a large logit overflows to infinity and the
/// whole distribution becomes NaN. Costs one extra pass, prevents a class of
/// bug that only appears on confident predictions.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = x[0];
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = expf(*v - max);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// SwiGLU feed-forward activation: `out = silu(out) * gate`.
pub fn swiglu(out: &mut [f32], gate: &[f32]) {
    for i in 0..out.len() {
        let v = out[i];
        out[i] = (v / (1.0 + expf(-v))) * gate[i];
    }
}

/// Rotary position embedding, applied in place to a head-major buffer.
pub fn rope(x: &mut [f32], head_size: usize, pos: usize, theta: f32) {
    let heads = x.len() / head_size;
    for h in 0..heads {
        let base = h * head_size;
        let mut i = 0;
        while i + 1 < head_size {
            let freq = 1.0 / powf(theta, i as f32 / head_size as f32);
            let angle = pos as f32 * freq;
            let (s, c) = (sinf(angle), cosf(angle));
            let a = x[base + i];
            let b = x[base + i + 1];
            x[base + i] = a * c - b * s;
            x[base + i + 1] = a * s + b * c;
            i += 2;
        }
    }
}

pub fn add_into(dst: &mut [f32], src: &[f32]) {
    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

pub fn scale(dst: &mut [f32], k: f32) {
    for v in dst.iter_mut() {
        *v *= k;
    }
}

pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

// --- math ---------------------------------------------------------------
//
// `core` has no `f32::exp` or `sqrt` in a freestanding build: those live in
// `std` and bottom out in libm, which we do not have. These are small
// implementations, accurate enough for inference, where a 1e-6 error in a
// logit changes nothing.

pub fn sqrtf(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    // sqrtss is a single instruction and exactly correct; no reason to
    // approximate when SSE2 is guaranteed.
    unsafe { _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x))) }
}

/// exp via range reduction: exp(x) = 2^k * exp(r), with r small.
pub fn expf(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > 88.0 {
        return f32::INFINITY;
    }
    if x < -88.0 {
        return 0.0;
    }
    const LOG2E: f32 = 1.442_695_f32;
    const LN2: f32 = 0.693_147_2_f32;

    let k = floorf(x * LOG2E + 0.5);
    let r = x - k * LN2;
    // Degree-5 Taylor of exp(r) on |r| <= ln2/2; error well under 1e-7.
    let p = 1.0
        + r * (1.0
            + r * (0.5 + r * (0.166_666_67 + r * (0.041_666_67 + r * 0.008_333_33))));
    p * exp2i(k as i32)
}

/// 2^n by constructing the float exponent directly.
fn exp2i(n: i32) -> f32 {
    if n < -126 {
        return 0.0;
    }
    if n > 127 {
        return f32::INFINITY;
    }
    f32::from_bits((((n + 127) as u32) & 0xFF) << 23)
}

pub fn floorf(x: f32) -> f32 {
    let t = x as i64 as f32;
    if t > x {
        t - 1.0
    } else {
        t
    }
}

pub fn powf(base: f32, exp: f32) -> f32 {
    if base <= 0.0 {
        return 0.0;
    }
    expf(exp * lnf(base))
}

/// ln via exponent extraction plus an atanh series on the mantissa.
pub fn lnf(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    // Mantissa forced into [1, 2).
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    // 2*atanh(t) = ln(m), converges fast because |t| <= 1/3.
    let s = 2.0 * t * (1.0 + t2 * (1.0 / 3.0 + t2 * (0.2 + t2 * (1.0 / 7.0))));
    s + e as f32 * 0.693_147_2
}

pub fn sinf(x: f32) -> f32 {
    const TWO_PI: f32 = 6.283_185_5;
    let mut a = x - TWO_PI * floorf(x / TWO_PI + 0.5);
    // a now in roughly [-pi, pi]; fold to [-pi/2, pi/2] for the series.
    if a > core::f32::consts::FRAC_PI_2 {
        a = core::f32::consts::PI - a;
    } else if a < -core::f32::consts::FRAC_PI_2 {
        a = -core::f32::consts::PI - a;
    }
    let a2 = a * a;
    a * (1.0 - a2 * (1.0 / 6.0 - a2 * (1.0 / 120.0 - a2 / 5040.0)))
}

pub fn cosf(x: f32) -> f32 {
    sinf(x + core::f32::consts::FRAC_PI_2)
}

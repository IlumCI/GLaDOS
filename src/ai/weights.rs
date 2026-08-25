//! Weight matrices that may be f32 or int8, read in place.
//!
//! The checkpoint is already in RAM when this runs -- `uefi::read_file` put it
//! in a LoaderData pool allocation before ExitBootServices, which the identity
//! map covers write-back. So nothing is copied: a `Mat` is a view into those
//! bytes. For a 135 MB model that saves both the copy and the 135 MB of heap
//! it would have needed.
//!
//! int8 data needs no alignment (`i8` is align 1), so the hot path casts
//! straight from the blob. The f32 side -- per-row scales, and the norm
//! weights -- is read with `from_le_bytes` instead of being cast, because a
//! misaligned `&[f32]` is undefined behaviour and the alignment of any
//! particular tensor is an accident of the shapes above it. Scales are read
//! once per output row rather than once per weight, so the cost is noise.
//!
//! Dequantisation is `sum(w[j] * x[j]) * scale[row]`, not
//! `sum(w[j] * scale * x[j])`: the scale is constant across the row, so it
//! belongs outside the loop. One multiply per output element instead of one
//! per weight.

use super::tensor;

#[derive(Clone, Copy)]
pub enum Mat<'a> {
    /// Row-major f32, `rows * cols` values.
    F32 { data: &'a [f32], rows: usize, cols: usize },
    /// Row-major int8 with one f32 scale per row.
    Q8 { data: &'a [i8], scales: &'a [u8], rows: usize, cols: usize },
}

#[inline]
pub fn f32_at(bytes: &[u8], i: usize) -> f32 {
    let o = i * 4;
    f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
}

impl Mat<'_> {
    /// `out = self * x`. `out` must have `rows` entries, `x` must have `cols`.
    pub fn matvec(&self, out: &mut [f32], x: &[f32]) {
        match self {
            Mat::F32 { data, rows, cols } => {
                tensor::matmul(&mut out[..*rows], &x[..*cols], data, *cols, *rows)
            }
            Mat::Q8 { data, scales, rows, cols } => {
                let f = crate::cpu::detected();
                // All three, and `avx2` is not optional: the kernel below is
                // `target_feature(avx2,fma)` and uses `_mm256_cvtepi8_epi32`,
                // which is an AVX2 instruction. Gating on AVX alone would take
                // a #UD on any part with AVX and FMA but no AVX2 -- AMD's
                // Piledriver, for one. `avx_enabled` is separate again and
                // means the OS has actually set CR4.OSXSAVE and XCR0, without
                // which every one of these faults regardless of what CPUID
                // advertises.
                if f.avx_enabled && f.avx2 && f.fma {
                    unsafe { q8_matvec_avx2(out, x, data, scales, *rows, *cols) }
                } else {
                    q8_matvec_scalar(out, x, data, scales, *rows, *cols)
                }
            }
        }
    }

    /// Copy row `r` out as f32. Used for the embedding lookup, which is a row
    /// fetch rather than a matrix-vector product.
    pub fn row_into(&self, r: usize, out: &mut [f32]) {
        match self {
            Mat::F32 { data, cols, .. } => {
                out[..*cols].copy_from_slice(&data[r * cols..(r + 1) * cols]);
            }
            Mat::Q8 { data, scales, cols, .. } => {
                let s = f32_at(scales, r);
                let row = &data[r * cols..(r + 1) * cols];
                for (o, v) in out[..*cols].iter_mut().zip(row.iter()) {
                    *o = *v as f32 * s;
                }
            }
        }
    }

    /// `out[t*rows + r] = row r . xs[t*cols ..]` for `tc` input rows.
    ///
    /// The whole point of this shape is what crosses the memory bus once:
    /// prefilled as `tc` separate matvecs, a prompt of N tokens streams every
    /// weight byte N times, which under TCG means re-translating the same
    /// 132 MiB of working set N times and on hardware means paying DRAM
    /// bandwidth N times for weights that were already in cache the second
    /// time around. Row-major outer order streams each weight row once and
    /// reuses it against every input row while it is hot; per output element
    /// the additions still run over `j` ascending into one accumulator, so
    /// results are bit-identical to calling `matvec` per position.
    pub fn matvec_batch(&self, out: &mut [f32], xs: &[f32], tc: usize) {
        match self {
            Mat::F32 { data, rows, cols } => {
                // Flat checkpoints are the tiny llama2.c test models; their
                // f32 path never got a weight-stationary kernel because there
                // was nothing for it to save. Per-position reuse keeps one
                // code path for the arithmetic.
                for t in 0..tc {
                    let x = &xs[t * *cols..(t + 1) * *cols];
                    let o = &mut out[t * *rows..(t + 1) * *rows];
                    tensor::matmul(o, x, data, *cols, *rows);
                }
            }
            Mat::Q8 { data, scales, rows, cols } => {
                let f = crate::cpu::detected();
                // Same gate as `matvec`: AVX2 is not optional (see above), and
                // `avx_enabled` means the OS actually enabled the state.
                if f.avx_enabled && f.avx2 && f.fma {
                    unsafe { q8_matvec_batch_avx2(out, xs, data, scales, *rows, *cols, tc) }
                } else {
                    q8_matvec_batch_scalar(out, xs, data, scales, *rows, *cols, tc)
                }
            }
        }
    }
}

pub fn q8_matvec_scalar(
    out: &mut [f32],
    x: &[f32],
    data: &[i8],
    scales: &[u8],
    rows: usize,
    cols: usize,
) {
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for j in 0..cols {
            acc += row[j] as f32 * x[j];
        }
        out[r] = acc * f32_at(scales, r);
    }
}

/// AVX2 int8 matrix-vector.
///
/// Sixteen weights per iteration: a 128-bit load of int8, widened to i32 in
/// two 256-bit registers, converted to f32, then two FMAs against x. The
/// widening is the expensive part and is why this is nowhere near sixteen
/// times the scalar version -- but the memory traffic is a quarter of f32's,
/// and that is what actually bounds generation.
///
/// # Safety
/// Requires AVX2 and FMA. Callers check `cpu::detected()`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn q8_matvec_avx2(
    out: &mut [f32],
    x: &[f32],
    data: &[i8],
    scales: &[u8],
    rows: usize,
    cols: usize,
) {
    use core::arch::x86_64::*;

    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let chunks = cols / 16;
        for c in 0..chunks {
            let base = c * 16;
            let packed = _mm_loadu_si128(row.as_ptr().add(base) as *const __m128i);
            // Sign-extend: cvtepi8_epi32 takes the low 8 bytes, so the high
            // half is shifted down first. Using an unsigned widen here would
            // turn every negative weight into a large positive one, which
            // looks like a plausible model that generates nonsense.
            let lo = _mm256_cvtepi8_epi32(packed);
            let hi = _mm256_cvtepi8_epi32(_mm_srli_si128(packed, 8));
            let lof = _mm256_cvtepi32_ps(lo);
            let hif = _mm256_cvtepi32_ps(hi);
            let x0 = _mm256_loadu_ps(x.as_ptr().add(base));
            let x1 = _mm256_loadu_ps(x.as_ptr().add(base + 8));
            acc0 = _mm256_fmadd_ps(lof, x0, acc0);
            acc1 = _mm256_fmadd_ps(hif, x1, acc1);
        }

        let sum = _mm256_add_ps(acc0, acc1);
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
        let mut total = lanes.iter().sum::<f32>();

        // Ragged tail. Every dimension in the models here is a multiple of 16,
        // so this is usually dead -- which is exactly why it has to be correct
        // rather than merely present.
        for j in chunks * 16..cols {
            total += row[j] as f32 * x[j];
        }

        out[r] = total * f32_at(scales, r);
    }
}

/// Weight-stationary batch of [`q8_matvec_scalar`]: row-major outer loop, one
/// accumulator per input position, additions over `j` ascending -- the same
/// order, so the same bits, as the per-position kernel.
pub fn q8_matvec_batch_scalar(
    out: &mut [f32],
    xs: &[f32],
    data: &[i8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    tc: usize,
) {
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let scale = f32_at(scales, r);
        for t in 0..tc {
            let x = &xs[t * cols..(t + 1) * cols];
            let mut acc = 0.0f32;
            for j in 0..cols {
                acc += row[j] as f32 * x[j];
            }
            out[t * rows + r] = acc * scale;
        }
    }
}

/// Weight-stationary batch of [`q8_matvec_avx2`].
///
/// The lane structure per (row, position) is exactly the single-position
/// kernel's -- two accumulators over 16-wide chunks, horizontal add, ragged
/// tail -- so results match it bit for bit. What is hoisted is nothing that
/// changes that order: the weight bytes are walked once per row instead of
/// once per row per position.
///
/// # Safety
/// Requires AVX2 and FMA. Callers check `cpu::detected()`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn q8_matvec_batch_avx2(
    out: &mut [f32],
    xs: &[f32],
    data: &[i8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    tc: usize,
) {
    use core::arch::x86_64::*;

    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let scale = f32_at(scales, r);
        let chunks = cols / 16;
        for t in 0..tc {
            let x = &xs[t * cols..(t + 1) * cols];
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();

            for c in 0..chunks {
                let base = c * 16;
                let packed = _mm_loadu_si128(row.as_ptr().add(base) as *const __m128i);
                let lo = _mm256_cvtepi8_epi32(packed);
                let hi = _mm256_cvtepi8_epi32(_mm_srli_si128(packed, 8));
                let lof = _mm256_cvtepi32_ps(lo);
                let hif = _mm256_cvtepi32_ps(hi);
                let x0 = _mm256_loadu_ps(x.as_ptr().add(base));
                let x1 = _mm256_loadu_ps(x.as_ptr().add(base + 8));
                acc0 = _mm256_fmadd_ps(lof, x0, acc0);
                acc1 = _mm256_fmadd_ps(hif, x1, acc1);
            }

            let sum = _mm256_add_ps(acc0, acc1);
            let mut lanes = [0.0f32; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
            let mut total = lanes.iter().sum::<f32>();

            for j in chunks * 16..cols {
                total += row[j] as f32 * x[j];
            }

            out[t * rows + r] = total * scale;
        }
    }
}

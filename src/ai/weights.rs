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
    pub fn rows(&self) -> usize {
        match self {
            Mat::F32 { rows, .. } | Mat::Q8 { rows, .. } => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Mat::F32 { cols, .. } | Mat::Q8 { cols, .. } => *cols,
        }
    }

    /// `out = self * x`. `out` must have `rows` entries, `x` must have `cols`.
    pub fn matvec(&self, out: &mut [f32], x: &[f32]) {
        match self {
            Mat::F32 { data, rows, cols } => {
                tensor::matmul(&mut out[..*rows], &x[..*cols], data, *cols, *rows)
            }
            Mat::Q8 { data, scales, rows, cols } => {
                let f = crate::cpu::detected();
                if f.avx_enabled && f.fma {
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
        let mut acc0 = unsafe { _mm256_setzero_ps() };
        let mut acc1 = unsafe { _mm256_setzero_ps() };

        let chunks = cols / 16;
        for c in 0..chunks {
            let base = c * 16;
            let packed = unsafe { _mm_loadu_si128(row.as_ptr().add(base) as *const __m128i) };
            // Sign-extend: cvtepi8_epi32 takes the low 8 bytes, so the high
            // half is shifted down first. Using an unsigned widen here would
            // turn every negative weight into a large positive one, which
            // looks like a plausible model that generates nonsense.
            let lo = unsafe { _mm256_cvtepi8_epi32(packed) };
            let hi = unsafe { _mm256_cvtepi8_epi32(_mm_srli_si128(packed, 8)) };
            let lof = unsafe { _mm256_cvtepi32_ps(lo) };
            let hif = unsafe { _mm256_cvtepi32_ps(hi) };
            let x0 = unsafe { _mm256_loadu_ps(x.as_ptr().add(base)) };
            let x1 = unsafe { _mm256_loadu_ps(x.as_ptr().add(base + 8)) };
            acc0 = unsafe { _mm256_fmadd_ps(lof, x0, acc0) };
            acc1 = unsafe { _mm256_fmadd_ps(hif, x1, acc1) };
        }

        let sum = unsafe { _mm256_add_ps(acc0, acc1) };
        let mut lanes = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sum) };
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

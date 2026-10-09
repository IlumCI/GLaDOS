//! Premultiplied alpha, and the four ways one row of pixels meets another.
//!
//! Q1 of the pixel pipeline. Until now a pixel was `Color` -- three bytes, no
//! alpha -- and every blend in the tree was an 8-bit `>> 8` that darkens by one
//! part in 256 every time it runs. Glass, soft shadows, anti-aliased edges and
//! fades are all alpha, so they all start here.
//!
//! ### The pixel
//!
//! A `u32` in the **screen's own byte order** with alpha in the top byte, where
//! the framebuffer keeps the byte it ignores. So a premultiplied pixel is the
//! word the screen wants with one more byte filled in, and an opaque one *is*
//! that word -- nothing converts on the way out. Both formats the firmware can
//! hand over (`Bgrx`, `Rgbx`) keep that byte on top, which is why every
//! operation here can be channel-wise without knowing which order it is in.
//!
//! **Premultiplied**, because OVER is then one multiply-add per channel with no
//! division, and because a premultiplied pixel can be filtered, scaled and
//! blurred without dark fringes -- straight alpha averaged with transparent
//! black goes grey. Hyprland and every compositor since Porter-Duff do it this
//! way.
//!
//! ### Exact division by 255
//!
//! `div255(x) = (x + 128 + ((x + 128) >> 8)) >> 8` is `round(x / 255)` for every
//! `x` in `0..=65025`, which is every product of two bytes. `>> 8` alone divides
//! by 256: a white pixel faded twice by a full-opacity mask comes back 254, and
//! that is the "slightly grey where it should be white" every blend here used
//! to leave behind. The claim checks all 65,026 inputs.
//!
//! ### AVX2, and the scalar twin it must equal
//!
//! Each operation has a scalar form and an AVX2 form doing eight pixels at a
//! time in 16-bit lanes, chosen at run time (`cpu::detected`). **They must agree
//! bit for bit** and the suite checks every width from 0 to 33 -- both ends of
//! the eight-pixel stride and every remainder -- on patterns that reach both
//! extremes of every channel. A tolerance would hide exactly the bug worth
//! finding, the same argument `smp.rs` makes about a split matvec.

/// Exact `round(x / 255)` for `x` in `0..=65025`.
#[inline]
pub const fn div255(x: u32) -> u32 {
    (x + 128 + ((x + 128) >> 8)) >> 8
}

/// An opaque screen word with an alpha, premultiplied: every colour channel
/// scaled by `a`. `raw` is a word from `Framebuffer::encode`.
#[inline]
pub const fn premul(raw: u32, a: u8) -> u32 {
    let a32 = a as u32;
    let c0 = div255((raw & 0xFF) * a32);
    let c1 = div255(((raw >> 8) & 0xFF) * a32);
    let c2 = div255(((raw >> 16) & 0xFF) * a32);
    c0 | c1 << 8 | c2 << 16 | a32 << 24
}

/// A fully opaque pixel from a screen word.
#[inline]
pub const fn opaque(raw: u32) -> u32 {
    raw | 0xFF00_0000
}

#[inline]
fn avx2() -> bool {
    let f = crate::cpu::detected();
    f.avx_enabled && f.avx2
}

// ------------------------------------------------------------- scalar forms

/// `s OVER d` for one premultiplied pixel.
#[inline]
pub fn over_px(s: u32, d: u32) -> u32 {
    let inv = 255 - (s >> 24);
    let mut out = 0;
    for sh in [0, 8, 16, 24] {
        let sc = (s >> sh) & 0xFF;
        let dc = (d >> sh) & 0xFF;
        out |= (sc + div255(dc * inv)) << sh;
    }
    out
}

/// Each channel of `p` scaled by `m / 255`.
#[inline]
pub fn scale_px(p: u32, m: u32) -> u32 {
    let mut out = 0;
    for sh in [0, 8, 16, 24] {
        out |= div255(((p >> sh) & 0xFF) * m) << sh;
    }
    out
}

/// Each channel of `p` times the matching channel of `f`, over 255.
#[inline]
pub fn mul_px(p: u32, f: u32) -> u32 {
    let mut out = 0;
    for sh in [0, 8, 16, 24] {
        out |= div255(((p >> sh) & 0xFF) * ((f >> sh) & 0xFF)) << sh;
    }
    out
}

fn over_scalar(dst: &mut [u32], src: &[u32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = over_px(s, *d);
    }
}

fn over_mask_scalar(dst: &mut [u32], mask: &[u8], color: u32) {
    for (d, &m) in dst.iter_mut().zip(mask) {
        *d = over_px(scale_px(color, m as u32), *d);
    }
}

fn over_solid_scalar(dst: &mut [u32], color: u32) {
    for d in dst.iter_mut() {
        *d = over_px(color, *d);
    }
}

fn multiply_scalar(dst: &mut [u32], factor: u32) {
    for d in dst.iter_mut() {
        *d = mul_px(*d, factor);
    }
}

// --------------------------------------------------------------- the API

/// `src OVER dst`, premultiplied, pixel by pixel. Lengths are the shorter.
pub fn over(dst: &mut [u32], src: &[u32]) {
    let n = dst.len().min(src.len());
    if avx2() {
        unsafe { avx::over(&mut dst[..n], &src[..n]) }
    } else {
        over_scalar(&mut dst[..n], &src[..n])
    }
}

/// A premultiplied `color` laid over `dst` through a coverage mask: what an
/// anti-aliased glyph or an edge is. Lengths are the shorter.
pub fn over_mask(dst: &mut [u32], mask: &[u8], color: u32) {
    let n = dst.len().min(mask.len());
    if avx2() {
        unsafe { avx::over_mask(&mut dst[..n], &mask[..n], color) }
    } else {
        over_mask_scalar(&mut dst[..n], &mask[..n], color)
    }
}

/// A premultiplied `color` laid over every pixel of `dst`: a translucent fill.
pub fn over_solid(dst: &mut [u32], color: u32) {
    if color >> 24 == 0xFF {
        dst.fill(color);
        return;
    }
    if color >> 24 == 0 && color == 0 {
        return;
    }
    if avx2() {
        unsafe { avx::over_solid(dst, color) }
    } else {
        over_solid_scalar(dst, color)
    }
}

/// Every channel of `dst` times the matching channel of `factor`, over 255: a
/// dim (`factor` grey) or a tint (`factor` coloured). Alpha is scaled by
/// `factor`'s top byte, so pass it as 255 to leave coverage alone.
pub fn multiply(dst: &mut [u32], factor: u32) {
    if factor == 0xFFFF_FFFF {
        return;
    }
    if avx2() {
        unsafe { avx::multiply(dst, factor) }
    } else {
        multiply_scalar(dst, factor)
    }
}

mod avx {
    use core::arch::x86_64::*;

    /// Exact `round(x / 255)` on sixteen 16-bit lanes, each at most 65025.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn div255(x: __m256i) -> __m256i {
        let t = _mm256_add_epi16(x, _mm256_set1_epi16(128));
        _mm256_srli_epi16(_mm256_add_epi16(t, _mm256_srli_epi16(t, 8)), 8)
    }

    /// The alpha of each pixel copied into all four of its 16-bit lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn alphas(x: __m256i) -> __m256i {
        // Lane 3 of each group of four, both halves of each 128-bit line.
        _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(x, 0xFF), 0xFF)
    }

    /// `s + d * (255 - sa) / 255` on 16-bit lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn over16(s: __m256i, d: __m256i) -> __m256i {
        let inv = _mm256_sub_epi16(_mm256_set1_epi16(255), alphas(s));
        _mm256_add_epi16(s, div255(_mm256_mullo_epi16(d, inv)))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn lo(x: __m256i) -> __m256i {
        _mm256_unpacklo_epi8(x, _mm256_setzero_si256())
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hi(x: __m256i) -> __m256i {
        _mm256_unpackhi_epi8(x, _mm256_setzero_si256())
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn over(dst: &mut [u32], src: &[u32]) {
        unsafe {
            let n = dst.len();
            let mut i = 0;
            while i + 8 <= n {
                let s = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
                let d = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
                let r = _mm256_packus_epi16(over16(lo(s), lo(d)), over16(hi(s), hi(d)));
                _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, r);
                i += 8;
            }
            super::over_scalar(&mut dst[i..], &src[i..]);
        }
    }

    /// Eight coverage bytes, each spread across its pixel's four bytes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn spread(mask: *const u8) -> __m256i {
        unsafe {
            let m = _mm_loadl_epi64(mask as *const __m128i);
            _mm256_mullo_epi32(_mm256_cvtepu8_epi32(m), _mm256_set1_epi32(0x0101_0101))
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn over_mask(dst: &mut [u32], mask: &[u8], color: u32) {
        unsafe {
            let n = dst.len();
            let c = _mm256_set1_epi32(color as i32);
            let (clo, chi) = (lo(c), hi(c));
            let mut i = 0;
            while i + 8 <= n {
                let m = spread(mask.as_ptr().add(i));
                let s_lo = div255(_mm256_mullo_epi16(clo, lo(m)));
                let s_hi = div255(_mm256_mullo_epi16(chi, hi(m)));
                let d = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
                let r = _mm256_packus_epi16(over16(s_lo, lo(d)), over16(s_hi, hi(d)));
                _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, r);
                i += 8;
            }
            super::over_mask_scalar(&mut dst[i..], &mask[i..], color);
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn over_solid(dst: &mut [u32], color: u32) {
        unsafe {
            let n = dst.len();
            let c = _mm256_set1_epi32(color as i32);
            let (clo, chi) = (lo(c), hi(c));
            let mut i = 0;
            while i + 8 <= n {
                let d = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
                let r = _mm256_packus_epi16(over16(clo, lo(d)), over16(chi, hi(d)));
                _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, r);
                i += 8;
            }
            super::over_solid_scalar(&mut dst[i..], color);
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn multiply(dst: &mut [u32], factor: u32) {
        unsafe {
            let n = dst.len();
            let f = _mm256_set1_epi32(factor as i32);
            let (flo, fhi) = (lo(f), hi(f));
            let mut i = 0;
            while i + 8 <= n {
                let d = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
                let r = _mm256_packus_epi16(
                    div255(_mm256_mullo_epi16(lo(d), flo)),
                    div255(_mm256_mullo_epi16(hi(d), fhi)),
                );
                _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, r);
                i += 8;
            }
            super::multiply_scalar(&mut dst[i..], factor);
        }
    }
}

/// `diag blend`: the arithmetic exact, and the fast forms equal to the slow.
pub fn selftest() -> bool {
    use crate::kprintln;
    use alloc::vec::Vec;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };

    let exact = (0..=65025u32).all(|x| {
        // round(x / 255) by integer arithmetic: halves round up.
        div255(x) == (2 * x + 255) / 510
    });
    claim("div255 is round(x/255) for every product of two bytes", exact);
    claim("white faded by full coverage stays white", scale_px(0xFFFF_FFFF, 255) == 0xFFFF_FFFF);
    claim("an opaque source replaces the destination", over_px(0xFF12_3456, 0xFFAB_CDEF) == 0xFF12_3456);
    claim("a transparent source leaves it alone", over_px(0, 0xFFAB_CDEF) == 0xFFAB_CDEF);
    claim("half black over white is mid grey, not dark grey",
          over_px(premul(0x0000_0000, 128), 0xFFFF_FFFF) == 0xFF7F_7F7F);
    claim("premultiplying keeps the alpha and scales the colour",
          premul(0x00FF_8000, 128) == 0x8080_4000);

    // A pattern reaching both ends of every channel, with every alpha from
    // transparent to opaque represented, so a lane mix-up or an overflow in
    // the 16-bit arithmetic cannot hide.
    let pattern = |n: usize, seed: u32| -> Vec<u32> {
        let mut x = seed.wrapping_mul(2_654_435_761) | 1;
        (0..n)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                match i % 7 {
                    0 => 0xFFFF_FFFF,
                    1 => 0,
                    2 => 0xFF00_0000,
                    _ => {
                        // Premultiplied: no channel exceeds its alpha.
                        let a = x >> 24;
                        let c = |v: u32| (v & 0xFF) * a / 255;
                        c(x) | c(x >> 8) << 8 | c(x >> 16) << 16 | a << 24
                    }
                }
            })
            .collect()
    };
    let fast = avx2();
    let mut agree = true;
    for n in 0..=33usize {
        let src = pattern(n, 7 + n as u32);
        let dst0 = pattern(n, 99 + n as u32);
        let mask: Vec<u8> = (0..n).map(|i| [0u8, 255, 128, 1, 254, 77][i % 6]).collect();
        let color = premul(0x0012_3456, 200);
        for op in 0..4 {
            let mut a = dst0.clone();
            let mut b = dst0.clone();
            match op {
                0 => {
                    over(&mut a, &src);
                    over_scalar(&mut b, &src);
                }
                1 => {
                    over_mask(&mut a, &mask, color);
                    over_mask_scalar(&mut b, &mask, color);
                }
                2 => {
                    over_solid(&mut a, color);
                    over_solid_scalar(&mut b, color);
                }
                _ => {
                    multiply(&mut a, 0xFFC0_8040);
                    multiply_scalar(&mut b, 0xFFC0_8040);
                }
            }
            agree &= a == b;
        }
    }
    if fast {
        claim("AVX2 equals scalar, bit for bit, at every width from 0 to 33, all four ops", agree);
    } else {
        claim("no AVX2 here, so the scalar forms ran (and agree with themselves)", agree);
    }
    ok
}

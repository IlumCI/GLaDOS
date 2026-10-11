//! A radix-2 FFT, for the envelope the style stage takes off Kokoro's audio
//! and puts back under its own source.
//!
//! One size matters, 1024 points at 24 kHz, and nothing here is tuned
//! beyond being correct: a sentence is a few hundred frames and a frame
//! is one forward and one inverse transform, which is nothing beside the
//! vocoder that produced it. The twiddles are computed once per call from
//! the kernel's own `sinf`/`cosf`; a table would save microseconds.
//!
//! `rfft` takes `n` reals to `n/2 + 1` complex bins; `irfft` is its
//! inverse, so `irfft(rfft(x)) == x` to float precision, which is the one
//! claim the module makes about itself.

use crate::ai::tensor;
use alloc::vec;
use alloc::vec::Vec;

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct C {
    pub re: f32,
    pub im: f32,
}

impl C {
    pub fn new(re: f32, im: f32) -> C {
        C { re, im }
    }
    pub fn mul(self, o: C) -> C {
        C { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
    pub fn abs2(self) -> f32 {
        self.re * self.re + self.im * self.im
    }
}

/// Sine and cosine to float precision. `tensor::sinf` is a short series
/// good to about 1e-4, which is fine for a Snake activation and is not
/// fine for twiddles: 1e-4 in each of a thousand multiplies is a
/// transform that does not invert. Quadrant reduction, then Taylor to
/// the eleventh power on |x| <= pi/4, whose next term is 2e-9.
pub fn sincos(x: f32) -> (f32, f32) {
    let q = tensor::roundf(x / core::f32::consts::FRAC_PI_2);
    let r = x - q * core::f32::consts::FRAC_PI_2;
    let r2 = r * r;
    let s = r * (1.0 + r2 * (-1.0 / 6.0 + r2 * (1.0 / 120.0 + r2 * (-1.0 / 5040.0 + r2 * (1.0 / 362_880.0 + r2 * (-1.0 / 39_916_800.0))))));
    let c = 1.0 + r2 * (-0.5 + r2 * (1.0 / 24.0 + r2 * (-1.0 / 720.0 + r2 * (1.0 / 40_320.0 + r2 * (-1.0 / 3_628_800.0)))));
    match (q as i64).rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// In-place complex FFT; `n` must be a power of two. `inverse` scales by
/// `1/n` so the pair is an identity. Twiddles come from a table filled by
/// `sincos` for this call, indexed and never recurred, so the last
/// butterfly is as exact as the first.
pub fn fft(x: &mut [C], inverse: bool) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let sign = if inverse { 1.0 } else { -1.0 };
    let tw: Vec<C> = (0..n / 2)
        .map(|k| {
            let (s, c) = sincos(core::f32::consts::TAU * k as f32 / n as f32);
            C::new(c, sign * s)
        })
        .collect();
    // bit reversal
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            x.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let step = n / len;
        let mut i = 0;
        while i < n {
            for k in 0..len / 2 {
                let w = tw[k * step];
                let u = x[i + k];
                let v = x[i + k + len / 2].mul(w);
                x[i + k] = C::new(u.re + v.re, u.im + v.im);
                x[i + k + len / 2] = C::new(u.re - v.re, u.im - v.im);
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        let s = 1.0 / n as f32;
        for v in x.iter_mut() {
            v.re *= s;
            v.im *= s;
        }
    }
}

/// `n` reals to `n/2 + 1` bins.
pub fn rfft(x: &[f32]) -> Vec<C> {
    let n = x.len();
    let mut buf: Vec<C> = x.iter().map(|&v| C::new(v, 0.0)).collect();
    fft(&mut buf, false);
    buf.truncate(n / 2 + 1);
    buf
}

/// `n/2 + 1` bins to `n` reals, assuming Hermitian symmetry.
pub fn irfft(bins: &[C], n: usize) -> Vec<f32> {
    let mut buf = vec![C::default(); n];
    for k in 0..=n / 2 {
        buf[k] = bins[k];
    }
    for k in 1..n / 2 {
        buf[n - k] = C::new(bins[k].re, -bins[k].im);
    }
    fft(&mut buf, true);
    buf.iter().map(|c| c.re).collect()
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let x: Vec<f32> = (0..1024).map(|i| tensor::sinf(0.05 * i as f32) + 0.3 * tensor::cosf(0.9 * i as f32)).collect();
    let y = irfft(&rfft(&x), 1024);
    let err = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    // a pure tone at bin 8 of 64 lands in bin 8 and nowhere else
    let tone: Vec<f32> = (0..64).map(|i| sincos(core::f32::consts::TAU * 8.0 * i as f32 / 64.0).1).collect();
    let b = rfft(&tone);
    let peak = b[8].re;
    let rest = b.iter().enumerate().filter(|(k, _)| *k != 8).map(|(_, c)| c.abs2()).fold(0f32, f32::max);
    let (s, c) = sincos(2.5);
    vec![
        ("sincos(2.5) is right to 1e-6", (s - 0.598_472_1).abs() < 1e-6 && (c + 0.801_143_6).abs() < 1e-6),
        ("irfft inverts rfft on 1024 points to 1e-4", err < 1e-4),
        ("a tone lands in its own bin with nothing elsewhere", (peak - 32.0).abs() < 1e-3 && rest < 1e-4),
    ]
}

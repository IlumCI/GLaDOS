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

/// In-place complex FFT; `n` must be a power of two. `inverse` scales by
/// `1/n` so the pair is an identity.
pub fn fft(x: &mut [C], inverse: bool) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
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
        let ang = core::f32::consts::TAU / len as f32 * if inverse { 1.0 } else { -1.0 };
        let wlen = C::new(tensor::cosf(ang), tensor::sinf(ang));
        let mut i = 0;
        while i < n {
            let mut w = C::new(1.0, 0.0);
            for k in 0..len / 2 {
                let u = x[i + k];
                let v = x[i + k + len / 2].mul(w);
                x[i + k] = C::new(u.re + v.re, u.im + v.im);
                x[i + k + len / 2] = C::new(u.re - v.re, u.im - v.im);
                w = w.mul(wlen);
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
    let tone: Vec<f32> = (0..64).map(|i| tensor::cosf(core::f32::consts::TAU * 8.0 * i as f32 / 64.0)).collect();
    let b = rfft(&tone);
    let peak = b[8].re;
    let rest = b.iter().enumerate().filter(|(k, _)| *k != 8).map(|(_, c)| c.abs2()).fold(0f32, f32::max);
    vec![
        ("irfft inverts rfft on 1024 points to 1e-4", err < 1e-4),
        ("a tone lands in its own bin with nothing elsewhere", (peak - 32.0).abs() < 1e-3 && rest < 1e-4),
    ]
}

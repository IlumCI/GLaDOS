//! Gradients without banding: interpolated in OKLab, positioned finely, dithered.
//!
//! `Framebuffer::vgrad` interpolates in sRGB with a position quantised to 256
//! steps and writes one colour per row, which is three separate ways to make a
//! band. A 1080-row wall from one navy to another moves its blue channel by
//! perhaps 40 values, so every value is a stripe some 27 rows tall with a hard
//! edge between stripes. That is the banding the old desktop wears.
//!
//! What fixes each:
//!
//! - **OKLab** (Ottosson, 2020) for the interpolation, because sRGB's midpoint
//!   between two colours is muddier and darker than either suggests: black to
//!   white crosses sRGB 128, which is perceptually well past the middle.
//!   OKLab's lightness is built to be perceptually uniform, so its midpoint
//!   is sRGB 99, where the eye puts it. Hyprland's border gradients use it for
//!   the same reason.
//! - **Position at pixel precision.** Each gradient is baked to one entry per
//!   pixel along its length, held in 8.8 fixed point, so nothing in the
//!   pipeline rounds a colour until the very last step.
//! - **An 8x8 ordered dither** at that last step: the fractional part of each
//!   channel decides, against a Bayer threshold, whether the pixel rounds up.
//!   The eye averages the pattern, so a ramp moving one value every 27 rows
//!   now moves one *eighth* of a value every few rows, and the stripes are
//!   gone. Ordered rather than random, so a redraw is bit-identical and the
//!   scene diff does not see a ramp as changed every frame.
//!
//! The dither is sized so that a colour that is *exactly* a value stays that
//! value: thresholds run -126..=126 in steps of four, so an error under one
//! 256th of a value (the float conversion's) can never flip a pixel. A flat
//! ramp is flat, which is asserted, because a gradient between two equal
//! colours that shimmers is worse than the banding this replaces.

use alloc::vec::Vec;

use super::{Color, Framebuffer};
/// sRGB transfer, decoding: a byte to linear light.
fn to_linear(c: u8) -> f64 {
    let v = c as f64 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        pow((v + 0.055) / 1.055, 2.4)
    }
}

/// sRGB transfer, encoding: linear light to a value on the 0..=255 scale,
/// unrounded.
fn from_linear(l: f64) -> f64 {
    let l = l.clamp(0.0, 1.0);
    let v = if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * pow(l, 1.0 / 2.4) - 0.055 };
    v * 255.0
}

// The conversion is done in f64, and that was measured rather than assumed.
// OKLab's inverse matrix subtracts terms near one to leave a linear value near
// zero for a dark channel in a bright colour, and the sRGB curve is steepest
// exactly there -- about 3,300 values per unit of linear light. In f32, with
// `tensor::lnf`'s 1e-5 relative error, (17, 238, 221) came back with its red
// at 16.94; with a longer series it was still 0.03 off at a channel of zero,
// which the dither turns into stray pixels. It runs once per pixel of a
// gradient's *length*, so f64 costs nothing that matters. These are local
// rather than changes to `tensor`'s, whose own note says altering them moves
// every logit the model produces.

fn pow(b: f64, e: f64) -> f64 {
    if b <= 0.0 {
        return 0.0;
    }
    exp(e * ln(b))
}

/// Natural log: exponent extracted, mantissa brought into [sqrt(1/2), sqrt(2))
/// so the atanh series' argument stays under 0.172 and nine terms reach f64.
fn ln(x: f64) -> f64 {
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let mut sum = 0.0;
    let mut k = 17.0;
    while k >= 1.0 {
        sum = sum * t2 + 1.0 / k;
        k -= 2.0;
    }
    2.0 * t * sum + e as f64 * core::f64::consts::LN_2
}

/// exp by range reduction to |r| <= ln2/2 and a degree-13 Taylor series.
fn exp(x: f64) -> f64 {
    if x < -700.0 {
        return 0.0;
    }
    let k = floor(x * core::f64::consts::LOG2_E + 0.5);
    let r = x - k * core::f64::consts::LN_2;
    let mut p = 1.0;
    let mut i = 13.0;
    while i >= 1.0 {
        p = 1.0 + r * p / i;
        i -= 1.0;
    }
    p * f64::from_bits(((k as i64 + 1023) as u64) << 52)
}

fn floor(x: f64) -> f64 {
    let t = x as i64 as f64;
    if t > x { t - 1.0 } else { t }
}

fn cbrt(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else if x < 0.0 {
        -pow(-x, 1.0 / 3.0)
    } else {
        pow(x, 1.0 / 3.0)
    }
}

/// A colour in OKLab: lightness and two opponent axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lab {
    pub l: f64,
    pub a: f64,
    pub b: f64,
}

/// sRGB to OKLab, with the matrices from Ottosson's reference implementation.
pub fn lab(c: Color) -> Lab {
    let (r, g, b) = (to_linear(c.r), to_linear(c.g), to_linear(c.b));
    let l = cbrt(0.412_221_470_8 * r + 0.536_332_536_3 * g + 0.051_445_992_9 * b);
    let m = cbrt(0.211_903_498_2 * r + 0.680_699_545_1 * g + 0.107_396_956_6 * b);
    let s = cbrt(0.088_302_461_9 * r + 0.281_718_837_6 * g + 0.629_978_700_5 * b);
    Lab {
        l: 0.210_454_255_3 * l + 0.793_617_785_0 * m - 0.004_072_046_8 * s,
        a: 1.977_998_495_1 * l - 2.428_592_205_0 * m + 0.450_593_709_9 * s,
        b: 0.025_904_037_1 * l + 0.782_771_766_2 * m - 0.808_675_766_0 * s,
    }
}

/// OKLab back to sRGB, as unrounded values on the 0..=255 scale. Out-of-gamut
/// results (a straight line in OKLab between two sRGB colours can leave the
/// sRGB cube slightly) are clamped per channel.
pub fn srgb(c: Lab) -> [f64; 3] {
    let l = c.l + 0.396_337_777_4 * c.a + 0.215_803_757_3 * c.b;
    let m = c.l - 0.105_561_345_8 * c.a - 0.063_854_172_8 * c.b;
    let s = c.l - 0.089_484_177_5 * c.a - 1.291_485_548_0 * c.b;
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    [
        from_linear(4.076_741_662_1 * l - 3.307_711_591_3 * m + 0.230_969_929_2 * s),
        from_linear(-1.268_438_004_6 * l + 2.609_757_401_1 * m - 0.341_319_396_5 * s),
        from_linear(-0.004_196_086_3 * l - 0.703_418_614_7 * m + 1.707_614_701_0 * s),
    ]
}

/// A colour in 8.8 fixed point per channel: 256 times its value.
type Fine = [u16; 3];

fn fine_of(c: Color) -> Fine {
    [(c.r as u16) << 8, (c.g as u16) << 8, (c.b as u16) << 8]
}

fn fine(v: [f64; 3]) -> Fine {
    let q = |x: f64| (x * 256.0 + 0.5).clamp(0.0, 65280.0) as u16;
    [q(v[0]), q(v[1]), q(v[2])]
}

/// A gradient, baked: one 8.8 colour per pixel along its length.
///
/// Stops are `(position, colour)` with position in `0.0..=1.0`, ascending; two
/// stops at one position are a hard step, which the old Luna captions needed
/// and Vista's glass gloss still does.
pub struct Ramp {
    lut: Vec<Fine>,
}

impl Ramp {
    /// Bake `stops` to `n` entries, entry `i` being the colour at `i / (n-1)`.
    pub fn bake(stops: &[(f32, Color)], n: usize) -> Ramp {
        let n = n.max(1);
        let mut lut = Vec::with_capacity(n);
        if stops.is_empty() {
            lut.resize(n, [0; 3]);
            return Ramp { lut };
        }
        let labs: Vec<Lab> = stops.iter().map(|s| lab(s.1)).collect();
        let mut k = 0usize;
        for i in 0..n {
            let t = if n == 1 { 0.0 } else { i as f32 / (n - 1) as f32 };
            while k + 1 < stops.len() && stops[k + 1].0 <= t {
                k += 1;
            }
            let e = if t <= stops[0].0 || k + 1 >= stops.len() {
                // Before the first stop or past the last: that stop, exactly.
                fine_of(stops[if t <= stops[0].0 { 0 } else { k }].1)
            } else {
                let (p0, c0) = stops[k];
                let (p1, c1) = stops[k + 1];
                if c0 == c1 {
                    // Exact rather than round-tripped through OKLab, so a
                    // flat run is flat to the last bit.
                    fine_of(c0)
                } else {
                    let f = (((t - p0) / (p1 - p0).max(1e-6)).clamp(0.0, 1.0)) as f64;
                    let (a, b) = (labs[k], labs[k + 1]);
                    if f == 0.0 {
                        fine_of(c0)
                    } else {
                        fine(srgb(Lab {
                            l: a.l + (b.l - a.l) * f,
                            a: a.a + (b.a - a.a) * f,
                            b: a.b + (b.b - a.b) * f,
                        }))
                    }
                }
            };
            lut.push(e);
        }
        Ramp { lut }
    }

    pub fn len(&self) -> usize {
        self.lut.len()
    }

    /// The undithered colour at entry `i`, in 8.8.
    pub fn at(&self, i: usize) -> [u16; 3] {
        self.lut[i.min(self.lut.len() - 1)]
    }
}

/// The 8x8 Bayer matrix, 0..64.
const BAYER: [[u8; 8]; 8] = [
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
];

/// The threshold at a pixel, in 256ths of a value: -126..=126, mean zero.
#[inline]
fn threshold(x: u32, y: u32) -> i32 {
    BAYER[(y & 7) as usize][(x & 7) as usize] as i32 * 4 - 126
}

/// One 8.8 colour rounded to a byte colour against the threshold at (x, y).
#[inline]
pub fn dither(c: [u16; 3], x: u32, y: u32) -> Color {
    let d = threshold(x, y) + 128;
    let ch = |v: u16| ((v as i32 + d) >> 8).clamp(0, 255) as u8;
    Color::new(ch(c[0]), ch(c[1]), ch(c[2]))
}

/// A vertical gradient over a box, top to bottom.
///
/// A row is no longer one colour -- the dither varies along it -- but it is
/// periodic in eight, so each row encodes eight pixels and repeats them.
pub fn vertical(fb: &Framebuffer, x: u32, y: u32, w: u32, h: u32, stops: &[(f32, Color)]) {
    if w == 0 || h == 0 {
        return;
    }
    let ramp = Ramp::bake(stops, h as usize);
    let end = y.saturating_add(h).min(fb.height());
    let mut row: Vec<u32> = alloc::vec![0; w as usize];
    for py in y..end {
        let c = ramp.at((py - y) as usize);
        // Keyed to screen coordinates, so two gradients that meet tile one
        // pattern rather than two that collide at the seam.
        let pat: [u32; 8] = core::array::from_fn(|i| {
            let px = x + i as u32;
            super::blend::opaque(fb.encode(dither(c, px, py)))
        });
        for (i, p) in row.iter_mut().enumerate() {
            *p = pat[i & 7];
        }
        fb.blit_span(x, py, &row);
    }
}

/// A linear gradient over a box at any angle: `angle` in degrees, 0 running
/// left to right and 90 top to bottom. The ramp spans the box's extent along
/// that direction, corner to corner, the way CSS's `linear-gradient` does.
pub fn linear(fb: &Framebuffer, x: u32, y: u32, w: u32, h: u32, angle: f32, stops: &[(f32, Color)]) {
    if w == 0 || h == 0 {
        return;
    }
    let a = angle * (core::f32::consts::PI / 180.0);
    let (dx, dy) = (cos(a), sin(a));
    // Projections of the four corners onto the direction bound the ramp.
    let proj = |px: f32, py: f32| px * dx + py * dy;
    let (wf, hf) = (w as f32 - 1.0, h as f32 - 1.0);
    let ps = [proj(0.0, 0.0), proj(wf, 0.0), proj(0.0, hf), proj(wf, hf)];
    let lo = ps.iter().cloned().fold(f32::MAX, f32::min);
    let hi = ps.iter().cloned().fold(f32::MIN, f32::max);
    let len = ((hi - lo) as usize + 1).max(2);
    let ramp = Ramp::bake(stops, len);
    // Position along the ramp in 16.16, stepped by addition across a row.
    let scale = (len - 1) as f32 / (hi - lo).max(1e-6);
    let step = (dx * scale * 65536.0) as i64;
    let end = y.saturating_add(h).min(fb.height());
    let mut row: Vec<u32> = alloc::vec![0; w as usize];
    for py in y..end {
        let mut t = (((py - y) as f32 * dy - lo) * scale * 65536.0) as i64;
        for (i, p) in row.iter_mut().enumerate() {
            let at = (t.max(0) >> 16) as usize;
            let frac = (t.max(0) & 0xffff) as u32;
            let (c0, c1) = (ramp.at(at), ramp.at(at + 1));
            let mix = |a: u16, b: u16| (a as i32 + ((b as i32 - a as i32) * frac as i32 >> 16)) as u16;
            let c = [mix(c0[0], c1[0]), mix(c0[1], c1[1]), mix(c0[2], c1[2])];
            *p = super::blend::opaque(fb.encode(dither(c, x + i as u32, py)));
            t += step;
        }
        fb.blit_span(x, py, &row);
    }
}

fn sin(x: f32) -> f32 {
    // Reduced to [-pi, pi] first; the series below is for small arguments.
    let tau = 2.0 * core::f32::consts::PI;
    let mut x = x - tau * crate::ai::tensor::floorf(x / tau + 0.5);
    if x > core::f32::consts::FRAC_PI_2 {
        x = core::f32::consts::PI - x;
    } else if x < -core::f32::consts::FRAC_PI_2 {
        x = -core::f32::consts::PI - x;
    }
    let x2 = x * x;
    x * (1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0 * (1.0 - x2 / 72.0))))
}

fn cos(x: f32) -> f32 {
    sin(x + core::f32::consts::FRAC_PI_2)
}

/// `video gradients`: the same ramps the old way and the new, side by side.
pub fn proof(fb: &Framebuffer) {
    let (w, h) = (fb.width(), fb.height());
    let half = w / 2;
    let navy = Color::new(12, 20, 38);
    let deep = Color::new(16, 64, 104);
    // Left: what vgrad does. Right: OKLab, fine position, dithered.
    fb.vgrad(0, 0, half, h, &[(0, navy), (255, deep)]);
    vertical(fb, half, 0, w - half, h, &[(0.0, navy), (1.0, deep)]);
    // Black to white across the bottom, sRGB on top and OKLab below.
    let (bx, by, bw, bh) = (w / 10, h * 7 / 10, w * 8 / 10, h / 12);
    for i in 0..bw {
        let v = (i * 255 / bw.max(1)) as u8;
        fb.rect(bx + i, by, 1, bh, Color::new(v, v, v));
    }
    linear(fb, bx, by + bh, bw, bh, 0.0, &[(0.0, Color::new(0, 0, 0)), (1.0, Color::new(255, 255, 255))]);
    // Hyprland's active border: a 45-degree ramp, here Aperture orange to gold.
    linear(fb, w / 10, h / 10, w * 3 / 10, h * 4 / 10, 45.0,
        &[(0.0, Color::new(255, 120, 20)), (1.0, Color::new(255, 214, 90))]);
    linear(fb, w * 6 / 10, h / 10, w * 3 / 10, h * 4 / 10, 45.0,
        &[(0.0, Color::new(40, 90, 255)), (0.5, Color::new(150, 60, 220)), (1.0, Color::new(255, 120, 20))]);
}

/// `diag gradient`.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };

    // OKLab round trip, over a grid of the cube.
    let mut worst = 0.0f64;
    for r in (0..=255).step_by(17) {
        for g in (0..=255).step_by(17) {
            for b in (0..=255).step_by(17) {
                let c = Color::new(r as u8, g as u8, b as u8);
                let back = srgb(lab(c));
                for (v, o) in back.iter().zip([c.r, c.g, c.b]) {
                    worst = worst.max((v - o as f64).abs());
                }
            }
        }
    }
    kprintln!("       round trip: worst channel moved {} thousandths of a value", (worst * 1000.0) as u32);
    claim("sRGB to OKLab and back moves no channel by a thousandth of a value", worst < 0.001);
    let white = lab(Color::new(255, 255, 255));
    claim("white is OKLab lightness one with no chroma",
        (white.l - 1.0).abs() < 1e-3 && white.a.abs() < 1e-3 && white.b.abs() < 1e-3);

    // The midpoint is perceptual: black to white crosses sRGB 99, not 128.
    let mid = Ramp::bake(&[(0.0, Color::new(0, 0, 0)), (1.0, Color::new(255, 255, 255))], 3).at(1);
    let m = mid[0] as i32 >> 8;
    claim("the middle of black to white is sRGB 99 (perceptual), not 128", (98..=100).contains(&m));

    // Ends are the stops exactly.
    let (a, b) = (Color::new(12, 20, 38), Color::new(16, 64, 104));
    let ramp = Ramp::bake(&[(0.0, a), (1.0, b)], 1080);
    claim("a ramp's ends are its stops, exactly", ramp.at(0) == fine_of(a) && ramp.at(1079) == fine_of(b));

    // A flat ramp survives the dither unchanged, at every threshold.
    let flat = Ramp::bake(&[(0.0, a), (1.0, a)], 64);
    let mut flat_ok = true;
    for y in 0..8 {
        for x in 0..8 {
            flat_ok &= dither(flat.at((x * 8 + y) as usize), x, y) == a;
        }
    }
    claim("a gradient between equal colours is flat after dithering", flat_ok);
    // And so does a value off by a 256th either way, which is the float's slack.
    let mut slack_ok = true;
    for e in [-1i32, 1] {
        let v = ((100i32 << 8) + e) as u16;
        for y in 0..8 {
            for x in 0..8 {
                slack_ok &= dither([v, v, v], x, y) == Color::new(100, 100, 100);
            }
        }
    }
    claim("an error of a 256th of a value never flips a pixel", slack_ok);

    // The dither preserves the mean over a tile.
    let mut mean_ok = true;
    for frac in [0u16, 32, 64, 128, 200, 255] {
        let v = (50u16 << 8) + frac;
        let sum: u32 = (0..64).map(|i| dither([v, v, v], i % 8, i / 8).r as u32).sum();
        let mean = sum as f32 * 256.0 / 64.0;
        mean_ok &= (mean - v as f32).abs() <= 4.0 + 0.01;
    }
    claim("over an 8x8 tile the dither keeps the mean to a 64th of a value", mean_ok);

    // No banding: a wall's row means track the ideal, and no band is tall.
    let (w, h) = (64u32, 1080u32);
    let mut px: Vec<u32> = alloc::vec![0; (w * h) as usize];
    let fb = unsafe { Framebuffer::over_ram(px.as_mut_ptr() as u64, w, h, w, super::Format::Bgrx) };
    vertical(&fb, 0, 0, w, h, &[(0.0, a), (1.0, b)]);
    let row_mean = |y: u32| -> f32 {
        (0..w).map(|x| ((px[(y * w + x) as usize]) & 0xff) as f32).sum::<f32>() / w as f32
    };
    let (mut longest, mut run, mut last) = (0u32, 0u32, -1.0f32);
    for y in 0..h {
        let m = row_mean(y);
        if m == last {
            run += 1;
        } else {
            run = 1;
            last = m;
        }
        longest = longest.max(run);
    }
    // Tracking is judged over eight rows, not one: a single Bayer row is not
    // zero-mean (row 0's thresholds average 21 of 64, row 7's 42), so one
    // row's mean sits up to a sixth of a value off by construction. The 8x8
    // tile is what the dither promises, and it is what the eye integrates.
    let mut track = 0.0f32;
    for by in 0..h / 8 {
        let got: f32 = (0..8).map(|i| row_mean(by * 8 + i)).sum::<f32>() / 8.0;
        let want: f32 = (0..8).map(|i| ramp.at((by * 8 + i) as usize)[2] as f32 / 256.0).sum::<f32>() / 8.0;
        track = track.max((got - want).abs());
    }
    let bands = (b.b - a.b) as u32;
    kprintln!("       blue moves {} values over {} rows: undithered bands would be {} rows; longest run here {}",
        bands, h, h / bands, longest);
    claim("dithered, no run of identical rows is a quarter of an undithered band", longest * 4 <= h / bands);
    kprintln!("       worst 8-row block is {} hundredths of a value from the ideal", (track * 100.0) as u32);
    claim("and every 8-row block's mean is within a tenth of a value of the ideal", track <= 0.1);
    claim("the box is filled and nothing past it", px.iter().all(|&p| p >> 24 == 0xff));

    // A 45-degree ramp is constant along the perpendicular.
    let (sw, sh) = (64u32, 64u32);
    let mut sq: Vec<u32> = alloc::vec![0; (sw * sh) as usize];
    let sfb = unsafe { Framebuffer::over_ram(sq.as_mut_ptr() as u64, sw, sh, sw, super::Format::Bgrx) };
    linear(&sfb, 0, 0, sw, sh, 45.0, &[(0.0, Color::new(0, 0, 0)), (1.0, Color::new(0, 0, 252))]);
    // Compare 8x8 tile means, which undo the dither, along the anti-diagonal.
    let tile = |tx: u32, ty: u32| -> u32 {
        (0..64).map(|i| sq[((ty * 8 + i / 8) * sw + tx * 8 + i % 8) as usize] & 0xff).sum()
    };
    let mut diag_ok = true;
    for k in 0..8 {
        let base = tile(k, 7 - k);
        diag_ok &= tile(7 - k, k).abs_diff(base) <= 64 && (k == 0 || tile(k, 7 - k).abs_diff(tile(0, 7)) <= 128);
    }
    claim("a 45-degree ramp is level along its perpendicular", diag_ok);
    claim("and runs from its first stop in one corner to its last in the other",
        tile(0, 0) / 64 <= 20 && tile(7, 7) / 64 >= 232);

    // A hard step stays a step.
    let step = Ramp::bake(&[(0.0, a), (0.5, a), (0.5, b), (1.0, b)], 256);
    claim("two stops at one position are a hard step",
        step.at(127) == fine_of(a) && step.at(128) == fine_of(b));
    ok
}

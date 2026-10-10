//! The wallpaper: an Aperture sun rising over rim-lit hills.
//!
//! Chosen by the operator from host mockups (2026-10-11): a deep night sky, the
//! Aperture mark as a full sun on the left, striped in its lower half the way
//! a synthwave sun is, with a warm halo; three hills in front of everything,
//! back to front, each crest a line of light -- blue, teal and the Aperture
//! orange. "A new day", so the whole sun is clear of the hills. The right side
//! is kept quiet because that is where windows sit.
//!
//! **Computed in linear light**, then encoded once with an ordered dither,
//! because every part of it is light being added: a halo over a sky, a rim
//! glow over a hill. Adding in sRGB makes every glow dirty and every overlap
//! too bright. The mock this was approved from did exactly this arithmetic,
//! so what was approved is what is drawn.
//!
//! It costs a few hundred milliseconds and is paid once per screen size --
//! `desk::wallpaper_cached` keeps the result, and the glass blur is taken from
//! it in the same breath.

use alloc::vec::Vec;

use super::{Color, Framebuffer};

/// sRGB byte to linear light.
fn lin(c: u8) -> f32 {
    let v = c as f64 / 255.0;
    (if v <= 0.04045 { v / 12.92 } else { super::grad::pow((v + 0.055) / 1.055, 2.4) }) as f32
}

fn lin3(c: (u8, u8, u8)) -> [f32; 3] {
    [lin(c.0), lin(c.1), lin(c.2)]
}

/// Linear light to sRGB in 8.8 fixed point, through a table: six million
/// `pow`s would be most of the cost of the whole picture.
struct Encode {
    t: Vec<u16>,
}

const ENC_N: usize = 16384;

impl Encode {
    fn new() -> Encode {
        let t = (0..=ENC_N)
            .map(|i| {
                let l = i as f64 / ENC_N as f64;
                let v = if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * super::grad::pow(l, 1.0 / 2.4) - 0.055 };
                (v * 255.0 * 256.0 + 0.5).min(65535.0) as u16
            })
            .collect();
        Encode { t }
    }

    /// Interpolated between the two nearest entries: the curve is steepest at
    /// the dark end, which is most of this picture.
    #[inline]
    fn at(&self, l: f32) -> u16 {
        let x = l.clamp(0.0, 1.0) * ENC_N as f32;
        let i = (x as usize).min(ENC_N - 1);
        let f = x - i as f32;
        (self.t[i] as f32 * (1.0 - f) + self.t[i + 1] as f32 * f) as u16
    }
}

/// Three box blurs of one f32 plane, in place: the glass blur's arithmetic
/// on light rather than on bytes, because a glow has no ceiling at 255.
fn blur_plane(p: &mut [f32], w: usize, h: usize, sigma: f32) {
    if sigma < 0.3 {
        return;
    }
    let mut tmp = alloc::vec![0f32; w * h];
    for size in super::glass::box_sizes(sigma, 3) {
        let r = (size / 2) as isize;
        // Rows.
        for y in 0..h {
            let row = &p[y * w..(y + 1) * w];
            let at = |i: isize| row[i.clamp(0, w as isize - 1) as usize];
            let mut s: f32 = (-r..=r).map(at).sum();
            for x in 0..w {
                tmp[y * w + x] = s / size as f32;
                s += at(x as isize + r + 1) - at(x as isize - r);
            }
        }
        // Columns.
        for x in 0..w {
            let at = |i: isize| tmp[(i.clamp(0, h as isize - 1) as usize) * w + x];
            let mut s: f32 = (-r..=r).map(at).sum();
            for y in 0..h {
                p[y * w + x] = s / size as f32;
                s += at(y as isize + r + 1) - at(y as isize - r);
            }
        }
    }
}

/// The layout, in fractions of the screen so it holds at any resolution.
pub struct Layout {
    pub sun_x: f32,
    pub sun_y: f32,
    pub sun_r: f32,
}

pub const LAYOUT: Layout = Layout { sun_x: 0.25, sun_y: 0.306, sun_r: 0.236 };

/// A hill's crest, as a fraction of the height at a fraction `u` of the width.
type Crest = fn(f32) -> f32;

fn sin(x: f32) -> f32 {
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

/// Back to front: crest, body colour, rim colour, rim strength.
const HILLS: [(Crest, (u8, u8, u8), (u8, u8, u8), f32); 3] = [
    (|u| 0.55 + 0.07 * sin(u * 3.2 + 2.0), (16, 36, 74), (70, 140, 255), 0.9),
    (|u| 0.60 + 0.08 * sin(u * 4.0 + 0.4), (10, 30, 56), (60, 220, 210), 1.0),
    (|u| 0.70 + 0.05 * sin(u * 4.6 + 3.6) - 0.03 * u, (6, 18, 34), (255, 150, 50), 0.9),
];

/// The sky, top to bottom, in OKLab (a ramp, so no band at any row).
const SKY: [(f32, Color); 3] = [
    (0.0, Color::new(4, 6, 20)),
    (0.5, Color::new(14, 22, 52)),
    (0.72, Color::new(40, 36, 78)),
];

/// Coverage of the sun at a pixel centre, stripes included.
fn sun_cov(x: f32, y: f32, cx: f32, cy: f32, r: f32) -> f32 {
    let c = super::splash::aperture_coverage(x - cx, y - cy, r);
    if c == 0.0 {
        return 0.0;
    }
    // Synthwave bars across the lower part, widening toward the bottom, each
    // edge anti-aliased by its distance in y.
    let t = (y - cy) / r;
    if t <= 0.1 {
        return c;
    }
    let k = ((t - 0.1) / 0.9).clamp(0.0, 1.0);
    let period = r * 0.12;
    let bar = period * (0.08 + 0.5 * k);
    let local = (y - cy) - period * crate::ai::tensor::floorf((y - cy) / period);
    // Signed distance to the bar occupying [0, bar) of each period: negative
    // inside it (to its nearer edge), positive outside (to this bar's end or
    // the next one's start).
    let d = if local < bar { -local.min(bar - local) } else { (local - bar).min(period - local) };
    let in_bar = (0.5 - d).clamp(0.0, 1.0);
    c * (1.0 - in_bar)
}

/// Paint the wallpaper into `fb`, whole. Answers the milliseconds it took.
pub fn paint(fb: &Framebuffer) -> u64 {
    let t0 = crate::time::rdtsc();
    let (w, h) = (fb.width() as usize, fb.height() as usize);
    if w == 0 || h == 0 {
        return 0;
    }
    let (wf, hf) = (w as f32, h as f32);
    let scale = hf / 1080.0;
    let (scx, scy, sr) = (LAYOUT.sun_x * wf, LAYOUT.sun_y * hf, LAYOUT.sun_r * hf);

    // The sky, one ramp entry per row, in linear light.
    let sky_ramp = super::grad::Ramp::bake(&SKY, h);
    let sky_row: Vec<[f32; 3]> = (0..h)
        .map(|y| {
            let c = sky_ramp.at(y);
            [lin((c[0] >> 8) as u8), lin((c[1] >> 8) as u8), lin((c[2] >> 8) as u8)]
        })
        .collect();

    // The halo: the sun's coverage, blurred, over a crop just big enough.
    let sigma = sr * 0.35;
    let pad = (3.0 * sigma) as i32 + 2;
    let (hx0, hy0) = ((scx - sr) as i32 - pad, (scy - sr) as i32 - pad);
    let hw = (2.0 * sr) as usize + 2 * pad as usize + 1;
    let mut halo = alloc::vec![0f32; hw * hw];
    for j in 0..hw {
        for i in 0..hw {
            let (x, y) = (hx0 + i as i32, hy0 + j as i32);
            halo[j * hw + i] = super::splash::aperture_coverage(x as f32 + 0.5 - scx, y as f32 + 0.5 - scy, sr);
        }
    }
    blur_plane(&mut halo, hw, hw, sigma);
    let warm = lin3((255, 120, 40));
    let (ink_top, ink_bot) = (lin3((255, 226, 120)), lin3((255, 84, 40)));

    // Picture and glow, three planes each.
    let mut img = alloc::vec![0f32; w * h * 3];
    let mut glow = alloc::vec![0f32; w * h * 3];
    let crests: Vec<[f32; 3]> = (0..w)
        .map(|x| {
            let u = (x as f32 + 0.5) / wf;
            [HILLS[0].0(u) * hf, HILLS[1].0(u) * hf, HILLS[2].0(u) * hf]
        })
        .collect();
    let bodies: Vec<[f32; 3]> = HILLS.iter().map(|hl| lin3(hl.1)).collect();
    let rims: Vec<[f32; 3]> = HILLS.iter().map(|hl| lin3(hl.2)).collect();
    for y in 0..h {
        let yc = y as f32 + 0.5;
        for x in 0..w {
            let xc = x as f32 + 0.5;
            let i = (y * w + x) * 3;
            let mut px = sky_row[y];
            // Halo, added to the sky.
            let (hi, hj) = (x as i32 - hx0, y as i32 - hy0);
            if hi >= 0 && hj >= 0 && (hi as usize) < hw && (hj as usize) < hw {
                let hv = halo[hj as usize * hw + hi as usize] * 0.55;
                for c in 0..3 {
                    px[c] += hv * warm[c];
                }
            }
            // The sun, over the sky.
            let sc = if (yc - scy).abs() <= sr + 1.0 && (xc - scx).abs() <= sr + 1.0 { sun_cov(xc, yc, scx, scy, sr) } else { 0.0 };
            if sc > 0.0 {
                let t = ((yc - (scy - sr)) / (2.0 * sr)).clamp(0.0, 1.0);
                for c in 0..3 {
                    let ink = ink_top[c] * (1.0 - t) + ink_bot[c] * t;
                    px[c] = px[c] * (1.0 - sc) + ink * sc;
                }
            }
            // Hills, back to front, each hiding what is behind it, its glow
            // included.
            let mut g = [0f32; 3];
            for (k, hl) in HILLS.iter().enumerate() {
                let d = yc - crests[x][k];
                let fill = (d + 0.5).clamp(0.0, 1.0);
                if fill > 0.0 {
                    let lift = 1.0 + 0.6 * crate::ai::tensor::expf(-d.max(0.0) / (90.0 * scale));
                    for c in 0..3 {
                        px[c] = px[c] * (1.0 - fill) + bodies[k][c] * lift * fill;
                        g[c] *= 1.0 - fill;
                    }
                }
                let e = d / (1.6 * scale);
                if e.abs() < 6.0 {
                    let line = crate::ai::tensor::expf(-e * e) * hl.3;
                    for c in 0..3 {
                        g[c] += line * rims[k][c];
                    }
                }
            }
            img[i..i + 3].copy_from_slice(&px);
            glow[i..i + 3].copy_from_slice(&g);
        }
    }

    // The glow twice: a little softened, and as a wide bloom.
    let mut plane = alloc::vec![0f32; w * h];
    for c in 0..3 {
        for (k, v) in plane.iter_mut().enumerate() {
            *v = glow[k * 3 + c];
        }
        let mut wide = plane.clone();
        blur_plane(&mut plane, w, h, 1.2 * scale);
        blur_plane(&mut wide, w, h, 30.0 * scale);
        for k in 0..w * h {
            img[k * 3 + c] += plane[k] + wide[k] * 0.6;
        }
    }

    // A vignette, then encode with the same ordered dither the gradients use.
    let enc = Encode::new();
    let mut row: Vec<u32> = alloc::vec![0; w];
    for y in 0..h {
        let v = (y as f32 + 0.5) / hf;
        for x in 0..w {
            let u = (x as f32 + 0.5) / wf;
            let vg = (1.0 - 0.45 * (((u - 0.5) * 1.6) * ((u - 0.5) * 1.6) + ((v - 0.45) * 1.4) * ((v - 0.45) * 1.4))).clamp(0.3, 1.0);
            let i = (y * w + x) * 3;
            let fine = [enc.at(img[i] * vg), enc.at(img[i + 1] * vg), enc.at(img[i + 2] * vg)];
            let c = super::grad::dither(fine, x as u32, y as u32);
            row[x] = super::blend::opaque(fb.encode(c));
        }
        fb.blit_span(0, y as u32, &row);
    }
    let mhz = crate::time::tsc_mhz().max(1) as u64;
    (crate::time::rdtsc() - t0) / mhz / 1000
}

/// `video wall`: the wallpaper alone, and how long it took.
pub fn proof(fb: &Framebuffer) {
    let ms = paint(fb);
    crate::kprintln!("  painted the wallpaper in {} ms", ms);
}

/// `diag wall`.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    // The anti-aliased mark is the boot screen's mark: draw both and compare
    // every pixel the coverage calls whole or empty.
    let (w, h) = (160u32, 160u32);
    let mut px: Vec<u32> = alloc::vec![0; (w * h) as usize];
    let fb = unsafe { Framebuffer::over_ram(px.as_mut_ptr() as u64, w, h, w, super::Format::Bgrx) };
    let (fg, bg) = (Color::new(255, 255, 255), Color::new(0, 0, 0));
    fb.rect(0, 0, w, h, bg);
    super::splash::aperture(&fb, 80, 80, 70, fg, bg);
    let cov = |x: i32, y: i32| super::splash::aperture_coverage(x as f32 + 0.5 - 80.0, y as f32 + 0.5 - 80.0, 70.0);
    let (mut agree, mut differ, mut edge) = (0u32, 0u32, 0u32);
    for y in 1..h as i32 - 1 {
        for x in 1..w as i32 - 1 {
            let c = cov(x, y);
            // An edge, or a pixel beside one: the triangle rasteriser rounds
            // its vertices to whole pixels, so it may put an edge a pixel from
            // where the exact geometry has it.
            let near = [cov(x - 1, y), cov(x + 1, y), cov(x, y - 1), cov(x, y + 1)];
            if (c > 0.0 && c < 1.0) || near.iter().any(|&n| n != c) {
                edge += 1;
                continue;
            }
            let lit = px[(y as u32 * w + x as u32) as usize] & 0xff > 127;
            if (c == 1.0) == lit {
                agree += 1;
            } else {
                differ += 1;
            }
        }
    }
    kprintln!("       {} pixels agree, {} on or beside an edge, {} differ", agree, edge, differ);
    claim("the anti-aliased mark is the boot screen's mark, off its edges", differ * 100 <= agree);

    // The whole sun is clear of the hills: a new day, not a sunset.
    let (wf, hf) = (1920.0f32, 1080.0f32);
    let (scx, scy, sr) = (LAYOUT.sun_x * wf, LAYOUT.sun_y * hf, LAYOUT.sun_r * hf);
    let mut clear = true;
    let mut x = scx - sr;
    while x <= scx + sr {
        let dy = crate::gfx::shape::libm_sqrt((sr * sr - (x - scx) * (x - scx)).max(0.0));
        let bottom = scy + dy;
        let crest = HILLS.iter().map(|hl| hl.0(x / wf) * hf).fold(f32::MAX, f32::min);
        clear &= bottom < crest;
        x += 4.0;
    }
    claim("the whole sun clears the nearest crest at every column", clear);

    // Deterministic, and every pixel written: two paints of a small screen.
    let (sw, sh) = (192u32, 108u32);
    let mut a: Vec<u32> = alloc::vec![0; (sw * sh) as usize];
    let mut b: Vec<u32> = alloc::vec![0xDEAD_BEEF; (sw * sh) as usize];
    let fa = unsafe { Framebuffer::over_ram(a.as_mut_ptr() as u64, sw, sh, sw, super::Format::Bgrx) };
    let fb2 = unsafe { Framebuffer::over_ram(b.as_mut_ptr() as u64, sw, sh, sw, super::Format::Bgrx) };
    paint(&fa);
    paint(&fb2);
    claim("two paints are identical, and every pixel is written", a == b);
    // The brightest pixel on the left is in the sun's disc and lit, and the
    // far corner is night: the layout landed where it was put. (A single
    // sample inside the disc can fall in a cut between blades, which is how
    // the first version of this claim failed on a correct picture.)
    let at = |x: u32, y: u32| {
        let p = a[(y * sw + x) as usize];
        ((p >> 16) & 0xff) + ((p >> 8) & 0xff) + (p & 0xff)
    };
    let (mut best, mut bx, mut by) = (0, 0, 0);
    for y in 0..sh {
        for x in 0..sw / 2 {
            if at(x, y) > best {
                (best, bx, by) = (at(x, y), x, y);
            }
        }
    }
    let (cx, cy, r) = (LAYOUT.sun_x * sw as f32, LAYOUT.sun_y * sh as f32, LAYOUT.sun_r * sh as f32);
    let (ddx, ddy) = (bx as f32 + 0.5 - cx, by as f32 + 0.5 - cy);
    let inside = ddx * ddx + ddy * ddy <= r * r;
    let night = at(sw - 8, 4);
    kprintln!("       brightest on the left {} at ({}, {}), far sky {}", best, bx, by, night);
    claim("the sun is the brightest thing on the left, and the far sky is night", inside && best > 400 && night < 60);
    ok
}

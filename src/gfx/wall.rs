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

pub(crate) fn lin3(c: (u8, u8, u8)) -> [f32; 3] {
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
fn sun_cov(x: f32, y: f32, cx: f32, cy: f32, r: f32, stripes: f32) -> f32 {
    let c = super::splash::aperture_coverage(x - cx, y - cy, r);
    if c == 0.0 {
        return 0.0;
    }
    // Synthwave bars across the lower part, widening toward the bottom, each
    // edge anti-aliased by its distance in y.
    let t = (y - cy) / r;
    if t <= 0.1 || stripes <= 0.0 {
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
    c * (1.0 - in_bar * stripes)
}

/// Paint the wallpaper into `fb`, whole. Answers the milliseconds it took.
///
/// The base and then the sun over it, through the same `overlay_sun` the boot
/// screen's glide uses for every one of its frames -- which is what makes the
/// glide's last frame this picture bit for bit rather than nearly.
pub fn paint(fb: &Framebuffer) -> u64 {
    let t0 = crate::time::rdtsc();
    paint_base(fb);
    let tables = Tables::new();
    overlay_sun(fb, &Sun::rest(fb.width(), fb.height()), &tables);
    let mhz = crate::time::tsc_mhz().max(1) as u64;
    (crate::time::rdtsc() - t0) / mhz / 1000
}

/// What the sun's overlay needs and should not rebuild per frame.
pub struct Tables {
    enc: Encode,
    dec: [f32; 256],
}

impl Tables {
    pub fn new() -> Tables {
        let mut dec = [0f32; 256];
        for (i, d) in dec.iter_mut().enumerate() {
            *d = lin(i as u8);
        }
        Tables { enc: Encode::new(), dec }
    }
}

/// The sun at some moment: where, how big, how striped, in what colours.
#[derive(Clone, Copy)]
pub struct Sun {
    pub cx: f32,
    pub cy: f32,
    pub r: f32,
    /// 0 none, 1 the wallpaper's full bars.
    pub stripes: f32,
    pub top: [f32; 3],
    pub bot: [f32; 3],
}

impl Sun {
    /// Where the wallpaper keeps it.
    pub fn rest(w: u32, h: u32) -> Sun {
        let (wf, hf) = (w as f32, h as f32);
        Sun {
            cx: LAYOUT.sun_x * wf,
            cy: LAYOUT.sun_y * hf,
            r: LAYOUT.sun_r * hf,
            stripes: 1.0,
            top: lin3((255, 226, 120)),
            bot: lin3((255, 84, 40)),
        }
    }
}

/// How far the halo reaches past the disc, as a multiple of its radius. It is
/// windowed to exactly zero at that distance: a logistic tail is still a few
/// tenths of a per cent of full light there, which on a night sky is a visible
/// square where the box the overlay works in stops.
const HALO_REACH: f32 = 1.05;

/// The halo at distance `d` from the centre of a sun of radius `r`: a disc
/// blurred by a Gaussian of 0.35 r, as the logistic approximation of its
/// edge profile, with the opening darker -- what blurring the mark's own
/// coverage gave, in closed form, so it can travel with the sun at no cost.
fn halo(d: f32, r: f32) -> f32 {
    let s = 0.35 * r;
    let edge = |rad: f32, w: f32| 1.0 / (1.0 + crate::ai::tensor::expf(1.702 * (d - rad) / w));
    let core = (edge(r, s) - 0.6 * edge(r * 0.46, s * 0.5)).max(0.0);
    let end = r * (1.0 + HALO_REACH);
    let window = ((end - d) / (0.35 * r)).clamp(0.0, 1.0);
    core * window * window
}

/// Lay the sun -- halo and striped mark -- over what `fb` already holds, in
/// linear light, behind the hills (which the base has drawn), vignetted as
/// the base is. Reads the pixels back, so `fb` must be memory.
pub fn overlay_sun(fb: &Framebuffer, sun: &Sun, t: &Tables) {
    overlay_sun_behind(fb, sun, t, 1.0);
}

/// The same, with the hills hiding the sun only `hills` (0..=1) as much --
/// the glide fades them in, and a sun cut by hills not yet visible is a sun
/// with a bite out of it.
pub fn overlay_sun_behind(fb: &Framebuffer, sun: &Sun, t: &Tables, hills: f32) {
    let (w, h) = (fb.width(), fb.height());
    let (wf, hf) = (w as f32, h as f32);
    let reach = sun.r * (1.0 + HALO_REACH);
    let x0 = ((sun.cx - reach).max(0.0)) as u32;
    let x1 = ((sun.cx + reach).max(0.0) as u32 + 1).min(w);
    let y0 = ((sun.cy - reach).max(0.0)) as u32;
    let y1 = ((sun.cy + reach).max(0.0) as u32 + 1).min(h);
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    let bgr = fb.format() == super::Format::Bgrx;
    let warm = lin3((255, 120, 40));
    // Per column, the nearest crest; per quarter pixel of distance, the halo.
    // Both are functions of one coordinate and were costing three sines and two
    // exponentials a pixel -- most of 75 ms a glide frame.
    let crest: Vec<f32> = (x0..x1)
        .map(|x| {
            let u = (x as f32 + 0.5) / wf;
            HILLS.iter().map(|hl| hl.0(u) * hf).fold(f32::MAX, f32::min)
        })
        .collect();
    let steps = (reach * 4.0) as usize + 2;
    let halo_at: Vec<f32> = (0..steps).map(|i| halo(i as f32 / 4.0, sun.r) * 0.55).collect();
    let job = Overlay {
        fb: *fb,
        sun: *sun,
        tables: t as *const Tables,
        hills,
        crest: crest.as_ptr(),
        halo: halo_at.as_ptr(),
        halo_len: halo_at.len(),
        x0,
        x1,
        y0,
        bgr,
        warm,
    };
    // Rows are the items: each is independent given the tables above, and
    // chunks are disjoint row ranges, so no two cores write one pixel. The
    // glide composes a million of these a frame and spent 31 ms doing it on
    // one core. Serial when there is nobody to help.
    let rows = (y1 - y0) as usize;
    let ctx = &job as *const Overlay as usize;
    if !crate::smp::parallel_split(ctx, overlay_rows, rows, (x1 - x0) as usize) {
        unsafe { overlay_rows(ctx, 0, rows) };
    }
}

/// Everything a chunk of the overlay reads, by pointer: `parallel_split` hands
/// a context through a `usize`, and it outlives the call because the call
/// does not return until every chunk has run.
#[derive(Clone, Copy)]
struct Overlay {
    fb: Framebuffer,
    sun: Sun,
    tables: *const Tables,
    hills: f32,
    crest: *const f32,
    halo: *const f32,
    halo_len: usize,
    x0: u32,
    x1: u32,
    y0: u32,
    bgr: bool,
    warm: [f32; 3],
}

/// Rows `lo..hi` of the overlay, counted from its top.
unsafe fn overlay_rows(ctx: usize, lo: usize, hi: usize) {
    let j = unsafe { *(ctx as *const Overlay) };
    let t = unsafe { &*j.tables };
    let width = (j.x1 - j.x0) as usize;
    let crest = unsafe { core::slice::from_raw_parts(j.crest, width) };
    let halo_at = unsafe { core::slice::from_raw_parts(j.halo, j.halo_len) };
    let (fb, sun) = (&j.fb, &j.sun);
    let (wf, hf) = (fb.width() as f32, fb.height() as f32);
    let mut row: Vec<u32> = alloc::vec![0; width];
    for y in j.y0 + lo as u32..j.y0 + hi as u32 {
        let yc = y as f32 + 0.5;
        let v = yc / hf;
        for x in j.x0..j.x1 {
            let xc = x as f32 + 0.5;
            let u = xc / wf;
            let p = fb.get(x, y);
            row[(x - j.x0) as usize] = p;
            // Behind the hills: whatever a hill covers here, the sun does not.
            // The nearest crest is the highest, and hills are opaque, so it is
            // the only one that matters.
            let cover = (yc - crest[(x - j.x0) as usize] + 0.5).clamp(0.0, 1.0) * j.hills;
            let open = 1.0 - cover;
            if open <= 0.0 {
                continue;
            }
            let (dx, dy) = (xc - sun.cx, yc - sun.cy);
            let d = crate::gfx::shape::libm_sqrt(dx * dx + dy * dy);
            let hv = halo_at.get((d * 4.0) as usize).copied().unwrap_or(0.0);
            let c = if dx.abs() <= sun.r + 1.0 && dy.abs() <= sun.r + 1.0 {
                sun_cov(xc, yc, sun.cx, sun.cy, sun.r, sun.stripes)
            } else {
                0.0
            } * open;
            if hv < 1e-4 && c == 0.0 {
                continue;
            }
            let vg = vignette(u, v);
            let lanes = if j.bgr { [2, 1, 0] } else { [0, 1, 2] };
            let ty = ((yc - (sun.cy - sun.r)) / (2.0 * sun.r)).clamp(0.0, 1.0);
            let mut fine = [0u16; 3];
            for k in 0..3 {
                let byte = ((p >> (8 * lanes[k])) & 0xff) as usize;
                let bg = t.dec[byte] + hv * j.warm[k] * vg * open;
                let ink = (sun.top[k] * (1.0 - ty) + sun.bot[k] * ty) * vg;
                fine[k] = t.enc.at(bg * (1.0 - c) + ink * c);
            }
            let col = super::grad::dither(fine, x, y);
            row[(x - j.x0) as usize] = super::blend::opaque(fb.encode(col));
        }
        fb.blit_span(j.x0, y, &row);
    }
}

#[inline]
fn vignette(u: f32, v: f32) -> f32 {
    let (a, b) = ((u - 0.5) * 1.6, (v - 0.45) * 1.4);
    (1.0 - 0.45 * (a * a + b * b)).clamp(0.3, 1.0)
}

/// Everything but the sun: sky, hills, their light, the vignette.
pub fn paint_base(fb: &Framebuffer) {
    let (w, h) = (fb.width() as usize, fb.height() as usize);
    if w == 0 || h == 0 {
        return;
    }
    let (wf, hf) = (w as f32, h as f32);
    let scale = hf / 1080.0;

    // The sky, one ramp entry per row, in linear light.
    let sky_ramp = super::grad::Ramp::bake(&SKY, h);
    let sky_row: Vec<[f32; 3]> = (0..h)
        .map(|y| {
            let c = sky_ramp.at(y);
            [lin((c[0] >> 8) as u8), lin((c[1] >> 8) as u8), lin((c[2] >> 8) as u8)]
        })
        .collect();

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
            let i = (y * w + x) * 3;
            let mut px = sky_row[y];
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

    // The vignette, then encode with the same ordered dither the gradients use.
    let enc = Encode::new();
    let mut row: Vec<u32> = alloc::vec![0; w];
    for y in 0..h {
        let v = (y as f32 + 0.5) / hf;
        for x in 0..w {
            let u = (x as f32 + 0.5) / wf;
            let vg = vignette(u, v);
            let i = (y * w + x) * 3;
            let fine = [enc.at(img[i] * vg), enc.at(img[i + 1] * vg), enc.at(img[i + 2] * vg)];
            let c = super::grad::dither(fine, x as u32, y as u32);
            row[x] = super::blend::opaque(fb.encode(c));
        }
        fb.blit_span(0, y as u32, &row);
    }
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

//! Anti-aliased geometry: rounded rectangles, rings, circles, lines.
//!
//! Q2 of the pixel pipeline. Every edge in the interface was a staircase -- a
//! rounded corner was a lookup table of whole pixels and a line was Bresenham --
//! and that, more than any colour, is what read as 1998.
//!
//! ### Coverage from a signed distance
//!
//! Each shape is a signed distance `d` from the pixel's centre to its edge,
//! negative inside, and the pixel's coverage is `clamp(0.5 - d, 0, 1)`. For a
//! straight edge that is *exactly* the fraction of the pixel inside it, and for
//! a curve whose radius is large against a pixel it is within a fraction of a
//! percent, which the area claim below measures. It is the box SDF Hyprland's
//! rounding shader uses, run on the CPU, with circular corners (its default
//! `rounding_power` of 2); its smoothstep softening is not copied, since a
//! smoothstep over one pixel is a contrast curve and Q4 owns those.
//!
//! ### Only the edge is computed
//!
//! A row of a rounded rectangle is solid apart from its first and last few
//! pixels, so the interior is filled as one span (`Framebuffer::over_span`)
//! and coverage is computed only where it can be fractional: the corner bands
//! and one pixel either side of a straight edge. A 1600x1000 window therefore
//! costs one SDF per edge pixel, not per pixel.
//!
//! Colours are premultiplied (`gfx::blend`), and every write goes through the
//! surface's scissor, so these draw correctly in a damage-limited frame.

use alloc::vec;
use alloc::vec::Vec;

use super::Framebuffer;

/// Coverage of a pixel whose centre is `d` from the edge, in 0..=255.
#[inline]
fn cov(d: f32) -> u8 {
    let c = 0.5 - d;
    if c <= 0.0 {
        0
    } else if c >= 1.0 {
        255
    } else {
        (c * 255.0 + 0.5) as u8
    }
}

/// Signed distance from `(px, py)` to a rounded box centred at `(cx, cy)` with
/// half-extents `(hx, hy)` and corner radius `r`. Negative inside.
#[inline]
fn sd_rbox(px: f32, py: f32, cx: f32, cy: f32, hx: f32, hy: f32, r: f32) -> f32 {
    let qx = (px - cx).abs() - hx + r;
    let qy = (py - cy).abs() - hy + r;
    let ox = qx.max(0.0);
    let oy = qy.max(0.0);
    libm_sqrt(ox * ox + oy * oy) + qx.max(qy).min(0.0) - r
}

/// `core` has no `floor` for floats; the i32 range is plenty for pixels.
#[inline]
fn floorf(x: f32) -> f32 {
    let i = x as i32 as f32;
    if i > x { i - 1.0 } else { i }
}

#[inline]
fn ceilf(x: f32) -> f32 {
    let i = x as i32 as f32;
    if i < x { i + 1.0 } else { i }
}

#[inline]
fn libm_sqrt(x: f32) -> f32 {
    // `sqrtss`: SSE is on for this target, and the result is correctly rounded.
    unsafe { core::arch::x86_64::_mm_cvtss_f32(core::arch::x86_64::_mm_sqrt_ss(core::arch::x86_64::_mm_set_ss(x))) }
}

/// A rectangle in pixels, possibly partly off the surface.
#[derive(Clone, Copy, Debug)]
pub struct Box {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

impl Box {
    pub const fn new(x: i32, y: i32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }
}

/// The rows and columns of `b` that land on `fb`, as `(x0, y0, x1, y1)`.
fn on_surface(fb: &Framebuffer, b: Box) -> Option<(i32, i32, i32, i32)> {
    let x0 = b.x.max(0);
    let y0 = b.y.max(0);
    let x1 = (b.x + b.w as i32).min(fb.width() as i32);
    let y1 = (b.y + b.h as i32).min(fb.height() as i32);
    (x1 > x0 && y1 > y0).then_some((x0, y0, x1, y1))
}

/// Fill a rounded rectangle with a premultiplied colour.
pub fn rrect(fb: &Framebuffer, b: Box, radius: u32, color: u32) {
    ring_or_fill(fb, b, radius, None, color)
}

/// The band between a rounded rectangle and the same shape inset by `width`:
/// a window's border. The inner radius follows the outer one in, so the band
/// is the same thickness all the way round the corner.
pub fn ring(fb: &Framebuffer, b: Box, radius: u32, width: u32, color: u32) {
    ring_or_fill(fb, b, radius, Some(width), color)
}

fn ring_or_fill(fb: &Framebuffer, b: Box, radius: u32, width: Option<u32>, color: u32) {
    let Some((x0, y0, x1, y1)) = on_surface(fb, b) else { return };
    let r = radius.min(b.w / 2).min(b.h / 2) as f32;
    let hx = b.w as f32 / 2.0;
    let hy = b.h as f32 / 2.0;
    let cx = b.x as f32 + hx;
    let cy = b.y as f32 + hy;
    let inner = width.map(|w| {
        let w = w as f32;
        (hx - w, hy - w, (r - w).max(0.0))
    });
    // How far in from each side a row is not plain solid: the corner band,
    // plus one pixel for the edge itself.
    let band = r as i32 + 1;
    let wpx = width.map(|w| w as i32 + 1);
    let mut mask: Vec<u8> = vec![0; (x1 - x0) as usize];
    for y in y0..y1 {
        let py = y as f32 + 0.5;
        let in_corner_rows = y < b.y + band || y >= b.y + b.h as i32 - band;
        // Columns needing a per-pixel answer on this row.
        // On a corner row of a ring the band must also cover the ring's own
        // width, which can be wider than a small radius's corner.
        let edge = match (in_corner_rows, wpx) {
            (true, None) => band,
            (true, Some(w)) => band.max(w),
            (false, None) => 1,
            (false, Some(w)) => w,
        };
        let solid_rows_ring = wpx.map_or(false, |w| y < b.y + w || y >= b.y + b.h as i32 - w);
        let coverage = |px: f32| -> u8 {
            let outer = cov(sd_rbox(px, py, cx, cy, hx, hy, r));
            match inner {
                None => outer,
                Some((ix, iy, ir)) => {
                    if ix <= 0.0 || iy <= 0.0 {
                        return outer;
                    }
                    let inn = cov(sd_rbox(px, py, cx, cy, ix, iy, ir));
                    outer.saturating_sub(inn)
                }
            }
        };
        // A ring's top and bottom bands are needed across the whole row; so is
        // any row of a box too narrow to have an interior.
        if solid_rows_ring || (b.w as i32) <= 2 * edge {
            for x in x0..x1 {
                mask[(x - x0) as usize] = coverage(x as f32 + 0.5);
            }
            fb.over_mask_row(x0 as u32, y as u32, &mask[..(x1 - x0) as usize], color);
            continue;
        }
        let left_end = (b.x + edge).min(x1);
        let right_start = (b.x + b.w as i32 - edge).max(x0);
        // Left edge.
        if left_end > x0 {
            for x in x0..left_end {
                mask[(x - x0) as usize] = coverage(x as f32 + 0.5);
            }
            fb.over_mask_row(x0 as u32, y as u32, &mask[..(left_end - x0) as usize], color);
        }
        // The interior: solid for a fill, empty for a ring.
        if width.is_none() {
            let a = left_end.max(x0);
            let z = right_start.min(x1);
            if z > a {
                fb.over_span(a as u32, y as u32, (z - a) as u32, color);
            }
        }
        // Right edge.
        let rs = right_start.max(left_end);
        if x1 > rs {
            for x in rs..x1 {
                mask[(x - rs) as usize] = coverage(x as f32 + 0.5);
            }
            fb.over_mask_row(rs as u32, y as u32, &mask[..(x1 - rs) as usize], color);
        }
    }
}

/// A filled circle of radius `r` centred at `(cx, cy)`, in pixels.
pub fn circle(fb: &Framebuffer, cx: f32, cy: f32, r: f32, color: u32) {
    let x0 = floorf(cx - r) as i32;
    let y0 = floorf(cy - r) as i32;
    let size = ceilf(2.0 * r) as u32 + 2;
    let b = Box::new(x0, y0, size, size);
    let Some((sx0, sy0, sx1, sy1)) = on_surface(fb, b) else { return };
    let mut mask: Vec<u8> = vec![0; (sx1 - sx0) as usize];
    for y in sy0..sy1 {
        let dy = y as f32 + 0.5 - cy;
        for x in sx0..sx1 {
            let dx = x as f32 + 0.5 - cx;
            mask[(x - sx0) as usize] = cov(libm_sqrt(dx * dx + dy * dy) - r);
        }
        fb.over_mask_row(sx0 as u32, y as u32, &mask, color);
    }
}

/// A line from `(ax, ay)` to `(bx, by)`, `width` pixels thick with round
/// ends: a capsule. Replaces Bresenham, whose one-pixel staircase is the
/// thinnest possible jagged edge.
pub fn line(fb: &Framebuffer, ax: f32, ay: f32, bx: f32, by: f32, width: f32, color: u32) {
    let hw = width / 2.0;
    let x0 = floorf(ax.min(bx) - hw) as i32 - 1;
    let y0 = floorf(ay.min(by) - hw) as i32 - 1;
    let x1 = ceilf(ax.max(bx) + hw) as i32 + 1;
    let y1 = ceilf(ay.max(by) + hw) as i32 + 1;
    let b = Box::new(x0, y0, (x1 - x0) as u32, (y1 - y0) as u32);
    let Some((sx0, sy0, sx1, sy1)) = on_surface(fb, b) else { return };
    let (ex, ey) = (bx - ax, by - ay);
    let len2 = (ex * ex + ey * ey).max(1e-6);
    let mut mask: Vec<u8> = vec![0; (sx1 - sx0) as usize];
    for y in sy0..sy1 {
        let py = y as f32 + 0.5;
        let mut any = false;
        for x in sx0..sx1 {
            let px = x as f32 + 0.5;
            let t = (((px - ax) * ex + (py - ay) * ey) / len2).clamp(0.0, 1.0);
            let (qx, qy) = (px - (ax + t * ex), py - (ay + t * ey));
            let c = cov(libm_sqrt(qx * qx + qy * qy) - hw);
            any |= c != 0;
            mask[(x - sx0) as usize] = c;
        }
        if any {
            fb.over_mask_row(sx0 as u32, y as u32, &mask, color);
        }
    }
}

/// A proof sheet for the eye: every shape at several sizes, a fan of lines at
/// every angle, and translucent overlaps. `video shapes` shows it. The claims
/// in `selftest` settle the arithmetic; only a look settles whether an edge
/// reads as smooth.
pub fn proof(fb: &Framebuffer) {
    use super::blend::{opaque, premul};
    let (w, h) = (fb.width(), fb.height());
    let enc = |r: u8, g: u8, b: u8| fb.encode(super::Color::new(r, g, b));
    for y in 0..h {
        let t = y * 255 / h.max(1);
        fb.over_span(0, y, w, opaque(enc(14, (24 + t / 6) as u8, (40 + t / 4) as u8)));
    }
    let white = opaque(enc(240, 244, 250));
    let orange = opaque(enc(255, 150, 40));
    // Laid out on a 960x540 grid and scaled to fit whichever axis is tighter,
    // so every specimen is on screen at any resolution.
    let k = (w as f32 / 960.0).min(h as f32 / 540.0);
    let px = |v: f32| (v * k) as i32;
    let ks = (k as u32).max(1);
    let mut x = 30.0;
    for r in [0u32, 3, 6, 10, 16, 28] {
        let (bw, bh) = (px(120.0) as u32, px(60.0) as u32);
        rrect(fb, Box::new(px(x), px(30.0), bw, bh), r * ks, white);
        ring(fb, Box::new(px(x), px(110.0), bw, bh), r * ks, ks + (r % 3), orange);
        x += 150.0;
    }
    let mut cx = 60.0;
    for r in [1.5f32, 3.0, 6.5, 12.0, 24.0, 40.0] {
        circle(fb, cx * k, 240.0 * k, r * k, white);
        cx += r * 2.0 + 30.0;
    }
    let (ox, oy) = (420.0 * k, 500.0 * k);
    for i in 0..=24 {
        let a = i as f32 * (core::f32::consts::FRAC_PI_2 / 24.0);
        let (c, sn) = (cos(a), sin(a));
        let width = [1.0f32, 1.5, 3.0][i % 3] * k;
        line(fb, ox, oy, ox + c * 200.0 * k, oy - sn * 200.0 * k, width, white);
    }
    for (i, (r, g, b)) in [(255u8, 60u8, 60u8), (60, 255, 90), (70, 120, 255)].iter().enumerate() {
        let (dx, dy) = [(0.0, 0.0), (45.0, 0.0), (22.0, 38.0)][i];
        circle(fb, (700.0 + dx) * k, (400.0 + dy) * k, 50.0 * k, premul(enc(*r, *g, *b), 128));
    }
}

/// Sine and cosine for the proof sheet's fan; a few terms is plenty for
/// placing lines, and keeps the shape code free of a maths dependency.
fn sin(x: f32) -> f32 {
    let x2 = x * x;
    x * (1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0 * (1.0 - x2 / 72.0))))
}

fn cos(x: f32) -> f32 {
    sin(x + core::f32::consts::FRAC_PI_2)
}

/// `diag shape`: coverage is exact where it can be, symmetric, and sums to the
/// shape's area.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    let (w, h) = (200u32, 160u32);
    let mut px: Vec<u32> = vec![0; (w * h) as usize];
    let surface = |px: &mut Vec<u32>| unsafe {
        Framebuffer::over_ram(px.as_mut_ptr() as u64, w, h, w, super::Format::Bgrx)
    };
    let alpha_sum = |px: &[u32]| px.iter().map(|&p| (p >> 24) as f64 / 255.0).sum::<f64>();
    let white = 0xFFFF_FFFFu32;

    // A rounded box: area w*h - (4 - pi) r^2.
    let fb = surface(&mut px);
    rrect(&fb, Box::new(20, 30, 120, 80), 18, white);
    let want = 120.0 * 80.0 - (4.0 - core::f64::consts::PI) * 18.0 * 18.0;
    let got = alpha_sum(&px);
    claim("a rounded box's coverage sums to its area within 0.5%", ((got - want) / want).abs() < 0.005);
    claim("its interior is fully covered", px[(70 * w + 80) as usize] == white);
    claim("outside it nothing is touched", px[(10 * w + 10) as usize] == 0 && px[(30 * w + 20) as usize] >> 24 < 40);
    let a = |px: &[u32], x: u32, y: u32| px[(y * w + x) as usize] >> 24;
    let symmetric = (30..110).all(|y| (20..140).all(|x| a(&px, x, y) == a(&px, 20 + 139 - x, y))) &&
        (30..110).all(|y| (20..140).all(|x| a(&px, x, y) == a(&px, x, 30 + 109 - y)));
    claim("it is mirror-symmetric both ways", symmetric);
    claim("a straight edge on a pixel boundary is exact: full inside, none outside",
          a(&px, 80, 30) == 255 && a(&px, 80, 29) == 0 && a(&px, 19, 70) == 0 && a(&px, 20, 70) == 255);

    // A circle: area pi r^2.
    px.iter_mut().for_each(|p| *p = 0);
    let fb = surface(&mut px);
    circle(&fb, 100.0, 80.0, 50.0, white);
    let want = core::f64::consts::PI * 2500.0;
    claim("a circle's coverage sums to pi r^2 within 0.5%", ((alpha_sum(&px) - want) / want).abs() < 0.005);

    // A ring: the outer box minus the inset one.
    px.iter_mut().for_each(|p| *p = 0);
    let fb = surface(&mut px);
    ring(&fb, Box::new(20, 30, 120, 80), 18, 3, white);
    let outer = 120.0 * 80.0 - (4.0 - core::f64::consts::PI) * 18.0 * 18.0;
    let inner = 114.0 * 74.0 - (4.0 - core::f64::consts::PI) * 15.0 * 15.0;
    let want = outer - inner;
    claim("a ring's coverage is the band between the two boxes, within 1%",
          ((alpha_sum(&px) - want) / want).abs() < 0.01);
    claim("and its middle is left untouched", a(&px, 80, 70) == 0);

    // A line: a 3-pixel horizontal capsule 100 long covers 100*3 + pi*1.5^2.
    px.iter_mut().for_each(|p| *p = 0);
    let fb = surface(&mut px);
    line(&fb, 50.0, 80.5, 150.0, 80.5, 3.0, white);
    let want = 300.0 + core::f64::consts::PI * 2.25;
    claim("a line's coverage sums to its capsule's area within 1%",
          ((alpha_sum(&px) - want) / want).abs() < 0.01);

    // Half coverage over a colour blends, and the scissor holds.
    px.iter_mut().for_each(|p| *p = 0xFF00_0000);
    let fb = surface(&mut px).clipped(0, 0, 60, h);
    rrect(&fb, Box::new(20, 30, 120, 80), 0, white);
    claim("a scissored fill stops at the scissor", a(&px, 59, 50) == 255 && px[(50 * w + 60) as usize] == 0xFF00_0000);
    ok
}

//! Soft shadows: a rounded box, blurred by a Gaussian, laid under a window.
//!
//! The shadow the desktop had was two five-pixel bands of a fixed shade, offset
//! three pixels -- a hard, rectangular silhouette that gave the rounded corners
//! away and is most of why a window looked pasted on rather than lifted. A
//! modern compositor casts something closer to light: a rounded box, blurred,
//! dark under the window's lower edge and gone a few dozen pixels out.
//!
//! **Blurred once, sliced forever.** A Gaussian over a window-sized box every
//! frame is millions of multiply-adds for a picture that never changes. But
//! a blurred box is a nine-slice by construction: far enough from the corners,
//! every row of its middle is the same row and every column the same column,
//! because the edges on the far side are more than three sigma away and the
//! kernel cannot reach them. So one template per `(radius, sigma)` -- a
//! rounded box exactly big enough that its middle is flat, blurred properly --
//! gives the shadow of *any* window at least that size: its corners are the
//! template's corners and its edges are the template's middle line repeated.
//! This is the nine-slice Levien and Wallace describe for blurred rounded
//! rectangles on the CPU; it is computed here by convolving the exact
//! coverage rather than by their closed form, which is cheaper to trust --
//! the template is built once, and its mass is checked against the box's area.
//!
//! A box smaller than the template (a tooltip, a menu) gets an exact render of
//! its own size, cached beside the templates.
//!
//! **Zeroed under the window**, as Hyprland does: the window covers that area
//! anyway when it is opaque, and when it is glass (Q6) a shadow underneath
//! would darken the glass from behind, which no light does.

use alloc::sync::Arc;
use alloc::vec::Vec;

use super::shape::{libm_sqrt, sd_rbox};
use super::Framebuffer;
use crate::sync::Spin;

/// What a shadow looks like: the corner radius of the box casting it and the
/// blur's standard deviation in pixels.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Spec {
    pub radius: u32,
    pub sigma: f32,
}

impl Spec {
    /// How far the shadow reaches past the box on every side: three sigma,
    /// beyond which a Gaussian is under a third of a percent.
    pub fn extent(self) -> u32 {
        (3.0 * self.sigma + 0.999) as u32
    }

    fn key(self) -> (u32, u32) {
        (self.radius, (self.sigma * 4.0 + 0.5) as u32)
    }
}

/// A blurred box: `w` by `h` of A8, the box itself occupying `[e, e+bw)`.
struct Image {
    w: u32,
    h: u32,
    a: Vec<u8>,
}

/// Blur the coverage of a `bw` x `bh` rounded box by `spec`, into an image
/// padded by the extent on every side.
fn render(bw: u32, bh: u32, spec: Spec) -> Image {
    let e = spec.extent();
    let (w, h) = (bw + 2 * e, bh + 2 * e);
    let r = spec.radius.min(bw / 2).min(bh / 2) as f32;
    // Exact coverage first, the same signed distance `shape` draws with, so a
    // shadow and the window casting it agree about where the corner is.
    let (cx, cy) = (e as f32 + bw as f32 / 2.0, e as f32 + bh as f32 / 2.0);
    let (hx, hy) = (bw as f32 / 2.0, bh as f32 / 2.0);
    let mut src = alloc::vec![0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let d = sd_rbox(x as f32 + 0.5, y as f32 + 0.5, cx, cy, hx, hy, r);
            src[(y * w + x) as usize] = (0.5 - d).clamp(0.0, 1.0);
        }
    }
    // A Gaussian is separable: two passes of one dimension each, rather than
    // one of two, which at a 91-tap kernel is the difference between thousands
    // of multiplies a pixel and under two hundred.
    let k = kernel(spec.sigma, e);
    let mut mid = alloc::vec![0f32; (w * h) as usize];
    for y in 0..h as usize {
        let row = &src[y * w as usize..(y + 1) * w as usize];
        for x in 0..w as usize {
            let mut s = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let xx = x as isize + i as isize - e as isize;
                if xx >= 0 && (xx as usize) < w as usize {
                    s += kv * row[xx as usize];
                }
            }
            mid[y * w as usize + x] = s;
        }
    }
    let mut a = alloc::vec![0u8; (w * h) as usize];
    for x in 0..w as usize {
        for y in 0..h as usize {
            let mut s = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let yy = y as isize + i as isize - e as isize;
                if yy >= 0 && (yy as usize) < h as usize {
                    s += kv * mid[yy as usize * w as usize + x];
                }
            }
            a[y * w as usize + x] = (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
    Image { w, h, a }
}

/// Normalised Gaussian taps, `2e + 1` of them.
fn kernel(sigma: f32, e: u32) -> Vec<f32> {
    let sigma = sigma.max(0.01);
    let mut k: Vec<f32> = (0..=2 * e)
        .map(|i| {
            let x = i as f32 - e as f32;
            exp(-(x * x) / (2.0 * sigma * sigma))
        })
        .collect();
    let sum: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    k
}

fn exp(x: f32) -> f32 {
    crate::ai::tensor::expf(x)
}

/// The side of the square box whose blurred middle is flat. A column is flat
/// when the kernel, reaching `extent` either way, touches neither box edge nor
/// a corner's arc; `2(r + e) + 1` leaves exactly one such column, and two more
/// make the middle three identical, so the slicing has a pixel of slack.
fn template_side(spec: Spec) -> u32 {
    2 * (spec.radius + spec.extent()) + 3
}

/// Templates by `(radius, quarter-sigma)`, and exact renders by size. A
/// handful of each -- a desktop has two shadow styles and a few menus -- so a
/// list searched in order is the whole cache, and the oldest goes when it is
/// full.
struct Cache {
    templates: Vec<((u32, u32), Arc<Image>)>,
    exact: Vec<((u32, u32, u32, u32), Arc<Image>)>,
}

static CACHE: Spin<Cache> = Spin::new(Cache { templates: Vec::new(), exact: Vec::new() });
const KEEP: usize = 8;

/// Paint the shadow of the box `(x, y, w, h)` at offset `(dx, dy)`, `alpha`
/// (0..=255) black at its darkest, zero wherever the box itself fully covers.
pub fn cast(fb: &Framebuffer, x: i32, y: i32, w: u32, h: u32, spec: Spec, dx: i32, dy: i32, alpha: u8) {
    if w == 0 || h == 0 || alpha == 0 {
        return;
    }
    let e = spec.extent() as i32;
    let t = template_side(spec);
    let nine = w >= t && h >= t;
    // Find or build what this box needs. Rendering happens outside the lock
    // (it is milliseconds of arithmetic, and the lock is taken with interrupts
    // off), and the image is shared out by reference count, so blending a
    // large shadow does not hold anybody else's paint behind it.
    let img: Arc<Image> = if nine {
        let key = spec.key();
        let found = CACHE.lock_irq().templates.iter().find(|(k, _)| *k == key).map(|(_, i)| i.clone());
        found.unwrap_or_else(|| {
            let im = Arc::new(render(t, t, spec));
            let mut c = CACHE.lock_irq();
            if c.templates.len() >= KEEP {
                c.templates.remove(0);
            }
            c.templates.push((key, im.clone()));
            im
        })
    } else {
        let key = (w, h, spec.key().0, spec.key().1);
        let found = CACHE.lock_irq().exact.iter().find(|(k, _)| *k == key).map(|(_, i)| i.clone());
        found.unwrap_or_else(|| {
            let im = Arc::new(render(w, h, spec));
            let mut c = CACHE.lock_irq();
            if c.exact.len() >= KEEP {
                c.exact.remove(0);
            }
            c.exact.push((key, im.clone()));
            im
        })
    };

    let ow = w + 2 * e as u32;
    let oh = h + 2 * e as u32;
    // Where the template's flat middle starts, in its own coordinates; the
    // middle line is the one just past it.
    let c = if nine { (img.w - 1) / 2 } else { 0 };
    let map = |o: u32, out_len: u32, len: u32| -> u32 {
        if !nine || o < c {
            o
        } else if o >= out_len - c {
            len - (out_len - o)
        } else {
            c
        }
    };
    let color = super::blend::premul(0, alpha);
    let r = spec.radius.min(w / 2).min(h / 2);
    let (sx0, sy0) = (x + dx - e, y + dy - e);
    // Only the rows and columns the surface will take: a repaint of a few
    // rows beside a window used to build every row of its shadow and let the
    // write throw all but those away.
    let (cx0, cy0, cx1, cy1) = fb.scissor();
    let ox0 = (cx0 as i64 - sx0 as i64).clamp(0, ow as i64) as u32;
    let ox1 = (cx1 as i64 - sx0 as i64).clamp(0, ow as i64) as u32;
    let oy0 = (cy0 as i64 - sy0 as i64).clamp(0, oh as i64) as u32;
    let oy1 = (cy1 as i64 - sy0 as i64).clamp(0, oh as i64) as u32;
    if ox0 >= ox1 || oy0 >= oy1 {
        return;
    }
    let mut row: Vec<u8> = alloc::vec![0; (ox1 - ox0) as usize];
    let left = sx0 + ox0 as i32;
    for oy in oy0..oy1 {
        let sy = sy0 + oy as i32;
        if sy < 0 || sy >= fb.height() as i32 {
            continue;
        }
        let ty = map(oy, oh, img.h);
        let src = &img.a[(ty * img.w) as usize..((ty + 1) * img.w) as usize];
        for (i, v) in row.iter_mut().enumerate() {
            *v = src[map(ox0 + i as u32, ow, img.w) as usize];
        }
        // Zero what the box itself covers on this row (in screen space, so
        // without the offset), interior only: an edge pixel the window only
        // partly covers keeps its shadow, since what shows between the window's
        // anti-aliased edge and the wall is meant to be shadow.
        if let Some((a, b)) = interior(sy - y, w, h, r) {
            let lo = (x + a as i32 - left).clamp(0, row.len() as i32) as usize;
            let hi = (x + b as i32 - left).clamp(0, row.len() as i32) as usize;
            if lo < hi {
                row[lo..hi].fill(0);
            }
        }
        let skip = (-left).max(0) as usize;
        if skip >= row.len() {
            continue;
        }
        fb.over_mask_row((left + skip as i32) as u32, sy as u32, &row[skip..], color);
    }
}

/// The columns of row `ry` (relative to the box's top) the box covers fully,
/// as `[a, b)`, or nothing if the row is outside it.
fn interior(ry: i32, w: u32, h: u32, r: u32) -> Option<(u32, u32)> {
    if ry < 0 || ry >= h as i32 {
        return None;
    }
    let ry = ry as u32;
    let edge = ry.min(h - 1 - ry);
    // Inside the rounded corners, the inset is where the arc is; a pixel is
    // fully covered once its whole width is inside, so it is rounded up.
    let inset = if edge >= r {
        0
    } else {
        let dy = r as f32 - (edge as f32 + 0.5);
        let half = libm_sqrt((r * r) as f32 - dy * dy);
        ((r as f32 - half) + 1.0) as u32
    };
    (2 * inset < w).then_some((inset, w - inset))
}

/// How far a shadow cast with `spec` at `(dx, dy)` reaches past its box, as
/// `(left, top, right, bottom)` -- what a layer's bounds must grow by.
pub fn reach(spec: Spec, dx: i32, dy: i32) -> (u32, u32, u32, u32) {
    let e = spec.extent() as i32;
    (
        (e - dx).max(0) as u32,
        (e - dy).max(0) as u32,
        (e + dx).max(0) as u32,
        (e + dy).max(0) as u32,
    )
}

/// `video shadows`: windows on a light ground and a dark one, two styles each.
pub fn proof(fb: &Framebuffer) {
    use super::Color;
    let (w, h) = (fb.width(), fb.height());
    super::grad::vertical(fb, 0, 0, w, h / 2, &[(0.0, Color::new(236, 240, 246)), (1.0, Color::new(214, 222, 234))]);
    super::grad::vertical(fb, 0, h / 2, w, h - h / 2, &[(0.0, Color::new(20, 44, 72)), (1.0, Color::new(14, 26, 44))]);
    let k = (w as f32 / 960.0).min(h as f32 / 540.0);
    let px = |v: f32| (v * k) as i32;
    let face = super::blend::opaque(fb.encode(Color::new(250, 251, 253)));
    let styles = [
        (Spec { radius: (8.0 * k) as u32, sigma: 14.0 * k / 1.25 }, 0, px(5.0), 100u8),
        (Spec { radius: (8.0 * k) as u32, sigma: 8.0 * k / 1.25 }, 0, px(3.0), 64u8),
        (Spec { radius: (6.0 * k) as u32, sigma: 4.0 * k / 1.25 }, 0, px(2.0), 90u8),
    ];
    for (half, base) in [(0, 60.0), (1, 330.0)] {
        let _ = half;
        let mut x = 60.0;
        for (spec, dx, dy, a) in styles {
            let (bx, by, bw, bh) = (px(x), px(base), px(240.0) as u32, px(150.0) as u32);
            cast(fb, bx, by, bw, bh, spec, dx, dy, a);
            super::shape::rrect(fb, super::shape::Box::new(bx, by, bw, bh), spec.radius, face);
            x += 300.0;
        }
        // A tooltip-sized box, which takes the exact path.
        let spec = Spec { radius: (4.0 * k) as u32, sigma: 6.0 * k / 1.25 };
        let (bx, by, bw, bh) = (px(60.0), px(base + 180.0), px(120.0) as u32, px(24.0) as u32);
        cast(fb, bx, by, bw, bh, spec, 0, px(2.0), 90);
        super::shape::rrect(fb, super::shape::Box::new(bx, by, bw, bh), spec.radius, face);
    }
}

/// `diag shadow`.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    let spec = Spec { radius: 8, sigma: 6.0 };
    let t = template_side(spec);
    let im = render(t, t, spec);
    let at = |x: u32, y: u32| im.a[(y * im.w + x) as usize] as i32;
    let n = im.w;
    let mid = (n - 1) / 2;

    // Mass: blurring moves coverage and creates none, so the image sums to
    // the box's own area.
    let r = spec.radius as f32;
    let area = (t * t) as f32 - (4.0 - core::f32::consts::PI) * r * r;
    let mass: f32 = im.a.iter().map(|&v| v as f32 / 255.0).sum();
    kprintln!("       template {}x{}, mass {:.1} against an area of {:.1}", n, n, mass, area);
    claim("a blurred box keeps its area to within one per cent", (mass - area).abs() / area < 0.01);
    claim("its middle is full and its outer border empty",
        at(mid, mid) >= 254 && at(0, mid) == 0 && at(mid, 0) == 0 && at(0, 0) == 0);
    let mut sym = true;
    for y in 0..n {
        for x in 0..n {
            sym &= at(x, y) == at(n - 1 - x, y) && at(x, y) == at(x, n - 1 - y) && at(x, y) == at(y, x);
        }
    }
    claim("it is symmetric four ways and across the diagonal", sym);
    // A straight edge blurred by a Gaussian is half intensity exactly at the edge.
    let e = spec.extent();
    let edge = (at(e - 1, mid) + at(e, mid)) / 2;
    claim("half intensity lies on the box's own edge", (edge - 128).abs() <= 4);
    let mut mono = true;
    for x in 1..=mid {
        mono &= at(x, mid) >= at(x - 1, mid);
    }
    claim("and it falls away monotonically outward", mono);
    // The middle is flat, which is what makes a nine-slice exact.
    let flat = (0..n).all(|y| at(mid - 1, y) == at(mid, y) && at(mid + 1, y) == at(mid, y));
    claim("the template's middle columns are identical, so slicing it is exact", flat);

    // Cast onto memory: a wide box's edge row equals an exact render's.
    let (fw, fh) = (300u32, 200u32);
    let mut px: Vec<u32> = alloc::vec![0xFFFF_FFFF; (fw * fh) as usize];
    let fb = unsafe { Framebuffer::over_ram(px.as_mut_ptr() as u64, fw, fh, fw, super::Format::Bgrx) };
    let (bx, by, bw, bh) = (60i32, 40i32, 180u32, 120u32);
    cast(&fb, bx, by, bw, bh, spec, 0, 0, 255);
    let exact = render(bw, bh, spec);
    let mut worst = 0i32;
    for oy in 0..exact.h {
        for ox in 0..exact.w {
            let (sx, sy) = (bx - e as i32 + ox as i32, by - e as i32 + oy as i32);
            if sx < 0 || sy < 0 || sx >= fw as i32 || sy >= fh as i32 {
                continue;
            }
            // Outside the box only; inside it is zeroed by design.
            if interior(sy - by, bw, bh, spec.radius).is_some_and(|(a, b)| (sx - bx) >= a as i32 && (sx - bx) < b as i32) {
                continue;
            }
            let got = 255 - (px[(sy as u32 * fw + sx as u32) as usize] & 0xff) as i32;
            let want = exact.a[(oy * exact.w + ox) as usize] as i32;
            worst = worst.max((got - want).abs());
        }
    }
    kprintln!("       nine-slice against an exact render: worst {} of 255", worst);
    claim("a sliced shadow matches an exact one to within two levels", worst <= 2);
    let centre = px[((by as u32 + bh / 2) * fw + (bx as u32 + bw / 2)) as usize];
    claim("nothing is cast under the box itself", centre & 0x00FF_FFFF == 0x00FF_FFFF);
    let far = px[(5 * fw + 5) as usize];
    claim("nothing is cast past its extent", far & 0x00FF_FFFF == 0x00FF_FFFF);
    let (l, t2, rr, b) = reach(spec, 0, 4);
    claim("an offset shadow reaches further on its own side", b == l + 4 && t2 + 4 == l && rr == l);
    ok
}

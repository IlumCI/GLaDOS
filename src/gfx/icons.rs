//! The desktop's pictograms, drawn as lit objects.
//!
//! The operator chose, from mockups, the object itself -- a folder, a palette,
//! a gear -- lit from above with a soft shadow and no tile behind it: Vista's
//! Explorer icons rather than its Start-menu tiles. They replace a set built
//! from axis-aligned rectangles, frames and one-pixel glosses, which is what
//! a 40-pixel icon looks like when nothing can draw a curve.
//!
//! **Every icon is a stack of signed-distance elements** in a 48-unit design
//! space: a shape, a vertical ramp to fill it, an optional rim and gloss. One
//! rasteriser draws them all at any size, anti-aliased, so the wall, the
//! taskbar and a title bar share one drawing and cannot drift. Each size of
//! each icon is drawn once into premultiplied ARGB and kept; laying one down
//! is then a row of OVER.

use alloc::sync::Arc;
use alloc::vec::Vec;

use super::shape::{libm_sqrt, sd_rbox};
use super::Framebuffer;
use crate::sync::Spin;

pub const TERM: usize = 0;
pub const PROGRAMS: usize = 1;
pub const FILES: usize = 2;
pub const TODO: usize = 3;
pub const NET: usize = 4;
pub const PAINT: usize = 5;
pub const WRITE: usize = 6;
pub const MINES: usize = 7;
pub const ORACLE: usize = 8;
pub const SETTINGS: usize = 9;
/// The Start orb, at rest and under the pointer: a sphere of dark glass with
/// the Aperture mark lit inside it -- Vista's orb, carrying this machine's
/// mark where Vista carried its flag.
pub const ORB: usize = 10;
pub const ORB_HOT: usize = 11;
/// The mark alone, for a caption: light for the focused window's glass and
/// dark for the pale glass of the others, as the title text beside it is.
pub const MARK: usize = 12;
pub const MARK_IDLE: usize = 13;
pub const COUNT: usize = 14;

type Rgba = [f32; 4];

const fn c(r: u8, g: u8, b: u8) -> Rgba {
    [r as f32, g as f32, b as f32, 255.0]
}

/// A premultiplied image being drawn, in f32 so layers blend exactly.
struct Canvas {
    n: usize,
    k: f32,
    px: Vec<[f32; 4]>,
}

impl Canvas {
    fn new(n: usize) -> Canvas {
        Canvas { n, k: n as f32 / 48.0, px: alloc::vec![[0.0; 4]; n * n] }
    }

    /// Fill the element whose signed distance (in design units) is `sd`,
    /// with a vertical ramp from `top` at `y0` to `bot` at `y1`, OVER what is
    /// there. Anti-aliased over a pixel's width.
    fn fill(&mut self, sd: &dyn Fn(f32, f32) -> f32, top: Rgba, bot: Rgba, y0: f32, y1: f32, alpha: f32) {
        let px_units = 1.0 / self.k;
        for j in 0..self.n {
            for i in 0..self.n {
                let (x, y) = ((i as f32 + 0.5) / self.k, (j as f32 + 0.5) / self.k);
                let d = sd(x, y) / px_units;
                let cov = (0.5 - d).clamp(0.0, 1.0) * alpha;
                if cov <= 0.0 {
                    continue;
                }
                let t = ((y - y0) / (y1 - y0).max(0.01)).clamp(0.0, 1.0);
                let col: Rgba = core::array::from_fn(|q| top[q] * (1.0 - t) + bot[q] * t);
                let a = col[3] / 255.0 * cov;
                let p = &mut self.px[j * self.n + i];
                for q in 0..3 {
                    p[q] = col[q] / 255.0 * a + p[q] * (1.0 - a);
                }
                p[3] = a + p[3] * (1.0 - a);
            }
        }
    }

    /// A soft shadow of the element `sd`, offset down, under what is there:
    /// the distance field smoothed rather than blurred, as the pointer's is.
    fn shadow(&mut self, sd: &dyn Fn(f32, f32) -> f32, dy: f32, soft: f32, alpha: f32) {
        for j in 0..self.n {
            for i in 0..self.n {
                let (x, y) = ((i as f32 + 0.5) / self.k, (j as f32 + 0.5) / self.k);
                let d = sd(x, y - dy);
                let t = ((d + soft) / (2.5 * soft)).clamp(0.0, 1.0);
                let a = (1.0 - t * t * (3.0 - 2.0 * t)) * alpha;
                let p = &mut self.px[j * self.n + i];
                // Under: only where the image is not already opaque.
                let under = a * (1.0 - p[3]);
                p[3] += under;
            }
        }
    }

    fn finish(self) -> Vec<u32> {
        self.px
            .iter()
            .map(|p| {
                let q = |v: f32| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u32;
                // BGRX lane order; `lay` swaps for an RGBX screen.
                q(p[2]) | q(p[1]) << 8 | q(p[0]) << 16 | q(p[3]) << 24
            })
            .collect()
    }
}

// --- signed distances, in design units ------------------------------------

fn rrect(x: f32, y: f32, x0: f32, y0: f32, x1: f32, y1: f32, r: f32) -> f32 {
    sd_rbox(x, y, (x0 + x1) / 2.0, (y0 + y1) / 2.0, (x1 - x0) / 2.0, (y1 - y0) / 2.0, r)
}

fn circle(x: f32, y: f32, cx: f32, cy: f32, r: f32) -> f32 {
    libm_sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)) - r
}

fn capsule(x: f32, y: f32, ax: f32, ay: f32, bx: f32, by: f32, r: f32) -> f32 {
    let (ex, ey) = (bx - ax, by - ay);
    let (px, py) = (x - ax, y - ay);
    let t = ((px * ex + py * ey) / (ex * ex + ey * ey).max(1e-6)).clamp(0.0, 1.0);
    libm_sqrt((px - ex * t) * (px - ex * t) + (py - ey * t) * (py - ey * t)) - r
}

/// An ellipse, approximately: a circle in a stretched space, which is exact
/// on the axes and close enough elsewhere for an icon.
fn ellipse(x: f32, y: f32, cx: f32, cy: f32, rx: f32, ry: f32) -> f32 {
    let (u, v) = ((x - cx) / rx, (y - cy) / ry);
    (libm_sqrt(u * u + v * v) - 1.0) * rx.min(ry)
}

fn union(a: f32, b: f32) -> f32 {
    a.min(b)
}

fn minus(a: f32, b: f32) -> f32 {
    a.max(-b)
}

/// The gloss every object carries: a white wash over its upper part.
const GLOSS_TOP: Rgba = [255.0, 255.0, 255.0, 90.0];
const GLOSS_BOT: Rgba = [255.0, 255.0, 255.0, 0.0];

fn draw(kind: usize, n: usize) -> Vec<u32> {
    let mut cv = Canvas::new(n);
    match kind {
        TERM => {
            let body = |x: f32, y: f32| rrect(x, y, 4.0, 8.0, 44.0, 40.0, 5.0);
            cv.shadow(&body, 2.0, 2.0, 0.45);
            cv.fill(&body, c(84, 92, 108), c(36, 40, 50), 8.0, 40.0, 1.0);
            let screen = |x: f32, y: f32| rrect(x, y, 7.0, 11.0, 41.0, 37.0, 3.5);
            cv.fill(&screen, c(20, 24, 34), c(8, 10, 16), 11.0, 37.0, 1.0);
            let chev = |x: f32, y: f32| union(capsule(x, y, 12.0, 18.0, 18.0, 23.0, 1.4), capsule(x, y, 18.0, 23.0, 12.0, 28.0, 1.4));
            cv.fill(&chev, c(255, 160, 60), c(255, 120, 30), 18.0, 28.0, 1.0);
            let cursor = |x: f32, y: f32| rrect(x, y, 21.0, 27.0, 30.0, 29.5, 1.0);
            cv.fill(&cursor, c(235, 240, 250), c(200, 210, 225), 27.0, 30.0, 1.0);
            let gloss = |x: f32, y: f32| rrect(x, y, 5.0, 9.0, 43.0, 20.0, 4.5);
            cv.fill(&gloss, [255.0, 255.0, 255.0, 40.0], GLOSS_BOT, 9.0, 20.0, 1.0);
        }
        PROGRAMS => {
            let disc = |x: f32, y: f32| circle(x, y, 24.0, 23.0, 19.0);
            cv.shadow(&disc, 2.0, 2.0, 0.45);
            // The Aperture mark, through the one geometry for it. Its coverage
            // anti-aliases over one unit, so it is asked in pixels, and the
            // answer handed back as the distance that gives that coverage.
            let k = cv.k;
            let mark = move |x: f32, y: f32| {
                let cov = super::splash::aperture_coverage((x - 24.0) * k, (y - 23.0) * k, 19.0 * k);
                (0.5 - cov) / k
            };
            cv.fill(&mark, c(255, 222, 120), c(240, 96, 30), 4.0, 42.0, 1.0);
            cv.fill(&|x, y| circle(x, y, 24.0, 23.0, 19.0).max(-(16.0 - y)), GLOSS_TOP, GLOSS_BOT, 4.0, 18.0, 0.6);
        }
        FILES => {
            let back = |x: f32, y: f32| union(rrect(x, y, 4.0, 12.0, 44.0, 40.0, 3.0), rrect(x, y, 4.0, 8.0, 20.0, 15.0, 2.5));
            cv.shadow(&back, 2.0, 2.0, 0.45);
            cv.fill(&back, c(214, 150, 40), c(176, 112, 24), 8.0, 40.0, 1.0);
            let paper = |x: f32, y: f32| rrect(x, y, 8.0, 14.0, 40.0, 30.0, 1.5);
            cv.fill(&paper, c(250, 250, 248), c(226, 228, 232), 14.0, 30.0, 1.0);
            let front = |x: f32, y: f32| rrect(x, y, 3.0, 19.0, 45.0, 40.0, 3.0);
            cv.fill(&front, c(255, 214, 110), c(240, 166, 44), 19.0, 40.0, 1.0);
            cv.fill(&|x, y| rrect(x, y, 4.0, 20.0, 44.0, 28.0, 2.5), GLOSS_TOP, GLOSS_BOT, 20.0, 28.0, 0.7);
        }
        TODO => {
            let pad = |x: f32, y: f32| rrect(x, y, 9.0, 6.0, 39.0, 43.0, 2.5);
            cv.shadow(&pad, 2.0, 2.0, 0.45);
            cv.fill(&pad, c(255, 255, 255), c(232, 236, 242), 6.0, 43.0, 1.0);
            for (k, yy) in [16.0f32, 24.0, 32.0].iter().enumerate() {
                let yy = *yy;
                let line = move |x: f32, y: f32| rrect(x, y, 21.0, yy - 0.8, 34.0, yy + 0.8, 0.8);
                cv.fill(&line, c(120, 140, 170), c(120, 140, 170), 0.0, 48.0, 1.0);
                let tick = move |x: f32, y: f32| union(capsule(x, y, 13.0, yy, 15.0, yy + 2.0, 1.0), capsule(x, y, 15.0, yy + 2.0, 18.5, yy - 2.5, 1.0));
                let col = if k < 2 { c(60, 170, 90) } else { c(200, 205, 215) };
                cv.fill(&tick, col, col, 0.0, 48.0, 1.0);
            }
            let clip = |x: f32, y: f32| rrect(x, y, 18.0, 3.0, 30.0, 9.0, 2.0);
            cv.fill(&clip, c(190, 196, 206), c(120, 128, 140), 3.0, 9.0, 1.0);
        }
        NET => {
            let globe = |x: f32, y: f32| circle(x, y, 24.0, 24.0, 18.0);
            cv.shadow(&globe, 2.0, 2.0, 0.45);
            cv.fill(&globe, c(110, 190, 250), c(20, 90, 180), 6.0, 42.0, 1.0);
            let meridians = |x: f32, y: f32| {
                let e1 = ellipse(x, y, 24.0, 24.0, 8.0, 18.0).abs() - 0.8;
                let e2 = (y - 24.0).abs() - 0.8;
                let e3 = (ellipse(x, y, 24.0, 16.0, 15.0, 3.0)).abs() - 0.7;
                let e4 = (ellipse(x, y, 24.0, 32.0, 15.0, 3.0)).abs() - 0.7;
                union(union(e1, e2), union(e3, e4)).max(circle(x, y, 24.0, 24.0, 17.0))
            };
            cv.fill(&meridians, [230.0, 245.0, 255.0, 170.0], [200.0, 230.0, 255.0, 120.0], 6.0, 42.0, 1.0);
            cv.fill(&|x, y| ellipse(x, y, 20.0, 14.0, 10.0, 6.0), GLOSS_TOP, GLOSS_BOT, 8.0, 20.0, 0.8);
        }
        PAINT => {
            let palette = |x: f32, y: f32| minus(ellipse(x, y, 24.0, 25.0, 20.0, 16.0), circle(x, y, 32.0, 32.0, 4.0));
            cv.shadow(&palette, 2.0, 2.0, 0.45);
            cv.fill(&palette, c(246, 232, 204), c(214, 186, 140), 9.0, 41.0, 1.0);
            let dots = [(14.0f32, 22.0f32, c(230, 70, 60)), (21.0, 15.0, c(250, 200, 50)), (30.0, 15.0, c(90, 190, 110)), (36.0, 22.0, c(60, 150, 240))];
            for (dx, dy, col) in dots {
                let dot = move |x: f32, y: f32| circle(x, y, dx, dy, 3.6);
                let lighter: Rgba = core::array::from_fn(|q| if q < 3 { (col[q] + 50.0).min(255.0) } else { 255.0 });
                cv.fill(&dot, lighter, col, dy - 3.6, dy + 3.6, 1.0);
            }
        }
        WRITE => {
            let page = |x: f32, y: f32| minus(rrect(x, y, 10.0, 5.0, 38.0, 43.0, 2.0), capsule(x, y, 38.0, 5.0, 38.0, 5.0, 7.0));
            cv.shadow(&page, 2.0, 2.0, 0.45);
            cv.fill(&page, c(255, 255, 255), c(228, 232, 238), 5.0, 43.0, 1.0);
            let fold = |x: f32, y: f32| rrect(x, y, 31.0, 5.0, 38.0, 12.0, 1.0).max(-(x - y - 26.0));
            cv.fill(&fold, c(210, 216, 226), c(180, 188, 200), 5.0, 12.0, 1.0);
            for yy in [17.0f32, 22.0, 27.0, 32.0] {
                let line = move |x: f32, y: f32| rrect(x, y, 14.0, yy - 0.7, 31.0, yy + 0.7, 0.7);
                cv.fill(&line, c(150, 162, 182), c(150, 162, 182), 0.0, 48.0, 1.0);
            }
            let pen = |x: f32, y: f32| capsule(x, y, 26.0, 40.0, 42.0, 22.0, 2.2);
            cv.fill(&pen, c(255, 170, 70), c(220, 110, 30), 22.0, 40.0, 1.0);
            let nib = |x: f32, y: f32| capsule(x, y, 24.5, 41.8, 26.0, 40.0, 1.2);
            cv.fill(&nib, c(40, 44, 52), c(40, 44, 52), 38.0, 43.0, 1.0);
        }
        MINES => {
            let spikes = |x: f32, y: f32| {
                let mut d = circle(x, y, 24.0, 25.0, 13.0);
                for k in 0..8 {
                    let a = k as f32 * core::f32::consts::FRAC_PI_4;
                    let (cs, sn) = (cos(a), sin(a));
                    d = union(d, capsule(x, y, 24.0, 25.0, 24.0 + cs * 19.0, 25.0 + sn * 19.0, 1.6));
                }
                d
            };
            cv.shadow(&spikes, 2.0, 2.0, 0.45);
            cv.fill(&spikes, c(70, 76, 88), c(18, 20, 26), 6.0, 44.0, 1.0);
            cv.fill(&|x, y| circle(x, y, 19.5, 20.5, 4.0), [255.0, 255.0, 255.0, 200.0], [255.0, 255.0, 255.0, 40.0], 16.5, 24.5, 1.0);
        }
        ORACLE => {
            let eye = |x: f32, y: f32| ellipse(x, y, 24.0, 24.0, 20.0, 11.0);
            cv.shadow(&eye, 2.0, 2.0, 0.45);
            cv.fill(&eye, c(255, 255, 255), c(220, 226, 236), 13.0, 35.0, 1.0);
            let iris = |x: f32, y: f32| circle(x, y, 24.0, 24.0, 8.5).max(eye(x, y));
            cv.fill(&iris, c(255, 196, 90), c(220, 96, 20), 15.5, 32.5, 1.0);
            let pupil = |x: f32, y: f32| circle(x, y, 24.0, 24.0, 3.8);
            cv.fill(&pupil, c(20, 18, 16), c(20, 18, 16), 20.0, 28.0, 1.0);
            cv.fill(&|x, y| circle(x, y, 21.0, 21.0, 1.8), [255.0, 255.0, 255.0, 230.0], [255.0, 255.0, 255.0, 230.0], 19.0, 23.0, 1.0);
        }
        ORB | ORB_HOT => {
            let hot = kind == ORB_HOT;
            let sphere = |x: f32, y: f32| circle(x, y, 24.0, 24.0, 22.0);
            cv.shadow(&sphere, 1.5, 1.5, 0.5);
            // Dark glass, deepest at the foot, as a sphere lit from above is.
            if hot {
                cv.fill(&sphere, c(58, 96, 150), c(14, 30, 58), 2.0, 46.0, 1.0);
            } else {
                cv.fill(&sphere, c(40, 66, 108), c(8, 16, 34), 2.0, 46.0, 1.0);
            }
            // The light that comes up through the bottom of Vista's orb:
            // a glow inside the sphere's lower half, strongest at the rim.
            let foot = |x: f32, y: f32| sphere(x, y).max(ellipse(x, y, 24.0, 40.0, 19.0, 12.0));
            let glow = if hot { [120.0, 210.0, 255.0, 190.0] } else { [80.0, 170.0, 240.0, 130.0] };
            cv.fill(&foot, [glow[0], glow[1], glow[2], 0.0], glow, 26.0, 46.0, 1.0);
            // The mark, through the one geometry for it, asked in pixels.
            let k = cv.k;
            let mark = move |x: f32, y: f32| {
                let cov = super::splash::aperture_coverage((x - 24.0) * k, (y - 24.0) * k, 14.5 * k);
                (0.5 - cov) / k
            };
            if hot {
                cv.fill(&mark, c(255, 240, 180), c(255, 140, 50), 9.0, 39.0, 1.0);
            } else {
                cv.fill(&mark, c(255, 222, 120), c(240, 96, 30), 9.0, 39.0, 1.0);
            }
            // The reflection: a bright cap over the top half, and a rim of
            // light all the way round where the glass turns away.
            let cap = |x: f32, y: f32| sphere(x, y).max(ellipse(x, y, 24.0, 14.0, 17.0, 11.0));
            cv.fill(&cap, [255.0, 255.0, 255.0, 150.0], GLOSS_BOT, 3.0, 24.0, 1.0);
            let rim = |x: f32, y: f32| sphere(x, y).abs() - 0.7;
            cv.fill(&rim, [210.0, 230.0, 255.0, 170.0], [120.0, 190.0, 255.0, 120.0], 2.0, 46.0, 1.0);
        }
        MARK | MARK_IDLE => {
            let k = cv.k;
            let mark = move |x: f32, y: f32| {
                let cov = super::splash::aperture_coverage((x - 24.0) * k, (y - 24.0) * k, 21.0 * k);
                (0.5 - cov) / k
            };
            if kind == MARK {
                // From the disc, not from `mark`: a distance rebuilt from
                // coverage is the same half-unit everywhere outside the logo,
                // so a shadow cast from it darkens the whole square.
                cv.shadow(&|x, y| circle(x, y, 24.0, 24.0, 21.0), 1.5, 1.5, 0.35);
                cv.fill(&mark, c(255, 255, 255), c(222, 232, 246), 3.0, 45.0, 1.0);
            } else {
                cv.fill(&mark, c(70, 92, 108), c(40, 60, 74), 3.0, 45.0, 1.0);
            }
        }
        _ => {
            // Settings: a gear, machined, with an orange hub.
            let gear = |x: f32, y: f32| {
                let (dx, dy) = (x - 24.0, y - 24.0);
                let r = libm_sqrt(dx * dx + dy * dy);
                let a = atan2(dy, dx);
                // Eight teeth: the radius steps out in each tooth's sector,
                // with the corners softened by blending the two radii.
                let s = cos(a * 8.0);
                let teeth = 15.0 + 4.0 * smooth((s - 0.2) * 3.0);
                minus(r - teeth, circle(x, y, 24.0, 24.0, 5.5))
            };
            cv.shadow(&gear, 2.0, 2.0, 0.45);
            cv.fill(&gear, c(214, 220, 230), c(120, 130, 146), 5.0, 43.0, 1.0);
            let hub = |x: f32, y: f32| minus(circle(x, y, 24.0, 24.0, 8.5), circle(x, y, 24.0, 24.0, 5.5));
            cv.fill(&hub, c(255, 170, 70), c(230, 110, 30), 15.5, 32.5, 1.0);
        }
    }
    cv.finish()
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

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

fn cos(x: f32) -> f32 {
    sin(x + core::f32::consts::FRAC_PI_2)
}

/// atan2 to a few thousandths of a radian, which a gear's teeth do not notice.
fn atan2(y: f32, x: f32) -> f32 {
    let (ax, ay) = (x.abs(), y.abs());
    if ax < 1e-9 && ay < 1e-9 {
        return 0.0;
    }
    let (t, swap) = if ay > ax { (ax / ay, true) } else { (ay / ax, false) };
    // A polynomial for atan on [0, 1].
    let t2 = t * t;
    let mut a = t * (0.999_866 + t2 * (-0.330_299_5 + t2 * (0.180_141 + t2 * (-0.085_133 + t2 * 0.020_835_1))));
    if swap {
        a = core::f32::consts::FRAC_PI_2 - a;
    }
    if x < 0.0 {
        a = core::f32::consts::PI - a;
    }
    if y < 0.0 { -a } else { a }
}

static CACHE: Spin<Vec<((usize, usize), Arc<Vec<u32>>)>> = Spin::new(Vec::new());

/// The icon `kind` at `n` pixels square, premultiplied ARGB in BGRX lane
/// order, drawn on first use and kept.
fn image(kind: usize, n: usize) -> Arc<Vec<u32>> {
    let key = (kind, n);
    if let Some(i) = CACHE.lock_irq().iter().find(|(k, _)| *k == key).map(|(_, i)| i.clone()) {
        return i;
    }
    let im = Arc::new(draw(kind, n));
    let mut c = CACHE.lock_irq();
    if c.len() >= 48 {
        c.remove(0);
    }
    c.push((key, im.clone()));
    im
}

/// Lay icon `kind`, `n` pixels square, with its top-left at `(x, y)`.
pub fn lay(fb: &Framebuffer, kind: usize, x: u32, y: u32, n: u32) {
    if n == 0 {
        return;
    }
    let im = image(kind.min(COUNT - 1), n as usize);
    let swap = fb.format() == super::Format::Rgbx;
    let mut row: Vec<u32> = alloc::vec![0; n as usize];
    for j in 0..n {
        let src = &im[(j * n) as usize..((j + 1) * n) as usize];
        if swap {
            for (o, &p) in row.iter_mut().zip(src.iter()) {
                *o = (p & 0xFF00_FF00) | (p >> 16 & 0xFF) | (p & 0xFF) << 16;
            }
            fb.over_row(x, y + j, &row);
        } else {
            fb.over_row(x, y + j, src);
        }
    }
}

/// `video icons`: every icon at the wall's size and the taskbar's, on the
/// wallpaper.
pub fn proof(fb: &Framebuffer) {
    super::desk::paint_wallpaper(fb);
    let (w, h) = (fb.width(), fb.height());
    let k = (w as f32 / 960.0).min(h as f32 / 540.0);
    for (row, size) in [(0u32, (48.0 * k) as u32), (1, (30.0 * k) as u32), (2, (20.0 * k) as u32)] {
        for kind in 0..COUNT {
            let x = (560.0 * k) as u32 + kind as u32 * (38.0 * k) as u32;
            let y = (300.0 * k) as u32 + row * (60.0 * k) as u32;
            lay(fb, kind, x, y, size);
        }
    }
}

/// `diag icons`.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    let mut all_inked = true;
    let mut corners_clear = true;
    let mut premul = true;
    for kind in 0..COUNT {
        let im = draw(kind, 48);
        let ink: u32 = im.iter().filter(|&&p| p >> 24 > 128).count() as u32;
        all_inked &= ink > 300;
        corners_clear &= [0usize, 47, 47 * 48, 48 * 48 - 1].iter().all(|&i| im[i] >> 24 < 16);
        premul &= im.iter().all(|&p| (0..3).all(|l| (p >> (8 * l)) & 0xff <= p >> 24));
    }
    claim("every icon draws a solid object at 48 pixels", all_inked);
    claim("and leaves its corners clear, so it sits on anything", corners_clear);
    claim("every pixel is premultiplied: no channel exceeds its alpha", premul);
    let a = image(SETTINGS, 30);
    let b = image(SETTINGS, 30);
    claim("a size drawn once is kept, not drawn again", Arc::ptr_eq(&a, &b));
    let edge = (atan2(1.0, 1.0) - core::f32::consts::FRAC_PI_4).abs() < 0.002
        && (atan2(-1.0, -1.0) + 3.0 * core::f32::consts::FRAC_PI_4).abs() < 0.002;
    claim("atan2 is right in two opposite quadrants", edge);
    ok
}

//! Flip 3D: the windows stood in a stack, leaning away, front to back.
//!
//! Vista's Super-Tab, as a homage rather than a reconstruction: 2.5D, not a
//! 3D pipeline. Each window is its image (phase 11) drawn into a *quad* --
//! a left and a right edge, each with its own height, so the near edge is
//! taller than the far one and the window reads as turned away from the
//! eye. That is the whole of the perspective, and it is what a rotation
//! about the vertical axis looks like at the angles the stack uses.
//!
//! `layout` is pure, so where every window stands for any stack position
//! and any instant of a cycle is checkable at boot without a desktop.
//! A cycle is a continuous `offset`: window `k` stands at depth
//! `(k - offset) mod n`, so turning the stack is one number moving, and the
//! window leaving the front fades out in front of the stack as it re-enters
//! at the back.

use alloc::vec::Vec;

use super::scale::Image;
use super::Framebuffer;
use crate::ai::tensor::{expf, floorf};

/// A window as drawn: its left and right edges, the height of each, the
/// height of its middle at each, and how opaque.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quad {
    pub x0: f32,
    pub x1: f32,
    pub cy0: f32,
    pub cy1: f32,
    pub h0: f32,
    pub h1: f32,
    pub alpha: f32,
}

impl Quad {
    /// A window lying flat where it is on the desktop.
    pub fn flat(x: f32, y: f32, w: f32, h: f32) -> Quad {
        Quad { x0: x, x1: x + w, cy0: y + h / 2.0, cy1: y + h / 2.0, h0: h, h1: h, alpha: 1.0 }
    }

    pub fn lerp(a: Quad, b: Quad, t: f32) -> Quad {
        let l = |p: f32, q: f32| p + (q - p) * t;
        Quad {
            x0: l(a.x0, b.x0),
            x1: l(a.x1, b.x1),
            cy0: l(a.cy0, b.cy0),
            cy1: l(a.cy1, b.cy1),
            h0: l(a.h0, b.h0),
            h1: l(a.h1, b.h1),
            alpha: l(a.alpha, b.alpha),
        }
    }

    /// The rectangle the quad covers, for damage and for clipping.
    pub fn bounds(&self) -> (f32, f32, f32, f32) {
        let top = (self.cy0 - self.h0 / 2.0).min(self.cy1 - self.h1 / 2.0);
        let bot = (self.cy0 + self.h0 / 2.0).max(self.cy1 + self.h1 / 2.0);
        (self.x0.min(self.x1), top, self.x0.max(self.x1), bot)
    }
}

/// How each step back shrinks a window, how far up and right it stands, and
/// how much it leans.
const SHRINK: f32 = 0.86;
const STEP_X: f32 = 0.085;
const STEP_Y: f32 = 0.072;
const LEAN: f32 = 0.11;

/// Where a window of `w` x `h` stands at depth `d` (0 the front; fractional
/// mid-cycle) on a `sw` x `sh` screen.
pub fn at_depth(w: f32, h: f32, d: f32, sw: f32, sh: f32) -> Quad {
    // Fitted into a box a little under half the screen, keeping its shape.
    let fit = (0.46 * sw / w.max(1.0)).min(0.50 * sh / h.max(1.0));
    let s = fit * expf(d * crate::ai::tensor::lnf(SHRINK));
    let (bw, bh) = (w * s, h * s);
    let cx = sw * (0.40 + STEP_X * d);
    let cy = sh * (0.56 - STEP_Y * d);
    // Turned away: narrower than it is wide flat, the near (left) edge
    // taller and the far one shorter.
    let pw = bw * 0.90;
    let alpha = if d < 0.0 { (1.0 + d).clamp(0.0, 1.0) } else { (1.0 - 0.09 * d).clamp(0.35, 1.0) };
    Quad {
        x0: cx - pw / 2.0,
        x1: cx + pw / 2.0,
        cy0: cy,
        cy1: cy - bh * 0.04,
        h0: bh * (1.0 + LEAN),
        h1: bh * (1.0 - LEAN),
        alpha,
    }
}

/// The depth window `k` of `n` stands at when the stack has turned by
/// `offset`. In `[0, n)`, except that the window on its way out of the
/// front stands at `(-1, 0)` -- in front of the stack and fading -- until it
/// is more than half way round.
pub fn depth(k: usize, n: usize, offset: f32) -> f32 {
    let n = n.max(1) as f32;
    let mut d = (k as f32 - offset) % n;
    if d < 0.0 {
        d += n;
    }
    if d > n - 1.0 {
        d -= n;
    }
    d
}

/// Every window's depth, deepest first: the order to draw them in.
pub fn order(n: usize, offset: f32) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = (0..n).map(|k| (k, depth(k, n, offset))).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    v
}

/// Lay `img` (premultiplied) into quad `q`, OVER what is there.
pub fn draw(fb: &Framebuffer, img: &Image, q: Quad) {
    paint(fb, None, &[(img, q)]);
}

/// One quad, made ready for rows: for every column it covers, where in the
/// image the column samples and where it starts and stops on screen. Both
/// edges of a quad are straight lines, so this is all that varies across it,
/// and computing it once per column takes a division off every pixel.
struct Prepared<'a> {
    img: &'a Image,
    xa: i32,
    /// Per column: source x in pixels, top, bottom, and image rows per
    /// screen row.
    cols: Vec<(f32, f32, f32, f32)>,
    o: u32,
    top: f32,
    bot: f32,
}

fn prepare<'a>(img: &'a Image, q: Quad, sx0: u32, sx1: u32) -> Option<Prepared<'a>> {
    if img.w == 0 || img.h == 0 || q.alpha <= 0.0 || q.x1 - q.x0 < 1.0 {
        return None;
    }
    let xa = floorf(q.x0.max(sx0 as f32)) as i32;
    let xb = -floorf(-q.x1.min(sx1 as f32)) as i32;
    if xa >= xb {
        return None;
    }
    let span = q.x1 - q.x0;
    let (iw, ih) = (img.w as f32, img.h as f32);
    let cols = (xa..xb)
        .map(|x| {
            let t = ((x as f32 + 0.5 - q.x0) / span).clamp(0.0, 1.0);
            let h = (q.h0 + (q.h1 - q.h0) * t).max(1.0);
            let cy = q.cy0 + (q.cy1 - q.cy0) * t;
            (t * iw - 0.5, cy - h / 2.0, cy + h / 2.0, ih / h)
        })
        .collect();
    let (_, top, _, bot) = q.bounds();
    Some(Prepared { img, xa, cols, o: (q.alpha.clamp(0.0, 1.0) * 255.0 + 0.5) as u32, top, bot })
}

/// Everything one band of rows needs, by pointer, for `smp::parallel_split`.
#[derive(Clone, Copy)]
struct Job {
    fb: Framebuffer,
    bg: *const u32,
    bg_w: u32,
    bg_alpha: u32,
    items: *const Prepared<'static>,
    n: usize,
    x0: u32,
    x1: u32,
    y0: u32,
}

/// The stack: the backdrop (`bg`, a screen-sized image, at an opacity), then
/// each quad, in the order given, row by row -- across every core there is,
/// since a stack covers the screen and every frame of a turn repaints it.
pub fn paint(fb: &Framebuffer, bg: Option<(&[u32], u32, u32)>, items: &[(&Image, Quad)]) {
    let (sx0, sy0, sx1, sy1) = fb.scissor();
    if sx0 >= sx1 || sy0 >= sy1 {
        return;
    }
    let prepared: Vec<Prepared> = items.iter().filter_map(|(img, q)| prepare(img, *q, sx0, sx1)).collect();
    let (bgp, bg_w, bg_alpha) = match bg {
        Some((px, w, a)) => (px.as_ptr(), w, a),
        None => (core::ptr::null(), 0, 0),
    };
    let job = Job {
        fb: *fb,
        bg: bgp,
        bg_w,
        bg_alpha,
        items: prepared.as_ptr() as *const Prepared<'static>,
        n: prepared.len(),
        x0: sx0,
        x1: sx1,
        y0: sy0,
    };
    let rows = (sy1 - sy0) as usize;
    let ctx = &job as *const Job as usize;
    if !crate::smp::parallel_split(ctx, rows_of, rows, (sx1 - sx0) as usize) {
        unsafe { rows_of(ctx, 0, rows) };
    }
}

/// Rows `lo..hi` of the stack, counted from the scissor's top.
unsafe fn rows_of(ctx: usize, lo: usize, hi: usize) {
    let j = unsafe { *(ctx as *const Job) };
    let items = unsafe { core::slice::from_raw_parts(j.items, j.n) };
    let width = (j.x1 - j.x0) as usize;
    let mut row: Vec<u32> = Vec::with_capacity(width);
    for r in lo..hi {
        let y = j.y0 + r as u32;
        if !j.bg.is_null() && j.bg_alpha > 0 {
            let src = unsafe { core::slice::from_raw_parts(j.bg.add((y * j.bg_w + j.x0) as usize), width) };
            if j.bg_alpha >= 255 {
                j.fb.blit_span(j.x0, y, src);
            } else {
                row.clear();
                row.extend(src.iter().map(|&p| super::blend::scale_px(p | 0xFF00_0000, j.bg_alpha)));
                j.fb.over_row(j.x0, y, &row);
            }
        }
        let py = y as f32 + 0.5;
        for it in items {
            if py < it.top || py >= it.bot {
                continue;
            }
            row.clear();
            let mut first: Option<i32> = None;
            let (w, h) = (it.img.w as i32, it.img.h as i32);
            for (i, &(u, top, bot, k)) in it.cols.iter().enumerate() {
                if py < top || py >= bot {
                    if first.is_some() {
                        row.push(0);
                    }
                    continue;
                }
                if first.is_none() {
                    first = Some(it.xa + i as i32);
                }
                let v = (py - top) * k - 0.5;
                let p = sample(&it.img.px, w, h, u, v);
                row.push(if it.o == 255 { p } else { super::blend::scale_px(p, it.o) });
            }
            if let Some(x) = first {
                // Trailing clear pixels are a no-op OVER and are left on.
                j.fb.over_row(x as u32, y, &row);
            }
        }
    }
}

/// Bilinear at `(u, v)` in pixel units, clamped at the edges; two lanes at
/// a time, as `blend` does.
#[inline]
fn sample(px: &[u32], w: i32, h: i32, u: f32, v: f32) -> u32 {
    let (u, v) = (u.max(0.0), v.max(0.0));
    let (x0, y0) = ((u as i32).min(w - 1), (v as i32).min(h - 1));
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let fx = ((u - x0 as f32).min(1.0) * 256.0) as u32;
    let fy = ((v - y0 as f32).min(1.0) * 256.0) as u32;
    let p = |x: i32, y: i32| px[(y * w + x) as usize];
    #[inline]
    fn lerp(a: u32, b: u32, f: u32) -> u32 {
        let g = 256 - f;
        let rb = (((a & 0x00FF_00FF) * g + (b & 0x00FF_00FF) * f + 0x0080_0080) >> 8) & 0x00FF_00FF;
        let ag = ((((a >> 8) & 0x00FF_00FF) * g + ((b >> 8) & 0x00FF_00FF) * f + 0x0080_0080)) & 0xFF00_FF00;
        rb | ag
    }
    lerp(lerp(p(x0, y0), p(x1, y0), fx), lerp(p(x0, y1), p(x1, y1), fx), fy)
}

/// `diag flip`.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    let (sw, sh) = (1280.0, 800.0);
    let front = at_depth(600.0, 500.0, 0.0, sw, sh);
    let back = at_depth(600.0, 500.0, 2.0, sw, sh);
    claim("a window further back is smaller", back.h0 < front.h0 && back.x1 - back.x0 < front.x1 - front.x0);
    claim("and stands up and to the right", back.cy0 < front.cy0 && back.x0 > front.x0);
    claim("its near edge is taller than its far one", front.h0 > front.h1);
    let (x0, y0, x1, y1) = front.bounds();
    claim("the front window is on the screen", x0 >= 0.0 && y0 >= 0.0 && x1 <= sw && y1 <= sh);
    claim(
        "depths at rest are whole and distinct",
        (0..5).map(|k| depth(k, 5, 0.0)).collect::<Vec<_>>() == alloc::vec![0.0, 1.0, 2.0, 3.0, 4.0],
    );
    claim("turning by one moves every window one place forward", (0..5).all(|k| depth(k, 5, 1.0) == if k == 0 { 4.0 } else { k as f32 - 1.0 }));
    claim("half way round, the front window is leaving in front of the stack", depth(0, 5, 0.5) < 0.0 && depth(0, 5, 0.5) > -1.0);
    let o = order(4, 0.0);
    claim("windows are drawn deepest first", o.windows(2).all(|p| p[0].1 >= p[1].1) && o.last().map(|l| l.0) == Some(0));
    let flat = Quad::flat(10.0, 20.0, 300.0, 200.0);
    claim("a quad lerps exactly onto both ends", Quad::lerp(flat, front, 0.0) == flat && Quad::lerp(flat, front, 1.0) == front);
    // Drawing: an opaque 2x2 into a flat quad covers exactly its rectangle.
    let mut buf = alloc::vec![0u32; 32 * 32];
    let fb = unsafe { Framebuffer::over_ram(buf.as_mut_ptr() as u64, 32, 32, 32, super::Format::Bgrx) };
    let img = Image { w: 2, h: 2, px: alloc::vec![0xFF00_FF00; 4] };
    draw(&fb, &img, Quad::flat(4.0, 6.0, 10.0, 8.0));
    let inside = (6..14).all(|y| (4..14).all(|x| buf[y * 32 + x] & 0x00FF_FFFF == 0x0000_FF00));
    let outside = buf[5 * 32 + 4] == 0 && buf[6 * 32 + 3] == 0 && buf[14 * 32 + 4] == 0 && buf[6 * 32 + 14] == 0;
    claim("a flat quad covers exactly its rectangle", inside && outside);
    ok
}

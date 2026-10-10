//! Scaling an image: bilinear sampling, and a mip chain for shrinking.
//!
//! What scales here is a window's image (phase 11 onwards): it pops in from
//! 87% of its size, it stands in a Flip 3D stack at a third of it, and it is a
//! thumbnail on the taskbar at a tenth. Hyprland samples with `GL_LINEAR`, and
//! so does this, in fixed point.
//!
//! **Interpolation happens on premultiplied pixels**, which is the only place
//! it is right. Blending a transparent pixel's colour into an opaque
//! neighbour's -- what interpolating *straight* alpha does -- pulls the edge
//! of every rounded window toward whatever colour its invisible corner
//! happened to hold, usually black, and the edge grows a dark fringe.
//! Premultiplied, an invisible pixel contributes nothing at all.
//!
//! **Bilinear alone is wrong below half size.** It reads four pixels around
//! each sample point, so at a tenth of the size it reads four of every
//! hundred and ignores the rest: a thumbnail of text shimmers as it moves and
//! drops thin lines entirely. The mip chain halves the image with a 2x2 box
//! until one more halving would go below the target, and bilinear does the
//! last step from there -- the arrangement every GPU uses, at a cost of a
//! third more memory per image.

use alloc::vec::Vec;

use super::Framebuffer;

/// A premultiplied ARGB image.
#[derive(Clone)]
pub struct Image {
    pub w: u32,
    pub h: u32,
    pub px: Vec<u32>,
}

impl Image {
    /// Half the size in each direction, each pixel the average of the 2x2
    /// block it covers (an odd last row or column averages what there is).
    pub fn half(&self) -> Image {
        let (w, h) = ((self.w / 2).max(1), (self.h / 2).max(1));
        let mut px = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let (x0, y0) = (x * 2, y * 2);
                let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
                let p = [
                    self.px[(y0 * self.w + x0) as usize],
                    self.px[(y0 * self.w + x1) as usize],
                    self.px[(y1 * self.w + x0) as usize],
                    self.px[(y1 * self.w + x1) as usize],
                ];
                let mut out = 0u32;
                for lane in 0..4 {
                    let s: u32 = p.iter().map(|&v| (v >> (8 * lane)) & 0xff).sum();
                    out |= ((s + 2) / 4) << (8 * lane);
                }
                px.push(out);
            }
        }
        Image { w, h, px }
    }
}

/// An image and its halvings, built on demand down to the size asked for.
pub struct Mips {
    levels: Vec<Image>,
}

impl Mips {
    pub fn new(base: Image) -> Mips {
        Mips { levels: alloc::vec![base] }
    }

    pub fn base(&self) -> &Image {
        &self.levels[0]
    }

    /// The smallest level still at least `w` x `h`, halving as far as needed.
    ///
    /// Searched, not taken from the end: levels are kept once built, so after
    /// a thumbnail has halved the chain down to a few dozen pixels the last
    /// level is the wrong answer for anything larger -- the first version
    /// returned it, and a pop-in at 39% was upscaled from a 120-pixel copy.
    pub fn level_for(&mut self, w: u32, h: u32) -> &Image {
        let (w, h) = (w.max(1), h.max(1));
        loop {
            let last = self.levels.last().unwrap();
            if last.w / 2 < w || last.h / 2 < h || last.w < 2 || last.h < 2 {
                break;
            }
            let next = last.half();
            self.levels.push(next);
        }
        let i = self.levels.iter().rposition(|l| l.w >= w && l.h >= h).unwrap_or(0);
        &self.levels[i]
    }
}

/// Lay `img` scaled into `(x, y, w, h)`, at `opacity` (0..=255), OVER what
/// is there. Bilinear; the caller hands in an image no more than twice the
/// target size (`Mips::level_for`) for anything smaller than half.
pub fn draw(fb: &Framebuffer, img: &Image, x: i32, y: i32, w: u32, h: u32, opacity: u8) {
    if w == 0 || h == 0 || img.w == 0 || img.h == 0 || opacity == 0 {
        return;
    }
    // Source position of a destination pixel's centre, in 16.16:
    // u = (dx + 0.5) * sw / dw - 0.5. Stepped by addition across a row.
    let step_x = ((img.w as u64) << 16) / w as u64;
    let step_y = ((img.h as u64) << 16) / h as u64;
    let start_x = (step_x as i64 / 2) - (1 << 15);
    let start_y = (step_y as i64 / 2) - (1 << 15);
    let mut row: Vec<u32> = alloc::vec![0; w as usize];
    let (sw, sh) = (img.w as i64, img.h as i64);
    for dy in 0..h {
        let sy = y + dy as i32;
        if sy < 0 || sy >= fb.height() as i32 {
            continue;
        }
        let v = start_y + dy as i64 * step_y as i64;
        let (v0, fy) = split(v, sh);
        let v1 = (v0 + 1).min(sh - 1);
        let r0 = &img.px[(v0 * sw) as usize..((v0 + 1) * sw) as usize];
        let r1 = &img.px[(v1 * sw) as usize..((v1 + 1) * sw) as usize];
        let mut u = start_x;
        for p in row.iter_mut() {
            let (u0, fx) = split(u, sw);
            let u1 = (u0 + 1).min(sw - 1);
            let top = lerp(r0[u0 as usize], r0[u1 as usize], fx);
            let bot = lerp(r1[u0 as usize], r1[u1 as usize], fx);
            *p = lerp(top, bot, fy);
            u += step_x as i64;
        }
        if opacity != 255 {
            for p in row.iter_mut() {
                *p = super::blend::scale_px(*p, opacity as u32);
            }
        }
        let skip = (-x).max(0) as usize;
        if skip < row.len() {
            fb.over_row((x + skip as i32) as u32, sy as u32, &row[skip..]);
        }
    }
}

/// A 16.16 position split into a clamped whole pixel and an 8-bit fraction.
#[inline]
fn split(v: i64, len: i64) -> (i64, u32) {
    if v <= 0 {
        return (0, 0);
    }
    let i = v >> 16;
    if i >= len - 1 {
        return (len - 1, 0);
    }
    (i, ((v >> 8) & 0xff) as u32)
}

/// Each byte lane of `a` toward `b` by `f`/256, exactly at the ends.
#[inline]
fn lerp(a: u32, b: u32, f: u32) -> u32 {
    if f == 0 {
        return a;
    }
    let mut out = 0u32;
    for lane in 0..4 {
        let (x, y) = ((a >> (8 * lane)) & 0xff, (b >> (8 * lane)) & 0xff);
        let v = (x * (256 - f) + y * f + 128) >> 8;
        out |= v << (8 * lane);
    }
    out
}

/// Draw `img` at scale `k` with its top-left at `(x, y)`, picking the mip
/// level for it: the whole of what a caller does with a window image.
pub fn draw_at(fb: &Framebuffer, mips: &mut Mips, x: i32, y: i32, k: f32, opacity: u8) {
    let base = mips.base();
    let (w, h) = (((base.w as f32 * k) + 0.5) as u32, ((base.h as f32 * k) + 0.5) as u32);
    let level = mips.level_for(w, h).clone();
    draw(fb, &level, x, y, w, h, opacity);
}

/// `video scale`: the wallpaper at five sizes, a window-ish image with
/// rounded transparent corners scaled up and down, and the same thumbnail
/// with and without mips.
pub fn proof(fb: &Framebuffer) {
    let (w, h) = (fb.width(), fb.height());
    super::grad::vertical(fb, 0, 0, w, h, &[(0.0, super::Color::new(30, 34, 44)), (1.0, super::Color::new(18, 20, 26))]);
    // A source: the wallpaper, read back from a surface it was painted into.
    let mut src: Vec<u32> = alloc::vec![0; (w * h) as usize];
    let surf = unsafe { Framebuffer::over_ram(src.as_mut_ptr() as u64, w, h, w, fb.format()) };
    super::desk::paint_wallpaper(&surf);
    // A rounded card with text on it over the middle, so thin detail exists.
    let card = super::shape::Box::new((w / 4) as i32, (h / 4) as i32, w / 2, h / 2);
    super::shape::rrect(&surf, card, 24, super::blend::opaque(surf.encode(super::Color::new(248, 248, 250))));
    for (i, line) in ["Thumbnails need a mip chain.", "Without one, thin text shimmers", "and hairlines vanish."].iter().enumerate() {
        super::text::draw(&surf, (w / 4 + 40) as f32, (h / 4 + 80 + i as u32 * 60) as i32, super::text::Face::Ui, 40.0, line, super::Color::new(20, 24, 32));
    }
    src.iter_mut().for_each(|p| *p |= 0xFF00_0000);
    let mut mips = Mips::new(Image { w, h, px: src });
    let mut x = 20i32;
    for k in [0.4f32, 0.25, 0.15, 0.08] {
        draw_at(fb, &mut mips, x, 20, k, 255);
        x += (w as f32 * k) as i32 + 20;
    }
    // The same 8% thumbnail without mips, beside it: the shimmer, frozen.
    let base = mips.base().clone();
    let (tw, th) = ((w as f32 * 0.08) as u32, (h as f32 * 0.08) as u32);
    draw(fb, &base, x, 20, tw, th, 255);
    // Pop-in: 87% at half opacity, as a window arrives.
    draw_at(fb, &mut mips, 20, (h as f32 * 0.45) as i32, 0.45 * 0.87, 128);
    // And an upscale of a small patch, which is where bilinear shows its blur.
    let patch_w = 60u32;
    let mut patch = Vec::with_capacity((patch_w * patch_w) as usize);
    for yy in 0..patch_w {
        for xx in 0..patch_w {
            // Over the first line of the card's text, where there is detail.
            patch.push(base.px[((h / 4 + 45 + yy) * w + w / 4 + 40 + xx) as usize]);
        }
    }
    let patch = Image { w: patch_w, h: patch_w, px: patch };
    draw(fb, &patch, (w as f32 * 0.5) as i32, (h as f32 * 0.45) as i32, patch_w * 6, patch_w * 6, 255);
}

/// `diag scale`.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    // A small premultiplied test image: a gradient with a transparent column.
    let (iw, ih) = (16u32, 12u32);
    let mut px = Vec::new();
    for y in 0..ih {
        for x in 0..iw {
            if x == 0 {
                px.push(0);
            } else {
                let v = (x * 13 + y * 7) & 0xff;
                px.push(0xFF00_0000 | v | v << 8 | (255 - v) << 16);
            }
        }
    }
    let img = Image { w: iw, h: ih, px };
    let (fw, fh) = (64u32, 48u32);
    let mut out: Vec<u32> = alloc::vec![0; (fw * fh) as usize];
    let fb = unsafe { Framebuffer::over_ram(out.as_mut_ptr() as u64, fw, fh, fw, super::Format::Bgrx) };

    draw(&fb, &img, 5, 3, iw, ih, 255);
    let ident = (0..ih).all(|y| (0..iw).all(|x| out[((y + 3) * fw + x + 5) as usize] == img.px[(y * iw + x) as usize]));
    claim("at one to one on whole pixels it is a copy, exactly", ident);
    let outside = out.iter().enumerate().all(|(i, &p)| {
        let (x, y) = (i as u32 % fw, i as u32 / fw);
        (x >= 5 && x < 5 + iw && y >= 3 && y < 3 + ih) || p == 0
    });
    claim("and writes nothing outside the rectangle it was given", outside);

    let flat = Image { w: 4, h: 4, px: alloc::vec![0xFF40_80C0; 16] };
    out.iter_mut().for_each(|p| *p = 0);
    draw(&fb, &flat, 0, 0, 37, 29, 255);
    let fl = (0..29).all(|y| (0..37).all(|x| out[(y * fw + x) as usize] == 0xFF40_80C0));
    claim("an upscaled flat image is flat, to the bit", fl);

    let premul_ok = {
        out.iter_mut().for_each(|p| *p = 0);
        draw(&fb, &img, 0, 0, 41, 33, 255);
        (0..33).all(|y| (0..41).all(|x| {
            let p = out[(y * fw + x) as usize];
            let a = p >> 24;
            (0..3).all(|l| (p >> (8 * l)) & 0xff <= a)
        }))
    };
    claim("interpolated pixels stay premultiplied: no channel exceeds alpha", premul_ok);

    // Mips: a 2x2 average is exact, and a mean survives the whole chain.
    let quad = Image { w: 2, h: 2, px: alloc::vec![0xFF00_0000, 0xFF00_0004, 0xFF00_0008, 0xFF00_000C] };
    claim("halving averages a 2x2 block, rounding to nearest", quad.half().px == alloc::vec![0xFF00_0006]);
    let mut big = Vec::new();
    for y in 0..64u32 {
        for x in 0..64u32 {
            big.push(0xFF00_0000 | (if (x + y) % 2 == 0 { 255 } else { 0 }));
        }
    }
    let mut mips = Mips::new(Image { w: 64, h: 64, px: big });
    let lvl = mips.level_for(5, 5).clone();
    let mean: u32 = lvl.px.iter().map(|&p| p & 0xff).sum::<u32>() / lvl.px.len() as u32;
    kprintln!("       a 64x64 checkerboard halved to {}x{}, mean {}", lvl.w, lvl.h, mean);
    claim("a checkerboard shrunk through the chain is its mean grey", (mean as i32 - 128).abs() <= 1 && lvl.w >= 5 && lvl.w < 10);
    out.iter_mut().for_each(|p| *p = 0);
    draw(&fb, &lvl, 0, 0, 5, 5, 255);
    let thumb = (0..5).all(|y| (0..5).all(|x| ((out[(y * fw + x) as usize] & 0xff) as i32 - 128).abs() <= 2));
    claim("and drawn at 5x5 shows grey, not a sampled pattern", thumb);

    let again = mips.level_for(30, 30).w;
    claim("a larger size asked for after a smaller one gets a larger level", again >= 30 && again < 64);

    out.iter_mut().for_each(|p| *p = 0xFF11_2233);
    draw(&fb, &img, 0, 0, 30, 20, 0);
    claim("opacity zero changes nothing", out.iter().all(|&p| p == 0xFF11_2233));
    ok
}

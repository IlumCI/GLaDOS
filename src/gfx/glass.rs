//! Vista glass: a blurred backdrop, colourised, with a gloss on top.
//!
//! The decision on record is **wallpaper-only blur** (Hyprland's `xray`):
//! glass shows the wallpaper behind it blurred, not the windows behind it.
//! That is what makes it affordable on a CPU -- the wallpaper changes when
//! somebody changes it, so it is blurred once into a second image, and a pane
//! of glass is then a per-pixel formula over two images, never a convolution
//! per frame. Glass over a window would need that window blurred whenever it
//! changed, which is the one cost this design is built to avoid.
//!
//! **The blur is three box blurs**, which converge on a Gaussian (the central
//! limit theorem doing the work), each a running sum and so O(1) a pixel
//! whatever the radius. The box widths come from the usual formula for
//! matching a Gaussian's variance with `n` boxes.
//!
//! **The colourisation is Vista's own**, as OpenGlass reconstructed it from the
//! DWM: `blurBalance * B + afterglowBalance * lum(B) * glow + colorBalance *
//! tint`, defaults 49, 43 and 8 per cent. The middle term is why Vista glass
//! reads as tinted light rather than as tinted paint: it carries the
//! backdrop's *brightness* in the glow colour, so a bright patch of sky
//! behind the glass brightens it in its own hue.
//!
//! **A fixed noise**, plus or minus three levels, keyed to screen position:
//! a large flat-ish glass surface otherwise bands exactly like an undithered
//! gradient, and a pattern that is a function of position redraws
//! bit-identically, so the scene diff does not see glass as changed.

use alloc::vec::Vec;

use super::{Color, Format, Framebuffer};
use crate::sync::Spin;

/// What a pane of glass looks like.
#[derive(Clone, Copy, Debug)]
pub struct Glass {
    /// The colourisation colour.
    pub tint: Color,
    /// The afterglow colour; Vista used the tint for it.
    pub glow: Color,
    /// Shares of the blurred backdrop, its luminance in the glow colour, and
    /// the flat tint, out of 100. Vista's defaults are 49, 43 and 8.
    pub blur: u32,
    pub afterglow: u32,
    pub color: u32,
    /// White added at the top, fading to nothing by `gloss_end` of the
    /// height, out of 255: the reflection that makes glass read as glass.
    pub gloss: u32,
    pub gloss_end: f32,
}

impl Glass {
    /// Vista's "Sky", the default colour scheme, `0x6b74b8fc`.
    pub const SKY: Glass = Glass {
        tint: Color::new(0x74, 0xb8, 0xfc),
        glow: Color::new(0x74, 0xb8, 0xfc),
        blur: 49,
        afterglow: 43,
        color: 8,
        gloss: 40,
        gloss_end: 0.5,
    };
}

/// The blurred backdrop, in the screen's own pixel format.
struct Backdrop {
    px: Vec<u32>,
    w: u32,
    h: u32,
}

static BACKDROP: Spin<Option<Backdrop>> = Spin::new(None);

/// Blur `px` (a `w` x `h` image) and keep it as the backdrop every pane of
/// glass samples. Called when the wallpaper is painted, which is once per
/// screen size. Answers the milliseconds it took.
pub fn set_backdrop(px: &[u32], w: u32, h: u32, sigma: f32) -> u64 {
    let t0 = crate::time::rdtsc();
    let blurred = blur(px, w, h, sigma);
    *BACKDROP.lock_irq() = Some(Backdrop { px: blurred, w, h });
    let mhz = crate::time::tsc_mhz().max(1) as u64;
    (crate::time::rdtsc() - t0) / mhz / 1000
}

/// Box widths for `n` boxes whose sum of variances is `sigma^2`.
pub(crate) fn box_sizes(sigma: f32, n: usize) -> Vec<usize> {
    let ideal = crate::ai::tensor::sqrtf(12.0 * sigma * sigma / n as f32 + 1.0);
    let mut wl = ideal as usize;
    if wl % 2 == 0 {
        wl = wl.saturating_sub(1);
    }
    let wl = wl.max(1);
    let wu = wl + 2;
    let m_ideal = (12.0 * sigma * sigma
        - (n * wl * wl) as f32
        - (4 * n * wl) as f32
        - (3 * n) as f32)
        / (-4.0 * wl as f32 - 4.0);
    let m = (m_ideal + 0.5).max(0.0) as usize;
    (0..n).map(|i| if i < m { wl } else { wu }).collect()
}

/// Three box blurs approximating a Gaussian of `sigma`, per byte lane, with
/// the edge pixel extended so a flat image stays exactly flat.
pub fn blur(px: &[u32], w: u32, h: u32, sigma: f32) -> Vec<u32> {
    let (w, h) = (w as usize, h as usize);
    let n = w * h;
    let mut planes: [Vec<u16>; 3] = core::array::from_fn(|c| px.iter().map(|&p| ((p >> (8 * c)) & 0xff) as u16).collect());
    let mut tmp = alloc::vec![0u16; n];
    if sigma > 0.25 {
        for size in box_sizes(sigma, 3) {
            let r = size / 2;
            for plane in planes.iter_mut() {
                box_pass(plane, &mut tmp, w, h, r, 1, w);
                box_pass(&tmp, plane, h, w, r, w, 1);
            }
        }
    }
    (0..n)
        .map(|i| 0xFF00_0000 | planes[0][i] as u32 | (planes[1][i] as u32) << 8 | (planes[2][i] as u32) << 16)
        .collect()
}

/// One box of radius `r` along every line of an image: `len` elements a line,
/// `step` between elements of a line, `lines` lines `stride` apart.
fn box_pass(src: &[u16], dst: &mut [u16], len: usize, lines: usize, r: usize, step: usize, stride: usize) {
    let size = (2 * r + 1) as u32;
    for line in 0..lines {
        let base = line * stride;
        let at = |i: isize| -> u32 { src[base + (i.clamp(0, len as isize - 1) as usize) * step] as u32 };
        let mut sum: u32 = (-(r as isize)..=r as isize).map(at).sum();
        for i in 0..len {
            dst[base + i * step] = ((sum + size / 2) / size) as u16;
            sum += at(i as isize + r as isize + 1);
            sum -= at(i as isize - r as isize);
        }
    }
}

/// A fixed per-position noise in -3..=3: a hash of the coordinates, so the
/// same pixel always gets the same value.
#[inline]
fn noise(x: u32, y: u32) -> i32 {
    let mut v = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77);
    v ^= v >> 15;
    v = v.wrapping_mul(0x2C1B_3C6D);
    v ^= v >> 12;
    (v % 7) as i32 - 3
}

/// Whether glass shows the blurred wallpaper (Settings; the quality
/// governor's first rung turns it off). Off, a pane is its tint over a flat
/// wall: the colourisation formula with nothing behind it.
static BLUR: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);

pub fn set_blur(on: bool) {
    BLUR.store(on, core::sync::atomic::Ordering::Relaxed);
}

/// Whether glass shows the blur now: the setting, unless the quality
/// governor has shed it.
pub fn blur_on() -> bool {
    blur_setting() && !super::quality::shed(1)
}

/// The setting, whatever the governor is doing.
pub fn blur_setting() -> bool {
    BLUR.load(core::sync::atomic::Ordering::Relaxed)
}

/// Paint a pane of glass over `(x, y, w, h)` with corners of `radius`.
///
/// Where there is no backdrop yet -- before the wallpaper has been painted, or
/// on a machine without the memory for it -- the tint stands in for the
/// blurred backdrop, which is the formula with a flat wall behind it.
pub fn fill(fb: &Framebuffer, x: i32, y: i32, w: u32, h: u32, radius: u32, g: Glass) {
    fill_with(fb, x, y, w, h, radius, g, blur_on())
}

/// `fill`, saying whether to use the blur rather than asking the setting
/// and the governor: the suite tests the formula, not the machine's mood.
fn fill_with(fb: &Framebuffer, x: i32, y: i32, w: u32, h: u32, radius: u32, g: Glass, blur: bool) {
    if w == 0 || h == 0 {
        return;
    }
    let bgr = fb.format() == Format::Bgrx;
    // Lane of each primary within a pixel word.
    let (ri, bi) = if bgr { (2, 0) } else { (0, 2) };
    let tint = [g.tint.r as i32, g.tint.g as i32, g.tint.b as i32];
    let glow = [g.glow.r as i32, g.glow.g as i32, g.glow.b as i32];
    let guard = BACKDROP.lock_irq();
    // Blur off (Settings, or the quality governor's first rung) is the
    // formula with a flat wall behind it, which is what no backdrop gives.
    let back = if blur { guard.as_ref() } else { None };
    let r = radius.min(w / 2).min(h / 2);
    // Only the part of the pane the surface will take. Every pixel is a
    // function of its own position alone -- the backdrop under it, its noise,
    // its corner -- so computing a clipped part gives exactly the pixels a
    // whole pane would have put there. It computed the whole pane and let
    // the write clip it, which made a clock tick on the taskbar recompute
    // all 1920 x 48 pixels of the bar's glass to change four digits.
    let (cx0, cy0, cx1, cy1) = fb.scissor();
    let rx0 = (cx0 as i64 - x as i64).clamp(0, w as i64) as u32;
    let rx1 = (cx1 as i64 - x as i64).clamp(0, w as i64) as u32;
    let ry0 = (cy0 as i64 - y as i64).clamp(0, h as i64) as u32;
    let ry1 = (cy1 as i64 - y as i64).clamp(0, h as i64) as u32;
    if rx0 >= rx1 || ry0 >= ry1 {
        return;
    }
    let cols = (rx1 - rx0) as usize;
    let mut row: Vec<u32> = alloc::vec![0; cols];
    let mut alpha: Vec<u8> = alloc::vec![0; cols];
    for ry in ry0..ry1 {
        let sy = y + ry as i32;
        if sy < 0 || sy >= fb.height() as i32 {
            continue;
        }
        // The gloss: white fading linearly to nothing at `gloss_end`.
        let t = ry as f32 / h as f32;
        let gloss = if t < g.gloss_end { (g.gloss as f32 * (1.0 - t / g.gloss_end)) as i32 } else { 0 };
        for rx in rx0..rx1 {
            let sx = x + rx as i32;
            let b = match back {
                Some(bd) if sx >= 0 && (sx as u32) < bd.w && (sy as u32) < bd.h => bd.px[(sy as u32 * bd.w + sx as u32) as usize],
                _ => {
                    let c = tint;
                    if bgr {
                        (c[2] | c[1] << 8 | c[0] << 16) as u32
                    } else {
                        (c[0] | c[1] << 8 | c[2] << 16) as u32
                    }
                }
            };
            let lane = |i: u32| ((b >> (8 * i)) & 0xff) as i32;
            let (br, bg, bb) = (lane(ri), lane(1), lane(bi));
            let lum = (br * 54 + bg * 183 + bb * 19) >> 8;
            let n = noise(sx as u32, sy as u32);
            let mix = |back: i32, k: usize| -> u32 {
                let v = (back * g.blur as i32 + lum * glow[k] / 255 * g.afterglow as i32 + tint[k] * g.color as i32) / 100;
                (v + gloss * (255 - v) / 255 + n).clamp(0, 255) as u32
            };
            let (or, og, ob) = (mix(br, 0), mix(bg, 1), mix(bb, 2));
            let i = (rx - rx0) as usize;
            row[i] = 0xFF00_0000 | if bgr { ob | og << 8 | or << 16 } else { or | og << 8 | ob << 16 };
            alpha[i] = corner_cov(rx, ry, w, h, r);
        }
        let left = x + rx0 as i32;
        let skip = (-left).max(0) as usize;
        if skip >= row.len() {
            continue;
        }
        let full = alpha[skip..].iter().all(|&a| a == 255);
        if full {
            fb.blit_span((left + skip as i32) as u32, sy as u32, &row[skip..]);
        } else {
            // Premultiply by coverage and lay over what is there: the corners.
            for (p, &a) in row.iter_mut().zip(alpha.iter()) {
                *p = super::blend::premul(*p & 0x00FF_FFFF, a);
            }
            fb.over_row((left + skip as i32) as u32, sy as u32, &row[skip..]);
        }
    }
}

/// Coverage of pixel `(rx, ry)` of a `w` x `h` box with corners of `r`.
#[inline]
fn corner_cov(rx: u32, ry: u32, w: u32, h: u32, r: u32) -> u8 {
    if r == 0 {
        return 255;
    }
    let near_x = rx < r || rx >= w - r;
    let near_y = ry < r || ry >= h - r;
    if !(near_x && near_y) {
        return 255;
    }
    let d = super::shape::sd_rbox(rx as f32 + 0.5, ry as f32 + 0.5, w as f32 / 2.0, h as f32 / 2.0,
                                  w as f32 / 2.0, h as f32 / 2.0, r as f32);
    super::shape::cov(d)
}

/// `video glass`: the desktop's own wallpaper with panes of glass on it.
pub fn proof(fb: &Framebuffer) {
    let (w, h) = (fb.width(), fb.height());
    // The real wallpaper, painted into this surface and blurred as the
    // desktop would.
    super::desk::paint_wallpaper(fb);
    let mut px: Vec<u32> = alloc::vec![0; (w * h) as usize];
    for yy in 0..h {
        for xx in 0..w {
            px[(yy * w + xx) as usize] = fb.get(xx, yy);
        }
    }
    let ms = set_backdrop(&px, w, h, 14.0);
    crate::kprintln!("  blurred the wallpaper in {} ms", ms);
    let k = (w as f32 / 960.0).min(h as f32 / 540.0);
    let p = |v: f32| (v * k) as i32;
    let orange = Glass { tint: Color::new(255, 140, 30), glow: Color::new(255, 170, 80), ..Glass::SKY };
    let graphite = Glass { tint: Color::new(60, 70, 84), glow: Color::new(120, 130, 150), ..Glass::SKY };
    // Three captions, one per tint, over the bright and dark parts of the wall.
    for (i, g) in [Glass::SKY, orange, graphite].iter().enumerate() {
        let x = p(60.0 + i as f32 * 300.0);
        fill(fb, x, p(60.0), p(260.0) as u32, p(30.0) as u32, (8.0 * k) as u32, *g);
        fill(fb, x, p(330.0), p(260.0) as u32, p(30.0) as u32, (8.0 * k) as u32, *g);
    }
    // A taskbar across the bottom and a Start menu above it.
    fill(fb, 0, h as i32 - p(36.0), w, p(36.0) as u32, 0, graphite);
    fill(fb, p(4.0), h as i32 - p(36.0) - p(300.0) - p(6.0), p(220.0) as u32, p(300.0) as u32, (8.0 * k) as u32, Glass::SKY);
}

/// `diag glass`.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    let (w, h) = (64u32, 48u32);
    let flat: Vec<u32> = alloc::vec![0xFF30_6090; (w * h) as usize];
    claim("blurring a flat image changes nothing", blur(&flat, w, h, 6.0) == flat);

    // An impulse: symmetric, monotone, and with the variance asked for.
    let (iw, ih) = (101u32, 101u32);
    let mut imp: Vec<u32> = alloc::vec![0xFF00_0000; (iw * ih) as usize];
    imp[(50 * iw + 50) as usize] = 0xFFFF_FFFF;
    // An impulse of 255 rounds to nothing after a few passes, so the measure
    // is taken on a block instead: a 21-pixel square of white.
    for yy in 40..61 {
        for xx in 40..61 {
            imp[(yy * iw + xx) as usize] = 0xFFFF_FFFF;
        }
    }
    let sigma = 5.0;
    let b = blur(&imp, iw, ih, sigma);
    let v = |x: u32, y: u32| (b[(y * iw + x) as usize] & 0xff) as i32;
    let mut sym = true;
    for yy in 0..ih {
        for xx in 0..iw {
            sym &= (v(xx, yy) - v(iw - 1 - xx, yy)).abs() <= 1 && (v(xx, yy) - v(xx, ih - 1 - yy)).abs() <= 1;
        }
    }
    claim("a blurred square is symmetric", sym);
    let before: i64 = imp.iter().map(|&p| (p & 0xff) as i64).sum();
    let after: i64 = b.iter().map(|&p| (p & 0xff) as i64).sum();
    kprintln!("       mass {} before and {} after", before, after);
    claim("and keeps its mass to a tenth of a per cent", (after - before).abs() * 1000 <= before);
    let mut mono = true;
    for xx in 51..iw {
        mono &= v(xx, 50) <= v(xx - 1, 50);
    }
    claim("falling away from its middle monotonically", mono);
    // The edge of the square is where a Gaussian blur leaves half intensity.
    let edge = (v(39, 50) + v(40, 50)) / 2;
    claim("with half intensity on the square's own edge", (edge - 128).abs() <= 6);
    let s = box_sizes(sigma, 3);
    let var: f32 = s.iter().map(|&b| ((b * b - 1) as f32) / 12.0).sum();
    kprintln!("       boxes {:?}, variance {:.1} for sigma^2 {:.1}", s, var, sigma * sigma);
    claim("three boxes sum to the Gaussian's variance within ten per cent", (var - sigma * sigma).abs() / (sigma * sigma) < 0.1);

    // Glass over a known backdrop is the formula, noise bounded.
    let saved = BACKDROP.lock_irq().take();
    let back: Vec<u32> = alloc::vec![0xFF80_8080; (w * h) as usize];
    *BACKDROP.lock_irq() = Some(Backdrop { px: back, w, h });
    let mut px: Vec<u32> = alloc::vec![0; (w * h) as usize];
    let fb = unsafe { Framebuffer::over_ram(px.as_mut_ptr() as u64, w, h, w, Format::Bgrx) };
    let g = Glass { gloss: 0, ..Glass::SKY };
    fill_with(&fb, 0, 0, w, h, 0, g, true);
    // 128 grey: lum 128; red channel = (128*49 + 128*116/255*43 + 116*8)/100.
    let want_r = (128 * 49 + 128 * 0x74 / 255 * 43 + 0x74 * 8) / 100;
    let worst = px.iter().map(|&p| (((p >> 16) & 0xff) as i32 - want_r).abs()).max().unwrap_or(99);
    kprintln!("       red should be {} before noise; worst distance {}", want_r, worst);
    claim("glass over a flat grey is Vista's formula, within the noise", worst <= 3);
    let first = px.clone();
    fill_with(&fb, 0, 0, w, h, 0, g, true);
    claim("and draws identically twice, so the scene diff sees no change", px == first);
    // Rounded corners touch nothing outside the shape.
    px.iter_mut().for_each(|p| *p = 0xFF00_0000);
    fill_with(&fb, 0, 0, w, h, 12, g, true);
    claim("a rounded pane leaves its corner pixel alone", px[0] == 0xFF00_0000 && px[(w - 1) as usize] == 0xFF00_0000);
    claim("and fills its middle", px[((h / 2) * w + w / 2) as usize] != 0xFF00_0000);
    *BACKDROP.lock_irq() = None;
    px.iter_mut().for_each(|p| *p = 0);
    fill_with(&fb, 0, 0, w, h, 0, g, true);
    claim("with no backdrop the tint stands in, and nothing faults", px[0] != 0);
    *BACKDROP.lock_irq() = saved;
    ok
}

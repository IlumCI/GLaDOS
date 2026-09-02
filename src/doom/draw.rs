//! Putting things on the screen, before there is a renderer.
//!
//! Two pictures, and neither is a renderer. They exist because the step after
//! this one is a BSP walk with perspective-correct texture mapping in it, and
//! debugging that on top of an untested palette path means two unknowns at
//! once. Everything here is checkable by eye:
//!
//!   * `palette` draws all 256 colours of the WAD's own `PLAYPAL` as a grid.
//!     If the byte order is wrong the ramps run the wrong way; if the encode
//!     into the screen's word order is wrong the reds are blue; if the row
//!     expansion is wrong the grid is sheared. Each of those is a distinct and
//!     obvious picture.
//!   * `overhead` draws the level's linedefs from above. That is the first
//!     thing to use the level reader's output rather than report it, and a
//!     square room that comes out square is a vertex decode, a linedef decode
//!     and a coordinate transform all confirmed at once.
//!
//! Both draw into a `port::Surface` -- an indexed frame with the game's own
//! palette -- which is exactly what the renderer will draw into. So the path
//! being tested is the path that will be used, rather than a convenient
//! substitute for it.

use super::level::Level;
use crate::port::Surface;

/// The index in `pal` closest to a wanted colour.
///
/// A WAD's palette is whatever the artist chose, so nothing here can name an
/// index and expect it to be the colour it wants -- index 4 is white in
/// DOOM's PLAYPAL and something else entirely in a total conversion's. The
/// search is 256 entries of integer arithmetic, run a handful of times when a
/// picture is set up rather than per pixel.
///
/// Squared distance in RGB, which is not how a person perceives colour and is
/// entirely sufficient for picking a red that reads as red.
pub fn nearest(pal: &[u8], r: u8, g: u8, b: u8) -> u8 {
    let mut best = 0u8;
    let mut best_d = u32::MAX;
    for i in 0..256usize {
        let j = i * 3;
        let (pr, pg, pb) = match (pal.get(j), pal.get(j + 1), pal.get(j + 2)) {
            (Some(a), Some(b2), Some(c)) => (*a as i32, *b2 as i32, *c as i32),
            _ => break,
        };
        let (dr, dg, db) = (pr - r as i32, pg - g as i32, pb - b as i32);
        let d = (dr * dr + dg * dg + db * db) as u32;
        if d < best_d {
            best_d = d;
            best = i as u8;
        }
    }
    best
}

/// How many 256-colour palettes a `PLAYPAL` holds.
///
/// DOOM ships fourteen: the normal one, then the reds it fades through when
/// you are hit and the golds for picking an item up. Everything here uses the
/// first, which is the one the world is actually drawn in.
pub fn palette_count(playpal: &[u8]) -> usize {
    playpal.len() / 768
}

/// All 256 colours as a sixteen-by-sixteen grid.
pub fn palette(surf: &mut Surface, playpal: &[u8]) {
    surf.set_palette_rgb(playpal);
    let (w, h) = (surf.width(), surf.height());
    let px = surf.pixels();
    for y in 0..h {
        let cy = y * 16 / h;
        for x in 0..w {
            let cx = x * 16 / w;
            px[y * w + x] = (cy * 16 + cx) as u8;
        }
    }
}

/// One pixel, if it is on the frame.
///
/// Bounds-checked rather than clipped by the caller, because the caller is a
/// line drawer whose endpoints have already been scaled from map coordinates
/// -- and a map coordinate is an `i16` from a file, so an off-by-one at the
/// edge is a wild write rather than a wrong pixel.
fn put(px: &mut [u8], w: usize, h: usize, x: i32, y: i32, c: u8) {
    if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
        px[y as usize * w + x as usize] = c;
    }
}

/// Bresenham, integer only.
fn line(px: &mut [u8], w: usize, h: usize, mut x0: i32, mut y0: i32, x1: i32, y1: i32, c: u8) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        put(px, w, h, x0, y0, c);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// The level seen from above: every linedef, and the player start.
///
/// Fitted to the frame with one scale for both axes, so a square room is drawn
/// square. Fitting each axis separately would fill the frame better and would
/// make a stretched picture indistinguishable from a level that really is
/// stretched -- which is the exact class of bug this picture is here to catch.
pub fn overhead(surf: &mut Surface, lv: &Level, playpal: &[u8]) {
    surf.set_palette_rgb(playpal);

    let ink_wall = nearest(playpal, 0xFF, 0xFF, 0xFF);
    let ink_two = nearest(playpal, 0xF2, 0x8C, 0x1E);
    let ink_start = nearest(playpal, 0x40, 0xFF, 0x40);
    let ink_bg = nearest(playpal, 0, 0, 0);

    surf.clear(ink_bg);

    if lv.vertexes.is_empty() {
        return;
    }
    let (mut minx, mut maxx) = (i32::MAX, i32::MIN);
    let (mut miny, mut maxy) = (i32::MAX, i32::MIN);
    for v in lv.vertexes.iter() {
        minx = minx.min(v.x as i32);
        maxx = maxx.max(v.x as i32);
        miny = miny.min(v.y as i32);
        maxy = maxy.max(v.y as i32);
    }
    let (w, h) = (surf.width() as i32, surf.height() as i32);
    let span_x = (maxx - minx).max(1);
    let span_y = (maxy - miny).max(1);
    // A margin, so a wall on the boundary is visible rather than clipped to
    // the frame's own edge and indistinguishable from it.
    let inset = 8;
    let sx = (w - inset * 2) * 256 / span_x;
    let sy = (h - inset * 2) * 256 / span_y;
    let s = sx.min(sy).max(1);
    let ox = (w - span_x * s / 256) / 2;
    let oy = (h - span_y * s / 256) / 2;

    // Map y grows north and screen y grows down, so the vertical axis is
    // flipped. Without this the level is drawn mirrored, which on a
    // symmetrical test room looks perfectly correct -- and would be found much
    // later, on a real map, as "the level is back to front".
    let to_screen = |vx: i32, vy: i32| -> (i32, i32) {
        (
            ox + (vx - minx) * s / 256,
            oy + (span_y - (vy - miny)) * s / 256,
        )
    };

    let (sw, sh) = (surf.width(), surf.height());
    let lines: alloc::vec::Vec<(i32, i32, i32, i32, u8)> = lv
        .linedefs
        .iter()
        .filter_map(|l| {
            let a = lv.vertexes.get(l.v1 as usize)?;
            let b = lv.vertexes.get(l.v2 as usize)?;
            let (x0, y0) = to_screen(a.x as i32, a.y as i32);
            let (x1, y1) = to_screen(b.x as i32, b.y as i32);
            let ink = if l.two_sided() { ink_two } else { ink_wall };
            Some((x0, y0, x1, y1, ink))
        })
        .collect();

    let start = lv.player_start().map(|t| to_screen(t.x as i32, t.y as i32));

    let px = surf.pixels();
    for (x0, y0, x1, y1, ink) in lines {
        line(px, sw, sh, x0, y0, x1, y1, ink);
    }
    if let Some((x, y)) = start {
        // A cross rather than a dot: one pixel on a 320-wide frame scaled up
        // is findable, and four are unmistakable.
        for d in -3..=3 {
            put(px, sw, sh, x + d, y, ink_start);
            put(px, sw, sh, x, y + d, ink_start);
        }
    }
}

/// One composed texture, as large as it will go and centred.
///
/// The proof-by-eye for the picture decoder, in the same spirit as `palette`
/// is for the blit path. A composed texture is the end of a long chain --
/// PNAMES, the TEXTURE1 directory, a patch's column offsets, the posts inside
/// a column and the tall-patch delta rule -- and every link in it fails as a
/// *plausible* picture rather than as an error. Drawing one at a size a person
/// can look at is the only check that settles them: the generated test WAD
/// puts a bright marker in a patch's top-left corner, offsets its vertical
/// joins course by course and cuts a hole in it, so a flip, a transposition
/// and a decoder that ignores posts each read as a different wrong picture.
///
/// Answers the zoom it used, or `None` if there is no such texture.
pub fn texture(
    surf: &mut Surface,
    pics: &super::pic::Pics,
    tex: usize,
    playpal: &[u8],
) -> Option<usize> {
    let def = pics.def(tex)?;
    let (tw, th) = (def.width, def.height);
    if tw == 0 || th == 0 {
        return None;
    }
    surf.set_palette_rgb(playpal);
    // A background that is neither the transparent index nor anything a
    // pattern uses, so a texture's own edges are visible and a hole in it is
    // told apart from the surround.
    let back = nearest(playpal, 0x30, 0x00, 0x30);
    surf.clear(back);

    let (w, h) = (surf.width(), surf.height());
    let zoom = (w / tw).min(h / th).max(1);
    let (dw, dh) = (tw * zoom, th * zoom);
    let (ox, oy) = ((w.saturating_sub(dw)) / 2, (h.saturating_sub(dh)) / 2);

    for sx in 0..dw.min(w) {
        let Some(col) = pics.column(tex, (sx / zoom) as i32) else { continue };
        for sy in 0..dh.min(h) {
            let ty = sy / zoom;
            let Some(v) = col.get(ty) else { continue };
            let (px_, py_) = (ox + sx, oy + sy);
            if px_ < w && py_ < h {
                let i = py_ * w + px_;
                surf.pixels()[i] = *v;
            }
        }
    }
    Some(zoom)
}

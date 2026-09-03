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

/// One flat, as large as it will go and centred.
///
/// The eye-check that a wall texture's cannot stand in for. A flat is
/// addressed by *world* position rather than by distance along a surface, so
/// it has two failure modes a wall does not: a transposed pair of coordinates,
/// and a mirrored axis -- DOOM negates world y to get the row, because north
/// is up in the world and down in the picture. Both are invisible on anything
/// symmetric, which is why `tools/mkwad.py` generates flats that are not.
///
/// Answers the zoom it used.
pub fn flat(surf: &mut Surface, pixels: &[u8], playpal: &[u8]) -> Option<usize> {
    let side = super::pic::FLAT_SIDE;
    if pixels.len() < side * side {
        return None;
    }
    surf.set_palette_rgb(playpal);
    let back = nearest(playpal, 0x00, 0x30, 0x30);
    surf.clear(back);

    let (w, h) = (surf.width(), surf.height());
    let zoom = (w / side).min(h / side).max(1);
    let (dw, dh) = (side * zoom, side * zoom);
    let (ox, oy) = ((w.saturating_sub(dw)) / 2, (h.saturating_sub(dh)) / 2);
    for sy in 0..dh.min(h) {
        for sx in 0..dw.min(w) {
            let v = pixels[(sy / zoom) * side + (sx / zoom)];
            let (px_, py_) = (ox + sx, oy + sy);
            if px_ < w && py_ < h {
                let i = py_ * w + px_;
                surf.pixels()[i] = v;
            }
        }
    }
    Some(zoom)
}

/// One sprite, blown up, over a background that is obviously not part of it.
///
/// The check `texture` and `flat` cannot make: a sprite is mostly *hole*, so a
/// decoder that filled the bounding box draws a rectangle where a barrel
/// should be, and one that lost the two pad bytes shears each post by a row.
/// Both are invisible on a wall texture, which is opaque everywhere.
///
/// The origin is marked, because `left` and `top` are the two fields a wall
/// never reads and the two a billboard is placed by.
pub fn sprite(surf: &mut Surface, p: &super::pic::Patch, playpal: &[u8]) {
    surf.set_palette_rgb(playpal);
    let back = nearest(playpal, 0x28, 0x00, 0x00);
    surf.clear(back);
    let (w, h) = (surf.width(), surf.height());
    if p.width == 0 || p.height == 0 {
        return;
    }
    let zoom = (w / p.width).min(h / p.height).max(1);
    let (dw, dh) = (p.width * zoom, p.height * zoom);
    let (ox, oy) = ((w.saturating_sub(dw)) / 2, (h.saturating_sub(dh)) / 2);

    for (cx, posts) in p.columns.iter().enumerate() {
        for post in posts.iter() {
            for (i, v) in post.pixels.iter().enumerate() {
                for zy in 0..zoom {
                    for zx in 0..zoom {
                        let px_ = ox + cx * zoom + zx;
                        let py_ = oy + (post.top + i) * zoom + zy;
                        if px_ < w && py_ < h {
                            let k = py_ * w + px_;
                            surf.pixels()[k] = *v;
                        }
                    }
                }
            }
        }
    }

    // The origin: where the thing's own position lands in the picture. A
    // cross rather than a dot, so it is legible over whatever is behind it.
    let mark = nearest(playpal, 0xFF, 0xFF, 0x00);
    let (gx, gy) = (ox + p.left.max(0) as usize * zoom, oy + p.top.max(0) as usize * zoom);
    for d in 0..6usize {
        for (a, b) in [(gx + d, gy), (gx.saturating_sub(d), gy), (gx, gy + d), (gx, gy.saturating_sub(d))] {
            if a < w && b < h {
                let k = b * w + a;
                surf.pixels()[k] = mark;
            }
        }
    }
}

/// All eight facings of one sprite, four across and two down.
///
/// The check the counts cannot make. A census says every thing found a
/// picture; it cannot say the picture is the *right* one, and a single frame
/// of a level shows each monster from exactly one bearing -- so a facing
/// picked from the wrong bucket, or a mirror drawn unmirrored, renders
/// perfectly and is wrong. Laid out in order, a correct set reads as one
/// figure turning through a full circle, and the second half mirrors the
/// first wherever the WAD stored one lump for two facings.
pub fn mugshot(
    surf: &mut Surface,
    frame: &super::sprite::Frame,
    playpal: &[u8],
) -> usize {
    surf.set_palette_rgb(playpal);
    let back = nearest(playpal, 0x20, 0x20, 0x28);
    surf.clear(back);
    let rule = nearest(playpal, 0x60, 0x60, 0x70);

    let (w, h) = (surf.width(), surf.height());
    let (cw, ch) = (w / 4, h / 2);
    let mut drawn = 0;
    for rot in 0..8usize {
        let (cx, cy) = ((rot % 4) * cw, (rot / 4) * ch);
        // A rule between the cells, so a sprite wider than its cell is
        // obviously overflowing rather than mysteriously clipped.
        for y in cy..(cy + ch).min(h) {
            if cx > 0 && cx < w {
                surf.pixels()[y * w + cx] = rule;
            }
        }
        let Some((patch, flip)) = frame.at(rot) else { continue };
        drawn += 1;
        // Feet on the cell's floor and centred on its middle, which is where
        // `left` and `top` put a sprite in the world too.
        let ox = cx as i32 + cw as i32 / 2 - patch.left as i32;
        let oy = cy as i32 + ch as i32 - 8 - patch.top as i32;
        for sx in 0..patch.width {
            let src = if flip { patch.width - 1 - sx } else { sx };
            let Some(posts) = patch.columns.get(src) else { continue };
            let px_ = ox + sx as i32;
            if px_ < 0 || px_ >= w as i32 {
                continue;
            }
            for post in posts.iter() {
                for (i, v) in post.pixels.iter().enumerate() {
                    let py_ = oy + (post.top + i) as i32;
                    if py_ < 0 || py_ >= h as i32 {
                        continue;
                    }
                    surf.pixels()[py_ as usize * w + px_ as usize] = *v;
                }
            }
        }
    }
    drawn
}

/// The weapon in your hands, over everything else.
///
/// DOOM's `R_DrawPSprite`, and on a 320 by 200 view it collapses to almost
/// nothing: the screen centre it adds and the 160 it subtracts cancel, so what
/// is left is the patch's own offsets against `sx` and `sy`. That is why a
/// pistol carries a left offset of -125 -- the number is not a nudge, it is
/// the whole of the placement, and a reader that ignored it would draw the gun
/// in a corner.
///
/// Integer-scaled to whatever the surface is, because DOOM's constants are for
/// 320 by 200 and the surface is not obliged to be. Repeated rather than
/// sampled, the choice the sprite path makes and for a reason that matters
/// more here: a gun fills a quarter of the screen, so a gap in it is a hole
/// you look through.
pub fn weapon(surf: &mut Surface, p: &super::pic::Patch, sx: f32, sy: f32) {
    let (w, h) = (surf.width(), surf.height());
    let zoom = (w / 320).max(1);
    let x0 = (sx - p.left as f32) as i32 * zoom as i32;
    let y0 = (sy - p.top as f32) as i32 * zoom as i32;
    for (cx, posts) in p.columns.iter().enumerate() {
        for post in posts.iter() {
            for (i, v) in post.pixels.iter().enumerate() {
                for zy in 0..zoom {
                    for zx in 0..zoom {
                        let px = x0 + (cx * zoom + zx) as i32;
                        let py = y0 + ((post.top + i) * zoom + zy) as i32;
                        if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                            let k = py as usize * w + px as usize;
                            surf.pixels()[k] = *v;
                        }
                    }
                }
            }
        }
    }
}

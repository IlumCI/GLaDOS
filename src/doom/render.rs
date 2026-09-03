//! Walls, drawn the way DOOM draws them: front to back, one column each.
//!
//! Adapted from room4doom (MIT), which is in turn a transliteration of the
//! original. The algorithm is not ours and is worth stating plainly because
//! every part of it is load-bearing:
//!
//! 1. **Walk the BSP from the viewer outwards.** At each node the viewer is on
//!    one side; that side is nearer, so it is visited first. The result is
//!    that every seg arrives in front-to-back order without sorting anything.
//! 2. **Keep a per-column window of what is still open.** `top[x]` and
//!    `bot[x]` bound the rows of column `x` nothing has claimed. A solid wall
//!    closes its columns outright; a portal narrows them by the step above and
//!    below the opening.
//! 3. **Draw each column once.** Because the traversal is front-to-back, the
//!    first thing to claim a row is the nearest thing, so there is no depth
//!    buffer and no overdraw -- which is how this ran on a 486.
//!
//! **Textures arrived after the geometry did, and the order was the point.**
//! The first version shaded each wall by distance and light and drew no art at
//! all, so a wrong picture was a geometry bug and could not be a texture bug.
//! Both halves work now, and the flat path is still here for a wall whose
//! texture is missing -- which is a legal thing for a WAD to contain and reads
//! better as a plain surface than as a hole.
//!
//! ### Where this departs from the original
//!
//! DOOM projects through angles: `R_PointToAngle` and a `viewangletox` table
//! turn a world point into a column directly, which on a 486 avoids a divide
//! per vertex. This transforms into view space and divides, because the divide
//! is now free and the angle tables are a source of subtle error that is hard
//! to see and harder to test. The geometry is identical; the arithmetic is
//! not.
//!
//! Everything about *lighting* is the original's, though, and had to be. A
//! texture pixel is a palette index, not a colour, so it cannot be darkened by
//! multiplying it -- an index scaled by 0.7 is an unrelated colour. The only
//! way to shade indexed art is to remap it through a table that says which
//! darker index each one becomes, and that table is COLORMAP.

use super::level::{Level, NONE, SUBSECTOR_BIT};
use super::math;
use super::pic::{Art, Pics, FLAT_SIDE, LIGHT_LEVELS};
use crate::port::Surface;

/// Where the eye is and which way it looks.
#[derive(Clone, Copy)]
pub struct View {
    pub x: f32,
    pub y: f32,
    /// Eye height in world units, absolute rather than above the floor.
    pub z: f32,
    /// Radians, 0 east and increasing anticlockwise, matching the map.
    pub angle: f32,
}

/// How far in front of the eye a vertex has to be to be projected.
///
/// Not zero: a vertex exactly on the eye plane divides by zero, and one just
/// in front of it projects to a column thousands of pixels off screen. Clipping
/// to a near plane is the standard answer and the constant only has to be
/// small enough not to clip anything a player can stand next to. A DOOM map
/// unit is roughly an inch.
const NEAR: f32 = 1.0;

/// The player's eye above the floor, which DOOM fixes at 41 units.
pub const EYE_HEIGHT: f32 = 41.0;

/// How many shades a wall is quantised into.
const SHADES: usize = 24;

/// Distance at which a wall reaches the darkest shade.
const FALLOFF: f32 = 1400.0;

/// A linedef whose upper texture hangs from the front ceiling rather than
/// rising from the back one.
const DONTPEG_TOP: u16 = 0x0008;
/// A linedef whose lower and middle textures are anchored at the bottom.
const DONTPEG_BOTTOM: u16 = 0x0010;

/// A column that no plane covers.
///
/// `i32::MAX` rather than a sentinel row, so that the span walk's `t <= b`
/// tests fail naturally on an empty column and the algorithm needs no special
/// case for one. DOOM uses `0xff` in a byte array for exactly this, and its
/// comparisons work the same way.
const UNSET: i32 = i32::MAX;

/// How many planes one frame may accumulate.
///
/// DOOM's `MAXVISPLANES` is 128 and it aborts with "no more visplanes" past
/// it, which was a reasonable thing for 1993 and is not one here. Running out
/// leaves a plane undrawn -- the same gap this whole change exists to close,
/// but bounded and rare rather than structural.
const MAX_PLANES: usize = 256;

/// One flat surface at one height, accumulated over the columns that show it.
///
/// **This is the structure the per-column shortcut could not have.** Drawing a
/// floor at the moment a wall claims a column only ever fills the part of that
/// column the wall left over, so a subsector no seg claims a column in gets
/// nothing. The fix is not a patch to that code: the information needed is
/// *which columns show this floor*, and no column can know it until the walk
/// is finished. So the walk records, and the drawing happens afterwards.
///
/// **How much that was costing is not established, and a claim that it was
/// was withdrawn.** A dark region in a FreeDoom frame was read as a hole and
/// asserted as one without being measured; it was distant geometry shaded
/// almost to black, and the count of pixels left at the sky colour is
/// *identical* before and after this change on both maps. So what is fixed
/// here is a defect the code is structurally able to have, not one anybody has
/// a picture of. The differential is the evidence that matters: the test map
/// renders **pixel-identical** through the new path, and FreeDoom's E1M1
/// differs in 0.8% of pixels, all of them small shifts from stepping the
/// texture coordinate along a span rather than recomputing it per pixel.
///
/// What it buys beyond correctness is the reason DOOM did it this way at all.
/// Distance is constant along a screen row of a level plane, so a whole
/// horizontal span costs one divide instead of one per pixel.
struct Plane {
    height: f32,
    /// The picture. Compared by address, which is what DOOM's `picnum`
    /// comparison amounts to: `Flats::by_name` hands back the same slice for
    /// the same name, so two planes wearing one flat share a pointer.
    flat: &'static [u8],
    light: i16,
    minx: i32,
    maxx: i32,
    /// The rows of each column this plane covers. `top[x] == UNSET` means it
    /// covers none of that column.
    top: alloc::vec::Vec<i32>,
    bot: alloc::vec::Vec<i32>,
}

pub struct Renderer {
    w: usize,
    h: usize,
    /// First row of each column not yet claimed, from the top.
    top: alloc::vec::Vec<i32>,
    /// Last row not yet claimed, from the bottom.
    bot: alloc::vec::Vec<i32>,
    /// Wall shades, bright to dark, and the same for the step faces so an
    /// upper reads as part of the same world rather than as a decal. Used
    /// only where a wall has no texture to draw.
    wall: [u8; SHADES],
    step: [u8; SHADES],
    /// `light[s][i]` is palette index `i` seen at shade `s`, `s` rising into
    /// the dark. DOOM's COLORMAP, and the only way indexed art can be shaded
    /// at all -- see the module note.
    light: alloc::vec::Vec<[u8; 256]>,
    /// Whether that table came out of the WAD or had to be built from the
    /// palette. Recorded rather than printed: nothing in this tree may name
    /// the printing macro, so the caller does the talking.
    pub lit_from_wad: bool,
    /// How many columns are still open, so a full screen can stop the walk.
    open: usize,
    /// Distance at which each column was closed by a solid surface, or
    /// infinity where nothing closed it.
    ///
    /// Sprites are drawn after the walk and need to know what is in front of
    /// them, and the per-column window cannot say: `top`/`bot` record *where*
    /// a column is still open and never *how far away* the thing that narrowed
    /// it was. One distance per column is the cheapest record that answers the
    /// question at all.
    solid: alloc::vec::Vec<f32>,
    /// The window each column was left with, saved before `close` wipes it.
    ///
    /// A doorway narrows a column without closing it, and a sprite beyond the
    /// doorway must not be drawn over its frame. Keeping the final opening
    /// alongside the distance that produced it covers that: past `near`, a
    /// sprite is only visible inside the opening.
    win_top: alloc::vec::Vec<i32>,
    win_bot: alloc::vec::Vec<i32>,
    /// Distance of the nearest surface that narrowed each column at all.
    near: alloc::vec::Vec<f32>,
    /// The planes this frame is accumulating.
    ///
    /// Pooled rather than reallocated. Each carries two arrays a column wide,
    /// and a busy frame wants a hundred of them, so allocating per frame would
    /// be several hundred allocations at 35 Hz to hold data whose shape never
    /// changes. `live` says how many are in use; the rest keep their buffers.
    planes: alloc::vec::Vec<Plane>,
    live: usize,
    /// The ceiling and floor plane of the subsector being walked.
    ///
    /// Mirrors DOOM's `ceilingplane`/`floorplane` globals, and for its reason
    /// rather than by imitation: `check_plane` can *replace* the plane a seg
    /// should mark into, and the replacement has to be visible to the next seg
    /// of the same subsector. Threading it through `seg`'s argument list would
    /// be two more `&mut` parameters on a function that already has nine.
    cur_ceil: Option<usize>,
    cur_floor: Option<usize>,
    /// Where the run currently open in each row began.
    spanstart: alloc::vec::Vec<i32>,
}

impl Renderer {
    pub fn new(surf: &Surface, art: &Art<'_>) -> Renderer {
        let (w, h) = (surf.width(), surf.height());
        let playpal = art.playpal;
        let mut wall = [0u8; SHADES];
        let mut step = [0u8; SHADES];
        for i in 0..SHADES {
            // A ramp built by asking the palette for the nearest match at each
            // brightness, rather than by walking indices. A WAD's palette is
            // whatever its artist chose, and consecutive indices in it are not
            // a ramp of anything in particular.
            let k = 255 - (i * 235 / (SHADES - 1)) as u32;
            wall[i] = super::draw::nearest(playpal, k as u8, k as u8, k as u8);
            let warm = (k * 3 / 4) as u8;
            step[i] = super::draw::nearest(playpal, k as u8, warm, (k / 2) as u8);
        }

        let mut light = alloc::vec::Vec::new();
        let from_wad = art.lighting_colormap();
        match from_wad {
            // The artist's own table, sampled down to however many shades this
            // renderer keeps. Map 0 is full brightness and map 31 is nearly
            // black, which is the order this ramp runs in too.
            Some(cm) => {
                for s in 0..SHADES {
                    let m = s * (LIGHT_LEVELS - 1) / (SHADES - 1);
                    let mut row = [0u8; 256];
                    row.copy_from_slice(&cm[m * 256..m * 256 + 256]);
                    light.push(row);
                }
            }
            // No table, or one that does not darken. Build one by asking the
            // palette which index is closest to each colour dimmed. 24 * 256
            // nearest-matches is a million and a half comparisons once at
            // startup and an array index every time afterwards.
            None => {
                for s in 0..SHADES {
                    let f = 1.0 - (s as f32 / (SHADES - 1) as f32) * 0.92;
                    let mut row = [0u8; 256];
                    for (i, slot) in row.iter_mut().enumerate() {
                        let j = i * 3;
                        let at = |k: usize| playpal.get(k).copied().unwrap_or(0) as f32 * f;
                        *slot = super::draw::nearest(
                            playpal,
                            at(j) as u8,
                            at(j + 1) as u8,
                            at(j + 2) as u8,
                        );
                    }
                    light.push(row);
                }
            }
        }

        Renderer {
            w,
            h,
            top: alloc::vec![0; w],
            bot: alloc::vec![h as i32 - 1; w],
            wall,
            step,
            light,
            lit_from_wad: from_wad.is_some(),
            open: w,
            solid: alloc::vec![f32::INFINITY; w],
            win_top: alloc::vec![0; w],
            win_bot: alloc::vec![h as i32 - 1; w],
            near: alloc::vec![f32::INFINITY; w],
            planes: alloc::vec::Vec::new(),
            live: 0,
            cur_ceil: None,
            cur_floor: None,
            spanstart: alloc::vec![0; h],
        }
    }

    fn reset(&mut self) {
        for x in 0..self.w {
            self.top[x] = 0;
            self.bot[x] = self.h as i32 - 1;
            self.solid[x] = f32::INFINITY;
            self.near[x] = f32::INFINITY;
            self.win_top[x] = 0;
            self.win_bot[x] = self.h as i32 - 1;
        }
        self.open = self.w;
        self.live = 0;
        self.cur_ceil = None;
        self.cur_floor = None;
    }

    /// The plane at this height wearing this picture under this light, making
    /// one if there is none.
    ///
    /// A linear scan, like DOOM's. The list is short because a plane is shared
    /// by every surface that agrees on all three -- one floor seen through
    /// four doorways is one plane, which is the whole economy of the idea.
    fn find_plane(&mut self, height: f32, flat: &'static [u8], light: i16) -> Option<usize> {
        for i in 0..self.live {
            let p = &self.planes[i];
            if p.height == height
                && p.light == light
                && core::ptr::eq(p.flat.as_ptr(), flat.as_ptr())
            {
                return Some(i);
            }
        }
        self.new_plane(height, flat, light)
    }

    fn new_plane(&mut self, height: f32, flat: &'static [u8], light: i16) -> Option<usize> {
        if self.live >= MAX_PLANES {
            return None;
        }
        let (i, w) = (self.live, self.w);
        if i == self.planes.len() {
            self.planes.push(Plane {
                height,
                flat,
                light,
                minx: w as i32,
                maxx: -1,
                top: alloc::vec![UNSET; w],
                bot: alloc::vec![0; w],
            });
        } else {
            let p = &mut self.planes[i];
            p.height = height;
            p.flat = flat;
            p.light = light;
            p.minx = w as i32;
            p.maxx = -1;
            // Only `top` is cleared. `bot` is never read where `top` is UNSET,
            // and clearing both would double the per-plane cost of a frame for
            // data nothing can observe.
            for t in p.top.iter_mut() {
                *t = UNSET;
            }
        }
        self.live += 1;
        Some(i)
    }

    /// Claim a range of columns in a plane, splitting it if they are taken.
    ///
    /// The subtle half of visplanes. One sector's floor can be visible in two
    /// disjoint parts of the screen -- through two doorways, or either side of
    /// a pillar -- and those are the *same* height, picture and light, so
    /// `find_plane` hands back the same plane for both. But a plane holds one
    /// top and one bottom per column, so two overlapping claims cannot both be
    /// recorded. When the new range touches a column already marked, this
    /// makes a second plane with identical keys instead. Without it the later
    /// claim silently overwrites the earlier and one of the two views loses
    /// its floor.
    fn check_plane(&mut self, idx: usize, start: i32, stop: i32) -> Option<usize> {
        let (intrl, unionl, intrh, unionh) = {
            let p = &self.planes[idx];
            let (il, ul) = if start < p.minx { (p.minx, start) } else { (start, p.minx) };
            let (ih, uh) = if stop > p.maxx { (p.maxx, stop) } else { (stop, p.maxx) };
            (il, ul, ih, uh)
        };
        let mut x = intrl;
        while x <= intrh {
            if self.planes[idx].top[x as usize] != UNSET {
                break;
            }
            x += 1;
        }
        if x > intrh {
            let p = &mut self.planes[idx];
            p.minx = unionl;
            p.maxx = unionh;
            return Some(idx);
        }
        let (h, f, l) = {
            let p = &self.planes[idx];
            (p.height, p.flat, p.light)
        };
        let n = self.new_plane(h, f, l)?;
        let p = &mut self.planes[n];
        p.minx = start;
        p.maxx = stop;
        Some(n)
    }

    /// Record that one column of a plane covers rows `a` to `b`.
    fn mark(&mut self, idx: usize, x: i32, a: i32, b: i32) {
        if a > b {
            return;
        }
        let p = &mut self.planes[idx];
        p.top[x as usize] = a.max(0);
        p.bot[x as usize] = b;
    }

    /// Which sector a subsector is in.
    ///
    /// Its first seg that has a sidedef, which is how the level reader answers
    /// the same question -- every seg of a subsector faces the same sector, so
    /// any of them will do and the first is cheapest.
    fn subsector_sector<'l>(
        &self,
        lv: &'l Level,
        ss: &super::level::SubSector,
    ) -> Option<&'l super::level::Sector> {
        for i in 0..ss.count as usize {
            let seg = lv.segs.get(ss.first as usize + i)?;
            if seg.linedef == NONE {
                continue;
            }
            let line = lv.linedefs.get(seg.linedef as usize)?;
            let side = if seg.side == 0 { line.right } else { line.left };
            if side == NONE {
                continue;
            }
            let sd = lv.sidedefs.get(side as usize)?;
            return lv.sectors.get(sd.sector as usize);
        }
        None
    }

    /// How dark a surface at this distance in this sector is, as a fraction of
    /// the way into the ramp.
    ///
    /// Two things darken a wall: how far away it is, and how dark its sector
    /// is. Combined multiplicatively, which is what makes a dim room stay dim
    /// when you walk up to a wall in it.
    fn level(dist: f32, light: i16) -> usize {
        let by_dist = 1.0 - (dist / FALLOFF).min(1.0);
        let by_light = (light.clamp(0, 255) as f32) / 255.0;
        let f = (by_dist * (0.25 + 0.75 * by_light)).clamp(0.0, 1.0);
        (((1.0 - f) * (SHADES - 1) as f32) as usize).min(SHADES - 1)
    }

    /// Draw one frame.
    pub fn frame(&mut self, surf: &mut Surface, lv: &Level, art: &Art<'_>, view: View, sky: u8) {
        self.reset();
        surf.clear(sky);
        // The projection scale for a 90-degree horizontal field of view, which
        // is DOOM's. `tan(45) == 1`, so it is simply half the width.
        let scale = self.w as f32 / 2.0;
        let (sin, cos) = (math::sin(view.angle), math::cos(view.angle));
        self.walk(surf, lv, art, &view, lv.root(), sin, cos, scale);
        // After the walls, and safely so: a plane only ever recorded rows no
        // wall had claimed, so painting them last cannot overwrite one.
        self.draw_planes(surf, &view, sin, cos, scale);
        self.record_windows();
    }

    /// The BSP, near side first.
    ///
    /// Recursion depth is bounded by the tree's height, which for a real map
    /// is a few dozen. A malformed tree could be a cycle, which is why the
    /// child indices were checked when the level loaded rather than here --
    /// checking here would mean checking per frame.
    #[allow(clippy::too_many_arguments)]
    fn walk(
        &mut self,
        surf: &mut Surface,
        lv: &Level,
        art: &Art<'_>,
        view: &View,
        node: u16,
        sin: f32,
        cos: f32,
        scale: f32,
    ) {
        if self.open == 0 {
            return;
        }
        if node & SUBSECTOR_BIT != 0 {
            let i = (node & !SUBSECTOR_BIT) as usize;
            if let Some(ss) = lv.subsectors.get(i) {
                // The two planes every seg of this subsector marks into.
                // Chosen here rather than per seg because a subsector is one
                // sector, so all its segs agree on height, picture and light --
                // and because `find_plane` is a scan, which per seg on a map
                // with two thousand of them is real work for one answer.
                match self.subsector_sector(lv, ss) {
                    Some(sec) => {
                        let (ch, cl, fh, fl) =
                            (sec.ceiling as f32, sec.light, sec.floor as f32, sec.light);
                        let cf = art.flat(&sec.ceiling_pic);
                        let ff = art.flat(&sec.floor_pic);
                        self.cur_ceil = cf.and_then(|f| self.find_plane(ch, f, cl));
                        self.cur_floor = ff.and_then(|f| self.find_plane(fh, f, fl));
                    }
                    None => {
                        self.cur_ceil = None;
                        self.cur_floor = None;
                    }
                }
                for s in 0..ss.count as usize {
                    let seg = match lv.segs.get(ss.first as usize + s) {
                        Some(g) => *g,
                        None => continue,
                    };
                    self.seg(surf, lv, art, view, &seg, sin, cos, scale);
                }
            }
            return;
        }
        let Some(n) = lv.nodes.get(node as usize) else { return };
        let near = n.side_of(view.x as i32, view.y as i32);
        let (a, b) = if near == 0 { (n.right, n.left) } else { (n.left, n.right) };
        self.walk(surf, lv, art, view, a, sin, cos, scale);
        self.walk(surf, lv, art, view, b, sin, cos, scale);
    }

    /// One seg: project it, clip it, and fill what it claims.
    #[allow(clippy::too_many_arguments)]
    fn seg(
        &mut self,
        surf: &mut Surface,
        lv: &Level,
        art: &Art<'_>,
        view: &View,
        seg: &super::level::Seg,
        sin: f32,
        cos: f32,
        scale: f32,
    ) {
        // A miniseg is a nodebuilder's own edge along a partition, not a wall.
        if seg.linedef == NONE {
            return;
        }
        let pics = art.pics;
        let Some(line) = lv.linedefs.get(seg.linedef as usize) else { return };
        // Which face of the line this seg is: side 0 runs with the linedef, so
        // its sidedef is the right one. Getting this backwards puts every
        // wall's far sector in front of it.
        let (front_i, back_i) =
            if seg.side == 0 { (line.right, line.left) } else { (line.left, line.right) };
        let Some(front_side) = lv.sidedefs.get(front_i as usize) else { return };
        let Some(front) = lv.sectors.get(front_side.sector as usize) else { return };
        let back = if back_i == NONE {
            None
        } else {
            lv.sidedefs
                .get(back_i as usize)
                .and_then(|s| lv.sectors.get(s.sector as usize))
        };

        let Some(v1) = lv.vertexes.get(seg.v1 as usize) else { return };
        let Some(v2) = lv.vertexes.get(seg.v2 as usize) else { return };

        // How long this seg is in world units, which is also how much texture
        // it spans: DOOM walls are one texel per unit and there is no scaling
        // field anywhere in the format.
        let (sdx, sdy) = ((v2.x - v1.x) as f32, (v2.y - v1.y) as f32);
        let seg_len = math::sqrt(sdx * sdx + sdy * sdy);

        // Into view space: +depth is forward, +lat is right.
        let to_view = |wx: f32, wy: f32| -> (f32, f32) {
            let (dx, dy) = (wx - view.x, wy - view.y);
            (dx * sin - dy * cos, dx * cos + dy * sin)
        };
        let (mut l1, mut d1) = to_view(v1.x as f32, v1.y as f32);
        let (mut l2, mut d2) = to_view(v2.x as f32, v2.y as f32);
        // Distance along the seg at each end, which becomes the horizontal
        // texture coordinate. Clipped by the same fraction as everything else
        // below, because it is linear in the same parameter.
        let (mut u1, mut u2) = (0.0f32, seg_len);

        // Entirely behind the eye.
        if d1 < NEAR && d2 < NEAR {
            return;
        }
        // Clip the crossing end to the near plane. Lateral is interpolated by
        // the same fraction, which is exact because both are linear in the
        // parameter along the segment -- it is only the *projection* that is
        // not.
        if d1 < NEAR {
            let t = (NEAR - d1) / (d2 - d1);
            l1 += (l2 - l1) * t;
            u1 += (u2 - u1) * t;
            d1 = NEAR;
        } else if d2 < NEAR {
            let t = (NEAR - d2) / (d1 - d2);
            l2 += (l1 - l2) * t;
            u2 += (u1 - u2) * t;
            d2 = NEAR;
        }

        let cx = self.w as f32 / 2.0;
        let x1f = cx + l1 / d1 * scale;
        let x2f = cx + l2 / d2 * scale;

        // Backface cull. A seg is wound so its front faces the sector it
        // belongs to; projected, that means x1 < x2. The reversed case is the
        // same wall seen from behind, which some other seg will draw.
        if x2f <= x1f {
            return;
        }

        let xa = x1f.max(0.0) as i32;
        let xb = (x2f.min(self.w as f32 - 1.0)) as i32;
        if xb < xa {
            return;
        }

        // Perspective correction, and the whole reason this is not a linear
        // interpolation: depth is not linear across the screen but its
        // reciprocal is. Interpolating depth directly bends every wall, and
        // interpolating the texture coordinate directly is the wobble every
        // early console port of this game had. `u/z` is the quantity that runs
        // straight, so that is what is stepped and `u` is recovered per column.
        // Claim these columns in the subsector's two planes before marking
        // any. `check_plane` can hand back a *different* plane -- see its own
        // note -- and the replacement has to stick for the rest of the
        // subsector, not just this seg.
        if let Some(pi) = self.cur_ceil {
            self.cur_ceil = self.check_plane(pi, xa, xb);
        }
        if let Some(pi) = self.cur_floor {
            self.cur_floor = self.check_plane(pi, xa, xb);
        }

        let inv1 = 1.0 / d1;
        let inv2 = 1.0 / d2;
        let base = seg.offset as f32 + front_side.x_off as f32;
        let uo1 = (u1 + base) * inv1;
        let uo2 = (u2 + base) * inv2;
        let span = x2f - x1f;

        let ceil = front.ceiling as f32;
        let floor = front.floor as f32;
        let cy = self.h as f32 / 2.0;
        let y_off = front_side.y_off as f32;

        // Which texture each part of this wall wears, resolved once for the
        // whole seg rather than per column.
        let named = |n: &super::wad::Name| -> Option<usize> {
            let p = pics?;
            // `-` is the WAD's own spelling for "no texture here", and it is
            // the common case on a two-sided line.
            if n.is("-") || n.as_str().is_empty() {
                return None;
            }
            p.index_of(n.as_str())
        };
        let mid_t = named(&front_side.middle);
        let up_t = named(&front_side.upper);
        let low_t = named(&front_side.lower);
        let height_of = |t: Option<usize>| -> f32 {
            t.and_then(|i| pics.and_then(|p| p.def(i))).map(|d| d.height as f32).unwrap_or(0.0)
        };

        // Where texture row zero sits in world height, which is the whole of
        // DOOM's "pegging". The default anchors a texture where the geometry
        // it decorates begins and the flag moves it to the other end, so that
        // a door's texture stays still while the door moves. Getting one of
        // these backwards is not subtle: every step in the map wears its
        // texture upside down.
        let peg_top = line.flags & DONTPEG_TOP != 0;
        let peg_bot = line.flags & DONTPEG_BOTTOM != 0;
        let mid_anchor = if peg_bot { floor + height_of(mid_t) } else { ceil } + y_off;
        // The two step anchors depend on the sector behind, which is fixed for
        // the whole seg. Computed here rather than per column, where they were
        // recomputing the same two numbers a few hundred times a wall.
        let (up_anchor, low_anchor) = match back {
            None => (0.0, 0.0),
            Some(b) => (
                if peg_top { ceil } else { b.ceiling as f32 + height_of(up_t) } + y_off,
                if peg_bot { ceil } else { b.floor as f32 } + y_off,
            ),
        };

        for x in xa..=xb {
            if self.top[x as usize] > self.bot[x as usize] {
                continue;
            }
            let t = (((x as f32 + 0.5) - x1f) / span).clamp(0.0, 1.0);
            let inv = inv1 + (inv2 - inv1) * t;
            if inv <= 0.0 {
                continue;
            }
            let dist = 1.0 / inv;
            let u = (uo1 + (uo2 - uo1) * t) * dist;
            let lit = Self::level(dist, front.light);

            let y_of = |world_z: f32| -> i32 {
                (cy - (world_z - view.z) * scale * inv) as i32
            };
            let wall_top = y_of(ceil);
            let wall_bot = y_of(floor);

            // Record the rows of this column that belong to the front
            // sector's ceiling and floor. *Recorded*, not drawn: the columns a
            // plane covers are not known until the walk is over, and drawing
            // per column here is exactly the shortcut that left a subsector no
            // seg reached with no floor at all.
            //
            // Read before the wall rather than after, because the window
            // `close` is about to shut is the same one that says which rows
            // are still unclaimed. Both branches below want this: above
            // `wall_top` is the front ceiling whether the wall is solid or a
            // portal, and likewise below `wall_bot`. Nothing is recorded
            // twice, because the window narrows past both immediately after.
            if let Some(pi) = self.cur_ceil {
                let (a, b) = (self.top[x as usize], (wall_top - 1).min(self.bot[x as usize]));
                self.mark(pi, x, a, b);
            }
            if let Some(pi) = self.cur_floor {
                let (a, b) = ((wall_bot + 1).max(self.top[x as usize]), self.bot[x as usize]);
                self.mark(pi, x, a, b);
            }

            match back {
                // A solid wall: it fills the column and closes it.
                None => {
                    // Nothing behind a solid wall is ever visible in this
                    // column again, which is what a sprite needs to know.
                    self.solid[x as usize] = self.solid[x as usize].min(dist);
                    self.near[x as usize] = self.near[x as usize].min(dist);
                    match mid_t {
                        Some(tex) => self.tex_fill(
                            surf, pics, tex, x, wall_top, wall_bot, u, mid_anchor, view.z, dist,
                            scale, lit,
                        ),
                        None => self.fill(surf, x, wall_top, wall_bot, self.wall[lit]),
                    }
                    self.close(x);
                }
                // A portal: only the steps above and below the opening are
                // wall. The gap between them is whatever is beyond, which the
                // traversal has yet to reach.
                //
                // A two-sided line's *middle* texture is deliberately not
                // drawn. It is the masked mid-wall -- a grate, a railing, the
                // bars on a window -- and it is the one wall in DOOM that is
                // drawn after everything behind it rather than before, because
                // you can see through it. Drawing it here with the solid path
                // would seal every open doorway that happens to name one.
                Some(b) => {
                    let mut closed = true;
                    self.near[x as usize] = self.near[x as usize].min(dist);
                    if b.ceiling < front.ceiling {
                        let step_bot = y_of(b.ceiling as f32);
                        match up_t {
                            Some(tex) => self.tex_fill(
                                surf, pics, tex, x, wall_top, step_bot, u, up_anchor, view.z,
                                dist, scale, lit,
                            ),
                            None => self.fill(surf, x, wall_top, step_bot, self.step[lit]),
                        }
                        self.top[x as usize] = self.top[x as usize].max(step_bot + 1);
                    } else {
                        self.top[x as usize] = self.top[x as usize].max(wall_top);
                    }
                    if b.floor > front.floor {
                        let step_top = y_of(b.floor as f32);
                        match low_t {
                            Some(tex) => self.tex_fill(
                                surf, pics, tex, x, step_top, wall_bot, u, low_anchor, view.z,
                                dist, scale, lit,
                            ),
                            None => self.fill(surf, x, step_top, wall_bot, self.step[lit]),
                        }
                        self.bot[x as usize] = self.bot[x as usize].min(step_top - 1);
                    } else {
                        self.bot[x as usize] = self.bot[x as usize].min(wall_bot);
                    }
                    // A door shut, or a sector whose floor has risen to its
                    // own ceiling: the opening has no height, so the column is
                    // as closed as a wall.
                    closed &= b.floor >= b.ceiling;
                    if closed {
                        self.close(x);
                    }
                }
            }
        }
    }

    /// Paint the open part of one column out of a texture.
    ///
    /// `anchor` is the world height of the texture's first row, `y_off`
    /// already folded in. Everything else follows from it: the world height at
    /// a screen row is linear in the row, so the texture coordinate is too,
    /// and one step per row does the whole column with no division inside the
    /// loop.
    #[allow(clippy::too_many_arguments)]
    fn tex_fill(
        &mut self,
        surf: &mut Surface,
        pics: Option<&Pics>,
        tex: usize,
        x: i32,
        y0: i32,
        y1: i32,
        u: f32,
        anchor: f32,
        view_z: f32,
        dist: f32,
        scale: f32,
        lit: usize,
    ) {
        let lo = y0.max(self.top[x as usize]).max(0);
        let hi = y1.min(self.bot[x as usize]).min(self.h as i32 - 1);
        if lo > hi {
            return;
        }
        let Some(p) = pics else { return };
        let Some(col) = p.column(tex, math::floor_i(u)) else { return };
        let th = col.len() as i32;
        if th <= 0 {
            return;
        }
        let dv = dist / scale;
        let cy = self.h as f32 / 2.0;
        let mut v = anchor - view_z - (cy - lo as f32) * dv;
        let map = &self.light[lit];
        let w = self.w;
        let px = surf.pixels();
        for y in lo..=hi {
            // Wrapped, because a wall taller than its texture repeats it --
            // that is how DOOM tiles -- and because `v` runs negative above a
            // texture's anchor, where a truncating cast would fold the top row
            // onto the second one.
            let row = math::floor_i(v).rem_euclid(th) as usize;
            px[y as usize * w + x as usize] = map[col[row] as usize];
            v += dv;
        }
    }

    /// Every plane this frame accumulated, as horizontal spans.
    fn draw_planes(&mut self, surf: &mut Surface, view: &View, sin: f32, cos: f32, scale: f32) {
        for i in 0..self.live {
            self.draw_plane(surf, i, view, sin, cos, scale);
        }
    }

    /// One plane, walked column by column, emitting a span wherever a run of
    /// rows ends.
    ///
    /// This is DOOM's `R_MakeSpans` and the four loops are its. Each row of the
    /// plane is a run that opens at some column and closes at another, and the
    /// only thing that can be known at column `x` is which runs ended at
    /// `x - 1`. So the walk compares this column's extent against the last
    /// one's: rows that were covered and no longer are get drawn, rows that are
    /// covered and were not get their start recorded. One extra column past the
    /// end closes whatever is still open.
    fn draw_plane(&mut self, surf: &mut Surface, i: usize, view: &View, sin: f32, cos: f32, scale: f32) {
        let (minx, maxx, height, light, flat) = {
            let p = &self.planes[i];
            (p.minx, p.maxx, p.height, p.light, p.flat)
        };
        if minx > maxx {
            return;
        }
        let col = |me: &Self, x: i32| -> (i32, i32) {
            if x < minx || x > maxx {
                return (UNSET, -1);
            }
            let p = &me.planes[i];
            let t = p.top[x as usize];
            if t == UNSET {
                (UNSET, -1)
            } else {
                (t, p.bot[x as usize])
            }
        };

        let (mut t1, mut b1) = (UNSET, -1);
        for x in minx..=maxx + 1 {
            let (mut t2, mut b2) = col(self, x);
            while t1 < t2 && t1 <= b1 {
                let from = self.spanstart[t1 as usize];
                self.map_row(surf, flat, height, light, t1, from, x - 1, view, sin, cos, scale);
                t1 += 1;
            }
            while b1 > b2 && b1 >= t1 {
                let from = self.spanstart[b1 as usize];
                self.map_row(surf, flat, height, light, b1, from, x - 1, view, sin, cos, scale);
                b1 -= 1;
            }
            while t2 < t1 && t2 <= b2 {
                self.spanstart[t2 as usize] = x;
                t2 += 1;
            }
            while b2 > b1 && b2 >= t2 {
                self.spanstart[b2 as usize] = x;
                b2 -= 1;
            }
            let (nt, nb) = col(self, x);
            t1 = nt;
            b1 = nb;
        }
    }

    /// One horizontal span of a level plane.
    ///
    /// **The whole point of visplanes is in the first two lines.** Distance is
    /// constant along a screen row of a level plane, so `z` is computed once
    /// for the row rather than once per pixel -- and with `z` fixed, the world
    /// position walks the row by addition. A pixel costs two adds and a table
    /// lookup where the per-column version cost a divide.
    #[allow(clippy::too_many_arguments)]
    fn map_row(
        &self,
        surf: &mut Surface,
        flat: &[u8],
        height: f32,
        light: i16,
        y: i32,
        x1: i32,
        x2: i32,
        view: &View,
        sin: f32,
        cos: f32,
        scale: f32,
    ) {
        if x1 > x2 || y < 0 || y >= self.h as i32 {
            return;
        }
        let (cx, cy) = (self.w as f32 / 2.0, self.h as f32 / 2.0);
        let dy = y as f32 + 0.5 - cy;
        // Negative above the horizon and positive below it, matching the sign
        // of `rise` for a ceiling and a floor respectively -- so one expression
        // covers both, and a row on the wrong side of the horizon for this
        // plane is one a level surface never reaches.
        let rise = view.z - height;
        if (rise < 0.0) != (dy < 0.0) {
            return;
        }
        let z = rise * scale / dy;
        if !(z > 0.0 && z < 1.0e6) {
            return;
        }
        let k = (x1 as f32 + 0.5 - cx) / scale;
        let mut wx = view.x + z * (k * sin + cos);
        // North is negated because a flat's rows run the other way: DOOM
        // computes `-viewy - ...` for the row against `viewx + ...` for the
        // column.
        let mut wy = -(view.y + z * (sin - k * cos));
        let (sx, sy) = (z * sin / scale, z * cos / scale);

        let mask = FLAT_SIDE as i32 - 1;
        let map = &self.light[Self::level(z, light)];
        let w = self.w;
        let px = surf.pixels();
        for x in x1..=x2 {
            if x < 0 || x >= w as i32 {
                wx += sx;
                wy += sy;
                continue;
            }
            let cxi = (math::floor_i(wx) & mask) as usize;
            let ryi = (math::floor_i(wy) & mask) as usize;
            if let Some(v) = flat.get(ryi * FLAT_SIDE + cxi) {
                px[y as usize * w + x as usize] = map[*v as usize];
            }
            wx += sx;
            wy += sy;
        }
    }

    /// Paint the open part of one column in one colour, and no more.
    fn fill(&mut self, surf: &mut Surface, x: i32, y0: i32, y1: i32, c: u8) {
        let (w, h) = (self.w, self.h);
        let lo = y0.max(self.top[x as usize]).max(0);
        let hi = y1.min(self.bot[x as usize]).min(h as i32 - 1);
        if lo > hi {
            return;
        }
        let px = surf.pixels();
        for y in lo..=hi {
            px[y as usize * w + x as usize] = c;
        }
    }

    fn close(&mut self, x: i32) {
        if self.top[x as usize] <= self.bot[x as usize] {
            self.open -= 1;
        }
        self.top[x as usize] = 1;
        self.bot[x as usize] = 0;
    }

    /// Keep what each column was left open to, for the sprite pass.
    fn record_windows(&mut self) {
        for x in 0..self.w {
            if self.top[x] <= self.bot[x] {
                self.win_top[x] = self.top[x];
                self.win_bot[x] = self.bot[x];
            }
        }
    }

    /// Every thing on the level, back to front, over the world already drawn.
    ///
    /// After the walls rather than during, because a billboard is not part of
    /// the BSP and has no place in a front-to-back order -- two sprites in one
    /// subsector can be any way round. So they are sorted by distance and
    /// painted far to near, which is the one place this renderer overdraws
    /// and the reason DOOM calls them *masked* rather than solid.
    pub fn things(
        &mut self,
        surf: &mut Surface,
        things: &super::sprite::Things,
        view: &View,
        scale: f32,
    ) {
        let billboards = &things.items;
        let (sin, cos) = (math::sin(view.angle), math::cos(view.angle));
        // Depth per thing, computed once, so the sort compares numbers rather
        // than recomputing a transform per comparison.
        let mut order: alloc::vec::Vec<(usize, f32)> = alloc::vec::Vec::new();
        for (i, b) in billboards.iter().enumerate() {
            let (dx, dy) = (b.x - view.x, b.y - view.y);
            let depth = dx * cos + dy * sin;
            if depth > NEAR {
                order.push((i, depth));
            }
        }
        // Far to near. `total_cmp` rather than `partial_cmp`, because a NaN
        // depth would make the comparator inconsistent and the sort's contract
        // is not defined for one -- and a NaN here would come from a thing at
        // exactly the eye, which the near-plane test above does not exclude.
        order.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        for (i, depth) in order {
            let b = &billboards[i];
            let (dx, dy) = (b.x - view.x, b.y - view.y);
            // Which of the thing's facings this bearing shows, and whether
            // the lump for it is the mirror of one the WAD stores once.
            let Some(art) = things.art.get(b.art) else { continue };
            let Some((patch, flip)) = art.pick(math::atan2(dy, dx), b.angle) else { continue };
            let lat = dx * sin - dy * cos;
            let inv = 1.0 / depth;
            let (cx, cy) = (self.w as f32 / 2.0, self.h as f32 / 2.0);
            let px_per_unit = scale * inv;

            // `left` is the distance from the picture's left edge to the
            // thing's own centre, and `top` from its top edge down to the
            // thing's feet. Ignoring either is not subtle: the first hangs
            // every object half its width to one side, the second buries it
            // in the floor or floats it.
            let x1f = cx + (lat - patch.left as f32) * px_per_unit;
            let gzt = b.z + patch.top as f32;
            let y1f = cy - (gzt - view.z) * px_per_unit;

            let xa = x1f.max(0.0) as i32;
            let xb = ((x1f + patch.width as f32 * px_per_unit) as i32).min(self.w as i32 - 1);
            if xb < xa {
                continue;
            }
            let lit = Self::level(depth, b.light);
            for x in xa..=xb {
                // Behind a solid wall in this column, or behind something that
                // narrowed it and outside what it left open.
                if depth >= self.solid[x as usize] {
                    continue;
                }
                let clipped = depth >= self.near[x as usize];
                let col_i = ((x as f32 + 0.5 - x1f) / px_per_unit) as usize;
                // Mirrored where the lump serves the opposite facing: a
                // monster is symmetric about the way it points, so id stored
                // one picture for each off-axis pair. Reading the column from
                // the far end is the whole of drawing the other one.
                let col_i = if flip {
                    match patch.width.checked_sub(1).and_then(|w| w.checked_sub(col_i)) {
                        Some(c) => c,
                        None => continue,
                    }
                } else {
                    col_i
                };
                let Some(posts) = patch.columns.get(col_i) else { continue };
                // Walk the *destination* rows and ask each which source row it
                // wants, rather than walking the source and placing each one.
                // The two agree only while a sprite is being shrunk: magnify
                // one and consecutive source rows land more than a pixel
                // apart, so the sprite comes out combed with background
                // showing between its own scanlines. It is invisible at any
                // distance a test map happens to place an object at and
                // obvious the moment you walk up to it.
                let y_lo = y1f.max(0.0) as i32;
                let y_hi = ((y1f + patch.height as f32 * px_per_unit)
                    .min(self.h as f32 - 1.0)) as i32;
                for y in y_lo..=y_hi {
                    if y < 0 {
                        continue;
                    }
                    if clipped && (y < self.win_top[x as usize] || y > self.win_bot[x as usize]) {
                        continue;
                    }
                    let src = ((y as f32 + 0.5 - y1f) / px_per_unit) as usize;
                    // Which post covers that row, if any. A sprite is mostly
                    // hole, so most rows of most columns land in no post at
                    // all and are simply left alone -- that is the whole of
                    // what makes a billboard masked rather than a rectangle.
                    for post in posts.iter() {
                        if src >= post.top && src < post.top + post.pixels.len() {
                            let v = post.pixels[src - post.top];
                            let c = self.light[lit][v as usize];
                            let w = self.w;
                            surf.pixels()[y as usize * w + x as usize] = c;
                            break;
                        }
                    }
                }
            }
        }
    }
}

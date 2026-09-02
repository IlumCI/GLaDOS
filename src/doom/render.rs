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
//! **No textures yet, deliberately.** Walls are shaded by distance and by
//! their sector's light. That is one slice: it proves the traversal, the
//! projection and the clipping, which is the hard half, without also needing
//! TEXTURE1, PNAMES and patch composition, which is the tedious half. A wrong
//! picture here is a geometry bug and cannot be a texture bug, and that is
//! worth a great deal when the picture is wrong.
//!
//! ### Where this departs from the original
//!
//! DOOM projects through angles: `R_PointToAngle` and a `viewangletox` table
//! turn a world point into a column directly, which on a 486 avoids a divide
//! per vertex. This transforms into view space and divides, because the divide
//! is now free and the angle tables are a source of subtle error that is hard
//! to see and harder to test. The geometry is identical; the arithmetic is
//! not.

use super::level::{Level, NONE, SUBSECTOR_BIT};
use super::math;
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

/// How many shades a wall colour is quantised into.
const SHADES: usize = 24;

/// Distance at which a wall reaches the darkest shade.
const FALLOFF: f32 = 1400.0;

pub struct Renderer {
    w: usize,
    h: usize,
    /// First row of each column not yet claimed, from the top.
    top: alloc::vec::Vec<i32>,
    /// Last row not yet claimed, from the bottom.
    bot: alloc::vec::Vec<i32>,
    /// Wall shades, bright to dark, and the same for the step faces so an
    /// upper reads as part of the same world rather than as a decal.
    wall: [u8; SHADES],
    step: [u8; SHADES],
    /// How many columns are still open, so a full screen can stop the walk.
    open: usize,
}

impl Renderer {
    pub fn new(surf: &Surface, playpal: &[u8]) -> Renderer {
        let (w, h) = (surf.width(), surf.height());
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
        Renderer {
            w,
            h,
            top: alloc::vec![0; w],
            bot: alloc::vec![h as i32 - 1; w],
            wall,
            step,
            open: w,
        }
    }

    fn reset(&mut self) {
        for x in 0..self.w {
            self.top[x] = 0;
            self.bot[x] = self.h as i32 - 1;
        }
        self.open = self.w;
    }

    fn shade(ramp: &[u8; SHADES], dist: f32, light: i16) -> u8 {
        // Two things darken a wall: how far away it is, and how dark its
        // sector is. Combined multiplicatively, which is what makes a dim room
        // stay dim when you walk up to a wall in it.
        let by_dist = 1.0 - (dist / FALLOFF).min(1.0);
        let by_light = (light.clamp(0, 255) as f32) / 255.0;
        let f = (by_dist * (0.25 + 0.75 * by_light)).clamp(0.0, 1.0);
        let i = ((1.0 - f) * (SHADES - 1) as f32) as usize;
        ramp[i.min(SHADES - 1)]
    }

    /// Draw one frame.
    pub fn frame(&mut self, surf: &mut Surface, lv: &Level, view: View, sky: u8) {
        self.reset();
        surf.clear(sky);
        // The projection scale for a 90-degree horizontal field of view, which
        // is DOOM's. `tan(45) == 1`, so it is simply half the width.
        let scale = self.w as f32 / 2.0;
        let (sin, cos) = (math::sin(view.angle), math::cos(view.angle));
        self.walk(surf, lv, &view, lv.root(), sin, cos, scale);
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
                for s in 0..ss.count as usize {
                    let seg = match lv.segs.get(ss.first as usize + s) {
                        Some(g) => *g,
                        None => continue,
                    };
                    self.seg(surf, lv, view, &seg, sin, cos, scale);
                }
            }
            return;
        }
        let Some(n) = lv.nodes.get(node as usize) else { return };
        let near = n.side_of(view.x as i32, view.y as i32);
        let (a, b) = if near == 0 { (n.right, n.left) } else { (n.left, n.right) };
        self.walk(surf, lv, view, a, sin, cos, scale);
        self.walk(surf, lv, view, b, sin, cos, scale);
    }

    /// One seg: project it, clip it, and fill what it claims.
    #[allow(clippy::too_many_arguments)]
    fn seg(
        &mut self,
        surf: &mut Surface,
        lv: &Level,
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

        // Into view space: +depth is forward, +lat is right.
        let to_view = |wx: f32, wy: f32| -> (f32, f32) {
            let (dx, dy) = (wx - view.x, wy - view.y);
            (dx * sin - dy * cos, dx * cos + dy * sin)
        };
        let (mut l1, mut d1) = to_view(v1.x as f32, v1.y as f32);
        let (mut l2, mut d2) = to_view(v2.x as f32, v2.y as f32);

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
            d1 = NEAR;
        } else if d2 < NEAR {
            let t = (NEAR - d2) / (d1 - d2);
            l2 += (l1 - l2) * t;
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
        // reciprocal is. Interpolating depth directly bends every wall.
        let inv1 = 1.0 / d1;
        let inv2 = 1.0 / d2;
        let span = x2f - x1f;

        let ceil = front.ceiling as f32;
        let floor = front.floor as f32;
        let cy = self.h as f32 / 2.0;

        for x in xa..=xb {
            if self.top[x as usize] > self.bot[x as usize] {
                continue;
            }
            let t = ((x as f32 + 0.5) - x1f) / span;
            let inv = inv1 + (inv2 - inv1) * t.clamp(0.0, 1.0);
            if inv <= 0.0 {
                continue;
            }
            let dist = 1.0 / inv;

            let y_of = |world_z: f32| -> i32 {
                (cy - (world_z - view.z) * scale * inv) as i32
            };
            let wall_top = y_of(ceil);
            let wall_bot = y_of(floor);

            match back {
                // A solid wall: it fills the column and closes it.
                None => {
                    let c = Self::shade(&self.wall, dist, front.light);
                    self.fill(surf, x, wall_top, wall_bot, c);
                    self.close(x);
                }
                // A portal: only the steps above and below the opening are
                // wall. The gap between them is whatever is beyond, which the
                // traversal has yet to reach.
                Some(b) => {
                    let mut closed = true;
                    if b.ceiling < front.ceiling {
                        let step_bot = y_of(b.ceiling as f32);
                        let c = Self::shade(&self.step, dist, front.light);
                        self.fill(surf, x, wall_top, step_bot, c);
                        self.top[x as usize] = self.top[x as usize].max(step_bot + 1);
                    } else {
                        self.top[x as usize] = self.top[x as usize].max(wall_top);
                    }
                    if b.floor > front.floor {
                        let step_top = y_of(b.floor as f32);
                        let c = Self::shade(&self.step, dist, front.light);
                        self.fill(surf, x, step_top, wall_bot, c);
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

    /// Paint the open part of one column, and no more.
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
}

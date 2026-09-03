//! Hitscan: what a shot meets, and what that does to it.
//!
//! Ported from room4doom's `gameplay/src/thing/shooting.rs`, cut hard. What is
//! here is a shot travelling in a straight line until a wall or a thing stops
//! it, and a radius attack for the one action that needs it.
//!
//! ### The blockmap is still not read, and this is the third place to say so
//!
//! DOOM traces a shot with `P_PathTraverse`, which walks the blockmap cell by
//! cell and only ever looks at the handful of lines and things in each. This
//! walks every linedef and every object. `blocked` does the same and so does
//! `use_lines`; the note is repeated because a shot is the first thing here
//! that runs the walk *per press* over the whole map rather than over a
//! 64-unit reach, and it is therefore where the cost will show up first.
//!
//! ### Boxes, not circles
//!
//! A thing is hit when the ray crosses the **square** of its radius, not the
//! circle. That is DOOM's, through `PIT_AddThingIntercepts`, and it is the
//! same square the pickup test uses -- so a barrel is very slightly easier to
//! hit on the diagonal than head on. Rounding it to a circle would be tidier
//! and would be a different game.
//!
//! ### What is missing, named rather than left to be discovered
//!
//! **There is no vertical aim.** DOOM's shot carries a slope, checks each
//! thing's top and bottom against it, and passes over anything too low or too
//! high; and a two-sided line stops a shot when the *opening* does not admit
//! it, not merely when it is shut. Here a shot is level: it stops at a wall
//! with no opening at all, and hits the first thing whose square it crosses.
//! On a map with a floor at one height that is the same answer. On a map with
//! a balcony it is not, and the difference is a shot that hits something
//! standing above you.
//!
//! **There is no randomness.** DOOM's pistol does `5 * (P_Random() % 3 + 1)`,
//! so 5, 10 or 15. This does 10 every time. The random table is DOOM's own
//! 256 bytes and inventing a different sequence would be a game that plays
//! differently from every other copy, so it arrives when the monsters that
//! need it do.

use alloc::vec::Vec;

use super::info;
use super::level::Level;
use super::math;
use super::play::Opening;
use super::thing::{Fired, Objs};

/// How far a shot reaches. DOOM's `MISSILERANGE`, 32 map blocks.
pub const RANGE: f32 = 32.0 * 64.0;

/// What the pistol takes off. See the note above about randomness -- this is
/// the middle of DOOM's three outcomes rather than one of them.
pub const PISTOL_DAMAGE: i32 = 10;

/// How far a barrel's blast reaches, and how much it does at the centre.
/// DOOM's `A_Explode` passes 128 for both, which is why one number does.
pub const BLAST: f32 = 128.0;

/// Where a ray first crosses a wall it cannot pass, as a distance.
///
/// A one-sided line always stops a shot. A two-sided one stops it only when
/// there is no opening at all -- a shut door, or a sector whose ceiling has
/// come down to its floor.
pub fn wall_range(lv: &Level, x: f32, y: f32, dx: f32, dy: f32) -> f32 {
    let mut best = RANGE;
    for l in lv.linedefs.iter() {
        let solid = match Opening::of(lv, l) {
            None => true,
            Some(o) => o.top <= o.bottom,
        };
        if !solid {
            continue;
        }
        let (Some(a), Some(b)) = (lv.vertexes.get(l.v1 as usize), lv.vertexes.get(l.v2 as usize))
        else {
            continue;
        };
        if let Some(t) = ray_meets_seg(
            x, y, dx, dy, a.x as f32, a.y as f32, b.x as f32, b.y as f32,
        ) {
            if t < best {
                best = t;
            }
        }
    }
    best
}

/// How far along a ray it crosses a segment, if it does.
///
/// The ray is infinite forward and the segment is not, so the two parameters
/// are bounded differently: `t >= 0` and `0 <= u <= 1`. Getting that the wrong
/// way round gives a shot stopped by the *extension* of a wall it is nowhere
/// near, which on a map full of collinear walls is almost every shot.
fn ray_meets_seg(
    x: f32,
    y: f32,
    dx: f32,
    dy: f32,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
) -> Option<f32> {
    let (sx, sy) = (bx - ax, by - ay);
    let denom = dx * sy - dy * sx;
    if denom == 0.0 {
        return None;
    }
    let (qx, qy) = (ax - x, ay - y);
    let t = (qx * sy - qy * sx) / denom;
    let u = (qx * dy - qy * dx) / denom;
    if t < 0.0 || !(0.0..=1.0).contains(&u) {
        return None;
    }
    Some(t)
}

/// How far along a ray it enters the square around a point, if it does.
///
/// The slab method, and the two degenerate cases are the whole of it: a ray
/// with no movement along an axis either starts inside that slab and is
/// unconstrained by it, or starts outside and can never enter.
fn ray_meets_box(x: f32, y: f32, dx: f32, dy: f32, cx: f32, cy: f32, r: f32) -> Option<f32> {
    let slab = |p: f32, d: f32, lo: f32, hi: f32| -> Option<(f32, f32)> {
        if d == 0.0 {
            return if p >= lo && p <= hi {
                Some((f32::NEG_INFINITY, f32::INFINITY))
            } else {
                None
            };
        }
        let (t1, t2) = ((lo - p) / d, (hi - p) / d);
        Some((t1.min(t2), t1.max(t2)))
    };
    let (nx, fx) = slab(x, dx, cx - r, cx + r)?;
    let (ny, fy) = slab(y, dy, cy - r, cy + r)?;
    let near = nx.max(ny).max(0.0);
    let far = fx.min(fy);
    if far < near {
        return None;
    }
    Some(near)
}

/// Fire one shot. Answers which object it hit, if any, and whether that killed
/// it.
///
/// The wall range is found first and the things are then tested against it, so
/// a shot cannot reach through a door to something behind it. That ordering is
/// the whole of the wall check: DOOM interleaves the two along the blockmap
/// walk and arrives at the same answer, one cell at a time.
pub fn fire(
    lv: &Level,
    objs: &mut Objs,
    x: f32,
    y: f32,
    angle: f32,
    damage: i32,
    out: &mut Vec<Fired>,
) -> Option<(usize, bool)> {
    let (dx, dy) = (math::cos(angle), math::sin(angle));
    let limit = wall_range(lv, x, y, dx, dy);
    let mut best: Option<(f32, usize)> = None;
    for (i, o) in objs.list.iter().enumerate() {
        if o.flags & info::MF_SHOOTABLE == 0 {
            continue;
        }
        let Some(t) = ray_meets_box(x, y, dx, dy, o.x, o.y, o.radius()) else { continue };
        if t > limit {
            continue;
        }
        if best.map(|(bt, _)| t < bt).unwrap_or(true) {
            best = Some((t, i));
        }
    }
    let (_, i) = best?;
    let mut acts: Vec<u8> = Vec::new();
    let died = objs.list[i].hurt(damage, &mut acts);
    let (hx, hy) = (objs.list[i].x, objs.list[i].y);
    for a in acts.drain(..) {
        out.push(Fired { action: a, x: hx, y: hy });
    }
    Some((i, died))
}

/// DOOM's `P_RadiusAttack`: everything near a point takes what is left of the
/// blast after the distance is taken off it.
///
/// The distance is `max(|dx|, |dy|)` less the thing's radius, which is a
/// square again and for the same reason -- and it means a barrel two hundred
/// units away diagonally is closer, to DOOM, than one a hundred and fifty away
/// in a straight line.
///
/// Sight is checked, and cheaply: a blast does not reach through a wall that
/// would have stopped a shot. DOOM uses `P_CheckSight`, which walks the BSP
/// and consults REJECT; this asks the same question the shot trace already
/// answers, which is less clever and gives the same answer for a blast that
/// only ever travels 128 units.
pub fn blast(lv: &Level, objs: &mut Objs, x: f32, y: f32, out: &mut Vec<Fired>) -> usize {
    let mut hurt = 0usize;
    let mut acts: Vec<u8> = Vec::new();
    for i in 0..objs.list.len() {
        let o = &objs.list[i];
        if o.flags & info::MF_SHOOTABLE == 0 {
            continue;
        }
        let (ex, ey) = ((o.x - x).abs(), (o.y - y).abs());
        let dist = (ex.max(ey) - o.radius()).max(0.0);
        if dist >= BLAST {
            continue;
        }
        let (tx, ty) = (o.x, o.y);
        if !visible(lv, x, y, tx, ty) {
            continue;
        }
        acts.clear();
        if objs.list[i].hurt((BLAST - dist) as i32, &mut acts) {
            hurt += 1;
        }
        for a in acts.drain(..) {
            out.push(Fired { action: a, x: tx, y: ty });
        }
    }
    hurt
}

/// Whether a straight line between two points meets no wall.
fn visible(lv: &Level, x1: f32, y1: f32, x2: f32, y2: f32) -> bool {
    let (dx, dy) = (x2 - x1, y2 - y1);
    let len = math::sqrt(dx * dx + dy * dy);
    if len <= 0.0 {
        return true;
    }
    wall_range(lv, x1, y1, dx / len, dy / len) >= len
}

/// What `diag doom` asks of the geometry a shot rests on.
///
/// Both of these are pure and neither needs a map, which is the point: a shot
/// that misses is indistinguishable from a shot that was never fired, and a
/// shot that stops at the extension of a wall it is nowhere near looks exactly
/// like a shot that hit something.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    // A ray crosses a segment in front of it and not the same segment placed
    // behind, and not the *extension* of one beside it.
    out.push((
        "a ray meets a segment ahead and neither one behind nor one beside",
        ray_meets_seg(0.0, 0.0, 1.0, 0.0, 50.0, -10.0, 50.0, 10.0) == Some(50.0)
            && ray_meets_seg(0.0, 0.0, 1.0, 0.0, -50.0, -10.0, -50.0, 10.0).is_none()
            && ray_meets_seg(0.0, 0.0, 1.0, 0.0, 50.0, 20.0, 50.0, 40.0).is_none(),
    ));

    // A ray parallel to a segment never meets it, whatever their spacing.
    out.push((
        "a parallel ray never meets it",
        ray_meets_seg(0.0, 0.0, 1.0, 0.0, 10.0, 5.0, 90.0, 5.0).is_none(),
    ));

    // The box, entered at its near face rather than its centre -- a shot stops
    // at the edge of what it hits, and using the centre would let it reach
    // through one thing to the one behind by its own radius.
    out.push((
        "a ray enters a box at its near face",
        ray_meets_box(0.0, 0.0, 1.0, 0.0, 100.0, 0.0, 20.0) == Some(80.0),
    ));

    // Missing it, above and behind.
    out.push((
        "a ray misses a box beside it and one behind it",
        ray_meets_box(0.0, 0.0, 1.0, 0.0, 100.0, 40.0, 20.0).is_none()
            && ray_meets_box(0.0, 0.0, 1.0, 0.0, -100.0, 0.0, 20.0).is_none(),
    ));

    // A ray that does not move along an axis is the degenerate case, and it
    // has both answers in it: inside that slab it is unconstrained, outside it
    // can never enter.
    out.push((
        "a ray with no movement along an axis is decided by where it starts",
        ray_meets_box(0.0, 0.0, 0.0, 1.0, 0.0, 100.0, 20.0) == Some(80.0)
            && ray_meets_box(50.0, 0.0, 0.0, 1.0, 0.0, 100.0, 20.0).is_none(),
    ));

    // The blast falls off with distance and stops. A barrel sixty units from
    // another takes 78 of its 20 health, which is why one going up takes the
    // next with it.
    let fall = |gap: f32, radius: f32| -> i32 { (BLAST - (gap - radius).max(0.0)) as i32 };
    out.push((
        "a blast falls off with distance and reaches 128 units",
        fall(0.0, 10.0) == 128 && fall(60.0, 10.0) == 78 && fall(138.0, 10.0) == 0,
    ));

    out
}

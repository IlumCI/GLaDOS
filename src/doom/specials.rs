// Ported from room4doom (MIT, Luke Jones): `gameplay/src/env/specials.rs`,
// the `EV_VerticalDoor` and `P_UseLines` paths.
//   https://github.com/flukejones/room4doom
//
// A linedef's `special` is a number in a table id fixed in 1993 and nobody has
// been free to change since -- 1 is a door you open by walking into it, 4 is
// one that opens when you cross the line, 11 ends the level. The numbers carry
// three things at once: what happens, what triggers it, and whether it can
// happen more than once. `Trigger` below pulls those apart, because the
// alternative is a match arm per number and DOOM has 141 of them.
//
// **What is not here, and is not silently wrong for being absent.** Doors 26,
// 27, 28, 32, 33 and 34 want a key, and there is no inventory yet -- so they
// fall through and stay shut, which is exactly what a locked door does to
// somebody with no key. That is the one absence in this file which is correct
// rather than merely unimplemented.

use super::level::{Level, NONE};
use super::math;
use super::thinker::{DoorKind, Thinkers};

/// How far ahead of the player a Use reaches. DOOM's `USERANGE`.
pub const USE_RANGE: f32 = 64.0;

/// How a special is set off.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    /// Walking across the line.
    Cross,
    /// Pressing Use while facing it.
    Use,
}

/// What a special does, once its trigger and repeatability are stripped off.
#[derive(Clone, Copy)]
enum Action {
    Door(DoorKind),
}

/// What this special is, or nothing if it is one we do not do yet.
///
/// The `bool` is whether it repeats. A once-only special has its number
/// cleared after it fires, which is how DOOM stops a `W1` line triggering
/// twice -- there is no separate "used" flag anywhere in the format.
fn lookup(special: u16, by: Trigger) -> Option<(Action, bool)> {
    use DoorKind::*;
    let (action, trigger, repeats) = match special {
        // Manual doors: the sector is the line's *back*, not a tag.
        1 => (Action::Door(Normal), Trigger::Use, true),
        31 => (Action::Door(Open), Trigger::Use, false),
        117 => (Action::Door(Normal), Trigger::Use, true),
        118 => (Action::Door(Open), Trigger::Use, false),
        // Walk-triggered, tagged.
        2 => (Action::Door(Open), Trigger::Cross, false),
        3 => (Action::Door(Close), Trigger::Cross, false),
        4 => (Action::Door(Normal), Trigger::Cross, false),
        16 => (Action::Door(Close30ThenOpen), Trigger::Cross, false),
        86 => (Action::Door(Open), Trigger::Cross, true),
        75 => (Action::Door(Close), Trigger::Cross, true),
        90 => (Action::Door(Normal), Trigger::Cross, true),
        76 => (Action::Door(Close30ThenOpen), Trigger::Cross, true),
        105 => (Action::Door(Normal), Trigger::Cross, true),
        106 => (Action::Door(Open), Trigger::Cross, true),
        107 => (Action::Door(Close), Trigger::Cross, true),
        108 => (Action::Door(Normal), Trigger::Cross, false),
        109 => (Action::Door(Open), Trigger::Cross, false),
        110 => (Action::Door(Close), Trigger::Cross, false),
        _ => return None,
    };
    if trigger != by {
        return None;
    }
    Some((action, repeats))
}

/// Which sector a manual door operates on.
///
/// The line's **back** sector, and the player must be on the front. That is
/// not a convention a reader can choose: a manual door line is drawn with its
/// front facing the room you stand in, so using it from the other side would
/// open the wall you are standing inside. DOOM checks the side and refuses.
fn manual_sector(lv: &Level, line: usize, px: f32, py: f32) -> Option<usize> {
    let l = lv.linedefs.get(line)?;
    if l.left == NONE {
        return None;
    }
    if side_of(lv, line, px, py)? != 0 {
        return None;
    }
    lv.sidedefs.get(l.left as usize).map(|sd| sd.sector as usize)
}

/// Which side of a line a point is on: 0 front (right), 1 back (left).
///
/// DOOM's `P_PointOnLineSide`, and the sign is the whole of it: front is
/// `cross > 0`, where cross is `(p - v1) x (v2 - v1)`. `Node::side_of` in
/// `level.rs` already spells the same test the same way, and this had it
/// inverted -- which does not fail loudly. It decided the player was standing
/// *behind* the only door on the map, so `EV_VerticalDoor` refused exactly as
/// it should when somebody tries to open a door from inside the wall, and the
/// symptom was a door that ignored every press with nothing logged.
fn side_of(lv: &Level, line: usize, px: f32, py: f32) -> Option<usize> {
    let l = lv.linedefs.get(line)?;
    let a = lv.vertexes.get(l.v1 as usize)?;
    let b = lv.vertexes.get(l.v2 as usize)?;
    let (dx, dy) = ((b.x - a.x) as f32, (b.y - a.y) as f32);
    let cross = (px - a.x as f32) * dy - (py - a.y as f32) * dx;
    Some(if cross > 0.0 { 0 } else { 1 })
}

/// Fire whatever this line does, if anything.
///
/// Answers whether something happened, which the caller wants for two reasons:
/// a Use that hit nothing should keep looking at the next line, and a once-only
/// special is only spent when it actually fired.
pub fn activate(
    lv: &mut Level,
    th: &mut Thinkers,
    line: usize,
    by: Trigger,
    px: f32,
    py: f32,
) -> bool {
    let Some(l) = lv.linedefs.get(line).copied() else { return false };
    let Some((action, repeats)) = lookup(l.special, by) else { return false };

    let mut fired = false;
    match action {
        Action::Door(kind) => {
            if by == Trigger::Use {
                if let Some(sector) = manual_sector(lv, line, px, py) {
                    // A manual door already moving is *reversed* rather than
                    // ignored, which is how you can pull a door back down on
                    // yourself in DOOM. Not modelled yet: it needs the thinker
                    // to be findable from the sector, which `busy` deliberately
                    // does not allow. Re-triggering is simply refused, which
                    // stops the two-thinkers-one-sector fight and loses only
                    // the ability to change your mind mid-open.
                    fired = th.spawn_door(lv, sector, kind);
                }
            } else {
                // Tagged: one line can open several doors at once.
                let sectors: alloc::vec::Vec<u16> = lv.tagged(l.tag as i32).to_vec();
                for s in sectors {
                    if th.spawn_door(lv, s as usize, kind) {
                        fired = true;
                    }
                }
            }
        }
    }

    // Spend a once-only special by clearing its number. There is no "already
    // used" bit in the format, so this *is* the mechanism -- and it is why the
    // clearing has to happen only when the special actually did something.
    if fired && !repeats {
        if let Some(m) = lv.linedefs.get_mut(line) {
            m.special = 0;
        }
    }
    fired
}

/// The line the player is trying to use, and use it.
///
/// DOOM's `P_UseLines` walks the blockmap along a 64-unit ray and takes the
/// first line it meets. This walks every linedef instead, for the reason
/// `blocked` does: the blockmap is not read yet, and on the maps this runs
/// against the scan is affordable. It is the same answer, arrived at more
/// slowly, and the note is here so the eventual blockmap work knows where to
/// look.
pub fn use_lines(lv: &mut Level, th: &mut Thinkers, px: f32, py: f32, angle: f32) -> bool {
    let (ex, ey) = (
        px + math::cos(angle) * USE_RANGE,
        py + math::sin(angle) * USE_RANGE,
    );
    let mut best: Option<(f32, usize)> = None;
    for i in 0..lv.linedefs.len() {
        let l = lv.linedefs[i];
        if l.special == 0 {
            continue;
        }
        let (Some(a), Some(b)) = (
            lv.vertexes.get(l.v1 as usize).copied(),
            lv.vertexes.get(l.v2 as usize).copied(),
        ) else {
            continue;
        };
        let Some(t) = cross_at(
            px, py, ex, ey, a.x as f32, a.y as f32, b.x as f32, b.y as f32,
        ) else {
            continue;
        };
        if best.map(|(bt, _)| t < bt).unwrap_or(true) {
            best = Some((t, i));
        }
    }
    match best {
        Some((_, i)) => activate(lv, th, i, Trigger::Use, px, py),
        None => false,
    }
}

/// How far along the first segment it crosses the second, if it does.
///
/// The standard two-segment intersection. Answers the parameter rather than
/// the point, because the caller wants the *nearest* line and a parameter
/// compares without a square root.
fn cross_at(
    px: f32,
    py: f32,
    qx: f32,
    qy: f32,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
) -> Option<f32> {
    let (rx, ry) = (qx - px, qy - py);
    let (sx, sy) = (bx - ax, by - ay);
    let denom = rx * sy - ry * sx;
    if denom == 0.0 {
        return None;
    }
    let t = ((ax - px) * sy - (ay - py) * sx) / denom;
    let u = ((ax - px) * ry - (ay - py) * rx) / denom;
    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
        Some(t)
    } else {
        None
    }
}

/// Fire the cross-triggered special of every line the player just walked over.
pub fn cross_lines(
    lv: &mut Level,
    th: &mut Thinkers,
    from: (f32, f32),
    to: (f32, f32),
) -> bool {
    let mut any = false;
    for i in 0..lv.linedefs.len() {
        let l = lv.linedefs[i];
        if l.special == 0 {
            continue;
        }
        let (Some(a), Some(b)) = (
            lv.vertexes.get(l.v1 as usize).copied(),
            lv.vertexes.get(l.v2 as usize).copied(),
        ) else {
            continue;
        };
        if cross_at(
            from.0, from.1, to.0, to.1, a.x as f32, a.y as f32, b.x as f32, b.y as f32,
        )
        .is_some()
            && activate(lv, th, i, Trigger::Cross, from.0, from.1)
        {
            any = true;
        }
    }
    any
}

/// What `diag doom` asks of the special table.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out: alloc::vec::Vec<(&'static str, bool)> = alloc::vec::Vec::new();
    out.push((
        "special 1 is a use-door and not a walk-door",
        lookup(1, Trigger::Use).is_some() && lookup(1, Trigger::Cross).is_none(),
    ));
    out.push((
        "special 4 is a walk-door and not a use-door",
        lookup(4, Trigger::Cross).is_some() && lookup(4, Trigger::Use).is_none(),
    ));
    out.push((
        "a DR door repeats and a D1 door does not",
        lookup(1, Trigger::Use).map(|(_, r)| r) == Some(true)
            && lookup(31, Trigger::Use).map(|(_, r)| r) == Some(false),
    ));
    out.push((
        "a key door is not handled, so it stays shut",
        lookup(26, Trigger::Use).is_none() && lookup(32, Trigger::Use).is_none(),
    ));
    out.push(("special 0 is nothing at all", lookup(0, Trigger::Use).is_none()));
    // The ray test, which decides what a Use reaches.
    out.push((
        "a use ray crosses a line in front and not one behind",
        cross_at(0.0, 0.0, 64.0, 0.0, 32.0, -16.0, 32.0, 16.0).is_some()
            && cross_at(0.0, 0.0, 64.0, 0.0, -32.0, -16.0, -32.0, 16.0).is_none(),
    ));
    out.push((
        "and not one beyond its reach",
        cross_at(0.0, 0.0, 64.0, 0.0, 96.0, -16.0, 96.0, 16.0).is_none(),
    ));
    out.push((
        "a parallel line is never crossed",
        cross_at(0.0, 0.0, 64.0, 0.0, 0.0, 8.0, 64.0, 8.0).is_none(),
    ));
    // Which side of a line a point is on, checked because getting it backwards
    // is silent: a door refuses, correctly, on the belief that the player is
    // standing inside the wall it is set into.
    let lv = Level {
        name: super::wad::Name::from_lump(b"TEST    "),
        sector_lines: alloc::vec::Vec::new(),
        tagged: alloc::vec::Vec::new(),
        things: alloc::vec::Vec::new(),
        // A line running north from the origin. Its right -- its front -- is
        // therefore east, which is +x.
        vertexes: alloc::vec![
            super::level::Vertex { x: 0, y: -64 },
            super::level::Vertex { x: 0, y: 64 }
        ],
        linedefs: alloc::vec![super::level::LineDef {
            v1: 0,
            v2: 1,
            flags: 4,
            special: 1,
            tag: 0,
            right: 0,
            left: 1
        }],
        sidedefs: alloc::vec::Vec::new(),
        segs: alloc::vec::Vec::new(),
        subsectors: alloc::vec::Vec::new(),
        nodes: alloc::vec::Vec::new(),
        sectors: alloc::vec::Vec::new(),
    };
    out.push((
        "east of a northward line is its front, and west is its back",
        side_of(&lv, 0, 25.0, 0.0) == Some(0) && side_of(&lv, 0, -25.0, 0.0) == Some(1),
    ));
    out
}

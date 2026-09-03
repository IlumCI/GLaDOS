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
use super::thinker::{DoorKind, FLOOR_SPEED, PlatKind, Thinkers};

/// How far ahead of the player a Use reaches. DOOM's `USERANGE`.
pub const USE_RANGE: f32 = 64.0;

/// How a special is set off.
///
/// A switch and a manual door are both "press Use at it", and DOOM tells them
/// apart by what they operate on rather than by how they are triggered: a
/// manual door acts on the sector behind the line, a switch on a *tag* and
/// swaps its own texture to show it was pressed. Both arrive here as `Use`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    /// Walking across the line.
    Cross,
    /// Pressing Use while facing it.
    Use,
}

/// Where a floor is being sent.
///
/// Named rather than numeric because every one of these is a *query* against
/// the neighbouring sectors, resolved when the special fires. A stair that
/// stored an absolute height would be wrong the moment anything around it
/// moved.
#[derive(Clone, Copy)]
enum FloorTo {
    /// The highest floor around it. Named for what it *queries*, not for
    /// which way the floor then moves: DOOM calls special 19 "lower floor to
    /// highest floor", and a sector below its neighbours would rise to the
    /// same destination. `spawn_floor` takes the direction from comparing the
    /// destination to where the floor is now, so the query and the direction
    /// stay separate and neither has to know about the other.
    HighestNeighbour,
    /// Down to the lowest floor around it.
    LowestNeighbour,
    /// Up to the lowest ceiling around it, or its own, whichever is lower.
    LowestNeighbourCeiling,
    /// Up to the next floor above the current one. What a staircase is built
    /// from.
    NextHighest,
    /// Up by a fixed amount.
    By(i16),
}

/// What a special does, once its trigger and repeatability are stripped off.
#[derive(Clone, Copy)]
enum Action {
    Door(DoorKind),
    Plat(PlatKind),
    Floor(FloorTo, i16),
    /// End the level.
    Exit,
}

/// What this special is, or nothing if it is one we do not do yet.
///
/// The `bool` is whether it repeats. A once-only special has its number
/// cleared after it fires, which is how DOOM stops a `W1` line triggering
/// twice -- there is no separate "used" flag anywhere in the format.
fn lookup(special: u16, by: Trigger) -> Option<(Action, bool)> {
    use DoorKind::*;
    use PlatKind::*;
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

        // Switches. Tagged, and they swap their own texture when pressed.
        29 => (Action::Door(Normal), Trigger::Use, false),
        63 => (Action::Door(Normal), Trigger::Use, true),
        103 => (Action::Door(Open), Trigger::Use, false),
        61 => (Action::Door(Open), Trigger::Use, true),
        50 => (Action::Door(Close), Trigger::Use, false),
        42 => (Action::Door(Close), Trigger::Use, true),

        // Lifts.
        10 => (Action::Plat(DownWaitUpStay), Trigger::Cross, false),
        88 => (Action::Plat(DownWaitUpStay), Trigger::Cross, true),
        21 => (Action::Plat(DownWaitUpStay), Trigger::Use, false),
        62 => (Action::Plat(DownWaitUpStay), Trigger::Use, true),
        53 => (Action::Plat(PerpetualRaise), Trigger::Cross, false),
        87 => (Action::Plat(PerpetualRaise), Trigger::Cross, true),

        // Floors.
        23 => (Action::Floor(FloorTo::LowestNeighbour, FLOOR_SPEED), Trigger::Use, false),
        60 => (Action::Floor(FloorTo::LowestNeighbour, FLOOR_SPEED), Trigger::Use, true),
        38 => (Action::Floor(FloorTo::LowestNeighbour, FLOOR_SPEED), Trigger::Cross, false),
        82 => (Action::Floor(FloorTo::LowestNeighbour, FLOOR_SPEED), Trigger::Cross, true),
        19 => (Action::Floor(FloorTo::HighestNeighbour, FLOOR_SPEED), Trigger::Cross, false),
        102 => (Action::Floor(FloorTo::HighestNeighbour, FLOOR_SPEED), Trigger::Use, false),
        45 => (Action::Floor(FloorTo::HighestNeighbour, FLOOR_SPEED), Trigger::Cross, true),
        70 => (Action::Floor(FloorTo::HighestNeighbour, FLOOR_SPEED * 4), Trigger::Cross, true),
        5 => (Action::Floor(FloorTo::LowestNeighbourCeiling, FLOOR_SPEED), Trigger::Cross, false),
        91 => (Action::Floor(FloorTo::LowestNeighbourCeiling, FLOOR_SPEED), Trigger::Cross, true),
        18 => (Action::Floor(FloorTo::NextHighest, FLOOR_SPEED), Trigger::Use, false),
        69 => (Action::Floor(FloorTo::NextHighest, FLOOR_SPEED), Trigger::Use, true),
        119 => (Action::Floor(FloorTo::NextHighest, FLOOR_SPEED), Trigger::Cross, false),
        128 => (Action::Floor(FloorTo::NextHighest, FLOOR_SPEED), Trigger::Cross, true),
        58 => (Action::Floor(FloorTo::By(24), FLOOR_SPEED), Trigger::Cross, false),
        92 => (Action::Floor(FloorTo::By(24), FLOOR_SPEED), Trigger::Cross, true),

        // The way out. 52 is the ordinary exit switch; 11 the one you walk
        // into; 51 and 124 go to the secret level, which is the same thing
        // here because there is no level sequence yet.
        11 => (Action::Exit, Trigger::Use, false),
        51 => (Action::Exit, Trigger::Use, false),
        52 => (Action::Exit, Trigger::Cross, false),
        124 => (Action::Exit, Trigger::Cross, false),

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
    // Which sectors this operates on. A tag names them; no tag on a Use-line
    // means the sector behind it. That single rule is what separates a manual
    // door from a switch, and it is read off the line rather than from the
    // special's number.
    let targets: alloc::vec::Vec<u16> = lv.tagged(l.tag as i32).to_vec();

    match action {
        Action::Exit => {
            lv.exited = true;
            fired = true;
        }
        Action::Plat(kind) => {
            for s in targets.iter() {
                if th.spawn_plat(lv, *s as usize, kind) {
                    fired = true;
                }
            }
        }
        Action::Floor(to, speed) => {
            for s in targets.iter() {
                let i = *s as usize;
                let Some(sec) = lv.sectors.get(i).copied() else { continue };
                let dest = match to {
                    FloorTo::LowestNeighbour => lv.lowest_neighbour_floor(i),
                    FloorTo::HighestNeighbour => lv.highest_neighbour_floor(i),
                    FloorTo::LowestNeighbourCeiling => {
                        lv.lowest_neighbour_ceiling(i).min(sec.ceiling)
                    }
                    FloorTo::NextHighest => lv.next_highest_neighbour_floor(i, sec.floor),
                    FloorTo::By(n) => sec.floor + n,
                };
                if th.spawn_floor(i, dest, speed, sec.floor) {
                    fired = true;
                }
            }
        }
        Action::Door(kind) => {
            if by == Trigger::Use && targets.is_empty() {
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
                for s in targets.iter() {
                    if th.spawn_door(lv, *s as usize, kind) {
                        fired = true;
                    }
                }
            }
        }
    }

    // A switch shows that it was pressed by swapping its own texture. Without
    // it a switch you have thrown looks exactly like one you have not, which
    // on a map with several is the difference between knowing where you are
    // and not.
    if fired && by == Trigger::Use && !targets.is_empty() {
        flip_switch(lv, line);
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

/// Swap a switch's texture between its off and on faces.
///
/// DOOM ships a table pairing them (`SW1BRN1` with `SW2BRN1`, and so on), but
/// every pair in it differs in exactly one character: the `1` or `2` after
/// `SW`. Following the rule rather than carrying the table means a WAD with
/// its own switches works without being listed, which is the same trade
/// `pic::classify` makes about flats -- and if a WAD ever breaks the
/// convention the switch simply does not change picture, which is cosmetic.
///
/// The face that carries the switch is whichever of the three textures is not
/// blank, checked middle-first: a switch on a one-sided wall is a middle, one
/// beside a step is a lower, and one over a doorway is an upper.
fn flip_switch(lv: &mut Level, line: usize) {
    let Some(l) = lv.linedefs.get(line).copied() else { return };
    let Some(sd) = lv.sidedefs.get_mut(l.right as usize) else { return };
    for slot in [&mut sd.middle, &mut sd.upper, &mut sd.lower] {
        let n = slot.as_str();
        if n.len() < 3 || !n.starts_with("SW") {
            continue;
        }
        let flipped = match n.as_bytes()[2] {
            b'1' => '2',
            b'2' => '1',
            _ => continue,
        };
        let mut buf = [b' '; 8];
        let b = n.as_bytes();
        buf[..b.len().min(8)].copy_from_slice(&b[..b.len().min(8)]);
        buf[2] = flipped as u8;
        *slot = super::wad::Name::from_lump(&buf);
        return;
    }
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

/// Whether this special number is one we act on, under either trigger.
///
/// For a caller that wants to report what a map asks for against what is
/// implemented -- which is the only honest measure of how far this has got.
/// Counting the specials we *do* implement says nothing; counting the ones a
/// real map uses that we do not is the number.
pub fn handled(special: u16) -> bool {
    lookup(special, Trigger::Use).is_some() || lookup(special, Trigger::Cross).is_some()
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
        exited: false,
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
        "a lift and a floor are told apart from a door by their number",
        matches!(lookup(21, Trigger::Use), Some((Action::Plat(_), _)))
            && matches!(lookup(23, Trigger::Use), Some((Action::Floor(_, _), _)))
            && matches!(lookup(1, Trigger::Use), Some((Action::Door(_), _))),
    ));
    out.push((
        "the exit is reachable by walking into it and by pressing it",
        matches!(lookup(52, Trigger::Cross), Some((Action::Exit, _)))
            && matches!(lookup(11, Trigger::Use), Some((Action::Exit, _))),
    ));
    out.push((
        "an SR switch repeats where its S1 twin does not",
        lookup(63, Trigger::Use).map(|(_, r)| r) == Some(true)
            && lookup(29, Trigger::Use).map(|(_, r)| r) == Some(false),
    ));
    out.push((
        "east of a northward line is its front, and west is its back",
        side_of(&lv, 0, 25.0, 0.0) == Some(0) && side_of(&lv, 0, -25.0, 0.0) == Some(1),
    ));
    out
}

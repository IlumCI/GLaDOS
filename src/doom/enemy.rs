//! Monsters: noticing, chasing, and attacking.
//!
//! Ported from room4doom's `gameplay/src/thing/enemy.rs` and the chase-movement
//! half of `thing/movement.rs`.
//!
//! ### A monster walks on a compass, not a heading
//!
//! DOOM does not steer. A monster picks one of **eight** directions, walks in
//! it for a random number of tics, and picks again when it is blocked or the
//! count runs out. `new_chase_dir` tries the direct route first, then the two
//! cardinal components, then the way it was already going, then every
//! remaining direction in a randomly chosen order -- and refuses to turn
//! straight around unless nothing else works.
//!
//! That is why DOOM's monsters move the way they do: they get stuck on
//! doorframes, they take corners in two moves, and they occasionally wander
//! off. Steering them toward the player with a heading would look smoother and
//! would not be DOOM.
//!
//! ### The one target is the player
//!
//! Upstream's `target` is a pointer, because a monster hit by another monster
//! turns on *it* -- DOOM's infighting. Here it is a `bool`: there is one
//! player and monsters cannot hurt each other, so "has a target" is the whole
//! of the information. Infighting needs a hitscan that can hit a monster fired
//! *by* a monster, which is a change to `shoot::fire` and not to this file.
//!
//! ### What attacks, and what only chases
//!
//! **Hitscan attackers work**: the zombieman, the shotgun guy and the chaingun
//! guy shoot, with DOOM's own spread and damage rolls. **Melee attackers
//! work**: the demon and the imp's claw.
//!
//! **Projectile attackers do not throw.** An imp's fireball, a cacodemon's
//! ball, a baron's plasma are all `P_SpawnMissile` -- a thing with momentum
//! that flies, collides and explodes. Nothing in this port moves under its own
//! power yet. Those monsters chase and melee correctly and never throw
//! anything, which is a monster that is easier than it should be rather than
//! one that behaves strangely.
//!
//! ### Sound, and the one place its absence changes behaviour
//!
//! There is no audio driver, which is ordinarily just silence. Here it is
//! not: DOOM wakes monsters with `P_NoiseAlert`, flooding the sound of a shot
//! through connected sectors until it has crossed two sound-blocking lines.
//! A monster with its back turned wakes because it *heard* you.
//!
//! So the flood is kept and the sound is dropped: `alert` wakes monsters in
//! the player's own sector and the ones adjoining it, which is a one-hop
//! version of the same idea. Without it a monster facing away is deaf, and
//! firing a gun in a crowded room wakes nobody.

use alloc::vec::Vec;

use super::info;
use super::level::Level;
use super::math;
use super::rng;
use super::thing::{Fired, Objs};

/// How close a melee attack reaches. DOOM's `MELEERANGE`.
pub const MELEE_RANGE: f32 = 64.0;

/// How far a monster's hitscan carries. DOOM's `MISSILERANGE`.
pub const MISSILE_RANGE: f32 = 32.0 * 64.0;

/// The tallest ledge a monster will walk down.
///
/// The same 24 units the player can climb, used the other way up. Without it
/// monsters walk off balconies, which on a map with any height to it is most
/// of them.
pub const MAX_DROP: f32 = 24.0;

/// Not going anywhere.
pub const NO_DIR: u8 = 8;

/// The eight directions, in DOOM's order: east, then anticlockwise by 45.
///
/// The diagonal component is **0.717**, not `1/sqrt(2)`. DOOM's table holds
/// 47000 of 65536, which is 0.71716 -- a hair over the true value, so a
/// monster moving diagonally covers slightly more ground than one moving
/// straight. Rounding it to the honest number would be a different game by a
/// fraction of a percent, and there is no reason to prefer that.
const DIAG: f32 = 47000.0 / 65536.0;
const XSPEED: [f32; 8] = [1.0, DIAG, 0.0, -DIAG, -1.0, -DIAG, 0.0, DIAG];
const YSPEED: [f32; 8] = [0.0, DIAG, 1.0, DIAG, 0.0, -DIAG, -1.0, -DIAG];

/// Which direction is straight back the way it came.
const OPPOSITE: [u8; 9] = [4, 5, 6, 7, 0, 1, 2, 3, NO_DIR];

/// North-west, north-east, south-west, south-east, indexed by the signs of the
/// two deltas.
const DIAGONALS: [u8; 4] = [3, 1, 5, 7];

/// Where the player is, copied.
///
/// A copy and not a borrow, because everything here walks the object list
/// mutably and the player is not in it. Copying four numbers once per monster
/// per tic is cheaper than the alternative and much cheaper than being wrong
/// about which of them changed.
#[derive(Clone, Copy)]
pub struct Quarry {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub alive: bool,
}

/// Whether a monster of this radius and height can stand here.
///
/// `play::blocked_for` decides the walls; this adds the two rules that are
/// about a *monster* rather than about the map. It will not walk into another
/// solid thing, and it will not walk off a ledge taller than it can climb.
fn can_stand(
    lv: &Level,
    objs: &Objs,
    me: usize,
    x: f32,
    y: f32,
    z: f32,
    radius: f32,
    height: f32,
) -> bool {
    if super::play::blocked_for(lv, x, y, z, radius, height) {
        return false;
    }
    // A drop it cannot climb back out of. DOOM asks this of `floorz` against
    // `dropoffz`; with one floor per sector the destination's floor is the
    // same question.
    match lv.sector_at(x as i32, y as i32) {
        None => return false,
        Some(sec) => {
            if z - sec.floor as f32 > MAX_DROP {
                return false;
            }
        }
    }
    // Another solid thing. The square again, and the same square the pickup
    // and the shot use -- DOOM compares bounding boxes and so does this.
    for (j, o) in objs.list.iter().enumerate() {
        if j == me || o.flags & info::MF_SOLID == 0 {
            continue;
        }
        let reach = radius + o.radius();
        if (x - o.x).abs() < reach && (y - o.y).abs() < reach {
            return false;
        }
    }
    true
}

/// One step in the direction it is facing. DOOM's `P_Move`.
fn do_move(lv: &Level, objs: &mut Objs, i: usize, q: &Quarry) -> bool {
    let (dir, speed, radius, height) = {
        let o = &objs.list[i];
        (o.movedir, o.speed(), o.radius(), o.height())
    };
    if dir >= NO_DIR {
        return false;
    }
    let (x, y, z) = {
        let o = &objs.list[i];
        (
            o.x + XSPEED[dir as usize] * speed,
            o.y + YSPEED[dir as usize] * speed,
            o.z,
        )
    };
    // The player is solid too, and is not in the object list.
    if q.alive {
        let reach = radius + 16.0;
        if (x - q.x).abs() < reach && (y - q.y).abs() < reach {
            return false;
        }
    }
    if !can_stand(lv, objs, i, x, y, z, radius, height) {
        return false;
    }
    let o = &mut objs.list[i];
    o.x = x;
    o.y = y;
    if let Some(sec) = lv.sector_at(x as i32, y as i32) {
        o.sector = lv.sector_index_at(x as i32, y as i32).unwrap_or(o.sector);
        let _ = sec;
    }
    true
}

/// Move, and if it worked commit to the direction for a while.
fn try_walk(lv: &Level, objs: &mut Objs, i: usize, q: &Quarry) -> bool {
    if !do_move(lv, objs, i, q) {
        return false;
    }
    objs.list[i].movecount = rng::p_random() & 15;
    true
}

/// Choose a new direction. DOOM's `P_NewChaseDir`, and the order is the whole
/// of how a monster behaves.
fn new_chase_dir(lv: &Level, objs: &mut Objs, i: usize, q: &Quarry) {
    let (ox, oy, old) = {
        let o = &objs.list[i];
        (o.x, o.y, o.movedir)
    };
    let turnaround = OPPOSITE[old.min(NO_DIR) as usize];
    let (dx, dy) = (q.x - ox, q.y - oy);

    // The two cardinal components, each dead only within ten units.
    let mut d1 = if dx > 10.0 {
        0 // east
    } else if dx < -10.0 {
        4 // west
    } else {
        NO_DIR
    };
    let mut d2 = if dy < -10.0 {
        6 // south
    } else if dy > 10.0 {
        2 // north
    } else {
        NO_DIR
    };

    // Straight at it, if both components want to move.
    if d1 != NO_DIR && d2 != NO_DIR {
        let k = ((dy < 0.0) as usize) << 1 | (dx > 0.0) as usize;
        objs.list[i].movedir = DIAGONALS[k];
        if objs.list[i].movedir != turnaround && try_walk(lv, objs, i, q) {
            return;
        }
    }

    // Otherwise try the longer component first -- usually. The draw happens
    // whatever the deltas are, because it costs a byte either way and DOOM
    // spends it here.
    if rng::p_random() > 200 || dy.abs() > dx.abs() {
        core::mem::swap(&mut d1, &mut d2);
    }
    if d1 == turnaround {
        d1 = NO_DIR;
    }
    if d2 == turnaround {
        d2 = NO_DIR;
    }
    for d in [d1, d2] {
        if d != NO_DIR {
            objs.list[i].movedir = d;
            if try_walk(lv, objs, i, q) {
                return;
            }
        }
    }
    // Carry on the way it was going.
    if old != NO_DIR {
        objs.list[i].movedir = old;
        if try_walk(lv, objs, i, q) {
            return;
        }
    }
    // Everything else, from one end or the other. Which end is a coin, so a
    // monster in a corner does not always squirm the same way out of it.
    let forwards = rng::p_random() & 1 != 0;
    for k in 0..8u8 {
        let d = if forwards { k } else { 7 - k };
        if d == turnaround {
            continue;
        }
        objs.list[i].movedir = d;
        if try_walk(lv, objs, i, q) {
            return;
        }
    }
    // Even backwards.
    if turnaround != NO_DIR {
        objs.list[i].movedir = turnaround;
        if try_walk(lv, objs, i, q) {
            return;
        }
    }
    objs.list[i].movedir = NO_DIR;
}

/// Whether this monster can see the player.
fn sees(lv: &Level, objs: &Objs, i: usize, q: &Quarry) -> bool {
    let o = &objs.list[i];
    q.alive && super::shoot::visible(lv, o.x, o.y, q.x, q.y)
}

/// DOOM's `P_AproxDistance`: the long side plus half the short one.
///
/// Not the Euclidean distance, and the difference is not rounding -- it
/// overestimates by up to 12% on the diagonal. Every range check in DOOM uses
/// it, so a monster's melee reach is genuinely shorter diagonally than it is
/// straight on. Replacing it with a square root would be tidier and would
/// change when monsters decide to attack.
fn approx_dist(dx: f32, dy: f32) -> f32 {
    let (dx, dy) = (dx.abs(), dy.abs());
    if dx < dy {
        dy + dx / 2.0
    } else {
        dx + dy / 2.0
    }
}

/// Close enough to claw. DOOM's `P_CheckMeleeRange`.
fn in_melee_range(lv: &Level, objs: &Objs, i: usize, q: &Quarry) -> bool {
    if !q.alive {
        return false;
    }
    let o = &objs.list[i];
    // Against the *player's* radius, and DOOM subtracts 20 from the range as
    // well -- so the reach is 64 - 20 + 16, not 64.
    let dist = approx_dist(q.x - o.x, q.y - o.y);
    if dist >= 16.0 + (MELEE_RANGE - 20.0) {
        return false;
    }
    sees(lv, objs, i, q)
}

/// Whether to take a shot this tic. DOOM's `P_CheckMissileRange`, and the
/// probability rising as the distance falls is the whole of it.
fn wants_to_shoot(lv: &Level, objs: &mut Objs, i: usize, q: &Quarry) -> bool {
    if !sees(lv, objs, i, q) {
        return false;
    }
    let o = &mut objs.list[i];
    // Just been hit: fight back, whatever the distance.
    if o.flags & info::MF_JUSTHIT != 0 {
        o.flags &= !info::MF_JUSTHIT;
        return true;
    }
    if o.reaction != 0 {
        return false;
    }
    let melee = o.row().map(|k| k.melee).unwrap_or(0);
    let mut dist = approx_dist(o.x - q.x, o.y - q.y) - 64.0;
    if melee == 0 {
        // Nothing to do up close, so shoot from further out.
        dist -= 128.0;
    }
    let mut d = dist.max(0.0) as i32;
    if d > 200 {
        d = 200;
    }
    // The nearer it is the likelier the draw clears it.
    rng::p_random() >= d
}

/// Turn to face the player, snapped to nothing -- this is an exact heading.
fn face_target(objs: &mut Objs, i: usize, q: &Quarry) {
    let o = &mut objs.list[i];
    o.angle = math::atan2(q.y - o.y, q.x - o.x);
}

/// One bullet from a monster, with DOOM's spread and damage.
///
/// `(P_Random() - P_Random()) << 20` is a spread of about +-5.6 degrees:
/// the difference of two draws is roughly +-255, and 255 << 20 of a 32-bit
/// turn is 5.6 degrees. Damage is `((P_Random() % 5) + 1) * 3`, so 3 to 15.
fn monster_bullet(lv: &Level, objs: &mut Objs, i: usize, q: &Quarry) -> i32 {
    let base = objs.list[i].angle;
    let spread = (rng::p_random_signed() as f32 / 255.0) * math::deg_to_rad(6);
    let damage = ((rng::p_random() % 5) + 1) * 3;
    let angle = base + spread;
    // Does the shot reach the player before a wall does? The monster's aim is
    // not modelled as a trace against the player's box -- it is a trace
    // against the *world*, and the player is hit when nothing solid stands
    // between them at the sprayed angle.
    let (ox, oy) = (objs.list[i].x, objs.list[i].y);
    let (dx, dy) = (math::cos(angle), math::sin(angle));
    let wall = super::shoot::wall_range(lv, ox, oy, dx, dy);
    let to_player = approx_dist(q.x - ox, q.y - oy);
    if to_player > wall || to_player > MISSILE_RANGE {
        return 0;
    }
    // Within the spread, at the distance. The cone widens with range, which is
    // why a zombieman across a courtyard misses and one in your face does not.
    let miss = (to_player * spread).abs();
    if miss > 16.0 {
        return 0;
    }
    damage
}

/// Wake every monster near enough to have heard a shot.
///
/// DOOM's `P_NoiseAlert` floods the sound through connected sectors, stopping
/// after it has crossed two sound-blocking lines. This is one hop: the
/// player's own sector and everything adjoining it. Without any of it a
/// monster facing away from you is deaf, and firing a gun in a crowded room
/// wakes nobody -- which is the one place having no audio driver changes
/// behaviour rather than merely being quiet.
pub fn alert(lv: &Level, objs: &mut Objs, px: f32, py: f32) {
    let Some(here) = lv.sector_index_at(px as i32, py as i32) else { return };
    // The sectors one line away. `neighbours` answers `&Sector` because every
    // other caller wants a *height*; the flood wants an identity, so the walk
    // is repeated here over indices rather than widening that one.
    let mut near: alloc::vec::Vec<usize> = alloc::vec::Vec::new();
    for &line in lv.sector_lines.get(here).map(|v| v.as_slice()).unwrap_or(&[]) {
        if let Some(other) = lv.across(line, here) {
            if !near.contains(&other) {
                near.push(other);
            }
        }
    }
    for o in objs.list.iter_mut() {
        if o.flags & info::MF_COUNTKILL == 0 || o.health <= 0 {
            continue;
        }
        if o.sector == here || near.contains(&o.sector) {
            o.target = true;
        }
    }
}

/// `A_Look`: stand still until the player is in view.
fn look(lv: &Level, objs: &mut Objs, i: usize, q: &Quarry, fired: &mut Vec<u8>) {
    objs.list[i].threshold = 0;
    // Woken by noise already, or able to see for itself. An ambushing monster
    // -- one placed with the ambush flag -- is deaf and must see you.
    let woken = objs.list[i].target && objs.list[i].flags & info::MF_AMBUSH == 0;
    if !woken && !sees(lv, objs, i, q) {
        return;
    }
    // Half the world is behind it. DOOM lets a monster notice anything within
    // melee range regardless, which is what stops you standing behind one
    // forever.
    if !woken {
        let o = &objs.list[i];
        let bearing = math::atan2(q.y - o.y, q.x - o.x);
        let mut off = bearing - o.angle;
        while off > math::PI {
            off -= math::TAU;
        }
        while off < -math::PI {
            off += math::TAU;
        }
        if off.abs() > math::PI / 2.0 && approx_dist(q.x - o.x, q.y - o.y) > MELEE_RANGE {
            return;
        }
    }
    objs.list[i].target = true;
    let see = objs.list[i].row().map(|k| k.see).unwrap_or(0);
    if see != 0 {
        objs.list[i].set_state(see, fired);
    }
}

/// `A_Chase`: the whole of a monster's behaviour once it has noticed you.
///
/// Answers the damage to deal to the player, which is always zero -- chasing
/// does not hit. The attacks are separate actions further along the chain.
fn chase(lv: &Level, objs: &mut Objs, i: usize, q: &Quarry, fired: &mut Vec<u8>) {
    {
        let o = &mut objs.list[i];
        if o.reaction > 0 {
            o.reaction -= 1;
        }
        if o.threshold > 0 {
            o.threshold -= 1;
        }
    }

    // Turn toward the direction it is walking, one 45-degree step per tic.
    // This is why a monster's sprite lags its movement round a corner.
    {
        let o = &mut objs.list[i];
        if o.movedir < NO_DIR {
            let want = o.movedir as f32 * (math::TAU / 8.0);
            let mut off = want - o.angle;
            while off > math::PI {
                off -= math::TAU;
            }
            while off < -math::PI {
                off += math::TAU;
            }
            let step = math::TAU / 8.0;
            o.angle += off.clamp(-step, step);
        }
    }

    // Lost the player, or never had them.
    if !objs.list[i].target || !q.alive {
        objs.list[i].target = false;
        let spawn = objs.list[i].row().map(|k| k.spawn).unwrap_or(0);
        if spawn != 0 {
            objs.list[i].set_state(spawn, fired);
        }
        return;
    }

    // Just attacked: take a step before deciding again, so a monster does not
    // stand still emptying a clip into you.
    if objs.list[i].flags & info::MF_JUSTATTACKED != 0 {
        objs.list[i].flags &= !info::MF_JUSTATTACKED;
        new_chase_dir(lv, objs, i, q);
        return;
    }

    // Close enough to claw?
    let melee = objs.list[i].row().map(|k| k.melee).unwrap_or(0);
    if melee != 0 && in_melee_range(lv, objs, i, q) {
        objs.list[i].set_state(melee, fired);
        return;
    }

    // Far enough to shoot? `movecount` gates it, which is what stops a
    // monster firing on the tic it starts moving.
    let missile = objs.list[i].row().map(|k| k.missile).unwrap_or(0);
    if missile != 0 && objs.list[i].movecount == 0 && wants_to_shoot(lv, objs, i, q) {
        objs.list[i].set_state(missile, fired);
        objs.list[i].flags |= info::MF_JUSTATTACKED;
        return;
    }

    // Otherwise walk.
    objs.list[i].movecount -= 1;
    if objs.list[i].movecount < 0 || !do_move(lv, objs, i, q) {
        new_chase_dir(lv, objs, i, q);
    }
}

/// Run one action for one object. Answers the damage to deal to the player.
///
/// The dispatcher, and it is where every monster action arrives. Anything not
/// listed does nothing, deliberately: an action that half-worked would be much
/// harder to notice than one that did not run.
pub fn act(
    a: u8,
    lv: &Level,
    objs: &mut Objs,
    i: usize,
    q: &Quarry,
    fired: &mut Vec<u8>,
) -> i32 {
    if i >= objs.list.len() {
        return 0;
    }
    match a {
        info::A_LOOK => {
            look(lv, objs, i, q, fired);
            0
        }
        info::A_CHASE => {
            chase(lv, objs, i, q, fired);
            0
        }
        info::A_FACETARGET => {
            if objs.list[i].target {
                face_target(objs, i, q);
            }
            0
        }
        // The hitscan attackers. One bullet, three, and one respectively --
        // the chaingun guy's `A_CPosAttack` is one shot per frame of a
        // two-frame loop, which is where its rate comes from.
        info::A_POSATTACK | info::A_CPOSATTACK => {
            if !objs.list[i].target {
                return 0;
            }
            face_target(objs, i, q);
            monster_bullet(lv, objs, i, q)
        }
        info::A_SPOSATTACK => {
            if !objs.list[i].target {
                return 0;
            }
            face_target(objs, i, q);
            let mut total = 0;
            for _ in 0..3 {
                total += monster_bullet(lv, objs, i, q);
            }
            total
        }
        // Melee. The imp's claw and the demon's bite differ only in the roll.
        info::A_TROOPATTACK | info::A_SARGATTACK | info::A_HEADATTACK
        | info::A_BRUISATTACK => {
            if !objs.list[i].target {
                return 0;
            }
            face_target(objs, i, q);
            if !in_melee_range(lv, objs, i, q) {
                // Out of reach. A missile-capable monster would throw here;
                // see the note at the top about why none of them do.
                return 0;
            }
            let roll = rng::p_random() % 8 + 1;
            match a {
                info::A_SARGATTACK => roll * 4,
                info::A_HEADATTACK => roll * 6,
                info::A_BRUISATTACK => roll * 10,
                _ => roll * 3,
            }
        }
        // A corpse stops blocking. Without this you cannot walk over what you
        // have killed, which in a corridor is a wall you made yourself.
        info::A_FALL => {
            objs.list[i].flags &= !info::MF_SOLID;
            0
        }
        _ => 0,
    }
}

/// What `diag doom` asks of the movement tables.
///
/// The tables are the part that is silently wrong when it is wrong: a monster
/// with a mistyped direction still walks, still chases, and simply never takes
/// the route you would expect.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    // Every direction has an opposite, and applying it twice comes home.
    out.push((
        "each direction's opposite is its own opposite",
        (0..8u8).all(|d| OPPOSITE[OPPOSITE[d as usize] as usize] == d)
            && OPPOSITE[NO_DIR as usize] == NO_DIR,
    ));

    // East is +x, north is +y, and the opposite of east is west. Getting the
    // sign of one axis wrong gives monsters that walk away from you, which
    // looks like fear rather than like a table.
    out.push((
        "east is +x and north is +y",
        XSPEED[0] == 1.0
            && YSPEED[0] == 0.0
            && XSPEED[2] == 0.0
            && YSPEED[2] == 1.0
            && XSPEED[4] == -1.0
            && YSPEED[6] == -1.0,
    ));

    // Every diagonal moves on both axes, and every cardinal on exactly one.
    out.push((
        "a diagonal moves on both axes and a cardinal on one",
        [1usize, 3, 5, 7].iter().all(|d| XSPEED[*d] != 0.0 && YSPEED[*d] != 0.0)
            && [0usize, 2, 4, 6]
                .iter()
                .all(|d| (XSPEED[*d] == 0.0) != (YSPEED[*d] == 0.0)),
    ));

    // The diagonal step is DOOM's 47000/65536 and not 1/sqrt(2). Worth an
    // assertion because the honest value is the one somebody would "fix" it to.
    out.push((
        "the diagonal step is DOOM's 0.717 and not 0.707",
        (DIAG - 0.717_163).abs() < 1e-5,
    ));

    // The approximate distance overestimates on the diagonal and is exact on
    // an axis. That is what makes a monster's reach shorter cornerwise.
    out.push((
        "the approximate distance is exact on an axis and long on the diagonal",
        (approx_dist(100.0, 0.0) - 100.0).abs() < 1e-3
            && (approx_dist(0.0, -100.0) - 100.0).abs() < 1e-3
            && (approx_dist(100.0, 100.0) - 150.0).abs() < 1e-3,
    ));

    // The diagonal picked from the signs of the deltas. North-east is up and
    // to the right, which is dy > 0 and dx > 0.
    let pick = |dx: f32, dy: f32| DIAGONALS[((dy < 0.0) as usize) << 1 | (dx > 0.0) as usize];
    out.push((
        "the direct route is chosen from the signs of the deltas",
        pick(1.0, 1.0) == 1 && pick(-1.0, 1.0) == 3 && pick(-1.0, -1.0) == 5 && pick(1.0, -1.0) == 7,
    ));

    out
}

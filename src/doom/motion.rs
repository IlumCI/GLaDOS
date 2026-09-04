//! Things that move under their own power.
//!
//! Ported from room4doom's `gameplay/src/thing/movement.rs`. Nothing in this
//! port carried momentum before it: a monster walks by having its position
//! set, a barrel never moves, and the player is teleported by their velocity
//! each tic. A missile cannot be written that way, so this is the substrate
//! underneath one.
//!
//! ### It lives here and not in `thing.rs`
//!
//! Moving needs the map -- the floor under a thing, the walls it can hit --
//! and `thing.rs` deliberately knows nothing about maps, which is what lets
//! its whole state machine be asserted at boot without one. So the state
//! machine stays there and the movement is here, and `play::tic` runs this
//! first.
//!
//! ### The move is stepped, and that *is* the collision detection
//!
//! A rocket travels 20 units a tic and a wall can be thinner than that, so
//! adding the momentum in one go would put it through the other side with
//! nothing between the two positions ever tested. DOOM halves the move until
//! it fits under `MAXMOVE/2` and tries each piece in turn. Porting the
//! addition without the loop gives a game that works everywhere except where
//! it matters.
//!
//! ### What friction is for
//!
//! Applied only on the floor and never to a missile -- a projectile that slowed
//! down would stop in mid-air. `STOPSPEED` is the other half: without a
//! threshold, a multiply by 0.90625 approaches zero and never reaches it, so
//! anything that ever moved would drift forever at a speed too small to see
//! and too large to stop.

use alloc::vec::Vec;

use super::info;
use super::level::Level;
use super::thing::Objs;

/// How much a thing falls per tic, per tic. DOOM's `GRAVITY`.
pub const GRAVITY: f32 = 1.0;

/// The most a thing may move in one tic before the step loop halves it.
pub const MAXMOVE: f32 = 30.0;

/// What a tic on the ground takes off. DOOM's `FRICTION`, 0xE800 of 65536.
pub const FRICTION: f32 = 0xE800 as f32 / 65536.0;

/// Below this, a slide is over. DOOM's `STOPSPEED`, 0x1000 of 65536.
pub const STOPSPEED: f32 = 0x1000 as f32 / 65536.0;

/// What a thing hit, if anything.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Hit {
    /// It moved.
    None,
    /// A wall, or a gap it does not fit through.
    Wall,
    /// Another thing, by index.
    Thing(usize),
}

/// Whether this thing can occupy this spot, and what stops it if not.
///
/// A **missile** asks a different question from a walker and the difference is
/// the whole of what makes one work: it ignores drop-offs, it ignores whether
/// something is solid, and it is stopped by anything *shootable* -- which is
/// how it hits a monster rather than sliding around one.
pub fn obstacle(lv: &Level, objs: &Objs, me: usize, x: f32, y: f32) -> Hit {
    let (z, radius, height, is_missile) = {
        let o = &objs.list[me];
        (o.z, o.radius(), o.height(), o.missile())
    };
    if super::play::blocked_for(lv, x, y, z, radius, height) {
        return Hit::Wall;
    }
    for (j, o) in objs.list.iter().enumerate() {
        if j == me {
            continue;
        }
        let interesting = if is_missile {
            o.flags & info::MF_SHOOTABLE != 0
        } else {
            o.flags & info::MF_SOLID != 0
        };
        if !interesting {
            continue;
        }
        let reach = radius + o.radius();
        if (x - o.x).abs() >= reach || (y - o.y).abs() >= reach {
            continue;
        }
        // Over it or under it. A rocket passes above a corpse, and a fireball
        // fired from a ledge sails over whatever is at the bottom of it.
        if is_missile && (z > o.z + o.height() || z + height < o.z) {
            continue;
        }
        return Hit::Thing(j);
    }
    Hit::None
}

/// One tic of horizontal movement. DOOM's `P_XYMovement`.
///
/// Answers what stopped it, which the caller needs: a missile explodes on
/// whatever it met and everything else simply stops.
pub fn xy_movement(lv: &Level, objs: &mut Objs, i: usize) -> Hit {
    let (mut xmove, mut ymove) = {
        let o = &objs.list[i];
        (
            o.momx.clamp(-MAXMOVE, MAXMOVE),
            o.momy.clamp(-MAXMOVE, MAXMOVE),
        )
    };
    if xmove == 0.0 && ymove == 0.0 {
        return Hit::None;
    }
    {
        let o = &mut objs.list[i];
        o.momx = xmove;
        o.momy = ymove;
    }

    let mut met = Hit::None;
    // Bounded as well as conditioned. The halving terminates on its own in
    // about six rounds from `MAXMOVE`, and a bound costs nothing next to a
    // kernel with no unwinder.
    for _ in 0..16 {
        if xmove == 0.0 && ymove == 0.0 {
            break;
        }
        let (stepx, stepy);
        if xmove > MAXMOVE / 2.0 || ymove > MAXMOVE / 2.0 {
            // Only the *positive* side is checked, which is DOOM's own quirk
            // and is kept: a thing moving hard in -x takes the whole step.
            stepx = xmove / 2.0;
            stepy = ymove / 2.0;
            xmove /= 2.0;
            ymove /= 2.0;
        } else {
            stepx = xmove;
            stepy = ymove;
            xmove = 0.0;
            ymove = 0.0;
        }
        let (tryx, tryy) = {
            let o = &objs.list[i];
            (o.x + stepx, o.y + stepy)
        };
        match obstacle(lv, objs, i, tryx, tryy) {
            Hit::None => {
                let o = &mut objs.list[i];
                o.x = tryx;
                o.y = tryy;
                if let Some(s) = lv.sector_index_at(tryx as i32, tryy as i32) {
                    o.sector = s;
                }
            }
            other => {
                met = other;
                let o = &mut objs.list[i];
                o.momx = 0.0;
                o.momy = 0.0;
                break;
            }
        }
    }

    // Friction, on the ground only, and never for a missile.
    let o = &mut objs.list[i];
    if o.missile() || o.z > o.floor_under(lv) {
        return met;
    }
    if o.momx.abs() < STOPSPEED && o.momy.abs() < STOPSPEED {
        o.momx = 0.0;
        o.momy = 0.0;
    } else {
        o.momx *= FRICTION;
        o.momy *= FRICTION;
    }
    met
}

/// One tic of vertical movement. DOOM's `P_ZMovement`.
///
/// Answers true when it hit the floor or the ceiling, which a missile treats
/// as having hit something.
pub fn z_movement(lv: &Level, objs: &mut Objs, i: usize) -> bool {
    let o = &mut objs.list[i];
    let floor = lv.sectors.get(o.sector).map(|s| s.floor as f32).unwrap_or(0.0);
    let ceiling = lv
        .sectors
        .get(o.sector)
        .map(|s| s.ceiling as f32)
        .unwrap_or(0.0);
    o.z += o.momz;

    if o.z <= floor {
        // **This is what carries a thing on a rising lift.** It used to be
        // free, because `z` was computed from the sector every time it was
        // asked for; now the floor coming up meets a thing that is below it
        // and pushes it back out, which is the same answer arrived at
        // deliberately.
        let landed = o.momz < 0.0 || o.z < floor;
        o.z = floor;
        if o.momz < 0.0 {
            o.momz = 0.0;
        }
        return landed && o.missile();
    }
    if o.falls() {
        // DOOM's first step of a fall is twice gravity, so a thing that walks
        // off a ledge drops rather than easing off it.
        o.momz -= if o.momz == 0.0 { GRAVITY * 2.0 } else { GRAVITY };
    }
    if o.z + o.height() > ceiling {
        o.z = ceiling - o.height();
        if o.momz > 0.0 {
            o.momz = 0.0;
        }
        return o.missile();
    }
    false
}

/// Move everything by one tic, and report what each missile met.
///
/// Runs before the state machines, which is DOOM's order: a missile that
/// arrives at a wall this tic explodes on it rather than a tic later.
pub fn run(lv: &Level, objs: &mut Objs, hits: &mut Vec<(usize, Hit)>) {
    for i in 0..objs.list.len() {
        if objs.list[i].remove {
            continue;
        }
        let met = xy_movement(lv, objs, i);
        let vertical = z_movement(lv, objs, i);
        if met != Hit::None {
            hits.push((i, met));
        } else if vertical {
            hits.push((i, Hit::Wall));
        }
    }
}

/// A level of exactly one sector, for a check that needs somewhere to stand.
fn one_sector(floor: i16, ceiling: i16) -> Level {
    Level {
        name: super::wad::Name::from_lump(b"MOTION  "),
        sector_lines: Vec::new(),
        tagged: Vec::new(),
        exited: false,
        things: Vec::new(),
        vertexes: Vec::new(),
        linedefs: Vec::new(),
        sidedefs: Vec::new(),
        segs: Vec::new(),
        subsectors: Vec::new(),
        nodes: Vec::new(),
        sectors: alloc::vec![super::level::Sector {
            floor,
            ceiling,
            floor_pic: super::wad::Name::from_lump(b"F       "),
            ceiling_pic: super::wad::Name::from_lump(b"F       "),
            light: 255,
            special: 0,
            tag: 0,
        }],
    }
}

/// What `diag doom` asks of the arithmetic.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    // The constants, which are DOOM's and not chosen. Friction especially:
    // 0.90625 is 0xE800/65536, and a rounder 0.9 would be a different game
    // for everything that slides.
    out.push((
        "the movement constants are DOOM's",
        GRAVITY == 1.0
            && MAXMOVE == 30.0
            && (FRICTION - 0.906_25).abs() < 1e-6
            && (STOPSPEED - 0.062_5).abs() < 1e-6,
    ));

    // The step loop halves until it fits, which is what stops a fast thing
    // tunnelling. Simulated here rather than driven, because what is being
    // checked is that the *sequence of offsets* sums to the whole move and
    // that no single step is large enough to skip a wall.
    let steps = |mut m: f32| -> (usize, f32, f32) {
        let (mut n, mut total, mut biggest) = (0usize, 0.0f32, 0.0f32);
        for _ in 0..16 {
            if m == 0.0 {
                break;
            }
            let step = if m > MAXMOVE / 2.0 { m / 2.0 } else { m };
            m = if m > MAXMOVE / 2.0 { m / 2.0 } else { 0.0 };
            total += step;
            if step > biggest {
                biggest = step;
            }
            n += 1;
        }
        (n, total, biggest)
    };
    let (n, total, biggest) = steps(MAXMOVE);
    out.push((
        "a full-speed move is split into steps that sum to it",
        n > 1 && (total - MAXMOVE).abs() < 1e-3 && biggest <= MAXMOVE / 2.0,
    ));
    let (n1, total1, _) = steps(4.0);
    out.push((
        "and a slow move is one step",
        n1 == 1 && (total1 - 4.0).abs() < 1e-6,
    ));

    // A missile's speed is fixed point and a walker's is not. The table says
    // 20 << 16 for a rocket and 8 for an imp, and dividing the wrong one gives
    // a rocket that crawls or a monster that teleports.
    let fixed = |v: i32| v as f32 / 65536.0;
    out.push((
        "a missile's speed is fixed point where a walker's is units",
        (fixed(1_310_720) - 20.0).abs() < 1e-3
            && (fixed(655_360) - 10.0).abs() < 1e-3
            && info::kind_of(3004).map(|k| k.speed) == Some(8),
    ));

    // **A thing on a rising floor rides it**, which is the property the whole
    // stored-`z` change had to preserve and the one the fixture cannot test:
    // its only moving surface is the door's *ceiling*, so nothing on that map
    // ever stands on a floor that moves.
    //
    // It used to be free. `z` was computed from the sector every time anybody
    // asked, so a floor that moved carried whatever stood on it with no code
    // at all. Now it comes from the clamp below -- the floor comes up, meets a
    // thing that is underneath it, and pushes it back out -- so the mechanism
    // is asserted here rather than assumed. A map with a lift would test the
    // same thing more convincingly and there is not one.
    {
        let mut lv = one_sector(0, 128);
        let mut objs = Objs::new();
        if let Some(o) = super::thing::Obj::spawn(
            info::row_of(2035).unwrap_or(0),
            0.0,
            0.0,
            0.0,
            0,
            0.0,
            128.0,
        ) {
            objs.list.push(o);
        }
        let started_on_the_floor = objs.list.first().map(|o| o.z) == Some(0.0);
        // The lift goes up by eight, twice.
        let mut rose = true;
        for step in 1..=2 {
            if let Some(sec) = lv.sectors.get_mut(0) {
                sec.floor = step * 8;
            }
            z_movement(&lv, &mut objs, 0);
            rose &= objs.list.first().map(|o| o.z) == Some((step * 8) as f32);
        }
        // And back down, where it falls under gravity rather than snapping --
        // which is the half a derived `z` got wrong in the other direction.
        //
        // **Two tics before it has moved at all**, and the first version of
        // this check asserted after one and failed. `z_movement` adds `momz`
        // at the top and applies gravity at the bottom, so the tic that
        // notices the floor has gone only sets the velocity; the tic after it
        // is the one that falls. That ordering is DOOM's, and a claim written
        // against the intuitive one is a claim that fails on correct code.
        if let Some(sec) = lv.sectors.get_mut(0) {
            sec.floor = 0;
        }
        z_movement(&lv, &mut objs, 0);
        let waited = objs.list.first().map(|o| o.z) == Some(16.0);
        z_movement(&lv, &mut objs, 0);
        let falling = objs.list.first().is_some_and(|o| o.z < 16.0 && o.z > 0.0);
        for _ in 0..8 {
            z_movement(&lv, &mut objs, 0);
        }
        let landed = objs.list.first().map(|o| o.z) == Some(0.0);
        let fell = waited && falling && landed;
        out.push((
            "a thing on a rising floor rides it, and falls when it drops",
            started_on_the_floor && rose && fell,
        ));
    }

    // Friction stops rather than approaching zero forever.
    let mut v = 1.0f32;
    let mut ticks = 0;
    while v.abs() >= STOPSPEED && ticks < 1000 {
        v *= FRICTION;
        ticks += 1;
    }
    out.push(("a slide stops in a bounded number of tics", ticks < 100));

    out
}

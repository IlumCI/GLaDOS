//! Standing in the level and walking about in it.
//!
//! The loop, the input and the collision. Adapted from room4doom and the
//! original in the parts that matter -- the tic rate, the movement speeds and
//! the idea of trying a move and refusing it -- and simplified everywhere it
//! can be without becoming a different thing.
//!
//! ### The tic
//!
//! DOOM runs its world at **35 Hz**, and that is not a frame rate. Every speed
//! in the game is expressed per tic, so a world running at some other rate is
//! a world where the player walks at the wrong speed. The renderer is free to
//! run at whatever rate it can; the world is not.
//!
//! 35 does not divide the kernel's 100 Hz timer, so a tic boundary is found
//! against the TSC rather than by counting interrupts -- `port::clock` exists
//! for exactly this, and `lapic::ticks()` at 10 ms would alias a 28.571 ms tic
//! badly enough to be visible as uneven walking.
//!
//! ### Collision, and what this is not
//!
//! DOOM's `P_TryMove` consults the blockmap, gathers the lines near the
//! destination, checks each for blocking, and then does the same for every
//! thing in range. This does the first half against every line in the level,
//! because the maps here have six of them and a blockmap lookup would be
//! machinery in front of a loop that is already instant.
//!
//! What it keeps is the part that matters to how it feels: a blocked move is
//! **retried along each axis** before being refused, which is what lets a
//! player slide along a wall instead of sticking to it. A collision routine
//! without that is one nobody enjoys, and it is two extra lines.

use super::level::{Level, NONE};
use super::math;
use super::pic::Art;
use super::specials;
use super::thinker::Thinkers;
use super::render::{Renderer, View, EYE_HEIGHT};
use crate::port::{keys, Surface};

/// The world's heartbeat, in microseconds. DOOM's 35 Hz.
const TIC_US: u64 = 1_000_000 / 35;

/// How far the player moves in one tic, walking and running. DOOM's own
/// figures: 25 map units a tic is about 875 a second.
const WALK: f32 = 25.0;
const RUN: f32 = 50.0;

/// Degrees turned per tic.
const TURN: f32 = 3.0;
const TURN_FAST: f32 = 6.0;

/// The player's width, which DOOM fixes at 16 units either side.
const RADIUS: f32 = 16.0;

/// How tall the player is, which DOOM fixes at 56 units.
///
/// Not the eye height -- the eye sits at 41 and the other 15 are the headroom
/// a doorway has to clear. A gap shorter than this is not a doorway.
const HEIGHT: f32 = 56.0;

/// The tallest step that can be walked up rather than blocked, DOOM's
/// `MAXSTEPMOVE`.
///
/// The whole of what makes a staircase a staircase and a ledge a wall. There
/// is no jump in DOOM: 24 units is the entire vertical vocabulary of the
/// player, and every stair in every map is built to it.
const MAX_STEP: f32 = 24.0;

/// Give up on catching up after this many missed tics.
///
/// Without a cap, a frame that took a long time -- a first frame with a cold
/// cache, or a moment when another task held the machine -- is repaid by
/// running the world at full speed until it catches up, which reads as the
/// player being flung across the room. Better to lose the time.
const MAX_CATCHUP: u32 = 4;

pub struct Player {
    pub x: f32,
    pub y: f32,
    /// Radians, 0 east, increasing anticlockwise.
    pub angle: f32,
    /// Height of the floor the player is standing on.
    pub floor: f32,
    /// Whether Use was held on the previous tic.
    ///
    /// Use is an *edge*, not a state. Holding the key against a door would
    /// otherwise re-trigger it thirty-five times a second, which for a
    /// repeatable door means it never finishes opening -- each tic refuses
    /// because the sector is busy, and the one tic it is not busy starts it
    /// again. Doors that open and immediately shut, forever.
    pub used: bool,
}

impl Player {
    pub fn at(lv: &Level) -> Option<Player> {
        let t = lv.player_start()?;
        let floor = lv.sector_at(t.x as i32, t.y as i32).map(|s| s.floor as f32).unwrap_or(0.0);
        Some(Player {
            x: t.x as f32,
            y: t.y as f32,
            angle: math::deg_to_rad(t.angle),
            floor,
            used: false,
        })
    }

    pub fn view(&self) -> View {
        View { x: self.x, y: self.y, z: self.floor + EYE_HEIGHT, angle: self.angle }
    }
}

/// How close a point comes to a line segment.
fn dist_to_seg(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let (dx, dy) = (bx - ax, by - ay);
    let len2 = dx * dx + dy * dy;
    if len2 <= 0.0 {
        let (ex, ey) = (px - ax, py - ay);
        return math::sqrt(ex * ex + ey * ey);
    }
    // The projection of the point onto the line, clamped to the segment --
    // clamped, because a wall is a segment and not an infinite line, and a
    // player standing beyond its end is not touching it.
    let t = (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0);
    let (cx, cy) = (ax + dx * t, ay + dy * t);
    let (ex, ey) = (px - cx, py - cy);
    math::sqrt(ex * ex + ey * ey)
}

/// What a two-sided line leaves open, vertically.
///
/// DOOM's `P_LineOpening`, ported from room4doom's `PortalZ` -- the lowest of
/// the two ceilings, the highest of the two floors, and the gap between them.
/// Three numbers that between them decide every vertical question a moving
/// thing can ask about a portal.
struct Opening {
    /// The lowest ceiling of the two sides.
    top: f32,
    /// The highest floor of the two sides -- the step you would climb.
    bottom: f32,
}

impl Opening {
    fn of(lv: &Level, l: &super::level::LineDef) -> Option<Opening> {
        if l.left == NONE {
            return None;
        }
        let side = |i: u16| -> Option<&super::level::Sector> {
            lv.sidedefs
                .get(i as usize)
                .and_then(|sd| lv.sectors.get(sd.sector as usize))
        };
        let (f, b) = (side(l.right)?, side(l.left)?);
        Some(Opening {
            top: (f.ceiling.min(b.ceiling)) as f32,
            bottom: (f.floor.max(b.floor)) as f32,
        })
    }
}

/// Would the player standing here be inside a wall?
fn blocked(lv: &Level, x: f32, y: f32, z: f32) -> bool {
    for l in lv.linedefs.iter() {
        // A one-sided line is always a wall, whatever its flags say -- there
        // is nothing on the other side to walk into. A two-sided one is a wall
        // if it is flagged impassable, and otherwise is decided by the gap.
        let wall = l.left == NONE || (l.flags & 1) != 0;
        let (Some(a), Some(b)) = (lv.vertexes.get(l.v1 as usize), lv.vertexes.get(l.v2 as usize))
        else {
            continue;
        };
        if dist_to_seg(x, y, a.x as f32, a.y as f32, b.x as f32, b.y as f32) >= RADIUS {
            continue;
        }
        if wall {
            return true;
        }
        // **The gap is what makes a door a door.** Until this existed a
        // two-sided line without the impassable flag was simply passable, so a
        // shut door -- which is a two-sided line whose back sector has been
        // squeezed to nothing -- was walked straight through. Nothing looked
        // wrong; the door was just scenery.
        let Some(o) = Opening::of(lv, l) else { return true };
        let too_short = o.top - o.bottom < HEIGHT;
        let no_headroom = o.top - z < HEIGHT;
        let step_too_high = o.bottom - z > MAX_STEP;
        if too_short || no_headroom || step_too_high {
            return true;
        }
    }
    false
}

/// Move if the destination is free, sliding along a wall if it is not.
fn try_move(lv: &Level, p: &mut Player, dx: f32, dy: f32) {
    let z = p.floor;
    if !blocked(lv, p.x + dx, p.y + dy, z) {
        p.x += dx;
        p.y += dy;
        return;
    }
    // The full move is refused, so try each axis alone. Against a wall running
    // north, the eastward half of a north-east move is what is blocked and the
    // northward half is not, so the player slides along it. Refusing outright
    // instead makes a wall something you stick to.
    if !blocked(lv, p.x + dx, p.y, z) {
        p.x += dx;
    } else if !blocked(lv, p.x, p.y + dy, z) {
        p.y += dy;
    }
}

/// One tic of the world.
fn tic(lv: &mut Level, p: &mut Player, th: &mut Thinkers) {
    let held = keys::snapshot();
    let fast = held.down(keys::SHIFT);
    let turn = math::deg_to_rad(if fast { TURN_FAST as i16 } else { TURN as i16 });
    let speed = if fast { RUN } else { WALK };

    if held.down(keys::LEFT) {
        p.angle += turn;
    }
    if held.down(keys::RIGHT) {
        p.angle -= turn;
    }
    if p.angle > math::PI {
        p.angle -= math::TAU;
    } else if p.angle < -math::PI {
        p.angle += math::TAU;
    }

    let (s, c) = (math::sin(p.angle), math::cos(p.angle));
    let mut fwd = 0.0;
    let mut side = 0.0;
    if held.down(keys::W) || held.down(keys::UP) {
        fwd += speed;
    }
    if held.down(keys::S) || held.down(keys::DOWN) {
        fwd -= speed;
    }
    // A and D strafe. DOOM puts strafe on comma and full stop by default and
    // nobody has played it that way since about 1997.
    if held.down(keys::D) {
        side += speed;
    }
    if held.down(keys::A) {
        side -= speed;
    }
    if fwd != 0.0 || side != 0.0 {
        // Forward is (cos, sin); right is (sin, -cos), which is forward turned
        // a quarter turn clockwise.
        let dx = c * fwd + s * side;
        let dy = s * fwd - c * side;
        let was = (p.x, p.y);
        try_move(&*lv, p, dx, dy);
        // Anything crossed on the way. The *path* rather than the destination,
        // because a trigger line is infinitely thin: at 25 units a tic a
        // player can be on one side at the start of a tic and the other at the
        // end without ever having been within a unit of the line itself.
        if was != (p.x, p.y) {
            specials::cross_lines(lv, th, was, (p.x, p.y));
        }
    }

    // Use, on the press rather than while held.
    let use_now = held.down(keys::USE);
    if use_now && !p.used {
        specials::use_lines(lv, th, p.x, p.y, p.angle);
    }
    p.used = use_now;

    th.tick(lv);
    // The floor under wherever we are, sampled **every** tic rather than only
    // on one where a movement key was held.
    //
    // It used to sit inside the branch above, which is correct for walking and
    // wrong for standing still on something that moves: a lift would rise
    // through a stationary player, who would keep the height of the floor as
    // it was when they last pressed a key. The whole point of a lift is that
    // you stand on it and do nothing.
    if let Some(sec) = lv.sector_at(p.x as i32, p.y as i32) {
        p.floor = sec.floor as f32;
    }
}

/// What a session did, for the caller to report.
pub struct Stats {
    pub frames: u32,
    pub tics: u32,
    pub ms: u64,
    /// Where the player finished, which is what makes a headless run
    /// checkable: hold one key for a known time and the destination is
    /// arithmetic, not an opinion about a screenshot.
    pub x: f32,
    pub y: f32,
    pub deg: f32,
    /// How many surfaces were still moving when the run ended, and how many
    /// specials fired over the whole of it. Reported because a door is a
    /// number rather than a picture: "the ceiling is at 124 after 70 tics" is
    /// a test and a screenshot of a doorway is not.
    pub moving: usize,
    /// The height of the sector the run was asked to watch, at the end.
    pub watched: i16,
    /// How many specials fired over the whole run.
    pub spawned: usize,
    /// How many times a sprite changed picture.
    pub flips: usize,
    /// Whether something ended the level.
    pub exited: bool,
    /// Whether the shading table came out of the WAD's own COLORMAP. Carried
    /// rather than printed, because nothing in this tree may reach the
    /// printing macro -- the shell does the talking.
    pub lit_from_wad: bool,
}

/// Walk around the level until Escape, or until `limit_ms` has passed.
///
/// The bounded form is not a convenience: the harness drives this machine over
/// a serial line and sends the next command when it sees a prompt, so a
/// program that runs until a keypress can never be tested -- the keystroke
/// that would end it is the one thing the harness cannot deliver.
#[allow(clippy::too_many_arguments)]
pub fn run(
    surf: &mut Surface,
    lv: &mut Level,
    art: &Art<'_>,
    things: &super::sprite::Things,
    limit_ms: u64,
    script: &[(u8, u64)],
    watch: usize,
) -> Option<Stats> {
    let mut p = Player::at(&*lv)?;
    surf.set_palette_rgb(art.playpal);
    let sky = super::draw::nearest(art.playpal, 0x10, 0x18, 0x60);
    let mut r = Renderer::new(surf, art);
    let mut th = Thinkers::new(&*lv);
    // The same projection scale the walls use: a 90-degree field of view, so
    // half the width. Read once rather than per frame, and from the surface
    // rather than from a constant, because a different resolution would give
    // the sprites a different field of view from the geometry.
    let surf_scale = surf.width() as f32 / 2.0;

    // Anything held when the game started belongs to whoever was typing, not
    // to the player.
    keys::clear();
    // Then whatever the caller scripted, which is the only way a held key
    // reaches this over a serial line. Applied *after* the clear rather than
    // before it -- the first version cleared on entry and silently threw the
    // script away, and the symptom was a player who stood perfectly still
    // while the tic counter said the world was running.
    //
    // **A key can be scheduled rather than only held**, and that is not a
    // convenience. Use is edge-triggered, so a script that holds it from the
    // first tic spends the press immediately -- at the spawn point, sixty-four
    // units being the whole reach, which on any real map is nowhere near the
    // thing you meant to open. The press has to happen *after* the walking,
    // and nothing in a held-key model can say that.
    for (k, at) in script {
        if *at == 0 {
            keys::force(*k, true);
        }
    }
    let mut pending: alloc::vec::Vec<(u8, u64)> =
        script.iter().copied().filter(|(_, at)| *at != 0).collect();

    let start = crate::port::now_us();
    let mut next_tic = start;
    let mut last_phase = things.phase(0);
    let mut stats = Stats {
        frames: 0,
        tics: 0,
        ms: 0,
        x: p.x,
        y: p.y,
        deg: 0.0,
        moving: 0,
        watched: 0,
        spawned: 0,
        flips: 0,
        exited: false,
        lit_from_wad: r.lit_from_wad,
    };

    loop {
        let now = crate::port::now_us();
        if keys::down(keys::ESC) {
            break;
        }
        if limit_ms != 0 && (now - start) / 1000 >= limit_ms {
            break;
        }
        // The level is over. Stopping here rather than inside the special is
        // what the flag exists for -- an exit fires from deep in a dispatch
        // and the screen has to be handed back by whoever took it.
        if lv.exited {
            break;
        }

        // Anything the script scheduled for by now. Pressed and left down: a
        // release would need a second time and nothing yet wants one.
        let elapsed = (now - start) / 1000;
        pending.retain(|(k, at)| {
            if elapsed >= *at {
                keys::force(*k, true);
                false
            } else {
                true
            }
        });

        // Catch up on whole tics, so the world advances at 35 Hz whatever the
        // renderer manages.
        let mut ran = 0;
        while now >= next_tic && ran < MAX_CATCHUP {
            tic(lv, &mut p, &mut th);
            next_tic += TIC_US;
            stats.tics += 1;
            ran += 1;
        }
        if ran == MAX_CATCHUP {
            // Too far behind to repay. Drop the debt rather than sprinting.
            next_tic = now + TIC_US;
        }

        let v = p.view();
        r.frame(surf, &*lv, art, v, sky);
        r.things(surf, &*lv, things, &v, surf_scale, stats.tics);
        // Whether anything actually changed picture this frame. Counted here
        // rather than asserted, because how many flips a run sees depends on
        // how long it ran and what is on the level -- what matters is that it
        // is not zero on a level with animated things on it.
        let phase = things.phase(stats.tics);
        if phase != last_phase {
            stats.flips += 1;
            last_phase = phase;
        }
        surf.present();
        stats.frames += 1;

        // Park until something happens. The 100 Hz timer guarantees a wake
        // long before the next tic is due.
        if crate::port::now_us() < next_tic {
            crate::port::idle();
        }
    }

    // And anything held when it ends does not belong to the shell either.
    keys::clear();
    stats.ms = (crate::port::now_us() - start) / 1000;
    stats.x = p.x;
    stats.y = p.y;
    let mut d = p.angle * (180.0 / math::PI);
    if d < 0.0 {
        d += 360.0;
    }
    stats.deg = d;
    stats.moving = th.len();
    stats.spawned = th.spawned;
    stats.exited = lv.exited;
    stats.watched = lv
        .sectors
        .get(watch)
        .map(|s| s.ceiling)
        .unwrap_or(0);
    Some(stats)
}

/// What `diag doom` asks of the line opening.
///
/// The three rules DOOM decides every portal with, checked as arithmetic
/// because on a map they are invisible: a door that is walked through looks
/// like a door somebody forgot to close, and a ledge that is climbed looks
/// like a map with no ledge.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out: alloc::vec::Vec<(&'static str, bool)> = alloc::vec::Vec::new();
    // `blocked` needs a whole level, so the rules are checked directly. They
    // are three comparisons and this is where they are stated once.
    let blocks = |top: f32, bottom: f32, z: f32| -> bool {
        top - bottom < HEIGHT || top - z < HEIGHT || bottom - z > MAX_STEP
    };
    out.push(("a shut door -- no gap at all -- blocks", blocks(64.0, 64.0, 0.0)));
    out.push(("an open doorway does not", !blocks(128.0, 0.0, 0.0)));
    out.push((
        "a gap shorter than the player blocks, even wide open below",
        blocks(40.0, 0.0, 0.0),
    ));
    out.push((
        "a step of 24 is climbed and 25 is not",
        !blocks(200.0, 24.0, 0.0) && blocks(200.0, 25.0, 0.0),
    ));
    out.push((
        "standing higher makes the same step climbable",
        !blocks(200.0, 25.0, 1.0),
    ));
    out.push((
        "a ceiling too low over where the player *is* blocks",
        blocks(50.0, 0.0, 0.0),
    ));
    out
}

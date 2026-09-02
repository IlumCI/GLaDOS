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

/// Would the player standing here be inside a wall?
fn blocked(lv: &Level, x: f32, y: f32) -> bool {
    for l in lv.linedefs.iter() {
        // Two-sided lines are openings unless flagged impassable. A one-sided
        // line is always a wall, whatever its flags say -- there is nothing on
        // the other side of it to walk into.
        let solid = l.left == NONE || (l.flags & 1) != 0;
        if !solid {
            continue;
        }
        let (Some(a), Some(b)) = (lv.vertexes.get(l.v1 as usize), lv.vertexes.get(l.v2 as usize))
        else {
            continue;
        };
        if dist_to_seg(x, y, a.x as f32, a.y as f32, b.x as f32, b.y as f32) < RADIUS {
            return true;
        }
    }
    false
}

/// Move if the destination is free, sliding along a wall if it is not.
fn try_move(lv: &Level, p: &mut Player, dx: f32, dy: f32) {
    if !blocked(lv, p.x + dx, p.y + dy) {
        p.x += dx;
        p.y += dy;
        return;
    }
    // The full move is refused, so try each axis alone. Against a wall running
    // north, the eastward half of a north-east move is what is blocked and the
    // northward half is not, so the player slides along it. Refusing outright
    // instead makes a wall something you stick to.
    if !blocked(lv, p.x + dx, p.y) {
        p.x += dx;
    } else if !blocked(lv, p.x, p.y + dy) {
        p.y += dy;
    }
}

/// One tic of the world.
fn tic(lv: &Level, p: &mut Player) {
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
        try_move(lv, p, dx, dy);
        // The floor under wherever we ended up, so a step up or down is
        // followed. There is no gravity and no step height yet: on a real map
        // this will teleport the eye up a cliff rather than refuse it.
        if let Some(sec) = lv.sector_at(p.x as i32, p.y as i32) {
            p.floor = sec.floor as f32;
        }
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
pub fn run(
    surf: &mut Surface,
    lv: &Level,
    art: &Art<'_>,
    limit_ms: u64,
    script: &[u8],
) -> Option<Stats> {
    let mut p = Player::at(lv)?;
    surf.set_palette_rgb(art.playpal);
    let sky = super::draw::nearest(art.playpal, 0x10, 0x18, 0x60);
    let mut r = Renderer::new(surf, art);

    // Anything held when the game started belongs to whoever was typing, not
    // to the player.
    keys::clear();
    // Then whatever the caller scripted, which is the only way a held key
    // reaches this over a serial line. Applied *after* the clear rather than
    // before it -- the first version cleared on entry and silently threw the
    // script away, and the symptom was a player who stood perfectly still
    // while the tic counter said the world was running.
    for k in script {
        keys::force(*k, true);
    }

    let start = crate::port::now_us();
    let mut next_tic = start;
    let mut stats =
        Stats { frames: 0, tics: 0, ms: 0, x: p.x, y: p.y, deg: 0.0, lit_from_wad: r.lit_from_wad };

    loop {
        let now = crate::port::now_us();
        if keys::down(keys::ESC) {
            break;
        }
        if limit_ms != 0 && (now - start) / 1000 >= limit_ms {
            break;
        }

        // Catch up on whole tics, so the world advances at 35 Hz whatever the
        // renderer manages.
        let mut ran = 0;
        while now >= next_tic && ran < MAX_CATCHUP {
            tic(lv, &mut p);
            next_tic += TIC_US;
            stats.tics += 1;
            ran += 1;
        }
        if ran == MAX_CATCHUP {
            // Too far behind to repay. Drop the debt rather than sprinting.
            next_tic = now + TIC_US;
        }

        r.frame(surf, lv, art, p.view(), sky);
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
    Some(stats)
}

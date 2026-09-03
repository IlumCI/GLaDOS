// Ported from room4doom (MIT, Luke Jones): `gameplay/src/env/doors.rs` and
// the `move_plane` half of `gameplay/src/env/specials.rs`.
//   https://github.com/flukejones/room4doom
//
// **Adapted, and the adaptation is the interesting part.** Upstream keeps
// DOOM's own arrangement: an intrusive doubly-linked list of thinkers, each a
// raw pointer into a custom arena, with `sector.specialdata` pointing back at
// whichever thinker owns that sector. That design exists because the original
// ran on a machine where a linked list was cheaper than anything else and
// where a thinker had to be removable from inside its own `think`.
//
// Here it is a `Vec` and an enum, for `Racy`'s reason -- one core, one address
// space, and nothing that runs between two tics. None of the pointer
// discipline that list exists to provide buys anything, and a `Vec<Thinker>`
// cannot dangle. Removal is deferred to the end of a tic rather than done
// inside the walk, which is what the linked list was for.
//
// `sector.specialdata` becomes `busy`, and it is not an optimisation: without
// it, pressing Use twice on one door spawns a second thinker for the same
// sector and the two fight over its ceiling, one raising while the other
// lowers, which reads as a door that judders and never opens.

use alloc::vec::Vec;

use super::level::Level;

/// How far a normal door moves in one tic. DOOM's `VDOORSPEED`.
pub const DOOR_SPEED: i16 = 2;

/// How long a door that closes itself waits at the top, in tics. DOOM's
/// `VDOORWAIT` -- 150 tics is a little over four seconds.
pub const DOOR_WAIT: i32 = 150;

/// The gap a door leaves under the lintel when it opens.
///
/// DOOM opens a door to the lowest neighbouring ceiling *less four*. Without
/// the four the door's top texture would have nothing to draw on and the
/// doorway would read as a hole in the wall rather than as an opening under a
/// header.
pub const DOOR_HEADROOM: i16 = 4;

/// What happened to a surface asked to move.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Plane {
    /// Still going.
    Moving,
    /// Something was in the way.
    Crushed,
    /// It arrived.
    PastDest,
}

/// Move one sector's ceiling toward a height, one tic's worth.
///
/// DOOM's `T_MovePlane` for the ceiling case. The shape matters: it moves by
/// `speed` unless that would overshoot, in which case it lands *exactly* on
/// the destination and says so. Clamping without reporting would leave a door
/// jittering around its top by less than a unit forever, because nothing would
/// ever tell it that it had arrived.
pub fn move_ceiling(sec: &mut super::level::Sector, speed: i16, dest: i16, dir: i8) -> Plane {
    match dir {
        -1 => {
            if sec.ceiling - speed <= dest {
                sec.ceiling = dest;
                Plane::PastDest
            } else {
                sec.ceiling -= speed;
                // A ceiling cannot pass its own floor. In DOOM this is where
                // a crusher decides whether it has caught something; here it
                // is the shut position of a door.
                if sec.ceiling < sec.floor {
                    sec.ceiling = sec.floor;
                    return Plane::PastDest;
                }
                Plane::Moving
            }
        }
        1 => {
            if sec.ceiling + speed >= dest {
                sec.ceiling = dest;
                Plane::PastDest
            } else {
                sec.ceiling += speed;
                Plane::Moving
            }
        }
        _ => Plane::Moving,
    }
}

/// The same for a floor, which lifts and stairs move.
pub fn move_floor(sec: &mut super::level::Sector, speed: i16, dest: i16, dir: i8) -> Plane {
    match dir {
        -1 => {
            if sec.floor - speed <= dest {
                sec.floor = dest;
                Plane::PastDest
            } else {
                sec.floor -= speed;
                Plane::Moving
            }
        }
        1 => {
            if sec.floor + speed >= dest {
                sec.floor = dest;
                Plane::PastDest
            } else {
                sec.floor += speed;
                if sec.floor > sec.ceiling {
                    sec.floor = sec.ceiling;
                    return Plane::PastDest;
                }
                Plane::Moving
            }
        }
        _ => Plane::Moving,
    }
}

/// Which of DOOM's door behaviours this is.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DoorKind {
    /// Opens, waits, closes itself.
    Normal,
    /// Opens and stays open.
    Open,
    /// Closes and stays closed.
    Close,
    /// Closes, waits thirty seconds, opens.
    Close30ThenOpen,
}

/// A ceiling on its way somewhere.
pub struct Door {
    pub sector: usize,
    pub kind: DoorKind,
    /// The height it opens to.
    pub top: i16,
    pub speed: i16,
    /// 1 rising, 0 waiting at the top, -1 lowering.
    pub direction: i8,
    pub topwait: i32,
    pub countdown: i32,
}

/// Anything that acts on its own over time.
///
/// One variant so far. It is an enum rather than a trait object because the
/// set is closed and small -- DOOM has about a dozen -- and a `Box<dyn Think>`
/// would put an allocation and a vtable in front of a match on a field.
pub enum Thinker {
    Door(Door),
}

/// Everything currently moving, and which sectors are spoken for.
pub struct Thinkers {
    list: Vec<Thinker>,
    /// One flag per sector: is something already moving it?
    ///
    /// DOOM's `sector.specialdata`, which is a pointer there and a bool here
    /// because nothing needs to find the thinker from the sector -- only to
    /// know that one exists. See the module note for what its absence costs.
    busy: Vec<bool>,
}

impl Thinkers {
    pub fn new(lv: &Level) -> Thinkers {
        Thinkers { list: Vec::new(), busy: alloc::vec![false; lv.sectors.len()] }
    }

    pub fn len(&self) -> usize {
        self.list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    pub fn busy(&self, sector: usize) -> bool {
        self.busy.get(sector).copied().unwrap_or(true)
    }

    /// Start a door on a sector, unless something is already moving it.
    ///
    /// Answers whether it started, because a use-activated door that is
    /// already moving should not re-trigger and the caller wants to know.
    pub fn spawn_door(&mut self, lv: &Level, sector: usize, kind: DoorKind) -> bool {
        if self.busy(sector) {
            return false;
        }
        let Some(sec) = lv.sectors.get(sector) else { return false };
        let top = lv.lowest_neighbour_ceiling(sector) - DOOR_HEADROOM;
        let (direction, countdown) = match kind {
            DoorKind::Close => (-1, 0),
            DoorKind::Close30ThenOpen => (-1, 0),
            _ => (1, 0),
        };
        // A door told to close that is already shut, or to open that is
        // already open, still gets a thinker: it arrives on its first tic and
        // removes itself, which is what the original does and is simpler than
        // a special case that has to agree with the state machine.
        let _ = sec;
        self.list.push(Thinker::Door(Door {
            sector,
            kind,
            top,
            speed: DOOR_SPEED,
            direction,
            topwait: DOOR_WAIT,
            countdown,
        }));
        if let Some(b) = self.busy.get_mut(sector) {
            *b = true;
        }
        true
    }

    /// One tic for everything moving.
    ///
    /// Finished thinkers are collected and dropped *after* the walk rather
    /// than removed during it. Removing inside would invalidate the index the
    /// loop is holding, which is the whole reason the original used a linked
    /// list -- and a `Vec` with a retain afterwards gets the same property for
    /// less.
    pub fn tick(&mut self, lv: &mut Level) {
        let mut done: Vec<usize> = Vec::new();
        for i in 0..self.list.len() {
            let finished = match &mut self.list[i] {
                Thinker::Door(d) => Self::door_tic(d, lv),
            };
            if finished {
                done.push(i);
            }
        }
        for &i in done.iter().rev() {
            let sector = match &self.list[i] {
                Thinker::Door(d) => d.sector,
            };
            if let Some(b) = self.busy.get_mut(sector) {
                *b = false;
            }
            self.list.remove(i);
        }
    }

    /// One door, one tic. Answers whether it is finished.
    fn door_tic(d: &mut Door, lv: &mut Level) -> bool {
        let Some(sec) = lv.sectors.get_mut(d.sector) else { return true };
        match d.direction {
            // Waiting at the top.
            0 => {
                d.countdown -= 1;
                if d.countdown <= 0 {
                    match d.kind {
                        DoorKind::Normal => d.direction = -1,
                        DoorKind::Close30ThenOpen => d.direction = 1,
                        _ => return true,
                    }
                }
                false
            }
            // Lowering.
            -1 => {
                let floor = sec.floor;
                match move_ceiling(sec, d.speed, floor, -1) {
                    Plane::PastDest => match d.kind {
                        DoorKind::Close30ThenOpen => {
                            d.direction = 0;
                            d.countdown = 35 * 30;
                            false
                        }
                        _ => true,
                    },
                    // Something is under it. A door that is not deliberately a
                    // crusher goes back up rather than through.
                    Plane::Crushed => {
                        if !matches!(d.kind, DoorKind::Close) {
                            d.direction = 1;
                        }
                        false
                    }
                    Plane::Moving => false,
                }
            }
            // Rising.
            1 => {
                let top = d.top;
                if move_ceiling(sec, d.speed, top, 1) == Plane::PastDest {
                    match d.kind {
                        DoorKind::Normal => {
                            d.direction = 0;
                            d.countdown = d.topwait;
                            false
                        }
                        _ => true,
                    }
                } else {
                    false
                }
            }
            _ => true,
        }
    }
}

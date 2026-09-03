//! The gun in your hands: its own state machine, and where it is drawn.
//!
//! Ported from room4doom's `gameplay/src/player_sprite.rs`. A player sprite is
//! a second state machine running beside the world's, with the same shape as a
//! thing's -- a state, a countdown, and an action on entry -- and it is what
//! decides how fast a weapon fires.
//!
//! ### The rate of fire is a table, not a constant
//!
//! Firing was edge-triggered while this file did not exist: one shot per
//! press, which is slower than DOOM and never faster. DOOM's rate is written
//! down in the states rather than chosen by anybody -- but it is **not** the
//! length of the attack chain, which is the obvious reading and is wrong.
//!
//! `A_ReFire` runs on *entry* to its state, and if the trigger is down it
//! restarts the chain there and then. So the state carrying it never spends
//! its own tics while you are holding the trigger, and the rate is the chain
//! up to that state and not through it:
//!
//! ```text
//! pistol    PISGA(4) PISGB(6, fire) PISGC(4) | PISGB(5, refire)   14 tics
//! chaingun  CHGGA(4, fire) CHGGB(4, fire)    | CHGGB(0, refire)    8 tics
//! ```
//!
//! Fourteen tics is 2.5 shots a second, which is exactly what DOOM's pistol
//! does. Nineteen -- the whole chain -- would be 1.8, and the difference was
//! measured rather than reasoned about: a run holding the trigger for three
//! seconds fired **eight** shots where the chain length predicted five, and
//! the prediction was what was wrong.
//!
//! Neither number appears in this file. `held_cycle` computes it, and the
//! claims assert what the table says rather than what anybody remembers.
//!
//! ### What fires and what only animates
//!
//! The **pistol and the chaingun** are wired: both are one bullet from one
//! clip round, which is all `shoot::fire` can do. Everything else animates
//! correctly, consumes nothing and hits nothing, and says so through
//! `Psprite::armed`. The shotgun needs seven pellets with a spread, which
//! needs the random table; the rocket launcher, the plasma rifle and the BFG
//! need projectiles, which are things that move and nothing here moves yet.
//! Wiring them to fire single bullets would be worse than leaving them: a
//! shotgun that behaves like a pistol is a bug that looks like a balance
//! decision.
//!
//! ### The gun does not bob, and that is visible
//!
//! DOOM's `A_WeaponReady` moves `sx` and `sy` around their resting values with
//! the player's own bob, so the weapon sways as you walk. Here they are
//! constants. The bob is a property of the player's movement -- `P_MovePlayer`
//! accumulates it from the momentum -- and there is no momentum yet: this
//! player is teleported by their velocity each tic rather than carrying one.
//! Faking a sway from the movement keys would look right and would be a second
//! source of truth about how fast the player is going.
//!
//! ### Weapon switching is absent
//!
//! `up` and `down` are in the table and nothing walks them, because there is
//! no key that selects a weapon. Picking one up records it and the gun in your
//! hands does not change.

use alloc::vec::Vec;

use super::info;
use super::pic::Patch;
use super::player::{Status, Weapon};
use super::sprite::{Frame, Sprites};

/// How far down the screen a raised weapon sits. DOOM's `WEAPONTOP`.
pub const TOP: f32 = 32.0;

/// The horizontal offset of a weapon at rest.
///
/// One, and not zero, and not a hundred and sixty. DOOM's `A_WeaponReady` sets
/// `psp->sx = FRACUNIT + bob`, so the resting value is 1 and the bob is added
/// to it -- and `R_DrawPSprite` then subtracts 160 and adds the screen centre
/// back, which cancels. The whole of the placement is therefore the patch's
/// own offsets, which is why a weapon sprite carries a left offset of -125.
pub const REST: f32 = 1.0;

/// The gun, as a state machine.
pub struct Psprite {
    pub state: u16,
    pub tics: i16,
    /// Where it is drawn, in DOOM's units before the patch offsets.
    pub sx: f32,
    pub sy: f32,
}

impl Psprite {
    /// A weapon, ready.
    pub fn ready(w: Weapon) -> Psprite {
        let s = info::WEAPONS[w as usize].ready;
        Psprite {
            state: s,
            tics: info::STATES.get(s as usize).map(|st| st.tics).unwrap_or(-1),
            sx: REST,
            sy: TOP,
        }
    }

    /// Whether this weapon's shot is implemented, as against merely animated.
    pub fn armed(w: Weapon) -> bool {
        matches!(w, Weapon::Pistol | Weapon::Chaingun)
    }

    /// Start the attack chain.
    pub fn attack(&mut self, w: Weapon) {
        self.set(info::WEAPONS[w as usize].attack);
    }

    /// Back to the ready state.
    pub fn rest(&mut self, w: Weapon) {
        self.set(info::WEAPONS[w as usize].ready);
    }

    /// DOOM's `P_SetPsprite`, with the same zero-tic fall-through a thing has
    /// and the same bound on it, for the same reason: this walks a generated
    /// table in a kernel with no unwinder.
    ///
    /// The action of each state entered is **not** run here. It is answered,
    /// so the caller -- which has the level, the objects and the player in
    /// hand -- can run it. Exactly the arrangement `thing.rs` uses, and for
    /// the same reason.
    fn set(&mut self, state: u16) {
        self.state = state;
        self.tics = info::STATES.get(state as usize).map(|s| s.tics).unwrap_or(-1);
    }

    /// One tic. Answers the action of the state it landed in, if any.
    ///
    /// A weapon in a state with negative `tics` is one waiting for the player
    /// rather than for the clock -- which is every ready state, since a gun
    /// held still does not animate.
    pub fn tick(&mut self) -> Option<u8> {
        if self.tics < 0 {
            return None;
        }
        self.tics -= 1;
        if self.tics > 0 {
            return None;
        }
        let next = info::STATES.get(self.state as usize).map(|s| s.next).unwrap_or(0);
        if next == 0 {
            return None;
        }
        self.set(next);
        info::STATES.get(next as usize).map(|s| s.action).filter(|a| *a != 0)
    }

    /// The action of the state it is in right now, for the first tic after a
    /// chain is started.
    pub fn action(&self) -> Option<u8> {
        info::STATES
            .get(self.state as usize)
            .map(|s| s.action)
            .filter(|a| *a != 0)
    }

    /// Whether this weapon has a round to spend.
    pub fn loaded(w: Weapon, st: &Status) -> bool {
        match w.ammo() {
            None => true,
            Some(a) => st.ammo[a as usize] > 0,
        }
    }
}

/// The pictures every weapon can show.
///
/// All nine, decoded once when a level loads. Decoding only what the player
/// owns would be cheaper and would have to be redone the moment they walked
/// over a shotgun, which is the sort of lazy loading that works until somebody
/// picks something up mid-corridor.
pub struct Guns {
    art: Vec<(u16, Frame)>,
}

impl Guns {
    pub fn none() -> Guns {
        Guns { art: Vec::new() }
    }

    /// Every state of every weapon's chains that this WAD has a picture for.
    pub fn load(sprites: &Sprites) -> Guns {
        let mut art: Vec<(u16, Frame)> = Vec::new();
        let mut want: Vec<u16> = Vec::new();
        for w in info::WEAPONS.iter() {
            for start in [w.up, w.down, w.ready, w.attack, w.flash] {
                let mut st = start;
                let mut steps = 0;
                while st != 0 && steps < 24 {
                    if want.contains(&st) {
                        break;
                    }
                    want.push(st);
                    let Some(row) = info::STATES.get(st as usize) else { break };
                    if row.tics < 0 {
                        break;
                    }
                    st = row.next;
                    steps += 1;
                }
            }
        }
        for st in want {
            let Some((name, letter)) = info::frame_of(st) else { continue };
            let Some(f) = Frame::load(sprites, name, letter) else { continue };
            art.push((st, f));
        }
        art.sort_unstable_by_key(|(s, _)| *s);
        Guns { art }
    }

    pub fn len(&self) -> usize {
        self.art.len()
    }

    pub fn is_empty(&self) -> bool {
        self.art.is_empty()
    }

    /// The picture for one state, if this WAD has it.
    pub fn frame(&self, state: u16) -> Option<&Patch> {
        let i = self.art.binary_search_by_key(&state, |(s, _)| *s).ok()?;
        // A weapon has one picture from every angle: it is always seen from
        // the same place, which is behind it.
        self.art.get(i).and_then(|(_, f)| f.at(0)).map(|(p, _)| p)
    }
}

/// How many tics between shots while the trigger is held.
///
/// The chain up to the state carrying `A_ReFire`, and **not** through it: that
/// action runs on entry and restarts the chain, so its own tics are never
/// spent while the trigger is down.
pub fn held_cycle(w: Weapon) -> i32 {
    let wi = &info::WEAPONS[w as usize];
    let (mut st, mut total, mut steps) = (wi.attack, 0i32, 0);
    while st != 0 && steps < 24 {
        let Some(row) = info::STATES.get(st as usize) else { break };
        if row.action == info::A_REFIRE {
            break;
        }
        if row.tics < 0 {
            break;
        }
        total += row.tics as i32;
        st = row.next;
        steps += 1;
    }
    total
}

/// The whole chain, which is what a single press costs.
pub fn full_cycle(w: Weapon) -> i32 {
    let wi = &info::WEAPONS[w as usize];
    let (mut st, mut total, mut steps) = (wi.attack, 0i32, 0);
    while st != 0 && steps < 24 {
        let Some(row) = info::STATES.get(st as usize) else { break };
        if row.tics < 0 {
            break;
        }
        total += row.tics as i32;
        if row.next == wi.ready {
            break;
        }
        st = row.next;
        steps += 1;
    }
    total
}

/// What `diag doom` asks of the weapon table.
///
/// All of it arithmetic over the generated states, because what would
/// otherwise check the rate of fire is somebody counting shots in a second.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    // How often a weapon fires while the trigger is held. Stops *before* the
    // refire state's own tics, because that action restarts the chain on entry
    // -- which is the difference between 14 tics and 19 for a pistol, and
    // therefore between DOOM's 2.5 shots a second and 1.8.
    out.push((
        "the pistol fires every 14 tics held, and the chaingun every 8",
        held_cycle(Weapon::Pistol) == 14 && held_cycle(Weapon::Chaingun) == 8,
    ));
    out.push((
        "letting go costs the rest of the chain, which is five more tics",
        full_cycle(Weapon::Pistol) == held_cycle(Weapon::Pistol) + 5,
    ));
    out.push((
        "and a shotgun is slower than a pistol by more than a factor of two",
        held_cycle(Weapon::Shotgun) > 2 * held_cycle(Weapon::Pistol),
    ));

    // Every attack chain ends by asking whether the trigger is still down.
    // That is the whole of automatic fire, and a chain missing it would be a
    // weapon that fires once and then stands there.
    let refires = |w: Weapon| -> bool {
        let info = &info::WEAPONS[w as usize];
        let (mut st, mut steps) = (info.attack, 0);
        while st != 0 && steps < 24 {
            let Some(row) = info::STATES.get(st as usize) else { break };
            if row.action == info::A_REFIRE {
                return true;
            }
            if row.tics < 0 || row.next == info.ready {
                return false;
            }
            st = row.next;
            steps += 1;
        }
        false
    };
    out.push((
        "every weapon asks whether the trigger is still down",
        (0..9).all(|i| {
            refires(match i {
                0 => Weapon::Fist,
                1 => Weapon::Pistol,
                2 => Weapon::Shotgun,
                3 => Weapon::Chaingun,
                4 => Weapon::Missile,
                5 => Weapon::Plasma,
                6 => Weapon::Bfg,
                7 => Weapon::Chainsaw,
                _ => Weapon::SuperShotgun,
            })
        }),
    ));

    // A ready state waits for the player rather than for the clock, which is
    // what `tics: -1`... is not. It is 1, and it loops back to itself, so the
    // ready action runs every single tic -- which is how DOOM notices the
    // trigger going down at all.
    let ready = info::WEAPONS[Weapon::Pistol as usize].ready;
    out.push((
        "a ready weapon re-asks every tic",
        info::STATES.get(ready as usize).is_some_and(|s| {
            s.tics == 1 && s.next == ready && s.action == info::A_WEAPONREADY
        }),
    ));

    // Starting the attack chain lands on its first state with its own tics.
    let mut p = Psprite::ready(Weapon::Pistol);
    let was = p.state;
    p.attack(Weapon::Pistol);
    out.push((
        "attacking leaves the ready state",
        was == ready && p.state == info::WEAPONS[Weapon::Pistol as usize].attack && p.tics > 0,
    ));

    // What `armed` claims must be what the table says. A weapon listed as
    // firing whose chain contains no firing action would be a claim in a
    // comment and nothing else -- and the failure would be a trigger that
    // animates, spends ammunition and hits nothing.
    let fires = |w: Weapon| -> bool {
        let wi = &info::WEAPONS[w as usize];
        let (mut st, mut steps) = (wi.attack, 0);
        while st != 0 && steps < 24 {
            let Some(row) = info::STATES.get(st as usize) else { break };
            if row.action == info::A_FIREPISTOL || row.action == info::A_FIRECGUN {
                return true;
            }
            if row.tics < 0 || row.action == info::A_REFIRE {
                return false;
            }
            st = row.next;
            steps += 1;
        }
        false
    };
    out.push((
        "the weapons said to fire are the ones whose chains fire",
        fires(Weapon::Pistol)
            && fires(Weapon::Chaingun)
            && Psprite::armed(Weapon::Pistol)
            && Psprite::armed(Weapon::Chaingun)
            && !Psprite::armed(Weapon::Shotgun)
            && !fires(Weapon::Shotgun)
            && !fires(Weapon::Bfg),
    ));

    // A pistol with no bullets is not loaded, and a fist always is.
    let mut st = Status::new();
    st.ammo[super::player::Ammo::Clip as usize] = 0;
    out.push((
        "an empty pistol is not loaded and a fist always is",
        !Psprite::loaded(Weapon::Pistol, &st) && Psprite::loaded(Weapon::Fist, &st),
    ));

    out
}

//! A thing, as an object rather than as a picture.
//!
//! Adapted from room4doom's `gameplay/src/thing/mod.rs`, cut to the part that
//! can be exercised today: an object, the state it is in, and the clock that
//! moves it to the next one. Momentum, the blockmap links, damage and the
//! action functions are upstream's and are not here yet -- there is nothing to
//! shoot and nothing to be shot by.
//!
//! ### Why an object has a state at all, when a tic would do
//!
//! It did not, until this file. `Anim` played a kind's spawn cycle as a pure
//! function of the world tic, which draws every barrel in the map correctly
//! and is *only* correct while every object of a kind stays in phase with
//! every other. That holds exactly as long as nothing joins the level late and
//! nothing leaves its cycle -- so it held, and it stops holding the moment
//! anything can be hurt, since a monster in pain is a monster whose animation
//! no longer agrees with its neighbour's.
//!
//! Moving the phase into the object is therefore not a refactor for its own
//! sake. It is the difference between an animation and a state machine, and
//! the state machine is what the next two phases hang off.
//!
//! ### The zero-tic loop, and the guard DOOM does not have
//!
//! A state with `tics == 0` is not a state that shows for no time. It is a
//! state that runs its action and *falls straight through* to the next one,
//! which is how DOOM writes a piece of logic into an animation. So setting a
//! state is a loop, not an assignment.
//!
//! Upstream's loop is unbounded, as id's was, on the reasonable grounds that
//! the shipped table has no zero-tic cycle in it. This one is bounded, because
//! the table here is *generated* and a generator is a thing that can be wrong:
//! there is no unwinder in this kernel and no watchdog, so an unbounded walk
//! over a bad chain is a machine that stops with no message at all. The bound
//! is the one failure mode this file can afford.

use alloc::vec::Vec;

use super::info;

/// How many zero-tic states may chain before the walk is called broken.
///
/// Generously above anything DOOM does -- the longest real run of them is a
/// handful -- because this is a backstop against a malformed table and not a
/// rule about content.
const CHAIN_CAP: usize = 64;

/// One thing on the level: where it is, what it is, and what it is doing.
pub struct Obj {
    /// Which row of `info::KINDS` this is.
    pub kind: u16,
    /// The state it is in.
    pub state: u16,
    /// Tics left before it leaves that state. `-1` never advances, which is
    /// every item and every piece of scenery.
    pub tics: i16,
    pub x: f32,
    pub y: f32,
    /// The way it is pointing, in radians.
    pub angle: f32,
    /// The sector it stands in, rather than the height and light that sector
    /// had when the level loaded.
    ///
    /// It was a copied `z` and `light`, which is correct exactly as long as no
    /// floor ever moves. A lift raising a sector would have left every thing
    /// standing in it hanging in the air at its load-time height, lit by a
    /// room that had since gone dark -- and nothing would have looked broken
    /// enough to investigate, because a barrel at the wrong height is still a
    /// barrel. An index cannot go stale.
    pub sector: usize,
    /// What is left of it.
    pub health: i16,
    /// Its own flags, copied from the kind at spawn.
    ///
    /// Its **own**, not the kind's, because dying changes them: a corpse is no
    /// longer shootable and no longer solid, and reading them off the shared
    /// row would make one barrel's death take every barrel's out of the world
    /// with it.
    pub flags: u32,
    /// Its own height, for the same reason. A corpse is a quarter as tall,
    /// which is what lets you walk over one.
    pub height: i16,
    /// Which of the eight compass directions it is walking in, or `NO_DIR`.
    pub movedir: u8,
    /// How many more tics to keep walking that way before choosing again.
    pub movecount: i32,
    /// Tics before it will consider shooting. Counted down by `A_Chase`, and
    /// it is why a monster that has just noticed you does not fire instantly.
    pub reaction: i16,
    /// How long it stays interested in its current target.
    pub threshold: i32,
    /// Whether it has noticed the player. See `enemy.rs` for why this is a
    /// flag and not a pointer.
    pub target: bool,
    /// Marked for removal at the end of the tic.
    ///
    /// Not removed where it is decided, which is the whole point: the caller
    /// dispatches actions by index, and a list that shrank underneath it would
    /// hand every later action to the wrong object.
    pub remove: bool,
}

impl Obj {
    /// A thing of this kind, standing here.
    ///
    /// Answers nothing for a kind whose spawn state is `S_NULL`, which is a
    /// real row rather than a missing one -- a teleport destination is an
    /// object whose whole job is to be a marker.
    pub fn spawn(kind: u16, x: f32, y: f32, angle: f32, sector: usize) -> Option<Obj> {
        let k = info::KINDS.get(kind as usize)?;
        if k.spawn == 0 {
            return None;
        }
        let mut o = Obj {
            kind,
            state: 0,
            tics: 0,
            x,
            y,
            angle,
            sector,
            health: k.health,
            flags: k.flags,
            height: k.height,
            movedir: super::enemy::NO_DIR,
            movecount: 0,
            reaction: k.reaction,
            threshold: 0,
            target: false,
            remove: false,
        };
        let mut fired = Vec::new();
        o.set_state(k.spawn, &mut fired).then_some(o)
    }

    /// Take damage. True means this killed it.
    ///
    /// DOOM's `P_DamageMobj`, without the thrust -- that needs momentum, and
    /// nothing here carries any.
    ///
    /// The **pain state** is here now that there is a random table to ask.
    /// `painchance` is a threshold out of 255 rather than a probability out of
    /// 100: a zombieman is 200, so it flinches about four times in five, and a
    /// baron of hell is 50. Getting the comparison the wrong way round gives
    /// monsters that flinch when they should not, which reads as them being
    /// easy rather than as a bug.
    ///
    /// The draw happens **whether or not** the thing has a pain state, because
    /// what matters is the index: DOOM spends the byte and then decides, so a
    /// version that checked first would consume a different number of bytes
    /// and every later roll in the game would differ.
    pub fn hurt(&mut self, amount: i32, fired: &mut Vec<u8>) -> bool {
        if self.flags & info::MF_SHOOTABLE == 0 || self.health <= 0 {
            return false;
        }
        self.health = self.health.saturating_sub(amount as i16);
        if self.health <= 0 {
            self.kill(fired);
            return true;
        }
        let chance = self.row().map(|k| k.painchance).unwrap_or(0);
        let roll = super::rng::p_random();
        if roll < chance as i32 && self.flags & info::MF_SKULLFLY == 0 {
            self.flags |= info::MF_JUSTHIT;
            if let Some(pain) = self.row().map(|k| k.pain).filter(|p| *p != 0) {
                self.set_state(pain, fired);
            }
        }
        false
    }

    /// DOOM's `P_KillMobj`, less the item a monster drops.
    fn kill(&mut self, fired: &mut Vec<u8>) {
        let Some(k) = self.row() else { return };
        // A corpse is not shot at again, does not block, and does not float.
        self.flags &= !(info::MF_SHOOTABLE | info::MF_SOLID | info::MF_FLOAT);
        self.flags |= info::MF_CORPSE | info::MF_DROPOFF;
        // A quarter as tall, which is what lets you walk over one.
        self.height /= 4;
        // Overkill has its own animation where there is one. The test is
        // against *negative* spawn health, so it takes twice the thing's
        // health in one hit -- a barrel has no gib state and takes the
        // ordinary one however hard it is hit.
        let state = if self.health < -k.health && k.xdeath != 0 {
            k.xdeath
        } else {
            k.death
        };
        self.set_state(state, fired);
    }

    /// DOOM's `P_SetMobjState`. False means the thing has reached `S_NULL` and
    /// should be taken off the level.
    ///
    /// The action functions are not called, because there are none yet. Where
    /// they go is here, between reading the state and following `next` --
    /// which is the ordering that matters, since an action may itself set a
    /// state and the loop has to see the one it left behind.
    pub fn set_state(&mut self, state: u16, fired: &mut Vec<u8>) -> bool {
        let mut next = state;
        for _ in 0..CHAIN_CAP {
            if next == 0 {
                self.state = 0;
                return false;
            }
            let Some(st) = info::STATES.get(next as usize) else {
                self.state = 0;
                return false;
            };
            self.state = next;
            self.tics = st.tics;
            // The action of the state just entered, recorded rather than
            // called. An action needs the level, the other objects and the
            // player, and reaching all three from here would put the world
            // inside the state machine -- which is precisely what keeps this
            // file checkable at boot without a map. The caller dispatches.
            if st.action != 0 {
                fired.push(st.action);
            }
            next = st.next;
            if self.tics != 0 {
                return true;
            }
        }
        // A cycle of zero-tic states. Stopping here leaves the thing in a
        // legal state showing a real picture, which is the least wrong answer
        // available and much better than not returning.
        true
    }

    /// One tic of this object's own clock.
    pub fn tick(&mut self, fired: &mut Vec<u8>) -> bool {
        if self.tics == -1 {
            return true;
        }
        self.tics -= 1;
        if self.tics <= 0 {
            let next = info::STATES.get(self.state as usize).map(|s| s.next).unwrap_or(0);
            return self.set_state(next, fired);
        }
        true
    }
}

impl Obj {
    /// The row of `info::KINDS` this is, if it is one.
    pub fn row(&self) -> Option<&'static info::Kind> {
        info::KINDS.get(self.kind as usize)
    }

    /// Its flags, which are its own and not the kind's once it has died.
    pub fn flags(&self) -> u32 {
        self.flags
    }

    /// Whether it has been killed.
    pub fn dead(&self) -> bool {
        self.flags & info::MF_CORPSE != 0
    }

    /// Whether walking over it picks it up.
    pub fn special(&self) -> bool {
        self.flags() & info::MF_SPECIAL != 0
    }

    /// Whether it can be walked through.
    pub fn solid(&self) -> bool {
        self.flags() & info::MF_SOLID != 0
    }

    /// Whether it can be shot.
    pub fn shootable(&self) -> bool {
        self.flags() & info::MF_SHOOTABLE != 0
    }

    /// Whether it hangs from the ceiling rather than standing on the floor.
    ///
    /// A hanging corpse, and the reason a thing's height is not simply its
    /// sector's floor. Getting this wrong buries every one of them in the
    /// ground, which on a map with a room full of them is unmistakable and on
    /// a map with one is not.
    pub fn hangs(&self) -> bool {
        self.flags() & info::MF_SPAWNCEILING != 0
    }

    pub fn radius(&self) -> f32 {
        self.row().map(|k| k.radius as f32).unwrap_or(0.0)
    }

    pub fn height(&self) -> f32 {
        self.height as f32
    }

    /// How far it walks in one tic.
    ///
    /// Units per tic for anything that walks, which is everything that walks.
    /// The field is fixed point for missiles and this is not one -- see the
    /// note on `Kind::speed`.
    pub fn speed(&self) -> f32 {
        self.row().map(|k| k.speed as f32).unwrap_or(0.0)
    }

    /// Whether it is a monster that counts toward the kill total.
    pub fn monster(&self) -> bool {
        self.flags & info::MF_COUNTKILL != 0
    }

    /// Where its feet are, looked up in the level.
    pub fn z_floor(&self, lv: &super::level::Level) -> f32 {
        match lv.sectors.get(self.sector) {
            Some(sec) => self.z(sec.floor as f32, sec.ceiling as f32),
            None => 0.0,
        }
    }

    /// The four-character sprite name the state it is in wears.
    pub fn sprite(&self) -> Option<&'static str> {
        info::frame_of(self.state).map(|(n, _)| n)
    }

    /// Where its feet are, given the sector it stands in.
    ///
    /// Takes the two heights rather than the level, so this file needs to know
    /// nothing about maps -- which is what keeps the state machine testable
    /// without one.
    pub fn z(&self, floor: f32, ceiling: f32) -> f32 {
        if self.hangs() {
            ceiling - self.height()
        } else {
            floor
        }
    }
}

/// An action that fired, by whom, and where.
///
/// **Both** the index and the position, because the two actions that exist
/// want different things and neither substitutes for the other: a monster's
/// `A_Chase` has to move the object, which needs the index, and a barrel's
/// `A_Explode` happens at a place the object is about to leave, which needs
/// the position.
///
/// The index is only sound because the sweep is deferred -- see `Objs::tick`.
pub struct Fired {
    pub action: u8,
    pub who: usize,
    pub x: f32,
    pub y: f32,
}

/// Every object on the level.
///
/// A `Vec` and a sweep, which is the same adaptation `Thinkers` makes of
/// DOOM's intrusive list and for the same reason: one core, one address space,
/// and none of the pointer discipline that list exists to provide.
pub struct Objs {
    pub list: Vec<Obj>,
}

impl Objs {
    pub fn new() -> Objs {
        Objs { list: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Advance every object by one tic and report every action that fired.
    ///
    /// **Nothing is removed here**, and that is the design rather than an
    /// oversight. Actions are dispatched by index after this returns, and the
    /// first version removed as it went -- so an index handed out early in a
    /// tic named a different object by the end of it, and the action most
    /// worth dispatching is fired by an object on its way off the level.
    /// Anything that ran out of states is *marked*, and `sweep` takes them
    /// once the caller has finished. DOOM removes its thinkers at the end of a
    /// tic for the same reason.
    pub fn tick(&mut self, out: &mut Vec<Fired>) {
        let mut scratch: Vec<u8> = Vec::new();
        for i in 0..self.list.len() {
            if self.list[i].remove {
                continue;
            }
            scratch.clear();
            let alive = self.list[i].tick(&mut scratch);
            for a in scratch.iter() {
                out.push(Fired {
                    action: *a,
                    who: i,
                    x: self.list[i].x,
                    y: self.list[i].y,
                });
            }
            if !alive {
                self.list[i].remove = true;
            }
        }
    }

    /// Take away everything marked during the tic.
    pub fn sweep(&mut self) {
        self.list.retain(|o| !o.remove);
    }

    /// How many of them have been killed.
    pub fn dead(&self) -> usize {
        self.list.iter().filter(|o| o.dead()).count()
    }

    /// How far the nearest live monster is from a point.
    ///
    /// The one number that says a monster *approached* rather than merely
    /// woke: it shoots from where it stood if `A_Chase` never moves it, and
    /// damage alone cannot tell those apart.
    pub fn nearest_monster(&self, x: f32, y: f32) -> f32 {
        let mut best = f32::INFINITY;
        for o in self.list.iter() {
            if !o.monster() || o.health <= 0 {
                continue;
            }
            let (dx, dy) = (o.x - x, o.y - y);
            let d = super::math::sqrt(dx * dx + dy * dy);
            if d < best {
                best = d;
            }
        }
        best
    }

    /// How many monsters are on the level, and how many have noticed you.
    pub fn monsters(&self) -> (usize, usize) {
        let live = self.list.iter().filter(|o| o.monster() && o.health > 0);
        (
            self.list.iter().filter(|o| o.monster()).count(),
            live.filter(|o| o.target).count(),
        )
    }

    /// The states every object is in, added up.
    ///
    /// A number that changes exactly when some object changes picture, which
    /// is what a run needs to report that animation is *live* rather than
    /// merely configured. Everything else can be right -- the chains walked,
    /// the durations read, the arithmetic asserted -- and the clock never
    /// reach the objects, which then show their spawn frame forever and look
    /// exactly like a level whose barrels happen not to move.
    ///
    /// A sum rather than a hash: two objects advancing in the same tic can
    /// cancel in principle, so this undercounts and never overcounts, which is
    /// the right way round for a figure whose whole job is to be non-zero.
    pub fn phase(&self) -> usize {
        self.list.iter().map(|o| o.state as usize).sum()
    }
}

impl Default for Objs {
    fn default() -> Objs {
        Objs::new()
    }
}

/// A thing standing nowhere, for a check that only cares about its states.
fn bare() -> Obj {
    Obj {
        kind: 0,
        state: 0,
        tics: 0,
        x: 0.0,
        y: 0.0,
        angle: 0.0,
        sector: 0,
        health: 0,
        flags: 0,
        height: 0,
        movedir: super::enemy::NO_DIR,
        movecount: 0,
        reaction: 0,
        threshold: 0,
        target: false,
        remove: false,
    }
}

/// What `diag doom` asks of the state machine.
///
/// Arithmetic over a generated table, so none of it needs a WAD and all of it
/// runs at boot -- which is the point: a state machine that stalls looks
/// exactly like content that does not move, and one that runs away looks like
/// a machine that has stopped.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    // A zombieman idles between two states of ten tics each. Twenty tics of
    // its own clock therefore returns it to where it started and ten does not,
    // which is the whole of what a state machine has to get right.
    let mut z = bare();
    let mut fired: Vec<u8> = Vec::new();
    let alive = z.set_state(info::kind_of(3004).map(|k| k.spawn).unwrap_or(0), &mut fired);
    let first = z.state;
    for _ in 0..10 {
        z.tick(&mut fired);
    }
    let after10 = z.state;
    for _ in 0..10 {
        z.tick(&mut fired);
    }
    out.push((
        "an idle monster returns to its first state after one cycle",
        alive && first != 0 && after10 != first && z.state == first,
    ));

    // A thing whose state never advances stays in it however long the level
    // runs. Every item in the game is this, so a `tics` of -1 read as an
    // ordinary countdown would march the entire contents of a map through
    // their death animations in the first second.
    let mut m = bare();
    m.set_state(info::kind_of(2012).map(|k| k.spawn).unwrap_or(0), &mut fired);
    let held = m.state;
    for _ in 0..1000 {
        m.tick(&mut fired);
    }
    out.push((
        "an item never leaves its spawn state",
        held != 0 && m.tics == -1 && m.state == held,
    ));

    // `S_NULL` takes a thing off the level rather than drawing it, and the
    // caller has to honour the return value. Nothing about the object looks
    // wrong afterwards, which is why this is asserted rather than trusted.
    out.push((
        "a thing set to state zero is removed",
        !bare().set_state(0, &mut fired),
    ));

    // A kind that spawns into `S_NULL` is not an object at all. A teleport
    // destination is exactly this: a real row with a real doomednum.
    out.push((
        "a marker is a kind that spawns into nothing",
        info::row_of(14).is_some_and(|r| Obj::spawn(r, 0.0, 0.0, 0.0, 0).is_none()),
    ));

    // And one that does not is. A barrel spawns, is alive, and has health to
    // take away later.
    out.push((
        "a barrel spawns with health and a state",
        info::row_of(2035)
            .and_then(|r| Obj::spawn(r, 0.0, 0.0, 0.0, 0))
            .is_some_and(|o| o.state != 0 && o.health > 0 && o.tics > 0),
    ));

    out
}

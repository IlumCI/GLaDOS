//! DOOM's randomness, which is not random.
//!
//! `P_Random` reads the next byte of a fixed 256-entry table and advances an
//! index. That is the whole of it. There is no seed, no entropy, no state
//! beyond one number that wraps -- and every copy of DOOM ever shipped draws
//! from the same bytes in the same order.
//!
//! ### Why this is better here than a real generator
//!
//! It was deferred twice while the port grew, with the note that inventing a
//! sequence would produce a game that plays differently from every other copy
//! and would say so nowhere. That is the correctness argument. The practical
//! one is larger: **a fixed table is what makes a run repeatable**, and this
//! whole tree is tested by driving a headless machine and reading numbers off
//! the end of it. A real generator would make "the zombieman fired twice in
//! four seconds" a fact about this boot rather than about the code.
//!
//! So `reset` exists, and every scripted run calls it. Two runs of the same
//! script produce the same shots, the same pain, the same monster wandering
//! into the same corner.
//!
//! ### One index, and it is global on purpose
//!
//! DOOM keeps `prndindex` as a file-static and reaches it from everywhere,
//! because the *order of draws* is part of the game's behaviour: a monster
//! deciding whether to flinch consumes a byte that the next damage roll then
//! does not get. Threading a generator through every call site would preserve
//! the values and destroy the order, which is the half that matters.
//!
//! `AtomicUsize` rather than a `Cell`, for the reason upstream gives: it is
//! sound interior mutability without `static mut`, and there is no contention
//! to pay for because there is one core running the game.
//!
//! DOOM has a *second* index, `rndindex`, drawn by `M_Random` for things
//! outside the simulation -- menu effects, and the demo-recording code. It is
//! separate precisely so that non-gameplay draws cannot shift the gameplay
//! sequence. Nothing here draws from it, so it is not carried: a second index
//! that nothing advances is a second index that cannot desynchronise anything.

use core::sync::atomic::{AtomicUsize, Ordering};

use super::info::RNDTABLE;

static INDEX: AtomicUsize = AtomicUsize::new(0);

/// The next byte. DOOM's `P_Random`.
pub fn p_random() -> i32 {
    let i = (INDEX.load(Ordering::Relaxed) + 1) & 0xFF;
    INDEX.store(i, Ordering::Relaxed);
    RNDTABLE[i] as i32
}

/// The difference of two draws, which is DOOM's idiom for a signed spread.
///
/// Written out because it appears everywhere -- aim jitter, damage, the
/// direction a monster flinches -- and because the *two* draws matter: the
/// index advances twice, so replacing this with one draw and a sign would
/// shift every later number in the game.
pub fn p_random_signed() -> i32 {
    p_random() - p_random()
}

/// Put the sequence back to the start.
///
/// Called at the top of a scripted run so the run is reproducible. DOOM does
/// this too, at the start of a level, for the same reason a demo needs it.
pub fn reset() {
    INDEX.store(0, Ordering::Relaxed);
}

/// Where the index stands, so a run can report how much randomness it spent.
///
/// A number worth printing: two runs of one script that disagree about it have
/// taken different paths through the game, and that is visible here before it
/// is visible anywhere else.
pub fn spent() -> usize {
    INDEX.load(Ordering::Relaxed)
}

/// What `diag doom` asks of the table.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out: alloc::vec::Vec<(&'static str, bool)> = alloc::vec::Vec::new();

    // The table is id's, and these are its first and last bytes. A generator
    // that emitted a table of the right shape and the wrong contents would
    // produce a game that runs perfectly and plays differently from every
    // other copy of DOOM, which is exactly the failure worth an assertion.
    out.push((
        "the random table is DOOM's own",
        RNDTABLE.len() == 256
            && RNDTABLE[0] == 0
            && RNDTABLE[1] == 8
            && RNDTABLE[2] == 109
            && RNDTABLE[255] == 249,
    ));

    // The first draw is the *second* entry, not the first: the index advances
    // before it reads. Off by one here shifts the whole game by one byte.
    reset();
    let first = p_random();
    let second = p_random();
    out.push((
        "the first draw is the table's second byte",
        first == 8 && second == 109,
    ));

    // It wraps, and wrapping returns to the same place. 256 draws from a reset
    // index land back where they started.
    reset();
    for _ in 0..256 {
        p_random();
    }
    out.push(("256 draws wrap to the start", spent() == 0));

    // Reset makes a run repeatable, which is what the harness rests on.
    reset();
    let a: alloc::vec::Vec<i32> = (0..8).map(|_| p_random()).collect();
    reset();
    let b: alloc::vec::Vec<i32> = (0..8).map(|_| p_random()).collect();
    out.push(("a reset sequence repeats exactly", a == b));

    // The signed spread draws twice. Checking the *cost* and not the value,
    // because what a caller replacing it with one draw would break is the
    // index, and the values would look perfectly reasonable either way.
    reset();
    p_random_signed();
    out.push(("a signed draw spends two bytes", spent() == 2));

    // Leave the sequence where a run expects to find it rather than wherever
    // these checks happened to stop.
    reset();
    out
}

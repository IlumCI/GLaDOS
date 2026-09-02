//! What a program written somewhere else may ask of GLaDOS.
//!
//! This is the seam. Everything under `src/doom/` -- and everything ported
//! after it -- reaches the machine through this module and through nothing
//! else, and that rule is checked rather than intended: `tools/portcheck.py`
//! scans for a `crate::` path under a ported tree that is not `crate::port`
//! and fails. There is no `build.rs` in this repository and there cannot be
//! one, so the check runs beside the build rather than inside it.
//!
//! ### Why a seam at all, for one consumer
//!
//! Normally this tree would refuse to build an interface before there are two
//! implementations of it -- inventing an abstraction for a user who does not
//! exist is how a codebase acquires layers nobody wanted. The exception is
//! made deliberately and for one reason: **the point of the first port is to
//! find out where the boundary is.** A port that reaches into `gfx`, `kbd`,
//! `sysbox` and `time` wherever it happens to need them is not a port, it is a
//! merge, and the second one starts from nothing.
//!
//! So the interface is small, it is shaped only by what the first consumer
//! actually needs, and it deliberately does *not* anticipate the next one.
//! What comes after DOOM is undecided -- a script interpreter and a binary
//! loader want very different things from a host -- and guessing would produce
//! exactly the speculative layer this paragraph opened by refusing.
//!
//! ### What is deliberately absent
//!
//! No sound: there is no audio driver anywhere in this kernel, and the one
//! reference to the PC speaker keeps it disconnected on purpose. No
//! networking, no threads, no memory protection -- a ported program runs in
//! ring 0 in the one address space, with the operator's own powers, because
//! that is what this machine is. A port is trusted code, and the trust is not
//! enforced by anything. That is a statement of fact rather than a design.

use alloc::vec::Vec;

pub mod clock;
pub mod files;
pub mod keys;
pub mod surface;

pub use clock::now_us;
pub use surface::Surface;

/// The name a ported program is known by, for diagnostics.
///
/// Not an identity, not a handle -- the desktop has no notion of a foreign
/// application and this does not invent one. It is what `port` prints when it
/// has something to say about who asked.
pub struct Program {
    pub name: &'static str,
}

impl Program {
    pub const fn new(name: &'static str) -> Program {
        Program { name }
    }
}

/// Take the screen for the duration of a call, and give it back afterwards
/// however the call leaves.
///
/// A full-screen program does not get the machine to itself just by drawing
/// over it. `desk::paint_clock` runs on the clock task at 10 Hz and
/// `desk::move_cursor` runs on whichever task is generating; both write the
/// framebuffer, and both take a paint claim that is private to `desk.rs` and
/// therefore unavailable here. Without this the clock stamps itself over a
/// game frame ten times a second.
///
/// `edit::run` has owned the screen the same way since it was written and has
/// this defect today -- which is how it was found.
pub fn with_screen<R>(f: impl FnOnce() -> R) -> R {
    crate::gfx::set_exclusive(true);
    // Blanked on the way in. A frame that does not divide the screen exactly
    // is letterboxed, and what shows in the letterbox is whatever the desktop
    // last composed there -- so a 320x200 frame on a 1920x1080 screen came up
    // framed by half a Program Manager and a column of icons.
    //
    // It also makes the flag above testable: against black, a clock that
    // failed to stand down is four digits in the corner rather than something
    // indistinguishable from the desktop it was already sitting on.
    if let Some(fb) = crate::gfx::primary() {
        fb.fill(crate::gfx::Color::new(0, 0, 0));
    }
    let out = f();
    crate::gfx::set_exclusive(false);
    // Whatever was underneath, put back -- and the shadow forgotten first,
    // which the comment below always said was necessary and the code did not
    // do. A `Surface` writes the aperture directly, so the compositor still
    // believes the desktop is on screen; `present` then finds every row
    // unchanged and repaints none of them, and the game stays on screen with
    // the terminal drawn over the top of it. Found by looking at a
    // screenshot taken after the picture was supposed to be gone.
    crate::gfx::compose::invalidate();
    crate::gfx::desk::draw();
    out
}

/// Wait until something happens.
///
/// A game loop that has drawn its frame and is waiting for the next tic should
/// not spin: this machine has other tasks -- the resident mind among them --
/// and a busy wait starves them for the whole of a session. `hlt` parks the
/// core until the next interrupt, and the 100 Hz timer guarantees one arrives.
///
/// A host service rather than something a ported program does for itself,
/// because halting a CPU is exactly the kind of thing only the machine should
/// decide it is safe to do.
pub fn idle() {
    unsafe { core::arch::asm!("hlt", options(nomem, nostack)) };
}

/// A scratch buffer sized once and reused, because a ported program allocating
/// per frame is a ported program that stutters.
pub(crate) fn sized<T: Clone + Default>(v: &mut Vec<T>, n: usize) {
    if v.len() != n {
        v.clear();
        v.resize(n, T::default());
    }
}

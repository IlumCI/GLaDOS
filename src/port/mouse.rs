//! How far the hand moved, and which buttons are down.
//!
//! The fifth thing in this seam, and the first one added because a *port*
//! asked for it rather than because the shape looked incomplete. DOOM turns
//! with the mouse; nothing else here ever needed a pointer, because the
//! desktop has a cursor and a cursor is a different quantity.
//!
//! ### Relative, and that is the whole reason this file exists
//!
//! `dev::mouse` tracks a cursor, so its position is **clamped to the screen**.
//! That is correct for a pointer and fatal for a player: turning right would
//! work until the notional cursor reached the right edge of the display, and
//! then stop -- a player who can face east and cannot keep going. So this
//! reads the unclamped accumulator instead, and drains it, because what it
//! reports is how far the hand moved since the last look and not where
//! anything is.
//!
//! ### Why there is no cursor here, and no absolute position
//!
//! A ported program owns the whole screen (`gfx::exclusive`), so there is no
//! cursor to place and nothing for an absolute coordinate to mean. Exposing
//! one would be exposing the desktop's idea of a pointer to a program that has
//! taken the desktop's screen away, which is a number that is always wrong.
//!
//! ### The typed equivalent
//!
//! `mouse move` and `mouse click` in the shell. That is not a convenience: the
//! serial line this machine is tested over cannot inject PS/2 packets, so a
//! capability reachable only by hand is one nothing ever checks -- which is
//! the argument `win keys` already makes for the keyboard, and the reason
//! `port bars` has a bounded form.

use crate::dev::mouse;

/// How far the hand moved since the last call, in mouse counts.
///
/// `dy` counts **up** away from the user, which is the sign the hardware
/// sends. The desktop negates it on the way to a screen coordinate, where down
/// is positive; nothing here does, because a program turning or looking wants
/// the hardware's own sense and inventing a second convention at the seam
/// would mean two places to be wrong.
pub fn motion() -> (i32, i32) {
    mouse::take_relative()
}

/// Which buttons are down, left and right.
///
/// Sampled rather than drained: a button is a *state*, like a held key, and a
/// program asking twice in one frame must get the same answer both times. That
/// is why this does not go through `mouse::take`, which clears the wheel and
/// the moved flag and would swallow a click the desktop has not seen.
pub fn buttons() -> (bool, bool) {
    let s = mouse::peek();
    (s.left, s.right)
}

/// Whether there is a pointer at all.
///
/// Worth asking rather than assuming: on the GF63 the touchpad answers, and
/// under QEMU it depends on what the command line asked for. A program that
/// turned with the mouse unconditionally would be a program that cannot be
/// steered at all where there is none.
pub fn present() -> bool {
    mouse::present()
}

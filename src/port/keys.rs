//! Which keys are held, for a program that needs a state and not a stream.
//!
//! The kernel's keyboard ring is a byte queue and says so in its own doc:
//! keys with no ASCII form arrive as bytes above 0x7F precisely so it never
//! has to grow an event type. That is the right shape for a shell and it
//! cannot answer "is forward held", which is the only question a game asks.
//!
//! So this reads the down-map beside the ring rather than the ring itself. The
//! map is filled in `kbd::decode`, which is the single decoder shared by the
//! PS/2 interrupt and the USB-HID path, so both hands feed it without either
//! knowing this module exists.
//!
//! **Set-1 make codes, not characters.** A program that wants WASD wants the
//! physical keys, and asking for `'w'` would give a different answer under a
//! different layout -- and no answer at all while shift is held, since the
//! character would be `'W'`.

use crate::dev::kbd;

/// The keys a ported program is likely to name, so it does not carry a table
/// of magic numbers. Set-1 make codes.
pub const W: u8 = kbd::SC_W;
pub const A: u8 = kbd::SC_A;
pub const S: u8 = kbd::SC_S;
pub const D: u8 = kbd::SC_D;
pub const SPACE: u8 = kbd::SC_SPACE;
pub const ESC: u8 = kbd::SC_ESC;
pub const LEFT: u8 = kbd::SC_LEFT;
pub const RIGHT: u8 = kbd::SC_RIGHT;
pub const UP: u8 = kbd::SC_UP;
pub const DOWN: u8 = kbd::SC_DOWN;
pub const CTRL: u8 = kbd::SC_LCTRL;
pub const SHIFT: u8 = kbd::SC_LSHIFT;

/// Is that key held right now?
pub fn down(code: u8) -> bool {
    kbd::is_down(code)
}

/// Every held key at one instant.
///
/// A program reading four keys to decide one movement vector should read them
/// from one moment, not from four -- otherwise a key released between the
/// second and the third read gives a frame that never happened.
pub fn snapshot() -> Held {
    let (lo, hi) = kbd::down_snapshot();
    Held { lo, hi }
}

/// A frozen copy of the down-map.
#[derive(Clone, Copy)]
pub struct Held {
    lo: u64,
    hi: u64,
}

impl Held {
    pub fn down(&self, code: u8) -> bool {
        if code < 64 {
            self.lo & (1u64 << code) != 0
        } else {
            self.hi & (1u64 << (code - 64)) != 0
        }
    }

    /// Nothing held. Distinguishable from "the keyboard is idle" only in that
    /// a program can act on it -- a paused game wants to know.
    pub fn is_empty(&self) -> bool {
        self.lo == 0 && self.hi == 0
    }
}

/// Hold or release a key without a keyboard.
///
/// The typed equivalent, which this tree requires of everything: serial can
/// send a keystroke but not a *held* one, because there is no make without a
/// break, so a program that reads held keys is one no harness could otherwise
/// drive. It goes through the same map the interrupt fills, so the scripted
/// path and the real one cannot disagree about what "held" means.
pub fn force(code: u8, down: bool) {
    kbd::force_down(code, down);
}

/// Forget every held key.
///
/// For the moment a program takes the keyboard and the moment it gives it
/// back. A key held when the game started is not held by the game, and a key
/// held when it exits must not still be held by the shell -- the release went
/// to whoever was reading at the time, and after a mode change that is nobody.
pub fn clear() {
    kbd::clear_down();
}

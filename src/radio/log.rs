//! A way for a ported tree to say something.
//!
//! `src/doom/mod.rs` records the cost of not having this: `kprintln!` lives at
//! the crate root, a ported tree may not name the crate root, so that tree
//! cannot print at all and its WAD parser carries an error type instead of
//! diagnostics. That was the right trade for a game and it is the wrong one
//! for a network stack, where the interesting failures are transient, happen
//! once, and are invisible afterwards -- an association that was refused with
//! a status code nobody saw is indistinguishable from one that never happened.
//!
//! So: three levels, and a switch. Ported code calls these; whether anything
//! reaches the console is the machine's decision, not the port's.

use core::sync::atomic::{AtomicU8, Ordering};

/// 0 quiet, 1 errors, 2 events, 3 frames.
///
/// Frames is genuinely unusable as a default -- a busy channel is hundreds of
/// beacons a second and the console is 128 columns -- so it exists for a
/// person who has asked for it and is watching.
static LEVEL: AtomicU8 = AtomicU8::new(1);

pub fn set_level(n: u8) {
    LEVEL.store(n.min(3), Ordering::Relaxed);
}

pub fn level() -> u8 {
    LEVEL.load(Ordering::Relaxed)
}

/// Something went wrong and somebody should see it.
pub fn error(what: &str) {
    if level() >= 1 {
        crate::kprintln!("[wlan] {}", what);
    }
}

/// A state change worth a line: scanned, authenticated, associated, keyed.
pub fn event(what: &str) {
    if level() >= 2 {
        crate::kprintln!("[wlan] {}", what);
    }
}

/// Per-frame detail. Off unless asked for.
pub fn frame(what: &str) {
    if level() >= 3 {
        crate::kprintln!("[wlan] {}", what);
    }
}

//! What an 802.11 stack written somewhere else may ask of GLaDOS.
//!
//! This is the second seam. Everything under `src/wlan/` -- the transliterated
//! OpenBSD `net80211` and the drivers under it -- reaches the machine through
//! this module and through nothing else, and `tools/portcheck.py` checks that
//! rather than trusting it.
//!
//! ### Why a second seam instead of widening the first
//!
//! `crate::port` is what a *program* asks of a machine: a screen, held keys,
//! relative pointer motion, a clock, the bytes of a file. A wireless stack
//! wants none of that. It wants a millisecond clock, a scheduler it can arm a
//! timeout against, entropy, a block cipher, and -- for the drivers -- config
//! space, a mapped BAR and memory the device can reach. The overlap is the
//! clock and nothing else.
//!
//! Widening `port` to cover both would have put a DMA allocator beside a
//! keyboard scancode table and called the result an interface. That is the
//! merge `port` exists to prevent, arriving through `port` itself.
//!
//! ### Why this one is allowed to be designed rather than discovered
//!
//! `port/mod.rs` argues that a seam for one consumer is speculative, and it is
//! right. This one has several from the outset -- the core, and every driver
//! behind it -- which is the condition that module names as the exception. It
//! is still shaped only by what those consumers actually need: there is no
//! method here that nothing calls.
//!
//! ### What is deliberately absent
//!
//! No mbuf chain. OpenBSD passes `struct mbuf *` everywhere and this passes
//! owned byte vectors, because `iface::Nic` already hands whole frames and a
//! chain exists to avoid copies this kernel is not yet fast enough to notice.
//! That is a real deviation from upstream and it touches every ported
//! signature, so it is stated here rather than discovered file by file.
//!
//! No interrupts. Every driver in this kernel polls, and `dev/e1000.rs` argues
//! that is a correctness property rather than a gap. A radio driver here polls
//! too, and pays for it in throughput.

// The seam is complete before the tree that consumes it, so every item here
// is currently unused and the build would carry forty warnings saying so.
// `src/doom/info.rs` states the objection to that exactly: "a build with 140
// warnings in it is a build where the one that matters is not read." The
// alternative -- adding methods one at a time as the port reaches them -- is
// worse, because then the shape of the seam is decided by whatever file is
// being transliterated on a given afternoon rather than designed.
#![allow(dead_code)]

use alloc::vec::Vec;

pub mod bus;
pub mod cipher;
pub mod clock;
pub mod entropy;
pub mod log;

#[allow(unused_imports)]
pub use clock::{now_ms, now_us};

/// A frame, owned, as it travels between the stack and a driver.
///
/// A type alias rather than a newtype on purpose: it is exactly a `Vec<u8>`,
/// every layer agrees on that, and wrapping it would buy nothing but a
/// conversion at each boundary.
pub type Frame = Vec<u8>;

/// A MAC address. The same six bytes `net::Mac` is, spelled again here because
/// a ported tree may not name `crate::net`.
pub type Mac = [u8; 6];

/// The broadcast address, which appears in enough ported code to be worth a
/// name rather than six literals.
pub const BROADCAST: Mac = [0xff; 6];

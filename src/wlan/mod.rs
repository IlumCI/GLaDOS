//! The 802.11 stack, ported.
//!
//! Written for GLaDOS.
//!
//! This file is the tree's root and its argument; everything it declares
//! came from somewhere else and says so.
//!
//! **This tree is not ours.** It is transliterated from OpenBSD's
//! `sys/net80211`, 24,665 lines across 31 files, by Atsushi Onoe, Sam Leffler,
//! Damien Bergamini, David Young, Reyk Floeter, Christian Ehrhardt, Stefan
//! Sperling and Theo Buehler, under ISC, BSD-2-Clause and BSD-3-Clause. Every
//! file says at the top what it came from, which version, who holds the
//! copyright and under what terms, and `tools/wlanprov.py` fails the build
//! rather than trusting that they do. Full licence texts are in `licenses/`.
//!
//! ### Why ported rather than written
//!
//! `src/net/ieee80211.rs` opens by arguing the opposite -- that a beacon has
//! the same shape coming out of every access point ever built, so it can be
//! written from the specification. That argument is right about frames and
//! wrong about scope. It produced 694 lines across two files, both correct,
//! neither able to associate with anything, because the volume in 802.11 is
//! not the frame layouts: it is the MLME state machine, the retry and timeout
//! behaviour, rate adaptation, the regulatory database, and the exact order in
//! which an access point expects to be spoken to. None of that gets smaller
//! for being written locally, and all of it is the part that is wrong in ways
//! no local test can find.
//!
//! OpenBSD specifically, out of four candidates, for one reason above the
//! others: **it runs the supplicant in the kernel.** Linux and FreeBSD both
//! put the WPA four-way handshake in a userland `wpa_supplicant`, and this
//! machine has no userland and no process to run one in. `ieee80211_pae_input.c`
//! is a station that can join a WPA2 network with nothing above the kernel,
//! which is the shape GLaDOS needs and the only mainstream stack that has it.
//!
//! ### The rule
//!
//! **Nothing in this tree may name `crate::` except `crate::radio`.**
//! `tools/portcheck.py` checks it. The cost is the same one `src/doom/mod.rs`
//! records -- this code cannot print, because `kprintln!` lives at the crate
//! root -- except that here it would be a much worse cost, since the
//! interesting failures in a network stack happen once and leave nothing
//! behind. So `radio::log` exists and is the only way out.
//!
//! ### Where this departs from upstream, at the level of the whole tree
//!
//! Three deviations touch every file, so they are stated once here rather than
//! repeated in each:
//!
//!   * **No mbuf chains.** Frames are owned `Vec<u8>`. `iface::Nic` already
//!     hands over whole frames, and a chain exists to avoid copies this kernel
//!     is not fast enough to notice.
//!   * **No `ifnet`.** Upstream's `struct ieee80211com` begins with a
//!     `struct arpcom`, and every driver casts between the three types freely.
//!     Rust will not express that and faking it with casts would be worse than
//!     the composition that replaces it.
//!   * **Station only.** Upstream's `IEEE80211_STA_ONLY` build is taken: no
//!     host AP, no IBSS. GLaDOS is a client.

// The tree is ahead of its consumer for the length of the port: nothing is
// wired to `iface::Nic` yet, so most of what is transliterated is reachable
// only from a selftest and the build would carry a warning per item saying so.
// `src/doom/info.rs` states the objection -- "a build with 140 warnings in it
// is a build where the one that matters is not read."
//
// This is a worse trade here than it is there, and the difference is worth
// naming: a generated table is *expected* to be mostly unused, while a ported
// stack has genuinely unreachable code in it -- an upstream path whose caller
// was not brought over is a real defect that this allow hides. So it comes off
// the day the stack reaches `Nic`, and that is a step in the port rather than
// a tidy-up to be remembered.
#![allow(dead_code)]

pub mod ccmp;
pub mod frame;

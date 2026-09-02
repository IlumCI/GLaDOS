//! DOOM, ported.
//!
//! **This tree is not ours.** It is adapted from
//! [room4doom](https://github.com/flukejones/room4doom) by Luke Jones, MIT
//! licensed, which is a software renderer rather than a GPU one and therefore
//! meets this machine where it is -- a linear framebuffer and nothing else.
//! Every file says at the top what it came from and what was changed.
//!
//! It is here to answer one question: **can software written somewhere else
//! run on GLaDOS?** Not whether a renderer can be written, which was never in
//! doubt, but whether a foreign program can be brought over without dissolving
//! into the kernel it now lives in. That is why the rule below exists, and why
//! it is checked by a script rather than remembered.
//!
//! ### The rule
//!
//! **Nothing in this tree may name `crate::` except `crate::port`.**
//!
//! `tools/portcheck.py` enforces it. A port that reaches into `gfx` for a
//! framebuffer, `kbd` for a key, `sysbox` for a file and `time` for a clock is
//! not a port -- it is a merge, and the next one starts from nothing. The
//! whole value of doing the first one carefully is that the second is cheap.
//!
//! ### What that costs, said plainly
//!
//! This code cannot print: the printing macro lives at the crate root and
//! naming it would cross the rule. A parser here returns structures and
//! whoever called it does the talking. That is a real
//! constraint and it is a good one: it is why the WAD parser has an `Error`
//! type with a `Display` rather than a scattering of diagnostic prints, and
//! the shell verb reads much better for it.
//!
//! ### Where it is going
//!
//! `wad` is the file format. `level`, `pic-data` and the software renderer
//! follow, in that order, each testable before the next exists. There is no
//! sound: this machine has no audio driver of any kind, and the one reference
//! to the PC speaker deliberately keeps it disconnected.
//!
//! The WAD itself is never in this repository. `DOOM1.WAD` belongs to id;
//! [FreeDoom](https://freedoom.github.io/) is the freely licensed one to test
//! against. Either is read off the boot volume at startup, by the same path
//! the model takes and for the same reason -- that is the only moment there is
//! a filesystem.

pub mod draw;
pub mod math;
pub mod render;
pub mod level;
pub mod wad;

/// The name the boot loader files a WAD under, and the name this module asks
/// for. One spelling, so a file that arrives cannot be a file nothing looks
/// for.
pub const WAD_FILE: &str = "doom.wad";

/// The WAD, if one was on the boot volume.
///
/// Parsed on every call rather than cached, and that is a deliberate choice
/// for now: the directory of a four-megabyte IWAD is about 2,300 entries of
/// sixteen bytes, which is a scan of 37 KB and one allocation. It is fast
/// enough for a command and honest about lifetime -- there is no global to go
/// stale, and no initialisation order to get wrong. A renderer opening it per
/// frame would want the opposite, and can have it when there is a renderer.
pub fn open() -> Option<Result<wad::Wad, wad::Error>> {
    crate::port::files::get(WAD_FILE).map(wad::Wad::parse)
}

//! What is on the screen, as a list, so two frames can be compared.
//!
//! The desktop used to repaint everything every frame -- wallpaper, every
//! window, every application, the whole terminal -- because repainting
//! everything is obviously correct and nothing else is. This keeps the
//! "obviously correct" and drops the "everything": `desk::layers` lists what
//! each frame contains as flat, z-ordered layers, each with the screen area it
//! covers and a fingerprint (`stamp`) of exactly what its painter reads, and
//! `diff` compares one frame's list with the last one's. What differs is the
//! damage; only the damage is repainted, by the same painters, clipped.
//!
//! It is the wlroots `wlr_scene` idea -- a scene the compositor can diff --
//! built the SerenityOS way: the scene is not a long-lived tree that every
//! mutation must remember to update, but a list derived afresh from the
//! desktop's state. A caller that changes the desktop does not report damage;
//! the next frame finds it. That keeps the property `desk::with` already has,
//! that touching the desktop is enough to get it repainted, and a fingerprint
//! that misses some state shows up in `render verify` as a mismatch rather
//! than in a screenshot as a stale pixel.
//!
//! A layer may carry **parts**: sub-areas with their own stamps, for content
//! that changes without the chrome around it changing -- an application's
//! client area, a terminal's status strip. A part whose stamp moved damages
//! the part alone.

use alloc::vec::Vec;
use core::hash::{Hash, Hasher};

use super::region::Region;
use super::theme::Rect;

/// Which thing a layer is, stable across frames, so the diff can pair them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayerId {
    Wallpaper,
    Icons,
    Taskbar,
    /// A window, by the serial it was given when it opened. Not its index:
    /// indices shift when a window closes or is raised.
    Window(u64),
    /// Tips, menus, the move hint and the Start menu.
    Overlay,
    /// A closed window fading out, by the serial it had.
    Ghost(u64),
    /// Flip 3D's stack, over everything.
    Flip,
}

#[derive(Clone, Debug)]
pub struct Layer {
    pub id: LayerId,
    /// Everything this layer may paint, its shadow included.
    pub bounds: Rect,
    /// A fingerprint of what the painter reads for everything except `parts`.
    pub stamp: u64,
    /// Sub-areas repainted on their own when only their stamp moved.
    pub parts: Vec<(Rect, u64)>,
}

/// FNV-1a, 64-bit. Not a cryptographic hash and does not need to be: a
/// collision costs one frame that should have repainted and did not, which
/// `render verify` exists to notice, and FNV is a multiply and an xor a byte.
pub struct Stamp(u64);

impl Stamp {
    pub fn new() -> Self {
        Stamp(0xcbf2_9ce4_8422_2325)
    }
    pub fn of(v: impl Hash) -> u64 {
        let mut s = Stamp::new();
        v.hash(&mut s);
        s.finish()
    }
}

impl Default for Stamp {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher for Stamp {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= *b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// Compare two frames' layers and add what changed to `out`.
///
/// - a layer only in one frame damages its bounds;
/// - a layer that moved or resized damages where it was and where it is;
/// - a layer whose own stamp changed damages its bounds;
/// - otherwise each part whose rectangle or stamp changed damages that part,
///   old and new;
/// - and a layer whose stacking changed -- something now above it that was
///   below it, or the reverse -- damages its bounds, because the overlap
///   between the two is now painted the other way round. Compared on the order
///   of the layers both frames share, so a layer appearing elsewhere in the
///   list does not count as everything else moving.
pub fn diff(prev: &[Layer], next: &[Layer], out: &mut Region) {
    let common_prev: Vec<LayerId> =
        prev.iter().map(|l| l.id).filter(|id| next.iter().any(|n| n.id == *id)).collect();
    let common_next: Vec<LayerId> =
        next.iter().map(|l| l.id).filter(|id| prev.iter().any(|p| p.id == *id)).collect();
    for p in prev {
        if !next.iter().any(|n| n.id == p.id) {
            out.add(p.bounds);
        }
    }
    for n in next {
        let Some(p) = prev.iter().find(|p| p.id == n.id) else {
            out.add(n.bounds);
            continue;
        };
        let moved_in_stack = common_prev.iter().position(|id| *id == n.id)
            != common_next.iter().position(|id| *id == n.id);
        if p.bounds != n.bounds || moved_in_stack || p.stamp != n.stamp {
            out.add(p.bounds);
            out.add(n.bounds);
            continue;
        }
        let parts = p.parts.len().max(n.parts.len());
        for k in 0..parts {
            match (p.parts.get(k), n.parts.get(k)) {
                (Some(a), Some(b)) if a == b => {}
                (a, b) => {
                    if let Some(a) = a {
                        out.add(a.0);
                    }
                    if let Some(b) = b {
                        out.add(b.0);
                    }
                }
            }
        }
    }
}

/// `diag scene`: the diff, on synthetic frames.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    let wall = Layer { id: LayerId::Wallpaper, bounds: Rect::new(0, 0, 1000, 800), stamp: 1, parts: Vec::new() };
    let win = |serial: u64, x: u32, y: u32, stamp: u64, part: u64| Layer {
        id: LayerId::Window(serial),
        bounds: Rect::new(x, y, 200, 100),
        stamp,
        parts: alloc::vec![(Rect::new(x + 10, y + 30, 180, 60), part)],
    };
    let frame = |v: &[Layer]| -> Region {
        let mut r = Region::new();
        diff(&[wall.clone(), win(1, 100, 100, 7, 0), win(2, 400, 100, 7, 0)], v, &mut r);
        r
    };

    let same = frame(&[wall.clone(), win(1, 100, 100, 7, 0), win(2, 400, 100, 7, 0)]);
    claim("an unchanged frame damages nothing", same.is_empty());

    let moved = frame(&[wall.clone(), win(1, 110, 100, 7, 0), win(2, 400, 100, 7, 0)]);
    let mut expect = Region::new();
    expect.add(Rect::new(100, 100, 200, 100));
    expect.add(Rect::new(110, 100, 200, 100));
    claim("a moved window damages exactly where it was and where it is",
          moved.area() == expect.area() && moved.bbox() == expect.bbox());

    let content = frame(&[wall.clone(), win(1, 100, 100, 7, 1), win(2, 400, 100, 7, 0)]);
    claim("a change of content damages the client area alone",
          content.area() == 180 * 60 && content.bbox() == Some(Rect::new(110, 130, 180, 60)));

    let chrome = frame(&[wall.clone(), win(1, 100, 100, 8, 0), win(2, 400, 100, 7, 0)]);
    claim("a change of chrome damages the whole window", chrome.area() == 200 * 100);

    let raised = frame(&[wall.clone(), win(2, 400, 100, 7, 0), win(1, 100, 100, 7, 0)]);
    claim("a change of stacking damages the windows whose order changed",
          raised.area() == 2 * 200 * 100);

    let closed = frame(&[wall.clone(), win(1, 100, 100, 7, 0)]);
    claim("a closed window damages where it was",
          closed.bbox() == Some(Rect::new(400, 100, 200, 100)));

    let opened = frame(&[wall.clone(), win(1, 100, 100, 7, 0), win(2, 400, 100, 7, 0), win(3, 0, 0, 1, 0)]);
    claim("an opened window damages where it is and nothing else moves",
          opened.bbox() == Some(Rect::new(0, 0, 200, 100)));

    claim("a stamp is a pure function of its input",
          Stamp::of(("a", 1u32)) == Stamp::of(("a", 1u32)) && Stamp::of(("a", 1u32)) != Stamp::of(("a", 2u32)));
    ok
}

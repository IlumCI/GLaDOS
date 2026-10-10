//! The quality governor: what to give up when frames run slow, and when to
//! take it back.
//!
//! The target is 60 frames a second while something moves, and a machine
//! that cannot reach it should look plainer rather than stutter -- motion
//! that drops frames reads as broken, and a pane without blur reads as a
//! choice. So effects are shed in a declared order, cheapest to lose first:
//!
//! ```text
//!   0  everything
//!   1  no blur        glass is its tint over a flat wall
//!   2  no dimming     unfocused windows are not darkened
//!   3  half motion    every animation takes half as long
//!   4  no motion      windows arrive, leave and move at once
//! ```
//!
//! **Only frames that were animating count.** An idle desktop composes the
//! tray clock ten times a second in a fraction of a millisecond, and a
//! governor fed those would always think the machine was fast. A frame of a
//! window arriving, a slide, a Flip turn is the frame the budget is about.
//!
//! **It drops quickly and climbs slowly**, which is the hysteresis: a window
//! of frames where a third missed the budget, twice running, sheds one effect, and five
//! windows in a row comfortably inside half of it -- *at the level it is
//! now* -- earn one back. Without the asymmetry it oscillates: shedding
//! blur makes frames fast, fast frames restore blur, blur makes them slow.
//!
//! `adjust` is pure, so all of that is asserted at boot (`diag quality`).

use core::sync::atomic::{AtomicU32, AtomicU8, Ordering};

/// Frames per decision.
///
/// Twelve, a fifth of a second at 60 Hz. It was thirty, and on the machine
/// the governor exists for that never decided anything: at 85 ms a frame a
/// 410 ms arrival is five frames, so a window of thirty animating frames
/// took six animations to fill -- measured under TCG, a whole session of
/// opens and Flip turns shed nothing at a p95 of 85 ms.
pub const WINDOW: usize = 12;
/// The frame budget: 60 Hz.
pub const BUDGET_US: u32 = 16_667;
/// How many good windows in a row earn a level back: five, so climbing takes
/// a second of easy frames at 60 Hz where dropping takes a fifth of one.
pub const CLIMB: u8 = 5;
pub const LEVELS: u8 = 5;

/// The governor's state between frames: its level, the frames of the
/// current window, and how many good windows in a row it has seen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gov {
    pub level: u8,
    pub frames: [u32; WINDOW],
    pub n: usize,
    pub good: u8,
    /// Slow windows in a row. Two shed a rung: one is a burst -- a window
    /// opening draws its image once, and three opening at once filled a
    /// window with slow frames on a machine with headroom, under KVM.
    pub bad: u8,
}

impl Gov {
    pub const fn new() -> Gov {
        Gov { level: 0, frames: [0; WINDOW], n: 0, good: 0, bad: 0 }
    }
}

/// One animating frame of `us` microseconds, under a floor and a ceiling
/// (`0..=4`). Answers the governor afterwards; its `level` is what to draw
/// at.
pub fn adjust(mut g: Gov, us: u32, floor: u8, ceiling: u8) -> Gov {
    let (floor, ceiling) = (floor.min(LEVELS - 1), ceiling.min(LEVELS - 1).max(floor.min(LEVELS - 1)));
    g.frames[g.n] = us;
    g.n += 1;
    if g.n < WINDOW {
        g.level = g.level.clamp(floor, ceiling);
        return g;
    }
    g.n = 0;
    let slow = g.frames.iter().filter(|&&f| f > BUDGET_US).count();
    let easy = g.frames.iter().filter(|&&f| f <= BUDGET_US / 2).count();
    if slow * 3 >= WINDOW {
        g.good = 0;
        g.bad += 1;
        if g.bad >= 2 {
            g.level = (g.level + 1).min(ceiling);
            g.bad = 0;
        }
    } else if easy * 10 >= WINDOW * 9 {
        g.good += 1;
        if g.good >= CLIMB && g.level > floor {
            g.level -= 1;
            g.good = 0;
        }
    } else {
        g.good = 0;
    }
    if slow * 3 < WINDOW {
        g.bad = 0;
    }
    g.level = g.level.clamp(floor, ceiling);
    g
}

static LEVEL: AtomicU8 = AtomicU8::new(0);
/// Floor and ceiling (Settings: `desk quality <floor> <ceiling>`).
static RANGE: AtomicU32 = AtomicU32::new(4);
static GOV: crate::sync::Spin<Gov> = crate::sync::Spin::new(Gov::new());

/// The level being drawn at.
pub fn level() -> u8 {
    LEVEL.load(Ordering::Relaxed)
}

pub fn set_range(floor: u8, ceiling: u8) {
    let (f, c) = (floor.min(LEVELS - 1), ceiling.min(LEVELS - 1).max(floor.min(LEVELS - 1)));
    RANGE.store((f as u32) << 8 | c as u32, Ordering::Relaxed);
    let mut g = GOV.lock_irq();
    g.level = g.level.clamp(f, c);
    LEVEL.store(g.level, Ordering::Relaxed);
}

pub fn range() -> (u8, u8) {
    let r = RANGE.load(Ordering::Relaxed);
    ((r >> 8) as u8, r as u8)
}

/// Whether effect `rung` (1 blur, 2 dimming, 3 half motion, 4 motion) has
/// been shed.
pub fn shed(rung: u8) -> bool {
    level() >= rung
}

/// The compositor's: one frame composed while something animated. Answers
/// whether the level moved, in which case everything is repainted.
pub fn observe(us: u32) -> bool {
    let (f, c) = range();
    let mut g = GOV.lock_irq();
    let before = g.level;
    *g = adjust(*g, us, f, c);
    LEVEL.store(g.level, Ordering::Relaxed);
    g.level != before
}

pub fn name(level: u8) -> &'static str {
    match level {
        0 => "everything",
        1 => "no blur",
        2 => "no blur or dimming",
        3 => "half motion",
        _ => "no motion",
    }
}

/// `diag quality`.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    let run = |mut g: Gov, us: u32, frames: usize, f: u8, c: u8| {
        for _ in 0..frames {
            g = adjust(g, us, f, c);
        }
        g
    };
    let g = run(Gov::new(), 30_000, WINDOW - 1, 0, 4);
    claim("nothing is decided before a whole window of frames", g.level == 0);
    let g = run(Gov::new(), 30_000, WINDOW, 0, 4);
    claim("one window of slow frames is a burst, and sheds nothing", g.level == 0);
    let g = run(Gov::new(), 30_000, WINDOW * 2, 0, 4);
    claim("two in a row shed one effect", g.level == 1);
    let g = run(Gov::new(), 30_000, WINDOW * 9, 0, 4);
    claim("and it keeps shedding, a rung every two windows, down to the ceiling", g.level == 4);
    let g = run(Gov::new(), 30_000, WINDOW * 9, 0, 2);
    claim("which it never passes", g.level == 2);
    let mut mixed = Gov::new();
    for i in 0..WINDOW {
        mixed = adjust(mixed, if i % 4 == 0 { 30_000 } else { 9_000 }, 0, 4);
    }
    claim("a quarter of frames over budget is not enough to shed", mixed.level == 0);
    let slow = run(Gov::new(), 30_000, WINDOW * 4, 0, 4);
    let once = run(slow, 4_000, WINDOW, 0, 4);
    claim("one good window does not earn anything back", once.level == 2);
    let back = run(slow, 4_000, WINDOW * CLIMB as usize, 0, 4);
    claim("five good windows in a row earn back one", back.level == 1);
    let middling = run(slow, 12_000, WINDOW * 6, 0, 4);
    claim("frames inside the budget but not easily so hold where it is", middling.level == 2);
    let floored = run(Gov::new(), 1_000, WINDOW * 9, 3, 4);
    claim("a floor is held however fast the frames are", floored.level == 3);
    ok
}

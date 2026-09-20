//! What actually reached the screen, and when it stopped.
//!
//! The desktop freezes during long foreground commands. That has been fixed
//! three times -- `pump_cursor` was added for `ask`, then called from the clock
//! task as well once `mine sweep` turned out to have the same shape, and
//! `paint_clock` was moved through the compositor -- and it is still happening,
//! which is the tell that the fixes were addressing call sites rather than a
//! cause.
//!
//! So before any of it is rearranged: an instrument. This counts what each
//! painter did and, more usefully, **the longest anybody went without one**.
//!
//! ### The gap is the measurement, not the rate
//!
//! A frames-per-second average over a session that froze for twenty seconds
//! and ran smoothly for forty reads as a perfectly healthy twenty. What a
//! person experiences is the *worst* interval, so that is what is recorded:
//! `present` keeps the largest span between two consecutive calls, and a
//! freeze is that number rather than an adjective.
//!
//! ### The contrast is the diagnosis
//!
//! Three painters are counted separately on purpose, because the symptom is
//! that they disagree. `paint_clock` and the cursor run on the clock task,
//! which wakes on its own quantum; `draw` and `present` run wherever somebody
//! remembered to call them, which is sixteen scattered sites and the shell's
//! idle loop. If a run comes back with thousands of clock paints and a present
//! gap of twenty seconds, the freeze is not a rendering bug at all -- it is
//! that nothing owns the frame.
//!
//! Counters only, and deliberately nothing else: this file changes no
//! behaviour, so a measurement taken with it cannot be an artefact of it.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Something changed and the screen does not know yet.
///
/// **This replaces sixteen scattered `desk::draw()` calls**, which is what the
/// freeze actually was: painting was push-model with no owner, so every
/// feature that changed the screen had to remember to repaint, and a task
/// inside a long command remembered nothing. Measured before this existed --
/// one frame composed in seventy-three seconds, the screen still for
/// fifty-nine of them, while the clock task painted a hundred and eighty-seven
/// times.
///
/// Marking is free and idempotent; the compositor decides when. A caller that
/// marks twice costs one frame, and a caller that forgets is the bug this is
/// meant to end -- so `invalidate` is cheap enough that the honest default is
/// to call it whenever anything might have moved.
static DIRTY: AtomicBool = AtomicBool::new(true);

/// Whether the compositor task paints at all.
///
/// Off is what the machine did before it existed, so the two can be compared
/// in one binary rather than across two builds -- which is the difference
/// between a paired measurement and two numbers from different machines, the
/// distinction `rails.py` exists to make.
static ENABLED: AtomicBool = AtomicBool::new(true);

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Release);
    if on {
        invalidate();
    }
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

/// Say the screen is out of date. Cheap, and safe from anywhere.
pub fn invalidate() {
    DIRTY.store(true, Ordering::Release);
}

/// Claim the pending repaint, if there is one.
pub fn take_dirty() -> bool {
    DIRTY.swap(false, Ordering::AcqRel)
}

/// Put it back, for a frame that could not be painted after all.
///
/// The compositor takes the flag *before* it tries, so a change arriving
/// during a frame is not swallowed by that frame -- but a frame refused
/// because a full-screen program owns the screen never happened, and dropping
/// the flag there would leave the desktop stale until something else moved.
pub fn restore_dirty() {
    DIRTY.store(true, Ordering::Release);
}

/// Calls to `desk::draw`, which composes a whole frame into the back buffer.
static DRAWS: AtomicU64 = AtomicU64::new(0);
/// Calls to `compose::present`, which copies changed spans to the aperture.
static PRESENTS: AtomicU64 = AtomicU64::new(0);
/// Presents that actually wrote a span. A present over an unchanged frame is
/// a 4 MB compare and no pixels, so the two counts differing is how much of
/// the work was a no-op.
static WROTE: AtomicU64 = AtomicU64::new(0);
/// Rows written, summed. A present that wrote one row and one that wrote a
/// thousand are the same event and very different amounts of screen.
static ROWS: AtomicU64 = AtomicU64::new(0);
/// The clock task's two small paints, which are what keep moving while
/// everything else is stopped.
static CLOCKS: AtomicU64 = AtomicU64::new(0);
static CURSORS: AtomicU64 = AtomicU64::new(0);
/// Paints refused because a full-screen program owns the screen. Counted so a
/// quiet run under `doom` is not mistaken for a freeze.
static REFUSED: AtomicU64 = AtomicU64::new(0);

/// `rdtsc` at the last present, and the largest gap between two of them.
static LAST: AtomicU64 = AtomicU64::new(0);
static MAX_GAP: AtomicU64 = AtomicU64::new(0);
/// When the window under measurement opened, so a rate can be computed.
static SINCE: AtomicU64 = AtomicU64::new(0);

/// Which task composed a frame, and whether more than one ever has.
///
/// **Single-writer has to be a printed number, not a belief.** Every stage
/// after this one rests on "nothing else paints", and the only honest way to
/// hold that is to record who did. `sync::audit` cannot answer it: `Racy::get`
/// is `#[track_caller]` and every desktop mutation goes through `desk::with`,
/// so all of them collapse to one line and the report says "shared" without
/// saying by whom.
static COMPOSER: AtomicU64 = AtomicU64::new(u64::MAX);
static COMPOSERS: AtomicU64 = AtomicU64::new(0);

pub fn drew() {
    DRAWS.fetch_add(1, Ordering::Relaxed);
    let me = crate::task::current() as u64;
    // A different composer than last time is what is being counted, not the
    // identity itself -- `task::current()` answers 0 for both task 0 and an
    // idle core, so the id alone is not trustworthy. A *change* in it is.
    let prev = COMPOSER.swap(me, Ordering::Relaxed);
    if prev != me {
        COMPOSERS.fetch_add(1, Ordering::Relaxed);
    }
}

/// How many times the composing task changed. One means one task has ever
/// composed a frame, which is the property the whole design rests on.
pub fn composer_changes() -> u64 {
    COMPOSERS.load(Ordering::Relaxed)
}

pub fn clock_painted() {
    CLOCKS.fetch_add(1, Ordering::Relaxed);
}

pub fn cursor_painted() {
    CURSORS.fetch_add(1, Ordering::Relaxed);
}

pub fn refused() {
    REFUSED.fetch_add(1, Ordering::Relaxed);
}

/// Called by `present`, with how many rows it wrote.
///
/// The gap is measured here rather than at `draw` because `present` is what
/// the screen sees. A frame composed into the back buffer and never presented
/// changed nothing a person could look at.
pub fn presented(rows: u64) {
    let now = crate::time::rdtsc();
    PRESENTS.fetch_add(1, Ordering::Relaxed);
    if rows > 0 {
        WROTE.fetch_add(1, Ordering::Relaxed);
        ROWS.fetch_add(rows, Ordering::Relaxed);
    }
    let prev = LAST.swap(now, Ordering::Relaxed);
    // Zero means this is the first present since a reset, and the interval
    // before it is not a gap anybody experienced.
    if prev != 0 && now > prev {
        let gap = now - prev;
        MAX_GAP.fetch_max(gap, Ordering::Relaxed);
    }
}

/// Start a fresh window. Everything below is measured from here.
pub fn reset() {
    for c in [&DRAWS, &PRESENTS, &WROTE, &ROWS, &CLOCKS, &CURSORS, &REFUSED, &MAX_GAP,
              &COMPOSERS] {
        c.store(0, Ordering::Relaxed);
    }
    // Not the identity -- resetting that would count the next frame as a new
    // composer and report two where there is one.
    COMPOSER.store(u64::MAX, Ordering::Relaxed);
    LAST.store(0, Ordering::Relaxed);
    SINCE.store(crate::time::rdtsc(), Ordering::Relaxed);
}

pub struct Stats {
    pub draws: u64,
    pub presents: u64,
    pub wrote: u64,
    pub rows: u64,
    pub clocks: u64,
    pub cursors: u64,
    pub refused: u64,
    /// Milliseconds, the longest anybody waited between two presents.
    pub max_gap_ms: u64,
    /// Milliseconds the window has been open.
    pub window_ms: u64,
}

pub fn stats() -> Stats {
    let mhz = crate::time::tsc_mhz().max(1) as u64;
    let since = SINCE.load(Ordering::Relaxed);
    let now = crate::time::rdtsc();
    // **The gap that is still open counts, and leaving it out read as zero.**
    //
    // `MAX_GAP` only ever sees the interval *between two* presents, so a
    // window containing one present contains no interval and the worst gap
    // came back as `0 ms` -- from a run where the screen was composed once in
    // sixty-eight seconds, which is the most frozen a screen can be. An
    // instrument answering zero for the total freeze is the failure this
    // whole file exists to avoid, arriving inside it.
    //
    // So the span from the last present to *now* is a gap too, and so is the
    // span from the reset to the first one when none has happened yet.
    let open = now.saturating_sub(if LAST.load(Ordering::Relaxed) != 0 {
        LAST.load(Ordering::Relaxed)
    } else {
        since
    });
    let worst = MAX_GAP.load(Ordering::Relaxed).max(open);
    Stats {
        draws: DRAWS.load(Ordering::Relaxed),
        presents: PRESENTS.load(Ordering::Relaxed),
        wrote: WROTE.load(Ordering::Relaxed),
        rows: ROWS.load(Ordering::Relaxed),
        clocks: CLOCKS.load(Ordering::Relaxed),
        cursors: CURSORS.load(Ordering::Relaxed),
        refused: REFUSED.load(Ordering::Relaxed),
        max_gap_ms: worst / mhz / 1000,
        window_ms: if since != 0 && now > since { (now - since) / mhz / 1000 } else { 0 },
    }
}

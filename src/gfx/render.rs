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
//! ### It stopped being counters only, and the watchdog is why
//!
//! This opened as counters and nothing else, so that a measurement taken with
//! it could not be an artefact of it. That held while there were several
//! painters and the diagnosis was that they disagreed. There is one painter
//! now, and a count that stops rising cannot say whether the task died, is
//! starved, or is blocked -- so the file also carries a heartbeat, the phase
//! the compositor was in, and what the scheduler makes of its task.
//!
//! It earned that immediately. Two readings across one `diag all` had the
//! compositor resumed 2,194 times and its loop turning once, which is a task
//! being scheduled constantly and blocking inside a single iteration -- a
//! different bug from the starvation the counts alone had suggested, and one
//! no rate could have named.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Something changed and the screen does not know yet.
///
/// **This was written up as replacing the scattered `desk::draw()` calls and
/// it replaced none of them.** It says so here because the claim was load
/// bearing: painting was push-model with no owner, every feature that changed
/// the screen had to remember to repaint, and a task inside a long command
/// remembered nothing -- one frame composed in seventy-three seconds, the
/// screen still for fifty-nine of them, while the clock task painted a hundred
/// and eighty-seven times. The compositor fixes *that*, because it repaints on
/// its own account whether or not anybody asked.
///
/// What it did not do is remove the other writers. `desk.rs` still calls
/// `draw()` directly in **fifty** places, so the push model is intact
/// alongside the pull one and any task that runs one of those sites composes a
/// frame. Measured: `composers` reads 1 after a `render probe` and 3 after
/// opening a single window, because `focus_terminal` ends with a `draw()` and
/// the shell calls it after every command. The two things that measure
/// `composers` open no windows, which is why it read 1 for so long.
///
/// Marking is free and idempotent and the compositor decides when. A caller
/// that marks twice costs one frame, and a caller that forgets is the bug this
/// is meant to end -- so `invalidate` is cheap enough that the honest default
/// is to call it whenever anything might have moved.
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
/// The taskbar's two readouts, and the pointer.
///
/// **These counted a different thing until the readouts moved.** They were
/// the clock task's paints, and the whole point of counting them separately
/// was that they kept moving while everything else was stopped -- a machine
/// running with nobody responsible for the picture. Both painters are the
/// compositor's now, so the contrast they existed to show is gone, and what
/// they measure instead is how much of the screen a frame did not have to
/// touch: a tenth of a second of uptime is a few thousand pixels where a
/// composed frame is 2,143 us.
///
/// The question they used to answer is the watchdog's now, and it answers it
/// better -- a beat that has stopped names the phase it stopped in, where a
/// count that kept rising only ever said somebody was still painting.
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

pub fn tray_painted() {
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
///
/// **A measurement follows the window; a record of a defect does not.** Those
/// are two different questions -- "how did the screen do during that command"
/// against "has this machine's only painter ever stopped" -- and the watchdog
/// holds one of each. `WORST_BEAT` is an interval, exactly like `MAX_GAP`
/// beside it, so a gap from boot would otherwise sit in every later reading
/// and hide a smaller one during the thing actually being measured. `BEATS`
/// and `STALLS` stay: throwing away evidence that the compositor once stopped
/// because somebody opened a fresh window would lose the most important thing
/// this file knows, and `render reset` is typed casually.
pub fn reset() {
    for c in [&DRAWS, &PRESENTS, &WROTE, &ROWS, &CLOCKS, &CURSORS, &REFUSED, &MAX_GAP,
              &COMPOSERS, &WORST_BEAT] {
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

// ---------------------------------------------------------------------------
// The watchdog
// ---------------------------------------------------------------------------

/// Where the compositor was when it last said anything.
///
/// **The phase is the whole value of this.** "Frames stopped" is one symptom
/// with four different causes, and they want four different answers: stuck
/// composing means an app's `draw_in` loops, stuck in the pointer means a
/// press handler ran something unbounded, stuck between turns means the
/// scheduler stopped handing this task the core, and never started means the
/// spawn failed at boot and the one line saying so scrolled past with three
/// hundred others. Without the phase, all four read as a black screen.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// Never beat. The task was never spawned, or has not reached its loop.
    Cold,
    /// Between turns: checking the deadline, standing down, or yielding.
    Turn,
    /// Applying ops posted by tasks that do not own the desktop.
    Ops,
    /// Reading the pointer, and running whatever a press started.
    Pointer,
    /// Inside `desk::draw`, composing.
    Frame,
    /// Painting the taskbar's uptime and charge.
    Tray,
}

impl Phase {
    fn code(self) -> u64 {
        match self {
            Phase::Cold => 0,
            Phase::Turn => 1,
            Phase::Ops => 2,
            Phase::Pointer => 3,
            Phase::Frame => 4,
            Phase::Tray => 5,
        }
    }

    fn of(code: u64) -> Phase {
        match code {
            1 => Phase::Turn,
            2 => Phase::Ops,
            3 => Phase::Pointer,
            4 => Phase::Frame,
            5 => Phase::Tray,
            _ => Phase::Cold,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Phase::Cold => "never started",
            Phase::Turn => "between turns",
            Phase::Ops => "applying posted ops",
            Phase::Pointer => "reading the pointer",
            Phase::Frame => "composing a frame",
            Phase::Tray => "painting the taskbar readouts",
        }
    }
}

/// Turns of the compositor loop, and when the last one was.
///
/// Separate from `DRAWS` on purpose, and the difference is the diagnosis. A
/// loop that turns while no frame lands is *healthy* and idle: nothing was
/// dirty. A loop that stops turning is the machine having lost its only
/// painter. The counters above cannot tell those apart, because both of them
/// look like "presents stopped".
static BEATS: AtomicU64 = AtomicU64::new(0);
static BEAT_AT: AtomicU64 = AtomicU64::new(0);
static PHASE: AtomicU64 = AtomicU64::new(0);
/// The worst beat-to-beat interval, same shape as `MAX_GAP` and including the
/// one still open. An instrument that answers zero during the total stall is
/// the failure this file already had once.
static WORST_BEAT: AtomicU64 = AtomicU64::new(0);

/// Whether the watchdog currently believes the compositor is gone, and how
/// many times it has come to believe that.
static STALLED: AtomicBool = AtomicBool::new(false);
static STALLS: AtomicU64 = AtomicU64::new(0);
/// When the watchdog was armed, so "never started" has a timebase. Zero until
/// the first `watch()`.
static WATCH_FROM: AtomicU64 = AtomicU64::new(0);

/// How long the compositor may go quiet before it is called dead.
///
/// Two seconds, and generously so. A frame is 2,143 us, and a press is allowed
/// to open a window or run an Aiksi program under `DRAW_BUDGET`, so a
/// legitimate turn of the loop can take a good fraction of a second. The cost
/// of being early here is an alarm about a machine that is fine, which teaches
/// people to ignore the alarm; the cost of being late is two seconds added to
/// a wait that is already indefinite.
const STALL_MS: u64 = 2000;

/// The compositor says it is alive, and what it is about to do.
///
/// Called at the four points where the loop turns or disappears into somebody
/// else's code. Three atomics and no allocation, because this runs at the top
/// of the hottest loop in the tree.
pub fn beat(phase: Phase) {
    let now = crate::time::rdtsc();
    PHASE.store(phase.code(), Ordering::Relaxed);
    BEATS.fetch_add(1, Ordering::Relaxed);
    let prev = BEAT_AT.swap(now, Ordering::Relaxed);
    if prev != 0 && now > prev {
        WORST_BEAT.fetch_max(now - prev, Ordering::Relaxed);
    }
}

pub struct Health {
    /// Turns of the compositor loop since boot.
    pub beats: u64,
    /// Milliseconds since the last one.
    pub quiet_ms: u64,
    /// The longest it has ever been quiet, including right now.
    pub worst_ms: u64,
    /// What it was doing when it last spoke.
    pub phase: Phase,
    /// True while the watchdog believes it is gone.
    pub stalled: bool,
    /// How many separate stalls have been seen.
    pub stalls: u64,
}

pub fn health() -> Health {
    let mhz = crate::time::tsc_mhz().max(1) as u64;
    let now = crate::time::rdtsc();
    let beats = BEATS.load(Ordering::Relaxed);
    // Before the first beat there is no interval to report, so the quiet is
    // measured from when the watchdog was armed. Answering "0 ms quiet" for a
    // compositor that never existed would be the instrument agreeing with the
    // bug.
    let from = match BEAT_AT.load(Ordering::Relaxed) {
        0 => WATCH_FROM.load(Ordering::Relaxed),
        b => b,
    };
    let quiet = if from == 0 { 0 } else { now.saturating_sub(from) };
    Health {
        beats,
        quiet_ms: quiet / mhz / 1000,
        worst_ms: WORST_BEAT.load(Ordering::Relaxed).max(quiet) / mhz / 1000,
        phase: if beats == 0 { Phase::Cold } else { Phase::of(PHASE.load(Ordering::Relaxed)) },
        stalled: STALLED.load(Ordering::Relaxed),
        stalls: STALLS.load(Ordering::Relaxed),
    }
}

/// What the watchdog decided this visit, when the decision changed.
pub enum Verdict {
    /// It has gone quiet for longer than a working compositor ever does.
    Stalled(Health),
    /// It started beating again.
    Recovered(Health),
}

/// Check on the compositor. Edge-triggered: answers only when the state flips.
///
/// **Deliberately not called by the compositor.** A task cannot notice its own
/// absence, so the one thing a watchdog must not depend on is the thing it
/// watches. The clock task does the asking, because it is independent of the
/// desktop, never blocks and is already awake once a second. If *that* stops
/// too, the uptime in the taskbar stops with it, which is a symptom somebody
/// is already looking straight at.
///
/// Reporting is all it does. There is no killing a stuck task here and no
/// restarting one: `desk::draw` holds the paint claim across a whole frame and
/// `Claim` releases on `Drop`, so anything that abandons that stack without
/// unwinding leaves `PAINTER` held by a task that no longer exists, and every
/// `desk::with` in the tree then yields forever waiting for it. That is why
/// there is no `recover::guard` around the frame either, which is otherwise
/// the obvious thing to reach for: `recover::land` is a longjmp and runs no
/// destructors, so catching a fault inside `draw` would trade a loud halt with
/// a full register dump for a silent machine-wide deadlock. A watchdog that
/// says what is wrong is worth more than one that tries to fix it and makes it
/// permanent.
pub fn watch() -> Option<Verdict> {
    let now = crate::time::rdtsc();
    if WATCH_FROM.load(Ordering::Relaxed) == 0 {
        WATCH_FROM.store(now, Ordering::Relaxed);
        return None;
    }
    let h = health();
    let bad = h.quiet_ms >= STALL_MS;
    if bad == h.stalled {
        return None;
    }
    STALLED.store(bad, Ordering::Release);
    if bad {
        STALLS.fetch_add(1, Ordering::Relaxed);
        Some(Verdict::Stalled(h))
    } else {
        Some(Verdict::Recovered(h))
    }
}

/// Which task the compositor is, so the alarm can say what became of it.
///
/// **This is the field that would have saved the investigation that found the
/// first real stall.** The watchdog said the loop had stopped turning and the
/// phase said it was not stuck inside anything, which leaves "it is not being
/// given the core" -- and that has three quite different causes the scheduler
/// tells apart and nothing else does. `ready` is a task the scheduler could
/// pick and does not, which is starvation. `running` is a task some core still
/// claims while not actually running it, which is unclaimable by every other
/// core by design. `handoff` is one stranded mid-switch, which is unclaimable
/// by anybody at all.
///
/// Told rather than guessed: `task::current()` answers 0 for both task 0 and
/// an idle core, so a compositor asking who it is cannot be trusted. `spawn`
/// already answers the index and `main` has it in hand.
static COMP_TASK: AtomicU64 = AtomicU64::new(u64::MAX);

pub fn watching(task: usize) {
    COMP_TASK.store(task as u64, Ordering::Release);
}

/// What the scheduler thinks of the compositor's task, if it is known.
pub fn comp_state() -> Option<&'static str> {
    let i = COMP_TASK.load(Ordering::Acquire);
    if i == u64::MAX {
        return None;
    }
    crate::task::snapshot(i as usize).map(|t| t.state.name())
}

/// How many times the scheduler has resumed the compositor.
///
/// **This is the discriminator the state alone could not give.** A task the
/// scheduler never picks and a task it picks constantly that never reaches the
/// top of its own loop both read as `ready` from outside, and they are
/// completely different bugs -- starvation against something inside the loop
/// swallowing every turn. The counts say which: beats frozen with resumes
/// climbing is the second, both frozen is the first.
///
/// Worth having as its own number rather than read off `tasks`, because that
/// verb takes a snapshot and the question is about a *rate*. Two readings of a
/// counter answer it; one reading of a table does not.
pub fn comp_switches() -> Option<u64> {
    let i = COMP_TASK.load(Ordering::Acquire);
    if i == u64::MAX {
        return None;
    }
    crate::task::snapshot(i as usize).map(|t| t.switches)
}

/// A deliberate stall, so the watchdog can be watched firing.
///
/// **An alarm nobody has seen go off is an alarm written in a comment.** Every
/// other claim about this watchdog is arithmetic -- the phase table, the
/// edge, the open interval -- and none of them answers the question that
/// matters, which is whether a compositor that genuinely stops turning
/// produces a line somebody reads. There is no way to ask that without
/// stopping one, and nothing here stops on its own to order.
///
/// So the compositor spins here on request. `rdtsc` and not a count, because
/// what is being reproduced is a wall-clock silence rather than an amount of
/// work, and the threshold it has to cross is in milliseconds.
static STALL_UNTIL: AtomicU64 = AtomicU64::new(0);

/// Ask the compositor to stop turning for a while. Shell-only.
///
/// Not an applet, and deliberately: `sysbox::APPLETS` is what the decoding
/// grammar is built from, so an applet here would be a route for the model to
/// stop the screen. Same rule `skill trust` and `work` follow.
pub fn stall_for(ms: u64) {
    let mhz = crate::time::tsc_mhz().max(1) as u64;
    STALL_UNTIL.store(crate::time::rdtsc() + ms * mhz * 1000, Ordering::Release);
}

/// Serve a pending stall, if there is one. Called by the compositor.
///
/// One relaxed load per frame when nothing is pending, which is about thirty a
/// second against a frame that costs two milliseconds. The phase is announced
/// before the spin rather than after, because a stall that reported its phase
/// on the way out would report it only once the stall was over.
pub fn stall_hook() {
    let until = STALL_UNTIL.load(Ordering::Acquire);
    if until == 0 {
        return;
    }
    beat(Phase::Frame);
    while crate::time::rdtsc() < until {
        core::hint::spin_loop();
    }
    STALL_UNTIL.store(0, Ordering::Release);
}

/// Claims about the watchdog, checked without a screen.
///
/// What is under test is the arithmetic and the edge, which is otherwise only
/// exercised on the day it matters.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };

    claim("every phase code round-trips", (0..6).all(|c| Phase::of(c).code() == c));
    // Out of range must land on `Cold` rather than on a neighbouring phase: a
    // stray read reporting "composing a frame" would send somebody into the
    // window manager after a compositor that was never spawned.
    claim("and an unknown one reads as never started", Phase::of(99) == Phase::Cold);

    // The compositor is running while this runs, so it must not be reported as
    // gone. This is the false-alarm check, and it decides whether the alarm is
    // worth having: one that cries wolf during `diag all` is one people learn
    // to skip past.
    let h = health();
    claim("the compositor has beaten at least once", h.beats > 0);
    claim("and is not quiet right now", !h.stalled && h.quiet_ms < STALL_MS);
    // `worst` includes the interval still open, so it can never be behind the
    // current quiet. That relationship is what the `MAX_GAP` bug broke.
    claim("worst quiet is never less than current quiet", h.worst_ms >= h.quiet_ms);

    // **This suite does not call `beat`, and the attempt to was a bug.**
    //
    // `beat` stamps `BEAT_AT`, which is the watchdog's whole measurement, and
    // this runs on the shell task inside `diag all` while the compositor is
    // beating on its own. Calling it once reset the compositor's quiet timer,
    // so the watchdog reported the screen recovering when what had moved was
    // the suite.
    //
    // Saving the three statics and putting them back afterwards looked like
    // the fix and is worse: the load and the store are a read-modify-write
    // across tasks, the shell is preempted in the middle of it, and every beat
    // the compositor makes inside that window is *discarded* -- `BEAT_AT` goes
    // back to a timestamp from before the suite ran. That reproduces a frozen
    // compositor exactly, out of an instrument measuring a healthy one, and it
    // cost a run to tell the artefact from the fault it was imitating.
    //
    // So the beat claims are gone rather than guarded. What they asserted was
    // that a `fetch_add` adds and an atomic reads back what was stored, which
    // is worth close to nothing against an instrument that lies about the one
    // number it exists to report. Everything above is a pure function of
    // values this suite owns.

    // A boot where `watching` was never called leaves the alarm able to say
    // the compositor stopped and unable to say what became of it, which is
    // precisely the half that is hard to get any other way.
    claim("the watchdog knows which task to ask about", comp_state().is_some());
    ok
}

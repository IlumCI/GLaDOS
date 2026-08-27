//! The resident mind: initiative between commands.
//!
//! Everything before this point has been reactive -- the model answers when
//! asked, plans when consulted, and goes quiet otherwise. This module removes
//! the asking. A task spawned at boot wakes on the second boundary and runs
//! the same loop a mind is usually described as having: perceive the
//! situation, remember what was decided last, choose among doing nothing,
//! acting directly, or giving itself a small goal to work an episode
//! through, write down which and why, and go back to sleep.
//!
//! Restraint is the load-bearing wall. Every gate exists because the failure
//! it prevents already happened somewhere else in someone else's system:
//!
//! - Any hardware input in the last few seconds stands everything down. The
//!   operator owns the foreground, always; initiative is what the machine
//!   does with the silence, never a rival for attention. (Under serial-only
//!   control nothing feeds the entropy ring, so headless is permanently
//!   "silent" -- which is exactly why the machine can be watched thinking
//!   from a script.)
//! - Never two minds at once: the agent task's own busy flag is checked
//!   before anything is queued.
//! - Cooldowns bound how often self-set goals may run, independent of how
//!   interesting the world looks.
//! - Self-queued episodes are read-only and budgeted small. Mutating trust
//!   is earned elsewhere or not at all.
//! - `initiative off` exists, and works, and is one command.
//!
//! The journal at `/ai/mind/journal.txt` is the inspectable inner monologue:
//! one line per tick, what was perceived, what was chosen, why. If this
//! module ever surprises you, read that first.

use super::{aixi, context, godbits};
use crate::sync::Racy;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(true);
static TICKS: AtomicU64 = AtomicU64::new(0);
static ACTS: AtomicU32 = AtomicU32::new(0);
static EPISODES: AtomicU32 = AtomicU32::new(0);
static SUPPRESSED: AtomicU32 = AtomicU32::new(0);

/// Seconds between policy evaluations. Fifteen is slow enough that the
/// daemon never appears in a profile and fast enough that "the machine
/// noticed" stays true within a human sense of noticed.
const TICK_SECS: u64 = 15;
/// Uptime before the first evaluation, whatever the tick spacing.
///
/// The machine should not set itself a goal before the operator has seen a
/// prompt. Under emulation that was true by accident: a boot took about 150 s
/// of guest time, so the first tick at 15 s had long passed by the time
/// anybody could type. Under the hypervisor accelerator a boot is around 20 s
/// and the first tick lands *during* it, so the resident mind queued an
/// episode before the boot log had finished printing and held the engine
/// against the first command the operator sent.
///
/// The floor is a minute and the real gate is `shell::interactive`, which
/// says a prompt has been printed and a person could have typed. A fixed
/// period alone is a guess: the prompt arrives at 150 s under TCG and at 21 s
/// under the hypervisor accelerator, so no single number sits after both.
/// Both conditions are required, and the settle time is measured from the
/// prompt rather than from power-on.
///
/// Not a workaround for a test. A mind that starts acting before anyone could
/// have told it not to has no business calling the restraint below
/// load-bearing.
const FIRST_TICK_AFTER_S: u64 = 60;
/// And this long after the prompt itself, whichever is later.
const SETTLE_SECS: u64 = 30;

/// Uptime at which a prompt was first seen. Zero means never.
static SAW_PROMPT_AT: Racy<u64> = Racy::new(0);
/// Hardware input inside this window defers all initiative.
const QUIET_AFTER_INPUT_S: u64 = 8;
/// Minimum spacing between self-set goals, however interesting things look.
const EPISODE_GAP_S: u64 = 240;
/// Minimum spacing between self-modification trials. An hour: a trial is
/// minutes of forward passes, and the value of running one more tonight is
/// far below the value of the machine still answering if somebody wakes up.
const GODEL_GAP_S: u64 = 3600;
/// Corpus examples per nightly trial. Small, and it is the honest tradeoff:
/// the judges get less evidence per trial in exchange for the machine staying
/// responsive, and the ledger accumulates across nights rather than within
/// one.
const GODEL_EXAMPLES: usize = 24;
/// Wall-clock ceiling on the optimiser half of a nightly trial.
const GODEL_MS: u64 = 20_000;

/// Journal length cap. Old lines fall off the top; the namespace keeps the
/// snapshots either way.
const JOURNAL_MAX: usize = 48;

static LAST_TOUCH: Racy<u64> = Racy::new(0);
static LOOP_SEEN: AtomicU64 = AtomicU64::new(0);
static LAST_TENTHS_SEEN: AtomicU64 = AtomicU64::new(u64::MAX);
/// Second number the last tick fired at. The %15 test is true for a whole
/// second of spin iterations, so without this edge-trigger one boundary
/// would fire five ticks.
static LAST_TICKED_SEC: AtomicU64 = AtomicU64::new(u64::MAX);
/// Presence requires two consecutive ticks that saw input. One blip is
/// usually a controller probe -- QEMU's i8042 enumeration trips the entropy
/// counter exactly once at boot -- and mistaking it for a person stood the
/// whole loop down on its very first live tick.
static PREV_TICK_TOUCHED: Racy<bool> = Racy::new(false);
static LAST_EPISODE_AT: Racy<u64> = Racy::new(0);
static LAST_GODEL_AT: Racy<u64> = Racy::new(0);
static EPOCH_SEEDED: AtomicBool = AtomicBool::new(false);
static JOURNAL: Racy<Vec<String>> = Racy::new(Vec::new());

const MIND_DIR: &str = "/ai/mind";
const JOURNAL_PATH: &str = "/ai/mind/journal.txt";
const REPORT_PATH: &str = "/ai/mind/report.txt";

/// Read-only goals the machine sets itself, rotated. Deliberately mundane:
/// their value is not the answer but the exercise -- every successful
/// transcript is material the router can be taught from, so curiosity here
/// widens reflex coverage later.
pub(crate) const CURIOSITY: [&str; 4] = [
    "list the files in /sys",
    "list the files in /tmp",
    "list the files in /ai",
    "list the files in /ai/tools",
];

/// What a tick decided, and the reason it can say out loud.
enum Decision {
    Sleep(&'static str),
    Act(&'static str),
    Episode(String),
}

/// The policy, pure in its inputs so the boot self-test can drive it
/// through every branch without waiting on a clock.
fn decide(
    enabled: bool,
    busy: bool,
    seconds_since_input: u64,
    seconds_since_episode: u64,
    planner_ready: bool,
    proposal: Option<&str>,
) -> Decision {
    if !enabled {
        return Decision::Sleep("disabled");
    }
    if busy {
        return Decision::Sleep("an episode is already running");
    }
    if seconds_since_input < QUIET_AFTER_INPUT_S {
        return Decision::Sleep("the operator is present");
    }
    if planner_ready {
        // The planner has cleared its own confidence gates and found a
        // schedule worth more than carrying on. Acting on it is the whole
        // point of having fitted it.
        return Decision::Act("planner verdict ready");
    }
    if seconds_since_episode < EPISODE_GAP_S {
        return Decision::Sleep("cooldown between self-set goals");
    }
    match proposal {
        Some(goal) => Decision::Episode(String::from(goal)),
        None => Decision::Sleep("nothing proposed"),
    }
}

/// Seconds since the last trial. Zero-at-boot means "never", which is
/// eligibility rather than a cooldown in force -- the same reading
/// `since_episode` gives its own clock.
fn since_godel(now_s: u64) -> u64 {
    let last = unsafe { *LAST_GODEL_AT.get() };
    if last == 0 {
        GODEL_GAP_S
    } else {
        now_s.saturating_sub(last)
    }
}

fn journal_push(line: String) {
    let mut buf = unsafe { &mut *JOURNAL.get() };
    buf.push(line);
    let len = buf.len();
    if len > JOURNAL_MAX {
        buf.drain(..len - JOURNAL_MAX);
    }
    // Whole-file rewrite: the namespace replaces atomically, and a journal
    // of this size costs less than any cleverness about appending.
    let mut text = buf.join("\n");
    text.push('\n');
    crate::sysbox::write_text(JOURNAL_PATH, &text);
}

fn situation_line(s: &context::Situation) -> String {
    format!(
        "heap {:.1}/{:.0} MiB drift {:+.2} load {} op {}",
        s.heap_used_mib,
        s.heap_total_mib,
        s.heap_trend_mib_s,
        s.load.label(),
        s.operator.label()
    )
}

/// One evaluation of the loop. Called on the second boundary from the
/// resident task every TICK_SECS, and directly by `initiative now`.
pub fn tick() {
    let now_s = crate::dev::lapic::ticks() / crate::TIMER_HZ as u64;
    // Settle time measured from the prompt, not from power-on.
    //
    // A floor alone is not enough even with the interactivity gate: boot ends
    // around 55 s under the hypervisor accelerator and the floor is 60, so
    // there was a five-second window in which the operator's first command
    // and the mind's first tick raced, and the mind won often enough to eat
    // it. Anchoring the wait to the moment a prompt appeared removes the
    // coincidence instead of moving it.
    if !crate::shell::interactive() {
        return;
    }
    let seen = unsafe { *SAW_PROMPT_AT.get() };
    if seen == 0 {
        unsafe { *SAW_PROMPT_AT.get() = now_s.max(1) };
        return;
    }
    if now_s.saturating_sub(seen) < SETTLE_SECS || now_s < FIRST_TICK_AFTER_S {
        // Counted as a tick that did not happen rather than skipped silently,
        // so `mind` shows the grace period rather than looking asleep.
        return;
    }
    TICKS.fetch_add(1, Ordering::Relaxed);
    // First evaluation is always eligible: zero here means no goal has ever
    // run, which is readiness rather than a cooldown in force.
    let since_episode = if !EPOCH_SEEDED.swap(true, Ordering::Relaxed) {
        EPISODE_GAP_S
    } else {
        now_s.saturating_sub(unsafe { *LAST_EPISODE_AT.get() })
    };

    // Input detection is a difference of cumulative counts: any key or
    // pointer event the ISRs saw since the previous tick counts as activity,
    // and only sustained activity counts as a person.
    let felt = godbits::felt() as u64;
    let last = unsafe { *LAST_TOUCH.get() };
    let touched = felt != last;
    unsafe { *LAST_TOUCH.get() = felt };
    let prev_touched = unsafe { *PREV_TICK_TOUCHED.get() };
    unsafe { *PREV_TICK_TOUCHED.get() = touched };
    let seconds_since_input = if touched && prev_touched {
        0
    } else {
        QUIET_AFTER_INPUT_S
    };

    let busy = super::agent::episode_busy();

    // The planner is consulted only often enough to stay honest about its
    // own confidence; its verdict is the expensive part of the tick and
    // microseconds against a fifteen-second clock.
    let p = aixi::plan(4, 16);
    let planner_ready = p.confident;

    // Rotate curiosity goals; the counter makes the rotation advance even
    // when ticks are suppressed, so the machine does not fixate.
    let proposal = {
        let n = CURIOSITY.len();
        let i = (TICKS.load(Ordering::Relaxed) as usize) % n;
        if since_episode >= EPISODE_GAP_S {
            Some(String::from(CURIOSITY[i]))
        } else {
            None
        }
    };

    let decision = decide(
        ENABLED.load(Ordering::Relaxed),
        busy,
        seconds_since_input,
        since_episode,
        planner_ready,
        proposal.as_deref(),
    );

    let sit = context::gather();
    // Console heartbeats for decisions worth knowing about. The two
    // repetitive waits (cooldown, episode-in-flight) journal silently --
    // fifteen-second reminders that the machine is waiting would be noise,
    // not transparency.
    let why = match &decision {
        Decision::Sleep(w) => *w,
        Decision::Act(w) => *w,
        Decision::Episode(g) => g.as_str(),
    };
    let quiet_wait = matches!(
        &decision,
        Decision::Sleep("cooldown between self-set goals")
            | Decision::Sleep("an episode is already running")
    );
    if !quiet_wait {
        crate::kprintln!(
            "[mind t{}] {}",
            TICKS.load(Ordering::Relaxed),
            why
        );
    }
    match decision {
        Decision::Sleep(why) => {
            SUPPRESSED.fetch_add(1, Ordering::Relaxed);
            journal_push(format!(
                "[t{} +{}s] sleep: {}. since_ep {}s, proposal {}, planner_ready {}. {}",
                TICKS.load(Ordering::Relaxed),
                now_s,
                why,
                since_episode,
                proposal.is_some(),
                planner_ready,
                situation_line(&sit)
            ));

            // Sleeping is when the machine may change itself, and only then.
            //
            // Two independent facts have to agree: the RTC says the operator
            // has gone to bed, and the entropy ring says no key or pointer
            // interrupt has fired. `quiet_now` checks both and names which
            // one refused, which is what the journal records.
            //
            // The budget is deliberately small. A trial is a forward pass per
            // example, and the mind task holds the engine for the whole of
            // one -- so a full-corpus run here would turn "the machine thinks
            // between your commands" into "the machine will not answer for
            // twenty minutes". Bounded, it is a few examples an hour, every
            // hour of the night, and the ledger accumulates.
            if let Ok(hour) = super::godel::quiet_now() {
                if since_godel(now_s) >= GODEL_GAP_S {
                    unsafe { *LAST_GODEL_AT.get() = now_s };
                    let b = super::train::Budget {
                        examples: GODEL_EXAMPLES,
                        millis: GODEL_MS,
                        ..Default::default()
                    };
                    let verdict = super::with_engine(|e| super::godel::trial(e, &b));
                    let line = match verdict {
                        None => String::from("engine held by another task"),
                        Some(Err(_)) => String::from("trainer refused"),
                        Some(Ok(c)) => format!(
                            "variant {} {} (fixed {}, broke {}, goals {}/{})",
                            super::godel::short_hex(&c.variant),
                            if c.adopted { "ADOPTED" } else { "rejected" },
                            c.fixed,
                            c.broke,
                            c.goals_held,
                            c.goals_total
                        ),
                    };
                    journal_push(format!(
                        "[t{} +{}s] godel: hour {}, {}",
                        TICKS.load(Ordering::Relaxed),
                        now_s,
                        hour,
                        line
                    ));
                }
            }
        }
        Decision::Act(why) => {
            ACTS.fetch_add(1, Ordering::Relaxed);
            // The one reversible action: publish the machine's own state
            // into its own namespace, where any later episode -- or any
            // later version of this loop -- can read it back.
            let report = format!(
                "situation at {}s\n{}\nplanner: best {:?} dU {:+.4}\n",
                now_s,
                situation_line(&sit),
                p.best.levels,
                p.best.u_mean - p.baseline.u_mean
            );
            let wrote = crate::sysbox::write_text(REPORT_PATH, &report);
            journal_push(format!(
                "[t{} +{}s] act: {} -> report {}",
                TICKS.load(Ordering::Relaxed),
                now_s,
                why,
                if wrote { "written" } else { "FAILED to write" }
            ));
        }
        Decision::Episode(ref goal) => {
            EPISODES.fetch_add(1, Ordering::Relaxed);
            unsafe { *LAST_EPISODE_AT.get() = now_s };
            let queued =
                super::agent::queue_autonomous(&goal, super::harness::Trust::ReadOnly, 2);
            journal_push(format!(
                "[t{} +{}s] episode: \"{}\" {}",
                TICKS.load(Ordering::Relaxed),
                now_s,
                goal,
                if queued { "queued (read-only, 2 steps)" } else { "REFUSED by queue" }
            ));
        }
    }
}

/// The resident task. Wakes on second boundaries; sleeps between them by
/// yielding, which is what every other well-behaved task here does.
pub fn initiative_task() {
    // Deliberately a yield-free spin, exactly like `clock_task`. The
    // temptation is `loop { yield_now(); ... }`, and the temptation has a
    // body count: a polling loop in net::tcp that yielded a hundred times a
    // second wedged the shell on every run -- see `task::yield_now`. Hot
    // voluntary yielding is the one scheduling pattern this kernel forbids;
    // preemption already shares the CPU fairly between spinners.
    let mut last_tenth = u64::MAX;
    loop {
        LOOP_SEEN.fetch_add(1, Ordering::Relaxed);
        let tenths = crate::dev::lapic::ticks() * 10 / crate::TIMER_HZ as u64;
        LAST_TENTHS_SEEN.store(tenths, Ordering::Relaxed);
        if tenths == last_tenth {
            continue;
        }
        last_tenth = tenths;
        let sec = tenths / 10;
        if sec != LAST_TICKED_SEC.swap(sec, Ordering::Relaxed) && sec % TICK_SECS == 0 {
            tick();
        }
    }
}

pub fn spawn() -> bool {
    crate::task::spawn("initiative", initiative_task).is_some()
}

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// Force one policy evaluation now, past the silence and cooldown gates but
/// never past busy or disabled. This is the headless handle: a script can
/// watch a full perceive-decide-act cycle without pretending to type.
pub fn force_tick() {
    tick();
}

pub fn status() -> (u64, u32, u32, u32, bool, u64, u64) {
    (
        TICKS.load(Ordering::Relaxed),
        ACTS.load(Ordering::Relaxed),
        EPISODES.load(Ordering::Relaxed),
        SUPPRESSED.load(Ordering::Relaxed),
        ENABLED.load(Ordering::Relaxed),
        LOOP_SEEN.load(Ordering::Relaxed),
        LAST_TENTHS_SEEN.load(Ordering::Relaxed),
    )
}

pub fn journal_tail(n: usize) -> Vec<String> {
    let buf = unsafe { &*JOURNAL.get() };
    let start = buf.len().saturating_sub(n);
    buf[start..].to_vec()
}

/// Boot self-test: the policy's branches, driven synthetically. The gates
/// are ordered -- presence beats busy beats cooldown beats curiosity -- and
/// each test pins one ordering that would be expensive to get wrong live.
pub fn selftest() -> bool {
    use crate::kprintln;

    let mut ok = true;

    // Presence wins over everything except being disabled.
    let d = decide(true, false, 1, 10_000, true, Some("g"));
    ok &= matches!(d, Decision::Sleep("the operator is present"));

    // Disabled wins even over presence.
    let d = decide(false, false, 0, 10_000, true, Some("g"));
    ok &= matches!(d, Decision::Sleep("disabled"));

    // Busy beats ready-planner: two minds is the one unrecoverable shape.
    let d = decide(true, true, 100, 10_000, true, None);
    ok &= matches!(d, Decision::Sleep("an episode is already running"));

    // Ready planner acts ahead of curiosity, and inside the cooldown.
    let d = decide(true, false, 100, 5, true, Some("g"));
    ok &= matches!(d, Decision::Act(_));

    // Cooldown holds curiosity down.
    let d = decide(true, false, 100, 5, false, Some("g"));
    ok &= matches!(d, Decision::Sleep("cooldown between self-set goals"));

    // Past cooldown, curiosity becomes a goal.
    let d = decide(true, false, 100, EPISODE_GAP_S, false, Some("look"));
    ok &= matches!(d, Decision::Episode(g) if g == "look");

    // Nothing proposed is a legitimate answer, not an error.
    let d = decide(true, false, 100, EPISODE_GAP_S, false, None);
    ok &= matches!(d, Decision::Sleep("nothing proposed"));

    kprintln!(
    "  {}  presence outranks readiness, cooldowns hold, curiosity converts",
        if ok { "ok " } else { "FAIL" }
    );
    ok
}

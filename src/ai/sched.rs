//! Who decodes first, decided between tokens.
//!
//! Contexts (`ai::ctx`) took away the reason two callers could not both use
//! the model; they did not take away that both are competing for the same
//! memory bandwidth on the same core. A forward pass reads the whole of the
//! weights, so two at once each run at roughly half speed -- an operator's
//! `ask` during an episode would answer, but slowly enough that "you can talk
//! to it while it works" would be true only on paper.
//!
//! So priority is a closed order, `Chat > Agent > Judge`, and a lower context
//! **parks at a token boundary** while a higher one has foreground work. The
//! boundary is the top of `Model::forward`: every decode loop in the tree goes
//! through it, so no loop has to remember to call anything, and parking there
//! leaves the parked context's cache exactly as it was -- its next token is the
//! one it would have computed anyway. A parked episode resumes where it
//! stopped, which is what makes this interruption rather than cancellation.
//!
//! What counts as foreground is declared by the work, not inferred: `ask`
//! holds a `Foreground` guard for the length of a turn, and the agent task
//! holds one for the length of a job. The resident mind deliberately does not
//! -- it is the machine talking to itself, and nothing waits on it.
//!
//! Prefill is a batch and does not pass a boundary until it finishes. A long
//! prompt prefilled by an episode therefore delays a chat turn by that one
//! prefill, which is bounded by the context length and measured in seconds,
//! not minutes.

use super::ctx::{self, Ctx};
use core::sync::atomic::{AtomicUsize, Ordering};

/// Foreground turns in progress, per context.
static ACTIVE: [AtomicUsize; ctx::COUNT] = [const { AtomicUsize::new(0) }; ctx::COUNT];
/// Times a context parked, per context. Read by `diag ctx` and `ctx`.
static PARKED: [AtomicUsize; ctx::COUNT] = [const { AtomicUsize::new(0) }; ctx::COUNT];

/// Foreground work in a context. Released by `Drop`, so a turn that returns
/// early cannot leave every lower context parked for the rest of the boot.
pub struct Foreground {
    c: Ctx,
}

impl Drop for Foreground {
    fn drop(&mut self) {
        ACTIVE[self.c as usize].fetch_sub(1, Ordering::AcqRel);
    }
}

/// Declare foreground work in the running task's context.
pub fn foreground() -> Foreground {
    let c = ctx::current();
    ACTIVE[c as usize].fetch_add(1, Ordering::AcqRel);
    Foreground { c }
}

fn outranked(me: Ctx) -> bool {
    // Lower discriminant is higher priority; the order is the enum's.
    Ctx::ALL[..me as usize].iter().any(|h| ACTIVE[*h as usize].load(Ordering::Acquire) > 0)
}

/// Park while a higher-priority context has foreground work.
///
/// Sleeps a tick at a time rather than spinning, so a parked episode costs
/// the core nothing; the sleeper re-checks on every wake because the guard
/// is dropped by another task and nobody wakes it by name.
pub fn boundary() {
    let me = ctx::current();
    if !outranked(me) {
        return;
    }
    PARKED[me as usize].fetch_add(1, Ordering::AcqRel);
    while outranked(me) {
        let now = crate::dev::lapic::ticks();
        crate::task::sleep_until(now + 1);
    }
}

pub fn parked(c: Ctx) -> usize {
    PARKED[c as usize].load(Ordering::Acquire)
}

pub fn active(c: Ctx) -> usize {
    ACTIVE[c as usize].load(Ordering::Acquire)
}

static HELPER_DONE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

fn helper() {
    ctx::adopt(Ctx::Agent);
    boundary();
    // Back to the default before the slot is reclaimed: the context table is
    // keyed by task id, and a reused slot must not inherit `Agent`.
    ctx::adopt(Ctx::Chat);
    HELPER_DONE.store(true, Ordering::Release);
}

/// Claims for `diag ctx`: parking happens, is seen from another task, and
/// ends when the foreground turn does. Seen from outside, because a task that
/// never parked and one that parked and woke both finish -- only the middle
/// tells them apart, which is `diag migrate`'s argument about sleeping.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut rows = alloc::vec::Vec::new();
    let before = parked(Ctx::Agent);
    HELPER_DONE.store(false, Ordering::Release);
    let fg = foreground();
    let spawned = crate::task::spawn("sched-check", helper).is_some();
    // Long enough for the helper to reach its boundary several times over.
    let until = crate::dev::lapic::ticks() + 20;
    crate::task::sleep_until(until);
    let held_back = !HELPER_DONE.load(Ordering::Acquire);
    let counted = parked(Ctx::Agent) > before;
    drop(fg);
    let deadline = crate::dev::lapic::ticks() + 200;
    while !HELPER_DONE.load(Ordering::Acquire) && crate::dev::lapic::ticks() < deadline {
        crate::task::sleep_until(crate::dev::lapic::ticks() + 1);
    }
    let released = HELPER_DONE.load(Ordering::Acquire);
    rows.push(("a lower context's task was spawned for the check", spawned));
    rows.push(("it parks at a token boundary while chat has a turn", held_back && counted));
    rows.push(("and resumes when the turn ends", released));
    rows.push(("nothing is left in the foreground afterwards", (0..ctx::COUNT).all(|i| ACTIVE[i].load(Ordering::Acquire) == 0)));
    rows
}

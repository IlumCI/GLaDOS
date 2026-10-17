//! Which Linux session a task belongs to, and what each session owns.
//!
//! A session is what `linux run` (or a background job) starts: one program as
//! pid 1, and whatever it forks or clones. Until this existed there was exactly
//! one, so its main task, its deadline, its syscall trace, its quarantine flag
//! and its thread pool's exit flag were each a single static -- and two
//! sessions at once would have ended each other's processes, since the first to
//! finish swept every forked child and stopped every thread on the machine.
//!
//! **Each session is a pid namespace.** Its main guest answers pid 1 to its own
//! processes, `kill` and `wait4` see only its own family, and nothing in one
//! session can name a process in another. That is what Linux containers do and
//! it is also the only arrangement that needs no renumbering: every session's
//! first program already believed it was pid 1.
//!
//! **What the timer interrupt reads is atomic, everything else is task-side.**
//! `overran` is asked from the timer handler while a guest is at ring 3, so a
//! session's deadline, main task and exit flag are arrays of atomics indexed by
//! session. The syscall trace is a `Vec` and lives behind `Racy`, which is only
//! ever touched from a syscall or from the task running the session.
//!
//! A task learns its session from whichever of the three doors into ring 3 it
//! took -- `run`, `run_thread`, a forked child's `body` -- and forgets it on the
//! way out, exactly as `IN_GUEST` does in `syscall`.

use super::syscall::Call;
use crate::sync::Racy;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// How many sessions may run at once. Each costs a main task, a 16 KiB syscall
/// stack and a share of the shared fork and thread pools, so the bound is about
/// those pools rather than about memory.
pub const MAX: usize = 4;

const NONE: usize = usize::MAX;

static TASK_SESSION: [AtomicUsize; crate::task::MAX_TASKS] =
    [const { AtomicUsize::new(NONE) }; crate::task::MAX_TASKS];

static LIVE: [AtomicBool; MAX] = [const { AtomicBool::new(false) }; MAX];
static MAIN_TASK: [AtomicUsize; MAX] = [const { AtomicUsize::new(NONE) }; MAX];
static MAIN_GUEST: [AtomicUsize; MAX] = [const { AtomicUsize::new(NONE) }; MAX];
/// Zero means no deadline, which is also what "not armed" reads as.
static DEADLINE: [AtomicU64; MAX] = [const { AtomicU64::new(0) }; MAX];
static EXITING: [AtomicBool; MAX] = [const { AtomicBool::new(false) }; MAX];
static QUARANTINED: [AtomicBool; MAX] = [const { AtomicBool::new(false) }; MAX];
static TRACE: Racy<[Vec<Call>; MAX]> = Racy::new([const { Vec::new() }; MAX]);
/// The most recent calls once the trace above is full, as a ring.
///
/// The trace keeps the *first* calls because those are where a program asks
/// for what it needs; a program that dies of corruption dies thousands of
/// calls later, and what it did just before is the only lead. So the two
/// halves answer different questions and both are kept.
static TAIL: Racy<[(Vec<Call>, usize); MAX]> = Racy::new([const { (Vec::new(), 0) }; MAX]);
const TAIL_CAP: usize = 256;

/// Take a free session number, or `None` when all are in use.
pub fn claim() -> Option<usize> {
    for (i, l) in LIVE.iter().enumerate() {
        if l.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            MAIN_TASK[i].store(NONE, Ordering::Release);
            MAIN_GUEST[i].store(NONE, Ordering::Release);
            DEADLINE[i].store(0, Ordering::Release);
            EXITING[i].store(false, Ordering::Release);
            QUARANTINED[i].store(false, Ordering::Release);
            unsafe { (*TRACE.get())[i].clear() };
            unsafe { (*TAIL.get())[i] = (Vec::new(), 0) };
            return Some(i);
        }
    }
    None
}

/// Give a session number back. Its trace is answered by `take_trace` first.
pub fn release(s: usize) {
    if s < MAX {
        MAIN_TASK[s].store(NONE, Ordering::Release);
        MAIN_GUEST[s].store(NONE, Ordering::Release);
        DEADLINE[s].store(0, Ordering::Release);
        LIVE[s].store(false, Ordering::Release);
    }
}

pub fn live_count() -> usize {
    LIVE.iter().filter(|l| l.load(Ordering::Acquire)).count()
}

fn task_index() -> usize {
    crate::task::current().min(crate::task::MAX_TASKS - 1)
}

/// This task speaks for session `s` from now on.
pub fn enter(s: usize) {
    TASK_SESSION[task_index()].store(s, Ordering::Release);
}

/// And no longer does.
pub fn leave() {
    TASK_SESSION[task_index()].store(NONE, Ordering::Release);
}

/// The session the running task belongs to, if it is inside one.
pub fn current() -> Option<usize> {
    let s = TASK_SESSION[task_index()].load(Ordering::Acquire);
    (s < MAX).then_some(s)
}

pub fn set_main(s: usize, task: usize, guest: usize) {
    MAIN_TASK[s].store(task, Ordering::Release);
    MAIN_GUEST[s].store(guest, Ordering::Release);
}

pub fn main_task_of(s: usize) -> Option<usize> {
    let t = MAIN_TASK.get(s)?.load(Ordering::Acquire);
    (t != NONE).then_some(t)
}

pub fn main_guest_of(s: usize) -> Option<usize> {
    let g = MAIN_GUEST.get(s)?.load(Ordering::Acquire);
    (g != NONE).then_some(g)
}

pub fn set_deadline(s: usize, tick: u64) {
    DEADLINE[s].store(tick, Ordering::Release);
}

pub fn deadline_of(s: usize) -> u64 {
    DEADLINE.get(s).map(|d| d.load(Ordering::Acquire)).unwrap_or(0)
}

pub fn begin_exit(s: usize) {
    EXITING[s].store(true, Ordering::Release);
}

pub fn exiting(s: usize) -> bool {
    EXITING.get(s).is_some_and(|e| e.load(Ordering::Acquire))
}

pub fn quarantine_main(s: usize, on: bool) {
    QUARANTINED[s].store(on, Ordering::Release);
}

pub fn main_quarantined(s: usize) -> bool {
    QUARANTINED.get(s).is_some_and(|q| q.load(Ordering::Acquire))
}

/// Append a call to the running task's session trace.
pub fn record(c: Call, cap: usize) {
    if let Some(s) = current() {
        let t = unsafe { &mut (*TRACE.get())[s] };
        if t.len() < cap {
            t.push(c);
            return;
        }
        let (ring, head) = unsafe { &mut (*TAIL.get())[s] };
        if ring.len() < TAIL_CAP {
            ring.push(c);
        } else {
            ring[*head] = c;
            *head = (*head + 1) % TAIL_CAP;
        }
    }
}

/// The last calls past the trace's cap, oldest first, emptied.
/// The tail as it stands, leaving it in place: what a crash report prints.
pub fn peek_tail(s: usize) -> Vec<Call> {
    let (ring, head) = unsafe { &(*TAIL.get())[s] };
    let mut out = Vec::with_capacity(ring.len());
    out.extend_from_slice(&ring[*head..]);
    out.extend_from_slice(&ring[..*head]);
    if out.is_empty() {
        out.extend_from_slice(unsafe { &(*TRACE.get())[s] });
    }
    out
}

pub fn take_tail(s: usize) -> Vec<Call> {
    let (ring, head) = core::mem::take(unsafe { &mut (*TAIL.get())[s] });
    let mut out = Vec::with_capacity(ring.len());
    out.extend_from_slice(&ring[head..]);
    out.extend_from_slice(&ring[..head]);
    out
}

/// The session's trace, emptied.
pub fn take_trace(s: usize) -> Vec<Call> {
    core::mem::take(unsafe { &mut (*TRACE.get())[s] })
}

/// Device-free checks of the claim and lookup arithmetic.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    let before = live_count();
    let a = claim();
    let b = claim();
    out.push(("two sessions can be claimed at once", a.is_some() && b.is_some() && a != b));
    if let (Some(a), Some(b)) = (a, b) {
        set_main(a, 7, 3);
        set_deadline(a, 100);
        begin_exit(b);
        out.push(("one session's main task is not the other's", main_task_of(a) == Some(7) && main_task_of(b).is_none()));
        out.push(("one session's deadline is not the other's", deadline_of(a) == 100 && deadline_of(b) == 0));
        out.push(("one session ending does not end the other", exiting(b) && !exiting(a)));
        release(a);
        release(b);
    }
    out.push(("released sessions are free again", live_count() == before));
    out.push(("a session past the table has no main task", main_task_of(MAX).is_none()));
    out
}

//! Work that runs beside the shell: a guest program on a task of its own.
//!
//! Until this existed a guest held the shell for its whole life. `linux run`
//! entered ring 3 *on the shell's task*, so the prompt came back only when the
//! program ended, and anything meant to keep running -- a Wayland client, an
//! audio daemon, a compositor -- could not be started at all without the
//! machine being unusable for as long as it ran.
//!
//! **Runtime-neutral on purpose.** Linux is the first of three guest runtimes
//! (macOS and NT are planned), and what a job is does not depend on which ABI
//! the program speaks: a title, a task, a way to end it, and what it said when
//! it ended. A runtime hands `start` a closure that does its own loading and
//! running and answers a one-line summary, plus a `kill` that knows how to end
//! a task inside that runtime. Everything ABI-shaped stays in the runtime.
//!
//! **The closure runs on a fresh task**, which is the whole mechanism. `spawn`
//! takes a bare `fn()`, so the work cannot be handed over as an argument: it
//! is parked in the table as `Starting`, and the task's entry claims the first
//! `Starting` job it finds. Several starts racing each other are fine, since
//! each entry claims exactly one under the lock and every job has a task.
//!
//! What a job does **not** get is the terminal. Output still reaches the
//! console, interleaved with whatever the shell prints, which is what a Unix
//! shell does with a background job too; input it never sees, since stdin is
//! the runtime's business and Linux's already reads as end-of-file.

use crate::sync::Spin;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// What a runtime supplies about itself.
pub struct Runtime {
    pub name: &'static str,
    /// End the guest running on `task`. Answers whether the request landed;
    /// the task then ends itself at its next safe point, which is the only way
    /// a guest can be ended here (`linux::syscall::doom` says why).
    pub kill: fn(task: usize) -> bool,
}

type Work = Box<dyn FnOnce() -> String + Send>;

enum State {
    Starting(Work),
    Running,
    Done(String),
}

struct Job {
    id: u32,
    runtime: &'static Runtime,
    title: String,
    task: Option<usize>,
    started: u64,
    ended: u64,
    state: State,
}

static JOBS: Spin<Vec<Job>> = Spin::new(Vec::new());
static NEXT_ID: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(1);

/// Finished jobs kept for `jobs` to report. Bounded so a script that starts
/// thousands of short jobs does not grow the table forever.
const KEEP_DONE: usize = 16;

/// Start `work` on a task of its own. Answers the job's number.
pub fn start(runtime: &'static Runtime, title: String, work: Work) -> Result<u32, &'static str> {
    let id = NEXT_ID.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    {
        let mut jobs = JOBS.lock_irq();
        // Drop the oldest finished ones first, never a live one.
        while jobs.iter().filter(|j| matches!(j.state, State::Done(_))).count() >= KEEP_DONE {
            let Some(i) = jobs.iter().position(|j| matches!(j.state, State::Done(_))) else { break };
            jobs.remove(i);
        }
        jobs.push(Job {
            id,
            runtime,
            title,
            task: None,
            started: crate::dev::lapic::ticks(),
            ended: 0,
            state: State::Starting(work),
        });
    }
    if crate::task::spawn("job", entry).is_none() {
        // Taken back out, so the table does not hold work no task will claim.
        JOBS.lock_irq().retain(|j| j.id != id);
        return Err("no free task to run it on");
    }
    Ok(id)
}

fn entry() {
    let me = crate::task::current();
    let claimed = {
        let mut jobs = JOBS.lock_irq();
        jobs.iter_mut().find(|j| matches!(j.state, State::Starting(_))).map(|j| {
            j.task = Some(me);
            let State::Starting(work) = core::mem::replace(&mut j.state, State::Running) else {
                unreachable!()
            };
            (j.id, work)
        })
    };
    let Some((id, work)) = claimed else { return };
    // Outside the lock: the work is a whole program's lifetime, and holding a
    // lock with interrupts masked across it would stop the machine.
    let summary = work();
    let title = {
        let mut jobs = JOBS.lock_irq();
        jobs.iter_mut().find(|j| j.id == id).map(|j| {
            j.ended = crate::dev::lapic::ticks();
            j.state = State::Done(summary.clone());
            j.task = None;
            j.title.clone()
        })
    };
    // **Held for the shell's next prompt rather than printed now**, which is
    // what a Unix shell does with "Done" and for the same reason: printed
    // from here it lands after a prompt already on screen, the operator's
    // half-typed line scrolls away under it, and `drive.py` -- which waits for
    // a prompt at the end of output -- never sees one again.
    if let Some(t) = title {
        NOTICES.lock_irq().push(alloc::format!("[job {}] {}: {}", id, t, summary));
    }
}

static NOTICES: Spin<Vec<String>> = Spin::new(Vec::new());

/// Jobs that have ended since the shell last asked, oldest first.
pub fn take_notices() -> Vec<String> {
    core::mem::take(&mut *NOTICES.lock_irq())
}

/// Whether any job of `runtime` is starting or running.
pub fn live(runtime: &'static Runtime) -> bool {
    JOBS.lock_irq()
        .iter()
        .any(|j| core::ptr::eq(j.runtime, runtime) && !matches!(j.state, State::Done(_)))
}

/// Ask job `id` to end. Answers what happened, in words.
pub fn kill(id: u32) -> &'static str {
    let mut jobs = JOBS.lock_irq();
    let Some(i) = jobs.iter().position(|j| j.id == id) else {
        return "no such job";
    };
    match (&jobs[i].state, jobs[i].task) {
        (State::Done(_), _) => "it has already ended",
        // Not yet claimed: nothing has run, so it can simply be withdrawn. The
        // task that would have claimed it finds nothing and returns.
        (State::Starting(_), _) => {
            jobs.remove(i);
            "withdrawn before it started"
        }
        (State::Running, Some(t)) => {
            if (jobs[i].runtime.kill)(t) {
                "asked to end; it stops at its next safe point"
            } else {
                "running, but not yet inside its runtime -- try again in a moment"
            }
        }
        (State::Running, None) => "running with no task, which should not happen",
    }
}

/// One row per job, for the `jobs` verb.
pub struct Row {
    pub id: u32,
    pub runtime: &'static str,
    pub title: String,
    pub task: Option<usize>,
    pub state: &'static str,
    pub summary: String,
    pub seconds: u64,
}

pub fn list() -> Vec<Row> {
    let now = crate::dev::lapic::ticks();
    JOBS.lock_irq()
        .iter()
        .map(|j| {
            let (state, summary, until) = match &j.state {
                State::Starting(_) => ("starting", String::new(), now),
                State::Running => ("running", String::new(), now),
                State::Done(s) => ("done", s.clone(), j.ended),
            };
            Row {
                id: j.id,
                runtime: j.runtime.name,
                title: j.title.clone(),
                task: j.task,
                state,
                summary,
                seconds: until.saturating_sub(j.started) / crate::TIMER_HZ as u64,
            }
        })
        .collect()
}

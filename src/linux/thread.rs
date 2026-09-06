//! Threads for a guest, which one address space makes easier rather than
//! harder.
//!
//! **`fork` needs two address spaces and this system has one, deliberately.**
//! Threads need the opposite: `CLONE_VM` says "share the address space", which
//! is a request this kernel can grant by doing nothing at all. So the founding
//! constraint that makes `fork` impossible here is the same one that makes a
//! thread nearly free, and that asymmetry is worth stating because it is easy
//! to read "no processes" as "no concurrency".
//!
//! What a thread actually needs is four things, and three of them are already
//! in this kernel:
//!
//! - **A stack.** The guest allocates it and hands it to `clone`.
//! - **A scheduler.** `task.rs` preempts at 100 Hz and has since long before
//!   any guest existed.
//! - **Its own `FS`**, for thread-local storage. `arch_prctl` already sets it;
//!   what is new is that it has to survive a context switch.
//! - **Its own syscall entry state**, and this is the piece that did not
//!   exist. `syscall.rs` says so in its own header: "that stack is one static,
//!   so the handler is not reentrant ... stage 0 runs one guest with no
//!   threads. Both of those stop being true later." This is later.
//!
//! ### The entry path is per-task now, and no assembly changed
//!
//! Three globals carry the syscall entry across the ring boundary: where the
//! guest's `rsp` went, which stack the handler runs on, and where to longjmp
//! back to. With one guest they are constants. With two threads they are three
//! ways to corrupt each other, since a thread that blocks inside a syscall
//! leaves them live while another thread enters one.
//!
//! They are saved and restored by the scheduler now, exactly as the FPU area
//! already was and in the same place. That is why the stub is untouched: a
//! global that is only read while its own task is running *is* per-task, as
//! long as somebody swaps it at the switch. The alternative was `swapgs` and a
//! per-thread block, which would have collided with `cpu::percpu` owning GS.
//!
//! ### A pool, not a task per thread
//!
//! `MAX_TASKS` is 24 and a kernel task that returns is not reclaimed -- it
//! spins in `yield_now` forever, which is fine for the half-dozen tasks this
//! kernel spawns at boot and fatal for a program that creates a thread per
//! frame. Reclaiming slots properly means teaching the scheduler about a task
//! that is finished, and the outgoing task's state is written unconditionally
//! in `schedule`, so that is real surgery on the most delicate loop here.
//!
//! So guest threads come from a pool that grows to `MAX_THREADS` and never
//! shrinks: a finished thread parks its task rather than ending it, and the
//! next `clone` takes it back. The limit is on *concurrent* threads instead of
//! on threads ever created, which is the honest shape, and a machine that
//! never runs one spawns nothing.

use crate::sync::Racy;
use alloc::alloc::{alloc, Layout};
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// How many guest threads may exist at once.
///
/// Eight against `MAX_TASKS` of 24, leaving room for the shell, the clock, the
/// resident mind and the agent. A ninth `clone` is `EAGAIN`, which is what
/// Linux answers when it is out of task structures and what every threading
/// library already handles.
pub const MAX_THREADS: usize = 8;

/// Sixteen KiB, the same as the main guest's, and for the same reason: the
/// dispatcher is ordinary Rust that formats and prints, so a stack sized for
/// the assembly is a handler that works until somebody adds a `kprintln!`.
const SYSCALL_STACK: usize = 16 * 1024;

/// The flags a thread is made of.
///
/// `CLONE_THREAD` is the one that matters: it says the child joins this thread
/// group rather than becoming a process. Without it `clone` is `fork`, which
/// needs a second address space, so it is refused rather than approximated.
pub const CLONE_VM: u64 = 0x0000_0100;
pub const CLONE_FS: u64 = 0x0000_0200;
pub const CLONE_FILES: u64 = 0x0000_0400;
pub const CLONE_SIGHAND: u64 = 0x0000_0800;
pub const CLONE_THREAD: u64 = 0x0001_0000;
pub const CLONE_SYSVSEM: u64 = 0x0004_0000;
pub const CLONE_SETTLS: u64 = 0x0008_0000;
pub const CLONE_PARENT_SETTID: u64 = 0x0010_0000;
pub const CLONE_CHILD_CLEARTID: u64 = 0x0020_0000;

/// What a slot is asked to run.
#[derive(Clone, Copy)]
pub struct Work {
    /// Where the child resumes, which is where the *parent* would have.
    ///
    /// A cloned thread does not start at an entry point: it returns from
    /// `clone` with `rax` zero, at the same instruction the parent returns to,
    /// on a different stack. `syscall` leaves that address in `rcx`, so the
    /// child's entry is the parent's return address and nothing has to be
    /// invented.
    pub rip: u64,
    pub rsp: u64,
    pub fs: u64,
    pub tid: u64,
    /// Where to write a zero and then wake, when this thread ends.
    ///
    /// `CLONE_CHILD_CLEARTID` is the whole of how `pthread_join` works: the
    /// joiner futex-waits on this word, and the kernel clearing it is the
    /// notification. A thread library given a kernel that ignores this waits
    /// forever on a thread that finished.
    pub ctid: u64,
}

#[derive(Clone, Copy)]
struct Slot {
    /// The kernel task backing this slot, once one has been spawned.
    task: Option<usize>,
    /// The handler stack that task's guest threads run their syscalls on.
    ///
    /// **On the slot and not on the `Task`, which is where it was.** A task
    /// stops carrying ring-3 state the moment its thread ends -- that is the
    /// whole point, so a parked pool task costs nothing at a switch -- and the
    /// stack went out with it. The second thread to use a slot therefore
    /// entered its first syscall with the handler stack at zero and pushed
    /// onto a null pointer. It ran perfectly once.
    stack: u64,
    work: Option<Work>,
    /// Whether a guest thread is running on it right now.
    live: bool,
}

static SLOTS: Racy<[Slot; MAX_THREADS]> =
    Racy::new([Slot { task: None, stack: 0, work: None, live: false }; MAX_THREADS]);

/// The next thread id to hand out.
///
/// From 2, because `getpid` answers 1 and the main thread's `gettid` has to
/// agree with it: on Linux a single-threaded process has `tid == pid`, and a
/// library that finds them different concludes it is not the main thread.
static NEXT_TID: AtomicU64 = AtomicU64::new(2);

/// How many guest threads are running, main thread excluded.
static LIVE: AtomicUsize = AtomicUsize::new(0);

pub fn live() -> usize {
    LIVE.load(Ordering::Acquire)
}

/// Set when the process is ending, so every other thread stops at its next
/// syscall.
///
/// A flag rather than reaching into the other tasks, because ending a thread
/// means longjmping out of *its* stack and only that thread can do that. It
/// checks on the way into every syscall, which bounds how long a thread
/// outlives `exit_group` by however long it takes to make one -- and the run
/// deadline is the backstop for a thread that makes none.
static EXITING: AtomicU64 = AtomicU64::new(0);

pub fn exiting() -> bool {
    EXITING.load(Ordering::Acquire) != 0
}

pub fn begin_exit() {
    EXITING.store(1, Ordering::Release);
}

pub fn reset() {
    EXITING.store(0, Ordering::Release);
    LIVE.store(0, Ordering::Release);
    let s = unsafe { &mut *SLOTS.get() };
    for slot in s.iter_mut() {
        slot.work = None;
        slot.live = false;
    }
}

/// Whether these flags describe a thread this kernel can make.
///
/// **Refused rather than approximated, and the reason is not caution.** A
/// `clone` without `CLONE_VM` wants a copy of the address space, and there is
/// one address space here by construction; a `clone` without `CLONE_THREAD`
/// wants a new process, and there are no processes. Answering either with a
/// thread would hand a program two names for one thing and let it write
/// through both.
pub fn is_thread(flags: u64) -> bool {
    flags & CLONE_VM != 0 && flags & CLONE_THREAD != 0
}

/// Start a thread. Answers its tid, or an errno.
pub fn spawn(flags: u64, stack: u64, ptid: u64, ctid: u64, tls: u64, rip: u64) -> Result<u64, u64> {
    const EAGAIN: u64 = (-11i64) as u64;
    const EINVAL: u64 = (-22i64) as u64;
    if stack == 0 || stack % 16 != 0 {
        // Linux does not check this and a misaligned stack is a `movaps` fault
        // several frames into the thread function, which reads as a bug in the
        // program. Refused here because this kernel can say so cheaply and the
        // alternative diagnostic is a register dump.
        return Err(EINVAL);
    }
    let tid = NEXT_TID.fetch_add(1, Ordering::Relaxed);
    let work = Work {
        rip,
        rsp: stack,
        fs: if flags & CLONE_SETTLS != 0 { tls } else { 0 },
        tid,
        ctid: if flags & CLONE_CHILD_CLEARTID != 0 { ctid } else { 0 },
    };

    let idx = {
        let s = unsafe { &mut *SLOTS.get() };
        let Some(i) = s.iter().position(|x| !x.live && x.work.is_none()) else {
            return Err(EAGAIN);
        };
        s[i].work = Some(work);
        s[i].live = true;
        i
    };
    LIVE.fetch_add(1, Ordering::Release);

    // The task is spawned on first use and kept afterwards. A machine that
    // never runs a threaded guest spawns none of these.
    let have = unsafe { (*SLOTS.get())[idx].task };
    if have.is_none() {
        let Some(t) = crate::task::spawn("guest thread", body) else {
            let s = unsafe { &mut *SLOTS.get() };
            s[idx].work = None;
            s[idx].live = false;
            LIVE.fetch_sub(1, Ordering::Release);
            return Err(EAGAIN);
        };
        let stack_top = match alloc_stack() {
            Some(a) => a,
            None => {
                let s = unsafe { &mut *SLOTS.get() };
                s[idx].work = None;
                s[idx].live = false;
                LIVE.fetch_sub(1, Ordering::Release);
                return Err(EAGAIN);
            }
        };
        let s = unsafe { &mut *SLOTS.get() };
        s[idx].task = Some(t);
        s[idx].stack = stack_top;
    }

    // `CLONE_PARENT_SETTID` writes the child's id where the parent can see it,
    // and the parent is guaranteed to observe it before `clone` returns -- so
    // it is written here rather than by the child, which may not have run yet.
    if flags & CLONE_PARENT_SETTID != 0 && ptid != 0 {
        unsafe { core::ptr::write_unaligned(ptid as *mut u32, tid as u32) };
    }
    Ok(tid)
}

fn alloc_stack() -> Option<u64> {
    let layout = Layout::from_size_align(SYSCALL_STACK, 16).ok()?;
    let p = unsafe { alloc(layout) };
    if p.is_null() {
        return None;
    }
    Some(p as u64 + SYSCALL_STACK as u64)
}

/// Which slot the running task is, if it is a guest thread.
fn my_slot() -> Option<usize> {
    let me = crate::task::current();
    let s = unsafe { &*SLOTS.get() };
    s.iter().position(|x| x.task == Some(me))
}

/// The tid of whatever is running.
///
/// The main thread answers 1, which is `getpid`'s answer, because on Linux a
/// process's first thread has `tid == pid` and a library that finds otherwise
/// concludes it is not the main thread and takes a different path.
pub fn current_tid() -> u64 {
    match my_slot() {
        Some(i) => unsafe { (*SLOTS.get())[i].work.map(|w| w.tid).unwrap_or(1) },
        None => 1,
    }
}

/// Where a guest thread's `clone` is answered and its life is spent.
fn body() {
    loop {
        let work = my_slot().and_then(|i| unsafe { (*SLOTS.get())[i].work });
        let Some(w) = work else {
            // Parked. `hlt` rather than a spin, so an idle pool costs one
            // wake-up per timer tick instead of every scheduler round.
            crate::port::idle();
            crate::task::yield_now();
            continue;
        };
        let stack = my_slot().map(|i| unsafe { (*SLOTS.get())[i].stack }).unwrap_or(0);
        crate::linux::syscall::run_thread(w, stack);
        if let Some(i) = my_slot() {
            let s = unsafe { &mut *SLOTS.get() };
            s[i].work = None;
            s[i].live = false;
        }
        LIVE.fetch_sub(1, Ordering::Release);
    }
}

/// What `diag linux` asks of the thread table.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out = alloc::vec::Vec::new();

    // The shape test, and both halves matter. Dropping `CLONE_VM` asks for a
    // copied address space and dropping `CLONE_THREAD` asks for a process;
    // this kernel has exactly one of the first and none of the second.
    let thread = CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD;
    out.push((
        "a clone is a thread only when it shares the address space and the group",
        is_thread(thread)
            && is_thread(thread | CLONE_SETTLS | CLONE_CHILD_CLEARTID)
            && !is_thread(thread & !CLONE_VM)
            && !is_thread(thread & !CLONE_THREAD)
            && !is_thread(0),
    ));
    // The flag values themselves, because they are copied from a header and a
    // digit wrong would make `CLONE_SETTLS` read as something else entirely.
    out.push((
        "and the flags are the ones the header carries, not ones near them",
        CLONE_VM == 0x100
            && CLONE_THREAD == 0x1_0000
            && CLONE_SETTLS == 0x8_0000
            && CLONE_CHILD_CLEARTID == 0x20_0000,
    ));
    out.push((
        "a stack that is not sixteen-aligned is refused, since the fault is far away",
        spawn(thread, 0, 0, 0, 0, 0).is_err() && spawn(thread, 0x1008, 0, 0, 0, 0).is_err(),
    ));
    out.push((
        "the main thread's tid is its pid, or a library decides it is not the main one",
        current_tid() == 1,
    ));
    out.push((
        "and the pool is bounded, leaving room for the shell and the resident mind",
        MAX_THREADS + 4 <= crate::task::MAX_TASKS,
    ));
    out
}

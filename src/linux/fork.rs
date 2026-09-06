//! `fork`, which this kernel spent four commits earning the right to write.
//!
//! `syscall.rs` called it "the one thing this system cannot grow into" and
//! `thread.rs` said the same, and both were true of a kernel with one set of
//! page tables. What made it possible was not this file: it was giving a guest
//! a page-table root of its own, and then giving every one of its regions an
//! address in an arena that every guest shares. A child is its parent's memory
//! at *the same virtual addresses*, and that sentence is either impossible or
//! nearly free depending on whether those addresses were chosen by the guest's
//! layout or by wherever the heap had room.
//!
//! What is copied and what is shared follows Linux rather than convenience.
//! Memory is copied, eagerly and in full -- no copy-on-write, because that
//! needs a fault handler that can service a write to a read-only page and this
//! kernel's every vector is fatal. Descriptors are *shared*, because `Fd` holds
//! an `Rc` around the open file description and Linux says a forked child
//! shares the parent's offsets. Both are stated here because the two obvious
//! guesses are wrong in opposite directions.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::syscall::{self, Frame, Resume};
use crate::sync::Racy;

/// How many children may be live at once.
///
/// A bound rather than a `Vec` because each one costs a kernel task, and
/// `task::MAX_TASKS` is 24 with no way to reclaim a finished task's slot --
/// the same constraint `thread`'s pool records. Four is what a shell needs to
/// run a pipeline.
pub const MAX_CHILDREN: usize = 4;

/// The syscall stack a child's ring-3 entry runs on.
///
/// Its own, and that is not an optimisation. The handler stack is one of the
/// three globals `schedule` swaps per task, so a child entering a syscall
/// while its parent sits inside one would otherwise overwrite where the
/// parent's `rsp` went -- which `syscall::run` already records as the reason
/// the main thread carries entry state at all.
const CHILD_STACK: usize = 16 * 1024;

#[derive(Clone, Copy)]
struct Slot {
    task: Option<usize>,
    stack: u64,
    /// The guest table entry this child speaks for.
    guest: usize,
    work: Option<Resume>,
    live: bool,
    pid: u64,
    /// What it exited with, once it has. `wait4` reads this.
    status: Option<i32>,
}

static SLOTS: Racy<[Slot; MAX_CHILDREN]> = Racy::new(
    [Slot { task: None, stack: 0, guest: 0, work: None, live: false, pid: 0, status: None };
        MAX_CHILDREN],
);

/// The next process id. From 2, because `getpid` answers 1 for the first guest.
static NEXT_PID: AtomicU64 = AtomicU64::new(2);

static LIVE: AtomicUsize = AtomicUsize::new(0);

pub fn live() -> usize {
    LIVE.load(Ordering::Acquire)
}

/// Forget every child. Called from `teardown`, and it clears the *work* only.
///
/// Deliberately not the tasks or their stacks: a pool task that has been
/// spawned is kept and reused, exactly as `thread`'s is, because a kernel task
/// that returns is not reclaimed and spawning one per fork would exhaust
/// `MAX_TASKS` in four runs.
pub fn reset() {
    let s = unsafe { SLOTS.get() };
    for slot in s.iter_mut() {
        slot.work = None;
        slot.live = false;
        slot.status = None;
    }
    LIVE.store(0, Ordering::Release);
}

/// Copy one of the parent's ranges into pages of the child's own.
///
/// **Read through the parent's *virtual* address, which is only legal because
/// the parent's root is installed for the whole syscall.** That is what makes
/// this possible without reaching into the loader for the backing `Exec`s: the
/// kernel is already looking at the guest's memory the way the guest sees it.
///
/// A page the parent does not have present is skipped rather than copied. A
/// linker reserves a span with `PROT_NONE` and lays libraries over part of it,
/// so reading the rest would take a `#PF` at ring 0 inside `fork` -- and this
/// kernel has no handler that could survive it. The child gets a hole where
/// the parent had a reservation, which faults the same way on access; the
/// honest difference is that the parent's would answer `EFAULT` from a bounds
/// check and the child's is simply absent.
fn dup_range(
    tables: &mut crate::mem::space::Space,
    at: u64,
    len: usize,
    owned: &mut Vec<(u64, usize)>,
) -> bool {
    let n = syscall::page_up(len);
    if n == 0 {
        return true;
    }
    let Some(phys) = syscall::alloc_pages(n) else { return false };
    let mut off = 0u64;
    while off < n as u64 {
        let there = crate::mem::paging::query(at + off).is_some_and(|p| p.present);
        if there {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    (at + off) as *const u8,
                    (phys + off) as *mut u8,
                    4096,
                )
            };
            let ok = if at + off >= crate::mem::space::WINDOW {
                tables.map_page(at + off, phys + off, true, true)
            } else {
                tables.map_low(at + off, phys + off, true, true).is_ok()
            };
            if !ok {
                syscall::free_pages(phys, n);
                return false;
            }
        }
        off += 4096;
    }
    owned.push((phys, n));
    true
}

/// `fork`. Answers the child's pid to the parent, and never returns in the
/// child -- the child's first instruction is the one after the parent's
/// `syscall`, on another task.
pub fn fork(f: &Frame) -> u64 {
    const EAGAIN: u64 = (-11i64) as u64;
    const ENOMEM: u64 = (-12i64) as u64;
    const ENOSYS: u64 = (-38i64) as u64;

    // **Refused without a space, and that is a fact about the guest rather
    // than about this call.** A child is a second address space; a guest
    // running on the kernel's root has nowhere to put one, and answering
    // anyway would hand back a "child" that shared every byte with its parent
    // -- two names for one process, and a program free to corrupt itself
    // through both.
    if syscall::guest_root() == 0 {
        return ENOSYS;
    }

    let parent = syscall::current_guest();
    let child = syscall::free_guest_slot();
    if child == parent {
        return EAGAIN;
    }

    let Some(mut tables) = crate::mem::space::Space::sharing_kernel() else { return ENOMEM };

    // Everything the parent owns, at the addresses it owns them at.
    let (regions, maps) = {
        let Some(sp) = (unsafe { syscall::guest_slot() }).as_ref() else { return ENOMEM };
        let mut r: Vec<(u64, usize)> = Vec::new();
        r.push((sp.image.at, sp.image.len));
        if let Some(i) = sp.interp {
            r.push((i.at, i.len));
        }
        r.push((sp.stack.at, sp.stack.len));
        r.push((sp.brk_start, (sp.brk_end - sp.brk_start) as usize));
        let m: Vec<(u64, usize)> = sp.maps.iter().map(|x| (x.at, x.len)).collect();
        (r, m)
    };

    let mut owned: Vec<(u64, usize)> = Vec::new();
    for (at, len) in regions.iter().chain(maps.iter()) {
        if !dup_range(&mut tables, *at, *len, &mut owned) {
            for (p, n) in owned {
                syscall::free_pages(p, n);
            }
            return ENOMEM;
        }
    }

    let pid = NEXT_PID.fetch_add(1, Ordering::Relaxed);

    // A pool slot, and its stack on first use.
    let idx = {
        let s = unsafe { SLOTS.get() };
        let Some(i) = s.iter().position(|x| !x.live && x.work.is_none()) else {
            for (p, n) in owned {
                syscall::free_pages(p, n);
            }
            return EAGAIN;
        };
        if s[i].stack == 0 {
            let Some(st) = syscall::alloc_pages(CHILD_STACK) else {
                for (p, n) in owned {
                    syscall::free_pages(p, n);
                }
                return ENOMEM;
            };
            s[i].stack = st + CHILD_STACK as u64;
        }
        i
    };

    // The child's entry state: the parent's, with `rax` zero. That zero is the
    // whole of how a program tells the two apart, and `rcx` and `r11` are
    // where `syscall` left the return address and flags -- so the child
    // resumes at the instruction after the parent's trap, which is what makes
    // one call return twice.
    let resume = Resume {
        rip: f.rip,
        rsp: syscall::guest_rsp(),
        rflags: f.rflags,
        rax: 0,
        rbx: f.rbx,
        rbp: f.rbp,
        r12: f.r12,
        r13: f.r13,
        r14: f.r14,
        r15: f.r15,
    };

    // The child's guest entry, cloned from the parent's and given the tables
    // and the pages that were just made for it. It frees them itself if it
    // cannot place them, so there is one owner at every moment rather than a
    // window where both this and it might.
    if !syscall::clone_guest(parent, child, tables, owned) {
        return ENOMEM;
    }

    {
        let s = unsafe { SLOTS.get() };
        s[idx].guest = child;
        s[idx].work = Some(resume);
        s[idx].live = true;
        s[idx].pid = pid;
        s[idx].status = None;
    }
    LIVE.fetch_add(1, Ordering::Release);

    let have = unsafe { (*SLOTS.get())[idx].task };
    if have.is_none() {
        let Some(t) = crate::task::spawn("guest child", body) else {
            let s = unsafe { SLOTS.get() };
            s[idx].work = None;
            s[idx].live = false;
            LIVE.fetch_sub(1, Ordering::Release);
            return EAGAIN;
        };
        unsafe { (*SLOTS.get())[idx].task = Some(t) };
    }
    pid
}

/// A pool task. Picks up whatever child it was given and runs it.
fn body() {
    loop {
        // **A kernel task must not run with interrupts masked, and this one
        // inherits them masked.** It is first scheduled from inside the
        // parent's `fork`, and a syscall runs with IF clear through `FMASK`,
        // so the flags this task resumes on are the parent's. Yield from
        // there and the scheduler parks the idle task on a `hlt` that never
        // wakes: no timer, no prompt, and the run deadline cannot fire
        // because firing is something an interrupt does.
        //
        // `run_thread` records this exact failure from the other direction
        // and it cost the whole machine there too. What made it invisible
        // here for one run is that both halves of `fork` worked perfectly --
        // the child ran, printed and exited 7 -- and then everything stopped,
        // which reads as a hung `wait4` rather than as a dead scheduler.
        crate::cpu::enable_interrupts();
        let me = crate::task::current();
        let found = {
            let s = unsafe { SLOTS.get() };
            s.iter().position(|x| x.task == Some(me) && x.work.is_some())
        };
        let Some(i) = found else {
            // `hlt` rather than a spin, so an idle pool costs one wake per
            // timer tick instead of one per scheduler round -- `thread`'s
            // pool makes the same trade.
            crate::port::idle();
            crate::task::yield_now();
            continue;
        };
        let (work, stack, guest) = {
            let s = unsafe { SLOTS.get() };
            (s[i].work.take().unwrap(), s[i].stack, s[i].guest)
        };

        // Flags around the run, for the reason `run_thread` records: `syscall`
        // clears IF through FMASK and a guest that exits leaves through a
        // longjmp that restores a stack rather than a processor state, so
        // without this the pool task goes back to its wait with interrupts off
        // and the core never wakes again.
        let flags: u64;
        unsafe {
            core::arch::asm!("pushfq; pop {}", out(reg) flags, options(nomem, preserves_flags))
        };

        // **This task speaks for the child now, and only while it runs.** The
        // guest table index and the page-table root move together here and
        // nowhere else: `set_current_guest` says which entry the syscall path
        // means, `set_root` says which tables the processor walks, and a
        // child running under its parent's root would be writing to its
        // parent's memory at every address they share -- which is all of them.
        let prev = syscall::current_guest();
        syscall::set_current_guest(guest);
        crate::task::set_root(me, syscall::guest_root());
        crate::task::ring3_active(true, stack);
        // Recorded above, installed here. See `set_syscall_stack`:
        // nothing switches between arming this task and entering the
        // child, so the global would still name the parent's stack.
        unsafe { syscall::set_syscall_stack(stack) };
        let code = unsafe { syscall::enter_resumed(&work) };
        crate::task::ring3_active(false, 0);
        crate::task::set_root(me, 0);
        syscall::set_current_guest(prev);

        {
            let s = unsafe { SLOTS.get() };
            s[i].status = Some((code & 0xFF) as i32);
            s[i].live = false;
        }
        LIVE.fetch_sub(1, Ordering::Release);
        if flags & (1 << 9) != 0 {
            crate::cpu::enable_interrupts();
        }
    }
}

/// `wait4`, in the one shape a shell needs: wait for any child, or for one.
///
/// No `WNOHANG` and no resource usage. Both are refusals this kernel can make
/// honestly -- there is no rusage to report and nothing here samples one --
/// and a shell that asked for either would rather be told than handed zeros.
pub fn wait(pid: i64, status: u64) -> u64 {
    const ECHILD: u64 = (-10i64) as u64;

    loop {
        let found = {
            let s = unsafe { SLOTS.get() };
            s.iter()
                .position(|x| x.pid != 0 && !x.live && x.status.is_some()
                    && (pid <= 0 || x.pid == pid as u64))
        };
        if let Some(i) = found {
            let s = unsafe { SLOTS.get() };
            let code = s[i].status.take().unwrap_or(0);
            let got = s[i].pid;
            s[i].pid = 0;
            if status != 0 && syscall::owns(status, 4) {
                // The wait status is encoded, not the exit code: bits 8..15
                // are what `WEXITSTATUS` shifts back down, and a shell that
                // read a bare code would report every exit as a signal.
                unsafe { core::ptr::write(status as *mut i32, code << 8) };
            }
            return got;
        }
        let any = {
            let s = unsafe { SLOTS.get() };
            s.iter().any(|x| x.pid != 0 && (pid <= 0 || x.pid == pid as u64))
        };
        if !any {
            return ECHILD;
        }
        // A yield loop rather than a wait queue, the bargain `futex` and
        // `nanosleep` already make here and for the same reason: there is no
        // guest scheduler to block against, so this costs the CPU it is not
        // using.
        if unsafe { syscall::kill_if_overdue() } {
            // The deadline is what ends a runaway, and a parent waiting on a
            // child that will never exit is exactly one. Answering `ECHILD`
            // rather than killing from here, because the guest is about to be
            // ended by the timer anyway and a syscall that does not return is
            // harder to read in a trace than one that refuses.
            return ECHILD;
        }
        crate::task::yield_now();
    }
}

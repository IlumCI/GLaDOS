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
    /// The entry that forked it, so its exit can be reported there.
    parent: usize,
    work: Option<Resume>,
    live: bool,
    pid: u64,
    /// What it exited with, once it has. `wait4` reads this.
    status: Option<i32>,
    /// Sealed into a quarantine field, invisible to everything outside it.
    ///
    /// A quarantined slot answers no pid lookup, no `kill`, no `wait4`: the
    /// process is still on its task until it dies, but as far as the rest of
    /// the machine is concerned it is already gone. `SIGQUARANTINE` sets it on
    /// a whole process family at once and then dooms every task in the field --
    /// which is what makes a self-replicating tree stoppable, since a fork that
    /// lands after the lookups are blinded has nobody to be seen by and is
    /// swept by the same doom.
    quarantined: bool,
}

static SLOTS: Racy<[Slot; MAX_CHILDREN]> = Racy::new(
    [Slot { task: None, stack: 0, guest: 0, parent: 0, work: None, live: false,
            pid: 0, status: None, quarantined: false };
        MAX_CHILDREN],
);

/// The next process id. From 2, because `getpid` answers 1 for the first guest.
static NEXT_PID: AtomicU64 = AtomicU64::new(2);

/// The next id, for a process *or* a thread. One counter, as Linux has one id
/// space: two counters both starting at 2 gave a thread and a forked child the
/// same number, and `wait4` or `kill` naming it would mean either.
pub fn next_id() -> u64 {
    NEXT_PID.fetch_add(1, Ordering::Relaxed)
}

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
///
/// **A child still running is left alone.** `run` dooms every child and waits
/// for them, but the wait is bounded, and a child that outlived it is still
/// on its task reading its own slot. Marking that slot free would hand it to
/// the next `fork` while its tenant is alive, and zeroing the count would make
/// `live` lie about it. It ends at its next safe point and tidies its own
/// slot then, exactly as it would have inside the bound.
pub fn reset() {
    let s = unsafe { SLOTS.get() };
    let mut still = 0;
    for slot in s.iter_mut() {
        if slot.live {
            still += 1;
            continue;
        }
        slot.work = None;
        slot.status = None;
        slot.pid = 0;
        slot.quarantined = false;
    }
    LIVE.store(still, Ordering::Release);
    MAIN_QUARANTINED.store(false, Ordering::Release);
}

/// Whether the session's first guest, pid 1, has been sealed into a quarantine
/// field. It has no `Slot`, so its flag lives here.
static MAIN_QUARANTINED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Seal a process and everyone connected to it by parent or child -- so its
/// parent, its children and its siblings, transitively -- into one quarantine
/// field, then terminate the whole field at once. Answers how many processes
/// were sealed.
///
/// **The field is the connected component, computed rather than assumed.** In
/// today's one-session model the parent chain of every child leads back to
/// pid 1, so the component is usually the whole session; the closure is walked
/// honestly anyway, so the day two unrelated guest trees exist, sealing one
/// leaves the other untouched. That is the whole promise: nothing outside the
/// field learns anything happened inside it.
///
/// Sealing comes before dooming, and the order is the point. Once every member
/// is invisible to pid lookup, a member that forks one more child in the gap
/// before it dies has handed that child to a family nobody can see, and the
/// child is caught by the same sweep -- which is exactly what a self-replicating
/// process is.
pub fn quarantine(target_guest: usize) -> usize {
    let s = unsafe { SLOTS.get() };
    // The connected component over parent/child links, to a fixpoint. Bounded
    // by the slot count plus the main guest, so the walk terminates.
    let mut field: Vec<usize> = alloc::vec![target_guest];
    let mut grew = true;
    while grew {
        grew = false;
        for slot in s.iter() {
            if !slot.live {
                continue;
            }
            if !(field.contains(&slot.guest) || field.contains(&slot.parent)) {
                continue;
            }
            for g in [slot.guest, slot.parent] {
                if !field.contains(&g) {
                    field.push(g);
                    grew = true;
                }
            }
        }
    }

    // Seal first, every member, so the lookups are blinded before anything dies
    // and a late fork has nowhere to escape to. Guest 0 is pid 1, the session
    // itself: sealing it ends the whole session, there being nothing above it
    // to be isolated from.
    let mut sealed = 0;
    if field.contains(&0) {
        MAIN_QUARANTINED.store(true, Ordering::Release);
        sealed += 1;
    }
    for slot in s.iter_mut() {
        if slot.live && field.contains(&slot.guest) {
            slot.quarantined = true;
            sealed += 1;
        }
    }

    // Then doom the whole field at once. Each task ends itself at its next safe
    // point -- a syscall boundary, a wait loop, or the timer finding it at ring
    // 3 -- so a member spinning without syscalls is ended by the timer just as
    // `SIGKILL` reaches one.
    if field.contains(&0) {
        if let Some(t) = syscall::main_task() {
            syscall::doom(t, syscall::SIGNALED | 9);
        }
    }
    for slot in s.iter() {
        if slot.live && field.contains(&slot.guest) {
            if let Some(t) = slot.task {
                syscall::doom(t, syscall::SIGNALED | 9);
            }
        }
    }
    sealed
}

/// Tell every running child to end itself with `code`. The session's end, and
/// nothing else, sweeps them all -- see `syscall::run`.
///
/// A child forked but not yet entered is not in a guest, so it cannot be
/// doomed -- it is cancelled instead: its work withdrawn before its task ever
/// picks it up, its entry freed, and its parent told it was killed.
pub fn doom_all(code: u64) {
    let s = unsafe { SLOTS.get() };
    for slot in s.iter_mut() {
        if !slot.live {
            continue;
        }
        if slot.work.take().is_some() {
            syscall::release_guest(slot.guest);
            slot.status = Some(syscall::wait_status(code));
            slot.live = false;
            LIVE.fetch_sub(1, Ordering::Release);
            continue;
        }
        if let Some(t) = slot.task {
            syscall::doom(t, code);
        }
    }
}

/// The task running the guest `pid` names, for a `SIGKILL` that has to reach a
/// child which may never make another syscall.
pub fn task_of_pid(pid: i64) -> Option<usize> {
    if pid == 1 {
        return (!MAIN_QUARANTINED.load(Ordering::Acquire))
            .then(syscall::main_task)
            .flatten();
    }
    unsafe { &*SLOTS.get() }
        .iter()
        .find(|x| x.live && x.work.is_none() && x.pid == pid as u64 && !x.quarantined)
        .and_then(|x| x.task)
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
    fork_with(f, None, None)
}

const SIGCHLD: u64 = 17;
const CLONE_VFORK: u64 = 0x0000_4000;
const CLONE_CHILD_SETTID: u64 = 0x0100_0000;

/// Whether a `clone` is asking for a process rather than a thread: no
/// `CLONE_THREAD`, an exit signal of `SIGCHLD` or none, and nothing among its
/// flags this kernel cannot honour. `CLONE_VM` is allowed only with
/// `CLONE_VFORK`, which is how `posix_spawn` asks -- see `clone_process`.
pub fn is_process(flags: u64) -> bool {
    use super::thread::{CLONE_CHILD_CLEARTID, CLONE_PARENT_SETTID, CLONE_THREAD, CLONE_VM};
    let signal = flags & 0xFF;
    let known = CLONE_VM | CLONE_VFORK | CLONE_CHILD_SETTID | CLONE_CHILD_CLEARTID | CLONE_PARENT_SETTID | 0xFF;
    flags & CLONE_THREAD == 0
        && (signal == SIGCHLD || signal == 0)
        && flags & !known == 0
        && (flags & CLONE_VM == 0 || flags & CLONE_VFORK != 0)
}

/// `clone` asking for a process, which is how glibc forks.
///
/// **glibc never calls `fork`.** Its `fork()` is `clone(CLONE_CHILD_SETTID |
/// CLONE_CHILD_CLEARTID | SIGCHLD)` with the child's tid written into the
/// child's own thread block, and `posix_spawn` -- which `system()` uses -- is
/// `clone(CLONE_VM | CLONE_VFORK | SIGCHLD)` on a stack of its own. This kernel
/// served `fork` and `vfork` by number, which is musl's way, and answered
/// `ENOSYS` to both of glibc's, so a glibc shell could run nothing.
///
/// `CLONE_VM | CLONE_VFORK` is served by a copy, not a share: the child of a
/// spawn runs on the stack it was given until it calls `execve` or exits,
/// which a copy does identically. What is lost is the child writing into the
/// parent's memory before `exec` -- `posix_spawn` reports a failed `exec` that
/// way, so here a spawn whose `exec` failed reads as started, and its exit
/// status of 127 is what says otherwise. `CLONE_CHILD_CLEARTID` is accepted
/// and not acted on: it clears a word in the child's own memory when the child
/// ends, and nothing outside that child waits on it.
pub fn clone_process(f: &Frame) -> u64 {
    use super::thread::CLONE_PARENT_SETTID;
    let (flags, stack, ptid, ctid) = (f.rdi, f.rsi, f.rdx, f.r10);
    let settid = (flags & CLONE_CHILD_SETTID != 0 && ctid != 0).then_some(ctid);
    let pid = fork_with(f, (stack != 0).then_some(stack), settid);
    if (pid as i64) > 0 && flags & CLONE_PARENT_SETTID != 0 && ptid != 0 && syscall::reachable(ptid, 4, true) {
        unsafe { core::ptr::write_volatile(ptid as *mut u32, pid as u32) };
    }
    pid
}

/// `fork`, running the child on `stack` if given, and writing its pid at
/// `settid` in the child's memory if given.
fn fork_with(f: &Frame, stack: Option<u64>, settid: Option<u64>) -> u64 {
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

    // The pid first, so it can be in the child's memory from its first
    // instruction: `CLONE_CHILD_SETTID` is the child's copy of a word, and the
    // simplest way to have a copy say something is to say it in the original
    // for the length of the copy.
    let pid = next_id();
    let settid = settid.filter(|&a| syscall::reachable(a, 4, true));
    let saved = settid.map(|a| unsafe { core::ptr::read_volatile(a as *const u32) });
    if let Some(a) = settid {
        unsafe { core::ptr::write_volatile(a as *mut u32, pid as u32) };
    }

    let mut owned: Vec<(u64, usize)> = Vec::new();
    let mut failed = false;
    for (at, len) in regions.iter().chain(maps.iter()) {
        if !dup_range(&mut tables, *at, *len, &mut owned) {
            failed = true;
            break;
        }
    }
    if let (Some(a), Some(v)) = (settid, saved) {
        unsafe { core::ptr::write_volatile(a as *mut u32, v) };
    }
    if failed {
        for (p, n) in owned {
            syscall::free_pages(p, n);
        }
        return ENOMEM;
    }

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
        // A spawn's child runs on the stack it was handed; a fork's on its copy
        // of the parent's.
        rsp: stack.unwrap_or_else(syscall::guest_rsp),
        rflags: f.rflags,
        rax: 0,
        rbx: f.rbx,
        rbp: f.rbp,
        r12: f.r12,
        r13: f.r13,
        r14: f.r14,
        r15: f.r15,
        // The parent's TLS, which a child's copy of memory needs pointed at.
        fs: syscall::guest_fs(),
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
        s[idx].parent = parent;
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
        let code = syscall::as_guest(|| unsafe { syscall::enter_resumed(&work) });
        crate::task::ring3_active(false, 0);
        crate::task::set_root(me, 0);
        syscall::set_current_guest(prev);

        // Said, because nothing else will say it: the parent learns only a
        // signal number from `wait4`, and a person watching learns nothing.
        // Printed here, off the fault handler's stack -- painting from inside
        // an interrupt gate is the console bug `cpu::idt` records.
        let pid = unsafe { (*SLOTS.get())[i].pid };
        if code & syscall::FAULTED != 0 {
            match syscall::take_task_fault(me) {
                Some(f) => crate::kprintln!(
                    "  [linux] child {} killed by fault {:#04x} at rip {:#x} (cr2 {:#x}), its parent carries on",
                    pid, f.regs.vector, f.regs.rip, f.cr2
                ),
                None => crate::kprintln!("  [linux] child {} killed by a fault, its parent carries on", pid),
            }
        }

        // **The child's memory goes back now, not never.** Its copy of the
        // parent and its tables lived in a guest entry nothing ever freed, so
        // every fork leaked a whole process and a shell running commands in a
        // loop would have run the heap dry. The root is off by now, which is
        // the order `Space` needs: tables are freed only once nothing walks
        // them.
        syscall::release_guest(guest);

        let parent = {
            let s = unsafe { SLOTS.get() };
            s[i].status = Some(syscall::wait_status(code));
            s[i].live = false;
            s[i].parent
        };
        // **A child finishing is the first signal this machine can honestly
        // raise.** Ignored by default, which is why a parent that never
        // installs a handler is not killed by its own children finishing.
        super::signal::raise_at(parent, super::signal::SIGCHLD);
        LIVE.fetch_sub(1, Ordering::Release);
        if flags & (1 << 9) != 0 {
            crate::cpu::enable_interrupts();
        }
    }
}

/// Which guest entry a pid names, for `kill`.
///
/// Pid 1 is the guest the shell started, which `getpid` has always answered
/// for and which has no slot here -- it was never forked.
/// The pid of the guest entry `guest`: a forked child's own, or 1 for the
/// guest `linux run` started.
pub fn pid_of_guest(guest: usize) -> u64 {
    unsafe { &*SLOTS.get() }
        .iter()
        .find(|s| s.live && s.guest == guest)
        .map(|s| s.pid)
        .unwrap_or(1)
}

/// `getpid`. It answered 1 to everybody, so a forked child named itself after
/// its parent -- and glibc checks exactly that after `fork`.
pub fn current_pid() -> u64 {
    pid_of_guest(syscall::current_guest())
}

/// `getppid`: a child's parent's pid, and 0 for the first guest, whose parent
/// is the kernel.
pub fn parent_pid() -> u64 {
    let me = syscall::current_guest();
    match unsafe { &*SLOTS.get() }.iter().find(|s| s.live && s.guest == me) {
        Some(s) => pid_of_guest(s.parent),
        None => 0,
    }
}

pub fn guest_of_pid(pid: i64) -> Option<usize> {
    if pid == 1 {
        // Sealed pid 1 is gone to the rest of the machine, as every sealed
        // process is.
        return (!MAIN_QUARANTINED.load(Ordering::Acquire)).then_some(0);
    }
    if pid <= 0 {
        return None;
    }
    let s = unsafe { SLOTS.get() };
    s.iter()
        .find(|x| x.pid == pid as u64 && !x.quarantined)
        .map(|x| x.guest)
}

/// `wait4`, in the one shape a shell needs: wait for any child, or for one.
///
/// No `WNOHANG` and no resource usage. Both are refusals this kernel can make
/// honestly -- there is no rusage to report and nothing here samples one --
/// and a shell that asked for either would rather be told than handed zeros.
/// `WNOHANG`: answer now, even when the answer is "nothing yet".
///
/// The one option a shell actually needs. `WUNTRACED` and `WCONTINUED` report
/// stops and continues, and nothing here can stop a guest short of ending it,
/// so a kernel accepting them would be promising events it cannot produce.
pub const WNOHANG: u64 = 1;

pub fn wait(pid: i64, status: u64, options: u64) -> u64 {
    const ECHILD: u64 = (-10i64) as u64;
    const EINVAL: u64 = (-22i64) as u64;

    // Refused rather than ignored. A caller that asked to be told about a
    // stopped child and is simply never told has been given a wait that looks
    // like it works and silently never fires -- the shape this tree keeps
    // recording as the worst kind of wrong.
    if options & !WNOHANG != 0 {
        return EINVAL;
    }

    loop {
        let found = {
            let s = unsafe { SLOTS.get() };
            s.iter()
                .position(|x| x.pid != 0 && !x.live && x.status.is_some()
                    && !x.quarantined && (pid <= 0 || x.pid == pid as u64))
        };
        if let Some(i) = found {
            let s = unsafe { SLOTS.get() };
            let code = s[i].status.take().unwrap_or(0);
            let got = s[i].pid;
            s[i].pid = 0;
            if status != 0 && syscall::owns(status, 4) {
                // Already a wait status, encoded by `syscall::wait_status`
                // when the child ended: an exit in bits 8..15, a signal in the
                // low seven. Encoded there rather than here because only the
                // end knows which of the two it was.
                unsafe { core::ptr::write(status as *mut i32, code) };
            }
            return got;
        }
        let any = {
            let s = unsafe { SLOTS.get() };
            s.iter().any(|x| x.pid != 0 && !x.quarantined && (pid <= 0 || x.pid == pid as u64))
        };
        if !any {
            return ECHILD;
        }
        // **Zero, and it has to be zero rather than `ECHILD`.** A child exists
        // and has not finished, which is a different fact from having no
        // children at all: a shell polling with `WNOHANG` treats `ECHILD` as
        // "it is gone, stop asking" and would drop the child on the floor.
        if options & WNOHANG != 0 {
            return 0;
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

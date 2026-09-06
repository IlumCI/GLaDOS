//! Signals, which stopped being impossible when `fork` landed.
//!
//! `syscall.rs` accepted `rt_sigaction` and never delivered anything, under a
//! reason that was true when written: "there is no other process to send one,
//! no terminal to generate one, and a fault ends the guest rather than being
//! offered to it". The first clause is what `fork` falsified. There are other
//! processes now, a child that exits is an event its parent is entitled to
//! hear about, and `kill` has somebody to talk to.
//!
//! ### Where delivery happens, and why it is the only sane place
//!
//! On the way **out of a syscall**, in `glados_syscall_dispatch`. That is the
//! one moment the guest's complete register state is already sitting in a
//! `Frame` the kernel owns, its stack pointer is parked in `GLADOS_GUEST_RSP`,
//! and the return path is about to `sysretq` somewhere -- so redirecting it
//! costs three stores and no new assembly. Delivering from the timer instead
//! would mean building a frame around an interrupted ring-3 context, which is
//! a second entry path to keep in step with the first.
//!
//! The cost is stated rather than hidden: **a guest that makes no syscalls
//! receives no signals.** A program spinning in a loop cannot be interrupted,
//! which on Linux it could. Nothing here needs that yet and the deadline still
//! ends a runaway, so this buys the whole feature for the price of one branch
//! on a path that already exists.
//!
//! ### What is deliberately not here
//!
//! The saved context lives in the kernel, not on the guest's stack. Linux
//! pushes a `ucontext` and a `siginfo` and a handler may read them; Wine reads
//! them, which is how it emulates memory it has taken away. So this serves an
//! ordinary handler and does not yet serve Wine, and **one signal at a time**
//! -- a second delivered before `rt_sigreturn` would overwrite the first one's
//! saved state, so it is refused rather than allowed to corrupt.

use core::sync::atomic::{AtomicU64, Ordering};

use super::syscall::{self, Frame};

/// The highest signal this answers for. Linux has 64; the low 32 are the ones
/// with names, and nothing here generates a real-time signal.
pub const NSIG: usize = 32;

pub const SIGKILL: u32 = 9;
pub const SIGUSR1: u32 = 10;
pub const SIGSEGV: u32 = 11;
pub const SIGUSR2: u32 = 12;
pub const SIGTERM: u32 = 15;
pub const SIGCHLD: u32 = 17;
pub const SIGSTOP: u32 = 19;

/// `SIG_DFL` is 0 and `SIG_IGN` is 1, which are addresses no handler can have.
const SIG_DFL: u64 = 0;
const SIG_IGN: u64 = 1;

/// What a guest asked for, per signal.
#[derive(Clone, Copy)]
pub struct Action {
    pub handler: u64,
    pub flags: u64,
    /// Where the handler returns to. glibc supplies one that does
    /// `mov rax, 15; syscall`, which is `rt_sigreturn`; a program with none
    /// cannot have its handler return and this refuses to deliver to it,
    /// because the alternative is a `ret` onto whatever the stack held.
    pub restorer: u64,
    pub mask: u64,
}

impl Default for Action {
    fn default() -> Self {
        Action { handler: SIG_DFL, flags: 0, restorer: 0, mask: 0 }
    }
}

/// Everything one guest's signal state is.
#[derive(Clone, Copy)]
pub struct State {
    pub actions: [Action; NSIG],
    pub pending: u64,
    pub blocked: u64,
    /// The frame a handler interrupted, restored by `rt_sigreturn`.
    ///
    /// In the kernel rather than on the guest's stack, which is the deviation
    /// this module's header names. One deep, so a second signal arriving
    /// before the first returns is left pending rather than delivered.
    pub saved: Option<(Frame, u64)>,
}

impl Default for State {
    fn default() -> Self {
        State {
            actions: [Action::default(); NSIG],
            pending: 0,
            blocked: 0,
            saved: None,
        }
    }
}

/// Signals a guest may not catch, block or ignore, as Linux has it.
///
/// Not a courtesy: `SIGKILL` exists so there is one thing a process cannot
/// argue with, and a kernel that let a handler take it has removed the only
/// guarantee the call makes.
fn uncatchable(sig: u32) -> bool {
    sig == SIGKILL || sig == SIGSTOP
}

fn bit(sig: u32) -> u64 {
    if sig == 0 || sig as usize > NSIG {
        0
    } else {
        1u64 << (sig - 1)
    }
}

/// How many signals have been delivered, for a claim that wants a number
/// rather than a story.
static DELIVERED: AtomicU64 = AtomicU64::new(0);

pub fn delivered() -> u64 {
    DELIVERED.load(Ordering::Relaxed)
}

/// `rt_sigaction`. Records a handler and answers the one it replaced.
pub fn sigaction(sig: u64, act: u64, old: u64, setsize: u64) -> u64 {
    const EINVAL: u64 = (-22i64) as u64;
    const EFAULT: u64 = (-14i64) as u64;
    if sig == 0 || sig as usize > NSIG {
        return EINVAL;
    }
    // Linux checks this and a libc passes 8. Refused rather than ignored,
    // because a caller using a different mask width would be handing over a
    // structure this reads the wrong number of bytes from.
    if setsize != 8 {
        return EINVAL;
    }
    if uncatchable(sig as u32) && act != 0 {
        return EINVAL;
    }
    let Some(sp) = (unsafe { syscall::guest_slot() }).as_mut() else { return EFAULT };
    let i = sig as usize - 1;
    let was = sp.signals.actions[i];
    if old != 0 {
        if !syscall::reachable(old, 32, true) {
            return EFAULT;
        }
        unsafe {
            let p = old as *mut u64;
            p.write(was.handler);
            p.add(1).write(was.flags);
            p.add(2).write(was.restorer);
            p.add(3).write(was.mask);
        }
    }
    if act != 0 {
        if !syscall::reachable(act, 32, false) {
            return EFAULT;
        }
        // The layout is `struct sigaction`: handler, flags, restorer, mask.
        // Four words, in that order, which is the kernel's own ABI rather
        // than glibc's userspace struct.
        let p = act as *const u64;
        let a = unsafe {
            Action {
                handler: p.read(),
                flags: p.add(1).read(),
                restorer: p.add(2).read(),
                mask: p.add(3).read(),
            }
        };
        sp.signals.actions[i] = a;
    }
    0
}

/// `rt_sigprocmask`. `how` is 0 block, 1 unblock, 2 set.
pub fn sigprocmask(how: u64, set: u64, old: u64, setsize: u64) -> u64 {
    const EINVAL: u64 = (-22i64) as u64;
    const EFAULT: u64 = (-14i64) as u64;
    if setsize != 8 {
        return EINVAL;
    }
    let Some(sp) = (unsafe { syscall::guest_slot() }).as_mut() else { return EFAULT };
    if old != 0 {
        if !syscall::reachable(old, 8, true) {
            return EFAULT;
        }
        unsafe { core::ptr::write(old as *mut u64, sp.signals.blocked) };
    }
    if set != 0 {
        if !syscall::reachable(set, 8, false) {
            return EFAULT;
        }
        let v = unsafe { core::ptr::read(set as *const u64) };
        // `SIGKILL` and `SIGSTOP` cannot be blocked, and Linux silently drops
        // them from the mask rather than refusing the call.
        let v = v & !(bit(SIGKILL) | bit(SIGSTOP));
        sp.signals.blocked = match how {
            0 => sp.signals.blocked | v,
            1 => sp.signals.blocked & !v,
            2 => v,
            _ => return EINVAL,
        };
    }
    0
}

/// Mark a signal pending on one guest. Answers false when there is no such
/// guest to mark.
pub fn raise_at(guest: usize, sig: u32) -> bool {
    let Some(sp) = (unsafe { syscall::slot_at(guest) }).as_mut() else { return false };
    if bit(sig) == 0 {
        return false;
    }
    sp.signals.pending |= bit(sig);
    true
}

/// `kill`. Only whole processes, and only ones this machine knows about.
pub fn kill(pid: i64, sig: u64) -> u64 {
    const ESRCH: u64 = (-3i64) as u64;
    const EINVAL: u64 = (-22i64) as u64;
    if sig as usize > NSIG {
        return EINVAL;
    }
    // Signal zero is the existence check every shell uses and delivers
    // nothing, which is worth answering correctly rather than treating as a
    // degenerate send.
    let Some(guest) = super::fork::guest_of_pid(pid) else { return ESRCH };
    if sig == 0 {
        return 0;
    }
    if !raise_at(guest, sig as u32) {
        return EINVAL;
    }
    0
}

/// Pick the next signal to deliver, or `None`.
fn next(sp: &super::syscall::Space) -> Option<u32> {
    let ready = sp.signals.pending & !sp.signals.blocked;
    if ready == 0 {
        return None;
    }
    for s in 1..=NSIG as u32 {
        if ready & bit(s) != 0 {
            return Some(s);
        }
    }
    None
}

/// Deliver a pending signal by redirecting the return from a syscall.
///
/// Called with the frame the stub is about to restore. Answers true when the
/// frame was rewritten, which the caller uses only for its trace.
pub fn deliver(f: &mut Frame) -> bool {
    let Some(sp) = (unsafe { syscall::guest_slot() }).as_mut() else { return false };
    // One at a time. A second signal arriving inside a handler would overwrite
    // the state `rt_sigreturn` restores, so it waits rather than corrupting.
    if sp.signals.saved.is_some() {
        return false;
    }
    let Some(sig) = next(sp) else { return false };
    let a = sp.signals.actions[sig as usize - 1];

    // Default and ignore, decided here rather than at `kill`, because a guest
    // may install a handler after the signal is already pending.
    if a.handler == SIG_IGN {
        sp.signals.pending &= !bit(sig);
        return false;
    }
    if a.handler == SIG_DFL {
        sp.signals.pending &= !bit(sig);
        // **What the default is depends on the signal, and getting it wrong is
        // the difference between a program that stops and one that does not.**
        // `SIGCHLD` is ignored by default, which is why a shell that never
        // installs a handler is not killed by its own children finishing.
        // Everything else here terminates.
        if sig == SIGCHLD {
            return false;
        }
        unsafe { syscall::kill_guest_now(128 + sig as u64) };
    }
    // A handler with nowhere to return to is refused rather than entered: the
    // `ret` at the end of it would take whatever the stack happened to hold.
    if a.restorer == 0 {
        sp.signals.pending &= !bit(sig);
        return false;
    }

    let rsp = syscall::guest_rsp();
    // Sixteen aligned before the push, so the handler sees the alignment the
    // ABI promises at a call boundary -- one word of return address below a
    // 16-aligned stack, exactly as a `call` leaves it.
    let sp_new = ((rsp - 128) & !0xF) - 8;
    if !syscall::reachable(sp_new, 8, true) {
        sp.signals.pending &= !bit(sig);
        return false;
    }
    unsafe { core::ptr::write(sp_new as *mut u64, a.restorer) };

    sp.signals.pending &= !bit(sig);
    sp.signals.saved = Some((
        Frame {
            rax: f.rax, rdi: f.rdi, rsi: f.rsi, rdx: f.rdx, r10: f.r10,
            r8: f.r8, r9: f.r9, rip: f.rip, rflags: f.rflags, rbx: f.rbx,
            rbp: f.rbp, r12: f.r12, r13: f.r13, r14: f.r14, r15: f.r15,
        },
        rsp,
    ));

    f.rip = a.handler;
    f.rdi = sig as u64;
    f.rsi = 0;
    f.rdx = 0;
    unsafe { syscall::set_guest_rsp(sp_new) };
    DELIVERED.fetch_add(1, Ordering::Relaxed);
    true
}

/// `rt_sigreturn`. Puts back what the handler interrupted.
///
/// Never returns a value to the guest -- whatever it answers is overwritten by
/// the restored `rax`, which is the point: the interrupted call's result has
/// to survive being interrupted.
pub fn sigreturn(f: &mut Frame) -> u64 {
    let Some(sp) = (unsafe { syscall::guest_slot() }).as_mut() else { return 0 };
    let Some((saved, rsp)) = sp.signals.saved.take() else {
        // A guest that called this without being in a handler. Linux kills it;
        // this refuses, because ending a guest for a bad syscall is a heavier
        // answer than the call deserves and the trace already records it.
        return (-22i64) as u64;
    };
    *f = saved;
    unsafe { syscall::set_guest_rsp(rsp) };
    f.rax
}

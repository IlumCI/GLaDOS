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
//! ### The frame goes on the guest's stack, in Linux's own layout
//!
//! It was in the kernel first, one deep, which served an ordinary handler and
//! served Wine not at all: Wine reads `uc_mcontext` to find out where a fault
//! happened and what the registers were, because that is how it emulates
//! memory it has taken away. A handler cannot read a context it cannot reach.
//!
//! So `rt_sigframe` is built where Linux builds it and with the offsets Linux
//! uses -- `sigcontext` is 256 bytes with a fixed register order, `ucontext`
//! is 304, and the whole frame is 440 with `siginfo` on the end. Those numbers
//! are an ABI rather than a choice: a handler compiled against the real header
//! reads `uc_mcontext.rip` at a fixed offset, and a frame one field short is
//! not a smaller frame, it is a different structure.
//!
//! Two things fall out of moving it. **Signals nest**, because each delivery
//! builds its own frame, where the kernel-side version had to refuse a second
//! one. And `rt_sigreturn` takes its state from the guest's stack, which means
//! a handler that edits `uc_mcontext` before returning changes where the
//! program resumes -- which is not a curiosity, it is precisely the mechanism
//! Wine's fault emulation depends on.

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

/// The signal frame, in the layout Linux uses on x86-64.
///
/// **These are an ABI and not a choice.** A handler built against the real
/// `<ucontext.h>` reads `uc_mcontext.rip` at a fixed distance from the pointer
/// it is handed, so a frame that is a field short is not a smaller frame, it
/// is a different structure and every field after the gap is misread.
///
///     rt_sigframe {
///         pretcode  +0     where the handler returns to
///         uc        +8     ucontext, 304 bytes
///         info      +312   siginfo, 128 bytes
///     }
///
///     ucontext {
///         uc_flags  +0
///         uc_link   +8
///         uc_stack  +16    stack_t, 24 bytes
///         uc_mcontext +40  sigcontext, 256 bytes
///         uc_sigmask  +296
///     }
mod frame {
    pub const UC: u64 = 8;
    pub const MCONTEXT: u64 = UC + 40;
    pub const SIGMASK: u64 = UC + 296;
    pub const INFO: u64 = 312;
    pub const SIZE: u64 = 440;

    // Offsets inside `sigcontext`, which is a fixed register order rather than
    // any order that would be natural. Written out because getting one wrong
    // reads a handler's `rip` out of its `rbx`.
    pub const R8: u64 = 0;
    pub const R9: u64 = 8;
    pub const R10: u64 = 16;
    pub const R11: u64 = 24;
    pub const R12: u64 = 32;
    pub const R13: u64 = 40;
    pub const R14: u64 = 48;
    pub const R15: u64 = 56;
    pub const RDI: u64 = 64;
    pub const RSI: u64 = 72;
    pub const RBP: u64 = 80;
    pub const RBX: u64 = 88;
    pub const RDX: u64 = 96;
    pub const RAX: u64 = 104;
    pub const RCX: u64 = 112;
    pub const RSP: u64 = 120;
    pub const RIP: u64 = 128;
    pub const EFLAGS: u64 = 136;
    pub const CSGSFSSS: u64 = 144;
    pub const ERR: u64 = 152;
    pub const TRAPNO: u64 = 160;
    pub const OLDMASK: u64 = 168;
    pub const CR2: u64 = 176;
    pub const FPSTATE: u64 = 184;
}

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
    /// How deep in handlers this guest is, for the trace and for a claim.
    ///
    /// Not a limit. Each delivery builds its own frame on the guest's stack,
    /// so nesting costs stack rather than kernel state -- which is the whole
    /// reason the frame moved there.
    pub depth: u32,
}

impl Default for State {
    fn default() -> Self {
        State {
            actions: [Action::default(); NSIG],
            pending: 0,
            blocked: 0,
            depth: 0,
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
    // 128 bytes of red zone left alone, which the ABI says a leaf function may
    // be using below its own stack pointer. Linux skips it for the same reason
    // and a kernel that did not would corrupt whatever was interrupted.
    //
    // The frame lands at 8 mod 16, so the handler starts with its stack in the
    // state a `call` leaves -- return address pushed, everything above it
    // 16-aligned. A handler doing anything with SSE reads that alignment.
    let base = (rsp - 128 - frame::SIZE) & !0xF;
    let at = base - 8;
    if !syscall::reachable(at, frame::SIZE as usize + 8, true) {
        sp.signals.pending &= !bit(sig);
        return false;
    }

    let m = at + frame::MCONTEXT;
    unsafe {
        let w = |off: u64, v: u64| core::ptr::write((at + off) as *mut u64, v);
        let r = |off: u64, v: u64| core::ptr::write((m + off) as *mut u64, v);
        // Zero the whole thing first: `siginfo`'s union, `uc_stack`, and the
        // reserved tail of `sigcontext` are all read by somebody and none of
        // them is written below.
        core::ptr::write_bytes(at as *mut u8, 0, frame::SIZE as usize + 8);
        w(0, a.restorer);
        r(frame::R8, f.r8);
        r(frame::R9, f.r9);
        r(frame::R10, f.r10);
        r(frame::R11, f.rflags);
        r(frame::R12, f.r12);
        r(frame::R13, f.r13);
        r(frame::R14, f.r14);
        r(frame::R15, f.r15);
        r(frame::RDI, f.rdi);
        r(frame::RSI, f.rsi);
        r(frame::RBP, f.rbp);
        r(frame::RBX, f.rbx);
        r(frame::RDX, f.rdx);
        r(frame::RAX, f.rax);
        r(frame::RCX, f.rip);
        r(frame::RSP, rsp);
        r(frame::RIP, f.rip);
        r(frame::EFLAGS, f.rflags);
        // `cs` 0x33 and `ss` 0x2b are what ring 3 runs on here, packed as four
        // `u16` in one word the way the structure has them.
        r(frame::CSGSFSSS, 0x33 | (0x2b << 48));
        r(frame::ERR, 0);
        r(frame::TRAPNO, 0);
        r(frame::OLDMASK, sp.signals.blocked);
        r(frame::CR2, 0);
        // No extended state saved, and said so rather than pointed at
        // rubbish: a handler that followed a stale pointer would `xrstor`
        // whatever was there.
        r(frame::FPSTATE, 0);
        w(frame::SIGMASK, sp.signals.blocked);
        // `siginfo`: signo, errno, code. `SI_USER` is 0, which is what a
        // signal raised by `kill` carries.
        core::ptr::write((at + frame::INFO) as *mut u32, sig);
        core::ptr::write((at + frame::INFO + 4) as *mut u32, 0);
        core::ptr::write((at + frame::INFO + 8) as *mut u32, 0);
    }

    sp.signals.pending &= !bit(sig);
    // Blocked for the length of the handler, as Linux does, plus whatever the
    // action asked for. `rt_sigreturn` puts the old mask back out of the
    // frame, which is why `uc_sigmask` had to be written.
    sp.signals.blocked |= a.mask | bit(sig);
    sp.signals.depth += 1;

    f.rip = a.handler;
    f.rdi = sig as u64;
    f.rsi = at + frame::INFO;
    f.rdx = at + frame::UC;
    unsafe { syscall::set_guest_rsp(at) };
    DELIVERED.fetch_add(1, Ordering::Relaxed);
    true
}

/// `rt_sigreturn`. Puts back what the handler interrupted.
///
/// Never returns a value to the guest -- whatever it answers is overwritten by
/// the restored `rax`, which is the point: the interrupted call's result has
/// to survive being interrupted.
pub fn sigreturn(f: &mut Frame) -> u64 {
    const EFAULT: u64 = (-14i64) as u64;
    let Some(sp) = (unsafe { syscall::guest_slot() }).as_mut() else { return 0 };
    // The handler's `ret` popped `pretcode`, so the guest's stack pointer is
    // now eight past the frame. That is how Linux finds it too, and it is the
    // reason `pretcode` exists at offset zero rather than being handed over in
    // a register.
    let at = syscall::guest_rsp() - 8;
    if !syscall::reachable(at, frame::SIZE as usize + 8, false) {
        return EFAULT;
    }
    let m = at + frame::MCONTEXT;
    let r = |off: u64| unsafe { core::ptr::read((m + off) as *const u64) };

    // **Read out of the guest's frame rather than out of kernel state, which
    // is what makes a handler able to change where it resumes.** Wine's fault
    // emulation edits `uc_mcontext` and returns; so does any program using
    // `setcontext`. A kernel that restored its own copy would silently ignore
    // the edit.
    f.r8 = r(frame::R8);
    f.r9 = r(frame::R9);
    f.r10 = r(frame::R10);
    f.r12 = r(frame::R12);
    f.r13 = r(frame::R13);
    f.r14 = r(frame::R14);
    f.r15 = r(frame::R15);
    f.rdi = r(frame::RDI);
    f.rsi = r(frame::RSI);
    f.rbp = r(frame::RBP);
    f.rbx = r(frame::RBX);
    f.rdx = r(frame::RDX);
    f.rax = r(frame::RAX);
    f.rip = r(frame::RIP);
    // Only the bits a guest is allowed to set. `sysretq` loads flags from
    // `r11`, so a frame claiming `IF` clear or `IOPL` 3 would be a guest
    // choosing its own interrupt state -- which is a way out of ring 3 that
    // does not involve a syscall.
    const SAFE: u64 = 0x0000_08D5 | (1 << 9);
    f.rflags = (r(frame::EFLAGS) & SAFE) | 0x202;
    unsafe { syscall::set_guest_rsp(r(frame::RSP)) };
    sp.signals.blocked = unsafe { core::ptr::read((at + frame::SIGMASK) as *const u64) };
    sp.signals.depth = sp.signals.depth.saturating_sub(1);
    f.rax
}

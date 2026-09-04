//! Surviving a fault in a program the machine wrote for itself.
//!
//! Every vector but `#BP` used to be fatal, which is defensible for a kernel
//! bug and indefensible for a bad program: this system writes its own skills,
//! composes its own routing cores, and compiles its own code, and a stray
//! index in one of those stopped the machine. The thing most likely to fault
//! here is the thing the machine produced five minutes ago.
//!
//! **What is recoverable is bounded and stated.** A fault is caught only while
//! a task is inside `guard`, which the interpreter and the code generator wrap
//! their execution in. A fault anywhere else stays fatal, because there is no
//! isolation in this kernel and a fault in the page tables or the allocator has
//! already corrupted whatever it was going to corrupt. Recovering from that
//! would produce a machine that keeps running and cannot be trusted, which is
//! worse than one that stops.
//!
//! **The handler does not return.** It restores a stack pointer and jumps,
//! rather than editing the interrupt frame and executing `iretq`. Editing the
//! frame means knowing whether the `x86-interrupt` ABI handed this code the
//! real frame or a copy of it, and being wrong there returns to an address
//! nobody chose. Jumping needs no such answer: the landing pad is on the same
//! task stack at a point that was live when `guard` was called, so the frame
//! and everything above it is simply abandoned. That also works when the fault
//! arrived on an interrupt stack, which the page-fault vector does.
//!
//! Interrupts are re-enabled at the pad, because the gate cleared them on the
//! way in and the pad is ordinary code.

use core::sync::atomic::{AtomicU64, Ordering};

/// Where to land, per task.
///
/// Per task rather than per core: a task can be preempted mid-program and
/// resumed on another core, so a core-indexed table would send the fault to
/// whatever the previous occupant of that core was doing.
#[repr(C)]
#[derive(Clone, Copy)]
struct Pad {
    /// **Every callee-saved register, not just the stack.**
    ///
    /// `guard`'s own epilogue restores whatever `guard` spilled, so for a long
    /// time saving `rsp` and `rbp` looked sufficient. It is not. A register the
    /// *caller* is using and `guard` never touched is one `guard` had no reason
    /// to spill, so nothing puts it back after a fault -- and the fault path is
    /// a large amount of kernel code that will happily use it.
    ///
    /// It surfaced as `diag recover` reporting FAILED with all five of its
    /// claims printing `ok`: the accumulator the claims were folded into lived
    /// in a callee-saved register, the caught fault clobbered it, and the
    /// verdict was garbage while every line of evidence said pass.
    ///
    /// The order here is the order the landing code reads them in, so changing
    /// one means changing both.
    rbx: u64,
    r12: u64,
    r13: u64,
    r14: u64,
    r15: u64,
    rbp: u64,
    rip: u64,
    rsp: u64,
    /// Set while this task is inside `guard`. Nothing is recovered otherwise.
    armed: u64,
}

const EMPTY: Pad =
    Pad { rbx: 0, r12: 0, r13: 0, r14: 0, r15: 0, rbp: 0, rip: 0, rsp: 0, armed: 0 };
const SLOTS: usize = crate::task::MAX_TASKS;

static PADS: crate::sync::Racy<[Pad; SLOTS]> = crate::sync::Racy::new([EMPTY; SLOTS]);

/// Why the last recovered fault happened, for the message a program gets.
static LAST: AtomicU64 = AtomicU64::new(0);
static COUNT: AtomicU64 = AtomicU64::new(0);

/// How many faults have been caught rather than fatal.
pub fn caught() -> u64 {
    COUNT.load(Ordering::Relaxed)
}

fn slot() -> Option<usize> {
    let t = crate::cpu::percpu::billed()?;
    if t < SLOTS {
        Some(t)
    } else {
        None
    }
}

/// Whether a fault on this task should be recovered, and where to land.
///
/// Called from the fault handler. Reads only, and clears the arm so a fault
/// while unwinding is fatal rather than an endless loop through the same pad.
/// The landing block for this task, if it is inside a guard.
///
/// Answers a pointer rather than the values, because the landing code restores
/// eight registers and passing eight through `asm!` operands would need eight
/// registers it is about to overwrite. Reading them from memory needs one.
///
/// The block lives in `PADS`, which is static, so it stays readable after the
/// landing code has moved `rsp` off the interrupt stack.
pub fn take(vector: u8) -> Option<*const u64> {
    // Only the vectors a program can plausibly cause. A machine check or a
    // double fault says the machine is wrong rather than the program.
    if !matches!(vector, 0 | 5 | 6 | 13 | 14 | 17 | 19) {
        return None;
    }
    let i = slot()?;
    let pads = unsafe { PADS.get() };
    if pads[i].armed == 0 {
        return None;
    }
    pads[i].armed = 0;
    LAST.store(vector as u64, Ordering::Relaxed);
    COUNT.fetch_add(1, Ordering::Relaxed);
    Some(&pads[i].rbx as *const u64)
}

/// What the last recovered fault was.
pub fn describe() -> &'static str {
    match LAST.load(Ordering::Relaxed) {
        0 => "divide error",
        5 => "bound range exceeded",
        6 => "invalid opcode",
        13 => "general protection fault",
        14 => "page fault",
        17 => "alignment check",
        19 => "SIMD floating point",
        _ => "fault",
    }
}

/// Run `f`, catching a fault inside it.
///
/// Answers `Err` with a description when one was caught. The closure's own
/// return value is lost in that case, which is the point: it did not finish.
///
/// # Safety
/// Recovering abandons everything the closure was in the middle of. Anything
/// it half-wrote stays half-written, and any lock it held stays held. So the
/// closure must not hold a lock, and callers here do not: the interpreter
/// takes the namespace and the console per operation rather than across a
/// program.
pub fn guard<F: FnOnce()>(f: F) -> Result<(), &'static str> {
    let Some(i) = slot() else {
        // No per-core storage yet, so nothing can be attributed and nothing
        // can be recovered. Run it plainly rather than pretending.
        f();
        return Ok(());
    };

    let (rsp, rbp, rbx, r12, r13, r14, r15): (u64, u64, u64, u64, u64, u64, u64);
    unsafe {
        core::arch::asm!("mov {}, rsp", out(reg) rsp, options(nomem, nostack, preserves_flags));
        core::arch::asm!("mov {}, rbp", out(reg) rbp, options(nomem, nostack, preserves_flags));
        core::arch::asm!("mov {}, rbx", out(reg) rbx, options(nomem, nostack, preserves_flags));
        core::arch::asm!("mov {}, r12", out(reg) r12, options(nomem, nostack, preserves_flags));
        core::arch::asm!("mov {}, r13", out(reg) r13, options(nomem, nostack, preserves_flags));
        core::arch::asm!("mov {}, r14", out(reg) r14, options(nomem, nostack, preserves_flags));
        core::arch::asm!("mov {}, r15", out(reg) r15, options(nomem, nostack, preserves_flags));
    }

    // The pad's address is taken with `lea` against a local label, so it is
    // the address of the landing code below rather than of this function.
    let land: u64;
    unsafe {
        core::arch::asm!(
            "lea {}, [rip + 3f]",
            out(reg) land,
            options(nomem, nostack, preserves_flags),
        );
    }

    {
        let pads = unsafe { PADS.get() };
        pads[i] = Pad { rbx, r12, r13, r14, r15, rbp, rip: land, rsp, armed: 1 };
    }

    f();

    {
        let pads = unsafe { PADS.get() };
        pads[i].armed = 0;
    }
    // Jumped over on the normal path. A recovered fault lands on `3:` with
    // rsp and rbp restored, and falls into the same tail.
    let faulted: u64;
    unsafe {
        core::arch::asm!(
            "xor {out}, {out}",
            "jmp 4f",
            "3:",
            "sti",
            "mov {out}, 1",
            "4:",
            out = out(reg) faulted,
            options(nostack),
        );
    }
    if faulted != 0 {
        Err(describe())
    } else {
        Ok(())
    }
}

pub fn selftest() -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    let before = caught();
    let mut ran = false;
    let r = guard(|| {
        ran = true;
    });
    claim(&mut ok, r.is_ok() && ran, "a closure that does not fault runs and reports nothing");
    claim(&mut ok, caught() == before, "and nothing was counted as caught");

    // The one that matters. A read of the unmapped page zero inside a guard
    // has to come back as an error rather than stopping the machine.
    let r = guard(|| unsafe {
        let p = 0x0 as *const u64;
        core::ptr::read_volatile(p);
    });
    claim(&mut ok, r.is_err(), "a fault inside a guard is caught");
    claim(&mut ok, caught() == before + 1, "and counted");

    // And the machine still works afterwards, which is the whole claim.
    let mut after = 0u64;
    let r = guard(|| {
        after = 42;
    });
    claim(&mut ok, r.is_ok() && after == 42, "the machine keeps running after catching one");
    ok
}

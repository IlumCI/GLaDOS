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
    /// **Every general-purpose register, and the reason is that `guard` gets
    /// inlined.**
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
    /// The list written then was **System V's** -- rbx, rbp, r12-r15 -- and on
    /// `x86_64-unknown-uefi` the ABI is Microsoft x64, where `rsi` and `rdi`
    /// are non-volatile as well. That was the second bug and it was still the
    /// wrong question, because a longjmp back into an *inlined* `guard`
    /// crosses no ABI boundary at all: the compiler is entitled to keep a
    /// caller's live value in `rax` or `r9` across a call that no longer
    /// exists, and it does. So the answer is not a shorter list chosen from a
    /// calling convention, it is all fifteen plus the stack.
    ///
    /// Both bugs were found by `mem::paging::checks`, which is the only thing
    /// in the tree that faults on purpose with real work live around it, and
    /// both presented as a wild pointer inside an unrelated subsystem: first a
    /// PML4 walk with an index out of `rsi`, then the heap's free list walked
    /// from a cursor out of `r9`. Neither was reproducible by reasoning about
    /// which registers *ought* to matter, and the second appeared because an
    /// unrelated file changed what the register allocator did.
    ///
    /// `xmm6`-`xmm15` are non-volatile under the same ABI and are deliberately
    /// **not** here: the target is built `-sse,+soft-float`, so ordinary Rust
    /// on this machine emits no SSE at all and nothing can have a live value in
    /// one. The SIMD kernels reach them only inside `#[target_feature]`
    /// functions, which save and restore their own. If this target ever gains
    /// hardware float, ten `movups` slots belong here.
    ///
    /// The order is the order the landing code reads them in, and the `const`
    /// block below asserts the offsets rather than trusting this sentence.
    rax: u64,
    rbx: u64,
    rcx: u64,
    rdx: u64,
    rsi: u64,
    rdi: u64,
    rbp: u64,
    r8: u64,
    r9: u64,
    r10: u64,
    r11: u64,
    r12: u64,
    r13: u64,
    r14: u64,
    r15: u64,
    rip: u64,
    rsp: u64,
    /// Set while this task is inside `guard`. Nothing is recovered otherwise.
    armed: u64,
}

const EMPTY: Pad = Pad {
    rax: 0, rbx: 0, rcx: 0, rdx: 0, rsi: 0, rdi: 0, rbp: 0, r8: 0, r9: 0, r10: 0,
    r11: 0, r12: 0, r13: 0, r14: 0, r15: 0, rip: 0, rsp: 0, armed: 0,
};

/// The offsets `idt.rs` reads the pad at, checked here rather than agreed by
/// hand across two files.
///
/// A comment saying "changing one means changing both" was true and was not
/// enough: adding two fields moves `rip` and `rsp`, and a landing pad reading
/// the old offsets jumps to whatever `r15` was and switches to a stack that is
/// really a register value. A `const` assertion fails the build, which is the
/// only useful kind of failure for a fact whose runtime symptom is a machine
/// that stops with nothing printed.
const _: () = {
    use core::mem::offset_of;
    assert!(offset_of!(Pad, rax) == 0);
    assert!(offset_of!(Pad, rbx) == 8);
    assert!(offset_of!(Pad, rcx) == 16);
    assert!(offset_of!(Pad, rdx) == 24);
    assert!(offset_of!(Pad, rsi) == 32);
    assert!(offset_of!(Pad, rdi) == 40);
    assert!(offset_of!(Pad, rbp) == 48);
    assert!(offset_of!(Pad, r8) == 56);
    assert!(offset_of!(Pad, r9) == 64);
    assert!(offset_of!(Pad, r10) == 72);
    assert!(offset_of!(Pad, r11) == 80);
    assert!(offset_of!(Pad, r12) == 88);
    assert!(offset_of!(Pad, r13) == 96);
    assert!(offset_of!(Pad, r14) == 104);
    assert!(offset_of!(Pad, r15) == 112);
    assert!(offset_of!(Pad, rip) == 120);
    assert!(offset_of!(Pad, rsp) == 128);
};

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
    Some(&pads[i].rax as *const u64)
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
#[inline(never)]
pub fn guard<F: FnOnce()>(f: F) -> Result<(), &'static str> {
    let Some(i) = slot() else {
        // No per-core storage yet, so nothing can be attributed and nothing
        // can be recovered. Run it plainly rather than pretending.
        f();
        return Ok(());
    };

    // Read as one block rather than fifteen `asm!`s, so nothing the compiler
    // emits between them can move a register the block claims to have read.
    // `rsp` is taken with `lea` because a `mov` would answer the value *after*
    // whatever this block itself needed, and the landing pad has to arrive at
    // the stack this function is standing on.
    let mut r = [0u64; 17];
    unsafe {
        core::arch::asm!(
            "mov [{p}], rax",
            "mov [{p} + 8], rbx",
            "mov [{p} + 16], rcx",
            "mov [{p} + 24], rdx",
            "mov [{p} + 32], rsi",
            "mov [{p} + 40], rdi",
            "mov [{p} + 48], rbp",
            "mov [{p} + 56], r8",
            "mov [{p} + 64], r9",
            "mov [{p} + 72], r10",
            "mov [{p} + 80], r11",
            "mov [{p} + 88], r12",
            "mov [{p} + 96], r13",
            "mov [{p} + 104], r14",
            "mov [{p} + 112], r15",
            // The pad's rip is the landing label below, taken against a local
            // label so it is the address of that code rather than of this
            // function.
            "lea {t}, [rip + 3f]",
            "mov [{p} + 120], {t}",
            "mov [{p} + 128], rsp",
            p = in(reg) r.as_mut_ptr(),
            t = out(reg) _,
            options(nostack, preserves_flags),
        );
    }

    {
        let pads = unsafe { PADS.get() };
        pads[i] = Pad {
            rax: r[0], rbx: r[1], rcx: r[2], rdx: r[3], rsi: r[4], rdi: r[5],
            rbp: r[6], r8: r[7], r9: r[8], r10: r[9], r11: r[10], r12: r[11],
            r13: r[12], r14: r[13], r15: r[14], rip: r[15], rsp: r[16],
            armed: 1,
        };
    }

    f();

    {
        let pads = unsafe { PADS.get() };
        pads[i].armed = 0;
    }
    // Jumped over on the normal path. A recovered fault lands on `3:` with
    // every register and the stack restored, and falls into the same tail.
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

    // What is *not* claimed here, said plainly rather than left as a gap.
    //
    // The failure this pad exists to stop is a caller's value sitting in a
    // non-volatile register that `guard` never spilled, so a probe written as
    // a function is the one shape that cannot catch it: the probe's own
    // prologue saves those registers and its epilogue puts them back, whether
    // or not the pad restored anything. Reproducing it needs the sentinel and
    // the guard inside one inlined body, which is what `mem::paging::checks`
    // is by accident -- it walks page tables with an index live in `rsi`
    // across the fault it takes on purpose, and it is what caught the missing
    // `rsi`/`rdi`. So the coverage is real and it lives in `diag paging`; a
    // claim here would be either flaky or vacuous, and both are worse.
    ok
}

//! Surviving a fault in a program the machine wrote for itself.
//!
//! Every vector but `#BP` used to be fatal, which is defensible for a kernel
//! bug and indefensible for a bad program: this system writes its own skills,
//! composes its own routing cores, and compiles its own code, and a stray
//! index in one of those stopped the machine. The thing most likely to fault
//! here is the thing the machine produced five minutes ago.
//!
//! **What is recoverable is bounded and stated.** A fault is caught only while
//! a task is inside `guard`. The code generator wraps compiled code in one
//! (`aiksi::jit`), the page-rights suite faults inside one on purpose, and
//! `main::section` wraps each boot selftest in one so an optional subsystem
//! cannot take the machine with it. The tree-walking interpreter does **not**
//! -- it is bounded by a step budget rather than guarded, and this said
//! otherwise for a long time. A fault anywhere else stays fatal, because there is no
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

/// How deep guards may nest on one task.
///
/// **One pad per task was a defect, not a simplification.** The inner guard's
/// exit cleared `armed`, which disarmed the outer one too, so a `guard` around
/// anything that itself guards -- `mem::paging::checks` does -- silently
/// stopped protecting the outer scope. Nothing failed; the protection was
/// simply gone, which is the shape of bug this module exists to prevent
/// elsewhere.
///
/// Four because the real nesting here is two -- a selftest wrapper around a
/// check that guards its own deliberate fault -- and a bound that is obviously
/// enough is better than one that is exactly enough.
const DEPTH: usize = 4;

static PADS: crate::sync::Racy<[[Pad; DEPTH]; SLOTS]> =
    crate::sync::Racy::new([[EMPTY; DEPTH]; SLOTS]);

/// How many guards each task is currently inside.
static DEPTHS: crate::sync::Racy<[usize; SLOTS]> = crate::sync::Racy::new([0; SLOTS]);

/// `LAST` when what was caught was a panic rather than a hardware exception.
/// Outside the vector range so it cannot collide with one.
const PANIC: u64 = 0xFFFF;

/// Whether a panic should be caught rather than halting the machine.
///
/// **Set only around the boot selftests, and cleared immediately after.** A
/// panic means a Rust invariant was violated, which is a weaker thing to
/// survive than a hardware exception -- so the window where it is survivable
/// is one block of code whose whole job is to try things that might not work.
/// Everywhere else, at every other time, a panic halts exactly as it did.
static SELFTEST: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Open or close the window in which a panic is recoverable.
pub fn selftest_window(on: bool) {
    SELFTEST.store(on, Ordering::Release);
}

pub fn in_selftest() -> bool {
    SELFTEST.load(Ordering::Acquire)
}

/// The landing pad for a panic, if one is armed and the window is open.
///
/// Unlike `take` there is no vector to filter on: a panic is a panic. The
/// window is the filter, and it is the whole of the restraint here.
pub fn take_panic() -> Option<*const u64> {
    if !in_selftest() {
        return None;
    }
    let i = slot()?;
    let pad = take_slot(i, PANIC)?;
    // A panic has no faulting instruction, and pointing at the last fault's
    // would be worse than saying nothing.
    LAST_RIP.store(0, Ordering::Relaxed);
    Some(pad)
}

/// Where the last recovered fault happened, as an absolute rip.
///
/// **Kept because recovering throws away the only evidence of *where*.** A
/// caught fault is attributed to whatever scope was guarded, which answers
/// "which check failed" and not "what broke". Those differ whenever a check
/// calls into something else: a fault inside the graphics stack, reached from
/// a power selftest, is a power failure by attribution and a graphics failure
/// in fact. Without the site there is nothing to tell them apart, and the
/// report would name the wrong subsystem with total confidence.
///
/// Zero when the last recovery was a panic, which has no faulting instruction
/// in the same sense -- `site()` answers `None` there rather than pointing at
/// address zero.
static LAST_RIP: AtomicU64 = AtomicU64::new(0);

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
pub fn take(vector: u8, rip: u64) -> Option<*const u64> {
    // Only the vectors a program can plausibly cause. A machine check or a
    // double fault says the machine is wrong rather than the program.
    if !matches!(vector, 0 | 5 | 6 | 13 | 14 | 17 | 19) {
        return None;
    }
    let i = slot()?;
    let pad = take_slot(i, vector as u64)?;
    LAST_RIP.store(rip, Ordering::Relaxed);
    Some(pad)
}

/// Where the last recovered fault was, if it was a fault rather than a panic.
pub fn site() -> Option<u64> {
    match LAST_RIP.load(Ordering::Relaxed) {
        0 => None,
        r => Some(r),
    }
}

/// The innermost armed pad for a task, popped.
///
/// **Popping rather than only disarming is what makes nesting safe.** A fault
/// taken while unwinding out of the inner guard now lands in the *outer* one
/// instead of being fatal, and it still cannot loop, because the depth strictly
/// decreases on every take and the outermost frame has nowhere left to go.
fn take_slot(i: usize, why: u64) -> Option<*const u64> {
    let depths = unsafe { DEPTHS.get() };
    let d = depths[i];
    if d == 0 {
        return None;
    }
    let top = d - 1;
    let pads = unsafe { PADS.get() };
    if pads[i][top].armed == 0 {
        return None;
    }
    pads[i][top].armed = 0;
    depths[i] = top;
    LAST.store(why, Ordering::Relaxed);
    COUNT.fetch_add(1, Ordering::Relaxed);
    Some(&pads[i][top].rax as *const u64)
}

/// Jump to a landing pad. Never returns.
///
/// **One copy of this exists and that is deliberate.** It restores *every*
/// general-purpose register, not a set chosen from a calling convention:
/// `guard` is inlined into its callers, so the longjmp crosses no ABI boundary
/// and a caller's live value can be sitting in `rax` or `r9` as easily as in
/// `rbx`. It was written out twice for a while -- once in the fault handler and
/// once for panics -- and two copies of a register list that must not drift is
/// the same bet `idt.rs` lost over the stub stride.
///
/// `rcx` is the cursor and is restored last, from its own slot through itself.
/// That leaves nothing to hold the jump target, so the target is pushed onto
/// the already-restored stack and `ret` takes it: eight bytes below `rsp`,
/// which nothing owns.
///
/// The offsets are asserted against `Pad` above.
///
/// # Safety
/// `pad` must be a pointer `take`/`take_panic` answered, and the frame it
/// belongs to must still be on the stack.
pub unsafe fn land(pad: *const u64) -> ! {
    unsafe {
        core::arch::asm!(
            "mov rax, [rcx]",
            "mov rbx, [rcx + 8]",
            "mov rdx, [rcx + 24]",
            "mov rsi, [rcx + 32]",
            "mov rdi, [rcx + 40]",
            "mov rbp, [rcx + 48]",
            "mov r8,  [rcx + 56]",
            "mov r9,  [rcx + 64]",
            "mov r10, [rcx + 72]",
            "mov r11, [rcx + 80]",
            "mov r12, [rcx + 88]",
            "mov r13, [rcx + 96]",
            "mov r14, [rcx + 104]",
            "mov r15, [rcx + 112]",
            "mov rsp, [rcx + 128]",
            "push [rcx + 120]",
            "mov rcx, [rcx + 16]",
            "ret",
            in("rcx") pad,
            options(noreturn),
        );
    }
}

/// Every name `describe` can produce.
///
/// Exists because `repair`'s clues match on these strings, so renaming a vector
/// here would silently stop every repair signature matching -- no error, no
/// fault, just a machine repairing itself worse than it used to. A claim over
/// this list is what turns that into a failure somebody sees.
pub fn names() -> &'static [&'static str] {
    &[
        "divide error",
        "bound range exceeded",
        "invalid opcode",
        "general protection fault",
        "page fault",
        "alignment check",
        "SIMD floating point",
        "panicked",
        "fault",
    ]
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
        PANIC => "panicked",
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
/// What `guarded` did, which `guard`'s `Result` cannot express.
///
/// **The third case is the one that mattered.** With no per-core storage, or
/// nested too deep, `guard` ran the closure unprotected and answered `Ok(())`
/// -- indistinguishable from having run it safely. A caller deciding whether a
/// subsystem is healthy would read "ran unprotected, and a fault would have
/// been fatal" as "passed".
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Caught {
    /// Ran under a pad and did not fault.
    Ran,
    /// Faulted, and was recovered. The string says what.
    Faulted(&'static str),
    /// Ran with no pad, so a fault would have been fatal. Not a pass.
    Unguarded(&'static str),
}

/// `guard`, with the three states it actually has.
#[inline(never)]
pub fn guarded<F: FnOnce()>(f: F) -> Caught {
    match guard_inner(f) {
        Ok(true) => Caught::Ran,
        Ok(false) => Caught::Unguarded("no landing pad, so this ran unprotected"),
        Err(e) => Caught::Faulted(e),
    }
}

/// The original two-state form, for callers that only need to know it survived.
#[inline(never)]
pub fn guard<F: FnOnce()>(f: F) -> Result<(), &'static str> {
    guard_inner(f).map(|_| ())
}

/// `Ok(true)` ran guarded, `Ok(false)` ran unguarded, `Err` was recovered.
#[inline(never)]
fn guard_inner<F: FnOnce()>(f: F) -> Result<bool, &'static str> {
    let Some(i) = slot() else {
        // No per-core storage yet, so nothing can be attributed and nothing
        // can be recovered. Run it plainly rather than pretending.
        f();
        return Ok(false);
    };

    // Deeper than the stack goes. Running unguarded is the honest answer and
    // is reported as such; refusing to run would turn a depth limit into a
    // behaviour change.
    {
        let depths = unsafe { DEPTHS.get() };
        if depths[i] >= DEPTH {
            f();
            return Ok(false);
        }
    }

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

    let d = {
        let depths = unsafe { DEPTHS.get() };
        let d = depths[i];
        let pads = unsafe { PADS.get() };
        pads[i][d] = Pad {
            rax: r[0], rbx: r[1], rcx: r[2], rdx: r[3], rsi: r[4], rdi: r[5],
            rbp: r[6], r8: r[7], r9: r[8], r10: r[9], r11: r[10], r12: r[11],
            r13: r[12], r14: r[13], r15: r[14], rip: r[15], rsp: r[16],
            armed: 1,
        };
        depths[i] = d + 1;
        d
    };

    // **Called behind an opaque condition, and that is not superstition.**
    // A closure the optimiser can prove diverges -- one that always panics,
    // or ends in `unimplemented!()` -- makes everything after this line
    // unreachable, so it is deleted. Including the landing block below, which
    // the pad's `rip` points at and which only a longjmp ever reaches. The
    // compiler cannot see that edge, so it is entitled to remove the target.
    //
    // Measured, before this line existed: a guarded closure whose body was a
    // bare `panic!` landed at rva 0x5e06e8, past the end of `.text`, and took
    // a #PF at cr2 = 8. The pad was correct and pointed at code that was no
    // longer there.
    //
    // `black_box` on the condition costs one compare and makes the tail
    // reachable in the compiler's view, which is all that is needed.
    if core::hint::black_box(true) {
        f();
    }

    // The normal exit pops. The faulting exit pops inside `take_slot`, so both
    // paths leave the depth where this frame found it.
    {
        let depths = unsafe { DEPTHS.get() };
        depths[i] = d;
        let pads = unsafe { PADS.get() };
        pads[i][d].armed = 0;
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
        // The closure was abandoned mid-flight. If it was inside a `kprintln!`
        // it left the console locked, and the next print -- which is the one
        // reporting this very fault -- would spin to `PATIENCE` and panic.
        // A recovered fault must not become a fatal one on the way out.
        unsafe { crate::gfx::console::release_locks() };
        // And the paint claim, for the same reason one line up. A guarded
        // scope that faulted inside `desk::with` never ran `Claim::drop`, and
        // the claim is reentrant for its holder -- so the task that leaked it
        // carries on working while every other task blocks forever in
        // `Claim::wait`. The compositor is the one that matters: it is the
        // only painter left, so a leak here is a screen that stops with the
        // machine still running, which is the exact failure this tree has been
        // chasing. Releases only this task's.
        unsafe { crate::gfx::desk::release_claim() };
        Err(describe())
    } else {
        Ok(true)
    }
}

pub fn selftest() -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    // The fault stubs are indexed by arithmetic rather than named one at a
    // time, so the stride has to be what it claims. Every stub begins with a
    // `push imm8`, which is `0x6a`, and a stride that had drifted would put
    // something else at one of these addresses -- silently, and only for the
    // vectors past the drift.
    claim(
        &mut ok,
        {
            let base = crate::cpu::idt::stub_base();
            (0..32u64).all(|v| unsafe {
                core::ptr::read_volatile(
                    (base + v * crate::cpu::idt::STUB_STRIDE) as *const u8,
                ) == 0x6a
            })
        },
        "every fault stub is where the stride says it is",
    );

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

    // ---- nesting, which was broken and silently so ------------------------
    //
    // **The bug this pair exists to catch.** One pad per task meant the inner
    // guard's *normal exit* cleared the arm, so an outer guard around anything
    // that itself guards stopped protecting anything at all. Nothing failed
    // and nothing printed; the protection was simply gone.
    //
    // Note what happens if the fix is wrong: the fault below is not caught,
    // and the machine halts here rather than printing FAIL. That is the honest
    // shape for this claim -- there is no way to ask "was I protected" except
    // by needing it.
    let mut inner_ran = false;
    let outer = guard(|| {
        let _ = guard(|| {
            inner_ran = true;
        });
        // The inner guard has exited normally. If its exit disarmed this one,
        // the machine stops on the next line.
        unsafe {
            core::ptr::read_volatile(0x0 as *const u64);
        }
    });
    claim(&mut ok, inner_ran, "a guard nested inside a guard runs");
    claim(
        &mut ok,
        outer.is_err(),
        "and the outer one still catches a fault after the inner one exited",
    );

    // A fault in the inner guard is caught by the inner guard, and the outer
    // one is left armed for its own.
    let mut inner_caught = false;
    let outer = guard(|| {
        inner_caught = guard(|| unsafe {
            core::ptr::read_volatile(0x0 as *const u64);
        })
        .is_err();
        unsafe {
            core::ptr::read_volatile(0x0 as *const u64);
        }
    });
    claim(&mut ok, inner_caught, "a fault in the inner guard is caught there");
    claim(&mut ok, outer.is_err(), "and the outer one catches its own afterwards");

    // The depth is where it started, or every guarded call leaks a slot and
    // the fifth one runs unprotected.
    claim(
        &mut ok,
        slot().map(|i| unsafe { DEPTHS.get() }[i] == 0).unwrap_or(false),
        "and the nesting depth is back to zero",
    );

    // `guarded` distinguishes the case `guard` could not: ran unprotected.
    claim(
        &mut ok,
        guarded(|| {}) == Caught::Ran,
        "guarded reports a clean run as having been guarded",
    );
    claim(
        &mut ok,
        matches!(
            guarded(|| unsafe { core::ptr::read_volatile(0x0 as *const u64); }),
            Caught::Faulted(_)
        ),
        "and a caught fault as faulted rather than as a pass",
    );

    // ---- panics, which bypassed every one of the above --------------------
    //
    // Hardware exceptions were the only thing recoverable, and an `assert!` is
    // how a selftest usually fails. So the majority of the cases this whole
    // mechanism looks like it covers, it did not.
    claim(
        &mut ok,
        !in_selftest() && take_panic().is_none(),
        "with the window shut, a panic finds no pad and stays fatal",
    );

    selftest_window(true);
    // Deliberately a closure that *always* panics, because that is the shape
    // that broke: the optimiser deletes the landing block when it can prove
    // the call diverges. `guard_inner` calls behind a `black_box`d condition
    // to stop it, and this is the claim that the defence is still there.
    let panicked = guard(|| {
        panic!("a selftest that asserts its way out");
    });
    // Shut immediately. Leaving it open would make every later panic in the
    // boot recoverable, which is exactly the blanket this is written to avoid.
    selftest_window(false);
    claim(&mut ok, panicked.is_err(), "with it open, a panic inside a guard is caught");
    claim(
        &mut ok,
        panicked == Err("panicked"),
        "and is named a panic rather than borrowing a fault's description",
    );
    claim(
        &mut ok,
        !in_selftest(),
        "and the window is shut again afterwards",
    );

    // The console was locked by the `kprintln!` this suite is made of, and the
    // claim above printed after the catch -- so the release worked. Said out
    // loud because the failure would be a hang rather than a wrong answer.
    claim(&mut ok, true, "and the console survived being abandoned mid-print");

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

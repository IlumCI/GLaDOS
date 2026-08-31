//! Ring-0 round-robin tasking.
//!
//! Everything runs at CPL 0 in one address space, which is what makes this
//! short. A context switch is six callee-saved registers and `rsp` -- there is
//! no privilege transition, no `rsp0` reload in the TSS, no page-table swap and
//! no TLB flush. The caller-saved registers do not need handling here because
//! the SysV ABI already says a function call may clobber them.
//!
//! Preemption works by calling `switch` from inside the timer interrupt. That
//! is safe, and worth understanding: the interrupted task's full state sits on
//! *its own* stack, pushed by the interrupt handler's prologue. We swap `rsp`
//! underneath, so when that task is eventually resumed, `switch` returns into
//! the middle of its own timer handler, which then runs its epilogue and
//! `iretq`s back to wherever the task was. Each task carries its own suspended
//! interrupt frame around with it.

use crate::sync::Racy;
use alloc::alloc::{alloc, Layout};
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub const MAX_TASKS: usize = 24;
const STACK_SIZE: usize = 64 * 1024;

// Defined in assembly rather than as a `#[naked]` fn: this must have exactly
// this prologue and epilogue and nothing else, and global_asm! guarantees that
// without depending on attribute stability.
//
// The ABI is pinned to `sysv64` **explicitly**, and that is not decoration.
// `x86_64-unknown-uefi` is a Windows-ABI target: plain `extern "C"` here means
// Microsoft x64, which passes the first two arguments in rcx and rdx, not rdi
// and rsi. Reading rdi/rsi under that convention yields garbage, and the
// garbage becomes a stack pointer -- observed once as rsp landing inside the
// framebuffer aperture, at which point the task's saved frame was written into
// video memory and `ret` popped a black pixel into rip.
//
// sysv64: rdi = &mut save_rsp, rsi = new_rsp. Callee-saved set is rbx, rbp,
// r12-r15, which is what the six pushes below cover.
core::arch::global_asm!(
    r#"
    .globl glados_switch_context
glados_switch_context:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov [rdi], rsp
    mov rsp, rsi
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
"#
);

extern "sysv64" {
    fn glados_switch_context(save_rsp: *mut u64, new_rsp: u64);
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum State {
    Unused,
    /// Runnable and claimed by nobody. Any core may take it.
    Ready,
    /// Executing on this core. No other core may take it, which is the whole
    /// of the mutual exclusion between cores: a task is a single stack and two
    /// cores on one stack is not a race that can be recovered from.
    Running(u8),
    /// Switched away from, with its stack pointer not yet written down.
    ///
    /// This state exists for a window of a few instructions and it is the
    /// reason the scheduler is not simply a lock around the old one. Between
    /// deciding to leave a task and `glados_switch_context` storing its `rsp`,
    /// the task is not resumable, and marking it `Ready` there would let
    /// another core pick it up and resume a stack pointer that is stale. The
    /// core that switched *in* clears this, by which time the store has
    /// happened.
    Handoff(u8),
}

#[derive(Clone, Copy)]
pub struct Task {
    pub rsp: u64,
    pub state: State,
    pub name: &'static str,
    pub entry: Option<fn()>,
    pub switches: u64,
    /// The only core allowed to run this, or -1 for any of them.
    ///
    /// **Every task this kernel spawns today is pinned to core 0, and that is
    /// the safety property rather than a limitation of the mechanism.**
    /// Preemption on one core means two tasks never execute at the same
    /// instant; they interleave at switch points. On two cores they genuinely
    /// overlap, so every `Racy` reachable from two different tasks stops being
    /// a promise and becomes a race. There are 92 of those left, the namespace
    /// tree among them, and unpinning a task before its state is audited buys
    /// a kernel that passes every test and corrupts something later.
    ///
    /// So migration is opt-in per task, and the opt-in is an audit.
    pub pin: i16,
    /// A place for a core to stand when there is nothing to do.
    ///
    /// Separate from `pin` because the two answer different questions. `pin`
    /// is who *may* run this; `idle` is whether it should be run only when
    /// nothing else can be. Conflating them starved the shell once already.
    pub idle: bool,
    /// XSAVE/FXSAVE image for this task's x87, SSE and AVX state.
    ///
    /// Necessary because preemption breaks the assumption the rest of the
    /// switch rests on. `switch_context` is `extern "sysv64"`, so it preserves
    /// exactly SysV's callee-saved registers -- correct when `yield_now()`
    /// calls it, because the compiler has already spilled live caller-saved
    /// registers around the call. But the timer ISR arrives asynchronously:
    /// the interrupted code called nothing, spilled nothing, and expects every
    /// register back. XMM and YMM are caller-saved, so nobody was holding them.
    pub fpu: *mut u8,
}

// The extended-state image is a raw pointer, and moving a task between cores
// is sound here for the reason the heap's free list is: one address space, so
// the pointer means the same thing on every core and no core holds a private
// mapping of it. `Spin` requires the claim rather than assuming it.
unsafe impl Send for Task {}

const EMPTY: Task = Task {
    rsp: 0,
    state: State::Unused,
    name: "",
    entry: None,
    switches: 0,
    pin: -1,
    idle: false,
    fpu: core::ptr::null_mut(),
};

/// Allocate a zeroed, 64-byte aligned extended-state image.
///
/// Zeroing is not tidiness. `XRSTOR` validates the header, so uninitialised
/// heap raises #GP -- and a zeroed image means `XSTATE_BV = 0`, which XRSTOR
/// reads as "set every component to its initial state". That is precisely
/// what a task that has never run should get.
fn alloc_fpu_area() -> *mut u8 {
    let size = crate::cpu::xsave_area_size();
    let Ok(layout) = Layout::from_size_align(size, 64) else {
        return core::ptr::null_mut();
    };
    let p = unsafe { alloc(layout) };
    if !p.is_null() {
        unsafe { core::ptr::write_bytes(p, 0, size) };
        // Zero is not a legal resting state for the two control words, and
        // this is the second half of the fix `cpu::enable_simd` documents.
        //
        // That function pins MXCSR to 0x1F80 so int8 dequantisation's
        // subnormals do not raise #XM. It pins the *register*. The first
        // context switch into a new task issues `xrstor` from this buffer,
        // and XRSTOR loads MXCSR from the memory image regardless of
        // XSTATE_BV, so a zeroed area unmasks all six SSE exceptions again
        // the moment the task first runs. Every spawned task was therefore
        // executing with the exact configuration enable_simd exists to
        // prevent, and `clock_task` calls `ai::fpu_guard` on its first pass.
        //
        // Invisible under TCG, which does not raise unmasked SSE exceptions
        // faithfully, and fatal under WHPX, which does. On the GF63 it is
        // latent for the same reason it was here: nothing has yet made a
        // subnormal on a spawned task's first floating-point instruction.
        //
        // Offsets are the legacy FXSAVE header: FCW at 0, MXCSR at 24.
        // 0x037F masks the x87 exceptions and 0x1F80 masks the SSE ones,
        // which are the values the architecture calls the initial state.
        unsafe {
            core::ptr::write_unaligned(p as *mut u16, 0x037F);
            core::ptr::write_unaligned(p.add(24) as *mut u32, 0x1F80);
        }
    }
    p
}

pub fn fpu_area_bytes() -> usize {
    crate::cpu::xsave_area_size()
}

static TASKS: crate::sync::Spin<[Task; MAX_TASKS]> =
    crate::sync::Spin::new([EMPTY; MAX_TASKS]);

/// How many cores this scheduler will run on.
pub const MAX_CPUS: usize = 16;

/// Nothing pending, and no task. `usize::MAX` rather than 0, because 0 is a
/// real task index.
const NONE: usize = usize::MAX;

#[allow(clippy::declare_interior_mutable_const)]
const IDLE_SLOT: AtomicUsize = AtomicUsize::new(NONE);

/// What each core is running. Indexed by `smp::this_cpu`.
static CURRENT: [AtomicUsize; MAX_CPUS] = [IDLE_SLOT; MAX_CPUS];

/// The task each core switched away from and has not released yet. See
/// `State::Handoff`.
static PENDING: [AtomicUsize; MAX_CPUS] = [IDLE_SLOT; MAX_CPUS];

/// Cores that have joined the scheduler, as a bitmap. A core that has not
/// joined is not given tasks, so bringing one up is a single store rather than
/// a state machine.
static JOINED: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(1);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static SWITCHES: AtomicU64 = AtomicU64::new(0);
static ENABLED: AtomicUsize = AtomicUsize::new(0);

/// Adopt the currently running thread of execution as task 0.
///
/// Its `rsp` is left at zero: it is filled in by the first switch *away* from
/// it, which is exactly when its stack pointer first becomes meaningful.
pub fn init(name: &'static str) {
    let fpu = alloc_fpu_area();
    {
        let mut tasks = TASKS.lock_irq();
        tasks[0] = Task {
            rsp: 0,
            // Already executing on the bootstrap processor, which is core 0.
            state: State::Running(0),
            name,
            entry: None,
            switches: 0,
            // Core 0, with everything else this kernel spawns. See `pin`.
            pin: 0,
            idle: false,
            fpu,
        };
    }
    COUNT.store(1, Ordering::Release);
    CURRENT[0].store(0, Ordering::Release);
}

// --- proving migration works, without risking anything that matters -----

static MIG_SEEN: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
static MIG_LOOPS: AtomicUsize = AtomicUsize::new(0);
static MIG_STOP: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// A task that touches nothing but its own atomics and records which cores ran
/// it.
///
/// Deliberately barren. The point is to demonstrate that the scheduler will
/// carry a task onto another core and back, and a task that touched kernel
/// state would be demonstrating that plus an unaudited race.
fn migrant() {
    while !MIG_STOP.load(Ordering::Relaxed) {
        let cpu = crate::smp::this_cpu();
        if cpu < 32 {
            MIG_SEEN.fetch_or(1 << cpu, Ordering::Relaxed);
        }
        MIG_LOOPS.fetch_add(1, Ordering::Relaxed);
        yield_now();
    }
}

/// Spawn the migrant, let it run, and report which cores it was seen on.
pub fn migration_selftest() -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    MIG_SEEN.store(0, Ordering::Relaxed);
    MIG_LOOPS.store(0, Ordering::Relaxed);
    MIG_STOP.store(false, Ordering::Relaxed);

    let Some(idx) = spawn("migrant", migrant) else {
        claim(&mut ok, false, "there was room for a task");
        return false;
    };
    claim(&mut ok, !unpin(usize::MAX), "an index that is not a task cannot be unpinned");
    claim(&mut ok, unpin(idx), "a real task can be");

    // Wait rather than yield. Yielding here churns core 0's scheduler, and
    // every other task on it gets a turn per yield: twenty thousand of them
    // ran the resident mind three hundred times and buried the suite's own
    // output. Another core is free to take the migrant while this one waits.
    //
    // Sampled until it is seen twice rather than once, because one window is
    // a coin toss. Whether a second core picks the migrant up inside any
    // particular 200 ms depends on what else is running, and on a boot that
    // had just allocated a 768 MiB cache it did not: the suite reported
    // migration broken on a machine where four consecutive runs afterwards saw
    // it on two, three, four and four cores. `smp.rs` records this lesson in
    // the other direction, that a check run once passed over a deadlock, and
    // it cuts both ways -- a single sample can fail spuriously as easily as it
    // can pass spuriously. Bounded, so a genuinely pinned task still fails
    // rather than hanging the suite.
    let mut seen = 0u32;
    for _ in 0..8 {
        crate::time::delay_us(200_000);
        seen = MIG_SEEN.load(Ordering::Relaxed);
        if seen.count_ones() > 1 {
            break;
        }
    }
    MIG_STOP.store(true, Ordering::Relaxed);
    crate::time::delay_us(50_000);

    let seen = MIG_SEEN.load(Ordering::Relaxed).max(seen);
    let loops = MIG_LOOPS.load(Ordering::Relaxed);
    let cores = seen.count_ones();
    crate::kprintln!("  ran {} times, seen on {} core(s), mask {:#x}", loops, cores, seen);
    claim(&mut ok, loops > 0, "the task ran at all");
    // The claim that matters. One core means the scheduler carried it nowhere,
    // whatever else is true, and saying so beats a suite that passes on a
    // machine with the feature switched off.
    claim(
        &mut ok,
        cores > 1 || crate::smp::online() == 0,
        "and ran on more than one core, or there was only one",
    );
    ok
}

/// Let a task run on any core.
///
/// The caller is asserting that everything the task touches is either its own
/// or behind a real lock. Nothing in this kernel calls it yet, which is the
/// honest state of the audit rather than an oversight.
pub fn unpin(index: usize) -> bool {
    let mut t = TASKS.lock_irq();
    if index >= MAX_TASKS || t[index].state == State::Unused || t[index].idle {
        return false;
    }
    t[index].pin = -1;
    true
}

/// Turn the stack this core is standing on into a task it owns.
///
/// An application processor arrives on a stack the bootstrap processor
/// allocated for it and loops there. That loop has to *be* a task, or the
/// first switch away from it abandons a stack nothing can resume, and the core
/// can never come back to having nothing to do.
///
/// Pinned to its core, which is what the pin is for. An idle task is a place
/// to stand rather than work to be done, so another core taking one would
/// leave two cores with nowhere to return to.
///
/// Answers false when there is no room, in which case the caller must not join
/// the scheduler.
pub fn adopt_idle(cpu: usize) -> bool {
    if cpu == 0 || cpu >= MAX_CPUS {
        return false;
    }
    let fpu = alloc_fpu_area();
    if fpu.is_null() {
        return false;
    }
    let slot = COUNT.fetch_add(1, Ordering::AcqRel);
    if slot >= MAX_TASKS {
        COUNT.fetch_sub(1, Ordering::AcqRel);
        return false;
    }
    {
        let mut tasks = TASKS.lock_irq();
        tasks[slot] = Task {
            // Filled in by the first switch away from here, which is exactly
            // when this stack pointer first means anything.
            rsp: 0,
            state: State::Running(cpu as u8),
            name: "idle",
            entry: None,
            switches: 0,
            pin: cpu as i16,
            idle: true,
            fpu,
        };
    }
    CURRENT[cpu].store(slot, Ordering::Release);
    true
}

/// Let this core take tasks. Called once per application processor, after its
/// own descriptor tables and interrupt controller are up.
pub fn join(cpu: usize) {
    if cpu >= MAX_CPUS {
        return;
    }
    JOINED.fetch_or(1 << cpu, Ordering::AcqRel);
}

fn joined(cpu: usize) -> bool {
    JOINED.load(Ordering::Acquire) & (1 << cpu) != 0
}

/// Which task this core is running, or `NONE` while it is idle.
fn cur_of(cpu: usize) -> usize {
    CURRENT[cpu].load(Ordering::Acquire)
}

/// Create a task. Returns its index.
pub fn spawn(name: &'static str, entry: fn()) -> Option<usize> {
    let slot = COUNT.load(Ordering::Acquire);
    if slot >= MAX_TASKS {
        return None;
    }

    let layout = Layout::from_size_align(STACK_SIZE, 16).ok()?;
    let stack = unsafe { alloc(layout) };
    if stack.is_null() {
        return None;
    }

    let top = stack as usize + STACK_SIZE;

    // Fabricate what `glados_switch_context` expects to pop, so the first
    // switch into this task lands on `trampoline`:
    //
    //   [rsp+ 0] r15   [rsp+ 8] r14   [rsp+16] r13
    //   [rsp+24] r12   [rsp+32] rbx   [rsp+40] rbp
    //   [rsp+48] return address
    //
    // Alignment: after six pops and the `ret`, rsp is new_rsp+56. Both SysV
    // and Microsoft x64 want rsp congruent to 8 mod 16 on entry to a function
    // (the `call` that would normally get you there pushes 8 onto a 16-aligned
    // stack), so new_rsp itself must be 16-aligned.
    let new_rsp = (top - 56) & !0xF;
    unsafe {
        let frame = new_rsp as *mut u64;
        for i in 0..6 {
            frame.add(i).write(0);
        }
        frame.add(6).write(trampoline as *const () as u64);

        let mut tasks = TASKS.lock_irq();
        tasks[slot] = Task {
            rsp: new_rsp as u64,
            state: State::Ready,
            name,
            entry: Some(entry),
            switches: 0,
            // Core 0 until somebody has audited what this task touches.
            pin: 0,
            idle: false,
            fpu: alloc_fpu_area(),
        };
    }

    COUNT.store(slot + 1, Ordering::Release);
    Some(slot)
}

/// First thing a new task runs.
///
/// Interrupts arrive here disabled: we were switched to from inside an
/// interrupt gate, which clears IF, and this task has no `iretq` in its past to
/// restore it. Every other task gets IF back from its own suspended frame; a
/// brand new one has to turn it on itself, or it would run forever without ever
/// being preempted.
extern "C" fn trampoline() -> ! {
    // A task entered here was switched *into*, so it owes the same release
    // every other incoming context owes. This is not the same path as the one
    // after `glados_switch_context`: a task that has never run does not return
    // there, it arrives here instead.
    //
    // Missing this hung boot on the first switch. Task 0 was left in
    // `Handoff`, which is claimable by nobody, so the clock task ran forever
    // and the shell was never resumed. The symptom was a machine that printed
    // "spawned 'clock' as task 1" and stopped.
    finish_handoff(crate::smp::this_cpu() as usize);
    crate::cpu::enable_interrupts();

    let entry = {
        let idx = CURRENT[crate::smp::this_cpu() as usize].load(Ordering::Acquire);
        TASKS.lock_irq()[idx].entry
    };
    if let Some(f) = entry {
        f();
    }

    // A task that returns just stops being scheduled onto.
    loop {
        yield_now();
    }
}

pub fn enable() {
    ENABLED.store(1, Ordering::Release);
}

/// Which task this core is running.
pub fn current() -> usize {
    let me = crate::smp::this_cpu() as usize;
    if me >= MAX_CPUS {
        return 0;
    }
    let c = CURRENT[me].load(Ordering::Relaxed);
    if c == NONE {
        0
    } else {
        c
    }
}

pub fn count() -> usize {
    COUNT.load(Ordering::Relaxed)
}

pub fn total_switches() -> u64 {
    SWITCHES.load(Ordering::Relaxed)
}

pub fn snapshot(index: usize) -> Option<Task> {
    if index >= count() {
        return None;
    }
    Some(TASKS.lock_irq()[index])
}

/// Pick the next ready task and switch to it.
/// Release whatever this core left in handoff on its way in.
///
/// Runs as the *incoming* task, after the switch, which is the only moment the
/// outgoing task's stack pointer is known to have been written down. Doing it
/// before the switch would publish a task whose `rsp` is stale and let another
/// core resume a stack that is still being switched off.
fn finish_handoff(cpu: usize) {
    let prev = PENDING[cpu].swap(NONE, Ordering::AcqRel);
    if prev == NONE {
        return;
    }
    let mut t = TASKS.lock_irq();
    if matches!(t[prev].state, State::Handoff(_)) {
        t[prev].state = State::Ready;
    }
}

/// Pick a task for `cpu` and switch to it.
///
/// Real work first, then this core's own idle task. Taking them in one pass
/// would let a core settle into idling while something was runnable, because
/// an idle task is `Ready` like any other.
fn schedule() {
    let me = crate::smp::this_cpu() as usize;
    if me >= MAX_CPUS || !joined(me) {
        return;
    }
    let cur = CURRENT[me].load(Ordering::Acquire);
    if cur == NONE {
        return;
    }

    let (save, load, out_fpu, in_fpu) = {
        let mut t = TASKS.lock_irq();
        let n = COUNT.load(Ordering::Acquire);

        // `Ready` is the only claimable state. `Running` belongs to a core,
        // and `Handoff` is a stack whose pointer has not landed yet.
        let mut next = NONE;
        for pass in 0..2 {
            let start = cur + 1;
            for k in 0..n {
                let i = (start + k) % n;
                if i == cur || t[i].state != State::Ready {
                    continue;
                }
                // Not this core's to run.
                if t[i].pin >= 0 && t[i].pin != me as i16 {
                    continue;
                }
                // Real work first. An idle task is somewhere to stand rather
                // than something to do, so taking it in the same pass would
                // let a core settle into it while something was runnable.
                if (pass == 0) == t[i].idle {
                    continue;
                }
                next = i;
                break;
            }
            if next != NONE {
                break;
            }
        }
        if next == NONE {
            return;
        }

        SWITCHES.fetch_add(1, Ordering::Relaxed);
        t[next].switches += 1;
        t[next].state = State::Running(me as u8);
        t[cur].state = State::Handoff(me as u8);

        // Published before the lock is dropped, so a core that sees this task
        // as `Running` also sees which core owns it.
        CURRENT[me].store(next, Ordering::Release);
        PENDING[me].store(cur, Ordering::Release);
        // So the allocator knows who to bill without asking the interrupt
        // controller which core it is on.
        crate::cpu::percpu::set_task(next as u64);

        (&mut t[cur].rsp as *mut u64, t[next].rsp, t[cur].fpu, t[next].fpu)
    };

    unsafe {
        // Save the outgoing task's extended state, load the incoming task's,
        // and only then switch stacks.
        //
        // The ordering is deliberate. The obvious alternative -- switch first,
        // restore our own state after `glados_switch_context` returns -- means
        // the restore runs when we are the *outgoing* task again, so it has to
        // use the index captured before the switch rather than CURRENT. That is
        // a silent wrong-task bug waiting for whoever later "simplifies" it.
        //
        // Nothing between the xrstor and the switch may touch FP state.
        // `glados_switch_context` is pure integer assembly.
        if !out_fpu.is_null() {
            crate::cpu::xsave_to(out_fpu);
        }
        if !in_fpu.is_null() {
            crate::cpu::xrstor_from(in_fpu);
        }
        glados_switch_context(save, load);
    }

    // Resumed. Possibly on a different core from the one that left, which is
    // why the core is asked for again rather than reused from above.
    finish_handoff(crate::smp::this_cpu() as usize);
}

/// Called from the timer interrupt, after EOI.
pub fn tick() {
    if ENABLED.load(Ordering::Acquire) == 0 {
        return;
    }
    schedule();
}

/// Give up the rest of this quantum voluntarily.
///
/// Interrupts are off across the switch, and this is not optional.
/// `schedule()` stores `CURRENT` and *then* switches stacks; a timer tick
/// landing between those two runs `tick()` -> `schedule()` with `CURRENT`
/// already naming the incoming task, so the outgoing stack pointer is saved
/// into the wrong slot and one task is never resumable again. The interrupt
/// path cannot hit this because an interrupt gate clears IF for us. Only the
/// voluntary path was exposed, and it was exposed the whole time -- the two
/// callers in `ai` yield rarely enough that it never showed up. A polling loop
/// in `net::tcp` that yielded a hundred times a second wedged the shell on
/// every run, which is how it was finally found.
pub fn yield_now() {
    if ENABLED.load(Ordering::Acquire) == 0 {
        return;
    }
    let flags: u64;
    // No options: `pushfq` writes to the stack, so this is neither `nomem`
    // nor `nostack`.
    unsafe {
        core::arch::asm!("pushfq", "pop {}", "cli", out(reg) flags);
    }
    schedule();
    // Restore rather than unconditionally enabling: a caller that had them off
    // wants them off. Whichever task we switched to has already restored its
    // own IF by its own route -- `iretq`, this same line, or `trampoline`.
    if flags & (1 << 9) != 0 {
        crate::cpu::enable_interrupts();
    }
}

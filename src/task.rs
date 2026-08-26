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

pub const MAX_TASKS: usize = 8;
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
    Ready,
}

#[derive(Clone, Copy)]
pub struct Task {
    pub rsp: u64,
    pub state: State,
    pub name: &'static str,
    pub entry: Option<fn()>,
    pub switches: u64,
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

const EMPTY: Task = Task {
    rsp: 0,
    state: State::Unused,
    name: "",
    entry: None,
    switches: 0,
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

static TASKS: Racy<[Task; MAX_TASKS]> = Racy::new([EMPTY; MAX_TASKS]);
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static SWITCHES: AtomicU64 = AtomicU64::new(0);
static ENABLED: AtomicUsize = AtomicUsize::new(0);

/// Adopt the currently running thread of execution as task 0.
///
/// Its `rsp` is left at zero: it is filled in by the first switch *away* from
/// it, which is exactly when its stack pointer first becomes meaningful.
pub fn init(name: &'static str) {
    let fpu = alloc_fpu_area();
    unsafe {
        let tasks = TASKS.get();
        tasks[0] = Task {
            rsp: 0,
            state: State::Ready,
            name,
            entry: None,
            switches: 0,
            fpu,
        };
    }
    COUNT.store(1, Ordering::Release);
    CURRENT.store(0, Ordering::Release);
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

        let tasks = TASKS.get();
        tasks[slot] = Task {
            rsp: new_rsp as u64,
            state: State::Ready,
            name,
            entry: Some(entry),
            switches: 0,
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
    crate::cpu::enable_interrupts();

    let entry = {
        let idx = CURRENT.load(Ordering::Acquire);
        unsafe { TASKS.get()[idx].entry }
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

pub fn current() -> usize {
    CURRENT.load(Ordering::Relaxed)
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
    Some(unsafe { TASKS.get()[index] })
}

/// Pick the next ready task and switch to it.
fn schedule() {
    let n = COUNT.load(Ordering::Acquire);
    if n < 2 {
        return;
    }

    let cur = CURRENT.load(Ordering::Acquire);
    let mut next = (cur + 1) % n;
    while next != cur {
        if unsafe { TASKS.get()[next].state } == State::Ready {
            break;
        }
        next = (next + 1) % n;
    }
    if next == cur {
        return;
    }

    SWITCHES.fetch_add(1, Ordering::Relaxed);

    unsafe {
        let tasks = TASKS.get();
        tasks[next].switches += 1;

        let out_fpu = tasks[cur].fpu;
        let in_fpu = tasks[next].fpu;
        let save = &mut tasks[cur].rsp as *mut u64;
        let load = tasks[next].rsp;

        // Save the outgoing task's extended state, load the incoming task's,
        // and only then switch stacks.
        //
        // The ordering is deliberate. The obvious alternative -- switch first,
        // restore our own state after `glados_switch_context` returns -- means
        // the restore runs when we are the *outgoing* task again, so it has to
        // use the index captured before the switch rather than CURRENT. That is
        // a silent wrong-task bug waiting for whoever later "simplifies" it.
        // Doing both halves before the switch leaves no code after it to get
        // wrong.
        //
        // Nothing between the xrstor and the switch may touch FP state.
        // `glados_switch_context` is pure integer assembly and the store below
        // is an atomic integer write.
        if !out_fpu.is_null() {
            crate::cpu::xsave_to(out_fpu);
        }
        if !in_fpu.is_null() {
            crate::cpu::xrstor_from(in_fpu);
        }

        CURRENT.store(next, Ordering::Release);
        glados_switch_context(save, load);
    }
    // Execution resumes here when someone switches back to `cur`. Our extended
    // state was already restored by whoever switched to us, so there is
    // deliberately nothing to do here.
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

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
    .globl sanctum_switch_context
sanctum_switch_context:
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
    fn sanctum_switch_context(save_rsp: *mut u64, new_rsp: u64);
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
}

const EMPTY: Task = Task {
    rsp: 0,
    state: State::Unused,
    name: "",
    entry: None,
    switches: 0,
};

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
    unsafe {
        let tasks = TASKS.get();
        tasks[0] = Task { rsp: 0, state: State::Ready, name, entry: None, switches: 0 };
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

    // Fabricate what `sanctum_switch_context` expects to pop, so the first
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

    CURRENT.store(next, Ordering::Release);
    SWITCHES.fetch_add(1, Ordering::Relaxed);

    unsafe {
        let tasks = TASKS.get();
        tasks[next].switches += 1;
        let save = &mut tasks[cur].rsp as *mut u64;
        let load = tasks[next].rsp;
        sanctum_switch_context(save, load);
    }
    // Execution resumes here when someone switches back to `cur`.
}

/// Called from the timer interrupt, after EOI.
pub fn tick() {
    if ENABLED.load(Ordering::Acquire) == 0 {
        return;
    }
    schedule();
}

/// Give up the rest of this quantum voluntarily.
pub fn yield_now() {
    if ENABLED.load(Ordering::Acquire) == 0 {
        return;
    }
    schedule();
}

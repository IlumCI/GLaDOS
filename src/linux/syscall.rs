//! Trapping `syscall` from code running at ring 0.
//!
//! ### Why this works at all, and what it costs
//!
//! `syscall` is normally the ring 3 to ring 0 door, and it is easy to assume
//! that is all it is. It is not: the instruction loads `rip` from `IA32_LSTAR`
//! and `cs` from `IA32_STAR[47:32]` **whatever the current privilege level**,
//! so a guest already at CPL 0 traps here exactly as one at CPL 3 would. That
//! is what makes stage 0 possible without building a userspace first.
//!
//! What it does *not* do is switch stacks. There is no `rsp0` reload, because
//! there is no privilege transition to trigger one -- and none of the three
//! things that normally make a syscall safe happen either:
//!
//!   - **The guest's `rsp` is whatever the guest left in it.** So the stub
//!     swaps to a stack of its own before touching anything, and a guest that
//!     wrecked its stack pointer still reaches the dispatcher.
//!   - **That stack is a single static, so the handler is not reentrant.**
//!     `IA32_FMASK` clears `IF` on entry, so nothing preempts it, and stage 0
//!     runs one guest with no threads. Both of those stop being true later,
//!     and this is where that will have to be paid for.
//!   - **A hostile guest is not contained by any of this.** At CPL 0 it can
//!     execute `wrmsr` and point `LSTAR` somewhere else, or `mov cr3`, or
//!     `cli`. That is the acknowledged shape of stage 0: it contains bugs,
//!     not malice, and a fault here is the measurement rather than a failure.
//!
//! ### Returning
//!
//! Not with `sysret`, which forces CPL 3 on the way out and would drop a guest
//! this kernel deliberately keeps at ring 0 into a privilege level it has no
//! descriptors for. `syscall` leaves the return address in `rcx` and the old
//! flags in `r11`, so the stub restores the flags and jumps to `rcx` -- which
//! is what `sysret` does minus the privilege change.

use super::elf;
use crate::sync::Racy;
use alloc::vec::Vec;

const IA32_EFER: u32 = 0xC000_0080;
const IA32_STAR: u32 = 0xC000_0081;
const IA32_LSTAR: u32 = 0xC000_0082;
const IA32_FMASK: u32 = 0xC000_0084;
/// `EFER.SCE`. Without it `syscall` is an invalid opcode.
const EFER_SCE: u64 = 1;

/// Linux's numbers, for the calls stage 0 answers or expects to meet first.
pub const SYS_WRITE: u64 = 1;
pub const SYS_BRK: u64 = 12;
pub const SYS_EXIT: u64 = 60;
pub const SYS_ARCH_PRCTL: u64 = 158;
pub const SYS_EXIT_GROUP: u64 = 231;

/// `-ENOSYS`, as Linux returns it: errors come back as small negatives in
/// `rax` rather than through any side channel.
const ENOSYS: u64 = (-38i64) as u64;
const EBADF: u64 = (-9i64) as u64;

/// What the guest had in its registers when it trapped.
///
/// Field order is the order the stub pushes them, so this is the stack frame
/// itself rather than a copy of it -- the dispatcher writes `rax` back here and
/// the stub pops it into the guest.
#[repr(C)]
pub struct Frame {
    pub rax: u64,
    pub rdi: u64,
    pub rsi: u64,
    pub rdx: u64,
    pub r10: u64,
    pub r8: u64,
    pub r9: u64,
    /// Where the guest resumes. `syscall` put it in `rcx`.
    pub rip: u64,
    /// The guest's flags. `syscall` put them in `r11`.
    pub rflags: u64,
}

// The stub, and every line of it is load-bearing.
//
// `mov rdi, rsp` is taken *before* the alignment `sub`, so the dispatcher gets
// the frame and not the padding. The `sub rsp, 8` is there because nine pushes
// from a 16-aligned top leaves rsp at 8 mod 16, and SysV wants it 16-aligned
// at the `call` -- getting that wrong does not fault, it misaligns every SSE
// spill the dispatcher makes, which on this machine surfaces as #GP inside
// unrelated Rust code.
core::arch::global_asm!(
    r#"
    .globl glados_syscall_entry
glados_syscall_entry:
    mov [rip + GLADOS_GUEST_RSP], rsp
    mov rsp, [rip + GLADOS_SYSCALL_STACK]
    push r11
    push rcx
    push r9
    push r8
    push r10
    push rdx
    push rsi
    push rdi
    push rax
    mov rdi, rsp
    sub rsp, 8
    call glados_syscall_dispatch
    add rsp, 8
    pop rax
    pop rdi
    pop rsi
    pop rdx
    pop r10
    pop r8
    pop r9
    pop rcx
    pop r11
    mov rsp, [rip + GLADOS_GUEST_RSP]
    push r11
    popfq
    jmp rcx

    .globl glados_enter_guest
glados_enter_guest:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov [rip + GLADOS_HOST_RSP], rsp
    mov rsp, rsi
    xor eax, eax
    xor ecx, ecx
    xor edx, edx
    xor ebx, ebx
    xor ebp, ebp
    xor r8d, r8d
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    xor r12d, r12d
    xor r13d, r13d
    xor r14d, r14d
    xor r15d, r15d
    mov rsi, rdi
    xor edi, edi
    jmp rsi

    .globl glados_leave_guest
glados_leave_guest:
    mov rsp, [rip + GLADOS_HOST_RSP]
    mov rax, rdi
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
    fn glados_syscall_entry();
    /// Jump into the guest. Answers the exit code, when the guest exits.
    fn glados_enter_guest(entry: u64, stack_top: u64) -> u64;
}

/// Where the guest's stack pointer went while the handler runs.
#[no_mangle]
static mut GLADOS_GUEST_RSP: u64 = 0;
/// The top of the handler's own stack.
#[no_mangle]
static mut GLADOS_SYSCALL_STACK: u64 = 0;
/// Where `glados_enter_guest` left the host, for `exit_group` to return to.
#[no_mangle]
static mut GLADOS_HOST_RSP: u64 = 0;

/// Sixteen KiB, aligned, and static rather than heap-allocated.
///
/// Static because the address has to be known to the stub, and 16 KiB because
/// the dispatcher below is ordinary Rust that formats and prints -- a stack
/// sized for the assembly rather than for what it calls is the classic way to
/// make a trap handler that works until somebody adds a `kprintln!`.
#[repr(align(16))]
struct Stack([u8; 16 * 1024]);
static mut SYSCALL_STACK: Stack = Stack([0; 16 * 1024]);

/// One recorded call. **The whole point of stage 0.**
#[derive(Clone, Copy)]
pub struct Call {
    pub nr: u64,
    pub args: [u64; 6],
    pub ret: u64,
    /// Whether this kernel actually answered, or recorded the question and
    /// returned `-ENOSYS`. Both are measurements and only one is a service.
    pub served: bool,
}

static TRACE: Racy<Vec<Call>> = Racy::new(Vec::new());
/// Bounded, because a guest in a loop on an unimplemented call would otherwise
/// grow this until the heap gave out -- and the first thousand entries are the
/// measurement anyway. What matters is which calls appear, not how often.
const TRACE_CAP: usize = 1024;

/// Arm the trap. Idempotent, and safe to call before any guest exists.
///
/// `STAR[47:32]` is the code selector `syscall` loads, and the data selector is
/// implicitly that plus eight -- which is exactly this kernel's `KERNEL_CS` and
/// `KERNEL_DS`, so the guest lands on the descriptors it was already running
/// under. That coincidence is not luck: it is what makes a same-ring trap cost
/// nothing to set up.
pub fn arm() {
    unsafe {
        let top = core::ptr::addr_of!(SYSCALL_STACK.0) as u64 + (16 * 1024);
        core::ptr::write(core::ptr::addr_of_mut!(GLADOS_SYSCALL_STACK), top);

        let star = (crate::cpu::gdt::KERNEL_CS as u64) << 32;
        crate::cpu::wrmsr(IA32_STAR, star);
        crate::cpu::wrmsr(IA32_LSTAR, glados_syscall_entry as usize as u64);
        // Clear IF, TF and DF on entry. IF because the handler runs on one
        // static stack and a timer tick landing inside it would re-enter that
        // stack; DF because Rust's memory intrinsics assume it is clear and a
        // guest is under no obligation to leave it that way.
        crate::cpu::wrmsr(IA32_FMASK, 0x700);
        let efer = crate::cpu::rdmsr(IA32_EFER);
        crate::cpu::wrmsr(IA32_EFER, efer | EFER_SCE);
    }
}

/// Whether the trap is armed, read back from the register rather than a flag.
pub fn armed() -> bool {
    unsafe {
        crate::cpu::rdmsr(IA32_EFER) & EFER_SCE != 0
            && crate::cpu::rdmsr(IA32_LSTAR) == glados_syscall_entry as usize as u64
    }
}

/// Everything the guest has asked for.
pub fn trace() -> Vec<Call> {
    unsafe { TRACE.get().clone() }
}

pub fn clear_trace() {
    unsafe { TRACE.get().clear() }
}

fn record(c: Call) {
    let t = unsafe { TRACE.get() };
    if t.len() < TRACE_CAP {
        t.push(c);
    }
}

/// The guest wrote something. Only the two standard descriptors, and every
/// other fd is `-EBADF` rather than silently accepted -- a write to fd 7 that
/// reported success would be a program whose output vanished.
fn sys_write(fd: u64, buf: u64, len: usize) -> u64 {
    if fd != 1 && fd != 2 {
        return EBADF;
    }
    let bytes = unsafe { core::slice::from_raw_parts(buf as *const u8, len) };
    for chunk in core::str::from_utf8(bytes) {
        crate::kprint!("{}", chunk);
    }
    len as u64
}

/// Called from the stub. Not public API: the only caller is three lines of
/// assembly above, and the `sysv64` pinning is why `rdi` is the frame.
#[no_mangle]
pub extern "sysv64" fn glados_syscall_dispatch(f: &mut Frame) {
    let nr = f.rax;
    let args = [f.rdi, f.rsi, f.rdx, f.r10, f.r8, f.r9];

    let (ret, served) = match nr {
        SYS_WRITE => (sys_write(f.rdi, f.rsi, f.rdx as usize), true),
        SYS_EXIT | SYS_EXIT_GROUP => {
            record(Call { nr, args, ret: 0, served: true });
            // Does not return. The host's stack and callee-saved registers
            // were parked by `glados_enter_guest`, so this is a longjmp back
            // into whoever started the guest -- there is no unwinder here and
            // returning normally from a process that has exited is not a
            // thing that can be expressed.
            unsafe { glados_leave_guest(f.rdi | EXITED) };
        }
        // Everything else is recorded and refused. That *is* the instrument:
        // a run that ends in `-ENOSYS` on call 47 has told us which call to
        // implement next, which is the question stage 0 exists to answer.
        _ => (ENOSYS, false),
    };
    record(Call { nr, args, ret, served });
    f.rax = ret;
}

extern "sysv64" {
    fn glados_leave_guest(code: u64) -> !;
}

/// Set on the way out so an exit code of zero is still distinguishable from
/// "the guest never called exit at all".
pub const EXITED: u64 = 1 << 32;

/// Run a loaded image until it exits.
///
/// # Safety
/// Jumps to an address derived from a file. Everything in the module docs
/// about stage 0 containing bugs rather than malice applies here and nowhere
/// more directly.
pub unsafe fn run(entry: u64, stack_top: u64) -> u64 {
    arm();
    unsafe { glados_enter_guest(entry, stack_top) }
}

/// What `diag linux` asks of the trap, without taking one.
///
/// Every claim here is about arithmetic or about a register, because the trap
/// itself cannot be exercised without a guest and a guest cannot be built at
/// boot. What is checked is the part that is silently wrong when it is wrong.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();

    // The frame is the stub's own stack layout. A field reordered here and not
    // in the assembly would hand the dispatcher `rsi` where it expects `rdi`
    // and produce a syscall trace that is subtly and consistently wrong.
    out.push((
        "the trap frame is nine words, in the order the stub pushes them",
        core::mem::size_of::<Frame>() == 72,
    ));
    let f = Frame { rax: 0, rdi: 1, rsi: 2, rdx: 3, r10: 4, r8: 5, r9: 6, rip: 7, rflags: 8 };
    let words = unsafe { core::slice::from_raw_parts(&f as *const Frame as *const u64, 9) };
    out.push((
        "the frame reads back as the register file it describes",
        words == [0, 1, 2, 3, 4, 5, 6, 7, 8],
    ));

    // The selector arithmetic. `syscall` derives SS from CS by adding eight,
    // so a STAR built from the wrong field loads a data selector that is not
    // this kernel's -- and the fault would land on the first push, inside the
    // stub, with no diagnostic that names the cause.
    let star = (crate::cpu::gdt::KERNEL_CS as u64) << 32;
    out.push((
        "STAR names this kernel's code selector, and its data selector follows",
        (star >> 32) as u16 == crate::cpu::gdt::KERNEL_CS
            && (star >> 32) as u16 + 8 == crate::cpu::gdt::KERNEL_DS,
    ));

    // Errors are small negatives in rax, which is the whole of Linux's error
    // convention. A positive ENOSYS would read to a guest as a successful call
    // that returned 38.
    out.push((
        "an unimplemented call answers a negative errno, not a plausible length",
        (ENOSYS as i64) < 0 && (EBADF as i64) < 0 && ENOSYS as i64 == -38 && EBADF as i64 == -9,
    ));

    // The exit marker has to live above anything an exit code can reach, or a
    // program exiting with 1 would be indistinguishable from one that never
    // exited.
    out.push((
        "the exit marker is clear of every exit code a guest can return",
        EXITED > u32::MAX as u64 && (255u64 | EXITED) != 255,
    ));

    // The handler's stack has to be aligned, because the stub's `sub rsp, 8`
    // assumes a 16-aligned top and corrects for exactly nine pushes.
    let top = unsafe { core::ptr::addr_of!(SYSCALL_STACK.0) as u64 + (16 * 1024) };
    out.push(("the handler's stack top is 16-byte aligned", top % 16 == 0));
    out.push((
        "nine pushes and the alignment slot leave rsp 16-aligned at the call",
        (top - 72 - 8) % 16 == 0,
    ));

    out
}

/// The initial stack Linux hands a process, as much of it as matters here.
///
/// `rsp` points at `argc`, then the argv pointers, a NULL, the envp pointers,
/// a NULL, and the auxiliary vector terminated by `AT_NULL`. A static binary
/// that never reads its arguments does not care -- and building it anyway
/// costs six words and means the first program that *does* read them finds
/// something shaped correctly rather than a fault.
pub fn build_stack(top: *mut u8, size: usize) -> u64 {
    let words = size / 8;
    let base = top as *mut u64;
    // Leave a gap below the very top: some prologues read a little above rsp.
    let at = words.saturating_sub(8);
    unsafe {
        let p = base.add(at);
        p.write(0); // argc
        p.add(1).write(0); // argv terminator
        p.add(2).write(0); // envp terminator
        p.add(3).write(0); // AT_NULL
        p.add(4).write(0);
        p as u64
    }
}

/// Name the calls stage 0 knows about, for a trace a person has to read.
pub fn name_of(nr: u64) -> &'static str {
    match nr {
        SYS_WRITE => "write",
        SYS_BRK => "brk",
        9 => "mmap",
        10 => "mprotect",
        11 => "munmap",
        SYS_EXIT => "exit",
        SYS_ARCH_PRCTL => "arch_prctl",
        218 => "set_tid_address",
        SYS_EXIT_GROUP => "exit_group",
        231.. => "?",
        _ => "?",
    }
}

/// Whether an image is one stage 0 can actually run, and why not when it is
/// not. Split out from the loader so the refusal is testable without a heap.
pub fn runnable(img: &elf::Image) -> Result<(), &'static str> {
    if img.dynamic() {
        return Err("dynamically linked: the entry point is ld.so, which stage 0 has no loader for");
    }
    if !img.relocatable() {
        return Err("fixed addresses: a single address space has no free range to promise it");
    }
    if img.segments.is_empty() {
        return Err("nothing to load");
    }
    Ok(())
}

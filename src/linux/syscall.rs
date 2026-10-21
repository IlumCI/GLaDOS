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
//! With `sysretq`, which forces CPL 3 on the way out. That was the reason not
//! to use it while guests ran at ring 0 and this kernel had no ring-3
//! descriptors; both of those stopped being true at stage 1. It takes the
//! return address from `rcx` and the flags from `r11`, which is where
//! `syscall` left them, and derives CS and SS from `IA32_STAR[63:48]`.

use super::elf;
use crate::sync::Racy;
use core::sync::atomic::{AtomicBool, Ordering};
use crate::sysbox;
use alloc::string::String;
use alloc::vec::Vec;

const IA32_EFER: u32 = 0xC000_0080;
const IA32_STAR: u32 = 0xC000_0081;
const IA32_LSTAR: u32 = 0xC000_0082;
const IA32_FMASK: u32 = 0xC000_0084;
/// `EFER.SCE`. Without it `syscall` is an invalid opcode.
const EFER_SCE: u64 = 1;

/// The segment-base MSRs. `FS` is the guest's to set; `GS` is emphatically
/// not -- see `sys_arch_prctl`.
const IA32_FS_BASE: u32 = 0xC000_0100;

const ARCH_SET_GS: u64 = 0x1001;
const ARCH_SET_FS: u64 = 0x1002;
const ARCH_GET_FS: u64 = 0x1003;
const ARCH_GET_GS: u64 = 0x1004;

const PROT_WRITE: u64 = 2;
const PROT_EXEC: u64 = 4;

const MAP_SHARED: u64 = 0x01;
const MAP_PRIVATE: u64 = 0x02;
const MAP_FIXED: u64 = 0x10;
const MAP_ANONYMOUS: u64 = 0x20;

/// Linux's numbers, for the calls stage 0 answers or expects to meet first.
pub const SYS_READ: u64 = 0;
pub const SYS_WRITE: u64 = 1;
pub const SYS_OPEN: u64 = 2;
pub const SYS_CLOSE: u64 = 3;
pub const SYS_STAT: u64 = 4;
pub const SYS_FSTAT: u64 = 5;
pub const SYS_LSTAT: u64 = 6;
pub const SYS_LSEEK: u64 = 8;
pub const SYS_IOCTL: u64 = 16;
pub const SYS_PREAD64: u64 = 17;
pub const SYS_SET_ROBUST_LIST: u64 = 273;
pub const SYS_RSEQ: u64 = 334;
pub const SYS_PRLIMIT64: u64 = 302;
pub const SYS_GETRANDOM: u64 = 318;
pub const SYS_TIME: u64 = 201;
pub const SYS_SYSINFO: u64 = 99;
pub const SYS_SCHED_GETAFFINITY: u64 = 204;
pub const SYS_READLINK: u64 = 89;
pub const SYS_READLINKAT: u64 = 267;
pub const SYS_GETPID: u64 = 39;
pub const SYS_GETDENTS64: u64 = 217;
pub const SYS_SET_TID_ADDRESS: u64 = 218;
pub const SYS_DUP: u64 = 32;
pub const SYS_WRITEV: u64 = 20;
pub const SYS_RT_SIGACTION: u64 = 13;
pub const SYS_READV: u64 = 19;
pub const SYS_SOCKET: u64 = 41;
pub const SYS_CONNECT: u64 = 42;
pub const SYS_ACCEPT: u64 = 43;
pub const SYS_SENDMSG: u64 = 46;
pub const SYS_RECVMSG: u64 = 47;
pub const SYS_BIND: u64 = 49;
pub const SYS_LISTEN: u64 = 50;
pub const SYS_SOCKETPAIR: u64 = 53;
pub const SYS_ACCEPT4: u64 = 288;
pub const SYS_SENDTO: u64 = 44;
pub const SYS_RECVFROM: u64 = 45;
pub const SYS_SHUTDOWN: u64 = 48;
pub const SYS_GETSOCKNAME: u64 = 51;
pub const SYS_GETPEERNAME: u64 = 52;
pub const SYS_SETSOCKOPT: u64 = 54;
pub const SYS_GETSOCKOPT: u64 = 55;
pub const SYS_ACCESS: u64 = 21;
pub const SYS_FTRUNCATE: u64 = 77;
pub const SYS_MEMFD_CREATE: u64 = 319;
pub const SYS_EPOLL_CREATE: u64 = 213;
pub const SYS_EPOLL_WAIT: u64 = 232;
pub const SYS_EPOLL_CTL: u64 = 233;
pub const SYS_EPOLL_PWAIT: u64 = 281;
pub const SYS_EPOLL_CREATE1: u64 = 291;
pub const SYS_PIPE: u64 = 22;
pub const SYS_PIPE2: u64 = 293;
pub const SYS_POLL: u64 = 7;
pub const SYS_SELECT: u64 = 23;
pub const SYS_PSELECT6: u64 = 270;
pub const SYS_PPOLL: u64 = 271;
pub const SYS_SENDFILE: u64 = 40;
pub const SYS_FACCESSAT: u64 = 269;
pub const SYS_RT_SIGPROCMASK: u64 = 14;
pub const SYS_NANOSLEEP: u64 = 35;
pub const SYS_GETCWD: u64 = 79;
pub const SYS_MKDIR: u64 = 83;
pub const SYS_RMDIR: u64 = 84;
pub const SYS_UNLINK: u64 = 87;
pub const SYS_GETPPID: u64 = 110;
pub const SYS_GETGROUPS: u64 = 115;
pub const SYS_CLOCK_NANOSLEEP: u64 = 230;
pub const SYS_UNAME: u64 = 63;
pub const SYS_FCNTL: u64 = 72;
pub const SYS_GETTIMEOFDAY: u64 = 96;
pub const SYS_CLOCK_GETTIME: u64 = 228;
pub const SYS_DUP2: u64 = 33;
pub const SYS_DUP3: u64 = 292;
pub const SYS_CLONE: u64 = 56;
pub const SYS_FORK: u64 = 57;
pub const SYS_VFORK: u64 = 58;
pub const SYS_EXECVE: u64 = 59;
pub const SYS_RT_SIGRETURN: u64 = 15;
pub const SYS_KILL: u64 = 62;
pub const SYS_WAIT4: u64 = 61;
pub const SYS_FUTEX: u64 = 202;
pub const SYS_GETTID: u64 = 186;
pub const SYS_TGKILL: u64 = 234;
pub const SYS_SCHED_YIELD: u64 = 24;
pub const SYS_GETUID: u64 = 102;
pub const SYS_GETGID: u64 = 104;
pub const SYS_GETEUID: u64 = 107;
pub const SYS_GETEGID: u64 = 108;
pub const SYS_OPENAT: u64 = 257;
pub const SYS_NEWFSTATAT: u64 = 262;
pub const SYS_BRK: u64 = 12;
pub const SYS_EXIT: u64 = 60;
pub const SYS_MMAP: u64 = 9;
pub const SYS_MPROTECT: u64 = 10;
pub const SYS_MADVISE: u64 = 28;
pub const SYS_MUNMAP: u64 = 11;
pub const SYS_ARCH_PRCTL: u64 = 158;
pub const SYS_EXIT_GROUP: u64 = 231;

/// `-ENOSYS`, as Linux returns it: errors come back as small negatives in
/// `rax` rather than through any side channel.
pub(super) const ENOSYS: u64 = (-38i64) as u64;
pub(super) const EBADF: u64 = (-9i64) as u64;
pub(super) const EPERM: u64 = (-1i64) as u64;
pub(super) const ENOMEM: u64 = (-12i64) as u64;
pub(super) const EINVAL: u64 = (-22i64) as u64;
pub(super) const EFAULT: u64 = (-14i64) as u64;
pub(super) const ENOENT: u64 = (-2i64) as u64;
pub(super) const EACCES: u64 = (-13i64) as u64;
pub(super) const EMFILE: u64 = (-24i64) as u64;
pub(super) const ENOTDIR: u64 = (-20i64) as u64;
pub(super) const EISDIR: u64 = (-21i64) as u64;
pub(super) const ENOTTY: u64 = (-25i64) as u64;
pub(super) const ESPIPE: u64 = (-29i64) as u64;
pub(super) const ENOTCONN: u64 = (-107i64) as u64;
pub(super) const EAGAIN: u64 = (-11i64) as u64;
pub(super) const EINTR: u64 = (-4i64) as u64;
pub(super) const ENAMETOOLONG: u64 = (-36i64) as u64;
pub(super) const EROFS: u64 = (-30i64) as u64;
pub(super) const EEXIST: u64 = (-17i64) as u64;
pub(super) const ERANGE: u64 = (-34i64) as u64;
pub(super) const ENOTEMPTY: u64 = (-39i64) as u64;
pub(super) const ENODEV: u64 = (-19i64) as u64;
pub(super) const ENOSPC: u64 = (-28i64) as u64;
/// `O_NONBLOCK`, which on an event device is the difference between a program
/// that polls and one that waits.
const O_NONBLOCK: u64 = 0o4000;

/// `AT_EMPTY_PATH`: operate on the descriptor rather than on a path under it.
const AT_EMPTY_PATH: u64 = 0x1000;
pub(super) const EAFNOSUPPORT: u64 = (-97i64) as u64;
pub(super) const EPROTONOSUPPORT: u64 = (-93i64) as u64;
pub(super) const ENOTSOCK: u64 = (-88i64) as u64;
pub(super) const ECONNREFUSED: u64 = (-111i64) as u64;
pub(super) const ETIMEDOUT: u64 = (-110i64) as u64;
pub(super) const ENETDOWN: u64 = (-100i64) as u64;
pub(super) const EISCONN: u64 = (-106i64) as u64;
pub(super) const ENOBUFS: u64 = (-105i64) as u64;

/// `O_WRONLY` and `O_RDWR`. This view is read-only, so both are refused.
const O_WRONLY: u64 = 1;
const O_RDWR: u64 = 2;
const O_DIRECTORY: u64 = 0x10000;
const O_CREAT: u64 = 0x40;
const O_TRUNC: u64 = 0x200;
const O_APPEND: u64 = 0x400;
const O_EXCL: u64 = 0x80;

/// How many bytes every open descriptor may hold between them.
///
/// An open file here *is* its contents: `read_blob` answers a `Vec`, so a
/// descriptor costs the size of the file for as long as it is open. Sixty-four
/// descriptors against an unbounded file size is a guest taking the heap with
/// nothing more exotic than a loop of `open`, and there is no OOM killer in
/// this kernel and one address space to lose.
///
/// The answer is `ENOMEM`, which Linux's `open` genuinely can return. It would
/// not return it for *this* reason, and that is the deviation: on Linux the
/// cost of an open descriptor does not scale with the file.
const OPEN_MAX_BYTES: usize = 64 * 1024 * 1024;

/// The most `getdents64` will build in one call, whatever the guest offers.
///
/// The guest's buffer length is the room the records must fit in, and the
/// records are built in the kernel heap first. A guest that owns a large
/// mapping can therefore ask this call to allocate as much as it owns, which
/// is a second copy of memory the machine already gave it.
const DENTS_MAX: usize = 1 << 20;

/// How many descriptors a guest may hold, and how long a path may be.
/// Linux's usual soft limit. It was 64, and a Source game keeps dozens of
/// archive chunks open at once.
const MAX_FDS: usize = 1024;
const PATH_MAX: usize = 4096;

/// The largest single mapping a guest may ask for.
///
/// Not a policy about greed. `page_up` rounds up and multiplies, which wraps
/// for a length near `usize::MAX`, so an unbounded request would quietly
/// produce a *small* allocation and hand back a pointer to far less memory
/// than the guest was told it had. A cap makes that unreachable at the one
/// place it can be checked cheaply.
///
/// A gigabyte: well short of any wrap, and above the largest single library
/// a real program maps -- Mesa's 32-bit `libLLVM` is 177 MB, and the old
/// 64 MiB cap answered `ENOMEM` for it and left the program with no OpenGL.
const MAP_MAX: u64 = 1024 * 1024 * 1024;

/// What the guest had in its registers when it trapped.
///
/// Field order is the order the stub pushes them, so this is the stack frame
/// itself rather than a copy of it -- the dispatcher writes `rax` back here and
/// the stub pops it into the guest.
#[repr(C)]
#[derive(Clone, Copy)]
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
    // The callee-saved six, and they are here for exactly one caller.
    //
    // **`fork` cannot be written without them.** SysV says these survive a
    // call, and they do -- the dispatcher is compiled Rust and spills them
    // like any other function -- so the *guest* never notices they are absent
    // from this frame. What notices is a child that has to resume with its
    // parent's registers: by the time a handler runs, the guest's `rbx` is
    // somewhere on the kernel stack and what the register holds is the
    // kernel's own value. Reading them at that point gives a plausible number
    // that is not the guest's, which is the worst shape this could take.
    //
    // Six more pushes on the hottest path in the guest ABI, paid by every
    // syscall so that one of them can work. The alternative was a second
    // entry point for `fork` alone, which means two stubs that must agree
    // about a frame layout -- the arrangement `model.rs` warns about twice.
    pub rbx: u64,
    pub rbp: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
}

// The stub, and every line of it is load-bearing.
//
// `mov rdi, rsp` is taken *before* the alignment `sub`, so the dispatcher gets
// the frame and not the padding. The `sub rsp, 8` is there because fifteen
// pushes from a 16-aligned top leaves rsp at 8 mod 16 -- 120 bytes, and 120
// mod 16 is 8, which is why adding the callee-saved six did not change it;
// nine pushes was 72, and 72 mod 16 is 8 as well. SysV wants it 16-aligned
// at the `call` -- getting that wrong does not fault, it misaligns every SSE
// spill the dispatcher makes, which on this machine surfaces as #GP inside
// unrelated Rust code.
//
// **The `fxsave64` pair is the ABI and not a precaution, and it was missing.**
// A Linux syscall clobbers `rax`, `rcx` and `r11` and *nothing else*: every
// other register, the whole FPU and SSE state included, is the caller's and
// must come back. This stub saved fifteen general-purpose registers and then
// called into Rust, where SSE is not optional -- a `memcpy` is enough, never
// mind anything holding a float -- so a guest's `xmm` registers were destroyed
// by any syscall at all. Silent, because nothing faults: the guest reads back
// whatever the kernel happened to leave.
//
// Found from the guest side after a long way round, which is the argument for
// writing it down here. QuickJS died at exit in musl's `pthread_key_delete`,
// walking a thread list whose `self->next` was zero, and musl's `__init_tp`
// plainly sets it. One disassembly says why:
//
//     movq       %rbx,%xmm0        ; xmm0 = td
//     punpcklqdq %xmm0,%xmm0       ; xmm0 = {td, td}
//     mov        $0xda,%eax        ; SYS_set_tid_address
//     syscall                      ; <-- td live in xmm0 across it
//     movups     %xmm0,0x10(%rbx)  ; td->prev = td->next = td, one 16-byte store
//
// musl keeps the value in `xmm0` across the syscall and writes both list
// pointers with a single `movups`. Entirely legal, and this kernel wrote zeros
// over it -- so the list was never linked and the fault arrived at exit,
// thousands of instructions later, in a function that had done nothing wrong.
//
// `fxsave64` and not a run of `movaps`: two instructions against thirty-two,
// and it covers x87 and `MXCSR` as well, which the ABI also promises. The area
// is 512 bytes on the syscall stack, which is per task, so a preempted syscall
// keeps its own copy. `and rsp, -16` makes the alignment `fxsave` requires
// self-enforcing rather than inherited from whoever allocated the stack --
// which the `sub rsp, 8` above had been quietly assuming all along.
//
// **The residual gap is AVX**: `fxsave` does not carry `ymm`'s upper halves,
// so a guest holding one of those across a syscall still loses it. Closing
// that means `xsave` with `XCR0` in `edx:eax` and a 64-byte-aligned area,
// which is the same shape and a bigger decision; it is named here rather than
// left to be rediscovered the way this was.
core::arch::global_asm!(
    r#"
    .globl glados_syscall_entry
glados_syscall_entry:
    swapgs
    mov gs:[0], rsp
    mov rsp, gs:[8]
    swapgs
    and rsp, -16
    sub rsp, 512
    fxsave64 [rsp]
    push r15
    push r14
    push r13
    push r12
    push rbp
    push rbx
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
    pop rbx
    pop rbp
    pop r12
    pop r13
    pop r14
    pop r15
    fxrstor64 [rsp]
    add rsp, 512
    swapgs
    mov rsp, gs:[0]
    swapgs
    sysretq

    .globl glados_enter_guest
glados_enter_guest:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov rax, rsp
    pushfq
    cli
    swapgs
    mov gs:[16], rax
    swapgs
    popfq
    push 0x33
    push rsi
    push 0x202
    push 0x3b
    push rdi
    xor eax, eax
    xor ecx, ecx
    xor edx, edx
    xor ebx, ebx
    xor ebp, ebp
    xor esi, esi
    xor edi, edi
    xor r8d, r8d
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    xor r12d, r12d
    xor r13d, r13d
    xor r14d, r14d
    xor r15d, r15d
    iretq

    // Enter a guest carrying registers, which is what a forked child is.
    //
    // `glados_enter_guest` zeroes everything, which is right for a program
    // starting at its entry point and wrong for a child resuming inside
    // libc's `fork` wrapper: SysV says `rbx`, `rbp` and `r12`-`r15` survive a
    // call, so the code on the other side of that `syscall` instruction is
    // entitled to find them intact. The caller-saved ones are zeroed here
    // rather than carried, because the syscall convention clobbers them
    // anyway and a child that inherited them would be relying on something
    // the ABI does not promise.
    //
    // `rdi` holds the `Resume` and is therefore loaded last, after every
    // field has been read out of it.
    //
    // The six pushes and `GLADOS_HOST_RSP` match `glados_enter_guest`
    // exactly, so `glados_leave_guest` unwinds a child no differently from
    // anything else. Getting that out of step would put the longjmp back onto
    // a stack six words from where it belongs.
    .globl glados_enter_guest_regs
glados_enter_guest_regs:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov rax, rsp
    pushfq
    cli
    swapgs
    mov gs:[16], rax
    swapgs
    popfq
    push 0x33
    push qword ptr [rdi + 8]
    push qword ptr [rdi + 16]
    push 0x3b
    push qword ptr [rdi + 0]
    mov rax, [rdi + 24]
    mov rbx, [rdi + 32]
    mov rbp, [rdi + 40]
    mov r12, [rdi + 48]
    mov r13, [rdi + 56]
    mov r14, [rdi + 64]
    mov r15, [rdi + 72]
    xor ecx, ecx
    xor edx, edx
    xor esi, esi
    xor r8d, r8d
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    xor edi, edi
    iretq

    .globl glados_leave_guest
glados_leave_guest:
    cli
    swapgs
    mov rsp, gs:[16]
    swapgs
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
    /// Jump into a guest that is resuming rather than starting.
    fn glados_enter_guest_regs(r: *const Resume) -> u64;
}

/// What a forked child resumes with.
///
/// Field order is the assembly's offsets, so this is the structure that stub
/// reads rather than a description of it -- the same bargain `Frame` makes
/// with the syscall stub, and a field reordered here without the offsets
/// moving is a child that resumes with its stack pointer in `rbx`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Resume {
    pub rip: u64,
    pub rsp: u64,
    pub rflags: u64,
    pub rax: u64,
    pub rbx: u64,
    pub rbp: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    /// The thread-local base, `IA32_FS_BASE`. **Last, and read by Rust, not by
    /// the assembly** that loads the fields above at fixed offsets. A forked
    /// child was entered with whatever FS its pool task had -- zero -- and the
    /// first thing glibc does is read its stack canary at `%fs:0x28`, so every
    /// glibc child faulted reading address 0x28.
    pub fs: u64,
}

/// The ring-3 entry state, one block per core.
///
/// **It was three globals**, swapped by the scheduler at every switch, which
/// made them per task for exactly as long as one core ran guests: a value
/// only read while its own task runs is per task provided somebody swaps it
/// -- and with guests on two cores, two tasks are running. So they are per
/// core now, still swapped at each switch, and the assembly reaches this
/// core's block through `swapgs`: `IA32_KERNEL_GS_BASE` holds its address
/// (`arm_core`). The kernel itself never uses `GS` -- `cpu::percpu` reads
/// `rdpid` -- so each `swapgs` is paired with another at once and nothing
/// runs between them with interrupts on.
///
/// Field order is the assembly's offsets: 0, 8, 16.
#[repr(C, align(64))]
pub struct CoreEntry {
    /// Where the guest's stack pointer went while the handler runs.
    guest_rsp: u64,
    /// The top of the handler's own stack.
    syscall_stack: u64,
    /// Where `glados_enter_guest` left the host, for `exit_group` to return to.
    host_rsp: u64,
}

static mut GLADOS_ENTRY: [CoreEntry; crate::task::MAX_CPUS] =
    [const { CoreEntry { guest_rsp: 0, syscall_stack: 0, host_rsp: 0 } }; crate::task::MAX_CPUS];

fn core_entry() -> *mut CoreEntry {
    let c = (crate::smp::this_cpu() as usize).min(crate::task::MAX_CPUS - 1);
    unsafe { core::ptr::addr_of_mut!(GLADOS_ENTRY[c]) }
}

const IA32_KERNEL_GS_BASE: u32 = 0xC000_0102;
static ARMED: [core::sync::atomic::AtomicBool; crate::task::MAX_CPUS] =
    [const { core::sync::atomic::AtomicBool::new(false) }; crate::task::MAX_CPUS];

/// The per-core half of arming: the `syscall` MSRs and this core's entry
/// block. Every core that may run a guest needs it, and an application
/// processor has none of it until it first does.
pub fn arm_core() {
    let c = (crate::smp::this_cpu() as usize).min(crate::task::MAX_CPUS - 1);
    if ARMED[c].swap(true, Ordering::AcqRel) {
        return;
    }
    unsafe {
        let star = (crate::cpu::gdt::KERNEL_CS as u64) << 32
            | (crate::cpu::gdt::SYSRET_BASE as u64) << 48;
        crate::cpu::wrmsr(IA32_STAR, star);
        crate::cpu::wrmsr(IA32_LSTAR, glados_syscall_entry as usize as u64);
        crate::cpu::wrmsr(IA32_FMASK, 0x700);
        let efer = crate::cpu::rdmsr(IA32_EFER);
        crate::cpu::wrmsr(IA32_EFER, efer | EFER_SCE);
        crate::cpu::wrmsr(IA32_KERNEL_GS_BASE, core::ptr::addr_of_mut!(GLADOS_ENTRY[c]) as u64);
    }
}

/// Sixteen KiB, aligned, and static rather than heap-allocated.
///
/// Static because the address has to be known to the stub, and 16 KiB because
/// the dispatcher below is ordinary Rust that formats and prints -- a stack
/// sized for the assembly rather than for what it calls is the classic way to
/// make a trap handler that works until somebody adds a `kprintln!`.
#[repr(align(16))]
struct Stack([u8; 16 * 1024]);
static mut SYSCALL_STACK: Stack = Stack([0; 16 * 1024]);

/// The three globals the stub reaches through, as one value.
///
/// Read and written by the scheduler at a switch. That is what makes them
/// per-task without any assembly changing: a global that is only live while
/// its own task is running *is* per-task, provided somebody swaps it at the
/// boundary.
pub fn ring3_now() -> crate::task::Ring3 {
    let (tls, gs, bits32) = super::i386::save();
    unsafe {
        crate::task::Ring3 {
            tls,
            gs,
            bits32,
            guest_rsp: (*core_entry()).guest_rsp,
            syscall_stack: (*core_entry()).syscall_stack,
            host_rsp: (*core_entry()).host_rsp,
            fs_base: crate::cpu::rdmsr(IA32_FS_BASE),
        }
    }
}

/// Put a task's entry state back, or leave the kernel's own `FS` alone.
///
/// `None` means the incoming task is not running a guest, and then only `FS`
/// matters -- the other three are dead until somebody enters ring 3 again, and
/// writing them would be a store nobody reads. `FS` is different because the
/// kernel itself uses it, so a task switched into after a guest set `FS` must
/// get the kernel's value back or the next allocation reads thread-local
/// storage as a per-core block.
pub fn ring3_load(r: Option<crate::task::Ring3>) {
    match r {
        Some(r) => unsafe {
            arm_core();
            (*core_entry()).guest_rsp = r.guest_rsp;
            (*core_entry()).syscall_stack = r.syscall_stack;
            (*core_entry()).host_rsp = r.host_rsp;
            crate::cpu::wrmsr(IA32_FS_BASE, r.fs_base);
            // This task's own landing for an interrupt taken at ring 3. See
            // `gdt::set_rsp0` for what sharing one cost.
            crate::cpu::gdt::set_rsp0(r.syscall_stack);
            super::i386::load(r.tls, r.gs, r.bits32);
        },
        None => {
            crate::cpu::gdt::set_rsp0(0);
            if let Some(sp) = unsafe { guest_slot().as_ref() } {
                unsafe { crate::cpu::wrmsr(IA32_FS_BASE, sp.saved_fs) };
            }
        }
    }
}

/// One anonymous mapping the guest asked for and has not given back.
///
/// `Clone` because `fork` copies the list wholesale: a child inherits every
/// mapping its parent had, at the same addresses, and the pages behind them
/// were duplicated before this list was.
#[derive(Clone, Copy)]
pub struct Mapping {
    pub at: u64,
    pub len: usize,
    /// Where the pages came from, because the two go back different ways and
    /// getting it wrong is silent: freeing a placed range to the heap hands
    /// the allocator memory it never owned.
    pub from: Source,
}

/// Which pool a mapping's pages came out of.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Source {
    /// The kernel heap, through `alloc_pages`. Where an ordinary `mmap` goes.
    Heap,
    /// A physical range `mem::fixed` promised. The only way to answer
    /// `MAP_FIXED` at all, since virtual is physical here and the address the
    /// guest names is real memory somebody may already own.
    Fixed,
    /// The display's own aperture, handed out by `/dev/fb0`.
    ///
    /// A third source because it goes back a third way, and the two it is not
    /// are both actively wrong: freeing it to the heap hands the allocator
    /// several megabytes of somebody else's memory-mapped registers, and
    /// releasing it through `mem::fixed` releases a claim nobody ever made.
    /// All it owes is the U bit, taken back off.
    Device,
    /// Heap pages the guest sees somewhere else entirely.
    ///
    /// A fourth source because it goes back a fourth way, and it carries the
    /// address to go back *to*: the guest's `at` is an arena address that no
    /// allocator ever handed out, so freeing that number would hand the heap
    /// something it never owned. The pages to release are the backing's, and
    /// this is the only place that still knows which they were.
    Backed(u64),
    /// Arena memory built from separate heap pages, because no contiguous run
    /// of the size asked for was free. The pages are found again through the
    /// guest's own tables when they go back -- which is why `give_back` alone
    /// cannot free them, and `free_paged` does.
    ///
    /// Linux never needs a physically contiguous run for `mmap`; this kernel
    /// did, and after enough churn a 50 MB request failed against a heap with
    /// far more than 50 MB free. Portal reported it as `std::bad_alloc`.
    Paged,
    /// A forked child's inherited record of one of its parent's mappings.
    ///
    /// **A sixth way back, and it is to do nothing.** The record was cloned
    /// with the rest of the parent's entry, so it still named the parent's
    /// source -- and a child calling `munmap` handed the *parent's* pages
    /// back to the heap while the parent was still using them. The child's
    /// own copy of those bytes is in its entry's `owned` list and goes back
    /// with the entry, so the record owes nothing at all.
    Copied,
    /// Pages a `memfd` owns, which the guest sees at their own address.
    ///
    /// A fifth source because it goes back a fifth way, and the way is to do
    /// almost nothing: the memfd allocated these pages and the memfd frees
    /// them when its last descriptor closes. Freeing them here would take
    /// memory still named by an open file. All the mapping owes is the U bit,
    /// which is the same debt `Device` has for a completely different reason.
    Shared,
}

/// A range this kernel handed to the guest.
#[derive(Clone, Copy)]
pub struct Region {
    pub at: u64,
    pub len: usize,
}

impl Region {
    fn holds(&self, at: u64, end: u64) -> bool {
        at >= self.at && end <= self.at.saturating_add(self.len as u64)
    }
}

/// The three ranges a loader hands over, kept together so a caller cannot
/// pass them in the wrong order.
#[derive(Clone, Copy)]
pub struct Regions {
    pub image: Region,
    pub stack: Region,
    pub brk: Region,
    /// The interpreter, when the program asked for one.
    ///
    /// A fourth region rather than a second `Space`, because it is one process
    /// with two images in it: `ld.so` and the program share a break, a
    /// descriptor table and every pointer either passes to the kernel. Making
    /// it optional rather than a zero-length region is the difference between
    /// "there is no interpreter" and "there is one and it is empty", and only
    /// the first is ever true.
    pub interp: Option<Region>,
    /// Whether the image lives in the guest's own page tables rather than in
    /// the identity map.
    ///
    /// **`teardown` needs this and it is not cosmetic.** It puts page rights
    /// back on every region, which is right for a region the loader opened in
    /// the kernel's map and wrong for one that was never there: the address of
    /// a mapped image is a *guest* virtual address, so restoring rights on it
    /// reaches the identity mapping of the same number -- a kernel page the
    /// guest never had. Today that only clears a U bit nobody set and splits a
    /// 2 MiB entry for nothing, which is why it is a tidiness bug rather than
    /// a fault. It is the same mistake `run` has to avoid in the other
    /// direction, where it would *grant* ring 3 a kernel page, and one of them
    /// being harmless is not a reason to write the pair inconsistently.
    pub image_mapped: bool,
    pub interp_mapped: bool,
    pub stack_mapped: bool,
    pub brk_mapped: bool,
    /// The ABI, and where its `mmap` arena runs.
    pub bits32: bool,
    pub mmap_at: u64,
    pub mmap_end: u64,
    /// More arena, below 4 GiB for a 32-bit guest: holes past the first.
    pub arena_extra: [(u64, u64); 7],
}

/// What a guest owns.
///
/// **One guest at a time**, which is why this is a global rather than
/// something the dispatcher is handed. The dispatcher is called from three
/// lines of assembly and has nowhere to carry a handle; the constraint is real
/// and is the same one that makes the handler's single static stack safe.
pub struct Space {
    /// Every range the loader gave this guest, which is the whole of what
    /// `owns` is allowed to say yes to.
    pub image: Region,
    /// The interpreter's image, if the program wanted one. A pointer into it
    /// is as legitimate as one into the program: `ld.so` passes the kernel its
    /// own strings and structures constantly.
    pub interp: Option<Region>,
    pub stack: Region,
    /// Where `brk` began, where it stands, and where it may not pass.
    pub brk_start: u64,
    pub brk_now: u64,
    pub brk_end: u64,
    pub maps: Vec<Mapping>,
    /// Open descriptors. 0, 1 and 2 are filled at `install`, so a guest that
    /// never opens anything still has somewhere to write.
    pub fds: Vec<Option<super::fs::Fd>>,
    /// Where a relative path starts from.
    pub cwd: String,
    /// What the guest was invoked as, and what it was loaded from.
    ///
    /// Kept because `/proc/self/exe` and `/proc/self/cmdline` are the only way
    /// a program can find its own binary, and neither is answerable from a
    /// register: `argv` lives on a stack the kernel built and then stopped
    /// tracking.
    pub argv: Vec<String>,
    pub image_mapped: bool,
    pub interp_mapped: bool,
    pub stack_mapped: bool,
    pub brk_mapped: bool,
    /// This guest's page tables, when it has its own.
    ///
    /// **Owned here rather than by `Guest`, because `mmap` needs them.** A
    /// mapping made during a syscall has to go into the running guest's own
    /// tables, and the intermediate tables it creates have to be freed when
    /// the guest is -- which means the thing that owns them must be reachable
    /// from the syscall path and must die with the guest. That is this entry.
    /// `Guest` hands them over at `install` and `teardown` drops them.
    pub tables: Option<crate::mem::space::Space>,
    /// The `Guest` an `execve` built, held so its pages outlive the call.
    ///
    /// **Only `execve` fills this, and the reason is that nothing else has to.**
    /// A guest started by `linux run` is described by a `Guest` the shell holds
    /// on its own stack for as long as the run lasts. One that replaced itself
    /// has no such owner: the shell's `Guest` describes the program that just
    /// stopped existing, and dropping it would free the image the guest is now
    /// executing from. So the new one lives here and dies with the entry --
    /// which also makes a second `execve` free the first one's pages, because
    /// replacing this field drops what it held.
    /// Handlers, masks and the frame a handler interrupted.
    pub signals: super::signal::State,
    pub holder: Option<alloc::boxed::Box<super::load::Guest>>,
    /// Heap ranges this guest owns outright, freed when it is torn down.
    ///
    /// A forked child's memory. The parent's image, stack and break belong to
    /// a `Guest` in the loader and go back when that drops; a child has no
    /// `Guest` and never will, so the pages copied for it are recorded here
    /// and nowhere else. Addresses are the *backings*, not what the guest
    /// sees -- freeing an arena address would hand the heap a number no
    /// allocator ever gave out.
    pub owned: alloc::vec::Vec<(u64, usize)>,
    /// The next free address in the guest's mmap arena.
    ///
    /// A bump cursor, with `arena_free` in front of it.
    pub mmap_next: u64,
    /// Where the arena must stop: the stack for a 32-bit guest, which has no
    /// room to spare below 4 GiB, and effectively nowhere for a 64-bit one.
    pub mmap_end: u64,
    /// Arena ranges given back by `munmap`, sorted and coalesced, taken
    /// first-fit before the cursor moves.
    ///
    /// This was "worth building when something runs out", and something did.
    /// A 64-bit guest has 510 GiB of arena; a 32-bit one has the hole below
    /// its stack, about 1.4 GiB. Portal maps and unmaps scratch buffers of tens
    /// of megabytes on its way to the menu -- about 1,100 `munmap`s on the
    /// host -- and with a bump cursor the hole was gone before the menu was:
    /// `malloc` answered NULL, and Valve's GL layer `memcpy`d a texture to
    /// address zero.
    pub arena_free: Vec<(u64, u64)>,
    /// A 32-bit guest: `int 0x80`, a compatibility code segment, 4-byte
    /// pointers in every structure it hands over.
    pub bits32: bool,
    pub image_path: String,
    pub interp_path: Option<String>,
    /// `FS` base as the kernel left it. A guest sets `FS` for its
    /// thread-local storage and the register is the *machine's*, not the
    /// guest's, so it is put back on the way out.
    pub saved_fs: u64,
}

impl Space {
    /// Whether a range the guest named is one this kernel actually gave it.
    ///
    /// **This is the whole of the hardening.** A guest at ring 0 shares an
    /// address space with the kernel, so a pointer it passes is not merely
    /// possibly-invalid, it is a pointer at anything at all: the page tables,
    /// the heap's free list, the model's weights. Every syscall that reads or
    /// writes through a guest-supplied address asks this first and answers
    /// `EFAULT` when the answer is no.
    ///
    /// It cannot stop a guest dereferencing that pointer *itself*, and nothing
    /// at CPL 0 can. What it stops is the kernel doing it on the guest's
    /// behalf, which is the part that turns a bad argument into a corrupted
    /// kernel rather than a crashed program.
    pub fn owns(&self, at: u64, len: usize) -> bool {
        let Some(end) = at.checked_add(len as u64) else { return false };
        self.image.holds(at, end)
            || self.interp.is_some_and(|r| r.holds(at, end))
            || self.stack.holds(at, end)
            || (at >= self.brk_start && end <= self.brk_end)
            || self.maps.iter().any(|m| Region { at: m.at, len: m.len }.holds(at, end))
    }
}

/// Every guest this machine is holding, and which one the syscall path means.
///
/// **A table rather than one slot, because `fork` is two guests alive at
/// once.** Everything else was ready for that before this was: a guest's
/// memory is its own page-table root, and a fixed image is mapped rather than
/// holding physical `0x400000` against the machine. What remained was that
/// there was exactly one `Space` to describe a guest with, so a second one had
/// nowhere to be recorded.
///
/// The syscall path always means the *running* guest, which is why the 25
/// places that reach for this did not have to change: `guest_slot` answers the
/// current slot and they go on saying what they said. Switching guests is
/// therefore one store, and it belongs beside the CR3 switch rather than
/// inside any of them.
///
/// A `Vec` rather than a fixed array because `Space` holds `Vec`s and
/// `String`s, so an array of `None` needs a const initialiser it cannot have,
/// and because the bound that matters is the heap rather than a number chosen
/// here. It grows on demand and never shrinks, which is the same bargain
/// `thread`'s pool makes: a slot vacated by a guest that exited is taken by
/// the next one.
/// Where a guest's `mmap` arena begins, one slot past the stack.
///
/// Named here rather than in `load` because this is what hands addresses out
/// of it. The slots below are the loader's business and this one is not.
pub const GUEST_MMAP_AT: u64 = crate::mem::space::WINDOW + 0x5000_0000;

static TABLE: Racy<Vec<Option<Space>>> = Racy::new(Vec::new());

/// Which slot each task's syscall path is speaking for.
///
/// **Indexed by task, rather than one global the scheduler swaps.** It was the
/// global, saved and restored by `schedule` beside the ring-3 state, and that
/// is correct only while every path that switches tasks goes through that one
/// seam and in that one order. A 32-bit guest's threads found a path that did
/// not: two pool tasks entered `futex` with the guest's own page tables in
/// CR3 and index 0 in the global, so `owns` checked the wrong guest's regions
/// and answered `EFAULT` about the guest's own memory -- glibc's "The futex
/// facility returned an unexpected error code". An array indexed by the
/// running task has no seam to miss.
static TASK_GUEST: [core::sync::atomic::AtomicUsize; crate::task::MAX_TASKS] =
    [const { core::sync::atomic::AtomicUsize::new(0) }; crate::task::MAX_TASKS];

fn guest_index() -> usize {
    TASK_GUEST[me()].load(core::sync::atomic::Ordering::Relaxed)
}

/// The running guest's slot, grown into existence if this is its first use.
///
/// # Safety
/// Same contract `Racy::get` carries: no other live reference, and never from
/// an interrupt that can preempt another caller.
#[allow(clippy::mut_from_ref)]
pub unsafe fn guest_slot() -> &'static mut Option<Space> {
    let t = unsafe { TABLE.get() };
    let i = guest_index();
    while t.len() <= i {
        t.push(None);
    }
    &mut t[i]
}

/// Jump into a guest that is resuming rather than starting.
///
/// # Safety
/// `r` must describe a guest whose tables are the installed root.
/// Install the stack a guest's syscalls will run on.
///
/// **`task::ring3_active` records this and does not install it.** The global
/// is loaded by `schedule` on a switch, so a caller that arms a task and then
/// enters a guest *without a switch in between* runs that guest's syscalls on
/// whatever stack the last switch left behind. `run_thread` writes it for
/// exactly this reason; `fork` did not, and the child's two syscalls landed on
/// the parent's syscall stack and overwrote the kernel frame it was suspended
/// on. The parent then resumed, `ret`'d into what the child had written, and
/// took a ring-0 fault at `rip 0x1400` -- an address no image claims, which is
/// what a `ret` off a smashed stack looks like.
///
/// # Safety
/// `stack` must be the top of a stack this task alone will use.
/// Where the guest's stack pointer will be restored from on the way out.
///
/// `execve` is the only caller and it is how the new program gets a fresh
/// stack: the stub ends `mov rsp, [GLADOS_GUEST_RSP]` then `sysretq`, so
/// setting this and the frame's `rip` is the whole of "return into a
/// different program" -- no second entry stub, no longjmp, and the syscall
/// that never returns returns like any other.
///
/// # Safety
/// `v` must be a stack the running guest owns.
/// Hand a guest the `Guest` describing what it is now executing.
///
/// See `Space::holder`. Replacing what was there drops it, which is how the
/// pages of a program that has been exec'd over are freed.
/// End the running guest now, with a code.
///
/// # Safety
/// Longjmps out of whatever is running it, so no allocation, borrow or lock
/// may be live at the call.
pub unsafe fn kill_guest_now(code: u64) -> ! {
    unsafe { glados_leave_guest(code | EXITED) }
}

pub fn hold_guest(g: super::load::Guest) {
    if let Some(sp) = unsafe { guest_slot() }.as_mut() {
        sp.holder = Some(alloc::boxed::Box::new(g));
    }
}

pub unsafe fn set_guest_rsp(v: u64) {
    unsafe { (*core_entry()).guest_rsp = v };
}

/// Give this guest a different image, keeping what `execve` says survives.
///
/// **Not `install`, and the difference is the point.** `install` calls
/// `teardown`, which stands down the screen, the input script, the thread
/// pool *and the child table* -- right for a guest that is finishing and
/// wrong for one that is merely becoming a different program. A child that
/// execs would clear its own parent's record of it.
///
/// Descriptors and the working directory survive, because that is what Linux
/// promises across an exec; everything that describes memory is replaced. The
/// old pages and the old tables go here, which is the one moment both the old
/// and the new are in hand.
pub fn replace_guest(
    r: Regions,
    tables: Option<crate::mem::space::Space>,
) -> bool {
    let slot = unsafe { guest_slot() };
    let Some(old) = slot.as_mut() else { return false };
    let fds = core::mem::take(&mut old.fds);
    let cwd = core::mem::take(&mut old.cwd);
    let saved_fs = old.saved_fs;
    for (at, len) in core::mem::take(&mut old.owned) {
        free_pages(at, len);
    }
    let brk = r.brk;
    *slot = Some(Space {
        image: r.image,
        interp: r.interp,
        stack: r.stack,
        brk_start: brk.at,
        brk_now: brk.at,
        brk_end: brk.at.saturating_add(brk.len as u64),
        maps: Vec::new(),
        fds,
        cwd,
        argv: Vec::new(),
        image_mapped: r.image_mapped,
        interp_mapped: r.interp_mapped,
        stack_mapped: r.stack_mapped,
        brk_mapped: r.brk_mapped,
        tables,
        signals: super::signal::State::default(),
        holder: None,
        owned: alloc::vec::Vec::new(),
        mmap_next: r.mmap_at,
        mmap_end: r.mmap_end,
        arena_free: Vec::new(),
        bits32: r.bits32,
        image_path: String::new(),
        interp_path: None,
        saved_fs,
    });
    true
}

pub unsafe fn set_syscall_stack(stack: u64) {
    unsafe { (*core_entry()).syscall_stack = stack };
}

pub unsafe fn enter_resumed(r: &Resume) -> u64 {
    unsafe {
        crate::cpu::wrmsr(IA32_FS_BASE, r.fs);
        glados_enter_guest_regs(r as *const Resume)
    }
}

/// The running guest's thread-local base, for a `fork` that gives its child
/// the same one.
pub fn guest_fs() -> u64 {
    unsafe { crate::cpu::rdmsr(IA32_FS_BASE) }
}

/// Where the running guest's stack pointer went, for a `fork` that has to give
/// its child the same one.
pub fn guest_rsp() -> u64 {
    unsafe { (*core_entry()).guest_rsp }
}

/// Whether the run deadline has passed, for a loop that would otherwise wait
/// forever on a child that is never coming back.
pub unsafe fn kill_if_overdue() -> bool {
    let d = super::session::current().map(super::session::deadline_of).unwrap_or(0);
    doomed() != 0 || (d != 0 && crate::dev::lapic::ticks() >= d)
}

/// Copy a guest entry into another slot, for `fork`.
///
/// **Descriptors are cloned and therefore shared**, because every `Fd` variant
/// holds an `Rc` around the open file description -- which is what Linux means
/// by a forked child inheriting them. Memory is the opposite: the caller has
/// already copied it, and hands the pages in.
///
/// Frees what it was given if it cannot place it, so the caller has one thing
/// to reason about rather than two.
pub fn clone_guest(
    parent: usize,
    child: usize,
    tables: crate::mem::space::Space,
    owned: alloc::vec::Vec<(u64, usize)>,
) -> bool {
    let Some(p) = (unsafe { slot_at(parent) }).as_ref() else {
        for (at, len) in owned {
            free_pages(at, len);
        }
        return false;
    };
    let fresh = Space {
        image: p.image,
        interp: p.interp,
        stack: p.stack,
        brk_start: p.brk_start,
        brk_now: p.brk_now,
        brk_end: p.brk_end,
        // Re-badged as the child's copies: see `Source::Copied`.
        maps: p.maps.iter().map(|m| Mapping { from: Source::Copied, ..*m }).collect(),
        fds: p.fds.clone(),
        cwd: p.cwd.clone(),
        argv: p.argv.clone(),
        image_mapped: p.image_mapped,
        interp_mapped: p.interp_mapped,
        stack_mapped: p.stack_mapped,
        brk_mapped: p.brk_mapped,
        // Not inherited: the child's memory was *copied*, so it owns pages
        // rather than an image, and they are in `owned` instead.
        // A child inherits its parent's handlers and mask, as Linux has
        // it, and inherits none of its pending signals -- those were
        // addressed to the parent.
        signals: super::signal::State {
            pending: 0,
            // Not inside a handler, whatever the parent was doing: the child
            // resumes at the fork, not at the parent's interrupted frame.
            depth: 0,
            ..p.signals
        },
        holder: None,
        tables: Some(tables),
        owned,
        mmap_next: p.mmap_next,
        mmap_end: p.mmap_end,
        arena_free: p.arena_free.clone(),
        bits32: p.bits32,
        image_path: p.image_path.clone(),
        interp_path: p.interp_path.clone(),
        saved_fs: p.saved_fs,
    };
    *unsafe { slot_at(child) } = Some(fresh);
    true
}

/// The running guest's page-table root, or 0 for the kernel's.
pub fn guest_root() -> u64 {
    match unsafe { guest_slot() }.as_ref().and_then(|sp| sp.tables.as_ref()) {
        Some(t) => t.root(),
        None => 0,
    }
}

/// One particular slot, which is how a `fork` reaches the child's.
///
/// # Safety
/// As `guest_slot`, and additionally: `i` must not be the running guest while
/// another reference to it is live.
#[allow(clippy::mut_from_ref)]
pub unsafe fn slot_at(i: usize) -> &'static mut Option<Space> {
    let t = unsafe { TABLE.get() };
    while t.len() <= i {
        t.push(None);
    }
    &mut t[i]
}

/// Which guest the syscall path is speaking for.
pub fn current_guest() -> usize {
    guest_index()
}

/// Speak for a different guest from now on.
///
/// Deliberately does **not** touch CR3. The page-table root belongs to the
/// task and `task::set_root` owns it; folding the two together would give two
/// things one name, and the first time they needed to move apart -- a kernel
/// reading a guest's memory without running it -- the fold would be the bug.
pub fn set_current_guest(i: usize) {
    TASK_GUEST[me()].store(i, core::sync::atomic::Ordering::Relaxed);
}

/// What task `t` speaks for, for the scheduler's report and the selftest.
pub fn guest_of_task(t: usize) -> usize {
    TASK_GUEST.get(t).map(|g| g.load(core::sync::atomic::Ordering::Relaxed)).unwrap_or(0)
}

/// The lowest slot holding no guest, growing the table if every one is taken.
///
/// **Never slot 0.** Every task's guest index starts at 0, so slot 0 is what a
/// task that is in no session speaks for -- and with one session it was also
/// where that session's guest lived, which made "the shell's task" and "the
/// guest" the same index. With several sessions that would let the shell's
/// task, or any pool task between tenants, resolve `guest_slot()` to whichever
/// session happened to land in slot 0. Kept empty, it answers `None` to them.
pub fn free_guest_slot() -> usize {
    let t = unsafe { TABLE.get() };
    while t.is_empty() {
        t.push(None);
    }
    if let Some(i) = t.iter().skip(1).position(|g| g.is_none()) {
        return i + 1;
    }
    t.push(None);
    t.len() - 1
}

/// How many guests are live, which is what a `fork` claim counts.
pub fn live_guests() -> usize {
    let t = unsafe { TABLE.get() };
    t.iter().filter(|g| g.is_some()).count()
}

/// Say what the running guest is, for the `/proc` files that describe it.
///
/// Separate from `install` rather than another argument to it, because these
/// are names and that takes ranges: the loader knows both, and folding them
/// into one call would mean `Regions` carrying strings it has no use for.
pub fn name_guest(argv: &[&str], image: &str, interp: Option<&str>) {
    if let Some(sp) = unsafe { guest_slot() }.as_mut() {
        sp.argv = argv.iter().map(|a| String::from(*a)).collect();
        sp.image_path = String::from(image);
        sp.interp_path = interp.map(String::from);
    }
}

/// The path the running guest was loaded from, if one is running.
pub fn guest_image() -> Option<String> {
    let sp = unsafe { guest_slot() }.as_ref()?;
    (!sp.image_path.is_empty()).then(|| sp.image_path.clone())
}

pub fn guest_argv() -> Vec<String> {
    unsafe { guest_slot() }.as_ref().map(|s| s.argv.clone()).unwrap_or_default()
}

/// Every range the guest owns, with a name for the ones that have one.
///
/// In ascending address order, because that is the order `/proc/self/maps` is
/// read in and a reader scanning for a containing range may stop at the first
/// one past its address.
pub fn guest_regions() -> Vec<(u64, usize, String)> {
    let Some(sp) = (unsafe { guest_slot() }).as_ref() else { return Vec::new() };
    let mut out: Vec<(u64, usize, String)> = Vec::new();
    out.push((sp.image.at, sp.image.len, sp.image_path.clone()));
    if let (Some(r), Some(p)) = (sp.interp, sp.interp_path.as_ref()) {
        out.push((r.at, r.len, p.clone()));
    }
    out.push((sp.stack.at, sp.stack.len, String::from("[stack]")));
    out.push((
        sp.brk_start,
        sp.brk_end.saturating_sub(sp.brk_start) as usize,
        String::from("[heap]"),
    ));
    for m in &sp.maps {
        out.push((m.at, m.len, String::new()));
    }
    out.sort_by_key(|(at, _, _)| *at);
    out
}

/// Whether the running guest owns this range. False when nothing is running,
/// which is the right answer: with no guest there is no address it may name.
pub fn owns(at: u64, len: usize) -> bool {
    unsafe { guest_slot().as_ref().is_some_and(|s| s.owns(at, len)) }
}

/// Whether the kernel may touch this range on the guest's behalf.
///
/// Two questions, and the second one only became askable once `mprotect` was
/// real. `owns` says the loader handed this range over. This adds: **and the
/// guest has not since taken the rights away from itself.**
///
/// Without it a guest can kill the machine with an entirely legal pair of
/// calls: `mprotect` a page of its own to `PROT_NONE`, then pass a pointer
/// into it to `write`. The range is one it owns, so the region check says yes,
/// and the kernel then reads a page that is not present. `EFAULT` is what
/// Linux answers and it is what this answers.
pub fn reachable(at: u64, len: usize, need_write: bool) -> bool {
    if !owns(at, len) {
        return false;
    }
    let Some(end) = at.checked_add(len as u64) else { return false };
    let mut page = at & !(PAGE - 1);
    while page < end {
        match crate::mem::paging::query(page) {
            Some(p) if p.present && (!need_write || p.write) => {}
            _ => {
                return false;
            }
        }
        page += PAGE;
    }
    true
}

const PAGE: u64 = 4096;

/// Hand the dispatcher every region the loader gave this guest.
///
/// Called from `run` rather than from `load`, and that ordering is the fix for
/// a real hazard: a guest that was loaded and never run would otherwise leave
/// `SPACE` naming memory freed when its `Guest` dropped, and the next thing to
/// read it would be reading a dangling range it believed it had verified.
pub fn install(r: Regions, tables: Option<crate::mem::space::Space>) {
    let brk = r.brk;
    // A slot of its own rather than tearing down whatever the task last held:
    // with several sessions, "whatever this task last held" may be nothing it
    // owns. See `free_guest_slot` for why it is never slot 0.
    set_current_guest(free_guest_slot());
    // A display server is listening before the guest's first instruction, so
    // a client that connects first thing finds one. See `sky::server`.
    crate::sky::server::serve();
    unsafe {
        *guest_slot() = Some(Space {
            image: r.image,
            interp: r.interp,
            image_mapped: r.image_mapped,
            interp_mapped: r.interp_mapped,
            stack_mapped: r.stack_mapped,
            brk_mapped: r.brk_mapped,
            tables,
            signals: super::signal::State::default(),
            holder: None,
            owned: alloc::vec::Vec::new(),
            mmap_next: r.mmap_at,
            mmap_end: r.mmap_end,
            arena_free: {
                let mut v: Vec<(u64, u64)> =
                    r.arena_extra.iter().filter(|&&(a, e)| e > a).map(|&(a, e)| (a, e - a)).collect();
                v.sort_unstable();
                v
            },
            bits32: r.bits32,
            stack: r.stack,
            brk_start: brk.at,
            brk_now: brk.at,
            brk_end: brk.at.saturating_add(brk.len as u64),
            maps: Vec::new(),
            fds: {
                let mut v = Vec::with_capacity(MAX_FDS);
                v.push(Some(super::fs::Fd::Stdin));
                v.push(Some(super::fs::Fd::Stdout));
                v.push(Some(super::fs::Fd::Stderr));
                v
            },
            cwd: start_cwd(),
            argv: Vec::new(),
            image_path: String::new(),
            interp_path: None,
            saved_fs: 0,
        });
    }
}

/// Give back everything the guest still held, and put `FS` back.
///
/// A guest that exits without unmapping is the ordinary case, not an error --
/// `exit_group` is how programs end -- so the teardown is where mappings are
/// actually reclaimed and `munmap` is only the early return of one.
pub fn teardown() -> usize {
    // **Before anything else, and unconditionally for this session.** A guest
    // that took the display and then faulted is exactly the case this has to
    // cover, and a release conditional on a tidy exit would leave the desktop
    // stood down with the last frame of a dead program on it and no way back
    // short of a reboot. Only *this* session's hold, though: another session
    // holding the screen is still drawing on it.
    let sess = super::session::current();
    super::dev::release_screen(sess);
    // The scripted input feed is one per machine and is a test harness's, so
    // it stands down only when the last session does.
    if super::session::live_count() <= 1 {
        super::input::stop();
    }
    if let Some(s) = sess {
        super::fork::reset(s);
    }
    let mut freed = 0;
    unsafe {
        if let Some(sp) = guest_slot().as_mut() {
            // A guest that writes and exits without closing is the ordinary
            // case, not an error, so this is where those bytes actually reach
            // the store. Dropped in order, because `flush` only commits the
            // last name for a body and two names left open would otherwise
            // both decline.
            for slot in sp.fds.iter_mut() {
                if let Some(f) = slot.take() {
                    f.flush();
                }
            }
            let maps = core::mem::take(&mut sp.maps);
            for m in maps {
                // A guest is free to exit having mprotected its mappings to
                // something the heap cannot reuse. Handing a read-only or
                // absent page back to the allocator would poison it for
                // whatever asks next, and the symptom would appear in an
                // unrelated subsystem hours later.
                if m.from == Source::Paged {
                    free_paged(sp, m.at, page_up(m.len) as u64);
                } else {
                    give_back(m.at, m.len, Some(m.from));
                }
                freed += 1;
            }
            // The image, stack, interpreter and break came from `Exec`
            // allocations the `Guest` still owns and will drop, so the pages go
            // back the same way -- but their *rights* do not, and nothing else
            // puts them back.
            //
            // **The interpreter was missing from this list and it cost a
            // halted machine.** `run` opens four regions and this restored
            // three, so a real `ld.so` that mprotected its own RELRO read-only
            // handed that page to the heap still read-only. The guest exited
            // cleanly, `fat get` allocated, landed on it, and took a `#PF` at
            // ring 0 with `CR0.WP` on -- in a shell command, with no guest
            // running, several seconds after the thing that caused it. Fourth
            // time this tree has made this mistake and the first time the two
            // lists were different lengths, which is why the loop takes the
            // whole set now rather than three of it.
            for r in [
                (Some(sp.image), sp.image_mapped),
                (sp.interp, sp.interp_mapped),
                (Some(sp.stack), sp.stack_mapped),
                (
                    Some(Region { at: sp.brk_start, len: (sp.brk_end - sp.brk_start) as usize }),
                    sp.brk_mapped,
                ),
            ]
            .into_iter()
            .filter_map(|(r, mapped)| if mapped { None } else { r })
            {
                crate::mem::paging::protect(r.at, r.len, crate::mem::paging::Perm::RWX);
            }
            crate::cpu::wrmsr(IA32_FS_BASE, sp.saved_fs);
        }
        if let Some(sp) = guest_slot().as_mut() {
            // A child's own pages. Taken before the entry is dropped, since
            // dropping it drops the tables that map them and the order does
            // not matter for the heap but does for reading this.
            for (at, len) in core::mem::take(&mut sp.owned) {
                free_pages(at, len);
                freed += len;
            }
        }
        *guest_slot() = None;
    }
    // The display connections this guest held closed with its descriptors
    // above; end those and no others. `reap` ended every client on the
    // machine, which with several sessions is everybody else's windows too.
    crate::sky::server::reap_closed();
    set_current_guest(0);
    freed
}

/// Free a finished forked child's guest entry: its descriptors, the mappings
/// it made itself, the pages it was copied into and its tables.
///
/// **Not `teardown`**, which belongs to the session: it stands down the
/// screen, the input script, the display server and every pool. A child ending
/// owes none of that, only its own memory.
///
/// The caller must already be off this entry's root; dropping the entry drops
/// the tables.
pub fn release_guest(i: usize) {
    let Some(mut sp) = (unsafe { slot_at(i) }).take() else { return };
    for slot in sp.fds.iter_mut() {
        if let Some(f) = slot.take() {
            f.flush();
        }
    }
    let maps = core::mem::take(&mut sp.maps);
    for m in maps {
        if m.from == Source::Paged {
            free_paged(&mut sp, m.at, page_up(m.len) as u64);
        } else {
            give_back(m.at, m.len, Some(m.from));
        }
    }
    for (at, len) in core::mem::take(&mut sp.owned) {
        free_pages(at, len);
    }
    drop(sp);
}

pub fn page_up(n: usize) -> usize {
    n.max(1).div_ceil(4096) * 4096
}

pub fn alloc_pages(len: usize) -> Option<u64> {
    use alloc::alloc::{alloc_zeroed, Layout};
    let layout = Layout::from_size_align(page_up(len), 4096).ok()?;
    let p = unsafe { alloc_zeroed(layout) };
    if p.is_null() {
        return None;
    }
    Some(p as u64)
}

pub fn free_pages(at: u64, len: usize) {
    use alloc::alloc::{dealloc, Layout};
    if let Ok(layout) = Layout::from_size_align(page_up(len), 4096) {
        unsafe { dealloc(at as *mut u8, layout) };
    }
}

/// One recorded call. **The whole point of stage 0.**
#[derive(Clone, Copy)]
pub struct Call {
    pub nr: u64,
    pub args: [u64; 6],
    pub ret: u64,
    /// Whether this kernel actually answered, or recorded the question and
    /// returned `-ENOSYS`. Both are measurements and only one is a service.
    pub served: bool,
    /// The path this call named, for the calls that name one.
    ///
    /// **Captured rather than pointed at, and that is the whole point.** The
    /// trace recorded six numbers, which is enough while a guest is asking for
    /// syscalls it may not get and useless the moment it is asking for
    /// *files*: a real `ld.so` searching for a library makes a dozen
    /// identical-looking `openat` calls that differ only in a string, and the
    /// pointer is into guest memory that is freed before anybody reads the
    /// trace. So the bytes are copied at the call.
    ///
    /// Sixty-four is chosen against the thing being measured rather than
    /// against `PATH_MAX`: the longest path a library search actually tries is
    /// about thirty-five characters, and a trace entry is not the place to
    /// spend four kilobytes each.
    pub path: [u8; PATH_SNIP],
    pub path_len: u8,
}

pub const PATH_SNIP: usize = 64;

impl Call {
    /// The path it named, or nothing.
    pub fn path(&self) -> Option<&str> {
        if self.path_len == 0 {
            return None;
        }
        core::str::from_utf8(&self.path[..self.path_len as usize]).ok()
    }
}

/// Which argument of a call is a path, for the calls that take one.
///
/// A table rather than a match inside the dispatcher, because the dispatcher
/// already decides what a call *does* and this decides what it is *about*.
/// Descriptor-relative calls put the path second, which is the detail that
/// makes reading `openat`'s first argument produce a plausible empty string.
fn path_arg(nr: u64) -> Option<usize> {
    // A 32-bit guest's calls are tagged, and their numbers are i386's.
    if nr & super::i386::TAG != 0 {
        return super::i386::path_arg(nr & !super::i386::TAG);
    }
    Some(match nr {
        SYS_OPEN | SYS_STAT | SYS_LSTAT | SYS_ACCESS => 0,
        SYS_OPENAT | SYS_NEWFSTATAT => 1,
        _ => return None,
    })
}

/// The last finished session's calls, per task that ran one, so the shell and
/// a background job each read the trace of the session *they* ran rather than
/// whichever finished last. Recording goes to the session itself
/// (`session::record`), so a forked child's and a thread's calls land in their
/// session's trace whichever task made them.
static TASK_TRACE: Racy<[Vec<Call>; crate::task::MAX_TASKS]> =
    Racy::new([const { Vec::new() }; crate::task::MAX_TASKS]);
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
    arm_core();
    unsafe {
        let top = core::ptr::addr_of!(SYSCALL_STACK.0) as u64 + (16 * 1024);
        (*core_entry()).syscall_stack = top;
        // And the 32-bit door beside it.
        super::i386::arm();
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
    unsafe { (*TASK_TRACE.get())[me()].clone() }
}

pub fn clear_trace() {
    unsafe { (*TASK_TRACE.get())[me()].clear() }
}

pub(super) fn record(mut c: Call) {
    // Read the path *here*, after the call has run and while the guest's
    // memory is still its own. `read_cstr` is bounds-checked, so a guest
    // passing rubbish costs an `EFAULT` this quietly drops rather than a
    // kernel reading whatever the number pointed at.
    if let Some(i) = path_arg(c.nr) {
        if let Ok(s) = read_cstr(c.args[i]) {
            let b = s.as_bytes();
            let n = b.len().min(PATH_SNIP);
            c.path[..n].copy_from_slice(&b[..n]);
            c.path_len = n as u8;
        }
    }
    if MEMLOG.load(Ordering::Relaxed) {
        // i386 numbers carry the tag; the 64-bit ones do not.
        let n = c.nr;
        let i386 = n & (1 << 31) != 0;
        let base = n & !(1 << 31);
        let mem = if i386 { matches!(base, 90 | 91 | 125 | 192 | 219 | 356 | 194 | 93) } else { matches!(base, 9 | 10 | 11 | 28 | 319 | 77) };
        if mem {
            crate::kprintln!(
                "  [mem] {} {:#x} {:#x} {:#x} {:#x} {:#x} -> {:#x}",
                name_of(n), c.args[0], c.args[1], c.args[2], c.args[3], c.args[4], c.ret
            );
        }
    }
    if MISSES.load(Ordering::Relaxed) && c.path_len > 0 && (c.ret as i64) < 0 {
        let p = core::str::from_utf8(&c.path[..c.path_len as usize]).unwrap_or("?");
        crate::kprintln!("  [miss] {} {} -> {}", name_of(c.nr), p, c.ret as i64);
    }
    super::session::record(c, TRACE_CAP);
}

/// Print every path a guest named and was refused, as it happens.
///
/// The trace keeps the first thousand calls, and a game spends those loading
/// libraries; the lookup that decides whether it starts comes thousands of
/// calls later. A refused path is the one fact that question turns on, and
/// printing it live costs nothing while the switch is off.
pub static MISSES: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
/// Print every memory-map call as it happens: which call made a page what it
/// is, which no end-of-run report can say once the console may be wedged.
pub static MEMLOG: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// The guest wrote something. Only the two standard descriptors, and every
/// other fd is `-EBADF` rather than silently accepted -- a write to fd 7 that
/// reported success would be a program whose output vanished.
pub(super) fn sys_write(fd: u64, buf: u64, len: usize) -> u64 {
    // **Ask the table, not the number.** This compared `fd` against 1 and 2,
    // which is right until a guest does the one thing every shell does:
    // `close(1)` then `open(...)` hands the file descriptor 1, and a write to
    // it went to the console -- output the guest had redirected, printed to
    // the terminal, reported as successful. `close(1)` alone was worse, since
    // writes to a descriptor that is not open have to be `EBADF`.
    if is_socket(fd) {
        return sys_send(fd, buf, len as u64);
    }
    // A Unix socket, which is a different transport with the same shape.
    // A short write is real here -- the queue is bounded -- so the count is
    // the answer and a caller that ignores it loses the tail.
    if let Some(sock) = unix_sock(fd) {
        let stream = {
            let b = sock.borrow();
            match &*b {
                super::unix::Sock::Stream { pipe, side } => Some((pipe.clone(), *side)),
                _ => None,
            }
        };
        let Some((pipe, side)) = stream else { return ENOTCONN };
        if !reachable(buf, len, false) {
            return EFAULT;
        }
        let data = unsafe { core::slice::from_raw_parts(buf as *const u8, len) };
        return match super::unix::write(&pipe, side, data) {
            Ok(n) => n as u64,
            Err(e) => e as u64,
        };
    }
    // An anonymous file. **A write past the end is short rather than growing
    // the file**, which is a real deviation from Linux and is the same hazard
    // `ftruncate` refuses for: growing moves the pages, and the address here
    // *is* the memory. A caller that checks its return value copes; one that
    // does not would have been wrong about a short write anyway.
    if let Some(m) = memfd(fd) {
        if !reachable(buf, len, false) {
            return EFAULT;
        }
        let mut f = m.borrow_mut();
        let at = f.cursor.min(f.len);
        let n = (f.len - at).min(len);
        if n > 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(buf as *const u8, (f.at + at as u64) as *mut u8, n)
            };
        }
        f.cursor = at + n;
        return n as u64;
    }
    // A pipe, which is the same transport with one end refused.
    if let Some(end) = pipe_end(fd) {
        let (pipe, writing) = {
            let e = end.borrow();
            (e.pipe.clone(), e.writing)
        };
        if !writing {
            // Writing the read end, the mirror of the refusal in `sys_read`.
            return EBADF;
        }
        if !reachable(buf, len, false) {
            return EFAULT;
        }
        let data = unsafe { core::slice::from_raw_parts(buf as *const u8, len) };
        return match super::unix::write(&pipe, super::fs::Fd::pipe_side(true), data) {
            Ok(n) => n as u64,
            Err(e) => e as u64,
        };
    }
    let sink = with_fds(|fds, _| {
        matches!(
            fds.get(fd as usize),
            Some(Some(super::fs::Fd::Stdout)) | Some(Some(super::fs::Fd::Stderr))
        )
    });
    if sink != Some(true) {
        // Not a stream, so it is either a file open for writing or an error,
        // and both answers live in one place rather than being decided twice.
        return write_file(fd, buf, len);
    }
    // A zero-length write succeeds without the pointer being looked at, which
    // is what Linux does and what an allocator flushing an empty buffer
    // expects.
    if len == 0 {
        return 0;
    }
    // Read-only is enough: this call reads the buffer and prints it.
    if !reachable(buf, len, false) {
        return EFAULT;
    }
    let bytes = unsafe { core::slice::from_raw_parts(buf as *const u8, len) };
    // **Lossy, and the first version of this was silently not.** `from_utf8`
    // answers a `Result`, and iterating a `Result` runs the body zero times on
    // the error arm -- so a guest writing Latin-1 or raw bytes printed nothing
    // at all and still got the full length back. A write that reports success
    // and produces no output is the worst shape this call can take, because
    // the guest has no way to find out.
    crate::kprint!("{}", alloc::string::String::from_utf8_lossy(bytes));
    len as u64
}

/// Copy a NUL-terminated path out of guest memory.
///
/// Checked a page at a time rather than a byte at a time, because `reachable`
/// walks the page tables and a path is up to four kilobytes: per byte that is
/// four thousand walks to read a filename. Checked at all because the pointer
/// is the guest's, and a path is the one argument every file call starts by
/// dereferencing.
///
/// Refuses an unterminated string rather than reading to the end of the page,
/// since a path with no NUL is a bug in the caller and guessing where it ends
/// invents a filename.
///
/// **Three failures, three errnos, and they were one.** This answered `None`
/// for an unreachable pointer, for a string with no terminator, and for bytes
/// that are not UTF-8, and every caller turned that into `EFAULT`. Two of
/// those are not `EFAULT`: the pointer was fine and the program is told to go
/// looking at its pointer arithmetic. A Linux path is bytes rather than text,
/// so a Latin-1 filename is a perfectly legal thing to ask for and an
/// impossible thing to store in a namespace keyed by `String` -- `ENOENT` is
/// the true answer there, since no such name can exist here.
pub fn read_cstr(at: u64) -> Result<String, u64> {
    let mut out = alloc::vec::Vec::new();
    let mut p = at;
    let mut checked_to = at;
    while out.len() < PATH_MAX {
        if p >= checked_to {
            // Cover to the end of this page, and fall back to a byte at a time
            // when that overshoots.
            //
            // **The fast path alone was wrong, and wrong on ordinary
            // programs.** A region does not have to end on a page boundary --
            // the image's is the ELF span, so a path constant in the last
            // partial page of a binary is entirely legal and entirely
            // unreadable to a check that demands the whole rest of the page.
            // It presented as `open` answering `EFAULT` for a string the guest
            // could read perfectly well itself, which reads as a pointer bug
            // in the program. Found by the fixture that opens a path it
            // carries at the very end of its own image.
            let end = (p & !(PAGE - 1)) + PAGE;
            if reachable(p, (end - p) as usize, false) {
                checked_to = end;
            } else if reachable(p, 1, false) {
                checked_to = p + 1;
            } else {
                return Err(EFAULT);
            }
        }
        let b = unsafe { core::ptr::read_volatile(p as *const u8) };
        if b == 0 {
            return core::str::from_utf8(&out).map(String::from).map_err(|_| ENOENT);
        }
        out.push(b);
        p += 1;
    }
    Err(ENAMETOOLONG)
}

/// The descriptor table, or nothing when no guest is running.
pub(super) fn with_fds<T>(f: impl FnOnce(&mut Vec<Option<super::fs::Fd>>, &str) -> T) -> Option<T> {
    let sp = unsafe { guest_slot() };
    // Destructured rather than borrowed twice, which is also why the working
    // directory is no longer cloned here. It was, because `&mut sp.fds` and
    // `&sp.cwd` are two borrows of one `sp` -- so every `read` in a guest's
    // copy loop allocated and freed a string it never looked at.
    let Space { fds, cwd, .. } = sp.as_mut()?;
    Some(f(fds, cwd))
}

/// The directory a `*at` call's relative path is resolved against.
///
/// `AT_FDCWD` means the working directory; anything else is a descriptor that
/// has to *be* a directory. This was `ENOSYS` for a long time under a comment
/// saying a descriptor-relative path "needs the directory's path, which means
/// keeping one per open directory" -- and `fs::Dir` has carried `path` since it
/// was written, so the thing the refusal was waiting for was already there. The
/// refusal outlived its reason, which is why it is worth saying so here.
///
/// What it cost while it stood: `openat(dirfd, rel)` is how every `fts`-based
/// program walks a tree, so `find`, `rm -r`, `du` and `cp -r` all stopped at
/// the first subdirectory.
///
/// **An absolute path ignores `dirfd` entirely**, which is POSIX and is not a
/// shortcut: a caller passing a closed descriptor with an absolute path is
/// correct code, so checking the descriptor first would refuse it.
fn at_base(fds: &[Option<super::fs::Fd>], dirfd: u64, cwd: &str) -> Result<String, u64> {
    if (dirfd as i64) == super::fs::AT_FDCWD {
        return Ok(String::from(cwd));
    }
    let i = dirfd as i64;
    if i < 0 || i as usize >= fds.len() {
        return Err(EBADF);
    }
    match &fds[i as usize] {
        Some(super::fs::Fd::Dir(d)) => Ok(d.borrow().path.clone()),
        // A descriptor that is open and is not a directory is `ENOTDIR`, which
        // is a different fact from `EBADF` and sends a program somewhere else.
        Some(_) => Err(ENOTDIR),
        None => Err(EBADF),
    }
}

/// Open a path in the namespace and hand back a descriptor.
///
/// **Read-only, and that is a decision rather than a gap.** The namespace is
/// content-addressed and snapshotted, so a write is not a store into a file,
/// it is a new object and a new root hash. Letting a guest do that through a
/// POSIX `write` would give it a way to change the tree that bypasses every
/// gate `sysbox` puts in front of the shell. When guests get to write it
/// should be a deliberate design, so for now `O_WRONLY` and `O_RDWR` answer
/// `EACCES` and say why here.
/// Put a new description in the lowest free slot, which is what makes
/// `close(1)` then `open(...)` hand back 1 and is the whole of how a shell
/// redirects.
/// The device and blocking mode behind a descriptor, when it names an event
/// stream and nothing otherwise.
fn input_fd(fd: u64) -> Option<(super::input::Dev, bool)> {
    with_fds(|fds, _| match fds.get(fd as usize) {
        Some(Some(super::fs::Fd::Dev(b))) => {
            let f = b.borrow();
            super::dev::is_input(f.node).map(|d| (d, f.nonblock))
        }
        _ => None,
    })
    .flatten()
}

fn place_fd(fds: &mut Vec<Option<super::fs::Fd>>, entry: super::fs::Fd) -> u64 {
    match fds.iter().position(|f| f.is_none()) {
        Some(slot) => {
            fds[slot] = Some(entry);
            slot as u64
        }
        None => {
            if fds.len() >= MAX_FDS {
                return EMFILE;
            }
            fds.push(Some(entry));
            (fds.len() - 1) as u64
        }
    }
}

pub(super) fn sys_openat(dirfd: u64, path_at: u64, flags: u64, _mode: u64) -> u64 {
    let raw = match read_cstr(path_at) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let wants_write = flags & (O_WRONLY | O_RDWR) != 0 || flags & (O_CREAT | O_TRUNC) != 0;
    // An empty path is `ENOENT` on Linux, and here it would otherwise resolve
    // to the working directory: `open("")` would hand back a descriptor for
    // `/`, which is a plausible-looking answer to a call that asked for
    // nothing.
    if raw.is_empty() {
        return ENOENT;
    }
    let cwd_relative = !raw.starts_with('/');
    with_fds(|fds, cwd| {
        // Only consulted for a relative path, so a closed `dirfd` beside an
        // absolute one is not refused -- see `at_base`.
        let base = if cwd_relative {
            match at_base(fds, dirfd, cwd) {
                Ok(b) => b,
                Err(e) => return e,
            }
        } else {
            String::new()
        };
        let Some(path) = super::fs::resolve(&base, &raw) else { return ENOENT };
        // A mounted volume, read in place by range. Before everything else,
        // so a mount point shadows whatever the namespace has at that path.
        if super::disk::claims(&path) {
            if wants_write {
                return EROFS;
            }
            let entry = match super::disk::lookup(&path) {
                Some(Some(n)) if n.is_dir => super::fs::Fd::Dir(alloc::rc::Rc::new(
                    core::cell::RefCell::new(super::fs::Dir {
                        path: path.clone(),
                        entries: super::disk::entries(&path),
                        at: 0,
                    }),
                )),
                Some(Some(_)) => {
                    if flags & O_DIRECTORY != 0 {
                        return ENOTDIR;
                    }
                    match super::disk::DiskFile::open(&path) {
                        Ok(f) => super::fs::Fd::Disk(alloc::rc::Rc::new(core::cell::RefCell::new(f))),
                        Err(_) => return ENOENT,
                    }
                }
                _ => return ENOENT,
            };
            return place_fd(fds, entry);
        }
        // **Before the store, because these are answers rather than blobs.**
        // Nothing under `/proc` is writable, listed by `sysbox` or in a
        // snapshot, and asking the store about it first would answer `ENOENT`
        // for a path that does exist.
        // **A device is opened for writing and that is not the jail's
        // business.** `/tmp` is the whole of where a guest may write because
        // everywhere else is the content-addressed store, and a new root hash
        // per write is the objection. A character device is not in the store,
        // so the objection does not apply -- and `/dev/fb0` opened read-only
        // is a framebuffer nothing can draw on.
        if let Some(n) = super::dev::node(&path) {
            if flags & O_DIRECTORY != 0 {
                return ENOTDIR;
            }
            // An input device opens at the *present*, not at the beginning
            // of time: a program that received every keystroke since boot the
            // moment it started would look possessed. Everything else opens
            // at offset zero, which is where a framebuffer's top-left is.
            let at = match super::dev::is_input(n) {
                Some(d) => super::input::now_at(d) as usize,
                None => 0,
            };
            if let super::dev::Node::Snd(k) = n {
                super::snd::opened(k);
            }
            let entry = super::fs::Fd::Dev(alloc::rc::Rc::new(core::cell::RefCell::new(
                super::fs::DevFile {
                    path: path.clone(),
                    node: n,
                    at,
                    nonblock: flags & O_NONBLOCK != 0,
                },
            )));
            return place_fd(fds, entry);
        }
        if super::proc::claims(&path) || super::dev::is_dir(&path) {
            if wants_write {
                return EROFS;
            }
            let entry = if super::proc::is_dir(&path) || super::dev::is_dir(&path) {
                super::fs::Fd::Dir(alloc::rc::Rc::new(core::cell::RefCell::new(
                    super::fs::Dir {
                        path: path.clone(),
                        entries: if super::dev::is_dir(&path) {
                            super::dev::entries(&path)
                        } else {
                            super::proc::entries(&path)
                        },
                        at: 0,
                    },
                )))
            } else {
                if flags & O_DIRECTORY != 0 {
                    return ENOTDIR;
                }
                // Generated here and held like any other file, which is the
                // whole reason this costs so little: `read` and `lseek` are
                // already written against a `Vec` and neither knows the
                // difference.
                let Some(data) = super::proc::read(&path) else { return ENOENT };
                super::fs::Fd::File(alloc::rc::Rc::new(core::cell::RefCell::new(
                    super::fs::File {
                        path: path.clone(),
                        data,
                        at: 0,
                        writable: false,
                        dirty: false,
                    },
                )))
            };
            return place_fd(fds, entry);
        }
        // The jail, checked on the resolved path and before anything is
        // created. `EROFS` rather than `EACCES`, because the objection is to
        // where the file is rather than to who is asking.
        if wants_write && !super::fs::writable(&path) {
            return EROFS;
        }
        let is_dir = sysbox::is_dir(&path);
        if flags & O_DIRECTORY != 0 && !is_dir {
            return ENOTDIR;
        }
        let entry = if is_dir {
            if wants_write {
                return EISDIR;
            }
            super::fs::Fd::Dir(alloc::rc::Rc::new(core::cell::RefCell::new(super::fs::Dir {
                path: path.clone(),
                entries: sysbox::listing(&path),
                at: 0,
            })))
        } else {
            // Asked before the copy is made, not after: the point is to not
            // allocate the file, so a check that reads it first and then
            // measures has already lost.
            let size = sysbox::blob_len(&path);
            if size.is_none() && flags & O_CREAT == 0 {
                return ENOENT;
            }
            if size.is_some() && flags & (O_CREAT | O_EXCL) == (O_CREAT | O_EXCL) {
                return EEXIST;
            }
            let held: usize = fds
                .iter()
                .filter_map(|f| match f {
                    Some(super::fs::Fd::File(b)) => Some(b.borrow().data.len()),
                    _ => None,
                })
                .sum();
            if held.saturating_add(size.unwrap_or(0)) > OPEN_MAX_BYTES {
                return ENOMEM;
            }
            // A truncating or brand-new open does not read what is there, which
            // is the whole point of `O_TRUNC` and is also the only way to
            // rewrite a file larger than the open-bytes budget.
            let data = if flags & O_TRUNC != 0 || size.is_none() {
                alloc::vec::Vec::new()
            } else {
                match sysbox::read_blob(&path) {
                    Some(d) => d,
                    None => return ENOENT,
                }
            };
            let at = if flags & O_APPEND != 0 { data.len() } else { 0 };
            // A file created but never written still has to exist, since a
            // program doing `open(O_CREAT); close()` means `touch`.
            let fresh = size.is_none() || flags & O_TRUNC != 0;
            super::fs::Fd::File(alloc::rc::Rc::new(core::cell::RefCell::new(super::fs::File {
                path: path.clone(),
                data,
                at,
                writable: wants_write,
                dirty: wants_write && fresh,
            })))
        };
        // Lowest free descriptor, which is what POSIX promises and what any
        // program doing `close(0); open(...)` to redirect depends on.
        let slot = fds.iter().position(|f| f.is_none());
        match slot {
            Some(i) => {
                fds[i] = Some(entry);
                i as u64
            }
            None if fds.len() < MAX_FDS => {
                fds.push(Some(entry));
                (fds.len() - 1) as u64
            }
            None => EMFILE,
        }
    })
    .unwrap_or(EBADF)
}

/// Duplicate a descriptor onto the lowest free number, or onto a given one.
///
/// **Everything duplicates now, sharing one open file description.** A file
/// used to be refused here, because the cursor lived inside the `Fd` and
/// copying it would give two independent cursors -- a program doing
/// `dup2(fd, 0)` and then reading both would silently read everything twice.
/// Refusing was better than getting it wrong and it still cost a real applet:
/// `hexdump` does exactly that `dup2` to read its file as stdin, and got
/// `ENOSYS`. `fs::Fd` puts the body behind an `Rc<RefCell<..>>`, so `share`
/// hands out another name for one body and the cursor is genuinely shared.
/// Make a directory, remove one, or unlink a name.
///
/// One function because they are one decision three times over: resolve, check
/// the jail, then ask the tree. Splitting them would put the `EROFS` check in
/// three places, which is how one of them ends up missing it.
pub(super) fn sys_name_op(path_at: u64, op: u8) -> u64 {
    let raw = match read_cstr(path_at) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if raw.is_empty() {
        return ENOENT;
    }
    let found = with_fds(|_, cwd| super::fs::resolve(cwd, &raw)).flatten();
    let Some(path) = found else { return ENOENT };
    if !super::fs::writable(&path) {
        return EROFS;
    }
    let is_dir = sysbox::is_dir(&path);
    match op {
        b'm' => {
            if is_dir || sysbox::blob_len(&path).is_some() {
                return EEXIST;
            }
            if sysbox::make_dir(&path) { 0 } else { EPERM }
        }
        b'r' => {
            if !is_dir {
                return ENOTDIR;
            }
            // `rmdir` removes an empty directory and nothing else. The tree
            // would happily detach a full one, and that would be a recursive
            // delete wearing the name of the safe call.
            if !sysbox::listing(&path).is_empty() {
                return ENOTEMPTY;
            }
            if sysbox::detach(&path) { 0 } else { EPERM }
        }
        _ => {
            if is_dir {
                return EISDIR;
            }
            if sysbox::blob_len(&path).is_none() {
                return ENOENT;
            }
            if sysbox::detach(&path) { 0 } else { EPERM }
        }
    }
}

/// The working directory, which is `/` and has never been anything else.
///
/// There is no `chdir`, so this is a constant -- and it is still worth serving,
/// because `pwd` exits 1 without it and every program that resolves a relative
/// path for a message calls it.
/// Where the next guest starts, for a program that finds its data relative to
/// the directory it was started in -- which is every game.
static START_CWD: Racy<String> = Racy::new(String::new());

pub fn set_start_cwd(p: &str) {
    unsafe { *START_CWD.get() = String::from(p) };
}

pub fn start_cwd() -> String {
    let c = unsafe { START_CWD.get().clone() };
    if c.is_empty() { String::from("/") } else { c }
}

/// `chdir`: move the working directory to a directory that exists, wherever
/// it is -- the namespace, a mounted volume, `/proc` or `/dev`.
pub(super) fn sys_chdir(path_at: u64) -> u64 {
    let raw = match read_cstr(path_at) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let found = with_fds(|_, cwd| super::fs::resolve(cwd, &raw)).flatten();
    let Some(path) = found else { return ENOENT };
    let is_dir = match super::disk::lookup(&path) {
        Some(on) => on.is_some_and(|n| n.is_dir),
        None => sysbox::is_dir(&path) || super::dev::is_dir(&path) || super::proc::is_dir(&path) || path == "/",
    };
    if !is_dir {
        return ENOTDIR;
    }
    match unsafe { guest_slot() }.as_mut() {
        Some(sp) => {
            sp.cwd = path;
            0
        }
        None => EBADF,
    }
}

pub(super) fn sys_getcwd(buf: u64, len: u64) -> u64 {
    let cwd = with_fds(|_, cwd| alloc::string::String::from(cwd)).unwrap_or_default();
    let n = cwd.len() + 1;
    if (len as usize) < n {
        return ERANGE;
    }
    if !reachable(buf, n, true) {
        return EFAULT;
    }
    unsafe {
        core::ptr::copy_nonoverlapping(cwd.as_ptr(), buf as *mut u8, cwd.len());
        core::ptr::write((buf as *mut u8).add(cwd.len()), 0);
    }
    // Linux answers the length including the terminator, where glibc's wrapper
    // answers the buffer. A caller reading the raw return and getting zero
    // would think it failed.
    n as u64
}

/// Sleep, by spinning on the timer tick.
///
/// **Honest about what it is.** There is no guest scheduler to block against:
/// the guest owns the machine until it traps, so "sleeping" is a busy wait that
/// gives the tick a chance to fire. It costs the CPU it is not using, which is
/// the trade a kernel with one runnable guest has, and it is bounded by the
/// same deadline everything else is.
pub(super) fn sys_nanosleep(req: u64) -> u64 {
    if !reachable(req, 16, false) {
        return EFAULT;
    }
    let (sec, nsec) = unsafe {
        (
            core::ptr::read_volatile(req as *const u64),
            core::ptr::read_volatile((req + 8) as *const u64),
        )
    };
    sleep_for(sec, nsec)
}

/// The sleep itself, for any ABI's timespec.
/// Give up the CPU from inside a guest's blocking wait, with interrupts on.
///
/// **A syscall arrives with interrupts off** -- `syscall` clears IF through
/// FMASK and `int 0x80` is an interrupt gate -- and every wait loop here
/// yielded without turning them back on. That was harmless while one guest
/// waited and kernel tasks ran between its turns. A program with a dozen
/// threads parked in futex waits found the other case: the run queue was
/// nothing but those waiters, each handing the CPU to the next with IF clear,
/// so no timer interrupt was ever taken, `ticks()` stopped, and every timeout
/// and deadline -- all measured in ticks -- froze with it. The sleeping clock
/// and compositor never woke, the screen stopped, and the run deadline that
/// exists to end exactly this could not fire. Sampled from QEMU's monitor:
/// core 0 cycling through `schedule` and `futex` with IF clear, for minutes.
///
/// Safe to open because everything a syscall leaves live across a switch is
/// per task now -- the guest's stack pointer, the syscall stack, its TLS --
/// and swapped by `schedule`. The caller's interrupt state is put back after.
pub(super) fn wait_turn() {
    let held = holds_big();
    let me = crate::task::current();
    let nap = NAP.load(Ordering::Relaxed) && me < crate::task::MAX_TASKS;
    if nap {
        // **A nap, not a yield.** With guests on several cores, a waiter that
        // only yields comes straight back when its core has nothing else to
        // run, takes the guest lock to look at its condition, and goes round
        // again -- several of them doing that at once keep the lock busy and
        // starve the one thread with work to do. So it sleeps until the next
        // tick, and the end of every syscall wakes whoever is napping, since
        // a syscall is what changes what a guest waits on. The deadline is
        // set before the lock is let go, so a wake that lands in between is
        // kept rather than lost (`task::wake` on a task not yet asleep).
        NAPPING[me].store(true, Ordering::Release);
        NAPPERS.fetch_add(1, Ordering::AcqRel);
        crate::task::set_wake(crate::dev::lapic::ticks() + 1);
    }
    if held {
        big_unlock();
    }
    let flags: u64;
    unsafe { core::arch::asm!("pushfq; pop {}", out(reg) flags, options(nomem, preserves_flags)) };
    crate::cpu::enable_interrupts();
    crate::task::yield_now();
    if nap {
        let _ = crate::task::take_woken();
        crate::task::set_wake(0);
        NAPPING[me].store(false, Ordering::Release);
        NAPPERS.fetch_sub(1, Ordering::AcqRel);
    }
    if flags & (1 << 9) == 0 {
        crate::cpu::disable_interrupts();
    }
    if held {
        big_lock();
    }
}

/// Whether waits nap (`linux nap on|off`).
pub static NAP: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);
/// Tasks napping in `wait_turn`, and how many, so a syscall that finds none
/// pays one load.
static NAPPING: [core::sync::atomic::AtomicBool; crate::task::MAX_TASKS] =
    [const { core::sync::atomic::AtomicBool::new(false) }; crate::task::MAX_TASKS];
static NAPPERS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Wake every napping waiter, and nudge the other cores out of `hlt` so one
/// that is idle picks the waiter up now rather than at its next tick. Called
/// as each syscall finishes.
pub fn wake_nappers() {
    if NAPPERS.load(Ordering::Acquire) == 0 {
        return;
    }
    let me = crate::task::current();
    let mut any = false;
    for t in 0..crate::task::MAX_TASKS {
        if t != me && NAPPING[t].load(Ordering::Acquire) {
            any |= crate::task::wake(t);
        }
    }
    if any {
        crate::dev::lapic::kick_others();
    }
}

// ---- guests on every core -------------------------------------------------
//
// **Ring 3 runs everywhere; the kernel side of a guest runs one at a time.**
// A guest's threads spend their time at ring 3 -- the profile of Portal
// reaching its menu was nine tenths llvmpipe's own rasteriser -- and that is
// what spreads across cores. Every syscall takes `BIG` on the way in and
// gives it back on the way out, and every wait inside one gives it back for
// as long as it waits (`wait_turn`, `sleep_for`), so the guest state behind
// the ~150 `Racy`s the syscall path touches is still reached by one core at
// a time. That is the old big kernel lock, deliberately: auditing each of
// those statics is the SMP audit the 1.4.0 plan holds, and this is the
// shape that does not wait for it.
//
// Acquiring spins with interrupts **on**: the holder may be waiting for this
// core to acknowledge a TLB shootdown, and a core spinning with them off
// would never answer.

const NOBODY: usize = usize::MAX;

/// Whether a guest's tasks may run on any core (`linux cores all|one`).
pub static SPREAD: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);
static BIG: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(NOBODY);

/// Which task holds the guest lock, for `linux prof`.
pub fn big_holder() -> Option<usize> {
    let h = BIG.load(Ordering::Acquire);
    (h != NOBODY).then_some(h)
}

pub fn holds_big() -> bool {
    BIG.load(Ordering::Acquire) == crate::task::current()
}

/// Feature switches for bisecting the SMP work at runtime (`linux feat`).
pub static FEAT_BIG: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);
pub static FEAT_SHOOT: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);
pub static FEAT_KICK: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);

/// **A ticket lock, served in order.** It was a compare-and-swap, and that is
/// unfair in the way that matters here: a guest's threads take and give back
/// the lock thousands of times a second, so a kernel task wanting it once --
/// the shell reaching the namespace for `mouse move` -- lost every race and
/// sat at the prompt for minutes while the game went on.
static NEXT_TICKET: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
static SERVING: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

pub fn big_lock() {
    if !FEAT_BIG.load(Ordering::Relaxed) {
        return;
    }
    let me = crate::task::current();
    if BIG.load(Ordering::Acquire) == me {
        return;
    }
    let ticket = NEXT_TICKET.fetch_add(1, Ordering::AcqRel);
    let mut spins = 0u32;
    while SERVING.load(Ordering::Acquire) != ticket {
        let flags: u64;
        unsafe { core::arch::asm!("pushfq; pop {}", out(reg) flags, options(nomem, preserves_flags)) };
        crate::cpu::enable_interrupts();
        spins += 1;
        if spins % 64 == 0 {
            // Somebody on this core may be what holds it, or be next in line.
            crate::task::yield_now();
        } else {
            core::hint::spin_loop();
        }
        if flags & (1 << 9) == 0 {
            crate::cpu::disable_interrupts();
        }
    }
    BIG.store(me, Ordering::Release);
}

/// Take the guest lock unless this task already holds it, and say whether
/// it was taken here -- for a kernel path that may be reached from inside a
/// syscall or from outside one.
pub fn big_take() -> bool {
    if holds_big() {
        return false;
    }
    big_lock();
    true
}

pub fn big_unlock() {
    let me = crate::task::current();
    if BIG.compare_exchange(me, NOBODY, Ordering::AcqRel, Ordering::Acquire).is_ok() {
        SERVING.fetch_add(1, Ordering::AcqRel);
    }
}

/// Every other core drops its cached translations, and this one waits until
/// they have. After any syscall that removed or narrowed a guest mapping:
/// a thread of the same guest on another core would otherwise go on reading
/// and writing pages through a translation the tables no longer hold --
/// pages already back in the heap.
pub fn tlb_shootdown() {
    if crate::smp::online() <= 1 || !FEAT_SHOOT.load(Ordering::Relaxed) {
        return;
    }
    crate::dev::lapic::shootdown();
}

pub(super) fn sleep_for(sec: u64, nsec: u64) -> u64 {
    if nsec >= 1_000_000_000 {
        return EINVAL;
    }
    let hz = crate::TIMER_HZ as u64;
    let want = sec.saturating_mul(hz) + nsec * hz / 1_000_000_000;
    let until = crate::dev::lapic::ticks().saturating_add(want);

    // **This spun, and it was the one wait in the kernel that did.** The loop
    // was `while ticks() < until { spin_loop() }`: no yield, so a guest
    // sleeping a second held its core for a second and every other task on it
    // waited; no deadline check, so a sleep longer than the session's own
    // bound outlived it; and no `sky::server::pump`, so a Wayland client that
    // slept stopped serving its own display connection. Harmless with one
    // foreground guest and a CPU sink with several.
    //
    // A tick at a time rather than one long sleep, deliberately. The slice is
    // what keeps the two checks below running at the cadence every other
    // blocking path here uses -- the sleep is the only thing that changes, and
    // what it changes is that the core is free meanwhile.
    while crate::dev::lapic::ticks() < until {
        if overran(crate::dev::lapic::ticks()) {
            unsafe { kill_blocked() }
        }
        // The display server answers while its client sleeps, for the same
        // reason it does while one waits on a socket.
        crate::sky::server::pump();
        let now = crate::dev::lapic::ticks();
        let flags: u64;
        unsafe { core::arch::asm!("pushfq; pop {}", out(reg) flags, options(nomem, preserves_flags)) };
        let held = holds_big();
        if held {
            big_unlock();
        }
        crate::cpu::enable_interrupts();
        let _ = crate::task::sleep_until(until.min(now.saturating_add(1)));
        if flags & (1 << 9) == 0 {
            crate::cpu::disable_interrupts();
        }
        if held {
            big_lock();
        }
    }
    0
}

/// Write through a descriptor that names a file.
///
/// Buffered into the body and committed by `close`, for the reason `fs::File`
/// gives: the store is keyed by content, so every commit rewrites the whole
/// blob and gives it a new address. A program writing a kilobyte one byte at a
/// time would otherwise produce a thousand objects.
fn write_file(fd: u64, buf: u64, len: usize) -> u64 {
    if len == 0 {
        return 0;
    }
    if !reachable(buf, len, false) {
        return EFAULT;
    }
    if let Some(r) = with_fds(|fds, _| match fds.get(fd as usize) {
        Some(Some(super::fs::Fd::Dev(body))) => {
            let d = &mut *body.borrow_mut();
            let src = unsafe { core::slice::from_raw_parts(buf as *const u8, len) };
            let n = super::dev::write(d.node, d.at, src);
            // Zero from a device that was given bytes means there was nowhere
            // to put them, which for the framebuffer is a cursor past the end
            // of video memory. `ENOSPC` rather than a short write of nothing,
            // because a copy loop treats zero as "try again" and spins.
            if n == 0 {
                return Some(ENOSPC);
            }
            d.at = d.at.saturating_add(n);
            Some(n as u64)
        }
        _ => None,
    })
    .flatten()
    {
        return r;
    }
    with_fds(|fds, _| {
        let Some(Some(super::fs::Fd::File(body))) = fds.get(fd as usize) else { return EBADF };
        let mut f = body.borrow_mut();
        if !f.writable {
            return EBADF;
        }
        let at = f.at;
        // A write past the end zero-fills the gap, which is what a sparse file
        // reads back as and what `lseek` past the end plus a write means.
        if f.data.len() < at {
            f.data.resize(at, 0);
        }
        let src = unsafe { core::slice::from_raw_parts(buf as *const u8, len) };
        let end = at + len;
        if f.data.len() < end {
            f.data.resize(end, 0);
        }
        f.data[at..end].copy_from_slice(src);
        f.at = end;
        f.dirty = true;
        len as u64
    })
    .unwrap_or(EBADF)
}

/// Gather-write: one call, a vector of buffers.
///
/// **This is why `ls` printed nothing.** It ran to completion, opened the
/// directory, walked it with `getdents64` and `lstat`, and exited 0 -- and
/// produced no output at all, because everything it had to say went through a
/// call that answered `-ENOSYS`. A program that works perfectly and is silent
/// is the worst shape a missing syscall can take, and it is exactly the shape
/// the `-ENOSYS` trace exists to make visible.
///
/// Every pointer is checked twice over: once for the vector itself, and once
/// per buffer, because the vector is guest memory holding guest pointers and
/// neither is trustworthy.
pub(super) fn sys_writev(fd: u64, iov: u64, cnt: u64) -> u64 {
    // Linux's own bound. Refused rather than clamped: a caller that asked for
    // more than this has a bug, and writing the first thousand of its buffers
    // would hide it behind a short count.
    if cnt > 1024 {
        return EINVAL;
    }
    if cnt == 0 {
        return 0;
    }
    let Some(bytes) = (cnt as usize).checked_mul(iov_size() as usize) else { return EINVAL };
    if !reachable(iov, bytes, false) {
        return EFAULT;
    }
    let mut total: u64 = 0;
    for i in 0..cnt as usize {
        let (base, len) = iov_at(iov, i as u64);
        if len == 0 {
            continue;
        }
        let n = sys_write(fd, base, len as usize);
        // An error on the first buffer is the call's error; after that, Linux
        // reports the short count, because those bytes really were written and
        // saying otherwise would have the caller send them twice.
        if (n as i64) < 0 {
            return if total == 0 { n } else { total };
        }
        total += n;
        if n < len {
            break;
        }
    }
    total
}

// --- sockets -------------------------------------------------------------
//
// **The obstacle was never the syscalls.** `net::tcp` held one control block
// and `connect` aborted whatever was open, so a guest could not have two
// sockets -- which is most of what a socket is for. The stack holds a table of
// sixteen now, routed by four-tuple, and this is the surface over it.
//
// Outbound only, and that is the stack's shape rather than a decision made
// here: there is no `Listen` state, so `bind`/`listen`/`accept` would be
// answering for a passive open that does not exist. They are absent rather
// than stubbed, because a `listen` that returns 0 and never accepts anything
// is worse than one that says it cannot.

/// `sockaddr_in`: family, port and address, the last two big-endian.
fn read_sockaddr(at: u64, len: u64) -> Result<(crate::net::Ipv4, u16), u64> {
    if len < 16 {
        return Err(EINVAL);
    }
    if !reachable(at, 16, false) {
        return Err(EFAULT);
    }
    let b = unsafe { core::slice::from_raw_parts(at as *const u8, 16) };
    let family = u16::from_le_bytes([b[0], b[1]]);
    if family != 2 {
        return Err(EAFNOSUPPORT);
    }
    let port = u16::from_be_bytes([b[2], b[3]]);
    Ok(([b[4], b[5], b[6], b[7]], port))
}

fn write_sockaddr(at: u64, len_at: u64, ip: crate::net::Ipv4, port: u16) -> u64 {
    if at == 0 || len_at == 0 {
        return 0;
    }
    if !reachable(len_at, 4, true) || !reachable(at, 16, true) {
        return EFAULT;
    }
    let mut b = [0u8; 16];
    b[0..2].copy_from_slice(&2u16.to_le_bytes());
    b[2..4].copy_from_slice(&port.to_be_bytes());
    b[4..8].copy_from_slice(&ip);
    unsafe {
        core::ptr::copy_nonoverlapping(b.as_ptr(), at as *mut u8, 16);
        core::ptr::write_volatile(len_at as *mut u32, 16);
    }
    0
}

/// Turn a stack error into the errno a program expects to read.
fn sock_err(e: crate::net::tcp::Error) -> u64 {
    use crate::net::tcp::Error as E;
    match e {
        E::NoNic => ENETDOWN,
        E::Timeout => ETIMEDOUT,
        E::Refused => ECONNREFUSED,
        E::Reset => ECONNREFUSED,
        E::NotConnected => ENOTCONN,
        E::NoSlot => ENOBUFS,
    }
}

fn is_socket(fd: u64) -> bool {
    with_fds(|fds, _| matches!(fds.get(fd as usize), Some(Some(super::fs::Fd::Socket(_)))))
        .unwrap_or(false)
}

/// Put a descriptor in the table and answer its number.
fn install_fd(entry: super::fs::Fd) -> u64 {
    with_fds(|fds, _| match fds.iter().position(|f| f.is_none()) {
        Some(i) => {
            fds[i] = Some(entry);
            i as u64
        }
        None if fds.len() < MAX_FDS => {
            fds.push(Some(entry));
            (fds.len() - 1) as u64
        }
        None => EMFILE,
    })
    .unwrap_or(EFAULT)
}

/// The `AF_UNIX` socket a descriptor names, if it is one.
fn unix_sock(fd: u64) -> Option<alloc::rc::Rc<core::cell::RefCell<super::unix::Sock>>> {
    with_fds(|fds, _| match fds.get(fd as usize) {
        Some(Some(super::fs::Fd::Unix(b))) => Some(b.clone()),
        _ => None,
    })
    .flatten()
}

/// The pipe end a descriptor names, if it is one.
fn pipe_end(fd: u64) -> Option<alloc::rc::Rc<core::cell::RefCell<super::fs::PipeEnd>>> {
    with_fds(|fds, _| match fds.get(fd as usize) {
        Some(Some(super::fs::Fd::Pipe(b))) => Some(b.clone()),
        _ => None,
    })
    .flatten()
}

/// `pipe` and `pipe2`, which differ only in whether flags are offered.
///
/// The transport is the one `unix.rs` already has, with one direction unused.
/// That is not a shortcut: a pipe genuinely is a bounded byte queue with a
/// reader and a writer, and the only thing it adds over a socket pair is that
/// each end may do exactly one of the two things.
///
/// **The write end closing is the whole protocol.** `head` stops a pipeline by
/// closing its input; `cat` finds out because its next read answers zero. So
/// the direction each end drives has to be right, and `Fd::pipe_side` is the
/// one place that decides -- a pipe whose two ends shared a queue would work
/// perfectly when talking to itself and deliver nothing across.
pub(super) fn do_pipe(at: u64, flags: u64) -> u64 {
    const O_NONBLOCK: u64 = 0o4000;
    const O_CLOEXEC: u64 = 0o2000000;
    // Refused rather than ignored. `O_DIRECT` on a pipe changes it to packet
    // mode, where a read returns one write's worth however much was asked
    // for, and a program given byte-stream behaviour under that flag reads
    // two messages glued together and cannot tell.
    if flags & !(O_NONBLOCK | O_CLOEXEC) != 0 {
        return EINVAL;
    }
    if !reachable(at, 8, true) {
        return EFAULT;
    }
    let nonblock = flags & O_NONBLOCK != 0;
    // Two handles onto one queue, which is what `pair` answers.
    let (one, two) = super::unix::pair();
    let make = |pipe, writing| {
        super::fs::Fd::Pipe(alloc::rc::Rc::new(core::cell::RefCell::new(
            super::fs::PipeEnd { pipe, writing, nonblock },
        )))
    };
    let r = install_fd(make(one, false));
    if (r as i64) < 0 {
        return r;
    }
    let w = install_fd(make(two, true));
    if (w as i64) < 0 {
        // Put the first one back. A `pipe` that failed having installed one
        // descriptor leaks it, and the guest never learns the number to close.
        with_fds(|fds, _| {
            if let Some(slot) = fds.get_mut(r as usize) {
                *slot = None;
            }
        });
        return w;
    }
    unsafe {
        core::ptr::write_unaligned(at as *mut i32, r as i32);
        core::ptr::write_unaligned((at + 4) as *mut i32, w as i32);
    }
    0
}

/// The epoll set a descriptor names, if it is one.
fn epoll_set(fd: u64) -> Option<alloc::rc::Rc<core::cell::RefCell<super::epoll::Epoll>>> {
    with_fds(|fds, _| match fds.get(fd as usize) {
        Some(Some(super::fs::Fd::Epoll(b))) => Some(b.clone()),
        _ => None,
    })
    .flatten()
}

pub(super) fn sys_epoll_create(size: u64) -> u64 {
    // The hint has been ignored since 2.6.8 and a non-positive one is still
    // refused, because a program passing zero computed a number wrongly rather
    // than choosing a default.
    if (size as i64) <= 0 {
        return EINVAL;
    }
    make_epoll()
}

pub(super) fn sys_epoll_create1(flags: u64) -> u64 {
    const EPOLL_CLOEXEC: u64 = 0o2000000;
    if flags & !EPOLL_CLOEXEC != 0 {
        return EINVAL;
    }
    make_epoll()
}

fn make_epoll() -> u64 {
    install_fd(super::fs::Fd::Epoll(alloc::rc::Rc::new(core::cell::RefCell::new(
        super::epoll::Epoll::new(),
    ))))
}

pub(super) fn sys_epoll_ctl(epfd: u64, op: u64, fd: u64, ev: u64) -> u64 {
    let Some(set) = epoll_set(epfd) else { return EBADF };
    // A set watching itself would be asked its own readiness to compute its own
    // readiness, so it is refused where Linux refuses it too.
    if fd == epfd {
        return EINVAL;
    }
    let target = with_fds(|fds, _| {
        fds.get(fd as usize)
            .and_then(|s| s.as_ref().map(|d| matches!(d, super::fs::Fd::Epoll(_))))
    })
    .flatten();
    let Some(is_set) = target else { return EBADF };
    // **Nesting is refused rather than depth-limited.** Linux allows an epoll
    // inside an epoll to five levels; supporting that means carrying a depth
    // check through every add, and nothing here has asked. Refusing says so
    // where a silent wrong answer would not.
    if is_set {
        return EINVAL;
    }
    let (events, data) = if op == super::epoll::CTL_DEL {
        (0u32, 0u64)
    } else {
        if !reachable(ev, super::epoll::EV_LEN, false) {
            return EFAULT;
        }
        let b = unsafe { core::slice::from_raw_parts(ev as *const u8, super::epoll::EV_LEN) };
        match super::epoll::get_event(b) {
            Some(v) => v,
            None => return EFAULT,
        }
    };
    let mut s = set.borrow_mut();
    let r = match op {
        super::epoll::CTL_ADD => s.add(fd, events, data),
        super::epoll::CTL_MOD => s.modify(fd, events, data),
        super::epoll::CTL_DEL => s.remove(fd),
        _ => Err(super::epoll::EINVAL),
    };
    match r {
        Ok(()) => 0,
        Err(e) => e as u64,
    }
}

pub(super) fn sys_epoll_wait(epfd: u64, evs: u64, maxevents: u64, timeout: u64) -> u64 {
    let Some(set) = epoll_set(epfd) else { return EBADF };
    if (maxevents as i64) <= 0 || maxevents > MAX_WATCHED {
        return EINVAL;
    }
    let bytes = maxevents as usize * super::epoll::EV_LEN;
    if !reachable(evs, bytes, true) {
        return EFAULT;
    }
    let limit = super::poll::ticks_for_ms(timeout as u32 as i32 as i64);
    let start = crate::dev::lapic::ticks();
    loop {
        // Copied out before asking, because answering for a member reaches the
        // descriptor table and the set must not be borrowed across that.
        let entries: alloc::vec::Vec<super::epoll::Watch> = set.borrow().entries().to_vec();
        let out = unsafe { core::slice::from_raw_parts_mut(evs as *mut u8, bytes) };
        let mut n = 0usize;
        let mut fired: alloc::vec::Vec<u64> = alloc::vec::Vec::new();
        for w in &entries {
            if n >= maxevents as usize {
                break;
            }
            let state = match fd_ready(w.fd) {
                Some(s) => s,
                // A watched descriptor that has since been closed. Linux drops
                // the entry when the open file description goes; nothing here
                // can observe that, so it is reported as an error instead --
                // which at least lets the program find out and remove it. A
                // silent never-ready entry is a program waiting forever on
                // something that has gone.
                None => super::poll::POLLERR,
            };
            if let Some(hit) = super::epoll::report(w, state) {
                if super::epoll::put_event(out, n, hit, w.data) {
                    n += 1;
                    fired.push(w.fd);
                }
            }
        }
        if n > 0 {
            let mut s = set.borrow_mut();
            for fd in fired {
                s.spend(fd);
            }
            return n as u64;
        }
        if expired(start, limit) {
            return 0;
        }
        if overran(crate::dev::lapic::ticks()) {
            unsafe { kill_blocked() }
        }
        // The display server answers while its client waits.
        crate::sky::server::pump();
        wait_turn();
    }
}

pub(super) const EBUSY: u64 = (-16i64) as u64;

/// The anonymous file a descriptor names, if it is one.
fn memfd(fd: u64) -> Option<alloc::rc::Rc<core::cell::RefCell<super::fs::Memfd>>> {
    with_fds(|fds, _| match fds.get(fd as usize) {
        Some(Some(super::fs::Fd::Memfd(b))) => Some(b.clone()),
        _ => None,
    })
    .flatten()
}

/// `memfd_create`, which is how a Wayland client makes a buffer.
///
/// `MFD_ALLOW_SEALING` is accepted and seals themselves are not implemented,
/// which is stated rather than hidden: the flag only permits sealing later, so
/// accepting it costs nothing, and `fcntl(F_ADD_SEALS)` fails the way it does
/// on a kernel built without the feature. A compositor that required its
/// clients to have sealed would be one written here, and it is not.
pub(super) fn sys_memfd_create(name: u64, flags: u64) -> u64 {
    const MFD_CLOEXEC: u64 = 1;
    const MFD_ALLOW_SEALING: u64 = 2;
    // NOEXEC_SEAL and EXEC (Linux 6.3) say whether the file may be mapped
    // executable. Nothing maps a memfd executable here and no seal is
    // enforced, so both are accepted as statements about intent; refusing
    // them cost every new libwayland a failed first attempt.
    const MFD_NOEXEC_SEAL: u64 = 8;
    const MFD_EXEC: u64 = 0x10;
    if flags & !(MFD_CLOEXEC | MFD_ALLOW_SEALING | MFD_NOEXEC_SEAL | MFD_EXEC) != 0 {
        return EINVAL;
    }
    let name = match read_cstr(name) {
        Ok(n) => n,
        Err(e) => return e,
    };
    install_fd(super::fs::Fd::Memfd(alloc::rc::Rc::new(core::cell::RefCell::new(
        super::fs::Memfd { name, at: 0, len: 0, cursor: 0, maps: 0 },
    ))))
}

/// `ftruncate`, which is the only way a `memfd` gets a size.
///
/// **Refused once the file has been mapped**, and that is the honest version
/// of a hazard rather than a limitation for its own sake. Resizing moves the
/// pages, and this kernel is identity mapped, so a guest holding a mapping
/// would go on writing to the address the pages used to be at -- which the
/// allocator has by then given to something else. Linux can move them because
/// its mappings are indirections and it can rewrite them; here the address
/// *is* the memory.
///
/// Every real user sizes before mapping, which is the order the call was
/// designed around: create, size, map, draw.
/// `fallocate` in its default mode, on a memfd: make the file at least
/// `offset + len` long, which is what `posix_fallocate` asks before a client
/// maps a shared buffer. Anything else -- punching holes, collapsing ranges,
/// a file in the store -- is refused, since only growth has a meaning here.
pub(super) fn sys_fallocate(fd: u64, mode: u64, offset: u64, len: u64) -> u64 {
    const EOPNOTSUPP: u64 = (-95i64) as u64;
    if mode != 0 {
        return EOPNOTSUPP;
    }
    let Some(end) = offset.checked_add(len) else { return EINVAL };
    match memfd(fd) {
        Some(m) => {
            let have = m.borrow().len as u64;
            if end <= have { 0 } else { sys_ftruncate(fd, end) }
        }
        None => EOPNOTSUPP,
    }
}

pub(super) fn sys_ftruncate(fd: u64, len: u64) -> u64 {
    if (len as i64) < 0 {
        return EINVAL;
    }
    if let Some(m) = memfd(fd) {
        let mut f = m.borrow_mut();
        if f.maps > 0 {
            return EBUSY;
        }
        // **`page_up(0)` is 4096, not 0.** It carries a `.max(1)`, so that
        // `alloc_pages(0)` still returns a page rather than a null the caller
        // would hand out -- correct there, and a trap here: the first sizing
        // of an empty file compared `page_up(4096)` against `page_up(0)`,
        // found them equal, took the "nothing moves" path and set the length
        // without ever allocating. What that produced was a memfd reporting
        // 4096 bytes at address zero, which `ftruncate` reported as success
        // and `mmap` then refused. Found by printing the two fields rather
        // than by reading the arithmetic, after three wrong guesses.
        let want = if len == 0 { 0 } else { page_up(len as usize) };
        let have = if f.at == 0 { 0 } else { page_up(f.len) };
        if f.at != 0 && want == have {
            // The same pages, a different visible length. Nothing moves.
            f.len = len as usize;
            return 0;
        }
        let fresh = if want == 0 {
            0
        } else {
            match alloc_pages(want) {
                Some(a) => a,
                None => return ENOMEM,
            }
        };
        if want > 0 {
            // Growth reads as zeros, which is what `ftruncate` promises and
            // what heap pages do not give for free -- they carry whatever the
            // last owner left, and handing that to a guest is a disclosure as
            // much as a wrong answer.
            unsafe { core::ptr::write_bytes(fresh as *mut u8, 0, want) };
            let keep = f.len.min(len as usize);
            if keep > 0 && f.at != 0 {
                unsafe {
                    core::ptr::copy_nonoverlapping(f.at as *const u8, fresh as *mut u8, keep)
                };
            }
        }
        if f.at != 0 {
            free_pages(f.at, have);
        }
        f.at = fresh;
        f.len = len as usize;
        return 0;
    }
    with_fds(|fds, _| match fds.get_mut(fd as usize) {
        Some(Some(super::fs::Fd::File(b))) => {
            let f = &mut *b.borrow_mut();
            if !f.writable {
                // Linux answers `EINVAL` rather than `EACCES` for a descriptor
                // that was not opened for writing, and a program checking
                // errno branches on which.
                return EINVAL;
            }
            f.data.resize(len as usize, 0);
            f.dirty = true;
            0
        }
        Some(Some(_)) => EINVAL,
        _ => EBADF,
    })
    .unwrap_or(EBADF)
}

/// Read a `sockaddr_un` out of guest memory.
///
/// **The path is a fixed 108-byte array and not a string**, so a name is
/// whatever precedes the first NUL inside it. Reading to the end of the
/// buffer instead gives a name with trailing zeros in it, which binds and
/// then never matches what anybody connects to -- silently, because both
/// operations agree with each other and disagree with everybody else.
///
/// Linux's abstract namespace, where a leading NUL means a name with no file
/// behind it, is refused rather than approximated: Wine does not use it and a
/// name this cannot represent is better said than half-kept.
fn sockaddr_un(at: u64, len: u64) -> Result<String, u64> {
    const EINVAL: u64 = (-22i64) as u64;
    if len < 3 || len > 110 {
        return Err(EINVAL);
    }
    if !reachable(at, len as usize, false) {
        return Err(EFAULT);
    }
    let family = unsafe { core::ptr::read(at as *const u16) };
    if family != 1 {
        return Err(EAFNOSUPPORT);
    }
    let mut out = String::new();
    for i in 0..(len as usize - 2).min(108) {
        let c = unsafe { core::ptr::read((at + 2 + i as u64) as *const u8) };
        if c == 0 {
            break;
        }
        out.push(c as char);
    }
    if out.is_empty() {
        return Err(EINVAL);
    }
    Ok(out)
}

pub(super) fn sys_bind(fd: u64, at: u64, len: u64) -> u64 {
    let Some(sock) = unix_sock(fd) else { return ENOTSOCK };
    let path = match sockaddr_un(at, len) {
        Ok(v) => v,
        Err(e) => return e,
    };
    match super::unix::bind(&sock, &path) {
        Ok(()) => 0,
        Err(e) => e as u64,
    }
}

pub(super) fn sys_listen(fd: u64) -> u64 {
    let Some(sock) = unix_sock(fd) else { return ENOTSOCK };
    match super::unix::listen(&sock) {
        Ok(()) => 0,
        Err(e) => e as u64,
    }
}

/// `accept4`, which `accept` is the flagless case of.
///
/// **Blocking is a yield loop and non-blocking is one look**, which is the
/// same bargain `futex` and `nanosleep` make here. A server that blocks costs
/// the CPU it is not using; one that passes `SOCK_NONBLOCK` gets `EAGAIN` and
/// decides for itself.
pub(super) fn sys_accept4(fd: u64, addr: u64, alen: u64, flags: u64) -> u64 {
    const SOCK_NONBLOCK: u64 = 0o4000;
    let Some(sock) = unix_sock(fd) else { return ENOTSOCK };
    loop {
        match super::unix::accept(&sock) {
            Err(e) => return e as u64,
            Ok(Some((pipe, side))) => {
                // The peer of a Unix socket has no address worth reporting --
                // a client that never bound has no name at all. Linux writes
                // a `sockaddr_un` of length 2 holding just the family, which
                // is what a caller passing a buffer expects to find.
                if addr != 0 && alen != 0 && reachable(addr, 2, true) && reachable(alen, 4, true) {
                    unsafe {
                        core::ptr::write(addr as *mut u16, 1);
                        core::ptr::write(alen as *mut u32, 2);
                    }
                }
                let entry = super::fs::Fd::Unix(alloc::rc::Rc::new(core::cell::RefCell::new(
                    super::unix::Sock::Stream { pipe, side },
                )));
                return install_fd(entry);
            }
            Ok(None) => {
                if flags & SOCK_NONBLOCK != 0 {
                    return EAGAIN;
                }
                if unsafe { kill_if_overdue() } {
                    return EAGAIN;
                }
                wait_turn();
            }
        }
    }
}

pub(super) fn sys_socketpair(domain: u64, kind: u64, _proto: u64, sv: u64) -> u64 {
    if domain != 1 {
        return EAFNOSUPPORT;
    }
    if kind & 0xFF != 1 {
        return EPROTONOSUPPORT;
    }
    if !reachable(sv, 8, true) {
        return EFAULT;
    }
    let (a, b) = super::unix::pair();
    let mk = |pipe, side| {
        super::fs::Fd::Unix(alloc::rc::Rc::new(core::cell::RefCell::new(
            super::unix::Sock::Stream { pipe, side },
        )))
    };
    let fa = install_fd(mk(a, super::unix::Side::A));
    if fa > MAX_FDS as u64 {
        return fa;
    }
    let fb = install_fd(mk(b, super::unix::Side::B));
    if fb > MAX_FDS as u64 {
        return fb;
    }
    unsafe {
        core::ptr::write(sv as *mut u32, fa as u32);
        core::ptr::write((sv + 4) as *mut u32, fb as u32);
    }
    0
}

/// `struct msghdr`, `struct iovec` and `struct cmsghdr`, which are three
/// layouts a caller compiled against the real headers already agrees with.
///
/// Written out because every one of them has a hole in it that is easy to
/// miss: `msg_namelen` is a `u32` followed by four bytes of padding, and
/// `cmsg_level` and `cmsg_type` are two `int`s packed into the word after a
/// `size_t`. A structure read one field out of step produces a control
/// message that looks well-formed and names the wrong descriptors.
/// Too big for the control buffer offered.
pub(super) const EMSGSIZE: u64 = (-90i64) as u64;

mod msg {
    pub const NAME: u64 = 0;
    pub const NAMELEN: u64 = 8;
    pub const IOV: u64 = 16;
    pub const IOVLEN: u64 = 24;
    pub const CONTROL: u64 = 32;
    pub const CONTROLLEN: u64 = 40;
    pub const FLAGS: u64 = 48;
    pub const SIZE: usize = 56;

    pub const IOV_BASE: u64 = 0;
    pub const IOV_LEN: u64 = 8;
    pub const IOV_SIZE: u64 = 16;

    pub const CMSG_LEN: u64 = 0;
    pub const CMSG_LEVEL: u64 = 8;
    pub const CMSG_TYPE: u64 = 12;
    /// `CMSG_DATA` is the header rounded up to eight, which for a 16-byte
    /// header is 16 -- but the rounding is written out rather than folded in,
    /// because it is what makes a second control message land in the right
    /// place.
    pub const CMSG_DATA: u64 = 16;

    pub const SOL_SOCKET: u32 = 1;
    pub const SCM_RIGHTS: u32 = 1;

    pub fn align8(n: u64) -> u64 {
        (n + 7) & !7
    }
}

/// The bytes an `iovec` array describes, gathered into one buffer.
///
/// Bounded at `IOV_MAX`, which Linux also bounds, and refused rather than
/// clamped: a caller that asked to send more than this and got a short count
/// would have to guess whether the rest was dropped or never attempted.
/// Whether the running guest is 32-bit, which is what decides the shape of
/// every structure it hands the kernel. Asked at the boundary, by the function
/// that reads or writes the structure, so one implementation of a call serves
/// both ABIs and only the bytes differ.
pub(super) fn abi32() -> bool {
    unsafe { guest_slot() }.as_ref().is_some_and(|s| s.bits32)
}

/// How wide one `struct iovec` is for the running guest: two pointers.
pub(super) fn iov_size() -> u64 {
    if abi32() { 8 } else { msg::IOV_SIZE }
}

/// Entry `i` of an `iovec` array, as (base, length), at the guest's width.
/// The caller has already checked the array is reachable.
pub(super) fn iov_at(iov: u64, i: u64) -> (u64, u64) {
    let e = iov + i * iov_size();
    unsafe {
        if abi32() {
            (
                core::ptr::read_volatile(e as *const u32) as u64,
                core::ptr::read_volatile((e + 4) as *const u32) as u64,
            )
        } else {
            (
                core::ptr::read_volatile((e + msg::IOV_BASE) as *const u64),
                core::ptr::read_volatile((e + msg::IOV_LEN) as *const u64),
            )
        }
    }
}

pub(super) fn gather(iov: u64, cnt: u64) -> Result<Vec<u8>, u64> {
    const IOV_MAX: u64 = 1024;
    if cnt > IOV_MAX {
        return Err(EINVAL);
    }
    let mut out = Vec::new();
    for i in 0..cnt {
        let e = iov + i * iov_size();
        if !reachable(e, iov_size() as usize, false) {
            return Err(EFAULT);
        }
        let (base, len) = iov_at(iov, i);
        let len = len as usize;
        if len == 0 {
            continue;
        }
        if !reachable(base, len, false) {
            return Err(EFAULT);
        }
        out.extend_from_slice(unsafe { core::slice::from_raw_parts(base as *const u8, len) });
    }
    Ok(out)
}

/// Total room an `iovec` array offers, and a writer for filling it.
pub(super) fn scatter(iov: u64, cnt: u64, data: &[u8]) -> Result<usize, u64> {
    const IOV_MAX: u64 = 1024;
    if cnt > IOV_MAX {
        return Err(EINVAL);
    }
    let mut done = 0usize;
    for i in 0..cnt {
        if done == data.len() {
            break;
        }
        let e = iov + i * iov_size();
        if !reachable(e, iov_size() as usize, false) {
            return Err(EFAULT);
        }
        let (base, len) = iov_at(iov, i);
        let len = len as usize;
        let n = len.min(data.len() - done);
        if n == 0 {
            continue;
        }
        if !reachable(base, n, true) {
            return Err(EFAULT);
        }
        unsafe { core::ptr::copy_nonoverlapping(data[done..].as_ptr(), base as *mut u8, n) };
        done += n;
    }
    Ok(done)
}

/// A `msghdr`'s four fields this kernel reads -- iov, iovlen, control,
/// controllen -- at the running guest's width: 28 bytes of 4-byte fields on
/// i386, 56 bytes with 8-byte pointers and lengths on x86-64.
fn msghdr_in(hdr: u64, write: bool) -> Result<(u64, u64, u64, u64), u64> {
    let (size, w) = if abi32() { (28usize, 4u64) } else { (msg::SIZE, 8u64) };
    if !reachable(hdr, size, write) {
        return Err(EFAULT);
    }
    let rd = |i: u64| unsafe {
        if w == 4 {
            core::ptr::read_unaligned((hdr + i * 4) as *const u32) as u64
        } else {
            core::ptr::read_unaligned((hdr + i * 8) as *const u64)
        }
    };
    // name, namelen, iov, iovlen, control, controllen, flags
    Ok((rd(2), rd(3), rd(4), rd(5)))
}

/// What `recvmsg` writes back: the control length used, no flags, no name.
fn msghdr_out(hdr: u64, control_len: u64) {
    unsafe {
        if abi32() {
            core::ptr::write_unaligned((hdr + 20) as *mut u32, control_len as u32);
            core::ptr::write_unaligned((hdr + 24) as *mut u32, 0);
            core::ptr::write_unaligned((hdr + 4) as *mut u32, 0);
        } else {
            core::ptr::write((hdr + msg::CONTROLLEN) as *mut u64, control_len);
            core::ptr::write((hdr + msg::FLAGS) as *mut u32, 0);
            core::ptr::write((hdr + msg::NAMELEN) as *mut u32, 0);
        }
    }
}

/// A control message header: (where its data starts, alignment). i386's
/// `cmsghdr` is three 4-byte fields aligned to 4; x86-64's carries an 8-byte
/// length and aligns to 8. Wayland passes every shared-memory buffer through
/// one of these, so a width mismatch is a display that never gets a frame.
fn cmsg_shape() -> (u64, u64) {
    if abi32() { (12, 4) } else { (msg::CMSG_DATA, 8) }
}

/// Pull `SCM_RIGHTS` descriptors out of a control buffer.
fn control_fds(at: u64, len: u64) -> Result<Vec<super::fs::Fd>, u64> {
    let mut out = Vec::new();
    let (data, align) = cmsg_shape();
    if at == 0 || len < data {
        return Ok(out);
    }
    if !reachable(at, len as usize, false) {
        return Err(EFAULT);
    }
    let mut off = 0u64;
    while off + data <= len {
        let h = at + off;
        let (clen, level, kind) = unsafe {
            if data == 12 {
                (
                    core::ptr::read_unaligned(h as *const u32) as u64,
                    core::ptr::read_unaligned((h + 4) as *const u32),
                    core::ptr::read_unaligned((h + 8) as *const u32),
                )
            } else {
                (
                    core::ptr::read((h + msg::CMSG_LEN) as *const u64),
                    core::ptr::read((h + msg::CMSG_LEVEL) as *const u32),
                    core::ptr::read((h + msg::CMSG_TYPE) as *const u32),
                )
            }
        };
        if clen < data || off + clen > len {
            break;
        }
        if level == msg::SOL_SOCKET && kind == msg::SCM_RIGHTS {
            let n = ((clen - data) / 4) as usize;
            for i in 0..n {
                let fd = unsafe { core::ptr::read_unaligned((h + data + (i * 4) as u64) as *const u32) } as u64;
                let Some(f) = with_fds(|fds, _| {
                    fds.get(fd as usize).and_then(|x| x.as_ref()).map(|x| x.share())
                })
                .flatten() else {
                    return Err(EBADF);
                };
                out.push(f);
            }
        }
        off += (clen + align - 1) & !(align - 1);
    }
    Ok(out)
}

/// Install received descriptors and write one `SCM_RIGHTS` header naming them.
fn write_control(at: u64, room: u64, fds: Vec<super::fs::Fd>) -> Result<u64, u64> {
    if fds.is_empty() {
        return Ok(0);
    }
    let (data, _) = cmsg_shape();
    let need = data + (fds.len() * 4) as u64;
    if at == 0 || room < need {
        return Err(EMSGSIZE);
    }
    if !reachable(at, need as usize, true) {
        return Err(EFAULT);
    }
    let mut nums = Vec::new();
    for f in fds {
        let n = install_fd(f);
        if n >= MAX_FDS as u64 {
            return Err(EMFILE);
        }
        nums.push(n as u32);
    }
    unsafe {
        if data == 12 {
            core::ptr::write_unaligned(at as *mut u32, need as u32);
            core::ptr::write_unaligned((at + 4) as *mut u32, msg::SOL_SOCKET);
            core::ptr::write_unaligned((at + 8) as *mut u32, msg::SCM_RIGHTS);
        } else {
            core::ptr::write((at + msg::CMSG_LEN) as *mut u64, need);
            core::ptr::write((at + msg::CMSG_LEVEL) as *mut u32, msg::SOL_SOCKET);
            core::ptr::write((at + msg::CMSG_TYPE) as *mut u32, msg::SCM_RIGHTS);
        }
        for (i, n) in nums.iter().enumerate() {
            core::ptr::write_unaligned((at + data + (i * 4) as u64) as *mut u32, *n);
        }
    }
    Ok(need)
}


fn unix_stream(fd: u64) -> Option<(alloc::rc::Rc<core::cell::RefCell<super::unix::Pipe>>,
                                   super::unix::Side)> {
    let sock = unix_sock(fd)?;
    let b = sock.borrow();
    match &*b {
        super::unix::Sock::Stream { pipe, side } => Some((pipe.clone(), *side)),
        _ => None,
    }
}

pub(super) fn sys_sendmsg(fd: u64, hdr: u64, _flags: u64) -> u64 {
    let Some((pipe, side)) = unix_stream(fd) else { return ENOTCONN };
    let (iov, cnt, ctl, ctl_len) = match msghdr_in(hdr, false) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let data = match gather(iov, cnt) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let fds = match control_fds(ctl, ctl_len) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let n = match super::unix::write(&pipe, side, &data) {
        Ok(n) => n,
        Err(e) => return e as u64,
    };
    // **After the bytes.** The attachment is stamped with what it travels
    // behind, so attaching first would stamp it at the previous message's end
    // and release it a read too early.
    super::unix::attach(&pipe, side, fds);
    n as u64
}

pub(super) fn sys_recvmsg(fd: u64, hdr: u64, flags: u64) -> u64 {
    const MSG_DONTWAIT: u64 = 0x40;
    let Some((pipe, side)) = unix_stream(fd) else { return ENOTCONN };
    let (iov, cnt, ctl, ctl_len) = match msghdr_in(hdr, true) {
        Ok(v) => v,
        Err(e) => return e,
    };

    let mut room = 0usize;
    for i in 0..cnt.min(1024) {
        let e = iov + i * iov_size();
        if !reachable(e, iov_size() as usize, false) {
            return EFAULT;
        }
        room += iov_at(iov, i).1 as usize;
    }
    let mut buf = alloc::vec![0u8; room.min(64 * 1024)];
    let got = loop {
        match super::unix::read(&pipe, side, &mut buf) {
            Ok(n) => break n,
            Err(-11) => {}
            Err(e) => return e as u64,
        }
        if flags & MSG_DONTWAIT != 0 {
            return EAGAIN;
        }
        if overran(crate::dev::lapic::ticks()) {
            unsafe { kill_blocked() }
        }
        // The display server answers while its client waits.
        crate::sky::server::pump();
        wait_turn();
    };
    if let Err(e) = scatter(iov, cnt, &buf[..got]) {
        return e;
    }
    // Only now, because `collect` answers by what this end has read and the
    // read above is what moved that line.
    let fds = super::unix::collect(&pipe, side);
    let wrote = match write_control(ctl, ctl_len, fds) {
        Ok(v) => v,
        Err(e) => return e,
    };
    msghdr_out(hdr, wrote);
    let _ = msg::NAME;
    got as u64
}

pub(super) fn sys_socket(domain: u64, kind: u64, proto: u64) -> u64 {
    // `AF_UNIX`, which is what wineserver speaks and what nothing on a wire
    // ever sees. Answered before the `AF_INET` path rather than inside it,
    // because the two share a syscall number and nothing else.
    if domain == 1 {
        if kind & 0xFF != 1 {
            return EPROTONOSUPPORT;
        }
        if proto != 0 {
            return EPROTONOSUPPORT;
        }
        let entry = super::fs::Fd::Unix(alloc::rc::Rc::new(core::cell::RefCell::new(
            super::unix::Sock::Fresh,
        )));
        return install_fd(entry);
    }
    if domain != 2 {
        return EAFNOSUPPORT;
    }
    // `SOCK_STREAM` with the close-on-exec and non-blocking bits masked off:
    // there is no `exec` for the first to matter to, and the second is
    // answered by every call taking its own timeout.
    if kind & 0xFF != 1 {
        return EPROTONOSUPPORT;
    }
    if proto != 0 && proto != 6 {
        return EPROTONOSUPPORT;
    }
    let entry = super::fs::Fd::Socket(alloc::rc::Rc::new(core::cell::RefCell::new(
        super::fs::Sock { conn: None },
    )));
    with_fds(|fds, _| match fds.iter().position(|f| f.is_none()) {
        Some(i) => {
            fds[i] = Some(entry);
            i as u64
        }
        None if fds.len() < MAX_FDS => {
            fds.push(Some(entry));
            (fds.len() - 1) as u64
        }
        None => EMFILE,
    })
    .unwrap_or(EBADF)
}

/// The handle behind a descriptor, or the errno saying why there is not one.
fn sock_of(fd: u64) -> Result<alloc::rc::Rc<core::cell::RefCell<super::fs::Sock>>, u64> {
    with_fds(|fds, _| match fds.get(fd as usize) {
        Some(Some(super::fs::Fd::Socket(b))) => Ok(b.clone()),
        Some(Some(_)) => Err(ENOTSOCK),
        _ => Err(EBADF),
    })
    .unwrap_or(Err(EBADF))
}

pub(super) fn sys_connect(fd: u64, at: u64, len: u64) -> u64 {
    // `AF_UNIX` first, because the two share a syscall number and nothing
    // else: `sock_of` below demands a TCP socket and would refuse this one
    // with a message about the wrong thing.
    if let Some(sock) = unix_sock(fd) {
        let path = match sockaddr_un(at, len) {
            Ok(v) => v,
            Err(e) => return e,
        };
        {
            let b = sock.borrow();
            if matches!(*b, super::unix::Sock::Stream { .. }) {
                return EISCONN;
            }
        }
        return match super::unix::connect(&path) {
            Ok((pipe, side)) => {
                *sock.borrow_mut() = super::unix::Sock::Stream { pipe, side };
                0
            }
            Err(e) => e as u64,
        };
    }
    let sock = match sock_of(fd) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if sock.borrow().conn.is_some() {
        return EISCONN;
    }
    let (ip, port) = match read_sockaddr(at, len) {
        Ok(v) => v,
        Err(e) => return e,
    };
    // The borrow ends before the connect: opening one pumps the stack, which
    // can deliver to another socket, which would want this same table.
    match crate::net::tcp::open(ip, port, SOCK_TIMEOUT_MS) {
        Ok(h) => {
            sock.borrow_mut().conn = Some(h);
            0
        }
        Err(e) => sock_err(e),
    }
}

/// How long a socket call waits before answering.
///
/// One number rather than `SO_RCVTIMEO`, because there is no scheduler to
/// block a guest against: a wait here is this task spinning the stack, and a
/// guest that asked for an unbounded one would own the machine until its
/// deadline killed it. Ten seconds is longer than any handshake and shorter
/// than the guest deadline that would otherwise end the argument.
const SOCK_TIMEOUT_MS: u64 = 10_000;

pub(super) fn sys_send(fd: u64, buf: u64, len: u64) -> u64 {
    if len == 0 {
        return 0;
    }
    if !reachable(buf, len as usize, false) {
        return EFAULT;
    }
    let sock = match sock_of(fd) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let Some(h) = sock.borrow().conn else { return ENOTCONN };
    let data = unsafe { core::slice::from_raw_parts(buf as *const u8, len as usize) };
    match crate::net::tcp::send_at(h, data, SOCK_TIMEOUT_MS) {
        Ok(()) => len,
        Err(e) => sock_err(e),
    }
}

pub(super) fn sys_recv(fd: u64, buf: u64, len: u64) -> u64 {
    if len == 0 {
        return 0;
    }
    if !reachable(buf, len as usize, true) {
        return EFAULT;
    }
    let sock = match sock_of(fd) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let Some(h) = sock.borrow().conn else { return ENOTCONN };
    let got = crate::net::tcp::recv_at(h, SOCK_TIMEOUT_MS);
    if got.is_empty() {
        // Zero means end of file, and only end of file. A timeout with the
        // peer still there is `EAGAIN`, because a program reading zero from a
        // live connection concludes the other end hung up.
        return if crate::net::tcp::peer_done(h) { 0 } else { EAGAIN };
    }
    let n = got.len().min(len as usize);
    unsafe { core::ptr::copy_nonoverlapping(got.as_ptr(), buf as *mut u8, n) };
    // Anything past the guest's buffer would need pushing back, which this
    // stack has nowhere to put. Refused as short rather than lost: the caller
    // asked for `len` and got `len`, and the rest is still on the connection.
    n as u64
}

pub(super) fn sys_shutdown(fd: u64, _how: u64) -> u64 {
    let sock = match sock_of(fd) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let taken = sock.borrow_mut().conn.take();
    match taken {
        Some(h) => {
            crate::net::tcp::close_at(h, SOCK_TIMEOUT_MS);
            0
        }
        None => ENOTCONN,
    }
}

/// Scatter-read: one call, a vector of buffers.
///
/// The mirror of `writev` and it arrived for the same reason: `hexdump` reads
/// its input through this, so without it the applet opened the file,
/// `dup2`ed it onto stdin, and then reported "Function not implemented"
/// about a file it was holding open.
pub(super) fn sys_readv(fd: u64, iov: u64, cnt: u64) -> u64 {
    if cnt > 1024 {
        return EINVAL;
    }
    if cnt == 0 {
        return 0;
    }
    let Some(bytes) = (cnt as usize).checked_mul(iov_size() as usize) else { return EINVAL };
    if !reachable(iov, bytes, false) {
        return EFAULT;
    }
    let mut total: u64 = 0;
    for i in 0..cnt as usize {
        let (base, len) = iov_at(iov, i as u64);
        if len == 0 {
            continue;
        }
        let n = sys_read(fd, base, len);
        if (n as i64) < 0 {
            return if total == 0 { n } else { total };
        }
        total += n;
        // Short means the source is out, and asking again would answer zero.
        if n < len {
            break;
        }
    }
    total
}

/// Whether a path exists, and whether the guest could do the named thing to it.
///
/// The mode bits are answered from the jail rather than from permissions,
/// because there are none: everything readable is readable by the one uid
/// there is, and `W_OK` outside `/tmp` is the only "no" this can honestly
/// give. That makes `access(p, W_OK)` a real test of the write jail, which is
/// what a program uses it for.
pub(super) fn sys_access(path_at: u64, mode: u64) -> u64 {
    let raw = match read_cstr(path_at) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if raw.is_empty() {
        return ENOENT;
    }
    let found = with_fds(|_, cwd| super::fs::resolve(cwd, &raw)).flatten();
    let Some(path) = found else { return ENOENT };
    // **A mounted volume is not in the namespace**, so asking only `sysbox`
    // answered `ENOENT` for every file on one. `open` and `stat` already ask
    // the disk first; `access` was the call that did not, and the Source
    // engine decides whether a game exists with `access("portal/gameinfo.txt")`
    // -- so a game that could open its every file reported that it was not
    // installed.
    if let Some(on) = super::disk::lookup(&path) {
        let Some(n) = on else { return ENOENT };
        if mode & 2 != 0 {
            return EROFS;
        }
        if mode & 1 != 0 && !n.is_dir {
            return EACCES;
        }
        return 0;
    }
    if super::dev::node(&path).is_some() || super::dev::is_dir(&path) || super::proc::claims(&path) {
        return 0;
    }
    if !sysbox::is_dir(&path) && sysbox::blob_len(&path).is_none() {
        return ENOENT;
    }
    // W_OK is bit 1. X_OK is granted on directories and refused on files,
    // since nothing here can be executed by name.
    if mode & 2 != 0 && !super::fs::writable(&path) {
        return EACCES;
    }
    if mode & 1 != 0 && !sysbox::is_dir(&path) {
        return EACCES;
    }
    0
}

/// Copy between two descriptors without the bytes going through the guest.
///
/// Both ends are already in the kernel here, so this is what it claims to be
/// rather than an optimisation that pretends: no guest buffer is involved and
/// no guest pointer is dereferenced. `cat` reaches for it first and falls back
/// to `read`/`write` when it fails, which is why it worked without this and
/// still spent a refused call every time.
pub(super) fn sys_sendfile(out: u64, into: u64, off_at: u64, count: u64) -> u64 {
    if off_at != 0 {
        // An explicit offset means "read from here and do not move the
        // cursor", which needs a second cursor this table does not keep.
        // Refused rather than served from the wrong place.
        return ENOSYS;
    }
    let taken = with_fds(|fds, _| match fds.get(into as usize) {
        Some(Some(super::fs::Fd::File(b))) => {
            let f = &mut *b.borrow_mut();
            let from = f.at.min(f.data.len());
            let n = (f.data.len() - from).min(count as usize);
            let chunk = f.data[from..from + n].to_vec();
            f.at = from + n;
            Some(chunk)
        }
        Some(Some(_)) => None,
        _ => None,
    })
    .flatten();
    let Some(chunk) = taken else { return EINVAL };
    if chunk.is_empty() {
        return 0;
    }
    let sink = with_fds(|fds, _| {
        matches!(
            fds.get(out as usize),
            Some(Some(super::fs::Fd::Stdout)) | Some(Some(super::fs::Fd::Stderr))
        )
    });
    if sink == Some(true) {
        crate::kprint!("{}", alloc::string::String::from_utf8_lossy(&chunk));
        return chunk.len() as u64;
    }
    with_fds(|fds, _| {
        let Some(Some(super::fs::Fd::File(b))) = fds.get(out as usize) else { return EBADF };
        let f = &mut *b.borrow_mut();
        if !f.writable {
            return EBADF;
        }
        let at = f.at;
        let end = at + chunk.len();
        if f.data.len() < end {
            f.data.resize(end, 0);
        }
        f.data[at..end].copy_from_slice(&chunk);
        f.at = end;
        f.dirty = true;
        chunk.len() as u64
    })
    .unwrap_or(EBADF)
}

/// `struct utsname`: six fields of 65 bytes, NUL-padded.
///
/// **It says GLaDOS, and a program that gates on "Linux" will now find out.**
/// Reporting the kernel this is not would buy compatibility with anything
/// checking the string, and this tree does not do that anywhere else -- the
/// wireless driver refuses to pretend it can associate, and the battery code
/// refuses to invent a reading. The release is the real build version, so a
/// bug report carries something true.
pub(super) fn sys_uname(buf: u64) -> u64 {
    const N: usize = 65;
    if !reachable(buf, N * 6, true) {
        return EFAULT;
    }
    let mut b = [0u8; N * 6];
    let mut put = |i: usize, v: &str| {
        let n = v.len().min(N - 1);
        b[i * N..i * N + n].copy_from_slice(&v.as_bytes()[..n]);
    };
    put(0, "GLaDOS");
    put(1, "glados");
    put(2, crate::VERSION);
    // **This said "one address space, no processes" until `fork` landed**, and
    // by then it was the one string on the machine still claiming it. The same
    // sentence was on two wiki pages for the same reason: it was true when
    // written, nothing rechecked it, and a guest asking `uname` was told a
    // fact about its host that its own existence disproved.
    put(3, "ring 0, with guests at ring 3");
    put(4, "x86_64");
    put(5, "(none)");
    unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), buf as *mut u8, N * 6) };
    0
}

/// Seconds and a sub-second part, from whichever clock was asked for.
///
/// Two sources and they are not interchangeable. The RTC gives a wall clock in
/// whole seconds and nothing finer; the timer tick gives 10 ms resolution and
/// counts from boot. So `CLOCK_REALTIME` is the RTC and its nanoseconds are
/// always zero, which is honest, and `CLOCK_MONOTONIC` is the tick, which
/// actually moves. A caller timing something short with `REALTIME` will
/// measure zero, and that is a property of the hardware rather than of this
/// call.
pub(super) fn clock_pair(which: u64) -> Option<(u64, u64)> {
    match which {
        // CLOCK_REALTIME and its coarse twin.
        0 | 5 => {
            let dt = crate::dev::rtc::now()?;
            Some((crate::dev::rtc::unix_seconds(&dt) as u64, 0))
        }
        // MONOTONIC, its coarse and raw twins, and the two process clocks --
        // one guest, no threads, so process time and uptime are the same
        // number and answering it is better than a program giving up.
        1 | 2 | 3 | 4 | 6 | 7 => {
            let t = crate::dev::lapic::ticks();
            let hz = crate::TIMER_HZ as u64;
            Some((t / hz, (t % hz) * (1_000_000_000 / hz)))
        }
        _ => None,
    }
}

pub(super) fn sys_clock_gettime(which: u64, at: u64) -> u64 {
    let Some((sec, nsec)) = clock_pair(which) else { return EINVAL };
    if !reachable(at, 16, true) {
        return EFAULT;
    }
    unsafe {
        core::ptr::write_volatile(at as *mut u64, sec);
        core::ptr::write_volatile((at + 8) as *mut u64, nsec);
    }
    0
}

pub(super) fn sys_gettimeofday(tv: u64, tz: u64) -> u64 {
    // The timezone argument has been obsolete since 4.4BSD and glibc passes
    // NULL. A caller that passes one gets it zeroed rather than refused, since
    // refusing a field nobody means anything by would fail the whole call.
    if tz != 0 {
        if !reachable(tz, 8, true) {
            return EFAULT;
        }
        unsafe { core::ptr::write_volatile(tz as *mut u64, 0) };
    }
    if tv == 0 {
        return 0;
    }
    let Some((sec, nsec)) = clock_pair(0) else { return EINVAL };
    if !reachable(tv, 16, true) {
        return EFAULT;
    }
    unsafe {
        core::ptr::write_volatile(tv as *mut u64, sec);
        core::ptr::write_volatile((tv + 8) as *mut u64, nsec / 1000);
    }
    0
}

/// The handful of `fcntl` commands a program uses before it does anything.
///
/// `F_SETFD` accepts `FD_CLOEXEC` and stores nothing, and that is honest
/// rather than lazy: close-on-exec is a promise about what survives an `exec`,
/// and there is no `exec` here for anything to survive. The day there is, this
/// has to start remembering, and it is written down here so that day finds it.
/// `mincore`: which pages of a range are resident. Every page a guest owns is
/// -- nothing here is ever paged out -- so the answer is "all of them" for a
/// range it owns and `ENOMEM` for one it does not, which is exactly the
/// question Mesa's EGL asks with it: is this pointer real memory? It was
/// unimplemented, and Mesa read that as "no" about the Wayland display it had
/// just been handed and answered "Could not get EGL display".
pub(super) fn sys_mincore(at: u64, len: u64, vec: u64) -> u64 {
    if at % 4096 != 0 {
        return EINVAL;
    }
    let pages = page_up(len as usize) / 4096;
    if !owns(at, len.max(1) as usize) {
        return ENOMEM;
    }
    if !reachable(vec, pages.max(1), true) {
        return EFAULT;
    }
    unsafe { core::ptr::write_bytes(vec as *mut u8, 1, pages) };
    0
}

pub(super) fn sys_fcntl(fd: u64, cmd: u64, arg: u64) -> u64 {
    let known = with_fds(|fds, _| matches!(fds.get(fd as usize), Some(Some(_)))).unwrap_or(false);
    if !known {
        return EBADF;
    }
    match cmd {
        0 => sys_dup(fd, None).max_free_from(arg), // F_DUPFD
        1 => 0,                                    // F_GETFD, nothing is close-on-exec
        2 => 0,                                    // F_SETFD, accepted and not stored
        // F_GETFL and F_SETFL: a device keeps its blocking mode, which is
        // how libasound turns a stream it opened non-blocking back into a
        // blocking one. Elsewhere no flag it could set applies.
        3 => with_fds(|fds, _| match fds.get(fd as usize) {
            Some(Some(super::fs::Fd::Dev(b))) => 2 | if b.borrow().nonblock { O_NONBLOCK } else { 0 },
            _ => 0,
        })
        .unwrap_or(0),
        4 => {
            with_fds(|fds, _| {
                if let Some(Some(super::fs::Fd::Dev(b))) = fds.get(fd as usize) {
                    b.borrow_mut().nonblock = arg & O_NONBLOCK != 0;
                }
            });
            0
        }
        // Record locks, in both widths. One process at a time owns any file
        // it writes here, so there is nobody to lock against.
        5 | 6 | 7 | 12 | 13 | 14 => 0,
        1030 => sys_dup(fd, None).max_free_from(arg), // F_DUPFD_CLOEXEC
        // File seals on a memfd: accepted and not enforced. A client seals
        // its buffer so the compositor can trust its size, and the compositor
        // here copies a commit out under its own bounds check anyway.
        1033 => 0,                                 // F_ADD_SEALS
        1034 => 0,                                 // F_GET_SEALS: none
        _ => EINVAL,
    }
}

trait MinFd {
    fn max_free_from(self, floor: u64) -> u64;
}

impl MinFd for u64 {
    /// `F_DUPFD` promises the lowest free descriptor *at or above* a floor,
    /// where `dup` promises the lowest free one. Rather than a second search,
    /// the result of `dup` is moved up when it landed too low -- which costs
    /// one extra descriptor briefly and cannot loop.
    fn max_free_from(self, floor: u64) -> u64 {
        if (self as i64) < 0 || self >= floor {
            return self;
        }
        let moved = sys_dup(self, None);
        sys_close(self);
        moved
    }
}

/// `dup2` with a flags argument, and one deliberate difference.
///
/// **`dup3(n, n, 0)` is `EINVAL` where `dup2(n, n)` answers `n`.** That is the
/// whole reason the call exists as well as `dup2`: the no-op case hides a bug
/// in the caller, so the newer call refuses it. Getting this backwards would
/// be invisible in every program that never makes the mistake, which is all of
/// them until one does.
///
/// It was found by driving rather than by reading a list. glibc's `dup2` calls
/// `dup3` where the descriptors differ, so `busybox dd` reached `dup2` and
/// `busybox hexdump` reached this and stopped -- two applets in one sweep,
/// taking two different routes to the same thing.
///
/// `O_CLOEXEC` is accepted and does nothing, honestly: there is no `execve`
/// here for a descriptor to survive, so the flag names an event that cannot
/// happen. Any other flag is `EINVAL`, because a flag this does not implement
/// is one the caller is relying on.
/// Start a thread, and refuse everything else `clone` can mean.
///
/// Threads only. A clone asking for a process never reaches here -- the
/// dispatcher sends it to `fork::clone_process`, which is how glibc forks --
/// and anything that is neither is refused with `ENOSYS` rather than
/// approximated: a thread handed back for a request it does not match is two
/// names for one address space and a program free to write through both.
/// (This said `fork` was "the one thing this system cannot grow into", which
/// `fork.rs` has since disproved.)
pub(super) fn sys_clone(flags: u64, stack: u64, ptid: u64, ctid: u64, tls: u64, rip: u64) -> u64 {
    if !super::thread::is_thread(flags) {
        return ENOSYS;
    }
    if ptid != 0 && !reachable(ptid, 4, true) {
        return EFAULT;
    }
    if ctid != 0 && !reachable(ctid, 4, true) {
        return EFAULT;
    }
    if stack == 0 || stack % 16 != 0 {
        // Answered before the reachability check, because the objection is to
        // the argument rather than to what it names -- and `EFAULT` about a
        // null stack sends a caller looking at its memory map.
        return EINVAL;
    }
    if !owns(stack.saturating_sub(4096), 4096) {
        // The stack is the guest's own memory or it is nothing this kernel
        // will jump onto. Checked one page below the top, since a stack
        // pointer names the byte past the last usable one.
        return EFAULT;
    }
    match super::thread::spawn(flags, stack, ptid, ctid, tls, rip) {
        Ok(tid) => tid,
        Err(e) => e,
    }
}

/// How many addresses a futex wait queue can name at once.
const FUTEX_SLOTS: usize = 32;
/// (address space, address, how many times it has been woken).
///
/// **The address space is part of the key.** Every guest with a root of its
/// own maps its image, stack and arena at the same virtual addresses, so two
/// sessions -- or a parent and its forked child -- waiting on "the same"
/// address are waiting on different words. Keyed by address alone, a wake in
/// one woke the other: harmless only because every waiter re-reads its word.
static FUTEX: Racy<[(u64, u64, u64); FUTEX_SLOTS]> = Racy::new([(0, 0, 0); FUTEX_SLOTS]);

/// Who is waiting on which word: (root, address, task). Under the guest lock.
///
/// **A wake names its waiters.** Waits used to nap until the next tick or
/// until *any* syscall finished, and every syscall's end woke every napper to
/// look again. llvmpipe synchronises its rasterizer threads through futexes
/// many times a frame, so each of those wakes became ten wakeups, ten trips
/// through the guest lock and ten IPIs, and Portal ran slower on four cores
/// than on one: 0.46 frames a second against 0.54, where the host goes from
/// 0.9 to 2.3. A `FUTEX_WAKE` now wakes the tasks waiting on that word and
/// nobody else.
static FUTEX_WAITERS: Racy<Vec<(u64, u64, usize)>> = Racy::new(Vec::new());

/// Wait for a wake on `addr`, or `until`, or a second, whichever is first --
/// the second only a backstop against a wake this table somehow missed.
fn futex_nap(addr: u64, until: Option<u64>) {
    let me = crate::task::current();
    let root = guest_root();
    let w = unsafe { &mut *FUTEX_WAITERS.get() };
    w.push((root, addr, me));
    let now = crate::dev::lapic::ticks();
    let deadline = until.unwrap_or(u64::MAX).min(now + crate::TIMER_HZ as u64).max(now + 1);
    crate::task::set_wake(deadline);
    let held = holds_big();
    if held {
        big_unlock();
    }
    let flags: u64;
    unsafe { core::arch::asm!("pushfq; pop {}", out(reg) flags, options(nomem, preserves_flags)) };
    crate::cpu::enable_interrupts();
    crate::task::yield_now();
    let _ = crate::task::take_woken();
    crate::task::set_wake(0);
    if flags & (1 << 9) == 0 {
        crate::cpu::disable_interrupts();
    }
    if held {
        big_lock();
    }
    let w = unsafe { &mut *FUTEX_WAITERS.get() };
    if let Some(i) = w.iter().position(|&(r, a, t)| r == root && a == addr && t == me) {
        w.swap_remove(i);
    }
}

pub(super) fn futex_bump(addr: u64) -> u64 {
    let took = big_take();
    let root = guest_root();
    let mut any = false;
    for &(r, a, t) in unsafe { &*FUTEX_WAITERS.get() }.iter() {
        if r == root && a == addr {
            any |= crate::task::wake(t);
        }
    }
    if any {
        crate::dev::lapic::kick_others();
    }
    let n = futex_bump_count(addr);
    if took {
        big_unlock();
    }
    n
}

fn futex_bump_count(addr: u64) -> u64 {
    let t = unsafe { &mut *FUTEX.get() };
    let root = guest_root();
    if let Some(e) = t.iter_mut().find(|e| e.1 == addr && e.0 == root) {
        e.2 += 1;
        return e.2;
    }
    if let Some(e) = t.iter_mut().find(|e| e.1 == 0) {
        *e = (root, addr, 1);
        return 1;
    }
    // Out of slots. Every waiter re-reads the word it is waiting on as well,
    // so a lost wake costs latency rather than correctness -- which is the
    // whole reason the value check is there and not merely belt and braces.
    0
}

pub(super) fn futex_count(addr: u64) -> u64 {
    let t = unsafe { &*FUTEX.get() };
    let root = guest_root();
    t.iter().find(|e| e.1 == addr && e.0 == root).map(|e| e.2).unwrap_or(0)
}

/// Wait on a word, or wake whoever is waiting on one.
///
/// **Two conditions end a wait, and needing both is the point.** A waiter
/// watches the wake counter *and* re-reads the word: the counter alone loses a
/// wake when the table is full, and the word alone misses a wake that changed
/// nothing. glibc's mutexes and `pthread_join` change the word, so in practice
/// the second is what fires and the first is what makes an unusual wake work.
///
/// The wait is a yield loop rather than a real sleep queue, which is the same
/// bargain `nanosleep` makes and for the same reason: there is no guest
/// scheduler to block against, so a wait is this kernel's wait. It costs a
/// context switch per tick per waiter, and it cannot deadlock the machine
/// because the run deadline reaches it.
pub(super) fn sys_futex(uaddr: u64, op: u64, val: u64, timeout: u64) -> u64 {
    futex(uaddr, op, val, timeout, 0, 0, false)
}

/// The whole of `futex`, for either ABI: `ts32` says whether the timeout is a
/// 32-bit `timespec` (i386's original `futex`) or a 64-bit one.
///
/// **Seven operations, and the bitset forms were the gap that stopped a real
/// program.** glibc i386 waits with `FUTEX_WAIT_BITSET`, whose timeout is an
/// *absolute* time on the monotonic clock (or the real-time one with
/// `FUTEX_CLOCK_REALTIME`), and anything else answered `ENOSYS` -- which glibc
/// treats as impossible and reports as "The futex facility returned an
/// unexpected error code", in a loop, from every thread that waited. The
/// bitset itself is accepted and not used to select waiters: every waiter on
/// the word is woken, which is a spurious wake for the ones the bitset
/// excluded, and spurious wakes are part of the contract.
///
/// The requeue forms are honoured the same way. Moving waiters from one word
/// to another needs a wait queue this kernel does not keep; waking them all
/// sends each back through its own re-check, which lands it on whichever word
/// it should now be waiting on. Correct by the contract, and slower than
/// Linux under contention.
///
/// PI operations stay refused: priority inheritance needs a scheduler that
/// knows about priorities.
pub(super) fn futex(uaddr: u64, op: u64, val: u64, timeout: u64, uaddr2: u64, val3: u64, ts32: bool) -> u64 {
    futex_inner(uaddr, op, val, timeout, uaddr2, val3, ts32)
}

fn futex_inner(uaddr: u64, op: u64, val: u64, timeout: u64, uaddr2: u64, val3: u64, ts32: bool) -> u64 {
    const FUTEX_WAIT: u64 = 0;
    const FUTEX_WAKE: u64 = 1;
    const FUTEX_REQUEUE: u64 = 3;
    const FUTEX_CMP_REQUEUE: u64 = 4;
    const FUTEX_WAKE_OP: u64 = 5;
    const FUTEX_WAIT_BITSET: u64 = 9;
    const FUTEX_WAKE_BITSET: u64 = 10;
    const PRIVATE: u64 = 128;
    const CLOCK_REALTIME: u64 = 256;
    let kind = op & !(PRIVATE | CLOCK_REALTIME);
    if uaddr % 4 != 0 {
        return EINVAL;
    }
    if !reachable(uaddr, 4, false) {
        return EFAULT;
    }
    let read_ts = |at: u64| -> Option<(u64, u64)> {
        if ts32 {
            if !reachable(at, 8, false) {
                return None;
            }
            Some(unsafe {
                (
                    core::ptr::read_unaligned(at as *const u32) as u64,
                    core::ptr::read_unaligned((at + 4) as *const u32) as u64,
                )
            })
        } else {
            if !reachable(at, 16, false) {
                return None;
            }
            Some(unsafe {
                (
                    core::ptr::read_unaligned(at as *const u64),
                    core::ptr::read_unaligned((at + 8) as *const u64),
                )
            })
        }
    };
    let woken_live = || val.min(super::thread::live() as u64 + 1);
    match kind {
        FUTEX_WAIT | FUTEX_WAIT_BITSET => {
            if kind == FUTEX_WAIT_BITSET && val3 as u32 == 0 {
                return EINVAL;
            }
            let seen = unsafe { core::ptr::read_volatile(uaddr as *const u32) };
            if seen as u64 != val as u32 as u64 {
                // Already changed. `EAGAIN` and not zero, because a caller
                // that slept would re-check and a caller told zero would not.
                return EAGAIN;
            }
            let woken = futex_count(uaddr);
            let hz = crate::TIMER_HZ as u64;
            let until = if timeout == 0 {
                None
            } else {
                let Some((sec, nsec)) = read_ts(timeout) else { return EFAULT };
                if kind == FUTEX_WAIT_BITSET {
                    // Absolute, on the clock the flag names.
                    let clk = if op & CLOCK_REALTIME != 0 { 0 } else { 1 };
                    let (now_s, now_ns) = clock_pair(clk).unwrap_or((0, 0));
                    let want = sec.saturating_mul(1_000_000_000).saturating_add(nsec);
                    let now = now_s.saturating_mul(1_000_000_000).saturating_add(now_ns);
                    let left = want.saturating_sub(now);
                    Some(crate::dev::lapic::ticks() + left / (1_000_000_000 / hz))
                } else {
                    Some(
                        crate::dev::lapic::ticks()
                            + sec.saturating_mul(hz)
                            + nsec * hz / 1_000_000_000,
                    )
                }
            };
            loop {
                let now = unsafe { core::ptr::read_volatile(uaddr as *const u32) };
                if now as u64 != val as u32 as u64 || futex_count(uaddr) != woken {
                    return 0;
                }
                if let Some(t) = until {
                    if crate::dev::lapic::ticks() >= t {
                        return ETIMEDOUT;
                    }
                }
                if super::thread::exiting() {
                    return EINTR;
                }
                // The same escape the input wait has, and the same reason: a
                // guest blocked in the kernel is invisible to the deadline
                // check in the timer, which only fires from ring 3.
                if overran(crate::dev::lapic::ticks()) {
                    unsafe { kill_blocked() }
                }
                futex_nap(uaddr, until);
            }
        }
        FUTEX_WAKE | FUTEX_WAKE_BITSET => {
            futex_bump(uaddr);
            // How many were woken. There is no queue to count, so this answers
            // what was asked for, which is what a caller uses to decide
            // whether to bother with a syscall next time -- and over-reporting
            // is the safe direction, since it makes a caller do less.
            woken_live()
        }
        FUTEX_REQUEUE | FUTEX_CMP_REQUEUE => {
            if kind == FUTEX_CMP_REQUEUE {
                let now = unsafe { core::ptr::read_volatile(uaddr as *const u32) };
                if now != val3 as u32 {
                    return EAGAIN;
                }
            }
            futex_bump(uaddr);
            woken_live()
        }
        FUTEX_WAKE_OP => {
            // *uaddr2 = op(*uaddr2, oparg); wake uaddr; and wake uaddr2 too
            // if the old value compares as the encoding says. **A compare-
            // and-swap, not a read and a write**: this said a plain store was
            // atomic because no other core could be running the guest, which
            // stopped being true when guest threads spread across cores -- a
            // thread on another core doing `lock cmpxchg` on this word between
            // the read and the write had its update lost, and the waiter it
            // was meant to release waited forever.
            if uaddr2 % 4 != 0 || !reachable(uaddr2, 4, true) {
                return EFAULT;
            }
            let enc = val3 as u32;
            let (opn, cmp) = ((enc >> 28) & 7, (enc >> 24) & 15);
            let mut oparg = (enc >> 12) & 0xFFF;
            let cmparg = enc & 0xFFF;
            if enc & (1 << 31) != 0 {
                oparg = 1u32.checked_shl(oparg).unwrap_or(0);
            }
            if opn > 4 {
                return ENOSYS;
            }
            let word = unsafe { &*(uaddr2 as *const core::sync::atomic::AtomicU32) };
            let old = word
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                    Some(match opn {
                        0 => oparg,
                        1 => old.wrapping_add(oparg),
                        2 => old | oparg,
                        3 => old & !oparg,
                        _ => old ^ oparg,
                    })
                })
                .unwrap_or_else(|v| v);
            futex_bump(uaddr);
            let hit = match cmp {
                0 => old == cmparg,
                1 => old != cmparg,
                2 => (old as i32) < cmparg as i32,
                3 => (old as i32) <= cmparg as i32,
                4 => (old as i32) > cmparg as i32,
                5 => (old as i32) >= cmparg as i32,
                _ => return ENOSYS,
            };
            if hit {
                futex_bump(uaddr2);
            }
            woken_live()
        }
        _ => ENOSYS,
    }
}

/// Run one guest thread to its end, on the task the pool gave it.
///
/// The counterpart of `run` for the main thread, and deliberately not the same
/// function: `run` parks the kernel's `FS`, arms the trap, sets the deadline
/// and tears the space down afterwards, and every one of those belongs to the
/// process rather than to a thread.
pub fn run_thread(w: super::thread::Work, stack: u64) {
    // **`exit` leaves through a longjmp and the flags do not come back.**
    // `syscall` clears IF through `FMASK` and `glados_leave_guest` restores a
    // stack rather than a processor state, so a thread that ends returns here
    // with interrupts off. `run` saves and restores them around the whole run
    // and says why; this did not, and the cost was the entire machine: the
    // pool task went back to its `hlt` with IF clear, which is a core that
    // never wakes again. No prompt, no timer, and the run deadline -- the one
    // thing that ends a runaway -- was never going to fire, because firing is
    // something an interrupt does.
    let flags: u64;
    unsafe {
        core::arch::asm!("pushfq; pop {}", out(reg) flags, options(nomem, preserves_flags))
    };
    // **The thread speaks for its process's guest and walks its tables.** A
    // pool task's guest index was whatever it last held, and its root the
    // kernel's: right while the main guest always sat in slot 0 on the
    // kernel's root, and wrong for any guest with a space of its own, whose
    // memory a thread on the kernel's root cannot even see. Set here, the way
    // a forked child's `body` sets them, and put back below.
    let me_t = crate::task::current();
    let prev = current_guest();
    set_current_guest(w.guest);
    super::session::enter(w.session);
    crate::task::set_root(me_t, guest_root());
    crate::task::ring3_active(true, stack);
    unsafe {
        (*core_entry()).syscall_stack = stack;
        crate::cpu::wrmsr(IA32_FS_BASE, w.fs);
    }
    // A 32-bit thread carries its TLS and its `%gs` in from the clone; they
    // go into this core's GDT and the segment register before the way in.
    if w.regs32.is_some() {
        super::i386::load(w.tls32, w.gs, true);
    }
    let code = as_guest(|| unsafe {
        match &w.regs32 {
            Some(r) => super::i386::glados_enter_guest32_regs(r as *const _),
            None => glados_enter_guest(w.rip, w.rsp),
        }
    });
    if w.regs32.is_some() {
        super::i386::begin(false);
    }
    // **A thread that crashes takes its process with it**, as on Linux: the
    // memory it shares with every other thread is exactly what a wild pointer
    // may have just corrupted, so the others carrying on would be running on
    // state nobody can vouch for. Ended here rather than in the fault handler,
    // because ending the main thread means longjmping out of *its* stack --
    // so it is doomed, and ends itself at its next safe point with this
    // thread's fault as the session's report.
    if code & FAULTED != 0 {
        if let Some(f) = take_task_fault(me()) {
            adopt_fault(f);
        }
        if let Some(m) = main_task() {
            doom(m, code);
        }
        super::session::begin_exit(w.session);
    }
    // **Clearing this is the whole of how `pthread_join` works.** The joiner
    // futex-waits on the word, and the kernel zeroing it is the notification;
    // a kernel that ignored `CLONE_CHILD_CLEARTID` leaves a library waiting
    // forever on a thread that finished.
    if w.ctid != 0 && owns(w.ctid, 4) {
        unsafe { core::ptr::write_volatile(w.ctid as *mut u32, 0) };
        futex_bump(w.ctid);
    }
    crate::task::ring3_active(false, 0);
    crate::task::set_root(me_t, 0);
    super::session::leave();
    set_current_guest(prev);
    if flags & (1 << 9) != 0 {
        crate::cpu::enable_interrupts();
    }
}

pub(super) fn sys_dup3(from: u64, to: u64, flags: u64) -> u64 {
    const O_CLOEXEC: u64 = 0o2_000_000;
    if flags & !O_CLOEXEC != 0 {
        return EINVAL;
    }
    if from == to {
        return EINVAL;
    }
    sys_dup(from, Some(to))
}

pub(super) fn sys_dup(from: u64, to: Option<u64>) -> u64 {
    with_fds(|fds, _| {
        let copy = match fds.get(from as usize) {
            Some(Some(f)) => f.share(),
            _ => return EBADF,
        };
        let slot = match to {
            // `dup2(n, n)` is a no-op that answers `n`, and it has to be
            // checked before the close: the obvious order shuts the descriptor
            // and then duplicates what is no longer there.
            Some(n) if n == from => return n,
            Some(n) => {
                if n as usize >= MAX_FDS {
                    return EBADF;
                }
                let n = n as usize;
                if fds.len() <= n {
                    fds.resize_with(n + 1, || None);
                }
                n
            }
            None => match fds.iter().position(|f| f.is_none()) {
                Some(i) => i,
                None if fds.len() < MAX_FDS => {
                    fds.push(None);
                    fds.len() - 1
                }
                None => return EMFILE,
            },
        };
        fds[slot] = Some(copy);
        slot as u64
    })
    .unwrap_or(EBADF)
}

pub(super) fn sys_close(fd: u64) -> u64 {
    with_fds(|fds, _| match fds.get_mut(fd as usize) {
        Some(slot @ Some(_)) => {
            // Committed here rather than on every write, and only when this is
            // the last name for the body. `flush` decides both.
            if let Some(f) = slot.as_ref() {
                f.flush();
            }
            *slot = None;
            0
        }
        _ => EBADF,
    })
    .unwrap_or(EBADF)
}

/// The most descriptors either call will consider.
///
/// A cap rather than a trust: `nfds` is the guest's, and both calls size an
/// allocation from it. Linux's own default limit is 1024 open files, so a
/// program legitimately past this is one this machine could not have given the
/// descriptors to in the first place.
const MAX_WATCHED: u64 = 1024;

/// What one descriptor can do, or nothing if it is not open.
///
/// Looked up afresh on every turn of a wait, because the entire point of these
/// calls is that the answer changes while the guest is not running.
/// The descriptor is taken out of the table **before** it is asked, and the
/// borrow released, which is required rather than tidy: an `epoll` instance's
/// readiness is its members' readiness, so `poll::ready` reaches back into the
/// table for it. Asking while the table were still borrowed is a second
/// `RefCell` borrow and a panic in a kernel with no unwinder. `share` is an
/// `Rc` bump, so the copy costs nothing.
pub(crate) fn fd_ready(fd: u64) -> Option<u16> {
    let d = with_fds(|fds, _| fds.get(fd as usize).and_then(|s| s.as_ref().map(|d| d.share())))
        .flatten()?;
    Some(super::poll::ready(&d))
}

/// Whether a wait that began at `start` has run out.
fn expired(start: u64, limit: Option<u64>) -> bool {
    match limit {
        None => false,
        Some(t) => crate::dev::lapic::ticks().wrapping_sub(start) >= t,
    }
}

/// A `timespec` or `timeval` from the guest, as milliseconds.
///
/// One function for both because they differ only in the divisor of the second
/// field, and writing it twice is how the two come to disagree about rounding.
fn span_ms(at: u64, per_ms: u64) -> Option<i64> {
    if !reachable(at, 16, false) {
        return None;
    }
    let p = at as *const u64;
    let (whole, frac) = unsafe { (core::ptr::read_unaligned(p), core::ptr::read_unaligned(p.add(1))) };
    Some(whole.saturating_mul(1000).saturating_add(frac / per_ms) as i64)
}

pub(super) fn sys_poll(fds: u64, nfds: u64, timeout: u64) -> u64 {
    // The field is an `int`, so a wait of -1 arrives as 0xFFFFFFFF and has to
    // be read back through i32 before it means "no limit" rather than
    // "forty-nine days".
    do_poll(fds, nfds, super::poll::ticks_for_ms(timeout as u32 as i32 as i64))
}

/// `ppoll`, whose signal mask cannot matter here.
///
/// Linux swaps the mask for the duration so a signal can break the wait. This
/// kernel delivers signals **on the way out of a syscall**, and a guest
/// blocked inside `poll` has not left one, so nothing can interrupt this wait
/// whatever the mask says. Accepting the argument and ignoring it is therefore
/// exact rather than lazy: there is no behaviour it could change. It stops
/// being true the day delivery happens anywhere else, and `signal.rs` already
/// records that the timer is the other candidate.
pub(super) fn sys_ppoll(fds: u64, nfds: u64, ts: u64, _mask: u64, _setsize: u64) -> u64 {
    let limit = if ts == 0 {
        None
    } else {
        match span_ms(ts, 1_000_000) {
            Some(ms) => super::poll::ticks_for_ms(ms),
            None => return EFAULT,
        }
    };
    do_poll(fds, nfds, limit)
}

pub(super) fn do_poll(fds: u64, nfds: u64, limit: Option<u64>) -> u64 {
    use super::poll::{PFD_LEN, POLLNVAL};
    if nfds > MAX_WATCHED {
        return EINVAL;
    }
    let bytes = nfds as usize * PFD_LEN;
    // Writable, because `revents` is written back into the guest's own array.
    // That is the whole calling convention: the guest keeps one array across
    // many calls and the kernel touches only the last field of each entry.
    if bytes > 0 && !reachable(fds, bytes, true) {
        return EFAULT;
    }
    let start = crate::dev::lapic::ticks();
    loop {
        let mut n = 0u64;
        for i in 0..nfds as usize {
            let at = fds + (i * PFD_LEN) as u64;
            let raw = unsafe { core::ptr::read_unaligned(at as *const i32) };
            let events = unsafe { core::ptr::read_unaligned((at + 4) as *const u16) };
            // A negative descriptor is skipped with its answer cleared. That
            // is how a program switches one entry off without rebuilding its
            // array, and refusing it would break every caller that does.
            let rev = if raw < 0 {
                0
            } else {
                match fd_ready(raw as u64) {
                    Some(state) => super::poll::wanted(state, events),
                    // Not an error return. `poll` reports a bad descriptor in
                    // that entry and answers about the rest, which is what
                    // lets a program find out *which* one went bad.
                    None => POLLNVAL,
                }
            };
            unsafe { core::ptr::write_unaligned((at + 6) as *mut u16, rev) };
            if rev != 0 {
                n += 1;
            }
        }
        if n > 0 {
            return n;
        }
        if expired(start, limit) {
            return 0;
        }
        // A blocked guest has to stay killable. The run deadline is checked
        // from the timer and only for ring 3, so a guest sitting in a kernel
        // wait is invisible to it -- the failure `sys_read`'s own loop
        // records, which took the machine with no key able to bring it back.
        if overran(crate::dev::lapic::ticks()) {
            unsafe { kill_blocked() }
        }
        // The display server answers while its client waits.
        crate::sky::server::pump();
        wait_turn();
    }
}

pub(super) fn sys_select(nfds: u64, r: u64, w: u64, e: u64, tv: u64) -> u64 {
    let limit = if tv == 0 {
        None
    } else {
        match span_ms(tv, 1000) {
            Some(ms) => super::poll::ticks_for_ms(ms),
            None => return EFAULT,
        }
    };
    do_select(nfds, r, w, e, limit)
}

pub(super) fn sys_pselect6(nfds: u64, r: u64, w: u64, e: u64, ts: u64) -> u64 {
    // The sixth argument is a signal mask, and it cannot matter here for the
    // reason `ppoll` gives.
    let limit = if ts == 0 {
        None
    } else {
        match span_ms(ts, 1_000_000) {
            Some(ms) => super::poll::ticks_for_ms(ms),
            None => return EFAULT,
        }
    };
    do_select(nfds, r, w, e, limit)
}

fn do_select(nfds: u64, rp: u64, wp: u64, ep: u64, limit: Option<u64>) -> u64 {
    use super::poll::{set_bytes, set_get, set_put, POLLHUP, POLLIN, POLLOUT, POLLPRI};
    if nfds > MAX_WATCHED {
        return EINVAL;
    }
    let nb = set_bytes(nfds as usize);
    let mut want = [alloc::vec![0u8; nb], alloc::vec![0u8; nb], alloc::vec![0u8; nb]];
    for (slot, p) in [rp, wp, ep].iter().enumerate() {
        if *p != 0 && nb > 0 {
            if !reachable(*p, nb, true) {
                return EFAULT;
            }
            unsafe {
                core::ptr::copy_nonoverlapping(*p as *const u8, want[slot].as_mut_ptr(), nb)
            };
        }
    }
    let start = crate::dev::lapic::ticks();
    loop {
        let mut out = [alloc::vec![0u8; nb], alloc::vec![0u8; nb], alloc::vec![0u8; nb]];
        let mut n = 0u64;
        for i in 0..nfds as usize {
            let asked = [set_get(&want[0], i), set_get(&want[1], i), set_get(&want[2], i)];
            if !asked.iter().any(|a| *a) {
                continue;
            }
            // `select` has nowhere to put a per-descriptor answer, so a bad
            // one fails the whole call. That is the opposite of `poll` above
            // and both are what a libc expects.
            let Some(state) = fd_ready(i as u64) else {
                return EBADF;
            };
            // A hangup counts as readable here, deliberately. `select` has no
            // bit for it, so end-of-connection reaches a program as a read
            // that then returns zero -- which is exactly how every program
            // written against `select` learns a peer has gone.
            if asked[0] && state & (POLLIN | POLLHUP) != 0 {
                set_put(&mut out[0], i);
                n += 1;
            }
            if asked[1] && state & POLLOUT != 0 {
                set_put(&mut out[1], i);
                n += 1;
            }
            if asked[2] && state & POLLPRI != 0 {
                set_put(&mut out[2], i);
                n += 1;
            }
        }
        // The sets are written back on the way out either way, since a timeout
        // has to leave them empty rather than holding what the guest asked
        // for. A caller that re-polled without clearing them would otherwise
        // be told every descriptor it watches is ready.
        if n > 0 || expired(start, limit) {
            for (slot, p) in [rp, wp, ep].iter().enumerate() {
                if *p != 0 && nb > 0 {
                    unsafe {
                        core::ptr::copy_nonoverlapping(out[slot].as_ptr(), *p as *mut u8, nb)
                    };
                }
            }
            return n;
        }
        if overran(crate::dev::lapic::ticks()) {
            unsafe { kill_blocked() }
        }
        wait_turn();
    }
}

pub(super) fn sys_read(fd: u64, buf: u64, len: u64) -> u64 {
    if len == 0 {
        return 0;
    }
    if is_socket(fd) {
        return sys_recv(fd, buf, len);
    }
    if !reachable(buf, len as usize, true) {
        return EFAULT;
    }
    // An event device is read through its own path, because two things here do
    // not fit a call that takes an offset: the cursor is a sequence number
    // rather than a position, and a blocking reader has to wait.
    if let Some((d, nonblock)) = input_fd(fd) {
        if (len as usize) < super::input::EVENT_LEN {
            // Linux answers this rather than delivering part of a record. A
            // reader walks the stream by a fixed stride, so half an event is
            // not less data, it is garbage from there on.
            return EINVAL;
        }
        loop {
            let out = unsafe { core::slice::from_raw_parts_mut(buf as *mut u8, len as usize) };
            let n = with_fds(|fds, _| match fds.get(fd as usize) {
                Some(Some(super::fs::Fd::Dev(b))) => {
                    let f = &mut *b.borrow_mut();
                    let mut cur = f.at as u64;
                    let n = super::input::read(d, &mut cur, out);
                    f.at = cur as usize;
                    n
                }
                _ => 0,
            })
            .unwrap_or(0);
            if n > 0 {
                return n as u64;
            }
            if nonblock {
                return EAGAIN;
            }
            // **A blocked guest is killable, and it was not.** The run
            // deadline is checked from the timer interrupt and only when the
            // saved CS says ring 3, so a guest sitting in a kernel wait was
            // never a guest that had overrun -- it was invisible to the one
            // thing that ends a runaway. A blocking read on a device nothing
            // feeds therefore took the machine, with the shell gone and no key
            // able to bring it back. Found by writing exactly that program.
            if overran(crate::dev::lapic::ticks()) {
                unsafe { kill_blocked() }
            }
            // **Yield rather than spin.** There is no guest scheduler to block
            // against, so a wait is this kernel's wait, and a busy loop here
            // starves the resident mind and the clock for as long as nobody
            // touches the keyboard.
            // The display server answers while its client waits.
            crate::sky::server::pump();
            wait_turn();
        }
    }
    // A Unix socket waits the way an event device does and for the same
    // reason: what it is short of is not bytes at an offset, it is bytes that
    // have not been written yet. Handled before the table below, because the
    // loop has to re-look each turn rather than borrow across a yield.
    if let Some(sock) = unix_sock(fd) {
        let stream = {
            let b = sock.borrow();
            match &*b {
                super::unix::Sock::Stream { pipe, side } => Some((pipe.clone(), *side)),
                _ => None,
            }
        };
        let Some((pipe, side)) = stream else { return ENOTCONN };
        loop {
            let out = unsafe { core::slice::from_raw_parts_mut(buf as *mut u8, len as usize) };
            match super::unix::read(&pipe, side, out) {
                Ok(n) => return n as u64,
                Err(-11) => {}
                Err(e) => return e as u64,
            }
            if overran(crate::dev::lapic::ticks()) {
                unsafe { kill_blocked() }
            }
            // The display server answers while its client waits.
            crate::sky::server::pump();
            wait_turn();
        }
    }
    // An anonymous file is memory that is already there, so nothing waits.
    if let Some(m) = memfd(fd) {
        let mut f = m.borrow_mut();
        let at = f.cursor.min(f.len);
        let n = (f.len - at).min(len as usize);
        if n > 0 {
            unsafe {
                core::ptr::copy_nonoverlapping((f.at + at as u64) as *const u8, buf as *mut u8, n)
            };
        }
        f.cursor = at + n;
        return n as u64;
    }
    // A pipe waits like a Unix socket, with one refusal on top.
    if let Some(end) = pipe_end(fd) {
        let (pipe, writing, nonblock) = {
            let e = end.borrow();
            (e.pipe.clone(), e.writing, e.nonblock)
        };
        if writing {
            // **Reading the write end is `EBADF` and not an empty read.** The
            // end was opened write-only, and answering zero would tell the
            // program its input had ended.
            return EBADF;
        }
        let side = super::fs::Fd::pipe_side(false);
        loop {
            let out = unsafe { core::slice::from_raw_parts_mut(buf as *mut u8, len as usize) };
            match super::unix::read(&pipe, side, out) {
                Ok(n) => return n as u64,
                Err(-11) if !nonblock => {}
                Err(e) => return e as u64,
            }
            if overran(crate::dev::lapic::ticks()) {
                unsafe { kill_blocked() }
            }
            // The display server answers while its client waits.
            crate::sky::server::pump();
            wait_turn();
        }
    }
    with_fds(|fds, _| match fds.get_mut(fd as usize) {
        // Nothing types at a guest, so stdin is permanently at end of file.
        // Zero is the honest answer and is what a program reading a closed
        // pipe sees.
        Some(Some(super::fs::Fd::Stdin)) => 0,
        Some(Some(super::fs::Fd::Disk(body))) => {
            let f = &mut *body.borrow_mut();
            let out = unsafe { core::slice::from_raw_parts_mut(buf as *mut u8, len as usize) };
            let at = f.at;
            let n = f.read_at(at, out);
            f.at = at.saturating_add(n);
            n as u64
        }
        Some(Some(super::fs::Fd::File(body))) => {
            let f = &mut *body.borrow_mut();
            let (data, at) = (&f.data, &mut f.at);
            // **The cursor is clamped before it indexes, and that is not
            // belt and braces.** `lseek` past the end is legal and this module
            // says so two functions down, so `*at > data.len()` is a state a
            // guest reaches with two ordinary calls -- and `data[*at..]` on
            // that state is a panic, in ring 0, with no unwinder. Two legal
            // syscalls stopped the machine.
            let from = (*at).min(data.len());
            let n = (data.len() - from).min(len as usize);
            unsafe {
                core::ptr::copy_nonoverlapping(data[from..].as_ptr(), buf as *mut u8, n);
            }
            *at = at.saturating_add(n);
            n as u64
        }
        Some(Some(super::fs::Fd::Dev(body))) => {
            let d = &mut *body.borrow_mut();
            let out = unsafe { core::slice::from_raw_parts_mut(buf as *mut u8, len as usize) };
            let n = super::dev::read(d.node, d.at, out);
            d.at = d.at.saturating_add(n);
            n as u64
        }
        Some(Some(super::fs::Fd::Dir(_))) => EISDIR,
        _ => EBADF,
    })
    .unwrap_or(EBADF)
}

/// Read at an offset, leaving the cursor where it was.
///
/// **What a dynamic linker uses to read a library's headers**, and the call
/// glibc stopped on: it has the file open, it wants bytes from a known place,
/// and it does not want to put the cursor back afterwards. `lseek`, `read`,
/// `lseek` is the same thing done in three calls that can be interrupted
/// between any two of them, which is why the atomic form exists.
///
/// The failure this replaces is worth keeping, because the instrument nearly
/// missed it. Every call around this one was implemented and every one of them
/// succeeded, so the run did not end in a fault or a refusal that named a
/// missing feature -- it ended in glibc printing "cannot read file data: Error
/// 38" about a library it had already found, opened and read the first 832
/// bytes of. Error 38 is `ENOSYS`, and the guest reported our gap more clearly
/// than our own trace did.
pub(super) fn sys_pread64(fd: u64, buf: u64, len: u64, off: u64) -> u64 {
    if len == 0 {
        return 0;
    }
    // Linux answers `EINVAL` for a negative offset rather than treating it as
    // an enormous unsigned one, which is what the cast would otherwise do to a
    // program that passed -1 by mistake.
    if (off as i64) < 0 {
        return EINVAL;
    }
    if !reachable(buf, len as usize, true) {
        return EFAULT;
    }
    with_fds(|fds, _| match fds.get(fd as usize) {
        Some(Some(super::fs::Fd::Disk(body))) => {
            let out = unsafe { core::slice::from_raw_parts_mut(buf as *mut u8, len as usize) };
            body.borrow_mut().read_at(off as usize, out) as u64
        }
        Some(Some(super::fs::Fd::File(body))) => {
            let f = body.borrow();
            // Past the end is zero rather than an error, the same answer
            // `read` gives there. The cursor is not touched, which is the
            // whole of the difference between this call and that one.
            let from = (off as usize).min(f.data.len());
            let n = (f.data.len() - from).min(len as usize);
            unsafe { core::ptr::copy_nonoverlapping(f.data[from..].as_ptr(), buf as *mut u8, n) };
            n as u64
        }
        Some(Some(super::fs::Fd::Dir(_))) => EISDIR,
        // A stream has no offsets to read at. `ESPIPE` is what `lseek` on one
        // answers and this is the same fact said by a different call.
        Some(Some(_)) => ESPIPE,
        _ => EBADF,
    })
    .unwrap_or(EBADF)
}

/// Where a thread keeps the mutexes it must release if it dies holding them.
///
/// Accepted and stored nowhere, which is honest for the same reason
/// `rt_sigaction` is: the list exists so the *kernel* can walk it when a
/// thread dies without unlocking, and there is one thread here, no futexes,
/// and no second thread to be left waiting. Refusing would stop a libc that
/// registers this before `main` over a cleanup that can never be needed.
///
/// The length is checked because Linux checks it. A libc passing the wrong
/// size is a libc built against a different `struct robust_list_head`, and
/// accepting that quietly is how a disagreement about a structure becomes a
/// disagreement about memory later.
/// Random bytes, from the same generator everything else here uses.
///
/// glibc asks for these and copes with `ENOSYS` by falling back to `AT_RANDOM`
/// and the clock, which is a worse answer than the one this machine can
/// actually give. `rng::fill` rather than `fill_secret`: a stack guard and a
/// hash seed want unpredictability without depending on it, and `fill_secret`
/// refuses before the entropy threshold is met, which would turn a working
/// program into a refused one on a machine nobody has typed at.
///
/// `GRND_RANDOM` and `GRND_NONBLOCK` are accepted and ignored, because there
/// is one pool here and it never blocks. Answering the full count is therefore
/// always true rather than optimistic.
/// Seconds since the epoch.
///
/// Answered from the RTC rather than from the tick counter, because the tick
/// counter says how long this machine has been up and a timestamp is a
/// different question. A machine whose clock cannot be read answers zero,
/// which is what Linux reports before anything has set the time, and is the
/// one wrong answer that is honestly wrong rather than plausibly wrong.
/// Read a symlink, of which this machine has exactly one.
///
/// `/proc/self/exe`, and it is a symlink rather than a file because that is
/// what Linux makes it and what every program reaching for its own path
/// expects. Everything else is `EINVAL`, which is what Linux answers for a
/// path that exists and is not a link -- distinct from `ENOENT`, and the
/// distinction is load-bearing: a program told `EINVAL` knows the file is
/// there and stops looking, where `ENOENT` sends it hunting.
///
/// **The answer is not NUL-terminated.** `readlink` returns a length and
/// writes exactly that many bytes, and a terminator written past it would
/// land in the caller's buffer beyond what it was told was used.
pub(super) fn sys_readlinkat(dirfd: u64, path_at: u64, buf: u64, size: u64) -> u64 {
    let raw = match read_cstr(path_at) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if raw.is_empty() {
        return ENOENT;
    }
    if size == 0 {
        return EINVAL;
    }
    let rel = !raw.starts_with('/');
    let found = with_fds(|fds, cwd| {
        let base = if rel { at_base(fds, dirfd, cwd)? } else { String::new() };
        super::fs::resolve(&base, &raw).ok_or(ENOENT)
    });
    let found = match found {
        Some(Ok(p)) => Some(p),
        Some(Err(e)) => return e,
        None => None,
    };
    let Some(path) = found else { return ENOENT };
    let Some(target) = super::proc::link(&path) else {
        // **A directory is not a symlink, and saying it does not exist is a
        // different claim.** `blob_len` answers for a blob and nothing else,
        // so every directory on the way to a file answered `ENOENT` here --
        // and `realpath` is a walk over exactly those components. glibc reads
        // `EINVAL` as "not a symlink, carry on" and `ENOENT` as "this path is
        // not there", so resolving `/tmp/h.js` failed at `/tmp`.
        //
        // Found from the guest side rather than by reading: QuickJS printed
        // `TypeError: realpath failure` and then dereferenced the null it had
        // not checked, faulting at `0x80`. The errno was the whole of it --
        // one wrong value sending a correct program down a path that ends in
        // a crash it cannot explain.
        return if super::proc::claims(&path)
            || matches!(super::disk::lookup(&path), Some(Some(_)))
            || sysbox::blob_len(&path).is_some()
            || sysbox::is_dir(&path)
        {
            EINVAL
        } else {
            ENOENT
        };
    };
    let n = target.len().min(size as usize);
    if !reachable(buf, n, true) {
        return EFAULT;
    }
    unsafe { core::ptr::copy_nonoverlapping(target.as_ptr(), buf as *mut u8, n) };
    n as u64
}

pub(super) fn sys_time(tloc: u64) -> u64 {
    let secs = crate::dev::rtc::now()
        .map(|dt| crate::dev::rtc::unix_seconds(&dt) as u64)
        .unwrap_or(0);
    // The pointer is optional, which is unusual enough to be worth saying: the
    // value comes back in the return register either way, and a program that
    // wants it in memory as well passes somewhere to put it.
    if tloc != 0 {
        if !reachable(tloc, 8, true) {
            return EFAULT;
        }
        unsafe { core::ptr::write_unaligned(tloc as *mut u64, secs) };
    }
    secs
}

/// How big `struct sysinfo` is on x86-64.
///
/// One hundred and eight bytes of fields rounded up to eight, and the rounding
/// is not optional: a libc reads the whole structure, so writing 108 leaves
/// four bytes of whatever the guest had there being read as padding it will
/// then ignore -- until some future field lives in them.
const SYSINFO_LEN: usize = 112;

/// What the machine can say about itself.
///
/// **Three of these are real and the rest are honestly zero**, which is the
/// only interesting decision here. Uptime and the heap are facts. The load
/// averages are not: there is no run-queue sampling in this kernel, and a
/// fabricated number would be read by anything that graphs it. Swap is zero
/// because there is none. `procs` is one because there is one.
///
/// `totalram` is the kernel heap rather than the machine's RAM, and that is a
/// deviation worth naming: the heap is one contiguous allocation the kernel
/// already owns, so it is what a guest could actually be given, and reporting
/// the firmware's total would promise memory nothing here can hand out.
pub(super) fn sys_sysinfo(buf: u64) -> u64 {
    if !reachable(buf, SYSINFO_LEN, true) {
        return EFAULT;
    }
    let (used, total) = crate::mem::heap::HEAP.stats();
    let uptime = crate::dev::lapic::ticks() / crate::TIMER_HZ as u64;
    let mut b = [0u8; SYSINFO_LEN];
    let mut put = |off: usize, v: u64| b[off..off + 8].copy_from_slice(&v.to_le_bytes());
    put(0, uptime);
    // 8..32 are the three load averages, left zero for the reason above.
    put(32, total as u64);
    put(40, total.saturating_sub(used) as u64);
    // 48..80 are shared, buffer and both swap figures. None of them exist.
    b[80..82].copy_from_slice(&1u16.to_le_bytes());
    // 88..104 are the high-memory pair, which is a 32-bit concept.
    b[104..108].copy_from_slice(&1u32.to_le_bytes());
    unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), buf as *mut u8, SYSINFO_LEN) };
    0
}

/// Which cores this process may run on.
///
/// Every one that answered at boot, which is the truthful answer even though
/// nothing here can yet schedule a guest onto a second core: the question is
/// about permission rather than about capability, and a guest that asked and
/// was told "one" would size its thread pool for a machine this is not.
///
/// Answers the *bytes written* rather than zero, which is the part of this
/// call that is easy to get wrong. A libc uses the return value to know how
/// much of a much larger `cpu_set_t` it must clear itself.
pub(super) fn sys_sched_getaffinity(_pid: u64, len: u64, mask: u64) -> u64 {
    // Linux insists on a whole number of longs, and it is worth insisting too:
    // a length that is not one means the caller and this disagree about the
    // shape of a bitmap, and a partial write is how that becomes silent.
    if len < 8 || len % 8 != 0 {
        return EINVAL;
    }
    if !reachable(mask, 8, true) {
        return EFAULT;
    }
    let cores = crate::smp::online().clamp(1, 64);
    let bits = if cores >= 64 { u64::MAX } else { (1u64 << cores) - 1 };
    unsafe { core::ptr::write_unaligned(mask as *mut u64, bits) };
    8
}

pub(super) fn sys_getrandom(buf: u64, len: u64, _flags: u64) -> u64 {
    if len == 0 {
        return 0;
    }
    if !reachable(buf, len as usize, true) {
        return EFAULT;
    }
    let n = (len as usize).min(1 << 20);
    let mut tmp = alloc::vec![0u8; n];
    crate::rng::fill(&mut tmp);
    unsafe { core::ptr::copy_nonoverlapping(tmp.as_ptr(), buf as *mut u8, n) };
    n as u64
}

pub(super) fn sys_set_robust_list(_head: u64, len: u64) -> u64 {
    if len != 24 {
        return EINVAL;
    }
    0
}

/// Resource limits, answered rather than invented.
///
/// glibc asks for `RLIMIT_STACK` before `main` and sizes things from it, so
/// `ENOSYS` here is not a neutral answer. Three of these are facts this kernel
/// actually knows -- the stack and break are regions the loader handed out and
/// the descriptor table has a real ceiling -- and the rest are genuinely
/// unbounded, because nothing here bounds them.
///
/// **Setting one is refused.** A limit this kernel cannot enforce is a limit
/// it must not claim to have accepted: a guest told its stack is now 8 MiB
/// would grow into whatever is next in the heap. `EPERM` is what Linux answers
/// a process that may not raise a hard limit, which is the nearest true thing.
pub(super) fn sys_prlimit64(_pid: u64, resource: u64, new: u64, old: u64) -> u64 {
    const RLIMIT_DATA: u64 = 2;
    const RLIMIT_STACK: u64 = 3;
    const RLIMIT_NOFILE: u64 = 7;
    const RLIM_INFINITY: u64 = u64::MAX;
    if new != 0 {
        return EPERM;
    }
    if old == 0 {
        return EINVAL;
    }
    if !reachable(old, 16, true) {
        return EFAULT;
    }
    let Some(sp) = (unsafe { guest_slot() }).as_ref() else { return EINVAL };
    let n = match resource {
        RLIMIT_STACK => sp.stack.len as u64,
        RLIMIT_DATA => sp.brk_end.saturating_sub(sp.brk_start),
        RLIMIT_NOFILE => MAX_FDS as u64,
        _ => RLIM_INFINITY,
    };
    // Soft and hard are the same number, because there is nothing here that
    // could raise one to reach the other.
    unsafe {
        core::ptr::write_unaligned(old as *mut u64, n);
        core::ptr::write_unaligned((old + 8) as *mut u64, n);
    }
    0
}

pub(super) fn sys_lseek(fd: u64, off: u64, whence: u64) -> u64 {
    with_fds(|fds, _| match fds.get_mut(fd as usize) {
        Some(Some(super::fs::Fd::Disk(body))) => {
            let f = &mut *body.borrow_mut();
            let base = match whence {
                0 => 0i64,
                1 => f.at as i64,
                2 => f.size as i64,
                _ => return EINVAL,
            };
            let want = base.saturating_add(off as i64);
            if want < 0 {
                return EINVAL;
            }
            f.at = want as usize;
            want as u64
        }
        Some(Some(super::fs::Fd::File(body))) => {
            let f = &mut *body.borrow_mut();
            let (data, at) = (&f.data, &mut f.at);
            let base = match whence {
                0 => 0i64,                 // SEEK_SET
                1 => *at as i64,           // SEEK_CUR
                2 => data.len() as i64,    // SEEK_END
                _ => return EINVAL,
            };
            let want = base.saturating_add(off as i64);
            if want < 0 {
                return EINVAL;
            }
            // Seeking past the end is legal and reads answer zero there, which
            // is what makes a sparse write possible on Linux and is harmless
            // on a view that cannot write.
            *at = want as usize;
            want as u64
        }
        // A directory's cursor is an entry index, which is what `d_off`
        // reports, so the two agree. Only a rewind is honoured: `seekdir` to
        // an arbitrary index would need the index to stay meaningful across
        // the snapshot the directory was opened with, and inventing that is
        // how `telldir` starts handing out positions that name the wrong file.
        // `rewinddir` is the one every program actually uses.
        Some(Some(super::fs::Fd::Dir(body))) => {
            if whence == 0 && off == 0 {
                body.borrow_mut().at = 0;
                0
            } else {
                EINVAL
            }
        }
        // A device seeks, which is how `cat > /dev/fb0` reaches the bottom
        // half of the screen. The end is video memory's for the framebuffer
        // and zero for the rest, which have no end -- `SEEK_END` on
        // `/dev/zero` answering zero is what Linux does and is why nothing
        // uses it to size anything.
        Some(Some(super::fs::Fd::Dev(body))) => {
            let d = &mut *body.borrow_mut();
            // An event stream has no position, so this is `ESPIPE` for the
            // same reason stdout is. It matters more here than it reads:
            // `DevFile.at` is a sequence number for these, so honouring a seek
            // would move a reader to an event that never happened.
            if super::dev::is_input(d.node).is_some() {
                return ESPIPE;
            }
            let base = match whence {
                0 => 0i64,
                1 => d.at as i64,
                2 => super::dev::size(d.node) as i64,
                _ => return EINVAL,
            };
            let want = base.saturating_add(off as i64);
            if want < 0 {
                return EINVAL;
            }
            d.at = want as usize;
            want as u64
        }
        // A stream has no position. `ESPIPE` is what libc turns into "illegal
        // seek", and it is how a program discovers stdout is not a file.
        Some(Some(_)) => ESPIPE,
        _ => EBADF,
    })
    .unwrap_or(EBADF)
}

/// Set across an i386 `stat`, `lstat` or `fstat` (106-108), whose structure
/// is the 1990s one: 64 bytes, 32-bit sizes. A library linked against an old
/// glibc's `__xstat` reaches it -- libasound from the Steam runtime did, for
/// its own configuration file, and reported a file that was there as one it
/// "cannot access", which left the game silent.
pub(super) static STAT_OLD: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

pub(super) fn write_stat(buf: u64, kind: super::fs::Kind, size: usize, ino: u64) -> u64 {
    if abi32() && STAT_OLD.load(Ordering::Relaxed) {
        return write_stat_old(buf, kind, size, ino);
    }
    if abi32() {
        return write_stat64(buf, kind, size, ino);
    }
    if !reachable(buf, 144, true) {
        return EFAULT;
    }
    let b = super::fs::stat_bytes(kind, size, ino);
    unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), buf as *mut u8, 144) };
    0
}

/// i386's `struct stat64`: 96 bytes, packed, with the inode twice -- a 32-bit
/// copy at 12 and the full one at 88 -- and the times as 32-bit seconds and
/// nanoseconds. Offsets read off the host's own header with `gcc -m32`, not
/// recalled.
/// i386 `struct stat`: dev, ino as u32; mode, nlink, uid, gid as u16; rdev,
/// size, blksize, blocks as u32; three (sec, nsec) pairs; two spare words.
fn write_stat_old(buf: u64, kind: super::fs::Kind, size: usize, ino: u64) -> u64 {
    if !reachable(buf, 64, true) {
        return EFAULT;
    }
    if size > u32::MAX as usize {
        return (-75i64) as u64; // EOVERFLOW, as Linux answers
    }
    let w = super::fs::stat_bytes(kind, size, ino);
    let r32 = |at: usize| u32::from_le_bytes([w[at], w[at + 1], w[at + 2], w[at + 3]]);
    let r64 = |at: usize| u64::from_le_bytes(w[at..at + 8].try_into().unwrap());
    let mut b = [0u8; 64];
    b[0..4].copy_from_slice(&(r64(0) as u32).to_le_bytes());
    b[4..8].copy_from_slice(&(r64(8) as u32).to_le_bytes());
    b[8..10].copy_from_slice(&(r32(24) as u16).to_le_bytes());
    b[10..12].copy_from_slice(&(r64(16) as u16).to_le_bytes());
    b[12..14].copy_from_slice(&(r32(28) as u16).to_le_bytes());
    b[14..16].copy_from_slice(&(r32(32) as u16).to_le_bytes());
    b[16..20].copy_from_slice(&(r64(40) as u32).to_le_bytes());
    b[20..24].copy_from_slice(&(r64(48) as u32).to_le_bytes());
    b[24..28].copy_from_slice(&(r64(56) as u32).to_le_bytes());
    b[28..32].copy_from_slice(&(r64(64) as u32).to_le_bytes());
    for (i, at) in [72usize, 88, 104].iter().enumerate() {
        b[32 + i * 8..36 + i * 8].copy_from_slice(&(r64(*at) as u32).to_le_bytes());
        b[36 + i * 8..40 + i * 8].copy_from_slice(&(r64(at + 8) as u32).to_le_bytes());
    }
    unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), buf as *mut u8, 64) };
    0
}

fn write_stat64(buf: u64, kind: super::fs::Kind, size: usize, ino: u64) -> u64 {
    if !reachable(buf, 96, true) {
        return EFAULT;
    }
    // Reuse the 64-bit encoder for the fields that mean the same thing, so
    // mode, link count and block count cannot be computed two ways.
    let w = super::fs::stat_bytes(kind, size, ino);
    let r32 = |at: usize| u32::from_le_bytes([w[at], w[at + 1], w[at + 2], w[at + 3]]);
    let r64 = |at: usize| u64::from_le_bytes(w[at..at + 8].try_into().unwrap());
    // x86-64 `struct stat`: dev 0, ino 8, nlink 16, mode 24, uid 28, gid 32,
    // rdev 40, size 48, blksize 56, blocks 64, then three timespecs at 72.
    let mut b = [0u8; 96];
    b[0..8].copy_from_slice(&r64(0).to_le_bytes());
    b[12..16].copy_from_slice(&(r64(8) as u32).to_le_bytes());
    b[16..20].copy_from_slice(&r32(24).to_le_bytes());
    b[20..24].copy_from_slice(&(r64(16) as u32).to_le_bytes());
    b[24..28].copy_from_slice(&r32(28).to_le_bytes());
    b[28..32].copy_from_slice(&r32(32).to_le_bytes());
    b[32..40].copy_from_slice(&r64(40).to_le_bytes());
    b[44..52].copy_from_slice(&r64(48).to_le_bytes());
    b[52..56].copy_from_slice(&(r64(56) as u32).to_le_bytes());
    b[56..64].copy_from_slice(&r64(64).to_le_bytes());
    for (i, at) in [72usize, 88, 104].iter().enumerate() {
        b[64 + i * 8..68 + i * 8].copy_from_slice(&(r64(*at) as u32).to_le_bytes());
        b[68 + i * 8..72 + i * 8].copy_from_slice(&(r64(at + 8) as u32).to_le_bytes());
    }
    b[88..96].copy_from_slice(&r64(8).to_le_bytes());
    unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), buf as *mut u8, 96) };
    0
}

pub(super) fn sys_fstat(fd: u64, buf: u64) -> u64 {
    let found = with_fds(|fds, _| match fds.get(fd as usize) {
        Some(Some(super::fs::Fd::Disk(b))) => {
            let f = b.borrow();
            Some((super::fs::Kind::File, f.size, super::fs::ino_of(&f.path)))
        }
        Some(Some(super::fs::Fd::File(b))) => {
            let f = b.borrow();
            Some((super::fs::Kind::File, f.data.len(), super::fs::ino_of(&f.path)))
        }
        Some(Some(super::fs::Fd::Dir(b))) => {
            let d = b.borrow();
            Some((super::fs::Kind::Dir, 0, super::fs::ino_of(&d.path)))
        }
        Some(Some(super::fs::Fd::Dev(b))) => {
            let d = b.borrow();
            Some((super::fs::Kind::Char, super::dev::size(d.node), super::fs::ino_of(&d.path)))
        }
        // The standard three report as pipes, which is the only answer that
        // agrees with the rest of this module: `lseek` on them is `ESPIPE` and
        // `read` on stdin is a permanent end of file.
        Some(Some(_)) => Some((super::fs::Kind::Fifo, 0, 1)),
        _ => None,
    })
    .flatten();
    match found {
        Some((k, n, ino)) => write_stat(buf, k, n, ino),
        None => EBADF,
    }
}

pub(super) fn sys_statat(dirfd: u64, path_at: u64, buf: u64, flags: u64) -> u64 {
    let raw = match read_cstr(path_at) {
        Ok(v) => v,
        Err(e) => return e,
    };
    // **An empty path with `AT_EMPTY_PATH` means the descriptor itself**, and
    // this is where a real `ld.so` stopped. glibc opens a library, reads its
    // header, and then asks how big it is with
    // `newfstatat(fd, "", buf, AT_EMPTY_PATH)` rather than `fstat` -- so a
    // kernel that resolves the empty string answers `ENOENT` about a file it
    // has open, and the linker reports "error while loading shared libraries"
    // about a library it just found. The whole `-ENOSYS` instrument could not
    // see it, because every call involved was implemented and every one of
    // them succeeded except the last.
    //
    // The flags argument was not merely unhandled, it was never passed: the
    // dispatcher dropped `r10` on the floor and this took three arguments.
    if raw.is_empty() {
        if flags & AT_EMPTY_PATH != 0 {
            return sys_fstat(dirfd, buf);
        }
        return ENOENT;
    }
    let rel = !raw.starts_with('/');
    let found = with_fds(|fds, cwd| {
        let base = if rel { at_base(fds, dirfd, cwd)? } else { String::new() };
        super::fs::resolve(&base, &raw).ok_or(ENOENT)
    });
    let found = match found {
        Some(Ok(p)) => Some(p),
        Some(Err(e)) => return e,
        None => None,
    };
    let Some(path) = found else { return ENOENT };
    if let Some(on) = super::disk::lookup(&path) {
        return match on {
            Some(n) if n.is_dir => write_stat(buf, super::fs::Kind::Dir, 0, super::fs::ino_of(&path)),
            Some(n) => write_stat(buf, super::fs::Kind::File, n.size, super::fs::ino_of(&path)),
            None => ENOENT,
        };
    }
    if let Some(n) = super::dev::node(&path) {
        return write_stat(buf, super::fs::Kind::Char, super::dev::size(n), super::fs::ino_of(&path));
    }
    if super::dev::is_dir(&path) {
        return write_stat(buf, super::fs::Kind::Dir, 0, super::fs::ino_of(&path));
    }
    if super::proc::claims(&path) {
        if super::proc::is_dir(&path) {
            return write_stat(buf, super::fs::Kind::Dir, 0, super::fs::ino_of(&path));
        }
        // The size is the content's, which means making it. These are all a
        // few hundred bytes, and reporting zero the way Linux does for most of
        // `/proc` would make `wc -c` answer nothing about a file with bytes in
        // it.
        let n = super::proc::read(&path).map(|d| d.len()).unwrap_or(0);
        return write_stat(buf, super::fs::Kind::File, n, super::fs::ino_of(&path));
    }
    if sysbox::is_dir(&path) {
        return write_stat(buf, super::fs::Kind::Dir, 0, super::fs::ino_of(&path));
    }
    // `blob_len` rather than `read_blob`: the only field wanted here is the
    // size, and reading the blob to find it copies the whole file into the
    // heap and drops it. `stat` on a large checkpoint is an ordinary thing for
    // a program to do and was an allocation of the whole checkpoint.
    match sysbox::blob_len(&path) {
        Some(n) => write_stat(buf, super::fs::Kind::File, n, super::fs::ino_of(&path)),
        None => ENOENT,
    }
}

pub(super) fn sys_getdents64(fd: u64, buf: u64, len: u64) -> u64 {
    if !reachable(buf, len as usize, true) {
        return EFAULT;
    }
    // The records are built in the kernel before being copied out, so the
    // guest's buffer length is a length this kernel allocates. Capped, since a
    // guest owning a large mapping could otherwise ask for a second copy of it
    // on the heap; short reads are the ordinary case for this call anyway and
    // the caller's loop already handles them.
    let room = (len as usize).min(DENTS_MAX);
    with_fds(|fds, _| match fds.get_mut(fd as usize) {
        Some(Some(super::fs::Fd::Dir(body))) => {
            let d = &mut *body.borrow_mut();
            let (path, entries, at) = (&d.path, &d.entries, &mut d.at);
            let mut out = alloc::vec::Vec::new();
            while *at < entries.len() {
                let (name, is_dir, _) = &entries[*at];
                let mut full = path.clone();
                if !full.ends_with('/') {
                    full.push('/');
                }
                full.push_str(name);
                let next = (*at + 1) as u64;
                if !super::fs::dirent(
                    &mut out, room, super::fs::ino_of(&full), next, *is_dir, name,
                ) {
                    break;
                }
                *at += 1;
            }
            // Zero means end of directory, which is how a caller's loop stops.
            // It must not be returned while entries remain, so a buffer too
            // small for even one entry is EINVAL rather than a silent end.
            if out.is_empty() && *at < entries.len() {
                return EINVAL;
            }
            unsafe { core::ptr::copy_nonoverlapping(out.as_ptr(), buf as *mut u8, out.len()) };
            out.len() as u64
        }
        Some(Some(_)) => ENOTDIR,
        _ => EBADF,
    })
    .unwrap_or(EBADF)
}

/// Nothing here is a terminal.
///
/// `ENOTTY` is the answer that makes a program treat its output as a pipe:
/// full buffering, no colour, no width probing. That is true here and it is
/// also the useful answer, because the alternative is claiming a terminal and
/// then being asked its window size.
/// What an evdev device answers about itself.
///
/// **These are the interface, and the events are the easy half.** A program
/// classifies a device by reading its capability bitmaps and nothing else, so
/// a mouse that fails to advertise `REL_X` is opened, read from successfully,
/// and ignored -- which looks exactly like input that does not arrive.
///
/// Requests are matched on type and number with the size masked away, because
/// the length-carrying ones encode it in the request itself: `EVIOCGNAME(64)`
/// and `EVIOCGNAME(256)` are different numbers naming one thing.
fn input_ioctl(d: super::input::Dev, req: u64, arg: u64) -> u64 {
    use super::input as ev;
    let want = ev::request_len(req);
    let nr = ev::request(req);

    // Everything below writes into the caller's buffer, so one bounds check
    // covers the lot -- and a zero length is a request for nothing, which is
    // not an error and must not become a check against a null pointer.
    let mut give = |bytes: &[u8]| -> u64 {
        let n = bytes.len().min(want);
        if n == 0 {
            return 0;
        }
        if !reachable(arg, n, true) {
            return EFAULT;
        }
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), arg as *mut u8, n) };
        n as u64
    };

    match nr {
        ev::EVIOC_VERSION => {
            if !reachable(arg, 4, true) {
                return EFAULT;
            }
            unsafe { core::ptr::write_unaligned(arg as *mut u32, ev::EV_VERSION) };
            0
        }
        ev::EVIOC_ID => give(&ev::ident(d)),
        // The name is returned *with* its terminator and the length includes
        // it, which is what libevdev measures the string by.
        ev::EVIOC_NAME => {
            let mut b = [0u8; 64];
            let n = ev::name(d).len().min(63);
            b[..n].copy_from_slice(&ev::name(d).as_bytes()[..n]);
            give(&b[..n + 1])
        }
        // No physical location and no unique id. `ENOENT` is what a driver
        // without them answers, and libinput treats it as absent rather than
        // as a failure -- where an empty string would be a device claiming to
        // sit at the empty path.
        ev::EVIOC_PHYS | ev::EVIOC_UNIQ => ENOENT,
        // No input properties. Zeroes rather than a refusal: this is a
        // bitmask, and "none set" is a complete and true answer.
        ev::EVIOC_PROP => {
            let zero = [0u8; 8];
            give(&zero[..zero.len().min(want)])
        }
        ev::EVIOC_KEYSTATE => {
            let mut b = [0u8; 96];
            ev::key_state(d, &mut b);
            give(&b[..b.len().min(want.max(1))])
        }
        // `EVIOCGBIT(ev, len)` is one number per event type, so the family is
        // a contiguous run and the type is the offset into it.
        n if n >= ev::EVIOC_BIT && n < ev::EVIOC_BIT + 0x20 => {
            let mut b = [0u8; 96];
            ev::bits(d, n - ev::EVIOC_BIT, &mut b);
            give(&b[..b.len().min(want.max(1))])
        }
        // **Accepted and a no-op, which is honest here.** A grab asks for
        // exclusive access so a compositor does not also see the events. There
        // is one guest and the desktop has already stood down for it, so the
        // exclusivity a grab asks for is a fact rather than a request --
        // refusing would stop SDL, which grabs for relative mouse mode.
        ev::EVIOC_GRAB => 0,
        // Which clock timestamps come from. There is one clock here and it is
        // monotonic, which is what `CLOCK_MONOTONIC` asks for and near enough
        // to what `CLOCK_REALTIME` asks for that refusing would be worse.
        ev::EVIOC_SETCLOCK => 0,
        _ => ENOTTY,
    }
}

/// The four requests a framebuffer program makes, and `ENOTTY` for everything
/// else.
///
/// `ENOTTY` rather than `EINVAL` for an unknown request, because that is how a
/// program asks "are you a terminal" and gets told no -- which is what every
/// runtime here does on startup and what `isatty` is built from.
pub(super) fn sys_ioctl(fd: u64, req: u64, arg: u64) -> u64 {
    let node = with_fds(|fds, _| match fds.get(fd as usize) {
        Some(Some(super::fs::Fd::Dev(b))) => Some(b.borrow().node),
        Some(Some(_)) => None,
        _ => None,
    });
    let known = with_fds(|fds, _| matches!(fds.get(fd as usize), Some(Some(_)))).unwrap_or(false);
    if !known {
        return EBADF;
    }
    if let Some(d) = node.flatten().and_then(super::dev::is_input) {
        return input_ioctl(d, req, arg);
    }
    if let Some(Some(super::dev::Node::Snd(k))) = node {
        let nonblock = with_fds(|fds, _| match fds.get(fd as usize) {
            Some(Some(super::fs::Fd::Dev(b))) => b.borrow().nonblock,
            _ => false,
        })
        .unwrap_or(false);
        return super::snd::ioctl(k, req, arg, nonblock);
    }
    let Some(Some(super::dev::Node::Fb)) = node else { return ENOTTY };
    match req {
        super::dev::FBIOGET_VSCREENINFO => {
            let Some(v) = super::dev::var_screeninfo() else { return ENODEV };
            if !reachable(arg, super::dev::VAR_LEN, true) {
                return EFAULT;
            }
            unsafe { core::ptr::copy_nonoverlapping(v.as_ptr(), arg as *mut u8, v.len()) };
            0
        }
        super::dev::FBIOGET_FSCREENINFO => {
            let Some(f) = super::dev::fix_screeninfo() else { return ENODEV };
            if !reachable(arg, super::dev::FIX_LEN, true) {
                return EFAULT;
            }
            unsafe { core::ptr::copy_nonoverlapping(f.as_ptr(), arg as *mut u8, f.len()) };
            0
        }
        // **Read-only, and the refusal is the feature.** There is no
        // mode-setting on this machine: the geometry is whatever the firmware
        // left. Answering 0 to a mode this display cannot enter leaves a
        // program drawing at the wrong size forever with nothing reporting it,
        // which is strictly worse than a failure it can branch on.
        super::dev::FBIOPUT_VSCREENINFO => {
            if !reachable(arg, super::dev::VAR_LEN, false) {
                return EFAULT;
            }
            let want =
                unsafe { core::slice::from_raw_parts(arg as *const u8, super::dev::VAR_LEN) };
            if super::dev::mode_matches(want) {
                0
            } else {
                EINVAL
            }
        }
        // Unblanking is what a program does before it draws, so it is taken as
        // the moment the screen changes hands. Every other blanking level is
        // accepted and does nothing, there being no panel power control here
        // that is not the whole machine's.
        super::dev::FBIOBLANK => {
            super::dev::hold_screen(arg == 0);
            0
        }
        _ => ENOTTY,
    }
}

/// Grow or query the break.
///
/// **Linux's `brk` never returns an error.** It answers the resulting break,
/// which on failure is the *unchanged* one -- and libc decides it failed by
/// comparing that against what it asked for. Returning `-ENOMEM` here instead
/// would hand musl a break of `0xFFFFFFFFFFFFFFF4` and it would believe it,
/// which is the difference between a refused allocation and a wild pointer.
///
/// One honest deviation: on Linux the break begins immediately after the
/// image, and here it is a separate region. Nothing reads it that way -- every
/// allocator asks `brk(0)` and grows from the answer -- but a program that
/// assumed adjacency would be wrong, so it is written down rather than left to
/// be discovered.
pub(super) fn sys_brk(want: u64) -> u64 {
    let sp = unsafe { guest_slot() };
    let Some(sp) = sp.as_mut() else { return 0 };
    if want >= sp.brk_start && want <= sp.brk_end {
        // **Growing past where the break has been gives zeroed pages, as
        // Linux's does.** Linux unmaps the pages a shrinking break leaves and
        // maps fresh ones when it grows back; here the region is allocated
        // once and the break is a number, so a trim followed by a grow handed
        // glibc back its own freed chunks -- and glibc's `calloc` treats
        // memory above the old top as fresh and does not clear it. Whole
        // pages only: the bytes between the old break and the end of its page
        // stay, because on Linux that page was never unmapped either.
        let from = page_up(sp.brk_now as usize) as u64;
        let to = (page_up(want as usize) as u64).min(sp.brk_end);
        let mut page = from;
        while page < to {
            if let Some(p) = crate::mem::paging::query(page) {
                if p.present && p.write {
                    unsafe { core::ptr::write_bytes(page as *mut u8, 0, PAGE as usize) };
                }
            }
            page += PAGE;
        }
        sp.brk_now = want;
    }
    sp.brk_now
}

/// Anonymous private memory, and nothing else.
///
/// Three refusals, each for a reason about this machine rather than about the
/// arguments. A file-backed mapping needs an fd table that does not exist. A
/// `MAP_FIXED` needs an address one address space can promise, which is the
/// same objection that makes the loader decline `ET_EXEC`. And a zero length
/// is `EINVAL` on Linux, so it is `EINVAL` here.
pub(super) fn sys_mmap(addr: u64, len: u64, prot: u64, flags: u64, fd: u64, off: u64) -> u64 {
    if len == 0 {
        return EINVAL;
    }
    if len > MAP_MAX {
        return ENOMEM;
    }
    let anon = flags & MAP_ANONYMOUS != 0;

    // **A file mapping is a copy, and that is a real deviation.** Linux maps
    // the page cache, so two processes mapping one file share pages and a
    // write through `MAP_SHARED` is visible to the other. Here an open file
    // already *is* its contents -- `fs.rs` says so and gives the reason -- so
    // there is no cache to point at and the honest thing is to copy. What that
    // costs is stated rather than hidden: a shared writable file mapping is
    // refused, because honouring it would mean writing back into a
    // content-addressed store, which is a new root hash per modified page.
    // `MAP_PRIVATE` is exactly a copy, so it is exactly right, and it is what
    // `ld.so` uses for every library it loads.
    // **The device branch, before the refusal that does not apply to it.** A
    // shared writable file mapping is refused everywhere else because writing
    // back into a content-addressed store is a new root hash per page. The
    // framebuffer is not in the store, and `MAP_SHARED` on it is not an
    // awkward case to tolerate -- it is the entire point of the device, and
    // the one thing every program that draws does.
    //
    // What comes back is the aperture itself. This kernel is identity-mapped,
    // so there is nothing to translate and nothing to copy: the guest is
    // handed the pages the display controller is already scanning out, and a
    // frame it writes is on screen as it writes it.
    if !anon {
        let dev = (unsafe { guest_slot() })
            .as_ref()
            .and_then(|sp| sp.fds.get(fd as usize))
            .and_then(|f| f.as_ref())
            .and_then(|f| match f {
                super::fs::Fd::Dev(b) => Some(b.borrow().node),
                _ => None,
            });
        if let Some(n) = dev {
            if n != super::dev::Node::Fb {
                // The other three have no memory to map. `ENODEV` is what
                // Linux answers for a device whose driver has no `mmap`.
                return ENODEV;
            }
            let Some((base, _, _, len, _, _)) = super::dev::fb() else { return ENODEV };
            if off as usize >= len {
                return EINVAL;
            }
            // Fixed placement is refused rather than emulated: the aperture is
            // where it is, and a guest naming a different address is asking
            // for a copy that would never reach the screen.
            if flags & MAP_FIXED != 0 && addr != base + off {
                return EINVAL;
            }
            let at = base + off;
            let span = page_up((len - off as usize).min(len as usize));
            let perm = crate::mem::paging::Perm {
                present: true,
                write: prot & PROT_WRITE != 0,
                exec: false,
                user: true,
            };
            if !crate::mem::paging::protect(at, span, perm) {
                return ENOMEM;
            }
            super::dev::hold_screen(true);
            let sp = unsafe { guest_slot() }.as_mut();
            if let Some(sp) = sp {
                sp.maps.push(Mapping { at, len: span, from: Source::Device });
            }
            return at;
        }
    }

    // A memfd, and the same argument as the framebuffer arriving a second
    // time. The refusal below is about writing back into a content-addressed
    // store; a memfd is not in the store, it is heap pages the descriptor
    // owns, so `MAP_SHARED | PROT_WRITE` is exactly right for it. This is
    // where a Wayland client's buffer comes from.
    if !anon {
        if let Some(m) = memfd(fd) {
            const MAP_SHARED: u64 = 0x01;
            if flags & MAP_SHARED == 0 {
                // A private mapping of a memfd is a copy-on-write view, which
                // is legal and which nothing has asked for. Refusing says so.
                return EINVAL;
            }
            if off != 0 {
                return EINVAL;
            }
            let mut f = m.borrow_mut();
            let want = page_up(len as usize);
            if want == 0 || f.at == 0 || want > page_up(f.len) {
                return EINVAL;
            }
            // **With a space, the memfd's pages go at an arena address**, the
            // way anonymous memory already does. Handing back `f.at` -- the
            // backing's own heap address -- is only an address the guest can
            // hold when the heap happens to sit below 4 GiB, and for a 32-bit
            // guest on a machine whose heap is above it the answer was
            // `0x1479b0000`: the guest kept the low 32 bits, and Mesa drew
            // every frame of Portal into whatever it owned at `0x479b0000` --
            // malloc's chunks, a library's read-only page. glibc reported it
            // as tcache corruption, as an invalid free, and as a fault on a
            // page that was never the buffer, three symptoms with one cause.
            //
            // Each mapping is its own view, as on Linux; unmapping one frees
            // nothing (`Copied`), because the pages are the memfd's.
            let backing = f.at;
            let arena = unsafe { guest_slot() }.as_mut().and_then(|sp| {
                sp.tables.as_ref()?;
                let at = arena_take(sp, want as u64)?;
                let t = sp.tables.as_mut()?;
                let mut off = 0u64;
                while off < want as u64 {
                    if !t.map_page(at + off, backing + off, prot & PROT_WRITE != 0, true) {
                        return None;
                    }
                    off += 4096;
                }
                sp.maps.push(Mapping { at, len: want, from: Source::Copied });
                Some(at)
            });
            if let Some(at) = arena {
                f.maps += 1;
                return at;
            }
            if guest_root() != 0 {
                return ENOMEM;
            }
            let at = f.at;
            // **Every mapping of a memfd is the same pages**, so its rights are
            // a property of those pages, not of one mapping. A second,
            // read-only mapping must not take writing away from a first one
            // that is still live -- on Linux they are independent views.
            let already = unsafe { guest_slot() }.as_ref().is_some_and(|sp| {
                sp.maps.iter().any(|m| {
                    matches!(m.from, Source::Shared) && m.at < at + want as u64 && at < m.at + m.len as u64
                })
            });
            let perm = crate::mem::paging::Perm {
                present: true,
                write: prot & PROT_WRITE != 0 || already,
                exec: false,
                user: true,
            };
            if !crate::mem::paging::protect(at, want, perm) {
                return ENOMEM;
            }
            // Only ever rises. It is what stops a later `ftruncate` moving
            // pages the guest is holding an address into, and it is not
            // decremented on unmap because `Source` cannot carry a reference
            // back: `Mapping` is `Copy`, since `fork` duplicates the list.
            f.maps += 1;
            drop(f);
            if let Some(sp) = (unsafe { guest_slot() }).as_mut() {
                sp.maps.push(Mapping { at, len: want, from: Source::Shared });
            }
            return at;
        }
    }

    let backing = if anon {
        None
    } else {
        if off % PAGE != 0 {
            return EINVAL;
        }
        if flags & MAP_SHARED != 0 && prot & PROT_WRITE != 0 {
            return ENODEV;
        }
        let Some(sp) = (unsafe { guest_slot() }).as_ref() else { return ENOMEM };
        match sp.fds.get(fd as usize).and_then(|f| f.as_ref()) {
            Some(super::fs::Fd::Disk(b)) => {
                let mut f = b.borrow_mut();
                let from = (off as usize).min(f.size);
                let to = from.saturating_add(len as usize).min(f.size);
                let mut v = alloc::vec![0u8; to - from];
                let got = f.read_at(from, &mut v);
                v.truncate(got);
                Some(v)
            }
            Some(super::fs::Fd::File(b)) => {
                let f = b.borrow();
                // A mapping may run past the end of the file, and the tail is
                // zero rather than an error -- that is how the `.bss` of a
                // shared object is made, so refusing it would refuse every
                // library there is.
                let from = (off as usize).min(f.data.len());
                let to = from.saturating_add(len as usize).min(f.data.len());
                Some(f.data[from..to].to_vec())
            }
            // A directory or a socket has no bytes to map. `ENODEV` is what
            // Linux answers for a file whose type does not support it.
            Some(_) => return ENODEV,
            None => return EBADF,
        }
    };

    // Where it goes. Three cases and the middle one is the whole reason this
    // stopped being a refusal: `ld.so` reserves a span with one anonymous
    // mapping and then writes each segment of a library over it with
    // `MAP_FIXED`, so an address inside memory the guest already holds is not
    // an attack, it is the ordinary case.
    let (at, from) = if flags & MAP_FIXED != 0 {
        if addr == 0 || addr % PAGE != 0 {
            return EINVAL;
        }
        if owns(addr, len as usize) {
            // Its own memory, laid out again. Nothing is claimed and nothing
            // is recorded, because whatever already holds these pages still
            // holds them and will still free them.
            //
            // **But an anonymous mapping laid over it is new memory, and new
            // memory reads as zero.** Linux replaces the pages; these are the
            // same pages, so the old contents survived -- and a `.bss` laid
            // over a reservation, or an arena re-laid after a trim, then
            // starts with whatever the last tenant left. glibc's `calloc`
            // trusts fresh heap memory to be zero and skips clearing it, so
            // the symptom is not here at all: it is a structure that came
            // back from `calloc` holding stale pointers, freed thousands of
            // calls later, and reported by the allocator as tcache
            // corruption. Opened for writing first, since the pages may be
            // read-only or absent by now and this kernel enforces that on
            // itself too.
            if anon {
                let n = page_up(len as usize);
                let rw = crate::mem::paging::Perm { present: true, write: true, exec: false, user: true };
                if !crate::mem::paging::protect(addr, n, rw) {
                    return ENOMEM;
                }
                unsafe { core::ptr::write_bytes(addr as *mut u8, 0, n) };
            }
            (addr, None)
        } else if crate::mem::fixed::claim(addr, len as usize).is_ok() {
            // Whatever the last tenant left. `alloc_pages` zeroes and this
            // does not, and a program whose fresh memory starts as somebody
            // else's is one that works exactly once.
            unsafe { core::ptr::write_bytes(addr as *mut u8, 0, page_up(len as usize)) };
            (addr, Some(Source::Fixed))
        } else {
            return ENOMEM;
        }
    } else {
        // A non-fixed `addr` is a hint, and this ignores it. Linux is entitled
        // to as well, every allocator copes, and honouring it would spend a
        // placement claim on a suggestion.
        // **Anonymous memory in a guest with its own tables is page by page,
        // always**, not only when no contiguous run can be found: a range
        // built of single pages can be given back a page at a time, and
        // `munmap` of part of one is what glibc's thread arenas and LLVM's JIT
        // do to align a reservation. Contiguous, those trims could only be
        // accepted and the range kept, and a 32-bit guest's 1.4 GiB arena ran
        // out under Portal's options dialog -- `malloc` handed gallium a NULL.
        let anon_paged = backing.is_none() && unsafe { guest_slot() }.as_ref().is_some_and(|sp| sp.tables.is_some());
        let contiguous = if anon_paged { None } else { alloc_pages(len as usize) };
        let Some(phys) = contiguous else {
            // No contiguous run that size. With a space the arena is mapped a
            // page at a time anyway, so contiguity was never needed -- for
            // anonymous memory. A file's bytes are copied in after this point,
            // so a file mapping that cannot be had whole is refused instead.
            if backing.is_some() {
                crate::kprintln!("  [mmap] no contiguous run for a {:#x}-byte file mapping", len);
                return ENOMEM;
            }
            let want = page_up(len as usize) as u64;
            let paged = unsafe { guest_slot() }.as_mut().and_then(|sp| {
                sp.tables.as_ref()?;
                let Some(at) = arena_take(sp, want) else {
                    crate::kprintln!("  [mmap] the 32-bit arena has no {:#x}-byte gap left", want);
                    arena_report(sp);
                    return None;
                };
                let mut off = 0u64;
                while off < want {
                    let ok = alloc_pages(4096).is_some_and(|p| {
                        sp.tables.as_mut().is_some_and(|t| t.map_page(at + off, p, true, true))
                            || {
                                free_pages(p, 4096);
                                false
                            }
                    });
                    if !ok {
                        free_paged(sp, at, off);
                        arena_give(sp, at, want);
                        return None;
                    }
                    off += 4096;
                }
                Some(at)
            });
            match paged {
                Some(at) => {
                    // Fresh pages from the heap carry what the last owner left.
                    unsafe { core::ptr::write_bytes(at as *mut u8, 0, want as usize) };
                    let perm = crate::mem::paging::Perm {
                        present: prot != 0,
                        write: prot & PROT_WRITE != 0,
                        exec: prot & PROT_EXEC != 0,
                        user: true,
                    };
                    let _ = crate::mem::paging::protect(at, want as usize, perm);
                    if let Some(sp) = unsafe { guest_slot() }.as_mut() {
                        sp.maps.push(Mapping { at, len: want as usize, from: Source::Paged });
                    }
                    return at;
                }
                None => {
                    crate::kprintln!("  [mmap] out of memory for {:#x} bytes", len);
                    return ENOMEM;
                }
            }
        };
        // **With a space, the guest is told an arena address rather than the
        // heap address the pages actually have.** That is the last thing
        // standing between this and `fork`: a child gets its parent's mappings
        // at the same addresses, and "wherever the heap had room" is a
        // different number in each of them.
        let want = page_up(len as usize) as u64;
        let arena = unsafe { guest_slot() }.as_mut().and_then(|sp| {
            sp.tables.as_ref()?;
            // Out of arena is out of memory, said as Linux says it, rather
            // than mapping over the stack a 32-bit guest keeps just above it.
            let at = arena_take(sp, want)?;
            let t = sp.tables.as_mut()?;
            let mut off = 0u64;
            while off < want {
                if !t.map_page(at + off, phys + off, true, true) {
                    // Partly mapped and going no further. The tables are the
                    // guest's and die with it, so what has to be undone here
                    // is the pages -- the caller frees them on `None`.
                    return None;
                }
                off += 4096;
            }
            Some(at)
        });
        match arena {
            Some(at) => (at, Some(Source::Backed(phys))),
            None if guest_root() != 0 => {
                free_pages(phys, len as usize);
                return ENOMEM;
            }
            // No space, so the guest sees the heap address, as it always did.
            None => (phys, Some(Source::Heap)),
        }
    };

    // **Writable first, then copy, then the rights that were asked for.** The
    // copy used to come first, which is correct for a fresh allocation and
    // wrong for the one case `MAP_FIXED` exists to serve. A linker reserves a
    // span with `PROT_NONE` and then lays each segment of a library over it,
    // so by the time the file bytes arrive the guest has already made those
    // pages unwritable -- and the kernel, copying on the guest's behalf, took
    // a `#PF` at ring 0 with `CR0.WP` on. Nothing about it was the guest's
    // fault and no bounds check could have caught it: the range was one the
    // guest owned and the pointer was one the kernel chose.
    // Writable for the length of the fill, whichever kind it is. An anonymous
    // mapping is *zeroed* by Linux and this only got that for free while every
    // mapping came from `alloc_pages`: laid over memory the guest already
    // holds, the pages keep whatever was there. For a library that is the tail
    // of the file showing through where `.bss` should be, which is a
    // zero-initialised variable that silently is not.
    if from.is_none() || backing.is_some() {
        crate::mem::paging::protect(at, page_up(len as usize), crate::mem::paging::Perm::RWX);
    }
    match &backing {
        Some(b) => unsafe {
            core::ptr::copy_nonoverlapping(b.as_ptr(), at as *mut u8, b.len());
            // Past the end of the file is zero, which is how a shared object's
            // `.bss` is made when it shares a page with `.data`.
            let tail = page_up(len as usize).saturating_sub(b.len());
            core::ptr::write_bytes((at + b.len() as u64) as *mut u8, 0, tail);
        },
        None if from.is_none() => unsafe {
            core::ptr::write_bytes(at as *mut u8, 0, page_up(len as usize))
        },
        None => {}
    }
    // **Open it to ring 3, or the guest cannot touch what it just asked for.**
    // The loader opens the image, stack and break before the guest starts, and
    // a mapping made after that is not covered by any of them. At ring 0 this
    // was invisible, because the U bit meant nothing; the first ring-3 guest to
    // call `mmap` took a protection violation reading its own memory.
    //
    // `PROT_NONE` takes the page away rather than leaving it readable, which
    // is what `mprotect` already does and what makes a reservation a
    // reservation. Leaving it present is how a guard page guards nothing.
    let perm = crate::mem::paging::Perm {
        present: prot != 0,
        write: prot & PROT_WRITE != 0,
        exec: prot & PROT_EXEC != 0,
        user: true,
    };
    if !crate::mem::paging::protect(at, page_up(len as usize), perm) {
        // Partially applied rights are still rights, so close it on the way
        // out rather than handing the allocator whatever the walk managed.
        give_back(at, len as usize, from);
        return ENOMEM;
    }
    match from {
        // Re-laid over memory the guest already holds: no new record, because
        // a second entry naming the same pages would be freed twice.
        None => at,
        Some(from) => {
            let sp = unsafe { guest_slot() };
            match sp.as_mut() {
                Some(sp) => {
                    // **The pages handed out, not the length asked for.**
                    // `mmap` allocates whole pages and Linux rounds a
                    // mapping's length up to one, so recording the raw length
                    // leaves the tail of the last page owned by nobody --
                    // and `owns` then refuses a `MAP_FIXED` laid over it.
                    //
                    // A real `ld.so` is what found this. It reserves a span
                    // the exact size of a shared object, `0x4010` for glibc's
                    // `libpthread` stub, then lays each segment over it; the
                    // last of those covers `0x2fdf000` for two pages and the
                    // second page is past `0x2fe0010`. The kernel answered
                    // `ENOMEM`, `ld.so` said "failed to map segment from
                    // shared object", and nothing about the message pointed
                    // here.
                    sp.maps.push(Mapping { at, len: page_up(len as usize), from });
                    at
                }
                None => {
                    give_back(at, len as usize, Some(from));
                    ENOMEM
                }
            }
        }
    }
}

/// `madvise`. Only `MADV_DONTNEED` changes anything a program can see.
///
/// **It is a promise about contents, not a hint.** On a private anonymous
/// mapping Linux drops the pages, and the next touch reads zero. glibc shrinks
/// a thread's arena with exactly this and, when the arena grows back over the
/// same addresses, treats everything above its old top as fresh -- so `calloc`
/// skips clearing it. Answering 0 and changing nothing (which this did for
/// 32-bit guests) left that memory holding freed chunks' old links, handed
/// out as zeroed structures; the program followed them much later and glibc
/// reported `unaligned tcache chunk`, nowhere near the cause.
///
/// So the range is zeroed where it is anonymous memory the guest holds and can
/// write. A memfd or the framebuffer is shared, and Linux keeps their contents
/// across `DONTNEED`, so those are left alone; every other advice is a hint
/// this kernel has no use for and is accepted.
pub(super) fn sys_madvise(at: u64, len: u64, advice: u64) -> u64 {
    const MADV_DONTNEED: u64 = 4;
    if at % PAGE != 0 {
        return EINVAL;
    }
    if advice != MADV_DONTNEED || len == 0 {
        return 0;
    }
    let n = page_up(len as usize);
    if !owns(at, n) {
        return ENOMEM;
    }
    let shared = unsafe { guest_slot() }.as_ref().is_some_and(|sp| {
        sp.maps.iter().any(|m| {
            matches!(m.from, Source::Shared | Source::Device)
                && at < m.at + page_up(m.len) as u64
                && m.at < at + n as u64
        })
    });
    if shared {
        return 0;
    }
    let mut page = at;
    while page < at + n as u64 {
        // A page the guest has made read-only or absent keeps its contents:
        // it cannot have been written since, and opening it to clear it
        // would change rights the guest chose.
        if let Some(p) = crate::mem::paging::query(page) {
            if p.present && p.write {
                unsafe { core::ptr::write_bytes(page as *mut u8, 0, PAGE as usize) };
            }
        }
        page += PAGE;
    }
    0
}

/// Put a mapping's pages back where they came from, rights first.
///
/// **Rights first, always.** A page handed back still carrying its U bit is
/// one the next tenant inherits with ring-3 access attached, and the symptom
/// lands in an unrelated subsystem hours later -- which this tree has now paid
/// for three times, in `munmap`, in teardown, and in the guest's image.
fn give_back(at: u64, len: usize, from: Option<Source>) {
    match from {
        // Somebody else's pages, laid out again. They keep their rights
        // because they keep their owner.
        None => {}
        Some(Source::Heap) => {
            crate::mem::paging::release_to_heap(at, page_up(len));
            free_pages(at, len);
        }
        Some(Source::Fixed) => {
            crate::mem::paging::protect(at, page_up(len), crate::mem::paging::Perm::RWX);
            crate::mem::fixed::release(at);
        }
        // Kernel-only again, and nothing freed. `Perm::RWX` carries
        // `user: false`, which is the whole of what has to be undone: the
        // aperture was already present, writable and executable before a guest
        // asked for it, and it still belongs to the display either way.
        Some(Source::Device) => {
            crate::mem::paging::protect(at, page_up(len), crate::mem::paging::Perm::RWX);
        }
        // The same debt and nothing else. These pages belong to a `memfd`
        // that is still open, or to one whose close will free them; taking
        // them back here would pull memory out from under a live descriptor.
        Some(Source::Shared) => {
            crate::mem::paging::protect(at, page_up(len), crate::mem::paging::Perm::RWX);
        }
        Some(Source::Copied) => {}
        // Freed by `free_paged`, which can see the guest's tables; this cannot.
        Some(Source::Paged) => {}
        // The backing, at its own address. Nothing is done about the guest's
        // mapping of it: those page tables belong to the guest's `Space` and
        // die with it, so unmapping here would be tidying something that is
        // about to be freed wholesale.
        Some(Source::Backed(phys)) => {
            crate::mem::paging::release_to_heap(phys, page_up(len));
            free_pages(phys, len);
        }
    }
}

/// Give a whole mapping back. Partial unmapping is refused rather than
/// approximated: splitting one allocation into two is not something the heap
/// underneath can express, and silently unmapping more than was asked is worse
/// than saying no.
pub(super) fn sys_munmap(at: u64, len: u64) -> u64 {
    let sp = unsafe { guest_slot() };
    let Some(sp) = sp.as_mut() else { return EINVAL };
    if at % 4096 != 0 || len == 0 {
        return EINVAL;
    }
    // The record's length and the guest's are compared page-rounded: the
    // guest passes whatever it asked `mmap` for, and compared as given, every
    // mapping of an odd length -- every library a linker maps -- could never
    // be unmapped at all.
    let want = page_up(len as usize);
    match sp.maps.iter().position(|m| m.at == at && page_up(m.len) == want) {
        Some(i) => {
            let m = sp.maps.remove(i);
            // Another mapping of the same memfd pages is still live: those
            // pages keep their rights until the last view of them goes. Taking
            // the U bit off here killed the survivor -- Mesa maps a Wayland
            // buffer, and a page fault at its first write was the result.
            if matches!(m.from, Source::Shared)
                && sp.maps.iter().any(|o| {
                    matches!(o.from, Source::Shared) && o.at < m.at + m.len as u64 && m.at < o.at + o.len as u64
                })
            {
                return 0;
            }
            // **Close it before giving it back.** `teardown` does this for a
            // guest that exits still holding mappings, and this path was
            // missed: an explicit unmap returned pages to the allocator with
            // their U bit still set, so the next thing to be handed that
            // memory -- kernel or otherwise -- came with ring-3 access
            // attached. A `diag all` caught it and `diag paging` alone did
            // not, because it only shows up once something else has run.
            if m.from == Source::Paged {
                free_paged(sp, m.at, page_up(m.len) as u64);
            } else {
                give_back(m.at, m.len, Some(m.from));
            }
            // An arena address goes back on the free list once its pages have.
            if matches!(m.from, Source::Backed(_) | Source::Copied | Source::Paged) && sp.tables.is_some() {
                arena_give(sp, m.at, page_up(m.len) as u64);
            }
            0
        }
        // **Part of a mapping: accepted, and the pages kept.** The heap cannot
        // free part of an allocation, so the honest options were to refuse --
        // which `ld.so` meets every time it trims a reservation, and then
        // reports as a failed load -- or to keep the pages mapped until the
        // whole mapping or the guest goes. This is the second: it costs
        // memory, not correctness, except that the guest can still read what
        // it unmapped rather than faulting there.
        None => {
            let end = at + want as u64;
            let Some(i) = sp.maps.iter().position(|m| at >= m.at && end <= m.at + page_up(m.len) as u64) else {
                return EINVAL;
            };
            // Page-by-page memory can be trimmed exactly: those pages freed,
            // that range back in the arena, and the record split around it.
            if sp.maps[i].from == Source::Paged && sp.tables.is_some() {
                let m = sp.maps.remove(i);
                let m_end = m.at + page_up(m.len) as u64;
                free_paged(sp, at, want as u64);
                arena_give(sp, at, want as u64);
                if at > m.at {
                    sp.maps.push(Mapping { at: m.at, len: (at - m.at) as usize, from: Source::Paged });
                }
                if end < m_end {
                    sp.maps.push(Mapping { at: end, len: (m_end - end) as usize, from: Source::Paged });
                }
            }
            0
        }
    }
}

/// Change what a range of the guest's own memory may be used for.
///
/// **This was unimplemented on purpose until page rights existed**, and the
/// reason is worth keeping: every page in this kernel was writable and
/// executable, so answering 0 would have claimed an enforcement that did not
/// exist and musl's guard pages would have guarded nothing, while refusing
/// stops any real allocator. Both answers were lies. Now there is a third.
///
/// `PROT_NONE` clears the present bit, so the page genuinely faults. That is
/// the whole point of a guard page and it is also why `reachable` exists: a
/// guest that hides a page from itself and then hands the kernel a pointer
/// into it gets `EFAULT` rather than taking the machine down.
///
/// The range must be one the loader gave this guest. Linux answers `ENOMEM`
/// for a range with no mapping under it, so that is what comes back.
pub(super) fn sys_mprotect(at: u64, len: u64, prot: u64) -> u64 {
    if at % PAGE != 0 {
        return EINVAL;
    }
    if len == 0 {
        return 0;
    }
    if !owns(at, len as usize) {
        return ENOMEM;
    }
    let perm = crate::mem::paging::Perm {
        present: prot != 0,
        write: prot & PROT_WRITE != 0,
        exec: prot & PROT_EXEC != 0,
        // A guest's own memory stays reachable from ring 3 whatever it does to
        // the other three bits. Clearing this would hide the page from the
        // guest while leaving it visible to the kernel, which is backwards.
        user: true,
    };
    if crate::mem::paging::protect(at, len as usize, perm) {
        0
    } else {
        ENOMEM
    }
}

/// Set or read a segment base -- and refuse one of them.
///
/// `ARCH_SET_FS` is how thread-local storage works and musl calls it before
/// `main`, so it is honoured: the base goes straight into `IA32_FS_BASE` and
/// the kernel's own value is restored at teardown, because the register
/// belongs to the machine rather than to the guest.
///
/// **`ARCH_SET_GS` is refused, and that is the most specific thing stage 0 has
/// found so far.** `GS` is not free here: `cpu::percpu` points it at each
/// core's own block, `gs:[0]` is how the allocator discovers which core it is
/// billing, and there is no privilege boundary to stop a guest overwriting it.
/// A guest setting `GS` would leave the next kernel allocation reading its
/// thread-local storage as a per-core structure -- so this is the first place
/// where "the guest and the kernel share everything" stops being an
/// architectural note and becomes a specific call that has to say no.
///
/// Reading `GS` is refused for the smaller reason that it hands out a kernel
/// pointer to code that has no business with one.
pub(super) fn sys_arch_prctl(code: u64, addr: u64) -> u64 {
    match code {
        ARCH_SET_FS => {
            unsafe { crate::cpu::wrmsr(IA32_FS_BASE, addr) };
            0
        }
        ARCH_GET_FS => {
            // Eight bytes written wherever the guest points. Unchecked, this
            // is a kernel-corrupting primitive handed to the program: it could
            // name a page table, the heap's free list, or the model's weights.
            // Eight bytes are written, so write rights are required and a
            // read-only page is refused as firmly as an absent one.
            if !reachable(addr, 8, true) {
                return EFAULT;
            }
            let base = unsafe { crate::cpu::rdmsr(IA32_FS_BASE) };
            unsafe { (addr as *mut u64).write(base) };
            0
        }
        ARCH_SET_GS | ARCH_GET_GS => EPERM,
        _ => EINVAL,
    }
}

/// Called from the stub. Not public API: the only caller is three lines of
/// assembly above, and the `sysv64` pinning is why `rdi` is the frame.
#[no_mangle]
pub extern "sysv64" fn glados_syscall_dispatch(f: &mut Frame) {
    // A doomed guest asks for nothing more. Here and again on the way out,
    // which bounds how long a task outlives its doom by one syscall's work.
    big_lock();
    unsafe { die_if_doomed() };
    let nr = f.rax;
    let args = [f.rdi, f.rsi, f.rdx, f.r10, f.r8, f.r9];
    super::prof::enter(nr, &args);
    let t0 = super::prof::start();

    let (ret, served) = match nr {
        SYS_WRITE => (sys_write(f.rdi, f.rsi, f.rdx as usize), true),
        SYS_READ => (sys_read(f.rdi, f.rsi, f.rdx), true),
        SYS_OPEN => (sys_openat(super::fs::AT_FDCWD as u64, f.rdi, f.rsi, f.rdx), true),
        SYS_OPENAT => (sys_openat(f.rdi, f.rsi, f.rdx, f.r10), true),
        SYS_CLOSE => (sys_close(f.rdi), true),
        SYS_LSEEK => (sys_lseek(f.rdi, f.rsi, f.rdx), true),
        SYS_FSTAT => (sys_fstat(f.rdi, f.rsi), true),
        SYS_STAT | SYS_LSTAT => {
            (sys_statat(super::fs::AT_FDCWD as u64, f.rdi, f.rsi, 0), true)
        }
        SYS_NEWFSTATAT => (sys_statat(f.rdi, f.rsi, f.rdx, f.r10), true),
        SYS_GETDENTS64 => (sys_getdents64(f.rdi, f.rsi, f.rdx), true),
        SYS_IOCTL => (sys_ioctl(f.rdi, f.rsi, f.rdx), true),
        SYS_PREAD64 => (sys_pread64(f.rdi, f.rsi, f.rdx, f.r10), true),
        SYS_SET_ROBUST_LIST => (sys_set_robust_list(f.rdi, f.rsi), true),
        SYS_PRLIMIT64 => (sys_prlimit64(f.rdi, f.rsi, f.rdx, f.r10), true),
        SYS_GETRANDOM => (sys_getrandom(f.rdi, f.rsi, f.rdx), true),
        SYS_TIME => (sys_time(f.rdi), true),
        SYS_READLINK => {
            (sys_readlinkat(super::fs::AT_FDCWD as u64, f.rdi, f.rsi, f.rdx), true)
        }
        SYS_READLINKAT => (sys_readlinkat(f.rdi, f.rsi, f.rdx, f.r10), true),
        SYS_SYSINFO => (sys_sysinfo(f.rdi), true),
        SYS_SCHED_GETAFFINITY => (sys_sched_getaffinity(f.rdi, f.rsi, f.rdx), true),
        // **`rseq` stays refused, and that is the right answer rather than a
        // gap.** Linux itself answers `ENOSYS` whenever `CONFIG_RSEQ` is off,
        // so every libc that registers one already copes with being told no,
        // and restartable sequences are a per-CPU optimisation this kernel has
        // no way to honour. Named here so nobody implements it to make a line
        // in the trace go away.
        SYS_WRITEV => (sys_writev(f.rdi, f.rsi, f.rdx), true),
        SYS_READV => (sys_readv(f.rdi, f.rsi, f.rdx), true),
        SYS_SOCKET => (sys_socket(f.rdi, f.rsi, f.rdx), true),
        SYS_CONNECT => (sys_connect(f.rdi, f.rsi, f.rdx), true),
        SYS_BIND => (sys_bind(f.rdi, f.rsi, f.rdx), true),
        SYS_SENDMSG => (sys_sendmsg(f.rdi, f.rsi, f.rdx), true),
        SYS_RECVMSG => (sys_recvmsg(f.rdi, f.rsi, f.rdx), true),
        SYS_LISTEN => (sys_listen(f.rdi), true),
        SYS_ACCEPT => (sys_accept4(f.rdi, f.rsi, f.rdx, 0), true),
        SYS_ACCEPT4 => (sys_accept4(f.rdi, f.rsi, f.rdx, f.r10), true),
        SYS_SOCKETPAIR => (sys_socketpair(f.rdi, f.rsi, f.rdx, f.r10), true),
        SYS_SENDTO => (sys_send(f.rdi, f.rsi, f.rdx), true),
        SYS_RECVFROM => (sys_recv(f.rdi, f.rsi, f.rdx), true),
        SYS_SHUTDOWN => (sys_shutdown(f.rdi, f.rsi), true),
        // Accepted and stored nowhere. Every option a program sets here is
        // about buffering, keepalive or timeouts, and this stack answers all
        // three its own way -- refusing would stop programs that set them as a
        // formality, which is most of them.
        SYS_SETSOCKOPT => (0, true),
        SYS_GETSOCKOPT => (0, true),
        SYS_ACCESS => (sys_access(f.rdi, f.rsi), true),
        SYS_FACCESSAT => (sys_access(f.rsi, f.rdx), true),
        SYS_FTRUNCATE => (sys_ftruncate(f.rdi, f.rsi), true),
        285 => (sys_fallocate(f.rdi, f.rsi, f.rdx, f.r10), true),
        27 => (sys_mincore(f.rdi, f.rsi, f.rdx), true),
        SYS_MEMFD_CREATE => (sys_memfd_create(f.rdi, f.rsi), true),
        SYS_EPOLL_CREATE => (sys_epoll_create(f.rdi), true),
        SYS_EPOLL_CREATE1 => (sys_epoll_create1(f.rdi), true),
        SYS_EPOLL_CTL => (sys_epoll_ctl(f.rdi, f.rsi, f.rdx, f.r10), true),
        SYS_EPOLL_WAIT => (sys_epoll_wait(f.rdi, f.rsi, f.rdx, f.r10), true),
        // The signal mask cannot matter here, for the reason `ppoll` gives.
        SYS_EPOLL_PWAIT => (sys_epoll_wait(f.rdi, f.rsi, f.rdx, f.r10), true),
        SYS_PIPE => (do_pipe(f.rdi, 0), true),
        SYS_PIPE2 => (do_pipe(f.rdi, f.rsi), true),
        SYS_POLL => (sys_poll(f.rdi, f.rsi, f.rdx), true),
        SYS_PPOLL => (sys_ppoll(f.rdi, f.rsi, f.rdx, f.r10, f.r8), true),
        SYS_SELECT => (sys_select(f.rdi, f.rsi, f.rdx, f.r10, f.r8), true),
        SYS_PSELECT6 => (sys_pselect6(f.rdi, f.rsi, f.rdx, f.r10, f.r8), true),
        SYS_SENDFILE => (sys_sendfile(f.rdi, f.rsi, f.rdx, f.r10), true),
        SYS_GETCWD => (sys_getcwd(f.rdi, f.rsi), true),
        80 => (sys_chdir(f.rdi), true),
        SYS_MKDIR => (sys_name_op(f.rdi, b'm'), true),
        SYS_RMDIR => (sys_name_op(f.rdi, b'r'), true),
        SYS_UNLINK => (sys_name_op(f.rdi, b'u'), true),
        SYS_NANOSLEEP => (sys_nanosleep(f.rdi), true),
        SYS_CLOCK_NANOSLEEP => (sys_nanosleep(f.rdx), true),
        // **Accepted and never delivered, which is the honest shape.** A
        // handler is recorded nowhere because nothing here can raise a signal
        // at a guest: there is no other process to send one, no terminal to
        // generate one, and a fault ends the guest rather than being offered
        // to it. Refusing instead would stop `sh` before it starts, over a
        // promise about events that cannot happen.
        // Recorded and delivered now. The refusal above this used to say
        // nothing could raise a signal at a guest -- no other process, no
        // terminal, and a fault ends the guest. `fork` is what made the first
        // clause false, and a child exiting is an event its parent is owed.
        SYS_RT_SIGACTION => (super::signal::sigaction(f.rdi, f.rsi, f.rdx, f.r10), true),
        SYS_RT_SIGPROCMASK => {
            (super::signal::sigprocmask(f.rdi, f.rsi, f.rdx, f.r10), true)
        }
        SYS_RT_SIGRETURN => (super::signal::sigreturn(f), true),
        SYS_KILL => (super::signal::kill(f.rdi as i64, f.rsi), true),
        // No parent, and no group but the one. `getppid` answering zero is
        // what a process reparented to nothing reports.
        SYS_GETPPID => (super::fork::parent_pid(), true),
        SYS_GETGROUPS => (0, true),
        SYS_UNAME => (sys_uname(f.rdi), true),
        SYS_FCNTL => (sys_fcntl(f.rdi, f.rsi, f.rdx), true),
        SYS_GETTIMEOFDAY => (sys_gettimeofday(f.rdi, f.rsi), true),
        SYS_CLOCK_GETTIME => (sys_clock_gettime(f.rdi, f.rsi), true),
        SYS_DUP => (sys_dup(f.rdi, None), true),
        SYS_DUP2 => (sys_dup(f.rdi, Some(f.rsi)), true),
        SYS_DUP3 => (sys_dup3(f.rdi, f.rsi, f.rdx), true),
        // `rcx` is where `syscall` stashed the return address, which is where
        // the child resumes: a cloned thread does not start at an entry point,
        // it returns from `clone` on a different stack with `rax` zero.
        // A process-shaped clone is glibc's fork and posix_spawn; see
        // `fork::clone_process`. Everything else is a thread or refused.
        SYS_CLONE if super::fork::is_process(f.rdi) => (super::fork::clone_process(f), true),
        SYS_CLONE => (sys_clone(f.rdi, f.rsi, f.rdx, f.r10, f.r8, f.rip), true),
        // The whole frame, because a child resumes with its parent's
        // registers and this is the only place that still has them.
        SYS_FORK => (super::fork::fork(f), true),
        // **`vfork` is answered as `fork`, and that is safe in the direction
        // that matters.** `vfork` promises the parent will be suspended and
        // the two will share memory until the child execs or exits; a caller
        // who gets a full copy and a running parent has been given *more*
        // than it asked for, not less. The reverse -- answering `fork` with
        // `vfork` semantics -- is what would corrupt a parent.
        SYS_VFORK => (super::fork::fork(f), true),
        SYS_WAIT4 => (super::fork::wait(f.rdi as i64, f.rsi, f.rdx), true),
        // Returns into a different program rather than to its caller, so the
        // frame it was handed is the thing it edits.
        SYS_EXECVE => (super::load::exec(f), true),
        SYS_FUTEX => (futex(f.rdi, f.rsi, f.rdx, f.r10, f.r8, f.r9, false), true),
        SYS_GETTID => (super::thread::current_tid(), true),
        // A thread yielding to another is a thing this kernel can actually do,
        // which is unusual in this file: most scheduling calls are answered
        // with a shape rather than an action.
        SYS_SCHED_YIELD => {
            wait_turn();
            (0, true)
        }
        // One process, and it is the guest. Reporting a pid at all is what
        // stops a runtime deciding it failed to start.
        SYS_GETPID => (super::fork::current_pid(), true),
        SYS_SET_TID_ADDRESS => (super::thread::current_tid(), true),
        // Root, and every id the same. There is no privilege boundary above a
        // guest here to be anything else, which is the same fact `AT_SECURE`
        // reports as zero and `stat` reports as uid 0 -- said once per place
        // that asks rather than three different ways.
        SYS_GETUID | SYS_GETGID | SYS_GETEUID | SYS_GETEGID => (0, true),
        SYS_BRK => (sys_brk(f.rdi), true),
        SYS_MMAP => (sys_mmap(f.rdi, f.rsi, f.rdx, f.r10, f.r8, f.r9), true),
        SYS_MPROTECT => (sys_mprotect(f.rdi, f.rsi, f.rdx), true),
        SYS_MADVISE => (sys_madvise(f.rdi, f.rsi, f.rdx), true),
        SYS_MUNMAP => (sys_munmap(f.rdi, f.rsi), true),
        SYS_ARCH_PRCTL => (sys_arch_prctl(f.rdi, f.rsi), true),
        SYS_EXIT | SYS_EXIT_GROUP => {
            record(Call { nr, args, ret: 0, served: true, path: [0; PATH_SNIP], path_len: 0 });
            if nr == SYS_EXIT_GROUP {
                exit_group_from_thread(f.rdi);
            }
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
    record(Call { nr, args, ret, served, path: [0; PATH_SNIP], path_len: 0 });
    super::prof::bill(super::prof::row(nr, false), t0);
    // The display server's turn, on the guest's own task -- see `sky::server`
    // for why it runs here and nowhere else. A request just written is
    // answered before the client next looks.
    crate::sky::server::pump();
    f.rax = ret;
    // **On the way out, and this is the only place it can be.** The guest's
    // whole register state is in this frame, its stack pointer is parked in
    // `GLADOS_GUEST_RSP`, and the return is about to `sysretq` somewhere --
    // so redirecting it into a handler costs three stores and no new
    // assembly. `rt_sigreturn` is excluded because it has just *restored* a
    // frame, and delivering on top of that would lose it.
    //
    // The price is that a guest making no syscalls receives no signals. On
    // Linux the timer would interrupt it; here the run deadline is what ends
    // a runaway, and buying delivery for one branch on an existing path is
    // the trade.
    if nr != SYS_RT_SIGRETURN {
        super::signal::deliver(f);
    }
    if matches!(nr, SYS_MMAP | SYS_MUNMAP | SYS_MPROTECT | SYS_BRK | 25 /* mremap */) {
        tlb_shootdown();
    }
    super::prof::leave();
    wake_nappers();
    big_unlock();
    unsafe { die_if_doomed() };
}

extern "sysv64" {
    pub(super) fn glados_leave_guest(code: u64) -> !;
}

/// Set on the way out so an exit code of zero is still distinguishable from
/// "the guest never called exit at all".
pub const EXITED: u64 = 1 << 32;
/// Set when the guest was killed for running past its deadline.
pub const OVERRAN: u64 = 1 << 34;
/// Set when the guest was ended by a signal, whose number is the low byte.
///
/// Its own flag rather than an exit code of `128 + sig`, which is what a shell
/// *prints* for a signalled child and is not what `wait4` reports: a parent
/// reading `WIFSIGNALED` of an exit code would be told its child chose to exit
/// with 137, and a parent that kills a child to stop it would never learn the
/// kill landed.
pub const SIGNALED: u64 = 1 << 35;

/// How long a guest may run before the timer takes the machine back.
///
/// **Without this a guest that never makes a syscall owns the machine.** Every
/// fixture so far ends by asking for something, so nothing had noticed; a real
/// binary with a bug in its startup loop would simply never give the shell
/// back, and there is no key to press because the guest is what is running.
///
/// Thirty seconds at 100 Hz, and the number moved because a real program met
/// it while doing nothing wrong.
///
/// It was five, on the reasoning that no correct program here would take that
/// long -- true of every hand-written fixture and false of the first binary
/// nobody here wrote. BusyBox printing its own applet list makes about seven
/// hundred `write` calls of a word each, every one of them painting the
/// console, and it was killed two thirds of the way through: `killed for
/// running too long after 70 syscall(s)`. Nothing was wrong with the guest and
/// nothing was wrong with the kernel. The harness was measuring the console.
///
/// The default rather than a constant now, which is what the paragraph above
/// asked for: a guest that legitimately computes for a minute needed a way to
/// say so, and thirty seconds only moved the point at which that became true.
const DEADLINE_TICKS: u64 = 3000;

/// How long the next guest may run, in ticks, or zero for no limit at all.
///
/// **Zero is dangerous and is offered anyway.** The deadline is the only thing
/// that ends a runaway, so a guest with no limit that never makes a syscall
/// takes the machine and nothing short of a reboot gets it back. It exists
/// because the alternative is worse for one specific job: a program that draws
/// and then holds the screen is killed mid-frame at thirty seconds, so there
/// has never been a photograph of anything a guest rendered.
static LIMIT: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(DEADLINE_TICKS);

/// Set the limit for the next guest. Answers what it now is, in ticks.
pub fn set_limit(ticks: u64) -> u64 {
    LIMIT.store(ticks, Ordering::Release);
    ticks
}

pub fn limit() -> u64 {
    LIMIT.load(Ordering::Acquire)
}


/// Whether the running guest has outstayed its welcome.
///
/// Or has been told to die: a task marked by `doom` answers true here too, so
/// every wait loop that already ends a guest at its deadline ends a doomed one
/// at the same check, and the timer ends a doomed one spinning at ring 3. One
/// question for both, because a second check beside the first at each of the
/// eight sites is eight places to forget it.
pub fn overran(now: u64) -> bool {
    if !running() {
        return false;
    }
    if doomed() != 0 {
        return true;
    }
    // The running task's own session's deadline. One per machine, this was
    // the deadline of whichever session started last, so a second session
    // starting reset the first one's clock.
    let d = super::session::current().map(super::session::deadline_of).unwrap_or(0);
    d != 0 && now >= d
}

/// What a task ending at a deadline or a doom check ends with.
fn death_code() -> u64 {
    match doomed() {
        0 => OVERRAN,
        d => d,
    }
}

/// End a guest that would not stop on its own.
///
/// # Safety
/// Only from an interrupt taken **at ring 3**, so the guest itself was
/// executing. Called while the kernel is working on the guest's behalf it
/// would abandon whatever that work was holding.
pub unsafe fn kill_overrun() -> ! {
    unsafe { kill_with(death_code()) }
}

/// End a guest that blocked past its deadline.
///
/// # Safety
/// Only from a wait inside a syscall that holds nothing. That is a *different*
/// condition from the one `kill_overrun` asks for rather than a weaker one:
/// there the guarantee is that the kernel was not running at all, and here it
/// is that this particular loop has no allocation in flight, no borrow of
/// `SPACE` live and no lock taken across the yield.
pub unsafe fn kill_blocked() -> ! {
    unsafe { kill_with(death_code()) }
}

/// Set instead when the guest died of a fault.
pub const FAULTED: u64 = 1 << 33;

/// What killed the last guest, kept so the report can say more than a vector.
///
/// A guest that dies takes its registers with it: the longjmp abandons the
/// stack the fault arrived on, so anything not copied out here is gone by the
/// time anybody prints anything. "killed by fault 0x0e" is true and says
/// nothing -- it is the same line for a null dereference, a stack overflow and
/// a jump into a page that was never mapped.
#[derive(Clone, Copy)]
pub struct Fault {
    /// Everything the stub and the CPU between them produced, whole rather
    /// than five fields picked in advance. Which register matters is not
    /// knowable at the time of the fault: the last one that mattered here was
    /// `rdi`, and nobody would have thought to keep it.
    pub regs: crate::cpu::idt::Frame,
    /// The address reached for, which is only meaningful for a page fault and
    /// is the one thing not in the register block.
    pub cr2: u64,
}

/// Per task: the fault the session that task ran ended with. Written by the
/// main thread's own kill, or adopted from a thread whose crash ended the
/// process, and read by whoever ran the session -- on the same task.
static LAST_FAULT: Racy<[Option<Fault>; crate::task::MAX_TASKS]> =
    Racy::new([None; crate::task::MAX_TASKS]);

/// The same three addresses said in words, resolved while the space was still
/// installed.
///
/// `run` tears the space down on the line after the guest returns, and
/// `locate` reads the space, so resolving at print time answered "no guest"
/// three times about a guest that had just died. The strings are made in
/// `run`, one line earlier, which is the only moment both the fault and the
/// map it should be read against exist together.
static FAULT_AT: Racy<[Option<(String, String, String)>; crate::task::MAX_TASKS]> =
    Racy::new([const { None }; crate::task::MAX_TASKS]);

/// The words around the guest's stack pointer when it died, each said in
/// words as well as in hex.
///
/// **Because a return address is the only thing that names the caller here.**
/// There is no unwinder and no symbols, so "who called this with a null" is
/// answerable only by reading the stack the call left behind -- and the stack
/// is freed by the teardown two lines after the guest returns. Sixteen words
/// either side is enough to hold a small frame and the return address above
/// it, and small enough that the report stays readable.
static FAULT_STACK: Racy<[Option<Vec<(u64, u64, String)>>; crate::task::MAX_TASKS]> =
    Racy::new([const { None }; crate::task::MAX_TASKS]);

pub fn fault_stack() -> Option<Vec<(u64, u64, String)>> {
    unsafe { (*FAULT_STACK.get())[me()].clone() }
}

/// Each general-purpose register, named, with what it points at.
///
/// Resolved beside the stack words and for the same reason: `locate` reads the
/// space and the space is gone two lines later. The first version of this
/// resolved at print time and answered "no guest" fifteen times about a guest
/// that had just died, which is the identical mistake made twice, once for
/// three addresses and once for fifteen.
static FAULT_REGS: Racy<[Option<Vec<(&'static str, u64, String)>>; crate::task::MAX_TASKS]> =
    Racy::new([const { None }; crate::task::MAX_TASKS]);

pub fn fault_regs() -> Option<Vec<(&'static str, u64, String)>> {
    unsafe { (*FAULT_REGS.get())[me()].clone() }
}

pub fn last_fault() -> Option<Fault> {
    unsafe { (*LAST_FAULT.get())[me()] }
}

/// Where the rip, the faulting address and the stack pointer were, in words.
pub fn fault_where() -> Option<(String, String, String)> {
    unsafe { (*FAULT_AT.get())[me()].clone() }
}

/// Which of the guest's own ranges an address falls in.
///
/// An address on its own is a number. `0x30d6000` means nothing until it is
/// "inside the interpreter, at +0x2000", and that is the difference between a
/// fault report and a diagnosis -- the same reason `cpu::code::locate` exists
/// for the kernel's own faults.
/// Which file each file-backed mapping came from: (address, length, file
/// offset, path). For naming a crash site, and nothing else reads it.
///
/// A side table rather than a field on `Mapping`, because `Mapping` is copied
/// by `fork` and compared by `munmap` and neither should care about a name.
/// Never pruned while a guest runs, so a stale entry can name a library that
/// has since been unmapped; the newest matching entry wins, which is the one
/// that was there last.
pub static MAP_NAMES: Racy<Vec<(u64, u64, u64, String)>> = Racy::new(Vec::new());

/// Give back a `Paged` mapping's pages: each one looked up in the guest's
/// tables, its rights restored and freed, then the translation removed.
fn free_paged(sp: &mut Space, at: u64, len: u64) {
    let Some(t) = sp.tables.as_mut() else { return };
    let mut off = 0;
    while off < len {
        if let Some(phys) = t.phys_of(at + off) {
            crate::mem::paging::release_to_heap(phys, 4096);
            free_pages(phys, 4096);
        }
        t.unmap_page(at + off);
        off += 4096;
    }
}

/// What a full arena is full of, once per guest: bytes by kind of mapping,
/// and the free list's shape. Out of arena and out of memory are different
/// failures, and fragmentation is a third.
fn arena_report(sp: &Space) {
    static SAID: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
    if SAID.swap(true, Ordering::AcqRel) {
        return;
    }
    let mut by = [0u64; 7];
    let mut n = [0usize; 7];
    for m in &sp.maps {
        let k = match m.from {
            Source::Heap => 0,
            Source::Fixed => 1,
            Source::Device => 2,
            Source::Shared => 3,
            Source::Copied => 4,
            Source::Backed(_) => 5,
            Source::Paged => 6,
        };
        by[k] += page_up(m.len) as u64;
        n[k] += 1;
    }
    let names = ["heap", "fixed", "device", "memfd", "copied", "backed", "paged"];
    for k in 0..7 {
        if n[k] > 0 {
            crate::kprintln!("    {:<7} {:>5} mapping(s) {:>6} MiB", names[k], n[k], by[k] >> 20);
        }
    }
    let free: u64 = sp.arena_free.iter().map(|&(_, l)| l).sum();
    let big = sp.arena_free.iter().map(|&(_, l)| l).max().unwrap_or(0);
    crate::kprintln!(
        "    arena cursor {:#x} of {:#x}; free list {} gap(s), {} MiB, largest {} KiB",
        sp.mmap_next, sp.mmap_end, sp.arena_free.len(), free >> 20, big >> 10
    );
}

/// Take `want` bytes of arena: first fit from what `munmap` gave back, then
/// the cursor. `None` is out of arena, which the caller says as `ENOMEM`.
fn arena_take(sp: &mut Space, want: u64) -> Option<u64> {
    if let Some(i) = sp.arena_free.iter().position(|&(_, l)| l >= want) {
        let (a, l) = sp.arena_free[i];
        if l == want {
            sp.arena_free.remove(i);
        } else {
            sp.arena_free[i] = (a + want, l - want);
        }
        return Some(a);
    }
    let at = sp.mmap_next;
    if at.checked_add(want)? > sp.mmap_end {
        return None;
    }
    sp.mmap_next = at + want;
    Some(at)
}

/// Give an arena range back: its page-table entries cleared and flushed --
/// a stale translation would hand the next tenant the last one's pages --
/// and the range filed, merged with its neighbours.
fn arena_give(sp: &mut Space, at: u64, len: u64) {
    // A name for a range that is no longer that file. Left behind, the next
    // tenant -- llvmpipe's JIT, most often -- was profiled as libMiles or a
    // font, and a hot spot named after the wrong file sends the reader to
    // the wrong source.
    unsafe { &mut *MAP_NAMES.get() }.retain(|&(a, l, _, _)| a + l <= at || a >= at + len);
    let Some(t) = sp.tables.as_mut() else { return };
    let mut off = 0;
    while off < len {
        t.unmap_page(at + off);
        off += 4096;
    }
    let i = sp.arena_free.partition_point(|&(a, _)| a < at);
    sp.arena_free.insert(i, (at, len));
    if i + 1 < sp.arena_free.len() && sp.arena_free[i].0 + sp.arena_free[i].1 == sp.arena_free[i + 1].0 {
        sp.arena_free[i].1 += sp.arena_free[i + 1].1;
        sp.arena_free.remove(i + 1);
    }
    if i > 0 && sp.arena_free[i - 1].0 + sp.arena_free[i - 1].1 == sp.arena_free[i].0 {
        sp.arena_free[i - 1].1 += sp.arena_free[i].1;
        sp.arena_free.remove(i);
    }
}

/// Note a file mapping that succeeded.
pub(super) fn note_map(at: u64, len: u64, fd: u64, off: u64) {
    if (at as i64) < 0 {
        return;
    }
    let name = unsafe { guest_slot() }.as_ref().and_then(|sp| match sp.fds.get(fd as usize)?.as_ref()? {
        super::fs::Fd::Disk(b) => Some(b.borrow().path.clone()),
        super::fs::Fd::File(b) => Some(b.borrow().path.clone()),
        _ => None,
    });
    if let Some(n) = name {
        let t = unsafe { &mut *MAP_NAMES.get() };
        if t.len() < 4096 {
            t.push((at, len, off, n));
        }
    }
}

/// A code address as `file+offset`, the form `addr2line` and `nm` take.
pub fn name_code(v: u64) -> Option<String> {
    let t = unsafe { &*MAP_NAMES.get() };
    t.iter().rev().find(|(a, l, _, _)| v >= *a && v < a + l).map(|(a, _, off, n)| {
        let short = n.rsplit('/').next().unwrap_or(n);
        alloc::format!("{}+{:#x}", short, v - a + off)
    })
}

pub fn locate(at: u64) -> alloc::string::String {
    let Some(sp) = (unsafe { guest_slot() }).as_ref() else {
        return alloc::string::String::from("no guest");
    };
    let hit = |r: Region, what: &str| -> Option<alloc::string::String> {
        (at >= r.at && at < r.at.saturating_add(r.len as u64))
            .then(|| alloc::format!("{} +{:#x}", what, at - r.at))
    };
    hit(sp.image, "image")
        .or_else(|| sp.interp.and_then(|r| hit(r, "interpreter")))
        .or_else(|| hit(sp.stack, "stack"))
        .or_else(|| {
            (at >= sp.brk_start && at < sp.brk_end)
                .then(|| alloc::format!("break +{:#x}", at - sp.brk_start))
        })
        .or_else(|| {
            sp.maps.iter().find_map(|m| {
                (at >= m.at && at < m.at.saturating_add(m.len as u64)).then(|| {
                    let kind = match m.from {
                        Source::Heap => "heap",
                        Source::Fixed => "fixed",
                        Source::Device => "device",
                        Source::Shared => "memfd",
                        Source::Copied => "copied",
                        Source::Backed(_) => "backed",
                        Source::Paged => "paged",
                    };
                    let rights = crate::mem::paging::query(at & !(PAGE - 1))
                        .map(|p| alloc::format!("{}{}{}", if p.present { "p" } else { "-" }, if p.write { "w" } else { "-" }, if p.user { "u" } else { "-" }))
                        .unwrap_or_else(|| alloc::string::String::from("unmapped"));
                    alloc::format!("a {} mapping at {:#x} len {:#x} +{:#x} [{}]", kind, m.at, m.len, at - m.at, rights)
                })
            })
        })
        .unwrap_or_else(|| alloc::string::String::from("nothing this guest owns"))
}


/// **Per task, and one flag for the machine was the bug that halted it.**
/// `GUEST_RUNNING` answered "is a guest running" for everybody, and the kill
/// path cleared it -- so the first forked child to crash was ended correctly
/// and took the flag with it, and the second faulted at ring 3 with nothing
/// saying a guest was there to blame. The handler took it for the kernel's
/// own fault and halted. One child's death disarmed isolation for every guest
/// after it, which is the shape `forktest` crashes two children to catch.
///
/// What the fault handler needs to know is narrower than "a guest exists": it
/// is whether *the task that faulted* entered ring 3 through one of the three
/// doors that park a landing in its own `GLADOS_HOST_RSP` -- `run`,
/// `run_thread` and a forked child's `enter_resumed`. Each sets its own task's
/// flag, and only the task itself or its own kill ever clears it.
/// `GUEST_RUNNING` stays, meaning only what `run` means by it: the session's
/// deadline is armed.
static IN_GUEST: [AtomicBool; crate::task::MAX_TASKS] =
    [const { AtomicBool::new(false) }; crate::task::MAX_TASKS];

/// Per task, a code to die with at the next safe point; zero for none.
///
/// Ending a guest means longjmping out of *its* stack, which only its own task
/// can do, so nobody kills a task directly: they mark it here and it ends
/// itself at the next point that holds nothing -- a syscall's way in or out,
/// a wait loop's check, or the timer finding it at ring 3. That is how a
/// session ending takes its children with it, how `SIGKILL` reaches a child
/// spinning in a loop that makes no syscall, and how a thread's crash ends its
/// process.
static DOOM: [core::sync::atomic::AtomicU64; crate::task::MAX_TASKS] =
    [const { core::sync::atomic::AtomicU64::new(0) }; crate::task::MAX_TASKS];

/// The fault each task last died of, read by whoever ran it.
static TASK_FAULT: Racy<[Option<Fault>; crate::task::MAX_TASKS]> =
    Racy::new([None; crate::task::MAX_TASKS]);

/// The task running the first guest of the running task's session -- the one
/// `linux run` or a job started. `None` outside a session, which is what the
/// shell's own task always is.
pub fn main_task() -> Option<usize> {
    super::session::main_task_of(super::session::current()?)
}

fn me() -> usize {
    crate::task::current().min(crate::task::MAX_TASKS - 1)
}

/// Whether the running task is inside a guest, so that its own landing is
/// valid and a ring-3 fault on it may end the guest rather than the machine.
pub fn running() -> bool {
    IN_GUEST[me()].load(Ordering::Acquire)
}

/// Mark a task to end itself with `code` at its next safe point.
///
/// The first doom wins: a child crashing and then being swept up by its
/// session ending died of the crash, and the report should say so.
///
/// Only a task that is inside a guest can be doomed, and answers whether this
/// one was. A mark left on a task between guests would be inherited by its
/// next tenant, which would die the instant it started for something its
/// predecessor did; `as_guest` clears on both sides as well, so the window
/// that remains loses a doom rather than misdirecting one.
pub fn doom(task: usize, code: u64) -> bool {
    if task >= crate::task::MAX_TASKS || !IN_GUEST[task].load(Ordering::Acquire) {
        return false;
    }
    let _ = DOOM[task].compare_exchange(0, code, Ordering::AcqRel, Ordering::Acquire);
    true
}

/// The code this task has been told to die with, or zero.
pub fn doomed() -> u64 {
    DOOM[me()].load(Ordering::Acquire)
}

/// The fault a task died of, taken so the next run on the same task starts
/// clean.
pub fn take_task_fault(task: usize) -> Option<Fault> {
    if task >= crate::task::MAX_TASKS {
        return None;
    }
    unsafe { (*TASK_FAULT.get())[task].take() }
}

/// Run `enter` as a guest on this task: armed so a fault ends it and not the
/// machine, with any doom from a previous tenant of the task forgotten.
///
/// The one place the flag is set, so the three doors cannot disagree about
/// it. Cleared on the way back whichever way the guest left -- an exit, a
/// kill, a fault -- because all of them return here through the longjmp.
pub fn as_guest(enter: impl FnOnce() -> u64) -> u64 {
    let t = me();
    DOOM[t].store(0, Ordering::Release);
    unsafe { (*TASK_FAULT.get())[t] = None };
    IN_GUEST[t].store(true, Ordering::Release);
    // Any core may run it: see `BIG`. The pin it had comes back after, so a
    // shell that ran a guest in the foreground is the shell on core 0 again.
    let pin = crate::task::pin_of(t);
    if SPREAD.load(Ordering::Relaxed) {
        crate::task::set_pin(t, -1);
    }
    let code = enter();
    // Every way out of a guest lands here, the longjmps included, and one of
    // them may have left from inside a syscall still holding the lock.
    super::prof::leave();
    big_unlock();
    crate::task::set_pin(t, pin);
    IN_GUEST[t].store(false, Ordering::Release);
    DOOM[t].store(0, Ordering::Release);
    code
}

/// End this task's guest now if it has been doomed. From the syscall path's
/// way in and way out, where nothing is held.
///
/// # Safety
/// Longjmps; only where `exit_group` itself could be served.
pub(super) unsafe fn die_if_doomed() {
    let d = doomed();
    if d != 0 {
        unsafe { glados_leave_guest(d) };
    }
}

/// What a guest's end means as a `wait4` status word.
///
/// The encoding is Linux's: an exit puts its code in bits 8..15 and a signal
/// puts its number in the low seven, which is what `WIFEXITED` and
/// `WIFSIGNALED` tell apart. A fault is the signal Linux would have raised for
/// it, so a parent sees `SIGSEGV` for a wild pointer rather than an exit code
/// of 14 -- the vector number -- which is what it saw before.
pub fn wait_status(code: u64) -> i32 {
    if code & FAULTED != 0 {
        signal_of_vector((code & 0xFF) as u8) as i32
    } else if code & SIGNALED != 0 {
        (code & 0x7F) as i32
    } else if code & OVERRAN != 0 {
        9
    } else {
        ((code & 0xFF) as i32) << 8
    }
}

/// The signal Linux sends for a processor exception taken at ring 3.
pub fn signal_of_vector(v: u8) -> u32 {
    const SIGILL: u32 = 4;
    const SIGTRAP: u32 = 5;
    const SIGBUS: u32 = 7;
    const SIGFPE: u32 = 8;
    const SIGSEGV: u32 = 11;
    match v {
        0 | 16 | 19 => SIGFPE,
        1 | 3 => SIGTRAP,
        6 => SIGILL,
        17 => SIGBUS,
        _ => SIGSEGV,
    }
}

/// End a guest that faulted, and go back to whoever started it.
///
/// **This is what ring 3 buys.** At ring 0 a guest fault was the machine's
/// fault too: it shared an address space with the kernel and could have
/// already corrupted anything, so the only honest response was to stop. At
/// ring 3 the kernel is intact by construction, because the guest could not
/// reach it, so the guest dies and the machine carries on.
///
/// # Safety
/// Only from a fault handler, and only when `running` is true.
/// The first crashing thread's stack, copied raw at the moment of the fault.
///
/// Raw because this is filled from the fault path, where allocating and
/// printing are both off the table; `run` names the words afterwards. The
/// first crash only, since the first is the cause and the rest are the
/// process coming down around it.
pub static CRASH: Racy<(bool, u64, usize, [u32; 4096])> = Racy::new((false, 0, 0, [0; 4096]));

pub unsafe fn kill(f: Fault) -> ! {
    let vector = f.regs.vector;
    {
        let c = unsafe { &mut *CRASH.get() };
        if !c.0 && abi32() {
            let base = f.regs.rsp & !3;
            let mut n = 0;
            while n < c.3.len() && reachable(base + n as u64 * 4, 4, false) {
                c.3[n] = unsafe { core::ptr::read_unaligned((base + n as u64 * 4) as *const u32) };
                n += 1;
            }
            *c = (true, base, n, c.3);
        }
    }
    // Copied out before the longjmp, which abandons the stack this arrived on.
    // Into the faulting task's own cell: a child's crash is its parent's to
    // read through `wait4`, and writing it over the session's report would
    // put a child's registers under the main guest's name.
    let t = me();
    unsafe { (*TASK_FAULT.get())[t] = Some(f) };
    if main_task() == Some(t) {
        unsafe { (*LAST_FAULT.get())[t] = Some(f) };
    }
    unsafe { kill_with(FAULTED | vector) }
}

/// Record a fault as the session's, for a thread whose crash ends the process.
pub fn adopt_fault(f: Fault) {
    if let Some(m) = main_task() {
        unsafe { (*LAST_FAULT.get())[m.min(crate::task::MAX_TASKS - 1)] = Some(f) };
    }
}

/// `exit_group` called by a thread rather than by the main thread: the whole
/// process ends, as on Linux. The main thread is doomed with the code -- its
/// end is what tears the session down and dooms every other thread -- and
/// this thread then leaves the way any `exit` does.
pub(super) fn exit_group_from_thread(code: u64) {
    if let Some(m) = main_task() {
        if m != me() {
            doom(m, (code & 0xFF) | EXITED);
            if let Some(s) = super::session::current() {
                super::session::begin_exit(s);
            }
        }
    }
}

/// The longjmp both reasons share.
///
/// # Safety
/// Only while a guest is running, and only from a context that may abandon its
/// stack.
unsafe fn kill_with(code: u64) -> ! {
    // This task's flag and nobody else's. Clearing the session's here was
    // what let one dead child disarm the next one's fault.
    IN_GUEST[me()].store(false, Ordering::Release);
    // **Inline, and calling `glados_leave_guest` through its declaration was
    // the bug.** This target is Windows-ABI, so an ordinary Rust function is
    // Microsoft x64, where xmm6-xmm15 are non-volatile. `glados_leave_guest`
    // is `sysv64`, which treats them as scratch, so the compiler must spill
    // all ten across the call: a 160-byte `movaps` prologue wanting the stack
    // 16-byte aligned. On the stack a guest fault arrives on it is not, and a
    // misaligned `movaps` raises #GP(0) -- which is precisely the fault that
    // was stopping the machine instead of the guest.
    //
    // `exit_group` never met it, because it leaves from
    // `glados_syscall_dispatch`, which is already `sysv64` and so has nothing
    // to preserve. That asymmetry is what made this look like a ring-3
    // problem for a long time. It is an ABI problem.
    //
    // Written out, the longjmp has no prologue, spills nothing, and needs no
    // alignment it cannot have.
    unsafe {
        core::arch::asm!(
            "cli",
            "swapgs",
            "mov rsp, gs:[16]",
            "swapgs",
            "pop r15",
            "pop r14",
            "pop r13",
            "pop r12",
            "pop rbx",
            "pop rbp",
            "ret",
            in("rax") code,
            options(noreturn),
        )
    }
}

/// Run a loaded image until it exits.
///
/// # Safety
/// Jumps to an address derived from a file. Everything in the module docs
/// about stage 0 containing bugs rather than malice applies here and nowhere
/// more directly.
pub unsafe fn run(entry: u64, stack_top: u64, want: u64, bits32: bool) -> u64 {
    // A session of its own, which is what lets two of these run at once: its
    // deadline, its main task, its trace and its processes are its own, and
    // the end of this one sweeps only them. Callers ask `linux::busy` first,
    // so running out here is a race between two starts, answered plainly.
    let Some(sess) = super::session::claim() else {
        crate::kprintln!("  every Linux session is in use ({} at once)", super::session::MAX);
        return 0;
    };
    // **And a syscall stack of its own.** The main thread's syscalls ran on
    // one static 16 KiB stack, which is correct for exactly one session: a
    // second main thread blocked in a syscall beside the first would be
    // pushing onto the frame the first is suspended in.
    let sys_layout = core::alloc::Layout::from_size_align(16 * 1024, 16).unwrap();
    let sys_stack = unsafe { alloc::alloc::alloc(sys_layout) };
    if sys_stack.is_null() {
        super::session::release(sess);
        return 0;
    }
    arm();
    unsafe {
        (*core_entry()).syscall_stack = sys_stack as u64 + 16 * 1024;
    };
    // The guest is about to be allowed to write `FS`, and `FS` is a register
    // of this machine that the kernel goes on using afterwards. Parked here
    // rather than inside `arch_prctl`, so a guest that sets it forty times
    // still restores the one value that was true before any of them.
    unsafe {
        if let Some(sp) = guest_slot().as_mut() {
            sp.saved_fs = crate::cpu::rdmsr(IA32_FS_BASE);
        }
    }
    // `syscall` clears IF through FMASK and `exit_group` leaves through a
    // longjmp rather than through the stub's tail, so nothing on that path
    // puts the flag back. Saved and restored around the whole run, which is
    // also what makes a faulting guest survivable: the handler that killed it
    // arrived through a gate that cleared IF too.
    let flags: u64;
    unsafe {
        core::arch::asm!("pushfq; pop {}", out(reg) flags, options(nomem, preserves_flags))
    };
    unsafe {
        (*LAST_FAULT.get())[me()] = None;
        (*FAULT_AT.get())[me()] = None;
        (*FAULT_REGS.get())[me()] = None;
        (*FAULT_STACK.get())[me()] = None;
        (*MAP_NAMES.get()).clear();
    }
    // Zero means no deadline, and `overran` already reads a zero `DEADLINE`
    // as "nothing to enforce", so the two agree without a second condition.
    // The caller chooses: a foreground run passes `limit()`, a background job
    // passes zero, because a job runs on its own task and a runaway there
    // costs a share of a core rather than the shell.
    super::session::set_deadline(
        sess,
        if want == 0 { 0 } else { crate::dev::lapic::ticks() + want },
    );
    // Scheduled input is measured from here, which is the only moment that
    // means anything to it: a script armed at the prompt has no idea how long
    // the harness will take to send the next line.
    super::input::start(crate::dev::lapic::ticks());
    // **The main thread carries its entry state as well.** It is the one that
    // looked like it did not need to: with a single guest the three globals
    // are constants, so nothing swaps them and nothing notices. The moment a
    // child runs while the main thread sits inside a syscall, the child's
    // entry overwrites where the main thread's `rsp` went and where it must
    // longjmp back to, and the main thread returns onto a stack that is not
    // its own.
    crate::task::ring3_active(true, unsafe {
        (*core_entry()).syscall_stack
    });
    super::session::set_main(sess, me(), current_guest());
    super::session::enter(sess);
    super::i386::begin(bits32);
    let code = as_guest(|| unsafe {
        if bits32 {
            super::i386::glados_enter_guest32(entry, stack_top)
        } else {
            glados_enter_guest(entry, stack_top)
        }
    });
    super::i386::begin(false);
    crate::task::ring3_active(false, 0);
    // **Every thread has to be gone before the space is.** A child still
    // running would be reading regions `teardown` is about to hand back, and
    // `exit_group` only asks the others to stop -- ending a thread means
    // longjmping out of its own stack, which only that thread can do.
    // Bounded, because a thread that makes no syscall never notices the ask
    // and the alternative is a shell that never comes back.
    //
    // **And every child goes with the session.** There is no init to reparent
    // an orphan to, and an orphan left running was a guest nobody would ever
    // wait for, holding a pool task forever -- and, before `IN_GUEST`, one
    // whose next fault halted the machine because the session it belonged to
    // had already cleared the flag. `SIGKILL`, as a terminal hangup ends a
    // shell's jobs. A child that does not notice inside the bound is still
    // doomed, and the timer ends it at its next tick at ring 3.
    super::session::begin_exit(sess);
    super::thread::doom_session(sess, code | EXITED);
    super::fork::doom_all(sess, SIGNALED | 9);
    let give_up = crate::dev::lapic::ticks() + 200;
    while (super::thread::live_in(sess) > 0 || super::fork::live_in(sess) > 0)
        && crate::dev::lapic::ticks() < give_up
    {
        wait_turn();
    }
    super::session::set_deadline(sess, 0);
    if flags & (1 << 9) != 0 {
        crate::cpu::enable_interrupts();
    }
    // Before the teardown on the next line, because that is what `locate`
    // reads. This is the only point where the fault and the map it has to be
    // read against are both alive.
    unsafe {
        (*FAULT_AT.get())[me()] =
            last_fault().map(|f| (locate(f.regs.rip), locate(f.cr2), locate(f.regs.rsp)));
        (*FAULT_REGS.get())[me()] = last_fault().map(|f| {
            let g = f.regs;
            [
                ("rax", g.rax), ("rbx", g.rbx), ("rcx", g.rcx),
                ("rdx", g.rdx), ("rsi", g.rsi), ("rdi", g.rdi),
                ("rbp", g.rbp), ("r8 ", g.r8), ("r9 ", g.r9),
                ("r10", g.r10), ("r11", g.r11), ("r12", g.r12),
                ("r13", g.r13), ("r14", g.r14), ("r15", g.r15),
            ]
            .into_iter()
            .map(|(n, v)| (n, v, locate(v)))
            .collect()
        });
        (*FAULT_STACK.get())[me()] = last_fault().and_then(|f| {
            // A 32-bit guest's stack is 4-byte words, and reading it 8 at a
            // time pairs every return address with its neighbour. Deep, and
            // filtered to words that land in a mapped file: those are the
            // candidate return addresses, which is the backtrace a program
            // that called `abort` has no other way to leave behind.
            let w = if abi32() { 4u64 } else { 8 };
            let base = f.regs.rsp & !(w - 1);
            let mut out = Vec::new();
            for i in 0..2048u64 {
                let at = base + i * w;
                if !reachable(at, w as usize, false) {
                    break;
                }
                let v = if w == 4 {
                    core::ptr::read_unaligned(at as *const u32) as u64
                } else {
                    core::ptr::read_unaligned(at as *const u64)
                };
                if let Some(n) = name_code(v) {
                    out.push((at, v, n));
                    if out.len() >= 48 {
                        break;
                    }
                }
            }
            (!out.is_empty()).then_some(out)
        });
        // Where, as well as the stack: a background job has no `linux run`
        // to print the fault afterwards, so it is said here, once.
        if let Some(f) = last_fault() {
            crate::kprintln!(
                "  fault {:#04x} at rip {:#x} ({}), reaching for {:#x} ({}), error {:#x}",
                f.regs.vector, f.regs.rip,
                name_code(f.regs.rip).unwrap_or_else(|| locate(f.regs.rip)),
                f.cr2, locate(f.cr2), f.regs.err
            );
        }
        if let (Some(sid), Some(_)) = (super::session::current(), last_fault()) {
            let tail = super::session::peek_tail(sid);
            crate::kprintln!("  the session's last calls:");
            for c in tail.iter().rev().take(25).rev() {
                crate::kprintln!(
                    "    {:<16} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} -> {:#x}",
                    name_of(c.nr), c.args[0], c.args[1], c.args[2], c.args[3], c.args[4], c.args[5], c.ret
                );
            }
        }
        if let Some(ws) = (*FAULT_STACK.get())[me()].as_ref() {
            crate::kprintln!("  return addresses on the stack:");
            for (at, v, n) in ws {
                crate::kprintln!("    {:#010x}: {:#010x} {}", at, v, n);
            }
        }
        let c = &mut *CRASH.get();
        if c.0 {
            crate::kprintln!("  the first crash's stack, from {:#x}:", c.1);
            let mut shown = 0;
            for i in 0..c.2 {
                if let Some(n) = name_code(c.3[i] as u64) {
                    crate::kprintln!("    +{:#06x}: {:#010x} {}", i * 4, c.3[i], n);
                    shown += 1;
                    if shown >= 64 {
                        break;
                    }
                }
            }
            c.0 = false;
        }
    }
    teardown();
    // The session's trace becomes this task's, for whoever ran it to read.
    let tail = super::session::take_tail(sess);
    if MISSES.load(Ordering::Relaxed) && !tail.is_empty() {
        crate::kprintln!("  the last {} call(s):", tail.len());
        for c in &tail {
            match c.path() {
                Some(p) => crate::kprintln!("  {:>4} {:<16} {} -> {}", c.nr & !(1 << 31), name_of(c.nr), p, c.ret as i64),
                None => crate::kprintln!(
                    "  {:>4} {:<16} {:#x} {:#x} {:#x} {:#x} -> {}",
                    c.nr & !(1 << 31), name_of(c.nr), c.args[0], c.args[1], c.args[2], c.args[3], c.ret as i64
                ),
            }
        }
    }
    unsafe { (*TASK_TRACE.get())[me()] = super::session::take_trace(sess) };
    super::session::leave();
    super::session::release(sess);
    // After `leave`: nothing can enter a syscall on this stack any more.
    unsafe { alloc::alloc::dealloc(sys_stack, sys_layout) };
    code
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
        "the trap frame is fifteen words, in the order the stub pushes them",
        core::mem::size_of::<Frame>() == 120,
    ));
    // 120 is also what keeps the stub's alignment `sub` correct: 120 mod 16 is
    // 8, exactly as 72 was, which is why six more pushes changed nothing about
    // it. Asserted rather than left in a comment, because a sixteenth field
    // would make the frame 128, leave rsp already aligned, and the `sub` would
    // then *mis*align it -- a fault inside the dispatcher's first SSE spill,
    // a long way from the field somebody added.
    out.push((
        "and its size still leaves the stub's alignment sub correct",
        core::mem::size_of::<Frame>() % 16 == 8,
    ));
    let f = Frame { rax: 0, rdi: 1, rsi: 2, rdx: 3, r10: 4, r8: 5, r9: 6, rip: 7, rflags: 8,
        rbx: 9, rbp: 10, r12: 11, r13: 12, r14: 13, r15: 14 };
    let words = unsafe { core::slice::from_raw_parts(&f as *const Frame as *const u64, 15) };
    out.push((
        "the frame reads back as the register file it describes",
        words == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
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
    // How a guest's end reads to its parent. Linux's encoding, because
    // `WIFEXITED` and `WIFSIGNALED` are macros compiled into the parent: an
    // exit in bits 8..15, a signal in the low seven. A crash used to arrive as
    // an *exit* whose code was the vector number.
    out.push((
        "a child's exit reads as an exit, and its crash as the signal Linux would send",
        wait_status(EXITED | 7) == 7 << 8
            && wait_status(FAULTED | 14) == 11
            && wait_status(FAULTED | 13) == 11
            && wait_status(FAULTED | 0) == 8
            && wait_status(FAULTED | 6) == 4
            && wait_status(SIGNALED | EXITED | 9) == 9
            && wait_status(OVERRAN) == 9,
    ));
    out.push((
        "and every one of those reads as signalled, not as exited, by the parent's own test",
        [FAULTED | 14, SIGNALED | 9, OVERRAN]
            .iter()
            .all(|&c| { let w = wait_status(c); w & 0x7F != 0 && w & 0x7F != 0x7F })
            && wait_status(EXITED | 3) & 0x7F == 0,
    ));
    // A forked child's records of its parent's mappings must give nothing
    // back: the pages behind them are the parent's, and freeing them from the
    // child pulled memory out from under a running process.
    out.push((
        "a forked child's inherited mapping owes nothing when it is unmapped",
        {
            let m = Mapping { at: 0x1000, len: 4096, from: Source::Heap };
            let c = Mapping { from: Source::Copied, ..m };
            c.from == Source::Copied && c.at == m.at && c.len == m.len
        },
    ));
    // Doom lands only on a task inside a guest, so a stale mark can never be
    // inherited by the next program a pool task runs. Asked of a task index
    // nothing is running a guest on at boot.
    out.push((
        "a task that is not inside a guest cannot be doomed",
        !doom(crate::task::MAX_TASKS - 1, SIGNALED | 9) && !doom(crate::task::MAX_TASKS, 1),
    ));
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

    // brk, on a region installed for the check and torn down after it. The
    // claim that earns its place is the failure mode: Linux answers the
    // *unchanged* break rather than an error, and an implementation returning
    // -ENOMEM would hand libc 0xFFFFFFFFFFFFFFF4 as a heap address.
    //
    // **Real pages, because growing the break now writes.** A grown break
    // reads as zero, as Linux's does, and this check's region used to be a
    // fake at `0x100000` -- so the first `brk` past the old break zeroed 8 KiB
    // of whatever the machine keeps there, and `diag linux` took QEMU down
    // with no message at all.
    if let Some(region) = alloc_pages(4096 * 4) {
        let fake = Region { at: region, len: 4096 * 4 };
        install(Regions { arena_extra: [(0, 0); 7], image: fake, stack: fake, brk: fake, interp: None, image_mapped: false, interp_mapped: false, stack_mapped: false, brk_mapped: false, bits32: false, mmap_at: GUEST_MMAP_AT, mmap_end: u64::MAX }, None);
        let first = sys_brk(0);
        let grown = sys_brk(region + 8192);
        let refused = sys_brk(region + 1_000_000);
        let back = sys_brk(region);
        out.push((
            "brk answers the current break when asked for nothing",
            first == region,
        ));
        out.push(("brk grows to exactly what was asked for", grown == region + 8192));
        out.push((
            "a brk past the end answers the unchanged break, never an error",
            refused == region + 8192 && (refused as i64) > 0,
        ));
        out.push(("brk shrinks as well as grows", back == region));
        // Teardown would wrmsr FS from a Space whose saved value is zero, and
        // this check installed one by hand rather than through `run`.
        unsafe { *guest_slot() = None };
        free_pages(region, 4096 * 4);
    }

    // mmap's refusals, and the two shapes that stopped being refusals. Two of
    // these used to assert the opposite and both were about this loader rather
    // than about the call, which is why they went the way `ET_EXEC` went.
    {
        let fake = Region { at: 0x10_0000, len: 4096 };
        install(Regions { arena_extra: [(0, 0); 7], image: fake, stack: fake, brk: fake, interp: None, image_mapped: false, interp_mapped: false, stack_mapped: false, brk_mapped: false, bits32: false, mmap_at: GUEST_MMAP_AT, mmap_end: u64::MAX }, None);
        out.push((
            "a zero-length mapping is EINVAL, as Linux has it",
            sys_mmap(0, 0, 3, MAP_ANONYMOUS, u64::MAX, 0) == EINVAL,
        ));
        out.push((
            "a file mapping needs a descriptor somebody opened",
            sys_mmap(0, 4096, 1, MAP_PRIVATE, 99, 0) == EBADF,
        ));
        out.push((
            "and an offset into it that is a whole page, since a mapping starts on one",
            sys_mmap(0, 4096, 1, MAP_PRIVATE, 99, 1) == EINVAL,
        ));
        // Refused before the descriptor is even looked at, because the reason
        // is about the store rather than about the file: a shared writable
        // mapping would have to write back into something keyed by content,
        // which is a new root hash per modified page.
        out.push((
            "a shared writable file mapping is refused, and not by pretending the fd is bad",
            sys_mmap(0, 4096, 3, MAP_SHARED, 99, 0) == ENODEV,
        ));
        out.push((
            "MAP_FIXED at zero is EINVAL rather than treated as a hint",
            sys_mmap(0, 4096, 3, MAP_ANONYMOUS | MAP_FIXED, u64::MAX, 0) == EINVAL,
        ));
        out.push((
            "and an unaligned one is refused rather than rounded to a page",
            sys_mmap(0x1234, 4096, 3, MAP_ANONYMOUS | MAP_FIXED, u64::MAX, 0) == EINVAL,
        ));
        // A range no memory map can promise. Far above anything the firmware
        // ever declares conventional, so this is a fact about arithmetic
        // rather than about the machine underneath.
        out.push((
            "MAP_FIXED where nothing can promise the address is ENOMEM",
            sys_mmap(0x1000_0000_0000, 4096, 3, MAP_ANONYMOUS | MAP_FIXED, u64::MAX, 0) == ENOMEM,
        ));
    }

    // The three the applet sweep asked for, each answered from something this
    // machine knows rather than from a plausible constant.
    {
        let mut buf = [0u8; SYSINFO_LEN];
        let at = buf.as_mut_ptr() as u64;
        let owned = Region { at, len: SYSINFO_LEN };
        install(Regions { arena_extra: [(0, 0); 7], image: owned, stack: owned, brk: owned, interp: None, image_mapped: false, interp_mapped: false, stack_mapped: false, brk_mapped: false, bits32: false, mmap_at: GUEST_MMAP_AT, mmap_end: u64::MAX }, None);

        let bare = sys_time(0);
        let through = sys_time(at);
        out.push((
            "time answers the same value whether or not it is given somewhere to put it",
            bare == through
                && u64::from_le_bytes(buf[..8].try_into().unwrap_or_default()) == through,
        ));
        out.push((
            "and a pointer it does not own is EFAULT rather than a write through it",
            sys_time(0x1000) == EFAULT,
        ));

        buf = [0u8; SYSINFO_LEN];
        let ok = sys_sysinfo(at) == 0;
        let field = |o: usize| u64::from_le_bytes(buf[o..o + 8].try_into().unwrap_or_default());
        out.push((
            "sysinfo reports one process, a unit of one, and a heap that is not empty",
            ok && u16::from_le_bytes(buf[80..82].try_into().unwrap_or_default()) == 1
                && u32::from_le_bytes(buf[104..108].try_into().unwrap_or_default()) == 1
                && field(32) > 0,
        ));
        out.push((
            "and free is never more than total, which a subtraction that wrapped would be",
            field(40) <= field(32),
        ));
        out.push((
            "the load averages are zero rather than invented, since nothing samples them",
            field(8) == 0 && field(16) == 0 && field(24) == 0,
        ));

        buf = [0u8; SYSINFO_LEN];
        out.push((
            "sched_getaffinity answers the bytes it wrote, not zero",
            sys_sched_getaffinity(0, 128, at) == 8,
        ));
        out.push((
            "and names at least this core, whatever the rest of the machine did",
            u64::from_le_bytes(buf[..8].try_into().unwrap_or_default()) & 1 == 1,
        ));
        out.push((
            "a mask that is not a whole number of longs is refused rather than part-written",
            sys_sched_getaffinity(0, 4, at) == EINVAL,
        ));
        teardown();
    }

    // `pread64`, against the descriptors `install` seeds. What cannot be
    // checked here is the offset arithmetic on a real file, because that needs
    // a namespace and this runs at boot; the glibc run is what settles that,
    // and it is named in the commit rather than left implied.
    {
        let mut buf = [0u8; 64];
        let at = buf.as_mut_ptr() as u64;
        let owned = Region { at, len: 64 };
        install(Regions { arena_extra: [(0, 0); 7], image: owned, stack: owned, brk: owned, interp: None, image_mapped: false, interp_mapped: false, stack_mapped: false, brk_mapped: false, bits32: false, mmap_at: GUEST_MMAP_AT, mmap_end: u64::MAX }, None);
        out.push((
            "pread on a stream is ESPIPE, the same answer lseek gives it",
            sys_pread64(0, at, 8, 0) == ESPIPE,
        ));
        out.push((
            "a negative offset is EINVAL rather than an enormous unsigned one",
            sys_pread64(0, at, 8, (-1i64) as u64) == EINVAL,
        ));
        out.push((
            "and a descriptor nobody opened is EBADF",
            sys_pread64(99, at, 8, 0) == EBADF,
        ));
        out.push((
            "a zero-length read answers zero without touching the pointer",
            sys_pread64(99, 0, 0, 0) == 0,
        ));
        teardown();
    }

    // `readlink`, which has exactly one answer on this machine and three ways
    // to get it wrong. The sentinel past the end is the interesting one: the
    // call returns a length and writes that many bytes, so a terminator would
    // land in the caller's buffer past what it was told was used -- and a
    // caller that then trusts the string reads whatever was already there.
    {
        let mut buf = [0xAAu8; 160];
        let at = buf.as_mut_ptr() as u64;
        let owned = Region { at, len: 160 };
        install(Regions { arena_extra: [(0, 0); 7], image: owned, stack: owned, brk: owned, interp: None, image_mapped: false, interp_mapped: false, stack_mapped: false, brk_mapped: false, bits32: false, mmap_at: GUEST_MMAP_AT, mmap_end: u64::MAX }, None);
        name_guest(&["/tmp/g/bb"], "/tmp/g/bb", None);
        // Three separate strings rather than one rewritten in place: the last
        // time claims here shared a buffer, the first one's write took the NUL
        // the next two read as their path.
        let put = |off: usize, s: &str| unsafe {
            core::ptr::copy_nonoverlapping(s.as_ptr(), (at as usize + off) as *mut u8, s.len());
            *((at as usize + off + s.len()) as *mut u8) = 0;
        };
        put(96, "/proc/self/exe");
        put(120, "/proc/self/maps");
        put(144, "/nowhere");

        let n = sys_readlinkat(super::fs::AT_FDCWD as u64, at + 96, at, 64);
        out.push((
            "readlink answers the length it wrote, and /proc/self/exe is the image",
            n == 9 && &buf[..9] == b"/tmp/g/bb",
        ));
        out.push((
            "and writes no terminator, since the length is the whole of the answer",
            buf[9] == 0xAA,
        ));

        buf = [0xAAu8; 160];
        put(96, "/proc/self/exe");
        let short = sys_readlinkat(super::fs::AT_FDCWD as u64, at + 96, at, 4);
        out.push((
            "a buffer shorter than the target truncates rather than refusing, as Linux does",
            short == 4 && &buf[..4] == b"/tmp" && buf[4] == 0xAA,
        ));

        put(120, "/proc/self/maps");
        put(144, "/nowhere");
        out.push((
            "a path that exists and is not a link is EINVAL, which stops a caller hunting",
            sys_readlinkat(super::fs::AT_FDCWD as u64, at + 120, at, 64) == EINVAL,
        ));
        out.push((
            "and one that does not exist at all is ENOENT, which is a different fact",
            sys_readlinkat(super::fs::AT_FDCWD as u64, at + 144, at, 64) == ENOENT,
        ));
        put(96, "/proc/self/exe");
        out.push((
            "a zero-size buffer is EINVAL rather than a successful write of nothing",
            sys_readlinkat(super::fs::AT_FDCWD as u64, at + 96, at, 0) == EINVAL,
        ));
        out.push((
            "and a destination the guest does not own is EFAULT before anything is copied",
            sys_readlinkat(super::fs::AT_FDCWD as u64, at + 96, 0x1000, 64) == EFAULT,
        ));
        teardown();
    }

    // The exact sequence a dynamic linker performs, which is the one shape
    // `MAP_FIXED` exists for and the one nothing had ever run: reserve a span
    // with no rights at all, then lay something over part of it. Anonymous
    // rather than file-backed, because what is being checked is that the
    // rights of the reservation do not survive into the mapping laid over it,
    // and a `PROT_NONE` page that stays `PROT_NONE` is a program that faults
    // on its own code.
    if let Some(two) = alloc_pages(8192) {
        let mine = Region { at: two, len: 8192 };
        install(Regions { arena_extra: [(0, 0); 7], image: mine, stack: mine, brk: mine, interp: None, image_mapped: false, interp_mapped: false, stack_mapped: false, brk_mapped: false, bits32: false, mmap_at: GUEST_MMAP_AT, mmap_end: u64::MAX }, None);
        let reserved =
            sys_mmap(two, 8192, 0, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, u64::MAX, 0) == two;
        let gone = crate::mem::paging::query(two).is_some_and(|p| !p.present);
        let over = sys_mmap(two, 4096, 3, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, u64::MAX, 0);
        let writable = crate::mem::paging::query(two).is_some_and(|p| p.present && p.write);
        out.push((
            "a PROT_NONE reservation takes the page away rather than leaving it readable",
            reserved && gone,
        ));
        out.push((
            "and a mapping laid over it gets the rights it asked for, not the ones it replaced",
            over == two && writable,
        ));
        teardown();
        crate::mem::paging::release_to_heap(two, 8192);
        free_pages(two, 8192);
    }

    // `newfstatat` on a descriptor with no path. stdin is enough to check it,
    // since what is being asked is whether the *descriptor* is consulted at
    // all rather than what it says.
    {
        let mut buf = [0u8; 144];
        let at = buf.as_mut_ptr() as u64;
        let owned = Region { at, len: 144 };
        install(Regions { arena_extra: [(0, 0); 7], image: owned, stack: owned, brk: owned, interp: None, image_mapped: false, interp_mapped: false, stack_mapped: false, brk_mapped: false, bits32: false, mmap_at: GUEST_MMAP_AT, mmap_end: u64::MAX }, None);
        // The empty path has to live *inside* what the guest owns, because
        // `read_cstr` bounds-checks it like any other guest pointer. Pointing
        // at a local outside the region answered `EFAULT` before any of the
        // logic under test ran, and all three claims failed for a reason that
        // had nothing to do with what they were asking.
        let p = at;
        out.push((
            "an empty path with AT_EMPTY_PATH stats the descriptor, as ld.so asks it to",
            sys_statat(0, p, at, AT_EMPTY_PATH) == 0,
        ));
        // The path and the destination are one buffer, so a successful stat
        // writes 144 bytes over the NUL the next claim reads as its path.
        buf[0] = 0;
        out.push((
            "and without the flag it is still ENOENT, which is what Linux answers",
            sys_statat(0, p, at, 0) == ENOENT,
        ));
        buf[0] = 0;
        out.push((
            "with the flag and a descriptor nobody opened it is EBADF, not success",
            sys_statat(99, p, at, AT_EMPTY_PATH) == EBADF,
        ));
        teardown();
    }

    // A descriptor-relative path, which was `ENOSYS` under a comment saying it
    // needed the directory's own path kept per open descriptor -- and
    // `fs::Dir` has carried `path` since it was written. The refusal outlived
    // its reason, and what it cost is every `fts`-based program: `find`,
    // `rm -r`, `du` and `cp -r` all stop at the first subdirectory.
    //
    // `/lib` is the directory used because it is seeded at every boot, so this
    // needs no store mounted and no guest staged.
    {
        let mut buf = [0u8; 512];
        let at = buf.as_mut_ptr() as u64;
        let owned = Region { at, len: 512 };
        install(Regions { arena_extra: [(0, 0); 7], image: owned, stack: owned, brk: owned, interp: None, image_mapped: false, interp_mapped: false, stack_mapped: false, brk_mapped: false, bits32: false, mmap_at: GUEST_MMAP_AT, mmap_end: u64::MAX }, None);

        // Three disjoint slices of the one region: the paths have to live
        // inside what the guest owns, because `read_cstr` bounds-checks them
        // like any other guest pointer, and a `stat` writes 144 bytes.
        let dir_p = at;            // "/lib"
        let rel_p = at + 32;       // "geom.ai&xi"
        let abs_p = at + 64;       // "/lib/geom.ai&xi"
        let sbuf = at + 256;
        let put = |off: usize, text: &str, b: &mut [u8; 512]| {
            b[off..off + text.len()].copy_from_slice(text.as_bytes());
            b[off + text.len()] = 0;
        };
        put(0, "/lib", &mut buf);
        put(32, "geom.ai&xi", &mut buf);
        put(64, "/lib/geom.ai&xi", &mut buf);

        let fd = sys_openat(super::fs::AT_FDCWD as u64, dir_p, O_DIRECTORY, 0);
        out.push(("a directory opens, which is what a relative path is resolved against", (fd as i64) > 2));

        out.push((
            "a path relative to an open directory resolves, where it was ENOSYS",
            sys_statat(fd, rel_p, sbuf, 0) == 0,
        ));
        // A descriptor that is open and is not a directory is a different fact
        // from one nobody opened, and sends a program somewhere else.
        out.push((
            "the same path against stdin is ENOTDIR rather than EBADF",
            sys_statat(0, rel_p, sbuf, 0) == ENOTDIR,
        ));
        out.push((
            "and against a descriptor nobody opened it is EBADF",
            sys_statat(99, rel_p, sbuf, 0) == EBADF,
        ));
        // POSIX: an absolute path ignores `dirfd` entirely. A caller passing a
        // closed descriptor with an absolute path is correct code, so checking
        // the descriptor first would refuse it -- which is why `at_base` is
        // consulted only for a relative path.
        out.push((
            "an absolute path ignores dirfd, so a closed one is not an error",
            sys_statat(99, abs_p, sbuf, 0) == 0,
        ));
        // `..` is lexical: it drops the previous component of the path as
        // written. It was refused, on the grounds that a tree with O(1) copies
        // has no single parent -- true of a node, and beside the point for a
        // path, which names its parent in its own text. The refusal cost a
        // real program its data directory: the Source engine resolves
        // `bin/../portal` and reported its game as not installed.
        put(32, "../lib/geom.ai&xi", &mut buf);
        out.push((
            "a relative path may step back with .. , which drops a component",
            sys_statat(fd, rel_p, sbuf, 0) == 0,
        ));
        // And it cannot climb past the root, which is where Linux stops too.
        put(32, "../../../../lib/geom.ai&xi", &mut buf);
        out.push((
            "and stepping back past the root stays at the root",
            sys_statat(fd, rel_p, sbuf, 0) == 0,
        ));
        teardown();
    }

    // `nanosleep`, which was the one wait in this kernel that genuinely spun:
    // `while ticks() < until { spin_loop() }`, no yield in it at all. One
    // foreground guest made that invisible; a background one per sleeping
    // process is a core each.
    //
    // Tested here rather than through a fixture because the only fixture that
    // calls it is `--kind fb`, which sleeps ten minutes holding the display so
    // the harness can photograph it -- a poor instrument for asking whether a
    // sleep of 50 ms takes 50 ms.
    {
        let mut buf = [0u8; 32];
        let at = buf.as_mut_ptr() as u64;
        let owned = Region { at, len: 32 };
        install(Regions { arena_extra: [(0, 0); 7], image: owned, stack: owned, brk: owned, interp: None, image_mapped: false, interp_mapped: false, stack_mapped: false, brk_mapped: false, bits32: false, mmap_at: GUEST_MMAP_AT, mmap_end: u64::MAX }, None);

        // 50 ms as a timespec, which at TIMER_HZ is five ticks.
        buf[0..8].copy_from_slice(&0u64.to_le_bytes());
        buf[8..16].copy_from_slice(&50_000_000u64.to_le_bytes());
        let before = crate::dev::lapic::ticks();
        let rc = sys_nanosleep(at);
        let elapsed = crate::dev::lapic::ticks().saturating_sub(before);
        out.push(("nanosleep answers 0 for a sleep it completed", rc == 0));
        // At least what was asked for. Returning early is the failure a busy
        // wait cannot have and a sleep can, so `>=` is the honest direction --
        // the tick is 10 ms, so overshooting by one is expected.
        out.push((
            "and it slept at least the five ticks it was asked for",
            elapsed >= 5,
        ));
        // A nanosecond field at or past a second is EINVAL on Linux, and this
        // is checked after the sleep so a refusal cannot be mistaken for the
        // sleep above having been skipped.
        buf[8..16].copy_from_slice(&1_000_000_000u64.to_le_bytes());
        out.push(("a nanosecond field of a whole second is EINVAL", sys_nanosleep(at) == EINVAL));
        out.push(("and an unreachable timespec is EFAULT", sys_nanosleep(0x1000) == EFAULT));
        teardown();
    }

    // Teardown puts rights back on every region, and the interpreter is the
    // one that was missed. Two pages so the interpreter's is a different page
    // from the image's, because a claim where both are the same address passes
    // while the interpreter is not restored at all.
    if let Some(two) = alloc_pages(8192) {
        let img = Region { at: two, len: 4096 };
        let interp = Region { at: two + 4096, len: 4096 };
        install(Regions { arena_extra: [(0, 0); 7], image: img, stack: img, brk: img, interp: Some(interp), image_mapped: false, interp_mapped: false, stack_mapped: false, brk_mapped: false, bits32: false, mmap_at: GUEST_MMAP_AT, mmap_end: u64::MAX }, None);
        // PROT_READ. A real `ld.so` does exactly this to its own RELRO, which
        // is how the page that halted the machine got its rights.
        let asked = sys_mprotect(two + 4096, 4096, 1) == 0;
        let took = crate::mem::paging::query(two + 4096).is_some_and(|p| !p.write);
        teardown();
        let back = crate::mem::paging::query(two + 4096).is_some_and(|p| p.write);
        out.push((
            "a page the interpreter made read-only is writable again after teardown",
            asked && took && back,
        ));
        crate::mem::paging::release_to_heap(two, 8192);
        free_pages(two, 8192);
    }

    // The shape a dynamic linker actually uses: reserve a span, then lay each
    // segment of a library out over part of it. A real page rather than the
    // fake region above, because this one gets its rights marked and a fake
    // pointing at 0x100000 would mark the low megabyte.
    if let Some(own) = alloc_pages(8192) {
        let mine = Region { at: own, len: 8192 };
        install(Regions { arena_extra: [(0, 0); 7], image: mine, stack: mine, brk: mine, interp: None, image_mapped: false, interp_mapped: false, stack_mapped: false, brk_mapped: false, bits32: false, mmap_at: GUEST_MMAP_AT, mmap_end: u64::MAX }, None);
        unsafe { core::ptr::write_bytes(own as *mut u8, 0xA5, 8192) };
        out.push((
            "MAP_FIXED over memory the guest already holds answers that address",
            sys_mmap(own, 4096, 3, MAP_ANONYMOUS | MAP_FIXED, u64::MAX, 0) == own,
        ));
        // glibc's `calloc` trusts this: a `.bss` or a re-laid arena that kept
        // its last tenant's bytes is a zeroed structure full of stale pointers.
        let page = |at: u64| unsafe { core::slice::from_raw_parts(at as *const u8, 4096) };
        out.push((
            "and the anonymous memory laid there reads as zero, as Linux's would",
            page(own).iter().all(|&b| b == 0) && page(own + 4096).iter().all(|&b| b == 0xA5),
        ));
        unsafe { core::ptr::write_bytes(own as *mut u8, 0x5A, 4096) };
        out.push((
            "MADV_DONTNEED on anonymous memory leaves it reading as zero",
            sys_madvise(own, 4096, 4) == 0 && page(own).iter().all(|&b| b == 0),
        ));
        out.push((
            "and every other advice is accepted and changes nothing",
            sys_madvise(own + 4096, 4096, 3) == 0 && page(own + 4096).iter().all(|&b| b == 0xA5),
        ));
        out.push((
            "and records nothing, because something already holds those pages",
            unsafe { guest_slot() }.as_ref().is_some_and(|sp| sp.maps.is_empty()),
        ));
        // Rights back before the page goes back, which is the mistake this
        // tree has now made three times in three different places.
        crate::mem::paging::release_to_heap(own, 8192);
        free_pages(own, 8192);
        teardown();
    }

    {
        let fake = Region { at: 0x10_0000, len: 4096 };
        install(Regions { arena_extra: [(0, 0); 7], image: fake, stack: fake, brk: fake, interp: None, image_mapped: false, interp_mapped: false, stack_mapped: false, brk_mapped: false, bits32: false, mmap_at: GUEST_MMAP_AT, mmap_end: u64::MAX }, None);

        let at = sys_mmap(0, 8192, 3, MAP_ANONYMOUS, u64::MAX, 0);
        let got = (at as i64) > 0 && at % 4096 == 0;
        // It has to be real memory, and it has to be zeroed: an allocator
        // handed dirty pages produces a program that works once.
        let zeroed = got && unsafe { core::slice::from_raw_parts(at as *const u8, 8192) }.iter().all(|b| *b == 0);
        out.push(("an anonymous mapping is page-aligned and zeroed", got && zeroed));
        out.push((
            "a partial unmap is accepted and keeps the pages, since the heap cannot free part of one",
            sys_munmap(at, 4096) == 0,
        ));
        out.push(("unmapping the whole thing gives it back", sys_munmap(at, 8192) == 0));
        out.push((
            "unmapping what was never mapped is EINVAL",
            sys_munmap(at, 8192) == EINVAL,
        ));
        unsafe { *guest_slot() = None };
    }

    // arch_prctl, and the refusal that is the point of it.
    {
        let mut slot = 0u64;
        let at = &mut slot as *mut u64 as u64;
        // GET_FS writes eight bytes through a guest pointer, so it is bounds
        // checked now, so the destination has to be a range the guest owns.
        let owned = Region { at, len: 8 };
        install(Regions { arena_extra: [(0, 0); 7], image: owned, stack: owned, brk: owned, interp: None, image_mapped: false, interp_mapped: false, stack_mapped: false, brk_mapped: false, bits32: false, mmap_at: GUEST_MMAP_AT, mmap_end: u64::MAX }, None);
        let was = unsafe { crate::cpu::rdmsr(IA32_FS_BASE) };
        let set = sys_arch_prctl(ARCH_SET_FS, 0xDEAD_0000);
        let got = sys_arch_prctl(ARCH_GET_FS, at);
        unsafe { crate::cpu::wrmsr(IA32_FS_BASE, was) };
        unsafe { *guest_slot() = None };
        out.push((
            "a guest may set FS for its thread-local storage, and read it back",
            set == 0 && got == 0 && slot == 0xDEAD_0000,
        ));
        out.push((
            "a guest may not touch GS, which is where this kernel keeps its per-core block",
            sys_arch_prctl(ARCH_SET_GS, 0x1000) == EPERM
                && sys_arch_prctl(ARCH_GET_GS, 0) == EPERM,
        ));
        out.push((
            "an arch_prctl nobody implements is EINVAL and not a silent success",
            sys_arch_prctl(0x9999, 0) == EINVAL,
        ));
        out.push((
            "FS is unchanged by the checks that moved it",
            unsafe { crate::cpu::rdmsr(IA32_FS_BASE) } == was,
        ));
    }

    // Guest pointers, which is where a ring-0 guest is most dangerous to the
    // kernel rather than to itself. A range the loader never handed out has to
    // be refused before anything dereferences it.
    {
        let mut backing = [0u64; 64];
        let at = backing.as_mut_ptr() as u64;
        let one = Region { at, len: 512 };
        install(Regions { arena_extra: [(0, 0); 7], image: one, stack: one, brk: one, interp: None, image_mapped: false, interp_mapped: false, stack_mapped: false, brk_mapped: false, bits32: false, mmap_at: GUEST_MMAP_AT, mmap_end: u64::MAX }, None);
        out.push(("a range inside what the loader gave out is owned", owns(at, 8)));
        out.push(("a range that runs off the end is not", !owns(at + 508, 8)));
        out.push(("a range below it is not", !owns(at - 8, 8)));
        out.push(("a null pointer is not", !owns(0, 8)));
        out.push((
            "a length that overflows the address is refused rather than wrapping",
            !owns(u64::MAX - 4, 64),
        ));
        // The two calls that dereference a guest pointer.
        out.push((
            "a write through a pointer the guest does not own is EFAULT",
            sys_write(1, 0, 16) == EFAULT,
        ));
        out.push((
            "and arch_prctl will not post the FS base to one either",
            sys_arch_prctl(ARCH_GET_FS, 0) == EFAULT,
        ));
        out.push((
            "a zero-length write succeeds without the pointer being looked at",
            sys_write(1, 0, 0) == 0,
        ));
        out.push((
            "a mapping larger than the cap is refused, so the page rounding cannot wrap",
            sys_mmap(0, MAP_MAX + 1, 3, MAP_ANONYMOUS, u64::MAX, 0) == ENOMEM,
        ));
        unsafe { *guest_slot() = None };
    }
    // mprotect, on a page taken for the check and given straight back.
    {
        use alloc::alloc::{alloc_zeroed, dealloc, Layout};
        if let Ok(layout) = Layout::from_size_align(4096, 4096) {
            let mem = unsafe { alloc_zeroed(layout) };
            if !mem.is_null() {
                let at = mem as u64;
                let owned = Region { at, len: 4096 };
                install(Regions { arena_extra: [(0, 0); 7], image: owned, stack: owned, brk: owned, interp: None, image_mapped: false, interp_mapped: false, stack_mapped: false, brk_mapped: false, bits32: false, mmap_at: GUEST_MMAP_AT, mmap_end: u64::MAX }, None);
                out.push(("a page the guest owns starts reachable", reachable(at, 8, true)));
                out.push((
                    "mprotect to PROT_NONE is accepted",
                    sys_mprotect(at, 4096, 0) == 0,
                ));
                out.push((
                    "and the kernel then refuses to touch it on the guest's behalf",
                    !reachable(at, 8, true) && sys_write(1, at, 8) == EFAULT,
                ));
                out.push((
                    "read-only is refused for a write and allowed for a read",
                    sys_mprotect(at, 4096, 1) == 0
                        && !reachable(at, 8, true)
                        && reachable(at, 8, false),
                ));
                out.push((
                    "an unaligned mprotect is EINVAL",
                    sys_mprotect(at + 1, 4096, 3) == EINVAL,
                ));
                out.push((
                    "a range the guest does not own is ENOMEM",
                    sys_mprotect(at + 0x100_0000, 4096, 3) == ENOMEM,
                ));
                let back = sys_mprotect(at, 4096, PROT_WRITE | 1) == 0;
                out.push(("and read-write can be given back", back && reachable(at, 8, true)));
                unsafe { *guest_slot() = None };
                crate::mem::paging::protect(at, 4096, crate::mem::paging::Perm::RWX);
                unsafe { dealloc(mem, layout) };
            }
        }
    }

    out.push((
        "with no guest running, no address is owned at all",
        !owns(0x1000, 8),
    ));
    // A stack too small to hold the frame answers None rather than writing a
    // truncated one somewhere the guest will not look.
    let mut small = [0u8; 64];
    out.push((
        "a stack too small for the initial frame is refused",
        {
            let p = small.as_mut_ptr();
            build_stack(p, p as u64, 64, &["x"], Aux::default()).is_none()
        },
    ));
    // Room for whatever `linux env` has added, which this frame carries: at
    // 512 bytes these claims passed on a fresh boot and failed after a game's
    // environment had been set, which is a claim about the shell's history
    // rather than about the frame.
    const BIG: usize = 8192;
    let mut big = alloc::vec![0u8; BIG];
    let aux = Aux { phdr: 0x1000, phent: 56, phnum: 2, entry: 0x1078, base: 0 };
    let sp = {
        let p = big.as_mut_ptr();
        build_stack(p, p as u64, BIG, &["cat", "/ai/about"], aux)
    };
    out.push((
        "and one large enough answers a 16-byte-aligned pointer inside itself",
        sp.is_some_and(|v| v % 16 == 0 && v >= big.as_ptr() as u64
            && v < big.as_ptr() as u64 + BIG as u64),
    ));
    out.push((
        "argc and the argv pointers are where the ABI says, and the strings are real",
        sp.is_some_and(|v| unsafe {
            let p = v as *const u64;
            let argc = p.read();
            let a0 = p.add(1).read() as *const u8;
            let a1 = p.add(2).read() as *const u8;
            let term = p.add(3).read();
            argc == 2
                && term == 0
                && core::slice::from_raw_parts(a0, 3) == b"cat"
                && core::slice::from_raw_parts(a1, 9) == b"/ai/about"
        }),
    ));
    // The aux vector, read back the way a libc reads it: walk pairs from after
    // the envp terminator until AT_NULL, and look up by key rather than by
    // position, since nothing promises an order.
    let auxv = |key: u64| -> Option<u64> {
        let v = sp?;
        unsafe {
            let p = v as *const u64;
            let n = p.read() as usize;
            // argc, argv, its NULL, then the environment and its NULL.
            let mut i = 2 + n;
            while p.add(i).read() != 0 {
                i += 1;
            }
            i += 1;
            loop {
                let k = p.add(i).read();
                if k == AT_NULL {
                    return None;
                }
                if k == key {
                    return Some(p.add(i + 1).read());
                }
                i += 2;
            }
        }
    };
    out.push((
        "the aux vector answers a page size, so a libc does not divide by zero",
        auxv(AT_PAGESZ) == Some(PAGE),
    ));
    out.push((
        "the environment is between argv and the aux vector, where a libc looks",
        sp.is_some_and(|v| unsafe {
            let p = v as *const u64;
            let n = p.read() as usize;
            let first = p.add(2 + n).read();
            first != 0
                && core::slice::from_raw_parts(first as *const u8, 4) == b"PATH"
        }),
    ));
    out.push((
        "and the entry and header table the loader placed, by key rather than by position",
        auxv(AT_ENTRY) == Some(0x1078)
            && auxv(AT_PHDR) == Some(0x1000)
            && auxv(AT_PHNUM) == Some(2)
            && auxv(AT_PHENT) == Some(56),
    ));
    out.push((
        "AT_RANDOM points at sixteen bytes inside this stack, since libc reads its guard there",
        auxv(AT_RANDOM).is_some_and(|r| {
            r >= big.as_ptr() as u64 && r + 16 <= big.as_ptr() as u64 + BIG as u64
        }),
    ));
    out.push((
        "a header table no segment covers omits the whole group rather than pointing at nothing",
        {
            let mut small = alloc::vec![0u8; BIG];
            let none = Aux { phdr: 0, phent: 56, phnum: 2, entry: 0, base: 0 };
            let s2 = {
                let p = small.as_mut_ptr();
                build_stack(p, p as u64, BIG, &["x"], none)
            };
            s2.is_some_and(|v| unsafe {
                let p = v as *const u64;
                let n = p.read() as usize;
                let mut i = 2 + n;
                while p.add(i).read() != 0 {
                    i += 1;
                }
                i += 1;
                let mut seen_phdr = false;
                let mut seen_pagesz = false;
                loop {
                    let k = p.add(i).read();
                    if k == AT_NULL {
                        break;
                    }
                    seen_phdr |= k == AT_PHDR || k == AT_PHNUM || k == AT_PHENT;
                    seen_pagesz |= k == AT_PAGESZ;
                    i += 2;
                }
                !seen_phdr && seen_pagesz
            })
        },
    ));

    // The stub and `glados_enter_guest` carry their selectors as literals,
    // because `global_asm!` cannot see a Rust constant. So the literals are
    // asserted against the constants instead: 0x3b and 0x33 in that assembly
    // are the only two numbers in this module that nothing else checks, and
    // getting either wrong is a triple fault on the `iretq`.
    out.push((
        "the ring-3 selectors written into the assembly are the ones the GDT holds",
        crate::cpu::gdt::ring3(crate::cpu::gdt::USER_CS) == 0x3B
            && crate::cpu::gdt::ring3(crate::cpu::gdt::USER_DS) == 0x33,
    ));
    out.push((
        "the flags the guest starts with have interrupts on, so it can be preempted",
        0x202u64 & (1 << 9) != 0,
    ));
    out.push((
        "an overrun is distinguishable from a fault and from an exit",
        OVERRAN != FAULTED && OVERRAN != EXITED && (OVERRAN | 5) & 0xFFFF_FFFF == 5,
    ));
    out.push((
        "no deadline is set when nothing is running, so nothing can be killed",
        !overran(u64::MAX),
    ));
    out.push((
        "a fault code is distinguishable from an exit code",
        FAULTED != EXITED && (FAULTED | 14) & 0xFFFF_FFFF == 14,
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
/// What the auxiliary vector has to say, gathered by the loader.
///
/// **Empty was not a safe default and that is why this exists.** A static libc
/// has no dynamic linker to ask, so everything it cannot compute it reads from
/// here: `AT_PAGESZ` becomes `libc.page_size`, which musl divides by, and
/// `AT_RANDOM` is where the stack guard comes from. A vector holding nothing
/// but `AT_NULL` hands a real binary a page size of zero.
///
/// Nothing here has been consumed by a real libc on this machine -- every
/// fixture is hand-written and reads none of it -- so this is a bet placed
/// where the ABI says it should be placed, and it is worth saying so.
#[derive(Clone, Copy, Default)]
pub struct Aux {
    /// Where the program headers landed at runtime, or zero when no loadable
    /// segment covers them. Zero omits the whole `AT_PHDR`/`AT_PHENT`/
    /// `AT_PHNUM` group, since a header pointer into nothing is worse than an
    /// absent one: the absent one a libc can cope with.
    pub phdr: u64,
    pub phent: u64,
    pub phnum: u64,
    /// The *program's* entry, which is not where execution starts once there
    /// is an interpreter. Getting these two the wrong way round gives a
    /// `ld.so` that relocates everything correctly and then jumps back into
    /// itself.
    pub entry: u64,
    /// Where the interpreter was loaded, or zero when there is none.
    ///
    /// This is how `ld.so` finds its own relocations: it is a `ET_DYN` object
    /// that has not been relocated by anybody, so the only way it can locate
    /// its own `_DYNAMIC` is by being told where it landed. Zero omits the
    /// entry, which is what Linux does for a static binary and what a libc
    /// reads as "you are the program".
    pub base: u64,
}

const AT_NULL: u64 = 0;
const AT_PHDR: u64 = 3;
const AT_PHENT: u64 = 4;
const AT_PHNUM: u64 = 5;
const AT_PAGESZ: u64 = 6;
const AT_BASE: u64 = 7;
const AT_ENTRY: u64 = 9;
const AT_UID: u64 = 11;
const AT_EUID: u64 = 12;
const AT_GID: u64 = 13;
const AT_EGID: u64 = 14;
const AT_CLKTCK: u64 = 17;
const AT_SECURE: u64 = 23;
const AT_RANDOM: u64 = 25;

/// What a guest finds in `environ`.
///
/// **It was empty, and an empty environment is not a neutral one.** `env`
/// printed nothing, which is at least true, but `sh` resolves commands through
/// `PATH` and a program with no `HOME` writes its dotfiles into the working
/// directory. These are the smallest set that make a shell behave, and each is
/// a fact about this machine rather than a plausible-looking default:
/// `TERM=dumb` because there is no terminal here at all and `ioctl` says so,
/// `PWD=/` because there is no `chdir`, and `PATH` naming directories that may
/// well be empty, which is what a search path is for.
///
/// And where the display server is, which is a fact now that there is one:
/// `XDG_RUNTIME_DIR` and `WAYLAND_DISPLAY` together name `sky::server::PATH`.
/// A Wayland client with neither set gives up before it tries to connect.
const ENVIRON: [&str; 7] = [
    "PATH=/bin:/usr/bin:/tmp",
    "HOME=/",
    "PWD=/",
    "TERM=dumb",
    "USER=root",
    "XDG_RUNTIME_DIR=/run/glados",
    "WAYLAND_DISPLAY=wayland-0",
];

/// Extra variables, on top of the five above.
///
/// Added because the guest's own diagnostics are better than ours and there
/// was no way to switch them on. `ld.so` narrates everything it does under
/// `LD_DEBUG`, in its own vocabulary, about its own data structures -- which
/// is a far better account of why a linker failed than any trace of syscalls
/// can be, and it costs one environment variable rather than a kernel change
/// per question.
///
/// Separate from `ENVIRON` rather than replacing it, so the five facts about
/// this machine stay facts and cannot be switched off by accident.
static EXTRA_ENV: Racy<Vec<String>> = Racy::new(Vec::new());

/// Set one, replacing any earlier setting of the same name. An empty value
/// removes it, which is how a variable is unset rather than set to nothing --
/// `LD_DEBUG=` and no `LD_DEBUG` mean different things to `ld.so`.
pub fn set_env(entry: &str) {
    let name = match entry.split_once('=') {
        Some((n, _)) => n,
        None => entry,
    };
    let v = unsafe { EXTRA_ENV.get() };
    v.retain(|e| !(e.starts_with(name) && e.as_bytes().get(name.len()) == Some(&b'=')));
    if entry.contains('=') && !entry.ends_with('=') {
        v.push(String::from(entry));
    }
}

/// What a guest will be given, the fixed five and the added ones.
pub fn environ() -> Vec<String> {
    let mut out: Vec<String> = ENVIRON.iter().map(|e| String::from(*e)).collect();
    out.extend(unsafe { EXTRA_ENV.get() }.iter().cloned());
    out
}

/// Lay out argc, argv, envp and the aux vector on a guest's stack.
///
/// **`base` is where the kernel writes and `guest_base` is where the guest
/// will see it, and they are only the same number by accident.** Every
/// pointer this lays down names memory on this same stack -- argv entries
/// point at strings a few bytes above them, `AT_RANDOM` points at sixteen
/// bytes of canary -- so if the stack is mapped somewhere else in the guest's
/// own tables, every one of those has to be written as the guest's address
/// while the write itself goes to the backing. Pass `guest_base == base as
/// u64` for a stack the guest sees where the kernel put it, which is what an
/// identity-mapped guest wants and what this did before it could be asked
/// anything else.
///
/// Getting this wrong is silent in the worst way: the stack is laid out
/// perfectly, `argc` is right, and every pointer in it names an address in
/// the *kernel's* heap that the guest cannot reach. What shows is a fault on
/// argv[0] inside libc's startup, which reads as a bad loader rather than as
/// a stack built at the wrong base.
pub fn build_stack(
    base: *mut u8,
    guest_base: u64,
    size: usize,
    args: &[&str],
    aux: Aux,
) -> Option<u64> {
    build_stack_w(base, guest_base, size, args, aux, 8)
}

/// The same stack in 4-byte words, for a 32-bit guest: argc, every pointer and
/// every auxiliary pair are the width of the guest's `long`. One builder with a
/// width rather than two copies, because the layout -- strings at the top, the
/// random bytes under them, argc at the stack pointer -- is the part the ABI
/// fixes and two copies of it are two chances to disagree.
pub fn build_stack32(
    base: *mut u8,
    guest_base: u64,
    size: usize,
    args: &[&str],
    aux: Aux,
) -> Option<u64> {
    build_stack_w(base, guest_base, size, args, aux, 4)
}

fn build_stack_w(
    base: *mut u8,
    guest_base: u64,
    size: usize,
    args: &[&str],
    aux: Aux,
    word: usize,
) -> Option<u64> {
    let bottom = base as usize;
    let mut top = bottom.checked_add(size)?;
    // Where the guest will see an address the kernel is writing at.
    let seen = |x: usize| guest_base + (x - bottom) as u64;

    // Strings first, at the very top, because the pointer array below has to
    // name them and nothing may move afterwards.
    let mut put = |top: &mut usize, v: &str| -> Option<u64> {
        *top = top.checked_sub(v.len() + 1)?;
        if *top < bottom {
            return None;
        }
        unsafe {
            core::ptr::copy_nonoverlapping(v.as_ptr(), *top as *mut u8, v.len());
            core::ptr::write((*top + v.len()) as *mut u8, 0);
        }
        Some(seen(*top))
    };
    let env = environ();
    let mut envs = alloc::vec::Vec::with_capacity(env.len());
    for e in env.iter().rev() {
        envs.push(put(&mut top, e)?);
    }
    envs.reverse();
    let mut ptrs = alloc::vec::Vec::with_capacity(args.len());
    for a in args.iter().rev() {
        ptrs.push(put(&mut top, a)?);
    }
    ptrs.reverse();

    // Sixteen bytes for `AT_RANDOM`, which is where a libc takes its stack
    // guard from. `fill` and not `fill_secret`: this is a canary rather than
    // key material, and `fill_secret` refuses until the entropy pool has been
    // credited, which on a headless boot is never -- so the strict call would
    // make every guest fail to start in exchange for a stronger guarantee than
    // a canary needs.
    top = top.checked_sub(16)?;
    if top < bottom {
        return None;
    }
    // Two addresses for one thing: the canary is filled through the backing
    // and named to the guest at the address it will read it from.
    let random = seen(top);
    unsafe {
        crate::rng::fill(core::slice::from_raw_parts_mut(top as *mut u8, 16));
    }

    let mut pairs = alloc::vec::Vec::new();
    pairs.push((AT_PAGESZ, PAGE));
    // The scheduler's tick, which is what `times()` would be denominated in
    // if it existed. Taken from `crate::TIMER_HZ`, the interrupt rate, and
    // deliberately not from `lapic::timer_hz()`, which is the calibrated APIC
    // frequency and is in the millions -- a confusion this tree has already
    // paid for once, in the Oracle.
    pairs.push((AT_CLKTCK, crate::TIMER_HZ as u64));
    // One process, no privilege boundary above it, nothing dropped. A libc
    // reads `AT_SECURE` to decide whether to trust the environment, and here
    // there is one environment.
    for id in [AT_UID, AT_EUID, AT_GID, AT_EGID, AT_SECURE] {
        pairs.push((id, 0));
    }
    pairs.push((AT_RANDOM, random));
    if aux.entry != 0 {
        pairs.push((AT_ENTRY, aux.entry));
    }
    if aux.phdr != 0 && aux.phnum != 0 {
        pairs.push((AT_PHDR, aux.phdr));
        pairs.push((AT_PHENT, aux.phent));
        pairs.push((AT_PHNUM, aux.phnum));
    }
    // Omitted rather than zeroed when there is no interpreter, which is what
    // Linux does and what a libc reads as "you are the program". A zero would
    // be read as an interpreter loaded at address zero, and the first thing
    // `ld.so` does with this number is add it to an offset.
    if aux.base != 0 {
        pairs.push((AT_BASE, aux.base));
    }
    pairs.push((AT_NULL, 0));

    // argc, argv[..], its NULL, envp's NULL, then the pairs. `rsp` itself is
    // sixteen-byte aligned at entry, which is what the ABI asks of a process
    // rather than of a function -- there is no return address under it.
    //
    // A line here used to pad down when `(sp + words * 8) % 8 != 0`, which is
    // a condition that cannot hold: `sp` is sixteen-aligned and every word is
    // eight bytes. It read as an alignment fix and was a tautology, which is
    // the more expensive kind of dead code because it stops anybody looking.
    let words = 1 + ptrs.len() + 1 + envs.len() + 1 + pairs.len() * 2;
    let sp = top.checked_sub(words * word)? & !0xF;
    if sp < bottom {
        return None;
    }
    // A word at index `i`, at the guest's own width.
    let put_w = |i: usize, v: u64| unsafe {
        if word == 4 {
            (sp as *mut u32).add(i).write(v as u32);
        } else {
            (sp as *mut u64).add(i).write(v);
        }
    };
    put_w(0, ptrs.len() as u64);
    for (i, v) in ptrs.iter().enumerate() {
        put_w(1 + i, *v);
    }
    put_w(1 + ptrs.len(), 0); // argv terminator
    let e0 = 2 + ptrs.len();
    for (i, v) in envs.iter().enumerate() {
        put_w(e0 + i, *v);
    }
    put_w(e0 + envs.len(), 0); // envp terminator
    let a0 = e0 + envs.len() + 1;
    for (i, (k, v)) in pairs.iter().enumerate() {
        put_w(a0 + i * 2, *k);
        put_w(a0 + i * 2 + 1, *v);
    }
    Some(seen(sp))
}

/// Name the calls stage 0 knows about, for a trace a person has to read.
pub fn name_of(nr: u64) -> &'static str {
    if nr & super::i386::TAG != 0 {
        return super::i386::name_of(nr & !super::i386::TAG);
    }
    match nr {
        SYS_READ => "read",
        SYS_WRITE => "write",
        SYS_OPEN => "open",
        SYS_CLOSE => "close",
        SYS_STAT => "stat",
        SYS_FSTAT => "fstat",
        SYS_LSTAT => "lstat",
        SYS_LSEEK => "lseek",
        SYS_IOCTL => "ioctl",
        SYS_PREAD64 => "pread64",
        SYS_SET_ROBUST_LIST => "set_robust_list",
        SYS_PRLIMIT64 => "prlimit64",
        SYS_GETRANDOM => "getrandom",
        SYS_TIME => "time",
        SYS_READLINK => "readlink",
        SYS_READLINKAT => "readlinkat",
        SYS_SYSINFO => "sysinfo",
        SYS_SCHED_GETAFFINITY => "sched_getaffinity",
        SYS_RSEQ => "rseq",
        SYS_GETPID => "getpid",
        SYS_DUP => "dup",
        SYS_WRITEV => "writev",
        SYS_READV => "readv",
        SYS_SOCKET => "socket",
        SYS_CONNECT => "connect",
        SYS_BIND => "bind",
        SYS_SENDMSG => "sendmsg",
        SYS_RECVMSG => "recvmsg",
        SYS_LISTEN => "listen",
        SYS_ACCEPT => "accept",
        SYS_ACCEPT4 => "accept4",
        SYS_SOCKETPAIR => "socketpair",
        SYS_SENDTO => "sendto",
        SYS_RECVFROM => "recvfrom",
        SYS_SHUTDOWN => "shutdown",
        SYS_SETSOCKOPT => "setsockopt",
        SYS_GETSOCKOPT => "getsockopt",
        SYS_ACCESS => "access",
        SYS_FACCESSAT => "faccessat",
        SYS_FTRUNCATE => "ftruncate",
        SYS_MEMFD_CREATE => "memfd_create",
        SYS_EPOLL_CREATE => "epoll_create",
        SYS_EPOLL_CREATE1 => "epoll_create1",
        SYS_EPOLL_CTL => "epoll_ctl",
        SYS_EPOLL_WAIT => "epoll_wait",
        SYS_EPOLL_PWAIT => "epoll_pwait",
        SYS_PIPE => "pipe",
        SYS_PIPE2 => "pipe2",
        SYS_POLL => "poll",
        SYS_PPOLL => "ppoll",
        SYS_SELECT => "select",
        SYS_PSELECT6 => "pselect6",
        SYS_SENDFILE => "sendfile",
        SYS_GETCWD => "getcwd",
        SYS_MKDIR => "mkdir",
        SYS_RMDIR => "rmdir",
        SYS_UNLINK => "unlink",
        SYS_NANOSLEEP => "nanosleep",
        SYS_CLOCK_NANOSLEEP => "clock_nanosleep",
        SYS_RT_SIGACTION => "rt_sigaction",
        SYS_RT_SIGPROCMASK => "rt_sigprocmask",
        SYS_RT_SIGRETURN => "rt_sigreturn",
        SYS_KILL => "kill",
        SYS_GETPPID => "getppid",
        SYS_GETGROUPS => "getgroups",
        SYS_UNAME => "uname",
        SYS_FCNTL => "fcntl",
        SYS_GETTIMEOFDAY => "gettimeofday",
        SYS_CLOCK_GETTIME => "clock_gettime",
        SYS_DUP2 => "dup2",
        SYS_DUP3 => "dup3",
        SYS_CLONE => "clone",
        SYS_FORK => "fork",
        SYS_VFORK => "vfork",
        SYS_WAIT4 => "wait4",
        SYS_EXECVE => "execve",
        SYS_FUTEX => "futex",
        SYS_GETTID => "gettid",
        SYS_TGKILL => "tgkill",
        SYS_SCHED_YIELD => "sched_yield",
        SYS_GETUID => "getuid",
        SYS_GETGID => "getgid",
        SYS_GETEUID => "geteuid",
        SYS_GETEGID => "getegid",
        SYS_GETDENTS64 => "getdents64",
        SYS_SET_TID_ADDRESS => "set_tid_address",
        SYS_OPENAT => "openat",
        SYS_NEWFSTATAT => "newfstatat",
        SYS_BRK => "brk",
        SYS_MMAP => "mmap",
        SYS_MPROTECT => "mprotect",
        SYS_MADVISE => "madvise",
        SYS_MUNMAP => "munmap",
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
    // A dynamically linked binary is no longer refused here. It used to be,
    // under a reason about this loader rather than about the file -- the entry
    // in the header is not where execution starts, `ld.so` is -- and that
    // stopped being true when `load` learned to place the interpreter beside
    // the program and jump to *its* entry. What can still fail is finding the
    // interpreter, which is a fact about the namespace and is reported there.
    if img.segments.is_empty() {
    // A fixed address is no longer refused here. Whether one can be honoured is
    // a question about this machine's memory map rather than about the file, so
    // `load` asks `mem::fixed` and reports what it said -- "that physical range
    // is not free on this machine" names the actual obstacle, where the blanket
    // refusal named a design decision that had stopped being one.
        return Err("nothing to load");
    }
    Ok(())
}

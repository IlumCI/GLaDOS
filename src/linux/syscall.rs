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
pub const SYS_SENDTO: u64 = 44;
pub const SYS_RECVFROM: u64 = 45;
pub const SYS_SHUTDOWN: u64 = 48;
pub const SYS_GETSOCKNAME: u64 = 51;
pub const SYS_GETPEERNAME: u64 = 52;
pub const SYS_SETSOCKOPT: u64 = 54;
pub const SYS_GETSOCKOPT: u64 = 55;
pub const SYS_ACCESS: u64 = 21;
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
pub const SYS_MUNMAP: u64 = 11;
pub const SYS_ARCH_PRCTL: u64 = 158;
pub const SYS_EXIT_GROUP: u64 = 231;

/// `-ENOSYS`, as Linux returns it: errors come back as small negatives in
/// `rax` rather than through any side channel.
const ENOSYS: u64 = (-38i64) as u64;
const EBADF: u64 = (-9i64) as u64;
const EPERM: u64 = (-1i64) as u64;
const ENOMEM: u64 = (-12i64) as u64;
const EINVAL: u64 = (-22i64) as u64;
const EFAULT: u64 = (-14i64) as u64;
const ENOENT: u64 = (-2i64) as u64;
const EACCES: u64 = (-13i64) as u64;
const EMFILE: u64 = (-24i64) as u64;
const ENOTDIR: u64 = (-20i64) as u64;
const EISDIR: u64 = (-21i64) as u64;
const ENOTTY: u64 = (-25i64) as u64;
const ESPIPE: u64 = (-29i64) as u64;
const ENOTCONN: u64 = (-107i64) as u64;
const EAGAIN: u64 = (-11i64) as u64;
const EINTR: u64 = (-4i64) as u64;
const ENAMETOOLONG: u64 = (-36i64) as u64;
const EROFS: u64 = (-30i64) as u64;
const EEXIST: u64 = (-17i64) as u64;
const ERANGE: u64 = (-34i64) as u64;
const ENOTEMPTY: u64 = (-39i64) as u64;
const ENODEV: u64 = (-19i64) as u64;
const ENOSPC: u64 = (-28i64) as u64;
/// `O_NONBLOCK`, which on an event device is the difference between a program
/// that polls and one that waits.
const O_NONBLOCK: u64 = 0o4000;

/// `AT_EMPTY_PATH`: operate on the descriptor rather than on a path under it.
const AT_EMPTY_PATH: u64 = 0x1000;
const EAFNOSUPPORT: u64 = (-97i64) as u64;
const EPROTONOSUPPORT: u64 = (-93i64) as u64;
const ENOTSOCK: u64 = (-88i64) as u64;
const ECONNREFUSED: u64 = (-111i64) as u64;
const ETIMEDOUT: u64 = (-110i64) as u64;
const ENETDOWN: u64 = (-100i64) as u64;
const EISCONN: u64 = (-106i64) as u64;
const ENOBUFS: u64 = (-105i64) as u64;

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
const MAX_FDS: usize = 64;
const PATH_MAX: usize = 4096;

/// The largest single mapping a guest may ask for.
///
/// Not a policy about greed. `page_up` rounds up and multiplies, which wraps
/// for a length near `usize::MAX`, so an unbounded request would quietly
/// produce a *small* allocation and hand back a pointer to far less memory
/// than the guest was told it had. A cap makes that unreachable at the one
/// place it can be checked cheaply.
const MAP_MAX: u64 = 64 * 1024 * 1024;

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
    sysretq

    .globl glados_enter_guest
glados_enter_guest:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov [rip + GLADOS_HOST_RSP], rsp
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

/// The three globals the stub reaches through, as one value.
///
/// Read and written by the scheduler at a switch. That is what makes them
/// per-task without any assembly changing: a global that is only live while
/// its own task is running *is* per-task, provided somebody swaps it at the
/// boundary.
pub fn ring3_now() -> crate::task::Ring3 {
    unsafe {
        crate::task::Ring3 {
            guest_rsp: core::ptr::read(core::ptr::addr_of!(GLADOS_GUEST_RSP)),
            syscall_stack: core::ptr::read(core::ptr::addr_of!(GLADOS_SYSCALL_STACK)),
            host_rsp: core::ptr::read(core::ptr::addr_of!(GLADOS_HOST_RSP)),
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
            core::ptr::write(core::ptr::addr_of_mut!(GLADOS_GUEST_RSP), r.guest_rsp);
            core::ptr::write(core::ptr::addr_of_mut!(GLADOS_SYSCALL_STACK), r.syscall_stack);
            core::ptr::write(core::ptr::addr_of_mut!(GLADOS_HOST_RSP), r.host_rsp);
            crate::cpu::wrmsr(IA32_FS_BASE, r.fs_base);
        },
        None => {
            if let Some(sp) = unsafe { SPACE.get().as_ref() } {
                unsafe { crate::cpu::wrmsr(IA32_FS_BASE, sp.saved_fs) };
            }
        }
    }
}

/// One anonymous mapping the guest asked for and has not given back.
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

static SPACE: Racy<Option<Space>> = Racy::new(None);

/// Say what the running guest is, for the `/proc` files that describe it.
///
/// Separate from `install` rather than another argument to it, because these
/// are names and that takes ranges: the loader knows both, and folding them
/// into one call would mean `Regions` carrying strings it has no use for.
pub fn name_guest(argv: &[&str], image: &str, interp: Option<&str>) {
    if let Some(sp) = unsafe { SPACE.get() }.as_mut() {
        sp.argv = argv.iter().map(|a| String::from(*a)).collect();
        sp.image_path = String::from(image);
        sp.interp_path = interp.map(String::from);
    }
}

/// The path the running guest was loaded from, if one is running.
pub fn guest_image() -> Option<String> {
    let sp = unsafe { SPACE.get() }.as_ref()?;
    (!sp.image_path.is_empty()).then(|| sp.image_path.clone())
}

pub fn guest_argv() -> Vec<String> {
    unsafe { SPACE.get() }.as_ref().map(|s| s.argv.clone()).unwrap_or_default()
}

/// Every range the guest owns, with a name for the ones that have one.
///
/// In ascending address order, because that is the order `/proc/self/maps` is
/// read in and a reader scanning for a containing range may stop at the first
/// one past its address.
pub fn guest_regions() -> Vec<(u64, usize, String)> {
    let Some(sp) = (unsafe { SPACE.get() }).as_ref() else { return Vec::new() };
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
fn owns(at: u64, len: usize) -> bool {
    unsafe { SPACE.get().as_ref().is_some_and(|s| s.owns(at, len)) }
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
fn reachable(at: u64, len: usize, need_write: bool) -> bool {
    if !owns(at, len) {
        return false;
    }
    let Some(end) = at.checked_add(len as u64) else { return false };
    let mut page = at & !(PAGE - 1);
    while page < end {
        match crate::mem::paging::query(page) {
            Some(p) if p.present && (!need_write || p.write) => {}
            _ => return false,
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
pub fn install(r: Regions) {
    let brk = r.brk;
    teardown();
    unsafe {
        *SPACE.get() = Some(Space {
            image: r.image,
            interp: r.interp,
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
            cwd: String::from("/"),
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
    // **Before anything else, and unconditionally.** A guest that took the
    // display and then faulted is exactly the case this has to cover, and a
    // release conditional on a tidy exit would leave the desktop stood down
    // with the last frame of a dead program on it and no way back short of a
    // reboot.
    super::dev::hold_screen(false);
    super::input::stop();
    super::thread::reset();
    let mut freed = 0;
    unsafe {
        if let Some(sp) = SPACE.get().as_mut() {
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
            for m in sp.maps.drain(..) {
                // A guest is free to exit having mprotected its mappings to
                // something the heap cannot reuse. Handing a read-only or
                // absent page back to the allocator would poison it for
                // whatever asks next, and the symptom would appear in an
                // unrelated subsystem hours later.
                give_back(m.at, m.len, Some(m.from));
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
                Some(sp.image),
                sp.interp,
                Some(sp.stack),
                Some(Region { at: sp.brk_start, len: (sp.brk_end - sp.brk_start) as usize }),
            ]
            .into_iter()
            .flatten()
            {
                crate::mem::paging::protect(r.at, r.len, crate::mem::paging::Perm::RWX);
            }
            crate::cpu::wrmsr(IA32_FS_BASE, sp.saved_fs);
        }
        *SPACE.get() = None;
    }
    freed
}

fn page_up(n: usize) -> usize {
    n.max(1).div_ceil(4096) * 4096
}

fn alloc_pages(len: usize) -> Option<u64> {
    use alloc::alloc::{alloc_zeroed, Layout};
    let layout = Layout::from_size_align(page_up(len), 4096).ok()?;
    let p = unsafe { alloc_zeroed(layout) };
    if p.is_null() {
        return None;
    }
    Some(p as u64)
}

fn free_pages(at: u64, len: usize) {
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
    Some(match nr {
        SYS_OPEN | SYS_STAT | SYS_LSTAT | SYS_ACCESS => 0,
        SYS_OPENAT | SYS_NEWFSTATAT => 1,
        _ => return None,
    })
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

        // Bits 47:32 are what `syscall` loads. Bits 63:48 are what `sysret`
        // counts from, and it counts to selectors that only exist because the
        // GDT was widened for them.
        let star = (crate::cpu::gdt::KERNEL_CS as u64) << 32
            | (crate::cpu::gdt::SYSRET_BASE as u64) << 48;
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

fn record(mut c: Call) {
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
    let t = unsafe { TRACE.get() };
    if t.len() < TRACE_CAP {
        t.push(c);
    }
}

/// The guest wrote something. Only the two standard descriptors, and every
/// other fd is `-EBADF` rather than silently accepted -- a write to fd 7 that
/// reported success would be a program whose output vanished.
fn sys_write(fd: u64, buf: u64, len: usize) -> u64 {
    // **Ask the table, not the number.** This compared `fd` against 1 and 2,
    // which is right until a guest does the one thing every shell does:
    // `close(1)` then `open(...)` hands the file descriptor 1, and a write to
    // it went to the console -- output the guest had redirected, printed to
    // the terminal, reported as successful. `close(1)` alone was worse, since
    // writes to a descriptor that is not open have to be `EBADF`.
    if is_socket(fd) {
        return sys_send(fd, buf, len as u64);
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
fn read_cstr(at: u64) -> Result<String, u64> {
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
fn with_fds<T>(f: impl FnOnce(&mut Vec<Option<super::fs::Fd>>, &str) -> T) -> Option<T> {
    let sp = unsafe { SPACE.get() };
    // Destructured rather than borrowed twice, which is also why the working
    // directory is no longer cloned here. It was, because `&mut sp.fds` and
    // `&sp.cwd` are two borrows of one `sp` -- so every `read` in a guest's
    // copy loop allocated and freed a string it never looked at.
    let Space { fds, cwd, .. } = sp.as_mut()?;
    Some(f(fds, cwd))
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

fn sys_openat(dirfd: u64, path_at: u64, flags: u64, _mode: u64) -> u64 {
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
    if cwd_relative && (dirfd as i64) != super::fs::AT_FDCWD {
        // A descriptor-relative open needs the directory's path, which means
        // keeping one per open directory. Refused rather than resolved against
        // the wrong place.
        return ENOSYS;
    }
    with_fds(|fds, cwd| {
        let Some(path) = super::fs::resolve(cwd, &raw) else { return ENOENT };
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
fn sys_name_op(path_at: u64, op: u8) -> u64 {
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
fn sys_getcwd(buf: u64, len: u64) -> u64 {
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
fn sys_nanosleep(req: u64) -> u64 {
    if !reachable(req, 16, false) {
        return EFAULT;
    }
    let (sec, nsec) = unsafe {
        (
            core::ptr::read_volatile(req as *const u64),
            core::ptr::read_volatile((req + 8) as *const u64),
        )
    };
    if nsec >= 1_000_000_000 {
        return EINVAL;
    }
    let hz = crate::TIMER_HZ as u64;
    let want = sec.saturating_mul(hz) + nsec * hz / 1_000_000_000;
    let until = crate::dev::lapic::ticks().saturating_add(want);
    while crate::dev::lapic::ticks() < until {
        core::hint::spin_loop();
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
fn sys_writev(fd: u64, iov: u64, cnt: u64) -> u64 {
    // Linux's own bound. Refused rather than clamped: a caller that asked for
    // more than this has a bug, and writing the first thousand of its buffers
    // would hide it behind a short count.
    if cnt > 1024 {
        return EINVAL;
    }
    if cnt == 0 {
        return 0;
    }
    let Some(bytes) = (cnt as usize).checked_mul(16) else { return EINVAL };
    if !reachable(iov, bytes, false) {
        return EFAULT;
    }
    let mut total: u64 = 0;
    for i in 0..cnt as usize {
        let at = iov + (i * 16) as u64;
        let (base, len) = unsafe {
            (
                core::ptr::read_volatile(at as *const u64),
                core::ptr::read_volatile((at + 8) as *const u64),
            )
        };
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

fn sys_socket(domain: u64, kind: u64, proto: u64) -> u64 {
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

fn sys_connect(fd: u64, at: u64, len: u64) -> u64 {
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

fn sys_send(fd: u64, buf: u64, len: u64) -> u64 {
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

fn sys_recv(fd: u64, buf: u64, len: u64) -> u64 {
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

fn sys_shutdown(fd: u64, _how: u64) -> u64 {
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
fn sys_readv(fd: u64, iov: u64, cnt: u64) -> u64 {
    if cnt > 1024 {
        return EINVAL;
    }
    if cnt == 0 {
        return 0;
    }
    let Some(bytes) = (cnt as usize).checked_mul(16) else { return EINVAL };
    if !reachable(iov, bytes, false) {
        return EFAULT;
    }
    let mut total: u64 = 0;
    for i in 0..cnt as usize {
        let at = iov + (i * 16) as u64;
        let (base, len) = unsafe {
            (
                core::ptr::read_volatile(at as *const u64),
                core::ptr::read_volatile((at + 8) as *const u64),
            )
        };
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
fn sys_access(path_at: u64, mode: u64) -> u64 {
    let raw = match read_cstr(path_at) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if raw.is_empty() {
        return ENOENT;
    }
    let found = with_fds(|_, cwd| super::fs::resolve(cwd, &raw)).flatten();
    let Some(path) = found else { return ENOENT };
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
fn sys_sendfile(out: u64, into: u64, off_at: u64, count: u64) -> u64 {
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
fn sys_uname(buf: u64) -> u64 {
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
    put(3, "one address space, no processes");
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
fn clock_pair(which: u64) -> Option<(u64, u64)> {
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

fn sys_clock_gettime(which: u64, at: u64) -> u64 {
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

fn sys_gettimeofday(tv: u64, tz: u64) -> u64 {
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
fn sys_fcntl(fd: u64, cmd: u64, arg: u64) -> u64 {
    let known = with_fds(|fds, _| matches!(fds.get(fd as usize), Some(Some(_)))).unwrap_or(false);
    if !known {
        return EBADF;
    }
    match cmd {
        0 => sys_dup(fd, None).max_free_from(arg), // F_DUPFD
        1 => 0,                                    // F_GETFD, nothing is close-on-exec
        2 => 0,                                    // F_SETFD, accepted and not stored
        3 => 0,                                    // F_GETFL, everything here is O_RDONLY
        4 => 0,                                    // F_SETFL, no flag it could set applies
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
/// **`fork` is the one thing this system cannot grow into.** One address space
/// is its founding claim rather than a shortcut, and a `clone` without
/// `CLONE_VM` is asking for a second one. `CLONE_THREAD` is the other half: a
/// clone without it becomes a process, and there are none. Both are refused
/// with `ENOSYS` rather than approximated, because a thread handed back for a
/// process request is two names for one address space and a program free to
/// write through both.
fn sys_clone(flags: u64, stack: u64, ptid: u64, ctid: u64, tls: u64, rip: u64) -> u64 {
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
/// (address, how many times it has been woken).
static FUTEX: Racy<[(u64, u64); FUTEX_SLOTS]> = Racy::new([(0, 0); FUTEX_SLOTS]);

fn futex_bump(addr: u64) -> u64 {
    let t = unsafe { &mut *FUTEX.get() };
    if let Some(e) = t.iter_mut().find(|e| e.0 == addr) {
        e.1 += 1;
        return e.1;
    }
    if let Some(e) = t.iter_mut().find(|e| e.0 == 0) {
        *e = (addr, 1);
        return 1;
    }
    // Out of slots. Every waiter re-reads the word it is waiting on as well,
    // so a lost wake costs latency rather than correctness -- which is the
    // whole reason the value check is there and not merely belt and braces.
    0
}

fn futex_count(addr: u64) -> u64 {
    let t = unsafe { &*FUTEX.get() };
    t.iter().find(|e| e.0 == addr).map(|e| e.1).unwrap_or(0)
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
fn sys_futex(uaddr: u64, op: u64, val: u64, timeout: u64) -> u64 {
    const FUTEX_WAIT: u64 = 0;
    const FUTEX_WAKE: u64 = 1;
    const PRIVATE: u64 = 128;
    const CLOCK_REALTIME: u64 = 256;
    let kind = op & !(PRIVATE | CLOCK_REALTIME);
    if uaddr % 4 != 0 {
        return EINVAL;
    }
    if !reachable(uaddr, 4, false) {
        return EFAULT;
    }
    match kind {
        FUTEX_WAIT => {
            let seen = unsafe { core::ptr::read_volatile(uaddr as *const u32) };
            if seen as u64 != val {
                // Already changed. `EAGAIN` and not zero, because a caller
                // that slept would re-check and a caller told zero would not.
                return EAGAIN;
            }
            let woken = futex_count(uaddr);
            // A relative timeout, and a zero pointer means forever.
            let until = if timeout != 0 && reachable(timeout, 16, false) {
                let (sec, nsec) = unsafe {
                    (
                        core::ptr::read_volatile(timeout as *const u64),
                        core::ptr::read_volatile((timeout + 8) as *const u64),
                    )
                };
                let hz = crate::TIMER_HZ as u64;
                Some(
                    crate::dev::lapic::ticks()
                        + sec.saturating_mul(hz)
                        + nsec * hz / 1_000_000_000,
                )
            } else {
                None
            };
            loop {
                let now = unsafe { core::ptr::read_volatile(uaddr as *const u32) };
                if now as u64 != val || futex_count(uaddr) != woken {
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
                crate::task::yield_now();
            }
        }
        FUTEX_WAKE => {
            futex_bump(uaddr);
            // How many were woken. There is no queue to count, so this answers
            // what was asked for, which is what a caller uses to decide
            // whether to bother with a syscall next time -- and over-reporting
            // is the safe direction, since it makes a caller do less.
            val.min(super::thread::live() as u64 + 1)
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
    crate::task::ring3_active(true, stack);
    unsafe {
        core::ptr::write(core::ptr::addr_of_mut!(GLADOS_SYSCALL_STACK), stack);
        crate::cpu::wrmsr(IA32_FS_BASE, w.fs);
    }
    let _ = unsafe { glados_enter_guest(w.rip, w.rsp) };
    // **Clearing this is the whole of how `pthread_join` works.** The joiner
    // futex-waits on the word, and the kernel zeroing it is the notification;
    // a kernel that ignored `CLONE_CHILD_CLEARTID` leaves a library waiting
    // forever on a thread that finished.
    if w.ctid != 0 && owns(w.ctid, 4) {
        unsafe { core::ptr::write_volatile(w.ctid as *mut u32, 0) };
        futex_bump(w.ctid);
    }
    crate::task::ring3_active(false, 0);
    if flags & (1 << 9) != 0 {
        crate::cpu::enable_interrupts();
    }
}

fn sys_dup3(from: u64, to: u64, flags: u64) -> u64 {
    const O_CLOEXEC: u64 = 0o2_000_000;
    if flags & !O_CLOEXEC != 0 {
        return EINVAL;
    }
    if from == to {
        return EINVAL;
    }
    sys_dup(from, Some(to))
}

fn sys_dup(from: u64, to: Option<u64>) -> u64 {
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

fn sys_close(fd: u64) -> u64 {
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

fn sys_read(fd: u64, buf: u64, len: u64) -> u64 {
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
            crate::task::yield_now();
        }
    }
    with_fds(|fds, _| match fds.get_mut(fd as usize) {
        // Nothing types at a guest, so stdin is permanently at end of file.
        // Zero is the honest answer and is what a program reading a closed
        // pipe sees.
        Some(Some(super::fs::Fd::Stdin)) => 0,
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
fn sys_pread64(fd: u64, buf: u64, len: u64, off: u64) -> u64 {
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
fn sys_readlinkat(dirfd: u64, path_at: u64, buf: u64, size: u64) -> u64 {
    let raw = match read_cstr(path_at) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if raw.is_empty() {
        return ENOENT;
    }
    if !raw.starts_with('/') && (dirfd as i64) != super::fs::AT_FDCWD {
        return ENOSYS;
    }
    if size == 0 {
        return EINVAL;
    }
    let found = with_fds(|_, cwd| super::fs::resolve(cwd, &raw)).flatten();
    let Some(path) = found else { return ENOENT };
    let Some(target) = super::proc::link(&path) else {
        return if super::proc::claims(&path) || sysbox::blob_len(&path).is_some() {
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

fn sys_time(tloc: u64) -> u64 {
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
fn sys_sysinfo(buf: u64) -> u64 {
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
fn sys_sched_getaffinity(_pid: u64, len: u64, mask: u64) -> u64 {
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

fn sys_getrandom(buf: u64, len: u64, _flags: u64) -> u64 {
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

fn sys_set_robust_list(_head: u64, len: u64) -> u64 {
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
fn sys_prlimit64(_pid: u64, resource: u64, new: u64, old: u64) -> u64 {
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
    let Some(sp) = (unsafe { SPACE.get() }).as_ref() else { return EINVAL };
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

fn sys_lseek(fd: u64, off: u64, whence: u64) -> u64 {
    with_fds(|fds, _| match fds.get_mut(fd as usize) {
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

fn write_stat(buf: u64, kind: super::fs::Kind, size: usize, ino: u64) -> u64 {
    if !reachable(buf, 144, true) {
        return EFAULT;
    }
    let b = super::fs::stat_bytes(kind, size, ino);
    unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), buf as *mut u8, 144) };
    0
}

fn sys_fstat(fd: u64, buf: u64) -> u64 {
    let found = with_fds(|fds, _| match fds.get(fd as usize) {
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

fn sys_statat(dirfd: u64, path_at: u64, buf: u64, flags: u64) -> u64 {
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
    if !raw.starts_with('/') && (dirfd as i64) != super::fs::AT_FDCWD {
        return ENOSYS;
    }
    let found = with_fds(|_, cwd| super::fs::resolve(cwd, &raw)).flatten();
    let Some(path) = found else { return ENOENT };
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

fn sys_getdents64(fd: u64, buf: u64, len: u64) -> u64 {
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
fn sys_ioctl(fd: u64, req: u64, arg: u64) -> u64 {
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
fn sys_brk(want: u64) -> u64 {
    let sp = unsafe { SPACE.get() };
    let Some(sp) = sp.as_mut() else { return 0 };
    if want >= sp.brk_start && want <= sp.brk_end {
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
fn sys_mmap(addr: u64, len: u64, prot: u64, flags: u64, fd: u64, off: u64) -> u64 {
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
        let dev = (unsafe { SPACE.get() })
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
            let sp = unsafe { SPACE.get() }.as_mut();
            if let Some(sp) = sp {
                sp.maps.push(Mapping { at, len: span, from: Source::Device });
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
        let Some(sp) = (unsafe { SPACE.get() }).as_ref() else { return ENOMEM };
        match sp.fds.get(fd as usize).and_then(|f| f.as_ref()) {
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
        let Some(at) = alloc_pages(len as usize) else { return ENOMEM };
        (at, Some(Source::Heap))
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
            let sp = unsafe { SPACE.get() };
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
    }
}

/// Give a whole mapping back. Partial unmapping is refused rather than
/// approximated: splitting one allocation into two is not something the heap
/// underneath can express, and silently unmapping more than was asked is worse
/// than saying no.
fn sys_munmap(at: u64, len: u64) -> u64 {
    let sp = unsafe { SPACE.get() };
    let Some(sp) = sp.as_mut() else { return EINVAL };
    match sp.maps.iter().position(|m| m.at == at && m.len == len as usize) {
        Some(i) => {
            let m = sp.maps.remove(i);
            // **Close it before giving it back.** `teardown` does this for a
            // guest that exits still holding mappings, and this path was
            // missed: an explicit unmap returned pages to the allocator with
            // their U bit still set, so the next thing to be handed that
            // memory -- kernel or otherwise -- came with ring-3 access
            // attached. A `diag all` caught it and `diag paging` alone did
            // not, because it only shows up once something else has run.
            give_back(m.at, m.len, Some(m.from));
            0
        }
        None => EINVAL,
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
fn sys_mprotect(at: u64, len: u64, prot: u64) -> u64 {
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
fn sys_arch_prctl(code: u64, addr: u64) -> u64 {
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
    let nr = f.rax;
    let args = [f.rdi, f.rsi, f.rdx, f.r10, f.r8, f.r9];

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
        SYS_SENDFILE => (sys_sendfile(f.rdi, f.rsi, f.rdx, f.r10), true),
        SYS_GETCWD => (sys_getcwd(f.rdi, f.rsi), true),
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
        SYS_RT_SIGACTION | SYS_RT_SIGPROCMASK => (0, true),
        // No parent, and no group but the one. `getppid` answering zero is
        // what a process reparented to nothing reports.
        SYS_GETPPID => (0, true),
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
        SYS_CLONE => (sys_clone(f.rdi, f.rsi, f.rdx, f.r10, f.r8, f.rip), true),
        SYS_FUTEX => (sys_futex(f.rdi, f.rsi, f.rdx, f.r10), true),
        SYS_GETTID => (super::thread::current_tid(), true),
        // A thread yielding to another is a thing this kernel can actually do,
        // which is unusual in this file: most scheduling calls are answered
        // with a shape rather than an action.
        SYS_SCHED_YIELD => {
            crate::task::yield_now();
            (0, true)
        }
        // One process, and it is the guest. Reporting a pid at all is what
        // stops a runtime deciding it failed to start.
        SYS_GETPID | SYS_SET_TID_ADDRESS => (1, true),
        // Root, and every id the same. There is no privilege boundary above a
        // guest here to be anything else, which is the same fact `AT_SECURE`
        // reports as zero and `stat` reports as uid 0 -- said once per place
        // that asks rather than three different ways.
        SYS_GETUID | SYS_GETGID | SYS_GETEUID | SYS_GETEGID => (0, true),
        SYS_BRK => (sys_brk(f.rdi), true),
        SYS_MMAP => (sys_mmap(f.rdi, f.rsi, f.rdx, f.r10, f.r8, f.r9), true),
        SYS_MPROTECT => (sys_mprotect(f.rdi, f.rsi, f.rdx), true),
        SYS_MUNMAP => (sys_munmap(f.rdi, f.rsi), true),
        SYS_ARCH_PRCTL => (sys_arch_prctl(f.rdi, f.rsi), true),
        SYS_EXIT | SYS_EXIT_GROUP => {
            record(Call { nr, args, ret: 0, served: true, path: [0; PATH_SNIP], path_len: 0 });
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
    f.rax = ret;
}

extern "sysv64" {
    fn glados_leave_guest(code: u64) -> !;
}

/// Set on the way out so an exit code of zero is still distinguishable from
/// "the guest never called exit at all".
pub const EXITED: u64 = 1 << 32;
/// Set when the guest was killed for running past its deadline.
pub const OVERRAN: u64 = 1 << 34;

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

static DEADLINE: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Whether the running guest has outstayed its welcome.
pub fn overran(now: u64) -> bool {
    let d = DEADLINE.load(Ordering::Relaxed);
    d != 0 && now >= d && GUEST_RUNNING.load(Ordering::Relaxed)
}

/// End a guest that would not stop on its own.
///
/// # Safety
/// Only from an interrupt taken **at ring 3**, so the guest itself was
/// executing. Called while the kernel is working on the guest's behalf it
/// would abandon whatever that work was holding.
pub unsafe fn kill_overrun() -> ! {
    unsafe { kill_with(OVERRAN) }
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
    unsafe { kill_with(OVERRAN) }
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

static LAST_FAULT: Racy<Option<Fault>> = Racy::new(None);

/// The same three addresses said in words, resolved while the space was still
/// installed.
///
/// `run` tears the space down on the line after the guest returns, and
/// `locate` reads the space, so resolving at print time answered "no guest"
/// three times about a guest that had just died. The strings are made in
/// `run`, one line earlier, which is the only moment both the fault and the
/// map it should be read against exist together.
static FAULT_AT: Racy<Option<(String, String, String)>> = Racy::new(None);

/// The words around the guest's stack pointer when it died, each said in
/// words as well as in hex.
///
/// **Because a return address is the only thing that names the caller here.**
/// There is no unwinder and no symbols, so "who called this with a null" is
/// answerable only by reading the stack the call left behind -- and the stack
/// is freed by the teardown two lines after the guest returns. Sixteen words
/// either side is enough to hold a small frame and the return address above
/// it, and small enough that the report stays readable.
static FAULT_STACK: Racy<Option<Vec<(u64, u64, String)>>> = Racy::new(None);

pub fn fault_stack() -> Option<Vec<(u64, u64, String)>> {
    unsafe { (*FAULT_STACK.get()).clone() }
}

/// Each general-purpose register, named, with what it points at.
///
/// Resolved beside the stack words and for the same reason: `locate` reads the
/// space and the space is gone two lines later. The first version of this
/// resolved at print time and answered "no guest" fifteen times about a guest
/// that had just died, which is the identical mistake made twice, once for
/// three addresses and once for fifteen.
static FAULT_REGS: Racy<Option<Vec<(&'static str, u64, String)>>> = Racy::new(None);

pub fn fault_regs() -> Option<Vec<(&'static str, u64, String)>> {
    unsafe { (*FAULT_REGS.get()).clone() }
}

pub fn last_fault() -> Option<Fault> {
    unsafe { *LAST_FAULT.get() }
}

/// Where the rip, the faulting address and the stack pointer were, in words.
pub fn fault_where() -> Option<(String, String, String)> {
    unsafe { (*FAULT_AT.get()).clone() }
}

/// Which of the guest's own ranges an address falls in.
///
/// An address on its own is a number. `0x30d6000` means nothing until it is
/// "inside the interpreter, at +0x2000", and that is the difference between a
/// fault report and a diagnosis -- the same reason `cpu::code::locate` exists
/// for the kernel's own faults.
pub fn locate(at: u64) -> alloc::string::String {
    let Some(sp) = (unsafe { SPACE.get() }).as_ref() else {
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
                (at >= m.at && at < m.at.saturating_add(m.len as u64))
                    .then(|| alloc::format!("a mapping at {:#x} +{:#x}", m.at, at - m.at))
            })
        })
        .unwrap_or_else(|| alloc::string::String::from("nothing this guest owns"))
}

/// Whether a guest is on the stack right now.
///
/// A plain atomic that Rust both writes and reads, rather than a look at the
/// stack pointer `glados_enter_guest` parked. That indirection is deliberate:
/// the fault handler asks this question from inside an interrupt gate on an
/// IST stack, having just come from ring 3, and it must not depend on reading
/// a `static mut` whose only writer is assembly. The first version did, and it
/// took a #GP inside the read on exactly that path.
static GUEST_RUNNING: AtomicBool = AtomicBool::new(false);

pub fn running() -> bool {
    GUEST_RUNNING.load(Ordering::Relaxed)
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
pub unsafe fn kill(f: Fault) -> ! {
    let vector = f.regs.vector;
    // Copied out before the longjmp, which abandons the stack this arrived on.
    unsafe { *LAST_FAULT.get() = Some(f) };
    unsafe { kill_with(FAULTED | vector) }
}

/// The longjmp both reasons share.
///
/// # Safety
/// Only while a guest is running, and only from a context that may abandon its
/// stack.
unsafe fn kill_with(code: u64) -> ! {
    GUEST_RUNNING.store(false, Ordering::Relaxed);
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
            "mov rsp, [rip + GLADOS_HOST_RSP]",
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
pub unsafe fn run(entry: u64, stack_top: u64) -> u64 {
    arm();
    // The guest is about to be allowed to write `FS`, and `FS` is a register
    // of this machine that the kernel goes on using afterwards. Parked here
    // rather than inside `arch_prctl`, so a guest that sets it forty times
    // still restores the one value that was true before any of them.
    unsafe {
        if let Some(sp) = SPACE.get().as_mut() {
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
        *LAST_FAULT.get() = None;
        *FAULT_AT.get() = None;
        *FAULT_REGS.get() = None;
    }
    GUEST_RUNNING.store(true, Ordering::Relaxed);
    // Zero means no deadline, and `overran` already reads a zero `DEADLINE`
    // as "nothing to enforce", so the two agree without a second condition.
    let want = limit();
    DEADLINE.store(
        if want == 0 { 0 } else { crate::dev::lapic::ticks() + want },
        Ordering::Relaxed,
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
        core::ptr::read(core::ptr::addr_of!(GLADOS_SYSCALL_STACK))
    });
    let code = unsafe { glados_enter_guest(entry, stack_top) };
    crate::task::ring3_active(false, 0);
    // **Every thread has to be gone before the space is.** A child still
    // running would be reading regions `teardown` is about to hand back, and
    // `exit_group` only asks the others to stop -- ending a thread means
    // longjmping out of its own stack, which only that thread can do.
    // Bounded, because a thread that makes no syscall never notices the ask
    // and the alternative is a shell that never comes back.
    super::thread::begin_exit();
    let give_up = crate::dev::lapic::ticks() + 200;
    while super::thread::live() > 0 && crate::dev::lapic::ticks() < give_up {
        crate::task::yield_now();
    }
    DEADLINE.store(0, Ordering::Relaxed);
    GUEST_RUNNING.store(false, Ordering::Relaxed);
    if flags & (1 << 9) != 0 {
        crate::cpu::enable_interrupts();
    }
    // Before the teardown on the next line, because that is what `locate`
    // reads. This is the only point where the fault and the map it has to be
    // read against are both alive.
    unsafe {
        *FAULT_AT.get() =
            last_fault().map(|f| (locate(f.regs.rip), locate(f.cr2), locate(f.regs.rsp)));
        *FAULT_REGS.get() = last_fault().map(|f| {
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
        *FAULT_STACK.get() = last_fault().and_then(|f| {
            let base = f.regs.rsp & !7;
            let mut out = Vec::new();
            for i in 0..24u64 {
                let at = base + i * 8;
                // Read only what the guest actually owns and can be read: the
                // stack pointer of a program that died badly is not
                // necessarily a stack pointer.
                if !reachable(at, 8, false) {
                    continue;
                }
                let v = core::ptr::read_unaligned(at as *const u64);
                out.push((at, v, locate(v)));
            }
            (!out.is_empty()).then_some(out)
        });
    }
    teardown();
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

    // brk, on a region installed for the check and torn down after it. The
    // claim that earns its place is the failure mode: Linux answers the
    // *unchanged* break rather than an error, and an implementation returning
    // -ENOMEM would hand libc 0xFFFFFFFFFFFFFFF4 as a heap address.
    {
        let region = 0x10_0000u64;
        let fake = Region { at: region, len: 4096 * 4 };
        install(Regions { image: fake, stack: fake, brk: fake, interp: None });
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
        unsafe { *SPACE.get() = None };
    }

    // mmap's refusals, and the two shapes that stopped being refusals. Two of
    // these used to assert the opposite and both were about this loader rather
    // than about the call, which is why they went the way `ET_EXEC` went.
    {
        let fake = Region { at: 0x10_0000, len: 4096 };
        install(Regions { image: fake, stack: fake, brk: fake, interp: None });
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
        install(Regions { image: owned, stack: owned, brk: owned, interp: None });

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
        install(Regions { image: owned, stack: owned, brk: owned, interp: None });
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
        install(Regions { image: owned, stack: owned, brk: owned, interp: None });
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
        install(Regions { image: mine, stack: mine, brk: mine, interp: None });
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
        install(Regions { image: owned, stack: owned, brk: owned, interp: None });
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

    // Teardown puts rights back on every region, and the interpreter is the
    // one that was missed. Two pages so the interpreter's is a different page
    // from the image's, because a claim where both are the same address passes
    // while the interpreter is not restored at all.
    if let Some(two) = alloc_pages(8192) {
        let img = Region { at: two, len: 4096 };
        let interp = Region { at: two + 4096, len: 4096 };
        install(Regions { image: img, stack: img, brk: img, interp: Some(interp) });
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
        install(Regions { image: mine, stack: mine, brk: mine, interp: None });
        out.push((
            "MAP_FIXED over memory the guest already holds answers that address",
            sys_mmap(own, 4096, 3, MAP_ANONYMOUS | MAP_FIXED, u64::MAX, 0) == own,
        ));
        out.push((
            "and records nothing, because something already holds those pages",
            unsafe { SPACE.get() }.as_ref().is_some_and(|sp| sp.maps.is_empty()),
        ));
        // Rights back before the page goes back, which is the mistake this
        // tree has now made three times in three different places.
        crate::mem::paging::release_to_heap(own, 8192);
        free_pages(own, 8192);
        teardown();
    }

    {
        let fake = Region { at: 0x10_0000, len: 4096 };
        install(Regions { image: fake, stack: fake, brk: fake, interp: None });

        let at = sys_mmap(0, 8192, 3, MAP_ANONYMOUS, u64::MAX, 0);
        let got = (at as i64) > 0 && at % 4096 == 0;
        // It has to be real memory, and it has to be zeroed: an allocator
        // handed dirty pages produces a program that works once.
        let zeroed = got && unsafe { core::slice::from_raw_parts(at as *const u8, 8192) }.iter().all(|b| *b == 0);
        out.push(("an anonymous mapping is page-aligned and zeroed", got && zeroed));
        out.push((
            "a partial unmap is refused rather than approximated",
            sys_munmap(at, 4096) == EINVAL,
        ));
        out.push(("unmapping the whole thing gives it back", sys_munmap(at, 8192) == 0));
        out.push((
            "unmapping what was never mapped is EINVAL",
            sys_munmap(at, 8192) == EINVAL,
        ));
        unsafe { *SPACE.get() = None };
    }

    // arch_prctl, and the refusal that is the point of it.
    {
        let mut slot = 0u64;
        let at = &mut slot as *mut u64 as u64;
        // GET_FS writes eight bytes through a guest pointer, so it is bounds
        // checked now, so the destination has to be a range the guest owns.
        let owned = Region { at, len: 8 };
        install(Regions { image: owned, stack: owned, brk: owned, interp: None });
        let was = unsafe { crate::cpu::rdmsr(IA32_FS_BASE) };
        let set = sys_arch_prctl(ARCH_SET_FS, 0xDEAD_0000);
        let got = sys_arch_prctl(ARCH_GET_FS, at);
        unsafe { crate::cpu::wrmsr(IA32_FS_BASE, was) };
        unsafe { *SPACE.get() = None };
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
        install(Regions { image: one, stack: one, brk: one, interp: None });
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
        unsafe { *SPACE.get() = None };
    }
    // mprotect, on a page taken for the check and given straight back.
    {
        use alloc::alloc::{alloc_zeroed, dealloc, Layout};
        if let Ok(layout) = Layout::from_size_align(4096, 4096) {
            let mem = unsafe { alloc_zeroed(layout) };
            if !mem.is_null() {
                let at = mem as u64;
                let owned = Region { at, len: 4096 };
                install(Regions { image: owned, stack: owned, brk: owned, interp: None });
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
                unsafe { *SPACE.get() = None };
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
        build_stack(small.as_mut_ptr(), 64, &["x"], Aux::default()).is_none(),
    ));
    let mut big = [0u8; 512];
    let aux = Aux { phdr: 0x1000, phent: 56, phnum: 2, entry: 0x1078, base: 0 };
    let sp = build_stack(big.as_mut_ptr(), 512, &["cat", "/ai/about"], aux);
    out.push((
        "and one large enough answers a 16-byte-aligned pointer inside itself",
        sp.is_some_and(|v| v % 16 == 0 && v >= big.as_ptr() as u64
            && v < big.as_ptr() as u64 + 512),
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
            r >= big.as_ptr() as u64 && r + 16 <= big.as_ptr() as u64 + 512
        }),
    ));
    out.push((
        "a header table no segment covers omits the whole group rather than pointing at nothing",
        {
            let mut small = [0u8; 512];
            let none = Aux { phdr: 0, phent: 56, phnum: 2, entry: 0, base: 0 };
            let s2 = build_stack(small.as_mut_ptr(), 512, &["x"], none);
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
const ENVIRON: [&str; 5] =
    ["PATH=/bin:/usr/bin:/tmp", "HOME=/", "PWD=/", "TERM=dumb", "USER=root"];

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

pub fn build_stack(base: *mut u8, size: usize, args: &[&str], aux: Aux) -> Option<u64> {
    let bottom = base as usize;
    let mut top = bottom.checked_add(size)?;

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
        Some(*top as u64)
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
    let random = top as u64;
    unsafe {
        crate::rng::fill(core::slice::from_raw_parts_mut(random as *mut u8, 16));
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
    let sp = top.checked_sub(words * 8)? & !0xF;
    if sp < bottom {
        return None;
    }
    unsafe {
        let p = sp as *mut u64;
        p.write(ptrs.len() as u64);
        for (i, v) in ptrs.iter().enumerate() {
            p.add(1 + i).write(*v);
        }
        p.add(1 + ptrs.len()).write(0); // argv terminator
        let e0 = 2 + ptrs.len();
        for (i, v) in envs.iter().enumerate() {
            p.add(e0 + i).write(*v);
        }
        p.add(e0 + envs.len()).write(0); // envp terminator
        let a0 = e0 + envs.len() + 1;
        for (i, (k, v)) in pairs.iter().enumerate() {
            p.add(a0 + i * 2).write(*k);
            p.add(a0 + i * 2 + 1).write(*v);
        }
    }
    Some(sp as u64)
}

/// Name the calls stage 0 knows about, for a trace a person has to read.
pub fn name_of(nr: u64) -> &'static str {
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
        SYS_SENDTO => "sendto",
        SYS_RECVFROM => "recvfrom",
        SYS_SHUTDOWN => "shutdown",
        SYS_SETSOCKOPT => "setsockopt",
        SYS_GETSOCKOPT => "getsockopt",
        SYS_ACCESS => "access",
        SYS_FACCESSAT => "faccessat",
        SYS_SENDFILE => "sendfile",
        SYS_GETCWD => "getcwd",
        SYS_MKDIR => "mkdir",
        SYS_RMDIR => "rmdir",
        SYS_UNLINK => "unlink",
        SYS_NANOSLEEP => "nanosleep",
        SYS_CLOCK_NANOSLEEP => "clock_nanosleep",
        SYS_RT_SIGACTION => "rt_sigaction",
        SYS_RT_SIGPROCMASK => "rt_sigprocmask",
        SYS_GETPPID => "getppid",
        SYS_GETGROUPS => "getgroups",
        SYS_UNAME => "uname",
        SYS_FCNTL => "fcntl",
        SYS_GETTIMEOFDAY => "gettimeofday",
        SYS_CLOCK_GETTIME => "clock_gettime",
        SYS_DUP2 => "dup2",
        SYS_DUP3 => "dup3",
        SYS_CLONE => "clone",
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

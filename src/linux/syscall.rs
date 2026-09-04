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
pub const SYS_GETPID: u64 = 39;
pub const SYS_GETDENTS64: u64 = 217;
pub const SYS_SET_TID_ADDRESS: u64 = 218;
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
const ENAMETOOLONG: u64 = (-36i64) as u64;

/// `O_WRONLY` and `O_RDWR`. This view is read-only, so both are refused.
const O_WRONLY: u64 = 1;
const O_RDWR: u64 = 2;
const O_DIRECTORY: u64 = 0x10000;

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

/// One anonymous mapping the guest asked for and has not given back.
pub struct Mapping {
    pub at: u64,
    pub len: usize,
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
            || self.stack.holds(at, end)
            || (at >= self.brk_start && end <= self.brk_end)
            || self.maps.iter().any(|m| Region { at: m.at, len: m.len }.holds(at, end))
    }
}

static SPACE: Racy<Option<Space>> = Racy::new(None);

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
pub fn install(image: Region, stack: Region, brk: Region) {
    teardown();
    unsafe {
        *SPACE.get() = Some(Space {
            image,
            stack,
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
    let mut freed = 0;
    unsafe {
        if let Some(sp) = SPACE.get().as_mut() {
            for m in sp.maps.drain(..) {
                // A guest is free to exit having mprotected its mappings to
                // something the heap cannot reuse. Handing a read-only or
                // absent page back to the allocator would poison it for
                // whatever asks next, and the symptom would appear in an
                // unrelated subsystem hours later.
                crate::mem::paging::protect(m.at, m.len, crate::mem::paging::Perm::RW);
                free_pages(m.at, m.len);
                freed += 1;
            }
            // The image, stack and break came from `Exec` allocations the
            // `Guest` still owns and will drop, so they go back the same way.
            for r in [sp.image, sp.stack, Region { at: sp.brk_start, len: (sp.brk_end - sp.brk_start) as usize }] {
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
    // **Ask the table, not the number.** This compared `fd` against 1 and 2,
    // which is right until a guest does the one thing every shell does:
    // `close(1)` then `open(...)` hands the file descriptor 1, and a write to
    // it went to the console -- output the guest had redirected, printed to
    // the terminal, reported as successful. `close(1)` alone was worse, since
    // writes to a descriptor that is not open have to be `EBADF`.
    let sink = with_fds(|fds, _| {
        matches!(
            fds.get(fd as usize),
            Some(Some(super::fs::Fd::Stdout)) | Some(Some(super::fs::Fd::Stderr))
        )
    });
    if sink != Some(true) {
        return EBADF;
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
fn sys_openat(dirfd: u64, path_at: u64, flags: u64, _mode: u64) -> u64 {
    let raw = match read_cstr(path_at) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if flags & (O_WRONLY | O_RDWR) != 0 {
        return EACCES;
    }
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
        let is_dir = sysbox::is_dir(&path);
        if flags & O_DIRECTORY != 0 && !is_dir {
            return ENOTDIR;
        }
        let entry = if is_dir {
            super::fs::Fd::Dir { path: path.clone(), entries: sysbox::listing(&path), at: 0 }
        } else {
            // Asked before the copy is made, not after: the point is to not
            // allocate the file, so a check that reads it first and then
            // measures has already lost.
            let Some(size) = sysbox::blob_len(&path) else { return ENOENT };
            let held: usize = fds
                .iter()
                .filter_map(|f| match f {
                    Some(super::fs::Fd::File { data, .. }) => Some(data.len()),
                    _ => None,
                })
                .sum();
            if held.saturating_add(size) > OPEN_MAX_BYTES {
                return ENOMEM;
            }
            let Some(data) = sysbox::read_blob(&path) else { return ENOENT };
            super::fs::Fd::File { path: path.clone(), data, at: 0 }
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

fn sys_close(fd: u64) -> u64 {
    with_fds(|fds, _| match fds.get_mut(fd as usize) {
        Some(slot @ Some(_)) => {
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
    if !reachable(buf, len as usize, true) {
        return EFAULT;
    }
    with_fds(|fds, _| match fds.get_mut(fd as usize) {
        // Nothing types at a guest, so stdin is permanently at end of file.
        // Zero is the honest answer and is what a program reading a closed
        // pipe sees.
        Some(Some(super::fs::Fd::Stdin)) => 0,
        Some(Some(super::fs::Fd::File { data, at, .. })) => {
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
        Some(Some(super::fs::Fd::Dir { .. })) => EISDIR,
        _ => EBADF,
    })
    .unwrap_or(EBADF)
}

fn sys_lseek(fd: u64, off: u64, whence: u64) -> u64 {
    with_fds(|fds, _| match fds.get_mut(fd as usize) {
        Some(Some(super::fs::Fd::File { data, at, .. })) => {
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
        Some(Some(super::fs::Fd::Dir { at, .. })) => {
            if whence == 0 && off == 0 {
                *at = 0;
                0
            } else {
                EINVAL
            }
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
        Some(Some(super::fs::Fd::File { path, data, .. })) => {
            Some((super::fs::Kind::File, data.len(), super::fs::ino_of(path)))
        }
        Some(Some(super::fs::Fd::Dir { path, .. })) => {
            Some((super::fs::Kind::Dir, 0, super::fs::ino_of(path)))
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

fn sys_statat(dirfd: u64, path_at: u64, buf: u64) -> u64 {
    let raw = match read_cstr(path_at) {
        Ok(v) => v,
        Err(e) => return e,
    };
    // The same two refusals `openat` makes, for the same reasons and stated
    // once each. They were absent here, so `newfstatat` would take a relative
    // path with a real descriptor and resolve it against the working
    // directory -- answering confidently about a file that exists and is not
    // the one asked for, which is worse than refusing.
    if raw.is_empty() {
        return ENOENT;
    }
    if !raw.starts_with('/') && (dirfd as i64) != super::fs::AT_FDCWD {
        return ENOSYS;
    }
    let found = with_fds(|_, cwd| super::fs::resolve(cwd, &raw)).flatten();
    let Some(path) = found else { return ENOENT };
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
        Some(Some(super::fs::Fd::Dir { path, entries, at })) => {
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
fn sys_ioctl(fd: u64, _req: u64, _arg: u64) -> u64 {
    let known = with_fds(|fds, _| matches!(fds.get(fd as usize), Some(Some(_)))).unwrap_or(false);
    if known {
        ENOTTY
    } else {
        EBADF
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
fn sys_mmap(addr: u64, len: u64, prot: u64, flags: u64, fd: u64, _off: u64) -> u64 {
    if len == 0 {
        return EINVAL;
    }
    if len > MAP_MAX {
        return ENOMEM;
    }
    if flags & MAP_ANONYMOUS == 0 || fd as i64 != -1 {
        return ENOSYS;
    }
    if flags & MAP_FIXED != 0 && addr != 0 {
        return ENOMEM;
    }
    let Some(at) = alloc_pages(len as usize) else { return ENOMEM };
    // **Open it to ring 3, or the guest cannot touch what it just asked for.**
    // The loader opens the image, stack and break before the guest starts, and
    // a mapping made after that is not covered by any of them. At ring 0 this
    // was invisible, because the U bit meant nothing; the first ring-3 guest to
    // call `mmap` took a protection violation reading its own memory.
    let perm = crate::mem::paging::Perm {
        present: true,
        write: prot & PROT_WRITE != 0,
        exec: prot & PROT_EXEC != 0,
        user: true,
    };
    if !crate::mem::paging::protect(at, page_up(len as usize), perm) {
        // Partially applied rights are still rights, so close it on the way
        // out rather than handing the allocator whatever the walk managed.
        crate::mem::paging::protect(at, page_up(len as usize), crate::mem::paging::Perm::RW);
        free_pages(at, len as usize);
        return ENOMEM;
    }
    let sp = unsafe { SPACE.get() };
    match sp.as_mut() {
        Some(sp) => sp.maps.push(Mapping { at, len: len as usize }),
        None => {
            free_pages(at, len as usize);
            return ENOMEM;
        }
    }
    at
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
            crate::mem::paging::protect(m.at, m.len, crate::mem::paging::Perm::RW);
            free_pages(m.at, m.len);
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
            (sys_statat(super::fs::AT_FDCWD as u64, f.rdi, f.rsi), true)
        }
        SYS_NEWFSTATAT => (sys_statat(f.rdi, f.rsi, f.rdx), true),
        SYS_GETDENTS64 => (sys_getdents64(f.rdi, f.rsi, f.rdx), true),
        SYS_IOCTL => (sys_ioctl(f.rdi, f.rsi, f.rdx), true),
        // One process, and it is the guest. Reporting a pid at all is what
        // stops a runtime deciding it failed to start.
        SYS_GETPID | SYS_SET_TID_ADDRESS => (1, true),
        SYS_BRK => (sys_brk(f.rdi), true),
        SYS_MMAP => (sys_mmap(f.rdi, f.rsi, f.rdx, f.r10, f.r8, f.r9), true),
        SYS_MPROTECT => (sys_mprotect(f.rdi, f.rsi, f.rdx), true),
        SYS_MUNMAP => (sys_munmap(f.rdi, f.rsi), true),
        SYS_ARCH_PRCTL => (sys_arch_prctl(f.rdi, f.rsi), true),
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
/// Set when the guest was killed for running past its deadline.
pub const OVERRAN: u64 = 1 << 34;

/// How long a guest may run before the timer takes the machine back.
///
/// **Without this a guest that never makes a syscall owns the machine.** Every
/// fixture so far ends by asking for something, so nothing had noticed; a real
/// binary with a bug in its startup loop would simply never give the shell
/// back, and there is no key to press because the guest is what is running.
///
/// Five seconds at 100 Hz. Long enough that no correct program here meets it,
/// short enough that meeting it is an inconvenience rather than a reboot. It
/// is a stage-1 number: a guest that legitimately computes for a minute needs
/// this to be a policy rather than a constant, and there is no such guest yet.
const DEADLINE_TICKS: u64 = 500;

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

/// Set instead when the guest died of a fault.
pub const FAULTED: u64 = 1 << 33;

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
pub unsafe fn kill(vector: u64) -> ! {
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
    GUEST_RUNNING.store(true, Ordering::Relaxed);
    DEADLINE.store(crate::dev::lapic::ticks() + DEADLINE_TICKS, Ordering::Relaxed);
    let code = unsafe { glados_enter_guest(entry, stack_top) };
    DEADLINE.store(0, Ordering::Relaxed);
    GUEST_RUNNING.store(false, Ordering::Relaxed);
    if flags & (1 << 9) != 0 {
        crate::cpu::enable_interrupts();
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
        install(fake, fake, fake);
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

    // mmap's three refusals, each about this machine rather than the argument.
    {
        let fake = Region { at: 0x10_0000, len: 4096 };
        install(fake, fake, fake);
        let file_backed = sys_mmap(0, 4096, 3, MAP_ANONYMOUS, 3, 0);
        let fixed = sys_mmap(0x40_0000, 4096, 3, MAP_ANONYMOUS | MAP_FIXED, u64::MAX, 0);
        let empty = sys_mmap(0, 0, 3, MAP_ANONYMOUS, u64::MAX, 0);
        out.push(("a file-backed mapping is refused, there being no fd table", (file_backed as i64) < 0));
        out.push(("MAP_FIXED is refused, for the reason ET_EXEC is", (fixed as i64) < 0));
        out.push(("a zero-length mapping is EINVAL, as Linux has it", empty == EINVAL));

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
        install(owned, owned, owned);
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
        install(
            Region { at, len: 512 },
            Region { at, len: 512 },
            Region { at, len: 512 },
        );
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
                install(owned, owned, owned);
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
    let aux = Aux { phdr: 0x1000, phent: 56, phnum: 2, entry: 0x1078 };
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
            let mut i = 3 + n;
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
            let none = Aux { phdr: 0, phent: 56, phnum: 2, entry: 0 };
            let s2 = build_stack(small.as_mut_ptr(), 512, &["x"], none);
            s2.is_some_and(|v| unsafe {
                let p = v as *const u64;
                let n = p.read() as usize;
                let mut i = 3 + n;
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
    pub entry: u64,
}

const AT_NULL: u64 = 0;
const AT_PHDR: u64 = 3;
const AT_PHENT: u64 = 4;
const AT_PHNUM: u64 = 5;
const AT_PAGESZ: u64 = 6;
const AT_ENTRY: u64 = 9;
const AT_UID: u64 = 11;
const AT_EUID: u64 = 12;
const AT_GID: u64 = 13;
const AT_EGID: u64 = 14;
const AT_CLKTCK: u64 = 17;
const AT_SECURE: u64 = 23;
const AT_RANDOM: u64 = 25;

pub fn build_stack(base: *mut u8, size: usize, args: &[&str], aux: Aux) -> Option<u64> {
    let bottom = base as usize;
    let mut top = bottom.checked_add(size)?;

    // Strings first, at the very top, because the pointer array below has to
    // name them and nothing may move afterwards.
    let mut ptrs = alloc::vec::Vec::with_capacity(args.len());
    for a in args.iter().rev() {
        top = top.checked_sub(a.len() + 1)?;
        if top < bottom {
            return None;
        }
        unsafe {
            core::ptr::copy_nonoverlapping(a.as_ptr(), top as *mut u8, a.len());
            core::ptr::write((top + a.len()) as *mut u8, 0);
        }
        ptrs.push(top as u64);
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
    pairs.push((AT_NULL, 0));

    // argc, argv[..], its NULL, envp's NULL, then the pairs. `rsp` itself is
    // sixteen-byte aligned at entry, which is what the ABI asks of a process
    // rather than of a function -- there is no return address under it.
    //
    // A line here used to pad down when `(sp + words * 8) % 8 != 0`, which is
    // a condition that cannot hold: `sp` is sixteen-aligned and every word is
    // eight bytes. It read as an alignment fix and was a tautology, which is
    // the more expensive kind of dead code because it stops anybody looking.
    let words = 1 + ptrs.len() + 1 + 1 + pairs.len() * 2;
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
        p.add(2 + ptrs.len()).write(0); // envp terminator
        for (i, (k, v)) in pairs.iter().enumerate() {
            p.add(3 + ptrs.len() + i * 2).write(*k);
            p.add(4 + ptrs.len() + i * 2).write(*v);
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
        SYS_GETPID => "getpid",
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
    if img.dynamic() {
        return Err("dynamically linked: the entry point is ld.so, which stage 0 has no loader for");
    }
    // A fixed address is no longer refused here. Whether one can be honoured is
    // a question about this machine's memory map rather than about the file, so
    // `load` asks `mem::fixed` and reports what it said -- "that physical range
    // is not free on this machine" names the actual obstacle, where the blanket
    // refusal named a design decision that had stopped being one.
    if img.segments.is_empty() {
        return Err("nothing to load");
    }
    Ok(())
}

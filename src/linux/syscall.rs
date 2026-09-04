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
pub const SYS_WRITE: u64 = 1;
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
    if fd != 1 && fd != 2 {
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
        build_stack(small.as_mut_ptr(), 64).is_none(),
    ));
    let mut big = [0u8; 512];
    let sp = build_stack(big.as_mut_ptr(), 512);
    out.push((
        "and one large enough answers a 16-byte-aligned pointer inside itself",
        sp.is_some_and(|v| v % 16 == 0 && v >= big.as_ptr() as u64
            && v < big.as_ptr() as u64 + 512),
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
pub fn build_stack(base: *mut u8, size: usize) -> Option<u64> {
    // Five words of frame plus eight of headroom. Answering `None` rather than
    // clamping, because a stack too small to hold the frame is a caller bug
    // and writing a truncated one would put argc somewhere the guest is not
    // looking.
    let words = size / 8;
    if words < 16 {
        return None;
    }
    let at = words - 8;
    unsafe {
        let p = (base as *mut u64).add(at);
        p.write(0); // argc
        p.add(1).write(0); // argv terminator
        p.add(2).write(0); // envp terminator
        p.add(3).write(0); // AT_NULL
        p.add(4).write(0);
        Some(p as u64)
    }
}

/// Name the calls stage 0 knows about, for a trace a person has to read.
pub fn name_of(nr: u64) -> &'static str {
    match nr {
        SYS_WRITE => "write",
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
    if !img.relocatable() {
        return Err("fixed addresses: a single address space has no free range to promise it");
    }
    if img.segments.is_empty() {
        return Err("nothing to load");
    }
    Ok(())
}

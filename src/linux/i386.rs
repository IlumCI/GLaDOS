//! 32-bit (i386) Linux guests: compatibility mode, `int 0x80`, and the
//! structures whose shape changes with the width of a pointer.
//!
//! A 64-bit guest enters ring 3 on the 64-bit user code segment and asks for
//! things with `syscall`. An i386 program cannot do either: it runs on a
//! *32-bit* code segment (GDT index 5, which existed only so `sysret` had
//! something to count from) and on Intel `syscall` is undefined outside 64-bit
//! mode, so it raises `int 0x80` -- which is why glibc i386 falls back to it
//! when it is handed no vDSO, and none is handed over here.
//!
//! **One implementation of each call, two doors.** The numbers differ
//! (`write` is 4 here and 1 there) and so do the argument registers (`ebx`,
//! `ecx`, `edx`, `esi`, `edi`, `ebp`), but what `write` *does* does not, so the
//! dispatcher below maps i386 numbers onto the same `sys_*` functions the
//! 64-bit door uses. Where a structure's layout depends on pointer width --
//! `stat`, `iovec` -- the function that reads or writes it asks
//! `syscall::abi32` and produces the right bytes, so there is no second copy
//! of the logic to drift.
//!
//! Numbers are `/usr/include/asm/unistd_32.h`'s and layouts are read off the
//! host's own headers with `gcc -m32`; neither is recalled.
//!
//! **Thread-local storage is `%gs`**, set up by `set_thread_area` writing a
//! descriptor into one of three GDT slots (`gdt::TLS_FIRST`). The kernel used
//! to find its per-core block through GS, which a guest loading `%gs` would
//! have clobbered; `cpu::percpu` uses `rdpid` now, and `load` refuses a 32-bit
//! guest on a machine where it cannot. The descriptors are per task and are
//! swapped by the scheduler beside the other ring-3 state.

use super::syscall::{self as sc, Call, PATH_SNIP};
use crate::cpu::gdt;

/// Tag on a trace entry's number, so a 32-bit `write` (4) is not read as a
/// 64-bit `stat` (4) by whoever prints the trace.
pub const TAG: u64 = 1 << 31;

/// The selectors a compatibility-mode guest runs on: GDT 5 (32-bit code) and
/// 6 (data), both at RPL 3. Literals in the assembly below, asserted against
/// the constants in `checks`.
pub const USER_CS32: u64 = gdt::USER_CS32 as u64 | 3;
pub const USER_DS: u64 = gdt::USER_DS as u64 | 3;

core::arch::global_asm!(
    r#"
    .globl glados_int80_entry
glados_int80_entry:
    push r15
    push r14
    push r13
    push r12
    push r11
    push r10
    push r9
    push r8
    push rbp
    push rdi
    push rsi
    push rdx
    push rcx
    push rbx
    push rax
    sub rsp, 512
    fxsave64 [rsp]
    lea rdi, [rsp + 512]
    call glados_syscall_dispatch32
    fxrstor64 [rsp]
    add rsp, 512
    pop rax
    pop rbx
    pop rcx
    pop rdx
    pop rsi
    pop rdi
    pop rbp
    pop r8
    pop r9
    pop r10
    pop r11
    pop r12
    pop r13
    pop r14
    pop r15
    iretq

    .globl glados_enter_guest32
glados_enter_guest32:
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
    mov ax, 0x33
    mov ds, ax
    mov es, ax
    push 0x33
    push rsi
    push 0x202
    push 0x2b
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
"#
);

// **`ds` and `es` are loaded with the user data selector on the way in, and
// that is not optional in compatibility mode.** In 64-bit mode their bases are
// ignored and nothing checks them; in compatibility mode segmentation is live,
// and an `iretq` to ring 3 *nulls* any data segment register whose DPL is
// below the new CPL -- so the kernel's 0x10 left in `ds` became a null `ds`
// and the guest's first memory access a #GP. DPL 3 survives the return, and
// the kernel, in 64-bit mode, does not care what is in them.

core::arch::global_asm!(
    r#"
    .globl glados_enter_guest32_regs
glados_enter_guest32_regs:
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
    mov ax, 0x33
    mov ds, ax
    mov es, ax
    push 0x33
    push qword ptr [rdi + 8]
    push qword ptr [rdi + 16]
    push 0x2b
    push qword ptr [rdi + 0]
    mov eax, [rdi + 24]
    mov ebx, [rdi + 32]
    mov ecx, [rdi + 40]
    mov edx, [rdi + 48]
    mov esi, [rdi + 56]
    mov ebp, [rdi + 72]
    xor r8d, r8d
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    xor r12d, r12d
    xor r13d, r13d
    xor r14d, r14d
    xor r15d, r15d
    mov edi, [rdi + 64]
    iretq
"#
);

extern "sysv64" {
    fn glados_int80_entry();
    pub fn glados_enter_guest32(entry: u64, stack_top: u64) -> u64;
    /// Enter a 32-bit guest with every register set, for a cloned thread.
    pub fn glados_enter_guest32_regs(r: *const super::thread::Regs32) -> u64;
}

/// What the entry stub pushed, in the order it pushed it, followed by the
/// frame the processor itself pushed for the interrupt.
#[repr(C)]
pub struct Frame32 {
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rbp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rip: u64,
    pub cs: u64,
    pub rflags: u64,
    pub rsp: u64,
    pub ss: u64,
}

/// Install the `int 0x80` gate. Idempotent; the table is shared by every core.
pub fn arm() {
    unsafe { crate::cpu::idt::set_user_gate(0x80, glados_int80_entry as *const ()) };
}

// ---- the running task's TLS, swapped by the scheduler -------------------

/// The running task's three TLS descriptors, and whether it is a 32-bit guest
/// at all. Globals swapped per task, the pattern `GLADOS_GUEST_RSP` and its
/// neighbours use: a value only read while its own task runs *is* per task as
/// long as the scheduler swaps it.
///
/// Per core, for the reason `syscall::CoreEntry` is: swapped per task only
/// works while one core runs guests.
static mut TLS_CORE: [[u64; gdt::TLS_SLOTS]; crate::task::MAX_CPUS] = [[0; gdt::TLS_SLOTS]; crate::task::MAX_CPUS];
static mut BITS32_CORE: [bool; crate::task::MAX_CPUS] = [false; crate::task::MAX_CPUS];

fn cpu() -> usize {
    (crate::smp::this_cpu() as usize).min(crate::task::MAX_CPUS - 1)
}

/// Begin or end carrying 32-bit state on the running task.
pub fn begin(on: bool) {
    unsafe {
        BITS32_CORE[cpu()] = on;
        TLS_CORE[cpu()] = [0; gdt::TLS_SLOTS];
        gdt::set_tls(&*core::ptr::addr_of!(TLS_CORE[cpu()]));
    }
}

/// What the scheduler saves for the outgoing task.
pub fn save() -> ([u64; gdt::TLS_SLOTS], u16, bool) {
    let gs: u16;
    unsafe { core::arch::asm!("mov {0:x}, gs", out(reg) gs, options(nomem, nostack, preserves_flags)) };
    unsafe { (*core::ptr::addr_of!(TLS_CORE[cpu()]), gs, BITS32_CORE[cpu()]) }
}

/// Put the incoming task's back: its descriptors into this core's GDT, and its
/// `%gs` reloaded so the hidden base is read from them again.
///
/// A selector is reloaded only when it names one of the TLS slots and that
/// slot holds a present descriptor. Loading anything else at ring 0 is a #GP
/// in the scheduler, which is a halted machine rather than a confused guest.
pub fn load(tls: [u64; gdt::TLS_SLOTS], gs: u16, bits32: bool) {
    unsafe {
        BITS32_CORE[cpu()] = bits32;
        TLS_CORE[cpu()] = tls;
    }
    if !bits32 {
        return;
    }
    gdt::set_tls(&tls);
    let idx = (gs >> 3) as usize;
    let present = |i: usize| tls[i] & (1 << 47) != 0;
    let sel = if gs & 3 == 3
        && (gdt::TLS_FIRST..gdt::TLS_FIRST + gdt::TLS_SLOTS).contains(&idx)
        && present(idx - gdt::TLS_FIRST)
    {
        gs
    } else {
        0
    };
    unsafe {
        core::arch::asm!(
            "mov ds, {d:x}",
            "mov es, {d:x}",
            "mov gs, {g:x}",
            d = in(reg) USER_DS as u16,
            g = in(reg) sel,
            options(nostack, preserves_flags),
        )
    };
}

/// `set_thread_area`: put a TLS descriptor in one of the three slots.
///
/// `struct user_desc` is 16 bytes: entry number, base, limit, and a word of
/// bit fields -- `seg_32bit`, `contents` (2 bits), `read_exec_only`,
/// `limit_in_pages`, `seg_not_present`, `useable`. An entry number of -1 asks
/// for any free slot and is answered by writing the chosen number back, which
/// glibc then shifts into the selector it loads into `%gs`.
fn set_thread_area(at: u64) -> u64 {
    if !sc::reachable(at, 16, true) {
        return sc::EFAULT;
    }
    let rd = |o: u64| unsafe { core::ptr::read_unaligned((at + o) as *const u32) };
    let entry = rd(0) as i32;
    let (base, limit, bits) = (rd(4), rd(8), rd(12));
    let contents = (bits >> 1) & 3;
    let not_present = bits & (1 << 5) != 0;
    let pages = bits & (1 << 4) != 0;
    let tls = unsafe { &mut *core::ptr::addr_of_mut!(TLS_CORE[cpu()]) };
    let slot = if entry == -1 {
        match tls.iter().position(|d| *d == 0) {
            Some(i) => i,
            None => return (-3i64) as u64, // ESRCH, as Linux answers a full table
        }
    } else {
        let e = entry as usize;
        if !(gdt::TLS_FIRST..gdt::TLS_FIRST + gdt::TLS_SLOTS).contains(&e) {
            return sc::EINVAL;
        }
        e - gdt::TLS_FIRST
    };
    // A code segment or a stack-like expand-down one is not thread-local
    // storage, and building one would give ring 3 a segment this kernel never
    // meant to grant.
    if contents != 0 {
        return sc::EINVAL;
    }
    // The "empty" descriptor clears the slot.
    tls[slot] = if base == 0 && limit == 0 && not_present {
        0
    } else {
        gdt::tls_descriptor(base, limit, pages, !not_present)
    };
    gdt::set_tls(tls);
    if entry == -1 {
        unsafe { core::ptr::write_unaligned(at as *mut u32, (gdt::TLS_FIRST + slot) as u32) };
    }
    0
}

// ---- threads ---------------------------------------------------------------

/// i386 `clone(flags, newsp, parent_tid, tls, child_tid)` -- note the order,
/// which is not x86-64's: the TLS argument is fourth and is a pointer to a
/// `struct user_desc`, not a base address.
///
/// Threads only. A clone without `CLONE_THREAD` is a new process, and a
/// 32-bit `fork` is not here yet; `ENOSYS` is what glibc's `posix_spawn` and
/// `fork` already know how to report.
fn clone(f: &Frame32, flags: u64, newsp: u64, ptid: u64, tls_desc: u64, ctid: u64) -> u64 {
    if !super::thread::is_thread(flags) {
        return sc::ENOSYS;
    }
    if ptid != 0 && !sc::reachable(ptid, 4, true) {
        return sc::EFAULT;
    }
    if ctid != 0 && !sc::reachable(ctid, 4, true) {
        return sc::EFAULT;
    }
    // Four-byte aligned rather than sixteen: glibc i386 subtracts 28 from a
    // sixteen-aligned stack before passing it, and the child pops back to
    // alignment itself.
    if newsp == 0 || newsp % 4 != 0 || !sc::owns(newsp.saturating_sub(64), 64) {
        return sc::EINVAL;
    }
    // The child's thread-local storage: the parent's descriptors, with the
    // one `CLONE_SETTLS` names replaced by the new thread's own.
    let mut tls = unsafe { *core::ptr::addr_of!(TLS_CORE[cpu()]) };
    if flags & super::thread::CLONE_SETTLS != 0 {
        if !sc::reachable(tls_desc, 16, false) {
            return sc::EFAULT;
        }
        let rd = |o: u64| unsafe { core::ptr::read_unaligned((tls_desc + o) as *const u32) };
        let (entry, base, limit, bits) = (rd(0) as usize, rd(4), rd(8), rd(12));
        if !(gdt::TLS_FIRST..gdt::TLS_FIRST + gdt::TLS_SLOTS).contains(&entry) {
            return sc::EINVAL;
        }
        tls[entry - gdt::TLS_FIRST] = gdt::tls_descriptor(base, limit, bits & (1 << 4) != 0, bits & (1 << 5) == 0);
    }
    let (_, gs, _) = save();
    let regs = super::thread::Regs32 {
        eip: f.rip,
        esp: newsp,
        eflags: f.rflags,
        eax: 0,
        ebx: f.rbx as u32 as u64,
        ecx: newsp,
        edx: f.rdx as u32 as u64,
        esi: f.rsi as u32 as u64,
        edi: f.rdi as u32 as u64,
        ebp: f.rbp as u32 as u64,
    };
    match super::thread::spawn32(flags, ptid, ctid, regs, tls, gs) {
        Ok(tid) => tid,
        Err(e) => e,
    }
}

// ---- the dispatcher ------------------------------------------------------

/// A 32-bit argument as the signed value a 64-bit implementation expects: a
/// descriptor of -1 (`AT_FDCWD` is -100) arrives as 0xFFFF_FFFF and has to
/// stay negative.
fn sx(v: u64) -> u64 {
    v as u32 as i32 as i64 as u64
}

fn put_u32(at: u64, v: u32) -> bool {
    if !sc::reachable(at, 4, true) {
        return false;
    }
    unsafe { core::ptr::write_unaligned(at as *mut u32, v) };
    true
}

fn put_u64(at: u64, v: u64) -> bool {
    if !sc::reachable(at, 8, true) {
        return false;
    }
    unsafe { core::ptr::write_unaligned(at as *mut u64, v) };
    true
}

/// A `struct timespec` with 32-bit fields, for the calls glibc still reaches
/// when the `_time64` form is refused.
fn put_ts32(at: u64, sec: u64, nsec: u64) -> u64 {
    if !sc::reachable(at, 8, true) {
        return sc::EFAULT;
    }
    unsafe {
        core::ptr::write_unaligned(at as *mut u32, sec as u32);
        core::ptr::write_unaligned((at + 4) as *mut u32, nsec as u32);
    }
    0
}

#[no_mangle]
pub extern "sysv64" fn glados_syscall_dispatch32(f: &mut Frame32) {
    sc::big_lock();
    unsafe { sc::die_if_doomed() };
    let nr = f.rax as u32 as u64;
    let t0 = super::prof::start();
    let a = [
        f.rbx as u32 as u64,
        f.rcx as u32 as u64,
        f.rdx as u32 as u64,
        f.rsi as u32 as u64,
        f.rdi as u32 as u64,
        f.rbp as u32 as u64,
    ];
    let [b, c, d, si, di, bp] = a;
    super::prof::enter(nr | TAG, &a);
    const AT_FDCWD: u64 = super::fs::AT_FDCWD as u64;

    let (ret, served) = match nr {
        1 | 252 => {
            sc::record(Call { nr: nr | TAG, args: a, ret: 0, served: true, path: [0; PATH_SNIP], path_len: 0 });
            if nr == 252 {
                sc::exit_group_from_thread(b);
            }
            unsafe { sc::glados_leave_guest(b | sc::EXITED) }
        }
        3 => (sc::sys_read(b, c, d), true),
        4 => (sc::sys_write(b, c, d as usize), true),
        5 => (sc::sys_openat(AT_FDCWD, b, c, d), true),
        6 => (sc::sys_close(b), true),
        12 => (sc::sys_chdir(b), true),
        13 => {
            let t = sc::sys_time(0);
            if b != 0 && !put_u32(b, t as u32) {
                (sc::EFAULT, true)
            } else {
                (t, true)
            }
        }
        19 => (sc::sys_lseek(b, sx(c), d), true),
        20 => (super::fork::current_pid(), true),
        24 | 47 | 49 | 50 | 199 | 200 | 201 | 202 => (0, true),
        33 => (sc::sys_access(b, c), true),
        41 => (sc::sys_dup(b, None), true),
        42 => (sc::do_pipe(b, 0), true),
        45 => (sc::sys_brk(b), true),
        54 => (sc::sys_ioctl(b, c, d), true),
        55 | 221 => (sc::sys_fcntl(b, c, d), true),
        60 => (0o022, true),
        63 => (sc::sys_dup(b, Some(c)), true),
        64 => (super::fork::parent_pid(), true),
        78 => {
            // gettimeofday: a 32-bit timeval, and a timezone nobody reads.
            match sc::clock_pair(0) {
                Some((s, ns)) if b != 0 => (put_ts32(b, s, ns / 1000), true),
                _ => (0, true),
            }
        }
        85 => (sc::sys_readlinkat(AT_FDCWD, b, c, d), true),
        90 => {
            // The old `mmap`: its six arguments in a struct, pointed at.
            if !sc::reachable(b, 24, false) {
                (sc::EFAULT, true)
            } else {
                let w = |i: u64| unsafe { core::ptr::read_unaligned((b + i * 4) as *const u32) } as u64;
                (sc::sys_mmap(w(0), w(1), w(2), w(3), sx(w(4)), w(5)), true)
            }
        }
        91 => (sc::sys_munmap(b, c), true),
        120 => (clone(f, b, c, d, si, di), true),
        122 => (sc::sys_uname(b), true),
        125 => (sc::sys_mprotect(b, c, d), true),
        140 => {
            // _llseek(fd, high, low, result*, whence): the offset in two
            // halves and the answer through a pointer, so a 32-bit libc can
            // seek past 4 GiB.
            let r = sc::sys_lseek(b, (c << 32) | d, di);
            if (r as i64) < 0 {
                (r, true)
            } else if !put_u64(si, r) {
                (sc::EFAULT, true)
            } else {
                (0, true)
            }
        }
        145 => (sc::sys_readv(b, c, d), true),
        146 => (sc::sys_writev(b, c, d), true),
        158 => {
            sc::wait_turn();
            (0, true)
        }
        162 => {
            // nanosleep with a 32-bit timespec.
            if !sc::reachable(b, 8, false) {
                (sc::EFAULT, true)
            } else {
                let (s, ns) = unsafe {
                    (
                        core::ptr::read_unaligned(b as *const u32) as u64,
                        core::ptr::read_unaligned((b + 4) as *const u32) as u64,
                    )
                };
                (sc::sleep_for(s, ns), true)
            }
        }
        168 => (sc::sys_poll(b, c, d), true),
        172 => (0, true),
        174 => {
            // rt_sigaction: accepted, and never delivered to a 32-bit guest --
            // there is no i386 signal frame here yet. The old action reads as
            // the default, which is the truth while nothing is ever delivered.
            if d != 0 && !(put_u64(d, 0) && put_u64(d + 8, 0) && put_u32(d + 16, 0)) {
                (sc::EFAULT, true)
            } else {
                (0, true)
            }
        }
        175 => (super::signal::sigprocmask(b, c, d, si), true),
        180 => (sc::sys_pread64(b, c, d, (bp << 32) | si), true),
        183 => (sc::sys_getcwd(b, c), true),
        186 => {
            // sigaltstack: no alternate stack, reported as disabled.
            if c != 0 && !(put_u32(c, 0) && put_u32(c + 4, 2) && put_u32(c + 8, 0)) {
                (sc::EFAULT, true)
            } else {
                (0, true)
            }
        }
        191 => {
            // ugetrlimit, with a 32-bit rlimit. The stack is what the loader
            // really gave; everything else is unlimited.
            let v: u32 = match b {
                3 => 1024 * 1024,
                7 => 1024,
                _ => u32::MAX,
            };
            (if put_u32(c, v) && put_u32(c + 4, v) { 0 } else { sc::EFAULT }, true)
        }
        192 => {
            let r = sc::sys_mmap(b, c, d, si, sx(di), bp.wrapping_mul(4096));
            if si & 0x20 == 0 {
                sc::note_map(r, c, di, bp.wrapping_mul(4096));
            }
            (r, true)
        }
        195 | 196 => (sc::sys_statat(AT_FDCWD, b, c, 0), true),
        // The old ones, with the old structure: see `STAT_OLD`.
        106 | 107 | 108 => {
            sc::STAT_OLD.store(true, core::sync::atomic::Ordering::Relaxed);
            let r = if nr == 108 { sc::sys_fstat(b, c) } else { sc::sys_statat(AT_FDCWD, b, c, 0) };
            sc::STAT_OLD.store(false, core::sync::atomic::Ordering::Relaxed);
            (r, true)
        }
        197 => (sc::sys_fstat(b, c), true),
        218 => (sc::sys_mincore(b, c, d), true),
        219 => (sc::sys_madvise(b, c, d), true),
        220 => (sc::sys_getdents64(b, c, d), true),
        224 => (super::thread::current_tid(), true),
        242 => (sc::sys_sched_getaffinity(b, c, d), true),
        243 => (set_thread_area(b), true),
        258 => (super::thread::current_tid(), true),
        267 => {
            // clock_nanosleep with a 32-bit timespec; TIMER_ABSTIME (1) asks
            // for a wake at an absolute time on that clock.
            if !sc::reachable(d, 8, false) {
                (sc::EFAULT, true)
            } else {
                let (s, ns) = unsafe {
                    (
                        core::ptr::read_unaligned(d as *const u32) as u64,
                        core::ptr::read_unaligned((d + 4) as *const u32) as u64,
                    )
                };
                if c & 1 != 0 {
                    let (now_s, now_ns) = sc::clock_pair(b).unwrap_or((0, 0));
                    // u64 nanoseconds, which last 584 years: this target has no
                    // working 128-bit division, so u128 is not to be reached for.
                    let want = s.saturating_mul(1_000_000_000).saturating_add(ns);
                    let now = now_s.saturating_mul(1_000_000_000).saturating_add(now_ns);
                    let left = want.saturating_sub(now);
                    (sc::sleep_for(left / 1_000_000_000, left % 1_000_000_000), true)
                } else {
                    (sc::sleep_for(s, ns), true)
                }
            }
        }
        265 => match sc::clock_pair(b) {
            Some((s, ns)) => (put_ts32(c, s, ns), true),
            None => (sc::EINVAL, true),
        },
        295 => (sc::sys_openat(sx(b), c, d, si), true),
        300 => (sc::sys_statat(sx(b), c, d, si), true),
        305 => (sc::sys_readlinkat(sx(b), c, d, si), true),
        307 | 439 => (sc::sys_access(c, d), true),
        // set_robust_list: accepted. The list is the libc's, and this kernel
        // walks it for nobody.
        311 => (0, true),
        330 => (sc::sys_dup3(b, c, d), true),
        331 => (sc::do_pipe(b, c), true),
        340 => (sc::sys_prlimit64(b, c, d, si), true),
        355 => (sc::sys_getrandom(b, c, d), true),
        // rseq: refused, as the 64-bit door refuses it and for its reason.
        386 => (sc::ENOSYS, true),
        // Sockets: i386 has had direct calls since 4.3, which is what glibc
        // uses; the arguments are the 64-bit door's, and `msghdr` and the
        // control headers are read at the guest's width inside.
        359 => (sc::sys_socket(b, c, d), true),
        360 => (sc::sys_socketpair(b, c, d, si), true),
        361 => (sc::sys_bind(b, c, d), true),
        362 => (sc::sys_connect(b, c, d), true),
        363 => (sc::sys_listen(b), true),
        364 => (sc::sys_accept4(b, c, d, si), true),
        365 | 366 => (0, true),
        369 => (sc::sys_send(b, c, d), true),
        370 => (sc::sys_sendmsg(b, c, d), true),
        371 => (sc::sys_recv(b, c, d), true),
        372 => (sc::sys_recvmsg(b, c, d), true),
        373 => (sc::sys_shutdown(b, c), true),
        356 => (sc::sys_memfd_create(b, c), true),
        // fallocate(fd, mode, offset as two halves, len as two halves).
        324 => (sc::sys_fallocate(b, c, (si << 32) | d, (bp << 32) | di), true),
        93 => (sc::sys_ftruncate(b, c), true),
        194 => (sc::sys_ftruncate(b, (d << 32) | c), true),
        414 => (sc::sys_ppoll(b, c, d, si, di), true),
        // ppoll with a 32-bit timespec, which libwayland's dispatch reaches.
        309 => {
            if d == 0 {
                (sc::do_poll(b, c, None), true)
            } else if !sc::reachable(d, 8, false) {
                (sc::EFAULT, true)
            } else {
                let (s, ns) = unsafe {
                    (
                        core::ptr::read_unaligned(d as *const u32) as i64,
                        core::ptr::read_unaligned((d + 4) as *const u32) as i64,
                    )
                };
                (sc::do_poll(b, c, super::poll::ticks_for_ms(s * 1000 + ns / 1_000_000)), true)
            }
        }
        254 => (sc::sys_epoll_create(b), true),
        329 => (sc::sys_epoll_create1(b), true),
        255 => (sc::sys_epoll_ctl(b, c, d, si), true),
        256 => (sc::sys_epoll_wait(b, c, d, si), true),
        403 => (sc::sys_clock_gettime(b, c), true),
        // clock_nanosleep_time64: the 64-bit timespec, so the same call as
        // x86-64's, flags and all. It dropped `TIMER_ABSTIME`, which turns a
        // sleep until a time into a sleep for that many seconds.
        407 => {
            // A thread asleep is as much a question as one parked on a lock:
            // a stalled game's main thread sleeps in a loop and renders
            // nothing, and only its stack says what it is waiting out.
            super::prof::note_stack(f.rsp as u32 as u64, f.rip as u32 as u64);
            (sc::sys_clock_nanosleep(b, c, d), true)
        }
        116 => (sysinfo32(b), true),
        240 => {
            if c & 0x7f == 0 || c & 0x7f == 9 {
                super::prof::note_stack(f.rsp as u32 as u64, f.rip as u32 as u64);
            }
            (sc::futex(b, c, d, si, di, bp, true), true)
        }
        422 => (sc::futex(b, c, d, si, di, bp, false), true),
        _ => (sc::ENOSYS, false),
    };
    sc::record(Call { nr: nr | TAG, args: a, ret, served, path: [0; PATH_SNIP], path_len: 0 });
    super::prof::bill(super::prof::row(nr, true), t0);
    let t1 = super::prof::start();
    crate::sky::server::pump();
    super::prof::bill(super::prof::PUMP, t1);
    f.rax = ret as u32 as u64;
    // mmap, mmap2, munmap, mprotect, brk, mremap: anything that may have
    // removed or narrowed a translation another core is still holding.
    if matches!(nr, 90 | 192 | 91 | 125 | 45 | 163) {
        sc::tlb_shootdown();
    }
    super::prof::leave();
    sc::wake_nappers();
    sc::big_unlock();
    unsafe { sc::die_if_doomed() };
}

/// Which argument of an i386 call is a path, for the trace.
pub fn path_arg(nr: u64) -> Option<usize> {
    Some(match nr {
        5 | 12 | 33 | 85 | 195 | 196 => 0,
        295 | 300 | 305 | 307 | 383 => 1,
        _ => return None,
    })
}

/// The i386 name of a call, for the trace.
pub fn name_of(nr: u64) -> &'static str {
    match nr {
        1 => "exit",
        3 => "read",
        4 => "write",
        5 => "open",
        6 => "close",
        13 => "time",
        19 => "lseek",
        20 => "getpid",
        33 => "access",
        45 => "brk",
        54 => "ioctl",
        85 => "readlink",
        90 => "mmap",
        91 => "munmap",
        120 => "clone",
        122 => "uname",
        116 => "sysinfo",
        125 => "mprotect",
        140 => "_llseek",
        145 => "readv",
        146 => "writev",
        162 => "nanosleep",
        174 => "rt_sigaction",
        175 => "rt_sigprocmask",
        180 => "pread64",
        183 => "getcwd",
        191 => "ugetrlimit",
        192 => "mmap2",
        106 => "stat",
        107 => "lstat",
        108 => "fstat",
        195 => "stat64",
        196 => "lstat64",
        197 => "fstat64",
        199 => "getuid32",
        220 => "getdents64",
        221 => "fcntl64",
        224 => "gettid",
        240 => "futex",
        243 => "set_thread_area",
        252 => "exit_group",
        258 => "set_tid_address",
        265 => "clock_gettime",
        267 => "clock_nanosleep",
        295 => "openat",
        300 => "fstatat64",
        311 => "set_robust_list",
        340 => "prlimit64",
        355 => "getrandom",
        356 => "memfd_create",
        359 => "socket",
        362 => "connect",
        370 => "sendmsg",
        372 => "recvmsg",
        414 => "ppoll_time64",
        309 => "ppoll",
        383 => "statx",
        386 => "rseq",
        403 => "clock_gettime64",
        422 => "futex_time64",
        _ => "?",
    }
}

/// Device-free checks: the selectors, the descriptor encoding, the stub's
/// frame order.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out = alloc::vec::Vec::new();
    out.push(("the 32-bit code selector the assembly loads is GDT 5 at RPL 3", USER_CS32 == 0x2b));
    out.push(("and the data selector is GDT 6 at RPL 3", USER_DS == 0x33));
    let d = gdt::tls_descriptor(0x1234_5678, 0xF_FFFF, true, true);
    let base = ((d >> 16) & 0xFF_FFFF) | (((d >> 56) & 0xFF) << 24);
    out.push(("a TLS descriptor carries its base split the way the processor reads it", base == 0x1234_5678));
    out.push(("and is present, DPL 3, writable data", (d >> 40) & 0xFF == 0xF2));
    out.push(("and 32-bit with page granularity", (d >> 52) & 0xC == 0xC));
    out.push(("a not-present descriptor is not present", gdt::tls_descriptor(0, 0, false, false) & (1 << 47) == 0));
    out.push(("a 32-bit -1 stays -1 when widened", sx(0xFFFF_FFFF) == u64::MAX));
    out.push(("AT_FDCWD survives the widening", sx((-100i32) as u32 as u64) == super::fs::AT_FDCWD as u64));
    out.push((
        "the stub's frame puts the processor's rip just past fifteen registers",
        core::mem::offset_of!(Frame32, rip) == 15 * 8,
    ));
    out
}

/// i386 `sysinfo`: the same facts as x86-64's in 32-bit longs, 64 bytes.
///
/// glibc's `sysconf(_SC_PHYS_PAGES)` is built on this, and Zink refuses to
/// create a screen when it cannot learn how much memory the machine has --
/// so a 32-bit GL program on the GPU fell back to software over a call this
/// table did not have. `mem_unit` scales the figures when the heap would not
/// fit a 32-bit count of bytes.
fn sysinfo32(buf: u64) -> u64 {
    if !sc::reachable(buf, 64, true) {
        return sc::EFAULT;
    }
    let (used, total) = crate::mem::heap::HEAP.stats();
    let free = total.saturating_sub(used) as u64;
    let total = total as u64;
    let unit: u64 = if total > u32::MAX as u64 { 4096 } else { 1 };
    let uptime = crate::dev::lapic::ticks() / crate::TIMER_HZ as u64;
    let mut b = [0u8; 64];
    let mut put = |off: usize, v: u32| b[off..off + 4].copy_from_slice(&v.to_le_bytes());
    put(0, uptime as u32);
    put(16, (total / unit) as u32);
    put(20, (free / unit) as u32);
    put(52, unit as u32);
    b[40..42].copy_from_slice(&1u16.to_le_bytes());
    unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), buf as *mut u8, 64) };
    0
}

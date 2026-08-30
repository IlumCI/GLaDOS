//! The other cores.
//!
//! One core did everything until now. `Racy<T>` says so in its safety comment,
//! and it is right: nothing in this kernel is prepared to be entered twice at
//! once. That is a real constraint and this module does not lift it.
//!
//! What it does instead is add a **compute fabric**. The application
//! processors are started, put in long mode, handed a stack, and parked. They
//! never allocate, never take an interrupt, never print, and never touch
//! anything a `Racy` guards. They exist to be handed a range of matrix rows
//! and to say when it is done. Every decision, every allocation and every byte
//! of kernel state stays on the bootstrap processor, so `Racy`'s "one core"
//! argument stays exactly as true as it was before this file existed.
//!
//! That is the whole design, and it is deliberate. General SMP would mean
//! auditing several hundred `Racy` uses and adding a lock discipline to a
//! kernel that has never had one; a fabric that only runs arithmetic on
//! pointers the BSP hands it needs none of that, and it is where all the time
//! goes anyway -- a token is a gigabyte of weights streamed through a dot
//! product.
//!
//! ## Why the trampoline is short
//!
//! A processor comes out of INIT-SIPI in **16-bit real mode** at `vector << 12`,
//! which is why the startup code has to live in a page below 1 MiB, and why
//! `mem::frame` has refused to allocate low memory since the day it was
//! written. It has to walk itself up to long mode by hand: protected mode,
//! PAE, a page table, EFER.LME, paging, then a far jump to 64-bit code.
//!
//! The part that is usually painful is that the code runs at a low physical
//! address and the kernel lives at a high virtual one, so the instant paging
//! comes on the instruction pointer is wrong and the trampoline has to jump
//! somewhere else in the same breath. **We are identity-mapped**, so that
//! problem does not exist here: 0x8000 is 0x8000 before and after `mov cr0`,
//! and execution simply continues. The single address space earns its keep.

use crate::dev::lapic;
use core::ptr::addr_of;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Physical address the startup code is copied to.
///
/// Must be page-aligned and below 1 MiB: the SIPI carries eight bits of
/// vector and the processor starts at `vector << 12`, so the address *is* the
/// message. 0x8000 is conventional low memory that `mem::frame::MIN_PHYS`
/// guarantees the allocator will never hand to anyone else.
const TRAMPOLINE: u64 = 0x8000;
const TRAMPOLINE_PAGE: u8 = (TRAMPOLINE >> 12) as u8;

/// The asm below hardcodes the address, because a 16-bit far jump needs it as
/// an immediate and there is nowhere to put a relocation.
const _: () = assert!(TRAMPOLINE == 0x8000);

/// Per-core stack. Nothing recursive runs here; this is room for a matvec
/// kernel's frame and the fault path if one ever fires.
const AP_STACK: usize = 64 * 1024;

/// Cores that have reached Rust and enabled their own SIMD state.
static ONLINE: AtomicUsize = AtomicUsize::new(0);

/// Set once every core has its own descriptor tables, per-core block and idle
/// task, and they may all begin scheduling. See `glados_ap_main`.
static RELEASED: AtomicBool = AtomicBool::new(false);

/// Let every core that is up start taking work.
pub fn release() {
    RELEASED.store(true, Ordering::Release);
}

core::arch::global_asm!(
    r#"
.globl ap_tramp_start
.globl ap_tramp_params
.globl ap_tramp_end

.code16
ap_tramp_start:
    cli
    cld
    xorw %ax, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %ss

    movw $(0x8000 + ap_gdt_ptr - ap_tramp_start), %si
    lgdtl (%si)

    movl %cr0, %eax
    orl  $1, %eax
    movl %eax, %cr0

    /* ljmpl $0x08, $ap_prot32 -- assembled by hand, because a far jump out of
       16-bit mode with a 32-bit offset is exactly the encoding an assembler in
       .code16 is least likely to agree with us about. */
    .byte 0x66, 0xEA
    .long 0x8000 + ap_prot32 - ap_tramp_start
    .word 0x08

.code32
ap_prot32:
    movw $0x10, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %ss
    movw %ax, %fs
    movw %ax, %gs

    /* PAE. Long mode does not exist without it. */
    movl %cr4, %eax
    orl  $(1 << 5), %eax
    movl %eax, %cr4

    /* The BSP's page tables, verbatim. One address space, so there is nothing
       to build and nothing to keep in step. */
    movl $(0x8000 + ap_cr3 - ap_tramp_start), %eax
    movl (%eax), %eax
    movl %eax, %cr3

    /* EFER.LME */
    movl $0xC0000080, %ecx
    rdmsr
    orl  $(1 << 8), %eax
    wrmsr

    /* CR0.PG. Identity mapping means the next instruction is still here. */
    movl %cr0, %eax
    orl  $(1 << 31), %eax
    movl %eax, %cr0

    .byte 0xEA
    .long 0x8000 + ap_long64 - ap_tramp_start
    .word 0x18

.code64
ap_long64:
    movw $0x10, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %ss
    movw %ax, %fs
    movw %ax, %gs

    movl $(0x8000 + ap_stack - ap_tramp_start), %eax
    movq (%rax), %rsp
    movl $(0x8000 + ap_entry - ap_tramp_start), %eax
    movq (%rax), %rax
    jmp  *%rax

/* Flat descriptors: 32-bit code/data to get out of real mode, and a 64-bit
   code segment to land in. The kernel's own GDT is not used because loading it
   would mean sharing its TSS, and one TSS across cores is a corrupted
   interrupt stack the first time two of them fault together. These cores do
   not take interrupts, so flat and separate is both simpler and safer. */
ap_gdt:
    .quad 0x0000000000000000
    .quad 0x00CF9A000000FFFF
    .quad 0x00CF92000000FFFF
    .quad 0x00AF9A000000FFFF
ap_gdt_end:
ap_gdt_ptr:
    .word ap_gdt_end - ap_gdt - 1
    .long 0x8000 + ap_gdt - ap_tramp_start

/* Patched in the copy at 0x8000, never here: .text is read-only. */
ap_tramp_params:
ap_cr3:
    .quad 0
ap_entry:
    .quad 0
ap_stack:
    .quad 0
ap_tramp_end:
"#,
    options(att_syntax)
);

extern "C" {
    static ap_tramp_start: u8;
    static ap_tramp_params: u8;
    static ap_tramp_end: u8;
}

/// Where an application processor arrives, and where it stays.
///
/// Not `pub`: the only thing that may call this is a SIPI.
#[no_mangle]
extern "C" fn glados_ap_main() -> ! {
    // CR4 and XCR0 are per-core. A core that skips this handshake takes #UD on
    // the first `vmulps` no matter what the BSP enabled for itself, so every
    // AVX kernel it ran would have to be the scalar fallback -- which is most
    // of the reason this file exists.
    crate::cpu::enable_simd_this_core();

    ONLINE.fetch_add(1, Ordering::SeqCst);

    // Descriptor tables, then the interrupt table, then this core's own
    // controller. In that order: the table entries name a code selector, so
    // taking an interrupt before the descriptor table matches would enter
    // whatever happens to sit at that index in the trampoline's table.
    // Loads only. Everything these need was allocated by the bootstrap
    // processor before this core was started, because a fault here has no
    // interrupt table to land in.
    let cpu = this_cpu() as usize;
    crate::cpu::gdt::adopt(cpu);
    crate::cpu::percpu::adopt(cpu);
    crate::cpu::idt::load_this_core();
    crate::dev::lapic::init_this_core();

    // Adopt this stack as a task, so the core always has something to be
    // running and something to return to. Without it, switching into work
    // would abandon the stack this loop is standing on with nothing able to
    // resume it.
    let adopted = crate::task::adopt_idle(cpu);

    // Wait for every core to finish coming up before any of them starts
    // scheduling.
    //
    // Bringing them up one at a time while the earlier ones were already
    // taking timer interrupts and contending for the heap hung the third core
    // inside its own allocation. Whatever the precise interleaving was, a
    // core initialising beside cores that are already scheduling is a race
    // nobody needs: they are started once, at boot, and waiting costs
    // milliseconds.
    while !RELEASED.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }
    if adopted {
        crate::dev::lapic::start_timer(crate::TIMER_HZ);
        crate::task::join(cpu);
        crate::cpu::enable_interrupts();
    }

    // Interrupts stay masked: there is no per-core IDT, and the kernel's
    // handlers would print, which is a shared console. So a parked core cannot
    // be woken by one, and has to be woken by the memory it is watching.
    let sleeper = crate::cpu::has_monitor();
    let mut seen = 0usize;
    loop {
        let g = GEN.load(Ordering::Acquire);
        if g != seen {
            seen = g;
            claim_and_run();
            continue;
        }
        if sleeper {
            unsafe {
                // Arm on the generation counter, then check it *again* before
                // sleeping. Without the recheck a job published between the
                // load above and the MONITOR is a wake-up that already
                // happened, and the core sleeps through its own work.
                monitor(&GEN as *const AtomicUsize as usize);
                if GEN.load(Ordering::Acquire) == seen {
                    mwait();
                }
            }
        } else {
            core::hint::spin_loop();
        }
    }
}

/// Watch a cache line. A store to it ends the next `mwait`.
#[inline]
unsafe fn monitor(addr: usize) {
    unsafe {
        core::arch::asm!(
            "monitor",
            in("rax") addr,
            in("rcx") 0usize,
            in("rdx") 0usize,
            options(nostack, preserves_flags)
        );
    }
}

/// Stop until the monitored line is written.
///
/// Idle cores that spin are not free. Seven of them halved this machine's
/// single-threaded throughput under emulation, and on a laptop they are also
/// a flat battery and a fan that never stops -- for cores that are, by
/// definition, doing nothing. `mwait` is the instruction for exactly this and
/// it wakes on the store that publishes the job, so dispatch stays immediate.
///
/// Gated on CPUID: without MONITOR support this is #UD, and an application
/// processor has no IDT to take it on.
#[inline]
unsafe fn mwait() {
    unsafe {
        core::arch::asm!(
            "mwait",
            in("rax") 0usize,
            in("rcx") 0usize,
            options(nostack, preserves_flags)
        );
    }
}

// --- the work fabric -------------------------------------------------------
//
// One job at a time, claimed dynamically rather than divided up front.
//
// Static division would be the obvious thing and it is wrong on this laptop:
// an i5-12450H has four performance cores and four efficiency cores, and an
// equal split finishes when the slowest core finishes. Handing out small
// chunks from a shared cursor lets a P-core take three while an E-core takes
// one, which is the same total work in less wall time and needs no knowledge
// of which core is which.

/// Bumped to publish a job. A worker compares it against what it last ran.
static GEN: AtomicUsize = AtomicUsize::new(0);
/// The chunk function, as a raw pointer. Zero means no job.
static FUNC: AtomicUsize = AtomicUsize::new(0);
/// Opaque argument, handed back to `FUNC` unchanged.
static CTX: AtomicUsize = AtomicUsize::new(0);
/// Next chunk index nobody has claimed.
static CURSOR: AtomicUsize = AtomicUsize::new(0);
/// Chunks finished. The job is over when this reaches `NCHUNKS`.
static DONE: AtomicUsize = AtomicUsize::new(0);
static NCHUNKS: AtomicUsize = AtomicUsize::new(0);
static CHUNK: AtomicUsize = AtomicUsize::new(0);
static ROWS: AtomicUsize = AtomicUsize::new(0);
/// Held for the length of one job, so a second caller falls back to serial
/// rather than overwriting the slots a running job is reading.
static BUSY: AtomicBool = AtomicBool::new(false);
/// Cores currently inside `claim_and_run`.
///
/// The slots are reused by every job, so publishing the next one while a
/// worker is still reading the last one is the whole hazard. It is not
/// theoretical: a worker caches `NCHUNKS` on entry, and if the cursor is reset
/// underneath it, its next claim returns an index that is valid for the new
/// job and out of range for the cached one. It breaks out **without counting
/// that chunk**, the new job's tally can never reach its target, and the
/// bootstrap processor waits forever.
///
/// That is why this counter exists rather than a second generation check: it
/// is the only way to know that nobody is looking any more.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// What a worker runs: `(ctx, lo, hi)` over a half-open row range.
pub type ChunkFn = unsafe fn(usize, usize, usize);

fn claim_and_run() {
    ACTIVE.fetch_add(1, Ordering::AcqRel);
    claim_inner();
    ACTIVE.fetch_sub(1, Ordering::AcqRel);
}

fn claim_inner() {
    // Zero means the job is closed. A core that arrives late reads this and
    // claims nothing, which is what makes it safe to publish the next job as
    // soon as `ACTIVE` falls to zero.
    let f = FUNC.load(Ordering::Acquire);
    if f == 0 {
        return;
    }
    let ctx = CTX.load(Ordering::Acquire);
    let n = NCHUNKS.load(Ordering::Acquire);
    let chunk = CHUNK.load(Ordering::Acquire);
    let rows = ROWS.load(Ordering::Acquire);
    let func: ChunkFn = unsafe { core::mem::transmute(f) };

    loop {
        let i = CURSOR.fetch_add(1, Ordering::Relaxed);
        if i >= n {
            break;
        }
        let lo = i * chunk;
        let hi = ((i + 1) * chunk).min(rows);
        if lo < hi {
            unsafe { func(ctx, lo, hi) };
        }
        // Unconditional, including for an empty range: the bootstrap
        // processor is counting chunks, not rows, and a chunk that was
        // claimed and skipped still has to be accounted for or it waits
        // forever.
        DONE.fetch_add(1, Ordering::Release);
    }
}

/// Split `count` independent items across every core and return when all of
/// it has run.
///
/// `width` is how much work one item is, and is used only to decide whether
/// the job is worth splitting at all. What an "item" means is the caller's
/// business: a row of a forward matvec, a column of its adjoint.
///
/// Answers false without doing anything if there is nobody to help, if
/// another job is in flight, or if the work is too small to be worth the
/// handshake -- the caller then runs it serially, so this is an optimisation
/// and never a requirement.
///
/// ## Why this is sound
///
/// `ctx` outlives the call because this function does not return until every
/// chunk has completed, so a context on the caller's stack is still there for
/// as long as anyone can read it. Chunks are disjoint half-open row ranges, so
/// two cores never write the same output element. And `BUSY` means a caller
/// preempted mid-job cannot have its slots rewritten underneath it: the second
/// caller sees the flag and goes serial rather than waiting, which is also why
/// there is no lock to deadlock on.
pub fn parallel_split(ctx: usize, func: ChunkFn, count: usize, width: usize) -> bool {
    let helpers = ONLINE.load(Ordering::Relaxed);
    if helpers == 0 || count < 2 {
        return false;
    }
    // Below this the handshake costs more than the arithmetic saves. Half a
    // million multiply-accumulates is a few hundred microseconds of work,
    // comfortably more than a cross-core round trip. Every projection in a
    // 0.6B is above it -- q alone is 2048x1024 -- while a 22-row classifier
    // head and the wide-head test fixture are below it and stay on one core.
    if count.saturating_mul(width) < 1 << 19 {
        return false;
    }
    if BUSY.swap(true, Ordering::Acquire) {
        return false;
    }

    // More chunks than cores, so a slow core holds up one chunk and not one
    // eighth of the matrix.
    let want = (helpers + 1) * 4;
    let chunk = count.div_ceil(want).max(8);
    let n = count.div_ceil(chunk);

    FUNC.store(func as usize, Ordering::Relaxed);
    CTX.store(ctx, Ordering::Relaxed);
    ROWS.store(count, Ordering::Relaxed);
    CHUNK.store(chunk, Ordering::Relaxed);
    NCHUNKS.store(n, Ordering::Relaxed);
    CURSOR.store(0, Ordering::Relaxed);
    DONE.store(0, Ordering::Relaxed);
    GEN.fetch_add(1, Ordering::Release);

    // The bootstrap processor is a worker too. With one core idle and seven
    // busy this is an eighth of the machine; it also means the job completes
    // even if every AP is somehow wedged.
    claim_and_run();

    let mut spins = 0u64;
    while DONE.load(Ordering::Acquire) < n {
        core::hint::spin_loop();
        spins += 1;
        if spins > 20_000_000_000 {
            // A core claimed a chunk and never finished it. Answer **false**,
            // not true: some rows of `out` may never have been written, and
            // saying the job is done would hand back a half-computed vector
            // that looks like a model gone strange. False makes the caller run
            // the whole thing serially, overwriting everything.
            crate::kprintln!("[smp] a core did not finish its chunk -- fabric off");
            ONLINE.store(0, Ordering::SeqCst);
            FUNC.store(0, Ordering::Release);
            BUSY.store(false, Ordering::Release);
            return false;
        }
    }

    // Close the job first, so a core entering from here on claims nothing,
    // then wait for anyone still inside to leave. Only then are the slots free
    // to be written again.
    FUNC.store(0, Ordering::Release);
    while ACTIVE.load(Ordering::Acquire) != 0 {
        core::hint::spin_loop();
    }

    BUSY.store(false, Ordering::Release);
    true
}

/// Busy-wait. There is no sleeping this early and nothing else to run.
fn udelay(us: u64) {
    let mhz = crate::time::tsc_mhz();
    if mhz < 2 {
        // Uncalibrated. Spin a fixed count rather than return immediately: a
        // zero-length delay between INIT and SIPI is a core that never starts,
        // and that failure looks identical to absent hardware.
        for _ in 0..(us * 2_000) {
            core::hint::spin_loop();
        }
        return;
    }
    let target = crate::time::rdtsc() + us * mhz;
    while crate::time::rdtsc() < target {
        core::hint::spin_loop();
    }
}

fn wait_for(target: usize, ms: u64) -> bool {
    for _ in 0..(ms * 10) {
        if ONLINE.load(Ordering::SeqCst) >= target {
            return true;
        }
        udelay(100);
    }
    false
}

/// Cores answering, including the bootstrap processor.
/// Which core is executing this, as a small dense index.
///
/// The local interrupt controller's identifier is the only thing a core can
/// ask about itself without per-core storage, and it is sparse: firmware
/// numbers cores however it likes. So it is read and looked up in a table
/// built while the cores were being started.
///
/// This costs a memory-mapped read and is therefore never on a hot path. It
/// is for reports and for the deadlock message.
pub fn this_cpu() -> u32 {
    let id = crate::dev::lapic::id() as usize;
    let t = unsafe { &*LAPIC_TO_CPU.get() };
    t[id] as u32
}

/// Sparse local-controller identifier to dense index. Written only while cores
/// are being brought up, and read-only afterwards.
static LAPIC_TO_CPU: crate::sync::Racy<[u8; 256]> = crate::sync::Racy::new([0u8; 256]);

/// Record that this controller identifier is core `index`.
fn map_cpu(lapic_id: u8, index: u8) {
    unsafe { LAPIC_TO_CPU.get()[lapic_id as usize] = index };
}

pub fn online() -> usize {
    ONLINE.load(Ordering::SeqCst) + 1
}

/// Start every application processor the firmware declared.
///
/// Returns how many answered. Serialised on purpose: one shared parameter
/// block is patched per core and the next INIT does not go out until the last
/// core has picked its stack up, which costs a few milliseconds once at boot
/// and removes the only race in the bring-up path.
pub fn init(acpi: &crate::acpi::Acpi) -> usize {
    let start = addr_of!(ap_tramp_start) as u64;
    let end = addr_of!(ap_tramp_end) as u64;
    let params_off = addr_of!(ap_tramp_params) as u64 - start;
    let len = (end - start) as usize;

    unsafe {
        core::ptr::copy_nonoverlapping(start as *const u8, TRAMPOLINE as *mut u8, len);
    }

    let params = (TRAMPOLINE + params_off) as *mut u64;
    unsafe {
        params.add(0).write_volatile(crate::cpu::read_cr3());
        params.add(1).write_volatile(glados_ap_main as usize as u64);
    }

    let me = lapic::id() as u32;
    // The bootstrap processor is core 0 whatever the firmware numbered it, so
    // `this_cpu` answers something meaningful before any AP exists.
    map_cpu(me as u8, 0);
    let mut started = 0usize;

    for i in 0..acpi.cpus.min(crate::acpi::MAX_CPUS) {
        let id = acpi.apic_ids[i];
        if id == me {
            continue;
        }
        // The ICR's destination field is eight bits. An x2APIC id above 255
        // needs the x2APIC MSR interface to address at all, and answering
        // "0 cores" is better than sending INIT to whoever id & 0xFF is.
        if id > 0xFF {
            continue;
        }

        let stack = alloc::vec![0u8; AP_STACK];
        let stack = alloc::boxed::Box::leak(stack.into_boxed_slice());
        // A heap pointer is its own physical address here, so the stack needs
        // no translation before a core in 64-bit mode can load it.
        let top = (stack.as_ptr() as u64 + AP_STACK as u64) & !0xF;
        unsafe { params.add(2).write_volatile(top) };

        // Mapped before the core is started rather than after: the core can
        // reach a lock, and therefore `this_cpu`, before `init` gets another
        // turn to record anything.
        map_cpu(id as u8, (started + 1) as u8);
        let want = ONLINE.load(Ordering::SeqCst) + 1;
        lapic::send_init(id);
        udelay(10_000);
        lapic::send_sipi(id, TRAMPOLINE_PAGE);
        udelay(200);

        // The second SIPI is not superstition: the first is allowed to be lost
        // if the core was still coming out of INIT, and the Intel sequence
        // says send it again before concluding anything.
        if !wait_for(want, 20) {
            lapic::send_sipi(id, TRAMPOLINE_PAGE);
            if !wait_for(want, 100) {
                continue;
            }
        }
        started += 1;
    }

    started
}

/// The fabric against one core, bit for bit.
///
/// Not "close enough": **identical**. Splitting by rows changes nothing about
/// the arithmetic -- `out[o]` is the same dot product over the same row in the
/// same order whether one core computes all of them or eight cores compute
/// eighths -- so any difference at all means the chunking is wrong, not that
/// floating point is unlucky. A tolerance here would hide exactly the bug this
/// is looking for.
///
/// The likely mistake it catches: an int8 matrix carries four bytes of scale
/// per *row*, so a core starting at row `lo` offsets its scales by `lo * 4`
/// and its data by `lo * cols`. Confusing those two reads a scale from the
/// middle of another row, which is wrong by a factor rather than by a fault.
pub fn selftest() -> bool {
    use crate::ai::weights::Mat;
    use alloc::vec;
    use alloc::vec::Vec;

    if ONLINE.load(Ordering::Relaxed) == 0 {
        return true;
    }

    // Big enough to clear the threshold in `parallel_rows`, and not a multiple
    // of the chunk size, so the last chunk is short.
    let (rows, cols) = (1029usize, 512usize);
    let data: Vec<i8> = (0..rows * cols).map(|i| (i as u32).wrapping_mul(2654435761).to_le_bytes()[0] as i8).collect();
    let scales: Vec<u8> = (0..rows)
        .flat_map(|r| (0.5 + (r % 13) as f32 * 0.125).to_le_bytes())
        .collect();
    let x: Vec<f32> = (0..cols).map(|i| (i % 17) as f32 * 0.0625 - 0.5).collect();
    let m = Mat::Q8 { data: &data, scales: &scales, rows, cols };

    // One core, by taking the helpers away rather than by calling a different
    // function: the reference has to go down the same path, or this compares
    // two implementations instead of one implementation split two ways.
    let saved = ONLINE.swap(0, Ordering::SeqCst);
    let mut one = vec![0.0f32; rows];
    m.matvec(&mut one, &x);
    ONLINE.store(saved, Ordering::SeqCst);

    // Run it many times, not once. A single job cannot reproduce the hazard
    // that matters: the slots are shared, so the interesting case is the next
    // job being published while a core is still inside the last one. Back to
    // back is exactly what a model forward does -- every projection of every
    // layer -- and it is how the first version of this deadlocked, after
    // passing a one-shot check at boot.
    let mut many = vec![0.0f32; rows];
    for round in 0..64 {
        for v in many.iter_mut() {
            *v = f32::NAN;
        }
        m.matvec(&mut many, &x);
        if one != many {
            let bad = one.iter().zip(many.iter()).position(|(a, b)| a != b).unwrap_or(0);
            crate::kprintln!(
                "  round {} row {} -- one core {}, {} cores {}",
                round,
                bad,
                one[bad],
                saved + 1,
                many[bad]
            );
            return false;
        }
    }
    // The adjoint too, which splits by column rather than by row and is
    // therefore a different piece of index arithmetic with the same
    // consequences. It is bit-exact for the same reason: for a given column
    // the accumulation runs over the same rows in the same order whether one
    // core does every column or eight cores take a stripe each.
    let g: Vec<f32> = (0..rows).map(|i| (i % 23) as f32 * 0.03125 - 0.25).collect();
    let saved = ONLINE.swap(0, Ordering::SeqCst);
    let mut wt_one = vec![0.0f32; cols];
    m.wt_matvec(&mut wt_one, &g);
    ONLINE.store(saved, Ordering::SeqCst);

    let mut wt_many = vec![0.0f32; cols];
    for _ in 0..64 {
        for v in wt_many.iter_mut() {
            *v = f32::NAN;
        }
        m.wt_matvec(&mut wt_many, &g);
        if wt_one != wt_many {
            let bad = wt_one.iter().zip(wt_many.iter()).position(|(a, b)| a != b).unwrap_or(0);
            crate::kprintln!(
                "  adjoint col {} -- one core {}, {} cores {}",
                bad,
                wt_one[bad],
                saved + 1,
                wt_many[bad]
            );
            return false;
        }
    }

    // And both have to have actually computed something, or this passes by
    // testing the serial path twice.
    many.iter().any(|v| *v != 0.0) && wt_many.iter().any(|v| *v != 0.0)
}

/// Time the same matvec on one core and on all of them.
///
/// In the kernel, on the real machine, because nothing else can measure this
/// honestly. Under an emulator the host's scheduler decides when each virtual
/// core runs, so a "parallel" pass is really eight threads competing for
/// whatever the host has left -- the answer it gives is about the host, not
/// about this kernel. A tight loop here has no such problem.
/// Allocate and free from every core at once, and check nothing was lost.
///
/// This is the claim the allocator's lock makes, tested rather than argued.
/// Each chunk allocates a block, writes a pattern the whole way through it,
/// reads it back, and frees it. A heap that handed the same block to two
/// cores fails the read-back; one whose free list corrupted fails to satisfy
/// a later request or returns overlapping blocks.
///
/// The sizes vary per chunk on purpose. A fixed size exercises one free-list
/// bucket, and the bug this is looking for is two cores splitting or merging
/// the same block, which needs sizes that actually split and merge.
unsafe fn hammer(_ctx: usize, lo: usize, hi: usize) {
    use alloc::vec::Vec;
    for i in lo..hi {
        let n = 16 + (i % 97) * 8;
        let mut v: Vec<u8> = Vec::with_capacity(n);
        let tag = (i & 0xFF) as u8;
        for _ in 0..n {
            v.push(tag);
        }
        // Read back before dropping. Two cores handed the same allocation see
        // each other's tag here, which is the failure this exists to catch.
        if v.iter().any(|b| *b != tag) {
            BAD.fetch_add(1, Ordering::Relaxed);
        }
        RAN.fetch_add(1, Ordering::Relaxed);
    }
}

static BAD: AtomicUsize = AtomicUsize::new(0);
static RAN: AtomicUsize = AtomicUsize::new(0);

/// The concurrency claims, which need more than one core to mean anything.
pub fn selftest_mt() -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    ok &= crate::sync::selftest();

    let cores = online();
    claim(&mut ok, this_cpu() == 0, "the bootstrap processor is core 0");
    if cores == 0 {
        crate::kprintln!("  no application processors; the rest needs more than one core");
        return ok;
    }

    // Sixty-four rounds, for the reason the split matvec runs sixty-four:
    // a single pass through a lock is not evidence about a lock.
    let before = crate::mem::heap::HEAP.stats().0;
    BAD.store(0, Ordering::Relaxed);
    RAN.store(0, Ordering::Relaxed);
    let mut ran_parallel = 0;
    for _ in 0..64 {
        // Width is chosen to clear the threshold below which the split
        // declines and does the work serially, so this is actually testing
        // several cores rather than one.
        if parallel_split(0, hammer, 4096, 256) {
            ran_parallel += 1;
        } else {
            unsafe { hammer(0, 0, 4096) };
        }
    }
    let after = crate::mem::heap::HEAP.stats().0;

    claim(&mut ok, ran_parallel > 0, "the allocator was exercised from several cores");
    claim(&mut ok, RAN.load(Ordering::Relaxed) == 64 * 4096, "every chunk ran exactly once");
    claim(
        &mut ok,
        BAD.load(Ordering::Relaxed) == 0,
        "no allocation was handed to two cores at once",
    );
    // A leak here is a lost block rather than a wrong answer, and it is the
    // failure a lock protects against on the free path specifically.
    //
    // This asked for exact equality until tasks began running on more than one
    // core, at which point it started failing: the mind and the clock allocate
    // too, and they now do it *during* this measurement rather than only
    // between reschedules on one core. The assumption was never written down,
    // which is how it survived until the system stopped satisfying it.
    //
    // So the bound is on the leak rather than on the total. This churns about
    // a hundred megabytes; losing even a thousandth of it would show as
    // hundreds of kilobytes, while another task's allocations over a few
    // milliseconds are a few kilobytes at most. The gap between those two is
    // wide enough to be a real check.
    const SLACK: usize = 64 * 1024;
    let grew = after.saturating_sub(before);
    claim(
        &mut ok,
        grew < SLACK,
        "and the heap did not leak, allowing for what other cores allocated",
    );
    if grew >= SLACK {
        crate::kprintln!("         grew by {} bytes", grew);
    }
    ok
}

pub fn bench() {
    use crate::ai::weights::Mat;
    use alloc::vec;
    use alloc::vec::Vec;

    // 16 MiB of weights, which is the point: this has to miss cache the way a
    // real projection does, or it measures the L2 and not the memory bus.
    let (rows, cols) = (4096usize, 4096usize);
    let reps = 8;

    crate::kprintln!("  building {} MiB...", rows * cols / (1024 * 1024));
    let data: Vec<i8> = (0..rows * cols)
        .map(|i| (i as u32).wrapping_mul(2654435761).to_le_bytes()[0] as i8)
        .collect();
    let scales: Vec<u8> = (0..rows)
        .flat_map(|r| (0.5 + (r % 13) as f32 * 0.125).to_le_bytes())
        .collect();
    let x: Vec<f32> = (0..cols).map(|i| (i % 17) as f32 * 0.0625 - 0.5).collect();
    let m = Mat::Q8 { data: &data, scales: &scales, rows, cols };
    let mut out = vec![0.0f32; rows];

    let mhz = crate::time::tsc_mhz().max(1);

    // Every repetition gets a different input and every result is folded into
    // a checksum. Both halves matter. Eight identical calls writing the same
    // buffer are eight dead stores and one live one, and a release build is
    // entitled to notice -- a benchmark that measures an optimised-away loop
    // reports a beautiful number for no work. And the checksums let the two
    // halves be compared, which is the only thing that makes the ratio mean
    // anything: a parallel pass that skipped rows would also be very fast.
    let mut run = |x: &mut Vec<f32>, out: &mut Vec<f32>| -> (u64, u64) {
        let t0 = crate::time::rdtsc();
        let mut sum = 0u64;
        for r in 0..reps {
            x[r % cols] = 0.25 + r as f32 * 0.03125;
            m.matvec(out, x);
            // Every element, not a sample. One element per repetition would
            // be satisfied by a split pass that computed the first rows and
            // left the rest -- which is precisely the failure that would also
            // make it look fast.
            for v in out.iter() {
                sum = sum.wrapping_add(v.to_bits() as u64);
            }
        }
        ((crate::time::rdtsc() - t0) / mhz, sum)
    };

    let mut x1 = x.clone();
    let saved = ONLINE.swap(0, Ordering::SeqCst);
    let (one, sum_one) = run(&mut x1, &mut out);
    ONLINE.store(saved, Ordering::SeqCst);

    let mut x2 = x.clone();
    let (many, sum_many) = run(&mut x2, &mut out);

    if sum_one != sum_many {
        crate::kprintln!("  FAIL -- the split answer differs from the whole one");
        return;
    }

    let bytes = (rows * cols * reps) as u64;
    crate::kprintln!("  1 core    {} us   {} MB/s", one, bytes / one.max(1));
    crate::kprintln!(
        "  {} cores   {} us   {} MB/s",
        saved + 1,
        many,
        bytes / many.max(1)
    );
    if many > 0 {
        // Tenths, without floating point in a diagnostic.
        let x10 = one * 10 / many;
        crate::kprintln!("  {}.{}x", x10 / 10, x10 % 10);
    }
}

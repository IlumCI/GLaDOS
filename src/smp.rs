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
use core::sync::atomic::{AtomicUsize, Ordering};

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

    // Parked. Interrupts stay masked: there is no per-core IDT, and the
    // kernel's handlers would print, which is a shared console.
    loop {
        core::hint::spin_loop();
    }
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

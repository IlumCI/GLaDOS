//! The IDT, and the most important code in the project so far.
//!
//! The GF63 has no serial port. When this kernel runs on the real laptop, a
//! fault handler that draws to the framebuffer is the *only* way any diagnostic
//! reaches a human. Without one, every bug looks identical: the screen freezes,
//! or the machine reboots. With one, a page fault tells you the faulting
//! address and the instruction pointer, and you are debugging instead of
//! guessing.
//!
//! Every handler here is currently fatal except `#BP`. Once demand paging or
//! task switching exists, `#PF` will need to become resumable.
//!
//! Known gap: the `x86-interrupt` ABI gives us the hardware-pushed frame, but
//! the general-purpose registers are already clobbered by the compiler's
//! prologue by the time Rust code runs. Capturing those needs a naked assembly
//! stub per vector that pushes all sixteen GPRs and passes a pointer to them.
//! Worth doing, deliberately not done yet -- RIP plus CR2 diagnoses the large
//! majority of early faults.

use super::gdt::{self, IST_DOUBLE_FAULT, IST_PAGE_FAULT};
use super::{read_cr2, read_cr3};
use crate::gfx::console;
use crate::sync::Racy;
use crate::{kprintln, serial_println};
use core::arch::asm;
use core::sync::atomic::{AtomicU64, Ordering};
use core::mem::size_of;

/// What the CPU pushes on entry to an interrupt gate, in long mode.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct InterruptStackFrame {
    pub rip: u64,
    pub cs: u64,
    pub rflags: u64,
    pub rsp: u64,
    pub ss: u64,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct Entry {
    offset_low: u16,
    selector: u16,
    ist: u8,
    type_attr: u8,
    offset_mid: u16,
    offset_high: u32,
    zero: u32,
}

impl Entry {
    const fn missing() -> Self {
        Self {
            offset_low: 0,
            selector: 0,
            ist: 0,
            type_attr: 0,
            offset_mid: 0,
            offset_high: 0,
            zero: 0,
        }
    }

    fn set(&mut self, handler: u64, ist: u8) {
        self.offset_low = handler as u16;
        self.selector = gdt::KERNEL_CS;
        self.ist = ist & 0x7;
        // Present, DPL 0, 64-bit interrupt gate (0xE). Interrupt gate rather
        // than trap gate, so IF is cleared on entry and a fault handler cannot
        // itself be interrupted.
        self.type_attr = 0x8E;
        self.offset_mid = (handler >> 16) as u16;
        self.offset_high = (handler >> 32) as u32;
        self.zero = 0;
    }
}

static IDT: Racy<[Entry; 256]> = Racy::new([Entry::missing(); 256]);

/// The address the firmware loaded this kernel at, captured from the Loaded
/// Image protocol before boot services exit. RIP alone names nothing under a
/// relocated load; `rip - IMAGE_BASE` is an RVA that the build tree's
/// disassembly resolves directly.
pub static IMAGE_BASE: AtomicU64 = AtomicU64::new(0);

/// How long that image is, when the firmware would say.
///
/// Zero means it was never learned -- `LoadedImage` was unavailable and the
/// base came from scanning backwards for an MZ header, which finds where the
/// image starts and says nothing about where it ends. Without this, `fault`
/// printed `rip - IMAGE_BASE` for *any* rip, so a wild jump into the heap
/// produced a large number that looks exactly like a real offset and that a
/// disassembly resolves to an unrelated function.
pub static IMAGE_SIZE: AtomicU64 = AtomicU64::new(0);

/// Shared reporting path for every fatal exception.
fn fault(frame: &InterruptStackFrame, vector: u8, name: &str, err: Option<u64>) -> ! {
    // Copy out of the packed/borrowed frame before formatting.
    let rip = frame.rip;
    let cs = frame.cs;
    let rflags = frame.rflags;
    let rsp = frame.rsp;
    let ss = frame.ss;
    let cr2 = read_cr2();
    let cr3 = read_cr3();

    // Everything below is `kprintln`, which paints nothing while the boot
    // screen owns the framebuffer. A fault during boot would then show a
    // progress bar and no diagnostic at all -- on a machine whose only output
    // device is that screen. Take it back first.
    crate::gfx::splash::abandon();

    console::set_color(console::LTRED);
    kprintln!("\n*** EXCEPTION {:#04x}  {} ***", vector, name);

    match err {
        Some(e) if vector == 14 => {
            kprintln!("  error {:#018x}  {}", e, describe_page_fault(e));
            kprintln!("  cr2   {:#018x}   <-- faulting address", cr2);
        }
        Some(e) => kprintln!("  error {:#018x}", e),
        None => {}
    }

    kprintln!("  rip   {:#018x}   cs  {:#06x}", rip, cs);
    kprintln!("  rsp   {:#018x}   ss  {:#06x}", rsp, ss);
    kprintln!("  flags {:#018x}", rflags);
    if vector != 14 {
        kprintln!("  cr2   {:#018x}", cr2);
    }
    kprintln!("  cr3   {:#018x}", cr3);
    // The firmware relocated the kernel, so RIP names nothing by itself.
    // Relative to the load base it is an offset into the very binary in the
    // build tree, and a disassembly answers which function it is -- but only
    // if it is in the image at all, which this used to print without asking.
    use super::code::Where;
    let base = IMAGE_BASE.load(Ordering::Relaxed);
    let size = IMAGE_SIZE.load(Ordering::Relaxed);
    match super::code::locate(rip, base, size, super::code::lookup(rip)) {
        Where::Generated { tag, off } => {
            kprintln!("  in generated code {:016x} at +{:#x}", tag, off);
        }
        Where::Image(rva) => {
            kprintln!("  rva   {:#018x}   <-- rip - image base", rva);
        }
        Where::Unverified(rva) => {
            kprintln!("  rva   {:#018x}   <-- rip - image base, extent unknown", rva);
        }
        Where::Elsewhere => {
            if base != 0 && size != 0 {
                kprintln!(
                    "  rip is outside the image {:#x}..{:#x} and no generated range claims it",
                    base,
                    base + size
                );
            }
        }
    }
    kprintln!("\n  halted.");

    super::halt()
}

/// Decode the #PF error code bits into something readable at 3am.
fn describe_page_fault(e: u64) -> &'static str {
    let present = e & 1 != 0;
    let write = e & 2 != 0;
    let user = e & 4 != 0;
    let reserved = e & 8 != 0;
    let fetch = e & 16 != 0;

    if reserved {
        return "reserved bit set in a page table entry";
    }
    match (present, write, user, fetch) {
        (false, _, _, true) => "instruction fetch from unmapped page",
        (false, true, _, _) => "write to unmapped page",
        (false, false, _, _) => "read from unmapped page",
        (true, _, _, true) => "instruction fetch from no-execute page",
        (true, true, _, _) => "write to read-only page",
        (true, false, _, _) => "protection violation on read",
    }
}

macro_rules! fatal {
    ($name:ident, $vec:expr, $msg:expr) => {
        extern "x86-interrupt" fn $name(frame: InterruptStackFrame) -> ! {
            fault(&frame, $vec, $msg, None)
        }
    };
}

macro_rules! fatal_err {
    ($name:ident, $vec:expr, $msg:expr) => {
        extern "x86-interrupt" fn $name(frame: InterruptStackFrame, err: u64) -> ! {
            fault(&frame, $vec, $msg, Some(err))
        }
    };
}

fatal!(divide_error, 0, "#DE divide error");
fatal!(debug_exception, 1, "#DB debug");
fatal!(nmi, 2, "NMI");
fatal!(overflow, 4, "#OF overflow");
fatal!(bound_range, 5, "#BR bound range exceeded");
fatal!(invalid_opcode, 6, "#UD invalid opcode");
fatal!(device_not_available, 7, "#NM device not available");
fatal_err!(double_fault, 8, "#DF double fault");
fatal_err!(invalid_tss, 10, "#TS invalid TSS");
fatal_err!(segment_not_present, 11, "#NP segment not present");
fatal_err!(stack_fault, 12, "#SS stack-segment fault");
fatal_err!(general_protection, 13, "#GP general protection fault");
fatal_err!(page_fault, 14, "#PF page fault");
fatal!(x87_floating_point, 16, "#MF x87 floating point");
fatal_err!(alignment_check, 17, "#AC alignment check");
fatal!(machine_check, 18, "#MC machine check");
fatal!(simd_floating_point, 19, "#XM SIMD floating point");
fatal!(virtualization, 20, "#VE virtualization");
fatal_err!(control_protection, 21, "#CP control protection");
fatal!(reserved_vector, 15, "reserved vector");

/// `int3`. Deliberately resumable -- it is a debugging aid, not a failure.
extern "x86-interrupt" fn breakpoint(frame: InterruptStackFrame) {
    let rip = frame.rip;
    console::set_color(console::YELLOW);
    kprintln!("[brk] int3 at {:#018x}", rip);
    console::set_color(console::LTGRAY);
}

pub fn init() {
    unsafe {
        let idt = IDT.get();

        idt[0].set(divide_error as *const () as u64, 0);
        idt[1].set(debug_exception as *const () as u64, 0);
        idt[2].set(nmi as *const () as u64, 0);
        idt[3].set(breakpoint as *const () as u64, 0);
        idt[4].set(overflow as *const () as u64, 0);
        idt[5].set(bound_range as *const () as u64, 0);
        idt[6].set(invalid_opcode as *const () as u64, 0);
        idt[7].set(device_not_available as *const () as u64, 0);
        // The two that must never run on the current stack.
        idt[8].set(double_fault as *const () as u64, IST_DOUBLE_FAULT);
        idt[10].set(invalid_tss as *const () as u64, 0);
        idt[11].set(segment_not_present as *const () as u64, 0);
        idt[12].set(stack_fault as *const () as u64, 0);
        idt[13].set(general_protection as *const () as u64, 0);
        idt[14].set(page_fault as *const () as u64, IST_PAGE_FAULT);
        idt[16].set(x87_floating_point as *const () as u64, 0);
        idt[17].set(alignment_check as *const () as u64, 0);
        idt[18].set(machine_check as *const () as u64, 0);
        idt[19].set(simd_floating_point as *const () as u64, 0);
        idt[20].set(virtualization as *const () as u64, 0);
        idt[21].set(control_protection as *const () as u64, 0);

        // Vectors 9, 15, 22..=31 are reserved. Catch them rather than letting
        // a stray one become a double fault with no explanation.
        for v in [9usize, 15, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] {
            idt[v].set(reserved_vector as *const () as u64, 0);
        }

        let ptr = gdt::DescriptorTablePointer {
            limit: (size_of::<[Entry; 256]>() - 1) as u16,
            base: idt.as_ptr() as u64,
        };
        asm!("lidt [{}]", in(reg) &ptr, options(readonly, nostack, preserves_flags));
    }
    serial_println!("glados: idt installed");
}

/// Point a vector at a handler after `init` has already run.
///
/// Safe to do with the IDT live: the CPU re-reads the table on every
/// interrupt, so there is nothing cached to invalidate.
///
/// # Safety
/// `handler` must be an `extern "x86-interrupt"` function with the signature
/// the CPU will actually use for this vector -- in particular, one that takes
/// an error code if and only if the vector pushes one.
pub unsafe fn set_handler(vector: u8, handler: *const (), ist: u8) {
    unsafe { IDT.get()[vector as usize].set(handler as u64, ist) };
}

/// Deliberately trigger a page fault at address 0.
///
/// This exists to be run on purpose. Testing the fault reporter *before* you
/// are relying on it is the difference between a debugger and a rumour.
pub fn trigger_page_fault() {
    kprintln!("\n[selftest] dereferencing null on purpose...");
    unsafe {
        let p = 0x0 as *mut u64;
        core::ptr::read_volatile(p);
    }
}

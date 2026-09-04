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

/// Set once the fault reporter has begun. See `fault`.
static REPORTING: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Everything a fault report says, gathered before anything tries to print
/// any of it.
///
/// Separated from the printing because the report has to be emitted twice,
/// once per sink, and the second sink can fail. See `fault`.
struct Report<'a> {
    vector: u8,
    name: &'a str,
    err: Option<u64>,
    rip: u64,
    cs: u64,
    rsp: u64,
    ss: u64,
    rflags: u64,
    cr2: u64,
    cr3: u64,
}

/// Write one fault report to one sink, a line at a time.
fn emit(out: &mut dyn FnMut(core::fmt::Arguments), r: &Report) {
    out(format_args!("
*** EXCEPTION {:#04x}  {} ***", r.vector, r.name));

    match r.err {
        Some(e) if r.vector == 14 => {
            out(format_args!("  error {:#018x}  {}", e, describe_page_fault(e)));
            out(format_args!("  cr2   {:#018x}   <-- faulting address", r.cr2));
        }
        Some(e) => out(format_args!("  error {:#018x}", e)),
        None => {}
    }

    out(format_args!("  rip   {:#018x}   cs  {:#06x}", r.rip, r.cs));
    out(format_args!("  rsp   {:#018x}   ss  {:#06x}", r.rsp, r.ss));
    out(format_args!("  flags {:#018x}", r.rflags));
    if r.vector != 14 {
        out(format_args!("  cr2   {:#018x}", r.cr2));
    }
    out(format_args!("  cr3   {:#018x}", r.cr3));

    // The firmware relocated the kernel, so RIP names nothing by itself.
    // Relative to the load base it is an offset into the very binary in the
    // build tree, and a disassembly answers which function it is -- but only
    // if it is in the image at all, which this used to print without asking.
    use super::code::Where;
    let base = IMAGE_BASE.load(Ordering::Relaxed);
    let size = IMAGE_SIZE.load(Ordering::Relaxed);
    match super::code::locate(r.rip, base, size, super::code::lookup(r.rip)) {
        Where::Generated { tag, off } => {
            out(format_args!("  in generated code {:016x} at +{:#x}", tag, off));
        }
        Where::Image(rva) => {
            out(format_args!("  rva   {:#018x}   <-- rip - image base", rva));
        }
        Where::Unverified(rva) => {
            out(format_args!(
                "  rva   {:#018x}   <-- rip - image base, extent unknown",
                rva
            ));
        }
        Where::Elsewhere => {
            if base != 0 && size != 0 {
                out(format_args!(
                    "  rip is outside the image {:#x}..{:#x} and no generated range claims it",
                    base,
                    base + size
                ));
            }
        }
    }
    out(format_args!("
  halted."));
}

/// Shared reporting path for every fatal exception.
///
/// The report goes out **twice, whole, serial before the console** -- not
/// interleaved a line at a time -- and the ordering is the entire point.
///
/// `kprint!` writes the console first and serial second, which is right
/// everywhere but here. The console paints, and painting from inside an
/// interrupt gate takes a #GP on this kernel: measured, repeatedly, as a
/// first line followed by an unbroken column of `EXCEPTION 0x0d`. With
/// console-first, that meant *no fault this kernel has ever taken produced a
/// readable report* -- the first line died in the console before serial was
/// reached, and what a person saw was a machine that went quiet.
///
/// So serial, which is a port write and cannot block or fault, gets the whole
/// thing before the console is touched at all. Then the console is attempted
/// anyway, because on the GF63 there is no UART and the framebuffer is the
/// only diagnostic that exists -- and if it fails there, the serial copy has
/// already been written and `REPORTING` turns the failure into one line and a
/// halt instead of an endless loop.
///
/// The console #GP itself is a real bug and is not fixed here. It is older
/// than any of this and belongs to the console, not to the reporter.
fn fault(frame: &InterruptStackFrame, vector: u8, name: &str, err: Option<u64>) -> ! {
    // **A guest fault should kill only the guest, and does not yet.**
    //
    // The isolation half works and is measured: a ring-3 guest reading a
    // kernel address takes a #PF with `cs=0x3b` and `cr2` at the address it
    // reached for, so it cannot read kernel memory. What does not work is
    // surviving that. Longjmping out of this handler through
    // `syscall::kill` reaches `glados_leave_guest` and then takes a #GP at
    // ring 0 somewhere on the way back, which the report below then cannot
    // print because the console faults inside an interrupt gate here.
    //
    // Three things were ruled out on the way and are worth not re-testing:
    // the branch is reached (`cs & 3 == 3` holds and a raw port write inside
    // it emits), `running()` answers true, and reading the asm-written
    // `GLADOS_HOST_RSP` from Rust was a genuine second bug -- the optimiser
    // can fold it, since nothing in Rust writes it -- which is why that
    // question is an `AtomicBool` now and not a look at the parked stack.
    //
    // So a guest fault still stops the machine, exactly as it did at stage 0.
    // That is not a regression, and it is the one thing stage 1 was supposed
    // to buy, so it stays named here rather than quietly missing.
    let _ = crate::linux::syscall::running();

    // A program the machine wrote for itself is the thing most likely to fault
    // here, and stopping the machine for one is the wrong answer. If the task
    // is inside a guard, land there instead of reporting.
    //
    // This does not return through `iretq`. It restores a stack pointer and
    // jumps, because editing the interrupt frame means knowing whether this
    // code was handed the real frame or a copy, and being wrong there returns
    // to an address nobody chose. The pad is on the same task stack at a point
    // that was live when the guard was set, so the frame and everything above
    // it is abandoned, which also works when the fault arrived on an interrupt
    // stack as the page-fault vector does.
    if let Some((rsp, rbp, rip)) = super::recover::take(vector) {
        unsafe {
            core::arch::asm!(
                "mov rsp, {rsp}",
                "mov rbp, {rbp}",
                "jmp {rip}",
                rsp = in(reg) rsp,
                rbp = in(reg) rbp,
                rip = in(reg) rip,
                options(noreturn),
            );
        }
    }

    // A fault taken *while reporting* one used to recurse: the report crashed
    // partway through, its own handler started another report, and that
    // crashed in the same place. One line and a halt is worth more than an
    // infinite number of identical ones, and the first report is the one that
    // says something.
    if REPORTING.swap(true, Ordering::Relaxed) {
        crate::serial_println!(
            "
*** {:#04x} {} while reporting a fault -- halting ***",
            vector,
            name
        );
        super::halt()
    }

    // Copy out of the packed/borrowed frame before formatting.
    let r = Report {
        vector,
        name,
        err,
        rip: frame.rip,
        cs: frame.cs,
        rsp: frame.rsp,
        ss: frame.ss,
        rflags: frame.rflags,
        cr2: read_cr2(),
        cr3: read_cr3(),
    };

    emit(&mut |a| crate::serial::_print(format_args!("{}
", a)), &r);

    // Now the framebuffer. `kprintln` paints nothing while the boot screen
    // owns it, so a fault during boot would show a progress bar and no
    // diagnostic at all -- on a machine whose only output device is that
    // screen. Take it back first, and stop pacing: 1200us a character turns a
    // report into half a second of typewriter, which to somebody watching is
    // indistinguishable from the hang it is explaining.
    crate::gfx::splash::abandon();
    console::set_pace(0);
    console::set_color(console::LTRED);
    emit(&mut |a| crate::gfx::console::_print(format_args!("{}
", a)), &r);

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

/// Point this core at the table `init` already built.
///
/// The table is shared and that is correct: a handler is code, and every core
/// wants the same handlers. What must be per-core is the task-state segment
/// the entries' IST indices resolve against, which `gdt::init_this_core`
/// provides.
pub fn load_this_core() {
    unsafe {
        let idt = IDT.get();
        let ptr = gdt::DescriptorTablePointer {
            limit: (size_of::<[Entry; 256]>() - 1) as u16,
            base: idt.as_ptr() as u64,
        };
        asm!("lidt [{}]", in(reg) &ptr, options(readonly, nostack, preserves_flags));
    }
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

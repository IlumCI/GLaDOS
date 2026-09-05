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
//! Every fault arrives through an assembly stub that pushes all fifteen
//! general-purpose registers first, so a report says what the machine was
//! holding and not only where it was. That was a named gap here for a long
//! time, on the reasoning that RIP plus CR2 diagnoses most early faults --
//! true, and the ones it does not diagnose are the expensive ones. The last
//! of them needed a disassembly of somebody else's dynamic linker to learn
//! that `rdi` was zero.

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
    f: &'a Frame,
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
            // **The translation, because a page fault report without it is
            // missing the only thing that explains a reserved-bit fault.**
            // Two hypotheses about one such fault were measured and both were
            // wrong, and the entry was never read because nothing printed it.
            // Four numbers, from the tables the faulting core was actually
            // using, which is the point: an audit run afterwards on another
            // core is a different question.
            let (w, n) = crate::mem::paging::walk(r.cr3, r.cr2);
            for (i, e) in w[..n].iter().enumerate() {
                out(format_args!("  {}  {:#018x}", ["pml4", "pdpt", "  pd", "  pt"][i], e));
            }
        }
        Some(e) => out(format_args!("  error {:#018x}", e)),
        None => {}
    }

    out(format_args!("  rip   {:#018x}   cs  {:#06x}", r.f.rip, r.f.cs));
    out(format_args!("  rsp   {:#018x}   ss  {:#06x}", r.f.rsp, r.f.ss));
    out(format_args!("  flags {:#018x}", r.f.rflags));
    // Three to a line, in the order a disassembly names them rather than the
    // order they were pushed, because the reader is holding an instruction
    // and asking what its operands were.
    let g = r.f;
    out(format_args!("  rax {:#018x}  rbx {:#018x}  rcx {:#018x}", g.rax, g.rbx, g.rcx));
    out(format_args!("  rdx {:#018x}  rsi {:#018x}  rdi {:#018x}", g.rdx, g.rsi, g.rdi));
    out(format_args!("  rbp {:#018x}  r8  {:#018x}  r9  {:#018x}", g.rbp, g.r8, g.r9));
    out(format_args!("  r10 {:#018x}  r11 {:#018x}  r12 {:#018x}", g.r10, g.r11, g.r12));
    out(format_args!("  r13 {:#018x}  r14 {:#018x}  r15 {:#018x}", g.r13, g.r14, g.r15));
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
    match super::code::locate(r.f.rip, base, size, super::code::lookup(r.f.rip)) {
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

/// Every register a fault can report, in the order the stub pushes them.
///
/// **This closes the gap named at the top of this file.** The
/// `extern "x86-interrupt"` ABI hands Rust the hardware frame and nothing
/// else: by the time the body runs, the compiler's prologue has been over the
/// general-purpose registers, so a report could say *where* a fault happened
/// and never *with what*. That has cost this tree real time -- the last one
/// took a disassembly of somebody else's dynamic linker to learn that `rdi`
/// was zero, which is a number the processor was holding the whole time.
///
/// The field order is the push order reversed, because a push moves down: the
/// last register pushed sits at the lowest address and so comes first here.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Frame {
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub r11: u64,
    pub r10: u64,
    pub r9: u64,
    pub r8: u64,
    pub rbp: u64,
    pub rdi: u64,
    pub rsi: u64,
    pub rdx: u64,
    pub rcx: u64,
    pub rbx: u64,
    pub rax: u64,
    /// Pushed by the stub, because the CPU does not say which vector it is.
    pub vector: u64,
    /// The CPU's error code, or a zero the stub pushed so that both shapes of
    /// vector produce one layout. `pushes_error` says which of the two it is,
    /// and printing a fabricated zero as though the CPU meant it is exactly
    /// the kind of confident wrong number this file exists to avoid.
    pub err: u64,
    pub rip: u64,
    pub cs: u64,
    pub rflags: u64,
    pub rsp: u64,
    pub ss: u64,
}

/// Which vectors push an error code. The rest get a zero from the stub.
fn pushes_error(v: u64) -> bool {
    matches!(v, 8 | 10 | 11 | 12 | 13 | 14 | 17 | 21 | 29 | 30)
}

fn vector_name(v: u64) -> &'static str {
    match v {
        0 => "#DE divide error",
        1 => "#DB debug",
        2 => "NMI",
        3 => "#BP breakpoint",
        4 => "#OF overflow",
        5 => "#BR bound range exceeded",
        6 => "#UD invalid opcode",
        7 => "#NM device not available",
        8 => "#DF double fault",
        10 => "#TS invalid TSS",
        11 => "#NP segment not present",
        12 => "#SS stack-segment fault",
        13 => "#GP general protection fault",
        14 => "#PF page fault",
        16 => "#MF x87 floating point",
        17 => "#AC alignment check",
        18 => "#MC machine check",
        19 => "#XM SIMD floating point",
        20 => "#VE virtualization",
        21 => "#CP control protection",
        _ => "reserved vector",
    }
}

/// How far apart the stubs are. Every one is padded to this, so the handler
/// for vector *v* is `glados_fault_stubs + v * STUB_STRIDE` and the table
/// needs no thirty-two symbols.
pub const STUB_STRIDE: u64 = 16;

// One stub per vector, each pushing a fake error code where the CPU pushes
// none so that a single tail can serve both shapes, then the vector number,
// then jumping to the common tail.
//
// **The alignment is load-bearing and is not obvious.** The CPU aligns the
// stack to sixteen before pushing, then pushes five words without an error
// code or six with one -- so the two shapes arrive eight bytes out of phase.
// The fake push puts them back in phase, the vector push and fifteen register
// pushes come to 128 bytes, and the tail therefore calls with `rsp` sixteen-
// aligned, which is what the System V ABI wants at a call. Getting this wrong
// gives a `movaps` fault inside the reporter, which this tree has already
// paid for once on a different path.
core::arch::global_asm!(
    r#"
.macro FSTUB vec, haserr
    .balign 16
    .if \haserr == 0
    push 0
    .endif
    push \vec
    jmp glados_fault_common
.endm

.globl glados_fault_stubs
.balign 16
glados_fault_stubs:
FSTUB 0, 0
FSTUB 1, 0
FSTUB 2, 0
FSTUB 3, 0
FSTUB 4, 0
FSTUB 5, 0
FSTUB 6, 0
FSTUB 7, 0
FSTUB 8, 1
FSTUB 9, 0
FSTUB 10, 1
FSTUB 11, 1
FSTUB 12, 1
FSTUB 13, 1
FSTUB 14, 1
FSTUB 15, 0
FSTUB 16, 0
FSTUB 17, 1
FSTUB 18, 0
FSTUB 19, 0
FSTUB 20, 0
FSTUB 21, 1
FSTUB 22, 0
FSTUB 23, 0
FSTUB 24, 0
FSTUB 25, 0
FSTUB 26, 0
FSTUB 27, 0
FSTUB 28, 0
FSTUB 29, 1
FSTUB 30, 1
FSTUB 31, 0

glados_fault_common:
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    cld
    mov rdi, rsp
    call glados_fault_entry
    ud2
"#
);

extern "C" {
    fn glados_fault_stubs();
}

/// Where the stub table starts, for the claim that checks its stride.
pub fn stub_base() -> u64 {
    glados_fault_stubs as *const () as u64
}

/// Where every fault arrives now.
///
/// `sysv64` and not the default: this target is Windows-ABI, so an ordinary
/// `extern "C"` would expect its argument in `rcx`, and the stub puts it in
/// `rdi`. That mistake is silent -- the report would name a frame made of
/// whatever was in `rcx` -- and it is the third time this tree has had to be
/// careful about exactly this.
///
/// # Safety
/// Called only from `glados_fault_common`, with `rdi` holding the stack
/// pointer at which that stub finished pushing.
#[no_mangle]
pub unsafe extern "sysv64" fn glados_fault_entry(f: *const Frame) -> ! {
    // The pointer is the stub's own stack pointer, so what it names outlives
    // every use below: nothing here returns.
    fault(unsafe { &*f })
}

/// The console #GP itself is a real bug and is not fixed here. It is older
/// than any of this and belongs to the console, not to the reporter.
fn fault(f: &Frame) -> ! {
    let vector = f.vector as u8;
    let name = vector_name(f.vector);
    let err = pushes_error(f.vector).then_some(f.err);
    // **A guest fault kills only the guest, and this is where.** That was
    // written here as not working for a while and it works now: a real
    // dynamically linked binary faulted at ring 3 and the shell printed its
    // trace afterwards, so the longjmp through `glados_leave_guest` returns to
    // whoever started the guest with the machine intact.
    //
    // Two things ruled out on the way and worth not re-testing: reading the
    // asm-written `GLADOS_HOST_RSP` from Rust was a genuine second bug, since
    // the optimiser may fold a `static mut` nothing in Rust writes, which is
    // why `running()` is an `AtomicBool` rather than a look at the parked
    // stack; and the branch is reached, which was checked with a raw port
    // write when the console could not be trusted from inside a gate.
    //
    // Everything the report needs is copied out *here*, because the longjmp
    // abandons this stack. A vector alone is the same line for a null
    // dereference, a stack overflow and a jump into nothing.
    if f.cs & 3 == 3 && crate::linux::syscall::running() {
        unsafe {
            crate::linux::syscall::kill(crate::linux::syscall::Fault {
                regs: *f,
                cr2: read_cr2(),
            })
        }
    }

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
    if let Some(pad) = super::recover::take(vector) {
        // Restores **every** general-purpose register, not a set chosen from
        // a calling convention: `guard` is inlined into its callers, so the
        // longjmp crosses no ABI boundary and a caller's live value can be
        // sitting in `rax` or `r9` as easily as in `rbx`.
        //
        // `rcx` is the cursor and is therefore restored last, from its own
        // slot through itself. That leaves nothing to hold the jump target, so
        // the target is pushed onto the already-restored stack and `ret` takes
        // it: eight bytes below `rsp`, which nothing owns, written with
        // interrupts still off because the gate cleared them and `guard` does
        // not `sti` until it has landed.
        //
        // The offsets are asserted against the structure in `recover.rs`.
        unsafe {
            core::arch::asm!(
                "mov rax, [rcx]",
                "mov rbx, [rcx + 8]",
                "mov rdx, [rcx + 24]",
                "mov rsi, [rcx + 32]",
                "mov rdi, [rcx + 40]",
                "mov rbp, [rcx + 48]",
                "mov r8,  [rcx + 56]",
                "mov r9,  [rcx + 64]",
                "mov r10, [rcx + 72]",
                "mov r11, [rcx + 80]",
                "mov r12, [rcx + 88]",
                "mov r13, [rcx + 96]",
                "mov r14, [rcx + 104]",
                "mov r15, [rcx + 112]",
                "mov rsp, [rcx + 128]",
                "push [rcx + 120]",
                "mov rcx, [rcx + 16]",
                "ret",
                in("rcx") pad,
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
    let r = Report { vector, name, err, f, cr2: read_cr2(), cr3: read_cr3() };

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

        // Every vector through its own stub, at a fixed stride from one
        // symbol. Thirty-two `set` lines naming thirty-two functions was a
        // list that had to agree with another list; this is arithmetic, and
        // `recover::selftest` checks the stride is what it says.
        let stubs = glados_fault_stubs as *const () as u64;
        for v in 0..32usize {
            let ist = match v {
                // The two that must never run on the current stack.
                8 => IST_DOUBLE_FAULT,
                14 => IST_PAGE_FAULT,
                _ => 0,
            };
            idt[v].set(stubs + v as u64 * STUB_STRIDE, ist);
        }
        // `int3` is the one that is not a failure, so it keeps a handler that
        // returns rather than one that halts.
        idt[3].set(breakpoint as *const () as u64, 0);

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

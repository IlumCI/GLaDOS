//! Our own GDT and TSS.
//!
//! UEFI left us a perfectly working GDT, but it belongs to firmware memory we
//! are about to reclaim, so we install our own before anything else.
//!
//! In long mode the segment descriptors are nearly vestigial -- the base and
//! limit are ignored for code and data. What we actually need out of this is
//! the TSS, and specifically its IST entries: a stack overflow faults on the
//! guard page, and if the fault handler tries to run on that same broken stack
//! it faults again and the CPU triple-faults, which on real hardware is an
//! instant silent reboot. IST gives those handlers a known-good stack so they
//! live long enough to print something.

use crate::sync::Racy;
use core::arch::asm;
use core::mem::size_of;

pub const KERNEL_CS: u16 = 0x08;
pub const KERNEL_DS: u16 = 0x10;
pub const TSS_SEL: u16 = 0x18;

/// The three ring-3 descriptors, and **their order is dictated by `sysret`
/// rather than chosen.**
///
/// `sysret` does not take selectors. It computes them from `IA32_STAR[63:48]`:
/// the stack selector is that value plus eight and the 64-bit code selector is
/// that value plus sixteen, both with RPL forced to 3. So the base slot has to
/// be a 32-bit code descriptor nobody uses, followed by user data, followed by
/// user code, in exactly that sequence. Reordering them to something tidier
/// gives a `sysret` that lands in a data segment.
///
/// They sit after the TSS because the TSS descriptor is sixteen bytes and
/// occupies two slots, and moving it would re-address `TSS_SEL` in two tables
/// that have to agree.
pub const SYSRET_BASE: u16 = 0x28;
/// Unused by anything: it exists because `sysret` counts from it.
pub const USER_CS32: u16 = 0x28;
pub const USER_DS: u16 = 0x30;
pub const USER_CS: u16 = 0x38;

/// A selector as ring 3 must see it. The requested privilege level is part of
/// the selector, so a descriptor at DPL 3 loaded with RPL 0 still faults.
pub const fn ring3(sel: u16) -> u16 {
    sel | 3
}

/// IST slot (1-based, as the IDT encodes it) for the double-fault handler.
pub const IST_DOUBLE_FAULT: u8 = 1;
/// IST slot for the page-fault handler, so a stack overflow is still reportable.
pub const IST_PAGE_FAULT: u8 = 2;

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct DescriptorTablePointer {
    pub limit: u16,
    pub base: u64,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct Tss {
    reserved0: u32,
    /// Stack pointers for CPL 0/1/2.
    ///
    /// `rsp[0]` is loaded by the processor when an interrupt arrives from a
    /// less privileged ring, so it is what a guest running at ring 3 depends
    /// on. It said "unused: we never leave ring 0" for a long time and that
    /// stopped being true.
    pub rsp: [u64; 3],
    reserved1: u64,
    /// The seven interrupt stack table entries.
    pub ist: [u64; 7],
    reserved2: u64,
    reserved3: u16,
    pub iomap_base: u16,
}

impl Tss {
    pub const fn new() -> Self {
        Self {
            reserved0: 0,
            rsp: [0; 3],
            reserved1: 0,
            ist: [0; 7],
            reserved2: 0,
            reserved3: 0,
            // Pointing the I/O map base past the end of the TSS means "no
            // bitmap", which is what we want at ring 0.
            iomap_base: size_of::<Tss>() as u16,
        }
    }
}

const STACK_SIZE: usize = 16 * 1024;

#[repr(C, align(16))]
struct Stack([u8; STACK_SIZE]);

static DF_STACK: Racy<Stack> = Racy::new(Stack([0; STACK_SIZE]));
static PF_STACK: Racy<Stack> = Racy::new(Stack([0; STACK_SIZE]));

static TSS: Racy<Tss> = Racy::new(Tss::new());
// 0: null, 1: code, 2: data, 3+4: TSS (system descriptors are 16 bytes),
// 5: user code32, 6: user data, 7: user code64. The last three are in that
// order because `sysret` derives its selectors by adding to SYSRET_BASE.
static GDT: Racy<[u64; 8]> = Racy::new([0; 8]);

/// Long-mode ring-0 code: present, DPL 0, code segment, L bit set.
const DESC_KERNEL_CODE: u64 = 0x00AF_9B00_0000_FFFF;
/// Long-mode ring-0 data: present, DPL 0, data segment, writable.
const DESC_KERNEL_DATA: u64 = 0x00CF_9300_0000_FFFF;
/// Compatibility-mode ring-3 code. Present only so `sysret` has something to
/// count from; nothing ever loads it.
const DESC_USER_CODE32: u64 = 0x00CF_FA00_0000_FFFF;
/// Ring-3 data: the same as the kernel's but DPL 3 (access 0x93 -> 0xF3).
const DESC_USER_DATA: u64 = 0x00CF_F300_0000_FFFF;
/// Long-mode ring-3 code: the same as the kernel's but DPL 3 (0x9B -> 0xFB).
const DESC_USER_CODE64: u64 = 0x00AF_FB00_0000_FFFF;

/// The stack the processor switches to when an interrupt arrives from ring 3.
///
/// Unlike the IST stacks this one is not for surviving a broken stack. It is
/// the ordinary kernel stack for a ring-3 entry, and without it an interrupt
/// taken in a guest would push its frame onto the guest's own stack, which is
/// the guest's to corrupt.
static RSP0_STACK: Racy<Stack> = Racy::new(Stack([0; STACK_SIZE]));

fn stack_top(s: &Stack) -> u64 {
    let base = s.0.as_ptr() as u64;
    // Stacks grow down, so hand out the far end, 16-byte aligned.
    (base + STACK_SIZE as u64) & !0xF
}

/// What `diag gdt` asks of the table, without loading anything.
///
/// Every claim here is about a bit field, and every one of them is a silent
/// triple fault if it is wrong: a descriptor at the wrong DPL, or the three
/// ring-3 slots in the wrong order, produces a machine that reboots with no
/// message the first time anything selects them.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out = alloc::vec::Vec::new();
    // The access byte lives at bits 40..48. DPL is bits 45..47.
    let dpl = |d: u64| ((d >> 45) & 3) as u8;
    let present = |d: u64| (d >> 47) & 1 == 1;
    let long = |d: u64| (d >> 53) & 1 == 1;

    out.push((
        "the kernel descriptors are DPL 0 and the user ones DPL 3",
        dpl(DESC_KERNEL_CODE) == 0
            && dpl(DESC_KERNEL_DATA) == 0
            && dpl(DESC_USER_CODE64) == 3
            && dpl(DESC_USER_DATA) == 3
            && dpl(DESC_USER_CODE32) == 3,
    ));
    out.push((
        "every descriptor is present, and only the 64-bit code ones set L",
        present(DESC_KERNEL_CODE)
            && present(DESC_USER_CODE64)
            && present(DESC_USER_DATA)
            && long(DESC_KERNEL_CODE)
            && long(DESC_USER_CODE64)
            && !long(DESC_USER_DATA)
            && !long(DESC_USER_CODE32),
    ));
    // The one that would be a silent reboot. `sysret` adds 8 for SS and 16 for
    // CS, so the three slots have to sit in that order and nowhere else.
    out.push((
        "sysret's arithmetic lands on the user data and code selectors",
        SYSRET_BASE + 8 == USER_DS && SYSRET_BASE + 16 == USER_CS,
    ));
    out.push((
        "and those selectors index the slots the table actually fills",
        USER_CS32 as usize / 8 == 5 && USER_DS as usize / 8 == 6 && USER_CS as usize / 8 == 7,
    ));
    out.push((
        "a ring-3 selector carries RPL 3, since a DPL-3 descriptor at RPL 0 still faults",
        ring3(USER_CS) == 0x3B && ring3(USER_DS) == 0x33 && ring3(KERNEL_CS) == 0x0B,
    ));
    out.push((
        "the ring-3 entry stack is set and 16-byte aligned",
        unsafe {
            let r = TSS.get().rsp[0];
            r != 0 && r % 16 == 0
        },
    ));
    out
}

/// Build the two halves of a 16-byte 64-bit TSS descriptor.
fn tss_descriptor(base: u64) -> (u64, u64) {
    let limit = (size_of::<Tss>() - 1) as u64;
    let mut low = limit & 0xFFFF;
    low |= (base & 0x00FF_FFFF) << 16;
    low |= 0x89 << 40; // present, DPL 0, type 9 = available 64-bit TSS
    low |= ((limit >> 16) & 0xF) << 48;
    low |= ((base >> 24) & 0xFF) << 56;
    (low, base >> 32)
}

/// One core's tables, built by the bootstrap processor and loaded by the core
/// itself.
struct CoreTables {
    gdt: [u64; 8],
    tss: Tss,
}

/// Room for the same number of cores the scheduler knows about.
const MAX_CPUS: usize = crate::task::MAX_CPUS;

static TABLES: Racy<[*mut CoreTables; MAX_CPUS]> = Racy::new([core::ptr::null_mut(); MAX_CPUS]);

/// Build tables for every core, on the bootstrap processor, before any of them
/// starts.
///
/// **Nothing here may run on an application processor**, and that is the whole
/// reason this is split from `adopt`. A core walks itself up from the
/// trampoline with no interrupt table of its own, so between arriving and
/// `idt::load_this_core` it has no handler for anything: a fault in that
/// window is a triple fault and the core is simply gone. Allocating there was
/// exactly that mistake, and it presented as the third core dying inside a
/// 16 KiB allocation while the first two came up, with the hypervisor
/// reporting an unexpected exit and nothing on the console at all.
///
/// So every allocation happens here, on a core that can report a fault, and
/// what the application processor does is load registers.
pub fn prepare(cores: usize) {
    use alloc::boxed::Box;
    for cpu in 0..cores.min(MAX_CPUS) {
        // The interrupt stacks are allocated as slices rather than as a large
        // value moved into a box, because the latter materialises sixteen
        // kilobytes on the caller's stack before it ever reaches the heap.
        let df = Box::leak(alloc::vec![0u8; STACK_SIZE].into_boxed_slice());
        let pf = Box::leak(alloc::vec![0u8; STACK_SIZE].into_boxed_slice());

        let t = Box::leak(Box::new(CoreTables { gdt: [0; 8], tss: Tss::new() }));
        t.tss.ist[(IST_DOUBLE_FAULT - 1) as usize] =
            (df.as_ptr() as u64 + STACK_SIZE as u64) & !0xF;
        t.tss.ist[(IST_PAGE_FAULT - 1) as usize] =
            (pf.as_ptr() as u64 + STACK_SIZE as u64) & !0xF;

        t.gdt[0] = 0;
        t.gdt[1] = DESC_KERNEL_CODE;
        t.gdt[2] = DESC_KERNEL_DATA;
        let (lo, hi) = tss_descriptor(&t.tss as *const Tss as u64);
        t.gdt[3] = lo;
        t.gdt[4] = hi;
        t.gdt[5] = DESC_USER_CODE32;
        t.gdt[6] = DESC_USER_DATA;
        t.gdt[7] = DESC_USER_CODE64;

        // Its own ring-3 entry stack, for the same reason it has its own IST
        // stacks: two cores sharing one would interleave their frames.
        let r0 = Box::leak(alloc::vec![0u8; STACK_SIZE].into_boxed_slice());
        t.tss.rsp[0] = (r0.as_ptr() as u64 + STACK_SIZE as u64) & !0xF;

        unsafe { TABLES.get()[cpu] = t as *mut CoreTables };
    }
}

/// Load the tables `prepare` built for this core. Allocates nothing.
///
/// The layout matches `init` exactly, selector for selector. That is not
/// tidiness: the interrupt descriptor table is shared between all cores and
/// its entries name a code selector, so a core whose table puts code anywhere
/// else takes its first interrupt into whatever happens to be at that index.
pub fn adopt(cpu: usize) -> bool {
    let t = unsafe { TABLES.get().get(cpu).copied().unwrap_or(core::ptr::null_mut()) };
    if t.is_null() {
        return false;
    }
    unsafe {
        let ptr = DescriptorTablePointer {
            limit: (size_of::<[u64; 5]>() - 1) as u16,
            base: (*t).gdt.as_ptr() as u64,
        };
        asm!("lgdt [{}]", in(reg) &ptr, options(readonly, nostack, preserves_flags));

        asm!(
            "push {sel}",
            "lea {tmp}, [rip + 2f]",
            "push {tmp}",
            "retfq",
            "2:",
            sel = in(reg) KERNEL_CS as u64,
            tmp = lateout(reg) _,
            options(preserves_flags),
        );

        asm!(
            "mov ds, {0:x}",
            "mov es, {0:x}",
            "mov ss, {0:x}",
            "mov fs, {0:x}",
            "mov gs, {0:x}",
            in(reg) KERNEL_DS,
            options(nostack, preserves_flags),
        );

        asm!("ltr {0:x}", in(reg) TSS_SEL, options(nostack, preserves_flags));
    }
    true
}

/// Install the GDT, reload every segment register, and load the task register.
pub fn init() {
    unsafe {
        let tss = TSS.get();
        tss.ist[(IST_DOUBLE_FAULT - 1) as usize] = stack_top(DF_STACK.get());
        tss.ist[(IST_PAGE_FAULT - 1) as usize] = stack_top(PF_STACK.get());

        let gdt = GDT.get();
        gdt[0] = 0;
        gdt[1] = DESC_KERNEL_CODE;
        gdt[2] = DESC_KERNEL_DATA;
        let (lo, hi) = tss_descriptor(tss as *const Tss as u64);
        gdt[3] = lo;
        gdt[4] = hi;
        gdt[5] = DESC_USER_CODE32;
        gdt[6] = DESC_USER_DATA;
        gdt[7] = DESC_USER_CODE64;

        tss.rsp[0] = stack_top(RSP0_STACK.get());

        let ptr = DescriptorTablePointer {
            limit: (size_of::<[u64; 8]>() - 1) as u16,
            base: gdt.as_ptr() as u64,
        };
        asm!("lgdt [{}]", in(reg) &ptr, options(readonly, nostack, preserves_flags));

        // CS cannot be loaded with a plain mov. Push the new selector and a
        // return address, then far-return into it.
        asm!(
            "push {sel}",
            "lea {tmp}, [rip + 2f]",
            "push {tmp}",
            "retfq",
            "2:",
            sel = in(reg) KERNEL_CS as u64,
            tmp = lateout(reg) _,
            options(preserves_flags),
        );

        asm!(
            "mov ds, {0:x}",
            "mov es, {0:x}",
            "mov ss, {0:x}",
            "mov fs, {0:x}",
            "mov gs, {0:x}",
            in(reg) KERNEL_DS,
            options(nostack, preserves_flags),
        );

        asm!("ltr {0:x}", in(reg) TSS_SEL, options(nostack, preserves_flags));
    }
}

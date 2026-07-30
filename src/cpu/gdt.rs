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
    /// Stack pointers for CPL 0/1/2. Unused: we never leave ring 0.
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
// 0: null, 1: code, 2: data, 3+4: TSS (system descriptors are 16 bytes).
static GDT: Racy<[u64; 5]> = Racy::new([0; 5]);

/// Long-mode ring-0 code: present, DPL 0, code segment, L bit set.
const DESC_KERNEL_CODE: u64 = 0x00AF_9B00_0000_FFFF;
/// Long-mode ring-0 data: present, DPL 0, data segment, writable.
const DESC_KERNEL_DATA: u64 = 0x00CF_9300_0000_FFFF;

fn stack_top(s: &Stack) -> u64 {
    let base = s.0.as_ptr() as u64;
    // Stacks grow down, so hand out the far end, 16-byte aligned.
    (base + STACK_SIZE as u64) & !0xF
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

        let ptr = DescriptorTablePointer {
            limit: (size_of::<[u64; 5]>() - 1) as u16,
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

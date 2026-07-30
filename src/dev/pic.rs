//! The legacy 8259 PIC -- initialised only so it can be silenced.
//!
//! We deliver interrupts through the IOAPIC, so the PIC has no job. But it
//! powers on with IRQs mapped onto vectors 0x08..0x0F, which collide with CPU
//! exception vectors: a stray IRQ0 would arrive as #DF and produce a wildly
//! misleading "double fault" report.
//!
//! So: remap it somewhere harmless first, *then* mask everything. Masking
//! without remapping is not enough, because a spurious IRQ7 is delivered even
//! when the line is masked.

use crate::cpu::port::{io_wait, outb};

const PIC1_CMD: u16 = 0x20;
const PIC1_DATA: u16 = 0x21;
const PIC2_CMD: u16 = 0xA0;
const PIC2_DATA: u16 = 0xA1;

/// Remapped clear of both the exception range and our APIC vectors.
const OFFSET1: u8 = 0x30;
const OFFSET2: u8 = 0x38;

pub fn disable() {
    unsafe {
        // ICW1: begin initialisation, ICW4 will follow.
        outb(PIC1_CMD, 0x11);
        io_wait();
        outb(PIC2_CMD, 0x11);
        io_wait();

        // ICW2: vector offsets.
        outb(PIC1_DATA, OFFSET1);
        io_wait();
        outb(PIC2_DATA, OFFSET2);
        io_wait();

        // ICW3: master has a slave on IRQ2; slave's cascade identity is 2.
        outb(PIC1_DATA, 4);
        io_wait();
        outb(PIC2_DATA, 2);
        io_wait();

        // ICW4: 8086/88 mode.
        outb(PIC1_DATA, 0x01);
        io_wait();
        outb(PIC2_DATA, 0x01);
        io_wait();

        // Mask every line on both chips.
        outb(PIC1_DATA, 0xFF);
        outb(PIC2_DATA, 0xFF);
    }
}

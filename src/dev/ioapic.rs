//! I/O APIC: route external interrupts to CPU vectors.
//!
//! Access is indirect. Write a register index to IOREGSEL at offset 0, then
//! read or write the value through IOWIN at offset 0x10. Each redirection
//! entry is 64 bits and therefore occupies two consecutive indices.

use crate::acpi::IoApicInfo;
use core::ptr::{read_volatile, write_volatile};

const IOREGSEL: u64 = 0x00;
const IOWIN: u64 = 0x10;

const REG_VER: u32 = 0x01;
const REG_REDIR_BASE: u32 = 0x10;

/// Bit 16 of the low dword masks the entry.
const ENTRY_MASKED: u32 = 1 << 16;
/// Bit 13: 1 = active low.
const ENTRY_ACTIVE_LOW: u32 = 1 << 13;
/// Bit 15: 1 = level triggered.
const ENTRY_LEVEL: u32 = 1 << 15;

unsafe fn read(base: u64, reg: u32) -> u32 {
    unsafe {
        write_volatile((base + IOREGSEL) as *mut u32, reg);
        read_volatile((base + IOWIN) as *const u32)
    }
}

unsafe fn write(base: u64, reg: u32, value: u32) {
    unsafe {
        write_volatile((base + IOREGSEL) as *mut u32, reg);
        write_volatile((base + IOWIN) as *mut u32, value);
    }
}

/// Number of redirection entries this IOAPIC supports.
pub fn max_redirection_entries(io: &IoApicInfo) -> u32 {
    // IOAPICVER bits 16..23 hold "max entry", one less than the count.
    (unsafe { read(io.addr, REG_VER) } >> 16 & 0xFF) + 1
}

/// Mask every line. Firmware may leave entries enabled from its own use, and
/// an unexpected interrupt arriving before we have a handler is a fault report
/// at best and a silent lockup at worst.
pub fn mask_all(io: &IoApicInfo) {
    let count = max_redirection_entries(io);
    for i in 0..count {
        unsafe {
            write(io.addr, REG_REDIR_BASE + i * 2, ENTRY_MASKED);
            write(io.addr, REG_REDIR_BASE + i * 2 + 1, 0);
        }
    }
}

/// Route `gsi` to `vector` on the CPU with local APIC id `apic_id`.
///
/// `flags` are the MADT interrupt source override flags: bits 0..1 polarity,
/// bits 2..3 trigger mode. A value of 0 in either field means "conforms to the
/// bus default", which for ISA lines is active-high and edge-triggered.
pub fn route(io: &IoApicInfo, gsi: u32, vector: u8, apic_id: u8, flags: u16) -> bool {
    if gsi < io.gsi_base {
        return false;
    }
    let index = gsi - io.gsi_base;
    if index >= max_redirection_entries(io) {
        return false;
    }

    let mut low = vector as u32; // fixed delivery, physical destination, unmasked
    if flags & 0b11 == 0b11 {
        low |= ENTRY_ACTIVE_LOW;
    }
    if (flags >> 2) & 0b11 == 0b11 {
        low |= ENTRY_LEVEL;
    }

    unsafe {
        // Program the high dword (destination) before unmasking the low one,
        // so the entry is never briefly live pointing at CPU 0 by accident.
        write(io.addr, REG_REDIR_BASE + index * 2 + 1, (apic_id as u32) << 24);
        write(io.addr, REG_REDIR_BASE + index * 2, low);
    }
    true
}

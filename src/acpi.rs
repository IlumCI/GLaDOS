//! ACPI table discovery: RSDP -> XSDT -> MADT / FADT / HPET / MCFG.
//!
//! Everything here reads by explicit byte offset with `read_unaligned`, never
//! by declaring a `#[repr(C)]` struct and dereferencing it. That is deliberate.
//! ACPI tables are byte-packed and several fields sit at offsets Rust would
//! never choose: the HPET base address is at offset 44, which is 4-aligned but
//! not 8-aligned, so a `repr(C)` struct would quietly place that `u64` at 48
//! and read four bytes of the wrong field. `repr(C, packed)` avoids the padding
//! but then taking a reference to a field is itself unsound. Offsets it is.
//!
//! Reference: ACPI Specification 6.5, sections 5.2.5 (RSDP), 5.2.8 (XSDT),
//! 5.2.12 (MADT).

#![allow(dead_code)]

use core::ffi::c_void;
use core::ptr::read_unaligned;

#[inline]
unsafe fn rd_u8(p: *const u8, off: usize) -> u8 {
    unsafe { read_unaligned(p.add(off)) }
}
#[inline]
unsafe fn rd_u16(p: *const u8, off: usize) -> u16 {
    unsafe { read_unaligned(p.add(off) as *const u16) }
}
#[inline]
unsafe fn rd_u32(p: *const u8, off: usize) -> u32 {
    unsafe { read_unaligned(p.add(off) as *const u32) }
}
#[inline]
unsafe fn rd_u64(p: *const u8, off: usize) -> u64 {
    unsafe { read_unaligned(p.add(off) as *const u64) }
}

/// All ACPI tables checksum to zero over their whole declared length.
unsafe fn checksum_ok(p: *const u8, len: usize) -> bool {
    let mut sum: u8 = 0;
    for i in 0..len {
        sum = sum.wrapping_add(unsafe { rd_u8(p, i) });
    }
    sum == 0
}

const SDT_HEADER_LEN: usize = 36;

#[derive(Clone, Copy, Default)]
pub struct IoApicInfo {
    pub id: u8,
    pub addr: u64,
    pub gsi_base: u32,
}

/// An MADT interrupt source override: "ISA IRQ `source` is really GSI `gsi`".
///
/// These exist because the legacy PIC IRQ numbering and the IOAPIC's global
/// system interrupt numbering are not the same map. On most machines the
/// timer's IRQ 0 is remapped; keyboard IRQ 1 usually is not, but assuming that
/// is how you write a keyboard driver that works in QEMU and not on hardware.
#[derive(Clone, Copy, Default)]
pub struct OverrideInfo {
    pub source: u8,
    pub gsi: u32,
    pub flags: u16,
}

pub const MAX_IOAPICS: usize = 4;
pub const MAX_OVERRIDES: usize = 24;

#[derive(Clone, Copy)]
pub struct Acpi {
    pub revision: u8,
    pub lapic_addr: u64,
    pub cpus: usize,
    pub ioapics: [IoApicInfo; MAX_IOAPICS],
    pub ioapic_count: usize,
    pub overrides: [OverrideInfo; MAX_OVERRIDES],
    pub override_count: usize,
    pub hpet: Option<u64>,
    pub mcfg: Option<u64>,
    /// ACPI PM timer I/O port. A fixed 3.579545 MHz clock, and a useful
    /// independent reference if PIT calibration ever looks wrong.
    pub pm_timer: Option<u32>,
}

impl Acpi {
    const fn new() -> Self {
        Self {
            revision: 0,
            lapic_addr: 0xFEE0_0000,
            cpus: 0,
            ioapics: [IoApicInfo { id: 0, addr: 0, gsi_base: 0 }; MAX_IOAPICS],
            ioapic_count: 0,
            overrides: [OverrideInfo { source: 0, gsi: 0, flags: 0 }; MAX_OVERRIDES],
            override_count: 0,
            hpet: None,
            mcfg: None,
            pm_timer: None,
        }
    }

    /// Resolve a legacy ISA IRQ to its global system interrupt, honouring any
    /// override. Returns `(gsi, flags)`; identity-mapped when no override says
    /// otherwise.
    pub fn gsi_for_irq(&self, irq: u8) -> (u32, u16) {
        for i in 0..self.override_count {
            let o = self.overrides[i];
            if o.source == irq {
                return (o.gsi, o.flags);
            }
        }
        (irq as u32, 0)
    }

    pub fn primary_ioapic(&self) -> Option<IoApicInfo> {
        if self.ioapic_count == 0 {
            None
        } else {
            Some(self.ioapics[0])
        }
    }
}

/// Walk the ACPI tables starting from the RSDP the firmware handed us.
///
/// # Safety
/// `rsdp` must be the pointer taken from the UEFI configuration table, and the
/// tables must be identity-mapped.
pub unsafe fn parse(rsdp: *const c_void) -> Option<Acpi> {
    if rsdp.is_null() {
        return None;
    }
    let p = rsdp as *const u8;

    unsafe {
        // "RSD PTR "
        let mut sig = [0u8; 8];
        for (i, b) in sig.iter_mut().enumerate() {
            *b = rd_u8(p, i);
        }
        if &sig != b"RSD PTR " {
            return None;
        }
        // The v1 checksum covers only the first 20 bytes, even on a v2 RSDP.
        if !checksum_ok(p, 20) {
            return None;
        }

        let mut acpi = Acpi::new();
        acpi.revision = rd_u8(p, 15);

        // Revision >= 2 means an XSDT with 64-bit pointers. Below that we only
        // have the 32-bit RSDT.
        let (sdt, entry_size) = if acpi.revision >= 2 {
            if !checksum_ok(p, rd_u32(p, 20) as usize) {
                return None;
            }
            (rd_u64(p, 24), 8usize)
        } else {
            (rd_u32(p, 16) as u64, 4usize)
        };

        if sdt == 0 {
            return None;
        }
        let sdt = sdt as *const u8;
        let sdt_len = rd_u32(sdt, 4) as usize;
        if sdt_len < SDT_HEADER_LEN {
            return None;
        }

        let count = (sdt_len - SDT_HEADER_LEN) / entry_size;
        for i in 0..count {
            let off = SDT_HEADER_LEN + i * entry_size;
            let table = if entry_size == 8 {
                rd_u64(sdt, off)
            } else {
                rd_u32(sdt, off) as u64
            };
            if table == 0 {
                continue;
            }
            let t = table as *const u8;
            let mut tsig = [0u8; 4];
            for (j, b) in tsig.iter_mut().enumerate() {
                *b = rd_u8(t, j);
            }
            let len = rd_u32(t, 4) as usize;

            match &tsig {
                b"APIC" => parse_madt(t, len, &mut acpi),
                b"HPET" => acpi.hpet = Some(rd_u64(t, 44)),
                b"MCFG" => {
                    // First allocation entry starts at 44 (36 header + 8 reserved).
                    if len >= 44 + 16 {
                        acpi.mcfg = Some(rd_u64(t, 44));
                    }
                }
                b"FACP" => {
                    // FADT: PM_TMR_BLK at offset 76, PM_TMR_LEN at 91.
                    if len >= 92 && rd_u8(t, 91) == 4 {
                        let port = rd_u32(t, 76);
                        if port != 0 {
                            acpi.pm_timer = Some(port);
                        }
                    }
                }
                _ => {}
            }
        }

        Some(acpi)
    }
}

unsafe fn parse_madt(t: *const u8, len: usize, acpi: &mut Acpi) {
    unsafe {
        acpi.lapic_addr = rd_u32(t, 36) as u64;

        let mut off = 44;
        while off + 2 <= len {
            let kind = rd_u8(t, off);
            let elen = rd_u8(t, off + 1) as usize;
            // A zero-length entry would spin here forever.
            if elen < 2 || off + elen > len {
                break;
            }

            match kind {
                // Processor Local APIC. Bit 0 of flags = enabled.
                0 => {
                    if rd_u32(t, off + 4) & 1 != 0 {
                        acpi.cpus += 1;
                    }
                }
                // I/O APIC.
                1 => {
                    if acpi.ioapic_count < MAX_IOAPICS {
                        acpi.ioapics[acpi.ioapic_count] = IoApicInfo {
                            id: rd_u8(t, off + 2),
                            addr: rd_u32(t, off + 4) as u64,
                            gsi_base: rd_u32(t, off + 8),
                        };
                        acpi.ioapic_count += 1;
                    }
                }
                // Interrupt Source Override.
                2 => {
                    if acpi.override_count < MAX_OVERRIDES {
                        acpi.overrides[acpi.override_count] = OverrideInfo {
                            source: rd_u8(t, off + 3),
                            gsi: rd_u32(t, off + 4),
                            flags: rd_u16(t, off + 8),
                        };
                        acpi.override_count += 1;
                    }
                }
                // Local APIC Address Override. A 64-bit address at offset 4,
                // which is exactly the misalignment that motivates reading by
                // offset rather than by struct.
                5 => acpi.lapic_addr = rd_u64(t, off + 4),
                // Processor Local x2APIC.
                9 => {
                    if rd_u32(t, off + 8) & 1 != 0 {
                        acpi.cpus += 1;
                    }
                }
                _ => {}
            }

            off += elen;
        }
    }
}

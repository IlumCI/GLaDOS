//! Our own identity map.
//!
//! TempleOS ran identity-mapped in a single address space and so do we:
//! physical address equals virtual address, one set of page tables for the
//! whole machine, no higher half, no per-process tables, no TLB shootdown to
//! design. Combined with ring-0-only execution this removes most of what makes
//! x86-64 memory management difficult.
//!
//! One deliberate exception to a pure identity map: **virtual page 0 is left
//! unmapped**. That costs one extra page table for the first 2 MiB and buys a
//! hard fault on every null dereference, which -- given the `#PF` reporter in
//! `cpu::idt` -- turns the most common bug in the kernel into a legible message
//! instead of a silent read of whatever the firmware happened to leave at zero.

use super::frame::EarlyFrames;
use super::{GIB, LARGE_PAGE_SIZE, PAGE_SIZE};

const PRESENT: u64 = 1 << 0;
const WRITABLE: u64 = 1 << 1;
/// PS bit. On a PD entry this means "this is a 2 MiB page", not a pointer to a PT.
const HUGE: u64 = 1 << 7;

const ENTRIES: usize = 512;
const ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

#[inline]
unsafe fn table(phys: u64) -> &'static mut [u64; ENTRIES] {
    unsafe { &mut *(phys as *mut [u64; ENTRIES]) }
}

/// Build an identity map covering `[0, limit)` and return the PML4's physical
/// address. Does not install it -- see `activate`.
///
/// `limit` is rounded up to a 1 GiB boundary. One PML4 entry spans 512 GiB, so
/// a single PDPT covers anything this machine could plausibly have.
pub fn build_identity_map(frames: &mut EarlyFrames, limit: u64) -> Option<u64> {
    let gibs = limit.div_ceil(GIB).max(1) as usize;
    if gibs > ENTRIES {
        return None; // Beyond 512 GiB we would need more than one PDPT.
    }

    let pml4_phys = frames.alloc()?;
    let pdpt_phys = frames.alloc()?;

    unsafe {
        table(pml4_phys)[0] = pdpt_phys | PRESENT | WRITABLE;
    }

    for gib in 0..gibs {
        let pd_phys = frames.alloc()?;
        unsafe {
            table(pdpt_phys)[gib] = pd_phys | PRESENT | WRITABLE;
        }

        for i in 0..ENTRIES {
            let phys = gib as u64 * GIB + i as u64 * LARGE_PAGE_SIZE;

            if phys == 0 {
                // The first 2 MiB, at 4 KiB granularity, so page 0 can stay absent.
                let pt_phys = frames.alloc()?;
                unsafe {
                    let pt = table(pt_phys);
                    // Deliberately start at 1: entry 0 stays zero (not present).
                    for (p, entry) in pt.iter_mut().enumerate().skip(1) {
                        *entry = (p as u64 * PAGE_SIZE) | PRESENT | WRITABLE;
                    }
                    table(pd_phys)[0] = pt_phys | PRESENT | WRITABLE;
                }
                continue;
            }

            unsafe {
                table(pd_phys)[i] = phys | PRESENT | WRITABLE | HUGE;
            }
        }
    }

    Some(pml4_phys)
}

/// Install the map. Everything currently executing -- code, stack, framebuffer
/// -- must already be covered by it, or this instruction is the last one.
///
/// # Safety
/// `pml4_phys` must point at a complete, correct PML4.
pub unsafe fn activate(pml4_phys: u64) {
    unsafe { crate::cpu::write_cr3(pml4_phys & ADDR_MASK) };
}

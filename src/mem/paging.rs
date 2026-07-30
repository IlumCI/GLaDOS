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
use crate::uefi::MemoryDescriptor;

const PRESENT: u64 = 1 << 0;
const WRITABLE: u64 = 1 << 1;
/// PWT -- page write-through.
const WRITE_THROUGH: u64 = 1 << 3;
/// PCD -- page cache disable. Together with PWT this gives strong uncacheable,
/// which is mandatory for memory-mapped device registers.
const CACHE_DISABLE: u64 = 1 << 4;
/// PS bit. On a PD entry this means "this is a 2 MiB page", not a pointer to a PT.
const HUGE: u64 = 1 << 7;

/// Memory types from the UEFI map that represent real RAM, and may therefore
/// be cached write-back. Everything else is treated as device memory.
fn is_ram_type(ty: u32) -> bool {
    matches!(ty, 1..=7 | 9 | 10)
}

/// Is this physical address backed by RAM according to the firmware?
///
/// Anything that is not gets mapped uncacheable. That is not a nicety: a
/// write-back mapping over device registers means a write to the IOAPIC's
/// index register can sit in a write buffer while the read of its data
/// register is answered from a cache line. On this laptop that made the
/// IOAPIC report 120 redirection entries instead of 24. QEMU does not model
/// caches, so it looked correct there for as long as we only tested there.
fn addr_is_ram(addr: u64, mmap: *const u8, mmap_size: usize, desc_size: usize) -> bool {
    if desc_size == 0 {
        return false;
    }
    for i in 0..(mmap_size / desc_size) {
        let d = unsafe { &*(mmap.add(i * desc_size) as *const MemoryDescriptor) };
        if !is_ram_type(d.ty) {
            continue;
        }
        let start = d.phys_start;
        let end = d.phys_start + d.num_pages * PAGE_SIZE;
        if addr >= start && addr < end {
            return true;
        }
    }
    false
}

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
pub fn build_identity_map(
    frames: &mut EarlyFrames,
    limit: u64,
    mmap: *const u8,
    mmap_size: usize,
    desc_size: usize,
    fb_start: u64,
    fb_end: u64,
) -> Option<u64> {
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

            let mut flags = PRESENT | WRITABLE | HUGE;

            // The framebuffer is device memory, but it is linear and we only
            // ever write to it, so write-back is both correct (the iGPU snoops)
            // and far faster than uncacheable would be. Everything else that is
            // not RAM is a register file and must not be cached.
            let overlaps_fb = phys < fb_end && (phys + LARGE_PAGE_SIZE) > fb_start;
            if !overlaps_fb && !addr_is_ram(phys, mmap, mmap_size, desc_size) {
                flags |= WRITE_THROUGH | CACHE_DISABLE;
            }

            unsafe {
                table(pd_phys)[i] = phys | flags;
            }
        }
    }

    Some(pml4_phys)
}

/// Add a mapping after boot, creating page tables as needed.
///
/// The boot-time map covers RAM plus the low 4 GiB plus the framebuffer, which
/// is everything known at the time. It is not enough: PCI BARs can be anywhere
/// in the 64-bit address space, and both machines this runs on put them far
/// outside that range -- QEMU's NVMe controller at 768 GiB, this laptop's
/// framebuffer at 256 GiB. Touching an unmapped BAR faults on the very first
/// register read, which is exactly how this function came to exist.
///
/// New page tables come from the heap. That works only because the address
/// space is identity-mapped, so a heap pointer is also its own physical
/// address; the allocation is page-aligned for the same reason.
///
/// Device memory is always mapped uncacheable -- see `addr_is_ram` for why a
/// write-back mapping over registers produces plausible garbage.
pub fn map_range(phys_start: u64, len: u64, uncached: bool) -> bool {
    if len == 0 {
        return true;
    }
    let pml4_phys = crate::cpu::read_cr3() & ADDR_MASK;
    let start = phys_start & !(LARGE_PAGE_SIZE - 1);
    let end = (phys_start + len).div_ceil(LARGE_PAGE_SIZE) * LARGE_PAGE_SIZE;

    let mut addr = start;
    while addr < end {
        if !map_one_large(pml4_phys, addr, uncached) {
            return false;
        }
        addr += LARGE_PAGE_SIZE;
    }

    // Reloading CR3 flushes the TLB. Heavy-handed compared to `invlpg` per
    // page, but this runs once per device, not in any hot path.
    unsafe { crate::cpu::write_cr3(pml4_phys) };
    true
}

fn alloc_table() -> Option<u64> {
    use alloc::alloc::{alloc_zeroed, Layout};
    let layout = Layout::from_size_align(4096, 4096).ok()?;
    let p = unsafe { alloc_zeroed(layout) };
    if p.is_null() {
        None
    } else {
        Some(p as u64)
    }
}

fn map_one_large(pml4_phys: u64, addr: u64, uncached: bool) -> bool {
    let i4 = ((addr >> 39) & 511) as usize;
    let i3 = ((addr >> 30) & 511) as usize;
    let i2 = ((addr >> 21) & 511) as usize;

    unsafe {
        let pml4 = table(pml4_phys);
        let pdpt_phys = if pml4[i4] & PRESENT == 0 {
            let Some(p) = alloc_table() else { return false };
            pml4[i4] = p | PRESENT | WRITABLE;
            p
        } else {
            pml4[i4] & ADDR_MASK
        };

        let pdpt = table(pdpt_phys);
        let pd_phys = if pdpt[i3] & PRESENT == 0 {
            let Some(p) = alloc_table() else { return false };
            pdpt[i3] = p | PRESENT | WRITABLE;
            p
        } else {
            // A 1 GiB page here would have to be split before we could map a
            // 2 MiB entry inside it. We never create those, so treat it as a
            // failure rather than silently corrupting the mapping.
            if pdpt[i3] & HUGE != 0 {
                return false;
            }
            pdpt[i3] & ADDR_MASK
        };

        let pd = table(pd_phys);
        let mut flags = PRESENT | WRITABLE | HUGE;
        if uncached {
            flags |= WRITE_THROUGH | CACHE_DISABLE;
        }
        pd[i2] = addr | flags;
    }
    true
}

/// Install the map. Everything currently executing -- code, stack, framebuffer
/// -- must already be covered by it, or this instruction is the last one.
///
/// # Safety
/// `pml4_phys` must point at a complete, correct PML4.
pub unsafe fn activate(pml4_phys: u64) {
    unsafe { crate::cpu::write_cr3(pml4_phys & ADDR_MASK) };
}

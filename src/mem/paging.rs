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
/// U/S. Clear means ring 3 may not touch this page at all.
const USER: u64 = 1 << 2;
/// Bit 63. Means no-execute, but only once `EFER.NXE` is on.
const NX: u64 = 1 << 63;
/// The physical address field of a 2 MiB entry is bits 51:21, not 51:12.
const ADDR_MASK_2M: u64 = 0x000F_FFFF_FFE0_0000;

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

/// What a range of pages may be used for.
#[derive(Clone, Copy, PartialEq)]
pub struct Perm {
    pub present: bool,
    pub write: bool,
    pub exec: bool,
    /// Whether ring 3 may touch it.
    ///
    /// **The U bit is ANDed down every level of the walk**, so a page marked
    /// user under a directory that is not stays unreachable from ring 3. That
    /// is why `protect` opens the PML4, PDPT and PD entries along the way and
    /// not only the leaf. It is also why doing so is safe: every other page
    /// under those directories still has a clear U bit of its own, and the
    /// leaf is the gate.
    pub user: bool,
}

impl Perm {
    pub const RW: Perm = Perm { present: true, write: true, exec: false, user: false };
    pub const RO: Perm = Perm { present: true, write: false, exec: false, user: false };
    pub const RX: Perm = Perm { present: true, write: false, exec: true, user: false };
    pub const RWX: Perm = Perm { present: true, write: true, exec: true, user: false };
    pub const NONE: Perm = Perm { present: false, write: false, exec: false, user: false };
    /// What a guest's own pages get, and the only pages in the machine ring 3
    /// can reach.
    pub const USER_RWX: Perm = Perm { present: true, write: true, exec: true, user: true };
    pub const USER_RW: Perm = Perm { present: true, write: true, exec: false, user: true };

    fn bits(&self) -> u64 {
        let mut f = 0;
        if self.present {
            f |= PRESENT;
        }
        if self.write {
            f |= WRITABLE;
        }
        if !self.exec {
            f |= NX;
        }
        if self.user {
            f |= USER;
        }
        f
    }
}

/// Turn one 2 MiB entry into five hundred and twelve 4 KiB ones covering the
/// same bytes with the same rights.
///
/// **This is the whole reason per-page permissions were not free here.** The
/// identity map is built out of 2 MiB pages because that is one entry per two
/// megabytes and no page tables to walk, which is exactly right for a map that
/// never changes. Changing the rights on a single 4 KiB page inside one means
/// the 2 MiB entry has to stop existing first.
///
/// The split is invisible: same physical bytes, same flags, same cacheability.
/// A reader who did not know it happened would see no difference, which is the
/// property that makes it safe to do underneath a running kernel.
unsafe fn split_large(pd: &mut [u64; ENTRIES], i2: usize) -> bool {
    let e = pd[i2];
    if e & PRESENT == 0 {
        return false;
    }
    if e & HUGE == 0 {
        return true;
    }
    let base = e & ADDR_MASK_2M;
    // Everything except the address and the size bit carries over, so
    // uncached device memory stays uncached through the split.
    let flags = e & !(ADDR_MASK_2M | HUGE);
    let Some(pt_phys) = alloc_table() else { return false };
    unsafe {
        let pt = table(pt_phys);
        for (j, slot) in pt.iter_mut().enumerate() {
            *slot = (base + (j as u64) * PAGE_SIZE) | flags;
        }
    }
    pd[i2] = pt_phys | PRESENT | WRITABLE;
    true
}

/// The entry governing one address, splitting a huge page if it has to.
unsafe fn entry_for(addr: u64, split: bool) -> Option<&'static mut u64> {
    unsafe { entry_for_user(addr, split, false) }
}

/// As `entry_for`, and when `open` is set every directory on the way down is
/// marked user-accessible so the leaf's own U bit can take effect.
///
/// Opening a directory grants nothing by itself. The processor ANDs the U bit
/// across all four levels, so a directory marked user still hands ring 3
/// exactly the leaves whose own U bit is set, which is none of the kernel's.
unsafe fn entry_for_user(addr: u64, split: bool, open: bool) -> Option<&'static mut u64> {
    let pml4_phys = crate::cpu::read_cr3() & ADDR_MASK;
    let (i4, i3, i2, i1) = (
        ((addr >> 39) & 511) as usize,
        ((addr >> 30) & 511) as usize,
        ((addr >> 21) & 511) as usize,
        ((addr >> 12) & 511) as usize,
    );
    unsafe {
        let pml4 = table(pml4_phys);
        if pml4[i4] & PRESENT == 0 {
            return None;
        }
        if open {
            pml4[i4] |= USER;
        }
        let pdpt = table(pml4[i4] & ADDR_MASK);
        if pdpt[i3] & PRESENT == 0 || pdpt[i3] & HUGE != 0 {
            return None;
        }
        if open {
            pdpt[i3] |= USER;
        }
        let pd = table(pdpt[i3] & ADDR_MASK);
        if pd[i2] & PRESENT == 0 {
            return None;
        }
        if pd[i2] & HUGE != 0 {
            if !split {
                // Report the 2 MiB entry itself, which is right for a query:
                // its rights are the rights of every address inside it.
                return Some(&mut pd[i2]);
            }
            if !split_large(pd, i2) {
                return None;
            }
        }
        // **After the split, and that ordering is the whole of it.** Marked
        // before, this bit is part of the flags `split_large` copies into all
        // five hundred and twelve new entries, so opening one page to ring 3
        // opened the entire 2 MiB region around it. A claim caught it: the
        // page beside the opened one came back user-accessible.
        if open {
            pd[i2] |= USER;
        }
        let pt = table(pd[i2] & ADDR_MASK);
        Some(&mut pt[i1])
    }
}

/// What a given address may be used for right now.
///
/// Reads the tables rather than a shadow of them, so it cannot disagree with
/// the hardware about a page somebody else changed.
pub fn query(addr: u64) -> Option<Perm> {
    let e = *unsafe { entry_for(addr, false) }?;
    Some(Perm {
        user: e & USER != 0,
        present: e & PRESENT != 0,
        write: e & WRITABLE != 0,
        // A page is executable when NX is clear, and also when NX means
        // nothing because the feature was never enabled. Reporting a page as
        // non-executable while the processor happily runs it would be a
        // comfortable lie.
        exec: e & NX == 0 || !crate::cpu::nx_on(),
    })
}

/// Make a range obey `perm`, one 4 KiB page at a time.
///
/// The range is rounded outward to whole pages, because a permission is a
/// property of a page and half a page cannot have one. Answers false on the
/// first page it cannot reach, having already changed the ones before it: a
/// partial application is visible in `query` rather than rolled back, since
/// unwinding page-table edits needs a second copy of the state that would be
/// exactly as likely to be wrong.
pub fn protect(at: u64, len: usize, perm: Perm) -> bool {
    if len == 0 {
        return true;
    }
    let start = at & !(PAGE_SIZE - 1);
    let Some(sum) = at.checked_add(len as u64) else { return false };
    let end = sum.div_ceil(PAGE_SIZE) * PAGE_SIZE;
    let keep = !(PRESENT | WRITABLE | NX | USER);
    let mut addr = start;
    while addr < end {
        let Some(e) = (unsafe { entry_for_user(addr, true, perm.user) }) else { return false };
        *e = (*e & keep) | perm.bits();
        // Per page rather than a CR3 reload, because a reload flushes every
        // translation in the machine and this can be called with a guest's
        // whole heap. Skipping it entirely is the failure that matters: the
        // old translation stays cached and the change is silently unenforced.
        unsafe { crate::cpu::invlpg(addr) };
        addr += PAGE_SIZE;
    }
    true
}

/// What `diag paging` asks of all of it.
///
/// The claim that earns its place writes to a page it has just made
/// read-only, under `recover::guard`, and requires the fault to arrive.
/// Everything else here is arithmetic; that one is the only evidence that any
/// of it is enforced rather than merely recorded.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    use alloc::alloc::{alloc_zeroed, dealloc, Layout};
    let mut out = alloc::vec::Vec::new();

    out.push(("ring 0 respects the read-only bit (CR0.WP)", crate::cpu::wp_on()));
    out.push((
        "no-execute is on, or this part does not implement it",
        crate::cpu::nx_on() || !crate::cpu::nx_supported(),
    ));

    let Ok(layout) = Layout::from_size_align(2 * PAGE_SIZE as usize, PAGE_SIZE as usize) else {
        out.push(("a page could not be laid out for the checks", false));
        return out;
    };
    let page = unsafe { alloc_zeroed(layout) };
    if page.is_null() {
        out.push(("a page could not be taken for the checks", false));
        return out;
    }
    let at = page as u64;

    let before = query(at);
    out.push((
        "an ordinary heap page starts present and writable",
        before.is_some_and(|p| p.present && p.write),
    ));

    // Splitting is invisible: the bytes under a 2 MiB entry survive being
    // described by five hundred and twelve entries instead of one.
    unsafe { core::ptr::write_volatile(page, 0xA5) };
    let split_ok = protect(at, PAGE_SIZE as usize, Perm::RW);
    let survived = unsafe { core::ptr::read_volatile(page) } == 0xA5;
    out.push(("splitting a huge page keeps the bytes underneath it", split_ok && survived));
    out.push((
        "and the neighbouring page inside the same 2 MiB entry is still writable",
        query(at + PAGE_SIZE).is_some_and(|p| p.present && p.write),
    ));

    // The one that matters.
    let ro = protect(at, PAGE_SIZE as usize, Perm::RO);
    out.push(("a page can be made read-only", ro && query(at).is_some_and(|p| !p.write)));
    let read_still_works = unsafe { core::ptr::read_volatile(page) } == 0xA5;
    out.push(("a read-only page can still be read", read_still_works));

    let caught = crate::cpu::recover::guard(|| unsafe {
        core::ptr::write_volatile(page, 0x5A);
    });
    out.push((
        "writing to a read-only page faults, from ring 0, which is the whole point",
        caught.is_err(),
    ));
    out.push((
        "and the write did not land",
        unsafe { core::ptr::read_volatile(page) } == 0xA5,
    ));

    // The U bit, and the property that makes it worth having.
    let opened = protect(at, PAGE_SIZE as usize, Perm::USER_RW);
    out.push((
        "a page can be opened to ring 3",
        opened && query(at).is_some_and(|p| p.user),
    ));
    out.push((
        "and the page beside it is not opened with it, since the leaf is the gate",
        query(at + PAGE_SIZE).is_some_and(|p| !p.user),
    ));
    out.push((
        "closing it takes the U bit back off",
        protect(at, PAGE_SIZE as usize, Perm::RW) && query(at).is_some_and(|p| !p.user),
    ));

    // Put it back, or the heap hands out a page nothing may write to.
    let restored = protect(at, 2 * PAGE_SIZE as usize, Perm::RW);
    out.push((
        "and it can be given back, so the heap is not poisoned by the check",
        restored && query(at).is_some_and(|p| p.write),
    ));
    unsafe { dealloc(page, layout) };
    out
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

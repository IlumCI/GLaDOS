//! Early physical frame allocator.
//!
//! A bump allocator walking the UEFI memory map. It never frees, which is
//! correct for what it is used for: the page tables and the initial heap, both
//! of which live for the lifetime of the kernel. A real allocator arrives in M3.

use super::PAGE_SIZE;
use crate::uefi::MemoryDescriptor;

/// Never hand out anything below 1 MiB. Firmware often marks it conventional,
/// but it holds the EBDA and assorted legacy structures, and SMP will want the
/// low pages later for AP trampolines (real-mode entry has to live under 1 MiB).
const MIN_PHYS: u64 = 0x10_0000;

/// How many separate handouts are remembered.
///
/// Boot makes very few: the page tables take one frame at a time and the heap
/// takes one span, so a dozen is generous. Runs of frames that abut are merged
/// rather than each taking a slot, which is what makes that true -- the page
/// table walk is one contiguous run in practice.
const MAX_HANDOUTS: usize = 16;

pub struct EarlyFrames {
    mmap: *const u8,
    mmap_size: usize,
    desc_size: usize,
    idx: usize,
    cursor: u64,
    allocated: usize,
    /// Every range handed out, so somebody else can compute what is left.
    ///
    /// A bump allocator never frees, so its history *is* its state, and that is
    /// the only reason a list this short can be complete. `mem::fixed` needs it
    /// because the forward-only cursor cannot answer questions about addresses
    /// behind it, and those are exactly the addresses a fixed-address binary
    /// asks for.
    handouts: [(u64, u64); MAX_HANDOUTS],
    handout_n: usize,
    /// Set when a handout could not be recorded, so the snapshot can refuse to
    /// be trusted rather than describe memory it does not know is taken.
    handouts_lost: bool,
}

impl EarlyFrames {
    /// # Safety
    /// `mmap` must point at `mmap_size` bytes of UEFI memory descriptors with
    /// the firmware-reported `desc_size` stride, and boot services must be gone.
    pub unsafe fn new(mmap: *const u8, mmap_size: usize, desc_size: usize) -> Self {
        Self {
            mmap,
            mmap_size,
            desc_size,
            idx: 0,
            cursor: 0,
            allocated: 0,
            handouts: [(0, 0); MAX_HANDOUTS],
            handout_n: 0,
            handouts_lost: false,
        }
    }

    /// Record a handout, merging it onto the previous one when they abut.
    fn note(&mut self, at: u64, len: u64) {
        if self.handout_n > 0 {
            let last = &mut self.handouts[self.handout_n - 1];
            if last.0 + last.1 == at {
                last.1 += len;
                return;
            }
        }
        if self.handout_n >= MAX_HANDOUTS {
            self.handouts_lost = true;
            return;
        }
        self.handouts[self.handout_n] = (at, len);
        self.handout_n += 1;
    }

    /// Every range this allocator gave out, or `None` if any was not recorded.
    ///
    /// `None` rather than a short list, because a caller subtracting these from
    /// the firmware's map would otherwise report memory as free that is holding
    /// the page tables. Wrong in the one direction that cannot be survived.
    pub fn handouts(&self) -> Option<&[(u64, u64)]> {
        if self.handouts_lost {
            None
        } else {
            Some(&self.handouts[..self.handout_n])
        }
    }

    fn count(&self) -> usize {
        if self.desc_size == 0 {
            0
        } else {
            self.mmap_size / self.desc_size
        }
    }

    fn desc(&self, i: usize) -> &MemoryDescriptor {
        // Stride by the firmware's descriptor_size. It is allowed to exceed
        // size_of::<MemoryDescriptor>() and on some firmware it does.
        unsafe { &*(self.mmap.add(i * self.desc_size) as *const MemoryDescriptor) }
    }

    pub fn allocated_frames(&self) -> usize {
        self.allocated
    }

    /// The largest contiguous run still available, in pages, without taking it.
    ///
    /// `alloc_contiguous` cannot answer "would this fit?" -- it advances `idx`
    /// on every region it rejects and never rewinds, so asking it costs the
    /// regions it walked past. That made the heap ladder above it a fiction:
    /// once the first rung failed, the allocator sat at the end of the map and
    /// every smaller rung failed instantly, so a size the machine could not
    /// satisfy produced no heap at all rather than a smaller one.
    ///
    /// Rewinding to fix that would be worse than the bug. `install_paging` runs
    /// before the heap and took its frames from this same allocator; resetting
    /// `idx` and `cursor` would offer the live page tables' own frames a second
    /// time. So this looks forward from wherever the cursor already is, touches
    /// nothing, and reports what a subsequent call could actually get.
    pub fn largest_span(&self) -> usize {
        let mut best = 0u64;
        let mut cursor = self.cursor;
        for i in self.idx..self.count() {
            let d = self.desc(i);
            let region_end = d.phys_start + d.num_pages * PAGE_SIZE;
            let start = d.phys_start.max(MIN_PHYS).max(cursor);
            // The cursor only applies to the region it is inside; past that,
            // each region starts from its own base.
            cursor = 0;
            if !d.is_conventional() || start >= region_end {
                continue;
            }
            best = best.max(region_end - start);
        }
        (best / PAGE_SIZE) as usize
    }

    /// Total conventional memory still available, in pages, without taking it.
    ///
    /// Reported alongside `largest_span` because the two answer different
    /// questions and only one of them bounds an allocation. A machine can have
    /// gigabytes free and refuse a 200 MiB request.
    pub fn total_free(&self) -> usize {
        let mut total = 0u64;
        let mut cursor = self.cursor;
        for i in self.idx..self.count() {
            let d = self.desc(i);
            let region_end = d.phys_start + d.num_pages * PAGE_SIZE;
            let start = d.phys_start.max(MIN_PHYS).max(cursor);
            cursor = 0;
            if !d.is_conventional() || start >= region_end {
                continue;
            }
            total += region_end - start;
        }
        (total / PAGE_SIZE) as usize
    }

    /// Take `pages` physically contiguous zeroed frames from a single region.
    ///
    /// The heap needs one unbroken span; the bump cursor alone would happily
    /// straddle a region boundary and hand back two disjoint runs that look
    /// contiguous. So this skips forward to a region with room for the whole
    /// request rather than stitching.
    pub fn alloc_contiguous(&mut self, pages: usize) -> Option<u64> {
        let want = pages as u64 * PAGE_SIZE;
        loop {
            if self.idx >= self.count() {
                return None;
            }

            let d = self.desc(self.idx);
            let region_end = d.phys_start + d.num_pages * PAGE_SIZE;
            let usable_start = d.phys_start.max(MIN_PHYS);

            if !d.is_conventional() || usable_start >= region_end {
                self.idx += 1;
                self.cursor = 0;
                continue;
            }
            if self.cursor < usable_start {
                self.cursor = usable_start;
            }
            if self.cursor + want > region_end {
                self.idx += 1;
                self.cursor = 0;
                continue;
            }

            let base = self.cursor;
            self.cursor += want;
            self.allocated += pages;
            self.note(base, want);
            unsafe { core::ptr::write_bytes(base as *mut u8, 0, want as usize) };
            return Some(base);
        }
    }

    /// Take one zeroed 4 KiB frame. Returns its physical address, which is also
    /// its virtual address while we are identity-mapped.
    pub fn alloc(&mut self) -> Option<u64> {
        loop {
            if self.idx >= self.count() {
                return None;
            }

            let d = self.desc(self.idx);
            let region_start = d.phys_start;
            let region_end = d.phys_start + d.num_pages * PAGE_SIZE;
            let usable_start = region_start.max(MIN_PHYS);

            if !d.is_conventional() || usable_start >= region_end {
                self.idx += 1;
                self.cursor = 0;
                continue;
            }

            if self.cursor < usable_start {
                self.cursor = usable_start;
            }
            if self.cursor + PAGE_SIZE > region_end {
                self.idx += 1;
                self.cursor = 0;
                continue;
            }

            let frame = self.cursor;
            self.cursor += PAGE_SIZE;
            self.allocated += 1;
            self.note(frame, PAGE_SIZE);

            // Page tables must start empty: a stale non-zero entry is a mapping
            // we never intended and will not be able to explain later.
            unsafe { core::ptr::write_bytes(frame as *mut u8, 0, PAGE_SIZE as usize) };
            return Some(frame);
        }
    }
}

/// Highest physical address backed by actual memory.
///
/// Deliberately excludes `MappedIo`/`MappedIoPortSpace`/`PalCode`. Including
/// them looks reasonable and is a trap: q35 (and real firmware) put a 64-bit
/// PCI window hundreds of gigabytes above RAM, so taking the max over every
/// descriptor produces a "top of memory" far beyond anything worth mapping --
/// past 512 GiB it no longer fits in a single PDPT and the map build fails
/// outright.
///
/// Apertures we genuinely need, above all the framebuffer, are mapped by
/// address explicitly rather than by sweeping everything in between.
pub fn max_ram_address(mmap: *const u8, mmap_size: usize, desc_size: usize) -> u64 {
    // An allowlist, not a blocklist. Excluding the obvious MMIO types is not
    // enough: OVMF hands back descriptors for reserved space all the way to the
    // top of the physical address range -- measured here, `Reserved` ran to
    // 0x100_0000_0000, a clean 1 TiB. Taking a max over "everything that is not
    // MMIO" therefore still produced a 1 TiB map limit, which does not fit in a
    // single PDPT, so the whole build failed.
    //
    // Only these types describe memory worth mapping.
    const USABLE: [u32; 9] = [
        1,  // LoaderCode -- us
        2,  // LoaderData -- us, and the memory map buffer
        3,  // BootServicesCode
        4,  // BootServicesData
        5,  // RuntimeServicesCode
        6,  // RuntimeServicesData
        7,  // Conventional
        9,  // AcpiReclaim
        10, // AcpiNvs
    ];

    let mut max = 0u64;
    if desc_size == 0 {
        return 0;
    }
    for i in 0..(mmap_size / desc_size) {
        let d = unsafe { &*(mmap.add(i * desc_size) as *const MemoryDescriptor) };
        if !USABLE.contains(&d.ty) {
            continue;
        }
        let end = d.phys_start + d.num_pages * PAGE_SIZE;
        if end > max {
            max = end;
        }
    }
    max
}

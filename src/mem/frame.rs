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

pub struct EarlyFrames {
    mmap: *const u8,
    mmap_size: usize,
    desc_size: usize,
    idx: usize,
    cursor: u64,
    allocated: usize,
}

impl EarlyFrames {
    /// # Safety
    /// `mmap` must point at `mmap_size` bytes of UEFI memory descriptors with
    /// the firmware-reported `desc_size` stride, and boot services must be gone.
    pub unsafe fn new(mmap: *const u8, mmap_size: usize, desc_size: usize) -> Self {
        Self { mmap, mmap_size, desc_size, idx: 0, cursor: 0, allocated: 0 }
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

            // Page tables must start empty: a stale non-zero entry is a mapping
            // we never intended and will not be able to explain later.
            unsafe { core::ptr::write_bytes(frame as *mut u8, 0, PAGE_SIZE as usize) };
            return Some(frame);
        }
    }
}

/// Highest physical address the memory map mentions, ignoring type.
///
/// MMIO windows count: the framebuffer lives in one, and an aperture we fail to
/// map is a screen we cannot draw on.
pub fn max_physical_address(mmap: *const u8, mmap_size: usize, desc_size: usize) -> u64 {
    let mut max = 0u64;
    if desc_size == 0 {
        return 0;
    }
    for i in 0..(mmap_size / desc_size) {
        let d = unsafe { &*(mmap.add(i * desc_size) as *const MemoryDescriptor) };
        let end = d.phys_start + d.num_pages * PAGE_SIZE;
        if end > max {
            max = end;
        }
    }
    max
}

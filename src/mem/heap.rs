//! Kernel heap: an address-sorted free list with coalescing.
//!
//! The one invariant that makes this simple enough to trust: **every block
//! address and every block size is a multiple of `GRAIN` (16 bytes)**. Regions
//! come in page-aligned, requests are rounded up, and splits therefore always
//! land on the grain too. That means a leftover fragment is never an awkward
//! 8 bytes -- it is either exactly zero or at least a whole minimum block, so
//! `alloc` never has to absorb a slack tail it cannot describe.
//!
//! Why that matters: `GlobalAlloc::dealloc` is handed the original `Layout`,
//! not a header we wrote. If `alloc` ever handed back more than the rounded
//! request, `dealloc` would return less than was taken and quietly leak the
//! difference on every single allocation. Keeping both sides rounded the same
//! way makes them exact inverses.

use core::alloc::{GlobalAlloc, Layout};
use core::ptr;

/// Allocation granularity. Also the minimum block size, and `size_of::<Block>()`.
const GRAIN: usize = 16;

#[inline]
const fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

/// A free block. Lives *inside* the free memory it describes.
#[repr(C)]
struct Block {
    /// Total bytes, header included. Always a multiple of `GRAIN`.
    size: usize,
    next: *mut Block,
}

pub struct Heap {
    /// Dummy node so insertion never special-cases the list head.
    head: Block,
    total: usize,
    used: usize,
}

impl Heap {
    pub const fn empty() -> Self {
        Self {
            head: Block { size: 0, next: ptr::null_mut() },
            total: 0,
            used: 0,
        }
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn used(&self) -> usize {
        self.used
    }

    /// Donate a region to the heap.
    ///
    /// # Safety
    /// `start..start+size` must be exclusively owned, mapped and writable.
    pub unsafe fn add_region(&mut self, start: usize, size: usize) {
        let aligned = align_up(start, GRAIN);
        let shrink = aligned - start;
        if size <= shrink {
            return;
        }
        let size = (size - shrink) & !(GRAIN - 1);
        if size < GRAIN {
            return;
        }
        self.total += size;
        unsafe { self.insert_free(aligned, size) };
    }

    /// Insert a block into the address-sorted list, merging with either
    /// neighbour it happens to touch.
    unsafe fn insert_free(&mut self, addr: usize, size: usize) {
        let mut prev: *mut Block = &mut self.head;

        unsafe {
            // Walk to the last block that starts below `addr`.
            while !(*prev).next.is_null() && ((*prev).next as usize) < addr {
                prev = (*prev).next;
            }

            let next = (*prev).next;
            let block = addr as *mut Block;
            (*block).size = size;
            (*block).next = next;
            (*prev).next = block;

            // Merge forward.
            if !next.is_null() && addr + size == next as usize {
                (*block).size += (*next).size;
                (*block).next = (*next).next;
            }

            // Merge backward. `prev` may be the dummy head, which never
            // touches anything because its address is not in the heap region.
            if prev as *const Block != &self.head as *const Block {
                let prev_end = prev as usize + (*prev).size;
                if prev_end == addr {
                    (*prev).size += (*block).size;
                    (*prev).next = (*block).next;
                }
            }
        }
    }

    unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
        let size = align_up(layout.size().max(1), GRAIN);
        let align = layout.align().max(GRAIN);

        let mut prev: *mut Block = &mut self.head;
        unsafe {
            loop {
                let cur = (*prev).next;
                if cur.is_null() {
                    return ptr::null_mut();
                }

                let start = cur as usize;
                let block_size = (*cur).size;
                let payload = align_up(start, align);
                // Every block start is GRAIN-aligned and every alignment is a
                // power of two >= GRAIN, so this padding is a multiple of GRAIN.
                let front = payload - start;

                if front + size > block_size {
                    prev = cur;
                    continue;
                }

                let tail = block_size - front - size;
                let next = (*cur).next;

                if front == 0 {
                    if tail == 0 {
                        (*prev).next = next;
                    } else {
                        let rest = (payload + size) as *mut Block;
                        (*rest).size = tail;
                        (*rest).next = next;
                        (*prev).next = rest;
                    }
                } else {
                    // Keep the front padding as a free block.
                    (*cur).size = front;
                    if tail == 0 {
                        (*cur).next = next;
                    } else {
                        let rest = (payload + size) as *mut Block;
                        (*rest).size = tail;
                        (*rest).next = next;
                        (*cur).next = rest;
                    }
                }

                self.used += size;
                return payload as *mut u8;
            }
        }
    }

    unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        if ptr.is_null() {
            return;
        }
        // Exactly the rounding `alloc` used, so this returns exactly what it took.
        let size = align_up(layout.size().max(1), GRAIN);
        self.used -= size;
        unsafe { self.insert_free(ptr as usize, size) };
    }
}

pub struct LockedHeap {
    inner: crate::sync::Racy<Heap>,
}

impl LockedHeap {
    pub const fn new() -> Self {
        Self { inner: crate::sync::Racy::new(Heap::empty()) }
    }

    /// # Safety
    /// See `Heap::add_region`.
    pub unsafe fn add_region(&self, start: usize, size: usize) {
        unsafe { self.inner.get().add_region(start, size) }
    }

    pub fn stats(&self) -> (usize, usize) {
        let h = unsafe { self.inner.get() };
        (h.used(), h.total())
    }
}

// Single core, and allocation never happens inside an interrupt handler in
// this kernel. When either of those stops being true this needs a real lock --
// it uses Racy precisely so it shows up in the same grep as everything else
// that SMP will break.
unsafe impl GlobalAlloc for LockedHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { self.inner.get().alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.inner.get().dealloc(ptr, layout) }
    }
}

#[global_allocator]
pub static HEAP: LockedHeap = LockedHeap::new();

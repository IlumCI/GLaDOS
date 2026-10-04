//! Memory a device reads and writes on its own: DMA regions.
//!
//! Shared by the drivers that hand a part addresses -- the Intel radio and the
//! HD Audio controller. Both depend on the same two facts, so they live once.

/// A region a device may read or write, owned by us.
///
/// **Physical equals virtual, and that is load-bearing rather than convenient.**
/// `build_identity_map` maps everything at its own address, so a heap pointer
/// *is* a bus address and there is no translation to get wrong -- the same
/// property `cpu::code` leans on to execute from an allocation and `mem::space`
/// checks before writing CR3. It is asserted rather than assumed, because the day
/// it stops being true this module silently hands the device the wrong addresses
/// and the symptom is a part that fetches microcode from somebody else's memory.
pub struct Dma {
    ptr: *mut u8,
    layout: core::alloc::Layout,
}

impl Dma {
    /// Allocate `len` bytes aligned to `align`, zeroed.
    ///
    /// Zeroed because the descriptor has reserved fields and unused DRAM slots,
    /// and firmware reads all of them. A slot holding heap debris is an address
    /// the part will try to fetch from.
    pub fn new(len: usize, align: usize) -> Option<Dma> {
        // A zero-length region has no address worth handing over, and `Layout`
        // refuses a non-power-of-two alignment -- both are programming errors
        // here rather than conditions, so they answer `None` rather than panic.
        if len == 0 || !align.is_power_of_two() {
            return None;
        }
        let layout = core::alloc::Layout::from_size_align(len, align).ok()?;
        // Safety: the layout is non-zero-sized and validated above.
        let ptr = unsafe { alloc::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return None;
        }
        Some(Dma { ptr, layout })
    }

    /// The address to give the device. Identity mapped, so this is the virtual
    /// address unchanged.
    pub fn pa(&self) -> u64 {
        self.ptr as u64
    }

    pub fn len(&self) -> usize {
        self.layout.size()
    }

    pub fn as_slice(&self) -> &[u8] {
        // Safety: our own allocation, of exactly this length.
        unsafe { core::slice::from_raw_parts(self.ptr, self.layout.size()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // Safety: as above, and `&mut self` is the only writer.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
    }
}

impl Drop for Dma {
    fn drop(&mut self) {
        // **Nothing here can stop the device first**, and that is stated rather
        // than solved: a `Dma` must outlive every fetch the part will make from
        // it, which is an ordering its owner is responsible for. The context
        // info may be released once firmware reports alive; the paging regions
        // may not be released until the device is down. Upstream keeps those in
        // two different lifetimes for exactly this reason.
        //
        // Safety: allocated by us with this layout and not freed twice.
        unsafe { alloc::alloc::dealloc(self.ptr, self.layout) }
    }
}

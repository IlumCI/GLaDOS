//! A second address space, and the first thing in this kernel to move CR3
//! after boot.
//!
//! `paging::activate` had exactly one caller, `main.rs`, at the moment the
//! identity map replaces the firmware's tables. Everything since ran on that
//! one root, so "can this machine switch address spaces at all" had never been
//! asked here -- and it is the first question `fork` and `execve` depend on.
//! `syscall.rs` says `fork` is the one thing this system cannot grow into,
//! which is true of a kernel with one root and is a statement about the tables
//! rather than about the machine.
//!
//! Two things live here and the order between them is the design.
//!
//! **Sharing.** `sharing_kernel` copies the kernel's 512 top-level entries,
//! which are *pointers* to the PDPTs it already built, so both roots walk the
//! identical tables underneath. There is exactly one copy of every mapping, a
//! `protect` through one root is visible through the other because it is the
//! same entry, and nothing can drift. That is what makes the switch itself
//! safe to prove first, on its own, before anything depends on it.
//!
//! **Divergence.** `map_page` gives a space a mapping the kernel's root does
//! not have, which is the thing a process actually needs: two spaces naming
//! one virtual address and reaching different physical memory. It is confined
//! to `WINDOW` and above, and the confinement is the whole safety argument --
//! see `map_page`.
//!
//! What is still missing for `fork` is placement rather than mechanism: a
//! process wants its own `0x400000`, and that address is inside the identity
//! map's own subtree. Privatising it means giving a space its own page
//! directories for a region the kernel is also using, and that is the next
//! step.

use core::sync::atomic::{AtomicU64, Ordering};

use super::paging::{ADDR_MASK, HUGE, PRESENT, USER, WRITABLE};

const ENTRIES: usize = 512;

/// Where a space's private mappings begin: 512 GiB, which is PML4 entry 1.
///
/// **Chosen because the identity map provably cannot reach it.**
/// `build_identity_map` puts everything under a single PDPT hung off entry 0,
/// and one PML4 entry spans 512 GiB, so every address the kernel maps has a
/// top-level index of zero. Anything at or above this is unmapped in the
/// kernel's root and in every space that has not asked for it, which means a
/// mistake here faults on an address nothing owns instead of quietly landing
/// in the heap. `checks` asserts the kernel really does not map it rather than
/// taking the arithmetic on trust.
pub const WINDOW: u64 = 1 << 39;

/// The root every kernel task runs on, sampled the first time one of these is
/// built.
///
/// Lazily rather than at boot because that needs a call in `main.rs` and this
/// can establish it truthfully on its own: `Space::sharing_kernel` is the only
/// producer, it records this before it can possibly have activated anything,
/// and nothing else in the tree moves CR3. A sample taken while a `Space` were
/// installed would record that space as the kernel's and every restore
/// afterwards would put back the wrong root, so the ordering inside
/// `sharing_kernel` is load-bearing rather than incidental.
static KERNEL_ROOT: AtomicU64 = AtomicU64::new(0);

fn page() -> Option<(*mut u8, u64)> {
    use alloc::alloc::{alloc_zeroed, Layout};
    let layout = Layout::from_size_align(4096, 4096).ok()?;
    let ptr = unsafe { alloc_zeroed(layout) };
    if ptr.is_null() {
        return None;
    }
    Some((ptr, ptr as u64))
}

fn give_back(ptr: *mut u8) {
    use alloc::alloc::{dealloc, Layout};
    if let Ok(layout) = Layout::from_size_align(4096, 4096) {
        unsafe { dealloc(ptr, layout) };
    }
}

/// A page-aligned PML4 of this kernel's own, and every table under it that
/// this space rather than the kernel owns.
pub struct Space {
    root: u64,
    ptr: *mut u8,
    /// Tables created by `map_page`. Freed on drop, and *only* these: the
    /// kernel's own tables are reachable from this root and must outlive it.
    owned: alloc::vec::Vec<*mut u8>,
}

impl Space {
    /// A root that maps everything the kernel's does, by sharing its entries.
    pub fn sharing_kernel() -> Option<Space> {
        // Before anything else, and see KERNEL_ROOT: this is the point at
        // which CR3 is known to be the kernel's.
        let kernel = kernel_root();

        // CR3 carries a physical address in bits 51:12, so the table has to
        // start on a page boundary; the low bits are flags and an unaligned
        // root would be read as a different address with attributes set.
        let (ptr, root) = page()?;

        // **The table has to be identity-mapped, and here it is checked rather
        // than assumed.** CR3 takes a physical address while every write to
        // the table below goes through a virtual one, so the two have to be
        // the same number. They are, because the heap lives inside the
        // identity map -- the same property `cpu::code` leans on to execute
        // from a heap allocation. Checked because the day that stops being
        // true, the failure is the processor walking whatever happens to live
        // at that physical address, which is not a diagnosable event.
        if super::paging::query(root).is_none() {
            give_back(ptr);
            return None;
        }

        // Copy the top level. These are pointers to the kernel's own PDPTs, so
        // this shares the whole tree rather than duplicating it: 512 words
        // against walking and cloning every table under them, and no second
        // copy of a mapping that could disagree with the first.
        unsafe {
            let from = &*(kernel as *const [u64; ENTRIES]);
            let to = &mut *(root as *mut [u64; ENTRIES]);
            to.copy_from_slice(from);
        }

        Some(Space { root, ptr, owned: alloc::vec::Vec::new() })
    }

    /// The physical address of this root, which is what CR3 wants.
    pub fn root(&self) -> u64 {
        self.root
    }

    /// True while the processor is walking this root.
    pub fn is_active(&self) -> bool {
        crate::cpu::read_cr3() & ADDR_MASK == self.root & ADDR_MASK
    }

    /// True if this space allocated `phys`, so writing to it changes nothing
    /// the kernel or another space can see.
    ///
    /// **This is the one invariant the low half rests on.** A table reached
    /// from this root is either one of ours or one we inherited by copying a
    /// pointer, and the two are indistinguishable from the entry alone. Every
    /// edit that could damage somebody else's map is gated on this answer
    /// rather than on where the address happens to be.
    fn owns_table(&self, phys: u64) -> bool {
        phys == self.root || self.owned.iter().any(|p| *p as u64 == phys)
    }

    /// Turn a 2 MiB entry in a table *we own* into a page table covering the
    /// same bytes with the same flags, and answer where it is.
    ///
    /// The 512 entries are written out rather than left absent, because the
    /// large page was mapping real memory and dropping it would unmap the
    /// other 2 MiB less a page. `paging::split_large` does this one level
    /// down for the kernel's own map; this is the same bargain against a
    /// table this space owns, which is what makes it safe to do at all.
    fn split(&mut self, table: u64, idx: usize) -> Option<u64> {
        if !self.owns_table(table) {
            return None;
        }
        let t = unsafe { &mut *(table as *mut [u64; ENTRIES]) };
        let old = t[idx];
        let base = old & ADDR_MASK;
        let flags = old & !ADDR_MASK & !HUGE;
        let (ptr, phys) = page()?;
        self.owned.push(ptr);
        let pt = unsafe { &mut *(phys as *mut [u64; ENTRIES]) };
        for (i, slot) in pt.iter_mut().enumerate() {
            *slot = (base + (i as u64) * 4096) | flags;
        }
        t[idx] = phys | PRESENT | WRITABLE | USER;
        Some(phys)
    }

    /// Take a private copy of a table this space is sharing, and point the
    /// parent's entry at it.
    ///
    /// Copy-on-write, one level at a time, and it is what makes a low mapping
    /// possible at all: the descent starts at the root, which this space
    /// always owns, so each step either finds a table it already owns or makes
    /// one. By the time a leaf is written, every table above it is this
    /// space's own and the kernel's are untouched. The parent must be owned or
    /// the rewrite would be visible to everybody, which is the same gate
    /// `split` applies and for the same reason.
    fn privatise(&mut self, parent: u64, idx: usize, shared: u64) -> Option<u64> {
        if !self.owns_table(parent) {
            return None;
        }
        let (ptr, mine) = page()?;
        self.owned.push(ptr);
        unsafe {
            let from = &*(shared as *const [u64; ENTRIES]);
            let to = &mut *(mine as *mut [u64; ENTRIES]);
            to.copy_from_slice(from);
            (&mut *(parent as *mut [u64; ENTRIES]))[idx] = mine | PRESENT | WRITABLE | USER;
        }
        Some(mine)
    }

    /// Find or create the next table down, and answer where it is.
    fn step(&mut self, table: u64, idx: usize) -> Option<u64> {
        let e = unsafe { (&*(table as *const [u64; ENTRIES]))[idx] };
        if e & PRESENT != 0 {
            if e & HUGE != 0 {
                // A large page covers this. Splitting a table we *share* would
                // edit the kernel's own map through a root nobody thinks of as
                // the kernel's, so `split` refuses unless this space owns the
                // table. Refusing is the only answer that cannot damage
                // somebody else's mappings.
                return self.split(table, idx);
            }
            let next = e & ADDR_MASK;
            if self.owns_table(next) {
                return Some(next);
            }
            return self.privatise(table, idx, next);
        }
        let t = unsafe { &mut *(table as *mut [u64; ENTRIES]) };
        let (ptr, phys) = page()?;
        self.owned.push(ptr);
        // The U bit is ANDed down all four levels, so an intermediate without
        // it makes every leaf underneath unreachable from ring 3 however the
        // leaf is marked. Set here and gated at the leaf, which is the
        // arrangement `paging::protect` already uses and argues for.
        t[idx] = phys | PRESENT | WRITABLE | USER;
        Some(phys)
    }

    /// Map one 4 KiB page at `at`, privately to this space.
    ///
    /// **Above `WINDOW` only.** Not because the tables could not take it --
    /// `step` privatises its way down and the kernel's are never touched --
    /// but because of what a low address *means*: the kernel is identity
    /// mapped, so shadowing virtual `0x2c00000` in a space points the kernel's
    /// own heap pointer at somebody else's page for as long as that space is
    /// installed. The tables would be perfectly correct and the machine would
    /// be reading the wrong memory. Whether an address is safe to shadow is a
    /// question about what the kernel is using, and `map_low` is the entry
    /// point that asks it.
    pub fn map_page(&mut self, at: u64, phys: u64, writable: bool, user: bool) -> bool {
        if at < WINDOW {
            return false;
        }
        self.map_at(at, phys, writable, user)
    }

    /// Map one page at a *low* address, the way a process needs.
    ///
    /// This is the placement half of `fork`, and the guard is the whole of it.
    /// `mem::fixed::is_free` answers whether anything on this machine is using
    /// that physical memory, which -- the kernel being identity mapped -- is
    /// exactly the question "is it safe to shadow this virtual address". It is
    /// a *query* and deliberately not a `claim`: two spaces both wanting
    /// `0x400000` is the ordinary case for processes and reserving it would
    /// refuse the second for no reason. What stays global is the physical
    /// page each of them maps to, and those come from the allocator.
    pub fn map_low(&mut self, at: u64, phys: u64, writable: bool, user: bool)
        -> Result<(), &'static str>
    {
        if at % 4096 != 0 {
            return Err("a low mapping has to start on a page boundary");
        }
        if !super::fixed::is_free(at, 4096) {
            return Err("the kernel is using that address, so shadowing it would move its own memory");
        }
        if self.map_at(at, phys, writable, user) {
            Ok(())
        } else {
            Err("the tables could not be built")
        }
    }

    fn map_at(&mut self, at: u64, phys: u64, writable: bool, user: bool) -> bool {
        if at % 4096 != 0 || phys % 4096 != 0 {
            return false;
        }
        let i4 = ((at >> 39) & 511) as usize;
        let (i3, i2, i1) = (
            ((at >> 30) & 511) as usize,
            ((at >> 21) & 511) as usize,
            ((at >> 12) & 511) as usize,
        );
        let root = self.root;
        let pdpt = match self.step(root, i4) {
            Some(p) => p,
            None => return false,
        };
        let pd = match self.step(pdpt, i3) {
            Some(p) => p,
            None => return false,
        };
        let pt = match self.step(pd, i2) {
            Some(p) => p,
            None => return false,
        };
        let mut flags = PRESENT;
        if writable {
            flags |= WRITABLE;
        }
        if user {
            flags |= USER;
        }
        unsafe { (&mut *(pt as *mut [u64; ENTRIES]))[i1] = (phys & ADDR_MASK) | flags };
        true
    }

    /// Run `f` with this space installed, and put the kernel's root back
    /// whatever happens.
    ///
    /// **Interrupts are off for the whole of it**, and not because the switch
    /// is delicate: `mov cr3` is one instruction and a shared root maps the
    /// same memory either side of it. It is off because a task switch here
    /// would leave *another* task running on this root, and with `map_page`
    /// that root now has mappings the kernel's does not -- which is exactly
    /// the thing that stops being harmless. Nothing inside may print: the
    /// console is a `Spin` taken with `lock_irq`, and taking it with
    /// interrupts already masked is fine, but a paint path that blocked would
    /// have no tick to be rescued by.
    pub fn with<R>(&self, f: impl FnOnce() -> R) -> R {
        crate::cpu::without_interrupts(|| {
            let prev = crate::cpu::read_cr3() & ADDR_MASK;
            unsafe { super::paging::activate(self.root) };
            let out = f();
            unsafe { super::paging::activate(prev) };
            out
        })
    }
}

impl Drop for Space {
    fn drop(&mut self) {
        // **Put the root back before giving the pages away.** Freeing a table
        // the processor is still walking hands the allocator memory that the
        // next translation will read, and the fault arrives somewhere with
        // nothing to do with page tables. This is the fourth place in this
        // tree that has had to say "restore before release" -- `munmap` and
        // the U bit, `teardown` and the interpreter's rights, `give_back` --
        // and it is written here rather than left to the caller because a
        // `Drop` is exactly where nobody remembers.
        if self.is_active() {
            unsafe { super::paging::activate(kernel_root()) };
        }
        for p in core::mem::take(&mut self.owned) {
            give_back(p);
        }
        give_back(self.ptr);
    }
}

/// The root the kernel booted on.
///
/// See `KERNEL_ROOT`: correct only because the first sample is taken by
/// `Space::sharing_kernel` before it activates anything.
pub fn kernel_root() -> u64 {
    let seen = KERNEL_ROOT.load(Ordering::Relaxed);
    if seen != 0 {
        return seen;
    }
    let now = crate::cpu::read_cr3() & ADDR_MASK;
    KERNEL_ROOT.store(now, Ordering::Relaxed);
    now
}

/// Claims. The ones that earn their place are the three that do real work
/// while a second root is installed, and the two that show divergence.
///
/// A switch asserted only by reading CR3 back proves the register accepted a
/// write. What has to be true is that the *machine* still works underneath it,
/// so the middle claims allocate, write and read through pointers taken
/// beforehand. Nothing inside `with` prints, for the reason `with` gives.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    use alloc::vec::Vec;
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    let boot = crate::cpu::read_cr3() & ADDR_MASK;
    let mut space = match Space::sharing_kernel() {
        Some(s) => s,
        None => {
            out.push(("a second root could be built", false));
            return out;
        }
    };
    out.push(("a second root could be built", true));
    out.push((
        "and the kernel's own root was sampled before anything moved",
        kernel_root() == boot,
    ));
    out.push((
        "the table is page-aligned, as CR3 requires",
        space.root() % 4096 == 0,
    ));
    out.push((
        "and it is identity-mapped, so its physical and virtual addresses agree",
        super::paging::query(space.root()).is_some(),
    ));
    out.push(("it is a different root from the kernel's", space.root() != boot));

    let same = unsafe {
        let a = &*(boot as *const [u64; ENTRIES]);
        let b = &*(space.root() as *const [u64; ENTRIES]);
        a.iter().zip(b.iter()).all(|(x, y)| x == y)
    };
    out.push(("and it shares all 512 top-level entries with it", same));

    // The switch, and work done while it is installed.
    let probe: alloc::boxed::Box<u64> = alloc::boxed::Box::new(0x5541_4C55_4531_u64);
    let addr = &*probe as *const u64;
    let (switched, read_back, allocated) = space.with(|| {
        let on = crate::cpu::read_cr3() & ADDR_MASK == space.root();
        let seen = unsafe { core::ptr::read_volatile(addr) };
        let v: Vec<u64> = (0..512).collect();
        (on, seen, v.len() == 512 && v[511] == 511)
    });
    out.push(("CR3 moved to it", switched));
    out.push((
        "a pointer taken before the switch read the same value after it",
        read_back == 0x5541_4C55_4531_u64,
    ));
    out.push(("and the heap still answered while it was installed", allocated));
    out.push((
        "the kernel's root is back afterwards",
        crate::cpu::read_cr3() & ADDR_MASK == boot,
    ));

    // ---- divergence ----
    //
    // The window has to be unmapped in the kernel's root or none of the rest
    // means anything: a claim that two spaces disagree about an address the
    // kernel also maps would be reading the kernel's page either time.
    out.push((
        "the private window is unmapped in the kernel's own root",
        super::paging::query(WINDOW).is_none(),
    ));
    out.push((
        "a mapping below the window is refused, since entry 0 is the kernel's",
        !space.map_page(0x400000, 0x400000, true, false),
    ));

    if let (Some((ap, aphys)), Some((bp, bphys)), Some(mut other)) =
        (page(), page(), Space::sharing_kernel())
    {
        // Distinct physical pages, distinguishable contents.
        unsafe {
            core::ptr::write_volatile(aphys as *mut u64, 0xAAAA_0000_1111_u64);
            core::ptr::write_volatile(bphys as *mut u64, 0xBBBB_0000_2222_u64);
        }
        let mapped_a = space.map_page(WINDOW, aphys, true, false);
        let mapped_b = other.map_page(WINDOW, bphys, true, false);
        out.push(("one virtual address maps in both spaces", mapped_a && mapped_b));
        out.push((
            "and to genuinely different physical pages",
            aphys != bphys,
        ));

        let seen_a = space.with(|| unsafe { core::ptr::read_volatile(WINDOW as *const u64) });
        let seen_b = other.with(|| unsafe { core::ptr::read_volatile(WINDOW as *const u64) });
        out.push((
            "the first space reads its own page through that address",
            seen_a == 0xAAAA_0000_1111_u64,
        ));
        out.push((
            "and the second reads a different one through the same address",
            seen_b == 0xBBBB_0000_2222_u64,
        ));

        // A write through the window must land in that space's page and
        // nowhere else. This is the claim that would catch a leaf pointing at
        // the wrong frame, which reads identically to a correct one until
        // something writes.
        space.with(|| unsafe {
            core::ptr::write_volatile(WINDOW as *mut u64, 0xCCCC_0000_3333_u64)
        });
        let a_changed = unsafe { core::ptr::read_volatile(aphys as *const u64) };
        let b_intact = unsafe { core::ptr::read_volatile(bphys as *const u64) };
        out.push((
            "a write through it lands in that space's page",
            a_changed == 0xCCCC_0000_3333_u64,
        ));
        out.push((
            "and leaves the other space's page alone",
            b_intact == 0xBBBB_0000_2222_u64,
        ));

        drop(other);
        give_back(ap);
        give_back(bp);
    } else {
        out.push(("a second space and two pages could be built", false));
    }

    out.push((
        "the window is still unmapped in the kernel's root",
        super::paging::query(WINDOW).is_none(),
    ));

    // ---- the low half, which is where a process actually lives ----
    //
    // 0x400000 is busybox's own base and every non-PIE binary's, and
    // `mem::fixed` reports it inside the placeable run. The claim that earns
    // its place is the refusal: shadowing an address the kernel is using
    // builds perfectly correct tables and points the kernel's own pointer at
    // somebody else's page, which is not a diagnosable failure.
    let heap_addr = (&*probe as *const u64 as u64) & !0xFFF;
    out.push((
        "an address the kernel is using is refused for a low mapping",
        Space::sharing_kernel()
            .map(|mut s| s.map_low(heap_addr, heap_addr, true, false).is_err())
            .unwrap_or(false),
    ));
    out.push((
        "and map_page still refuses the low half outright",
        Space::sharing_kernel()
            .map(|mut s| !s.map_page(0x400000, 0x400000, true, false))
            .unwrap_or(false),
    ));

    if let (Some((cp, cphys)), Some((dp, dphys)), Some(mut low_a), Some(mut low_b)) =
        (page(), page(), Space::sharing_kernel(), Space::sharing_kernel())
    {
        unsafe {
            core::ptr::write_volatile(cphys as *mut u64, 0x1111_2222_3333_u64);
            core::ptr::write_volatile(dphys as *mut u64, 0x4444_5555_6666_u64);
        }
        let ra = low_a.map_low(0x400000, cphys, true, false);
        let rb = low_b.map_low(0x400000, dphys, true, false);
        out.push((
            "two spaces each map 0x400000, the address a non-PIE binary insists on",
            ra.is_ok() && rb.is_ok(),
        ));
        let sa = low_a.with(|| unsafe { core::ptr::read_volatile(0x400000 as *const u64) });
        let sb = low_b.with(|| unsafe { core::ptr::read_volatile(0x400000 as *const u64) });
        out.push((
            "the first reads its own page through 0x400000",
            sa == 0x1111_2222_3333_u64,
        ));
        out.push((
            "and the second reads a different one through the same address",
            sb == 0x4444_5555_6666_u64,
        ));
        // The kernel's own map must be exactly as it was. This is what the
        // copy-on-write descent is for, and a failure here means a table was
        // edited in place that somebody else was sharing.
        out.push((
            "the kernel's own root still maps 0x400000 as it always did",
            super::paging::query(0x400000).is_some(),
        ));
        let heap_still: alloc::vec::Vec<u64> = (0..256).collect();
        out.push((
            "and the kernel's heap is unharmed by any of it",
            heap_still.len() == 256 && heap_still[255] == 255,
        ));
        drop(low_a);
        drop(low_b);
        give_back(cp);
        give_back(dp);
    } else {
        out.push(("two low spaces and two pages could be built", false));
    }

    // Dropping a live space must restore before it frees. Installed through a
    // raw activate rather than `with`, then dropped: the failure this catches
    // is a root freed while the processor walks it, which does not produce a
    // diagnosable fault.
    if let Some(d) = Space::sharing_kernel() {
        crate::cpu::without_interrupts(|| {
            unsafe { super::paging::activate(d.root()) };
            drop(d);
        });
    }
    out.push((
        "and dropping a space that was still installed put it back",
        crate::cpu::read_cr3() & ADDR_MASK == boot,
    ));

    out
}

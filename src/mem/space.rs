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

    /// Find or create the next table down, and answer where it is.
    fn step(&mut self, table: u64, idx: usize) -> Option<u64> {
        let t = unsafe { &mut *(table as *mut [u64; ENTRIES]) };
        let e = t[idx];
        if e & PRESENT != 0 {
            if e & HUGE != 0 {
                // A large page already covers this. Splitting one is
                // `paging::split_large`'s job and it would be splitting a
                // table this space may be *sharing*, so refusing is the only
                // answer that cannot damage the kernel's map.
                return None;
            }
            return Some(e & ADDR_MASK);
        }
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
    /// **Refused below `WINDOW`, and that refusal is the safety argument for
    /// this whole module.** The top-level entries were *copied* from the
    /// kernel, so they point at the kernel's own PDPTs: creating a table under
    /// entry 0 would create it inside the kernel's map, every space would see
    /// it, and the private mapping would not be private at all. Worse, it
    /// would be an edit to live kernel page tables made through a root nobody
    /// thinks of as the kernel's. A space that wants its own low addresses has
    /// to privatise that subtree first, which is a different and larger piece
    /// of work.
    pub fn map_page(&mut self, at: u64, phys: u64, writable: bool, user: bool) -> bool {
        if at < WINDOW || at % 4096 != 0 || phys % 4096 != 0 {
            return false;
        }
        let i4 = ((at >> 39) & 511) as usize;
        if i4 == 0 {
            return false;
        }
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

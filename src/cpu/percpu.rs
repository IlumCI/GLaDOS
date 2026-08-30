//! Per-core storage, reachable in one instruction.
//!
//! A core has no register that says which core it is. The local interrupt
//! controller knows, and asking it is a memory-mapped read, which is fine for
//! a report and far too slow for anything on a hot path: the allocator would
//! be paying more to find out who to bill than to allocate.
//!
//! The architectural answer is a segment base. `GS` has one in long mode, it
//! is per-core because it lives in a per-core register, and `gs:[0]` is a
//! single load. So each core points GS at a small structure of its own and
//! everything that needs to know where it is reads from there.
//!
//! **Nothing may read it before it is set.** GS base is zero at reset, so a
//! read through it lands at address zero, which is deliberately unmapped, and
//! this kernel does not recover from faults. `ARMED` is checked first and is
//! only set once every core that exists has its own block. That check is an
//! atomic load, which is cheap enough for the paths that need it and is the
//! reason the allocator can bill anybody at all.

use core::sync::atomic::{AtomicBool, Ordering};

/// MSR holding the base address `gs:` offsets are added to.
const IA32_GS_BASE: u32 = 0xC000_0101;

/// What every core keeps about itself.
///
/// `repr(C)` and the field order are load-bearing: the offsets are written
/// into inline assembly below, so reordering these silently changes what
/// `cpu_id` and `task` read.
#[repr(C)]
pub struct PerCpu {
    /// Dense core index, the same one `smp::this_cpu` reports.
    pub cpu: u64,
    /// The task this core is running, for anything that needs to attribute
    /// work to somebody. Updated by the scheduler on every switch.
    pub task: u64,
}

static ARMED: AtomicBool = AtomicBool::new(false);

/// Whether per-core storage may be read.
#[inline(always)]
pub fn armed() -> bool {
    ARMED.load(Ordering::Relaxed)
}

/// Give this core its own block and point GS at it.
///
/// Leaked deliberately. The processor holds the address for as long as it
/// runs, and freeing it would hand the block the core reads on every
/// allocation back to the allocator.
/// Room for the same number of cores the scheduler knows about.
const MAX_CPUS: usize = crate::task::MAX_CPUS;

static BLOCKS: crate::sync::Racy<[*mut PerCpu; MAX_CPUS]> =
    crate::sync::Racy::new([core::ptr::null_mut(); MAX_CPUS]);

/// Build a block for every core, on the bootstrap processor.
///
/// Allocating on an application processor before it has an interrupt table is
/// a triple fault waiting to happen, which `gdt::prepare` explains at length.
/// Same rule here, same reason.
pub fn prepare(cores: usize) {
    for cpu in 0..cores.min(MAX_CPUS) {
        let block =
            alloc::boxed::Box::leak(alloc::boxed::Box::new(PerCpu { cpu: cpu as u64, task: 0 }));
        unsafe { BLOCKS.get()[cpu] = block as *mut PerCpu };
    }
}

/// Point this core's GS at the block `prepare` built for it.
pub fn adopt(cpu: usize) -> bool {
    let b = unsafe { BLOCKS.get().get(cpu).copied().unwrap_or(core::ptr::null_mut()) };
    if b.is_null() {
        return false;
    }
    unsafe { super::wrmsr(IA32_GS_BASE, b as u64) };
    true
}

/// Declare that every core has its block. Called once, from the bootstrap
/// processor, after the others are up.
pub fn arm() {
    ARMED.store(true, Ordering::Release);
}

/// This core's index, in one load.
///
/// # Safety
/// Only valid once `armed()`. Callers that cannot guarantee that must use
/// `smp::this_cpu`, which asks the interrupt controller and is always safe.
#[inline(always)]
pub fn cpu_id() -> u64 {
    let v: u64;
    unsafe {
        core::arch::asm!("mov {}, gs:[0]", out(reg) v, options(nostack, preserves_flags));
    }
    v
}

/// The task this core is running.
#[inline(always)]
pub fn task() -> u64 {
    let v: u64;
    unsafe {
        core::arch::asm!("mov {}, gs:[8]", out(reg) v, options(nostack, preserves_flags));
    }
    v
}

/// Record which task this core is now running.
#[inline(always)]
pub fn set_task(index: u64) {
    if !armed() {
        return;
    }
    unsafe {
        core::arch::asm!("mov gs:[8], {}", in(reg) index, options(nostack, preserves_flags));
    }
}

/// Who to bill for work happening right now, or `None` before per-core storage
/// exists.
#[inline(always)]
pub fn billed() -> Option<usize> {
    if !armed() {
        return None;
    }
    Some(task() as usize)
}

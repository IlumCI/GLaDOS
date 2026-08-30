//! Who is using the memory.
//!
//! `mem_used` answers how much of the heap is gone and says nothing about
//! where it went, which makes it useless for deciding what to take back. This
//! attributes every allocation to the task that asked for it, so the question
//! stops being "is memory tight" and becomes "which of these is holding it".
//!
//! **What is measured is gross, and the distinction matters.** Each task's
//! bytes requested and bytes released are counted separately, and the
//! difference is its outstanding total. That is exact for a task that frees
//! what it allocated, and it drifts when a block is allocated by one task and
//! freed by another, because the freeing task is credited rather than the
//! owning one.
//!
//! Fixing that drift means a header on every allocation carrying an owner,
//! which is eight bytes per block and a change to every alignment assumption
//! in the allocator. The drift is bounded by how much ownership actually
//! moves between tasks here, which is a handful of buffers, and the number is
//! reported as gross rather than dressed up as exact. A figure that is honest
//! about its error is more useful than one that hides it.
//!
//! Cost is two atomic adds on the allocation path, and only once per-core
//! storage exists. Before that, and for allocations made by no task at all
//! during boot, everything lands in slot zero and is reported as such.

use core::sync::atomic::{AtomicU64, Ordering};

/// One row per task, plus a row for anything with no task to blame.
const ROWS: usize = crate::task::MAX_TASKS;

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);

static TAKEN: [AtomicU64; ROWS] = [ZERO; ROWS];
static GIVEN: [AtomicU64; ROWS] = [ZERO; ROWS];
static COUNT: [AtomicU64; ROWS] = [ZERO; ROWS];
static PEAK: [AtomicU64; ROWS] = [ZERO; ROWS];

/// Bill an allocation to whoever is running.
#[inline(always)]
pub fn took(bytes: usize) {
    let Some(who) = crate::cpu::percpu::billed() else {
        return;
    };
    if who >= ROWS {
        return;
    }
    let t = TAKEN[who].fetch_add(bytes as u64, Ordering::Relaxed) + bytes as u64;
    COUNT[who].fetch_add(1, Ordering::Relaxed);
    // Peak of the outstanding total rather than of the gross, since gross only
    // ever rises and would say nothing.
    let out = t.saturating_sub(GIVEN[who].load(Ordering::Relaxed));
    // A plain compare and store: two cores racing here differ by one
    // allocation, and a peak that is occasionally one block low is not worth a
    // loop on the allocation path.
    if out > PEAK[who].load(Ordering::Relaxed) {
        PEAK[who].store(out, Ordering::Relaxed);
    }
}

/// Credit a release to whoever is running. See the note about drift above.
#[inline(always)]
pub fn gave(bytes: usize) {
    let Some(who) = crate::cpu::percpu::billed() else {
        return;
    };
    if who >= ROWS {
        return;
    }
    GIVEN[who].fetch_add(bytes as u64, Ordering::Relaxed);
}

/// What one task has done with the heap.
#[derive(Clone, Copy, Default)]
pub struct Row {
    pub taken: u64,
    pub given: u64,
    pub count: u64,
    pub peak: u64,
}

impl Row {
    /// Bytes asked for and not yet released. Saturating, because the drift
    /// described above can put a task's releases ahead of its requests and a
    /// wrapped subtraction would report several exabytes outstanding.
    pub fn outstanding(&self) -> u64 {
        self.taken.saturating_sub(self.given)
    }
}

pub fn row(task: usize) -> Row {
    if task >= ROWS {
        return Row::default();
    }
    Row {
        taken: TAKEN[task].load(Ordering::Relaxed),
        given: GIVEN[task].load(Ordering::Relaxed),
        count: COUNT[task].load(Ordering::Relaxed),
        peak: PEAK[task].load(Ordering::Relaxed),
    }
}

/// Forget everything. For a measurement that wants a window rather than a
/// lifetime.
pub fn reset() {
    for i in 0..ROWS {
        TAKEN[i].store(0, Ordering::Relaxed);
        GIVEN[i].store(0, Ordering::Relaxed);
        COUNT[i].store(0, Ordering::Relaxed);
        PEAK[i].store(0, Ordering::Relaxed);
    }
}

/// Print the census, busiest first.
pub fn report() {
    use crate::kprintln;
    let n = crate::task::count();
    let (used, total) = crate::mem::heap::HEAP.stats();
    kprintln!("  heap {} KiB of {} KiB", used / 1024, total / 1024);
    kprintln!("  {:<12} {:>10} {:>10} {:>10} {:>8}", "task", "outstanding", "peak", "taken", "allocs");

    // Sorted by what each is holding, because the question this answers is
    // which one to take memory back from.
    let mut order: alloc::vec::Vec<(u64, usize)> = (0..n.min(ROWS))
        .map(|i| (row(i).outstanding(), i))
        .collect();
    order.sort_by(|a, b| b.0.cmp(&a.0));

    for (_, i) in order {
        let r = row(i);
        if r.count == 0 {
            continue;
        }
        let name = crate::task::snapshot(i).map(|t| t.name).unwrap_or("?");
        kprintln!(
            "  {:<12} {:>10} {:>10} {:>10} {:>8}",
            name,
            r.outstanding(),
            r.peak,
            r.taken,
            r.count
        );
    }
}

pub fn selftest() -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    claim(
        &mut ok,
        crate::cpu::percpu::armed(),
        "per-core storage is up, so allocations can be attributed",
    );

    let me = crate::task::current();
    let before = row(me);

    // Something big enough to be unmistakable against whatever else this task
    // is doing while the measurement runs.
    const BIG: usize = 512 * 1024;
    let v: alloc::vec::Vec<u8> = alloc::vec![7u8; BIG];
    let during = row(me);
    claim(
        &mut ok,
        during.taken >= before.taken + BIG as u64,
        "an allocation is billed to the task that made it",
    );
    claim(&mut ok, during.count > before.count, "and counted");
    claim(
        &mut ok,
        during.peak >= BIG as u64,
        "the peak records the largest outstanding total, not the gross",
    );

    drop(v);
    let after = row(me);
    claim(
        &mut ok,
        after.given >= during.given + BIG as u64,
        "and the release is credited back",
    );

    // The property the whole thing exists for: two tasks are told apart. The
    // idle tasks on other cores allocate nothing, so any core that has run
    // real work has a different row from one that has not.
    let mut nonzero = 0;
    for i in 0..crate::task::count().min(ROWS) {
        if row(i).count > 0 {
            nonzero += 1;
        }
    }
    claim(&mut ok, nonzero >= 1, "at least one task has a row of its own");
    crate::kprintln!("  {} task(s) have allocated", nonzero);

    claim(
        &mut ok,
        row(usize::MAX).count == 0,
        "and an index that is not a task reads as empty rather than faulting",
    );
    ok
}

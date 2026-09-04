//! Physical ranges a fixed-address image may be placed at.
//!
//! **Why this exists at all.** `linux::load` refused `ET_EXEC` with "a single
//! address space has no free range to promise it", and that was true as a
//! statement about what the kernel *knew* rather than about the machine. A
//! non-PIE binary insists on its own addresses -- classically `0x400000` --
//! and in an identity map those are real physical bytes, so honouring one
//! means being able to answer "does anything own four megabytes at four
//! megabytes". Nothing here could answer that, so the honest response was to
//! decline.
//!
//! It matters more than it sounds. Nearly every prebuilt static binary in the
//! world is non-PIE, busybox's own included, so the refusal was not a corner
//! case: it was most of the software this loader exists to run.
//!
//! **Why the frame allocator cannot answer it.** `EarlyFrames` is a bump
//! allocator that walks the UEFI map forward and never rewinds, and its own
//! doc comment explains why rewinding would be worse than the bug it has. By
//! the time boot is done its cursor is past the heap, three hundred megabytes
//! up, and everything behind it reads as unavailable -- including large
//! conventional regions it merely stepped over looking for one span big enough
//! for the heap. Asking it about `0x400000` gets "no" for a region that is in
//! fact untouched.
//!
//! So the free set is computed the other way round: every conventional region
//! the firmware declared, minus the handful of ranges boot actually took. The
//! allocator records its handouts, which it can do exactly because it never
//! frees -- a bump allocator's history is a short list.
//!
//! **The snapshot is taken once, after the heap.** The UEFI memory map lives
//! in memory boot services owned and nothing promises it survives, so this
//! copies what it needs while the map is still being held, and the map is
//! never read again.

use super::PAGE_SIZE;
use crate::sync::Racy;

/// Free conventional ranges, and what has been claimed out of them.
///
/// Sized for a real firmware map rather than QEMU's: OVMF reports a handful of
/// conventional regions and the GF63 reports rather more, so the table is
/// generous and the count is checked. Overflow drops the *tail*, which is the
/// safe direction -- a range that is free and not listed is refused, and a
/// range that is listed is genuinely free.
const MAX_RANGES: usize = 48;

#[derive(Clone, Copy)]
struct Range {
    at: u64,
    end: u64,
}

struct Free {
    ranges: [Range; MAX_RANGES],
    n: usize,
    /// What a guest has taken out of the free set and not yet given back.
    ///
    /// One entry, because one guest runs at a time -- the same assumption the
    /// syscall stack and `SPACE` already rest on, and the one that has to stop
    /// being true before threads arrive. Stated here rather than left implicit
    /// so it turns up in the same grep as the others.
    claim: Option<Range>,
    /// Whether `snapshot` ever ran. A machine that never called it must refuse
    /// everything rather than treat an empty table as "nothing is free", which
    /// is the same answer for the opposite reason.
    ready: bool,
}

static FREE: Racy<Free> = Racy::new(Free {
    ranges: [Range { at: 0, end: 0 }; MAX_RANGES],
    n: 0,
    claim: None,
    ready: false,
});

/// Record the conventional ranges the firmware declared, less what boot took.
///
/// # Safety
/// `mmap` must point at `mmap_size` bytes of UEFI memory descriptors with the
/// firmware's `desc_size` stride, and `taken` must be every range the early
/// allocator handed out.
pub unsafe fn snapshot(
    mmap: *const u8,
    mmap_size: usize,
    desc_size: usize,
    taken: &[(u64, u64)],
) {
    let f = unsafe { FREE.get() };
    f.n = 0;
    f.claim = None;
    f.ready = true;
    if desc_size == 0 {
        return;
    }
    for i in 0..mmap_size / desc_size {
        let d = unsafe { &*(mmap.add(i * desc_size) as *const crate::uefi::MemoryDescriptor) };
        if !d.is_conventional() {
            continue;
        }
        // The low megabyte is excluded for the reasons `frame::MIN_PHYS` gives
        // -- the EBDA, assorted legacy structures, and the AP trampoline that
        // has to live in real-mode reach. A guest asking for an address down
        // there is asking for the thing that starts the other cores.
        let mut at = d.phys_start.max(0x10_0000);
        let end = d.phys_start + d.num_pages * PAGE_SIZE;
        if at >= end {
            continue;
        }
        // Subtract the handouts. They are sorted by construction (a bump
        // allocator only moves forward) but that is not relied on: each region
        // is walked against every handout, which is a few dozen comparisons
        // once per boot.
        loop {
            let mut cut = None;
            for &(t_at, t_len) in taken {
                let t_end = t_at + t_len;
                if t_at < end && t_end > at {
                    cut = Some((t_at, t_end));
                    break;
                }
            }
            match cut {
                None => {
                    push(f, at, end);
                    break;
                }
                Some((t_at, t_end)) => {
                    if t_at > at {
                        push(f, at, t_at);
                    }
                    if t_end >= end {
                        break;
                    }
                    at = t_end;
                }
            }
        }
    }
}

fn push(f: &mut Free, at: u64, end: u64) {
    if end <= at || f.n >= MAX_RANGES {
        return;
    }
    f.ranges[f.n] = Range { at, end };
    f.n += 1;
}

/// Take `len` bytes at exactly `at`, or answer why not.
///
/// Rounded outward to whole pages, because the caller is going to mark page
/// rights on it and a permission is a property of a page.
pub fn claim(at: u64, len: usize) -> Result<(), &'static str> {
    let f = unsafe { FREE.get() };
    if !f.ready {
        return Err("the free physical ranges were never recorded");
    }
    if f.claim.is_some() {
        return Err("another fixed-address image already holds a range");
    }
    let start = at & !(PAGE_SIZE - 1);
    let Some(sum) = at.checked_add(len as u64) else {
        return Err("the range runs past the end of the address space");
    };
    let end = sum.div_ceil(PAGE_SIZE) * PAGE_SIZE;
    if !f.ranges[..f.n].iter().any(|r| start >= r.at && end <= r.end) {
        // Deliberately one message rather than "not conventional" versus
        // "already taken": the two are indistinguishable from here and
        // guessing which would be inventing a reason.
        return Err("that physical range is not free on this machine");
    }
    f.claim = Some(Range { at: start, end });
    Ok(())
}

/// Give a claimed range back. Answers false when nothing held it.
pub fn release(at: u64) -> bool {
    let f = unsafe { FREE.get() };
    match f.claim {
        Some(r) if r.at == (at & !(PAGE_SIZE - 1)) => {
            f.claim = None;
            true
        }
        _ => false,
    }
}

/// Free conventional bytes, and the largest single run, for a report.
pub fn totals() -> (u64, u64) {
    let f = unsafe { FREE.get() };
    let mut total = 0;
    let mut best = 0;
    for r in &f.ranges[..f.n] {
        total += r.end - r.at;
        best = best.max(r.end - r.at);
    }
    (total, best)
}

/// Print the table, which is the only way to find out whether a given address
/// is placeable on a machine that cannot be debugged any other way.
pub fn report() {
    let f = unsafe { FREE.get() };
    if !f.ready {
        crate::kprintln!("  (never recorded)");
        return;
    }
    for r in &f.ranges[..f.n] {
        crate::kprintln!(
            "  {:#012x}..{:#012x}  {} MiB",
            r.at,
            r.end,
            (r.end - r.at) / 1024 / 1024
        );
    }
    match f.claim {
        Some(c) => crate::kprintln!("  claimed {:#x}..{:#x}", c.at, c.end),
        None => crate::kprintln!("  {} range(s), nothing claimed", f.n),
    }
}

/// What `diag mem` asks of the placement table.
///
/// Every claim here is about the arithmetic rather than about this machine's
/// map, so they hold on the GF63 too -- which is the point, since the map is
/// exactly what cannot be reproduced here.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out = alloc::vec::Vec::new();
    let f = unsafe { FREE.get() };
    let saved = (f.ranges, f.n, f.claim, f.ready);

    // A synthetic map: one region with a hole punched in the middle, which is
    // the shape a handout makes and the one the subtraction has to get right.
    f.n = 0;
    f.claim = None;
    f.ready = true;
    push(f, 0x10_0000, 0x40_0000);
    push(f, 0x50_0000, 0x90_0000);

    out.push((
        "a range inside a free run is claimable",
        claim(0x60_0000, 0x1000).is_ok(),
    ));
    out.push((
        "and a second claim is refused while the first stands",
        claim(0x20_0000, 0x1000).is_err(),
    ));
    out.push(("releasing the claim gives it back", release(0x60_0000)));
    out.push((
        "releasing something nobody claimed is false rather than a panic",
        !release(0x60_0000),
    ));
    out.push((
        "a range in the hole between two runs is refused",
        claim(0x45_0000, 0x1000).is_err(),
    ));
    out.push((
        "a range that starts inside a run and ends past it is refused",
        claim(0x3f_0000, 0x20_000).is_err(),
    ));
    out.push((
        "an unaligned request is rounded outward, so it is judged on whole pages",
        {
            let _ = release(0);
            let r = claim(0x8f_f000 + 1, 0x1000);
            let bad = r.is_err();
            let _ = release(0x8f_f000);
            bad
        },
    ));
    out.push((
        "a length that overflows the address space is refused rather than wrapping",
        claim(u64::MAX - 4095, 1 << 20).is_err(),
    ));

    // And with nothing recorded, everything is refused: an empty table must
    // not read as "the whole machine is free".
    f.ready = false;
    out.push((
        "a machine that never recorded its map places nothing",
        claim(0x60_0000, 0x1000).is_err(),
    ));

    f.ranges = saved.0;
    f.n = saved.1;
    f.claim = saved.2;
    f.ready = saved.3;
    out
}

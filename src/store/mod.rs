//! Persistent storage: block layer, hashing, and the checkpoint store.

pub mod block;
pub mod cas;
pub mod sha256;

use crate::sync::Racy;

static STORE: Racy<Option<cas::Store>> = Racy::new(None);

pub fn mounted() -> bool {
    unsafe { STORE.get().is_some() }
}

pub fn with<R>(f: impl FnOnce(&mut cas::Store) -> R) -> Option<R> {
    unsafe { STORE.get().as_mut().map(f) }
}

/// Minimum region worth formatting: two superblocks plus room to be useful.
pub const MIN_REGION_BLOCKS: u64 = 2048;

#[derive(Debug)]
pub enum InitError {
    NoDevice,
    NoRoom,
    Store(cas::Error),
}

/// Find unclaimed space, format it, and mount it.
///
/// This is the one place NVMe writes get unlocked, and only after
/// `find_store_region` has named a target: either a partition tagged with the
/// GLaDOS type GUID, or -- on a bare image with no partition table -- unclaimed
/// space past every partition. `Store::format` then re-checks that the region
/// is inside our own partition or overlaps nothing at all.
///
/// On a disk fully allocated to Windows with no GLaDOS partition, there is no
/// such region and this fails. That is the intended outcome, not an
/// inconvenience.
pub fn init() -> Result<(u64, u64), InitError> {
    if !crate::dev::nvme::present() {
        return Err(InitError::NoDevice);
    }
    let (start, blocks) = cas::find_store_region(MIN_REGION_BLOCKS).ok_or(InitError::NoRoom)?;

    // Check before unlocking, not after. `format` re-checks internally, but it
    // needs writes already enabled to lay down a superblock -- so unlocking
    // first meant a region that failed the check returned an error with the
    // disk left writable. Verifying here keeps the unlock from happening at all
    // in the case that matters most.
    cas::verify_region_safe(start, blocks).map_err(InitError::Store)?;

    crate::dev::nvme::unlock_writes(0xD15EA5E);
    match cas::Store::format(start, blocks) {
        Ok(s) => {
            unsafe { *STORE.get() = Some(s) };
            Ok((start, blocks))
        }
        Err(e) => {
            // Any other failure -- I/O, a short region -- puts the lock back
            // too. Leaving it open on the way out of an error path is how a
            // safety mechanism becomes decorative.
            crate::dev::nvme::lock_writes();
            Err(InitError::Store(e))
        }
    }
}

/// Allow writes to an already-mounted store.
///
/// Mounting is deliberately read-only, so a store attached at boot cannot be
/// written to until someone asks. This re-runs the same region check `format`
/// uses rather than trusting that mounting implied permission.
pub fn unlock() -> Result<(u64, u64), cas::Error> {
    let Some((start, blocks)) = with(|s| (s.sb.region_start, s.sb.region_blocks)) else {
        return Err(cas::Error::NotFormatted);
    };
    cas::verify_region_safe(start, blocks)?;
    crate::dev::nvme::unlock_writes(0xD15EA5E);
    Ok((start, blocks))
}

/// Attach to an already-formatted region without writing anything.
pub fn mount(region_start: u64) -> Result<(), cas::Error> {
    let s = cas::Store::mount(region_start)?;
    unsafe { *STORE.get() = Some(s) };
    Ok(())
}

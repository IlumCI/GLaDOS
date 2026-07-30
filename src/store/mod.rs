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
/// `find_free_region` has confirmed the target overlaps no partition. On a disk
/// fully allocated to Windows there is no such region and this fails, which is
/// the intended outcome rather than an inconvenience.
pub fn init() -> Result<(u64, u64), InitError> {
    if !crate::dev::nvme::present() {
        return Err(InitError::NoDevice);
    }
    let (start, blocks) = cas::find_free_region(MIN_REGION_BLOCKS).ok_or(InitError::NoRoom)?;

    crate::dev::nvme::unlock_writes(0xD15EA5E);

    let s = cas::Store::format(start, blocks).map_err(InitError::Store)?;
    unsafe { *STORE.get() = Some(s) };
    Ok((start, blocks))
}

/// Attach to an already-formatted region without writing anything.
pub fn mount(region_start: u64) -> Result<(), cas::Error> {
    let s = cas::Store::mount(region_start)?;
    unsafe { *STORE.get() = Some(s) };
    Ok(())
}

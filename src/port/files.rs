//! The bytes of a file, borrowed rather than copied.
//!
//! A ported program's data is large and read constantly -- a WAD is four
//! megabytes and a renderer asks it for a texture several times a frame. So
//! the shape that matters here is not "how do I read a file", it is **who owns
//! the bytes**.
//!
//! `sysbox::read_blob` clones on every call. That is correct for its callers,
//! which read a configuration file once, and catastrophic for this one: a
//! lump lookup would allocate four megabytes each time it was asked.
//!
//! So a ported program's data is read **once, before `ExitBootServices`**, by
//! the same path the model takes, and lives in the firmware's LoaderData pool
//! for the life of the machine. `uefi::read_file` already does the hard part
//! -- it tries `allocate_pool` and falls back to `allocate_pages`, because
//! firmware pool allocators refuse large requests and a 1.8 GB checkpoint is
//! what taught it that. Nothing is freed, and nothing is copied, so the cost
//! to the heap is zero.
//!
//! What this module is, therefore, is a registry: a place for `main` to put
//! the blobs it read at boot, and a way for a port to ask for one by name.

use alloc::vec::Vec;

/// One file, as bytes that outlive everything that looks at them.
///
/// `'static` because that is the truth: these come from a pool the firmware
/// allocated and nothing releases. Saying so lets a port hold a slice of a
/// texture for as long as it likes without a lifetime threaded through its
/// whole renderer, which is the kind of thing that turns a port into a
/// rewrite.
struct Entry {
    name: &'static str,
    bytes: &'static [u8],
}

static FILES: crate::sync::Racy<Vec<Entry>> = crate::sync::Racy::new(Vec::new());

/// Register a blob read at boot.
///
/// Called from `main`, before the shell exists. Not public beyond the crate:
/// a port consumes files, it does not publish them.
pub(crate) fn provide(name: &'static str, bytes: &'static [u8]) {
    let v = unsafe { &mut *FILES.get() };
    if v.iter().any(|e| e.name == name) {
        return;
    }
    v.push(Entry { name, bytes });
}

/// The bytes of a named file, or `None` if it was not on the boot volume.
///
/// Absence is the ordinary case and is not an error: a machine with no WAD on
/// its ESP is a machine that cannot run DOOM, and the honest response is for
/// the command to say which file it wanted and where it looked.
pub fn get(name: &str) -> Option<&'static [u8]> {
    let v = unsafe { &*FILES.get() };
    v.iter().find(|e| e.name == name).map(|e| e.bytes)
}

/// What is available, for a command that wants to say so.
pub fn names() -> Vec<&'static str> {
    let v = unsafe { &*FILES.get() };
    v.iter().map(|e| e.name).collect()
}

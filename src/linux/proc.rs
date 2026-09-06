//! The handful of `/proc` files this machine can answer honestly.
//!
//! **Not a filesystem, and the difference is the whole design.** Every other
//! path in this system resolves to a content-addressed blob: the inode is a
//! hash of the path, the size comes from `blob_len`, and a snapshot commits
//! the lot to one root. A `/proc` file has none of that. It resolves to a
//! function, its content depends on when it is read, and it has no hash. So
//! this is a short table of answers consulted *before* the store, and nothing
//! here is ever written, listed by `sysbox`, or included in a snapshot.
//!
//! What makes it cheap is a property `fs.rs` already had for its own reasons:
//! an open file holds its whole contents, so a synthetic file is only a
//! different way of filling that `Vec`. There is no second kind of descriptor,
//! no second `read` path, and `lseek` works on these for free.
//!
//! ### The rule that keeps this from becoming fiction
//!
//! **A field this machine does not know means the file does not exist.**
//!
//! That is the one decision worth arguing. `/proc/stat`, `/proc/loadavg` and
//! `/proc/cpuinfo` are text formats with no way to say "I do not know": omit a
//! field and a parser breaks, invent one and it gets believed. There is no
//! `Option` in a text file. A missing `/proc/stat` sends a program down a
//! fallback it already has, and a fabricated one sends it down a confident
//! wrong path -- which is the more expensive of the two and is the mistake
//! this tree has paid for repeatedly with plausible-looking numbers.
//!
//! So the table is short on purpose and grows when something actually asks,
//! which is the `-ENOSYS` discipline applied to paths instead of call numbers.
//!
//! ### What is deliberately absent
//!
//! `/proc/meminfo` and `/proc/cpuinfo`, because `sysinfo` and `CPUID` answer
//! those without inventing the dozen fields neither can fill.
//! `/proc/uptime`, because its second field is idle time and nothing here
//! measures it. `/proc/stat`, for the same reason several times over. And
//! every other process, because there is one.

use super::syscall;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The directories that exist so a path can be walked to a file in them.
///
/// `/proc/self` and not `/proc/1`: there is one process, `getpid` answers 1,
/// and both spellings would be the same thing said twice. `self` is what
/// programs actually use.
pub fn is_dir(path: &str) -> bool {
    matches!(path, "/proc" | "/proc/self")
}

/// Whether this path is one this module answers at all.
///
/// **A property of the path and never of what is running.** The first version
/// asked `link` and `read`, which consult the live guest, so `/proc/self/exe`
/// was claimed while a guest ran and unclaimed otherwise -- a table that
/// changes shape depending on the weather. `diag linux` caught it, running
/// with no guest installed, and it would have been much worse the other way
/// round: a listing offering a name that `openat` then routed to the store,
/// which answers `ENOENT` about a file the same command had just printed.
///
/// Producing the *content* is still allowed to fail, and that is `read`'s
/// answer to give.
pub fn claims(path: &str) -> bool {
    is_dir(path) || is_file(path)
}

/// The synthetic files, as a table rather than as four scattered matches.
pub fn is_file(path: &str) -> bool {
    matches!(
        path,
        "/proc/self/exe" | "/proc/self/cmdline" | "/proc/self/environ" | "/proc/self/maps"
    )
}

/// Where a synthetic symlink points.
///
/// **`/proc/self/exe` is the reason this module exists.** There is no syscall
/// that answers "where is my own binary", so a program that wants to find its
/// data directory beside itself has this and nothing else. It is a symlink on
/// Linux, so `readlink` is the usual way in, and `open` on it gives the binary
/// -- both work here, because `read` below resolves it to the real path.
pub fn link(path: &str) -> Option<String> {
    match path {
        "/proc/self/exe" => syscall::guest_image(),
        _ => None,
    }
}

/// The bytes of a synthetic file, or nothing when this machine cannot say.
pub fn read(path: &str) -> Option<Vec<u8>> {
    match path {
        // Opening the link opens what it points at, which is what Linux does
        // and what a program expecting to read its own image relies on.
        "/proc/self/exe" => {
            let real = syscall::guest_image()?;
            crate::sysbox::read_blob(&real)
        }
        // Both of these are NUL-separated with a trailing NUL, which is the
        // detail that matters: a reader splits on the separator and a missing
        // final one silently drops the last entry.
        "/proc/self/cmdline" => Some(joined(&syscall::guest_argv())),
        "/proc/self/environ" => Some(joined(&syscall::environ())),
        "/proc/self/maps" => Some(maps().into_bytes()),
        _ => None,
    }
}

fn joined(parts: &[String]) -> Vec<u8> {
    let mut out = Vec::new();
    for p in parts {
        out.extend_from_slice(p.as_bytes());
        out.push(0);
    }
    out
}

/// What the guest owns, in the shape `/proc/self/maps` has.
///
/// **Truthful and therefore strange**, which is worth knowing before reading
/// it. A guest here shares one address space with the kernel, so these are the
/// real addresses of real pages, and they sit wherever the heap put them
/// rather than at the tidy `0x400000` a reader might expect. What is *not*
/// listed is everything the guest does not own, which is correct from the
/// guest's side: those pages carry no U bit and it cannot reach them.
///
/// The rights come from the page tables rather than from what was asked for at
/// `mmap`, because `mprotect` moves them afterwards and a linker spends its
/// last act doing exactly that.
fn maps() -> String {
    let mut out = String::new();
    for (at, len, name) in syscall::guest_regions() {
        if len == 0 {
            continue;
        }
        let p = crate::mem::paging::query(at);
        let r = if p.is_some_and(|p| p.present) { 'r' } else { '-' };
        let w = if p.is_some_and(|p| p.write) { 'w' } else { '-' };
        let x = if p.is_some_and(|p| p.exec) { 'x' } else { '-' };
        // Always private: nothing here is shared, and `mmap` refuses a shared
        // writable file mapping for a reason it states.
        let ino = if name.starts_with('/') { super::fs::ino_of(&name) } else { 0 };
        // The end is rounded up to a page, which is *more* truthful rather
        // than less: rights are applied per page, so the guest owns the whole
        // of its last partial one. It also matters to whoever reads this --
        // every parser of this file was written against a kernel whose ranges
        // are page-aligned, and an image whose span ends mid-page (busybox's
        // does, at +0xbda78) would be the first one any of them had seen.
        let end = (at + len as u64).next_multiple_of(4096);
        out.push_str(&format!(
            "{:012x}-{:012x} {}{}{}p 00000000 00:00 {} {}\n",
            at,
            end,
            r,
            w,
            x,
            ino,
            name
        ));
    }
    out
}

/// What a listing of one of the synthetic directories answers.
///
/// Name, whether it is a directory, and size, which is the shape `Dir` wants.
/// The size is always zero, for two reasons that agree: `getdents64` ignores
/// the field, and Linux reports zero for nearly everything under `/proc`
/// anyway. Filling it truthfully would mean generating every file to list the
/// directory, and for `exe` that is the whole binary.
pub fn entries(dir: &str) -> Vec<(String, bool, usize)> {
    let file = |n: &str| (n.to_string(), false, 0usize);
    match dir {
        "/proc" => alloc::vec![("self".to_string(), true, 0usize)],
        "/proc/self" => {
            alloc::vec![file("exe"), file("cmdline"), file("environ"), file("maps")]
        }
        _ => Vec::new(),
    }
}

/// What `diag linux` asks of the table.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();

    out.push((
        "the synthetic directories exist so a path can be walked into them",
        is_dir("/proc") && is_dir("/proc/self") && !is_dir("/proc/self/exe"),
    ));
    out.push((
        "and nothing outside them is claimed, so the store still answers for every other path",
        !claims("/proc/meminfo") && !claims("/proc/stat") && !claims("/tmp/x") && !claims("/"),
    ));
    // The rule the module exists to hold. `/proc/cpuinfo` and friends are
    // formats whose honest content is mostly "I do not know", and a text file
    // has no way to say that.
    out.push((
        "a file whose fields this machine cannot fill is absent rather than invented",
        read("/proc/cpuinfo").is_none()
            && read("/proc/uptime").is_none()
            && read("/proc/loadavg").is_none(),
    ));
    out.push((
        "only /proc/self/exe is a link, since it is the only one with no other way in",
        is_file("/proc/self/exe") && link("/proc/self/maps").is_none(),
    ));
    // What the table says exists cannot depend on whether a guest is running,
    // which is exactly the bug the first version had: `claims` consulted the
    // live space, so this suite -- which runs with none -- saw a `/proc/self`
    // holding three of the four entries it had just listed.
    out.push((
        "and what is claimed is a property of the path rather than of the moment",
        claims("/proc/self/exe") && syscall::guest_image().is_none(),
    ));

    let joined_pair = joined(&alloc::vec!["ab".to_string(), "c".to_string()]);
    out.push((
        "a NUL-separated list ends with one, or a reader drops its last entry",
        joined_pair == alloc::vec![b'a', b'b', 0, b'c', 0],
    ));
    out.push((
        "and an empty list is empty rather than a single separator",
        joined(&[]).is_empty(),
    ));

    out.push((
        "every name a listing offers is a path that answers",
        entries("/proc/self")
            .iter()
            .all(|(n, is_dir, _)| !is_dir && claims(&format!("/proc/self/{}", n)))
            && entries("/proc").len() == 1
            && entries("/proc")[0].1,
    ));
    out.push((
        "a directory nobody synthesised lists nothing rather than something empty-looking",
        entries("/tmp").is_empty() && entries("/proc/self/exe").is_empty(),
    ));
    out
}

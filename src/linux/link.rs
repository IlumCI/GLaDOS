//! Symbolic links: the guest's own, and a path walk that follows a link
//! wherever it is.
//!
//! **The layer had no links at all.** The root image carries the ones its
//! packages ship and its reader follows them, but only inside the image: a
//! link the guest made could not exist (`symlink` was `ENOSYS`), and a path
//! through an image link into something the guest had written could not be
//! found -- `/lib` is `usr/lib` in every Arch root, so a library pacman put in
//! `/usr/lib` was missing as `/lib/...`. Nearly every package ships
//! `libfoo.so -> libfoo.so.1`, so a package manager in a guest needs both.
//!
//! A link the guest makes is a blob under `LINKS` holding its target, beside
//! the layer the way whiteouts are (`ns::GONE`): the store has files and
//! directories and nothing else, and teaching it a third kind of node would
//! change every walk of it for the sake of one owner.
//!
//! `walk` is the one answer to "where does this path really go", and
//! `fs::resolve` runs it over every component but the last. The last is the
//! caller's: `open` and `stat` follow it, `lstat`, `readlink`, `unlink` and
//! `rename` act on the link itself, which is the line Linux draws.

use crate::sysbox;
use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Where the guest's own links live, by guest path.
pub const LINKS: &str = "/linux/link";

/// Linux's own bound on links followed in one lookup; past it is `ELOOP`.
const MAX_HOPS: usize = 40;

fn store(guest: &str) -> String {
    format!("{}{}", LINKS, guest)
}

/// The target of a link the guest made at this path.
pub fn own(guest: &str) -> Option<String> {
    if guest == "/" {
        return None;
    }
    sysbox::read_blob(&store(guest)).map(|b| String::from_utf8_lossy(&b).into_owned())
}

/// Make a link. The caller has checked the path is free and writable.
pub fn make(guest: &str, target: &str) -> bool {
    super::ns::unwhite(guest);
    sysbox::write_blob(&store(guest), Vec::from(target.as_bytes()))
}

/// Remove the guest's link at this path, answering whether there was one.
pub fn remove(guest: &str) -> bool {
    own(guest).is_some() && sysbox::detach(&store(guest))
}

/// Move the guest's link from one path to another.
pub fn rename(from: &str, to: &str) -> bool {
    match own(from) {
        Some(t) => {
            remove(from);
            make(to, &t)
        }
        None => false,
    }
}

/// The names of the guest's links in a directory, for a listing.
pub fn names_in(dir: &str) -> Vec<(String, usize)> {
    let d = if dir == "/" { String::from(LINKS) } else { store(dir) };
    sysbox::listing(&d)
        .into_iter()
        .filter(|(_, is_dir, _)| !is_dir)
        .map(|(n, _, len)| (n, len))
        .collect()
}

/// Whether the layer has something real at this path, which hides a link the
/// image has there: a guest that replaced `/lib` with a directory has a
/// directory.
fn shadowed(guest: &str) -> bool {
    super::ns::blob_len(guest).is_some() || super::ns::whited_out(guest) || (super::ns::layered() && super::ns::is_dir(guest) && !super::disk::image_lookup(guest, false).is_some_and(|n| n.is_dir))
}

fn components(path: &str) -> VecDeque<String> {
    path.split('/').filter(|c| !c.is_empty() && *c != ".").map(String::from).collect()
}

fn joined(parts: &[String]) -> String {
    let mut s = String::from("/");
    s.push_str(&parts.join("/"));
    s
}

/// Where `path` (absolute, already normalised) really goes, every link on the
/// way followed -- the last component's only when `last` is set. `Err` is
/// the errno: `ELOOP` past `MAX_HOPS`.
///
/// Links are looked for in the guest's layer first, then in the image, and
/// the earliest one on the path is followed; a relative target is read
/// against the directory the link is in and `..` takes a component back, the
/// way the kernel this imitates does it.
pub fn walk(path: &str, last: bool) -> Result<String, u64> {
    let mut parts: Vec<String> = components(path).into_iter().collect();
    for _ in 0..=MAX_HOPS {
        if parts.is_empty() {
            return Ok(String::from("/"));
        }
        let limit = if last { parts.len() } else { parts.len() - 1 };
        // The guest's own links, prefix by prefix: a sysbox read each, so
        // cheap, and they win over anything the image says at the same path.
        let mut found: Option<(usize, String)> = None;
        for i in 1..=limit {
            if let Some(t) = own(&joined(&parts[..i])) {
                found = Some((i, t));
                break;
            }
        }
        // The image's, in one walk, and only up to where a layer link was.
        let full = joined(&parts);
        if let Some((j, t)) = super::disk::image_first_link(&full, last) {
            if found.as_ref().is_none_or(|(i, _)| j < *i) && !shadowed(&joined(&parts[..j])) {
                found = Some((j, t));
            }
        }
        let Some((i, target)) = found else { return Ok(full) };
        // Replace the link's component with its target, keep the rest.
        let rest: Vec<String> = parts[i..].to_vec();
        let mut next: Vec<String> = if target.starts_with('/') { Vec::new() } else { parts[..i - 1].to_vec() };
        for c in target.split('/') {
            match c {
                "" | "." => {}
                ".." => {
                    next.pop();
                }
                c => next.push(String::from(c)),
            }
        }
        next.extend(rest);
        parts = next;
    }
    Err(super::syscall::ELOOP)
}

/// `walk` with the last component followed, and the path unchanged when the
/// walk fails: the caller's own lookup then reports what is missing.
pub fn follow(path: String) -> String {
    walk(&path, true).unwrap_or(path)
}

/// Whether the path names a link -- the guest's, or the image's not hidden by
/// the layer -- without following it.
pub fn is_link(guest: &str) -> bool {
    own(guest).is_some()
        || (!shadowed(guest) && matches!(super::disk::link_target(guest), Some(Some(_))))
}

/// What `diag linux` can check with no guest and no image.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    let base = "/tmp/.linkcheck";
    let _ = sysbox::detach(&store(base));
    out.push(("a path with no links walks to itself", walk("/tmp/.linkcheck/a/b", true).as_deref() == Ok("/tmp/.linkcheck/a/b")));
    make(&format!("{}/rel", base), "x/y");
    out.push(("a relative link is read against its own directory", walk("/tmp/.linkcheck/rel/z", false).as_deref() == Ok("/tmp/.linkcheck/x/y/z")));
    make(&format!("{}/abs", base), "/etc");
    out.push(("an absolute one starts again at the root", walk("/tmp/.linkcheck/abs/passwd", false).as_deref() == Ok("/etc/passwd")));
    make(&format!("{}/up", base), "../../usr");
    out.push(("and `..` in a target takes a component back", walk("/tmp/.linkcheck/up/lib", false).as_deref() == Ok("/usr/lib")));
    out.push(("the last component is followed only when asked", walk("/tmp/.linkcheck/abs", false).as_deref() == Ok("/tmp/.linkcheck/abs") && walk("/tmp/.linkcheck/abs", true).as_deref() == Ok("/etc")));
    make(&format!("{}/loop", base), "loop");
    out.push(("a link to itself is ELOOP, not a hang", walk("/tmp/.linkcheck/loop", true) == Err(super::syscall::ELOOP)));
    out.push(("a listing names the links in a directory", names_in(base).len() == 4));
    out.push(("and removing one forgets it", remove(&format!("{}/loop", base)) && own(&format!("{}/loop", base)).is_none()));
    let _ = sysbox::detach(&store(base));
    out
}

//! The namespace as a guest sees it: a writable layer over the root image.
//!
//! **With no root image mounted, this is the identity**, and a guest path is a
//! namespace path exactly as it always was -- so every fixture, `diag linux`
//! and a session started the old way mean what they meant.
//!
//! **With one mounted, the guest's writes go to a subtree of their own**,
//! `/linux/root`, and never to the kernel's namespace at the same path. That
//! is not tidiness. The kernel keeps its own `/etc` (the operator's grants
//! among it), `/ai` and `/linux/bin` -- the last being the set of programs the
//! model may invoke -- and a guest free to write `/etc` or `/linux/bin` would
//! be editing the machine's permissions and the model's action vocabulary
//! from ring 3. `/tmp` and `/run` are shared, as they are meant to be: `/tmp`
//! is how files cross between the shell and a guest, and `/run` is where the
//! display server's socket lives.
//!
//! Reads look here first and fall through to the image (`disk::under` makes
//! that decision); writes land here, copying a file up from the image first
//! when it is opened for anything but a truncation. Deleting a file the image
//! has leaves a **whiteout** under `/linux/gone`, because the image cannot
//! forget it and the next lookup would otherwise find it again.
//!
//! Held in memory, like the rest of the namespace, and durable the way the
//! rest of it is: through a snapshot.

use crate::sysbox;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Where a guest's own files live when a root image is mounted.
pub const UPPER: &str = "/linux/root";
/// Whiteouts: a path here hides the image's file at the same guest path.
pub const GONE: &str = "/linux/gone";
/// Permission bits a guest has set on a path, as four little-endian bytes.
///
/// **Beside the files, as whiteouts are, because the store has no modes.**
/// Every file reported 0644, so nothing looked executable: gpgme searched
/// PATH for gpgconf with `access(X_OK)`, found none, and pacman could check
/// no signature. `chmod` was a no-op, so a package's binaries, which its
/// extraction marks 0755, stayed 0644. A path with no entry here answers its
/// image inode's mode, or the default for its kind.
pub const MODES: &str = "/linux/mode";

/// Where a path's bits are kept: one flat entry per path, named by a hash
/// of it.
///
/// **It mirrored the path, and a mirror cannot hold a directory's bits and
/// its children's at once.** `/linux/mode/a/.git` became a four-byte blob the
/// moment `.git` was chmodded, after which `/linux/mode/a/.git/config.lock`
/// needed `.git` to be a directory and the write failed: git's `chmod` of its
/// own lock file answered EIO, and every AUR package built from git stopped
/// there. A flat name cannot collide with anything.
fn mode_path(guest: &str) -> String {
    let h = crate::store::sha256::hash(guest.as_bytes());
    let mut name = String::with_capacity(32);
    for b in &h[..16] {
        name.push_str(&format!("{:02x}", b));
    }
    format!("{}/{}", MODES, name)
}

/// Where the mirrored layout kept it, read so bits set before the change are
/// still honoured.
fn old_mode_path(guest: &str) -> String {
    format!("{}{}", MODES, guest)
}

/// The permission bits set on a guest path, if any were.
pub fn mode_of(guest: &str) -> Option<u32> {
    let b = sysbox::read_blob(&mode_path(guest)).or_else(|| sysbox::read_blob(&old_mode_path(guest)))?;
    (b.len() == 4).then(|| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0o7777)
}

pub fn set_mode(guest: &str, perm: u32) -> bool {
    sysbox::write_blob(&mode_path(guest), Vec::from((perm & 0o7777).to_le_bytes()))
}

pub fn forget_mode(guest: &str) {
    for p in [mode_path(guest), old_mode_path(guest)] {
        if sysbox::blob_len(&p).is_some() {
            sysbox::detach(&p);
        }
    }
}

/// The trees a guest shares with the kernel rather than layering.
const SHARED: [&str; 2] = ["/tmp", "/run"];

fn under(path: &str, prefix: &str) -> bool {
    path == prefix || path.strip_prefix(prefix).is_some_and(|r| r.starts_with('/'))
}

/// Whether the guests' root is an image, which is what turns layering on.
pub fn layered() -> bool {
    super::disk::root_mounted()
}

/// The namespace path that backs a guest path.
pub fn backing(guest: &str) -> String {
    if !layered() || SHARED.iter().any(|s| under(guest, s)) {
        return String::from(guest);
    }
    if guest == "/" {
        String::from(UPPER)
    } else {
        format!("{}{}", UPPER, guest)
    }
}

fn gone_path(guest: &str) -> String {
    format!("{}{}", GONE, guest)
}

/// Whether the image's file at this path has been deleted.
pub fn whited_out(guest: &str) -> bool {
    layered() && sysbox::blob_len(&gone_path(guest)).is_some()
}

/// Hide the image's file at this path, and say whether the mark was made.
pub fn white_out(guest: &str) -> bool {
    sysbox::write_blob(&gone_path(guest), Vec::new())
}

/// A path written again is no longer deleted.
pub fn unwhite(guest: &str) {
    if layered() {
        let g = gone_path(guest);
        if sysbox::blob_len(&g).is_some() {
            sysbox::detach(&g);
        }
    }
}

pub fn blob_len(guest: &str) -> Option<usize> {
    sysbox::blob_len(&backing(guest))
}

pub fn is_dir(guest: &str) -> bool {
    sysbox::is_dir(&backing(guest))
}

pub fn read_blob(guest: &str) -> Option<Vec<u8>> {
    sysbox::read_blob(&backing(guest))
}

pub fn listing(guest: &str) -> Vec<(String, bool, usize)> {
    sysbox::listing(&backing(guest))
}

pub fn write_blob(guest: &str, data: Vec<u8>) -> bool {
    unwhite(guest);
    sysbox::write_blob(&backing(guest), data)
}

pub fn make_dir(guest: &str) -> bool {
    unwhite(guest);
    sysbox::make_dir(&backing(guest))
}

pub fn detach(guest: &str) -> bool {
    sysbox::detach(&backing(guest))
}

pub fn rename(from: &str, to: &str) -> bool {
    unwhite(to);
    // The bits travel with the file; whatever `to` had set goes with it.
    forget_mode(to);
    if let Some(m) = mode_of(from) {
        set_mode(to, m);
        forget_mode(from);
    }
    sysbox::rename(&backing(from), &backing(to))
}

/// Whether a guest may write at this path. Shared trees always; anything but
/// the kernel's own device and process trees once there is a layer to write
/// into; only `/tmp` without one, as before.
pub fn writable(guest: &str) -> bool {
    if under(guest, "/tmp") {
        return true;
    }
    if !layered() {
        return false;
    }
    !(under(guest, "/proc") || under(guest, "/dev") || under(guest, "/sys") || under(guest, "/mnt"))
}

/// Whether any write a guest can make lands on this *namespace* path.
///
/// The question a jail is really asked. Under a root image a guest may write
/// almost any of its own paths, and that is right -- every such write lands
/// in the layer (`UPPER`, `GONE`), never on the namespace path of the same
/// name. So "may a guest write `/linux/bin`" stopped being the property; "can
/// a guest's write reach the kernel's `/linux/bin`" is, and it is no either
/// way. Without a layer, only `/tmp` is reachable, as before.
pub fn reaches(ns_path: &str) -> bool {
    if under(ns_path, "/tmp") {
        return true;
    }
    layered() && (under(ns_path, UPPER) || under(ns_path, GONE) || SHARED.iter().any(|s| under(ns_path, s)))
}

/// Copy a file up from the root image into the layer, so it can be written
/// in place. Answers whether the layer now holds it.
pub fn copy_up(guest: &str) -> bool {
    if blob_len(guest).is_some() {
        return true;
    }
    let Some(bytes) = super::disk::read_all(guest) else { return false };
    // The image file's mode comes with it: a program copied up to be edited
    // is still a program.
    if mode_of(guest).is_none() {
        if let Some(m) = super::disk::image_lookup(guest, true).and_then(|n| n.inode).map(|i| i.mode) {
            set_mode(guest, m);
        }
    }
    write_blob(guest, bytes)
}

/// What `diag linux` can check of the mapping itself.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    out.push(("with no root image a guest path is its own namespace path", layered() || backing("/etc/x") == "/etc/x"));
    out.push(("the shared trees are never layered", backing("/tmp/a") == "/tmp/a" && backing("/run/glados/x") == "/run/glados/x"));
    out.push(("a name that only begins like a shared tree is not one", !under("/tmpx", "/tmp") && under("/tmp", "/tmp")));
    out.push(("without a layer only /tmp is writable", layered() || (writable("/tmp/a") && !writable("/etc/a"))));
    out
}

//! Volumes a guest reads in place: files served by range from disk.
//!
//! The namespace a guest normally sees is the content-addressed store, held in
//! memory, and an open file there *is* its whole contents (`fs.rs` says so and
//! why). That is right for configuration and text and impossible for a game or
//! an office suite: a Source engine install is 4 GB of archives, and the
//! machine has less heap than that. So a partition on the NVMe disk can be
//! mounted at a guest path, and a file under it opens as a `DiskFile` that
//! reads only the bytes asked for.
//!
//! Two kinds of volume. **FAT**, for a game's or a user's own files, mounted
//! at a path like `/mnt`. **EROFS**, the root filesystem `tools/rootfs.py`
//! builds from a distribution's packages, mounted at `/` -- and underneath
//! everything, which is the point of it: `/proc`, `/dev`, `/run` and `/tmp`
//! stay this kernel's own, a file in the namespace wins over the image's at
//! the same path (so what a program writes, or an operator staged, is what it
//! reads back), and everything else -- `/usr`, `/etc/fonts`, `/lib64` -- is a
//! normal Linux system where a program expects one. Nothing has to be staged
//! by hand for a program to find its data.
//!
//! **Read-only, and a volume is assumed not to change while mounted.** That is
//! what makes caching safe: a FAT directory listing is read once and kept, and
//! a file's cluster chain is built once at open.
//!
//! **FAT folds case**, because it stores names case-insensitively and a
//! program written for Windows asks for `Materials/Foo.VMT` where the archive
//! says `materials/foo.vmt`. EROFS does not: it is a Linux filesystem and its
//! names are exact.

use crate::store::{erofs, fat};
use crate::sync::Racy;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

enum Vol {
    Fat {
        vol: fat::Volume,
        /// Listings already read, by cluster. Reading a directory is a walk of
        /// its cluster chain, and a program probing for files does it
        /// thousands of times on the same few directories.
        dirs: BTreeMap<u32, Rc<Vec<fat::DirEntry>>>,
    },
    Erofs(erofs::Image),
}

struct Mount {
    /// Guest path the volume appears at, without a trailing slash; empty for
    /// the root image.
    prefix: String,
    vol: Vol,
}

static MOUNTS: Racy<Vec<Mount>> = Racy::new(Vec::new());

/// The guest lock, held for the length of a public call.
///
/// **`MOUNTS` is read by guests inside their syscalls, under that lock, and
/// changed or read from the shell outside it.** `linux mount` pushing a
/// second volume reallocated the list while a session service on another
/// core walked it: the service failed its lookups and exited 127, and the
/// shell took a page fault in the middle of the mount. A guest already holds
/// the lock and this is a no-op for it; for the shell, and for `start_job`
/// reading a program, it is the exclusion every guest's syscall already has.
struct Big(bool);

impl Drop for Big {
    fn drop(&mut self) {
        if self.0 {
            super::syscall::big_unlock();
        }
    }
}

fn big() -> Big {
    Big(super::syscall::big_take())
}

/// The trees that are this kernel's own and never the root image's.
const OWN: [&str; 4] = ["/proc", "/dev", "/run", "/tmp"];

/// Mount partition `index` (as `block::scan` numbers them) at `prefix`. FAT or
/// EROFS, told apart by the superblock; only an EROFS image may be the root.
pub fn mount(prefix: &str, index: u32) -> Result<(), &'static str> {
    let prefix = prefix.trim_end_matches('/');
    if !prefix.is_empty() && (!prefix.starts_with('/') || prefix.len() < 2) {
        return Err("a mount point is an absolute path");
    }
    if prefix.is_empty() && index == 0 {
        return Err("no partition zero");
    }
    let layout = crate::store::block::scan().map_err(|_| "cannot read the partition table")?;
    let p = layout
        .partitions
        .iter()
        .find(|p| p.index == index)
        .ok_or("no partition with that number -- 'fat' lists them")?;
    let vol = match erofs::Image::mount(p.start_lba) {
        Ok(img) => Vol::Erofs(img),
        Err(_) if prefix.is_empty() => return Err("only an EROFS root image can be mounted at /"),
        Err(_) => Vol::Fat {
            vol: fat::Volume::mount(p.start_lba).map_err(|_| "that partition is neither FAT nor EROFS")?,
            dirs: BTreeMap::new(),
        },
    };
    let _g = big();
    let m = unsafe { MOUNTS.get() };
    m.retain(|x| x.prefix != prefix);
    m.push(Mount { prefix: String::from(prefix), vol });
    Ok(())
}

/// Mount point, kind, and block or cluster size, for the shell.
pub fn mounts() -> Vec<(String, &'static str, u32)> {
    let _g = big();
    unsafe { MOUNTS.get() }
        .iter()
        .map(|m| {
            let p = if m.prefix.is_empty() { String::from("/") } else { m.prefix.clone() };
            match &m.vol {
                Vol::Fat { vol, .. } => (p, "fat", vol.cluster_bytes()),
                Vol::Erofs(img) => (p, "erofs", img.block_size()),
            }
        })
        .collect()
}

fn own(path: &str) -> bool {
    OWN.iter().any(|o| path == *o || path.strip_prefix(o).is_some_and(|r| r.starts_with('/')))
}

/// The mount `path` is under, and the rest of the path inside it. A named
/// mount wins over the root image, and the root image yields to everything
/// listed at the top of this file.
fn under(path: &str) -> Option<(usize, &str)> {
    let m = unsafe { MOUNTS.get() };
    if let Some(hit) = m.iter().enumerate().find_map(|(i, x)| {
        if x.prefix.is_empty() {
            return None;
        }
        let rest = path.strip_prefix(x.prefix.as_str())?;
        (rest.is_empty() || rest.starts_with('/')).then_some((i, rest))
    }) {
        return Some(hit);
    }
    let (i, root) = m.iter().enumerate().find(|(_, x)| x.prefix.is_empty())?;
    if own(path) || super::ns::blob_len(path).is_some() || super::ns::whited_out(path) {
        return None;
    }
    // A directory in the guest's layer that the image also has is the
    // image's, so a walk through `/usr` or `/etc` keeps going (and `entries`
    // merges the layer's files into its listing); one only the layer has
    // stays the layer's.
    if super::ns::is_dir(path) {
        let Vol::Erofs(img) = &root.vol else { return None };
        if !img.lookup(path, true).is_ok_and(|n| n.is_dir()) {
            return None;
        }
    }
    Some((i, path))
}

/// Whether the guests' root filesystem is an image.
pub fn root_mounted() -> bool {
    let _g = big();
    unsafe { MOUNTS.get() }.iter().any(|m| m.prefix.is_empty())
}

/// Whether a path is answered by the root image rather than a named mount.
/// Those are the paths a write may copy up into the guest's layer; a named
/// mount is read-only all the way down.
pub fn root_claims(path: &str) -> bool {
    let _g = big();
    under(path).is_some_and(|(i, _)| unsafe { MOUNTS.get()[i].prefix.is_empty() })
}

/// The root image's own answer for a path, ignoring the guest's layer but
/// honouring its whiteouts: what `mkdir`, `unlink` and `rename` need to know
/// about the file underneath. `None` with no root image or nothing there.
pub fn image_lookup(path: &str, follow: bool) -> Option<Node> {
    let _g = big();
    if own(path) || super::ns::whited_out(path) {
        return None;
    }
    let m = unsafe { MOUNTS.get() };
    let root = m.iter().find(|x| x.prefix.is_empty())?;
    let Vol::Erofs(img) = &root.vol else { return None };
    img.lookup(path, follow).ok().map(erofs_node)
}

/// Whether a path is on a mounted volume at all.
pub fn claims(path: &str) -> bool {
    let _g = big();
    under(path).is_some()
}

fn listing(dirs: &mut BTreeMap<u32, Rc<Vec<fat::DirEntry>>>, vol: &fat::Volume, cluster: u32) -> Option<Rc<Vec<fat::DirEntry>>> {
    if let Some(l) = dirs.get(&cluster) {
        return Some(l.clone());
    }
    let l = Rc::new(vol.list(cluster).ok()?);
    dirs.insert(cluster, l.clone());
    Some(l)
}

/// What a path names on a mounted volume.
#[derive(Clone)]
pub struct Node {
    pub is_dir: bool,
    pub size: usize,
    /// The FAT start cluster; zero on EROFS.
    pub cluster: u32,
    /// The EROFS inode, when it is one.
    pub inode: Option<erofs::Inode>,
    /// A symlink, as `lstat` sees one: only when looked up without following.
    pub is_link: bool,
}

fn fat_lookup(dirs: &mut BTreeMap<u32, Rc<Vec<fat::DirEntry>>>, vol: &fat::Volume, rest: &str) -> Option<Node> {
    let mut node = Node { is_dir: true, size: 0, cluster: 0, inode: None, is_link: false };
    for part in rest.split('/').filter(|p| !p.is_empty() && *p != ".") {
        if !node.is_dir {
            return None;
        }
        let l = listing(dirs, vol, node.cluster)?;
        let e = l.iter().find(|e| e.name.eq_ignore_ascii_case(part))?;
        node = Node { is_dir: e.is_dir, size: e.size as usize, cluster: e.cluster, inode: None, is_link: false };
    }
    Some(node)
}

fn erofs_node(i: erofs::Inode) -> Node {
    Node { is_dir: i.is_dir(), size: i.size as usize, cluster: 0, inode: Some(i), is_link: i.is_link() }
}

fn lookup_with(path: &str, follow: bool) -> Option<Option<Node>> {
    let (i, rest) = under(path)?;
    let m = unsafe { &mut MOUNTS.get()[i] };
    Some(match &mut m.vol {
        Vol::Fat { vol, dirs } => fat_lookup(dirs, vol, rest),
        Vol::Erofs(img) => img.lookup(rest, follow).ok().map(erofs_node),
    })
}

/// Look a path up, following symlinks. `None` when it is not on a mount;
/// `Some(None)` when it is and nothing is there.
pub fn lookup(path: &str) -> Option<Option<Node>> {
    let _g = big();
    lookup_with(path, true)
}

/// The same without following a symlink in the last component: `lstat`.
pub fn lookup_nofollow(path: &str) -> Option<Option<Node>> {
    let _g = big();
    lookup_with(path, false)
}

/// A symlink's target. `None` when the path is not on a mount; `Some(None)`
/// when it is and is not a link.
pub fn link_target(path: &str) -> Option<Option<String>> {
    let _g = big();
    let (i, rest) = under(path)?;
    let m = unsafe { &MOUNTS.get()[i] };
    let Vol::Erofs(img) = &m.vol else { return Some(None) };
    Some(img.lookup(rest, false).ok().filter(|n| n.is_link()).and_then(|n| {
        img.read_all(&n).ok().map(|b| String::from_utf8_lossy(&b).into_owned())
    }))
}

/// A directory's entries as `fs::Dir` wants them: name, is a directory, size.
/// A symlink to a directory counts as one, which is what `d_type` cannot say
/// and what a program walking the tree wants to know.
pub fn entries(path: &str) -> Vec<(String, bool, usize)> {
    let _g = big();
    let Some(Some(n)) = lookup(path) else { return Vec::new() };
    if !n.is_dir {
        return Vec::new();
    }
    let (i, _) = under(path).unwrap();
    let m = unsafe { &mut MOUNTS.get()[i] };
    match &mut m.vol {
        Vol::Fat { vol, dirs } => listing(dirs, vol, n.cluster)
            .map(|l| l.iter().map(|e| (e.name.clone(), e.is_dir, e.size as usize)).collect())
            .unwrap_or_default(),
        Vol::Erofs(img) => {
            let Some(ino) = n.inode else { return Vec::new() };
            let base = path.trim_end_matches('/');
            let mut out: Vec<(String, bool, usize)> = img
                .dir(&ino)
                .map(|d| {
                    d.into_iter()
                        .filter(|(name, _, _)| name != "." && name != "..")
                        .filter(|(name, _, _)| !super::ns::whited_out(&alloc::format!("{}/{}", base, name)))
                        .filter_map(|(name, nid, _)| {
                            let mut c = img.inode(nid).ok()?;
                            if c.is_link() {
                                let full = alloc::format!("{}/{}", path.trim_end_matches('/'), name);
                                c = img.lookup(&full, true).unwrap_or(c);
                            }
                            Some((name, c.is_dir(), c.size as usize))
                        })
                        .collect()
                })
                .unwrap_or_default();
            // The guest's own files at this directory: what it wrote wins
            // over the image's entry of the same name.
            if super::ns::layered() && m.prefix.is_empty() {
                for e in super::ns::listing(path) {
                    out.retain(|x| x.0 != e.0);
                    out.push(e);
                }
                out.sort_by(|a, b| a.0.cmp(&b.0));
            }
            out
        }
    }
}

enum Body {
    Fat {
        chain: Vec<u32>,
        /// The last run read, so a program reading a few hundred bytes at a
        /// time does not cost a disk command per call.
        cache_first: usize,
        cache: Vec<u8>,
    },
    Erofs(erofs::Inode),
}

/// An open file on a mounted volume.
pub struct DiskFile {
    pub path: String,
    pub size: usize,
    pub at: usize,
    mount: usize,
    body: Body,
}

/// How many clusters one read may cover at most: a megabyte at 32 KiB
/// clusters, which is what one NVMe command list can carry here.
const RUN_MAX: usize = 32;

impl DiskFile {
    pub fn open(path: &str) -> Result<DiskFile, &'static str> {
        let _g = big();
        let (i, _) = under(path).ok_or("not on a mounted volume")?;
        let Some(Some(n)) = lookup(path) else { return Err("no such file") };
        if n.is_dir {
            return Err("a directory");
        }
        let m = unsafe { &MOUNTS.get()[i] };
        let body = match &m.vol {
            Vol::Fat { vol, .. } => Body::Fat {
                chain: if n.size == 0 { Vec::new() } else { vol.chain(n.cluster).map_err(|_| "the volume's table is damaged")? },
                cache_first: usize::MAX,
                cache: Vec::new(),
            },
            Vol::Erofs(_) => Body::Erofs(n.inode.ok_or("no inode")?),
        };
        Ok(DiskFile { path: String::from(path), size: n.size, at: 0, mount: i, body })
    }

    /// Copy up to `out.len()` bytes from offset `off`. Answers how many.
    pub fn read_at(&mut self, off: usize, out: &mut [u8]) -> usize {
        let _g = big();
        if off >= self.size || out.is_empty() {
            return 0;
        }
        let m = unsafe { &MOUNTS.get()[self.mount] };
        match (&m.vol, &mut self.body) {
            (Vol::Erofs(img), Body::Erofs(ino)) => img.read(ino, off as u64, out).unwrap_or(0),
            (Vol::Fat { vol, .. }, Body::Fat { chain, cache_first, cache }) => {
                let cb = vol.cluster_bytes() as usize;
                let want = out.len().min(self.size - off);
                let mut done = 0;
                while done < want {
                    let pos = off + done;
                    let ci = pos / cb;
                    // Serve from the cached run if it covers this position.
                    let run_bytes = cache.len();
                    if *cache_first != usize::MAX && ci >= *cache_first && pos < *cache_first * cb + run_bytes {
                        let within = pos - *cache_first * cb;
                        let n = (run_bytes - within).min(want - done);
                        out[done..done + n].copy_from_slice(&cache[within..within + n]);
                        done += n;
                        continue;
                    }
                    // Otherwise read the run of consecutive clusters starting here.
                    let Some(&first) = chain.get(ci) else { break };
                    let mut count = 1;
                    while count < RUN_MAX
                        && ci + count < chain.len()
                        && chain[ci + count] == first + count as u32
                        // Do not read past what the caller and the file can use.
                        && (ci + count) * cb < off + want
                    {
                        count += 1;
                    }
                    cache.resize(count * cb, 0);
                    if vol.read_run(first, count as u32, cache).is_err() {
                        *cache_first = usize::MAX;
                        break;
                    }
                    *cache_first = ci;
                }
                done
            }
            _ => 0,
        }
    }
}

/// A whole file, for the loader: a program or an interpreter on a mount.
pub fn read_all(path: &str) -> Option<Vec<u8>> {
    let _g = big();
    let mut f = DiskFile::open(path).ok()?;
    let mut v = alloc::vec![0u8; f.size];
    let n = f.read_at(0, &mut v);
    (n == v.len()).then_some(v)
}

/// The bytes of a program wherever it lives: a mounted volume or the
/// namespace.
pub fn program_bytes(path: &str) -> Option<Vec<u8>> {
    if claims(path) {
        return read_all(path);
    }
    super::ns::read_blob(path)
}

/// What `diag linux` can check with no volume mounted.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    out.push(("a path under no mount is not claimed", !claims("/tmp/x")));
    out.push(("a mount point must be absolute", mount("rel", 1).is_err()));
    out.push(("the kernel's own trees are never the root image's", own("/proc/self/maps") && own("/tmp") && !own("/tmpx")));
    out
}

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
    /// This mount's own number, never reused: what an open file names it by.
    id: u64,
}

static MOUNTS: Racy<Vec<Mount>> = Racy::new(Vec::new());

/// Mounts replaced at their path while files may still be open on them.
///
/// **An open file names its volume by `Mount::id`, and that is the fix for
/// a crash that looked like the dynamic loader's.** It named it by position
/// in `MOUNTS`, and `mount` takes the old entry at a path out before adding
/// the new one -- so remounting `/` slid every later mount down a place.
/// Boot mounts the root first and starts the session bus on it; a `linux
/// mount / 2` typed while dbus was still loading its libraries left every
/// library it had open naming `/disk/1`, a FAT volume, where the reads found
/// no file of theirs and answered zero bytes. `mmap` laid those zeros down
/// as the library's dynamic section and `ld.so` took a null pointer out of
/// it -- on about half of boots, the half where dbus was still loading.
/// A replaced mount is kept here, as Linux keeps a lazily unmounted
/// filesystem alive for the files open on it; mounts are rare and a volume's
/// state is small, so they are never freed.
static RETIRED: Racy<Vec<Mount>> = Racy::new(Vec::new());
static NEXT_MOUNT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);

/// The mount an open file names, wherever it now is.
fn by_id(id: u64) -> Option<&'static mut Mount> {
    let m = unsafe { MOUNTS.get() };
    if let Some(i) = m.iter().position(|x| x.id == id) {
        return Some(&mut m[i]);
    }
    unsafe { RETIRED.get() }.iter_mut().find(|x| x.id == id)
}

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
    // Out of the table and into `RETIRED`, never dropped: files may be open
    // on it (see `RETIRED`).
    while let Some(i) = m.iter().position(|x| x.prefix == prefix) {
        let old = m.remove(i);
        unsafe { RETIRED.get() }.push(old);
    }
    let id = NEXT_MOUNT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    m.push(Mount { prefix: String::from(prefix), vol, id });
    Ok(())
}

/// Find the guests' root filesystem on the attached disks and mount it, the
/// way a person typed `linux mount / 2` for every session until now.
///
/// Every partition is offered to `mount` at `/`, which refuses anything that
/// is not EROFS, so the first that takes is the root and nothing else is
/// touched. Answers the partition number, or `None` when there is no root
/// image -- the ordinary case on a machine without one, and not an error.
/// Already mounted is answered as the partition it already is.
pub fn auto_root() -> Option<u32> {
    if root_mounted() {
        return Some(0);
    }
    // A root on the disk first: an installed system's. Then the one the ISO
    // carried into memory, answered as partition 0.
    let on_disk = crate::store::block::scan().ok().and_then(|layout| {
        let indices: Vec<u32> = layout.partitions.iter().map(|p| p.index).collect();
        indices.into_iter().find(|&i| mount("/", i).is_ok())
    });
    on_disk.or_else(|| mount_ram().ok().map(|_| 0))
}

/// Mount the in-memory root (`erofs::set_ram`) at `/`.
fn mount_ram() -> Result<(), &'static str> {
    let img = erofs::Image::mount(erofs::RAM_BASE).map_err(|_| "no root image in memory")?;
    let _g = big();
    let m = unsafe { MOUNTS.get() };
    let id = NEXT_MOUNT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    m.push(Mount { prefix: String::new(), vol: Vol::Erofs(img), id });
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
    if own(path) || super::ns::blob_len(path).is_some() || super::ns::whited_out(path) || super::link::own(path).is_some() {
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

/// The first link in the root image on the way to `path`: how many of its
/// components lead to it, and its target. `None` with no root image, for a
/// path in the kernel's own trees, or when the image has no link on the way.
pub fn image_first_link(path: &str, last: bool) -> Option<(usize, String)> {
    let _g = big();
    if own(path) {
        return None;
    }
    let m = unsafe { MOUNTS.get() };
    let root = m.iter().find(|x| x.prefix.is_empty())?;
    // A named mount over part of the path is not the image's to answer.
    if m.iter().any(|x| !x.prefix.is_empty() && (path == x.prefix || path.strip_prefix(x.prefix.as_str()).is_some_and(|r| r.starts_with('/')))) {
        return None;
    }
    let Vol::Erofs(img) = &root.vol else { return None };
    img.first_link(path, last).ok().flatten()
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
/// Image directory listings already read, by (mount, directory inode). `Rc`
/// because a guest's whole session runs under the guest lock on one core at a
/// time -- the lock `entries` holds -- the bargain `fs` makes for descriptors.
static LISTINGS: Racy<alloc::collections::BTreeMap<(u64, u64), Listing>> =
    Racy::new(alloc::collections::BTreeMap::new());
/// Entries kept across every cached listing before the cache starts again.
const LISTING_CAP: usize = 200_000;

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
            let key = (m.id, ino.nid);
            // **The image's own listing is computed once.** It is an inode read
            // per entry, for the size and to follow a link to a directory, and
            // `opendir` on `/usr/lib` -- 2,559 entries -- took 270 ms; a vim
            // install opened image directories 10,057 times at 304 us each,
            // pacman checking for conflicts. The image never changes, so neither
            // does this; what the guest's layer adds or deletes is applied on
            // top, fresh, every time.
            let image = image_listing(img, key, &ino, path);
            let image = image.0;
            // Deletions, read once for the directory rather than asked after
            // each name.
            let gone = gone_in(path);
            let mut out: Vec<(String, bool, usize)> =
                image.iter().filter(|e| !gone.contains(&e.0)).cloned().collect();
            // The guest's own files at this directory: what it wrote wins
            // over the image's entry of the same name.
            if super::ns::layered() && m.prefix.is_empty() {
                let mine = super::ns::listing(path);
                if !mine.is_empty() {
                    let names: alloc::collections::BTreeSet<&str> = mine.iter().map(|e| e.0.as_str()).collect();
                    out.retain(|x| !names.contains(x.0.as_str()));
                    out.extend(mine.iter().cloned());
                    out.sort_by(|a, b| a.0.cmp(&b.0));
                }
            }
            out
        }
    }
}

/// The names the guest has deleted from the image in this directory.
fn gone_in(path: &str) -> alloc::collections::BTreeSet<String> {
    if !super::ns::layered() {
        return alloc::collections::BTreeSet::new();
    }
    crate::sysbox::listing(&alloc::format!("{}{}", super::ns::GONE, path.trim_end_matches('/')))
        .into_iter()
        .filter(|(_, is_dir, _)| !is_dir)
        .map(|(name, _, _)| name)
        .collect()
}

type Listing = (alloc::rc::Rc<Vec<(String, bool, usize)>>, alloc::rc::Rc<alloc::collections::BTreeSet<String>>);

/// The image's own listing of a directory and which of its names are links,
/// read once (see `LISTINGS`).
fn image_listing(img: &erofs::Image, key: (u64, u64), ino: &erofs::Inode, path: &str) -> Listing {
    let cached = unsafe { LISTINGS.get() }.get(&key).cloned();
            match cached {
                Some(l) => l,
                None => {
                    let raw = img.dir(ino).unwrap_or_default();
                    let links: alloc::collections::BTreeSet<String> =
                        raw.iter().filter(|(_, _, t)| *t == 7).map(|(n, _, _)| n.clone()).collect();
                    let l: Vec<(String, bool, usize)> = Some(raw)
                        .map(|d| {
                            d.into_iter()
                                .filter(|(name, _, _)| name != "." && name != "..")
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
                    let l = (alloc::rc::Rc::new(l), alloc::rc::Rc::new(links));
                    let c = unsafe { LISTINGS.get() };
                    let total: usize = c.values().map(|v| v.0.len()).sum();
                    if total + l.0.len() > LISTING_CAP {
                        c.clear();
                    }
                    c.insert(key, l.clone());
                    l
                }
            }
}

/// Which names in a directory are links, for `getdents64`'s type byte: the
/// guest's own, and the image's that the layer has not replaced or deleted.
///
/// **Asked once for the directory, not once per name.** `d_type` for a link
/// was a whole `link::is_link` per entry -- the namespace twice, the image
/// once -- and a glob of `/usr/lib` spent 18.7 ms in each `getdents64` call:
/// two hundred of them took 20.7 s. The image's directory blocks already say
/// which entries are links; the type byte is read off them when the listing
/// is cached.
pub fn link_names(path: &str) -> alloc::collections::BTreeSet<String> {
    let _g = big();
    let mut out: alloc::collections::BTreeSet<String> =
        super::link::names_in(path).into_iter().map(|(n, _)| n).collect();
    let Some((i, _)) = under(path) else { return out };
    let m = unsafe { &MOUNTS.get()[i] };
    let Vol::Erofs(img) = &m.vol else { return out };
    let Some(Some(n)) = lookup(path) else { return out };
    let Some(ino) = n.inode else { return out };
    let (_, links) = image_listing(img, (m.id, ino.nid), &ino, path);
    if links.is_empty() {
        return out;
    }
    let gone = gone_in(path);
    let mine: alloc::collections::BTreeSet<String> = if super::ns::layered() && m.prefix.is_empty() {
        super::ns::listing(path).into_iter().map(|e| e.0).collect()
    } else {
        alloc::collections::BTreeSet::new()
    };
    for l in links.iter() {
        if !gone.contains(l) && !mine.contains(l) {
            out.insert(l.clone());
        }
    }
    out
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
    /// The mount's id, not its place in the table: see `RETIRED`.
    mount: u64,
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
        Ok(DiskFile { path: String::from(path), size: n.size, at: 0, mount: m.id, body })
    }

    /// What reading this file needs, copied out so the read can be made
    /// without the guest lock: the image's handle and the inode. `None` on
    /// FAT, whose reads go through state behind that lock.
    pub fn erofs_source(&self) -> Option<(erofs::Image, erofs::Inode)> {
        let Body::Erofs(ino) = &self.body else { return None };
        let _g = big();
        match by_id(self.mount).map(|m| &m.vol) {
            Some(Vol::Erofs(img)) => Some((*img, *ino)),
            _ => None,
        }
    }

    /// The permission bits the image gives this file; `None` on FAT.
    pub fn image_mode(&self) -> Option<u32> {
        match &self.body {
            Body::Erofs(ino) => Some(ino.mode & 0o7777),
            _ => None,
        }
    }

    /// Copy up to `out.len()` bytes from offset `off`. Answers how many.
    pub fn read_at(&mut self, off: usize, out: &mut [u8]) -> usize {
        let _g = big();
        if off >= self.size || out.is_empty() {
            return 0;
        }
        let Some(m) = by_id(self.mount) else { return 0 };
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
    // **A name is found by binary search, which is only right if the image
    // sorts its directories**, as `mkfs.erofs` does. Checked against a scan
    // of every entry in a few real directories, when a root image is
    // mounted -- an image built some other way would lose files silently, so
    // this is where it should say so instead.
    let _g = big();
    let m = unsafe { MOUNTS.get() };
    if let Some(Vol::Erofs(img)) = m.iter().find(|x| x.prefix.is_empty()).map(|x| &x.vol) {
        let mut agree = true;
        let mut seen = 0usize;
        for d in ["", "usr/bin", "usr/lib", "etc"] {
            let Ok(dir) = img.lookup(d, true) else { continue };
            let Ok(list) = img.dir(&dir) else { continue };
            for (name, nid, _) in &list {
                seen += 1;
                agree &= img.child_nid(&dir, name) == Some(*nid);
                // Twice through the cache: the first fills it, the second is
                // answered from it, and both must be the search's answer.
                agree &= img.child_cached_nid(&dir, name) == Some(*nid);
                agree &= img.child_cached_nid(&dir, name) == Some(*nid);
            }
            agree &= img.child_nid(&dir, "\u{7f}no such name") == None && img.child_by_scan(&dir, "\u{7f}no such name").is_none();
        }
        out.push(("a directory lookup by binary search finds every entry a scan finds", agree && seen > 0));
    }
    out
}

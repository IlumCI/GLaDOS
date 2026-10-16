//! A FAT volume a guest can read in place: files served by range from disk.
//!
//! The namespace a guest normally sees is the content-addressed store, held in
//! memory, and an open file there *is* its whole contents (`fs.rs` says so and
//! why). That is right for configuration and text and impossible for a game: a
//! Source engine install is 4 GB of archives, and the machine has less heap
//! than that. So a FAT partition on the NVMe disk can be mounted at a guest
//! path, and a file under it opens as a `DiskFile` that reads only the bytes
//! asked for -- straight from the clusters, with runs of consecutive clusters
//! read in one command.
//!
//! **Read-only, and the volume is assumed not to change while mounted.** That
//! is what makes caching safe: a directory listing is read once and kept, and
//! a file's cluster chain is built once at open. Writing through this would
//! need the opposite assumption everywhere, and nothing a game reads from its
//! install directory is written back there.
//!
//! **Case is folded**, because FAT stores names case-insensitively and a
//! program written for Windows asks for `Materials/Foo.VMT` where the archive
//! says `materials/foo.vmt`. That is a property of the format rather than a
//! convenience; the Linux build of Source carries its own `pathmatch` shim for
//! exactly this.

use crate::store::fat;
use crate::sync::Racy;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

struct Mount {
    /// Guest path the volume appears at, without a trailing slash.
    prefix: String,
    vol: fat::Volume,
    /// Listings already read, by cluster. Reading a directory is a walk of its
    /// cluster chain, and a program probing for files does it thousands of
    /// times on the same few directories.
    dirs: BTreeMap<u32, Rc<Vec<fat::DirEntry>>>,
}

static MOUNTS: Racy<Vec<Mount>> = Racy::new(Vec::new());

/// Mount FAT partition `index` (as `block::scan` numbers them) at `prefix`.
pub fn mount(prefix: &str, index: u32) -> Result<(), &'static str> {
    let prefix = prefix.trim_end_matches('/');
    if !prefix.starts_with('/') || prefix.len() < 2 {
        return Err("a mount point is an absolute path, and not /");
    }
    let layout = crate::store::block::scan().map_err(|_| "cannot read the partition table")?;
    let p = layout
        .partitions
        .iter()
        .find(|p| p.index == index)
        .ok_or("no partition with that number -- 'fat' lists them")?;
    let vol = fat::Volume::mount(p.start_lba).map_err(|_| "that partition is not a FAT volume")?;
    let m = unsafe { MOUNTS.get() };
    m.retain(|x| x.prefix != prefix);
    m.push(Mount { prefix: String::from(prefix), vol, dirs: BTreeMap::new() });
    Ok(())
}

pub fn mounts() -> Vec<(String, u32)> {
    unsafe { MOUNTS.get() }.iter().map(|m| (m.prefix.clone(), m.vol.cluster_bytes())).collect()
}

/// The mount `path` is under, and the rest of the path inside it.
fn under(path: &str) -> Option<(usize, &str)> {
    let m = unsafe { MOUNTS.get() };
    m.iter().enumerate().find_map(|(i, x)| {
        let rest = path.strip_prefix(x.prefix.as_str())?;
        (rest.is_empty() || rest.starts_with('/')).then_some((i, rest))
    })
}

/// Whether a path is on a mounted volume at all.
pub fn claims(path: &str) -> bool {
    under(path).is_some()
}

fn listing(m: &mut Mount, cluster: u32) -> Option<Rc<Vec<fat::DirEntry>>> {
    if let Some(l) = m.dirs.get(&cluster) {
        return Some(l.clone());
    }
    let l = Rc::new(m.vol.list(cluster).ok()?);
    m.dirs.insert(cluster, l.clone());
    Some(l)
}

/// What a path names on a mounted volume.
#[derive(Clone)]
pub struct Node {
    pub is_dir: bool,
    pub size: usize,
    pub cluster: u32,
}

/// Look a path up. `None` when it is not on a mount; `Some(None)` when it is
/// and nothing is there.
pub fn lookup(path: &str) -> Option<Option<Node>> {
    let (i, rest) = under(path)?;
    let m = unsafe { &mut MOUNTS.get()[i] };
    let mut node = Node { is_dir: true, size: 0, cluster: 0 };
    for part in rest.split('/').filter(|p| !p.is_empty() && *p != ".") {
        if !node.is_dir {
            return Some(None);
        }
        let Some(l) = listing(m, node.cluster) else { return Some(None) };
        let Some(e) = l.iter().find(|e| e.name.eq_ignore_ascii_case(part)) else {
            return Some(None);
        };
        node = Node { is_dir: e.is_dir, size: e.size as usize, cluster: e.cluster };
    }
    Some(Some(node))
}

/// A directory's entries as `fs::Dir` wants them: name, is a directory, size.
pub fn entries(path: &str) -> Vec<(String, bool, usize)> {
    let Some(Some(n)) = lookup(path) else { return Vec::new() };
    if !n.is_dir {
        return Vec::new();
    }
    let (i, _) = under(path).unwrap();
    let m = unsafe { &mut MOUNTS.get()[i] };
    listing(m, n.cluster)
        .map(|l| l.iter().map(|e| (e.name.clone(), e.is_dir, e.size as usize)).collect())
        .unwrap_or_default()
}

/// An open file on a mounted volume.
pub struct DiskFile {
    pub path: String,
    pub size: usize,
    pub at: usize,
    mount: usize,
    chain: Vec<u32>,
    /// The last run read, so a program reading a few hundred bytes at a time
    /// does not cost a disk command per call.
    cache_first: usize,
    cache: Vec<u8>,
}

/// How many clusters one read may cover at most: a megabyte at 32 KiB
/// clusters, which is what one NVMe command list can carry here.
const RUN_MAX: usize = 32;

impl DiskFile {
    pub fn open(path: &str) -> Result<DiskFile, &'static str> {
        let (i, _) = under(path).ok_or("not on a mounted volume")?;
        let Some(Some(n)) = lookup(path) else { return Err("no such file") };
        if n.is_dir {
            return Err("a directory");
        }
        let m = unsafe { &MOUNTS.get()[i] };
        let chain = if n.size == 0 { Vec::new() } else { m.vol.chain(n.cluster).map_err(|_| "the volume's table is damaged")? };
        Ok(DiskFile {
            path: String::from(path),
            size: n.size,
            at: 0,
            mount: i,
            chain,
            cache_first: usize::MAX,
            cache: Vec::new(),
        })
    }

    /// Copy up to `out.len()` bytes from offset `off`. Answers how many.
    pub fn read_at(&mut self, off: usize, out: &mut [u8]) -> usize {
        if off >= self.size || out.is_empty() {
            return 0;
        }
        let m = unsafe { &MOUNTS.get()[self.mount] };
        let cb = m.vol.cluster_bytes() as usize;
        let want = out.len().min(self.size - off);
        let mut done = 0;
        while done < want {
            let pos = off + done;
            let ci = pos / cb;
            // Serve from the cached run if it covers this position.
            let run_bytes = self.cache.len();
            if self.cache_first != usize::MAX
                && ci >= self.cache_first
                && pos < self.cache_first * cb + run_bytes
            {
                let within = pos - self.cache_first * cb;
                let n = (run_bytes - within).min(want - done);
                out[done..done + n].copy_from_slice(&self.cache[within..within + n]);
                done += n;
                continue;
            }
            // Otherwise read the run of consecutive clusters starting here.
            let Some(&first) = self.chain.get(ci) else { break };
            let mut count = 1;
            while count < RUN_MAX
                && ci + count < self.chain.len()
                && self.chain[ci + count] == first + count as u32
                // Do not read past what the caller and the file can use.
                && (ci + count) * cb < off + want
            {
                count += 1;
            }
            self.cache.resize(count * cb, 0);
            if m.vol.read_run(first, count as u32, &mut self.cache).is_err() {
                self.cache_first = usize::MAX;
                break;
            }
            self.cache_first = ci;
        }
        done
    }
}

/// A whole file, for the loader: a program or an interpreter on a mount.
pub fn read_all(path: &str) -> Option<Vec<u8>> {
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
    crate::sysbox::read_blob(path)
}

/// What `diag linux` can check with no volume mounted.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    out.push(("a path under no mount is not claimed", !claims("/tmp/x") || !mounts().iter().any(|(p, _)| p == "/tmp")));
    out.push(("a mount point must be absolute and not the root", mount("rel", 1).is_err() && mount("/", 1).is_err()));
    out
}

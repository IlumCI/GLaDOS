//! The stored tree, read without restoring it.
//!
//! Everything else in `sysbox` works on the **working tree**, which is
//! `Node::Blob(Vec<u8>)` all the way down and therefore entirely resident.
//! That is right for a namespace and wrong for a corpus: `read_node` walks
//! every node at restore and SHA-256s every blob, so a forest that fits on
//! disk and not in memory cannot be reached at all.
//!
//! This is the other door. It resolves a path by walking **directory chunks
//! only**, and reads bytes out of a blob's chunk with `cas::read_blocks` --
//! which has been sitting in the tree, finished, with nothing calling it.
//!
//! ### The two trades, which are different for directories and for blobs
//!
//! A directory chunk is read with `Store::get`: small, verified against its
//! own address, and there are few of them. A blob is read with `read_blocks`:
//! ranged, allocation-free, and **unverified**, because the hash covers the
//! whole blob and a range is not the whole blob. `cas.rs` says so where the
//! function is declared. So a body read this way is trusted on the strength of
//! having been verified when it was imported, which is the trade streaming
//! always makes and the reason the index is built from paths the directory
//! walk produced rather than from anything a body claims about itself.
//!
//! ### One scratch buffer, which is the fix `cas::dma` asks for
//!
//! `cas::dma` allocates per call through `alloc_zeroed` and never frees --
//! measured at 4,096 bytes plus the rounded length per blob. Its own note says
//! one static scratch would serve every caller. This is that, narrowed to the
//! callers here: 64 KiB taken the first time somebody asks and reused for the
//! life of the boot, so a million ranged reads cost what one does.
//!
//! **It is claimed rather than merely documented.** A single shared buffer
//! that two callers may enter is not a caution, it is a silent corruption: the
//! scheduler preempts at 100 Hz, `read_blob` is on the shell's path and on the
//! resident mind's, and a second caller landing mid-read would overwrite the
//! first one's bytes and hand both of them a plausible answer assembled from
//! two different blobs. So entry takes an `AtomicBool` and a contended read is
//! **refused and counted**, which is a failure somebody can see.
//!
//! That also serialises this path's device commands as a side effect, which is
//! worth stating precisely because it is not the guarantee it looks like:
//! `nvme::CONTROLLER` is a `Racy` and not a lock, so `cas::get` -- which has
//! its own freshly allocated buffer -- can still interleave with anything.
//! This narrows one hazard rather than removing the class.

use alloc::string::String;
use alloc::vec::Vec;

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::store::{self, block, cas};
use crate::sync::Racy;

use super::tree;

/// 64 KiB, which is 128 blocks at the usual size. Large enough that an
/// ordinary forest node is one command and small enough to be unremarkable
/// against a heap that starts at 320 MiB.
const SCRATCH_BYTES: usize = 64 * 1024;

/// The scratch buffer's address, or zero before anything has asked for it.
static SCRATCH: Racy<u64> = Racy::new(0);

/// Whether somebody is inside a ranged read right now.
static BUSY: AtomicBool = AtomicBool::new(false);

/// How many reads were refused because somebody already was.
///
/// Expected to stay at zero. It is a counter rather than a panic because a
/// refused read is recoverable -- the caller gets nothing and says so -- and
/// because a number that has moved is evidence, where a machine that stopped
/// is only a machine that stopped.
static CONTENDED: AtomicU64 = AtomicU64::new(0);

/// Released on every path out, including the early returns inside the loop.
struct Held;

impl Drop for Held {
    fn drop(&mut self) {
        BUSY.store(false, Ordering::Release);
    }
}

fn hold() -> Option<Held> {
    if BUSY.swap(true, Ordering::Acquire) {
        let n = CONTENDED.fetch_add(1, Ordering::Relaxed) + 1;
        // The first few, and then quiet. This should never happen, so it is
        // worth a line; if it happens in a loop, a line per iteration would
        // bury the thing that caused it.
        if n <= 3 {
            crate::gfx::console::set_color(crate::gfx::console::LTRED);
            crate::kprintln!("  [stored] a ranged read was refused: another one is in flight");
            crate::gfx::console::set_color(crate::gfx::console::LTGRAY);
        }
        return None;
    }
    Some(Held)
}

/// How many ranged reads have been refused for contention. Zero on a healthy
/// machine, and `diag forest` asserts it has not moved.
pub fn contended() -> u64 {
    CONTENDED.load(Ordering::Relaxed)
}

/// Does the buffer actually refuse a second caller?
///
/// **A guard that has never refused anything is indistinguishable from no
/// guard.** That is the objection `smp.rs` records about its own one-shot
/// canary, which passed over a deadlock; a counter sitting at zero is exactly
/// as convincing whether the check works or was deleted. So the hold is taken
/// here on purpose, the read underneath it must fail, and -- the half that is
/// easy to leave out -- the next read must succeed, or the guard is a one-way
/// door that breaks the machine the first time it fires.
///
/// The red line it prints on the way through is that refusal happening, and is
/// expected rather than a fault.
pub fn refuses_while_held(r: &cas::ChunkRef) -> bool {
    let before = CONTENDED.load(Ordering::Relaxed);
    let held = match hold() {
        Some(h) => h,
        // Somebody else is inside one right now, which is itself the thing
        // being tested for -- but it means this cannot run, so it must not
        // report success.
        None => return false,
    };
    let refused = read_at(r, 0, 1).is_none();
    drop(held);
    let released = read_at(r, 0, 1).is_some();
    refused && released && CONTENDED.load(Ordering::Relaxed) == before + 1
}

fn bs() -> u64 {
    block::block_size() as u64
}

/// The one scratch buffer, allocated on first use.
///
/// # Safety
/// The returned slice aliases a static. One caller at a time, never from an
/// interrupt -- see the module header.
fn scratch() -> Option<&'static mut [u8]> {
    unsafe {
        let slot = SCRATCH.get();
        if *slot == 0 {
            *slot = crate::dev::nvme::alloc_dma(SCRATCH_BYTES)? as u64;
        }
        Some(core::slice::from_raw_parts_mut(*slot as *mut u8, SCRATCH_BYTES))
    }
}

/// Resolve a path in the stored tree, answering its chunk and its kind.
///
/// Reads one directory chunk per path component and **never reads a blob**,
/// which is the whole point: finding out where a 40 KB node lives costs the
/// directories above it and nothing else.
pub fn find(path: &str) -> Option<(cas::ChunkRef, u8)> {
    let parts = super::path_of(path);
    store::with(|st| {
        let m = st.read_manifest(&st.sb.root).ok()?;
        let mut cur = m.entries.first()?.chunk;
        let mut kind = tree::KIND_DIR;
        for p in &parts {
            if kind != tree::KIND_DIR {
                return None;
            }
            let raw = st.get(&cur).ok()?;
            let entries = tree::decode_dir(&raw)?;
            let hit = entries.into_iter().find(|(n, ..)| n == p)?;
            kind = tree::base_kind(hit.1);
            cur = hit.2;
        }
        Some((cur, kind))
    })?
}

/// The chunk of a stored blob, or nothing for a directory or a missing path.
pub fn locate(path: &str) -> Option<cas::ChunkRef> {
    match find(path)? {
        (r, tree::KIND_BLOB) => Some(r),
        _ => None,
    }
}

/// Every blob under a stored directory, as (path, chunk), in name order.
///
/// One walk of the directory chunks rather than a `locate` per node, which is
/// the difference between reading the directories once and reading them once
/// per node per level. Nothing in it reads a body.
pub fn locate_under(root: &str) -> Vec<(String, cas::ChunkRef)> {
    let mut out = Vec::new();
    let start = match find(root) {
        Some((r, tree::KIND_DIR)) => r,
        _ => return out,
    };
    store::with(|st| collect(st, &start, String::from(root), 0, &mut out));
    out
}

fn collect(
    st: &cas::Store,
    r: &cas::ChunkRef,
    path: String,
    depth: usize,
    out: &mut Vec<(String, cas::ChunkRef)>,
) {
    if depth > tree::MAX_DEPTH {
        return;
    }
    let raw = match st.get(r) {
        Ok(v) => v,
        Err(_) => return,
    };
    let entries = match tree::decode_dir(&raw) {
        Some(v) => v,
        None => return,
    };
    for (name, kind, cr, _) in entries {
        let kind = tree::base_kind(kind);
        let mut p = path.clone();
        p.push('/');
        p.push_str(&name);
        if kind == tree::KIND_BLOB {
            out.push((p, cr));
        } else {
            collect(st, &cr, p, depth + 1, out);
        }
    }
}

/// Bytes `off..off+len` of a stored blob, without the blob becoming resident.
///
/// Byte-granular on the outside and block-granular underneath, so asking for
/// the first eighty characters of a node reads one block. A request running
/// past the end is **clamped rather than refused**, because the length is a
/// property of the chunk the caller already holds and a short answer is the
/// truthful one; a request starting past the end answers nothing at all.
pub fn read_at(r: &cas::ChunkRef, off: u64, len: usize) -> Option<Vec<u8>> {
    read_at_with(r, off, len, SCRATCH_BYTES as u64 / bs())
}

/// The same read with the window size supplied.
///
/// **This exists so the loop can be tested.** With the real 64 KiB scratch, a
/// window holds 128 blocks and every read anybody has made here fits in one
/// command -- so the multi-window path, which is the only part of this
/// function with arithmetic worth getting wrong, had no coverage at all. A
/// claim passing `1` walks it a block at a time over a real blob and requires
/// the same bytes back.
pub fn read_at_with(r: &cas::ChunkRef, off: u64, len: usize, cap: u64) -> Option<Vec<u8>> {
    if len == 0 || off >= r.len {
        return Some(Vec::new());
    }
    let len = len.min((r.len - off) as usize);
    let b = bs();
    let cap = cap.max(1);
    // Named, not `_`: `let _ = hold()?` drops the guard immediately and the
    // buffer would be unheld for the whole read it was taken for.
    let _held = hold()?;
    let buf = scratch()?;
    store::with(|st| {
        let mut out = Vec::with_capacity(len);
        let mut pos = off;
        while out.len() < len {
            let blk = pos / b;
            let skip = (pos % b) as usize;
            let want = len - out.len();
            let blocks = ((skip + want) as u64).div_ceil(b).min(cap);
            st.read_blocks(r, blk, blocks as u32, &mut buf[..]).ok()?;
            let avail = blocks as usize * b as usize - skip;
            let take = want.min(avail);
            out.extend_from_slice(&buf[skip..skip + take]);
            pos += take as u64;
        }
        Some(out)
    })?
}

/// The whole of a stored blob, through the ranged path.
///
/// Distinct from `Store::get` in two ways that both matter: it allocates one
/// buffer of the answer's size rather than a fresh DMA region that is never
/// freed, and it does not verify. Use `get` where the answer's integrity is
/// the question and this where its size is.
pub fn read_all(r: &cas::ChunkRef) -> Option<Vec<u8>> {
    read_at(r, 0, r.len as usize)
}

/// The first line of a stored blob, which is what an index holds.
///
/// One block, whatever the blob's size. That is the number the whole retrieval
/// design rests on: a forest node's head is `subject | concept | terms` and
/// its body is everything else, so an index over a corpus that does not fit in
/// memory costs one block per node to build and nothing per node to keep.
///
/// A head longer than a block comes back truncated rather than continued.
/// `tools/forest.py` writes one line per head by construction; this bounds a
/// malformed one instead of reading a body to find out.
pub fn head_line(r: &cas::ChunkRef) -> Option<String> {
    let want = (r.len as usize).min(bs() as usize);
    let raw = read_at(r, 0, want)?;
    let text = String::from_utf8_lossy(&raw);
    let first = text.lines().next()?;
    Some(String::from(first.strip_prefix("head ").unwrap_or(first)))
}

/// Is there a store to read from at all?
pub fn available() -> bool {
    store::with(|st| st.read_manifest(&st.sb.root).is_ok()).unwrap_or(false)
}

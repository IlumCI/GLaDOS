//! Append-only, content-addressed checkpoint store.
//!
//! Three properties, each chosen to survive a specific failure:
//!
//! **Append-only.** Chunks are never rewritten. A checkpoint that fails
//! halfway through cannot damage an earlier one, because it never touches the
//! blocks an earlier one occupies.
//!
//! **Content-addressed.** Every chunk carries the SHA-256 of its contents, and
//! `get` verifies it. Corruption is detected at read time rather than
//! propagating into whatever is being restored.
//!
//! **Dual superblocks.** The obvious design is one root pointer, updated
//! atomically. That is not actually safe: a single 512-byte write is not
//! guaranteed atomic across power loss, so a torn sector could destroy the
//! only root and take every checkpoint with it. Instead two superblocks
//! alternate, each with a sequence number and a checksum over its own
//! contents. Mounting picks the highest sequence number that checksums, so a
//! torn write costs the newest checkpoint and nothing else. This is the same
//! reasoning behind ZFS uberblocks and F2FS checkpoints.
//!
//! Checkpoints form an immutable chain: each manifest names its predecessor by
//! hash and location, so history is walkable and rollback is a matter of
//! pointing the superblock at an older manifest.

#![allow(dead_code)]

use super::block;
use super::sha256;
use crate::dev::nvme;
use crate::gfx::console;
use crate::kprintln;
use alloc::vec;
use alloc::vec::Vec;

const SB_MAGIC: &[u8; 8] = b"GLADOSCP";
const MF_MAGIC: &[u8; 8] = b"GLADOSMF";
const VERSION: u32 = 1;

/// Bytes of the superblock covered by its checksum.
const SB_CHECKED: usize = 480;
const SB_HASH_OFF: usize = 480;

pub const ENTRY_SIZE: usize = 64;
pub const NAME_LEN: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Error {
    NoDevice,
    NotFormatted,
    Io(block::Error),
    Corrupt,
    HashMismatch,
    Full,
    Unsafe,
    TooManyEntries,
    /// A collection's two reachability walks disagreed, so it freed nothing.
    Disagree,
}

impl From<block::Error> for Error {
    fn from(e: block::Error) -> Self {
        match e {
            // **A refused write is not an I/O error and must not read as one.**
            // `nvme::write` answers `0xFFFC` when writes have never been
            // unlocked, which is the ordinary state of every boot that did not
            // run `store init` -- and it arrived here as `Io(Io(65532))`, a
            // number that sends an operator to look at the disk. `Unsafe`
            // already had the right words attached to it and simply never
            // reached them.
            block::Error::Io(nvme::ERR_LOCKED) => Error::Unsafe,
            other => Error::Io(other),
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct ChunkRef {
    pub hash: [u8; 32],
    pub lba: u64,
    pub len: u64,
}

impl ChunkRef {
    pub fn is_none(&self) -> bool {
        self.len == 0 && self.lba == 0
    }
}

#[derive(Clone, Copy)]
pub struct Superblock {
    pub seq: u64,
    pub region_start: u64,
    pub region_blocks: u64,
    pub alloc_next: u64,
    pub root: ChunkRef,
    pub checkpoints: u64,
}

pub struct Entry {
    pub name: [u8; NAME_LEN],
    pub chunk: ChunkRef,
}

pub struct Manifest {
    pub seq: u64,
    pub prev: ChunkRef,
    pub entries: Vec<Entry>,
    /// Seconds since 1970, or zero when unknown.
    ///
    /// Stored in the four bytes between the entry count and the entries, which
    /// were already being written as zero -- so this costs no format change and
    /// every manifest written before it reads back as "no time recorded"
    /// rather than as a wrong one.
    pub time: u32,
}

fn bs() -> u64 {
    block::block_size() as u64
}

fn blocks_for(bytes: u64) -> u64 {
    bytes.div_ceil(bs())
}

/// A page-aligned scratch buffer sized to whole blocks, freed when dropped.
///
/// **It used to be leaked**, and what that cost is worth keeping: every `put`,
/// `get`, `commit`, `format` and `mount` took `4096 + ceil(size / 512) * 512`
/// bytes of heap for good. Measured under QEMU, a first snapshot of 812 blocks
/// moved the heap by 3,512,864 bytes, and a forest import of 8,913 nodes by
/// 52 MB. Survivable for a snapshot of notes; fatal for the installer, where a
/// desktop's worth of packages is ~1.8 GB written through here, so roughly the
/// same again of heap gone.
///
/// **Not one static scratch, which the old note proposed, and the reason is
/// preemption.** A guest reading a stored file goes through `get` on its own
/// task while the shell's autosnap goes through `put`; with one buffer, the
/// first to be preempted mid-I/O has its bytes overwritten by the second. A
/// buffer owned by its scope cannot be shared by accident.
pub(crate) struct DmaBuf {
    ptr: *mut u8,
    layout: core::alloc::Layout,
    len: usize,
}

impl DmaBuf {
    /// `bytes` rounded up to whole blocks, at least one.
    pub(crate) fn new(bytes: usize) -> Result<Self, Error> {
        let rounded = (blocks_for(bytes as u64) as usize).max(1) * bs() as usize;
        let layout = core::alloc::Layout::from_size_align(rounded, 4096).map_err(|_| Error::Full)?;
        let ptr = unsafe { alloc::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(Error::NoDevice);
        }
        Ok(Self { ptr, layout, len: rounded })
    }
}

impl core::ops::Deref for DmaBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl core::ops::DerefMut for DmaBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for DmaBuf {
    fn drop(&mut self) {
        unsafe { alloc::alloc::dealloc(self.ptr, self.layout) };
    }
}

fn dma(bytes: usize) -> Result<DmaBuf, Error> {
    DmaBuf::new(bytes)
}

/// The most a single `put` or `get` holds in scratch at once. A blob larger
/// than this streams through it, so writing a 600 MB model costs 2 MiB of
/// heap rather than a second copy of the model.
pub const STREAM: usize = 2 << 20;

#[inline]
fn put_u64(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}
#[inline]
fn get_u64(b: &[u8], o: usize) -> u64 {
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(v)
}
#[inline]
fn put_u32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
#[inline]
fn get_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

// --- superblock ---------------------------------------------------------

fn encode_sb(sb: &Superblock, out: &mut [u8]) {
    for v in out.iter_mut() {
        *v = 0;
    }
    out[0..8].copy_from_slice(SB_MAGIC);
    put_u32(out, 8, VERSION);
    put_u64(out, 16, sb.seq);
    put_u64(out, 24, sb.region_start);
    put_u64(out, 32, sb.region_blocks);
    put_u64(out, 40, sb.alloc_next);
    put_u64(out, 48, sb.root.lba);
    put_u64(out, 56, sb.root.len);
    out[64..96].copy_from_slice(&sb.root.hash);
    put_u64(out, 96, sb.checkpoints);
    let h = sha256::hash(&out[..SB_CHECKED]);
    out[SB_HASH_OFF..SB_HASH_OFF + 32].copy_from_slice(&h);
}

fn decode_sb(b: &[u8]) -> Option<Superblock> {
    if &b[0..8] != SB_MAGIC || get_u32(b, 8) != VERSION {
        return None;
    }
    // The checksum is what makes a torn write detectable rather than merely
    // unlikely.
    let want = sha256::hash(&b[..SB_CHECKED]);
    if want != b[SB_HASH_OFF..SB_HASH_OFF + 32] {
        return None;
    }
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&b[64..96]);
    Some(Superblock {
        seq: get_u64(b, 16),
        region_start: get_u64(b, 24),
        region_blocks: get_u64(b, 32),
        alloc_next: get_u64(b, 40),
        root: ChunkRef { lba: get_u64(b, 48), len: get_u64(b, 56), hash },
        checkpoints: get_u64(b, 96),
    })
}

/// Superblocks alternate by sequence parity, so a failed write never lands on
/// the slot holding the last good one.
fn sb_lba(region_start: u64, seq: u64) -> u64 {
    region_start + (seq % 2)
}

// --- the store ----------------------------------------------------------

pub struct Store {
    pub sb: Superblock,
    /// Runs of blocks below `alloc_next` that nothing retained refers to, as
    /// `(first, blocks)`, sorted. **Only in memory**: a GC pass recomputes it
    /// from what is reachable, so the on-disk format does not change and an
    /// older kernel reads a collected store as it reads any other. Empty after
    /// mount, which is the safe direction -- the store just appends until it
    /// next runs out of tail and collects.
    pub free: Vec<(u64, u64)>,
    /// Blocks `put` has written this boot, for reporting what a snapshot
    /// cost now that a write may land below `alloc_next`.
    pub written_blocks: u64,
}

/// The free runs in `[lo, hi)` given the runs in use, which may overlap and
/// come in any order. Pure, so its claims need no disk.
pub fn sweep(lo: u64, hi: u64, used: &mut Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    used.sort_unstable();
    let mut free = Vec::new();
    let mut at = lo;
    for &(s, n) in used.iter() {
        let e = s.saturating_add(n);
        if e <= at {
            continue;
        }
        if s > at && s <= hi {
            free.push((at, s - at));
        }
        at = at.max(e);
        if at >= hi {
            break;
        }
    }
    if at < hi {
        free.push((at, hi - at));
    }
    free
}

/// First fit: `need` contiguous blocks out of `free`, which shrinks. A blob is
/// stored as one run, so a run too small for it is skipped, not split.
pub fn take(free: &mut Vec<(u64, u64)>, need: u64) -> Option<u64> {
    let i = free.iter().position(|&(_, n)| n >= need)?;
    let (s, n) = free[i];
    if n == need {
        free.remove(i);
    } else {
        free[i] = (s + need, n - need);
    }
    Some(s)
}

impl Store {
    /// Format a region. Refuses to touch anything a partition claims.
    pub fn format(region_start: u64, region_blocks: u64) -> Result<Self, Error> {
        verify_region_safe(region_start, region_blocks)?;
        if region_blocks < 8 {
            return Err(Error::Full);
        }

        let sb = Superblock {
            seq: 1,
            region_start,
            region_blocks,
            // Slots 0 and 1 are the superblocks; the chunk area starts after.
            alloc_next: region_start + 2,
            root: ChunkRef::default(),
            checkpoints: 0,
        };

        let mut buf = dma(bs() as usize)?;
        encode_sb(&sb, &mut buf[..bs() as usize]);
        block::write(sb_lba(region_start, sb.seq), 1, &buf)?;
        Ok(Self { sb, free: Vec::new(), written_blocks: 0 })
    }

    /// Read both superblocks and adopt the newest that verifies.
    pub fn mount(region_start: u64) -> Result<Self, Error> {
        let mut buf = dma(2 * bs() as usize)?;
        block::read(region_start, 2, &mut buf)?;
        let a = decode_sb(&buf[..bs() as usize]);
        let b = decode_sb(&buf[bs() as usize..2 * bs() as usize]);
        let sb = match (a, b) {
            (Some(x), Some(y)) => Some(if x.seq >= y.seq { x } else { y }),
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (None, None) => None,
        }
        .ok_or(Error::NotFormatted)?;
        Ok(Self { sb, free: Vec::new(), written_blocks: 0 })
    }

    pub fn free_blocks(&self) -> u64 {
        (self.sb.region_start + self.sb.region_blocks).saturating_sub(self.sb.alloc_next)
            + self.free.iter().map(|&(_, n)| n).sum::<u64>()
    }

    /// Append a chunk. Returns its content address and location.
    pub fn put(&mut self, data: &[u8]) -> Result<ChunkRef, Error> {
        let need = blocks_for(data.len() as u64).max(1);
        // A collected run first, then the tail. Taking from the tail moves
        // `alloc_next`, which is committed with the next superblock; taking a
        // run does not, because the run is below it.
        let tail = (self.sb.region_start + self.sb.region_blocks).saturating_sub(self.sb.alloc_next);
        let (lba, from_tail) = match take(&mut self.free, need) {
            Some(l) => (l, false),
            None if need <= tail => (self.sb.alloc_next, true),
            None => return Err(Error::Full),
        };
        let mut buf = dma(data.len().min(STREAM))?;
        let per = buf.len();
        let mut off = 0usize;
        // At least one pass, so an empty blob still owns its one block.
        loop {
            let n = (data.len() - off).min(per);
            buf[..n].copy_from_slice(&data[off..off + n]);
            // The tail of the last block, and a stale tail of the scratch from
            // the previous piece, must not reach the disk as data.
            buf[n..].fill(0);
            let blocks = blocks_for(n as u64).max(1);
            block::write(lba + (off as u64 / bs()), blocks as u32, &buf)?;
            off += per;
            if off >= data.len() {
                break;
            }
        }
        if from_tail {
            self.sb.alloc_next += need;
        }
        self.written_blocks += need;
        Ok(ChunkRef { hash: sha256::hash(data), lba, len: data.len() as u64 })
    }

    /// Read a chunk and verify it against its own address.
    pub fn get(&self, r: &ChunkRef) -> Result<Vec<u8>, Error> {
        if r.len == 0 {
            return Ok(Vec::new());
        }
        let len = r.len as usize;
        let mut buf = dma(len.min(STREAM))?;
        let per = buf.len();
        let mut data = Vec::with_capacity(len);
        let mut off = 0usize;
        while off < len {
            let n = (len - off).min(per);
            let blocks = blocks_for(n as u64).max(1);
            block::read(r.lba + (off as u64 / bs()), blocks as u32, &mut buf)?;
            data.extend_from_slice(&buf[..n]);
            off += n;
        }
        if sha256::hash(&data) != r.hash {
            return Err(Error::HashMismatch);
        }
        Ok(data)
    }

    /// Read part of a chunk, into a buffer the caller already owns.
    ///
    /// The whole point of streaming, and the reason it is cheap here: `put`
    /// writes a blob to one contiguous run of blocks, so any range inside it
    /// is a contiguous run too. Reading the middle of a 30 GiB model costs one
    /// command, and the model never has to be resident to be read from.
    ///
    /// **Block-granular, and allocation-free, on purpose.** `get` allocates a
    /// scratch per call, which is fine for a manifest and wasteful for a path
    /// that runs once per layer per token, so this takes the caller's buffer
    /// and allocates nothing. Page-align it (`nvme::alloc_dma`)
    /// or pay an extra PRP per command.
    ///
    /// No hash is checked, and cannot be: the hash covers the whole blob. A
    /// streamed weight is verified once when it is imported and trusted after
    /// that, which is the trade streaming always makes.
    pub fn read_blocks(
        &self,
        r: &ChunkRef,
        first_block: u64,
        blocks: u32,
        buf: &mut [u8],
    ) -> Result<(), Error> {
        if blocks == 0 {
            return Ok(());
        }
        // Past the end of the blob is a caller bug, not a short read: it would
        // otherwise return whatever the next chunk holds, which is a plausible
        // looking tensor belonging to something else.
        let have = blocks_for(r.len).max(1);
        if first_block + blocks as u64 > have {
            return Err(Error::Io(block::Error::TooSmall));
        }
        if buf.len() < blocks as usize * bs() as usize {
            return Err(Error::Io(block::Error::TooSmall));
        }
        block::read(r.lba + first_block, blocks, buf).map_err(Error::Io)
    }

    /// Write a new checkpoint naming the current one as its predecessor.
    ///
    /// Order matters: the manifest chunk is written and its data is on disk
    /// before any superblock update points at it. A crash between the two
    /// leaves an orphaned chunk, which wastes space and breaks nothing.
    pub fn commit(&mut self, entries: &[Entry]) -> Result<ChunkRef, Error> {
        let seq = self.sb.seq + 1;
        let size = 72 + entries.len() * ENTRY_SIZE;
        let mut m = vec![0u8; size];
        m[0..8].copy_from_slice(MF_MAGIC);
        put_u64(&mut m, 8, seq);
        put_u64(&mut m, 16, self.sb.root.lba);
        put_u64(&mut m, 24, self.sb.root.len);
        m[32..64].copy_from_slice(&self.sb.root.hash);
        put_u32(&mut m, 64, entries.len() as u32);
        let now = crate::dev::rtc::now().map(|d| crate::dev::rtc::unix_seconds(&d)).unwrap_or(0);
        put_u32(&mut m, 68, now);
        for (i, e) in entries.iter().enumerate() {
            let o = 72 + i * ENTRY_SIZE;
            m[o..o + NAME_LEN].copy_from_slice(&e.name);
            m[o + 16..o + 48].copy_from_slice(&e.chunk.hash);
            put_u64(&mut m, o + 48, e.chunk.lba);
            put_u64(&mut m, o + 56, e.chunk.len);
        }

        let root = self.put(&m)?;

        let mut sb = self.sb;
        sb.seq = seq;
        sb.root = root;
        sb.checkpoints += 1;

        let mut buf = dma(bs() as usize)?;
        encode_sb(&sb, &mut buf[..bs() as usize]);
        block::write(sb_lba(sb.region_start, seq), 1, &buf)?;
        self.sb = sb;
        Ok(root)
    }

    pub fn read_manifest(&self, r: &ChunkRef) -> Result<Manifest, Error> {
        let m = self.get(r)?;
        if m.len() < 72 || &m[0..8] != MF_MAGIC {
            return Err(Error::Corrupt);
        }
        let count = get_u32(&m, 64) as usize;
        if m.len() < 72 + count * ENTRY_SIZE {
            return Err(Error::Corrupt);
        }
        let mut prev_hash = [0u8; 32];
        prev_hash.copy_from_slice(&m[32..64]);
        let mut entries = Vec::new();
        for i in 0..count {
            let o = 72 + i * ENTRY_SIZE;
            let mut name = [0u8; NAME_LEN];
            name.copy_from_slice(&m[o..o + NAME_LEN]);
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&m[o + 16..o + 48]);
            entries.push(Entry {
                name,
                chunk: ChunkRef { hash, lba: get_u64(&m, o + 48), len: get_u64(&m, o + 56) },
            });
        }
        Ok(Manifest {
            seq: get_u64(&m, 8),
            prev: ChunkRef {
                lba: get_u64(&m, 16),
                len: get_u64(&m, 24),
                hash: prev_hash,
            },
            entries,
            time: get_u32(&m, 68),
        })
    }

    /// Point the store back at an earlier checkpoint.
    ///
    /// Nothing is erased. The newer manifests remain on disk and remain
    /// reachable by hash -- rolling back is itself an append, so it is
    /// reversible. That is the property that makes this usable as a recovery
    /// tool rather than a destructive one.
    pub fn rollback_to(&mut self, target: ChunkRef) -> Result<(), Error> {
        // Verify before committing to it: refusing to roll back to a corrupt
        // checkpoint is the whole job.
        let _ = self.read_manifest(&target)?;
        let mut sb = self.sb;
        sb.seq += 1;
        sb.root = target;
        let mut buf = dma(bs() as usize)?;
        encode_sb(&sb, &mut buf[..bs() as usize]);
        block::write(sb_lba(sb.region_start, sb.seq), 1, &buf)?;
        self.sb = sb;
        Ok(())
    }
}

/// Confirm a region is one we are allowed to write to.
///
/// Two acceptable cases: it overlaps no partition at all, or it lies entirely
/// inside a partition tagged as ours. Everything else is refused, including
/// spilling past the end of our own partition into a neighbour.
///
/// Kept separate from `format` because unlocking writes on an already-mounted
/// store has to make exactly the same check, and a safety test that exists in
/// two copies eventually exists in two versions.
pub fn verify_region_safe(region_start: u64, region_blocks: u64) -> Result<(), Error> {
    let layout = block::scan().map_err(Error::Io)?;
    if let Some(p) = layout.overlaps(region_start, region_blocks) {
        if p.type_guid != GLADOS_TYPE_GUID
            || region_start < p.start_lba
            || region_start + region_blocks > p.end_lba()
        {
            return Err(Error::Unsafe);
        }
    }
    Ok(())
}

/// GPT type GUID for a GLaDOS store partition, in on-disk mixed-endian form.
///
/// Text form: b7e1f4a2-9c3d-4e58-a061-2f8d7c4b93e5
///
/// Tagging the partition means the store is found by *identity* rather than by
/// inferring which space looks unused. That matters on a disk that is fully
/// allocated: the space freed by shrinking C: lands between C: and the
/// recovery partition, not at the end of the disk, so "unclaimed tail" finds
/// nothing. It is also simply safer -- an explicit tag cannot be confused with
/// a region some vendor tool quietly uses without declaring a partition.
pub const GLADOS_TYPE_GUID: [u8; 16] = [
    0xa2, 0xf4, 0xe1, 0xb7, 0x3d, 0x9c, 0x58, 0x4e, 0xa0, 0x61, 0x2f, 0x8d, 0x7c, 0x4b, 0x93, 0xe5,
];

/// Where the checkpoint store lives.
///
/// A partition tagged as ours is preferred and is the real answer on hardware.
/// The unclaimed-tail fallback exists so a bare QEMU image works without
/// anyone having to build a partition table first.
pub fn find_store_region(min_blocks: u64) -> Option<(u64, u64)> {
    if let Ok(layout) = block::scan() {
        if let Some(p) = layout
            .partitions
            .iter()
            .find(|p| p.type_guid == GLADOS_TYPE_GUID)
        {
            if p.block_count >= min_blocks {
                return Some((p.start_lba, p.block_count));
            }
        }
    }
    find_free_region(min_blocks)
}

/// Choose a region beyond every partition.
///
/// "Unclaimed" is not automatically "safe" -- firmware and vendor tools
/// sometimes use the tail of a disk without declaring a partition -- but on a
/// disk fully allocated to Windows this returns nothing at all, which is the
/// behaviour that matters.
pub fn find_free_region(min_blocks: u64) -> Option<(u64, u64)> {
    let layout = block::scan().ok()?;
    let total = block::block_count();
    // Leave the last MiB alone: GPT keeps its backup header and entry array
    // there, and overwriting it would make the disk look unpartitioned.
    let reserve_tail = (1024 * 1024) / block::block_size() as u64;
    let start = layout.highest_used_lba().max(2048);
    let end = total.saturating_sub(reserve_tail);
    if end <= start {
        return None;
    }
    let blocks = end - start;
    if blocks < min_blocks {
        return None;
    }
    Some((start, blocks))
}

/// A blob larger than one command, and a range out of the middle of it.
///
/// Both halves matter. The round trip proves a PRP-list transfer actually
/// lands -- before the list existed anything over 8 KiB was refused outright,
/// so a checkpoint could only ever be moved in two-page pieces. The ranged
/// read proves the thing streaming is built on: that the middle of a blob can
/// be fetched without the blob being resident, and that it is byte-for-byte
/// what the whole-blob read would have given.
/// What `read_blocks` actually costs, which nothing here has ever measured.
///
/// **The paged KV cache turns on this number and on no other.** Holding the
/// trained 40,960-token context outright is 2,520 MiB, which this machine does
/// not have; keeping a per-page min/max index resident and fetching the few
/// pages a query wants is 72 MiB at page 32, which it does. The host has
/// already measured the half that can be measured there -- four or more pages
/// fetched gives perfect recall at 6.25% of the cache -- and the half left is
/// whether the disk can deliver 12 MiB inside a decode step.
///
/// So this reports two things that answer opposite questions, and reporting
/// only the first is how a design like this gets built and then found to be
/// slow. **Sequential** large reads give bandwidth, which is the number if
/// pages are laid out so one read fetches many. **Scattered** small reads give
/// per-read latency, which is the number if they are not -- and the naive
/// layout needs 6 pages x 8 kv heads x 28 layers = 1,344 separate reads a
/// step. Between those two the design is either free or impossible.
///
/// Nothing is written. It reads blocks the store has already committed, which
/// are real data rather than a freshly written pattern, so no write gate is
/// touched and the figure is not flattered by reading back what is still in a
/// controller's write cache.
pub fn read_bench(reps: usize) -> bool {
    // The whole formatted region rather than just `alloc_next`. This times the
    // disk and never looks at what the bytes mean, so an unallocated block is
    // as good as a committed one -- and `alloc_next` after one empty snapshot
    // is 142 blocks, which silently skipped every size above 32 KiB on the
    // first run and printed a table that looked complete.
    let Some((start, end)) = crate::store::with(|s| {
        (s.sb.region_start, s.sb.region_start + s.sb.region_blocks)
    }) else {
        kprintln!("  no store mounted -- 'store init', then 'snap', then this");
        return false;
    };
    let bsz = bs();
    let span = end.saturating_sub(start);
    if span < 64 {
        kprintln!("  only {} block(s) committed -- 'snap' something first", span);
        return false;
    }

    // One buffer, page-aligned, allocated once and freed on the way out.
    const BIG: usize = 2 * 1024 * 1024;
    let Ok(mut owned) = DmaBuf::new(BIG) else {
        kprintln!("  could not get a {} B DMA buffer", BIG);
        return false;
    };
    let buf: &mut [u8] = &mut owned;

    let mhz = crate::time::tsc_mhz().max(1);
    if let Some(h) = crate::cpu::hypervisor() {
        console::set_color(console::LTRED);
        // The same caution `dev::power` states about its MSRs, and for the
        // same reason: an emulator answers, and the answer is about the
        // emulator rather than about the disk.
        kprintln!(
            "  a hypervisor is present ({}), so this measures the host's",
            core::str::from_utf8(&h).unwrap_or("?")
        );
        kprintln!("  page cache and not this machine's NVMe. Run it on the GF63.");
        console::set_color(console::WHITE);
    }
    kprintln!("  reading committed blocks {}..{}, nothing written", start, end);

    let mut ok = true;
    kprintln!("  sequential, best of {}:", reps);
    for kib in [4u64, 32, 128, 512, 2048] {
        let blocks = (kib * 1024 / bsz).max(1) as u32;
        if blocks as u64 > span {
            kprintln!("    {:>5} KiB   skipped, the region is only {} KiB",
                      kib, span * bsz / 1024);
            continue;
        }
        let mut best = u64::MAX;
        for r in 0..reps {
            // A different offset each repeat, so a controller that cached the
            // last read cannot answer the next one from it.
            let off = (r as u64 * blocks as u64) % (span - blocks as u64).max(1);
            let cr = ChunkRef { hash: [0u8; 32], lba: start + off, len: span * bsz };
            let t0 = crate::time::rdtsc();
            let got = crate::store::with(|s| s.read_blocks(&cr, 0, blocks, buf));
            let dt = crate::time::rdtsc() - t0;
            if !matches!(got, Some(Ok(()))) {
                ok = false;
                break;
            }
            best = best.min(dt);
        }
        if best == u64::MAX {
            continue;
        }
        let us = (best / mhz).max(1);
        let mbs = (blocks as u64 * bsz) * 1_000_000 / us / (1024 * 1024);
        kprintln!("    {:>5} KiB   {:>7} us   {:>5} MB/s", kib, us, mbs);
    }

    // --- which layout, which is what the first run turned this into ------
    //
    // A decode step at page 32 wants 6 pages per kv head per layer, and what
    // that costs depends entirely on how a page sits on disk. The first run
    // measured 261 us for a 9 KiB read against 112 us for a 32 KiB one:
    // per-command overhead dominates and bandwidth is nearly irrelevant at
    // these sizes, so the layout is what decides whether this is affordable.
    // Three candidates, timed against each other rather than one assumed:
    //
    //   per head    P tokens, one head, one layer. 6*8*28 reads. Finest
    //               selection, worst I/O.
    //   per layer   P tokens, all 8 heads, one layer. 6*28 reads. The heads
    //               in a layer must then agree on which pages they want.
    //   whole       P tokens, every head and layer. 6 reads. One selection
    //               for the entire model.
    //
    // The trade is real and is **not** resolved here: a coarser layout reads
    // far less and chooses far worse, and how much worse is a host
    // measurement nobody has taken. This says only what each one costs.
    let page_tokens = 32u64;
    let per_tok_head = 2 * (128 + (128 / 32) * 4);
    let layouts: [(&str, u64, u64); 3] = [
        ("per head ", 6 * 8 * 28, page_tokens * per_tok_head),
        ("per layer", 6 * 28, page_tokens * per_tok_head * 8),
        ("whole    ", 6, page_tokens * per_tok_head * 8 * 28),
    ];
    kprintln!("  what one decode step costs at page {}, by layout:", page_tokens);
    for (name, reads, bytes) in layouts {
        let rb = bytes.div_ceil(bsz) as u32;
        if rb as u64 > span {
            kprintln!("    {}  skipped, one read exceeds the region", name);
            continue;
        }
        let mut best = u64::MAX;
        for r in 0..reps {
            let t0 = crate::time::rdtsc();
            for i in 0..reads {
                let off = (i * 2_654_435_761 + r as u64 * 7919)
                    % (span - rb as u64).max(1);
                let cr = ChunkRef { hash: [0u8; 32], lba: start + off, len: span * bsz };
                if !matches!(
                    crate::store::with(|s| s.read_blocks(&cr, 0, rb, buf)),
                    Some(Ok(()))
                ) {
                    ok = false;
                    break;
                }
            }
            best = best.min(crate::time::rdtsc() - t0);
        }
        let us = (best / mhz).max(1);
        let total = reads * rb as u64 * bsz;
        console::set_color(if us < 20_000 { console::LTGREEN } else { console::YELLOW });
        kprintln!(
            "    {}  {:>4} reads x {:>4} KiB = {:>5} KiB   {:>7} us   {:>4} us/read",
            name, reads, rb as u64 * bsz / 1024, total / 1024, us, us / reads
        );
        console::set_color(console::WHITE);
    }
    kprintln!("  against a decode step of roughly 200 ms on this checkpoint");
    ok
}

/// What writing costs: raw sequential block writes, and SHA-256 over the same
/// bytes, timed apart because a system-slot write does both and either could
/// be the one that bounds it.
///
/// **Nothing is consumed.** The blocks written are in the store's unallocated
/// tail, past `alloc_next`, which no checkpoint references and the next `put`
/// overwrites. So the bench can be run as often as wanted without the store
/// growing, which matters while the store has no GC.
pub fn write_bench(mib: u64) -> bool {
    let Some((next, end)) = crate::store::with(|s| (s.sb.alloc_next, s.sb.region_start + s.sb.region_blocks))
    else {
        kprintln!("  no store mounted");
        return false;
    };
    if !nvme::holds(nvme::Holder::Store) {
        kprintln!("  writes are locked -- 'store unlock' first");
        return false;
    }
    let bsz = bs();
    let want = mib * 1024 * 1024 / bsz;
    // Leave the last superblock-sized margin alone and refuse rather than wrap.
    if end.saturating_sub(next) < want + 8 {
        kprintln!("  only {} MiB unallocated -- ask for less", end.saturating_sub(next) * bsz >> 20);
        return false;
    }
    let Ok(mut buf) = DmaBuf::new(STREAM) else {
        kprintln!("  could not get a {} B buffer", STREAM);
        return false;
    };
    for (i, b) in buf.iter_mut().enumerate() {
        *b = (i as u32).wrapping_mul(2654435761).to_le_bytes()[3];
    }
    let per = (STREAM as u64 / bsz) as u32;
    let mhz = crate::time::tsc_mhz().max(1);

    let t0 = crate::time::rdtsc();
    let mut done = 0u64;
    while done < want {
        let n = (want - done).min(per as u64) as u32;
        if block::write(next + done, n, &buf).is_err() {
            kprintln!("  write failed at block {}", next + done);
            return false;
        }
        done += n as u64;
    }
    let w_us = ((crate::time::rdtsc() - t0) / mhz).max(1);

    let t1 = crate::time::rdtsc();
    let mut h = sha256::Sha256::new();
    for _ in 0..(done * bsz).div_ceil(STREAM as u64) {
        h.update(&buf);
    }
    let _ = h.finish();
    let h_us = ((crate::time::rdtsc() - t1) / mhz).max(1);

    let bytes = done * bsz;
    let rate = |us: u64| bytes * 1_000_000 / us / (1024 * 1024);
    if crate::cpu::hypervisor().is_some() {
        console::set_color(console::YELLOW);
        kprintln!("  under a hypervisor the disk is a host file, so this measures the host");
        console::set_color(console::WHITE);
    }
    kprintln!("  {} MiB written raw     {:>8} us   {:>5} MiB/s", bytes >> 20, w_us, rate(w_us));
    kprintln!("  {} MiB hashed SHA-256  {:>8} us   {:>5} MiB/s", bytes >> 20, h_us, rate(h_us));
    kprintln!("  so a 6 GiB system slot is about {} s at the slower of the two",
        (6u64 << 30) / (rate(w_us).min(rate(h_us)).max(1) << 20));
    true
}

pub fn stream_selftest() -> bool {
    use alloc::vec::Vec;

    let Some(bs) = crate::dev::nvme::with(|n| n.block_size as usize) else {
        return true;
    };
    let pattern = |n: usize, salt: u32| -> Vec<u8> {
        (0..n).map(|i| ((i as u32 ^ salt).wrapping_mul(2654435761) >> 24) as u8).collect()
    };

    let mut ok = true;
    // Two sizes. The small one is past the old two-page ceiling and not a
    // whole number of blocks. The large one is two full `STREAM` pieces and an
    // odd tail, because a blob under 2 MiB never runs the streaming loop at
    // all, and an off-by-one piece offset there writes a plausible blob with
    // its second piece in the wrong place.
    for (n, salt) in [(300 * 1024 + 7, 0u32), (2 * STREAM + 4100 + 3, 0x5a5a)] {
        let data = pattern(n, salt);
        crate::store::with(|s| {
            let Ok(r) = s.put(&data) else {
                crate::kprintln!("  could not store {} B", n);
                ok = false;
                return;
            };
            match s.get(&r) {
                Ok(back) if back == data => {}
                Ok(_) => {
                    crate::kprintln!("  a {} B blob did not survive the round trip", n);
                    ok = false;
                    return;
                }
                Err(_) => {
                    crate::kprintln!("  a {} B blob could not be read back", n);
                    ok = false;
                    return;
                }
            }
            // The middle by block, without the whole -- and for the large blob,
            // straddling the boundary between its first and second pieces.
            let blocks = 40u32;
            let first = if n > STREAM { (STREAM / bs) as u64 - 20 } else { 17 };
            let Ok(mut owned) = DmaBuf::new(blocks as usize * bs) else {
                return;
            };
            let buf: &mut [u8] = &mut owned;
            if s.read_blocks(&r, first, blocks, buf).is_err() {
                crate::kprintln!("  ranged read refused");
                ok = false;
                return;
            }
            let lo = first as usize * bs;
            if buf[..blocks as usize * bs] != data[lo..lo + blocks as usize * bs] {
                crate::kprintln!("  a ranged read of a {} B blob disagreed with the whole", n);
                ok = false;
            }
        });
        if !ok {
            return false;
        }
    }

    // And the store must give back what it takes. Eight round trips of the
    // large blob through `put` and `get` (16 MiB of I/O) used to cost the heap
    // ~16 MiB for good; now only the returned copies, which are dropped.
    let (before, _) = crate::mem::heap::HEAP.stats();
    let data = pattern(2 * STREAM + 4100 + 3, 7);
    for _ in 0..8 {
        let r = crate::store::with(|s| s.put(&data).ok().and_then(|r| s.get(&r).ok().map(|_| ())));
        if r.flatten().is_none() {
            crate::kprintln!("  the leak check could not round-trip its blob");
            return false;
        }
    }
    drop(data);
    let (after, _) = crate::mem::heap::HEAP.stats();
    let grew = after.saturating_sub(before);
    // One block of slack per round trip, against the ~4.3 MB a trip leaked.
    if grew > 8 * 4096 {
        crate::kprintln!("  eight round trips grew the heap by {} B -- the store is leaking", grew);
        return false;
    }
    ok
}

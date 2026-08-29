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
}

impl From<block::Error> for Error {
    fn from(e: block::Error) -> Self {
        Error::Io(e)
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

/// A page-aligned scratch buffer sized to whole blocks.
fn dma(bytes: usize) -> Result<&'static mut [u8], Error> {
    let rounded = (blocks_for(bytes as u64) as usize).max(1) * bs() as usize;
    let p = nvme::alloc_dma(rounded + 4096).ok_or(Error::NoDevice)?;
    Ok(unsafe { core::slice::from_raw_parts_mut(p, rounded) })
}

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

        let buf = dma(bs() as usize)?;
        encode_sb(&sb, &mut buf[..bs() as usize]);
        block::write(sb_lba(region_start, sb.seq), 1, buf)?;
        Ok(Self { sb })
    }

    /// Read both superblocks and adopt the newest that verifies.
    pub fn mount(region_start: u64) -> Result<Self, Error> {
        let buf = dma(2 * bs() as usize)?;
        block::read(region_start, 2, buf)?;
        let a = decode_sb(&buf[..bs() as usize]);
        let b = decode_sb(&buf[bs() as usize..2 * bs() as usize]);
        let sb = match (a, b) {
            (Some(x), Some(y)) => Some(if x.seq >= y.seq { x } else { y }),
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (None, None) => None,
        }
        .ok_or(Error::NotFormatted)?;
        Ok(Self { sb })
    }

    pub fn free_blocks(&self) -> u64 {
        (self.sb.region_start + self.sb.region_blocks).saturating_sub(self.sb.alloc_next)
    }

    /// Append a chunk. Returns its content address and location.
    pub fn put(&mut self, data: &[u8]) -> Result<ChunkRef, Error> {
        let need = blocks_for(data.len() as u64).max(1);
        if need > self.free_blocks() {
            return Err(Error::Full);
        }
        let lba = self.sb.alloc_next;
        let buf = dma(data.len())?;
        buf[..data.len()].copy_from_slice(data);
        for v in buf[data.len()..].iter_mut() {
            *v = 0;
        }
        block::write(lba, need as u32, buf)?;
        self.sb.alloc_next += need;
        Ok(ChunkRef { hash: sha256::hash(data), lba, len: data.len() as u64 })
    }

    /// Read a chunk and verify it against its own address.
    pub fn get(&self, r: &ChunkRef) -> Result<Vec<u8>, Error> {
        if r.len == 0 {
            return Ok(Vec::new());
        }
        let need = blocks_for(r.len).max(1);
        let buf = dma(r.len as usize)?;
        block::read(r.lba, need as u32, buf)?;
        let data = buf[..r.len as usize].to_vec();
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
    /// **Block-granular, and allocation-free, on purpose.** `get` calls `dma`,
    /// which calls `alloc_zeroed` and hands back a `&'static mut [u8]` -- it
    /// never frees. That is survivable for a manifest read at boot and fatal
    /// for a path that runs once per layer per token, so this takes the
    /// caller's buffer and allocates nothing. Page-align it (`nvme::alloc_dma`)
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

        let buf = dma(bs() as usize)?;
        encode_sb(&sb, &mut buf[..bs() as usize]);
        block::write(sb_lba(sb.region_start, seq), 1, buf)?;
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
        let buf = dma(bs() as usize)?;
        encode_sb(&sb, &mut buf[..bs() as usize]);
        block::write(sb_lba(sb.region_start, sb.seq), 1, buf)?;
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
pub fn stream_selftest() -> bool {
    use alloc::vec;
    use alloc::vec::Vec;

    let Some(bs) = crate::dev::nvme::with(|n| n.block_size as usize) else {
        return true;
    };
    // Comfortably past the old two-page ceiling and not a whole number of
    // blocks, so the tail is short.
    let n = 300 * 1024 + 7;
    let data: Vec<u8> = (0..n)
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
        .collect();

    let mut ok = true;
    crate::store::with(|s| {
        let Ok(r) = s.put(&data) else {
            crate::kprintln!("  could not store {} B", n);
            ok = false;
            return;
        };
        match s.get(&r) {
            Ok(back) => {
                if back != data {
                    crate::kprintln!("  a {} B blob did not survive the round trip", n);
                    ok = false;
                    return;
                }
            }
            Err(_) => {
                ok = false;
                return;
            }
        }
        // Now the middle, by block, without reading the whole thing.
        let first = 17u64;
        let blocks = 40u32;
        let Some(buf) = crate::dev::nvme::alloc_dma(blocks as usize * bs) else {
            return;
        };
        let buf = unsafe { core::slice::from_raw_parts_mut(buf, blocks as usize * bs) };
        if s.read_blocks(&r, first, blocks, buf).is_err() {
            crate::kprintln!("  ranged read refused");
            ok = false;
            return;
        }
        let lo = first as usize * bs;
        if buf[..blocks as usize * bs] != data[lo..lo + blocks as usize * bs] {
            crate::kprintln!("  a ranged read disagreed with the whole-blob read");
            ok = false;
        }
    });
    ok
}

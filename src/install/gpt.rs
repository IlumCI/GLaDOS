//! A GUID partition table, written.
//!
//! `store::block` has always read one; nothing in this tree had ever written
//! one, because until the installer nothing was allowed to touch LBA 0. This is
//! the encoder and nothing else: it turns a list of partitions into the exact
//! bytes of the primary and backup tables, and never sees a device. What gets
//! written, and whether it may be, is `install::plan`'s and the write gate's.
//!
//! **Checked against the kernel's own reader, not a reader of its own.** The
//! claims encode a table and hand the entry array to
//! `block::parse_gpt_entries`, the function `block::scan` runs at boot. A
//! writer verified only against a parser written beside it is two copies of
//! one misunderstanding, and the first reader to disagree would be the
//! firmware, at the next power-on, with nothing printed.
//!
//! Three things are silent when wrong, and each has a claim:
//!
//! - **The last LBA of an entry is inclusive.** One past it is a partition a
//!   block longer than it says, overlapping its neighbour by exactly that
//!   block, which no tool notices until something writes there.
//! - **The header CRC is over `header_size` bytes with its own field zeroed**,
//!   and the entry-array CRC is over the whole declared array, empty slots
//!   included. Firmware that finds either wrong falls back to the backup, and
//!   if that is wrong too the disk is unpartitioned.
//! - **The backup is not a copy.** Its `my_lba` and `alternate_lba` are
//!   swapped and its entry array sits *before* it at the end of the disk, so
//!   its `entry_lba` differs, so its CRC differs.

use alloc::vec;
use alloc::vec::Vec;

/// Slots in the entry array. 128 is what every tool writes and what firmware
/// expects; fewer is legal and some implementations refuse it.
pub const ENTRIES: usize = 128;
pub const ENTRY_SIZE: usize = 128;
const ARRAY_BYTES: usize = ENTRIES * ENTRY_SIZE;
const HEADER_SIZE: usize = 92;

/// The standard reflected CRC-32 (IEEE 802.3), which is GPT's.
///
/// Bitwise rather than tabled: it runs over sixteen kilobytes twice per
/// install, and a table is 1 KiB of constants to get wrong for no measurable
/// gain.
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
        }
    }
    !c
}

/// One partition to write.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry {
    pub type_guid: [u8; 16],
    pub unique_guid: [u8; 16],
    pub first_lba: u64,
    /// Blocks, not an end LBA, so the inclusive-last rule lives in one place.
    pub blocks: u64,
    pub name: &'static str,
}

/// The geometry every table on a disk shares.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub block_size: u32,
    pub blocks: u64,
}

impl Geometry {
    /// Blocks the entry array takes.
    pub fn array_blocks(&self) -> u64 {
        (ARRAY_BYTES as u64).div_ceil(self.block_size as u64)
    }
    /// The first LBA a partition may start at: after the protective MBR, the
    /// primary header and the primary array.
    pub fn first_usable(&self) -> u64 {
        2 + self.array_blocks()
    }
    /// The last LBA a partition may end on, inclusive: before the backup array
    /// and the backup header in the disk's last block.
    pub fn last_usable(&self) -> u64 {
        self.blocks - 2 - self.array_blocks()
    }
    pub fn backup_header_lba(&self) -> u64 {
        self.blocks - 1
    }
    pub fn backup_array_lba(&self) -> u64 {
        self.blocks - 1 - self.array_blocks()
    }
}

/// What to write, and where.
pub struct Image {
    /// LBA 0 through the end of the primary array.
    pub primary: Vec<u8>,
    /// The backup array then the backup header, ending at the disk's last LBA.
    pub backup: Vec<u8>,
    pub backup_lba: u64,
}

fn put_u32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}
fn get_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// The entry array, all 128 slots, with unused ones left zero.
///
/// Entries carry their slot, because a kept partition must stay in the slot it
/// had: some tools name partitions by index. Two entries in one slot refuse.
pub fn encode_array(entries: &[(usize, Entry)]) -> Option<Vec<u8>> {
    let mut a = vec![0u8; ARRAY_BYTES];
    for (k, &(i, ref e)) in entries.iter().enumerate() {
        if e.blocks == 0 || i >= ENTRIES || entries[..k].iter().any(|(j, _)| *j == i) {
            return None;
        }
        let s = &mut a[i * ENTRY_SIZE..(i + 1) * ENTRY_SIZE];
        s[0..16].copy_from_slice(&e.type_guid);
        s[16..32].copy_from_slice(&e.unique_guid);
        put_u64(s, 32, e.first_lba);
        put_u64(s, 40, e.first_lba + e.blocks - 1);
        // Attributes at 48 stay zero. Name: UTF-16LE, 36 units at most; the
        // names here are ASCII by construction, and truncation is refusal
        // rather than a quietly shorter label.
        let name = e.name.as_bytes();
        if name.len() > 36 || !name.is_ascii() {
            return None;
        }
        for (j, &c) in name.iter().enumerate() {
            s[56 + j * 2] = c;
        }
    }
    Some(a)
}

fn header(g: &Geometry, disk_guid: &[u8; 16], my: u64, alt: u64, array_lba: u64, array_crc: u32) -> Vec<u8> {
    let mut h = vec![0u8; g.block_size as usize];
    h[0..8].copy_from_slice(b"EFI PART");
    put_u32(&mut h, 8, 0x0001_0000);
    put_u32(&mut h, 12, HEADER_SIZE as u32);
    // 16: header CRC, filled last over the header with this field zero.
    put_u64(&mut h, 24, my);
    put_u64(&mut h, 32, alt);
    put_u64(&mut h, 40, g.first_usable());
    put_u64(&mut h, 48, g.last_usable());
    h[56..72].copy_from_slice(disk_guid);
    put_u64(&mut h, 72, array_lba);
    put_u32(&mut h, 80, ENTRIES as u32);
    put_u32(&mut h, 84, ENTRY_SIZE as u32);
    put_u32(&mut h, 88, array_crc);
    let crc = crc32(&h[..HEADER_SIZE]);
    put_u32(&mut h, 16, crc);
    h
}

/// The protective MBR: one 0xEE entry covering the disk, capped at what 32
/// bits can say, so a tool that reads only MBR sees a disk that is entirely
/// spoken for rather than an empty one it may offer to format.
fn protective_mbr(g: &Geometry) -> Vec<u8> {
    let mut m = vec![0u8; g.block_size as usize];
    let e = &mut m[446..462];
    e[1..4].copy_from_slice(&[0x00, 0x02, 0x00]); // CHS of LBA 1, by convention
    e[4] = 0xEE;
    e[5..8].copy_from_slice(&[0xFF, 0xFF, 0xFF]);
    put_u32(e, 8, 1);
    put_u32(e, 12, (g.blocks - 1).min(0xFFFF_FFFF) as u32);
    m[510] = 0x55;
    m[511] = 0xAA;
    m
}

/// Both tables for `entries` on a disk of geometry `g`.
///
/// Refuses anything that does not fit between the usable bounds or overlaps,
/// so a bad plan cannot become bytes. The planner checks this too; the second
/// check is here because this is the last function before the disk.
pub fn encode(g: &Geometry, disk_guid: &[u8; 16], entries: &[(usize, Entry)]) -> Option<Image> {
    let bs = g.block_size as usize;
    if bs < 512 || !bs.is_power_of_two() || g.blocks <= 2 * (g.array_blocks() + 2) {
        return None;
    }
    for (i, (_, a)) in entries.iter().enumerate() {
        let last = a.first_lba.checked_add(a.blocks)?.checked_sub(1)?;
        if a.first_lba < g.first_usable() || last > g.last_usable() {
            return None;
        }
        for (_, b) in &entries[i + 1..] {
            if a.first_lba < b.first_lba + b.blocks && b.first_lba < a.first_lba + a.blocks {
                return None;
            }
        }
    }
    let array = encode_array(entries)?;
    let array_crc = crc32(&array);
    let mut padded = array.clone();
    padded.resize(g.array_blocks() as usize * bs, 0);

    let mut primary = protective_mbr(g);
    primary.extend_from_slice(&header(g, disk_guid, 1, g.backup_header_lba(), 2, array_crc));
    primary.extend_from_slice(&padded);

    let mut backup = padded;
    backup.extend_from_slice(&header(
        g,
        disk_guid,
        g.backup_header_lba(),
        1,
        g.backup_array_lba(),
        array_crc,
    ));
    Some(Image { primary, backup, backup_lba: g.backup_array_lba() })
}

/// Whether a header block and its array are internally consistent: magic,
/// header CRC, array CRC. Not a parser -- `block::scan` is the parser.
pub fn verify(header_block: &[u8], array: &[u8]) -> bool {
    if header_block.len() < HEADER_SIZE || &header_block[0..8] != b"EFI PART" {
        return false;
    }
    let size = get_u32(header_block, 12) as usize;
    if size < HEADER_SIZE || size > header_block.len() {
        return false;
    }
    let mut h = header_block[..size].to_vec();
    let want = get_u32(&h, 16);
    h[16..20].fill(0);
    let n = get_u32(header_block, 80) as usize * get_u32(header_block, 84) as usize;
    crc32(&h) == want && array.len() >= n && crc32(&array[..n]) == get_u32(header_block, 88)
}

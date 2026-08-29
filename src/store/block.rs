//! Block layer: chunked I/O and partition table parsing.
//!
//! Two jobs. First, turn the NVMe driver's small single-command transfers into
//! reads and writes of arbitrary length. Second -- and this is the one that
//! matters right now -- work out where the existing partitions are.
//!
//! That second job is a safety gate, not a convenience. The only NVMe device
//! in this laptop holds Windows. Before GLaDOS writes a single sector we need
//! to know which ranges are spoken for, so that "somewhere safe" is a fact
//! read off the disk rather than an assumption.

#![allow(dead_code)]

use crate::dev::nvme;
use alloc::vec::Vec;

/// One NVMe command here covers two pages. Larger transfers would need a PRP
/// list, so callers are chunked to this instead.
/// Fallback when the device has not been asked yet. The real figure comes
/// from `Nvme::max_transfer_blocks`, which is MDTS bounded by what one page of
/// PRP entries can describe -- 2 MiB rather than the 8 KiB this used to be.
const BLOCKS_PER_IO: u32 = 16;

fn per_io() -> u32 {
    crate::dev::nvme::with(|c| c.max_transfer_blocks).unwrap_or(BLOCKS_PER_IO).max(1)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Error {
    NoDevice,
    Io(u16),
    TooSmall,
    Unaligned,
}

pub fn block_size() -> u32 {
    nvme::with(|n| n.block_size).unwrap_or(512)
}

pub fn block_count() -> u64 {
    nvme::with(|n| n.block_count).unwrap_or(0)
}

/// Read `blocks` blocks starting at `lba`.
///
/// `buf` should be page-aligned (see `nvme::alloc_dma`). It is not required --
/// PRP2 is computed from the buffer offset -- but a straddling buffer costs an
/// extra page per command for no benefit.
pub fn read(lba: u64, blocks: u32, buf: &mut [u8]) -> Result<(), Error> {
    let bs = block_size() as usize;
    if buf.len() < blocks as usize * bs {
        return Err(Error::TooSmall);
    }
    let mut done = 0u32;
    while done < blocks {
        let n = (blocks - done).min(per_io());
        let off = done as usize * bs;
        let ptr = unsafe { buf.as_mut_ptr().add(off) };
        nvme::with(|c| c.read(lba + done as u64, n as u16, ptr))
            .ok_or(Error::NoDevice)?
            .map_err(Error::Io)?;
        done += n;
    }
    Ok(())
}

/// Write `blocks` blocks. Fails unless NVMe writes have been unlocked.
pub fn write(lba: u64, blocks: u32, buf: &[u8]) -> Result<(), Error> {
    let bs = block_size() as usize;
    if buf.len() < blocks as usize * bs {
        return Err(Error::TooSmall);
    }
    let mut done = 0u32;
    while done < blocks {
        let n = (blocks - done).min(per_io());
        let off = done as usize * bs;
        let ptr = unsafe { buf.as_ptr().add(off) };
        nvme::with(|c| c.write(lba + done as u64, n as u16, ptr))
            .ok_or(Error::NoDevice)?
            .map_err(Error::Io)?;
        done += n;
    }
    Ok(())
}

// --- partition tables ---------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Scheme {
    None,
    Mbr,
    Gpt,
}

#[derive(Clone, Copy)]
pub struct Partition {
    pub index: u32,
    pub start_lba: u64,
    pub block_count: u64,
    /// MBR partition type byte, or 0 for GPT.
    pub mbr_type: u8,
    /// GPT type GUID in on-disk (mixed-endian) form, or zeroes for MBR.
    pub type_guid: [u8; 16],
}

impl Partition {
    pub fn end_lba(&self) -> u64 {
        self.start_lba + self.block_count
    }

    pub fn kind(&self) -> &'static str {
        if self.type_guid != [0u8; 16] {
            return gpt_type_name(&self.type_guid);
        }
        match self.mbr_type {
            0x00 => "empty",
            0x07 => "NTFS/exFAT",
            0x0B | 0x0C => "FAT32",
            0x83 => "Linux",
            0x82 => "Linux swap",
            0xEE => "GPT protective",
            0xEF => "EFI System",
            _ => "unknown",
        }
    }
}

/// GPT type GUIDs are stored mixed-endian: the first three fields are
/// little-endian, the last two are byte order as written. These constants are
/// in on-disk form so comparison is a plain byte match.
const GUID_ESP: [u8; 16] = [
    0x28, 0x73, 0x2a, 0xc1, 0x1f, 0xf8, 0xd2, 0x11, 0xba, 0x4b, 0x00, 0xa0, 0xc9, 0x3e, 0xc9, 0x3b,
];
const GUID_MS_BASIC: [u8; 16] = [
    0xa2, 0xa0, 0xd0, 0xeb, 0xe5, 0xb9, 0x33, 0x44, 0x87, 0xc0, 0x68, 0xb6, 0xb7, 0x26, 0x99, 0xc7,
];
const GUID_MS_RESERVED: [u8; 16] = [
    0x16, 0xe3, 0xc9, 0xe3, 0x5c, 0x0b, 0xb8, 0x4d, 0x81, 0x7d, 0xf9, 0x2d, 0xf0, 0x02, 0x15, 0xae,
];
const GUID_WIN_RECOVERY: [u8; 16] = [
    0xa4, 0xbb, 0x94, 0xde, 0xd1, 0x06, 0x40, 0x4d, 0xa1, 0x6a, 0xbf, 0xd5, 0x01, 0x79, 0xd6, 0xac,
];
const GUID_LINUX_FS: [u8; 16] = [
    0xaf, 0x3d, 0xc6, 0x0f, 0x83, 0x84, 0x72, 0x47, 0x8e, 0x79, 0x3d, 0x69, 0xd8, 0x47, 0x7d, 0xe4,
];

fn gpt_type_name(g: &[u8; 16]) -> &'static str {
    match *g {
        super::cas::GLADOS_TYPE_GUID => "GLaDOS store",
        GUID_ESP => "EFI System",
        GUID_MS_BASIC => "Microsoft basic data",
        GUID_MS_RESERVED => "Microsoft reserved",
        GUID_WIN_RECOVERY => "Windows recovery",
        GUID_LINUX_FS => "Linux filesystem",
        _ => "unrecognised",
    }
}

pub struct Layout {
    pub scheme: Scheme,
    pub partitions: Vec<Partition>,
}

impl Layout {
    /// Highest block used by any partition. Anything past this is unclaimed --
    /// though unclaimed is not the same as safe, since firmware and vendor
    /// tools sometimes use the tail of a disk without declaring a partition.
    pub fn highest_used_lba(&self) -> u64 {
        self.partitions.iter().map(|p| p.end_lba()).max().unwrap_or(0)
    }

    pub fn overlaps(&self, lba: u64, blocks: u64) -> Option<&Partition> {
        self.partitions
            .iter()
            .find(|p| lba < p.end_lba() && (lba + blocks) > p.start_lba)
    }
}

#[inline]
fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

#[inline]
fn rd_u64(b: &[u8], o: usize) -> u64 {
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(v)
}

/// Read and parse whatever partition table the disk carries.
pub fn scan() -> Result<Layout, Error> {
    let bs = block_size() as usize;
    let Some(dma) = nvme::alloc_dma(8192) else {
        return Err(Error::NoDevice);
    };
    let buf = unsafe { core::slice::from_raw_parts_mut(dma, 8192) };

    // LBA 0 and 1 together: the MBR and, if present, the GPT header.
    read(0, 2, buf)?;

    let mbr = &buf[..bs];
    let sig = u16::from_le_bytes([mbr[510], mbr[511]]);
    if sig != 0xAA55 {
        return Ok(Layout { scheme: Scheme::None, partitions: Vec::new() });
    }

    // "EFI PART" in LBA 1 means the MBR is only a protective stub.
    let gpt = &buf[bs..bs * 2];
    if gpt.len() >= 92 && &gpt[0..8] == b"EFI PART" {
        let entry_lba = rd_u64(gpt, 72);
        let entry_count = rd_u32(gpt, 80).min(256);
        let entry_size = rd_u32(gpt, 84).max(128) as usize;

        let mut parts = Vec::new();
        let per_block = bs / entry_size;
        if per_block == 0 {
            return Err(Error::Unaligned);
        }
        let blocks_needed = (entry_count as usize).div_ceil(per_block) as u32;

        let Some(edma) = nvme::alloc_dma(blocks_needed as usize * bs + 4096) else {
            return Err(Error::NoDevice);
        };
        let ebuf =
            unsafe { core::slice::from_raw_parts_mut(edma, blocks_needed as usize * bs) };
        read(entry_lba, blocks_needed, ebuf)?;

        for i in 0..entry_count as usize {
            let e = &ebuf[i * entry_size..i * entry_size + entry_size];
            let mut guid = [0u8; 16];
            guid.copy_from_slice(&e[0..16]);
            if guid == [0u8; 16] {
                continue; // unused slot
            }
            let first = rd_u64(e, 32);
            let last = rd_u64(e, 40);
            if last < first {
                continue;
            }
            parts.push(Partition {
                index: i as u32 + 1,
                start_lba: first,
                // GPT stores the last block inclusive.
                block_count: last - first + 1,
                mbr_type: 0,
                type_guid: guid,
            });
        }
        return Ok(Layout { scheme: Scheme::Gpt, partitions: parts });
    }

    // Plain MBR: four 16-byte entries starting at offset 446.
    let mut parts = Vec::new();
    for i in 0..4 {
        let e = &mbr[446 + i * 16..446 + i * 16 + 16];
        let ty = e[4];
        let start = rd_u32(e, 8) as u64;
        let count = rd_u32(e, 12) as u64;
        if ty == 0 || count == 0 {
            continue;
        }
        parts.push(Partition {
            index: i as u32 + 1,
            start_lba: start,
            block_count: count,
            mbr_type: ty,
            type_guid: [0u8; 16],
        });
    }
    Ok(Layout { scheme: Scheme::Mbr, partitions: parts })
}

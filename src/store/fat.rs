//! Read-only FAT16/FAT32.
//!
//! The firmware can read files, and does -- that is how the model gets loaded.
//! But its filesystem driver dies with `ExitBootServices`, so from the moment
//! the kernel owns the machine it can address every block on the disk and
//! understand none of them. That is the gap between a system that boots and a
//! system you could use to fix a machine.
//!
//! FAT because it is what the ESP is, which is where boot configuration lives
//! and where things go wrong. It is also small enough to implement correctly:
//! a chain of 32-bit numbers, a table of 32-byte records, and one awkward
//! extension for long names.
//!
//! Read-only, deliberately. Writing FAT means allocating clusters, updating
//! two copies of the table, and keeping directory entries consistent across a
//! power cut. A rescue tool that can read a broken disk is useful; one that
//! can half-write it is worse than nothing.

use super::block;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Error {
    Io,
    NotFat,
    Unsupported,
    NotFound,
    TooLarge,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Fat16,
    Fat32,
}

pub struct Volume {
    /// LBA of the partition's first sector, so every offset here is relative.
    base: u64,
    kind: Kind,
    bytes_per_sector: u32,
    sectors_per_cluster: u32,
    reserved: u32,
    fat_start: u64,
    sectors_per_fat: u32,
    data_start: u64,
    total_clusters: u32,
    /// FAT32 keeps the root in a normal cluster chain; FAT16 keeps it in a
    /// fixed area before the data region, which is the one real structural
    /// difference between them.
    root_cluster: u32,
    root_dir_sectors: u32,
    root_dir_start: u64,
    /// How many copies of the table this volume keeps. The writer updates all
    /// of them, because copies that disagree are a volume some drivers repair
    /// and others refuse.
    num_fats: u32,
}

#[inline]
fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

#[inline]
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

impl Volume {
    /// Parse the boot sector of a partition starting at `base`.
    pub fn mount(base: u64) -> Result<Self, Error> {
        let mut buf = vec![0u8; block::block_size() as usize];
        block::read(base, 1, &mut buf).map_err(|_| Error::Io)?;

        // The signature is necessary but nowhere near sufficient -- plenty of
        // non-FAT sectors end in 0x55AA -- so the geometry is checked too, and
        // anything that does not add up is refused rather than guessed at.
        if u16_at(&buf, 510) != 0xAA55 {
            return Err(Error::NotFat);
        }
        let bytes_per_sector = u16_at(&buf, 0x0B) as u32;
        let sectors_per_cluster = buf[0x0D] as u32;
        let reserved = u16_at(&buf, 0x0E) as u32;
        let num_fats = buf[0x10] as u32;
        let root_entries = u16_at(&buf, 0x11) as u32;
        let total_16 = u16_at(&buf, 0x13) as u32;
        let fat_16 = u16_at(&buf, 0x16) as u32;
        let total_32 = u32_at(&buf, 0x20);
        let fat_32 = u32_at(&buf, 0x24);

        if bytes_per_sector == 0
            || !bytes_per_sector.is_power_of_two()
            || sectors_per_cluster == 0
            || !sectors_per_cluster.is_power_of_two()
            || reserved == 0
            || num_fats == 0
        {
            return Err(Error::NotFat);
        }
        if bytes_per_sector != block::block_size() {
            // A 4Kn disk with a 512-byte FAT, or the reverse. Handling it means
            // translating every offset, and getting that subtly wrong on a
            // rescue tool is exactly the wrong place to be clever.
            return Err(Error::Unsupported);
        }

        let sectors_per_fat = if fat_16 != 0 { fat_16 } else { fat_32 };
        let total_sectors = if total_16 != 0 { total_16 } else { total_32 };
        if sectors_per_fat == 0 || total_sectors == 0 {
            return Err(Error::NotFat);
        }

        // Root directory is a fixed area on FAT16 and a cluster chain on FAT32;
        // `root_entries` being zero is what distinguishes them.
        let root_dir_sectors = (root_entries * 32).div_ceil(bytes_per_sector);
        let data_start_rel = reserved + num_fats * sectors_per_fat + root_dir_sectors;
        if data_start_rel as u64 >= total_sectors as u64 {
            return Err(Error::NotFat);
        }
        let total_clusters = (total_sectors - data_start_rel) / sectors_per_cluster;

        // The cluster count is what actually defines the FAT width; the
        // thresholds are from the specification and are not negotiable, which
        // is why a volume is not classified by its label or its sectors-per-fat
        // field.
        let kind = if total_clusters < 4085 {
            return Err(Error::Unsupported); // FAT12: different, rare, not worth it
        } else if total_clusters < 65525 {
            Kind::Fat16
        } else {
            Kind::Fat32
        };

        Ok(Self {
            base,
            kind,
            bytes_per_sector,
            sectors_per_cluster,
            reserved,
            fat_start: base + reserved as u64,
            sectors_per_fat,
            data_start: base + data_start_rel as u64,
            total_clusters,
            root_cluster: if kind == Kind::Fat32 { u32_at(&buf, 0x2C) } else { 0 },
            root_dir_sectors,
            root_dir_start: base + (reserved + num_fats * sectors_per_fat) as u64,
            num_fats,
        })
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn cluster_bytes(&self) -> u32 {
        self.bytes_per_sector * self.sectors_per_cluster
    }

    pub fn total_clusters(&self) -> u32 {
        self.total_clusters
    }

    pub fn bytes_per_sector(&self) -> u32 {
        self.bytes_per_sector
    }

    pub fn sectors_per_cluster(&self) -> u32 {
        self.sectors_per_cluster
    }

    pub fn fat_start(&self) -> u64 {
        self.fat_start
    }

    pub fn sectors_per_fat(&self) -> u32 {
        self.sectors_per_fat
    }

    pub fn num_fats(&self) -> u32 {
        self.num_fats
    }

    /// Where a directory listing starts. FAT32 keeps the root in a cluster
    /// chain like any other directory; FAT16 keeps it in a fixed area, which
    /// the writer does not handle and says so.
    pub fn root_cluster(&self) -> u32 {
        self.root_cluster
    }

    pub fn cluster_lba_of(&self, cluster: u32) -> u64 {
        self.cluster_lba(cluster)
    }

    fn cluster_lba(&self, cluster: u32) -> u64 {
        self.data_start + ((cluster - 2) as u64) * self.sectors_per_cluster as u64
    }

    /// The next cluster in a chain, or `None` at the end.
    fn next_cluster(&self, cluster: u32) -> Result<Option<u32>, Error> {
        let entry_bytes = if self.kind == Kind::Fat32 { 4u64 } else { 2 };
        let off = cluster as u64 * entry_bytes;
        let sector = self.fat_start + off / self.bytes_per_sector as u64;
        if sector >= self.fat_start + self.sectors_per_fat as u64 {
            return Err(Error::Io);
        }
        let mut buf = vec![0u8; self.bytes_per_sector as usize];
        block::read(sector, 1, &mut buf).map_err(|_| Error::Io)?;
        let within = (off % self.bytes_per_sector as u64) as usize;

        let next = match self.kind {
            Kind::Fat32 => u32_at(&buf, within) & 0x0FFF_FFFF,
            Kind::Fat16 => u16_at(&buf, within) as u32,
        };
        let eoc = if self.kind == Kind::Fat32 { 0x0FFF_FFF8 } else { 0xFFF8 };
        // Below 2 means free or reserved; either in the middle of a chain
        // means the filesystem is damaged, and following it would read
        // arbitrary blocks.
        if next >= eoc || next < 2 {
            Ok(None)
        } else {
            Ok(Some(next))
        }
    }

    /// Read a whole cluster chain, up to `limit` bytes.
    fn read_chain(&self, start: u32, limit: usize) -> Result<Vec<u8>, Error> {
        let mut out: Vec<u8> = Vec::new();
        out.try_reserve(limit.min(1 << 20)).map_err(|_| Error::TooLarge)?;

        let mut cluster = start;
        let per = self.cluster_bytes() as usize;
        let mut scratch = vec![0u8; per];
        // A corrupt table can form a loop; without a bound this reads until it
        // runs out of memory rather than reporting the damage.
        let mut guard = 0u32;

        loop {
            if cluster < 2 || cluster - 2 >= self.total_clusters {
                return Err(Error::Io);
            }
            block::read(self.cluster_lba(cluster), self.sectors_per_cluster, &mut scratch)
                .map_err(|_| Error::Io)?;
            let take = per.min(limit.saturating_sub(out.len()));
            out.extend_from_slice(&scratch[..take]);
            if out.len() >= limit {
                break;
            }
            match self.next_cluster(cluster)? {
                Some(n) => cluster = n,
                None => break,
            }
            guard += 1;
            if guard > self.total_clusters + 1 {
                return Err(Error::Io);
            }
        }
        Ok(out)
    }

    /// Every 32-byte record of a directory, chain or fixed area.
    fn read_dir_raw(&self, cluster: u32) -> Result<Vec<u8>, Error> {
        if cluster == 0 && self.kind == Kind::Fat16 {
            let bytes = self.root_dir_sectors as usize * self.bytes_per_sector as usize;
            let mut buf = vec![0u8; bytes];
            block::read(self.root_dir_start, self.root_dir_sectors, &mut buf)
                .map_err(|_| Error::Io)?;
            return Ok(buf);
        }
        let start = if cluster == 0 { self.root_cluster } else { cluster };
        // A directory has no size field, so the chain is followed to its end.
        self.read_chain(start, usize::MAX / 2)
    }
}

#[derive(Clone)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u32,
    pub cluster: u32,
}

const ATTR_DIR: u8 = 0x10;
const ATTR_VOLUME: u8 = 0x08;
const ATTR_LFN: u8 = 0x0F;

/// Decode the UCS-2 fragments a long-name entry carries.
///
/// The thirteen characters are split across three ranges of the record rather
/// than being contiguous, which is the single most error-prone detail in the
/// format.
fn lfn_part(rec: &[u8], out: &mut Vec<u16>) {
    const RANGES: [(usize, usize); 3] = [(1, 5), (14, 6), (28, 2)];
    for (off, count) in RANGES {
        for i in 0..count {
            let c = u16_at(rec, off + i * 2);
            if c == 0x0000 || c == 0xFFFF {
                return;
            }
            out.push(c);
        }
    }
}

fn short_name(rec: &[u8]) -> String {
    let mut s = String::new();
    for &b in &rec[0..8] {
        if b == b' ' {
            break;
        }
        s.push(b as char);
    }
    let ext: String = rec[8..11]
        .iter()
        .take_while(|b| **b != b' ')
        .map(|b| *b as char)
        .collect();
    if !ext.is_empty() {
        s.push('.');
        s.push_str(&ext);
    }
    s
}

impl Volume {
    pub fn list(&self, cluster: u32) -> Result<Vec<DirEntry>, Error> {
        let raw = self.read_dir_raw(cluster)?;
        let mut out = Vec::new();
        let mut lfn: Vec<u16> = Vec::new();

        for rec in raw.chunks_exact(32) {
            match rec[0] {
                0x00 => break,      // no entry here or after
                0xE5 => {
                    lfn.clear();    // deleted; any name fragments were its own
                    continue;
                }
                _ => {}
            }
            let attr = rec[11];
            if attr == ATTR_LFN {
                // Fragments are stored last-first, so each new one belongs in
                // front of what has been gathered so far.
                let mut part = Vec::new();
                lfn_part(rec, &mut part);
                part.extend_from_slice(&lfn);
                lfn = part;
                continue;
            }
            if attr & ATTR_VOLUME != 0 && attr & ATTR_DIR == 0 {
                lfn.clear();
                continue;
            }

            let name = if lfn.is_empty() {
                short_name(rec)
            } else {
                String::from_utf16_lossy(&lfn)
            };
            lfn.clear();

            if name == "." || name == ".." {
                continue;
            }
            let hi = u16_at(rec, 20) as u32;
            let lo = u16_at(rec, 26) as u32;
            out.push(DirEntry {
                name,
                is_dir: attr & ATTR_DIR != 0,
                size: u32_at(rec, 28),
                cluster: (hi << 16) | lo,
            });
        }
        Ok(out)
    }

    /// Resolve a slash-separated path to an entry.
    pub fn find(&self, path: &str) -> Result<DirEntry, Error> {
        let mut cluster = 0u32; // root
        let mut current = DirEntry {
            name: String::from("/"),
            is_dir: true,
            size: 0,
            cluster: 0,
        };
        for part in path.split(['/', '\\']).filter(|p| !p.is_empty()) {
            let entries = self.list(cluster)?;
            let found = entries
                .into_iter()
                // FAT is case-insensitive, and a rescue tool that cannot find
                // EFI/BOOT because the operator typed efi/boot is useless.
                .find(|e| e.name.eq_ignore_ascii_case(part))
                .ok_or(Error::NotFound)?;
            cluster = found.cluster;
            current = found;
        }
        Ok(current)
    }

    pub fn read_file(&self, entry: &DirEntry) -> Result<Vec<u8>, Error> {
        if entry.is_dir {
            return Err(Error::NotFound);
        }
        if entry.size == 0 {
            return Ok(Vec::new());
        }
        self.read_chain(entry.cluster, entry.size as usize)
    }
}

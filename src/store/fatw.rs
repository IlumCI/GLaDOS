//! Writing FAT, so files can leave this machine.
//!
//! The reader next door has been enough to boot from and to fetch a corpus
//! with. It is not enough to be an operating system somebody uses: a person
//! who makes something here and cannot put it on a stick and read it elsewhere
//! has a demonstration rather than a computer.
//!
//! **Every write goes through the same gate as every other write.** NVMe
//! writes are locked until an operator claims a range, and nothing here lifts
//! that: a `put` outside the claimed window is refused by the driver exactly
//! as a stray write would be. That is deliberate and it is the reason this can
//! exist at all. An updater or a file writer built on a global unlock is a
//! whole-disk writer wearing a smaller name.
//!
//! **Short names only, and that is a stated limit rather than an oversight.**
//! Reading honours long names because images this project builds carry them;
//! writing produces 8.3 entries, because generating a long-name chain means
//! generating its checksum, its ordering and its unicode, and getting any of
//! the three wrong produces a directory that one operating system reads and
//! another does not. A name that does not fit is refused with the name it
//! would have needed.
//!
//! What is not here: no truncation of an existing file to a shorter length
//! without rewriting it, no directory creation, no timestamps beyond zero, and
//! no FSInfo free-count maintenance. The last is worth naming: FSInfo is a
//! hint and a stale one is legal, but a reader that trusts it will report the
//! wrong free space until something recomputes it.

use super::block;
use super::fat::{Error, Kind, Volume};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// A cluster value meaning "this is the last one".
const EOC: u32 = 0x0FFF_FFF8;

/// Turn a path's final component into an 8.3 name, or say why it will not fit.
///
/// Answers the eleven packed bytes a directory entry wants: eight of name and
/// three of extension, space padded, upper case.
pub fn short_name(name: &str) -> Result<[u8; 11], String> {
    let (stem, ext) = match name.rsplit_once('.') {
        Some((s, e)) => (s, e),
        None => (name, ""),
    };
    if stem.is_empty() || stem.len() > 8 || ext.len() > 3 {
        return Err(alloc::format!(
            "'{}' needs a short name: at most eight characters, a dot, and three",
            name
        ));
    }
    let mut out = [b' '; 11];
    for (i, c) in stem.chars().enumerate() {
        if !ok_char(c) {
            return Err(alloc::format!("'{}' cannot be written to FAT", c));
        }
        out[i] = c.to_ascii_uppercase() as u8;
    }
    for (i, c) in ext.chars().enumerate() {
        if !ok_char(c) {
            return Err(alloc::format!("'{}' cannot be written to FAT", c));
        }
        out[8 + i] = c.to_ascii_uppercase() as u8;
    }
    Ok(out)
}

fn ok_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || "$%'-_@~`!(){}^#&".contains(c)
}

/// Read, change and write back one FAT entry.
///
/// Every copy of the table is updated. A volume with two FATs whose copies
/// disagree is one that some drivers repair and others refuse, and picking
/// which copy is right is not this code's decision to make.
fn set_entry(v: &Volume, cluster: u32, value: u32, fats: u32) -> Result<(), Error> {
    let entry_bytes = if v.kind() == Kind::Fat32 { 4u64 } else { 2 };
    let off = cluster as u64 * entry_bytes;
    let bps = v.bytes_per_sector() as u64;
    let within = (off % bps) as usize;

    for copy in 0..fats.max(1) {
        let sector = v.fat_start() + copy as u64 * v.sectors_per_fat() as u64 + off / bps;
        let mut buf = vec![0u8; bps as usize];
        block::read(sector, 1, &mut buf).map_err(|_| Error::Io)?;
        match v.kind() {
            Kind::Fat32 => {
                // The top four bits are reserved and belong to whoever set
                // them. Preserving them is not politeness: some drivers use
                // them and clearing them is a change nobody asked for.
                let old = u32::from_le_bytes([
                    buf[within],
                    buf[within + 1],
                    buf[within + 2],
                    buf[within + 3],
                ]);
                let merged = (old & 0xF000_0000) | (value & 0x0FFF_FFFF);
                buf[within..within + 4].copy_from_slice(&merged.to_le_bytes());
            }
            _ => {
                let v16 = (value & 0xFFFF) as u16;
                buf[within..within + 2].copy_from_slice(&v16.to_le_bytes());
            }
        }
        block::write(sector, 1, &buf).map_err(|_| Error::Io)?;
    }
    Ok(())
}

fn entry_of(v: &Volume, cluster: u32) -> Result<u32, Error> {
    let entry_bytes = if v.kind() == Kind::Fat32 { 4u64 } else { 2 };
    let off = cluster as u64 * entry_bytes;
    let bps = v.bytes_per_sector() as u64;
    let sector = v.fat_start() + off / bps;
    let within = (off % bps) as usize;
    let mut buf = vec![0u8; bps as usize];
    block::read(sector, 1, &mut buf).map_err(|_| Error::Io)?;
    Ok(match v.kind() {
        Kind::Fat32 => {
            u32::from_le_bytes([buf[within], buf[within + 1], buf[within + 2], buf[within + 3]])
                & 0x0FFF_FFFF
        }
        _ => u16::from_le_bytes([buf[within], buf[within + 1]]) as u32,
    })
}

/// Find `count` free clusters and chain them together.
///
/// The whole chain is built before anything is written into it, so a volume
/// that runs out halfway is left exactly as it was rather than holding a
/// half-allocated chain nothing points at.
fn allocate(v: &Volume, count: u32, fats: u32) -> Result<Vec<u32>, Error> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut found: Vec<u32> = Vec::new();
    // Cluster 0 and 1 are reserved by the format and are not storage.
    for c in 2..v.total_clusters() + 2 {
        if entry_of(v, c)? == 0 {
            found.push(c);
            if found.len() as u32 == count {
                break;
            }
        }
    }
    if (found.len() as u32) < count {
        return Err(Error::Io);
    }
    for i in 0..found.len() {
        let next = if i + 1 == found.len() { EOC } else { found[i + 1] };
        set_entry(v, found[i], next, fats)?;
    }
    Ok(found)
}

/// Release a chain back to the volume.
fn release(v: &Volume, start: u32, fats: u32) -> Result<(), Error> {
    let mut c = start;
    // Bounded by the cluster count, because a corrupt chain that loops would
    // otherwise free the volume forever.
    let mut guard = v.total_clusters() + 2;
    while c >= 2 && c < 0x0FFF_FFF8 && guard > 0 {
        let next = entry_of(v, c)?;
        set_entry(v, c, 0, fats)?;
        c = next;
        guard -= 1;
    }
    Ok(())
}

/// Write `data` to `path`, replacing whatever was there.
///
/// The path is a directory chain the reader already understands, and only the
/// final component is created. Directories are not made on the way, because a
/// half-made path after a failure is worse than a refusal.
pub fn put(v: &Volume, path: &str, data: &[u8]) -> Result<(), String> {
    let (dir_path, name) = match path.rsplit_once('/') {
        Some((d, n)) => (d, n),
        None => ("", path),
    };
    let short = short_name(name)?;

    // Where the directory's entries live.
    let dir_cluster = if dir_path.is_empty() || dir_path == "/" {
        v.root_cluster()
    } else {
        match v.find(dir_path) {
            Ok(e) => e.cluster,
            Err(_) => return Err(alloc::format!("no directory '{}'", dir_path)),
        }
    };

    let fats = v.num_fats();
    let per = v.cluster_bytes() as usize;
    let need = data.len().div_ceil(per.max(1)) as u32;

    // An existing entry has its old chain released first, so a rewrite does
    // not leak the space the previous contents held.
    let existing = find_entry(v, dir_cluster, &short).map_err(|_| String::from("read failed"))?;
    if let Some((sector, off, old_cluster)) = existing {
        if old_cluster >= 2 {
            release(v, old_cluster, fats).map_err(|_| String::from("could not free the old chain"))?;
        }
        let chain = allocate(v, need, fats).map_err(|_| String::from("no room on the volume"))?;
        write_chain(v, &chain, data).map_err(|_| String::from("write failed"))?;
        update_entry(v, sector, off, chain.first().copied().unwrap_or(0), data.len() as u32)
            .map_err(|_| String::from("could not update the directory"))?;
        return Ok(());
    }

    let chain = allocate(v, need, fats).map_err(|_| String::from("no room on the volume"))?;
    write_chain(v, &chain, data).map_err(|_| String::from("write failed"))?;
    add_entry(v, dir_cluster, &short, chain.first().copied().unwrap_or(0), data.len() as u32)
        .map_err(|_| String::from("the directory is full"))?;
    Ok(())
}

/// Copy the payload into an already-linked chain.
fn write_chain(v: &Volume, chain: &[u32], data: &[u8]) -> Result<(), Error> {
    let per = v.cluster_bytes() as usize;
    let spc = v.sectors_per_cluster();
    for (i, c) in chain.iter().enumerate() {
        let start = i * per;
        let end = (start + per).min(data.len());
        let mut buf = vec![0u8; per];
        buf[..end - start].copy_from_slice(&data[start..end]);
        let lba = v.cluster_lba_of(*c);
        block::write(lba, spc, &buf).map_err(|_| Error::Io)?;
    }
    Ok(())
}

/// Walk a directory's entries looking for a short name. Answers where the
/// entry sits so it can be rewritten in place.
fn find_entry(
    v: &Volume,
    dir_cluster: u32,
    short: &[u8; 11],
) -> Result<Option<(u64, usize, u32)>, Error> {
    each_entry(v, dir_cluster, |sector, off, raw| {
        if raw[0] == 0x00 {
            return Some(None);
        }
        if raw[0] == 0xE5 || raw[11] == 0x0F {
            return None;
        }
        if &raw[..11] == &short[..] {
            let hi = u16::from_le_bytes([raw[20], raw[21]]) as u32;
            let lo = u16::from_le_bytes([raw[26], raw[27]]) as u32;
            return Some(Some((sector, off, (hi << 16) | lo)));
        }
        None
    })
}

/// Call `f` for every 32-byte slot in a directory until it answers.
fn each_entry<T>(
    v: &Volume,
    dir_cluster: u32,
    mut f: impl FnMut(u64, usize, &[u8]) -> Option<T>,
) -> Result<T, Error>
where
    T: Default,
{
    let bps = v.bytes_per_sector() as usize;
    let spc = v.sectors_per_cluster();
    let mut cluster = dir_cluster;
    let mut guard = v.total_clusters() + 2;
    loop {
        let base = v.cluster_lba_of(cluster);
        for s in 0..spc as u64 {
            let mut buf = vec![0u8; bps];
            block::read(base + s, 1, &mut buf).map_err(|_| Error::Io)?;
            for off in (0..bps).step_by(32) {
                if let Some(t) = f(base + s, off, &buf[off..off + 32]) {
                    return Ok(t);
                }
            }
        }
        match entry_of(v, cluster)? {
            n if n >= 2 && n < 0x0FFF_FFF8 && guard > 0 => {
                cluster = n;
                guard -= 1;
            }
            _ => return Ok(T::default()),
        }
    }
}

fn update_entry(v: &Volume, sector: u64, off: usize, cluster: u32, size: u32) -> Result<(), Error> {
    let bps = v.bytes_per_sector() as usize;
    let mut buf = vec![0u8; bps];
    block::read(sector, 1, &mut buf).map_err(|_| Error::Io)?;
    buf[off + 20..off + 22].copy_from_slice(&((cluster >> 16) as u16).to_le_bytes());
    buf[off + 26..off + 28].copy_from_slice(&(cluster as u16).to_le_bytes());
    buf[off + 28..off + 32].copy_from_slice(&size.to_le_bytes());
    block::write(sector, 1, &buf).map_err(|_| Error::Io)
}

fn add_entry(
    v: &Volume,
    dir_cluster: u32,
    short: &[u8; 11],
    cluster: u32,
    size: u32,
) -> Result<(), Error> {
    // The first slot that is free or deleted. A directory that is full answers
    // an error rather than growing, because growing one means allocating a
    // cluster and linking it, and a half-grown directory is the worst thing to
    // leave behind.
    let spot: Option<(u64, usize)> = each_entry(v, dir_cluster, |sector, off, raw| {
        if raw[0] == 0x00 || raw[0] == 0xE5 {
            Some(Some((sector, off)))
        } else {
            None
        }
    })?;
    let Some((sector, off)) = spot else { return Err(Error::Io) };

    let bps = v.bytes_per_sector() as usize;
    let mut buf = vec![0u8; bps];
    block::read(sector, 1, &mut buf).map_err(|_| Error::Io)?;
    let e = &mut buf[off..off + 32];
    e.fill(0);
    e[..11].copy_from_slice(&short[..]);
    // An ordinary file. Not read-only, not hidden, not a directory.
    e[11] = 0x20;
    e[20..22].copy_from_slice(&((cluster >> 16) as u16).to_le_bytes());
    e[26..28].copy_from_slice(&(cluster as u16).to_le_bytes());
    e[28..32].copy_from_slice(&size.to_le_bytes());
    block::write(sector, 1, &buf).map_err(|_| Error::Io)
}

/// Remove a file, freeing what it held.
pub fn remove(v: &Volume, path: &str) -> Result<(), String> {
    let (dir_path, name) = match path.rsplit_once('/') {
        Some((d, n)) => (d, n),
        None => ("", path),
    };
    let short = short_name(name)?;
    let dir_cluster = if dir_path.is_empty() || dir_path == "/" {
        v.root_cluster()
    } else {
        match v.find(dir_path) {
            Ok(e) => e.cluster,
            Err(_) => return Err(alloc::format!("no directory '{}'", dir_path)),
        }
    };
    let found = find_entry(v, dir_cluster, &short).map_err(|_| String::from("read failed"))?;
    let Some((sector, off, cluster)) = found else {
        return Err(alloc::format!("no file '{}'", name));
    };
    if cluster >= 2 {
        release(v, cluster, v.num_fats()).map_err(|_| String::from("could not free the chain"))?;
    }
    let bps = v.bytes_per_sector() as usize;
    let mut buf = vec![0u8; bps];
    block::read(sector, 1, &mut buf).map_err(|_| String::from("read failed"))?;
    // The convention is to mark the slot rather than blank it, so an
    // undelete tool still has the name and the size.
    buf[off] = 0xE5;
    block::write(sector, 1, &buf).map_err(|_| String::from("write failed"))?;
    Ok(())
}

//! A FAT32 volume, made from scratch with its files already in it.
//!
//! The installer's ESP. `fatw` edits an existing volume one cluster at a time
//! and cannot make a directory; this lays out a whole volume in one pass, the
//! way `tools/mkiso.py::build_fat` does for the ISO, and is a port of it:
//! same reserved area, same short-name and long-name rules, same rule that
//! every chain is contiguous because nothing is ever deleted from a volume
//! built in one go. Two builders that disagree about a FAT would be two
//! chances for firmware to refuse one of them, so where `build_fat` made a
//! choice this makes the same one.
//!
//! What differs: the volume fills the partition it is given rather than being
//! padded to its contents, so files written later (a staged update, a second
//! model) have room; and the cluster size is the largest that still leaves
//! FAT32's minimum of 65,525 clusters, because below that some firmware reads
//! the volume as FAT16 and misparses all of it.
//!
//! Split in two on purpose. `layout` is pure -- every cluster, every directory
//! entry and the allocation table are decided from sizes alone -- so its
//! claims run at boot with no disk. `write` only copies bytes to places
//! `layout` already chose.

use super::block;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

const SECTOR: usize = 512;
const RESERVED: u32 = 32;
/// FAT32 is *defined* by cluster count; this is the floor.
pub const MIN_CLUSTERS: u64 = 65_525;
const MAX_CLUSTERS: u64 = 0x0FFF_FFF5;

/// One file to put on the volume, by `/`-separated path.
pub struct File<'a> {
    pub path: &'a str,
    pub data: &'a [u8],
}

struct Dir {
    name: Option<String>,
    parent: usize,
    dirs: Vec<usize>,
    files: Vec<usize>,
    cluster: u32,
    clusters: u32,
}

/// Where everything goes.
pub struct Layout {
    pub total_sectors: u32,
    pub spc: u32,
    pub fat_sectors: u32,
    pub clusters: u32,
    /// Directory contents, already serialised, with their first cluster.
    pub dirs: Vec<(u32, Vec<u8>)>,
    /// `(first cluster, file index)`; zero-length files have cluster 0.
    pub files: Vec<(u32, usize)>,
    /// The allocation table, one copy.
    pub fat: Vec<u8>,
}

impl Layout {
    /// Sector of cluster `c`, relative to the volume.
    pub fn sector_of(&self, c: u32) -> u64 {
        RESERVED as u64 + 2 * self.fat_sectors as u64 + (c as u64 - 2) * self.spc as u64
    }
    pub fn cluster_bytes(&self) -> usize {
        self.spc as usize * SECTOR
    }
}

/// The 8.3 name for `name`, with a `~N` tail when the real name will not fit.
/// `taken` is this directory's short names so far.
pub fn short_name(name: &str, taken: &mut Vec<[u8; 11]>) -> Option<[u8; 11]> {
    let (stem, ext) = match name.rfind('.') {
        Some(i) if i > 0 => (&name[..i], &name[i + 1..]),
        _ => (name, ""),
    };
    let clean = |s: &str, ok: fn(char) -> bool| -> String {
        s.chars().map(|c| c.to_ascii_uppercase()).filter(|&c| ok(c)).collect()
    };
    let stem = clean(stem, |c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    let mut ext = clean(ext, |c| c.is_ascii_alphanumeric());
    ext.truncate(3);
    let make = |s: &str| {
        let mut out = [b' '; 11];
        for (i, b) in s.bytes().take(8).enumerate() {
            out[i] = b;
        }
        for (i, b) in ext.bytes().enumerate() {
            out[8 + i] = b;
        }
        out
    };
    if stem.len() <= 8 {
        let c = make(&stem);
        if !taken.contains(&c) {
            taken.push(c);
            return Some(c);
        }
    }
    for n in 1..1000 {
        let tail = alloc::format!("~{}", n);
        let keep = 8 - tail.len();
        let cand = alloc::format!("{}{}", &stem[..stem.len().min(keep)], tail);
        let c = make(&cand);
        if !taken.contains(&c) {
            taken.push(c);
            return Some(c);
        }
    }
    None
}

/// The checksum tying long-name entries to their short entry. Specified as
/// exactly this rotate-and-add.
pub fn lfn_checksum(short: &[u8; 11]) -> u8 {
    let mut s = 0u8;
    for &c in short {
        s = (((s & 1) << 7) | (s >> 1)).wrapping_add(c);
    }
    s
}

fn raw(short: &[u8; 11], attr: u8, cluster: u32, size: u32) -> [u8; 32] {
    let mut e = [0u8; 32];
    e[..11].copy_from_slice(short);
    e[11] = attr;
    e[20..22].copy_from_slice(&((cluster >> 16) as u16).to_le_bytes());
    e[26..28].copy_from_slice(&(cluster as u16).to_le_bytes());
    e[28..32].copy_from_slice(&size.to_le_bytes());
    e
}

/// Whether `name` is exactly what its short entry spells, so no long name is
/// needed. Case matters: `model.bin` is not `MODEL.BIN` to a reader that
/// honours long names, and the payload is looked up in lower case.
fn spelled_by(name: &str, short: &[u8; 11]) -> bool {
    let stem = core::str::from_utf8(&short[..8]).unwrap_or("").trim_end();
    let ext = core::str::from_utf8(&short[8..]).unwrap_or("").trim_end();
    let s = if ext.is_empty() { String::from(stem) } else { alloc::format!("{}.{}", stem, ext) };
    s == name
}

fn entries_for(name: &str, short: &[u8; 11], attr: u8, cluster: u32, size: u32, out: &mut Vec<u8>) {
    if !spelled_by(name, short) {
        let mut chars: Vec<u16> = name.encode_utf16().collect();
        chars.push(0);
        // 0xFFFF is the specified filler; zero would read as a terminator.
        while chars.len() % 13 != 0 {
            chars.push(0xFFFF);
        }
        let total = chars.len() / 13;
        let sum = lfn_checksum(short);
        for i in (1..=total).rev() {
            let part = &chars[(i - 1) * 13..i * 13];
            let mut e = [0u8; 32];
            e[0] = i as u8 | if i == total { 0x40 } else { 0 };
            for (j, c) in part[..5].iter().enumerate() {
                e[1 + 2 * j..3 + 2 * j].copy_from_slice(&c.to_le_bytes());
            }
            e[11] = 0x0F;
            e[13] = sum;
            for (j, c) in part[5..11].iter().enumerate() {
                e[14 + 2 * j..16 + 2 * j].copy_from_slice(&c.to_le_bytes());
            }
            for (j, c) in part[11..13].iter().enumerate() {
                e[28 + 2 * j..30 + 2 * j].copy_from_slice(&c.to_le_bytes());
            }
            out.extend_from_slice(&e);
        }
    }
    out.extend_from_slice(&raw(short, attr, cluster, size));
}

/// Decide everything for a volume of `total_sectors` holding `files`.
pub fn layout(total_sectors: u64, files: &[File]) -> Result<Layout, &'static str> {
    if total_sectors > u32::MAX as u64 {
        return Err("a FAT32 volume is at most 2 TiB");
    }
    // The tree, from the paths.
    let mut dirs = vec![Dir { name: None, parent: 0, dirs: Vec::new(), files: Vec::new(), cluster: 0, clusters: 0 }];
    for (fi, f) in files.iter().enumerate() {
        if f.data.len() > u32::MAX as usize {
            return Err("a FAT32 file is at most 4 GiB");
        }
        let parts: Vec<&str> = f.path.split('/').filter(|s| !s.is_empty()).collect();
        let (leaf, within) = parts.split_last().ok_or("an empty path")?;
        let mut at = 0usize;
        for p in within {
            at = match dirs[at].dirs.iter().copied().find(|&d| dirs[d].name.as_deref() == Some(*p)) {
                Some(d) => d,
                None => {
                    let d = dirs.len();
                    dirs.push(Dir { name: Some(String::from(*p)), parent: at, dirs: Vec::new(), files: Vec::new(), cluster: 0, clusters: 0 });
                    dirs[at].dirs.push(d);
                    d
                }
            };
        }
        if dirs[at].files.iter().any(|&o| files[o].path == f.path) {
            return Err("the same path twice");
        }
        let _ = leaf;
        dirs[at].files.push(fi);
    }

    // Cluster size: the largest that leaves FAT32 its minimum cluster count.
    let mut chosen = None;
    for spc in [8u32, 4, 2, 1] {
        let mut fat_sectors = 1u64;
        loop {
            let data = total_sectors.saturating_sub(RESERVED as u64 + 2 * fat_sectors);
            let clusters = data / spc as u64;
            let need = ((clusters + 2) * 4).div_ceil(SECTOR as u64);
            if need <= fat_sectors {
                if (MIN_CLUSTERS..=MAX_CLUSTERS).contains(&clusters) {
                    chosen = Some((spc, fat_sectors as u32, clusters as u32));
                }
                break;
            }
            fat_sectors = need;
        }
        if chosen.is_some() {
            break;
        }
    }
    let (spc, fat_sectors, clusters) = chosen.ok_or("too small for a FAT32 volume")?;
    let cb = spc as usize * SECTOR;

    // Directory sizes do not depend on cluster numbers, so count entries with
    // placeholders first.
    let size_of = |d: &Dir| -> usize {
        let mut taken = Vec::new();
        let mut buf = Vec::new();
        let mut n = 2; // '.'/'..', or the volume label and slack in the root
        for &c in &d.dirs {
            let name = dirs[c].name.as_deref().unwrap_or("");
            let s = short_name(name, &mut taken).unwrap_or([b' '; 11]);
            entries_for(name, &s, 0x10, 0, 0, &mut buf);
        }
        for &f in &d.files {
            let name = files[f].path.rsplit('/').next().unwrap_or("");
            let s = short_name(name, &mut taken).unwrap_or([b' '; 11]);
            entries_for(name, &s, 0x20, 0, 0, &mut buf);
        }
        n += buf.len() / 32;
        (n * 32).div_ceil(cb).max(1)
    };
    let sizes: Vec<u32> = dirs.iter().map(|d| size_of(d) as u32).collect();
    let mut next = 2u32;
    for (d, n) in dirs.iter_mut().zip(sizes) {
        d.cluster = next;
        d.clusters = n;
        next += n;
    }
    let mut placed = Vec::new();
    for d in &dirs {
        for &f in &d.files {
            let n = files[f].data.len().div_ceil(cb) as u32;
            placed.push((if n == 0 { 0 } else { next }, f));
            next += n;
        }
    }
    if (next - 2) as u64 > clusters as u64 {
        return Err("the files do not fit in the volume");
    }
    let cluster_of = |f: usize| placed.iter().find(|(_, i)| *i == f).map(|(c, _)| *c).unwrap_or(0);

    // Serialise every directory.
    let mut built = Vec::new();
    for d in &dirs {
        let mut out = Vec::new();
        let mut taken = Vec::new();
        match d.name {
            Some(_) => {
                out.extend_from_slice(&raw(b".          ", 0x10, d.cluster, 0));
                // '..' naming the root is written as 0, the one place FAT does
                // not use the real cluster number.
                let up = if d.parent == 0 { 0 } else { dirs[d.parent].cluster };
                out.extend_from_slice(&raw(b"..         ", 0x10, up, 0));
            }
            None => out.extend_from_slice(&raw(b"GLADOS     ", 0x08, 0, 0)),
        }
        for &c in &d.dirs {
            let name = dirs[c].name.as_deref().unwrap_or("");
            let s = short_name(name, &mut taken).ok_or("no free short name")?;
            entries_for(name, &s, 0x10, dirs[c].cluster, 0, &mut out);
        }
        for &f in &d.files {
            let name = files[f].path.rsplit('/').next().unwrap_or("");
            let s = short_name(name, &mut taken).ok_or("no free short name")?;
            entries_for(name, &s, 0x20, cluster_of(f), files[f].data.len() as u32, &mut out);
        }
        out.resize(d.clusters as usize * cb, 0);
        built.push((d.cluster, out));
    }

    // The allocation table: every chain contiguous.
    let mut fat = vec![0u8; fat_sectors as usize * SECTOR];
    let mut set = |c: u32, v: u32| fat[c as usize * 4..c as usize * 4 + 4].copy_from_slice(&v.to_le_bytes());
    set(0, 0x0FFF_FFF8);
    set(1, 0x0FFF_FFFF);
    let mut chain = |start: u32, n: u32| {
        for i in 0..n {
            set(start + i, if i + 1 == n { 0x0FFF_FFFF } else { start + i + 1 });
        }
    };
    for d in &dirs {
        chain(d.cluster, d.clusters);
    }
    for &(c, f) in &placed {
        if c != 0 {
            chain(c, files[f].data.len().div_ceil(cb) as u32);
        }
    }

    Ok(Layout {
        total_sectors: total_sectors as u32,
        spc,
        fat_sectors,
        clusters,
        dirs: built,
        files: placed,
        fat,
    })
}

/// The boot sector and FSInfo, which are also written again at sector 6.
pub fn boot_sectors(l: &Layout, hidden: u32) -> [u8; 2 * SECTOR] {
    let mut b = [0u8; 2 * SECTOR];
    let bs = &mut b[..SECTOR];
    bs[0..3].copy_from_slice(&[0xEB, 0x58, 0x90]);
    bs[3..11].copy_from_slice(b"GLADOS  ");
    bs[11..13].copy_from_slice(&(SECTOR as u16).to_le_bytes());
    bs[13] = l.spc as u8;
    bs[14..16].copy_from_slice(&(RESERVED as u16).to_le_bytes());
    bs[16] = 2; // FATs
    bs[21] = 0xF8;
    bs[24..26].copy_from_slice(&63u16.to_le_bytes());
    bs[26..28].copy_from_slice(&255u16.to_le_bytes());
    bs[28..32].copy_from_slice(&hidden.to_le_bytes());
    bs[32..36].copy_from_slice(&l.total_sectors.to_le_bytes());
    bs[36..40].copy_from_slice(&l.fat_sectors.to_le_bytes());
    bs[44..48].copy_from_slice(&2u32.to_le_bytes()); // root cluster
    bs[48..50].copy_from_slice(&1u16.to_le_bytes()); // FSInfo
    bs[50..52].copy_from_slice(&6u16.to_le_bytes()); // backup boot sector
    bs[64] = 0x80;
    bs[66] = 0x29;
    bs[67..71].copy_from_slice(&0x474C_4144u32.to_le_bytes());
    bs[71..82].copy_from_slice(b"GLADOS     ");
    bs[82..90].copy_from_slice(b"FAT32   ");
    bs[510] = 0x55;
    bs[511] = 0xAA;
    let fsi = &mut b[SECTOR..];
    fsi[0..4].copy_from_slice(b"RRaA");
    fsi[484..488].copy_from_slice(b"rrAa");
    // Free count and next-free unknown: firmware ignores them and a tool
    // recounts, where a wrong count is a volume that reports space it lacks.
    fsi[488..496].fill(0xFF);
    fsi[510] = 0x55;
    fsi[511] = 0xAA;
    b
}

/// Write the volume at `start` (an LBA of 512-byte blocks), inside a write
/// window the caller holds. `say` hears progress.
pub fn write(start: u64, l: &Layout, files: &[File], say: &mut dyn FnMut(&str)) -> Result<(), String> {
    if block::block_size() as usize != SECTOR {
        return Err("FAT32 here assumes 512-byte blocks".into());
    }
    let put = |lba: u64, data: &[u8]| crate::install::apply::write_at(lba, data);
    let boot = boot_sectors(l, start.min(u32::MAX as u64) as u32);
    let mut reserved = vec![0u8; RESERVED as usize * SECTOR];
    reserved[..2 * SECTOR].copy_from_slice(&boot);
    reserved[6 * SECTOR..8 * SECTOR].copy_from_slice(&boot);
    put(start, &reserved)?;
    put(start + RESERVED as u64, &l.fat)?;
    put(start + RESERVED as u64 + l.fat_sectors as u64, &l.fat)?;
    for (c, bytes) in &l.dirs {
        put(start + l.sector_of(*c), bytes)?;
    }
    for &(c, f) in &l.files {
        if c != 0 {
            put(start + l.sector_of(c), files[f].data)?;
            say(&alloc::format!("  boot    {} ({} KiB)", files[f].path, files[f].data.len() / 1024));
        }
    }
    Ok(())
}

/// Pure claims about `layout`, run at boot with no disk.
pub fn checks() -> Vec<(String, bool)> {
    let mut c: Vec<(String, bool)> = Vec::new();
    let mut taken = Vec::new();
    c.push(("an 8.3 name that fits is used as it is".into(),
        short_name("roots.der", &mut taken) == Some(*b"ROOTS   DER")));
    c.push(("a long name gets a ~1 tail".into(),
        short_name("tokenizer.bin", &mut taken) == Some(*b"TOKENI~1BIN")));
    c.push(("and a second one that collides gets ~2".into(),
        short_name("tokenizer.old.bin", &mut taken) == Some(*b"TOKENI~2BIN")));
    // The checksum of "FOO     BAR" as Linux's vfat computes it.
    c.push(("the long-name checksum is the specified rotate-and-add".into(),
        lfn_checksum(b"FOO     BAR") == {
            let mut s = 0u8;
            for &ch in b"FOO     BAR" { s = (((s & 1) << 7) | (s >> 1)).wrapping_add(ch); }
            s
        } && lfn_checksum(b"BOOTX64 EFI") != lfn_checksum(b"BOOTX65 EFI")));
    c.push(("a lower-case name needs a long-name entry, an upper-case 8.3 one does not".into(),
        !spelled_by("model.bin", b"MODEL   BIN") && spelled_by("BOOTX64.EFI", b"BOOTX64 EFI")));

    static K: [u8; 9000] = [7; 9000];
    static M: [u8; 70000] = [9; 70000];
    let files = [
        File { path: "EFI/BOOT/BOOTX64.EFI", data: &K },
        File { path: "GLADOS/model.bin", data: &M },
        File { path: "GLADOS/tokenizer.bin", data: &K[..100] },
        File { path: "GLADOS/FW/iwlwifi-so-a0-gf-a0-89.ucode", data: &K[..10] },
        File { path: "GLADOS/empty", data: &[] },
    ];
    // A 4 GiB partition, as the installer makes.
    let total = (4u64 << 30) / 512;
    match layout(total, &files) {
        Ok(l) => {
            c.push(("a 4 GiB volume takes 4 KiB clusters".into(), l.spc == 8));
            c.push(("and has at least FAT32's minimum cluster count".into(), l.clusters as u64 >= MIN_CLUSTERS));
            c.push(("the volume ends inside the partition".into(),
                RESERVED as u64 + 2 * l.fat_sectors as u64 + l.clusters as u64 * l.spc as u64 <= total));
            c.push(("the table is big enough for every cluster".into(),
                l.fat.len() >= (l.clusters as usize + 2) * 4));
            c.push(("the root directory is cluster 2".into(), l.dirs[0].0 == 2));
            let get = |c: u32| u32::from_le_bytes(l.fat[c as usize * 4..c as usize * 4 + 4].try_into().unwrap());
            let (mc, _) = l.files.iter().copied().find(|&(_, f)| f == 1).unwrap();
            let n = 70000u32.div_ceil(4096);
            c.push(("a file's chain is contiguous and ends in end-of-chain".into(),
                (0..n - 1).all(|i| get(mc + i) == mc + i + 1) && get(mc + n - 1) == 0x0FFF_FFFF));
            c.push(("an empty file has no cluster".into(),
                l.files.iter().any(|&(c, f)| f == 4 && c == 0)));
            c.push(("no two items share a cluster".into(), {
                let mut spans: Vec<(u32, u32)> = l.dirs.iter().map(|(c, b)| (*c, (b.len() / 4096) as u32)).collect();
                for &(c, f) in &l.files {
                    if c != 0 { spans.push((c, files[f].data.len().div_ceil(4096) as u32)); }
                }
                spans.sort();
                spans.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0)
            }));
            let root = &l.dirs[0].1;
            c.push(("the root carries the volume label first".into(), &root[..11] == b"GLADOS     " && root[11] == 0x08));
            let b = boot_sectors(&l, 2048);
            c.push(("the boot sector is signed and says FAT32".into(),
                b[510] == 0x55 && b[511] == 0xAA && &b[82..90] == b"FAT32   "));
            c.push(("the boot sector records the partition's offset".into(),
                u32::from_le_bytes(b[28..32].try_into().unwrap()) == 2048));
        }
        Err(e) => c.push((alloc::format!("a 4 GiB volume lays out ({})", e), false)),
    }
    c.push(("a volume too small for FAT32 is refused".into(), layout(30_000, &files).is_err()));
    // 600 copies of a 70 KB file is 42 MB, against a 34 MB volume.
    let names: Vec<String> = (0..600).map(|i| alloc::format!("f{}", i)).collect();
    let many: Vec<File> = names.iter().map(|n| File { path: n, data: &M }).collect();
    c.push(("files that do not fit are refused".into(), layout(70_000, &many).is_err()));
    c.push(("the same path twice is refused".into(),
        layout(total, &[File { path: "a/b", data: &K }, File { path: "a/b", data: &K }]).is_err()));
    c
}

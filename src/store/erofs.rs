//! EROFS, read-only and uncompressed: the root filesystem Linux guests see.
//!
//! **Why this format.** A program written for Linux expects a Linux system
//! under it -- `/usr/share` data, libraries it `dlopen`s, symlinks such as
//! `/lib64 -> usr/lib`, modes and owners -- and a FAT volume can carry none of
//! the last three. EROFS is the Linux kernel's own read-only filesystem: a
//! superblock, fixed-size inodes addressed by number, directories as sorted
//! blocks of entries, and file data in plain blocks. `tools/rootfs.py` builds
//! the image from a distribution's packages with `mkfs.erofs`, which is the
//! reference this reader is checked against.
//!
//! **Uncompressed only, and refused rather than misread otherwise.** The
//! image is written with no compressor, so every file is either one run of
//! blocks or one run plus a tail kept right after its inode. A compressed or
//! chunk-indexed inode is an error naming its layout: decoding one as plain
//! blocks would hand a program bytes that look like a file and are not.
//!
//! **Symlinks are followed here**, inside the image: Arch's `/lib`, `/bin` and
//! `/lib64` are links into `/usr`, so nothing would resolve without it. An
//! absolute target restarts at the image's own root, and forty hops is a loop
//! -- Linux's own limit.

use crate::store::block;
use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

const MAGIC: u32 = 0xE0F5_E1E2;
const SB_OFFSET: u64 = 1024;
const SECTOR: u64 = 512;
const MAX_HOPS: u32 = 40;

/// Data layouts, `i_format >> 1 & 7`.
const FLAT_PLAIN: u8 = 0;
const FLAT_INLINE: u8 = 2;

pub const S_IFMT: u32 = 0o170000;
pub const S_IFDIR: u32 = 0o040000;
pub const S_IFLNK: u32 = 0o120000;
pub const S_IFREG: u32 = 0o100000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Io,
    NotErofs,
    NotFound,
    NotDir,
    Loop,
    /// A layout this reader declines: compressed or chunk-indexed.
    Layout(u8),
}

/// One inode, with where its data lives already worked out.
#[derive(Clone, Copy, Debug)]
pub struct Inode {
    pub nid: u64,
    pub mode: u32,
    pub size: u64,
    pub uid: u32,
    pub gid: u32,
    pub nlink: u32,
    pub mtime: u64,
    pub rdev: u32,
    layout: u8,
    /// First block of the data, for both flat layouts.
    blkaddr: u32,
    /// Byte offset of the inline tail, right after inode and xattrs.
    tail_at: u64,
}

impl Inode {
    pub fn is_dir(&self) -> bool {
        self.mode & S_IFMT == S_IFDIR
    }
    pub fn is_link(&self) -> bool {
        self.mode & S_IFMT == S_IFLNK
    }
}

pub struct Image {
    /// Partition start, in 512-byte sectors.
    base: u64,
    blkbits: u32,
    /// Byte offset of the metadata area, where inode `n` is at `n * 32`.
    meta: u64,
    root: u64,
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    u32_at(b, o) as u64 | (u32_at(b, o + 4) as u64) << 32
}

impl Image {
    /// Mount the image at partition sector `base`, refusing anything that is
    /// not an EROFS superblock.
    pub fn mount(base: u64) -> Result<Image, Error> {
        let mut sb = [0u8; 128];
        Self::read_raw(base, SB_OFFSET, &mut sb)?;
        if u32_at(&sb, 0) != MAGIC {
            return Err(Error::NotErofs);
        }
        let blkbits = sb[12] as u32;
        if !(9..=16).contains(&blkbits) {
            return Err(Error::NotErofs);
        }
        let img = Image { base, blkbits, meta: (u32_at(&sb, 40) as u64) << blkbits, root: u16_at(&sb, 14) as u64 };
        // The root must be a directory, or this is not an image to trust.
        if !img.inode(img.root)?.is_dir() {
            return Err(Error::NotErofs);
        }
        Ok(img)
    }

    pub fn block_size(&self) -> u32 {
        1 << self.blkbits
    }

    pub fn root(&self) -> Result<Inode, Error> {
        self.inode(self.root)
    }

    /// Bytes at a byte offset in the partition, read through whole sectors.
    fn read_raw(base: u64, off: u64, out: &mut [u8]) -> Result<(), Error> {
        if out.is_empty() {
            return Ok(());
        }
        let first = off / SECTOR;
        let last = (off + out.len() as u64 - 1) / SECTOR;
        let n = (last - first + 1) as usize;
        let mut buf = vec![0u8; n * SECTOR as usize];
        // In pieces, so one read never asks for more than a command carries.
        let mut done = 0usize;
        while done < n {
            let take = (n - done).min(256);
            block::read(base + first + done as u64, take as u32, &mut buf[done * SECTOR as usize..(done + take) * SECTOR as usize])
                .map_err(|_| Error::Io)?;
            done += take;
        }
        let skip = (off - first * SECTOR) as usize;
        out.copy_from_slice(&buf[skip..skip + out.len()]);
        Ok(())
    }

    fn read_at(&self, off: u64, out: &mut [u8]) -> Result<(), Error> {
        Self::read_raw(self.base, off, out)
    }

    pub fn inode(&self, nid: u64) -> Result<Inode, Error> {
        let at = self.meta + nid * 32;
        let mut b = [0u8; 64];
        self.read_at(at, &mut b)?;
        let format = u16_at(&b, 0);
        let extended = format & 1 != 0;
        let layout = ((format >> 1) & 7) as u8;
        let icount = u16_at(&b, 2) as u64;
        let xattrs = if icount == 0 { 0 } else { 12 + (icount - 1) * 4 };
        let (size, uid, gid, nlink, mtime, isize) = if extended {
            (u64_at(&b, 8), u32_at(&b, 24), u32_at(&b, 28), u32_at(&b, 44), u64_at(&b, 32), 64)
        } else {
            (u32_at(&b, 8) as u64, u16_at(&b, 24) as u32, u16_at(&b, 26) as u32, u16_at(&b, 6) as u32, 0, 32)
        };
        let raw = u32_at(&b, 16);
        Ok(Inode {
            nid,
            mode: u16_at(&b, 4) as u32,
            size,
            uid,
            gid,
            nlink,
            mtime,
            rdev: raw,
            layout,
            blkaddr: raw,
            tail_at: at + isize + xattrs,
        })
    }

    /// Up to `out.len()` bytes of a file from `off`. Answers how many.
    pub fn read(&self, ino: &Inode, off: u64, out: &mut [u8]) -> Result<usize, Error> {
        if off >= ino.size || out.is_empty() {
            return Ok(0);
        }
        let want = (out.len() as u64).min(ino.size - off) as usize;
        let bs = self.block_size() as u64;
        match ino.layout {
            FLAT_PLAIN => {
                self.read_at(((ino.blkaddr as u64) << self.blkbits) + off, &mut out[..want])?;
            }
            FLAT_INLINE => {
                // Whole blocks first, then the tail kept after the inode.
                let full = (ino.size / bs) * bs;
                let mut done = 0usize;
                if off < full {
                    let n = ((full - off) as usize).min(want);
                    self.read_at(((ino.blkaddr as u64) << self.blkbits) + off, &mut out[..n])?;
                    done = n;
                }
                if done < want {
                    let in_tail = off + done as u64 - full;
                    self.read_at(ino.tail_at + in_tail, &mut out[done..want])?;
                }
            }
            other => return Err(Error::Layout(other)),
        }
        Ok(want)
    }

    pub fn read_all(&self, ino: &Inode) -> Result<Vec<u8>, Error> {
        let mut v = vec![0u8; ino.size as usize];
        let n = self.read(ino, 0, &mut v)?;
        v.truncate(n);
        Ok(v)
    }

    /// A directory's entries: name, inode number, and the file type byte
    /// (1 regular, 2 directory, 7 symlink, as in `linux_dirent64`).
    ///
    /// A directory is a run of blocks, each its own table: entries of twelve
    /// bytes -- nid, name offset, type -- and then the names packed after
    /// them. The first entry's name offset divided by twelve is how many
    /// there are, and a name runs to the next one's offset or to the end of
    /// the block, where the last may be padded with NULs.
    pub fn dir(&self, ino: &Inode) -> Result<Vec<(String, u64, u8)>, Error> {
        if !ino.is_dir() {
            return Err(Error::NotDir);
        }
        let data = self.read_all(ino)?;
        let bs = self.block_size() as usize;
        let mut out = Vec::new();
        for blk in data.chunks(bs) {
            if blk.len() < 12 {
                break;
            }
            let count = u16_at(blk, 8) as usize / 12;
            for i in 0..count {
                let e = i * 12;
                if e + 12 > blk.len() {
                    break;
                }
                let nid = u64_at(blk, e);
                let start = u16_at(blk, e + 8) as usize;
                let end = if i + 1 < count { u16_at(blk, e + 20) as usize } else { blk.len() };
                if start > end || end > blk.len() {
                    break;
                }
                let mut name = &blk[start..end];
                while let [rest @ .., 0] = name {
                    name = rest;
                }
                out.push((String::from_utf8_lossy(name).into_owned(), nid, blk[e + 10]));
            }
        }
        Ok(out)
    }

    fn child(&self, dir: &Inode, name: &str) -> Result<Inode, Error> {
        let entries = self.dir(dir)?;
        let (_, nid, _) = entries.iter().find(|(n, _, _)| n == name).ok_or(Error::NotFound)?;
        self.inode(*nid)
    }

    /// Resolve a path inside the image. `follow` says whether a symlink in
    /// the last component is followed, which is `stat` against `lstat`.
    pub fn lookup(&self, path: &str, follow: bool) -> Result<Inode, Error> {
        let root = self.root()?;
        let mut cur = root;
        let mut todo: VecDeque<String> =
            path.split('/').filter(|p| !p.is_empty() && *p != ".").map(String::from).collect();
        let mut hops = 0;
        while let Some(part) = todo.pop_front() {
            if !cur.is_dir() {
                return Err(Error::NotDir);
            }
            let next = self.child(&cur, &part)?;
            if next.is_link() && (follow || !todo.is_empty()) {
                hops += 1;
                if hops > MAX_HOPS {
                    return Err(Error::Loop);
                }
                let target = self.read_all(&next)?;
                let target = String::from_utf8_lossy(&target).into_owned();
                if target.starts_with('/') {
                    cur = root;
                }
                for p in target.split('/').filter(|p| !p.is_empty() && *p != ".").rev() {
                    todo.push_front(String::from(p));
                }
                continue;
            }
            cur = next;
        }
        Ok(cur)
    }
}

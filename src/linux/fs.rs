//! A POSIX view of a namespace that is not one.
//!
//! `sysbox::tree` says it plainly in its own header: the namespace is "not a
//! filesystem in any sense a POSIX program would recognise". It is a
//! content-addressed Merkle tree where a copy is O(1), a snapshot is a hash,
//! and `rm` detaches a name rather than destroying anything. None of that has
//! an `open`.
//!
//! A Linux program has the opposite set of expectations: a path resolves to an
//! inode, an inode has a size and a mode, a descriptor is a small integer with
//! a cursor in it, and reading advances the cursor. This module is the
//! translation, and it is a *view* rather than a second store. Nothing here
//! owns any bytes; every read goes to the tree and every listing comes from
//! `sysbox::listing`.
//!
//! ### What is deliberately simplified, and what it costs
//!
//! **An open file holds its whole contents.** `read_blob` answers a `Vec`, so
//! the honest options were to keep that or to teach the store ranged reads.
//! Keeping it makes `read` a slice and `lseek` an integer, and it means a
//! guest opening a 600 MB model file would take 600 MB of heap. Files a guest
//! reads today are configuration and text. When that stops being true this is
//! the first thing to change, and it is written down here rather than
//! discovered by an allocation failure.
//!
//! **There are no permissions, owners or times.** Everything reports mode 0644
//! or 0755, uid 0, and a zero timestamp. A program that branches on any of
//! those gets a consistent answer rather than a true one, which is the right
//! trade while the alternative is inventing a field the store does not have.
//!
//! **There are no links, no devices and no `..`.** `resolve` walks names
//! forward. A path containing `..` is refused rather than normalised, because
//! a tree with O(1) copies has no single parent to walk back to.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// `AT_FDCWD`, the descriptor that means "relative to the working directory".
pub const AT_FDCWD: i64 = -100;

/// What a descriptor refers to.
///
/// The three standard ones are not files and never become files, which is why
/// they are variants rather than an entry in the table with a magic path: a
/// guest that `lseek`s on stdout should get `ESPIPE` from the type system
/// rather than from a path comparison.
pub enum Fd {
    Stdin,
    Stdout,
    Stderr,
    File {
        path: String,
        data: Vec<u8>,
        at: usize,
    },
    Dir {
        path: String,
        /// Name, whether it is a directory, and size. Snapshotted at `open`,
        /// because a directory that changed under a half-finished
        /// `getdents64` would hand the guest a shifting list and there is no
        /// cursor the tree could offer instead.
        entries: Vec<(String, bool, usize)>,
        at: usize,
    },
}

impl Fd {
    pub fn is_dir(&self) -> bool {
        matches!(self, Fd::Dir { .. })
    }
}

/// Resolve a guest path against a working directory.
///
/// Answers `None` for anything with a `..` in it. The tree has O(1) copies and
/// no single parent, so walking back up is not a question it can answer, and
/// silently normalising the path would resolve to somewhere the guest did not
/// name.
pub fn resolve(cwd: &str, path: &str) -> Option<String> {
    if path.split('/').any(|c| c == "..") {
        return None;
    }
    let joined = if path.starts_with('/') {
        path.to_string()
    } else {
        let mut s = String::from(cwd);
        if !s.ends_with('/') {
            s.push('/');
        }
        s.push_str(path);
        s
    };
    // Collapse `.` and empty components so `/a//./b` is `/a/b`.
    let mut out = String::from("/");
    for c in joined.split('/') {
        if c.is_empty() || c == "." {
            continue;
        }
        if out.len() > 1 {
            out.push('/');
        }
        out.push_str(c);
    }
    Some(out)
}

/// `S_IFREG | 0644`.
pub const MODE_FILE: u32 = 0o100_644;
/// `S_IFDIR | 0755`.
pub const MODE_DIR: u32 = 0o040_755;
/// `S_IFIFO | 0600`, which is what the standard three are here.
///
/// They were reported as empty regular files, under a comment saying that
/// reporting them as empty regular files is what makes a program believe
/// stdout is seekable. The comment was right and was describing the code
/// beside it. A pipe is the shape that agrees with the rest of this module:
/// `lseek` on one answers `ESPIPE`, `read` on stdin answers zero forever, and
/// libc picks full buffering for it, all of which are true here.
pub const MODE_FIFO: u32 = 0o010_600;

/// What kind of thing a `stat` is describing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    File,
    Dir,
    Fifo,
}

impl Kind {
    pub fn mode(self) -> u32 {
        match self {
            Kind::File => MODE_FILE,
            Kind::Dir => MODE_DIR,
            Kind::Fifo => MODE_FIFO,
        }
    }
}

/// Linux's `struct stat` for x86-64, filled in as far as this store can.
///
/// 144 bytes, and the layout is fixed by the ABI rather than chosen. Writing
/// it a field short is not a smaller answer, it is a different structure, and
/// libc reads past the end of what was written.
pub fn stat_bytes(kind: Kind, size: usize, ino: u64) -> [u8; 144] {
    let mut b = [0u8; 144];
    let put64 = |b: &mut [u8; 144], at: usize, v: u64| {
        b[at..at + 8].copy_from_slice(&v.to_le_bytes());
    };
    let put32 = |b: &mut [u8; 144], at: usize, v: u32| {
        b[at..at + 4].copy_from_slice(&v.to_le_bytes());
    };
    put64(&mut b, 0, 1); // st_dev
    put64(&mut b, 8, ino); // st_ino
    put64(&mut b, 16, 1); // st_nlink
    put32(&mut b, 24, kind.mode());
    put32(&mut b, 28, 0); // st_uid
    put32(&mut b, 32, 0); // st_gid
    put64(&mut b, 48, size as u64); // st_size
    put64(&mut b, 56, 4096); // st_blksize
    // Blocks are 512-byte units and libc's `du`-shaped callers divide by that
    // rather than by st_blksize, so a file of one byte is one block.
    put64(&mut b, 64, size.div_ceil(512) as u64);
    b
}

/// One `linux_dirent64`, appended to `out`. Answers false when it will not fit.
///
/// The record length is padded to eight because the kernel does, and a guest
/// walking the buffer adds `d_reclen` to its cursor. An unpadded record leaves
/// the next one misaligned and the guest reads a name out of the middle of an
/// inode.
pub fn dirent(out: &mut Vec<u8>, room: usize, ino: u64, next: u64, is_dir: bool, name: &str) -> bool {
    let len = (19 + name.len() + 1).next_multiple_of(8);
    if out.len() + len > room {
        return false;
    }
    let start = out.len();
    out.extend_from_slice(&ino.to_le_bytes());
    // `d_off` is the cursor a later `lseek` would restore to reach the *next*
    // entry, and here the cursor is an entry index rather than a byte offset.
    // It was the offset of the next record inside this buffer, which is a
    // number that means nothing outside the one call that produced it -- so
    // `telldir` would hand back a position `seekdir` could not use, and the
    // two would disagree silently.
    out.extend_from_slice(&next.to_le_bytes());
    out.extend_from_slice(&(len as u16).to_le_bytes());
    out.push(if is_dir { 4 } else { 8 }); // DT_DIR / DT_REG
    out.extend_from_slice(name.as_bytes());
    out.push(0);
    while out.len() < start + len {
        out.push(0);
    }
    true
}

/// A stable-ish inode number for a path.
///
/// The tree has no inodes. Programs use the number to tell two paths apart and
/// to spot hard links, so a hash of the path answers both: distinct paths get
/// distinct numbers, and the same path gets the same number twice running.
/// Zero is avoided because some callers treat it as absent.
pub fn ino_of(path: &str) -> u64 {
    let h = crate::store::sha256::hash(path.as_bytes());
    let n = u64::from_le_bytes([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]]);
    n | 1
}

/// What `diag linux` asks of the projection.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();

    out.push((
        "an absolute path ignores the working directory",
        resolve("/ai", "/tmp/x").as_deref() == Some("/tmp/x"),
    ));
    out.push((
        "a relative path joins onto it",
        resolve("/ai", "notes").as_deref() == Some("/ai/notes"),
    ));
    out.push((
        "doubled and dotted separators collapse",
        resolve("/", "/a//./b/").as_deref() == Some("/a/b"),
    ));
    out.push((
        "a path that walks upwards is refused rather than normalised",
        resolve("/ai", "../etc/passwd").is_none() && resolve("/", "a/../b").is_none(),
    ));
    out.push((
        "the root resolves to itself",
        resolve("/", "/").as_deref() == Some("/") && resolve("/", "").as_deref() == Some("/"),
    ));

    // The stat block is an ABI, so its size and the fields a program actually
    // branches on are asserted rather than assumed.
    let f = stat_bytes(Kind::File, 1234, 7);
    let d = stat_bytes(Kind::Dir, 0, 9);
    let s3 = stat_bytes(Kind::Fifo, 0, 1);
    out.push(("a stat block is the 144 bytes the ABI fixes", f.len() == 144));
    let mode = |b: &[u8; 144]| u32::from_le_bytes([b[24], b[25], b[26], b[27]]);
    out.push((
        "st_mode says regular or directory, and st_size is where libc looks",
        mode(&f) == MODE_FILE
            && mode(&d) == MODE_DIR
            && u64::from_le_bytes(f[48..56].try_into().unwrap()) == 1234,
    ));
    out.push((
        "the standard three are pipes, which is what agrees with ESPIPE on them",
        mode(&s3) == MODE_FIFO && mode(&s3) != MODE_FILE,
    ));
    out.push((
        "a one-byte file is one 512-byte block, since that is the unit callers divide by",
        u64::from_le_bytes(stat_bytes(Kind::File, 1, 1)[64..72].try_into().unwrap()) == 1,
    ));

    // Directory entries, and the padding a guest's cursor depends on.
    let mut buf = Vec::new();
    let ok1 = dirent(&mut buf, 4096, 1, 1, true, "ai");
    let first = buf.len();
    let ok2 = dirent(&mut buf, 4096, 2, 2, false, "a-rather-longer-name.txt");
    out.push(("two entries fit and both are eight-byte multiples", ok1 && ok2 && first % 8 == 0 && buf.len() % 8 == 0));
    out.push((
        "the record length in the first entry steps exactly to the second",
        u16::from_le_bytes([buf[16], buf[17]]) as usize == first,
    ));
    out.push((
        "d_off is the cursor that reaches the next entry, not a place in this buffer",
        u64::from_le_bytes(buf[8..16].try_into().unwrap()) == 1
            && u64::from_le_bytes(buf[first + 8..first + 16].try_into().unwrap()) == 2,
    ));
    out.push((
        "the type byte separates a directory from a file",
        buf[18] == 4 && buf[first + 18] == 8,
    ));
    out.push((
        "an entry that will not fit is refused rather than truncated",
        !dirent(&mut Vec::new(), 8, 1, 1, false, "toolong"),
    ));
    out.push((
        "and a room of zero refuses without touching the buffer",
        {
            let mut v = Vec::new();
            !dirent(&mut v, 0, 1, 1, false, "x") && v.is_empty()
        },
    ));

    out.push((
        "an inode number is stable for a path and different between paths",
        ino_of("/a") == ino_of("/a") && ino_of("/a") != ino_of("/b") && ino_of("/a") != 0,
    ));
    out
}

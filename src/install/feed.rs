//! Writing a system image into the spare slot as it downloads.
//!
//! A system image is 0.7 to 2 GB, and the kernel's own HTTPS client holds a
//! whole body in the heap with a 32 KiB window -- not a download for this.
//! So the kernel does not fetch it. It **arms** a write-only device,
//! `/dev/glados/slot`, with the length and digest a signed manifest named,
//! and any guest program that can copy bytes (`curl -o`, `cp`, `dd`) fills
//! it. The kernel is the only party that decides anything:
//!
//! - the device exists only while armed, and only aimed at the slot this boot
//!   is **not** running, through `nvme::Holder::Slot`'s window and no other;
//! - every byte is hashed on the way to the disk, and a byte past the armed
//!   length is refused rather than written;
//! - when the last byte lands the slot is read back, hashed again and
//!   test-mounted, and only then is a slot record written naming it.
//!
//! A guest feeding the wrong bytes therefore gets a slot that is never
//! named, and the machine boots what it booted before. The guest is a pipe,
//! not a judge.

use super::apply::{read_at, write_at};
use crate::dev::nvme::{self, Holder, CONFIRM};
use crate::store::{block, sha256};
use crate::sync::Spin;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Bytes gathered before a write goes to the disk: whole blocks, large.
const BATCH: usize = 4 << 20;

pub struct Armed {
    pub target: u8,
    pub start: u64,
    pub blocks: u64,
    pub len: u64,
    pub digest: [u8; 32],
    pub version: String,
    pub snapshot: String,
    pub with_kernel: bool,
}

struct Feed {
    a: Armed,
    written: u64,
    hash: sha256::Sha256,
    buf: Vec<u8>,
    /// Where the next batch lands, in blocks from the slot's start.
    at: u64,
    failed: Option<String>,
    done: Option<Result<String, String>>,
}

static FEED: Spin<Option<Feed>> = Spin::new(None);

pub fn armed() -> bool {
    FEED.lock().as_ref().is_some_and(|f| f.done.is_none() && f.failed.is_none())
}

/// What the feed is doing, for `update system`.
pub fn status() -> Option<String> {
    let g = FEED.lock();
    let f = g.as_ref()?;
    Some(match (&f.done, &f.failed) {
        (Some(Ok(s)), _) => s.clone(),
        (Some(Err(e)), _) | (None, Some(e)) => format!("slot {}: failed -- {}", slot_name(f.a.target), e),
        (None, None) => format!(
            "slot {} armed for {} {}: {} of {} MiB written -- copy the image to /dev/glados/slot",
            slot_name(f.a.target),
            f.a.version,
            if f.a.with_kernel { "(with its kernel)" } else { "" },
            f.written >> 20,
            f.a.len >> 20
        ),
    })
}

pub fn slot_name(i: u8) -> char {
    (b'A' + i) as char
}

/// The slot a system update may write: whichever this boot is not running.
pub fn spare() -> Result<(u8, block::Partition), String> {
    let running = super::slot::chosen().ok_or("this boot did not come from an installed disk, so there is no spare slot")?;
    let slots = super::slot::slots();
    if slots.len() < 2 {
        return Err("this disk has one system slot, so there is nowhere to put a new system beside the running one".into());
    }
    let t = 1 - running.min(1);
    Ok((t, slots[t as usize]))
}

/// Arm the device for one image. Holds the slot's write window until the feed
/// finishes or is called off.
pub fn arm(len: u64, digest: [u8; 32], version: &str, snapshot: &str, with_kernel: bool) -> Result<String, String> {
    if FEED.lock().as_ref().is_some_and(|f| f.done.is_none() && f.failed.is_none()) {
        return Err("a system image is already being written -- 'update system cancel' first".into());
    }
    let (target, p) = spare()?;
    let bs = block::block_size() as u64;
    if len == 0 || len > p.block_count * bs {
        return Err(format!("{} MiB does not fit slot {} ({} MiB)", len >> 20, slot_name(target), (p.block_count * bs) >> 20));
    }
    if crate::linux::disk::slot_mounted(p.start_lba) {
        return Err(format!("slot {} is mounted -- refusing to write under a running system", slot_name(target)));
    }
    if !nvme::claim(CONFIRM, Holder::Slot, p.start_lba, p.block_count) {
        return Err("the write gate refused the slot".into());
    }
    crate::store::erofs::forget(p.start_lba);
    *FEED.lock() = Some(Feed {
        a: Armed {
            target,
            start: p.start_lba,
            blocks: p.block_count,
            len,
            digest,
            version: version.into(),
            snapshot: snapshot.into(),
            with_kernel,
        },
        written: 0,
        hash: sha256::Sha256::new(),
        buf: Vec::with_capacity(BATCH),
        at: 0,
        failed: None,
        done: None,
    });
    Ok(format!("slot {} armed for {} MiB -- /dev/glados/slot is open for writing", slot_name(target), len >> 20))
}

pub fn cancel() -> bool {
    let had = FEED.lock().take().is_some();
    nvme::release(Holder::Slot);
    had
}

fn flush(f: &mut Feed, last: bool) -> Result<(), String> {
    let bs = block::block_size() as usize;
    let whole = if last { f.buf.len().div_ceil(bs) * bs } else { f.buf.len() / bs * bs };
    if whole == 0 {
        return Ok(());
    }
    f.buf.resize(f.buf.len().max(whole), 0);
    write_at(f.a.start + f.at, &f.buf[..whole])?;
    f.at += (whole / bs) as u64;
    f.buf.drain(..whole);
    Ok(())
}

/// Bytes from a guest's `write`. Answers how many were taken: all of them,
/// or none once the feed is full, failed or not armed.
pub fn write(data: &[u8]) -> usize {
    let mut g = FEED.lock();
    let Some(f) = g.as_mut() else { return 0 };
    if f.done.is_some() || f.failed.is_some() {
        return 0;
    }
    let room = (f.a.len - f.written) as usize;
    if data.len() > room {
        f.failed = Some(format!("{} B past the {} the manifest names -- not this image", data.len() - room, f.a.len));
        nvme::release(Holder::Slot);
        return 0;
    }
    f.hash.update(data);
    f.buf.extend_from_slice(data);
    f.written += data.len() as u64;
    let last = f.written == f.a.len;
    if f.buf.len() >= BATCH || last {
        if let Err(e) = flush(f, last) {
            f.failed = Some(e);
            nvme::release(Holder::Slot);
            return 0;
        }
    }
    if last {
        // Out of the lock for the read-back: a 2 GB slot is seconds of I/O,
        // and `status` from the shell should not wait behind it.
        let mut own = g.take().unwrap();
        drop(g);
        let r = finish(&mut own);
        nvme::release(Holder::Slot);
        match &r {
            Ok(s) => crate::kprintln!("[update] {}", s),
            Err(e) => crate::kprintln!("[update] system image refused: {}", e),
        }
        own.done = Some(r);
        *FEED.lock() = Some(own);
    }
    data.len()
}

/// The last byte has landed: hash it, read it back, mount it, name it.
fn finish(f: &mut Feed) -> Result<String, String> {
    let got = core::mem::replace(&mut f.hash, sha256::Sha256::new()).finish();
    if got != f.a.digest {
        return Err("the bytes written are not the image the manifest names".into());
    }
    let bs = block::block_size() as u64;
    let mut h = sha256::Sha256::new();
    let mut off = 0u64;
    while off < f.a.len {
        let n = (f.a.len - off).min(BATCH as u64) as usize;
        let b = read_at(f.a.start + off / bs, n)?;
        h.update(&b);
        off += n as u64;
        // Seconds of hashing in a guest's syscall: let the desktop draw.
        crate::task::yield_now();
    }
    if h.finish() != f.a.digest {
        return Err("the slot did not read back as the image that was written".into());
    }
    crate::store::erofs::forget(f.a.start);
    crate::store::erofs::Image::mount(f.a.start).map_err(|_| String::from("the written slot does not mount"))?;
    let running = super::slot::chosen().unwrap_or(1 - f.a.target);
    // A kernel staged beside this image is part of the same release: it
    // rolls back with it. Read off the ESP rather than asked for, so a kernel
    // staged before, during or after the download is counted the same way
    // -- and one that is not staged is never restored over.
    let with_kernel = f.a.with_kernel
        || crate::update::stage::find_esp().is_ok_and(|e| e.volume.find("/GLADOS/UPDATE.FLG").is_ok());
    let rec = super::slot::pending(
        super::slot::current().as_ref(),
        running,
        f.a.target,
        f.a.len,
        f.a.digest,
        &f.a.version,
        &f.a.snapshot,
        with_kernel,
    );
    super::slot::write_now(&rec)?;
    Ok(format!(
        "system {} is in slot {}, verified -- restart to try it (the running system stays if it fails)",
        f.a.version,
        slot_name(f.a.target)
    ))
}

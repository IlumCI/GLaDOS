//! Which system slot boots, and how a new one is tried and kept or undone.
//!
//! An installed disk carries two system slots (`plan::Role::SlotA`, `SlotB`).
//! One is running; a system update writes the other whole and then names it
//! in a **slot record**. The record lives on the ESP as two alternating
//! 512-byte files, `GLADOS\SLOT0.REC` and `SLOT1.REC`: a write always goes to
//! the file holding the *older* sequence number, so a write torn by a power
//! cut damages the copy nobody needed and the other one still says where the
//! machine was. Each carries a CRC, and the valid one with the higher
//! sequence wins.
//!
//! The record is read and decided **before ExitBootServices**, beside
//! `update::hook`, for the reason that hook gives: the firmware's FAT driver
//! is the one writer of the ESP that exists then, and an attempt has to be
//! counted before the risky part of the boot, not after it. It is marked kept
//! later, at the shell, through this kernel's own FAT writer -- the window is
//! the whole boot, as `repairs::survived` has it.
//!
//! The decision is pure, in `update::decide`'s shape, so every state is a
//! claim at boot with no disk.
//!
//! **A kernel and an image never run mismatched across a rollback.** A
//! release that moves both stages the kernel through the existing updater and
//! marks the record `with_kernel`; a failed trial of the image restores
//! `BOOTX64.OLD` too, and a kernel the updater rolled back takes the image
//! back with it.

use alloc::string::String;
use alloc::vec::Vec;

pub const MAGIC: &[u8; 8] = b"GLADSLOT";
/// The newest store format this kernel reads and writes: 2 is directory
/// entries carrying attributes (installer P3). A release that writes a
/// later one is refused by `update system` rather than installed under a
/// kernel that would misread its store.
pub const STORE_FORMAT: u32 = 2;
pub const FORMAT: u32 = 1;
pub const LEN: usize = 512;
pub const FILES: [&str; 2] = ["\\GLADOS\\SLOT0.REC", "\\GLADOS\\SLOT1.REC"];
/// The same two, as this kernel's own FAT writer names them.
pub const FAT_FILES: [&str; 2] = ["/GLADOS/SLOT0.REC", "/GLADOS/SLOT1.REC"];

/// Boots a new slot may take before it is undone. One, for `update::MAX_TRIES`'
/// reason: an image that fails early fails every time.
pub const MAX_TRIES: u8 = 1;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Record {
    pub seq: u32,
    /// 0 is slot A, 1 is slot B: their order in the partition table.
    pub active: u8,
    /// Where a failed trial goes back to.
    pub previous: u8,
    /// `None` when kept; the boots taken so far while on trial.
    pub trial: Option<u8>,
    /// The kernel was staged with this image and rolls back with it.
    pub with_kernel: bool,
    pub image_len: u64,
    pub digest: [u8; 32],
    /// The system version and the Arch snapshot it was built against,
    /// NUL-padded.
    pub version: [u8; 32],
    pub snapshot: [u8; 32],
    /// The same three for `previous`, so a rollback describes the system it
    /// went back to rather than the one that failed.
    pub prev_len: u64,
    pub prev_digest: [u8; 32],
    pub prev_version: [u8; 32],
    pub prev_snapshot: [u8; 32],
}

fn fixed(s: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    let b = s.as_bytes();
    let n = b.len().min(32);
    out[..n].copy_from_slice(&b[..n]);
    out
}

fn unfixed(b: &[u8; 32]) -> &str {
    let n = b.iter().position(|&c| c == 0).unwrap_or(32);
    core::str::from_utf8(&b[..n]).unwrap_or("?")
}

impl Record {
    pub fn version(&self) -> &str {
        unfixed(&self.version)
    }
    pub fn snapshot(&self) -> &str {
        unfixed(&self.snapshot)
    }

    pub fn encode(&self) -> [u8; LEN] {
        let mut o = [0u8; LEN];
        o[0..8].copy_from_slice(MAGIC);
        o[8..12].copy_from_slice(&FORMAT.to_le_bytes());
        o[12..16].copy_from_slice(&self.seq.to_le_bytes());
        o[16] = self.active;
        o[17] = self.previous;
        o[18] = self.trial.is_some() as u8;
        o[19] = self.trial.unwrap_or(0);
        o[20] = self.with_kernel as u8;
        o[24..32].copy_from_slice(&self.image_len.to_le_bytes());
        o[32..64].copy_from_slice(&self.digest);
        o[64..96].copy_from_slice(&self.version);
        o[96..128].copy_from_slice(&self.snapshot);
        o[128..136].copy_from_slice(&self.prev_len.to_le_bytes());
        o[136..168].copy_from_slice(&self.prev_digest);
        o[168..200].copy_from_slice(&self.prev_version);
        o[200..232].copy_from_slice(&self.prev_snapshot);
        let crc = super::gpt::crc32(&o[..LEN - 4]);
        o[LEN - 4..].copy_from_slice(&crc.to_le_bytes());
        o
    }

    /// A record, or nothing: wrong size, wrong magic, a later format, a slot
    /// number that is not 0 or 1, or a CRC that does not match.
    pub fn decode(b: &[u8]) -> Option<Record> {
        if b.len() != LEN || &b[0..8] != MAGIC {
            return None;
        }
        if u32::from_le_bytes(b[8..12].try_into().ok()?) != FORMAT {
            return None;
        }
        if super::gpt::crc32(&b[..LEN - 4]) != u32::from_le_bytes(b[LEN - 4..].try_into().ok()?) {
            return None;
        }
        if b[16] > 1 || b[17] > 1 || b[18] > 1 {
            return None;
        }
        Some(Record {
            seq: u32::from_le_bytes(b[12..16].try_into().ok()?),
            active: b[16],
            previous: b[17],
            trial: (b[18] == 1).then_some(b[19]),
            with_kernel: b[20] == 1,
            image_len: u64::from_le_bytes(b[24..32].try_into().ok()?),
            digest: b[32..64].try_into().ok()?,
            version: b[64..96].try_into().ok()?,
            snapshot: b[96..128].try_into().ok()?,
            prev_len: u64::from_le_bytes(b[128..136].try_into().ok()?),
            prev_digest: b[136..168].try_into().ok()?,
            prev_version: b[168..200].try_into().ok()?,
            prev_snapshot: b[200..232].try_into().ok()?,
        })
    }

    /// The file a successor of this record goes to: the one this record is
    /// *not* in, so the copy that describes the running machine survives a
    /// torn write.
    pub fn next_file(&self) -> usize {
        ((self.seq + 1) % 2) as usize
    }
}

/// The record two files agree on: the valid one with the higher sequence.
pub fn newest(a: Option<&[u8]>, b: Option<&[u8]>) -> Option<Record> {
    let ra = a.and_then(Record::decode);
    let rb = b.and_then(Record::decode);
    match (ra, rb) {
        (Some(x), Some(y)) => Some(if y.seq > x.seq { y } else { x }),
        (x, y) => x.or(y),
    }
}

/// What the kernel updater did this boot, as far as a slot cares.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kernel {
    Quiet,
    /// A staged kernel was written this boot and has not run yet.
    AppliedNow,
    /// The kernel on trial did not prove itself and `BOOTX64.OLD` was put
    /// back.
    RolledBack,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Choice {
    /// Which slot to mount.
    pub mount: u8,
    /// The record to write before going on, if the state moved.
    pub write: Option<Record>,
    /// Put `BOOTX64.OLD` back as well.
    pub restore_kernel: bool,
    pub say: Option<&'static str>,
}

fn back(r: &Record, restore_kernel: bool, say: &'static str) -> Choice {
    let w = Record {
        seq: r.seq + 1,
        active: r.previous,
        previous: r.previous,
        trial: None,
        with_kernel: false,
        image_len: r.prev_len,
        digest: r.prev_digest,
        version: r.prev_version,
        snapshot: r.prev_snapshot,
        ..*r
    };
    Choice { mount: r.previous, write: Some(w), restore_kernel, say: Some(say) }
}

/// The whole boot-time decision.
pub fn decide(rec: Option<Record>, kernel: Kernel) -> Choice {
    let Some(r) = rec else {
        // No record: a disk the installer laid out before records existed,
        // or one whose two copies are both unreadable. Slot A is what the
        // installer writes, so it is the one that was ever known good.
        return Choice { mount: 0, write: None, restore_kernel: false, say: None };
    };
    let Some(tries) = r.trial else {
        return Choice { mount: r.active, write: None, restore_kernel: false, say: None };
    };
    // A kernel that was undone takes its image with it, whatever the count:
    // the image was built for the kernel that just failed.
    if kernel == Kernel::RolledBack {
        return back(&r, false, "system update: the new kernel failed, so the previous system image is back too");
    }
    // A kernel was written this boot and the machine restarts into it
    // before anything mounts. Counting a try here would spend the image's
    // only one on a boot that never ran it -- and for a release that moved
    // both, would try the image under the old kernel, the mismatch this
    // module exists to prevent. Wait for the restart.
    if kernel == Kernel::AppliedNow {
        return Choice { mount: r.previous, write: None, restore_kernel: false, say: Some("system update: waiting for the new kernel") };
    }
    if tries >= MAX_TRIES {
        return back(&r, r.with_kernel, "system update: the new system did not reach the desktop -- rolled back");
    }
    let w = Record { seq: r.seq + 1, trial: Some(tries + 1), ..r };
    Choice { mount: r.active, write: Some(w), restore_kernel: false, say: Some("system update: the new system image is on trial") }
}

/// The record that keeps a trial, or nothing to write when there is none.
pub fn settle(r: &Record) -> Option<Record> {
    r.trial.map(|_| Record {
        seq: r.seq + 1,
        trial: None,
        previous: r.active,
        prev_len: r.image_len,
        prev_digest: r.digest,
        prev_version: r.version,
        prev_snapshot: r.snapshot,
        ..*r
    })
}

/// The record naming a freshly written slot, from whatever the machine had.
pub fn pending(cur: Option<&Record>, running: u8, target: u8, len: u64, digest: [u8; 32], version: &str, snapshot: &str, with_kernel: bool) -> Record {
    Record {
        seq: cur.map(|r| r.seq + 1).unwrap_or(1),
        active: target,
        previous: running,
        trial: Some(0),
        with_kernel,
        image_len: len,
        digest,
        version: fixed(version),
        snapshot: fixed(snapshot),
        // What is running now, which is what a failed trial goes back to.
        prev_len: cur.map(|r| r.image_len).unwrap_or(0),
        prev_digest: cur.map(|r| r.digest).unwrap_or([0; 32]),
        prev_version: cur.map(|r| r.version).unwrap_or([0; 32]),
        prev_snapshot: cur.map(|r| r.snapshot).unwrap_or([0; 32]),
    }
}

// ---------------------------------------------------------------------------
// The impure halves.
// ---------------------------------------------------------------------------

use crate::sync::Racy;

/// What this boot decided, for `auto_root`, `settle_now` and the shell.
static NOW: Racy<(Option<Record>, Option<u8>)> = Racy::new((None, None));

static RESTORED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Whether `at_boot` put the previous kernel back, so this boot is running
/// the one that failed and has to restart.
pub fn restored_kernel() -> bool {
    RESTORED.load(core::sync::atomic::Ordering::Relaxed)
}

/// The slot this boot chose, or `None` on media with no record and no choice
/// (live ISO, or before `at_boot` ran).
pub fn chosen() -> Option<u8> {
    unsafe { NOW.get() }.1
}

pub fn current() -> Option<Record> {
    unsafe { NOW.get() }.0
}

/// Before ExitBootServices: read both copies, decide, write what moved.
pub fn at_boot(bs: &crate::uefi::BootServices, image: crate::uefi::Handle, kernel: Kernel) -> Option<&'static str> {
    let a = crate::uefi::read_file(bs, image, FILES[0]);
    let b = crate::uefi::read_file(bs, image, FILES[1]);
    // An installed disk carries INSTALL.ID; anything without one is media,
    // where there is no slot to choose.
    let installed = crate::uefi::read_file(bs, image, "\\GLADOS\\INSTALL.ID").is_some();
    let rec = newest(a.as_ref().map(|x| x.as_slice()), b.as_ref().map(|x| x.as_slice()));
    let c = decide(rec, kernel);
    let mut said = c.say;
    let mut kept = rec;
    if let Some(w) = c.write {
        // Counted before the risky part of the boot, so a system that dies
        // in it has had its try.
        if crate::uefi::write_file(bs, image, FILES[w.next_file_from(rec)], &w.encode()) {
            kept = Some(w);
        } else {
            said = Some("system update: the slot record would not write -- staying on the running slot");
            let mount = rec.map(|r| if r.trial.is_some() { r.previous } else { r.active }).unwrap_or(0);
            unsafe { *NOW.get() = (rec, installed.then_some(mount)) };
            return said;
        }
    }
    if c.restore_kernel {
        if crate::update::restore_old(bs, image) {
            RESTORED.store(true, core::sync::atomic::Ordering::Relaxed);
        } else {
            said = Some("system update: rolled back the image, and the old kernel would not write -- boot the USB");
        }
    }
    unsafe { *NOW.get() = (kept, (installed || rec.is_some()).then_some(c.mount)) };
    said
}

impl Record {
    /// Which file a record written over `prev` goes to.
    fn next_file_from(&self, prev: Option<Record>) -> usize {
        match prev {
            Some(p) => p.next_file(),
            None => (self.seq % 2) as usize,
        }
    }
}

/// Write a record after ExitBootServices, through our own FAT writer, and
/// make it the one this boot reports.
pub fn write_now(w: &Record) -> Result<(), String> {
    let esp = crate::update::stage::find_esp()?;
    let file = w.next_file_from(current());
    crate::update::stage::put_one(&esp, FAT_FILES[file], &w.encode())?;
    let ch = chosen();
    unsafe { *NOW.get() = (Some(*w), ch) };
    Ok(())
}

/// The machine reached the shell on a slot that was on trial: keep it.
pub fn settle_now() -> Option<String> {
    let r = current()?;
    r.trial?;
    // Reaching the shell is not enough if the system it reached is the
    // other one: `auto_root` falls back to the running slot when the new one
    // will not mount, and keeping a trial that never mounted would name a
    // slot that does not boot. Undo it now rather than one boot later.
    let want = slots().get(r.active as usize).map(|p| p.start_lba);
    if want.is_none() || crate::linux::disk::root_base() != want {
        let w = back(&r, false, "").write?;
        return Some(match write_now(&w) {
            Ok(()) => alloc::format!("system update: {} did not mount, so slot {} stays", r.version(), (b'A' + r.previous) as char),
            Err(e) => alloc::format!("system update: {} did not mount, and the record would not write: {}", r.version(), e),
        });
    }
    let w = settle(&r)?;
    Some(match write_now(&w) {
        Ok(()) => alloc::format!("system update: {} reached the desktop and is kept", r.version()),
        Err(e) => alloc::format!("system update: could not mark the new system kept: {}", e),
    })
}

/// The two system slots on the disk, in table order, as `(start, blocks)`.
pub fn slots() -> Vec<crate::store::block::Partition> {
    crate::store::block::scan()
        .map(|l| l.partitions.into_iter().filter(|p| p.type_guid == super::plan::ROOT_TYPE_GUID).collect())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Claims.
// ---------------------------------------------------------------------------

pub fn checks() -> Vec<(String, bool)> {
    let mut c: Vec<(String, bool)> = Vec::new();
    let mut p = |s: &str, ok: bool| c.push((String::from(s), ok));
    let base = pending(None, 0, 1, 1 << 30, [7; 32], "1.4.6", "2026/11/15", false);

    let e = base.encode();
    p("a record round-trips through its encoding", Record::decode(&e) == Some(base));
    let mut torn = e;
    torn[40] ^= 1;
    p("a flipped bit is refused by the CRC", Record::decode(&torn).is_none());
    let mut later = e;
    later[8] = 2;
    p("a later format is refused rather than misread", Record::decode(&later).is_none());
    p("the version and snapshot read back as text", base.version() == "1.4.6" && base.snapshot() == "2026/11/15");

    let older = Record { seq: 4, ..base };
    let newer = Record { seq: 5, active: 0, ..base };
    p("the newer of two good copies wins", newest(Some(&older.encode()), Some(&newer.encode())) == Some(newer));
    p("a torn newer copy leaves the older one in charge", newest(Some(&older.encode()), Some(&torn)) == Some(older));
    p("a successor goes to the other file", older.next_file() != newer.next_file());

    p("no record: slot A, nothing written", decide(None, Kernel::Quiet) == Choice { mount: 0, write: None, restore_kernel: false, say: None });
    let kept = Record { trial: None, ..base };
    p("a kept record mounts its slot and writes nothing", decide(Some(kept), Kernel::Quiet).mount == 1 && decide(Some(kept), Kernel::Quiet).write.is_none());
    p("a kernel-only rollback leaves a kept slot alone", decide(Some(kept), Kernel::RolledBack).write.is_none());

    let t0 = decide(Some(base), Kernel::Quiet);
    p("a fresh trial mounts the new slot and counts the boot", t0.mount == 1 && t0.write.map(|w| w.trial) == Some(Some(1)));
    p("the count goes up the sequence", t0.write.map(|w| w.seq) == Some(base.seq + 1));
    let t1 = Record { trial: Some(1), ..base };
    let r = decide(Some(t1), Kernel::Quiet);
    p("a trial that came back unkept is undone", r.mount == 0 && r.write.map(|w| (w.active, w.trial)) == Some((0, None)));
    let named = pending(Some(&kept), 1, 0, 9, [3; 32], "1.4.7", "s2", false);
    let undone = decide(Some(Record { trial: Some(1), ..named }), Kernel::Quiet).write.unwrap();
    p("and the record then describes the system it went back to", undone.version() == "1.4.6" && undone.digest == kept.digest && undone.image_len == kept.image_len);
    p("an image-only rollback leaves the kernel", !r.restore_kernel);
    let tk = Record { trial: Some(1), with_kernel: true, ..base };
    p("a release that moved the kernel takes it back too", decide(Some(tk), Kernel::Quiet).restore_kernel);
    let rk = decide(Some(Record { with_kernel: true, ..base }), Kernel::RolledBack);
    p("a failed kernel takes its image back", rk.mount == 0 && rk.write.map(|w| w.trial) == Some(None) && !rk.restore_kernel);
    let hold = decide(Some(Record { with_kernel: true, ..base }), Kernel::AppliedNow);
    p("the image waits for its kernel and spends no try", hold.mount == 0 && hold.write.is_none());
    p("no try is spent on a boot that restarts into a new kernel", decide(Some(base), Kernel::AppliedNow).write.is_none());

    p("settling a trial keeps the new slot", settle(&t1).map(|w| (w.active, w.previous, w.trial)) == Some((1, 1, None)));
    p("there is nothing to settle on a kept record", settle(&kept).is_none());
    let after = pending(Some(&kept), 1, 0, 5, [1; 32], "1.4.7", "s", true);
    p("the next update targets the spare and remembers the running one", after.active == 0 && after.previous == 1 && after.seq == kept.seq + 1);
    c
}

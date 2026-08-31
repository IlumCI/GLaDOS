//! Putting a staged image where the boot hook will find it.
//!
//! `hook` runs before `ExitBootServices` and applies whatever is staged, using
//! the firmware's own FAT driver. This is the other end: writing those files
//! from a kernel that is already running, through our FAT writer, over NVMe.
//!
//! ### The gate stays ranged
//!
//! NVMe writes are locked until somebody claims a range, and nothing here
//! lifts that further than it has to: the claim covers the boot partition's
//! LBA window and is dropped again on every path out, including the ones that
//! failed halfway. An updater built on a global unlock is a whole-disk writer
//! that happens to be aiming carefully, and it only has to miss once.
//!
//! ### Finding the right volume, not the first one
//!
//! "The first partition that parses as FAT" is the right default for a person
//! browsing a disk and the wrong one for this. Writing three files to
//! somebody's data volume and reporting success is the worst outcome available
//! here -- it would look exactly like an update that worked, until the next
//! boot changed nothing. So the volume has to carry `\EFI\BOOT\BOOTX64.EFI`
//! before it is written to.

use crate::dev::nvme;
use crate::store::{block, fat, fatw, sha256};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// The three files `hook` looks for, spelled the way the FAT writer wants
/// them. `update`'s own constants are UEFI paths with backslashes because that
/// is what the firmware's file protocol takes; this layer walks the directory
/// chain itself. `find` accepts either separator, `put` splits on '/'.
const STAGED: &str = "/GLADOS/STAGED.EFI";
const STAGED_SIG: &str = "/GLADOS/STAGED.SIG";
const FLAG: &str = "/GLADOS/UPDATE.FLG";

/// What makes a FAT partition the one this machine boots from, rather than
/// merely a FAT partition.
const MARKER: &str = "/EFI/BOOT/BOOTX64.EFI";

/// The magic `unlock_writes` wants, so a stray call cannot open the gate.
const CONFIRM: u64 = 0xD15EA5E;

pub struct Esp {
    pub index: u32,
    pub start_lba: u64,
    pub blocks: u64,
    pub volume: fat::Volume,
}

/// Find the volume this machine boots from.
///
/// The failure messages are the point of this function as much as the success
/// is: "no FAT filesystem at all" and "a FAT filesystem that is not a boot
/// volume" have completely different fixes, and the first is what a live ISO
/// boot looks like from in here.
pub fn find_esp() -> Result<Esp, String> {
    let layout = block::scan().map_err(|e| format!("cannot read the partition table: {:?}", e))?;

    let mut saw_fat = false;
    for p in layout.partitions.iter() {
        let Ok(v) = fat::Volume::mount(p.start_lba) else {
            continue;
        };
        saw_fat = true;
        if v.find(MARKER).is_err() {
            continue;
        }
        // Refused with the reason rather than attempted: FAT16 keeps its root
        // directory outside the cluster chain, and every routine in the writer
        // would address the wrong sectors for it.
        if v.kind() != fat::Kind::Fat32 {
            return Err(format!(
                "partition {} is the boot volume and is {:?} -- writing is FAT32 only",
                p.index,
                v.kind()
            ));
        }
        return Ok(Esp {
            index: p.index,
            start_lba: p.start_lba,
            blocks: p.block_count,
            volume: v,
        });
    }

    Err(String::from(if saw_fat {
        "no FAT partition on this disk carries \\EFI\\BOOT\\BOOTX64.EFI, \
         so none of them is a volume this machine boots from"
    } else {
        "there is no FAT filesystem on this disk. A live ISO cannot update \
         itself -- ISO 9660 is read-only and there is no writable ESP. \
         Install to disk first"
    }))
}

/// Write, then read back and compare.
///
/// A short write is a legal FAT outcome rather than an error, and `hook` makes
/// the same check when it swaps the boot image for the same reason: the file
/// being written is the one the next boot will run, so "the call returned Ok"
/// is not enough to know it is there.
fn put_verified(esp: &Esp, path: &str, data: &[u8]) -> Result<(), String> {
    fatw::put(&esp.volume, path, data).map_err(|e| format!("{}: {}", path, e))?;

    let entry = esp
        .volume
        .find(path)
        .map_err(|_| format!("{} is not there after being written", path))?;
    let back = esp
        .volume
        .read_file(&entry)
        .map_err(|_| format!("{} could not be read back", path))?;

    if back.len() != data.len() {
        return Err(format!(
            "{} read back as {} B of {} -- a short write",
            path,
            back.len(),
            data.len()
        ));
    }
    if sha256::hash(&back) != sha256::hash(data) {
        return Err(format!("{} read back as the right length of different bytes", path));
    }
    Ok(())
}

fn write_all(esp: &Esp, image: &[u8], sig: &[u8]) -> Result<(), String> {
    // The flag goes last, and that ordering is the whole crash-safety story:
    // a machine that loses power partway through has a half-written image and
    // no flag telling the next boot to believe in it. `hook` refuses a flag
    // with no staged image, so the reverse order would arm a boot against a
    // file that does not exist yet.
    put_verified(esp, STAGED, image)?;
    put_verified(esp, STAGED_SIG, sig)?;
    put_verified(esp, FLAG, b"1\n")
}

/// Stage an image and its signature, arming the next boot to apply them.
///
/// The caller is expected to have verified both already -- this does not
/// re-check the signature, because a staging function that verifies is a
/// staging function somebody will call *instead of* verifying. `hook` verifies
/// again at boot regardless, against the same pinned key, which is the check
/// that actually decides anything.
pub fn stage(image: &[u8], sig: &[u8]) -> Result<String, String> {
    let esp = find_esp()?;

    if !nvme::unlock_writes(CONFIRM, esp.start_lba, esp.blocks) {
        return Err(String::from(
            "the write gate refused the claim -- no NVMe controller, or no such range",
        ));
    }

    let outcome = write_all(&esp, image, sig);

    // Dropped on every path, including the ones that failed halfway. A gate
    // left open by an error is a gate that was never closed.
    nvme::lock_writes();

    outcome?;
    Ok(format!(
        "staged {} B to partition {} (lba {}..{}); it is applied at the next boot",
        image.len(),
        esp.index,
        esp.start_lba,
        esp.start_lba + esp.blocks
    ))
}

/// Remove a staged update, so a machine can be disarmed without booting it.
///
/// The flag first: while it is gone the other two are inert, so an interrupted
/// clear leaves a machine that boots what it already had.
pub fn unstage() -> Result<String, String> {
    let esp = find_esp()?;

    if !nvme::unlock_writes(CONFIRM, esp.start_lba, esp.blocks) {
        return Err(String::from("the write gate refused the claim"));
    }
    let mut gone: Vec<&str> = Vec::new();
    for path in [FLAG, STAGED_SIG, STAGED] {
        if esp.volume.find(path).is_ok() && fatw::remove(&esp.volume, path).is_ok() {
            gone.push(path);
        }
    }
    nvme::lock_writes();

    if gone.is_empty() {
        return Ok(String::from("nothing was staged"));
    }
    Ok(format!("removed {}", gone.join(", ")))
}

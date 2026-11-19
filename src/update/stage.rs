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

/// What makes a boot volume *this* machine's rather than one it can merely see.
///
/// A Windows ESP carries the same marker as its fallback path, so the marker
/// alone matches it perfectly -- and `block::scan` reads NVMe and nothing else,
/// so a machine booted from the USB stick walks the internal disk here and
/// finds exactly that. Everything downstream would then read one volume and
/// write another, since `repairs::at_boot` goes through the firmware, which
/// reads the medium the image was actually loaded from.
const PAYLOAD: &str = "/GLADOS";

use crate::dev::nvme::CONFIRM;

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
    let mut not_ours: Option<u32> = None;
    for p in layout.partitions.iter() {
        let Ok(v) = fat::Volume::mount(p.start_lba) else {
            continue;
        };
        saw_fat = true;
        let Ok(marker) = v.find(MARKER) else {
            continue;
        };
        // **Is this the volume we actually booted from?**
        //
        // `block::scan` reads NVMe and nothing else, so on a machine booted
        // from removable media this loop walks the *internal* disk -- and a
        // Windows ESP carries `\EFI\BOOT\BOOTX64.EFI` as its fallback path,
        // so it matches the marker perfectly. Everything downstream then reads
        // one volume and writes another: `repairs::at_boot` goes through the
        // firmware, which reads the medium the image was loaded from, while
        // `record` would write here. Split-brain, on somebody else's ESP.
        //
        // The payload directory settles it, and comparing sizes does not.
        // `LoadedImage::ImageSize` is the image *in memory*, sections expanded
        // and aligned, where a directory entry is the file on disk -- 6,356,992
        // against 5,119,488 for one build, which is not a mismatch but two
        // different quantities. A GLaDOS boot volume carries the payload beside
        // the loader; a Windows ESP does not.
        let _ = marker;
        if !v.find(PAYLOAD).map(|e| e.is_dir).unwrap_or(false) {
            not_ours = Some(p.index);
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

    // Named rather than folded into "no boot volume", because the two have
    // completely different fixes and this one is the ordinary consequence of
    // booting the USB stick: nothing is wrong with the machine, the writable
    // ESP is simply not on the disk this layer can see.
    if let Some(idx) = not_ours {
        return Err(format!(
            "partition {} is a boot volume carrying no {} directory, so it is somebody else's ESP rather than this machine's -- writing to it would be writing to theirs. Booted from removable media? block::scan reads NVMe and nothing else, so the volume this image came from is not visible from here at all",
            idx, PAYLOAD
        ));
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

/// Write one file to the boot volume, verified, over the ranged gate.
///
/// Exposed because `repairs` needs exactly this and nothing else about
/// staging. Keeping the gate here rather than letting a second module claim its
/// own is the point: there is one place that opens the write window on this
/// disk, and one place that closes it on every path out.
pub fn put_one(esp: &Esp, path: &str, data: &[u8]) -> Result<String, String> {
    if !nvme::claim(CONFIRM, nvme::Holder::Esp, esp.start_lba, esp.blocks) {
        return Err(String::from(
            "the write gate refused the claim -- no NVMe controller, or no such range",
        ));
    }
    let outcome = put_verified(esp, path, data);
    nvme::release(nvme::Holder::Esp);
    outcome?;
    Ok(format!("wrote {} B to {}", data.len(), path))
}

/// Remove files that are there, quietly ignoring the ones that are not.
///
/// Order is the caller's and it matters: `unstage` and `repairs::clear` both
/// pass the arming flag first, so an interrupted removal leaves a machine that
/// does what it already did rather than a half-disarmed one.
pub fn remove_some(esp: &Esp, paths: &[&str]) -> Result<String, String> {
    if !nvme::claim(CONFIRM, nvme::Holder::Esp, esp.start_lba, esp.blocks) {
        return Err(String::from("the write gate refused the claim"));
    }
    let mut gone: Vec<String> = Vec::new();
    for path in paths {
        if esp.volume.find(path).is_ok() && fatw::remove(&esp.volume, path).is_ok() {
            gone.push(String::from(*path));
        }
    }
    nvme::release(nvme::Holder::Esp);

    if gone.is_empty() {
        return Ok(String::from("there was nothing to remove"));
    }
    Ok(format!("removed {}", gone.join(", ")))
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

    if !nvme::claim(CONFIRM, nvme::Holder::Esp, esp.start_lba, esp.blocks) {
        return Err(String::from(
            "the write gate refused the claim -- no NVMe controller, or no such range",
        ));
    }

    let outcome = write_all(&esp, image, sig);

    // Dropped on every path, including the ones that failed halfway. A gate
    // left open by an error is a gate that was never closed.
    nvme::release(nvme::Holder::Esp);

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

    if !nvme::claim(CONFIRM, nvme::Holder::Esp, esp.start_lba, esp.blocks) {
        return Err(String::from("the write gate refused the claim"));
    }
    let mut gone: Vec<&str> = Vec::new();
    for path in [FLAG, STAGED_SIG, STAGED] {
        if esp.volume.find(path).is_ok() && fatw::remove(&esp.volume, path).is_ok() {
            gone.push(path);
        }
    }
    nvme::release(nvme::Holder::Esp);

    if gone.is_empty() {
        return Ok(String::from("nothing was staged"));
    }
    Ok(format!("removed {}", gone.join(", ")))
}

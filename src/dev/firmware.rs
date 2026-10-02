//! Firmware images a device asks for by name.
//!
//! **One store for every driver, because "where is my blob" is not a driver's
//! question.** Intel's wireless parts want a `.ucode`, and every other wireless
//! family `dev::registry` names wants one too -- brcmfmac, mt76, rtw89 -- so a
//! loader per driver would be the arrangement the registry exists to replace,
//! arriving one layer down.
//!
//! **Read before `ExitBootServices`, all of it.** That is the only moment a
//! filesystem exists here, which is why the model is read then too. Which file a
//! part wants depends on what its registers say -- the GF63's controller is Snow
//! Owl, its radio is Harrier, and those two pick `so-a0-hr-b0` -- and reading a
//! register means MMIO the boot path has no business doing before the memory map
//! is taken. So the directory is read whole and the choice is made later. It is
//! a few megabytes, held in `LoaderData` for the life of the machine.
//!
//! **A namespace copy wins over the boot volume's**, deliberately. `fat get` a
//! file to `/fw/<name>` and the next lookup sees it, which is what iterating on
//! a bring-up needs: a redeploy and a reboot per attempt is a bare-metal trip per
//! attempt. The precedence is the one decision here with a claim of its own,
//! because getting it backwards fails silently: the operator stages a fixed image
//! and the driver keeps loading the old one.

use crate::sync::Racy;
use crate::uefi::{self, Blob, BootServices, Handle};
use alloc::string::String;
use alloc::vec::Vec;

/// Where the images live on the boot volume.
pub const DIR: &str = "\\GLADOS\\FW";
/// Where an operator stages one in the namespace.
pub const NS_DIR: &str = "/fw";

/// How many images, and how large. Bounded because this is read before
/// anything can report a pool exhausted by a directory somebody filled with the
/// wrong files; the largest image any driver here wants is Intel's at 1.5 MB.
const MAX_FILES: usize = 24;
const MAX_FILE: u64 = 8 << 20;
const MAX_TOTAL: u64 = 32 << 20;
const MAX_NAME: usize = 64;

#[derive(Clone, Copy)]
struct Entry {
    name: [u8; MAX_NAME],
    len: u8,
    blob: Blob,
}

static TABLE: Racy<[Option<Entry>; MAX_FILES]> = Racy::new([None; MAX_FILES]);
/// What was refused while reading, so `fw` can say so rather than the boot log
/// alone: on the GF63 there is no serial line to read the boot log from.
static SKIPPED: Racy<usize> = Racy::new(0);

/// Read every file under `DIR` into the table. Before `ExitBootServices` only.
pub fn load(bs: &BootServices, image: Handle) {
    let mut names: [([u8; MAX_NAME], u8, u64); MAX_FILES] = [([0; MAX_NAME], 0, 0); MAX_FILES];
    let mut found = 0usize;
    let mut skipped = 0usize;
    // Names first, then reads: opening a file while a directory read is in
    // progress is legal and is also exactly the kind of thing a firmware FAT
    // driver gets wrong.
    uefi::for_each_file(bs, image, DIR, |name, size| {
        if found == MAX_FILES || name.len() > MAX_NAME || size == 0 || size > MAX_FILE {
            skipped += 1;
            return;
        }
        names[found].0[..name.len()].copy_from_slice(name.as_bytes());
        names[found].1 = name.len() as u8;
        names[found].2 = size;
        found += 1;
    });
    let mut total = 0u64;
    let table = unsafe { &mut *TABLE.get() };
    let mut held = 0usize;
    for (raw, len, size) in names.iter().take(found) {
        let name = core::str::from_utf8(&raw[..*len as usize]).unwrap_or("");
        if total + size > MAX_TOTAL {
            skipped += 1;
            continue;
        }
        let mut path = [0u8; 96];
        let dir = DIR.as_bytes();
        path[..dir.len()].copy_from_slice(dir);
        path[dir.len()] = b'\\';
        path[dir.len() + 1..dir.len() + 1 + name.len()].copy_from_slice(name.as_bytes());
        let path = core::str::from_utf8(&path[..dir.len() + 1 + name.len()]).unwrap_or("");
        match uefi::read_file(bs, image, path) {
            Some(blob) => {
                total += blob.len as u64;
                table[held] = Some(Entry { name: *raw, len: *len, blob });
                held += 1;
                crate::serial_println!("glados: firmware {} ({} bytes)", name, blob.len);
            }
            None => skipped += 1,
        }
    }
    unsafe { *SKIPPED.get() = skipped };
}

/// An image, from wherever it was found.
pub enum Image {
    /// Off the boot volume, held for the life of the machine.
    Boot(&'static [u8]),
    /// Staged into the namespace, which takes precedence.
    Staged(Vec<u8>),
}

impl Image {
    pub fn bytes(&self) -> &[u8] {
        match self {
            Image::Boot(b) => b,
            Image::Staged(v) => v,
        }
    }
    pub fn source(&self) -> &'static str {
        match self {
            Image::Boot(_) => "boot volume",
            Image::Staged(_) => "namespace",
        }
    }
}

/// The boot volume's copy, by name.
///
/// Case-insensitive, because FAT is: a volume written on one host and read on
/// another may hand back `IWLWIFI-SO-A0-HR-B0-89.UCODE` for a file that was
/// copied in lower case, and the name a driver computes is lower case.
fn on_boot_volume(name: &str) -> Option<&'static [u8]> {
    let table = unsafe { &*TABLE.get() };
    table.iter().flatten().find_map(|e| {
        let have = core::str::from_utf8(&e.name[..e.len as usize]).ok()?;
        have.eq_ignore_ascii_case(name).then(|| e.blob.as_slice())
    })
}

/// The rule, pure: a staged copy wins.
fn pick(staged: Option<Vec<u8>>, boot: Option<&'static [u8]>) -> Option<Image> {
    match (staged, boot) {
        (Some(v), _) => Some(Image::Staged(v)),
        (None, Some(b)) => Some(Image::Boot(b)),
        (None, None) => None,
    }
}

/// The image a driver asked for, or `None` when nobody has provided it.
pub fn get(name: &str) -> Option<Image> {
    let staged = crate::sysbox::read_blob(&alloc::format!("{}/{}", NS_DIR, name));
    pick(staged, on_boot_volume(name))
}

/// What the boot volume provided, as `(name, bytes)`.
pub fn list() -> Vec<(String, usize)> {
    let table = unsafe { &*TABLE.get() };
    table
        .iter()
        .flatten()
        .map(|e| (String::from_utf8_lossy(&e.name[..e.len as usize]).into_owned(), e.blob.len))
        .collect()
}

/// How many files under `DIR` were refused: too many, too large, or unreadable.
pub fn skipped() -> usize {
    unsafe { *SKIPPED.get() }
}

/// The rule, with no volume and no namespace.
pub fn checks() -> Vec<(&'static str, bool)> {
    static BOOT: [u8; 3] = [1, 2, 3];
    let mut out = Vec::new();
    out.push((
        "a staged copy is preferred over the boot volume's, so a fix can be tried without a redeploy",
        matches!(pick(Some(alloc::vec![9]), Some(&BOOT)), Some(Image::Staged(ref v)) if v[..] == [9]),
    ));
    out.push((
        "the boot volume's copy is used when nothing is staged",
        matches!(pick(None, Some(&BOOT)), Some(Image::Boot(b)) if b == &BOOT[..]),
    ));
    out.push(("and an image nobody provided is absent rather than empty", pick(None, None).is_none()));
    out.push((
        "a name that is not on the boot volume is not found there",
        on_boot_volume("glados-no-such-firmware.bin").is_none(),
    ));
    // The table's caps have to admit what the one driver that uses it needs.
    out.push((
        "the caps admit Intel's 1.5 MB image with room, and its name",
        MAX_FILE > 1_527_004 && "iwlwifi-so-a0-hr-b0-89.ucode".len() <= MAX_NAME,
    ));
    out
}

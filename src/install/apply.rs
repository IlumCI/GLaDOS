//! Putting a plan on the disk.
//!
//! `plan` decides and `gpt` encodes; this is the only file in the installer
//! that writes, and every write goes through a named `nvme::Holder` window
//! that is claimed for exactly the blocks in question and released on every
//! path out.
//!
//! **The operator confirms the plan they were shown, not a mode.** `install
//! plan erase` prints the exact table and a code that is a digest of it and of
//! the disk it was computed against; `install erase <code>` recomputes both
//! and refuses on any difference. So a disk that changed between the two
//! commands -- a stick pulled, a partition made elsewhere -- cannot be written
//! under a confirmation that was given for something else. `update stage`
//! makes the same bargain about an image.
//!
//! Order of an erase, and why:
//!
//! 1. Everything that can refuse, refuses first. Nothing is written until the
//!    namespace has been cut loose from the old store, because that is the
//!    step that can fail and the disk must still be whole if it does.
//! 2. Tables: backup, then primary, then both read back through the parser
//!    boot uses. A table that does not read back stops the install there.
//! 3. System slot A, written from the image the boot carried, hashed on the
//!    way out and again read back off the disk, then mounted as a check.
//! 4. Slot B's head is zeroed, so a stale image from a previous install can
//!    never be mistaken for a system.
//! 5. The store is formatted and the namespace committed into it.
//!
//! The ESP is P2; until then an installed disk carries a system and a store
//! and does not boot on its own.

use super::gpt;
use super::plan::{self, Disk, Mode, Plan, Role};
use crate::dev::nvme::{self, Holder, CONFIRM};
use crate::store::block::{self, Scheme};
use crate::store::cas::{DmaBuf, STREAM};
use crate::store::sha256;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// The disk as it is now.
pub fn survey() -> Result<Disk, String> {
    if !nvme::present() {
        return Err("no NVMe disk".into());
    }
    let layout = block::scan().map_err(|e| format!("cannot read the partition table: {:?}", e))?;
    Ok(Disk {
        geometry: gpt::Geometry { block_size: block::block_size(), blocks: block::block_count() },
        scheme: layout.scheme,
        partitions: layout.partitions,
    })
}

fn gib(blocks: u64, bs: u32) -> String {
    let bytes = blocks * bs as u64;
    if bytes >= 10 << 30 {
        format!("{} GiB", bytes >> 30)
    } else {
        format!("{} MiB", bytes >> 20)
    }
}

/// The plan as the operator reads it, one line per partition, existing ones
/// first. The confirmation code is a digest of exactly these lines.
pub fn describe(disk: &Disk, p: &Plan) -> Vec<String> {
    let bs = disk.geometry.block_size;
    let mut out = Vec::new();
    out.push(format!(
        "disk: {} ({} blocks of {} B), {}",
        gib(disk.geometry.blocks, bs),
        disk.geometry.blocks,
        bs,
        match disk.scheme {
            Scheme::Gpt => "GPT",
            Scheme::Mbr => "MBR",
            Scheme::None => "no partition table",
        }
    ));
    match p.mode {
        Mode::Erase => {
            if !disk.partitions.is_empty() {
                out.push(format!("ERASED: {} existing partition(s) and everything on them", disk.partitions.len()));
                for e in &disk.partitions {
                    out.push(format!("  gone  #{} {} at {}, {}", e.index, e.kind(), e.start_lba, gib(e.block_count, bs)));
                }
            }
        }
        Mode::Alongside => {
            for e in &p.kept {
                out.push(format!("  kept  #{} {} at {}, {}", e.index, e.kind(), e.start_lba, gib(e.block_count, bs)));
            }
        }
    }
    for n in &p.new {
        out.push(format!("  new   {} at {}, {}", n.role.name(), n.first_lba, gib(n.blocks, bs)));
    }
    out
}

/// Eight hex characters naming this plan on this disk.
pub fn code(disk: &Disk, p: &Plan) -> String {
    let mut h = sha256::Sha256::new();
    for line in describe(disk, p) {
        h.update(line.as_bytes());
        h.update(b"\n");
    }
    let d = h.finish();
    let hx = sha256::short_hex(&d);
    String::from(core::str::from_utf8(&hx[..8]).unwrap_or("????????"))
}

/// What stops an install before any byte is written. Pure over the facts, so
/// the wizard and the verb refuse for the same reasons in the same words.
pub fn blockers(disk: &Disk, root_len: Option<usize>, disk_in_use: bool, has_kernel: bool, p: &Plan) -> Vec<String> {
    let mut why = Vec::new();
    if !has_kernel {
        why.push("this boot did not keep its own kernel file to copy".into());
    }
    // The EROFS reader counts in 512-byte sectors; on a 4Kn disk it would
    // read a slot at eight times its real offset. The planner handles 4K;
    // the reader is what is owed.
    if disk.geometry.block_size != 512 {
        why.push(format!(
            "the disk has {}-byte blocks and the system reader only understands 512 yet",
            disk.geometry.block_size
        ));
    }
    if disk_in_use {
        why.push("a volume on this disk is mounted -- reboot from the install image and try again".into());
    }
    match (root_len, p.part(Role::RootA)) {
        (None, _) => why.push("this boot carried no system image to install".into()),
        (Some(n), Some(a)) if n as u64 > a.blocks * disk.geometry.block_size as u64 => {
            why.push(format!("the system image ({} MiB) is larger than a slot", n >> 20))
        }
        _ => {}
    }
    why
}

fn random_guid() -> [u8; 16] {
    let mut g = [0u8; 16];
    crate::rng::fill(&mut g);
    // RFC 4122 version 4, variant 1, so tools that print GUIDs read it as one.
    g[7] = (g[7] & 0x0F) | 0x40;
    g[8] = (g[8] & 0x3F) | 0x80;
    g
}

/// Write `data` at `lba` inside `holder`'s window, which the caller claimed.
pub(crate) fn write_at(lba: u64, data: &[u8]) -> Result<(), String> {
    let bs = block::block_size() as usize;
    let mut buf = DmaBuf::new(data.len().min(STREAM)).map_err(|_| String::from("out of memory for a write buffer"))?;
    let per = buf.len();
    let mut off = 0usize;
    while off < data.len() {
        let n = (data.len() - off).min(per);
        buf[..n].copy_from_slice(&data[off..off + n]);
        buf[n..].fill(0);
        let blocks = n.div_ceil(bs) as u32;
        block::write(lba + (off / bs) as u64, blocks, &buf)
            .map_err(|e| format!("write at block {} failed: {:?}", lba + (off / bs) as u64, e))?;
        off += n;
    }
    Ok(())
}

pub(crate) fn read_at(lba: u64, len: usize) -> Result<Vec<u8>, String> {
    let bs = block::block_size() as usize;
    let mut buf = DmaBuf::new(len.min(STREAM)).map_err(|_| String::from("out of memory for a read buffer"))?;
    let per = buf.len();
    let mut out = Vec::with_capacity(len);
    let mut off = 0usize;
    while off < len {
        let n = (len - off).min(per);
        block::read(lba + (off / bs) as u64, n.div_ceil(bs) as u32, &mut buf)
            .map_err(|e| format!("read at block {} failed: {:?}", lba + (off / bs) as u64, e))?;
        out.extend_from_slice(&buf[..n]);
        off += n;
    }
    Ok(out)
}

/// Both tables, written and read back. The windows are released whatever
/// happens.
fn write_tables(g: &gpt::Geometry, img: &gpt::Image) -> Result<(), String> {
    let first = g.first_usable();
    if !nvme::claim(CONFIRM, Holder::Table, 0, first)
        || !nvme::claim(CONFIRM, Holder::TableTail, img.backup_lba, g.blocks - img.backup_lba)
    {
        nvme::release(Holder::Table);
        nvme::release(Holder::TableTail);
        return Err("the write gate refused the table windows".into());
    }
    // Backup first: a crash between the two leaves the old primary, which
    // still describes the old disk, rather than a new primary whose backup
    // is missing.
    let r = write_at(img.backup_lba, &img.backup).and_then(|_| write_at(0, &img.primary));
    nvme::release(Holder::Table);
    nvme::release(Holder::TableTail);
    r?;

    let bs = g.block_size as usize;
    let prim = read_at(0, img.primary.len())?;
    let back = read_at(img.backup_lba, img.backup.len())?;
    let arr = &prim[2 * bs..2 * bs + gpt::ENTRIES * gpt::ENTRY_SIZE];
    if !gpt::verify(&prim[bs..2 * bs], arr) {
        return Err("the primary table did not read back intact".into());
    }
    let tail = back.len() - bs;
    if !gpt::verify(&back[tail..], &back[..tail]) {
        return Err("the backup table did not read back intact".into());
    }
    if prim != img.primary || back != img.backup {
        return Err("the tables read back differently from what was written".into());
    }
    Ok(())
}

/// Write a whole system image into a slot, verified three ways: hashed as it
/// goes out, hashed again as it comes back off the disk, and mounted.
///
/// Used by the installer for slot A and, from P5, by system updates for the
/// spare slot -- one writer, so the two cannot disagree about what "written"
/// means.
pub fn write_slot(start: u64, blocks: u64, image: &[u8], say: &mut dyn FnMut(&str)) -> Result<(), String> {
    let bs = block::block_size() as usize;
    if image.len() as u64 > blocks * bs as u64 {
        return Err("the image is larger than the slot".into());
    }
    if !nvme::claim(CONFIRM, Holder::Slot, start, blocks) {
        return Err("the write gate refused the slot".into());
    }
    let want = sha256::hash(image);
    let step = (image.len() / 10).max(STREAM);
    let mut off = 0usize;
    let mut next = step;
    let mut r = Ok(());
    while off < image.len() {
        let n = (image.len() - off).min(STREAM * 8);
        if let Err(e) = write_at(start + (off / bs) as u64, &image[off..off + n]) {
            r = Err(e);
            break;
        }
        off += n;
        if off >= next {
            say(&format!("  system  {}%", off * 100 / image.len()));
            next += step;
        }
    }
    nvme::release(Holder::Slot);
    r?;

    say("  system  reading back");
    let mut h = sha256::Sha256::new();
    let mut off = 0usize;
    while off < image.len() {
        let n = (image.len() - off).min(STREAM * 8);
        let b = read_at(start + (off / bs) as u64, n)?;
        h.update(&b);
        off += n;
    }
    if h.finish() != want {
        return Err("the system slot did not read back as the image that was written".into());
    }
    crate::store::erofs::Image::mount(start).map_err(|_| String::from("the written slot does not mount"))?;
    Ok(())
}

/// Erase the disk and install, under the code `install plan erase` printed.
pub fn erase(given: &str, say: &mut dyn FnMut(&str)) -> Result<(), String> {
    install(Mode::Erase, given, say)
}

/// Install into the largest free stretch beside what is on the disk, under
/// the code `install plan alongside` printed. Nothing that exists moves.
pub fn alongside(given: &str, say: &mut dyn FnMut(&str)) -> Result<(), String> {
    install(Mode::Alongside, given, say)
}

/// The disk's own GPT, raw: its GUID and every 128-byte entry by slot. Only
/// the standard shape (128 entries of 128 bytes, array right after the
/// header) is accepted, and both CRCs must check: alongside rewrites the
/// table around somebody else's partitions, and a table this code did not
/// fully understand is one it refuses to rewrite.
fn raw_table() -> Result<([u8; 16], Vec<(usize, [u8; gpt::ENTRY_SIZE])>), String> {
    let bs = block::block_size() as usize;
    let h = read_at(1, bs)?;
    let n = u32::from_le_bytes(h[80..84].try_into().unwrap()) as usize;
    let size = u32::from_le_bytes(h[84..88].try_into().unwrap()) as usize;
    let at = u64::from_le_bytes(h[72..80].try_into().unwrap());
    if &h[0..8] != b"EFI PART" || n != gpt::ENTRIES || size != gpt::ENTRY_SIZE || at != 2 {
        return Err("the disk's partition table is not the standard shape, so it is not rewritten".into());
    }
    let array = read_at(at, n * size)?;
    if !gpt::verify(&h, &array) {
        return Err("the disk's partition table fails its own checksums -- repair it first".into());
    }
    let mut guid = [0u8; 16];
    guid.copy_from_slice(&h[56..72]);
    let mut out = Vec::new();
    for i in 0..n {
        let e = &array[i * size..(i + 1) * size];
        if e[0..16].iter().any(|&b| b != 0) {
            let mut raw = [0u8; gpt::ENTRY_SIZE];
            raw.copy_from_slice(e);
            out.push((i, raw));
        }
    }
    Ok((guid, out))
}

fn install(mode: Mode, given: &str, say: &mut dyn FnMut(&str)) -> Result<(), String> {
    let disk = survey()?;
    let word = if mode == Mode::Erase { "erase" } else { "alongside" };
    let p = plan::plan(&disk, mode, &plan::Want::DEFAULT).map_err(|r| r.say())?;
    let want = code(&disk, &p);
    if given != want {
        return Err(format!(
            "that code is not this plan's -- run 'install plan {}' and type the code it prints (now {})",
            word, want
        ));
    }
    let root = crate::store::erofs::ram();
    let carried = super::carried();
    let has_kernel = carried.iter().any(|(p, _)| p == "EFI/BOOT/BOOTX64.EFI");
    let why = blockers(&disk, root.map(|r| r.len()), crate::linux::disk::uses_disk_outside(&p.kept.iter().map(|k| (k.start_lba, k.block_count)).collect::<Vec<_>>()), has_kernel, &p);
    if let Some(w) = why.first() {
        return Err(w.clone());
    }
    let root = root.unwrap_or(&[]);
    // Read before anything is written: what alongside keeps, exactly.
    let kept_raw = if mode == Mode::Alongside { Some(raw_table()?) } else { None };

    // 1. Free the namespace from the old store before anything is written.
    if crate::store::mounted() {
        let (n, bytes) = crate::sysbox::detach_from_store()
            .map_err(|p| format!("could not bring {} into memory, so nothing was written", p))?;
        say(&format!("  memory  {} stored file(s), {} KiB, brought into memory", n, bytes / 1024));
        crate::store::unmount();
    }

    // 2. Tables.
    let g = disk.geometry;
    let ents = p.entries(|_| random_guid());
    let img = match &kept_raw {
        None => gpt::encode(&g, &random_guid(), &ents),
        Some((guid, raw)) => {
            // Every entry the disk has must be one the plan kept, in its slot.
            let kept: Vec<(usize, [u8; gpt::ENTRY_SIZE])> =
                raw.iter().filter(|(i, _)| p.kept.iter().any(|k| k.index as usize - 1 == *i)).cloned().collect();
            if kept.len() != raw.len() || kept.len() != p.kept.len() {
                return Err("the disk changed since it was surveyed -- plan again".into());
            }
            gpt::encode_over(&g, guid, &ents, &kept)
        }
    }
    .ok_or("the plan did not encode")?;
    write_tables(&g, &img)?;
    say(&format!("  tables  written and read back{}", if mode == Mode::Alongside { format!(", {} existing partition(s) kept byte for byte", p.kept.len()) } else { String::new() }));

    // 3. The boot partition.
    let e = p.part(Role::Esp).ok_or("the plan has no boot partition")?;
    // Slot A, kept: the record a later system update starts from. Written
    // with the ESP rather than after, so there is no installed disk that
    // lacks one.
    let rec = super::slot::Record {
        seq: 1,
        active: 0,
        previous: 0,
        trial: None,
        with_kernel: false,
        image_len: root.len() as u64,
        digest: sha256::hash(root),
        prev_len: root.len() as u64,
        prev_digest: sha256::hash(root),
        ..super::slot::pending(None, 0, 0, 0, [0; 32], crate::VERSION, "", false)
    }
    .encode();
    write_esp(e.first_lba, e.blocks, &carried, &rec, say)?;

    // 4. System slot A.
    let a = p.part(Role::RootA).ok_or("the plan has no system slot")?;
    write_slot(a.first_lba, a.blocks, root, say)?;
    say(&format!("  system  {} MiB in slot A, verified and mountable", root.len() >> 20));

    // 5. Slot B's head, so no earlier install's image looks like a system.
    if let Some(b) = p.part(Role::RootB) {
        if !nvme::claim(CONFIRM, Holder::Slot, b.first_lba, 8) {
            return Err("the write gate refused slot B".into());
        }
        let r = write_at(b.first_lba, &[0u8; 4096]);
        nvme::release(Holder::Slot);
        r?;
    }

    // 6. The store, and the namespace into it.
    let (start, blocks) = crate::store::init().map_err(|e| format!("the store did not format: {:?}", e))?;
    let s = p.part(Role::Store).ok_or("the plan has no store")?;
    if start != s.first_lba || blocks != s.blocks {
        return Err(format!("the store formatted at {}+{}, not where the plan put it", start, blocks));
    }
    say(&format!("  store   {} formatted", gib(blocks, g.block_size)));
    crate::sysbox::snap_now();

    // 7. The firmware's boot entry, by the ESP's GUID as the table now says.
    let esp = block::scan()
        .ok()
        .and_then(|l| l.partitions.into_iter().find(|q| q.start_lba == e.first_lba))
        .ok_or("the boot partition is not in the table that was just written")?;
    match super::bootopt::register(&esp) {
        Ok(n) => say(&format!("  firmware Boot{:04X} \"GLaDOS\" is first in the boot order", n)),
        // Not fatal: the kernel is on the removable-media path, so firmware
        // boots it from this disk regardless. Said, because on a laptop with
        // another system the entry is what puts GLaDOS first.
        Err(w) => say(&format!("  firmware {} -- the disk still boots by the fallback path", w)),
    }
    Ok(())
}

/// The ESP: formatted with every file this boot carried plus an install
/// identity, then every file read back through `store::fat`, the reader the
/// kernel uses for FAT everywhere else.
fn write_esp(start: u64, blocks: u64, carried: &[(String, &'static [u8])], rec: &[u8], say: &mut dyn FnMut(&str)) -> Result<(), String> {
    use crate::store::fatfmt;
    let mut id = Vec::new();
    let mut raw = [0u8; 16];
    crate::rng::fill(&mut raw);
    for b in raw {
        id.extend_from_slice(format!("{:02x}", b).as_bytes());
    }
    id.extend_from_slice(b"\n");
    let mut files: Vec<fatfmt::File> = carried.iter().map(|(p, d)| fatfmt::File { path: p, data: d }).collect();
    files.push(fatfmt::File { path: "GLADOS/INSTALL.ID", data: &id });
    files.push(fatfmt::File { path: "GLADOS/SLOT1.REC", data: rec });
    let l = fatfmt::layout(blocks, &files).map_err(String::from)?;
    if !nvme::claim(CONFIRM, Holder::Esp, start, blocks) {
        return Err("the write gate refused the boot partition".into());
    }
    let r = fatfmt::write(start, &l, &files, say);
    nvme::release(Holder::Esp);
    r?;
    let vol = crate::store::fat::Volume::mount(start).map_err(|_| String::from("the new boot partition does not mount"))?;
    for f in &files {
        let ent = vol.find(f.path).map_err(|_| format!("{} is not on the new boot partition", f.path))?;
        let back = vol.read_file(&ent).map_err(|_| format!("{} does not read back", f.path))?;
        if sha256::hash(&back) != sha256::hash(f.data) {
            return Err(format!("{} read back differently", f.path));
        }
    }
    say(&format!("  boot    {} file(s), read back through the FAT reader", files.len()));
    Ok(())
}

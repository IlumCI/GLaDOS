//! Installing GLaDOS onto a disk.
//!
//! In order of what exists: `plan` decides the partition table, `gpt` encodes
//! it. Writing it, laying down the ESP and the system image, and formatting
//! the store come after -- and nothing here writes until every claim below
//! passes, because the first table this tree writes lands on a disk that may
//! hold somebody's other operating system.
//!
//! `design/installer.md` is the plan this module is the first piece of.

pub mod apply;
pub mod bootopt;
pub mod feed;
pub mod gpt;
pub mod plan;
pub mod slot;

use crate::store::block::{Partition, Scheme, GUID_ESP};
use crate::sync::Racy;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use gpt::Geometry;
use plan::{Disk, Mode, Refusal, Role, Want};

/// What this boot read off its own volume and an install copies to the new
/// one, as `(path on the ESP, bytes)`. Filled before ExitBootServices, which is
/// before the heap, so a fixed table rather than a `Vec`; the blobs are the
/// firmware's pools and live for the machine's life.
static CARRIED: Racy<[Option<(&'static str, &'static [u8])>; 8]> = Racy::new([None; 8]);

/// Record a file the boot volume provided. Called from `main` only.
pub fn carry(path: &'static str, data: &'static [u8]) {
    let t = unsafe { CARRIED.get() };
    if let Some(slot) = t.iter_mut().find(|s| s.is_none()) {
        *slot = Some((path, data));
    }
}

/// Everything the boot carried that belongs on an installed ESP: the table
/// above and every firmware file, the latter under `GLADOS/FW/`.
pub fn carried() -> Vec<(String, &'static [u8])> {
    let mut out: Vec<(String, &'static [u8])> =
        unsafe { CARRIED.get() }.iter().flatten().map(|(p, d)| (String::from(*p), *d)).collect();
    for (name, data) in crate::dev::firmware::boot_files() {
        out.push((format!("GLADOS/FW/{}", name), data));
    }
    out
}

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

fn part(index: u32, start: u64, blocks: u64, ty: [u8; 16]) -> Partition {
    Partition { index, start_lba: start, block_count: blocks, mbr_type: 0, type_guid: ty, unique_guid: [index as u8; 16] }
}

/// A 512 GB disk shaped like a Windows install with 60 GiB freed by shrinking
/// C: -- which lands *between* C: and the recovery partition, not at the end,
/// and that is the case a "tail of the disk" rule gets wrong.
fn windows_disk(gap_gib: u64) -> Disk {
    let g = Geometry { block_size: 512, blocks: 1_000_215_216 };
    let mib = MIB / 512;
    let esp = part(1, 2048, 100 * mib, GUID_ESP);
    let msr = part(2, esp.end_lba(), 16 * mib, [0x16; 16]);
    let rec_blocks = GIB / 512;
    let rec_start = (g.last_usable() + 1 - rec_blocks) / mib * mib;
    let c_end = rec_start - gap_gib * (GIB / 512);
    let c = part(3, msr.end_lba(), c_end - msr.end_lba(), [0xa2; 16]);
    let rec = part(4, rec_start, rec_blocks, [0xa4; 16]);
    Disk { geometry: g, scheme: Scheme::Gpt, partitions: vec![esp, msr, c, rec] }
}

fn overlaps(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 < b.0 + b.1 && b.0 < a.0 + a.1
}

/// Every claim, named. Pure: no device, no write, nothing staged.
pub fn checks() -> Vec<(String, bool)> {
    let mut c: Vec<(String, bool)> = Vec::new();
    let want = Want::DEFAULT;

    c.push(("crc32 is the IEEE one GPT uses (check value of 123456789)".into(),
        gpt::crc32(b"123456789") == 0xCBF4_3926));

    // --- erase, 512-byte blocks -------------------------------------------
    let g = Geometry { block_size: 512, blocks: 500_118_192 }; // a 256 GB SSD
    let blank = Disk { geometry: g, scheme: Scheme::None, partitions: Vec::new() };
    match plan::plan(&blank, Mode::Erase, &want) {
        Ok(p) => {
            let a = MIB / 512;
            c.push(("erase lays out exactly ESP, system A, system B, store".into(),
                p.new.iter().map(|x| x.role).collect::<Vec<_>>()
                    == [Role::Esp, Role::RootA, Role::RootB, Role::Store]));
            c.push(("every new partition starts and ends on 1 MiB".into(),
                p.new.iter().all(|x| x.first_lba % a == 0 && (x.first_lba + x.blocks) % a == 0)));
            c.push(("every new partition is inside the usable area".into(),
                p.new.iter().all(|x| x.first_lba >= g.first_usable()
                    && x.first_lba + x.blocks - 1 <= g.last_usable())));
            c.push(("no two new partitions overlap, and they are contiguous".into(),
                p.new.windows(2).all(|w| w[0].first_lba + w[0].blocks == w[1].first_lba)));
            let store = p.part(Role::Store).unwrap();
            c.push(("the store takes the rest, up to the last aligned block".into(),
                g.last_usable() + 1 - (store.first_lba + store.blocks) < a));
            c.push(("erase keeps nothing that was there".into(), p.kept.is_empty()));

            // Encode it and read it back through the kernel's own parser.
            let ents = p.entries(|i| [i as u8 + 1; 16]);
            match gpt::encode(&g, &[0x5a; 16], &ents) {
                Some(img) => {
                    let bs = 512usize;
                    let arr = &img.primary[2 * bs..2 * bs + gpt::ENTRIES * gpt::ENTRY_SIZE];
                    let back = crate::store::block::parse_gpt_entries(arr, gpt::ENTRIES, gpt::ENTRY_SIZE);
                    c.push(("block::scan's parser reads back the four partitions written".into(),
                        back.len() == 4 && back.iter().zip(&p.new).all(|(b, n)|
                            b.start_lba == n.first_lba && b.block_count == n.blocks
                                && b.type_guid == n.role.type_guid())));
                    c.push(("each partition's own GUID reads back as the one written".into(),
                        back.iter().enumerate().all(|(i, b)| b.unique_guid == [i as u8 + 1; 16])));
                    c.push(("the protective MBR is a single 0xEE entry and parses as one".into(), {
                        let m = crate::store::block::parse_mbr(&img.primary[..bs]);
                        m.len() == 1 && m[0].mbr_type == 0xEE && m[0].start_lba == 1
                    }));
                    c.push(("the primary header and array pass their CRCs".into(),
                        gpt::verify(&img.primary[bs..2 * bs], arr)));
                    let tail = img.backup.len() - bs;
                    c.push(("the backup header and array pass their CRCs".into(),
                        gpt::verify(&img.backup[tail..], &img.backup[..tail])));
                    c.push(("the backup ends on the disk's last block".into(),
                        img.backup_lba + (img.backup.len() / bs) as u64 == g.blocks));
                    let h = &img.backup[tail..];
                    let my = u64::from_le_bytes(h[24..32].try_into().unwrap());
                    let alt = u64::from_le_bytes(h[32..40].try_into().unwrap());
                    let arr_lba = u64::from_le_bytes(h[72..80].try_into().unwrap());
                    c.push(("the backup swaps my/alternate and points at its own array".into(),
                        my == g.blocks - 1 && alt == 1 && arr_lba == img.backup_lba));
                    let mut bad = img.primary[bs..2 * bs].to_vec();
                    bad[48] ^= 1; // last_usable, one bit
                    c.push(("a header altered by one bit fails its CRC".into(), !gpt::verify(&bad, arr)));
                    let mut bad_arr = arr.to_vec();
                    bad_arr[127 * gpt::ENTRY_SIZE] = 1; // an *empty* slot
                    c.push(("an altered empty slot fails the array CRC -- empty slots count".into(),
                        !gpt::verify(&img.primary[bs..2 * bs], &bad_arr)));
                    c.push(("the last LBA written is inclusive".into(), {
                        let e = &arr[0..gpt::ENTRY_SIZE];
                        let first = u64::from_le_bytes(e[32..40].try_into().unwrap());
                        let last = u64::from_le_bytes(e[40..48].try_into().unwrap());
                        last - first + 1 == p.new[0].blocks
                    }));
                }
                None => c.push(("the erase plan encodes".into(), false)),
            }
        }
        Err(r) => c.push((format!("erase on a 256 GB disk plans ({})", r.say()), false)),
    }

    // --- the encoder's own refusals -----------------------------------------
    let e = |first, blocks| gpt::Entry { type_guid: [1; 16], unique_guid: [2; 16],
        first_lba: first, blocks, name: "x" };
    c.push(("the encoder refuses overlapping entries".into(),
        gpt::encode(&g, &[0; 16], &[(0, e(2048, 4096)), (1, e(4000, 4096))]).is_none()));
    c.push(("the encoder refuses an entry over the backup array".into(),
        gpt::encode(&g, &[0; 16], &[(0, e(g.last_usable() - 10, 20))]).is_none()));
    c.push(("the encoder refuses an entry over the primary array".into(),
        gpt::encode(&g, &[0; 16], &[(0, e(10, 4096))]).is_none()));
    c.push(("the encoder refuses two entries in one slot".into(),
        gpt::encode(&g, &[0; 16], &[(0, e(2048, 2048)), (0, e(8192, 2048))]).is_none()));

    // --- 4K sectors -----------------------------------------------------------
    let g4 = Geometry { block_size: 4096, blocks: 62_514_774 };
    c.push(("4K blocks: the array is four blocks and partitions may start at 6".into(),
        g4.array_blocks() == 4 && g4.first_usable() == 6));
    let d4 = Disk { geometry: g4, scheme: Scheme::None, partitions: Vec::new() };
    c.push(("4K blocks: erase aligns to 256 blocks".into(),
        plan::plan(&d4, Mode::Erase, &want).is_ok_and(|p| p.new.iter().all(|x| x.first_lba % 256 == 0))));

    // --- refusals -----------------------------------------------------------
    let small = Disk { geometry: Geometry { block_size: 512, blocks: 16 * GIB / 512 },
        scheme: Scheme::None, partitions: Vec::new() };
    c.push(("a 16 GiB disk is refused as too small, with both figures".into(),
        matches!(plan::plan(&small, Mode::Erase, &want), Err(Refusal::DiskTooSmall { .. }))));
    let mbr = Disk { geometry: g, scheme: Scheme::Mbr,
        partitions: vec![Partition { index: 1, start_lba: 2048, block_count: 1 << 20,
            mbr_type: 0x07, type_guid: [0; 16], unique_guid: [0; 16] }] };
    c.push(("alongside on an MBR disk is refused, not converted".into(),
        plan::plan(&mbr, Mode::Alongside, &want).err() == Some(Refusal::NotGpt)));
    c.push(("erase on that same MBR disk is allowed".into(), plan::plan(&mbr, Mode::Erase, &want).is_ok()));
    c.push(("an odd block size is refused".into(), plan::plan(&Disk {
        geometry: Geometry { block_size: 520, blocks: 1 << 30 }, scheme: Scheme::None,
        partitions: Vec::new() }, Mode::Erase, &want).err() == Some(Refusal::Geometry)));

    // --- alongside Windows ----------------------------------------------------
    let win = windows_disk(60);
    match plan::plan(&win, Mode::Alongside, &want) {
        Ok(p) => {
            let (gs, ge) = (win.partitions[2].end_lba(), win.partitions[3].start_lba);
            c.push(("alongside lands in the gap between C: and recovery".into(),
                p.new.iter().all(|x| x.first_lba >= gs && x.first_lba + x.blocks <= ge)));
            c.push(("alongside overlaps no existing partition".into(),
                p.new.iter().all(|x| win.partitions.iter().all(|o|
                    !overlaps((x.first_lba, x.blocks), (o.start_lba, o.block_count))))));
            c.push(("alongside keeps all four existing entries".into(), p.kept.len() == 4));
            let ents = p.entries(|i| [i as u8 + 1; 16]);
            c.push(("kept entries stay in their own slots and new ones take 4..7".into(),
                ents.iter().map(|(s, _)| *s).collect::<Vec<_>>() == [0, 1, 2, 3, 4, 5, 6, 7]
                    && ents[..4].iter().zip(&win.partitions)
                        .all(|((_, e), o)| e.first_lba == o.start_lba && e.blocks == o.block_count)));
            c.push(("GLaDOS gets its own ESP; Windows' is left alone".into(),
                ents.iter().filter(|(_, e)| e.type_guid == GUID_ESP).count() == 2));
            c.push(("the whole table encodes with nothing overlapping".into(),
                gpt::encode(&win.geometry, &[0; 16], &ents).is_some()));
        }
        Err(r) => c.push((format!("alongside a 60 GiB gap plans ({})", r.say()), false)),
    }
    let tight = windows_disk(20);
    c.push(("a 20 GiB gap is refused, naming the largest space there is".into(),
        matches!(plan::plan(&tight, Mode::Alongside, &want),
            Err(Refusal::NoRoom { largest, .. }) if largest / GIB == 20)));
    let mut again = windows_disk(60);
    if let Ok(p) = plan::plan(&windows_disk(60), Mode::Alongside, &want) {
        let s = p.part(Role::Store).unwrap();
        again.partitions.push(part(5, s.first_lba, s.blocks, crate::store::cas::GLADOS_TYPE_GUID));
    }
    c.push(("a disk already carrying GLaDOS is refused alongside".into(),
        plan::plan(&again, Mode::Alongside, &want).err() == Some(Refusal::AlreadyInstalled)));
    // 126 entries: the four of a Windows disk and 122 one-block ones. Only
    // the slot count is being asked, and it is asked before any gap is.
    let mut crowded = windows_disk(60);
    for i in 0..122u32 {
        crowded.partitions.push(part(5 + i, 40 + i as u64, 1, [9; 16]));
    }
    c.push(("a table with no four free slots is refused".into(),
        plan::plan(&crowded, Mode::Alongside, &want).err() == Some(Refusal::NoSlots)));

    c
}

pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut n = 0usize;
    let all = checks()
        .into_iter()
        .chain(crate::store::fatfmt::checks())
        .chain(bootopt::checks())
        .chain(slot::checks());
    for (what, good) in all {
        n += 1;
        if !good {
            kprintln!("    FAIL: {}", what);
            ok = false;
        }
    }
    kprintln!("    {} claim(s)", n);
    ok
}

/// `install`: what the disk is, what an install would do to it, and doing it.
pub fn cmd(rest: &str) {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED, WHITE, YELLOW};
    use crate::kprintln;
    let mut words = rest.split_whitespace();
    let verb = words.next().unwrap_or("");
    let arg = words.next().unwrap_or("");
    let mode = |w: &str| match w {
        "" | "erase" => Some(Mode::Erase),
        "alongside" => Some(Mode::Alongside),
        _ => None,
    };
    match verb {
        "" | "plan" => {
            let Some(m) = mode(arg) else {
                kprintln!("  install plan [erase|alongside]");
                return;
            };
            let disk = match apply::survey() {
                Ok(d) => d,
                Err(e) => {
                    console::set_color(LTRED);
                    kprintln!("  {}", e);
                    console::set_color(WHITE);
                    return;
                }
            };
            match plan::plan(&disk, m, &Want::DEFAULT) {
                Ok(p) => {
                    for line in apply::describe(&disk, &p) {
                        kprintln!("  {}", line);
                    }
                    let why = apply::blockers(
                        &disk,
                        crate::store::erofs::ram().map(|r| r.len()),
                        crate::linux::disk::uses_disk(),
                        carried().iter().any(|(p, _)| p == "EFI/BOOT/BOOTX64.EFI"),
                        &p,
                    );
                    if why.is_empty() {
                        console::set_color(YELLOW);
                        match m {
                            Mode::Erase => kprintln!(
                                "  to do this: install erase {}",
                                apply::code(&disk, &p)
                            ),
                            Mode::Alongside => kprintln!("  installing alongside is not built yet"),
                        }
                    } else {
                        console::set_color(LTRED);
                        for w in why {
                            kprintln!("  cannot install: {}", w);
                        }
                    }
                    console::set_color(WHITE);
                }
                Err(r) => {
                    console::set_color(LTRED);
                    kprintln!("  {}", r.say());
                    console::set_color(WHITE);
                }
            }
        }
        "erase" => {
            if arg.is_empty() {
                kprintln!("  'install plan erase' first, then 'install erase <code>'");
                return;
            }
            console::set_color(LTGRAY);
            let r = apply::erase(arg, &mut |line| kprintln!("{}", line));
            match r {
                Ok(()) => {
                    console::set_color(LTGREEN);
                    kprintln!("  installed -- remove the install medium and restart");
                }
                Err(e) => {
                    console::set_color(LTRED);
                    kprintln!("  stopped: {}", e);
                }
            }
            console::set_color(WHITE);
        }
        _ => kprintln!("  install [plan [erase|alongside]] | install erase <code>"),
    }
}

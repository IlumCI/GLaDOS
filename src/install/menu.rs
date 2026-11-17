//! The startup menu: which system to start, when there is more than one.
//!
//! The other systems are found where the firmware keeps them, in its own
//! `Boot####` entries -- Windows registers "Windows Boot Manager" there when
//! it installs, and so does every Linux. Probing other ESPs for loader files
//! would find the same thing less reliably and would need the firmware's
//! file system protocol for each volume.
//!
//! **Picking another system does not chainload it.** It sets `BootNext` to
//! that entry and restarts, so the other system is started by the firmware
//! from a clean reset, exactly as if it had been chosen in the firmware's own
//! menu. That is what keeps BitLocker quiet: Windows' measured boot sees the
//! firmware start `bootmgfw.efi`, not a loader started by another loader.
//! Loading it from here is kept as the fallback for a firmware that will not
//! take the variable.
//!
//! The menu shows only on an installed disk, only when there is another
//! system, and only for a few seconds; otherwise a boot does not wait at all.

use alloc::string::String;
use alloc::vec::Vec;

const LOAD_OPTION_ACTIVE: u32 = 0x1;
const LOAD_OPTION_HIDDEN: u32 = 0x8;
/// Category bits: zero is "boot", anything else (an application such as a
/// firmware setup entry) is not a system.
const LOAD_OPTION_CATEGORY: u32 = 0x1F00;

/// Seconds the menu waits before starting GLaDOS.
pub const WAIT_SECONDS: u32 = 4;

#[derive(Clone, Debug, PartialEq)]
pub struct Other {
    pub number: u16,
    pub name: String,
    /// The loader's path, for the screen and for the fallback.
    pub file: String,
    pub device_path: Vec<u8>,
}

fn ucs2(b: &[u8]) -> (String, usize) {
    let mut s = String::new();
    let mut i = 0;
    while i + 1 < b.len() {
        let c = u16::from_le_bytes([b[i], b[i + 1]]);
        i += 2;
        if c == 0 {
            return (s, i);
        }
        s.push(char::from_u32(c as u32).unwrap_or('?'));
    }
    (s, i)
}

/// An `EFI_LOAD_OPTION` that starts a system from a file on a disk
/// partition: active, not hidden, a boot entry, and a device path holding a
/// hard-drive node followed by a file node. The firmware's own entries for
/// whole devices, the network, its shell and its setup do not have that
/// shape, and are not systems a person means.
pub fn parse(number: u16, b: &[u8]) -> Option<Other> {
    if b.len() < 6 {
        return None;
    }
    let attr = u32::from_le_bytes(b[0..4].try_into().ok()?);
    let fp_len = u16::from_le_bytes([b[4], b[5]]) as usize;
    if attr & LOAD_OPTION_ACTIVE == 0 || attr & LOAD_OPTION_HIDDEN != 0 || attr & LOAD_OPTION_CATEGORY != 0 {
        return None;
    }
    let (name, used) = ucs2(&b[6..]);
    let dp = b.get(6 + used..6 + used + fp_len)?;
    let mut i = 0;
    let (mut hd, mut file) = (false, None);
    while i + 4 <= dp.len() {
        let (t, st, len) = (dp[i], dp[i + 1], u16::from_le_bytes([dp[i + 2], dp[i + 3]]) as usize);
        if len < 4 || i + len > dp.len() {
            return None;
        }
        match (t, st) {
            (0x04, 0x01) => hd = true,
            (0x04, 0x04) if hd => file = Some(ucs2(&dp[i + 4..i + len]).0),
            (0x7F, 0xFF) => break,
            _ => {}
        }
        i += len;
    }
    let file = file?;
    Some(Other { number, name, file, device_path: dp.to_vec() })
}

/// The systems to offer, in the firmware's boot order, without GLaDOS itself.
pub fn others(order: &[u16], entries: &[(u16, Vec<u8>)]) -> Vec<Other> {
    order
        .iter()
        .filter_map(|n| entries.iter().find(|(m, _)| m == n))
        .filter_map(|(n, b)| parse(*n, b))
        .filter(|o| o.name != super::bootopt::DESCRIPTION)
        .collect()
}

/// A key on the menu: which entry it picks, if any. `1`..`9` pick another
/// system by its place in the list; Enter or `0` picks GLaDOS.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Pick {
    Glados,
    Other(usize),
    Nothing,
}

pub fn key(c: char, n_others: usize) -> Pick {
    match c {
        '\r' | '\n' | '0' | 'g' | 'G' => Pick::Glados,
        '1'..='9' => {
            let i = c as usize - '1' as usize;
            if i < n_others { Pick::Other(i) } else { Pick::Nothing }
        }
        _ => Pick::Nothing,
    }
}

// ---------------------------------------------------------------------------
// The firmware's half.
// ---------------------------------------------------------------------------

fn name16(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(core::iter::once(0)).collect()
}

/// Every `Boot####` the boot order names, read through the runtime services
/// (which answer before ExitBootServices as well as after).
pub fn read_entries() -> (Vec<u16>, Vec<(u16, Vec<u8>)>) {
    let g = super::bootopt::GLOBAL;
    let mut buf = alloc::vec![0u8; 1024];
    let order: Vec<u16> = match crate::cpu::efi_get_variable(&name16("BootOrder"), &g, &mut buf) {
        Some(n) => buf[..n & !1].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
        None => Vec::new(),
    };
    let mut entries = Vec::new();
    let mut b = alloc::vec![0u8; 4096];
    for &n in &order {
        if let Some(len) = crate::cpu::efi_get_variable(&super::bootopt::var_name(n), &g, &mut b) {
            entries.push((n, b[..len].to_vec()));
        }
    }
    (order, entries)
}

/// Restart into another system by setting `BootNext`. Answers only on
/// failure: the firmware refused the variable or did not restart.
pub fn restart_into(o: &Other) -> String {
    let ok = crate::cpu::efi_set_variable(&name16("BootNext"), &super::bootopt::GLOBAL, &o.number.to_le_bytes());
    if !ok {
        return alloc::format!("the firmware would not set BootNext for {}", o.name);
    }
    crate::cpu::reboot();
}

#[repr(C)]
struct InputKey {
    scan: u16,
    ch: u16,
}

#[repr(C)]
struct TextInput {
    reset: extern "efiapi" fn(*mut TextInput, bool) -> crate::uefi::Status,
    read_key: extern "efiapi" fn(*mut TextInput, *mut InputKey) -> crate::uefi::Status,
}

/// The fallback: load the other system's loader from here and start it. The
/// device path in a boot entry is usually the short form (hard drive, file),
/// which `LoadImage` resolves on most firmware; where it does not, this says
/// so and GLaDOS starts instead.
fn chainload(bs: &crate::uefi::BootServices, image: crate::uefi::Handle, o: &Other) -> &'static str {
    type Load = extern "efiapi" fn(bool, crate::uefi::Handle, *const u8, *const u8, usize, *mut crate::uefi::Handle) -> crate::uefi::Status;
    type Start = extern "efiapi" fn(crate::uefi::Handle, *mut usize, *mut *mut u16) -> crate::uefi::Status;
    let load: Load = unsafe { core::mem::transmute(bs.load_image) };
    let start: Start = unsafe { core::mem::transmute(bs.start_image) };
    let mut h: crate::uefi::Handle = core::ptr::null_mut();
    if load(false, image, o.device_path.as_ptr(), core::ptr::null(), 0, &mut h) != 0 {
        return "and loading it from here failed too";
    }
    let mut n = 0usize;
    start(h, &mut n, core::ptr::null_mut());
    "it returned"
}

/// Before ExitBootServices, on an installed disk: offer the other systems,
/// if there are any, and wait a few seconds for a key.
fn say(st: &crate::uefi::SystemTable, s: &str) {
    crate::serial_println!("glados: {}", s.trim_start());
    let out = st.con_out;
    if out.is_null() {
        return;
    }
    let w: Vec<u16> = s.encode_utf16().chain("\r\n".encode_utf16()).chain(core::iter::once(0)).collect();
    unsafe { ((*out).output_string)(out, w.as_ptr()) };
}

pub fn at_boot(st: &crate::uefi::SystemTable, bs: &crate::uefi::BootServices, image: crate::uefi::Handle) {
    let (order, entries) = read_entries();
    let list = others(&order, &entries);
    if list.is_empty() {
        return;
    }
    let input = st.con_in as *mut TextInput;
    if input.is_null() {
        return;
    }
    say(st, "");
    say(st, "  Start which system?");
    say(st, "    [Enter] GLaDOS");
    for (i, o) in list.iter().enumerate().take(9) {
        say(st, &alloc::format!("    [{}] {}", i + 1, o.name));
    }
    unsafe { ((*input).reset)(input, false) };
    let ticks = WAIT_SECONDS * 20;
    let mut pick = Pick::Nothing;
    for t in 0..ticks {
        if t % 20 == 0 {
            crate::serial_println!("glados: menu: GLaDOS in {} s", WAIT_SECONDS - t / 20);
        }
        let mut k = InputKey { scan: 0, ch: 0 };
        if unsafe { ((*input).read_key)(input, &mut k) } == 0 {
            pick = key(char::from_u32(k.ch as u32).unwrap_or('\0'), list.len());
            if pick != Pick::Nothing {
                break;
            }
        }
        (bs.stall)(50_000);
    }
    if let Pick::Other(i) = pick {
        let o = &list[i];
        crate::serial_println!("glados: menu: restarting into {} (Boot{:04X})", o.name, o.number);
        say(st, &alloc::format!("  Restarting into {}...", o.name));
        let why = restart_into(o);
        say(st, &alloc::format!("  {} -- loading it directly", why));
        let r = chainload(bs, image, o);
        say(st, &alloc::format!("  {}; starting GLaDOS", r));
    }
}

pub fn checks() -> Vec<(String, bool)> {
    let mut c: Vec<(String, bool)> = Vec::new();
    let mut p = |s: &str, ok: bool| c.push((String::from(s), ok));
    let g = [0x22u8; 16];
    let win_dp = super::bootopt::device_path(1, 2048, 532480, &g, "\\EFI\\Microsoft\\Boot\\bootmgfw.efi");
    let win = super::bootopt::load_option("Windows Boot Manager", &win_dp);
    let ours = super::bootopt::load_option(super::bootopt::DESCRIPTION, &super::bootopt::device_path(5, 9, 9, &g, super::bootopt::KERNEL_PATH));
    // A whole-device entry, as OVMF makes for a disk: PciRoot/Pci/NVMe, no file.
    let mut dev = Vec::new();
    dev.extend_from_slice(&[0x02, 0x01, 12, 0, 0xD0, 0x41, 0x03, 0x0A, 0, 0, 0, 0]);
    dev.extend_from_slice(&[0x7F, 0xFF, 4, 0]);
    let whole = super::bootopt::load_option("UEFI QEMU NVMe Ctrl", &dev);
    let mut hidden = win.clone();
    hidden[0] |= LOAD_OPTION_HIDDEN as u8;
    let mut app = win.clone();
    app[1] = 0x01; // category 0x100: an application, not a system
    let w = parse(1, &win);
    p("Windows' entry is read as a system: its name and loader", w.as_ref().is_some_and(|o| o.name == "Windows Boot Manager" && o.file == "\\EFI\\Microsoft\\Boot\\bootmgfw.efi"));
    p("a whole-device entry is not a system", parse(2, &whole).is_none());
    p("nor a hidden one, nor an application", parse(3, &hidden).is_none() && parse(4, &app).is_none());
    let entries = alloc::vec![(5u16, ours), (1u16, win), (2u16, whole)];
    let list = others(&[5, 2, 1], &entries);
    p("GLaDOS does not offer itself, and the rest keep the boot order", list.len() == 1 && list[0].number == 1);
    p("Enter starts GLaDOS, a number starts that system", key('\r', 1) == Pick::Glados && key('1', 1) == Pick::Other(0));
    p("a number past the list does nothing", key('2', 1) == Pick::Nothing && key('x', 1) == Pick::Nothing);
    p("a truncated entry is refused rather than read past", parse(1, &entries[1].1[..20]).is_none());
    c
}

/// `boot`: the systems this machine can start, and restarting into one.
pub fn cmd(rest: &str) {
    use crate::kprintln;
    let mut w = rest.split_whitespace();
    match w.next().unwrap_or("") {
        "" | "list" => {
            let (order, entries) = read_entries();
            let list = others(&order, &entries);
            kprintln!("  boot order: {}", order.iter().map(|n| alloc::format!("{:04X}", n)).collect::<Vec<_>>().join(" "));
            for (n, b) in &entries {
                let (name, _) = ucs2(b.get(6..).unwrap_or(&[]));
                let mark = if list.iter().any(|o| o.number == *n) { "system" } else if name == super::bootopt::DESCRIPTION { "this" } else { "" };
                kprintln!("  Boot{:04X}  {:<7} {}", n, mark, name);
            }
            if list.is_empty() {
                kprintln!("  no other system -- the startup menu stays out of the way");
            } else {
                kprintln!("  'boot next <number>' restarts into one; the startup menu offers them for {} s", WAIT_SECONDS);
            }
        }
        "next" => {
            let Some(n) = w.next().and_then(|s| u16::from_str_radix(s.trim_start_matches("Boot"), 16).ok()) else {
                kprintln!("  boot next <number>, as 'boot' lists it");
                return;
            };
            let (order, entries) = read_entries();
            let Some(o) = others(&order, &entries).into_iter().find(|o| o.number == n) else {
                kprintln!("  Boot{:04X} is not a system this machine offers", n);
                return;
            };
            let _ = crate::sysbox::commit_now();
            kprintln!("  restarting into {}", o.name);
            let why = restart_into(&o);
            kprintln!("  {}", why);
        }
        "add" => {
            let (Some(i), Some(path)) = (w.next().and_then(|s| s.parse::<u32>().ok()), w.next()) else {
                kprintln!("  boot add <partition> <\\EFI\\...\\loader.efi> <name>");
                return;
            };
            let name: Vec<&str> = w.collect();
            let name = if name.is_empty() { String::from("Other system") } else { name.join(" ") };
            let Some(p) = crate::store::block::scan().ok().and_then(|l| l.partitions.into_iter().find(|p| p.index == i)) else {
                kprintln!("  no partition {}", i);
                return;
            };
            match super::bootopt::add(&p, path, &name) {
                Ok(n) => kprintln!("  Boot{:04X} \"{}\" added at the end of the boot order", n, name),
                Err(e) => kprintln!("  {}", e),
            }
        }
        _ => kprintln!("  boot [list | next <number> | add <partition> <path> <name>]"),
    }
}

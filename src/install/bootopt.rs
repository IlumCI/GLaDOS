//! A firmware boot entry for the installed disk: `Boot####` and `BootOrder`.
//!
//! The kernel lives at `\EFI\BOOT\BOOTX64.EFI` on GLaDOS's *own* ESP, the
//! removable-media path, so a firmware that loses or never honours the entry
//! still finds it -- and the staged updater, which swaps exactly that path,
//! keeps working unchanged. The entry is what puts GLaDOS first and names it
//! in the firmware's boot menu.
//!
//! Encoding is pure (`load_option`) and its layout is the UEFI spec's
//! `EFI_LOAD_OPTION` with a HardDrive media node naming the partition by its
//! GPT GUID; writing variables is the impure half.
//!
//! **A free `Boot####` number is found by asking whether a variable exists,
//! never by whether reading it succeeded.** A read into a small buffer fails
//! with "buffer too small" for an entry that is very much there, and treating
//! that as free overwrites somebody's boot entry -- Windows', on this laptop.

use alloc::string::String;
use alloc::vec::Vec;

/// EFI_GLOBAL_VARIABLE: 8be4df61-93ca-11d2-aa0d-00e098032b8c.
pub const GLOBAL: crate::uefi::Guid =
    crate::uefi::Guid { d1: 0x8be4_df61, d2: 0x93ca, d3: 0x11d2, d4: [0xaa, 0x0d, 0x00, 0xe0, 0x98, 0x03, 0x2b, 0x8c] };

pub const LOAD_OPTION_ACTIVE: u32 = 1;
pub const DESCRIPTION: &str = "GLaDOS";
pub const KERNEL_PATH: &str = "\\EFI\\BOOT\\BOOTX64.EFI";

fn ucs2(s: &str, out: &mut Vec<u8>) {
    for c in s.encode_utf16() {
        out.extend_from_slice(&c.to_le_bytes());
    }
    out.extend_from_slice(&[0, 0]);
}

/// The device path: HardDrive(partition, start, size, GPT GUID) /
/// File(path) / End.
pub fn device_path(part: u32, start: u64, size: u64, guid: &[u8; 16], path: &str) -> Vec<u8> {
    let mut d = Vec::new();
    // Media device path, HardDrive, 42 bytes.
    d.extend_from_slice(&[0x04, 0x01, 42, 0]);
    d.extend_from_slice(&part.to_le_bytes());
    d.extend_from_slice(&start.to_le_bytes());
    d.extend_from_slice(&size.to_le_bytes());
    d.extend_from_slice(guid);
    d.push(0x02); // partition format: GPT
    d.push(0x02); // signature type: GUID
    // Media device path, File path.
    let mut name = Vec::new();
    ucs2(path, &mut name);
    let len = (4 + name.len()) as u16;
    d.extend_from_slice(&[0x04, 0x04]);
    d.extend_from_slice(&len.to_le_bytes());
    d.extend_from_slice(&name);
    // End of entire device path.
    d.extend_from_slice(&[0x7F, 0xFF, 4, 0]);
    d
}

/// The bytes of `Boot####`.
pub fn load_option(desc: &str, path: &[u8]) -> Vec<u8> {
    let mut o = Vec::new();
    o.extend_from_slice(&LOAD_OPTION_ACTIVE.to_le_bytes());
    o.extend_from_slice(&(path.len() as u16).to_le_bytes());
    ucs2(desc, &mut o);
    o.extend_from_slice(path);
    o
}

/// `Boot0007` and the like, NUL-terminated, with the spec's upper-case hex.
pub fn var_name(n: u16) -> [u16; 9] {
    let mut out = [0u16; 9];
    for (i, c) in "Boot".encode_utf16().enumerate() {
        out[i] = c;
    }
    let hex = b"0123456789ABCDEF";
    for i in 0..4 {
        out[4 + i] = hex[((n >> (12 - 4 * i)) & 0xF) as usize] as u16;
    }
    out
}

/// `BootOrder` with `ours` first and nothing else moved or lost.
pub fn order_with(existing: &[u16], ours: u16) -> Vec<u16> {
    let mut o = Vec::with_capacity(existing.len() + 1);
    o.push(ours);
    o.extend(existing.iter().copied().filter(|&n| n != ours));
    o
}

const ORDER: [u16; 10] = [
    b'B' as u16, b'o' as u16, b'o' as u16, b't' as u16, b'O' as u16, b'r' as u16, b'd' as u16, b'e' as u16,
    b'r' as u16, 0,
];

fn read_order() -> Vec<u16> {
    let mut buf = [0u8; 512];
    match crate::cpu::efi_get_variable(&ORDER, &GLOBAL, &mut buf) {
        Some(n) => buf[..n & !1].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
        None => Vec::new(),
    }
}

/// Create (or reuse) GLaDOS's entry for the ESP `part` and put it first.
/// Answers the entry number.
pub fn register(part: &crate::store::block::Partition) -> Result<u16, String> {
    let dp = device_path(part.index, part.start_lba, part.block_count, &part.unique_guid, KERNEL_PATH);
    let opt = load_option(DESCRIPTION, &dp);
    // Reuse an entry that already says exactly this, so reinstalling does not
    // leave a trail of identical entries in the firmware menu.
    let mut free = None;
    let mut buf = alloc::vec![0u8; 4096];
    for n in 0..0x100u16 {
        let name = var_name(n);
        match crate::cpu::efi_variable_exists(&name, &GLOBAL) {
            Some(true) => {
                if crate::cpu::efi_get_variable(&name, &GLOBAL, &mut buf) == Some(opt.len())
                    && buf[..opt.len()] == opt[..]
                {
                    free = Some((n, true));
                    break;
                }
            }
            Some(false) if free.is_none() => free = Some((n, false)),
            Some(false) => {}
            None => return Err("the firmware's variable service did not answer".into()),
        }
    }
    let (n, existed) = free.ok_or("no free Boot#### number")?;
    if !existed && !crate::cpu::efi_set_variable(&var_name(n), &GLOBAL, &opt) {
        return Err("the firmware refused the boot entry".into());
    }
    let order = order_with(&read_order(), n);
    let bytes: Vec<u8> = order.iter().flat_map(|v| v.to_le_bytes()).collect();
    if !crate::cpu::efi_set_variable(&ORDER, &GLOBAL, &bytes) {
        return Err("the firmware refused the boot order".into());
    }
    // Read both back: a firmware that accepts a write and keeps nothing is
    // the failure that matters, and it is silent until the next power-on.
    if crate::cpu::efi_get_variable(&var_name(n), &GLOBAL, &mut buf) != Some(opt.len()) || buf[..opt.len()] != opt[..] {
        return Err("the boot entry did not read back".into());
    }
    if read_order().first() != Some(&n) {
        return Err("the boot order did not keep GLaDOS first".into());
    }
    Ok(n)
}

/// Add an entry for another loader -- `efibootmgr -c` -- at the end of the
/// boot order, so it is offered and never takes over. Answers its number.
pub fn add(part: &crate::store::block::Partition, path: &str, desc: &str) -> Result<u16, String> {
    let opt = load_option(desc, &device_path(part.index, part.start_lba, part.block_count, &part.unique_guid, path));
    let mut n = None;
    for i in 0..0x100u16 {
        match crate::cpu::efi_variable_exists(&var_name(i), &GLOBAL) {
            Some(false) => {
                n = Some(i);
                break;
            }
            Some(true) => {}
            None => return Err("the firmware's variable service did not answer".into()),
        }
    }
    let n = n.ok_or("no free Boot#### number")?;
    if !crate::cpu::efi_set_variable(&var_name(n), &GLOBAL, &opt) {
        return Err("the firmware refused the boot entry".into());
    }
    let mut order = read_order();
    order.retain(|&x| x != n);
    order.push(n);
    let bytes: Vec<u8> = order.iter().flat_map(|v| v.to_le_bytes()).collect();
    if !crate::cpu::efi_set_variable(&ORDER, &GLOBAL, &bytes) {
        return Err("the firmware refused the boot order".into());
    }
    Ok(n)
}

/// Pure claims about the encoding.
pub fn checks() -> Vec<(String, bool)> {
    let mut c: Vec<(String, bool)> = Vec::new();
    let g = [0x11u8; 16];
    let dp = device_path(1, 2048, 8_388_608, &g, KERNEL_PATH);
    c.push(("the HardDrive node is 42 bytes, typed media/hard drive".into(), dp[0] == 4 && dp[1] == 1 && dp[2] == 42));
    c.push(("it names partition 1 at 2048 for 8388608 blocks".into(),
        u32::from_le_bytes(dp[4..8].try_into().unwrap()) == 1
            && u64::from_le_bytes(dp[8..16].try_into().unwrap()) == 2048
            && u64::from_le_bytes(dp[16..24].try_into().unwrap()) == 8_388_608));
    c.push(("by GPT GUID".into(), dp[24..40] == g && dp[40] == 2 && dp[41] == 2));
    let flen = 4 + (KERNEL_PATH.len() + 1) * 2;
    c.push(("the file node carries the path in UCS-2 and its own length".into(),
        dp[42] == 4 && dp[43] == 4 && u16::from_le_bytes([dp[44], dp[45]]) as usize == flen
            && dp[46] == b'\\' && dp[47] == 0));
    c.push(("and the path ends in an end node".into(), dp[dp.len() - 4..] == [0x7F, 0xFF, 4, 0] && dp.len() == 42 + flen + 4));
    let o = load_option(DESCRIPTION, &dp);
    let dlen = (DESCRIPTION.len() + 1) * 2;
    c.push(("the load option is active and states the path's length".into(),
        u32::from_le_bytes(o[0..4].try_into().unwrap()) == 1 && u16::from_le_bytes([o[4], o[5]]) as usize == dp.len()));
    c.push(("its description is GLaDOS, NUL-terminated, then the path".into(),
        o[6] == b'G' && o[6 + dlen - 2..6 + dlen] == [0, 0] && o[6 + dlen..] == dp[..]));
    c.push(("Boot entry names are upper-case hex".into(),
        String::from_utf16_lossy(&var_name(0x00AB)[..8]) == "Boot00AB" && var_name(0)[8] == 0));
    c.push(("GLaDOS goes first and nothing else moves".into(), order_with(&[3, 0, 1], 7) == [7, 3, 0, 1]));
    c.push(("reinstalling does not list it twice".into(), order_with(&[3, 7, 1], 7) == [7, 3, 1]));
    c
}

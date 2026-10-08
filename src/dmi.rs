//! Who this machine is, read from the firmware rather than hardcoded.
//!
//! A boot screen that says "MSI GF63" is a lie on every other machine. The
//! vendor, product and board come from the SMBIOS tables the firmware
//! publishes in the EFI configuration table -- the same place `find_rsdp`
//! finds ACPI -- and the processor name from the CPUID brand leaves, which
//! every x86-64 part carries.
//!
//! Read once, early, **while boot services are still up**: the SMBIOS tables
//! can sit in boot-services memory that `ExitBootServices` reclaims, so the
//! strings are copied into our own buffers here and nothing holds a pointer
//! into firmware memory afterwards. The POST intro that consumes this draws
//! before the model read, which is before the exit, so the order is natural
//! rather than arranged.
//!
//! The parser is split from the pointer-walking so it can be fed a synthetic
//! structure table and asserted -- `selftest` does exactly that, because a
//! string-table walk off by one byte reads a plausible-looking wrong model
//! name, which is the failure a boot screen can least afford.

use crate::sync::Racy;
use crate::uefi::{Guid, SystemTable};

/// SMBIOS 2.x entry point. EB9D2D31-2D88-11D3-9A16-0090273FC14D.
const SMBIOS_GUID: Guid = Guid {
    d1: 0xEB9D_2D31,
    d2: 0x2D88,
    d3: 0x11D3,
    d4: [0x9A, 0x16, 0x00, 0x90, 0x27, 0x3F, 0xC1, 0x4D],
};
/// SMBIOS 3.x entry point. F2FD1544-9794-4A2C-992E-E5BBCF20E394.
const SMBIOS3_GUID: Guid = Guid {
    d1: 0xF2FD_1544,
    d2: 0x9794,
    d3: 0x4A2C,
    d4: [0x99, 0x2E, 0xE5, 0xBB, 0xCF, 0x20, 0xE3, 0x94],
};

/// Per-field buffer. Firmware strings are short; 63 printable bytes plus room
/// is more than any real vendor or model uses, and a longer one is truncated
/// rather than allowed to run off the end.
const N: usize = 64;

struct Info {
    v: [u8; N],
    vl: usize,
    p: [u8; N],
    pl: usize,
    b: [u8; N],
    bl: usize,
    c: [u8; N],
    cl: usize,
}

impl Info {
    const fn empty() -> Self {
        Self { v: [0; N], vl: 0, p: [0; N], pl: 0, b: [0; N], bl: 0, c: [0; N], cl: 0 }
    }
}

static INFO: Racy<Info> = Racy::new(Info::empty());

/// Copy a firmware string in, sanitised. SMBIOS strings are ASCII by spec, but
/// a byte outside printable ASCII would make the `from_utf8_unchecked` the
/// accessors use unsound, so anything else becomes '?': a wrong character on
/// screen is a far better failure than a wrong invariant.
fn store(dst: &mut [u8; N], len: &mut usize, src: &[u8]) {
    let n = src.len().min(N);
    for k in 0..n {
        let c = src[k];
        dst[k] = if (0x20..0x7f).contains(&c) { c } else { b'?' };
    }
    *len = n;
}

fn trim(s: &[u8]) -> &[u8] {
    let mut a = 0;
    let mut b = s.len();
    while a < b && s[a] == b' ' {
        a += 1;
    }
    while b > a && s[b - 1] == b' ' {
        b -= 1;
    }
    &s[a..b]
}

/// The `idx`-th (1-based) NUL-terminated string in a structure's string set.
/// Index 0 is "no string" by SMBIOS convention. The last string in the set may
/// not carry its own NUL within the slice handed here, so that case is checked
/// after the loop rather than assumed.
fn string_at(strings: &[u8], idx: u8) -> Option<&[u8]> {
    if idx == 0 {
        return None;
    }
    let mut n = 1u8;
    let mut start = 0usize;
    let mut k = 0usize;
    while k < strings.len() {
        if strings[k] == 0 {
            if n == idx {
                return Some(&strings[start..k]);
            }
            n += 1;
            start = k + 1;
        }
        k += 1;
    }
    if n == idx && start < strings.len() {
        return Some(&strings[start..]);
    }
    None
}

/// Walk SMBIOS structures for Type 1 (system) manufacturer + product and
/// Type 2 (baseboard) product. `bytes` is the structure table, which under
/// SMBIOS 3 is only bounded by a *maximum* size, so the type-127 end marker is
/// what stops the walk rather than the length.
fn parse(bytes: &[u8]) -> (Option<&[u8]>, Option<&[u8]>, Option<&[u8]>) {
    let (mut vendor, mut product, mut board) = (None, None, None);
    let mut i = 0usize;
    while i + 4 <= bytes.len() {
        let typ = bytes[i];
        let flen = bytes[i + 1] as usize;
        // type 127 is the end of table; a formatted area shorter than its
        // four-byte header, or one running past the buffer, is corruption.
        if typ == 127 || flen < 4 || i + flen > bytes.len() {
            break;
        }
        let formatted = &bytes[i..i + flen];
        // The string set follows the formatted area, ending at a double NUL.
        let str_start = i + flen;
        let mut j = str_start;
        while j + 1 < bytes.len() && !(bytes[j] == 0 && bytes[j + 1] == 0) {
            j += 1;
        }
        let strings = &bytes[str_start..j.min(bytes.len())];
        match typ {
            1 if flen > 5 => {
                vendor = vendor.or_else(|| string_at(strings, formatted[4]));
                product = product.or_else(|| string_at(strings, formatted[5]));
            }
            2 if flen > 5 => {
                board = board.or_else(|| string_at(strings, formatted[5]));
            }
            _ => {}
        }
        i = j + 2; // past the double NUL
    }
    (vendor, product, board)
}

#[inline]
unsafe fn ru16(p: *const u8) -> u16 {
    unsafe { (p as *const u16).read_unaligned() }
}
#[inline]
unsafe fn ru32(p: *const u8) -> u32 {
    unsafe { (p as *const u32).read_unaligned() }
}
#[inline]
unsafe fn ru64(p: *const u8) -> u64 {
    unsafe { (p as *const u64).read_unaligned() }
}

/// (structure-table address, length) from an SMBIOS entry point of either
/// generation, told apart by anchor. SMBIOS 2.1 carries a 32-bit table address
/// and an exact length; SMBIOS 3.0 a 64-bit address and a maximum size.
unsafe fn entry(ep: *const u8) -> Option<(u64, usize)> {
    let anchor = unsafe { core::slice::from_raw_parts(ep, 5) };
    if &anchor[..4] == b"_SM_" {
        let len = unsafe { ru16(ep.add(0x16)) } as usize;
        let addr = unsafe { ru32(ep.add(0x18)) } as u64;
        Some((addr, len))
    } else if anchor == b"_SM3_" {
        let max = unsafe { ru32(ep.add(0x0C)) } as usize;
        let addr = unsafe { ru64(ep.add(0x10)) };
        Some((addr, max))
    } else {
        None
    }
}

fn find(st: &SystemTable) -> Option<*const u8> {
    // Prefer the 3.x table: on a machine that publishes both it is the
    // authoritative one, and the 2.1 table is there for old consumers.
    for want in [&SMBIOS3_GUID, &SMBIOS_GUID] {
        for i in 0..st.number_of_table_entries {
            let e = unsafe { &*st.configuration_table.add(i) };
            if e.vendor_guid == *want {
                return Some(e.vendor_table as *const u8);
            }
        }
    }
    None
}

fn read_cpu() {
    // The brand string lives in extended leaves 0x80000002..=0x80000004, four
    // registers each, 48 bytes in all. Leaf 0x80000000 says whether they are
    // implemented; everything since the Pentium 4 carries them, but a part
    // that does not would otherwise hand back whatever the lower leaves put in
    // those registers.
    let max = crate::cpu::cpuid(0x8000_0000, 0)[0];
    if max < 0x8000_0004 {
        return;
    }
    let mut buf = [0u8; 48];
    for (n, leaf) in [0x8000_0002u32, 0x8000_0003, 0x8000_0004].iter().enumerate() {
        let r = crate::cpu::cpuid(*leaf, 0);
        for (w, val) in r.iter().enumerate() {
            let off = n * 16 + w * 4;
            buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
    }
    // NUL-terminated and, on many parts, padded on the left with spaces.
    let end = buf.iter().position(|&b| b == 0).unwrap_or(48);
    let info = unsafe { INFO.get() };
    store(&mut info.c, &mut info.cl, trim(&buf[..end]));
}

/// Read system identity from the firmware. Call once, before `ExitBootServices`.
pub fn init(st: &SystemTable) {
    if let Some(ep) = find(st) {
        unsafe {
            if let Some((addr, len)) = entry(ep) {
                // A plausible table: non-null, and not so large it is a bad
                // read rather than a real table. 1 MiB is far past any real
                // SMBIOS set and bounds a corrupt length.
                if addr != 0 && (1..(1 << 20)).contains(&len) {
                    let bytes = core::slice::from_raw_parts(addr as *const u8, len);
                    let (v, p, b) = parse(bytes);
                    let info = INFO.get();
                    if let Some(v) = v {
                        store(&mut info.v, &mut info.vl, trim(v));
                    }
                    if let Some(p) = p {
                        store(&mut info.p, &mut info.pl, trim(p));
                    }
                    if let Some(b) = b {
                        store(&mut info.b, &mut info.bl, trim(b));
                    }
                }
            }
        }
    }
    read_cpu();
}

fn field(buf: &[u8; N], len: usize) -> &'static str {
    // SAFETY: filled once at `init` from sanitised printable-ASCII bytes and
    // never mutated after, and `INFO` is static, so the bytes outlive any
    // caller. `store` guarantees the UTF-8 validity this skips checking.
    unsafe { core::str::from_utf8_unchecked(core::slice::from_raw_parts(buf.as_ptr(), len)) }
}

pub fn vendor() -> &'static str {
    let i = unsafe { INFO.get() };
    field(&i.v, i.vl)
}
pub fn product() -> &'static str {
    let i = unsafe { INFO.get() };
    field(&i.p, i.pl)
}
pub fn board() -> &'static str {
    let i = unsafe { INFO.get() };
    field(&i.b, i.bl)
}
pub fn cpu() -> &'static str {
    let i = unsafe { INFO.get() };
    field(&i.c, i.cl)
}

/// Assert the parser against a synthetic SMBIOS structure table, because the
/// real one cannot be relied on to exercise the edges: a machine with no
/// baseboard string, a last string without its own trailing NUL in the slice,
/// and index 0 meaning "no string" are all cases QEMU may or may not present.
pub fn selftest() -> bool {
    // Type 1: manufacturer=1, product=2, version=0 (none), serial=0.
    // Type 2: manufacturer=1 ("BoardCo"), product=2 ("MB-X1").
    // Type 127: end of table.
    let blob: &[u8] = &[
        // --- Type 1, formatted length 8 ---
        1, 8, 0x01, 0x00, 1, 2, 0, 0,
        b'A', b'C', b'M', b'E', 0, // string 1
        b'S', b'u', b'p', b'e', b'r', b'S', b'e', b'r', b'v', b'e', b'r', 0, // string 2
        0, // end of this structure's strings
        // --- Type 2, formatted length 8 ---
        2, 8, 0x02, 0x00, 1, 2, 0, 0,
        b'B', b'o', b'a', b'r', b'd', b'C', b'o', 0, // string 1
        b'M', b'B', b'-', b'X', b'1', 0, // string 2
        0, // end
        // --- Type 127, end of table ---
        127, 4, 0x03, 0x00, 0, 0,
    ];
    let (v, p, b) = parse(blob);
    let ok_v = v == Some(&b"ACME"[..]);
    let ok_p = p == Some(&b"SuperServer"[..]);
    let ok_b = b == Some(&b"MB-X1"[..]);

    // index 0 is "no string"
    let ok_zero = string_at(b"a\0b\0", 0).is_none();
    // a last string with no trailing NUL in the slice still resolves
    let ok_last = string_at(b"a\0bc", 2) == Some(&b"bc"[..]);
    // an index past the set is None rather than a wrong string
    let ok_over = string_at(b"a\0b\0", 5).is_none();

    // garbage / empty table yields nothing rather than a confident wrong answer
    let (gv, gp, gb) = parse(&[0u8; 3]);
    let ok_empty = gv.is_none() && gp.is_none() && gb.is_none();

    // sanitiser maps a non-printable byte to '?', keeping the accessors sound
    let mut buf = [0u8; N];
    let mut len = 0usize;
    store(&mut buf, &mut len, &[b'o', b'k', 0x07, b'!']);
    let ok_san = &buf[..len] == b"ok?!";

    ok_v && ok_p && ok_b && ok_zero && ok_last && ok_over && ok_empty && ok_san
}

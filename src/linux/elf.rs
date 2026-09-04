//! ELF64, only as much of it as a loader needs.
//!
//! The format has sections, symbols, relocations, a dynamic table and a
//! string table, and a *loader* needs none of them. What the kernel needs is
//! the program headers: which byte ranges of the file go at which addresses,
//! how big they are once there, and where to jump. Sections are for linkers.
//!
//! So this parses the header, walks `PT_LOAD`, and answers three questions the
//! rest of the port turns on:
//!
//!   - **Is it 64-bit x86 little-endian?** Everything else is a different
//!     machine and the honest answer is to say so rather than to load half of
//!     it and fault somewhere unrecognisable.
//!   - **Is it position-independent?** `ET_DYN` can be placed anywhere;
//!     `ET_EXEC` insists on its own addresses, which in a single address space
//!     means exactly one instance of it can ever exist. That constraint is a
//!     fact about this kernel rather than about the file, and the caller is
//!     the one who can weigh it.
//!   - **Does it want an interpreter?** A `PT_INTERP` means dynamically
//!     linked: the real entry point is `ld.so` and the loader's job is much
//!     larger than this. Reported rather than attempted, because a dynamic
//!     binary loaded as though it were static jumps into a PLT stub that has
//!     never been filled in, and the fault lands nowhere near the cause.
//!
//! Nothing here maps anything or trusts anything. Every field that indexes the
//! file is checked against the file's own length, because the input is a
//! *foreign binary* -- the first thing in this tree that was not written here
//! and not signed by the key compiled in.

use alloc::string::String;
use alloc::vec::Vec;

/// `\x7fELF`.
const MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];
/// `ELFCLASS64`.
const CLASS64: u8 = 2;
/// `ELFDATA2LSB`.
const LSB: u8 = 1;
/// `EM_X86_64`.
const X86_64: u16 = 0x3E;

/// An executable with fixed addresses.
pub const ET_EXEC: u16 = 2;
/// A shared object, which a position-independent executable also is.
pub const ET_DYN: u16 = 3;

const PT_LOAD: u32 = 1;
const PT_INTERP: u32 = 3;

pub const PF_X: u32 = 1;
pub const PF_W: u32 = 2;
pub const PF_R: u32 = 4;

/// One span of the file that has to exist in memory before the entry runs.
pub struct Segment {
    pub offset: usize,
    pub vaddr: u64,
    /// Bytes to copy from the file.
    pub filesz: usize,
    /// Bytes the segment occupies once loaded.
    ///
    /// Larger than `filesz` for anything with a `.bss`, and **the difference
    /// must be zeroed rather than left as whatever was there**. A C program
    /// whose globals start as the previous tenant's heap is a program that
    /// works until it does not.
    pub memsz: usize,
    pub flags: u32,
    pub align: u64,
}

impl Segment {
    pub fn writable(&self) -> bool {
        self.flags & PF_W != 0
    }
    pub fn executable(&self) -> bool {
        self.flags & PF_X != 0
    }
}

/// What a loader needs to know about a file, and nothing else.
pub struct Image {
    pub entry: u64,
    pub kind: u16,
    pub segments: Vec<Segment>,
    /// The interpreter this binary wants, if it wants one.
    pub interp: Option<String>,
}

impl Image {
    /// Whether it can be placed anywhere, or insists on its own addresses.
    pub fn relocatable(&self) -> bool {
        self.kind == ET_DYN
    }

    /// Whether running it means running a dynamic linker first.
    pub fn dynamic(&self) -> bool {
        self.interp.is_some()
    }

    /// The lowest and highest virtual addresses any segment claims.
    ///
    /// The span a single address space has to find room for. `None` when
    /// there are no loadable segments, which is a file that parses and cannot
    /// run -- a distinction worth keeping, because the two have very different
    /// causes.
    pub fn span(&self) -> Option<(u64, u64)> {
        let lo = self.segments.iter().map(|s| s.vaddr).min()?;
        let hi = self.segments.iter().map(|s| s.vaddr + s.memsz as u64).max()?;
        Some((lo, hi))
    }
}

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(at..at + 8)?.try_into().ok()?))
}

/// Read a file into the little a loader needs from it.
///
/// Every offset is bounds-checked against `bytes` rather than trusted, and a
/// failure names the field. "not an ELF" and "program headers past the end of
/// the file" are different accusations, and a loader that reports the first
/// for both sends the reader to look at the wrong thing.
pub fn parse(bytes: &[u8]) -> Result<Image, &'static str> {
    if bytes.len() < 64 {
        return Err("shorter than an ELF header");
    }
    if bytes[0..4] != MAGIC {
        return Err("not an ELF");
    }
    if bytes[4] != CLASS64 {
        return Err("not 64-bit");
    }
    if bytes[5] != LSB {
        return Err("not little-endian");
    }
    let kind = u16_at(bytes, 16).ok_or("truncated e_type")?;
    if u16_at(bytes, 18) != Some(X86_64) {
        return Err("not x86-64");
    }
    if kind != ET_EXEC && kind != ET_DYN {
        return Err("neither an executable nor a shared object");
    }
    let entry = u64_at(bytes, 24).ok_or("truncated e_entry")?;
    let phoff = u64_at(bytes, 32).ok_or("truncated e_phoff")? as usize;
    let phentsize = u16_at(bytes, 54).ok_or("truncated e_phentsize")? as usize;
    let phnum = u16_at(bytes, 56).ok_or("truncated e_phnum")? as usize;
    // A program header smaller than the fields read below would let a short
    // entry read into its neighbour and produce a segment nobody wrote.
    if phentsize < 56 {
        return Err("program header entries are too small to hold one");
    }
    // Computed with checked arithmetic: a `phnum` of 65535 against a large
    // `phoff` overflows on 64-bit only if the file claims something absurd,
    // and "absurd" is exactly what a hostile file claims.
    let end = phnum
        .checked_mul(phentsize)
        .and_then(|n| n.checked_add(phoff))
        .ok_or("program header table overflows")?;
    if end > bytes.len() {
        return Err("program headers past the end of the file");
    }

    let mut segments = Vec::new();
    let mut interp = None;
    for i in 0..phnum {
        let at = phoff + i * phentsize;
        let ty = u32_at(bytes, at).ok_or("truncated p_type")?;
        let flags = u32_at(bytes, at + 4).ok_or("truncated p_flags")?;
        let offset = u64_at(bytes, at + 8).ok_or("truncated p_offset")? as usize;
        let vaddr = u64_at(bytes, at + 16).ok_or("truncated p_vaddr")?;
        let filesz = u64_at(bytes, at + 32).ok_or("truncated p_filesz")? as usize;
        let memsz = u64_at(bytes, at + 40).ok_or("truncated p_memsz")? as usize;
        let align = u64_at(bytes, at + 48).ok_or("truncated p_align")?;

        if ty == PT_INTERP {
            let to = offset.checked_add(filesz).ok_or("interpreter path overflows")?;
            let raw = bytes.get(offset..to).ok_or("interpreter path past the end of the file")?;
            // The path is NUL-terminated inside its own segment.
            let cut = raw.iter().position(|b| *b == 0).unwrap_or(raw.len());
            interp = core::str::from_utf8(&raw[..cut]).ok().map(String::from);
            continue;
        }
        if ty != PT_LOAD {
            continue;
        }
        // A segment that claims more file than exists would copy whatever
        // follows the buffer into the guest's address space.
        let to = offset.checked_add(filesz).ok_or("segment overflows the file")?;
        if to > bytes.len() {
            return Err("segment past the end of the file");
        }
        // `memsz` under `filesz` is not merely odd -- it means the zero-fill
        // length below goes negative, and the subtraction that computes it is
        // where that would surface as a very large number.
        if memsz < filesz {
            return Err("a segment smaller in memory than on disk");
        }
        segments.push(Segment { offset, vaddr, filesz, memsz, flags, align });
    }
    Ok(Image { entry, kind, segments, interp })
}

/// What `diag linux` asks of the parser.
///
/// Every negative here is a file that a loader trusting its fields would map
/// something wrong from, so each one is built by taking a good header and
/// breaking exactly one field.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();

    // A minimal, valid ET_EXEC with one PT_LOAD, built by hand so the claims
    // below do not depend on a toolchain being installed.
    let good = |mutate: &dyn Fn(&mut Vec<u8>)| -> Vec<u8> {
        let mut b = alloc::vec![0u8; 64 + 56];
        b[0..4].copy_from_slice(&MAGIC);
        b[4] = CLASS64;
        b[5] = LSB;
        b[6] = 1;
        b[16..18].copy_from_slice(&ET_EXEC.to_le_bytes());
        b[18..20].copy_from_slice(&X86_64.to_le_bytes());
        b[24..32].copy_from_slice(&0x40_1000u64.to_le_bytes());
        b[32..40].copy_from_slice(&64u64.to_le_bytes());
        b[54..56].copy_from_slice(&56u16.to_le_bytes());
        b[56..58].copy_from_slice(&1u16.to_le_bytes());
        let ph = 64;
        b[ph..ph + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
        b[ph + 4..ph + 8].copy_from_slice(&(PF_R | PF_X).to_le_bytes());
        b[ph + 8..ph + 16].copy_from_slice(&0u64.to_le_bytes());
        b[ph + 16..ph + 24].copy_from_slice(&0x40_0000u64.to_le_bytes());
        b[ph + 32..ph + 40].copy_from_slice(&120u64.to_le_bytes());
        b[ph + 40..ph + 48].copy_from_slice(&8192u64.to_le_bytes());
        b[ph + 48..ph + 56].copy_from_slice(&4096u64.to_le_bytes());
        mutate(&mut b);
        b
    };

    let ok = good(&|_| {});
    let img = parse(&ok);
    out.push((
        "a minimal executable parses to one loadable segment and an entry",
        img.as_ref().is_ok_and(|i| {
            i.segments.len() == 1 && i.entry == 0x40_1000 && !i.dynamic() && !i.relocatable()
        }),
    ));
    out.push((
        "a segment with a .bss occupies more memory than it does file",
        img.as_ref().is_ok_and(|i| i.segments[0].memsz > i.segments[0].filesz),
    ));
    out.push((
        "the span covers every byte the segments claim",
        img.as_ref().ok().and_then(|i| i.span()) == Some((0x40_0000, 0x40_0000 + 8192)),
    ));

    // Each negative breaks exactly one field of the same good file.
    out.push(("a file that is not an ELF is refused", parse(&good(&|b| b[1] = b'X')).is_err()));
    out.push(("a 32-bit ELF is refused", parse(&good(&|b| b[4] = 1)).is_err()));
    out.push((
        "an ELF for another machine is refused",
        parse(&good(&|b| b[18..20].copy_from_slice(&0xB7u16.to_le_bytes()))).is_err(),
    ));
    out.push((
        "program headers past the end of the file are refused",
        parse(&good(&|b| b[56..58].copy_from_slice(&99u16.to_le_bytes()))).is_err(),
    ));
    out.push((
        "a segment claiming more file than exists is refused",
        parse(&good(&|b| b[64 + 32..64 + 40].copy_from_slice(&99_999u64.to_le_bytes()))).is_err(),
    ));
    out.push((
        "a segment smaller in memory than on disk is refused",
        parse(&good(&|b| b[64 + 40..64 + 48].copy_from_slice(&1u64.to_le_bytes()))).is_err(),
    ));
    out.push((
        "a program header entry too small to hold one is refused",
        parse(&good(&|b| b[54..56].copy_from_slice(&8u16.to_le_bytes()))).is_err(),
    ));
    out.push((
        "a truncated file is refused rather than read short",
        parse(&ok[..40]).is_err(),
    ));

    // The two facts the rest of the port turns on, which a loader that only
    // asked "did it parse" would carry straight into a fault.
    let pie = good(&|b| b[16..18].copy_from_slice(&ET_DYN.to_le_bytes()));
    out.push((
        "a position-independent executable says it can be placed anywhere",
        parse(&pie).is_ok_and(|i| i.relocatable()),
    ));
    out.push((
        "a fixed executable says it cannot",
        parse(&ok).is_ok_and(|i| !i.relocatable()),
    ));

    out
}

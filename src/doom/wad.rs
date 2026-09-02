// Ported from room4doom's `wad` crate, MIT licensed.
//   https://github.com/flukejones/room4doom  --  wad/src/wad.rs
//
// The format handling is theirs: the header layout, the sixteen-byte directory
// entry, and the rule that a lookup searches from the end so a later lump wins
// -- which is how a PWAD overrides an IWAD and is not obvious from the format
// alone.
//
// Three things are changed, and each is a property of running in a kernel
// rather than a preference:
//
//   * **Lumps borrow, they do not own.** The original stores `Vec<u8>` per
//     lump and copies the bytes out of the file. Here the whole WAD already
//     lives in the firmware's LoaderData pool for the life of the machine, so
//     a lump is an offset and a length into it. On a four-megabyte IWAD with
//     ~2,300 lumps that is the difference between one copy of the file and
//     two, and between 2,300 heap allocations and none.
//   * **Nothing panics.** The original does `expect("Invalid lump name")` on a
//     name that is not UTF-8, and slices `file[offset..offset + size]`
//     unchecked. Both are fine with an operating system underneath and fatal
//     here: there is no unwinder, so a malformed WAD would halt the machine
//     rather than be rejected. Every read is bounds-checked and every failure
//     is an `Error`.
//   * **Names are fixed, not `String`.** Eight bytes inline against a heap
//     allocation each, for something compared far more often than it is
//     printed.

use alloc::vec::Vec;

/// The eight bytes a WAD gives a lump, trimmed and made printable.
///
/// Fixed rather than `String` because there are thousands of them and they are
/// compared far more often than displayed. Sanitised at parse time so
/// `as_str` cannot fail: a byte outside printable ASCII becomes `?`, since a
/// lump name is for a human to read and a strange byte in one is not a reason
/// to refuse the file it names.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Name {
    raw: [u8; 8],
    len: u8,
}

impl Name {
    /// The eight bytes a *lump body* spells a name in -- a sector's floor
    /// texture, a sidedef's wall. Same encoding as a directory entry, so the
    /// same reader, exposed because the level lumps carry them too.
    pub fn from_lump(b: &[u8]) -> Name {
        Name::from_bytes(b)
    }

    fn from_bytes(b: &[u8]) -> Name {
        let mut raw = [0u8; 8];
        let mut len = 0usize;
        for i in 0..8 {
            let c = b.get(i).copied().unwrap_or(0);
            // A name is NUL-padded, and the padding is where it ends.
            if c == 0 {
                break;
            }
            raw[i] = if (0x20..0x7F).contains(&c) { c.to_ascii_uppercase() } else { b'?' };
            len = i + 1;
        }
        // Trailing spaces are padding too, and some tools emit them.
        while len > 0 && raw[len - 1] == b' ' {
            len -= 1;
        }
        Name { raw, len: len as u8 }
    }

    pub fn as_str(&self) -> &str {
        // Every byte was forced into printable ASCII above, so this cannot
        // fail. `unwrap_or` rather than `unwrap` because a panic here would be
        // a halt, and "" is a truthful rendering of a name we could not read.
        core::str::from_utf8(&self.raw[..self.len as usize]).unwrap_or("")
    }

    pub fn is(&self, s: &str) -> bool {
        let b = s.as_bytes();
        if b.len() != self.len as usize {
            return false;
        }
        b.iter().zip(self.raw.iter()).all(|(a, c)| a.to_ascii_uppercase() == *c)
    }
}

impl core::fmt::Display for Name {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One directory entry: a name and where its bytes are.
#[derive(Clone, Copy)]
pub struct Entry {
    pub name: Name,
    at: usize,
    len: usize,
}

impl Entry {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Which kind of WAD this is.
///
/// An IWAD is a complete game and a PWAD is a patch over one. The distinction
/// matters to a loader that stacks several; it is recorded here and acted on
/// nowhere yet.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Iwad,
    Pwad,
}

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Iwad => "IWAD",
            Kind::Pwad => "PWAD",
        }
    }
}

pub enum Error {
    /// Smaller than a header. Almost always a truncated copy.
    TooSmall,
    /// The first four bytes are neither `IWAD` nor `PWAD`.
    NotAWad([u8; 4]),
    /// The directory is not inside the file.
    DirectoryOutside { at: usize, count: usize, len: usize },
    /// A lump claims bytes past the end. Names the entry, because on a
    /// truncated download it is always the last few and saying which turns
    /// "corrupt" into "short by this much".
    LumpOutside { index: usize, name: Name, at: usize, len: usize },
    /// A lump's offset or size is negative.
    ///
    /// Its own variant rather than folding into the one above, because the
    /// two are different diagnoses and the message is the whole product here.
    /// Cast to `usize` first, -4 renders as 18446744073709551612, which reads
    /// as an absurdly large lump -- a truncated file -- when what it actually
    /// indicates is a byte order or a field width that is wrong. Saying
    /// "negative" points at the right half of the problem.
    NegativeLump { index: usize, name: Name, at: i32, len: i32 },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::TooSmall => f.write_str("shorter than a WAD header"),
            Error::NotAWad(m) => {
                f.write_str("not a WAD: magic is ")?;
                for c in m.iter() {
                    let c = if (0x20..0x7F).contains(c) { *c as char } else { '?' };
                    core::fmt::Write::write_char(f, c)?;
                }
                Ok(())
            }
            Error::DirectoryOutside { at, count, len } => write!(
                f,
                "directory of {} entries at {} runs past the file ({} bytes)",
                count, at, len
            ),
            Error::LumpOutside { index, name, at, len } => write!(
                f,
                "lump {} '{}' wants {} bytes at {}, past the end",
                index, name, len, at
            ),
            Error::NegativeLump { index, name, at, len } => write!(
                f,
                "lump {} '{}' offset or size is negative ({} at {}); byte-swapped or not a WAD",
                index, name, len, at
            ),
        }
    }
}

const HEADER: usize = 12;
const DIR_ENTRY: usize = 16;

/// A WAD, parsed but not copied.
pub struct Wad {
    bytes: &'static [u8],
    kind: Kind,
    lumps: Vec<Entry>,
}

impl Wad {
    /// Parse the directory of a WAD already in memory.
    ///
    /// `'static` is the honest lifetime rather than a convenience: these bytes
    /// come from a firmware pool nothing releases, and saying so lets a
    /// renderer hold a slice of a texture for as long as it likes without a
    /// lifetime threaded through every structure it owns -- which is one of
    /// the things that turns a port into a rewrite.
    pub fn parse(bytes: &'static [u8]) -> Result<Wad, Error> {
        if bytes.len() < HEADER {
            return Err(Error::TooSmall);
        }
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        let kind = match &magic {
            b"IWAD" => Kind::Iwad,
            b"PWAD" => Kind::Pwad,
            _ => return Err(Error::NotAWad(magic)),
        };
        let count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let at = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;

        // Checked before allocating, so a header claiming four billion lumps
        // asks for nothing rather than asking for 64 GB and taking the heap
        // with it.
        let need = count.checked_mul(DIR_ENTRY).and_then(|n| n.checked_add(at));
        match need {
            Some(end) if end <= bytes.len() => {}
            _ => {
                return Err(Error::DirectoryOutside { at, count, len: bytes.len() });
            }
        }

        let mut lumps = Vec::new();
        if lumps.try_reserve_exact(count).is_err() {
            return Err(Error::DirectoryOutside { at, count, len: bytes.len() });
        }
        for i in 0..count {
            let e = at + i * DIR_ENTRY;
            let pos = i32::from_le_bytes([bytes[e], bytes[e + 1], bytes[e + 2], bytes[e + 3]]);
            let size =
                i32::from_le_bytes([bytes[e + 4], bytes[e + 5], bytes[e + 6], bytes[e + 7]]);
            let name = Name::from_bytes(&bytes[e + 8..e + 16]);
            // Negative is not merely invalid, it is the shape a truncated or
            // byte-swapped file takes, so it is rejected as an offence rather
            // than cast into a huge unsigned number.
            if pos < 0 || size < 0 {
                return Err(Error::NegativeLump { index: i, name, at: pos, len: size });
            }
            let (pos, size) = (pos as usize, size as usize);
            match pos.checked_add(size) {
                Some(end) if end <= bytes.len() => {}
                _ => return Err(Error::LumpOutside { index: i, name, at: pos, len: size }),
            }
            lumps.push(Entry { name, at: pos, len: size });
        }
        Ok(Wad { bytes, kind, lumps })
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn len(&self) -> usize {
        self.lumps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lumps.is_empty()
    }

    /// The whole file, for anything that wants to know how big it was.
    pub fn size(&self) -> usize {
        self.bytes.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Entry> {
        self.lumps.iter()
    }

    /// The bytes of one lump. Always in range: the constructor rejected any
    /// entry that was not.
    pub fn data(&self, e: &Entry) -> &'static [u8] {
        &self.bytes[e.at..e.at + e.len]
    }

    /// A lump by name, **searched from the end**.
    ///
    /// Last match wins, which is the format's own override rule and the reason
    /// a PWAD loaded after an IWAD replaces its lumps rather than being
    /// ignored. It is not deducible from the file layout and it is the one
    /// piece of WAD semantics that is easy to get backwards.
    pub fn find(&self, name: &str) -> Option<&Entry> {
        self.lumps.iter().rev().find(|l| l.name.is(name))
    }

    /// The bytes of a lump by name.
    pub fn lump(&self, name: &str) -> Option<&'static [u8]> {
        self.find(name).map(|e| self.data(e))
    }

    /// The index of a lump, for anything that needs to read the lumps that
    /// follow it -- which is how a map is stored: a marker named `E1M1` and
    /// then ten unnamed-by-position lumps after it.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.lumps.iter().rposition(|l| l.name.is(name))
    }

    pub fn at(&self, i: usize) -> Option<&Entry> {
        self.lumps.get(i)
    }
}

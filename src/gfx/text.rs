//! Outline fonts: the faces, their kerning, and their licences.
//!
//! 1.4.1 replaces the 8x8 bitmap font for everything but the panic path and
//! early boot (where nothing may allocate) with outline fonts rasterised by
//! `fontdue` -- the one outside crate, and why it is allowed is the policy at
//! the top of Cargo.toml. This module is the floor of that: the three faces
//! compiled in, the kerning `tools/fontsubset.py` flattened out of GPOS, and the
//! licence texts the OFL requires to travel with the fonts. Layout, the glyph
//! atlas and drawing build on it (phase Q4).
//!
//! **Every face is parsed once and kept.** `Font::from_bytes` walks the whole
//! file and builds its tables; doing that per string would cost more than the
//! rasterising.

use alloc::vec::Vec;

use crate::sync::Spin;

/// Noto Sans, the interface face, in two weights.
const NOTO_REGULAR: &[u8] = include_bytes!("../../assets/fonts/NotoSans-Regular.ttf");
const NOTO_SEMIBOLD: &[u8] = include_bytes!("../../assets/fonts/NotoSans-SemiBold.ttf");
/// Cascadia Code, the terminal's, unmodified (its name is reserved).
const CASCADIA: &[u8] = include_bytes!("../../assets/fonts/CascadiaCode-Regular.ttf");

const KERN_REGULAR: &[u8] = include_bytes!("../../assets/fonts/NotoSans-Regular.kern");
const KERN_SEMIBOLD: &[u8] = include_bytes!("../../assets/fonts/NotoSans-SemiBold.kern");

/// The OFL, with each font's copyright line. Compiled in because the licence
/// must accompany every copy of the fonts and the kernel image *is* the copy;
/// `licences` prints them.
pub const OFL_NOTO: &str = include_str!("../../assets/fonts/OFL-NotoSans.txt");
pub const OFL_CASCADIA: &str = include_str!("../../assets/fonts/OFL-CascadiaCode.txt");

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Face {
    /// Noto Sans Regular: body text, labels.
    Ui,
    /// Noto Sans SemiBold: captions, headings, the focused thing.
    UiStrong,
    /// Cascadia Code: the terminal and anything fixed-width.
    Mono,
}

impl Face {
    fn bytes(self) -> &'static [u8] {
        match self {
            Face::Ui => NOTO_REGULAR,
            Face::UiStrong => NOTO_SEMIBOLD,
            Face::Mono => CASCADIA,
        }
    }

    fn kern_table(self) -> Option<&'static [u8]> {
        match self {
            Face::Ui => Some(KERN_REGULAR),
            Face::UiStrong => Some(KERN_SEMIBOLD),
            // A monospaced face is not kerned: every advance is the cell.
            Face::Mono => None,
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

const FACES: [Face; 3] = [Face::Ui, Face::UiStrong, Face::Mono];

struct Loaded {
    fonts: Vec<fontdue::Font>,
}

static LOADED: Spin<Option<Loaded>> = Spin::new(None);

/// Parse every face, once. Needs the heap. Answers false if a face did not
/// parse, which for a file whose digest is pinned means the build is broken.
pub fn init() -> bool {
    let mut l = LOADED.lock_irq();
    if l.is_some() {
        return true;
    }
    let mut fonts = Vec::new();
    for f in FACES {
        let settings = fontdue::FontSettings { scale: 40.0, ..fontdue::FontSettings::default() };
        match fontdue::Font::from_bytes(f.bytes(), settings) {
            Ok(font) => fonts.push(font),
            Err(_) => return false,
        }
    }
    *l = Some(Loaded { fonts });
    true
}

/// Run `f` with a parsed face, parsing them all first if nothing has yet.
pub fn with_face<R>(face: Face, f: impl FnOnce(&fontdue::Font) -> R) -> Option<R> {
    if LOADED.lock_irq().is_none() && !init() {
        return None;
    }
    let l = LOADED.lock_irq();
    l.as_ref().map(|l| f(&l.fonts[face.index()]))
}

/// The kerning between two characters, in thousandths of an em -- the table's
/// own units, since every face here is 1000 units per em -- or 0.
///
/// A binary search over `GLADOSK1` (the format is documented in
/// `tools/fontsubset.py`): sorted `(left, right)` pairs of six bytes each.
pub fn kern(face: Face, left: char, right: char) -> i16 {
    let Some(t) = face.kern_table() else { return 0 };
    let (Ok(l), Ok(r)) = (u16::try_from(left as u32), u16::try_from(right as u32)) else {
        return 0;
    };
    let Some(n) = table_len(t) else { return 0 };
    let key = (l as u32) << 16 | r as u32;
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = (lo + hi) / 2;
        let at = 16 + mid * 6;
        let a = u16::from_le_bytes([t[at], t[at + 1]]) as u32;
        let b = u16::from_le_bytes([t[at + 2], t[at + 3]]) as u32;
        let k = a << 16 | b;
        if k == key {
            return i16::from_le_bytes([t[at + 4], t[at + 5]]);
        }
        if k < key {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    0
}

/// The pair count of a `GLADOSK1` table, if it is one and its length agrees.
fn table_len(t: &[u8]) -> Option<usize> {
    if t.len() < 16 || &t[..8] != b"GLADOSK1" {
        return None;
    }
    let n = u32::from_le_bytes([t[12], t[13], t[14], t[15]]) as usize;
    (16 + n * 6 == t.len()).then_some(n)
}

/// `diag fonts`: the faces parse, rasterise, measure and kern.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    claim("all three faces parse", init());
    for f in FACES {
        let cov = with_face(f, |font| {
            let (m, bitmap) = font.rasterize('A', 16.0);
            (m.width, m.height, bitmap.iter().map(|&v| v as u32).sum::<u32>())
        });
        let good = matches!(cov, Some((w, h, s)) if w > 4 && h > 6 && s > 0);
        claim(
            match f {
                Face::Ui => "Noto Sans draws an 'A' with real coverage",
                Face::UiStrong => "Noto Sans SemiBold draws one too",
                Face::Mono => "and so does Cascadia Code",
            },
            good,
        );
    }
    // Every character a cell of the terminal may hold is in the mono face.
    let missing = with_face(Face::Mono, |font| {
        ['\u{2500}', '\u{2502}', '\u{250C}', '\u{2588}', '\u{2014}', 'e', '\u{00E9}']
            .iter()
            .filter(|&&c| font.lookup_glyph_index(c) == 0)
            .count()
    });
    claim("Cascadia Code has box drawing, blocks, dashes and accents", missing == Some(0));
    // The kerning values HarfBuzz gives the unsubset font (tools/fontsubset.py
    // checked 20,008 pairs against it); reading the same here proves the
    // embedded table and the search, not the flattening.
    claim("'AV' kerns -40", kern(Face::Ui, 'A', 'V') == -40);
    claim("'To' kerns -70", kern(Face::Ui, 'T', 'o') == -70);
    claim("'ab' does not kern", kern(Face::Ui, 'a', 'b') == 0);
    claim("the terminal's face is never kerned", kern(Face::Mono, 'A', 'V') == 0);
    claim("a character outside the table's plane is no pair", kern(Face::Ui, '\u{1F600}', 'A') == 0);
    claim("both licences are compiled in, with their copyright lines",
          OFL_NOTO.contains("Noto Project Authors") && OFL_CASCADIA.contains("Microsoft Corporation"));
    ok
}

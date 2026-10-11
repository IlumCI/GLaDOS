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
    // Asked as its own statement: in `lock().is_none() && !init()` the guard
    // lives to the end of the condition, and `init` takes the same lock.
    let loaded = LOADED.lock_irq().is_some();
    if !loaded && !init() {
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

// ---------------------------------------------------------------------------
// Layout and drawing (Q4).
//
// **Positions are kept to a third of a pixel, and the coverage there is
// exact.** A glyph drawn only at whole pixels sits up to half a pixel from
// where its advances put it, and at UI sizes that is visible as uneven
// spacing -- "rn" crowding into "m". fontdue offers no fractional origin, but
// it does rasterise at three times the horizontal resolution
// (`rasterize_indexed_subpixel`), and those columns are true area coverage of
// a third of a pixel each. Averaging three of them is therefore *exactly* the
// coverage of a whole pixel, at any of three phases: a cache entry per glyph
// per phase, and no resampling anywhere.
//
// The phase arithmetic has one trap, found reading fontdue rather than
// guessed at. Its 3x canvas places the outline's left edge at `fract(xmin)`
// *sub*-columns, not three times that, so the sub-column origin carries a
// `2 * fract(xmin)` term a naive `3 * pen` misses -- up to two thirds of a
// pixel, different per glyph, which reads as letters that wobble. `place`
// works it through; the claims check a stem moves by a third a phase.
//
// **Vertically, the baseline is snapped to a whole pixel** and nothing else
// is: the light hinting FreeType calls "vertical only". Horizontal snapping
// distorts letterforms; vertical snapping is what keeps a line of text from
// sitting on a blurred baseline.
// ---------------------------------------------------------------------------

use alloc::collections::BTreeMap;
use alloc::string::String;

use super::{Color, Framebuffer};

/// One rasterised glyph at one size and phase: coverage, and where it goes.
struct Cached {
    /// Rows above the baseline to the bitmap's top.
    top: i32,
    w: u32,
    h: u32,
    cov: Vec<u8>,
}

/// The atlas. Keyed `(face, quarter-pixels, glyph, phase)`; cleared whole when
/// it outgrows its budget rather than evicted piecemeal, because a clear is
/// one line that cannot be wrong and the working set of a desktop is a few
/// hundred glyphs -- refilling it costs a frame, once.
struct Atlas {
    map: BTreeMap<u64, Cached>,
    bytes: usize,
}

static ATLAS: Spin<Atlas> = Spin::new(Atlas { map: BTreeMap::new(), bytes: 0 });
const ATLAS_BUDGET: usize = 4 << 20;

static HITS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static MISSES: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static CLEARS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Atlas counters: hits, misses, entries, bytes, clears.
pub fn atlas_stats() -> (u64, u64, usize, usize, u64) {
    use core::sync::atomic::Ordering::Relaxed;
    let a = ATLAS.lock_irq();
    (HITS.load(Relaxed), MISSES.load(Relaxed), a.map.len(), a.bytes, CLEARS.load(Relaxed))
}

fn key(face: Face, px: f32, glyph: u16, phase: u32) -> u64 {
    let q = (px * 4.0 + 0.5) as u64 & 0xffff;
    (face.index() as u64) << 40 | q << 24 | (glyph as u64) << 8 | phase as u64
}

/// Rasterise `glyph` at phase `r` (0..3): pixel `i` of the result averages
/// sub-columns `3i - r .. 3i - r + 3` of fontdue's 3x bitmap.
fn rasterise(font: &fontdue::Font, glyph: u16, px: f32, r: u32) -> Cached {
    let (m, sub) = font.rasterize_indexed_subpixel(glyph, px);
    let sw = m.width * 3;
    let w = if m.width == 0 { 0 } else { m.width + (r > 0) as usize };
    let mut cov = alloc::vec![0u8; w * m.height];
    for y in 0..m.height {
        let row = &sub[y * sw..(y + 1) * sw];
        for i in 0..w {
            let mut sum = 0u32;
            for k in 0..3 {
                let j = (3 * i + k) as isize - r as isize;
                if j >= 0 && (j as usize) < sw {
                    sum += row[j as usize] as u32;
                }
            }
            cov[y * w + i] = ((sum + 1) / 3) as u8;
        }
    }
    Cached { top: m.ymin + m.height as i32, w: w as u32, h: m.height as u32, cov }
}

/// Where a glyph goes for a pen at `pen` (pixels, fractional): the whole-pixel
/// column its bitmap's column 0 lands on, and its phase. The left bearing is
/// already in that column -- it is the `3 floor(bx)` below.
///
/// Derivation: fontdue's sub-column s holds outline x = bx + (s - f)/3, where
/// bx is the glyph's scaled left bearing and f = fract(bx). Pixel X therefore
/// takes sub-columns from 3X - D, D = 3 pen + 3 floor(bx) + 2 f. Rounding D
/// to an integer d = 3q + r costs at most a sixth of a pixel; the bitmap for
/// phase r then starts at whole pixel q.
fn place(pen: f32, bx: f32) -> (i32, u32) {
    let fl = crate::ai::tensor::floorf(bx);
    let f = bx - fl;
    let d = crate::ai::tensor::floorf(3.0 * pen + 3.0 * fl + 2.0 * f + 0.5) as i32;
    let r = d.rem_euclid(3);
    ((d - r) / 3, r as u32)
}

/// Line metrics for a face at a size: ascent, descent (negative), line gap.
pub fn line(face: Face, px: f32) -> (f32, f32, f32) {
    with_face(face, |f| {
        f.horizontal_line_metrics(px).map(|m| (m.ascent, m.descent, m.line_gap))
    })
    .flatten()
    .unwrap_or((px * 0.8, -px * 0.2, 0.0))
}

/// The advance of every character of `s` in turn, kerning included, calling
/// `each(char, glyph index, pen before it, its scaled left bearing)`.
fn walk(font: &fontdue::Font, face: Face, px: f32, s: &str, mut each: impl FnMut(char, u16, f32, f32)) -> f32 {
    let em = font.units_per_em().max(1.0);
    let mut pen = 0.0f32;
    let mut prev: Option<char> = None;
    for c in s.chars() {
        if let Some(p) = prev {
            pen += kern(face, p, c) as f32 * px / em;
        }
        let g = font.lookup_glyph_index(c);
        let m = font.metrics_indexed(g, px);
        each(c, g, pen, m.bounds.xmin);
        pen += m.advance_width;
        prev = Some(c);
    }
    pen
}

/// The width of `s` set in `face` at `px` pixels, kerned.
pub fn measure(face: Face, px: f32, s: &str) -> f32 {
    with_face(face, |f| walk(f, face, px, s, |_, _, _, _| {})).unwrap_or(0.0)
}

/// `s`, or as much of it as fits in `max` pixels with an ellipsis after it.
pub fn ellipsize(face: Face, px: f32, s: &str, max: f32) -> String {
    if measure(face, px, s) <= max {
        return String::from(s);
    }
    let dots = measure(face, px, "\u{2026}");
    let mut out = String::new();
    let mut fits = 0usize;
    with_face(face, |f| {
        let mut ends = Vec::new();
        let total = walk(f, face, px, s, |_, _, pen, _| ends.push(pen));
        ends.push(total);
        // ends[i] is where character i starts; the prefix of i characters is
        // ends[i] wide (kerning against the dropped character excluded).
        for i in (0..ends.len()).rev() {
            if ends[i] + dots <= max {
                fits = i;
                break;
            }
        }
    });
    out.extend(s.chars().take(fits));
    // A trailing space before the ellipsis reads as a gap, not a word cut.
    while out.ends_with(' ') {
        out.pop();
    }
    out.push('\u{2026}');
    out
}

/// Gamma applied to coverage by the text colour's luminance.
///
/// Light text on a dark ground blended in sRGB looks thinner than its
/// outline, and dark on light heavier, because coverage is linear and the
/// encoding is not. Skia answers with a contrast table per luminance; this is
/// the same idea at four levels -- gamma 1.0 for black text down to 0.7 for
/// white -- and the figure is a first guess, to be judged by eye on the GF63's
/// panel, which is the only display that matters.
fn contrast_lut(c: Color) -> [u8; 256] {
    static LUTS: Spin<Option<[[u8; 256]; 4]>> = Spin::new(None);
    let lum = (c.r as u32 * 54 + c.g as u32 * 183 + c.b as u32 * 19) >> 8;
    let level = (lum * 4 / 256).min(3) as usize;
    let mut l = LUTS.lock_irq();
    let t = l.get_or_insert_with(|| {
        let mut t = [[0u8; 256]; 4];
        for (k, lut) in t.iter_mut().enumerate() {
            let gamma = 1.0 - 0.1 * k as f64;
            for (i, v) in lut.iter_mut().enumerate() {
                *v = (super::grad::pow(i as f64 / 255.0, gamma) * 255.0 + 0.5) as u8;
            }
            lut[0] = 0;
            lut[255] = 255;
        }
        t
    });
    t[level]
}

/// Draw `s` with its pen starting at `x` (fractional pixels) and its baseline
/// on row `baseline`. Answers the advance, so runs can be chained.
pub fn draw(fb: &Framebuffer, x: f32, baseline: i32, face: Face, px: f32, s: &str, color: Color) -> f32 {
    let raw = super::blend::opaque(fb.encode(color));
    let lut = contrast_lut(color);
    let mut row: Vec<u8> = Vec::new();
    let mut atlas = ATLAS.lock_irq();
    let adv = with_face(face, |font| {
        walk(font, face, px, s, |c, g, pen, bx| {
            if c == ' ' {
                return;
            }
            let (q, r) = place(x + pen, bx);
            let k = key(face, px, g, r);
            if !atlas.map.contains_key(&k) {
                MISSES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                let e = rasterise(font, g, px, r);
                if atlas.bytes + e.cov.len() > ATLAS_BUDGET {
                    atlas.map.clear();
                    atlas.bytes = 0;
                    CLEARS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                }
                atlas.bytes += e.cov.len();
                atlas.map.insert(k, e);
            } else {
                HITS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            }
            let e = &atlas.map[&k];
            let gx = q;
            let gy = baseline - e.top;
            for yy in 0..e.h as i32 {
                let py = gy + yy;
                if py < 0 || py >= fb.height() as i32 {
                    continue;
                }
                let src = &e.cov[(yy as u32 * e.w) as usize..((yy as u32 + 1) * e.w) as usize];
                // Clip on the left by hand: a pen can sit left of the screen.
                let skip = (-gx).max(0) as usize;
                if skip >= src.len() {
                    continue;
                }
                row.clear();
                row.extend(src[skip..].iter().map(|&v| lut[v as usize]));
                fb.over_mask_row((gx + skip as i32) as u32, py as u32, &row, raw);
            }
        })
    })
    .unwrap_or(0.0);
    adv
}

/// `video text`: sizes, weights, the terminal face, kerning, phases, ellipsis.
pub fn proof(fb: &Framebuffer) {
    let (w, h) = (fb.width(), fb.height());
    let k = (w as f32 / 960.0).min(h as f32 / 540.0);
    super::grad::vertical(fb, 0, 0, w, h / 2, &[(0.0, Color::new(14, 22, 40)), (1.0, Color::new(20, 50, 84))]);
    super::grad::vertical(fb, 0, h / 2, w, h - h / 2, &[(0.0, Color::new(246, 247, 250)), (1.0, Color::new(226, 230, 238))]);
    let light = Color::new(236, 240, 248);
    let dark = Color::new(24, 28, 36);
    let orange = Color::new(255, 140, 30);
    let mut y = 34.0 * k;
    for (i, size) in [10.0f32, 12.0, 14.0, 16.0, 20.0, 26.0].iter().enumerate() {
        let px = size * k;
        y += px * 1.3;
        let face = if i % 2 == 1 { Face::UiStrong } else { Face::Ui };
        let label = alloc::format!("{} px  The quick brown fox jumps over the lazy dog. AV To Wa 0123456789", size);
        draw(fb, 24.0 * k, y as i32, face, px, &label, light);
    }
    y += 20.0 * k;
    let mono = 13.0 * k;
    for line in ["glados> diag all   \u{2500}\u{2500}\u{2500} 82 suites \u{2500}\u{2500}\u{2500}  \u{2502} ok \u{2502}",
                 "fn main() { let caf\u{e9} = \"na\u{ef}ve \u{2014} r\u{e9}sum\u{e9}\"; } // 0O 1lI"] {
        y += mono * 1.4;
        draw(fb, 24.0 * k, y as i32, Face::Mono, mono, line, orange);
    }
    // The light half: dark text, and the same phrase at three phases.
    let mut y = h as f32 / 2.0 + 24.0 * k;
    for (i, size) in [11.0f32, 14.0, 18.0].iter().enumerate() {
        let px = size * k;
        y += px * 1.4;
        let face = if i == 1 { Face::UiStrong } else { Face::Ui };
        draw(fb, 24.0 * k, y as i32, face, px, "Settings \u{2014} Appearance and Personalization", dark);
    }
    let px = 14.0 * k;
    for p in 0..3 {
        y += px * 1.5;
        draw(fb, 24.0 * k + p as f32 / 3.0, y as i32, Face::Ui, px,
             "illuminati minimum rn m  (pen moved a third of a pixel per line)", dark);
    }
    y += px * 1.5;
    let cut = ellipsize(Face::Ui, px, "A label far too long for the box it has been given to live in", 260.0 * k);
    draw(fb, 24.0 * k, y as i32, Face::Ui, px, &cut, dark);
    // Kerning, on and off, side by side.
    let big = 44.0 * k;
    let kx = 520.0 * k;
    let ky = (h as f32 / 2.0 + 80.0 * k) as i32;
    draw(fb, kx, ky, Face::UiStrong, big, "AVATAR To Wa", dark);
    let mut pen = kx;
    for c in "AVATAR To Wa".chars() {
        let mut b = [0u8; 4];
        pen += draw(fb, pen, ky + (60.0 * k) as i32, Face::UiStrong, big, c.encode_utf8(&mut b), Color::new(170, 60, 40));
    }
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

    // Layout.
    claim("nothing is no width", measure(Face::Ui, 16.0, "") == 0.0);
    let (a, b, ab) = (measure(Face::Ui, 16.0, "a"), measure(Face::Ui, 16.0, "b"), measure(Face::Ui, 16.0, "ab"));
    claim("an unkerned pair is the sum of its advances", (ab - (a + b)).abs() < 1e-3);
    let (av, a1, v1) = (measure(Face::Ui, 20.0, "AV"), measure(Face::Ui, 20.0, "A"), measure(Face::Ui, 20.0, "V"));
    claim("'AV' at 20 px is 0.8 px narrower than A and V apart", ((a1 + v1 - av) - 0.8).abs() < 1e-3);
    let m10 = measure(Face::Mono, 16.0, "0123456789");
    claim("the terminal face is monospaced", (m10 - 10.0 * measure(Face::Mono, 16.0, "i")).abs() < 1e-3
        && (measure(Face::Mono, 16.0, "W") - measure(Face::Mono, 16.0, "i")).abs() < 1e-3);
    let (asc, desc, _) = line(Face::Ui, 20.0);
    claim("a line has an ascent above and a descent below", asc > 10.0 && desc < -2.0);

    // Ellipsis.
    let long = "A label far too long for the box";
    let cut = ellipsize(Face::Ui, 14.0, long, 120.0);
    claim("an ellipsized label fits, and says it was cut",
        measure(Face::Ui, 14.0, &cut) <= 120.0 && cut.ends_with('\u{2026}') && cut.len() < long.len());
    claim("one that already fits is left alone", ellipsize(Face::Ui, 14.0, "Short", 120.0) == "Short");
    claim("and a box too small for anything is just the ellipsis",
        ellipsize(Face::Ui, 14.0, long, 1.0) == "\u{2026}");

    // Drawing, into memory. A stem's centroid says where a glyph landed. In
    // black on white: black's contrast table is the identity, and any other
    // bends edge coverage, so neither a centroid nor an ink total would be
    // linear in position -- which is how this claim failed the first time.
    let (w, h) = (96u32, 48u32);
    let mut px: Vec<u32> = alloc::vec![0; (w * h) as usize];
    let fb = unsafe { Framebuffer::over_ram(px.as_mut_ptr() as u64, w, h, w, super::Format::Bgrx) };
    let mut ink = |x: f32, s: &str, face: Face| -> (f32, f32, f32, f32) {
        px.iter_mut().for_each(|p| *p = 0xffff_ffff);
        draw(&fb, x, 34, face, 24.0, s, Color::new(0, 0, 0));
        let (mut sum, mut sx, mut above, mut below) = (0f32, 0f32, 0f32, 0f32);
        for y in 0..h {
            for xx in 0..w {
                let v = (255 - (px[(y * w + xx) as usize] & 0xff)) as f32;
                sum += v;
                sx += v * (xx as f32 + 0.5);
                if y < 34 { above += v } else { below += v }
            }
        }
        (sum, sx / sum.max(1.0), above, below)
    };
    let (s0, c0, _, _) = ink(20.0, "l", Face::Ui);
    let (s1, c1, _, _) = ink(20.0 + 1.0 / 3.0, "l", Face::Ui);
    let (s2, c2, _, _) = ink(20.0 + 2.0 / 3.0, "l", Face::Ui);
    let (s3, c3, _, _) = ink(21.0, "l", Face::Ui);
    kprintln!("       stem centroid {:.3} {:.3} {:.3} {:.3}, ink {} {} {} {}", c0, c1, c2, c3, s0 as u32, s1 as u32, s2 as u32, s3 as u32);
    claim("a stem drawn a third of a pixel later lands a third later",
        ((c1 - c0) - 1.0 / 3.0).abs() < 0.06 && ((c2 - c1) - 1.0 / 3.0).abs() < 0.06 && ((c3 - c0) - 1.0).abs() < 0.02);
    claim("and carries the same ink at every phase",
        [s1, s2, s3].iter().all(|s| (s - s0).abs() / s0 < 0.01));
    let (_, _, xa, xb) = ink(10.0, "x", Face::Ui);
    let (_, _, ga, gb) = ink(10.0, "g", Face::Ui);
    claim("an 'x' sits on the baseline and a 'g' hangs below it", xb == 0.0 && xa > 0.0 && gb > 0.0 && ga > 0.0);
    let before = atlas_stats();
    let _ = ink(10.0, "x", Face::Ui);
    let after = atlas_stats();
    claim("drawing it again rasterises nothing", after.1 == before.1 && after.0 > before.0);
    let (sl, _, _, _) = ink(-200.0, "x", Face::Ui);
    claim("a pen left of the surface draws nothing and faults nothing", sl == 0.0);
    ok
}

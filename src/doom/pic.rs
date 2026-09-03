// Ported from room4doom (MIT, Luke Jones), `wad/src/types.rs` -- the patch
// column-post decoder and the TEXTURE1/PNAMES composite reader.
//   https://github.com/flukejones/room4doom
//
// The decoding is theirs, and one line of it is the reason this was ported
// rather than written from the specification:
//
//     if y <= top { top += y } else { top = y }
//
// A post's `topdelta` is normally an absolute row. But a patch taller than 254
// cannot express its lower posts that way, so the convention -- DeePsea's, and
// universal since -- is that a delta *not greater than the previous one* is
// relative to it. Every published description of the format that predates tall
// patches says "topdelta is the y offset", full stop. Writing this from the
// spec produces a decoder that is correct on every patch in DOOM and wrong on
// half the patches in anything made after 1997, which is the worst kind of
// wrong: it works until it does not, on somebody else's data.
//
// Changed here, for the reasons the WAD and level readers were: nothing
// panics, every offset is bounds-checked, and a column keeps its palette
// indices as `u8` rather than `u16` -- a palette index is a byte, and the
// wider type costs twice the memory for a texture atlas that is already the
// largest thing this port allocates.

use alloc::vec::Vec;

use super::wad::{Name, Wad};

/// One run of opaque pixels in a column.
pub struct Post {
    /// The row this run starts at.
    pub top: usize,
    pub pixels: Vec<u8>,
}

/// A picture: DOOM's patch format.
pub struct Patch {
    pub name: Name,
    pub width: usize,
    pub height: usize,
    pub left: i16,
    pub top: i16,
    /// One list of posts per column. A column with none is entirely
    /// see-through, which is normal -- it is how a grate or a railing is
    /// drawn.
    pub columns: Vec<Vec<Post>>,
}

const END: u8 = 0xFF;

fn le16(b: &[u8], at: usize) -> Option<i16> {
    Some(i16::from_le_bytes([*b.get(at)?, *b.get(at + 1)?]))
}

fn le32(b: &[u8], at: usize) -> Option<i32> {
    Some(i32::from_le_bytes([
        *b.get(at)?,
        *b.get(at + 1)?,
        *b.get(at + 2)?,
        *b.get(at + 3)?,
    ]))
}

impl Patch {
    pub fn parse(name: Name, d: &[u8]) -> Option<Patch> {
        let width = le16(d, 0)? as usize;
        let height = le16(d, 2)? as usize;
        let left = le16(d, 4)?;
        let top = le16(d, 6)?;
        // A patch is at most 4096 wide in every tool that has ever written
        // one; a header claiming more is a lump that is not a patch, and
        // reserving for it would be the allocation a malformed file wants.
        if width == 0 || height == 0 || width > 4096 || height > 4096 {
            return None;
        }

        // `width` is a 16-bit field clamped above, so `8 + x * 4` below cannot
        // overflow and the reservation cannot be a denial of service.
        let mut columns = Vec::new();
        columns.try_reserve_exact(width).ok()?;

        for x in 0..width {
            let mut at = le32(d, 8 + x * 4)? as usize;
            let mut posts = Vec::new();
            // The previous post's top, for the relative-delta rule above.
            let mut prev: i32 = -1;
            loop {
                let delta = *d.get(at)?;
                if delta == END {
                    break;
                }
                let cur = if prev >= 0 && (delta as i32) <= prev {
                    prev + delta as i32
                } else {
                    delta as i32
                };
                prev = cur;
                let len = *d.get(at + 1)? as usize;
                // Two pad bytes, one either side of the run. They are not
                // alignment -- the original reads one pixel beyond each end
                // when it smooths a column -- and a reader that omits them is
                // off by one from the first post onward, which shows as every
                // texture sheared by a row per post.
                let from = at + 3;
                let to = from.checked_add(len)?;
                let run = d.get(from..to)?;
                let mut pixels = Vec::new();
                pixels.try_reserve_exact(len).ok()?;
                pixels.extend_from_slice(run);
                // A post that starts past the bottom is dropped rather than
                // refused: it is what a slightly wrong editor emits, and one
                // stray run is not a reason to refuse a whole wall.
                if (cur as usize) < height {
                    posts.push(Post { top: cur as usize, pixels });
                }
                at = to + 1;
            }
            columns.push(posts);
        }
        Some(Patch { name, width, height, left, top, columns })
    }
}

/// One patch placed into a texture.
pub struct Placement {
    pub x: i16,
    pub y: i16,
    pub patch: usize,
}

/// A wall texture: a name, a size, and the patches pasted into it.
pub struct TexDef {
    pub name: Name,
    pub width: usize,
    pub height: usize,
    pub patches: Vec<Placement>,
}

/// Everything a level needs to draw a wall.
pub struct Pics {
    pub patch_names: Vec<Name>,
    pub textures: Vec<TexDef>,
    /// Composed textures, column-major: `columns[t][x]` is one column of
    /// `height` bytes, top to bottom.
    composed: Vec<Vec<Vec<u8>>>,
    /// Indices into `textures`, ordered by name.
    ///
    /// A lookup is on the frame path: a seg asks for up to three textures by
    /// name and a room has hundreds of segs, so a scan of the directory is
    /// hundreds of thousands of string comparisons a frame. DOOM's own answer
    /// was `R_InitTextures` building a hash table once; this is the same trade
    /// with a binary search, which for the registered game's 125 textures is
    /// seven comparisons instead of a hundred and twenty-five.
    order: Vec<u32>,
}

impl Pics {
    /// Read `PNAMES` and `TEXTURE1` (and `TEXTURE2`, which the registered
    /// game splits its list across), then compose every texture.
    ///
    /// Composed once at load rather than per frame. A texture is a handful of
    /// kilobytes and a wall column is read several times a frame from several
    /// distances; building it on demand is the trade a machine with no memory
    /// makes, and this one has 320 MiB.
    pub fn load(wad: &Wad) -> Option<Pics> {
        let pn = wad.lump("PNAMES")?;
        // Every count in this format is a signed 32- or 16-bit field, so a
        // corrupt one is not merely large -- as a `usize` it is around 1.8e19,
        // and an offset computed from it wraps rather than exceeding anything
        // a bounds check would notice. Clamping each count to what its own
        // lump has room for removes every one of those paths at the top,
        // which reads far better than a `checked_add` at each of the nine
        // places an index is formed.
        let room = |len: usize, header: usize, each: usize| len.saturating_sub(header) / each;
        let count = (le32(pn, 0)?.max(0) as usize).min(room(pn.len(), 4, 8));
        let mut patch_names = Vec::new();
        patch_names.try_reserve_exact(count).ok()?;
        for i in 0..count {
            let at = 4 + i * 8;
            patch_names.push(Name::from_lump(pn.get(at..at + 8)?));
        }

        let mut textures = Vec::new();
        for lump in ["TEXTURE1", "TEXTURE2"] {
            let Some(t) = wad.lump(lump) else { continue };
            let n = (le32(t, 0)?.max(0) as usize).min(room(t.len(), 4, 4));
            for i in 0..n {
                let off = le32(t, 4 + i * 4)?.max(0) as usize;
                // A `maptexture_t` is 22 bytes before its first patch.
                if off > t.len().saturating_sub(22) {
                    continue;
                }
                let name = Name::from_lump(t.get(off..off + 8)?);
                // Fields 8..12 are `masked` and 16..20 `columndirectory`,
                // neither read by any engine since 1993. Skipped by width
                // rather than by name, because landing mid-field on the patch
                // count gives a texture with tens of thousands of patches.
                let width = le16(t, off + 12)? as usize;
                let height = le16(t, off + 14)? as usize;
                let np = (le16(t, off + 20)?.max(0) as usize).min(room(t.len(), off + 22, 10));
                if width == 0 || height == 0 || width > 4096 || height > 4096 {
                    continue;
                }
                let mut patches = Vec::new();
                for j in 0..np {
                    let e = off + 22 + j * 10;
                    let x = le16(t, e)?;
                    let y = le16(t, e + 2)?;
                    let p = le16(t, e + 4)? as usize;
                    if p < patch_names.len() {
                        patches.push(Placement { x, y, patch: p });
                    }
                }
                textures.push(TexDef { name, width, height, patches });
            }
        }

        let mut order: Vec<u32> = (0..textures.len() as u32).collect();
        order.sort_unstable_by(|a, b| {
            textures[*a as usize].name.as_str().cmp(textures[*b as usize].name.as_str())
        });
        let mut pics = Pics { patch_names, textures, composed: Vec::new(), order };
        pics.compose(wad);
        Some(pics)
    }

    fn compose(&mut self, wad: &Wad) {
        for t in self.textures.iter() {
            // Index 0 is the transparent hole in every DOOM palette and is
            // what an uncovered part of a texture reads as. A texture with
            // gaps is legal -- it is how a two-sided middle texture makes a
            // fence -- and filling with anything else invents pixels.
            let mut cols: Vec<Vec<u8>> = alloc::vec![alloc::vec![0u8; t.height]; t.width];
            for pl in t.patches.iter() {
                let Some(pname) = self.patch_names.get(pl.patch) else { continue };
                let Some(bytes) = wad.lump(pname.as_str()) else { continue };
                let Some(p) = Patch::parse(*pname, bytes) else { continue };
                for (px, posts) in p.columns.iter().enumerate() {
                    let tx = pl.x as i32 + px as i32;
                    if tx < 0 || tx as usize >= t.width {
                        continue;
                    }
                    let col = &mut cols[tx as usize];
                    for post in posts.iter() {
                        for (i, v) in post.pixels.iter().enumerate() {
                            let ty = pl.y as i32 + post.top as i32 + i as i32;
                            if ty >= 0 && (ty as usize) < t.height {
                                col[ty as usize] = *v;
                            }
                        }
                    }
                }
            }
            self.composed.push(cols);
        }
    }

    /// Which texture wears that name, or nothing.
    ///
    /// Case-sensitive, like every DOOM engine: a lump name is stored
    /// upper-case and a sidedef spells it the same way, so folding case here
    /// would let two textures that a real engine tells apart collide.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        let mut lo = 0usize;
        let mut hi = self.order.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            let i = self.order[mid] as usize;
            match self.textures[i].name.as_str().cmp(name) {
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
                core::cmp::Ordering::Equal => return Some(i),
            }
        }
        None
    }

    pub fn def(&self, i: usize) -> Option<&TexDef> {
        self.textures.get(i)
    }

    /// One column of a texture, wrapped.
    ///
    /// Wrapped rather than clamped, because a wall wider than its texture
    /// repeats it -- that is how DOOM tiles -- and because `u` arrives from a
    /// projection that can be a little outside the wall at its very edge.
    pub fn column(&self, tex: usize, x: i32) -> Option<&[u8]> {
        let t = self.textures.get(tex)?;
        let cols = self.composed.get(tex)?;
        let w = t.width as i32;
        let i = x.rem_euclid(w) as usize;
        cols.get(i).map(|c| c.as_slice())
    }
}

/// Everything a frame needs that is not geometry.
///
/// One structure rather than three arguments because the three travel
/// together everywhere and two of them are optional: a WAD with no TEXTURE1
/// still has to draw, and so does one whose COLORMAP does not light.
pub struct Art<'a> {
    pub playpal: &'a [u8],
    pub colormap: Option<&'a [u8]>,
    pub pics: Option<&'a Pics>,
    pub flats: Option<&'a Flats>,
}

impl<'a> Art<'a> {
    /// The pixels of a sector's floor or ceiling picture.
    pub fn flat(&self, name: &Name) -> Option<&'static [u8]> {
        self.flats?.by_name(name)
    }
}

/// A flat is 64 by 64 and always exactly that.
///
/// Not a header anywhere -- the size is the format. A flat lump carries no
/// dimensions at all, which is why the only test for one is that it is 4096
/// bytes, and why a wrong lump read as a flat is a picture rather than an
/// error.
pub const FLAT_SIDE: usize = 64;
pub const FLAT_BYTES: usize = FLAT_SIDE * FLAT_SIDE;

/// The name a ceiling wears when it is not a ceiling.
///
/// DOOM has no sky flag: a sector whose ceiling names this is open to the sky,
/// and the renderer draws the sky texture through the hole instead of a
/// ceiling. It is a string comparison in the original too.
pub const SKY_FLAT: &str = "F_SKY1";

/// What one directory entry means to the flat namespace.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mark {
    /// A `*_START` marker: everything after it is in the namespace.
    Open,
    /// A `*_END` marker.
    Close,
    /// A lump the right size to be a flat.
    Flat,
    /// A marker opening or closing *some other* namespace.
    ForeignOpen,
    ForeignClose,
    Other,
}

/// The whole of the namespace rule, as a function of one entry.
///
/// Pulled out because it is four conditions that all have to be right at once
/// and none of them fails loudly:
///
/// * **Empty.** A marker is a zero-length lump. Without this a *flat* named
///   something ending in `_END` would close the namespace it is inside.
/// * **Begins with F.** `P_START` and `S_START` bracket patches and sprites,
///   and both end in `_START` too. A rule that missed this would pull every
///   sprite into the flat set -- and a sprite is a patch, so the ones that
///   happened to be 4096 bytes would be *kept*, as garbage floors.
/// * **Any `F*`, not just `F_`.** The registered game nests `F1_START` and
///   `F2_START` inside the outer pair, and a PWAD spells it `FF_START`. This
///   counts depth rather than matching one spelling, so all three work and a
///   nested pair does not close the outer one.
/// * **Exactly 4096 bytes.** A flat has no header at all, so its size is the
///   only test there is.
pub fn classify(name: &str, empty: bool, len: usize) -> Mark {
    if empty {
        let f = name.starts_with('F');
        if name.ends_with("_START") {
            return if f { Mark::Open } else { Mark::ForeignOpen };
        }
        if name.ends_with("_END") {
            return if f { Mark::Close } else { Mark::ForeignClose };
        }
    }
    if len == FLAT_BYTES {
        Mark::Flat
    } else {
        Mark::Other
    }
}

/// Floor and ceiling pictures.
///
/// Borrowed rather than copied. A `Wad` hands back `&'static [u8]` because the
/// file is in LoaderData and never moves, so a few hundred flats cost a few
/// hundred slice headers instead of a megabyte -- which is the opposite of the
/// trade `Pics` makes, and for the opposite reason: a wall texture is
/// *composed* from patches and has to be built somewhere, while a flat is
/// already exactly the bytes a renderer wants.
pub struct Flats {
    names: Vec<Name>,
    data: Vec<&'static [u8]>,
    /// Lumps the right size to be a flat that no namespace claimed.
    loose: Vec<(Name, &'static [u8])>,
    /// Whether the WAD declared a flat namespace at all.
    pub marked: bool,
}

impl Flats {
    /// Every flat in the WAD.
    ///
    /// Flats live in a *namespace* -- between `F_START` and `F_END` -- rather
    /// than being identifiable by anything in the lump, because there is
    /// nothing in the lump: 4096 bytes of pixels and no header. The registered
    /// game nests `F1_START`/`F2_START` pairs inside the outer one and a PWAD
    /// usually spells it `FF_START`, so this counts depth over any `F*_START`
    /// rather than matching one spelling.
    ///
    /// A WAD with no markers at all still works: `by_name` falls back to a
    /// direct lookup, which is what a PWAD that simply drops a flat beside its
    /// map needs. The fallback is second because a name can collide across
    /// namespaces -- a patch and a flat may share one, and only the markers
    /// tell them apart.
    pub fn load(wad: &Wad) -> Flats {
        let mut names = Vec::new();
        let mut data = Vec::new();
        let mut loose = Vec::new();
        let mut depth = 0usize;
        // How deep inside somebody else's namespace we are. A sprite is a
        // patch and a patch can be any size, so one that happens to be exactly
        // 4096 bytes would otherwise be adopted here as a stray flat -- which
        // is precisely the name collision the loose list's own doc warns
        // about, arriving from the one direction it did not guard.
        let mut foreign = 0usize;
        let mut marked = false;
        for i in 0..wad.len() {
            let Some(e) = wad.at(i) else { continue };
            match classify(e.name.as_str(), e.is_empty(), e.len()) {
                Mark::Open => {
                    depth += 1;
                    marked = true;
                }
                Mark::Close => depth = depth.saturating_sub(1),
                Mark::ForeignOpen => foreign += 1,
                Mark::ForeignClose => foreign = foreign.saturating_sub(1),
                Mark::Flat if depth > 0 => {
                    names.push(e.name);
                    data.push(wad.data(e));
                }
                // Right size, and somebody else has claimed it. Not a flat.
                Mark::Flat if foreign > 0 => {}
                // Right size, no namespace at all. Kept separately and
                // consulted only after the namespace, because 4096 bytes is
                // the whole of the test available -- a flat has no header --
                // and a name can legitimately belong to something else.
                Mark::Flat => loose.push((e.name, wad.data(e))),
                Mark::Other => {}
            }
        }
        Flats { names, data, loose, marked }
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    pub fn name_at(&self, i: usize) -> Option<&Name> {
        self.names.get(i)
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        // Searched from the end, like `Wad::find`, so a PWAD's flat replaces
        // the IWAD's of the same name rather than losing to it.
        self.names.iter().rposition(|n| n.is(name))
    }

    pub fn at(&self, i: usize) -> Option<&'static [u8]> {
        self.data.get(i).copied()
    }

    /// The pixels of a named flat, or nothing -- including for the sky, which
    /// is a name and not a picture.
    pub fn by_name(&self, name: &Name) -> Option<&'static [u8]> {
        if name.is(SKY_FLAT) {
            return None;
        }
        if let Some(i) = self.index_of(name.as_str()) {
            return self.data.get(i).copied();
        }
        self.loose.iter().rev().find(|(n, _)| *n == *name).map(|(_, d)| *d)
    }
}

/// How many light levels a DOOM `COLORMAP` holds before the two special ones
/// (invulnerability, then all black).
pub const LIGHT_LEVELS: usize = 32;

impl<'a> Art<'a> {
    /// The WAD's own `COLORMAP`, but only if it lights anything.
    ///
    /// A COLORMAP is *the* answer to shading indexed art -- it is a table the
    /// artist made, and picking nearest-neighbours out of the palette instead
    /// is a guess at what they meant. So it is preferred whenever there is
    /// one. But a generated WAD can carry an identity table (`tools/mkwad.py`
    /// shipped one for a while, deliberately), and a renderer that used it
    /// would draw every wall at full brightness at every distance -- which
    /// reads exactly like a lighting bug in the renderer. Comparing the
    /// brightest map against the darkest settles it in 256 bytes.
    pub fn lighting_colormap(&self) -> Option<&'a [u8]> {
        let cm = self.colormap?;
        if cm.len() < LIGHT_LEVELS * 256 {
            return None;
        }
        let dark = LIGHT_LEVELS - 1;
        let lights = (0..256).any(|i| cm[i] != cm[dark * 256 + i]);
        if lights {
            Some(cm)
        } else {
            None
        }
    }
}

/// One column of a patch, from `(topdelta, pixels)` pairs.
///
/// Written out byte by byte rather than by calling anything the decoder uses,
/// because an encoder sharing code with its decoder agrees with itself about
/// a mistake in both.
fn column_bytes(posts: &[(u8, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    for (delta, run) in posts {
        out.push(*delta);
        out.push(run.len() as u8);
        // The two pad bytes, deliberately *not* equal to the pixels beside
        // them, so a decoder that read them as picture is caught by the pixel
        // claim rather than passing on a lucky duplicate.
        out.push(0xAB);
        out.extend_from_slice(run);
        out.push(0xCD);
    }
    out.push(END);
    out
}

/// A whole patch lump from a list of encoded columns.
fn patch_bytes(w: usize, h: usize, cols: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(w as i16).to_le_bytes());
    out.extend_from_slice(&(h as i16).to_le_bytes());
    out.extend_from_slice(&0i16.to_le_bytes());
    out.extend_from_slice(&0i16.to_le_bytes());
    let mut at = 8 + 4 * w;
    for c in cols.iter() {
        out.extend_from_slice(&(at as i32).to_le_bytes());
        at += c.len();
    }
    for c in cols.iter() {
        out.extend_from_slice(c);
    }
    out
}

/// What `diag doom` asks, as claims rather than as printed lines.
///
/// Nothing in this tree may name the printing macro, so a check here answers
/// a name and a verdict and the caller does the talking. That is the same
/// bargain the WAD reader's `Error` type makes, and it is the reason both read
/// as well as they do. Nothing here unwraps either: there is no unwinder in
/// this kernel, so a `.expect()` in a selftest halts the machine instead of
/// reporting the failure it just found.
///
/// **What this covers and what it does not.** It covers `Patch::parse`, which
/// is where a decoding mistake produces a plausible picture instead of an
/// error, and the two pure decisions beside it. It does *not* cover the
/// TEXTURE1 field offsets or composition -- those need a whole WAD, and they
/// are checked end to end by `doom tex` against `tools/mkwad.py`, which
/// verifies the same file from the other side.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let n = Name::from_lump(b"TEST\0\0\0\0");

    // Two columns: one solid run, and one with a gap in the middle.
    let solid = column_bytes(&[(0, &[1, 2, 3, 4])]);
    let split = column_bytes(&[(0, &[9, 9]), (4, &[7, 7])]);
    let d = patch_bytes(2, 8, &[solid, split]);
    match Patch::parse(n, &d) {
        None => out.push(("a patch decodes at all", false)),
        Some(p) => {
            out.push(("a patch decodes at all", true));
            out.push(("with the width and height its header claims", p.width == 2 && p.height == 8));
            out.push(("one column per width", p.columns.len() == 2));
            let run = p.columns.first().and_then(|c| c.first());
            out.push((
                "a run keeps its pixels and skips both pad bytes",
                run.map(|r| r.pixels.as_slice()) == Some(&[1u8, 2, 3, 4][..])
                    && run.map(|r| r.top) == Some(0),
            ));
            let gap = p.columns.get(1);
            out.push(("a gap leaves two posts rather than one", gap.map(|c| c.len()) == Some(2)));
            out.push((
                "and the second starts where its delta says",
                gap.and_then(|c| c.get(1)).map(|r| r.top) == Some(4),
            ));
        }
    }

    // The tall-patch rule, which is the whole reason the decoder was ported
    // rather than written from the format description. A delta greater than
    // the last one is absolute; a delta not greater than it is *relative*,
    // because a patch over 254 tall cannot name row 300 in a byte. Every
    // published description of the format predates that convention and says
    // the field is simply the row.
    let tall = column_bytes(&[(200, &[1]), (50, &[2])]);
    let d = patch_bytes(1, 300, &[tall]);
    out.push((
        "a delta that does not rise is relative to the last (row 250, not 50)",
        Patch::parse(n, &d)
            .and_then(|p| p.columns.first().and_then(|c| c.get(1)).map(|r| r.top))
            == Some(250),
    ));

    let rising = column_bytes(&[(10, &[1]), (100, &[2])]);
    let d = patch_bytes(1, 200, &[rising]);
    out.push((
        "and a delta that does rise is still absolute",
        Patch::parse(n, &d)
            .and_then(|p| p.columns.first().and_then(|c| c.get(1)).map(|r| r.top))
            == Some(100),
    ));

    // A run that starts below the picture is dropped rather than refused: it
    // is what a slightly wrong editor emits, and one stray post is not a
    // reason to lose a wall.
    let over = column_bytes(&[(0, &[1]), (250, &[2])]);
    let d = patch_bytes(1, 8, &[over]);
    out.push((
        "a post past the bottom is dropped, not kept",
        Patch::parse(n, &d).map(|p| p.columns[0].len()) == Some(1),
    ));

    // Refusals. A lump cut short must not be read past its end, and a header
    // claiming no picture is not a picture.
    let full = patch_bytes(2, 8, &[column_bytes(&[(0, &[1, 2])]), column_bytes(&[(0, &[3])])]);
    out.push((
        "a truncated lump is refused rather than read past",
        (1..full.len()).all(|k| Patch::parse(n, &full[..k]).is_none()),
    ));
    out.push(("a zero-width header is refused", Patch::parse(n, &patch_bytes(0, 8, &[])).is_none()));
    out.push(("so is an empty lump", Patch::parse(n, &[]).is_none()));

    // A column offset pointing outside the lump is the corruption that reads
    // as a picture rather than as damage: whatever bytes it lands on decode
    // as perfectly valid posts.
    let mut wild = full.clone();
    wild[8..12].copy_from_slice(&1_000_000i32.to_le_bytes());
    out.push(("a column offset outside the lump is refused", Patch::parse(n, &wild).is_none()));

    // The COLORMAP test, which decides whether a WAD's own shading table is
    // used or one is built from the palette. An identity table is a legal
    // lump that lights nothing, and a renderer trusting it draws every wall at
    // full brightness -- which reads as a bug in the renderer.
    let pal = [0u8; 768];
    let ident: Vec<u8> = (0..34).flat_map(|_| 0..=255u8).collect();
    let art = Art { playpal: &pal, colormap: Some(&ident), pics: None, flats: None };
    out.push(("an identity COLORMAP is not used for lighting", art.lighting_colormap().is_none()));
    let mut lit = ident.clone();
    lit[31 * 256] = 7;
    let art = Art { playpal: &pal, colormap: Some(&lit), pics: None, flats: None };
    out.push(("one that darkens anything is used", art.lighting_colormap().is_some()));
    let art = Art { playpal: &pal, colormap: Some(&ident[..100]), pics: None, flats: None };
    out.push(("a COLORMAP too short to hold 32 maps is refused", art.lighting_colormap().is_none()));

    // The flat namespace. Four conditions that all have to hold at once, and
    // the one that matters most is the third: `S_START` brackets sprites, a
    // sprite is a patch, and a patch that happens to be 4096 bytes would be
    // adopted as a floor by a rule that only looked for `_START`.
    out.push((
        "F_START opens the flat namespace and F_END closes it",
        classify("F_START", true, 0) == Mark::Open && classify("F_END", true, 0) == Mark::Close,
    ));
    out.push((
        "so do the nested and PWAD spellings",
        classify("F1_START", true, 0) == Mark::Open
            && classify("FF_START", true, 0) == Mark::Open
            && classify("F2_END", true, 0) == Mark::Close,
    ));
    out.push((
        "a sprite or patch marker opens somebody else's namespace instead",
        classify("S_START", true, 0) == Mark::ForeignOpen
            && classify("P_START", true, 0) == Mark::ForeignOpen
            && classify("P_END", true, 0) == Mark::ForeignClose,
    ));
    out.push((
        "a marker must be empty, so a flat cannot close its own namespace",
        classify("FWATER_END", false, FLAT_BYTES) == Mark::Flat,
    ));
    out.push((
        "and only 4096 bytes is a flat",
        classify("FLOOR4_8", false, FLAT_BYTES) == Mark::Flat
            && classify("FLOOR4_8", false, FLAT_BYTES - 1) == Mark::Other
            && classify("FLOOR4_8", false, FLAT_BYTES + 1) == Mark::Other,
    ));

    // The sprite claims moved to `sprite::checks`, where they belong now that
    // that module has its own. They were here because it did not.

    // `floor_i`, because a texture coordinate wraps and the negative side of
    // the seam is exactly where a truncating cast is wrong.
    use super::math::floor_i;
    out.push((
        "floor_i rounds down on both sides of zero",
        floor_i(2.7) == 2 && floor_i(-0.5) == -1 && floor_i(-1.0) == -1 && floor_i(-1.2) == -2,
    ));

    out
}

//! Things, drawn as billboards.
//!
//! A sprite is a patch -- the same column-and-post encoding `pic.rs` already
//! decodes for wall textures -- so nothing here re-reads a picture. What is
//! here is the three things a wall never needs: finding a sprite by the
//! convention its *name* encodes, deciding which sprite a map's thing wants,
//! and clipping a billboard against geometry that was drawn before it.
//!
//! ### The name is the directory
//!
//! There is no table in a WAD saying which lumps are frames of what. A sprite
//! lump is named `NNNNFR`: four characters of sprite name, a frame letter from
//! `A`, and a rotation digit. Rotation `0` means *this picture from every
//! angle*, which is what every item and every piece of scenery uses, and 1 to
//! 8 are the eight facings a monster has. Two frames can share one lump
//! through an eight-character name -- `SARGA1D1` is frame A rotation 1 and
//! also frame D rotation 1 mirrored -- and that is not handled here.
//!
//! ### Why the mapping table is small and says so
//!
//! DOOM's `mobjinfo` is several hundred entries of state machine, of which the
//! only field a renderer needs is the first sprite. Copying the table would be
//! copying a game's content, and deriving it is not possible -- a doomednum
//! means what id decided it means. So this carries the ones a map is actually
//! built from and draws nothing for a number it does not know, which is
//! visible as a missing object rather than as a wrong one.

use alloc::vec::Vec;

use super::level::Level;
use super::pic::Patch;
use super::wad::{Name, Wad};

/// A doomednum, and the sprite a thing of that kind wears.
///
/// Frame A rotation 0 in every case, because everything here is scenery or an
/// item and those have exactly one picture. A monster would need its facing,
/// which is the angle from the viewer to the thing minus the thing's own --
/// and needs the eight rotations to exist in the WAD to be worth computing.
const KINDS: &[(i16, &str)] = &[
    // Scenery, which is what a map is decorated with.
    (2035, "BAR1"), // exploding barrel
    (2028, "COLU"), // floor lamp
    (34, "CAND"),   // candle
    (35, "CBRA"),   // candelabra
    (54, "TRE2"),   // large tree
    (43, "TRE1"),   // burnt tree
    (2014, "BON1"), // health bonus
    (2015, "BON2"), // armour bonus
    (2011, "STIM"), // stimpack
    (2012, "MEDI"), // medikit
    (2018, "ARM1"), // armour
    (2019, "ARM2"), // megaarmour
    (2007, "CLIP"), // clip
    (2008, "SHEL"), // shells
    (2010, "ROCK"), // rocket
    (2047, "CELL"), // cell
    (2048, "AMMO"), // box of bullets
    (2049, "SBOX"), // box of shells
    (2046, "BROK"), // box of rockets
    (17, "CELP"),   // cell pack
    (8, "BPAK"),    // backpack
    (2001, "SHOT"), // shotgun
    (2002, "MGUN"), // chaingun
    (2003, "LAUN"), // rocket launcher
    (2004, "PLAS"), // plasma rifle
    (2005, "CSAW"), // chainsaw
    (2006, "BFUG"), // BFG
    (2013, "SOUL"), // soulsphere
    (2022, "PINV"), // invulnerability
    (2023, "PSTR"), // berserk
    (2024, "PINS"), // invisibility
    (2025, "SUIT"), // radiation suit
    (2026, "PMAP"), // computer map
    (2045, "PVIS"), // light amplification
    (5, "BKEY"),    // blue key card
    (6, "YKEY"),    // yellow key card
    (13, "RKEY"),   // red key card
    // Monsters. Frame A rotation 0 is wrong for these -- they have eight
    // rotations and no rotation-0 lump at all -- so they resolve to nothing
    // and draw nothing until facings are implemented. Listed anyway, because
    // the absence is then a missing picture rather than a missing table row.
    (3004, "POSS"), // zombieman
    (9, "SPOS"),    // shotgun guy
    (3001, "TROO"), // imp
    (3002, "SARG"), // demon
    (3005, "HEAD"), // cacodemon
    (3006, "SKUL"), // lost soul
    (3003, "BOSS"), // baron of hell
];

/// Things that are map data rather than objects: player and deathmatch
/// starts, and the teleport destination. They have no sprite and never had.
fn invisible(kind: i16) -> bool {
    matches!(kind, 1..=4 | 11 | 14 | 88 | 89)
}

pub fn sprite_for(kind: i16) -> Option<&'static str> {
    if invisible(kind) {
        return None;
    }
    KINDS.iter().find(|(k, _)| *k == kind).map(|(_, n)| *n)
}

/// Every sprite lump in the WAD, by the name it is filed under.
pub struct Sprites {
    names: Vec<Name>,
    data: Vec<&'static [u8]>,
    pub marked: bool,
}

impl Sprites {
    pub fn load(wad: &Wad) -> Sprites {
        let mut names = Vec::new();
        let mut data = Vec::new();
        let mut depth = 0usize;
        let mut marked = false;
        for i in 0..wad.len() {
            let Some(e) = wad.at(i) else { continue };
            let n = e.name.as_str();
            // The sprite namespace, `S_START`/`S_END` or a PWAD's `SS_START`.
            // Same shape as the flat rule and separate from it for the reason
            // that rule exists: the two namespaces overlap in nothing but
            // their punctuation, and a lump has to be in the right one.
            if e.is_empty() && n.starts_with('S') {
                if n.ends_with("_START") {
                    depth += 1;
                    marked = true;
                    continue;
                }
                if n.ends_with("_END") {
                    depth = depth.saturating_sub(1);
                    continue;
                }
            }
            if depth > 0 && !e.is_empty() {
                names.push(e.name);
                data.push(wad.data(e));
            }
        }
        Sprites { names, data, marked }
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
        self.names.iter().rposition(|n| n.is(name))
    }

    pub fn bytes(&self, i: usize) -> Option<&'static [u8]> {
        self.data.get(i).copied()
    }

    /// Frame `A`, rotation 0, of a sprite by its four-character name.
    ///
    /// Rotation 0 only, and that is the whole of what this draws. Anything
    /// with eight facings has no rotation-0 lump, so it resolves to nothing --
    /// which is a monster that does not appear rather than a monster drawn
    /// facing the wrong way, and the first is much easier to notice.
    pub fn still(&self, name: &str) -> Option<usize> {
        let mut buf = [0u8; 8];
        let b = name.as_bytes();
        if b.len() != 4 {
            return None;
        }
        buf[..4].copy_from_slice(b);
        buf[4] = b'A';
        buf[5] = b'0';
        let full = core::str::from_utf8(&buf[..6]).ok()?;
        self.index_of(full)
    }
}

/// One thing, decoded and ready to place.
pub struct Billboard {
    pub patch: Patch,
    /// Where it stands.
    pub x: f32,
    pub y: f32,
    /// The floor it stands on.
    pub z: f32,
    /// The light of the sector it is in, so it is lit by the room rather than
    /// by itself -- an object at full brightness in a dark room reads as a
    /// sprite pasted on top of the picture, which is exactly what it is.
    pub light: i16,
}

/// Decode every drawable thing on the level, once.
///
/// Once rather than per frame: a patch decode allocates a vector per column,
/// and a map has hundreds of things. The same trade `Pics` makes, for the same
/// reason, and it is why this returns owned `Patch`es rather than indices.
pub fn collect(lv: &Level, sprites: &Sprites) -> Vec<Billboard> {
    let mut out = Vec::new();
    for t in lv.things.iter() {
        let Some(name) = sprite_for(t.kind) else { continue };
        let Some(i) = sprites.still(name) else { continue };
        let Some(bytes) = sprites.bytes(i) else { continue };
        let Some(nm) = sprites.name_at(i).copied() else { continue };
        let Some(patch) = Patch::parse(nm, bytes) else { continue };
        // The floor of the sector it is standing in. A thing whose sector
        // cannot be found is dropped rather than placed at zero, which in a
        // map with a raised floor would bury it.
        let Some(sector) = lv.sector_at(t.x as i32, t.y as i32) else { continue };
        out.push(Billboard {
            patch,
            x: t.x as f32,
            y: t.y as f32,
            z: sector.floor as f32,
            light: sector.light,
        });
    }
    out
}

/// How many things on this level would draw, and how many were skipped for
/// each of the two reasons. For a caller that wants to say so.
pub fn census(lv: &Level, sprites: &Sprites) -> (usize, usize, usize) {
    let (mut drawn, mut no_kind, mut no_lump) = (0, 0, 0);
    for t in lv.things.iter() {
        match sprite_for(t.kind) {
            None => {
                if !invisible(t.kind) {
                    no_kind += 1;
                }
            }
            Some(n) => match sprites.still(n) {
                Some(_) => drawn += 1,
                None => no_lump += 1,
            },
        }
    }
    (drawn, no_kind, no_lump)
}

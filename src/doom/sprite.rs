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
use super::math;
use super::pic::Patch;
use super::wad::{Name, Wad};

/// A doomednum, and the sprite a thing of that kind wears.
///
/// Frame `A` in every case. Whether that frame has one picture or eight is a
/// property of the WAD rather than of this table -- an item has a rotation-0
/// lump and a monster has eight facings -- and `Frame` finds out by looking.
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
    // Monsters, which have eight facings and no rotation-0 lump. They draw
    // now: `Frame` decodes all eight and `pick` chooses by bearing.
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

    /// Frame `A` at one of the eight facings, and whether that lump is the
    /// mirror of the one wanted.
    ///
    /// A sprite lump can serve two frames at once through an eight-character
    /// name: `SARGA1D1` is frame A rotation 1, *and* frame D rotation 1 drawn
    /// mirrored. id did that because a monster is symmetric about its facing,
    /// so the picture of it turned 45 degrees left is the picture of it turned
    /// 45 degrees right flipped -- which halves the art for the four
    /// off-axis pairs. A reader that only matched the first pair would find
    /// four of the eight facings and drop the rest.
    ///
    /// A scan rather than a lookup. The list is every sprite lump in the WAD
    /// -- 853 in FreeDoom -- and this runs eight times per *sprite kind* when
    /// a level loads, not per thing and never per frame.
    pub fn facing(&self, name: &str, rot: u8) -> Option<(usize, bool)> {
        let want = name.as_bytes();
        if want.len() != 4 || !(1..=8).contains(&rot) {
            return None;
        }
        let digit = b'0' + rot;
        for (i, n) in self.names.iter().enumerate() {
            let s = n.as_str().as_bytes();
            if s.len() < 6 || s[..4] != want[..4] {
                continue;
            }
            if s[4] == b'A' && s[5] == digit {
                return Some((i, false));
            }
            if s.len() == 8 && s[6] == b'A' && s[7] == digit {
                return Some((i, true));
            }
        }
        None
    }
}

/// Every picture frame `A` of one sprite has.
///
/// A frame is either one picture seen from every angle or eight facings, and
/// which it is comes from the WAD rather than from any table: a rotation-0
/// lump means the first, its absence means the second. Decoded once per
/// *kind* and shared by every thing wearing it, because a map has hundreds of
/// things and twenty sorts of them -- decoding per thing would multiply the
/// work by fifteen and the memory with it.
pub struct Frame {
    /// One picture for every angle.
    all: Option<Patch>,
    /// The eight facings, and whether each lump is mirrored. Index 0 is the
    /// front, which is rotation *1* in a lump name.
    rot: Vec<Option<(Patch, bool)>>,
}

impl Frame {
    pub fn load(sprites: &Sprites, name: &str) -> Option<Frame> {
        let one = |i: usize| -> Option<Patch> {
            Patch::parse(sprites.name_at(i).copied()?, sprites.bytes(i)?)
        };
        let all = sprites.still(name).and_then(one);
        let mut rot = Vec::new();
        for r in 1..=8u8 {
            rot.push(
                sprites
                    .facing(name, r)
                    .and_then(|(i, flip)| Some((one(i)?, flip))),
            );
        }
        if all.is_none() && rot.iter().all(|r| r.is_none()) {
            return None;
        }
        Some(Frame { all, rot })
    }

    /// Whether this frame turns at all, for a caller that wants to say so.
    pub fn turns(&self) -> bool {
        self.all.is_none()
    }

    /// One facing by index, 0 being the front. For looking at all eight.
    pub fn at(&self, rot: usize) -> Option<(&Patch, bool)> {
        if let Some(p) = self.all.as_ref() {
            return Some((p, false));
        }
        self.rot.get(rot).and_then(|o| o.as_ref()).map(|(p, f)| (p, *f))
    }

    /// How many of the eight the WAD actually stores a lump for, and how many
    /// of those are mirrors. `SARGA2A8` is one lump serving two facings, so a
    /// monster with five lumps has eight facings and three mirrors.
    pub fn found(&self) -> (usize, usize) {
        let have = self.rot.iter().filter(|r| r.is_some()).count();
        let mirrored = self.rot.iter().filter(|r| matches!(r, Some((_, true)))).count();
        (have, mirrored)
    }

    /// Which picture the viewer sees, and whether to draw it mirrored.
    ///
    /// `to_thing` is the bearing from the eye to the thing and `facing` is the
    /// way the thing is pointing, both in radians. DOOM's own expression is
    ///
    /// ```text
    /// rot = (ang - thing->angle + (unsigned)(ANG45/2)*9) >> 29
    /// ```
    ///
    /// and the constant is what makes it work: `ANG45/2 * 9` is 202.5 degrees,
    /// which is 180 to turn "where the viewer is" into "which way the thing
    /// shows", plus 22.5 to put the *centre* of the front facing at the
    /// boundary-free middle of its bucket rather than on the edge. Without the
    /// half-bucket the sprite would flip between two facings whenever the
    /// viewer stood exactly in front of it.
    pub fn pick(&self, to_thing: f32, facing: f32) -> Option<(&Patch, bool)> {
        if let Some(p) = self.all.as_ref() {
            return Some((p, false));
        }
        let eighth = math::TAU / 8.0;
        let mut a = to_thing - facing + eighth * 4.5;
        a -= math::TAU * math::floor_i(a / math::TAU) as f32;
        let want = ((a / eighth) as usize).min(7);
        // Outward from the facing wanted, so a WAD missing one draws its
        // nearest neighbour rather than nothing. Searching one direction only
        // would bias every gap the same way round, which on a monster reads as
        // it snapping to face you.
        for k in 0..8usize {
            let step = k.div_ceil(2);
            let j = if k % 2 == 0 { want + step } else { want + 8 - step };
            if let Some((p, f)) = self.rot[j % 8].as_ref() {
                return Some((p, *f));
            }
        }
        None
    }
}

/// One thing, placed and pointing.
pub struct Billboard {
    /// Which entry of `Things::art` this wears.
    pub art: usize,
    /// The way it is pointing, in radians -- which for anything with eight
    /// facings decides which of them the viewer sees.
    pub angle: f32,
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
/// The things of a level, and the pictures they share.
pub struct Things {
    /// One per distinct sprite kind on the level.
    pub art: Vec<Frame>,
    pub items: Vec<Billboard>,
}

impl Things {
    /// Nothing to draw, for a WAD with no sprite namespace at all.
    pub fn none() -> Things {
        Things { art: Vec::new(), items: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// How many of the kinds on this level turn to face the viewer.
    pub fn turning(&self) -> usize {
        self.art.iter().filter(|f| f.turns()).count()
    }
}

pub fn collect(lv: &Level, sprites: &Sprites) -> Things {
    let mut art: Vec<Frame> = Vec::new();
    // Which `art` index each sprite name took, so a second imp shares the
    // first one's eight decoded facings.
    let mut seen: Vec<(&'static str, usize)> = Vec::new();
    let mut items = Vec::new();
    for t in lv.things.iter() {
        let Some(name) = sprite_for(t.kind) else { continue };
        let idx = match seen.iter().find(|(n, _)| *n == name) {
            Some((_, i)) => *i,
            None => {
                let Some(f) = Frame::load(sprites, name) else { continue };
                art.push(f);
                seen.push((name, art.len() - 1));
                art.len() - 1
            }
        };
        // The floor of the sector it is standing in. A thing whose sector
        // cannot be found is dropped rather than placed at zero, which in a
        // map with a raised floor would bury it.
        let Some(sector) = lv.sector_at(t.x as i32, t.y as i32) else { continue };
        items.push(Billboard {
            art: idx,
            angle: math::deg_to_rad(t.angle),
            x: t.x as f32,
            y: t.y as f32,
            z: sector.floor as f32,
            light: sector.light,
        });
    }
    Things { art, items }
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
            // A kind draws if it has *any* picture: a rotation-0 lump, or
            // at least one of the eight facings.
            Some(n) => match Frame::load(sprites, n) {
                Some(_) => drawn += 1,
                None => no_lump += 1,
            },
        }
    }
    (drawn, no_kind, no_lump)
}

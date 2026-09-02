// Ported from room4doom (MIT, Luke Jones) -- the record layouts and the rule
// that a map's lumps follow its marker by position.
//   https://github.com/flukejones/room4doom
//
// Changed here, for the same reasons the WAD reader was: nothing panics, every
// index is checked before it is stored rather than when it is used, and the
// structures borrow the file instead of copying it where they can.

use alloc::vec::Vec;

use super::wad::{Name, Wad};

/// A map's lumps follow its marker **in the directory**, not by name.
///
/// This is the single thing most easily got wrong about the format, and it is
/// silent when it is: a WAD with thirty-six maps has thirty-six lumps called
/// `THINGS`, so looking one up by name gives whichever the lookup rule
/// happens to favour -- for `Wad::find`, which searches from the end so a PWAD
/// can override an IWAD, that is always the *last* map's. Every map would load
/// with E3M9's monsters in it.
///
/// So: find the marker's index, then walk forward taking lumps by name only
/// until something that is not a map lump. Walking forward rather than
/// assuming a fixed order because the order is a convention, not a rule --
/// Hexen inserts `BEHAVIOR`, node builders add `GL_*`, and some editors emit
/// `REJECT` and `BLOCKMAP` in the other order.
const MAP_LUMPS: [&str; 11] = [
    "THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SEGS", "SSECTORS", "NODES", "SECTORS",
    "REJECT", "BLOCKMAP", "BEHAVIOR",
];

const THING: usize = 10;
const LINEDEF: usize = 14;
const SIDEDEF: usize = 30;
const VERTEX: usize = 4;
const SEG: usize = 12;
const SSECTOR: usize = 4;
const NODE: usize = 28;
const SECTOR: usize = 26;

/// A sidedef or child that is deliberately absent.
pub const NONE: u16 = 0xFFFF;
/// Set in a node's child to say "this is a subsector, not another node".
pub const SUBSECTOR_BIT: u16 = 0x8000;

pub enum Error {
    NoSuchMap,
    /// A lump the map cannot do without is not there.
    Missing(&'static str),
    /// A lump is not a whole number of records, which means the file is
    /// truncated or the record size is not what we think it is.
    BadSize { lump: &'static str, bytes: usize, record: usize },
    /// Something names something that does not exist. Checked at load rather
    /// than at use: a renderer that indexes a vertex per column would find out
    /// mid-frame, and in this kernel finding out mid-frame is a halt.
    Dangling { what: &'static str, index: usize, names: &'static str, at: usize, of: usize },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::NoSuchMap => f.write_str("no such map in this WAD"),
            Error::Missing(l) => write!(f, "the map has no {} lump", l),
            Error::BadSize { lump, bytes, record } => write!(
                f,
                "{} is {} bytes, not a multiple of {}",
                lump, bytes, record
            ),
            Error::Dangling { what, index, names, at, of } => write!(
                f,
                "{} {} names {} {} of {}",
                what, index, names, at, of
            ),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Vertex {
    pub x: i16,
    pub y: i16,
}

#[derive(Clone, Copy)]
pub struct Thing {
    pub x: i16,
    pub y: i16,
    /// Degrees, with 0 east and 90 north.
    pub angle: i16,
    pub kind: i16,
    pub flags: i16,
}

#[derive(Clone, Copy)]
pub struct LineDef {
    pub v1: u16,
    pub v2: u16,
    pub flags: u16,
    pub special: u16,
    pub tag: u16,
    pub right: u16,
    pub left: u16,
}

impl LineDef {
    /// A wall with a sidedef on both faces, which is what a doorway or a step
    /// between two sectors is. The flag exists too (bit 2) but the sidedefs
    /// are the truth: an editor can leave the flag wrong.
    pub fn two_sided(&self) -> bool {
        self.left != NONE
    }
}

#[derive(Clone, Copy)]
pub struct SideDef {
    pub x_off: i16,
    pub y_off: i16,
    pub upper: Name,
    pub lower: Name,
    pub middle: Name,
    pub sector: u16,
}

#[derive(Clone, Copy)]
pub struct Seg {
    pub v1: u16,
    pub v2: u16,
    /// A binary angle: a full turn is 65536 and east is zero.
    pub angle: u16,
    pub linedef: u16,
    /// 0 if the seg runs the same way as its linedef, 1 if reversed.
    pub side: u16,
    pub offset: i16,
}

#[derive(Clone, Copy)]
pub struct SubSector {
    pub count: u16,
    pub first: u16,
}

#[derive(Clone, Copy)]
pub struct Node {
    pub x: i16,
    pub y: i16,
    pub dx: i16,
    pub dy: i16,
    /// `(top, bottom, left, right)` -- y before x, and top before bottom,
    /// which is the opposite order to the one the words usually come in.
    pub right_box: [i16; 4],
    pub left_box: [i16; 4],
    pub right: u16,
    pub left: u16,
}

impl Node {
    /// Which side of this node's partition a point falls on: 0 right, 1 left.
    ///
    /// This is DOOM's own test and the arithmetic is load-bearing. Getting the
    /// comparison the wrong way round mirrors the level, which does not look
    /// like an arithmetic mistake -- it looks like a renderer bug.
    pub fn side_of(&self, px: i32, py: i32) -> usize {
        let dx = px - self.x as i32;
        let dy = py - self.y as i32;
        if (self.dy as i32) * dx > dy * (self.dx as i32) {
            0
        } else {
            1
        }
    }
}

#[derive(Clone, Copy)]
pub struct Sector {
    pub floor: i16,
    pub ceiling: i16,
    pub floor_pic: Name,
    pub ceiling_pic: Name,
    pub light: i16,
    pub special: i16,
    pub tag: i16,
}

pub struct Level {
    pub name: Name,
    pub things: Vec<Thing>,
    pub vertexes: Vec<Vertex>,
    pub linedefs: Vec<LineDef>,
    pub sidedefs: Vec<SideDef>,
    pub segs: Vec<Seg>,
    pub subsectors: Vec<SubSector>,
    pub nodes: Vec<Node>,
    pub sectors: Vec<Sector>,
}

fn le16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn le16i(b: &[u8], at: usize) -> i16 {
    i16::from_le_bytes([b[at], b[at + 1]])
}

fn records(
    lump: &'static str,
    bytes: Option<&'static [u8]>,
    size: usize,
) -> Result<(&'static [u8], usize), Error> {
    let b = bytes.ok_or(Error::Missing(lump))?;
    if b.len() % size != 0 {
        return Err(Error::BadSize { lump, bytes: b.len(), record: size });
    }
    Ok((b, b.len() / size))
}

impl Level {
    /// Load one map by its marker name.
    pub fn load(wad: &Wad, map: &str) -> Result<Level, Error> {
        let start = wad.index_of(map).ok_or(Error::NoSuchMap)?;
        let name = wad.at(start).ok_or(Error::NoSuchMap)?.name;

        // Walk forward from the marker, taking only what a map is made of and
        // stopping at the first lump that is not. The stop is what keeps this
        // from wandering into the next map: a marker is empty and its name is
        // not in `MAP_LUMPS`, so the loop ends there on its own.
        let mut find = |want: &str| -> Option<&'static [u8]> {
            let mut i = start + 1;
            while let Some(e) = wad.at(i) {
                if !MAP_LUMPS.iter().any(|m| e.name.is(m)) {
                    break;
                }
                if e.name.is(want) {
                    return Some(wad.data(e));
                }
                i += 1;
            }
            None
        };

        let (tb, nthings) = records("THINGS", find("THINGS"), THING)?;
        let (vb, nverts) = records("VERTEXES", find("VERTEXES"), VERTEX)?;
        let (lb, nlines) = records("LINEDEFS", find("LINEDEFS"), LINEDEF)?;
        let (sb, nsides) = records("SIDEDEFS", find("SIDEDEFS"), SIDEDEF)?;
        let (gb, nsegs) = records("SEGS", find("SEGS"), SEG)?;
        let (ub, nsubs) = records("SSECTORS", find("SSECTORS"), SSECTOR)?;
        let (nb, nnodes) = records("NODES", find("NODES"), NODE)?;
        let (cb, nsectors) = records("SECTORS", find("SECTORS"), SECTOR)?;

        let mut things = Vec::new();
        for i in 0..nthings {
            let e = i * THING;
            things.push(Thing {
                x: le16i(tb, e),
                y: le16i(tb, e + 2),
                angle: le16i(tb, e + 4),
                kind: le16i(tb, e + 6),
                flags: le16i(tb, e + 8),
            });
        }

        let mut vertexes = Vec::new();
        for i in 0..nverts {
            let e = i * VERTEX;
            vertexes.push(Vertex { x: le16i(vb, e), y: le16i(vb, e + 2) });
        }

        let mut sectors = Vec::new();
        for i in 0..nsectors {
            let e = i * SECTOR;
            sectors.push(Sector {
                floor: le16i(cb, e),
                ceiling: le16i(cb, e + 2),
                floor_pic: Name::from_lump(&cb[e + 4..e + 12]),
                ceiling_pic: Name::from_lump(&cb[e + 12..e + 20]),
                light: le16i(cb, e + 20),
                special: le16i(cb, e + 22),
                tag: le16i(cb, e + 24),
            });
        }

        let mut sidedefs = Vec::new();
        for i in 0..nsides {
            let e = i * SIDEDEF;
            let sector = le16(sb, e + 28);
            if sector as usize >= nsectors {
                return Err(Error::Dangling {
                    what: "sidedef",
                    index: i,
                    names: "sector",
                    at: sector as usize,
                    of: nsectors,
                });
            }
            sidedefs.push(SideDef {
                x_off: le16i(sb, e),
                y_off: le16i(sb, e + 2),
                upper: Name::from_lump(&sb[e + 4..e + 12]),
                lower: Name::from_lump(&sb[e + 12..e + 20]),
                middle: Name::from_lump(&sb[e + 20..e + 28]),
                sector,
            });
        }

        let mut linedefs = Vec::new();
        for i in 0..nlines {
            let e = i * LINEDEF;
            let (v1, v2) = (le16(lb, e), le16(lb, e + 2));
            let (right, left) = (le16(lb, e + 10), le16(lb, e + 12));
            for v in [v1, v2] {
                if v as usize >= nverts {
                    return Err(Error::Dangling {
                        what: "linedef",
                        index: i,
                        names: "vertex",
                        at: v as usize,
                        of: nverts,
                    });
                }
            }
            for s in [right, left] {
                if s != NONE && s as usize >= nsides {
                    return Err(Error::Dangling {
                        what: "linedef",
                        index: i,
                        names: "sidedef",
                        at: s as usize,
                        of: nsides,
                    });
                }
            }
            linedefs.push(LineDef {
                v1,
                v2,
                flags: le16(lb, e + 4),
                special: le16(lb, e + 6),
                tag: le16(lb, e + 8),
                right,
                left,
            });
        }

        let mut segs = Vec::new();
        for i in 0..nsegs {
            let e = i * SEG;
            let (v1, v2, li) = (le16(gb, e), le16(gb, e + 2), le16(gb, e + 6));
            for v in [v1, v2] {
                if v as usize >= nverts {
                    return Err(Error::Dangling {
                        what: "seg",
                        index: i,
                        names: "vertex",
                        at: v as usize,
                        of: nverts,
                    });
                }
            }
            // A miniseg -- one a nodebuilder emits along a partition rather
            // than along a wall -- carries `NONE` here and is legal.
            if li != NONE && li as usize >= nlines {
                return Err(Error::Dangling {
                    what: "seg",
                    index: i,
                    names: "linedef",
                    at: li as usize,
                    of: nlines,
                });
            }
            segs.push(Seg {
                v1,
                v2,
                angle: le16(gb, e + 4),
                linedef: li,
                side: le16(gb, e + 8),
                offset: le16i(gb, e + 10),
            });
        }

        let mut subsectors = Vec::new();
        for i in 0..nsubs {
            let e = i * SSECTOR;
            let (count, first) = (le16(ub, e), le16(ub, e + 2));
            if first as usize + count as usize > nsegs {
                return Err(Error::Dangling {
                    what: "subsector",
                    index: i,
                    names: "segs ending at",
                    at: first as usize + count as usize,
                    of: nsegs,
                });
            }
            subsectors.push(SubSector { count, first });
        }

        let mut nodes = Vec::new();
        for i in 0..nnodes {
            let e = i * NODE;
            let mut bb = |at: usize| {
                [le16i(nb, at), le16i(nb, at + 2), le16i(nb, at + 4), le16i(nb, at + 6)]
            };
            let right_box = bb(e + 8);
            let left_box = bb(e + 16);
            let (right, left) = (le16(nb, e + 24), le16(nb, e + 26));
            for c in [right, left] {
                let ok = if c & SUBSECTOR_BIT != 0 {
                    ((c & !SUBSECTOR_BIT) as usize) < nsubs
                } else {
                    (c as usize) < nnodes
                };
                if !ok {
                    return Err(Error::Dangling {
                        what: "node",
                        index: i,
                        names: if c & SUBSECTOR_BIT != 0 { "subsector" } else { "node" },
                        at: (c & !SUBSECTOR_BIT) as usize,
                        of: if c & SUBSECTOR_BIT != 0 { nsubs } else { nnodes },
                    });
                }
            }
            nodes.push(Node {
                x: le16i(nb, e),
                y: le16i(nb, e + 2),
                dx: le16i(nb, e + 4),
                dy: le16i(nb, e + 6),
                right_box,
                left_box,
                right,
                left,
            });
        }

        Ok(Level {
            name,
            things,
            vertexes,
            linedefs,
            sidedefs,
            segs,
            subsectors,
            nodes,
            sectors,
        })
    }

    /// Where the player starts, if the map says. Thing type 1 is player one.
    pub fn player_start(&self) -> Option<&Thing> {
        self.things.iter().find(|t| t.kind == 1)
    }

    /// The root of the BSP tree.
    ///
    /// The last node, which is a convention rather than a field -- a
    /// nodebuilder emits children before parents, so the tree's root is
    /// whatever it wrote last. A map small enough to be a single convex space
    /// has no nodes at all, and then the root *is* subsector zero; vanilla
    /// gets this wrong by computing `numnodes - 1` on an unsigned zero, and
    /// saying so here is cheaper than rediscovering it.
    pub fn root(&self) -> u16 {
        if self.nodes.is_empty() {
            SUBSECTOR_BIT
        } else {
            (self.nodes.len() - 1) as u16
        }
    }

    /// The sector a point stands in, by way of the subsector containing it.
    ///
    /// A subsector does not record its sector -- the format leaves it implied
    /// by the segs, every one of which faces into it -- so this takes the
    /// front sector of the first seg. A miniseg has no linedef and therefore
    /// no sector, so it is skipped rather than trusted.
    pub fn sector_at(&self, x: i32, y: i32) -> Option<&Sector> {
        let ss = self.subsector_at(x, y)?;
        for i in 0..ss.count as usize {
            let seg = self.segs.get(ss.first as usize + i)?;
            if seg.linedef == NONE {
                continue;
            }
            let line = self.linedefs.get(seg.linedef as usize)?;
            let side = if seg.side == 0 { line.right } else { line.left };
            if side == NONE {
                continue;
            }
            let sd = self.sidedefs.get(side as usize)?;
            return self.sectors.get(sd.sector as usize);
        }
        None
    }

    /// Walk the tree to the subsector containing a point.
    pub fn subsector_at(&self, x: i32, y: i32) -> Option<&SubSector> {
        let mut n = self.root();
        let mut guard = 0;
        while n & SUBSECTOR_BIT == 0 {
            // A malformed tree can be a cycle, and a renderer that followed
            // one would hang the machine with no message. The bound is the
            // node count: a well-formed descent visits each at most once.
            guard += 1;
            if guard > self.nodes.len() {
                return None;
            }
            let node = self.nodes.get(n as usize)?;
            n = if node.side_of(x, y) == 0 { node.right } else { node.left };
        }
        self.subsectors.get((n & !SUBSECTOR_BIT) as usize)
    }
}

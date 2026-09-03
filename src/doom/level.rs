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
    /// Which linedefs touch each sector, by index.
    ///
    /// The map stores the arrow the other way -- a linedef names its sidedefs
    /// and a sidedef names its sector -- and every `EV_Do*` needs to walk it
    /// backwards: "the lowest ceiling among this sector's neighbours" is the
    /// height a door opens to, and there is no way to ask it without knowing
    /// which lines bound the sector. Built once at load rather than scanned
    /// per query, because a door asks on the tic it opens and a map has
    /// thousands of lines.
    pub sector_lines: Vec<Vec<u16>>,
    /// Which sectors carry each non-zero tag.
    ///
    /// A linedef's special names a *tag*, not a sector, and several sectors
    /// can share one -- that is how a switch opens four doors at once. Vanilla
    /// scans the sector list per activation; this is the same answer without
    /// the scan.
    ///
    /// Note the tag types disagree in the format: a linedef's is `u16` and a
    /// sector's is `i16`. Everything here compares as `i32`.
    pub tagged: Vec<(i32, Vec<u16>)>,
    /// Whether something has ended the level.
    ///
    /// A flag rather than an immediate stop, because the exit fires from
    /// inside a special and the run has to unwind to the loop that owns the
    /// screen before it can hand it back.
    pub exited: bool,
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

        let sector_lines = build_sector_lines(&linedefs, &sidedefs, sectors.len());
        let tagged = build_tag_index(&sectors);
        Ok(Level {
            name,
            sector_lines,
            tagged,
            exited: false,
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
    /// Every sector carrying this tag.
    ///
    /// Empty for tag 0, deliberately: zero means "no tag" and a special that
    /// asked for it would otherwise operate on every untagged sector in the
    /// map at once.
    pub fn tagged(&self, tag: i32) -> &[u16] {
        if tag == 0 {
            return &[];
        }
        self.tagged
            .iter()
            .find(|(t, _)| *t == tag)
            .map(|(_, v)| v.as_slice())
            .unwrap_or(&[])
    }

    /// The sector on the other side of a line from this one, if there is one.
    fn across(&self, line: u16, from: usize) -> Option<usize> {
        let l = self.linedefs.get(line as usize)?;
        let side = |i: u16| -> Option<usize> {
            if i == NONE {
                return None;
            }
            self.sidedefs.get(i as usize).map(|sd| sd.sector as usize)
        };
        let (a, b) = (side(l.right), side(l.left));
        match (a, b) {
            (Some(x), Some(y)) if x == from => Some(y),
            (Some(x), Some(y)) if y == from => Some(x),
            _ => None,
        }
    }

    /// The lowest ceiling among a sector's neighbours, and the height a door
    /// in it opens to.
    ///
    /// DOOM's `P_FindLowestCeilingSurrounding` less four units, which is the
    /// gap a door leaves under the lintel so its top texture has something to
    /// draw on. A sector with no neighbour at all answers its own ceiling.
    pub fn lowest_neighbour_ceiling(&self, sector: usize) -> i16 {
        let mut best: Option<i16> = None;
        for &line in self.sector_lines.get(sector).map(|v| v.as_slice()).unwrap_or(&[]) {
            let Some(other) = self.across(line, sector) else { continue };
            let Some(sec) = self.sectors.get(other) else { continue };
            best = Some(match best {
                Some(b) if b <= sec.ceiling => b,
                _ => sec.ceiling,
            });
        }
        best.unwrap_or_else(|| self.sectors.get(sector).map(|s| s.ceiling).unwrap_or(0))
    }

    /// Every sector sharing a two-sided line with this one.
    ///
    /// The unit every moving-surface kind is defined in: a lift goes to the
    /// *lowest neighbouring floor*, a door to the *lowest neighbouring
    /// ceiling*, a stair to the *next highest*. DOOM has one of these per
    /// question; this is the walk they share.
    fn neighbours(&self, sector: usize) -> impl Iterator<Item = &Sector> + '_ {
        self.sector_lines
            .get(sector)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
            .iter()
            .filter_map(move |&line| self.across(line, sector))
            .filter_map(move |i| self.sectors.get(i))
    }

    /// The highest floor among a sector's neighbours, or its own if it has
    /// none. DOOM's `P_FindHighestFloorSurrounding`.
    pub fn highest_neighbour_floor(&self, sector: usize) -> i16 {
        let own = self.sectors.get(sector).map(|s| s.floor).unwrap_or(0);
        // Vanilla seeds this at -500 rather than at the sector's own height,
        // and the difference is visible: a sector lower than everything around
        // it lowers to -500 instead of staying put. Seeded from the neighbours
        // and falling back to its own height, which is what the seed was
        // standing in for.
        self.neighbours(sector).map(|s| s.floor).max().unwrap_or(own)
    }

    /// The lowest floor among a sector's neighbours. Where a lift goes down to.
    pub fn lowest_neighbour_floor(&self, sector: usize) -> i16 {
        let own = self.sectors.get(sector).map(|s| s.floor).unwrap_or(0);
        self.neighbours(sector).map(|s| s.floor).min().unwrap_or(own)
    }

    /// The lowest neighbouring floor that is still *above* a height.
    ///
    /// DOOM's `P_FindNextHighestFloor`, and what builds a staircase: each step
    /// rises to the next one up rather than to the highest, so a flight of
    /// eight becomes eight steps instead of one wall.
    pub fn next_highest_neighbour_floor(&self, sector: usize, above: i16) -> i16 {
        let own = self.sectors.get(sector).map(|s| s.floor).unwrap_or(above);
        self.neighbours(sector)
            .map(|s| s.floor)
            .filter(|&h| h > above)
            .min()
            .unwrap_or(own)
    }

    /// Which sector a point is in, by index.
    ///
    /// The same walk as `sector_at`, answering the index instead of the
    /// record. Anything that has to survive a sector *moving* wants this one:
    /// a `&Sector` is a snapshot of a value that changes, and an index is not.
    pub fn sector_index_at(&self, x: i32, y: i32) -> Option<usize> {
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
            if (sd.sector as usize) < self.sectors.len() {
                return Some(sd.sector as usize);
            }
        }
        None
    }

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

/// Which linedefs touch each sector.
///
/// Both sides of every line are walked, so a two-sided line appears in the
/// list of both sectors it separates -- which is what a neighbour query needs
/// and is the whole reason the index exists.
fn build_sector_lines(
    linedefs: &[LineDef],
    sidedefs: &[SideDef],
    nsectors: usize,
) -> Vec<Vec<u16>> {
    let mut out: Vec<Vec<u16>> = Vec::new();
    out.resize_with(nsectors, Vec::new);
    for (i, l) in linedefs.iter().enumerate() {
        for side in [l.right, l.left] {
            if side == NONE {
                continue;
            }
            let Some(sd) = sidedefs.get(side as usize) else { continue };
            let Some(list) = out.get_mut(sd.sector as usize) else { continue };
            if !list.contains(&(i as u16)) {
                list.push(i as u16);
            }
        }
    }
    out
}

/// Sectors grouped by tag, skipping tag 0.
fn build_tag_index(sectors: &[Sector]) -> Vec<(i32, Vec<u16>)> {
    let mut out: Vec<(i32, Vec<u16>)> = Vec::new();
    for (i, s) in sectors.iter().enumerate() {
        let t = s.tag as i32;
        if t == 0 {
            continue;
        }
        match out.iter_mut().find(|(k, _)| *k == t) {
            Some((_, v)) => v.push(i as u16),
            None => out.push((t, alloc::vec![i as u16])),
        }
    }
    out
}

/// What `diag doom` asks of the level indexes.
///
/// Claims rather than printed lines, because nothing in this tree may name the
/// printing macro -- the same bargain the WAD reader's `Error` type makes.
///
/// These two indexes are the sort that fail silently. A tag lookup that
/// answered the wrong sector opens the wrong door, which on a real map looks
/// like a mapper's mistake; a neighbour walk that missed one side of a line
/// gives a door that opens to the wrong height, which looks like a texture
/// problem. Neither produces an error.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let sec = |ceiling: i16, tag: i16| Sector {
        floor: 0,
        ceiling,
        floor_pic: Name::from_lump(b"F       "),
        ceiling_pic: Name::from_lump(b"C       "),
        light: 160,
        special: 0,
        tag,
    };

    // Three sectors: two sharing tag 7, one untagged.
    let sectors = alloc::vec![sec(128, 7), sec(96, 0), sec(64, 7)];
    let idx = build_tag_index(&sectors);
    out.push((
        "sectors sharing a tag group together",
        idx.iter().find(|(t, _)| *t == 7).map(|(_, v)| v.as_slice()) == Some(&[0u16, 2][..]),
    ));
    out.push(("and an untagged sector joins no group", idx.len() == 1));

    // A two-sided line must appear in *both* the sectors it separates, which
    // is the whole point of the index -- a neighbour query that only walked
    // the front side would never see the sector on the other side of a door.
    let side = |sector: u16| SideDef {
        x_off: 0,
        y_off: 0,
        upper: Name::from_lump(b"-       "),
        lower: Name::from_lump(b"-       "),
        middle: Name::from_lump(b"-       "),
        sector,
    };
    let sidedefs = alloc::vec![side(0), side(1)];
    let line = |right: u16, left: u16| LineDef {
        v1: 0,
        v2: 1,
        flags: 0,
        special: 0,
        tag: 0,
        right,
        left,
    };
    let linedefs = alloc::vec![line(0, 1)];
    let by_sector = build_sector_lines(&linedefs, &sidedefs, 3);
    out.push((
        "a two-sided line is listed under both its sectors",
        by_sector[0].as_slice() == [0u16] && by_sector[1].as_slice() == [0u16],
    ));
    out.push(("and under no others", by_sector[2].is_empty()));

    let lv = Level {
        name: Name::from_lump(b"TEST    "),
        sector_lines: by_sector,
        tagged: idx,
        exited: false,
        things: Vec::new(),
        vertexes: alloc::vec![Vertex { x: 0, y: 0 }, Vertex { x: 64, y: 0 }],
        linedefs,
        sidedefs,
        segs: Vec::new(),
        subsectors: Vec::new(),
        nodes: Vec::new(),
        sectors,
    };
    out.push(("tag 0 matches nothing, rather than everything untagged", lv.tagged(0).is_empty()));
    out.push(("a tag names its sectors", lv.tagged(7) == [0u16, 2]));
    out.push((
        "a sector's neighbour is found across a two-sided line",
        lv.lowest_neighbour_ceiling(0) == 96,
    ));
    out.push((
        "and a sector with no neighbour answers its own ceiling",
        lv.lowest_neighbour_ceiling(2) == 64,
    ));
    out
}

"""Build a WAD with a real map in it, and read it back.

The kernel's WAD and level readers need something true to parse before there
is an IWAD on the machine, and they need the malformed cases far more than the
good one -- a parser whose error paths have never run is a parser with no
error paths, and in a kernel with no unwinder those paths are the difference
between "that file is truncated" and a halted machine.

    python tools/mkwad.py out/test.wad --verify
    python tools/mkwad.py out/bad.wad --break trunc-dir
    python tools/mkwad.py --list

The reader in `--verify` is deliberately *not* the writer: a shared
misunderstanding of the format then shows up as a disagreement rather than as
two halves of one bug agreeing. Same bargain `tokenizer.py --verify` makes.

--------------------------------------------------------------------------
WHAT THE MAP IS, AND WHAT IT IS NOT
--------------------------------------------------------------------------

`E1M1` is a single square room, 1024 units on a side, one sector, floor at 0
and ceiling at 128, with a player 1 start at the origin facing north.

It is a **complete** map in the sense that every lump vanilla expects is
present and structurally valid: THINGS, LINEDEFS, SIDEDEFS, VERTEXES, SEGS,
SSECTORS, NODES, SECTORS, REJECT, BLOCKMAP, in that order, each a whole number
of records.

The BSP is hand-built rather than produced by a nodebuilder, and that shapes
what it can be used for. The room is six vertices rather than four so that a
vertical partition through the middle splits it into two convex halves without
having to split any wall: three walls fall either side, one node, two
subsectors, and node traversal is genuinely exercised.

**Three things it does not have, said here so they are not diagnosed as
renderer bugs later:**

  * **No minisegs.** A real nodebuilder emits segs along the partition itself,
    with linedef 0xFFFF, so each subsector is a closed polygon. These
    subsectors are open along x=0. Wall rendering does not care; anything that
    fills flats by walking a subsector's edges will.
  * **No textures.** The names in the sidedefs and sector are the usual ones
    (STARTAN3, FLOOR4_8, CEIL3_5) but there are no TEXTURE1, PNAMES or patch
    lumps in this WAD, so nothing can look them up. A renderer needs a real
    IWAD, or those lumps added here.
  * **REJECT is all zero**, meaning no sector pair is rejected. That is the
    permissive answer and is always safe; it just does no work.

The palette is real: 256 entries as a legible sixteen-by-sixteen ramp, so
`PLAYPAL` can be pushed through the framebuffer and checked by eye.
"""

import argparse
import math
import struct
import sys
from pathlib import Path

HEADER = 12
DIR_ENTRY = 16

# Record sizes, which are also what `--verify` checks each lump against.
SIZES = {
    "THINGS": 10,
    "LINEDEFS": 14,
    "SIDEDEFS": 30,
    "VERTEXES": 4,
    "SEGS": 12,
    "SSECTORS": 4,
    "NODES": 28,
    "SECTORS": 26,
}

# DOOM's blockmap grid is 128 units, and the size is not stored in the file --
# it is compiled into the engine. A generator that picks its own would produce
# a blockmap every reader disagrees with.
BLOCK = 128

NO_SIDE = 0xFFFF
SUBSECTOR_BIT = 0x8000


def name8(s):
    return s.encode("ascii")[:8].ljust(8, b"\x00")


def playpal():
    """256 RGB triples: sixteen hues by sixteen brightnesses.

    Legible on purpose. A palette of noise proves the bytes arrived; a ramp
    proves they arrived *in order*, which is the failure a palette path
    actually has.
    """
    out = bytearray()
    for i in range(256):
        hue, lev = divmod(i, 16)
        v = lev * 17
        table = [
            (v, v, v), (v, 0, 0), (0, v, 0), (0, 0, v),
            (v, v, 0), (0, v, v), (v, 0, v), (v, v // 2, 0),
        ]
        out += bytes(table[hue] if hue < 8 else (v, (hue - 8) * 32, 255 - v))
    return bytes(out)


def colormap():
    """34 light levels of 256 bytes. Identity, which is wrong and harmless.

    A real COLORMAP darkens; this one maps every index to itself at every
    level, so a renderer using it draws at full brightness everywhere. That is
    a visibly flat picture rather than a broken one, and it is obvious enough
    that nobody will mistake it for a lighting bug in their own code.
    """
    return bytes(bytes(range(256)) * 34)


# --------------------------------------------------------------------------
# The map
# --------------------------------------------------------------------------

R = 512  # half-width of the room

# Six, not four. The two extra sit at x=0 on the top and bottom walls, so the
# partition can pass between them without splitting a linedef -- which is the
# whole reason a nodebuilder is normally required.
VERTEXES = [
    (-R, -R),   # 0
    (0, -R),    # 1
    (R, -R),    # 2
    (R, R),     # 3
    (0, R),     # 4
    (-R, R),    # 5
]

# Anticlockwise, so the right side of each faces into the room -- which is
# what makes the sidedef on the right the one that exists.
LINEDEFS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]

# Which linedefs fall on which side of the partition x=0.
RIGHT_LINES = [1, 2, 3]   # x >= 0
LEFT_LINES = [4, 5, 0]    # x <= 0


def bam(x0, y0, x1, y1):
    """A DOOM binary angle: a full turn is 65536, and east is zero."""
    a = math.atan2(y1 - y0, x1 - x0)
    return int(round(a / (2 * math.pi) * 65536)) & 0xFFFF


def build_things():
    # type 1 is the player 1 start; angle is in degrees with 90 as north;
    # flags 7 is "present on all three skill levels".
    return struct.pack("<hhhhh", 0, 0, 90, 1, 7)


def build_vertexes():
    return b"".join(struct.pack("<hh", x, y) for x, y in VERTEXES)


def build_linedefs():
    out = b""
    for i, (a, b) in enumerate(LINEDEFS):
        # flag 1 is impassable, which every one-sided wall is.
        out += struct.pack("<HHHHHHH", a, b, 1, 0, 0, i, NO_SIDE)
    return out


def build_sidedefs():
    out = b""
    for _ in LINEDEFS:
        out += struct.pack("<hh", 0, 0)
        out += name8("-") + name8("-") + name8("STARTAN3")
        out += struct.pack("<H", 0)
    return out


def build_sectors():
    return (
        struct.pack("<hh", 0, 128)
        + name8("FLOOR4_8")
        + name8("CEIL3_5")
        + struct.pack("<hhh", 160, 0, 0)
    )


def build_segs():
    """One seg per wall, grouped by subsector so SSECTORS can name a run.

    No minisegs along the partition -- see the module docstring. Order matters:
    a subsector names its segs as a first index and a count, so the segs of one
    subsector have to be contiguous.
    """
    out = b""
    for group in (RIGHT_LINES, LEFT_LINES):
        for li in group:
            a, b = LINEDEFS[li]
            (x0, y0), (x1, y1) = VERTEXES[a], VERTEXES[b]
            # direction 0: the seg runs the same way as its linedef, so its
            # side is the linedef's right side, which is the one with a sidedef.
            out += struct.pack("<HHHHHh", a, b, bam(x0, y0, x1, y1), li, 0, 0)
    return out


def build_ssectors():
    return struct.pack("<HH", len(RIGHT_LINES), 0) + struct.pack(
        "<HH", len(LEFT_LINES), len(RIGHT_LINES)
    )


def build_nodes():
    """One node, splitting the room down x=0.

    The partition runs from (0,-512) to (0,512), so dx=0 and dy=1024. DOOM's
    own side test is `(dy * px) < (dx * py)` after translating to the node's
    origin, which with dx=0 puts positive x on the **right**. Getting that
    backwards mirrors the level, which is a bug that looks like a renderer
    problem and is not one.

    A bounding box is (top, bottom, left, right) -- y first, and top before
    bottom, which is the opposite of the order the words usually come in.
    """
    right_bbox = (R, -R, 0, R)
    left_bbox = (R, -R, -R, 0)
    return struct.pack(
        "<hhhh" + "hhhh" + "hhhh" + "HH",
        0, -R, 0, 2 * R,
        *right_bbox,
        *left_bbox,
        SUBSECTOR_BIT | 0,
        SUBSECTOR_BIT | 1,
    )


def build_reject(nsectors):
    bits = nsectors * nsectors
    return b"\x00" * ((bits + 7) // 8)


def build_blockmap():
    """A real blockmap: every linedef listed in every 128-unit cell it crosses.

    Brute force, by bounding box overlap. That over-reports a diagonal, which
    is allowed -- the blockmap is a broad-phase filter and a false positive
    costs one extra intersection test, where a false negative is a wall you
    walk through.
    """
    xs = [v[0] for v in VERTEXES]
    ys = [v[1] for v in VERTEXES]
    ox, oy = min(xs), min(ys)
    cols = (max(xs) - ox) // BLOCK + 1
    rows = (max(ys) - oy) // BLOCK + 1

    cells = []
    for r in range(rows):
        for c in range(cols):
            bx0, by0 = ox + c * BLOCK, oy + r * BLOCK
            bx1, by1 = bx0 + BLOCK, by0 + BLOCK
            here = []
            for i, (a, b) in enumerate(LINEDEFS):
                (x0, y0), (x1, y1) = VERTEXES[a], VERTEXES[b]
                if max(x0, x1) < bx0 or min(x0, x1) > bx1:
                    continue
                if max(y0, y1) < by0 or min(y0, y1) > by1:
                    continue
                here.append(i)
            cells.append(here)

    # Offsets are in 16-bit words from the start of the blockmap, and the
    # header is four words plus one per cell.
    head_words = 4 + cols * rows
    offsets = []
    body = []
    cursor = head_words
    for here in cells:
        offsets.append(cursor)
        # Every list starts with a 0 and ends with 0xFFFF. The leading zero is
        # not a linedef index -- it is a quirk of the original format that
        # every reader relies on.
        body.append([0] + here + [0xFFFF])
        cursor += len(here) + 2

    out = struct.pack("<hhHH", ox, oy, cols, rows)
    out += b"".join(struct.pack("<H", o) for o in offsets)
    for lst in body:
        out += b"".join(struct.pack("<H", v) for v in lst)
    return out


def map_lumps():
    """The marker and its ten lumps, in the order vanilla expects them."""
    return [
        ("E1M1", b""),
        ("THINGS", build_things()),
        ("LINEDEFS", build_linedefs()),
        ("SIDEDEFS", build_sidedefs()),
        ("VERTEXES", build_vertexes()),
        ("SEGS", build_segs()),
        ("SSECTORS", build_ssectors()),
        ("NODES", build_nodes()),
        ("SECTORS", build_sectors()),
        ("REJECT", build_reject(1)),
        ("BLOCKMAP", build_blockmap()),
    ]


def all_lumps():
    return (
        [("PLAYPAL", playpal()), ("COLORMAP", colormap())]
        + map_lumps()
        # Two with one name, to exercise the rule that a lookup searches from
        # the end and the later one wins.
        + [("DUPE", b"first"), ("DUPE", b"second")]
    )


BREAKS = {
    "none": "a valid WAD",
    "magic": "first four bytes are XWAD",
    "short": "truncated to eight bytes, shorter than a header",
    "trunc-dir": "directory offset points past the end",
    "huge-count": "header claims 4 billion lumps",
    "lump-past-end": "one lump's offset+size runs past the file",
    "negative": "a lump size is negative",
    "odd-linedefs": "LINEDEFS is not a whole number of records",
    "bad-vertex": "a linedef names a vertex that does not exist",
}


def build(kind=b"IWAD", how="none"):
    if how == "short":
        return b"IWAD\x00\x00\x00\x00"

    lumps = all_lumps()
    if how == "odd-linedefs":
        lumps = [(n, d + b"\x00" if n == "LINEDEFS" else d) for n, d in lumps]
    if how == "bad-vertex":
        patched = []
        for n, d in lumps:
            if n == "LINEDEFS":
                d = struct.pack("<H", 999) + d[2:]
            patched.append((n, d))
        lumps = patched

    body = bytearray()
    entries = []
    for name, data in lumps:
        entries.append((HEADER + len(body), len(data), name))
        body += data

    dir_at = HEADER + len(body)
    magic = b"XWAD" if how == "magic" else kind
    count = 0xFFFF_FFFF if how == "huge-count" else len(lumps)
    at = dir_at + 0x1000 if how == "trunc-dir" else dir_at

    out = bytearray()
    out += magic
    out += struct.pack("<II", count, at)
    out += body
    for pos, size, name in entries:
        if how == "lump-past-end" and name == "SECTORS":
            size = 1 << 20
        if how == "negative" and name == "SECTORS":
            size = -4
        out += struct.pack("<ii", pos, size)
        out += name8(name)
    return bytes(out)


# --------------------------------------------------------------------------
# The reader, which is deliberately not the writer
# --------------------------------------------------------------------------


def read_back(data):
    if len(data) < HEADER:
        raise ValueError("shorter than a header")
    magic = data[:4]
    if magic not in (b"IWAD", b"PWAD"):
        raise ValueError(f"bad magic {magic!r}")
    count, at = struct.unpack("<II", data[4:12])
    if at + count * DIR_ENTRY > len(data):
        raise ValueError(f"directory of {count} at {at} runs past {len(data)}")
    lumps = []
    for i in range(count):
        e = at + i * DIR_ENTRY
        pos, size = struct.unpack("<ii", data[e:e + 8])
        name = data[e + 8:e + 16].rstrip(b"\x00").decode("ascii", "replace")
        if pos < 0 or size < 0 or pos + size > len(data):
            raise ValueError(f"lump {i} {name!r} wants {size} at {pos}")
        lumps.append((name, data[pos:pos + size]))
    return magic.decode(), lumps


def check_map(lumps):
    """Everything the level reader will later assert, asserted here first."""
    by = {}
    for n, d in lumps:
        by.setdefault(n, d)

    for name, rec in SIZES.items():
        if name not in by:
            raise ValueError(f"no {name} lump")
        if len(by[name]) % rec:
            raise ValueError(
                f"{name} is {len(by[name])} bytes, not a multiple of {rec}"
            )

    nverts = len(by["VERTEXES"]) // SIZES["VERTEXES"]
    nsides = len(by["SIDEDEFS"]) // SIZES["SIDEDEFS"]
    nsectors = len(by["SECTORS"]) // SIZES["SECTORS"]
    nlines = len(by["LINEDEFS"]) // SIZES["LINEDEFS"]
    nsegs = len(by["SEGS"]) // SIZES["SEGS"]
    nsub = len(by["SSECTORS"]) // SIZES["SSECTORS"]
    nnodes = len(by["NODES"]) // SIZES["NODES"]

    for i in range(nlines):
        e = i * SIZES["LINEDEFS"]
        a, b, _fl, _sp, _tag, right, left = struct.unpack(
            "<HHHHHHH", by["LINEDEFS"][e:e + 14]
        )
        if a >= nverts or b >= nverts:
            raise ValueError(f"linedef {i} names vertex {a}/{b} of {nverts}")
        if right != NO_SIDE and right >= nsides:
            raise ValueError(f"linedef {i} names sidedef {right} of {nsides}")
        if left != NO_SIDE and left >= nsides:
            raise ValueError(f"linedef {i} names sidedef {left} of {nsides}")

    for i in range(nsides):
        e = i * SIZES["SIDEDEFS"] + 28
        (sec,) = struct.unpack("<H", by["SIDEDEFS"][e:e + 2])
        if sec >= nsectors:
            raise ValueError(f"sidedef {i} names sector {sec} of {nsectors}")

    for i in range(nsegs):
        e = i * SIZES["SEGS"]
        a, b, _ang, li, _dir, _off = struct.unpack("<HHHHHh", by["SEGS"][e:e + 12])
        if a >= nverts or b >= nverts:
            raise ValueError(f"seg {i} names vertex {a}/{b} of {nverts}")
        if li != NO_SIDE and li >= nlines:
            raise ValueError(f"seg {i} names linedef {li} of {nlines}")

    seen = 0
    for i in range(nsub):
        e = i * SIZES["SSECTORS"]
        cnt, first = struct.unpack("<HH", by["SSECTORS"][e:e + 4])
        if first + cnt > nsegs:
            raise ValueError(f"subsector {i} runs to seg {first + cnt} of {nsegs}")
        seen += cnt
    if seen != nsegs:
        raise ValueError(f"subsectors cover {seen} segs of {nsegs}")

    for i in range(nnodes):
        e = i * SIZES["NODES"] + 24
        r, l = struct.unpack("<HH", by["NODES"][e:e + 4])
        for child in (r, l):
            if child & SUBSECTOR_BIT:
                if (child & ~SUBSECTOR_BIT) >= nsub:
                    raise ValueError(f"node {i} names subsector {child & 0x7FFF}")
            elif child >= nnodes:
                raise ValueError(f"node {i} names node {child} of {nnodes}")

    (px, py, ang, kind, _flags) = struct.unpack("<hhhhh", by["THINGS"][:10])
    if kind != 1:
        raise ValueError(f"first thing is type {kind}, not a player start")

    return {
        "verts": nverts, "lines": nlines, "sides": nsides,
        "sectors": nsectors, "segs": nsegs, "subsectors": nsub,
        "nodes": nnodes, "start": (px, py, ang),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--break", dest="how", default="none", choices=sorted(BREAKS))
    ap.add_argument("--pwad", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in sorted(BREAKS.items()):
            print(f"  {k:14} {v}")
        return 0
    if not args.out:
        ap.error("need an output path (or --list)")

    data = build(b"PWAD" if args.pwad else b"IWAD", args.how)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    print(f"{p}  {len(data)} bytes  ({BREAKS[args.how]})")

    if not args.verify:
        return 0

    try:
        kind, lumps = read_back(data)
        stats = check_map(lumps)
    except ValueError as e:
        if args.how == "none":
            print(f"FAIL: a WAD we wrote does not read back: {e}")
            return 1
        print(f"  rejected, as intended: {e}")
        return 0

    if args.how != "none":
        print(f"FAIL: {args.how} was supposed to be rejected and parsed fine")
        return 1

    print(f"  {kind}, {len(lumps)} lumps")
    for name, d in lumps:
        print(f"    {name:9} {len(d):>6} B")
    print(
        "  map: {verts} vertexes, {lines} linedefs, {sides} sidedefs, "
        "{sectors} sector(s)".format(**stats)
    )
    print(
        "       {segs} segs, {subsectors} subsectors, {nodes} node(s), "
        "start at {start}".format(**stats)
    )
    last = [d for n, d in lumps if n == "DUPE"][-1]
    assert last == b"second", "last DUPE should win"
    print("  last duplicate wins: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

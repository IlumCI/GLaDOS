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
  * **No flats.** The sector names FLOOR4_8 and CEIL3_5 and there is no
    F_START/F_END namespace here, so floors and ceilings have nothing to look
    up. Walls do: `STARTAN3` is a real composed texture in this file.
  * **REJECT is all zero**, meaning no sector pair is rejected. That is the
    permissive answer and is always safe; it just does no work.

The palette is real: 256 entries as a legible sixteen-by-sixteen ramp, so
`PLAYPAL` can be pushed through the framebuffer and checked by eye. So is the
COLORMAP, and so are the wall textures -- the *art* in them is generated,
because id's is not ours to ship, but the *encoding* is the real one: columns
of posts with real transparency, a PNAMES list and a TEXTURE1 directory. The
patterns are chosen so that a decoding mistake reads as an obviously wrong
picture rather than as slightly odd art; `make_patch` says which mistake each
one catches.
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
    """34 light levels of 256 bytes, and it darkens.

    This shipped as an identity table for a while, on the reasoning that a
    flat picture is obviously flat. It is not: a renderer that used it drew
    every wall at full brightness at every distance, which reads exactly like
    a lighting bug in the renderer -- and a test file whose failure mode is
    indistinguishable from the bug it is meant to catch is worse than no test
    file. So it darkens for real now, and the kernel prefers a WAD's own table
    only when it can see that it does.

    A COLORMAP is *the* answer to shading indexed art, because a palette index
    is not a colour and cannot be dimmed by arithmetic. Map 0 is full
    brightness, map 31 is nearly black, map 32 is the invulnerability inverse
    and map 33 is unused and all zero, which is the layout every DOOM engine
    indexes by number.
    """
    pal = playpal()
    out = bytearray()
    for level in range(32):
        f = 1.0 - level / 31.0
        for i in range(256):
            j = i * 3
            out.append(
                _nearest(pal, int(pal[j] * f), int(pal[j + 1] * f), int(pal[j + 2] * f))
            )
    # 32: the inverse ramp the invulnerability sphere draws through, keyed on
    # luminance, since this palette has no grey ramp to index directly.
    for i in range(256):
        j = i * 3
        g = 255 - (pal[j] * 30 + pal[j + 1] * 59 + pal[j + 2] * 11) // 100
        out.append(_nearest(pal, g, g, g))
    # 33: all black. Unused by anything, and present because a reader indexing
    # by map number expects 34 of them.
    out += bytes(256)
    return bytes(out)


def _nearest(pal, r, g, b):
    """The palette index closest to a colour, by squared distance in RGB.

    The same search the kernel's `doom::draw::nearest` runs, deliberately: a
    generated COLORMAP built by a different rule than the fallback the kernel
    builds would make the two paths disagree about the same WAD, and telling
    which one was in force is exactly what this file exists to make easy.
    """
    best, best_d = 0, 1 << 30
    for i in range(256):
        j = i * 3
        dr, dg, db = pal[j] - r, pal[j + 1] - g, pal[j + 2] - b
        d = dr * dr + dg * dg + db * db
        if d < best_d:
            best, best_d = i, d
            if d == 0:
                break
    return best


# --------------------------------------------------------------------------
# Patches and textures
# --------------------------------------------------------------------------
#
# The *art* here is generated and the *format* is not. There is no way to ship
# id's patches, so the pictures are drawn by this file -- but they are drawn
# into the real column-and-post encoding, with real transparency, so the
# kernel's decoder is exercised by exactly the structure a commercial WAD has.
#
# The patterns are chosen to make a decoding mistake visible rather than
# merely wrong:
#
#   * a bright marker in the **top-left 8x8** -- upside-down puts it at the
#     bottom, mirrored puts it on the right, and both are instant to see;
#   * horizontal mortar lines and vertical joins, so a column mapped to the
#     wrong x smears rather than blending in;
#   * a vertical gradient, so a v coordinate that runs backwards is obvious
#     even where the bricks happen to line up;
#   * a **transparent hole**, because a patch with no gaps never exercises the
#     post loop at all -- a decoder that ignored posts entirely and copied
#     `width * height` bytes would pass on a solid patch.

PATCH_W, PATCH_H = 64, 128

# Palette indices into `playpal` above: hue * 16 + level.
def _idx(hue, lev):
    return hue * 16 + max(0, min(15, lev))


TRANSPARENT = None


def brick_pixel(x, y):
    """One pixel of the wall patch, or None where it is see-through."""
    # A hole, so the post encoding has something to encode. Off-centre, so it
    # also shows a mirrored patch.
    if 40 <= x < 56 and 16 <= y < 40:
        return TRANSPARENT
    # The corner marker: bright green, top-left.
    if x < 8 and y < 8:
        return _idx(2, 15)
    # Mortar: a course every 16 rows, and a vertical join every 32 columns
    # offset on alternate courses -- which is what makes a wrongly ordered
    # column obvious rather than merely different.
    course = y // 16
    if y % 16 < 2:
        return _idx(0, 6)
    if (x + (0 if course % 2 == 0 else 16)) % 32 < 2:
        return _idx(0, 6)
    # The brick face, darkening down the patch.
    return _idx(7, 14 - (y * 10 // PATCH_H))


def stripe_pixel(x, y):
    """A second patch, so composition has two things to compose."""
    if x % 16 < 8:
        return _idx(3, 4 + (y * 10 // PATCH_H))
    return _idx(5, 4 + (y * 10 // PATCH_H))


def make_patch(w, h, fn, left=0, top=0):
    """Encode a picture in DOOM's column-and-post format.

    A column is a series of posts, each `topdelta, length, pad, pixels..., pad`
    and the whole terminated by a topdelta of 255. The two pad bytes are not
    padding in any useful sense -- the renderer reads one pixel before and
    after each post for its own smoothing -- but they are part of the format
    and a reader that omits them is off by one from the first post onward.
    """
    columns = []
    for x in range(w):
        col = bytearray()
        y = 0
        while y < h:
            # Skip transparent runs.
            while y < h and fn(x, y) is TRANSPARENT:
                y += 1
            if y >= h:
                break
            run_start = y
            run = bytearray()
            while y < h and fn(x, y) is not TRANSPARENT and len(run) < 254:
                run.append(fn(x, y))
                y += 1
            col.append(run_start)          # topdelta
            col.append(len(run))           # length
            col.append(run[0])             # leading pad
            col += run
            col.append(run[-1])            # trailing pad
        col.append(0xFF)                   # end of column
        columns.append(bytes(col))

    header = struct.pack("<hhhh", w, h, left, top)
    table_len = 4 * w
    offsets = []
    cursor = len(header) + table_len
    for c in columns:
        offsets.append(cursor)
        cursor += len(c)
    out = bytearray(header)
    for o in offsets:
        out += struct.pack("<i", o)
    for c in columns:
        out += c
    return bytes(out)


def make_pnames(names):
    out = struct.pack("<i", len(names))
    for n in names:
        out += name8(n)
    return bytes(out)


def make_texture1(textures, pnames):
    """The texture directory.

    `maptexture_t` carries two fields no engine has read since 1993 --
    `masked` and `columndirectory` -- and they are written as zero here
    because a reader that skips them by the wrong width lands mid-field on the
    patch count and produces a texture with tens of thousands of patches.
    """
    idx = {n: i for i, n in enumerate(pnames)}
    bodies = []
    for name, w, h, patches in textures:
        b = bytearray(name8(name))
        b += struct.pack("<i", 0)              # masked, unused
        b += struct.pack("<hh", w, h)
        b += struct.pack("<i", 0)              # columndirectory, unused
        b += struct.pack("<h", len(patches))
        for ox, oy, pname in patches:
            b += struct.pack("<hhhhh", ox, oy, idx[pname], 1, 0)
        bodies.append(bytes(b))

    head = struct.pack("<i", len(bodies))
    table = 4 * len(bodies)
    offsets = []
    cursor = len(head) + table
    for b in bodies:
        offsets.append(cursor)
        cursor += len(b)
    out = bytearray(head)
    for o in offsets:
        out += struct.pack("<i", o)
    for b in bodies:
        out += b
    return bytes(out)


PATCHES = [
    ("WALL01", lambda: make_patch(PATCH_W, PATCH_H, brick_pixel)),
    ("WALL02", lambda: make_patch(PATCH_W, PATCH_H, stripe_pixel)),
]

# `STARTAN3` is the name the map's sidedefs already carry, so the walls
# resolve without touching the map. `BIGWALL` is two patches side by side,
# which is the case a single-patch texture never tests: an origin that is
# ignored draws both at x=0 and the right half stays empty.
TEXTURES = [
    ("STARTAN3", PATCH_W, PATCH_H, [(0, 0, "WALL01")]),
    ("BIGWALL", PATCH_W * 2, PATCH_H, [(0, 0, "WALL01"), (PATCH_W, 0, "WALL02")]),
]


def texture_lumps():
    names = [n for n, _ in PATCHES]
    out = [
        ("PNAMES", make_pnames(names)),
        ("TEXTURE1", make_texture1(TEXTURES, names)),
        # The patch namespace. Not consulted by a name lookup, but a WAD
        # without the markers is one a real editor will not open.
        ("P_START", b""),
    ]
    for n, build in PATCHES:
        out.append((n, build()))
    out.append(("P_END", b""))
    return out


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

# Clockwise, and the direction is not cosmetic.
#
# DOOM's front side is the *right* one, so for a one-sided wall the right side
# has to face into its sector -- which means walking the boundary with the room
# on your right, which is clockwise. Anticlockwise puts every front face
# outward, and the symptom is not a mirrored room or a stripe: it is an
# entirely empty screen, because the renderer's backface cull correctly
# discards every wall in the level.
#
# This was anticlockwise, with a comment claiming it was the other way, and it
# cost a debugging session on a renderer that was right all along.
LINEDEFS = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (0, 5)]

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
        + texture_lumps()
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

    # Winding. Every one-sided wall's right side must face into its sector,
    # and the failure mode is an empty screen rather than a wrong picture --
    # which is much harder to diagnose, because an empty screen looks like a
    # renderer that has not been finished yet.
    for i in range(nlines):
        e = i * SIZES["LINEDEFS"]
        a, b, _fl, _sp, _tag, right, left = struct.unpack(
            "<HHHHHHH", by["LINEDEFS"][e:e + 14]
        )
        if left != NO_SIDE:
            continue
        (ax, ay) = struct.unpack("<hh", by["VERTEXES"][a * 4:a * 4 + 4])
        (bx, by_) = struct.unpack("<hh", by["VERTEXES"][b * 4:b * 4 + 4])
        dx, dy = bx - ax, by_ - ay
        # A step from the midpoint along the right-hand normal, which for a
        # convex room centred on the origin lands inside it.
        mx, my = (ax + bx) / 2, (ay + by_) / 2
        n = (dx * dx + dy * dy) ** 0.5 or 1
        px_, py_ = mx + dy / n * 8, my - dx / n * 8
        if not (abs(px_) < R and abs(py_) < R):
            raise ValueError(
                f"linedef {i} is wound the wrong way: its right side faces out"
            )

    # The COLORMAP has to actually darken, which is the property the kernel
    # tests before it will use one -- so a regression here would silently send
    # every render down the fallback path instead.
    if "COLORMAP" in by:
        cm = by["COLORMAP"]
        if len(cm) != 34 * 256:
            raise ValueError(f"COLORMAP is {len(cm)} bytes, not 34 maps of 256")
        if cm[:256] == cm[31 * 256:32 * 256]:
            raise ValueError("COLORMAP's brightest and darkest maps are identical")
        if any(b != 0 for b in cm[33 * 256:]):
            raise ValueError("COLORMAP's last map should be all black")

    # The texture directory, checked the same way the map is: every reference
    # resolved, every offset inside the lump.
    if "TEXTURE1" in by and "PNAMES" in by:
        pn = by["PNAMES"]
        (npatch,) = struct.unpack("<i", pn[:4])
        if 4 + npatch * 8 > len(pn):
            raise ValueError(f"PNAMES claims {npatch} patches, past the lump")
        patch_names = [
            pn[4 + i * 8:12 + i * 8].rstrip(b"\x00").decode("ascii", "replace")
            for i in range(npatch)
        ]
        for n in patch_names:
            if not any(ln == n for ln, _ in lumps):
                raise ValueError(f"PNAMES names {n}, which is not in the WAD")

        t = by["TEXTURE1"]
        (ntex,) = struct.unpack("<i", t[:4])
        for i in range(ntex):
            (off,) = struct.unpack("<i", t[4 + i * 4:8 + i * 4])
            if off + 22 > len(t):
                raise ValueError(f"texture {i} at {off} runs past TEXTURE1")
            tname = t[off:off + 8].rstrip(b"\x00").decode("ascii", "replace")
            tw, th = struct.unpack("<hh", t[off + 12:off + 16])
            (np_,) = struct.unpack("<h", t[off + 20:off + 22])
            if tw <= 0 or th <= 0:
                raise ValueError(f"texture {tname} is {tw}x{th}")
            for j in range(np_):
                e = off + 22 + j * 10
                ox, oy, pi, _sd, _cm = struct.unpack("<hhhhh", t[e:e + 10])
                if not 0 <= pi < npatch:
                    raise ValueError(f"texture {tname} names patch {pi} of {npatch}")

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

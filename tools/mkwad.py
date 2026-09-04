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

`E1M1` is a single convex room -- a hexagon about 1024 units across, one
sector, floor at 0 and ceiling at 128 -- with a player 1 start at the origin
facing north, two barrels and a floor lamp. It said "square" here for a while
and the paragraph below said six vertices, which is the sort of disagreement
that costs somebody an afternoon.

It is a **complete** map in the sense that every lump vanilla expects is
present and structurally valid: THINGS, LINEDEFS, SIDEDEFS, VERTEXES, SEGS,
SSECTORS, NODES, SECTORS, REJECT, BLOCKMAP, in that order, each a whole number
of records.

The BSP is hand-built rather than produced by a nodebuilder, and that shapes
what it can be used for. The room is six vertices rather than four so that a
vertical partition through the middle splits it into two convex halves without
having to split any wall: three walls fall either side, one node, two
subsectors, and node traversal is genuinely exercised.

**Every picture in this file is generated here and none of it is id's.**
Stating it at the top because the distinction is easy to lose: what was
*ported* into the kernel is the code that reads DOOM's formats -- the patch
column-and-post decoder, the TEXTURE1/PNAMES reader, the flat and sprite
namespaces. What is written *here* is test art in those formats, drawn by the
functions below. It looks like a test pattern because it is one: a green corner
marker, mortar joins offset course by course, a transparent hole, a wedge that
is asymmetric under both a transpose and a mirror. Each exists to make one
specific decoding mistake visible, and none of it is meant to be looked at for
its own sake.

DOOM's own art is not in this repository and cannot be. Load an IWAD or
FreeDoom to see the real thing; the renderer needs no change for it, and that
claim is **untested**, because there is no WAD on the machine this was built
on.

**Three things it does not have, said here so they are not diagnosed as
renderer bugs later:**

  * **No minisegs.** A real nodebuilder emits segs along the partition itself,
    with linedef 0xFFFF, so each subsector is a closed polygon. These
    subsectors are open along x=0. Wall rendering does not care; anything that
    fills flats by walking a subsector's edges will.
  * **No sprites.** There is no S_START/S_END namespace and no sprite lumps,
    so a THING has nothing to draw. Walls, floors and ceilings all have real
    pictures: `STARTAN3` is a composed texture and `FLOOR4_8`/`CEIL3_5` are
    real flats.
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


# --------------------------------------------------------------------------
# Flats
# --------------------------------------------------------------------------
#
# A flat is 64 by 64 raw pixels and nothing else -- no header, no dimensions,
# no name inside it. The size *is* the format, which is why a flat lives in a
# namespace (`F_START` to `F_END`) instead of being identifiable by content,
# and why the only test a reader can apply is that the lump is 4096 bytes.
#
# So the patterns here have to catch what a wall texture's cannot. A wall is
# addressed by (u along the wall, v down it) and a floor by (world x, world y),
# and the second has two failure modes the first does not:
#
#   * **A transposed pair.** Feeding x where y belongs and vice versa is
#     invisible on anything with four-fold symmetry -- a checkerboard, a grid,
#     a noise field. So the pattern is deliberately different along each axis.
#   * **A mirrored axis.** DOOM negates world y to get the flat's row, because
#     north is up in the world and down in the picture. Getting that wrong
#     mirrors every floor in the game across the east-west axis, which on a
#     symmetric pattern is nothing at all.
#
# Hence a wedge: a diagonal that thickens toward one corner. Transposed it
# leans the other way, mirrored it points the other way, and both read at a
# glance.

FLAT_SIDE = 64


def floor_pixel(x, y):
    """The floor: a plated grid with a wedge marking one corner."""
    # The tile seam, so the 64-unit repeat is visible on a large floor and a
    # wrong world-to-flat scale shows as the wrong number of tiles.
    if x < 2 or y < 2:
        return _idx(0, 5)
    # The wedge. Filled below the diagonal, so the mass is toward the low-x,
    # high-y corner -- and neither a transpose nor a mirror leaves it there.
    if y > x + 8:
        return _idx(5, 12 - (y - x) // 8)
    # A dot grid, on a different pitch in each axis so the two cannot be
    # swapped without the spacing changing.
    if x % 16 == 8 and y % 8 == 4:
        return _idx(4, 14)
    return _idx(0, 2 + (x // 16))


def ceil_pixel(x, y):
    """The ceiling: darker, and unmistakably not the floor.

    A renderer that swapped a sector's two pictures would otherwise draw a
    perfectly plausible room.
    """
    if x < 2 or y < 2:
        return _idx(0, 3)
    if (x // 8 + y // 8) % 2 == 0:
        return _idx(3, 5 + (y // 16))
    return _idx(6, 3 + (x // 16))


def make_flat(fn):
    out = bytearray()
    for y in range(FLAT_SIDE):
        for x in range(FLAT_SIDE):
            out.append(fn(x, y))
    return bytes(out)


# The names the map's sector already carries, so the flats resolve without
# touching the map.
FLATS = [
    ("FLOOR4_8", floor_pixel),
    ("CEIL3_5", ceil_pixel),
]


def flat_lumps():
    out = [("F_START", b"")]
    for n, fn in FLATS:
        out.append((n, make_flat(fn)))
    out.append(("F_END", b""))
    return out


# --------------------------------------------------------------------------
# Sprites
# --------------------------------------------------------------------------
#
# A sprite is a patch -- the same column-and-post encoding a wall texture is
# built from -- living in its own namespace and named by a convention rather
# than by a directory: four characters of sprite name, a frame letter, and a
# rotation digit. `BAR1A0` is sprite BAR1, frame A, rotation 0, and rotation 0
# means "this picture from every angle", which is what every item and every
# piece of scenery uses.
#
# Two fields matter here that a wall texture never reads, and both are in the
# patch header this file already writes:
#
#   * **`left`** is how far the sprite's *centre* is from its left edge. A
#     renderer that ignores it hangs every sprite off to one side by half its
#     width, which on a symmetric picture looks like a positioning bug
#     somewhere else entirely.
#   * **`top`** is how far the sprite's *origin* is below its top edge, and the
#     origin sits at the thing's feet. So for something standing on the floor
#     it is the full height, and getting it wrong sinks every object into the
#     ground or floats it.
#
# So the pictures here are asymmetric left-to-right on purpose, and they do not
# fill their own bounding box -- a sprite that was a solid rectangle would say
# nothing about whether transparency survived the trip.

SPRITE_W, SPRITE_H = 48, 64


def barrel_pixel(x, y):
    """A drum with a bite out of one side, so a mirror shows and a column
    can hold more than one post."""
    # The body: an oval, so the corners of the patch are transparent and the
    # post encoding has to describe a different run in every column.
    cx, cy = SPRITE_W / 2.0, SPRITE_H / 2.0
    dx, dy = (x - cx) / 20.0, (y - cy) / 30.0
    if dx * dx + dy * dy > 1.0:
        return TRANSPARENT
    # A bite out of the right side. Two jobs: it is the asymmetry that makes a
    # mirrored sprite obvious, and it is the only thing here that puts *two*
    # posts in one column -- an oval gives every column a single run, so a
    # decoder that read one post per column and stopped would draw this
    # perfectly.
    if 30 <= x < 40 and 22 <= y < 34:
        return TRANSPARENT
    # Hoops, so a vertically squashed sprite is obvious.
    if y % 16 < 3:
        return _idx(7, 13)
    # A highlight down the left of the body, which is the other half of
    # telling a mirror: the lit side and the handle are on opposite sides.
    if x < cx - 6:
        return _idx(1, 11 - (y * 4 // SPRITE_H))
    return _idx(1, 7 - (y * 4 // SPRITE_H))


def lamp_pixel(x, y):
    """A column with a lit top: tall, thin, and nothing like the barrel."""
    if 20 <= x < 28:
        return _idx(0, 4 + (y * 6 // SPRITE_H))
    if y < 12 and 14 <= x < 34:
        return _idx(4, 15 - y)
    return TRANSPARENT


def barrel_b_pixel(x, y):
    """The barrel's second frame: the same drum with its hoops moved.

    **A second frame is not decoration here.** A barrel's cycle is `BAR1A` then
    `BAR1B`, and while this file shipped only the first, every barrel in the
    fixture went invisible for six tics in twelve -- the state machine advanced
    to a frame with no lump. FreeDoom could never have found that, because it
    has every frame anything asks for.

    Moved rather than redrawn, so the difference between the two frames is
    unmistakable on screen and is *only* the thing that should differ.
    """
    cx, cy = SPRITE_W / 2.0, SPRITE_H / 2.0
    dx, dy = (x - cx) / 20.0, (y - cy) / 30.0
    if dx * dx + dy * dy > 1.0:
        return TRANSPARENT
    if 30 <= x < 40 and 22 <= y < 34:
        return TRANSPARENT
    if (y + 8) % 16 < 3:
        return _idx(7, 13)
    if x < cx - 6:
        return _idx(1, 11 - (y * 4 // SPRITE_H))
    return _idx(1, 7 - (y * 4 // SPRITE_H))


def clip_pixel(x, y):
    """A magazine: small, squat, and sitting on the floor."""
    if 18 <= x < 30 and SPRITE_H - 14 <= y < SPRITE_H:
        return _idx(6, 6 + ((x - 18) * 6 // 12))
    return TRANSPARENT


def medi_pixel(x, y):
    """A box with a cross on it, and the cross is off-centre on purpose."""
    if not (14 <= x < 36 and SPRITE_H - 24 <= y < SPRITE_H):
        return TRANSPARENT
    # The cross, pushed left of the box's middle so a mirrored draw shows.
    if 18 <= x < 22 and SPRITE_H - 20 <= y < SPRITE_H - 8:
        return _idx(4, 12)
    if 16 <= x < 26 and SPRITE_H - 16 <= y < SPRITE_H - 12:
        return _idx(4, 12)
    return _idx(0, 6)


def key_pixel(bright):
    """A key card: a tall thin tag, in two brightnesses so it blinks.

    A key's cycle is `BKEYA` then `BKEYB` and the two differ only in how lit
    they are, which is what makes a key catch the eye on a dark map. Both are
    emitted, so this fixture exercises a two-frame cycle where *both* frames
    exist -- the barrel above covers the case where one does not.
    """

    def px(x, y):
        if 20 <= x < 26 and SPRITE_H - 30 <= y < SPRITE_H - 4:
            return _idx(11, 9 if bright else 13)
        # A notch on one side, so a mirror shows.
        if 26 <= x < 30 and SPRITE_H - 26 <= y < SPRITE_H - 22:
            return _idx(11, 9 if bright else 13)
        return TRANSPARENT

    return px


# A zombieman, in every frame its state chains reach.
#
# **Twenty-one frames, and all of them rotation 0.** A real zombieman has eight
# facings per frame and this has one seen from everywhere, which is wrong and
# is deliberately wrong: the eight-facing path is already exercised by FreeDoom
# (`doom mug POSS` draws all eight), and what the fixture is for is the *state
# machine* -- whether the thing looks, chases, shoots, flinches and dies. A
# frame that does not exist is a frame the object is dropped for, so the whole
# alphabet has to be here or the test is about nothing.
#
# The poses are crude and the *distinctions* are not: the walk cycle moves its
# legs, the attack frame puts an arm out and the one after it adds a muzzle
# flash, pain leans, and the two death chains sink. Anything that plays the
# wrong chain is visible rather than merely different.
ZOMBIE_FRAMES = "ABCDEFGHIJKLMNOPQRSTU"


def zombie_pixel(frame):
    """One frame of the zombieman, as a pixel function."""
    walk = "ABCD".find(frame)          # the four-step walk cycle
    fire = "EF".find(frame)            # arm out, then the flash
    hurt = frame == "G"
    die = "HIJKL".find(frame)          # ordinary death
    gib = "MNOPQRSTU".find(frame)      # the messy one

    # How far the body has collapsed, 0 upright to 1 flat.
    down = 0.0
    if die >= 0:
        down = (die + 1) / 5.0
    elif gib >= 0:
        down = (gib + 1) / 9.0

    def px(x, y):
        cx = SPRITE_W // 2
        floor = SPRITE_H - 1
        # Standing height shrinks as it falls.
        tall = int(44 * (1.0 - down))
        top = floor - tall
        if tall < 4:
            # A heap on the ground, and the gib chain leaves a wider one.
            wide = 16 if gib >= 0 else 10
            if floor - 4 <= y <= floor and abs(x - cx) < wide:
                return _idx(1, 4)
            return TRANSPARENT

        # Head.
        if top <= y < top + 8 and abs(x - cx) < 5:
            return _idx(7, 9 if not hurt else 12)
        # Torso.
        if top + 8 <= y < floor - 12 and abs(x - cx) < 7:
            # Leaning when in pain, which is the whole of that frame.
            lean = 3 if hurt else 0
            if abs(x - cx - lean) < 7:
                return _idx(2, 6 + (y % 3))
            return TRANSPARENT
        # Legs, alternating with the walk cycle.
        if floor - 12 <= y <= floor:
            step = 0 if walk < 0 else (walk % 2) * 3
            if abs(x - (cx - 4 + step)) < 3 or abs(x - (cx + 4 - step)) < 3:
                return _idx(3, 5)
            return TRANSPARENT
        # The arm, out to the right while attacking.
        if fire >= 0 and top + 10 <= y < top + 16 and cx + 6 <= x < cx + 18:
            return _idx(2, 8)
        # The flash, on the second attack frame only.
        if fire == 1 and top + 9 <= y < top + 17 and cx + 18 <= x < cx + 24:
            return _idx(4, 15)
        return TRANSPARENT

    return px


# `left` is half the width, so the sprite is centred on the thing. `top` is the
# full height, so the bottom edge sits on the floor.
SPRITES = [
    ("BAR1A0", barrel_pixel, SPRITE_W // 2, SPRITE_H),
    ("BAR1B0", barrel_b_pixel, SPRITE_W // 2, SPRITE_H),
    ("COLUA0", lamp_pixel, SPRITE_W // 2, SPRITE_H),
    ("CLIPA0", clip_pixel, SPRITE_W // 2, SPRITE_H),
    ("MEDIA0", medi_pixel, SPRITE_W // 2, SPRITE_H),
    ("BKEYA0", key_pixel(True), SPRITE_W // 2, SPRITE_H),
    ("BKEYB0", key_pixel(False), SPRITE_W // 2, SPRITE_H),
] + [
    ("POSS%s0" % f, zombie_pixel(f), SPRITE_W // 2, SPRITE_H) for f in ZOMBIE_FRAMES
]


def sprite_lumps():
    out = [("S_START", b"")]
    for n, fn, left, top in SPRITES:
        out.append((n, make_patch(SPRITE_W, SPRITE_H, fn, left, top)))
    out.append(("S_END", b""))
    return out


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
LINEDEFS = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (0, 5), (1, 4)]

# Linedef 6 is the divider, and it is the only two-sided line in the file.
#
# **It is here because without it half the renderer had never run.** The room
# was one sector bounded by six one-sided walls, so `back` was `None` at every
# seg and the entire portal branch -- the upper step, the lower step, the
# pegging rules for each, and the shut-door case -- was code no test had ever
# reached. A map that exercises only the easy half of a renderer is a map that
# reports the renderer works.
#
# It runs from vertex 1 (0, -512) to vertex 4 (0, 512), which are the two
# vertices already sitting on the BSP's partition, so the tree does not change:
# one node splitting x=0, one subsector either side, and now a sector either
# side too.
#
# **And it is a door.** Direction decides which sidedef is which: going north
# from (0,-512) to (0,512), the right of north is east, so the *right* sidedef
# faces x>0 -- the half the player stands in -- and the *left* faces x<0, the
# sector that opens.
#
# That orientation is not cosmetic and it is not a free choice. A manual door
# (special 1) operates on the line's **back** sector and refuses when the
# player is on the back side, because a door drawn the other way round would
# open the wall you are standing inside. This line ran south until doors
# existed, which put the player's own half on the back and made the door
# refuse from the only place you can reach it.
DIVIDER = 6

# DR Door Blue: open, wait, close -- repeatable, manual, untagged, and locked.
#
# It was special 1, the same door without the lock, and the lock is what makes
# the fixture test a *sequence* rather than a single rule: the player must walk
# over the key before the door will do anything. A run that opens it has proved
# the pickup fired, the key was recorded, and the door consulted it, and a run
# that does not says which of those failed -- the key count is in the report
# beside the door's height.
#
# 26 rather than 32 because 26 is the repeatable form. A once-only door that
# had already been opened would make every later run in a session meaningless.
DOOR_SPECIAL = 26

# Which sector each half is. 0 is the right (x>0), where the player stands.
RIGHT_SECTOR, LEFT_SECTOR = 0, 1

# Which linedefs bound each subsector, and for the divider, which way its seg
# runs. `(line, side)`, where side 1 means the seg runs against the linedef and
# therefore takes the *left* sidedef.
RIGHT_LINES = [(1, 0), (2, 0), (3, 0), (DIVIDER, 0)]   # x >= 0
LEFT_LINES = [(4, 0), (5, 0), (0, 0), (DIVIDER, 1)]    # x <= 0

# Which sector each one-sided line belongs to, by the half it is in.
LINE_SECTOR = {
    0: LEFT_SECTOR, 1: RIGHT_SECTOR, 2: RIGHT_SECTOR,
    3: RIGHT_SECTOR, 4: LEFT_SECTOR, 5: LEFT_SECTOR,
}


def bam(x0, y0, x1, y1):
    """A DOOM binary angle: a full turn is 65536, and east is zero."""
    a = math.atan2(y1 - y0, x1 - x0)
    return int(round(a / (2 * math.pi) * 65536)) & 0xFFFF


def build_vertexes():
    return b"".join(struct.pack("<hh", x, y) for x, y in VERTEXES)


# The player start, plus scenery to look at. Positions are inside the hexagon
# (radius 512) and off-axis, so a sprite that is drawn at the wrong world point
# does not land somewhere that happens to look plausible.
#
# 2035 is a barrel and 2028 a floor lamp, which are the doomednums every DOOM
# map uses for them -- the mapping from that number to a sprite name is the
# kernel's table, and using the real numbers is what makes the table testable
# with something other than itself.
THINGS = [
    # The player stands in the right half and faces *west*, at the divider, so
    # the first thing a `doom view` shows is the portal: a step up, a ceiling
    # coming down, and the far room's own light and flats through the gap.
    # Not at the origin any more -- the origin is exactly on the partition,
    # where which sector a point is in has no answer.
    (300, 0, 180, 1),
    # Two barrels close enough to set each other off, with the first of them
    # directly west of the player so a shot fired without turning hits it.
    #
    # 60 units apart, and the arithmetic is the point rather than the aesthetic:
    # a blast does `128 - (max(|dx|,|dy|) - radius)`, so at 60 with a radius of
    # 10 the second barrel takes 78 against 20 health and goes up too. A pair
    # far enough apart to survive would test a barrel that explodes and nothing
    # else, which is a chain reaction with one link.
    (100, 0, 0, 2035),      # shot directly
    (100, 60, 0, 2035),     # taken out by the first one's blast
    (-200, -150, 0, 2035),  # one in the far half, standing 32 higher
    (-260, 200, 0, 2028),   # a lamp in the far half
    # Three pickups on the straight line west from the player to the door, so
    # a run that holds one key walks over all of them in order. Each is here
    # for a different answer:
    #
    #   the clip     is taken           -- 50 bullets become 60
    #   the medikit  is *refused*       -- at full health it stays on the floor
    #   the key      is taken, and is the only way through the door
    #
    # The refusal is the one worth having. A pickup that always disappears
    # cannot tell a working rule from `return true`, and "the medikit is still
    # there" is a fact a headless run can report as a number.
    (250, 0, 0, 2007),      # a clip
    (200, 0, 0, 2012),      # a medikit, which full health will not take
    (150, 0, 0, 5),         # the blue key the door wants
    # A zombieman, placed where it can see the player and pointed at them.
    #
    # Both halves matter. `A_Look` wants sight *and* the player inside its
    # front 180 degrees, so a monster facing the wall would never notice
    # anybody and the test would report a monster that does nothing -- which
    # is exactly what a broken `A_Look` reports too. The bearing from here to
    # the player start is 45 degrees, so that is where it is pointed.
    (100, -200, 45, 3004)
]


def build_things():
    out = b""
    for (x, y, ang, kind) in THINGS:
        # Flag 7 is "on every skill level"; a thing with no skill flags is
        # invisible in play, which would make an absent sprite and a filtered
        # thing look exactly alike.
        out += struct.pack("<hhhhh", x, y, ang, kind, 7)
    return out


def build_linedefs():
    out = b""
    for i, (a, b) in enumerate(LINEDEFS):
        if i == DIVIDER:
            # Flag 4 is ML_TWOSIDED, special 1 is DR Door Open Wait Close --
            # the commonest line in DOOM and the one you press to open. Tag 0
            # because a manual door names no tag: it operates on the sector
            # behind the line it is written on.
            # Sidedef 6 is the right (facing the player's half), 7 the left.
            out += struct.pack("<HHHHHHH", a, b, 4, DOOR_SPECIAL, 0, 6, 7)
        else:
            # flag 1 is impassable, which every one-sided wall is.
            out += struct.pack("<HHHHHHH", a, b, 1, 0, 0, i, NO_SIDE)
    return out


def build_sidedefs():
    """One sidedef per one-sided line, then the divider's two.

    A two-sided line names nothing in its *middle* slot here. That is the
    masked mid-wall -- a grate or a railing -- and it is the one surface DOOM
    draws after what is behind it rather than before. Naming a texture there
    that the renderer then drew as solid would seal the opening, so it is left
    as `-` until there is something that can draw it properly.
    """
    out = b""
    for i, _ in enumerate(LINEDEFS):
        if i == DIVIDER:
            continue
        out += struct.pack("<hh", 0, 0)
        out += name8("-") + name8("-") + name8("STARTAN3")
        out += struct.pack("<H", LINE_SECTOR[i])
    # 6: the divider's right side, facing the right half where the player
    # stands. This is the door's face -- its *upper* is what you look at while
    # the door is shut, because a shut door is a ceiling lowered to the floor
    # and the whole opening is upper texture.
    out += struct.pack("<hh", 0, 0)
    out += name8("BIGWALL") + name8("STARTAN3") + name8("-")
    out += struct.pack("<H", RIGHT_SECTOR)
    # 7: its left side, inside the door sector.
    out += struct.pack("<hh", 0, 0)
    out += name8("BIGWALL") + name8("STARTAN3") + name8("-")
    out += struct.pack("<H", LEFT_SECTOR)
    return out


def build_sectors():
    """Two, differing in every field a renderer reads.

    Floor, ceiling, light and both pictures all change across the divider, so a
    renderer that took any of them from the wrong sector draws something
    visibly wrong rather than something subtly off. The flats are swapped
    rather than new: a ceiling that is the floor's picture is unmistakable, and
    it costs no more art.
    """
    out = struct.pack("<hh", 0, 128)
    out += name8("FLOOR4_8") + name8("CEIL3_5")
    out += struct.pack("<hhh", 160, 0, 0)
    # The far half is the **door**, and it starts shut: its ceiling is on its
    # floor, so the opening has no height at all and `P_LineOpening` refuses
    # to let anybody through.
    #
    # Its floor is 24 rather than 32 on purpose. Twenty-four is exactly DOOM's
    # step limit, so the moment the door opens the player can walk in -- at 32
    # the door would open onto a step too high to climb, and a test that opened
    # the door correctly would still report the player stuck, which is the
    # worst kind of fixture.
    out += struct.pack("<hh", 24, 24)
    out += name8("CEIL3_5") + name8("FLOOR4_8")
    out += struct.pack("<hhh", 240, 0, 0)
    return out


def build_segs():
    """One seg per wall, grouped by subsector so SSECTORS can name a run.

    No minisegs along the partition -- see the module docstring. Order matters:
    a subsector names its segs as a first index and a count, so the segs of one
    subsector have to be contiguous.
    """
    out = b""
    for group in (RIGHT_LINES, LEFT_LINES):
        for (li, side) in group:
            a, b = LINEDEFS[li]
            if side == 1:
                # The seg runs against its linedef, so its vertices are
                # reversed and it takes the *left* sidedef. This is how one
                # two-sided line is a wall in two subsectors at once, seen from
                # opposite directions.
                a, b = b, a
            (x0, y0), (x1, y1) = VERTEXES[a], VERTEXES[b]
            out += struct.pack("<HHHHHh", a, b, bam(x0, y0, x1, y1), li, side, 0)
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
        + flat_lumps()
        + sprite_lumps()
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

    # The portal. Without a two-sided line the whole `back` branch of the
    # renderer is unreachable, so what is asserted is not just that the line
    # exists but that it *steps*: same heights either side and the upper and
    # lower are zero pixels tall, which draws identically to no portal at all.
    two = [i for i in range(nlines)
           if struct.unpack("<HHHHHHH", by["LINEDEFS"][i * 14:i * 14 + 14])[6] != NO_SIDE]
    if len(two) != 1:
        raise ValueError(f"expected exactly one two-sided line, found {len(two)}")
    (_a, _b, fl, _sp, _tg, r, l) = struct.unpack(
        "<HHHHHHH", by["LINEDEFS"][two[0] * 14:two[0] * 14 + 14])
    if not fl & 4:
        raise ValueError("the two-sided line does not carry ML_TWOSIDED")
    secs = []
    for sd in (r, l):
        e = sd * SIZES["SIDEDEFS"] + 28
        secs.append(struct.unpack("<H", by["SIDEDEFS"][e:e + 2])[0])
    if secs[0] == secs[1]:
        raise ValueError("both sides of the portal name the same sector")
    hts = []
    for sc in secs:
        e = sc * SIZES["SECTORS"]
        hts.append(struct.unpack("<hh", by["SECTORS"][e:e + 4]))
    if hts[0][0] == hts[1][0]:
        raise ValueError("the portal has no step: both floors are equal")
    if hts[0][1] == hts[1][1]:
        raise ValueError("the portal has no header: both ceilings are equal")
    # The door has to start shut, or the fixture tests an opening rather than
    # an opening *door*: a sector whose ceiling is already above its floor
    # would let the player through before anything was triggered, and the run
    # would pass whether or not a single special ever fired.
    door_special = struct.unpack("<H", by["LINEDEFS"][two[0] * 14 + 6:two[0] * 14 + 8])[0]
    if fl & 4 and door_special in (1, 26, 27, 28, 32, 33, 34):
        shut = [h for h in hts if h[0] == h[1]]
        if not shut:
            raise ValueError("the door sector does not start shut")
    if door_special != DOOR_SPECIAL:
        raise ValueError(
            f"the two-sided line carries special {door_special}, not {DOOR_SPECIAL}"
        )

    # A locked door needs its key on the map, and the key has to be somewhere a
    # run can reach *before* the door. Both halves are asserted, because either
    # one alone makes the fixture pass for the wrong reason: a key nobody can
    # reach turns the door test into a test that doors refuse, and a key with
    # no door turns the pickup test into a test that nothing crashes.
    LOCK_KEYS = {26: 5, 32: 5, 27: 6, 34: 6, 28: 13, 33: 13}
    if door_special in LOCK_KEYS:
        want = LOCK_KEYS[door_special]
        things = []
        for i in range(len(by["THINGS"]) // 10):
            x, y, ang, kind, flags = struct.unpack_from("<5h", by["THINGS"], i * 10)
            things.append((x, y, kind))
        start = [t for t in things if t[2] == 1]
        if len(start) != 1:
            raise ValueError(f"expected one player start, found {len(start)}")
        sx, sy, _ = start[0]
        keys = [t for t in things if t[2] == want]
        if not keys:
            raise ValueError(
                f"door special {door_special} wants doomednum {want} and no thing is one"
            )
        # The door is the partition, at x = 0, and the player starts east of it
        # facing west. So "on the way" is: the same row, and between the two.
        for kx, ky, _ in keys:
            if ky != sy or not (0 < kx < sx):
                raise ValueError(
                    f"the key at {kx},{ky} is not on the straight line from "
                    f"{sx},{sy} to the door, so holding one movement key "
                    f"would not walk over it"
                )

    # The chain reaction the shooting test depends on. Two barrels in the near
    # half, one of them on the line the player fires along, and close enough
    # that the first one's blast kills the second. All three are properties of
    # where they were put, so all three are asserted -- moving a barrel is a
    # one-character edit that would turn the test into one that passes because
    # nothing happened.
    BARREL, BLAST, BARREL_R, BARREL_HP = 2035, 128, 10, 20
    barrels = []
    for i in range(len(by["THINGS"]) // 10):
        x, y, _a, kind, _f = struct.unpack_from("<5h", by["THINGS"], i * 10)
        if kind == BARREL:
            barrels.append((x, y))
    start = [
        struct.unpack_from("<5h", by["THINGS"], i * 10)[:2]
        for i in range(len(by["THINGS"]) // 10)
        if struct.unpack_from("<5h", by["THINGS"], i * 10)[3] == 1
    ][0]
    online = [b for b in barrels if b[1] == start[1] and 0 < b[0] < start[0]]
    if not online:
        raise ValueError(
            f"no barrel on the line west from {start}, so a shot without a turn hits nothing"
        )
    chained = False
    for bx, by_ in barrels:
        for cx, cy in barrels:
            if (bx, by_) == (cx, cy):
                continue
            gap = max(abs(bx - cx), abs(by_ - cy)) - BARREL_R
            if BLAST - gap > BARREL_HP:
                chained = True
    if not chained:
        raise ValueError("no two barrels are close enough for one to set off the other")

    # A monster, and every frame its state chains can reach. The first frame
    # alone is not enough: a monster whose *death* frames are missing dies into
    # nothing, and one whose walk frames are missing stops being drawn the
    # moment it notices you -- both of which look like the AI failing rather
    # than like a WAD that is short of a lump.
    ZOMBIE = 3004
    monsters = [
        i for i in range(len(by["THINGS"]) // 10)
        if struct.unpack_from("<5h", by["THINGS"], i * 10)[3] == ZOMBIE
    ]
    if not monsters:
        raise ValueError("no monster on the map, so nothing can be chased by anything")
    have = {n for n, _ in lumps}
    for f in ZOMBIE_FRAMES:
        want = "POSS%s0" % f
        if want not in have:
            raise ValueError(f"the zombieman needs {want} and this WAD has no such lump")
    # And it has to be able to see the player, or it never wakes. Both are in
    # the near half, which is one convex space, so line of sight is the sign
    # of x and nothing more.
    mx, my, _mk = struct.unpack_from("<5h", by["THINGS"], monsters[0] * 10)[:3]
    if mx <= 0 or sx <= 0:
        raise ValueError(
            f"the monster at {mx},{my} and the player at {sx},{sy} are not in the "
            f"same half, so the divider stands between them"
        )

    # Every pickup placed must have the sprite lump its first frame wants, or
    # the thing is dropped at level load and the run reports nothing rather
    # than reporting a failure. The table is small and explicit on purpose:
    # deriving it would mean carrying a copy of `mobjinfo` in this file.
    FIRST_FRAME = {
        2007: "CLIPA0", 2012: "MEDIA0", 5: "BKEYA0",
        2035: "BAR1A0", 2028: "COLUA0", 3004: "POSSA0",
    }
    have = {n for n, _ in lumps}
    for i in range(len(by["THINGS"]) // 10):
        _x, _y, _a, kind, _f = struct.unpack_from("<5h", by["THINGS"], i * 10)
        want = FIRST_FRAME.get(kind)
        if want and want not in have:
            raise ValueError(f"thing {kind} needs sprite {want}, which is not in this WAD")

    # Flats, which have no header to check -- so what is checked is the one
    # property a reader depends on: exactly 4096 bytes, inside the namespace,
    # and named by the sector that wants them.
    inside = False
    seen = set()
    for ln, body in lumps:
        if ln == "F_START":
            inside = True
            continue
        if ln == "F_END":
            inside = False
            continue
        if inside:
            if len(body) != 4096:
                raise ValueError(f"flat {ln} is {len(body)} bytes, not 64x64")
            seen.add(ln)
    if seen:
        (_fh, _ch, fpic, cpic) = (
            struct.unpack("<hh", by["SECTORS"][:4])
            + (
                by["SECTORS"][4:12].rstrip(b"\x00").decode("ascii", "replace"),
                by["SECTORS"][12:20].rstrip(b"\x00").decode("ascii", "replace"),
            )
        )
        for want in (fpic, cpic):
            if want not in seen:
                raise ValueError(f"the sector names flat {want}, which is not in F_START..F_END")
        # A flat that is the same in both axes cannot catch a transposed
        # lookup, which is the mistake this file exists to make visible.
        for ln, body in lumps:
            if ln in seen:
                flipped = bytes(
                    body[x * 64 + y] for y in range(64) for x in range(64)
                )
                if flipped == body:
                    raise ValueError(f"flat {ln} is symmetric, so a transpose would not show")
                mirrored = bytes(
                    body[(63 - y) * 64 + x] for y in range(64) for x in range(64)
                )
                if mirrored == body:
                    raise ValueError(f"flat {ln} is mirror-symmetric, so a flip would not show")

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

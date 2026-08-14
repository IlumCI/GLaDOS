#!/usr/bin/env python3
"""Draw the GLaDOS aperture mark for the web, from the kernel's own geometry.

This is a port of `gfx::splash::aperture`, deliberately step for step, so the
favicon and the boot screen are the same mark rather than two drawings that
resemble each other. If the kernel's version changes, this one is wrong and
should be re-ported.

What the mark is, and the three ways to get it wrong
---------------------------------------------------
It is a **solid disc with wedges cut out of it** -- not blades drawn onto a
background, and not a ring with spokes.

  * Stroking the blade edges as lines gives a wireframe, and full chords
    between evenly spaced points always make a star.
  * Leaving the middle solid gives a flower: six petals around a hub, rather
    than six blades around a hole. The middle is *open*, and its corners are
    where consecutive blade edges meet, so the opening is a hexagon.
  * Fanning each slash out into a triangle wide enough to read at the rim
    makes it far too wide where it meets the opening, and it eats the blades.
    A slash is a constant-width quadrilateral.

The sweep is what sells it: each slash starts at a corner of the hexagonal
opening and runs to the rim 45 degrees ahead of that corner, so the blades left
over appear to spiral, exactly as a camera iris does.

No image library, for the same reason there is no ISO tool here: PNG is a zlib
stream of filtered scanlines in four CRC'd chunks, and zlib is in the standard
library. Antialiasing is 4x4 supersampling, because a hard-edged rasteriser
makes this look like a staircase at 32 pixels -- which is the size at which a
favicon is judged.

Usage:
    mklogo.py [--out docs/img]
"""

import argparse
import binascii
import math
import struct
import zlib
from pathlib import Path

AMBER = (0xF2, 0x8C, 0x1E)
DARK = (0x0E, 0x0E, 0x0E)

BLADES = 7
# Fractions of the disc radius.
R_IN = 0.46             # the opening: large, and the thing every wrong version shrinks
SLASH = 0.035           # half-width of the cut between blades
DEG = 360.0 / BLADES


def unit(a):
    return math.cos(a), math.sin(a)


def in_opening(x, y, cx, cy, rin):
    """True inside the central opening.

    The opening is the polygon bounded by the same tangent lines the cuts run
    along -- the intersection of seven half-planes -- not a circle. A circular
    hole leaves a nub where each straight cut meets the curve; the polygon is
    what gives the blades their sharp points, because each blade's inner edge
    *is* one of those lines.
    """
    for i in range(BLADES):
        ux, uy = unit(math.radians(i * DEG))
        if (x - cx) * ux + (y - cy) * uy > rin:
            return False
    return True


def in_tri(p, a, b, c):
    """Point-in-triangle by consistent sign of the three edge cross products."""
    def side(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = side(p, a, b), side(p, b, c), side(p, c, a)
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))


def slashes(cx, cy, r):
    """The cuts between blades, as triangle pairs.

    Each cut lies along a line **tangent to the central opening**, running from
    its tangent point out to the rim. That tangency is the whole mark: a cut
    aimed radially just notches the disc, while a tangent one leaves a blade
    whose inner edge is a straight chord, which is what makes seven of them
    read as a spiral. Every earlier attempt here got this wrong.
    """
    tris = []
    rin = r * R_IN
    half = r * SLASH
    reach = math.sqrt(max(0.0, r * r - rin * rin)) * 1.06   # past the rim, so it cuts clean
    for i in range(BLADES):
        a = math.radians(i * DEG)
        ux, uy = unit(a)
        px, py = cx + rin * ux, cy + rin * uy       # tangent point on the opening
        tx, ty = -uy, ux                            # tangent direction
        ex, ey = px + tx * reach, py + ty * reach   # out at the rim
        # Constant width: offset both ends along the normal, which here is the
        # radial direction.
        ox, oy = ux * half, uy * half
        a0, a1 = (px + ox, py + oy), (px - ox, py - oy)
        b0, b1 = (ex + ox, ey + oy), (ex - ox, ey - oy)
        tris.append((a0, a1, b1))
        tris.append((a0, b1, b0))
    return tris


def render(size, bg=None, ss=4):
    """RGBA buffer of the mark at `size` pixels, supersampled `ss`x."""
    cx = cy = size / 2.0
    r = size / 2.0 * 0.96
    rin = r * R_IN
    tris = slashes(cx, cy, r)
    px = bytearray(size * size * 4)
    inv = 1.0 / (ss * ss)

    for py_ in range(size):
        for px_ in range(size):
            hits = 0
            for sy in range(ss):
                for sx in range(ss):
                    x = px_ + (sx + 0.5) / ss
                    y = py_ + (sy + 0.5) / ss
                    if math.hypot(x - cx, y - cy) > r:
                        continue
                    if in_opening(x, y, cx, cy, rin):
                        continue
                    if any(in_tri((x, y), *t) for t in tris):
                        continue
                    hits += 1
            i = (py_ * size + px_) * 4
            a = hits * inv * 255
            if a > 0:
                px[i], px[i + 1], px[i + 2] = AMBER
            px[i + 3] = int(a)

    if bg:
        flat = bytearray(size * size * 4)
        for i in range(0, len(px), 4):
            al = px[i + 3] / 255.0
            for k in range(3):
                flat[i + k] = int(px[i + k] * al + bg[k] * (1 - al))
            flat[i + 3] = 255
        return flat
    return px


def png(path, w, h, rgba):
    stride = w * 4
    raw = b"".join(b"\x00" + bytes(rgba[y * stride:(y + 1) * stride]) for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b""))


def svg():
    """The same construction as vectors: outer disc, then the opening and the
    seven cuts punched through it with fill-rule evenodd."""
    cx = cy = 50.0
    r = 48.0
    rin = r * R_IN
    d = [f"M {cx - r},{cy} a {r},{r} 0 1,0 {2 * r},0 a {r},{r} 0 1,0 {-2 * r},0"]
    # Vertices of the opening: where consecutive tangent lines meet. For a
    # regular polygon of inradius rin that is circumradius rin/cos(pi/n).
    circ = rin / math.cos(math.pi / BLADES)
    pts = []
    for i in range(BLADES):
        a = math.radians(i * DEG + DEG / 2)
        pts.append(f"{cx + circ * math.cos(a):.2f},{cy + circ * math.sin(a):.2f}")
    d.append("M " + " L ".join(pts) + " Z")
    for t_ in slashes(cx, cy, r):
        d.append("M {} L {} L {} Z".format(
            *[f"{x:.2f},{y:.2f}" for x, y in t_]))
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" '
            'role="img" aria-label="GLaDOS aperture mark">'
            f'<path d="{" ".join(d)}" '
            f'fill="#{AMBER[0]:02X}{AMBER[1]:02X}{AMBER[2]:02X}" '
            'fill-rule="evenodd"/></svg>')


def social(w, h, out):
    """Open Graph card: the mark on a dark field. No text -- rendering a
    wordmark means embedding a font, and every platform that shows this image
    shows the page title beside it."""
    px = bytearray(w * h * 4)
    for i in range(0, len(px), 4):
        px[i], px[i + 1], px[i + 2], px[i + 3] = DARK[0], DARK[1], DARK[2], 255
    size = int(h * 0.66)
    m = render(size, bg=DARK)
    ox, oy = (w - size) // 2, (h - size) // 2
    for y in range(size):
        d0 = ((oy + y) * w + ox) * 4
        s0 = (y * size) * 4
        px[d0:d0 + size * 4] = m[s0:s0 + size * 4]
    png(out, w, h, px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/img")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "logo.svg").write_text(svg(), encoding="utf-8")
    made = ["logo.svg"]
    for s in (32, 180, 192, 512):
        png(out / f"icon-{s}.png", s, s, render(s))
        made.append(f"icon-{s}.png")
    social(1200, 630, out / "og.png")
    made.append("og.png")
    for f in made:
        print(f"  {f}  {(out / f).stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()

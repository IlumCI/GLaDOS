#!/usr/bin/env python3
"""Draw the GLaDOS aperture mark for the web.

This and `gfx::splash::aperture` in the kernel are the same construction, in
integer arithmetic there and floating point here. The kernel was ported from
this file, so if the geometry changes it changes here first and the constants
in splash.rs (BLADE_DIR, OPEN_DIR, OPEN_PCT, CUT_PCT) follow. Two drawings that
merely resemble each other is the thing being avoided: the boot screen, the
desktop wallpaper and the favicon are one mark.

What the mark is, and the ways it has been got wrong
---------------------------------------------------
A solid disc with wedges cut out of it: eight blades around a large open
middle. Not blades drawn onto a background, and not a ring with spokes.

  * Stroking the blade edges as lines gives a wireframe, and full chords
    between evenly spaced points always make a star.
  * Leaving the middle solid gives a flower, petals around a hub rather than
    blades around a hole. The middle is open, and largely so: the opening is
    46% of the radius.
  * Cutting radially. This was the long-lived one, and it survived three
    attempts. A cut aimed out from the centre only notches the disc, and the
    result reads as a wheel. The cuts are **tangent to the opening**, so each
    blade's inner edge is a straight chord and the blades left over appear to
    spiral. That tangency is the whole mark; without it the blade count hardly
    matters.
  * A circular opening. It leaves a nub where each straight cut meets the
    curve. The opening is the polygon bounded by those same tangent lines,
    which is what gives the blades their points.

No image library, for the same reason there is no ISO tool here: PNG is a zlib
stream of filtered scanlines in four CRC'd chunks, and zlib is in the standard
library. Antialiasing is 4x4 supersampling, because a hard-edged rasteriser
makes this a staircase at 32 pixels, which is the size a favicon is judged at.

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

BLADES = 8
# Fractions of the disc radius. The opening is a regular octagon whose
# vertices sit at 20/48 of the radius, which puts its inradius here.
R_VERT = 20.0 / 48.0
R_IN = R_VERT * math.cos(math.pi / BLADES)
SLASH = 1.6 / 48.0      # half-width of the cut between blades
DEG = 360.0 / BLADES


def unit(a):
    return math.cos(a), math.sin(a)


def in_opening(x, y, cx, cy, rin):
    """True inside the central opening: the octagon bounded by the same lines
    the cuts run along, so each blade's inner edge is one of them."""
    for i in range(BLADES):
        ux, uy = unit(math.radians(i * DEG))
        if (x - cx) * ux + (y - cy) * uy > rin:
            return False
    return True


def vertices(cx, cy, r):
    rv = r * R_VERT
    return [(cx + rv * math.cos(math.radians(DEG / 2 + i * DEG)),
             cy + rv * math.sin(math.radians(DEG / 2 + i * DEG)))
            for i in range(BLADES)]


def cuts(cx, cy, r):
    """The cuts between blades, one per octagon edge.

    Cut k starts at vertex k, runs along the edge through vertex k+1 and on
    out past the rim. It starts *at* the vertex with a square end and never
    behind it -- the first version of this mark ran each cut as two
    overlapping triangles under an even-odd fill, and every overlap punched
    the colour back in, which is what made it look messy.
    """
    v = vertices(cx, cy, r)
    out = []
    for k in range(BLADES):
        a, b = v[k], v[(k + 1) % BLADES]
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy)
        out.append((a, (dx / n, dy / n)))
    return out


def in_cut(x, y, cs, half):
    for (ax, ay), (dx, dy) in cs:
        t = (x - ax) * dx + (y - ay) * dy
        if t >= 0 and abs((x - ax) * dy - (y - ay) * dx) < half:
            return True
    return False


def render(size, bg=None, ss=4):
    """RGBA buffer of the mark at `size` pixels, supersampled `ss`x."""
    cx = cy = size / 2.0
    r = size / 2.0 * 0.96
    rin = r * R_IN
    cs = cuts(cx, cy, r)
    half = r * SLASH
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
                    if in_cut(x, y, cs, half):
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
    """The same construction as vectors: a disc under a mask that punches out
    the octagon and the eight cuts. A mask rather than an even-odd path,
    because an even-odd path fills wherever two cuts overlap."""
    cx = cy = 50.0
    r = 48.0
    v = vertices(cx, cy, r)
    poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in v)
    lines = []
    for (ax, ay), (dx, dy) in cuts(cx, cy, r):
        lines.append(f'<line x1="{ax:.2f}" y1="{ay:.2f}" '
                     f'x2="{ax + dx * 80:.2f}" y2="{ay + dy * 80:.2f}"/>')
    fill = f"#{AMBER[0]:02X}{AMBER[1]:02X}{AMBER[2]:02X}"
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" '
            'role="img" aria-label="GLaDOS aperture mark">'
            '<defs><mask id="m"><rect width="100" height="100" fill="#fff"/>'
            f'<polygon points="{poly}" fill="#000"/>'
            f'<g stroke="#000" stroke-width="{2 * r * SLASH:.1f}" stroke-linecap="butt">'
            + "".join(lines) + '</g></mask></defs>'
            f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" fill="{fill}" mask="url(#m)"/></svg>')


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

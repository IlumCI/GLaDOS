#!/usr/bin/env python3
"""The desktop's fonts: subset, kerning flattened, and every byte accounted for.

1.4.1 replaces the 8x8 bitmap font with outline fonts rasterised in the kernel
by `fontdue`. This builds the files `include_bytes!` embeds, into
`assets/fonts/`, from sources whose digests are pinned below -- so a font in the
tree can always be traced back to the bytes it was made from, and a rebuild
from different bytes refuses rather than quietly producing a different face.

### Two fonts, two different licence positions

**Noto Sans** (Regular and SemiBold) is OFL-1.1 with *no* Reserved Font Name,
so a subset may keep its name. It is subset here: the full face is 620 KB per
weight and most of it is scripts the desktop never draws.

**Cascadia Code** is OFL-1.1 *with* a Reserved Font Name. Under the OFL a subset
is a Modified Version, and a Modified Version may not be distributed under the
reserved name -- so it is copied **unmodified**, at the cost of 598 KB, rather
than subset and renamed. Renaming a font to save half a megabyte is a licence
question to decide on purpose, not a side effect of a build script.

### Kerning is flattened here, and why

Noto Sans keeps all of its kerning in GPOS pair-positioning lookups and has no
legacy `kern` table, and neither fontdue nor ab_glyph reads GPOS -- measured:
both answered zero for `AV`, `To` and `Wa`. A GPOS parser in the kernel would be
the largest single piece of the text stack for one table. So the pairs that
matter are resolved *here*, by fontTools, into a table the kernel reads with a
binary search (`GLADOSK1`, below), and GSUB/GPOS are then dropped from the
subset since nothing reads them.

Resolution follows OpenType: within one lookup the first subtable that
*lists* the pair decides it -- a format-2 subtable whose second class is 0
does not, and resolution moves on -- and separate lookups add.

**Checked against HarfBuzz rather than against itself.** 20,008 pairs over
the kerned ranges (20,000 drawn at random, seed 1, plus `AV`, `To` and the
like), shaped by `uharfbuzz` against the *unsubset* source with ligatures
off: 535 of them kern, and the table disagrees with HarfBuzz on **none**. The
first version stopped at the first subtable whose coverage held the left
glyph, answered zero for `AV`, `To` and `Wa`, and kept 3,987 pairs where the
right answer is 8,155 -- a table that reads fine and kerns half the text.

### GLADOSK1

    0   8   magic  b"GLADOSK1"
    8   2   units per em, u16 little-endian
    10  2   reserved, zero
    12  4   pair count, u32
    16  ..  pairs, each: left u16, right u16 (codepoints, BMP), value i16 (font
            units), sorted by (left, right) so the kernel can binary-search

Usage:
    python3 tools/fontsubset.py build [--cascadia DIR] [--noto-ofl FILE]
    python3 tools/fontsubset.py check        # outputs against fonts.txt
    python3 tools/fontsubset.py --selftest
"""

import hashlib
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "fonts"

# Where the inputs come from and what they must be. A source that does not
# match its digest is refused: a different Noto release would subset to a
# different face, and the kerning table would describe glyphs it no longer has.
SOURCES = {
    "NotoSans-Regular": ("/usr/share/fonts/noto/NotoSans-Regular.ttf",
                         "478c558ea716033cd60c03438f628dfa75694dcf6b5f6d505a2f05fd2b4f3823"),
    "NotoSans-SemiBold": ("/usr/share/fonts/noto/NotoSans-SemiBold.ttf",
                          "a4e91fd530ac2b4ef5367240144ff37d7d65d66cf76f2e9a2187b93c676f92d0"),
    # microsoft/cascadia-code, release v2407.24, CascadiaCode-2407.24.zip
    # (sha256 e67a68ee...8474), ttf/static/CascadiaCode-Regular.ttf
    "CascadiaCode-Regular": ("ttf/static/CascadiaCode-Regular.ttf",
                             "c33ef522cdfeff99907fb54f3e97152bb18bfb9b56ec2fef4d4ceec51c8974a4"),
}
LICENCES = {
    # notofonts/latin-greek-cyrillic, OFL.txt
    "OFL-NotoSans.txt": "cee9892f9f0cc8fe882c9e9537ee6a89621d86ee7ceaf70b02e2b2b1c25c061a",
    # microsoft/cascadia-code v2407.24, LICENSE
    "OFL-CascadiaCode.txt": "51882cd3cdba4e16f220f44ddb08a635c38c44ea6e0975db2574f4be6f958238",
}

# What the interface draws. Latin with both extensions, Greek, Cyrillic, the
# general punctuation block (dashes, quotes, ellipsis), currency, letterlike
# symbols, arrows, a handful of mathematical operators. Box drawing is the
# terminal's and lives in Cascadia Code, which is not subset at all.
RANGES = [
    (0x0020, 0x007E), (0x00A0, 0x00FF), (0x0100, 0x024F),
    (0x0370, 0x03FF), (0x0400, 0x04FF),
    (0x2000, 0x206F), (0x20A0, 0x20CF), (0x2100, 0x214F),
    (0x2190, 0x21FF), (0x2200, 0x22FF), (0x25A0, 0x25FF),
]
# The characters kerning is resolved over: letters, digits and the punctuation
# that sits beside them. Every pair among these that the font kerns is kept.
KERN_RANGES = [(0x0020, 0x007E), (0x00A0, 0x00FF), (0x0100, 0x017F),
               (0x0370, 0x03FF), (0x0400, 0x045F), (0x2010, 0x2027)]

MAGIC = b"GLADOSK1"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def codepoints(ranges):
    out = []
    for a, b in ranges:
        out.extend(range(a, b + 1))
    return out


def gpos_pairs(font, chars):
    """Every non-zero horizontal kern between two of `chars`, in font units."""
    cmap = font.getBestCmap()
    glyph_of = {c: cmap[c] for c in chars if c in cmap}
    if "GPOS" not in font:
        return {}
    gpos = font["GPOS"].table
    kern_lookups = set()
    for fr in gpos.FeatureList.FeatureRecord:
        if fr.FeatureTag == "kern":
            kern_lookups.update(fr.Feature.LookupListIndex)
    # glyph -> chars, since several codepoints can share a glyph.
    chars_of = {}
    for c, g in glyph_of.items():
        chars_of.setdefault(g, []).append(c)
    glyphs = list(chars_of)
    total = {}
    for li in sorted(kern_lookups):
        lookup = gpos.LookupList.Lookup[li]
        subtables = []
        for st in lookup.SubTable:
            if lookup.LookupType == 9:
                st = st.ExtSubTable
            if getattr(st, "LookupType", 2) == 2 or lookup.LookupType in (2, 9):
                subtables.append(st)
        decided = {}
        for g1 in glyphs:
            for g2 in glyphs:
                for st in subtables:
                    cov = st.Coverage.glyphs
                    if g1 not in cov:
                        continue
                    # A subtable decides a pair only when it actually lists
                    # it: a PairValueRecord for format 1, a second glyph in a
                    # class other than 0 for format 2. Otherwise resolution
                    # moves on to the next subtable -- stopping at the first
                    # whose coverage merely held the left glyph answered zero
                    # for AV, To and Wa, which Noto plainly kerns.
                    if st.Format == 1:
                        ps = st.PairSet[cov.index(g1)]
                        rec = next((r for r in ps.PairValueRecord if r.SecondGlyph == g2), None)
                        if rec is None:
                            continue
                    elif st.Format == 2:
                        c2 = st.ClassDef2.classDefs.get(g2, 0)
                        if c2 == 0:
                            continue
                        c1 = st.ClassDef1.classDefs.get(g1, 0)
                        rec = st.Class1Record[c1].Class2Record[c2]
                    else:
                        continue
                    v = getattr(rec.Value1, "XAdvance", 0) if rec.Value1 else 0
                    decided[(g1, g2)] = v or 0
                    break
        for k, v in decided.items():
            total[k] = total.get(k, 0) + v
    pairs = {}
    for (g1, g2), v in total.items():
        if v == 0:
            continue
        for c1 in chars_of[g1]:
            for c2 in chars_of[g2]:
                pairs[(c1, c2)] = v
    return pairs


def pack_kern(pairs, upem):
    out = bytearray(MAGIC)
    out += struct.pack("<HHI", upem, 0, len(pairs))
    for (a, b) in sorted(pairs):
        if a > 0xFFFF or b > 0xFFFF:
            continue
        out += struct.pack("<HHh", a, b, pairs[(a, b)])
    return bytes(out)


def unpack_kern(data):
    if data[:8] != MAGIC:
        raise ValueError("not a GLADOSK1 table")
    upem, _, n = struct.unpack_from("<HHI", data, 8)
    # Before reading a single pair: the count is a claim the file makes about
    # itself, and a short file is exactly the case where it is false.
    if 16 + 6 * n != len(data):
        raise ValueError("length does not match the count")
    pairs = {}
    for i in range(n):
        a, b, v = struct.unpack_from("<HHh", data, 16 + 6 * i)
        pairs[(a, b)] = v
    return upem, pairs


def subset(src, dst):
    from fontTools import subset as fs
    from fontTools.ttLib import TTFont
    opts = fs.Options()
    opts.layout_features = []
    opts.drop_tables += ["GPOS", "GSUB", "GDEF", "kern", "DSIG", "STAT"]
    opts.hinting = False
    opts.name_IDs = ["*"]
    opts.name_languages = [0x409]
    opts.notdef_outline = True
    opts.recalc_bounds = True
    font = TTFont(src)
    sub = fs.Subsetter(opts)
    sub.populate(unicodes=codepoints(RANGES))
    sub.subset(font)
    font.save(dst)


def build(argv):
    from fontTools.ttLib import TTFont
    cascadia = Path(argv[argv.index("--cascadia") + 1]) if "--cascadia" in argv else None
    noto_ofl = Path(argv[argv.index("--noto-ofl") + 1]) if "--noto-ofl" in argv else None
    if cascadia is None or noto_ofl is None:
        raise SystemExit("build needs --cascadia <unzipped release dir> and --noto-ofl <OFL.txt>")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []

    def src_path(name):
        p = Path(SOURCES[name][0])
        return p if p.is_absolute() else cascadia / p

    for name, (_, want) in SOURCES.items():
        got = sha256(src_path(name))
        if got != want:
            raise SystemExit(f"{name}: source digest {got[:16]} is not the pinned {want[:16]}")
    for lic, want in LICENCES.items():
        p = noto_ofl if lic == "OFL-NotoSans.txt" else cascadia / "LICENSE"
        if sha256(p) != want:
            raise SystemExit(f"{lic}: digest does not match the pinned one")
        (OUT / lic).write_bytes(p.read_bytes())
        manifest.append((lic, want, "licence, copied"))

    kern_chars = codepoints(KERN_RANGES)
    for name in ("NotoSans-Regular", "NotoSans-SemiBold"):
        src = src_path(name)
        dst = OUT / f"{name}.ttf"
        subset(src, dst)
        f = TTFont(src)
        pairs = gpos_pairs(f, kern_chars)
        table = pack_kern(pairs, f["head"].unitsPerEm)
        (OUT / f"{name}.kern").write_bytes(table)
        manifest.append((f"{name}.ttf", sha256(dst), f"subset of {SOURCES[name][1][:16]}"))
        manifest.append((f"{name}.kern", sha256(OUT / f"{name}.kern"),
                         f"{len(pairs)} pairs flattened from GPOS"))
        print(f"  {name}: {src.stat().st_size} -> {dst.stat().st_size} B, "
              f"{len(pairs)} kern pairs ({len(table)} B)")
    cc = src_path("CascadiaCode-Regular")
    (OUT / "CascadiaCode-Regular.ttf").write_bytes(cc.read_bytes())
    manifest.append(("CascadiaCode-Regular.ttf", SOURCES["CascadiaCode-Regular"][1], "unmodified"))
    print(f"  CascadiaCode-Regular: {cc.stat().st_size} B, copied unmodified")

    lines = ["# assets/fonts, written by tools/fontsubset.py -- name sha256 note"]
    lines += [f"{n} {d} {note}" for n, d, note in manifest]
    (OUT / "fonts.txt").write_text("\n".join(lines) + "\n")
    print(f"  wrote {OUT / 'fonts.txt'}")


def check(_argv):
    bad = 0
    for line in (OUT / "fonts.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, digest = line.split()[:2]
        got = sha256(OUT / name)
        ok = got == digest
        bad += not ok
        print(f"  {'ok ' if ok else 'BAD'}  {name}")
    for k in OUT.glob("*.kern"):
        upem, pairs = unpack_kern(k.read_bytes())
        print(f"  ok   {k.name} reads back: {len(pairs)} pairs at {upem} units per em")
    raise SystemExit(1 if bad else 0)


def selftest():
    ok = True

    def claim(what, good):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'ok ' if good else 'FAIL'}  {what}")

    pairs = {(0x41, 0x56): -80, (0x54, 0x6F): -120, (0x41, 0x54): -30}
    data = pack_kern(pairs, 1000)
    upem, back = unpack_kern(data)
    claim("a kern table round-trips", back == pairs and upem == 1000)
    order = [struct.unpack_from("<HH", data, 16 + 6 * i) for i in range(len(pairs))]
    claim("pairs are sorted by (left, right), which the kernel's search needs", order == sorted(order))
    try:
        unpack_kern(data[:-2])
        claim("a truncated table is refused", False)
    except ValueError:
        claim("a truncated table is refused", True)
    claim("every kerned character is also in the subset",
          set(codepoints(KERN_RANGES)) <= set(codepoints(RANGES)))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        selftest()
    elif a[:1] == ["build"]:
        build(a)
    elif a[:1] == ["check"]:
        check(a)
    else:
        print(__doc__)

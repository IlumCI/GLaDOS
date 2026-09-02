"""Build a tiny WAD, and read one back.

The kernel's WAD parser needs something to parse before there is a real IWAD
on the machine, and it needs the malformed cases far more than it needs the
good one -- a parser that only ever sees valid input is a parser whose error
paths have never run. In a kernel with no unwinder those paths are the
difference between "that file is truncated" and a halted machine.

    python tools/mkwad.py out/test.wad              # a small valid IWAD
    python tools/mkwad.py out/test.wad --verify     # write it, read it back
    python tools/mkwad.py out/bad.wad --break trunc-dir
    python tools/mkwad.py --list                    # what can be broken

This is the same bargain `tokenizer.py --verify` makes: the reader here is
deliberately not the writer, so a shared misunderstanding of the format shows
up as a disagreement rather than as two halves of one bug agreeing.
"""

import argparse
import struct
import sys
from pathlib import Path

HEADER = 12
DIR_ENTRY = 16

# What a lump is called and what is in it. PLAYPAL is the one lump every IWAD
# has and the first thing a renderer asks for, so the fake has a real one: 256
# RGB triples, a legible ramp rather than noise.
def playpal():
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


LUMPS = [
    ("PLAYPAL", playpal()),
    ("E1M1", b""),                       # a map marker is empty by design
    ("THINGS", b"\x00" * 10),
    ("LINEDEFS", b"\x00" * 14),
    ("SIDEDEFS", b"\x00" * 30),
    ("VERTEXES", struct.pack("<hhhh", 0, 0, 64, 64)),
    ("SECTORS", b"\x00" * 26),
    ("COLORMAP", bytes(range(256)) * 2),
    # Two with the same name, to exercise the rule that a lookup searches from
    # the end and the later one wins.
    ("DUPE", b"first"),
    ("DUPE", b"second"),
]

BREAKS = {
    "none": "a valid WAD",
    "magic": "first four bytes are XWAD",
    "short": "truncated to eight bytes, shorter than a header",
    "trunc-dir": "directory offset points past the end",
    "huge-count": "header claims 4 billion lumps",
    "lump-past-end": "one lump's offset+size runs past the file",
    "negative": "a lump size is negative",
}


def build(kind=b"IWAD", how="none"):
    if how == "short":
        return b"IWAD\x00\x00\x00\x00"

    body = bytearray()
    entries = []
    for name, data in LUMPS:
        entries.append((HEADER + len(body), len(data), name))
        body += data

    dir_at = HEADER + len(body)
    out = bytearray()
    magic = b"XWAD" if how == "magic" else kind
    count = len(LUMPS)
    if how == "huge-count":
        count = 0xFFFF_FFFF
    at = dir_at
    if how == "trunc-dir":
        at = dir_at + 0x1000

    out += magic
    out += struct.pack("<II", count, at)
    out += body
    for i, (pos, size, name) in enumerate(entries):
        if how == "lump-past-end" and name == "SECTORS":
            size = 1 << 20
        if how == "negative" and name == "SECTORS":
            size = -4
        out += struct.pack("<ii", pos, size)
        out += name.encode("ascii")[:8].ljust(8, b"\x00")
    return bytes(out)


def read_back(data):
    """A reader that is deliberately not the writer above."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", help="where to write it")
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

    if args.verify:
        try:
            kind, lumps = read_back(data)
        except ValueError as e:
            if args.how == "none":
                print(f"FAIL: a WAD we wrote does not read back: {e}")
                return 1
            print(f"  reads back as broken, as intended: {e}")
            return 0
        if args.how != "none":
            print(f"FAIL: {args.how} was supposed to be rejected and parsed fine")
            return 1
        print(f"  {kind}, {len(lumps)} lumps")
        for name, d in lumps:
            print(f"    {name:8} {len(d):>6} B")
        # The override rule, checked here so the kernel's answer has something
        # to be right or wrong against.
        last = [d for n, d in lumps if n == "DUPE"][-1]
        assert last == b"second", "last DUPE should win"
        print("  last duplicate wins: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

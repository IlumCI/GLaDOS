#!/usr/bin/env python3
"""Read an Intel wireless firmware container, and write one.

**The second reader.** `src/dev/iwx/fw.rs` is the kernel's, and two
implementations of one format do not stay agreeing unless something makes them --
which is the argument `tools/v4.py` makes for the checkpoint format and
`tokenizer.py --verify` makes for the tokeniser. This is the same bargain for the
firmware container: written from `iwl-fw-file.h` rather than transcribed from the
Rust, so the two can disagree and be caught.

A container is 88 bytes of header then a stream of type-length-value records,
each padded to a multiple of four. The header's leading word is zero, which is how
a TLV image is told from the v1 format it replaced -- no valid combination of
major/minor/API/serial is zero.

**It walks and never seeks, and asserts it lands on the last byte.** A firmware
image carries no table of contents, so a reader that disagreed with the writer
about one length would find perfectly valid records of the wrong thing after it --
and a misparsed section is not a section that fails to load, it is the right
number of bytes written to the wrong address in a radio's memory.

    python tools/iwxfw.py --selftest
    python tools/iwxfw.py /lib/firmware/iwlwifi-QuZ-a0-hr-b0-77.ucode
    python tools/iwxfw.py --emit out/fixture.ucode
"""
import argparse
import struct
import sys

MAGIC = 0x0A4C5749
HEADER = 88
HUMAN = 64

CPU1_CPU2_SEPARATOR = 0xFFFFCCCC
PAGING_SEPARATOR = 0xAAAABBBB

# Only the types this kernel acts on are named. Everything else is reported by
# number, because "a record we ignored" and "a record nobody has heard of" are
# the same thing to a loader and naming them all would be a table to keep in step.
NAMED = {
    19: "SEC_RT",
    20: "SEC_INIT",
    27: "NUM_OF_CPU",
    32: "PAGING",
    36: "FW_VERSION",
    52: "IML",
}
LOADABLE = {19, 20}

# Types from 0x1000005 up are IWL_UCODE_TLV_DEBUG_BASE and its successors, the
# debug-region descriptors. A real image is mostly these -- the AX201's own
# firmware carries 98 of them out of 181 records -- and a loader ignores every
# one, so they are reported by number rather than named.


class Bad(Exception):
    pass


def parse(b: bytes) -> dict:
    """Walk the container. Every refusal names what it found."""
    if len(b) < HEADER:
        raise Bad(f"{len(b)} bytes is shorter than the {HEADER}-byte header")
    zero, magic = struct.unpack_from("<II", b, 0)
    if zero != 0:
        raise Bad(f"leading word {zero:#010x} is not zero, so this is a v1 image")
    if magic != MAGIC:
        raise Bad(f"magic {magic:#010x} is not {MAGIC:#010x}")
    human = b[8:8 + HUMAN].split(b"\0", 1)[0].decode("utf-8", "replace")
    ver, build = struct.unpack_from("<II", b, 8 + HUMAN)

    recs, sections, cpus = [], [], None
    at, n = HEADER, 0
    while at < len(b):
        left = len(b) - at
        if left < 8:
            raise Bad(f"{left} byte(s) left over, too few for a record")
        t, ln = struct.unpack_from("<II", b, at)
        body = at + 8
        if ln > len(b) - body:
            raise Bad(f"record {n} declares {ln} bytes with {len(b) - body} left")
        recs.append((t, ln))
        if t in LOADABLE:
            if ln < 4:
                raise Bad(f"record {n} is a section of {ln} bytes, too short for a destination")
            off = struct.unpack_from("<I", b, body)[0]
            sections.append({"type": t, "offset": off, "at": body + 4, "len": ln - 4})
        elif t == 27 and ln >= 4:
            cpus = struct.unpack_from("<I", b, body)[0]
        # Padding is part of the record: rounding up is how the next one is
        # found, and advancing by `ln` alone lands one to three bytes early.
        step = 8 + ((ln + 3) & ~3)
        if step > left:
            raise Bad(f"record {n} with padding needs {step} bytes, {left} left")
        at += step
        n += 1
    if at != len(b):
        raise Bad(f"{len(b) - at} byte(s) left over")
    if not sections:
        raise Bad("parsed whole, and declares nothing to load")
    return {
        "human": human, "ver": ver, "build": build,
        "records": recs, "sections": sections, "cpus": cpus,
    }


def build(human: str, ver: int, build_no: int, recs) -> bytes:
    out = bytearray()
    out += struct.pack("<II", 0, MAGIC)
    h = human.encode()[:HUMAN]
    out += h + b"\0" * (HUMAN - len(h))
    out += struct.pack("<IIQ", ver, build_no, 0)
    assert len(out) == HEADER, len(out)
    for t, body in recs:
        out += struct.pack("<II", t, len(body)) + body
        while len(out) % 4:
            out += b"\0"
    return bytes(out)


def section(offset: int, data: bytes) -> bytes:
    return struct.pack("<I", offset) + data


def fixture() -> bytes:
    """A container shaped like a real one: two CPUs, a separator, an ignored record."""
    return build(
        "77.1a2b3c4d.0 QuZ-a0-hr-b0-77", 0x01000000, 4242,
        [
            (27, struct.pack("<I", 2)),
            (19, section(0x00800000, b"\xAA" * 16)),
            (19, section(CPU1_CPU2_SEPARATOR, b"")),
            (19, section(0x00400000, b"\xBB" * 12)),
            (61, b"phy integration"),
        ],
    )


def report(b: bytes, path: str) -> int:
    try:
        img = parse(b)
    except Bad as e:
        print(f"REFUSED {path}: {e}")
        return 1
    print(f"{path}: {len(b):,} bytes")
    print(f"  version   {img['human']!r}")
    print(f"  ver/build {img['ver']:#010x} / {img['build']}")
    print(f"  cpus      {img['cpus'] if img['cpus'] is not None else 'not stated'}")
    print(f"  records   {len(img['records'])}")
    load = [s for s in img["sections"]
            if s["offset"] not in (CPU1_CPU2_SEPARATOR, PAGING_SEPARATOR)]
    print(f"  sections  {len(img['sections'])} ({len(load)} loadable, "
          f"{sum(s['len'] for s in load):,} bytes)")
    for s in img["sections"]:
        if s["offset"] == CPU1_CPU2_SEPARATOR:
            print("    -- cpu1/cpu2 separator --")
        elif s["offset"] == PAGING_SEPARATOR:
            print("    -- paging separator --")
        else:
            print(f"    {NAMED.get(s['type'], s['type'])} -> {s['offset']:#010x}  "
                  f"{s['len']:,} bytes at {s['at']}")
    seen = {}
    for t, ln in img["records"]:
        seen.setdefault(t, [0, 0])
        seen[t][0] += 1
        seen[t][1] += ln
    print("  every record type, including the ignored ones:")
    for t in sorted(seen):
        c, tot = seen[t]
        print(f"    {NAMED.get(t, f'type {t}'):14} x{c:<3} {tot:,} bytes")
    return 0


def selftest() -> int:
    ok = [0, 0]

    def claim(good, what):
        ok[0] += 1
        if good:
            print(f"ok    {what}")
        else:
            ok[1] += 1
            print(f"FAIL  {what}")

    img = fixture()
    # The round trip, which is the only reason a writer is here.
    p = parse(img)
    claim(p["human"] == "77.1a2b3c4d.0 QuZ-a0-hr-b0-77", "the version string round-trips")
    claim(p["build"] == 4242, "and the build number")
    claim(p["cpus"] == 2, "the cpu count comes from its own record")
    claim(len(p["records"]) == 5, "every record is seen, including the ignored one")
    claim(any(t == 61 for t, _ in p["records"]), "an unnamed type is carried, not dropped")

    load = [s for s in p["sections"] if s["offset"] not in
            (CPU1_CPU2_SEPARATOR, PAGING_SEPARATOR)]
    claim(len(p["sections"]) == 3, "three section records")
    claim(len(load) == 2, "of which two are loadable")
    claim(sum(s["len"] for s in load) == 28, "and the loadable bytes are the bodies alone")
    claim(img[load[0]["at"]] == 0xAA, "a section's bytes are where it says they are")

    # Every refusal, because the walk being exact is the whole design.
    def refuses(b, what):
        try:
            parse(b)
            claim(False, what)
        except Bad:
            claim(True, what)

    refuses(b"\0" * 8, "a short buffer is refused")
    refuses(b"\x01" + img[1:], "a v1 image is refused by its leading word")
    refuses(img[:4] + b"\0\0\0\0" + img[8:], "a wrong magic is refused")
    over = bytearray(img)
    struct.pack_into("<I", over, HEADER + 4, len(img) * 2)
    refuses(bytes(over), "a record longer than the file is refused")
    huge = bytearray(img)
    struct.pack_into("<I", huge, HEADER + 4, 0xFFFFFFFF)
    refuses(bytes(huge), "a length of 0xFFFFFFFF is refused rather than wrapping")
    refuses(img + b"\0\0\0\0", "a trailing fragment is refused")
    refuses(build("x", 1, 1, [(19, b"\0\0")]),
            "a section shorter than its destination is refused")
    refuses(build("x", 1, 1, [(61, b"only a capability")]),
            "a container with nothing to load is refused")

    # Padding is part of the record, which is the arithmetic most easily got
    # wrong: a nine-byte body is followed by three bytes that belong to it.
    odd = build("x", 1, 1, [(19, section(0x1000, b"\xCD" * 9)),
                            (19, section(0x2000, b"\xEF" * 4))])
    q = parse(odd)
    claim(len(q["sections"]) == 2, "an odd-length record is padded and still parses")
    claim(q["sections"][1]["offset"] == 0x2000,
          "and the record after it is found at the right place")
    claim(len(odd) % 4 == 0, "the whole container is a multiple of four")

    print(f"\n{ok[0]} claim(s), {ok[1]} failed")
    return 1 if ok[1] else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="a .ucode container to read")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit", metavar="OUT", help="write a fixture container")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.emit:
        b = fixture()
        with open(a.emit, "wb") as f:
            f.write(b)
        print(f"wrote {a.emit}, {len(b)} bytes")
        return report(b, a.emit)
    if not a.path:
        ap.print_help()
        return 2
    with open(a.path, "rb") as f:
        return report(f.read(), a.path)


if __name__ == "__main__":
    sys.exit(main())

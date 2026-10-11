#!/usr/bin/env python3
"""Pack CMUdict into GLADOSL1, the pronouncing dictionary `src/voice` reads.

A word's pronunciation is a lookup before it is a rule. English spelling is
too irregular for letter-to-sound rules to be the first resort -- `though`,
`through`, `tough` and `thought` share four letters and no vowel -- so the
voice asks this table first and falls back to rules (`voice::g2p`) only for
a word it does not have.

CMUdict (Carnegie Mellon University, BSD 2-clause, NOTICE.md) has 135,000
words in ARPAbet with lexical stress. Alternative pronunciations, `word(2)`,
are dropped: the first is the common one, and a voice that chose among
them would need a reason this file cannot give it.

Format, little-endian:

    magic   8   "GLADOSL1"
    count   u32
    blob    u32 bytes of entry data that follows the index
    index   u32 * count     offset of each entry within the data
    data    per entry: u8 word length, word bytes (a-z and '),
            u8 phone count, phones as u8 each

Entries are sorted by word bytes so the kernel binary-searches the index.
A phone byte is `base | stress << 6`, base an index into PHONES and stress
0/1/2 for vowels, 3 for a consonant (which has none).

    python3 tools/lexicon.py build cmudict.dict assets/voice/lexicon.bin
    python3 tools/lexicon.py look assets/voice/lexicon.bin hello world
    python3 tools/lexicon.py --selftest
"""
import struct
import sys
from pathlib import Path

MAGIC = b"GLADOSL1"
# The 39 ARPAbet phones CMUdict uses, in an order `src/voice/phone.rs` copies.
PHONES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH",
    "T", "TH", "V", "W", "Y", "Z", "ZH",
]
CODE = {p: i for i, p in enumerate(PHONES)}
VOWELS = set(PHONES[:15])


def encode(phones):
    out = []
    for ph in phones:
        base = ph.rstrip("012")
        stress = int(ph[-1]) if ph[-1].isdigit() else 3
        if base not in CODE:
            raise ValueError(f"unknown phone {ph}")
        if (base in VOWELS) != (stress != 3):
            raise ValueError(f"stress on a consonant or none on a vowel: {ph}")
        out.append(CODE[base] | stress << 6)
    return out


def decode(codes):
    out = []
    for c in codes:
        base, stress = PHONES[c & 63], c >> 6
        out.append(base + ("" if stress == 3 else str(stress)))
    return out


def parse(text: str):
    words = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        word, *phones = line.split()
        if "(" in word:
            continue
        if not all(c.isascii() and (c.isalpha() or c == "'") for c in word):
            continue
        w = word.lower()
        if w not in words:
            words[w] = encode(phones)
    return words


def pack(words) -> bytes:
    items = sorted((w.encode(), p) for w, p in words.items())
    data = bytearray()
    index = []
    for w, p in items:
        if len(w) > 255 or len(p) > 255:
            continue
        index.append(len(data))
        data += bytes([len(w)]) + w + bytes([len(p)]) + bytes(p)
    return MAGIC + struct.pack("<II", len(index), len(data)) + struct.pack(f"<{len(index)}I", *index) + bytes(data)


def look(blob: bytes, word: str):
    if blob[:8] != MAGIC:
        raise ValueError("not GLADOSL1")
    n, size = struct.unpack_from("<II", blob, 8)
    idx = struct.unpack_from(f"<{n}I", blob, 16)
    base = 16 + 4 * n
    if base + size != len(blob):
        raise ValueError("size does not match")
    key = word.lower().encode()
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        o = base + idx[mid]
        wl = blob[o]
        w = blob[o + 1:o + 1 + wl]
        if w < key:
            lo = mid + 1
        elif w > key:
            hi = mid
        else:
            pl = blob[o + 1 + wl]
            return decode(blob[o + 2 + wl:o + 2 + wl + pl])
    return None


def selftest() -> bool:
    ok = True

    def claim(what, good):
        nonlocal ok
        print(("ok  " if good else "FAIL"), what)
        ok &= bool(good)

    sample = "hello HH AH0 L OW1\nhello(2) HH EH0 L OW1\nworld W ER1 L D\n'bout B AW1 T\nzebra Z IY1 B R AH0 # a comment\n"
    words = parse(sample)
    claim("alternatives are dropped, the first kept", decode(words["hello"]) == ["HH", "AH0", "L", "OW1"])
    blob = pack(words)
    claim("every word round-trips through the packed table", all(look(blob, w) == decode(p) for w, p in words.items()))
    claim("an apostrophe word is kept and found", look(blob, "'bout") == ["B", "AW1", "T"])
    claim("a word not in the table is None, not a near neighbour", look(blob, "helloo") is None and look(blob, "a") is None)
    claim("lookup ignores case", look(blob, "WORLD") == ["W", "ER1", "L", "D"])
    try:
        encode(["K1"])
        claim("a stressed consonant is refused", False)
    except ValueError:
        claim("a stressed consonant is refused", True)
    return ok


def main() -> int:
    a = sys.argv[1:]
    if a[:1] == ["--selftest"]:
        return 0 if selftest() else 1
    if a[:1] == ["build"] and len(a) == 3:
        words = parse(Path(a[1]).read_text(encoding="latin-1"))
        blob = pack(words)
        Path(a[2]).parent.mkdir(parents=True, exist_ok=True)
        Path(a[2]).write_bytes(blob)
        print(f"  {len(words)} words, {len(blob)} bytes -> {a[2]}")
        return 0
    if a[:1] == ["look"] and len(a) >= 3:
        blob = Path(a[1]).read_bytes()
        for w in a[2:]:
            print(f"  {w}: {' '.join(look(blob, w) or ['(not in the table)'])}")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())

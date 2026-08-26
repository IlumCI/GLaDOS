#!/usr/bin/env python3
"""Read, check and convert GLaDOS adapter files.

`v4.py` is to checkpoints what this is to adapters: the host-side reader that
exists so a kernel-side writer has something independent to be wrong against.
An adapter is float32 all the way down and carries no names or shapes in its
payload, so a length read wrongly produces finite, plausible, wrong weights and
nothing downstream complains. Both readers therefore walk and never seek, and
assert they land on the last byte.

Layout -- integers little-endian u32, floats little-endian f32:

    0    8   magic "GLADOSA1"
    8    4   rank r
    12   4   alpha, as f32 bits
    16   4   n_layers
    20   4   dim            (the classifier site's k_in, 0 if absent)
    24   4   vocab          (the classifier site's out, 0 if absent)

`dim` and `vocab` are there so a file can be identified without walking it, and
are deliberately not what the kernel checks against: every site carries its own
k_in and out, and those are what must match the model exactly.
    28   4   site count
    32       sites

    site:
    0    4   kind   0=q, 1=k, 2=v, 3=classifier
    4    4   layer  (ignored for the classifier)
    8    4   k_in
    12   4   out
    16   4   rows stored
    20       a: r*k_in floats
             then, per stored row: u32 index, r floats of B, one float of m

Rows are sparse because they are sparse in fact. Only rows whose low-rank
factors or magnitude have moved are written -- a row with zero B and a default
magnitude is bit-identical to no adapter -- and the decision layer moves about
a hundred rows out of a vocabulary of fifty thousand.

Relationship to RustLMHub's LoAA
--------------------------------
The two formats make the same promises and are not byte-compatible, and both
halves of that matter. Same: a magic that is refused rather than guessed at,
dimensions in the header checked for exact equality, flat little-endian f32,
and an adapter file that does not contain and does not touch the frozen model.
Different, unavoidably: LoAA is LoRA over gate/up/down, this is DoRA over the
attention path and the classifier, so every site here also carries per-row
magnitudes that LoAA has nowhere to put.

`--export-lora` writes one site as a LoAA-shaped flat A/B pair -- the same
magic, the same header order, the same dense little-endian f32 payload. Two
things it is not, both worth saying plainly rather than discovering:

  * It is *lossy*. The magnitudes are dropped, because LoRA has nowhere to put
    them, so what arrives is the low-rank direction without the per-row
    rescaling DoRA trained alongside it. Reimporting the result does not give
    back the adapter that was exported.
  * It is *one site*. `FfnLora::load` expects gate, up and down in a single
    file and checks its dims against a block shape; a one-site export is not a
    drop-in for it. What this produces is a file that tool's reader can be
    pointed at without a parser being written first, which is the part that
    was actually costing anything.
"""

import argparse
import struct
import sys
from pathlib import Path

MAGIC = b"GLADOSA1"
HEADER = 32
SITE_HEADER = 20
KINDS = {0: "q", 1: "k", 2: "v", 3: "classifier"}


class Site:
    def __init__(self, kind, layer, k_in, out, a, rows):
        self.kind, self.layer = kind, layer
        self.k_in, self.out = k_in, out
        self.a, self.rows = a, rows

    @property
    def name(self):
        if self.kind == 3:
            return "classifier"
        return f"layer {self.layer} {KINDS[self.kind]}"


class Adapter:
    def __init__(self, r, alpha, n_layers, dim, vocab, sites):
        self.r, self.alpha = r, alpha
        self.n_layers, self.dim, self.vocab = n_layers, dim, vocab
        self.sites = sites


def read(blob):
    """Walk an adapter back. Raises on anything that does not add up."""
    if len(blob) < HEADER or blob[:8] != MAGIC:
        raise ValueError("not a GLADOSA1 adapter")
    r, alpha_bits, n_layers, dim, vocab, n_sites = struct.unpack_from("<IIIIII", blob, 8)
    alpha = struct.unpack("<f", struct.pack("<I", alpha_bits))[0]

    off, sites = HEADER, []
    for i in range(n_sites):
        if off + SITE_HEADER > len(blob):
            raise ValueError(f"site {i}: truncated header")
        kind, layer, k_in, out, n_rows = struct.unpack_from("<IIIII", blob, off)
        off += SITE_HEADER
        if kind not in KINDS:
            raise ValueError(f"site {i}: unknown kind {kind}")

        need = r * k_in * 4 + n_rows * (4 + r * 4 + 4)
        if off + need > len(blob):
            raise ValueError(f"site {i}: claims {need} bytes, {len(blob) - off} remain")

        a = list(struct.unpack_from(f"<{r * k_in}f", blob, off))
        off += r * k_in * 4
        rows = []
        for _ in range(n_rows):
            (idx,) = struct.unpack_from("<I", blob, off)
            off += 4
            if idx >= out:
                raise ValueError(f"site {i}: row {idx} outside {out}")
            brow = list(struct.unpack_from(f"<{r}f", blob, off))
            off += r * 4
            (m,) = struct.unpack_from("<f", blob, off)
            off += 4
            rows.append((idx, brow, m))
        sites.append(Site(kind, layer, k_in, out, a, rows))

    # Landing anywhere else means a length was written that nothing read.
    if off != len(blob):
        raise ValueError(f"{len(blob) - off} trailing bytes after {n_sites} sites")
    return Adapter(r, alpha, n_layers, dim, vocab, sites)


def write(ad):
    """The writer, so the reader has something to disagree with."""
    out = bytearray(MAGIC)
    out += struct.pack(
        "<IIIIII",
        ad.r,
        struct.unpack("<I", struct.pack("<f", ad.alpha))[0],
        ad.n_layers,
        ad.dim,
        ad.vocab,
        len(ad.sites),
    )
    for s in ad.sites:
        out += struct.pack("<IIIII", s.kind, s.layer, s.k_in, s.out, len(s.rows))
        out += struct.pack(f"<{len(s.a)}f", *s.a)
        for idx, brow, m in s.rows:
            out += struct.pack("<I", idx)
            out += struct.pack(f"<{len(brow)}f", *brow)
            out += struct.pack("<f", m)
    return bytes(out)


def report(ad, size):
    print(f"  GLADOSA1  rank {ad.r}, alpha {ad.alpha:g}, {ad.n_layers} layers")
    print(f"  classifier geometry: dim {ad.dim} -> vocab {ad.vocab}")
    print(f"  {len(ad.sites)} site(s), {size} bytes on disk")
    total_dense = 0
    for s in ad.sites:
        # What this site would cost with every row written out: A, then a
        # full B and a full m.
        dense = len(s.a) * 4 + s.out * (ad.r + 1) * 4
        total_dense += dense
        moved = len(s.rows)
        pct = 100.0 * moved / s.out if s.out else 0.0
        print(f"    {s.name:<16} {s.k_in} -> {s.out}, {moved} row(s) moved ({pct:.2f}%)")
    if total_dense:
        # Said as a ratio because the sparse encoding is the reason an adapter
        # is small enough to snapshot before every change, and a ratio is the
        # only form of that claim which stays true as the vocabulary grows.
        print(f"  dense would be {total_dense} bytes -- {total_dense / max(size, 1):.1f}x this file")


LOAA_MAGIC = 0x4C6F4141


def export_lora(ad, site, path):
    """Write one site's A and B in RustLMHub's flat LoAA shape.

    Lossy, and the loss is the whole DoRA half: `m` is dropped, because LoAA
    is LoRA and has nowhere to record a per-row magnitude. What arrives is the
    low-rank direction alone. B is expanded to dense here -- a LoRA reader
    expects every row present, and the rows this file omits are zero, which is
    exactly what a dense B says about them.
    """
    b = [0.0] * (site.out * ad.r)
    for idx, brow, _m in site.rows:
        b[idx * ad.r:(idx + 1) * ad.r] = brow
    out = bytearray()
    out += struct.pack("<IIII", LOAA_MAGIC, site.k_in, site.out, ad.r)
    out += struct.pack(f"<{len(site.a)}f", *site.a)
    out += struct.pack(f"<{len(b)}f", *b)
    Path(path).write_bytes(bytes(out))
    dropped = sum(1 for _, _, m in site.rows if m != 1.0)
    print(f"  wrote {path}  ({len(out)} bytes, LoAA header + A + dense B)")
    print(f"  DROPPED {dropped} per-row magnitude(s): this is the direction, not the adapter")


def selftest():
    """Round-trip the writer through the reader, then break it on purpose."""
    ok = True

    def claim(what, passed):
        nonlocal ok
        ok = ok and passed
        print(f"  {'ok  ' if passed else 'FAIL'}  {what}")

    r = 3
    cls = Site(3, 0, 8, 40, [0.01 * i - 0.1 for i in range(r * 8)],
               [(2, [0.5, 0.4, 0.3], 1.5), (17, [0.1, 0.2, 0.3], 5.75)])
    q = Site(0, 1, 8, 12, [0.02 * i for i in range(r * 8)], [(5, [1.0, 2.0, 3.0], 2.0)])
    ad = Adapter(r, 6.0, 2, 8, 40, [q, cls])
    blob = write(ad)

    back = read(blob)
    shape = (
        back.r == r and back.alpha == 6.0 and back.n_layers == 2
        and back.dim == 8 and back.vocab == 40 and len(back.sites) == 2
        and all(
            a.kind == b.kind and a.layer == b.layer and a.k_in == b.k_in
            and a.out == b.out and len(a.a) == len(b.a)
            and [i for i, _, _ in a.rows] == [i for i, _, _ in b.rows]
            for a, b in zip(ad.sites, back.sites)
        )
    )
    claim("writer and reader agree on every site, shape and row index", shape)

    # The values are compared by rewriting rather than by ==. The fixture is
    # written in Python floats, which are f64, and the file is f32 -- so a
    # direct comparison would fail on rounding that is not an error, and
    # would go on failing however correct the format was. Byte equality after
    # a second pass is the claim that actually means "nothing was lost".
    claim("a re-written adapter is byte-identical", write(back) == blob)

    for name, mutate in [
        ("one byte short is caught", lambda b: b[:-1]),
        ("a byte past the last site is caught", lambda b: b + b"\x00"),
        ("a wrong magic is refused", lambda b: b"GLADOSC1" + b[8:]),
    ]:
        try:
            read(mutate(blob))
            claim(name, False)
        except ValueError:
            claim(name, True)

    # A site whose declared row count runs past the end must be refused before
    # the floats are read, not after a plausible prefix has been consumed.
    broken = bytearray(blob)
    struct.pack_into("<I", broken, HEADER + 16, 9999)
    try:
        read(bytes(broken))
        claim("an impossible row count is refused", False)
    except ValueError:
        claim("an impossible row count is refused", True)

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--export-lora", type=Path, default=None,
                    help="write one site as a flat LoAA A/B pair (lossy: drops m)")
    ap.add_argument("--site", default="classifier",
                    help="which site to export: 'classifier', or 'q/k/v:<layer>'")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if not args.path:
        ap.error("a path, or --selftest")

    blob = args.path.read_bytes()
    ad = read(blob)
    report(ad, len(blob))

    if args.export_lora:
        if args.site == "classifier":
            picked = next((s for s in ad.sites if s.kind == 3), None)
        else:
            kind_name, _, layer = args.site.partition(":")
            kinds = {v: k for k, v in KINDS.items()}
            picked = next(
                (s for s in ad.sites
                 if s.kind == kinds.get(kind_name) and str(s.layer) == layer),
                None,
            )
        if picked is None:
            raise SystemExit(f"  no such site: {args.site}")
        export_lora(ad, picked, args.export_lora)


if __name__ == "__main__":
    main()

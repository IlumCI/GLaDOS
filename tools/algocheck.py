#!/usr/bin/env python3
"""The oracle for the GPU mining algorithms.

Every algorithm in `cuda/` carries an expected digest, and this is where that
digest comes from. It is computed with `hashlib` -- an implementation nobody
in this repository wrote -- because an algorithm checked against itself is not
checked at all, and this project has already lost weeks to a transformer that
was fast, well behaved, and wrong.

    algocheck.py                 print the expected digest for every algorithm
    algocheck.py --patch         write them into the .cuh files
    algocheck.py --selftest      check this file against published vectors

The same bargain `tokenizer.py --verify` makes: the reader is deliberately not
the writer.
"""

import argparse
import hashlib
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CUDA = os.path.join(os.path.dirname(HERE), "cuda")

# Block 125552. The nonce is already in the header, so this is the digest the
# device must produce for the nonce word it is handed.
BTC_HEADER = bytes([
    0x01,0x00,0x00,0x00,
    0x81,0xcd,0x02,0xab,0x7e,0x56,0x9e,0x8b,0xcd,0x93,0x17,0xe2,
    0xfe,0x99,0xf2,0xde,0x44,0xd4,0x9a,0xb2,0xb8,0x85,0x1b,0xa4,
    0xa3,0x08,0x00,0x00,0x00,0x00,0x00,0x00,
    0xe3,0x20,0xb6,0xc2,0xff,0xfc,0x8d,0x75,0x04,0x23,0xdb,0x8b,
    0x1e,0xb9,0x42,0xae,0x71,0x0e,0x95,0x1e,0xd7,0x97,0xf7,0xaf,
    0xfc,0x88,0x92,0xb0,0xf1,0xfc,0x12,0x2b,
    0xc7,0xf5,0xd7,0x4d, 0xf2,0xb9,0x44,0x1a, 0x42,0xa1,0x46,0x95])

# The BLAKE2s block, mirroring cuda/blake2s.cuh exactly. The device replaces
# message word 15, which is bytes 60..64 read little-endian.
B2S_SEED = (b"GLaDOS blake2s vector "
            + b"0" * 60)[:64]
B2S_VERIFY_NONCE = 0x12345678


def sha256d_expect():
    return hashlib.sha256(hashlib.sha256(BTC_HEADER).digest()).digest().hex()


def blake2s_expect():
    blk = bytearray(B2S_SEED)
    blk[60:64] = B2S_VERIFY_NONCE.to_bytes(4, "little")
    return hashlib.blake2s(bytes(blk), digest_size=32).digest().hex()


ALGOS = [
    ("sha256d", "sha256d.cuh", sha256d_expect),
    ("blake2s", "blake2s.cuh", blake2s_expect),
]


def selftest():
    """This file, against vectors it did not choose.

    A generator of expected values is exactly the thing that must not be
    self-consistent-and-wrong, so it is checked against two published digests
    before it is trusted to produce any others.
    """
    ok = True

    def claim(what, good):
        nonlocal ok
        if not good:
            ok = False
            print("  FAIL %s" % what)

    # RFC 7693's own BLAKE2s-256 vector for "abc".
    claim("blake2s('abc') matches RFC 7693",
          hashlib.blake2s(b"abc", digest_size=32).hexdigest() ==
          "508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982")

    # Block 125552, whose hash is public and checkable against any explorer.
    # Bitcoin displays it reversed, which is the trap this claim pins down.
    d = hashlib.sha256(hashlib.sha256(BTC_HEADER).digest()).digest()
    claim("block 125552 hashes to its published id",
          d[::-1].hex() ==
          "00000000000000001e8d6829a8a21adc5d38d0a473b144b6765798e61f98bd1d")
    claim("the raw digest is the reverse of the displayed id",
          d.hex() == "1dbd981fe6985776b644b173a4d0385ddc1aa2a829688d1e"
                     "0000000000000000")
    claim("the blake2s block is 64 bytes", len(B2S_SEED) == 64)

    print("algocheck selftest: %s" % ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch", action="store_true",
                    help="write the expected digests into the .cuh files")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    for name, fname, fn in ALGOS:
        want = fn()
        print("%-10s %s" % (name, want))
        if not a.patch:
            continue
        path = os.path.join(CUDA, fname)
        if not os.path.exists(path):
            print("  no %s" % path)
            continue
        text = io.open(path, encoding="utf-8").read()
        new = re.sub(r'return "[@0-9a-fA-F][^"]*";',
                     'return "%s";' % want, text, count=1)
        if new != text:
            io.open(path, "w", encoding="utf-8", newline="\n").write(new)
            print("  patched %s" % fname)
        else:
            print("  %s already current" % fname)
    return 0


if __name__ == "__main__":
    sys.exit(main())

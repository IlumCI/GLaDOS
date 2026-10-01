#!/usr/bin/env python3
"""HeavyHash, as Optical Bitcoin (OBTC) defines it -- the oracle.

    python3 tools/heavyhash.py --selftest
    python3 tools/heavyhash.py HEADER_HEX          # 80 bytes, as serialised

Transcribed from PoWx-Org/obtc-core: `CBlockHeader::GetPoWHash`
(src/primitives/block.cpp), `GenerateHeavyHashMatrix` (src/hash.cpp),
`CHeavyHash` and `MultiplyUsing4bitPrecision` (src/crypto/heavyhash.cpp),
`XoShiRo256PlusPlus` (src/crypto/xoshiro256pp.h). zpool's `heavyhash` port
mines this coin, which is why it is here.

    seed    = SHA3-256(hashPrevBlock)                  32 bytes, as stored
    matrix  = 64x64 nibbles from xoshiro256++(seed), 16 a word, low bits
              first, regenerated until full rank
    h1      = SHA3-256(80-byte header)
    v       = the 64 nibbles of h1, high nibble of each byte first
    p[i]    = (sum_j M[i][j] * v[j]) >> 10             always < 16
    h1'     = h1 XOR (p[0]<<4|p[1], p[2]<<4|p[3], ...)
    pow     = SHA3-256(h1')                            compared little-endian

**SHA3-256 and not cSHAKE256**, which is the whole difference from Kaspa's
kHeavyHash and the reason this has an oracle at all: `hashlib.sha3_256` is
CPython's own implementation and is not a second copy of anything written
here. `cuda/kheavy.cu`'s matrix step is the same arithmetic in the same nibble
order, and was blocked only on the outer hash.

**The rank test is exact here and floating-point upstream.** obtc-core runs an
SVD in `double` and asks whether 64 singular values clear a tolerance. This
eliminates over the rationals instead. The two can only disagree on a matrix
that is singular by less than the tolerance -- for 4096 random nibbles, a
question with no practical cases, and one the selftest cannot exercise.

The selftest is OBTC's mainnet genesis block, whose every header field is in
`chainparams.cpp` and whose hash the node asserts at startup:
0000000000115c7a7e3ff65d77ee96de527953ca6e43e77246929741408f95c0.
"""
import hashlib
import struct
import sys
from fractions import Fraction

MASK = (1 << 64) - 1


def sha3(b: bytes) -> bytes:
    return hashlib.sha3_256(b).digest()


def xoshiro(seed: bytes):
    # uint256::GetUint64(i) reads eight bytes little-endian from offset 8*i.
    s = list(struct.unpack("<4Q", seed))
    rotl = lambda x, k: ((x << k) | (x >> (64 - k))) & MASK
    while True:
        result = (rotl((s[0] + s[3]) & MASK, 23) + s[0]) & MASK
        t = (s[1] << 17) & MASK
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl(s[3], 45)
        yield result


def full_rank(m) -> bool:
    a = [[Fraction(x) for x in row] for row in m]
    n = len(a)
    for c in range(n):
        p = next((r for r in range(c, n) if a[r][c] != 0), None)
        if p is None:
            return False
        a[c], a[p] = a[p], a[c]
        for r in range(c + 1, n):
            if a[r][c]:
                f = a[r][c] / a[c][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return True


def matrix(prev_hash: bytes):
    g = xoshiro(sha3(prev_hash))
    while True:
        m = []
        for _ in range(64):
            row = []
            for _ in range(4):
                v = next(g)
                row += [(v >> (4 * k)) & 0xF for k in range(16)]
            m.append(row)
        if full_rank(m):
            return m


def heavy_step(m, h: bytes) -> bytes:
    v = []
    for b in h:
        v += [b >> 4, b & 0xF]
    p = [sum(m[i][j] * v[j] for j in range(64)) >> 10 for i in range(64)]
    assert all(x < 16 for x in p), "a product escaped four bits"
    return bytes(h[k] ^ ((p[2 * k] << 4) | p[2 * k + 1]) for k in range(32))


def heavyhash(header: bytes, m=None) -> bytes:
    """The PoW digest, in memory order. Reverse it to read it as a hash."""
    assert len(header) == 80
    if m is None:
        m = matrix(header[4:36])
    return sha3(heavy_step(m, sha3(header)))


def header(version, prev_hex, merkle_hex, time, bits, nonce) -> bytes:
    """An 80-byte header from display-order hashes, as Bitcoin serialises it."""
    return (struct.pack("<i", version) + bytes.fromhex(prev_hex)[::-1]
            + bytes.fromhex(merkle_hex)[::-1] + struct.pack("<III", time, bits, nonce))


GENESIS = dict(version=1, prev_hex="00" * 32,
               merkle_hex="c4a47847658174dff39f23e69c2246e7e611752884ceb600694a8619adbbfef5",
               time=1616765395, bits=0x1c00ffff, nonce=1120945927)
GENESIS_HASH = "0000000000115c7a7e3ff65d77ee96de527953ca6e43e77246929741408f95c0"


def selftest() -> bool:
    ok = True

    def claim(c, what):
        nonlocal ok
        ok &= bool(c)
        print(("ok    " if c else "FAIL  ") + what)

    h = header(**GENESIS)
    got = heavyhash(h)[::-1].hex()
    claim(got == GENESIS_HASH, f"OBTC mainnet genesis hashes to its asserted hash ({got[:16]}..)")
    # Negative controls: each one breaks exactly one thing the positive needs,
    # so a pass above cannot be a coincidence of the check.
    claim(heavyhash(header(**{**GENESIS, "nonce": GENESIS["nonce"] + 1}))[::-1].hex() != GENESIS_HASH,
          "one nonce away is a different hash")
    plain = sha3(sha3(h))[::-1].hex()
    claim(plain != GENESIS_HASH, "and the heavy step is not a no-op: SHA3 twice alone does not match")
    m = matrix(h[4:36])
    claim(all(0 <= x < 16 for row in m for x in row), "the matrix is nibbles")
    claim(heavyhash(h, m) == heavyhash(h), "a supplied matrix is the generated one")
    return ok


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(0 if selftest() else 1)
    if len(sys.argv) == 2:
        print(heavyhash(bytes.fromhex(sys.argv[1]))[::-1].hex())
        sys.exit(0)
    print(__doc__)
    sys.exit(2)

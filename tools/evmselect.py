#!/usr/bin/env python3
"""Keccak-256, so a function selector is derived rather than trusted.

Written out because the numbers this produces are going onto a public page, and
this project's rule is that a figure nobody checked is a figure that becomes a
decision later. A selector taken from a 4-byte directory is somebody else's
table; this is the hash.

Checked against the empty-string vector before it is used for anything.
"""
import sys

RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
    0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]
ROT = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]
M = (1 << 64) - 1


def rol(x, n):
    n %= 64
    return ((x << n) | (x >> (64 - n))) & M


def keccak_f(a):
    for rnd in range(24):
        c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ rol(c[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                a[x][y] ^= d[x]
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = rol(a[x][y], ROT[x][y])
        for x in range(5):
            for y in range(5):
                a[x][y] = b[x][y] ^ ((~b[(x + 1) % 5][y] & M) & b[(x + 2) % 5][y])
        a[0][0] ^= RC[rnd]
    return a


def keccak256(data):
    rate = 136
    pad = bytearray(data)
    pad.append(0x01)                      # keccak padding, not SHA-3's 0x06
    while len(pad) % rate != 0:
        pad.append(0x00)
    pad[-1] |= 0x80
    a = [[0] * 5 for _ in range(5)]
    for off in range(0, len(pad), rate):
        blk = pad[off:off + rate]
        for i in range(rate // 8):
            lane = int.from_bytes(blk[i * 8:(i + 1) * 8], "little")
            a[i % 5][i // 5] ^= lane
        a = keccak_f(a)
    out = bytearray()
    while len(out) < 32:
        for i in range(rate // 8):
            if len(out) >= 32:
                break
            out += a[i % 5][i // 5].to_bytes(8, "little")
    return bytes(out[:32])


def selector(sig):
    return keccak256(sig.encode()).hex()[:8]


if __name__ == "__main__":
    want = "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    got = keccak256(b"").hex()
    if got != want:
        print("FAIL keccak256(\"\") = %s" % got)
        sys.exit(1)
    print("ok    keccak256 matches the empty-string vector")
    for sig in sys.argv[1:]:
        print("0x%s  %s" % (selector(sig), sig))

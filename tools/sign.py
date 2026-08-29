#!/usr/bin/env python3
"""Sign a GLaDOS update image with P-256 ECDSA.

    sign.py --keygen                       print a fresh keypair
    sign.py <image> <out.sig> --key <hex>  sign an image

The kernel verifies with `crypto::p256::verify`, which the boot selftest
already checks against published ECDSA vectors -- so a signature this produces
and that the kernel accepts is a signature the kernel's *validated* verifier
accepted. That is what makes this small implementation trustworthy enough to
be the other half of the test: it is never the thing being trusted, only the
thing being checked.

The signature file is deliberately tiny and fixed-length:

    "GLADOSIG"   8   magic
    u32          4   format version, 1
    u32          4   curve, 0 = P-256
    u8[32]      32   r
    u8[32]      32   s
                --
                80 bytes

The digest signed is SHA-256 over the whole image, so the signature commits to
every byte and a truncated image is a failed verification rather than a
shorter valid one.
"""
import hashlib
import os
import struct
import sys

# NIST P-256.
P = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
A = P - 3
B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
GX = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
GY = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5


def inv(a, m):
    return pow(a, m - 2, m)


def add(p, q):
    if p is None:
        return q
    if q is None:
        return p
    (x1, y1), (x2, y2) = p, q
    if x1 == x2 and (y1 + y2) % P == 0:
        return None
    if p == q:
        lam = (3 * x1 * x1 + A) * inv(2 * y1, P) % P
    else:
        lam = (y2 - y1) * inv(x2 - x1, P) % P
    x3 = (lam * lam - x1 - x2) % P
    return (x3, (lam * (x1 - x3) - y1) % P)


def mul(k, p):
    r = None
    while k:
        if k & 1:
            r = add(r, p)
        p = add(p, p)
        k >>= 1
    return r


def keygen():
    d = int.from_bytes(os.urandom(32), "big") % (N - 1) + 1
    q = mul(d, (GX, GY))
    return d, q


def sign(d, digest):
    z = int.from_bytes(digest, "big")
    while True:
        # A random nonce, not RFC 6979. Reusing one leaks the key, so it comes
        # from the OS generator and nowhere else.
        k = int.from_bytes(os.urandom(32), "big") % (N - 1) + 1
        pt = mul(k, (GX, GY))
        r = pt[0] % N
        if r == 0:
            continue
        s = inv(k, N) * (z + r * d) % N
        if s == 0:
            continue
        return r, s


def pack(r, s):
    return (
        b"GLADOSIG"
        + struct.pack("<II", 1, 0)
        + r.to_bytes(32, "big")
        + s.to_bytes(32, "big")
    )


def main():
    if "--keygen" in sys.argv:
        d, q = keygen()
        pub = b"\x04" + q[0].to_bytes(32, "big") + q[1].to_bytes(32, "big")
        print("private (keep this off the machine being updated):")
        print("  " + format(d, "064x"))
        print("public (paste into UPDATE_KEY in src/update.rs):")
        rows = [pub[i:i + 8] for i in range(0, len(pub), 8)]
        for row in rows:
            print("    " + " ".join("0x%02x," % b for b in row))
        return

    if len(sys.argv) < 3 or "--key" not in sys.argv:
        raise SystemExit(__doc__)
    image = sys.argv[1]
    out = sys.argv[2]
    d = int(sys.argv[sys.argv.index("--key") + 1], 16)

    data = open(image, "rb").read()
    digest = hashlib.sha256(data).digest()
    r, s = sign(d, digest)
    open(out, "wb").write(pack(r, s))
    print(f"  signed {len(data)} B of {image}")
    print(f"  sha256 {digest.hex()}")
    print(f"  wrote  {out} ({80} B)")


if __name__ == "__main__":
    main()

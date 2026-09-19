#!/usr/bin/env python3
"""Sign a GLaDOS update image with P-256 ECDSA.

    sign.py --keygen [--out FILE]          make a keypair
    sign.py <image> <out.sig> --key-file F sign an image

Prefer `--keygen --out FILE`. Without it the private half goes to stdout,
and a private half that has been on a terminal is not a private half. That
is how one of them died, and the zeroing that followed is why this line used
to end "and UPDATE_KEY has been zeroed ever since" -- true when written and
long out of date, which is the same drift the key's own doc comment carried.

**There are two anchors now and this signs for either.** `UPDATE_KEY` for a
kernel image, `VERDICT_KEY` for a verdict coming home from `propose.yml`, and
the point of the second is that the workflow any allowlisted device can start
does not hold the first. Which key a signature answers to is decided entirely
by which private half `--key-file` is given, so keep them in separate files
and separate secrets.

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
import re
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



# --- does this private half match the point the kernel pins? --------------

#: Where the anchors are declared. One file, and the check reads it rather than
#: carrying a copy, for the reason `forest_retrieve.py` reads `lex.rs`: a second
#: copy agrees on the day it is written and then drifts.
ANCHORS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "src", "update", "mod.rs")


def anchor(symbol, path=None):
    """The 65-byte uncompressed point named `symbol`, out of the Rust.

    The bytes are found by searching rather than by splitting on delimiters,
    and that is the detail `manifest.py` paid for before this existed: one of
    the two anchors is written on a single line, where its first element reads
    `[0x04` -- which a scan for a `0x` *prefix* drops. 64 bytes, a key that
    looks unprovisioned, and a release failing its own verify step over a key
    that was perfectly good. A substring search sees both spellings alike.
    """
    src = open(path or ANCHORS, encoding="utf-8").read()
    pat = r"pub const " + re.escape(symbol) + r"\s*:\s*\[u8;\s*65\]\s*=\s*\[(.*?)\];"
    m = re.search(pat, src, re.S)
    if not m:
        raise SystemExit(f"  no `const {symbol}: [u8; 65]` in {path or ANCHORS}")
    vals = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{1,2})", m.group(1))]
    if len(vals) != 65 or vals[0] != 0x04:
        if not vals or all(v == 0 for v in vals):
            raise SystemExit(f"  {symbol} is not provisioned in {path or ANCHORS}")
        raise SystemExit(
            f"  {symbol} parsed as {len(vals)} bytes starting {vals[0]:#04x}; "
            "it must be 65 starting 0x04"
        )
    return bytes(vals)


def public_of(d):
    """The uncompressed public point for a private scalar."""
    x, y = mul(d, (GX, GY))
    return b"\x04" + x.to_bytes(32, "big") + y.to_bytes(32, "big")


def check_anchor(symbol, d, path=None):
    """Answers 0 when this key is the one that symbol pins.

    **The check that turns a wrong secret from a silent refusal into a loud
    one.** A signature made with the wrong private half is perfectly well
    formed; the only thing that notices is the machine it eventually reaches,
    which answers "not a signature over this image by this key" and files
    nothing. That is a correct refusal about a configuration mistake made days
    earlier and three systems away.

    Public points only ever leave this function, so a mismatch prints both and
    says which is which. The private half is never rendered, never compared as
    text, and never reaches a log.
    """
    want = anchor(symbol, path)
    got = public_of(d)
    if got == want:
        print(f"  ok     this key is the one {symbol} pins")
        print(f"         {got.hex()[:24]}...")
        return 0
    print(f"  FAIL   this key is not the one {symbol} pins")
    print(f"         pinned  {want.hex()}")
    print(f"         this is {got.hex()}")
    print("         a signature from it is well formed and will be refused by")
    print("         every machine, which is why this is checked here instead")
    return 1


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



SIG_LEN = 80


def verify_sig(pub, data, blob):
    """Check an 80-byte GLADOSIG against an uncompressed public key.

    The other half of `sign`, and deliberately the *only* other half: a
    caller that wants to know whether these bytes will be accepted asks the
    arithmetic rather than re-deriving a key and comparing points, because
    a matching key says nothing about whether the signature over it is whole.
    """
    if len(blob) != 80 or blob[:8] != b"GLADOSIG":
        raise ValueError("not a GLADOSIG signature")
    version = int.from_bytes(blob[8:12], "little")
    curve = int.from_bytes(blob[12:16], "little")
    if version != 1 or curve != 0:
        raise ValueError("a signature format this does not implement")
    r = int.from_bytes(blob[16:48], "big")
    s = int.from_bytes(blob[48:80], "big")
    if not (0 < r < N and 0 < s < N):
        raise ValueError("r or s is out of range")

    if len(pub) != 65 or pub[0] != 0x04:
        raise ValueError("the public key is not uncompressed 0x04||X||Y")
    q = (int.from_bytes(pub[1:33], "big"), int.from_bytes(pub[33:65], "big"))

    z = int.from_bytes(hashlib.sha256(data).digest(), "big")
    w = inv(s, N)
    p = add(mul(z * w % N, (GX, GY)), mul(r * w % N, q))
    if p is None:
        raise ValueError("not a signature over these bytes by this key")
    if p[0] % N != r:
        raise ValueError("not a signature over these bytes by this key")


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
        if "--out" in sys.argv:
            path = sys.argv[sys.argv.index("--out") + 1]
            # 0600 at creation rather than a chmod afterwards, so the key
            # is never briefly readable by anyone else.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                print(format(d, "064x"), file=f)
            print(f"private -> {path}, and not to this terminal")
        else:
            print("private (keep this off the machine being updated):")
            print("  " + format(d, "064x"))
            print("  ^ that is in your scrollback now. --out FILE avoids it.")
        print("public (paste into UPDATE_KEY in src/update/mod.rs):")
        rows = [pub[i:i + 8] for i in range(0, len(pub), 8)]
        for row in rows:
            print("    " + " ".join("0x%02x," % b for b in row))
        return

    # `--check FILE SIG --anchor SYMBOL` reads back what was just produced,
    # with the verifier, against the point the kernel pins. That is the
    # sentence `release.yml` already carries about a manifest, and it is a
    # strictly stronger question than `--anchor` alone: a key that matches
    # says nothing about whether the signature over it is whole.
    if "--check" in sys.argv:
        at = sys.argv.index("--check")
        if len(sys.argv) <= at + 2 or "--anchor" not in sys.argv:
            raise SystemExit("  usage: sign.py --check FILE SIG --anchor SYMBOL")
        data = open(sys.argv[at + 1], "rb").read()
        blob = open(sys.argv[at + 2], "rb").read()
        symbol = sys.argv[sys.argv.index("--anchor") + 1]
        try:
            verify_sig(anchor(symbol), data, blob)
        except ValueError as e:
            print(f"  FAIL   {e}")
            print(f"         {symbol} would refuse this on every machine")
            return 1
        print(f"  ok     {len(data)} B verified against {symbol}")
        return 0

    # `--anchor SYMBOL` checks a key against a pinned point and signs nothing.
    # Its own mode rather than a flag on signing, so a caller that wants the
    # check cannot accidentally also produce a signature, and so it can run in
    # CI before the thing it is guarding.
    if "--anchor" in sys.argv:
        at = sys.argv.index("--anchor")
        if len(sys.argv) <= at + 1:
            raise SystemExit("  usage: sign.py --anchor SYMBOL --key-file FILE")
        symbol = sys.argv[at + 1]
        if "--key-file" in sys.argv:
            d = int(open(sys.argv[sys.argv.index("--key-file") + 1]).read().strip(), 16)
        elif "--key" in sys.argv:
            d = int(sys.argv[sys.argv.index("--key") + 1], 16)
        else:
            raise SystemExit("  --anchor needs --key-file FILE")
        return check_anchor(symbol, d)

    if len(sys.argv) < 3 or not any(a in sys.argv for a in ("--key", "--key-file")):
        raise SystemExit(__doc__)
    image = sys.argv[1]
    out = sys.argv[2]
    if "--key-file" in sys.argv:
        path = sys.argv[sys.argv.index("--key-file") + 1]
        d = int(open(path).read().strip(), 16)
    else:
        d = int(sys.argv[sys.argv.index("--key") + 1], 16)

    data = open(image, "rb").read()
    digest = hashlib.sha256(data).digest()
    r, s = sign(d, digest)
    open(out, "wb").write(pack(r, s))
    print(f"  signed {len(data)} B of {image}")
    print(f"  sha256 {digest.hex()}")
    print(f"  wrote  {out} ({80} B)")


if __name__ == "__main__":
    sys.exit(main() or 0)

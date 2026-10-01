#!/usr/bin/env python3
"""Is this payout address real? The host's answer, written to disagree.

`src/mine/addr.rs` refuses a miner image whose worker name is an address with a
failing checksum. This is the same question answered by a second implementation,
for `tokenizer.py --verify`'s reason: the checksums below are the only thing
standing between a mistyped address and a night of hashing for nobody, and one
implementation checking itself establishes nothing.

Keccak-256 is written out here rather than imported. `hashlib` has `sha3_256`,
which is *not* the same function -- it differs in one padding byte -- and
`eth_utils` is not installed on this host and would be a third party in the one
place this project does not take one. Ninety lines, against published vectors.

    python3 tools/addrcheck.py --selftest
    python3 tools/addrcheck.py 0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed
    python3 tools/addrcheck.py --miner esp/GLADOS/MINER.TXT

The last form reads a real config and answers about the `worker` line in it,
which is the check to run before cutting an image rather than after booting one.
"""

import argparse
import sys

# --- Keccak-256 ------------------------------------------------------------

_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]
_MASK = (1 << 64) - 1


def _rotl(x, n):
    return ((x << n) | (x >> (64 - n))) & _MASK


def _f1600(a):
    """Keccak-f[1600], written by lane coordinate rather than by flat index.

    Deliberately a different arrangement from the Rust, which carries the
    displaced lane through a single cycle. Two spellings of one permutation is
    the point: a shared off-by-one in the rotation table would survive being
    transcribed and does not survive being written from the coordinates.
    """
    for rnd in range(24):
        c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ _rotl(c[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                a[x][y] ^= d[x]

        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                # Rho and Pi in one step, with the offset from `_RHO` -- derived
                # by walking the lane sequence rather than tabulated, so there is
                # no table here to mistype.
                b[y][(2 * x + 3 * y) % 5] = _rotl(a[x][y], _RHO[x][y])
        for x in range(5):
            for y in range(5):
                a[x][y] = b[x][y] ^ ((~b[(x + 1) % 5][y] & _MASK) & b[(x + 2) % 5][y])

        a[0][0] ^= _RC[rnd]
    return a


def _build_rho():
    """Rho's offsets from the walk that defines them, not from a table."""
    r = [[0] * 5 for _ in range(5)]
    x, y = 1, 0
    for t in range(24):
        r[x][y] = ((t + 1) * (t + 2) // 2) % 64
        x, y = y, (2 * x + 3 * y) % 5
    return r


_RHO = _build_rho()


def keccak256(msg: bytes) -> bytes:
    rate = 136
    a = [[0] * 5 for _ in range(5)]
    pad = bytearray(msg)
    pad.append(0x01)
    while len(pad) % rate != 0:
        pad.append(0x00)
    pad[-1] ^= 0x80
    for off in range(0, len(pad), rate):
        block = pad[off:off + rate]
        for i in range(rate // 8):
            lane = int.from_bytes(block[i * 8:(i + 1) * 8], "little")
            a[i % 5][i // 5] ^= lane
        _f1600(a)
    out = bytearray()
    i = 0
    while len(out) < 32:
        out += a[i % 5][i // 5].to_bytes(8, "little")
        i += 1
    return bytes(out[:32])


# --- the three forms -------------------------------------------------------

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
HRPS = ("bc", "tb", "bcrt", "ltc", "tltc", "rltc")


def _sha256d(b):
    import hashlib
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def evm(s):
    if not (s.startswith("0x") or s.startswith("0X")):
        return None
    body = s[2:]
    if len(body) != 40 or any(c not in "0123456789abcdefABCDEF" for c in body):
        return None
    letters = [c for c in body if c.isalpha()]
    if not letters or all(c.islower() for c in letters) or all(c.isupper() for c in letters):
        return ("unchecked", "EVM hex")
    h = keccak256(body.lower().encode())
    for i, c in enumerate(body):
        if not c.isalpha():
            continue
        nib = (h[i // 2] >> 4) if i % 2 == 0 else (h[i // 2] & 0x0F)
        if (nib >= 8) != c.isupper():
            return ("broken", "EVM hex")
    return ("checked", "EVM hex")


def _polymod(values):
    gen = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for v in values:
        top = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ v
        for i in range(5):
            if (top >> i) & 1:
                chk ^= gen[i]
    return chk


def bech32(s):
    if s != s.lower() and s != s.upper():
        return None
    low = s.lower()
    if not 8 <= len(low) <= 90 or "1" not in low:
        return None
    sep = low.rfind("1")
    hrp, data = low[:sep], low[sep + 1:]
    if hrp not in HRPS or len(data) < 7:
        return None
    vals = [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]
    payload = []
    for c in data:
        if c not in CHARSET:
            return None
        payload.append(CHARSET.index(c))
        vals.append(CHARSET.index(c))
    want = 1 if payload[0] == 0 else 0x2BC830A3
    return ("checked" if _polymod(vals) == want else "broken", "bech32")


def base58(s):
    if not 26 <= len(s) <= 35:
        return None
    n = 0
    for c in s:
        if c not in B58:
            return None
        n = n * 58 + B58.index(c)
    body = n.to_bytes((n.bit_length() + 7) // 8 or 1, "big")
    lead = 0
    for c in s:
        if c == "1":
            lead += 1
        else:
            break
    body = b"\x00" * lead + body
    if len(body) != 25:
        return None
    return ("checked" if _sha256d(body[:21])[:4] == body[21:] else "broken", "base58check")


def judge(worker: str):
    head = worker.split(".", 1)[0]
    for f in (evm, bech32, base58):
        v = f(head)
        if v is not None:
            return v
    return ("name", "-")


# --- the two front ends ----------------------------------------------------

SAY = {
    "checked": "checks out",
    "unchecked": "well formed, and carries no checksum to read",
    "broken": "CHECKSUM FAILS -- this pays nobody",
    "name": "a worker name, so the pool's roster decides where this pays",
}


def report(worker: str) -> int:
    verdict, kind = judge(worker)
    tag = kind if kind != "-" else ""
    print(f"{worker}\n  {verdict:<9} {tag:<12} {SAY[verdict]}")
    return 1 if verdict == "broken" else 0


def selftest() -> int:
    ok = True

    def claim(good, what):
        nonlocal ok
        print(f"  {'ok  ' if good else 'FAIL'}  {what}")
        ok &= bool(good)

    claim(keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
          "keccak256 of the empty string")
    claim(keccak256(b"abc").hex() == "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45",
          "keccak256 of 'abc'")
    import hashlib
    claim(keccak256(b"") != hashlib.sha3_256(b"").digest(),
          "and it is not hashlib's sha3_256, which is the whole reason it is here")
    claim(keccak256(b"a" * 136) != keccak256(b"a" * 137), "a message filling the rate pads into its own block")
    # Rho's offsets, derived here and tabulated in the Rust. The published
    # sequence is the one thing both spellings have to agree on.
    claim([_RHO[1][0], _RHO[0][2], _RHO[2][1]] == [1, 3, 6], "rho's first offsets match the published walk")

    # The same fixtures the kernel asserts, so a disagreement is visible as a
    # disagreement rather than as two suites that happen to pass.
    cases = [
        ("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "checked", "base58check"),
        ("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNb", "broken", "base58check"),
        ("3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy", "checked", "base58check"),
        ("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4", "checked", "bech32"),
        ("bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0", "checked", "bech32"),
        ("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t5", "broken", "bech32"),
        ("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed", "checked", "EVM hex"),
        ("0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359", "checked", "EVM hex"),
        ("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1Beaed", "broken", "EVM hex"),
        ("0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed", "unchecked", "EVM hex"),
        ("0x5AAEB6053F3E94C9B9A09F33669435E7EF1BEAED", "unchecked", "EVM hex"),
        ("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed.rig1", "checked", "EVM hex"),
        ("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa.gf63", "checked", "base58check"),
        ("rig1", "name", "-"),
        ("gf63-miner", "name", "-"),
        ("1rig", "name", "-"),
        ("bc-alpha1", "name", "-"),
        ("MixedCaseWorkerName123456789", "name", "-"),
        ("", "name", "-"),
    ]
    for s, want_v, want_k in cases:
        got_v, got_k = judge(s)
        claim((got_v, got_k) == (want_v, want_k),
              f"{s or '(empty)'} -> {want_v} {want_k}" + ("" if (got_v, got_k) == (want_v, want_k) else f" (got {got_v} {got_k})"))

    print("\nselftest passed" if ok else "\nselftest FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("address", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--miner", help="a MINER.TXT to read the worker line out of")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if a.miner:
        worker = None
        with open(a.miner, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("worker") and len(line.split(None, 1)) == 2:
                    worker = line.split(None, 1)[1].strip()
        if worker is None:
            print(f"{a.miner} names no worker, so the kernel would not start a miner from it")
            return 1
        print(f"{a.miner}:")
        return report(worker)

    if not a.address:
        ap.error("give an address, --miner FILE, or --selftest")
    rc = 0
    for s in a.address:
        rc |= report(s)
    return rc


if __name__ == "__main__":
    sys.exit(main())

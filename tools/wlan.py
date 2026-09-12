"""The numeric oracle for the wireless stack, and deliberately not the kernel's code.

`tokenizer.py --verify` makes the bargain this file makes: reimplement the
kernel's algorithm independently, run both over the same input, and diff. A
second implementation that agrees is evidence; one written by transcribing the
first is a spelling check.

So this is pure Python with no dependencies -- AES from the FIPS-197 tables,
CCM from RFC 3610, CMAC from RFC 4493 -- and it exists because a bug in CCMP
produces frames that look perfectly well-formed and do not decrypt, which is
indistinguishable from a radio problem, a key problem, or an access point
being difficult.

    python tools/wlan.py --selftest        # published vectors, both ways
    python tools/wlan.py --ccmp            # an 802.11 frame, encrypted
"""

import argparse
import sys

SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
]


def xtime(a):
    a <<= 1
    if a & 0x100:
        a = (a ^ 0x1B) & 0xFF
    return a & 0xFF


def gmul(a, b):
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        a = xtime(a)
        b >>= 1
    return p & 0xFF


def expand(key):
    nk = len(key) // 4
    rounds = {4: 10, 8: 14}[nk]
    total = 4 * (rounds + 1)
    w = [list(key[4 * i:4 * i + 4]) for i in range(nk)]
    rcon = 1
    for i in range(nk, total):
        t = list(w[i - 1])
        if i % nk == 0:
            t = [SBOX[t[1]] ^ rcon, SBOX[t[2]], SBOX[t[3]], SBOX[t[0]]]
            rcon = xtime(rcon)
        elif nk > 6 and i % nk == 4:
            t = [SBOX[x] for x in t]
        w.append([w[i - nk][k] ^ t[k] for k in range(4)])
    return [bytes(b for c in range(4) for b in w[r * 4 + c]) for r in range(rounds + 1)], rounds


def encrypt_block(rk, rounds, block):
    s = bytearray(x ^ y for x, y in zip(block, rk[0]))
    for r in range(1, rounds + 1):
        s = bytearray(SBOX[b] for b in s)
        # ShiftRows, on the column-major state AES actually uses.
        t = bytearray(16)
        for c in range(4):
            for row in range(4):
                t[c * 4 + row] = s[((c + row) % 4) * 4 + row]
        s = t
        if r != rounds:
            t = bytearray(16)
            for c in range(4):
                col = s[c * 4:c * 4 + 4]
                t[c * 4 + 0] = gmul(col[0], 2) ^ gmul(col[1], 3) ^ col[2] ^ col[3]
                t[c * 4 + 1] = col[0] ^ gmul(col[1], 2) ^ gmul(col[2], 3) ^ col[3]
                t[c * 4 + 2] = col[0] ^ col[1] ^ gmul(col[2], 2) ^ gmul(col[3], 3)
                t[c * 4 + 3] = gmul(col[0], 3) ^ col[1] ^ col[2] ^ gmul(col[3], 2)
            s = t
        s = bytearray(x ^ y for x, y in zip(s, rk[r]))
    return bytes(s)


class Aes:
    def __init__(self, key):
        self.rk, self.rounds = expand(key)

    def e(self, block):
        return encrypt_block(self.rk, self.rounds, block)


def xor(a, b):
    return bytes(x ^ y for x, y in zip(a, b))


def ccm_encrypt(key, nonce, aad, plain, mlen=8, llen=2):
    """RFC 3610. Returns (ciphertext, mic)."""
    aes = Aes(key)
    assert len(nonce) == 15 - llen

    # B_0: flags, nonce, length.
    flags = (0x40 if aad else 0) | (((mlen - 2) // 2) << 3) | (llen - 1)
    b0 = bytes([flags]) + nonce + len(plain).to_bytes(llen, "big")
    x = aes.e(b0)

    if aad:
        # Length-prefixed, then zero-padded to a block.
        if len(aad) < 0xFF00:
            pre = len(aad).to_bytes(2, "big")
        else:
            pre = b"\xff\xfe" + len(aad).to_bytes(4, "big")
        blk = pre + aad
        blk += b"\x00" * ((-len(blk)) % 16)
        for i in range(0, len(blk), 16):
            x = aes.e(xor(x, blk[i:i + 16]))

    pad = plain + b"\x00" * ((-len(plain)) % 16)
    for i in range(0, len(pad), 16):
        x = aes.e(xor(x, pad[i:i + 16]))
    tag = x[:mlen]

    # CTR. A_0 encrypts the tag; A_1.. encrypt the payload.
    def a(i):
        return bytes([llen - 1]) + nonce + i.to_bytes(llen, "big")

    s0 = aes.e(a(0))
    mic = xor(tag, s0[:mlen])

    out = bytearray()
    for i in range(0, len(plain), 16):
        s = aes.e(a(i // 16 + 1))
        chunk = plain[i:i + 16]
        out += xor(chunk, s[:len(chunk)])
    return bytes(out), mic


def ccm_decrypt(key, nonce, aad, cipher, mic, mlen=8, llen=2):
    """Returns plaintext, or None if the MIC does not verify."""
    aes = Aes(key)

    def a(i):
        return bytes([llen - 1]) + nonce + i.to_bytes(llen, "big")

    out = bytearray()
    for i in range(0, len(cipher), 16):
        s = aes.e(a(i // 16 + 1))
        chunk = cipher[i:i + 16]
        out += xor(chunk, s[:len(chunk)])
    plain = bytes(out)
    _, want = ccm_encrypt(key, nonce, aad, plain, mlen, llen)
    # Constant time is not the point in an offline oracle; correctness is.
    return plain if want == mic else None


def cmac(key, msg):
    """RFC 4493, AES-CMAC."""
    aes = Aes(key)
    zero = bytes(16)
    L = aes.e(zero)

    def dbl(b):
        n = int.from_bytes(b, "big") << 1
        n &= (1 << 128) - 1
        if b[0] & 0x80:
            n ^= 0x87
        return n.to_bytes(16, "big")

    k1 = dbl(L)
    k2 = dbl(k1)

    if len(msg) and len(msg) % 16 == 0:
        last = xor(msg[-16:], k1)
        body = msg[:-16]
    else:
        pad = msg[len(msg) - len(msg) % 16:]
        pad = pad + b"\x80" + b"\x00" * (15 - len(pad))
        last = xor(pad, k2)
        body = msg[:len(msg) - len(msg) % 16]

    x = bytes(16)
    for i in range(0, len(body), 16):
        x = aes.e(xor(x, body[i:i + 16]))
    return aes.e(xor(x, last))


def h(s):
    return bytes.fromhex(s.replace(" ", "").replace("\n", ""))


def selftest():
    ok = True

    def claim(name, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
        ok = ok and cond

    # FIPS-197 C.1, the AES-128 worked example.
    a = Aes(h("000102030405060708090a0b0c0d0e0f"))
    claim("aes-128  FIPS-197 C.1",
          a.e(h("00112233445566778899aabbccddeeff")) == h("69c4e0d86a7b0430d8cdb78070b4c55a"))

    # FIPS-197 C.3, AES-256, because CCMP is 128 but the key wrap path is not.
    a = Aes(h("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"))
    claim("aes-256  FIPS-197 C.3",
          a.e(h("00112233445566778899aabbccddeeff")) == h("8ea2b7ca516745bfeafc49904b496089"))

    # RFC 3610 packet vector #1.
    key = h("c0c1c2c3c4c5c6c7c8c9cacbcccdcecf")
    nonce = h("00000003020100a0a1a2a3a4a5")
    aad = h("0001020304050607")
    plain = h("08090a0b0c0d0e0f101112131415161718191a1b1c1d1e")
    c, m = ccm_encrypt(key, nonce, aad, plain, mlen=8, llen=2)
    claim("ccm      RFC 3610 vector #1 ciphertext",
          c == h("588c979a61c663d2f066d0c2c0f989806d5f6b61dac384"))
    claim("ccm      RFC 3610 vector #1 mic", m == h("17e8d12cfdf926e0"))
    claim("ccm      decrypt round-trips", ccm_decrypt(key, nonce, aad, c, m) == plain)
    bad = bytearray(c); bad[0] ^= 1
    claim("ccm      a flipped ciphertext bit is rejected",
          ccm_decrypt(key, nonce, aad, bytes(bad), m) is None)

    # RFC 3610 packet vector #2, which differs in length so a hardcoded
    # block count passes #1 and fails here.
    nonce2 = h("00000004030201a0a1a2a3a4a5")
    plain2 = h("08090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
    c2, m2 = ccm_encrypt(key, nonce2, aad, plain2, mlen=8, llen=2)
    claim("ccm      RFC 3610 vector #2",
          c2 == h("72c91a36e135f8cf291ca894085c87e3cc15c439c9e43a3b")
          and m2 == h("a091d56e10400916"))

    # RFC 4493 AES-CMAC.
    ck = h("2b7e151628aed2a6abf7158809cf4f3c")
    claim("cmac     RFC 4493 empty", cmac(ck, b"") == h("bb1d6929e95937287fa37d129b756746"))
    claim("cmac     RFC 4493 16 bytes",
          cmac(ck, h("6bc1bee22e409f96e93d7e117393172a")) == h("070a16b46b4d4144f79bdd9dd04a287c"))
    claim("cmac     RFC 4493 40 bytes",
          cmac(ck, h("6bc1bee22e409f96e93d7e117393172a"
                     "ae2d8a571e03ac9c9eb76fac45af8e51"
                     "30c81c46a35ce411")) == h("dfa66747de9ae63030ca32611497c827"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest or True:
        print("[wlan.py] published vectors:")
        ok = selftest()
        print(f"[wlan.py] {'all passed' if ok else 'FAILURES above'}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

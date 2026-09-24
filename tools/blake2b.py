#!/usr/bin/env python3
"""The oracle for `cuda/blake2b.cuh`, and it is not a second implementation.

`hashlib.blake2b` is CPython's own C BLAKE2b and it takes both of the
parameter-block fields Equihash needs -- `digest_size` and `person` -- so
nothing here reimplements anything. That is the whole point: `algo.cuh` argues
that a fast wrong hash is indistinguishable from a fast right one from the
inside, and the only way out is to check against something written by somebody
else.

    python3 tools/blake2b.py

Prints the same four labels `cuda/blake2btest.cu` prints. Diff them.
"""
import hashlib
import sys

PERSON = b"ZcashPoW" + (192).to_bytes(4, "little") + (7).to_bytes(4, "little")


def line(label, msg, outlen, person=None):
    kw = {"digest_size": outlen}
    if person is not None:
        kw["person"] = person
    print("%-28s %s" % (label, hashlib.blake2b(msg, **kw).hexdigest()))


def main():
    big = bytes(range(200 % 256)) if False else bytes(i & 0xFF for i in range(200))
    line("abc/64", b"abc", 64)
    line("empty/64", b"", 64)
    line("200bytes/64", big, 64)
    line("abc/48/ZcashPoW-192-7", b"abc", 48, PERSON)

    # RFC 7693 Appendix A publishes exactly one BLAKE2b digest, and it is the
    # only value here that comes from the standard rather than from a library.
    # Checked so that a CPython whose BLAKE2b was wrong could not make both
    # sides of this comparison agree with each other and with nothing else.
    rfc = ("ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1"
           "7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923")
    got = hashlib.blake2b(b"abc", digest_size=64).hexdigest()
    print()
    if got == rfc:
        print("ok    hashlib agrees with RFC 7693 Appendix A on abc/64")
        return 0
    print("FAIL  hashlib does not match RFC 7693; the oracle itself is wrong")
    print("      rfc %s" % rfc)
    print("      got %s" % got)
    return 1


if __name__ == "__main__":
    sys.exit(main())

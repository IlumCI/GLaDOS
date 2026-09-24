#!/usr/bin/env python3
"""Equihash, host-side: generate, verify, and solve slowly enough to be right.

`design/equihash.md` names the hazards of writing the GPU solver and four of the
five are invisible without this file:

  - the tree walk-back recovering 2^k indices from a chain of pair pointers, in
    the right order, where "an off-by-one produces solutions that look
    structurally perfect -- right length, right index count -- and fail
    verification. No partial credit, no useful error message."
  - ordering canonicalisation, which "get it wrong and solutions are valid
    XOR-wise and rejected by the pool", the same silent signature as a wrong
    personalisation string.
  - bucket overflow, which "does not crash and does not corrupt: it *lowers the
    solution rate*, which is indistinguishable from bad luck."
  - the per-coin personalisation, which is the one that bites first.

Every one of those is a solver that produces plausible garbage. So the verifier
comes first, and it is written from the specification rather than from the
solver, because a verifier derived from the thing it checks agrees with it and
with nothing else -- `model.rs` makes that objection twice and `differ.rs` was
written before there was anything to differ.

### Why it is generic over (n, k) when only 192,7 is wanted

Because 192,7 cannot be solved here. It needs 2^25 initial hashes and Wagner's
rounds over them, which is hours in Python and gigabytes. But (48,5) is 512
entries and (96,5) is 131,072, and both are *the same algorithm* -- the
parameters appear only as bit widths. So the verifier is exercised against real
solutions found by a reference solver at sizes where a laptop finds them in
milliseconds, and then the identical verifier is what the GPU's 192,7 solutions
are judged by.

That is the only way to have a trustworthy verifier before there is anything to
verify.

### The hash agrees with the device already

The generator here is `hashlib.blake2b` with `digest_size` and `person`, which is
what `cuda/blake2b.cuh` was checked against -- four vectors, identical, including
48 bytes out under `ZcashPoW`/192/7. So this file and the GPU share a hash
without sharing an implementation.

    python3 tools/equihash.py --selftest
    python3 tools/equihash.py --solve 48 5
"""
import argparse
import hashlib
import sys


def params(n, k):
    """The four numbers every other function needs, derived rather than tabled.

    `collision` is n/(k+1) and everything else falls out of it. A table would be
    a second place for 192,7 to be written down, and `design/mining.md` records
    what a per-coin parameter table costs when it disagrees with the wire.
    """
    if n % 8 or k < 1 or n % (k + 1):
        raise ValueError("n must be a multiple of 8 and of (k+1)")
    collision = n // (k + 1)
    per_output = 512 // n
    if per_output < 1:
        raise ValueError("n too large for one BLAKE2b output")
    return {
        "n": n,
        "k": k,
        "collision": collision,
        "per_output": per_output,
        # Bytes of digest asked for. Not 64: the length enters BLAKE2b's state
        # before the message, so this is a different function from a truncated
        # 64-byte one, which is the thing `cuda/blake2b.cuh` says at length.
        "hash_len": per_output * (n // 8),
        "slice_len": n // 8,
        "index_bits": collision + 1,
        "solution_len": 1 << k,
    }


def person(n, k):
    """`"ZcashPoW" || le32(n) || le32(k)`.

    A coin that uses its own string passes it instead. This is the hazard
    `design/equihash.md` lists first, and what getting it wrong looks like is
    every share rejected on some chains and accepted on others, with the solver
    correct and the solutions structurally perfect -- solutions to a different
    puzzle. So it is never a constant anywhere below.
    """
    return b"ZcashPoW" + n.to_bytes(4, "little") + k.to_bytes(4, "little")


def generate(header, index, p, pers):
    """The n/8 bytes that index `index` contributes.

    One BLAKE2b covers `per_output` indices, which is why the counter fed to the
    hash is `index // per_output` and the slice taken is `index % per_output`.
    Hashing the index directly would be a different and much slower algorithm
    that happens to produce plausible output.
    """
    g = hashlib.blake2b(digest_size=p["hash_len"], person=pers)
    g.update(header)
    g.update((index // p["per_output"]).to_bytes(4, "little"))
    d = g.digest()
    lo = (index % p["per_output"]) * p["slice_len"]
    return d[lo:lo + p["slice_len"]]


def bit_slice(b, start, length):
    """`length` bits of `b` starting at bit `start`, most significant first."""
    v = int.from_bytes(b, "big")
    total = len(b) * 8
    if start + length > total:
        raise ValueError("bit range past the end")
    return (v >> (total - start - length)) & ((1 << length) - 1)


def verify(header, indices, n, k, pers=None):
    """Is this a solution? Answers `(ok, why)` rather than a bool.

    The reason matters more here than in most predicates: a solver under
    development produces solutions that fail for exactly one of five reasons and
    `design/equihash.md` says each failure is silent. "No" is not a useful answer
    to somebody holding 400 bytes of indices.
    """
    p = params(n, k)
    pers = pers if pers is not None else person(n, k)

    if len(indices) != p["solution_len"]:
        return False, "wrong length: %d indices, wanted %d" % (len(indices), p["solution_len"])
    limit = 1 << p["index_bits"]
    for i in indices:
        if not 0 <= i < limit:
            return False, "index %d is outside 0..2^%d" % (i, p["index_bits"])
    if len(set(indices)) != len(indices):
        # The condition that stops a solution being padded with repeats. Cheap
        # to check and the one a solver breaks by forgetting to drop pairs that
        # share an index.
        return False, "indices are not distinct"

    # Each leaf carries its own index list, so ordering can be checked at every
    # level rather than only at the leaves. This is the "algorithm binding"
    # condition, and it is what makes a permutation of a solution not a second
    # solution.
    rows = [(generate(header, i, p, pers), [i]) for i in indices]

    for r in range(1, k + 1):
        if len(rows) % 2:
            return False, "round %d has an odd number of rows" % r
        nxt = []
        for j in range(0, len(rows), 2):
            (ha, ia), (hb, ib) = rows[j], rows[j + 1]
            lo = (r - 1) * p["collision"]
            if bit_slice(ha, lo, p["collision"]) != bit_slice(hb, lo, p["collision"]):
                return False, ("round %d pair %d does not collide on bits %d..%d"
                               % (r, j // 2, lo, lo + p["collision"]))
            # **Ordering, and this is the check that is invisible when wrong.**
            # A solution whose halves are swapped XORs to exactly the same value
            # and is not a valid solution. Comparing the first index of each half
            # is the canonical form, so a solver that emits either order is
            # producing one valid answer and one that a pool silently rejects.
            if ia[0] >= ib[0]:
                return False, ("round %d pair %d is out of order: %d then %d"
                               % (r, j // 2, ia[0], ib[0]))
            nxt.append((bytes(x ^ y for x, y in zip(ha, hb)), ia + ib))
        rows = nxt

    if len(rows) != 1:
        return False, "reduced to %d rows rather than one" % len(rows)
    if any(rows[0][0]):
        return False, "the final XOR is not zero"
    return True, "ok"


def solve(header, n, k, pers=None, limit=None):
    """A reference solver: Wagner's algorithm, written for clarity.

    Not competitive and not meant to be -- it exists so the verifier above has
    real solutions to be checked against, at parameters where that is seconds
    rather than hours. The GPU solver `design/equihash.md` describes is a
    different program and this is what it will be judged by.

    It keeps whole index lists rather than pair pointers, which is precisely the
    part the GPU cannot afford and the part the brief calls the hardest
    correctness problem there -- the tree walk-back. Doing it the expensive way
    here means a disagreement between the two is the walk-back's fault and
    nothing else's.
    """
    p = params(n, k)
    pers = pers if pers is not None else person(n, k)
    total = 1 << p["index_bits"]

    rows = [(generate(header, i, p, pers), [i]) for i in range(total)]

    for r in range(1, k + 1):
        lo = (r - 1) * p["collision"]
        buckets = {}
        for h, idx in rows:
            buckets.setdefault(bit_slice(h, lo, p["collision"]), []).append((h, idx))
        nxt = []
        for group in buckets.values():
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    ha, ia = group[a]
                    hb, ib = group[b]
                    # Sharing an index makes the XOR collapse and the solution
                    # invalid; dropping here is cheaper than at the end.
                    if set(ia) & set(ib):
                        continue
                    # Canonical order, so the solver cannot emit the swapped
                    # form the verifier refuses.
                    if ia[0] < ib[0]:
                        nxt.append((bytes(x ^ y for x, y in zip(ha, hb)), ia + ib))
                    else:
                        nxt.append((bytes(x ^ y for x, y in zip(hb, ha)), ib + ia))
        rows = nxt
        if not rows:
            return []

    out = []
    for h, idx in rows:
        if not any(h) and len(set(idx)) == len(idx):
            out.append(idx)
            if limit and len(out) >= limit:
                break
    return out


def pack(indices, index_bits):
    """Indices to the wire form: big-endian, `index_bits` each, no padding."""
    v = 0
    for i in indices:
        v = (v << index_bits) | i
    width = (len(indices) * index_bits + 7) // 8
    return v.to_bytes(width, "big")


def unpack(blob, count, index_bits):
    v = int.from_bytes(blob, "big")
    mask = (1 << index_bits) - 1
    out = []
    for j in range(count):
        shift = (count - 1 - j) * index_bits
        out.append((v >> shift) & mask)
    return out


def selftest():
    bad = 0

    def ok(c, w):
        nonlocal bad
        print("%s  %s" % ("ok  " if c else "FAIL", w))
        if not c:
            bad += 1

    # The parameters, derived. 192,7 is the target and is checked here even
    # though nothing below solves it, because getting these four numbers wrong
    # is the cheapest possible way to waste a GPU solver.
    p = params(192, 7)
    ok(p["collision"] == 24, "192,7 collides on 24 bits")
    ok(p["per_output"] == 2, "and packs 2 indices per BLAKE2b output")
    ok(p["hash_len"] == 48, "so it asks BLAKE2b for 48 bytes, not 64")
    ok(p["index_bits"] == 25, "an index is 25 bits")
    ok(p["solution_len"] == 128, "and a solution is 128 of them")
    ok(pack(range(128), 25).__len__() == 400, "which is 400 bytes on the wire")
    ok(unpack(pack([1, 2, 3, 4], 25), 4, 25) == [1, 2, 3, 4], "and the packing round-trips")

    ok(person(192, 7) == b"ZcashPoW\xc0\x00\x00\x00\x07\x00\x00\x00",
       "the personalisation is ZcashPoW with n and k little-endian")
    # The vector `cuda/blake2b.cuh` was checked against, so this file and the
    # device are pinned to the same hash without sharing a line of code.
    ok(hashlib.blake2b(b"abc", digest_size=48, person=person(192, 7)).hexdigest()
       == "d936b5ac0dab75f94b06dc76b438ea4abd78202206812a1d9289039c3f37c08c"
          "87e86de2593095442ec9fa635a0939d2",
       "and it reproduces the digest the CUDA header was checked against")

    ok(bit_slice(b"\xff\x00", 0, 8) == 0xFF, "bit slicing reads from the top")
    ok(bit_slice(b"\x0f\xf0", 4, 8) == 0xFF, "and across a byte boundary")

    # The real work: solve small, then verify with the other function.
    for n, k in ((48, 5), (96, 5)):
        # **Nonces are ground, because that is what the algorithm is.** A given
        # header having no solution is the ordinary case and not a defect: the
        # expected count per nonce is around one, so a test that fixed the
        # header would fail for whichever parameters drew unlucky. Written as a
        # fixed header first and (48,5) duly had none -- 512 entries colliding
        # on 8 bits, and no amount of correct code makes a solution exist.
        #
        # How many it took is printed, because that count is the closest thing
        # here to a solution rate and a solver whose rate silently halved is
        # `design/equihash.md`'s bucket-overflow hazard exactly.
        sols, tries = [], 0
        for nonce in range(64):
            header = b"equihash selftest" + nonce.to_bytes(15, "little")
            tries += 1
            sols = solve(header, n, k, limit=1)
            if sols:
                break
        ok(bool(sols), "a solution exists for %d,%d within 64 nonces (took %d)" % (n, k, tries))
        if not sols:
            continue
        s = sols[0]
        good, why = verify(header, s, n, k)
        ok(good, "and the verifier accepts it (%d,%d): %s" % (n, k, why))

        # Each refusal, driven. These are the four silent failures
        # `design/equihash.md` warns about, made loud.
        swapped = list(s)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        g, why = verify(header, swapped, n, k)
        ok(not g and "out of order" in why,
           "swapping a pair is refused for ordering, not XOR (%s)" % why)

        dup = list(s)
        dup[-1] = dup[0]
        g, why = verify(header, dup, n, k)
        ok(not g, "a repeated index is refused (%s)" % why)

        short = list(s)[:-1]
        g, why = verify(header, short, n, k)
        ok(not g and "wrong length" in why, "a short solution is refused (%s)" % why)

        other = b"equihash selftest, but not the same header"
        g, why = verify(other, s, n, k)
        ok(not g, "and the same indices under another header are refused (%s)" % why)

        # The wrong personalisation, which is hazard one. Same indices, same
        # header, a string one byte different.
        g, why = verify(header, s, n, k, pers=person(n, k)[:-1] + b"\x01")
        ok(not g, "a solution under the wrong personalisation is refused (%s)" % why)

        blob = pack(s, params(n, k)["index_bits"])
        back = unpack(blob, len(s), params(n, k)["index_bits"])
        ok(back == list(s), "and the wire form round-trips at %d,%d" % (n, k))

    print()
    print("%d claim(s) failed" % bad if bad else "every claim held")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--solve", nargs=2, type=int, metavar=("N", "K"))
    ap.add_argument("--header", default="equihash")
    ap.add_argument("--limit", type=int, default=1)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.solve:
        n, k = a.solve
        header = a.header.encode()
        sols = solve(header, n, k, limit=a.limit)
        print("%d solution(s) for %d,%d" % (len(sols), n, k))
        for s in sols:
            good, why = verify(header, s, n, k)
            print("  %s  %s" % ("ok  " if good else "FAIL", why))
            print("  %s" % pack(s, params(n, k)["index_bits"]).hex())
        return 0 if sols else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

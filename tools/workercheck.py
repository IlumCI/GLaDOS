#!/usr/bin/env python3
"""Register a worker name against the live `worker` function, and release it again.

**Why this exists rather than a curl in the README.** The non-custodial claim in
`design/runbook.md` rests on one exchange: a miner proves it holds the key for a
payout address, and the pool records the name against it. Nothing in this tree
had ever *performed* that exchange -- `supabase/README.md` documents the four
routes and `pool/deploy/run-pool.sh` reads the map, so the write side was
described and never driven. A route that answers 405 to a GET has told you it is
deployed and nothing about whether it works.

**The signer is written here and the verifier is `_shared/evm.js`**, which is the
differ idiom this tree uses everywhere two implementations must agree: one in
Python over `hashlib`-free keccak, one in JavaScript over Deno, and if they
disagree about a recovery id or a checksum the exchange fails rather than
half-working.

**It rebuilds the message instead of signing what it was handed.** The nonce
route answers `message`, and signing that is what a careless client does: a
server -- or anything between -- could ask for a signature over different text
and a client that signs the string it received would provide it. So the SIWE
text is reconstructed from the fields, compared byte for byte, and a mismatch
refuses *before* anything is signed. That check is the only thing here a real
wallet does by showing the text to a person.

**The key is thrown away and the name is released.** Nothing about this run
should survive it: the address is derived from a fixed test key, which is
therefore public and must never hold anything, and the name is released on the
way out so the workers table is left as it was found. `--keep` skips the release
for looking at the row.

    python3 tools/workercheck.py --selftest
    python3 tools/workercheck.py                       # against what channel.rs pins
    python3 tools/workercheck.py --base http://localhost:54321/functions/v1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from addrcheck import keccak256  # noqa: E402  the one we already have two readers for

# secp256k1. Written out rather than imported: `coincurve` is not installed here
# and a check that needs a package nobody has is a check nobody runs.
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


def _add(p, q):
    if p is None:
        return q
    if q is None:
        return p
    (x1, y1), (x2, y2) = p, q
    if x1 == x2 and (y1 + y2) % P == 0:
        return None
    if p == q:
        lam = 3 * x1 * x1 * pow(2 * y1, P - 2, P) % P
    else:
        lam = (y2 - y1) * pow(x2 - x1, P - 2, P) % P
    x3 = (lam * lam - x1 - x2) % P
    return (x3, (lam * (x1 - x3) - y1) % P)


def _mul(k, p=(GX, GY)):
    r = None
    while k:
        if k & 1:
            r = _add(r, p)
        p = _add(p, p)
        k >>= 1
    return r


def address_of(priv: int) -> str:
    """The address, which is the last 20 bytes of the keccak of the public point."""
    x, y = _mul(priv)
    return "0x" + keccak256(x.to_bytes(32, "big") + y.to_bytes(32, "big"))[12:].hex()


def to_checksum(addr: str) -> str:
    """EIP-55: the case of each hex letter is a bit of a keccak over the lowercase.

    Reimplemented rather than imported from `_shared/evm.js`, because that is the
    implementation under test: two readers of one rule, which is the only
    arrangement that catches one of them being wrong.
    """
    low = addr.lower().removeprefix("0x")
    h = keccak256(low.encode()).hex()
    return "0x" + "".join(c.upper() if c.isalpha() and int(h[i], 16) >= 8 else c
                          for i, c in enumerate(low))


def sign(priv: int, message: str) -> str:
    """The signature alone, which is all a caller outside the selftest wants."""
    return sign_parts(priv, message)[0]


def sign_parts(priv: int, message: str) -> tuple[str, bool]:
    """An EIP-191 personal_sign signature, and whether `s` had to be normalised.

    **The flag is here because it cannot be recovered from the signature.** After
    normalisation `s` is always in the lower half, so a finished signature carries
    no trace of which branch produced it -- and the selftest's first attempt to
    assert coverage read the flag off the output and concluded "16 flipped, 0 not"
    on a run that was plainly half and half. A branch a test cannot observe is a
    branch a test cannot claim to have taken, so the signer reports it.

    **`s` is forced into the lower half and the recovery bit adjusted with it.**
    Both `(r, s)` and `(r, N - s)` verify, so a high-`s` signature is valid and is
    rejected by a verifier enforcing malleability rules -- and which of those
    `_shared/evm.js` does is not something to find out from an intermittent
    failure.
    """
    pre = f"\x19Ethereum Signed Message:\n{len(message.encode())}".encode() + message.encode()
    z = int.from_bytes(keccak256(pre), "big")
    # Deterministic k, in the spirit of RFC 6979 without being it: this signs a
    # throwaway test key over a nonce the server just issued, so the danger a
    # random k protects against -- two signatures sharing one k revealing the key
    # -- is about a key that matters. Derived from the digest and the key so a
    # failure is reproducible, which is worth more here than being RFC 6979.
    for bump in range(256):
        k = int.from_bytes(keccak256(priv.to_bytes(32, "big") + z.to_bytes(32, "big")
                                     + bytes([bump])), "big") % N
        if k == 0:
            continue
        pt = _mul(k)
        r = pt[0] % N
        if r == 0:
            continue
        s = pow(k, N - 2, N) * (z + r * priv) % N
        if s == 0:
            continue
        flip = s > N // 2
        # **The parity is flipped on the recovery bit, before 27 is added.** The
        # first version of this wrote `v ^ 1` on the already-offset value, and
        # `27 ^ 1` is 26 -- not a recovery id at all, so `_shared/evm.js` refused
        # every signature that happened to need the low-`s` normalisation and
        # accepted every one that did not. It ran correctly the first time, which
        # is the worst number of times for a thing to work: the selftest signed one
        # message, that message did not need the flip, and the claim asserting
        # `v in (27, 28)` passed on a branch it never entered.
        rec = (pt[1] & 1) | (2 if pt[0] >= N else 0)
        if s > N // 2:
            s, rec = N - s, rec ^ 1
        v = 27 + rec
        return ("0x" + r.to_bytes(32, "big").hex() + s.to_bytes(32, "big").hex()
                + bytes([v]).hex(), flip)
    raise AssertionError("no signature after 256 bumps, which cannot happen")


def siwe(domain: str, address: str, worker: str, verb: str, nonce: str,
         chain_id: int, issued: str, expires: str) -> str:
    """The message `worker/index.ts` builds, rebuilt from its parts.

    Kept line for line against that function on purpose: this is the comparison,
    so a difference of one space has to show up as a refusal rather than being
    accommodated here.
    """
    return "\n".join([
        f"{domain} wants you to sign in with your Ethereum account:",
        to_checksum(address),
        "",
        f'{verb} the mining worker name "{worker}" for this address.',
        "This signature costs nothing, moves nothing, and approves no transaction.",
        "",
        f"URI: https://{domain}/pool/",
        "Version: 1",
        f"Chain ID: {chain_id}",
        f"Nonce: {nonce}",
        f"Issued At: {issued}",
        f"Expiration Time: {expires}",
    ])


def post(base: str, route: str, body: dict, timeout: float = 30.0):
    req = urllib.request.Request(
        f"{base.rstrip('/')}/{route}", method="POST",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw or "{}")
        except json.JSONDecodeError:
            return e.code, {"error": raw[:200]}


def get(base: str, route: str, timeout: float = 30.0):
    try:
        with urllib.request.urlopen(f"{base.rstrip('/')}/{route}", timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:200]}


def pinned_base() -> str:
    """The origin the kernel asks, so a check cannot test a host no image names."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "src", "update", "channel.rs"), encoding="utf-8").read()
    m = re.search(r'pub\s+const\s+DEFAULT_SOURCE\s*:\s*&\s*str\s*=\s*"([^"]*)"\s*;', src)
    if not m:
        sys.exit("no DEFAULT_SOURCE in src/update/channel.rs")
    return m.group(1).rstrip("/") + "/functions/v1"


# A key that is published here and therefore worthless. It exists so the address
# is stable across runs: a fresh key per run would leave a new row behind every
# time the release leg failed, and the table would fill with addresses nobody
# could ever explain.
TEST_KEY = 0x0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF


def run(base: str, worker: str, keep: bool, domain: str, chain_id: int) -> int:
    addr = address_of(TEST_KEY)
    print(f"base    {base}")
    print(f"address {to_checksum(addr)}  (a published test key -- holds nothing)")
    print(f"worker  {worker}")
    bad = 0
    replay = None

    def step(label, ok, detail=""):
        nonlocal bad
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f" -- {detail}" if detail else ""))
        if not ok:
            bad += 1
        return ok

    for verb_route, verb_word in (("claim", "Claim"), ("release", "Release")):
        if verb_route == "release" and keep:
            print("  --keep, so the name is left registered")
            break
        print(f"== {verb_route}")
        code, body = post(base, "worker/nonce",
                          {"address": addr, "worker": worker,
                           "verb": "release" if verb_route == "release" else "claim"})
        if not step("a nonce was issued", code == 200 and "nonce" in body,
                    f"http {code} {body.get('error', '')}"):
            return 1

        # The comparison this script exists for: what the server wants signed,
        # against what its own documented fields say it should want.
        mine = siwe(domain, addr, worker, verb_word, body["nonce"], chain_id,
                    body.get("issued_at", ""), body.get("expires_at", ""))
        theirs = body.get("message", "")
        same = mine == theirs
        if not step("the message it asks for is the message the fields describe",
                    same, "" if same else "rebuilt text differs -- NOT signing"):
            print("    theirs:", json.dumps(theirs)[:300])
            print("    mine:  ", json.dumps(mine)[:300])
            return 1

        sig = sign(TEST_KEY, theirs)
        if verb_route == "claim":
            replay = sig          # the real thing, for the leg below
        code, body2 = post(base, f"worker/{verb_route}",
                           {"address": addr, "worker": worker, "signature": sig})
        step("the signature was accepted and the name "
             + ("claimed" if verb_route == "claim" else "released"),
             code == 200, f"http {code} {body2.get('error', body2)}")

        code, m = get(base, "worker/map")
        listed = worker in (m.get("workers") or {})
        want = verb_route == "claim"
        step(f"the map {'lists' if want else 'no longer lists'} it",
             listed == want, f"http {code} workers={list((m.get('workers') or {}).keys())[:5]}")

    # **The replay is the first claim's own signature, sent again.**
    # The first version signed arbitrary text, which is refused at *recovery* --
    # so it passed while saying nothing about nonce burning, the property that
    # actually stops a captured signature being reused. A signature that was
    # genuinely accepted a moment ago is the only input that tells the two apart.
    print("== replay")
    if replay is None:
        print("  ..    skipped: no accepted signature to replay (--keep, or a leg failed)")
        print()
        print("every leg answered" if bad == 0 else f"{bad} leg(s) did not")
        return 1 if bad else 0
    code, body = post(base, "worker/claim",
                      {"address": addr, "worker": worker, "signature": replay})
    why = body.get("error", "")
    step("the signature that was just accepted is refused the second time",
         code >= 400, f"http {code} {why}")
    # **There are two honest refusals here and the checker cannot pick between
    # them, for a reason about the function worth writing down.** `spendNonce`
    # takes the *newest unused* nonce for the address, and nothing ever deletes an
    # unused one -- so a run that asked for a nonce and then failed before
    # spending it leaves that nonce outstanding forever. A replay then gets 401
    # "does not belong", because the message was rebuilt around some older
    # nonce, rather than 400 "no unused nonce". Both mean the replay failed; only
    # the second means it failed *because the nonce was burned*.
    # Asserting one of them made this leg fail on a table carrying two abandoned
    # nonces from earlier runs of this very script -- a check that depended on
    # nothing having gone wrong before it. So the claim is that the refusal is one
    # of the two, which can still fail on a 200, a 500 or a refusal about something
    # else entirely.
    ok_reasons = ("nonce" in why.lower(), "does not belong" in why.lower())
    step("and for one of the two reasons a spent nonce produces",
         code >= 400 and any(ok_reasons), f"http {code} {why}")
    if ok_reasons[1] and not ok_reasons[0]:
        print("    (an older unused nonce is outstanding for this address, so the")
        print("     rebuild used that one -- abandoned nonces are never collected)")

    print()
    print("every leg answered" if bad == 0 else f"{bad} leg(s) did not")
    return 1 if bad else 0


def selftest() -> int:
    """What can be checked with no server, which is the whole of the crypto."""
    bad = 0

    def claim(ok, label):
        nonlocal bad
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
        if not ok:
            bad += 1

    # The published vector for this key, so the curve arithmetic is checked
    # against something rather than against itself.
    claim(address_of(1) == "0x7e5f4552091a69125d5dfcb7b8c2659029395bdf",
          "the generator's own address is the published one")
    claim(address_of(2) == "0x2b5ad5c4795c026514f8317c7a215e218dccd6cf",
          "and so is the second point's")
    # EIP-55, against the checksummed form in the standard itself.
    claim(to_checksum("0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed")
          == "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed",
          "EIP-55 reproduces the standard's own example")
    claim(to_checksum("0xfb6916095ca1df60bb79ce92ce3ea74c37c5d359")
          == "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359",
          "and its second one")
    # A signature is only checked by recovering from it, which is what the
    # function under test does -- so recover here too and require the address back.
    msg = 'Claim the mining worker name "rig-a" for this address.'
    sig = sign(TEST_KEY, msg)  # the wrapper, so the ordinary call path is covered too
    claim(len(sig) == 2 + 65 * 2, "a signature is 65 bytes")
    r = int(sig[2:66], 16)
    s = int(sig[66:130], 16)
    v = int(sig[130:132], 16)
    claim(s <= N // 2, "s is in the lower half, so the signature is canonical")
    claim(v in (27, 28), "v is 27 or 28 for a key with no chain replay protection")
    # **Recovery over a run of messages, not one.** One message exercises one of
    # the two `s`-normalisation branches and says nothing about the other, which
    # is exactly how the `v ^ 1` bug above survived its own selftest. Sixteen
    # messages take both, and the suite asserts that both were taken -- a claim
    # that passed without entering a branch is the canary argument `differ.rs`
    # makes, arriving on arithmetic.
    want = address_of(TEST_KEY)
    flipped = plain = 0
    bad_rec = []
    for i in range(16):
        m = f'Claim the mining worker name "rig-{i}" for this address.'
        sg, flip = sign_parts(TEST_KEY, m)
        rr = int(sg[2:66], 16)
        ss = int(sg[66:130], 16)
        vv = int(sg[130:132], 16)
        if vv not in (27, 28):
            bad_rec.append(("v", i, vv))
            continue
        pre = f"\x19Ethereum Signed Message:\n{len(m.encode())}".encode() + m.encode()
        zz = int.from_bytes(keccak256(pre), "big")
        yy = pow((pow(rr, 3, P) + 7) % P, (P + 1) // 4, P)
        if (yy & 1) != (vv - 27):
            yy = P - yy
        pb = _add(_mul(pow(rr, N - 2, N) * ss % N, (rr, yy)),
                  _mul(N - pow(rr, N - 2, N) * zz % N))
        got = "0x" + keccak256(pb[0].to_bytes(32, "big") + pb[1].to_bytes(32, "big"))[12:].hex()
        if got != want:
            bad_rec.append(("addr", i, got))
        flipped += flip
        plain += not flip
    claim(not bad_rec, f"recovery returns the signing address for all 16 messages ({bad_rec[:2]})")
    claim(flipped and plain,
          f"and both s-normalisation branches were taken ({flipped} flipped, {plain} not)")
    # The rebuild check has to be able to fail, or it is decoration. Same
    # argument `differ.rs` makes about its canary.
    a = siwe("d", address_of(1), "w", "Claim", "n", 1, "i", "e")
    b = siwe("d", address_of(1), "w", "Release", "n", 1, "i", "e")
    claim(a != b, "the verb is inside the signed text, so one verb's signature is not the other's")
    c = siwe("d", address_of(1), "w2", "Claim", "n", 1, "i", "e")
    claim(a != c, "and so is the worker name")

    print()
    print("selftest passed" if bad == 0 else f"selftest FAILED: {bad}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", help="functions base URL; default is what channel.rs pins")
    ap.add_argument("--worker", default="checkrig", help="the name to register")
    ap.add_argument("--keep", action="store_true", help="do not release it afterwards")
    ap.add_argument("--domain", default="glados.aperture.institute")
    ap.add_argument("--chain-id", type=int, default=4663)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return run(a.base or pinned_base(), a.worker, a.keep, a.domain, a.chain_id)


if __name__ == "__main__":
    raise SystemExit(main())

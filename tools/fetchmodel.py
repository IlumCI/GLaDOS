#!/usr/bin/env python3
"""Fetch what a manifest in `tools/models/` names, and refuse anything else.

### Why this is not three lines of curl in a workflow

`payload.py` already argues the case one directory over: a truncated transfer
produces an ISO that builds, boots, and then cannot load the model, because
nothing else in the build knows how long the file should be. The same hazard
arrives here wearing different clothes -- a short GGUF is a server that starts
and answers nonsense, and a short tarball is a `llama-server` that is not
there at all.

So the digest is checked before anything runs, and a mismatch is a refusal
rather than a warning.

### The URL lives in the manifest, once

A `# from <url>` line, parsed rather than read as prose. The alternative is
the URL in the workflow and the digest in the manifest, which is two places
that must agree about one file and will eventually not -- the drift
`site.py` exists to remove from the download page, arriving on weights.

The three data columns stay exactly `payload.py`'s, so `payload.parse` reads
these files unchanged and there is one format for "bytes and what they must
hash to" in the tree rather than two.

    python3 tools/fetchmodel.py tools/models/bonsai-4b.txt --out out/models
    python3 tools/fetchmodel.py --selftest
"""

import argparse
import hashlib
import io
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Refused(Exception):
    """Something that is not what the manifest said it would be."""


def parse(text):
    """(url, [(name, size, sha256)]) out of a manifest."""
    url = None
    rows = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("# from "):
            candidate = s[len("# from "):].strip()
            # **A directive must name a URL, or it is prose.** This file's own
            # header wrapped the words "a build / from source is three to six
            # minutes" onto a line beginning `# from `, and the parser read it
            # as a second source and refused the manifest. Requiring a scheme
            # is what separates the two, and it fails the safe way: a typo'd
            # URL becomes prose, and then the "no source" refusal fires loudly
            # rather than something being fetched from a guess.
            if "://" not in candidate:
                continue
            if url is not None:
                raise Refused("the manifest names two sources")
            url = candidate
            continue
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) != 3:
            raise Refused(f"not sha256/size/name: {s!r}")
        digest, size, name = parts
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise Refused(f"{digest!r} is not a sha256")
        if not size.isdigit():
            raise Refused(f"{size!r} is not a size")
        rows.append((name, int(size), digest))
    if url is None:
        raise Refused("the manifest has no `# from <url>` line")
    if len(rows) != 1:
        raise Refused(f"expected exactly one file, found {len(rows)}")
    return url, rows


def digest_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(path, size, want):
    """Size first: it is free, and a truncation is the likely failure."""
    got = os.path.getsize(path)
    if got != size:
        raise Refused(f"{os.path.basename(path)} is {got} bytes, the manifest "
                      f"says {size} -- a short transfer, not a wrong file")
    d = digest_of(path)
    if d != want:
        raise Refused(f"{os.path.basename(path)} hashes to {d[:16]}..., the "
                      f"manifest says {want[:16]}...")
    return d


def fetch(manifest, out_dir, force=False):
    url, rows = parse(io.open(manifest, encoding="utf-8").read())
    name, size, want = rows[0]
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, name)

    # An existing file that already verifies is the cache hit, and it is
    # re-checked rather than trusted: a restored cache is a transfer too.
    if os.path.exists(dest) and not force:
        try:
            check(dest, size, want)
            print(f"  {name} is already here and verifies")
            return dest
        except Refused as e:
            print(f"  refetching: {e}")

    import urllib.request
    print(f"  fetching {name} ({size / 1e6:.1f} MB)")
    req = urllib.request.Request(url, headers={"User-Agent": "glados-loop"})
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
    # Verified before it takes the real name, so a failed fetch never leaves
    # something behind that the next run mistakes for a cache hit.
    check(tmp, size, want)
    os.replace(tmp, dest)
    print(f"  {name} verifies")
    return dest


# ----------------------------------------------------------------- selftest

GOOD = """# a manifest
# from file:///dev/null
{digest}  {size}  thing.bin
"""


def selftest():
    claims = []

    def claim(ok, what):
        claims.append(bool(ok))
        print("  %-4s %s" % ("ok" if ok else "FAIL", what))

    def refuses(fn, needle, what):
        try:
            fn()
            claim(False, what + " (it was accepted)")
        except Refused as e:
            claim(needle in str(e), what)

    with tempfile.TemporaryDirectory() as tmp:
        body = b"the bytes a manifest describes"
        p = os.path.join(tmp, "thing.bin")
        with open(p, "wb") as f:
            f.write(body)
        d = hashlib.sha256(body).hexdigest()

        man = GOOD.format(digest=d, size=len(body))
        url, rows = parse(man)
        claim(url == "file:///dev/null" and rows == [("thing.bin", len(body), d)],
              "a manifest parses to one source and one file")
        claim(check(p, len(body), d) == d, "and a matching file verifies")

        # The canary: every way the bytes could be wrong must be refused,
        # because a checker that has only ever said yes is one that reads
        # nothing. Size and digest are separate claims on purpose -- size is
        # free and catches the likely failure, which is a short transfer.
        refuses(lambda: check(p, len(body) + 1, d), "short transfer",
                "a file of the wrong length is refused as a truncation")
        refuses(lambda: check(p, len(body), "0" * 64), "hashes to",
                "and one of the right length and wrong content is refused")

        refuses(lambda: parse("# from https://a/x\n# from https://b/y\n"
                              + man.split("\n")[2]),
                "two sources", "a manifest naming two URLs is refused")
        refuses(lambda: parse(f"{d}  {len(body)}  thing.bin\n"),
                "no `# from", "a manifest with no source is refused")
        refuses(lambda: parse(man + f"{d}  1  other.bin\n"),
                "exactly one file", "a manifest naming two files is refused")
        refuses(lambda: parse("# from x://y\nnot-a-digest 1 n\n"),
                "is not a sha256", "a row whose digest is not one is refused")

        # Prose that begins like a directive is prose. This manifest's own
        # header wrapped "a build / from source is three to six minutes" onto
        # such a line and the first version of this parser refused the file
        # for naming two sources.
        prose = man.replace("# a manifest\n",
                            "# a manifest, rather than a build\n"
                            "# from source is slower than a download\n")
        u, r = parse(prose)
        claim(u == "file:///dev/null" and len(r) == 1,
              "a wrapped sentence beginning `# from ` is not a second source")

        # And the half that matters for a cache: a part-file left by a failed
        # fetch must not be mistaken for the real thing next time.
        claim(not os.path.exists(os.path.join(tmp, "thing.bin.part")),
              "a verified fetch leaves no .part behind")

    print()
    if all(claims):
        print("  fetchmodel passed (%d claims)" % len(claims))
        return 0
    print("  fetchmodel FAILED")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", nargs="?")
    ap.add_argument("--out", default="out/models")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.manifest:
        print("  which manifest?", file=sys.stderr)
        return 2
    try:
        print(fetch(a.manifest, a.out, a.force))
    except Refused as e:
        print(f"::error::{e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

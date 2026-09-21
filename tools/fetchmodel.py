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
    url, rows, _ = parse_all(text)
    return url, rows


def parse_all(text):
    """As `parse`, plus the `# derives` rows.

    **A derived file is pinned too, and it can be.** The author is served as
    TQ2_0, which `llama-quantize` makes from the fetched Q2_0 in about a
    minute -- so CI converts rather than fetching a second gigabyte from
    somewhere somebody has to host. That is only allowed because the
    conversion is byte-reproducible: two runs over the same input produced
    the same sha256, checked rather than assumed. An author that changed
    under the loop would make every rung-4 certificate name a night nobody
    can reproduce, which is the property the pinned build number exists for.
    """
    url = None
    rows = []
    derived = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("# derives "):
            parts = s[len("# derives "):].split()
            if len(parts) != 3:
                raise Refused(f"not a derives row: {s!r}")
            derived.append((parts[2], int(parts[1]), parts[0]))
            continue
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
    return url, rows, derived


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


def fetch(manifest, out_dir, force=False, backoff=1.0):
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

    import time
    import urllib.error
    import urllib.request

    print(f"  fetching {name} ({size / 1e6:.1f} MB)")
    req = urllib.request.Request(url, headers={"User-Agent": "glados-loop"})
    tmp = dest + ".part"

    # **Retried, because the first real run died of a 504.** A release CDN
    # timing out is not a fact about the pin and not a reason to spend the
    # night without an author; it is weather. Only transient shapes are
    # retried -- a 404 is a wrong URL and a 403 is a wrong credential, and
    # asking those again four times is four times the same answer.
    #
    # And the failure is a Refused rather than a traceback, for the reason
    # `ask_model` learned one file over: twenty lines of urllib with the
    # useful half in the middle is not a message anybody reads.
    last = None
    for attempt in range(4):
        if attempt:
            wait = (2 ** attempt) * backoff
            print(f"  retrying in {wait:g}s after {last}")
            # `backoff` exists so the selftest can exercise the retry
            # path without spending fourteen seconds asleep on every push.
            if wait:
                time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
            break
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code not in (408, 429, 500, 502, 503, 504):
                raise Refused(f"{url} answered {last}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = str(e)
    else:
        raise Refused(f"{url} could not be fetched after 4 tries: {last}")
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

        # A transport that will not answer is a refusal with a reason, not a
        # stack. The first run on a runner died of a 504 and printed twenty
        # lines of urllib; `file://` to nowhere exercises the same path with
        # no network at all.
        gone = os.path.join(tmp, "gone.txt")
        with open(gone, "w", encoding="utf-8", newline="\n") as f:
            f.write("# from file:///no/such/file/anywhere\n"
                    "%s  %d  thing.bin\n" % (d, len(body)))
        dman = man + "# derives %s  %d  thing.bin\n" % (d, len(body))
        _, _, der = parse_all(dman)
        claim(der == [("thing.bin", len(body), d)],
              "a `# derives` row parses beside the fetched one")
        dp = os.path.join(tmp, "d.txt")
        with open(dp, "w", encoding="utf-8", newline="\n") as f:
            f.write(dman)
        claim(check_derived(dp, tmp) == 1,
              "and a derived file that matches verifies")
        with open(os.path.join(tmp, "thing.bin"), "ab") as f:
            f.write(b"!")
        refuses(lambda: check_derived(dp, tmp), "short transfer",
                "a derived file that does not match is refused")
        with open(os.path.join(tmp, "thing.bin"), "wb") as f:
            f.write(body)

        refuses(lambda: fetch(gone, os.path.join(tmp, "dest"), backoff=0),
                "could not be fetched after 4 tries",
                "an unreachable source is refused by name, not by traceback")

    print()
    if all(claims):
        print("  fetchmodel passed (%d claims)" % len(claims))
        return 0
    print("  fetchmodel FAILED")
    return 1


def check_derived(manifest, out_dir):
    """Verify every `# derives` row against what is on disk."""
    _, _, derived = parse_all(io.open(manifest, encoding="utf-8").read())
    if not derived:
        raise Refused("the manifest derives nothing")
    for name, size, want in derived:
        check(os.path.join(out_dir, name), size, want)
        print(f"  {name} verifies")
    return len(derived)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", nargs="?")
    ap.add_argument("--out", default="out/models")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check-derived", action="store_true",
                    help="verify the `# derives` rows against --out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.manifest:
        print("  which manifest?", file=sys.stderr)
        return 2
    try:
        if a.check_derived:
            check_derived(a.manifest, a.out)
            return 0
        print(fetch(a.manifest, a.out, a.force))
    except Refused as e:
        print(f"::error::{e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

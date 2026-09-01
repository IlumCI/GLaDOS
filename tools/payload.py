#!/usr/bin/env python3
"""The ISO payload: recording what it is, and checking what arrived.

`mkiso.py --payload DIR` copies a directory into `\\GLADOS\\` on the image, and
that directory is the model, the tokenizer and the root bundle -- six hundred
megabytes that exist nowhere in this repository. So an ISO built anywhere but
this machine has to fetch them first, and a fetch is a thing that can go wrong
quietly.

### The failure this exists to prevent

A truncated download produces an ISO that builds without complaint, boots
without complaint, and then cannot load the model. Nothing in the build says
so, because nothing in the build knows how long `model.bin` was supposed to be.
The same is true of a byte flipped in transit: the loader reads a header it
believes and indexes into weights that moved.

So the sizes and digests are recorded here, in the repository, where they are a
few hundred bytes and travel with the source that expects them.

### What this checks, and what it does not

It checks that the bytes that arrived are the bytes that were recorded. That
is transport, and transport is what a CI download gets wrong.

It does **not** check that the payload is correct -- that `model.bin` is a
well-formed checkpoint, or that the tokenizer matches it. `convert.py` and
`tokenizer.py --verify` are what answer those, on the machine that produced
them, and no digest can substitute for either. Recording a digest of a wrong
file preserves the wrongness perfectly.

Usage:

    python tools/payload.py record esp/GLADOS payload/qwen3-0.6b.txt
    python tools/payload.py verify esp/GLADOS payload/qwen3-0.6b.txt
    python tools/payload.py --selftest
"""

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

# Read in chunks rather than whole. `model.bin` is 598 MB and a CI runner has
# seven gigabytes of RAM shared with everything else the build is doing;
# reading it whole to hash it is the kind of thing that works locally and dies
# on the runner.
CHUNK = 1 << 20


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def scan(root: Path):
    """Every file under `root`, sorted, with its size and digest.

    Sorted because the manifest is compared as text by a person as often as by
    this tool, and a listing whose order depends on the filesystem is one that
    shows spurious changes in a diff.
    """
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out.append((p.relative_to(root).as_posix(), p.stat().st_size, digest(p)))
    return out


def render(entries) -> str:
    lines = ["# payload manifest -- sha256, size, name", "#",
             "# Written by tools/payload.py. The bytes an ISO build must fetch",
             "# before it can run mkiso.py, and what they have to hash to."]
    for name, size, sha in entries:
        lines.append(f"{sha}  {size}  {name}")
    return "\n".join(lines) + "\n"


def parse(text: str):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            raise ValueError(f"malformed line: {line!r}")
        sha, size, name = parts
        out.append((name, int(size), sha))
    return out


def cmd_record(args) -> int:
    root = Path(args.directory)
    if not root.is_dir():
        print(f"  no such directory: {root}", file=sys.stderr)
        return 1
    entries = scan(root)
    if not entries:
        print(f"  {root} has no files in it", file=sys.stderr)
        return 1
    dst = Path(args.manifest)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(render(entries), encoding="utf-8")
    total = sum(e[1] for e in entries)
    print(f"  wrote {dst}")
    for name, size, _ in entries:
        print(f"    {name:24} {size:>12,} B")
    print(f"  {len(entries)} file(s), {total / 1e6:.1f} MB")
    return 0


def cmd_verify(args) -> int:
    root = Path(args.directory)
    want = parse(Path(args.manifest).read_text(encoding="utf-8"))

    bad = 0
    for name, size, sha in want:
        p = root / name
        if not p.is_file():
            print(f"  MISSING  {name}", file=sys.stderr)
            bad += 1
            continue
        # Size first: it is a stat and it catches the truncation case without
        # reading six hundred megabytes to find out.
        got_size = p.stat().st_size
        if got_size != size:
            print(f"  SHORT    {name}: {got_size:,} B, expected {size:,} B", file=sys.stderr)
            bad += 1
            continue
        got = digest(p)
        if got != sha:
            print(f"  DIGEST   {name}: {got[:16]}.. expected {sha[:16]}..", file=sys.stderr)
            bad += 1
            continue
        print(f"  ok       {name}  {size:,} B")

    # Anything present that the manifest does not name is refused too. An ISO
    # is built from the whole directory, so a stray file is a stray file on the
    # image -- and the one time that matters is when it is somebody's private
    # key sitting in the payload directory by accident.
    named = {n for n, _, _ in want}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if rel not in named:
                print(f"  EXTRA    {rel}: present but not in the manifest", file=sys.stderr)
                bad += 1

    if bad:
        print(f"  {bad} problem(s); refusing", file=sys.stderr)
        return 1
    print(f"  {len(want)} file(s) match")
    return 0


def selftest() -> int:
    ok = True

    def claim(what, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
        if not cond:
            ok = False

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "payload"
        (root / "sub").mkdir(parents=True)
        (root / "a.bin").write_bytes(b"hello" * 1000)
        (root / "sub" / "b.bin").write_bytes(b"world")
        man = Path(td) / "m.txt"

        class A:
            directory, manifest = str(root), str(man)

        claim("record writes a manifest", cmd_record(A()) == 0)
        claim("and it verifies against what it recorded", cmd_verify(A()) == 0)
        claim("nested files are included", "sub/b.bin" in man.read_text())

        # The case this whole file exists for.
        (root / "a.bin").write_bytes(b"hello" * 999)
        claim("a truncated file is refused", cmd_verify(A()) == 1)

        # Same length, different bytes: size alone would wave this through, so
        # it is the claim that says the digest is actually consulted.
        (root / "a.bin").write_bytes(b"hellO" * 1000)
        claim("a same-length corruption is refused", cmd_verify(A()) == 1)

        (root / "a.bin").write_bytes(b"hello" * 1000)
        claim("and it passes again once restored", cmd_verify(A()) == 0)

        (root / "a.bin").unlink()
        claim("a missing file is refused", cmd_verify(A()) == 1)

        (root / "a.bin").write_bytes(b"hello" * 1000)
        (root / "stray.key").write_bytes(b"oops")
        claim("a file not in the manifest is refused", cmd_verify(A()) == 1)

        # Round-trip through the text, since CI reads it back as text.
        (root / "stray.key").unlink()
        entries = scan(root)
        claim("the manifest round-trips through its own rendering",
              parse(render(entries)) == entries)

    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("record", "verify"):
        s = sub.add_parser(name)
        s.add_argument("directory")
        s.add_argument("manifest")

    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.cmd == "record":
        return cmd_record(args)
    if args.cmd == "verify":
        return cmd_verify(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Pack a directory into a GLaDOS package.

Deliberately not compressed. A compressed archive would mean carrying a
decompressor in the kernel for the sake of files the store already deduplicates
by content -- two packages sharing a file share it on disk whatever the archive
did on the way in.

And no checksum field. tlrc downloads a zip and then verifies a SHA256 carried
beside it; here the store addresses content by its hash, so a package whose
bytes are wrong is a different package rather than a corrupt one. There is no
window between having the bytes and knowing they are right, so there is nothing
for a checksum field to add.

    mkpkg.py <dir> <out.pkg> --name tldr --version 2.1 --summary "..."
"""

import argparse
import struct
import sys
from pathlib import Path

MAGIC = b"GLADOSPK"
VERSION = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--version", default="0")
    ap.add_argument("--summary", default="")
    ap.add_argument("--requires", default="")
    ap.add_argument("--max-bytes", type=int, default=4 * 1024 * 1024)
    args = ap.parse_args()

    if "/" in args.name or "\\" in args.name:
        sys.exit("a package name may not contain a path separator")

    files = []
    total = 0
    for p in sorted(args.dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(args.dir).as_posix()
        if rel.startswith("/") or ".." in rel.split("/"):
            sys.exit(f"refusing member outside the package: {rel}")
        data = p.read_bytes()
        total += len(data)
        if total > args.max_bytes:
            sys.exit(f"package exceeds {args.max_bytes} bytes; the whole thing "
                     "is read into the namespace at install time")
        files.append((rel, data))

    if not files:
        sys.exit(f"no files under {args.dir}")

    meta = (
        f"name: {args.name}\n"
        f"version: {args.version}\n"
        f"summary: {args.summary}\n"
        f"requires: {args.requires}\n"
    ).encode()

    body = bytearray()
    body += MAGIC
    body += struct.pack("<I", VERSION)
    body += struct.pack("<I", len(meta))
    body += meta
    body += struct.pack("<I", len(files))
    for rel, data in files:
        r = rel.encode()
        body += struct.pack("<H", len(r))
        body += r
        body += struct.pack("<I", len(data))
        body += data

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(body)
    print(f"  {args.name} {args.version}: {len(files)} files, {total:,} B "
          f"-> {args.out} ({len(body):,} B)")


if __name__ == "__main__":
    main()

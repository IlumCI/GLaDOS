#!/usr/bin/env python3
"""Fetch the two dynamic linkers, and a dynamically linked program for each.

Why this exists
---------------
`linux::load` reads `PT_INTERP` and loads whatever is at that path, so musl and
glibc are both supported by construction and neither is chosen by this kernel.
That claim has been checked against two hand-assembled fixtures, which prove
the mechanics and prove nothing at all about what a real dynamic linker asks
for. The `-ENOSYS` trace is what answers that, and it needs a real one.

So: `ld-musl-x86_64.so.1` out of Alpine's `musl` package, and
`ld-linux-x86-64.so.2` plus `libc.so.6` out of Debian's `libc6`. Both are
freely redistributable (musl is MIT, glibc is LGPL-2.1) and **neither goes in
this repository**, for the same reason no WAD and no checkpoint does: they are
somebody else's bytes. They land in `out/`, which is gitignored, and the sizes
and digests are recorded so a truncated download fails here rather than as a
linker that gets four instructions in.

A dynamically linked `busybox` comes from each distribution as the program to
run under them. Alpine's is linked against musl and Debian's against glibc,
which makes the pair the cleanest available comparison: one program, two
libcs, two traces.

What the report says, and why it is the interesting part
--------------------------------------------------------
For every file it reads back the ELF type, the interpreter it names, and its
`DT_NEEDED` list. That last one is the difference the whole libc question turns
on and it is visible in one line: musl's linker *is* the libc, one file that
needs nothing, while glibc's is a loader that then goes looking for `libc.so.6`
and whatever else the program wants. A kernel that can open one file is not
yet a kernel that can open the transitive closure of six.

Usage:

    python tools/libc.py fetch          # download and extract into out/libc
    python tools/libc.py report         # read back what is there
    python tools/libc.py --selftest     # the archive readers, on made-up input
"""

import argparse
import hashlib
import io
import re
import struct
import sys
import tarfile
import urllib.request
import zlib
from pathlib import Path

ALPINE = "https://dl-cdn.alpinelinux.org/alpine/v3.20/main/x86_64/"
DEBIAN = "https://deb.debian.org/debian/pool/main/"

# Pinned rather than resolved at fetch time. A version discovered from an index
# makes every run a different experiment, and the whole point of the trace is
# comparing two of them -- so the version is part of the record, and moving it
# is an edit somebody makes on purpose.
WANT = [
    # (name, url, member inside the archive, where it goes)
    ("musl", ALPINE + "musl-1.2.5-r3.apk",
     "lib/ld-musl-x86_64.so.1", "musl/lib/ld-musl-x86_64.so.1"),
    ("busybox-musl", ALPINE + "busybox-1.36.1-r31.apk",
     "bin/busybox", "musl/bin/busybox"),
    # Debian keeps the real linker under `lib/` and makes `lib64` a symlink to
    # it; the path a binary's `PT_INTERP` actually names is the symlink, so the
    # file is placed where the binaries look rather than where the package
    # filed it.
    ("glibc-ld", DEBIAN + "g/glibc/libc6_2.36-9+deb12u14_amd64.deb",
     "lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
     "glibc/lib64/ld-linux-x86-64.so.2"),
    ("glibc-libc", DEBIAN + "g/glibc/libc6_2.36-9+deb12u14_amd64.deb",
     "lib/x86_64-linux-gnu/libc.so.6", "glibc/lib/x86_64-linux-gnu/libc.so.6"),
    # Found by the closure below rather than known in advance, which is the
    # argument for computing it: Debian's busybox needs the resolver as well,
    # and a fetch list written from expectation would have been one file short
    # and failed at the third `open` inside `ld.so`.
    ("glibc-resolv", DEBIAN + "g/glibc/libc6_2.36-9+deb12u14_amd64.deb",
     "lib/x86_64-linux-gnu/libresolv.so.2",
     "glibc/lib/x86_64-linux-gnu/libresolv.so.2"),
    ("busybox-glibc", DEBIAN + "b/busybox/busybox_1.35.0-4+deb12u1+b1_amd64.deb",
     "bin/busybox", "glibc/bin/busybox"),
]

OUT = Path("out/libc")


# ------------------------------------------------------------------ archives

def gunzip_all(blob: bytes) -> bytes:
    """Decompress every gzip member, not just the first.

    An APK is three concatenated gzip streams -- signature, control, data --
    and every gzip reader in the standard library stops at the end of the
    first one. A reader that stops there finds an APKINDEX where it expected a
    filesystem and reports the package as empty, which looks exactly like a
    package that does not contain what you asked for.
    """
    out = bytearray()
    rest = blob
    while rest[:2] == b"\x1f\x8b":
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        out += d.decompress(rest)
        out += d.flush()
        rest = d.unused_data
    return bytes(out)


def ar_members(blob: bytes):
    """Walk a Unix `ar` archive, which is what a .deb is.

    Sixty-byte headers, even-byte alignment, and a size field that is decimal
    text rather than a number. Written out because `ar` is not on this machine
    and the format is thirty lines.
    """
    assert blob[:8] == b"!<arch>\n", "not an ar archive"
    at = 8
    while at + 60 <= len(blob):
        name = blob[at:at + 16].decode("ascii", "replace").strip()
        size = int(blob[at + 48:at + 58].decode("ascii").strip())
        body = blob[at + 60:at + 60 + size]
        yield name.rstrip("/"), body
        at += 60 + size + (size & 1)


def unpack(url: str, blob: bytes) -> tarfile.TarFile:
    """Answer the tar inside a .apk or a .deb, whichever this is."""
    if url.endswith(".apk"):
        return tarfile.open(fileobj=io.BytesIO(gunzip_all(blob)))
    if url.endswith(".deb"):
        for name, body in ar_members(blob):
            if name.startswith("data.tar"):
                if name.endswith(".zst"):
                    raise SystemExit(
                        "this .deb uses zstd, which is not in the standard library;\n"
                        "pick a release whose data.tar is .xz or .gz"
                    )
                return tarfile.open(fileobj=io.BytesIO(body))
        raise SystemExit("no data.tar member in the .deb")
    raise SystemExit(f"do not know how to open {url}")


def member(tf: tarfile.TarFile, want: str) -> bytes:
    """One file out of the tar, tolerating a leading './' and following one
    level of symlink -- which `libc.so.6` and `busybox` both are on some
    releases, and a reader that did not follow them would write a file
    containing a path."""
    names = {n.lstrip("./"): n for n in tf.getnames()}
    for _ in range(4):
        real = names.get(want)
        if real is None:
            raise SystemExit(f"{want} is not in this package")
        info = tf.getmember(real)
        if not info.issym() and not info.islnk():
            f = tf.extractfile(info)
            if f is None:
                raise SystemExit(f"{want} has no contents")
            return f.read()
        target = info.linkname
        if not target.startswith("/") and "/" in want:
            target = str(Path(want).parent / target)
        want = target.lstrip("/").replace("\\", "/")
    raise SystemExit(f"{want} is a symlink chain deeper than this follows")


# ---------------------------------------------------------------- elf, read

PT_LOAD, PT_DYNAMIC, PT_INTERP = 1, 2, 3
DT_NEEDED, DT_STRTAB, DT_SONAME = 1, 5, 14
KINDS = {1: "ET_REL", 2: "ET_EXEC", 3: "ET_DYN", 4: "ET_CORE"}


def describe(blob: bytes) -> dict:
    """Type, interpreter and DT_NEEDED, read straight out of the file.

    A third ELF reader in this tree, and deliberately: `elf.rs` is the kernel's
    and `mkelf.py --verify` checks fixtures this one has never seen. What is
    wanted here is the *dynamic* half, which neither of those reads, because
    until now nothing had a dynamic binary to read.
    """
    if blob[:4] != b"\x7fELF":
        return {"error": "not an ELF"}
    kind = struct.unpack_from("<H", blob, 16)[0]
    entry = struct.unpack_from("<Q", blob, 24)[0]
    phoff = struct.unpack_from("<Q", blob, 32)[0]
    phent, phnum = struct.unpack_from("<HH", blob, 54)

    loads, interp, dyn = [], None, None
    for i in range(phnum):
        at = phoff + i * phent
        ty = struct.unpack_from("<I", blob, at)[0]
        off, va = struct.unpack_from("<QQ", blob, at + 8)[0], struct.unpack_from("<Q", blob, at + 16)[0]
        fsz = struct.unpack_from("<Q", blob, at + 32)[0]
        if ty == PT_LOAD:
            loads.append((off, va, fsz))
        elif ty == PT_INTERP:
            interp = blob[off:off + fsz].split(b"\x00")[0].decode("ascii", "replace")
        elif ty == PT_DYNAMIC:
            dyn = (off, fsz)

    def at_vaddr(va):
        """File offset for a runtime address. The dynamic array holds
        addresses, and every one of them has to come back through here."""
        for off, base, fsz in loads:
            if base <= va < base + fsz:
                return off + (va - base)
        return None

    needed, soname = [], None
    if dyn:
        off, fsz = dyn
        entries = [struct.unpack_from("<qQ", blob, off + i * 16)
                   for i in range(fsz // 16)]
        strtab = next((v for t, v in entries if t == DT_STRTAB), None)
        base = at_vaddr(strtab) if strtab is not None else None
        if base is not None:
            def s(o):
                end = blob.index(b"\x00", base + o)
                return blob[base + o:end].decode("ascii", "replace")
            needed = [s(v) for t, v in entries if t == DT_NEEDED]
            soname = next((s(v) for t, v in entries if t == DT_SONAME), None)
    return {
        "kind": KINDS.get(kind, str(kind)),
        "entry": entry,
        "interp": interp,
        "needed": needed,
        "soname": soname,
        "size": len(blob),
    }


# ------------------------------------------------------------------ commands

def fetch() -> int:
    cache: dict[str, bytes] = {}
    for name, url, inside, dest in WANT:
        if url not in cache:
            print(f"  fetching {url.rsplit('/', 1)[-1]}")
            with urllib.request.urlopen(url, timeout=120) as r:
                cache[url] = r.read()
            print(f"    {len(cache[url]):,} bytes")
        blob = member(unpack(url, cache[url]), inside)
        path = OUT / dest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        print(f"  {name:<14} {len(blob):>9,} B  -> {path.as_posix()}")
    print()
    print("  Not committed, and never will be: these are somebody else's bytes,")
    print("  the same rule that keeps every WAD and every checkpoint out.")
    return 0


def report() -> int:
    if not OUT.exists():
        print("  nothing fetched yet: run 'python tools/libc.py fetch'")
        return 1
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    seen = {}
    for p in files:
        d = describe(p.read_bytes())
        rel = p.relative_to(OUT).as_posix()
        if "error" in d:
            print(f"  {rel}: {d['error']}")
            continue
        seen[rel] = d
        print(f"  {rel}")
        print(f"      {d['kind']}, {d['size']:,} B, entry {d['entry']:#x}"
              + (f", soname {d['soname']}" if d["soname"] else ""))
        print(f"      interpreter: {d['interp'] or 'none, it is one'}")
        print(f"      needs: {', '.join(d['needed']) if d['needed'] else 'nothing'}")
        print(f"      sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")

    # What each program actually costs to start, which is the number the libc
    # question turns on. A kernel that can open one file is not yet a kernel
    # that can open the transitive closure of six, and the closure is not
    # something to guess at: `DT_NEEDED` names direct dependencies only, and
    # every one of them has its own.
    print()
    print("  what it takes to start each program:")
    by_soname = {d["soname"]: (rel, d) for rel, d in seen.items() if d["soname"]}
    for rel, d in sorted(seen.items()):
        if not d["interp"]:
            continue
        need, out, missing = list(d["needed"]), [], []
        while need:
            name = need.pop(0)
            if name in [n for n, _ in out] or name in missing:
                continue
            hit = by_soname.get(name)
            if hit is None:
                missing.append(name)
                continue
            out.append((name, hit[0]))
            need += hit[1]["needed"]
        # Count distinct *files*, not distinct names. musl's linker carries the
        # soname of the libc, so a program asks for two names and both resolve
        # to one file -- and counting names would report musl as costing three
        # opens when it costs two. That is the whole difference being measured,
        # so it is the one number worth getting exactly right.
        paths = {f for _, f in out} | {rel}
        total = sum(seen[f]["size"] for f in paths)
        print(f"    {rel}")
        print(f"      interpreter {d['interp']}")
        print(f"      names {len(out)} object(s): {', '.join(n for n, _ in out) or 'none'}")
        if missing:
            print(f"      MISSING: {', '.join(missing)}")
        print(f"      {len(paths)} distinct file(s), {total:,} B resident")
    return 0


def selftest() -> int:
    ok = True

    def claim(what, good):
        nonlocal ok
        print(("  ok   " if good else "  FAIL ") + what)
        ok = ok and good

    # Three gzip members back to back, which is the shape that defeats every
    # reader that stops at the first.
    parts = [b"one", b"two", b"three"]
    blob = b""
    for part in parts:
        c = zlib.compressobj(9, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
        blob += c.compress(part) + c.flush()
    claim("every gzip member is read, not only the first",
          gunzip_all(blob) == b"".join(parts))

    # An ar archive with an odd-sized member, so the padding byte is exercised.
    def entry(name, body):
        h = name.ljust(16) + " " * 12 + "0".ljust(6) + "0".ljust(6) + "100644".ljust(8)
        h += str(len(body)).ljust(10) + "`\n"
        return h.encode() + body + (b"\n" if len(body) & 1 else b"")

    ar = b"!<arch>\n" + entry("debian-binary", b"2.0\n") + entry("data.tar", b"xyz")
    got = dict(ar_members(ar))
    claim("an ar archive with an odd-length member does not lose its alignment",
          got.get("debian-binary") == b"2.0\n" and got.get("data.tar") == b"xyz")

    claim("a file that is not an ELF says so rather than decoding noise",
          describe(b"not an elf at all")["error"] == "not an ELF")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", default="report",
                    choices=["fetch", "report"])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return fetch() if a.action == "fetch" else report()


if __name__ == "__main__":
    sys.exit(main())

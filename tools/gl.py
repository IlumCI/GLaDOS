#!/usr/bin/env python3
"""Fetch a real OpenGL for the guest, and it is 8 MiB rather than 8,000 lines.

**GL is not kernel work and that is the whole finding.** Every OpenGL on Linux
is a userspace shared object: `libGL.so` in software mode rasterises into
ordinary memory and asks the kernel for nothing but pages. So "implement GL"
here is not a rasteriser to write, it is a library to get into the guest and a
syscall trace to answer -- and this machine already has the framebuffer to show
the result, the input to drive it, and the threads to run it on.

### Which Mesa, and why an eight-year-old one

Debian bookworm's `libosmesa6` closure is **188 MiB**, of which 112 is
`libLLVM15` and another 57 is LLVM's own dependencies. It is there for
llvmpipe, the JIT rasteriser, and Debian builds llvmpipe and softpipe into one
object so the JIT cannot be declined. Against `OPEN_MAX_BYTES` of 64 MiB --
`fs.rs` holds an open file's whole contents -- that is not close.

Mesa 13 predates that. Its `libOSMesa` is the *classic* software rasteriser,
no gallium and no LLVM, and the whole closure is **8.2 MiB across six
objects**, the largest of them 4 MiB. It is stretch's, so it is archived
rather than current, and it is pinned here for the reason `libc.py` pins
everything: a version discovered at fetch time makes every run a different
experiment.

What it costs is version. Mesa 13 is OpenGL 3.0-era, which is a decade behind
and is still three whole major versions past the 1.1 anybody asked for.

### What OSMesa is, since it is not what a program usually links

`libGL.so` gets its drawing surface from GLX, which needs an X server, or from
EGL, which needs DRM. There is neither here. `libOSMesa` is the third door: a
context bound to *a buffer the caller owns*, with `OSMesaMakeCurrent` taking a
pointer and a size. Every `gl*` call after that writes into that buffer, and
the buffer goes to `/dev/fb0` with one `write`.

That is exactly the shape this machine can serve, and it is why this is the
library to fetch rather than `libgl1`, whose closure drags in X11 as well.

    .\\tools\\venv\\Scripts\\python.exe tools\\gl.py --fetch
    .\\tools\\venv\\Scripts\\python.exe tools\\gl.py --report
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import libc  # noqa: E402  (the path has to be set first)

ARCHIVE = "https://archive.debian.org/debian/pool/main/"

# Pinned, not resolved. See the module docstring, and `libc.py`'s WANT for the
# same argument made once already.
WANT = [
    ("osmesa", ARCHIVE + "m/mesa/libosmesa6_13.0.6-1+b2_amd64.deb",
     "usr/lib/x86_64-linux-gnu/libOSMesa.so.8.0.0",
     "gl/usr/lib/x86_64-linux-gnu/libOSMesa.so.8"),
    ("glapi", ARCHIVE + "m/mesa/libglapi-mesa_13.0.6-1+b2_amd64.deb",
     "usr/lib/x86_64-linux-gnu/libglapi.so.0.0.0",
     "gl/usr/lib/x86_64-linux-gnu/libglapi.so.0"),
    # Mesa is C++ in places, so the runtime comes with it. Stretch's rather
    # than bookworm's, because a newer `libstdc++` is a superset and the older
    # one is what this Mesa was linked against.
    ("stdc++", ARCHIVE + "g/gcc-6/libstdc++6_6.3.0-18+deb9u1_amd64.deb",
     "usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.22",
     "gl/usr/lib/x86_64-linux-gnu/libstdc++.so.6"),
    # Surprising until you look: Mesa 13 links libgcrypt for its shader cache
    # hashing. Kept because the closure says so rather than because anybody
    # predicted it, which is the same way `libresolv` turned up for busybox.
    ("gcrypt", ARCHIVE + "libg/libgcrypt20/libgcrypt20_1.7.6-2+deb9u3_amd64.deb",
     "lib/x86_64-linux-gnu/libgcrypt.so.20.1.6",
     "gl/lib/x86_64-linux-gnu/libgcrypt.so.20"),
    ("gpg-error", ARCHIVE + "libg/libgpg-error/libgpg-error0_1.26-2_amd64.deb",
     "lib/x86_64-linux-gnu/libgpg-error.so.0.21.0",
     "gl/lib/x86_64-linux-gnu/libgpg-error.so.0"),
    # `libgcc_s` comes from bookworm, matching the glibc already staged: it is
    # the unwinder and the one object where mixing vintages would matter.
    ("gcc_s", libc.DEBIAN + "g/gcc-12/libgcc-s1_12.2.0-14+deb12u1_amd64.deb",
     "lib/x86_64-linux-gnu/libgcc_s.so.1",
     "gl/lib/x86_64-linux-gnu/libgcc_s.so.1"),
    # The three glibc satellites `libc.py` never needed. Since glibc 2.34
    # `libdl` and `libpthread` are empty stubs -- their contents moved into
    # `libc.so.6` -- and they exist only so an object linked before the merge
    # still resolves its `DT_NEEDED`. Fourteen kilobytes each, and without
    # them `ld.so` stops on a name it cannot find. `libm` is real.
    ("libm", libc.DEBIAN + "g/glibc/libc6_2.36-9+deb12u14_amd64.deb",
     "lib/x86_64-linux-gnu/libm.so.6", "gl/lib/x86_64-linux-gnu/libm.so.6"),
    ("libdl", libc.DEBIAN + "g/glibc/libc6_2.36-9+deb12u14_amd64.deb",
     "lib/x86_64-linux-gnu/libdl.so.2", "gl/lib/x86_64-linux-gnu/libdl.so.2"),
    ("libpthread", libc.DEBIAN + "g/glibc/libc6_2.36-9+deb12u14_amd64.deb",
     "lib/x86_64-linux-gnu/libpthread.so.0",
     "gl/lib/x86_64-linux-gnu/libpthread.so.0"),
]

OUT = Path("out/gl")

# What a libc already staged provides, so the closure does not report it as
# missing. These are the names `libc.py` puts under `glibc/`.
FROM_LIBC = {
    "libc.so.6",
    "libresolv.so.2",
    "ld-linux-x86-64.so.2",
}


def fetch() -> int:
    cache: dict[str, bytes] = {}
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, url, inside, dest in WANT:
        if url not in cache:
            print(f"  {url.rsplit('/', 1)[-1]}")
            cache[url] = libc.urllib.request.urlopen(url, timeout=300).read()
        tf = libc.unpack(url, cache[url])
        blob = libc.member(tf, inside)
        if blob is None:
            print(f"  !! {name}: {inside} is not in that archive")
            return 1
        p = OUT / dest
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)
        total += len(blob)
        print(f"    {name:10} {len(blob):>9,} B  {dest}")
    print(f"\n  {total:,} bytes in {len(WANT)} object(s) under {OUT}")
    return 0


def report() -> int:
    if not OUT.exists():
        print("nothing fetched yet; run with --fetch")
        return 1
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    if not files:
        print("nothing fetched yet; run with --fetch")
        return 1

    have: dict[str, Path] = {}
    for p in files:
        have[p.name] = p

    print(f"{'object':34} {'size':>11}  {'soname':26} needs")
    print("-" * 100)
    total = 0
    wanted: set[str] = set()
    for p in files:
        blob = p.read_bytes()
        d = libc.describe(blob)
        total += len(blob)
        wanted |= set(d.get("needed", []))
        needs = ", ".join(d.get("needed", [])) or "-"
        print(f"{p.name:34} {len(blob):>11,}  {str(d.get('soname')):26} {needs}")
        print(f"{'':34} {'sha256':>11}  {hashlib.sha256(blob).hexdigest()[:32]}")

    # The closure, computed rather than assumed. `libc.py` records what
    # assuming cost: a list written from expectation was one file short and
    # failed at the third `open` inside `ld.so`.
    missing = sorted(n for n in wanted if n not in have and n not in FROM_LIBC)
    print("-" * 100)
    print(f"  {total:,} bytes, {len(files)} object(s)")
    print(f"  satisfied by the staged glibc: {', '.join(sorted(wanted & FROM_LIBC)) or 'none'}")
    if missing:
        print(f"  !! MISSING: {', '.join(missing)}")
        print("     the closure is short; add them to WANT before staging")
        return 1
    print("  closure is complete: nothing it names is absent")

    # The arithmetic that decided which Mesa. Printed rather than remembered,
    # because it is the number that moves if `fs.rs` ever grows ranged reads.
    biggest = max(p.stat().st_size for p in files)
    print()
    print(f"  largest single object {biggest:,} B against OPEN_MAX_BYTES of 67,108,864")
    print(f"  whole closure {total:,} B, so every object may be open at once")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true", help="download and unpack")
    ap.add_argument("--report", action="store_true", help="sizes, sonames and the closure")
    a = ap.parse_args()
    if a.fetch:
        rc = fetch()
        if rc:
            return rc
        print()
        return report()
    if a.report:
        return report()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

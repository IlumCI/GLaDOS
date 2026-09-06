#!/usr/bin/env python3
"""A C compiler that targets the guest, in one download and no installation.

**There is no C toolchain on this machine and no usable WSL**, which for a
long time meant this project could only ever run software somebody else had
already compiled for Linux. That was survivable while the guests were busybox
and a hand-assembled fixture, and it stops being survivable at SDL: there is
no prebuilt SDL2 anywhere with a framebuffer video driver, because upstream
does not have one.

`zig cc` is clang with a bundled libc for every target it knows, so one
archive turns this Windows host into a cross-compiler for
`x86_64-linux-gnu` -- the exact triple of the glibc already staged in the
guest. Nothing is installed, nothing goes on `PATH`, and removing it is
deleting a directory.

**Pinned and hash-checked**, for the reason `libc.py` pins its libcs: a
version resolved at fetch time makes every run a different experiment, and a
compiler is the last thing that should move underneath a measurement. The
digest is the one ziglang.org publishes in its own index.

    .\\tools\\venv\\Scripts\\python.exe tools\\zig.py --fetch
    .\\tools\\venv\\Scripts\\python.exe tools\\zig.py --check

`--check` compiles and links a Linux binary and reads it back with
`libc.py`'s ELF reader, which is deliberately not the compiler: a toolchain
that produced the wrong thing would otherwise be found by a guest fault half
an hour later.
"""

import argparse
import hashlib
import io
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import libc  # noqa: E402

VERSION = "0.14.1"
URL = f"https://ziglang.org/download/{VERSION}/zig-x86_64-windows-{VERSION}.zip"
SHA256 = "554f5378228923ffd558eac35e21af020c73789d87afeabf4bfd16f2e6feed2c"
SIZE = 82_229_343

OUT = Path("out/zig")
# The triple the guest actually runs. `gnu` and not `musl`: the staged
# interpreter is glibc's, and a musl binary would name a loader that is not
# there. Both libcs are installed, so this is a choice rather than a limit.
TARGET = "x86_64-linux-gnu"


def exe() -> Path:
    return OUT / f"zig-x86_64-windows-{VERSION}" / "zig.exe"


def fetch() -> int:
    if exe().exists():
        print(f"  already here: {exe()}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"  {URL}")
    blob = urllib.request.urlopen(URL, timeout=1200).read()
    got = hashlib.sha256(blob).hexdigest()
    print(f"  {len(blob):,} bytes, sha256 {got}")
    if len(blob) != SIZE or got != SHA256:
        # Refused rather than reported, since the next thing that happens is
        # running it. A compiler whose bytes are not the ones published is not
        # a compiler this project has any way to reason about.
        print("  !! digest or size does not match what ziglang.org published")
        return 1
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(OUT)
    if not exe().exists():
        print(f"  !! extracted, but {exe()} is not there")
        return 1
    print(f"  {exe()}")
    return 0


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(exe())] + args, capture_output=True, text=True, **kw)


def check() -> int:
    if not exe().exists():
        print("nothing fetched yet; run with --fetch")
        return 1
    v = run(["version"])
    print(f"  zig {v.stdout.strip()}")
    if v.stdout.strip() != VERSION:
        print(f"  !! expected {VERSION}")
        return 1

    src = Path("out/zig/hello.c")
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        "#include <stdio.h>\n"
        "int main(int argc, char **argv) {\n"
        '    printf("compiled here, ran there: %d argument(s)\\n", argc);\n'
        "    return 0;\n"
        "}\n"
    )
    out = Path("out/zig/hello.elf")
    r = run(["cc", "-target", TARGET, "-O2", str(src), "-o", str(out)])
    if r.returncode != 0:
        print("  !! compile failed")
        print(r.stderr[:2000])
        return 1

    # Read it back with the reader that is not the writer, the bargain
    # `tokenizer.py --verify` makes. What matters is the interpreter: a binary
    # naming a loader this machine does not have is one the guest refuses with
    # a message about the namespace rather than about the compiler.
    d = libc.describe(out.read_bytes())
    print(f"  {out}  {d['size']:,} B  {d['kind']}")
    print(f"    interp {d['interp']}")
    print(f"    needs  {', '.join(d['needed']) or '-'}")
    ok = (
        d["kind"] in ("ET_DYN", "ET_EXEC")
        and d["interp"] == "/lib64/ld-linux-x86-64.so.2"
        and "libc.so.6" in d["needed"]
    )
    print("  it names the loader and the libc this guest has"
          if ok else "  !! it does not name what the guest has staged")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        rc = fetch()
        return rc or check()
    if a.check:
        return check()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

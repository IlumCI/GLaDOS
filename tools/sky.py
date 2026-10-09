#!/usr/bin/env python3
"""Build Skywalker's test client and stage it, with its libraries, for QEMU.

`tools/sky/skytest.c` is an ordinary Wayland client against the real
libwayland-client: it binds wl_compositor, wl_shm and xdg_wm_base, makes a
window, and draws frames into a memfd buffer paced by frame callbacks. It is
built here with the host's gcc and run in GLaDOS with the host's own glibc --
the interpreter, libc, libwayland-client and libffi, copied out of the host --
because that is the whole of what a real client needs and none of it is this
tree's to write.

The xdg-shell protocol is fetched at a pinned tag and checked by digest, then
turned into C by `wayland-scanner`, as every client's build does. Nothing
fetched or built goes in the repository: it lands in `out/sky/`.

Usage:
    python3 tools/sky.py build            # out/sky/skytest and the closure
    python3 tools/sky.py stage            # into .qemu/nvme.img, 8.3 names
    python3 tools/sky.py commands         # what to type in GLaDOS

Then, for example:
    mapfile -t C < <(python3 tools/sky.py commands)
    python3 tools/drive.py --no-payload "${C[@]}" "linux run /tmp/wl/skytest 90"
"""

import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "sky"
SRC = ROOT / "tools" / "sky" / "skytest.c"
XDG_URL = ("https://gitlab.freedesktop.org/wayland/wayland-protocols/-/raw/1.45/"
           "stable/xdg-shell/xdg-shell.xml")
XDG_SHA256 = "9d9e27038ca722f159a361293f8c6cd270dab907ab52de5b8ecfbb8d6b9651c2"

# The closure, as (where it is on the host, its 8.3 name on the FAT image,
# where it goes in GLaDOS). The interpreter goes where the binary's PT_INTERP
# says; the libraries under /tmp/wl, which LD_LIBRARY_PATH names, since a guest
# may write only inside /tmp.
CLOSURE = [
    ("/usr/lib64/ld-linux-x86-64.so.2", "LDSO.BIN", "/lib64/ld-linux-x86-64.so.2"),
    ("/usr/lib/libc.so.6", "LIBC.BIN", "/tmp/wl/libc.so.6"),
    ("/usr/lib/libwayland-client.so.0", "LIBWL.BIN", "/tmp/wl/libwayland-client.so.0"),
    ("/usr/lib/libffi.so.8", "LIBFFI.BIN", "/tmp/wl/libffi.so.8"),
]


def build() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    xml = OUT / "xdg-shell.xml"
    if not xml.exists() or hashlib.sha256(xml.read_bytes()).hexdigest() != XDG_SHA256:
        data = urllib.request.urlopen(XDG_URL, timeout=30).read()
        got = hashlib.sha256(data).hexdigest()
        if got != XDG_SHA256:
            print(f"  xdg-shell.xml digest {got}, expected {XDG_SHA256}", file=sys.stderr)
            return 1
        xml.write_bytes(data)
    for kind, name in (("client-header", "xdg-shell.h"), ("private-code", "xdg-shell.c")):
        subprocess.run(["wayland-scanner", kind, str(xml), str(OUT / name)], check=True)
    subprocess.run(["gcc", "-O2", f"-I{OUT}", "-o", str(OUT / "skytest"), str(SRC),
                    str(OUT / "xdg-shell.c"), "-lwayland-client"], check=True)
    # The closure the binary actually needs, checked against what is copied:
    # a library added upstream would otherwise be missing at run time with a
    # loader error as the only clue.
    ldd = subprocess.run(["ldd", str(OUT / "skytest")], capture_output=True, text=True).stdout
    wanted = {Path(line.split()[0]).name for line in ldd.splitlines() if "=>" in line}
    have = {Path(h).name for h, _, _ in CLOSURE}
    missing = wanted - have - {"linux-vdso.so.1"}
    if missing:
        print(f"  the binary needs {sorted(missing)}, which CLOSURE does not copy", file=sys.stderr)
        return 1
    for host, short, _ in CLOSURE:
        shutil.copyfile(Path(host).resolve(), OUT / short)
    shutil.copyfile(OUT / "skytest", OUT / "SKYTEST.ELF")
    print(f"  built {OUT / 'skytest'} and its closure of {len(CLOSURE)}")
    return 0


def stage() -> int:
    files = [str(OUT / "SKYTEST.ELF")] + [str(OUT / s) for _, s, _ in CLOSURE]
    for f in files:
        if not Path(f).exists():
            print(f"  {f} missing: run 'build' first", file=sys.stderr)
            return 1
    subprocess.run([sys.executable, str(ROOT / "tools" / "mkfat.py"),
                    str(ROOT / ".qemu" / "nvme.img")] + files, check=True)
    return 0


def commands() -> list:
    out = [f"fat get /{short} {dest}" for _, short, dest in CLOSURE]
    out.append("fat get /SKYTEST.ELF /tmp/wl/skytest")
    # LD_BIND_NOW for the reason CLAUDE.md gives: glibc's lazy binding reads a
    # GOT word this kernel's loader path leaves zero.
    out += ["linux env LD_LIBRARY_PATH=/tmp/wl", "linux env LD_BIND_NOW=1"]
    return out


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        return build()
    if cmd == "stage":
        return stage()
    if cmd == "commands":
        # One per line, for `mapfile`: each line is one shell command.
        for c in commands():
            print(c)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build the root filesystem Linux guests see, as an EROFS image.

Why this exists
---------------
Every hurdle in getting LibreOffice to run was the machine failing to provide
something a Linux program takes for granted: a data file under `/usr/share`,
a library the program `dlopen`s rather than links, a writable cache, a name
in the case it was written, a symlink. Each was fixed by hand for one program
(`out/office/mkoffice.sh` is the record of that), and the next program would
have needed the same again. A program should find a normal Linux system where
it expects one, so this builds one.

It is a distribution's own packages, unpacked: Arch's, resolved by the host's
`pacman` against a *private* copy of its sync databases, so that what the host
happens to have installed does not hide a dependency. `pacman -Sp` only
prints URLs and needs no root; the packages are fetched into a cache and
unpacked here, symlinks and modes intact, which is what a FAT staging volume
could never carry.

The image is EROFS -- Linux's own read-only filesystem, simple when
uncompressed, with symlinks, modes and owners -- written by `mkfs.erofs`,
which `out/rootfs/tools/` holds out of Arch's package rather than the host
having it installed. Nothing fetched goes in this repository, for the reason no
checkpoint and no game does: it is somebody else's bytes.

Package scripts are not run -- there is no chroot here -- so the steps they
exist for that produce *data* are done directly against the tree: compiled
GSettings schemas and the MIME database. Caches a program can build for itself
at run time (fontconfig) are left to it.

Usage:

    python3 tools/rootfs.py resolve        # what the set expands to
    python3 tools/rootfs.py build          # fetch, unpack, image
    python3 tools/rootfs.py --selftest
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "rootfs"
CACHE = OUT / "cache"
TREE = OUT / "root"
DB = OUT / "db"
IMAGE = ROOT / ".qemu" / "root.erofs"
MKFS = OUT / "tools" / "x" / "usr" / "bin" / "mkfs.erofs"

# What a desktop program expects to find, and the first programs we want.
# Dependencies are pacman's to resolve; this names intent, not closure.
PACKAGES = [
    "filesystem", "glibc", "bash", "coreutils",
    "fontconfig", "ttf-liberation", "ttf-dejavu",
    "gtk3", "adwaita-icon-theme", "adwaita-icon-theme-legacy", "hicolor-icon-theme",
    "shared-mime-info", "xkeyboard-config", "gsettings-desktop-schemas",
    "dbus",
    # The ordinary userland a script or a person expects to find.
    "grep", "sed", "gawk", "findutils", "diffutils", "which", "less", "nano",
    "tar", "gzip", "xz", "file", "procps-ng", "util-linux",
    # The shell sci-fish starts from.
    "fish",
    "libreoffice-fresh",
]

# Packages replaced by an older build, and why. gdk-pixbuf 2.43 and later
# decode every image format in a glycin helper process inside a bwrap
# sandbox, spoken to over D-Bus; 2.42 decodes in-process and keeps the soname.
PINS = {
    "gdk-pixbuf2": "https://archive.archlinux.org/packages/g/gdk-pixbuf2/gdk-pixbuf2-2.42.12-2-x86_64.pkg.tar.zst",
}

# Packages never wanted in a guest root, whatever pulls them in: they exist
# to run under a Linux kernel's init or to sandbox, neither of which is here.
SKIP = {"systemd", "systemd-libs-dummy", "bubblewrap", "glycin", "linux-api-headers"}


MIRROR = "https://geo.mirror.pkgbuild.com"
REPOS = ["core", "extra", "multilib"]


def private_db():
    """Fresh sync databases from the mirror, and an empty local database, so
    pacman resolves the whole closure rather than what the host lacks.

    Fetched rather than copied from the host: the host's are as old as its
    last `-Sy`, the mirror has since rotated packages out, and a set resolved
    against a stale database names files that answer 404. Refreshing the
    host's own would need root."""
    (DB / "local").mkdir(parents=True, exist_ok=True)
    sync = DB / "sync"
    sync.mkdir(parents=True, exist_ok=True)
    for repo in REPOS:
        url = f"{MIRROR}/{repo}/os/x86_64/{repo}.db"
        with urllib.request.urlopen(url) as r, open(sync / f"{repo}.db", "wb") as f:
            shutil.copyfileobj(r, f)


def resolve(pkgs):
    private_db()
    r = subprocess.run(
        ["pacman", "--dbpath", str(DB), "--noconfirm", "-Sp", *pkgs],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"pacman could not resolve the set:\n{r.stderr}")
    urls = [l.strip() for l in r.stdout.splitlines() if l.startswith(("http", "file:"))]
    out = []
    for u in urls:
        name = pkgname(u.rsplit("/", 1)[-1])
        if name in SKIP:
            continue
        out.append(PINS.get(name, u))
    return out


def pkgname(filename):
    """`gdk-pixbuf2-2.44.7-1-x86_64.pkg.tar.zst` -> `gdk-pixbuf2`: the name is
    everything before the last three dash-separated fields."""
    base = filename.split(".pkg.tar")[0]
    return base.rsplit("-", 3)[0]


def fetch(url):
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / url.rsplit("/", 1)[-1]
    if not dst.exists() or dst.stat().st_size == 0:
        tmp = dst.with_suffix(".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(dst)
    return dst


def unpack(pkg):
    # pacman's own metadata files at the top are not part of the system.
    r = subprocess.run(
        ["tar", "-C", str(TREE), "--exclude=.PKGINFO", "--exclude=.BUILDINFO",
         "--exclude=.MTREE", "--exclude=.INSTALL", "--exclude=.CHANGELOG",
         "-xpf", str(pkg)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"could not unpack {pkg.name}: {r.stderr}")


def readable():
    """Give the owner read on files and read and search on directories that
    lack them. A few setuid helpers ship as `---s--x---`, readable only by a
    root this build is not; `mkfs.erofs` must read them to copy them. Adding
    the owner's read bit changes nothing a guest relies on, since nothing
    here runs setuid."""
    for dirpath, dirnames, filenames in os.walk(TREE):
        for d in dirnames:
            p = Path(dirpath) / d
            if not p.is_symlink():
                m = p.stat().st_mode
                if m & 0o500 != 0o500:
                    p.chmod(m | 0o500)
        for f in filenames:
            p = Path(dirpath) / f
            if not p.is_symlink():
                m = p.stat().st_mode
                if not m & 0o400:
                    p.chmod(m | 0o400)


def settings():
    """Defaults the image ships, written where the programs read them.

    GTK 3's icon theme is AdwaitaLegacy, the PNG set: Adwaita itself is SVG
    now, and the gdk-pixbuf pinned above decodes SVG only through a loader
    this image does not carry -- so GTK found no `image-missing` and aborted
    on its first icon."""
    gtk = TREE / "etc/gtk-3.0"
    gtk.mkdir(parents=True, exist_ok=True)
    (gtk / "settings.ini").write_text("[Settings]\ngtk-icon-theme-name=AdwaitaLegacy\n")


def data_hooks():
    """What package scripts would have produced that is data rather than a
    service: compiled schemas and the MIME database."""
    schemas = TREE / "usr/share/glib-2.0/schemas"
    if schemas.is_dir():
        subprocess.run(["glib-compile-schemas", str(schemas)], check=True)
    mime = TREE / "usr/share/mime"
    if mime.is_dir():
        subprocess.run(["update-mime-database", str(mime)], check=True, capture_output=True)


def image():
    if not MKFS.exists():
        sys.exit(f"no mkfs.erofs at {MKFS}; fetch erofs-utils into out/rootfs/tools first")
    IMAGE.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed and with 4 KiB blocks, so the kernel's reader is a reader
    # of plain blocks; `-T0` pins every timestamp so one set of packages makes
    # one image, byte for byte.
    subprocess.run([str(MKFS), "-T0", "-b4096", "--all-root", str(IMAGE), str(TREE)], check=True)


def build(reuse=False):
    if reuse and TREE.exists():
        readable()
        data_hooks()
        image()
        print(f"{IMAGE}  {IMAGE.stat().st_size / 2**20:.0f} MiB")
        return
    urls = resolve(PACKAGES)
    print(f"{len(urls)} package(s)")
    if TREE.exists():
        shutil.rmtree(TREE)
    TREE.mkdir(parents=True)
    for i, u in enumerate(urls, 1):
        p = fetch(u)
        unpack(p)
        if i % 25 == 0:
            print(f"  {i} unpacked")
    readable()
    settings()
    data_hooks()
    image()
    size = IMAGE.stat().st_size
    print(f"{IMAGE}  {size / 2**20:.0f} MiB")


def selftest():
    claims = [
        ("a plain package name is everything before version, release and arch",
         pkgname("gtk3-1:3.24.52-1-x86_64.pkg.tar.zst") == "gtk3"),
        ("and a name with dashes in it keeps them",
         pkgname("gdk-pixbuf2-2.44.7-1-x86_64.pkg.tar.zst") == "gdk-pixbuf2"),
        ("a pin's own filename names the package it replaces",
         all(pkgname(u.rsplit("/", 1)[-1]) == k for k, u in PINS.items())),
        ("nothing pinned is also skipped", not (set(PINS) & SKIP)),
    ]
    bad = 0
    for what, ok in claims:
        print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
        bad += not ok
    sys.exit(1 if bad else 0)


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("cmd", nargs="?", choices=["resolve", "build"])
    a.add_argument("--selftest", action="store_true")
    a.add_argument("--reuse", action="store_true", help="image the tree already unpacked")
    args = a.parse_args()
    if args.selftest:
        selftest()
    elif args.cmd == "resolve":
        urls = resolve(PACKAGES)
        for u in urls:
            print(u)
        print(f"{len(urls)} package(s)", file=sys.stderr)
    elif args.cmd == "build":
        build(args.reuse)
    else:
        a.print_help()


if __name__ == "__main__":
    main()

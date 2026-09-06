#!/usr/bin/env python3
"""Build Xash3D-FWGS for the guest: a Half-Life engine with no OpenGL in it.

**This is the route to Half-Life that does not need a GPU.** Valve's own
`hl_linux` is OpenGL only, so it wants a GL context, which wants EGL or GLX,
which wants DRM or an X server. Xash3D-FWGS is an open reimplementation of the
GoldSrc engine that runs the real game data, and it ships `ref_soft` -- a pure
software renderer with no GL at all. Everything it needs, this machine now
has: SDL2 with a framebuffer driver, evdev, threads, and a filesystem.

Four objects come out, in dependency order, and each is a separate `.so`
because that is how the engine loads them at runtime:

    libfilesystem_stdio.so   paths, packfiles, the search order
    libref_soft.so           the software rasteriser
    libxash.so               the engine
    xash3d                   the launcher, which dlopens the rest

**waf is not used and cannot be.** It is a Python build system that wants to
run compilers to probe for features, and the compiler here is a
cross-compiler for a machine that is not this one -- every probe would answer
about Windows. So the file lists are explicit, the way `sdl.py`'s are, and for
the same reason: a component that quietly picked up `ref/gl` would fail at
link time with a message about OpenGL, which is a long way from the cause.

    .\\tools\\venv\\Scripts\\python.exe tools\\xash.py --build
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import libc  # noqa: E402
import sdl as sdlbuild  # noqa: E402
import zig  # noqa: E402

SRC = Path("out/xash/xash3d-fwgs-master")
OUT = Path("out/guest")

# The roots every component gets. **Deliberately not one shared list**, and
# that is not tidiness -- there are two files called `ref_common.h` in this
# tree, `engine/client`'s and `ref/common`'s, and both use the guard
# `REF_COMMON_H`. A shared list with `engine/client` in it hands the renderer
# the client's header, the guard is satisfied, and `ref_api.h` is never
# reached: every engine type the renderer uses becomes an unknown type name,
# with nothing in the message pointing at an include path.
#
# waf never had the problem because it gives each component only the roots it
# declares a use of, so this mirrors that instead of inventing a superset.
BASE = [
    "public",
    "common",
    "pm_shared",
    "engine",
    "filesystem",
    # `build.h` and `buildenums.h` live here, and finding that took a while:
    # they are in the `library-suffix` submodule, and a GitHub tarball
    # excludes submodules. The engine's whole platform and architecture
    # detection is in that one header, so without it every file fails on the
    # same missing include and none of them says where it went.
    "3rdparty/library_suffix/include",
]

# Defines every object shares. The platform is detected from compiler macros
# in `public/build.h`, so nothing here says "linux" -- saying it would be a
# second source of truth for a question the compiler already answers.
COMMON = [
    "-DXASH_SDL=2",
    # waf fills these from `git describe` and there is no git here. Named
    # after where the source came from rather than left empty, so a crash
    # report out of the guest still says which tree it was built from.
    '-DXASH_BUILD_COMMIT="glados-master"',
    '-DXASH_BUILD_BRANCH="master"',
    '-DXASH_BUILD_COMMIT_DATE="0"',
    "-D_GNU_SOURCE",
    # waf probes for these by compiling; a cross-compiler would answer about
    # the *host*, so they are asserted from what glibc is known to have.
    # `crtlib.h` uses each to decide between an inline wrapper and an extern
    # declaration, so a missing one is a symbol nothing defines -- which
    # `--no-undefined` turns into a build error instead of a guest that dies
    # at its first lookup.
    "-DHAVE_STRCASECMP=1",
    "-DHAVE_STRCHRNUL=1",
    "-DHAVE_STRCASESTR=1",
    "-DHAVE_TGMATH_H=1",
    "-DNDEBUG",
]

COMPONENTS = {
    # The smallest and the one nothing else depends on, so it is first: if the
    # include roots or the defines are wrong, they are wrong here in fourteen
    # files rather than in two hundred.
    "libfilesystem_stdio.so": {
        "dirs": ["filesystem", "public", "3rdparty/library_suffix/src"],
        "includes": [],
        "defines": ["-DXASH_FILESYSTEM_STDIO"],
    },
    # `ref/common` first, and `engine/client` absent: see the note on BASE.
    "libref_soft.so": {
        "dirs": ["ref/soft", "ref/common", "public", "3rdparty/library_suffix/src"],
        "includes": ["ref/common", "ref/soft", "engine/common"],
        "defines": ["-DREF_DLL"],
    },
    "libxash.so": {
        # The engine is deeper than one level. `engine/client` alone leaves
        # out the input, sound, parse, vgui and dll_int layers, which surface
        # as a hundred and seventy undefined symbols with nothing saying they
        # are directories rather than a missing library.
        "dirs": [
            "engine/client",
            "engine/client/avi",
            "engine/client/dll_int",
            "engine/client/input",
            "engine/client/parse",
            "engine/client/sound",
            "engine/client/soundlib",
            "engine/client/soundlib/libmpg",
            "engine/client/vgui",
            "engine/common",
            "engine/common/http",
            "engine/common/imagelib",
            "engine/common/soundlib",
            "engine/server",
            # One platform backend and no more. The other thirteen are for
            # machines this is not, and a glob over `platform/` would compile
            # every one of them.
            "engine/platform/sdl2",
            "engine/platform/linux",
            "engine/platform/posix",
            "engine/platform/misc",
            "public",
            "3rdparty/library_suffix/src",
            "3rdparty/MultiEmulator/src",
            "3rdparty/bzip2/bzip2",
        ],
        "includes": ["engine/client", "engine/client/vgui", "engine/common",
                     "engine/common/imagelib", "engine/common/soundlib",
                     "engine/client/sound", "engine/client/soundlib",
                     "engine/server", "engine/platform",
                     # Another submodule a tarball leaves empty. The client
                     # includes its header unconditionally, so it is fetched
                     # rather than defined away.
                     "3rdparty/MultiEmulator/include",
                     # bzip2, for the compressed fragments a server sends.
                     # Another empty submodule, and it is fetched rather than
                     # defined away because `net_chan.c` includes it flat.
                     "3rdparty/bzip2/bzip2"],
        # `kmalloc.c` and `sbrk.c` are the custom-swap allocator, and they
        # include a header that only exists when that option is on. waf builds
        # them only in that configuration; this leaves them out for the same
        # reason rather than defining the option to satisfy an include.
        # bzip2's directory holds its command-line tools beside its
        # library. waf names the seven library files explicitly; this names
        # the rest, which is the same list said the other way round.
        "skip": {"kmalloc.c", "sbrk.c",
                 "bzip2.c", "bzip2recover.c", "dlltest.c", "mk251.c",
                 "spewG.c", "unzcrash.c",
                 },
        "defines": ["-DENGINE_DLL"],
        # The platform layer calls SDL directly, so it links against it. A
        # `-shared` link leaves undefined symbols alone by default, which is
        # how the first build came out naming no SDL at all and would have
        # died at the first symbol lookup in the guest -- the same edge
        # dynapi had.
        "libs": ["out/guest/libSDL2-2.0.so.0"],
    },
}


def sources(dirs: list[str], skip: set = frozenset()) -> list[Path]:
    out = []
    for d in dirs:
        p = SRC / d
        if not p.exists():
            continue
        out += [f for f in sorted(p.glob("*.c")) if f.name not in skip]
    return out


def compile_one(name: str, spec: dict) -> int:
    files = sources(spec["dirs"], spec.get("skip", frozenset()))
    if not files:
        print(f"  !! {name}: no sources under {spec['dirs']}")
        return 1
    so = OUT / name
    args = ["cc", "-target", zig.TARGET, "-O2", "-fPIC", "-shared", "-o", str(so)]
    args += COMMON + spec["defines"]
    # The component's own roots first, so a name it owns wins over the same
    # name somewhere else in the tree.
    for i in spec.get("includes", []) + BASE:
        args.append(f"-I{SRC / i}")
    args.append(f"-I{sdlbuild.SRC / 'include'}")
    # Xash is old-fashioned C and warns a great deal; the warnings are not the
    # subject and drowning the errors in them makes the first real one
    # unfindable.
    args += ["-w", "-lm", "-lpthread", "-ldl", "-lrt"]
    args += [str(f) for f in files]
    args += spec.get("libs", [])
    # Nothing may be left dangling: a shared object that links with holes in
    # it is one that fails at the first lookup inside the guest, a long way
    # from the build that made it.
    args += ["-Wl,--no-undefined"]

    r = subprocess.run([str(zig.exe())] + args, capture_output=True, text=True)
    if r.returncode != 0:
        seen, uniq = set(), []
        for chunk in re.split(r"(?=error:)", r.stderr):
            c = chunk.strip()
            if not c.startswith("error:"):
                continue
            key = c.split("error:", 1)[1].strip().splitlines()[0][:100]
            if key not in seen:
                seen.add(key)
                uniq.append(key)
        print(f"  !! {name}: {len(files)} file(s), {len(uniq)} distinct error(s):")
        for l in uniq[:12]:
            print("    " + l)
        return 1
    d = libc.describe(so.read_bytes())
    print(f"  {name:26} {len(files):>3} file(s)  {d['size']:>10,} B  "
          f"needs {', '.join(d['needed'])}")
    return 0


def build() -> int:
    if not SRC.exists():
        print(f"  !! {SRC} is not there; the source has to be fetched first")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, spec in COMPONENTS.items():
        if compile_one(name, spec):
            bad += 1
            # Stop at the first, because the include roots and defines are
            # shared: a failure in one is nearly always a failure in all, and
            # three copies of it says nothing the first did not.
            break
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.build:
        return build()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

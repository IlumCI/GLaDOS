#!/usr/bin/env python3
"""Build SDL2 for the guest, with a video driver that writes to `/dev/fb0`.

**SDL2 has no framebuffer backend**, and that is the one thing standing
between this machine and a large amount of existing software. Its video
drivers are X11, Wayland, KMSDRM, offscreen and dummy; fbdev was an SDL 1.2
thing and did not survive the rewrite. So `/dev/fb0` being perfect does not
put an unmodified SDL2 program on screen -- the program has to be given a
driver that knows about it.

`tools/guest/sdl/` is that driver, and it is a small delta on SDL's own dummy:
`SDL_nullframebuffer.c` already allocates a surface the size of the window and
hands back its pixels, and its `UpdateWindowFramebuffer` does nothing. Ours
writes. Window creation, the surface, the software renderer and the whole 2D
stack above are SDL's and are untouched.

**No cmake and no autotools**, because neither is on this host. SDL already
ships hand-written configs for Linux targets that never had a build system
either -- `SDL_config_pandora.h` is one -- so `SDL_config_glados.h` is that
arrangement with a different set of drivers. The file list below is what a
Linux build with these drivers needs, and it is explicit rather than a
recursive glob for the reason `payload.py` verifies digests: a build that
silently picked up `src/video/x11` would fail at link time with a message
about X, which is a long way from the cause.

    .\\tools\\venv\\Scripts\\python.exe tools\\sdl.py --build
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import libc  # noqa: E402
import zig  # noqa: E402

VERSION = "2.30.12"
SRC = Path("out/sdl") / f"SDL2-{VERSION}"
MINE = Path("tools/guest/sdl")
OUT = Path("out/guest")

# Every directory whose `*.c` goes in. Named rather than walked, so a driver
# nobody asked for cannot arrive by being in the tree.
DIRS = [
    "src",
    "src/atomic",
    "src/audio",
    "src/audio/dummy",
    "src/cpuinfo",
    "src/dynapi",
    "src/events",
    "src/file",
    "src/filesystem/unix",
    "src/haptic",
    "src/haptic/dummy",
    "src/joystick",
    "src/joystick/dummy",
    "src/joystick/virtual",
    "src/libm",
    "src/loadso/dlopen",
    "src/locale",
    "src/locale/dummy",
    "src/misc",
    "src/misc/dummy",
    "src/power",
    "src/render",
    "src/render/software",
    "src/sensor",
    "src/sensor/dummy",
    "src/stdlib",
    "src/thread",
    "src/thread/pthread",
    "src/timer",
    "src/timer/unix",
    "src/video",
    "src/video/dummy",
    "src/video/glados",
    "src/video/yuv2rgb",
    # evdev: SDL's own reader of `/dev/input/event*`, unmodified.
    "src/core/linux",
]

# Files inside those directories that must not be compiled, each for a reason.
SKIP = {
    # Needs dbus, ibus, udev or libusb, none of which are here. SDL guards
    # most of this by ifdef, and these are the ones that do not build clean
    # without their headers.
    "SDL_dbus.c",
    "SDL_ibus.c",
    "SDL_fcitx.c",
    "SDL_udev.c",
    "SDL_ime.c",
    "SDL_sandbox.c",
    "SDL_system_theme.c",
}


def vendor() -> int:
    """Put our driver where SDL expects to find one."""
    if not SRC.exists():
        print(f"  !! {SRC} is not there; the source has to be fetched first")
        return 1
    dst = SRC / "src" / "video" / "glados"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("SDL_gladosvideo.c", "SDL_gladosvideo.h",
                 "SDL_gladosframebuffer.c", "SDL_gladosframebuffer_c.h"):
        shutil.copy(MINE / name, dst / name)
    shutil.copy(MINE / "SDL_config_glados.h", SRC / "include" / "SDL_config_glados.h")

    # Two edits to SDL's own source, and both are the ones upstream makes for
    # every driver it has: name the config, and put the bootstrap in the list
    # `SDL_VideoInit` walks. A driver that is compiled and not listed is dead
    # code that reports itself missing.
    cfg = SRC / "include" / "SDL_config.h"
    text = cfg.read_text(encoding="utf-8", errors="replace")
    hook = '#include "SDL_config_glados.h"'
    if hook not in text:
        # Made the first arm of the platform chain, which leaves the rest
        # unreachable -- exactly what cmake's generated config does.
        marker = "#if defined(__WIN32__)"
        if marker not in text:
            print("  !! SDL_config.h does not look the way this expects")
            return 1
        text = text.replace(
            marker, "#if 1\n" + hook + "\n#elif defined(__WIN32__)", 1)
        cfg.write_text(text, encoding="utf-8")

    vid = SRC / "src" / "video" / "SDL_video.c"
    text = vid.read_text(encoding="utf-8", errors="replace")
    if "GLADOS_bootstrap" not in text:
        a = "#ifdef SDL_VIDEO_DRIVER_DUMMY\n    &DUMMY_bootstrap,"
        b = ("#ifdef SDL_VIDEO_DRIVER_GLADOS\n    &GLADOS_bootstrap,\n#endif\n"
             "#ifdef SDL_VIDEO_DRIVER_DUMMY\n    &DUMMY_bootstrap,")
        if a not in text:
            print("  !! the bootstrap list does not look the way this expects")
            return 1
        text = text.replace(a, b, 1)
        vid.write_text(text, encoding="utf-8")

    # The declaration lives in `SDL_sysvideo.h` with every other driver's,
    # rather than beside the list that uses them. Getting that wrong is two
    # errors that both say the same thing: the symbol is undeclared, and the
    # array it is in therefore has no size.
    hdr = SRC / "src" / "video" / "SDL_sysvideo.h"
    text = hdr.read_text(encoding="utf-8", errors="replace")
    if "GLADOS_bootstrap" not in text:
        c = "extern VideoBootStrap DUMMY_bootstrap;"
        if c not in text:
            print("  !! SDL_sysvideo.h does not declare the bootstraps here")
            return 1
        text = text.replace(c, "extern VideoBootStrap GLADOS_bootstrap;\n" + c, 1)
        hdr.write_text(text, encoding="utf-8")
    # **dynapi off, by editing the header, because that is what it insists
    # on.** It exists so a system SDL can override a bundled one at load time,
    # routing every public call through a table of `_REAL` symbols; there is
    # no system SDL here to defer to. The indirection has a sharp edge: a
    # source file left out of the build still gets its `_REAL` entry
    # generated, so the library links and then dies at the first symbol
    # lookup inside the guest, which is a long way from "you skipped a file".
    # `-DSDL_DYNAMIC_API=0` is refused on purpose -- "Nope, you have to edit
    # this file to force this off" -- so this edits it.
    dyn = SRC / "src" / "dynapi" / "SDL_dynapi.h"
    text = dyn.read_text(encoding="utf-8", errors="replace")
    if "GLADOS_NO_DYNAPI" not in text:
        c = "#ifdef SDL_DYNAMIC_API /* Tried to force it on the command line? */"
        if c not in text:
            print("  !! SDL_dynapi.h does not guard the way this expects")
            return 1
        text = text.replace(
            c, "#define GLADOS_NO_DYNAPI 1\n#define SDL_DYNAMIC_API 0\n#if 0\n" + c, 1)
        text = text.replace(
            "#error Nope, you have to edit this file to force this off.\n#endif",
            "#error unreachable\n#endif\n#endif", 1)
        dyn.write_text(text, encoding="utf-8")

    print(f"  vendored into {dst}")
    return 0


def sources() -> list[Path]:
    out = []
    for d in DIRS:
        p = SRC / d
        if not p.exists():
            continue
        for f in sorted(p.glob("*.c")):
            if f.name in SKIP:
                continue
            out.append(f)
    return out


def build() -> int:
    if vendor():
        return 1
    files = sources()
    OUT.mkdir(parents=True, exist_ok=True)
    so = OUT / "libSDL2-2.0.so.0"
    print(f"  {len(files)} source file(s)")
    args = [
        "cc", "-target", zig.TARGET, "-O2", "-fPIC", "-shared",
        "-o", str(so),
        f"-I{SRC / 'include'}",
        # SDL builds its shared library with a version script upstream; the
        # symbols it exports here are simply everything, which costs a larger
        # dynamic table and nothing else.
        "-Wl,-soname,libSDL2-2.0.so.0",
        "-lm", "-lpthread", "-ldl",
    ] + [str(f) for f in files]
    r = subprocess.run([str(zig.exe())] + args, capture_output=True, text=True)
    if r.returncode != 0:
        # Only the first few, because one missing define produces the same
        # error in forty files and the first is the informative one.
        # zig compiles in parallel and the messages interleave, so the raw
        # stderr arrives shredded across file boundaries. Split on the marker
        # and keep one of each: forty files failing on one missing define
        # produce forty copies of the same sentence.
        seen, uniq = set(), []
        for chunk in re.split(r"(?=error:)", r.stderr):
            c = chunk.strip()
            if not c.startswith("error:"):
                continue
            key = c.split("error:", 1)[1].strip()[:90]
            if key not in seen:
                seen.add(key)
                uniq.append(key)
        print(f"  !! {len(uniq)} distinct error(s):")
        for l in uniq[:14]:
            print("    " + l)
        return 1
    d = libc.describe(so.read_bytes())
    print(f"  {so}  {d['size']:,} B  {d['kind']}  soname {d['soname']}")
    print(f"    needs {', '.join(d['needed'])}")
    return 0


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

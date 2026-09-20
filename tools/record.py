#!/usr/bin/env python3
"""Film GLaDOS running, at a framerate somebody would watch.

    python tools/record.py out/demo.mp4 [--scene boot|desktop|doom|all] [--dry]

### Why this does not go through QEMU

`drive.py --screenshot` is the only capture this harness had, and it is a
monitor round-trip: `screendump` writes a 3 MB PPM with a two-second settle in
front of it and a one-and-a-half-second wait behind. That is **0.28 frames a
second**. It is the right instrument for one screenshot and hopeless for video,
and tuning does not fix it, because the cost is the round trip rather than the
encoding.

So the guest gets an ordinary SDL window (`drive.py --window`) and ffmpeg films
the host desktop through `gdigrab`. Thirty frames a second, h264, no round trip
at all.

### The mode is chosen so the capture is not resampled

A guest larger than the host screen has its window clipped, and the part hanging
off the edge records as **black** -- which reads as a kernel that painted
nothing rather than as a window that did not fit. This host is 1536x864, so the
guest runs at 1280x720: it fits at 1:1 with room for the title bar, so every
guest pixel is one video pixel, and it is already a standard video mode.

Text is what this machine mostly draws, and resampled text is the difference
between a demo somebody watches and one they scroll past.

### The scene table is the script

A demo that races is a demo nobody can read. Each row is a command, how long to
linger on it afterwards, and the caption to burn in -- so the pacing is
declared in one place rather than emerging from how fast the serial line
happens to drain. A row with no command is a beat: it films whatever is on
screen without touching it, which is how the boot log gets watched.
"""

import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "tools/venv/Scripts/python.exe"

# The window `drive.py --window` produces. QEMU titles it from `-name`, and
# `gdigrab`'s `title=` is an exact FindWindow match, so this string is an
# interface between two files and is checked rather than assumed.
WINDOW = "QEMU (GLaDOS-0)"

RES = "1280x720"


def ffmpeg():
    """Find an ffmpeg that can capture, encode and draw text.

    Not installed by this repo and not vendored. Windows machines tend to
    already carry several -- ShareX, Krita and OBS each bundle one -- and the
    ShareX build is a full n8 with `gdigrab`, `libx264`, `drawtext` and
    `xfade`, which is the whole of what this needs.
    """
    for p in (
        Path("C:/Program Files/ShareX/ffmpeg.exe"),
        Path("C:/Program Files/Krita (x64)/bin/ffmpeg.exe"),
        Path("C:/ProgramData/chocolatey/bin/ffmpeg.exe"),
    ):
        if p.exists():
            return p
    from shutil import which
    found = which("ffmpeg")
    if found:
        return Path(found)
    raise SystemExit(
        "record.py: no ffmpeg.\n"
        "  It is not vendored here on purpose. Install one, or point this at\n"
        "  the copy ShareX/OBS/Krita already shipped."
    )


# (command, dwell seconds, caption). An empty command is a beat: film what is
# already there. `None` as a caption draws nothing.
SCENES = {
    # The boot log is the best thing this machine has to show and it needs no
    # driving at all -- 29 selftest sections scrolling past is the argument.
    "boot": [
        ("", 30.0, "GLaDOS: a language model living in ring 0"),
        ("", 18.0, "29 selftest sections, every boot"),
    ],
    # `paint`, `mines` and `oracle` are top-level verbs rather than `win open
    # X`; `open_app` hands focus back to the terminal on purpose, which is what
    # lets one command follow another here at all.
    "desktop": [
        ("paint", 4.0, "Paintbrush"),
        ("mines", 4.5, "Minesweeper, in the kernel"),
        ("oracle", 6.0, "The Oracle: a fitted model of the machine's own future"),
        ("font", 5.0, "325 glyphs, composed rather than drawn"),
    ],
    # What the machine says about itself, which is the part that is actually
    # novel. Slower dwells: this is text somebody has to read.
    "mind": [
        ("outcome", 7.0, "How far apart two applets are, measured by what they printed"),
        ("bench report", 6.0, "Fourteen rails, one machine-readable block"),
        ("diag all", 25.0, "67 suites, 0 failed"),
    ],
    "doom": [
        ("doom view 0 60000", 20.0, "DOOM, ported not emulated"),
    ],
}

ORDER = ["boot", "desktop", "mind"]


def build(out, scene_names, dry):
    ff = ffmpeg()
    scenes = []
    for n in scene_names:
        if n not in SCENES:
            raise SystemExit(f"record.py: no scene {n!r}; have {sorted(SCENES)}")
        scenes += SCENES[n]

    cmds, marks, t = [], [], 0.0
    for cmd, dwell, caption in scenes:
        # The dwell has to reach the *guest*, not just the caption track.
        # drive.py sends the next line the instant a prompt returns, so without
        # a beat behind each command the desktop scenes flash past in under a
        # second and the captions describe windows that are already gone.
        if cmd:
            cmds += [cmd]
        cmds += [f"@wait {dwell:.1f}"]
        marks.append((t, t + dwell, caption))
        t += dwell

    print(f"[record] {len(scenes)} scene(s), about {t:.0f}s of guest time")
    for a, b, c in marks:
        print(f"  {a:6.1f} - {b:6.1f}  {c or ''}")
    if dry:
        return

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + "-raw.mp4")

    drive = [
        str(PY), str(ROOT / "tools/drive.py"),
        "--window", "--res", RES,
        "--qemu-extra", "-accel whpx -cpu max",
        "--timeout", str(int(t) + 400),
        "initiative off", "agent stop",
        *cmds,
    ]
    print("[record] " + " ".join(shlex.quote(x) for x in drive))
    guest = subprocess.Popen(drive, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)

    # Wait for the window rather than sleeping a guessed amount. Boot under
    # WHPX is around 150 s with a real checkpoint and much less with the small
    # one, and a fixed sleep is wrong in both directions.
    if not wait_for_window():
        guest.kill()
        raise SystemExit("record.py: the guest never opened a window")

    box = client_crop()
    if box is None:
        guest.kill()
        raise SystemExit("record.py: the window vanished before filming started")
    w, h, dx, dy = box
    print(f"[record] window is up; filming its client area {w}x{h} at +{dx}+{dy}")
    cap = subprocess.Popen(
        [str(ff), "-hide_banner", "-loglevel", "error",
         # The window and nothing else -- not `desktop`, so nothing on the host
         # can wander into the shot.
         "-f", "gdigrab", "-framerate", "30", "-i", f"title={WINDOW}",
         # ...and inside that window, the guest and not its title bar.
         "-vf", f"crop={w}:{h}:{dx}:{dy}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
         "-pix_fmt", "yuv420p", "-y", str(raw)],
        stdin=subprocess.PIPE)

    try:
        guest.wait(timeout=int(t) + 500)
    except subprocess.TimeoutExpired:
        guest.kill()
    finally:
        # `q` on stdin is how ffmpeg is asked to finish the file. Killing it
        # leaves an mp4 with no moov atom, which nothing will play.
        try:
            cap.communicate(input=b"q", timeout=20)
        except Exception:
            cap.kill()

    if not raw.exists() or raw.stat().st_size == 0:
        raise SystemExit("record.py: capture produced nothing")
    print(f"[record] raw {raw} ({raw.stat().st_size / 1e6:.1f} MB)")
    return raw


def wait_for_window(limit=420.0):
    """Block until QEMU's window exists, or give up."""
    import ctypes
    user32 = ctypes.windll.user32
    t0 = time.time()
    while time.time() - t0 < limit:
        if user32.FindWindowW(None, WINDOW):
            # A window exists before it has painted anything. One second is
            # enough for the firmware's first frame, and filming a moment of
            # black is better than missing the splash.
            time.sleep(1.0)
            return True
        time.sleep(1.0)
    return False


def client_crop():
    """Where the guest's pixels are inside QEMU's window.

    `gdigrab title=` captures the whole **window**, which is the guest plus a
    title bar and a border. Those are not the demo, so they are cropped off --
    and the offsets are asked of Windows rather than guessed, because a title
    bar is not a fixed height once display scaling is involved.

    It also answers whether the capture is honest. The client area must be
    exactly the mode the guest was given; anything else means the window was
    clipped by the screen or scaled by the compositor, and a resampled
    framebuffer is the one thing this whole approach exists to avoid. So a
    mismatch is reported rather than cropped around.
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, WINDOW)
    if not hwnd:
        return None

    win, cli, org = wintypes.RECT(), wintypes.RECT(), wintypes.POINT(0, 0)
    user32.GetWindowRect(hwnd, ctypes.byref(win))
    user32.GetClientRect(hwnd, ctypes.byref(cli))
    user32.ClientToScreen(hwnd, ctypes.byref(org))

    w, h = cli.right - cli.left, cli.bottom - cli.top
    dx, dy = org.x - win.left, org.y - win.top
    want_w, want_h = (int(v) for v in RES.split("x"))
    if (w, h) != (want_w, want_h):
        print(
            f"[record] WARNING: client area is {w}x{h} where the guest was "
            f"given {want_w}x{want_h}.\n"
            f"           The window is being clipped or scaled, so the capture "
            f"is not pixel-perfect.",
            file=sys.stderr,
        )
    # Even values, or yuv420p refuses.
    return (w - w % 2, h - h % 2, dx, dy)


def main():
    argv = sys.argv[1:]
    dry = "--dry" in argv
    if dry:
        argv.remove("--dry")
    names = ORDER
    if "--scene" in argv:
        i = argv.index("--scene")
        v = argv[i + 1]
        names = list(SCENES) if v == "all" else v.split(",")
        del argv[i:i + 2]
    out = Path(argv[0]) if argv else ROOT / "out/demo.mp4"
    build(out, names, dry)


if __name__ == "__main__":
    main()

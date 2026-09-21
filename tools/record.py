#!/usr/bin/env python3
"""Film GLaDOS running, at a framerate somebody would watch.

    python tools/record.py out/demo.mp4 [--scene settle|desktop|screen|mind|all] [--dry]

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
guest runs at 1280x800: it fits at 1:1 with room for the title bar, so every
guest pixel is one video pixel.

**1280x800 and not 1280x720**, which was tried first and silently produced a
640x480 capture. 720 is not a mode this VGA device offers, so the guest fell
back to the smallest one it had -- and the recording looked like a working
demo of a tiny screen rather than a misconfiguration. `client_crop` asserts the
client area equals the mode that was asked for, which is what caught it.

Text is what this machine mostly draws, and resampled text is the difference
between a demo somebody watches and one they scroll past.

### The scene table is the script, and the captions are timed by the clock

Each row is a command, how long to linger afterwards, and the caption to burn
in. A row with no command is a beat: it films whatever is on screen without
touching it, which is how the boot log gets watched.

**The caption timings are measured rather than assumed, and that was a real
defect.** The first version computed each caption's span by adding up dwells,
which silently assumes a command is instant. `diag all` has a 25 s dwell and
runs for **88 s**, so every caption after it was out by over a minute -- and
none of it showed, because the ffmpeg filter chain was `crop` alone and the
computed marks were printed and then thrown away. A script that is not applied
is a document, not a script.

So the guest's own output is timestamped as it arrives, a scene ends when its
beat ends, and the captions are burned in a second pass over the raw capture.
A dry run can only estimate, and says so.

### The stdout pump is not tidiness

`drive.py` is launched with a pipe and boot alone writes far more than the
64 KB Windows gives one. Nothing read it until the process exited, so the guest
blocked on a full pipe partway through boot and the recording filmed a machine
that had stopped for reasons entirely of this script's making. It is drained by
a thread from the moment it starts.
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# **Both ends of the pipe, not just the one that bit first.** Decoding the
# guest was fixed with `encoding="utf-8"` on the Popen; echoing it then failed
# the same way in the other direction, because this process's own stdout is
# cp1252 and the guest prints a Greek alphabet in its glyph sheet. One codec
# per direction, and a stray byte stops being able to take a capture down.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "tools/venv/Scripts/python.exe"

# The window `drive.py --window` produces. QEMU titles it from `-name`, and
# `gdigrab`'s `title=` is an exact FindWindow match, so this string is an
# interface between two files and is checked rather than assumed.
WINDOW = "QEMU (GLaDOS-0)"

RES = "1280x800"

# Burned in at the foot of the frame. Segoe is on every Windows machine and is
# not the guest's own font on purpose: a caption that looked like something the
# kernel had printed would be a caption nobody could trust.
CAPTION_PT = 30
CAPTION_PAD = 44

# **Everything before the first prompt is boot, and no beat can cover it.**
# `drive.py` sends nothing until the shell exists, so a scene row cannot
# describe the boot log -- the first version had two that tried, and their
# beats fired *after* boot, captioning a settled desktop with "29 selftest
# sections". The pre-prompt footage gets its own caption and its own clock.
BOOT_CAPTION = "Boot: 29 selftest sections, every one of them run"

# And it gets sped up, because it is around six minutes under WHPX with a real
# checkpoint and almost all of it is text scrolling at a readable-but-slow
# rate. Eight makes it forty-odd seconds, which is long enough to see what it
# is and short enough that nobody scrolls past. Everything after the prompt
# plays at 1x, because that half is the demo.
BOOT_SPEED = 8.0


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


def caption_font(into: Path):
    """Put a font beside the filter script, and answer its bare name.

    **Copied rather than referenced, because of one character.** A filter graph
    separates options with colons, so an absolute Windows path carries one in
    `C:` and ffmpeg parses the drive letter as the end of the value --
    `No option name near '/Windows/Fonts/segoeui.ttf'`. The documented escape
    does not survive a `-filter_complex_script` file. A relative name has no
    colon in it, which is the whole of the fix.
    """
    for src in (
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if src.exists():
            dst = into / "caption.ttf"
            shutil.copyfile(src, dst)
            return dst.name
    raise SystemExit("record.py: no caption font found under C:/Windows/Fonts")


# (command, dwell seconds, caption). An empty command is a beat: film what is
# already there. `None` as a caption draws nothing.
SCENES = {
    # The desktop as boot leaves it. These are beats rather than commands --
    # they film what is already there -- and they are deliberately *not* about
    # the boot log, which has finished by the time any beat can fire.
    "settle": [
        ("", 6.0, "GLaDOS: a language model living in ring 0"),
        ("", 5.0, "A window manager, in the kernel, with no processes under it"),
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
    # One task owns the screen. This is the part of the machine that changed
    # most recently and there was nothing filming it.
    "screen": [
        ("render reset", 1.0, "One task owns the screen, and it says so"),
        ("render probe 6", 9.0,
         "A busy command that never yields: the desktop keeps painting"),
        ("render", 7.0, "composers 1 -- one task has ever composed a frame"),
    ],
    # What the machine says about itself, which is the part that is actually
    # novel. Slower dwells: this is text somebody has to read.
    "mind": [
        ("outcome", 7.0, "How far apart two applets are, measured by what they printed"),
        ("bench report", 6.0, "Fourteen rails, one machine-readable block"),
        # The count is read off `diag.rs`'s `SLOTS`, which is asserted against
        # `SUITES.len()` at compile time. It said 67 here for a while after a
        # suite was added, which is the ordinary way a number in a caption goes
        # stale -- there is nothing to check a string in a demo script.
        ("diag all", 25.0, "67 suites, 0 failed"),
    ],
}

SCENES["net"] = [
    # Raw footage of the network panel animating. No captions: this is cut into
    # a gif rather than watched, and the panel only animates while its window
    # has focus, which is what the alt-tab is for -- `open_app` hands focus back
    # to the terminal on purpose.
    # `oracle net` captures on open, so nothing has to be typed at the window.
    # Driving a key into it is not possible anyway: the shell re-focuses the
    # terminal after every command, so an injected key never reaches the app
    # that was just raised.
    ("oracle net", 15.0, None),
]

ORDER = ["settle", "desktop", "screen", "mind"]


def scenes_for(names):
    scenes = []
    for n in names:
        if n not in SCENES:
            raise SystemExit(f"record.py: no scene {n!r}; have {sorted(SCENES)}")
        scenes += SCENES[n]
    return scenes


def build(out, scene_names, dry):
    ff = ffmpeg()
    scenes = scenes_for(scene_names)

    cmds = []
    for cmd, dwell, _ in scenes:
        # The dwell has to reach the *guest*, not just the caption track.
        # drive.py sends the next line the instant a prompt returns, so without
        # a beat behind each command the desktop scenes flash past in under a
        # second and the captions describe windows that are already gone.
        if cmd:
            cmds += [cmd]
        cmds += [f"@wait {dwell:.1f}"]

    dwell_total = sum(d for _, d, _ in scenes)
    print(f"[record] {len(scenes)} scene(s), {dwell_total:.0f}s of dwell plus "
          f"however long the commands take")
    for cmd, dwell, cap in scenes:
        print(f"  {dwell:5.1f}s  {cmd or '(beat)':<18}  {cap or ''}")
    if dry:
        print("[record] --dry: caption spans are measured during a real run, "
              "not estimated here")
        return

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + "-raw.mp4")

    drive = [
        str(PY), str(ROOT / "tools/drive.py"),
        "--window", "--res", RES,
        "--qemu-extra", "-accel whpx -cpu max",
        "--timeout", str(int(dwell_total) + 900),
        "initiative off", "agent stop",
        *cmds,
    ]
    print("[record] " + " ".join(shlex.quote(x) for x in drive))
    # **`encoding` and `errors` are load-bearing, not tidiness.** `text=True`
    # alone decodes with the locale codec, cp1252 here, and this guest prints
    # bytes it has no mapping for -- `font` draws a 325-glyph coverage sheet of
    # box drawing and accents, which is exactly where the first capture died.
    guest = subprocess.Popen(drive, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                             encoding="utf-8", errors="replace")

    # Drained from the moment it starts. See the module docstring: nothing read
    # this until the process exited, and boot writes far more than the pipe
    # holds, so the guest blocked partway through booting.
    log = []
    lock = threading.Lock()

    def pump(stream):
        # **Nothing this thread hits may stop it draining.** A decoder that
        # raised took a whole capture down: the pump died, the pipe filled, the
        # guest blocked, and the window closed under the camera thirteen
        # minutes in. The encoding is fixed below, but a bare drain behind it
        # is cheap against eleven minutes of guest time.
        try:
            for line in stream:
                with lock:
                    log.append((time.time(), line.rstrip("\n")))
                sys.stdout.write(line)
                sys.stdout.flush()
        except Exception as e:
            print(f"[record] the log pump stopped: {e!r}", file=sys.stderr)
            try:
                for _ in stream:
                    pass
            except Exception:
                pass

    threading.Thread(target=pump, args=(guest.stdout,), daemon=True).start()

    # Wait for the window rather than sleeping a guessed amount. Boot under
    # WHPX is around 370 s with a real checkpoint and much less with the small
    # one, and a fixed sleep is wrong in both directions.
    box = wait_for_mode()
    if box is None:
        guest.kill()
        raise SystemExit("record.py: the guest never opened a window")
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
    # The capture's own zero. Every caption span below is measured against it,
    # so it is taken as close to the first frame as this can manage.
    t0 = time.time()

    try:
        guest.wait(timeout=int(dwell_total) + 1000)
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

    with lock:
        lines = list(log)
    marks, boot_end = marks_from(lines, scenes, t0)
    if not marks:
        print("[record] no beats were seen, so the raw capture is the output",
              file=sys.stderr)
        return raw
    return burn(ff, raw, out, marks, boot_end)


BEAT = re.compile(r"^\[drive\] beat: ([0-9.]+)s")


def marks_from(lines, scenes, t0):
    """When each scene was actually on screen, from the guest's own output.

    A scene ends when its beat ends, and the next begins there -- so a caption
    covers its command running *and* the dwell after it, which is what somebody
    watching sees as one moment. `drive.py` prints the beat when it starts and
    then sleeps, so the end is arithmetic on a timestamp rather than another
    line to wait for.

    Scenes with no beat behind them are dropped rather than guessed at. A
    caption over the wrong part of a video is worse than no caption: one of
    them is missing and the other is wrong.
    """
    marks = []
    start = 0.0
    first = None
    i = 0
    for when, line in lines:
        m = BEAT.match(line.strip())
        if not m:
            continue
        if i >= len(scenes):
            break
        if first is None:
            # The first beat is the first moment the shell existed and took a
            # line, so everything before it is boot.
            first = max(0.0, when - t0)
        end = (when - t0) + float(m.group(1))
        _, _, caption = scenes[i]
        if caption and end > start:
            marks.append((max(0.0, start), end, caption))
        start = end
        i += 1
    return marks, (first or 0.0)


def burn(ff, raw, out, marks, boot_end):
    """Second pass: timelapse the boot, burn the captions, fade the ends.

    Separate from the capture because a live `gdigrab` with a text filter on it
    drops frames under load, and because the spans are not known until the run
    is over.

    **The boot half and the demo half play at different speeds**, so every
    caption time has to be remapped rather than used as measured. Filming
    starts when QEMU opens its window, which is seconds in, and the shell does
    not exist for another six minutes -- so a video that played all of it at
    1x would be mostly a progress bar. `trim`/`setpts`/`concat` cuts it in two,
    runs the first piece at `BOOT_SPEED` and the second at 1x, and `at()` puts
    each caption where its frames ended up.
    """
    work = out.parent / (out.stem + "-caps")
    work.mkdir(parents=True, exist_ok=True)
    font = caption_font(work)

    fast = boot_end > 8.0

    def at(t):
        """Where a moment in the capture lands in the finished cut."""
        if not fast or t <= boot_end:
            return (t / BOOT_SPEED) if fast else t
        return boot_end / BOOT_SPEED + (t - boot_end)

    graph = []
    if fast:
        # `setpts` alone leaves the stream claiming its old duration, so the
        # trim on the second piece has to come from the source and be rebased.
        graph.append(f"[0:v]trim=0:{boot_end:.3f},setpts=PTS/{BOOT_SPEED}[b0]")
        graph.append(f"[0:v]trim={boot_end:.3f},setpts=PTS-STARTPTS[b1]")
        graph.append("[b0][b1]concat=n=2:v=1:a=0[base]")
        src = "[base]"
    else:
        src = "[0:v]"

    chain = []
    if fast:
        # The boot caption belongs to footage no scene row can describe, so it
        # is placed here rather than coming out of the table.
        bf = work / "cap-boot.txt"
        bf.write_text(BOOT_CAPTION, encoding="utf-8")
        chain.append(_text(font, bf.name, 0.0, at(boot_end)))
    for n, (a, b, text) in enumerate(marks):
        # The text goes in a file, so a caption containing a comma, a colon or
        # an apostrophe cannot break the filter graph. Every one of those is
        # ordinary English and two of them are filter syntax.
        cf = work / f"cap{n:02d}.txt"
        cf.write_text(text, encoding="utf-8")
        chain.append(_text(font, cf.name, at(a), at(b)))

    end = at(marks[-1][1])
    chain.append("fade=t=in:st=0:d=0.7")
    chain.append(f"fade=t=out:st={max(0.0, end - 0.9):.2f}:d=0.9")
    graph.append(src + ",".join(chain) + "[v]")

    script = work / "filter.txt"
    script.write_text(";".join(graph), encoding="utf-8")

    if fast:
        print(f"[record] boot is {boot_end:.0f}s, shown at {BOOT_SPEED:g}x "
              f"({boot_end / BOOT_SPEED:.0f}s)")
    print(f"[record] burning {len(marks)} caption(s), cutting at {end:.1f}s")
    for a, b, text in marks:
        print(f"  {at(a):6.1f} - {at(b):6.1f}  {text}")

    r = subprocess.run(
        [str(ff), "-hide_banner", "-loglevel", "error",
         "-i", str(raw.resolve()),
         "-filter_complex_script", str(script.resolve()),
         "-map", "[v]",
         # Cut where the last caption ends. What follows is the guest being
         # shut down, which is not part of the demo.
         "-t", f"{end:.2f}",
         "-c:v", "libx264", "-preset", "slow", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-y", str(out.resolve())],
        # Relative `fontfile=` and `textfile=` resolve against this, which is
        # how the drive-letter colon is kept out of the filter graph.
        cwd=work)
    if r.returncode != 0 or not out.exists():
        print("[record] the caption pass failed; the raw capture is still at "
              f"{raw}", file=sys.stderr)
        return raw
    print(f"[record] {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def _text(font, textfile, a, b):
    """One caption, as a lower third."""
    return (
        f"drawtext=fontfile={font}:textfile={textfile}"
        f":fontsize={CAPTION_PT}:fontcolor=white"
        f":box=1:boxcolor=black@0.78:boxborderw=18"
        f":x=(w-text_w)/2:y=h-text_h-{CAPTION_PAD}"
        f":enable='between(t,{a:.2f},{b:.2f})'"
    )


def wait_for_mode(limit=900.0):
    """Block until the guest is at the mode it was given, not merely on screen.

    **Waiting for the window was the bug, and the assertion below caught it and
    was ignored.** QEMU opens its window at the firmware's default 640x480 and
    the guest switches to 1280x800 a little later, so filming from the moment
    the window existed locked `gdigrab`'s crop at 640x480 -- and what came out
    was the **top-left quarter** of the desktop at 1:1, with the taskbar, the
    tray and half of every window simply absent. Thirteen minutes of it, and
    the warning had printed on line one.

    So the crop is not taken until the client area *is* the mode. What that
    costs is the firmware splash, which is a progress bar on a blank screen;
    what it buys is every pixel of the part anybody wants to see.

    Answers the crop box, or `None` if the mode never arrived.
    """
    want_w, want_h = (int(v) for v in RES.split("x"))
    t0 = time.time()
    seen = None
    while time.time() - t0 < limit:
        box = client_crop(quiet=True)
        if box is not None:
            seen = box
            if (box[0], box[1]) == (want_w, want_h):
                # A mode is set a moment before anything is drawn in it.
                time.sleep(0.6)
                return client_crop()
        time.sleep(0.5)
    if seen is not None:
        print(
            f"[record] the guest never reached {want_w}x{want_h}; it is at "
            f"{seen[0]}x{seen[1]}. Filming anyway, and the capture is not "
            f"pixel-perfect.",
            file=sys.stderr,
        )
    return seen


def client_crop(quiet=False):
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
    if (w, h) != (want_w, want_h) and not quiet:
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

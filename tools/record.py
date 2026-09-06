#!/usr/bin/env python3
"""Record what is on the guest's screen, as a GIF.

`drive.py --screenshot` takes one picture and takes it *after* the harness
stops, which for anything full-screen means a photograph of the desktop the
teardown just restored. That has been enough to show a frame arrived and is
useless for showing a frame arriving *again*: a program that renders once and
one that renders sixty times look identical in a still.

QEMU's monitor answers `screendump` at any moment, so this connects to it
while the guest is running and pulls frames on a fixed cadence. Nothing in
the guest knows it is being recorded, which is the point -- the alternative
would be the program writing its own frames somewhere, which measures the
program rather than the machine.

    .\\tools\\venv\\Scripts\\python.exe tools\\record.py --frames 40 --every 0.25 \\
        --out out/spin.gif -- "linux deadline off" "linux run /tmp/spin 25"

Everything after `--` is handed to `drive.py` as its command list. The guest
has to outlive the recording, so a `linux deadline` long enough for it is the
caller's business and is not assumed here.
"""

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

MONITOR = ("127.0.0.1", 45455)


def monitor_connect(timeout: float) -> socket.socket:
    """Wait for QEMU's monitor to exist, then take it.

    Polled rather than assumed ready: `drive.py` launches QEMU and the monitor
    port opens a moment later, and connecting too early is a refusal that
    reads like QEMU never started.
    """
    until = time.time() + timeout
    while time.time() < until:
        try:
            s = socket.create_connection(MONITOR, timeout=5)
            s.settimeout(5)
            return s
        except OSError:
            time.sleep(0.5)
    raise SystemExit("the QEMU monitor never opened on 45455")


def dump(sock: socket.socket, path: Path) -> None:
    # QEMU wants a host path and writes PPM. Backslashes are fine, but the
    # monitor takes the rest of the line verbatim so the path must not be
    # quoted.
    sock.sendall(f"screendump {path}\n".encode())
    try:
        sock.recv(65536)
    except OSError:
        pass


def stop(proc: subprocess.Popen) -> None:
    """End `drive.py` **and the QEMU under it**.

    `terminate()` alone leaks the emulator, and the leak is expensive out of
    all proportion to itself: the serial and monitor ports are fixed at 45454
    and 45455, so the survivor holds them and the *next* run attaches to the
    wrong guest's serial and sits there until it times out with every command
    unsent. That reads exactly like a kernel that hung on boot, and the only
    tell is `.qemu/qemu-stderr.log` saying "Failed to find an available port".
    It cost two runs before it was worth writing this.

    On Windows `terminate()` is `TerminateProcess`, which is immediate and
    gives `drive.py` no chance to clean up after itself, so the child has to be
    taken explicitly -- `taskkill /T` is the tree. Elsewhere the process group
    is the same idea.
    """
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--every", type=float, default=0.25, help="seconds between frames")
    ap.add_argument("--wait", type=float, default=900.0,
                    help="how long to let the guest boot before recording")
    ap.add_argument("--settle", type=float, default=0.0,
                    help="seconds to wait after the last command before the first frame")
    ap.add_argument("--out", default="out/spin.gif")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    cmds = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    if not cmds:
        ap.print_help()
        return 1

    shots = Path("out/record")
    for p in shots.glob("*.ppm"):
        p.unlink()
    shots.mkdir(parents=True, exist_ok=True)

    py = Path("tools/venv/Scripts/python.exe")
    drive = [str(py), "tools/drive.py", "--qemu-extra", "-accel whpx -cpu max",
             "--timeout", str(int(a.wait + a.frames * a.every + 120))] + cmds
    log = Path("out/record/drive.log")
    proc = subprocess.Popen(drive, stdout=log.open("w"), stderr=subprocess.STDOUT)

    sock = monitor_connect(60)
    # The guest is not up yet; the log is what says when it is. Watching for
    # the last command to be sent is the only signal available from here.
    needle = f"sent: {cmds[-1]}"
    until = time.time() + a.wait
    while time.time() < until:
        if proc.poll() is not None:
            print("drive.py exited before the guest ran; see out/record/drive.log")
            return 1
        try:
            if needle in log.read_text(errors="replace"):
                break
        except OSError:
            pass
        time.sleep(1.0)
    else:
        print(f"the guest never reached {cmds[-1]!r} within {a.wait:.0f}s")
        stop(proc)
        return 1

    if a.settle:
        time.sleep(a.settle)
    print(f"recording {a.frames} frame(s) every {a.every}s")
    got = []
    for i in range(a.frames):
        p = (shots / f"f{i:04d}.ppm").resolve()
        dump(sock, p)
        time.sleep(a.every)
        if p.exists():
            got.append(p)
    print(f"  {len(got)} frame(s)")

    stop(proc)

    if not got:
        return 1
    try:
        from PIL import Image
    except ImportError:
        print("  Pillow is not in the venv; the frames are in out/record")
        return 1

    imgs = []
    for p in got:
        im = Image.open(p).convert("RGB")
        if a.scale != 1.0:
            im = im.resize((int(im.width * a.scale), int(im.height * a.scale)),
                           Image.LANCZOS)
        imgs.append(im)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # A palette per frame rather than one for the whole animation: the scene
    # is a few saturated colours on a dark ground, and a shared 256-colour
    # palette bands the gouraud interpolation into stripes -- which would look
    # like a bug in the rasteriser rather than in the encoder.
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=int(a.every * 1000), loop=0, optimize=False)
    print(f"  {out}  {out.stat().st_size:,} B  {imgs[0].width}x{imgs[0].height}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

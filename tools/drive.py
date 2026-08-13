#!/usr/bin/env python3
"""Boot GLaDOS in QEMU and drive its shell over a serial socket.

Why a socket rather than stdio
------------------------------
QEMU's Windows stdio chardev reads console handles directly, so piping a script
into it does nothing at all -- silently. Redirecting the output loses it the
same way. A TCP chardev behaves like a socket on every platform, which makes
the boot log capturable and the shell scriptable, and that is the difference
between "the selftests probably pass" and knowing.

Usage:
    drive.py [--timeout N] [--memory 2048M] [cmd ...]

Each positional argument is one shell line. With none, it just captures the
boot log and exits at the first prompt.
"""

import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 45454
MONITOR_PORT = 45455
PROMPT = b"glados> "


def find_qemu():
    for c in [
        Path("C:/Program Files/qemu/qemu-system-x86_64.exe"),
        Path.home() / "scoop/apps/qemu/current/qemu-system-x86_64.exe",
        Path("C:/Program Files (x86)/qemu/qemu-system-x86_64.exe"),
    ]:
        if c.exists():
            return str(c)
    raise SystemExit("qemu-system-x86_64 not found")


def find_firmware():
    """Firmware arguments, with NVRAM reset to pristine every run.

    The vars image accumulates boot entries. A stale Boot0001 pointing at a
    device that is no longer attached sends the firmware to the UEFI shell
    instead of to `\\EFI\\BOOT\\BOOTX64.EFI`, which presents as the system not
    booting rather than as leftover state. A scripted run wants the same
    starting conditions every time, so this copies a fresh one.
    """
    share = Path(find_qemu()).parent / "share"
    for name in ("edk2-x86_64-code.fd", "OVMF_CODE.fd"):
        code = share / name
        if not code.exists():
            continue
        for vname in ("edk2-i386-vars.fd", "OVMF_VARS.fd"):
            pristine = share / vname
            if pristine.exists():
                scratch = ROOT / ".qemu/drive-vars.fd"
                scratch.parent.mkdir(parents=True, exist_ok=True)
                scratch.write_bytes(pristine.read_bytes())
                return [
                    "-drive", f"if=pflash,format=raw,unit=0,readonly=on,file={code}",
                    "-drive", f"if=pflash,format=raw,unit=1,file={scratch}",
                ]
    combined = share / "OVMF.fd"
    if combined.exists():
        return ["-bios", str(combined)]
    raise SystemExit(f"no UEFI firmware in {share}")


def capture(dest):
    """Ask QEMU's monitor for a screenshot, and convert it to PNG.

    QEMU writes PPM, which nothing on this machine opens. The conversion is
    hand-rolled -- PNG is a zlib stream of filtered scanlines wrapped in four
    CRC'd chunks, and `zlib` is in the standard library -- rather than adding a
    dependency to a repo whose entire point is not having any.
    """
    import binascii
    import struct as _s
    import zlib

    dest = Path(dest)
    ppm = dest.with_suffix(".ppm")
    try:
        mon = socket.create_connection(("127.0.0.1", MONITOR_PORT), timeout=5)
    except OSError as e:
        print(f"[drive] no monitor: {e}", file=sys.stderr)
        return
    # Let the guest finish whatever it is painting. A repaint here is a
    # million stores and the capture is asynchronous, so without this the
    # screenshot can land mid-frame -- which reads as a broken window manager
    # rather than as a broken screenshot, and cost a bisect to rule out.
    time.sleep(2.0)
    with mon:
        mon.settimeout(2.0)
        try:
            mon.recv(4096)
        except OSError:
            pass
        # Forward slashes: the monitor treats a backslash as an escape.
        mon.sendall(f"screendump {ppm.as_posix()}\n".encode())
        time.sleep(1.5)
        try:
            mon.recv(4096)
        except OSError:
            pass

    if not ppm.exists():
        print("[drive] screendump produced nothing", file=sys.stderr)
        return

    raw = ppm.read_bytes()
    # P6 header: magic, width height, maxval -- each possibly separated by any
    # whitespace, with # comments allowed between them.
    fields, i = [], 2
    while len(fields) < 3:
        while i < len(raw) and raw[i : i + 1].isspace():
            i += 1
        if raw[i : i + 1] == b"#":
            while i < len(raw) and raw[i] != 0x0A:
                i += 1
            continue
        j = i
        while j < len(raw) and not raw[j : j + 1].isspace():
            j += 1
        fields.append(int(raw[i:j]))
        i = j
    w, h, _maxval = fields
    pix = raw[i + 1 :]

    stride = w * 3
    lines = b"".join(b"\x00" + pix[y * stride : (y + 1) * stride] for y in range(h))

    def chunk(tag, data):
        return (
            _s.pack(">I", len(data))
            + tag
            + data
            + _s.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", _s.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(lines, 6))
        + chunk(b"IEND", b"")
    )
    dest.write_bytes(png)
    ppm.unlink(missing_ok=True)
    print(f"[drive] screenshot {dest} ({w}x{h})")


def main():
    argv = sys.argv[1:]
    timeout = 240
    memory = "2048M"
    shot = None
    if "--screenshot" in argv:
        i = argv.index("--screenshot")
        shot = Path(argv[i + 1])
        del argv[i:i + 2]
    if "--timeout" in argv:
        i = argv.index("--timeout")
        timeout = int(argv[i + 1])
        del argv[i:i + 2]
    if "--memory" in argv:
        i = argv.index("--memory")
        memory = argv[i + 1]
        del argv[i:i + 2]
    commands = argv

    esp = ROOT / "esp"

    # Stage the binary the same way run.ps1 does. Without this the firmware
    # finds no bootloader and reports "Not Found", which looks nothing like
    # "you forgot to copy the build".
    built = ROOT / "target/x86_64-unknown-uefi/release/glados.efi"
    if not built.exists():
        built = ROOT / "target/x86_64-unknown-uefi/debug/glados.efi"
    if not built.exists():
        raise SystemExit(f"no build artifact under {ROOT / 'target'}; run cargo build first")
    boot = esp / "EFI/BOOT"
    boot.mkdir(parents=True, exist_ok=True)
    (boot / "BOOTX64.EFI").write_bytes(built.read_bytes())

    # QEMU's VVFAT is FAT16 with a fixed geometry, and 516 MB is the whole disk.
    # Say so here rather than letting the -drive parser refuse with a number
    # that looks arbitrary.
    total = sum(f.stat().st_size for f in esp.rglob("*") if f.is_file())
    if total > 500 * 1024 * 1024:
        raise SystemExit(
            f"esp/ holds {total / 1024 / 1024:.0f} MB; QEMU's VVFAT caps at 516 MB.\n"
            "Stage a smaller checkpoint to exercise the kernel here -- a model "
            "this size can only be run on the GF63."
        )

    args = [
        find_qemu(),
        "-machine", "q35",
        "-m", memory,
        *find_firmware(),
        # Plain VVFAT, which is FAT16. `fat:32:` raises the 516 MB ceiling in
        # principle, but QEMU says outright that its FAT32 is untested and the
        # firmware cannot read the directory it produces -- the guest boots to
        # the UEFI shell having found no bootloader. So the ceiling stands, and
        # a model larger than it is not testable here at all. See the size
        # check above.
        "-drive", f"format=raw,file=fat:rw:{esp}",
        "-drive", f"file={ROOT / '.qemu/nvme.img'},if=none,id=nvm0,format=raw",
        "-device", "nvme,serial=GLADOSQEMU0001,drive=nvm0",
        # A USB controller to develop against. QEMU emulates xHCI faithfully
        # enough to bring up rings and enumerate, which is the whole reason the
        # USB work can be done before the dongle is involved at all.
        "-device", "qemu-xhci,id=xhci",
        # Something to enumerate. usb-net is also the eventual goal: a USB
        # network device is what the dongle will look like once its driver
        # exists, so proving enumeration against one is not a detour.
        "-device", "usb-net,bus=xhci.0",
        "-serial", f"tcp:127.0.0.1:{PORT},server=on,wait=on",
        # The monitor is how a screenshot happens. The serial transcript proves
        # a panel's *behaviour*; it says nothing about whether the thing on
        # screen is legible, and a GUI that has never been looked at is a GUI
        # nobody has tested.
        "-monitor", f"tcp:127.0.0.1:{MONITOR_PORT},server=on,wait=off",
        "-display", "none",
        "-net", "none",
        "-no-reboot",
    ]

    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    sock = None
    for _ in range(100):
        try:
            sock = socket.create_connection(("127.0.0.1", PORT), timeout=5)
            break
        except OSError:
            time.sleep(0.1)
    if sock is None:
        proc.kill()
        # QEMU's own complaint is the only useful thing here, and discarding it
        # turns every launch problem into the same opaque message.
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        raise SystemExit(f"could not connect to the serial socket\n{err.strip()}")

    sock.settimeout(1.0)
    buf = bytearray()
    deadline = time.time() + timeout
    queue = list(commands)
    sent_all = not queue
    idle_prompts = 0

    try:
        while time.time() < deadline:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                chunk = b""
            except OSError:
                break
            if chunk:
                buf += chunk
                sys.stdout.write(chunk.decode("utf-8", "replace"))
                sys.stdout.flush()

            # The shell echoes a prompt when it is ready for the next line.
            if buf.endswith(PROMPT) or (not chunk and buf.rstrip().endswith(PROMPT.strip())):
                if queue:
                    line = queue.pop(0)
                    sock.sendall(line.encode() + b"\r")
                    buf.clear()
                    time.sleep(0.2)
                else:
                    idle_prompts += 1
                    sent_all = True
                    if idle_prompts >= 2:
                        if shot:
                            capture(shot)
                        break
                    time.sleep(0.5)
    finally:
        try:
            sock.close()
        except OSError:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    if not sent_all:
        print(f"\n[drive] TIMEOUT after {timeout}s with {len(queue)} commands unsent",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

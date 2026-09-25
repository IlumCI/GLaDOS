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
    drive.py --stage-iso MODEL.BIN [--tokenizer TOK.BIN] [--memory 3072M] [cmd ...]
    drive.py --no-payload "diag all"      # no checkpoint, tokenizer or roots

Each positional argument is one shell line. With none, it just captures the
boot log and exits at the first prompt.

VVFAT cannot hold more than ~500 MB, which excludes the real Qwen3.5
checkpoint. `--stage-iso` assembles the same ESP tree into a FAT32 image,
wraps it El Torito via tools/mkiso.py, and boots off -cdrom, which has no
size cap; guest RAM still has to cover the weights, so raise --memory.
"""

# ### Booting this on Linux, which had never been done
#
# The firmware search below was written against Windows and Debian and missed
# Arch entirely -- wrong directory and a different spelling of the same file --
# so a perfectly ordinary Arch host reached "no UEFI firmware found" with OVMF
# two directories away. Fixed in `find_firmware`.
#
# The recipe, which is the WHPX one from CLAUDE.md with the accelerator
# changed. KVM is to Linux what WHPX is to Windows and `-cpu host` is what
# `-cpu max` was for -- without it the guest sees no AVX2 and `train` declines:
#
#     cargo build --release
#     python3 tools/drive.py --no-payload --qemu-extra "-accel kvm -cpu host" \
#         "diag all"
#
# Measured on a 12th Gen i7-12650H with /dev/kvm at 666: boots clean, answers
# the prompt, and `diag all` reads **68 passed, 0 failed**.
#
# **`--no-payload` is not a full verification and the tally does not say so.**
# With no checkpoint `ai::init` returns early, and the boot prints about
# sixteen `[selftest]` sections where CLAUDE.md documents twenty-nine. The
# *suites* still all run -- `diag all` is 68 of 68 including the AI arithmetic
# ones -- so the green tally is true and narrower than it looks. This is the
# hazard `.github/actions/verify-boot` grew a section-count-by-name check for,
# reproduced here rather than read about.
#
# And counting those headings is not quite the stale-proof measure CLAUDE.md
# calls it: a clean boot prints eighteen `[selftest]` lines, of which one is a
# continuation (`survived int3 -- idt is live.`) and one is `crypto:` a second
# time. Grep the count and subtract two, or count the colons.


import codecs
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 45454
MONITOR_PORT = 45455
PROMPT = b"glados> "
NVME_IMAGE_BYTES = 192 * 1024 * 1024


def find_qemu():
    """The emulator, wherever this host keeps it.

    PATH first, because that is where every packaged qemu is and this has to
    run on a CI runner as well as on the development laptop -- the Windows
    paths below were the whole of the search and a Linux host had no way in.
    """
    on_path = shutil.which("qemu-system-x86_64")
    if on_path:
        return on_path
    for c in [
        Path("C:/Program Files/qemu/qemu-system-x86_64.exe"),
        Path.home() / "scoop/apps/qemu/current/qemu-system-x86_64.exe",
        Path("C:/Program Files (x86)/qemu/qemu-system-x86_64.exe"),
    ]:
        if c.exists():
            return str(c)
    raise SystemExit("qemu-system-x86_64 not found on PATH or in the usual places")


def find_firmware():
    """Firmware arguments, with NVRAM reset to pristine every run.

    The vars image accumulates boot entries. A stale Boot0001 pointing at a
    device that is no longer attached sends the firmware to the UEFI shell
    instead of to `\\EFI\\BOOT\\BOOTX64.EFI`, which presents as the system not
    booting rather than as leftover state. A scripted run wants the same
    starting conditions every time, so this copies a fresh one.
    """
    # Several directories, because a packaged qemu does not keep its firmware
    # beside itself. A Windows build has `share/` next to the exe; Debian and
    # Ubuntu put OVMF in its own directory under `/usr/share` and the split
    # code/vars pair is the only form there. Searched in order and the first
    # complete pair wins, so a host with both keeps whichever it had.
    shares = [Path(find_qemu()).parent / "share"]
    shares += [Path(p) for p in (
        "/usr/share/OVMF",
        "/usr/share/OVMF/x64",
        "/usr/share/ovmf",
        # Arch keeps the pair one directory further down than Debian does, and
        # spells the size differently again -- `/usr/share/ovmf/x64/`, holding
        # `OVMF_CODE.4m.fd`. Neither the directory nor the name was in this
        # list, so a perfectly ordinary Arch host reached "no UEFI firmware
        # found" while the firmware sat two directories away.
        "/usr/share/ovmf/x64",
        "/usr/share/qemu",
        "/usr/share/edk2/ovmf",
        "/usr/share/edk2/x64",
        "/usr/share/edk2-ovmf/x64",
    )]
    for share in shares:
        # **The 4 MB pair is tried before the 2 MB one.** A host carrying both
        # wants the larger, and more to the point `OVMF.fd` at the bottom of
        # this function is a *combined* image whose vars are not writable --
        # so falling through to it loses the pristine-NVRAM reset this whole
        # function exists to perform.
        #
        # `.4m.fd` and `_4M.fd` are the same firmware under two spellings, one
        # Arch and one Debian. Listing both is the entire fix; guessing at a
        # pattern would be a glob that also matches `OVMF_CODE.secboot.4m.fd`,
        # which is a different firmware that refuses to boot an unsigned image
        # -- and presents as this kernel not booting rather than as the wrong
        # file being chosen.
        for name in ("edk2-x86_64-code.fd", "OVMF_CODE.4m.fd", "OVMF_CODE_4M.fd",
                     "OVMF_CODE.fd"):
            code = share / name
            if not code.exists():
                continue
            for vname in ("edk2-i386-vars.fd", "OVMF_VARS.4m.fd", "OVMF_VARS_4M.fd",
                          "OVMF_VARS.fd"):
                pristine = share / vname
                if pristine.exists():
                    scratch = ROOT / ".qemu/drive-vars.fd"
                    scratch.parent.mkdir(parents=True, exist_ok=True)
                    scratch.write_bytes(pristine.read_bytes())
                    return [
                        "-drive", f"if=pflash,format=raw,unit=0,readonly=on,file={code}",
                        "-drive", f"if=pflash,format=raw,unit=1,file={scratch}",
                    ]
    for share in shares:
        combined = share / "OVMF.fd"
        if combined.exists():
            return ["-bios", str(combined)]
    raise SystemExit(
        "no UEFI firmware found; looked in " + ", ".join(str(s) for s in shares)
    )


def monitor(lines):
    """Send commands to QEMU's monitor.

    The mouse is the reason this exists separately from `capture`: there is no
    way to inject a PS/2 packet over the serial console, so the only way to
    test a pointer headlessly is to ask the emulator to move it.
    """
    try:
        mon = socket.create_connection(("127.0.0.1", MONITOR_PORT), timeout=5)
    except OSError as e:
        print(f"[drive] no monitor: {e}", file=sys.stderr)
        return
    with mon:
        mon.settimeout(2.0)
        try:
            mon.recv(4096)
        except OSError:
            pass
        for line in lines:
            mon.sendall((line + "\n").encode())
            time.sleep(0.35)
        try:
            mon.recv(4096)
        except OSError:
            pass


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

    w, h = ppm_to_png(ppm, dest)
    print(f"[drive] screenshot {dest} ({w}x{h})")


def ppm_to_png(ppm, dest):
    """Convert one QEMU screendump, and answer its size.

    Split out of `capture` because the recorder needs it too, and two copies
    of a hand-rolled PNG writer is exactly the kind of thing that ends with
    one of them subtly wrong.
    """
    import binascii
    import struct as _s
    import zlib

    ppm = Path(ppm)
    dest = Path(dest)
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
    # Not when they are the same file. A destination named `.ppm` made this
    # delete the PNG it had just written, and reported success doing it.
    if ppm != dest:
        ppm.unlink(missing_ok=True)
    return w, h


def record(dest_dir, frames, gap):
    """Dump a numbered PNG sequence while the guest carries on running.

    A timelapse and not a screen recording, and the distinction is forced by
    the mechanism: `screendump` is a monitor round-trip that writes a 3 MB PPM,
    so a few frames a second is the ceiling and anything the machine does in
    under a second is invisible to it. That is fine for what this is for --
    the machine writing an application takes minutes, and 120 frames four
    seconds apart is the whole run in five seconds at 24fps.

    One connection for the whole sequence rather than one per frame: the
    per-frame cost is otherwise two socket handshakes and `capture`'s 3.5s of
    conservative sleeping, which is most of the interval.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ppm = dest_dir / "_frame.ppm"
    try:
        mon = socket.create_connection(("127.0.0.1", MONITOR_PORT), timeout=5)
    except OSError as e:
        print(f"[drive] no monitor: {e}", file=sys.stderr)
        return
    kept = 0
    with mon:
        mon.settimeout(2.0)
        try:
            mon.recv(4096)
        except OSError:
            pass
        for i in range(frames):
            t0 = time.time()
            ppm.unlink(missing_ok=True)
            mon.sendall(f"screendump {ppm.as_posix()}\n".encode())
            # Waited for by watching the file settle rather than by sleeping a
            # fixed amount: the dump is asynchronous, and a fixed sleep is
            # either a torn frame or most of the interval spent idle.
            #
            # **And not by waiting for the monitor prompt either**, which was
            # tried, looks obviously right, and is wrong: QEMU answers `(qemu)`
            # before the file is flushed and closed, so the converter gets half
            # a frame and the next iteration cannot even delete it -- on
            # Windows that is `WinError 32`, the file still being open. The
            # comment above already said the dump was asynchronous. It was
            # read as a description of the sleep rather than as the reason for
            # it, and the correction cost a fifteen-minute run.
            #
            # The poll is 30 ms rather than 80 because at four seconds between
            # frames the difference was invisible and at a tenth of a second it
            # is most of the budget.
            size, stable, deadline = -1, 0, time.time() + 5.0
            while time.time() < deadline:
                time.sleep(0.03)
                now = ppm.stat().st_size if ppm.exists() else -1
                if now == size and now > 0:
                    stable += 1
                    if stable >= 2:
                        break
                else:
                    stable = 0
                size = now
            try:
                mon.recv(4096)
            except OSError:
                pass
            if not ppm.exists() or ppm.stat().st_size == 0:
                continue
            try:
                ppm_to_png(ppm, dest_dir / f"f{i:04d}.png")
            except Exception as e:  # a torn frame is a dropped frame
                print(f"[drive] frame {i} unreadable: {e}", file=sys.stderr)
                continue
            kept += 1
            if kept % 10 == 0:
                print(f"[drive] recorded {kept}/{frames}")
            left = gap - (time.time() - t0)
            if left > 0:
                time.sleep(left)
    ppm.unlink(missing_ok=True)
    print(f"[drive] recorded {kept} frame(s) into {dest_dir}")


# One decoder for the whole session rather than one per socket read.
#
# UTF-8 spreads a character over up to four bytes and `recv` splits wherever
# it likes, so decoding each chunk on its own turned every accented letter
# into two replacement characters and every box-drawing character into three
# -- which reads exactly like a guest that cannot print them. It could not:
# the guest was right and the log was wrong. An incremental decoder carries
# the partial sequence into the next chunk, which is the same thing the
# console itself had to learn to do.
_DECODER = codecs.getincrementaldecoder("utf-8")("replace")


def emit(chunk):
    sys.stdout.write(_DECODER.decode(chunk))
    sys.stdout.flush()


def ports_held():
    """Which of the two fixed ports something else already owns.

    Asked *before* launching, because afterwards the failure is invisible. The
    serial chardev is `server=on,wait=on`, so a second QEMU cannot bind the
    port and exits -- and the `create_connection` below then succeeds anyway,
    against the **stale** QEMU that still owns it. What that looks like is a
    log with no boot output at all followed by a timeout with every command
    unsent, which reads exactly like a guest that died early.

    The existing `sock is None` guard cannot catch it: a socket was connected,
    just to the wrong machine. This has cost two sessions, and the second one
    spent two ten-minute runs on it.
    """
    held = []
    for what, port in (("serial", PORT), ("monitor", MONITOR_PORT)):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Deliberately no SO_REUSEADDR. On Windows it permits binding a port
        # another process is listening on, which is the opposite of the
        # question being asked here.
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            held.append(f"{what} on {port}")
        finally:
            s.close()
    return held


def main():
    # The default cp1252 stdout refuses characters the guest can now print,
    # which kills the session mid-run when output is redirected to a file.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
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
    # A real, writable ESP on a real block device.
    #
    # VVFAT's read-write mode cannot create or delete files, so anything that
    # asks the firmware to *write* the ESP -- the update hook -- reads
    # correctly and silently fails to change anything. This boots a raw FAT32
    # image instead, and the guest's writes stay in it across runs, which is
    # what makes the two-boot apply/prove flow observable.
    esp_on_nvme = "--esp-on-nvme" in argv
    if esp_on_nvme:
        argv.remove("--esp-on-nvme")
    esp_image = None
    if "--esp-image" in argv:
        i = argv.index("--esp-image")
        esp_image = Path(argv[i + 1])
        del argv[i:i + 2]
    esp_force = "--esp-rebuild" in argv
    if esp_force:
        argv.remove("--esp-rebuild")
    iso = None
    if "--iso" in argv:
        i = argv.index("--iso")
        iso = Path(argv[i + 1])
        del argv[i:i + 2]
    # Stage a checkpoint too large for VVFAT by building a one-shot ISO from
    # the same tree the VVFAT path would have assembled, and booting that off
    # -cdrom instead. This is how the real Qwen3.5 reaches QEMU at all: the
    # hybrid is 723 MB against VVFAT's 516, and FAT32-in-ISO has no such cap.
    stage_iso = None
    if "--stage-iso" in argv:
        i = argv.index("--stage-iso")
        stage_iso = Path(argv[i + 1])
        del argv[i:i + 2]
        if iso:
            raise SystemExit("--iso and --stage-iso are mutually exclusive")
    # Raw passthrough to the QEMU command line, shell-split. The default
    # qemu64 CPU model hides every SIMD extension, which costs an order of
    # magnitude on the int8 kernels; "-cpu max" exposes what the host has,
    # and "-accel whpx" swaps TCG for the Windows hypervisor where available.
    # Resending is off by default now, and `--resend` puts it back.
    #
    # It was added for genuine wire loss: bytes arriving at the guest UART and
    # never coming out, more often after a long silent boot. Every instance of
    # that was under TCG. Under the hypervisor accelerator the guest is
    # frequently and legitimately quiet -- the shell has printed a prompt and
    # is busy, or an episode holds the engine -- and the resend then puts a
    # second copy of the line into a buffer that already holds the first. The
    # result is `agent stopwin listwin list` on one line, which reads as a
    # guest fault and is this script's doing.
    #
    # A lost command is visible: the session times out naming it. A duplicated
    # one is not: it runs as garbage and the operator debugs the kernel.
    resend = "--resend" in argv
    if resend:
        argv.remove("--resend")
    qemu_extra = []
    if "--qemu-extra" in argv:
        i = argv.index("--qemu-extra")
        import shlex
        qemu_extra = shlex.split(argv[i + 1])
        del argv[i:i + 2]
    # 1920x1080 instead of whatever mode OVMF picks on its own, which is
    # 1280x800. A flag and not the default: every figure `video bench` has ever
    # recorded is at the default mode, and the graphics path is span fills and
    # a memcmp, so it scales with pixel count -- changing the default would
    # silently invalidate every one of them.
    #
    # `-vga none` first is not optional. `-global VGA.xres=` on the default
    # device does not set the mode, it breaks it: OVMF then cannot publish a
    # GOP at all, and the kernel boots with no framebuffer and no message.
    hd = "--hd" in argv
    if hd:
        argv.remove("--hd")
    # An explicit mode, because a window larger than the screen is not a window.
    #
    # `--hd` asks for 1920x1080 and that is right for a headless screenshot,
    # where nothing has to fit anywhere. A *recording* is captured off the host
    # desktop, so a guest bigger than the host screen has its window clipped and
    # the part hanging off the edge records as black -- which looks exactly like
    # a kernel that painted nothing. This host is 1536x864, so the recorder asks
    # for 1280x720: it fits at 1:1 with room for the title bar, which means the
    # capture is pixel-perfect rather than resampled, and it is already a
    # standard video mode on the way out.
    res = None
    if "--res" in argv:
        i = argv.index("--res")
        res = argv[i + 1].lower().split("x")
        del argv[i:i + 2]

    # A real window, for a camera to point at.
    #
    # `screendump` is the only capture this harness had, and it is a monitor
    # round-trip that writes a 3 MB PPM with a two-second settle in front of it
    # and a one-and-a-half-second wait behind: about 0.28 frames a second. That
    # is right for *a* screenshot and hopeless for video, and no amount of
    # tuning fixes it, because the cost is the round trip rather than the
    # encoding. So recording does not go through QEMU at all -- the guest gets
    # an ordinary SDL window and something else films it.
    #
    # `-name` is what titles that window, which is how `gdigrab` finds it.
    window = "--window" in argv
    if window:
        argv.remove("--window")
    rec_dir, rec_frames, rec_gap = None, 120, 4.0
    # When to start, measured from the moment the last command was *sent*.
    #
    # Zero means the old behaviour: wait for the prompt to come back. That is
    # right for a command that returns immediately and leaves the machine
    # working -- `initiative now`, `author` -- and useless for one that holds
    # the screen until it is finished, because by then the thing worth
    # filming is over. `edit` is the second kind, and so is `port bars`.
    #
    # Measured from the send and not from the first frame, so it has to cover
    # whatever the command spends before it draws: loading an app's own art
    # and sprites takes a couple of seconds on its own.
    rec_after = 0.0
    if "--record" in argv:
        i = argv.index("--record")
        rec_dir = argv[i + 1]
        del argv[i:i + 2]
    if "--record-n" in argv:
        i = argv.index("--record-n")
        rec_frames = int(argv[i + 1])
        del argv[i:i + 2]
    if "--record-after" in argv:
        i = argv.index("--record-after")
        rec_after = float(argv[i + 1])
        del argv[i:i + 2]
    if "--record-gap" in argv:
        i = argv.index("--record-gap")
        rec_gap = float(argv[i + 1])
        del argv[i:i + 2]
    mouse = []
    while "--mouse" in argv:
        i = argv.index("--mouse")
        mouse.append(argv[i + 1])
        del argv[i:i + 2]
    if "--memory" in argv:
        i = argv.index("--memory")
        memory = argv[i + 1]
        memory_given = True
        del argv[i:i + 2]
    else:
        memory_given = False
    # An override for checking a checkpoint the default staging cannot hold.
    # The port of Qwen3.5 needs a *hybrid* to exercise at all, and the real one
    # is 723 MB against VVFAT's 516; `tools/hybtest.py` builds a small one
    # shaped to hit every path it does.
    model_src = ROOT / "out/smollm2-135m.bin"
    # Boot with none of the ESP payload: no checkpoint, no tokenizer, no root
    # bundle. See the refusal below for why the default is to insist.
    no_payload = "--no-payload" in argv
    if no_payload:
        argv.remove("--no-payload")
    # **`--iso` implies it**, and without this a miner ISO could not be booted at
    # all: the refusal below insisted on a checkpoint in `out/` for a run whose
    # boot medium is the ISO and whose `\GLADOS\` came from whatever
    # `mkiso.py` was given. The staged ESP is not what such a guest reads.
    #
    # It also *clears* the staged files rather than merely skipping them, which
    # is the half that matters here: `.qemu/esp` persists between runs, so a
    # stale `model.bin` left beside an ISO is a second `\GLADOS\` for the
    # firmware to find, and which one wins is not a thing to leave to chance.
    if iso is not None and not no_payload:
        no_payload = True
        print("[drive] --iso, so nothing is staged: the image carries its own payload")
    model_given = False
    if "--model" in argv:
        i = argv.index("--model")
        model_src = Path(argv[i + 1])
        model_given = True
        del argv[i:i + 2]
    # The tokenizer has to match the checkpoint: Qwen3.5's vocabulary is
    # 248k against Qwen3's 152k, so staging the small tokenizer beside a big
    # model hands it ids that name different rows of the embedding.
    tokenizer_src = ROOT / "out/smollm2-tokenizer.bin"
    if "--tokenizer" in argv:
        i = argv.index("--tokenizer")
        tokenizer_src = Path(argv[i + 1])
        del argv[i:i + 2]
    # --stage-iso names the checkpoint it stages. Saying the model twice would
    # invite staging one while booting the other, so the flag carries both
    # meanings and an explicit --model still wins.
    if stage_iso is not None and not model_given:
        model_src = stage_iso
    # A staged checkpoint has to fit in guest RAM beside the firmware, the
    # heap ladder and the KV cache. Forgetting --memory reads as "no model at
    # \GLADOS\model.bin" because the pool allocation fails first, which is a
    # message about the wrong thing. Size for it here: weights plus ~2.3 GiB
    # of everything else, rounded up to a whole GiB.
    if stage_iso is not None and not memory_given:
        mb = model_src.stat().st_size / (1024 * 1024)
        gib = max(3, int(mb / 1024) + 3)
        memory = f"{gib * 1024}M"
        print(f"[drive] model is {mb:.0f} MB; guest RAM auto-set to {memory}")
    commands = argv

    # A QEMU-only ESP, assembled here rather than borrowing esp/.
    #
    # esp/ is the deploy staging directory and holds Qwen3, which is 574 MB and
    # cannot be hosted by VVFAT at all -- so testing used to mean copying
    # SmolLM2 over it and remembering to put Qwen3 back. Forgetting leaves the
    # GF63 staged with the wrong model, which is a silent and slow way to be
    # wrong. Build a separate tree instead: the small checkpoint is what QEMU
    # can run, and deploy staging is left alone.
    def differs(a, b):
        """Streamed content comparison. These files reach 723 MB; reading
        both whole into host RAM to compare them was acceptable at 135 MB
        and is not here."""
        if a.stat().st_size != b.stat().st_size:
            return True
        with open(a, 'rb') as fa, open(b, 'rb') as fb:
            while True:
                ca = fa.read(1 << 20)
                cb = fb.read(1 << 20)
                if ca != cb:
                    return True
                if not ca:
                    return False

    esp = ROOT / ".qemu/esp"
    (esp / "GLADOS").mkdir(parents=True, exist_ok=True)
    # The root bundle is optional in a way the other two are not. A missing
    # checkpoint is
    # silent and consequential -- `ai::init` returns early and takes eleven
    # boot selftest sections with it -- so it is refused. A missing root bundle
    # is neither: the kernel says so itself, at boot, and the state it leaves
    # is a real one that `Identity::NoTrustStore` names. So it is skipped with
    # a line rather than refused, which is what lets a CI runner boot at all.
    staged = [
        (model_src, "model.bin"),
        (tokenizer_src, "tokenizer.bin"),
    ]
    optional = [(ROOT / "esp/GLADOS/roots.der", "roots.der")]
    # The WAD staging went with `src/doom/`. A stale copy on the ESP would be
    # a file nothing reads, so it is cleared rather than left.
    old_wad = esp / "GLADOS" / "DOOM.WAD"
    if old_wad.exists():
        old_wad.unlink()

    for src, dst in optional:
        target = esp / "GLADOS" / dst
        if src.exists():
            if not target.exists() or differs(src, target):
                target.write_bytes(src.read_bytes())
        else:
            if target.exists():
                target.unlink()
            print(f"[drive] no {dst} -- TLS will encrypt and authenticate nothing")

    for src, dst in staged:
        # **Refused rather than skipped, unless somebody said so.** A run that
        # quietly booted without the checkpoint would look like a run where the
        # model was loaded and answered badly, which is the "drive.py prefers
        # the release artifact" failure by another route.
        #
        # `--no-payload` is the deliberate form, and it exists because a CI
        # runner has none of this: the weights are not in this repository and
        # never will be, so a gate that insisted on them could not run at all.
        # It removes rather than merely skips, because `.qemu/esp` persists
        # between runs and a stale copy is exactly what the flag is for
        # avoiding. Measured: `diag all` is 59 of 59 either way, because every
        # suite tests machinery and none of them asks the model a question.
        if no_payload or not src.exists():
            if not no_payload:
                raise SystemExit(f"missing {src}")
            stale = esp / "GLADOS" / dst
            if stale.exists():
                stale.unlink()
            print(f"[drive] no {dst} (--no-payload)")
            continue
        target = esp / "GLADOS" / dst
        # Content-compare rather than always copying: the copy is the slowest
        # thing in a run that is otherwise seconds.
        if not target.exists() or differs(src, target):
            target.write_bytes(src.read_bytes())

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

    # QEMU's VVFAT is FAT16 with a fixed geometry, and 516 MB is the whole
    # disk. Say so here rather than letting the -drive parser refuse with a
    # number that looks arbitrary. ISO staging has no such cap and skips it.
    total = sum(f.stat().st_size for f in esp.rglob("*") if f.is_file())
    if not stage_iso and total > 500 * 1024 * 1024:
        raise SystemExit(
            f"esp/ holds {total / 1024 / 1024:.0f} MB; QEMU's VVFAT caps at 516 MB.\n"
            "Stage a smaller checkpoint to exercise the kernel here, or pass "
            "--stage-iso to boot a larger one off an El Torito image."
        )

    if esp_image is not None:
        import mkesp

        # Only when it is not already there. The guest writes into this image
        # and those writes are the thing being observed, so rebuilding between
        # boots would erase the state the next boot is supposed to find --
        # which is precisely the two-boot flow the image exists to test.
        if esp_image.exists() and not esp_force:
            print(f"[drive] reusing {esp_image} (--esp-rebuild to start clean)")
        else:
            fat, tot = mkesp.build(esp, esp_image)
            print(f"[drive] built {esp_image} ({tot / 1024 / 1024:.0f} MB, writable)")

    # The NVMe scratch disk, which nothing had ever created.
    #
    # `-drive file=.qemu/nvme.img` is passed unconditionally below, `/.qemu` is
    # gitignored, and `mkfat.py` only ever *writes* that path when somebody
    # runs it by hand. So the file existed on the development machine because
    # it had been made there once, months ago, and existed nowhere else --
    # every fresh checkout got `Could not open ... nvme.img` out of QEMU before
    # the guest drew a single line. That is a checkout-shaped failure and it
    # went unnoticed for exactly as long as nothing ran on a fresh checkout.
    #
    # Only when it is absent, for the reason the ESP image above gives: this is
    # where `mkfat.py` stages corpus bundles, ACPI tables and ELF fixtures, so
    # rebuilding between boots would delete the thing the next command reads.
    #
    # Sparse and blank, which is the honest empty state rather than a
    # simulation of a populated one. `cas::find_free_region` needs the disk to
    # be larger than 3 MiB and reserves the last MiB for a GPT backup header
    # it will not find; 192 MiB is what the development machine has, so a CI
    # boot and a local one look at the same topology instead of two.
    if not esp_on_nvme:
        nvme = ROOT / ".qemu/nvme.img"
        if not nvme.exists():
            nvme.parent.mkdir(parents=True, exist_ok=True)
            with open(nvme, "wb") as f:
                f.truncate(NVME_IMAGE_BYTES)
            print(f"[drive] blank {nvme} "
                  f"({NVME_IMAGE_BYTES / 1024 / 1024:.0f} MB, no store on it) "
                  f"-- mkfat.py puts fixtures here")

    if stage_iso:
        import mkiso

        out_iso = ROOT / ".qemu/staged.iso"
        out_iso.parent.mkdir(parents=True, exist_ok=True)
        root = mkiso.Entry(None, 0)
        efi_dir = mkiso.Entry('EFI', 0)
        boot_dir = mkiso.Entry('BOOT', 0)
        boot_dir.children.append(
            mkiso.Entry('BOOTX64.EFI', built.stat().st_size, built))
        efi_dir.children.append(boot_dir)
        root.children.append(efi_dir)
        g = mkiso.Entry('GLADOS', 0)
        for f in sorted((esp / 'GLADOS').iterdir()):
            if f.is_file():
                g.children.append(mkiso.Entry(f.name, f.stat().st_size, f))
        root.children.append(g)

        cluster = 512
        while cluster < 32768 and total > 60000 * cluster:
            cluster *= 2

        esp_offset = 24 * mkiso.ISO_SECTOR
        expected = None
        with open(out_iso, 'wb') as fh:
            fh.write(b'\x00' * esp_offset)
            size = mkiso.build_fat(root, fh, cluster)
            tail = fh.tell() % mkiso.ISO_SECTOR
            if tail:
                fh.write(b'\x00' * (mkiso.ISO_SECTOR - tail))
            expected = fh.tell()
        # A short write means the host disk filled underneath us, and the
        # firmware's complaint about such an image ("Not Found") names
        # nothing resembling the cause.
        actual = out_iso.stat().st_size
        if actual != expected:
            raise SystemExit(
                f"{out_iso} is {actual} bytes, wanted {expected} -- "
                "the write did not complete; check free disk space"
            )
        mkiso.build_iso(out_iso, esp_offset, size, 'GLADOS')
        print(f"[drive] staged {total / 1024 / 1024:.0f} MB as {out_iso}")
        iso = out_iso

    # Four cores by default, because one is a suite that fails every time.
    #
    # QEMU gives a guest one vCPU unless told otherwise, so `smp::init` finds
    # no application processors to start and two of `diag mt`'s claims cannot
    # pass: "the sharing audit notices a second toucher" and "the allocator
    # was exercised from several cores" are both false on a machine with one
    # core. The suite printed `mt FAILED` on every clean boot this project has
    # ever driven, and a check that always fails is read as one nobody has to
    # look at -- the objection `smp.rs` already makes about its own canary.
    #
    # Four rather than two. Two satisfies the claims and exercises nothing
    # beyond a single contender for the chunk cursor, and the failure `smp.rs`
    # records there -- a worker claiming an index valid for the *next* job --
    # needs several claimants before it is likely.
    #
    # It costs nothing measurable. Best of nine decodes on SmolLM2 under WHPX:
    # 50,819 us/token at one core, 49,463 at two, 50,818 at four. That is a 3%
    # spread with no trend in it, and the single-sample figures that suggested
    # otherwise (65 ms against 95 ms) were the host's scheduler, which is the
    # error `video bench` was rewritten to stop making.
    #
    # And `logits 7 11 3` is bit-identical across all three, which is what
    # `smp.rs` claims for a split matvec and is worth checking rather than
    # believing, since splitting is live above 2^19 element-operations and the
    # classifier is far above it.
    smp = [] if any(a == "-smp" or a.startswith("-smp=") for a in qemu_extra) else ["-smp", "4"]

    args = [
        find_qemu(),
        "-machine", "q35",
        "-m", memory,
        *smp,
        *qemu_extra,
        *find_firmware(),
        # Plain VVFAT, which is FAT16. `fat:32:` raises the 516 MB ceiling in
        # principle, but QEMU says outright that its FAT32 is untested and the
        # firmware cannot read the directory it produces -- the guest boots to
        # the UEFI shell having found no bootloader. So the ceiling stands, and
        # a model larger than it is not testable here at all. See the size
        # check above.
        # An ISO boots the same kernel through El Torito instead of VVFAT,
        # which is the only way to test that the image tools/mkiso.py produces
        # is actually bootable rather than merely well-formed.
        *(["-cdrom", str(iso)] if iso else
          [] if esp_on_nvme else
          ["-drive", f"format=raw,file={esp_image}"] if esp_image is not None else
          ["-drive", f"format=raw,file=fat:rw:{esp}"]),
        # --esp-on-nvme puts the boot volume on the NVMe controller, which is
        # where it lives on the GF63: one disk, with the ESP as a partition of
        # it. The default topology has the ESP on its own drive, so the kernel's
        # own block layer -- which reads NVMe and nothing else -- cannot see the
        # volume it booted from, and anything that writes the ESP from a running
        # machine is untestable. OVMF enumerates NVMe as a boot device, so
        # nothing else has to change.
        "-drive", f"file={esp_image if esp_on_nvme else ROOT / '.qemu/nvme.img'},if=none,id=nvm0,format=raw",
        "-device", "nvme,serial=GLADOSQEMU0001,drive=nvm0",
        # A USB controller to develop against. QEMU emulates xHCI faithfully
        # enough to bring up rings and enumerate, which is the whole reason the
        # USB work can be done before the dongle is involved at all.
        "-device", "qemu-xhci,id=xhci",
        # Something to enumerate. usb-net is also the eventual goal: a USB
        # network device is what the dongle will look like once its driver
        # exists, so proving enumeration against one is not a detour.
        "-netdev", "user,id=usbnet",
        "-device", "usb-net,bus=xhci.0,netdev=usbnet",
        "-serial", f"tcp:127.0.0.1:{PORT},server=on,wait=on",
        # The monitor is how a screenshot happens. The serial transcript proves
        # a panel's *behaviour*; it says nothing about whether the thing on
        # screen is legible, and a GUI that has never been looked at is a GUI
        # nobody has tested.
        "-monitor", f"tcp:127.0.0.1:{MONITOR_PORT},server=on,wait=off",
        *(["-display", "sdl", "-name", "GLaDOS"] if window else ["-display", "none"]),
        *(["-vga", "none", "-device", f"VGA,xres={res[0]},yres={res[1]}"] if res else
          ["-vga", "none", "-device", "VGA,xres=1920,yres=1080"] if hd else []),
        "-no-reboot",
    ]

    held = ports_held()
    if held:
        raise SystemExit(
            "[drive] " + " and ".join(held) + " already in use.\n"
            "        Another drive.py, or a QEMU one left behind, still owns "
            "it.\n"
            "        Stop that first: this run would otherwise connect to "
            "*that* guest's\n"
            "        serial and time out with every command unsent."
        )

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
    pending = None
    # Set by a beat and cleared by the send it causes, so a pause does not
    # depend on a prompt coming back afterwards. See the beat below.
    force_send = False

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
                emit(chunk)
                # Fresh output means the guest is still working. Without this
                # the idle counter survives from the post-queue prompt of an
                # async command -- 'agent' returns its prompt immediately --
                # and the next quiet moment ends the session in the middle of
                # the episode it was supposed to be watching.
                idle_prompts = 0
                # It also means the command was not lost, so the resend clock
                # starts again. The Enter-echo only appears when the *shell*
                # reads the line, and the shell can be busy for a long time:
                # an agent episode holds the engine while it runs. Under TCG
                # the boot was slow enough that this never collided, and under
                # whpx the first command lands while the resident mind is
                # still in its first episode, gets no echo for eight seconds,
                # and is resent into a UART buffer that already holds it. The
                # result is 'initiative offinitiative offecho aecho aecho a'
                # on one line, which reads as a guest fault and is a driver
                # bug the emulator's slowness was hiding.
                if pending:
                    pending["at"] = time.time()

            # Acknowledgement watch: the guest's Enter-echo is the receipt.
            if pending:
                if len(buf) > pending["mark"]:
                    pending = None  # the guest saw the bytes
                elif resend and time.time() - pending["at"] > 25.0:
                    # Twenty-five seconds and one retry, raised from eight and
                    # three. The wire loss this recovers from was observed
                    # under TCG, where a boot was slow enough that a quiet
                    # eight seconds meant something was wrong. Under the
                    # hypervisor accelerator the guest is often legitimately
                    # silent for longer than that -- the shell has printed a
                    # prompt and is busy, or an episode holds the engine --
                    # and a resend into a UART buffer that already holds the
                    # line concatenates the two. A duplicated command is worse
                    # than a lost one, because a lost one is visible.
                    if pending["retries"] < 1:
                        pending["retries"] += 1
                        pending["at"] = time.time()
                        print(f"[drive] no echo -- resending "
                              f"(attempt {pending['retries']}): {pending['line']}")
                        # One screendump at first retry: the taskbar clock
                        # ticks every second, so two dumps tell alive from
                        # dead without touching the guest.
                        if pending["retries"] == 1:
                            try:
                                mon = socket.create_connection(
                                    ("127.0.0.1", MONITOR_PORT), timeout=5)
                                mon.settimeout(2.0)
                                mon.sendall(b"screendump .qemu/stall1.ppm\n")
                                time.sleep(1.0)
                                mon.close()
                                print("[drive] stall screendump 1")
                            except OSError:
                                pass
                        if pending["retries"] == 2:
                            try:
                                mon = socket.create_connection(
                                    ("127.0.0.1", MONITOR_PORT), timeout=5)
                                mon.settimeout(2.0)
                                mon.sendall(b"info chardev\n")
                                time.sleep(0.5)
                                info = b""
                                while True:
                                    try:
                                        info += mon.recv(4096)
                                    except socket.timeout:
                                        break
                                print("[drive] info chardev:",
                                      info.decode("utf-8", "replace")[:400])
                                mon.sendall(b"sendkey h\n")
                                time.sleep(0.3)
                                mon.sendall(b"screendump .qemu/stall2.ppm\n")
                                time.sleep(1.0)
                                mon.close()
                                a = Path(".qemu/stall1.ppm").read_bytes()
                                b = Path(".qemu/stall2.ppm").read_bytes()
                                print(f"[drive] stall screendump 2 (after "
                                      f"sendkey h); frames identical: "
                                      f"{a == b} (True = shell deaf to "
                                      f"keyboard too -> loop stuck; "
                                      f"False = shell alive, UART wedged)")
                            except OSError:
                                pass
                        sock.sendall(pending["line"].encode() + b"\r")
                        pending["mark"] = len(buf)
                    else:
                        print(f"[drive] giving up on: {pending['line']}",
                              file=sys.stderr)
                        pending = None

            # The shell echoes a prompt when it is ready for the next line.
            if (force_send or buf.endswith(PROMPT)
                    or (not chunk and buf.rstrip().endswith(PROMPT.strip()))):
                if queue:
                    line = queue.pop(0)
                    # `@wait N` is a beat rather than a command: it is handled
                    # here and never reaches the guest.
                    #
                    # This loop sends the next line the moment a prompt comes
                    # back, which is right for a test and wrong for a demo --
                    # a window that opens and is replaced a fifth of a second
                    # later is not something anybody can watch. The guest has
                    # no `sleep` verb to abuse for this, and adding one would
                    # put a do-nothing applet in the grammar the model decodes
                    # against, which is a real cost for a presentational
                    # problem. So the pause lives in the harness.
                    if line.startswith("@wait "):
                        try:
                            secs = float(line.split(None, 1)[1])
                        except (IndexError, ValueError):
                            print(f"[drive] bad beat: {line}", file=sys.stderr)
                            continue
                        print(f"[drive] beat: {secs}s")
                        time.sleep(secs)
                        # **The next line is sent because the beat is over,
                        # not because a prompt came back.**
                        #
                        # This used to leave `buf` alone, on the reasoning that
                        # the guest is idle during a beat and will print
                        # nothing further, so the prompt already sitting in
                        # there would match on the next pass. That was true of
                        # every guest this harness had driven and stopped being
                        # true the moment a background task learned to report
                        # something: the compositor watchdog prints from the
                        # clock task during exactly the pause a beat creates,
                        # those bytes land behind the prompt, and the match
                        # below can never fire again. Every remaining command
                        # then goes unsent, which reads as a guest that hung
                        # rather than as a harness that stopped asking.
                        #
                        # Standing a fresh prompt in `buf` does not fix it
                        # either, for the same reason one pass later -- the
                        # alarm is still in the socket and arrives on top of
                        # it. So the beat stops consulting the buffer at all.
                        # It already knows what a prompt would have told it:
                        # the guest was at one when the wait began, and the
                        # wait is what the pause was for.
                        force_send = True
                        continue
                    force_send = False
                    print(f"[drive] sent: {line}")
                    sock.sendall(line.encode() + b"\r")
                    buf.clear()
                    # The wire loses whole commands intermittently -- bytes
                    # arrive at the guest UART and never come out, more often
                    # after a long silent boot. The guest's own Enter-echo is
                    # the acknowledgement: if it has not appeared within 8s,
                    # the command is gone, and resending is the only honest
                    # recovery. Capped, and reset by any fresh output.
                    pending = {"line": line, "at": time.time(),
                               "retries": 0, "mark": len(buf)}
                    # A recording that must start *during* a command rather
                    # than after it. Blocking, deliberately: the guest is busy
                    # holding the screen and there is nothing on the serial
                    # line to miss, and a recorder racing the reader for the
                    # monitor would tear frames.
                    if rec_dir and rec_after > 0 and not queue:
                        time.sleep(rec_after)
                        record(rec_dir, rec_frames, rec_gap)
                        rec_dir = None
                    time.sleep(0.2)
                else:
                    idle_prompts += 1
                    sent_all = True
                    if idle_prompts >= 2:
                        if mouse:
                            monitor(mouse)
                            # Keep reading afterwards. Anything the guest says
                            # in response to a pointer event is emitted after
                            # the last prompt, and the loop that reads the
                            # socket has already decided it is done -- so
                            # without this the bytes sit in the buffer and the
                            # socket closes on top of them. That cost a run
                            # whose whole purpose was a trace, and read as the
                            # interrupt never firing.
                            deadline2 = time.time() + 2.0
                            while time.time() < deadline2:
                                try:
                                    extra = sock.recv(4096)
                                except socket.timeout:
                                    continue
                                except OSError:
                                    break
                                if extra:
                                    emit(extra)
                        if rec_dir:
                            record(rec_dir, rec_frames, rec_gap)
                        if shot:
                            capture(shot)
                            shot = None
                        break
                    time.sleep(0.5)
    finally:
        # A screenshot that was asked for is taken on every exit path, not
        # only the tidy one. It used to hang off the idle branch alone, so a
        # session that ran to its timeout -- which is most of the interesting
        # ones -- threw away the only evidence about what was on screen, and
        # the guest was gone by the time anybody noticed.
        if shot:
            try:
                capture(shot)
            except Exception as e:
                print(f"[drive] screenshot failed: {e}", file=sys.stderr)
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
        # A guest crash usually leaves its cause in QEMU's own log -- a `-d
        # int` trace carries the vector and RIP of every exception. Discarding
        # stderr here is how a fault stays mysterious.
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        if err.strip():
            Path(".qemu/qemu-stderr.log").write_text(err, encoding="utf-8")
            print(f"[drive] qemu stderr ({len(err)} bytes) -> .qemu/qemu-stderr.log",
                  file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

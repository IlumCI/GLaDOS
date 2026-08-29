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

Each positional argument is one shell line. With none, it just captures the
boot log and exits at the first prompt.

VVFAT cannot hold more than ~500 MB, which excludes the real Qwen3.5
checkpoint. `--stage-iso` assembles the same ESP tree into a FAT32 image,
wraps it El Torito via tools/mkiso.py, and boots off -cdrom, which has no
size cap; guest RAM still has to cover the weights, so raise --memory.
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
    print(f"[drive] screenshot {dest} ({w}x{h})")


def main():
    # The serial stream is UTF-8, and a multibyte character split across a
    # socket read decodes to U+FFFD under errors="replace" -- which the
    # default cp1252 stdout then refuses to encode when output is redirected
    # to a file, killing the session mid-run.
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
    for src, dst in [
        (model_src, "model.bin"),
        (tokenizer_src, "tokenizer.bin"),
        (ROOT / "esp/GLADOS/roots.der", "roots.der"),
    ]:
        if not src.exists():
            raise SystemExit(f"missing {src}")
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

    args = [
        find_qemu(),
        "-machine", "q35",
        "-m", memory,
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
          ["-drive", f"format=raw,file=fat:rw:{esp}"]),
        "-drive", f"file={ROOT / '.qemu/nvme.img'},if=none,id=nvm0,format=raw",
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
        "-display", "none",
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
    pending = None

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
            if buf.endswith(PROMPT) or (not chunk and buf.rstrip().endswith(PROMPT.strip())):
                if queue:
                    line = queue.pop(0)
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
                                    sys.stdout.write(extra.decode("utf-8", "replace"))
                                    sys.stdout.flush()
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

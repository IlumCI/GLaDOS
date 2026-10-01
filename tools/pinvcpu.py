#!/usr/bin/env python3
"""Pin each QEMU vCPU thread to the host CPU of the same index.

**Why this exists.** No hypervisor reports CPUID leaf 0x1A to a guest, and none
should: a vCPU has no core type, because the host scheduler moves it between a
performance core and an efficiency core whenever it likes. So a hybrid-aware
placement cannot be measured under QEMU at all -- the premise it rests on is
false while the vCPUs float.

Pinning makes it true. One vCPU thread to one host CPU, permanently, and guest
core N is host CPU N for the life of the run. The guest still cannot *read* which
kind it got, which is what `mine cores <mask>` is for: the operator supplies what
the machine cannot see.

Reads the mapping from the monitor rather than assuming it. QEMU creates vCPU
threads in index order, so `CPU #N` and the Nth thread agree in practice -- but
"in practice" is how a measurement ends up describing the wrong core, and
`info cpus` states it.

Usage:  python tools/pinvcpu.py [--port 45455] [--wait 60]
"""
import argparse
import re
import socket
import subprocess
import sys
import time


def monitor_text(port, command, timeout=3.0):
    """One round trip, returning whatever the monitor said."""
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as mon:
        mon.settimeout(timeout)
        # The banner, which arrives unprompted and is not the answer.
        try:
            mon.recv(65536)
        except OSError:
            pass
        mon.sendall((command + "\n").encode())
        out = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                chunk = mon.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
            # The monitor prompt is how a reply ends.
            if b"(qemu)" in out[-16:]:
                break
        return out.decode(errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=45455)
    ap.add_argument("--wait", type=float, default=60.0,
                    help="seconds to keep trying while QEMU starts")
    args = ap.parse_args()

    deadline = time.time() + args.wait
    text = None
    while time.time() < deadline:
        try:
            text = monitor_text(args.port, "info cpus")
            if "CPU #" in text:
                break
        except OSError:
            pass
        time.sleep(1.0)
    if not text or "CPU #" not in text:
        print("no monitor, or it never listed any cpus", file=sys.stderr)
        return 1

    # `* CPU #0: ... thread_id=12345`, one line each. The star marks the
    # current cpu and carries no meaning here.
    pairs = []
    for line in text.splitlines():
        m = re.search(r"CPU #(\d+)", line)
        t = re.search(r"thread_id=(\d+)", line)
        if m and t:
            pairs.append((int(m.group(1)), int(t.group(1))))
    if not pairs:
        print("cpus listed but no thread_id -- is this a KVM run?", file=sys.stderr)
        print(text, file=sys.stderr)
        return 1

    ok = 0
    for cpu, tid in sorted(pairs):
        r = subprocess.run(["taskset", "-pc", str(cpu), str(tid)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            ok += 1
        else:
            print(f"  cpu {cpu} tid {tid}: {r.stderr.strip()}", file=sys.stderr)
    print(f"pinned {ok} of {len(pairs)} vcpu thread(s), guest core N -> host cpu N")
    return 0 if ok == len(pairs) else 1


if __name__ == "__main__":
    sys.exit(main())

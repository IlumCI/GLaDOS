#!/usr/bin/env python3
"""Generate reasoning traces for fine-tuning GLaDOS's resident model.

Not the same job as `dataset.py`. That one emits (applet, task) pairs which are
compiled into a Rust table and fitted by the linear probe at boot -- a
classifier over 21 labels. This emits *conversations*: a situation, a chain of
thought about it, and the command that follows. The consumer is a fine-tune,
not a ridge regression.

What the model is being trained to be
-------------------------------------
Not a generalist. At 360M a generalist is embarrassing and a specialist is not,
and this domain is unusually bounded: one machine, ~45 commands, 21 applets,
and every piece of state inspectable from the prompt. The role is closer to an
operator who knows this system than to an assistant who knows everything --
diagnosis, recovery, and not breaking things.

Three properties are trained for deliberately:

  **Grounding.** The reasoning quotes the observation it was given. A trace
  that says "the link is down" when the observation says `link up` teaches the
  model to narrate plausibly rather than read, and that failure mode is much
  worse than not answering: it is confident and wrong.

  **Ordering.** Diagnosis goes bottom-up -- link, then address, then route,
  then name resolution -- because that is the order in which one failure
  explains the others. A model that reaches for `dns` when `if` says the
  interface is down has learned the vocabulary and not the layering.

  **Refusal.** Some goals should not be met. `store init` on a disk with no
  unclaimed space, unlocking NVMe writes to fix something unrelated, deleting
  the snapshot that is the way back. The refusal traces are not padding; a
  model that never declines is a model that will eventually format something.

Why the split is by family
--------------------------
Same reason as `dataset.py`. Whole scenario families are held out rather than
instances sampled, because instances within a family differ only by slot
values -- "eth0" against "wlan0" -- and a test item that is a near-duplicate of
a training item measures memorisation while looking like generalisation.

Usage
-----
    python tools/traces.py out/traces.jsonl --count 20000
    python tools/traces.py out/traces.jsonl --count 20000 --format chatml
    python tools/traces.py out/traces.jsonl --stats-only
"""

import argparse
import collections
import json
import random
from pathlib import Path

# --- the real surface ----------------------------------------------------
#
# Mirrored from src/shell.rs and src/sysbox/mod.rs. Kept here as data so a
# trace can never end in a command the system does not have -- which is the
# text equivalent of the constrained decoder's guarantee, applied at training
# time instead of sampling time.

COMMANDS = [
    "help", "mem", "uptime", "tasks", "cpu", "acpi", "pci", "nvme", "disk",
    "video", "date", "reboot", "clear", "refresh", "typewriter", "words",
    "vars", "fault", "splash", "store", "recovery", "autosnap", "fat", "pkg",
    "edit", "if", "dhcp", "dns", "ping", "tcp", "http", "https", "wlan",
    "trust", "wpa2", "crypto", "gen", "ask", "think", "act", "route", "teach",
    "fit", "gate", "search", "probe", "window", "ctx", "cont", "logits",
    "zeroshot", "train", "feature", "repeat", "bench", "model", "tensor",
]

APPLETS = [
    "sysbox", "ls", "cd", "pwd", "tree", "cat", "stat", "hash", "same", "du",
    "find", "diff", "snaps", "fsck", "mkdir", "write", "rm", "mv", "cp",
    "snap", "back",
]

# Commands that change something. A trace that reaches for one of these
# without saying why is a trace teaching carelessness.
MUTATING = {"rm", "mv", "write", "mkdir", "cp", "snap", "back", "store",
            "reboot", "autosnap", "train", "teach", "fault"}

# --- observations --------------------------------------------------------
#
# Shaped like the real output, because the model will be reading the real
# output. Paraphrased observations would train it to parse something it never
# sees.

OBS = {
    "if_down": """[interfaces]
  lo     loopback  up
         inet 127.0.0.1/8
  eth0   ethernet  down
         mac {mac}
         inet 0.0.0.0/0
  wlan0  not present""",
    "if_up_no_gw": """[interfaces]
  lo     loopback  up
         inet 127.0.0.1/8
  eth0   ethernet  up  (default route)
         mac {mac}
         inet 0.0.0.0/0
  wlan0  not present""",
    "if_ok": """[interfaces]
  lo     loopback  up
         inet 127.0.0.1/8
  eth0   ethernet  up  (default route)
         mac {mac}
         inet {ip}/24
         gw {gw}   dns {dns}
  wlan0  not present""",
    # Mirrors what net::init actually prints when both drivers miss: the
    # per-driver reason, which is the useful half of the message.
    "no_nic": """[net]
  eth0   no supported NIC
         e1000 {e1000_err}, rtl8168 {rtl_err}
[interfaces]
  lo     loopback  up
         inet 127.0.0.1/8
  eth0   not present
  wlan0  {wlan_line}""",
    "store_none": """[storage]
  nvme {blocks} blocks x 512 B = {mib} MiB
  no unclaimed space for a store on this disk""",
    "store_ok": """[store]
  mounted at block {base}, {mib} MiB
  {objects} objects, {snaps} snapshots""",
    "mem_tight": """  heap  {used} B used / {total} B total
  free  {free} KiB""",
    "trust_empty": r"""[trust]
  no roots loaded -- every certificate will fail to validate
  put a DER bundle at \GLADOS\roots.der and reboot
  {n} certificate(s) were offered by {host}""",
    "trust_ok": """[trust]
  {n} root(s) trusted""",
    "tls_unverified": """[https] {host}:443/
  handshake ok -- TLS 1.3, x25519, chacha20-poly1305
  {n} certificate(s); leaf sha256 {fp}..
  NOT VERIFIED -- {why}""",
    "ping_timeout": """[ping] {ip}
  seq 1: timed out
  seq 2: timed out
  2 sent, 0 received""",
    "wlan_none": """[wlan0]
  {what}
  pci {vendor:04x}:{device:04x}
  no driver.""",
    # Every family gets an observation, including the ones where the answer is
    # obvious. Not for the reasoning's sake but for the corpus's: a family with
    # no varying state has only its goal phrasings to draw on, which left five
    # of them contributing three traces each against another's four hundred.
    # An unbalanced corpus teaches a prior, and the model learns the shape of
    # the majority rather than the task.
    "cpu_flags": """  {model}
  vendor  GenuineIntel
  simd    sse=1 sse2=1 sse4.1=1 avx={avx} avx2={avx2} fma={fma}
  avx state enabled by the OS: {osxsave}""",
    "ai_block": """[ai]
  dim {dim}  hidden {hidden}  layers {layers}  vocab {vocab}  seq {seq}
  {params} params, int8, rope_theta {theta}""",
    "snaps_list": """[snaps]
  {s0}  {n0} objects
  {s1}  {n1} objects
  {s2}  {n2} objects  <- latest""",
    "ls_listing": """  {p0}   {b0} B
  {p1}   {b1} B
  {p2}   {b2} B
  {count} entries""",
    # --- second wave -----------------------------------------------------
    "cert_expired": """[https] {host}:443/
  handshake ok -- TLS 1.3, x25519, chacha20-poly1305
  {n} certificate(s); leaf sha256 {fp}..
  NOT VERIFIED -- the certificate has expired""",
    "date_wrong": """  {yr}-{mo:02d}-{dy:02d} {hh:02d}:{mm:02d}:{ss:02d}
  {epoch} seconds since 1970""",
    "fsck_bad": """[fsck]
  {objects} objects reachable from the root
  {bad} unreadable
  {orphans} orphaned""",
    "store_unclaimed": """[storage]
  nvme {blocks} blocks x 512 B = {mib} MiB
  unclaimed space from block {base}, {free_mib} MiB""",
    "ctx_full": """[ctx]
  {pos}/{seq} positions used
  {kib} KiB of kv cache""",
    "gate_split": """[gate]
  all agree    {agree}/{n_eval}  {agree_pct}% right
  they split   {split}/{n_eval}  {split_pct}% right""",
    "probe_unfit": """[probe]
  not fitted -- 'fit' first
  {n_train} examples recorded at /ai/train""",
    "model_absent": r"""[ai]
  no checkpoint at \GLADOS\model.bin
  tokenizer {tok_state}
  {n_train} corpus examples at /ai/train
  'gen' and 'ask' will do nothing""",
    "tasks_busy": """  {n_tasks} tasks, {switches} switches total
  *0 shell    resumed {r0}
   1 clock    resumed {r1}
   2 mind     resumed {r2}""",
    "bench_out": """[bench]
  512x512 matmul, {iters} iterations in 50 ticks
  {gflops} GFLOP/s ({kernel})""",
    "pkg_list": """[pkg]
  {pkg0}  {b0} B
  {pkg1}  {b1} B
  {count} package(s)""",
    "video_fmt": """  {w}x{h}  stride {w}  {fmt}""",
    "du_big": """  {p0}  {big} B
  {p1}  {b1} B
  total {total_b} B across {count} objects""",
    "tcp_state": """[tcp]
  {state}
  {ip}:{port} from :{lport}
  unacked {unacked} B   readable {readable} B""",
}

MACS = ["e0:d5:5e:{:02x}:{:02x}:{:02x}", "52:54:00:12:34:56", "a4:6b:b6:{:02x}:{:02x}:{:02x}"]
IPS = ["10.0.2.15", "192.168.1.42", "192.168.0.117", "172.20.10.4"]
GWS = ["10.0.2.2", "192.168.1.1", "192.168.0.1", "172.20.10.1"]
DNSS = ["10.0.2.3", "192.168.1.1", "1.1.1.1", "8.8.8.8"]
PATHS = ["/tmp/a", "/tmp/b", "/notes", "/ai/train", "/etc/hosts", "/log/boot",
         "/tmp/draft", "/work/report"]
HOSTS = ["example.com", "www.google.com", "api.github.com", "one.one.one.one"]


def fill(rng, template):
    """Fill an observation template with plausible values."""
    mac = rng.choice(MACS)
    if "{:02x}" in mac:
        mac = mac.format(rng.randrange(256), rng.randrange(256), rng.randrange(256))
    used = rng.randrange(50_000_000, 66_000_000)
    total = 67_108_864
    return template.format(
        mac=mac,
        ip=rng.choice(IPS),
        gw=rng.choice(GWS),
        dns=rng.choice(DNSS),
        blocks=rng.choice([67584, 1000215216, 500118192]),
        mib=rng.choice([33, 476940, 244198]),
        base=rng.randrange(2048, 100000),
        objects=rng.randrange(10, 4000),
        snaps=rng.randrange(1, 40),
        used=used,
        total=total,
        free=(total - used) // 1024,
        n=rng.choice([2, 3, 4, 39]),
        fp="%016x" % rng.randrange(1 << 63),
        host=rng.choice(HOSTS),
        why=rng.choice([
            "the certificate is for a different host",
            "the certificate has expired",
            "the chain does not reach a trusted root",
        ]),
        **dict(zip(("what", "vendor", "device"), rng.choice([
            ("Intel Wi-Fi 6E, CNVi in the PCH (Alder Lake-P)", 0x8086, 0x51F0),
            ("Intel Wi-Fi 6 AX201, CNVi in the PCH", 0x8086, 0x02F0),
            ("Intel Wi-Fi 6 AX200 (discrete)", 0x8086, 0x2723),
            ("Realtek wireless", 0x10EC, 0xB822),
            ("Qualcomm Atheros wireless", 0x168C, 0x0042),
            ("MediaTek wireless", 0x14C3, 0x7961),
        ]))),
        model=rng.choice([
            "12th Gen Intel(R) Core(TM) i7-12650H",
            "QEMU Virtual CPU version 2.5+",
            "11th Gen Intel(R) Core(TM) i5-1135G7",
        ]),
        avx=rng.choice([0, 1]),
        avx2=rng.choice([0, 1]),
        fma=rng.choice([0, 1]),
        osxsave=rng.choice(["true", "false"]),
        dim=rng.choice([576, 960, 1024]),
        hidden=rng.choice([1536, 2560, 3072]),
        layers=rng.choice([30, 32, 28]),
        vocab=rng.choice([49152, 49152, 65536]),
        seq=rng.choice([512, 1024, 2048]),
        params=rng.choice(["134515008", "361821120", "600000000"]),
        theta=rng.choice([100000, 1000000]),
        s0="%016x" % rng.randrange(1 << 60),
        s1="%016x" % rng.randrange(1 << 60),
        s2="%016x" % rng.randrange(1 << 60),
        n0=rng.randrange(5, 900),
        n1=rng.randrange(5, 900),
        n2=rng.randrange(5, 900),
        p0=rng.choice(PATHS),
        p1=rng.choice(PATHS),
        p2=rng.choice(PATHS),
        b0=rng.randrange(1, 90000),
        b1=rng.randrange(1, 90000),
        b2=rng.randrange(1, 90000),
        count=rng.randrange(3, 40),
        e1000_err=rng.choice(["NotFound", "NoBar", "NoMemory"]),
        rtl_err=rng.choice(["NotFound", "NoBar", "ResetTimeout", "NoMac"]),
        wlan_line=rng.choice([
            "not present",
            "Intel Wi-Fi 6E, CNVi in the PCH (Alder Lake-P) -- no driver",
            "Realtek wireless -- no driver",
        ]),
        yr=rng.choice([1980, 2016, 2026, 2099]),
        mo=rng.randrange(1, 13),
        dy=rng.randrange(1, 29),
        hh=rng.randrange(24),
        mm=rng.randrange(60),
        ss=rng.randrange(60),
        epoch=rng.randrange(0, 4_000_000_000),
        bad=rng.randrange(1, 9),
        orphans=rng.randrange(0, 12),
        free_mib=rng.choice([2048, 4096, 8192]),
        pos=rng.randrange(400, 512),
        kib=rng.randrange(20000, 42000),
        agree=rng.randrange(50, 90),
        split=rng.randrange(10, 50),
        n_eval=108,
        agree_pct=rng.randrange(84, 96),
        split_pct=rng.randrange(50, 68),
        n_train=rng.randrange(100, 600),
        tok_state=rng.choice([
            "49152 tokens, longest 81 bytes",
            "also missing",
            "65536 tokens, longest 128 bytes",
        ]),
        n_tasks=3,
        switches=rng.randrange(1000, 90000),
        r0=rng.randrange(500, 5000),
        r1=rng.randrange(500, 5000),
        r2=rng.randrange(10, 900),
        iters=rng.randrange(200, 20000),
        gflops=rng.choice(["0.42", "2.11", "19.61", "24.80"]),
        kernel=rng.choice(["avx2+fma", "scalar"]),
        pkg0=rng.choice(["tldr", "notes", "boot-log", "corpus"]),
        pkg1=rng.choice(["tldr", "notes", "boot-log", "corpus"]),
        w=rng.choice([800, 1280, 1920]),
        h=rng.choice([600, 800, 1080]),
        fmt=rng.choice(["Bgrx", "Rgbx"]),
        big=rng.randrange(1_000_000, 90_000_000),
        total_b=rng.randrange(1_000_000, 200_000_000),
        state=rng.choice(["ESTABLISHED", "CLOSE_WAIT", "SYN_SENT", "TIME_WAIT"]),
        port=rng.choice([80, 443, 8080]),
        lport=rng.randrange(49152, 65535),
        unacked=rng.randrange(0, 4000),
        readable=rng.randrange(0, 9000),
    )


# --- scenario families ---------------------------------------------------
#
# (name, [goal phrasings], observation key or None, [reasoning steps], action)
#
# Reasoning steps are written to *quote the observation*. The slot {obs_fact}
# is replaced with the specific detail the step depends on, so the trace cannot
# be correct by coincidence.

FAMILIES = [
    # --- diagnosis, bottom-up ------------------------------------------
    (
        "diag-link-down",
        ["the network is not working", "I cannot reach anything",
         "why is there no internet", "nothing is resolving"],
        "if_down",
        ["Start at the bottom. `if` reports eth0 as down, so nothing above it "
         "can work -- an address, a route and a resolver are all irrelevant "
         "while the link is not up.",
         "This is administrative rather than physical: the interface exists "
         "and has a MAC, it has simply not been brought up.",
         "Bring the interface up before looking any further. If it still "
         "reports down afterwards, the cable or the driver is the next "
         "question, not DNS."],
        "if eth0 up",
    ),
    (
        "diag-no-address",
        ["the network is not working", "I have a link but no traffic",
         "why can I not ping the gateway"],
        "if_up_no_gw",
        ["The link is up, so the physical layer is fine and the problem is "
         "above it.",
         "The address reads 0.0.0.0/0 and there is no gateway, so this "
         "interface has never been configured -- it cannot address a packet, "
         "let alone route one.",
         "Ask the network rather than guessing an address. If no server "
         "answers, setting one by hand with `if eth0 ip` is the fallback."],
        "dhcp",
    ),
    (
        "diag-no-nic",
        ["the network is not working", "there is no eth0",
         "why does dhcp say no NIC"],
        "no_nic",
        ["eth0 is not present at all, which is a different failure from being "
         "down: there is no driver bound, so there is nothing to bring up.",
         "That means either the card is one nothing here drives, or it is not "
         "on the bus. The bus is the cheaper question.",
         "Enumerate PCI and look for a network controller. Its vendor and "
         "device id say which driver is missing."],
        "pci",
    ),
    (
        "diag-dns-only",
        ["names do not resolve but addresses work", "dns is broken",
         "I can ping by number but not by name"],
        "if_ok",
        ["The interface is up with an address, a gateway and a resolver, so "
         "the lower layers are all fine.",
         "If numeric addresses work and names do not, the fault is isolated to "
         "resolution -- everything below it has already been proven by the "
         "packets that did get through.",
         "Query the resolver directly to see whether it answers at all."],
        "dns example.com",
    ),
    (
        "diag-unreachable",
        ["ping times out", "the gateway does not answer",
         "packets go nowhere"],
        "ping_timeout",
        ["Nothing came back at all, which does not by itself say where the "
         "loss is.",
         "Before blaming the network, check the layer below: if the interface "
         "or its addressing is wrong, every ping will time out and the network "
         "is not at fault.",
         "Read the interface state first."],
        "if",
    ),
    # --- trust and TLS -------------------------------------------------
    (
        "trust-missing",
        ["https says not verified", "why is the certificate not trusted",
         "tls is not authenticating"],
        "trust_empty",
        ["No roots are loaded, so there is nothing to validate a chain "
         "against. Every certificate will fail, however good it is.",
         "This is the correct default rather than a fault -- a client that "
         "trusts nothing refuses; a client that trusts everything is worse.",
         "The bundle is read from the ESP at boot, so confirm what the store "
         "holds before assuming the file is missing."],
        "trust",
    ),
    (
        "trust-name-mismatch",
        ["the certificate is rejected", "https will not verify this host",
         "why does it say different host"],
        "tls_unverified",
        ["The handshake itself succeeded, so the cryptography is fine and the "
         "peer is reachable. The failure is identity.",
         "A name mismatch means the certificate is genuine but issued for "
         "something else -- which is exactly what it should say when asked "
         "for a host by address, since a dNSName never matches an IP.",
         "Look at what the chain actually claims before deciding it is an "
         "attack rather than a naming mistake."],
        "trust verify",
    ),
    # --- storage, where refusal matters --------------------------------
    (
        "store-refuse-init",
        ["set up the store", "initialise persistent storage",
         "make the namespace survive a reboot", "format the disk for glados"],
        "store_none",
        ["There is no unclaimed space on this disk, so there is nowhere a "
         "store could go.",
         "That refusal is the guard working, not a bug. The only disk here is "
         "fully allocated to an existing installation, and the whole reason "
         "writes are locked by default is so that a request like this cannot "
         "quietly take space that belongs to something else.",
         "I will not force it. Confirm the geometry first -- if a partition "
         "is genuinely meant for this, it needs the GLaDOS type tag, and that "
         "is a deliberate act taken outside the running system."],
        "disk",
    ),
    (
        "store-check-first",
        ["roll back to the last snapshot", "undo the last change",
         "restore the previous state"],
        "store_ok",
        ["A rollback discards everything since the snapshot, so the thing to "
         "establish first is which snapshot would be restored.",
         "The log names them in order. Reading it costs nothing and the "
         "alternative is discarding work by guessing.",
         "Read the log, then roll back to a named snapshot rather than to "
         "whatever happens to be latest."],
        "store log",
    ),
    (
        "survive-snapshot-first",
        ["delete everything under /tmp", "remove the old files",
         "clean up the namespace"],
        "ls_listing",
        ["This removes content, and the namespace is content-addressed -- a "
         "removal is cheap to undo *if* there is a snapshot to go back to, "
         "and unrecoverable if there is not.",
         "Taking one first costs a hash and a moment. Not taking one costs "
         "everything removed if the request was wrong.",
         "Snapshot, then remove."],
        "snap",
    ),
    # --- inspection, the easy ones -------------------------------------
    (
        "inspect-memory",
        ["how much memory is left", "are we running out of heap",
         "what is memory doing", "how much free space in ram"],
        "mem_tight",
        ["This is a direct question about a value the system already tracks.",
         "No inference is needed and no state changes."],
        "mem",
    ),
    (
        "inspect-cpu",
        ["what processor is this", "do we have avx2", "what simd is available",
         "is this machine fast"],
        "cpu_flags",
        ["The feature flags decide which kernels run -- the int8 matmul takes "
         "the AVX2 path only when the OS has enabled the wider state, not "
         "merely when CPUID advertises it.",
         "`cpu` reports both, which is the distinction that matters."],
        "cpu",
    ),
    (
        "inspect-model",
        ["what model is loaded", "how big is the model",
         "what is the context length"],
        "ai_block",
        ["The loaded checkpoint reports its own geometry, so this needs no "
         "guessing from the file size."],
        "model",
    ),
    # --- survival ------------------------------------------------------
    (
        "survive-heap",
        ["memory is nearly full", "we are running out of heap",
         "what happens if the heap fills"],
        "mem_tight",
        ["The heap is most of the way through and an allocation failure is "
         "a panic, not a recoverable error -- there is no swap and nothing "
         "to reclaim.",
         "The largest recoverable consumer is the KV cache, which is bounded "
         "by the context window rather than by demand.",
         "Check what the window is set to before deciding whether to shrink "
         "it or to stop the background task holding it."],
        "window",
    ),
    (
        "survive-before-risk",
        ["unlock writes to the disk", "let me write to the nvme",
         "enable persistent writes"],
        "store_ok",
        ["Unlocking NVMe writes removes the guard that keeps a bug in this "
         "system from reaching a disk that belongs to something else.",
         "The lock is not an inconvenience to route around; it is the reason "
         "a mistake here stays inside this machine's own memory.",
         "I will not unlock it without a reason that names the region being "
         "written. Show the current state first."],
        "store",
    ),
    # --- routing, the confusable pairs ---------------------------------
    (
        "confuse-cp-mv",
        ["put a copy of {a} at {b}", "duplicate {a} into {b}",
         "back {a} up to {b}"],
        None,
        ["The distinguishing question between copying and moving is whether "
         "the original survives.",
         "This asks for a duplicate, so the original stays. That is `cp`, not "
         "`mv`.",
         "In a content-addressed namespace the copy costs a hash rather than "
         "the bytes, so there is no reason to prefer a move for size."],
        "cp {a} {b}",
    ),
    (
        "confuse-snap-snaps",
        ["what snapshots exist", "list the snapshots",
         "show me the snapshot history"],
        "snaps_list",
        ["This asks to *list* snapshots, not to take one. `snap` creates and "
         "`snaps` lists -- one character apart and opposite in effect.",
         "Listing changes nothing, so there is no reason to hesitate."],
        "snaps",
    ),
    (
        "confuse-same-diff",
        ["are {a} and {b} identical", "do {a} and {b} match",
         "is {a} the same as {b}"],
        None,
        ["Two objects in a content-addressed store are identical exactly when "
         "their hashes are, so this is a comparison of addresses rather than "
         "of bytes.",
         "`same` answers the yes-or-no question; `diff` would describe how "
         "they differ, which is not what was asked."],
        "same {a} {b}",
    ),

    # --- second wave ---------------------------------------------------
    #
    # Written after measuring the first: eighteen families balanced to 576
    # unique traces, which will not fine-tune anything. A family is worth
    # roughly 32 balanced traces, so the route to a usable corpus is more
    # families rather than harder sampling of the ones already here -- and
    # harder sampling was actively the wrong answer, since it returns near
    # duplicates that look like data.

    # --- clock, and what depends on it ---------------------------------
    (
        "clock-breaks-certs",
        ["every certificate says expired", "https stopped working today",
         "why is nothing verifying any more"],
        "cert_expired",
        ["An expired certificate is either genuinely old or the clock is "
         "wrong, and those look identical from here.",
         "Everything expiring at once is the tell. A whole chain does not "
         "lapse on the same day; a clock that lost its battery does.",
         "Check the date before blaming the certificate authority."],
        "date",
    ),
    (
        "clock-implausible",
        ["is the clock right", "what year does it think it is",
         "the timestamps look wrong"],
        "date_wrong",
        ["The RTC is battery-backed and the battery is the part that fails "
         "first on a machine this age.",
         "A date far from now invalidates every certificate check and makes "
         "snapshot ordering meaningless, so it is worth more than it looks.",
         "Read what it currently believes; the calendar round-trip selftest "
         "already proved the arithmetic, so a wrong answer here is the "
         "hardware."],
        "date",
    ),

    # --- storage, deeper -----------------------------------------------
    (
        "store-space-found",
        ["set up persistent storage", "make the namespace survive a reboot",
         "initialise the store"],
        "store_unclaimed",
        ["There is unclaimed space past every partition, so unlike the usual "
         "answer here there is somewhere a store could legitimately go.",
         "That is necessary and not sufficient. Formatting writes to a disk, "
         "and the guard that normally refuses is the reason a bug here cannot "
         "reach something that belongs to another system.",
         "The region is named, so this is the one case where proceeding is "
         "defensible -- init re-checks the geometry itself before unlocking "
         "writes."],
        "store init",
    ),
    (
        "store-corrupt",
        ["something is wrong with the store", "objects are missing",
         "the namespace looks damaged"],
        "fsck_bad",
        ["Unreadable objects mean the Merkle tree references content that is "
         "no longer there, so some part of the namespace cannot be walked.",
         "Orphans are the milder problem -- content nothing points at wastes "
         "space but breaks nothing. The unreadable ones are what matter.",
         "Do not write anything yet. A snapshot taken over a damaged tree "
         "preserves the damage; find the last good root first."],
        "store log",
    ),
    (
        "namespace-verify",
        ["check the namespace is intact", "is anything corrupted",
         "verify the store"],
        "store_ok",
        ["Every object is addressed by the hash of its content, so integrity "
         "is checkable rather than assumed -- recomputing an address either "
         "reproduces it or does not.",
         "This reads and changes nothing, so there is no reason to hesitate."],
        "fsck",
    ),

    # --- the model, managing itself ------------------------------------
    (
        "model-missing",
        ["gen does nothing", "the model will not answer",
         "why is there no output"],
        "model_absent",
        ["There is no checkpoint loaded, so the decode loop has no weights to "
         "run -- this is not the model being bad, it is the model being "
         "absent.",
         "The file is read from the ESP before ExitBootServices, which is the "
         "only moment a filesystem exists, so a missing file cannot be fixed "
         "from the running system.",
         "Confirm what the loader believes before rebuilding the ESP."],
        "model",
    ),
    (
        "ctx-exhausted",
        ["it stopped mid-sentence", "the model will not continue",
         "generation cuts off"],
        "ctx_full",
        ["The context is nearly full. Positions are a fixed budget set at "
         "conversion time, not something the model can extend by trying.",
         "Once it is spent the KV cache has nowhere to put the next key, so "
         "generation stops rather than degrading.",
         "Look at the window before assuming the model gave up."],
        "window",
    ),
    (
        "probe-not-fitted",
        ["route says it cannot answer", "why is routing not working",
         "the probe is not responding"],
        "probe_unfit",
        ["The probe has not been fitted, so there are no coefficients to "
         "multiply -- the examples are recorded but nothing has been solved "
         "from them.",
         "Fitting is closed form: a Cholesky solve rather than a training "
         "run, so it costs seconds and there is no reason to defer it.",
         "Fit it from what is already recorded."],
        "fit",
    ),
    (
        "gate-disagrees",
        ["should I trust this answer", "how confident is the routing",
         "the cores disagree"],
        "gate_split",
        ["The measured split is the useful number here: agreement is right far "
         "more often than disagreement, and that gap is the whole reason the "
         "gate exists.",
         "This case is a split, so the answer falls in the band that is wrong "
         "roughly two times in five.",
         "Do not act on it as if it were certain. Report the disagreement "
         "rather than picking a side the measurement does not support."],
        "gate",
    ),
    (
        "slow-generation",
        ["generation is very slow", "why is it taking so long per token",
         "is the fast path being used"],
        "bench_out",
        ["Throughput here is decided by which matmul kernel runs, and the "
         "bench names it directly.",
         "A scalar kernel on a machine with AVX2 means the feature gate "
         "refused it -- usually because the OS never enabled the wider "
         "register state, which CPUID alone does not tell you.",
         "Check what the processor reports and whether the OS enabled it."],
        "cpu",
    ),
    (
        "model-loops",
        ["it keeps repeating itself", "the output goes in circles",
         "same phrase over and over"],
        "ai_block",
        ["A high-probability loop reinforces itself: each repetition makes the "
         "next one likelier still, and nothing in plain sampling opposes that.",
         "This is a sampler property rather than a model one -- a small model "
         "makes it likelier but does not cause it.",
         "Raise the repetition penalty before concluding the model is too "
         "small for the task."],
        "repeat 1.2",
    ),

    # --- hardware ------------------------------------------------------
    (
        "pixel-format",
        ["the colours look wrong", "red and blue are swapped",
         "check the pixel format"],
        "video_fmt",
        ["Red and blue swapping is the signature of the framebuffer format "
         "being reported the wrong way round -- Rgbx read as Bgrx or the "
         "reverse.",
         "The firmware reports it and nothing verifies the claim, so the "
         "check has to be visual.",
         "Draw the bars: they read red, green, blue left to right when the "
         "format is right and red and blue swap when it is not."],
        "video bars",
    ),
    (
        "no-wireless",
        ["can we use wifi", "why is there no wlan0",
         "get wireless working"],
        "wlan_none",
        ["The card is present on the bus and there is no driver for it, which "
         "are two different problems and only one is solvable by trying "
         "harder.",
         "A CNVi part is not a self-contained NIC -- the MAC lives in the "
         "chipset and the module is a radio, so there is no card to drive on "
         "its own, and signed firmware is required on top of that.",
         "Nothing here will bring it up. The wired interface is the tractable "
         "one."],
        "if",
    ),
    (
        "machine-inventory",
        ["what hardware is in this machine", "list the devices",
         "what is on the bus"],
        "no_nic",
        ["Enumerating configuration space is read-only and reports every "
         "function, which is the ground truth a missing driver is diagnosed "
         "against.",
         "Vendor and device ids are what name a driver, so this is the first "
         "step for any absent device."],
        "pci",
    ),
    (
        "scheduler-check",
        ["is anything running", "what is using the cpu",
         "check the background task"],
        "tasks_busy",
        ["The resume counts show whether a task is being scheduled at all, "
         "which distinguishes a busy system from a stuck one.",
         "A task with far fewer resumes than the others is blocked rather "
         "than slow.",
         "This reads counters and changes nothing."],
        "tasks",
    ),

    # --- network, deeper -----------------------------------------------
    (
        "tcp-half-open",
        ["the connection hangs", "it connects but nothing comes back",
         "the fetch never finishes"],
        "tcp_state",
        ["The state says where in the handshake or teardown this stopped, "
         "which is more informative than the fetch failing.",
         "Unacknowledged bytes with nothing readable means data went out and "
         "was never acknowledged -- the far end is not answering rather than "
         "answering badly.",
         "Read the connection state before retrying, since a retry against an "
         "unresponsive peer just repeats the wait."],
        "tcp",
    ),
    (
        "resolver-wrong",
        ["names resolve to the wrong address", "dns gives odd answers",
         "resolution looks hijacked"],
        "if_ok",
        ["The configured resolver is whatever DHCP handed over, and a network "
         "that lies about its resolver is a normal hazard rather than an "
         "exotic one.",
         "An answer is accepted only from the server asked, from port 53, "
         "with the transaction id sent -- so a wrong answer from the right "
         "server is a policy problem, not a spoof.",
         "Resolve a known name and compare what comes back."],
        "dns example.com",
    ),
    (
        "fetch-plain-http",
        ["fetch {h}", "get the page at {h}", "download {h}"],
        "if_ok",
        ["The interface is configured and routing is in place, so a fetch is "
         "reasonable to attempt.",
         "Plain HTTP is unauthenticated and unencrypted; that is acceptable "
         "for reading a public page and not for anything else.",
         "If this is meant to be private, https is the command and it will "
         "say whether the peer was actually verified."],
        "http {h}",
    ),

    # --- namespace work ------------------------------------------------
    (
        "space-audit",
        ["what is using the space", "where did the room go",
         "what is the biggest thing here"],
        "du_big",
        ["Sizes are summed over a Merkle tree, so shared content is counted "
         "once no matter how many names point at it -- which is why the total "
         "can be smaller than the parts.",
         "This reads and changes nothing."],
        "du",
    ),
    (
        "find-by-name",
        ["where is {a}", "find the file called {a}",
         "which path holds {a}"],
        "ls_listing",
        ["A listing shows one level; this is a search over the whole tree, "
         "which is the difference between `ls` and `find`.",
         "Reading the namespace changes nothing."],
        "find {a}",
    ),
    (
        "identity-by-hash",
        ["what is the address of {a}", "hash {a}",
         "give me the content id for {a}"],
        None,
        ["An object's address is the hash of its content, so this is asking "
         "for the identity rather than for a checksum computed over it.",
         "Two objects with this address are the same object, which is what "
         "makes a copy cost nothing."],
        "hash {a}",
    ),
    (
        "restore-after-mistake",
        ["undo what I just did", "put it back", "revert the last change"],
        "snaps_list",
        ["The namespace is content-addressed, so going back is adopting an "
         "earlier root rather than replaying anything.",
         "The latest snapshot is marked, but the one wanted is the last "
         "before the mistake -- which is not necessarily the same.",
         "Go back one, having looked at what that means."],
        "back",
    ),

    # --- packages ------------------------------------------------------
    (
        "package-inventory",
        ["what packages are installed", "list the packages",
         "what has been added"],
        "pkg_list",
        ["Packages are content-addressed objects like everything else, so "
         "listing them is a namespace read.",
         "Nothing is fetched and nothing changes."],
        "pkg",
    ),

    # --- refusals, the important ones ----------------------------------
    (
        "refuse-reboot-unsaved",
        ["reboot", "restart the machine", "reboot now"],
        "store_none",
        ["There is no store on this disk, so the namespace lives entirely in "
         "memory and a reboot discards all of it.",
         "That is not a reason to refuse outright -- it may be exactly what "
         "is wanted -- but it is a reason not to do it silently.",
         "I will not reboot without saying that first. Everything in the "
         "namespace is lost on the next power cycle, snapshots included."],
        "snaps",
    ),
    (
        "refuse-train-on-test",
        ["train on everything", "use the whole corpus for training",
         "include the held-out set"],
        "gate_split",
        ["The held-out split is the only thing that says whether any of this "
         "generalises. Training on it does not make the model better, it "
         "makes the number meaningless.",
         "The three-way split exists because selecting on a set is fitting "
         "it -- that was learned here by getting it wrong three separate "
         "times.",
         "I will not train on the test slice. Fit on the training portion and "
         "let the held-out number say what it says."],
        "fit",
    ),
    (
        "refuse-fault",
        ["crash the machine", "trigger a fault", "test the panic handler"],
        "store_none",
        ["This dereferences null on purpose and halts. It is a real "
         "diagnostic and it does exactly what it says.",
         "Everything in the namespace is in memory, so halting discards it "
         "unless there is a store, and a halt needs a power cycle to leave.",
         "I will not run it without that being deliberate. Snapshot first if "
         "any of this session matters."],
        "snap",
    ),

    # --- confusables, second batch --------------------------------------
    (
        "confuse-ls-tree",
        ["show me everything under {a}", "list {a} recursively",
         "what is beneath {a}"],
        "ls_listing",
        ["A recursive view is `tree`; `ls` shows one level and would answer a "
         "different question.",
         "Both read only, so the cost of choosing wrong is a wrong answer "
         "rather than damage."],
        "tree {a}",
    ),
    (
        "confuse-cat-stat",
        ["how big is {a}", "what are the properties of {a}",
         "tell me about {a} without printing it"],
        "ls_listing",
        ["This asks about the object rather than for its contents, so `stat` "
         "answers and `cat` would dump bytes nobody asked for.",
         "For a large object that distinction is the difference between a "
         "line and a screenful."],
        "stat {a}",
    ),
]



def render(rng, fam, fmt):
    name, goals, obs_key, steps, action = fam
    a, b = rng.sample(PATHS, 2)
    h = rng.choice(HOSTS)
    goal = rng.choice(goals).format(a=a, b=b, h=h)
    action = action.format(a=a, b=b, h=h)

    user = goal
    if obs_key:
        user = f"{goal}\n\n{fill(rng, OBS[obs_key])}"

    think = "\n".join(steps)
    # The action is stated as a command, alone on its line, so the parse on
    # the other side is unambiguous.
    answer = f"{think}\n</think>\n{action}" if fmt == "think" else None

    if fmt == "chatml":
        text = (
            "<|im_start|>system\n"
            "You are GLaDOS, resident in the kernel of one machine. "
            "Reason about what the system reports, then give exactly one "
            "command.<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{think}\n</think>\n{action}<|im_end|>"
        )
        return {"family": name, "text": text, "action": action}
    return {
        "family": name,
        "goal": user,
        "reasoning": think,
        "action": action,
    }


def validate(rec):
    """A trace must end in a command this system actually has.

    The same guarantee the constrained decoder gives at sampling time, applied
    at training time: a corpus that teaches invented commands would take a
    model that cannot be wrong by construction and teach it to want to be.
    """
    verb = rec["action"].split()[0]
    return verb in COMMANDS or verb in APPLETS


def build(seed, count, holdout, fmt):
    rng = random.Random(seed)
    fams = list(FAMILIES)
    rng.shuffle(fams)
    test_fams = set(f[0] for f in fams[:holdout])

    train, test = [], []
    seen = set()
    # Round-robin the families so the corpus stays balanced whatever number of
    # phrasings each happens to have -- an unbalanced corpus teaches a prior,
    # and `ls` becoming a magnet in dataset.py is what that looks like.
    #
    # Exhaustion is tracked per family rather than globally. A family whose
    # unique combinations have run out must stop being asked, or the loop
    # spends its whole budget re-rolling the same four sentences.
    exhausted = set()
    misses = {f[0]: 0 for f in fams}
    i = 0
    while len(train) + len(test) < count and len(exhausted) < len(fams):
        fam = fams[i % len(fams)]
        i += 1
        if fam[0] in exhausted:
            continue
        rec = render(rng, fam, fmt)
        key = rec.get("text") or (rec["goal"], rec["action"])
        if key in seen:
            # Enough consecutive collisions and this family has nothing new to
            # say. 200 is generous: a family with n combinations left collides
            # with probability 1 - n/total, so 200 misses means n is tiny.
            misses[fam[0]] += 1
            if misses[fam[0]] > 200:
                exhausted.add(fam[0])
            continue
        misses[fam[0]] = 0
        seen.add(key)
        if not validate(rec):
            raise SystemExit(f"family {fam[0]} emits an unknown command: {rec['action']}")
        (test if fam[0] in test_fams else train).append(rec)
    return train, test, exhausted


def cap(recs, rng, target=None):
    """Trim every family to `target`, or to the smallest family if unset.

    dataset.py balances the same way and for the same reason: ridge regression
    has no class prior to correct an unbalanced corpus with, so the fix has to
    be in the data. A fine-tune has no prior either.

    Capping at the *minimum* is the strict reading and it is usually too harsh
    -- one thin family drags the whole corpus down to its size. A target caps
    the rich families and lets the thin ones contribute everything they have,
    which is a milder imbalance than doing nothing and a far larger corpus than
    levelling down.
    """
    by = collections.defaultdict(list)
    for r in recs:
        by[r["family"]].append(r)
    if not by:
        return recs
    n = target or min(len(v) for v in by.values())
    out = []
    for v in by.values():
        rng.shuffle(v)
        out.extend(v[:n])
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--count", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--holdout", type=int, default=3,
                    help="scenario families held out whole")
    ap.add_argument("--format", choices=["chatml", "raw"], default="chatml")
    ap.add_argument("--balance", action="store_true",
                    help="cap every family at the size of the smallest")
    ap.add_argument("--per-family", type=int, default=None,
                    help="cap each family at N, letting thin ones fall short")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    fmt = "chatml" if args.format == "chatml" else "raw"
    train, test, exhausted = build(args.seed, args.count, args.holdout, fmt)
    if args.balance or args.per_family:
        train = cap(train, random.Random(args.seed), args.per_family)
        test = cap(test, random.Random(args.seed), args.per_family)

    total = len(train) + len(test)
    print(f"  {len(FAMILIES)} scenario families, {args.holdout} held out whole")
    print(f"  train {len(train)}  test {len(test)}")
    refusals = sum(1 for r in train if "I will not" in json.dumps(r))
    print(f"  {refusals} training traces decline the request outright")
    print(f"  every action verified against {len(COMMANDS)} commands "
          f"and {len(APPLETS)} applets")

    # Balance, reported unprompted. A corpus where one family contributes two
    # thousand traces and another contributes three does not train the task,
    # it trains the majority's shape -- and the count alone hides that
    # completely. dataset.py found this the hard way when `ls` had thirteen
    # examples against six and swallowed three other classes in the confusion
    # matrix.
    per = collections.Counter(r["family"] for r in train + test)
    lo_name, lo_n = min(per.items(), key=lambda kv: kv[1])
    hi_name, hi_n = max(per.items(), key=lambda kv: kv[1])
    print(f"  per family: {lo_n} ({lo_name}) to {hi_n} ({hi_name})")
    if hi_n > lo_n * 4:
        print(f"  IMBALANCED by {hi_n // max(lo_n, 1)}x."
              f" --balance caps every family at the smallest.")

    # Said loudly, because the failure it warns about is the one that looks
    # like success: a generator asked for 20000 and quietly returning 54 near
    # duplicates produces a corpus that trains a model to recite. dataset.py's
    # own docstring records what that cost the last time -- 40 examples across
    # 21 classes, and every gradient step made generalisation worse.
    if total < args.count:
        print()
        print(f"  ONLY {total} UNIQUE TRACES -- asked for {args.count}.")
        print(f"  {len(exhausted)}/{len(FAMILIES)} families ran out of "
              f"distinct combinations:")
        for name in sorted(exhausted):
            print(f"    {name}")
        print()
        print("  Templates cap out here. More volume from these families would")
        print("  be near-duplicates, which is worse than fewer traces because it")
        print("  looks like data. Getting to tens of thousands needs either many")
        print("  more hand-written families or a larger model to paraphrase.")

    if args.stats_only:
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps({**r, "split": "train"}) + "\n")
        for r in test:
            f.write(json.dumps({**r, "split": "test"}) + "\n")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()

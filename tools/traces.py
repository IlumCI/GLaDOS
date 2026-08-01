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
    "no_nic": """[interfaces]
  lo     loopback  up
         inet 127.0.0.1/8
  eth0   not present
  wlan0  not present""",
    "store_none": """[storage]
  nvme {blocks} blocks x 512 B = {mib} MiB
  no unclaimed space for a store on this disk""",
    "store_ok": """[store]
  mounted at block {base}, {mib} MiB
  {objects} objects, {snaps} snapshots""",
    "mem_tight": """  heap  {used} B used / {total} B total
  free  {free} KiB""",
    "trust_empty": """[trust]
  no roots loaded -- every certificate will fail to validate""",
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
}

MACS = ["e0:d5:5e:{:02x}:{:02x}:{:02x}", "52:54:00:12:34:56", "a4:6b:b6:{:02x}:{:02x}:{:02x}"]
IPS = ["10.0.2.15", "192.168.1.42", "192.168.0.117", "172.20.10.4"]
GWS = ["10.0.2.2", "192.168.1.1", "192.168.0.1", "172.20.10.1"]
DNSS = ["10.0.2.3", "192.168.1.1", "1.1.1.1", "8.8.8.8"]
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
        what="Intel Wi-Fi 6E, CNVi in the PCH (Alder Lake-P)",
        vendor=0x8086,
        device=0x51F0,
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
        None,
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
        None,
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
        None,
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
        None,
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
        None,
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
]

PATHS = ["/tmp/a", "/tmp/b", "/notes", "/ai/train", "/etc/hosts", "/log/boot",
         "/tmp/draft", "/work/report"]


def render(rng, fam, fmt):
    name, goals, obs_key, steps, action = fam
    a, b = rng.sample(PATHS, 2)
    goal = rng.choice(goals).format(a=a, b=b)
    action = action.format(a=a, b=b)

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
    # Round-robin the families so the corpus stays balanced no matter how many
    # phrasings each happens to have -- an unbalanced corpus teaches a prior,
    # and `ls` becoming a magnet in dataset.py is what that looks like.
    tries = 0
    while len(train) + len(test) < count and tries < count * 40:
        tries += 1
        fam = fams[(len(train) + len(test)) % len(fams)]
        rec = render(rng, fam, fmt)
        key = rec.get("text") or (rec["goal"], rec["action"])
        if key in seen:
            continue
        seen.add(key)
        if not validate(rec):
            raise SystemExit(f"family {fam[0]} emits an unknown command: {rec['action']}")
        (test if fam[0] in test_fams else train).append(rec)
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--count", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--holdout", type=int, default=3,
                    help="scenario families held out whole")
    ap.add_argument("--format", choices=["chatml", "raw"], default="chatml")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    fmt = "chatml" if args.format == "chatml" else "raw"
    train, test = build(args.seed, args.count, args.holdout, fmt)

    print(f"  {len(FAMILIES)} scenario families, {args.holdout} held out whole")
    print(f"  train {len(train)}  test {len(test)}")
    refusals = sum(1 for r in train if "I will not" in json.dumps(r))
    print(f"  {refusals} training traces decline the request outright")
    print(f"  every action verified against {len(COMMANDS)} commands "
          f"and {len(APPLETS)} applets")

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

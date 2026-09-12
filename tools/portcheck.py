"""The seams, checked rather than intended.

A ported tree reaches this machine through one named seam and through nothing
else. That is the whole value of doing the first port carefully: a port that
reaches into `gfx`, `kbd`, `sysbox` and `time` wherever it happens to need
them is not a port, it is a merge, and the second one starts from nothing.

There are two seams now, and the second is why `TREES` became a mapping.
`src/doom` reaches `crate::port`, which is what a *program* asks of a machine.
`src/wlan` reaches `crate::radio`, which is what an 802.11 stack asks of one --
a different vocabulary with almost no overlap. Widening `crate::port` to cover
both would have put a DMA allocator next to a keyboard scancode table and
called it an interface.

A rule with no check is a habit, and a habit does not survive a long debugging
session at two in the morning. So this scans and fails.

    python tools/portcheck.py                 # every ported tree
    python tools/portcheck.py --tree src/doom
    python tools/portcheck.py --tree src/wlan

There is no `build.rs` in this repository and there cannot be one -- the
machine has no host linker, which `Cargo.toml` records in detail -- so this
runs beside the build rather than inside it, the same arrangement
`tokenizer.py --verify` and `payload.py` already use.

What it looks for, and what it deliberately does not:

  * `crate::x` where `x` is not `port`. The direct reach.
  * `super::super::` climbing out of the tree. The indirect one, which is what
    somebody writes ten minutes after being told about the first.
  * `use crate::...` in any form, including grouped and aliased imports.

It does *not* try to parse Rust. A comment or a string containing `crate::gfx`
will be reported, and that is the right trade: a false positive costs one line
of `# portcheck: ok` and a false negative costs the property the file exists
to protect.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The trees that are ported code, each with the seam it is allowed to name.
# Adding one here is the whole of enrolling it.
#
# It is a mapping rather than a list because the second tree did not want the
# first one's seam. `src/port` is what a *program* asks of this machine -- a
# screen, held keys, a clock, the bytes of a file -- and a wireless stack wants
# none of that and several things it does not have. One global `ALLOWED` would
# have forced the two vocabularies into one module, which is the merge this
# check exists to prevent, arriving through the check itself.
TREES = {
    "src/doom": {"port"},
    "src/wlan": {"radio"},
}

# `crate::` followed by an identifier, and `super::super::` which is the way
# out of a submodule without naming `crate`.
CRATE = re.compile(r"\bcrate::([A-Za-z_][A-Za-z0-9_]*)")
CLIMB = re.compile(r"\bsuper::super::")

# An explicit escape hatch, so a genuine exception is visible in the diff
# rather than achieved by rewording.
WAIVER = re.compile(r"#\s*portcheck:\s*ok")


def scan(path: Path, allowed):
    """Every violation in one file, as (line number, text, why)."""
    out = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return [(0, "", f"unreadable: {e}")]
    for n, line in enumerate(text.splitlines(), 1):
        if WAIVER.search(line):
            continue
        for m in CRATE.finditer(line):
            if m.group(1) not in allowed:
                out.append((n, line.strip(), f"reaches crate::{m.group(1)}"))
        if CLIMB.search(line):
            out.append((n, line.strip(), "climbs out with super::super::"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", action="append", help="override the tree list")
    args = ap.parse_args()

    # An overridden tree that this file does not know about has no declared
    # seam, so it is checked against the union of every seam there is. That is
    # deliberately the lenient direction: `--tree` is a debugging aid, and a
    # false failure there would teach somebody to stop running it.
    if args.tree:
        every = set().union(*TREES.values())
        trees = {t: TREES.get(t, every) for t in args.tree}
    else:
        trees = TREES

    total = 0
    looked = 0

    for t, allowed in trees.items():
        d = ROOT / t
        if not d.is_dir():
            # Not an error. The tree is created by the port that needs it, and
            # this script exists before it does.
            print(f"[portcheck] {t}: not present yet")
            continue
        files = sorted(d.rglob("*.rs"))
        looked += len(files)
        for f in files:
            bad = scan(f, allowed)
            for n, line, why in bad:
                rel = f.relative_to(ROOT).as_posix()
                print(f"{rel}:{n}: {why}")
                print(f"    {line}")
            total += len(bad)

    if total:
        print()
        print(f"[portcheck] {total} violation(s) in {looked} file(s).")
        for t, allowed in trees.items():
            seams = " or ".join(f"crate::{a}" for a in sorted(allowed))
            print(f"[portcheck] {t} may name {seams} and nothing else.")
        print("[portcheck] If one is genuinely justified, append '# portcheck: ok'")
        print("[portcheck] to that line so the exception is in the diff.")
        return 1

    print(f"[portcheck] {looked} file(s) clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

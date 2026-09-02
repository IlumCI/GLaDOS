"""The seam, checked rather than intended.

A ported program reaches this machine through `crate::port` and through
nothing else. That is the whole value of doing the first port carefully: a
port that reaches into `gfx`, `kbd`, `sysbox` and `time` wherever it happens
to need them is not a port, it is a merge, and the second one starts from
nothing.

A rule with no check is a habit, and a habit does not survive a long debugging
session at two in the morning. So this scans and fails.

    python tools/portcheck.py                 # every ported tree
    python tools/portcheck.py --tree src/doom

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

# The trees that are ported code. Adding one here is the whole of enrolling it.
TREES = ["src/doom"]

ALLOWED = {"port"}

# `crate::` followed by an identifier, and `super::super::` which is the way
# out of a submodule without naming `crate`.
CRATE = re.compile(r"\bcrate::([A-Za-z_][A-Za-z0-9_]*)")
CLIMB = re.compile(r"\bsuper::super::")

# An explicit escape hatch, so a genuine exception is visible in the diff
# rather than achieved by rewording.
WAIVER = re.compile(r"#\s*portcheck:\s*ok")


def scan(path: Path):
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
            if m.group(1) not in ALLOWED:
                out.append((n, line.strip(), f"reaches crate::{m.group(1)}"))
        if CLIMB.search(line):
            out.append((n, line.strip(), "climbs out with super::super::"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", action="append", help="override the tree list")
    args = ap.parse_args()

    trees = args.tree or TREES
    total = 0
    looked = 0

    for t in trees:
        d = ROOT / t
        if not d.is_dir():
            # Not an error. The tree is created by the port that needs it, and
            # this script exists before it does.
            print(f"[portcheck] {t}: not present yet")
            continue
        files = sorted(d.rglob("*.rs"))
        looked += len(files)
        for f in files:
            bad = scan(f)
            for n, line, why in bad:
                rel = f.relative_to(ROOT).as_posix()
                print(f"{rel}:{n}: {why}")
                print(f"    {line}")
            total += len(bad)

    if total:
        print()
        print(f"[portcheck] {total} violation(s) in {looked} file(s).")
        print("[portcheck] A ported tree may name crate::port and nothing else.")
        print("[portcheck] If one is genuinely justified, append '# portcheck: ok'")
        print("[portcheck] to that line so the exception is in the diff.")
        return 1

    print(f"[portcheck] {looked} file(s) clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

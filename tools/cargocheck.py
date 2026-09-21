#!/usr/bin/env python3
"""Cargo config keys a nested crate cannot escape.

### The bug this is the check for

`pool/.cargo/config.toml` held `rustflags = []` under a comment saying
"Cleared, not inherited", and `miner/` held the same. **Cargo merges config
files by joining arrays**, so an empty list takes nothing away and the
parent's flags arrive whole. The parent is the kernel's, and it carried
`-C link-arg=/MAP:target/glados.map` -- an MSVC flag that `link.exe` accepts
and `rust-lld -flavor gnu` reads as a filename:

    rust-lld: error: cannot open /MAP:target/glados.map: No such file

So the pool built on the development machine, where the default target is
Windows, and could not link for musl. `pool.yml` had already been fixed once
for a different inherited key (`build.target`, whose escape *does* work,
because a string is overridden by the closest config) and failed again on
this one immediately afterwards.

The comment is the part worth preventing. Nobody re-checks a belief that is
written down beside the code, so the fix is not to correct the sentence but
to make the arrangement refusable.

### The rule

**A config with crates beneath it may not declare an array key under
`[build]`.** Those keys -- `rustflags`, `rustdocflags` -- are the only cargo
config values that *accumulate* across files, which makes them the only ones
a nested crate cannot opt out of. Scalars are fine: the closest config wins,
which is exactly what `build.target` relies on.

Flags that are really about one target belong under `[target.<triple>]`,
where they reach the target they describe and nothing else. That is where the
kernel's live now.

    python3 tools/cargocheck.py
    python3 tools/cargocheck.py --selftest
"""

import argparse
import os
import sys
import tempfile
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The `[build]` keys cargo joins rather than overrides. If cargo ever grows
#: another, it belongs here -- and the failure of forgetting is silent, which
#: is why the list is named rather than inferred from the value's type: an
#: array-valued key nobody has thought about should be a refusal to look at,
#: not a pass.
INHERITED_ARRAYS = ("rustflags", "rustdocflags")


def configs(root):
    """Every `.cargo/config.toml`, nearest the root first."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in ("target", ".git", "node_modules")]
        if os.path.basename(dirpath) == ".cargo" and "config.toml" in filenames:
            out.append(os.path.join(dirpath, "config.toml"))
    return sorted(out, key=lambda p: p.count(os.sep))


def crate_dir(config_path):
    """The directory a config governs: the parent of its `.cargo`."""
    return os.path.dirname(os.path.dirname(config_path))


def check(root, quiet=False):
    bad = []
    found = configs(root)
    dirs = [crate_dir(p) for p in found]
    for path, d in zip(found, dirs):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        try:
            with open(path, "rb") as f:
                cfg = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            bad.append("%s: will not parse: %s" % (rel, e))
            continue

        # Does anything live underneath this one?
        below = [o for o in dirs if o != d and o.startswith(d + os.sep)]
        build = cfg.get("build") or {}
        for key in INHERITED_ARRAYS:
            if key not in build:
                continue
            if below:
                names = ", ".join(
                    os.path.relpath(o, root).replace(os.sep, "/") for o in below)
                bad.append(
                    "%s declares `build.%s`, which cargo JOINS rather than "
                    "overrides, so %s cannot escape it -- move it under "
                    "`[target.<triple>]`" % (rel, key, names))
            elif build[key] == []:
                # The other half of the same misunderstanding: an empty list
                # that was written to clear something and clears nothing.
                bad.append(
                    "%s sets `build.%s = []`, which clears nothing -- cargo "
                    "joins arrays, so this is either a no-op or a belief that "
                    "will be wrong the day a parent declares one" % (rel, key))
        if not quiet and not any(b.startswith(rel) for b in bad):
            print("  ok   %s" % rel)
    if not found and not quiet:
        print("  no cargo config in the tree")
    return bad


# ----------------------------------------------------------------- selftest

PARENT_BAD = """
[build]
target = "x86_64-unknown-uefi"
rustflags = ["-C", "link-arg=/MAP:target/glados.map"]
"""

PARENT_OK = """
[build]
target = "x86_64-unknown-uefi"

[target.x86_64-unknown-uefi]
rustflags = ["-C", "link-arg=/MAP:target/glados.map"]
"""

CHILD_CLEARING = """
[build]
target = "x86_64-pc-windows-msvc"
rustflags = []
"""

CHILD_OK = """
[build]
target = "x86_64-pc-windows-msvc"
"""


def selftest():
    claims = []

    def claim(ok, what):
        claims.append(bool(ok))
        print("  %-4s %s" % ("ok" if ok else "FAIL", what))

    with tempfile.TemporaryDirectory() as tmp:
        def write(rel, text):
            p = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)

        # The exact arrangement that broke the pool.
        write(".cargo/config.toml", PARENT_BAD)
        write("pool/.cargo/config.toml", CHILD_CLEARING)
        bad = check(tmp, quiet=True)
        claim(any("cannot escape it" in b for b in bad),
              "a parent's `build.rustflags` is refused when a crate sits below")
        claim(any("clears nothing" in b for b in bad),
              "and the child's `rustflags = []` is named as clearing nothing")

        # The fix, which must be accepted -- a check that refuses the
        # correction as well as the defect is a check nobody can satisfy.
        write(".cargo/config.toml", PARENT_OK)
        write("pool/.cargo/config.toml", CHILD_OK)
        claim(not check(tmp, quiet=True),
              "and the same flags under `[target.<triple>]` are accepted")

        # A leaf may hold build.rustflags: nothing is beneath it to inherit.
        write("pool/.cargo/config.toml",
              CHILD_OK + '\nrustflags = ["-C", "opt-level=2"]\n')
        claim(not check(tmp, quiet=True),
              "a crate with nothing below it may declare its own flags")

        write(".cargo/config.toml", "[build]\nrustflags = [\n")
        claim(any("will not parse" in b for b in check(tmp, quiet=True)),
              "a config that will not parse is refused")

    print()
    if all(claims):
        print("  cargocheck passed (%d claims)" % len(claims))
        return 0
    print("  cargocheck FAILED")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    bad = check(a.root)
    if bad:
        print()
        for b in bad:
            print("::error::%s" % b)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

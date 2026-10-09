#!/usr/bin/env python3
"""Wireless firmware for the boot volume and the install image.

`dev::firmware` reads every file under `\\GLADOS\\FW\\` before boot services
exit, and `dev::iwx` asks it for the image its registers say it needs. This
puts those images there: out of the host's `/lib/firmware`, decompressed, one
per firmware *base* the driver can name, at the highest API the driver
understands.

### Why a base and an API, and who decides which

An Intel image is named `iwlwifi-<base>-<api>.ucode`. The base is a fact about
the silicon -- `so-a0-hr-b0` is a Snow Owl controller at step A with a Harrier
radio at step B, the GF63's -- and the kernel derives it from `CSR_HW_REV` and
`CSR_HW_RF_ID`. The API is a fact about the firmware's command layouts, and
the driver understands layouts up to one version and not past it.

So the **kernel** chooses among what is present by base, taking the highest API
it finds, and **this** decides what is present: the highest API at or below
`MAX_API`. The ceiling is written twice, here and as `iwx::MAX_API`, and the
selftest reads the kernel's out of the source and refuses a disagreement --
the same bargain `knob.py check` makes, because a ceiling raised on one side
alone ships an image the driver then misreads with no error anywhere.

### What it does not check

That an image boots. `iwxfw.py` parses the container and `iwx ctxt` builds the
boot structures from it under emulation; neither proves the firmware runs, and
nothing short of the laptop does.

Usage:

    python3 tools/wifi_fw.py stage                # into esp/GLADOS/FW
    python3 tools/wifi_fw.py stage --for so-a0-hr-b0
    python3 tools/wifi_fw.py record               # payload/firmware.txt
    python3 tools/wifi_fw.py --selftest
"""

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path("/lib/firmware")
DEST = ROOT / "esp" / "GLADOS" / "FW"
MANIFEST = ROOT / "payload" / "firmware.txt"
LICENCE_NAME = "LICENCE.iwlwifi_firmware"
LICENCE_SOURCES = [
    Path("/usr/share/licenses/linux-firmware-intel") / LICENCE_NAME,
    Path("/usr/share/licenses/linux-firmware-other") / LICENCE_NAME,
    SOURCE / LICENCE_NAME,
]

# The API window the driver searches. Must equal `MIN_API`/`MAX_API` in
# src/dev/iwx/mod.rs; the selftest checks.
MAX_API = 89
MIN_API = 50

# What `dev::firmware` will read: past these it skips files in directory order,
# which on the ISO is sorted order -- so an oversized set does not fail, it
# quietly drops whichever images sort last, the GF63's among them. Must equal
# the constants in src/dev/firmware.rs; the selftest checks.
MAX_FILES = 24
MAX_FILE = 8 << 20
MAX_TOTAL = 32 << 20
MAX_NAME = 64

# What `stage` may delete from its destination. Anything else there was not put
# there by this tool, and a `--dest` one directory too high is the model.
OURS = re.compile(r"^(iwlwifi-[A-Za-z0-9-]+\.(ucode|pnvm)|" + re.escape("LICENCE.iwlwifi_firmware") + r")$")

# Every base `iwx::firmware_base` can answer. Bz is absent because the driver
# refuses that family; the 22000 bases are present because the driver names
# them, even though its boot path does not take that family yet -- the image is
# what lets somebody with the part try.
BASES = [
    # AX210 family: Snow Owl with each radio, and Typhoon Peak.
    "so-a0-hr-b0",
    "so-a0-gf-a0",
    "so-a0-jf-b0",
    "ty-a0-gf-a0",
    # Ma, which both references file under the AX210 family.
    "ma-b0-hr-b0",
    "ma-b0-gf-a0",
    # 22000 family.
    "cc-a0",
    "Qu-b0-hr-b0",
    "Qu-c0-hr-b0",
    "QuZ-a0-hr-b0",
    "Qu-b0-jf-b0",
    "Qu-c0-jf-b0",
    "QuZ-a0-jf-b0",
]

# Gale Force radios want a platform NVM beside the image; Harrier and Jefferson
# have none, which is why the GF63 runs without one.
PNVM = ["so-a0-gf-a0", "ty-a0-gf-a0", "ma-b0-gf-a0"]


def pick(base: str, names) -> str | None:
    """The file to ship for one base: highest API at or below the ceiling.

    Pure over a list of names, so it is checked against invented directories.
    A compressed and an uncompressed copy of one API are the same image.
    """
    pat = re.compile(r"^iwlwifi-" + re.escape(base) + r"-(\d+)\.ucode(\.zst|\.xz)?$")
    best = None
    for n in names:
        m = pat.match(n)
        if not m:
            continue
        api = int(m.group(1))
        if api > MAX_API or api < MIN_API:
            continue
        if best is None or api > best[0] or (api == best[0] and m.group(2) is None):
            best = (api, n)
    return best[1] if best else None


def plain_name(n: str) -> str:
    return re.sub(r"\.(zst|xz)$", "", n)


def decompress(src: Path, dst: Path):
    """Into a temporary beside `dst`, then renamed: a corrupt input must not
    destroy the good image already there."""
    tmp = dst.with_name(dst.name + ".part")
    try:
        if src.suffix == ".zst":
            subprocess.run(["zstd", "-dqf", str(src), "-o", str(tmp)], check=True)
        elif src.suffix == ".xz":
            with open(tmp, "wb") as out:
                subprocess.run(["xz", "-dc", str(src)], check=True, stdout=out)
        else:
            shutil.copyfile(src, tmp)
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def over_limits(sizes: dict) -> list:
    """What the kernel would refuse or skip, as sentences. Pure, for the selftest."""
    out = []
    if len(sizes) > MAX_FILES:
        out.append(f"{len(sizes)} files, and the kernel reads {MAX_FILES}")
    for n, sz in sizes.items():
        if sz > MAX_FILE:
            out.append(f"{n} is {sz:,} B, over the kernel's {MAX_FILE:,} per file")
        if len(n) > MAX_NAME:
            out.append(f"{n} is a longer name than the kernel's {MAX_NAME}")
    total = sum(sizes.values())
    if total > MAX_TOTAL:
        out.append(f"{total:,} B in all, over the kernel's {MAX_TOTAL:,}")
    return out


def plan(source: Path, only=None):
    names = [p.name for p in source.iterdir()] if source.is_dir() else []
    out, missing = [], []
    for base in BASES:
        if only and base not in only:
            continue
        n = pick(base, names)
        if n is None:
            missing.append(base)
            continue
        out.append((source / n, plain_name(n)))
        if base in PNVM:
            for suf in ("", ".zst", ".xz"):
                p = f"iwlwifi-{base}.pnvm{suf}"
                if p in names:
                    out.append((source / p, plain_name(p)))
                    break
    return out, missing


def cmd_stage(args) -> int:
    only = set(args.only) if args.only else None
    if only:
        unknown = sorted(only - set(BASES))
        if unknown:
            print(f"  not a base the driver names: {', '.join(unknown)}", file=sys.stderr)
            return 1
    source = Path(args.source)
    if not source.is_dir():
        print(f"  {source} is not a directory", file=sys.stderr)
        return 1
    dest = Path(args.dest)
    # The destination is a firmware directory and nothing else: the prune below
    # would otherwise take whatever lives beside the images.
    if dest.is_symlink() or dest.name != "FW":
        print(f"  refusing {dest}: the destination is a real directory named FW", file=sys.stderr)
        return 1
    lic = next((p for p in LICENCE_SOURCES if p.is_file()), None)
    if lic is None:
        print(f"  no {LICENCE_NAME} on this host; refusing to stage images without it",
              file=sys.stderr)
        return 1
    files, missing = plan(source, only)
    if not files:
        print(f"  no image for any base in {source}; nothing staged, nothing removed",
              file=sys.stderr)
        return 1
    dest.mkdir(parents=True, exist_ok=True)
    keep = set()
    for src, name in files:
        try:
            decompress(src, dest / name)
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"  {src} would not decompress ({e}); {dest / name} left as it was",
                  file=sys.stderr)
            return 1
        keep.add(name)
        print(f"  {name:36} {(dest / name).stat().st_size:>10,} B")
    shutil.copyfile(lic, dest / LICENCE_NAME)
    keep.add(LICENCE_NAME)
    # Anything else of ours in the directory goes: the kernel loads all of it,
    # and an image this run did not choose is an image nobody chose. Not with
    # `--for`, which adds one base to a set rather than replacing the set.
    if not only:
        for p in dest.iterdir():
            if p.is_file() and p.name not in keep and OURS.match(p.name):
                p.unlink()
                print(f"  removed {p.name}, which this run did not choose")
    for b in missing:
        print(f"  no image for {b} at API {MIN_API}..={MAX_API} in {source}")
    sizes = {p.name: p.stat().st_size for p in dest.iterdir() if p.is_file()}
    print(f"  {len(sizes)} file(s), {sum(sizes.values()) / 1e6:.1f} MB in {dest}")
    bad = over_limits(sizes)
    for line in bad:
        print(f"  {line}", file=sys.stderr)
    return 1 if bad else 0


def cmd_record(args) -> int:
    sys.path.insert(0, str(ROOT / "tools"))
    import payload
    dest = Path(args.dest)
    entries = [e for e in payload.scan(dest)]
    if not entries:
        print(f"  {dest} is empty; run 'stage' first", file=sys.stderr)
        return 1
    # Release assets are flat and the kernel reads one directory, so a nested
    # name is a line no CI download could ever verify.
    stray = [e[0] for e in entries if "/" in e[0] or not OURS.match(e[0])]
    if stray or not (dest / LICENCE_NAME).is_file():
        print(f"  refusing {dest}: it must be flat, hold only images, and carry {LICENCE_NAME}",
              file=sys.stderr)
        for n in stray:
            print(f"    not firmware: {n}", file=sys.stderr)
        return 1
    manifest = Path(args.manifest)
    text = payload.render(entries).replace(
        "# Written by tools/payload.py. The bytes an ISO build must fetch\n"
        "# before it can run mkiso.py, and what they have to hash to.",
        "# Written by tools/wifi_fw.py, in payload.py's format: wireless firmware\n"
        "# for \\GLADOS\\FW\\, redistributable in binary form on condition the\n"
        "# licence beside it travels with it.")
    manifest.write_text(text, encoding="utf-8")
    print(f"  wrote {manifest}, {len(entries)} file(s)")
    return 0


def kernel_const(path: str, name: str):
    """A numeric constant out of the kernel source, `8 << 20` included."""
    src = (ROOT / path).read_text(encoding="utf-8")
    m = re.search(r"const " + name + r": \w+ = ([0-9 <]+);", src)
    if not m:
        return None
    parts = [int(x) for x in m.group(1).split("<<")]
    return parts[0] << parts[1] if len(parts) == 2 else parts[0]


def kernel_max_api():
    return kernel_const("src/dev/iwx/mod.rs", "MAX_API")


def selftest() -> int:
    ok = True

    def claim(what, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
        ok &= bool(cond)

    names = [f"iwlwifi-so-a0-hr-b0-{n}.ucode.zst" for n in (72, 77, 86, 89, 93)]
    claim("the highest API at or below the ceiling is chosen, not the newest",
          pick("so-a0-hr-b0", names) == "iwlwifi-so-a0-hr-b0-89.ucode.zst")
    claim("a base with nothing at or below the ceiling has no image",
          pick("so-a0-hr-b0", ["iwlwifi-so-a0-hr-b0-93.ucode"]) is None)
    # `so-a0-gf-a0` is a prefix of `so-a0-gf4-a0` only by eye; the pattern is
    # anchored on the API, so one base's choice cannot be another's file.
    claim("a neighbouring base is not mistaken for this one",
          pick("so-a0-gf-a0", ["iwlwifi-so-a0-gf4-a0-89.ucode"]) is None)
    claim("an uncompressed copy is preferred over a compressed one of the same API",
          pick("cc-a0", ["iwlwifi-cc-a0-77.ucode.zst", "iwlwifi-cc-a0-77.ucode"])
          == "iwlwifi-cc-a0-77.ucode")
    claim("the GF63's image is the one the kernel was measured against",
          pick("so-a0-hr-b0", [f"iwlwifi-so-a0-hr-b0-{n}.ucode.zst" for n in (72, 86, 89)])
          == "iwlwifi-so-a0-hr-b0-89.ucode.zst")
    claim("the ceiling here is the kernel's ceiling", kernel_max_api() == MAX_API)
    claim("and the floor is the kernel's floor",
          kernel_const("src/dev/iwx/mod.rs", "MIN_API") == MIN_API)
    claim("an image below the floor is one the kernel never asks for",
          pick("cc-a0", ["iwlwifi-cc-a0-46.ucode"]) is None)
    fw = "src/dev/firmware.rs"
    claim("the limits here are the firmware store's limits",
          (kernel_const(fw, "MAX_FILES"), kernel_const(fw, "MAX_FILE"),
           kernel_const(fw, "MAX_TOTAL"), kernel_const(fw, "MAX_NAME"))
          == (MAX_FILES, MAX_FILE, MAX_TOTAL, MAX_NAME))
    claim("a set the kernel would cut short is reported",
          over_limits({f"f{i}": 1 for i in range(MAX_FILES + 1)}) != []
          and over_limits({"a": MAX_TOTAL // 2, "b": MAX_TOTAL // 2 + 1}) != []
          and over_limits({"a": 1}) == [])
    claim("the prune touches images and the licence and nothing else",
          OURS.match("iwlwifi-so-a0-hr-b0-89.ucode") and OURS.match("iwlwifi-ty-a0-gf-a0.pnvm")
          and OURS.match(LICENCE_NAME) and not OURS.match("model.bin")
          and not OURS.match("roots.der") and not OURS.match("iwlwifi-x.ucode.part"))

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        for n in ("iwlwifi-so-a0-gf-a0-89.ucode", "iwlwifi-so-a0-gf-a0.pnvm",
                  "iwlwifi-so-a0-hr-b0-89.ucode"):
            (src / n).write_bytes(n.encode())
        files, missing = plan(src)
        got = sorted(n for _, n in files)
        claim("a Gale Force base brings its platform NVM and Harrier does not",
              got == ["iwlwifi-so-a0-gf-a0-89.ucode", "iwlwifi-so-a0-gf-a0.pnvm",
                      "iwlwifi-so-a0-hr-b0-89.ucode"])
        claim("and every base with no image is reported rather than skipped silently",
              "ty-a0-gf-a0" in missing and "so-a0-hr-b0" not in missing)

    print("  selftest", "passed" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--source", default=str(SOURCE))
    s.add_argument("--dest", default=str(DEST))
    s.add_argument("--for", dest="only", action="append", metavar="BASE")
    r = sub.add_parser("record")
    r.add_argument("--dest", default=str(DEST))
    r.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args()
    return cmd_stage(args) if args.cmd == "stage" else cmd_record(args)


if __name__ == "__main__":
    sys.exit(main())

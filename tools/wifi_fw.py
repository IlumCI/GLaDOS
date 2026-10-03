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

# The highest firmware API the driver understands. Must equal `MAX_API` in
# src/dev/iwx/mod.rs; the selftest checks.
MAX_API = 89

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
        if api > MAX_API:
            continue
        if best is None or api > best[0] or (api == best[0] and m.group(2) is None):
            best = (api, n)
    return best[1] if best else None


def plain_name(n: str) -> str:
    return re.sub(r"\.(zst|xz)$", "", n)


def decompress(src: Path, dst: Path):
    if src.suffix == ".zst":
        subprocess.run(["zstd", "-dqf", str(src), "-o", str(dst)], check=True)
    elif src.suffix == ".xz":
        with open(dst, "wb") as out:
            subprocess.run(["xz", "-dc", str(src)], check=True, stdout=out)
    else:
        shutil.copyfile(src, dst)


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
    files, missing = plan(Path(args.source), only)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    keep = set()
    for src, name in files:
        decompress(src, dest / name)
        keep.add(name)
        print(f"  {name:36} {(dest / name).stat().st_size:>10,} B")
    lic = next((p for p in LICENCE_SOURCES if p.is_file()), None)
    if lic is None:
        print(f"  no {LICENCE_NAME} on this host; refusing to stage images without it",
              file=sys.stderr)
        return 1
    shutil.copyfile(lic, dest / LICENCE_NAME)
    keep.add(LICENCE_NAME)
    # Anything else in the directory goes: the kernel loads all of it, and an
    # image this run did not choose is an image nobody chose.
    for p in dest.iterdir():
        if p.is_file() and p.name not in keep:
            p.unlink()
            print(f"  removed {p.name}, which this run did not choose")
    for b in missing:
        print(f"  no image for {b} at API <= {MAX_API} in {args.source}")
    total = sum((dest / n).stat().st_size for n in keep)
    print(f"  {len(keep)} file(s), {total / 1e6:.1f} MB in {dest}")
    return 0


def cmd_record(args) -> int:
    sys.path.insert(0, str(ROOT / "tools"))
    import payload
    dest = Path(args.dest)
    entries = [e for e in payload.scan(dest)]
    if not entries:
        print(f"  {dest} is empty; run 'stage' first", file=sys.stderr)
        return 1
    text = payload.render(entries).replace(
        "# Written by tools/payload.py. The bytes an ISO build must fetch\n"
        "# before it can run mkiso.py, and what they have to hash to.",
        "# Written by tools/wifi_fw.py, in payload.py's format: wireless firmware\n"
        "# for \\GLADOS\\FW\\, redistributable in binary form on condition the\n"
        "# licence beside it travels with it.")
    MANIFEST.write_text(text, encoding="utf-8")
    print(f"  wrote {MANIFEST}, {len(entries)} file(s)")
    return 0


def kernel_max_api():
    src = (ROOT / "src" / "dev" / "iwx" / "mod.rs").read_text(encoding="utf-8")
    m = re.search(r"pub const MAX_API: u32 = (\d+);", src)
    return int(m.group(1)) if m else None


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
    args = ap.parse_args()
    return cmd_stage(args) if args.cmd == "stage" else cmd_record(args)


if __name__ == "__main__":
    sys.exit(main())

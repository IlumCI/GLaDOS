#!/usr/bin/env python3
"""Build a raw, *writable* ESP disk image for QEMU.

    mkesp.py <esp-dir> <out.img>

### Why this exists

`drive.py` hands QEMU `-drive file=fat:rw:<dir>`, which is VVFAT: a synthetic
FAT16 volume projected over a host directory. Its read-write mode can modify
the contents of a file that was there when the guest booted, and it cannot do
directory operations -- creating a file, deleting one, growing a directory.
That is enough for every test this project ran until an update mechanism
needed the firmware to *write* the ESP, and then it is enough for nothing:

    glados: the flag is set and there is no staged image     (guest saw it)
    Test-Path .qemu/esp/GLADOS/UPDATE.FLG -> True             (host still has it)

The guest read the flag correctly and could not clear it. So the whole
pre-ExitBootServices swap -- copy the running image aside, overwrite
BOOTX64.EFI, drop a health flag -- was untestable, and "untestable" was being
recorded as "the firmware will not write VVFAT" without anybody having
checked which half was true.

A real FAT32 filesystem on a real block device has no such limit. The
firmware's own FAT driver does the allocation, which is exactly the
arrangement the kernel is relying on in production.

### What it produces

An MBR with one partition of type 0xEF (EFI System) starting at LBA 2048,
holding a FAT32 volume built by `mkiso.build_fat` -- the same writer whose
output the firmware already boots through El Torito, so the filesystem half is
not new code on trial.

The image is a *disk*, not a projection: the guest's writes stay in it across
boots. That is the point. It is what makes the two-boot flow -- apply, reboot,
prove healthy -- observable at all, and it is why this does not rebuild an
image that already exists unless asked.
"""
import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mkiso  # noqa: E402

SECTOR = 512
PART_START = 2048
# EFI System. The firmware looks for this type; 0x0C would be found by some
# firmwares and not others, and "some" is not a property to build on.
PART_TYPE = 0xEF


def tree_of(root_dir):
    """Mirror a host directory as `mkiso.Entry` nodes, sorted."""

    def walk(path, name):
        e = mkiso.Entry(name, 0)
        for child in sorted(path.iterdir(), key=lambda p: p.name.upper()):
            if child.is_dir():
                e.children.append(walk(child, child.name))
            elif child.is_file():
                e.children.append(mkiso.Entry(child.name, child.stat().st_size, child))
        return e

    return walk(Path(root_dir), None)


def mbr(part_sectors):
    b = bytearray(SECTOR)
    e = 446
    b[e + 0] = 0x80
    b[e + 1:e + 4] = b"\xfe\xff\xff"
    b[e + 4] = PART_TYPE
    b[e + 5:e + 8] = b"\xfe\xff\xff"
    struct.pack_into("<I", b, e + 8, PART_START)
    struct.pack_into("<I", b, e + 12, part_sectors)
    struct.pack_into("<H", b, 510, 0xAA55)
    return bytes(b)


def build(esp_dir, out_path, slack_mb=16):
    root = tree_of(esp_dir)
    total = sum(f.stat().st_size for f in Path(esp_dir).rglob("*") if f.is_file())

    # Cluster size from the payload, the same ladder drive.py uses for the ISO.
    cluster = 512
    while cluster < 32768 and total > 60000 * cluster:
        cluster *= 2

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    off = PART_START * SECTOR
    with open(out, "wb") as fh:
        fh.write(b"\x00" * off)
        size = mkiso.build_fat(root, fh, cluster)
        # Room for the guest to write into. Without it the volume is exactly
        # as large as what was staged, and the first file the firmware creates
        # -- a rollback copy of the boot image -- has nowhere to go.
        fh.write(b"\x00" * (slack_mb * 1024 * 1024))
        end = fh.tell()

    part_sectors = (end - off) // SECTOR
    with open(out, "r+b") as fh:
        fh.write(mbr(part_sectors))

    # A short write means the host disk filled underneath us, and the
    # firmware's complaint about such an image names nothing resembling that.
    actual = out.stat().st_size
    if actual != end:
        raise SystemExit(
            f"{out} is {actual} bytes, wanted {end} -- the write did not "
            "complete; check free disk space"
        )
    return size, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("esp", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--slack-mb", type=int, default=16,
                    help="free space left in the volume for the guest to write")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the image exists (discards guest writes)")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        print(f"  {args.out} exists; --force to rebuild (this discards guest writes)")
        return

    fat, total = build(args.esp, args.out, args.slack_mb)
    print(f"  fat32 {fat / 1024 / 1024:.1f} MB, image {total / 1024 / 1024:.1f} MB")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a small partitioned FAT16 disk image, to test the kernel's FAT reader.

Written by hand rather than produced with mkfs or a VHD because both of those
need either a Linux host or Administrator, and a test that cannot be run is not
a test. Everything here is laid out from the specification, so a disagreement
between this and the kernel is a disagreement about the format rather than
about a tool.

Includes the awkward cases on purpose: a subdirectory, a file whose contents
span more than one cluster, and a long filename, which is stored as UCS-2
fragments split across three non-contiguous ranges of each record and is the
single most error-prone part of the format.
"""

import struct
import sys
from pathlib import Path

SECTOR = 512
SPC = 4                 # sectors per cluster -> 2 KiB clusters
RESERVED = 1
NUM_FATS = 2
ROOT_ENTRIES = 512
PART_START = 2048       # conventional 1 MiB alignment
PART_SECTORS = 65536    # 32 MiB, comfortably in FAT16 territory

ROOT_SECTORS = (ROOT_ENTRIES * 32 + SECTOR - 1) // SECTOR
FAT_SECTORS = 64        # 32768 entries of 2 bytes
DATA_START = RESERVED + NUM_FATS * FAT_SECTORS + ROOT_SECTORS


def boot_sector():
    b = bytearray(SECTOR)
    b[0:3] = b"\xeb\x3c\x90"
    b[3:11] = b"GLADOS  "
    struct.pack_into("<H", b, 0x0B, SECTOR)
    b[0x0D] = SPC
    struct.pack_into("<H", b, 0x0E, RESERVED)
    b[0x10] = NUM_FATS
    struct.pack_into("<H", b, 0x11, ROOT_ENTRIES)
    # The 16-bit count is zero whenever the volume exceeds 65535 sectors, and
    # the 32-bit field at 0x20 carries it instead. Writing the real number here
    # is not merely wrong, it does not fit -- which is how this was found.
    struct.pack_into("<H", b, 0x13, 0)
    struct.pack_into("<I", b, 0x20, PART_SECTORS)
    b[0x15] = 0xF8                                   # fixed disk
    struct.pack_into("<H", b, 0x16, FAT_SECTORS)
    struct.pack_into("<H", b, 0x18, 63)              # sectors per track
    struct.pack_into("<H", b, 0x1A, 255)             # heads
    b[0x26] = 0x29                                   # extended boot signature
    struct.pack_into("<I", b, 0x27, 0x12345678)
    b[0x2B:0x36] = b"RESCUE     "
    b[0x36:0x3E] = b"FAT16   "
    struct.pack_into("<H", b, 510, 0xAA55)
    return b


def mbr():
    b = bytearray(SECTOR)
    # One primary partition, type 0x06 (FAT16 >32 MiB). CHS fields are left as
    # placeholders: nothing here reads them, and nothing modern should.
    e = 446
    b[e + 0] = 0x80                    # bootable
    b[e + 1:e + 4] = b"\xfe\xff\xff"
    b[e + 4] = 0x06
    b[e + 5:e + 8] = b"\xfe\xff\xff"
    struct.pack_into("<I", b, e + 8, PART_START)
    struct.pack_into("<I", b, e + 12, PART_SECTORS)
    struct.pack_into("<H", b, 510, 0xAA55)
    return b


def lfn_checksum(short11):
    s = 0
    for c in short11:
        s = (((s & 1) << 7) + (s >> 1) + c) & 0xFF
    return s


def short_entry(name8, ext3, attr, cluster, size):
    e = bytearray(32)
    e[0:8] = name8.ljust(8).encode()
    e[8:11] = ext3.ljust(3).encode()
    e[11] = attr
    struct.pack_into("<H", e, 26, cluster)
    struct.pack_into("<I", e, 28, size)
    return e


def lfn_entries(long_name, short11):
    """The 13 characters of each record live at 1..11, 14..26 and 28..32."""
    chars = list(long_name.encode("utf-16-le"))
    units = [chars[i] | (chars[i + 1] << 8) for i in range(0, len(chars), 2)]
    units.append(0x0000)
    while len(units) % 13:
        units.append(0xFFFF)

    chunks = [units[i:i + 13] for i in range(0, len(units), 13)]
    csum = lfn_checksum(short11)
    out = []
    # Stored last-first, with the final fragment flagged 0x40.
    for i, chunk in reversed(list(enumerate(chunks))):
        e = bytearray(32)
        e[0] = (i + 1) | (0x40 if i == len(chunks) - 1 else 0)
        e[11] = 0x0F
        e[13] = csum
        for k, (off, count) in enumerate([(1, 5), (14, 6), (28, 2)]):
            base = sum(c for _, c in [(1, 5), (14, 6), (28, 2)][:k])
            for j in range(count):
                struct.pack_into("<H", e, off + j * 2, chunk[base + j])
        out.append(e)
    return b"".join(bytes(e) for e in out)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tools/out/rescue.img")
    disk = bytearray(PART_START * SECTOR + PART_SECTORS * SECTOR)
    disk[0:SECTOR] = mbr()

    part = PART_START * SECTOR
    disk[part:part + SECTOR] = boot_sector()

    cluster_bytes = SPC * SECTOR
    fat = [0] * 32768
    fat[0] = 0xFFF8
    fat[1] = 0xFFFF

    next_free = 2
    data_off = part + DATA_START * SECTOR

    def alloc(payload):
        """Write payload into a fresh chain, return its first cluster."""
        nonlocal next_free
        n = max(1, (len(payload) + cluster_bytes - 1) // cluster_bytes)
        first = next_free
        for i in range(n):
            c = first + i
            off = data_off + (c - 2) * cluster_bytes
            chunk = payload[i * cluster_bytes:(i + 1) * cluster_bytes]
            disk[off:off + len(chunk)] = chunk
            fat[c] = 0xFFFF if i == n - 1 else c + 1
        next_free += n
        return first

    # A file larger than one cluster, so the chain is actually followed.
    big = ("GLaDOS rescue volume.\n" + "".join(
        f"line {i:04} padding to force a multi-cluster chain\n" for i in range(80)
    )).encode()
    big_cluster = alloc(big)

    hello = b"hello from a real filesystem\n"
    hello_cluster = alloc(hello)

    # A subdirectory containing one file.
    inner = b"nested file contents\n"
    inner_cluster = alloc(inner)

    sub_entries = bytearray()
    sub_entries += short_entry(".", "", 0x10, 0, 0)
    sub_entries += short_entry("..", "", 0x10, 0, 0)
    sub_entries += short_entry("INNER", "TXT", 0x20, inner_cluster, len(inner))
    sub_cluster = alloc(bytes(sub_entries).ljust(cluster_bytes, b"\x00"))

    root = bytearray()
    root += short_entry("HELLO", "TXT", 0x20, hello_cluster, len(hello))
    root += short_entry("BIG", "TXT", 0x20, big_cluster, len(big))
    root += short_entry("EFI", "", 0x10, sub_cluster, 0)
    # Long name, with its 8.3 alias following as the format requires.
    root += lfn_entries("a rather long filename.txt", b"ALONGF~1TXT")
    root += short_entry("ALONGF~1", "TXT", 0x20, hello_cluster, len(hello))

    root_off = part + (RESERVED + NUM_FATS * FAT_SECTORS) * SECTOR
    disk[root_off:root_off + len(root)] = root

    packed = b"".join(struct.pack("<H", v) for v in fat)
    for i in range(NUM_FATS):
        off = part + (RESERVED + i * FAT_SECTORS) * SECTOR
        disk[off:off + FAT_SECTORS * SECTOR] = packed[:FAT_SECTORS * SECTOR]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(disk)
    print(f"  {out}  {len(disk):,} B")
    print(f"  partition at LBA {PART_START}, {PART_SECTORS} sectors, FAT16")
    print(f"  data starts at sector {DATA_START} of the partition")
    print(f"  files: HELLO.TXT, BIG.TXT ({len(big)} B, multi-cluster),")
    print("         EFI/INNER.TXT, and 'a rather long filename.txt'")


if __name__ == "__main__":
    main()

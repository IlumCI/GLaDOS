#!/usr/bin/env python3
"""Build a bootable GLaDOS ISO: a FAT32 EFI System Partition inside El Torito.

Why this exists rather than a call to xorriso
---------------------------------------------
xorriso and mkisofs are not on this machine and are not on most Windows
machines, and the Windows ADK's oscdimg is a 1.5 GB install to produce a 600 MB
file. Both formats needed here are published and neither is large: FAT32 is a
boot sector, two allocation tables and a cluster chain, and the El Torito part
of ISO 9660 is three volume descriptors and a boot catalog. Writing them costs
less than depending on a toolchain the reader does not have.

The layout, and why it is this one
----------------------------------
UEFI firmware boots an optical device by reading the El Torito boot catalog,
finding an entry whose platform id is 0xEF, and treating the sectors it points
at as a FAT filesystem -- an EFI System Partition that happens to live inside an
ISO. So the ISO 9660 structure around it is almost ceremonial: it exists so the
disc mounts and shows its contents, not so it boots. Both are built anyway,
because a disc that boots and appears empty when mounted looks broken.

Long file names are not optional here. The kernel opens `\\GLADOS\\tokenizer.bin`
and that base name is nine characters, so it cannot be expressed as 8.3 -- a
short-name-only FAT would silently present it as TOKENI~1.BIN and the kernel
would fail to find its tokenizer at boot, on real hardware, with no filesystem
to debug from. So VFAT long name entries are generated.

Everything streams. A 600 MB image assembled in a bytearray is 600 MB of RAM
plus whatever the interpreter copies on the way; the file is written in order
instead, and the only thing held whole is the FAT itself.

Usage:
    mkiso.py OUT.iso --efi path/to/BOOTX64.EFI [--payload DIR] [--label NAME]

`--payload` is copied to \\GLADOS\\ on the image. Without one the ISO boots to a
kernel that finds no model and says so.
"""

import argparse
import struct
import sys
from pathlib import Path

SECTOR = 512
ISO_SECTOR = 2048


# --- FAT32 ----------------------------------------------------------------

class Entry:
    """One file to place on the image."""

    def __init__(self, name, size, source=None):
        self.name = name
        self.size = size
        self.source = source          # Path, or None for a directory
        self.children = []
        self.cluster = 0

    @property
    def is_dir(self):
        return self.source is None


def short_name(name, taken):
    """The 8.3 name, with a ~N tail when the real name will not fit.

    The tail is what makes a long name entry *necessary* rather than
    decorative: firmware that ignores long names still has to find a unique
    short name, and two files whose first six characters agree would otherwise
    collide.
    """
    stem, _, ext = name.rpartition('.')
    if not stem:
        stem, ext = ext, ''
    stem = ''.join(c for c in stem.upper() if c.isalnum() or c in '_-')
    ext = ''.join(c for c in ext.upper() if c.isalnum())[:3]

    if len(stem) <= 8:
        cand = stem.ljust(8)
        if (cand, ext) not in taken:
            taken.add((cand, ext))
            return cand, ext.ljust(3)

    for n in range(1, 1000):
        tail = '~' + str(n)
        cand = (stem[:8 - len(tail)] + tail).ljust(8)
        if (cand, ext) not in taken:
            taken.add((cand, ext))
            return cand, ext.ljust(3)
    raise ValueError('no free short name for ' + name)


def lfn_checksum(short11):
    """The checksum tying long name entries to their short entry.

    Specified as this exact rotate-and-add; it is not a hash anyone chose for
    its properties, and reimplementing it "better" breaks compatibility.
    """
    s = 0
    for c in short11:
        s = (((s & 1) << 7) + (s >> 1) + c) & 0xFF
    return s


def dir_entries(entry):
    """Serialise one directory's entries, long names included."""
    out = bytearray()
    taken = set()

    def raw(short11, attr, cluster, size):
        return (short11
                + bytes([attr, 0, 0])
                + struct.pack('<HH', 0, 0)          # create time/date
                + struct.pack('<H', 0)              # access date
                + struct.pack('<H', cluster >> 16)
                + struct.pack('<HH', 0, 0)          # write time/date
                + struct.pack('<H', cluster & 0xFFFF)
                + struct.pack('<I', size))

    # '.' and '..' come first in every directory but the root. Their cluster
    # numbers are the directory's own and its parent's; '..' pointing at the
    # root is written as 0, not 2, which is the one place FAT does not use the
    # real cluster number.
    if entry.name is not None:
        out += raw(b'.          ', 0x10, entry.cluster, 0)
        out += raw(b'..         ', 0x10, entry.parent_cluster, 0)

    for child in entry.children:
        stem, ext = short_name(child.name, taken)
        short11 = (stem + ext).encode('ascii')
        need_lfn = child.name != (stem.strip() + ('.' + ext.strip() if ext.strip() else ''))

        if need_lfn:
            chars = child.name.encode('utf-16-le') + b'\x00\x00'
            # Pad to a multiple of 13 characters with 0xFFFF, which is the
            # specified filler -- zero would read as a terminator.
            per = 13 * 2
            while len(chars) % per:
                chars += b'\xff\xff'
            total = len(chars) // per
            csum = lfn_checksum(short11)
            for i in range(total, 0, -1):
                part = chars[(i - 1) * per: i * per]
                seq = i | (0x40 if i == total else 0)
                out += (bytes([seq])
                        + part[0:10]
                        + bytes([0x0F, 0, csum])
                        + part[10:22]
                        + b'\x00\x00'
                        + part[22:26])

        attr = 0x10 if child.is_dir else 0x20
        out += raw(short11, attr, child.cluster, 0 if child.is_dir else child.size)

    return bytes(out)


def build_fat(root, out, cluster_size):
    """Write a FAT32 filesystem containing `root` to the open file `out`.

    Returns the image size in bytes.
    """
    spc = cluster_size // SECTOR
    reserved = 32

    # Walk the tree assigning clusters. Directories are sized by serialising
    # them once with placeholder clusters, which is safe because the number of
    # entries -- and so the number of clusters -- does not depend on the
    # cluster numbers themselves.
    order = []

    def walk(d, parent_cluster):
        d.parent_cluster = parent_cluster
        order.append(d)
        for c in d.children:
            if c.is_dir:
                walk(c, 0)          # patched below once d.cluster is known
        return d

    walk(root, 0)

    def dir_size(d):
        n = len(dir_entries(d))
        return max(1, (n + cluster_size - 1) // cluster_size)

    next_cluster = 2
    for d in order:
        d.cluster = next_cluster
        next_cluster += dir_size(d)
    for d in order:
        for c in d.children:
            if c.is_dir:
                c.parent_cluster = d.cluster if d.name is not None else 0

    files = []
    for d in order:
        for c in d.children:
            if not c.is_dir:
                c.cluster = next_cluster if c.size else 0
                next_cluster += max(0, (c.size + cluster_size - 1) // cluster_size)
                files.append(c)

    data_clusters = next_cluster - 2
    # FAT32 is defined by having at least 65525 clusters; below that some
    # firmware reads the volume as FAT16 and misparses everything. Pad rather
    # than pick a smaller cluster size, which would only move the boundary.
    total_clusters = max(data_clusters, 65540)
    fat_sectors = ((total_clusters + 2) * 4 + SECTOR - 1) // SECTOR
    fat_sectors = (fat_sectors + spc - 1) // spc * spc
    total_sectors = reserved + 2 * fat_sectors + total_clusters * spc

    # Boot sector.
    bs = bytearray(SECTOR)
    bs[0:3] = b'\xeb\x58\x90'
    bs[3:11] = b'GLADOS  '
    struct.pack_into('<HBHBHHBHHHII', bs, 11,
                     SECTOR, spc, reserved, 2, 0, 0, 0xF8, 0,
                     63, 255, 0, total_sectors)
    struct.pack_into('<IHHIHH', bs, 36, fat_sectors, 0, 0, 2, 1, 6)
    bs[64] = 0x80
    bs[66] = 0x29
    struct.pack_into('<I', bs, 67, 0x474C4144)
    bs[71:82] = b'GLADOS     '
    bs[82:90] = b'FAT32   '
    bs[510:512] = b'\x55\xaa'
    out.write(bs)

    # FSInfo. Firmware does not need it, but a volume without one is flagged
    # dirty by every tool that inspects it afterwards.
    fsi = bytearray(SECTOR)
    fsi[0:4] = b'RRaA'
    fsi[484:488] = b'rrAa'
    struct.pack_into('<II', fsi, 488, 0xFFFFFFFF, 0xFFFFFFFF)
    fsi[510:512] = b'\x55\xaa'
    out.write(fsi)

    out.write(b'\x00' * (SECTOR * 4))
    out.write(bs)                                   # backup boot sector at 6
    out.write(fsi)
    out.write(b'\x00' * (SECTOR * (reserved - 8)))

    # The allocation table. Every chain is contiguous here because nothing is
    # ever deleted from an image built in one pass, so each entry simply points
    # at its successor.
    fat = bytearray(fat_sectors * SECTOR)
    struct.pack_into('<II', fat, 0, 0x0FFFFFF8, 0x0FFFFFFF)

    def chain(start, count):
        for i in range(count):
            v = 0x0FFFFFFF if i == count - 1 else start + i + 1
            struct.pack_into('<I', fat, (start + i) * 4, v)

    for d in order:
        chain(d.cluster, dir_size(d))
    for f in files:
        if f.size:
            chain(f.cluster, (f.size + cluster_size - 1) // cluster_size)

    out.write(fat)
    out.write(fat)

    # Data region, in cluster order.
    written = 2
    for item in sorted(order + files, key=lambda x: x.cluster):
        if item.cluster == 0:
            continue
        assert item.cluster == written, (item.name, item.cluster, written)
        if item.is_dir:
            data = dir_entries(item)
            n = dir_size(item)
            out.write(data.ljust(n * cluster_size, b'\x00'))
            written += n
        else:
            n = (item.size + cluster_size - 1) // cluster_size
            with open(item.source, 'rb') as f:
                left = item.size
                while left:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        raise IOError('short read from ' + str(item.source))
                    out.write(chunk)
                    left -= len(chunk)
            pad = n * cluster_size - item.size
            out.write(b'\x00' * pad)
            written += n

    out.write(b'\x00' * ((total_clusters - (written - 2)) * cluster_size))
    return total_sectors * SECTOR


# --- ISO 9660 + El Torito -------------------------------------------------

def both(n, size):
    """A value stored little-endian then big-endian, as ISO 9660 does."""
    if size == 2:
        return struct.pack('<H', n) + struct.pack('>H', n)
    return struct.pack('<I', n) + struct.pack('>I', n)


def dir_record(name, extent, length, is_dir, root=False):
    ident = b'\x00' if root else name.encode('ascii')
    ln = 33 + len(ident)
    pad = ln % 2
    return (bytes([ln + pad, 0])
            + both(extent, 4)
            + both(length, 4)
            + bytes([125, 1, 1, 0, 0, 0, 0])        # date: arbitrary, fixed
            + bytes([2 if is_dir else 0, 0, 0])
            + both(1, 2)
            + bytes([len(ident)])
            + ident
            + b'\x00' * pad)


def build_iso(out_path, efi_img, efi_size, label):
    """Wrap an already-written ESP image in ISO 9660 with an El Torito EFI entry."""
    esp_sectors = (efi_size + ISO_SECTOR - 1) // ISO_SECTOR

    PVD_LBA, BOOT_LBA, TERM_LBA = 16, 17, 18
    PATH_L, PATH_M, ROOT_LBA, CATALOG_LBA = 19, 20, 21, 22
    ESP_LBA = 24
    total = ESP_LBA + esp_sectors

    root_rec = dir_record('', ROOT_LBA, ISO_SECTOR, True, root=True)

    with open(out_path, 'r+b') as out:
        out.seek(0)
        out.write(b'\x00' * (ISO_SECTOR * 16))

        # Primary volume descriptor.
        pvd = bytearray(ISO_SECTOR)
        pvd[0] = 1
        pvd[1:6] = b'CD001'
        pvd[6] = 1
        pvd[8:40] = b' ' * 32
        pvd[40:72] = label.upper().ljust(32).encode('ascii')[:32]
        pvd[80:88] = both(total, 4)
        pvd[120:124] = both(1, 2)
        pvd[124:128] = both(1, 2)
        pvd[128:132] = both(ISO_SECTOR, 2)
        pvd[132:140] = both(ISO_SECTOR, 4)
        pvd[140:144] = struct.pack('<I', PATH_L)
        pvd[148:152] = struct.pack('>I', PATH_M)
        pvd[156:156 + len(root_rec)] = root_rec
        for a, b in ((190, 318), (318, 446), (446, 574), (574, 702)):
            pvd[a:b] = b' ' * (b - a)
            pvd[702:813] = b' ' * 111
        pvd[813:1395] = b' ' * 582
        pvd[881] = 1
        out.write(pvd)

        # Boot record volume descriptor: says where the catalog is.
        brvd = bytearray(ISO_SECTOR)
        brvd[0] = 0
        brvd[1:6] = b'CD001'
        brvd[6] = 1
        brvd[7:39] = b'EL TORITO SPECIFICATION'.ljust(32)[:32]
        brvd[71:75] = struct.pack('<I', CATALOG_LBA)
        out.write(brvd)

        term = bytearray(ISO_SECTOR)
        term[0] = 0xFF
        term[1:6] = b'CD001'
        term[6] = 1
        out.write(term)

        # Path tables, one per endianness. A single root entry each.
        pt = bytes([1, 0]) + struct.pack('<I', ROOT_LBA) + struct.pack('<H', 1) + b'\x00\x00'
        out.write(pt.ljust(ISO_SECTOR, b'\x00'))
        pt = bytes([1, 0]) + struct.pack('>I', ROOT_LBA) + struct.pack('>H', 1) + b'\x00\x00'
        out.write(pt.ljust(ISO_SECTOR, b'\x00'))

        # Root directory: itself and its parent.
        rd = dir_record('', ROOT_LBA, ISO_SECTOR, True, root=True)
        rd += dir_record('\x01', ROOT_LBA, ISO_SECTOR, True)[:34] + b'\x00'
        out.write(rd.ljust(ISO_SECTOR, b'\x00'))

        # Boot catalog: a validation entry, then the EFI entry.
        cat = bytearray(ISO_SECTOR)
        cat[0] = 1
        cat[1] = 0xEF                               # platform: EFI
        cat[28:30] = b'\x55\xaa'
        # The validation entry carries a checksum making the first 16 words sum
        # to zero. Firmware does check it, and a wrong one is a disc that is
        # simply not offered as bootable, with nothing said about why.
        s = sum(struct.unpack('<16H', bytes(cat[0:32]))) & 0xFFFF
        struct.pack_into('<H', cat, 28, (-s) & 0xFFFF)
        cat[30:32] = b'\x55\xaa'

        cat[32] = 0x88                              # bootable
        cat[33] = 0                                 # no emulation
        struct.pack_into('<H', cat, 34, 0)
        cat[36] = 0
        # Sector count is in 512-byte virtual sectors. Firmware ignores it for
        # no-emulation EFI entries and reads the FAT's own geometry instead,
        # but a zero here is rejected outright by some implementations.
        struct.pack_into('<H', cat, 38, 1)
        struct.pack_into('<I', cat, 40, ESP_LBA)
        out.write(cat)

        out.write(b'\x00' * ISO_SECTOR)             # LBA 23, spare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output')
    ap.add_argument('--efi', required=True, help='BOOTX64.EFI to boot')
    ap.add_argument('--payload', help='directory copied to \\GLADOS\\')
    ap.add_argument('--label', default='GLADOS')
    ap.add_argument('--cluster', type=int, default=0,
                    help='cluster size; 0 picks the smallest that fits')
    args = ap.parse_args()

    efi = Path(args.efi)
    if not efi.exists():
        raise SystemExit('no such file: ' + str(efi))

    root = Entry(None, 0)
    efi_dir = Entry('EFI', 0)
    boot_dir = Entry('BOOT', 0)
    boot_dir.children.append(Entry('BOOTX64.EFI', efi.stat().st_size, efi))
    efi_dir.children.append(boot_dir)
    root.children.append(efi_dir)

    payload_bytes = 0
    if args.payload:
        pdir = Path(args.payload)
        if not pdir.is_dir():
            raise SystemExit('not a directory: ' + str(pdir))
        g = Entry('GLADOS', 0)
        for f in sorted(pdir.iterdir()):
            if f.is_file():
                g.children.append(Entry(f.name, f.stat().st_size, f))
                payload_bytes += f.stat().st_size
        root.children.append(g)

    # FAT32 is *defined* as having at least 65525 clusters -- below that the
    # volume is FAT16 and firmware that trusts the count misparses everything.
    # So the cluster size sets a floor on the image: 4 KiB clusters cannot
    # produce a volume smaller than ~256 MB, which for a kernel-only image is
    # 255 MB of zeroes. Take the largest cluster whose floor still fits
    # inside the data: past that bound a bigger cluster only trips the floor
    # and pads the image, while below it the data itself sets the size and a
    # bigger cluster merely shrinks the FAT tables.
    cluster = args.cluster
    if not cluster:
        total = efi.stat().st_size + payload_bytes
        cluster = 512
        while cluster < 32768 and 65525 * cluster * 2 <= total:
            cluster *= 2

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # The ESP is written at its final offset so the ISO headers can be filled
    # in afterwards without moving 600 MB of data.
    esp_offset = 24 * ISO_SECTOR
    with open(out, 'wb') as f:
        f.write(b'\x00' * esp_offset)
        size = build_fat(root, f, cluster)
        # ISO images are a whole number of 2048-byte sectors.
        tail = f.tell() % ISO_SECTOR
        if tail:
            f.write(b'\x00' * (ISO_SECTOR - tail))

    build_iso(out, esp_offset, size, args.label)

    mb = out.stat().st_size / 1024 / 1024
    print(f'{out}  {mb:.1f} MB')
    print(f'  BOOTX64.EFI  {efi.stat().st_size / 1024:.0f} KiB')
    if args.payload:
        print(f'  GLADOS/      {payload_bytes / 1024 / 1024:.1f} MB')
    else:
        print('  no payload -- boots to a kernel with no model')


if __name__ == '__main__':
    sys.exit(main())

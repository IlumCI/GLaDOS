#!/usr/bin/env python3
"""Read an NVIDIA VBIOS and find what booting the GPU's firmware needs from it.

The host-side reader for the native NVIDIA path, in the role reference.py
plays for the model: the kernel's parser is checked against this one, so a
disagreement about an offset shows up as two programs answering differently
rather than as a falcon that silently refuses to start.

What it reads, in the order a boot needs it:

  images   A VBIOS is several PCI expansion-ROM images back to back (x86,
           EFI, and on Turing and later an NVIDIA image of type 0xE0 that
           carries falcon firmware). Each is found from the last one's size.
  BIT      NVIDIA's table of tables, found by its signature, "\\xff\\xb8BIT".
  'p'      The BIT token pointing at the falcon ucode table.
  0x85     The entry in that table that is FWSEC: the NVIDIA-signed firmware
           that, run on the GSP falcon, carves out the protected region GSP-RM
           is later loaded into. Its descriptor says how big each part is.

**Pointers above the first image are not file offsets.** A pointer read out
of a table that lands past image 0 is relative to the start of the 0xE0
image instead -- nouveau's `nvbios_addr` does exactly this adjustment, and
reading such a pointer as a plain offset lands in the middle of the EFI
driver and decodes perfectly plausible garbage.

The layouts are the ones nouveau's nvkm/subdev/bios/{image,pcir,npde,bit,pmu}.c
and nvkm/subdev/gsp/fwsec.c read (MIT, Red Hat); the numbers are the ROM's.

    python3 tools/nvbios.py out/gpu/nv/vbios.rom
    python3 tools/nvbios.py --selftest
"""

import struct
import sys

PMU_FWSEC = 0x85


def u8(d, o):
    return d[o]


def u16(d, o):
    return struct.unpack_from("<H", d, o)[0]


def u32(d, o):
    return struct.unpack_from("<I", d, o)[0]


def images(d):
    """Every ROM image, as (base, size, type, last). Stops at the first one
    that is not an image, which is how a ROM padded with 0xff ends."""
    out, base = [], 0
    while base + 0x1a <= len(d):
        if u16(d, base) not in (0xAA55, 0xBB77, 0x564E):
            break
        p = u16(d, base + 0x18)
        if not p or base + p + 0x16 > len(d):
            break
        pcir = base + p
        if u32(d, pcir) not in (0x52494350, 0x53494752, 0x5344504E):  # PCIR RGIS NPDS
            break
        hdr = u16(d, pcir + 0x0A)
        size = u16(d, pcir + 0x10) * 512
        typ = u8(d, pcir + 0x14)
        last = bool(u8(d, pcir + 0x15) & 0x80)
        if typ != 0x70:
            # NVIDIA's own extension follows the PCIR, sixteen-aligned, and
            # where present its size and last flag are the ones to believe.
            npde = (pcir + hdr + 0x0F) & ~0x0F
            if npde + 0x0B <= len(d) and u32(d, npde) == 0x4544504E:  # NPDE
                size = u16(d, npde + 0x08) * 512
                last = bool(u8(d, npde + 0x0A) & 0x80)
        else:
            last = True
        if size == 0:
            break
        out.append((base, size, typ, last))
        if last:
            break
        base += size
    return out


class Bios:
    def __init__(self, d):
        self.d = d
        self.images = images(d)
        self.image0 = self.images[0][1] if self.images else 0
        self.imaged = next((b for b, _, t, _ in self.images[1:] if t == 0xE0), 0)
        self.bit = d.find(b"\xff\xb8BIT")
        if self.bit < 0:
            raise ValueError("no BIT signature: not an NVIDIA VBIOS, or not all of one")

    def addr(self, a, size=4):
        """A table pointer as an offset in the file -- nvbios_addr."""
        if a >= self.image0 and self.imaged:
            a = a - self.image0 + self.imaged
        if a + size > len(self.d):
            raise ValueError("pointer %#x runs past the end of the ROM (%#x)" % (a, len(self.d)))
        return a

    def r8(self, a):
        return u8(self.d, self.addr(a, 1))

    def r16(self, a):
        return u16(self.d, self.addr(a, 2))

    def r32(self, a):
        return u32(self.d, self.addr(a, 4))

    def bit_entries(self):
        """(id, version, length, offset) for every BIT token."""
        b = self.bit
        hdr, size, count = u8(self.d, b + 8), u8(self.d, b + 9), u8(self.d, b + 10)
        e = b + 12 if hdr == 0 else b + hdr
        out = []
        for _ in range(count):
            out.append((chr(u8(self.d, e)), u8(self.d, e + 1), u16(self.d, e + 2), u16(self.d, e + 4)))
            e += size
        return out

    def bit_entry(self, ident):
        return next((e for e in self.bit_entries() if e[0] == ident), None)

    def version(self):
        i = self.bit_entry("i")
        if not i or i[2] < 5:
            return None
        o = i[3]
        return "%02x.%02x.%02x.%02x.%02x" % (self.r8(o + 3), self.r8(o + 2), self.r8(o + 1), self.r8(o), self.r8(o + 4))

    def falcon_ucodes(self):
        """The 'p' table: (type, data pointer) for each falcon ucode."""
        p = self.bit_entry("p")
        if not p or p[1] != 2 or p[2] < 4:
            raise ValueError("no version-2 'p' token, so no falcon ucode table")
        t = self.r32(p[3])
        ver, hdr, ln, cnt = self.r8(t), self.r8(t + 1), self.r8(t + 2), self.r8(t + 3)
        out = []
        for i in range(cnt):
            e = t + hdr + i * ln
            out.append((self.r8(e), self.r32(e + 2)))
        return {"table": t, "version": ver, "entries": out}

    def fwsec(self):
        """FWSEC's descriptor, decoded, with the file offsets of its parts."""
        ucodes = self.falcon_ucodes()["entries"]
        hit = [data for typ, data in ucodes if typ == PMU_FWSEC]
        if not hit:
            raise ValueError("no FWSEC (type 0x85) in the falcon ucode table")
        a = self.addr(hit[0])
        hdr = u32(self.d, a)
        if not hdr & 1:
            raise ValueError("FWSEC descriptor header %#x lacks its valid bit" % hdr)
        size, vers = (hdr >> 16) & 0xFFFF, (hdr >> 8) & 0xFF
        f = {"desc": a, "desc_size": size, "version": vers}
        if vers == 3:
            names = ["Hdr", "StoredSize", "PKCDataOffset", "InterfaceOffset", "IMEMPhysBase",
                     "IMEMLoadSize", "IMEMVirtBase", "DMEMPhysBase", "DMEMLoadSize"]
            for i, n in enumerate(names):
                f[n] = u32(self.d, a + 4 * i)
            f["EngineIdMask"] = u16(self.d, a + 36)
            f["UcodeId"] = u8(self.d, a + 38)
            f["SignatureCount"] = u8(self.d, a + 39)
            f["SignatureVersions"] = u16(self.d, a + 40)
            # The image follows the descriptor: IMEM then DMEM, and the
            # signatures sit just after the descriptor's fixed part at
            # +0x2c, 96 words apiece (fwsec.c, nvkm_falcon_fw_sign).
            f["image"] = a + size
            f["image_len"] = f["IMEMLoadSize"] + f["DMEMLoadSize"]
            f["signatures"] = a + 0x2C
        elif vers == 2:
            names = ["Hdr", "StoredSize", "UncompressedSize", "VirtualEntry", "InterfaceOffset",
                     "IMEMPhysBase", "IMEMLoadSize", "IMEMVirtBase", "IMEMSecBase", "IMEMSecSize",
                     "DMEMOffset", "DMEMPhysBase", "DMEMLoadSize", "altIMEMLoadSize", "altDMEMLoadSize"]
            for i, n in enumerate(names):
                f[n] = u32(self.d, a + 4 * i)
            f["image"] = a + size
            f["image_len"] = f["IMEMLoadSize"] + f["DMEMLoadSize"]
        else:
            raise ValueError("FWSEC descriptor version %d is neither 2 nor 3" % vers)
        if f["image"] + f["image_len"] > len(self.d):
            raise ValueError("FWSEC's image runs past the end of the ROM")
        return f

    def appif(self, f):
        """The DMEM mapper FWSEC exposes for its command, as fwsec.c patches
        it: interface header at InterfaceOffset into DMEM, then entries."""
        dmem = f["image"] + (f["IMEMLoadSize"] if f["version"] == 3 else f["DMEMOffset"])
        h = dmem + f["InterfaceOffset"]
        ver, hdr, ln, cnt = self.d[h], self.d[h + 1], self.d[h + 2], self.d[h + 3]
        if ver != 1:
            raise ValueError("application interface header version %d, not 1" % ver)
        for i in range(cnt):
            e = h + hdr + i * ln
            ident, base = u32(self.d, e), u32(self.d, e + 4)
            if ident == 4:  # DMEMMAPPER
                m = dmem + base
                return {"dmem": dmem, "mapper": m, "signature": u32(self.d, m),
                        "version": u16(self.d, m + 4), "cmd_in": u32(self.d, m + 8),
                        "cmd_in_size": u32(self.d, m + 12), "init_cmd": u32(self.d, m + 40)}
        raise ValueError("no DMEM mapper among %d interface entries" % cnt)


def report(path):
    d = open(path, "rb").read()
    b = Bios(d)
    print("%s: %d bytes" % (path, len(d)))
    for base, size, typ, last in b.images:
        name = {0x00: "x86", 0x03: "EFI", 0x70: "NBSI", 0xE0: "NVIDIA falcon"}.get(typ, "?")
        print("  image at %#07x  %6d bytes  type %#04x %-14s%s" % (base, size, typ, name, "  last" if last else ""))
    covered = sum(s for _, s, _, _ in b.images)
    if covered > len(d):
        print("  ** the images claim %d bytes and the file has %d: this dump is short" % (covered, len(d)))
    print("  BIT at %#x, version %s" % (b.bit, b.version()))
    print("  tokens: " + " ".join("%s%d@%#x" % (i, v, o) for i, v, _, o in b.bit_entries()))
    if not b.imaged and not b.images[-1][3]:
        # What the PCI expansion-ROM window shows (sysfs `rom`) stops after
        # the EFI image on Ampere: the falcon image is only reachable through
        # BAR0's PROM window, which a dump taken beside a live driver lacks.
        print("  the last image is not marked last and there is no 0xE0 image:")
        print("  this is the expansion-ROM window's view, without the falcon firmware.")
        print("  Take the ROM through BAR0 + 0x300000 (PROM) for the whole of it.")
        return
    t = b.falcon_ucodes()
    print("  falcon ucode table at %#x, version %d:" % (t["table"], t["version"]))
    for typ, data in t["entries"]:
        print("    type %#04x -> %#x%s" % (typ, data, "   FWSEC" if typ == PMU_FWSEC else ""))
    f = b.fwsec()
    print("  FWSEC: descriptor v%d at %#x (%d bytes)" % (f["version"], f["desc"], f["desc_size"]))
    for k in ("IMEMPhysBase", "IMEMLoadSize", "DMEMPhysBase", "DMEMLoadSize", "InterfaceOffset",
              "PKCDataOffset", "EngineIdMask", "UcodeId", "SignatureCount", "SignatureVersions"):
        if k in f:
            print("    %-18s %#x" % (k, f[k]))
    print("    image              %#x + %d bytes" % (f["image"], f["image_len"]))
    m = b.appif(f)
    print("  DMEM mapper at DMEM+%#x: signature %#x version %d, command buffer DMEM+%#x (%d bytes), init_cmd %#x"
          % (m["mapper"] - m["dmem"], m["signature"], m["version"], m["cmd_in"], m["cmd_in_size"], m["init_cmd"]))


def selftest():
    """Claims about the parser against a synthetic ROM, so a wrong layout
    fails here rather than on a falcon."""
    ok = True

    def claim(c, what):
        nonlocal ok
        print("  %s  %s" % ("ok  " if c else "FAIL", what))
        ok &= bool(c)

    def image(typ, size, last, body=b""):
        img = bytearray(size)
        img[0:2] = b"\x55\xaa"
        struct.pack_into("<H", img, 0x18, 0x40)
        struct.pack_into("<IHH", img, 0x40, 0x52494350, 0x10DE, 0x25A2)
        struct.pack_into("<H", img, 0x4A, 0x18)
        struct.pack_into("<H", img, 0x50, size // 512)
        img[0x54] = typ
        img[0x55] = 0x80 if last else 0
        img[0x60:0x60 + len(body)] = body
        return img

    # Image 0 holds BIT and the 'p' table; the table's FWSEC pointer is
    # past image 0, so it must be read relative to the 0xE0 image.
    i0 = image(0x00, 1024, False)
    i1 = image(0x03, 512, False)
    i2 = image(0xE0, 2048, True)
    bit = 0x100
    i0[bit:bit + 6] = b"\xff\xb8BIT\x00"
    i0[bit + 8], i0[bit + 9], i0[bit + 10] = 12, 6, 1
    struct.pack_into("<BBHH", i0, bit + 12, ord("p"), 2, 4, 0x200)
    struct.pack_into("<I", i0, 0x200, 0x210)                 # 'p' -> table at 0x210
    i0[0x210:0x214] = bytes([1, 4, 6, 1])                     # ver hdr len cnt
    struct.pack_into("<BBI", i0, 0x214, PMU_FWSEC, 0, 1024 + 0x100)  # past image 0
    desc = 0x100
    struct.pack_into("<I", i2, desc, (0x2C + 4 * 96 << 16) | (3 << 8) | 1)
    struct.pack_into("<IIIIIII", i2, desc + 8, 0, 0x40, 0, 0x100, 0, 0, 0x80)
    rom = bytes(i0 + i1 + i2)
    b = Bios(rom)
    claim([t for _, _, t, _ in b.images] == [0, 3, 0xE0], "three images found by walking sizes")
    claim(b.imaged == 1536, "the 0xE0 image is located")
    claim(b.addr(1024 + 0x100) == 1536 + 0x100, "a pointer past image 0 is relative to the 0xE0 image")
    claim(b.addr(0x210) == 0x210, "a pointer inside image 0 is a plain offset")
    f = b.fwsec()
    claim(f["version"] == 3 and f["desc"] == 1536 + 0x100, "FWSEC found through the 'p' table")
    claim(f["image"] == f["desc"] + 0x2C + 4 * 96, "the image starts after the descriptor and its signatures")
    try:
        Bios(rom[:1024 + 512] + b"\xff" * 2048).fwsec()
        claim(False, "a ROM missing its 0xE0 image is refused")
    except ValueError:
        claim(True, "a ROM missing its 0xE0 image is refused")
    print("selftest: %s" % ("passed" if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(0 if selftest() else 1)
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    report(sys.argv[1])

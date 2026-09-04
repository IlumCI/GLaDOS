#!/usr/bin/env python3
"""Build the smallest real Linux binaries the loader has to cope with.

Why hand-assembled rather than compiled
---------------------------------------
There is no cross toolchain in this repository and adding one to produce a
forty-seven byte program would make the fixture depend on more than the thing
it tests. Every byte here is written out, so the file is exactly what the
loader will meet and nothing about it is a compiler's choice.

It is also the only way to get the *negative* fixtures. A compiler will not
emit a dynamically linked binary that is otherwise identical to a static one,
and "identical apart from the field under test" is the whole point of a
negative: the loader must refuse it for the stated reason and not because it
happened to differ some other way.

    python tools/mkelf.py out/hello.elf
    python tools/mkelf.py out/dyn.elf   --kind dynamic
    python tools/mkelf.py out/fixed.elf --kind fixed
    python tools/mkelf.py out/hello.elf --verify
"""

import argparse
import struct
import sys
from pathlib import Path

ET_EXEC, ET_DYN = 2, 3
PT_LOAD, PT_INTERP = 1, 3
PF_X, PF_W, PF_R = 1, 2, 4

EHDR = 64
PHENT = 56

MESSAGE = b"hello from ring 0\n"
# Distinctive, and not 0 or 1: an exit code the harness reads back has to be
# distinguishable from "the guest never ran" and from "something returned
# success by accident".
EXIT_CODE = 5


def code(msg_rva, entry_rva, exit_code):
    """write(1, msg, len); exit_group(EXIT_CODE); hlt

    RIP-relative throughout, so the program is genuinely position-independent
    rather than merely marked as such -- which matters, because the loader
    places it at whatever the heap returned and an absolute address would send
    it into the kernel.
    """
    out = bytearray()
    out += b"\x48\xc7\xc0" + struct.pack("<i", 1)            # mov rax, 1 (write)
    out += b"\x48\xc7\xc7" + struct.pack("<i", 1)            # mov rdi, 1 (stdout)
    lea_at = entry_rva + len(out)
    out += b"\x48\x8d\x35" + struct.pack("<i", 0)            # lea rsi, [rip+d] -- patched
    lea_end = entry_rva + len(out)
    out += b"\x48\xc7\xc2" + struct.pack("<i", len(MESSAGE))  # mov rdx, len
    out += b"\x0f\x05"                                        # syscall
    out += b"\x48\xc7\xc0" + struct.pack("<i", 231)          # mov rax, 231 (exit_group)
    out += b"\x48\xc7\xc7" + struct.pack("<i", exit_code)    # mov rdi, code
    out += b"\x0f\x05"                                        # syscall
    out += b"\xf4"                                            # hlt -- must be unreachable
    # The displacement is measured from the end of the lea, which is the one
    # thing about RIP-relative addressing that is easy to get wrong by four.
    disp = msg_rva - lea_end
    out[lea_at - entry_rva + 3:lea_at - entry_rva + 7] = struct.pack("<i", disp)
    return bytes(out), lea_end, disp


def build(kind="static"):
    interp = b"/lib64/ld-linux-x86-64.so.2\x00"
    phnum = 2 if kind == "dynamic" else 1
    entry = EHDR + PHENT * phnum
    # Lay the body out first so the message address is known before the code
    # that points at it is emitted.
    body_at = entry
    probe, _, _ = code(0, 0, EXIT_CODE)
    msg_rva = body_at + len(probe)
    text, lea_end, disp = code(msg_rva, entry, EXIT_CODE)

    body = text + MESSAGE
    if kind == "dynamic":
        interp_off = body_at + len(body)
        body += interp
    total = body_at + len(body)

    e_type = ET_EXEC if kind == "fixed" else ET_DYN
    # A fixed executable is placed where its headers insist, and 0x400000 is
    # where the toolchains put one. The loader must refuse it for that reason
    # and no other, so everything else about this file matches the static one.
    vbase = 0x400000 if kind == "fixed" else 0

    h = bytearray(EHDR)
    h[0:4] = b"\x7fELF"
    h[4], h[5], h[6] = 2, 1, 1                     # 64-bit, little-endian, v1
    h[7] = 0                                       # System V ABI
    struct.pack_into("<H", h, 16, e_type)
    struct.pack_into("<H", h, 18, 0x3E)            # x86-64
    struct.pack_into("<I", h, 20, 1)
    struct.pack_into("<Q", h, 24, vbase + entry)   # e_entry
    struct.pack_into("<Q", h, 32, EHDR)            # e_phoff
    struct.pack_into("<H", h, 52, EHDR)            # e_ehsize
    struct.pack_into("<H", h, 54, PHENT)           # e_phentsize
    struct.pack_into("<H", h, 56, phnum)           # e_phnum
    struct.pack_into("<H", h, 58, 64)              # e_shentsize

    phs = bytearray()
    ph = bytearray(PHENT)
    struct.pack_into("<I", ph, 0, PT_LOAD)
    struct.pack_into("<I", ph, 4, PF_R | PF_X)
    struct.pack_into("<Q", ph, 8, 0)               # p_offset
    struct.pack_into("<Q", ph, 16, vbase)          # p_vaddr
    struct.pack_into("<Q", ph, 24, vbase)          # p_paddr
    struct.pack_into("<Q", ph, 32, total)          # p_filesz
    struct.pack_into("<Q", ph, 40, total)          # p_memsz
    struct.pack_into("<Q", ph, 48, 0x1000)         # p_align
    phs += ph
    if kind == "dynamic":
        pi = bytearray(PHENT)
        struct.pack_into("<I", pi, 0, PT_INTERP)
        struct.pack_into("<I", pi, 4, PF_R)
        struct.pack_into("<Q", pi, 8, interp_off)
        struct.pack_into("<Q", pi, 16, vbase + interp_off)
        struct.pack_into("<Q", pi, 24, vbase + interp_off)
        struct.pack_into("<Q", pi, 32, len(interp))
        struct.pack_into("<Q", pi, 40, len(interp))
        struct.pack_into("<Q", pi, 48, 1)
        phs += pi

    blob = bytes(h) + bytes(phs) + body
    assert len(blob) == total, (len(blob), total)
    return blob, dict(entry=entry, msg=msg_rva, disp=disp, lea_end=lea_end, size=total)


def verify(path):
    """Read it back the way the kernel will, and check the facts it depends on.

    A separate reader, deliberately: checking the writer against itself would
    pass on any self-consistent mistake, which is exactly the class the ELF
    header invites.
    """
    b = Path(path).read_bytes()
    ok = True

    def claim(what, good):
        nonlocal ok
        print(("  ok   " if good else "  FAIL ") + what)
        if not good:
            ok = False

    claim("it begins with the ELF magic", b[:4] == b"\x7fELF")
    claim("it is 64-bit little-endian x86-64",
          b[4] == 2 and b[5] == 1 and struct.unpack_from("<H", b, 18)[0] == 0x3E)
    e_type = struct.unpack_from("<H", b, 16)[0]
    entry = struct.unpack_from("<Q", b, 24)[0]
    phoff = struct.unpack_from("<Q", b, 32)[0]
    phentsize = struct.unpack_from("<H", b, 54)[0]
    phnum = struct.unpack_from("<H", b, 56)[0]
    claim("the program header table is where the header says and fits the file",
          phoff + phnum * phentsize <= len(b))

    loads, interp = [], None
    for i in range(phnum):
        at = phoff + i * phentsize
        ty, _fl = struct.unpack_from("<II", b, at)
        off, va = struct.unpack_from("<QQ", b, at + 8)[0], struct.unpack_from("<Q", b, at + 16)[0]
        fsz, msz = struct.unpack_from("<Q", b, at + 32)[0], struct.unpack_from("<Q", b, at + 40)[0]
        if ty == PT_LOAD:
            loads.append((off, va, fsz, msz))
        elif ty == PT_INTERP:
            interp = b[off:off + fsz]
    claim("there is exactly one loadable segment", len(loads) == 1)
    off, va, fsz, msz = loads[0]
    claim("the segment covers the whole file and no more", off == 0 and fsz == len(b) and msz == fsz)
    claim("the entry point is inside the segment", va <= entry < va + msz)

    # The instruction the whole fixture turns on: the message must be where the
    # RIP-relative lea says it is. Off by four here and the guest writes
    # whatever follows, which reads as a working loader printing garbage.
    rel_entry = entry - va
    lea_at = rel_entry + 14
    claim("the lea is where the layout put it", b[lea_at:lea_at + 3] == b"\x48\x8d\x35")
    disp = struct.unpack_from("<i", b, lea_at + 3)[0]
    target = lea_at + 7 + disp
    claim("the lea points at the message and not four bytes off",
          b[target:target + len(MESSAGE)] == MESSAGE)
    claim("the program ends in hlt, so a syscall that returns is visible",
          b[target - 1:target] == b"\xf4")

    if e_type == ET_DYN and interp is None:
        claim("a static fixture names no interpreter", True)
    if interp is not None:
        claim("the dynamic fixture names an interpreter, NUL-terminated",
              interp.endswith(b"\x00"))
    if e_type == ET_EXEC:
        claim("the fixed fixture insists on a non-zero base", va != 0)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--kind", choices=["static", "dynamic", "fixed"], default="static")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if a.verify:
        sys.exit(0 if verify(a.out) else 1)

    blob, info = build(a.kind)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(blob)
    print("%s: %s, %d bytes, entry +%d, message at +%d (lea disp %d)"
          % (a.out, a.kind, info["size"], info["entry"], info["msg"], info["disp"]))
    if not verify(a.out):
        sys.exit(1)


if __name__ == "__main__":
    main()

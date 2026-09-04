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

MESSAGE = b"hello from ring 3\n"
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


# A very small emitter, for the one fixture that needs more than a straight
# line. Every instruction here is fixed-length, so laying the program out and
# patching the two displacements afterwards needs no second pass.
REG = dict(rax=0, rcx=1, rdx=2, rbx=3, rsp=4, rbp=5, rsi=6, rdi=7,
           r8=8, r9=9, r10=10, r11=11)


def _rex(w=1, r=0, x=0, b=0):
    return 0x40 | (w << 3) | (r << 2) | (x << 1) | b


def mov_imm(reg, imm):
    """mov reg, imm32 (sign-extended to 64)."""
    n = REG[reg]
    return bytes([_rex(b=n >> 3), 0xC7, 0xC0 | (n & 7)]) + struct.pack("<i", imm)


def mov_rr(dst, src):
    d, sr = REG[dst], REG[src]
    return bytes([_rex(r=sr >> 3, b=d >> 3), 0x89, 0xC0 | ((sr & 7) << 3) | (d & 7)])


def add_imm(reg, imm):
    n = REG[reg]
    return bytes([_rex(b=n >> 3), 0x81, 0xC0 | (n & 7)]) + struct.pack("<i", imm)


def store8(base, imm8):
    """mov byte [base], imm8 -- base must be a low register that is not rsp/rbp."""
    n = REG[base]
    assert n < 8 and n not in (4, 5)
    return bytes([0xC6, n & 7, imm8])


def load64(dst, base):
    """mov dst, [base]"""
    d, b = REG[dst], REG[base]
    assert b < 8 and b not in (4, 5)
    return bytes([_rex(r=d >> 3), 0x8B, ((d & 7) << 3) | (b & 7)])


def cmp_rr(a, b):
    ra, rb = REG[a], REG[b]
    return bytes([_rex(r=rb >> 3, b=ra >> 3), 0x39, 0xC0 | ((rb & 7) << 3) | (ra & 7)])


def lea_rip(reg, disp):
    n = REG[reg]
    return bytes([_rex(r=n >> 3), 0x8D, 0x05 | ((n & 7) << 3)]) + struct.pack("<i", disp)


def cmp_imm8(reg, imm):
    """cmp reg, imm8 sign-extended to 64."""
    n = REG[reg]
    return bytes([_rex(b=n >> 3), 0x83, 0xF8 | (n & 7), imm & 0xFF])


def load_abs(dst, addr):
    """mov dst, [addr] with an absolute 32-bit address.

    Needs a SIB byte: 64-bit mode has no plain absolute addressing, so
    modrm 0x04 selects SIB and SIB 0x25 selects "no base, no index, disp32".
    """
    n = REG[dst]
    return bytes([_rex(r=n >> 3), 0x8B, 0x04 | ((n & 7) << 3), 0x25]) + struct.pack("<I", addr)


SYSCALL = b"\x0f\x05"
HLT = b"\xf4"
MSG_OK = b"brk, mmap and arch_prctl all answered; FS read back\n"
MSG_GUARDED = b"both wild pointers were refused with EFAULT\n"
MSG_UNGUARDED = b"a wild pointer got through\n"
MSG_PROT = b"mprotect PROT_NONE took the page away from the kernel too\n"
MSG_NOPROT = b"the page was still reachable after PROT_NONE\n"
MSG_ESCAPED = b"the guest read kernel memory and lived\n"
MSG_BAD = b"FS did not read back\n"


def mem_code(entry_rva, ok_rva, bad_rva):
    """Ask for memory three different ways, and prove each one answered.

    The shape is deliberate: every call's result is *used* rather than merely
    received. The break is written to, the mapping is written to and read back,
    and FS is set and then read back through a different call -- so a stub that
    returned a plausible number without doing anything fails here rather than
    passing quietly.
    """
    c = bytearray()
    # brk(0), then grow by a page. rbx keeps the first break.
    c += mov_imm("rax", 12) + mov_imm("rdi", 0) + SYSCALL
    c += mov_rr("rbx", "rax")
    c += mov_rr("rdi", "rax") + add_imm("rdi", 4096)
    c += mov_imm("rax", 12) + SYSCALL
    # Write into the break region. If brk handed back an address that is not
    # real memory, this is where the machine stops.
    c += store8("rbx", 0x41)

    # mmap(NULL, 8192, RW, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
    c += mov_imm("rax", 9)
    c += mov_imm("rdi", 0)
    c += mov_imm("rsi", 8192)
    c += mov_imm("rdx", 3)
    c += mov_imm("r10", 0x22)
    c += mov_imm("r8", -1)
    c += mov_imm("r9", 0)
    c += SYSCALL
    c += mov_rr("rbx", "rax")          # rbx = the mapping

    # arch_prctl(ARCH_SET_FS, mapping), then ARCH_GET_FS into the mapping.
    c += mov_imm("rax", 158) + mov_imm("rdi", 0x1002) + mov_rr("rsi", "rbx") + SYSCALL
    c += mov_imm("rax", 158) + mov_imm("rdi", 0x1003) + mov_rr("rsi", "rbx") + SYSCALL

    # Did the base come back through a different call than set it?
    c += load64("rax", "rbx")
    c += cmp_rr("rax", "rbx")
    jne_at = len(c)
    c += b"\x0f\x85" + struct.pack("<i", 0)   # patched below

    # ok: give the mapping back, say so, exit 7.
    c += mov_rr("rdi", "rbx") + mov_imm("rsi", 8192) + mov_imm("rax", 11) + SYSCALL
    c += mov_imm("rax", 1) + mov_imm("rdi", 1)
    ok_lea = len(c)
    c += lea_rip("rsi", 0)
    ok_end = len(c)
    c += mov_imm("rdx", len(MSG_OK)) + SYSCALL
    c += mov_imm("rax", 231) + mov_imm("rdi", 7) + SYSCALL + HLT

    bad_at = len(c)
    c += mov_imm("rax", 1) + mov_imm("rdi", 1)
    bad_lea = len(c)
    c += lea_rip("rsi", 0)
    bad_end = len(c)
    c += mov_imm("rdx", len(MSG_BAD)) + SYSCALL
    c += mov_imm("rax", 231) + mov_imm("rdi", 8) + SYSCALL + HLT

    # Three displacements, all measured from the end of their own instruction,
    # which is the one thing about RIP-relative addressing that is easy to get
    # wrong by exactly four.
    struct.pack_into("<i", c, jne_at + 2, bad_at - (jne_at + 6))
    struct.pack_into("<i", c, ok_lea + 3, ok_rva - (entry_rva + ok_end))
    struct.pack_into("<i", c, bad_lea + 3, bad_rva - (entry_rva + bad_end))
    return bytes(c)


def rogue_code(entry_rva, ok_rva, bad_rva):
    """Hand the kernel two pointers nothing ever gave us.

    0x1000 is a real, mapped, kernel-owned page in an identity-mapped machine,
    which is exactly what makes it the right probe: it is not a wild address
    that would fault on its own, it is a *valid* address the guest has no
    business naming. An unchecked kernel would happily print whatever lives
    there, or post the FS base into it.

    Both calls must answer -14 (EFAULT). The program exits 9 when the guard
    held and 10 when it did not, so the harness reads a number rather than
    a paragraph.
    """
    c = bytearray()
    # write(1, 0x1000, 16) through a page nothing handed us
    c += mov_imm("rax", 1) + mov_imm("rdi", 1) + mov_imm("rsi", 0x1000)
    c += mov_imm("rdx", 16) + SYSCALL
    c += cmp_imm8("rax", -14)
    j1 = len(c)
    c += b"\x0f\x85" + struct.pack("<i", 0)

    # arch_prctl(ARCH_GET_FS, 0x1000): eight bytes posted into kernel memory
    c += mov_imm("rax", 158) + mov_imm("rdi", 0x1003) + mov_imm("rsi", 0x1000) + SYSCALL
    c += cmp_imm8("rax", -14)
    j2 = len(c)
    c += b"\x0f\x85" + struct.pack("<i", 0)

    c += mov_imm("rax", 1) + mov_imm("rdi", 1)
    ok_lea = len(c)
    c += lea_rip("rsi", 0)
    ok_end = len(c)
    c += mov_imm("rdx", len(MSG_GUARDED)) + SYSCALL
    c += mov_imm("rax", 231) + mov_imm("rdi", 9) + SYSCALL + HLT

    bad_at = len(c)
    c += mov_imm("rax", 1) + mov_imm("rdi", 1)
    bad_lea = len(c)
    c += lea_rip("rsi", 0)
    bad_end = len(c)
    c += mov_imm("rdx", len(MSG_UNGUARDED)) + SYSCALL
    c += mov_imm("rax", 231) + mov_imm("rdi", 10) + SYSCALL + HLT

    struct.pack_into("<i", c, j1 + 2, bad_at - (j1 + 6))
    struct.pack_into("<i", c, j2 + 2, bad_at - (j2 + 6))
    struct.pack_into("<i", c, ok_lea + 3, ok_rva - (entry_rva + ok_end))
    struct.pack_into("<i", c, bad_lea + 3, bad_rva - (entry_rva + bad_end))
    return bytes(c)


def protect_code(entry_rva, ok_rva, bad_rva):
    """Hide a page from yourself, then check the kernel agrees.

    The probe is arch_prctl(ARCH_GET_FS, p) rather than a write, because that
    call posts eight bytes *into* p and so needs the page present and
    writable. It answers 0 while the mapping is ordinary and -14 once the
    guest has mprotected it away.

    That pair is the whole point. A kernel that only checked "did the loader
    hand this range over" would still say yes after PROT_NONE, then read a
    page that is not present, and take the machine down on a pair of entirely
    legal calls.
    """
    c = bytearray()
    # mmap(NULL, 8192, RW, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
    c += mov_imm("rax", 9) + mov_imm("rdi", 0) + mov_imm("rsi", 8192)
    c += mov_imm("rdx", 3) + mov_imm("r10", 0x22) + mov_imm("r8", -1)
    c += mov_imm("r9", 0) + SYSCALL
    c += mov_rr("rbx", "rax")

    jumps = []

    def expect(imm):
        c.extend(cmp_imm8("rax", imm))
        jumps.append(len(c))
        c.extend(b"\x0f\x85" + struct.pack("<i", 0))

    # It is reachable to begin with.
    c += mov_imm("rax", 158) + mov_imm("rdi", 0x1003) + mov_rr("rsi", "rbx") + SYSCALL
    expect(0)
    # Take it away: mprotect(p, 4096, PROT_NONE)
    c += mov_rr("rdi", "rbx") + mov_imm("rsi", 4096) + mov_imm("rdx", 0)
    c += mov_imm("rax", 10) + SYSCALL
    expect(0)
    # And now the kernel must refuse to touch it.
    c += mov_imm("rax", 158) + mov_imm("rdi", 0x1003) + mov_rr("rsi", "rbx") + SYSCALL
    expect(-14)

    c += mov_imm("rax", 1) + mov_imm("rdi", 1)
    ok_lea = len(c)
    c += lea_rip("rsi", 0)
    ok_end = len(c)
    c += mov_imm("rdx", len(MSG_PROT)) + SYSCALL
    c += mov_imm("rax", 231) + mov_imm("rdi", 11) + SYSCALL + HLT

    bad_at = len(c)
    c += mov_imm("rax", 1) + mov_imm("rdi", 1)
    bad_lea = len(c)
    c += lea_rip("rsi", 0)
    bad_end = len(c)
    c += mov_imm("rdx", len(MSG_NOPROT)) + SYSCALL
    c += mov_imm("rax", 231) + mov_imm("rdi", 12) + SYSCALL + HLT

    for j in jumps:
        struct.pack_into("<i", c, j + 2, bad_at - (j + 6))
    struct.pack_into("<i", c, ok_lea + 3, ok_rva - (entry_rva + ok_end))
    struct.pack_into("<i", c, bad_lea + 3, bad_rva - (entry_rva + bad_end))
    return bytes(c)


def wild_code(entry_rva, msg_rva, _unused):
    """Read a kernel page directly, with no syscall in the way.

    0x1000 is mapped and belongs to the kernel, and its U bit is clear, so at
    ring 3 this instruction must fault. Nothing here checks a return value
    because there is nothing to return to: the kernel is expected to kill this
    guest at the load.

    Reaching the write below therefore means the guest read kernel memory and
    carried on, which is isolation not working. It says so and exits 13, and a
    run that reports 13 is a run that failed.
    """
    c = bytearray()
    c += load_abs("rax", 0x1000)
    c += mov_imm("rax", 1) + mov_imm("rdi", 1)
    lea_at = len(c)
    c += lea_rip("rsi", 0)
    lea_end = len(c)
    c += mov_imm("rdx", len(MSG_ESCAPED)) + SYSCALL
    c += mov_imm("rax", 231) + mov_imm("rdi", 13) + SYSCALL + HLT
    struct.pack_into("<i", c, lea_at + 3, msg_rva - (entry_rva + lea_end))
    return bytes(c)


def spin_code(entry_rva, _a, _b):
    """Loop forever, asking for nothing.

    Two bytes: `jmp .`. It makes no syscall, so nothing the kernel can refuse
    ever happens, and before the deadline existed this owned the machine with
    no key left to press.
    """
    return bytes([0xEB, 0xFE])


def build(kind="static"):
    interp = b"/lib64/ld-linux-x86-64.so.2\x00"
    phnum = 2 if kind == "dynamic" else 1
    entry = EHDR + PHENT * phnum
    # Lay the body out first so the message address is known before the code
    # that points at it is emitted.
    body_at = entry
    if kind == "spin":
        text = spin_code(entry, 0, 0)
        body = text
        msg_rva, disp, lea_end = body_at, 0, 0
    elif kind == "wild":
        probe = wild_code(0, 0, 0)
        msg_rva = body_at + len(probe)
        text = wild_code(entry, msg_rva, 0)
        assert len(text) == len(probe), (len(text), len(probe))
        body = text + MSG_ESCAPED
        disp, lea_end = 0, 0
    elif kind == "protect":
        probe = protect_code(0, 0, 0)
        ok_rva = body_at + len(probe)
        bad_rva = ok_rva + len(MSG_PROT)
        text = protect_code(entry, ok_rva, bad_rva)
        assert len(text) == len(probe), (len(text), len(probe))
        body = text + MSG_PROT + MSG_NOPROT
        msg_rva, disp, lea_end = ok_rva, 0, 0
    elif kind == "rogue":
        probe = rogue_code(0, 0, 0)
        ok_rva = body_at + len(probe)
        bad_rva = ok_rva + len(MSG_GUARDED)
        text = rogue_code(entry, ok_rva, bad_rva)
        assert len(text) == len(probe), (len(text), len(probe))
        body = text + MSG_GUARDED + MSG_UNGUARDED
        msg_rva, disp, lea_end = ok_rva, 0, 0
    elif kind == "memory":
        probe = mem_code(0, 0, 0)
        ok_rva = body_at + len(probe)
        bad_rva = ok_rva + len(MSG_OK)
        text = mem_code(entry, ok_rva, bad_rva)
        assert len(text) == len(probe), (len(text), len(probe))
        body = text + MSG_OK + MSG_BAD
        msg_rva, disp, lea_end = ok_rva, 0, 0
    else:
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
    if len(b) == entry + 2 and b[entry:] == bytes([0xEB, 0xFE]):
        claim("it is two bytes of jmp-to-self and nothing else", True)
        claim("it makes no syscall at all", SYSCALL not in b)
        return ok
    if MSG_ESCAPED in b:
        claim("it reads an absolute kernel address with no syscall in the way",
              bytes([0x48, 0x8B, 0x04, 0x25, 0x00, 0x10, 0x00, 0x00]) in b)
        claim("the fault is expected before the write, so reaching it is the failure",
              b.index(bytes([0x48, 0x8B, 0x04, 0x25])) < b.index(MSG_ESCAPED))
        claim("it exits 13 if it gets that far", bytes([0x48, 0xC7, 0xC7, 13, 0, 0, 0]) in b)
        return ok
    if MSG_PROT in b:
        claim("it maps, hides and re-probes the same page",
              b.count(SYSCALL) == 8)
        claim("it checks two answers against 0 and one against EFAULT",
              b.count(bytes([0x48, 0x83, 0xF8, 0x00])) == 2
              and b.count(bytes([0x48, 0x83, 0xF8, 0xF2])) == 1)
        claim("it exits 11 when the kernel agreed and 12 when it did not",
              bytes([0x48, 0xC7, 0xC7, 11, 0, 0, 0]) in b
              and bytes([0x48, 0xC7, 0xC7, 12, 0, 0, 0]) in b)
        return ok
    if MSG_GUARDED in b:
        claim("it probes two guest pointers it was never given",
              b.count(b"\x00\x10\x00\x00") >= 2)
        claim("it compares each answer against EFAULT",
              b.count(bytes([0x48, 0x83, 0xF8, 0xF2])) == 2)
        claim("it exits 9 when guarded and 10 when not",
              bytes([0x48, 0xC7, 0xC7, 9, 0, 0, 0]) in b
              and bytes([0x48, 0xC7, 0xC7, 10, 0, 0, 0]) in b)
        claim("it ends in hlt on both paths", b.count(b"\xf4") >= 2)
        return ok
    if MSG_OK in b:
        # The memory fixture: two messages, two leas, and the check that
        # matters is that both point at their own text.
        for msg in (MSG_OK, MSG_BAD):
            claim("a message is present and NUL-free: %r" % msg[:18],
                  msg in b and b"\x00" not in msg)
        # Ten: brk twice, mmap, arch_prctl twice, munmap, then a write and
        # an exit on each of the two branches. Counted rather than asserted
        # loosely, because an emitter that dropped one instruction would
        # still produce a file that parses, runs, and does less than it says.
        claim("it makes exactly the ten syscalls it is written to make",
              b.count(SYSCALL) == 10)
        claim("it ends in hlt on both paths", b.count(b"\xf4") >= 2)
        return ok
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
    ap.add_argument("--kind",
                    choices=["static", "dynamic", "fixed", "memory", "rogue",
                             "protect", "wild", "spin"],
                    default="static")
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

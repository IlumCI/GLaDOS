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
           r8=8, r9=9, r10=10, r11=11, r12=12, r13=13, r14=14, r15=15)


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


def load_stack(dst, disp8):
    """mov dst, [rsp+disp8]. Needs a SIB byte because rm=100 means SIB."""
    n = REG[dst]
    return bytes([_rex(r=n >> 3), 0x8B, 0x40 | ((n & 7) << 3) | 4, 0x24, disp8])


def sub_imm(reg, imm):
    n = REG[reg]
    return bytes([_rex(b=n >> 3), 0x81, 0xE8 | (n & 7)]) + struct.pack("<i", imm)


def jcc(cc, rel=0):
    """Two-byte conditional jump with a 32-bit displacement."""
    return bytes([0x0F, cc]) + struct.pack("<i", rel)


JL, JLE, JNE = 0x8C, 0x8E, 0x85


def jmp(rel=0):
    return bytes([0xE9]) + struct.pack("<i", rel)


def load_byte(dst, base, index):
    """movzx dst, byte [base + index].

    A disp8 of zero rather than mod=00, which costs a byte and removes a
    footgun: with mod=00 a SIB base of 101 means "no base, disp32 follows",
    so an addressing mode that works for every other register silently reads
    an absolute address when the base happens to be rbp.
    """
    d, bs, ix = REG[dst], REG[base], REG[index]
    assert ix != REG["rsp"], "an index of 100 means no index at all"
    return bytes([_rex(r=d >> 3, x=ix >> 3, b=bs >> 3), 0x0F, 0xB6,
                  0x40 | ((d & 7) << 3) | 4, ((ix & 7) << 3) | (bs & 7), 0x00])


def load_q(dst, base, disp8):
    d, bs = REG[dst], REG[base]
    return bytes([_rex(r=d >> 3, b=bs >> 3), 0x8B,
                  0x40 | ((d & 7) << 3) | (bs & 7), disp8 & 0xFF])


def store_q(base, disp8, src):
    sr, bs = REG[src], REG[base]
    return bytes([_rex(r=sr >> 3, b=bs >> 3), 0x89,
                  0x40 | ((sr & 7) << 3) | (bs & 7), disp8 & 0xFF])


def add_rr(dst, src):
    d, sr = REG[dst], REG[src]
    return bytes([_rex(r=sr >> 3, b=d >> 3), 0x01, 0xC0 | ((sr & 7) << 3) | (d & 7)])


def sub_rr(dst, src):
    d, sr = REG[dst], REG[src]
    return bytes([_rex(r=sr >> 3, b=d >> 3), 0x29, 0xC0 | ((sr & 7) << 3) | (d & 7)])


JE, JGE, JG = 0x84, 0x8D, 0x8F


def cmp_imm32(reg, imm):
    n = REG[reg]
    return bytes([_rex(b=n >> 3), 0x81, 0xF8 | (n & 7)]) + struct.pack("<i", imm)


def or_imm32(reg, imm):
    n = REG[reg]
    return bytes([_rex(b=n >> 3), 0x81, 0xC8 | (n & 7)]) + struct.pack("<i", imm)


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


MSG_USAGE = b"cat: needs a path\n"


def cat_code(entry_rva, usage_rva, _b):
    """Open argv[1], read it in chunks, write each chunk to stdout.

    A real `cat` in eighty-odd bytes. It is the smallest program that proves
    the whole filesystem projection at once: a path travels from the shell
    through argv into `openat`, the namespace resolves it, `read` advances a
    cursor across several calls, and end of file is a zero return rather than
    an error.

    argv[1] is loaded *before* the buffer is carved off the stack, because
    `sub rsp` moves the very thing being indexed.
    """
    c = bytearray()
    c += load_stack("rbx", 16)             # argv[1]
    c += mov_imm("rax", 0)
    c += cmp_rr("rbx", "rax")
    j_usage = len(c)
    c += jcc(JLE if False else 0x84)       # je -> no argument
    c += sub_imm("rsp", 512)               # a read buffer

    c += mov_imm("rax", 2) + mov_rr("rdi", "rbx")
    c += mov_imm("rsi", 0) + mov_imm("rdx", 0) + SYSCALL
    c += cmp_imm8("rax", 0)
    j_bad = len(c)
    c += jcc(JL)
    c += mov_rr("rbx", "rax")              # the descriptor

    loop = len(c)
    c += mov_imm("rax", 0) + mov_rr("rdi", "rbx")
    c += mov_rr("rsi", "rsp") + mov_imm("rdx", 256) + SYSCALL
    c += cmp_imm8("rax", 0)
    j_done = len(c)
    c += jcc(JLE)
    c += mov_rr("rdx", "rax")
    c += mov_imm("rax", 1) + mov_imm("rdi", 1) + mov_rr("rsi", "rsp") + SYSCALL
    j_loop = len(c)
    c += jmp()

    done = len(c)
    c += mov_imm("rax", 3) + mov_rr("rdi", "rbx") + SYSCALL
    c += mov_imm("rax", 231) + mov_imm("rdi", 3) + SYSCALL + HLT

    bad = len(c)
    c += mov_imm("rax", 231) + mov_imm("rdi", 14) + SYSCALL + HLT

    usage = len(c)
    c += mov_imm("rax", 1) + mov_imm("rdi", 2)
    u_lea = len(c)
    c += lea_rip("rsi", 0)
    u_end = len(c)
    c += mov_imm("rdx", len(MSG_USAGE)) + SYSCALL
    c += mov_imm("rax", 231) + mov_imm("rdi", 15) + SYSCALL + HLT

    struct.pack_into("<i", c, j_usage + 2, usage - (j_usage + 6))
    struct.pack_into("<i", c, j_bad + 2, bad - (j_bad + 6))
    struct.pack_into("<i", c, j_done + 2, done - (j_done + 6))
    struct.pack_into("<i", c, j_loop + 1, loop - (j_loop + 5))
    struct.pack_into("<i", c, u_lea + 3, usage_rva - (entry_rva + u_end))
    return bytes(c)


DOTS = b"/../etc/passwd\x00"


def fsabuse_code(entry_rva, empty_rva, dots_rva):
    """Ask the filesystem for every wrong thing, and report which one it got
    wrong as a bitmask.

    The other fixtures each prove one property. This one exists because a
    projection over a store that is not a filesystem fails at the *edges*, and
    an edge is exactly what no program written to use the thing normally will
    ever reach. So the negatives are the subject: a cursor past the end, a
    negative seek, a write to a read-only store, a path that walks upwards, a
    pointer the guest does not own, a directory call on a file, and the shell's
    redirect, which is the one arrangement in which a descriptor number and the
    thing behind it stop agreeing.

    A bitmask instead of an early exit, because the first failure is not the
    only interesting one and a fixture that stops at it takes a boot to report
    each subsequent one. Zero means every negative held.

    Every check runs against a descriptor that is still open afterwards, and
    two of them (`3` and `4`) are there to say so: a kernel that survived the
    abuse by quietly breaking the file would pass everything else.
    """
    c = bytearray()
    bit = [0]

    def fails_unless(cc, n):
        """Emit `jcc over; or rbp, 1 << n; over:` -- so rbp accumulates the
        checks that did *not* answer what Linux answers."""
        j = len(c)
        c.extend(jcc(cc))
        c.extend(or_imm32("rbp", 1 << n))
        struct.pack_into("<i", c, j + 2, len(c) - (j + 6))
        bit[0] = max(bit[0], n + 1)

    def call(nr, rdi=None, rsi=None, rdx=None):
        c.extend(mov_imm("rax", nr))
        for reg, v in (("rdi", rdi), ("rsi", rsi), ("rdx", rdx)):
            if v is None:
                continue
            c.extend(mov_rr(reg, v) if isinstance(v, str) else mov_imm(reg, v))
        c.extend(SYSCALL)

    c += load_stack("r12", 16)                 # argv[1], the path
    c += cmp_imm8("r12", 0)
    j_bad = len(c)
    c += jcc(JE)

    c += sub_imm("rsp", 256)
    c += mov_rr("r13", "rsp")                  # a scratch buffer
    c += mov_imm("rbp", 0)                     # the failure mask

    call(2, "r12", 0, 0)                       # open(path, O_RDONLY)
    c += cmp_imm8("rax", 0)
    j_bad2 = len(c)
    c += jcc(JL)
    c += mov_rr("rbx", "rax")

    # 0: seeking past the end is legal and answers the position asked for.
    call(8, "rbx", 0x100000, 0)
    c += cmp_imm32("rax", 0x100000)
    fails_unless(JE, 0)

    # 1: and reading there answers zero. This is the one that mattered: the
    # cursor indexed a slice directly, so two legal calls panicked in ring 0.
    call(0, "rbx", "r13", 16)
    c += cmp_imm8("rax", 0)
    fails_unless(JE, 1)

    # 2: a seek before the start is EINVAL rather than a huge unsigned cursor.
    call(8, "rbx", -1, 0)
    c += cmp_imm8("rax", -22)
    fails_unless(JE, 2)

    # 3, 4: the descriptor still works, so the abuse above broke nothing.
    call(8, "rbx", 0, 0)
    c += cmp_imm8("rax", 0)
    fails_unless(JE, 3)
    call(0, "rbx", "r13", 16)
    c += cmp_imm8("rax", 0)
    fails_unless(JG, 4)

    # 5: opening for writing is refused, since a write here is a new root hash.
    call(2, "r12", 1, 0)
    c += cmp_imm8("rax", -13)
    fails_unless(JE, 5)

    # 6: an empty path is ENOENT, not a descriptor for the working directory.
    c += mov_imm("rax", 2)
    e_lea = len(c)
    c += lea_rip("rdi", 0)
    e_end = len(c)
    c += mov_imm("rsi", 0) + mov_imm("rdx", 0) + SYSCALL
    c += cmp_imm8("rax", -2)
    fails_unless(JE, 6)

    # 7: and one that walks upwards is refused rather than normalised.
    c += mov_imm("rax", 2)
    d_lea = len(c)
    c += lea_rip("rdi", 0)
    d_end = len(c)
    c += mov_imm("rsi", 0) + mov_imm("rdx", 0) + SYSCALL
    c += cmp_imm8("rax", -2)
    fails_unless(JE, 7)

    # 8: a read into a page the guest does not own is EFAULT. 0x1000 is real,
    # mapped and the kernel's, which is the point -- a wild address would fault
    # on its own and prove nothing about the check.
    call(0, "rbx", 0x1000, 16)
    c += cmp_imm8("rax", -14)
    fails_unless(JE, 8)

    # 9: a directory call on a file is ENOTDIR.
    call(217, "rbx", "r13", 128)
    c += cmp_imm8("rax", -20)
    fails_unless(JE, 9)

    # 10, 11: fstat answers, and says regular file. st_uid follows st_mode and
    # is zero, so eight bytes at 24 is the mode alone.
    call(5, "rbx", "r13", None)
    c += cmp_imm8("rax", 0)
    fails_unless(JE, 10)
    c += load_q("rax", "r13", 24)
    c += cmp_imm32("rax", 0o100644)
    fails_unless(JE, 11)

    # 12, 13, 14: the redirect. Closing stdout and opening a file must hand
    # back descriptor 1, and a write to it must be EBADF -- the arrangement in
    # which a call that only looked at the *number* printed a guest's
    # redirected output to the terminal and reported success.
    call(3, 1, None, None)
    c += cmp_imm8("rax", 0)
    fails_unless(JE, 12)
    call(2, "r12", 0, 0)
    c += cmp_imm8("rax", 1)
    fails_unless(JE, 13)
    call(1, 1, "r13", 4)
    c += cmp_imm8("rax", -9)
    fails_unless(JE, 14)

    c += mov_imm("rax", 231) + mov_rr("rdi", "rbp") + SYSCALL + HLT

    bad = len(c)
    c += mov_imm("rax", 231) + mov_imm("rdi", 255) + SYSCALL + HLT

    struct.pack_into("<i", c, j_bad + 2, bad - (j_bad + 6))
    struct.pack_into("<i", c, j_bad2 + 2, bad - (j_bad2 + 6))
    struct.pack_into("<i", c, e_lea + 3, empty_rva - (entry_rva + e_end))
    struct.pack_into("<i", c, d_lea + 3, dots_rva - (entry_rva + d_end))
    return bytes(c)


MSG_GREP = b"grep: needs a pattern and a path\n"


def grep_code(entry_rva, usage_rva, _b):
    """Print every line of argv[2] containing argv[1].

    Where `cat` proves a path can travel from the shell into the namespace and
    come back as bytes, this proves the bytes are *right*: a naive substring
    search over a line buffer answers differently for every one-byte change in
    the file, so a read that lost a byte, doubled one, or stopped early shows
    up as a wrong set of lines rather than as plausible output.

    It is a real frame: rbp is the buffer, and the locals live underneath it.
    The alternative was juggling seven live values in registers across a
    `write`, and the ABI only promises rbx, rbp, r10-r12 and rsi/rdi/rdx back
    -- of which two are the arguments the write needs.

    Exit codes are grep's own: 0 when something matched, 1 when nothing did,
    2 on an error. A program answering 0 for "no matches" would make the
    difference between an empty file and a working search invisible.
    """
    c = bytearray()
    c += load_stack("rbx", 16)             # argv[1], the pattern
    c += load_stack("rdi", 24)             # argv[2], the path
    c += cmp_imm8("rbx", 0)
    j_u1 = len(c)
    c += jcc(JE)
    c += cmp_imm8("rdi", 0)
    j_u2 = len(c)
    c += jcc(JE)

    c += mov_imm("rax", 2) + mov_imm("rsi", 0) + mov_imm("rdx", 0) + SYSCALL
    c += cmp_imm8("rax", 0)
    j_bad = len(c)
    c += jcc(JL)
    c += mov_rr("r9", "rax")               # the descriptor

    # The buffer, then the locals below it. rsp is left below both, so the
    # frame is whole and nothing a syscall does can reach into it.
    c += sub_imm("rsp", 4096)
    c += mov_rr("rbp", "rsp")
    c += sub_imm("rsp", 64)

    c += mov_imm("rax", 0) + mov_rr("rdi", "r9") + mov_rr("rsi", "rbp")
    c += mov_imm("rdx", 4096) + SYSCALL
    c += mov_rr("r10", "rax")              # bytes read

    # Close before scanning: the file is in hand, and holding a descriptor
    # open across the search would prove nothing the open already proved.
    c += mov_imm("rax", 3) + mov_rr("rdi", "r9") + SYSCALL

    c += mov_imm("rax", 0) + store_q("rbp", -16, "rax")   # matches so far
    c += cmp_imm8("r10", 0)
    j_empty = len(c)
    c += jcc(JLE)

    c += mov_imm("rsi", 0)                 # the line starts here
    line = len(c)
    c += cmp_rr("rsi", "r10")
    j_fin = len(c)
    c += jcc(JGE)
    c += mov_rr("rdi", "rsi")
    nl = len(c)
    c += cmp_rr("rdi", "r10")
    j_have1 = len(c)
    c += jcc(JGE)
    c += load_byte("r8", "rbp", "rdi")
    c += cmp_imm8("r8", 10)
    j_have2 = len(c)
    c += jcc(JE)
    c += add_imm("rdi", 1)
    j_nl = len(c)
    c += jmp()

    # The line is [rsi, rdi). Try the pattern at every start inside it.
    have = len(c)
    c += mov_rr("rdx", "rsi")
    tryat = len(c)
    c += cmp_rr("rdx", "rdi")
    j_nextline1 = len(c)
    c += jcc(JG)
    c += mov_rr("rax", "rdx") + mov_imm("rcx", 0)
    cmpl = len(c)
    c += load_byte("r8", "rbx", "rcx")
    c += cmp_imm8("r8", 0)                 # off the end of the pattern: a hit
    j_match = len(c)
    c += jcc(JE)
    c += cmp_rr("rax", "rdi")              # off the end of the line: no hit
    j_nextat1 = len(c)
    c += jcc(JGE)
    c += load_byte("r9", "rbp", "rax")
    c += cmp_rr("r8", "r9")
    j_nextat2 = len(c)
    c += jcc(JNE)
    c += add_imm("rax", 1) + add_imm("rcx", 1)
    j_cmpl = len(c)
    c += jmp()

    nextat = len(c)
    c += add_imm("rdx", 1)
    j_tryat = len(c)
    c += jmp()

    match = len(c)
    c += store_q("rbp", -24, "rsi") + store_q("rbp", -32, "rdi")
    c += mov_rr("rdx", "rdi") + sub_rr("rdx", "rsi")
    c += cmp_rr("rdi", "r10")              # a final line carries no newline
    j_nonl = len(c)
    c += jcc(JGE)
    c += add_imm("rdx", 1)
    nonl = len(c)
    c += add_rr("rsi", "rbp")
    c += mov_imm("rax", 1) + mov_imm("rdi", 1) + SYSCALL
    c += load_q("rax", "rbp", -16) + add_imm("rax", 1) + store_q("rbp", -16, "rax")
    c += load_q("rsi", "rbp", -24) + load_q("rdi", "rbp", -32)

    # One line printed per matching line, not per occurrence, which is what
    # grep does and why the match arm falls through to here.
    nextline = len(c)
    c += mov_rr("rsi", "rdi") + add_imm("rsi", 1)
    j_line = len(c)
    c += jmp()

    fin = len(c)
    c += load_q("rax", "rbp", -16)
    c += cmp_imm8("rax", 0)
    j_none = len(c)
    c += jcc(JE)
    c += mov_imm("rax", 231) + mov_imm("rdi", 0) + SYSCALL + HLT

    none = len(c)
    c += mov_imm("rax", 231) + mov_imm("rdi", 1) + SYSCALL + HLT

    bad = len(c)
    c += mov_imm("rax", 231) + mov_imm("rdi", 2) + SYSCALL + HLT

    usage = len(c)
    c += mov_imm("rax", 1) + mov_imm("rdi", 2)
    u_lea = len(c)
    c += lea_rip("rsi", 0)
    u_end = len(c)
    c += mov_imm("rdx", len(MSG_GREP)) + SYSCALL
    c += mov_imm("rax", 231) + mov_imm("rdi", 2) + SYSCALL + HLT

    for site, target in ((j_u1, usage), (j_u2, usage), (j_bad, bad),
                         (j_empty, fin), (j_fin, fin), (j_have1, have),
                         (j_have2, have), (j_nextline1, nextline),
                         (j_match, match), (j_nextat1, nextat),
                         (j_nextat2, nextat), (j_nonl, nonl), (j_none, none)):
        struct.pack_into("<i", c, site + 2, target - (site + 6))
    for site, target in ((j_nl, nl), (j_cmpl, cmpl), (j_tryat, tryat),
                         (j_line, line)):
        struct.pack_into("<i", c, site + 1, target - (site + 5))
    struct.pack_into("<i", c, u_lea + 3, usage_rva - (entry_rva + u_end))
    return bytes(c)


def build(kind="static"):
    interp = b"/lib64/ld-linux-x86-64.so.2\x00"
    phnum = 2 if kind == "dynamic" else 1
    entry = EHDR + PHENT * phnum
    # Lay the body out first so the message address is known before the code
    # that points at it is emitted.
    body_at = entry
    if kind == "fsabuse":
        probe = fsabuse_code(0, 0, 0)
        empty_rva = body_at + len(probe)
        text = fsabuse_code(entry, empty_rva, empty_rva + 1)
        assert len(text) == len(probe), (len(text), len(probe))
        body = text + b"\x00" + DOTS
        msg_rva, disp, lea_end = empty_rva, 0, 0
    elif kind == "grep":
        probe = grep_code(0, 0, 0)
        usage_rva = body_at + len(probe)
        text = grep_code(entry, usage_rva, 0)
        assert len(text) == len(probe), (len(text), len(probe))
        body = text + MSG_GREP
        msg_rva, disp, lea_end = usage_rva, 0, 0
    elif kind == "cat":
        probe = cat_code(0, 0, 0)
        usage_rva = body_at + len(probe)
        text = cat_code(entry, usage_rva, 0)
        assert len(text) == len(probe), (len(text), len(probe))
        body = text + MSG_USAGE
        msg_rva, disp, lea_end = usage_rva, 0, 0
    elif kind == "spin":
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
    if DOTS in b:
        claim("it opens the path it was handed, read-only",
              bytes([0x4C, 0x8B, 0x64, 0x24, 0x10]) in b            # r12 <- argv[1]
              and bytes([0x48, 0xC7, 0xC0, 2, 0, 0, 0]) in b)
        claim("it seeks a megabyte past the end and expects that to be allowed",
              bytes([0x48, 0xC7, 0xC6, 0x00, 0x00, 0x10, 0x00]) in b
              and bytes([0x48, 0x3D]) not in b)                     # cmp via 81 /7
        claim("it asks for every refusal by its own errno",
              all(bytes([0x48, 0x83, 0xF8, e & 0xFF]) in b
                  for e in (-22, -13, -2, -14, -20, -9)))
        claim("it names a page it does not own, and a real one",
              bytes([0x48, 0xC7, 0xC6, 0x00, 0x10, 0x00, 0x00]) in b)
        claim("it checks st_mode against S_IFREG|0644 read out of the block",
              bytes([0x49, 0x8B, 0x45, 0x18]) in b                  # [r13+24]
              and struct.pack("<i", 0o100644) in b)
        claim("fifteen checks, each folding one bit into the mask it exits with",
              sum(1 for i in range(15)
                  if bytes([0x48, 0x81, 0xCD]) + struct.pack("<i", 1 << i) in b) == 15)
        claim("it exits with the mask rather than a constant, and 255 on misuse",
              bytes([0x48, 0x89, 0xEF]) in b                        # rdi <- rbp
              and bytes([0x48, 0xC7, 0xC7, 255, 0, 0, 0]) in b)
        return ok
    if MSG_GREP in b:
        claim("it reads a pattern and a path off the stack before moving rsp",
              bytes([0x48, 0x8B, 0x5C, 0x24, 0x10]) in b
              and bytes([0x48, 0x8B, 0x7C, 0x24, 0x18]) in b)
        claim("it builds a frame: 4096 bytes of buffer, then locals under it",
              bytes([0x48, 0x81, 0xEC, 0x00, 0x10, 0x00, 0x00]) in b
              and bytes([0x48, 0x89, 0xE5]) in b
              and bytes([0x48, 0x81, 0xEC, 0x40, 0x00, 0x00, 0x00]) in b)
        claim("it reads the file and the pattern a byte at a time, indexed",
              bytes([0x4C, 0x0F, 0xB6, 0x44, 0x3D, 0x00]) in b     # [rbp+rdi]
              and bytes([0x4C, 0x0F, 0xB6, 0x44, 0x0B, 0x00]) in b # [rbx+rcx]
              and bytes([0x4C, 0x0F, 0xB6, 0x4C, 0x05, 0x00]) in b)# [rbp+rax]
        claim("it splits on 0x0a and nothing else",
              bytes([0x49, 0x83, 0xF8, 0x0A]) in b)
        claim("it saves the two loop registers a write would take, and reloads them",
              bytes([0x48, 0x89, 0x75, 0xE8]) in b     # [rbp-24] <- rsi
              and bytes([0x48, 0x89, 0x7D, 0xE0]) in b # [rbp-32] <- rdi
              and bytes([0x48, 0x8B, 0x75, 0xE8]) in b
              and bytes([0x48, 0x8B, 0x7D, 0xE0]) in b)
        claim("it makes the calls a grep makes: open, read, close, write, exit",
              b.count(SYSCALL) == 9)
        claim("it answers grep's own codes: 0 matched, 1 did not, 2 could not",
              bytes([0x48, 0xC7, 0xC7, 0, 0, 0, 0]) in b
              and bytes([0x48, 0xC7, 0xC7, 1, 0, 0, 0]) in b
              and bytes([0x48, 0xC7, 0xC7, 2, 0, 0, 0]) in b)
        return ok
    if MSG_USAGE in b:
        claim("it reads argv[1] off the stack before moving rsp",
              bytes([0x48, 0x8B, 0x5C, 0x24, 0x10]) in b)
        claim("it carves a read buffer out of the stack",
              bytes([0x48, 0x81, 0xEC, 0x00, 0x02, 0x00, 0x00]) in b)
        # open, read, write, close, exit on the good path; exit on the failed
        # open; write and exit on the no-argument path. Counted because an
        # emitter that dropped one still produces a file that runs and does
        # less than it claims.
        claim("it makes the eight calls its three paths add up to",
              b.count(SYSCALL) == 8)
        claim("it exits 3 on success, 14 when the open fails, 15 with no argument",
              bytes([0x48, 0xC7, 0xC7, 3, 0, 0, 0]) in b
              and bytes([0x48, 0xC7, 0xC7, 14, 0, 0, 0]) in b
              and bytes([0x48, 0xC7, 0xC7, 15, 0, 0, 0]) in b)
        return ok
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
                             "protect", "wild", "spin", "cat", "grep",
                             "fsabuse"],
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

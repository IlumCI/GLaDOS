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
    python tools/mkelf.py out/prog.elf  --kind interp   # wants /tmp/loader
    python tools/mkelf.py out/loader.elf --kind loader  # and is what runs first
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


def load_d(dst, base, disp8):
    """mov dst32, [base+disp8]. No REX.W, so the upper half is zeroed.

    Thirty-two bits because the thing worth reading four bytes of is an ELF
    magic, and a 64-bit load would drag in the class and endianness bytes after
    it -- which are also fixed, but comparing eight bytes against a constant
    tests four fields while claiming to test one.
    """
    d, bs = REG[dst], REG[base]
    pre = bytes([_rex(w=0, r=d >> 3, b=bs >> 3)]) if (d >> 3 or bs >> 3) else b""
    return pre + bytes([0x8B, 0x40 | ((d & 7) << 3) | (bs & 7), disp8 & 0xFF])


def store_d(base, disp8, src):
    """mov [base+disp8], src32. No REX.W, so four bytes rather than eight."""
    sr, bs = REG[src], REG[base]
    pre = bytes([_rex(w=0, r=sr >> 3, b=bs >> 3)]) if (sr >> 3 or bs >> 3) else b""
    return pre + bytes([0x89, 0x40 | ((sr & 7) << 3) | (bs & 7), disp8 & 0xFF])


def shl_cl(reg):
    """shl reg, cl. The one instruction a variable pixel layout needs.

    A framebuffer program is told where each colour channel sits and shifts by
    that, rather than assuming. Assuming is how red and blue get swapped, and
    a swapped picture is the failure that looks like art.
    """
    n = REG[reg]
    return bytes([_rex(b=n >> 3), 0xD3, 0xE0 | (n & 7)])


def shl_imm(reg, imm8):
    n = REG[reg]
    return bytes([_rex(b=n >> 3), 0xC1, 0xE0 | (n & 7), imm8 & 0xFF])


def shr_imm(reg, imm8):
    n = REG[reg]
    return bytes([_rex(b=n >> 3), 0xC1, 0xE8 | (n & 7), imm8 & 0xFF])


def imul_rr(dst, src):
    """imul dst, src. The screen's area is a multiplication and there is no
    other way to get one."""
    d, sr = REG[dst], REG[src]
    return bytes([_rex(r=d >> 3, b=sr >> 3), 0x0F, 0xAF, 0xC0 | ((d & 7) << 3) | (sr & 7)])


def movss_rip(xmm, disp=0):
    """movss xmm, [rip+disp32]. Four floats is what `glClearColor` wants.

    The System V convention puts floating-point arguments in `xmm0`-`xmm7`,
    which is the one thing in this file that cannot be done with the integer
    registers -- and a GL call that takes colours takes floats.
    """
    return bytes([0xF3, 0x0F, 0x10, 0x05 | (xmm << 3)]) + struct.pack("<i", disp)


def call_rip(disp=0):
    """call [rip+disp32]: an indirect call through a GOT slot.

    How a dynamically linked program reaches a library, with no PLT stub in
    between. That works here because the loader is asked to bind everything at
    load time, so the slot holds the real address before anything runs.
    """
    return bytes([0xFF, 0x15]) + struct.pack("<i", disp)


def jmp_reg(reg):
    """jmp reg. The one instruction an interpreter cannot do without."""
    n = REG[reg]
    pre = bytes([_rex(w=0, b=n >> 3)]) if n >> 3 else b""
    return pre + bytes([0xFF, 0xE0 | (n & 7)])


SYSCALL = b"\x0f\x05"
HLT = b"\xf4"
MSG_OK = b"brk, mmap and arch_prctl all answered; FS read back\n"
MSG_GUARDED = b"both wild pointers were refused with EFAULT\n"
MSG_UNGUARDED = b"a wild pointer got through\n"
MSG_PROT = b"mprotect PROT_NONE took the page away from the kernel too\n"
MSG_NOPROT = b"the page was still reachable after PROT_NONE\n"
MSG_ESCAPED = b"the guest read kernel memory and lived\n"
MSG_BAD = b"FS did not read back\n"
MSG_LD = b"ld: an interpreter ran, with a base and an entry\n"
# Where `--kind interp` says its interpreter is. A path in the namespace
# rather than a real one, because the point is to exercise the mechanism
# with something whose behaviour is known to the byte.
INTERP_FIXTURE = b"/tmp/loader\x00"
INTERP_REAL = b"/lib64/ld-linux-x86-64.so.2\x00"


def maps_code(entry_rva, _a, _b):
    """Ask for memory the two ways `ld.so` does, and prove each answered.

    `mmap` served exactly one shape for a long time: anonymous, no address, no
    file. That is enough for an allocator and nothing else, and it is the pair
    of refusals a dynamic linker meets on its first two calls -- it reserves a
    span, then writes each segment of a library over it with `MAP_FIXED`, and
    every one of those segments is file-backed.

    Eleven checks, each folding one bit into a mask, and the exit code is the
    mask -- `fsabuse`'s idiom, for its reason: zero means every one of them
    answered what Linux answers, and any other number says exactly which did
    not. The negatives are most of the list, because the positives are the
    calls a program makes when everything is fine and the refusals are what
    nothing written to use this normally will ever reach.

    It maps **its own file**, through `argv[0]`, so the fixture needs nothing
    staged beside it and the bytes it checks are ones it can be certain of: a
    file-backed mapping of an ELF begins with an ELF header or the mapping is
    not of that file.
    """
    PROT_READ = 1
    MAP_SHARED, MAP_PRIVATE, MAP_FIXED, MAP_ANON = 1, 2, 0x10, 0x20
    EBADF, ENODEV, EINVAL = -9, -19, -22
    out = bytearray()

    def fwd(at):
        out[at + 2:at + 6] = struct.pack("<i", len(out) - (at + 6))

    def mmap_call(addr_reg, addr_imm, length, prot, flags, fd_reg, fd_imm):
        b = bytearray()
        b += mov_imm("rax", 9)
        b += mov_rr("rdi", addr_reg) if addr_reg else mov_imm("rdi", addr_imm)
        b += mov_imm("rsi", length)
        b += mov_imm("rdx", prot)
        b += mov_imm("r10", flags)
        b += mov_rr("r8", fd_reg) if fd_reg else mov_imm("r8", fd_imm)
        b += mov_imm("r9", 0)
        b += SYSCALL
        return bytes(b)

    def fold(bit, cc):
        """Fold `bit` unless the comparison just made took `cc`."""
        ok = len(out)
        out.extend(jcc(cc, 0))
        out.extend(or_imm32("rbp", 1 << bit))
        fwd(ok)

    out += mov_imm("rbp", 0)

    # 0. The reservation. Two pages, so the fixed mapping below lands inside it
    #    rather than replacing the whole thing.
    out += mmap_call(None, 0, 0x2000, 3, MAP_PRIVATE | MAP_ANON, None, -1)
    out += mov_rr("r13", "rax")
    out += cmp_imm32("rax", 0)
    fold(0, JG)

    # 1. And laying a page of it out again in place, which is the call that was
    #    refused outright and the one a linker cannot do without.
    out += mmap_call("r13", 0, 0x1000, 3, MAP_PRIVATE | MAP_ANON | MAP_FIXED, None, -1)
    out += cmp_rr("rax", "r13")
    fold(1, JE)

    # 2. It is real memory afterwards, not merely an address.
    out += mov_imm("rcx", 0x5A5A)
    out += store_q("r13", 0, "rcx")
    out += load_q("rdx", "r13", 0)
    out += cmp_rr("rdx", "rcx")
    fold(2, JE)

    # 3. Its own file, by the path it was invoked with.
    out += mov_imm("rax", 2)
    out += load_stack("rdi", 8)                      # argv[0]
    out += mov_imm("rsi", 0)
    out += mov_imm("rdx", 0)
    out += SYSCALL
    out += mov_rr("r14", "rax")
    out += cmp_imm32("rax", 0)
    fold(3, JGE)

    # 4. A private file mapping.
    out += mmap_call(None, 0, 0x1000, PROT_READ, MAP_PRIVATE, "r14", 0)
    out += mov_rr("r15", "rax")
    out += cmp_imm32("rax", 0)
    past = len(out)
    out += jcc(JG, 0)
    out += or_imm32("rbp", 1 << 4)
    out += or_imm32("rbp", 1 << 5)
    skip = len(out)
    out += jmp(0)
    fwd(past)

    # 5. And it holds this file. Reading through a mapping that was refused
    #    would fault, so this is only reached when 4 held -- which is why the
    #    branch above folds both bits and jumps over it.
    out += load_d("rcx", "r15", 0)
    out += cmp_imm32("rcx", 0x464C457F)
    fold(5, JE)
    out[skip + 1:skip + 5] = struct.pack("<i", len(out) - (skip + 5))

    # 6. A shared writable file mapping is refused, because honouring it means
    #    writing back into a store keyed by content.
    out += mmap_call(None, 0, 0x1000, 3, MAP_SHARED, "r14", 0)
    out += cmp_imm8("rax", ENODEV)
    fold(6, JE)

    # 7. An unaligned fixed address is refused rather than rounded. Rounding
    #    would put a library's segment a page off its own headers.
    out += mmap_call(None, 0x1234, 0x1000, 3, MAP_PRIVATE | MAP_ANON | MAP_FIXED, None, -1)
    out += cmp_imm8("rax", EINVAL)
    fold(7, JE)

    # 8. And so is a fixed address of zero, which is how a null pointer arrives
    #    at this call.
    out += mmap_call(None, 0, 0x1000, 3, MAP_PRIVATE | MAP_ANON | MAP_FIXED, None, -1)
    out += cmp_imm8("rax", EINVAL)
    fold(8, JE)

    # 9. A file mapping on a descriptor nobody opened.
    out += mmap_call(None, 0, 0x1000, PROT_READ, MAP_PRIVATE, None, 99)
    out += cmp_imm8("rax", EBADF)
    fold(9, JE)

    # 10. The reservation comes back whole, which also says the fixed mapping
    #     laid over part of it left one record rather than two.
    out += mov_imm("rax", 11)
    out += mov_rr("rdi", "r13")
    out += mov_imm("rsi", 0x2000)
    out += SYSCALL
    out += cmp_imm32("rax", 0)
    fold(10, JE)

    out += mov_imm("rax", 231)
    out += mov_rr("rdi", "rbp")
    out += SYSCALL
    out += HLT
    return bytes(out)


def loader_code(entry_rva, msg_rva, _b):
    """Stand in for ld.so: read the aux vector, check it, jump to the program.

    This is the smallest thing that is genuinely an interpreter. It does no
    linking, because linking is not what the kernel side of `PT_INTERP` gets
    wrong -- what it gets wrong is the three numbers, and all three are silent
    when wrong. `AT_ENTRY` swapped with the interpreter's own entry gives a
    linker that re-enters itself forever. `AT_BASE` absent gives an `ET_DYN`
    that cannot find its own `_DYNAMIC` and relocates against zero. And a
    kernel that jumped to the program rather than the interpreter would run
    the program correctly, which looks exactly like success.

    So each is checked and each has its own exit code, and the last thing it
    does is jump where it was told. **`rsp` is never touched**: the program on
    the other side of that jump expects to find `argc` where the kernel left
    it, so the walk uses `rbx` and gives the stack back untouched.
    """
    AT_NULL, AT_BASE, AT_ENTRY = 0, 7, 9
    out = bytearray()

    def patch_fwd(at):
        """Point a jcc emitted at `at` here."""
        out[at + 2:at + 6] = struct.pack("<i", len(out) - (at + 6))

    def patch_back(at, to):
        out[at + 1:at + 5] = struct.pack("<i", to - (at + 5))

    out += mov_rr("rbx", "rsp")
    out += add_imm("rbx", 8)                    # past argc
    # argv then envp, each a NULL-terminated array of pointers, so one loop
    # shape does both and the count in argc is never needed.
    for _ in range(2):
        top = len(out)
        out += load_q("rcx", "rbx", 0)
        out += cmp_imm32("rcx", 0)
        done = len(out)
        out += jcc(JE, 0)
        out += add_imm("rbx", 8)
        back = len(out)
        out += jmp(0)
        patch_back(back, top)
        patch_fwd(done)
        out += add_imm("rbx", 8)                # step over the NULL itself

    # rbx now stands on the first aux key. Walk it by key rather than by
    # position, which is the only way that is right: the kernel is free to
    # order these however it likes and a reader that counted would break the
    # day one was added.
    out += mov_imm("r12", 0)                    # AT_ENTRY
    out += mov_imm("r13", 0)                    # AT_BASE
    top = len(out)
    out += load_q("rcx", "rbx", 0)
    out += cmp_imm32("rcx", AT_NULL)
    done = len(out)
    out += jcc(JE, 0)
    out += cmp_imm32("rcx", AT_BASE)
    skip = len(out)
    out += jcc(JNE, 0)
    out += load_q("r13", "rbx", 8)
    patch_fwd(skip)
    out += cmp_imm32("rcx", AT_ENTRY)
    skip = len(out)
    out += jcc(JNE, 0)
    out += load_q("r12", "rbx", 8)
    patch_fwd(skip)
    out += add_imm("rbx", 16)
    back = len(out)
    out += jmp(0)
    patch_back(back, top)
    patch_fwd(done)

    def exit_with(n):
        """exit_group(n), for a check that did not hold."""
        return mov_imm("rax", 231) + mov_imm("rdi", n) + SYSCALL

    # AT_BASE present.
    out += cmp_imm32("r13", 0)
    ok = len(out)
    out += jcc(JNE, 0)
    out += exit_with(21)
    patch_fwd(ok)
    # And pointing at the interpreter's own ELF header, which is the check that
    # makes it a base rather than a number: the first loadable segment starts
    # at file offset zero, so the header is in memory at exactly this address.
    out += load_d("rcx", "r13", 0)
    out += cmp_imm32("rcx", 0x464C457F)
    ok = len(out)
    out += jcc(JE, 0)
    out += exit_with(22)
    patch_fwd(ok)
    # AT_ENTRY present.
    out += cmp_imm32("r12", 0)
    ok = len(out)
    out += jcc(JNE, 0)
    out += exit_with(23)
    patch_fwd(ok)

    out += mov_imm("rax", 1)
    out += mov_imm("rdi", 1)
    lea_at = len(out)
    out += lea_rip("rsi", 0)
    lea_end = entry_rva + len(out)
    out += mov_imm("rdx", len(MSG_LD))
    out += SYSCALL
    out[lea_at + 3:lea_at + 7] = struct.pack("<i", msg_rva - lea_end)

    # And hand over. `syscall` clobbers rcx and r11 and leaves r12 alone, which
    # is why the entry lives there.
    out += jmp_reg("r12")
    out += HLT
    return bytes(out)


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


MSG_FORK_CHILD = b"fork: the child is here\n"
MSG_FORK_PARENT = b"fork: the parent reaped exactly its own child\n"


def fork_code(entry_rva, msg_rva, _unused):
    """One call that returns twice, and a parent that proves which child it got.

    The shape is the whole test. `fork` answers zero in the child and a pid in
    the parent, so the branch below is the only thing telling two identical
    instruction streams apart -- if the child came back with the parent's
    return value both halves would take the same path and the run would look
    like a program that simply printed twice.

    The parent waits for **that** pid rather than for any child, and exits 9
    only when `wait4` hands back the number `fork` gave it. A kernel that
    invented a pid, or reaped the wrong entry, or answered the wait before the
    child had run, fails there rather than printing something plausible.

    Exit codes are the report: 9 is the parent having reaped its own child, 7
    is the child, and 3 is a parent whose wait disagreed with its fork.
    """
    c = bytearray()
    c += mov_imm("rax", 57) + SYSCALL          # fork()
    c += cmp_imm8("rax", 0)
    jne_at = len(c)
    c += jcc(JNE, 0)                            # non-zero -> parent
    jne_end = len(c)

    # ---- the child ----
    c += mov_imm("rax", 1) + mov_imm("rdi", 1)
    lea1_at = len(c)
    c += lea_rip("rsi", 0)
    lea1_end = len(c)
    c += mov_imm("rdx", len(MSG_FORK_CHILD)) + SYSCALL
    c += mov_imm("rax", 60) + mov_imm("rdi", 7) + SYSCALL + HLT

    # ---- the parent ----
    parent_at = len(c)
    c += mov_rr("r12", "rax")                   # keep the pid fork gave us
    c += mov_imm("rax", 61)                     # wait4
    c += mov_rr("rdi", "r12")                   # that child, not any child
    c += mov_imm("rsi", 0)                      # no status word
    c += mov_imm("rdx", 0) + mov_imm("r10", 0)
    c += SYSCALL
    c += cmp_rr("rax", "r12")                   # the same pid back?
    jne2_at = len(c)
    c += jcc(JNE, 0)
    jne2_end = len(c)
    c += mov_imm("rax", 1) + mov_imm("rdi", 1)
    lea2_at = len(c)
    c += lea_rip("rsi", 0)
    lea2_end = len(c)
    c += mov_imm("rdx", len(MSG_FORK_PARENT)) + SYSCALL
    c += mov_imm("rax", 60) + mov_imm("rdi", 9) + SYSCALL + HLT
    bad_at = len(c)
    c += mov_imm("rax", 60) + mov_imm("rdi", 3) + SYSCALL + HLT

    struct.pack_into("<i", c, jne_at + 2, parent_at - jne_end)
    struct.pack_into("<i", c, jne2_at + 2, bad_at - jne2_end)
    struct.pack_into("<i", c, lea1_at + 3, msg_rva - (entry_rva + lea1_end))
    struct.pack_into("<i", c, lea2_at + 3,
                     msg_rva + len(MSG_FORK_CHILD) - (entry_rva + lea2_end))
    return bytes(c)


EXEC_PATH = b"/tmp/fixed\x00"
MSG_EXEC_BACK = b"execve: it came back, so it failed\n"


def exec_code(entry_rva, msg_rva, _unused):
    """Become another program, and say so if it does not happen.

    **Success is somebody else's output.** `execve` that works produces no
    line from this program at all -- what appears is `/tmp/fixed` printing
    `hello from ring 3` and exiting 5, from an image this one never had. That
    is a much stronger signal than a message here would be, because there is
    no way to fake it from inside a program that failed to be replaced.

    The failure path is the one that prints. `execve` returning at all means
    it did not happen, so the message says exactly that and exits 4, and a run
    reporting 4 is a run that failed however plausible its output looks.
    """
    c = bytearray()
    c += mov_imm("rax", 59)                     # execve
    lea_at = len(c)
    c += lea_rip("rdi", 0)                      # the path
    lea_end = len(c)
    c += mov_imm("rsi", 0)                      # no argv
    c += mov_imm("rdx", 0)                      # no envp
    c += SYSCALL
    # Only reached when execve refused.
    c += mov_imm("rax", 1) + mov_imm("rdi", 1)
    lea2_at = len(c)
    c += lea_rip("rsi", 0)
    lea2_end = len(c)
    c += mov_imm("rdx", len(MSG_EXEC_BACK)) + SYSCALL
    c += mov_imm("rax", 60) + mov_imm("rdi", 4) + SYSCALL + HLT

    struct.pack_into("<i", c, lea_at + 3, msg_rva - (entry_rva + lea_end))
    struct.pack_into("<i", c, lea2_at + 3,
                     msg_rva + len(EXEC_PATH) - (entry_rva + lea2_end))
    return bytes(c)


MSG_POLLED = b"wait4: WNOHANG said not-yet, then handed over the child\n"
MSG_INSTANT = b"wait4: the child was already gone on the first poll\n"


def wnohang_code(entry_rva, msg_rva, _unused):
    """Poll for a child instead of blocking on it.

    **The parent never yields between `fork` and its first poll**, so the
    child -- which lives on another task -- cannot have run yet, and a correct
    `WNOHANG` has to answer zero. Answering the pid there would mean reaping a
    child that had not finished; answering `ECHILD` would mean claiming it
    never existed, and a shell reads that as "stop asking" and drops it.

    The count is the evidence. `r13` tallies the not-yet answers and the exit
    code reports which of two things happened, because only one of them tests
    anything: exit 9 means `WNOHANG` genuinely returned zero at least once and
    then produced the right pid, exit 5 means the child had already finished by
    the first poll and the run proved only that blocking was not required.
    """
    c = bytearray()
    c += mov_imm("rax", 57) + SYSCALL           # fork()
    c += cmp_imm8("rax", 0)
    jne_at = len(c)
    c += jcc(JNE, 0)
    jne_end = len(c)
    # ---- the child: leave at once, so the parent's poll has something to
    # find without the child needing to do anything observable ----
    c += mov_imm("rax", 60) + mov_imm("rdi", 7) + SYSCALL + HLT

    parent_at = len(c)
    c += mov_rr("r12", "rax")                   # the pid
    c += mov_imm("r13", 0)                      # not-yet answers
    poll_at = len(c)
    c += mov_imm("rax", 61)                     # wait4
    c += mov_rr("rdi", "r12")
    c += mov_imm("rsi", 0)                      # no status
    c += mov_imm("rdx", 1)                      # WNOHANG
    c += mov_imm("r10", 0)
    c += SYSCALL
    c += cmp_imm8("rax", 0)
    jne2_at = len(c)
    c += jcc(JNE, 0)                            # something came back
    jne2_end = len(c)
    c += add_imm("r13", 1)
    back_at = len(c)
    c += jmp(0)
    back_end = len(c)

    got_at = len(c)
    c += cmp_rr("rax", "r12")                   # our child, not another
    jne3_at = len(c)
    c += jcc(JNE, 0)
    jne3_end = len(c)
    c += cmp_imm8("r13", 0)
    je_at = len(c)
    c += jcc(0x84, 0)                           # JE -> the instant path
    je_end = len(c)
    c += mov_imm("rax", 1) + mov_imm("rdi", 1)
    lea1_at = len(c)
    c += lea_rip("rsi", 0)
    lea1_end = len(c)
    c += mov_imm("rdx", len(MSG_POLLED)) + SYSCALL
    c += mov_imm("rax", 60) + mov_imm("rdi", 9) + SYSCALL + HLT

    instant_at = len(c)
    c += mov_imm("rax", 1) + mov_imm("rdi", 1)
    lea2_at = len(c)
    c += lea_rip("rsi", 0)
    lea2_end = len(c)
    c += mov_imm("rdx", len(MSG_INSTANT)) + SYSCALL
    c += mov_imm("rax", 60) + mov_imm("rdi", 5) + SYSCALL + HLT

    bad_at = len(c)
    c += mov_imm("rax", 60) + mov_imm("rdi", 3) + SYSCALL + HLT

    struct.pack_into("<i", c, jne_at + 2, parent_at - jne_end)
    struct.pack_into("<i", c, jne2_at + 2, got_at - jne2_end)
    struct.pack_into("<i", c, back_at + 1, poll_at - back_end)
    struct.pack_into("<i", c, jne3_at + 2, bad_at - jne3_end)
    struct.pack_into("<i", c, je_at + 2, instant_at - je_end)
    struct.pack_into("<i", c, lea1_at + 3, msg_rva - (entry_rva + lea1_end))
    struct.pack_into("<i", c, lea2_at + 3,
                     msg_rva + len(MSG_POLLED) - (entry_rva + lea2_end))
    return bytes(c)


def spin_code(entry_rva, _a, _b):
    """Loop forever, asking for nothing.

    Two bytes: `jmp .`. It makes no syscall, so nothing the kernel can refuse
    ever happens, and before the deadline existed this owned the machine with
    no key left to press.
    """
    return bytes([0xEB, 0xFE])


MSG_USAGE = b"cat: needs a path\n"


GL_IMPORTS = [
    "OSMesaCreateContext",
    "OSMesaMakeCurrent",
    "glClearColor",
    "glClear",
    "glFinish",
]

# Cyan, and chosen so nothing rounds. `glClearColor` takes floats in 0..1 and
# a half would land on 127 or 128 depending on how Mesa converts; ones and
# zeroes come out as 255 and 0 under any rule, so the byte this checks is the
# byte Mesa meant. Distinct from the red-green-blue the framebuffer fixture
# paints, so a screenshot cannot confuse the two.
GL_CLEAR = (0.0, 1.0, 1.0, 1.0)
# **The low 24 bits, and the alpha is dropped on purpose.** `load_d` zero-
# extends a dword into a 64-bit register while `cmp rax, imm32` *sign*-extends
# its immediate, so any expected value with bit 31 set can never compare equal
# -- and 0xFF00FFFF has it, because alpha is 255. Masking to B, G and R avoids
# the trap and drops nothing that matters: the fourth byte is not scanned out.
GL_EXPECT = 0x00FFFF            # B,G,R = 255,255,0, which is cyan


def gl_code(text_at, got_at, rodata_at):
    """Ask a real Mesa to clear the screen, and put the result on it.

    Eight checks folding into a mask, and what it proves is not any one of them:
    it is that `libOSMesa` -- four megabytes of somebody else's C, fetched from
    a Debian archive and touched by nothing here -- loaded through this
    kernel's `ld.so` path, ran at ring 3, and wrote pixels into a buffer this
    program owns.

    **`OSMESA_BGRA` is why the blit is a `write` and not a conversion.** Mesa
    will hand back any byte order asked for, and the one this display uses is
    the one `dev.rs` reports in `fb_var_screeninfo` -- so the buffer Mesa fills
    is already in the framebuffer's own layout and goes to `/dev/fb0`
    unmodified. Getting that argument wrong costs a colour swap and nothing
    else, which is exactly the failure the band fixture was built to catch.

    The width comes from `xres_virtual` rather than `xres`, deliberately.
    `xres_virtual` is the stride in pixels, so `w * 4` is exactly
    `line_length` and the blit lands square; using the visible width on a
    display whose stride is wider shears the picture one row at a time.
    """
    O_RDWR, PROT_RW, MAP_PRIVATE_ANON = 2, 3, 0x22
    OSMESA_BGRA, GL_UNSIGNED_BYTE, GL_COLOR_BUFFER_BIT = 1, 0x1401, 0x4000
    # **A plain number, not an `_IOR` encoding.** `linux/fb.h` spells the
    # framebuffer ioctls as bare constants 0x4600 upward rather than through
    # the `_IOC` macros, so there is no direction or size in the top bits. The
    # encoded form was answered with `ENOTTY`, the geometry stayed zero, the
    # `mmap` for the pixel buffer came back `EINVAL`, and this program read
    # `[-22]`. Every part of that was correct except the constant.
    FBIOGET_VSCREENINFO = 0x4600
    SYS_MMAP, SYS_OPEN, SYS_IOCTL, SYS_WRITE, SYS_EXIT = 9, 2, 16, 1, 231
    out = bytearray()

    def fwd(at):
        out[at + 2:at + 6] = struct.pack("<i", len(out) - (at + 6))

    def fold(bit, cc):
        ok = len(out)
        out.extend(jcc(cc, 0))
        out.extend(or_imm32("rbp", 1 << bit))
        fwd(ok)

    def sysc(nr, *args):
        b = bytearray()
        b += mov_imm("rax", nr)
        for reg, a in zip(("rdi", "rsi", "rdx", "r10", "r8", "r9"), args):
            b += mov_rr(reg, a) if isinstance(a, str) else mov_imm(reg, a)
        b += SYSCALL
        return bytes(b)

    def call(name):
        """One indirect call through this import's GOT slot."""
        i = GL_IMPORTS.index(name)
        here = text_at + len(out)
        return call_rip(got_at + 8 * i - (here + 6))

    def flt(i):
        """The rip-relative displacement of one float in the read-only data."""
        return rodata_at + 4 * i - (text_at + len(out) + 8)

    out += mov_imm("rbp", 0)
    out += sub_imm("rsp", 0x120)
    out += mov_rr("r15", "rsp")            # fb_var_screeninfo scratch, 204 bytes
    out += mov_rr("r14", "rsp")
    out += add_imm("r14", 208)             # "/dev/fb0"

    out += mov_imm("rax", int.from_bytes(b"/dev", "little"))
    out += store_d("r14", 0, "rax")
    out += mov_imm("rax", int.from_bytes(b"/fb0", "little"))
    out += store_d("r14", 4, "rax")
    out += mov_imm("rax", 0)
    out += store_d("r14", 8, "rax")

    # 0. The screen, opened for writing.
    out += sysc(SYS_OPEN, "r14", O_RDWR, 0)
    out += mov_rr("r13", "rax")
    out += cmp_imm8("r13", 0)
    fold(0, JGE)

    # 1. And its geometry, asked for rather than assumed.
    out += mov_imm("rax", SYS_IOCTL)
    out += mov_rr("rdi", "r13")
    out += mov_imm("rsi", FBIOGET_VSCREENINFO)
    out += mov_rr("rdx", "r15")
    out += SYSCALL
    out += cmp_imm8("rax", 0)
    fold(1, JE)

    out += load_d("r10", "r15", 8)         # xres_virtual, the stride in pixels
    out += load_d("r11", "r15", 4)         # yres
    out += mov_rr("r12", "r10")
    out += imul_rr("r12", "r11")
    out += shl_imm("r12", 2)               # bytes, four per pixel

    # 2. Somewhere for Mesa to draw. Its own buffer, not the aperture: OSMesa
    #    reads back as well as writes, and the screen is not a scratch pad.
    out += sysc(SYS_MMAP, 0, "r12", PROT_RW, MAP_PRIVATE_ANON, -1, 0)
    out += mov_rr("rbx", "rax")
    out += cmp_imm8("rbx", 0)
    fold(2, JG)

    # 3. A context, in the display's own byte order.
    out += mov_imm("rdi", OSMESA_BGRA)
    out += mov_imm("rsi", 0)
    out += call("OSMesaCreateContext")
    out += mov_rr("r12", "rax")
    out += cmp_imm8("r12", 0)
    fold(3, JG)

    # 4. Bound to the buffer. Everything after this draws into it.
    out += load_d("r10", "r15", 8)
    out += load_d("r11", "r15", 4)
    out += mov_rr("rdi", "r12")
    out += mov_rr("rsi", "rbx")
    out += mov_imm("rdx", GL_UNSIGNED_BYTE)
    out += mov_rr("rcx", "r10")
    out += mov_rr("r8", "r11")
    out += call("OSMesaMakeCurrent")
    out += cmp_imm8("rax", 0)
    fold(4, JG)

    # 5. Four floats, in the four registers the ABI names for them. This is
    #    the one call in any fixture here that cannot be made with integers.
    for i in range(4):
        out += movss_rip(i, flt(i))
    out += call("glClearColor")
    out += mov_imm("rdi", GL_COLOR_BUFFER_BIT)
    out += call("glClear")
    out += call("glFinish")

    # 6. The claim. Mesa wrote into a buffer this program allocated, in the
    #    byte order it was asked for, and the first pixel is the colour.
    out += load_d("rax", "rbx", 0)
    out += shl_imm("rax", 40)
    out += shr_imm("rax", 40)
    out += cmp_imm32("rax", GL_EXPECT)
    fold(5, JE)
    # 7. And the last pixel too, so a clear that touched one row is not
    #    mistaken for a clear that touched the screen.
    out += mov_rr("rcx", "rbx")
    out += mov_rr("rdx", "r12")
    out += load_d("r10", "r15", 8)
    out += load_d("r11", "r15", 4)
    out += mov_rr("rdx", "r10")
    out += imul_rr("rdx", "r11")
    out += shl_imm("rdx", 2)
    out += sub_imm("rdx", 4)
    out += add_rr("rcx", "rdx")
    out += load_d("rax", "rcx", 0)
    out += shl_imm("rax", 40)
    out += shr_imm("rax", 40)
    out += cmp_imm32("rax", GL_EXPECT)
    fold(6, JE)

    # 8. On the screen, with no conversion in between.
    out += mov_rr("rdx", "r10")
    out += imul_rr("rdx", "r11")
    out += shl_imm("rdx", 2)
    out += mov_rr("rax", "rdx")
    out += mov_imm("rdi", SYS_WRITE)
    out += mov_rr("r9", "rax")
    out += mov_imm("rax", SYS_WRITE)
    out += mov_rr("rdi", "r13")
    out += mov_rr("rsi", "rbx")
    out += mov_rr("rdx", "r9")
    out += SYSCALL
    out += cmp_rr("rax", "r9")
    fold(7, JE)

    out += sysc(SYS_EXIT, "rbp")
    out += HLT
    return bytes(out)


def thread_code(entry_rva, _a, _b):
    """Start a thread, let it write to shared memory, and join it.

    Ten checks folding into a mask, and the shape of the program is the test:
    `clone` returns *twice*, in two threads, at the same instruction, and the
    only thing that tells them apart is `rax`. Everything after that fork in
    the code is one path or the other.

    The child arrives with **every register zero except `rsp`**, because
    `glados_enter_guest` clears them, so it cannot be handed anything in a
    register. The parent leaves the shared address on the child's own stack
    before cloning and the child reads it from `[rsp-16]` -- sixteen and not
    eight, since the child's first `push` would land on eight.

    Joining is `CLONE_CHILD_CLEARTID` plus a futex, which is exactly what
    `pthread_join` is underneath: the parent sets the word to a sentinel,
    waits on it, and the kernel zeroing it when the thread ends is the whole
    of the notification.
    """
    PROT_RW, MAP_PRIVATE_ANON = 3, 0x22
    CLONE_THREAD_SHAPE = 0x100 | 0x200 | 0x400 | 0x800 | 0x10000 | 0x200000
    FUTEX_WAIT_PRIVATE = 128
    MARK = 0x1234
    ENOSYS, EINVAL = -38, -22
    SYS_MMAP, SYS_CLONE, SYS_FUTEX = 9, 56, 202
    SYS_GETTID, SYS_EXIT, SYS_YIELD, SYS_EXIT_GROUP = 186, 60, 24, 231
    out = bytearray()

    def fwd(at):
        out[at + 2:at + 6] = struct.pack("<i", len(out) - (at + 6))

    def fold(bit, cc):
        ok = len(out)
        out.extend(jcc(cc, 0))
        out.extend(or_imm32("rbp", 1 << bit))
        fwd(ok)

    def sysc(nr, *args):
        b = bytearray()
        b += mov_imm("rax", nr)
        for reg, a in zip(("rdi", "rsi", "rdx", "r10", "r8", "r9"), args):
            b += mov_rr(reg, a) if isinstance(a, str) else mov_imm(reg, a)
        b += SYSCALL
        return bytes(b)

    out += mov_imm("rbp", 0)

    # 0. Two pages. The first holds the word the child writes and the word the
    #    join waits on; the second is the child's stack, which grows down from
    #    its top.
    out += sysc(SYS_MMAP, 0, 0x2000, PROT_RW, MAP_PRIVATE_ANON, -1, 0)
    out += mov_rr("rbx", "rax")
    out += cmp_imm8("rbx", 0)
    fold(0, JG)

    # The child's stack top, and the shared address left on it where the child
    # can reach it with no registers of its own.
    out += mov_rr("r13", "rbx")
    out += add_imm("r13", 0x2000)
    out += mov_rr("r14", "r13")
    out += sub_imm("r14", 16)
    out += store_q("r14", 0, "rbx")

    # The sentinel the join waits on. Linux does not write this at clone time;
    # a thread library sets it to the child's id and this sets it to one, since
    # what matters is only that the kernel's zero is distinguishable.
    out += mov_imm("rax", 1)
    out += store_d("rbx", 8, "rax")
    out += mov_imm("rax", 0)
    out += store_d("rbx", 0, "rax")

    # 1. A clone that is not a thread. `CLONE_THREAD` dropped asks for a
    #    process, and there are none here, so it is refused rather than
    #    answered with a thread -- which would be two names for one address
    #    space and a program free to write through both.
    out += mov_rr("r15", "rbx")
    out += add_imm("r15", 8)
    out += sysc(SYS_CLONE, CLONE_THREAD_SHAPE & ~0x10000, "r13", 0, "r15", 0)
    out += cmp_imm32("rax", ENOSYS)
    fold(1, JE)

    # 2. And a stack of zero, whose fault would otherwise land several frames
    #    into the thread function and read as a bug in the program.
    out += sysc(SYS_CLONE, CLONE_THREAD_SHAPE, 0, 0, "r15", 0)
    out += cmp_imm32("rax", EINVAL)
    fold(2, JE)

    # 3. The real one. Both threads return here; only `rax` tells them apart.
    out += sysc(SYS_CLONE, CLONE_THREAD_SHAPE, "r13", 0, "r15", 0)
    out += cmp_imm8("rax", 0)
    child = len(out)
    out += jcc(JE, 0)

    # ---- parent ----------------------------------------------------------
    out += mov_rr("r12", "rax")
    out += cmp_imm8("r12", 1)
    fold(3, JG)

    # 4. The main thread's id is its process id, or a library concludes it is
    #    not the main thread and takes a different path entirely.
    out += sysc(SYS_GETTID)
    out += cmp_imm8("rax", 1)
    fold(4, JE)

    # 5. A futex on a word that is not four-aligned cannot be one.
    out += mov_rr("rdi", "rbx")
    out += add_imm("rdi", 9)
    out += sysc(SYS_FUTEX, "rdi", FUTEX_WAIT_PRIVATE, 1, 0)
    out += cmp_imm32("rax", EINVAL)
    fold(5, JE)

    # 6. Yielding is a thing this kernel can really do, which is unusual in
    #    this file -- most scheduling calls get a shape rather than an action.
    out += sysc(SYS_YIELD)
    out += cmp_imm8("rax", 0)
    fold(6, JE)

    # 7. The join. Blocks while the word is the sentinel and returns when the
    #    kernel clears it, which is `pthread_join` with the library removed.
    #
    #    **`EAGAIN` is a pass and finding that out cost a run.** It means the
    #    word was already not the sentinel, which here means the child finished
    #    before the parent got round to waiting -- a race the futex interface
    #    exists to make safe rather than an error. A real `pthread_join` loops
    #    on exactly this, so a fixture that treated it as a failure would be
    #    testing that the child is slow.
    out += sysc(SYS_FUTEX, "r15", FUTEX_WAIT_PRIVATE, 1, 0)
    out += cmp_imm8("rax", 0)
    already = len(out)
    out += jcc(JE, 0)
    out += cmp_imm32("rax", -11)
    fold(7, JE)
    fwd(already)

    # 8. Cleared by the kernel rather than by anybody in this program.
    out += load_d("rax", "rbx", 8)
    out += cmp_imm8("rax", 0)
    fold(8, JE)

    # 9. And the child really ran, in the parent's own address space. This is
    #    the whole claim: one page, two threads, and a value that could only
    #    have been put there by the other one.
    out += load_d("rax", "rbx", 0)
    out += cmp_imm32("rax", MARK)
    fold(9, JE)

    out += sysc(SYS_EXIT_GROUP, "rbp")
    out += HLT

    # ---- child -----------------------------------------------------------
    fwd(child)
    # Nothing but `rsp` survived, so the shared address comes off the stack.
    # `load_stack` and not `load_q`: rm=100 means a SIB byte follows, so `rsp`
    # as a base needs the form that writes one. `load_q` would encode an
    # index-scaled address nobody asked for.
    out += load_stack("rbx", 0xF0)             # [rsp-16]
    out += mov_imm("rax", MARK)
    out += store_d("rbx", 0, "rax")
    # `exit` and not `exit_group`: this thread is done and the process is not.
    out += sysc(SYS_EXIT, 0)
    out += HLT
    return bytes(out)


def ev_code(entry_rva, _a, _b):
    """Open the keyboard, ask it what it is, and block until somebody types.

    Twelve checks folding into a mask, and the interesting one is not a check
    at all: the sixth `read` has no `O_NONBLOCK` and nothing has happened yet,
    so the guest goes to sleep inside the kernel and comes back when the timer
    delivers a scheduled scancode. That path -- block, feed, wake -- is the
    whole of what an input device is for, and nothing else in this tree
    exercises a guest waiting on anything.

    Run it with `linux feed shift@300 -shift@900` armed beforehand. Shift
    rather than a letter on purpose: `kbd::decode` returns early for the
    modifiers before it pushes a character, so the feed leaves nothing in the
    shell's own input ring for `drive.py` to trip over.

    The field checks are one 32-bit compare each, which falls out of the
    layout: `type` and `code` are adjacent `__u16` at offset 16, so a press of
    the left shift key is exactly `0x002A0001` read as a dword.
    """
    O_RDONLY = 0
    EINVAL, ESPIPE = -22, -29
    EV_KEY_LEFTSHIFT = (42 << 16) | 1        # type EV_KEY, code KEY_LEFTSHIFT
    SYS_OPEN, SYS_READ, SYS_IOCTL, SYS_LSEEK, SYS_EXIT = 2, 0, 16, 8, 231
    out = bytearray()

    def fwd(at):
        out[at + 2:at + 6] = struct.pack("<i", len(out) - (at + 6))

    def fold(bit, cc):
        ok = len(out)
        out.extend(jcc(cc, 0))
        out.extend(or_imm32("rbp", 1 << bit))
        fwd(ok)

    def mov_u32(reg, v):
        """mov reg, v as an unsigned 32-bit value.

        `mov_imm` sign-extends, and every evdev request has bit 31 set, so a
        plain load would hand the kernel `0xFFFFFFFF80044501` where Linux is
        handed `0x80044501`. The masking in the dispatcher would forgive it and
        that is exactly why it is not forgiven here: a fixture should pass what
        a real program passes.
        """
        b = bytearray()
        b += mov_imm(reg, v - (1 << 32) if v >= (1 << 31) else v)
        b += shl_imm(reg, 32)
        b += shr_imm(reg, 32)
        return bytes(b)

    def sysc(nr, *args):
        b = bytearray()
        b += mov_imm("rax", nr)
        for reg, a in zip(("rdi", "rsi", "rdx", "r10", "r8", "r9"), args):
            if isinstance(a, str):
                b += mov_rr(reg, a)
            elif isinstance(a, bytes):
                b += a if reg in a.decode("latin-1", "ignore") else a
            else:
                b += mov_imm(reg, a)
        b += SYSCALL
        return bytes(b)

    out += mov_imm("rbp", 0)
    # 384 bytes: an eight-event read buffer at +0, and the path and the ioctl
    # scratch past it. Reached through registers because rm=100 means a SIB
    # byte and none of these emitters writes one.
    out += sub_imm("rsp", 0x180)
    out += mov_rr("rbx", "rsp")
    out += mov_rr("r15", "rsp")
    out += add_imm("r15", 256)

    for i, part in enumerate((b"/dev", b"/inp", b"ut/e", b"vent")):
        out += mov_imm("rax", int.from_bytes(part, "little"))
        out += store_d("r15", i * 4, "rax")
    out += mov_imm("rax", int.from_bytes(b"0\x00\x00\x00", "little"))
    out += store_d("r15", 16, "rax")
    out += mov_rr("rdi", "r15")

    # 0. Blocking, deliberately. `O_NONBLOCK` would turn the whole point of
    #    this fixture into a spin loop.
    out += sysc(SYS_OPEN, "rdi", O_RDONLY, 0)
    out += mov_rr("r12", "rax")
    out += cmp_imm8("r12", 0)
    fold(0, JGE)

    # Scratch for the ioctls, clear of the path.
    out += mov_rr("r14", "r15")
    out += add_imm("r14", 64)

    # 1. EVIOCGVERSION, which is the one request whose answer is a constant.
    out += mov_imm("rax", SYS_IOCTL)
    out += mov_rr("rdi", "r12")
    out += mov_u32("rsi", 0x80044501)
    out += mov_rr("rdx", "r14")
    out += SYSCALL
    out += load_d("rax", "r14", 0)
    out += cmp_imm32("rax", 0x0001_0001)
    fold(1, JE)

    # 2. EVIOCGBIT(0, 8): which event types exist. A keyboard is EV_SYN and
    #    EV_KEY and nothing else, so the first byte is exactly 3 -- and a 4 in
    #    there would be EV_REL, which would make this a mouse.
    out += mov_imm("rax", 0)
    out += store_d("r14", 0, "rax")
    out += mov_imm("rax", SYS_IOCTL)
    out += mov_rr("rdi", "r12")
    out += mov_u32("rsi", 0x80084520)
    out += mov_rr("rdx", "r14")
    out += SYSCALL
    out += load_d("rax", "r14", 0)
    out += cmp_imm8("rax", 3)
    fold(2, JE)

    # 3. EVIOCGNAME(32) answers the length it wrote, terminator included.
    out += mov_imm("rax", SYS_IOCTL)
    out += mov_rr("rdi", "r12")
    out += mov_u32("rsi", 0x80204506)
    out += mov_rr("rdx", "r14")
    out += SYSCALL
    out += cmp_imm8("rax", 0)
    fold(3, JG)

    # 4. A buffer too small for one record is EINVAL rather than a short read.
    #    A reader walks this stream by a fixed stride, so half a record is not
    #    less data, it is garbage from there on.
    out += sysc(SYS_READ, "r12", "rbx", 16)
    out += cmp_imm32("rax", EINVAL)
    fold(4, JE)

    # 5. And an event stream has no position.
    out += sysc(SYS_LSEEK, "r12", 0, 0)
    out += cmp_imm32("rax", ESPIPE)
    fold(5, JE)

    # 6-9. The read that blocks. Nothing has happened yet, so the guest sleeps
    #    inside the kernel and returns when the timer delivers the press.
    out += sysc(SYS_READ, "r12", "rbx", 192)
    out += cmp_imm8("rax", 0)
    fold(6, JG)
    out += load_d("rax", "rbx", 16)
    out += cmp_imm32("rax", EV_KEY_LEFTSHIFT)
    fold(7, JE)
    out += load_d("rax", "rbx", 20)
    out += cmp_imm8("rax", 1)
    fold(8, JE)
    # The SYN that closes the packet. A reader batches until it sees one, so a
    # device that never sends it delivers nothing while appearing to work.
    out += load_d("rax", "rbx", 40)
    out += cmp_imm8("rax", 0)
    fold(9, JE)

    # 10-11. Block again for the release, which is the event that matters
    #    most: a key whose release nobody delivered is a key held forever.
    out += sysc(SYS_READ, "r12", "rbx", 192)
    out += load_d("rax", "rbx", 16)
    out += cmp_imm32("rax", EV_KEY_LEFTSHIFT)
    fold(10, JE)
    out += load_d("rax", "rbx", 20)
    out += cmp_imm8("rax", 0)
    fold(11, JE)

    out += sysc(SYS_EXIT, "rbp")
    out += HLT
    return bytes(out)


def fb_code(entry_rva, _a, _b):
    """Open the screen, ask it what it is, and draw three bands on it.

    Thirteen checks folding into a mask, `fsabuse`'s idiom for its reason:
    zero means every one answered what Linux answers. But this fixture has a
    second job the others do not, and it is the more important one --

    **the bands are the only thing that can settle the pixel format.** A
    driver reports where each colour channel sits inside a word, and a driver
    that has red and blue the wrong way round produces no error anywhere: the
    ioctls succeed, the mapping is real, every check passes, and the picture
    is merely a strange colour. So this shifts by the offsets it was *told*
    rather than by ones it assumes, and paints red, green and blue in that
    order down the screen. A run that comes back blue-green-red has found a
    bug that no exit code could.

    Held on screen only when given an argument, because the harness takes its
    screenshot after the guest has exited and `teardown` puts the desktop
    back. `linux run /tmp/fb hold` sleeps instead, so a short `--timeout` is
    what photographs it -- the recipe `doom play` already needed.
    """
    O_RDWR = 2
    MAP_SHARED = 1
    GETV, PUTV, GETF = 0x4600, 0x4601, 0x4602
    TCGETS = 0x5401
    EINVAL, ENOTTY = -22, -25
    SYS_OPEN, SYS_IOCTL, SYS_MMAP, SYS_NANOSLEEP, SYS_EXIT = 2, 16, 9, 35, 231
    out = bytearray()

    def fwd(at):
        out[at + 2:at + 6] = struct.pack("<i", len(out) - (at + 6))

    def fold(bit, cc):
        """Fold `bit` unless the comparison just made took `cc`."""
        ok = len(out)
        out.extend(jcc(cc, 0))
        out.extend(or_imm32("rbp", 1 << bit))
        fwd(ok)

    def sysc(nr, *args):
        """One syscall. Each argument is an int or a register name."""
        b = bytearray()
        b += mov_imm("rax", nr)
        for reg, a in zip(("rdi", "rsi", "rdx", "r10", "r8", "r9"), args):
            b += mov_rr(reg, a) if isinstance(a, str) else mov_imm(reg, a)
        b += SYSCALL
        return bytes(b)

    # argc first, before the frame moves. A fixture that reads it afterwards
    # reads whatever it just allocated.
    out += load_stack("rax", 0)
    out += mov_imm("rbp", 0)
    # 320 bytes: fb_var_screeninfo at +0, fb_fix_screeninfo at +160, and the
    # 48 past that for the path, the timespec and argc. Both structures are
    # reached through a base register rather than rsp, because rm=100 means a
    # SIB byte and none of these emitters writes one.
    out += sub_imm("rsp", 0x140)
    out += mov_rr("rbx", "rsp")
    out += mov_rr("r15", "rsp")
    out += add_imm("r15", 160)
    out += store_q("r15", 112, "rax")

    # "/dev/fb0", two dwords and a terminator, since `mov_imm` is imm32.
    out += mov_imm("rax", int.from_bytes(b"/dev", "little"))
    out += store_d("r15", 96, "rax")
    out += mov_imm("rax", int.from_bytes(b"/fb0", "little"))
    out += store_d("r15", 100, "rax")
    out += mov_imm("rax", 0)
    out += store_d("r15", 104, "rax")
    out += mov_rr("rdi", "r15")
    out += add_imm("rdi", 96)

    # 0. Opened for writing, which the `/tmp` jail has to make an exception
    #    for: a device is not the store, so a new root hash is not the cost.
    out += sysc(SYS_OPEN, "rdi", O_RDWR, 0)
    out += mov_rr("r12", "rax")
    out += cmp_imm8("r12", 0)
    fold(0, JGE)

    # 1-3. The fixed half: what cannot change while the display is running.
    out += sysc(SYS_IOCTL, "r12", GETF, "r15")
    out += cmp_imm8("rax", 0)
    fold(1, JE)
    out += load_d("rcx", "r15", 48)                       # line_length
    out += cmp_imm8("rcx", 0)
    fold(2, JG)
    out += load_d("r13", "r15", 24)                       # smem_len
    out += cmp_rr("r13", "rcx")
    fold(3, JGE)

    # 4-6. The variable half.
    out += sysc(SYS_IOCTL, "r12", GETV, "rbx")
    out += cmp_imm8("rax", 0)
    fold(4, JE)
    out += load_d("rax", "rbx", 24)                       # bits_per_pixel
    out += cmp_imm8("rax", 32)
    fold(5, JE)
    out += load_d("rax", "rbx", 0)                        # xres
    out += cmp_imm8("rax", 0)
    fold(6, JG)

    # 7. Handing back the mode it was just given is accepted, or every
    #    well-behaved program that sets the mode it already has fails.
    out += sysc(SYS_IOCTL, "r12", PUTV, "rbx")
    out += cmp_imm8("rax", 0)
    fold(7, JE)

    # 8. And a mode this display cannot enter is refused. The important one:
    #    a driver that says yes and does nothing leaves the program drawing at
    #    a geometry the hardware does not have, forever, with no error.
    out += load_d("rax", "rbx", 0)
    out += add_imm("rax", 16)
    out += store_d("rbx", 0, "rax")
    out += sysc(SYS_IOCTL, "r12", PUTV, "rbx")
    out += cmp_imm32("rax", EINVAL)
    fold(8, JE)
    out += load_d("rax", "rbx", 0)
    out += sub_imm("rax", 16)
    out += store_d("rbx", 0, "rax")

    # 9. A request this is not answers ENOTTY, which is how `isatty` works.
    out += sysc(SYS_IOCTL, "r12", TCGETS, "rbx")
    out += cmp_imm32("rax", ENOTTY)
    fold(9, JE)

    # 10-11. Shared *and* writable, which is refused for every file in this
    #    namespace and is the whole point of a framebuffer. And what comes
    #    back is `smem_start` itself -- the guest is handed the pages the
    #    display is scanning out, not a copy of them.
    out += sysc(SYS_MMAP, 0, "r13", 3, MAP_SHARED, "r12", 0)
    out += mov_rr("r14", "rax")
    out += cmp_imm8("r14", 0)
    fold(10, JG)
    out += load_q("rcx", "r15", 16)
    out += cmp_rr("r14", "rcx")
    fold(11, JE)

    def band(offset_at, end_reg_setup):
        """255 in one channel, written eight bytes at a time from r9 to r10."""
        b = bytearray()
        b += load_d("rcx", "rbx", offset_at)
        b += mov_imm("rax", 255)
        b += shl_cl("rax")
        # Two pixels per store. The shifted value has a zero low half after
        # `shl 32`, so an add is an or and saves an emitter.
        b += mov_rr("rdx", "rax")
        b += shl_imm("rdx", 32)
        b += add_rr("rax", "rdx")
        b += end_reg_setup
        top = len(b)
        b += store_q("r9", 0, "rax")
        b += add_imm("r9", 8)
        b += cmp_rr("r9", "r10")
        rel = top - (len(b) + 6)
        b += jcc(JL, rel)
        return bytes(b)

    # A quarter of video memory, rounded down to a whole number of stores.
    # Shifts rather than a divide, because there is no divide here and a
    # quarter is exactly what two of them give.
    out += mov_rr("r8", "r13")
    out += shr_imm("r8", 5)
    out += shl_imm("r8", 3)
    out += mov_rr("r9", "r14")
    out += mov_rr("r10", "r14")
    out += add_rr("r10", "r8")
    out += band(32, b"")                                  # red, top quarter
    # 12. What was written is there to be read back, which is the check that
    #     the mapping is memory rather than a hole that swallows stores.
    out += load_q("rdx", "r14", 0)
    out += cmp_rr("rdx", "rax")
    fold(12, JE)
    out += band(44, add_rr("r10", "r8"))                  # green
    out += band(56, add_rr("r10", "r8"))                  # blue

    # Held only when asked, since the harness photographs what is on screen
    # after the guest has gone and `teardown` puts the desktop back.
    out += load_q("rax", "r15", 112)
    out += cmp_imm8("rax", 1)
    keep = len(out)
    out += jcc(JLE, 0)
    # Ten minutes, which is longer than any harness run. The exit path for a
    # held fixture is deliberately the *timeout*, because `drive.py` takes its
    # screenshot when it stops and `teardown` puts the desktop back the moment
    # the guest returns -- a shorter sleep is a race with the harness for the
    # one frame worth photographing.
    out += mov_imm("rax", 600)
    out += store_q("r15", 96, "rax")
    out += mov_imm("rax", 0)
    out += store_q("r15", 104, "rax")
    out += mov_rr("rdi", "r15")
    out += add_imm("rdi", 96)
    out += sysc(SYS_NANOSLEEP, "rdi", 0)
    fwd(keep)

    out += sysc(SYS_EXIT, "rbp")
    out += HLT
    return bytes(out)


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


# What a loader has to be handed, and no more.
DT_NULL, DT_NEEDED, DT_HASH, DT_STRTAB, DT_SYMTAB = 0, 1, 4, 5, 6
DT_RELA, DT_RELASZ, DT_RELAENT, DT_STRSZ, DT_SYMENT = 7, 8, 9, 10, 11
DT_FLAGS, DF_BIND_NOW = 30, 0x08
# Every real linker emits this for an executable and it looks optional, which
# is how it got left out. `ld.so` writes the address of its own debug
# structure into the slot, and glibc reaches for `l_info[DT_DEBUG]` -- index 21
# -- while setting up. A register dump named it exactly: `rdi` held 0xe8, and
# 0xe8 is `&l->l_info[21]` for a `link_map` at zero.
DT_DEBUG = 21
R_X86_64_GLOB_DAT = 6
PT_DYNAMIC, PT_PHDR = 2, 6


def build_linked(imports, soname, code, rodata=b""):
    """A real dynamically linked ELF, hand-built, with no linker anywhere.

    **This is the thing that makes a library testable without a compiler**, and
    it is the same argument this file already makes about static fixtures: no
    toolchain will emit a binary that differs from another in exactly one
    field, and none of them will emit one small enough to read.

    The minimum a loader needs turns out to be six tables and nine dynamic
    entries. `.dynstr` holds the names, `.dynsym` one undefined `FUNC` per
    import, `.hash` a System V hash table -- which is required even when
    nothing looks anything up, since `ld.so` refuses an object with neither
    hash -- `.rela.dyn` one `R_X86_64_GLOB_DAT` per import pointing at a `.got`
    slot, and `.dynamic` naming all of it.

    No PLT and no lazy binding: `DF_BIND_NOW` asks the loader to fill every
    slot before the program runs, so a call is `call [rip+slot]` and there is
    no resolver trampoline to get right. That is also the arrangement this
    machine already runs glibc under, for a reason recorded in CLAUDE.md.

    The hash table is the part with a trap in it. One bucket means every name
    collides, which is fine and is what makes it constructible: the chain then
    walks every symbol in order, so `bucket[0]` is 1 and `chain[i]` is `i+1`
    until the last, which is `STN_UNDEF` and terminates the walk. `nchain`
    *must* equal the symbol count -- it is how the loader sizes `.dynsym` --
    and getting it short reads as a symbol that does not exist.
    """
    interp = INTERP_REAL
    nsym = len(imports) + 1

    # Laid out before anything is emitted, because every table holds runtime
    # addresses of the others and `.dynamic` holds addresses of them all.
    ehdr = EHDR
    # Four, and the first of them is the one that took a disassembly to find.
    # glibc computes the main program's load address in exactly one place --
    # `case PT_PHDR: main_map->l_addr = (Addr) phdr - ph->p_vaddr;` -- and
    # there is no other. Without a `PT_PHDR` the base stays zero, so every
    # address the loader derives from a `p_vaddr` is used raw: the fault was a
    # `strcmp` against `PT_INTERP`'s string at 0xe8, which is 64 for the header
    # plus three program headers, the file offset with nothing added to it.
    #
    # Every real linker emits one, which is why nothing else here needed it and
    # why leaving it out looked harmless.
    phnum = 4
    at = ehdr + PHENT * phnum

    def align(x, a):
        return (x + a - 1) & ~(a - 1)

    interp_at = at
    at += len(interp)

    dynstr = bytearray(b"\x00")
    soname_off = len(dynstr)
    dynstr += soname.encode() + b"\x00"
    name_offs = []
    for n in imports:
        name_offs.append(len(dynstr))
        dynstr += n.encode() + b"\x00"
    dynstr_at = at
    at += len(dynstr)

    at = align(at, 8)
    dynsym_at = at
    at += 24 * nsym

    hash_at = at
    at += 8 + 4 + 4 * nsym          # nbucket, nchain, one bucket, nchain chains

    at = align(at, 8)
    rela_at = at
    at += 24 * len(imports)

    dynamic_at = at
    ndyn = 12
    at += 16 * ndyn

    got_at = at
    at += 8 * len(imports)

    rodata_at = at
    at += len(rodata)

    text_at = align(at, 16)

    # The code is emitted twice: once to learn its length, once with every
    # displacement resolved. `mkelf.py` does this for the static fixtures too.
    body = code(text_at, got_at, rodata_at)
    total = text_at + len(body)

    dsym = bytearray(24 * nsym)      # entry 0 is reserved and stays zero
    for i, off in enumerate(name_offs, start=1):
        struct.pack_into("<IBBHQQ", dsym, 24 * i, off, 0x12, 0, 0, 0, 0)

    h = bytearray()
    h += struct.pack("<II", 1, nsym)
    h += struct.pack("<I", 1)                       # bucket[0] -> symbol 1
    h += struct.pack("<I", 0)                       # chain[0], unused
    for i in range(1, nsym):
        h += struct.pack("<I", 0 if i == nsym - 1 else i + 1)

    rela = bytearray()
    for i in range(len(imports)):
        rela += struct.pack("<QQq", got_at + 8 * i, ((i + 1) << 32) | R_X86_64_GLOB_DAT, 0)

    dyn = bytearray()
    for tag, val in (
        (DT_NEEDED, soname_off),
        (DT_HASH, hash_at),
        (DT_STRTAB, dynstr_at),
        (DT_SYMTAB, dynsym_at),
        (DT_STRSZ, len(dynstr)),
        (DT_SYMENT, 24),
        (DT_RELA, rela_at),
        (DT_RELASZ, len(rela)),
        (DT_RELAENT, 24),
        (DT_FLAGS, DF_BIND_NOW),
        (DT_DEBUG, 0),
        (DT_NULL, 0),
    ):
        dyn += struct.pack("<qQ", tag, val)
    assert len(dyn) == 16 * ndyn, (len(dyn), ndyn)

    out = bytearray(total)
    out[interp_at:interp_at + len(interp)] = interp
    out[dynstr_at:dynstr_at + len(dynstr)] = dynstr
    out[dynsym_at:dynsym_at + len(dsym)] = dsym
    out[hash_at:hash_at + len(h)] = h
    out[rela_at:rela_at + len(rela)] = rela
    out[dynamic_at:dynamic_at + len(dyn)] = dyn
    out[rodata_at:rodata_at + len(rodata)] = rodata
    out[text_at:text_at + len(body)] = body

    hdr = bytearray(EHDR)
    hdr[0:4] = b"\x7fELF"
    hdr[4], hdr[5], hdr[6] = 2, 1, 1
    hdr[7] = 0
    struct.pack_into("<HHI", hdr, 16, ET_DYN, 0x3E, 1)
    struct.pack_into("<QQQ", hdr, 24, text_at, ehdr, 0)
    struct.pack_into("<IHHHHHH", hdr, 48, 0, EHDR, PHENT, phnum, 0, 0, 0)
    out[0:EHDR] = hdr

    # One segment, read-write-execute, covering everything. This kernel maps
    # every page that way regardless, and a fixture that pretended otherwise
    # would be describing a machine it is not running on.
    p = bytearray()
    # First, and the ELF specification says it must be: "if it is present, it
    # must precede any loadable segment entry".
    p += struct.pack("<IIQQQQQQ", PT_PHDR, 4, ehdr, ehdr, ehdr,
                     PHENT * phnum, PHENT * phnum, 8)
    p += struct.pack("<IIQQQQQQ", PT_LOAD, 7, 0, 0, 0, total, total, 0x1000)
    p += struct.pack("<IIQQQQQQ", PT_DYNAMIC, 6, dynamic_at, dynamic_at,
                     dynamic_at, len(dyn), len(dyn), 8)
    p += struct.pack("<IIQQQQQQ", PT_INTERP, 4, interp_at, interp_at,
                     interp_at, len(interp), len(interp), 1)
    out[EHDR:EHDR + len(p)] = p
    return bytes(out)


def build(kind="static"):
    if kind == "gl":
        import struct as _s
        rodata = b"".join(_s.pack("<f", v) for v in GL_CLEAR)
        blob = build_linked(GL_IMPORTS, "libOSMesa.so.8", gl_code, rodata)
        entry = _s.unpack_from("<Q", blob, 24)[0]
        return blob, dict(entry=entry, msg=0, disp=0, lea_end=0, size=len(blob))

    interp = INTERP_FIXTURE if kind == "interp" else INTERP_REAL
    phnum = 2 if kind in ("dynamic", "interp") else 1
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
    elif kind == "thread":
        text = thread_code(entry, 0, 0)
        body = text
        msg_rva, disp, lea_end = body_at, 0, 0
    elif kind == "ev":
        text = ev_code(entry, 0, 0)
        body = text
        msg_rva, disp, lea_end = body_at, 0, 0
    elif kind == "fb":
        text = fb_code(entry, 0, 0)
        body = text
        msg_rva, disp, lea_end = body_at, 0, 0
    elif kind == "maps":
        text = maps_code(entry, 0, 0)
        body = text
        msg_rva, disp, lea_end = body_at, 0, 0
    elif kind == "loader":
        probe = loader_code(0, 0, 0)
        msg_rva = body_at + len(probe)
        text = loader_code(entry, msg_rva, 0)
        assert len(text) == len(probe), (len(text), len(probe))
        body = text + MSG_LD
        disp, lea_end = 0, 0
    elif kind == "spin":
        text = spin_code(entry, 0, 0)
        body = text
        msg_rva, disp, lea_end = body_at, 0, 0
    elif kind == "wnohang":
        probe = wnohang_code(0, 0, 0)
        msg_rva = body_at + len(probe)
        text = wnohang_code(entry, msg_rva, 0)
        assert len(text) == len(probe), (len(text), len(probe))
        body = text + MSG_POLLED + MSG_INSTANT
        disp, lea_end = 0, 0
    elif kind == "exec":
        probe = exec_code(0, 0, 0)
        msg_rva = body_at + len(probe)
        text = exec_code(entry, msg_rva, 0)
        assert len(text) == len(probe), (len(text), len(probe))
        body = text + EXEC_PATH + MSG_EXEC_BACK
        disp, lea_end = 0, 0
    elif kind == "fork":
        probe = fork_code(0, 0, 0)
        msg_rva = body_at + len(probe)
        text = fork_code(entry, msg_rva, 0)
        assert len(text) == len(probe), (len(text), len(probe))
        body = text + MSG_FORK_CHILD + MSG_FORK_PARENT
        disp, lea_end = 0, 0
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
    if kind in ("dynamic", "interp"):
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
    if kind in ("dynamic", "interp"):
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
    # ENODEV is asked for by nothing else in this file, which makes it the one
    # byte pattern that identifies this fixture rather than merely suiting it.
    if bytes([0x48, 0x83, 0xF8, 0xED]) in b:
        claim("it reserves a span and then lays a page of it out again in place",
              bytes([0x48, 0xC7, 0xC6, 0x00, 0x20, 0x00, 0x00]) in b     # rsi = 0x2000
              and bytes([0x49, 0xC7, 0xC2, 0x22, 0, 0, 0]) in b          # r10 = PRIVATE|ANON
              and bytes([0x49, 0xC7, 0xC2, 0x32, 0, 0, 0]) in b          # and again with FIXED
              and bytes([0x4C, 0x89, 0xEF]) in b)                        # rdi <- r13
        claim("it maps its own file, by the path it was invoked with",
              bytes([0x48, 0x8B, 0x7C, 0x24, 0x08]) in b                 # rdi <- [rsp+8]
              and bytes([0x48, 0xC7, 0xC0, 2, 0, 0, 0]) in b)            # open
        claim("and checks the mapping holds an ELF header rather than merely existing",
              bytes([0x41, 0x8B, 0x4F, 0x00]) in b                       # ecx <- [r15]
              and struct.pack("<i", 0x464C457F) in b)
        claim("it asks for every refusal by its own errno",
              all(bytes([0x48, 0x83, 0xF8, e & 0xFF]) in b
                  for e in (-9, -19, -22)))
        claim("eleven checks, each folding one bit into the mask it exits with",
              sum(1 for i in range(11)
                  if bytes([0x48, 0x81, 0xCD]) + struct.pack("<i", 1 << i) in b) == 11)
        claim("it exits with the mask rather than a constant",
              bytes([0x48, 0x89, 0xEF]) in b
              and bytes([0x48, 0xC7, 0xC0, 231, 0, 0, 0]) in b)
        return ok
    if MSG_LD in b:
        claim("it walks the stack in rbx and never moves rsp, which the program needs",
              bytes([0x48, 0x89, 0xE3]) in b                       # mov rbx, rsp
              and bytes([0x48, 0x89, 0xE5]) not in b               # no frame pointer
              and bytes([0x48, 0x81, 0xEC]) not in b)              # no sub rsp
        claim("it looks the aux vector up by key, so the order it arrives in cannot matter",
              all(bytes([0x48, 0x81, 0xF9]) + struct.pack("<i", k) in b
                  for k in (0, 7, 9)))
        claim("it reads the value beside each key, not the key after it",
              bytes([0x4C, 0x8B, 0x6B, 0x08]) in b                 # r13 <- [rbx+8]
              and bytes([0x4C, 0x8B, 0x63, 0x08]) in b             # r12 <- [rbx+8]
              and bytes([0x48, 0x81, 0xC3, 0x10, 0, 0, 0]) in b)   # rbx += 16
        claim("it checks AT_BASE points at an ELF header rather than merely being set",
              bytes([0x41, 0x8B, 0x4D, 0x00]) in b                 # ecx <- [r13]
              and struct.pack("<i", 0x464C457F) in b)
        claim("each check has an exit code of its own, so a failure says which",
              all(bytes([0x48, 0xC7, 0xC7, n, 0, 0, 0]) in b for n in (21, 22, 23)))
        claim("and the last thing it does is jump to the entry it was handed",
              b.rindex(bytes([0x41, 0xFF, 0xE4])) > b.rindex(SYSCALL))
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
    # A dynamically linked fixture, identified by the one import no other
    # thing in this file names.
    if b"OSMesaCreateContext" in b:
        # Read back through a parser that is deliberately not the writer, the
        # bargain `tokenizer.py --verify` makes. A dynamic section is a chain
        # of addresses into other tables, so a writer that got one offset
        # wrong produces a file `ld.so` rejects with no message at all.
        phoff = struct.unpack_from("<Q", b, 32)[0]
        phent, phnum = struct.unpack_from("<HH", b, 54)
        loads, dyn, interp = [], None, None
        for i in range(phnum):
            at = phoff + i * phent
            ty = struct.unpack_from("<I", b, at)[0]
            off = struct.unpack_from("<Q", b, at + 8)[0]
            va = struct.unpack_from("<Q", b, at + 16)[0]
            fsz = struct.unpack_from("<Q", b, at + 32)[0]
            if ty == PT_LOAD:
                loads.append((off, va, fsz))
            elif ty == 2:
                dyn = (off, fsz)
            elif ty == PT_INTERP:
                interp = b[off:off + fsz]
        phdrseg = None
        for i in range(phnum):
            at2 = phoff + i * phent
            if struct.unpack_from("<I", b, at2)[0] == 6:
                phdrseg = struct.unpack_from("<Q", b, at2 + 16)[0]
        claim("it carries a PHDR, a LOAD, a DYNAMIC and an INTERP segment",
              len(loads) == 1 and dyn is not None and interp == INTERP_REAL
              and phdrseg is not None)
        # The whole of how a loader finds the base: `AT_PHDR` minus this.
        # Absent, the base is zero and every derived address is a file offset
        # pretending to be a pointer.
        claim("and PT_PHDR comes first and names the header table, or the base is zero",
              struct.unpack_from("<I", b, phoff)[0] == 6 and phdrseg == EHDR)

        ents = dict()
        order = []
        off, fsz = dyn
        for i in range(fsz // 16):
            t, v = struct.unpack_from("<qQ", b, off + i * 16)
            order.append(t)
            ents[t] = v
        claim("the dynamic array ends in DT_NULL and nothing follows it",
              order[-1] == 0 and order.count(0) == 1)
        claim("it names every table a loader has to be handed",
              all(t in ents for t in (1, 4, 5, 6, 7, 8, 9, 10, 11, 21)))
        # Lazy binding needs a PLT and a resolver, and this has neither. The
        # flag is what makes `call [rip+slot]` safe: every slot is filled
        # before the program runs.
        claim("and asks for everything to be bound before it runs, since it has no PLT",
              ents.get(30, 0) & 0x08 != 0 and b"\xff\x15" in b)

        strtab = ents[5]
        def cstr(o):
            e = b.index(b"\x00", strtab + o)
            return b[strtab + o:e].decode("ascii", "replace")
        claim("it needs libOSMesa and nothing else, the rest arriving through it",
              [cstr(v) for t, v in zip(order, [ents[t] for t in order]) if t == 1] == []
              or cstr(ents[1]) == "libOSMesa.so.8")

        nsym = ents[8] // ents[9] + 1
        names = [cstr(struct.unpack_from("<I", b, ents[6] + 24 * i)[0])
                 for i in range(1, nsym)]
        claim("it imports the five entry points it calls, and only those",
              names == GL_IMPORTS)

        nbucket, nchain = struct.unpack_from("<II", b, ents[4])
        # The trap. `nchain` is how a loader sizes `.dynsym`, so a count one
        # short is a symbol that silently does not exist -- and with a single
        # bucket every name collides, which is what makes the chain walk the
        # whole table and what makes it constructible by hand at all.
        claim("the hash table's chain count is the symbol count, or a symbol vanishes",
              nbucket == 1 and nchain == nsym)
        first = struct.unpack_from("<I", b, ents[4] + 8)[0]
        chain = [struct.unpack_from("<I", b, ents[4] + 12 + 4 * i)[0] for i in range(nsym)]
        claim("and the chain walks every symbol once and then terminates",
              first == 1 and chain[1:] == list(range(2, nsym)) + [0])

        rels = [struct.unpack_from("<QQq", b, ents[7] + 24 * i) for i in range(ents[8] // 24)]
        claim("one GLOB_DAT relocation per import, each naming its own symbol",
              len(rels) == len(GL_IMPORTS)
              and all(r[1] & 0xFFFFFFFF == 6 for r in rels)
              and [r[1] >> 32 for r in rels] == list(range(1, nsym)))

        claim("eight checks, each folding one bit into the mask it exits with",
              sum(1 for i in range(8)
                  if bytes([0x48, 0x81, 0xCD]) + struct.pack("<i", 1 << i) in b) == 8)
        # The four floats, in the four registers the ABI names. This is the one
        # call in any fixture here that integers cannot make.
        claim("it loads four floats into xmm0 through xmm3 for the clear colour",
              all(bytes([0xF3, 0x0F, 0x10, 0x05 | (i << 3)]) in b for i in range(4)))
        claim("it asks Mesa for the display's own byte order, so the blit is a copy",
              bytes([0x48, 0xC7, 0xC7, 1, 0, 0, 0]) in b)
        claim("and ends in hlt, so a syscall that returns is visible",
              b.endswith(HLT))
        return ok
    # The thread shape is a constant nothing else here loads, which makes it
    # the identifier rather than merely a hint.
    if bytes([0x48, 0xC7, 0xC7]) + struct.pack("<I", 0x210F00) in b:
        claim("it asks for a thread: shared address space, shared thread group",
              bytes([0x48, 0xC7, 0xC7]) + struct.pack("<I", 0x210F00) in b)
        claim("and for the two things that are not, so both refusals are checked",
              bytes([0x48, 0xC7, 0xC7]) + struct.pack("<I", 0x210F00 & ~0x10000) in b)
        claim("ten checks, each folding one bit into the mask it exits with",
              sum(1 for i in range(10)
                  if bytes([0x48, 0x81, 0xCD]) + struct.pack("<i", 1 << i) in b) == 10)
        # The child cannot be handed anything in a register, so the only way it
        # reaches shared memory is off its own stack.
        claim("the child reads the shared address from its stack, having no registers",
              bytes([0x48, 0x8B, 0x5C, 0x24, 0xF0]) in b)
        claim("the child exits and the parent exit_groups, which are different calls",
              bytes([0x48, 0xC7, 0xC0, 60, 0, 0, 0]) in b
              and bytes([0x48, 0xC7, 0xC0, 231, 0, 0, 0]) in b)
        claim("it joins with a futex rather than by spinning on the word",
              bytes([0x48, 0xC7, 0xC0, 202, 0, 0, 0]) in b
              and bytes([0x48, 0xC7, 0xC6, 128, 0, 0, 0]) in b)
        claim("it exits with the mask rather than a constant",
              bytes([0x48, 0x89, 0xEF]) in b)
        claim("and ends in hlt on both paths, so a syscall that returns is visible",
              b.count(HLT) >= 2)
        return ok
    # `EVIOCGVERSION` sign-corrected: nothing else in this file loads that
    # constant, which makes it the identifier rather than merely a hint.
    if bytes([0x48, 0xC7, 0xC6]) + struct.pack("<i", 0x80044501 - (1 << 32)) in b:
        claim("it names /dev/input/event0, in dwords because mov_imm is imm32",
              bytes([0x48, 0xC7, 0xC0]) + b"/dev" in b
              and bytes([0x48, 0xC7, 0xC0]) + b"/inp" in b
              and bytes([0x48, 0xC7, 0xC0]) + b"vent" in b)
        claim("it asks the version, the type bitmap and the name",
              bytes([0x48, 0xC7, 0xC6]) + struct.pack("<i", 0x80044501 - (1 << 32)) in b
              and bytes([0x48, 0xC7, 0xC6]) + struct.pack("<i", 0x80084520 - (1 << 32)) in b
              and bytes([0x48, 0xC7, 0xC6]) + struct.pack("<i", 0x80204506 - (1 << 32)) in b)
        # The request has bit 31 set and `mov_imm` sign-extends, so the pair of
        # shifts is what makes the guest pass what a real program passes
        # rather than the same number with 32 ones on the front.
        claim("and clears the sign extension, so the request is the one Linux gets",
              bytes([0x48, 0xC1, 0xE6, 0x20]) in b and bytes([0x48, 0xC1, 0xEE, 0x20]) in b)
        claim("it opens blocking, since a non-blocking read would test nothing",
              bytes([0x48, 0xC7, 0xC6, 0, 0, 0, 0]) in b)
        claim("twelve checks, each folding one bit into the mask it exits with",
              sum(1 for i in range(12)
                  if bytes([0x48, 0x81, 0xCD]) + struct.pack("<i", 1 << i) in b) == 12)
        # type and code are adjacent __u16 at offset 16, so a press of the left
        # shift key is one dword. A fixture comparing them separately would
        # pass on a stream whose fields had been swapped.
        claim("it checks type and code as one dword, which the layout allows",
              bytes([0x48, 0x81, 0xF8]) + struct.pack("<I", (42 << 16) | 1) in b)
        claim("it reads twice, so the release is checked and not only the press",
              b.count(bytes([0x48, 0xC7, 0xC2, 192, 0, 0, 0])) == 2)
        claim("it makes the nine calls its checks add up to", b.count(SYSCALL) == 9)
        claim("it exits with the mask rather than a constant",
              bytes([0x48, 0x89, 0xEF]) in b)
        claim("and ends in hlt, so a syscall that returns is visible",
              b.endswith(HLT))
        return ok
    # `mov rsi, 0x4600` is FBIOGET_VSCREENINFO and nothing else in this file
    # asks for it, which makes it the identifier rather than merely a hint.
    if bytes([0x48, 0xC7, 0xC6, 0x00, 0x46, 0x00, 0x00]) in b:
        claim("it names /dev/fb0, in two dwords because mov_imm is imm32",
              bytes([0x48, 0xC7, 0xC0]) + b"/dev" in b
              and bytes([0x48, 0xC7, 0xC0]) + b"/fb0" in b)
        claim("it asks the framebuffer all three of its questions",
              bytes([0x48, 0xC7, 0xC6, 0x00, 0x46, 0x00, 0x00]) in b   # GET_VSCREEN
              and bytes([0x48, 0xC7, 0xC6, 0x01, 0x46, 0x00, 0x00]) in b  # PUT
              and bytes([0x48, 0xC7, 0xC6, 0x02, 0x46, 0x00, 0x00]) in b) # GET_FSCREEN
        claim("thirteen checks, each folding one bit into the mask it exits with",
              sum(1 for i in range(13)
                  if bytes([0x48, 0x81, 0xCD]) + struct.pack("<i", 1 << i) in b) == 13)
        # MAP_SHARED with PROT_READ|PROT_WRITE, which every other file in this
        # namespace is refused and which is the entire point of this device.
        claim("it maps the screen shared and writable, not private",
              bytes([0x48, 0xC7, 0xC2, 3, 0, 0, 0]) in b
              and bytes([0x49, 0xC7, 0xC2, 1, 0, 0, 0]) in b)
        # The check the picture exists to make. A fixture that shifted by a
        # constant would draw the right bands on a screen laid out the way it
        # assumed and say nothing about one that is not.
        claim("it shifts by the channel offset it was told, rather than a constant",
              bytes([0x48, 0xD3, 0xE0]) in b)
        claim("it makes the nine calls its checks and its hold add up to",
              b.count(SYSCALL) == 9)
        claim("it exits with the mask rather than a constant",
              bytes([0x48, 0x89, 0xEF]) in b)
        claim("and ends in hlt, so a syscall that returns is visible",
              b.endswith(HLT))
        return ok
    if MSG_POLLED in b:
        claim("it asks for WNOHANG rather than blocking",
              mov_imm("rdx", 1) in b)
        claim("it polls the pid fork gave it", mov_rr("rdi", "r12") in b)
        claim("it counts the not-yet answers, which is the evidence",
              add_imm("r13", 1) in b)
        claim("and reports the two outcomes apart",
              MSG_POLLED in b and MSG_INSTANT in b)
        # fork and the child's exit, the wait4, then two writes and three
        # exits across the tails: polled, already-gone, and disagreed.
        claim("it makes the eight calls its four paths add up to",
              b.count(SYSCALL) == 8)
        claim("its code ends in hlt, so a syscall that returns is visible",
              b[b.index(MSG_POLLED) - 1:b.index(MSG_POLLED)] == HLT)
        return ok
    if MSG_EXEC_BACK in b:
        claim("it names the program it is becoming", EXEC_PATH in b)
        # Two calls on the failure path and one on the success path, which is
        # the whole of it: a fixture that wrote something before the execve
        # would make a working exec indistinguishable from a broken one.
        claim("it makes three calls: the execve and the two it only reaches "
              "when that returns",
              b.count(SYSCALL) == 3)
        claim("and it asks for execve rather than something adjacent",
              mov_imm("rax", 59) in b)
        claim("it exits 4 on the path that means failure",
              mov_imm("rdi", 4) in b)
        claim("its code ends in hlt, so a syscall that returns is visible",
              b[b.index(EXEC_PATH) - 1:b.index(EXEC_PATH)] == HLT)
        return ok
    # Identified by its own message, the way every other branch here is:
    # `verify` reads the file back and has no idea what it was asked to build.
    if MSG_FORK_CHILD in b:
        # The branch is the whole fixture. Two identical instruction streams
        # are told apart by one comparison, so a fixture without it would
        # print twice and look like a working fork.
        claim("it branches on what fork answered, which is all that separates "
              "the two halves",
              bytes([0x48, 0x83, 0xF8, 0x00]) in b)
        claim("it waits for the pid fork gave it rather than for any child",
              mov_rr("rdi", "r12") in b)
        claim("and compares what wait4 answered against that same pid",
              cmp_rr("rax", "r12") in b)
        claim("the child's message is there", MSG_FORK_CHILD in b)
        claim("and the parent's, which only a correct wait reaches",
              MSG_FORK_PARENT in b)
        # fork, then write+exit in the child, then wait4+write+exit in the
        # parent, then the exit its disagreeing path takes. Seven, and the
        # seventh is the one a fixture without a failure branch would not have.
        claim("it makes the seven calls its three paths add up to",
              b.count(SYSCALL) == 7)
        claim("it carries three exits, so a disagreeing wait is reported "
              "rather than silent",
              b.count(mov_imm("rax", 60)) == 3)
        # The two messages sit after the code, so the file does not end in
        # `hlt` and this has to look where the code actually ends.
        claim("its code ends in hlt, so a syscall that returns is visible",
              b[b.index(MSG_FORK_CHILD) - 1:b.index(MSG_FORK_CHILD)] == HLT)
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
        if interp == INTERP_FIXTURE:
            claim("and this one names a path the namespace can actually hold",
                  interp.startswith(b"/tmp/"))
    if e_type == ET_EXEC:
        claim("the fixed fixture insists on a non-zero base", va != 0)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--kind",
                    choices=["static", "dynamic", "interp", "loader", "maps",
                             "fixed", "memory", "rogue",
                             "protect", "wild", "spin", "fork", "exec", "wnohang", "cat", "grep",
                             "fsabuse", "fb", "ev", "thread", "gl"],
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

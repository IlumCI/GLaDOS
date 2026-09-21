# 1.3.3

Six commits since 1.3.2, and `git diff v1.3.2 HEAD -- src/` is 1,675 insertions
across 13 files. One subject: a program this kernel did not compile now runs at
ring 3, opens files by name, and gets Linux's answers when it asks for things
that do not work.

Three bugs came out of it that had nothing to do with the feature, and two of
them could stop the machine. Those are the interesting part.

## Ring 3, in one address space

A guest runs at CPL 3 and the identity map is unchanged: no second set of page
tables, no CR3 swap, no TLB shootdown to design. What separates the guest from
the kernel is the U bit, and it is on the guest's own pages and on nothing else
in the machine.

The GDT grew from five entries to eight and their order is dictated rather than
chosen, because `sysret` takes no selectors -- it derives them from
`IA32_STAR[63:48]`, so the layout is a 32-bit code descriptor nobody uses, then
user data, then user code. Tidying it gives a `sysret` that lands in a data
segment. In is `iretq` with a hand-built frame; out is `sysretq`, which was the
wrong instruction while guests ran at ring 0.

A guest fault kills the guest and nothing else:

    linux run /tmp/wild
      killed by fault 0x0e after 0 syscall(s), machine intact

And a guest that never makes a syscall no longer owns the machine. The timer
enforces a deadline against a ring-3 frame, which is the only thing that could
have taken the terminal back, since the guest is what is running and there is no
key to press.

## A filesystem a program written for Linux recognises

The store underneath is not a filesystem: a content-addressed Merkle tree where
a copy is O(1), a snapshot is a hash, and `rm` detaches a name. None of that has
an `open`. `src/linux/fs.rs` is the translation, and it owns no bytes -- every
read goes to the tree.

Twelve calls, plus two that exist so a runtime does not conclude it failed to
start. `cat` and `grep` are hand-assembled fixtures, and `grep` is the one that
checks the bytes are *right* rather than merely present: a naive substring
search over a line buffer answers differently for every one-byte change in the
file, so a read that lost a byte or stopped early shows as a wrong set of lines
instead of plausible output.

    linux run /tmp/grep the /tmp/lines.txt
    the quick brown fox
    jumps over the lazy dog
        2 open   0x2c5cff1 -> 3      0 read 0x1000 -> 54    3 close -> 0
        1 write  0x1 buf+6  0x14 -> 20
        1 write  0x1 buf+31 0x17 -> 23
      exited 0 after 6 syscall(s)

Every number checks against the 54-byte file: +6 is past `alpha\n`, +31 past
`beta\n`, 20 is the third line with its newline and 23 is the last without one,
because the file ends there. Exit codes are grep's own, so `grep zebra` exits 1
having written nothing and a missing path exits 2.

Opening for writing is refused. A write to a content-addressed store is a new
root hash, so honouring `O_WRONLY` would hand a guest binary a route to the
namespace that goes around every gate `sysbox` puts in front of the shell.

## Three bugs, and what found each one

**A guest at ring 3 could halt the kernel with two ordinary syscalls.** `lseek`
past the end of a file is legal -- the module says so, in a comment two
functions above the bug -- and `read` then indexed a slice at the cursor without
clamping it.

    *** PANIC *** panicked at src\linux\syscall.rs:725:52:
    range start index 1048576 out of range for slice of length 54

Measured that way on purpose, against a build with the one-line fix reverted,
because the fix is not the interesting part. Nothing about ring 3, page rights
or the pointer checks was wrong; the pointer never left the kernel. It was an
ordinary Rust index on a value the guest chooses, and the only thing that finds
those is a program written to choose badly. So there is one now: `--kind
fsabuse` runs fifteen negatives and folds each failure into a bit of its exit
code. It found five more, including a `write` that looked at the descriptor
*number* and not the descriptor -- so `close(1)` then `open(...)` printed a
guest's redirected output to the terminal and reported success.

**A longjmp into inlined code has no calling convention to lean on.**
`recover::guard` saved the registers a callee must preserve, which is the right
list for a function boundary and the wrong question: `guard` is inlined into its
callers, so there is no boundary, and the compiler keeps a caller's live value
in `rax` or `r9` as readily as in `rbx`.

Found twice, both times as a wild pointer somewhere with nothing to do with
faults. First a PML4 walk with the index out of `rsi` -- `rsi` and `rdi` are
non-volatile under Microsoft x64 and the saved list was System V's, which is the
third time this tree has paid for that difference. Fixing that moved the symptom
rather than removing it: the heap's free list, walked from a cursor out of `r9`.
The second appeared only because an unrelated file changed what the register
allocator did, which is the tell that a list of registers chosen from a calling
convention was never going to be the answer. It saves all fifteen now, and the
landing code restores `rcx` last through itself with the jump target pushed onto
the already-restored stack, so it needs no scratch register at all.

**`read_cstr` refused a legal path.** It checks reachability a page at a time,
which is right for the speed and demands the whole rest of the page be owned --
and a region need not end on a page boundary, the image's being the ELF span. A
path constant in the last partial page of a binary is entirely legal and got
`EFAULT`, which reads as a pointer bug in the program rather than a bounds check
being too eager. `cat` and `grep` never hit it because their paths arrive in
`argv`; the abuse fixture carries one at the end of its own image.

## The aux vector was empty, and that is not a safe default

A static libc has no dynamic linker to ask, so what it cannot compute it reads
from there: `AT_PAGESZ` becomes musl's `libc.page_size`, which it divides by,
and `AT_RANDOM` is where the stack guard comes from. A vector holding nothing
but `AT_NULL` hands a real binary a page size of zero.

`AT_PHDR` is a runtime address, so it is the segment containing the header table
plus the offset into it. `base + phoff` is the same number only when the first
segment starts at file offset zero, which is true of every fixture here and is
not a property of the format -- and where no loadable segment covers the table
the whole group is omitted rather than pointed at zero, because
`dl_iterate_phdr` walks what it is given either way.

**None of it has been read by a real libc on this machine.** Every fixture is
hand-written and consumes none of it, so that part is a bet placed where the ABI
says to place it rather than evidence.

## What is honestly still missing

Every fixture is hand-assembled, so they test what somebody thought to test. The
`-ENOSYS` trace has never met a program that surprised it. Dynamically linked
binaries and `ET_EXEC` are refused for reasons about the address space rather
than the loader. There is no `write` path into the namespace, no `chdir`, no
descriptor-relative `openat`, and one process. WSL is registered on the
development machine and its disk is gone, so no real Linux binary has been
reachable from here to try.

## Verified

    fsabuse   exited 0 after 16 syscall(s)      fifteen negatives, all correct
    cat       exited 3 after 6 syscall(s)
    grep      exited 0 / 1 / 2 on match, no match, error
    diag linux                                  86 claims
    diag all                                    36 passed, 0 failed

The sweep was run after four guests had already run, and `mem` reported an
intact heap between them.

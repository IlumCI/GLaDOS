# 1.3.4

**It runs busybox.**

An unmodified static binary, fetched from busybox.net and touched by nothing
here, executing at ring 3 in a kernel with one address space and a language
model in it.

    linux run /tmp/busybox uname -a
    GLaDOS glados 1.3.4 one address space, no processes x86_64 GNU/Linux

    linux run /tmp/busybox hexdump -C /tmp/lines.txt
    00000000  61 6c 70 68 61 0a 74 68  65 20 71 75 69 63 6b 20  |alpha.the quick |
    00000010  62 72 6f 77 6e 20 66 6f  78 0a 62 65 74 61 0a 6a  |brown fox.beta.j|
    00000020  75 6d 70 73 20 6f 76 65  72 20 74 68 65 20 6c 61  |umps over the la|
    00000030  7a 79 20 64 6f 67                                 |zy dog|

    linux run /tmp/busybox sha256sum /tmp/lines.txt
    6ecb6686ad1673a0f3c021fc356890758853980fd6465b3b5a04fd859682733a

That digest is the one the host computes over the same file. The bytes went
from FAT into the namespace, out through the POSIX projection, through
busybox's own hash at ring 3, and back through `writev`, and every one
survived. It is the first end-to-end check of this path against an
implementation nobody here wrote.

Working today: `echo`, `cat`, `ls -la`, `grep`, `wc`, `date`, `id`, `pwd`,
`env`, `hexdump`, `sha256sum`, `head`, `sort`, `sed`, `tr`, `du`, `stat`,
`find`, `cp`, `mkdir`, `rm`, `uptime`, `printf`, `sleep`.

## Forty-six syscalls, and not one of them was guessed

Stage 0's whole argument was that Linux has no specification you can test
against, so what is worth owning is a record of what a real binary asks for.
That record did its job: sixteen applets were swept, every gap they named was
implemented, and nothing else was. Twenty-one calls became forty-six, and after
the work `sysinfo` was the only call left unserved in the entire sweep.

The *shape* of two findings is what makes the case for the instrument.

`ls` ran perfectly on its first attempt. It opened the directory, walked it
with two `getdents64` calls, `lstat`ed every entry, and exited 0 -- and printed
nothing at all, because everything it had to say went through `writev`, which
answered `-ENOSYS`. **A program that works and is silent is the worst shape a
missing syscall can take**, and it is exactly the shape a trace makes visible
and nothing else does.

`hexdump` reported "Function not implemented" about a file it was holding open,
because it does `dup2(fd, 0)` to read its input as stdin. That was a
limitation this project had written down the day before as a deliberate
refusal, and it took one applet to turn the note into a bug worth fixing.

## Fixed-address binaries are placed, not refused

Busybox's prebuilt is `ET_EXEC` with an entry at `0x4038b1`, and so is nearly
every prebuilt static binary in the world. The loader refused all of them, so
that was not a corner case: it was most of the software the loader exists to
run.

The refusal read "a single address space has no free range to promise it",
which was true of what the kernel *knew* rather than of the machine. Nothing
could answer "does anything own four megabytes at four megabytes".

**The frame allocator cannot answer it, and why is the interesting part.** It
is a bump allocator whose cursor is forward-only by design, so by the end of
boot it sits past the heap, three hundred megabytes up, and everything behind
it reads as unavailable -- including large conventional regions it merely
stepped over while looking for one span big enough for the heap. `0x400000` is
exactly such an address: untouched, and invisible to the only thing you would
think to ask.

So the free set is computed the other way round: every region the firmware
called conventional, minus the handful boot actually took. That is computable
because a bump allocator never frees, so its history is a short list.

    [boot] placeable  7 MiB free below the heap and above it, largest run 6 MiB

## A guest can write, inside a jail

Writes were refused everywhere on the grounds that a write to a
content-addressed store is a new root hash, so an unrestricted `O_WRONLY`
routes a guest binary around every gate `sysbox` puts in front of the shell.
The reason was sound and what it argued for was a jail rather than a refusal.

    cp /tmp/lines.txt /tmp/copy.txt   then   ls /tmp
    busybox  copy.txt  lines.txt  newdir

    mkdir /ai/nope
    mkdir: can't create directory '/ai/nope': Read-only file system

`/tmp` because that is already the scratch area. Everywhere else is `EROFS`,
checked on the resolved path so a relative one cannot be written to climb out.
Writes buffer and commit on the last `close`, because the store is keyed by
content and every commit re-addresses the whole blob.

## Two bugs in the fault path, and neither was about Linux

**A longjmp into inlined code has no calling convention to lean on.**
`recover::guard` saved the registers a callee must preserve, which is the right
list for a function boundary and the wrong question: `guard` is inlined into
its callers, so the compiler keeps a caller's live value in `rax` as readily as
in `rbx`. Found twice, both times as a wild pointer in a subsystem with nothing
to do with faults -- a PML4 walk with its index out of `rsi`, then the heap's
free list walked from a cursor out of `r9`. The second only appeared because an
unrelated file changed what the register allocator did, which is the tell that
a list chosen from a calling convention was never going to be the answer. All
fifteen are saved now.

**A guest at ring 3 could halt the kernel with two ordinary syscalls.** `lseek`
past the end of a file is legal, and `read` then indexed a slice at the cursor
without clamping it. Measured against a build with the fix reverted, because
the fix is one `min` and the interesting part is that nothing about ring 3,
page rights or the pointer checks was wrong. It was an ordinary Rust index on a
value the guest chooses.

## What is still missing, and it is one thing

`fork`, `execve` and `wait4`: `sh` running anything that is not a builtin. That
is not a syscall away. `fork` needs two address spaces, and one address space
is the founding claim of this system rather than a shortcut it took. Nothing
else in the measured surface is blocked on a decision that large, and it
deserves to be made deliberately rather than arrived at.

Also absent and smaller: dynamic linking, threads, sockets, and signal
*delivery* -- `rt_sigaction` is accepted and never fires, which is honest here
because nothing can raise a signal at a guest.

## Verified

    busybox   24 applets exercised, every one exiting 0
    sha256sum matches the host digest byte for byte
    diag linux                                  91 claims
    diag place                                   9 claims
    diag all                                    37 passed, 0 failed

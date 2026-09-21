# GLaDOS 1.3.5 — it runs software it did not compile

Fifty-two commits since 1.3.4. The short version: this kernel now loads and
runs unmodified x86-64 Linux binaries, and one of them draws on the screen.

## The two measurements that matter

**busybox, fetched from busybox.net and touched by nothing here, hashing its
own file at ring 3 under glibc:**

```
linux run /tmp/busybox sha256sum /tmp/busybox
6e123e7f3202a8c1e9b1f94d8941580a25135382b99e8d3e34fb858bba311348
```

That digest is the one the host computes over the same bytes. It is the first
end-to-end check of this path against an implementation nobody here wrote: the
bytes went from FAT into the namespace, out through the POSIX projection,
through busybox's own hash at ring 3, and back through `writev`.

**An unmodified SDL2 program, on a machine with no display server:**

```
video driver: glados
display: 1280x800, format SDL_PIXELFORMAT_RGB888
surface: 1280x800 pitch 5120
frame on screen
```

The program is ordinary SDL2 and knows nothing about this machine. The
screenshot is eight bars with a diagonal across them, and the diagonal is the
point: bars alone look correct under a wrong stride as long as it is a multiple
of the bar width, and a diagonal cannot. It runs straight, corner to corner.

## What arrived

**Processes.** `fork`, `execve`, `wait4` with `WNOHANG`, and signals with a
real `ucontext` on the guest's own stack in Linux's exact layout. That last one
matters beyond correctness: a handler that edits `uc_mcontext` changes where
the program resumes, which is the mechanism Wine's fault emulation runs on.

`fork` needed two address spaces, and one address space had been the founding
claim of this system rather than a shortcut. `src/mem/space.rs` is what
removed it: a second page-table root, per-task, with the confinement above a
512 GiB window as the entire safety argument.

**Communication.** Unix domain sockets, the socket syscall surface, and
`SCM_RIGHTS`, so a file descriptor can travel over a connection. A descriptor
is attached to a *position in the byte stream* rather than to the connection,
and getting that ordering right is most of what the work was.

**Waiting.** `poll`, `select`, `ppoll`, `pselect6`, and the whole `epoll`
family. Every remaining target sits inside one of these: Wine's server loop,
SDL's event pump, and every Wayland client.

**Pipes**, which is the `|` behind half of all software, and `memfd_create`
with `ftruncate` and shared mappings that genuinely alias — which is how a
Wayland client hands a compositor a buffer.

**A screen and a keyboard for guests.** `/dev/fb0` gives a guest the actual
framebuffer rather than a copy, and `/dev/input/event0` and `event1` are real
evdev devices fed from the same place this kernel's own drivers converge.

**Skywalker**, the display server, begun. The Wayland wire format, the object
space, and `wl_display` with `wl_registry` — everything a client does before it
has asked for anything. Nothing draws through it yet.

**Elsewhere:** a device registry that says what is fitted and what would drive
it, gigabyte pages, a fault reporter that keeps every register, OpenGL through
OSMesa, and the honest finding that OpenGL turned out not to be kernel work at
all.

## What this is not

It is not full Linux compatibility and the number says so: **97 syscalls of
Linux's ~350.** Of the hundred or so calls a normal program actually reaches,
fifteen are still missing — `chdir`, `rename`, `statx`, `fsync`, `madvise`,
`eventfd`, `clone3` and others where a program either has a fallback or does
not care.

Deviations that are decisions rather than gaps, each argued in the code: a
guest may write only inside `/tmp`; there are no permissions, owners or times;
`..` in a path is refused rather than normalised; an open file holds its whole
contents with a 64 MiB cap; signals arrive only on the way out of a syscall, so
a program spinning in a loop cannot be interrupted; glibc needs `LD_BIND_NOW`.

There is no audio driver, no GPU driver, and no 32-bit support. A 32-bit Linux
binary is refused at parse with `not 64-bit`, which rules out most older games.

## Two bugs worth naming

`unix::close` was called from nowhere in the kernel. A descriptor went away and
the far end was never told, so a reader got `EAGAIN` forever where it was owed
a zero. Every fixture so far closed both ends by exiting, so nothing had been
in a position to notice. For a socket that is a hang; for a pipe it is the
entire protocol.

`page_up(0)` is 4096 rather than 0, because it carries a `.max(1)` so a
zero-length allocation still gets a page. A first `ftruncate` therefore
compared the new size against the old, found them equal, and set the length
without allocating. Three hypotheses were refuted by reading before printing
two fields settled it in one line.

## Verification

`diag all` runs 42 suites; `diag linux` alone is 255 claims. The gate for this
release is both sweeps in one boot plus every fixture, which passes with the
machine alive afterwards. Every syscall added here has a hand-assembled fixture
that exercises it from ring 3, because a claim about pure arithmetic and a
syscall that has actually run are different kinds of evidence.

One known defect is carried forward rather than closed: `diag paging` followed
by `diag smp` has faulted in the past on a page-table entry, needing the sweep
twice in one boot to appear. It did not reproduce across three attempts here.
Not reproducing is not a fix, so it stays on the list.

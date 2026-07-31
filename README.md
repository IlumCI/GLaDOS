# GLaDOS

A from-scratch, non-Unix, ring-0 operating system for one specific laptop —
an **MSI Thin GF63 12UC**, board **MS-16R8** — built around a language model
that lives *inside* the kernel rather than on top of it.

There is no user/kernel split, no syscalls, no process isolation, and no
address space but the one. The model runs at CPL 0 with the same view of
memory as the page fault handler. When it selects a tool, there is no IPC,
no serialisation, and no sandbox to cross — the call is a function call.

TempleOS is the lineage: single address space, everything at ring 0, no
syscalls, written by one person from nothing. It is not the target. TempleOS
put a programming language where the shell goes. GLaDOS puts a model there.

## The idea

Every "AI operating system" so far is an application: a model in userspace,
talking to the kernel through the same syscall interface as a text editor,
handed a sandbox and a JSON tool schema and asked to pretend it is an agent.
The boundary is inherited from an assumption — that untrusted code runs here —
which does not apply when there is exactly one program and one user.

Take the boundary away and things that are normally hard become arithmetic:

- **The KV cache is a filesystem object.** Not a serialised copy of one — the
  same bytes, in the content-addressed store, snapshot-able and rollback-able
  like any other object. Forking a conversation is a hash copy.
- **Tool selection cannot emit an invalid tool.** The applet name is decoded
  under a grammar built from the live applet table, so a name that does not
  exist is not an improbable output, it is an unreachable one. Read-only mode
  is enforced by removing mutating applets from the reachable set, before
  sampling, not by checking afterwards.
- **The model is a scheduler task.** `think` runs it in the background against
  the same run queue as the shell and the clock. It is preempted like anything
  else, and its FPU state is saved like anything else.
- **It can read its own machine.** `peek64`, `inb`, the page tables, its own
  weights. There is no privileged operation to ask permission for.

The safety story is not isolation, because there is none. It is that the
dangerous operations are gated on measurements, and the measurements are in
the repository.

## Watch it work

```
glados> route trusted make a folder for notes
[route]
  mkdir
  all 3 cores agree (measured 90% right when they do)
  1602 us, no transformer involved
  that applet mutates content

glados> http example.com /
[http] example.com:80/
  104.20.23.154
  HTTP/1.1 200 OK
  828 B in 38 ms, 559 B of body
  | <!doctype html><html lang="en"><head><title>Example Domain</title>...

glados> i=0 while(i<8){ rect(i*40,300,32,32,i+1) i=i+1 }
```

Anything typed at the prompt that is not a command is evaluated as code.
Integers are 64-bit, strings concatenate with `+`, and the builtins include
framebuffer drawing, timing, and — because everything is ring 0 — raw
`peek`/`poke`/`in`/`out`. A bad `peek` faults, and the fault reporter names
the address; that is the intended debugging loop rather than an accident. An
evaluation-step budget stops `while(1){}` from wedging the shell, since there
is no Ctrl-C to rescue you.

## State

All of it verified running under QEMU, not merely compiling.

| | | |
|---|---|---|
| M0–M1 | Toolchain, ESP staging, UEFI entry, GOP framebuffer, `ExitBootServices`, own PML4 | done |
| M2–M3 | Framebuffer console, GDT+TSS, IDT with fault reporting, frame allocator, heap | done |
| M4 | ACPI, LAPIC/IOAPIC, APIC timer, i8042 keyboard, preemptive tasking, shell | done |
| M5 | Lexer, parser, tree-walking interpreter with kernel builtins | done |
| M6 | TSC microsecond timing, console scroll as a memmove, typewriter pacing | done |
| M7 | `sysbox` — a Merkle namespace and its applet set | done |
| M8 | NVMe driver, content-addressed store, snapshot and rollback | done |
| M9 | Model loader, tokenizer, sampling, generation | done |
| M10 | Constrained decoding under a live grammar, tool-selection harness | done |
| M11–M12 | Adapter head, KV cache as a store object, model as a resident task | done |
| M13–M14 | Synthetic tool-selection corpus, honest evaluation protocol | done |
| M15 | Ridge probe and the three-core council | done |
| M16 | Verified self-modification search | done |
| M17 | Wall-clock RTC, self-snapshotting | done |
| M18 | Content-addressed package manager, modal editor, FAT16/32 reader | done |
| M19 | e1000 NIC, ARP, IPv4, ICMP echo | done |
| M20 | TCP: active open, retransmission, HTTP | done |
| M21 | UDP, DNS, DHCP | done |
| M22 | Named interfaces, routing, loopback | done |
| M23 | TLS 1.3, with certificate validation | done |
| M24 | WPA2 supplicant crypto (no driver to run it on) | done |
| M25 | A wireless driver, once the GF63 names its card | next |

## The model

**SmolLM2-135M**, 30 layers, dim 576, 9 query heads over 3 KV heads, vocab
49152, RoPE θ=100000 — quantised to int8 with a per-row scale, 129 MB on the
ESP. The weights are the one artifact here that this project did not produce;
`tools/convert.py` flattens them from safetensors into the layout
`ai::model::offsets` indexes by arithmetic, because parsing JSON and
rearranging 134 M values inside a kernel with no debugger is a bad trade
against 200 lines of Python that runs once.

The loader identifies the format by content rather than filename, so
karpathy's llama2.c checkpoints still load from the same path.

## The interesting result

The obvious way to route a task to a tool is to ask the transformer. It works:
`act` decodes the applet name token by token under the grammar, and gets there.
It is also slow, and it is not the best answer.

`route` instead reads one hidden state out of the model and hands it to a
**closed-form ridge regression** — Widrow-Hoff, 1960 — solved by Cholesky
in-kernel. 12,672 parameters, no epochs, nothing to overfit:

```
[probe]
  357 train, 108 held out, 21 classes
  seen      99%
  held out  77%   <- the one that counts
  chance is 4%
```

Three "cores" vote: the probe, a multinomial naive Bayes over hashed
character n-grams, and a lexical matcher. **Their agreement is the useful
signal, not their vote** — the council does not beat the probe on accuracy,
but when all three agree the answer is right 90% of the time versus 61% when
they split. That gap is what the confidence gate acts on.

```
[gate]
  all agree    72/108  90% right
  they split   36/108  61% right
  overall     80%
```

A 65-year-old linear method, on features the transformer computed, answering
in **1.6 ms with no transformer forward pass at inference**. That result came
from being told to stop reading only recent papers.

### On the evaluation

The three-way split (`SEED_TRAIN` / `SEED_VAL_END` / test) exists because
selection-on-held-out was done wrong three separate times: a grid sweep scored
on the test set, cross-validation folded by template family, and a test set
that *moved* every time the corpus was appended to, so 77.6% → 75.0% read as a
regression when nothing had regressed. `search` now reports the validation
number as spent and reads the test set once:

```
  adopted: lambda 1/10, rule majority
  validation 87%  (spent -- selected on, so optimistic)
  test       77%  (54 items, read once)
  a configuration is adopted only when measured better, never argued better
```

Negative results are kept in the tree rather than deleted. Training the
adapter head *hurts* at this data scale. The Product-of-Experts council does
not improve accuracy. Both are in the repository, because the reason to know
them is the same reason they were worth measuring.

## Build and run

```powershell
.\scripts\run.ps1                # debug build, boot in QEMU
.\scripts\run.ps1 -Release
.\scripts\run.ps1 -Gdb           # pause for gdb on :1234
.\scripts\run.ps1 -TraceFaults   # log every exception -- finds triple faults
```

Bare metal, once an ESP exists on the USB SSD:

```powershell
.\scripts\deploy.ps1 -EspDrive S:
```

Then reboot and hold **F11**. `deploy.ps1` writes exactly one file, creates no
partitions, and formats nothing. The internal Windows NVMe is never involved.

## Design

**No bootloader stage.** UEFI already delivers long mode, CPL 0 and an identity
map, so this UEFI application *is* the kernel. That removes ELF loading,
relocation, and a handoff ABI — three entire categories of bug, for free.

**Ring 0, single address space, identity-mapped.** No TSS `RSP0` juggling, no
SYSCALL/SYSRET setup, no per-process page tables, no TLB shootdown design. A
context switch is GPRs plus `RSP`, because there is no privilege change.

**Page 0 is deliberately not mapped.** The first 2 MiB is mapped at 4 KiB
granularity purely so that entry 0 can stay absent. Costs one page table, turns
every null dereference into a `#PF` the reporter can name.

**Faults report to the framebuffer.** The GF63 has no serial port. On real
hardware, pixels are the only channel by which a diagnostic reaches a human, so
the exception handlers draw. Under QEMU everything is mirrored to COM1.

**Storage is content-addressed.** Objects are named by SHA-256 of their
contents and assembled into Merkle trees, so a copy is O(1), identical data is
stored once, and a snapshot is a single root hash. The hash covers content
only and never block locations — otherwise moving a block would change an
object's name.

**NVMe writes are locked by default.** `store::init` unlocks them only after
`find_store_region` has named a target — a partition tagged with the GLaDOS
type GUID, or unclaimed space past every partition on a bare image — and
`Store::format` re-checks that the region is inside our own partition or
overlaps nothing at all. On a disk fully allocated to Windows with no GLaDOS
partition, there is no such region and initialisation fails. That is the
intended outcome, not an inconvenience. Every error path re-locks; leaving it
open on the way out is how a safety mechanism becomes decorative.

**No dependencies.** The only code in the kernel this project did not write is
the Rust `core` library. That is mostly a choice and partly a constraint:
there is no MSVC `link.exe` on this machine, so anything that runs on the host
at build time cannot be built. A plain library crate is fine — `typenum`
compiles for the UEFI target in under 4 seconds — but a proc-macro crate dies
with *"linker `link.exe` not found"*, and so does `-Zbuild-std`, which is why
the custom target in `x86_64-glados.json` is still unusable. That is the real
reason vendoring a TLS stack is an obstacle rather than a preference. The
host-side tools in `tools/` do use Python and numpy; they run once, on a real
computer, and none of their output is trusted without being re-checked
in-kernel.

## Hardware notes

Surveyed from the running Windows install, not assumed.

| | |
|---|---|
| CPU | i7-12650H, Alder Lake, 6 P-core + 4 E-core. **BSP only** — SMP across asymmetric cores is its own project |
| Firmware | UEFI, **Secure Boot off**, so unsigned `BOOTX64.EFI` loads directly |
| Boot mode | No CSM assumed: no real mode, no INT 10h, no VGA text mode, no `0xB8000` |
| Display | Intel UHD via GOP. The RTX 3050 is mux-less and irrelevant — the iGPU owns the panel |
| Keyboard | **i8042** (`ACPI\MSI0007`) — ports `0x60`/`0x64`. Most 2022 laptops route the internal keyboard over USB-HID, which would mean an xHCI + HID stack before you could type one character. This one does not |
| Storage | Boots from a USB SSD. UEFI's own drivers load us *before* `ExitBootServices`, so no USB driver is needed to boot. Persistence targets the internal NVMe (~600 lines) rather than USB Mass Storage (several times that) |
| Network | Unidentified. The driver written so far is for the **e1000**, which is what QEMU emulates; the GF63 is most likely a Realtek RTL8168 and is not yet supported |

## The boot disk is counterfeit

The USB SSD advertises **976.56 GB** and actually holds **14.67 GB**. It was
sold with a generic `VendorCo ProductCode` bridge, and the controller accepts
writes past the real end of flash, discards them, and returns garbage on read.

This was not obvious. It presented as a series of unrelated Windows failures —
`diskpart` refusing to format, then `format.com` reporting *"Invalid media or
Track 0 bad - disk unusable"* — because every failing write happened to be at
the 966 GB mark, where the original shrink-based layout put the ESP.
`scripts/probe-media.ps1` settled it by writing a self-identifying pattern and
reading it back; `scripts/find-capacity.ps1` binary-searched the boundary in
15 probes.

Consequences baked into `scripts/build-layout.ps1`:

- **MBR, not GPT.** GPT keeps a backup header and partition array in the last
  sectors of the *reported* size — 976 GB, inside flash that does not exist.
  The disk would be permanently missing half its partition table. MBR keeps
  everything in LBA 0.
- **ESP first, at 1 MB**, in the most thoroughly verified region.
- Everything above the layout is left unpartitioned so Windows never offers
  the fictional capacity to anything.

Do not put anything you care about on this device. It is acceptable as a boot
target only because the entire payload is reproducible: a 745 KB binary from
git, plus a model file regenerable from `tools/convert.py`.

## Bugs worth remembering

**`extern "C"` on this target is Microsoft x64, not System V.**
`x86_64-unknown-uefi` is a Windows-ABI target. The context switch assembly read
its arguments from `rdi`/`rsi`, but the compiler was passing them in
`rcx`/`rdx`. The garbage in `rdi` became a stack pointer that happened to land
in the framebuffer aperture, so the task's saved frame was written into video
memory and `ret` popped a black pixel into `rip`. The context switch is now
pinned to `extern "sysv64"` explicitly rather than inheriting a target default.

**Do not take the max over every UEFI memory descriptor.** OVMF describes
`Reserved` space to `0x100_0000_0000` — a clean 1 TiB. Using that as a map
limit exceeds what one PDPT covers, so the identity map silently failed to
build and we fell back to firmware page tables, which map page 0 — which in
turn made the null-dereference self-test pass without faulting. A masked
failure disabling the test meant to catch it. `max_ram_address` now uses an
allowlist of genuinely-RAM types plus a hard clamp.

**A guarded match arm placed after the arms it guards is unreachable.** The
exclusion check in `engine_free()` sat below the patterns it was supposed to
constrain, so it never ran — for several commits, while the compiler said so
in a warning nobody read. Exclusion is now enforced in `with_engine` by task
id, where it cannot be bypassed by pattern order.

## Known gaps

- **None of the AI stack has run on the real machine.** Three claims are
  waiting there: the AVX2 int8 kernel has never executed (QEMU's default CPU
  reports `avx2=0`), the attention window has never run at a realistic size,
  and the generation baseline of ~8 s/token is very likely a QEMU artifact.
- **DHCP does not renew.** The lease time comes back and is reported, and then
  nothing watches it. A machine left running past its lease keeps using an
  address the server believes is free.
- **TLS validates but does not enforce.** The chain, the transcript signature,
  the dates and the name are all checked, and the result is *reported* —
  `https` prints the body either way. That suits a system whose purpose is
  inspection and is the opposite of what a browser should do; a caller that
  cares must check `identity.ok()`.
- **No revocation.** No CRL, no OCSP, so a withdrawn certificate is accepted
  until it expires. No `iPAddress` subjectAltName either, which is why
  `https 1.1.1.1` is reported as a name mismatch rather than matched.
- **Roots are the host's.** `scripts/fetch-roots.ps1` copies the Windows
  trusted-root store, so GLaDOS inherits whatever that machine trusts —
  including anything added by management software. `trust` lists them; with no
  bundle, nothing validates, which is the right default.
- **Entropy is the TSC.** Private keys and nonces come from a cycle counter,
  which is not a random number generator. `RDRAND` exists on this CPU and is
  the fix.
- **No wireless driver.** `wlan0` is a slot that identifies hardware and
  refuses to pretend. The *supplicant* now exists — PBKDF2 for the PMK, the
  802.11 PRF for the PTK, RFC 3394 key unwrap for the GTK, and the four-way
  handshake, all checked against the IEEE 802.11i vectors at boot — but there
  is nothing to send a frame over. What remains is a signed firmware blob, an
  asynchronous host-command protocol, and 802.11 framing; see the header of
  `net/wifi.rs`. None of it can be started under QEMU, which emulates no
  wireless hardware at all, so the card must name itself on the GF63 first.
- **No plain-HTTP alternative to adopting a TLS stack.** Vendoring one
  will not be as simple as adding a dependency — rustls needs proc macros and
  ring builds C, neither of which this toolchain can compile. It would have to
  be hand-vendored.
- **TCP is a client, and one at a time.** No listening socket, no reassembly
  queue, no congestion control beyond a fixed four-segment flight cap, and the
  connection only advances while the shell is idle or a call is blocking —
  there is no interrupt-driven receive.
- **`Racy<T>` is not a lock.** It is single-core interior mutability, and the
  designated grep target for the day SMP arrives.
- **Exception handlers do not dump general-purpose registers.** The
  `x86-interrupt` ABI hands us the hardware frame, but the compiler's prologue
  has already clobbered the GPRs. Capturing them needs a naked assembly stub
  per vector. `RIP` plus `CR2` diagnoses most early faults, so this is
  deliberately deferred rather than faked.
- **The framebuffer is mapped write-back, not write-combining.** Correct, just
  slower than it could be. Needs PAT setup.
- **No NX.** Every mapping is `PRESENT | WRITABLE`. Requires `EFER.NXE` plus
  splitting text from data.
- `BootServicesCode`/`Data` are not reclaimed. They are genuinely free after
  exit, but our current stack is probably sitting in one of them, so the early
  frame allocator restricts itself to `Conventional` memory.

## Layout

~19,500 lines of Rust.

```
src/
  main.rs        UEFI entry and boot sequence
  uefi.rs        hand-written firmware bindings -- field order IS the ABI
  sync.rs        Racy<T>: single-core interior mutability. Grep target for SMP
  shell.rs       command dispatch; anything unrecognised is evaluated as code
  task.rs        preemptive round-robin tasking, fxsave per task
  time.rs        TSC calibrated against the LAPIC timer
  edit.rs        modal editor, nvim-shaped
  pkg.rs         content-addressed package manager
  net/           interfaces and routing, ARP, IPv4, ICMP, TCP, UDP, DNS,
                 DHCP, TLS 1.3 + X.509 + trust store, WPA2 supplicant,
                 and an honest wlan0
  crypto/        sha-1/256/384, hmac/hkdf, chacha20-poly1305, x25519,
                 aes + rfc3394, rsa, ecdsa p-256/384, montgomery bigint --
                 eleven vector sets, all re-run at every boot
  recovery.rs    console reachable when the store will not mount
  acpi.rs        RSDP/XSDT walk: MADT, MCFG, FADT
  cpu/           GDT+TSS with IST stacks, IDT and the fault reporter, port I/O
  mem/           frame allocator over the UEFI map, identity map, kernel heap
  gfx/           linear framebuffer, 8x8 font at 2x, scrolling console
  lang/          lexer, parser, tree-walking interpreter
  dev/           lapic, ioapic, pic, i8042, pci, nvme, rtc, e1000
  store/         block layer, SHA-256, content-addressed store, FAT16/32 reader
  sysbox/        the Merkle namespace and its applets
  ai/            model, weights, tokenizer, sampling, constrained decoding,
                 corpus, ridge probe, council, harness
tools/           host-side: checkpoint conversion, corpus generation, evaluation
scripts/         build, run, deploy, and the disk-forensics scripts
```

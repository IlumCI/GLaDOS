# GLaDOS

A from-scratch, non-Unix, ring-0 operating system in Rust, built around a
language model that runs inside the kernel.

Inside the kernel there is no privilege boundary and nothing to marshal across
one. The consequence that matters is what a tool call becomes: the sampler
picks a name from the live applet table under a grammar compiled from that
table, and the kernel calls the function. Nothing is serialised and nothing is
parsed in between.

**There are two stories on this machine and neither is a retreat from the
other.** The kernel has no boundary because removing it is the whole idea. A
program written somewhere else has not earned the machine, so it gets ring 3,
its own page-table root and a syscall surface -- 93 of Linux's roughly 350,
enough that unmodified busybox, an SDL2 program and a dynamic linker all run
here without knowing where they are.

It boots as a UEFI application, which means it *is* the kernel. Firmware
already delivers long mode, CPL 0 and an identity map, so the boot path has no
bootloader, no ELF loading, no relocation and no handoff ABI. The ELF loader
that exists is for running other people's programs rather than for starting
this one. The model, tokenizer and root certificates are read before
`ExitBootServices`, because that call is the last moment a filesystem exists.

198 files of Rust, about 132,000 lines, of which 178 files and 118,000 lines
were written for this project; the rest is a DOOM port and one file of Realtek
register tables, both credited below. TempleOS is the obvious ancestor, and the
identity map here exists for the reason Terry Davis gave for his.

---

## Status

A research kernel for one specific laptop, built to answer one question: what
changes when a language model is a kernel primitive? The table is what works.
The list after it is what does not, which on a project like this is the more
informative half.

Works, and verified:

| | |
|---|---|
| Boot | UEFI application, own page tables, GDT/IDT, APIC timer, i8042 keyboard |
| Memory | Physical frame allocator, identity paging to 4 GiB, coalescing heap |
| Tasks | Cooperative and preemptive at 100 Hz, `sysv64` context switch |
| Ring 3 | Guests at CPL 3 on a page-table root of their own, entered by `iretq` and left by `sysretq`, with a guest fault ending the guest and not the machine |
| Linux ABI | 93 syscalls: `fork`, `execve`, `wait4`, threads and futexes, signals with a real `ucontext` on the guest's stack, pipes, `epoll`, Unix sockets with `SCM_RIGHTS`, `memfd`, `/proc`, `/dev/fb0`, evdev |
| Ported | DOOM on FreeDoom, unmodified busybox under both musl and glibc, an SDL2 program drawing through the framebuffer |
| Graphics | GOP framebuffer, a Frutiger Aero desktop, window manager, taskbar, ten apps |
| Text | 325 glyphs at 8x8, UTF-8 in the console, Latin-1, Greek, box drawing and maths |
| Storage | NVMe, content-addressed object store, Merkle trees, snapshots, ranged write gate |
| Network | ARP, IPv4, ICMP, UDP, TCP, DHCP, DNS, TLS 1.3 with chain validation |
| Drivers | e1000, RTL8168, xHCI (USB 3), CDC-ECM USB Ethernet, USB keyboards and mice |
| Crypto | SHA-1/256/384, HMAC/HKDF, AES, ChaCha20-Poly1305, X25519, RSA, ECDSA |
| Model | Qwen3, Qwen3.5 hybrids and SmolLM2, int8, in-kernel inference |
| Routing | Constrained decoding over the live applet table, plus a closed-form probe |
| Agent | Propose, validate, execute, observe, with the grammar as the permission system |
| Initiative | A resident task that decides for itself when to act and when to stay quiet |
| Language | Aiksi: lexer, parser, tree-walking interpreter, records, types, capabilities |
| Codegen | An x86-64 back end for the integer subset, checked against the interpreter |
| Formats | Detection and structure for text, markdown, json, jsonl, xml, csv, ini and eight languages |
| Power | Digital thermal sensor, measured frequency, HWP governors, all behind a CPUID gate |
| ACPI | An AML interpreter: namespace, evaluator, operation regions, embedded controller |
| Battery | Charge, health and time remaining from the firmware's own bytecode, and S5 power-off |
| Training | Gradients, Adam, QDoRA over the classifier and over every q/k/v site |
| Self-modification | Variant lineage, a judge council, an append-only ledger, O(1) rollback |
| Updates | Signed staged images swapped before `ExitBootServices`, with rollback |
| SMP | Per-core GDT/TSS/APIC, a real spinlock, shared heap and console, tasks that migrate |
| Accounting | Every allocation billed to the task that made it, with peak and outstanding |
| Mining | Midstate-cached SHA-256d with an honest scoreboard |

Does not work yet:

- **Wireless.** The built-in card is CNVi, so the MAC lives in the PCH and the
  M.2 module is only a radio, reachable through an undocumented signed-firmware
  protocol. The WPA2 supplicant is complete and checked against IEEE 802.11i
  vectors at every boot, and has never had hardware to run on. A USB dongle
  driver (RTL8188EU) has its register layer and power-on sequence, and stops
  short of PHY, radio and firmware upload.
- **Real tasks on more than one core.** The machinery is done and proven: every
  core has its own descriptor table, task-state segment, per-core block and
  idle task, the scheduler carries a task between cores and back, and `diag
  migrate` demonstrates one running on two of them. What is missing is the
  audit. Preemption on one core means two tasks never execute at the same
  instant; on two cores they genuinely overlap, so every `Racy` reachable from
  two different tasks becomes a live race, the namespace tree among them. So
  every task this kernel spawns is pinned to core 0 on purpose, `unpin` exists,
  and nothing calls it. Unpinning before the audit buys a kernel that passes
  every test and corrupts something later.
- **A hardware entropy source.** The generator is a fast-key-erasure ChaCha20
  DRBG fed by keyboard and mouse interrupt timing and by NVMe completion
  latency, and it refuses to answer for key material until it has seen enough
  events. That refusal is correct and it is also a limitation: a machine that
  boots, touches no disk and shuts down with no key pressed never seeds, and
  TLS then falls back to timing-derived keys and says so during the handshake.
  The CPU has RDRAND and this does not use it, because trusting an opaque
  instruction is a different argument from trusting interrupt timing.
- **A fault report on the framebuffer.** Painting from inside an interrupt gate
  raises a general protection fault here, so the report goes to the serial port
  in full before the console is attempted at all. On the laptop there is no
  UART, which means a fatal fault there currently prints one line and halts.
  The bug belongs to the console and it is now visible instead of silent.
- **An authored application reaching adoption.** The machine writes drafts and
  never adopts one. The planner's output is likewise stringified into a report
  rather than gating how much the loop attempts.
- **A Wayland client on screen.** `src/sky/` has the wire format, the object
  table, `wl_display`, `wl_registry` and `wl_callback`, and it says so at the
  top of its own module: **nothing draws yet.** `wl_compositor`, `wl_surface`
  and `wl_shm` come next, then `xdg_shell`. The compositor half is not the
  problem -- `gfx::compose` already owns the screen and stacks windows -- and
  until a client actually paints, none of that is evidence of anything.

---

## The machine that changes itself

The capabilities above that are newest deserve a paragraph each, because
together they are the point of the project.

**The model trains inside the kernel.** `train adapter` fits a QDoRA adapter
over the classifier from the corpus in the namespace. The base weights stay
frozen, so every hidden state is a constant and gets cached once per example;
an epoch after that costs no forward passes at all. Restricted cross-entropy
zeroes the gradient outside the grammar's candidate set, so the only rows that
move are ones the decoder can reach. On SmolLM2 that is 132 rows out of 49,152,
which is what makes the whole exercise affordable on this hardware. Real-corpus
training refuses to run without AVX2, because scalar emulation would turn every
hyperparameter judgement into a judgement about the clock.

**Training goes deeper than the classifier now.** `deeptrain` moves every q/k/v
site through a taped forward pass and a backward walk over the layers. The
economics are different enough to deserve a different command: the classifier
path caches hidden states once and an epoch costs no forward passes, while
moving a projection in layer three moves every state after it, so every epoch
pays a forward pass per example. Measured under emulation on SmolLM2, 24
forward-and-backward passes in 110 s.

It goes through the judges as well, and how is the interesting part. Those
judges rest on cached features and a deep adapter moves the features, so a
trial prepared beforehand cannot judge what came out of it. Re-preparing one
afterwards does not work either, because decisions are recorded along the
baseline's own decode path and a change that alters that path alters how many
decisions there are. The comparison pairs on *routing* instead: one entry per
example, the same examples both times.

**A trained adapter is an object in the namespace.** `adapter save` writes a
`GLADOSA1` blob holding the adapter alone, never the checkpoint. Rows are
stored sparsely, because they are sparse in fact: on the measured decision
layer that is 23.7 KB against 1.79 MB dense, a factor of 75. Save, detach,
reload and save again reproduces the file byte for byte, which is checked by
comparing content addresses. `tools/adapter.py` reads the format on the host.

**Self-modification runs on certificates.** Schmidhuber's Gödel machine adopts
a rewrite once it can prove the rewrite raises expected utility, and it carries
a theorem prover to do it. There is no theorem prover here and none could be
built for a quantised transformer's future reward. What is substituted is a
certificate that is cheaper to refute than it was to produce, over
content-addressed inputs, so any later run re-derives the same verdict bit for
bit. That is affordable for one specific reason: producing a variant costs a
forward pass per example, and checking somebody else's claim about one costs a
dot product per cached decision. Four orders of magnitude between making a
claim and testing it is the asymmetry a proof system provides, reached by
another road.

Four judges have to agree, each covering a different way a variant can be bad:

- **J1, margin.** Both variants answer the same cached decisions, so the
  comparison is paired and McNemar's test applies. Nine repairs against two
  breaks scores 3.27 and fails the bar; twelve against one scores 7.69 and
  passes. Comparing two percentages would call both of those a win. The floor
  is symmetric, because "not significantly worse" alone once adopted a routing
  rule on a measured four repairs against ten breaks.
- **J2, its own goals.** The four goals the machine sets itself are cached
  along the path the frozen baseline walks for them. A variant that reroutes
  "list the files in /tmp" toward something that mutates the disk is
  catastrophic, and aggregate accuracy would never show it.
- **J3, structure.** Finite factors, positive scales, finite logits across
  every cached decision. This catches the variant whose validation accuracy
  improved while carrying a scale that overflows on the first unfamiliar
  prompt.
- **J4, cost.** Rank and resident bytes. The heap is one physically contiguous
  allocation on a ladder, so a lineage that grows with every adoption is a
  lineage that eventually fails to boot.

Adoption is a pointer swap and the parent stays addressed, so rollback costs a
pointer write. Every trial appends a line to `/ai/godel/ledger.txt` whether it
was adopted or rejected, with all four judges' numbers in it.

**The night branch rotates over every axis that has a judge.** It knew two jobs
and one always won the tie, so the adapter grid was walked to exhaustion while
the routing rule, deep training, a compiled skill and a core the machine wrote
were never tried unattended at all.

The rotation is a ranking rather than a round, and what it ranks by is worth
stating: `next_proposal` reaches for the axis whose next verdict the ledger can
*least* predict. An axis that has said yes to everything and one that has said
no to everything are equally predictable and equally uninformative, so a
Laplace-smoothed Beta posterior folded to a distance from the coin-flip puts
the axis nearest 50% first. Fairness is traded for information deliberately.
Smoothing is what stops that becoming starvation, ties break by slot so the
order is total and re-derivable, and the composed core is exempt and always
last, because producing its candidate costs a dozen decodes and is reached for
when everything cheaper is out of moves.

Two details that are load-bearing. The test slice carries a budget that lives
in the ledger, because a loop which improves itself forever reads the held-out
set forever and each read makes the reported figure more optimistic; past the
budget a test number prints as stale and is marked unquotable. And before the
judges run, the machine records whether training-set gain predicted a win, so
the ledger accumulates an answer to a question worth settling. Nothing acts on
that prediction, and at the current sample size it means nothing, which the
ledger says out loud.

Trials run when two independent facts agree: the real-time clock says the hour
falls inside a quiet window, and the entropy ring says no key or pointer
interrupt has fired. A clock alone does not know somebody is working late, and
silence at noon is a coffee break.

**Skills are judged before they are adopted.** A successful episode compiled
into a program used to be adopted by the act of writing it. A candidate is now
stored by content address and put through four judges of its own: it parses, it
runs under the powers an unadopted skill actually has, it repeats with the same
value and step count and touched set, and it is cheap. Trust is keyed to the
hash of the file, so editing a trusted skill revokes its trust by construction,
and granting trust is a shell command that is never an applet.

---

## Running programs this kernel did not compile

`src/linux/` began as a **measuring instrument rather than a loader.** The
expensive unknown in porting the Linux ABI is not the loader, which is a
weekend; it is that Linux has no specification you can test against, so a
subtly wrong `mmap` flag surfaces as a crash in a different subsystem an hour
later. gVisor needed 237 of Linux's roughly 350 calls to run containers. Before
committing to a privilege model or a syscall subset, the thing worth owning is
a trace of what a real binary actually asks for -- so every unimplemented call
is recorded and refused with `-ENOSYS`, and a run that ends there has named the
next call to implement.

**The first sixty-two were not guessed.** Sixteen busybox applets were swept
under musl and twenty-eight more under glibc, every gap they named was
implemented, and nothing else was. The thirty-one after those are the one place
this discipline was relaxed, and it is worth saying which way: processes,
pipes, `epoll` and `memfd` were implemented for targets that have *not* run
here yet -- Wine's server loop and a Wayland client -- rather than because a
trace demanded them. SDL2 is the only one of that group that has since been
run and confirmed. The shape of the findings was the useful part twice over. `ls` ran perfectly on its first attempt -- opened the
directory, walked it, `lstat`ed every entry, exited 0 -- and printed nothing at
all, because everything it had to say went through `writev`. A program that
works and is silent is the worst shape a missing syscall can take.

The evidence that this is real rather than a demo is a hash:

```
linux run /tmp/busybox sha256sum /tmp/busybox
b01eaede758499526db8c8ccd159b0f773ef0ecb29c25952e5c1042f5168e4ec
```

That is the digest the development machine computes over the same bytes. A
dynamically linked binary relocated a 1.9 MB libc through this kernel's `mmap`,
hashed its own file at ring 3 under a private page-table root, and got it right.

**Which libc is not a decision this kernel makes.** A libc is userspace: it is
linked into the binary or named by it in `PT_INTERP`, and the loader reads that
path and loads whatever is there. So musl and glibc are both installed and each
program takes its own -- which works today and needed no code. Writing a dynamic
linker was never the answer; loading `ld.so` was, and `dlopen` then becomes its
problem rather than ours.

**`fork` needed two address spaces, and one address space was the founding
claim of this system rather than a shortcut it took.** `src/mem/space.rs` is
what removed it. The decomposition that made it small is worth keeping: only
the image has an address the file demands, because the stack, the break and
every `mmap` are at addresses the kernel chooses. So `0x400000` is the whole
problem, and it is one region.

A guest fault ends the guest and nothing else:

```
linux run /tmp/wild
  ring 3, one address space -- only its own pages carry the U bit
  killed by fault 0x0e after 0 syscall(s), machine intact
```

At ring 0 the same fault stopped the machine, because a guest sharing an
address space with the kernel might already have corrupted anything. At ring 3
the kernel is intact by construction, so ending the guest is the honest
response.

**What this does not get you, said plainly.** There is no isolation from a
*hostile* guest worth the name, and stage 0 said so: contained bugs, not
malice. There is no scheduler that puts a guest on a second core. A guest that
makes no syscalls receives no signals, because delivery happens on the way out
of one -- a program spinning in a loop cannot be interrupted, which on Linux it
could, and the run deadline is what ends a runaway instead.

---

## Aiksi, the system language

`src/aiksi/` is the language everything above the kernel is written in, and the
intended relationship is C to Unix or HolyC to TempleOS. A program is
`code.ai&xi`, an extension chosen because nothing on the host has it.

```
use "/lib/text"

rec Host { name: str, port: int }

fn reachable(h: Host): int {
  if (tcp_connect(h.name, h.port, 600)) { tcp_close() return 1 }
  return 0
}
```

Records are values, so `b = a` copies and there is nothing to say about
aliasing. Types are optional and never inferred, and they are checked where a
value crosses a boundary somebody annotated. `use` is textual inclusion that
happens once, and the imported program runs with the importer's capabilities,
so an import can never be an escalation.

**`BUILTINS` is an allowlist and that is load-bearing.** Every row is a name, a
touch class and an arity range, and dispatch is refused for anything absent
from it before the arguments are examined. An arm added to the match without a
row is unreachable dead code; a row without an arm answers "no implementation",
which is broken and not dangerous. It replaced two denylists that were correct
until the language was wired to the network, and a denylist grants by default,
so the builtin anyone forgets is the one that matters.

**A code generator exists for the integer subset.** `src/aiksi/jit.rs` compiles
one function of integer arithmetic, `if`, `while` and `return` to x86-64, emits
it into a page-aligned heap buffer and enters it through a `sysv64` pointer.
Everything outside that slice is refused rather than approximated. It runs only
inside the differential harness and never on a live path, because a code
generation bug in a ring-0 image with no fault recovery gets exactly one
mistake.

What made that measurable came first. An Aiksi step costs about 14 ns, measured
for the first time by `core bench`, and a routing vote spends 20 steps against
a budget of 20,000. So the tree walk was never the cost the plan assumed, and
three rounds of removing allocations instead made a vote 2.9 times faster: the
kernel record table was being rebuilt per interpreter, a function's whole AST
was deep-copied per call, and a core's declarations were re-run per vote.

---

## Hardware

Developed against an MSI Thin GF63 12UC (board MS-16R8). That is the only
machine it has been meaningfully tested on.

It should boot on most x86-64 UEFI systems, since the graphics path is plain
GOP and the boot path assumes nothing vendor-specific. Storage and networking
are a different matter, because a driver has to match a chip, and the
memory-map handling has been tuned against one firmware. Expect a shell and a
working model on other hardware, and treat your disk and your network card as
open questions.

**The boot disk on the development machine is counterfeit**: it advertises
976 GB and holds 14.67. That is why the layout tooling uses MBR, a GPT backup
header having no real flash to land in, and why it carries a `SafeLimitGB`.

---

## Getting a model

The model is not in this repository. It is around 570 MB, it is not this
project's work, and a git repository is the wrong place for it. The ISO ships
with one; building from source means supplying your own.

The kernel reads three files from the EFI System Partition:

```
<ESP>/EFI/BOOT/BOOTX64.EFI      the kernel
<ESP>/GLADOS/model.bin          the model, converted
<ESP>/GLADOS/tokenizer.bin      the tokenizer, converted
<ESP>/GLADOS/roots.der          root certificates (optional)
```

Without `roots.der`, TLS still encrypts but authenticates nothing.

Download a checkpoint and convert it:

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir tools/qwen3
python tools/convert.py tools/qwen3 esp/GLADOS/model.bin --seq 512
python tools/tokenizer.py tools/qwen3/tokenizer.json esp/GLADOS/tokenizer.bin --verify
```

`--seq` sets the context window, and what bounds it is KV cache size instead of
the model. Qwen3-0.6B costs 112 MiB of kernel heap at 512 tokens, and
`convert.py` prints that figure so the decision gets made where `--seq` is
chosen, before it can surface as an allocation failure at boot.

**Always pass `--verify` to `tokenizer.py`.** It reimplements the kernel's
algorithm and diffs it against the reference `tokenizers` library, token for
token. A tokenizer that is subtly wrong produces text that still looks like
text, so nothing downstream will catch it for you.

Three families load. `convert.py` dispatches on `model_type`: `llama`, `qwen2`
and `qwen3` take the dense path and produce a v3 file, while `qwen3_5` produces
a v4 file with a layer-major body, because three layers in four hold a gated
DeltaNet mixer and the fourth holds full attention. Qwen3.5-MoE is refused at
load with `LoadError::Unsupported`, since the smallest published one is 71.9 GB
and nothing that size reaches a UEFI pool on this laptop; a forward pass for it
could never be run and contradicted, so none is written.

SmolLM2-135M also works, and it is the quickest checkpoint to develop against.

---

## Building

Rust nightly targeting `x86_64-unknown-uefi`:

```bash
cargo build --release
```

The artifact is `target/x86_64-unknown-uefi/release/glados.efi`.

### Running under QEMU

```bash
python tools/drive.py "initiative off" "agent stop" "diag all"
```

`drive.py` boots QEMU, stages the binary, resets NVRAM to pristine, and drives
the shell over a serial socket. It assembles its own ESP from the small
checkpoint, so it never disturbs deploy staging. **It prefers the release
artifact**, so a debug build alone leaves a stale binary staged and the change
under test never boots.

Pass `--qemu-extra "-accel whpx -cpu max"`. WHPX is the Windows hypervisor and
it is about 160 times faster than TCG on this workload; `-cpu max` alongside it
is what exposes AVX2, without which the trainer declines. It also raises
unmasked SSE exceptions faithfully where TCG does not, which is how a real bug
in per-task FPU initialisation was found.

Pass `initiative off` and then `agent stop` as the first commands: the resident
task wakes fifteen seconds in and holds the engine for a whole episode, and
stopping future ticks does not cancel the one already in flight.

**The large checkpoints do run here.** The synthetic FAT path is FAT16 on a
fixed geometry with a 516 MB ceiling, so it cannot stage one. `--stage-iso`
has no such cap: it builds a one-shot bootable image and boots that instead,
which is how any large checkpoint reaches emulation. Saying these models were
runnable only on the laptop was wrong on both halves, because the image path
existed and the hypervisor made it fast enough to matter.

Testing the staged-update path needs a real disk rather than the synthetic one,
which cannot do directory operations at all. `tools/mkesp.py` builds a raw
FAT32 image with an MBR, and `--esp-image` reuses it across boots so what the
guest wrote is still there next time.

### Deploying to hardware

```powershell
.\scripts\deploy.ps1 -EspDrive S: -Release
```

Then reboot and hold **F11** for the boot menu.

### Staged updates

The boot image is replaced by the *next* boot rather than the running one,
because the firmware's FAT driver is the only writer of the ESP that exists
while a boot image can still be swapped. Put three files on the ESP and reboot:

```
GLADOS/STAGED.EFI     the new image
GLADOS/STAGED.SIG     its detached GLADOSIG signature
GLADOS/UPDATE.FLG     any contents; presence is the request
```

**A key is provisioned and the updater is live.** This said `UPDATE_KEY` was
all zeroes for a while after it stopped being true, and the cost was real: it
sent a session looking for why the update channel could not publish, on the day
it published. `UPDATE_KEY` in `src/update/mod.rs` holds a real P-256 point, the
release workflow signs and uploads, and `update check`, `update fetch` and
`update stage` are separate verbs because claiming a write range on the boot
partition is the most dangerous thing this system does.

To rotate: `tools/sign.py --keygen --out FILE`, paste the public rows into
`src/update/mod.rs`, and rebuild -- adopting a signer is itself a kernel change,
which is the point. **Use `--out`**, because without it the private half goes
to stdout, which is exactly how the last one died. The first build carrying a
new key cannot be delivered by this system, since no kernel in the field trusts
it yet; that one ships as an ISO.

The rollback copy is taken and read back before anything is overwritten, the
health flag is cleared before the window rather than after, the written image
is verified by digest, and a mismatch puts the old image straight back. The
decision function is pure, so all eight of its states are asserted at boot
without staging anything.

### Building an ISO

```bash
python tools/mkiso.py glados.iso \
    --efi target/x86_64-unknown-uefi/release/glados.efi \
    --payload esp/GLADOS
```

`mkiso.py` writes a FAT32 EFI System Partition and wraps it in ISO 9660 with an
El Torito EFI boot entry. Both formats are built from scratch, since xorriso
and oscdimg are absent from most Windows machines and neither format is large
enough to justify the dependency.

It generates VFAT long-name entries, which is not optional: the kernel opens
`\GLADOS\tokenizer.bin`, whose base name is nine characters and cannot be
expressed as 8.3. A short-name-only image presents it as `TOKENI~1.BIN`, and
the kernel then fails to find its tokenizer at boot, on real hardware, with no
filesystem left to debug from.

Test the image the same way anything else here is tested:

```bash
python tools/drive.py --iso glados.iso "ls /"
```

---

## Testing

**There is no `cargo test`.** This is a `no_std` UEFI binary with no host test
runner, so verification is the boot selftests plus driving QEMU.

At boot the system runs twenty-six selftest sections, printing `ok` or `FAIL`
per line: heap, timer, clock, the namespace's Merkle addressing, fifteen sets
of published cipher vectors, fault handling, running machine code from the
heap, file type detection, the glyph table and the UTF-8 decoder, constrained
decoding, the agent loop, the linear probe, the situation planner, the
initiative policy, the self-modification gate, corpus bundles, QDoRA adapters,
the backward kernels, and the trainer's arithmetic.

**Forty-two** suites can be re-run on demand with `diag all` or `diag <name>`.
Registration is deliberately awkward: `SLOTS` is one number, and a suite added
without a slot in the results table fails a compile-time assertion rather than
silently never recording a verdict. That guard has earned its place -- it once
compared `SUITES.len()` against a *literal* while the array's length was a
separate literal beside it, so the thirty-third suite passed the guard and then
panicked at the store, which is precisely the failure the guard's own comment
says it prevents.

Both of those numbers were wrong here for several releases, and in different
directions from the ones in the project's own notes. They are counted now
rather than remembered:

```bash
python tools/drive.py "initiative off" "agent stop" "diag"   # then count
grep -c 'const SLOTS' src/diag.rs                            # 42, asserted
```

**A bare `diag` is not a clean sweep.** It prints the table with `-` beside
everything that has not run and a tally reading `0 passed, 0 failed, 42 not
run`, which is easy to read as the opposite of what it is. `diag all` runs
them.

**That output is the test suite.** It is easy to scroll past and it does catch
real bugs. An ECDSA break sat visible in `[selftest] crypto` for an entire
debugging cycle while the log was being sliced down to look at something else,
and a failing sparsity claim shipped for one commit the same way.

`diag differ` is the newest instrument and the one with the most opinion in it.
It runs one program two ways and requires agreement on value, step count and
error text, bit for bit, sixty-four times over. It also runs a pair that is
*supposed* to disagree and fails if that is not caught, because a harness which
has never reported a difference is indistinguishable from one that compares
nothing.

`tools/reference.py` is a NumPy oracle for the dense models and `tools/v4.py`
for the hybrids. Both read the *converted* file, so a `convert.py` bug shows up
there too and only a Rust bug shows up as a mismatch.

---

## Design notes

A few decisions that are load-bearing and non-obvious.

Constrained decoding makes invalid output unreachable. The grammar is built
from the live applet table, and read-only mode works by removing mutating
applets from the reachable set *before* sampling, so no sequence of sampling
outcomes names a tool that does not exist. Permission enforcement and output
validity end up being the same piece of code, which leaves nowhere for the two
to disagree.

The closed-form router beats the transformer. Two paths exist: decoding an
applet name token-by-token under a grammar, and reading one hidden state into a
closed-form ridge regression (Widrow-Hoff, 1960) solved by Cholesky in-kernel.
The regression is 12,672 parameters, runs in about 1.6 ms with no transformer
forward pass, and scores 54.7% on the held-out split against 32.1% for nearest
neighbour and 5.7% for the 135M model asked the same question.

Agreement carries more information than the vote. Three cores classify
independently. Combining them into a single better answer was measured and does
not work, scoring 76.9% against 77.8% for the best core alone. What their
agreement predicts is correctness: 90.3% right where all three pick the same
applet, 50% where they split. That gap is what the gate acts on, and a router
that knows when it is guessing is worth more than the point the ensemble was
supposed to buy.

Freezing the base is what makes training affordable. A frozen weight means a
hidden state is a constant, a constant can be cached, and a cached decision can
be replayed against any number of candidate adapters for the price of a dot
product. Every capability above rests on that one property, including the
judging: it is why a verdict this machine reaches at three in the morning can
be re-checked by anybody, later, for almost nothing. Anything that proposes to
train the attention path is proposing to give this up, which is a real trade
and worth naming before it is made.

The content hash covers content only, never block locations. Otherwise moving a
block would rename an object.

NVMe writes are locked by default, and the unlock names a range. It was one bit
for a long time: unlocking said writes were allowed and nothing said where, so
from the moment initialisation succeeded every LBA on the device was writable,
including the partition table and the Windows volume that is still the only
other thing on this disk. The window is enforced on every write and printed
beside the word UNLOCKED, and `diag wgate` asserts the whole decision without a
device and without writing anything.

A model can be wrong without being broken. RoPE pairing, QK-Norm, head width,
RMSNorm epsilon and the pre-tokenizer regex each produce a network that loads,
runs, stays numerically well-behaved, and writes fluent text. There is no error
to catch. The kernel used the wrong RoPE convention for a long time and nothing
looked broken, because both conventions are norm-preserving rotations by the
same angles; the model stayed fluent and attended by a scrambled notion of
distance, which is indistinguishable from the outside from a small model being
small.

Negative results stay in the tree. Training the **SGD** head *hurts* at this
data scale -- 30% held-out untrained, 10% after two epochs, 0% after eight,
because forty examples across twenty-one classes leave gradient descent nothing
to do but memorise. That is the head the ridge probe replaced, and it is **not**
the QDoRA adapter; the two have nothing to do with each other. This sentence
said "the adapter head" for a long time and it cost a session, which is the
whole argument for naming the thing precisely in the summary and not only in
the file it belongs to. The Product-of-Experts council does not improve
accuracy either. A
renderer survey ranked volatile framebuffer writes as the single largest
constant-factor loss and the measurement disagreed; the real win was that blank
console cells were being painted one pixel at a time under a background that
had just been filled, which took a frame from 2,376 us to 1,629 us. All of them
are kept, because the reason to know them is the reason they were worth
measuring, and because a deleted experiment gets repeated.

The measurements here are small, and they say so where they appear. The adapter
trainer has been exercised on subsamples of a few dozen decisions, which
establishes that the machinery composes and establishes nothing about how much
it helps.

For a long time a full-corpus pass was believed to need the laptop, because a
forward-pass group took 286 s under emulation. It took 1.8 s the first time
anybody tried the hypervisor accelerator instead of the interpreter, and the
whole corpus is about twenty minutes. The belief was never measured, and it
shaped which questions seemed answerable.

---

## Copyright

**Copyright © 2026. All rights reserved.**

No licence is granted. This source is published to be read and not to be
reused. You may not copy, modify, redistribute or create derivative works from
it without written permission.

### One exception, and it is not ours to reserve

`src/dev/rtl8188eu_tables.rs` contains 509 hardware initialisation constants
transcribed from `drivers/net/wireless/realtek/rtl8xxxu/8188e.c` in the Linux
kernel, which is **GPL-2.0**. Those values are not this project's work and the
reservation above does not apply to them. They are isolated in that one file,
marked at the top, and nothing else in the tree is copied from anywhere.

If you intend to reuse anything here, that file's licence is Linux's.

### Model weights

Model weights are not distributed in this repository. The ISO includes a
converted Qwen3-0.6B, which is **Apache-2.0** and belongs to Alibaba Cloud. Its
licence travels with it.

### Trademarks

GLaDOS, Aperture Science and Portal are properties of Valve Corporation. This
project is independent and is not affiliated with, endorsed by, or connected to
Valve in any way.

That line used to say "an independent, non-commercial homage". The project has
a token attached to it, which makes the non-commercial clause untrue, and a
disclaimer carrying a false clause is worse than a shorter one. What was doing
the work is the rest of the sentence.

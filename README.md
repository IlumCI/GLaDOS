# GLaDOS

A from-scratch, non-Unix, ring-0 operating system in Rust, built around a
language model that runs inside the kernel.

Everything sits at the same privilege level, in one address space, with no
syscalls and no userspace to put anything in. The consequence that matters is
what a tool call becomes: the sampler picks a name from the live applet table
under a grammar compiled from that table, and the kernel calls the function.
Nothing is serialised and nothing is parsed in between.

It boots as a UEFI application, which means it *is* the kernel. Firmware
already delivers long mode, CPL 0 and an identity map, so there is no
bootloader, no ELF loading, no relocation and no handoff ABI. The model,
tokenizer and root certificates are read before `ExitBootServices`, because
that call is the last moment a filesystem exists.

108 files of Rust, about 50,000 lines. TempleOS is the obvious ancestor, and
the identity map here exists for the reason Terry Davis gave for his.

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
| Graphics | GOP framebuffer, Windows-3.1-styled desktop, window manager, taskbar |
| Storage | NVMe, content-addressed object store, Merkle trees, snapshots |
| Network | ARP, IPv4, ICMP, UDP, TCP, DHCP, DNS, TLS 1.3 with chain validation |
| Drivers | e1000, RTL8168, xHCI (USB 3), CDC-ECM USB Ethernet |
| Crypto | SHA-1/256/384, HMAC/HKDF, AES, ChaCha20-Poly1305, X25519, RSA, ECDSA |
| Model | Qwen3, Qwen3.5 hybrids and SmolLM2, int8, in-kernel inference |
| Routing | Constrained decoding over the live applet table, plus a closed-form probe |
| Agent | Propose, validate, execute, observe, with the grammar as the permission system |
| Initiative | A resident task that decides for itself when to act and when to stay quiet |
| Training | Gradients, Adam and a QDoRA adapter over the classifier, all in-kernel |
| Self-modification | Variant lineage, a judge council, an append-only ledger, O(1) rollback |
| Language | Lexer, parser, tree-walking interpreter with kernel builtins |
| Mining | Midstate-cached SHA-256d with an honest scoreboard |

Does not work yet:

- Wireless. The built-in card is CNVi, so the MAC lives in the PCH and the M.2
  module is only a radio, reachable through an undocumented signed-firmware
  protocol. Our WPA2 supplicant is complete and checked against IEEE 802.11i
  vectors at every boot, and has never had hardware to run on. A USB dongle
  driver (RTL8188EU) has its register layer and power-on sequence, and stops
  short of PHY, radio and firmware upload.
- SMP. Single core. `sync::Racy<T>` gives interior mutability with no locking
  of any kind, and carries that name so one grep finds every place that assumed
  a single core on the day one stops being enough.
- A hardware entropy source. The generator is a fast-key-erasure ChaCha20
  DRBG fed by keyboard and mouse interrupt timing and by NVMe completion
  latency, and it refuses to answer for key material until it has seen enough
  events. That refusal is correct and it is also a limitation: a machine that
  boots, touches no disk and shuts down with no key pressed never seeds, and
  TLS then falls back to timing-derived keys and says so during the
  handshake. The CPU has RDRAND and we do not use it, because trusting an
  opaque instruction is a different argument from trusting interrupt timing.
- Attention-path training. The adapter moves the classifier. Every activation
  adjoint needed to go deeper exists and is checked at boot, and nothing yet
  composes them into a backward pass through the layers.

---

## The machine that changes itself

The three capabilities above that are newest deserve a paragraph each, because
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

**A trained adapter is an object in the namespace.** `adapter save` writes a
`GLADOSA1` blob holding the adapter alone, never the checkpoint. Rows are
stored sparsely, because they are sparse in fact: on the measured decision
layer that is 23.7 KB against 1.79 MB dense, a factor of 75. Save, detach,
reload and save again reproduces the file byte for byte, which we check by
comparing content addresses. `tools/adapter.py` reads the format on the host.

**Self-modification runs on certificates.** Schmidhuber's Gödel machine adopts
a rewrite once it can prove the rewrite raises expected utility, and it carries
a theorem prover to do it. We have no theorem prover and could not build one
for a quantised transformer's future reward. What we substitute is a
certificate that is cheaper to refute than it was to produce, over
content-addressed inputs, so any later run re-derives the same verdict bit for
bit. That is affordable here for one specific reason: producing a variant costs
a forward pass per example, and checking somebody else's claim about one costs
a dot product per cached decision. Four orders of magnitude between making a
claim and testing it is the asymmetry a proof system provides, reached by
another road.

Four judges have to agree, each covering a different way a variant can be bad:

- **J1, margin.** Both variants answer the same cached decisions, so the
  comparison is paired and McNemar's test applies. Nine repairs against two
  breaks scores 3.27 and fails the bar; twelve against one scores 7.69 and
  passes. Comparing two percentages would call both of those a win.
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

Two details we consider load-bearing. The test slice carries a budget that
lives in the ledger, because a loop that improves itself forever reads the
held-out set forever and each read makes the reported figure more optimistic;
past the budget a test number is printed as stale and marked unquotable. And
before the judges run, the machine records whether training-set gain predicted
a win, so the ledger accumulates an answer to a question we actually want
settled. Nothing acts on that prediction, and at the current sample size it
means nothing, which the ledger says out loud.

Trials run when two independent facts agree: the real-time clock says the hour
falls inside a quiet window, and the entropy ring says no key or pointer
interrupt has fired. A clock alone does not know somebody is working late, and
silence at noon is a coffee break.

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

**The boot disk on our development machine is counterfeit**: it advertises
976 GB and holds 14.67. That is why the layout tooling uses MBR, a GPT backup
header having no real flash to land in, and why it carries a `SafeLimitGB`.

---

## Getting a model

The model is not in this repository. It is around 570 MB, it is not our work,
and a git repository is the wrong place for it. The ISO ships with one;
building from source means supplying your own.

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
could never be run and contradicted, so we decline to write one.

SmolLM2-135M also works, and it is the checkpoint to reach for under QEMU,
which cannot host the larger ones (see below).

---

## Building

Rust nightly targeting `x86_64-unknown-uefi`:

```bash
cargo build --release
```

The artifact is `target/x86_64-unknown-uefi/release/glados.efi`.

### Running under QEMU

```bash
python tools/drive.py "tensor" "model" "ask -n 20 hello"
```

`drive.py` boots QEMU, stages the binary, resets NVRAM to pristine, and drives
the shell over a serial socket. It assembles its own ESP from the small
checkpoint, so it never disturbs deploy staging.

Two things to know before a session goes sideways. Pass `initiative off` as the
first command: the resident task wakes fifteen seconds in and holds the engine
for a whole episode, which presents as a timeout with commands unsent. And pass
`--qemu-extra "-cpu max"` for anything that trains, since the default `qemu64`
model hides every SIMD extension and the trainer declines without AVX2.

QEMU cannot run the 0.6B model. Its VVFAT is FAT16 on a fixed geometry and the
whole disk is 516 MB. `fat:32:` raises that in principle, but QEMU says its
FAT32 is untested and the firmware cannot read the directory it produces. So
Qwen3 runs only on real hardware, and QEMU work uses SmolLM2. The same cap
applies to the hybrids, so `tools/hybtest.py` builds a small one shaped to hit
every path the real one does and prints what the logits should say.

### Deploying to hardware

```powershell
.\scripts\deploy.ps1 -EspDrive S: -Release
```

Then reboot and hold **F11** for the boot menu.

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

At boot the system runs eighteen selftest sections carrying seventy-one
individual claims, printing `ok` or `FAIL` per line: heap, timer, clock, the
namespace's Merkle addressing, fifteen sets of published cipher vectors, fault
handling, constrained decoding, the agent loop, the linear probe, the situation
planner, the initiative policy, the self-modification gate, corpus bundles,
QDoRA adapters, the backward kernels, and the trainer's arithmetic.

**That output is the test suite.** It is easy to scroll past and it does catch
real bugs. An ECDSA break sat visible in `[selftest] crypto` for an entire
debugging cycle while the log was being sliced down to look at something else,
and while writing the adapter format we shipped a failing sparsity claim for
one commit by doing exactly the same thing.

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
be re-checked by anybody, later, for almost nothing.

The content hash covers content only, never block locations. Otherwise moving a
block would rename an object.

NVMe writes are locked by default. They unlock only after a target region is
named, the formatter re-checks before touching anything, and every error path
re-locks. On a disk fully allocated to Windows there is no such region and init
fails, which is the intended outcome. Leaving the lock open on a failure path
is how a safety mechanism becomes decorative.

A model can be wrong without being broken. RoPE pairing, QK-Norm, head width,
RMSNorm epsilon and the pre-tokenizer regex each produce a network that loads,
runs, stays numerically well-behaved, and writes fluent text. There is no error
to catch. The kernel used the wrong RoPE convention for a long time and nothing
looked broken, because both conventions are norm-preserving rotations by the
same angles; the model stayed fluent and attended by a scrambled notion of
distance, which is indistinguishable from the outside from a small model being
small.

Negative results stay in the tree. Training the adapter head *hurts* at this
data scale, and the Product-of-Experts council does not improve accuracy. Both
are kept, because the reason to know them is the reason they were worth
measuring, and because a deleted experiment gets repeated.

The measurements we have are small, and we say so where they appear. The
adapter trainer has been exercised on subsamples of a few dozen decisions
because a full-corpus pass is a forward pass per example, which is seconds on
the laptop and most of a day under emulation. Those runs establish that the
machinery composes. They establish nothing about how much it helps, and no
figure from them belongs in a claim.

---

## Copyright

**Copyright © 2026. All rights reserved.**

No licence is granted. This source is published to be read and not to be
reused. You may not copy, modify, redistribute or create derivative works from
it without written permission.

### One exception, and it is not ours to reserve

`src/dev/rtl8188eu_tables.rs` contains 509 hardware initialisation constants
transcribed from `drivers/net/wireless/realtek/rtl8xxxu/8188e.c` in the Linux
kernel, which is **GPL-2.0**. Those values are not our work and the reservation
above does not apply to them. They are isolated in that one file, marked at the
top, and nothing else in the tree is copied from anywhere.

If you intend to reuse anything here, that file's licence is Linux's.

### Model weights

Model weights are not distributed in this repository. The ISO includes a
converted Qwen3-0.6B, which is **Apache-2.0** and belongs to Alibaba Cloud. Its
licence travels with it.

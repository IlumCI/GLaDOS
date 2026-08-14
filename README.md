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

89 files of Rust, about 36,000 lines. TempleOS is the obvious ancestor, and the
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
| Tasks | Cooperative + preemptive at 100 Hz, `sysv64` context switch |
| Graphics | GOP framebuffer, Windows-3.1-styled desktop, window manager, taskbar |
| Storage | NVMe, content-addressed object store, Merkle trees, snapshots |
| Network | ARP, IPv4, ICMP, UDP, TCP, DHCP, DNS, TLS 1.3 with chain validation |
| Drivers | e1000, RTL8168, xHCI (USB 3), CDC-ECM USB Ethernet |
| Crypto | SHA-1/256/384, HMAC/HKDF, AES, ChaCha20-Poly1305, X25519, RSA, ECDSA |
| Model | Qwen3-0.6B or SmolLM2-135M, int8, in-kernel inference |
| Language | Lexer, parser, tree-walking interpreter with kernel builtins |

Does not work yet:

- Wireless. The built-in card is CNVi, so the MAC lives in the PCH and the M.2
  module is only a radio, reachable through an undocumented signed-firmware
  protocol. The WPA2 supplicant is complete and checked against IEEE 802.11i
  vectors at every boot, and has never had hardware to run on. A USB dongle
  driver (RTL8188EU) has its register layer and power-on sequence, and stops
  short of PHY, radio and firmware upload.
- SMP. Single core. `sync::Racy<T>` gives interior mutability with no locking
  of any kind, and carries that name so one grep finds every place that assumed
  a single core on the day one stops being enough.
- The agent loop. The model proposes and a keystroke adopts. Given the name
  over the door, this is the one feature the project is in no hurry to add.

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

`--seq` sets the context window, and what bounds it is KV cache size and not
the model. Qwen3-0.6B costs 112 MiB of kernel heap at 512 tokens, and
`convert.py` prints that figure so the decision gets made where `--seq` is
chosen, instead of surfacing as an allocation failure at boot.

**Always pass `--verify` to `tokenizer.py`.** It reimplements the kernel's
algorithm and diffs it against the reference `tokenizers` library, token for
token. A tokenizer that is subtly wrong produces text that still looks like
text, so nothing downstream will catch it for you.

SmolLM2-135M also works, and it is the checkpoint to reach for under QEMU,
which cannot host the larger one (see below).

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

QEMU cannot run the 0.6B model. Its VVFAT is FAT16 on a fixed geometry and the
whole disk is 516 MB. `fat:32:` raises that in principle, but QEMU says its
FAT32 is untested and the firmware cannot read the directory it produces. So
Qwen3 runs only on real hardware, and QEMU work uses SmolLM2.

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

At boot the system runs heap, timer, clock, namespace, crypto (11 vector sets
from published RFCs and FIPS documents), constrained-decoding and probe
selftests, printing `ok` or `FAIL` per line. **That output is the test suite.**
It is easy to scroll past and it does catch real bugs: an ECDSA break sat
visible in `[selftest] crypto` for an entire debugging cycle while the log was
being sliced down to look at something else.

`tools/reference.py` is a NumPy oracle for the model. It reads the *converted*
file, so a `convert.py` bug shows up there too and only a Rust bug shows up as
a mismatch.

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

---

## Copyright

**Copyright © 2026. All rights reserved.**

No licence is granted. This source is published to be read, not to be reused.
You may not copy, modify, redistribute or create derivative works from it
without written permission.

### One exception, and it is not mine to reserve

`src/dev/rtl8188eu_tables.rs` contains 509 hardware initialisation constants
transcribed from `drivers/net/wireless/realtek/rtl8xxxu/8188e.c` in the Linux
kernel, which is **GPL-2.0**. Those values are not this project's work and the
reservation above does not apply to them. They are isolated in that one file,
marked at the top, and nothing else in the tree is copied from anywhere.

If you intend to reuse anything here, that file's licence is Linux's, not mine.

### Model weights

Model weights are not distributed in this repository. The ISO includes a
converted Qwen3-0.6B, which is **Apache-2.0** and belongs to Alibaba Cloud, not
to this project. Its licence travels with it.

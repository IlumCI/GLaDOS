# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GLaDOS: a from-scratch, non-Unix, ring-0 operating system in Rust for one
specific laptop (MSI Thin GF63 12UC, board MS-16R8), built around a language
model that lives *inside* the kernel rather than on top of it. No user/kernel
split, no syscalls, no process isolation, one address space. A tool call from
the model is a function call.

The only code in the kernel this project did not write is Rust `core` — and
`src/dev/rtl8188eu_tables.rs`, which is 509 hardware initialisation constants
transcribed from Linux's GPL-2.0 rtl8xxxu driver because there is no other
source for them. It is one file, marked as such at the top, and nothing else in
the tree is copied from anywhere.

## Commands

Build and run under QEMU:

```powershell
.\scripts\run.ps1                 # debug build, boot in QEMU
.\scripts\run.ps1 -Release
.\scripts\run.ps1 -Gdb            # pause for gdb on :1234
.\scripts\run.ps1 -TraceFaults    # log every exception; finds triple faults
```

Cargo directly (rustup lives under scoop, not on PATH by default):

```powershell
$env:PATH = "$env:USERPROFILE\scoop\persist\rustup-msvc\.cargo\bin;$env:PATH"
cargo build            # or --release
```

Deploy to the USB SSD, then reboot and hold **F11**:

```powershell
.\scripts\deploy.ps1 -EspDrive S: -Release
```

`deploy.ps1` builds first and copies both `BOOTX64.EFI` *and* `esp\GLADOS\`
(model, tokenizer, roots). Without `roots.der` TLS encrypts but authenticates
nothing.

### Python tooling

Use the project venv — there is no Python on PATH:

```powershell
.\tools\venv\Scripts\python.exe tools\traces.py out\traces.jsonl --count 40000 --per-family 300
.\tools\venv\Scripts\python.exe tools\dataset.py out\corpus.json --rust src\ai\corpus.rs
.\tools\venv\Scripts\python.exe tools\convert.py tools\qwen3 esp\GLADOS\model.bin --seq 512
.\tools\venv\Scripts\python.exe tools\tokenizer.py tools\qwen3\tokenizer.json esp\GLADOS\tokenizer.bin --verify
```

`tools/qwen3/` and `tools/hf/` hold safetensors checkpoints. `convert.py <src>
<dst> [--f32] [--seq N]` flattens one into the `GLADOSM3` layout
`ai::model::offsets` indexes by arithmetic. `--seq` sets the context window and
is bounded by **KV cache size, not by the model**: Qwen3-0.6B costs 112 MiB of
kernel heap at 512, and convert.py prints that figure so it is decided where
`--seq` is chosen rather than discovered as an allocation failure at boot.

Always run `tokenizer.py` with `--verify`. It reimplements the kernel's
algorithm and diffs it against the reference `tokenizers` library; a tokenizer
that is subtly wrong produces text that still looks like text.

**Qwen3.5 writes v4, not v3.** `convert.py` dispatches on `model_type`:
`llama`/`qwen2`/`qwen3` take the dense path and still produce a byte-identical
v3 file, while `qwen3_5`/`qwen3_5_moe` take `convert_hybrid` and produce a
160-byte header plus a **layer-major** body — three layers in four hold
`linear_attn.*` and the fourth holds `self_attn.*`, so there is no single
stride to multiply and grouping by tensor stops being possible. The layer
schedule travels as an explicit bitmap rather than being derived from
`full_attention_interval`, so a checkpoint that breaks the pattern fails
instead of loading and running wrong.

The kernel runs the hybrid, `Arch::Qwen35`. **MoE is refused at load**
(`LoadError::Unsupported`) rather than half-implemented: the smallest published
one is 71.9 GB, nothing that size reaches a UEFI pool on the GF63, so a forward
pass for it could never be run and contradicted.

**QEMU cannot run Qwen3.5-0.8B either** -- 723 MB against VVFAT's 516 -- so the
kernel port is checked against a *small* hybrid instead of deferring every bug
to hardware. `tools/hybtest.py` builds one shaped to hit every path the real
one does (both layer kinds, packed cache indices, partial RoPE, 4 value heads
over 2 key heads, GQA, an untied classifier) and prints what `logits` should
say:

```powershell
.\tools\venv\Scripts\python.exe tools\hybtest.py out\hybtest.bin --build
.\tools\venv\Scripts\python.exe tools\drive.py --model out\hybtest.bin "logits 7 11 3"
```

`--schedule FFFF` / `LLLL` isolates the two mixers, and `--zero` zeroes every
projection back into the residual stream so the loader, the final norm and the
classifier can be checked apart from any layer. `drive.py --model` overrides
the staged checkpoint.

`tools/v4.py` reads a v4 file back, and is the oracle's front end the way
`reference.py` is for v2/v3:

```powershell
.\tools\venv\Scripts\python.exe tools\v4.py --selftest      # writer/reader round-trip
.\tools\venv\Scripts\python.exe tools\ref35.py --converted out\q35-0.8b.bin
```

The v4 body has no names, shapes or lengths in it, so a writer/reader
disagreement about one dimension leaves everything after it as perfectly valid
float32 garbage. Both readers therefore **walk and never seek**, and assert
they land on the last byte. `convert_hybrid` makes the same bargain on input:
every tensor must be written or explicitly skipped, and anything else is an
error rather than a silent omission.

Root certificate bundle, built from the host's store:

```powershell
.\scripts\fetch-roots.ps1          # -List to see what would be exported
```

### Testing

There is no `cargo test` — this is a `no_std` UEFI binary with no host test
runner. **Verification is the boot selftests plus driving QEMU.** At boot the
system runs heap, timer, clock, namespace, crypto (11 RFC vector sets),
constrained-decoding and probe selftests, and prints `ok` / `FAIL` per line.
Read that output; it is the test suite.

Shell commands that re-run tests on demand: `tensor`, `model`, `crypto`,
`trust verify`, `fit`, `gate`, `search`, `wpa2`, `video bars`. `tensor` and
`model` are **not** part of the boot sequence and hold the checks for the
pre-tokenizer and the wide-head attention geometry.

`tools/drive.py` boots QEMU and drives the shell over a serial socket:

```powershell
.\tools\venv\Scripts\python.exe tools\drive.py "tensor" "model" "ask -n 20 hello"
```

It stages `BOOTX64.EFI`, resets NVRAM to pristine (a stale boot entry sends the
firmware to the UEFI shell, which looks like the system not booting), and
attaches serial as TCP — QEMU's Windows stdio chardev reads console handles,
not redirected files, so piping a script into it silently does nothing.

**QEMU cannot run the real model.** VVFAT is FAT16 on a fixed geometry and the
whole disk is 516 MB; `fat:32:` raises that in principle but QEMU says its
FAT32 is untested and the firmware cannot read the directory it produces. So
Qwen3-0.6B is only runnable on the GF63, and QEMU work uses the SmolLM2
checkpoint in `out/`. Guest RAM must also cover the weights, which are read
whole into a pool before `ExitBootServices` — `run.ps1 -Memory` defaults to 2G.

`tools/reference.py` is the numeric oracle and the way to check the real model
without hardware. It reads the *converted* file, so a `convert.py` bug shows up
there too and only a Rust bug shows up as a mismatch:

```powershell
.\tools\venv\Scripts\python.exe tools\reference.py out\qwen3-0.6b.bin --tokenizer tools\qwen3\tokenizer.json --generate 40 --prompt "..."
```

Compare `logits <ids>` in GLaDOS against the same ids here. Coherent generated
text is the cheap end of the same check: an 0.6B instruction-tuned model whose
attention path is wired correctly writes real sentences.

**Boot selftest output is easy to skip past and it does catch real bugs.** An
ECDSA break was visible in `[selftest] crypto` for a whole debugging cycle
while the output was being sliced away.

## Architecture

### Boot

UEFI already delivers long mode, CPL 0 and an identity map, so this UEFI
application *is* the kernel — no ELF loading, relocation or handoff ABI.
`main.rs` reads the model, tokenizer and root bundle **before**
`ExitBootServices`, because that is the only moment a filesystem exists.
Everything after runs on our own page tables.

`gfx::splash` owns the framebuffer during boot; the console writes to its RAM
shadow grid without painting, and `finish()` repaints the whole log. Anything
that draws during boot must check `splash::active()`, and the fault reporter
and panic handler call `splash::abandon()` first — on the GF63 the framebuffer
is the only diagnostic channel there is.

### Concurrency

`sync::Racy<T>` is **not a lock** — it is single-core interior mutability and
the designated grep target for the day SMP arrives.

`task::yield_now` disables interrupts across the context switch. This is not
optional: `schedule()` stores `CURRENT` and *then* switches stacks, so a timer
tick landing between them saves the outgoing stack pointer into the wrong slot
and one task becomes unresumable. The interrupt path is safe because a gate
clears IF for it.

### Networking (`src/net/`)

Interfaces live in `iface`: `lo`, `eth0`, `wlan0`. A driver implements
`iface::Nic`; `net::init` tries e1000 (QEMU) then rtl8168 (the GF63's real
card, `10ec:8168`). Routing picks an interface by destination, and every layer
above asks for a source address rather than assuming one exists.

**`poll` never dispatches into a transport state machine — it queues.**
Sending calls `send_ipv4` → `resolve`, and `resolve` calls `poll` while waiting
for ARP. Running a state machine from there would let a connection re-enter its
own control block while an earlier borrow is live. TCP and UDP drain their own
inboxes.

TCP advances only while the shell is idle (`tcp::service` from the idle loop)
or inside a blocking call. There is no interrupt-driven receive.

The wireless card is CNVi — the MAC is in the PCH and the M.2 module is a
radio. `net/wifi.rs` identifies hardware and refuses to pretend; the WPA2
supplicant in `net/wpa2.rs` is complete and verified against IEEE vectors but
has nothing to run on.

### Crypto (`src/crypto/`)

Written from scratch, and this is the one place where that is a liability
rather than a virtue: a bug here produces output that works perfectly and is
not secure. Primitives were chosen for checkability — ChaCha20 over AES-GCM
(no key-dependent table lookups), X25519 over a NIST curve. Every one is
checked against published RFC vectors at boot.

ECDSA uses **Jacobian** coordinates. Affine cost an inversion per point
operation — a full modexp — which for P-384 meant ~460,000 allocating
multiplies per signature and exhausted the heap. `Mont::inv_prime` takes and
returns *ordinary* values; passing it something already in Montgomery form
computes the wrong thing silently. Use `Curve::inv_m`.

TLS 1.3 validates the chain, the transcript signature, dates and name — and
**reports** rather than enforces. A caller that cares must check
`identity.ok()`. No revocation. Key material still comes from the TSC, not
`RDRAND`.

### The model (`src/ai/`)

Qwen3-0.6B, int8, ~570 MB on the ESP, referenced in place in the LoaderData
pool rather than copied to the heap. SmolLM2-135M still loads and is the small
checkpoint to reach for when something needs to run under QEMU.

**Qwen3 is not a Llama, and neither difference fails loudly.** Its head width
is *stated* (128) rather than derived (1024/16 = 64), so `wq` is `[2048, 1024]`
and the attention path is wider than the residual stream; and it RMSNorms each
head's query and key before RoPE. Ignore either and the model loads, runs, and
generates confident nonsense. `Config::head_dim` and `Config::qk_norm` carry
them, and the `GLADOSM3` header (v3) records them per checkpoint. v2 files still
load: their defaults are exactly the Llama ones.

**RoPE pairs `i` with `i + head_dim/2`, not `2i` with `2i+1`.** This is
`rotate_half` in HuggingFace's modeling code, and therefore what every
checkpoint trained through transformers expects -- Qwen3 and SmolLM2 alike.
The kernel used the interleaved convention for a long time and nothing looked
broken, because both are norm-preserving rotations by the same angles: no NaN,
no drift, no error. The model stays fluent and attends by a scrambled notion of
distance, which is indistinguishable from a small model being small. It cost
SmolLM2 `"The capital of France."` followed by blank lines where the corrected
path gives `"The capital of France is Paris. Paris is a city known for..."`.
`Config::rope_interleaved` is true only for genuine llama2.c checkpoints.

Generation is memory-bandwidth bound -- bytes read per token is roughly the
model size -- so 570 MB against 135 MB is about 4.4x the time per token. The
classifier is 155 MB of that, and constrained decoding only ever needs logits
for the reachable set, so restricting that matvec is the obvious win when it
matters.

`ask` closes the `<think>` block itself unless given `-t`. Qwen3 left alone
reasons at length, which is the model working as designed and useless at a
64-token budget. `has_think_token()` decides by asking whether the tokenizer
knows `<think>` as one token -- a property of the vocabulary rather than a
guess from a name.

The tokenizer carries which pre-tokenizer regex the checkpoint trained with.
SmolLM2 is the GPT-2 pattern; Qwen3 spells out the cl100k one, where a word may
be led by any non-alphanumeric (`(x` is one piece), digits come one at a time,
and punctuation swallows following newlines. Using the wrong one moved ~12% of
tokens on the training corpus -- again with no error, just a model fed
sequences it never saw.

Two routing paths, and the interesting result is that the older one wins:
`act` decodes an applet name token-by-token under a grammar; `route` reads one
hidden state and hands it to a closed-form ridge regression (Widrow-Hoff, 1960)
solved by Cholesky in-kernel — 12,672 parameters, ~1.6 ms, **no transformer
forward pass**, and better held-out accuracy.

**Constrained decoding makes invalid output unreachable, not improbable.** The
grammar is built from the live applet table; read-only mode works by removing
mutating applets from the reachable set *before* sampling, not by checking
after.

Three "cores" vote (probe, hashed-n-gram Bayes, lexical). Their *agreement* is
the signal, not their vote: 90% right when all three agree against 61% when
they split. That gap is what `gate` acts on.

### Storage (`src/store/`, `src/sysbox/`)

Content-addressed: objects named by SHA-256 of their contents, assembled into
Merkle trees. A copy is O(1), a snapshot is one root hash. **The content hash
covers content only and never block locations** — otherwise moving a block
would rename an object.

NVMe writes are locked by default. `store::init` unlocks only after
`find_store_region` names a target, and `Store::format` re-checks. On a disk
fully allocated to Windows there is no such region and init fails — that is the
intended outcome. Every error path re-locks; leaving it open is how a safety
mechanism becomes decorative.

## Evaluation discipline

This project measures rather than argues, and the harness exists because the
measurement was got wrong three separate times: a grid sweep scored on the test
set, cross-validation folded by template family, and a test set that *moved*
whenever the corpus was appended to.

There are **three** splits (`SEED_TRAIN` / `SEED_VAL_END` / test). Validation
is spent freely; the test slice is read once. `search` adopts a configuration
only when measured better.

Corpora hold out **whole template families**, never sampled instances —
instances within a family differ only by slot values, so an instance split
measures memorisation while looking like generalisation.

Negative results stay in the tree. Training the adapter head *hurts* at this
data scale; the Product-of-Experts council does not improve accuracy. Both are
kept because the reason to know them is the reason they were worth measuring.

`tools/traces.py` reports what it could **not** produce and the per-family
imbalance unprompted — a generator asked for 20,000 that quietly returns 54
near-duplicates yields a corpus that trains a model to recite.

## Gotchas that have already cost time

- **`extern "C"` on `x86_64-unknown-uefi` is Microsoft x64, not System V.** The
  context switch is pinned to `extern "sysv64"` explicitly.
- **Do not take the max over every UEFI memory descriptor.** OVMF describes
  `Reserved` space to 1 TiB; using it as a map limit exceeds one PDPT and the
  identity map silently fails, falling back to firmware tables that map page 0
  — which made the null-dereference selftest pass without faulting.
- **A guarded match arm placed after the arms it guards is unreachable.** The
  compiler said so in a warning nobody read, for several commits.
- **Ethernet pads frames to 60 bytes**, so a bare 40-byte ACK carries garbage.
  IPv4 payloads must be trimmed to the length the header declares.
- **A feature gate must test the feature the code needs.** The AVX2 kernel was
  gated on `avx_enabled && fma` and never on `avx2`.
- **DER `expect(tag)` must not consume on mismatch**, and "try for the value,
  skip if that failed" throws it away when the optional field is absent.
- **A model can be wrong without being broken.** RoPE pairing, QK-Norm, head
  width, RMSNorm epsilon and the pre-tokenizer regex all produce a network that
  loads, runs, stays numerically well-behaved and writes fluent text. There is
  no error to catch, so the only thing that settles any of them is comparing
  against `tools/reference.py` or reading generated output that is supposed to
  contain a known fact.
- **The kernel heap is a ladder, not a constant** (`HEAP_LADDER`). It is one
  physically contiguous allocation, the GF63 cannot be tested from here, and a
  fixed size its memory map cannot satisfy is an unbootable system. Boot prints
  the size it got and says when it had to come down a rung.
- The boot disk is **counterfeit**: it advertises 976 GB and holds 14.67. Hence
  MBR (a GPT backup header would land in flash that does not exist) and hence
  `SafeLimitGB` in `build-layout.ps1`. Do not put anything you care about on it.

## Conventions

Comments explain *why*, and specifically why an obvious alternative was
rejected — several record measurements that overturned a confident assumption.
Match that register; do not add narration of what the code plainly does.

Commit messages follow the same shape: what changed, what it cost to find out,
and what is still not verified. State plainly when something is untested — the
RTL8168 driver cannot be exercised in QEMU (which emulates the 8139), and says
so in its own commit.

Git identity is not configured in this repo. Every commit needs it passed
explicitly, matching the existing author:

```bash
git -c user.name=glados -c user.email=research@euroswarms.eu commit ...
```

Note the enclosing `C:\` drive is itself a git repo — confirm the working
directory before staging.

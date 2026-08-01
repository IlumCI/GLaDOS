# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GLaDOS: a from-scratch, non-Unix, ring-0 operating system in Rust for one
specific laptop (MSI Thin GF63 12UC, board MS-16R8), built around a language
model that lives *inside* the kernel rather than on top of it. No user/kernel
split, no syscalls, no process isolation, one address space. A tool call from
the model is a function call.

The only code in the kernel this project did not write is Rust `core`.

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
.\tools\venv\Scripts\python.exe tools\convert.py tools\hf esp\GLADOS\model.bin
```

`tools/hf/` holds the safetensors checkpoint. `convert.py <src> <dst> [--f32]
[--seq N]` flattens it into the `GLADOSM2` layout `ai::model::offsets` indexes
by arithmetic. `--seq` sets the context window and is bounded by KV cache size,
not by the model.

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

Shell commands that re-run tests on demand: `crypto`, `trust verify`,
`fit`, `gate`, `search`, `wpa2`, `video bars`.

To drive QEMU non-interactively, attach to the serial port as a TCP socket —
QEMU's Windows stdio chardev reads console handles, not redirected files, so
piping a script into it silently does nothing.

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

SmolLM2-135M, int8, ~129 MB on the ESP, referenced in place in the LoaderData
pool rather than copied to the heap.

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

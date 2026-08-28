# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this is

GLaDOS: a from-scratch, non-Unix, ring-0 operating system in Rust for one
specific laptop (MSI Thin GF63 12UC, board MS-16R8), built around a language
model that lives *inside* the kernel. No user/kernel split, no syscalls, no
process isolation, one address space. A tool call from the model is a function
call.

108 files, roughly 50,000 lines. The only code in the kernel we did not write
is Rust `core`, and `src/dev/rtl8188eu_tables.rs`, which is the RTL8188EU
initialisation tables plus the RF and descriptor register constants, taken
from Linux's GPL-2.0 rtl8xxxu driver because there is no other source for
them. It is one file, marked as such at the top, and nothing else in the tree
is copied from anywhere. `tools/rtlconv.py` regenerates the second half from
a checkout and states its provenance; the tables came the same way.

## Commands

Build and run under QEMU:

```powershell
.\scripts\run.ps1                 # debug build, boot in QEMU
.\scripts\run.ps1 -Release
.\scripts\run.ps1 -Gdb            # pause for gdb on :1234
.\scripts\run.ps1 -TraceFaults    # log every exception; finds triple faults
```

Cargo directly (rustup lives under scoop, absent from PATH by default):

```powershell
$env:PATH = "$env:USERPROFILE\scoop\persist\rustup-msvc\.cargo\bin;$env:PATH"
cargo build            # or --release
```

**`drive.py` prefers the release artifact.** It stages
`target/x86_64-unknown-uefi/release/glados.efi` when one exists and falls back
to debug otherwise, so a `cargo build` alone leaves a stale release binary in
place and the change under test never boots. Build `--release` before driving.

Deploy to the USB SSD, then reboot and hold **F11**:

```powershell
.\scripts\deploy.ps1 -EspDrive S: -Release
```

`deploy.ps1` builds first and copies both `BOOTX64.EFI` *and* `esp\GLADOS\`
(model, tokenizer, roots). Without `roots.der` TLS encrypts but authenticates
nothing.

### Python tooling

Use the project venv, since there is no Python on PATH:

```powershell
.\tools\venv\Scripts\python.exe tools\traces.py out\traces.jsonl --count 40000 --per-family 300
.\tools\venv\Scripts\python.exe tools\dataset.py out\corpus.json --rust src\ai\corpus.rs
.\tools\venv\Scripts\python.exe tools\convert.py tools\qwen3 esp\GLADOS\model.bin --seq 512
.\tools\venv\Scripts\python.exe tools\tokenizer.py tools\qwen3\tokenizer.json esp\GLADOS\tokenizer.bin --verify
```

`tools/qwen3/` and `tools/hf/` hold safetensors checkpoints. `convert.py <src>
<dst> [--f32] [--seq N]` flattens one into the `GLADOSM3` layout
`ai::model::offsets` indexes by arithmetic. `--seq` sets the context window and
is bounded by **KV cache size instead of by the model**: Qwen3-0.6B costs
112 MiB of kernel heap at 512, and convert.py prints that figure so it is
decided where `--seq` is chosen, before it can surface as an allocation failure
at boot.

Always run `tokenizer.py` with `--verify`. It reimplements the kernel's
algorithm and diffs it against the reference `tokenizers` library; a tokenizer
that is subtly wrong produces text that still looks like text.

**Qwen3.5 writes v4.** `convert.py` dispatches on `model_type`:
`llama`/`qwen2`/`qwen3` take the dense path and still produce a byte-identical
v3 file, while `qwen3_5`/`qwen3_5_moe` take `convert_hybrid` and produce a
160-byte header plus a **layer-major** body. Three layers in four hold
`linear_attn.*` and the fourth holds `self_attn.*`, so there is no single
stride to multiply and grouping by tensor stops being possible. The layer
schedule travels as an explicit bitmap instead of being derived from
`full_attention_interval`, so a checkpoint that breaks the pattern fails
loudly.

The kernel runs the hybrid, `Arch::Qwen35`. **MoE is refused at load**
(`LoadError::Unsupported`) instead of being half-implemented: the smallest
published one is 71.9 GB, nothing that size reaches a UEFI pool on the GF63, so
a forward pass for it could never be run and contradicted.

**QEMU cannot run Qwen3.5-0.8B either**, 723 MB against VVFAT's 516, so the
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
error instead of a silent omission.

### The corpus, and getting one into a running machine

`/ai/train` holds the routing corpus as one blob per example, `applet<tab>task`.
It is seeded at boot from `src/ai/corpus.rs` (465 examples, compiled in so the
system can route before anything is mounted) and `teach` appends to it.

`dataset.py --blobs` writes the same examples a second way, as a `GLADOSC1`
bundle in the shape `/ai/train` actually stores, so a corpus can be replaced on
a running machine. The kernel side is `teach bundle`:

```powershell
.\tools\venv\Scripts\python.exe tools\dataset.py out\corpus.json --blobs out\corpus.bin
.\tools\venv\Scripts\python.exe tools\mkfat.py .qemu\nvme.img out\corpus.bin
.\tools\venv\Scripts\python.exe tools\drive.py "initiative off" "fat get /CORPUS.BIN /tmp/corpus.bin" "teach bundle /tmp/corpus.bin"
```

Under QEMU the ESP is VVFAT on a different device from the one `fat` scans, so
the bundle travels in the NVMe test image; on the GF63 the ESP is a partition
on the same disk and `fat get` reads it directly from `esp\GLADOS\`.

**`teach bundle` replaces the corpus, and it has to.** The bundle carries split
*positions* in its header, and the kernel takes its held-out boundaries from
those (`vocab::splits`) once one has been imported. Appending would leave the
boundaries describing a corpus that no longer exists, which is the same
"test set that moved" failure the three-way split exists to prevent, arriving
by a different route. `teach` on a live system still appends, and anything past
the recorded length trains.

### Training the model's own decision layer

Two different things in this tree are called training, and confusing them makes
every number ambiguous:

- **`fit` and `train [epochs]`** move the linear probe and its head. Closed-form
  ridge regression over hidden states; the checkpoint is never touched.
- **`train adapter`** moves a QDoRA adapter over the model's *classifier*. This
  is the model learning, in the only sense the word applies here.

```
train adapter [-e epochs] [-n examples] [-ms budget] [-r rank] [-lr rate]
adapter [status|save|load|off] [path]
```

**It refuses to run without AVX2/FMA** (`train::hardware_ok`). The scalar
kernels are correct and would produce the same adapter, slowly enough that
every hyperparameter judgement made from the run would be about the clock. So
under QEMU it needs `--qemu-extra "-cpu max"`; the default `qemu64` model hides
every SIMD extension and the command declines with the reason printed.

Three facts make it affordable in a kernel, and each is exact instead of
approximate. They are stated in `src/ai/train.rs` and worth knowing before
changing anything there:

- Only the classifier is adapted, so the hidden state at every decision is a
  constant and is cached once per example. An epoch after that costs no forward
  passes at all.
- Restricted cross-entropy zeroes the gradient outside the grammar's candidate
  set, so only rows the decoder can reach ever move. Measured on SmolLM2:
  **132 rows out of 49,152**. The trainer dequantises exactly those into an f32
  scratch and never touches the int8 classifier again.
- Teacher forcing keeps the whole spelling cacheable, and the chain of
  candidate sets is a property of the applet name instead of the task, so the
  vocabulary scan that finds them runs 21 times per trial.

`-n` **strides** through the corpus instead of taking a prefix. The splits are
positional, so the first N examples are all training examples and a short run
would report held-out accuracy over an empty set.

**Prep dominates.** Building the chains and dequantising the rows is a fixed
cost; caching the features is a forward pass per example. The report splits the
two so it is obvious which number `-n` moves.

Measured, with `-accel whpx -cpu max`, on the whole corpus: the fixed half is
739 ms and the 465 examples plus four guard goals took 1,301,778 ms, so a
forward-pass group is about 2.8 s and a full run is twenty-two minutes.

A 25-example run gives 1.8 s per group, and extrapolating that to 469 groups
predicted fourteen minutes. It was out by 55%. Small-sample extrapolation is
the same error this file warns about for accuracy figures, and it is just as
wrong about time.

Under TCG the same group took 286 s. That is where the belief a full run
needed the GF63 came from, and the belief was never measured.

`Trial` in `src/ai/train.rs` is the reusable object underneath all of this.
`prepare` builds it (expensive, once), and `score`, `paired`, `train`,
`scatter`, `gather` and `guards_hold` all run against it without touching the
model again.

### Adapters on disk

`adapter save` writes a `GLADOSA1` blob into the namespace, the adapter alone
and never the checkpoint. `tools/adapter.py` is the host-side reader, and the
format is documented there in full:

```powershell
.\tools\venv\Scripts\python.exe tools\adapter.py --selftest
.\tools\venv\Scripts\python.exe tools\adapter.py out\adapter.bin
.\tools\venv\Scripts\python.exe tools\adapter.py out\adapter.bin --export-lora out\dir.bin
```

Rows are stored **sparsely**, because they are sparse in fact: a row with a
zero low-rank factor and a default magnitude is bit-identical to no adapter. On
the measured decision layer that is 23.7 KB against 1.79 MB dense, 75x. `s` is
never stored, being `m/|W0 + BA|`, derived from a frozen weight the file does
not contain; storing it would let a file and a checkpoint disagree about a
value with exactly one correct answer.

The layout follows RustLMHub's `FfnLora::save` in every decision that could
have gone either way: a magic first and refused instead of guessed at, dims in
the header checked for exact equality, flat little-endian f32, no base weights
in the file. Byte compatibility was never available, since LoAA is LoRA over
gate/up/down while this is DoRA over the attention path and the classifier, and
every site here carries per-row magnitudes LoAA has nowhere to put.
`--export-lora` bridges the gap for one site and prints what it dropped.

### Self-modification

`godel` is the loop that lets the machine change itself, and `src/ai/godel.rs`
opens with why it departs from Schmidhuber's construction. The short version:
we have no theorem prover and could not build one for this, so proof is
replaced by a certificate cheaper to refute than to produce, over
content-addressed inputs, re-derivable bit for bit by any later run.

```
godel [status|now [n]|ledger [n]|rollback|on|off]
```

Four judges, unanimity required, each a different failure mode:

- **J1** is paired (McNemar) over the same cached decisions both variants
  answer, which is what a comparison of two percentages cannot be. It needs
  roughly six repaired validation decisions with none broken, so it needs a
  corpus subsample large enough to reach the held-out slice at all.
- **J2** replays the machine's own curiosity goals along the path the frozen
  baseline walks, and is never subsampled by `-n`.
- **J3** is structural: finite factors, positive scales, finite logits.
- **J4** is cost: rank and resident bytes, because `HEAP_LADDER` is one
  physically contiguous allocation that comes down a rung when the memory map
  cannot satisfy it.

Trials run only when the RTC hour falls in the quiet window (02:00 to 06:00)
**and** `godbits::felt()` shows no hardware input. `initiative::tick` fires one
from its sleep branch at most hourly, bounded to 24 examples and 20 s of
optimiser time, because the mind task holds the engine for the whole of a trial
and an unbounded one would take the terminal away.

Adoption is a pointer swap; the parent stays addressed and `godel rollback`
costs a pointer write. `/ai/godel/ledger.txt` gets a line per trial either way.

**The test slice carries a budget.** It is consulted only after a variant has
already won on validation, never to decide whether it won, and the ledger
counts the reads. Past three, a test figure is printed as stale and marked
unquotable. A loop that improves itself forever reads the held-out set forever,
and this tree's measurement discipline does not survive that unless somebody
counts.

`Variant.lambda`, `Variant.rule` and `Variant.skills` are hashed into the node
identity and nothing varies them yet. They are hooks for widening the search
space beyond "retrain the same thing", which is the current limitation: the DAG
will be a chain until something varies more than the random seed.

Root certificate bundle, built from the host's store:

```powershell
.\scripts\fetch-roots.ps1          # -List to see what would be exported
```

### Testing

There is no `cargo test`. This is a `no_std` UEFI binary with no host test
runner, so **verification is the boot selftests plus driving QEMU.**

At boot the system runs **eighteen selftest sections carrying seventy-three
claims** (the `aiksi` section covers the capability gate by name and never by
calling -- half that table pokes memory, drives I/O ports or paints over the
screen, and a suite that called every row to prove it exists would be
scribbling on the machine to do it), printing `ok` or `FAIL` per line: heap, timer, clock, the namespace's
Merkle addressing, fifteen sets of published cipher vectors, fault handling,
constrained decoding, the agent loop, the linear probe, the situation planner,
the initiative policy, the self-modification gate, corpus bundles, QDoRA
adapters, the backward kernels, and the trainer's arithmetic. Read that output;
it is the test suite.

Shell commands that re-run checks on demand: `tensor`, `model`, `crypto`,
`trust verify`, `fit`, `gate`, `search`, `wpa2`, `video bars`. `tensor` and
`model` are **absent from the boot sequence** and hold the checks for the
pre-tokenizer and the wide-head attention geometry.

`tools/drive.py` boots QEMU and drives the shell over a serial socket:

```powershell
.\tools\venv\Scripts\python.exe tools\drive.py "initiative off" "tensor" "model" "ask -n 20 hello"
```

It stages `BOOTX64.EFI`, resets NVRAM to pristine (a stale boot entry sends the
firmware to the UEFI shell, which looks like the system failing to boot), and
attaches serial as TCP, because QEMU's Windows stdio chardev reads console
handles and ignores redirected files, so piping a script into it silently does
nothing.

Three things worth knowing before a session goes sideways:

- **`initiative off` then `agent stop`, in that order, first.** The resident
  mind wakes fifteen seconds in and holds the engine for a whole episode, which
  presents as `drive.py` timing out with commands unsent, or as every engine
  command answering "another task holds it".
  `initiative off` stops future ticks and does not cancel the one already in
  flight, and under emulation the first tick and the first shell prompt arrive
  together: boot takes around 150 s of guest time, the tick fires at 150 s, and
  an episode is queued in the same moment the prompt appears. Two constrained
  decodes then run for minutes, and `agent stop` is what actually clears it.
- **`--qemu-extra "-accel whpx -cpu max"`, always.** WHPX is the Windows
  hypervisor and it is roughly **160x** faster than TCG on this workload:
  a forward-pass group measured 286,370 ms under TCG and 1,795 ms under WHPX,
  and a boot plus four cheap commands went from over 24 minutes to 61 seconds.
  Everything this project treated as "too slow to test here" was an untested
  assumption about the emulator, for months.
  `-cpu max` is needed alongside it: WHPX alone reports `avx2=0 fma=0` and
  `train::hardware_ok` declines. Together they report `avx2=1 fma=1 avx
  enabled=1`.
  Two things to know. WHPX raises unmasked SSE exceptions faithfully where TCG
  does not, so it surfaces real `#XM` faults that TCG hides; the first one it
  found was a genuine kernel bug in `task::alloc_fpu_area`. And at WHPX speed
  `drive.py`'s serial pacing races, so two commands can arrive concatenated on
  one line. Put a cheap command between anything that must not merge.
- **Build `--release`.** `drive.py` prefers the release artifact, so a debug
  build alone leaves a stale binary staged and the change under test never runs.

**QEMU cannot run the real model.** VVFAT is FAT16 on a fixed geometry and the
whole disk is 516 MB; `fat:32:` raises that in principle but QEMU says its
FAT32 is untested and the firmware cannot read the directory it produces. So
the VVFAT path cannot stage a checkpoint larger than the disk, and SmolLM2
in `out/` is what fits. `--stage-iso` has no such cap: it builds a one-shot
El Torito image and boots that instead, which is how any large checkpoint
reaches QEMU. Saying the big models were "only runnable on the GF63" was
wrong on both halves, since the ISO path existed and the hypervisor
accelerator made it fast enough to matter. Guest RAM must also cover the weights, which are read
whole into a pool before `ExitBootServices`, and `run.ps1 -Memory` defaults
to 2G.

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
while the output was being sliced away. It happened again while the adapter
format was being written: a sparsity claim that compared a whole file against
only the part sparsity can shrink failed for one commit, on an encoding that
was working correctly, because the log was being grepped down to the section
under active work. Read the whole thing, or grep for `FAIL` across all of it.

## Architecture

### Aiksi, the system language

`src/aiksi/` is the language everything above the kernel is written in, and
the intended relationship is C to Unix or HolyC to TempleOS: GLaDOS is written
in Rust, and Aiksi is how anything that is not the kernel reaches it. A program
is `code.ai&xi`. The extension is deliberately unusual and costs nothing --
nothing on the host has it, the shell does not parse `&`, and the path resolver
is a plain splitter.

Source -> tokens -> AST -> tree-walking evaluation, in `lex.rs`, `parse.rs`,
`eval.rs`. The single-pass code generator that replaces `eval` can be written
against the same AST and checked against the same results, which is much easier
than debugging a code generator with nothing to compare against.

**Builtin naming is a rule, not a taste.** A builtin is named after the Rust
path it calls, flattened: `crate::net::tcp::connect` is `tcp_connect`,
`crate::dev::rtc::now` is `rtc_now`. The audience is a 0.6B model and whoever
is reading the kernel source beside it, and both can apply a rule they were
told once to a subsystem they have never seen. A hand-picked name per builtin
reads better in isolation and has to be memorised one at a time, which is the
cost that actually matters. Where the rule reads badly the rule still wins: one
exception means every name has to be checked against a list again.

`eval.rs` owns the gate, the arity check and the table; `kernel.rs` owns the
arms that reach subsystems. The split keeps "what may a program do" to one
screen, and means adding a subsystem cannot accidentally edit the gate.

**`BUILTINS` is an allowlist and that is load-bearing.** Every row is
`(name, Touch, min args, max args)`, and `builtin` refuses anything absent from
it *before* dispatch. An arm added to the match without a row is unreachable --
dead code rather than an ungated builtin -- and a row without an arm answers
"no implementation", which is broken and not dangerous. It replaced two
denylists that were correct for eleven raw builtins and stopped being correct
the moment the language was wired to the network: a denylist grants by default,
so the builtin anyone forgets is the one that matters.

`Touch` has seven classes but **the sandbox question stays binary**:
Pure/Read/Write are allowed to a stored program, everything else needs
`app trust`. That follows `Manifest.raw`, which carries one bit for the reason
it states -- an operator approving a request has to hold the whole of it in
their head, and "may write outside itself but not open sockets" is a sentence
nobody can check against a program. The line for Net is whether a packet leaves
the machine, so `net_ifaces` is Read and `tcp_connect` is not.

`words` prints the table grouped by class. It is the reference.

What Aiksi reaches today: text (split/join/substr/find/replace/upper/lower/
trim/starts/ends/contains/chr/ord/repeat/pad/hexenc/hexdec), integer arithmetic
(no floats -- adding them for one builtin changes every arithmetic path), lists
(sort/reverse/slice/index/remove/range/push/set/get/len), the namespace
(read/write/ls/exists/rm/size/is_dir/hash_of/applet), the clock and counters
(rtc_now/rtc_unix/uptime/tsc/tsc_mhz/ticks/hz), tasks and memory, `pci_list`,
network status, sockets (dns_resolve/tcp_*/http_get/https_get/udp_send/ping),
the model (`ask`), the framebuffer, and raw memory and I/O ports.

Three bounds worth knowing before changing anything there. `range` and `repeat`
are capped at 65,536 because they are the easiest way for a generated program
to ask for a billion-element list, and this kernel has no OOM killer and one
address space, so the step budget never sees the single call that takes the
heap. Socket timeouts are clamped to 30 s because an unbounded one in a repaint
path hangs the desktop and the step budget cannot see a blocking call. And
`app::document` runs an application's `rows()` on **every repaint**, so that
path takes `with_step_budget(DRAW_BUDGET)` rather than the full one.

There is one TCP connection. `tcp` holds a single TCB and `connect` aborts
whatever was open before it; the builtins expose that rather than handing back
a descriptor that corresponds to nothing.

`app::migrate_extension` carries programs written before the rename across by
moving the bytes under the new name. Identity is the hash of the file contents,
so manifests, grants and lineage survive untouched. It runs after every
namespace init rather than once, because a restored snapshot can be older than
the rename.

### Graphics and the desktop

Rendering is composed, then diffed. `desk::draw` repaints everything
(wallpaper, icons, every window back to front) into `gfx::compose`'s heap back
buffer, and `present()` writes to the framebuffer only the row spans that
differ from the shadow of what is already on screen. Total repaint keeps the
window manager obviously correct; the diff is why nothing flashes and why hover
feedback on pointer motion is affordable. The console bypasses the next present
through `compose::flush_rect` so shell output stays immediate; both paths
update the shadow, so they cannot disagree about what is on screen.

The pointer's whole vocabulary lives in `desk::press_at`, and every layout it
hit-tests (`task_layout`, `chrome`, `Panel::rects`, `Browser::metrics`,
`dropdown_rows`) is the same function the paint pass draws from. A control that
highlights in one place and presses in another is the class of bug that split
forbids. Everything the pointer does a keystroke also does, because serial
cannot inject PS/2 packets and `win keys` is how the desktop gets tested
headlessly. Screenshots come from `drive.py --screenshot out/x.png`, pointer
events from `--mouse "mouse_move dx dy"` and `--mouse "mouse_button 1"` (QEMU
monitor, relative moves from (0,0) at boot).

The look is 98 plus 3.1 plus the palette from the sign over our door: icons and
Start and gradient titles from 98, bevels and dialogs that hug their content
from 3.1. `todo` (shell) and the ToDo window share one list. It is the hand-off
note for what to test at the GF63, since the machine that builds this is a
different machine from the one that runs it.

**Apps are `Content::App(Box<dyn DeskApp>)`** (`gfx/mod.rs`): a window whose
client area belongs to a program, being Paintbrush (`paint.rs`), Write
(`write.rs`) and Minesweeper (`mines.rs`). Six methods (draw, key, press,
right_press, drag, release/wheel); every handler returns whether it consumed
the event so unclaimed keys fall through to the window manager. `draw_in` takes
`&self`; layout facts discovered while drawing go in `Cell`s, the Browser's
pattern. Held-button motion is forwarded to the pressed app (`APP_PRESS` in
desk.rs), which is what a brush stroke is, and the second button goes to the app
before it means the system menu, which is how Minesweeper flags.

Four lessons already paid for. Do not relearn them:

- **`open_app` returns focus to the terminal**, like `open` and `open_browser`.
  The desktop takes *every* key while a non-terminal window has focus, so an
  app that kept focus ate the next serial command line. Minesweeper consumed
  `echo after-mines` a byte at a time and flagged a cell on the `f`.
- **Alt-Tab swaps the top two and does not rotate.** Rotating made a second
  Alt-Tab land on a third window; scripts and habit both need over-and-back to
  be two presses. Headless recipe: every `win keys` line that drives an app must
  be self-contained, as in `alttab,...,alttab`, because between commands the
  focused app would swallow the next line.
- **QEMU monitor `mouse_move` deltas must stay within +-255 per axis.** Bigger
  deltas set the PS/2 overflow bit and the driver correctly discards the
  packet, so the pointer simply does not move, which reads as a dead drag
  instead of a clamped one.
- **`font::GLYPH_H` is 8 and not 16** (glyphs are 8x8, doubled by
  `CHROME_SCALE`). `TITLE_H` is therefore 24, and the caption buttons are 12x12
  at the bar's right end. Choreographing clicks from remembered metrics instead
  of a `[desk] press` trace cost two full test cycles aimed 40 pixels left of
  the close box.

`write` is two things told apart by shape: with `<path> <text>` it is the
sysbox applet, with at most a path it opens the editor (decided in
`shell::execute` *before* sysbox dispatch, which would otherwise claim the bare
form and print usage). Paint saves `/draw/painting.ppm` (P6); `tree::put`
creates parent directories, so no mkdir ceremony.

### The Oracle (God Says, made honest)

`src/ai/futures.rs` and `src/gfx/oracle.rs` are the TempleOS "God Says"
descendant. Terry drew uniform words from Vocab.DD seeded by `KbdMsEvtTime`,
the timing of the operator's own hands. We keep the entropy (`src/ai/godbits.rs`:
every keyboard and mouse ISR deposits `rdtsc() >> GOD_BAD_BITS`, folded into the
sampler) and change the subject from hallucinated words to the one future that
is actually knowable, which is this machine's.

`futures::sample()` runs once a second from the clock task, recording heap,
task-switch rate, the operator's touch rate and task count into a ring. On
consult, a linear dynamical model `v_next = a + b*v + c*u` is fitted per
variable by the router's own Cholesky (`probe::ridge_solve`), and the state is
rolled forward under three interventions: `do(activity := 0 / mean / high)`,
the counterfactual "left alone, carried on, put under load". The window plots
forked timelines, solid white history to the `now` line and three coloured
projections after. It is genuinely causal, being a controlled linear system
fitted from real telemetry, and it is never prophecy. The word-prophecy first
draft was scrapped for exactly that reason.

Two gotchas paid for here:

- **`lapic::ticks()` is the timer-interrupt count at `TIMER_HZ` (100/s)**, and
  is not `lapic::timer_hz()` (the calibrated APIC frequency, in the millions).
  Dividing uptime by the latter put every reading at 0s. `mem` and `uptime` use
  `TIMER_HZ`; so must anything converting ticks to seconds.
- **`win keys` bypasses the hardware ISR**, so scripted keystrokes do not feed
  the entropy ring. Only real hardware events do. That is correct, since the
  entropy *is* hardware timing, and it means headless tests show "fed by ~1
  touches" while the ring lights up on the GF63.

### Boot

UEFI already delivers long mode, CPL 0 and an identity map, so this UEFI
application *is* the kernel. There is no ELF loading, relocation or handoff
ABI. `main.rs` reads the model, tokenizer and root bundle **before**
`ExitBootServices`, because that is the only moment a filesystem exists.
Everything after runs on our own page tables.

`gfx::splash` owns the framebuffer during boot; the console writes to its RAM
shadow grid without painting, and `finish()` repaints the whole log. Anything
that draws during boot must check `splash::active()`, and the fault reporter
and panic handler call `splash::abandon()` first, because on the GF63 the
framebuffer is the only diagnostic channel there is.

### Concurrency

`sync::Racy<T>` is **not a lock.** It is single-core interior mutability and
the designated grep target for the day SMP arrives.

`task::yield_now` disables interrupts across the context switch. This is
required: `schedule()` stores `CURRENT` and *then* switches stacks, so a timer
tick landing between them saves the outgoing stack pointer into the wrong slot
and one task becomes unresumable. The interrupt path is safe because a gate
clears IF for it.

Long-running work in a resident task is fine and does not freeze the machine,
since the scheduler preempts at 100 Hz. What it does do is hold the engine, and
`with_engine` refuses every other task while one holds it. That is why the
trial the initiative loop runs at night is bounded to a small budget: an
unbounded one would leave the shell answering "another task holds it" for the
length of it.

### Networking (`src/net/`)

Interfaces live in `iface`: `lo`, `eth0`, `wlan0`. A driver implements
`iface::Nic`; `net::init` tries e1000 (QEMU) then rtl8168 (the GF63's real
card, `10ec:8168`). Routing picks an interface by destination, and every layer
above asks for a source address instead of assuming one exists.

**`poll` never dispatches into a transport state machine. It queues.** Sending
calls `send_ipv4` then `resolve`, and `resolve` calls `poll` while waiting for
ARP. Running a state machine from there would let a connection re-enter its own
control block while an earlier borrow is live. TCP and UDP drain their own
inboxes.

TCP advances only while the shell is idle (`tcp::service` from the idle loop)
or inside a blocking call. There is no interrupt-driven receive.

The wireless card is CNVi, so the MAC is in the PCH and the M.2 module is a
radio. `net/wifi.rs` identifies hardware and refuses to pretend; `hardware()`
lists every network part on PCI and USB with what drives each, and boot prints
it.

For the USB dongle, everything above the transport is finished and checked at
boot: `net/ieee80211.rs` builds probe requests and parses beacons,
`dev/rtl8188eu.rs::desc` builds and reads the TX and RX descriptors, and
`net/wpa2.rs` runs the handshake. `bring_up` applies all four initialisation
tables including the radio, over the serial interface the radio actually needs.
What is missing is the transport in between: LLT, the FIFO boundary that gates
the MAC TX/RX enables, channel selection, efuse, firmware, and handing a
descriptor to a bulk endpoint. None of the chip-facing half can be exercised
here, since QEMU has no model of the part.

### Crypto (`src/crypto/`)

Written from scratch, and this is the one place where that is a liability
instead of a virtue: a bug here produces output that works perfectly and is not
secure. Primitives were chosen for checkability, ChaCha20 over AES-GCM (no
key-dependent table lookups) and X25519 over a NIST curve. Every one is checked
against published RFC vectors at boot.

ECDSA uses **Jacobian** coordinates. Affine cost an inversion per point
operation, a full modexp, which for P-384 meant around 460,000 allocating
multiplies per signature and exhausted the heap. `Mont::inv_prime` takes and
returns *ordinary* values; passing it something already in Montgomery form
computes the wrong thing silently. Use `Curve::inv_m`.

TLS 1.3 validates the chain, the transcript signature, dates and name, then
**reports** instead of enforcing. A caller that cares must check
`identity.ok()`. There is no revocation.

Key material comes from `src/rng`, a fast-key-erasure ChaCha20 DRBG built over
the `chacha::apply` the boot selftest already checks against RFC 8439. One
64-byte block per step: the first 32 bytes overwrite the key and only the last
32 leave the module, so the state that produced an output is gone before the
caller sees it. That is the only way a kernel with one address space and no
process isolation gets backtracking resistance.

Two entropy sources, and the second exists because the first has a blind spot.
Keyboard and mouse interrupt timing arrives through `godbits::ins`; NVMe
completion latency arrives through `rng::add_device_entropy`, which is
deliberately **not** routed through `godbits`. That function also feeds
`godbits::felt`, which `initiative` and `godel` read to decide whether a person
is present, so disk traffic going through it would make an unattended machine
look occupied and stand down the loop that only runs when nobody is there.

One bit is credited per event whatever the source, and 256 events are needed
before `fill_secret` will answer. That figure is an assumption and not a
measurement, and it is the weakest link in the module. `fill` still answers
below the threshold for anything that wants unpredictability without depending
on it; key material takes `fill_secret`, which refuses, because a generator
that quietly degrades for a private key is the failure this section exists to
warn about.

### The model (`src/ai/`)

Qwen3-0.6B, int8, around 570 MB on the ESP, referenced in place in the
LoaderData pool instead of being copied to the heap. SmolLM2-135M still loads
and is the small checkpoint to reach for when something needs to run under
QEMU. Qwen3.5 hybrids load through the v4 path.

The module map, since `src/ai/` is now twenty-three files:

| | |
|---|---|
| `model.rs` `weights.rs` `tensor.rs` | The forward pass, `Mat`, and the kernels |
| `tokenizer.rs` `vocab.rs` `corpus.rs` | Text in, and the routing corpus |
| `constrain.rs` `harness.rs` `sample.rs` | The grammar, the decode loop, the splits |
| `probe.rs` `council.rs` `deliberate.rs` | The closed-form router and its confidence |
| `agent.rs` `context.rs` `initiative.rs` | Episodes, situation, the resident mind |
| `aixi.rs` `futures.rs` `godbits.rs` | Planning over fitted dynamics, and the Oracle |
| `adapter.rs` `backward.rs` `train.rs` | QDoRA, the adjoints, and the trainer |
| `godel.rs` | Variants, judges, ledger, adoption |

**Qwen3 differs from Llama in two ways and neither fails loudly.** Its head
width is *stated* (128) instead of derived (1024/16 = 64), so `wq` is
`[2048, 1024]` and the attention path is wider than the residual stream; and it
RMSNorms each head's query and key before RoPE. Ignore either and the model
loads, runs, and generates confident nonsense. `Config::head_dim` and
`Config::qk_norm` carry them, and the `GLADOSM3` header (v3) records them per
checkpoint. v2 files still load, their defaults being exactly the Llama ones.

**RoPE pairs `i` with `i + head_dim/2`.** The interleaved convention, `2i` with
`2i+1`, is wrong for anything trained through transformers, where the reference
is `rotate_half`. The kernel used interleaved for a long time and nothing
looked broken, because both are norm-preserving rotations by the same angles:
no NaN, no drift, no error. The model stays fluent and attends by a scrambled
notion of distance, which is indistinguishable from a small model being small.
It cost SmolLM2 `"The capital of France."` followed by blank lines where the
corrected path gives `"The capital of France is Paris. Paris is a city known
for..."`. `Config::rope_interleaved` is true only for genuine llama2.c
checkpoints.

Generation is memory-bandwidth bound, since bytes read per token is roughly the
model size, so 570 MB against 135 MB is about 4.4x the time per token. The
classifier is 155 MB of that, and constrained decoding only ever needs logits
for the reachable set, so restricting that matvec is the obvious win when it
matters. `train.rs` takes exactly that win: it dequantises the 132 reachable
rows once and never reads the int8 classifier again.

`ask` closes the `<think>` block itself unless given `-t`. Qwen3 left alone
reasons at length, which is the model working as designed and useless at a
64-token budget. `has_think_token()` decides by asking whether the tokenizer
knows `<think>` as one token, a property of the vocabulary instead of a guess
from a name.

The tokenizer carries which pre-tokenizer regex the checkpoint trained with.
SmolLM2 is the GPT-2 pattern; Qwen3 spells out the cl100k one, where a word may
be led by any non-alphanumeric (`(x` is one piece), digits come one at a time,
and punctuation swallows following newlines. Using the wrong one moved around
12% of tokens on the training corpus, again with no error, just a model fed
sequences it never saw.

Two routing paths, and the interesting result is that the older one wins. `act`
decodes an applet name token-by-token under a grammar; `route` reads one hidden
state and hands it to a closed-form ridge regression (Widrow-Hoff, 1960) solved
by Cholesky in-kernel, 12,672 parameters, around 1.6 ms, **no transformer
forward pass**, and better held-out accuracy.

**Constrained decoding makes invalid output unreachable, where merely making
it improbable would leave it reachable.** The grammar is built from the live applet table; read-only mode
works by removing mutating applets from the reachable set *before* sampling,
and never by checking after.

Three "cores" vote (probe, hashed-n-gram Bayes, lexical). Their *agreement* is
the signal instead of their vote: 90% right when all three agree against 61%
when they split. That gap is what `gate` acts on.

**The frozen base is the load-bearing property.** Nothing above the adapter
moves, so a hidden state is a constant, a constant can be cached, and a cached
decision can be replayed against any number of candidate adapters for the price
of a dot product. Training is affordable because of it, judging is nearly free
because of it, and any verdict the machine reaches can be re-checked later for
almost nothing because of it. Anything that proposes to train the attention
path is proposing to give this up, which is a real trade and worth naming
before it is made.

### Storage (`src/store/`, `src/sysbox/`)

Content-addressed: objects named by SHA-256 of their contents, assembled into
Merkle trees. A copy is O(1), a snapshot is one root hash. **The content hash
covers content only and never block locations**, since otherwise moving a block
would rename an object.

Directory entries are kept sorted, so `children()` returns lexicographic order.
That is why `vocab::record` zero-pads blob names to four digits: sorted order
becomes insertion order, and every positional split boundary depends on it.
Past 9999 the padding truncates and the property fails silently, which is why
`dataset.py` refuses to emit a larger bundle.

NVMe writes are locked by default. `store::init` unlocks only after
`find_store_region` names a target, and `Store::format` re-checks. On a disk
fully allocated to Windows there is no such region and init fails, which is the
intended outcome. Every error path re-locks; leaving it open is how a safety
mechanism becomes decorative.

## Evaluation discipline

This project measures instead of arguing, and the harness exists because the
measurement was got wrong three separate times: a grid sweep scored on the test
set, cross-validation folded by template family, and a test set that *moved*
whenever the corpus was appended to.

There are **three** splits, and `vocab::splits()` is the single place anything
asks for them. It returns the compiled `SEED_TRAIN` and `SEED_VAL_END` until a
bundle is imported over the corpus, and the imported boundaries after. Reading
the constants directly is the bug that arrangement exists to prevent.
Validation is spent freely; the test slice is read once. `search` adopts a
configuration only when measured better.

Corpora hold out **whole template families** and never sampled instances, since
instances within a family differ only by slot values, so an instance split
measures memorisation while looking like generalisation.

**A loop breaks this, and `godel` is a loop.** A machine that improves itself
every night reads the held-out set every night, and each read makes the
reported figure more optimistic. So the test slice carries a budget in
`/ai/godel/test-budget`, it is consulted only after a variant has already won
on validation, and past three reads the figure prints as stale and marked
unquotable. Any future loop that touches the test slice must go through
`godel::read_test` for the same reason.

Negative results stay in the tree. Training the adapter head *hurts* at this
data scale; the Product-of-Experts council does not improve accuracy. Both are
kept because the reason to know them is the reason they were worth measuring.

`tools/traces.py` reports what it could **not** produce and the per-family
imbalance unprompted. A generator asked for 20,000 that quietly returns 54
near-duplicates yields a corpus that trains a model to recite.

Sample sizes get stated wherever a figure appears. The adapter trainer has been
exercised on subsamples of a few dozen decisions, which establishes that the
machinery composes and establishes nothing about how much it helps. Numbers
from those runs do not belong in a claim.

## Gotchas that have already cost time

- **`extern "C"` on `x86_64-unknown-uefi` is Microsoft x64 and not System V.**
  The context switch is pinned to `extern "sysv64"` explicitly.
- **Do not take the max over every UEFI memory descriptor.** OVMF describes
  `Reserved` space to 1 TiB; using it as a map limit exceeds one PDPT and the
  identity map silently fails, falling back to firmware tables that map page 0,
  which made the null-dereference selftest pass without faulting.
- **A guarded match arm placed after the arms it guards is unreachable.** The
  compiler said so in a warning nobody read, for several commits. Anything
  added to `shell::execute` with a guard goes *before* the bare arm.
- **`prefill` and `forward` must agree about adapters.** For a long time
  `prefill` ignored them entirely, so an adapted model prefilled its prompt
  through the frozen weights and decoded through the adapted ones. The same
  position computed two different things depending on which path reached it,
  with nothing faulting and no logit going non-finite. It was unreachable until
  the first adapter anybody would keep got attached.
- **`drive.py` prefers the release artifact.** A `cargo build` alone leaves a
  stale release binary staged and the change under test never boots, which
  presents as a change that mysteriously did nothing.
- **A `debug_assert` is only checked in debug builds, and this tree is driven
  in release.** `Dora::refresh` asserted `k * r == b.len()` where `b` is
  `out * r`, so every debug build attaching any Qwen3 q/k/v site would have
  panicked on a claim about the wrong dimension. It never fired because nothing
  runs debug under QEMU.
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
- **The kernel heap is a ladder and not a constant** (`HEAP_LADDER`). It is one
  physically contiguous allocation, the GF63 cannot be tested from here, and a
  fixed size its memory map cannot satisfy is an unbootable system. Boot prints
  the size it got and says when it had to come down a rung.
- The boot disk is **counterfeit**: it advertises 976 GB and holds 14.67. Hence
  MBR (a GPT backup header would land in flash that does not exist) and hence
  `SafeLimitGB` in `build-layout.ps1`. Do not put anything you care about on it.

## Conventions

Comments explain *why*, and specifically why an obvious alternative was
rejected. Several of them record measurements that overturned a confident
assumption. Match that register, and do not add narration of what the code
plainly does.

Commit messages follow the same shape: what changed, what it cost to find out,
and what is still unverified. State plainly when something is untested. The
RTL8168 driver cannot be exercised in QEMU (which emulates the 8139) and says
so in its own commit.

Git identity is not configured in this repo. Every commit needs it passed
explicitly:

```bash
git -c user.name=IlumCI -c user.email=ilumbackup@gmail.com commit ...
```

Nothing else goes in the trailer. No co-author lines, no tool attribution.

Note the enclosing `C:\` drive is itself a git repository. Confirm the working
directory before staging, because `git add` from the wrong one stages a
different tree entirely.

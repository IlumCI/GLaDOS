# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this is

GLaDOS: a from-scratch, non-Unix, ring-0 operating system in Rust for one
specific laptop (MSI Thin GF63 12UC, board MS-16R8), built around a language
model that lives *inside* the kernel. No user/kernel split, no syscalls, no
process isolation, one address space. A tool call from the model is a function
call.

170 files, roughly 108,000 lines. **Three** things in the kernel we did not
write, and the list has grown twice, so it is worth stating precisely rather
than approximately:

- **Rust `core`.**
- **`src/dev/rtl8188eu_tables.rs`**: the RTL8188EU initialisation tables plus
  the RF and descriptor register constants, taken from Linux's GPL-2.0
  rtl8xxxu driver because there is no other source for them. One file, marked
  as such at the top. `tools/rtlconv.py` regenerates the second half from a
  checkout and states its provenance; the tables came the same way.
- **`src/doom/`**: 9,900 lines adapted from
  [room4doom](https://github.com/flukejones/room4doom) (MIT, Luke Jones),
  which is itself a transliteration of id's DOOM. Eleven of its thirteen files
  say at the top what they came from and what changed. This is the largest
  piece and it is the *point* of that directory rather than an exception to
  the rule: `src/doom/` exists to find out whether software written somewhere
  else can be brought over without dissolving into the kernel, and code we
  wrote ourselves would answer nothing.

Of that last one, **`src/doom/info.rs` is generated rather than copied**, and
the distinction matters. It is 3,900 lines of DOOM's *content* -- every sprite
frame, how long it shows, what it becomes next -- emitted by
`tools/doominfo.py` from an upstream checkout, the same arrangement
`rtlconv.py` has with the wireless tables. Nine hundred and sixty-seven states
is not a table anybody writes by hand correctly, and it can be re-derived by
anybody with the checkout.

No WAD is in this repository and none ever will be. What `src/doom/` reads is
`DOOM1.WAD`, which belongs to id, or FreeDoom, which does not; every byte of
art it draws comes off the boot volume at runtime.

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
It is seeded at boot from `src/ai/corpus.rs` (717 examples, compiled in so the
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
- **`train adapter`** moves a QDoRA adapter over the model's *classifier*.
- **`deeptrain`** moves every q/k/v site as well, through `forward_taped` and
  `Model::backward`. Different economics and therefore a different command:
  `Trial::train` rests on hidden states being constants below the classifier,
  so it caches them once and an epoch costs no forward passes; move a
  projection in layer three and every state after it moves, so there is nothing
  to cache and every epoch pays a forward pass per example. Measured under
  QEMU on SmolLM2-135M: 24 forward-and-backward passes in 110 s. That figure is
  from an emulator and is not evidence about the GF63.

  Its objective is full-vocabulary cross-entropy on the applet name's first
  token, which is weaker than `Trial`'s -- that scores every step of the
  spelling under a grammar that has already removed unreachable applets.

  **It goes through the judges now.** It was the one training path with none
  in front of it -- it walked gradients into the live model and kept whatever
  came out. It is a `ProposalKind::Deep` proposal, so J1-J4 decide, the
  incumbent is copied out before training and put back on every path that does
  not adopt, and a rejection prints "rejected, and reverted".

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
**and** `godbits::felt()` shows no hardware input.

**The night branch rotates over every axis that has a judge.** It knew two
jobs and godel always won the tie, so the adapter grid was walked to
exhaustion while the routing rule, deep training, a skill the agent compiled
and a core the machine wrote were never tried unattended at all -- "search
space exhausted" was the end of self-improvement, eight points and then
nothing, every night forever.

`godel::next_proposal` starts from the number of verdicts already recorded and
takes the first kind from there that has work, so which axis a given night
takes is a function of the ledger rather than of a coin -- the same
re-derivability argument that makes `frontier` walk a declared grid. An
exhausted axis costs one skipped slot rather than an idle night, and the loop
stops only when every axis is out of moves. Order is cheap-and-declared before
expensive-and-composed: a grid point and a rule change are minutes, a deep
trial is two passes over the corpus, and composing a core spends a dozen
decodes writing something that may not survive its first judge.

`godel next` reports where the rotation stands without taking a turn. It
deliberately does not ask the last slot whether it has work, because finding
out costs those decodes -- a command answering "what would you do tonight"
must not spend the night doing it.

Widening this had to come last, and the ordering is the point rather than an
accident: an axis in the rotation without a judge in front of it is a machine
adopting things nobody measured.

Still not done from that plan item: an authored application is left as a draft
and never adopted, and `aixi`'s plan is still stringified to a report rather
than gating how much the loop attempts.

That question -- is anybody here -- is `quiet_hours()`, and it is shared with
the other unattended job. `initiative::tick`'s sleep branch also writes an
application from `WORKS`, leaving a draft it never adopts. Each job owns its own
switch (`godel off` does not stand down the writer) and `NIGHT_BUSY` claims the
whole block, because `tick_inner` is reentrant across two tasks -- the resident
mind and the shell's `initiative now` -- and a local flag is not a rule. The
journal caught that: two entries under one tick number with clocks fifty-four
seconds apart.

**One queue, one busy flag, one abort, one task.** `agent::Job` is either an
episode or an application to write, and both run on the resident agent task.
That is not tidiness: a second task running a second kind of work would need a
second entry in the engine's exclusion check, which is the stale-call-site
failure that check's own doc comment warns about, and there the failure is two
forward passes interleaving in one KV cache rather than an error message.
`agent stop` therefore cancels either kind without knowing which, and `author`
returns to the prompt immediately.

**The engine has one holder.** `HOLDER` records the task, `with_engine`
*claims* it for the length of a call rather than consulting somebody else's
busy flag, and `claim_engine()` returns an RAII `EngineClaim` for work spanning
many calls. That distinction is the point: two `&mut Engine` at once is
undefined behaviour and the per-call claim prevents it with nobody having to
remember a flag; somebody else decoding *between* two of your calls is not UB
but corrupts the KV cache, `pos` and `last_token`, and produces confident
nonsense -- so the mind and the agent task each hold a claim for a whole
episode or authoring run.

It replaced a flag-and-id pair per task, which is why a third holder was
invisible: the nightly `godel` trial runs on the initiative task, set neither
flag, and `with_engine` handed a second `&mut Engine` to anyone who asked
during its twenty seconds. Adding a third pair would have made the next
omission just as quiet. The claim is reentrant within a task, deliberately:
nesting is still forbidden, but a claim that refused its own holder would turn
any nesting that does exist into a silent `None`.

`engine_refusal()` says which of the two reasons a borrow failed. Every caller
used to print "no model loaded" for both, which became actively misleading once
`author` started returning to the prompt immediately -- the next `ask` reported
the model absent while it was loaded and working.

A run publishes `author::Progress` and calls `desk::draw()` per step, which the
"Writing" window shows: step N of M, clauses met, and the last verdict
verbatim. **No progress bar and no estimate** -- a step can be a skeleton that
lands instantly or a decode that takes seconds, and the loop ends when the
checks pass rather than after a known amount of work. `godel.rs` already paid
for that lesson. Stop is a button and also `author stop`, because serial cannot
inject PS/2 packets and a control with no typed equivalent never gets tested.

Windows that hand focus back must be placed clear of the terminal
(`desk::clear_of_terminal`). `open_app` centres, which is right for something
opened by a click and wrong for something opened on the machine's own
initiative: handing the keyboard back also raises the terminal over it.

`initiative now` bypasses the settle window and nothing else, which is what
makes any of this testable; QEMU's `-rtc base=...T03:00:00` puts the guest
inside the quiet window. Note the first forced tick is consumed recording when
the prompt appeared, so it takes two to get one. `initiative::tick` fires one
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

**There is no random seed, and that was the bug.** `Dora::new` starts at all
zeros, nothing in the training path is random, and `scatter` builds a
classifier-only adapter so the cached features do not move either. Both callers
passed `Budget::default()`, so every trial trained a bit-identical adapter with
the same content hash, and after the first adoption each later one was compared
against itself: nothing repaired, nothing broken, rejected, forever.

The fix is not randomness -- determinism is what lets any later run re-derive a
verdict, which is the claim the module rests on. Instead `trial` takes a
`Proposal` naming every knob, and `frontier()` walks a declared `GRID` in a
fixed order, skipping points marked in `/ai/godel/tried`. The search is
therefore re-derivable rather than merely repeatable: the next point is a
function of the markers, not a coin. `godel space` shows what is left,
`godel forget` walks it again.

`Proposal::render` uses six decimal places where `Variant::render` uses two.
A proposal is identified by its rendering alone, so 3e-4 and 2e-4 rendered at
two places would be one point; a variant carries its adapter's hash as well, so
there the imprecision is cosmetic. `Variant::render` keeps `push_f2` because
changing it would re-address every node already stored.

**A core is a proposal now, not only an operator command.** `core trial
<hash>` runs `harness::core_bench`'s three judges and then does what `core
install` never did: writes a node, a ledger line, and something `godel
rollback` can undo. `Variant` gained `core`, and `rollback` restores it --
without that, rolling back an adopted core left it installed and voting, so
the pointer said one thing and the machine did another. Deliberately a
sibling of `trial` rather than a branch inside it: `trial` is a training run
whose judges read cached features, a core changes no weights, and folding
two economics behind one name is what `deeptrain` was split out to avoid.

**`deeptrain` records itself.** It moves every q/k/v site and used to touch
neither head nor ledger, so `ensure_head` wrote a node describing a
classifier-only variant -- not "unknown", which would have been honest, but
*wrong*. `Variant.deep` is read off the adapter (`qkv.iter().any(...)`), so
it cannot disagree with what is attached, and the node is written when the
training happens rather than when the next trial notices.

It is judged now, and how is the interesting part. Those judges rest on cached
features and a deep adapter moves the features, so a `Trial` prepared before
the run cannot judge what came out of it -- and re-preparing one afterwards
does not work either, because its decisions are recorded along the baseline's
own decode path, so a change that alters that path alters how many decisions
there are and the two lists stop lining up item for item.
`harness::route_snapshot` pairs on *routing* instead: one entry per example,
the same examples both times, and the four curiosity goals recomputed on both
sides. Two full passes over the corpus, which is the frozen-base trade with a
number on it -- as is J4 reporting 2,646 KiB resident at rank 4 against a few
KiB for a classifier adapter.

**Both new `Variant` fields render only when non-default.** Adding a field to
a hashed structure re-addresses every object that already exists unless the
rendering omits it when the object omits it. An unconditional line would have
re-addressed every node in every lineage, making `head` name something that
no longer reproduces -- the change meant to extend re-derivability breaking
it instead. Three selftest claims assert this rather than trusting it.

`Variant.skills` carries an adopted skill's address now, and `rule` is
searchable at last -- see below.

**The routing rule needed a different judge, which is why it sat unsearchable
behind a comment.** Every other proposal is selected by J1, a net repair beyond
the noise, and a rule change is mostly not that: what it moves is
*calibration*, how much better the council's confident answers are than its
unconfident ones. `agreement` counts how many of the three cores landed on the
winner, and the winner is what the rule decides, so accuracy and confidence
move together and a judge watching one adopts the trade without noticing it was
made. `harness::rule_bench` evaluates both rules per item from one fitted probe
-- paired, because fitting twice would give two rules two slightly different
councils and call the difference an effect -- and `godel::trial_config` judges:

- **J1 do no harm.** Requiring a *win* here is exactly what made the axis
  unsearchable. But "not significantly worse" alone is too weak in the losing
  direction: it adopted `ProbeOnly` on a measured `fixed 4 broke 10`, a net
  loss of six items out of 180, because chi reached 1.79 against 3.84. The
  floor is symmetric now -- `MIN_FIXED` says a net repair under four is not a
  repair, and just as well that a net loss over four is not nothing.
- **J2 must improve.** The confidence gap has to widen by `MIN_CAL_GAIN`, and
  the confident set must not collapse to four fifths of what it was. A rule
  beautifully calibrated over six items has stopped answering, not improved.

Measured: `probe` and `lexical` both cost accuracy and are refused; `majority`
and `withcore` change nothing and are refused for having improved nothing.

**An adopted core is inert unless the rule is `withcore`.** `core_vote` returns
`None` when `!rule.needs_core()`, and `rule_in_force()` defaults to `Majority`
-- so a core can pass all three of its judges, be installed, and never be asked
anything, which from the ledger looks exactly like a core that is working.
`core judge` says so now, and `godel rule withcore` is the judged way to change
it.

**Nodes record the rule in force, not the proposal's.** It was `p.rule`, and
every grid point carries 0 -- `ProbeOnly` -- while the machine has been running
the default `Majority` throughout. Every node in every lineage therefore
recorded a rule its variant was never measured under. `rollback` restores the
rule too, but only when the two nodes disagree about it: unconditional
restoration would switch a lineage full of those legacy zeroes to a rule none
of them ever ran. `rule` reaches the variant
from the proposal now but nothing varies it, because J1 is a paired test over
routing decisions and the rule changes `Verdict::confident` -- how much the
council will claim, not what it answers. Varying it without a judge that
measures it would be search without selection.

Root certificate bundle, built from the host's store:

```powershell
.\scripts\fetch-roots.ps1          # -List to see what would be exported
```

### Staged updates

The boot image is replaced by the *next* boot, not the running one: the
firmware's FAT driver is the only writer of the ESP that exists while a boot
image can still be swapped, so `update::hook` runs before `ExitBootServices`
and after `cpu::set_runtime` (the reboot goes through the runtime table).

Stage one by putting three files on the ESP and rebooting:

```
GLADOS/STAGED.EFI     the new image
GLADOS/STAGED.SIG     its detached GLADOSIG signature
GLADOS/UPDATE.FLG     any contents; presence is the request
```

**A key is provisioned and the updater is live.** This said "`UPDATE_KEY` is
all zeroes, so `verify` answers `NoKey`" for a long time after it stopped being
true, and the cost was real: it is why a session went looking for why the
update channel could not publish, on the day it published. `UPDATE_KEY` in
`src/update/mod.rs` holds a real P-256 point, tags push, and `release.yml`
signs and uploads.

To rotate: `tools/sign.py --keygen --out FILE`, paste the public rows into
`src/update/mod.rs`, and rebuild -- adopting a signer is itself a kernel
change, which is the point. **Use `--out`**: without it the private half goes
to stdout, and that is exactly how the last one died. And the first build
carrying a new key cannot be delivered by this system, because no kernel in
the field trusts it yet; that one ships as an ISO.

A consequence worth stating before it surprises somebody: the first build
carrying the key cannot be delivered by this system, because no kernel in the
field trusts the key yet. That one ships as an ISO like every release before it.

### The updater that stages them

`update.rs` became `src/update/` when it grew a client. The boot half is
unchanged; what is new is everything on either side of it.

| | |
|---|---|
| `mod.rs` | `verify`, `decide`, `hook`, `mark_healthy` -- untouched |
| `manifest.rs` | the signed manifest: parse, verify, compare versions |
| `fetch.rs` | DHCP if needed, resolve, TLS, and every refusal named |
| `stage.rs` | ranged unlock, three files, read back, re-lock |
| `channel.rs` | source origin, channel, device code, what was last seen |

```
update              running version, channel, what was last seen
update check        ask the channel what it offers
update fetch        download it, verify it, hold it
update stage <hex>  write it to the boot volume
update unstage      call a staged update off
update source <url> | channel <name> | link <code> | unlink | verify
```

Separate verbs rather than one `update now`, for the reason `fat unlock` is
separate: claiming a write range on the boot partition is the most dangerous
thing this system does. `update stage` wants eight characters of the image
digest typed back, in the `app trust` idiom.

**A signed manifest is its text followed by exactly 80 bytes of GLADOSIG.** One
object, because there is one TCP connection and no pipelining, and because two
objects can be served out of step and produce a signature failure that is
really a deployment race. `tools/manifest.py` writes it and `--verify` reads it
back with a reimplementation of the kernel's parser, the bargain
`tokenizer.py --verify` makes.

**The manifest is signed as well as the image, and the two catch different
things.** The image signature proves the bytes came from the signer; it says
nothing about *which* signed image was offered, and an old one with a known
hole in it verifies perfectly. `Manifest::is_upgrade` is the other half, and it
is the only anti-rollback there is.

**The source URL is configurable and that is safe.** It looks like a trust
anchor and is not one: everything it serves is signed by the key compiled in
here, so a wrong or hostile source can deny service and reveal which version is
being asked for, and can install nothing. `channel.rs` stores an *origin* only
-- both channel paths are compiled in, so switching channel cannot be done by
pointing at a different host.

**`fetch` refuses anything short of `Identity::Verified`.** The `https` verb
prints the verdict and shows the body anyway, which is right for a person
reading a page and wrong for a machine deciding what to boot. No `roots.der`
means no update.

Three limits that are stated rather than discovered:

- **A live ISO cannot update itself.** ISO 9660 is read-only and there is no
  writable ESP. `find_esp` says so in those words rather than failing obscurely,
  and it looks for a volume carrying `\EFI\BOOT\BOOTX64.EFI` rather than
  taking the first FAT partition -- writing three files to somebody's data
  volume and reporting success is the worst outcome available here.
- **FAT32 only**, because `fatw` refuses FAT16 for a reason it states.
- **Kernel images only, never weights.** 570 MB to 1.9 GB through a 32 KB
  receive window with no Range requests, no resume and the whole body in the
  heap is not a download. Weights change by reinstalling.

`https_get` was replaced by `tls::https_fetch`, which takes a deadline, honours
`Content-Length` and reports whether the body is whole. The old one had a fixed
fifteen-second deadline and **returned a truncated body with no error**, which
the signature check would then have blamed on the signer. It also removed the
third inline copy of response-splitting; `http_response` is gone, since
`https_fetch` never produces an unsplit response.

### Install media, built in CI

`release.yml` has two jobs. `publish` builds, signs and ships the kernel image
plus its manifest; `iso` fans out over a matrix and assembles install media.
The second `needs` the first and nothing needs the second, which is the right
shape: every machine in the field is waiting on the manifest and nothing at
all is waiting on an ISO.

**The weights are not in this repository and still are not.** They live in
pinned releases (`payload-qwen3-0.6b-v1`, `payload-q35-2b-v1`), which CI
downloads. Not the repo, which would carry 600 MB to 1.9 GB of binary forever;
not LFS, whose free bandwidth this exhausts in a few builds; not the Supabase
bucket the images go to, because that egress is metered and public release
assets are not.

`tools/payload.py` records sizes and digests into `payload/*.txt` and verifies
after the download. That step is the reason the rest exists: a truncated
transfer produces an ISO that builds, boots, and then cannot load the model,
because nothing else in the build knows how long `model.bin` should be. Nine
claims, and the two that earn their place are truncation and a *same-length*
corruption, which size alone waves through.

**Context is a four-byte stamp, not a conversion.** `--seq` changes no weight;
it sets `seq_len` in the header and the kernel sizes its KV cache from that.
The three 2B files in `out/` differ in exactly one byte. So N context variants
cost one upload and N stamps: `tools/ctxstamp.py` writes the i32 at offset 36
-- where `convert.py` packs it and `v4.py` reads it -- and reads the result
back *through `v4.py`*, the reader that is deliberately not the writer.
Stamping `q35-2b.bin` from 512 to 32768 produces a file byte-identical to a
full `convert.py --seq 32768` run, which is how that claim was settled.

**Verify runs before the stamp**, and that order is not cosmetic: stamping
changes four bytes, so a digest taken afterwards could never match.

Two figures worth knowing before adding a payload. The 2B ISO is 1.90 GB
against GitHub's **2 GB per-asset limit**, so about 100 MB of headroom and a
larger model does not fit this route. And `HEAP_LADDER`'s 320 MiB is the first
*contiguous* region rather than the heap -- boot reports `heap 320 MiB` then
`+1020 MiB across 1 more regions (1340 MiB total)` -- while the kernel holds
the KV cache **int8** where `convert.py` reports it f32. Both of those were got
wrong first: the arithmetic said the 2B could not run at 32k, and measurement
said it uses 347 MiB of 1341 and boots.

The server side is `supabase/` and `.github/workflows/{release,experimental}.yml`.
`supabase/README.md` has the setup. Two things from it worth knowing here: the
Edge Function's P-256 signer is plain JavaScript so `node` can check it against
`tools/manifest.py`, and both workflows publish the image **before** the
manifest, because a manifest naming an object that is not there yet is a window
of 404s for an update every machine was just told about.

**Testing it under QEMU needs a real disk, not VVFAT.** `-drive file=fat:rw:`
projects a host directory as a synthetic FAT16 volume; its read-write mode can
change the contents of a file that was there at boot and cannot do directory
operations at all. The guest read `UPDATE.FLG` correctly and could not delete
it, and "the firmware will not write the ESP" was being recorded as the reason
without anybody having checked which half was true. `tools/mkesp.py` builds a
raw FAT32 image with an MBR and an 0xEF partition, using the same
`mkiso.build_fat` writer whose output the firmware already boots:

```powershell
.\tools\venv\Scripts\python.exe tools\drive.py --esp-image .qemu/esp.img --esp-rebuild ...
.\tools\venv\Scripts\python.exe tools\drive.py --esp-image .qemu/esp.img ...   # reuse; guest writes persist
```

Reuse is the point: the image is a disk, so what the guest wrote is still there
next boot, which is what makes the apply/trial/settle sequence observable.
`--esp-rebuild` starts clean and discards it.

The whole flow has been driven this way, with a throwaway key and two builds a
version apart:

    boot 1 (0.1.0)  signature signed by the update key
                    copying the running image aside
                    writing the new boot image
                    applied -- reboot to run it
    boot 2 (0.1.1)  this image is on trial          <- the swap landed
    boot 3 (0.1.1)  silent                          <- the trial settled

Three real bugs came out of that and none was reachable any other way: the hook
runs before `init_heap`, so verifying a signature died at `memory allocation of
32 bytes failed` (there is a static early arena now); the staged image was read
and verified on *every* boot including trials, which is 2.8 MB and an ECDSA
verification to answer a question already settled; and `ResetSystem` through the
runtime table, called while boot services are still up, left the machine silent
-- so the hook no longer reboots and the swap simply takes effect next boot.

**What the health flag can and cannot catch.** The window it can write in ends
at `ExitBootServices`, so the question it asks is whether the new image got
from the firmware handoff to just before the memory map -- covering a binary
that will not run, an early fault, and a model or tokenizer that will not
load. It does not cover a failure after that line, because there is no
filesystem left to record one in; and it cannot cover an image that faults
before reaching the hook, because nothing on the machine gets a turn. The
recovery for that is the USB stick, and no software scheme can do better.

The ordering is the design and is argued in the module. The short version: the
rollback copy is taken and *read back* before anything is overwritten, the flag
is cleared before the window rather than after, the written image is verified
by digest, and a mismatch puts `BOOTX64.OLD` straight back. `decide()` is a
pure function, so all eight of its states are asserted at boot without staging
anything -- including that an image already on trial refuses to apply a further
update on top of itself.

### Ported programs, and the seam they reach through

`src/port/` is everything a program written somewhere else may ask of this
machine: an indexed `Surface` with its own palette, held keys, **relative
pointer motion**, a monotonic clock, and the bytes of a file. **Anything under a ported tree may name
`crate::port` and nothing else**, and that is checked rather than intended:

```powershell
.\tools\venv\Scripts\python.exe tools\portcheck.py
```

It scans for `crate::x` where x is not `port`, and for `super::super::`, which
is what somebody writes ten minutes after being told about the first. A line
with a genuine exception carries `# portcheck: ok` so the exception is in the
diff rather than achieved by rewording. There is no `build.rs` here and there
cannot be one, so this runs beside the build the way `tokenizer.py --verify`
does.

The reason for a seam with one consumer, stated because this tree normally
refuses to build an interface before there are two: the point of the first
port is to find out where the boundary is. A port that reaches into `gfx`,
`kbd`, `sysbox` and `time` wherever it needs them is not a port, it is a
merge, and the second one starts from nothing.

**The pictures are ported, not redrawn.** `src/doom/pic.rs` is room4doom's
patch and TEXTURE1/PNAMES decoder, and one line of it is the reason it was
brought over rather than written from the format description:

```rust
if y <= top { top += y } else { top = y }
```

A post's `topdelta` is normally the absolute row it starts at. But a patch
taller than 254 cannot say row 300 in a byte, so the convention -- DeePsea's,
and universal since -- is that a delta which does **not rise** above the
previous one is relative to it. Every published description of the format
predates that and says the field is simply the row. Writing this from the
specification gives a decoder correct on every patch in DOOM and wrong on half
the patches in anything made after 1997, which is the worst kind of wrong: it
works until it does not, on somebody else's data. `diag doom` asserts exactly
that case, both directions.

**A texture pixel is an index, not a colour**, so it cannot be darkened by
arithmetic -- an index scaled by 0.7 is an unrelated colour. The only way to
shade indexed art is to remap it, and the table that says how is COLORMAP.
`Art::lighting_colormap` prefers the WAD's own, and checks that it *lights*
before using it: an identity table is a legal lump, and a renderer trusting one
draws every wall at full brightness at every distance, which reads exactly like
a lighting bug in the renderer. `tools/mkwad.py` shipped an identity table for
a while on the reasoning that a flat picture is obviously flat; it was not, and
it emits a real one now. Where there is no usable table, one is built by asking
the palette for the nearest match to each colour dimmed.

**Test it against FreeDoom, and that is not optional any more.** The generated
WAD is a fixture with two textures in it; a real IWAD has a thousand, and the
difference found things nothing else could:

```powershell
# 24 MB, freely licensed, and the release carries a signed CHECKSUM -- verify
# the SHA-256 against it rather than trusting the transfer.
.\tools\venv\Scripts\python.exe tools\drive.py --wad out\freedoom\freedoom-0.13.0\freedoom1.wad `
  --qemu-extra "-accel whpx -cpu max -smp 4" --timeout 600 `
  "initiative off" "agent stop" "doom view 0 900000"
```

The WAD is **not in this repository** and must not be: it is 28.8 MB of
somebody else's art, and the same rule that keeps `DOOM1.WAD` out keeps this
out. `out/freedoom/` is where it lands.

What the real file exercises that the fixture structurally cannot: **14
palettes** against one, **963 textures** so `TEXTURE2` is read at all,
**1,049 patch names**, **`F1_START` nested immediately inside `F_START`** --
which is the whole reason `pic::classify` counts depth instead of matching one
spelling -- **654 two-sided linedefs** against one, **681 BSP nodes** against
one, sector heights spanning 704 units against a flat 128, and `F_SKY1`
ceilings, which nothing in the fixture has and which the flat reader has to
decline to draw so the cleared sky shows through.

Every count the kernel printed matched an independent host-side parse exactly,
including the thing census decomposing to 292. What that decomposition *was*
is the reason the sprite table stopped being hand-written:

    before   179 drawn, 49 of a doomednum the table does not carry,
             52 monsters with no rotation-0 lump, 12 invisible
    after    280 drawn, 12 start(s), 0 of unknown kind, 0 with no picture

**The sprite table is generated, and the hand-written one was the problem.**
It carried 44 doomednums, on the argument that copying `mobjinfo` would be
copying a game's content. The argument was sound and the conclusion was wrong:
a doomednum means what id decided it means, so a *partial* table is not a
smaller version of the right answer, it is a level with 49 things missing from
it. `tools/doominfo.py` emits `src/doom/info.rs` -- 967 states, 137 kinds, 118
doomednums -- and the third figure above is the one worth watching, because it
would rise again the moment somebody loaded a PWAD with custom things in it,
which is the honest answer rather than a bug.

The 12 remaining are exactly the four player starts and eight deathmatch
starts. Those are **placeholders and not things**: positions the map format
defines, with no `mobjinfo` row, from which nothing ever spawns. A teleport
destination looks like it belongs with them and does not -- it is a real
object with a real doomednum whose whole job is to be a marker, and what makes
it invisible is that its row spawns into `S_NULL`. `S_NULL` carries `sprite:
TROO, frame: 0` like every other row because the array needs *something*
there, so a reader trusting the fields would draw an imp on every teleport pad
in the game.

Three generator mistakes, each recorded where it was fixed, and the pattern in
them is the useful part -- **the two that would not compile were the cheap
ones**:

- The flag names were hand-written beside the parsed table. `NotDeathmatch` is
  spelled `Notdmatch` upstream, so the constant emitted and the constant
  referred to were different identifiers, which is a build failure. But
  `Translation` is `0xC000000`, a two-bit colour field and not a bit at all,
  so a positional list assigning it bit 26 produced a constant that compiled
  perfectly and meant something else. Names *and* values are read out of
  upstream's `bitflags` block now, and every flag the table names is checked
  against the declaration.
- `speed` is units per tic for a monster and **fixed point** for a missile,
  one field with two units in it. A rocket reads 655360 where an imp reads 8,
  which does not fit an `i16` -- the only reason anybody noticed.
- `mass` is a divisor in the damage thrust, so Commander Keen and the boss
  brain carry ten million to mean *immovable*. Also not an `i16`, for a
  completely unrelated reason.

**A thing has a state of its own, and it did not at first.** The first version
played a kind's spawn cycle as a pure function of the world tic, which draws
every barrel on the map correctly and is right *only* while every object of a
kind stays in phase with every other. That holds while nothing joins the level
late and nothing leaves its cycle, and it stops holding the moment anything
can be hurt -- a monster in pain is a monster whose animation no longer agrees
with its neighbour's. So `thing.rs` carries `Obj` with a state pointer and its
own countdown, `sprite.rs` keys the decoded pictures by **state** rather than
by kind, and the renderer asks the object which state it is in. On E1M1 that
is 44 kinds over 66 states, of which 13 animate.

Two details of `P_SetMobjState` worth knowing before touching it. A state with
`tics == 0` is not a state that shows for no time -- it runs its action and
falls straight through to the next one, which is how DOOM writes logic into an
animation, so setting a state is a loop rather than an assignment. And that
loop is **bounded here where upstream's is not**: id could rely on the shipped
table having no zero-tic cycle, but this table is *generated*, there is no
unwinder in this kernel and no watchdog, so an unbounded walk over a bad chain
is a machine that stops with no message at all.

**There is an inventory, and the pickups that can refuse are the half worth
having.** `src/doom/player.rs` carries health, armour, ammunition, keys and
which weapons are owned. The table is keyed by **sprite name**, which is id's
own choice and looks like a mistake until the reason lands: a dropped weapon
and a placed one are different objects with different doomednums and the same
sprite, and "walking over `SHOT` gives you a shotgun" is true of both. A
medikit at full health stays on the floor; a pocket at 200 bullets leaves the
clip; a shotgun already owned is still taken, for the shells. A pickup that
always disappears cannot tell a working rule from `return true`.

Specials 26/27/28 and 32/33/34 are handled, which closed the last gap on
FreeDoom E1M1: **8 distinct specials on 42 lines, 8 handled, 0 missing**, where
it was 7 and 4 lines of blue door refused for want of an inventory to check.
The colour is read off the original special number and the check runs *before*
anything moves -- a lock tested after the door had been started would open it
and then report that it had not. A refused door is also not **spent**: clearing
a once-only special on a refusal would make the door forget it was ever a door
and refuse forever with the key in hand.

**Shooting is a hitscan, and two deviations are named rather than left to be
found.** No vertical aim: DOOM's shot carries a slope and a two-sided line
stops it when the *opening* does not admit it, where here a shot is level. No
pellet spread, so the shotgun's seven-bullet cone does not exist. And boxes
rather than circles, twice -- a thing is hit when the ray crosses the *square*
of its radius and a blast falls off by `max(|dx|,|dy|)` less the radius, both
of which are DOOM's and both the same square the pickup test uses.

**The rate of fire was predicted wrong and measured right, which is the whole
argument for measuring.** The obvious reading is that a weapon fires once per
attack chain, so a pistol's 19 tics would be 1.8 shots a second. A run holding
the trigger for three seconds fired **eight** shots, not five. `A_ReFire` runs
on *entry* to its state and restarts the chain there, so that state never
spends its own tics while the trigger is down: the real cycle is `4+6+4 = 14`
tics, which is 2.5 shots a second, which is exactly what DOOM's pistol does.
`weapon::held_cycle` computes it and the claim asserts 14 -- the number is
never written down, so a table that changed would fail there rather than
quietly changing how the game plays.

Only the **pistol and chaingun** actually fire; both are one bullet from one
clip round, which is all the hitscan can do. The rest animate correctly and hit
nothing, and `Psprite::armed` says which is which. A shotgun wired to fire a
single bullet would be a bug that looks like a balance decision.

**Monsters look, chase and shoot.** `src/doom/enemy.rs` is DOOM's own AI, and
the shape of it is the part worth knowing: a monster does not steer. It picks
one of **eight** compass directions, walks it for a random number of tics, and
picks again when blocked -- trying the direct route, then the two cardinal
components, then the way it was already going, then everything else in a
randomly chosen order, refusing to turn straight around unless nothing works.
That is why DOOM's monsters catch on doorframes and take corners in two moves.
Steering them with a heading would look smoother and would not be DOOM.

Three things are deliberately absent and each is named in the file. **No
infighting**: upstream's `target` is a pointer because a monster hit by another
turns on it, and here it is a `bool`, because there is one player and monsters
cannot hurt each other. **No projectiles**: an imp's fireball is a thing with
momentum and nothing in this port moves under its own power, so the imp chases
and claws and never throws -- easier than it should be rather than strange.
And **no sound**, which for once changes behaviour rather than being quiet:
DOOM wakes monsters by flooding the noise of a shot through connected sectors,
so `alert` keeps the flood and drops the sound, waking whatever is in the
player's sector or adjoining it. Without it a monster facing away is deaf.

**The random table is DOOM's, and it is what makes a run repeatable.** It was
deferred twice with the note that inventing a sequence would produce a game
that plays differently from every other copy. The larger reason turned out to
be the harness: `rng::reset` at the top of every run means two runs of one
script take the same path, and `spent()` reports how much of the table was
used, so two runs that *disagree* about that number have diverged before
anything else shows it. Measured: two identical scripts spent 17 draws each and
reported the same damage to the byte; a third that also fired spent 48.

That number is why a coincidence did not become a claim. Three runs each
reported exactly **27 damage taken**, which looked like a cap. It was not:
6 seconds gives 27, 15 gives 57, 30 gives 114 and kills the player. Two short
runs happened to land on the same total, and the roll counter is what made it
cheap to find out rather than plausible to assume.

Measured, on the fixture and on FreeDoom E1M1:

    fixture   1 monster, 1 awake, nearest 42 (from 283), 27 damage in 6 s
    E1M1      53 monsters, 126 damage in 10 s walking forward, PLAYER DIED

The fixture's zombieman is placed *pointed at* the player, and that is not
decoration: `A_Look` wants sight **and** the player inside its front 180
degrees, so a monster facing the wall never notices anybody -- which is exactly
what a broken `A_Look` looks like too. `mkwad.py --verify` asserts the
placement, the line of sight, and that every one of the 21 frames its state
chains can reach exists, because a monster missing its death frames dies into
nothing and one missing its walk frames vanishes the moment it notices you.

**Removal is deferred to the end of a tic**, and it had to be. Actions are
dispatched by index, and the first version removed objects as it swept -- so an
index handed out early in a tic named a different object by the end of it, and
the action most worth dispatching is fired by something on its way off the
level. `Objs::tick` marks, the caller dispatches, `sweep` takes them. DOOM
removes its thinkers at the end of a tic for the same reason.

**`port::mouse` is the fifth thing in the seam and the first added because a
port asked.** It is relative and unclamped, which is the whole reason it
exists: `dev::mouse` tracks a cursor, so its position stops at the edge of the
screen -- right for a pointer, fatal for a player, who would turn until they
faced the right-hand wall and then stop. The accumulator is fed inside `apply`,
the one point PS/2 and USB HID already converge on. `MOUSE_TURN` is derived and
not chosen: `G_BuildTiccmd` does `angleturn -= mousex * 8` and `angleturn`'s
unit is 1/65536 of a turn, so eight of them is 0.0439 degrees. Measured:
`mouse=400` turned the player from 180 degrees to 162, against 17.6 predicted.

A `mouse` verb exists for the same reason `win keys` does -- serial cannot
inject PS/2 packets -- and **the play script can release a key** now
(`fire@200 -fire@400 fire@600`). It had to learn to the moment anything was
edge-triggered: firing twice needs the trigger let go in between, and no
held-key model can say that.

**The weapon is drawn from the patch's own offsets and nothing else.** On a
320 by 200 view DOOM's `R_DrawPSprite` collapses: the screen centre it adds and
the 160 it subtracts cancel, leaving `x = sx - left` and `y = sy - top`. That
is why a pistol carries a left offset of -125 and a top of -97 -- those numbers
are not a nudge, they are the whole of the placement, and a reader that ignored
them would draw the gun in a corner. Verified by looking, which is the only
thing that settles it.

**Which frame is showing gets a number, not a screenshot.** `doom play`
reports how many times a sprite changed picture, and non-zero is the whole
claim: every part of this can be right -- the chains walked, the durations
read, the arithmetic asserted at boot -- and the tic never reach the objects,
which then show their spawn frame forever and look exactly like a level whose
barrels happen not to move. Measured on E1M1: `140 tics, 13 of 44 kind(s)
animate, 58 frame change(s)`.

**A claim made from a glance at that frame was wrong, and is recorded here
because it is the exact error this file spends pages warning about.** A dark
region left of centre on E1M1 was read as the per-column plane shortcut
leaving a hole, and asserted as one -- into a release note -- without being
measured. It is distant geometry shaded almost to black: the pixels are
`(0,0,0)` and greys, the sky colour is `(0,0,23)`, and the count of
sky-coloured pixels is *identical* before and after visplanes on both maps.
Nothing was missing.

The limitation is real in the code and visplanes do fix it; what does not
exist is a picture of it happening. Judge that change on its differential
instead: the test map renders **pixel-identical** through the span path, and
E1M1 differs in 0.8% of pixels, all small shifts from stepping the texture
coordinate along a span rather than recomputing it per pixel.

**What is ported is the reader, and what is generated is the art. They are
easy to confuse and worth separating once, plainly.** Ported: the patch
column-and-post decoder, the TEXTURE1/PNAMES composite reader, the flat and
sprite namespaces, and the BSP wall algorithm. Generated by `tools/mkwad.py`:
the palette, the COLORMAP, every patch, every flat, every sprite and the map
itself. **No byte of id's data is in this repository**, and none ever will be.

This paragraph used to end "the renderer has never once been run against real
DOOM art ... and 'should' is carrying real weight in that sentence, because
nothing has tested it." That was true when written and stopped being true the
day FreeDoom was fetched, and it sat here contradicting the section above it
for a while afterwards -- which is the ordinary way a file like this goes
wrong. It needed no change, as the recipe above records.

The generated test WAD carries real column-and-post patches with real
transparency. The *art* in them is generated -- id's is not ours to ship -- but
the patterns are chosen so a decoding mistake reads as an obviously wrong
picture rather than as slightly odd art: a bright marker in the top-left 8x8
(a flip moves it), mortar courses with joins offset course by course (a
transposed column smears), a vertical gradient (a reversed `v` is visible even
where the bricks line up), and a hole, because a solid patch never exercises
the post loop at all -- a decoder that ignored posts and copied `width *
height` bytes would pass on one. `doom tex [name]` composes one and blows it up
to look at, which is the only check that settles a picture decoder.

**Floors and ceilings go through visplanes**, which is DOOM's own structure and
was not the first shape here. They were drawn column-wise at the moment a wall
claimed a column, which is a dozen lines against several hundred and cost a
divide per pixel -- affordable on this machine where it was not on a 486. What
that shape cannot do is fill a column no seg ever claims, because the fact
needed is *which columns show this floor* and no column knows it until the walk
is over. So the walk records into a plane keyed by height, picture and light,
and the drawing happens afterwards.

Two pieces of it are worth knowing before touching either. `find_plane` shares
one plane between every surface agreeing on all three keys, which is the whole
economy of the idea -- one floor seen through four doorways is one plane. And
`check_plane` forks a second plane with *identical* keys when a claim overlaps
columns already marked, because a plane holds one top and bottom per column and
a sector's floor can appear either side of a pillar; without the fork the
second view silently overwrites the first and loses its floor.

**A flat is 64 by 64 and carries no header at all**, so its size *is* the
format and the only way to know one is the `F_START`/`F_END` namespace it sits
in. `pic::classify` is that rule as one function, because it is four conditions
that all have to hold at once and none fails loudly -- the one that matters
most is that a marker must begin with `F`, since `S_START` brackets sprites, a
sprite is a patch, and any patch that happened to be 4096 bytes would be
adopted as a floor by a rule that only looked for `_START`. Flats are
*borrowed* from the WAD rather than copied, the opposite of the trade `Pics`
makes and for the opposite reason: a wall texture is composed from patches and
has to be built somewhere, while a flat is already exactly the bytes a renderer
wants. `doom flat [name]` blows one up, and it catches two failures `doom tex`
cannot -- a flat is addressed by world position, so its coordinates can be
transposed or one axis mirrored (DOOM negates world y for the row), and neither
shows on a symmetric picture. `mkwad.py --verify` refuses to emit a flat that
is symmetric under either.

**A full-screen program's own launch keystroke used to close it.** `wait_or`
polled the keyboard ring without draining it first, so the Enter that ran
`doom view` was still queued when the first frame landed and the picture was
gone before anybody saw it -- which reads as a program that drew nothing rather
than one that exited. It showed as three screenshot runs in five coming back
with the desktop already restored, and it is a defect on the keyboard too, not
an artefact of driving this over a serial line.

`Surface` applies the palette in software, because the framebuffer is 32bpp
only and there is no mode-setting -- the resolution and format are whatever
UEFI handed over. It is cheap anyway: the palette is pre-encoded once into the
screen's word order, a row is expanded into a `u32` scratch and blitted whole
through `blit_span` (a `copy_nonoverlapping` on the aperture), and integer
scaling repeats that row rather than rebuilding it. `gfx/paint.rs` has the
other approach -- run-length plus a `rect` per run -- which is right for a
drawing program and wrong for a rendered scene where every pixel differs from
its neighbour.

### Testing

There is no `cargo test`. This is a `no_std` UEFI binary with no host test
runner, so **verification is the boot selftests plus driving QEMU.**

At boot the system runs **twenty-six selftest sections**, and `diag` offers
**thirty-three named suites** on demand, most of them the same checks (the `aiksi` section covers the capability gate by name and never by
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

**`diag` on its own lists the suites; `diag all` runs them.** A bare `diag`
prints a table with `-` beside everything that has not run this boot and a
tally reading `0 passed, 0 failed, 33 not run`, which is easy to read as a
clean sweep. It is the opposite of one.

**The list and its verdict table are one number now, and were not.** `RESULTS`
is a fixed array indexed by a suite's position, and the `const` assertion
guarding it compared `SUITES.len()` against a *literal* while the array's
length was a separate literal beside it. So adding the thirty-third suite
passed the guard and then panicked at the store -- `index out of bounds: the
len is 32 but the index is 32` -- which is exactly the failure the guard's own
comment says it prevents. `SLOTS` is the one number; a `static` cannot be read
in a `const` context, so naming its length is as close as this gets to
measuring the array directly.

**The tooling gives the guest four cores, and two suites need more than one.**
`drive.py` and `run.ps1` pass `-smp 4` by default; `--qemu-extra "-smp N"` and
`run.ps1 -Smp N` override it. QEMU's own default is one vCPU, and with one
`diag mt` and `diag migrate` cannot pass at all -- "the allocator was exercised
from several cores" and "a task carried onto another core and back" are false
statements about a machine with one core, so both printed `FAILED` on every
clean boot this project had ever driven. A check that always fails is read as
one nobody has to look at, which is the objection `smp.rs` makes about its own
canary.

**`video bench` is only comparable against a run with the same text on
screen.** `console redraw_all` skips blank cells, so its cost scales with how
much output is sitting in the terminal: the same build measured 497 us after a
bare boot and 823 us after `diag all` had filled the console, which reads as a
40% regression and is a full scrollback. Take the before and after with the
identical command prefix, and read `full-screen rect` as the control -- nothing
above the framebuffer can touch it, so what it moves by is the noise floor.
Between boots on the development machine that is about 10%.

**Run `video bench` at `-smp 1`.** The extra cores cost the graphics path 30
to 40% while doing nothing at all: `desk::draw + present` measures 1,541 us at
one core and 2,107 at four, `full-screen rect` 158 against 221, `present, no
change` 270 against 358. Those three figures **predate the Frutiger Aero
reskin** and are not the number to compare a change against: the same command
on the same prefix now reads 2,143 us at one core, which is what gradients,
rounded corners and shadows cost. Take a matched baseline by stashing rather
than reaching for a figure written down here -- a session comparing the
terminal status strip against 1,541 read a 41% regression that was entirely
the reskin, and the strip itself measured 1.5% on a stashed pair, against a
control that moved 4% the same way. That is the same contention `smp bench` records --
one core reads 4570 MB/s alone and 3526 MB/s with seven merely idling beside
it -- and it lands here because the whole graphics path is span fills and a
memcmp, which is to say memory bandwidth and nothing else. The figures
elsewhere in this file predate the `-smp` default and are one-core figures;
comparing a four-core run against them reads as a renderer that regressed by a
third.

Four rather than two, because two leaves a single contender for the chunk
cursor and the bug `smp.rs` records there needs several. It costs nothing on
the decode path, which is what was measured when the default was chosen:
best of nine decodes on SmolLM2 under WHPX read 50,819 us/token at one core,
49,463 at two and 50,818 at four, and `logits 7 11 3` is bit-identical across
all three. The single-sample figures that suggested a cost (65 ms against
95 ms) were the host's scheduler, which is the error `video bench` was
rewritten to stop making.

**Two `drive.py` runs at once is a failure that looks like a hung guest.**
The serial and monitor ports are fixed at 45454 and 45455, so the second
launch gets neither, and what it prints is a log with **no boot output at all**
followed by `TIMEOUT after Ns with N commands unsent` -- which reads exactly
like a guest that died early. The tell is the empty log: a real hang prints
the firmware banner and the boot sequence first. `.qemu/qemu-stderr.log` says
`Failed to find an available port`. Check for a running QEMU before launching,
especially when the first run is in the background.

**A screenshot is taken when `drive.py` exits, which for a full-screen program
means you get the desktop.** The bounded `ms` form returns before the harness
does, so `--screenshot` catches whatever is on screen *after* the program gave
the screen back -- a terminal, every time. To photograph the program itself,
give it a duration longer than the harness will wait and let the **timeout** be
the exit path: `doom play 600000` with `--timeout 420` lands about two minutes
into the run. Boot alone is around 300 s against FreeDoom under WHPX, so a
timeout under that photographs the boot log instead. Two runs and fifteen
minutes were spent learning this.

**A full-screen program needs a bounded form or nothing can test it.**
`drive.py` sends the next command when it sees a prompt, so a program that
owns the screen until somebody presses a key never gives the prompt back --
and the keystroke that would end it is the one command the harness cannot
deliver. `port bars` deadlocked exactly there on its first run, with two
commands unsent. Hence `port bars <ms>`: the interactive form waits for a key,
the bounded form returns on its own, and only the second one is ever driven.
Anything full-screen that follows needs the same.

**`gfx::exclusive` is how a full-screen program keeps the screen.** Owning the
framebuffer is not enough: `desk::paint_clock` runs on the clock task at 10 Hz
and `desk::move_cursor` runs on whichever task is generating, and both write
the framebuffer under a paint claim that is private to `desk.rs` and therefore
unreachable from outside it. So instead of contending for the claim, the two
periodic painters stand down while the flag is set. `port::with_screen` sets
it, blanks the screen and calls `desk::draw()` on the way out. `edit::run` has
owned the screen the same way since it was written and had this defect the
whole time -- the clock painted over the editor.

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

**Records, types and `use`.**

```
use "/lib/text"

rec Host { name: str, port: int }

fn reachable(h: Host): int {
  if (tcp_connect(h.name, h.port, 600)) { tcp_close() return 1 }
  return 0
}
```

A record is a declaration and a constructor in one: the name becomes callable
with the fields in order, so the constructor's arity is the declaration's by
construction rather than by agreement. Records are **values**, like lists --
`b = a` copies, and `a.x = 9` afterwards leaves `b` alone. That is why nothing
in this language has to explain aliasing. It is also why `a.b.c = 1` is refused
rather than silently discarded: there is no shared object to reach through, so
only a plain variable can be assigned back to.

Types are **optional and never inferred**. Absent means `any`, so every
application written before they existed still means what it meant. They are
checked where a value crosses a boundary somebody annotated -- a call, a
return, a record field at construction and at assignment. Inference would mean
a solver; the thing worth having is much smaller, which is that a model passing
a string where a number belongs gets `f wants int for 'a', got str` instead of
`int()` quietly answering 0 and the wrong number appearing four calls later.

`use` is textual inclusion that happens once, not a module system. There is
nothing to qualify against, and inventing a prefix would mean inventing a
spelling and then explaining it. **The imported program runs with the
importer's capabilities**, which is the security property: caps live on the
interpreter and there is one interpreter, so an import can never be an
escalation. The jail on it -- a sandboxed program may `use` only its own files
or `/lib` -- is therefore about legibility, not safety: it keeps a stored
program's dependencies somewhere a person can find them. Cycles terminate
because a path is marked imported *before* it is evaluated, and running out of
stack in ring 0 with no guard page is a triple fault rather than an error
message.

`eval::KERNEL_RECS` declares the record types the kernel itself returns, and
they are registered in every interpreter at construction so an annotation
checks against something real. `pci_list` answers a list of `Device` rather
than lines of text -- it answered text only because there was nowhere to put a
field, and every caller then wrote the same fragile `split` to take it apart. A
program may not redeclare one of these: a builtin would go on returning the
kernel's shape while every annotation checked a different one of the same name.

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

Everything that is *actually a struct* answers a record: `pci_list` gives
`Device`, `rtc_now` gives `Time`, `net_ifaces` gives `Iface`, plus `net_config`,
`mem_stats`, `task_list`, `stat` and `tcp_status`. `ls` answers a list of names
rather than newline-joined text.

Atomic answers stayed atomic. `mem_used()` is not improved by becoming
`mem_stats().used`, and converting scalars into records to be uniform would make
the language worse to make a rule tidy. The test is whether a caller would
otherwise re-parse: `substr(rtc_now(), 11, 2)` to get an hour, or a
character-by-character line counter to count what `ls` reported -- the seeded
`/ai/tools/count` tool contained exactly that, and now calls `len(ls(path))`.

`kernel::rec` builds one by name and checks it against `KERNEL_RECS`, so an arm
that adds a field without adding it to the shape, or gets the order wrong,
fails there rather than handing back an `Iface` whose `.ip` is its netmask --
a mistake invisible at a glance, because both are strings.

Two shapes answer `nil` rather than a filled-in record: `rtc_now` when the clock
cannot be read, and `stat` on a path that does not exist. A record whose fields
all read as "absent" is indistinguishable from a real empty one, and a program
checking existence would have to know which field to trust.

Three bounds worth knowing before changing anything there. `range` and `repeat`
are capped at 65,536 because they are the easiest way for a generated program
to ask for a billion-element list, and this kernel has no OOM killer and one
address space, so the step budget never sees the single call that takes the
heap. Socket timeouts are clamped to 30 s because an unbounded one in a repaint
path hangs the desktop and the step budget cannot see a blocking call. And
`app::document` takes `with_step_budget(DRAW_BUDGET)` rather than the full one
-- though **not** because it runs per repaint, which its own comment claimed
for a long time. `desk::refresh_routed` is the only caller and rebuilds a
window's panel after a command runs, since a command is the only thing that
changes what a route would produce; `draw` paints the stored panel. The bound
is right and the reason was wrong.

**What a step costs, measured at last.** `core bench` is the first wall-clock
calibration of Aiksi in this project's history, best of nine like `video
bench`. Every step budget in the tree was a number chosen by comparison to
another number, and nobody had ever timed one:

    one step                    13.7 ns      (~37 cycles at 2.67 GHz)
    Interp::new()               2,132 ns

    VOTE_BUDGET  20k    274 us      DRAW_BUDGET 200k    2 ms
    SKILL_BUDGET  5M     68 ms      STEP_BUDGET  20M    274 ms

So the budgets are sane as safety bounds, and `voter::Core::vote` -- the
genuinely hot path, on every routing decision -- spends **20 steps against a
20,000 ceiling**, a tenth of a percent of what it is allowed.

The same command breaks one vote into its parts, and the result contradicted
a confident prediction written into the plan that asked for the measurement:

    build interpreter    1,757 ns     31%
    arm (run top level)    623 ns     11%
    call vote            3,244 ns     57%
    total                5,658 ns

The walk looks dominant and is not. Twenty steps at 13.7 ns is about 275 ns of
dispatch -- **8% of that call and 5% of the vote**. The rest is work no code
generator removes: `lower` allocating a fresh string, `contains` scanning it,
and the argument string plus a 23-element list cloned into the frame. Against
that, the 2,380 ns of setup paid per decision is **8x larger than the entire
tree-walk**, and it is setup for a program whose whole body is one assignment
and three `if (contains(t, "word")) { return N }`.

That is why F1 comes before any compiler, and why a compiler for this tree
cannot be justified as an optimisation of this path. The plan's own text said
so in advance -- "unless F0 shows the tree-walk itself is not the cost" -- and
F0 showed exactly that.

**The setup was `KERNEL_RECS`, and it is gone.** `Interp::new` copied the
eight kernel record shapes into the program's own `recs` map at every
construction: 49 `String` allocations and eight tree inserts to reproduce
immutable kernel data, identical in every interpreter that has ever existed.
Lookup consults two tables now (`fields_of`), which is safe because
`Stmt::Rec` already refuses a name the kernel returns -- a guard that was
tidiness when both lived in one map and is load-bearing now that they are
two.

    Interp::new()      2,132 ns -> 22 ns          one vote  5,658 -> 3,343 ns
    build interpreter  1,757 ns -> 42 ns          setup       42% -> 19%

The freshness the doc in `voter.rs` argues for is untouched, because only
*program* state ever had to be fresh. What was being rebuilt per vote was the
kernel's half.

**The second copy was the function itself.** `funcs.get(name).cloned()` deep
-copied a `Func` -- its whole statement tree, and every expression tree under
it -- at every user function call, on a structure that is immutable from the
moment `Stmt::Fn` declared it. `funcs` holds `Rc<Func>` now, so a call is a
refcount bump. `Rc` and not `Arc`: one interpreter per call chain, nothing
crossing a task, the same single-core assumption `Racy` rests on.

    call vote  -25%       arm  unchanged       one vote  ~2.9 us

Judged against the control rather than on the raw figures, and the trade is
worth stating because it is a trade: one `Rc::new` is *added* per declaration
to remove about nineteen allocations per call. The vote is the worst case for
it -- a seven-statement function called once -- and it still wins, because the
saving scales with the body and the cost does not.

**And the declaration itself only had to happen once.** `Core::vote` ran the
core's top level on every routing decision to register a function that had
been the same function since the core was parsed. `Interp::prepare` runs a
declarative top level once and `adopt` seeds a fresh interpreter from it.

    arm  813 ns -> ~75 ns  (-92%)          one vote  -20%

Two conditions make that a saving rather than a semantic change, and both are
in the code as the reason:

- `is_declarative` is the whole gate. A declaration's only effect is to
  register itself, so running such a top level once and copying the result is
  indistinguishable from running it again. An assignment, a call that could
  ask the clock, a `use` that lexes another file -- any of those would be
  frozen at prepare time, so those programs keep arming. Nothing `compose`
  writes is one, but an operator could install one.
- `Prepared` carries the top level's **step count**. `steps` is what the
  budget stops and what a verdict records, so a prepared run that skipped
  those ticks would answer identically and report itself cheaper -- two paths
  through one program disagreeing about a number the judges read. The
  selftest compares prepared against armed on value *and* cost, which is the
  differential check in miniature and the only reason this is allowed on a
  path a judge reads.

Cumulatively, a vote went from about 418 step-equivalents to 146: **2.9x**,
none of it from compiling anything. Every one of the three wins was
allocation, and the two the plan predicted -- that the tree-walk was the cost
and that construction was near-certainly the fix -- were both wrong in the
same direction.

**Running code from the heap.** `src/cpu/code.rs` is the substrate a code
generator would need, and it works: `diag code` writes seven bytes into a
page-aligned buffer, serialises, and calls them. Every page in this kernel is
`PRESENT | WRITABLE` and executable -- there is no NX constant in
`mem::paging` and EFER.NXE is never enabled -- so getting somewhere to run
from is a `Layout::from_size_align(n, 4096)` and no page-table work at all.
That is exactly why the rest is careful.

- **`cpu::serialize()`.** Nothing in this tree serialised before it: no
  `wbinvd`, `clflush`, `mfence` or `cpuid`-as-barrier anywhere. `CPUID` does
  both halves -- the processor drops what it prefetched, and since `cpuid`'s
  `asm!` declares neither `nomem` nor `readonly` the compiler cannot sink the
  stores that filled the buffer past it.
- **`Compiled` is the ABI, declared once.** `unsafe extern "sysv64" fn(u64)
  -> u64`, so no call site spells the convention. `extern "C"` here is
  Microsoft x64; `task.rs` records what that cost when it was got wrong. The
  selftest's stub is `mov rax, rdi / add rax, rax / ret` called with 21,
  because a no-argument stub passes under either convention and proves
  nothing -- both agree on a bare return in rax. Only the argument tells them
  apart.
- **The registry, and a fault reporter that stops lying.** `fault()` printed
  `rip - IMAGE_BASE` whenever the base was known, with no check that rip was
  in the image -- so a wild jump produced a number indistinguishable from a
  real RVA, which a disassembly resolves to an unrelated function.
  `IMAGE_SIZE` was read from `LoadedImage` at boot and printed and never
  stored; it is stored now. `code::locate` is a pure function over (rip, base,
  size, registry answer) with all five of its states asserted at boot, the way
  `update::decide` is. Heap-resident generated code is named by tag and
  offset; an rip in neither is said to be in neither.

None of this prevents anything. There is no fault recovery here -- every IDT
vector but `#BP` is `-> !` -- so a bad jump is still a halted machine. What
the registry buys is that the one diagnostic which survives says something
true.

**And it does, verified by faulting on purpose.** `fault code` emits a null
dereference into an `Exec`, arms it and jumps in. The buffer is deliberately
`mem::forget`ed, because `fault` reads the registry from inside the handler
and dropping it would unregister the range on the way:

    *** EXCEPTION 0x0e  #PF page fault ***
      rip   0x0000000002bf2003   cs  0x0008
      in generated code fa17000000000003 at +0x3

The buffer was at `0x2bf2000`, so `+0x3` is exact.

**Getting that took fixing something much older: no fault this kernel ever
took produced a readable report.** `kprint!` writes the console first and
serial second, and painting from inside an interrupt gate takes a #GP here --
so the first line of every report died in the console before serial was
reached, and what a person saw was a machine that simply went quiet. Plain
`fault` was silent the same way, long before Phase F existed.

The report is emitted **twice, whole, serial before the console** -- not
interleaved a line at a time, which was the first attempt and still truncated
after one line. Serial is a port write and cannot block or fault. The console
is attempted afterwards regardless, because on the GF63 there is no UART and
the framebuffer is the only diagnostic there is; and `REPORTING` makes a fault
*while reporting* print one line and halt rather than recurse, which it did,
as an unbroken column of `EXCEPTION 0x0d`. Pacing is also turned off, since
1200us a character makes a report indistinguishable from the hang it explains.

**The console #GP inside an interrupt gate is a real bug and is not fixed.**
It belongs to the console rather than to the reporter, it predates all of
this, and it is now visible instead of silent.

**And there is a code generator now.** `src/aiksi/jit.rs` compiles one
function of integer arithmetic, `if`, `while` and `return` to x86-64, emits it
into an `Exec`, and calls it through the `sysv64` pointer `cpu::code` pins.
No builtins, no strings, no records, no `use`, no calls. Anything outside that
slice is **refused** -- `compile` answers `None` and the interpreter remains
the only thing that ran it -- and five claims check that refusing actually
happens, because a generator that quietly compiled a string return would be
answering a question nobody asked.

It is reached only from `differ`, never from a live path. Nothing routes
through it and `voter` does not know it exists.

**The step count is the hard part, not the arithmetic.** Twenty-one functions
run three ways -- armed, prepared, compiled -- and all three must agree on the
value, the cost and the error text, 64 rounds. The cases that earn their place
are the short-circuit pair, where whether the right side's ticks happen at all
is decided by a runtime value, and the runaway, which has to hit the budget at
the *same step*, not merely also stop. Failure text is compared too: division
by zero is a status in compiled code that becomes the interpreter's own words,
because a compiler with perfect arithmetic and the wrong error string passes
any test that only reads answers.

The harness caught the first mismatch immediately, on `fn f(): int { return 7
}`, and it was the harness's fault rather than the compiler's: `observe` runs
the top level before invoking, so the interpreter had already charged one tick
for *executing the `fn` statement* that declares the function. Compiled code
never runs a top level, so it owes exactly that tick -- and `only_fn`
guarantees the top level is one statement, so the number is one and not an
estimate. The budget it is given is short by one for the same reason.

A caution recorded because it cost three runs: two of the "still failing"
results after that fix were stale binaries, from commands that ran `cargo`
from the wrong directory. The `Bash` working directory is not the repo and
does not persist between calls.

The tick rule is written down on `Interp::tick` because anything executing
this language by another route has to match it exactly: **once entering
`stmt`, once entering `expr`, one extra per `while` iteration, and nothing
else.** A builtin costs one step however much work it does. The budget is a
safety bound, so a program that got more room by being run a different way
would be a runaway one path stops and another does not.

**`diag differ` is the gate, and it exists before the thing it gates.**
`src/aiksi/differ.rs` runs one program two ways and requires them to agree on
**value, step count and error text**, bit for bit with no tolerance -- the
reason `smp.rs` gives about a split matvec, that any difference at all is a
bug and a tolerance hides the one worth finding. Step count is in there
because it is what the budget stops and what a verdict records.

It was written before a code generator on purpose. `model.rs` makes the
objection twice -- two implementations that are supposed to agree do not stay
agreeing -- and a harness written afterwards is a harness shaped by whatever
the second implementation happens to do.

The second route today is `prepare`/`adopt` against `run`, which is a real
pair rather than a placeholder and is the one every routing decision now
depends on. A compiled route becomes a third `Route` and every case applies
to it unchanged.

**The canary is the part that makes it a harness.** A suite that has never
reported a difference is indistinguishable from one that compares nothing,
which is exactly how `smp.rs`'s one-shot check passed over a deadlock. So the
suite runs two programs differing by one unused declaration: same answer, one
more top-level statement, step counts one apart. It **fails if that is not
caught**. That is the difference a comparison looking only at answers waves
through, and the one a code generator with nearly-right ticks produces.

Eleven cases, sixty-four rounds, including the three failure modes and a
`while` loop for the tick most easily got wrong. Two limits are printed
rather than left to be assumed: stored cores and seeded tools are compared
too, but all three seeded tools have computing top levels, so the prepared
route declines them and the stored half of the corpus contributed **nothing**
on a fresh machine -- the line says `0 agreed, 3 declined`. And console
output is not compared, the same blind spot `skill.rs` records for J3: a
program of `println`s answers nil however it behaved.

Two honesty notes on those figures. `arm` and `call vote` are unchanged by
this work and the movements in them are host noise. And `fields_of` adds up
to eight string comparisons to every builtin call that misses, where the old
map cost about three; measured flat, and recorded because it is a real
regression on the call path even if nothing can see it.

**How big the noise is, measured rather than guessed.** The first version of
this bench timed *one* vote per sample, best of nine, and it could not
resolve its own subject: a change worth about a tenth of a vote read as an
18% regression on a boot whose step loop ran 25% faster than the boot it was
compared against. Each sample runs 200 votes now.

That fixes per-sample noise and not the boot-to-boot kind, so the bench
carries a control. `build interpreter` is `Interp::new` and nothing about how
a program is stored or called can touch it, so a difference in that line
between two builds is measurement error and nothing else. Across the pair
that judged the `Rc` change it read **16%** -- which is what "within noise"
is allowed to mean here, and it is a number instead of an adjective.

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

The look is Frutiger Aero in Aperture's colours, and it was 98 plus 3.1 until
1.3.0. `theme.rs` owns all of it and says so: changing the look is changing
that file rather than every caller, which is what made a whole reskin mostly a
change of numbers.

**The rule, where a rule was needed.** Aero is aqua, glass and saturation and
this machine's colour is orange, so each got where it belongs: the surfaces the
machine speaks through are warm -- captions, selection, links, focus -- and the
room it sits in is cool -- wallpaper, taskbar glass, fields, menus. The wall is
where they meet, a horizon from deep water to gold, with the Aperture mark lit
as the sun on it.

Four primitives carry it, all in `gfx/mod.rs` and all span-based, which is why
a gradient interface costs about what a flat one did. `vgrad` is a vertical
ramp, one `fill_span` per row, allocating nothing -- vertical for exactly that
reason, since a row of a vertical ramp is one colour and a row of a horizontal
one is a pattern that has to be built and blitted. **Two stops one position
apart is a hard step**, and every gloss break in the interface is one. `tint`
and `glass` blend toward a colour, `glass` ramping the opacity rather than the
colour. `shade` darkens, for the drop shadows. `ramp_at` and `tri_spans` are
shared lookups, so a shape and the hole cut in it cannot rasterise differently.

`theme::chrome` is the one formula for where a window's parts are, and
`theme::Popup` the one formula for a menu's. Both were several copies agreeing
by hand across two files, and both were unified in the commit that moved the
metrics, because that is the change that would otherwise have broken the
agreement silently -- as a title bar you can see and cannot grab.

The Start menu has a query row at its foot -- nearest the Start button, where
this menu opens upwards out of the taskbar and where Windows 7 put its search
box. Typing anywhere in the menu goes to it; Enter runs `open <query>`, the same
dispatcher the search panel uses. `KEY_STARTMENU` (`win keys start`) opens the
menu, which previously had no key at all and so could not be driven headlessly.

**A window arrives over about a tenth of a second, and only its chrome does.**
`open_flourish` plays six frames from a small rectangle at the window's own
centre before the window is drawn for the first time, and `ARRIVING` is what
tells `draw` to skip the window while its own animation is on screen. Chrome
only is a constraint rather than a shortcut: there is no way to blend a window
against a backdrop the back-to-front repaint has already overwritten, and
`Console::reflow` **discards rows when it shrinks**, so animating a terminal's
real geometry would destroy its scrollback to decorate its opening. `open_rect`
is pure and selftested, and the claim that earns its place is that the last
frame is the target *exactly* -- a final rectangle off by a pixel leaves a seam
of chrome the real window does not cover, and nothing repaints it until
something else happens to.

**The taskbar is drawn before the windows**, which mattered for the first time
when a hover tip had to stand above it: drawn from inside the bar's own painter
it went under the terminal, visible only in the strip of wallpaper to its left.
`taskbar_tip` is called at the end of the frame instead. Task buttons stay
pictogram-only -- every button the same width is what lets nine of them read as
a row -- and the tip is the third state between "no label" and "a label on
every button". A window the bar had no room for now shows as a `+N` chip rather
than being dropped in silence.

`todo` (shell) and the ToDo window share one list. It is the hand-off
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
  focused app would swallow the next line. **Alt-Shift-Tab is not the other
  direction of that** -- a swap is its own inverse, so it would be the same
  operation under a second name. It raises the *deepest* visible window
  instead, which is the one plain Alt-Tab can never reach, and it is
  `KEY_ALTTAB_BACK` rather than a modifier the desktop reads, because the
  desktop is handed bytes and has no view of what is held down. `win prev` and
  `win keys alt-shift-tab` are the typed forms.
- **QEMU monitor `mouse_move` deltas must stay within +-255 per axis.** Bigger
  deltas set the PS/2 overflow bit and the driver correctly discards the
  packet, so the pointer simply does not move, which reads as a dead drag
  instead of a clamped one.
- **`font::GLYPH_H` is 8 and not 16** (glyphs are 8x8, doubled by
  `CHROME_SCALE`). `TITLE_H` is `GLYPH_H * CHROME_SCALE + 14` = 30, `MENU_H` is
  22 and is **not** `TITLE_H` -- it was, over in `desk`, and that was a
  coincidence rather than a fact until the caption grew and took every menu row
  with it. Caption buttons are 21x21. Choreographing clicks from remembered
  metrics instead of a `[desk] press` trace cost two full test cycles aimed 40
  pixels left of the close box, and every metric on this line has moved since
  then.

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

**`sync::Spin<T>` is a lock**, and every conversion from one to the other is a
claim that a second core reaches that state. The claims are made one at a time
and each is verified, because converting all of them at once produces a kernel
where nothing is known to be right rather than one where a few things are. Two
have been made so far, taking the count from 93 to 92: the heap and the
console.

**`lock_irq` is not optional on either.** A lock taken by ordinary code and
also by an interrupt handler on the same core deadlocks against itself, and
both of those are in that position: allocation can happen under an interrupt,
and the clock task prints from a timer tick. A plain `lock` there is a hang
that appears under load and never in a test.

Nothing about the holder is recorded, deliberately: naming a core means reading
the local controller over memory-mapped I/O, which costs more than the lock it
would describe, and the allocator takes one on every allocation. A spin that
reaches its patience limit panics with the waiter and the lock address rather
than hanging, which is what makes converting the console safe to attempt: a
paint path that printed would take the lock twice and say so on the first line
of boot.

`diag mt` is the evidence for the heap. Sixty-four rounds of 4,096 allocations
across the cores, each writing a per-chunk pattern through its whole block and
reading it back. A heap that handed one block to two cores fails the read-back,
one whose free list corrupted fails a later request, and one that lost a block
fails the closing check that the heap is exactly where it started.

### ACPI, and the battery under it

`src/acpi/` is an AML interpreter. It exists because battery state on a laptop
lives behind bytecode the firmware ships in its DSDT, and there is no way to
read a charge without running that bytecode. Two cheaper routes were declined
and the reason was testability rather than effort: Linux's `msi-ec` approach of
hardcoded Embedded Controller offsets is three hundred lines and works on one
laptop, and QEMU emulates no MSI controller, so it would have shipped in the
state the RTL8188EU driver is in.

**The parser must be exact and the evaluator may be partial.** Those are
opposite obligations and separating them is what made this bounded. AML carries
package lengths inside itself, so one misread length desynchronises the rest of
a table: a parser that is ninety per cent right produces a complete-looking
namespace full of names the firmware never wrote. So the only acceptable result
is consuming the table to its last byte, and that is asserted. The evaluator
runs only methods somebody names, and an opcode with no arm is one method
returning an error that carries the opcode and its offset.

**The walk never enters a method body.** Everything that declares a name is
package-delimited, so bodies are stepped over by length. That matters because
the one hard problem in AML parsing lives inside them: a bare name followed by
arguments is a call whose argument count depends on a declaration that may be
in a table not yet loaded. ACPICA needs multiple passes for it. By the time the
evaluator meets it, the namespace is complete and the arity is known.

**Test against real firmware, not QEMU's.** QEMU's DSDT is nine kilobytes with
no battery. The GF63's is 575, and Windows hands it over through
`GetSystemFirmwareTable`. `-acpitable` caps an injected table at 65,535 bytes,
so it travels the way a corpus bundle does:

```powershell
.\tools\venv\Scripts\python.exe tools\mkfat.py .qemu\nvme.img out\gf63.aml
```

then `fat get /GF63.AML /tmp/gf63` and `acpi load /tmp/gf63` in the shell. That
found four opcodes QEMU never exercises: a region offset given as a name, one
computed with `Add`, a bare top-level `Store`, and the `CreateWordField`
family. Nodes went 70, 2491, 3367, 5110, 5889.

Bounded on three axes, because this is firmware bytecode in ring 0 and a fault
outside a guard is fatal. A step budget, since `While (One) {}` is legal AML. A
depth cap, since a method may call itself and there is no guard page. And
nothing runs unasked: building the namespace executes nothing, which is why a
top-level `Store` is stepped over rather than executed even though ACPI says it
should run at table load.

Region writes are off until `acpi unlock`. Reading a battery needs none, and a
stray write to an embedded controller is a fan that stops or a charge threshold
that moves, on hardware, permanently.

**The unit is a field of the thing it measures.** `_BIF` element zero says
whether the whole set is milliwatts or milliamps and machines differ, so a
capacity in mAh over a rate in mW gives a number that looks like a time and is
wrong by the battery's voltage. Everything converts once at the boundary. The
percentage is computed *before* conversion, since remaining and last-full always
share a unit and converting first would round twice.

`0xFFFFFFFF` is ACPI's "unknown" and is refused rather than believed.

Verbs: `acpi tables`, `acpi ns [path]`, `acpi eval <path> [args]`, `acpi load
<blob>`, `acpi s5`, `acpi off`, `acpi unlock`, `battery`, `ec`. Suites `acpi`
and `battery`.

**What cannot be tested here.** QEMU models no embedded controller, so the EC's
success path has never run: the ports answer 0xFF, both status bits look set
forever, and only the timeout is proven. Every battery figure seen under
emulation is the firmware's fallback branch rather than a reading.

### Text, and the font

`src/gfx/font.rs` draws 325 glyphs at 8x8 and `src/gfx/console.rs` decodes
UTF-8 to reach them. The console stored one byte per cell and drew one glyph
per byte, so every character above 0x7E the model wrote arrived as a run of
hollow boxes -- an em dash as two, a box corner as three.

**The cell is still two bytes, and where it is built is the reason.**
`console::init` runs twelve lines before `cpu::idt::init`, so the grid is
constructed at a point in boot where a fault is a triple fault: no message, no
register dump, an instant reboot. Two consoles of 128x72 cells cost 36 KB
today; a `char` per cell takes that to 147 KB and puts an extra 18 KB
temporary on the boot stack in exactly the window where running out of stack
explains nothing at all. So a cell packs a twelve-bit glyph index beside a
four-bit colour, which addresses four thousand glyphs against the three
hundred that exist.

The index is resolved when a character is **printed**, and the paint path is
an array lookup. `redraw_all` visits nine thousand cells a frame, so a search
per cell per frame would pay for the lookup nine thousand times to save it
once.

**The decoder is incremental because the console is fed bytes.** `put_char`
is also the keyboard's path and the recovery console's, and neither has a
decoded character to hand. Overlong forms are refused at the lead byte
instead of being decoded and then judged: an overlong sequence decodes to a
perfectly ordinary codepoint, so a check made afterwards is a check somebody
can forget, and the one place it is easy to forget is the one that matters.
The case that earns its place in the suite is a truncated sequence followed by
a newline, since a decoder that swallows the byte it could not use eats the
newline and the rest of the line with it.

**Most of the glyphs are not drawings.** `tools/font.py` composes an accented
letter from the letter this font already has plus a mark, and parses the ASCII
table out of `font.rs` rather than keeping a second copy that would drift. The
rule is uniform: the mark occupies rows 0 and 1 and the letter occupies rows 2
to 7, which for lowercase is the existing glyph untouched and for uppercase is
a six-row form with one interior row removed. Box drawing is generated from
segment tables and uses all eight bits of the cell, breaking the font's own
5-wide grid on purpose, because a line that stops short of the cell edge does
not join the line in the cell beside it.

```powershell
.\tools\venv\Scripts\python.exe tools\font.py --proof          # look at every glyph
.\tools\venv\Scripts\python.exe tools\font.py --proof 0xE9      # or just one
.\tools\venv\Scripts\python.exe tools\font.py --emit           # rewrite the table
```

`font` in the shell prints the whole coverage sheet on the panel. That is the
only check that settles a bitmap font, and the only one that reaches the GF63,
where there is no UART and the framebuffer is the entire diagnostic. `diag
text` checks what a program can check: that the table is sorted and every
entry is reachable by search, that an accented letter still contains the
letter it was composed from, that a cell is still two bytes, and ten claims
about malformed input.

**What it does not cover, said here rather than discovered later.** No CJK,
which needs a 16x16 cell and a table three orders of magnitude larger. No
combining marks, no shaping, no bidirectional text and no grapheme clusters:
one codepoint is one cell, so an 'e' followed by U+0301 draws as two cells and
not as an accented e. Anything with no glyph draws a hollow box, deliberately,
because a font that quietly substituted something close would be lying about
what it has.

**The terminal has a scrollback, and the view is not the grid.** 512 rows in a
heap ring, allocated on the *first scroll* and never in `console::init` --
which runs twelve lines before `cpu::idt::init`, where a fault is a silent
triple fault and where the fixed `[[Cell; 128]; 72]` sizing is load-bearing for
that reason. If the allocation fails the console behaves exactly as it did
before there was one.

`row_at` is the one answer to "what is on screen at row r", and everything that
draws goes through it, because `cells[r]` and screen row `r` stopped being the
same thing. `draw_cell` refuses while the view is back -- a character echoed
then would land in a row it has nothing to do with -- and `rows_starting`, the
caret and the status strip all read through the view too.

**Output does not yank the view to the bottom; a keystroke does.** That is what
every terminal does and here it is also free: when a row scrolls off while the
view is back, the history grows by one at the end and the view grows by one at
the start, and `row_at` resolves every screen row to exactly what it resolved
to before. Nothing is repainted because nothing moved. At the cap it cannot
follow any further and `redraw_all` runs instead. `set_col` is the keystroke
path, since the shell's `redraw` ends with it.

`win scroll <n|end>` is the typed equivalent, and it exists for the reason
`win keys` does: serial cannot inject PS/2 packets, so a scrollback reachable
only by a wheel is one nothing ever checks. Note it must read the view
**before** printing -- `kprintln!` writes, and the first version reported zero
every time because the message snapped the view home while rendering itself.

**A byte count stopped being a column count, and that broke code that had
been right for years.** Truncating a label with `&s[..room]` where `room` is a
column count does not produce a mangled label, it panics.
`theme::head_chars` and `tail_chars` are the safe forms and every display
truncation goes through them. `Write` wrapped its lines by byte and split
characters in half; `edit` decoded a file into `Vec<char>` correctly and then
mapped everything above 127 to a question mark on the last step before the
screen; `browse` carried a comment saying byte indexing was safe because
everything reaching the screen was Latin-1, which was true right up until it
was not.

### File formats

`src/fmt/` answers what a file is and hands back its structure. The namespace
stored bytes and nothing above it knew what any of them were.

**Extension first, contents second, and never a guess.** A name carrying a
known extension is that kind, because the operator said so by naming it, so
`notes.txt` full of JSON is still text. A name carrying none is sniffed.
Anything surviving both is `Text` when it decodes as UTF-8 and `Binary` when it
does not. Telling a program a file is JSON when it is prose earns it a parse
failure it cannot explain.

**One tokenizer, a table per language.** C, C++, C#, Rust, JavaScript, Python,
Aiksi and shell differ in comment markers, string delimiters and keyword lists,
and do not differ in lexical structure, so adding a language is adding a
`Syntax` row. The cases that earn their keep are the ones naive highlighters
fail: Rust block comments nest and C's do not, a Rust lifetime is an apostrophe
that never closes and must not open a literal that eats the line, a Python
docstring is a tripled delimiter, and a comment marker inside a string is not a
comment. That last set caught a real bug on the first run: opening a block
comment set the carry and re-entered the loop with the cursor still on the
opener, so the nesting scanner read it twice and `/* x` ended the line at depth
two and never closed.

`fmt::xml` is its own reader rather than `net::html`'s. HTML has void elements,
optional end tags and a parse algorithm defined by what browsers do, and
importing that leniency into a data format turns a malformed file into a
plausible tree. It refuses a mismatched close, a second root, an unquoted
attribute, and a DOCTYPE with an internal subset, because honouring part of a
DTD is how a reader disagrees with every other reader.

`fmt::table` covers CSV, TSV, JSON Lines and INI. The quoting rules are the
whole of CSV: a field may contain the delimiter, a newline and the quote
character, and splitting on commas gives a reader that works on every file
anybody tests it with and corrupts the first real one. JSON Lines reports the
line number of anything that will not parse and keeps the rest.

`fmt::outline` exists for the model more than the operator. A model reading a
forty kilobyte source file spends its context on it and answers worse than one
told the file defines nine functions and their names. Derived by scanning
rather than parsing, which is the right trade here: an outline occasionally
missing an entry is useful, a parser occasionally wrong about a program is not.

Six Aiksi builtins reach all of it (`fmt_kind`, `fmt_outline`, `csv_rows`,
`ini_get`, `xml_text`, `jsonl_count`) and the `file` verb prints kind and
outline. `diag fmt`.

### Temperature and frequency

`src/dev/power.rs` reads the sensor, measures the clock and sets a governor,
and **the gate in front of it matters more than the feature**. `rdmsr` on a
register the part does not implement raises #GP, every vector here is fatal,
and the result is a halted machine reporting a register nobody asked for.

Three conditions, all of them, before any MSR is touched: the vendor is Intel,
CPUID says the feature exists, and no hypervisor is present. The third is in no
manual. It is there because an emulator may advertise a capability in CPUID and
not implement the register behind it, and on real silicon the first two suffice
while under emulation they are a guess that does not return. **So none of the
readings can be checked under QEMU**, which reports "vendor intel, hypervisor
yes" and then declines with its reason. `power force` overrides it and says
what it is risking.

Frequency comes from delivered against reference cycles rather than a register
claiming one, because a part that changes its clock thousands of times a second
has no single frequency and the ratio describes the interval somebody cares
about. Governors are policies rather than frequencies, since naming a frequency
pretends to know better than the part about a decision hardware-managed states
exist to take over. The thermal policy has a gap between its thresholds because
one threshold oscillates, and it never touches turbo, because a machine that
quietly disabled it and forgot would look broken in a way nothing reports.

### The other cores

`smp::init` starts every application processor the MADT declares, walks it up
to long mode through a trampoline at physical 0x8000, and parks it. `smp` in
the shell reports how many answered; `smp bench` times a 16 MiB matvec on one
core against all of them.

This began as a **compute fabric rather than general SMP**, and it is moving.
The extra cores can now allocate and print, because those two structures are
behind real locks. They still never take an interrupt and still never run a
task, and the reason is specific: an application processor runs on the
trampoline's flat descriptor table with no task-state segment, so its code
selector does not match the one the interrupt table's entries name. Preempting
a task there needs a per-core GDT and TSS, and `smp.rs` already records why one
TSS cannot be shared. Running tasks cooperatively without a timer needs
neither, and is the shorter road if it is wanted. They wait on a generation counter
with MONITOR/MWAIT, run a range of a matrix, and go back to sleep. Every
decision and every byte of kernel state stays on the bootstrap processor, so
`Racy`'s safety argument is untouched -- which is the point. General SMP means
auditing several hundred `Racy` uses and inventing a lock discipline; this
needed none of that, and it is where the time goes anyway.

`smp::parallel_split(ctx, func, count, width)` is the whole interface. It
answers false -- meaning "do it yourself" -- if there are no helpers, if
another job is in flight, or if `count * width` is under 2^19. So it is always
an optimisation and never a requirement, and every caller keeps a serial path
that still works.

Two things it is easy to get wrong, both already paid for:

- **The slots are reused by every job.** A worker that caches the chunk count
  and then has the cursor reset underneath it will claim an index valid for the
  *next* job and out of range for the cached one, break out without counting
  that chunk, and the next job's tally never completes. `ACTIVE` counts cores
  inside the claim loop and the job is closed before that count is waited on.
  One job cannot reproduce this; the selftest runs 64 back to back.
- **Splitting changes no arithmetic, so the check is `==`.** A forward row and
  a backward column are each computed over the same values in the same order
  whichever core does them. Any difference at all is an index bug, and a
  tolerance would hide exactly that. Note the int8 scale offset is `lo * 4`,
  not `lo * cols`.

What is split: `Mat::matvec` (every projection, forward) and `Mat::wt_matvec`
(the adjoint, seven times per layer on a training step -- by column, because it
accumulates down rows and a row split would need per-core partials and a
reduction). What is **not**: `matvec_batch`, the prefill path, because it is
weight-stationary and a per-token split would multiply memory traffic by the
core count.

Measuring this under QEMU does not work and the numbers say so: one core reads
4570 MB/s alone and 3526 MB/s with seven cores merely *idling* beside it, so
both halves of any comparison are contaminated by the host. `smp bench` exists
to be run on the GF63.

**A foreground generation pumps the pointer.** `poll_mouse` is reached from
one place, the shell's idle loop, and `generate` only yields between tokens
when `opts.yielding` is set -- which the mind task sets and a foreground `ask`
does not. So for the whole of an answer the shell was inside the command and
the pointer was never polled, while the clock task kept its own quantum and
went on painting the uptime straight to the aperture. Frozen windows above a
moving clock, and it was a scheduling bug wearing a rendering costume: the
frame is 2.9 ms, measured.

`desk::pump_cursor` is **motion only, deliberately**. A press handled there
would run `press_at`, which can open or close a window or start an app --
re-entering the desktop, and possibly the engine, from inside a generation
that already holds it. Buttons stay latched for `poll_mouse` afterwards.

`video bench` is **best of nine** and prints the maximum beside the minimum.
It timed each operation once, and two consecutive `desk::draw` calls on an
idle machine measured 2,679 us and 24,318 us -- under emulation a single
sample measures the host's scheduler. It also times `console redraw_all`
separately, which is what named the console as the dominant cost of a frame:
1,672 us of 2,376.

**Blank cells were the whole of it.** `draw_cell` renders an 8x8 glyph
whatever the character is, so a blank cell wrote 64 scaled pixels of
background one `put` at a time -- and `desk::draw` had filled that same
background with a bulk span fill immediately before. About 1.2 million
per-pixel stores a frame, nearly all writing the colour already there.
`redraw_all` paints the background once as spans and then only cells with
something in them, for identical output:

    console redraw_all    1,672 us -> 592 us
    desk::draw + present  2,376 us -> 1,629 us

The survey that ranked this work missed it because it looked for per-cell
dirty tracking, which `desk::draw` defeats by erasing the client area every
frame -- and that is why A1 was written off as invalid as designed. The
redundancy was never in repainting cells that had not changed; it was in
painting *nothing*, pixel by pixel.

**Damage-rect present is not worth building, and the measurement says so.**
`present, no change` is 264 us of a 1,629 us frame, and that is a `memcmp`
over 4 MB -- about 15 GB/s, already at memory bandwidth. Skipping rows needs
to know which rows changed, and `desk::draw` repaints everything
unconditionally, so its damage *is* the whole screen: a damage-rect API would
be handed all of it every frame. The premise fails until `draw` itself becomes
incremental, which is a much larger change than the plan described.

Two renderer changes that are correct and measured **flat**: `put` no longer
goes volatile into RAM, and the title gradient is row-major rather than
`w * h` one-pixel spans. Together 13% off the console path and nothing off
the frame. Kept because they are strictly less work for identical output, and
recorded because the survey that proposed them called volatile `put` "the
single largest constant-factor loss" and the measurement disagreed.

**The clock draws through the compositor now.** It was the last thing writing
straight to the aperture, so the shadow went on describing whatever had been
there before and `present` -- finding `back` and `shadow` equal over that
rectangle -- wrote nothing, leaving digits on screen the desktop had already
decided to paint over. It takes the same claim the cursor does, since it runs
on the clock task and `draw` runs on the shell's, and they would otherwise
both write the back buffer through a `&mut` neither knows the other holds.
Losing the claim costs one tick of clock.

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

The module map, since `src/ai/` is now thirty files:

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
| `work.rs` | Workflows: the plan graph, the manager, roles, autonomy |
| `skill.rs` `study.rs` `abstraction.rs` `voter.rs` | Judged skills, the corpus study, abstraction, the cores |

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
the signal instead of their vote: 90.3% right when all three agree against 50%
when they split. That gap is what `gate` acts on.

**Training below the classifier now has a verified gradient.** `Tape` keeps the
residual stream entering each layer and nothing else; `Model::backward` walks
layers last-to-first, recomputing each from that stream and composing the
adjoints in `backward.rs` (which until then had exactly one caller: their own
selftest). `Dora::backward_x` supplies the input gradient those adjoints need
and which `Dora::backward` never produced -- the single absence that made a
classifier adapter trainable and a q/k/v adapter not.

`State::new_exact` keeps the KV cache in f32. It exists because `KvLayer`
stores int8, which makes the loss piecewise constant in anything upstream of a
cached key or value: differencing a layer-0 query through the quantised forward
reported -0.305 against an analytic gradient of order 1e-6. A switch, not a
second forward -- the drift argument that applies to the tape applies here too.
Serving still quantises, so training against these gradients is a
straight-through estimate, which is the usual bargain and is written down
rather than discovered later.

The gradient check is **directional** and uses cross-entropy, both for
resolution. A per-entry difference asks f32 to resolve `grad * 2h`, which for a
deep site is below the loss's own rounding -- it reported -1000/65536, a float
quantum wearing the costume of a derivative. Stepping every parameter along the
gradient asks it to resolve `2 eps |g|^2` instead, and checks the whole vector:
1.148 analytic against 1.135 numeric over 57,600 entries.

**The frozen base is the load-bearing property.** Nothing above the adapter
moves, so a hidden state is a constant, a constant can be cached, and a cached
decision can be replayed against any number of candidate adapters for the price
of a dot product. Training is affordable because of it, judging is nearly free
because of it, and any verdict the machine reaches can be re-checked later for
almost nothing because of it. Anything that proposes to train the attention
path is proposing to give this up, which is a real trade and worth naming
before it is made.

### The conversation

`ask` is one continuing conversation, not a question asked into the void.
`ask new` forgets it; `about <text>` appends to `/ai/about`, which is read into
the system turn of every new conversation.

**It resumes the KV cache instead of re-sending a transcript.** Every chat
program re-sends its whole history each turn because the model is behind an API
and the cache belongs to somebody else; here the cache is ours and it stays, so
a turn costs the tokens of that turn and the tenth exchange is as cheap as the
first. Measured under QEMU: position went 135 to 150 across two turns, growing
by the new turn alone -- a rebuild would have re-fed the ~120-token system turn
and landed back where it started.

Nothing here is new machinery. `GenOpts::resume`, `ctx_save`/`ctx_load` and
`set_window` all existed and were simply never joined up.

**Autosnap is on by default and the conversation is deliberately not in it.**
A fact told to the model (`remember`, `about`) is a few hundred bytes and is
carried automatically. The KV context is not: parking it writes the whole cache
as a blob, and this store is append-only -- `alloc_next` only rises, nothing
reclaims -- so with autosnap running, two turns of a *512-slot* cache wrote
16,375 then 19,625 blocks and took half a 27 MiB region. A 0.6B at 8k has a
cache three orders of magnitude larger. There is no cadence that makes that
affordable, so it is on request: `ctx save live`, which `revive` reads at boot.
Removing the per-turn park took the same measurement from 16,375 blocks to
**1**.

Defaulting autosnap on is safe only because the *write* gate is separate and
stays manual: mounting a store deliberately does not unlock it, so a machine
nobody has run `store unlock` on behaves exactly as before. That also means
`remember` has three outcomes and now says which -- no store, mounted but
locked, or kept and written within the interval. Proven across two boots with
**zero manual snaps**: `remember the passphrase is cathedral`, `[autosnap]
snapshot 2, 1 block(s)`, reboot, `about` -> `the passphrase is cathedral`.

**Durability is a snapshot, not a write.** `sysbox::write_blob` puts a blob in
the working tree, which is memory; it survives a reboot only once `snap` has
committed the tree, and only if a store is mounted at all. The first version of
`park()` returned a bool, reported success for a RAM-only write, and lost the
conversation at the next boot having said it was saved -- so it answers
`Durable`/`Volatile`/`Failed` now. A companion that says it will remember and
then does not is worse than one that admits it cannot.

Verified across two boots with a store mounted: `snapshot 2 root
0216018537036ad6`, then `the conversation from last boot is still here (162
tokens in)` and `about` reading back what was told to it.

**The conversation does not end at the context wall.** Within 64 positions of
the trained length it turns the cache into a ring (4 sinks, per StreamingLLM)
and evicts its oldest turns instead of stopping. Measured: `cache now a ring of
511 (468 positions kept)`, then `position 649, streaming through a 511-slot
cache`.

That is only safe because of a coincidence worth knowing. Unwindowed, a
position lives at the slot with its own number; windowed, at
`sinks + (abs - sinks) % ring`. Those are the same address while
`abs - sinks < ring`, so switching *before* the ring would first wrap needs no
entry re-seated, and the buffers are already larger than the new capacity.
`set_window` takes that path only under those conditions and clears the cache
otherwise, because the general case genuinely cannot be re-seated.

The first version of this got it wrong in the loudest possible way and the
output still looked fine: `set_window` cleared unconditionally, so a feature
announcing "it now forgets its oldest turns" forgot *all* of them, position fell
from 468 to 72, and the model carried on answering fluently. Only `ctx` showed
it.

**The system turn is pinned, and that is the same mechanism.** `slot_of`
returns `j` unchanged for `j < n_sinks`, so a sink is not merely a privileged
position -- it is a slot that never recycles. Setting the sink count to the
system turn's *token length* therefore pins the instructions, the applet list
and `/ai/about` for as long as the conversation runs, in their original slots
and with their original RoPE angles. Four sinks buy stability; the whole turn
buys memory of what the model is.

The count is taken by encoding the system turn the way `generate` will (same
BOS, same tokenizer), not estimated -- a count that ran short would pin part of
a turn and leave the rest to scroll. It is clamped to a third of the trained
length, and `ctx` prints span and pinned separately so a clamp shows up as the
two disagreeing. The cost is that a pinned slot never recycles, so the recent
window is shorter by exactly the system turn: a fifth of the cache at 512,
noise at 8192. Measured: 120 positions with no `about`, 154 after adding a line
to it.

**`remember <text>` is an applet, not a parse of the model's prose.** It is in
`sysbox::APPLETS`, so the decoding grammar carries it and the model reaches it
by the same route as `ls` -- an answer that merely says it will remember cannot
be mistaken for one that did. It refuses duplicates, since the file is read into
every system turn.

**The resident mind speaks as a turn.** It always generated into this same KV
cache (`resume: true`), but unframed: a thought it had on its own spliced into
the middle of its last sentence to the operator, and the next question read as a
continuation of it. `companion::interject_frame` closes the open turn and opens
a labelled one, so neither the operator nor the model has to guess who said
what.

### Skills, and who is allowed to be the operator

Every program under `/ai/tools` used to run on `TOOLS`, which is
`Interp::new()` -- operator capabilities: raw memory, I/O ports, the network,
the model, the framebuffer. An *app* has been jailed since `app::call` was
written; a *tool* was not, and `agent learn` writes tools, and a skill shared
by a stranger is a tool. It was open by omission rather than by argument.

`cmd_run` now follows `app::call` exactly: operator powers only for bytes the
operator has named, everything else fresh and sandboxed in its own subtree
under `SKILL_BUDGET`. Identity is the SHA-256 of the file, so **editing a
trusted skill revokes its trust by construction** -- the same property
`app::manifest` gets from putting `raw` inside the hash.

`skill list|trust <hash>|untrust` is shell-only and never an applet, for the
reason `app trust` is: a model that could grant itself trust would have
defeated the gate by using it. An ambiguous prefix is refused, not resolved
to the first match.

The three seeded tools need only Pure and Read builtins and go on working
sandboxed. Nothing that ships needs operator powers, and now nothing has them
until asked.

**Writing a skill is no longer adopting it.** `agent learn` compiled a
successful episode into `/ai/tools` and that was the whole of adoption -- the
file appeared and `run` would execute it, with nothing having asked whether it
was any good. `skill judge <path>` stores a candidate by content address under
`/ai/skills` and runs it through `godel::run` as a `ProposalKind::Skill`;
adoption copies it to `/ai/tools/learned-<hash>.ai&xi`, writes a node, and
leaves something `godel rollback` can undo.

Four judges, and what they are is constrained by what a replay skill *is*. It
takes no arguments and dispatches a fixed sequence, so "does it work on a task
it has not seen" is not a question it can answer -- a judge for that could
never fail here, which is worse than not having one. What is left is
admission, not improvement: it parses (J1), it runs under the powers an
unadopted skill actually has (J2 -- a replay of an episode the *operator*
drove often depends on a mutating applet a sandbox refuses), it repeats (J3),
and it is cheap (J4).

J3's blind spot is written down rather than discovered later: it compares the
value the program answered, its step count and the objects it touched, and
**not what it printed**. A replay is a sequence of `println(applet(...))`,
which answers nil however the applets behaved. The selftest claim for J3 was
itself wrong twice for related reasons -- first printing the clock instead of
answering it, then answering a 100 Hz clock that does not move between two
adjacent runs.

**`run`'s argument is decoded under a grammar.** The applet *name* always was,
so an applet that does not exist is unreachable; its arguments were free text,
which is right for a filename to write and wrong for `run`, where the whole
space is enumerable. The model had to spell `/ai/tools/learned-3f2a91c4.ai&xi`
exactly, so a skill it could not spell was a skill it could not use however
good the judges said it was -- adoption that put a tool in the toolkit nothing
could pick up. `agent::skill_choices` is the list and the grammar is built from
it. Not exercised through a real episode: that needs a model and minutes, and
what is checked at boot is that every choice is a full path to a program.

### Multi-agent workflows (`src/ai/work.rs`)

A manager delegating to workers, which is a different relationship from the
two multi-agent systems this tree already had. `council.rs` and `godel`'s
judges are independent evaluators whose *disagreement* is the product; this is
about getting something done.

**Three facts decided the shape before any code, and none is negotiable.**
`HOLDER` is one `AtomicUsize`, so two `&mut Engine` is undefined behaviour and
concurrent inference is unavailable rather than slow. `agent::Job` is one queue
on one task on purpose. And ensembling for accuracy was measured here and lost,
77.8% for the best single core against 76.9% for a product of three. So what
multi-agent buys here is specialisation, and never concurrency.

**Memory is the namespace, because a conversation is the one thing that cannot
be afforded.** `ctx_save` on a 512-slot cache measures in thousands of store
blocks, so swapping a context per worker costs more than the work. The
namespace is already a content-addressed Merkle tree, so the graph inherits
dedup, constant-time copy, `same` over whole subtrees, and `snap` versioning
everything as one root hash. A context switch becomes a namespace read.

```text
  /ai/work/<run>/plan            the plan tree, re-read at every step
  /ai/work/<run>/steps/NNNN      one summary per completed step
  /ai/work/<run>/artifacts/<n>   what is being built
  /ai/roles/<role>/ex/NNNN       harvested examples for a role
  /ai/roles/<role>/adapter       the role's adapter, if one was kept
  /ai/autonomy                   one hex line per granted plan
```

```
work                      list runs, each with its root hash
work new <run> <goal>     one step, decided by the worker
work plan <run> <goal>    the manager: decompose and write the actions down
work run <run> [budget]   execute, read-only
work cmp <a> <b>          decisions and observations, reported separately
work harvest              transcripts into per-role example sets
work roles | train <role> the Stage 3 measurement
work autonomy <run> ... | check | trust | untrust | trusted | night
```

**Execution is free, and that is the whole result.** `agent::prompt_for`
includes every prior step, so an episode re-encodes a growing prompt and spends
O(N^2) tokens over a run. `harness::plan_actions` encodes once and leaves the
engine positioned, so N actions cost one prefill and N decodes. And a worker
handed a pre-decided action decodes nothing. Measured on a four-step goal with
Qwen3-0.6B: `work plan` spent 4 decode calls, `work run` spent **0**.

A written action is re-checked against the trust level at dispatch rather than
trusted for being written down. A plan is a file, and a file can be edited by
anything that can write.

**A run has an address, and only half of it is re-derivable.** Stage 1 was
written expecting two runs of one plan to produce the same root hash, and they
did not: a run writes its steps under `/ai/work`, which is inside the `/ai` a
worker then lists, so the second run legitimately saw a directory the first had
changed. So a step records `action`, which is the decision and must agree
across runs, apart from `observation`, which is a reading of a world that
moves. `work cmp` reports them separately because decisions differing is a
defect and observations differing is Tuesday.

**Two prompt findings that cost a run each.** The manager's prompt has to
*demonstrate*: "one tool per line, then done" in words was ignored by both
checkpoints, and a two-line worked example took Qwen3-0.6B from zero steps to
four. And the grammar consumes an applet name exactly, so a prompt that does
not emit a separator after it leaves the model continuing mid-word -- every
step of the first run planned `find - - - -`. `decode_args` never had the bug
because its prompt ends `"name "`.

**Role adapters do not work, and the reason is structural.** `work harvest`
builds a per-role set from transcripts and `work train` puts it through
`godel`'s four judges, called rather than copied. It was measured on 24
workflows: 23 examples from 23 runs, base 97.3% on the training slice and 100%
held out, `fixed 0 broke 0`. **A harvested label is the base model's own
argmax** -- `choose` decodes at temperature 0, so the action in a transcript is
what the classifier already ranks first, and training on it asks the adapter to
reproduce whatever produced it. Filtering on `ok` does not escape that, because
`ok` records that the applet ran and never that it was the right one. On this
evidence a role is a naming convention, and a role adapter needs a label from
somewhere else: `teach`, a judged outcome, or a larger model as teacher.

The split is by *run* and never by step, since steps inside one run share a
goal and a step split measures memorisation while looking like generalisation.
Same argument as holding out whole template families.

**Autonomy takes a declaration and a grant.** `work autonomy <run>
unattended` writes the declaration, which grants nothing on its own: a plan is
a file, so a workflow declaring its own autonomy would be the gated party
writing its own permission. `work trust <run> <hex8>` is the operator's half,
in the `update stage` idiom. `work` is absent from `sysbox::APPLETS`, so no
grammar can spell it and the model has no route to the command.

**The grant names the plan without its statuses**, and that detail is the
feature. Hashing the file as written would mean the first step revoked the
grant that let it take that step, which is a gate that works exactly once and
looks like it is holding. `work check` re-runs before every unattended step,
because a grant is evidence the operator approved an intent rather than
evidence the intent is still admissible, and `admitted` comes from the live
applet table.

**The unattended branch is not gated on `spent`, and that is deliberate.**
`initiative.rs` records what happened the last time a job went behind the
one-expensive-job-per-tick rule: godel won the tie every night and the job
behind it was never tried at all. A pre-decided step costs no model call, so it
does not compete for what the other two compete for; a step the worker must
still decide waits for an unspent tick. `work night` takes the step the quiet
tick would take, which is how any of this is testable -- the first quiet tick
queues an episode in the same moment the prompt appears, and under emulation
that stands the whole block down for minutes.

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

**`sandbox` copies nothing up front.** It opened by deep-copying the whole
namespace so it had something to restore from, which meant every run paid for
a clone of every object in the tree to undo a program that usually touches one
file -- and held `&mut Sysbox` across a full recursive walk with interrupts
disabled, twice. `note` already runs immediately before each mutation with the
path in hand, so each path's pre-image is saved exactly there and the cost is
proportional to the change. It records the *shallowest* path that did not
exist, not the one asked for: `tree::put` creates intermediate directories, so
undoing only the named file would report the run as reverted and leave
directories behind.

The read-through overlay -- a namespace handle threaded through `with`, so a
run is invisible to other tasks rather than merely undoable -- is still not
built, and the reason is now a decision rather than a deferral. It has to be
paid for in the type: `Node` owns its children, so a persistent tree sharing
unmodified subtrees needs reference counting through every accessor in the
kernel. What it would buy is isolation the jail already provides, since a
sandboxed skill can only write under its own scratch subtree.

**The unlock names a range, and the range is enforced.** It was one bit for a
long time: unlocking said writes were allowed and nothing said *where*, so from
the moment `store::init` succeeded every LBA on the device was writable -- the
partition table at zero, the ESP, and the Windows volume that is still the only
other thing on this disk. `nvme::write` checks `may_write(lba, count)` against
the window the unlock claimed, `store` prints the window beside "UNLOCKED"
rather than leaving an operator to assume it means the disk, and `diag wgate`
asserts the whole decision without a device and without writing anything --
including that a write starting inside the window and overrunning it is
refused, and that a length overflowing `u64` does not wrap into a pass.

This is a prerequisite for anything that writes the ESP, not a separate
concern: an ESP updater built on a global unlock is a whole-disk writer.

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
configuration only when measured better, writes it to `/ai/config`, and spends
one read against `godel`'s test budget like anything else that touches the
slice. The router honours what it adopted through `harness::decide`, which is
the one implementation of a rule; `rule_in_force()` defaults to `Majority`, so
a machine that has never searched routes exactly as it did.

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

Negative results stay in the tree. The **gradient-descent classifier head**
that `probe.rs` replaced made held-out accuracy *worse* with every epoch --
30% untrained, 10% after two, 0% after eight -- because 40 examples across 21
classes leaves SGD nothing to do but memorise. The Product-of-Experts council
does not improve accuracy. Both are kept because the reason to know them is
the reason they were worth measuring.

That first one said "the adapter head" here for a long time and it cost a
session. It is the **SGD head**, which the ridge probe replaced; it is not the
QDoRA adapter, and the two have nothing to do with each other. `probe.rs:3`
has always been precise about it. This line was not, and the summary is what
got read.

**What the QDoRA initialisation fix does invalidate is separate**: every
verdict `godel`'s adapter grid ever recorded. Until `Dora::new` seeded A, both
low-rank factors were zero and the branch had identically zero gradient, so
every rejected grid point was rejected on an adapter that could only rescale
rows. Those rejections are not evidence any more. J4's cost figures stand --
the shapes did not change.

`tools/traces.py` reports what it could **not** produce and the per-family
imbalance unprompted. A generator asked for 20,000 that quietly returns 54
near-duplicates yields a corpus that trains a model to recite.

Sample sizes get stated wherever a figure appears. The adapter trainer has been
exercised on subsamples of a few dozen decisions, which establishes that the
machinery composes and establishes nothing about how much it helps. Numbers
from those runs do not belong in a claim.

## Gotchas that have already cost time

- **A byte count is not a column count.** Every glyph is one cell wide, so a
  width is a *character* count, and the two were the same number only while
  the font was ASCII. `s.len()` for a label width overstates it by one per
  accent; `&s[..n]` with a column count panics outright. Use
  `theme::text_w_of` and `theme::head_chars`/`tail_chars`.
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
- **Per-core storage is not only about other cores.** `init_smp` returned
  early when the firmware declared one CPU, or declared no ACPI tables at all,
  and both returns came before `cpu::percpu::arm()`. So `armed()` stayed false
  forever, `billed()` answered `None`, and the two things that read per-core
  state for reasons that have nothing to do with parallelism both died
  silently: `mem::census` could bill no allocation, and `recover::slot` could
  find no landing pad, which made **every fault inside a guard fatal** on a
  machine whose stated reason for having guards is that it runs programs it
  wrote itself. Measured at `-smp 1`: `diag recover` halted the machine at its
  third claim, twice out of twice, and `diag census` failed six of seven. The
  bug was invisible because nothing in the tooling had ever started the guest
  with more than one core, so the only configuration anybody ran was the
  broken one, and it was found while giving the guest a second core for an
  unrelated suite.
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

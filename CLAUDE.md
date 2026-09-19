# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this is

GLaDOS: a from-scratch, non-Unix, ring-0 operating system in Rust for one
specific laptop (MSI Thin GF63 12UC, board MS-16R8), built around a language
model that lives *inside* the kernel. No user/kernel split, no syscalls, no
process isolation, one address space. A tool call from the model is a function
call.

198 files, roughly 132,000 lines, of which 178 files and 118,000 lines were
written here. **Three** things in the kernel we did not
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

The self-improvement half, all four with a `--selftest` that needs no kernel:

```powershell
.\tools\venv\Scripts\python.exe tools\rails.py judge before.txt after.txt --claims host.retrieval
.\tools\venv\Scripts\python.exe tools\knob.py check          # the table against the real source
.\tools\venv\Scripts\python.exe tools\knob.py show|apply|revert PATCH
.\tools\venv\Scripts\python.exe tools\retrieval.py out\forest --dump out\retrieval-a.tsv
```

A verdict travels as its text followed by exactly 80 bytes of `GLADOSIG`, the
one object `manifest.py` already argues for: two objects can be served out of
step and produce a signature failure that is really a deployment race. Sign it
with `sign.py verdict.txt verdict.sig --key-file update.key` and concatenate.

`knob.py check` is the one CI must run. Nothing in the kernel can compare its
own table against the source, so a row that has gone stale produces a patch
that applies to nothing -- the marker written, the point never measured, and
the loop quietly spending nights on a constant that moved.

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
godel [next|space|forget|lib|rule <name>|window <from> <until>]
godel [clade|reconsider]          where to grow from, and going there
godel [source|push [--dry]|verdict <path>]   a change something else must build
bench report                      every rail, one block, for tools/rails.py
bench bpb [bytes]                 how well it predicts its own history
```

Four judges, unanimity required, each a different failure mode:

- **J1** is paired (McNemar) over the same cached decisions both variants
  answer, which is what a comparison of two percentages cannot be. It needs
  roughly six repaired validation decisions with none broken, so it needs a
  corpus subsample large enough to reach the held-out slice at all.
- **J2** asks whether the machine still sends its own curiosity goals where
  they were declared to go, walking each one *under the candidate*, and is
  never subsampled by `-n`. It replayed them along the frozen baseline's own
  cached path until it was measured -- see below.
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

**The rotation is a ranking now, not a round.** `next_proposal` reaches for
the axis whose next verdict the ledger can *least* predict:
`axis_uncertainty` is a Laplace-smoothed Beta posterior folded to a distance
from the coin-flip, so an axis that has said yes to everything and one that
has said no to everything are equally predictable and equally uninformative,
and the axis near 50% is where the information is. Fairness traded for
information, stated as such in the code.

Smoothing is what stops that becoming starvation -- `+1` over `+2` keeps a
saturated axis strictly above zero -- and ties break by slot, so the order is
total and a later run reconstructs the same one rather than a plausible one.
The composed core is **exempt and always last**: producing its candidate costs
a dozen decodes, so it is reached for when everything cheaper is out of moves,
never because it looked uncertain.

The counts come out of the ledger, which could not previously say which axis a
line came from -- five judges wrote lines in the same shape. Certificates carry
an `axis=` now, and the line carries the MAP-Elites `cell=` the variant lands
in, derived at render time from columns already there. Lines written before
that field count for nothing, which undercounts the early history rather than
guessing at it: an axis whose record is invisible reads as untried, an untried
axis is maximally uncertain, so it gets reached for and measured.

On a fresh machine this is a no-op. Every axis is untried, therefore maximally
uncertain, therefore ties break by slot and `adapter` goes first -- exactly
what `ledger_len() % KINDS` did.

`godel next` reports where the ranking stands without taking a turn, and says
whether the bar is frozen or open. It deliberately does not ask the last slot
whether it has work, because finding out costs those decodes -- a command
answering "what would you do tonight" must not spend the night doing it.

**The criterion holds still while the agent moves.** A loop that may rewrite
its own bar at any moment converges on an evaluator that says yes to
everything, and a ledger full of those adoptions would be *true*. So the bar is
frozen inside an epoch of `EPOCH_LEN` = 5 trials -- one per other axis -- and
questionable only at the edge. `is_boundary` is a pure function of ledger
length, so where the loop stands is re-derivable from the record; genesis is
excluded, because at length zero there is no agent to protect.

At a boundary the loop may propose a bar in `[0.5, 12.0]` off a declared
six-point grid. Coarse deliberately: a continuum lets the bar creep down by a
hundredth a night, every step honestly certified, and the sum of them is
exactly the drift the axis exists to catch.

`judge_verdict` is pure and asks three questions of a cross-evaluation matrix
-- the same candidate under both bars, plus a held-out anchor reading neither
bar can see. **Sane**, finite and in range. **Moves**, the two bars actually
disagree about this candidate, since a bar that admits and refuses what the
standing one did is a certificate with no content. And **honest**: loosening is
admitted only when the newly-admitted variant genuinely improves held-out
accuracy, tightening only when the newly-refused one genuinely was not a gain.
Without the third this axis is a machine for lowering its own bar.

All seven outcomes are asserted at boot with no model and no NVMe, the way
`update::decide` is.

**The price is the test-slice budget, and charging it there is the point.**
Every other axis reads the anchor after a variant has already won, as
confirmation. Here the anchor *is* the evidence, because both bars are what is
under suspicion -- so `read_anchor` spends one read to score both sides, and
after three a criterion change cannot be grounded at all. The one
non-renewable resource in the building is what a moving criterion is billed
against. A bar adopted on a stale read is refused.

**`bar_in_force()` reaches the judges**, or the axis would be a certificate
about nothing: J1 in `trial`, `trial_deep` and `trial_config` reads it rather
than `MCNEMAR_95`, which stays the default and the constant. A stored bar
outside the range is *ignored rather than clamped* -- clamping would let a
hand-edited file slide the bar to the nearest legal number and report nothing.

**The archive illuminates instead of hill-climbing.** Twelve cells, four rank
bands crossed with three repair behaviours, where behaviour is the *fraction*
of touched decisions that were repairs. A variant earns a cell by beating
whatever is in that cell rather than by beating the champion, so a rank-4
adapter that repairs a different set of decisions than the rank-32 one survives
on its own terms instead of being discarded over a margin inside the noise.
Cells hold addresses in the same DAG the ledger names, so an elite from three
weeks ago is still reachable and still re-derivable.

`godel storm [n]` is the whole apparatus in one command: one `prepare`, n grid
points against those cached features, descendants trained from the incumbent,
chimeras bred by blending the low-rank factors of same-rank survivors (capped
at six, paired in generation order), everything scored on validation, cell
winners offered to the archive, and the best of the generation put in front of
the same J1 the nightly loop uses.

Chimeras carry an honest `Fit` of zero epochs and zero loss because they were
never trained -- and **averaging factors is not averaging the function they
compute**, since `B.A` is bilinear. A chimera is a cheap mutation operator that
lands near two things that worked, which is exactly why it is judged rather
than assumed. `s` is recomputed rather than blended, because it caches
`m / |W0 + B.A|` against the frozen rows and a blended one would describe
neither parent.

Widening this had to come last, and the ordering is the point rather than an
accident: an axis in the rotation without a judge in front of it is a machine
adopting things nobody measured.

Still not done from that plan item: an authored application is left as a draft
and never adopted, and `aixi`'s plan is still stringified to a report rather
than gating how much the loop attempts.

### Five defects in the loop itself, and four of them chained

All five were found by reading and every one is now closed. Worth knowing
before touching `godel.rs`, because each was invisible from every other
vantage point.

**`trial_lib` judged and adopted nothing.** It built a certificate with
`adopted: false` hardcoded, with no `store`, no `set_head` and no ledger line,
so a library function could pass all four judges and be discarded. The
consequences chained: no ledger line means `axis_counts()[4]` stays `(0, 0)`,
so `axis_uncertainty(0, 0)` is 1.0 forever, so `lib` sorts first in
`surprise_order()` whenever it has work -- and `next_lib` was the one `next_*`
that did not consult `/ai/godel/tried`, while `run` has always written a
marker for a `Lib` proposal. So the loop re-offered one refused candidate
every night, in front of every other axis, indefinitely.

**The judge axis corrupted its own lineage.** `trial_judge` packed the bar
into the node's `rule` byte as hundredths over a grid of
`[1.0, 2.0, 3.0, 5.0, 8.0, 12.0]`. `as` saturates, so four of the six became
255, and 100, 200 and 255 are none of them a `Rule`. Every judge node named a
routing rule this kernel does not have and `rollback` onto one fails saying
exactly that. `Variant.bar` is a conditional `Option<f32>` now -- the
`core`/`deep`/`lib` pattern, so existing nodes still render to the bytes they
were stored as -- and **every axis records `bar_in_force()`**, not only the
judge axis, because a lineage that drops the criterion on the next adapter
trial does not record it.

**`storm` reported an adoption that never happened.** The shell printed "the
best was accepted" off a field with no `set_head`, no ledger line and no
`TRIALS` increment behind it. The words were fixed rather than the behaviour,
and that is the argument: a storm is an *explorer*, its product is the archive
it really does populate, and it weighs the generation against J1 alone.
Adopting there would put a weaker gate than the nightly four-judge unanimity
in front of the head. The field is `cleared_bar`.

**`trial_skill` and `trial_lib` bypassed `ensure_head`.** Neither takes an
engine, so both read `head()` directly and could write a node whose parent did
not describe the running mind -- after an out-of-band `adapter load` or `core
install`, both of which touch neither head nor ledger. `run` calls
`ensure_head` once before dispatching now, which is where the rule belongs.

**And the nightly trial could not pass J1 at all.** Not rarely: never. Not the
`no validation decisions` veto either -- `Trial::prepare` already strides, so
24 examples reach the held-out slice correctly. Measured, from the report's
own lines:

    examples   validation decisions   incumbent wrong   J1 needs
    24         15                     5                 6
    96         56                     26                6

`clean_fixes_needed()` is **six** and not `MIN_FIXED`, because Yates'
correction subtracts one before squaring. So at 24 examples the judge asked
for six clean repairs where only five wrong answers existed. An
arithmetically impossible trial, every night, for as long as the axis has
existed; what accumulated across nights was rejections. `GODEL_EXAMPLES` is 96
now, where the headroom is 26 against a need of 6, at a prepare cost of 321 s
against 63 s under WHPX with SmolLM2 -- the half `GODEL_MS` does not bound.

`Certificate.wrong` carries that ceiling and the report prints it beside the
requirement, so a budget that cannot pass says so in one line rather than
looking like a run of bad candidates. `Trial::paired` had always computed it
and thrown it away. It is an `Option`: an axis with no paired test says
nothing rather than claiming zero, which would read as "the incumbent was
already perfect".

**J2 still vetoes** on the curiosity goals at both budgets measured, so the
budget fix makes a nightly adoption *reachable* rather than likely. What J2
turned out to be is its own section.

### J2 was protecting the incumbent's mistakes, and could not say where a goal went

The judge that asks whether a variant has changed the machine's character
replayed four curiosity goals against **the incumbent's own cached decode
path** and required all four to land the same way. Three things were wrong
with that and only the first was suspected.

**It had no ground truth.** The incumbent's answer was the thing to preserve
whether or not it was right, so a candidate that routed a goal *correctly*
where the baseline had been routing it somewhere else was vetoed for having
changed the machine's character -- by the same trial whose J1 rewards exactly
that repair. Two judges pointing in opposite directions on the same handful of
items. `CURIOSITY` carries the applet each goal ought to reach now, and a goal
the baseline gets wrong is simply not counted: it has nothing to protect,
changing it cannot be a loss, and whether it is a gain is J1's question, asked
over a corpus with ground truth for every item.

**It could say a goal moved and not where it went.** A night's work was
refused for a change of character that no line in the transcript named, and
the cache genuinely cannot answer it -- the hidden states were collected
walking the baseline's path, so a candidate that diverges at step one leaves
nothing to score its own steps two onward against. So J2 walks the candidate:
`Trial::guards_where` decodes each goal again under the adapter, through the
*same* `walk_goal` the cache came from, which is why that function takes an
`Option<&Dora>` rather than there being a second decode loop to drift from the
first. It costs one prefill per goal against a forward pass per corpus example,
which is a few per cent, and it buys a judge whose verdict names its own
subject. `GuardStep` kept its cache because J3's `logits_finite` reads it; it
lost `chosen`, which was J2's whole mechanism.

**What the walk found on its first run is the argument for all of it.** A
rank-8 adapter over 96 examples, which the old J2 reported as `1/3 still route
where they did`, was rerouting:

    goal: list the files in /ai      -> mv    <- moved, and J2 protects this one
    goal: search for the word godel  -> tree  <- moved, and J2 protects this one

`mv` **changes things**. A goal the machine sets itself unasked had moved from
listing a directory to renaming, and the judge whose entire job is to catch
that could say only that something had moved. J1 vetoed the same variant
independently (chi 2.28 against 3.84), so J2 was never the sole blocker in
anything measured here -- but it was right, and now the transcript says why.

That also exposed a hole in the check beside it, and closing it caught
something worse on the first run. "None of the goals may be routing to a
mutating applet in the first place" walked the **incumbent's** answers, and the
incumbent is not what is being judged. `guards_read_only` asks it of the
candidate's answers as well, over *every* goal and not only the protected ones.
The same variant, on the same trial:

    a goal now reaches 'mv', which changes things
    a goal now reaches 'rm', which changes things
    a goal now reaches 'mv', which changes things

The `rm` is an **unprotected** goal -- one the baseline routes wrongly, so it
is outside J2's count -- and under the old check it was invisible twice over:
not counted because unprotected, and not caught by the mutation rule because
that rule only ever looked at where the *incumbent* sent things, which was
`ls` and `find`. A variant that would have the machine reaching for `rm` on a
goal it set itself passed both halves of that judge.

**And the goals were four spellings of one question**, all expecting `ls`, so
"the machine's character" meant "does it still say ls". Eight distinct
read-only applets now, and widening became safe only once the ground truth
existed: a goal this checkpoint cannot route is not protected, so a broader
list cannot make J2 unfairly strict on a model that routes badly.

Measured while choosing them, and the figure is about the instrument rather
than about the shipped model: on **SmolLM2-135M**, the checkpoint that fits
under QEMU, *twelve of fourteen candidate goals routed to `ls`*. `du`, `cat`,
`hash`, `snaps`, `pwd`, `fsck`, `tree`, `stat` and `same` all collapsed onto
it and only `find` reached its own applet. That is `repair.rs`'s finding
arriving on a second table: an applet's name carries probability mass that has
nothing to do with what the applet does, and `ls` is short, common and first.
Concluding anything about the 0.6B from it would be the small-sample
extrapolation this file warns about elsewhere.

**`godel lib` is the operator path that axis never had**, and its absence is
why `trial_lib` was never driven end to end: it was reachable only from a
rotation four ranked axes away that needs the quiet window. The verification
is three commands after `redqueen 3` offers candidates:

    godel lib     rejected, and a ledger line: axis=lib cell=0 n=8 ... reject
    godel next    lib surprise 66 -- it was 100, and 100 forever
    godel lib     a different candidate: fixed 1 where the first had fixed 0

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

**And what a rollback decides is asserted now, because `clade::reconsider`
calls it unattended.** Three decisions lived inside a hundred-and-fifty-line
function needing an engine, a store and a real lineage to reach: which core to
end on, whether to put a routing rule back, whether to restore a bar. Each has
a recorded history of having been got wrong and each is wrong *silently*, and
sixty-odd claims sat around that function without touching any of them.

They are pure functions of two nodes now -- `core_move`, `rule_move`,
`bar_move` -- in the shape `update::decide` has, and `rollback` calls them
rather than carrying its own copy of the branches. Fourteen claims cover every
state with no model and no disk. The one the shape exists for is the core: a
parent that *said* it had none and a parent that said nothing at all are
different facts, and `CoreMove::Leave` is the third outcome that keeps them
apart.

That hole could not be closed by driving. A real trial that gets rejected never
moves the head, so there is no lineage to roll back along, and a synthetic head
names nodes that are not stored -- which is exactly what `Rebased` reports.

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
the same examples both times, and the curiosity goals recomputed on both sides
against the applet each was declared to want. Two full passes over the corpus, which is the frozen-base trade with a
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

### The loop that ships: a change the machine wrote, built somewhere else

**The kernel cannot compile and cannot sign, and neither is an omission.**
`jit.rs` emits integer functions into a heap page and there is no PE32+ writer
anywhere in the tree; `UPDATE_KEY` is a public point with no private half on
the machine, which is the property that makes the updater worth having. So the
loop is not "the machine rebuilds itself". It is:

```
   machine authors a source change  ->  an envelope in /ai/godel/outbox
              |                                   |
              |                          CI applies it, builds both arms,
              |                          boots each, measures the rail it
              |                          claimed, and answers
              v                                   |
      a signed verdict comes home  <--------------+
```

**`ProposalKind::Source` is the authoring half.** `src/ai/knob.rs` declares a
closed table of tunable constants -- file, symbol, the value it has now, the
values it may take, and **the rail each claims to move** -- so a proposal that
claims nothing is refused before anything is built. Four rows today over
`LEN_B`, `TF_K1`, `IDF_POW` and `MIX`. The patch is generated
mechanically, so it is valid Rust by construction, which is the same argument
`constrain.rs` makes about applet names being unreachable rather than
improbable.

`godel source` picks the next untried point and writes an **envelope**: who is
asking, from which lineage, against which corpus hash, how many tests that
corpus has already paid for, and then the patch. `godel push [--dry]` sends
it. Nothing in the kernel can build it and the shell says so in those words.

**It is deliberately not a ledger line.** Every line in `ledger.txt` is a
verdict by judges that ran here, and a source proposal has none on this side --
writing one would put an unjudged entry among judged ones, which is the "axis
with no judge in front of it" failure the module opens by warning about.
`godel verdict <path>` is the return leg: signature verified **before** the
text is parsed, then a ledger line.

**And that signature is checked against a second anchor, not the update key.**
`src/update/mod.rs` pins `VERDICT_KEY` beside `UPDATE_KEY`, and
`update::verify_verdict` is its own entry point rather than a flag on `verify`
-- two questions that must never be answerable by one call with a different
argument somebody could get wrong.

The separation is about where the private halves are held. `release.yml` signs
kernel images and is reached by pushing a tag, which needs write access to the
repository. `propose.yml` signs verdicts and is reached by
`workflow_dispatch` from the `proposal` function, which **any allowlisted
device can call** -- that reachability is the entire point of the loop. One key
for both would have put the key that ships a kernel to every machine in the
field into a workflow a machine in the field can start. The verdict key's whole
power is to tell one machine that a proposal it made was adopted or refused.

Driven rather than asserted, on the same verdict bytes signed twice, and
re-driven after the key was rotated because the first transcript was about a
point the tree no longer pins. The proposal came out of the machine itself --
`godel source` picked `LEN_B = 0.25`, `godel push --dry` printed the envelope,
and the point in the verdict is the one it named:

    godel verdict /tmp/vbad    (signed by the update key)
      refused: not a signature over this image by this key
    godel verdict /tmp/vgood   (signed by the verdict key)
      737f9c0a moved same on host.retrieval -- ... net under 4
      not adopted, and the ledger says so

    godel ledger 3
      1 h12 parent=root.... variant=737f9c0a axis=source rail=host.retrieval
        moved=same corpus=f330c22c ... reject

**The two blobs are byte-identical except for their 80-byte tails**, which is
what makes this a test of the anchor rather than of the parser: same text, same
digest, same everything the kernel reads before it reaches the signature. Only
which private half signed it differs, and that decides whether a line is
written. 29 boot sections, `diag all` 65 of 65, no boot report, alive
afterwards.

Five claims in `diag update` cover what can be checked without a private half
on the machine, which there is not and must not be: that a verdict key is
pinned at all, that it is not the update key, that both anchors refuse a
malformed signature and a future format identically -- which is what says they
share one implementation rather than two that will drift -- and that the two
verifications consult different points.

`sign.py` signs for either, decided entirely by which private half `--key-file`
is given, so the two live in separate files and separate secrets.

**And a secret nobody here can read is checked before it is used.** A signature
made with the wrong private half is perfectly well formed: nothing in signing
or publishing can tell, and the only thing that ever notices is the machine it
eventually reaches, days later and three systems away, which answers "not a
signature over this image by this key" and files nothing. A correct refusal,
about a configuration mistake it has no way to name.

So `sign.py --anchor SYMBOL --key-file F` derives the public point from a
private half and compares it against what `src/update/mod.rs` pins, signing
nothing -- its own mode rather than a flag, so a caller wanting the check
cannot also produce a signature, and so CI can run it before the thing it
guards. `--check FILE SIG --anchor SYMBOL` verifies the signature itself,
which subsumes it: a key that matches says nothing about whether the signature
over it is whole. Public halves are all either one renders. `propose.yml` runs
the first before signing and the second after, and removes the key through a
`trap` on every path out, the failing one included.

`release.yml` and `experimental.yml` gained the same read-back for the image's
**own** detached signature, which `manifest.py --verify` never covered -- it
reads the manifest and checks the digest in it against the image, so a bad
`.efi.sig` publishes and the refusal arrives at `update stage` on every
machine in the field with nothing in the build to say why.

`verify_sig` and `public_of` came down into `sign.py` with it, and the layering
was backwards rather than merely duplicated: a manifest is a file format and
that is the primitive underneath it, which is what a caller with no manifest
needs. `manifest.py --selftest` still exercises every one, so the move is
covered rather than asserted.

**What it found on its first run is the argument for it.** The pinned
`VERDICT_KEY` had been rotated in the working tree and not committed, so HEAD
pinned a point whose private half exists nowhere -- `verdict.key` derived to
the worktree's point and `update.key` to neither. A kernel built from HEAD
would have refused every verdict anybody could sign, correctly, about a
mismatch made three hours and one commit earlier. The check answers that in a
second where reading the diff does not answer it at all.

**Rails are what "better" refers to.** `bench report` emits one machine-readable
block, `[rail] v1`, one rail per line as `name value unit want`. Thirteen of
them: five graphics, four interpreter, `ai.matmul`, `ai.bpb`, and two memory
bandwidth. `tools/rails.py` compares two blocks with a declared noise floor per
group, the group's **control divided out**, and a third verdict -- `UNSTABLE`,
exit 2 -- for "a control drifted and these two readings do not compare". That
is not a refusal: a "no" from an invalid measurement is as wrong as a yes.

The noise floors are measured across boots of one binary and were inherited
wrong by up to 7x from within-boot figures in this file. `video.rect` and
`core.new` are the controls; `ai.*` and `smp.*` have none, which is a hole and
is named as one.

**And the declared floors are a floor rather than the answer.** Each verify
boot takes *three* rail readings and keeps the second and third, so a
comparison can measure what a build does against itself and use the larger of
that and the table: `--again` hands each arm its own second reading, and a
floor can only ever widen. A quiet day cannot make the judge stricter than the
figure that was measured across three boots and written down.

**The refusal keeps the declared floor, and a driven run is why.** Widening a
rail's floor is right for a verdict and exactly backwards for the
not-comparable check: a run noisy enough to need a wide floor is a run to
refuse, and the first version of this used the wide floor there and turned a
correct "take it again" into four regressions. What the second reading buys
*here* is the other direction -- a build whose own two readings move the
control past the limit does not agree with itself, and nothing can be compared
with it.

**A within-boot spread is not a between-boot spread**, and the same run put a
number on it: `smp.all_cores` moved 1.7% and 12.2% within its own boots and
67.3% between them. For a controlled group the control covers that gap; for
`ai.*` and `smp.*` nothing does, and the only thing that would is a second
*boot* per arm, which doubles what the gate costs. Read a verdict on those two
as weaker than one on a controlled rail.

**Nothing visual is in the transformation space and it is now checked.**
Screenshots are captured and never compared -- there is no image diff anywhere
in this tree -- so a knob under `src/gfx/`, `src/doom/` or `src/port/` would
produce a patch that builds, boots, reads `same` on every rail there is, and
gets adopted having checked nothing about the only thing it changed.
`knob::UNJUDGEABLE` and `knob.py`'s copy of the rule both refuse such a row,
and two claims check that the gate can refuse rather than merely passing
everything.

**`.github/workflows/propose.yml` is the half that can compile.** Two builds
on one runner from one checkout -- baseline as the tree stands, candidate with
the patch applied -- two verify-boots, two rail collections, and
`rails.py judge` against the rail the proposal claimed. Both arms in one job is
not tidiness: two arms on two runners is a comparison of two machines with the
difference called an effect.

`.github/actions/verify-boot` is what made green mean something. It boots the
built image under QEMU with KVM and reads **six** things, not one: no timeout,
the machine still answering afterwards, a tally reading zero failures, no
`FAIL` anywhere including the boot selftests, the boot report empty, and
**the section count and four sections by name**. That last check exists because
its absence hid eleven of them: the first version staged no model, `ai::init`
returned early, and every selftest behind it -- the godel machine among them --
silently did not run while the tally read 59 of 59 and the gate was green.
`hybtest.py --build-only` writes a 353 KB fixture checkpoint so that cannot
happen again.

**The whole loop has been driven end to end on one machine**, and the verdict
it produced is the argument for paired testing arriving as a measurement:

    godel source        src/ai/lex.rs LEN_B = 0.25 (was 0.5)
    godel push --dry    point 737f9c0a, corpus f330c22c, tests 0
    knob.py apply       the constant, checked against what it said it was
    retrieval.py        600 real queries, both arms, per item
    rails.py judge      host.retrieval same: fixed 4 broke 4 of 600
    godel verdict       signature verified, a ledger line, not adopted

The aggregate was **identical both ways**, 62.7%, while eight items moved four
each way. Two percentages would have called that nothing.

**Driven again once every piece above existed**, both arms gated by `diag all`,
three rail readings a boot, and the judge reading `--again`. It refused, and
what refused it is the interesting part:

    video.*         unstable  the candidate's own two readings moved
                              video.rect by 69.2%, so it does not agree
                              with itself
    core.*          unstable  core.new moved -72.2% between the two builds
    ai.bpb          same      +0.0%, inside the 2% floor
    smp.all_cores   same      -54.9%, inside the 132% floor measured today
    host.retrieval  same      fixed 1 broke 2 of 250, net under 4

    J1 claimed  VETO        J2 the rest  pass
    NOT COMPARABLE -- 9 rail(s) had a control that drifted

Three of tonight's mechanisms are visible in that block and each did the job
it was built for. The **self-disagreement** refusal fired on a build whose own
two readings moved the graphics control by 69%, which nothing could say before.
The **measured floor** called `smp.all_cores` at -54.9% noise where the
declared 29% would have made it a regression, so **J2 passed** where it used to
veto on the day rather than on the proposal. And `ai.bpb` moved by exactly
nothing, which is what a deterministic rail on an unchanged model reading an
unchanged history should say.

Both arms passed `diag all` 65 of 65 before anything was judged, which is the
gate working: a candidate that broke something never reaches the rails. The
verdict was signed (267 B), carried in on the NVMe image, verified before it
was parsed, and filed:

    1 h7 parent=root.... variant=737f9c0a axis=source rail=host.retrieval
      moved=unstable corpus=f330c22c
      host.retrieval same fixed 1 broke 2 of 250, net under 4 reject

**One thing that reads oddly and is worth a decision rather than a fix at the
keyboard.** An unstable comparison is filed as `reject`, so a measurement that
did not compare and a change that was judged and refused leave the same verb in
the ledger. `moved=unstable` distinguishes them to a reader, and nothing
automatic reads that field. The honest alternative is a fourth outcome that
consumes no lineage and asks for the reading again, which is what the third
verdict already means one level down.

**The evidence budget is family-wise now.** Every judged comparison at
`bar_in_force()` is a test at p < 0.05; run one nightly for a year and roughly
one adoption in twenty is noise, permanently, by construction. So each trial
debits alpha from a declared series -- `0.05 * 6/(pi^2 k^2)`, which sums to
exactly 0.05 -- the bar rises as it depletes, past the table the loop refuses
outright, and only a new corpus hash refills it.

**`src/ai/clade.rs` chooses where to grow from** and **`src/ai/oops.rs`
chooses how much to spend.** See below; the first is the thing that lets the
loop go back, the second is the thing that lets it not know how big a question
is.

### The CI Godel machine

A second self-improvement loop, resident in GitHub Actions, whose substrate
is git -- which is the content-addressed Merkle DAG `godel.rs` had to build
from scratch in ring 0. A branch head is `/ai/godel/head`, revert is
rollback, the descendants of a commit are the clade. `tools/godel.py`
re-derives the kernel's rules (the alpha SPEND series, frozen-bar epochs,
OOPS doubling, deterministic Thompson clade selection) and its `--selftest`
holds 56 claims, among them: the SPEND table recomputed from the formula
against the kernel's 32 literals, the OOPS bounds as arithmetic, splitmix64
bit-for-bit, and both machines' ledger grammars round-tripped.

**Two machines, distinct authorities.** The kernel machine's ledger is
`/ai/godel/ledger.txt`; the CI machine's is content-named certificate files
under `loop/ledger/entries/` on the `loop/main` branch, fast-forward only,
one adoption at a time under one Actions concurrency group. The CI token
never holds `UPDATE_SIGNING_KEY` and has no `workflows` write, so GitHub
itself rejects any push touching the judge; `main` is reachable only through
the rolling audit PR, which a human merges. Certificates reference **git
tree hashes, never commit hashes**, so history surgery renames nothing that
matters.

**A proposal's identity carries no account state**, and that was re-learned
here the hard way rather than inherited: the first envelope format hashed
`alpha-k` and the minute budget into the point, every trial moved the
account, and the tried-marker walk re-proposed the same constant forever
under fresh names. The kernel's rule is verbatim the fix -- a proposal is
identified by its rendering alone, and `Proposal::render` carries knobs,
never `Budget`. Every account field is a pure function of the ledger and is
recomputed where needed; the certificate records what a judge actually
spent.

**Jobs are a closed kind table** (tune, cleanup, bugfix, test, feature,
rewrite enabled; deps/docs/eval designed and disabled; `event` is
certificate vocabulary for superseded/rollback entries and admits nowhere).
Each row declares scope masks, diff budgets and its judge beyond the common
gate: a bugfix must carry a witness that fails on the baseline arm and
passes on the candidate; a cleanup must delete more than it adds and improve
a cost rail; a rewrite claiming nothing is refused, the knob rule
generalised. Containment is five layers -- admission budgets before a runner
is spent, monotonic section/suite/claim counts in every certificate,
protected and generated paths, `git apply --check` hygiene (no binaries,
symlinks, renames, mode changes), and the machine only ever being able to
ruin `loop/*`.

The workflows: `ci.yml` (the first workflow to compile the kernel on a push
to main in this repository's history; also re-verifies every loop/** push
with a job the loop does not control), `probe-kvm.yml` (temporary; settles
whether ubuntu-latest has KVM, on which everything depends),
`loop-night.yml` (follow main with `superseded` entries where the operator
overrode an adoption -> reconsider -> mark tried BEFORE trying -> stage the
candidate as apply(parent-tree, patch) through plumbing) and
`loop-judge.yml` (propose.yml's two-arm shape generalised, no secrets,
`contents: read`, candidate tree asserted equal to the envelope's derivation
before a boot is spent, `rails.py judge --bar` at the alpha floor in force).
`rails.py --bar` raises the counted-rail threshold and can only ever raise
it.

**The rest of the suite, in one paragraph each.** Four evidence factories
(`evidence-boot/floors/fuzz/sweep`) produce data and never verdicts -- the
fuzz one is the differ idiom host-side, two independent readers over one
stream of mangled envelopes, and every family carries a known-bad seed that
must refuse. The verdict return leg exists at last (`migration 0005`,
`functions/verdict`): ingest verifies the GLADOSIG against the pinned point
in `_shared/verdict_key.js` BEFORE storing, ci.yml asserts that pin equals
`mod.rs`'s via `sign.anchor` on every push, and the offline tests' two
fixtures are the exact blobs a real boot filed and refused. Rung 2's
`godel.py discover` lists 586 candidate constants and says the row-adding
lane waits on the `eval` kind's enablement; rung 4a's `godel.py author`
asks GitHub Models for one patch under a fence contract whose injection
drill is a selftest claim (a diff aimed at `.github/` refuses by name).
`rail none` judges to an explicit refusal naming its missing judge (the
cost rails), so the model lane is drivable end to end without one
unmeasured adoption. `boundary.yml` is the only evaluator lane: epoch gate,
Sane/Moves/Honest over archived anchor pairs (refusing by name while none
are archived), environment-gated token, output a PR to main and never a
push. `.github/RULESETS.md` is the operator's half.

**None of it has run on a runner** -- that requires the push -- and what
could be driven locally was: every subcommand against the real tree, the
candidate re-derivation producing `LEN_B = 0.25` in a tree the worktree
never saw, fsck catching a certificate that lies about its candidate tree,
the tried-walk moving to the next value, the --bar floor turning a chi
23 win into `same` at a bar of 30, 500 envelope mangles with two readers
agreeing and the fixpoint holding, and the verdict verifier agreeing with
the kernel on all three of accept, wrong key and tamper. The crons ship
commented out; a schedule is the last thing a loop earns.

Still owed from the plan, named rather than implied: the kernel
counterpart (`godel verdicts` + the fourth night block, polling before the
trial so a filed verdict cannot move the epoch boundary mid-night), the
cost rails that give `cleanup` its J1, the repair-persist boot-matrix leg,
the anchor-pair harvest that arms Moves/Honest, and `godel.py --verify`'s
kernel-rendered fixture session.

Root certificate bundle, built from the host's store:

```powershell
.\scripts\fetch-roots.ps1          # -List to see what would be exported
```

### Which node to grow from, and how well the machine predicts its own life

Two Phase 7 pieces, and they are the two the loop was missing: a way to choose
where to search from, and something dense to want.

**`clade.rs`: the loop had never made the first choice.** Every trial extended
`head`, so the machine was a hill climber that could not go back -- a lineage
that walked into a dead end spent every later night proposing children of the
dead end, and the only way out was an operator typing `godel rollback`.

The Huxley-Gödel Machine (arXiv 2510.21614) is about exactly that choice and
its *finding* is the useful part: an agent's own score predicts its
descendants' badly, and what predicts them is what its descendants already did.
So the quantity to select on is the **clade** -- every trial at or below a
node, and how many were adopted. `axis_uncertainty` was already a
Laplace-smoothed Beta posterior pointed at axes; this points one at nodes.

An ancestor's clade contains its child's, and that is the mechanism. The head
starts cheap and confident, one trial and one adoption, and gets worse as
refusals accumulate under it; an ancestor carries the same refusals *plus* the
productive stretch before them. `godel clade` on a nine-line lineage:

    head aaaa0002  clade 1 of 7 adopted   draw 0.031
    back aaaa0001  clade 2 of 9 adopted   draw 0.277
    back 00000000  clade 2 of 9 adopted   draw 0.159

The last two have identical counts and different draws: independent samples
from one posterior keyed by node, which is what makes it sampling rather than
a ranking with extra arithmetic.

**Thompson sampling is a coin and everything around it here is built on the
opposite property**, so the draw is seeded from the record it is about -- the
ledger's length and the head's own hash. A later reader with the same ledger
draws the same numbers and reaches the same node. Exploration a reader can
reconstruct, which is the bargain `frontier` makes by walking a declared grid.

Beta is sampled exactly rather than approximated -- Gamma of integer shape as a
sum of exponentials, and the ratio of two -- because the counts *are* integers
and small, and an approximation tuned for large shapes is at its worst on the
arm with one trial, which is the arm whose uncertainty is the point.

Two declared floors. A head with fewer than six trials below it is left alone,
because a Beta with a handful of observations is its own prior wearing a
result's clothes. And one night may unwind at most four, because the decision
is re-made tomorrow and a night that unwound twenty adoptions on one draw is a
night nobody could review before it happened.

`godel reconsider` acts on it and the nightly branch calls it before choosing
an axis. It chooses among the head's **ancestors** and not the whole DAG,
because the only mechanism for moving the machine is `rollback`, which walks
one step to a parent; reaching a sibling means restoring an arbitrary node,
which is `rollback` generalised rather than repeated.

**Driving it found the thing reading it would not.** `godel clade` offered a
backtrack, `godel reconsider` answered "staying", and the head was silently at
the root afterwards -- because `reconsider` calls `ensure_head` and the head
file did not describe the mind that was running. Those two outcomes look
identical from outside and mean opposite things, so `Rebased` is its own answer
now.

**`progress.rs`: every judge in this tree measures routing accuracy.** J1 is
McNemar over applet choices, J2 replays eight curiosity goals, `core_bench` and
`rule_bench` score the same thing again. A binary rail at one bit an item needs
a thousand items before a paired test sees anything, and the whole of what the
loop can want is "pick a better applet more often".

Bits per byte is the dense alternative and the figure is measured: `--task bpb`
reaches t = 6.61 on 256 windows where GSM8K needed 1,319 questions to reach
chi 3.86. Every token is an observation.

Schmidhuber's 1991 signal is a **difference** -- `bits(history at t) - bits(the
same history at t+1)` -- which is two builds, one corpus, one number each, and
that is exactly what `rails.py` already does. So it ships as the rail `ai.bpb`
and the subtraction is somebody else's job that is already done. A rail that
computed its own progress would have to remember what it read last time, which
is a second record free to disagree with the first.

The history is the machine's **own**: the journal the night writes and the
ledger the judges write. A variant that predicts its own life better has
learned something about itself, which is the only reading of curiosity a kernel
with one address space can honestly make. It is not held out and cannot be, so
what keeps the comparison honest is that both arms are handed one text taken
once -- and `rails.py` declares `ai.bpb`'s 2% floor as an assumption about
*that*, since the instrument itself contributes no noise at all.

Log-sum-exp with the maximum subtracted, and the claim that earns its place is
the one the naive form fails: `exp(300)` is infinity in f32, so `inf - inf` is
NaN on exactly the confident predictions a working model makes.

It runs on a **scratch state sized to the window**, not the live one.
`State::new` allocates by `live_cap`, so a full one for the 0.6B is 112 MiB of
KV cache to score 256 positions; a config with `seq_len` cut to the window is
the same state three orders of magnitude smaller. A prefill cannot do this and
the reason is what makes prefill worth having: it is weight-stationary and
materialises only the last position's logits, where this needs a distribution
at every one. So it costs what generation costs -- nightly, never interactive.

`bench bpb` takes the reading and breaks it into parts. On the CI fixture,
whose 256-token byte vocabulary and random weights make it a good negative, it
reads **11.5 bits per byte** -- worse than the 8 a uniform guess would spend.
The plumbing works and the fixture knows nothing, which is what it should say.

### How much a night may spend

**A constant could not be right for two axes and the measurement says so.**
`GODEL_EXAMPLES` was 24, where J1 was arithmetically impossible: five wrong
validation decisions against a requirement of six clean repairs, so the judge
was asking for more repairs than there were wrong answers to repair. Every
night, for as long as the axis had existed. It became 96, which fixed that and
charged every other axis four times over for a problem it does not have --
`lib` judges a library function against a solver and pays 321 seconds of
prepare to reach a verdict it could have reached at 24.

The honest answer is not a better constant. It is to not know, and to search.

Schmidhuber's Optimal Ordered Problem Solver (2004) is exactly this shape of
question, and the arithmetic is the whole of it. Reaching level `L` by trying
every level below it costs `base * (2^(L+1) - 1)`, which is under twice what an
oracle that already knew `L` would have spent: **not knowing costs a factor of
two, forever, whatever L turns out to be.** The other half is that the schedule
never commits -- half the nights extend at the axis's level and half start at
the base -- which bounds the average however high the level has climbed.
Multiply the two and the whole scheme wastes at most four times an oracle's
budget, which is the constant the module exists to be able to state. Both
halves of that are asserted as arithmetic rather than cited.

**What raises a level is only a trial the budget could not have decided.**
`Certificate.wrong` is the ceiling on repairs available and
`clean_fixes_needed()` is what the bar asks; a trial whose ceiling is below the
requirement did not fail, it was never asked. A trial that had the evidence and
was refused anyway is a *candidate* failure, and doubling for it would be the
loop spending more and more on the same ground because it did not like the
answer -- so any verdict reached on sufficient evidence drops that axis back to
the base. And a line that says nothing about its ceiling is not starved: "did
not say" is not "could not pass", and reading it as the latter would raise
every axis on the strength of the early history being silent.

The level is **the largest budget this axis starved at, and the smallest it
decided at**: if it has decided something above every starvation, that is the
level that works and there is nothing left to double for. It is per axis,
because one question being unanswerable at 24 examples says nothing about the
next, and a shared counter would charge `lib` for `adapter`'s problem.

The first version counted the *trailing run* of starved trials, and it
converged only by a coincidence: the `Fresh` nights kept injecting starvations
at the base, and it would have stopped working the day they stopped failing. A
schedule whose convergence depends on something continuing to go wrong is a
schedule that fails silently.

Reading it directly needed a field the ledger did not have. **`ex=` is the
subsample a trial was given**, where `n=` is validation decisions -- derived
from the budget and not invertibly -- so a schedule that inferred one would be
a second account of the record free to disagree with it.

It is **scoped to one corpus**, or it is a ratchet: an axis that starved at 192
once would stay there forever, including after the corpus grew enough that the
base would do. The ledger already records which body of evidence each line was
paid for out of, for the family-wise budget's sake, and the same field answers
this.

The half is a function of the ledger's length, so a later reader can say which
half any past night was on -- the objection `axis_counts` already makes about a
counter in its own file.

The journal line carries what the night was allowed to spend beside what it
got for it. A refusal at 24 and a refusal at 192 are different facts and
looked identical.

**Driven, which nothing about the nightly path had been.** With the clock wound
to 03:00 and SmolLM2 loaded, forced ticks until one fell in the sleep branch:

    [t35 +33s] godel: hour 3, extending at n=24,
               variant ca6f18a4 rejected (fixed 2, broke 1, goals 1/3)

    1 h3 parent=root.... variant=ca6f18a4 axis=adapter corpus=f330c22c
      cell=4 n=15 pred=win J1[fix=2 broke=1 wrong=5 ex=24 chi=0.00
      net repair below the floor no] J2[goals=1/3 no] J3[ok] ep=20
      J4[r=8 kib=24 ok] reject

Every piece is in those two lines. `extending at n=24` is the schedule taking
its Extend half at the base, where the old constant would have spent 96.
`ex=24` is the budget in the record. `wrong=5` against a requirement of seven
clean repairs makes this a **starved** trial by definition, so the next
extending night on this axis runs at 48 -- the doubling, demonstrated on a
real verdict rather than on a fixture. No `grew from N back`, because with one
node there is nothing to choose between and `reconsider` said so.

Two things that cost a run each to find. The forced tick has to land in the
*sleep* branch, and it only does so once an episode is in cooldown and the
agent is genuinely idle, so `agent stop` needs a command after it before the
next tick. And **the CI fixture is a hybrid**: `hybtest.py` builds one to
exercise the Qwen3.5 path, the trainer refuses hybrids by design, so a nightly
trial on a CI runner answers `refused: the model is a hybrid the trainer will
not touch`. That is correct and is said in `verify-boot` now -- that job gates
builds, and the loop runs where there is a dense checkpoint.

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

**The image carries the kernel, the weights and their licences, and that is
the whole list.** Ported engines are measurements: `src/doom/` exists to find
out whether software written somewhere else survives being brought over, and
`tools/xash.py` and `tools/guest/spin.c` exist to find out whether a binary
this kernel did not compile can reach the screen. What they establish is a
fact about the kernel, and none of them is a thing to install on somebody's
laptop.

Two mechanisms, and they cover different halves. `payload.py verify` refuses a
file the manifest does not name, which covers CI because it runs before
`mkiso.py`. The hand-run case had nothing, so `mkiso.py` now reads the union of
`payload/*.txt` and refuses to place anything outside it -- an **allowlist**,
for the reason `eval.rs` gives about `BUILTINS`: a denylist naming `xash*` and
`*.wad` grants by default, and the first thing it misses is a name like
`libref_soft.so` with nothing in it saying what it belongs to. `--allow NAME`
is the deliberate exception, and a file that genuinely belongs gets recorded
with `payload.py` instead.

The licence half is the durable reason rather than the scope half.
Xash3D-FWGS is GPL-3.0-or-later and this kernel is not, so shipping it on the
install image puts obligations on the whole disc that nobody has decided to
take on; Half-Life's own data is Valve's, under exactly the rule that keeps
`DOOM1.WAD` out of this repository. Neither is a thing to discover after a
release is cut.

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

### Page rights, and whether they are enforced

Two bits decide whether a permission means anything here, and neither was on.

**`CR0.WP` is on now.** Without it a write from ring 0 ignores the R/W bit
entirely, and every instruction in this kernel runs at ring 0, as does every
guest binary. A page marked read-only without `CR0.WP` is a page marked
read-only in a comment. It also catches a class of kernel bug for free once
anything is read-only: writing through a stale pointer into constant data
becomes a fault at the write rather than a wrong answer somewhere later.

**`EFER.NXE` is on**, gated on `CPUID.80000001H:EDX[20]` for the reason
`dev::power` gates its MSRs -- writing a reserved bit of `EFER` raises #GP and
every vector but `#BP` here is fatal. Boot prints `page rights  wp=1  nx=1`.

Turning both on changes nothing the day it happens, which is the point:
everything is mapped writable and nothing had ever set bit 63, so the map means
exactly what it meant a moment earlier.

**The identity map is built from 2 MiB pages, and that is why this was not
free.** One entry per two megabytes with no page table to walk is exactly right
for a map that never changes. Changing the rights on one 4 KiB page inside it
means the 2 MiB entry has to stop existing first, so `split_large` replaces it
with 512 entries covering the same bytes with the same flags, including
cacheability, so uncached device memory survives the split. A reader who did not
know it happened would see no difference, which is what makes it safe to do
underneath a running kernel.

`paging::protect(at, len, perm)` applies rights per page and `invlpg`s each one.
Per page rather than a CR3 reload, because a reload flushes every translation in
the machine and this gets called with a guest's whole heap. Skipping it is the
failure that matters: the old translation stays cached and the change is
silently unenforced. `paging::query` reads the tables back rather than a shadow,
so it cannot disagree with the hardware.

**`diag paging` is 13 claims and one of them faults on purpose.** It makes a
heap page read-only, writes to it inside `cpu::recover::guard`, and requires the
fault to arrive and the write not to land. Everything else about page tables can
be asserted by reading them back; enforcement cannot, and a permission nobody
has watched the processor refuse is a permission written in a comment. The check
puts the page back afterwards, because leaving a read-only page in the heap
poisons whatever asks for it next.

### Ring 3, in one address space

Stage 1. A guest runs at CPL 3 and the identity map stays exactly as it was:
no second set of page tables, no CR3 swap, no TLB shootdown to design. What
separates the guest from the kernel is the U bit, and it is on the guest's own
pages and on nothing else in the machine.

**The GDT grew from five entries to eight, and their order is dictated rather
than chosen.** `sysret` takes no selectors. It derives them from
`IA32_STAR[63:48]`: stack is that plus eight, 64-bit code is that plus sixteen,
both with RPL forced to 3. So the layout is a 32-bit code descriptor nobody
uses, then user data, then user code, in that sequence. Tidying it gives a
`sysret` that lands in a data segment. `TSS.rsp[0]` is set for the first time,
because an interrupt taken in a guest would otherwise push its frame onto the
guest's own stack.

`diag gdt` checks the bit fields rather than loading anything, because every
one of these is a silent triple fault: an instant reboot with nothing printed.

**The U bit is ANDed down all four levels**, so a leaf marked user under a
directory that is not stays unreachable. `protect` therefore opens the PML4,
PDPT and PD entries along the way when the leaf is going to be user-accessible.
That is safe for the same reason it is necessary: every other page under those
directories still has a clear U bit of its own, and the leaf is the gate.

In: `iretq` with a hand-built ring-3 frame. Out: `sysretq`, which was the wrong
instruction while guests ran at ring 0 and is the right one now. The selectors
are literals in the assembly because `global_asm!` cannot see a Rust constant,
so a claim asserts the literals against the constants.

**Two real bugs surfaced only because the guest moved to ring 3.**

`sys_mmap` never marked what it handed out as user-accessible. The loader opens
the image, stack and break before the guest starts, and a mapping made after
that is covered by none of them. At ring 0 the U bit meant nothing so this was
invisible; the first ring-3 guest to call `mmap` took a protection violation
reading its own memory.

And **a `static mut` written only by assembly can be folded away.**
`GLADOS_HOST_RSP` is written by `glados_enter_guest` and by nothing in Rust, so
the optimiser sees an initialiser of zero, no writer, and is entitled to
constant-fold every read to `false`. It did. The question "is a guest running"
is an `AtomicBool` that Rust both writes and reads now, and reading the parked
stack pointer from Rust is gone.

**A guest fault kills the guest and nothing else.**

    linux run /tmp/wild
      ring 3, one address space -- only its own pages carry the U bit
      killed by fault 0x0e after 0 syscall(s), machine intact

The guest reached for `0x1000`, which is mapped, kernel-owned and has a clear
U bit, took a `#PF` at ring 3, and was ended. `mem` answers afterwards and
`diag paging` still passes, which is the part worth checking: at ring 0 the
same fault stopped the machine, because a guest sharing an address space with
the kernel might already have corrupted anything. At ring 3 the kernel is
intact by construction, so ending the guest is the honest response.

**Getting there took finding an ABI bug that looked like a ring-3 bug for a
long time.** `syscall::kill` was calling `glados_leave_guest` through its
`extern "sysv64"` declaration. This target is Windows-ABI, so an ordinary Rust
function is Microsoft x64, where `xmm6`-`xmm15` are non-volatile; `sysv64`
treats them as scratch. The compiler therefore spilled all ten across the call,
a 160-byte `movaps` prologue wanting the stack 16-byte aligned, and on the
stack a guest fault arrives on it is not. A misaligned `movaps` raises
`#GP(0)`, which is exactly the fault that was stopping the machine.

`exit_group` never met it, because it leaves from `glados_syscall_dispatch`,
which is already `sysv64` and so has nothing to preserve. That asymmetry is
what made it look like ring 3 was at fault. The longjmp is written out inline
now: no prologue, nothing spilled, no alignment it cannot have.

What settled it was `llvm-objdump` on the image at the faulting RVA, which the
fault reporter prints. Reasoning about segments and stacks produced four wrong
hypotheses first; one disassembly produced the answer.

### Running a binary this kernel did not compile

`src/linux/` is stage 0 of multi-binary support, and it is a **measuring
instrument rather than a loader**. The expensive unknown in porting the Linux
ABI is not the loader, which is a weekend; it is that Linux has no
specification you can test against, so a subtly wrong `mmap` flag surfaces as a
crash in a different subsystem an hour later. gVisor needed 237 of Linux's ~350
calls to run containers. Before committing to a privilege model or a syscall
subset, the thing worth owning is a trace of what a real binary actually asks
for.

**`syscall` traps from ring 0, and that is what makes this possible without
building a userspace first.** The instruction loads `rip` from `IA32_LSTAR` and
`cs` from `IA32_STAR[47:32]` whatever the current privilege level, so a guest
already at CPL 0 traps exactly as one at CPL 3 would. `STAR[47:32]` is set to
`KERNEL_CS`, and `syscall` derives SS as that plus eight, which is `KERNEL_DS` --
so the guest lands on the descriptors it was already running under. That is not
luck; it is what makes a same-ring trap cost nothing to set up.

Three things a real syscall gets that this does not, each written down in
`syscall.rs` rather than left to be discovered:

- **No stack switch.** There is no privilege transition, so no `rsp0` reload.
  The stub swaps to a stack of its own before touching anything, so a guest
  that wrecked its stack pointer still reaches the dispatcher.
- **That stack is one static, so the handler is not reentrant.** `IA32_FMASK`
  clears `IF` on entry and stage 0 runs one guest with no threads. Both of
  those stop being true later.
- **A hostile guest is not contained.** At CPL 0 it can `wrmsr` and move
  `LSTAR`, or `mov cr3`, or `cli`. Stage 0 contains bugs, not malice, and a
  fault *is* the measurement.

Return is **not** `sysret`, which forces CPL 3 on the way out and would drop
the guest into a privilege level this kernel has no descriptors for. The stub
restores flags from `r11` and jumps to `rcx`, which is what `sysret` does minus
the privilege change. `exit_group` is a longjmp: `glados_enter_guest` parks the
host's stack and callee-saved registers, and `glados_leave_guest` restores them,
because there is no unwinder here and returning normally from a process that
has exited is not a thing that can be expressed.

Both of the loader's original refusals are gone, and each went the same way:
the reason given was true of what the kernel *knew* rather than of the machine,
and the fix was to make the thing knowable.

- **Fixed-address executables.** An `ET_EXEC` insists on its own addresses,
  classically `0x400000`, and this kernel is identity-mapped with one address
  space, so those are real physical bytes. Nothing could answer "does anything
  own four megabytes at four megabytes" until `mem::fixed` did. See the
  placement section below.
- **Dynamically linked binaries.** The entry in the header is not where
  execution starts, `ld.so` is, and loading one as though it were static jumps
  into a PLT stub nobody filled in. The answer is not to implement linking, it
  is to load the linker. See below.

`tools/mkelf.py` builds the fixtures by hand rather than by compiling, for the
reason `mkwad.py` generates art: the negatives need to differ from the positive
in **exactly one field**, and no compiler will emit a dynamically linked binary
otherwise identical to a static one. `--verify` reads each back with a separate
parser.

```powershell
.\tools\venv\Scripts\python.exe tools\mkelf.py out\hello.elf
.\tools\venv\Scripts\python.exe tools\mkelf.py out\dyn.elf --kind dynamic
.\tools\venv\Scripts\python.exe tools\mkelf.py out\fixed.elf --kind fixed
.\tools\venv\Scripts\python.exe tools\mkelf.py out\mem.elf --kind memory
.\tools\venv\Scripts\python.exe tools\mkelf.py out\rogue.elf --kind rogue
.\tools\venv\Scripts\python.exe tools\mkfat.py .qemu\nvme.img out\hello.elf out\dyn.elf out\fixed.elf
.\tools\venv\Scripts\python.exe tools\drive.py --qemu-extra "-accel whpx -cpu max" `
  "initiative off" "fat get /HELLO.ELF /tmp/hello" "linux run /tmp/hello"
```

Measured, and every figure cross-checks against what `mkelf.py` printed:

    [linux] 185 byte(s), 1 segment(s), 185 byte span at 0x17f5000, entry 0x17f5078
    hello from ring 0
        1 write            0x1 0x17f50a7 0x12 -> 18
      231 exit_group       0x5 0x17f50a7 0x12 -> 0
      exited 5 after 2 syscall(s)

Entry is base + 120, the buffer is base + 167, `0x12` is the message's 18 bytes,
and 5 is the exit code it was built with.

**Memory: `brk`, `mmap`, `munmap`, `arch_prctl`.** The four a static musl
binary reaches before `main`, and three of them carry a decision worth knowing.

`brk` **never returns an error**, because Linux's does not. It answers the
resulting break, which on failure is the *unchanged* one, and libc decides it
failed by comparing that against what it asked for. Returning `-ENOMEM` instead
would hand musl `0xFFFFFFFFFFFFFFF4` as a heap address and it would believe it.
One honest deviation: the break is a separate region rather than the bytes
after the image, because in an identity map those bytes belong to whatever the
frame allocator gave them to. Every allocator asks `brk(0)` and grows from the
answer, so nothing notices -- but a program assuming adjacency would be wrong.

`mmap` serves anonymous private memory and refuses three things for reasons
about this machine rather than about the arguments: a file-backed mapping needs
an fd table that does not exist, `MAP_FIXED` needs an address one address space
can promise (the same objection that makes the loader decline `ET_EXEC`), and a
zero length is `EINVAL` because it is `EINVAL` on Linux. `munmap` refuses a
partial unmap rather than approximating it -- splitting one allocation in two
is not something the heap underneath can express. A guest that exits still
holding mappings is the ordinary case, so teardown is where they are actually
reclaimed.

**`arch_prctl` is where "the guest and the kernel share everything" stops being
an architectural note and becomes a specific call that has to say no.**
`ARCH_SET_FS` is honoured, because that is how thread-local storage works and
musl calls it before `main`; the base goes into `IA32_FS_BASE` and the kernel's
own value is parked by `run` and restored at teardown, since the register
belongs to the machine and not to the guest.

`ARCH_SET_GS` is **refused**, and not out of caution. `cpu::percpu` points GS at
each core's own block and `gs:[0]` is how the allocator discovers which core it
is billing. There is no privilege boundary here to stop a guest overwriting it,
so a guest setting GS would leave the next kernel allocation reading its
thread-local storage as a per-core structure. Reading GS is refused too, for
the smaller reason that it hands a kernel pointer to code with no business
holding one.

That restore is checked rather than assumed: `diag census` -- whose first claim
is "per-core storage is up, so allocations can be attributed" -- passes on a
boot where a guest has already set FS.

Measured, on the `--kind memory` fixture, which uses every result rather than
merely receiving it (the break is written to, the mapping is written and read
back, and FS is set through one call and read through another):

    brk, mmap and arch_prctl all answered; FS read back
       12 brk              0x0 ...        -> 46190592
       12 brk              0x2c0e000 ...  -> 46194688
        9 mmap             0x0 0x2000 0x3 -> 25141248
      158 arch_prctl       0x1002 ...     -> 0
      158 arch_prctl       0x1003 ...     -> 0
       11 munmap           0x17fa000 ...  -> 0
        1 write            0x1 ... 0x34   -> 52
      231 exit_group       0x7 ...        -> 0
      exited 7 after 8 syscall(s)

The break grew by exactly 4096, the mapping came back page-aligned, and exit 7
is the branch the guest only reaches if the FS base it set came back through
the call that reads it.

**Every guest pointer is bounds-checked, and that is the whole of the
hardening.** A guest at ring 0 shares an address space with the kernel, so a
pointer it passes is not merely possibly-invalid: it is a pointer at anything
at all, the page tables and the model's weights included. `Space` records every
range the loader handed out (image, stack, break, and each live mapping) and
`owns` is asked before any syscall reads or writes through a guest address.
`EFAULT` otherwise, as Linux has it.

It cannot stop a guest dereferencing a bad pointer *itself*, and nothing at CPL
0 can. What it stops is the kernel doing it on the guest's behalf, which is the
difference between a crashed program and a corrupted kernel.

Two calls needed it and neither had it: `write` built a slice straight from
`rsi`, and `arch_prctl(ARCH_GET_FS)` wrote eight bytes wherever `rsi` pointed,
which is a kernel-corrupting primitive handed to the program.

`--kind rogue` proves it from the guest's side rather than only in a claim. It
asks for `0x1000`, which is deliberately a *real, mapped, kernel-owned* page:
not a wild address that would fault on its own, a valid one the guest has no
business naming. An unchecked kernel prints whatever lives there.

    both wild pointers were refused with EFAULT
        1 write        0x1 0x1000 0x10 -> -14
      158 arch_prctl   0x1003 0x1000 0x10 -> -14
        1 write        0x1 0x17f511f 0x2c -> 44
      231 exit_group   0x9 -> 0
      exited 9 after 4 syscall(s)

**`sys_write` used to swallow bytes and report success.** `for chunk in
core::str::from_utf8(bytes)` iterates a `Result`, so the body ran zero times on
the error arm: a guest writing Latin-1 or raw bytes printed *nothing* and still
got the full length back. That is the worst shape this call can take, because
the guest has no way to find out. Lossy conversion now.

Three more gates, each closing an arithmetic hole a hostile file could reach:
a segment whose `vaddr + memsz` overflows is refused at parse, where the field
is read; an image span over 64 MiB is refused by the loader with a reason about
the file rather than about the machine; and `mmap` caps its length, because
page rounding multiplies and would wrap for a length near `usize::MAX`, handing
back a small allocation for a huge request.

`install` moved from `load` to `run`. A guest that was loaded and never run
would otherwise leave `SPACE` naming memory freed when its `Guest` dropped, so
the next thing to consult it would be reading a dangling range it believed it
had verified.

**`mprotect` is real now, and making it real meant page rights existed.** It
was unimplemented on purpose, because every page in this kernel was writable
and executable: answering 0 would have claimed an enforcement that did not
exist and musl's guard pages would have guarded nothing, while refusing stops
any real allocator. Both answers were lies. See the page-rights section above
for the third option.

`PROT_NONE` clears the present bit, so the page genuinely faults. That is what
a guard page is for, and it is also why `reachable` had to grow: a guest that
hides a page from itself and then hands the kernel a pointer into it now gets
`EFAULT` rather than taking the machine down on two entirely legal calls.
`--kind protect` proves exactly that from the guest's side:

        9 mmap        0x0 0x2000 0x3  -> 46436352
      158 arch_prctl  0x1003 0x2c49000 -> 0
       10 mprotect    0x2c49000 0x1000 0x0 -> 0
      158 arch_prctl  0x1003 0x2c49000 -> -14
      exited 11 after 6 syscall(s)

`teardown` puts the rights back before freeing. A guest is free to exit having
mprotected its mappings to something the heap cannot reuse, and handing a
read-only or absent page back to the allocator would poison it for whatever
asks next, with the symptom appearing in an unrelated subsystem hours later.

**Every unimplemented call is recorded and refused with `-ENOSYS`, and that is
the instrument.** A run ending in `-ENOSYS` on call 47 has said which call to
implement next, which is the question stage 0 exists to answer. The trace is
bounded at 1024 entries, because what matters is *which* calls appear rather
than how often.

`linux` reports whether the trap is armed, `linux run <path> [args...]` loads
and runs, `linux trace` prints what the last guest asked for, `linux libc` says
which interpreters are installed, and `linux env NAME=VALUE` adds a variable to
what the next guest is handed. `diag linux` is 134 claims.

**The trace records the path, not only the pointer.** A row read `257 openat
0x2d23faa 0x0 0x0` and the useful half of it was in the guest's memory, so
finding out which file a run failed on meant reading a register dump against a
disassembly. `Call` carries the first 48 bytes of a path argument now, filled
where the call is recorded rather than where it is dispatched, and `path_arg`
is the one table saying which register holds one. It named `newfstatat` as the
call that had silently dropped its flags, in a single run.

### It runs busybox

An unmodified static binary, fetched from busybox.net and touched by nothing
here, running at ring 3.

    linux run /tmp/busybox uname -a
    GLaDOS glados 1.3.5 ring 0, with guests at ring 3 x86_64 GNU/Linux

    linux run /tmp/busybox hexdump -C /tmp/lines.txt
    00000000  61 6c 70 68 61 0a 74 68  65 20 71 75 69 63 6b 20  |alpha.the quick |
    00000010  62 72 6f 77 6e 20 66 6f  78 0a 62 65 74 61 0a 6a  |brown fox.beta.j|
    00000020  75 6d 70 73 20 6f 76 65  72 20 74 68 65 20 6c 61  |umps over the la|
    00000030  7a 79 20 64 6f 67                                 |zy dog|

    linux run /tmp/busybox sha256sum /tmp/lines.txt
    6ecb6686ad1673a0f3c021fc356890758853980fd6465b3b5a04fd859682733a

**That digest is the one the host computes over the same file**, which is the
first end-to-end check of this path against an implementation nobody here
wrote: the bytes went from FAT into the namespace, out through the projection,
through busybox's own hash at ring 3, and back through `writev`.

**`ET_EXEC` is placed rather than refused, and busybox is why.** Its prebuilt
is non-PIE with an entry at `0x4038b1`, and so is nearly every prebuilt static
binary in the world, so the blanket refusal was not a corner case -- it was
most of the software this loader exists to run. See the placement section
below.

**Sixty-two syscalls, and not one of them was guessed.** Sixteen applets were
swept under musl and twenty-eight more under glibc, every gap they named was
implemented, and nothing else was. That is what stage 0 was built to make
possible.

The second sweep is the one worth reading, because the surface it named was
*small*: twenty-eight applets across two boots left exactly three calls
unserved, and each was answered from something this machine knows rather than
from a plausible constant. `time` reads the RTC. `sysinfo` reports uptime and
the kernel heap, and leaves the three load averages **zero rather than
invented**, because nothing here samples a run queue and a fabricated number
is read by whatever graphs it. `sched_getaffinity` answers the cores that came
up at boot -- which is a question about permission rather than about
capability, so it is honest even though nothing can yet schedule a guest onto
a second core -- and returns the *bytes it wrote*, which is what a libc uses
to know how much of a larger `cpu_set_t` it must clear itself.

`rseq` is refused on purpose and is the one refusal in the list. It is a
per-thread structure the kernel writes to from the scheduler, and answering 0
without doing that gives glibc a sequence number that never moves. `-ENOSYS`
is a value glibc's own startup handles; a stale `rseq` area is not.

The shape of the finding was the useful part twice over. `ls` ran perfectly on
its first attempt -- opened the directory, walked it with two `getdents64`
calls, `lstat`ed every entry, exited 0 -- and printed nothing at all, because
everything it had to say went through `writev`. A program that works and is
silent is the worst shape a missing syscall can take. And `hexdump` reported
"Function not implemented" about a file it was holding open, because it does
`dup2(fd, 0)` to read its input as stdin.

**A guest may write, inside `/tmp`.** Writes were refused everywhere on the
grounds that a write to a content-addressed store is a new root hash, so an
unrestricted `O_WRONLY` routes a guest binary around every gate `sysbox` puts
in front of the shell. The reason was sound and what it argued for was a jail
rather than a refusal. `/tmp` because that is already the scratch area, and
everywhere else is `EROFS` -- checked on the *resolved* path, so a relative one
cannot be written to climb out, which works because `resolve` refuses `..`
outright.

    cp /tmp/lines.txt /tmp/copy.txt  then  ls /tmp
    busybox  copy.txt  lines.txt  newdir
    mkdir /ai/nope
    mkdir: can't create directory '/ai/nope': Read-only file system

Writes buffer in the body and commit on the **last** `close`, because the store
is keyed by content: every commit rewrites the whole blob and gives it a new
address, so a program writing a kilobyte a byte at a time would leave a
thousand objects behind. The `Rc` count is what "last" means. `teardown`
flushes as well, since exiting without closing is the ordinary case.

**A descriptor's body lives behind `Rc<RefCell<..>>`, and that is what makes
`dup` correct.** On Linux a duplicated descriptor shares one open file
description, so the two numbers share a cursor. The body used to sit inside the
`Fd`, where `dup` could only copy it, and two independent cursors is a program
reading everything twice. `Rc` and not `Arc`, for the reason `Interp` gives
about its functions: one guest, one task, nothing crossing a core.

**The environment is five variables and each is a fact rather than a default.**
It was empty, which is not neutral: `sh` resolves through `PATH`, and a program
with no `HOME` writes its dotfiles into the working directory. `TERM=dumb`
because `ioctl` already says there is no terminal, and `PWD=/` because there is
no `chdir`.

`nanosleep` spins on the timer tick, because there is no guest scheduler to
block against, so it costs the CPU it is not using.

**Signals used to be accepted and never delivered, and that paragraph is worth
keeping because of how it stopped being true.** It read: "nothing here can
raise a signal at a guest: no other process to send one, no terminal to
generate one, and a fault ends the guest rather than being offered to it." The
reasoning was sound and `fork` falsified the first clause. There are other
processes now, so a child exiting is an event its parent is owed, and `kill`
has somebody to talk to. See below.

### Signals

`src/linux/signal.rs`. `rt_sigaction`, `rt_sigprocmask`, `rt_sigreturn` and
`kill`, with delivery **on the way out of a syscall** -- the one moment the
guest's whole register state is already in a `Frame` the kernel owns, its stack
pointer is parked in `GLADOS_GUEST_RSP`, and the return is about to `sysretq`
somewhere. Redirecting it into a handler costs three stores and no new
assembly. Delivering from the timer would mean building a frame around an
interrupted ring-3 context, which is a second entry path to keep in step with
the first.

The price is stated rather than hidden: **a guest that makes no syscalls
receives no signals.** A program spinning in a loop cannot be interrupted,
which on Linux it could. The run deadline still ends a runaway.

Three refusals that are decisions:

- **`SIGKILL` and `SIGSTOP` cannot be caught, blocked or ignored.** Not a
  courtesy -- `SIGKILL` exists so there is one thing a process cannot argue
  with, and a kernel that let a handler take it has removed the only guarantee
  the call makes.
- **A handler with no `sa_restorer` is not entered.** The `ret` ending it would
  take whatever the stack happened to hold. glibc always supplies one; a
  hand-written program that does not gets its signal dropped rather than a wild
  jump.
**The frame is on the guest's stack, in Linux's own layout.** It was in the
kernel first, one deep, which served an ordinary handler and served Wine not
at all -- Wine reads `uc_mcontext` to find out where a fault happened and what
the registers were, and a handler cannot read a context it cannot reach.

`sigcontext` is 256 bytes with a fixed register order, `ucontext` is 304, the
whole `rt_sigframe` is 440 with `siginfo` on the end. Those are an **ABI rather
than a choice**: a handler compiled against the real header reads
`uc_mcontext.rip` at a fixed offset, and a frame one field short is not a
smaller frame, it is a different structure with everything after the gap
misread. 128 bytes of red zone are stepped over, and the frame lands at 8 mod
16 so the handler starts with its stack exactly as a `call` leaves it.

Two things fall out. **Signals nest**, since each delivery builds its own
frame. And `rt_sigreturn` reads its state back *out of that frame*, so a
handler that edits `uc_mcontext` changes where the program resumes -- which is
not a curiosity, it is the mechanism Wine's fault emulation runs on.

`rt_sigreturn` masks the flags it will restore. `sysretq` loads them from
`r11`, so a frame claiming `IF` clear or `IOPL` 3 would be a guest choosing its
own interrupt state, which is a way out of ring 3 that does not involve a
syscall.

The fixture tests exactly that mechanism rather than a flag: **the handler
writes to `uc_mcontext.rax` and returns, and the program checks `rax`
afterwards.** Four things have to be right at once for that to work -- the
handler receives a real `ucontext` in `rdx`, the offset is Linux's 144, the
frame is on the guest's own stack and writable, and `rt_sigreturn` reads back
out of it rather than out of a kernel copy. There is no arrangement of
three-right-one-wrong that passes.

    signal: the handler edited uc_mcontext and it stuck
      exited 9 after 5 syscall(s)

`SIGCHLD` is the first signal this machine can honestly raise, and it is
ignored by default -- which is why a parent that installs no handler is not
killed by its own children finishing.

Measured, on `mkelf.py --kind signal`:

    13 rt_sigaction  0xa 0x8010000191 0x0 -> 0
    62 kill          0x1 0xa 0x0 -> 0
    15 rt_sigreturn  0xa 0x0 0x0 -> 0
    signal: the handler ran and rt_sigreturn put it back
      exited 9 after 5 syscall(s)

Exit 9 needs both halves and neither can fake the other. A flag only the
handler writes proves it ran; *reaching the comparison at all* proves
`rt_sigreturn` restored `rip` and `rsp`, since the handler ends in `ret` onto
the restorer. The flag's address is held in `rbx` across the signal on purpose,
so a `rt_sigreturn` that lost the callee-saved registers reads it from
somewhere else and fails rather than passing quietly.

**`fork`, `execve` and `wait4` have landed**, and the two paragraphs that used
to sit here are worth keeping in summary because of the shape of how they went
stale. The first said the three calls were "not a syscall away: `fork` needs
two address spaces, and one address space is the founding claim of this system
rather than a shortcut it took." The second said that had become half true,
because `src/mem/space.rs` existed and what remained was *placement*.

Both are now history. `src/mem/space.rs` gives a guest its own page-table root,
placement is solved by mapping a fixed image rather than placing it, and the
three calls are in the dispatch table alongside threads, futexes and signals.
The surface is 93 calls of Linux's roughly 350, counted from the match in
`glados_syscall_dispatch` rather than remembered:

    sed -n '/fn glados_syscall_dispatch/,/^}/p' src/linux/syscall.rs | grep -oE '^ +SYS_[A-Z0-9_| ]+=>' | grep -oE 'SYS_[A-Z0-9_]+' | sort -u | wc -l

The lesson the pair of them teaches is the one to keep: **a limitation stated
as a founding claim is still a limitation, and it will be removed by somebody
who did not read the claim as permanent.**

### A second address space

`src/mem/space.rs`, and the thing it removed was an assumption rather than a
line of code. `paging::activate` had exactly one caller in the whole tree,
`main.rs`, at the moment the identity map replaces the firmware's -- so
"can this machine switch address spaces at all" had never been asked, and
`fork` was refused on the strength of an answer nobody had measured.

Two pieces, and the order between them is the design.

**Sharing.** `Space::sharing_kernel` copies the kernel's 512 top-level
entries, which are *pointers* to the PDPTs it already built, so both roots walk
the identical tables underneath. Exactly one copy of every mapping exists, a
`protect` through one root is visible through the other because it is the same
entry, and nothing can drift. That is what made the switch safe to prove on its
own before anything depended on it.

**Divergence.** `map_page` gives a space a mapping the kernel's root does not
have, which is what a process actually needs. It is confined to `WINDOW`
(512 GiB, PML4 entry 1) **and the confinement is the entire safety argument**:
`build_identity_map` hangs everything off entry 0, so every address the kernel
maps has a top-level index of zero, and anything at or above the window is
unmapped in every root that has not asked for it. A mistake there faults on an
address nothing owns instead of quietly landing in the heap.

A mapping below the window is **refused**, and that refusal is the point. The
top-level entries were copied, so they point at the kernel's own PDPTs:
creating a table under entry 0 would create it inside the kernel's map, every
space would see it, and the "private" mapping would not be private at all.

Three details that are silent when wrong, each with a claim:

- **The root has to be identity-mapped**, because CR3 takes a physical address
  while every write to the table goes through a virtual one. It is, because the
  heap is inside the identity map, the same property `cpu::code` leans on to
  execute from a heap allocation. Checked rather than assumed, since the day it
  stops being true the processor walks whatever lives at that physical address.
- **`Drop` restores CR3 before it frees.** Freeing a table the processor is
  still walking hands the allocator memory the next translation will read. The
  fourth instance in this tree of "put it back before giving it away".
- **The U bit is set on intermediates** and gated at the leaf, because it is
  ANDed down all four levels.

Twenty claims, and the five that earn their place are the divergent ones: two
spaces map one address, to genuinely different physical pages, each reads its
own, a write through one lands in that space's page, and it leaves the other
alone. That last one is what would catch a leaf pointing at the wrong frame,
which reads identically to a correct one until something writes.

**And the low half, which is where a process actually lives.** `map_low` maps
`0x400000` -- busybox's own base, and every non-PIE binary's -- privately per
space. Two spaces each hold it, each reads its own page through it, and the
kernel's map is untouched.

`step` privatises its way down, one table at a time: the descent starts at the
root, which a space always owns, so each step either finds a table it already
owns or takes a private copy and repoints the parent. By the time a leaf is
written every table above it belongs to this space. A 2 MiB entry in the way is
split into 512 real entries carrying the same flags, so the rest of the large
page keeps mapping what it mapped -- `paging::split_large`'s bargain, one level
down, against a table this space owns. **Both are gated on `owns_table`**, and
that single predicate is what stops any of it editing a table somebody else is
sharing.

**The guard on a low address is about meaning rather than about tables.** The
tables would be perfectly correct either way; the hazard is that the kernel is
identity mapped, so shadowing virtual `0x2c00000` in a space points the
kernel's own heap pointer at somebody else's page for as long as that space is
installed. `mem::fixed::is_free` answers whether anything is using that
physical memory, which is exactly the question, and it is a **query rather than
a `claim`**: two spaces both wanting `0x400000` is the ordinary case for
processes, and reserving it would refuse the second for no reason. `claim` and
`is_free` share their two predicates so they cannot disagree about one address.
Measured after a run: `4 range(s), nothing claimed`.

**A task runs on its own root now.** `Task` carries one (0 meaning the
kernel's), `schedule` writes CR3 immediately before switching stacks, and
`set_root` takes effect at once on the running task. Safe there for one reason,
stated rather than implied: every root a task may carry maps everything the
kernel's does, so the code executing the write, the stack under it and the
incoming stack are mapped identically either side. `sharing_kernel` makes that
true and `map_low`'s guard is what stops divergence taking it away.

**The claim caught a real bug in the scheduler change, which is what it was
for.** `schedule` skips the CR3 write when both roots read zero, and that is
only sound while a task's recorded root describes the CR3 it is on. Clearing
the running task's root to the kernel's left both sides reading zero, the
branch untaken, and the machine on a root nothing named any more. The tell was
the asymmetry: the switch *to* a private root passed and the switch back
failed. Hence `set_root` activating immediately for the current task.

The test is shaped so a failure is a wrong value rather than a dead machine.
`0x400000` is mapped in **both** roots -- the identity map reaches the real
physical page, the space reaches a private one -- so whichever way the switch
goes the read is legal and the value says what happened. A page mapped only in
the space would fault at ring 0 with no recovery, which is a suite that halts
instead of reporting.

Thirty-two claims.

**A guest runs on a root of its own**, behind `linux space on|off`, off by
default. The space *shares* every mapping with the kernel's, so on and off
should be indistinguishable -- which is exactly what makes the switch worth
having. A fixture behaving identically both ways says the guest lifecycle
survives a non-kernel CR3, and that has to hold before anything diverges;
defaulting it on would make the first divergence bug and the first
"does this work at all" bug arrive together with nothing to tell them apart.

Measured, on unmodified busybox under glibc:

    uname -a     exited 0 after  60 syscall(s)   both ways
    sha256sum    exited 0 after 251 syscall(s)   both ways
    b01eaede758499526db8c8ccd159b0f773ef0ecb29c25952e5c1042f5168e4ec

That digest is the host's over the same bytes, so a real dynamically linked
binary relocated a 1.9 MB libc through `mmap`, hashed its own file at ring 3
under a private root, and got it right.

**One diff looked real and was not**, which is worth recording because it will
happen again: an earlier pair differed by `[mind t1] disabled` printed from
another task *into the middle of a line*, splitting `257` into `25` and `7`.
Console interleaving between tasks, and the tell was that the split fell
mid-token rather than at a boundary.

Cleanup order is the load-bearing part: `set_root(me, 0)` puts the kernel's
root back immediately, and only then may the space drop and free its tables.
Reversed, the allocator gets the page the processor is walking. `syscall::run`
returns on both paths that exist -- a guest that exits and a guest killed by a
fault both leave through the longjmp -- so it runs in the case that matters.

**A fixed image is mapped rather than placed**, when a guest has a space.
`Image::Mapped` backs it with ordinary heap pages and maps them at the address
the headers insist on, inside that guest's own tables.

The decomposition that made this small is worth keeping: **only the image has
an address the file demands.** The stack, the break and every `mmap` are at
addresses the *kernel* chooses, so two guests never collide there, and a PIE
already goes wherever the heap has room. `0x400000` is the whole problem, and
it is one region.

    space off   [linux] 185 byte(s) ... at 0x400000
    space on    image mapped at 0x400000, backed by heap pages at 0x2c17000
    space on    image mapped at 0x400000, backed by heap pages at 0x2c15000

All three printed `hello from ring 3` and exited 5 after 2 syscalls. The third
line is the result: one virtual address, two different physical backings, which
is exactly what two guests at once will need. `mem::fixed` is not consulted at
all on that path, so neither the machine-wide claim on `0x400000` nor the 6 MiB
placeable run bounds a fixed image any more.

**A mapped region must not be `protect`ed and that is required, not an
optimisation.** Its U bit is set at the leaf by `map_low`, inside the guest's
tables. `paging::protect` edits whatever CR3 names, so running it on
`0x400000` from the kernel's root would open the *identity* mapping of that
address -- a page the guest was never given.

**And rights for the regions that are *not* mapped go on under the guest's own
root, which cost a reproduction to learn.** `entry_for_user` opens the U bit at
every level down to the leaf, starting with `pml4[i4]` of whatever `read_cr3()`
names, and `Space::sharing_kernel` **copies** the kernel's PML4 entries when it
is built. Protect first and the U bit lands on the kernel's entry 0 while the
guest's copy of that entry keeps it clear -- and the bit is ANDed down all four
levels, so every page under it is unreachable from ring 3 however the leaf is
marked.

What it looked like is the part worth remembering. The **first** guest of a
boot died fetching its interpreter's first instruction, `error 0x15` (present,
user, fetch refused), and the second identical command worked. Three things
made it hard:

- **The evidence names the wrong level.** A fault reports an address, not which
  table denied it, so a leaf with correct rights looks like the whole story.
- **It read as a property of the command.** `uname -a` failed and `sha256sum`
  passed in one boot, which is about *ordering* and looks like it is about the
  programs. Two observations differing in more than one thing cannot say which
  one mattered -- the same coincidence the DOOM damage counter records.
- **It heals itself.** The first guest's `protect` leaves U set on the kernel's
  entry 0 for good, so every later space copies it already open. A bug that
  cannot happen twice is invisible to any test that runs twice.

Doing it under the guest's root is also strictly tighter: the U bits land in
the space's own PML4 and the kernel's entry 0 is left alone, so ring 3 reaches
those pages only through the root the guest actually runs on.

`Guest` declares `space` **before** the images it maps, so it drops first: the
tables point at the backing's pages and freeing those while a root still names
them would leave the next translation reading the allocator's memory. Nothing
is installed by then, so it is tidiness rather than a live hazard, and it is
the same ordering `give_back` exists to get right.

The three calls landed after this was written. What is still true is the
`SPACE` note: it is a single static, so nothing yet demonstrates two live
guests sharing `0x400000` at the same instant even though the memory allows it.
**`smp::init` passes CR3 to a starting application processor**
(`smp.rs:557`), which is harmless today because APs start at boot before any
space exists, and would not be if anything ever started one later.

### OpenGL, which turns out not to be kernel work at all

**Every OpenGL on Linux is a userspace shared object.** In software mode
`libGL` rasterises into ordinary memory and asks the kernel for nothing but
pages, so "implement GL" here is not a rasteriser to write, it is a library to
get into the guest and a syscall trace to answer. The estimate that started
this was 8,000 to 15,000 lines of kernel code, and it was wrong by all of it.

`tools/gl.py` fetches one, and which one is decided by arithmetic rather than
by preference. `fs.rs` holds an open file's whole contents and caps the total
at 64 MiB, so:

| | packages | installed |
|---|---|---|
| bookworm `libosmesa6` | 16 | **188 MiB** (112 of it `libLLVM15`) |
| bookworm `libgl1` | 49 | 210 MiB, and drags X11 |
| **stretch `libosmesa6` 13.0.6** | **6** | **8.2 MiB, no LLVM at all** |

LLVM is there for llvmpipe, the JIT rasteriser, and Debian builds llvmpipe and
softpipe into one object so the JIT cannot be declined. Mesa 13 predates that
and its `libOSMesa` is the *classic* software rasteriser: no gallium, no LLVM,
largest object 4 MiB. The whole closure with its glibc satellites is 8,033,864
bytes across nine objects, and `gl.py --report` prints that beside the cap so
the decision stays checkable.

**OSMesa is the third door.** `libGL` gets its surface from GLX, which needs an
X server, or EGL, which needs DRM, and there is neither here. `OSMesaMakeCurrent`
takes *a buffer the caller owns*: every `gl*` call after it writes there, and
the buffer reaches `/dev/fb0` with one `write`. Asking for `OSMESA_BGRA` makes
Mesa's byte order the display's own, so the blit is a copy rather than a
conversion -- which is the pixel-format work from `/dev/fb0` paying for itself.

The closure is computed from `DT_NEEDED` rather than from package metadata,
and it earned that again: nothing would have predicted `libOSMesa` needing
`libgcrypt`, which it uses to hash its shader cache.

### Calling a library with no compiler: a linked ELF, built by hand

`mkelf.py --kind gl` and `build_linked`. There is no C toolchain on the
development host and no usable WSL, so the choice was to fetch one or to teach
the fixture builder to emit a dynamically linked object. The second is what
this file already exists for: no toolchain will emit a binary that differs
from another in exactly one field, and none will emit one small enough to read.

The minimum a loader needs is six tables and twelve dynamic entries.
`.dynstr`, `.dynsym` with one undefined `FUNC` per import, `.hash` -- required
even when nothing is looked up, since `ld.so` refuses an object with neither
hash -- `.rela.dyn` with one `R_X86_64_GLOB_DAT` per import, `.got`, and
`.dynamic` naming all of it. No PLT and no lazy binding: `DF_BIND_NOW` has the
loader fill every slot before the program runs, so a call is `call [rip+slot]`
and there is no resolver trampoline to get right.

Two traps, and both are silent:

- **The hash table's `nchain` is how a loader sizes `.dynsym`.** One bucket
  means every name collides, which is what makes it constructible by hand: the
  chain then walks every symbol in order. A count one short is a symbol that
  does not exist.
- **`PT_PHDR`, and this one cost a disassembly.** glibc computes the main
  program's load address in exactly one place, `case PT_PHDR: main_map->l_addr
  = (Addr) phdr - ph->p_vaddr;`, and there is no fallback. Without it the base
  stays zero and every address the loader derives from a `p_vaddr` is used raw.
  The fault was a `#PF` reading `0xe8`, `ld.so` was in an SSE `strcmp`, and
  `0xe8` is 64 for the ELF header plus three program headers -- the file offset
  of `PT_INTERP`'s string with nothing added to it. Every real linker emits a
  `PT_PHDR`, which is why nothing else here ever needed one.

`DT_DEBUG` was the first theory and the run refuted it: the fault came back
byte for byte identical, same rip and same fifteen registers. That is what the
register dump is for.

### Threads, which one address space makes easier rather than harder

`src/linux/thread.rs`. The asymmetry is worth stating because it is easy to
read "no processes" as "no concurrency": `fork` needs two address spaces and
this system has one, while `CLONE_VM` asks to *share* one, which this kernel
grants by doing nothing at all. So the constraint that makes `fork` impossible
is the same one that makes a thread nearly free.

A thread needs four things and three were already here. A stack, which the
guest allocates. A scheduler, which `task.rs` has been preempting at 100 Hz
since long before any guest existed. Its own `FS`, which `arch_prctl` already
sets and which now has to survive a switch. And **its own syscall entry
state**, which is the piece that did not exist -- `syscall.rs` says so in its
own header: "that stack is one static, so the handler is not reentrant ...
stage 0 runs one guest with no threads. Both of those stop being true later."

**The entry path is per-task now and no assembly changed.** Three globals
carry a syscall across the ring boundary: where the guest's `rsp` went, which
stack the handler runs on, and where to longjmp back to. With one guest they
are constants; with two threads they are three ways to corrupt each other,
because a thread that blocks inside a syscall leaves them live while another
enters one. They are saved and restored by `schedule`, beside the FPU area and
in the same order, which is what makes them per-task: a global only read while
its own task is running *is* per-task as long as somebody swaps it. The
alternative was `swapgs` and a per-thread block, which collides with
`cpu::percpu` owning GS -- the same reason a guest is refused `ARCH_SET_GS`.

**A pool, not a task per thread.** `MAX_TASKS` is 24 and a kernel task that
returns is not reclaimed, it spins in `yield_now` forever. Reclaiming slots
means teaching the scheduler about a finished task, and the outgoing task's
state is written unconditionally in `schedule`, so that is surgery on the most
delicate loop here. Instead a finished thread parks its task and the next
`clone` takes it back: the limit is eight *concurrent* threads rather than
eight ever created, and a machine that never runs one spawns nothing.

`clone` returns twice, in two threads, at the same instruction, and only `rax`
tells them apart. The child arrives with every register zero except `rsp`,
because `glados_enter_guest` clears them, so it cannot be handed anything in a
register -- and its entry is the *parent's* return address, which `syscall`
left in `rcx`, so nothing has to be invented.

`futex` is `WAIT` and `WAKE`, and a waiter watches two things: the wake counter
*and* the word itself. Needing both is the point -- the counter alone loses a
wake when its small table is full, and the word alone misses a wake that
changed nothing. It is a yield loop rather than a sleep queue, the same bargain
`nanosleep` makes and for the same reason.

Joining is `CLONE_CHILD_CLEARTID` plus that futex, which is what `pthread_join`
is underneath: the kernel writing zero to the word when a thread ends is the
whole of the notification, and a kernel that ignores it leaves a library
waiting forever on a thread that finished.

    linux run /tmp/th
        9 mmap        0x0 0x2000 0x3 -> 46600192
       56 clone       0x200f00 ... -> -38    no CLONE_THREAD, so a process
       56 clone       0x210f00 0x0 -> -22    no stack
       56 clone       0x210f00 ... -> 4      a thread, and its id
      186 gettid                  -> 1       the main thread's id is its pid
      202 futex       ...+9 0x80  -> -22     a word that is not aligned
       24 sched_yield             -> 0
       60 exit        0x0         -> 0       <- the child, on its own stack
      202 futex       ...+8 0x80  -> 0       woken by the kernel clearing ctid
      231 exit_group  0x0         -> 0
      exited 0 after 10 syscall(s)

Zero is the mask, so all ten checks answered, and the last of them is the
whole claim: one page, two threads, and a value in it that could only have
been put there by the other one. Three runs in one boot, and the interesting
difference between them is the ordering -- in two the child finished before
the parent waited and the futex correctly answered `EAGAIN`, in one the parent
genuinely blocked and was woken. Both are right and a fixture that accepted
only the second would have been testing that the child is slow.

**Two bugs, and the first took the whole machine.** `run` saves and restores
`RFLAGS` around a guest because `syscall` clears `IF` through `FMASK` and
`exit` leaves through a longjmp that restores a stack rather than a processor
state. `run_thread` did not, so a pool task went back to its `hlt` with
interrupts off, which is a core that never wakes -- no prompt, no timer, and
the run deadline could not fire because firing is something an interrupt does.
And the handler stack lived on the `Task`, which stops carrying ring-3 state
the moment a thread ends: the second thread to use a pool slot entered its
first syscall with the stack at zero and pushed onto a null pointer. It ran
perfectly once, which is the worst number of times for a thing to work.

### Loading the loader

`PT_INTERP` names a path, the path is a file in the namespace, and a second
image at a second base is the whole of it. `load` places the program, places
the interpreter beside it, and jumps to the *interpreter's* entry. `dlopen`
then works because it is `ld.so`'s problem rather than ours, which is the whole
reason to load an interpreter instead of writing a linker -- and `dlopen` is
what Half-Life needs, since the game logic lives in `hl.so`.

**Three numbers have to be right and all three are silent when wrong.**
`AT_ENTRY` is the *program's* entry, not the interpreter's, or `ld.so`
relocates everything correctly and jumps back into itself. `AT_BASE` is where
the interpreter landed, and it is the only way an unrelocated `ET_DYN` can find
its own `_DYNAMIC`; it is **omitted rather than zeroed** when there is no
interpreter, because zero reads as "loaded at address zero" and the first thing
done with it is add it to an offset. And the address jumped to is the
interpreter's, which is the one of the three that is obvious when wrong.

`mem::fixed` had to learn to hold more than one range first. It held exactly
one, on the argument that one guest runs at a time -- an argument about
*guests* where the thing being counted is *ranges*, and one guest stops being
one range the moment it is dynamically linked. Sixteen now, with an overlap
check, which the single-entry version got for free by refusing everything: an
interpreter placed over the program it was loaded to run does not fault, the
second copy simply wins, and what shows is a jump into the middle of somebody
else's code.

**Two hand-assembled fixtures, because the pair is the test.**
`mkelf.py --kind loader` is a stand-in for `ld.so`: it walks the aux vector by
key (never by position -- the kernel may order them however it likes), checks
`AT_BASE` is present *and points at an ELF header*, checks `AT_ENTRY` is
present, prints, and jumps. Each check has its own exit code, so a failure says
which. It never touches `rsp`, because the program on the other side of that
jump expects to find `argc` where the kernel left it. `--kind interp` is the
ordinary hello program with a `PT_INTERP` naming `/tmp/loader`.

    glados> linux run /tmp/prog
    [linux] 253 byte(s), 2 segment(s), 253 byte span at 0x2c18000, entry 0x2c19078
    ld: an interpreter ran, with a base and an entry
    hello from ring 3
        1 write       0x1 0x2c19194 0x31 -> 49
        1 write       0x1 0x2c180df 0x12 -> 18
      231 exit_group  0x5 -> 0
      exited 5 after 3 syscall(s)

Every number is checkable against what `mkelf.py` printed. The interpreter was
placed at `0x2c19000` and its entry is `+0x78`, which is the `0x2c19078` the
loader reports -- an address outside the program's own 253 bytes, which is the
whole claim. The first write comes from `+404` of the interpreter and the
second from `+223` of the program, which are the two message offsets the
fixture builder named, and 5 is the program's own exit code.

Two negatives, and both matter more than the positive. Running the interpreter
*alone* exits **21**, which is its own code for "there was no `AT_BASE`" --
that is the check that the entry is omitted for a static binary rather than
zeroed. And a binary naming a real `/lib64/ld-linux-x86-64.so.2` is refused
with **"the interpreter this binary names is not in the namespace"**, which is
a fact about the machine rather than a design decision.

### The two calls a linker makes before anything else

`mmap` served exactly one shape for a long time: anonymous, no address, no
file. That is enough for an allocator and nothing else, and it is precisely the
pair of refusals a dynamic linker meets on its first two calls. It reserves a
span with one anonymous mapping, then writes each segment of each library over
part of that span with `MAP_FIXED`, and every one of those segments is
file-backed.

Both refusals went the way `ET_EXEC` went, and for the same reason: each was
true of what the kernel *knew* rather than of the machine.

**`MAP_FIXED` has three cases and the middle one is the point.** An address
inside memory the guest already holds is re-laid in place, which is not an
attack but the ordinary case, and it records nothing because whatever holds
those pages still holds them. An address nothing holds goes to
`mem::fixed::claim`, the only thing here that can promise a virtual address,
since virtual is physical. Anything else is `ENOMEM`. Zero and unaligned are
`EINVAL` rather than hints: rounding would put a library's segment a page off
its own headers.

**A file mapping is a copy, and that is a real deviation.** Linux maps the page
cache, so two processes mapping one file share pages. Here an open file already
*is* its contents -- `fs.rs` says so and gives the reason -- so there is no
cache to point at. `MAP_PRIVATE` is exactly a copy and so is exactly right,
which is what `ld.so` uses for every library it loads. A **shared writable**
file mapping is refused with `ENODEV`, because honouring it means writing back
into a store keyed by content, which is a new root hash per modified page. A
mapping running past the end of the file is zero-filled rather than refused,
since that is how a shared object's `.bss` is made.

`Mapping` carries where its pages came from, because the two go back different
ways and getting it wrong is silent: freeing a placed range to the heap hands
the allocator memory it never owned. `give_back` is the one place that knows,
and it puts the **rights back first** -- the mistake this tree has now made in
three separate places.

`mkelf.py --kind maps` is eleven checks folding into a mask, `fsabuse`'s idiom
for its reason: zero means every one answered what Linux answers. It maps **its
own file** through `argv[0]`, so it needs nothing staged beside it and the
bytes it checks are ones it can be certain of.

    glados> linux run /tmp/maps
        9 mmap    0x0        0x2000 0x3 -> 46522368
        9 mmap    0x2c5e000  0x1000 0x3 -> 46522368
        2 open    0x2c5cfbd  0x0    0x0 -> 3
        9 mmap    0x0        0x1000 0x1 -> 46534656
        9 mmap    0x0        0x1000 0x3 -> -19
        9 mmap    0x1234     0x1000 0x3 -> -22
        9 mmap    0x0        0x1000 0x3 -> -22
        9 mmap    0x0        0x1000 0x1 -> -9
       11 munmap  0x2c5e000  0x2000     -> 0
      231 exit_group 0x0 -> 0
      exited 0 after 10 syscall(s)

46522368 is `0x2c5e000`, so the reservation and the fixed mapping over it
answered the same address. Then a private file mapping, and four refusals by
their own errno: `ENODEV` for shared-and-writable, `EINVAL` twice for an
unaligned fixed address and a fixed address of zero, `EBADF` for a descriptor
nobody opened. The `munmap` of the whole reservation returning 0 is the check
that the fixed mapping laid over part of it left **one** record rather than
two.

**Two claims in `diag linux` used to assert the opposite** and both said so in
words about this loader: "a file-backed mapping is refused, there being no fd
table" and "MAP_FIXED is refused, for the reason ET_EXEC is". They are nine
claims now, about the refusals that remain and about the shape a linker uses.

**Which libc is not a decision this kernel makes.** A libc is userspace: it is
linked into the binary or named by it in `PT_INTERP`, and `load` reads that
path and loads whatever is there. So musl and glibc can both be installed and
each program takes its own, which works today and needed no code. What cannot
happen is two of them inside one process, so "use whichever is faster" is a
per-program question and never a per-call one. `linux libc` says which are
installed, a successful run names the interpreter it used and where it landed,
and a refusal names the path the binary wanted -- which the error itself cannot
do, its type being a `&'static str`.

What glibc will additionally ask of the syscall surface is a *prediction* and
is written down as one in `out/release/PLAN-native-linux.md`, which is a
working note rather than repository content, like every planning document
here. The short version: musl is a subset, so doing it first is the first half
of the same road and glibc later is additive rows in the `-ENOSYS` trace. The
one thing that cannot be worked around is a binary built against glibc that
cannot be rebuilt, and the game logic this target eventually loads is exactly
that kind of object.

**Both real interpreters have run, and this said they had not for a while
after they did.** `tools/libc.py` fetches musl's and glibc's from Alpine and
Debian, computes the *closure* rather than a list somebody wrote down (which
is how Debian's busybox turned out to need `libresolv.so.2` as well), and
stages a dynamic busybox for each. Neither is in this repository, for the
reason no WAD is.

    linux run /tmp/m/busybox uname -a         # musl, ld-musl-x86_64.so.1
    linux run /tmp/g/busybox sha256sum /tmp/g/busybox

The second is the check that settles it: glibc's linker relocated a 772 KB
binary through this kernel's `mmap`, and busybox then hashed *its own file*
byte for byte to the digest the host computes over the same bytes.

**glibc needs `LD_BIND_NOW=1` and musl does not**, which is a real limit and
is stated rather than worked around. busybox is BIND_NOW already; `libc.so.6`
is lazily bound, so its first call through the PLT enters
`_dl_runtime_resolve_xsavec`, which reads `GOT[1]` for the link map -- and
`GOT[1]` is zero here while `GOT[2]` beside it is not. That was located
precisely (glibc's `DT_PLTGOT` is `0x1d2fe8`, inside a `PT_GNU_RELRO` ending
`0x1d3000`, so the two words are the last sixteen bytes of the RELRO range)
and the boundary hypothesis it suggests was **tested and refuted**: three new
`diag paging` claims pass, and `protect` loses no bytes at a range end. Why
the linker leaves that word zero is open. `LD_BIND_NOW` resolves everything up
front and is arguably what a game wants anyway.

Four kernel bugs came out of that pair of runs and none was reachable any
other way. `teardown` did not restore the *interpreter's* page rights, so a
guest's RELRO page went back to the heap read-only and the next `fat get`
faulted at ring 0 -- the fourth instance in this tree of the same class, which
is why `give_back` now says so at the top. `newfstatat`'s flags never reached
it, so `AT_EMPTY_PATH` could not work. `mmap` copied file bytes *before*
applying rights, so glibc's `PROT_NONE` reservation with a `MAP_FIXED` overlay
made the kernel fault on the guest's behalf. And `locate` answered "no guest"
about a guest that had just died, because `run` calls `teardown` on the line
after the guest returns -- made twice, once for three addresses and once for
fifteen registers, and fixed the same way both times by resolving before
teardown.

**The fault report carries every register now, and that is what found the
last one.** `Fault` used to keep five fields picked in advance; which one
matters is not knowable at the time, and the one that mattered was `rdi`
holding a zero nobody had thought to ask about. `cpu::idt` emits 32 stubs from
one `global_asm!` macro at a fixed 16-byte stride, each pushing a fake error
code where the CPU pushes none so both shapes are in phase, then fifteen GPRs
-- vector plus fifteen pushes is 128 bytes, which is what leaves the tail
16-aligned for the SysV call. Get that wrong and the reporter takes a `movaps`
#GP inside itself. `diag recover` asserts the stride by reading the first byte
of each stub. The entry is `extern "sysv64"` and not `extern "C"`, for the
fourth time in this tree.

### The four `/proc` files that can be answered honestly

`src/linux/proc.rs`, and the rule it opens with is the whole design:

> **A field this machine does not know means the file does not exist.**

`/proc/stat`, `/proc/loadavg`, `/proc/cpuinfo` and `/proc/uptime` are text
formats with no way to say "I do not know": omit a field and a parser breaks,
invent one and it gets believed, and there is no `Option` in a text file. A
missing `/proc/stat` sends a program down a fallback it already has. So the
table is four entries and grows when something asks, which is the `-ENOSYS`
discipline applied to paths instead of call numbers.

**`/proc/self/exe` is the reason the module exists**, being the only one of
the four with no syscall alternative: there is no call that answers "where is
my own binary", so a program looking for its data directory beside itself has
this and nothing else. It is a symlink, so `readlink` is the usual way in, and
`open` on it gives the image, both of which work here.

What makes it cheap is a property `fs.rs` already had for its own reasons: an
open file *is* its whole contents, so a synthetic file is only a different way
of filling that `Vec`. No second kind of descriptor, no second `read` path,
and `lseek` works on these for free. The hook sits **before** the store in
`sys_openat`, since asking a content-addressed tree about `/proc` answers
`ENOENT` about a path that does exist.

`/proc/self/maps` is truthful and therefore strange. A guest shares one
address space with the kernel, so these are the real addresses of real pages
sitting wherever the heap put them rather than at the tidy `0x400000` a reader
expects; and the rights come from `paging::query` rather than from what was
asked for at `mmap`, because `mprotect` moves them afterwards and a linker
spends its last act doing exactly that. Ends are rounded up to a page, which
is *more* truthful and not less -- rights are applied per page, so the guest
owns the whole of its last partial one, and every parser of this file was
written against a kernel whose ranges are page-aligned.

    linux run /tmp/g/busybox cat /proc/self/maps
    000002d20000-000002d24000 rwxp 00000000 00:00 0 [stack]
    00000300c000-0000030ca000 rwxp 00000000 00:00 14267663290916959693 /tmp/g/busybox
    0000030ff000-000003134000 rwxp 00000000 00:00 9676539629617231489 /lib64/ld-linux-x86-64.so.2
    000003134000-000003174000 rwxp 00000000 00:00 0 [heap]

    linux run /tmp/g/busybox readlink /proc/self/exe   ->  /tmp/g/busybox
    linux run /tmp/g/busybox wc -c /proc/self/cmdline  ->  40

That 40 is the check worth keeping, because busybox computed it: the four
argv strings are 14, 2, 2 and 18 characters and each carries a NUL, and a
`joined` that dropped the final separator would answer 39 while looking
perfectly reasonable.

**`claims` is a property of the path and never of what is running**, which it
was not at first: it asked `link` and `read`, both of which consult the live
guest, so `/proc/self/exe` was claimed while a guest ran and unclaimed
otherwise. `diag linux` caught that, running with no guest installed. The
other direction would have been much worse -- a listing offering a name that
`openat` then routed to the store.

### `/dev`, a screen to draw on and a keyboard to read

`src/linux/dev.rs`. Seven nodes, synthetic like `/proc` and under the same
rule, and one of them is the display: `/dev/fb0`, the smallest well-specified way for
a program written somewhere else to put pixels on this machine.

There is no `/dev/tty`, no `/dev/dri` and no `/dev/snd`, and that is the rule
rather than an omission -- each is a real interface with real semantics nothing
here can supply, and a node that opens and then does nothing is worse than an
absent one, which sends a program down a fallback it already has.

**A guest gets the actual framebuffer, not a shadow.** This kernel is
identity-mapped, so virtual is physical and `smem_start` can be the aperture's
own address rather than a lie about a buffer somewhere else; a frame the guest
writes is on screen as it writes it, with no copy anywhere. The price is that
the desktop must stand down while that is true, which is `gfx::exclusive` --
the same flag `port::with_screen` sets for DOOM and the editor, split into two
halves here because a guest holds the screen across many syscalls rather than
for the length of one call. It is taken on the first write or mapping rather
than at `open`, since a program that merely `stat`s the device has not asked
for the machine, and released in `teardown` **unconditionally**, because a
guest that took the display and then faulted is exactly the case that matters.

Three decisions where the alternative was worse:

- **`MAP_SHARED` with `PROT_WRITE` is refused for every file here** and is the
  entire point of this device. The objection everywhere else is that writing
  back into a content-addressed store is a new root hash per page; the
  framebuffer is not in the store, so the objection does not apply. The device
  branch of `sys_mmap` therefore sits *before* that refusal.
- **`FBIOPUT_VSCREENINFO` accepts only the mode already running.** There is no
  mode-setting on this machine: the geometry is whatever the firmware left.
  Answering 0 to a mode this display cannot enter is the worst of the three
  available answers, because the program then draws at a geometry the hardware
  does not have, forever, with nothing reporting it.
- **A mapping of the aperture is `Source::Device`**, a third kind, because the
  two it is not are both actively wrong: freeing it to the heap hands the
  allocator several megabytes of the display's memory, and releasing it
  through `mem::fixed` releases a claim nobody made. All it owes is the U bit,
  taken back off.

**The pixel layout is the field that is silently wrong.** `Format` names the
order of *bytes in memory* and `fb_bitfield` names bit positions inside a
little-endian word, so the two run in opposite directions and reversing them
swaps red and blue -- with no error, no fault, and a picture that is merely a
strange colour. `mkelf.py --kind fb` is the answer: thirteen checks folding
into a mask, and then three bands painted by shifting 255 by the offsets the
driver *told* it, in the order red, green, blue. A run that comes back
blue-green-red has found a bug no exit code could.

    glados> linux run /tmp/fb
        2 open   /dev/fb0 -> 3
       16 ioctl  0x3 0x4602 0x2c5bdf0 -> 0        FBIOGET_FSCREENINFO
       16 ioctl  0x3 0x4600 0x2c5bd50 -> 0        FBIOGET_VSCREENINFO
       16 ioctl  0x3 0x4601 0x2c5bd50 -> 0        the mode it was just given
       16 ioctl  0x3 0x4601 0x2c5bd50 -> -22      a mode this display has not
       16 ioctl  0x3 0x5401 0x2c5bd50 -> -25      TCGETS, so ENOTTY
        9 mmap   0x0 0x3e8000 0x3 -> 2147483648
      exited 0 after 8 syscall(s)

Zero is the mask, so all thirteen answered. `0x3e8000` is 4,096,000, which is
1280 x 800 x 4, and `2147483648` is `0x80000000`, the aperture itself rather
than a copy of it -- which is one of the checks rather than a coincidence, and
another is reading back through the mapping what was written through it.

`--kind fb` with any argument sleeps for ten minutes instead of exiting, and
that is how the picture gets photographed: `drive.py` screenshots when it
stops, and `teardown` puts the desktop back the instant the guest returns, so
the exit path for a held fixture has to be the **timeout**. The same recipe
`doom play` needed.

The other four nodes are cheap and each has exactly one correct behaviour.
`/dev/random` and `/dev/urandom` are one node, which is a deviation with a date
on it: they were different devices until Linux 5.6 and have behaved alike
since. It reads through `rng::fill` rather than `fill_secret`, deliberately --
the secret form refuses below the entropy threshold, and glibc's fallback for a
failing `getrandom` is to open this and read it, so a refusal here leaves a
program with no third option.

Driven under glibc: `busybox ls -l /dev` reports all five as `crw-rw-rw-`,
which is an independent reader agreeing they are character devices, and
`busybox dd if=/dev/urandom of=/tmp/r.bin bs=16 count=1` reports `1+0 records
in, 1+0 records out` and exits 0.

**That sweep found `dup3` as well.** glibc's `dup2` calls `dup3` when the two
descriptors differ, so `dd` reached `dup2` and `hexdump` reached `dup3` and
stopped -- two applets in one run taking different routes to the same thing.
The one difference that matters is inverted on purpose: `dup3(n, n, 0)` is
`EINVAL` where `dup2(n, n)` answers `n`, because the no-op case hides a bug in
the caller and the newer call refuses it.

**What this does not get you, said plainly.** SDL2 has no framebuffer backend
-- its video drivers are X11, Wayland, KMSDRM, offscreen and dummy -- so
`/dev/fb0` does not put an unmodified SDL2 program on screen. What it does is
make the class of program that talks to a framebuffer directly work, and give
everything visual afterwards something to stand on.

### The other half: `/dev/input/event0` and `event1`

`src/linux/input.rs`. A screen with no input is a picture. Two evdev devices,
a keyboard and a pointer, fed from the two places this kernel's own drivers
already converge: `kbd::decode` for every scancode and `mouse::apply` for
every packet, the latter being the point `port::mouse` already hangs off for
the reason it gives -- PS/2 and USB HID both arrive there and a second copy
would be the one nobody tested.

**The ioctls are the interface and the events are the easy half.** A program
classifies a device by reading its capability bitmaps and nothing else, so a
mouse that fails to advertise `REL_X` is opened, read from successfully, and
ignored, which looks exactly like input that does not arrive. The name is
decoration.

Three things are silent when wrong and each has a claim:

- **`SYN_REPORT` is load-bearing.** A reader batches events until it sees one
  and treats the batch as a single state change, so a device that never
  synchronises delivers nothing while working perfectly at the `read` level.
- **evdev's `REL_Y` grows downward** where `mouse::apply`'s `dy` grows up, so
  the sign is flipped on the way in. Getting it wrong gives a game whose mouse
  look is inverted, which reads as a preference somebody forgot to expose.
- **A Linux keycode *is* a set-1 scancode, for the main block only.** That is
  historical rather than lucky: the keycodes were assigned to match, so
  `KEY_ESC` is 1 and `KEY_A` is 30 exactly as the wire has them. It stops at
  the `E0`-prefixed keys, which is why those need a table and only those do.

A descriptor opens at the *present*, so a program does not receive every key
pressed since boot the moment it starts, and a reader that falls behind the
256-entry ring is told `SYN_DROPPED` rather than handed a gap -- a release
nobody delivered is a key held forever.

**`linux feed` is how any of this is driven, and it had to exist.** A guest
owns the machine while it runs and `drive.py` sends the next command only when
it sees a prompt, so anything typed arrives before the guest starts or after
it has gone -- and a device opening at the present means events pushed
beforehand are events nobody sees. It is the same answer `win keys` and
`doom play`'s timed script are, and it borrows their spelling:

```
linux feed shift@400 -shift@1200
linux run /tmp/ev
```

Delivery is from the **timer interrupt**, the only thing that gets a turn
while a guest runs, and it goes through `kbd::inject_scancode` rather than
straight into the ring, so what is exercised is the path a real key takes.
Shift and control rather than letters, because `kbd::decode` returns early for
the modifiers before it pushes a character and the feed therefore leaves
nothing in the shell's own input ring for the harness to trip over.

`mkelf.py --kind ev` is twelve checks folding into a mask, and the interesting
one is not a check: the sixth `read` has no `O_NONBLOCK` and nothing has
happened yet, so the guest sleeps inside the kernel and comes back when the
timer delivers. Nothing else in this tree exercises a guest waiting on
anything.

    linux feed shift@400 -shift@1200
    linux run /tmp/ev
        2 open   /dev/input/event0 -> 3
       16 ioctl  0x3 0x80044501 -> 0     EVIOCGVERSION
       16 ioctl  0x3 0x80084520 -> 8     EVIOCGBIT(0): EV_SYN|EV_KEY, so a keyboard
       16 ioctl  0x3 0x80204506 -> 19    EVIOCGNAME
        0 read   0x3 ... 0x10   -> -22   a buffer too small for one record
        8 lseek  0x3 0x0 0x0    -> -29   an event stream has no position
        0 read   0x3 ... 0xc0   -> 48    blocked, then the press and its SYN
        0 read   0x3 ... 0xc0   -> 48    blocked again, then the release
      exited 0 after 9 syscall(s)

**Two real bugs came out of driving it and one is much bigger than evdev.**

`overran` is checked from the timer interrupt and only when the saved CS says
ring 3, so **a guest blocked in a syscall was invisible to the one thing that
ends a runaway**. A blocking read on a device nothing feeds took the machine:
shell gone, no key able to bring it back, reboot the only way out. The wait
loop checks the deadline itself now and ends the guest through `kill_blocked`,
whose safety note is deliberately a *different* condition from
`kill_overrun`'s rather than a weaker one -- there the guarantee is that the
kernel was not running at all, here it is that this particular loop holds no
allocation, no borrow of `SPACE` and no lock across the yield.

And **"at teardown" is not "at the end of a run"**: `install` calls `teardown`
at the head of itself to abandon any previous space, so the `stop` that swept
a spent script swept it a microsecond *before* the guest it was armed for
began. The counters found that in one run where reasoning had produced three
wrong theories -- `service` reported 9,346 ticks seen and 0 delivered, which
says the timer was fine and the script was gone. Anything else that hangs
cleanup off `teardown` has the same trap waiting.

### A filesystem a program written for Linux recognises

`src/linux/fs.rs`. The store underneath is not a filesystem and `sysbox::tree`
says so in its own header: a content-addressed Merkle tree where a copy is
O(1), a snapshot is a hash, and `rm` detaches a name rather than destroying
anything. None of that has an `open`. A Linux program expects the opposite set
of things -- a path resolves to an inode, an inode has a size and a mode, a
descriptor is a small integer with a cursor in it, and reading advances the
cursor. This module is the translation, and it is a **view rather than a second
store**: nothing here owns any bytes, every read goes to the tree and every
listing comes from `sysbox::listing`.

The descriptor table lives in `Space` beside the memory regions and is seeded
with the three standard ones at `install`, so a guest that never opens anything
still has somewhere to write. They are `Fd` variants rather than table entries
with a magic path, which is what makes `lseek` on stdout answer `ESPIPE` from
the type system instead of from a string comparison.

Twelve calls, plus two that exist to stop a runtime concluding it failed to
start: `read`, `open`, `openat`, `close`, `lseek`, `fstat`, `stat`, `lstat`,
`newfstatat`, `getdents64`, `ioctl`, and `getpid`/`set_tid_address` answering 1
because there is one process and it is the guest.

**Opening for writing is refused, and that is a decision rather than a gap.** A
write to a content-addressed store is a new root hash, so honouring `O_WRONLY`
would give a guest binary a route to the namespace that goes around every gate
`sysbox` puts in front of the shell -- the sandbox, the applet table, the
snapshot. `EACCES`, and the day a guest needs to write, what it needs is a
scratch subtree with the same jail an Aiksi program gets, not this call quietly
growing a second meaning.

**A descriptor-relative `openat` is refused too**, with `ENOSYS` and for a
smaller reason: resolving one needs the directory's own path kept per open
descriptor, and resolving it against the working directory instead would open a
real file that is not the one the guest named. `AT_FDCWD` and absolute paths
are the whole of what works.

**An open file holds its whole contents.** `read_blob` answers a `Vec`, so the
honest options were to keep that or to teach the store ranged reads. Keeping it
makes `read` a slice and `lseek` an integer, and it means a guest opening a
600 MB model file takes 600 MB of heap. What a guest reads today is
configuration and text. When that stops being true this is the first thing to
change, and it is written down here rather than discovered by an allocation
failure.

**There are no permissions, owners or times.** Everything reports mode 0644 or
0755, uid 0 and a zero timestamp. A program branching on any of those gets a
consistent answer rather than a true one, which is the right trade while the
alternative is inventing a field the store does not have.

**A path containing `..` is refused rather than normalised.** A tree with O(1)
copies has no single parent to walk back to, so there is no correct answer, and
a normalised path resolves to somewhere the guest did not name. `.` and doubled
separators do collapse, since those have one answer.

**`struct stat` is 144 bytes and the layout is an ABI rather than a choice.**
Writing it a field short is not a smaller answer, it is a different structure,
and libc reads past the end of what was written. `st_blocks` is in 512-byte
units because that is what `du`-shaped callers divide by, so a one-byte file is
one block and not zero.

**A `linux_dirent64` is padded to eight because the kernel pads.** A guest
walking the buffer adds `d_reclen` to its cursor, so an unpadded record leaves
the next one misaligned and the guest reads a name out of the middle of an
inode. `getdents64` snapshots the listing at `open`, since a directory that
changed under a half-finished walk would hand out a shifting list and the tree
has no cursor to offer instead.

**Inode numbers are the first eight bytes of the SHA-256 of the path**, with
the low bit set so none is zero. The tree has no inodes; what programs use the
number for is telling two paths apart and spotting hard links, and a hash of
the path answers both.

`build_stack` lays out real argc, argv, envp and auxv, and the shell passes the
path as argv[0] because that is what a program expects and what busybox
dispatches on.

**Two programs, and they are the measurement.** `tools/mkelf.py --kind cat` and
`--kind grep` are hand-assembled the way every other fixture is. `cat` is the
smallest thing that exercises the whole projection at once: a path travels from
the shell through argv into `openat`, the namespace resolves it, `read`
advances a cursor across several calls, and end of file is a zero return rather
than an error.

    glados> linux run /tmp/cat /tmp/lines.txt
    alpha
    the quick brown fox
    beta
    jumps over the lazy dog
        2 open      0x2c5cff1 0x0 0x0 -> 3
        0 read      0x3 0x2c5cdb0 0x100 -> 54
        1 write     0x1 0x2c5cdb0 0x36 -> 54
        0 read      0x3 0x2c5cdb0 0x100 -> 0
        3 close     0x3 -> 0
      exited 3 after 6 syscall(s)

`grep` is the one that checks the bytes are *right* rather than merely present.
A naive substring search over a line buffer answers differently for every
one-byte change in the file, so a read that lost a byte, doubled one or stopped
early shows up as a wrong set of lines instead of as plausible output:

    glados> linux run /tmp/grep the /tmp/lines.txt
    the quick brown fox
    jumps over the lazy dog
        2 open      0x2c5cff1 0x0 0x0 -> 3
        0 read      0x3 0x2c5bfa0 0x1000 -> 54
        3 close     0x3 -> 0
        1 write     0x1 0x2c5bfa6 0x14 -> 20
        1 write     0x1 0x2c5bfbf 0x17 -> 23
      exited 0 after 6 syscall(s)

Every number is checkable against the file. The buffer is at `0x2c5bfa0`, so
the first match is written from +6, which is exactly past `alpha\n`, and the
second from +31, which is past `beta\n`; 20 is `the quick brown fox` with its
newline and 23 is `jumps over the lazy dog` without one, because the file ends
there and a final line carries no separator.

The exit codes are grep's own -- 0 matched, 1 did not, 2 could not -- which is
what makes the negatives worth running. `grep zebra` exits 1 having written
nothing, `grep` with no arguments prints usage to fd 2 and exits 2, and a
missing file gets `-2` out of `open` and never reaches `read` or `close`.

The fixture is a real frame rather than register juggling: `rbp` is the buffer
and the locals sit underneath it. The alternative was keeping seven live values
in registers across a `write`, and the syscall ABI hands back only `rbx`,
`rbp`, `r10`-`r15` and `rsi`/`rdi`/`rdx` -- of which two are the arguments the
write needs.

### What a guest gets for asking wrongly

`--kind fsabuse` is the third program and it is the one that matters, because
`cat` and `grep` only ever ask for things that work. A projection over a store
that is not a filesystem fails at the *edges*, and an edge is exactly what no
program written to use the thing normally will reach. So the negatives are the
subject: fifteen checks, each folding one bit into a mask, and the exit code is
the mask. Zero means every one of them answered what Linux answers.

**One of them stopped the machine, from ring 3, with two ordinary calls.**
`lseek` past the end of a file is legal -- this module says so, in a comment,
two functions above the bug -- and `read` then indexed a slice at the cursor
without clamping it:

    linux run /tmp/fsabuse /tmp/lines.txt
    *** PANIC *** panicked at src\linux\syscall.rs:725:52:
    range start index 1048576 out of range for slice of length 54

That is a guest at CPL 3 halting a kernel that has no unwinder, and there is
nothing exotic in it: seek, then read. The measurement is the point rather than
the fix, which is one `min`. Nothing about ring 3, page rights or bounds checks
was wrong; the pointer never left the kernel. It was an ordinary Rust index on
a value the guest chooses, and the only thing that finds those is a program
written to choose badly.

Four more the same run turned up, none of which crashes and all of which lie:

- **`write` looked at the descriptor number and not at the descriptor.** It
  compared `fd` against 1 and 2, which is right until a guest does the thing
  every shell does: `close(1)` then `open(...)` hands the file descriptor 1,
  and writing to it printed the guest's redirected output to the terminal and
  reported success. `close(1)` alone was worse, since a write to a descriptor
  that is not open has to be `EBADF`.
- **`fstat` reported stdin, stdout and stderr as empty regular files**, under a
  comment saying that reporting them as empty regular files is what makes a
  program believe stdout is seekable. The comment was right and was describing
  the code beside it. They are pipes now, which is the answer that agrees with
  `lseek` on them returning `ESPIPE`.
- **`newfstatat` ignored its directory descriptor.** `openat` refuses a
  relative path against a real one, because resolving it needs the directory's
  own path; `newfstatat` resolved it against the working directory instead and
  answered confidently about a file that exists and is not the one asked for.
- **`read_cstr` refused a legal path sitting near the end of an image.** It
  checked reachability a page at a time, which is right for the speed -- a
  4 KB path checked per byte is four thousand page-table walks -- and demands
  that the whole rest of the page be owned. A region does not have to end on a
  page boundary: the image's is the ELF span, so a path constant in the last
  partial page of a binary is entirely legal and got `EFAULT`, which reads as
  a pointer bug in the program rather than a bounds check being too eager. It
  falls back to a byte at a time when the page-wide check overshoots. Found by
  the fixture opening a path it carries at the very end of its own image,
  which is a shape `cat` and `grep` do not have because their paths arrive in
  `argv`.
- **And it had one failure with three causes.** An unreachable pointer, a
  string with no terminator and bytes that are not UTF-8 all became `EFAULT`,
  which sends a program to look at its pointer arithmetic for a filename that
  was merely Latin-1. `EFAULT`, `ENAMETOOLONG` and `ENOENT` now, the last
  because a Linux path is bytes and a namespace keyed by `String` genuinely
  cannot hold one.

And two costs rather than bugs. `stat` learned a file's size by reading the
file, so `stat` on a 600 MB checkpoint allocated 600 MB to look at a `usize`;
`sysbox::blob_len` answers it without the copy. And an open descriptor *is* its
contents here, so sixty-four of them against an unbounded file size is a guest
taking the heap with a loop of `open` -- `OPEN_MAX_BYTES` caps the total at
64 MiB and answers `ENOMEM`, which Linux can return and would not return for
this reason.

**The aux vector was empty and that was not a safe default.** A static libc has
no dynamic linker to ask, so everything it cannot compute it reads from there:
`AT_PAGESZ` becomes musl's `libc.page_size`, which it divides by, and
`AT_RANDOM` is where the stack guard comes from. A vector holding nothing but
`AT_NULL` hands a real binary a page size of zero. It now carries `AT_PAGESZ`,
`AT_CLKTCK` (from `crate::TIMER_HZ`, the interrupt rate, and deliberately not
from `lapic::timer_hz()`, which is the calibrated APIC frequency and is in the
millions), the four ids and `AT_SECURE` as zero, `AT_RANDOM` pointing at
sixteen bytes of `rng::fill` on the guest's own stack, `AT_ENTRY`, and the
`AT_PHDR`/`AT_PHENT`/`AT_PHNUM` group.

Two details there are load-bearing. `AT_PHDR` is a *runtime* address, so it is
the segment containing the header table plus the offset into it -- `base +
phoff` is the same number only when the first segment starts at file offset
zero, which is true of every fixture here and is not a property of the format.
And when no loadable segment covers the table the whole group is omitted rather
than pointed at zero, because `dl_iterate_phdr` walks what it is given either
way and an absent vector is the one a libc knows how to cope with.

**None of it has been read by a real libc on this machine**, since every
fixture is hand-written and consumes none of it. That is a bet placed where the
ABI says to place it, and it is worth saying so rather than letting the section
read like evidence.

One piece of dead code came out with it. `build_stack` padded down when
`(sp + words * 8) % 8 != 0`, which cannot hold: `sp` is sixteen-aligned and
every word is eight bytes. It read as an alignment fix and was a tautology,
which is the more expensive kind of dead code because it stops anybody looking.

### Placing an image where it insists

`src/mem/fixed.rs`. A non-PIE binary demands the addresses in its own headers,
and this kernel is identity-mapped, so those are real physical bytes. That was
refused under a reason true of what the kernel *knew* rather than of the
machine: nothing could answer "does anything own four megabytes at four
megabytes".

**The frame allocator cannot answer it, and why is the interesting part.**
`EarlyFrames` is a bump allocator whose cursor is forward-only by design -- its
own doc explains why rewinding would be worse than the bug it has -- so by the
end of boot it sits past the heap, three hundred megabytes up, and everything
behind it reads as unavailable. That includes large conventional regions it
merely stepped over while looking for one span big enough for the heap.
`0x400000` is exactly such an address: untouched, and invisible to the only
thing you would think to ask.

So the free set is computed the other way round: every conventional region the
firmware declared, minus the handful of ranges boot actually took. The
allocator records its handouts, which it can do precisely because it never
frees -- a bump allocator's history is a short list. `handouts()` answers `None`
if one was ever dropped and the snapshot is skipped rather than approximated,
because a free set missing a taken range would place a guest on top of the live
page tables.

    [boot] placeable  7 MiB free below the heap and above it, largest run 6 MiB

Seven megabytes on this map, which is what the firmware calls conventional less
the page tables and the heap. `mem` prints the table and `diag place` asserts
the arithmetic against a synthetic map -- synthetic on purpose, since a claim
written against the real one would assert something about QEMU and fail on the
GF63 for a correct reason.

### Testing

There is no `cargo test`. This is a `no_std` UEFI binary with no host test
runner, so **verification is the boot selftests plus driving QEMU.**

At boot the system runs **twenty-nine selftest sections** -- count the
`[selftest]` headings in a boot log, which is the only figure that cannot go
stale -- **seventeen** of which are wrapped in `main::section` so one that
breaks marks itself unavailable instead of taking the machine, and `diag`
offers **sixty-five named suites** on demand (`diag.rs`'s `SLOTS`, asserted
against `SUITES.len()`), most of them the same checks (the `aiksi` section covers the capability gate by name and never by
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
tally reading `0 passed, 0 failed, 48 not run`, which is easy to read as a
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

**That figure said "about 10% between boots" and it is wrong by a factor of
seven.** Three boots of one binary, `--no-payload`, `-smp 4`, the same command
prefix, read through `bench report`:

    rail               1 vs 2    1 vs 3    2 vs 3 (post-control)
    video.rect         -69.4%    -62.0%    +24.4%     (the control itself)
    video.*            ...       ...       -20 to -30%
    core.new           -73.7%    -72.7%     +3.9%     (the control itself)
    core.*             ...       ...       +0.2 to +1.5%
    ai.matmul           +2.5%    -32.5%    -34.1%
    smp.*              -50%      -61%      -21.7%, -5.1%

Two things fall out and both are protocol rather than tuning.

**Discard the first reading after a build.** Run 1 was two to three times
slower than runs 2 and 3 on every timing rail, which is the host's page cache
meeting a freshly written 5 MB image. Anything compared against it is
measuring the build system. The CI verify job takes two readings and keeps the
second for exactly this reason.

**The control works, where there is a real one.** After dividing `core.new`
out, the three interpreter rails agree to within **1.5%** across boots -- the
design doing exactly what it is for. `video.rect` only half works: the
graphics group still moves 20 to 30% once it is divided out, because
`desk::draw` depends on what is on screen and a rectangle does not. And
`ai.matmul` and `smp.*` have no control at all, so on a busier day they report
a regression that is the day, with nothing to divide out and nothing to say
so. `tools/rails.py` declares that hole rather than leaving it to be noticed.

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
trim/starts/ends/contains/chr/ord/repeat/pad/hexenc/hexdec), a four-rung
numeric tower (below), lists
(sort/reverse/slice/index/remove/range/push/set/get/len), the namespace
(read/write/ls/exists/rm/size/is_dir/hash_of/applet), the clock and counters
(rtc_now/rtc_unix/uptime/tsc/tsc_mhz/ticks/hz), tasks and memory, `pci_list`,
network status, sockets (dns_resolve/tcp_*/http_get/https_get/udp_send/ping),
the model (`ask`), the framebuffer, and raw memory and I/O ports.

### Numbers, in four rungs

This said "no floats -- adding them for one builtin changes every arithmetic
path" for a long time, and the objection was right about the *path* and wrong
about the conclusion. What it argued for was making inexactness opt-in rather
than absent:

    Int(i64)            10/3 is 3, as it always was
    Rat(i64, i64)       rat(10, 3) is 10/3, exactly
    Qty(i64, i64, Dim)  qty(9, "m/s^2") is 9 m/s^2, and adding seconds is an error
    Approx(f32)         real(2) is ~2, and the tilde is in the rendering

**Every rung is reached through a named builtin and by no other route**, which
is what keeps the promise that nothing written earlier moved: `rat` is the only
way to make a fraction, `qty` the only way to attach a unit, and `real` plus the
transcendentals the only ways to make an approximation. There is no float
*literal*, deliberately -- that is what keeps `.` unambiguously field access.

Each rung has a canonical form and that is load-bearing. `rational` collapses a
denominator of one back to `Int`, `quantity` collapses an empty dimension back
to a plain number, so `3` and `3/1` and `3 dimensionless` are one value and not
three that render, hash and compare differently.

Two things worth knowing before touching the arithmetic:

- **This target cannot divide a 128-bit integer.** `__udivti3` and `__umodti3`
  link and do not return; one `u128 %` in a gcd loop stopped boot with no fault
  and no output, three sections before the shell. `rat_binary` therefore
  cross-reduces in `i64` and says so. A 128-bit *comparison* is fine -- it is a
  subtract -- which is why the orderings may use one.
- **Branch order in `binary` is Qty, Approx, Rat, Int**, and it is not
  cosmetic: a quantity meeting a fraction would otherwise route through
  `rat_binary` and silently lose its dimension.

`num_cmp` is one ordering that `min`, `max`, `clamp` and `sort` all read. It
agrees with `<` everywhere except NaN, deliberately, and both sites say to read
the other before changing either.

### The standard library at `/lib`

Seven files of Aiksi source, compiled into the image with `include_str!` and
seeded into `/lib` at every boot from `aiksi::LIBS`. Compiled in for the reason
the routing corpus is: `/lib` is where a sandboxed program's dependencies are
allowed to live, so a machine that has never mounted a store still has to have
one.

| | |
|---|---|
| `prob` | combinatorics and the exact half of statistics; a probability *is* a fraction |
| `geom` | plane geometry over exact coordinates, and a convex hull that cannot contradict itself |
| `mat` | linear algebra; Gaussian elimination where "is this pivot zero" has an honest answer |
| `num` | number theory, and modular arithmetic with no 128-bit intermediate anywhere |
| `poly` | polynomials, including the two operations whole numbers cannot express |
| `phys` | physics in quantities that carry their units |
| `chem` | formulas parsed, molar masses summed exactly |

**`/lib` belongs to the image and not to the machine.** It is re-seeded
unconditionally at boot and after restore, so an edit made there does not
survive a reboot; a program needing its own version keeps it in its own subtree,
where the jail admits it anyway. The guard that used to be there asked whether
`/lib` was *empty*, which froze every snapshotted machine at whatever library
set it had when it first snapshotted -- the seventh library simply never
appeared, with nothing said.

**Three structural claims walk `LIBS` at `diag lib`** and they exist because of
one trap: a user function **shadows a builtin**, deliberately, and `use` is
textual inclusion. So a library declaring `trim` silently takes the string
builtin away from every program that imports it. The claims are that every
library imports and runs its top level, that no declared name is a builtin, and
that no two libraries declare one name. `/lib/poly` was written with exactly
that `trim` and with an `area` colliding with `/lib/geom`'s, and this is what
found both.

The headline measurement is `inv(hilb(3))`. The Hilbert matrix is the standard
ill-conditioned example and its inverse is a matrix of *whole numbers*, which no
floating-point library recovers at any precision. This one does.

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

None of this prevents anything **here**. Faults are recoverable inside
`cpu::recover::guard` and generated code is usually not called from inside one,
so a bad jump is still a halted machine. What the registry buys is that the one
diagnostic which survives says something true -- and that matters more now that
`boot_report` turns a boot-time halt into a transcript, since a transcript
naming an offset into an anonymous buffer is a different thing from one naming
an rva in nothing.

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

**And there is a code generator now.** `src/aiksi/jit.rs` compiles a program
of integer functions -- arithmetic, `if`, `while`, `return`, and **calls
between them, recursion included** -- to x86-64, emits it into an `Exec`, and
calls it through the `sysv64` pointer `cpu::code` pins. No builtins, no
strings, no records, no `use`. Anything outside that slice is **refused** --
`compile` answers `None` and the interpreter remains the only thing that ran
it -- and ten claims check that refusing actually happens, each naming its own
subject, because a generator that quietly compiled a string return would be
answering a question nobody asked.

It is reached only from `differ`, never from a live path. Nothing routes
through it and `voter` does not know it exists.

**Locals live on the machine stack and they had to move there.** They were a
flat array inside the context structure, which is correct for exactly as long
as one frame exists: a recursive call would have written its parameters over
its caller's, and the tell would have been a `fib` that answers confidently and
wrongly with every step count still matching. So a function gets a real frame
-- `rbp`, locals under it, arguments pushed by the caller and copied in by the
prologue -- and the depth cap is **read from `eval::MAX_DEPTH`** rather than
copied, because two numbers that have to agree and are written down twice are
two numbers that will not. No guard page here, so running off the stack is a
triple fault rather than a message.

**A call costs no tick of its own**, which is a fact about `call_user` rather
than a convenience: it pushes a frame, checks the depth and runs the body, and
ticks at none of them. Charging one per call is the obvious thing and is wrong
by one per call, on a number the judges read.

**Nil is a value the interpreter has and the compiler does not.** A function
that falls off its end yields it, and inside an expression that means
reproducing what the interpreter says about `nil + 1`. So a *called* function
must declare `: int`, checked at the call site, which is exactly where the Nil
could escape; an unannotated entry may still fall off, because nothing consumes
what it answers. A callee that yields nothing is refused by the interpreter
**by name**, so the blame index travels with the status -- a message naming the
entry when a callee three frames down was at fault is the same failure
`boot_report` carries a field to avoid.

**And the first measurement of why any of this is worth doing.** `core bench`
runs `fib(18)` both ways from one parse and prints the ratio only when the two
routes agreed on the value *and* the step count:

    fib(18) = 2584, 83606 steps, two ways:
      tree-walk            5406 us
      generated code       102 us
      ratio                52x
      parse and compile    16 us, paid once

That does not contradict the finding above that the tree-walk is a twentieth
of a vote. A vote is twenty steps of walk behind fixed setup; this is 83,606
of it. The ratio is what says which kind of program a compiler is for, and the
compile is repaid three hundred times over by a single run of one.

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
on a fresh machine -- the line says `0 agreed, 3 declined`.

Console output **is** compared, and that sentence read "is not" here for a
while after it stopped being true. It is the hole this harness documented
against itself and then closed with `begin_capture`, which the console had had
all along: a program of `println`s answers nil however it behaved, so two
routes could disagree about everything a person would notice and agree on
every field. `skill.rs`'s J3 had the identical hole and is closed the same
way.

**And comparing it found a defect in the console rather than in either
route.** `diag differ` failed on `a runaway, stopped by the budget at the same
step` -- steps, value and error agreeing exactly at 100,001, and only the
console differing, on a program that has no builtins and therefore cannot
print. That shape is contamination and not divergence.

`CAPTURE` was one stack for the whole machine, so a line printed by *any* task
landed in whichever capture happened to be innermost. The runaway spends
100,001 interpreter steps inside a capture, about twenty-eight timer ticks, and
one boot in several had something land in that window. A capture carries the
task that opened it now: `_print` writes to the innermost capture belonging to
the *current* task and anything else falls through to the console, which is
where it went before captures existed.

`end_capture` pops this task's innermost rather than the stack's top, or two
tasks capturing at once would hand one task's output to the other. And
`serial::_print` suppresses the log only while *this* task is capturing, which
is the same correction one layer down: the clock task's output was being
dropped from the transcript whenever the shell happened to be capturing.

`CLAUDE.md` had already recorded this class once -- `[mind t1] disabled`
splitting `257` into `25` and `7` -- as an interleaving artefact rather than as
a bug with a fix. It was both.

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

`todo` (shell) and the ToDo window share one list. It is the hand-off note for
what to test at the GF63, and this said "the machine that builds this is a
different machine from the one that runs it", which is false: `Win32_ComputerSystem`
reports `Thin GF63 12UC`, board `MS-16R8`, so the development host **is** the
target. The hand-off is real and the reason was wrong. What is true is that the
two cannot run at once -- `deploy.ps1 -EspDrive S:` writes the external SSD and
testing means rebooting into it on F11, which takes the editor, the browser and
this file away with it. So anything you wanted to look up while GLaDOS is
running has to have been written down first, which is exactly what the list is
for.

**Apps are `Content::App(Box<dyn DeskApp>)`** (`gfx/mod.rs`): a window whose
client area belongs to a program. There are ten: Paintbrush (`paint.rs`), Write
(`write.rs`), Minesweeper (`mines.rs`), ToDo, Ask, Flows, Improve, the Oracle,
the agent log and the authoring progress window (`agentwin.rs` holds the last
two). Six methods (draw, key, press,
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

  **That last sentence was wrong for as long as there have been other cores,
  and it is the reason this entry is worth reading twice.** `TICKS` is one
  global counter and `timer_isr` incremented it unconditionally -- while every
  application processor starts its own periodic timer on the same vector, with
  the same handler, because `init_this_core` deliberately does not re-register
  per-core entries. So `ticks()` advanced at **N x TIMER_HZ**, and every
  duration derived from it was wrong by the core count.

  The expensive one was `tcp::wait_until`, whose deadline is
  `ticks() + ms * TIMER_HZ / 1000`: **every network timeout in the kernel was
  short by the core count.** A 15 s TLS deadline was 3.75 s under the tooling's
  default `-smp 4`, and would be under a second on the GF63's sixteen logical
  processors -- which is a candidate explanation for fetch and handshake
  failures on hardware that never reproduced here. `uptime` was wrong the other
  way, and the model selftest's tokens/sec under-reported by the same factor,
  since it both sampled a shorter window and divided by too many ticks.

  Measured: a 4-core guest reported **55.58 s of uptime during a 25 s run**,
  longer than the whole invocation including QEMU startup, against 8.01 s for
  the same script at `-smp 1`. Only the bootstrap processor increments now.

  Two things about how it hid for so long. The `[selftest] timer` line printed
  "N ticks in ~0.5 s" where the 0.5 was **a constant in the format string**, so
  it read identically however fast the counter was really moving; it is timed
  against the TSC now and fails if the two clocks disagree.

  **And for a long time after that it still could not have caught this**, which
  is worth more than the original bug. `time::calibrate` derived the microsecond
  from `ticks()` itself -- `elapsed_us = ticks * 1_000_000 / TIMER_HZ` -- so a
  tick rate wrong by a whole core count scaled the calibration and the check
  alike and divided straight back out. The comparison was arithmetic wearing a
  measurement's clothes.

  Driven rather than argued, by making the ISR increment by two:

      before   tsc 1345 MHz (a true 2688)
               ok   50 ticks in 492 ms -- the two clocks agree
      after    tsc 2688 MHz, measured against PIT
               FAIL 50 ticks in 248 ms -- ticks() disagrees with the TSC

  `uptime` read 22.54 s at 11.3 s of real time on both. A check with a hundred
  per cent false-negative rate for the one bug its own message names, passing
  cheerfully on a machine where every `ticks()`-derived timeout was half what
  it should be.

  The fix is that the TSC now has a reference that owes nothing to `ticks()`.
  `lapic::calibrate` already busy-waits on the PIT across an exact 10 ms window
  to measure the APIC timer, so two `rdtsc` reads around that same loop give a
  TSC frequency for free; `calibrate_pm` does the same against the PM timer,
  whose rate is architecturally fixed. `time::calibrate` prefers it and keeps
  the tick-derived loop only as a fallback -- and when that fallback is what
  ran, the boot check **says the two cannot be compared** rather than reporting
  agreement, because agreement would be a statement about division.

  It is also the steadier clock, which is measurable rather than asserted.
  Across ten boots before and eight after, the TSC figure went from 2672-2771
  MHz (3.7%) to 2687-2694 (**0.26%**), fourteen times tighter. That matters
  because the check reads that figure: a `tsc_mhz` inflated by a host stall
  inside the old 50 ms calibration window made every later reading
  proportionally short, which is what produced an intermittent FAIL at roughly
  **one boot in twelve** under WHPX -- 344 ms against a floor of 350, where a
  healthy boot reads 492 to 560. Nine post-fix boots have not reproduced it,
  which is not enough to call a one-in-twelve event gone; what is established
  is that its dominant cause is fourteen times smaller.

  The band stays 350..=750 deliberately. Two cores read 250 ms and three read
  167, so the floor keeps a factor of 1.4 under the smallest error worth
  catching, and widening it to quiet the flake would have given back the
  detection that was just bought. And the benchmarks
  that *are* trustworthy -- `smp bench`, `video bench`, `core bench` and the
  decode figures -- all use `rdtsc`/`tsc_mhz`, which is exactly why the decode
  numbers came out consistent across 1, 2 and 4 cores. Had they been
  tick-based they would have differed fourfold. **Anything measuring a duration
  should use `rdtsc`; `ticks()` is for wall-clock-ish elapsed time and nothing
  else.**
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

### Surviving a selftest, and repairing what broke

**The first bare-metal boot on the GF63 died of a thermometer.** A `#GP` in
`dev::power::hwp_range` -- reading `IA32_HWP_CAPABILITIES`, which is gated
behind `IA32_PM_ENABLE` and which nothing downstream needs -- halted the
machine before the shell existed. No storage, no namespace, no model. The only
positional information on screen was `rva 0xaa013`, which needed a second
computer and `llvm-objdump` to turn into a function name.

Four pieces answer that, and each is useful without the ones after it.

**Symbolication.** `.cargo/config.toml` passes `/MAP:target/glados.map` --
`lld-link` takes MSVC flags -- and `tools/symbols.py` turns the map into
`src/cpu/symbols.rs`: an rva-sorted table and one names blob, 13,559 symbols
and 546 KB, which takes the image from 4,221 to 4,916 KB. `cpu::code::symbol`
binary-searches it and allocates nothing, because it is called from a fault
handler. Two traps paid for: v0 Rust mangling is **length-prefixed**, so a
greedy regex returns `glados` for everything and it needs a scanner; and the
build stamp is a content hash of the table rather than the linker timestamp,
which never converged and named the *previous* link.

`scripts/deploy.ps1` builds, regenerates the table and relinks -- two passes,
necessarily, since the table describes the image it is then compiled into. The
stamp is what says whether the one in the binary belongs to it.

**Guards fit to wrap a selftest.** `PADS` is a per-task stack of depth 4, so a
guard inside a guard no longer disarms the outer one on its *normal* exit.
`guarded` answers three states rather than two: `Ran`, `Faulted(why)`, and
`Unguarded`, which means the closure ran with no landing pad and therefore
**proved nothing** -- treating that as a pass is exactly how a check that never
protected anything looks like one that passed. `selftest_window(bool)` lets the
panic handler consult a pad, and only there; everywhere else a panic halts.

Two hazards that are silent, both already paid for. A fault mid-`kprintln!`
abandons `CAPTURE` and `CONSOLES` held, and `Spin`'s patience limit **panics**
on the next acquire -- a recovered fault becoming a fatal one -- so the caught
path calls `console::release_locks()`, which is entitled to exactly those two
and says so. And **a provably-divergent closure loses its landing pad**: the
optimiser deletes the unreachable tail, longjmp target included, and the guard
lands at an rva past `.text`. `guard_inner` calls through
`if core::hint::black_box(true)`, and the suite's case is a bare `panic!` so it
asserts the defence rather than working around it.

**Criticality is per subsystem.** `main::section(name, need, f)` wraps a boot
check; `Need::Vital` halts with a reason, `Need::Optional` records the failure,
prints one red line and carries on. **Seventeen are wrapped today**, where
seven were: `sysbox`, `crypto` and `rng` vital, and `heap`, `version`, `timer`,
`clock`, `json`, `websocket`, `html`, `css`, `power`, `fmt`, `usbhid`, `text`,
`mining` and `code` optional.

`Optional` on all ten of the newly wrapped ones deliberately. What wrapping
buys is that a failure is *recorded and named* rather than fatal or silent;
escalating any of them to `Vital` is a separate decision wanting evidence from
the GF63 about, for instance, how wide the timer's band really is on hardware,
and a `Vital` false positive is an unbootable machine, which is exactly as bad
as the miss it would be protecting against.

**Two are still unwrapped and the reason is a type rather than an oversight.**
`acpi::selftest` and `acpi::aml_selftest` both take `acpi_ref`, so a closure
around either captures, and `section` wants a `fn` precisely because a check
that cannot be re-run is a check no repair can be judged against. Making them
re-runnable means giving `acpi` a handle that outlives that call.

It takes `fn() -> bool` rather than `impl FnOnce()`, and both halves of that
are load-bearing. The `fn` is because **a failure you cannot re-run is a
failure you cannot repair**, and re-running the check is the only judge a
repair has.

**The `-> bool` is newer, and its absence was the largest hole in this
machine's view of itself.** Every subsystem wrapped here answers a verdict --
`sysbox::selftest`, `crypto::selftest`, `rng::selftest`, `fmt`, `usbhid` and
`code` all return one -- and every call site discarded it; `|| {
sysbox::selftest(); }` was literally what was written. So the boot and repair
loop was a *liveness* oracle wearing a correctness one's name: a change that
made ChaCha20 return the wrong bytes without faulting was not recorded in
`boot_report`, not counted by `outstanding()`, never offered to `repair`, and
did not stop the boot even when `Vital`.

The two failures are recorded identically now and the report says which. A
fault is a subsystem that is **gone**; a `false` is one that is **wrong**, and
wrong is the more dangerous of the two everywhere a wrong answer still looks
like an answer. `power` is the one section that answers `true` unconditionally,
and that is honest rather than left over: what it can observe is whether
reading the registers takes the machine down, which is the GF63's bug exactly,
and every value claim `dev::power` can make without hardware is already `diag
power`.

Verified by injection, since nothing else settles it. With `crypto::selftest`
returning `false` from an otherwise untouched run, all 26 of its own claims
printed `ok` and then:

    [selftest] crypto failed its own checks -- and this machine needs it
    [boot] 1 subsystem(s) did not survive their own selftest, 1 still broken:
      crypto         failed its own checks  (vital)
    [boot] crypto is vital, so this machine will not continue.

with no site line, since there is no faulting instruction to point at. Before
the change the same boot reached the shell and answered `alive`.

**`repair::judge` had the same shape and needed the same fix.** It asked
`matches!(.., Caught::Ran)`, which is whether the check *finished*, so an
action that left a subsystem alive and answering wrongly would have been
adopted, marked as the repair that worked, and written to the boot volume for
every boot after. It asks for both now, and `diag repair` carries the claim for
the outcome that could not be expressed before: runs to the end, answers no.

And `boot_report::record` takes the rip as an argument rather than reading
`recover::site()` itself. `LAST_RIP` is a global that outlives the fault that
set it, so a check answering `false` would otherwise be filed at whatever
address broke last -- the "report names the wrong subsystem" failure the field
exists to prevent, arriving from the other side. `recover::take_panic` zeroes
it on the same argument.

**And the wrapping found two bugs on its first run, which is the argument for
doing it.**

The heap check had been printing `after drop: 399104 B LEAKED` **in red on
every boot** and nothing consumed the verdict. It compared the heap against
*zero* after its own objects dropped, which is the same question only while
nothing else in the kernel has ever allocated -- and the console's scrollback
ring, among others, is long since resident by the time the selftests run. What
it means is whether *this block's* allocations came back, so it is a delta
against a baseline taken a line earlier. It reads `back to 399104 B` now.

And `diag census` started failing, deterministically, on every boot, because
ten functions were added to `main.rs`. Nothing about the allocator or the
census changed. **Its 512 KiB test allocation was being deleted by the
optimiser**: Rust marks the allocator functions so an alloc/dealloc pair with
no observable effect can be removed outright, and the vector was filled once
and dropped. The tell was which claims failed -- `taken`, `count` and `given`
all stood still while `peak` passed on history alone, which is the signature of
an allocation that never happened rather than one billed to the wrong row.
`core::hint::black_box(v.as_mut_ptr())` makes the pointer escape so the
allocation has to exist. Same defence `recover::guard_inner` needs against the
optimiser deleting a landing pad, and worth stating as a rule: **a claim whose
subject the compiler can prove is unnecessary is not a claim**, and it will
pass until an unrelated edit somewhere else in the crate flips it.

**The site is recorded, because recovering throws away the only evidence of
where.** A caught fault is attributed to whatever scope was guarded, which
answers *which check failed* and not *what broke*. Those differ the moment a
check calls into something else -- a fault inside the graphics stack reached
from a power selftest is a power failure by attribution and a graphics one in
fact. `recover::site()` carries the rip through, `boot_report` symbolicates it
at print time, and the report prints both. Verified by faulting inside
`doom::pic` from the `power` section: the report named section `power`, site
`doom::pic::Art::flat +0x27`.

**The repair loop is the Troubleshooter's bargain**, which was a good one: a
fixed set of deterministic actions, one applied, and then **checked**. It never
ran during POST either -- it booted, looked, fixed, and made the fix stick.
That ordering is forced here anyway, since `ai::init` needs the namespace and
the selftests run long before either.

`src/repair.rs` is an allowlist, two rows today, which is the honest size of
the set of knobs that exist rather than a claim the machine can fix anything.
`retry` earns a row because "it did not happen the second time" is a real
outcome worth recording as the repair that worked. `skip-hwp` is the GF63 bug
with a switch in front of it. Most useful repairs turn something off or down;
so did the Troubleshooter's.

**The fixed rule was built first, deliberately**, and it is still the fallback:
try each offered action in table order, keep the first whose judge passes,
revert the ones that did not, so whatever is left standing is exactly the one
that worked. Proving apply/judge/revert somewhere a decode cannot be blamed for
had to come before a decode was allowed near it.

**The fault's own signature picks the repair, and no model is asked.** A fault
carries a vector and a symbolicated site, and those two say far more about which
knob is wrong than any amount of reasoning about names --
`dev::power::hwp_range +0x13` is not a hint, it is the answer. So `Clue` is
`Fault(name)` or `SiteContains(part)`, every clue on an action must hold, and
`skip-hwp` carries the GF63's own: a general protection fault inside
`dev::power`. Both halves matter -- a *page* fault there is some other bug and
this knob would not touch it, and a `#GP` outside `dev::power` is not an MSR
gate problem.

`rank` sorts what is offered into three groups and **drops none of them**:
matched, then actions asking for nothing, then actions that asked and did not
get it. The third group is kept because a clue is evidence about what is likely
and never a proof about what is possible, and discarding would turn a wrong
guess about a signature into a repair the machine can no longer reach. The judge
still decides, so the ordering is allowed to be a guess -- being wrong costs an
apply and a revert, never a bad repair.

**Plain table order was wrong, and this is what made it visible.** `retry` was
first, and retrying can only ever help a *transient* fault, so on a
deterministic one the first attempt was guaranteed waste. An action with no
clues is a fallback now and sinks below anything whose clues held.

`rank` is pure over the failure record, so nine claims assert it with no model,
no disk and nothing injected -- the `update::decide` discipline, and the reason
this replaced a decode rather than sitting beside one. Among them: a different
fault in the same subsystem does not match a signature, the right fault at the
wrong site does not either, and every vector a clue names is one
`recover::describe` can actually report. That last is a list rather than a match
arm now, because renaming a vector would otherwise stop every signature matching
with no error at all -- a machine quietly repairing itself worse than it used
to.

**The model is still there and off by default.** `author::choose` over the same
table, behind `repair model on`, falling back to the ranking on no model, a busy
engine, or three decodes that will not commit. It is kept rather than deleted
for the reason the SGD head and the role adapters are kept: the apparatus is
what lets somebody cheaply re-ask the question on a checkpoint that is not the
one it was measured on.

`offered_for` is narrow on purpose, and `diag repair` is built around that: a
chooser that could pick a power register knob for a graphics fault is one
decision away from a second fault in a subsystem nobody was repairing. Both
halves of that claim are **derived from each row's own list** rather than
written down, so renaming a row cannot leave the suite passing while testing
something that no longer exists; and the judge claims run against synthetic
checks belonging to no subsystem, so what they measure is the judge rather than
a particular repair.

Four more claims are that rule arriving on the other side of the loop.
`offered` stops the wrong knob being *tried*; these stop the model being *told*
about a fault other than the one that happened -- the prompt must name the
subsystem that failed, carry the real symbolicated site, mention no other
subsystem the table knows about, and offer exactly the rows this subsystem is
offered. `boot_report::site_of` is one function for that reason: a chooser shown
less than the operator is guessing about a fault somebody else can see, and one
shown *different* words makes decisions nobody can check against what was
printed.

**The transcript says who chose**, because three different things land on index
zero and they mean different things: `forced` (one option), `table` (the model
is switched off), `no model`, `undecided` (three decodes that would not commit)
and `model`. Without that column a boot where the engine was busy reads exactly
like a boot where the model picked the first row.

`judge` saves and restores the selftest window instead of closing it. It is
callable from inside a suite -- `diag repair` does exactly that -- and closing
it would silently take panic recovery away from every check after it.

Measured under QEMU, with a fault injected into the `power` section that
`skip-hwp` genuinely fixes:

    [selftest] power page fault -- this subsystem is unavailable
    [repair] 1 subsystem(s) to try
      power          repaired by 'skip-hwp', and the check now passes
    [boot] 1 subsystem(s) did not survive their own selftest, 0 still broken:
      power          page fault  (skip-hwp)
    glados> echo alive
      alive

A repaired subsystem stays listed, because "was broken and is now repaired" is
a different fact from "never broke" and an operator is owed both. The header
counts what is *still* broken.

**A repair survives a reboot**, in `\GLADOS\REPAIRS.TXT` on the ESP -- the
only durable channel that needs no human, since a namespace write lands in
memory and reaching NVMe needs `store unlock`, which is a person, once per boot.
`repairs::at_boot` runs inside `update::hook`, which is the earliest point there
is: every subsystem a repair could protect initialises later, and a repair
adopted after `power` has already faulted arrives one boot late.

**Nothing the file says is executed.** Two words are resolved against
`repair::ACTIONS`, the row is what runs, and `offered_for` is checked on the
way -- which is not decoration when the strings come off a FAT partition
anything can edit. That is the same bargain `author::choose` makes: the chooser
names a row, the kernel owns what the row does.

**The safety property is `update`'s health flag with one change, and the change
is the interesting part.** That flag is resolved before `ExitBootServices`,
because the firmware's FAT driver is the only writer of the ESP that exists
while a *boot image* can still be swapped. Nothing here swaps anything, so the
constraint does not apply -- and this kernel writes its own ESP afterwards over
NVMe, which `update stage` has always done. Clearing early would have bought a
window from the hook to the memory map: long enough to catch a repair that stops
the model loading, and blind to every repair that faults a subsystem, which is
the entire population this table can produce. `repairs::survived()` clears it at
the shell instead, so the window is the whole boot.

A machine that cannot write its ESP therefore withdraws one repair per boot.
That is the safe direction -- a machine nobody can talk to reverts to
unmodified -- and it is written down rather than left to be found.

**`--esp-on-nvme` is what made any of this testable.** The harness put the ESP
on its own drive and the NVMe image on another, so the kernel's own block layer
-- which reads NVMe and nothing else -- could not see the volume it booted from,
and `find_esp` correctly answered that no NVMe partition carries
`BOOTX64.EFI`. The GF63 has one disk with the ESP as a partition of it, and OVMF
enumerates NVMe as a boot device, so the fix was topology rather than code.
Driven over six boots on a real FAT32 volume: recorded, applied at the next
boot, applied again at the one after (so the trial clears), killed at 12 s
before the shell, **withdrawn automatically** on the boot after that, and clean
on the boot after that.

**`retry` is deliberately not persistable.** It applies nothing, so recording it
asks the next boot to run a check that boot runs anyway -- a line in a capped
file that can never change an outcome and would push a real repair out of the
eighth slot. `Action::persist` is that distinction and `record` refuses without
it.

**And a repair never silently replaces a fix.** A subsystem held up by a repair
and one passing because somebody fixed the bug look identical from everywhere
else, so `recheck_persisted` takes the repair away, runs the check again,
reports if it passes, and puts it back whatever the answer -- withdrawing a
repair the machine has been relying on is the operator's decision, not a side
effect of looking. That needs the check, which only the *failures* were keeping,
so `section` registers every check now. QEMU is the honest demonstration, since
it reports `hwp no` and `power` genuinely does not need the repair there.

The operator's half is the `repair` verb: `repair` says what is applied, what
the boot volume records and what actions exist at all; `repair record` and
`repair clear [n]` write one down and take it back; `repair off` reverts what is
applied without touching the file, since undoing a repair for this boot and
forgetting it forever are different decisions.

**The prompt shape was the whole difference, and it cost a run to find.** The
first version glossed every row -- `skip-hwp (stop reading the hardware-managed
performance registers)` -- and came back `no choice among 2 after 0 step(s)`
three times running. Zero steps means the decode never entered an alternative at
all: it laid out whitespace until the idle allowance ran out. The prompts that
work in this tree are one short sentence ending in a question, which is what
`voter` asks and what `author::choose`'s own note describes. Reshaped, it
commits. The `about` column is therefore *not* prompt text any more, and says so
where it is declared.

**And then the measurement, which is the part worth reading.** Six boots with a
fault injected into `power` that only `skip-hwp` fixes -- three with the table
in its own order, three with the offered list reversed:

    [retry, skip-hwp]     retry     retry     retry
    [skip-hwp, retry]     skip-hwp  retry     retry

It picked `retry` **five times in six, wherever `retry` sat**. Reversing the
list changed the answer once, which is what rules out the obvious reading: this
is not a model taking whatever is listed first, it is a model preferring a
*name*, and the name it prefers cannot fix this fault.

The mechanism matters before anybody adds a row. `retry` is one common English
token; `skip-hwp` is several uncommon pieces. Under a constrained grammar the
cheapest first-token path wins, so **an action's name carries probability mass
that has nothing to do with what the action does**. Naming a row is not
cosmetic here.

So the decode bought nothing on this table: table order tries `retry` first
too, and six boots of choosing produced exactly what the fixed rule produces,
for the price of a prefill. What it did not do is any harm -- the machine was
repaired on all six, because the judge caught the bad pick and the loop moved
on, which is the whole argument for this arrangement arriving as a measurement
instead of a claim.

**And the deterministic rule beats it on the one case there is evidence for.**
Measured against a fixture reproducing the GF63's shape -- a `#GP` raised inside
`dev::power` -- the ranking repairs it in **one** attempt where the model took
two:

    model   retry (wrong) -> skip-hwp    2 attempts
    rule    skip-hwp                     1 attempt

So the model was switched off rather than deleted. The measurement was on
SmolLM2-135M, the checkpoint that fits under QEMU and not the 0.6B the machine
runs, and concluding anything about the shipped model from it would be the
small-sample extrapolation this file warns about elsewhere. `repair model on` is
how somebody re-asks, and `repair log` is the transcript.

**Three things the testing turned up, none of them about repairs.**

`tools/symbols.py` **writes nothing without `--emit`.** A whole session's worth
of "regenerate the symbols" parsed the map, printed a count and left the file
alone, so the table was stale against every build -- and a convergence check
comparing a file nothing was writing converged instantly and meant nothing.
`deploy.ps1` passes `--emit` and was always correct, so this cost testing time
and never a deploy. Emitting the table changes the layout it describes, so it
takes two or three passes to reach a fixed point.

**`cpu::code::symbol` can name the wrong function, and the offset is the tell.**
The table holds public symbols, so anything inlined has no entry and the search
returns whatever precedes it. A deliberate fault in a small `dev::power` helper
was reported as `doom::play::dispatch +0x14a2` -- five kilobytes into an
unrelated function in an unrelated subsystem. Average spacing over 13,817
symbols is about a hundred bytes, so an offset in the thousands means the real
function is *absent* rather than enormous. `#[inline(never)]` puts one back in
the table. Not corrected silently, because the map carries no sizes and there is
nothing to correct it to.

**QEMU answers zero for a read of an unimplemented MSR instead of raising
`#GP`.** The first fixture was an `rdmsr` on a reserved register -- the real bug
-- and it did not fault at all. That is the same emulator gap `dev::power`
already records as the reason its gate cannot be checked here, arriving one
level down. The fixture uses a non-canonical address instead, which is `#GP` by
architecture rather than by model.

**What is not built.** No `Probe` is fitted over `/ai/repair/log`. That is the
point of keeping the transcript -- one line per attempt, symptom and action and
outcome -- but with zero examples a fitted router has nothing to beat a grammar
decode with, so it waits until the corpus exists.

**And the end-to-end test no emulator can produce.** Every mechanism above has
been driven under QEMU with an injected fault, which is not the same as the real
one: QEMU reports `hwp no`, so the GF63's actual `#GP` cannot reproduce here at
all. The sequence of that machine faulting, being repaired by `skip-hwp`,
persisting it and booting clean is still owed, and it needs the laptop.

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
not implement the register behind it, and under emulation the first two are a
guess that does not return. **So none of the readings can be checked under
QEMU**, which reports "vendor intel, hypervisor yes" and then declines with its
reason. `power force` overrides it and says what it is risking.

**"On real silicon the first two suffice" is what this used to say, and the
first boot on the GF63 disproved it.** `power` printed every line down to the
governor and then took a #GP; the reporter's `rva` disassembled to `rdmsr` with
`rcx = 0x771`, which is `IA32_HWP_CAPABILITIES`.

CPUID saying HWP exists is not permission to read that register. It is gated
behind `IA32_PM_ENABLE` bit 0, and this laptop supports HWP and boots with it
off, so the very first read faulted. The gate asked whether the register
*exists* where the processor's rule is about whether it is *enabled* -- two
different questions, and only the first was being asked. `hwp_enabled()` is
the fourth condition, and it reads the one register in the group that is safe
on a part advertising HWP, because being the architectural enable is what
`IA32_PM_ENABLE` is for.

`set_governor` had the same fault by a second route: it called `hwp_range()`
*before* setting `PM_ENABLE`. Enabling now happens first, which is honest there
because that function exists to change the policy -- and deliberately does not
happen in `hwp_range`, since a status command that switched the machine's power
management on as a side effect of being asked a question would be the worse
bug.

None of this was reachable under emulation: `allowed` declines under a
hypervisor before any of it runs, and QEMU reports `hwp no` regardless. **It is
the first bug in this tree that only bare metal could find**, and the thing that
found it was the `rva` line the fault reporter prints.

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

This began as a **compute fabric rather than general SMP**, and it has moved.
The extra cores can allocate and print, because those two structures are behind
real locks.

**They take interrupts and they run tasks now, and this file said otherwise for
longer than it was true.** It read: "They still never take an interrupt and
still never run a task, and the reason is specific: an application processor
runs on the trampoline's flat descriptor table with no task-state segment, so
its code selector does not match the one the interrupt table's entries name.
Preempting a task there needs a per-core GDT and TSS." That was an accurate
description of the obstacle and the obstacle was removed. `smp::init` calls
`gdt::adopt`, `percpu::adopt`, `idt::load_this_core` and
`lapic::init_this_core` on every application processor and then starts its
timer, so each core has its own descriptor table, task-state segment, per-core
block and idle task.

`diag migrate` is the evidence and it is worth knowing what it actually does:
it spawns a task, calls `unpin` on it, and waits until the core it has been
seen on has more than one bit set -- sampled until seen twice rather than once,
because whether a second core picks the migrant up inside any particular 200 ms
depends on what else is running, and a single sample can fail spuriously as
easily as it can pass spuriously.

**The audit is per task, and there is exactly one task that has passed it.**
`Task.pin` carries the only core allowed to run a task. Preemption on one core
means two tasks never execute at the same instant; on two they genuinely
overlap, so every `Racy` reachable from two tasks stops being a promise and
becomes a race. There are 92 of those, the namespace tree among them. Unpinning
before the audit buys a kernel that passes every test and corrupts something
later, so migration is opt-in per task and the opt-in *is* the audit.

This said "nothing outside the selftest calls `unpin`" until the mining slices
did. `mine::client::set_slices` is the first real caller, and what the claim
cost is worth knowing, because it is the price of every future one: the
miner's shared surface was audited to a single object -- its journal, which was
a `Racy<Vec<String>>` and is a `Spin<Vec<String>>` -- and everything else a
slice touches is an atomic, a `Spin`, or its own stack. `mine::work`'s table is
one `Spin` per slot for the same reason. The other 91 `Racy`s are untouched and
every other task is still pinned to core 0.

The evidence that it works is arithmetic rather than a passing test: four
slices summed to 256% of one slice's hash rate, and tasks sharing a core sum to
100% however many there are. `tasks` shows them being resumed independently.

The compute-fabric path is unchanged underneath all of that: helpers wait on a
generation counter with MONITOR/MWAIT, run a range of a matrix, and go back to
sleep, with every decision and every byte of kernel state still on the
bootstrap processor.

**A caution for whoever reads this next.** The README's status table says
"per-core GDT/TSS/APIC ... tasks that migrate" while its limitations list says
every task is pinned to core 0. Those looked like a contradiction and were not:
the mechanism worked and was deliberately unused. The limitations line is now
genuinely stale rather than deliberately conservative -- the mining slices
migrate -- and the honest edit is "every task but the mining slices", not
deleting the line. Do not resolve it the other way by unpinning something else
to make the README true; the pinning is the audit.

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

### Which driver claims which device

`src/dev/registry.rs`. Every driver used to answer that for itself: sweep all
256 PCI buses, compare against a private array of ids, take the first hit.
Nine drivers, nine sweeps, nine private lists, and **no single place that could
say what the machine contains**. That is survivable on one laptop and stops
being survivable the moment the answer has to be right on a machine nobody here
has ever booted.

**The matching rule is data.** A row is a bus, a rule, a role, a name and a
support level, and adding hardware support is adding a row. Three kinds of
rule, because hardware identifies itself three ways and which one applies is a
property of the part rather than a choice: NVMe reports a programming interface
every conforming controller reports, so one row drives every SSD in the world;
an RTL8188EU dongle reports a vendor-specific interface and no useful class at
all, so the id list *is* the detection. Getting that backwards is how a driver
misses hardware it could drive, or claims hardware it cannot.

**Specificity is ranked rather than resolved by table order.** An e1000
satisfies both `Ids(0x8086, ...)` and `Class(0x02, 0x00)` and the specific one
has to win. Order works right up until somebody inserts a row in the wrong
place and a generic "ethernet controller, unrecognised" silently shadows the
driver that would have worked, so a row may be added anywhere.

**Three levels of support, because there really are three.** `Driver` works.
`Known` means recognised with nothing behind it, and carries the *reason* --
"needs a signed blob" and "nobody has written it yet" are completely different
futures, and the reason is the most valuable field in the file. `Partial` is
the middle this tree keeps landing in: the RTL8188EU dongle is identified, its
registers are readable, its firmware parses, and it cannot carry a frame.
Filing that under `Driver` is a lie an operator only discovers when the network
does not work; filing it under `Known` throws the work away.

    glados> devices
      8 device(s)
      USB usb5.1    0525:a4a2 driven   usb-ecm    CDC ethernet adapter
      PCI 00:00.0   8086:29c0 known    -          host bridge
      PCI 00:01.0   1234:1111 partial  gfx        VGA-compatible display controller
      PCI 00:02.0   1b36:0010 driven   nvme       NVMe controller
      PCI 00:03.0   1b36:000d driven   xhci       xHCI USB 3 controller
      PCI 00:1f.2   8086:2922 known    -          SATA controller in AHCI mode
      what is missing:
        SATA controller in AHCI mode  no AHCI driver yet, and this is why some machines have no disk

That last line is the point of the whole file. It is the largest hole in the
table, it is entirely tractable -- AHCI is published and needs no firmware --
and until the registry existed nothing said it: a laptop with a SATA SSD and no
NVMe reaches a shell with no store, no model and nowhere to save, and the boot
log gave no hint why.

**USB is pushed, never swept.** Enumerating the bus resets the controller and
drops whatever link is on it, so a registry that went and looked would take the
network down to find out what the network is. `xhci::note_device` is called by
whoever was already enumerating, with zeros for the class triple -- which is
exactly what a device deferring to its interfaces reports, and which
deliberately matches no class rule -- and again with the real triple once a
configuration has been parsed.

**`net::init` no longer carries a preference list.** It was a nested match:
e1000, else rtl8168, else USB. That is a preference written for one laptop; on
a machine carrying only the Realtek it paid for a full Intel sweep to learn
nothing, and on a machine carrying neither it reported "no supported NIC"
without ever saying what *was* there. `drivers_for(Role::Ethernet)` answers in
bus order, every refusal is listed by name, and the ethernet and wireless gaps
print underneath.

`wifi.rs` kept a second copy of both id lists under a comment arguing the
duplication was deliberate. The objection it made -- that a probe saying
"supported" while the driver fails for another reason is worse than a short
list -- is answered by `Partial`, and by the registry never claiming a device
works, only that a row describes it. Both copies are gone and `hardware()`
reads the registry.

`diag devices` is 23 claims over 36 rows, nine of which are about which
*hypervisor* this is. Two of them are about the table
rather than about any device, and they are the ones worth knowing: **every row
is reachable** by some device, because a rule that cannot win against the rest
of the table is documentation pretending to be code, and **no two rows of equal
specificity overlap**. Everything is asserted against synthetic idents rather
than against the real bus, for the reason `mem::fixed` gives about its own map:
a claim about the machine underneath would pass here and fail on the next
laptop, which is precisely the failure this module exists to stop.

**What it deliberately does not do is bind.** `lookup` answers which row
describes a device; whether that driver then initialises is the driver's own
business and it can still fail for a dozen reasons a table cannot predict. The
registry says "this is an e1000 and `e1000` claims it", never "the network
works".

### Networking (`src/net/`)

Interfaces live in `iface`: `lo`, `eth0`, `wlan0`. A driver implements
`iface::Nic`; `net::init` asks `dev::registry` which drivers the fitted
hardware wants and tries those, in bus order, rather than the hardcoded
e1000-then-rtl8168 chain it used to carry. Routing picks an interface by
destination, and every layer above asks for a source address instead of
assuming one exists.

**`poll` never dispatches into a transport state machine. It queues.** Sending
calls `send_ipv4` then `resolve`, and `resolve` calls `poll` while waiting for
ARP. Running a state machine from there would let a connection re-enter its own
control block while an earlier borrow is live. TCP and UDP drain their own
inboxes.

TCP advances only while the shell is idle (`tcp::service` from the idle loop)
or inside a blocking call. There is no interrupt-driven receive.

### What is actually in this laptop, read off it rather than remembered

Enumerated from Windows on the GF63 itself (`Get-PnpDevice -Class Net`), not
recalled:

| | |
|---|---|
| **Intel Wi-Fi 6 AX201 160MHz** | `8086:51f0`, **00:14.3** -- CNVi, so the MAC is a PCH function |
| Realtek RTL8168 GbE | `10ec:8168`, driven |
| Intel Wireless Bluetooth | USB `8087:0026`, no HCI here |
| *(no USB wireless dongle is plugged in)* | |

`0x51f0` is in `INTEL_CNVI_IDS`, so the registry names the part correctly.

**CNVi is not the obstacle, and saying it was sent at least one session down
the wrong road.** The registry used to read "the radio is in the PCH over an
undocumented interface". CNVio -- the link between the chipset and the radio
module -- is undocumented, and *the host never speaks it*. From here the part
is an ordinary PCIe function with BARs and MSI-X, driven the way `iwlwifi`
drives a discrete card. The cost is `iwlwifi`'s cost: a firmware image in
Intel's TLV container, the context-info structure that bootstraps it, then a
host command protocol -- PHY and MAC contexts, bindings, stations, time events
-- before one frame moves. Thousands of lines, every step
match-it-exactly-or-silence, and no emulator models any of it.

And "a signed firmware blob that is not redistributable" was simply **wrong**.
`LICENCE.iwlwifi_firmware` permits redistribution and use in binary form
without modification, which is why Debian ships it in `non-free-firmware` --
the label for redistributable-and-not-free. Not modifiable and not open source
are different objections from the one that had been recorded.

**The cheap paths, in order.** A phone in USB tethering mode presents CDC-ECM
or RNDIS, both `Support::Driver` today, and `dev::registry` already calls that
"the closest thing to a universal wireless driver there is" -- it works now and
needs no new code. After that, an RTL8188EU dongle: `xhci` already identifies
one, reads its chip id and runs `bring_up`, and the efuse decoder, LLT chain,
firmware container parser, channel plan and both descriptor formats are written
and asserted at boot. Bulk endpoints are not missing either, since `xhci`
configures and drives them for CDC and RNDIS. What is left there is the on-wire
sequence and `impl Radio`. The AX201 is the largest of the three and the only
one that uses the laptop's own radio.

`net/wifi.rs` identifies hardware and refuses to pretend; `hardware()` projects
`dev::registry` down to the network parts, so the naming lives in one table
rather than two, and boot prints it.

### Wireless: a seam, a shared layer, and no drivers

**`net::iface::Nic` is Ethernet-shaped, and that was the finding that decided
the architecture.** `transmit(&[u8])` takes an Ethernet frame, which suits a
FullMAC part whose firmware hides 802.11 -- and `dev::registry` names six
wireless families, nearly every one of them SoftMAC. On those the radio moves
*802.11* frames and the **host** does association, sequencing and crypto. So
the generalisation is not a driver; it is a seam one layer down, plus
everything above it written once:

```
      net::iface::Nic        ethernet frames, TCP/IP above
            ^
   +--------+--------+
 net::mlme          a FullMAC part plugs in here, because
 net::softmac       its firmware already did all of this
      |
 dev::radio::Radio  <- the seam: 802.11 frames in and out
      ^
 rtl8188eu, ath9k, mt76, ...
```

| | |
|---|---|
| `dev/radio.rs` | the seam -- name, caps, mac, start/stop, channel, tx, rx -- and the channel plan, which is the same for every part in the world and used to live in one driver |
| `net/softmac.rs` | Ethernet over 802.11: SNAP, the four address layouts, sequence numbers, CCMP when the chip does not |
| `net/mlme.rs` | the station state machine: scan, authenticate, associate, four-way, install keys |
| `net/ccmp.rs` | the link cipher: AAD masks, packet numbers, replay |
| `crypto/ccm.rs` | AES-CCM, against RFC 3610 packet vector 1 |
| `net/wpa2.rs` | the handshake, **both** halves |

A driver's whole obligation is `impl Radio` and one call to
`net::attach_radio`, which makes it `wlan0`. It **refuses a FullMAC part**:
`Caps::softmac` is how a chip says its firmware ran the MLME, and such a part
implements `Nic` directly like the wired card. `Nic::wireless()` answers
`Option<&mut dyn Wlan>` with a default of `None`, so there is one driver and
one handle to it; the shell's `wifi scan | join | leave` goes through that.

**All of it is asserted at boot with nothing plugged in** -- `diag radio`,
`softmac`, `ccmp`, `ccm`, `mlme` -- against a `Loopback` radio and a fake
access point that authenticates, associates and runs the authenticator's half
of the four-way handshake. The whole path scan to encrypted data runs in a
loop with no delay, because `Station::poll(now_ms)` **takes the clock as an
argument**: every state is a deadline, so a machine reading its own clock could
only be tested by waiting.

Four things that cost a run each and are silent when wrong:

- **The association response and message 1 arrive in one batch**, and the
  supplicant that reads message 1 does not exist until the response has been
  read. Acting on frames in arrival order drops message 1 of every handshake --
  and what that looks like is a network that scans, authenticates, associates,
  returns an AID, and then times out with nothing visibly wrong. `poll` drains,
  handles management, *then* feeds EAPOL.
- **The key goes in after message 4**, never before. Installing earlier stops
  the link accepting the plaintext EAPOL the handshake is made of.
- **`Supplicant::on_frame` had never been executed** before `wpa2::Authenticator`
  existed. The PMK was checked against Annex H.4 and the PTK against its own
  symmetry; the state machine that installs a key had no frames to be fed.
- **`aes::key_wrap` was missing**, so `key_unwrap` had only itself and one
  vector. The counter is xored into A after encryption and before decryption,
  which is the one asymmetry in RFC 3394 and the only place the two directions
  can silently disagree.

Owed, and written at the top of `ccmp.rs` rather than only here: an IEEE
802.11-2016 Annex J CCMP vector. The cipher is checked against RFC 3610; the
*framing* is structural and round-trip only.

For the rtl8188eu dongle specifically, more exists than a summary here once
claimed: `xhci` identifies the part, reads its chip id and calls `bring_up`,
which applies all four initialisation tables including the radio over the
register interface, and `efuse_decode`, `efuse_mac`, `llt_chain`, `fw_parse`,
`fw_pages`, `channel_mhz` and both descriptor formats are written and asserted
at boot. **Bulk endpoints are not missing** -- `xhci::configure_bulk`,
`bulk_in` and `bulk_out` exist and carry CDC and RNDIS traffic today. What is
left is the on-wire sequence that uses those pieces (write the LLT, set the
FIFO boundary that gates the MAC TX/RX enables, read the efuse, upload the
firmware, select a channel) and then `impl Radio` over the descriptors. None of
the chip-facing half can be exercised here, since QEMU models no wireless part
at all.

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

**J3 compares what the candidate printed, and for a long time it could not.**
It weighed the value the program answered, its step count and the objects it
touched, and a replay is a sequence of `println(applet(...))` which answers
nil however the applets behaved -- so the one judge that matters for the one
shape `agent learn` actually produces agreed with itself by construction. The
reason recorded for leaving it was that closing it needed a capturing console,
"which does not exist". It did: `gfx::console::begin_capture` is a stack, it
was already carrying `applet`, the agent's observations and `differ`'s own
comparison, and the gap was a claim about the tree that had gone stale rather
than a missing mechanism.

The selftest claim for J3 was itself wrong twice for related reasons -- first
printing the clock instead of answering it, then answering a 100 Hz clock that
does not move between two adjacent runs. Both versions are claims now, and the
first one is the interesting one: `println(tsc())` answers nil twice, spends
the same steps twice and touches nothing twice, so removing the console
comparison is the only thing in the suite that makes it fail. Checked by
removing it.

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

**`Node::Away(ChunkRef)` is a blob that is on disk and not in memory**, and it
is the reason a corpus can be in the namespace at all. Its chunk reference is
everything the tree needs: `hash` is the blob's content address, because
`content_hash` of a blob is exactly `sha256(bytes)` and that is exactly what
`Store::put` computes, and `len` is its length. So an away node hashes,
compares, counts and serialises **identically** to the blob it stands for, and
every address from it to the root is bit-identical whether or not the bytes are
here. That is not a coincidence; it is the property the store was designed
around, and it is why the variant cost nothing anywhere else.

`read_node` therefore does not read a blob at all, and restore costs one chunk
read per *directory* instead of one per node plus a SHA-256 over every byte on
the disk. Measured on the 8,913-node forest:

    before   never reached a prompt, with a 900-second deadline
    after    39 s for the whole run -- QEMU start, boot, selftests, four commands
             8913 of 8913 file(s) on disk, 7,705,500 B that never came into memory
             heap 32 MB used, against ~100 MB with the same forest resident

Two things the compiler could not catch, both `_ => None` arms that compile
perfectly on a match over two variants of three: `read_blob` would have reported
every restored file as **missing**, and `blob_len` the same. Adding a variant
finds the exhaustive matches for you and not these.

`sysbox::fetch` is the read path, and unlike a bare ranged read it **verifies**:
the whole blob is read and `ChunkRef::hash` is its content address, so one
SHA-256 settles whether these are the bytes the tree names. Nothing caches the
result, deliberately -- a cache with no eviction is exactly how a corpus moved
out of memory ends up back in it, one read at a time, with nothing saying so.
`du` reports how much of a subtree is away, which is the figure that says
whether any of it worked.

**`src/sysbox/stored.rs` is the door underneath that.** It resolves a path by
walking directory chunks only and reads bytes out of a blob with
`cas::read_blocks`.

### Routing a question to a branch

`src/ai/route.rs`, and it is the Phase 3 **baseline** rather than the router --
the thing a fitted probe has to beat, in the shape the plan asked for.

A branch vector is the mean of its nodes' pooled head embeddings, and a query
is pooled the same way and compared by cosine. **No forward pass anywhere**:
`vocab::pool_text` averages embedding-table rows, so there is no attention, no
layer and no KV cache, which is what makes routing nine thousand nodes
affordable here at all. `forest embed` builds and scores the table, `forest
route <question>` asks it.

Two decisions exist so the number is not a lie:

- **The subject is never pooled.** A head is `subject | concept | terms` and
  the subject *is* the branch path, so a descriptor built from it would be
  reading the label off the back. Concept and terms only, on both sides.
- **Accuracy is leave-one-out, and it is free.** A branch vector is a mean, so
  removing one node is `(sum - v) / (n - 1)` exactly -- every node is a held-out
  test point against a branch that never saw it, with no split to arrange. The
  table therefore stores **sums and counts**, not means: a mean throws away
  precisely what scoring after the fact needs.

Measured on the 8,913-node forest with SmolLM2-135M, dim 576, deterministic
across runs:

    8913 node(s) over 16 subject(s) -- 8157 had a shard folded in
    plain      top-1 17.0%   top-3 47.2%
    centred    top-1 28.6%   top-3 44.2%
    6.2% is chance over 16 subject(s)
    built and scored in 9164 ms

**The first measurement found a design error, which is the reason to take one.**
Keyed by *branch* it read 7.6%, and two of every three classes were `part-00`
and `part-01` of one category -- `tools/forest.py` shards a directory to bound
fanout, so that asked the router to split one subject in half at an arbitrary
point. `route::subject_of` folds a `part-NN` component and `forest embed`
reports how many it folded, so a forest that plainly has shards and folds none
says so.

**Centring is measured, not assumed.** Mean-pooled English shares a large common
direction, which left every cosine bunched in 0.66-0.76; subtracting the
centroid spreads them from 0.69 down to 0.17. It is scored both ways over
identical vectors in one walk -- the only comparison that means anything -- and
it sharpens top-1 by 11.6 points while costing 3.0 on top-3. It is on because
top-1 won, and **that choice is budget-dependent**: a retrieval loading three
subjects would prefer the other one. Both figures print every run so it can be
revisited with evidence.

### Retrieval, and the measurement that condemned the first attempt

`src/ai/lex.rs`. **Mean-pooled embeddings scored 0.5% on real retrieval**, and
that number is the reason everything below exists.

The task is known-item retrieval and it is built so it cannot be gamed: a node's
`concept` is the *first sentence* of its `text`, so the index holds the concept
and its terms while the query is everything after that first sentence. Prose
from the same node that the index has never seen, one right answer in 8,913,
chance 0.011%. `forest bench [n]` runs it, and asks **two ways**: the long
query is the whole body tail, the short one is its first eight words, which is
what a person types. 198 queries, same candidates and same nodes throughout:

    method               L r@1   L r@5   L MRR     S r@1   S r@5   S MRR
    mean pool             0.5%    1.0%  0.0098      8.0%   16.1%  0.1097
    idf pool              4.5%   11.1%  0.0766     34.3%   43.4%  0.3893
    terms b=0.25 idf     86.3%   93.9%  0.8968     48.4%   66.1%  0.5660
    terms b=0.50 idf     89.3%   94.4%  0.9166     48.9%   65.6%  0.5722
    terms b=0.75 idf     87.3%   93.4%  0.9039     46.4%   64.1%  0.5529
    terms b=0.25 idf^2   90.9%   93.4%  0.9245     52.0%   71.7%  0.6016
    terms b=0.50 idf^2   91.9%   93.9%  0.9310     52.5%   72.2%  0.6080   <- ships
    terms b=0.75 idf^2   91.4%   93.9%  0.9275     50.5%   68.6%  0.5957
    bm25 k=1.2 b=0.50    80.3%   89.3%  0.8529     43.9%   61.1%  0.5157
    mix a=0.10           90.4%   93.9%  0.9242     52.5%   72.2%  0.6095
    mix a=0.30           71.2%   80.3%  0.7551     51.0%   71.2%  0.5944

**Two named gaps closed, one of them with a negative result.**

*Term-frequency saturation made it worse.* It was the obvious fix for a ranking
that looked like it was counting words, and BM25 measures 79.2% against 87.8%.
The reason is in the corpus: these are short questions with almost no term
repetition, so `tf` is nearly always 1 and the saturation collapses to a
constant -- what is left of BM25 is its *length* normalisation, which sits
**inside** the saturation where `k1` multiplies it, and that measured worse than
charging the sum directly. The constant, the `tf` column and the grid row all
stay so the day the corpus grows longer documents the answer is one command
away.

*Short queries are much harder and change nothing.* 47.9% against 87.8% is the
honest cost of a terse question, and `b = 0.50` wins both columns -- so the
worry that a constant tuned on long queries was tuned on the wrong distribution
was worth checking and came back clean. The interesting half of that table is
the embedding: 5.5% on long queries and **30.8%** on short ones, which closes
most of the gap and still never opens one.

**The families are three different normalisations, and conflating two of them
cost a measurement.** The first grid had only `bm25 k=0` where `terms b=X`
belonged -- and at `k1 = 0` BM25's length term drops out entirely, because it
only ever appears multiplied by `k1`. Three rows came back identical, the
discount looked like it did nothing, and the row that had actually won was
missing. `M::Terms`, `M::Bm25` and `M::Mix` are separate for that reason.

Three corrections, each visible in a row above:

- **Inverse document frequency.** `vocab::pool` averages every token's embedding
  equally, so in "what is the derivative of a polynomial" the four function
  words outvote the two that carry the question -- and they are the four that
  appear in every other document too. Weighting by `ln((N+1)/(df+1))` and
  scaling each row to unit length first took 0.5% to 5.5%.
- **An inverted index.** An embedding match is a similarity; a term match is a
  fact. Names and numbers either appear or they do not, and on a corpus of
  questions that is most of the signal. 5.5% to 81.8%, for 1.3 MB of postings.
- **A length discount.** Asked for the derivative of a polynomial, the first
  answer was about *roulette* -- a long node holding `what`, `is`, `of` and `a`.
  A long document collects more small weights than a short one carrying the
  words that mattered. Charging for that: 81.8% to 87.8%.

**`LEN_B` is 0.5 and not the textbook 0.75**, because the sweep has an interior
optimum there -- which is the difference between a value chosen and a value
defaulted to. These are short questions of similar shape and they wanted less of
a charge than web documents do.

**`MIX` is 0**, so the embedding channel is off for node retrieval. Every weight
above zero measured worse. That is not a claim that embeddings are useless: it
is one checkpoint's table, pooled with no forward pass, against exact term
matching -- a different model or a rank-based fusion may move it, and the sweep
is one command that prints every rung. The subject router still uses embeddings
and improved for free when the pooling did, from 28.6% to **34.7%** top-1 against
6.2% chance.

### `forest why`, and the two causes it found

The entry about *differentiating* a polynomial was not first, and two guesses
had already been wrong about it. `forest why <question>` prints what the scorer
actually saw: every query token with its document frequency and weight, then
each top result with its length, its length charge, and which tokens it matched
for how much. Both remaining causes fell out of one run.

**A tokenisation boundary.** A byte-level BPE spells `what` mid-sentence as
`' what'` and at the start of a string as `'what'` -- different ids. So the
first word of every query and every indexed document was a *different token*
from the same word anywhere else, and therefore rare: `'what'` measured a
document frequency of **2** and an IDF of **7.9968**, against `' derivative'`
at 8.4022. A stopword worth as much as the rarest content word in nine thousand
nodes, and a question about roulette outranking one about derivatives for
starting with it. `lex::prep` puts a space in front on **both sides**; `' what'`
is df 74 and IDF 4.78, and the sweep moved 87.8% to 89.3%.

**Linear IDF does not suppress a stopword *set*.** With that fixed, a node
matching `what is the of a` and no content word at all still came first: its
matched mass was 9.56 against 12.20 for the node holding the only `derivative`
in the corpus, close enough for the length charge to overturn -- 0.876 against
1.177. Squared, the same five sum to 28.6 against 75.4. That is the weighting a
tf-idf cosine uses, it says rarity counts more than linearly, and it moved
89.3% to **91.9%** long and 48.9% to **52.5%** short with r@5 going 65.6% to
72.2%.

The derivative entry now ranks **first**, paying the highest length charge in
the list, and no stopword-only match survives in the top five.

**That flag is a whole-number exponent now, `IDF_POW`, and the host rail does
not re-confirm the two.** Widening it to an `f32` would put `expf(p * lnf(v))`
on the shipped path where the host mirror uses libm, so the two would disagree
about every weight -- the divergence `forest_retrieve.py` was just corrected
for. Whole powers are exact on both sides, so the rung is 1, 2, 3.

Two is byte-identical to the `true` it replaced, checked on 250 queries with
the dumps compared as files. What is new is the sweep through the rail that
actually judges a source proposal about this constant, on the 7,560-node
forest at 250 queries:

    short   pow 1  60.8%  fixed 5 broke 3     pow 3  59.6%  fixed 4 broke 5
    long    pow 1  96.4%  fixed 4 broke 0, chi 2.25   (shipped 94.8%)

All three are `same` under the paired test, so the loop refuses all three and
the shipped value stands. But the long row leans the *other way* from the
sweep above it, cleanly, and falls short only on the bar. They are different
scorers -- BPE ids against words, a different forest, a different query set --
so the honest reading is that two is **not re-confirmed by the host rail**
rather than that it is wrong. A verdict about this row carries that caveat
where one about `LEN_B` does not: a length charge means the same thing under
either tokenisation and a rarity exponent does not, and `knob.rs` says so
beside the row.

### Budgeted retrieval

`src/ai/recall.rs`. Phase 4: pick nodes for a question and render as many as a
token budget admits. `forest recall [budget] [subjects] <question>`.

**The budget is counted, never estimated.** `fill` takes the counter as a
closure and re-encodes the accumulated block after every candidate, because
tokenisation is not additive at a boundary -- summing per-node counts drifts,
always in the direction of admitting one node too many. Measured: budget 1500,
used 1496, nine entries kept, fifteen skipped, identical across runs.

Two decisions the suite pins down. A candidate that does not fit is **skipped
rather than ending the fill** -- the list is sorted by score and not by size, so
stopping at the first overflow throws away every smaller entry behind one large
one and leaves the budget unspent with nothing saying why. And the preamble is
only paid for once something fits beneath it: a heading promising entries with
nothing under it is worse than silence.

**Scoring everything is the default, because the measurement said so.**
`recall::Nodes` is one pooled vector per node, 21 MB at dim 576 over 8,913
nodes, written by `forest embed` and cached after one load. With it resident,
nine thousand cosines is five million multiply-adds and no disk at all -- so
routing first costs recall and buys nothing at this size. Measured on the same
question at budget 1500: routing to three subjects of sixteen found **four of
the nine** entries a full scan chose, for the same 1496 tokens. Routing is
opt-in, and prints that price every time. It earns its keep when the vectors
stop fitting, and not before.

Both sides are centred or neither is: the query is centred against the subject
table, so `Nodes::centre_with` applies the same centroid at load. Comparing a
centred query to raw node vectors answers a perfectly plausible cosine to a
different question.

**And the silent unpinning is closed.** `sink_count` clamps the pinned span to a
third of the trained length, and when that bit, the *tail* of the system turn
stopped being pinned and scrolled out with no message -- `/ai/about` is appended
to and is therefore always at the end, so what was lost was precisely what the
operator had most recently asked to be remembered. `companion::turn` now says so
at the one moment both numbers are known:

    (the system turn is 199 tokens and only 170 can be pinned --
     the last 29 will scroll; '/ai/about' is what grows it)

`widen_if_near_the_wall` only warned when the pinned span fell to `MIN_SINKS`,
which is a different and much later failure, so a system turn one token over the
ceiling looked exactly like one that fitted.

`find`, `locate` and `locate_under` do the walking; `read_at`, `read_all` and
`head_line` are the byte-granular side. `read_blocks` had been finished and
unreachable for as long as it had existed. `forest::index_at` is the other
consumer: heads and chunk references in memory, bodies on disk, 2.6 MB of index
against 7.7 MB of bodies on that same forest, built in 30 s with nothing
resident.

Two trades are opposite on purpose. A directory goes through `Store::get`:
small, verified against its own address, few of them. A blob goes through
`read_blocks`: ranged, allocation-free and **unverified**, because the hash
covers the whole blob and a range is not the whole blob. And there is **one
static 64 KiB scratch buffer**, which is the fix `cas::dma`'s own note asks for
-- that one allocates per call and never frees, measured at 4,096 bytes plus the
rounded length per blob.

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

**And a fourth, of a kind the other three do not cover: a harness that was not
asking the question at all.** GSM8K read 0.0% on every checkpoint this project
has ever run, and that number was quoted as evidence about small models and
arithmetic. It was four defects in `tools/lm_eval.py`:

- `--hf-tokenizer` defaulted to SmolLM2's **49,152**-token vocabulary, which is
  right for SmolLM2 and wrong for every Qwen checkpoint here (151,669 and
  248,320). Handed the wrong one, a model receives ids belonging to another
  vocabulary and answers with a degenerate run of one token. Observed, at 0.0%.
- `--max-new` was **64** against answers that are **133 tokens on average**, so
  two thirds of every completion was cut off before the line the score is read
  from.
- Nothing told it to stop, so the extractor read its number out of a fabricated
  next question.
- The dense runner was llama2-shaped -- derived head width, interleaved RoPE, no
  QK-Norm -- and raised `operands could not be broadcast` on Qwen3, so it never
  produced a figure for one at all. It is a thin adapter over `reference.py`
  now; two dense implementations do not stay agreeing.

**The lesson is the instrument, not the four bugs.** Every one of them is
obvious the moment you look at what the model actually returned, and nothing
ever printed it -- the harness recorded a score and threw the text away. `--show
N` prints the raw completion and stays for that reason. A rail that reads zero
is not a result until its output has been read; a score with no transcript
behind it is an assertion.

**And a fifth, in the rail beside it, found while re-running what the first
four owed.** `run_mmlu` did `parquet_rows(find_file(d, "test"))[:100]`, and
`find_file` answers `sorted(rglob(...))[0]`. The MMLU snapshot has one
directory per subject and no combined config, so that is 100 questions of
`abstract_algebra` and nothing else -- **every MMLU figure this project has
recorded, for the life of the rail**, including the 43.3% quoted in
`design/benchmarks.md` as a number about the 2B. Not a wrong answer: a rail
quietly answering a different question from the one its name claims, which is
the same family as the other four and the fourth time it has been the
*composition* of a set that nobody printed. It reads 14,042 over 57 subjects
now, prints what it is scoring every run, and draws a seeded sample for
`--limit` because the rows arrive grouped by subject. The resident 0.6B is
**41.0%** over the whole thing against 25% chance.

**The rail that stops this costing a week each time.** Every one of those five
took thousands of items to surface because a binary rail throws away almost
everything the model did. `--task bpb` scores held-out bits per byte over a
corpus pinned by content hash, defaulting to this kernel's own source: one
forward pass per window, no generation, every token an observation. Measured
against GSM8K on a change both can see -- the kernel's int8 KV cache -- it
reaches t = 6.61 on 256 windows where 1,319 GSM8K questions reach chi 3.86
against a 3.84 bar, and two dozen windows in twelve seconds still carry more
evidence than the full test set does in seventy minutes. Reach for it first;
the binary rails are for the figure you quote, not the one you iterate on.

**And the first figure about GLaDOS rather than about the checkpoint: 37.1%.**
The kernel holds its KV cache `Vec<i8>` and every host number here was f32.
`--kv8` round-trips it at the point `model::State` stores, and the same whole
test set paired reads 39.2% against 37.1%, 74 fixed and 101 broke, chi 3.86.
Two points, clearing the bar by 0.02, so probably real and not settled -- and
175 of 1,319 answers moved for a net of 27, so int8 is a large perturbation
that mostly cancels rather than a small one that rarely matters.

**And the number it was hiding: 39.2%.** Qwen3-0.6B, GSM8K 5-shot greedy, on
the **whole 1,319-question test set**, through the fixed harness. The row read
0.0% on every checkpoint this project has ever run and was quoted as evidence
about small models and arithmetic; the model had been doing the arithmetic the
whole time and nothing was reading the answer.

The last of the four defects needed a runner rather than a fix.
`tools/fastdense.py` is `reference.py`'s arithmetic with the position loop
turned into a matrix dimension: prefill **92x**, weights dequantised once
instead of per call per layer per token, and the same five-shot prefix computed
once rather than 1,319 times. 8.7 s/question against about 180, so the whole
set is 3.2 hours where the oracle could not finish it.

It is allowed to exist only because `fastdense.py --check` runs it against the
oracle on a deliberately **ragged** batch and prints the largest disagreement
per row; `lm_eval.py --oracle` takes the slow path, which is what a
disagreement is diagnosed with. Two dense implementations do not stay agreeing
unless something makes them.

**Three figures from partial runs are withdrawn and worth knowing about.**
28.0%, 36.0% and 37.5% were all measured at n around 25 on this same
configuration, and the spread is the plus-or-minus-18-point interval doing
exactly what it says. None of them should have been quoted, and one of them
was, here. The full set has no interval worth stating.

And a fourth: cross-question batching was reported as buying nothing, from two
runs that were **both batch 1** because `lm_eval` never passed `batch` through
to the task. At full scale it is worth about 25%, which is far less than its
throughput figure suggests and not nothing. The tell was printed every time --
`1319 group(s) of one length` cannot happen at batch 8, and reads `222` once
the flag arrives. Before comparing two configurations, print something that
must differ between them.

`design/benchmarks.md` carries the figure and the transcript it came from.

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
- **Processor state that is per-core has to be adopted on every core, and the
  failure looks like memory corruption somewhere else.** `diag paging` then
  `diag smp` faulted with "reserved bit set in a page table entry", two
  confident hypotheses were measured and both were wrong, and the third
  attempt read the entry:

      error 0x0000000000000009  reserved bit set in a page table entry
        pt  0x8000000002c12063

  Bit 63, which is `NX` and is entirely legal -- boot prints `nx=1`. But
  **`EFER.NXE` is per-core**: the trampoline sets `EFER.LME` to reach long
  mode and stops, so nothing ever gave an application processor the rest. One
  entry therefore meant no-execute on the bootstrap processor and *reserved*
  on every other core, and the fault arrived on whichever core read it second.
  `cpu::adopt_page_rights` does `CR0.WP` and `EFER.NXE` on every core and says
  so when it cannot.

  **Fixing it exposed the other half**: `paging::checks` borrowed a heap page,
  made it read-only to watch the processor refuse a write, and restored it
  with `Perm::RW`, which is precisely non-executable. Every run of the suite
  left an `NX` page in the heap, and `diag code` and the Aiksi JIT both
  allocate one and jump into it. `RWX` now, and the claim checks `exec` too --
  "restored" was true of a page that had quietly lost a right nobody was
  asking about yet.

  Two lessons rather than one. It hid this long because the identity map is
  built from 2 MiB pages, which never carried bit 63: `diag paging` is the
  only thing in the tree that writes `NX` into a 4 KiB entry, and it frees
  those pages straight back to the heap. And **`pagemap` found nothing,
  correctly** -- it runs on the bootstrap processor, where the entry is legal.
  The question was never what the entry said, it was which core was reading
  it, which is a thing no single-core diagnostic can be asked.

  Fixed in `d015e0d`. `diag all` twice in one boot passes.
- **A longjmp into inlined code has no calling convention to lean on.**
  `recover::guard` saved the registers a callee must preserve, which is the
  right list for a function boundary and the wrong question entirely: `guard`
  is inlined into its callers, so there is no boundary, and the compiler is
  free to keep a caller's live value in `rax` or `r9` across a call that no
  longer exists. It saves **every** general-purpose register now, and the
  landing code restores every one -- `rcx` last, through itself, with the jump
  target pushed onto the already-restored stack so `ret` can take it.
  This was found twice, both times as a wild pointer in a subsystem with
  nothing to do with faults: a PML4 walk with an index out of `rsi`, then the
  heap's free list walked from a cursor out of `r9`. The second appeared
  because an unrelated file changed what the register allocator did, which is
  the tell that a list of registers was never going to be the answer.
  `mem::paging::checks` is the only thing in the tree that faults on purpose
  with real work live around it, so `diag paging` is where this shows up and
  `diag recover` passes throughout. A probe written as a function cannot catch
  it: the probe's own prologue saves those registers and its epilogue puts them
  back whether or not the pad restored anything.
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

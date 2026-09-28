# GLaDOS 1.3.9

A hundred and forty-seven commits, and the first release with something to
download that is not an operating system: the mining pool and its miner ship as
binaries, and a 32 MB kernel-only image ships beside the model ones.

The theme is money, and specifically that nearly every figure this project had
about its own earnings was wrong in a way that only arithmetic could catch. The
algorithm it had been optimising for eight commits pays **three dollars a day
across every miner on Earth**. Finding that out was worth more than the 1.51x
the optimisation bought.

## The pool and the miner are releasable

`glados-pool` and `glados-miner` are attached to this release as static-pie musl
binaries with no `PT_INTERP`, so they run on any x86_64 Linux rather than on one
with a recent enough glibc. Nothing published them before: `pool.yml` has built
and tested both on every relevant push for months and there was never an artefact
at the end of it.

- **The pool passes its own 37 claims from the published artefact**, not from a
  debug build of it, and the last of those is a real socket exchange over
  loopback -- so the binary that ships is the one known to listen.
- **A `nomodel` image, which the download page has been describing all along.**
  `tools/site.py`'s `VARIANTS` has carried a "Kernel only" row at the top of the
  table, with a RAM figure, for a long time; the release matrix never built it.
  Nothing in `src/mine/` needs a checkpoint, so 32 MB boots to a machine that
  can mine where the smallest model image is 600 MB.
- The miner's own cargo config still named `x86_64-pc-windows-msvc`, and its
  comment had predicted the consequence: "this crate has never been built for a
  non-Windows target, so here it was only ever latent". It stopped being latent
  the day somebody built a miner to hand out.

## The algorithm pays three dollars a day, and that is the whole finding

`design/mining.md` had the kernel's CPU out-earning an RTX 3050 by 1,800x on
yespower, which was true and irrelevant. Read off zpool's own `actual_last24h`:
yespower's entire network pays about **$3/day** to everybody on it. A miner
taking *all* of it earns three dollars, and no hashrate, hardware or free
electricity moves that number.

**yescrypt's network pays about $21,600/day**, and the pool is configured for it
now. The two numbers that matter and are not the same:

- Per unit of work, yescrypt's rate is **1,602x** yespower's (0.16021 against
  0.00010, read 2026-09-28).
- Per *this machine*, earnings go from **$0.0114 to $0.0404 a day, which is
  3.5x** -- because yescrypt's network is 1,745,879 H/s against yespower's
  49,988, so the same hashrate is a much smaller slice of a much larger pot.

Quote the second one. The first was reported here as the improvement and it is
the answer to a different question.

**It is a different string and not a different machine.** Same family, same
2 MiB working set at n2048 r8; `--bench` reads 3,557 us a share against
yespower's 3,520, so the bits, the CPU budget and the share target all carry
over. `yescrypt` is one of four aliases `parse_algo` byte-verifies against
upstream's TESTS-OK, and it is deliberately not `yescryptr8` -- BitZeny's -- where
a "Client Key" personalisation makes a different function at identical N and r.

**And it cleared the gate that priced the whole design dead.** The bridge floor
said that at yespower's rate the $30 minimum batch worth bridging to chain 4663
would take *years*, "which is another way of saying the algorithm has to change
before the bridge is even a question". It changed.

## Mining, optimised and then measured against the right thing

- **pwxform's lanes moved to SSE2** and the gap to `cpuminer-opt` closed to
  **15%**, from 1.51x before.
- **Sixteen slices, because seven execution units was the binding constraint**,
  not the core count. And an *idle efficiency core beats a busy performance
  core*, measured -- so placement puts slices on P-cores first and then takes
  whatever is idle.
- **Vectorising the bulk XOR bought nothing**, and the reason is the point: at
  these settings one hash does 2,097,152 word accesses and the XOR was never
  where the time went.
- **A sweep with no control measured mostly its baseline's noise**, which is the
  same correction `video bench` already carries.
- **Core 0 mines only when nothing is drawing on it.** Both branches are
  asserted in one boot, because the selftest runs before the compositor exists.
- **The payout address is checked before a night is spent hashing to it.**
  `src/mine/addr.rs` judges base58check, bech32/bech32m and EIP-55 and answers
  four verdicts rather than a bool: a form whose checksum holds, one that
  carries no checksum, one whose checksum fails, and a worker name. A new
  Keccak-256 sits under the EIP-55 half, pinned partly by *not* matching
  SHA3-256's empty-string digest.

Two figures in `design/mining.md` were corrected in place rather than in an
appendix, because an appendix is how a wrong number stays quotable. One of them
was a single-core memory-hard figure multiplied by sixteen.

## The Intel radio in this laptop is not the one the driver was written for

`src/dev/iwx/` is new, and the first thing it established is that the part
reported as an "AX201" is **AX210-family Snow Owl silicon** -- confirmed against
the machine's own kernel log, `rev=0x370`, `rfid=0x10a100`, `so-a0-hr-b0-89.ucode`.
That is a different boot path entirely, gen3 rather than 22000's.

What works, with no radio present and therefore checkable: Intel's own 1.4 MB
firmware container parses in-kernel -- 212 records, 50 loadable sections, a
13,944-byte image loader -- the gen3 context-info descriptor is built from it and
read back, the reset-and-power-up sequence is a table so its ordering can be
asserted, and the ALIVE notification, NVM read, capability bitmaps and power
table are all written. 476 claims in `diag iwx`.

Three things the reference got wrong or misleading are recorded beside the
numbers. The one worth repeating: `iwx_set_ltr`'s scale mask and its shift
disagree, so upstream writes a scale of zero -- 250 ns where its own comment says
250 microseconds.

**One attribution obligation, and it is not code.** The register offsets and
field encodings were read from OpenBSD's `iwx(4)`, which took them from Intel's
dual BSD/GPLv2 headers. The BSD arm is what is used, `NOTICE.md` reproduces the
notice in full, and the two files carrying the numbers say so.

## The contract was read by somebody who did not write it, and so was the harness

`design/audit.md`'s last open items are closed. Its first stated dependency was
the V2 pair's 0.3% fee, which the contract writes out as `997/1000` because a
pair does not publish it -- now **bracketed from both sides against the deployed
pair on chain 4663**, because one side is not enough: the distributor asks for
exactly what 997/1000 predicts, so a successful claim rules out a *higher* fee
and says nothing about a lower one silently shortchanging every claimant.

**Finding 9 is in the harness rather than the contract, and it matters more.**
`fork.mjs`'s `call()` reports whether a call reverted and does not undo it, so a
probe the pair's `k` check must refuse left its reentrancy flag cleared and the
next three answered `LOCKED` where the first answered `K`. The first version of
the fee sweep read all four as the pair refusing and measured a floor from three
answers that were the harness's own -- and it passed until the reasons were
printed.

`.github/workflows/contracts.yml` is new, and closes the audit's own complaint
that "the 93 claims run only when somebody remembers". There are 106; this file
said 60.

## Smaller things worth knowing

- **Two NUL bytes made two files invisible to `grep`** -- one in a Rust literal,
  one in prose about that literal. Four searches for a function plainly present
  returned nothing, and 60 KB of notes could not be searched at all.
- **The supabase half is deployed**, and its nonce replies now carry `issued_at`.
  They did not, while `Issued At` is a line *inside* the message being signed, so
  no caller could rebuild what it was being asked to sign. Found by
  `tools/workercheck.py`, which is new and drives the worker-registration path
  end to end with a signer written independently of the verifier.
- **`tools/origin.py` said a paused project was gone.** NXDOMAIN cannot tell a
  suspended free-tier project from a deleted one and the repairs differ, so it
  reports the measurement and names both causes.
- **A published sample ledger had a window 4.3x smaller than the work inside
  it** -- reachable, misconfigured, and exactly what the pool warns about at
  startup, which a published ledger carries no record of. `ledgercheck.py` says
  so now, as a note rather than a claim.
- **`equihash`'s verifier landed before the solver it exists to judge**, and
  `NSLOTS` is chosen from measured bucket occupancy rather than guessed.

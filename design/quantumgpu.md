# QuantumGPU Stage 1: SHA-256d on the RTX 3050, measured

Status: first measurements. Host-side, CUDA 13.3 on the GF63's own RTX 3050
Laptop (GA107, sm_86, 16 SMs, 45 W cap). Correctness established against
`hashlib` before anything was timed; every figure below is from
`cuda/sha256d.cu` and is reproducible from it.

The plan this answers predicted 0.5-0.9 GH/s and expected ILP widening to be
worth 1.2-1.8x. One of those was right.

## Numbers

| | |
|---|---|
| **Sustained, 4.9 s, host CPU idle** | **0.645 GH/s** |
| Same run with the CPU loaded | 0.493 GH/s |
| Burst, 1.4 s | 0.674 GH/s |
| Short, 70 ms | 0.448 GH/s |
| Clock under sustained load | ~1695 MHz |
| Power | 44.8 W, against a 45 W cap |
| Temperature | 78-79 C |
| Hashes per GPU cycle | 0.381 |
| Hashes per SM per cycle | 0.0238 |
| Energy | ~14.4 MH/J |

**The sustained figure is the real one.** Mining runs forever, so a number
taken before the part reaches its power limit is not a mining number.

**And the host CPU has to be idle, which cost a wrong figure before it was
noticed.** Every measurement in the first version of this file was taken while
QEMU was driving a gödel grid walk on the CPU, and the same run repeated
against an idle host gives 0.645 rather than 0.493 -- **31% of the answer was
the other workload.** The clocks and power were nearly identical between the
two (1670/44.5 against 1695/44.8), so this is not the dynamic power budget
being shared; it is the host thread that issues the launches being starved.
State the host's load beside any figure here or the figure is not
reproducible.

## What the durations mean, because they nearly produced a wrong answer

The first sweep ran 70 ms kernels and measured 0.448 GH/s. The same kernel over
6.4 s measures 0.493, and over 1.4 s measures 0.674. Three different answers
from one binary, spanning 50%.

At 70 ms the GPU never leaves its idle clock: sampled mid-run it read
**1072 MHz and 14 W**, on a part rated to 2100 MHz and 45 W. At 1.4 s it has
boosted to ~1725 MHz and has not yet hit the power limit -- that is the
0.674 peak, and it is not sustainable. By 6.4 s it is pinned at 44.5 W and
82 C and has settled to 0.493.

So the burst figure is the *thermal headroom* and the sustained figure is the
*machine*. A benchmark of this workload that does not say how long it ran has
not said anything.

## ILP width: no effect, and the reason is the interesting part

The plan's first-ranked experiment. Each thread carries N independent nonces so
the warp always has an eligible instruction, on the theory that the message
schedule's dependency chain leaves the scheduler starved.

Measured, all sustained, all ~0.94 G hashes, warm-up run discarded:

| nonces/thread | registers | spills | GH/s |
|---|---|---|---|
| 1 | 40 | 0 | 0.672 |
| 2 | 40 | 0 | 0.673 |
| 4 | 64 | 0 | 0.674 |
| 8 | 96 | 0 | 0.628 |

Flat within noise to 4, then a 7% regression at 8.

**The premise was wrong and `ptxas -v` says why.** The plan's ILP hypothesis
came from a published profile showing 128 registers per thread, 33% occupancy
and 47.6% "no eligible" warp stalls. That profile is of a *chained SHA-256 plus
RIPEMD-160* kernel. This one uses **40 registers**, which on sm_86 puts it at
the 1536-thread cap -- about 100% occupancy -- with zero spills.

There were no starved schedulers to feed. ILP substitutes for occupancy only
when occupancy is the constraint, and here it never was. At 8 nonces the
register count reaches 96, occupancy falls to ~44%, and the trade goes
negative.

The rolling 16-word message schedule is what kept it there. The textbook
64-word array would have cost the registers the experiment was designed to buy
back, so the fix was in before the problem was measured -- by accident rather
than by foresight.

## Against the plan's estimates

| | predicted | measured |
|---|---|---|
| Throughput | 0.5-0.9 GH/s | 0.493 sustained, 0.674 burst |
| ILP widening | 1.2-1.8x | 1.00x |
| Energy vs an S21 ASIC | ~4,300x worse | ~5,100x worse |

The throughput band held. The ILP prediction was wrong by its whole margin,
and it was wrong because it reasoned from somebody else's register count
instead of compiling this kernel and reading its own.

## What is not measured yet

- **Instructions per hash.** The derived figure depends on an assumed ALU
  utilisation, so it is not quoted. Nsight Compute gives it directly and that
  is the next run.
- **Undervolting.** The plan ranks it second and the telemetry supports the
  premise: this part is **power-limited, not clock-limited**, sitting at 44.8 W
  of a 45 W budget with 100% utilisation. Energy per operation converts into
  clock here, so the guardband lever is real.

  **Blocked on privilege, not on method.** `nvidia-smi -lgc` and `-pl` both
  answer *"the current user does not have permission to change clocks"*, and
  the experiment needs an elevated shell. What makes it worth running when
  somebody has one is that this kernel can grade itself: `sha256d.exe` with no
  arguments prints block 125552's digest, so the error rate at each setting is
  measurable rather than assumed, which is the whole premise of operating
  outside the guardband.
- **The uniform datapath.** Untouched.
- **A persistent megakernel is worth about 6%, not the large win the plan
  implied.** Measured directly by holding total work fixed and cutting the
  launch count 30-fold: 3000 launches gives 0.599 GH/s, 300 gives 0.618, 100
  gives 0.634. Launch overhead is real and it is small. That bounds what
  eliminating it entirely can return, and it is worth knowing before building
  the thing whose whole justification was that GLaDOS has no TDR watchdog.
- **Anything on the ring-0 side.** This is all host CUDA. The GLaDOS probe is
  committed but has still never seen the GPU, because that needs a reboot.

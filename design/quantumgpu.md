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
6.4 s under a loaded host measured 0.493, and over 1.4 s measures 0.674. Three different answers
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
| Throughput | 0.5-0.9 GH/s | 0.645 sustained, idle host |
| ILP widening | 1.2-1.8x | 1.00x |
| Energy vs an S21 ASIC | ~4,300x worse | ~4,000x worse |

The throughput band held. The ILP prediction was wrong by its whole margin,
and it was wrong because it reasoned from somebody else's register count
instead of compiling this kernel and reading its own.

## The efficiency curve

Measured with `nvidia-smi -lgc` from an elevated shell, sustained runs, host
CPU idle, correctness checked at every setting.

| clock | GH/s | watts | temp | MH/J | digest |
|---|---|---|---|---|---|
| 900 | 0.366 | 15.4 | 70 | **23.8** | ok |
| 1200 | 0.495 | 21.0 | 73 | 23.6 | ok |
| 1400 | 0.586 | 27.9 | 77 | 21.0 | ok |
| 1600 | 0.671 | 39.8 | 80 | 16.9 | ok |
| 1800 | **0.696** | 44.0 | 82 | 15.8 | ok |
| 2000 | 0.691 | 44.2 | 83 | 15.6 | ok |
| 2100 | 0.692 | 45.0 | 84 | 15.4 | ok |

**Throughput stops at 1800 MHz and the power cap is why.** Locking higher buys
nothing -- 1800, 2000 and 2100 all land within noise of each other at 44-45 W,
because the part is already spending its whole budget. The clock lock is a
ceiling, not a floor, and above 1800 the ceiling is not what binds.

**Peak efficiency is at 900 MHz and it is 51% better than peak throughput.**
23.8 MH/J against 15.8. Half the hash rate for a third of the power. For
anything that runs continuously that is the operating point, because joules are
the budget and peak rate is not the objective.

Even there the gap to purpose-built silicon is about 2,400x rather than the
4,000x at full clock, which is the most flattering way this hardware can be
described and still leaves it three orders of magnitude out.

**Correctness held at every clock**, 900 through 2100, checked against block
125552 each time. That is the expected result and worth stating precisely so it
is not mistaken for a guardband finding: `-lgc` pins the clock and the GPU
still applies the voltage its own curve specifies. Nothing was undervolted.
Testing the guardband means holding a clock at *less* voltage than the curve
asks for, which `nvidia-smi` cannot express -- Afterburner's curve editor is
the only route, and it is untried.

## Two algorithms

`cuda/miner.cu` runs either, and refuses to benchmark one whose digest does not
match `tools/algocheck.py`. Sustained, idle host:

| algorithm | best GH/s | npt 1 | npt 2 | npt 4 | registers |
|---|---|---|---|---|---|
| sha256d | 0.704 | 0.633 | 0.632 | -- | 40 |
| blake2s | **1.185** | 1.001 | 1.138 | 1.157 | 168 |

**BLAKE2s is about 1.7x SHA-256d**, which is less than instruction counting
predicts. Ten ARX rounds against sixty-four plus a second compression suggests
nearer 2.3x; the shortfall is the sigma permutation, which indexes the message
array per round and does not resolve to registers as cleanly as SHA-256's
rolling schedule.

**And ILP width matters for one of them and not the other.** BLAKE2s gains 16%
from one nonce per thread to four; SHA-256d is flat. That is the plan's first
experiment turning out to be right about the mechanism and wrong about the
kernel: SHA-256d sits at 40 registers and roughly full occupancy with nothing
to buy back, while BLAKE2s at 168 has room. The optimal width is a property of
the algorithm, which is why it stays a per-algorithm knob rather than a
constant.

Run-to-run variance is real and thermal: SHA-256d measured 0.704 and 0.632 on
the same settings minutes apart. Quote a range or quote the conditions.

## Do the tensor cores help? Measured: barely

The plan closed the tensor path for SHA-256 on the grounds that BMMA cannot do
carries. kHeavyHash is the fair test of the other case -- its heavy step *is* a
matrix multiply, 64x64 over GF(16), which is the shape a tensor core exists
for. `cuda/kheavy.cu` implements it three ways and refuses to time any of them
until all three match `tools/algocheck.py`.

| path | Mstep/s | vs scalar |
|---|---|---|
| scalar, int32 | 293 | 1.00x |
| dp4a, int8 on the INT pipe | 371 | 1.26x |
| wmma, int8 tensor cores | **383** | **1.31x** |
| everything except the matmul | 7628 | -- |

**The tensor core beats dp4a by 3%, and costs a batch of sixteen nonces to do
it.** `wmma::mma_sync` is warp-wide, so there is no single-nonce form; using it
forces a shape the algorithm does not otherwise want. dp4a needs no batching,
no shared memory and no staging, and lands within noise of it.

**The Amdahl explanation was wrong and the last row is how that was found.**
The obvious reading of a 1.31x result is that the matmul is a small part of the
step -- so the step was run with the multiply removed, and everything else
takes 41 ms of 1073. The matrix is **96%** of the work. Amdahl is not
available as an excuse.

What is left is that the problem is too small per warp. Each warp does
64x64x16, which is sixteen `mma` ops against three fragment loads each plus a
strided read back out of the accumulator, so the fixed cost of getting data
into and out of the tensor unit never amortises. Tensor cores want large
matrices; a proof-of-work step is a small one repeated forever, which is the
opposite shape.

Taken with the BMMA analysis, the conclusion for this hardware is the same from
two directions. SHA-256 cannot use the tensor cores because its critical path
is additions and they cannot carry. kHeavyHash can use them and gains almost
nothing because its matrix is too small. **Use `dp4a`.**

Two caveats, both real. This is the heavy step, not kHeavyHash: the full
algorithm wraps it in two cSHAKE256 passes, which are not implemented here
because cSHAKE is not in hashlib and an unverifiable hash has no business in
this tree. Those passes would only make the matmul a smaller share of the
whole, which weakens the tensor case further rather than strengthening it. And
the input feeding the benchmark is a cheap xorshift standing in for the Keccak
that would really produce it -- the matrix step does not care what it is
handed, only that it varies.

## What is not measured yet

- **Instructions per hash.** The derived figure depends on an assumed ALU
  utilisation, so it is not quoted. Nsight Compute gives it directly and that
  is the next run.
- **True undervolting.** The clock sweep above is not it, for the reason given
  there. Afterburner's curve editor is the only tool that can hold a clock
  below its curve voltage, and the harness is ready for it: the miner grades
  itself against hashlib at every setting, so the error rate is measurable
  rather than assumed.
- **The uniform datapath.** Untouched.
- **A persistent megakernel is worth about 6%, not the large win the plan
  implied.** Measured directly by holding total work fixed and cutting the
  launch count 30-fold: 3000 launches gives 0.599 GH/s, 300 gives 0.618, 100
  gives 0.634. Launch overhead is real and it is small. That bounds what
  eliminating it entirely can return, and it is worth knowing before building
  the thing whose whole justification was that GLaDOS has no TDR watchdog.
- **Anything on the ring-0 side.** This is all host CUDA. The GLaDOS probe is
  committed but has still never seen the GPU, because that needs a reboot.

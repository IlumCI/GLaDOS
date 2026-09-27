# Mining many coins at once, and which ones

Status: yespower is implemented in ring 0 and measured; see "Measured" below.
No coin has been mined. Every *network* figure is still absent on purpose
rather than guessed at, for the reason the last section gives. What is settled
here is the shape of the problem, the licence gate, and which algorithm family
the arithmetic points at.

## The idea

GLaDOS stays on x86-64 and mines several coins simultaneously, carving itself
into slices sized to what the low-power devices those networks are made of
actually have. The point is not to beat ASIC farms; it is to take a meaningful
share of a dozen small networks at once rather than a negligible share of one
large one. Monero is out by decision: it is the crowded one.

A GLaDOS-only pool aggregates all of it, converts, and pays out in $GLADOS, so
a miner sees one token rather than twelve coins.

**Owning the pool is what makes the kernel side small, and that is the
non-obvious part.** The pool speaks every upstream dialect -- Bitcoin's
`mining.notify` with coinb1/coinb2/merkle_branch to one coin, Monero's
`login`/`getjob` blob shape to another -- and GLaDOS speaks one protocol to us.
The right shape for that protocol is Monero's rather than Bitcoin's: the pool
pre-builds a blob, the kernel mutates a nonce inside it and hashes. **The
kernel then never learns what a coinbase or a merkle branch is, for any coin.**

The Stratum V1 client already in `src/mine/` is not wasted by that. It moves to
the pool as its *upstream* client, host-side, where it is ordinary code.

| | work |
|---|---|
| Kernel | N hash algorithms, one blob protocol, a supervisor with real resource budgets |
| Pool | N stratum dialects, N upstream connections, share accounting, conversion, payout |

## The budget being spent is L3, not cores and not RAM

Three limits, in the order they bite.

**Task slots bite first and are the easiest to design around.** `MAX_TASKS` is
24, `MAX_CPUS` is 16, every application processor takes a slot through
`adopt_idle`, and slots are never reclaimed. On the GF63 roughly twenty of the
twenty-four are spoken for before any mining starts, and a `mine off` / `mine
on` cycle burns more. So the unit is **a job per slice, with one supervisor
rotating which coin each slice works** -- never a task per coin, which caps out
in the low single digits and then bricks the machine.

**Heap is not the limit.** Boot reports `heap 320 MiB` plus `+1020 MiB across 1
more regions (1340 MiB total)`. At yespower's 8 MiB that is over a hundred
jobs' worth. Memory is not what runs out.

**L3 cache is the limit, and it is the interesting one.** yespower is built to
hammer L2 -- that is precisely the mechanism that makes it GPU-unfriendly --
and its 2 to 16 MiB working set exceeds any single core's L2 by design.
Concurrent jobs therefore contend in L3. Upstream's own PERFORMANCE file says
so directly:

> running 8 threads results in substantial slowdown

**The GF63's CPU is an i7-12650H: 10 cores, 16 threads, 24 MB of L3.** This
document said i5-12450H and 12 MB, and so did `src/smp.rs`, and the figure was
never read off the machine. At the 2 MiB setting 24 MB is **about twelve
concurrent jobs** before they steal from each other, not four to six; at 8 MiB,
about three.

That correction runs the wrong way from the usual one -- the budget is twice
what this document has been planning against, and the core count is four times
what QEMU shows. Every yespower figure below is therefore a floor rather than
an estimate, and by a margin nobody has measured.

**That figure is arithmetic and not a measurement**, and `mine sweep` is the
command that would settle it. Under QEMU it cannot, and the reason has got
worse now the real core count is known -- the guest sees four of sixteen:

    [mine sweep] yespower 1.0 N=2048 r=8
      1 slice(s)  387 H/s  (100% of one)
      2 slice(s)  704 H/s  (181%)
      3 slice(s)  992 H/s  (256%)
      4 slice(s)  993 H/s  (256%)

The curve bends hard at three and goes perfectly flat after it -- and that is
**core count rather than cache**. The guest has four vCPUs and core 0 is
carrying the shell and the clock, so three is all there ever was. An emulator
with four cores cannot find a wall that only appears when jobs outnumber the
cache, so this measurement belongs on the GF63 and is in the hardware runbook
(`todo`) accordingly.

What the sweep *does* settle here is that the slices genuinely run in parallel:
2.56x on three of them is impossible for tasks sharing one core, which would
sum to 1x however many there were.

## yespower, measured upstream

Memory is `128 * N * r` bytes. Upstream's recommended settings:

| setting | N, r | memory |
|---|---|---|
| 1 MiB | 1024, 8 | 1 MiB |
| 2 MiB | 2048, 8 | 2 MiB |
| 4 MiB | 1024, 32 | 4 MiB |
| 8 MiB | 2048, 32 | 8 MiB |
| 16 MiB | 4096, 32 | 16 MiB |

Throughput, from upstream's PERFORMANCE file, on an i7-4770K (2013) with four
threads:

| version | N=2048, r=8 (2 MiB) | N=2048, r=32 (8 MiB) |
|---|---|---|
| yespower 0.5 | 3700-3800 H/s | ~803 H/s |
| yespower 1.0 | ~3995 H/s | ~831 H/s |

Roughly 1,000 H/s per core at 2 MiB and 200 at 8 MiB, on hardware a decade
older than the GF63. Those are four-thread figures on a four-core part, so they
already include some contention.

## Measured, in ring 0

**Everything in this section is a floor, and three separate factors say so.**
The figures are QEMU under WHPX with `-smp 4`, running `yespower-ref.c`
transliterated, on a guest that sees four of the machine's sixteen logical
processors. Against that: the host is an i7-12650H with 24 MB of L3, upstream's
own optimised implementation runs about 1000 H/s per core on bare metal, and
this one runs 125. **The eight-times gap is emulation and the reference
implementation together**, and neither has been separated from the other.

So a bare-metal, optimised, sixteen-thread figure is not knowable from here and
is not guessed at. What is knowable is the direction: every yespower number in
this document and in `tools/payrate.py` understates the machine, and `mine
sweep` on the GF63 is the one command that settles by how much. It is in the
hardware runbook for exactly that reason.

`mine bench <ms>` hashes flat out with no pool and no network. QEMU under WHPX,
`-smp 4`, best of one, five-second samples:

| algorithm | H/s | working set |
|---|---|---|
| sha256d | 88,297 | 0 |
| yespower 1.0, N=2048 r=8 | **125** | 2,146 KiB |
| yespower 0.5, N=2048 r=8 | 97 | 2,058 KiB |
| yespower 1.0, N=2048 r=32 | 33 | 8,296 KiB |

Two internal consistency checks fall out and both hold. **r=32 is 3.8x slower
than r=8**, where upstream's own figures give 4.8x -- the same shape, since `r`
scales the blockmix work linearly. And **0.5 is slower than 1.0 at identical
parameters**, which it must be: 0.5 runs salsa20/8 with PWXrounds=6 where 1.0
runs salsa20/2 with PWXrounds=3, so 1.0 is simply less work per hash. A build
where those two came out the other way round would have the versions swapped
somewhere.

The working set is what `128 * N * r` predicts plus the S-boxes, so
`Hasher::footprint` is telling the truth and the slice budget can be trusted to
it.

**Against upstream, we are about 8x slow**, and that is expected rather than
alarming: upstream's ~1000 H/s per core is `yespower-opt.c` on bare metal, and
this is `yespower-ref.c` under an emulator. The reference is what the vectors
are checked against; speed is a later change with them still passing.

### The number that matters is not the bench one

The same algorithm, measured **inside the mining loop** rather than flat out:

    mine bench   125 H/s   over 632 hashes in 5039 ms, 5 tasks
    mine (loop)   40 H/s   over 240 hashes in 5955 ms, 7 tasks

**Three times less, on the same machine and the same algorithm.** The bench
runs on the shell task while everything else is blocked; the miner is one
runnable task among seven and gets a round-robin share of one core.

Both numbers are honest and they answer different questions. The bench says
what the algorithm costs, which is what a comparison against upstream or
against another algorithm needs. The loop says what a slice actually delivers,
which is what the supervisor's accounting and every expected-value figure must
use. **Quoting the bench number in an EV calculation would overstate earnings
by 3x**, and `mine ev` therefore reads the loop's counter and prints the task
count beside it.

### And a pinned slice is not an unpinned one

Those two figures were taken with the miner pinned to core 0 like every other
task this kernel has ever spawned. A slice on a core of its own reads **387
H/s** -- three times the bench and nearly ten times the pinned loop.

Two things changed at once there and it would be dishonest to credit one: the
slice was unpinned *and* the sweep rests the shell on `hlt` rather than letting
it poll. So the honest statement is that a slice with a core to itself and
nothing competing does about 387 H/s under emulation, and how that splits
between the two causes has not been measured.

**Every rate in this section is a QEMU rate.** The kernel says so itself now:
anything printing a hash rate asks `dev::power::virtualised()` and appends a
note when a hypervisor is present, because a figure that does not say what it
is worth gets quoted as though it were hardware.

## Many coins at once, which is now a thing the kernel does

`src/mine/work.rs` is the table: four slots, each with its own label, its own
algorithm and its own job, and a supervisor saying which slice works which slot.
`mine coin <n> <label> <algo...>` installs one, `mine coin <n> off` removes it,
`mine coins` prints the table.

Measured under QEMU with three slices over two coins, which is the whole claim
of this section in one command:

    slot  label       slices  source   rate            algorithm
    1     zeny        2       fixture  676 H/s         yespower 1.0 N=2048 r=8
    2     bitcoin     1       fixture  238759 H/s      sha256d

Two different proof-of-work functions, on two different cores, at the same
instant. Three, with BLAKE2s added:

    slot  label       slices  source   rate            algorithm
    0     pool        0       pool     no job yet      blake2s (RFC 7693)
    1     zeny        1       fixture  351 H/s         yespower 1.0 N=2048 r=8
    2     verge       1       fixture  5756621 H/s     blake2s (RFC 7693)
    3     bitcoin     1       fixture  246235 H/s      sha256d

Slot 0 has no job and correctly gets no slice. That was a bug first: `assign`
spread slices over slots that *existed* rather than slots that were *workable*,
so a jobless slot took two of three slices and they parked. Workability is
having a job, and `set_template` and `drop_template` re-run the supervisor.

**The yespower row moved when the others started, and that is the whole L3
question showing up.** It read 390 H/s alone and 351 with a BLAKE2s slice and a
sha256d slice beside it -- 10% off for two neighbours that are not memory-bound
in anything like the same way. Under emulation that number means little; on the
GF63, with sixteen logical processors and a 12 MB L3, it is the measurement
`design/mining.md` has been pointing at from the start. Clearing the second put all three slices back on the first and it read
968 H/s, against 676 on two -- so the supervisor's re-spread is visible in the
figures rather than only in the table.

**The aggregate hashrate had to be abandoned, and that is the interesting
part.** One `HASHES` counter was a fair summary while every slice computed the
same function. Sum those two rows and the answer is 239,435 "H/s", which is
238,759 with rounding: the yespower work -- the work that is actually scarce and
actually worth something -- disappears entirely into a number dominated by the
cheap algorithm. So there is a counter per slot and the report prints a row per
coin with no total anywhere. `client::HASHES` survives only as the *sweep's*
counter, where one coin is in the table by construction.

**A slot's figures are forgotten whenever what produced them changes** -- the
algorithm, or its share of the slices. That was found rather than designed:
clearing a coin took slot 1 from two slices to three and it reported 792 H/s,
which is neither the two-slice rate nor the three-slice one and looks perfectly
plausible as either.

**Assignment is sticky, and that is a memory decision rather than a policy.** A
slice keeps its hasher across batches because `Yespower` owns up to 8 MiB of
working set and a batch at that setting is eight hashes; rotating a slice
between coins per batch would throw that allocation away and take it again
several times a second, spending more time in the allocator than in the
algorithm. So a slice stays on a coin until the table changes.

**A share carries its slot and is checked twice.** Only slot 0 has a connection
behind it. A fixture slot exists to be measured and its target is one nothing
meets -- but the day one does, through a mistyped difficulty or a target of all
ones, submitting it would send the pool a share for a header it never issued,
which is how a worker gets banned. The miner declines to queue it and
`drain_shares` declines to send it.

**And the fixtures are no longer the only filling.** This section used to end
by saying every slot but the pool's was fed by a fixture, because there was one
Stratum connection and no protocol that could carry several coins down it. That
protocol exists now: `src/mine/proto.rs`, `mine pool <host> glados`, and
`design/pool.md` for the argument. Measured, three coins on three algorithms
from one connection, 181 shares all accepted.

What is still true is that no chain sits behind any of it. The pool assembles
its own headers, so a share beating a network target would be worth nothing.
`mine coin` and its fixtures also stay, and are still the only way to measure a
coin without a pool at all -- which is what the L3 question on the GF63 needs.

## BLAKE2s, and why a third algorithm was worth the day

Not for the coins. It is that `algo.rs`'s claim -- one `Algo`, one `Hasher`,
and the header assembly, merkle fold, target comparison and Stratum client
shared unchanged -- cannot be established by two algorithms, because with two
there is no telling a seam from a coincidence. Nothing above `Hasher` moved.

It is written from RFC 7693 rather than ported, which was available here and
was not for yespower: BLAKE2s has a published specification with its own test
vectors, so a from-scratch implementation can be settled against something that
is not somebody's source file. The licence gate below never comes up.

**And it falsified a comment.** `algo.rs` said SHA-256d was "the only one with
a usable midstate, because it is the only one whose first 64 header bytes can
be absorbed once." The reasoning was right and the conclusion was about SHA-256
rather than about midstates: BLAKE2s is also a 64-byte block function over the
same 80-byte header with the nonce at offset 76, so block 0 is constant across
nonces and is compressed once. What actually makes yespower different is that
it puts all eighty bytes through PBKDF2 before the expensive part begins, so
there is no prefix to absorb.

Measured: **2,219,757 H/s** benched, **5.7 million** in a slice with a core to
itself -- the same bench-versus-slice gap every other algorithm here shows, and
for the same reason.

Five claims, against digests this kernel did not compute, and two of them are
about the failure that produces a hash function looking entirely healthy. The
counter in section 3.2 is the *message length* and not the block index, so a
short final block that is zero-padded without moving the counter makes `"a"`
and `"a\0"` collide; and a message of exactly one block must keep that block
for the `last` flag rather than compressing it as an interior one. Both are
asserted, in the kernel and in `tools/algocheck.py`, which holds the same
digests so they live in two places that have to agree.

## The licence gate

This decides where implementations may come from, and it is a gate rather than
a preference. The reasoning is the one `mkiso.py` already applies to Xash3D:
a GPL source read to write a kernel this project does not GPL is an obligation
nobody has decided to take on.

| source | licence | usable |
|---|---|---|
| [openwall/yespower](https://github.com/openwall/yespower) | **2-clause BSD** (Colin Percival, Alexander Peslyak) | **yes** |
| [tevador/RandomX](https://github.com/tevador/RandomX) | BSD-3 (tevador, Monero Project) | yes |
| [JayDDee/cpuminer-opt](https://github.com/JayDDee/cpuminer-opt) | **GPL-2** | **no** |

That last row is the trap. cpuminer-opt is where reference implementations of
most niche algorithms live, including yespower, GhostRider and dozens more, and
it is the first thing anybody searching for "how do I implement X" will find.

**The rule: go to each algorithm's own upstream, never to the multi-algo
miner.** yespower has its own BSD repository. RandomX has its own. Where an
algorithm exists *only* inside a GPL miner it costs a clean-room implementation
from the specification, or it does not get done -- and that cost belongs in the
candidate table beside the algorithm, not discovered afterwards.

BSD-2 needs the copyright notice retained, which is exactly the arrangement
`src/doom/` already has with room4doom's MIT: one file, marked at the top,
saying where it came from.

## Candidates, ranked by what they cost us

| algorithm | coins | cost | notes |
|---|---|---|---|
| SHA-256d | many | **done** | in ring 0, pinned by block 125552 |
| BLAKE2s | several | **done** | in ring 0, from RFC 7693, with a midstate |
| kHeavyHash | Kaspa family | **~done** | written in `exp/xpu`, oracle exists |
| **yespower / yescrypt** | **BitZeny, Yenten, Koto, WAVI, Veco, PRiVCY** | **low** | BSD upstream, scrypt-derived, 1-16 MiB |
| VerusHash | Verus | low-moderate | Haraka512 over AES-NI |
| Argon2d | Nimiq | moderate | well specified, memory-hard |
| AstroBWT | Dero | moderate | Burrows-Wheeler + Salsa20 + SHA3 |
| CryptoNight family | Conceal, others | heavy | five hash functions, 2 MB scratchpad |
| GhostRider | Raptoreum | **heavy** | x16r *plus* CryptoNight family, and the reference is GPL |
| RandomX | Monero, Zephyr, SAL | **heaviest** | a VM with a JIT; 2 GB fast mode does not fit the heap ladder, 256 MB light mode is ~10x slower |

**yespower wins on five axes at once**, which is why it goes first: cheapest to
implement, smallest working set so the most jobs fit, licence-clean at source,
CPU-only by construction rather than merely ASIC-unfriendly, and attached to
the smallest networks in the table. Nothing else scores well on all five.

**That paragraph is wrong, and the axis it is wrong about is the fifth one.**
"Attached to the smallest networks" was listed as an advantage, because a small
network means blocks a small miner can actually find. It is the same fact as
"nobody trades this", and the second reading is the one that decides whether
mining it is worth anything. Measured, `tools/prices.py`, two independent
sources:

    coin       algo        usd            24h vol usd    age
    bitcoin    sha256d     78141          30,017,587,394 1.1 h
    verge      blake2s     0.00264568     6,058,540      1.1 h
    kaspa      kheavyhash  0.03637        14,205,519     1.1 h
    digibyte   sha256d     0.0046092      2,447,078      1.0 h
    monero     --          510.17         145,506,516    1.0 h
    bitzeny    yespower    0.00023968     0              1493 d
    koto       yespower    0.00003222     2              1328 d
    yenten     yespower    0.00303093     36             440 d
    veco       yespower    --             0              134 d
    privcy     yespower    0.00002741     0              122 d
    wavi       yespower    0.00001598     0              2450 d

**Every yespower coin in the table above has no live market.** The most recent
of the six was last priced four hundred and forty days ago at thirty-six
dollars of daily volume; the oldest was priced in 2019. CoinPaprika does not
list BitZeny, Koto or Yenten at all, which is what a delisting looks like from
the outside, and Veco's two quotes are a factor of 4.6 apart because the two
sources are describing two different assets that share a ticker.

So the cheapest algorithm to implement is attached to coins that cannot be
sold, and yespower in ring 0 -- which is done, correct, and checked against
thirteen upstream vectors -- buys a share of nothing. That is not wasted work:
it is the first algorithm here that is CPU-only by construction, and the
measurement apparatus around it is what every later one inherits. But it is not
the coin to point the machine at, and this document said it was.

**The two algorithms with live markets are the two the GPU already has.**
BLAKE2s is one of Verge's five, at six million dollars a day, and kHeavyHash is
Kaspa at fourteen. Both are written in `cuda/`; BLAKE2s is also in ring 0.

**And that paragraph made the same mistake in the other direction, one section
after correcting it.** It went on to call kHeavyHash "the single highest-value
item left in this document" on the strength of that fourteen million a day.
Liquidity says a coin can be sold. It says nothing about whether this machine
can win a share of one, and those are two axes. Measured the next day, off
`api.kaspa.org`:

    network difficulty    1.678e16
    network hashrate      3.36e17 H/s   (336 PH/s)
    this RTX 3050         3.83e8 H/s    upper bound -- see below
    share                 1.14e-9,  one part in 878 million

The card's figure is `design/xpu.md`'s best kHeavyHash path, and it is an
*upper* bound rather than a rate: that number is the heavy step alone, and the
full algorithm wraps it in two cSHAKE256 passes that are not implemented. The
real one is lower and the conclusion does not depend on how much.

Put in money, because a ratio that small stops meaning anything: for this
device to earn **ten cents a day** on Kaspa, the network would have to issue
$88M of new coin daily, which is 8.7% of Kaspa's entire market capitalisation
every day. It is not close, and no amount of tuning a CUDA kernel moves a
number by nine orders of magnitude. Kaspa has had dedicated ASICs since 2023
and that is what 336 PH/s is.

The derivation is worth keeping because it also settled a fact about the chain:
`difficulty x 2 / 0.1 s` reproduces the API's own hashrate to within 0.15%,
which confirms Kaspa is running at ten blocks a second rather than one.

**Verge's BLAKE2s is the open question and it is open for a boring reason.**
Four Verge explorer endpoints answered 404 or did not resolve, so the
per-algorithm difficulty is not quoted here. That is not a gap needing a
scraper: `nbits` off a live `mining.notify` *is* the network target, `mine
probe` was built to read exactly that, and most Stratum pools hand out work to
anybody who authorises with a payout address. So the instrument exists and what
it needs is a connection rather than code.

Until it is measured, this document ranks nothing. What the price survey
established is which coins can be **sold**, and that is one of the two
questions.

## Both of those rankings asked the wrong question, and one number answers it

Three corrections now sit above this line and they are all the same shape.
Yespower was chosen for implementation cost. It was then refused for coin
liquidity. kHeavyHash was chosen for market size and refused for network
hashrate. Every one of those is a proxy, and the thing being proxied is
**revenue per unit of this machine's hardware**.

A multi-coin auto-exchange pool publishes exactly that. yiimp's `/api/status`
carries `estimate_current` per algorithm: what a unit of hashrate earned in a
day, in BTC, *after the pool sold whatever it mined*. Price, network
difficulty, block reward and the per-chain decimals constant are all already
inside it, because the pool did the selling and is quoting the proceeds.

That dissolves the blocker `pool/src/market.rs` records against computing an
expected value -- "how many base units make a coin is not on the wire" -- by
not needing the constant. And it dissolves the liquidity objection above:
**on an auto-exchange pool the coin's own market is the pool's problem, not
the miner's.** It pays in BTC whatever it mined. A dead coin is a risk zpool
carries, and its `actual_last24h` is what it managed to realise.

`tools/payrate.py` asks. Measured against zpool, with BTC at $78,141:

    algo        ours H/s      USD/day       what it is
    sha256      6.3e8         $0.000031     RTX 3050, through the pool
    blake2s     1.28e9        --            zpool does not serve it
    heavyhash   3.83e8        --            served, pays nothing, no miners
    yespower    342           $0.013645     one kernel slice, ring 0
    yespower    1368          $0.054581     four kernel slices

**The kernel's CPU on yespower out-earns the RTX 3050 on sha256d by about
eighteen hundred times.** That is the ranking, and it is the opposite of what
the last two sections concluded. The reason is not subtle once the right
question is asked: sha256d is ASIC territory and a laptop GPU is a rounding
error in it, while yespower is CPU-only *by construction*, so a laptop core is
a real participant in a small field. The original document said exactly that
under "CPU-only by construction rather than merely ASIC-unfriendly", and it was
the axis that mattered all along.

**Verge is unreachable for a different reason than being unprofitable.** Every
Verge-specific BLAKE2s pool named in every guide -- `xvg.blake2s.com`,
`xvg.antminepool.com`, `cryptocartel.one`, `pool.verge-blockchain.com` -- fails
to resolve, and zpool does not carry blake2s among its seventy-five
algorithms. The coin has a live market and the algorithm has no pool. So the
GPU's second algorithm currently has nowhere to go, which is a fact about
infrastructure rather than about arithmetic and could change next month.

**The unit convention is derived and therefore checked.** zpool documents none
of it; `estimate_current` being BTC per `mbtc_mh_factor` MH/s per day was read
off the data, and a wrong reading moves every figure by three orders of
magnitude while still printing plausible pennies. `payrate.py --selftest`
checks it against a quantity nobody in the exchange controls: zpool's SHA-256
farm is a known fraction of Bitcoin's hashrate and Bitcoin's issuance is
published, so the pool's share of one must be its share of the other. Measured
ratio: **1.02**.

**And the conclusion has not moved once.** Five cents a day against an assumed
thirty-six cents of electricity is seven times underwater. Three rankings have
been overturned and the original objection at the top of this document is
untouched by all of them. What changed is which part of the machine is least
bad at it, and the answer is the part with no GPU in it.

**The finding took a field that has to be asked for.** CoinGecko answers
BitZeny with `0.00023968` and no error; `last_updated_at` is what says the
number is from August 2022, and it is opt-in. A fetcher that did not request it
would have written four-year-old prices into the pool and every expected value
after that would have been confidently wrong, with nothing anywhere reporting
it. `prices.py` requests it, prints it beside every figure, and writes a stale
coin into the file **marked** rather than omitting it -- an absent coin reads as
one nobody asked about.

Three algorithms are already most of the way there. Porting a hash kernel from
the CUDA in `exp/xpu` to CPU Rust is far cheaper than writing one, and
`tools/algocheck.py` transfers unchanged as the oracle.

## What is deliberately absent, and how to get it

**~~No network hashrates, no prices, no coins-per-day.~~ Prices are solved.**
The original note read: "miningpoolstats, poolbay and bitinfocharts all render
those in JavaScript and none of them yielded a figure." That was true of those
*sites* and it was never checked against the market -- CoinGecko and
CoinPaprika both answer plain JSON, with no key and no account, and
`tools/prices.py` reads both. The gap was a fact about three web pages that had
been recorded as a fact about the world, which is the same shape as this file's
own note about "too slow to test here" being an untested assumption about the
emulator.

Network *hashrate* is still not fetched and does not need to be: `nbits` off a
live `mining.notify` is the network target, which is the quantity the
arithmetic actually wants, and a hashrate is only that divided by a block time
somebody else assumed.

**The kernel already reads the real number off the wire.** `mine probe` pulls
`nbits` out of a live `mining.notify`, and `nbits` *is* the network target;
`ev::coinbase_value` sums the block's own outputs; `mine ev` turns those plus a
measured hashrate into coins per day. That is a measurement against the live
network rather than a table somebody typed.

So the way to score a coin is to point `mine probe` at its pool and read what
prints. The catch is circular and worth stating: a pool will not hand out work
a miner cannot do, so the algorithm has to exist before the coin can be scored.
Which fixes the order.

## The cache is a budget now, and the kernel reads it

`assign` handed out slices against the core count and nothing else. Slices run
at once, so their working sets are resident at once, and a memory-bound
algorithm exists to exceed a core's private cache -- that is the mechanism, not
a side effect. Handing out more slices than the last level holds is therefore
not neutral: it buys thrashing, and the report shows a rate that went *down*
when more of the machine was given to it.

`cpu::last_level_cache` reads CPUID leaf 4 and takes the largest cache it
enumerates. **Read rather than tabulated**, for the reason `mem::fixed` gives
about the memory map: a constant would be a claim about one laptop asserted in
a kernel meant to boot on another. It is also how the i5/i7 error above got
found -- the number had never been read, only written down.

    glados> mine slices
      1 slice(s) wanted, 0 spawned, 4 is the ceiling
      last-level cache 24576 KiB
        slot 1 wants 2146 KiB a slice, so 11 fit
        slot 2 wants 8296 KiB a slice, so 2 fit

Three things agree there and none was made to. 24576 KiB is the host's real L3
arriving through `-cpu max`; 2146 and 8296 KiB are what `Algo::working_set`
computes; and they are the same figures `Yespower::footprint` reports in the
table above, which a boot claim checks across five parameter sets.

Two refusals are the interesting half. **A cache the processor will not
describe leaves the request alone** rather than throttling it, which is what
every build before this did -- a budget that silently capped a machine it could
not measure would be worse than no budget. And **a working set larger than the
whole cache still gets one slice**, never none, because zero would turn a
large-parameter coin into a silent no-op that reads from the report exactly
like a pool that has gone quiet.

An arithmetic-bound algorithm costs nothing against this budget. sha256d's
whole state is a few hundred bytes, so a slice on one is free however many are
running -- which is the practical form of `Algo::bound`, and the reason a mixed
table is worth more than a uniform one.

**What is still not measured is whether the mix actually wins.** Nothing has
run an arithmetic-bound and a memory-bound coin together and compared the pair
against each alone. That measurement does not belong under QEMU for the same
reason `mine sweep` does not: the guest sees four of sixteen threads, so the
curve bends on cores long before it could bend on cache. It goes in the
hardware runbook beside the sweep.

## NeoScrypt, which is the fourth algorithm and the first one measured end to end

Three implementations, one oracle, and the oracle came first.

    tools/neoscrypt.py     the reference, transliterated from ghostlander's C
    cuda/neoscrypt.cu      the RTX 3050, 190 kH/s
    src/mine/neoscrypt.rs  ring 0, 34,144 bytes of working set

All three are checked by the same two blocks, and the ring-0 one is checked in
ring 0: `diag mine` is **111 claims** now, up from 89, and the Feathercoin
vectors are among them. The cache budget from the previous section picks the
new algorithm up for free -- `mine slices` reports `slot 1 wants 33 KiB a
slice, so 737 fit`, against yespower's 2,146 KiB and eleven.

**Upstream ships no test vectors, so the chain is the vector.** Feathercoin
forked to NeoScrypt at block 432,000, and a block is only on that chain because
its digest beat the target its own `nbits` declares -- so reproducing one is a
coincidence at one in `2^256 / target`. Six blocks across the whole NeoScrypt
era, at odds from 1 in 7e7 to 1 in 8e10. The header assembly is pinned
separately by a property of the coin: Feathercoin is a Litecoin fork, so its
*block* hash is SHA-256d while its *proof of work* is NeoScrypt, and the
explorer's own block hash therefore confirms all eighty bytes before NeoScrypt
is asked anything.

**The vector earned its place on the first run.** `blkmix`'s output
permutation is evens-then-odds, which at r=2 is a swap of the middle two
chunks; the first version read upstream's comment and swapped chunks 1 and 3.
It returned 32 bytes, deterministically, and one flipped nonce bit moved 133 of
256 output bits -- textbook avalanche. Every internal claim passed. Only a real
block said no.

### What the card does with it, and two null results

    full kernel      176.2 ms / 32768 hashes     186 kH/s
    FastKDF only      38.7 ms                    846 kH/s

So **SMix is 78% of a hash** and a perfect FastKDF is worth 1.28x at most.

Two optimisations were predicted, written, measured and rejected, which is
worth more than the 190 kH/s figure:

**Interleaving the scratchpad is a 46% loss.** Every memory-bound CUDA kernel
wants per-thread arrays interleaved so a warp's thirty-two reads of "word i of
my own array" fall in one cache line. Written, and it measured 105 kH/s against
194. It cannot help here because the index is *data-dependent per thread* --
SMix reads `V[64 * (X[48] & 127)]` and `X[48]` is thirty-two different values
across a warp -- so the reads scatter whatever the layout, and interleaving
destroys the locality that is available.

**Occupancy is not the bound either.** The kernel uses all 255 registers a
thread may have, which caps a multiprocessor at 256 threads. Capping registers
by hand to buy warps back:

    -maxrregcount    64      96     128     168    none
    kH/s            159     188     188     194     191

Four times the resident warps and 22% of the rate. The machine is waiting on
something more warps do not hide.

**This is 190 kH/s and a tuned miner does five to ten times that.** The gap is
known and is not addressed: the scrypt-family technique splits the sixteen
words of each Salsa or ChaCha block across four cooperating threads and
exchanges them with warp shuffles. NeoScrypt chains its four chunks serially,
so the cooperation has to happen inside a block rather than across them, which
is a rewrite rather than a tuning pass.

### Equihash 192,7 was researched and refused

See `design/equihash.md`, which is kept in full. The short version is that the
refusal is architectural rather than about effort: everything in `src/mine/`
answers "given a header and a nonce, what is the digest", and Equihash answers
"given a header and a nonce, which 400-byte solutions exist, if any". The wire,
the miner, the pool's validator and `cuda/algo.cuh`'s contract would all have
to change. Memory is survivable at 3.26 GiB against 3,836 free.

One finding from it belongs here rather than only there, because it is the same
failure this document keeps recording: **zpool serves one `equihash192` bucket
spanning four coins with three different BLAKE2b personalisation strings.** A
solver with one hardcoded prefix is wrong for two of the four, at full speed,
producing structurally perfect solutions that every pool rejects with no
diagnostic.

### And the refusal has a price now, which it did not when it was made

`$/worker/day` is a second ranking axis, and it is the one that says what a
field is worth entering rather than what this machine gets out of it. It is the
pool's own last-24h payout for an algorithm divided by the workers behind it,
which `payrate.py` already computes. Measured against zpool, restricted to rows
that are reachable -- not ASIC, no growing DAG, fits 3,836 MiB -- and not yet
implemented:

    equihash192      $0.4742 a worker a day    569 workers
    equihash144      $0.4721                    69
    yespowerEQPAY    $0.0499                   111
    verthash         $0.0300                   676
    yespowerURX      $0.0128                   129
    yescryptR16      $0.0121                   304
    yescrypt         $0.0111                  1843

For contrast, of the three algorithms actually built: neoscrypt pays $0.0271 a
worker a day over 62 workers and yespower $0.0058 over 332.

**So the two Equihash variants are roughly 7x anything else reachable, and an
order of magnitude above everything this repository has implemented.** That does
not overturn the refusal, which was never about the money -- it was that a
solver does not fit an abstraction built around `Hasher::hash` returning
`[u8; 32]`. What it does is put a number on the cost of that abstraction, which
is the honest way to hold a decision open: the wire needs a solution field, and
what buying it gets is the top of this table.

**A single reading cannot say which of these rankings are real, so there are
two, seven minutes apart.** `actual_last24h` is a trailing window over a pool
whose miners come and go:

    equihash192      $0.4742 -> $0.4903    +3.4%
    equihash144      $0.4721 -> $0.4689    -0.7%
    verthash         $0.0300 -> $0.0314    +4.7%
    yespowerEQPAY    $0.0499 -> $0.0325     -35%   and swapped rank with verthash

The Equihash pair's position is robust; third place is not. Anything with a
hundred-odd workers is noise at this resolution, which is the same objection
this file makes about single-worker rows and is worth making twice.

### What a pool-average worker is, which decides whether that table flatters us

Within one algorithm the hashrate unit is comparable, so the pool's
H/s-per-worker can be held against this machine's measured rate:

    yespower     pool 137.1 H/s      ours 1,368 H/s (emulated floor)   10x bigger
    neoscrypt    pool 403,400 H/s    ours 190,000 H/s                   0.47x
    sha256       pool 5.012e12 H/s   ours 0.63e9 H/s                    0.00013x

The CPU rows therefore **understate** what this machine would earn, before the
emulation floor is even lifted; the GPU row roughly halves it. And the sha256
row is the classifier's own conclusion arriving by a different route: an average
worker eight thousand times this GPU is an ASIC farm, so per-worker revenue
there is not an opportunity, it is a description of somebody else's hardware.

**No such calibration exists for equihash192 or equihash144**, because nothing
here has ever measured a rate on them -- which is exactly why `payrate.py`
leaves `ours $/d` blank for those rows rather than filling it in. The
pool-average equihash192 worker is 49.57 Sol/s. Whether an RTX 3050 Laptop
reaches that is unmeasured, and it is the single measurement standing between
$0.47 a worker a day and any claim about this machine.

## What mining costs the model, and standing down for it

The hash loop was built to avoid `smp::parallel_split` precisely so it would
not stall inference, and that argument is in `client.rs` and is correct. What
nobody had measured is what it costs anyway, simply by being runnable tasks on
a machine with four cores.

Measured under WHPX at `-smp 4`, three decodes of twelve tokens per condition,
one boot, SmolLM2:

    no mining              692  751  686 ms/token    median  692
    4 slices, no yield    1538 1452 1347             median 1452
    4 slices, yielding    1074 1109 1089             median 1089

**Mining doubles the cost of a token.** That is not a scheduling subtlety, it
is half the machine.

`YIELD_TO_MODEL` is the answer and it is on by default. A slice checks
`ai::engine_holder()` immediately before it hashes and parks if anybody holds
the engine. That recovers `(1452 - 1089) / (1452 - 692)` = **48% of the
penalty**.

**Why only half, stated rather than left to be found.** `with_engine` claims
the engine for one call, so a decode releases it between tokens and a slice
correctly works in the gap; and a parked slice keeps its working set resident,
which on a memory-bound algorithm costs the model bandwidth whether or not
anybody is hashing. Getting the rest means holding a claim across a whole
generation, which is a change to the engine and not to the miner.

**The default is an argument about what this machine is for.** The model in
ring 0 is the reason the kernel exists; mining is a side job that pays for the
token. A miner that silently takes half the arithmetic has inverted that, and
it does so invisibly -- the model does not visibly stall, it is merely slower
than it should be. `mine yield off` is there for anybody who disagrees, and it
prints what the choice costs.

**The first version of this measurement was worthless and that is worth
recording.** One sample per condition on the `bench` matmul put idle anywhere
between 2.64 and 5.30 GFLOP/s -- a 2x spread across a measurement of nothing
changing, entirely the host's scheduler. It is the same error `video bench` was
rewritten to stop making, made again, in a different subsystem, by somebody who
had read that note. Three samples with tight within-condition spreads is what
made the figures above quotable.

## Splitting one device across a field, which is mechanism and not policy

A card at half a gigahash is not one coin's worth of hashrate. The question is
how to divide it, and the answer has two halves that must not be one function:
**how** a device is shared out, and **what** each coin is worth. This is the
first half. Nothing here decides the second.

`miner/src/main.rs::choose` is the whole of it -- least-virtual-time over the
slots that have work this device can do, weight per slot, ties by slot number.
Three things in that are decisions.

**The unit is time and not hashes.** A yespower hash is roughly a thousand
times the work of a sha256d hash, so a scheduler counting hashes hands the
machine to the slow algorithm: it looks permanently behind however long it has
run. Seconds are the only quantity that means the same thing on both, and it is
the same objection this document's own measurement section makes about a
"total hashrate" across algorithms -- adding those two numbers and calling the
sum a speed. The per-slot report keeps them apart for that reason.

**The batch is measured rather than chosen, per algorithm.** `SLICE` is 250 ms
and the loop asks each backend for however many nonces that buys, learned from
the last scan. That matters because the two ends are four orders of magnitude
apart: the same constant has to mean 128M nonces on the GPU and about eighty on
one core of yespower. An algorithm the device has not timed gets a 4096-nonce
probe, small in absolute terms so the probe is never the thing that blocks, and
two scans take it to a full slice. The GPU figure it converges on is 128M,
which is what `design/xpu.md`'s batch benchmark chose independently.

**Weights are the entire policy interface, and they are all 1 today.** The
scheduler cannot ask what a coin is worth, because nothing in this tree yet
knows: `mine ev` reads `nbits` off the wire and a coinbase value out of the
block, and there is still no price. When there is, a weight is where it goes,
and this file's rule about invented figures is why that number is not being
guessed at now. Measured, on the RTX 3050 against a two-coin pool:

    weights   btc (sha256d)      verge (blake2s)
    1 : 1     49% of the device  47%
    2 : 1     64%                32%
    3 : 1     73%                24%

**And the socket is charged to the device too.** The loop reads the pool once
per scan, so the read timeout is time the card is not hashing: fifty
milliseconds against a 250 ms slice measured as exactly that, two coins summing
to 73% of the wall clock with nothing accounting for the rest. Two milliseconds
now, with a separate 50 ms park for the case a long timeout was really there
for -- a miner with every slot refused, which should sleep rather than spin. 97%
afterwards.

## Sequencing

1. ~~**yespower in ring 0.**~~ **Done.** `tools/yespower.py` came first and
   carries upstream's own thirteen TESTS-OK vectors; `src/mine/yespower.rs`
   matches three of them, one verbatim. Wired to the loop and measured above.
2. ~~**Score the six coins with `mine probe` and `mine ev`.**~~ **Answered, and
   the answer is no.** The six are the yespower coins and none of them has a
   live price, so there is no expected value to compute -- see the correction
   under Candidates. `tools/prices.py` is the measurement. The calibration set
   is Verge and Kaspa instead, and scoring those needs a pool account and a
   payout address rather than more code.
3. ~~**Measure the concurrency curve.**~~ **`mine sweep` exists and runs.**
   The measurement itself is a GF63 job and is in the hardware runbook, because
   a four-vCPU guest plateaus on cores before it can reach cache. Still the
   largest unmeasured claim in this document.
4. **The supervisor**, allocating slices against that measured budget rather
   than against core count. Slices exist and are unpinned; what does not exist
   is anything that gives them *different coins*, which is the whole idea.
   **The host miner has this now** -- `choose`, above -- and the kernel does
   not. The mechanism transfers directly; what does not is the batch, since a
   kernel slice is preempted at 100 Hz rather than blocking in a scan.
5. **The pool**, which is independent of all of the above and could start in
   parallel: proxy first, device-agnostic, `xmrig` on somebody's Pi as its
   first client. **Built** -- see `design/pool.md`.
6. ~~**kHeavyHash**, which the price survey moved to the top of the list.~~
   **Withdrawn the day after it was written.** Kaspa is 336 PH/s of ASIC and
   this card would hold one part in 878 million of it -- see the correction
   under Candidates. The hash is written and measured either way, so nothing is
   lost by not wiring it up.
7. ~~**Measure Verge's BLAKE2s difficulty with `mine probe`.**~~ **Overtaken.**
   `tools/payrate.py` answers the ranking question directly off a pool's own
   published payout rate, and no BLAKE2s pool is reachable to probe anyway. The
   difficulty is still the right number for a coin we mine *directly*; it is
   not needed to decide what to point the machine at.
8. **Point the kernel at zpool's yespower port and take a real share.** This
   is the only step left that has never been done: everything green in this
   tree is green against our own stub. It needs the address in hand and
   nothing else.

Items 1 to 4 are kernel work and item 5 is not, so they do not block each
other. The pool can exist and take miners before GLaDOS is a useful client at
it, and that is the right order rather than a compromise: a pool needs miners,
and xmrig users exist today.

## The question none of this answers

Who mines there. A pool with no miners is a server bill, and the recruitment
story -- market-rate payouts in the mined coin, plus $GLADOS and RWA on top,
funded by the operator's fee rather than by passing miner proceeds through --
is an offer that still needs somebody to hear it. That is not a technical
problem and nothing in this document addresses it.

## Measured against a live upstream, 2026-09-25 -- and yespower is a dead end

Everything above points at yespower. The figures below were read off zpool's
own API and off this machine, and they say the algorithm this tree implements
is worth **three dollars a day across every miner on it**.

None of this was written down before, which is why it was measured twice. An
earlier session reached "roughly one times underwater" and left it in a
transcript, so a later one re-derived it from scratch. The numbers are here now
for that reason rather than because they are pretty.

### What this machine does

`glados-pool --bench`, best of nine, on the i7-12650H (10 cores, 16 threads):

    sha256d            1.643 us/hash      608,642 /s per core
    blake2s            0.484 us/hash    2,066,115 /s per core
    yespower 2 MiB  1647 us/hash              607 /s per core
    yespower 8 MiB  6548 us/hash              153 /s per core

**Settled readings, first discarded.** The first reading after a build is the
host's page cache and it is roughly double: three consecutive runs of the same
binary read 2031, 1233 and 1042 us. `CLAUDE.md` records this for `video bench`
and it was walked into here anyway, so an earlier version of this section quoted
1804 us / 554 H/s and 7025 us / 142 H/s. Those were contaminated first samples.

**And a per-core figure must not be multiplied by the core count.** yespower is
memory-hard, so sixteen 8 MiB working sets thrash a 24 MiB L3. cpuminer-opt on
this CPU measures 862 H/s at sixteen threads against ~285 on one: **three times
the throughput for sixteen times the threads**, and the per-thread rate collapses
81%. An earlier version of this section multiplied 554 by 16, called it 8,864 H/s
and derived revenue from it -- ten times too high, and it made a dead algorithm
look merely bad.

### What that earns, and the ceiling nobody had noticed

zpool, 2026-09-25, `actual_last24h`, at BTC $83,767:

| algo | network hashrate | pays **all** miners |
|---|---|---|
| equihash | 2,692,831 | ~$80,900/day |
| yescrypt | 1,745,879 | ~$21,600/day |
| equihash192 | 33,281 | ~$503/day |
| **yespower** | **49,988** | **$3/day** |
| yespowerr16 | 58,550 | $6/day |

**862 H/s is 2.0% of zpool's entire yespower network.** One laptop. The
algorithm pays three dollars a day in total, so a miner taking *all* of it
earns three dollars a day -- and no amount of hashrate, hardware or free
electricity moves that number. `glados-pool.service` is configured for
`bitzeny:yespower-10-2048-8`, which is this.

**And the "cross-check" this section used to claim was circular.** Share of
network times total payout, and per-MH/s times hashrate, are the same two API
fields rearranged -- they cannot disagree, so their agreeing proved nothing. What
is actually pinned is the *share*: 862 H/s against a 43,772 H/s network is 2.0%,
and that is unit-free. The dollars are not pinned, and the plausibility test
below is the reason to distrust them.

**It is not right for every algo, and that trap is worth stating.** Reading
`actual_last24h` as BTC per MH/s per day uniformly gives sha256 a total payout
of $143 *trillion* per day. The field's unit tracks each algorithm's own
natural hashrate scale, so cross-algo ratios from this API mean nothing until
each one is pinned separately. The four rows above are ordered by implied
total, which is the comparison that survives; treat the absolute dollars for
anything but yespower as unconfirmed.

### Supporting an algorithm means verifying it, not mining it fast

The pool hashes **once per share** -- `--bench` says so in its own output -- so
the cost of accepting a new algorithm is a correct verifier and nothing else.
A 1,804 us validation is already fine at any share rate an operator would set.
Miners bring their own software; cpuminer-opt speaks yescrypt.

That makes yescrypt the cheap move rather than the ambitious one, because
**yespower is yescrypt's child and `src/mine/yespower.rs` already has the hard
half**: `salsa20`, `pwxform`, `blockmix_salsa`, `blockmix_pwxform`,
`integerify`, `p2floor`, `wrap`, `pbkdf2_1`, and sha256/hmac through
`crate::store`. What is missing is the KDF wrapper around them. Published test
vectors exist, so it can be checked the way every other primitive here is
rather than by mining something and hoping.

Equihash is worth four times more and is not the cheap move: a different
algorithm, a Wagner solver, and the GPU work already deferred once in
`cuda/`.

### Where the money actually goes, read off the code

`pool/src/upstream.rs` authorises **one** username per coin, from the spec's
`@host:port,user,pass`, and forwards every miner's share under it. There is no
per-miner username and **no fee code anywhere in `pool/src`**. So the upstream
pays the operator for all work by all miners, the operator owes the miners a
share, and the split is a policy choice in `tools/distribute.py` rather than
anything the pool enforces. The "non-custodial" claim in `design/runbook.md`
describes the *distributor* -- once an epoch is open the operator cannot
withhold a claim -- and not the revenue path.

That is what funds the GLADOS side: the bonus does not come out of the
operator's savings, it comes out of work the pool was paid for.

### The bridge floor, which caps the literal loop

zpool pays in BTC on Bitcoin's chain. An epoch is funded in WETH on chain
4663. So real revenue reaches an epoch only as payout -> exchange -> bridge,
and every hop has a fixed fee. At roughly $0.04/day for yespower it would take
*years* to accumulate the $30 that is the smallest batch worth bridging, which is
another way of saying the algorithm has to change before the bridge is even a
question.

So the honest arrangement separates two things that were being conflated: the
ledger says *who earned what*, and the treasury says *where the money comes
from*. Epochs are funded from treasury WETH; upstream revenue accumulates
off-chain and tops the treasury up in rare large batches. The mining is real
and the funding is pooled, which is how pools work anyway -- and it means the
payout contract can be deployed and used before any bridge has ever run.

### One operator's electricity is not every miner's

This operator pays nothing for power, so yespower's 862 H/s is about $0.04/day of
pure profit and the earlier "underwater" finding does not apply to them. It
still applies to everybody else: a miner paying $0.30/kWh spends about
$0.58/day to run sixteen threads. The pool's attractiveness depends on *their*
arithmetic, not on the operator's, and a pool whose only algorithm pays $3/day
across the whole network has nothing to offer either of them.

### The two measurement errors this section was written with

Both are corrected in place above rather than left as an appendix, because a
figure somebody has to read three sections to correct is a figure that will be
quoted wrong. They are named here because each is a repeat of something this tree
already had written down.

**The first reading after a build is the page cache**, not the code. Every figure
here is now the settled reading with the first discarded. `CLAUDE.md` says this
about `video bench` in as many words.

**A per-core rate times the core count is not a throughput**, for anything
memory-hard. cpuminer-opt measures it directly on this CPU: 285 H/s on one
thread, 862 on sixteen. This file warns about small-sample extrapolation in four
other places and then did it.

### Where this kernel's advantage actually is, and what it is worth

Like-for-like on one core at identical parameters:

    cpuminer-opt, hand-tuned AVX2    ~285 H/s
    src/mine/yespower.rs, scalar      153 H/s
    src/mine/yespower.rs, SSE2 lane   248 H/s   <- since the lane was vectorised

`grep -cE 'avx|sse|simd|_mm_|target_feature' src/mine/yespower.rs` answers
**0**. So the gap is 2.0x and it is entirely the inner loop; 2x is what scalar
against AVX2 costs on a salsa20/pwxform kernel, which means the implementation
is sound rather than sloppy.

The advantage is not in the inner loop and never was. It is that **the
bottleneck at thread scale is cache, and this kernel has levers Linux does
not**: `work::cache_budget` already sizes slices to the cache instead of running
N thrashing threads, which cpuminer-opt has no equivalent for; the identity map
is 2 MiB pages, so an 8 MiB random-access working set costs almost no TLB
pressure where a Linux process without hugepages pays continuously; nothing else
on the machine evicts L3; and there are no speculation mitigations and no
preemption inside a hash.

Sized honestly: three uncontended slices at AVX2 speed is about 855 H/s, and
TLB plus exclusivity gains put the ceiling near **940-1,110 H/s against
cpuminer's 862**. That is **tens of per cent, not multiples**, and none of it is
reachable before the inner loop is vectorised. An order-of-magnitude claim for
ring 0 alone would be false.

### The three levers, ranked by what they are actually worth

Measured on this CPU at 16 threads, against the per-kH/s rates above:

| lever | worth |
|---|---|
| **choosing the algorithm** | **~1,300x** |
| vectorising the inner loop | 2x |
| the ring-0 advantages | ~1.2x |

    minotaurx     4,031 H/s    ~$57/day
    yescrypt      3,740 H/s    ~$46/day
    yespowerR16     698 H/s    ~$0.07/day
    yespower        862 H/s    ~$0.04/day   <- what this tree implements

So the order of work is settled by arithmetic rather than by preference:
implement minotaurx and yescrypt first, vectorise second, and the ring-0 levers
are the last few tens of per cent on top. `rinhash` is the one candidate with no
reading -- cpuminer-opt knows the name but produced no rate here -- and it is
also the highest per-kH/s figure in the table, so it is worth one more attempt
before the list is called final.

### AVX2 was not the next lever, and the measurement says why

The SSE2 pwxform lane bought 1.58x and closed the gap to cpuminer-opt from 1.87x
to 1.15x. The obvious next step was to widen it, and the obvious target was the
bulk XORs, because at the yescrypt settings one hash does **2,097,152 word
XORs** -- more words than the pwxform lanes touch -- and `objdump` on `smix2`
found 188 `mov`, 11 scalar `xor` and 37 bounds-check branches with no SIMD
instruction anywhere in it.

It bought nothing. Interleaved A/B, two binaries alternating on one machine,
`yespower` 2 MiB:

    lane only        2014   1050   1060   2014   1158   1062 us
    lane + SIMD XOR  1066   1070   1055   2047   1110   1062 us

The ~2014 readings appear in **both** arms, which is what says they are the host
and not the change. The medians are 1061 and 1066 us: identical within noise.

**Because those XORs are DRAM-latency bound, not instruction bound.** Each one
reads a 256-word block from a pseudo-random offset in a 2 MiB or 8 MiB array, so
the cost is the cache miss and the arithmetic is free. Vectorising the arithmetic
removes no memory traffic whatsoever. That is yescrypt behaving exactly as
designed -- it is a *memory*-hard function, and the scalar XOR was already
keeping up with the memory system.

So the change was reverted rather than kept: it added unsafe pointer arithmetic
for no measured gain, and the SSE2 lane -- which is compute and L1 bound, and
where the 1.58x came from -- stays.

**Two failed attempts on the way, both worth keeping because each looked
right.** `iter_mut().zip()` in an `inline(always)` helper produced no SIMD at all
and ran ~6% slower: `noalias` is a property of the call boundary, and
`inline(always)` deletes the boundary, handing LLVM back two slices of one
structure it cannot separate. The same helper as `inline(never)` did vectorise
and was slower still, 1147 us against 1042, because `blockmix_pwxform` calls it
**65,536 times per hash** on 16-word blocks and a real call costs more than the
four SSE2 operations it wraps.

**What is left for AVX2, stated honestly.** Only the pwxform lane, processing two
lanes per 256-bit register. It is legal at the 0.5 settings, where lanes are
independent because there is no S-box writeback, and illegal at 1.0 where the
writeback couples them. The arithmetic would halve; the two S-box loads would
not, since the two lanes read different addresses and must be combined with an
insert or a gather. Against L1-resident 8 KiB S-boxes that is plausibly 1.1-1.25x
and it costs runtime feature detection in a file shared with the pool, which
cannot reach the kernel's CPUID. Worth doing *after* the ring-0 levers, which are
worth more and need no detection at all.

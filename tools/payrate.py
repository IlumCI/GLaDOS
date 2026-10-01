#!/usr/bin/env python3
"""What this machine would actually be paid, asked of a pool that pays it.

    payrate.py                    the survey: every algorithm, and why each is
                                  reachable or not
    payrate.py --mine             only the ones we can already compute
    payrate.py --vram 3836        against a different amount of video memory
    payrate.py --power 0.36       against a different daily electricity cost
    payrate.py --selftest         check the derived unit convention
    payrate.py --selftest --offline    only the claims that need no network

### Why this replaces the arithmetic it was going to be

`pool/src/market.rs` says an expected value is `price x reward / (2^256 /
network target)` and records why it was not written: a coinbase output is in
the chain's own base unit, how many of those make a coin is not on the wire,
and writing 1e8 because Bitcoin uses it is an invented figure.

**A multi-coin auto-exchange pool publishes the answer directly.** yiimp's
`/api/status` gives `estimate_current` per algorithm -- what a unit of hashrate
earned per day, in BTC, after the pool sold whatever it mined. Price, network
difficulty, block reward and the decimals constant are all already inside it,
because the pool did the selling and is quoting the proceeds.

### Three filters, and the second is the one nobody expects

A high payout per worker is not an opportunity if the worker is an ASIC. So:

1. **What dominates the algorithm.** Hand-classified below, from what the
   algorithm *is*, and deliberately left `unknown` where this file cannot say.
   Per-worker hashrate is printed beside it as corroboration and is never the
   classifier -- 42 Sol/s of equihash and 42 MH/s of kawpow are not comparable
   quantities and a threshold over both would be nonsense.
2. **Video memory.** The card here has 4096 MiB, and a fixed demand is judged
   against it. A *growing* one is refused whatever the card has: an
   Ethash-family DAG rises every epoch forever, so implementing ProgPoW on
   4 GiB is a race against a clock that does not stop. That refusal needs no
   figure, which is the point -- the first draft of this table carried three
   invented DAG sizes and they promptly put `firopow` top of the build queue.
3. **Whether we have a measured rate.** For an algorithm this project has not
   implemented there is no honest projection of earnings, so none is printed --
   `src/mine/ev.rs`'s rule, that a field this machine does not know means the
   answer does not exist.

### The 2% nobody quotes, and why it is already in these numbers

Stock yiimp carries `YAAMP_FEES_EXCHANGE = 2`, and it has exactly one use in the
mining path (`core/backend/markets.php`):

    $coin->price = $market->price * (1 - YAAMP_FEES_EXCHANGE/100);

So every non-BTC coin is *stored* at 2% below what it can actually be sold for,
and every earning is stamped with that discounted price. It is not a line item
anywhere and no pool advertises it -- a 2% haircut on the price rather than a
fee on the payout.

**It is already inside every figure this file reports**, and that is the whole
reason to write it down here rather than add it to a spreadsheet later.
`estimate_current` is what the pool *credits*, after selling, so the haircut has
already been taken by the time the number is published. Adding it again would
double-count.

The trap is for anybody building a cost model from the pool's fee page: the
stated fee is 1%, the real deduction is 1% plus a 2% price haircut, and only the
first is written down. That it lands inside the measurement here is luck rather
than design -- it is a consequence of reading credited proceeds instead of
computing expected value, which was chosen for a different reason entirely.

### The unit convention is derived, so it is checked

zpool documents none of it. Read off the data: `estimate_current` is BTC per
day per `mbtc_mh_factor` MH/s, and `actual_last24h` is the same in mBTC. A
wrong reading moves every figure by three orders of magnitude while still
printing plausible pennies, so `--selftest` checks it against a quantity nobody
in the exchange controls: zpool's SHA-256 farm is a known fraction of Bitcoin's
hashrate and Bitcoin's issuance is published, so the pool's share of one must
be its share of the other. Measured ratio: 1.02.
"""

import argparse
import json
import sys
import urllib.request

UA = {"User-Agent": "glados-pool/payrate.py (+https://aperture.institute)"}
DEFAULT_POOL = "https://zpool.ca/api/status"

# The GF63's RTX 3050 Laptop, from `nvidia-smi`: 4096 MiB total, 3836 free.
DEFAULT_VRAM_MIB = 3836
# A laptop drawing ~60 W flat out is ~1.4 kWh a day; EU domestic is around
# EUR 0.26/kWh. Stated as an assumption and overridable, because it is the one
# number here that is not measured and it is the one the conclusion rests on.
DEFAULT_POWER_USD_DAY = 0.36
# Below this, `$/worker/day` is one or two rigs and a coincidence. The first
# run put `firopow` top of the table on a single worker, which is the
# sample-size discipline this project applies to accuracy figures arriving at
# a payout figure instead.
MIN_WORKERS = 5

# What this machine computes, in hashes a second, and where each figure came
# from. **Measured rather than rated**: every one is a number this project
# printed.
#
# **Where it was measured is not decoration.** The GPU rows are native Windows
# through the pool on a *busy* host, which `design/xpu.md` records as costing
# about a fifth -- conservative, and within a fifth of the truth. The yespower
# row is something else entirely: QEMU with four of the machine's sixteen
# logical processors, running the transliterated `yespower-ref.c`. Upstream's
# optimised implementation does about 1000 H/s per core on bare metal against
# this one's 125, and the host's 24 MB of L3 would hold about twelve concurrent
# 2 MiB jobs where the guest could only ever run three.
#
# So that figure is a **floor by an unmeasured margin**, and the tool says so
# rather than scaling it: a multiplier picked to feel right is the invented
# figure this file already got caught on once. `mine sweep` on the GF63 is what
# settles it.
OURS = {
    "sha256": (0.63e9, "native", "RTX 3050, cuda/xpu.cu, through the pool"),
    "blake2s": (1.28e9, "native", "RTX 3050, cuda/xpu.cu, through the pool"),
    "heavyhash": (0.383e9, "native", "RTX 3050, cuda/kheavy.cu heavy step -- an upper bound"),
    "neoscrypt": (0.190e6, "native", "RTX 3050, cuda/neoscrypt.cu, 32768 threads"),
    "yespower": (1368.0, "emulated", "4 kernel slices under QEMU on 4 of 16 threads, reference code"),
    # **Measured natively, and the number is about contention rather than about
    # the algorithm.** Eight `glados-miner --cpu` processes against a
    # yescrypt-only pool on the i7-12650H, 2026-09-28: 3,086 H/s in total, where
    # *one* process alone did 838. Eight times the processes bought 3.7 times the
    # hashrate, because eight 2 MiB working sets do not fit the 24 MB of L3 and
    # yescrypt is memory-hard by construction -- the same contention `smp bench`
    # records as one core reading 4570 MB/s alone and 3526 MB/s with seven merely
    # idling beside it.
    #
    # So this is a *measurement of this host* and not a rating of the algorithm,
    # and it is deliberately the sum of per-process bests, which is an upper bound:
    # the windows are not aligned, so a figure taken over one common window would
    # be slightly lower. Extrapolating the single-process 838 H/s by sixteen
    # threads would have given 13,400 -- the invented multiplier this file already
    # records being caught on, wrong here by a factor of four.
    #
    # This is the algorithm the pool is configured for, so its absence was the
    # ranking being blind to the one row that matters.
    "yescrypt": (3086.0, "native", "8 glados-miner --cpu processes on 16 threads, i7-12650H"),
}

# Who wins each algorithm, and what it costs to join.
#
# **Classified from what the algorithm is, never from the numbers.** A pool's
# per-worker hashrate corroborates this and cannot establish it: the units are
# not comparable across algorithms, so any threshold over all of them is a
# category error. Anything this file cannot say is `unknown` and prints as
# unknown, because "we did not classify it" and "nobody can mine it" are
# different facts and the first is an invitation to look.
#
# `vram_mib` is what a GPU needs resident, and `None` means memory is not the
# binding constraint.
#
# **The ProgPoW family is refused without a figure, on purpose.** The first
# draft of this table carried 3500 and 4400 MiB for those rows -- numbers
# nobody measured, which promptly put `firopow` at the top of the build queue.
# That is the invented-figure failure this repository documents at length,
# arriving inside the tool written to prevent it.
#
# The honest refusal needs no number: an Ethash-family DAG grows monotonically,
# a fixed amount per epoch, forever. A 4 GiB card either cannot hold today's or
# will not hold next year's, and implementing a proof-of-work whose memory
# demand only rises is a race against a clock that does not stop. `dag: True`
# says that, and it is a fact about the construction rather than a reading.
FIELD = {
    # --- ASIC, and not close ---
    "sha256":        ("asic", None, "Bitcoin ASICs"),
    "sha256csm":     ("asic", None, "SHA-256 variant, same silicon"),
    "scrypt":        ("asic", None, "Litecoin ASICs"),
    "x11":           ("asic", None, "Dash ASICs"),
    "qubit":         ("asic", None, "ASIC-mined since 2015"),
    "myr-gr":        ("asic", None, "Myriad-Groestl ASICs"),
    "groestl":       ("asic", None, "Groestl ASICs"),
    "skein":         ("asic", None, "Skein ASICs"),
    "lyra2v2":       ("asic", None, "Lyra2REv2 ASICs"),
    "x13":           ("asic", None, "chained-hash ASICs"),
    "lbry":          ("asic", None, "LBRY ASICs"),
    "equihash":      ("asic", None, "Equihash 200,9 -- Antminer Z15 class"),
    "heavyhash":     ("asic", None, "Kaspa-family kHeavyHash, ASICs since 2023"),
    "odocrypt":      ("fpga", None, "DigiByte's Odocrypt, which rewrites itself for FPGAs"),

    # --- a DAG that grows forever, which no 4 GiB card wins ---
    "kawpow":        ("dag", None, "Ravencoin, ProgPoW over a DAG that grows every epoch"),
    "meowpow":       ("dag", None, "Meowcoin, ProgPoW over a DAG that grows every epoch"),
    "evrprogpow":    ("dag", None, "Evrmore, ProgPoW over a DAG that grows every epoch"),
    "firopow":       ("dag", None, "Firo, ProgPoW over a DAG that grows every epoch"),

    # --- GPU, with a fixed memory demand judged against the card ---
    "equihash144":   ("gpu", 2000, "Equihash 144,5 -- Flux family"),
    "equihash192":   ("gpu", 2800, "Equihash 192,7"),
    "neoscrypt":     ("gpu", None, "scrypt variant, GPU territory"),
    "verthash":      ("gpu", 1200, "Vertcoin, anti-ASIC by design, 1.2 GB data file"),
    "blake2s":       ("gpu", None, "what cuda/xpu.cu already computes"),
    "equihash192":   ("gpu", 3336, "Equihash 192,7 -- a solver, not a hash; see design/equihash.md"),

    # --- CPU by construction ---
    "yespower":      ("cpu", None, "memory-hard, cache-resident, CPU-only by design"),
    "yespowerR16":   ("cpu", None, "yespower variant"),
    "yespowerADVC":  ("cpu", None, "yespower variant"),
    "yespowerEQPAY": ("cpu", None, "yespower variant"),
    "yespowerLTNCG": ("cpu", None, "yespower variant"),
    "yespowerMGPC":  ("cpu", None, "yespower variant"),
    "yespowerSUGAR": ("cpu", None, "yespower variant"),
    "yespowerTIDE":  ("cpu", None, "yespower variant"),
    "yespowerURX":   ("cpu", None, "yespower variant"),
    "yescrypt":      ("cpu", None, "yespower's ancestor"),
    "yescryptR8":    ("cpu", None, "yescrypt variant"),
    "yescryptR16":   ("cpu", None, "yescrypt variant"),
    "yescryptR32":   ("cpu", None, "yescrypt variant"),
    "ghostrider":    ("cpu", None, "Raptoreum, CPU-favouring by design"),
    "argon2d500":    ("cpu", None, "Argon2d, memory-hard"),
    "argon2d1000":   ("cpu", None, "Argon2d, memory-hard"),
    "argon2d4096":   ("cpu", None, "Argon2d, memory-hard"),
    "argon2d16000":  ("cpu", None, "Argon2d, memory-hard"),
    "minotaurx":     ("cpu", None, "CPU-favouring chained hash"),
    "m7m":           ("cpu", None, "Magi, CPU"),
    "balloon":       ("cpu", None, "balloon hashing, memory-hard"),
}

# Which device runs it and which bottleneck it waits on, mirroring
# `Algo::bound` in the kernel. **Both halves are needed and the first was
# missing at first**: two memory-bound algorithms contend only if they are on
# the same silicon, and grouping by bound alone put the GPU's NeoScrypt and the
# CPU's yespower in one bucket as though they competed. They do not share so
# much as a cache.
WHERE = {
    "sha256": ("gpu", "arithmetic"),
    "blake2s": ("gpu", "arithmetic"),
    "heavyhash": ("gpu", "arithmetic"),
    "neoscrypt": ("gpu", "memory"),
    "yespower": ("cpu", "memory"),
    # **The same bucket as yespower, and that is the point rather than an
    # omission.** They are one family at the same N and r, so they contend for
    # exactly the same thing -- the measurement that put `yescrypt` in `OURS`
    # showed it directly: eight processes bought 3.7x one process's hashrate
    # because eight 2 MiB working sets do not fit 24 MB of L3.
    #
    # Sharing a bucket means the concurrency section will not add them together,
    # which is correct: running both would split one memory system between two
    # jobs, not earn both figures. Without this row `yescrypt` printed as device
    # `?` and `waits on unknown`, so the one algorithm the pool is configured for
    # was the one the "what can run at once" answer could not reason about.
    "yescrypt": ("cpu", "memory"),
}


def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))


def btc_per_day(entry, hashes_per_second, field="estimate_current"):
    """The derived convention, in one place so the selftest checks what runs.

    The quoted unit is `mbtc_mh_factor` megahashes a second. `estimate_current`
    is BTC per that per day; `actual_last24h` is the same in milli-BTC.
    """
    factor = float(entry["mbtc_mh_factor"])
    if factor <= 0:
        return None
    rate = float(entry[field])
    if field == "actual_last24h":
        rate /= 1000.0
    return (hashes_per_second / 1e6 / factor) * rate


def btc_price():
    """Spot BTC, from the source `prices.py` already cross-checks.

    One source is enough here and would not be in `prices.py`: this figure
    scales every row identically, so it cannot change which algorithm wins, and
    an error in it shows as every number being wrong together.
    """
    u = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    return float(get(u)["bitcoin"]["usd"])


def reachable(algo, vram_mib):
    """Whether this machine could join that field, and the reason if not."""
    who, need, note = FIELD.get(algo, ("unknown", None, ""))
    if who == "asic":
        return False, "ASIC: " + note
    if who == "fpga":
        return False, "FPGA: " + note
    if who == "dag":
        # No figure, and none needed. See the note on FIELD.
        return False, "growing DAG: " + note + ", so a %d MiB card loses eventually" % vram_mib
    if who == "unknown":
        return None, "unclassified -- worth a look"
    if need is not None and need > vram_mib:
        return False, "needs ~%d MiB of video memory, the card has %d" % (need, vram_mib)
    return True, note


def selftest(offline=False):
    """The derived unit convention against Bitcoin's own issuance.

    Nothing here controls either side: the pool reports its own SHA-256
    hashrate and its own payout rate, Bitcoin issues a published amount per
    day, and the pool's share of the network must be its share of the issuance.
    A convention wrong by the usual factor -- 1e3, 1e6 -- fails by the same
    factor and could not be mistaken for noise.
    """
    ok = True

    def claim(name, cond, detail=""):
        nonlocal ok
        print("%-4s  %s%s" % ("ok" if cond else "FAIL", name, ("  [%s]" % detail) if detail else ""))
        ok = ok and cond

    # A synthetic entry first, so the arithmetic is checked with no network.
    # `--offline` is what CI runs: a step that fetched a third party's API would
    # fail on their uptime rather than on a defect, which is the objection
    # `prices.py` records about its own selftest.
    fake = {"mbtc_mh_factor": 1000.0, "estimate_current": "0.002", "actual_last24h": "4.0"}
    claim("one unit of hashrate earns one rate", abs(btc_per_day(fake, 1e9) - 0.002) < 1e-12)
    claim("ten units earn ten times", abs(btc_per_day(fake, 1e10) - 0.02) < 1e-12)
    claim(
        "actual is read as milli-BTC and estimate as BTC",
        abs(btc_per_day(fake, 1e9, "actual_last24h") - 0.004) < 1e-12,
    )
    claim("a zero factor is refused rather than dividing", btc_per_day({"mbtc_mh_factor": 0}, 1e9) is None)

    # The filters, which are the half that decides what gets built.
    claim("an ASIC field is refused", reachable("sha256", 99999)[0] is False)
    claim(
        "a growing DAG is refused on its construction and not on a guessed size",
        reachable("kawpow", 3836)[0] is False and reachable("kawpow", 65536)[0] is False,
    )
    claim(
        "a fixed memory demand is still judged against the card",
        reachable("equihash192", 3836)[0] is True and reachable("equihash192", 1024)[0] is False,
    )
    claim("an unclassified algorithm is neither claimed nor refused", reachable("nosuch", 3836)[0] is None)
    claim("and a reachable one is reachable", reachable("neoscrypt", 3836)[0] is True)

    if offline:
        return ok

    d = get(DEFAULT_POOL)
    sha = d["sha256"]
    pool_hs = float(sha["hashrate"])
    claim("the pool reports a sha256 farm at all", pool_hs > 0, "%.4g H/s" % pool_hs)
    mine = btc_per_day(sha, pool_hs)
    # Round numbers on purpose. Bitcoin's hashrate moves and its issuance is
    # 3.125 BTC per block at ten minutes; both are quoted coarsely because the
    # claim is about a factor of a thousand, not a percent.
    network_hs = 1.0e21
    issuance = 3.125 * 6 * 24
    expected = pool_hs / network_hs * issuance
    ratio = mine / expected if expected else 0.0
    claim(
        "the derived unit agrees with Bitcoin's issuance within 10x",
        0.1 < ratio < 10.0,
        "%.4f BTC/day derived against %.4f expected, ratio %.2f" % (mine, expected, ratio),
    )
    return ok


def survey(d, btc, vram_mib):
    """One row per algorithm the pool serves and somebody mines."""
    rows = []
    for k, v in d.items():
        hs, w, f = float(v["hashrate"]), int(v["workers"]), float(v["mbtc_mh_factor"])
        if w == 0 or hs <= 0 or f <= 0:
            continue
        thin = w < MIN_WORKERS
        pool_usd = btc_per_day(v, hs, "actual_last24h") * btc
        can, why = reachable(k, vram_mib)
        ours = OURS.get(k)
        rows.append({
            "algo": k,
            "per_worker": pool_usd / w,
            "hs_per_worker": hs / w,
            "workers": w,
            "reachable": can,
            "why": why,
            "thin": thin,
            "ours_usd": btc_per_day(v, ours[0], "actual_last24h") * btc if ours else None,
            "ours_where": ours[1] if ours else None,
            "ours_note": ours[2] if ours else None,
        })
    rows.sort(key=lambda r: -r["per_worker"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=DEFAULT_POOL, help="a yiimp-style /api/status (default %(default)s)")
    ap.add_argument("--vram", type=int, default=DEFAULT_VRAM_MIB,
                    help="usable video memory, MiB (default %(default)s)")
    ap.add_argument("--power", type=float, default=DEFAULT_POWER_USD_DAY,
                    help="assumed electricity, USD a day (default %(default)s)")
    ap.add_argument("--mine", action="store_true", help="only algorithms we already compute")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="with --selftest, run only the claims that need no network")
    a = ap.parse_args()

    if a.selftest:
        return 0 if selftest(a.offline) else 1

    d = get(a.pool)
    btc = btc_price()
    rows = survey(d, btc, a.vram)

    if a.mine:
        rows = [r for r in rows if r["ours_usd"] is not None]

    print("%-15s %-11s %-13s %-7s %-11s %s"
          % ("algo", "$/worker/d", "H/s a worker", "workers", "ours $/d", "verdict"))
    for r in rows:
        if r["reachable"] is False:
            verdict = r["why"]
        elif r["reachable"] is None:
            verdict = r["why"]
        elif r["thin"]:
            verdict = "only %d worker(s), so the rate is a coincidence" % r["workers"]
        elif r["ours_usd"] is None:
            # No measured rate, so no projection. Naming the absence rather
            # than filling it is `ev.rs`'s rule and the whole reason this table
            # can be trusted where it does print a number.
            verdict = "reachable, no rate measured here yet"
        else:
            verdict = r["ours_note"]
            if r["ours_where"] == "emulated":
                verdict += "  [FLOOR]"
        print("%-15s $%-10.4f %-13.4g %-7d %-11s %s"
              % (r["algo"], r["per_worker"], r["hs_per_worker"], r["workers"],
                 ("$%.6f" % r["ours_usd"]) if r["ours_usd"] is not None else "--",
                 verdict))

    have = [r for r in rows if r["ours_usd"] is not None]
    open_ = [r for r in rows if r["reachable"] is True and r["ours_usd"] is None and not r["thin"]]
    unknown = [r for r in rows if r["reachable"] is None]
    print()
    print("%d algorithms with miners on them; %d reachable and unmeasured, %d unclassified."
          % (len(rows), len(open_), len(unknown)))

    if open_:
        print()
        print("reachable and not implemented, best first -- this is the build queue:")
        for r in open_[:8]:
            print("  %-15s $%-9.4f a worker a day   %s" % (r["algo"], r["per_worker"], r["why"]))

    # **Concurrency pays only across bottlenecks, and only on one device at a
    # time.** Two algorithms competing for the same resource simply halve each
    # other, so the total is fixed and splitting averages the rates down; two
    # waiting on different things can overlap. `(device, bound)` is the pair
    # that decides it -- the GPU's NeoScrypt and the CPU's yespower are both
    # memory-bound and share nothing at all.
    if have:
        print()
        print("what can run at once, by device and by what it waits on:")
        best = {}
        for r in have:
            k = WHERE.get(r["algo"], ("?", "unknown"))
            if r["ours_usd"] > best.get(k, (0.0, None))[0]:
                best[k] = (r["ours_usd"], r["algo"])
        total = 0.0
        for (dev, b), (usd, algo) in sorted(best.items()):
            print("  %-4s %-11s %-12s $%.6f a day" % (dev, b, algo, usd))
            total += usd
        print("  %-4s %-11s %-12s $%.6f a day" % ("", "", "together", total))
        print()
        print("  That total is a prediction, not a measurement: nothing has yet run")
        print("  two of these at once and compared the pair against each alone.")
        print()
        print("against $%.2f a day of electricity assumed: %.0fx underwater."
              % (a.power, a.power / total if total else float("inf")))
        floors = [r["algo"] for r in have if r["ours_where"] == "emulated"]
        if floors:
            # The one direction the error is known to run in. Saying which rows
            # are floors is not the same as scaling them, and scaling them
            # would be a number nobody measured driving a conclusion -- which
            # is the failure this file was caught making about DAG sizes.
            print("[FLOOR] %s measured under emulation on a quarter of the cores,"
                  % ", ".join(floors))
            print("        running reference rather than optimised code. The real"
                  " figure is higher")
            print("        by a margin only `mine sweep` on the hardware can say.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

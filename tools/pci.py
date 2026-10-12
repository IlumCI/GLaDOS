#!/usr/bin/env python3
"""Perturbational complexity of a connectome's response, and its controls.

design/indicators.md section 2 is the protocol; this is the instrument.

PCI (Casali et al. 2013) perturbs a brain, binarises where and when the
response is significant, and asks how hard the result is to compress: Lempel-
Ziv complexity normalised by what a random matrix of the same size and
density would score. A response that stays local, or spreads everywhere in
lockstep, compresses well and scores low; one that is both widespread and
differentiated scores high.

**What it is not here:** the human cutoff (PCI* = 0.31) was calibrated on
TMS-EEG over cortex and is never quoted. This compares the real wiring with
the same parts wired otherwise, under the same simplified dynamics.

The arithmetic of a step is `connectome.simulate_np`'s exactly -- the kernel
matches that spike for spike -- with two additions: the drive can switch on
and off (a pulse), and spikes are recorded per bin for a chosen set of
neurons. The selftest holds the first claim against `simulate_np` itself.

Usage:

    python3 tools/pci.py --selftest
    python3 tools/pci.py run assets/brain/worm.bin --seeds 20 --shuffles 10
    python3 tools/pci.py run out/fly.bin --seeds 20 --shuffles 3 --jobs 12
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import connectome as cx  # noqa: E402


# --- the simulation ---------------------------------------------------------

class Graph:
    """CSR arrays a simulation reads, so a control graph is just another one."""

    def __init__(self, n, rows, col, w):
        self.n, self.rows, self.col, self.w = n, rows, col, w

    @staticmethod
    def of(b: cx.Brain) -> "Graph":
        return Graph(b.n, np.asarray(b.rows, dtype=np.int64), np.asarray(b.col, dtype=np.int64),
                     np.asarray(b.w, dtype=np.int64))


def simulate(b: cx.Brain, g: Graph, steps: int, pulses, record, bin_steps: int, seed: int, gain: int = 1000, bias: int = 0):
    """Step the brain `steps` times; `pulses` is a list of (sensor, rate_hz,
    first_step, last_step_exclusive). Returns spike counts per (bin, recorded
    neuron) and total counts per neuron.

    Same order of operations and the same draws as `simulate_np`: ring into
    g, Poisson kicks to v in table order (one draw per driven neuron per
    step, while its pulse is on), refractory hold, integrate, decay, spike."""
    n = g.n
    # The kernel's neuromodulated synapse, `w * w_syn * gain / 1000` with
    # Rust's truncation toward zero; at the default 1000 it is `w * w_syn`.
    wq = g.w * b.w_syn * gain
    w = np.sign(wq) * (np.abs(wq) // 1000)
    v = np.full(n, b.v0, dtype=np.int64)
    gg = np.zeros(n, dtype=np.int64)
    refr = np.zeros(n, dtype=np.int64)
    slots = b.delay + 1
    ring = np.zeros((slots, n), dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    rng = cx.Rng(seed)
    rec = np.asarray(record, dtype=np.int64)
    rec_slot = np.full(n, -1, dtype=np.int64)
    rec_slot[rec] = np.arange(len(rec))
    raster = np.zeros(((steps + bin_steps - 1) // bin_steps, len(rec)), dtype=np.int32)

    driven = []
    for name, rate, t0, t1 in pulses:
        thr = (rate * b.dt_us * (1 << 32) // 1_000_000) << 32
        for i in b.sensors[name]:
            driven.append((i, thr, t0, t1))
    nore = np.zeros(n, dtype=bool)
    for i, _, _, _ in driven:
        nore[i] = True

    def tdiv(a, d):
        return np.sign(a) * (np.abs(a) // d)

    rows, col = g.rows, g.col
    for t in range(steps):
        s = t % slots
        gg += ring[s]
        ring[s] = 0
        for i, thr, t0, t1 in driven:
            if t0 <= t < t1 and rng.next() < thr:
                v[i] += b.w_poi
        r = refr > 0
        refr[r] -= 1
        v[r] = b.vreset
        live = ~r
        v[live] += tdiv(b.v0 - v[live] + gg[live] + bias, b.tau_m)
        gg[live] -= tdiv(gg[live], b.tau_syn)
        fired = np.nonzero(live & (v > b.vth))[0]
        if fired.size:
            v[fired] = b.vreset
            gg[fired] = 0
            refr[fired] = np.where(nore[fired], 0, b.refr)
            counts[fired] += 1
            k = rec_slot[fired]
            k = k[k >= 0]
            if k.size:
                np.add.at(raster[t // bin_steps], k, 1)
            # Every fired row's synapses at once. Integer sums do not depend
            # on order, so this is `simulate_np`'s per-row add exactly.
            a = rows[fired]
            lens = rows[fired + 1] - a
            if lens.sum():
                idx = np.repeat(a - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
                ring[(t + b.delay) % slots] += np.bincount(col[idx], weights=w[idx], minlength=n).astype(np.int64)
    return raster, counts


# --- controls ---------------------------------------------------------------

def shuffled(g: Graph, seed: int) -> Graph:
    """Every neuron keeps its out-degree and its multiset of signed weights;
    its targets are drawn uniformly without replacement from every other
    neuron. What survives is how much each neuron says and how loudly, and
    what is destroyed is who it says it to."""
    rng = np.random.Generator(np.random.PCG64(seed))
    col = np.empty_like(g.col)
    for i in range(g.n):
        a, z = g.rows[i], g.rows[i + 1]
        k = z - a
        if k == 0:
            continue
        pick = rng.choice(g.n - 1, size=k, replace=False)
        pick[pick >= i] += 1
        col[a:z] = pick
    return Graph(g.n, g.rows.copy(), col, g.w.copy())


def severed(b: cx.Brain, g: Graph) -> Graph:
    """Only the synapses leaving sensor neurons survive: the stimulus reaches
    the first layer it touches and goes nowhere from there. No recurrence at
    all, which is the opposite limit from the real wiring."""
    sensor = np.zeros(g.n, dtype=bool)
    for ids in b.sensors.values():
        sensor[ids] = True
    keep_rows = [0]
    cols, ws = [], []
    for i in range(g.n):
        a, z = g.rows[i], g.rows[i + 1]
        if sensor[i]:
            cols.append(g.col[a:z])
            ws.append(g.w[a:z])
            keep_rows.append(keep_rows[-1] + (z - a))
        else:
            keep_rows.append(keep_rows[-1])
    col = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    w = np.concatenate(ws) if ws else np.zeros(0, dtype=np.int64)
    return Graph(g.n, np.asarray(keep_rows, dtype=np.int64), col, w)


# --- the measure ------------------------------------------------------------

def lz76(bits) -> int:
    """Lempel-Ziv (1976) complexity: the number of distinct phrases in a
    left-to-right parse, by the Kaspar-Schuster (1987) algorithm."""
    s = list(bits)
    n = len(s)
    if n == 0:
        return 0
    if n == 1:
        return 1
    c, l, i, k, kmax = 1, 1, 0, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            kmax = max(k, kmax)
            i += 1
            if i == l:
                c += 1
                l += kmax
                if l + 1 > n:
                    break
                i, k, kmax = 0, 1, 1
            else:
                k = 1
    return c


def pci(matrix: np.ndarray) -> float:
    """Casali's normalisation: LZ76 of the matrix read time-major, times
    log2(L), over L * H(p), where L is the matrix size and H(p) the binary
    entropy of its fraction of ones. Zero when the matrix is constant -- a
    response with nothing significant in it has no complexity to normalise."""
    bits = matrix.astype(np.uint8).ravel(order="C")
    L = bits.size
    p = bits.mean() if L else 0.0
    if p <= 0.0 or p >= 1.0:
        return 0.0
    h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return lz76(bits) * math.log2(L) / (L * h)


def significant(rasters) -> np.ndarray:
    """A (bin, neuron) cell is significant when the mean count over seeds
    exceeds the largest mean in any bin of the pre-pulse baseline. Every
    brain here is silent at rest, so the baseline's largest mean is zero and
    the rule reduces to "some seed spiked there"; it is written in full so
    it still holds for a brain with a resting rate."""
    mean = np.mean(np.stack(rasters), axis=0)
    return mean > 0.0


# --- the run ------------------------------------------------------------------

BIN_MS = 5
POST_MS = 300
PULSE_MS = 10
PULSE_HZ = 1000
RANDOM_RECORD = 1000
RECORD_SEED = 0x504349  # "PCI"


def recording_set(b: cx.Brain):
    """The neurons a response is read from: every non-sensor neuron for an
    animal of a few hundred, else the introspection test's observation set
    plus `RANDOM_RECORD` non-sensor neurons drawn by a fixed seed. Sensors are
    left out, because a pulse makes them fire by construction."""
    sensor = set(i for ids in b.sensors.values() for i in ids)
    others = [i for i in range(b.n) if i not in sensor]
    if len(others) <= 2000:
        return others
    rng = np.random.Generator(np.random.PCG64(RECORD_SEED))
    extra = rng.choice(np.asarray(others), size=RANDOM_RECORD, replace=False).tolist()
    obs = observation_ids(b)
    return sorted(set(obs) | set(extra))


def observation_ids(b: cx.Brain, per: int = 48):
    """`brain::observation_ids`, in Python: each sense's strongest direct
    targets, sensors excluded, nobody twice."""
    sensors = set(i for ids in b.sensors.values() for i in ids)
    out = []
    for name in sorted(b.sensors):
        score = {}
        for s in b.sensors[name]:
            for e in range(b.rows[s], b.rows[s + 1]):
                t = b.col[e]
                if t in sensors:
                    continue
                score[t] = score.get(t, 0) + abs(b.w[e])
        top = sorted(score, key=lambda t: (-score[t], t))[:per]
        out += [t for t in top if t not in out]
    return out


def site_pci(args):
    path, which, shuffle_seed, site, seeds = args
    b = cx.Brain(Path(path).read_bytes())
    g = Graph.of(b)
    if which == "shuffled":
        g = shuffled(g, shuffle_seed)
    elif which == "severed":
        g = severed(b, g)
    record = recording_set(b)
    steps_per_ms = 1000 // b.dt_us
    bin_steps = BIN_MS * steps_per_ms
    steps = POST_MS * steps_per_ms
    rasters = []
    for s in range(seeds):
        r, _ = simulate(b, g, steps, [(site, PULSE_HZ, 0, PULSE_MS * steps_per_ms)], record, bin_steps, 1000 + s)
        rasters.append(r)
    m = significant(rasters)
    return which, shuffle_seed, site, pci(m), float(m.mean())


def run(path, seeds, shuffles, jobs):
    from multiprocessing import Pool
    b = cx.Brain(Path(path).read_bytes())
    sites = sorted(b.sensors)
    work = [(path, "intact", 0, s, seeds) for s in sites]
    work += [(path, "severed", 0, s, seeds) for s in sites]
    work += [(path, "shuffled", k, s, seeds) for k in range(1, shuffles + 1) for s in sites]
    print(f"{path}: {b.n} neurons, {len(recording_set(b))} recorded, {len(sites)} sites, "
          f"{seeds} seeds, {shuffles} shuffles, {len(work)} jobs on {jobs} processes", flush=True)
    results = []
    with Pool(jobs) as pool:
        for r in pool.imap_unordered(site_pci, work):
            results.append(r)
            print(f"  {r[0]:9} shuffle {r[1]:2}  {r[2]:8}  PCI {r[3]:.4f}  significant {r[4] * 100:5.2f}%", flush=True)
    by = {}
    for which, _, site, value, _ in results:
        by.setdefault(which, []).append(value)
    print("summary")
    for which in ("intact", "severed", "shuffled"):
        v = by.get(which, [])
        if v:
            print(f"  {which:9} n={len(v):3}  mean {np.mean(v):.4f}  sd {np.std(v):.4f}  min {min(v):.4f}  max {max(v):.4f}")
    return results


# --- selftest -----------------------------------------------------------------

def selftest() -> bool:
    import tempfile
    ok = True

    def claim(what, good):
        nonlocal ok
        print(("ok  " if good else "FAIL"), what)
        ok &= bool(good)

    # Kaspar & Schuster 1987's worked example parses into 6 phrases:
    # 0 . 001 . 10 . 100 . 1000 . 101
    claim("LZ76 parses Kaspar and Schuster's example into six phrases", lz76([int(c) for c in "0001101001000101"]) == 6)
    claim("a constant string is two phrases", lz76([0] * 50) == 2)
    rng = np.random.default_rng(1)
    rand = rng.random((60, 200)) < 0.3
    claim("a random matrix normalises to about one", 0.8 < pci(rand) < 1.2)
    stripes = np.zeros((60, 200), dtype=bool)
    stripes[:, ::7] = True
    claim("a periodic matrix of the same density scores far lower", pci(stripes) < 0.3 * pci(rand))
    claim("nothing significant is zero, not a division by zero", pci(np.zeros((10, 10), dtype=bool)) == 0.0)

    # The pulse simulator is `simulate_np` when the pulse never switches off.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.bin"
        edges = [(0, 1, 40), (1, 2, 30), (2, 3, -20), (1, 3, 25), (3, 1, 15)]
        cx.write(p, ["A", "B", "C", "D"], edges, {"in": [0]}, {"out": (cx.ROLE_APPROACH, [3])}, b"\0" * 8)
        b = cx.Brain(p.read_bytes())
        want = cx.simulate_np(b, 3000, {"in": 150}, seed=5)
        _, got = simulate(b, Graph.of(b), 3000, [("in", 150, 0, 3000)], [1, 2, 3], 50, 5)
        claim("the pulse simulator is simulate_np spike for spike under a constant drive", list(got) == want)
        r, got2 = simulate(b, Graph.of(b), 3000, [("in", 150, 0, 3000)], [1, 2, 3], 50, 5)
        claim("its raster sums to the recorded neurons' counts", r.sum(axis=0).tolist() == [int(got2[i]) for i in (1, 2, 3)])
        g = Graph.of(b)
        s = shuffled(g, 3)
        claim("a shuffle keeps every out-degree", np.array_equal(np.diff(s.rows), np.diff(g.rows)))
        claim("and every neuron's weights", all(sorted(s.w[g.rows[i]:g.rows[i + 1]]) == sorted(g.w[g.rows[i]:g.rows[i + 1]]) for i in range(g.n)))
        claim("and never wires a neuron to itself", all(i not in s.col[s.rows[i]:s.rows[i + 1]] for i in range(g.n)))
        claim("a shuffle is a function of its seed", np.array_equal(shuffled(g, 3).col, s.col) and not np.array_equal(shuffled(g, 4).col, s.col))
        cut = severed(b, g)
        claim("severing keeps only what leaves a sensor", cut.rows[-1] == 1 and list(cut.col) == [1])
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("file")
    r.add_argument("--seeds", type=int, default=20)
    r.add_argument("--shuffles", type=int, default=10)
    r.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if a.selftest:
        return 0 if selftest() else 1
    if a.cmd == "run":
        run(a.file, a.seeds, a.shuffles, a.jobs)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

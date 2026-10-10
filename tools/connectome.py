#!/usr/bin/env python3
"""Build a connectome into GLADOSB1, and simulate it the way the kernel does.

A connectome is a wiring diagram read off an electron microscope: which
neuron synapses onto which, and how many times. It is not a brain's
dynamics -- the strengths, the time constants and the neuromodulation are not
in it -- so this file pairs one with the dynamics the field already uses for
exactly that gap: Shiu et al. 2024 (Nature 634:210), a leaky integrate-and-
fire neuron per cell, weight = synapse count x a sign from the predicted
transmitter, one set of constants for every neuron. The constants below are
theirs.

Two halves, and the second is the reason the first can be trusted:

    build    a GLADOSB1 file the kernel loads (src/brain/)
    sim      the reference simulation, in integers, step for step what the
             kernel computes -- `diag brain` requires the kernel's spike counts
             to equal this program's with `==`, which is why it is integer
             arithmetic with truncation spelled out rather than floats

Rungs, smallest first:

    worm   C. elegans hermaphrodite, Cook et al. 2019 (Nature 571:63), via
           OpenWorm's ConnectomeToolbox (MIT). 300-odd neurons. Transmitter
           identity from Wang et al. 2024 (eLife 95402, CC BY 4.0): eat-4 is
           glutamate, unc-17 acetylcholine, unc-25/unc-47 GABA.
    fly    adult Drosophila, FlyWire v783 (Dorkenwald et al. 2024, CC BY 4.0),
           signs from Eckstein & Bates 2024. Not built by this version yet.

Simplifications, stated so nobody has to discover them:

- Gap junctions are dropped. The worm has many and they matter; Shiu's fly
  model drops them too, and adding them is a conductance term, not a format.
- Sign is per presynaptic neuron. GABA inhibits; acetylcholine and glutamate
  excite. Glutamate inhibits through GluCl in both animals, and a per-synapse
  sign needs receptor data neither dataset carries -- this is the first thing
  wrong with the model and it is wrong the same way every published LIF
  connectome model of the worm is.
- A neuron with no transmitter call is excitatory, and the build prints how
  many there were.
- Integration is forward Euler in integer nanovolts at dt = 0.1 ms, which is
  not what Brian2 computes. The kernel and this file agree with each other
  exactly; neither claims to agree with Brian2 to the last spike.

    python3 tools/connectome.py fetch worm out/worm-src
    python3 tools/connectome.py build worm out/worm-src esp/GLADOS/BRAIN.BIN
    python3 tools/connectome.py sim esp/GLADOS/BRAIN.BIN --poke noxious --ms 500
    python3 tools/connectome.py --selftest
"""
import argparse
import csv
import hashlib
import struct
import sys
from pathlib import Path

MAGIC = b"GLADOSB1"
VERSION = 1
HEADER = 128

# Shiu et al. 2024, model.py, in the units the integer simulation uses:
# nanovolts and steps of 0.1 ms.
DT_US = 100
V0_NV = -52_000_000          # rest, and reset
V_TH_NV = -45_000_000
TAU_M = 200                  # 20 ms
TAU_SYN = 50                 # 5 ms
REFR = 22                    # 2.2 ms
DELAY = 18                   # 1.8 ms
W_SYN_NV = 275_000           # 0.275 mV per synapse
W_POI_NV = 250 * W_SYN_NV    # Shiu's f_poi = 250: one input spike is a spike

ROLE_APPROACH, ROLE_AVOID, ROLE_AROUSAL = 0, 1, 2

# What the OS's senses reach in the worm. Named cells, each a textbook
# function; the kernel's `brain::senses` table points at these by name.
WORM_SENSORS = {
    # Gentle body touch (Chalfie et al. 1985).
    "touch": ["ALML", "ALMR", "AVM", "PLML", "PLMR", "PVM"],
    # Thermosensation (Mori & Ohshima 1995).
    "thermal": ["AFDL", "AFDR"],
    # Attractive chemosensation -- salt (ASE) and odour (AWC). The fly's
    # sugar neurons play this part in the fly rung.
    "food": ["ASEL", "ASER", "AWCL", "AWCR"],
    # The polymodal nociceptor (Kaplan & Horvitz 1993; Hilliard et al. 2005).
    "noxious": ["ASHL", "ASHR", "ADLL", "ADLR"],
    # Head touch, which is the closest thing a worm has to being addressed.
    "voice": ["FLPL", "FLPR", "OLQDL", "OLQDR", "OLQVL", "OLQVR"],
}

# What the kernel reads back. Forward command interneurons are approach,
# reversal command interneurons are avoidance (Chalfie et al. 1985), and
# arousal is the whole command layer.
WORM_READOUTS = {
    "forward": (ROLE_APPROACH, ["AVBL", "AVBR", "PVCL", "PVCR"]),
    "reverse": (ROLE_AVOID, ["AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"]),
    # AIY promotes forward runs and suppresses turns (Tsalik & Hobert 2003;
    # Gray et al. 2005) -- the one interneuron pair here that food reaches
    # without passing through gap junctions this model drops.
    "seek": (ROLE_APPROACH, ["AIYL", "AIYR"]),
    "command": (ROLE_AROUSAL, ["AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER", "PVCL", "PVCR"]),
}

WORM_FILES = {
    "herm_full_edgelist.csv": "142693f17556148d7f962835b18ac6dd5af18b7467eef61815ebc1dd5474c0ca",
    "elife-95402-supp2-v1.xlsx": None,  # recorded on first fetch; see fetch()
}
WORM_URL = "https://raw.githubusercontent.com/openworm/ConnectomeToolbox/main/cect/data/"


# --- the format -----------------------------------------------------------

def pack_name(s: str, n: int = 16) -> bytes:
    b = s.encode("ascii")
    if len(b) > n:
        raise ValueError(f"name too long for the format: {s}")
    return b + b"\0" * (n - len(b))


def write(path, labels, edges, sensors, readouts, source_digest: bytes):
    """labels: list of str (<= 8 chars) or int; edges: list of (pre, post, w)
    with w a signed synapse count; sensors: {name: [ids]}; readouts:
    {name: (role, [ids])}."""
    n = len(labels)
    edges = sorted(edges)
    rows = [0] * (n + 1)
    for pre, _, _ in edges:
        rows[pre + 1] += 1
    for i in range(n):
        rows[i + 1] += rows[i]
    for _, _, w in edges:
        if not -32768 <= w <= 32767:
            raise ValueError("a weight does not fit i16")

    body = bytearray()
    body += struct.pack(f"<{n + 1}I", *rows)
    body += struct.pack(f"<{len(edges)}I", *[p for _, p, _ in edges])
    body += struct.pack(f"<{len(edges)}h", *[w for _, _, w in edges])
    while len(body) % 4:
        body += b"\0"
    for name, ids in sensors.items():
        body += pack_name(name) + struct.pack(f"<I{len(ids)}I", len(ids), *ids)
    for name, (role, ids) in readouts.items():
        body += pack_name(name) + struct.pack(f"<II{len(ids)}I", role, len(ids), *ids)
    for lab in labels:
        if isinstance(lab, int):
            body += struct.pack("<Q", lab)
        else:
            body += lab.encode("ascii")[:8].ljust(8, b"\0")

    hdr = bytearray(HEADER)
    struct.pack_into(
        "<8sIIIiiiIIIIiiII8s", hdr, 0,
        MAGIC, VERSION, n, len(edges),
        V0_NV, V_TH_NV, V0_NV,
        TAU_M, TAU_SYN, REFR, DELAY,
        W_SYN_NV, W_POI_NV,
        len(sensors), len(readouts),
        source_digest[:8],
    )
    struct.pack_into("<I", hdr, 76, DT_US)
    Path(path).write_bytes(bytes(hdr) + bytes(body))


class Brain:
    """A GLADOSB1 file, read back by walking -- never seeking -- and asserting
    the walk lands on the last byte, for v4.py's reason: a reader that seeks
    can disagree with a writer about one length and still read valid-looking
    integers everywhere after it."""

    def __init__(self, data: bytes):
        (magic, ver, n, e, v0, vth, vreset, taum, tausyn, refr, delay,
         wsyn, wpoi, ns, nr, digest) = struct.unpack_from("<8sIIIiiiIIIIiiII8s", data, 0)
        if magic != MAGIC or ver != VERSION:
            raise ValueError("not a GLADOSB1 v1 file")
        (self.dt_us,) = struct.unpack_from("<I", data, 76)
        self.n, self.e = n, e
        self.v0, self.vth, self.vreset = v0, vth, vreset
        self.tau_m, self.tau_syn, self.refr, self.delay = taum, tausyn, refr, delay
        self.w_syn, self.w_poi = wsyn, wpoi
        self.digest = digest
        p = HEADER
        self.rows = list(struct.unpack_from(f"<{n + 1}I", data, p)); p += 4 * (n + 1)
        self.col = list(struct.unpack_from(f"<{e}I", data, p)); p += 4 * e
        self.w = list(struct.unpack_from(f"<{e}h", data, p)); p += 2 * e
        p = (p + 3) & ~3
        self.sensors = {}
        for _ in range(ns):
            name = data[p:p + 16].rstrip(b"\0").decode(); p += 16
            (k,) = struct.unpack_from("<I", data, p); p += 4
            self.sensors[name] = list(struct.unpack_from(f"<{k}I", data, p)); p += 4 * k
        self.readouts = {}
        for _ in range(nr):
            name = data[p:p + 16].rstrip(b"\0").decode(); p += 16
            role, k = struct.unpack_from("<II", data, p); p += 8
            self.readouts[name] = (role, list(struct.unpack_from(f"<{k}I", data, p))); p += 4 * k
        self.labels = [data[p + 8 * i:p + 8 * i + 8] for i in range(n)]
        p += 8 * n
        if p != len(data):
            raise ValueError(f"walk ended at {p} of {len(data)} bytes")
        if self.rows[-1] != e or any(self.rows[i] > self.rows[i + 1] for i in range(n)):
            raise ValueError("row pointers are not a CSR index")
        if any(c >= n for c in self.col):
            raise ValueError("a target is out of range")


# --- the simulation ---------------------------------------------------------

def tdiv(a: int, b: int) -> int:
    """Integer division truncating toward zero, which is what Rust's `/` on
    i64 does. Python's `//` floors, and the two differ on every negative
    numerator -- which, with a resting potential of -52 mV, is most of them."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


class Rng:
    """splitmix64, the generator src/brain uses. Deterministic and seeded, so
    a Poisson train is a function of the seed and the step."""

    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)


def simulate(b: Brain, steps: int, drive: dict, seed: int = 1):
    """Run `steps` steps. `drive` maps sensor name -> rate in Hz.

    One step, in exactly this order (the kernel's `lif::step` is the same):
      1. deliver the delay ring's slot for this step into g
      2. Poisson input to driven sensors, in table order, one draw each
      3. v integrates g (or is held at reset while refractory)
      4. a neuron at or over threshold spikes: reset, refractory, and its
         row's weights enter the ring DELAY steps ahead
      5. g decays
    Returns spike counts per neuron."""
    n = b.n
    v = [b.v0] * n
    g = [0] * n
    refr = [0] * n
    slots = b.delay + 1
    ring = [[0] * n for _ in range(slots)]
    counts = [0] * n
    rng = Rng(seed)
    # p(spike in one step) = rate * dt, as a threshold on a 64-bit draw.
    driven = []
    for name, rate in drive.items():
        thr = (rate * b.dt_us * (1 << 32) // 1_000_000) << 32
        for i in b.sensors[name]:
            driven.append((i, thr))
    for t in range(steps):
        slot = ring[t % slots]
        for i in range(n):
            if slot[i]:
                g[i] += slot[i]
                slot[i] = 0
        for i, thr in driven:
            if rng.next() < thr:
                g[i] += b.w_poi
        fired = []
        for i in range(n):
            if refr[i] > 0:
                refr[i] -= 1
                v[i] = b.vreset
            else:
                v[i] += tdiv(b.v0 - v[i] + g[i], b.tau_m)
                if v[i] >= b.vth:
                    v[i] = b.vreset
                    refr[i] = b.refr
                    counts[i] += 1
                    fired.append(i)
        if fired:
            out = ring[(t + b.delay) % slots]
            for i in fired:
                for k in range(b.rows[i], b.rows[i + 1]):
                    out[b.col[k]] += b.w[k] * b.w_syn
        for i in range(n):
            g[i] -= tdiv(g[i], b.tau_syn)
    return counts


def oracle_digest(counts) -> int:
    """FNV-1a over every neuron's spike count as a little-endian u32, so one
    number says whether two runs agreed about every neuron and not merely
    about the total."""
    h = 0xCBF29CE484222325
    for c in counts:
        for byte in struct.pack("<I", c):
            h ^= byte
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def rates(b: Brain, counts, steps):
    secs = steps * b.dt_us / 1e6
    out = {}
    for name, (role, ids) in b.readouts.items():
        out[name] = sum(counts[i] for i in ids) / max(1, len(ids)) / secs
    return out


# --- the worm -----------------------------------------------------------------

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch_worm(dst: Path):
    import urllib.request
    dst.mkdir(parents=True, exist_ok=True)
    for name, want in WORM_FILES.items():
        p = dst / name
        if not p.exists():
            urllib.request.urlretrieve(WORM_URL + name.replace(" ", "%20"), p)
        got = sha256(p)
        if want and got != want:
            raise SystemExit(f"{name}: sha256 {got}, pinned {want}")
        print(f"  {name}  {got}")


def worm_transmitters(xlsx: Path):
    """Neuron -> 'ach' | 'glu' | 'gaba' | None, from Wang et al. 2024's
    hermaphrodite table, read from its summary column ("Neurotransmitter(s)").

    The per-gene reporter columns read back empty -- the calls in them are
    cell formatting, not values -- so the summary is the column a program can
    actually read, and it is the authors' own synthesis of those reporters.
    GABA wins over an excitatory call, because a GABAergic neuron that also
    packages a second transmitter is still the inhibition in its circuit. A
    neuron whose summary names only a monoamine (or nothing) is None."""
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    head = next(i for i, r in enumerate(rows) if r and len(r) > 2 and r[1] == "Class")
    col_n = rows[head].index("Neuron")
    col_t = next(j for j, c in enumerate(rows[head]) if c and str(c).startswith("Neurotransmitter"))
    out = {}
    for r in rows[head + 1:]:
        if len(r) <= col_t or not r[col_n]:
            continue
        t = str(r[col_t] or "")
        name = worm_name(str(r[col_n]))
        if "GABA" in t:
            out[name] = "gaba"
        elif "ACh" in t:
            out[name] = "ach"
        elif "Glu" in t:
            out[name] = "glu"
        else:
            out[name] = None
    return out


def worm_name(s: str) -> str:
    """One spelling per neuron. The edge list zero-pads the motor neurons
    (`AS01`, `VD09`) and the transmitter table does not (`AS1`), and without
    this the whole ventral cord silently fell out of the build -- 233 neurons
    of 302, with nothing to say which were missing."""
    import re
    return re.sub(r"^([A-Z]+)0(\d)$", r"\1\2", s.strip())


def build_worm(src: Path, dst: Path):
    tx = worm_transmitters(src / "elife-95402-supp2-v1.xlsx")
    edges_raw = {}
    names = set()
    with open(src / "herm_full_edgelist.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["Type"].strip() != "chemical":
                continue
            a, b = worm_name(row["Source"]), worm_name(row["Target"])
            # Neurons only: the edge list also runs onto muscles and other
            # cells, which have no transmitter row and no dynamics here.
            if a not in tx or b not in tx:
                continue
            names.update((a, b))
            edges_raw[(a, b)] = edges_raw.get((a, b), 0) + int(row["Weight"])
    labels = sorted(names)
    absent = sorted(set(tx) - names)
    if absent:
        print(f"  {len(absent)} neuron(s) in the transmitter table with no chemical synapse to another neuron: {' '.join(absent)}")
    idx = {nm: i for i, nm in enumerate(labels)}
    unknown = [nm for nm in labels if tx.get(nm) is None]
    edges = []
    for (a, b), w in edges_raw.items():
        sign = -1 if tx.get(a) == "gaba" else 1
        edges.append((idx[a], idx[b], sign * w))
    sensors = {k: [idx[x] for x in v if x in idx] for k, v in WORM_SENSORS.items()}
    readouts = {k: (role, [idx[x] for x in v if x in idx]) for k, (role, v) in WORM_READOUTS.items()}
    for k, ids in sensors.items():
        if not ids:
            raise SystemExit(f"sensor {k} resolves to no neuron in this dataset")
    digest = hashlib.sha256()
    for name in sorted(WORM_FILES):
        digest.update((src / name).read_bytes())
    write(dst, labels, edges, sensors, readouts, digest.digest())
    inh = sum(1 for _, _, w in edges if w < 0)
    print(f"  {len(labels)} neurons, {len(edges)} chemical connections "
          f"({sum(abs(w) for _, _, w in edges)} synapses), {inh} inhibitory")
    print(f"  {len(unknown)} neuron(s) with no transmitter call, counted excitatory")
    print(f"  gap junctions dropped; sign per presynaptic neuron (GABA -, ACh/Glu +)")
    print(f"  wrote {dst} ({dst.stat().st_size} bytes)")


# --- selftest -------------------------------------------------------------------

def selftest() -> bool:
    import tempfile
    ok = True

    def claim(what, good):
        nonlocal ok
        print(("ok  " if good else "FAIL"), what)
        ok &= bool(good)

    claim("truncating division agrees with Rust on negatives", tdiv(-7, 2) == -3 and tdiv(7, -2) == -3 and tdiv(-6, 3) == -2)
    r = Rng(1)
    claim("splitmix64 matches its published first output for seed 1", r.next() == 0x910A2DEC89025CC1)

    # Three neurons: 0 drives 1 (excitatory, 40 synapses), 1 inhibits 2.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.bin"
        write(p, ["A", "B", "C"], [(0, 1, 40), (1, 2, -40)], {"in": [0], "quiet": [2]},
              {"b": (ROLE_APPROACH, [1]), "c": (ROLE_AVOID, [2])}, b"\0" * 8)
        b = Brain(p.read_bytes())
        claim("a written file reads back by walking to its last byte", b.n == 3 and b.e == 2)
        claim("its sensor and readout tables survive", b.sensors == {"in": [0], "quiet": [2]} and b.readouts["c"] == (ROLE_AVOID, [2]))
        silent = simulate(b, 2000, {})
        claim("with no input nothing fires -- no basal activity, as in Shiu", sum(silent) == 0)
        driven = simulate(b, 2000, {"in": 150})
        claim("a driven sensor fires", driven[0] > 0)
        claim("and drives its excitatory target", driven[1] > 0)
        claim("an inhibited neuron with nothing else stays silent", driven[2] == 0)
        again = simulate(b, 2000, {"in": 150})
        claim("the same seed gives the same spikes", again == driven)
        claim("a different seed gives different ones", simulate(b, 2000, {"in": 150}, seed=2) != driven)
        # One spike, by hand: at 150 Hz input weight W_POI pushes v from rest
        # by (W_POI) / TAU_M in the step it lands -- 343,750 nV.
        b2v = b.v0 + tdiv(b.v0 - b.v0 + b.w_poi, b.tau_m)
        claim("one input spike moves v by W_POI / TAU_M in its own step", b2v - b.v0 == 343_750)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    f = sub.add_parser("fetch"); f.add_argument("rung"); f.add_argument("dst")
    bld = sub.add_parser("build"); bld.add_argument("rung"); bld.add_argument("src"); bld.add_argument("dst")
    s = sub.add_parser("sim")
    s.add_argument("file"); s.add_argument("--poke", action="append", default=[])
    s.add_argument("--rate", type=int, default=150); s.add_argument("--ms", type=int, default=500)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--emit", action="store_true", help="print the spike counts diag brain asserts, as Rust")
    a = ap.parse_args()
    if a.selftest:
        return 0 if selftest() else 1
    if a.cmd == "fetch":
        if a.rung != "worm":
            raise SystemExit("only the worm rung is built by this version")
        fetch_worm(Path(a.dst)); return 0
    if a.cmd == "build":
        if a.rung != "worm":
            raise SystemExit("only the worm rung is built by this version")
        build_worm(Path(a.src), Path(a.dst)); return 0
    if a.cmd == "sim":
        b = Brain(Path(a.file).read_bytes())
        steps = a.ms * 1000 // b.dt_us
        counts = simulate(b, steps, {k: a.rate for k in a.poke}, a.seed)
        if a.emit:
            print(f"// connectome.py sim --poke {' --poke '.join(a.poke)} --rate {a.rate} --ms {a.ms} --seed {a.seed}")
            print(f"pub const ORACLE_STEPS: u64 = {steps};")
            print(f"pub const ORACLE_TOTAL: u64 = {sum(counts)};")
            print(f"pub const ORACLE_DIGEST: u64 = 0x{oracle_digest(counts):016x};")
            return 0
        print(f"  {b.n} neurons, {b.e} connections, {steps} steps, {sum(counts)} spikes")
        for name, hz in rates(b, counts, steps).items():
            print(f"  {name:10} {hz:8.1f} Hz")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

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
    fly    adult Drosophila, FlyWire v783 (Dorkenwald et al. 2024, CC BY 4.0):
           138,639 neurons, 15.1M connections, every synapse Shiu's model
           keeps, signed as Shiu signed them (Eckstein & Bates 2024's
           transmitter predictions; acetylcholine excites, GABA and
           glutamate inhibit). Read from Shiu's own Connectivity_783.parquet
           so the canary below is his experiment, not an imitation of it.
           Cell types from Schlegel et al. 2024's annotations.

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
# Neuromodulator populations: the animal's own aminergic cells, whose
# activity the kernel turns into global gain and excitability.
ROLE_DOPAMINE, ROLE_OCTOPAMINE, ROLE_SEROTONIN = 3, 4, 5

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
    # The worm's aminergic neurons (Chase & Koelle 2007): dopamine in the
    # mechanosensory CEP/ADE/PDE, serotonin in NSM/ADF/HSN, and octopamine
    # (RIC) with tyramine (RIM) standing in for the fly's octopamine -- the
    # invertebrate noradrenaline, arousal.
    "dopamine": (ROLE_DOPAMINE, ["CEPDL", "CEPDR", "CEPVL", "CEPVR", "ADEL", "ADER", "PDEL", "PDER"]),
    "octopamine": (ROLE_OCTOPAMINE, ["RICL", "RICR", "RIML", "RIMR"]),
    "serotonin": (ROLE_SEROTONIN, ["NSML", "NSMR", "ADFL", "ADFR", "HSNL", "HSNR"]),
}

WORM_FILES = {
    "herm_full_edgelist.csv": "142693f17556148d7f962835b18ac6dd5af18b7467eef61815ebc1dd5474c0ca",
    "elife-95402-supp2-v1.xlsx": None,  # recorded on first fetch; see fetch()
}
WORM_URL = "https://raw.githubusercontent.com/openworm/ConnectomeToolbox/main/cect/data/"

FLY_FILES = {
    "Connectivity_783.parquet": "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
    "Completeness_783.csv": "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
    "annotations.tsv": "b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd",
}
FLY_URLS = {
    "Connectivity_783.parquet": "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet",
    "Completeness_783.csv": "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Completeness_783.csv",
    "annotations.tsv": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
}

# Shiu et al. 2024's own neuron lists (figures.ipynb), so the canary is his
# experiment: labellar sugar GRNs, right hemisphere, drive MN9.
FLY_SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
    720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
    720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
FLY_BITTER = [
    720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093, 720575940626287336,
    720575940618600651, 720575940627692048, 720575940630195909, 720575940646212996, 720575940610483162,
    720575940645743412, 720575940627578156, 720575940622298631, 720575940621008895, 720575940629146711,
    720575940610259370, 720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118,
    720575940604027168,
]
FLY_MN9 = [720575940660219265]  # Shiu's v783 example; CB0701 in the annotations


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

# Short-term synaptic depression (Tsodyks & Markram 1997), added to Shiu's
# model because without it the whole fly brain has a self-sustaining state:
# a heat or odour input ignites ~10,000 neurons that then fire forever
# (design/attractor.md). Per presynaptic neuron, `xr` is the fraction of
# resources available in parts per million. A spike delivers its weights
# scaled by xr, then uses STD_U of what is left; every step xr recovers by
# (1e6 - xr) / STD_TAU. At rest xr is 1, so an isolated spike has exactly its
# published weight. Sensor neurons are exempt: their spikes come from the
# periphery, and the canary (sugar drives MN9) depends on them. Chosen on
# the host against two checks: the heat state ends within 100 ms of its
# input, and sugar still drives MN9 (50 Hz, from 70 without).
STD_U = 100_000
STD_TAU_MS = 1000
PPM = 1_000_000


def std_tau_steps(b) -> int:
    return STD_TAU_MS * 1000 // b.dt_us


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
      2. Poisson input to driven sensors, in table order, one draw each --
         a kick to v, not to g (`PoissonInput(target_var='v')`)
      3. a refractory neuron is held at reset and its g does not decay --
         Shiu's model.py marks both equations `(unless refractory)`; any
         other neuron integrates v, then g decays
      4. a neuron strictly over threshold (`v > v_th`, as model.py has it)
         spikes: v to reset *and g to zero* (`eq_rst`), refractory unless it
         is being driven (model.py sets `rfc = 0` for Poisson targets), and
         its row's weights enter the ring DELAY steps ahead

    Read off model.py line by line, after the first version got three of
    these wrong and ran away: g kept across a spike meant every neuron that
    fired once fired again the moment its refractory period ended, and 100
    ms of sugar became 14,900 neurons at 200 Hz where Shiu's notebook reports
    about 13,000 spikes in a whole second.
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
    nore = {i for i, _ in driven}
    xr = [PPM] * n
    exempt = set(i for ids in b.sensors.values() for i in ids)
    tau_rec = std_tau_steps(b)
    for t in range(steps):
        slot = ring[t % slots]
        for i in range(n):
            if slot[i]:
                g[i] += slot[i]
                slot[i] = 0
        for i, thr in driven:
            if rng.next() < thr:
                v[i] += b.w_poi
        fired = []
        for i in range(n):
            if refr[i] > 0:
                refr[i] -= 1
                v[i] = b.vreset
            else:
                v[i] += tdiv(b.v0 - v[i] + g[i], b.tau_m)
                g[i] -= tdiv(g[i], b.tau_syn)
                if v[i] > b.vth:
                    v[i] = b.vreset
                    g[i] = 0
                    refr[i] = 0 if i in nore else b.refr
                    counts[i] += 1
                    fired.append(i)
        if fired:
            out = ring[(t + b.delay) % slots]
            for i in fired:
                for k in range(b.rows[i], b.rows[i + 1]):
                    out[b.col[k]] += tdiv(b.w[k] * b.w_syn * xr[i], PPM)
                if i not in exempt:
                    xr[i] -= xr[i] * STD_U // PPM
        for i in range(n):
            xr[i] += (PPM - xr[i]) // tau_rec
    return counts


def simulate_np(b: Brain, steps: int, drive: dict, seed: int = 1):
    """`simulate`, vectorised, for an animal too large to loop over in
    Python. Same arithmetic, same order of draws; the selftest requires the
    two to agree exactly on a graph small enough for both."""
    import numpy as np
    n = b.n
    rows = np.asarray(b.rows, dtype=np.int64)
    col = np.asarray(b.col, dtype=np.int64)
    w = np.asarray(b.w, dtype=np.int64) * b.w_syn
    v = np.full(n, b.v0, dtype=np.int64)
    g = np.zeros(n, dtype=np.int64)
    refr = np.zeros(n, dtype=np.int64)
    slots = b.delay + 1
    ring = np.zeros((slots, n), dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    rng = Rng(seed)
    driven = []
    for name, rate in drive.items():
        thr = (rate * b.dt_us * (1 << 32) // 1_000_000) << 32
        for i in b.sensors[name]:
            driven.append((i, thr))

    nore = np.zeros(n, dtype=bool)
    for i, _ in driven:
        nore[i] = True

    def tdiv_np(a, d):
        return np.sign(a) * (np.abs(a) // d)

    xr = np.full(n, PPM, dtype=np.int64)
    exempt = np.zeros(n, dtype=bool)
    for ids in b.sensors.values():
        exempt[ids] = True
    tau_rec = std_tau_steps(b)
    for t in range(steps):
        s = t % slots
        g += ring[s]
        ring[s] = 0
        for i, thr in driven:
            if rng.next() < thr:
                v[i] += b.w_poi
        r = refr > 0
        refr[r] -= 1
        v[r] = b.vreset
        live = ~r
        v[live] += tdiv_np(b.v0 - v[live] + g[live], b.tau_m)
        g[live] -= tdiv_np(g[live], b.tau_syn)
        fired = np.nonzero(live & (v > b.vth))[0]
        if fired.size:
            v[fired] = b.vreset
            g[fired] = 0
            refr[fired] = np.where(nore[fired], 0, b.refr)
            counts[fired] += 1
            out = ring[(t + b.delay) % slots]
            for i in fired:
                a, z = rows[i], rows[i + 1]
                np.add.at(out, col[a:z], tdiv_np(w[a:z] * xr[i], PPM))
            dep = fired[~exempt[fired]]
            xr[dep] -= xr[dep] * STD_U // PPM
        xr += (PPM - xr) // tau_rec
    return [int(c) for c in counts]


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


def fetch_fly(dst: Path):
    import urllib.request
    dst.mkdir(parents=True, exist_ok=True)
    for name, want in FLY_FILES.items():
        p = dst / name
        if not p.exists():
            urllib.request.urlretrieve(FLY_URLS[name], p)
        got = sha256(p)
        if got != want:
            raise SystemExit(f"{name}: sha256 {got}, pinned {want}")
        print(f"  {name}  {got}")


def build_fly(src: Path, dst: Path):
    import pyarrow.parquet as pq
    for name, want in FLY_FILES.items():
        if sha256(src / name) != want:
            raise SystemExit(f"{name} does not match its pinned digest")
    ids = [int(l.split(",")[0]) for l in (src / "Completeness_783.csv").read_text().splitlines()[1:]]
    idx = {r: i for i, r in enumerate(ids)}
    t = pq.read_table(src / "Connectivity_783.parquet",
                      columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    pre = t.column(0).to_numpy()
    post = t.column(1).to_numpy()
    w = t.column(2).to_numpy()
    edges = list(zip(pre.tolist(), post.tolist(), w.tolist()))

    ann = list(csv.DictReader(open(src / "annotations.tsv"), delimiter="\t"))
    def where(pred):
        return sorted(idx[int(r["root_id"])] for r in ann if pred(r) and int(r["root_id"]) in idx)
    def ids_of(roots, need_all=False):
        # Shiu's lists were written against v630 and a root id changes when
        # its neuron is re-proofread, so a few may not exist in v783. Dropped
        # and said, never substituted -- guessing which v783 neuron a stale
        # id became is exactly the silent mapping this file refuses to make.
        missing = [r for r in roots if r not in idx]
        if missing:
            print(f"  {len(missing)} of {len(roots)} named neuron(s) absent from v783, dropped: {missing}")
        if need_all and missing or len(missing) * 10 > len(roots):
            raise SystemExit("too many named neurons are missing to call this the same experiment")
        return [idx[r] for r in roots if r in idx]

    sensors = {
        # What a fly hears with: the auditory Johnston's organ neurons.
        "voice": where(lambda r: r["cell_class"] == "mechanosensory" and r["cell_sub_class"] == "auditory"),
        # Head bristle mechanosensory neurons.
        "touch": where(lambda r: r["cell_class"] == "mechanosensory" and r["cell_sub_class"] == "head bristle"),
        # Labellar sugar GRNs -- Shiu's list.
        "food": ids_of(FLY_SUGAR),
        # Labellar bitter GRNs -- Shiu's list.
        "noxious": ids_of(FLY_BITTER),
        # Antennal hot cells (TRN_VP2).
        "thermal": where(lambda r: r["cell_type"] == "TRN_VP2"),
    }
    readouts = {
        # Proboscis extension, the feeding motor neuron Shiu's model drives
        # from sugar -- the canary.
        # MN9 is annotated as cell type CB0701, one per side. Shiu's v630
        # list named the other side by an id v783 no longer has; selecting
        # by type finds both, and the check below insists the one his v783
        # example uses is among them, so the canary is still his neuron.
        "feeding": (ROLE_APPROACH, where(lambda r: r["cell_type"] == "CB0701")),
        # The giant fibre and its escape partners.
        "escape": (ROLE_AVOID, where(lambda r: r["cell_type"] in ("DNp01", "DNp02", "DNp11"))),
        # The persistent internal state circuit (Deutsch et al. 2020).
        "pC1-aIPg": (ROLE_AROUSAL, where(lambda r: r["cell_type"] in ("pC1a", "pC1b", "pC1c", "pC1d", "pC1e")
                                         or r["cell_type"].startswith("aIPg") or r["hemibrain_type"].startswith("aIPg"))),
        # Everything that leaves the brain for the body.
        "descending": (ROLE_AROUSAL, where(lambda r: r["super_class"] == "descending")),
        # Aminergic cells by identified type, not by transmitter prediction:
        # the predictions call Kenyon cells dopaminergic, a known artefact.
        "dopamine": (ROLE_DOPAMINE, where(lambda r: any(t.startswith(p) for t in (r["cell_type"], r["hemibrain_type"])
                                                      for p in ("PAM", "PPL1", "PPL2", "PPM")))),
        "octopamine": (ROLE_OCTOPAMINE, where(lambda r: r["cell_type"].startswith("OA-") or r["hemibrain_type"].startswith("OA-"))),
        "serotonin": (ROLE_SEROTONIN, where(lambda r: any(t.startswith("5-HT") or t == "CSD"
                                                      for t in (r["cell_type"], r["hemibrain_type"])))),
    }
    if idx[FLY_MN9[0]] not in readouts["feeding"][1]:
        raise SystemExit("Shiu's MN9 is not among the CB0701 neurons -- the canary would not be his experiment")
    for k, v in list(sensors.items()) + [(k, v[1]) for k, v in readouts.items()]:
        if not v:
            raise SystemExit(f"{k} resolves to no neuron")
    digest = hashlib.sha256()
    for name in sorted(FLY_FILES):
        digest.update(bytes.fromhex(FLY_FILES[name]))
    write(dst, ids, edges, sensors, readouts, digest.digest())
    inh = int((w < 0).sum())
    print(f"  {len(ids)} neurons, {len(edges)} connections ({int(abs(w).sum())} synapses), {inh} inhibitory")
    for k, v in sensors.items():
        print(f"  sense {k:8} {len(v)} neurons")
    for k, (_, v) in readouts.items():
        print(f"  readout {k:10} {len(v)} neurons")
    print(f"  wrote {dst} ({dst.stat().st_size} bytes)")


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


# --- check ----------------------------------------------------------------------

# What the kernel asks a staged file for. Senses by name (`brain::Sense`);
# readouts by role, since affect sums every readout of a role and never asks
# a name. A file lacking one still loads and the kernel goes without, so a
# release built from it would ship a brain with valence or arousal stuck at
# neutral and nothing saying why.
SENSES_WANTED = ["voice", "touch", "food", "noxious", "thermal"]
ROLES_WANTED = {ROLE_APPROACH: "approach", ROLE_AVOID: "avoid", ROLE_AROUSAL: "arousal"}


def check(path: Path) -> int:
    """Read a built file by walking it, and require the populations the kernel
    reads. Standard library only, so a release job can run it with no numpy:
    the digest says these are the bytes that were recorded, and this says the
    bytes recorded were a brain."""
    try:
        b = Brain(path.read_bytes())
    except ValueError as why:
        print(f"FAIL {path}: {why}")
        return 1
    print(f"  {path}: {b.n} neurons, {b.e} connections, dt {b.dt_us} us")
    have = {role for role, cells in b.readouts.values() if cells}
    missing = [s for s in SENSES_WANTED if not b.sensors.get(s)] + \
              [f"a readout for {name}" for role, name in ROLES_WANTED.items() if role not in have]
    for name, cells in b.sensors.items():
        print(f"  sense   {name:12} {len(cells)}")
    for name, (_, cells) in b.readouts.items():
        print(f"  readout {name:12} {len(cells)}")
    if missing:
        print(f"FAIL empty or absent: {', '.join(missing)}")
        return 1
    print("ok   walks to its last byte; every sense, and approach, avoid and arousal readouts, populated")
    return 0


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
        claim("the vectorised oracle agrees with the loop exactly", simulate_np(b, 2000, {"in": 150}) == driven)
        # One input spike, by hand: a 68.75 mV kick to v from rest, then one
        # Euler step back toward rest, is still far over threshold -- Shiu's
        # f_poi is chosen to be "sufficient to cause spiking".
        kicked = b.v0 + b.w_poi
        kicked += tdiv(b.v0 - kicked, b.tau_m)
        claim("one input spike alone takes a resting neuron over threshold", kicked > b.vth)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    f = sub.add_parser("fetch"); f.add_argument("rung"); f.add_argument("dst")
    bld = sub.add_parser("build"); bld.add_argument("rung"); bld.add_argument("src"); bld.add_argument("dst")
    c = sub.add_parser("check", help="read a built file back and name what the kernel will look for")
    c.add_argument("file")
    s = sub.add_parser("sim")
    s.add_argument("file"); s.add_argument("--poke", action="append", default=[])
    s.add_argument("--rate", type=int, default=150); s.add_argument("--ms", type=int, default=500)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--emit", action="store_true", help="print the spike counts diag brain asserts, as Rust")
    a = ap.parse_args()
    if a.selftest:
        return 0 if selftest() else 1
    if a.cmd == "fetch":
        {"worm": fetch_worm, "fly": fetch_fly}[a.rung](Path(a.dst)); return 0
    if a.cmd == "build":
        {"worm": build_worm, "fly": build_fly}[a.rung](Path(a.src), Path(a.dst)); return 0
    if a.cmd == "check":
        return check(Path(a.file))
    if a.cmd == "sim":
        b = Brain(Path(a.file).read_bytes())
        steps = a.ms * 1000 // b.dt_us
        sim = simulate if b.n < 2000 else simulate_np
        counts = sim(b, steps, {k: a.rate for k in a.poke}, a.seed)
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

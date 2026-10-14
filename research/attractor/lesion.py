"""Causal test: the heat poke with one pathway removed.
  dan_out : every synapse leaving a dopaminergic neuron (cell_class DAN) zeroed
  kc_kc   : every Kenyon cell -> Kenyon cell synapse zeroed
  dan_kc  : only DAN -> Kenyon cell synapses zeroed
  kc_out  : every synapse leaving a Kenyon cell zeroed
Counts neurons still firing 500-1000 ms after a 500 ms poke ends."""
import sys, csv, time
sys.path.insert(0, "/home/lummy/Projects/GLaDOS/tools")
import numpy as np, connectome as cx, pci
from pathlib import Path
b = cx.Brain(Path("/home/lummy/Projects/GLaDOS/out/fly.bin").read_bytes())
ids = [int(l.split(",")[0]) for l in open("../fly-src/Completeness_783.csv").read().splitlines()[1:]]
cls = {}
with open("../fly-src/annotations.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        cls[int(r["root_id"])] = r["cell_class"]
kc = np.array([cls.get(r) == "Kenyon_Cell" for r in ids])
dan = np.array([cls.get(r) == "DAN" for r in ids])
sup = {}
with open("../fly-src/annotations.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        sup[int(r["root_id"])] = r["super_class"]
sensory = np.array([sup.get(r) == "sensory" for r in ids])
nts = {}
with open("../fly-src/annotations.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        nts[int(r["root_id"])] = (r["top_nt"], r["known_nt"])
def is_gaba(r):
    t, k = nts.get(r, ("", ""))
    return k.startswith("gaba") or (not k and t == "gaba")
gaba = np.array([is_gaba(r) for r in ids])
g = pci.Graph.of(b)
src = np.repeat(np.arange(b.n), np.diff(g.rows))
v = sys.argv[1]
def matched(seed):
    """Random non-sensor neurons, one per core-30 neuron, each with an
    out-degree within 10% of the core neuron it stands for."""
    core = np.load("core-30.npy")
    deg = np.diff(g.rows)
    sens = np.zeros(b.n, bool)
    for ids_ in b.sensors.values():
        sens[ids_] = True
    rng = np.random.default_rng(seed)
    pick = np.zeros(b.n, bool)
    order = np.argsort(deg)
    for i in np.nonzero(core)[0]:
        lo, hi = deg[i] * 0.9, deg[i] * 1.1
        cand = np.nonzero((deg >= lo) & (deg <= hi) & ~core & ~sens & ~pick)[0]
        pick[rng.choice(cand)] = True
    return pick
if v.startswith("rand"):
    sel = matched(int(v[4:]))
    v_cut = sel[src]
if v == "resign_gaba":
    flip = gaba[src] & (g.w > 0)
    g.w = np.where(flip, -g.w, g.w)
    cut = flip
else:
  cut = {"dan_out": dan[src], "kc_kc": kc[src] & kc[g.col], "dan_kc": dan[src] & kc[g.col], "kc_out": kc[src], "onto_sensory": sensory[g.col], "core30": np.load("core-30.npy")[src], "core89": np.load("core-89.npy")[src], **({v: v_cut} if v.startswith("rand") else {}), "none": np.zeros(len(src), bool)}[v]
if v != "resign_gaba":
    g.w = np.where(cut, 0, g.w)
spm = 1000 // b.dt_us
t0 = time.time()
r, _ = pci.simulate(b, g, 1500 * spm, [("thermal", 150, 0, 500 * spm)], list(range(b.n)), 500 * spm, 7)
on, late = r[0] > 0, r[2] > 0
print(f"{v:8} cut {cut.sum():8} synapse rows | during poke {on.sum():6} | late {late.sum():6} ({int(r[2].sum())} spikes) | KCs late {(late&kc).sum()} DANs late {(late&dan).sum()} | {time.time()-t0:.0f} s", flush=True)

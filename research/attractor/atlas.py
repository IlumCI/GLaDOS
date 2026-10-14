"""Every annotated cell type with any synapse onto the core, through the
reduced core model. Splits work by index modulo argv[2] of argv[1]."""
import sys, csv, collections, json, numpy as np
sys.path.insert(0, ".")
import reduced as rd
part, parts = int(sys.argv[1]), int(sys.argv[2])
ids = [int(l.split(",")[0]) for l in open("../fly-src/Completeness_783.csv").read().splitlines()[1:]]
types = collections.defaultdict(list)
with open("../fly-src/annotations.tsv") as f:
    ann = {int(r["root_id"]): r["cell_type"] for r in csv.DictReader(f, delimiter="\t")}
for i, r in enumerate(ids):
    t = ann.get(r)
    if t: types[t].append(i)
src = np.repeat(np.arange(rd.b.n), np.diff(rd.rows))
hit = np.zeros(rd.b.n, bool); hit[src[rd.core[rd.col]]] = True   # neurons with a synapse onto the core
names = sorted(t for t, m in types.items() if hit[m].any() and not rd.core[m].all())
out = {}
for k, t in enumerate(names):
    if k % parts != part: continue
    out[t] = {"n": len(types[t]), "core_late": rd.ignites(types[t])}
json.dump(out, open(f"atlas-{part}.json", "w"))
print(f"part {part}: {len(out)} types, {sum(v['core_late']>100 for v in out.values())} ignite", flush=True)

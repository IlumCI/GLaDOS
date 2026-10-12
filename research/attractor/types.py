"""Cell types of the persistent heat state and of the predicted core."""
import csv, collections, numpy as np
ids = [int(l.split(",")[0]) for l in open("../fly-src/Completeness_783.csv").read().splitlines()[1:]]
ann = {}
with open("../fly-src/annotations.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        ann[int(r["root_id"])] = r
def key(i, field):
    r = ann.get(ids[i])
    return (r[field] or "?") if r else "unannotated"
late = np.load("late-thermal.npy") > 0
core = np.load("core-89.npy")
tight = np.load("core-30.npy")
n = len(ids)
for title, mask in (("persistent heat state", late), ("predicted core @89 Hz", core), ("tightest core @30 Hz", tight)):
    print(f"\n== {title}: {mask.sum()} neurons")
    for field in ("super_class", "cell_class", "top_nt"):
        c = collections.Counter(key(i, field) for i in np.nonzero(mask)[0])
        allc = collections.Counter(key(i, field) for i in range(n))
        top = c.most_common(8)
        print(f"  {field}: " + ", ".join(f"{k} {v} (x{v/mask.sum()/(allc[k]/n):.1f})" for k, v in top))
    c = collections.Counter(key(i, "cell_type") for i in np.nonzero(mask)[0])
    print("  cell_type: " + ", ".join(f"{k} {v}" for k, v in c.most_common(15)))

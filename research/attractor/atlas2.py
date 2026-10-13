import sys, json, csv, collections
sys.path.insert(0, ".")
import reduced2 as r2
part, parts = int(sys.argv[1]), int(sys.argv[2])
atlas = json.load(open("atlas.json"))
ids = [int(l.split(",")[0]) for l in open("../fly-src/Completeness_783.csv").read().splitlines()[1:]]
ann = {int(r["root_id"]): r["cell_type"] for r in csv.DictReader(open("../fly-src/annotations.tsv"), delimiter="\t")}
types = collections.defaultdict(list)
for i, r in enumerate(ids):
    if ann.get(r): types[ann[r]].append(i)
yes = sorted(t for t, v in atlas.items() if v["core_late"] > 100)
out = {}
for k, t in enumerate(yes):
    if k % parts != part: continue
    out[t] = r2.ignites(types[t])[0]
json.dump(out, open(f"atlas2-{part}.json", "w"))

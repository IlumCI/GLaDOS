import sys, json, time
sys.path.insert(0, "/home/lummy/Projects/GLaDOS/tools")
import numpy as np, connectome as cx, pci
from pathlib import Path
b = cx.Brain(Path("/home/lummy/Projects/GLaDOS/out/fly.bin").read_bytes())
t = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
b.sensors[t] = json.load(open("receptor-groups.json"))[t]
spm = 1000 // b.dt_us
t0 = time.time()
r, _ = pci.simulate(b, pci.Graph.of(b), 1000 * spm, [(t, 150, 0, 500 * spm)], list(range(b.n)), 500 * spm, seed)
print(f"{t:14} seed {seed} during {int((r[0]>0).sum()):6}  late {int((r[1]>0).sum()):6} ({int(r[1].sum())} spikes)  {time.time()-t0:.0f} s", flush=True)

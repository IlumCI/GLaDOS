"""Exploratory: every neuron, 20 ms bins, over a 500 ms poke and 500 ms after."""
import sys, time
sys.path.insert(0, "/home/lummy/Projects/GLaDOS/tools")
import numpy as np, connectome as cx, pci
from pathlib import Path
b = cx.Brain(Path("/home/lummy/Projects/GLaDOS/out/fly.bin").read_bytes())
spm = 1000 // b.dt_us
s = sys.argv[1]
t0 = time.time()
r, _ = pci.simulate(b, pci.Graph.of(b), 1000 * spm, [(s, 150, 0, 500 * spm)], list(range(b.n)), 20 * spm, 7)
np.save(f"dyn-{s}.npy", r.astype(np.uint16))
print(f"{s}: {time.time()-t0:.0f} s", flush=True)

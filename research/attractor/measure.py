"""The persistent set: which fly neurons still fire after a stimulus ends.
Host oracle arithmetic (pci.simulate), Shiu's constants, no modulation.
Saves per-neuron spike counts over the last 500 ms of a 1.5 s run with a
500 ms poke at 150 Hz, for each sense given on the command line."""
import sys, time
sys.path.insert(0, "/home/lummy/Projects/GLaDOS/tools")
import numpy as np, connectome as cx, pci
from pathlib import Path
b = cx.Brain(Path("/home/lummy/Projects/GLaDOS/out/fly.bin").read_bytes())
spm = 1000 // b.dt_us
for sense in sys.argv[1:]:
    t0 = time.time()
    r, _ = pci.simulate(b, pci.Graph.of(b), 1500 * spm, [(sense, 150, 0, 500 * spm)], list(range(b.n)), 500 * spm, 7)
    late = r[2]  # 1000-1500 ms: half a second after the poke ended
    np.save(f"/home/lummy/Projects/GLaDOS/out/attractor/late-{sense}.npy", late)
    print(f"{sense}: {int((late>0).sum())} neurons firing late, {int(late.sum())} spikes, {time.time()-t0:.0f} s", flush=True)

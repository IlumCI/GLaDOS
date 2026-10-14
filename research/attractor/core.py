"""The self-sustaining core, predicted from the wiring alone.

Mean field for Shiu's LIF: v relaxes to rest + g; each presynaptic spike adds
w * w_syn to g, which decays with tau_syn, so a source firing steadily at r
holds g at w * w_syn * r * tau_syn. A neuron fires on its own only if that
sum, over the set it belongs to, clears vth - v0. Prune every neuron that
does not clear it, repeat until nothing changes: what survives is the
largest set that can hold itself up at rate r. Inhibition (negative w)
counts against, from inside the set."""
import sys
sys.path.insert(0, "/home/lummy/Projects/GLaDOS/tools")
import numpy as np, connectome as cx
from pathlib import Path
from scipy.sparse import csr_matrix
b = cx.Brain(Path("/home/lummy/Projects/GLaDOS/out/fly.bin").read_bytes())
rows = np.asarray(b.rows); col = np.asarray(b.col); w = np.asarray(b.w, dtype=np.float64)
src = np.repeat(np.arange(b.n), np.diff(rows))
# W[target, source] in nV of steady g per Hz of source rate.
tau_syn_s = b.tau_syn * b.dt_us * 1e-6  # tau_syn is in steps
W = csr_matrix((w * b.w_syn * tau_syn_s, (col, src)), shape=(b.n, b.n))
theta = b.vth - b.v0
print(f"n {b.n}, edges {len(w)}, tau_syn {tau_syn_s*1000:.1f} ms, threshold gap {theta/1e6:.1f} mV")
for r in [float(x) for x in sys.argv[1:]]:
    alive = np.ones(b.n, dtype=bool)
    sensors = np.zeros(b.n, dtype=bool)
    for ids in b.sensors.values():
        sensors[ids] = True
    alive &= ~sensors          # sensors are driven from outside, never self-sustained
    while True:
        drive = W @ (alive * r)
        keep = alive & (drive > theta)
        if keep.sum() == alive.sum():
            break
        alive = keep
    np.save(f"core-{int(r)}.npy", alive)
    print(f"rate {r:6.1f} Hz: self-sustaining core of {int(alive.sum())} neurons")

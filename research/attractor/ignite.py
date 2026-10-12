"""Followers, and a structural prediction of ignition.

Followers: of the active neurons outside the core, how many receive more
than threshold from the core firing at r?

Ignition: bootstrap forward from a stimulus. Sensors fire at the stimulus
rate; any neuron whose drive from the currently active set exceeds threshold
joins it (at rate r); repeat to closure. Then remove the stimulus and prune as
core.py does: what survives is the predicted persistent state for that
stimulus. A sense ignites when that is non-empty."""
import sys
sys.path.insert(0, "/home/lummy/Projects/GLaDOS/tools")
import numpy as np, connectome as cx
from pathlib import Path
from scipy.sparse import csr_matrix
b = cx.Brain(Path("/home/lummy/Projects/GLaDOS/out/fly.bin").read_bytes())
rows = np.asarray(b.rows); col = np.asarray(b.col); w = np.asarray(b.w, dtype=np.float64)
src = np.repeat(np.arange(b.n), np.diff(rows))
tau = b.tau_syn * b.dt_us * 1e-6
W = csr_matrix((w * b.w_syn * tau, (col, src)), shape=(b.n, b.n))
theta = b.vth - b.v0
r = 89.0
sensor = np.zeros(b.n, bool)
for ids in b.sensors.values():
    sensor[ids] = True

late = np.load("late-thermal.npy") > 0
core = np.load("core-89.npy")
fol = late & ~core
fed = (W @ (core * r)) > theta
print(f"active outside the core: {fol.sum()}; driven over threshold by the core alone: {(fol & fed).sum()} ({(fol & fed).sum()/fol.sum()*100:.1f}%)")
print(f"inactive neurons the core alone would drive over threshold: {(~late & fed & ~sensor).sum()}")

def prune(alive):
    while True:
        keep = alive & ((W @ (alive * r)) > theta)
        if keep.sum() == alive.sum():
            return keep
        alive = keep

for name in sorted(b.sensors):
    stim = np.zeros(b.n); stim[b.sensors[name]] = 150.0
    act = np.zeros(b.n, bool)
    while True:
        new = (~sensor) & ((W @ (stim + act * r)) > theta)
        if (new | act).sum() == act.sum():
            break
        act |= new
    left = prune(act.copy())
    print(f"  {name:8} recruits {act.sum():6} while on; predicted to persist after: {left.sum():6}  (core overlap {int((left & core).sum())})")

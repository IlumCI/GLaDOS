"""How many core neurons does a stimulus push past threshold by itself?
Drive on core neuron i from stimulus set S firing at 150 Hz: the steady g,
sum_j w_ij * w_syn * 150 Hz * tau_syn, against the 7 mV gap."""
import sys, json, re, numpy as np
sys.path.insert(0, "/home/lummy/Projects/GLaDOS/tools")
import connectome as cx
from pathlib import Path
from scipy.sparse import csr_matrix
b = cx.Brain(Path("/home/lummy/Projects/GLaDOS/out/fly.bin").read_bytes())
rows=np.asarray(b.rows); col=np.asarray(b.col); w=np.asarray(b.w, dtype=float)
src=np.repeat(np.arange(b.n), np.diff(rows))
tau=b.tau_syn*b.dt_us*1e-6; theta=b.vth-b.v0
W=csr_matrix((w*b.w_syn*tau,(col,src)),shape=(b.n,b.n))
core=np.load("core-30.npy")
groups=dict(b.sensors); groups.update(json.load(open("receptor-groups.json")))
def seeds(members, frac=1.0):
    x=np.zeros(b.n); x[members]=150.0
    d=W@x
    return int(((d>theta*frac)&core).sum()), int(((d>theta*frac)&~core).sum())
# measured outcome: neurons firing late
out={}
for f in Path(".").glob("m-*.txt"):
    m=re.match(r"(\w+): (\d+) neurons firing late", f.read_text()); out[m.group(1)]=int(m.group(2))
for f in Path(".").glob("r-*.txt"):
    t=f.name[2:-4]; m=re.search(r"late +(\d+)", f.read_text()); out[t]=int(m.group(1))
rows_=[]
for t,mem in groups.items():
    if t not in out: continue
    c,o=seeds(mem); c2,_=seeds(mem,0.5)
    rows_.append((c,c2,o,t,len(mem),out[t]))
for c,c2,o,t,n,late in sorted(rows_):
    print(f"  {t:14} n={n:4} core seeds {c:4} (at half threshold {c2:4}) other seeds {o:5} -> late {late:6} {'IGNITED' if late>1000 else ''}")

# Refined: how much do the seeded core neurons excite the rest of the core?
coreW = W[core][:, core]                      # target x source, inside the core
out_into_core = np.asarray((coreW.maximum(0)).sum(axis=0)).ravel() * 30.0 / theta  # per seed, at 30 Hz, in thresholds
core_idx = np.nonzero(core)[0]
print("\nseed output into the core (sum over seeds, in units of threshold at 30 Hz):")
res=[]
for t,mem in groups.items():
    if t not in out: continue
    x=np.zeros(b.n); x[mem]=150.0
    d=(W@x)
    s_mask = (d[core_idx] > theta)
    res.append((float(out_into_core[s_mask].sum()), int(s_mask.sum()), t, out[t]))
for v,c,t,late in sorted(res):
    print(f"  {t:14} seeds {c:3}  seed output into core {v:8.1f}  -> late {late:6} {'IGNITED' if late>1000 else ''}")

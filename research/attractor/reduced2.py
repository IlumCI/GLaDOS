"""The core plus its inhibitory loop: every neuron that takes input from the
core (or the stimulus) and sends inhibition back onto it. Same test as
reduced.py, against every full-brain case measured so far."""
import sys, json, re, time, copy, numpy as np
sys.path.insert(0, "/home/lummy/Projects/GLaDOS/tools")
import connectome as cx, pci
from pathlib import Path
b = cx.Brain(Path("/home/lummy/Projects/GLaDOS/out/fly.bin").read_bytes())
rows=np.asarray(b.rows); col=np.asarray(b.col); w=np.asarray(b.w)
src=np.repeat(np.arange(b.n), np.diff(rows))
core=np.load("core-30.npy")
inhib_onto_core = np.zeros(b.n, bool); inhib_onto_core[src[core[col] & (w < 0)]] = True
fed_by_core = np.zeros(b.n, bool); fed_by_core[col[core[src] & (w > 0)]] = True
loop = inhib_onto_core & fed_by_core & ~core

def model(members):
    keep = core | loop
    fed_by_stim = np.zeros(b.n, bool); s = np.zeros(b.n, bool); s[members] = True
    fed_by_stim[col[s[src] & (w > 0)]] = True
    keep = keep | (inhib_onto_core & fed_by_stim); keep[members] = True
    idx = np.nonzero(keep)[0]; remap = -np.ones(b.n, int); remap[idx] = np.arange(len(idx))
    r2=[0]; c2=[]; w2=[]
    for i in idx:
        a,z=rows[i],rows[i+1]; t=remap[col[a:z]]; m=t>=0
        c2.append(t[m]); w2.append(w[a:z][m]); r2.append(r2[-1]+int(m.sum()))
    g=pci.Graph(len(idx), np.asarray(r2), np.concatenate(c2), np.concatenate(w2))
    bb=copy.copy(b); bb.n=len(idx); bb.sensors={"stim":[int(remap[m]) for m in members]}
    return bb, g, remap

def ignites(members, seed=7):
    bb, g, remap = model(members)
    spm = 1000 // b.dt_us
    r, _ = pci.simulate(bb, g, 1000*spm, [("stim",150,0,500*spm)], list(range(bb.n)), 500*spm, seed)
    return int((r[1][remap[core]] > 0).sum()), bb.n

if __name__ == "__main__":
    print(f"inhibitory loop neurons: {loop.sum()}")
    groups=dict(b.sensors); groups.update(json.load(open("receptor-groups.json")))
    out={}
    for f in Path(".").glob("m-*.txt"):
        m=re.match(r"(\w+): (\d+) neurons firing late", f.read_text()); out[m.group(1)]=int(m.group(2))
    for f in Path(".").glob("r-*.txt"):
        out[f.name[2:-4]]=int(re.search(r"late +(\d+)", f.read_text()).group(1))
    agree=0
    for t,mem in sorted(groups.items()):
        if t not in out: continue
        c, n = ignites(mem)
        full = out[t] > 1000; red = c > 100
        agree += full == red
        print(f"  {t:16} model n={n:5} core late {c:4}  full brain late {out[t]:6}  {'agree' if full==red else 'DISAGREE'}", flush=True)
    print(f"agreement {agree} of {len(out)}")

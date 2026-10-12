"""The core alone: simulate only the 477-neuron core plus the stimulated
neurons, every synapse among them kept, nothing else. Does it reproduce the
full brain's ignition outcome?"""
import sys, json, re, time, copy, numpy as np
sys.path.insert(0, "/home/lummy/Projects/GLaDOS/tools")
import connectome as cx, pci
from pathlib import Path
b = cx.Brain(Path("/home/lummy/Projects/GLaDOS/out/fly.bin").read_bytes())
rows=np.asarray(b.rows); col=np.asarray(b.col); w=np.asarray(b.w)
core=np.load("core-30.npy")
groups=dict(b.sensors); groups.update(json.load(open("receptor-groups.json")))

def reduced(members):
    keep = core.copy(); keep[members] = True
    idx = np.nonzero(keep)[0]; remap = -np.ones(b.n, int); remap[idx] = np.arange(len(idx))
    r2=[0]; c2=[]; w2=[]
    for i in idx:
        a,z=rows[i],rows[i+1]; t=remap[col[a:z]]; m=t>=0
        c2.append(t[m]); w2.append(w[a:z][m]); r2.append(r2[-1]+int(m.sum()))
    g=pci.Graph(len(idx), np.asarray(r2), np.concatenate(c2), np.concatenate(w2))
    bb=copy.copy(b); bb.n=len(idx); bb.sensors={"stim":[int(remap[m]) for m in members]}
    return bb, g, remap

def ignites(members, seed=7):
    bb, g, remap = reduced(members)
    spm = 1000 // b.dt_us
    r, _ = pci.simulate(bb, g, 1000*spm, [("stim",150,0,500*spm)], list(range(bb.n)), 500*spm, seed)
    corepos = remap[core]
    return int((r[1][corepos] > 0).sum())

if __name__ == "__main__":
    out={}
    for f in Path(".").glob("m-*.txt"):
        m=re.match(r"(\w+): (\d+) neurons firing late", f.read_text()); out[m.group(1)]=int(m.group(2))
    for f in Path(".").glob("r-*.txt"):
        out[f.name[2:-4]]=int(re.search(r"late +(\d+)", f.read_text()).group(1))
    agree=0
    for t,mem in sorted(groups.items()):
        if t not in out: continue
        t0=time.time(); c=ignites(mem)
        full = out[t] > 1000; red = c > 100
        agree += full == red
        print(f"  {t:14} reduced core late {c:4}  full brain late {out[t]:6}  {'agree' if full==red else 'DISAGREE'}  {time.time()-t0:.1f} s", flush=True)
    print(f"agreement {agree} of {len(out)}")

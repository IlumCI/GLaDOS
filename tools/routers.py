#!/usr/bin/env python3
"""How good can the path that never runs the transformer get?

The evaluator found the 135M model scoring 5.7% on held-out tool selection
while a nearest-neighbour lookup over pooled embeddings got 32.1%. That is not
a speed/accuracy tradeoff -- the cheap path is better *and* about four orders of
magnitude cheaper -- so it is worth finding out where its ceiling actually is.

Everything here is old. Nearest-class-mean is Rocchio (1971). Ridge regression
onto one-hot targets is Widrow-Hoff (1960) and is what a linear probe is when
you solve it directly instead of by gradient descent -- which also sidesteps
the overfitting that killed the earlier SGD head. The binary variant is
Kanerva's sparse distributed memory (1988) and hyperdimensional computing:
project to high dimension, take signs, compare by Hamming distance. That last
one matters because Hamming distance over packed bits is popcount, which on
this CPU is one instruction per 64 dimensions -- a 21-way decision in roughly a
microsecond, with no floating point anywhere.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as ev  # noqa: E402
from reference import load  # noqa: E402

MODEL = "tools/out/smollm2-135m-q8.bin"
TOKENIZER = "tools/out/smollm2-tok.bin"
DATA = "tools/out/tools-dataset.json"
ev.HF_TOKENIZER = "tools/hf/tokenizer.json"


def features(w, tok, items):
    return np.stack([ev.pooled(w, tok, e["task"]) for e in items])


def report(name, right, total, note=""):
    print(f"  {name:44} {right/total:6.1%}  {note}")


def main():
    cfg, w, _ = load(MODEL)
    tok = ev.Tok(TOKENIZER)
    data = json.loads(Path(DATA).read_text(encoding="utf-8"))
    train, test = data["train"], data["test"]
    names = ev.NAMES
    idx = {n: i for i, n in enumerate(names)}

    Xtr = features(w, tok, train)
    Xte = features(w, tok, test)
    ytr = np.array([idx[e["applet"]] for e in train])
    yte = np.array([idx[e["applet"]] for e in test])

    print(f"  {len(train)} train, {len(test)} test, {len(names)} classes, "
          f"chance {100/len(names):.1f}%")
    print(f"  feature dim {Xtr.shape[1]}\n")

    # Centring is not cosmetic: pooled embeddings share a huge common component
    # (every sentence has the same function words), and without removing it
    # every pair is ~0.99 similar and the argmax is noise.
    mu = Xtr.mean(axis=0)
    A = Xtr - mu
    B = Xte - mu

    def unit(M):
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

    An, Bn = unit(A), unit(B)

    # --- 1-NN and k-NN, Rocchio's neighbourhood idea ---
    S = Bn @ An.T
    report("1-nn cosine", int((ytr[S.argmax(1)] == yte).sum()), len(yte))
    for k in (3, 5):
        top = np.argsort(-S, axis=1)[:, :k]
        votes = np.zeros((len(yte), len(names)))
        for r in range(len(yte)):
            for c, j in enumerate(top[r]):
                votes[r, ytr[j]] += S[r, j]
        report(f"{k}-nn cosine (similarity-weighted)",
               int((votes.argmax(1) == yte).sum()), len(yte))

    # --- nearest class mean (Rocchio 1971) ---
    cent = np.stack([An[ytr == c].mean(axis=0) if (ytr == c).any()
                     else np.zeros(An.shape[1]) for c in range(len(names))])
    cent = unit(cent)
    report("nearest class mean", int(((Bn @ cent.T).argmax(1) == yte).sum()), len(yte))

    # --- ridge regression onto one-hot (Widrow-Hoff, solved directly) ---
    Y = np.eye(len(names))[ytr]
    for lam in (0.1, 1.0, 10.0):
        d = A.shape[1]
        W = np.linalg.solve(A.T @ A + lam * np.eye(d), A.T @ Y)
        report(f"ridge probe (lambda={lam})",
               int(((B @ W).argmax(1) == yte).sum()), len(yte),
               "closed form, no SGD to overfit")

    # --- hyperdimensional: project, binarise, compare by Hamming ---
    # The whole model becomes a table of D-bit class vectors. A decision is
    # D/64 popcounts per class.
    rng = np.random.default_rng(0)
    for D in (2048, 8192):
        P = rng.standard_normal((A.shape[1], D)).astype(np.float32) / np.sqrt(D)
        HA = (A @ P) > 0
        HB = (B @ P) > 0
        proto = np.stack([
            (HA[ytr == c].mean(axis=0) > 0.5) if (ytr == c).any()
            else np.zeros(D, dtype=bool) for c in range(len(names))
        ])
        # Hamming distance; argmin over classes.
        pred = np.array([np.count_nonzero(proto ^ h, axis=1).argmin() for h in HB])
        bits = D // 8
        report(f"hyperdimensional binary, D={D}",
               int((pred == yte).sum()), len(yte),
               f"{bits} B/class, {D//64} popcounts per class")

    print("\n  for scale: the transformer scored 5.7% on this same split,")
    print("  at roughly 189 tokens of prefill per decision.")


if __name__ == "__main__":
    main()

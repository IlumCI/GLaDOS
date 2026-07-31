#!/usr/bin/env python3
"""Decide where to widen the corpus, instead of widening it blindly.

Three questions, and expanding without answering them risks spending effort on
the wrong axis entirely:

  1. Where does the probe actually fail? Expanding uniformly wastes most of the
     work on classes that are already fine. The confusion pairs say where the
     decision boundary is genuinely unclear.

  2. Is the learning curve still climbing? Accuracy at 25/50/75/100% of the
     training data shows whether more examples would help at all. A curve that
     has flattened means the ceiling is the features, and more data is wasted
     effort no matter how well targeted.

  3. Is it a feature ceiling? Mean-pooling weights "the" exactly as heavily as
     "unlink", which is obviously wrong for deciding intent. Inverse document
     frequency weighting and removing the dominant principal component are the
     two cheapest fixes -- together they are essentially Arora et al. (2017)'s
     SIF baseline, which is famously hard to beat with anything more clever.

If the curve is flat and better features help, widening is not the priority.
If the curve is steep, it is.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as ev  # noqa: E402
from reference import load  # noqa: E402

MODEL = "tools/out/smollm2-135m-q8.bin"
TOKENIZER = "tools/out/smollm2-tok.bin"
DATA = "tools/out/tools-dataset.json"
ev.HF_TOKENIZER = "tools/hf/tokenizer.json"


def token_ids(tok, items):
    return [ev.encode(tok, e["task"]) for e in items]


def build_idf(id_lists, vocab_size):
    """Inverse document frequency over the training tasks."""
    df = Counter()
    for ids in id_lists:
        for t in set(ids):
            df[t] += 1
    n = len(id_lists)
    idf = np.ones(vocab_size, dtype=np.float32)
    for t, c in df.items():
        idf[t] = np.log((n + 1) / (c + 1)) + 1.0
    return idf


def embed(w, id_lists, idf=None):
    dim = w["embed"].shape[1]
    out = np.zeros((len(id_lists), dim), dtype=np.float32)
    for i, ids in enumerate(id_lists):
        if not ids:
            continue
        rows = w["embed"][ids]
        if idf is None:
            out[i] = rows.mean(axis=0)
        else:
            wt = idf[ids][:, None]
            out[i] = (rows * wt).sum(axis=0) / wt.sum()
    return out


def ridge(A, y, classes, lam):
    Y = np.eye(classes)[y]
    d = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(d), A.T @ Y)


def fit_score(Xtr, ytr, Xte, yte, classes, lam=1.0, drop_pc=0):
    mu = Xtr.mean(axis=0)
    A, B = Xtr - mu, Xte - mu
    if drop_pc:
        # Remove the dominant directions. In pooled embeddings the top
        # component is essentially "this is English prose" and carries no
        # information about which tool is wanted.
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        P = Vt[:drop_pc]
        A = A - (A @ P.T) @ P
        B = B - (B @ P.T) @ P
    W = ridge(A, ytr, classes, lam)
    pred = (B @ W).argmax(1)
    return float((pred == yte).mean()), pred


def main():
    cfg, w, _ = load(MODEL)
    tok = ev.Tok(TOKENIZER)
    data = json.loads(Path(DATA).read_text(encoding="utf-8"))
    train, test = data["train"], data["test"]
    names = ev.NAMES
    idx = {n: i for i, n in enumerate(names)}
    C = len(names)

    tr_ids = token_ids(tok, train)
    te_ids = token_ids(tok, test)
    ytr = np.array([idx[e["applet"]] for e in train])
    yte = np.array([idx[e["applet"]] for e in test])
    idf = build_idf(tr_ids, w["embed"].shape[0])

    print(f"  {len(train)} train / {len(test)} test, {C} classes, chance {100/C:.1f}%\n")

    # --- 3. feature variants ---
    print("  features (ridge lambda=1, same split):")
    variants = {
        "mean pooled (current)": (embed(w, tr_ids), embed(w, te_ids), 0),
        "idf weighted": (embed(w, tr_ids, idf), embed(w, te_ids, idf), 0),
        "idf + drop 1 PC": (embed(w, tr_ids, idf), embed(w, te_ids, idf), 1),
        "idf + drop 3 PC": (embed(w, tr_ids, idf), embed(w, te_ids, idf), 3),
    }
    best_name, best_acc, best_pred, best_feats = None, -1, None, None
    for label, (Xtr, Xte, pcs) in variants.items():
        acc, pred = fit_score(Xtr, ytr, Xte, yte, C, 1.0, pcs)
        flag = ""
        if acc > best_acc:
            best_name, best_acc, best_pred, best_feats = label, acc, pred, (Xtr, Xte, pcs)
            flag = "  <-"
        print(f"    {label:26} {acc:6.1%}{flag}")

    # --- 2. learning curve, on the best features ---
    print(f"\n  learning curve ({best_name}):")
    Xtr, Xte, pcs = best_feats
    rng = np.random.default_rng(0)
    order = rng.permutation(len(ytr))
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = max(C, int(len(order) * frac))
        sel = order[:k]
        acc, _ = fit_score(Xtr[sel], ytr[sel], Xte, yte, C, 1.0, pcs)
        bar = "#" * int(acc * 40)
        print(f"    {int(frac*100):3}%  n={k:4}  {acc:6.1%}  {bar}")

    # --- 1. where it fails ---
    print(f"\n  confusions ({best_name}, {best_acc:.1%}):")
    pairs = Counter()
    per_class_wrong = Counter()
    per_class_total = Counter()
    for t, p in zip(yte, best_pred):
        per_class_total[names[t]] += 1
        if t != p:
            pairs[(names[t], names[p])] += 1
            per_class_wrong[names[t]] += 1
    print("    worst pairs (want -> got):")
    for (a, b), n in pairs.most_common(10):
        print(f"      {a:8} -> {b:8}  x{n}")

    print("\n    per-class recall:")
    rows = []
    for n in names:
        tot = per_class_total.get(n, 0)
        if tot == 0:
            continue
        ok = tot - per_class_wrong.get(n, 0)
        rows.append((ok / tot, n, ok, tot))
    rows.sort()
    for acc, n, ok, tot in rows:
        mark = "  <- widen" if acc < 0.5 else ""
        print(f"      {n:8} {ok}/{tot} {acc:6.0%}{mark}")


if __name__ == "__main__":
    main()

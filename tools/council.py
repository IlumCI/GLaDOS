#!/usr/bin/env python3
"""A council of cheap cores, combined as a product of experts.

The single ridge probe reaches 69.4% on the independent EVAL set, and its
remaining errors are semantic rather than lexical -- cp -> same, du -> ls,
sysbox -> fsck. Those are cases where one mean-pooled vector genuinely cannot
separate two meanings, which argues for a second *view* of the request rather
than more examples of the first.

Hinton's products of experts (1999/2002) is the combination rule, not a mixture.
In a mixture one expert wins and the rest are discarded; in a product every
expert multiplies its opinion in, so a core that is confident sharpens the
result and a core that has no idea emits a near-flat distribution and
effectively abstains. That is the "cores that balance each other" behaviour,
and it falls out of summing log-probabilities rather than needing a gate.

Each core is given a genuinely different view, because two experts looking at
the same features disagree only by noise:

  semantic   mean-pooled embeddings, ridge. Knows that "duplicate" and "copy"
             are related. Blind to which exact word was used.
  lexical    multinomial naive Bayes over token counts. Knows exactly which
             words appeared and nothing about what they mean. Trained by
             counting, so there is no matrix to invert.
  character  hashed character 3-grams, same naive Bayes. Sees morphology --
             "duplicating" and "duplicate" share most of their trigrams even
             though they may be different tokens entirely.

Per-core weights are the temperature knob: a weight near zero makes a core
abstain regardless of how loud it is. They are swept here rather than assumed.
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


def log_softmax(M):
    M = M - M.max(axis=1, keepdims=True)
    return M - np.log(np.exp(M).sum(axis=1, keepdims=True))


# --- semantic core ------------------------------------------------------

def semantic(w, tr_ids, te_ids, ytr, C, lam=1.0):
    dim = w["embed"].shape[1]

    def pool(ids_list):
        out = np.zeros((len(ids_list), dim), dtype=np.float32)
        for i, ids in enumerate(ids_list):
            if ids:
                out[i] = w["embed"][ids].mean(axis=0)
        return out

    A, B = pool(tr_ids), pool(te_ids)
    mu = A.mean(axis=0)
    A, B = A - mu, B - mu
    Y = np.eye(C)[ytr]
    W = np.linalg.solve(A.T @ A + lam * np.eye(dim), A.T @ Y)
    return A @ W, B @ W


# --- counting cores -----------------------------------------------------

def naive_bayes(tr_counts, te_counts, ytr, C, V, alpha=0.2):
    """Multinomial naive Bayes. Training is counting; there is nothing to solve.

    Returns log-probabilities directly, which is what the product wants -- no
    calibration step needed, unlike the ridge scores.
    """
    counts = np.full((C, V), alpha, dtype=np.float64)
    for row, y in zip(tr_counts, ytr):
        for f, n in row.items():
            counts[y, f] += n
    logp = np.log(counts / counts.sum(axis=1, keepdims=True))

    prior = np.bincount(ytr, minlength=C).astype(np.float64)
    prior = np.log(prior / prior.sum())

    def score(rows):
        out = np.zeros((len(rows), C))
        for i, row in enumerate(rows):
            s = prior.copy()
            for f, n in row.items():
                s += n * logp[:, f]
            out[i] = s
        return out

    return score(tr_counts), score(te_counts)


def token_counts(ids_list, vocab):
    out = []
    for ids in ids_list:
        c = Counter()
        for t in ids:
            if t in vocab:
                c[vocab[t]] += 1
        out.append(dict(c))
    return out


def fnv1a(s):
    """Deterministic 32-bit hash.

    Python's `hash` is salted per process unless PYTHONHASHSEED is set, so
    using it here would make the character core's buckets -- and therefore its
    accuracy -- change between runs. It also has to be reimplementable in the
    kernel, and FNV-1a is a dozen lines anywhere.
    """
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def char_counts(texts, buckets=4096, n=3):
    out = []
    for t in texts:
        s = f"  {t.lower()}  "
        c = Counter()
        for i in range(len(s) - n + 1):
            c[fnv1a(s[i:i + n]) % buckets] += 1
        out.append(dict(c))
    return out


def accuracy(scores, y):
    return float((scores.argmax(1) == y).mean())


def main():
    cfg, w, _ = load(MODEL)
    tok = ev.Tok(TOKENIZER)
    data = json.loads(Path(DATA).read_text(encoding="utf-8"))
    train, test = data["train"], data["test"]
    names = ev.NAMES
    idx = {n: i for i, n in enumerate(names)}
    C = len(names)

    tr_text = [e["task"] for e in train]
    te_text = [e["task"] for e in test]
    ytr = np.array([idx[e["applet"]] for e in train])
    yte = np.array([idx[e["applet"]] for e in test])

    tr_ids = [ev.encode(tok, t) for t in tr_text]
    te_ids = [ev.encode(tok, t) for t in te_text]

    print(f"  {len(train)} train / {len(test)} test, {C} classes, "
          f"chance {100/C:.1f}%, resolution {100/len(test):.1f}%\n")

    cores = {}

    _, sem_te = semantic(w, tr_ids, te_ids, ytr, C)
    cores["semantic"] = log_softmax(sem_te)

    # Only tokens actually seen in training get a column; the rest can never
    # carry evidence and would only add smoothing mass.
    seen = sorted({t for ids in tr_ids for t in ids})
    vmap = {t: i for i, t in enumerate(seen)}
    _, lex_te = naive_bayes(token_counts(tr_ids, vmap), token_counts(te_ids, vmap),
                            ytr, C, len(seen))
    cores["lexical"] = log_softmax(lex_te)

    B = 4096
    _, chr_te = naive_bayes(char_counts(tr_text, B), char_counts(te_text, B),
                            ytr, C, B)
    cores["character"] = log_softmax(chr_te)

    print("  each core alone:")
    for name, s in cores.items():
        print(f"    {name:12} {accuracy(s, yte):6.1%}")

    # --- the product -----------------------------------------------------
    #
    # Weights are chosen by cross-validation on the *training* set and then
    # applied to EVAL exactly once. Sweeping a 216-point grid against EVAL and
    # reporting the maximum is not an estimate of anything: with 49 items it
    # selects noise, and it did -- the best combination that way dropped the
    # semantic core entirely, which no honest procedure supports.
    print("\n  products:")
    grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    folds = 5
    rng = np.random.default_rng(0)
    order = rng.permutation(len(ytr))

    fold_scores = {k: [] for k in cores}
    fold_y = []
    for f in range(folds):
        held = order[f::folds]
        keep = np.setdiff1d(order, held)
        sub_ids = [tr_ids[i] for i in keep]
        sub_y = ytr[keep]
        h_ids = [tr_ids[i] for i in held]

        _, s_sem = semantic(w, sub_ids, h_ids, sub_y, C)
        fold_scores["semantic"].append(log_softmax(s_sem))

        s_seen = sorted({t for ids in sub_ids for t in ids})
        s_map = {t: i for i, t in enumerate(s_seen)}
        _, s_lex = naive_bayes(token_counts(sub_ids, s_map),
                               token_counts(h_ids, s_map), sub_y, C, len(s_seen))
        fold_scores["lexical"].append(log_softmax(s_lex))

        _, s_chr = naive_bayes(char_counts([tr_text[i] for i in keep], B),
                               char_counts([tr_text[i] for i in held], B),
                               sub_y, C, B)
        fold_scores["character"].append(log_softmax(s_chr))
        fold_y.append(ytr[held])

    cv = {k: np.concatenate(v) for k, v in fold_scores.items()}
    cv_y = np.concatenate(fold_y)

    best = (None, -1.0)
    for a in grid:
        for b in grid:
            for c in grid:
                if a == b == c == 0:
                    continue
                s = a * cv["semantic"] + b * cv["lexical"] + c * cv["character"]
                acc = accuracy(s, cv_y)
                if acc > best[1]:
                    best = ((a, b, c), acc)
    a, b, c = best[0]
    print(f"    weights {best[0]} chosen by {folds}-fold CV on train "
          f"({best[1]:.1%} there)")

    chosen = a * cores["semantic"] + b * cores["lexical"] + c * cores["character"]
    print(f"    council on EVAL             {accuracy(chosen, yte):6.1%}")

    equal = sum(cores.values())
    print(f"    equal weights on EVAL       {accuracy(equal, yte):6.1%}")
    print(f"    semantic alone (baseline)   {accuracy(cores['semantic'], yte):6.1%}")

    # --- answer rules ----------------------------------------------------
    #
    # A product is not the only way to use three cores, and the kernel turned
    # up a case that made the distinction concrete: for "delete that file
    # please" the probe answered cat while both counting cores answered rm.
    # Under "the probe answers" that request routes wrongly even though two
    # cores out of three had it right. Majority vote is a different rule and
    # has to be measured rather than reasoned about.
    print("\n  answer rules:")
    picks = np.stack([cores[k].argmax(1) for k in ("semantic", "lexical", "character")])

    def majority(prefer_row):
        out = np.empty(len(yte), dtype=int)
        for i in range(len(yte)):
            col = picks[:, i]
            counts = Counter(col.tolist())
            top, n = counts.most_common(1)[0]
            # Ties go to the preferred core rather than to whichever class
            # happens to sort first.
            out[i] = top if n > 1 else col[prefer_row]
        return out

    for label, pred in (
        ("probe answers (current)", picks[0]),
        ("majority, ties to probe", majority(0)),
        ("majority, ties to lexical", majority(1)),
    ):
        print(f"    {label:28} {float((pred == yte).mean()):6.1%}")
    print(f"    {'product, equal weights':28} {accuracy(equal, yte):6.1%}")

    # The subset that matters: where the probe stands alone against the other
    # two. If the pair is usually right there, deferring to them is free
    # accuracy; if not, the current rule is correct as it stands.
    alone = (picks[0] != picks[1]) & (picks[1] == picks[2])
    if alone.any():
        probe_ok = float((picks[0][alone] == yte[alone]).mean())
        pair_ok = float((picks[1][alone] == yte[alone]).mean())
        print(f"\n  probe outvoted 2-1 on {int(alone.sum())} items:")
        print(f"    probe right there  {probe_ok:6.1%}")
        print(f"    pair right there   {pair_ok:6.1%}")

    # --- where the cores disagree ----------------------------------------
    # The interesting property is not only the score. If the cores agree on
    # most items, a cascade can answer those instantly and escalate only the
    # rest, which is the whole point of having more than one.
    picks = {k: s.argmax(1) for k, s in cores.items()}
    unanimous = np.ones(len(yte), dtype=bool)
    for k in picks:
        unanimous &= picks[k] == picks["semantic"]
    n_un = int(unanimous.sum())
    if n_un:
        print(f"\n  all three agree on {n_un}/{len(yte)} items, "
              f"{accuracy(np.eye(C)[picks['semantic'][unanimous]], yte[unanimous]):.1%} correct there")
    if n_un < len(yte):
        d = ~unanimous
        s = sum(cores.values())
        print(f"  they disagree on {int(d.sum())}, "
              f"{accuracy(s[d], yte[d]):.1%} correct after the product")


if __name__ == "__main__":
    main()

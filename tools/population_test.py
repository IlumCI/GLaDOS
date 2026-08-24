#!/usr/bin/env python3
"""Population over decodes: the surviving form of the globular idea.

The latent-field variant is dead, measured three ways: latent re-entry is
destructive (Coconut result), and single-position fitness is structurally
blind because the routing signal is multi-positional. What survives is
population + selection where the fitness is honest -- sample K candidate
actions from the constrained distribution at temperature, score each with
the FULL multi-position continuation, keep the best. The kernel already
has the fork machinery (KV snapshot/restore); this measures whether the
population buys accuracy over the greedy decode's 33.3%.

Baseline is greedy constrained decode on the same items.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import v4  # noqa: E402
import traces  # noqa: E402
from lm_eval import Hybrid35, build_alphabet  # noqa: E402
from evaluate import Tok, constrained_pick  # noqa: E402
from tokenizers import Tokenizer as HFTok  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def sampled_pick(runner, alphabet, names, logits, rng, temperature):
    """Constrained decode with sampling instead of argmax. Same walk as
    constrained_pick; the choice among admissible candidates is a softmax
    draw, so repeated runs with different rng explore the neighbourhood of
    the greedy answer."""
    alts = [n.encode() + b"\n" for n in names]
    produced = b""
    started = False
    for _ in range(24):
        cands = []
        for i, piece in enumerate(alphabet):
            if not piece:
                continue
            p = piece
            if not started:
                if all(c == 0x20 for c in p):
                    continue
                if p[:1] == b" ":
                    p = p[1:]
                    if not p:
                        continue
            n = len(produced)
            if any(len(a) >= n + len(p) and a.startswith(produced) and a[n:].startswith(p)
                   for a in alts):
                cands.append(i)
        if not cands:
            return None
        vals = np.asarray([logits[i] for i in cands], dtype=np.float64) / temperature
        vals -= vals.max()
        p = np.exp(vals)
        p /= p.sum()
        best = cands[int(rng.choice(len(cands), p=p))]
        piece = alphabet[best]
        if not started:
            if piece[:1] == b" ":
                piece = piece[1:]
            started = True
        produced += piece
        for a in alts:
            if a == produced:
                return a[: -len(b"\n")].decode("utf-8", "replace")
        if not any(a.startswith(produced) for a in alts):
            return None
        logits = runner.feed([best])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tokenizer")
    ap.add_argument("--hf-tokenizer", required=True)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tensors, cfg = v4.load(args.model)
    runner = Hybrid35(tensors, cfg, max_len=1024)
    hf = HFTok.from_file(args.hf_tokenizer)
    tok = Tok(args.tokenizer)
    alphabet = build_alphabet(tok)
    names = traces.COMMANDS + traces.APPLETS

    items = [json.loads(l) for l in
             (ROOT / "out/traces.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    test = [e for e in items if e.get("split") == "test"][: args.limit]

    greedy_right = 0
    pop_right = 0
    oracle_right = 0
    t0 = time.time()

    for i, e in enumerate(test):
        text = e["text"]
        if text.endswith("<|im_end|>"):
            text = text[: -len("<|im_end|>")]
        cut = text.rfind("</think>")
        if cut != -1:
            text = text[: cut + len("</think>") + 1]
        ids = hf.encode(text, add_special_tokens=False).ids

        runner.reset()
        logits = runner.feed(ids)
        snap = runner.snapshot()
        # The fork is taken AFTER the prompt; every candidate decode restores
        # to here, so each explores the same prefix.

        greedy = constrained_pick(runner, tok, alphabet, names, logits)
        runner.restore(snap)

        rng = np.random.RandomState(args.seed * 99991 + i)
        best = None
        best_score = -1e30
        any_right = False
        for k in range(args.k):
            cand = sampled_pick(runner, alphabet, names, logits, rng, args.temperature)
            runner.restore(snap)
            if cand is None:
                continue
            # honest fitness: the full multi-position continuation logprob of
            # this action, conditioned on the prompt.
            a_ids = hf.encode(cand, add_special_tokens=False).ids
            runner.restore(snap)
            lg = runner.feed(a_ids)
            m = float(lg.max())
            score = float((lg - (m + np.log(np.exp(lg - m).sum())))[a_ids[-1]])
            if score > best_score:
                best_score = score
                best = cand
            any_right = any_right or cand == e["action"]

        greedy_right += greedy == e["action"]
        if best is not None:
            pop_right += best == e["action"]
        oracle_right += any_right
        if i < 6:
            m1 = "ok " if greedy == e["action"] else "-- "
            m2 = "ok " if best == e["action"] else "-- "
            print(f"  {m1}greedy {str(greedy):10} {m2}pop {str(best):10} "
                  f"want {e['action']:10}")

    n = len(test)
    print(f"\n  greedy constrained decode      : {greedy_right}/{n} = {greedy_right/n:6.1%}")
    print(f"  population best-of-{args.k} (T={args.temperature}) : "
          f"{pop_right}/{n} = {pop_right/n:6.1%}")
    print(f"  oracle (any candidate right)   : {oracle_right}/{n} = {oracle_right/n:6.1%}")
    print(f"  elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

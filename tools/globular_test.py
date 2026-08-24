#!/usr/bin/env python3
"""Host prototype of the Globular Field, GLaDOS overhaul.

Measures, on the route rail, whether a population of latent hypothesis
agents refined by core/halo gravity and a *wired-in* genetic search beats
the single hidden state the model normally decodes from.

Fairness: both sides are scored by the same single-position scorer -- the
sum of the action-name token logits read off one hidden state through the
model's own head. The baseline scores the model's h; the field scores the
fittest agent after S steps. No new parameters anywhere: agents live in
the model's hidden space, fitness comes from the model's own head.

The original this overhauls (Euroswarms-Institute/Stable-Cognition)
declares genetic selection, novelty search, species, meta-memory and
coevolution; its forward path uses none of them. Here the genetic
operators are wired in and each earns its place, or the flag comes out.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ref35  # noqa: E402
import v4  # noqa: E402
import traces  # noqa: E402
from tokenizers import Tokenizer as HFTok  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def rmsnorm_plain(x, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def action_token_ids(hf, names):
    """First-token decomposition of every action name, precomputed once."""
    table = []
    for name in names:
        ids = hf.encode(name, add_special_tokens=False).ids
        table.append((name, ids))
    return table


def score_actions(logits, table):
    """Single-position scorer: sum of length-normalised log-softmax over each
    action's tokens. Raw logit sums are systematically biased toward names
    whose pieces happen to carry high weights -- the first version of this
    scorer picked 'wpa2' for every item under every hidden state. The
    partition function is shared across names at one position, so log-softmax
    removes the scale; length normalisation removes the remaining arity
    bias."""
    m = float(logits.max())
    logp = logits - (m + np.log(np.exp(logits - m).sum()))
    out = []
    for _, ids in table:
        s = 0.0
        for t in ids:
            if t < logp.shape[0]:
                s += float(logp[t])
        out.append(s / max(len(ids), 1))
    return np.asarray(out, dtype=np.float32)


def globular_refine(h, head_w, table, cfg, rng):
    """The field. K hypotheses, S steps: gravity, anchoring, and a genetic
    search that is actually wired in. Returns (best_agent, diagnostics)."""
    K, S = cfg["agents"], cfg["steps"]
    dim = h.shape[0]
    eps = 1e-6

    # Seeded perturbations: orthogonal-ish starts via fixed-seed gaussians,
    # scaled to the hidden state's own magnitude so sigma means something.
    scale = float(np.linalg.norm(h)) / np.sqrt(dim)
    agents = h[None, :] + scale * cfg["sigma0"] * rng.standard_normal((K, dim)) / np.sqrt(dim)
    agents[0] = h  # agent 0 is always the un-perturbed anchor

    energies = np.zeros(K, dtype=np.float32)
    history = []

    for s in range(S):
        sigma = cfg["sigma0"] * (cfg["decay"] ** s)  # annealing

        # -- fitness: the model's own head scores each hypothesis against
        #    the constrained action set. Lower energy = better hypothesis.
        for i in range(K):
            logits = head_w @ rmsnorm_plain(agents[i], eps)
            energies[i] = -score_actions(logits, table).max()

        conf = np.exp(-(energies - energies.min()))
        conf /= conf.sum()

        # -- core/halo gravity: the fittest quarter form the core, and their
        #    centroid pulls the population in. The globular-cluster physics,
        #    kept from the original and central here.
        k_core = max(2, int(K * cfg["core_ratio"]))
        core_idx = np.argsort(-conf)[:k_core]
        core = agents[core_idx]
        centroid = core.mean(axis=0)
        for i in range(K):
            agents[i] += cfg["eta"] * (centroid - agents[i]) * conf[i]

        # -- context anchor: nobody drifts far from what the model actually
        #    thought. Replaces the original's learned input-influence
        #    attention with a convex pull.
        agents = (1.0 - cfg["beta"]) * agents + cfg["beta"] * h

        # -- the genetic search, wired in this time. Bottom half replaced by
        #    BLX-alpha crossover of two core parents plus annealed gaussian
        #    mutation; fitness sharing (crowding) keeps the population from
        #    collapsing onto one point.
        order = np.argsort(energies)  # ascending energy = best first
        n_replace = K // 2
        for j in range(n_replace):
            idx = order[K - 1 - j]  # worst
            p1, p2 = core[rng.randint(k_core)], core[rng.randint(k_core)]
            lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
            alpha = rng.uniform(cfg["alpha_min"], cfg["alpha_max"])
            child = p1 + alpha * (hi - lo) * (2 * rng.random(dim) - 1)
            child = (child + p2) / 2.0
            child = child + sigma * scale * rng.standard_normal(dim) / np.sqrt(dim)
            # fitness sharing: crowd out duplicates
            for i in range(K):
                if i == idx:
                    continue
                d = float(np.dot(child, agents[i]) /
                          (np.linalg.norm(child) * np.linalg.norm(agents[i]) + 1e-9))
                if d > 0.995:
                    child = child + 0.5 * sigma * scale * rng.standard_normal(dim) / np.sqrt(dim)
                    break
            agents[idx] = child

        history.append(float(energies.min()))

    # final scoring for readout
    best = int(np.argmin(energies))
    logits = head_w @ rmsnorm_plain(agents[best], eps)
    return agents[best], {"history": history, "final_energies": energies.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tokenizer")
    ap.add_argument("--hf-tokenizer", required=True)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--sigma0", type=float, default=0.35)
    ap.add_argument("--decay", type=float, default=0.7)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tensors, cfg = v4.load(args.model)
    hf = HFTok.from_file(args.hf_tokenizer)
    head_w = np.asarray(
        tensors.get("lm_head.weight",
                    tensors["model.language_model.embed_tokens.weight"]),
        dtype=np.float32)
    eps = cfg["rms_norm_eps"]

    names = traces.COMMANDS + traces.APPLETS
    table = action_token_ids(hf, names)

    items = [json.loads(l) for l in
             (ROOT / "out/traces.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    test = [e for e in items if e.get("split") == "test"][: args.limit]

    field_cfg = {"agents": args.agents, "steps": args.steps,
                 "sigma0": args.sigma0, "decay": args.decay,
                 "eta": args.eta, "beta": args.beta,
                 "core_ratio": 0.25, "alpha_min": 0.3, "alpha_max": 0.7}

    base_right = 0
    field_right = 0
    agree = 0
    t0 = time.time()

    for i, e in enumerate(test):
        text = e["text"]
        if text.endswith("<|im_end|>"):
            text = text[: -len("<|im_end|>")]
        cut = text.rfind("</think>")
        if cut != -1:
            text = text[: cut + len("</think>") + 1]
        ids = hf.encode(text, add_special_tokens=False).ids

        # One whole-sequence pass gives the anchor: the normed final hidden
        # of the last position, exactly what the head reads in the kernel.
        cap = {}
        ref35.forward(ids, tensors, cfg, capture=cap)
        h = np.asarray(cap["final_norm"][0])

        rng = np.random.RandomState(args.seed * 100003 + i)

        # baseline: score the model's own hidden
        base_logits = head_w @ rmsnorm_plain(h, eps)
        base_scores = score_actions(base_logits, table)
        base_pick = names[int(np.argmax(base_scores))]

        # field: refine, then score the fittest agent
        best_agent, diag = globular_refine(h, head_w, table, field_cfg, rng)
        field_logits = head_w @ rmsnorm_plain(best_agent, eps)
        field_scores = score_actions(field_logits, table)
        field_pick = names[int(np.argmax(field_scores))]

        base_right += base_pick == e["action"]
        field_right += field_pick == e["action"]
        agree += base_pick == field_pick
        if i < 8:
            m1 = "ok " if base_pick == e["action"] else "-- "
            m2 = "ok " if field_pick == e["action"] else "-- "
            print(f"  {m1}base {base_pick:10} {m2}field {field_pick:10} "
                  f"want {e['action']:10}")

    n = len(test)
    print(f"\n  baseline (h as-is)      : {base_right}/{n} = {base_right/n:6.1%}")
    print(f"  globular field (K={args.agents}, S={args.steps}) : "
          f"{field_right}/{n} = {field_right/n:6.1%}")
    print(f"  agreement between picks : {agree}/{n}")
    print(f"  elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()


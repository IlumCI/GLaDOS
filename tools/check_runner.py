#!/usr/bin/env python3
"""Check evaluate.Runner against reference.forward, and see what the model wants.

evaluate.py rewrote the forward pass to be incremental and vectorised, so that
a KV cache could be snapshotted and reused across items. That is a different
implementation from reference.forward, and reference.forward is the one already
proven equal to the kernel -- so before believing any accuracy number out of
the evaluator, the two have to agree.

The second half prints the unconstrained top tokens for a real selection
prompt. It separates two very different diagnoses: if the model's own top
choices are sensible tool names then the constrained decoder is at fault, and
if they are unrelated then the prompt or the forward pass is.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as ev  # noqa: E402
from reference import forward, load  # noqa: E402

MODEL = "tools/out/smollm2-135m-q8.bin"
TOKENIZER = "tools/out/smollm2-tok.bin"
ev.HF_TOKENIZER = "tools/hf/tokenizer.json"


def main():
    cfg, w, kv = load(MODEL)
    tok = ev.Tok(TOKENIZER)

    ids = [1, 9690, 1355]
    want = forward(cfg, w, kv, ids)

    runner = ev.Runner(cfg, w, kv)
    got = runner.feed(ids)

    delta = float(np.abs(want - got).max())
    scale = float(np.abs(want).max())
    print(f"  reference vs Runner: max |delta| {delta:.5f} on logits up to {scale:.2f}")
    print(f"  reference top5 {np.argsort(-want)[:5].tolist()}")
    print(f"  runner    top5 {np.argsort(-got)[:5].tolist()}")
    agree = np.array_equal(np.argsort(-want)[:5], np.argsort(-got)[:5])
    print(f"  same ordering: {agree}\n")
    if not agree:
        print("  the evaluator's forward pass is wrong; its accuracies mean nothing")
        return

    # Snapshot/restore has to be exact too, or every item after the first is
    # conditioned on leftover state from the previous one.
    runner2 = ev.Runner(cfg, w, kv)
    runner2.feed(ids[:1])
    snap = runner2.snapshot()
    a = runner2.feed(ids[1:])
    runner2.restore(snap)
    b = runner2.feed(ids[1:])
    print(f"  snapshot/restore reproduces logits: {np.array_equal(a, b)}\n")

    # What does the model actually want to say?
    prompt = ev.zero_shot_prefix() + ev.suffix("delete the file")
    r = ev.Runner(cfg, w, kv, max_len=1024)
    logits = r.feed(ev.encode(tok, prompt))
    print(f"  prompt is {len(ev.encode(tok, prompt))} tokens")
    print("  unconstrained top 10 continuations:")
    for i in np.argsort(-logits)[:10]:
        piece = bytes(tok.vocab[int(i)])
        print(f"    id {int(i):6}  {piece!r:24} logit {float(logits[i]):7.3f}")

    # And what the constrained decoder is limited to at step one.
    alphabet = ev.build_alphabet(tok)
    alts = [n.encode() + b"\n" for n in ev.NAMES]
    first = []
    for i, piece in enumerate(alphabet):
        if not piece:
            continue
        p = piece[1:] if piece[:1] == b" " else piece
        if not p:
            continue
        if any(a.startswith(p) and len(a) >= len(p) for a in alts):
            first.append(i)
    print(f"\n  {len(first)} tokens admissible at step 1")
    ranked = sorted(first, key=lambda i: -logits[i])[:10]
    print("  best admissible:")
    for i in ranked:
        print(f"    id {i:6}  {bytes(tok.vocab[i])!r:24} logit {float(logits[i]):7.3f}")


if __name__ == "__main__":
    main()

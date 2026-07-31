#!/usr/bin/env python3
"""Measure how well SmolLM2 picks a sysbox applet, before deciding what the
kernel should do.

Three strategies, on the held-out split:

  knn        nearest training example by pooled embedding. No transformer at
             all. This is the ceiling of what the trained head can reach, since
             the head is a linear map over the same features.
  zero-shot  the model reads the tool list and constrained-decodes a name.
  few-shot   the same, with retrieved examples in the prompt.

The point is to find out whether the trained head is still worth having now
that a model exists which can follow an instruction. It was built when the only
option was a story generator whose features had a measured separation gap of
zero; that is no longer the situation, and carrying the head forward on
momentum would be the wrong call.

The tool list is identical for every item, so it is prefilled once and the KV
cache is reused -- the same trick the kernel does with `ctx save`. Without it
this spends 250 tokens of prefill per item and takes twenty times as long.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from reference import load, rmsnorm  # noqa: E402

# Mirrors sysbox::APPLETS. Kept here rather than parsed out of the Rust so the
# evaluation can run without a build, at the cost of having to stay in step.
APPLETS = [
    ("sysbox", False, "this list"),
    ("ls", False, "list a directory"),
    ("cd", False, "change directory"),
    ("pwd", False, "print working directory"),
    ("tree", False, "list recursively"),
    ("cat", False, "print a file"),
    ("stat", False, "address, kind, size"),
    ("hash", False, "content address"),
    ("same", False, "compare two subtrees in one step"),
    ("du", False, "apparent bytes vs bytes that exist"),
    ("find", False, "search names and content"),
    ("diff", False, "compare snapshots"),
    ("snaps", False, "list snapshots"),
    ("fsck", False, "verify every stored object"),
    ("mkdir", True, "create a directory and its parents"),
    ("write", True, "write a file"),
    ("rm", True, "detach a name; content survives"),
    ("mv", True, "rename"),
    ("cp", True, "copy; constant time at any size"),
    ("snap", True, "commit the working tree as a snapshot"),
    ("back", True, "load a past snapshot into the working tree"),
]
NAMES = [a[0] for a in APPLETS]


# --- tokenizer (the converted v2 file, so this uses what the kernel uses) ---

class Tok:
    def __init__(self, path):
        b = Path(path).read_bytes()
        assert b[:8] == b"GLADOSTK", "not a v2 tokenizer"
        (ver, size, self.max_len, self.flags, self.bos, self.eos, self.unk) = struct.unpack_from(
            "<IIIIIII", b, 8
        )
        o = 36
        self.byte_table = list(struct.unpack_from(f"<{256}I", b, o))
        o += 256 * 4
        (n_spec,) = struct.unpack_from("<I", b, o)
        o += 4
        self.specials = list(struct.unpack_from(f"<{n_spec}I", b, o)) if n_spec else []
        o += n_spec * 4
        self.vocab, self.scores = [], []
        for _ in range(size):
            score, ln = struct.unpack_from("<fI", b, o)
            o += 8
            self.vocab.append(b[o:o + ln])
            self.scores.append(score)
            o += ln
        self.lookup = {}
        for i, v in enumerate(self.vocab):
            self.lookup.setdefault(v, i)


def encode(tok, text):
    """Uses the reference library, already proven equal to the kernel's."""
    from tokenizers import Tokenizer as HFTok
    global _HF
    try:
        _HF
    except NameError:
        _HF = HFTok.from_file(str(HF_TOKENIZER))
    return _HF.encode(text, add_special_tokens=False).ids


# --- incremental forward ------------------------------------------------

class Runner:
    def __init__(self, cfg, w, kv, max_len=512):
        self.cfg, self.w, self.kv = cfg, w, kv
        self.head_size = cfg["dim"] // cfg["heads"]
        self.kv_mul = cfg["heads"] // cfg["kv_heads"]
        self.k = np.zeros((cfg["layers"], max_len, kv), dtype=np.float32)
        self.v = np.zeros((cfg["layers"], max_len, kv), dtype=np.float32)
        self.pos = 0
        # RoPE angles do not depend on the token, only the position, so they
        # are computed once instead of per layer per token.
        d = cfg["dim"]
        hd = np.arange(0, d, 2) % self.head_size
        self.freq = 1.0 / (cfg["theta"] ** (hd / self.head_size))

    def snapshot(self):
        return (self.k[:, : self.pos].copy(), self.v[:, : self.pos].copy(), self.pos)

    def restore(self, snap):
        k, v, pos = snap
        self.k[:, :pos] = k
        self.v[:, :pos] = v
        self.pos = pos

    def feed(self, tokens):
        logits = None
        for t in tokens:
            logits = self._step(t)
        return logits

    def _step(self, token):
        cfg, w, kv = self.cfg, self.w, self.kv
        d, hs = cfg["dim"], self.head_size
        pos = self.pos
        ang = pos * self.freq
        cos, sin = np.cos(ang), np.sin(ang)

        x = w["embed"][token].copy()
        for li in range(cfg["layers"]):
            xb = rmsnorm(x, w["rms_att"][li])
            q = w["wq"][li] @ xb
            k = w["wk"][li] @ xb
            v = w["wv"][li] @ xb

            q0, q1 = q[0::2].copy(), q[1::2].copy()
            q[0::2] = q0 * cos - q1 * sin
            q[1::2] = q0 * sin + q1 * cos
            nk = kv // 2
            k0, k1 = k[0::2].copy(), k[1::2].copy()
            k[0::2] = k0 * cos[:nk] - k1 * sin[:nk]
            k[1::2] = k0 * sin[:nk] + k1 * cos[:nk]

            self.k[li, pos] = k
            self.v[li, pos] = v

            out = np.empty(d, dtype=np.float32)
            scale = 1.0 / np.sqrt(hs)
            for h in range(cfg["heads"]):
                qo = h * hs
                ko = (h // self.kv_mul) * hs
                K = self.k[li, : pos + 1, ko:ko + hs]
                s = (K @ q[qo:qo + hs]) * scale
                s = np.exp(s - s.max())
                s /= s.sum()
                out[qo:qo + hs] = s @ self.v[li, : pos + 1, ko:ko + hs]

            x = x + w["wo"][li] @ out
            xb = rmsnorm(x, w["rms_ffn"][li])
            hb = w["w1"][li] @ xb
            hb = hb / (1.0 + np.exp(-hb)) * (w["w3"][li] @ xb)
            x = x + w["w2"][li] @ hb

        self.pos += 1
        self.hidden = rmsnorm(x, w["rms_final"])
        return w["wcls"] @ self.hidden


# --- constrained decoding ----------------------------------------------

def build_alphabet(tok):
    return [bytes(v) for v in tok.vocab]


def constrained_pick(runner, tok, alphabet, allowed_names, logits):
    """Decode an applet name, admitting only tokens that keep one reachable."""
    alts = [n.encode() + b"\n" for n in allowed_names]
    produced = b""
    started = False
    for _ in range(16):
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
        best = max(cands, key=lambda i: logits[i])
        piece = alphabet[best]
        if not started:
            if piece[:1] == b" ":
                piece = piece[1:]
            started = True
        produced += piece
        for a, name in zip(alts, allowed_names):
            if produced == a:
                return name
        logits = runner.feed([best])
    return None


# --- strategies ---------------------------------------------------------

def tool_list():
    return "\n".join(f"{n} - {h}" for n, _, h in APPLETS)


def zero_shot_prefix():
    return (
        "<|im_start|>user\n"
        "Choose exactly one tool for the task. Reply with only the tool name.\n\n"
        f"Tools:\n{tool_list()}\n\n"
    )


def few_shot_prefix(examples):
    body = "\n".join(f"Task: {e['task']}\nTool: {e['applet']}" for e in examples)
    return (
        "<|im_start|>user\n"
        "Choose exactly one tool for the task. Reply with only the tool name.\n\n"
        f"Tools:\n{tool_list()}\n\nExamples:\n{body}\n\n"
    )


def suffix(task):
    return f"Task: {task}<|im_end|>\n<|im_start|>assistant\n"


def log_softmax(v):
    m = v.max()
    return v - m - np.log(np.exp(v - m).sum())


def score_pick(runner, item_base, tok, names, prompt_logits):
    """Pick by likelihood instead of by decoding.

    Constrained decoding asks the model to *emit* a name, which it will not do:
    its top continuations for a selection prompt are 'You', 'The', 'If' -- it
    wants to explain, and every tool name sits about three logits below that.
    Restricting the sample to tool names therefore reads off a distribution
    that is mostly about prose style, which is why the answer barely moves with
    the task.

    Scoring asks a different question -- how likely is this exact name, given
    the prompt -- and never requires the model to have chosen the format. That
    is standard for classification with small models, and it is also far
    cheaper here: the prompt is already in the cache, so each candidate costs
    only its own one-to-three tokens.

    Normalised by length, or short names win purely for being short.
    """
    best_name, best = None, -1e30
    for name in names:
        runner.restore(item_base)
        ids = encode(tok, name)
        if not ids:
            continue
        logits = prompt_logits
        total = 0.0
        for t in ids:
            total += float(log_softmax(logits)[t])
            logits = runner.feed([t])
        total /= len(ids)
        if total > best:
            best, best_name = total, name
    return best_name


def pooled(w, tok, text):
    ids = encode(tok, text)
    if not ids:
        return np.zeros(w["embed"].shape[1], dtype=np.float32)
    return w["embed"][ids].mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tokenizer")
    ap.add_argument("dataset")
    ap.add_argument("--hf-tokenizer", default="tools/hf/tokenizer.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shots", type=int, default=8)
    args = ap.parse_args()

    global HF_TOKENIZER
    HF_TOKENIZER = args.hf_tokenizer

    cfg, w, kv = load(args.model)
    tok = Tok(args.tokenizer)
    alphabet = build_alphabet(tok)
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    train, test = data["train"], data["test"]
    if args.limit:
        test = test[: args.limit]

    print(f"  {len(test)} held-out items, {len(NAMES)} applets, chance {100/len(NAMES):.1f}%\n")

    # --- knn: no transformer, the ceiling for a linear head over these
    #     features ---
    tr_raw = np.stack([pooled(w, tok, e["task"]) for e in train])
    # Centred before comparing. Pooled embeddings share a large common
    # component -- every sentence contains the same function words -- and
    # without removing it every cosine is ~0.99 and the argmax is whichever
    # vector happens to be longest, giving the same neighbour for every query.
    centre = tr_raw.mean(axis=0)
    tr_vec = tr_raw - centre
    tr_vec /= np.linalg.norm(tr_vec, axis=1, keepdims=True) + 1e-9

    right = 0
    for e in test:
        q = pooled(w, tok, e["task"]) - centre
        q /= np.linalg.norm(q) + 1e-9
        nearest = train[int(np.argmax(tr_vec @ q))]["applet"]
        right += nearest == e["applet"]
    print(f"  knn (pooled embeddings, no transformer) : {right/len(test):6.1%}")

    # --- zero-shot ---
    for label, prefix, shots in (
        ("zero-shot constrained decode", zero_shot_prefix(), None),
        (f"few-shot ({args.shots}) constrained decode", None, args.shots),
    ):
        if shots is not None:
            # Same examples for every item, so the prefix can still be cached.
            picked = []
            seen = set()
            for e in train:
                if e["applet"] not in seen:
                    picked.append(e)
                    seen.add(e["applet"])
                if len(picked) >= shots:
                    break
            prefix = few_shot_prefix(picked)

        runner = Runner(cfg, w, kv, max_len=1024)
        runner.feed(encode(tok, prefix))
        base = runner.snapshot()

        decoded = 0
        scored = 0
        for i, e in enumerate(test):
            runner.restore(base)
            logits = runner.feed(encode(tok, suffix(e["task"])))
            item_base = runner.snapshot()

            got = constrained_pick(runner, tok, alphabet, NAMES, logits)
            decoded += got == e["applet"]

            runner.restore(item_base)
            got_s = score_pick(runner, item_base, tok, NAMES, logits)
            scored += got_s == e["applet"]

            if i < 6:
                m1 = "ok " if got == e["applet"] else "-- "
                m2 = "ok " if got_s == e["applet"] else "-- "
                print(f"      decode {m1}{str(got):8} score {m2}{str(got_s):8} "
                      f"want {e['applet']:7} | {e['task'][:38]}")
        print(f"  {label + ', decoded':50}: {decoded/len(test):6.1%}")
        print(f"  {label + ', scored':50}: {scored/len(test):6.1%}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Host-side measurement rails for the resident checkpoints.

Four tasks, one tool, two backends:

  mmlu   multiple choice, scored by comparing the logprob of the four letter
         continuations after "Answer:" -- one prefill per question, no
         generation. This is the cheap standard trick, not the official
         harness; the number is for tracking deltas, not for quoting against
         leaderboards.
  gsm8k  few-shot chain-of-thought, greedy generation, exact match on the
         last number. The expensive rail; keep --limit small.
  niah   synthetic needle-in-a-haystack at several context lengths and
         depths. Host-side this tops out around 2k tokens -- beyond that the
         numpy attention is quadratic and the point is better made by the
         kernel on real hardware anyway.
  route  the traces corpus (applet choice given a machine state), scored
         with the same constrained decode the kernel uses. This is the
         rail that decides whether the loop is worth iterating on.

Backends are chosen by the file's magic: GLADOSM2/3 dense through
reference.load and evaluate.Runner, GLADOSM4 hybrid through an incremental
runner written here that mirrors the kernel's State -- full layers keep a KV
cache, linear layers keep the fixed-size recurrent state and conv ring.
`--check` proves the incremental hybrid against ref35's whole-sequence
forward before anything is measured with it.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ref35  # noqa: E402
import v4  # noqa: E402
from evaluate import Tok, Runner as DenseRunner, build_alphabet, constrained_pick  # noqa: E402
from reference import load as dense_load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = Path.home() / ".cache/huggingface/hub"


# --- backends ---------------------------------------------------------------


def magic_of(path):
    return Path(path).read_bytes()[:8]


class Hybrid35:
    """One token per step for the Qwen3.5 hybrid, state held across steps.

    Written to agree with ref35.forward to numerical noise; `--check` is the
    proof, and everything downstream of it is measurement rather than
    architecture. Full-attention layers keep a KV cache; linear layers keep
    the recurrent state and the conv ring, neither of which grows with
    context -- the property the whole architecture exists for.
    """

    def __init__(self, tensors, cfg, max_len):
        self.t = tensors
        self.cfg = cfg
        self.pre = "model.language_model."
        self.max_len = max_len
        self.pos = 0
        hd = cfg["head_dim"]
        nkv = cfg["num_key_value_heads"]
        self.full_cache = {}
        self.lin = {}
        for i, kind in enumerate(cfg["layer_types"]):
            if kind == "full_attention":
                self.full_cache[i] = (
                    np.zeros((max_len, nkv, hd), dtype=np.float32),
                    np.zeros((max_len, nkv, hd), dtype=np.float32),
                )
            else:
                hk = cfg["linear_key_head_dim"]
                hv = cfg["linear_value_head_dim"]
                nv = cfg["linear_num_value_heads"]
                kern = cfg["linear_conv_kernel_dim"]
                qkv_dim = 3 * hk * cfg["linear_num_key_heads"]
                self.lin[i] = {
                    "state": np.zeros((nv, hk, hv), dtype=np.float64),
                    # Ring of the last kern-1 qkv rows; the conv sees them
                    # plus the current one and nothing after.
                    "ring": np.zeros((kern - 1, qkv_dim), dtype=np.float32),
                }
        self._rope = {}

    def w(self, name):
        return np.asarray(self.t[self.pre + name], dtype=np.float32)

    def rope(self, pos):
        if pos not in self._rope:
            self._rope[pos] = ref35.rope_tables(
                np.array([pos]), self.cfg["head_dim"],
                self.cfg["partial_rotary_factor"], self.cfg["rope_theta"])
        return self._rope[pos]

    def feed(self, tokens):
        logits = None
        for t in tokens:
            logits = self._step(t)
        return logits

    def reset(self):
        """Fresh context. Stale cache entries beyond `pos` are never read --
        attention slices to the live prefix -- so only the counters and the
        recurrent state need clearing."""
        self.pos = 0
        for _, st in self.lin.items():
            st["state"][:] = 0
            st["ring"][:] = 0
        self._rope.clear()

    def _step(self, token):
        cfg = self.cfg
        eps = cfg["rms_norm_eps"]
        x = np.asarray(self.t[self.pre + "embed_tokens.weight"][token],
                       dtype=np.float32).copy()
        for i, kind in enumerate(cfg["layer_types"]):
            lp = f"layers.{i}."
            # The mixer sees the *normed* stream; the residual stays raw.
            # Omitting this was wrong from layer 0 with no downstream
            # complaint -- the exact failure mode the fixture exists for.
            xn = ref35.rms_norm(x[None, :],
                                self.w(lp + "input_layernorm.weight"), eps)[0]
            if kind == "full_attention":
                x = x + self._full(xn, lp, i)
            else:
                x = x + self._linear(xn, lp, i)
            xb = ref35.rms_norm(x[None, :],
                                self.w(lp + "post_attention_layernorm.weight"), eps)[0]
            h1 = (ref35.silu(xb @ self.w(lp + "mlp.gate_proj.weight").T)
                  * (xb @ self.w(lp + "mlp.up_proj.weight").T))
            x = x + h1 @ self.w(lp + "mlp.down_proj.weight").T
        # Advance once per token, after every layer has used the position.
        # Without this each token attended at position 0 and the KV cache
        # wrote over itself -- fluent nonsense, in the usual way.
        self.pos += 1
        h = ref35.rms_norm(x[None, :], self.w("norm.weight"), eps)
        head = self.t.get("lm_head.weight")
        W = (np.asarray(head, np.float32) if head is not None
             else self.w("embed_tokens.weight"))
        # 1-D, like the dense runner's contract: rails index logits[j] as a
        # flat vocabulary row.
        return (h @ W.T)[0]

    def _full(self, h, lp, li):
        cfg = self.cfg
        hd = cfg["head_dim"]
        nq = cfg["num_attention_heads"]
        nkv = cfg["num_key_value_heads"]
        eps = cfg["rms_norm_eps"]
        w = lambda n: self.w(lp + n)

        qg = (h @ w("self_attn.q_proj.weight").T).reshape(nq, hd * 2)
        q, gate = qg[:, :hd], qg[:, hd:].reshape(nq * hd)
        q = ref35.rms_norm(q[None, :], w("self_attn.q_norm.weight"), eps)[0]
        k = ref35.rms_norm(
            (h @ w("self_attn.k_proj.weight").T).reshape(1, nkv, hd),
            w("self_attn.k_norm.weight"), eps)[0]
        v = (h @ w("self_attn.v_proj.weight").T).reshape(nkv, hd)

        cos, sin = self.rope(self.pos)
        q = ref35.apply_rope(q[None, :, :], cos, sin)[0]
        k = ref35.apply_rope(k[None, :, :], cos, sin)[0]

        kc, vc = self.full_cache[li]
        kc[self.pos], vc[self.pos] = k, v
        n = self.pos + 1

        groups = nq // nkv
        scale = hd ** -0.5
        out = np.empty((nq, hd), dtype=np.float32)
        for hh in range(nq):
            K = kc[:n, hh // groups]
            s = (K @ q[hh]) * scale
            s = s - s.max()
            p = np.exp(s)
            p /= p.sum()
            out[hh] = p @ vc[:n, hh // groups]

        ogate = out.reshape(nq * hd) * ref35.sigmoid(gate)
        return ogate @ w("self_attn.o_proj.weight").T

    def _linear(self, h, lp, li):
        cfg = self.cfg
        hk = cfg["linear_key_head_dim"]
        hv = cfg["linear_value_head_dim"]
        nk = cfg["linear_num_key_heads"]
        nv = cfg["linear_num_value_heads"]
        kdim = hk * nk
        kern = cfg["linear_conv_kernel_dim"]
        eps = cfg["rms_norm_eps"]
        w = lambda n: self.w(lp + n)
        st = self.lin[li]

        qkv = h @ w("linear_attn.in_proj_qkv.weight").T
        z = h @ w("linear_attn.in_proj_z.weight").T
        b = h @ w("linear_attn.in_proj_b.weight").T
        a = h @ w("linear_attn.in_proj_a.weight").T

        cw = w("linear_attn.conv1d.weight")[:, 0, :]
        conv = np.zeros(qkv.shape, dtype=np.float32)
        for j in range(kern):
            row = st["ring"][j] if j < kern - 1 else qkv
            conv += row * cw[:, j]
        conv = ref35.silu(conv)
        # The current row joins the ring for the token after this one.
        st["ring"][:-1] = st["ring"][1:]
        st["ring"][-1] = qkv

        q = conv[:kdim].reshape(nk, hk)
        k = conv[kdim:2 * kdim].reshape(nk, hk)
        v = conv[2 * kdim:].reshape(nv, hv)

        beta = ref35.sigmoid(b)
        A = np.exp(w("linear_attn.A_log").astype(np.float64))
        g = -A * ref35.softplus(a.astype(np.float64)
                                + w("linear_attn.dt_bias").astype(np.float64))

        if nv // nk > 1:
            q = np.repeat(q, nv // nk, axis=0)
            k = np.repeat(k, nv // nk, axis=0)

        q = ref35.l2norm(q[None, :])[0] * (hk ** -0.5)
        k = ref35.l2norm(k[None, :])[0]

        gt = np.exp(g)[:, None, None]
        bt = beta[:, None]
        kt = k.astype(np.float64)
        vt = v.astype(np.float64)
        state = st["state"] * gt
        kv_mem = (state * kt[:, :, None]).sum(axis=1)
        delta = (vt - kv_mem) * bt
        st["state"] = state + kt[:, :, None] * delta[:, None, :]
        out = (st["state"] * q.astype(np.float64)[:, :, None]).sum(axis=1)

        core = out.reshape(nv, hv).astype(np.float32)
        core = ref35.rms_norm_gated(
            core, w("linear_attn.norm.weight"),
            z.reshape(nv, hv), eps)
        return core.reshape(nv * hv) @ w("linear_attn.out_proj.weight").T


def dequantize(v):
    """reference.load keeps int8 tensors as (data, per-row scales) tuples,
    per layer. evaluate.Runner predates that format and wants plain floats;
    the whole 135M checkpoint in f32 is ~540 MB, which the host can afford
    and the kernel cannot -- that difference is why this is the host rail."""
    if (isinstance(v, tuple) and len(v) == 2
            and hasattr(v[0], "astype") and hasattr(v[1], "astype")
            and v[0].ndim == 2 and v[1].ndim == 1
            and v[0].shape[0] == v[1].shape[0]):
        return v[0].astype(np.float32) * v[1][:, None]
    if isinstance(v, dict):
        return {k: dequantize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(dequantize(x) for x in v)
    return v


def make_backend(model_path, max_len):
    """Returns (runner, note). runner.feed(tokens) -> logits of the last token.

    Dense and hybrid share the GLADOSM2 magic; the version field at offset 8
    is what separates them (4 = v4 hybrid). Dispatching on magic alone sent
    the hybrid into the dense loader, which refused with a version error --
    caught here rather than in a measurement, which is the point of saying
    so."""
    import struct
    with open(model_path, "rb") as f:
        head = f.read(16)
    magic, version = head[:8], struct.unpack_from("<I", head, 8)[0]
    if magic == b"GLADOSM2" and version >= 4:
        tensors, cfg = v4.load(model_path)
        note = f"hybrid arch {cfg['arch']}, {len(cfg['layer_types'])} layers"
        return Hybrid35(tensors, cfg, max_len), note
    cfg, w = dense_load(str(model_path))
    w = dequantize(w)
    kv = cfg["dim"] // cfg["heads"] * cfg["kv_heads"]
    note = f"dense dim {cfg['dim']}, {cfg['layers']} layers"
    return DenseRunner2(cfg, w, kv, max_len=max_len), note


class DenseRunner2(DenseRunner):
    def reset(self):
        self.pos = 0

    """evaluate.Runner with reference.py's *current* signatures.

    evaluate.py predates two changes to reference.py -- load() returning two
    values and rmsnorm() taking eps -- and has not been run since. Rather
    than patch it call-site by call-site, the step lives here where the eps
    comes from the model's own config. Flagged in the commit so the staleness
    is somebody's known problem instead of a surprise.
    """

    def _step(self, token):
        cfg, w, kv = self.cfg, self.w, self.kv
        d, hs = cfg["dim"], self.head_size
        eps = cfg["eps"]
        pos = self.pos
        ang = pos * self.freq
        cos, sin = np.cos(ang), np.sin(ang)

        def rn(x, weight):
            return x / np.sqrt((x * x).mean() + eps) * weight

        x = w["embed"][token].copy()
        for li in range(cfg["layers"]):
            xb = rn(x, w["rms_att"][li])
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
            xb = rn(x, w["rms_ffn"][li])
            hb = w["w1"][li] @ xb
            hb = hb / (1.0 + np.exp(-hb)) * (w["w3"][li] @ xb)
            x = x + w["w2"][li] @ hb

        self.pos += 1
        self.hidden = rn(x, w["rms_final"])
        return w["wcls"] @ self.hidden


def check_hybrid(model_path, n=24):
    """Incremental vs whole-sequence, at every position. The gate the rest
    of this tool stands on."""
    tensors, cfg = v4.load(model_path)
    tokens = [760, 6511, 314, 9338, 369, 271, 4870, 432, 315, 6790,
              13, 1079, 527, 4902, 11, 1079, 527, 264, 6790, 13,
              946, 1122, 527, 4912][:n]
    whole = np.asarray(ref35.forward(tokens, tensors, cfg))
    inc = Hybrid35(tensors, cfg, max_len=len(tokens) + 8)
    worst = 0.0
    for pos in range(len(tokens)):
        got = inc._step(tokens[pos])
        want = whole[pos]
        d = float(np.abs(got - want).max())
        rel = d / max(float(np.abs(want).max()), 1e-9)
        worst = max(worst, rel)
    print(f"[check] hybrid incremental vs whole-sequence, {n} positions: "
          f"worst rel {worst:.3e}")
    if worst > 2e-3:
        raise SystemExit("incremental runner disagrees -- do not measure with it")
    print("[check] ok")


# --- data -------------------------------------------------------------------


def snapshot_dir(name):
    base = HF_CACHE / name / "snapshots"
    snaps = sorted(p for p in base.iterdir() if p.is_dir())
    if not snaps:
        raise SystemExit(f"dataset not cached: {name}")
    return snaps[0]


def parquet_rows(path):
    import pyarrow.parquet as pq
    return pq.read_table(str(path)).to_pylist()


def find_file(dirpath, prefix):
    hits = sorted(Path(dirpath).rglob(f"{prefix}*.parquet"))
    if not hits:
        raise SystemExit(f"no {prefix} parquet under {dirpath}")
    return hits[0]


# --- rails ------------------------------------------------------------------


def letter_ids(hf, letters=(" A", " B", " C", " D")):
    ids = []
    for s in letters:
        t = hf.encode(s, add_special_tokens=False).ids
        if len(t) != 1:
            raise SystemExit(f"letter continuation {s!r} is not one token: {t}")
        ids.append(t[0])
    return ids


def run_mmlu(backend, hf, limit):
    d = snapshot_dir("datasets--cais--mmlu")
    rows = parquet_rows(find_file(d, "test"))[: limit or 100]
    lids = letter_ids(hf)
    right = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        backend.reset()
        prompt = (f"The following are multiple choice questions (with answers) "
                  f"about {r['subject']}.\n\nQuestion: {r['question']}\n"
                  f"A. {r['choices'][0]}\nB. {r['choices'][1]}\n"
                  f"C. {r['choices'][2]}\nD. {r['choices'][3]}\nAnswer:")
        logits = backend.feed(hf.encode(prompt, add_special_tokens=False).ids)
        pick = int(np.argmax([logits[j] for j in lids]))
        right += pick == r["answer"]
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"  [{i + 1}/{len(rows)}] acc {right / (i + 1):6.1%}  "
                  f"({(time.time() - t0) / (i + 1):.1f}s/q)")
    print(f"  mmlu (0-shot letter logprob, n={len(rows)}): {right / len(rows):6.1%}")
    return right / len(rows)


def detok(tok, ids):
    return b"".join(bytes(tok.vocab[i]) for i in ids).decode("utf-8", "replace")


NUM_RE = re.compile(r"(-?[\d,]+(?:\.\d+)?)")


def last_number(text):
    hits = NUM_RE.findall(text.replace(",", ""))
    return hits[-1] if hits else None


def gold_number(answer):
    return last_number(answer.split("####")[-1])


def run_gsm8k(backend, hf, tok, limit, max_new):
    d = snapshot_dir("datasets--gsm8k")
    train = parquet_rows(find_file(d, "train"))
    test = parquet_rows(find_file(d, "test"))[: limit or 25]

    shots = train[:5]
    prefix = "".join(
        f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots)

    eos = tok.eos
    right = 0
    t0 = time.time()
    for i, r in enumerate(test):
        backend.reset()
        ids = hf.encode(prefix + f"Question: {r['question']}\nAnswer:",
                        add_special_tokens=False).ids
        logits = backend.feed(ids)
        gen = []
        for _ in range(max_new):
            nxt = int(np.argmax(logits))
            if nxt == eos:
                break
            gen.append(nxt)
            logits = backend.feed([nxt])
        text = detok(tok, gen)
        got = last_number(text)
        want = gold_number(r["answer"])
        hit = got is not None and want is not None and abs(float(got) - float(want)) < 1e-4
        right += hit
        if i < 5:
            print(f"      {'ok ' if hit else '-- '}got {got}  want {want}")
        print(f"  [{i + 1}/{len(test)}] acc {right / (i + 1):6.1%}  "
              f"({(time.time() - t0) / (i + 1):.1f}s/q)", end="\r")
    print()
    print(f"  gsm8k (5-shot greedy, n={len(test)}, <= {max_new} new): {right / len(test):6.1%}")
    return right / len(test)


FILLER = [
    "The old lighthouse keeper kept a journal of every ship that passed.",
    "Rain fell on the harbour in patterns the fishermen could read like text.",
    "A single gull circled the pier, uninterested in the day's catch.",
    "The market opened at six, and by seven the best stalls were taken.",
    "Somewhere inland a church bell counted an hour nobody had asked for.",
    "The tide brought in kelp, and the dogs argued with it at the shoreline.",
    "Every window on the street had its own opinion about the weather.",
    "The ferry ran late, as ferries do, and nobody was surprised by it.",
    "Bread, cheese and a knife: the whole picnic fit inside one basket.",
    "The map was older than the road it described, which explained a lot.",
]


def run_niah(backend, hf, tok, contexts, limit):
    """Needle at 25/50/75% depth; the model must recall a number it saw once.
    Filler is accumulated to a token budget, not a line count -- a line is
    roughly fourteen tokens, and conflating the two overshot every context
    by an order of magnitude on the first attempt."""
    rng = np.random.RandomState(7)
    results = {}
    for ctx in contexts:
        depths = [0.25, 0.5, 0.75] if ctx <= 1024 else [0.5]
        for depth in depths:
            backend.reset()
            word = f"gravel-{rng.randint(1000, 9999)}"
            magic = str(rng.randint(1000000, 9999999))
            needle = f"One of the special magic numbers for {word} is {magic}."
            question = (f"What is the special magic number for {word} "
                        f"mentioned in the text?\nAnswer:")
            budget = ctx - len(hf.encode(needle + question,
                                         add_special_tokens=False).ids) - 8
            ids = []
            i = 0
            while len(ids) < budget:
                line = FILLER[i % len(FILLER)]
                i += 1
                ids = hf.encode(line + "\n", add_special_tokens=False).ids + ids if not ids \
                    else ids + hf.encode(line + "\n", add_special_tokens=False).ids
            at = max(1, int(len(ids) * depth))
            needle_ids = hf.encode(needle + "\n", add_special_tokens=False).ids
            ids = ids[:at] + needle_ids + ids[at:]
            ids = ids[: ctx - 16]
            prompt_ids = ids + hf.encode("\n\n" + question,
                                         add_special_tokens=False).ids
            logits = backend.feed(prompt_ids)
            gen = []
            for _ in range(8):
                nxt = int(np.argmax(logits))
                gen.append(nxt)
                logits = backend.feed([nxt])
            text = detok(tok, gen)
            hit = magic in text
            results[(ctx, depth)] = hit
            print(f"  ctx {ctx:5d} depth {depth:4.0%} ({len(prompt_ids)} tok): "
                  f"{'ok  ' if hit else 'MISS'} (want {magic})")
    hits = sum(results.values())
    print(f"  niah (greedy, {hits}/{len(results)} found)")
    return hits / len(results)


def run_route(backend, hf, tok, alphabet, limit, shots=0):
    d = ROOT / "out" / "traces.jsonl"
    if not d.exists():
        raise SystemExit("no out/traces.jsonl -- run tools/traces.py first")
    items = [json.loads(l) for l in d.read_text(encoding="utf-8").splitlines() if l.strip()]
    test = [e for e in items if e.get("split") == "test"][: limit or 100]
    if not test:
        test = items[: limit or 100]
    # Few-shot exemplars come from the train split, whole traces including
    # their actions -- the model sees the mapping it is asked to perform, not
    # just the format. Fixed selection: first N distinct actions.
    prefix = ""
    if shots:
        picked, seen = [], set()
        for e in items:
            if e.get("split") != "train" or e["action"] in seen:
                continue
            picked.append(e)
            seen.add(e["action"])
            if len(picked) >= shots:
                break
        prefix = "".join(p["text"] + "\n" for p in picked)
    # The corpus's action space is the shell's command set plus the sysbox
    # applets -- not evaluate.NAMES, which is applets only and would exclude
    # the right answer for every trace that reaches a bare command.
    import traces
    names = traces.COMMANDS + traces.APPLETS
    right = 0
    t0 = time.time()
    for i, e in enumerate(test):
        backend.reset()
        text = e["text"]
        if text.endswith("<|im_end|>"):
            text = text[: -len("<|im_end|>")]
        # The corpus text includes the gold action after </think>; feeding it
        # would hand the model its own answer and then ask it to continue.
        # Cut at the end of reasoning so the decode *is* the choice.
        cut = text.rfind("</think>")
        if cut != -1:
            text = text[: cut + len("</think>") + 1]
        ids = hf.encode(prefix + text, add_special_tokens=False).ids
        logits = backend.feed(ids)
        got = constrained_pick(backend, tok, alphabet, names, logits)
        right += got == e["action"]
        if i < 5:
            print(f"      {'ok ' if got == e['action'] else '-- '}got {str(got):10} "
                  f"want {e['action']:10}")
        if (i + 1) % 25 == 0 or i + 1 == len(test):
            print(f"  [{i + 1}/{len(test)}] acc {right / (i + 1):6.1%}  "
                  f"({(time.time() - t0) / (i + 1):.1f}s/q)")
    print(f"  route (constrained decode, n={len(test)}, {len(names)} actions): "
          f"{right / len(test):6.1%}")
    return right / len(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tokenizer")
    ap.add_argument("--task", default="", choices=["", "mmlu", "gsm8k", "niah", "route"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--check", action="store_true",
                    help="prove the incremental hybrid against ref35, then exit")
    ap.add_argument("--hf-tokenizer", default="")
    args = ap.parse_args()

    if args.check:
        check_hybrid(args.model)
        return

    from tokenizers import Tokenizer as HFTok
    hf_path = args.hf_tokenizer
    if not hf_path:
        # The converted tokenizer is the kernel's; the HF one is the
        # reference's. Encoding uses the reference, detokenisation the kernel.
        hf_path = "tools/hf/tokenizer.json"
    hf = HFTok.from_file(hf_path)
    tok = Tok(args.tokenizer)

    max_len = {"mmlu": 2048, "gsm8k": 1024, "niah": max(args.contexts) + 64,
               "route": 2048}[args.task]
    backend, note = make_backend(args.model, max_len)
    print(f"[lm_eval] {Path(args.model).name}: {note}, task {args.task}")

    if args.task == "mmlu":
        run_mmlu(backend, hf, args.limit)
    elif args.task == "gsm8k":
        run_gsm8k(backend, hf, tok, args.limit, args.max_new)
    elif args.task == "niah":
        run_niah(backend, hf, tok, args.contexts, args.limit)
    elif args.task == "route":
        run_route(backend, hf, tok, build_alphabet(tok), args.limit)


if __name__ == "__main__":
    main()





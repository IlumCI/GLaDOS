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
import random
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
        self.latent_k = 0

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

    def snapshot(self):
        """Fork point: everything the state needs to explore a candidate and
        come back. Full layers copy their live KV prefix; linear layers copy
        the fixed-size state and ring. This is the kernel's mind-forking,
        host-side."""
        full = {i: (kc[: self.pos].copy(), vc[: self.pos].copy())
                for i, (kc, vc) in self.full_cache.items()}
        lin = {i: (st["state"].copy(), st["ring"].copy())
               for i, st in self.lin.items()}
        return (full, lin, self.pos)

    def restore(self, snap):
        full, lin, pos = snap
        for i, (kc, vc) in self.full_cache.items():
            k, v = full[i]
            kc[:pos], vc[:pos] = k, v
        for i, st in self.lin.items():
            st["state"], st["ring"] = lin[i][0].copy(), lin[i][1].copy()
        self.pos = pos

    def _step(self, token):
        cfg = self.cfg
        eps = cfg["rms_norm_eps"]
        x = np.asarray(self.t[self.pre + "embed_tokens.weight"][token],
                       dtype=np.float32).copy()
        x = self._body(x, commit=True)
        # Coconut-flavoured, training-free: K extra passes over the body with
        # the last hidden state as the input -- no token decoded, position not
        # advanced, recurrent state and KV prefix frozen. The refinement
        # happens in the continuous stream only. Whether an untrained
        # backbone benefits is exactly what the measurement is for.
        for _ in range(self.latent_k):
            x = self._body(x, commit=False)
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

    def _body(self, x, commit):
        cfg = self.cfg
        eps = cfg["rms_norm_eps"]
        for i, kind in enumerate(cfg["layer_types"]):
            lp = f"layers.{i}."
            # The mixer sees the *normed* stream; the residual stays raw.
            # Omitting this was wrong from layer 0 with no downstream
            # complaint -- the exact failure mode the fixture exists for.
            xn = ref35.rms_norm(x[None, :],
                                self.w(lp + "input_layernorm.weight"), eps)[0]
            if kind == "full_attention":
                x = x + self._full(xn, lp, i, commit)
            else:
                x = x + self._linear(xn, lp, i, commit)
            xb = ref35.rms_norm(x[None, :],
                                self.w(lp + "post_attention_layernorm.weight"), eps)[0]
            h1 = (ref35.silu(xb @ self.w(lp + "mlp.gate_proj.weight").T)
                  * (xb @ self.w(lp + "mlp.up_proj.weight").T))
            x = x + h1 @ self.w(lp + "mlp.down_proj.weight").T
        return x

    def _full(self, h, lp, li, commit=True):
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
        if commit:
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

    def _linear(self, h, lp, li, commit=True):
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
        # Frozen during latent passes: refinement reads the context, it
        # does not move it.
        if commit:
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
        new_state = state + kt[:, :, None] * delta[:, None, :]
        if commit:
            st["state"] = new_state
        state = new_state
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


def backend_vocab(backend):
    """How many rows the classifier has, whichever loader built it."""
    for probe in (
        lambda: backend.cfg["vocab_size"],
        lambda: backend.cfg["vocab"],
        lambda: backend.cfg["n_vocab"],
        lambda: backend.cfg["vocab_len"],
    ):
        try:
            v = probe()
            if isinstance(v, int) and v > 0:
                return v
        except Exception:
            pass
    return None


class DenseRef:
    """The dense path, borrowed from `tools/reference.py` rather than rebuilt.

    **There were two dense implementations here and only one of them was
    right.** This file carried its own, written for llama2: head width derived
    as `dim // heads`, RoPE pairing `2i` with `2i+1`, and no QK-Norm. Qwen3 has
    none of those -- it *states* a head width of 128 where the derivation gives
    64, it pairs `i` with `i + head_dim/2`, and it RMSNorms every head's query
    and key before the rotation. CLAUDE.md names all three as the mistakes that
    produce a model which loads, runs and writes confident nonsense.

    Here it did not even get that far: the shapes stopped agreeing and the
    dense path raised `operands could not be broadcast together`. Which means
    it has never produced a Qwen3 number at all, and any figure attributed to
    one did not come through this code.

    `reference.py` is the numeric oracle this project already checks the kernel
    against, and it has all three right. So the second implementation is gone
    rather than repaired -- `model.rs` makes the objection twice: two things
    that are supposed to agree do not stay agreeing.

    **And it is too slow to benchmark with**, which is a separate fact and
    took a while to be worth stating: the oracle walks positions in Python, so
    a 726-token 5-shot prefill is minutes and a full run is about ninety
    minutes. `fastdense.Dense` is the same arithmetic with the position loop
    turned into a matrix dimension, and it is the default here **only because
    `fastdense.py --check` runs it against this oracle on real ids and refuses
    to be used if the logits disagree.** `--oracle` takes this path instead,
    which is what a disagreement is diagnosed with.
    """

    def __init__(self, path, max_len):
        import reference as R

        self.R = R
        self.cfg, self.w = R.load(path)
        self.max_len = max_len
        self.reset()

    def reset(self):
        self.cache = self.R.new_cache(self.cfg, self.max_len)
        self.pos = 0

    def feed(self, ids):
        ids = list(ids)
        if self.pos + len(ids) > self.max_len:
            raise ValueError(
                f"{self.pos + len(ids)} tokens into a {self.max_len} context"
            )
        logits = self.R.forward(self.cfg, self.w, ids, self.cache, self.pos)
        self.pos += len(ids)
        return logits


def make_dense(path, max_len, oracle, batch=1, device="auto", dtype="f32",
               kv_dtype=None):
    """The fast runner, or the oracle it was proven against."""
    if oracle:
        return DenseRef(str(path), max_len)
    import fastdense

    return fastdense.Dense(str(path), max_len, batch=batch, device=device,
                           dtype=dtype, kv_dtype=kv_dtype, verbose=True)


def make_backend(model_path, max_len, oracle=False, batch=1,
                 device="auto", dtype="f32", kv_dtype=None):
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
    return make_dense(model_path, max_len, oracle, batch, device, dtype,
                      kv_dtype)


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


# Where a 5-shot completion stops being an answer.
#
# **Without this the score is read out of the wrong text.** Nothing makes a
# base model stop after one answer; it carries straight on with "Question: ..."
# and invents the next one, and `last_number` then returns a number out of
# whatever it drifted into. The stop is the same one every published GSM8K
# harness uses, and the blank line matters as much as the word because the
# few-shot prefix separates its examples with one.
STOP = ("\nQuestion:", "\n\n")


def cut_at_stop(text):
    for sep in STOP:
        j = text.find(sep)
        if j >= 0:
            text = text[:j]
    return text


def pick(rows, limit, seed):
    """`limit` of them, drawn rather than sliced.

    It was `rows[:limit]`, and a prefix is not a sample. Nothing suggests
    GSM8K's test set is ordered by difficulty, but "probably not ordered" is
    not a sampling method, and a figure from 25 items has to be able to say
    where the 25 came from. Seeded, so the same `--limit` is the same
    questions on every run and two checkpoints are compared on one set.
    """
    if not limit or limit >= len(rows):
        return list(rows), len(rows)
    idx = sorted(random.Random(seed).sample(range(len(rows)), limit))
    return [rows[i] for i in idx], len(rows)


def run_gsm8k(backend, hf, tok, limit, max_new, show=0, shots=5, batch=1, seed=0):
    d = snapshot_dir("datasets--gsm8k")
    train = parquet_rows(find_file(d, "train"))
    rows = parquet_rows(find_file(d, "test"))
    test, total = pick(rows, limit if limit is not None else 25, seed)

    shots = train[:shots]
    prefix = "".join(
        f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots)

    # A prompt that does not fit is a prompt whose front falls off, and what
    # falls off first is the examples that make it few-shot. Said out loud
    # rather than scored: a silent 0% looks exactly like a model that cannot
    # do arithmetic.
    room = getattr(backend, "max_len", None)
    if room:
        need = max(
            len(hf.encode(prefix + f"Question: {r['question']}\nAnswer:",
                          add_special_tokens=False).ids)
            for r in test) + max_new
        if need > room:
            print(f"  the longest prompt plus {max_new} new is {need} tokens "
                  f"and the context is {room} -- raise --seq or lower --max-new")
            return 0.0

    eos = tok.eos
    right = 0
    t0 = time.time()
    shown = 0
    done_n = 0
    # A backend that can hold several sequences at once decodes them together.
    # Decode is one token at a time whatever you do, so a lone sequence drags
    # every weight past the processor to compute one row; sixteen rows pay
    # that once. Measured on CPU at 8.8 tok/s for one against 40.2 for
    # sixteen. `Hybrid35` keeps a recurrent state per sequence and has no
    # batched form, so it takes the second path unchanged.
    wide = getattr(backend, "prefill", None)

    # Encoded once, up front, so the grouping below can see the lengths. It is
    # a tokenizer pass over 1,319 short strings and costs nothing next to one
    # forward pass.
    enc = [hf.encode(prefix + f"Question: {r['question']}\nAnswer:",
                     add_special_tokens=False).ids for r in test]

    # --- the shared prefix, and the reason it needs checking ----------------
    #
    # Every prompt begins with the same five worked examples, 688 of its 726
    # tokens, and prefilling them 1,319 times is the single largest cost in
    # the run. `Dense.hold_prefix` computes them once.
    #
    # **But `encode(a + b)` is not always `encode(a) + encode(b)`.** A
    # byte-level BPE merges across a boundary, and this project has already
    # paid for that once: `lex.rs` found `'what'` and `' what'` were different
    # tokens with wildly different document frequencies, and the fix was to
    # treat both sides the same way. Here the failure would be silent and
    # worse -- the held prefix would be a set of keys for a tokenisation the
    # prompt does not have. So the split is verified against every prompt and
    # abandoned entirely if any one of them disagrees.
    pre_ids = hf.encode(prefix, add_special_tokens=False).ids
    shared = all(ids[:len(pre_ids)] == pre_ids for ids in enc) if wide else False
    if wide and not shared:
        print("      the prompts do not all start with the encoded prefix, so "
              "it is recomputed per question (tokeniser boundary)")

    # A held prefix must sit at the same columns in every row, so a group has
    # to be one length -- see `Dense.hold_prefix`. Grouping by exact length
    # costs nothing and buys both that and zero padding.
    if wide and shared:
        by_len = {}
        for r, ids in zip(test, enc):
            by_len.setdefault(len(ids), []).append((r, ids))
        groups = [bucket[i:i + batch]
                  for bucket in by_len.values()
                  for i in range(0, len(bucket), batch)]
    elif wide and batch > 1:
        pairs = list(zip(test, enc))
        groups = [pairs[i:i + batch] for i in range(0, len(pairs), batch)]
    else:
        groups = [[(r, ids)] for r, ids in zip(test, enc)]

    if wide and shared:
        print(f"      prefix {len(pre_ids)} tok held once, "
              f"{len(groups)} group(s) of one length")

    for pair in groups:
        g = [r for r, _ in pair]
        prompts = [ids for _, ids in pair]
        backend.reset()
        if shown < show:
            print("      prompt %d tok, budget %d new"
                  % (max(len(p) for p in prompts), max_new))
        if wide:
            logits = (backend.prefill(prompts, prefix=pre_ids) if shared
                      else backend.prefill(prompts))
        else:
            logits = np.asarray([backend.feed(prompts[0])])
        B = len(g)
        gen = [[] for _ in range(B)]
        texts = [""] * B
        fin = [False] * B
        for _ in range(max_new):
            nxt = [int(np.argmax(logits[b])) for b in range(B)]
            for b in range(B):
                if fin[b]:
                    continue
                if nxt[b] == eos:
                    fin[b] = True
                    continue
                gen[b].append(nxt[b])
                texts[b] = detok(tok, gen[b])
                # Checked on the decoded text rather than on a token, because
                # a stop is a string and the tokeniser is free to split it
                # across two pieces. Re-decoding each step costs nothing next
                # to a forward pass.
                if any(sep in texts[b] for sep in STOP):
                    fin[b] = True
            if all(fin):
                break
            # A finished row is still stepped, with a token whose output is
            # thrown away. Dropping it from the batch would mean rebuilding
            # the cache around the hole, which costs more than the wasted
            # column -- and the batch ends when its *longest* answer does
            # either way.
            feed = [nxt[b] if not fin[b] else eos for b in range(B)]
            if wide:
                logits = backend.step(feed)
            else:
                logits = np.asarray([backend.feed([feed[0]])])

        for b, r in enumerate(g):
            text = cut_at_stop(texts[b])
            if shown < show:
                shown += 1
                print("      --- what it actually said (%d tok) ---" % len(gen[b]))
                print("      " + text.replace(chr(10), chr(10) + "      ")[:1400])
                print("      --- end ---")
            got = last_number(text)
            want = gold_number(r["answer"])
            hit = (got is not None and want is not None
                   and abs(float(got) - float(want)) < 1e-4)
            right += hit
            done_n += 1
            if done_n <= 5:
                print(f"      {'ok ' if hit else '-- '}got {got}  want {want}")
        print(f"  [{done_n}/{len(test)}] acc {right / done_n:6.1%}  "
              f"({(time.time() - t0) / done_n:.1f}s/q)", end="\r")
    print()
    frac = f" ({len(test)}/{total} = {len(test)/total:.1%} of the set)"
    print(f"  gsm8k ({len(shots)}-shot greedy, n={len(test)}{frac}, "
          f"<= {max_new} new): {right / len(test):6.1%}")
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
    ap.add_argument("--shots", type=int, default=0,
                    help="few-shot examples; gsm8k used a hardcoded 5 and "
                         "ignored this, which is 684 tokens of prompt")
    ap.add_argument("--latent", type=int, default=0,
                    help="Coconut-style frozen-state refinement passes per token (hybrid only)")
    # **64 was less than half an answer.** Measured over the test slice with
    # the harness's own tokeniser: a GSM8K answer is 133 tokens on average and
    # 213 at worst, so two thirds of every completion was being cut off before
    # it reached the line the score is read from -- and `last_number` then
    # returned a figure out of the middle of the reasoning. That is most of
    # why this task has read 0.0% on every checkpoint ever put through it.
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--show", type=int, default=0,
                    help="print the raw completion for the first N questions")
    ap.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--check", action="store_true",
                    help="prove the incremental hybrid against ref35, then exit")
    ap.add_argument("--batch", type=int, default=1,
                    help="decode this many questions at once. Decode is "
                         "memory-bound, so a batch is close to free per extra "
                         "sequence; dense backends only.")
    ap.add_argument("--all", action="store_true",
                    help="every item in the task's set rather than --limit")
    ap.add_argument("--seed", type=int, default=0,
                    help="which sample --limit draws. Fixed, so two runs and "
                         "two checkpoints see the same questions.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--kv-dtype", default=None, dest="kv_dtype",
                    choices=["f32", "bf16", "fp16"],
                    help="cache precision, separately from the weights. The "
                         "cache is what grows with batch (264 MB per sequence "
                         "here against 3.0 GB of weights), so this is the "
                         "knob that decides how big a batch fits.")
    ap.add_argument("--dtype", default="f32", choices=["f32", "bf16", "fp16"],
                    help="weight precision. f32 is what the oracle check "
                         "passes at; the narrow ones hold the argmax and "
                         "reorder the tail, so measure before quoting one.")
    ap.add_argument("--oracle", action="store_true",
                    help="run the dense path through reference.py itself -- "
                         "correct, and about 40x slower. For diagnosing a "
                         "disagreement, not for producing a figure.")
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

    # gsm8k was 1024 and the 5-shot prompt is 749 tokens on average, 800 at
    # worst -- which fit while the budget was 64 and stops fitting the moment
    # it is large enough to hold an answer. The two numbers have to move
    # together, so they are worked out together.
    max_len = {"mmlu": 2048, "gsm8k": 2048, "niah": max(args.contexts) + 64,
               "route": 2048}[args.task]
    # The KV cache is `layers * batch * max_len * kv_dim * 2`, so a batch
    # multiplies it: 2048 at batch 16 is 7.5 GB on this checkpoint and 1152
    # is 4.2. The 5-shot prompt is 800 tokens at worst and the budget is 256,
    # so 1152 has room and `need` below refuses loudly if it does not.
    if args.task == "gsm8k" and args.batch > 1:
        max_len = min(max_len, 1152)
    made = make_backend(args.model, max_len, args.oracle, batch=args.batch,
                        device=args.device, dtype=args.dtype,
                        kv_dtype=args.kv_dtype)
    backend, note = made if isinstance(made, tuple) else (
        made,
        f"dense dim {made.cfg['dim']}, {made.cfg['layers']} layers"
        + (", qk-norm" if made.cfg.get("qk_norm") else "")
        + f", head_dim {made.cfg['head_dim']}"
        + (", oracle" if args.oracle else ", batched"),
    )
    if hasattr(backend, "latent_k"):
        backend.latent_k = args.latent
        if args.latent:
            note += f", latent x{args.latent}"
    # **The encoder has to be the model's own, and nothing was checking.**
    #
    # `--hf-tokenizer` defaults to `tools/hf/tokenizer.json`, which is
    # SmolLM2's 49,152-token vocabulary. Handed a Qwen3.5 checkpoint, whose
    # vocabulary is 151,669, every prompt was encoded into ids that mean
    # something else entirely -- and the model answered the only way it could,
    # with a degenerate run of one token. It scored 0.0%, and that figure went
    # into this project's notes as a fact about the model.
    #
    # There is no way to notice this from the output: wrong ids produce
    # confident nonsense, not an error. So it is checked here, and the run is
    # refused rather than scored.
    mv = backend_vocab(backend)
    hv = hf.get_vocab_size()
    if mv and abs(mv - hv) > max(64, mv // 100):
        print(f"[lm_eval] the model's vocabulary is {mv} and "
              f"{Path(hf_path).parent.name}/{Path(hf_path).name} holds {hv}.")
        print("          Those are different tokenizers, so every prompt would be")
        print("          encoded into ids that mean something else. Pass")
        print("          --hf-tokenizer <the model's own tokenizer.json>.")
        sys.exit(2)
    if mv and tok.vocab and abs(len(tok.vocab) - mv) > max(64, mv // 100):
        print(f"[lm_eval] the model's vocabulary is {mv} and the kernel tokenizer "
              f"holds {len(tok.vocab)} -- detokenisation would be unreadable.")
        sys.exit(2)

    print(f"[lm_eval] {Path(args.model).name}: {note}, task {args.task}")

    if args.task == "mmlu":
        run_mmlu(backend, hf, args.limit)
    elif args.task == "gsm8k":
        run_gsm8k(backend, hf, tok, 0 if args.all else args.limit,
                  args.max_new, args.show,
                  args.shots if args.shots else 5,
                  batch=args.batch, seed=args.seed)
    elif args.task == "niah":
        run_niah(backend, hf, tok, args.contexts, args.limit)
    elif args.task == "route":
        run_route(backend, hf, tok, build_alphabet(tok), args.limit)


if __name__ == "__main__":
    main()






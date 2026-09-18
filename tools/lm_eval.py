#!/usr/bin/env python3
"""Host-side measurement rails for the resident checkpoints.

Five tasks, one tool, two backends:

  bpb    held-out bits per byte over a corpus pinned by content hash, which
         defaults to this kernel's own source. Dense: every token is an
         observation rather than one bit per question, so it resolves a
         change the binary rails need thousands of items to see. One forward
         pass per window and no generation. Read the note above `corpus_blob`
         before quoting one against a different checkpoint.

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
import forest_retrieve as fr  # noqa: E402
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
               kv_dtype=None, kv8=None):
    """The fast runner, or the oracle it was proven against."""
    if oracle:
        return DenseRef(str(path), max_len)
    import fastdense

    return fastdense.Dense(str(path), max_len, batch=batch, device=device,
                           dtype=dtype, kv_dtype=kv_dtype, kv8=kv8,
                           verbose=True)


def make_backend(model_path, max_len, oracle=False, batch=1,
                 device="auto", dtype="f32", kv_dtype=None, kv8=None):
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
                      kv_dtype, kv8)


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


# The block's preamble is part of the budget, so the string counted has to be
# the string rendered. `forest_retrieve.LABEL` is the default and this task
# wants its own, since what it retrieves is material rather than worked
# examples -- so it is named once and passed to both.
MMLU_LABEL = "Related material"


def mmlu_rows():
    """Every MMLU test question, all 57 subjects.

    **This rail scored `abstract_algebra` and called it MMLU, for its whole
    life.** `find_file` answers `sorted(rglob(...))[0]`, the snapshot has one
    directory per subject and no combined config, and `abstract_algebra` sorts
    first -- so `parquet_rows(find_file(d, "test"))` was 100 questions of
    undergraduate group theory. Every MMLU figure this project has recorded,
    including the 43.3% in `design/benchmarks.md`, is that.

    It is the fifth defect of the shape the evaluation-discipline section
    already lists four of: not a wrong answer, a rail quietly answering a
    different question from the one its name claims. The tell was available
    from the first run and nothing printed it, which is why the composition is
    printed now.
    """
    d = snapshot_dir("datasets--cais--mmlu")
    files = sorted(Path(d).rglob("test-*.parquet"))
    if not files:
        raise SystemExit(f"no MMLU test parquet under {d}")
    rows = []
    for f in files:
        rows.extend(parquet_rows(f))
    return rows, len(files)


def backend_score(backend, hf, text):
    """Summed NLL of `text` under the model, in nats. Fresh context each time.

    A thin wrapper so the cloze protocol reads the same whichever backend is
    underneath, and so `reset` can never be forgotten -- a stale KV prefix
    would score the choice against the previous question and the number would
    look entirely reasonable.
    """
    backend.reset()
    ids = hf.encode(text, add_special_tokens=False).ids
    return backend.score(ids)


def run_mmlu(backend, hf, limit, seed=0, dump="", forest=None, fbudget=0,
             fk=4, cloze=""):
    rows, nsub = mmlu_rows()
    total_all = len(rows)
    # Drawn rather than sliced, for the reason `pick` gives and for a sharper
    # one here: the rows arrive grouped by subject, so a prefix is one subject
    # and the next -- exactly the failure this function was built out of.
    rows, _ = pick(rows, limit if limit is not None else 100, seed)
    subj = {}
    for r in rows:
        subj[r["subject"]] = subj.get(r["subject"], 0) + 1
    print(f"  {total_all} question(s) over {nsub} subject(s); "
          f"scoring {len(rows)} over {len(subj)}")
    lids = letter_ids(hf)
    letters = "ABCD"
    right = 0
    right_cloze = 0
    right_norm = 0
    refused = 0
    record = []
    enc_len = lambda s: len(hf.encode(s, add_special_tokens=False).ids)
    t0 = time.time()
    for i, r in enumerate(rows):
        backend.reset()
        block = ""
        if forest is not None:
            nodes, ref = forest.retrieve(r["question"], k=fk, budget=fbudget,
                                         count=enc_len, label=MMLU_LABEL)
            refused += ref
            block = fr.render_block(nodes, MMLU_LABEL)
        prompt = (f"The following are multiple choice questions (with answers) "
                  f"about {r['subject']}.\n\n{block}Question: {r['question']}\n"
                  f"A. {r['choices'][0]}\nB. {r['choices'][1]}\n"
                  f"C. {r['choices'][2]}\nD. {r['choices'][3]}\nAnswer:")
        # --- the three protocols, on the same question ------------------
        #
        # **Letter-logprob asks the model to do the task and then two more
        # things.** It has to know the answer, map it to a position in a list,
        # and map that position to the token " A". The last two hops are
        # indirection with nothing to do with the subject, and indirection is
        # what a 0.6B is worst at -- so a model that knows perfectly well what
        # the ribosome does can still pick the wrong letter. It is the
        # standard protocol because it costs one prefill, which is a fact
        # about harness budgets rather than about measurement.
        #
        # The cloze protocol deletes both hops: score each choice as a
        # continuation of the question and take the likeliest. `acc_norm`
        # divides by the choice's own length, because summed log-probability
        # is monotonically punished by every extra token and the shortest
        # option would otherwise win by default.
        #
        # Five forward passes against one. All three are computed on the same
        # question so the comparison is paired rather than three runs.
        logits = backend.feed(hf.encode(prompt, add_special_tokens=False).ids)
        chose = int(np.argmax([logits[j] for j in lids]))
        hit = chose == r["answer"]
        right += hit

        if cloze:
            stem = (f"The following are multiple choice questions (with "
                    f"answers) about {r['subject']}.\n\n{block}"
                    f"Question: {r['question']}\nAnswer:")
            base, _ = backend_score(backend, hf, stem)
            nll, per = [], []
            for c in list(r["choices"])[:4]:
                tot, _ = backend_score(backend, hf, stem + " " + str(c))
                # The stem's own cost is the same for all four, so it cancels
                # in the ranking -- subtracted anyway because the normalised
                # variant needs the continuation's cost on its own.
                nll.append(tot - base)
                per.append(max(1, len(str(c).encode("utf-8"))))
            c_raw = int(np.argmin(nll))
            c_norm = int(np.argmin([n / b for n, b in zip(nll, per)]))
            right_cloze += c_raw == r["answer"]
            right_norm += c_norm == r["answer"]
            chose = c_norm if cloze == "norm" else chose

        record.append((qid(r), letters[r["answer"]], letters[chose],
                       int(bool(hit)), 1))
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            extra = ""
            if cloze:
                extra = (f"  cloze {right_cloze / (i + 1):6.1%}  "
                         f"norm {right_norm / (i + 1):6.1%}")
            print(f"  [{i + 1}/{len(rows)}] acc {right / (i + 1):6.1%}{extra}  "
                  f"({(time.time() - t0) / (i + 1):.1f}s/q)")
    if forest is not None:
        print(f"  {refused} node(s) refused for carrying the question "
              f"being scored")
    n = len(rows)
    print(f"  mmlu (0-shot letter logprob, n={n}): {right / n:6.1%}")
    if cloze:
        print(f"  mmlu (0-shot cloze, likeliest choice text): "
              f"{right_cloze / n:6.1%}")
        print(f"  mmlu (0-shot cloze, length-normalised):     "
              f"{right_norm / n:6.1%}")
    if dump:
        write_dump(dump, "mmlu", record)
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


# --- keeping the per-question record ---------------------------------------
#
# **A percentage is not enough to compare two configurations with.** Two runs
# at 39.2% and 37.4% differ by 24 questions out of 1,319, and the interval on
# each of those figures taken alone is wide enough to swallow the difference.
# Taken as pairs it is not: the same question under both configurations either
# agrees or it does not, and only the disagreements carry any information. That
# is McNemar's test, it needs per-question outcomes, and this harness was
# throwing them away the moment it printed an accuracy.
#
# So `--dump` writes one line per question, keyed by a hash of the question
# text rather than by its position. Position moves when `--limit` or `--seed`
# changes; the question does not, so two dumps taken at different samples still
# pair on whatever they have in common.
def qid(row):
    """A stable id for a question, keyed on what makes it distinct.

    **A hash of the question text alone paired 14,042 MMLU questions as
    13,869.** 173 stems repeat, and they repeat two ways: across subjects, and
    within one subject with *different choices* -- "what is the source of the
    material that causes meteor showers" appears twice in astronomy with two
    different option lists, and those are two questions. So the subject and
    the choices are in the key. The gold answer never is; an id that depends
    on the answer is an id you cannot compute without it.

    What still collapses is 27 rows that are byte-identical to another row in
    every field, and that is right: two copies of one question are not two
    independent observations, and pairing them as such would double-count.

    GSM8K rows carry neither field, so their ids are unchanged and every
    gsm8k dump written before this still pairs. An mmlu dump written before
    it does not, which is the honest outcome, since those dumps were short.
    """
    import hashlib
    key = row["question"]
    if row.get("subject"):
        key = row["subject"] + "\n" + key
    if row.get("choices") is not None:
        key += "\n" + "\n".join(str(c) for c in row["choices"])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def write_dump(path, task, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {task}\tqid\twant\tgot\thit\tntok\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"  per-question outcomes -> {path}")


def run_gsm8k(backend, hf, tok, limit, max_new, show=0, shots=5, batch=1,
              seed=0, dump="", forest=None, fbudget=0, fk=4):
    d = snapshot_dir("datasets--gsm8k")
    train = parquet_rows(find_file(d, "train"))
    rows = parquet_rows(find_file(d, "test"))
    test, total = pick(rows, limit if limit is not None else 25, seed)

    shots = train[:shots]
    prefix = "".join(
        f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots)

    # --- retrieval, if there is a forest ------------------------------------
    #
    # The retrieved block sits between the five-shot prefix and the question,
    # which keeps the prefix first and so keeps it shareable. It costs
    # grouping: prompts were all one length and now differ by whatever was
    # retrieved, so exact-length groups get smaller and the run gets slower.
    # That is the price of the measurement and it is stated rather than hidden.
    tails = [f"Question: {r['question']}\nAnswer:" for r in test]
    if forest is not None:
        enc_len = lambda s: len(hf.encode(s, add_special_tokens=False).ids)
        refused = 0
        got_n = 0
        for i, r in enumerate(test):
            nodes, ref = forest.retrieve(r["question"], k=fk,
                                         budget=fbudget, count=enc_len)
            refused += ref
            got_n += len(nodes)
            tails[i] = fr.render_block(nodes) + tails[i]
        print(f"      forest: {len(forest.nodes)} node(s), "
              f"{got_n / len(test):.1f} retrieved per question, "
              f"budget {fbudget or 'none'} tok")
        # Printed whatever it is, because zero is the only value that makes
        # the rest of the run mean anything -- see `forest_retrieve`'s guard.
        print(f"      {refused} node(s) refused for carrying the question "
              f"being scored")

    # A prompt that does not fit is a prompt whose front falls off, and what
    # falls off first is the examples that make it few-shot. Said out loud
    # rather than scored: a silent 0% looks exactly like a model that cannot
    # do arithmetic.
    room = getattr(backend, "max_len", None)
    if room:
        need = max(len(hf.encode(prefix + t, add_special_tokens=False).ids)
                   for t in tails) + max_new
        if need > room:
            print(f"  the longest prompt plus {max_new} new is {need} tokens "
                  f"and the context is {room} -- raise --seq or lower --max-new")
            if forest is not None:
                print("  (a retrieved block is in that prompt; --forest-budget "
                      "bounds it, and an unbounded one cannot be planned for)")
            return 0.0

    eos = tok.eos
    right = 0
    t0 = time.time()
    shown = 0
    done_n = 0
    record = []
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
    enc = [hf.encode(prefix + t, add_special_tokens=False).ids for t in tails]

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
            record.append((qid(r), want, got, int(bool(hit)), len(gen[b])))
            if done_n <= 5:
                print(f"      {'ok ' if hit else '-- '}got {got}  want {want}")
        print(f"  [{done_n}/{len(test)}] acc {right / done_n:6.1%}  "
              f"({(time.time() - t0) / done_n:.1f}s/q)", end="\r")
    print()
    frac = f" ({len(test)}/{total} = {len(test)/total:.1%} of the set)"
    print(f"  gsm8k ({len(shots)}-shot greedy, n={len(test)}{frac}, "
          f"<= {max_new} new): {right / len(test):6.1%}")
    if dump:
        write_dump(dump, "gsm8k", record)
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


def run_niah(backend, hf, tok, contexts, limit, new_tok=24):
    """Needle at 25/50/75% depth; the model must recall a number it saw once.
    Filler is accumulated to a token budget, not a line count -- a line is
    roughly fourteen tokens, and conflating the two overshot every context
    by an order of magnitude on the first attempt."""
    rng = np.random.RandomState(7)
    results = {}
    for ctx in contexts:
        # **Every context gets every depth.** This read
        # `[0.25, 0.5, 0.75] if ctx <= 1024 else [0.5]`, so the longest
        # context -- the only one where long-range recall is actually under
        # test -- contributed a single item, and a headline like "7/7" rested
        # on one observation at 2048 with six easy ones padding it out. That
        # was affordable caution when the runner was the NumPy oracle; on
        # `fastdense` a 4k prefill is under a second, so the cap now only
        # makes the evidence thinner than it looks.
        depths = [0.25, 0.5, 0.75]
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
            # **Budgeted in tokens, and the old budget was eight.** Qwen3's
            # pre-tokenizer takes digits *one at a time*, so the seven-digit
            # answer is seven tokens before any leading space or newline the
            # model chooses to emit first. Eight left no room to be wrong in,
            # and a truncated correct answer scores exactly like a wrong one.
            for _ in range(new_tok):
                nxt = int(np.argmax(logits))
                if nxt == tok.eos:
                    break
                gen.append(nxt)
                logits = backend.feed([nxt])
            text = detok(tok, gen)
            hit = magic in text
            results[(ctx, depth)] = hit
            # **Printed, always.** This rail read 0/7 for SmolLM2 and 0/6 for
            # Qwen3-0.6B and was quoted as a fact about the models, with
            # nothing anywhere recording what they actually said -- which is
            # precisely how GSM8K stayed at 0.0% for the life of the project.
            # Six items, so there is no reason to ever not show them.
            print(f"  ctx {ctx:5d} depth {depth:4.0%} ({len(prompt_ids)} tok): "
                  f"{'ok  ' if hit else 'MISS'} (want {magic})")
            print(f"        said ({len(gen)} tok): "
                  + repr(text)[:160])
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


# --- the compression rail ---------------------------------------------------
#
# **Every rail here is coarse, and that is the thing that slows the work
# down.** GSM8K needed all 1,319 questions to resolve the int8 cache at chi
# 3.86 against a bar of 3.84. Retrieval on MMLU moved 39 of 100 answers and
# netted one. The latent sweep read 34/38/32/42/34 at n=50, which is noise
# against a standard error of 7. In each case the machine plainly did
# something and the instrument could not say what.
#
# Held-out log-loss is the dense alternative: every token is an observation
# rather than one bit per question, it needs one forward pass and no
# generation, and it cannot leak an answer key because there is no answer.
# Huang et al. (COLM 2024) measured bits-per-character against twelve
# benchmarks over 31 base models and report about -0.93, and -0.92 against
# GSM8K specifically.
#
# **The caveat that applies to this checkpoint in particular.** That paper
# excluded the Qwen series from its maths fit as outliers, inferring GSM8K
# and MATH training-data exposure. Qwen3-0.6B is what runs here. So the
# published correlation is evidence that the rail is worth having and is not
# evidence about how it behaves on a contaminated model, and the way to
# settle that is to check whether this rail agrees with GSM8K on a change
# both can see. `--kv8` is the first such change.
#
# **Bits per byte and never bits per token.** A token is a property of the
# tokenizer, so a per-token figure cannot compare two checkpoints with
# different vocabularies -- and comparing checkpoints is most of why a rail
# exists. Bytes are a property of the text.

# What a corpus directory is walked for.
BPB_GLOB = "*.rs"


def corpus_blob(path, pattern=BPB_GLOB):
    """The exact bytes to be scored, and their content address.

    **A rail whose corpus can move is the "test set that moved" failure in a
    new costume**, which `CLAUDE.md` records as one of the three ways
    measurement was got wrong here. `src/` is under active development, so
    two runs a day apart would score different text and the difference would
    be reported as a change in the model. The hash is printed on every run
    and `paired.py` refuses to compare two runs that do not share it.

    Files are joined in sorted order with a single newline and no path
    headers. A header would be a thing the model can learn to predict cheaply
    and would flatter the figure.
    """
    import hashlib
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.rglob(pattern))
    if not files:
        raise SystemExit(f"  no {pattern} under {path}")
    blob = b"\n".join(f.read_bytes() for f in files)
    return blob, hashlib.sha256(blob).hexdigest()[:16], len(files)


def byte_offsets(text):
    """Character index to byte index, so predicted *bytes* can be exact.

    The tokenizer answers character offsets and the rail is denominated in
    bytes. Those agree on ASCII and this corpus is 244 lines short of being
    ASCII, so the fast path is taken where it holds and the map is built
    where it does not.
    """
    if text.isascii():
        return None
    out = np.empty(len(text) + 1, dtype=np.int64)
    out[0] = 0
    np.cumsum([len(c.encode("utf-8")) for c in text], out=out[1:])
    return out


def run_bpb(backend, hf, corpus, window=1024, limit=0, seed=0, dump=""):
    blob, digest, nfiles = corpus_blob(corpus)
    text = blob.decode("utf-8", "replace")
    bmap = byte_offsets(text)
    enc = hf.encode(text, add_special_tokens=False)
    ids, offs = enc.ids, enc.offsets

    room = getattr(backend, "max_len", None)
    if room and window > room:
        print(f"  a {window}-token window into a {room}-token context "
              f"-- raise --seq or lower --window")
        return 0.0

    # Whole windows only. A ragged tail would be a shorter window with less
    # context per token, which reads as the model doing worse at the end of
    # the corpus.
    starts = list(range(0, len(ids) - window + 1, window))
    total_windows = len(starts)
    if limit and limit < len(starts):
        # Drawn rather than sliced, for the reason `pick` gives: a prefix of
        # this corpus is `src/acpi/` and nothing else, so a prefix would
        # measure how well the model does at AML interpreters.
        starts = sorted(random.Random(seed).sample(starts, limit))

    nats = 0.0
    nbytes = 0
    ntok = 0
    record = []
    t0 = time.time()
    for n, s in enumerate(starts):
        backend.reset()
        chunk = ids[s:s + window]
        got, pred = backend.score(chunk)
        # Token `s` is context and is predicted by nothing, so the bytes it
        # covers are not on the bill. Everything from the start of token
        # `s+1` to the end of the last token is.
        c0, c1 = offs[s + 1][0], offs[s + window - 1][1]
        b = (int(bmap[c1] - bmap[c0]) if bmap is not None else c1 - c0)
        nats += got
        nbytes += b
        ntok += pred
        record.append((f"{digest}:{s}", b, pred, f"{got:.6f}"))
        if (n + 1) % 50 == 0 or n + 1 == len(starts):
            print(f"  [{n + 1}/{len(starts)}] bpb {nats / nbytes / LN2:6.4f}  "
                  f"({(time.time() - t0) / (n + 1):.2f}s/window)")

    bpb = nats / nbytes / LN2
    print(f"  corpus {corpus} {digest}: {nfiles} file(s), {len(blob)} byte(s), "
          f"{len(ids)} token(s), {len(ids) / len(blob):.3f} tok/byte")
    print(f"  bpb ({window}-token windows, {len(starts)}/{total_windows} "
          f"scored, {nbytes} byte(s) predicted): {bpb:6.4f}")
    if dump:
        with open(dump, "w", encoding="utf-8", newline="\n") as f:
            f.write("# bpb\tchunk\tbytes\ttokens\tnats\n")
            for r in record:
                f.write("\t".join(str(x) for x in r) + "\n")
        print(f"  per-window outcomes -> {dump}")
    return bpb


LN2 = float(np.log(2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tokenizer")
    ap.add_argument("--task", default="",
                    choices=["", "mmlu", "gsm8k", "niah", "route", "bpb"])
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
    ap.add_argument("--kv8", action="store_true",
                    help="quantise the KV cache to int8 the way the kernel "
                         "does. Without it a host figure is about the "
                         "checkpoint; with it, about what GLaDOS would score.")
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
    ap.add_argument("--corpus", default="src",
                    help="what the bpb rail scores. Pinned by content hash, "
                         "because a corpus that moves is a test set that "
                         "moves. Default is the kernel's own source, which "
                         "postdates every checkpoint here by a year.")
    ap.add_argument("--cloze", default="", choices=["", "raw", "norm"],
                    help="also score mmlu by the likelihood of each choice's "
                         "own text, which deletes the content-to-letter-to-"
                         "token indirection the standard protocol adds. Five "
                         "forward passes against one. The value picks which "
                         "variant the --dump records; all three print.")
    ap.add_argument("--sparse", type=int, default=0,
                    help="attend to only this many keys per head at decode. "
                         "The question the paged cache rests on: a cache on "
                         "disk is affordable only if a step reads a little of "
                         "it. 0 attends to everything.")
    ap.add_argument("--sparse-page", type=int, default=0, dest="sparse_page",
                    help="rank keys by Quest's per-page min/max bound with "
                         "pages this size, which is what a real index can do. "
                         "0 ranks by the true score, which nothing can do and "
                         "which is therefore the ceiling.")
    ap.add_argument("--sparse-sinks", type=int, default=4, dest="sparse_sinks")
    ap.add_argument("--sparse-local", type=int, default=64, dest="sparse_local")
    ap.add_argument("--window", type=int, default=1024,
                    help="tokens per scored window on the bpb rail")
    ap.add_argument("--hf-tokenizer", default="")
    ap.add_argument("--dump", default="",
                    help="write one line per question -- qid, gold, answer, "
                         "hit -- so two runs can be compared as pairs rather "
                         "than as two percentages that overlap.")
    ap.add_argument("--forest", default="",
                    help="retrieve from this forest before answering. The "
                         "measurement is the delta against the same --seed "
                         "without it, so run both.")
    ap.add_argument("--forest-k", type=int, default=4, dest="forest_k",
                    help="nodes per question, before the budget cuts it")
    ap.add_argument("--forest-budget", type=int, default=0,
                    dest="forest_budget",
                    help="token ceiling on the retrieved block, measured by "
                         "encoding it as the prompt will. 0 for no ceiling.")
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
               "route": 2048, "bpb": args.window + 8}[args.task]
    # The KV cache is `layers * batch * max_len * kv_dim * 2`, so a batch
    # multiplies it: 2048 at batch 16 is 7.5 GB on this checkpoint and 1152
    # is 4.2. The 5-shot prompt is 800 tokens at worst and the budget is 256,
    # so 1152 has room and `need` below refuses loudly if it does not.
    if args.task == "gsm8k" and args.batch > 1:
        # Plus whatever retrieval is allowed to add, because the cap and the
        # retrieval budget are two halves of one number and leaving them apart
        # means every --forest run refuses before it starts: the 8-question
        # smoke read "the longest prompt plus 256 new is 1272 tokens and the
        # context is 1152", which is the check working and the ceiling being
        # wrong. A retrieved block is bounded by --forest-budget by
        # construction, so the room it needs is known here rather than guessed.
        max_len = min(max_len, 1152 + (args.forest_budget if args.forest else 0))
    made = make_backend(args.model, max_len, args.oracle, batch=args.batch,
                        device=args.device, dtype=args.dtype,
                        kv_dtype=args.kv_dtype,
                        kv8=True if args.kv8 else None)
    backend, note = made if isinstance(made, tuple) else (
        made,
        f"dense dim {made.cfg['dim']}, {made.cfg['layers']} layers"
        + (", qk-norm" if made.cfg.get("qk_norm") else "")
        + f", head_dim {made.cfg['head_dim']}"
        + (", oracle" if args.oracle else ", batched"),
    )
    if args.sparse and hasattr(backend, "sparse"):
        backend.sparse = args.sparse
        backend.sparse_page = args.sparse_page
        backend.sparse_sinks = args.sparse_sinks
        backend.sparse_local = args.sparse_local
        how = (f"pages of {args.sparse_page}" if args.sparse_page
               else "the true score (a ceiling, not implementable)")
        note += (f", sparse {args.sparse} keys/head by {how}"
                 f", {args.sparse_sinks} sinks + {args.sparse_local} local")
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

    forest = None
    if args.forest:
        forest = fr.Forest.load(args.forest)
        comp = {}
        for n in forest.nodes:
            s = n.source.rsplit("/", 1)[-1]
            comp[s] = comp.get(s, 0) + 1
        print("[lm_eval] forest splits: "
              + ", ".join(f"{k} {v}" for k, v in sorted(comp.items())))
        # The builder refuses `test` and this is a different program reading
        # what the builder left behind, so it is checked again from here.
        if "test" in comp:
            print("[lm_eval] this forest holds a test split and is an answer "
                  "key for that rail. Rebuild it without --allow-test.")
            sys.exit(2)

    if args.task == "mmlu":
        run_mmlu(backend, hf, 0 if args.all else args.limit, args.seed,
                 dump=args.dump, forest=forest,
                 fbudget=args.forest_budget, fk=args.forest_k,
                 cloze=args.cloze)
    elif args.task == "gsm8k":
        run_gsm8k(backend, hf, tok, 0 if args.all else args.limit,
                  args.max_new, args.show,
                  args.shots if args.shots else 5,
                  batch=args.batch, seed=args.seed, dump=args.dump,
                  forest=forest, fbudget=args.forest_budget,
                  fk=args.forest_k)
    elif args.task == "niah":
        run_niah(backend, hf, tok, args.contexts, args.limit, args.max_new)
    elif args.task == "route":
        run_route(backend, hf, tok, build_alphabet(tok), args.limit)
    elif args.task == "bpb":
        run_bpb(backend, hf, args.corpus, args.window,
                0 if args.all else (args.limit or 256), args.seed, args.dump)


if __name__ == "__main__":
    main()






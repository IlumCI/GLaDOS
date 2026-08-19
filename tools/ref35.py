#!/usr/bin/env python3
"""Qwen3.5 forward pass in NumPy, checked against the golden fixture.

Ported from `transformers/models/qwen3_5/modeling_qwen3_5.py` step by step,
not reconstructed from a paper. Gated DeltaNet has a dozen plausible-looking
recurrences and only one right one, and a wrong one still produces fluent
text, so the only defensible way to write this is to transcribe the reference
and then prove the transcription against numbers the reference produced.

`fixture.py` captured those numbers. This runs the same prompt through the
same weights and diffs at every layer boundary. The first layer that disagrees
is the one with the mistake; a single logit diff would say only that one
exists somewhere.

Five things were learned by reading the reference, every one of which is a way
to get this wrong that would not have thrown:

  * `q_proj` is **twice** as wide as the query. It emits query and gate
    together, and the gate multiplies the attention output through a sigmoid
    before `o_proj`. Sizing it as `heads * head_dim` raises a shape error,
    which is the lucky outcome; sizing it right and dropping the gate does not.
  * RoPE covers **64 of 256** head dimensions. The remaining 192 pass through
    untouched.
  * mRoPE reduces to ordinary RoPE for text. All three position streams are
    the same tensor, so interleaving them is a no-op. Established by reading
    `apply_interleaved_mrope`, not assumed.
  * Query and key are **L2-normalised inside** the delta rule, and the
    `1/sqrt(head_k_dim)` scale goes on the query only.
  * The convolution is depthwise and causal, kernel 4, with SiLU after it.

Usage:
    ref35.py [--fixture out/fixture-qwen35.npz] [--model Qwen/Qwen3.5-0.8B]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# --- primitives -----------------------------------------------------------

def _norm(x, eps):
    return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps))


def rms_norm(x, w, eps):
    """Qwen3_5RMSNorm: scales by **(1 + w)**, and the parameter is initialised
    to zeros rather than ones.

    This is not the usual convention and it is the bug that phase 0 was built
    to catch. Using `w` directly produces perfectly plausible activations that
    are wrong from the first layer onward, and nothing downstream complains.
    Used by input_layernorm, post_attention_layernorm, q_norm, k_norm and the
    final norm.
    """
    return _norm(x, eps) * (1.0 + w)


def rms_norm_plain(x, w, eps):
    """The *other* RMSNorm in the same model: scales by `w`, initialised to
    ones. Only `linear_attn.norm` uses it, via `rms_norm_gated`. Two norms with
    different conventions inside one architecture is a trap worth naming."""
    return _norm(x, eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    # log1p(exp(x)), without overflowing for large x.
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def l2norm(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def rms_norm_gated(x, w, gate, eps):
    """Norm first, then gate. The order matters and the class name does not
    say which way round it goes. Uses the plain `w` convention, unlike every
    other norm in this model."""
    return rms_norm_plain(x, w, eps) * silu(gate)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def rope_tables(positions, head_dim, partial, theta):
    """cos/sin for the rotated prefix only.

    The rotated width is `head_dim * partial_rotary_factor` and `inv_freq` has
    half that many entries, so the tables come out that width after doubling.
    Everything past it in the head is never rotated.
    """
    dim = int(head_dim * partial)
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    freqs = np.outer(positions.astype(np.float64), inv)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(x, cos, sin):
    """x is [tokens, heads, head_dim]; cos and sin are [tokens, rotary_dim]."""
    r = cos.shape[-1]
    rot, passthru = x[..., :r], x[..., r:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    return np.concatenate([rot * c + rotate_half(rot) * s, passthru], axis=-1)


# --- the two mixers -------------------------------------------------------

def full_attention(h, w, cfg):
    """Causal attention with grouped keys and an output gate."""
    n = h.shape[0]
    hd = cfg["head_dim"]
    nq = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    eps = cfg["rms_norm_eps"]

    # Query and gate leave one projection together, query first.
    qg = (h @ w["self_attn.q_proj.weight"].T).reshape(n, nq, hd * 2)
    q, gate = qg[..., :hd], qg[..., hd:]
    gate = gate.reshape(n, nq * hd)

    q = rms_norm(q, w["self_attn.q_norm.weight"], eps)
    k = (h @ w["self_attn.k_proj.weight"].T).reshape(n, nkv, hd)
    k = rms_norm(k, w["self_attn.k_norm.weight"], eps)
    v = (h @ w["self_attn.v_proj.weight"].T).reshape(n, nkv, hd)

    cos, sin = rope_tables(np.arange(n), hd,
                           cfg["partial_rotary_factor"], cfg["rope_theta"])
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    groups = nq // nkv
    k = np.repeat(k, groups, axis=1)
    v = np.repeat(v, groups, axis=1)

    scale = hd ** -0.5
    out = np.empty((n, nq, hd), dtype=np.float32)
    mask = np.triu(np.full((n, n), -np.inf, dtype=np.float32), 1)
    for hh in range(nq):
        s = (q[:, hh] @ k[:, hh].T) * scale + mask
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        out[:, hh] = p @ v[:, hh]

    out = out.reshape(n, nq * hd) * sigmoid(gate)
    return out @ w["self_attn.o_proj.weight"].T


def gated_delta_net(h, w, cfg):
    """Linear attention: a depthwise causal conv, then the gated delta rule.

    The state is [k_head_dim, v_head_dim] per head and does not grow with the
    sequence. That is the entire reason this architecture is interesting for a
    kernel that decodes one token at a time.
    """
    n = h.shape[0]
    hk = cfg["linear_key_head_dim"]
    hv = cfg["linear_value_head_dim"]
    nk = cfg["linear_num_key_heads"]
    nv = cfg["linear_num_value_heads"]
    kdim = hk * nk
    kern = cfg["linear_conv_kernel_dim"]
    eps = cfg["rms_norm_eps"]

    qkv = h @ w["linear_attn.in_proj_qkv.weight"].T
    z = h @ w["linear_attn.in_proj_z.weight"].T
    b = h @ w["linear_attn.in_proj_b.weight"].T
    a = h @ w["linear_attn.in_proj_a.weight"].T

    # Depthwise causal convolution over time, then SiLU. Left-padded by
    # kernel-1 so position t sees t-3..t and nothing after it.
    cw = w["linear_attn.conv1d.weight"][:, 0, :]
    padded = np.concatenate(
        [np.zeros((kern - 1, qkv.shape[1]), dtype=np.float32), qkv], axis=0)
    conv = np.zeros_like(qkv)
    for j in range(kern):
        conv += padded[j:j + n] * cw[:, j]
    qkv = silu(conv)

    q = qkv[:, :kdim].reshape(n, nk, hk)
    k = qkv[:, kdim:2 * kdim].reshape(n, nk, hk)
    v = qkv[:, 2 * kdim:].reshape(n, nv, hv)

    beta = sigmoid(b)
    A = np.exp(w["linear_attn.A_log"].astype(np.float64))
    g = -A * softplus(a.astype(np.float64)
                      + w["linear_attn.dt_bias"].astype(np.float64))

    if nv // nk > 1:
        q = np.repeat(q, nv // nk, axis=1)
        k = np.repeat(k, nv // nk, axis=1)

    q = l2norm(q) * (hk ** -0.5)
    k = l2norm(k)

    state = np.zeros((nv, hk, hv), dtype=np.float64)
    out = np.zeros((n, nv, hv), dtype=np.float64)
    for t in range(n):
        gt = np.exp(g[t])[:, None, None]
        bt = beta[t][:, None]
        kt = k[t].astype(np.float64)
        vt = v[t].astype(np.float64)
        state = state * gt
        kv_mem = (state * kt[:, :, None]).sum(axis=1)
        delta = (vt - kv_mem) * bt
        state = state + kt[:, :, None] * delta[:, None, :]
        out[t] = (state * q[t].astype(np.float64)[:, :, None]).sum(axis=1)

    core = out.reshape(n * nv, hv).astype(np.float32)
    core = rms_norm_gated(core, w["linear_attn.norm.weight"],
                          z.reshape(n * nv, hv), eps)
    return core.reshape(n, nv * hv) @ w["linear_attn.out_proj.weight"].T


def mlp(h, w):
    return (silu(h @ w["mlp.gate_proj.weight"].T)
            * (h @ w["mlp.up_proj.weight"].T)) @ w["mlp.down_proj.weight"].T


LAYER_KEYS = {
    "linear_attention": [
        "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight",
        "linear_attn.conv1d.weight", "linear_attn.A_log",
        "linear_attn.dt_bias", "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    ],
    "full_attention": [
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    ],
}


def forward(tokens, tensors, cfg, capture=None):
    pre = "model.language_model."

    def W(name):
        return np.asarray(tensors[pre + name], dtype=np.float32)

    h = W("embed_tokens.weight")[tokens]
    if capture is not None:
        capture["embed"] = h.copy()

    eps = cfg["rms_norm_eps"]
    for i, kind in enumerate(cfg["layer_types"]):
        lp = f"layers.{i}."
        w = {s: W(lp + s) for s in LAYER_KEYS[kind]}

        x = rms_norm(h, W(lp + "input_layernorm.weight"), eps)
        mixed = (gated_delta_net(x, w, cfg) if kind == "linear_attention"
                 else full_attention(x, w, cfg))
        if capture is not None:
            tag = "linear_attn" if kind == "linear_attention" else "self_attn"
            capture[f"layer{i:02d}_{tag}"] = mixed.copy()
        h = h + mixed

        x = rms_norm(h, W(lp + "post_attention_layernorm.weight"), eps)
        m = mlp(x, w)
        if capture is not None:
            capture[f"layer{i:02d}_mlp"] = m.copy()
        h = h + m
        if capture is not None:
            capture[f"layer{i:02d}_out"] = h.copy()

    h = rms_norm(h, W("norm.weight"), eps)
    if capture is not None:
        capture["final_norm"] = h.copy()
    # Tied embeddings: the classifier is the embedding matrix.
    return h @ W("embed_tokens.weight").T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="out/fixture-qwen35.npz")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    # Loaded through torch because safetensors' numpy backend refuses bfloat16
    # outright, and every weight in this checkpoint is bf16. Only the file read
    # depends on torch; the forward pass below is numpy throughout, which is
    # the part that has to be portable to `reference.py`.
    import torch
    from safetensors.torch import load_file

    z = np.load(args.fixture)
    meta = json.loads(bytes(z["meta_json"]).decode())
    tokens = z["input_ids"][0]

    cfg = {k: meta[k] for k in
           ("hidden_size", "head_dim", "num_attention_heads",
            "num_key_value_heads", "rms_norm_eps", "linear_key_head_dim",
            "linear_value_head_dim", "linear_num_key_heads",
            "linear_num_value_heads", "linear_conv_kernel_dim")}
    cfg["layer_types"] = meta["layer_types"]
    rp = meta["rope_parameters"]
    cfg["rope_theta"] = rp["rope_theta"]
    cfg["partial_rotary_factor"] = rp["partial_rotary_factor"]

    print(f"[ref35] {len(cfg['layer_types'])} layers, {len(tokens)} tokens")
    path = snapshot_download(args.model,
                             allow_patterns=["*.safetensors", "*.json"])
    files = sorted(Path(path).glob("*.safetensors"))
    tensors = {}
    for p in files:
        for k, v in load_file(str(p)).items():
            tensors[k] = v.float().numpy()
    print(f"[ref35] {len(tensors)} tensors from {len(files)} file(s)")

    cap = {}
    logits = forward(tokens, tensors, cfg, capture=cap)

    order = ["embed"]
    for i, kind in enumerate(cfg["layer_types"]):
        tag = "linear_attn" if kind == "linear_attention" else "self_attn"
        order += [f"layer{i:02d}_{tag}", f"layer{i:02d}_mlp", f"layer{i:02d}_out"]
    order += ["final_norm"]

    print()
    print(f"{'key':24s} {'max abs':>11s} {'rel':>10s}")
    first_bad = None
    for key in order:
        if key not in z or key not in cap:
            continue
        want, got = z[key][0], cap[key]
        d = float(np.abs(want - got).max())
        rel = d / max(float(np.abs(want).max()), 1e-9)
        bad = rel >= args.tol
        print(f"{key:24s} {d:11.3e} {rel:10.2e}{'   <-- DIVERGES' if bad else ''}")
        if bad:
            first_bad = key
            break

    print()
    if first_bad:
        print(f"[ref35] first divergence at {first_bad}")
        return 1

    want = z["logits"][0]
    d = float(np.abs(want - logits).max())
    rel = d / max(float(np.abs(want).max()), 1e-9)
    print(f"{'logits':24s} {d:11.3e} {rel:10.2e}")
    agree = bool((want.argmax(-1) == logits.argmax(-1)).all())
    print(f"[ref35] argmax agrees at every position: {agree}")
    ok = rel < args.tol and agree
    print("[ref35] MATCHES the reference implementation" if ok
          else "[ref35] logits diverge")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

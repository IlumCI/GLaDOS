#!/usr/bin/env python3
"""Bisect one Gated DeltaNet layer: torch reference vs the numpy port.

The layer-level diff says the mixer is wrong and nothing about which step.
This recomputes layer 0 in torch, following the reference forward line by
line, and diffs every intermediate against the numpy version so the first
disagreeing stage names itself.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import ref35


def cmp(name, a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        print(f"{name:22s} SHAPE {a.shape} vs {b.shape}")
        return
    d = np.abs(a - b).max()
    rel = d / max(np.abs(a).max(), 1e-9)
    print(f"{name:22s} {d:11.3e} {rel:10.2e}{'   <--' if rel > 1e-4 else ''}")


def main():
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    z = np.load("out/fixture-qwen35.npz")
    meta = json.loads(bytes(z["meta_json"]).decode())
    cfg = {k: meta[k] for k in
           ("hidden_size", "head_dim", "num_attention_heads",
            "num_key_value_heads", "rms_norm_eps", "linear_key_head_dim",
            "linear_value_head_dim", "linear_num_key_heads",
            "linear_num_value_heads", "linear_conv_kernel_dim")}
    cfg["layer_types"] = meta["layer_types"]
    cfg["rope_theta"] = meta["rope_parameters"]["rope_theta"]
    cfg["partial_rotary_factor"] = meta["rope_parameters"]["partial_rotary_factor"]

    path = snapshot_download("Qwen/Qwen3.5-0.8B",
                             allow_patterns=["*.safetensors", "*.json"])
    tens = {}
    for p in sorted(Path(path).glob("*.safetensors")):
        for k, v in load_file(str(p)).items():
            tens[k] = v.float()

    pre = "model.language_model."
    W = lambda n: tens[pre + n]

    # Layer 0's mixer input: the embedding, normed.
    emb = torch.tensor(z["embed"][0])
    h = ref35.rms_norm(emb.numpy(), W("layers.0.input_layernorm.weight").numpy(),
                       cfg["rms_norm_eps"])
    ht = torch.tensor(h)
    n = h.shape[0]

    hk, hv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    kdim = hk * nk
    kern = cfg["linear_conv_kernel_dim"]
    eps = cfg["rms_norm_eps"]
    L = "layers.0.linear_attn."

    # --- torch, mirroring the reference forward -------------------------
    qkv_t = ht @ W(L + "in_proj_qkv.weight").T
    z_t = ht @ W(L + "in_proj_z.weight").T
    b_t = ht @ W(L + "in_proj_b.weight").T
    a_t = ht @ W(L + "in_proj_a.weight").T

    mixed = qkv_t.unsqueeze(0).transpose(1, 2)
    cw = W(L + "conv1d.weight").squeeze(1)
    conv_t = F.conv1d(mixed, weight=cw.unsqueeze(1), bias=None,
                      padding=kern - 1, groups=mixed.shape[1])[:, :, :n]
    conv_t = F.silu(conv_t).transpose(1, 2)[0]

    q_t = conv_t[:, :kdim].reshape(n, nk, hk)
    k_t = conv_t[:, kdim:2 * kdim].reshape(n, nk, hk)
    v_t = conv_t[:, 2 * kdim:].reshape(n, nv, hv)

    beta_t = b_t.sigmoid()
    g_t = -W(L + "A_log").float().exp() * F.softplus(a_t.float() + W(L + "dt_bias").float())

    # --- numpy, same order ----------------------------------------------
    qkv_n = h @ W(L + "in_proj_qkv.weight").numpy().T
    z_n = h @ W(L + "in_proj_z.weight").numpy().T
    b_n = h @ W(L + "in_proj_b.weight").numpy().T
    a_n = h @ W(L + "in_proj_a.weight").numpy().T

    cwn = W(L + "conv1d.weight").numpy()[:, 0, :]
    padded = np.concatenate(
        [np.zeros((kern - 1, qkv_n.shape[1]), dtype=np.float32), qkv_n], axis=0)
    conv_n = np.zeros_like(qkv_n)
    for j in range(kern):
        conv_n += padded[j:j + n] * cwn[:, j]
    conv_n = ref35.silu(conv_n)

    q_n = conv_n[:, :kdim].reshape(n, nk, hk)
    k_n = conv_n[:, kdim:2 * kdim].reshape(n, nk, hk)
    v_n = conv_n[:, 2 * kdim:].reshape(n, nv, hv)
    beta_n = ref35.sigmoid(b_n)
    A = np.exp(W(L + "A_log").numpy().astype(np.float64))
    g_n = -A * ref35.softplus(a_n.astype(np.float64)
                              + W(L + "dt_bias").numpy().astype(np.float64))

    print(f"{'stage':22s} {'max abs':>11s} {'rel':>10s}")
    cmp("in_proj_qkv", qkv_t.numpy(), qkv_n)
    cmp("conv+silu", conv_t.numpy(), conv_n)
    cmp("beta", beta_t.numpy(), beta_n)
    cmp("g", g_t.numpy(), g_n)
    cmp("z", z_t.numpy(), z_n)

    # --- the recurrence, torch's own function ---------------------------
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    core_t, _ = M.torch_recurrent_gated_delta_rule(
        q_t.unsqueeze(0), k_t.unsqueeze(0), v_t.unsqueeze(0),
        g=g_t.unsqueeze(0), beta=beta_t.unsqueeze(0),
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True)
    core_t = core_t[0]

    qn = ref35.l2norm(q_n) * (hk ** -0.5)
    kn = ref35.l2norm(k_n)
    state = np.zeros((nv, hk, hv), dtype=np.float64)
    core_n = np.zeros((n, nv, hv), dtype=np.float64)
    for t in range(n):
        gt = np.exp(g_n[t])[:, None, None]
        bt = beta_n[t][:, None]
        kt = kn[t].astype(np.float64)
        vt = v_n[t].astype(np.float64)
        state = state * gt
        kv_mem = (state * kt[:, :, None]).sum(axis=1)
        delta = (vt - kv_mem) * bt
        state = state + kt[:, :, None] * delta[:, None, :]
        core_n[t] = (state * qn[t].astype(np.float64)[:, :, None]).sum(axis=1)
    cmp("recurrence", core_t.numpy(), core_n)

    # --- chunked form, which is what the fixture actually ran -----------
    core_c, _ = M.torch_chunk_gated_delta_rule(
        q_t.unsqueeze(0), k_t.unsqueeze(0), v_t.unsqueeze(0),
        g=g_t.unsqueeze(0), beta=beta_t.unsqueeze(0),
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True)
    cmp("chunk vs recurrent", core_c[0].numpy(), core_t.numpy())

    # --- gated norm and out_proj ----------------------------------------
    gn = M.Qwen3_5RMSNormGated(hv, eps=eps)
    gn.weight.data = W(L + "norm.weight")
    out_t = gn(core_t.reshape(-1, hv), z_t.reshape(-1, hv))
    out_n = ref35.rms_norm_gated(core_n.reshape(-1, hv).astype(np.float32),
                                 W(L + "norm.weight").numpy(),
                                 z_n.reshape(-1, hv), eps)
    cmp("gated norm", out_t.detach().numpy(), out_n)

    fin_t = out_t.reshape(n, -1) @ W(L + "out_proj.weight").T
    fin_n = out_n.reshape(n, -1) @ W(L + "out_proj.weight").numpy().T
    cmp("out_proj", fin_t.detach().numpy(), fin_n)
    cmp("vs fixture", z["layer00_linear_attn"][0], fin_n)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Capture golden activations from the reference Qwen3.5 implementation.

Why this exists
---------------
Every other model in this project could be checked by reading its output. A
wrong RoPE convention still writes fluent English, but ask it for the capital
of France and a correct implementation says Paris. Gated DeltaNet has no such
tell. Its recurrence can be wrong in a dozen ways that all produce plausible
text, and if `reference.py` and `model.rs` are wrong the *same* way they agree
with each other and nothing catches it.

So the architecture is pinned to a file before any of it is reimplemented.
This runs the real implementation once and saves what it computed at every
layer boundary. `reference.py` is then developed against these numbers, and
only once it matches does the kernel port begin.

Captured in float32, not bfloat16. The point is to isolate architectural
mistakes from quantisation error, and comparing against a bf16-rounded target
would put a floor on agreement well above what a real bug looks like.

Per-layer, not just final logits. A single logit diff says something is wrong
and nothing about where; hidden states at every layer boundary say which layer,
and the first layer that disagrees is the one with the bug.

Usage:
    fixture.py [--model Qwen/Qwen3.5-0.8B] [--out out/fixture-qwen35.npz]
               [--prompt "..."] [--tokens N]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# A prompt with a checkable fact in it, so the fixture doubles as a sanity
# check on the download: if the reference itself cannot answer this, the
# problem is upstream of anything written here.
PROMPT = "The capital of France is"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", default="out/fixture-qwen35.npz")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--tokens", type=int, default=8,
                    help="how many tokens to generate greedily after the prompt")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    print(f"[fixture] loading {args.model}")
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu")
    model.eval()

    # The text tower. A conditional-generation wrapper puts the decoder one
    # level down, and hooking the wrapper would capture nothing useful.
    text = getattr(model, "model", model)
    text = getattr(text, "language_model", text)
    layers = text.layers
    print(f"[fixture] {len(layers)} layers")

    tcfg = getattr(cfg, "text_config", cfg)
    layer_types = list(getattr(tcfg, "layer_types", []))
    print(f"[fixture] layer_types: "
          f"{ {t: layer_types.count(t) for t in set(layer_types)} }")

    ids = tok(args.prompt, return_tensors="pt").input_ids
    print(f"[fixture] {ids.shape[1]} prompt tokens: {ids[0].tolist()}")

    store = {}

    def hook(name):
        def fn(_mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(h):
                store[name] = h.detach().float().numpy()
        return fn

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(hook(f"layer{i:02d}_out")))
        # One representative of each kind, in detail. Enough to localise a bug
        # inside a layer rather than merely to it.
        for sub in ("linear_attn", "self_attn", "mlp"):
            m = getattr(layer, sub, None)
            if m is not None:
                handles.append(m.register_forward_hook(hook(f"layer{i:02d}_{sub}")))

    emb = getattr(text, "embed_tokens", None)
    if emb is not None:
        handles.append(emb.register_forward_hook(hook("embed")))
    norm = getattr(text, "norm", None)
    if norm is not None:
        handles.append(norm.register_forward_hook(hook("final_norm")))

    print("[fixture] forward pass over the prompt")
    out = model(ids, use_cache=False)
    logits = out.logits.detach().float().numpy()
    store["logits"] = logits
    store["input_ids"] = ids.numpy()

    for h in handles:
        h.remove()

    # Greedy continuation, so the fixture also records what the model actually
    # says. A reimplementation that matches the activations but produces
    # different text has a bug in sampling or in the head, not in the layers.
    print("[fixture] greedy continuation")
    gen = model.generate(ids, max_new_tokens=args.tokens, do_sample=False)
    text_out = tok.decode(gen[0], skip_special_tokens=True)
    store["generated_ids"] = gen.numpy()
    print(f"[fixture] {text_out!r}")

    meta = {
        "model": args.model,
        "prompt": args.prompt,
        "layer_types": layer_types,
        "generated": text_out,
    }
    for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "vocab_size", "rms_norm_eps",
              "attn_output_gate", "linear_conv_kernel_dim",
              "linear_key_head_dim", "linear_value_head_dim",
              "linear_num_key_heads", "linear_num_value_heads",
              "num_experts", "num_experts_per_tok", "moe_intermediate_size",
              "shared_expert_intermediate_size", "intermediate_size"):
        if hasattr(tcfg, k):
            meta[k] = getattr(tcfg, k)
    rp = getattr(tcfg, "rope_parameters", None)
    if rp is not None:
        meta["rope_parameters"] = rp if isinstance(rp, dict) else dict(rp)
    store["meta_json"] = np.frombuffer(
        json.dumps(meta, default=str).encode("utf-8"), dtype=np.uint8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **store)
    mb = out_path.stat().st_size / 1e6
    print(f"[fixture] wrote {out_path}  {mb:.1f} MB  {len(store)} arrays")

    keys = sorted(k for k in store if k.startswith("layer"))
    print(f"[fixture] {len(keys)} layer captures, e.g. {keys[:3]}")
    print(f"[fixture] logits {logits.shape}")


if __name__ == "__main__":
    sys.exit(main())

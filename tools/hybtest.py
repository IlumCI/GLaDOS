#!/usr/bin/env python3
"""Build a small Qwen3.5 hybrid and say what the kernel should print for it.

Qwen3.5-0.8B is 723 MB converted and QEMU's VVFAT tops out at 516 MB on a
FAT16 geometry, so the real checkpoint cannot be the thing that checks the
kernel port. The alternative is not "check it on hardware later" -- that
defers every bug to a machine with no debugger -- it is to build a model small
enough to run here and shaped to exercise every path the big one does:

  * both layer kinds, and enough of each that the packed cache index has to be
    right. Layers 3 and 7 are the full-attention ones, so they want caches 0
    and 1; using the layer number would index past the end of a two-entry
    vector on layer 7 and into nothing at all on layer 3.
  * `rotary_dim` 8 of `head_dim` 16, so RoPE rotates half of each head and
    passes the rest through. A port that rotates the whole head still produces
    fluent-looking numbers.
  * 4 value heads over 2 key heads, so the delta rule has to share each
    normalised key across two value heads rather than assuming they pair up.
  * grouped-query attention, 4 query heads over 2 KV heads.
  * an untied classifier, which the tied 0.8B never exercises.

Weights are random, which is fine: this checks that two implementations of the
same arithmetic agree, and random weights make an accidental agreement less
likely rather than more.

    python tools/hybtest.py out/hybtest.bin --ids 7 11 3

prints the top-5 the kernel should report. Then:

    python tools/drive.py --model out/hybtest.bin "logits 7 11 3"
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import convert
import ref35
import v4

PREFIX = "model.language_model."


def build(dst, seed=7, schedule="LLLFLLLF", zero=False):
    dim, layers, heads, kv_heads, head_dim = 64, 8, 4, 2, 16
    hk, hv, nk, nv, kern = 8, 8, 2, 4, 4
    hidden, vocab, seq = 128, 256, 32
    kinds = ["full_attention" if ch == "F" else "linear_attention" for ch in schedule]
    layers = len(kinds)

    cfg = {
        "hidden_size": dim, "num_hidden_layers": layers,
        "num_attention_heads": heads, "num_key_value_heads": kv_heads,
        "vocab_size": vocab, "head_dim": head_dim, "rms_norm_eps": 1e-6,
        # Untied, so lm_head is a separate tensor and the negative-vocab path
        # in the header gets used.
        "tie_word_embeddings": False,
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        "linear_key_head_dim": hk, "linear_value_head_dim": hv,
        "linear_num_key_heads": nk, "linear_num_value_heads": nv,
        "linear_conv_kernel_dim": kern, "layer_types": kinds,
        "intermediate_size": hidden,
    }

    kdim, vdim = hk * nk, hv * nv
    conv_dim, q_dim, kv_dim = 2 * kdim + vdim, heads * head_dim, kv_heads * head_dim
    rng = np.random.default_rng(seed)
    w = {}

    def put(n, *shape, scale=1.0):
        v = (rng.standard_normal(shape) * scale).astype(np.float32)
        # Truncate to bf16 so writing the shards below is lossless and the
        # oracle sees exactly the numbers the converter quantised.
        w[PREFIX + n] = (v.view(np.uint32) & 0xFFFF0000).view(np.float32)

    put("embed_tokens.weight", vocab, dim, scale=0.1)
    for l, kind in enumerate(kinds):
        lp = f"layers.{l}."
        put(lp + "input_layernorm.weight", dim, scale=0.1)
        if kind == "linear_attention":
            a = lp + "linear_attn."
            put(a + "in_proj_qkv.weight", conv_dim, dim, scale=0.1)
            put(a + "in_proj_z.weight", vdim, dim, scale=0.1)
            put(a + "in_proj_a.weight", nv, dim, scale=0.1)
            put(a + "in_proj_b.weight", nv, dim, scale=0.1)
            put(a + "conv1d.weight", conv_dim, 1, kern, scale=0.5)
            # A_log is exponentiated and then negated, so a large positive
            # value here empties the state on the first token and the test
            # would pass without the recurrence carrying anything.
            put(a + "A_log", nv, scale=0.3)
            put(a + "dt_bias", nv, scale=0.3)
            put(a + "norm.weight", hv, scale=0.2)
            put(a + "out_proj.weight", dim, vdim, scale=0.1)
        else:
            a = lp + "self_attn."
            put(a + "q_norm.weight", head_dim, scale=0.1)
            put(a + "k_norm.weight", head_dim, scale=0.1)
            put(a + "q_proj.weight", 2 * q_dim, dim, scale=0.1)
            put(a + "k_proj.weight", kv_dim, dim, scale=0.1)
            put(a + "v_proj.weight", kv_dim, dim, scale=0.1)
            put(a + "o_proj.weight", dim, q_dim, scale=0.1)
        put(lp + "post_attention_layernorm.weight", dim, scale=0.1)
        m = lp + "mlp."
        put(m + "gate_proj.weight", hidden, dim, scale=0.1)
        put(m + "down_proj.weight", dim, hidden, scale=0.1)
        put(m + "up_proj.weight", hidden, dim, scale=0.1)
    put("norm.weight", dim, scale=0.1)
    put("lm_head.weight", vocab, dim, scale=0.3)
    w["lm_head.weight"] = w.pop(PREFIX + "lm_head.weight")

    # With every path back into the residual stream zeroed, the stream is
    # exactly the embedding row and the logits are
    # `lm_head @ rmsnorm_1p(embed[t])`. That isolates the loader, the final
    # norm and the classifier from every layer, which is the first fork in any
    # bisect of "the numbers are wrong and I do not know where".
    if zero:
        for k in list(w):
            if k.endswith(("out_proj.weight", "o_proj.weight", "down_proj.weight")):
                w[k] = np.zeros_like(w[k])

    src = dst.parent / (dst.stem + "-src")
    src.mkdir(parents=True, exist_ok=True)
    v4._write_shards(src, w)
    convert.convert_hybrid(cfg, convert.read_sharded(src), dst, True, seq, "qwen3_5")
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dst", nargs="?", default="out/hybtest.bin")
    ap.add_argument("--ids", type=int, nargs="+", default=[7, 11, 3])
    ap.add_argument("--schedule", default="LLLFLLLF",
                    help="one letter per layer, F full or L linear")
    ap.add_argument("--zero", action="store_true",
                    help="zero every projection back into the residual stream")
    ap.add_argument("--build", action="store_true",
                    help="regenerate the checkpoint before scoring it")
    args = ap.parse_args()

    dst = Path(args.dst)
    if args.build or not dst.exists():
        build(dst, schedule=args.schedule, zero=args.zero)

    tensors, cfg = v4.load(dst)
    logits = ref35.forward(np.array(args.ids), tensors, cfg)[-1]

    print(f"\n[oracle] {dst}, ids {' '.join(map(str, args.ids))}")
    print("  the kernel's `logits` should print these five lines:")
    order = np.argsort(-logits)[:5]
    for rank, i in enumerate(order, 1):
        milli = int(logits[i] * 1000.0)
        print(f"  {rank}. id {i:6}  logit {milli // 1000}.{abs(milli % 1000):03}")
    gap = float(logits[order[4]] - logits[order[5 - 1]]) if len(order) > 4 else 0.0
    sixth = float(np.sort(-logits)[5] * -1)
    print(f"\n  5th is {logits[order[4]]:.4f}, 6th is {sixth:.4f}, "
          f"margin {logits[order[4]] - sixth:.4f}")
    print("  (the kernel quantises its KV cache and the oracle does not, so "
          "expect the last digits to differ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

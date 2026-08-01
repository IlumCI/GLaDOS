#!/usr/bin/env python3
"""Convert a Hugging Face Llama- or Qwen-architecture checkpoint into a GLaDOS model file.

Why this exists
---------------
GLaDOS reads its weights off the ESP before ExitBootServices, as one flat
buffer that `offsets()` indexes by arithmetic. Safetensors is a dict of named
tensors in bf16. Something has to do the flattening, and it cannot be GLaDOS:
parsing JSON and rearranging 134M values inside a kernel with no debugger is a
poor trade against ~200 lines of Python that runs once.

Generation is memory-bandwidth bound -- bytes read per token is roughly the
model size -- so quantisation buys throughput as directly as it buys space.
int8 with a per-row scale is about 4x f32 on both counts. Qwen3-0.6B lands at
~600 MB against an 8 GB ESP; the constraint that used to be storage is now
purely speed.

Output format (GLADOSM3)
------------------------
A 64-byte header, then tensors in exactly the order `ai::model::offsets`
expects, so the loader stays a small delta on the one that already works:

    token_embedding, rms_att[L], (q_norm[L], k_norm[L],)
    wq[L], wk[L], wv[L], wo[L],
    rms_ffn[L], w1[L], w2[L], w3[L], rms_final

The QK-Norm pair is present only for architectures that have it, and sits with
the other attention norms so its absence is a gap of length zero.

Qwen3 is not a Llama in two ways that both fail silently rather than loudly:
its head width is stated rather than derived (128, where hidden/heads gives 64),
and it RMSNorms each head's query and key before RoPE. Neither produces a shape
error if ignored -- the model loads, runs, and generates confident nonsense.

Norm weights stay f32. They are ~35K values out of 134M, quantising them buys
nothing measurable, and they are the tensors where precision actually tells --
an RMSNorm scale is applied to every activation that passes through it.

A quantised tensor is stored as `n_rows` f32 scales followed by the int8 values,
row-major. Dequantisation is `value * scale[row]`, so the scale multiply
happens once per output element rather than once per weight.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"GLADOSM2"
# v2 described a Llama and stopped at byte 48. v3 fills three of the sixteen
# bytes that were already spare: head_dim, norm_eps, and a flags word. The magic
# is deliberately unchanged -- the kernel accepts both, so a v2 file already on
# an ESP keeps working.
VERSION = 3
HEADER_BYTES = 64

QUANT_F32 = 0
QUANT_I8 = 1

FLAG_QK_NORM = 1 << 0
# Which dimensions RoPE pairs. Clear means `rotate_half` -- i with i+head_dim/2
# -- which is what HuggingFace's modeling code does and therefore what every
# checkpoint trained through transformers expects. Set means llama2.c's
# adjacent-pair convention. Both are rotations by the same angles, so the wrong
# one produces a fluent model that attends by a scrambled notion of distance.
FLAG_ROPE_INTERLEAVED = 1 << 1

# Architectures whose weights are laid out the way this writer emits them.
# Qwen3 differs from Llama in exactly two ways that matter here, both handled
# below: an explicit head_dim, and QK-Norm.
SUPPORTED = {"llama", "qwen2", "qwen3"}


def read_safetensors(path):
    """Parse safetensors without the library.

    The format is small enough to not warrant a dependency: an 8-byte
    little-endian header length, that many bytes of JSON describing each
    tensor's dtype, shape and byte range, then the raw data.
    """
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        raw = f.read()

    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        chunk = raw[start:end]
        dtype = meta["dtype"]
        if dtype == "BF16":
            # numpy has no bfloat16. It does not need one: bf16 is the top 16
            # bits of an f32, so widening is a shift, exactly representable and
            # lossless in this direction.
            u16 = np.frombuffer(chunk, dtype="<u2").astype(np.uint32)
            arr = (u16 << 16).view(np.float32)
        elif dtype == "F32":
            arr = np.frombuffer(chunk, dtype="<f4")
        elif dtype == "F16":
            arr = np.frombuffer(chunk, dtype="<f2").astype(np.float32)
        else:
            raise SystemExit(f"{name}: unsupported dtype {dtype}")
        out[name] = arr.reshape(meta["shape"])
    return out


def quantise_rows(mat):
    """int8 with one f32 scale per row.

    Per-row rather than per-tensor because the rows of an attention or MLP
    projection routinely differ in magnitude by an order of magnitude, and a
    single tensor-wide scale would spend most of the int8 range on the largest
    row while flattening the rest into a handful of levels.
    """
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    rows = mat.reshape(mat.shape[0], -1)
    peak = np.abs(rows).max(axis=1)
    # A row of exact zeros would divide by zero; its values quantise to zero
    # under any scale, so the scale itself is arbitrary.
    scale = np.where(peak == 0, 1.0, peak / 127.0).astype(np.float32)
    q = np.rint(rows / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale


def emit(buf, arr, quant, stats):
    """Append one tensor, quantised or not, and account for it."""
    if quant and arr.ndim == 2:
        q, scale = quantise_rows(arr)
        buf.append(scale.tobytes())
        buf.append(q.tobytes())
        stats["quantised"] += q.size
        stats["bytes"] += scale.nbytes + q.nbytes
        # Report the worst relative error introduced, so a bad tensor is
        # visible here rather than as mysteriously poor output later.
        deq = q.astype(np.float32) * scale[:, None]
        denom = np.abs(arr).max()
        if denom > 0:
            err = np.abs(deq - arr.reshape(q.shape)).max() / denom
            stats["worst_err"] = max(stats["worst_err"], float(err))
    else:
        f = np.ascontiguousarray(arr, dtype=np.float32)
        buf.append(f.tobytes())
        stats["kept_f32"] += f.size
        stats["bytes"] += f.nbytes


def main():
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: convert.py <hf-dir> <out.bin> [--f32] [--seq N]"
        )
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    quant = "--f32" not in sys.argv
    seq_len = 512
    if "--seq" in sys.argv:
        seq_len = int(sys.argv[sys.argv.index("--seq") + 1])

    cfg = json.loads((src / "config.json").read_text())
    arch = cfg.get("model_type")
    if arch not in SUPPORTED:
        raise SystemExit(f"model_type {arch!r} is not one of {sorted(SUPPORTED)}")
    for flag in ("attention_bias", "mlp_bias"):
        if cfg.get(flag):
            raise SystemExit(f"{flag} is set; the flat layout has no room for biases")
    # A sliding window would change which keys attention may see, which is a
    # property of the checkpoint the kernel has no field for. Qwen3-0.6B sets
    # use_sliding_window false; refuse rather than quietly ignore it.
    if cfg.get("use_sliding_window") and cfg.get("sliding_window"):
        raise SystemExit("sliding-window attention is not implemented")
    if cfg.get("rope_scaling"):
        raise SystemExit(f"rope_scaling {cfg['rope_scaling']!r} is not implemented")

    dim = cfg["hidden_size"]
    hidden = cfg["intermediate_size"]
    layers = cfg["num_hidden_layers"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    vocab = cfg["vocab_size"]
    theta = float(cfg.get("rope_theta", 10000.0))
    tied = bool(cfg.get("tie_word_embeddings", False))
    norm_eps = float(cfg.get("rms_norm_eps", 1e-5))
    # Llama derives this; Qwen3 states it, and for the 0.6B the two disagree
    # (128 stated against 1024/16 = 64 derived). Trusting the derivation gives a
    # set of shapes that are wrong, mutually consistent, and load without error.
    head_dim = int(cfg.get("head_dim") or (dim // heads))
    # SmolLM2 states this explicitly as false; Qwen3 omits it, and the
    # transformers default is false either way. There is no HuggingFace model
    # here that wants the interleaved form.
    rope_interleaved = bool(cfg.get("rope_interleaved", False))
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim

    if heads % kv_heads:
        raise SystemExit("head geometry does not divide evenly")
    if head_dim % 2:
        raise SystemExit(f"head_dim {head_dim} is odd; RoPE rotates pairs")
    # The trained context can be far longer than anything we can afford: the KV
    # cache is n_layers * seq_len * kv_dim * 4 bytes twice over, so SmolLM2's
    # 8192 would be 377 MB of cache alone.
    trained = cfg.get("max_position_embeddings", seq_len)
    if seq_len > trained:
        raise SystemExit(f"seq {seq_len} exceeds the trained {trained}")

    w = read_safetensors(src / "model.safetensors")

    def take(name, shape=None):
        if name not in w:
            raise SystemExit(f"missing tensor {name}\nhave: {sorted(w)[:8]} ...")
        arr = w[name]
        # Shapes are checked against the *config* rather than against each
        # other. The whole failure mode this guards is a plausible geometry
        # derived from the wrong rule, which is perfectly self-consistent and
        # only disagrees with what is actually in the file.
        if shape is not None and tuple(arr.shape) != tuple(shape):
            raise SystemExit(f"{name} is {tuple(arr.shape)}, config implies {tuple(shape)}")
        return arr

    embed = take("model.embed_tokens.weight", (vocab, dim))

    # `tie_word_embeddings` and a saved `lm_head.weight` can both be present --
    # Qwen3-0.6B ships both. If they ever disagreed, honouring the flag would
    # silently use the wrong classifier, so check rather than assume.
    if tied and "lm_head.weight" in w:
        head = w["lm_head.weight"]
        if head.shape != embed.shape or not np.array_equal(head, embed):
            raise SystemExit(
                "tie_word_embeddings is set but lm_head.weight differs from the "
                "embedding; the checkpoint is not actually tied"
            )

    qk_norm = any(f"model.layers.{l}.self_attn.q_norm.weight" in w for l in range(layers))
    if qk_norm:
        missing = [
            l
            for l in range(layers)
            for t in ("q_norm", "k_norm")
            if f"model.layers.{l}.self_attn.{t}.weight" not in w
        ]
        if missing:
            raise SystemExit(f"QK-Norm present on some layers but not {sorted(set(missing))}")

    stats = {"quantised": 0, "kept_f32": 0, "bytes": 0, "worst_err": 0.0}
    body = []

    # Order must match ai::model::offsets exactly. The QK-Norm pair sits with
    # the other attention norms so that a model without them leaves a gap of
    # length zero and every later offset is unchanged.
    groups = [("input_layernorm.weight", False, (dim,))]
    if qk_norm:
        groups += [
            ("self_attn.q_norm.weight", False, (head_dim,)),
            ("self_attn.k_norm.weight", False, (head_dim,)),
        ]
    groups += [
        ("self_attn.q_proj.weight", True, (q_dim, dim)),
        ("self_attn.k_proj.weight", True, (kv_dim, dim)),
        ("self_attn.v_proj.weight", True, (kv_dim, dim)),
        ("self_attn.o_proj.weight", True, (dim, q_dim)),
        ("post_attention_layernorm.weight", False, (dim,)),
        ("mlp.gate_proj.weight", True, (hidden, dim)),
        ("mlp.down_proj.weight", True, (dim, hidden)),
        ("mlp.up_proj.weight", True, (hidden, dim)),
    ]

    emit(body, embed, quant, stats)
    for grp, quantise, shape in groups:
        for l in range(layers):
            emit(body, take(f"model.layers.{l}.{grp}", shape), quant and quantise, stats)
    emit(body, take("model.norm.weight", (dim,)), False, stats)

    if not tied:
        emit(body, take("lm_head.weight", (vocab, dim)), quant, stats)

    header = bytearray(HEADER_BYTES)
    header[0:8] = MAGIC
    struct.pack_into(
        "<Iiiiiiii f I",
        header,
        8,
        VERSION,
        dim,
        hidden,
        layers,
        heads,
        kv_heads,
        # Negative vocab means an untied classifier, the same convention
        # llama2.c uses, so the loader needs no new flag.
        vocab if tied else -vocab,
        seq_len,
        theta,
        QUANT_I8 if quant else QUANT_F32,
    )
    # v3 fields, in what was spare space at the end of the 64-byte header.
    struct.pack_into(
        "<i f I",
        header,
        48,
        head_dim,
        norm_eps,
        (FLAG_QK_NORM if qk_norm else 0)
        | (FLAG_ROPE_INTERLEAVED if rope_interleaved else 0),
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(header)
        for chunk in body:
            f.write(chunk)

    total = dst.stat().st_size
    print(f"  {arch}: dim {dim}  hidden {hidden}  layers {layers}  heads {heads}/{kv_heads} kv")
    print(f"  head_dim {head_dim} ({'stated' if cfg.get('head_dim') else 'derived'}), "
          f"q_dim {q_dim}, kv_dim {kv_dim}, QK-Norm {qk_norm}")
    print(f"  vocab {vocab}  seq {seq_len}  rope_theta {theta:g}  eps {norm_eps:g}  tied {tied}")
    print(f"  rope pairing: {'interleaved (2i,2i+1)' if rope_interleaved else 'rotate_half (i,i+d/2)'}")
    print(f"  quantised {stats['quantised']:,} values, kept {stats['kept_f32']:,} as f32")
    if quant:
        print(f"  worst relative error {stats['worst_err']:.4%}")
    print(f"  wrote {dst}  {total:,} B ({total / 1024 / 1024:.1f} MiB)")

    # The KV cache is heap, not ESP, and for a 28-layer model with 1024-wide
    # keys it is the largest single allocation in the system -- larger than the
    # entire heap was before this model existed. Report it here, where seq_len
    # is chosen, rather than leaving it to be discovered as an allocation
    # failure at boot.
    kv_bytes = 2 * layers * seq_len * kv_dim * 4
    print(f"  KV cache at seq {seq_len}: {kv_bytes / 1024 / 1024:.0f} MiB of kernel heap")

    esp_mb = 8192
    if total > esp_mb * 1024 * 1024 * 0.9:
        print(f"  WARNING: {total / 1024 / 1024:.0f} MiB against a {esp_mb} MiB ESP")


if __name__ == "__main__":
    main()

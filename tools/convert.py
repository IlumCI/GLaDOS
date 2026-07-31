#!/usr/bin/env python3
"""Convert a Hugging Face Llama-architecture checkpoint into a GLaDOS model file.

Why this exists
---------------
GLaDOS reads its weights off the ESP before ExitBootServices, as one flat
buffer that `offsets()` indexes by arithmetic. Safetensors is a dict of named
tensors in bf16. Something has to do the flattening, and it cannot be GLaDOS:
parsing JSON and rearranging 134M values inside a kernel with no debugger is a
poor trade against ~200 lines of Python that runs once.

Two things force quantisation rather than merely recommending it:

  * The ESP is 508 MB. SmolLM2-135M at f32 is 538 MB and does not fit.
  * Generation is memory-bandwidth bound -- bytes read per token is roughly
    the model size -- so f32 would be a few tokens per second at best.

int8 with a per-row scale takes it to ~134 MB, which fits with room and is
about 4x the throughput.

Output format (GLADOSM2)
------------------------
A 64-byte header, then tensors in exactly the order `ai::model::offsets`
expects, so the loader stays a small delta on the one that already works:

    token_embedding, rms_att[L], wq[L], wk[L], wv[L], wo[L],
    rms_ffn[L], w1[L], w2[L], w3[L], rms_final

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
VERSION = 2
HEADER_BYTES = 64

QUANT_F32 = 0
QUANT_I8 = 1


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
    if cfg.get("model_type") != "llama":
        raise SystemExit(f"model_type {cfg.get('model_type')!r} is not llama")
    for flag in ("attention_bias", "mlp_bias"):
        if cfg.get(flag):
            raise SystemExit(f"{flag} is set; the flat layout has no room for biases")

    dim = cfg["hidden_size"]
    hidden = cfg["intermediate_size"]
    layers = cfg["num_hidden_layers"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    vocab = cfg["vocab_size"]
    theta = float(cfg.get("rope_theta", 10000.0))
    tied = bool(cfg.get("tie_word_embeddings", False))

    if dim % heads or heads % kv_heads:
        raise SystemExit("head geometry does not divide evenly")
    # The trained context can be far longer than anything we can afford: the KV
    # cache is n_layers * seq_len * kv_dim * 4 bytes twice over, so SmolLM2's
    # 8192 would be 377 MB of cache alone.
    trained = cfg.get("max_position_embeddings", seq_len)
    if seq_len > trained:
        raise SystemExit(f"seq {seq_len} exceeds the trained {trained}")

    w = read_safetensors(src / "model.safetensors")

    def take(name):
        if name not in w:
            raise SystemExit(f"missing tensor {name}\nhave: {sorted(w)[:8]} ...")
        return w[name]

    embed = take("model.embed_tokens.weight")
    if embed.shape != (vocab, dim):
        raise SystemExit(f"embedding is {embed.shape}, expected {(vocab, dim)}")

    stats = {"quantised": 0, "kept_f32": 0, "bytes": 0, "worst_err": 0.0}
    body = []

    # Order must match ai::model::offsets exactly.
    emit(body, embed, quant, stats)
    for grp, quantise in (
        ("input_layernorm.weight", False),
        ("self_attn.q_proj.weight", True),
        ("self_attn.k_proj.weight", True),
        ("self_attn.v_proj.weight", True),
        ("self_attn.o_proj.weight", True),
        ("post_attention_layernorm.weight", False),
        ("mlp.gate_proj.weight", True),
        ("mlp.down_proj.weight", True),
        ("mlp.up_proj.weight", True),
    ):
        for l in range(layers):
            emit(body, take(f"model.layers.{l}.{grp}"), quant and quantise, stats)
    emit(body, take("model.norm.weight"), False, stats)

    if not tied:
        emit(body, take("lm_head.weight"), quant, stats)

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

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(header)
        for chunk in body:
            f.write(chunk)

    total = dst.stat().st_size
    print(f"  dim {dim}  hidden {hidden}  layers {layers}  heads {heads}/{kv_heads} kv")
    print(f"  vocab {vocab}  seq {seq_len}  rope_theta {theta:g}  tied {tied}")
    print(f"  quantised {stats['quantised']:,} values, kept {stats['kept_f32']:,} as f32")
    if quant:
        print(f"  worst relative error {stats['worst_err']:.4%}")
    print(f"  wrote {dst}  {total:,} B ({total / 1024 / 1024:.1f} MiB)")

    # The ESP is the constraint that forced quantisation in the first place, so
    # say plainly whether the result actually fits.
    esp = 508 * 1024 * 1024
    if total > esp * 0.9:
        print(f"  WARNING: {total / 1024 / 1024:.0f} MiB against a 508 MiB ESP")


if __name__ == "__main__":
    main()

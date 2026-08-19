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
# Qwen3.5 splits a double-width q_proj into query and gate, and multiplies the
# attention output by sigmoid(gate) before o_proj.
FLAG_ATTN_OUTPUT_GATE = 1 << 2

# Architectures whose weights are laid out the way this writer emits them.
# Qwen3 differs from Llama in exactly two ways that matter here, both handled
# below: an explicit head_dim, and QK-Norm.
DENSE = {"llama", "qwen2", "qwen3"}
HYBRID = {"qwen3_5", "qwen3_5_moe"}
SUPPORTED = DENSE | HYBRID

ARCH_DENSE = 0
ARCH_QWEN35 = 1
ARCH_QWEN35_MOE = 2

# v4 carries a second architecture whose layers are not all the same, so it
# needs both a wider header and a different body layout. v2 and v3 files are
# untouched: a dense checkpoint still writes VERSION 3 into a 64-byte header at
# exactly the offsets it always did.
VERSION_V4 = 4
V4_HEADER = 160
# One bit per layer, set for full attention. 256 layers is far past anything
# published and costs 32 bytes.
V4_BITMAP_AT = 112
V4_BITMAP_WORDS = 8

# Rows are quantised in blocks of roughly this many values. Big enough that the
# per-block numpy overhead is nothing, small enough that the scratch buffers
# stay a few MB whatever the tensor.
BLOCK_VALUES = 1 << 22


class Shards:
    """Every tensor in a checkpoint, materialised one at a time.

    Safetensors is small enough to not warrant a dependency: an 8-byte
    little-endian header length, that many bytes of JSON giving each tensor's
    dtype, shape and byte range, then the raw data. So this parses it, and
    memory-maps rather than reads.

    Reading the whole file was fine while the largest checkpoint here was
    Qwen3-0.6B, 1.2 GB of bf16 that widens to 2.4 GB of f32. Qwen3.5-35B-A3B
    is 71.9 GB and widens to about 143 GB, which is not a thing to hold in
    order to write it back out a tensor at a time. Through a memmap the peak
    is the largest single tensor instead of the model, and `expert()` slices
    an MoE bank before widening so it is one expert rather than all 256.

    Qwen3.5 also ships `model.safetensors-00001-of-00001.safetensors` plus an
    index even at one shard. Opening exactly `model.safetensors` -- which is
    all this used to do -- failed there with "missing tensor", which reads
    like a bad checkpoint rather than a reader that never opened the file.
    """

    CODES = {"BF16": "<u2", "F16": "<f2", "F32": "<f4"}

    def __init__(self, paths):
        self._meta = {}
        self._maps = {}
        for path in paths:
            with open(path, "rb") as f:
                (header_len,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(header_len))
            base = 8 + header_len
            for name, m in header.items():
                if name == "__metadata__":
                    continue
                if m["dtype"] not in self.CODES:
                    raise SystemExit(f"{name}: unsupported dtype {m['dtype']}")
                self._meta[name] = (path, m["dtype"], tuple(m["shape"]),
                                    base + m["data_offsets"][0])

    def __contains__(self, name):
        return name in self._meta

    def __iter__(self):
        return iter(self._meta)

    def __len__(self):
        return len(self._meta)

    def shape(self, name):
        return self._meta[name][2]

    def _view(self, name):
        if name not in self._meta:
            raise KeyError(name)
        path, dtype, shape, off = self._meta[name]
        mm = self._maps.get(path)
        if mm is None:
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            self._maps[path] = mm
        code = self.CODES[dtype]
        n = int(np.prod(shape)) if shape else 1
        seg = mm[off:off + n * np.dtype(code).itemsize]
        return seg.view(code).reshape(shape), dtype

    @staticmethod
    def _widen(arr, dtype):
        if dtype == "BF16":
            # numpy has no bfloat16 and does not need one: bf16 is the top 16
            # bits of an f32, so widening is a shift, exactly representable
            # and lossless in this direction.
            #
            # Shifted in place. Written as `(asarray(...) << 16).view(f32)` it
            # allocates the u32 twice -- once to widen, once for the shift
            # result -- which on Qwen3.5's 254M-value embedding is 2 GB of
            # peak to produce 1 GB of tensor.
            u = np.asarray(arr, dtype=np.uint32)
            u <<= 16
            return u.view(np.float32)
        return np.asarray(arr, dtype=np.float32)

    def __getitem__(self, name):
        arr, dtype = self._view(name)
        return self._widen(arr, dtype)

    def expert(self, name, i):
        """One slice of a batched 3-D expert bank, without widening the rest."""
        arr, dtype = self._view(name)
        return self._widen(arr[i], dtype)


def read_sharded(src):
    """Every tensor in a checkpoint, whether it is one file or fourteen."""
    single = src / "model.safetensors"
    if single.exists():
        return Shards([single])

    index = src / "model.safetensors.index.json"
    if not index.exists():
        shards = sorted(src.glob("*.safetensors"))
        if not shards:
            raise SystemExit(f"no safetensors under {src}")
    else:
        wm = json.loads(index.read_text())["weight_map"]
        shards = sorted({src / n for n in wm.values()})
        for s in shards:
            if not s.exists():
                raise SystemExit(f"the index names {s.name} but it is not here")

    print(f"  mapping {len(shards)} shard(s)")
    return Shards(shards)


class Sink:
    """A write-through stand-in for the list the body used to be.

    Accumulating every tensor and writing at the end was fine while the output
    was 570 MB. It is not fine at 36 GB: the peak became the whole output file
    plus the largest widened input tensor, for no reason at all, since nothing
    in the header is derived from the body. So the header space is reserved
    first and each tensor goes straight to disk as it is produced.

    Writing to a temporary and renaming at the end keeps the old property that
    a failed conversion leaves no file, rather than a truncated one that looks
    like a model.
    """

    def __init__(self, dst, header_bytes):
        self.dst = dst
        self.tmp = dst.with_name(dst.name + ".partial")
        dst.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.tmp, "wb")
        self.f.seek(header_bytes)

    def append(self, chunk):
        self.f.write(chunk)

    def finish(self, header):
        self.f.seek(0)
        self.f.write(header)
        self.f.close()
        self.tmp.replace(self.dst)


def quantise_rows(mat):
    """int8 with one f32 scale per row.

    Per-row rather than per-tensor because the rows of an attention or MLP
    projection routinely differ in magnitude by an order of magnitude, and a
    single tensor-wide scale would spend most of the int8 range on the largest
    row while flattening the rest into a handful of levels.
    """
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    rows = mat.reshape(mat.shape[0], -1)
    # max(|x|) over a row is max(max(x), -min(x)), and both of those reduce
    # without building anything. `np.abs(rows).max(axis=1)` reads the same but
    # materialises a second copy of the whole tensor first, which on the
    # embedding was a gigabyte spent to find 248,320 numbers.
    peak = np.maximum(rows.max(axis=1), -rows.min(axis=1))
    # A row of exact zeros would divide by zero; its values quantise to zero
    # under any scale, so the scale itself is arbitrary.
    scale = np.where(peak == 0, 1.0, peak / 127.0).astype(np.float32)

    # In blocks, and writing into a preallocated output, because the one-liner
    # this replaces allocated three full-size f32 temporaries -- the divide,
    # the rint and the clip. On Qwen3-0.6B's 254M-value embedding that is
    # 3 GB of scratch to produce 254 MB of int8, and on a 35B MoE expert bank
    # it is the difference between converting and being killed.
    q = np.empty(rows.shape, dtype=np.int8)
    err, denom = 0.0, float(np.abs(peak).max())
    step = max(1, BLOCK_VALUES // max(rows.shape[1], 1))
    for i in range(0, rows.shape[0], step):
        blk = rows[i:i + step] / scale[i:i + step, None]
        np.rint(blk, out=blk)
        np.clip(blk, -127, 127, out=blk)
        q[i:i + step] = blk
        # The round-trip error, measured on the block still in cache rather
        # than by dequantising the whole tensor again afterwards.
        blk *= scale[i:i + step, None]
        blk -= rows[i:i + step]
        err = max(err, float(np.abs(blk).max()))
    return q, scale, (err / denom if denom > 0 else 0.0)


def emit(buf, arr, quant, stats):
    """Append one tensor, quantised or not, and account for it."""
    if quant and arr.ndim == 2:
        q, scale, err = quantise_rows(arr)
        # The arrays go to the sink as-is. `.tobytes()` here copied every
        # quantised tensor a second time on the way out, which for the
        # embedding is 254 MB of peak for no gain -- a contiguous ndarray is
        # already a buffer that write() accepts.
        buf.append(scale)
        buf.append(q)
        stats["quantised"] += q.size
        stats["bytes"] += scale.nbytes + q.nbytes
        # The worst relative error introduced, so a bad tensor is visible here
        # rather than as mysteriously poor output later.
        stats["worst_err"] = max(stats["worst_err"], err)
    else:
        f = np.ascontiguousarray(arr, dtype=np.float32)
        buf.append(f)
        stats["kept_f32"] += f.size
        stats["bytes"] += f.nbytes


def convert_hybrid(cfg, w, dst, quant, seq_len, model_type):
    """Write a v4 file for Qwen3.5, dense or MoE.

    Two things differ from the dense writer beyond the obvious.

    **The body is layer-major.** The dense format groups by tensor and then by
    layer, so `offsets()` can multiply. A hybrid cannot: three layers in four
    hold `linear_attn.*` and the fourth holds `self_attn.*`, so there is no
    single stride. Emitting each layer's tensors together makes the loader walk
    the layers once and record where each began, which it has to do anyway, and
    it puts everything the forward pass reads for one layer in one contiguous
    run.

    **The layer schedule travels as a bitmap.** `full_attention_interval` is 4
    and describes every published checkpoint exactly, and deriving the schedule
    from it would mean a checkpoint that breaks the pattern loads and runs
    wrong. The list in the config is written down instead.
    """
    tc = cfg.get("text_config", cfg)
    moe = model_type == "qwen3_5_moe"
    # Detected rather than hardcoded. Every published Qwen3.5 nests the text
    # model under a vision-capable wrapper, but that is a property of these
    # releases and not of the architecture, and the failure if it ever changes
    # is a wall of "missing tensor" naming a prefix that does not exist.
    pre = ("model.language_model." if any(
        k.startswith("model.language_model.") for k in w) else "model.")

    dim = tc["hidden_size"]
    layers = tc["num_hidden_layers"]
    heads = tc["num_attention_heads"]
    kv_heads = tc["num_key_value_heads"]
    vocab = tc["vocab_size"]
    head_dim = tc["head_dim"]
    norm_eps = float(tc["rms_norm_eps"])
    tied = bool(tc.get("tie_word_embeddings", False))

    rp = tc.get("rope_parameters", {})
    theta = float(rp.get("rope_theta", 10000.0))
    # Resolved here rather than stored as a fraction: the kernel wants a count
    # of dimensions, and 0.25 * 256 is a question with one answer.
    rotary_dim = int(head_dim * float(rp.get("partial_rotary_factor", 1.0)))
    if rotary_dim % 2:
        raise SystemExit(f"rotary_dim {rotary_dim} is odd; RoPE rotates pairs")

    hk = tc["linear_key_head_dim"]
    hv = tc["linear_value_head_dim"]
    nk = tc["linear_num_key_heads"]
    nv = tc["linear_num_value_heads"]
    kern = tc["linear_conv_kernel_dim"]
    kdim, vdim = hk * nk, hv * nv
    conv_dim = 2 * kdim + vdim

    if moe:
        experts = tc["num_experts"]
        per_tok = tc["num_experts_per_tok"]
        hidden = tc["moe_intermediate_size"]
        shared = tc["shared_expert_intermediate_size"]
    else:
        experts = per_tok = shared = 0
        hidden = tc["intermediate_size"]

    kinds = list(tc["layer_types"])
    if len(kinds) != layers:
        raise SystemExit(f"layer_types has {len(kinds)} entries for {layers} layers")
    if layers > V4_BITMAP_WORDS * 32:
        raise SystemExit(f"{layers} layers exceeds the {V4_BITMAP_WORDS * 32}-bit schedule")
    n_full = kinds.count("full_attention")
    n_lin = kinds.count("linear_attention")
    if n_full + n_lin != layers:
        raise SystemExit(f"unknown layer kinds: {sorted(set(kinds))}")

    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim

    consumed = set()

    def check(full, shape):
        """Validate a tensor's shape from the index, without reading it."""
        if full not in w:
            raise SystemExit(f"missing tensor {full}")
        got = w.shape(full)
        if shape is not None and got != tuple(shape):
            raise SystemExit(f"{full} is {got}, config implies {tuple(shape)}")
        consumed.add(full)
        return full

    def take(name, shape=None):
        return w[check(pre + name, shape)]

    stats = {"quantised": 0, "kept_f32": 0, "bytes": 0, "worst_err": 0.0}
    body = Sink(dst, V4_HEADER)

    emit(body, take("embed_tokens.weight", (vocab, dim)), quant, stats)

    for l, kind in enumerate(kinds):
        lp = f"layers.{l}."
        emit(body, take(lp + "input_layernorm.weight", (dim,)), False, stats)

        if kind == "linear_attention":
            a = lp + "linear_attn."
            emit(body, take(a + "in_proj_qkv.weight", (conv_dim, dim)), quant, stats)
            emit(body, take(a + "in_proj_z.weight", (vdim, dim)), quant, stats)
            # a and b set the recurrence's decay and write strength. They are
            # 32x2048 -- a rounding error in the size of the whole file -- and
            # they feed a loop that carries its own output forward, so error
            # compounds step over step instead of averaging out. Keep f32.
            emit(body, take(a + "in_proj_a.weight", (nv, dim)), False, stats)
            emit(body, take(a + "in_proj_b.weight", (nv, dim)), False, stats)
            emit(body, take(a + "conv1d.weight", (conv_dim, 1, kern)).reshape(conv_dim, kern),
                 False, stats)
            emit(body, take(a + "A_log", (nv,)), False, stats)
            emit(body, take(a + "dt_bias", (nv,)), False, stats)
            emit(body, take(a + "norm.weight", (hv,)), False, stats)
            emit(body, take(a + "out_proj.weight", (dim, vdim)), quant, stats)
        else:
            a = lp + "self_attn."
            emit(body, take(a + "q_norm.weight", (head_dim,)), False, stats)
            emit(body, take(a + "k_norm.weight", (head_dim,)), False, stats)
            # Double width: query and gate leave this projection together.
            emit(body, take(a + "q_proj.weight", (2 * q_dim, dim)), quant, stats)
            emit(body, take(a + "k_proj.weight", (kv_dim, dim)), quant, stats)
            emit(body, take(a + "v_proj.weight", (kv_dim, dim)), quant, stats)
            emit(body, take(a + "o_proj.weight", (dim, q_dim)), quant, stats)

        emit(body, take(lp + "post_attention_layernorm.weight", (dim,)), False, stats)

        m = lp + "mlp."
        if moe:
            # The router decides *which experts run*. An int8 error there is
            # not a small numeric perturbation, it is a different computation,
            # so it stays f32 however large the expert bank gets.
            emit(body, take(m + "gate.weight", (experts, dim)), False, stats)
            gu = check(pre + m + "experts.gate_up_proj", (experts, 2 * hidden, dim))
            dn = check(pre + m + "experts.down_proj", (experts, dim, hidden))
            for e in range(experts):
                # One expert at a time. The 35B bank is 2.1 GB once widened and
                # nothing here needs more of it than the slice being written.
                emit(body, w.expert(gu, e), quant, stats)
                emit(body, w.expert(dn, e), quant, stats)
            s = m + "shared_expert."
            emit(body, take(s + "gate_proj.weight", (shared, dim)), quant, stats)
            emit(body, take(s + "up_proj.weight", (shared, dim)), quant, stats)
            emit(body, take(s + "down_proj.weight", (dim, shared)), quant, stats)
            emit(body, take(m + "shared_expert_gate.weight", (1, dim)).reshape(dim),
                 False, stats)
        else:
            emit(body, take(m + "gate_proj.weight", (hidden, dim)), quant, stats)
            emit(body, take(m + "down_proj.weight", (dim, hidden)), quant, stats)
            emit(body, take(m + "up_proj.weight", (hidden, dim)), quant, stats)

    emit(body, take("norm.weight", (dim,)), False, stats)
    if not tied:
        # Unprefixed even where everything else is nested, checked against the
        # published 35B-A3B index rather than assumed.
        emit(body, w[check("lm_head.weight", (vocab, dim))], quant, stats)

    # Nothing may go missing quietly. The body is positional, so a tensor this
    # writer forgot is not a short file or a load error -- it is a model with a
    # layer of noise in it. Anything neither written nor deliberately skipped
    # is a bug in this function.
    ignored = {k for k in w if k.startswith("model.visual.") or k.startswith("mtp.")}
    left = sorted(set(w) - consumed - ignored)
    if left:
        listed = "".join(f"\n  {k}" for k in left[:8])
        raise SystemExit(f"{len(left)} tensor(s) neither written nor skipped:{listed}")
    skipped = len(ignored)

    header = bytearray(V4_HEADER)
    header[0:8] = MAGIC
    struct.pack_into(
        "<Iiiiiiii f I", header, 8,
        VERSION_V4, dim, hidden, layers, heads, kv_heads,
        vocab if tied else -vocab, seq_len, theta,
        QUANT_I8 if quant else QUANT_F32,
    )
    struct.pack_into(
        "<i f I", header, 48,
        head_dim, norm_eps,
        # QK-Norm is on the full-attention layers only, which the bitmap
        # already says; the flag stays for readers that only look at flags.
        FLAG_QK_NORM | FLAG_ATTN_OUTPUT_GATE,
    )
    struct.pack_into(
        "<Iiiiiiiiiiii", header, 60,
        ARCH_QWEN35_MOE if moe else ARCH_QWEN35,
        rotary_dim, hk, hv, nk, nv, kern,
        experts, per_tok, shared, n_full, n_lin,
    )
    words = [0] * V4_BITMAP_WORDS
    for l, kind in enumerate(kinds):
        if kind == "full_attention":
            words[l // 32] |= 1 << (l % 32)
    struct.pack_into("<" + "I" * V4_BITMAP_WORDS, header, V4_BITMAP_AT, *words)

    body.finish(header)

    total = dst.stat().st_size
    schedule = "".join("F" if k == "full_attention" else "L" for k in kinds)
    print(f"  {model_type}: dim {dim}  hidden {hidden}  layers {layers}  "
          f"heads {heads}/{kv_heads} kv")
    print(f"  head_dim {head_dim}, rotary_dim {rotary_dim} of {head_dim}, q_proj is 2x wide")
    print(f"  linear: {nk}x{hk} keys, {nv}x{hv} values, conv kernel {kern}, conv_dim {conv_dim}")
    if moe:
        print(f"  moe: {experts} experts, top-{per_tok}, inner {hidden}, shared {shared}")
    print(f"  schedule {schedule}  ({n_full} full, {n_lin} linear)")
    print(f"  vocab {vocab}  seq {seq_len}  theta {theta:g}  eps {norm_eps:g}  tied {tied}")
    print(f"  skipped {skipped} vision and mtp tensors")
    print(f"  quantised {stats['quantised']:,} values, kept {stats['kept_f32']:,} as f32")
    if quant:
        print(f"  worst relative error {stats['worst_err']:.4%}")
    print(f"  wrote {dst}  {total:,} B ({total / 1024 / 1024:.1f} MiB)")

    # The whole argument for this architecture. Only full-attention layers
    # carry a cache; the linear ones carry a state that is the same size at
    # token 32 as at token 32,768.
    kv_bytes = 2 * n_full * seq_len * kv_dim * 4
    state = n_lin * (nv * hk * hv * 4 + conv_dim * (kern - 1) * 4)
    print(f"  KV cache at seq {seq_len}: {kv_bytes / 1024 / 1024:.0f} MiB "
          f"({n_full} of {layers} layers)")
    print(f"  recurrent state: {state / 1024 / 1024:.1f} MiB, independent of context")


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

    # Hybrids take a different writer entirely. Everything below this point
    # assumes every layer holds the same tensors, which is exactly what
    # Qwen3.5 stops being true.
    if arch in HYBRID:
        return convert_hybrid(cfg, read_sharded(src), dst, quant, seq_len, arch)

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

    w = read_sharded(src)

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
    body = Sink(dst, HEADER_BYTES)

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

    body.finish(header)

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

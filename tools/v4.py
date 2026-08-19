#!/usr/bin/env python3
"""Read a v4 (hybrid) GLaDOS checkpoint back into named tensors.

This is the other half of `convert.py`'s `convert_hybrid`, and it exists to be
run rather than to be read. The v4 body is laid out by arithmetic with nothing
self-describing in it: no names, no shapes, no per-tensor lengths. If the
writer and the reader disagree about one dimension, everything after that
point is still perfectly valid float32 and is garbage.

So the file is **walked, never seeked**, exactly as `reference.py` walks v2 and
v3, and the walk ends with

    if pos != len(raw): raise

Landing anywhere but the exact end means a shape is wrong somewhere, which
turns the silent failure into a loud one. That single assertion is most of the
value of this module; the dequantisation is the easy part.

The names produced match the HuggingFace ones exactly, prefix included, so
`ref35.forward` cannot tell a converted file from the original safetensors and
the same diff runs against both.
"""

import json
import struct

import numpy as np

MAGIC = b"GLADOSM2"
HEADER = 160
BITMAP_AT = 112
BITMAP_WORDS = 8

ARCH_DENSE = 0
ARCH_QWEN35 = 1
ARCH_QWEN35_MOE = 2

PREFIX = "model.language_model."


class Walker:
    """A cursor over the body that can only move forward."""

    def __init__(self, raw, pos, quant):
        self.raw = raw
        self.pos = pos
        self.quant = quant

    def f32(self, *shape):
        n = int(np.prod(shape))
        end = self.pos + n * 4
        if end > len(self.raw):
            raise SystemExit(f"f32{shape} runs {end - len(self.raw)} B past the end")
        arr = np.frombuffer(self.raw, dtype="<f4", count=n, offset=self.pos)
        self.pos = end
        return arr.reshape(shape)

    def q(self, rows, cols):
        """One quantised 2-D tensor: `rows` scales, then rows*cols int8."""
        if not self.quant:
            return self.f32(rows, cols)
        scales = self.f32(rows)
        end = self.pos + rows * cols
        if end > len(self.raw):
            raise SystemExit(f"i8[{rows},{cols}] runs {end - len(self.raw)} B past the end")
        q = np.frombuffer(self.raw, dtype=np.int8, count=rows * cols, offset=self.pos)
        self.pos = end
        return q.reshape(rows, cols).astype(np.float32) * scales[:, None]


def load(path):
    """Return `(tensors, cfg)` for a v4 file. `cfg` matches ref35's dict."""
    raw = open(path, "rb").read()
    if raw[:8] != MAGIC:
        raise SystemExit(f"{path}: not a GLaDOS model file")
    (version,) = struct.unpack_from("<I", raw, 8)
    if version != 4:
        raise SystemExit(f"{path}: version {version}, this reads v4 only")

    (dim, hidden, layers, heads, kv_heads, vocab_signed, seq_len) = \
        struct.unpack_from("<iiiiiii", raw, 12)
    (theta,) = struct.unpack_from("<f", raw, 40)
    (quant,) = struct.unpack_from("<I", raw, 44)
    (head_dim,) = struct.unpack_from("<i", raw, 48)
    (norm_eps,) = struct.unpack_from("<f", raw, 52)
    (flags,) = struct.unpack_from("<I", raw, 56)
    (arch,) = struct.unpack_from("<I", raw, 60)
    (rotary_dim, hk, hv, nk, nv, kern, experts, per_tok, shared,
     n_full, n_lin) = struct.unpack_from("<iiiiiiiiiii", raw, 64)
    words = struct.unpack_from("<" + "I" * BITMAP_WORDS, raw, BITMAP_AT)

    if arch not in (ARCH_QWEN35, ARCH_QWEN35_MOE):
        raise SystemExit(f"{path}: arch {arch} is not a hybrid")
    moe = arch == ARCH_QWEN35_MOE
    tied = vocab_signed > 0
    vocab = abs(vocab_signed)

    kinds = ["full_attention" if words[i // 32] >> (i % 32) & 1
             else "linear_attention" for i in range(layers)]
    if kinds.count("full_attention") != n_full or kinds.count("linear_attention") != n_lin:
        raise SystemExit("bitmap disagrees with the layer counts in the header")

    kdim, vdim = hk * nk, hv * nv
    conv_dim = 2 * kdim + vdim
    q_dim, kv_dim = heads * head_dim, kv_heads * head_dim

    w = Walker(raw, HEADER, quant == 1)
    t = {}

    def put(name, arr):
        t[PREFIX + name] = arr

    put("embed_tokens.weight", w.q(vocab, dim))

    for l, kind in enumerate(kinds):
        lp = f"layers.{l}."
        put(lp + "input_layernorm.weight", w.f32(dim))

        if kind == "linear_attention":
            a = lp + "linear_attn."
            put(a + "in_proj_qkv.weight", w.q(conv_dim, dim))
            put(a + "in_proj_z.weight", w.q(vdim, dim))
            put(a + "in_proj_a.weight", w.f32(nv, dim))
            put(a + "in_proj_b.weight", w.f32(nv, dim))
            # Restored to the reference's [C, 1, K]: the forward pass indexes
            # it as `[:, 0, :]`, and matching that is cheaper than a special
            # case in the only consumer.
            put(a + "conv1d.weight", w.f32(conv_dim, kern).reshape(conv_dim, 1, kern))
            put(a + "A_log", w.f32(nv))
            put(a + "dt_bias", w.f32(nv))
            put(a + "norm.weight", w.f32(hv))
            put(a + "out_proj.weight", w.q(dim, vdim))
        else:
            a = lp + "self_attn."
            put(a + "q_norm.weight", w.f32(head_dim))
            put(a + "k_norm.weight", w.f32(head_dim))
            put(a + "q_proj.weight", w.q(2 * q_dim, dim))
            put(a + "k_proj.weight", w.q(kv_dim, dim))
            put(a + "v_proj.weight", w.q(kv_dim, dim))
            put(a + "o_proj.weight", w.q(dim, q_dim))

        put(lp + "post_attention_layernorm.weight", w.f32(dim))

        m = lp + "mlp."
        if moe:
            put(m + "gate.weight", w.f32(experts, dim))
            gate_up = np.empty((experts, 2 * hidden, dim), dtype=np.float32)
            down = np.empty((experts, dim, hidden), dtype=np.float32)
            for e in range(experts):
                gate_up[e] = w.q(2 * hidden, dim)
                down[e] = w.q(dim, hidden)
            put(m + "experts.gate_up_proj", gate_up)
            put(m + "experts.down_proj", down)
            s = m + "shared_expert."
            put(s + "gate_proj.weight", w.q(shared, dim))
            put(s + "up_proj.weight", w.q(shared, dim))
            put(s + "down_proj.weight", w.q(dim, shared))
            put(m + "shared_expert_gate.weight", w.f32(dim).reshape(1, dim))
        else:
            put(m + "gate_proj.weight", w.q(hidden, dim))
            put(m + "down_proj.weight", w.q(dim, hidden))
            put(m + "up_proj.weight", w.q(hidden, dim))

    put("norm.weight", w.f32(dim))
    if not tied:
        t["lm_head.weight"] = w.q(vocab, dim)

    if w.pos != len(raw):
        raise SystemExit(
            f"layout walked to {w.pos} but the file is {len(raw)} "
            f"({len(raw) - w.pos:+d}); a shape is wrong")

    cfg = {
        "hidden_size": dim, "intermediate_size": hidden,
        "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": kv_heads, "head_dim": head_dim,
        "rms_norm_eps": norm_eps, "rope_theta": theta,
        "partial_rotary_factor": rotary_dim / head_dim,
        "linear_key_head_dim": hk, "linear_value_head_dim": hv,
        "linear_num_key_heads": nk, "linear_num_value_heads": nv,
        "linear_conv_kernel_dim": kern,
        "layer_types": kinds,
        "num_experts": experts, "num_experts_per_tok": per_tok,
        "shared_expert_intermediate_size": shared,
        "vocab_size": vocab, "tie_word_embeddings": tied,
        "seq_len": seq_len, "quant": quant, "flags": flags, "arch": arch,
    }
    return t, cfg


def _synthetic(moe):
    """A tiny checkpoint with the real tensor names and shapes.

    The MoE writer has no other check available. The smallest published MoE is
    35B-A3B at 71.9 GB, so there is no fixture for it and will not be one on
    this machine, and `ref35.forward` has no expert routing to compare against
    anyway. What *can* be checked without any of that is the part that fails
    silently: whether the writer and the reader agree on the order and shape of
    every tensor. Shapes here are chosen small and all distinct, so a
    transposed or swapped pair cannot round-trip by coincidence.
    """
    dim, layers, heads, kv_heads, head_dim, vocab = 32, 4, 4, 2, 16, 64
    hk, hv, nk, nv, kern = 8, 12, 2, 2, 4
    kinds = ["linear_attention"] * 3 + ["full_attention"]
    cfg = {
        "hidden_size": dim, "num_hidden_layers": layers,
        "num_attention_heads": heads, "num_key_value_heads": kv_heads,
        "vocab_size": vocab, "head_dim": head_dim, "rms_norm_eps": 1e-6,
        "tie_word_embeddings": not moe,
        "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.25},
        "linear_key_head_dim": hk, "linear_value_head_dim": hv,
        "linear_num_key_heads": nk, "linear_num_value_heads": nv,
        "linear_conv_kernel_dim": kern, "layer_types": kinds,
    }
    if moe:
        cfg.update(num_experts=4, num_experts_per_tok=2,
                   moe_intermediate_size=20, shared_expert_intermediate_size=24)
        hidden = 20
    else:
        cfg["intermediate_size"] = 40
        hidden = 40

    kdim, vdim = hk * nk, hv * nv
    conv_dim, q_dim, kv_dim = 2 * kdim + vdim, heads * head_dim, kv_heads * head_dim
    rng = np.random.default_rng(0)
    w = {}

    def put(n, *shape):
        v = rng.standard_normal(shape).astype(np.float32)
        # Truncated to what bf16 can hold exactly, so the shards below can be
        # written in bf16 -- the dtype every real checkpoint uses -- and still
        # compared for equality rather than for closeness.
        w[PREFIX + n] = (v.view(np.uint32) & 0xFFFF0000).view(np.float32)

    put("embed_tokens.weight", vocab, dim)
    for l, kind in enumerate(kinds):
        lp = f"layers.{l}."
        put(lp + "input_layernorm.weight", dim)
        if kind == "linear_attention":
            a = lp + "linear_attn."
            put(a + "in_proj_qkv.weight", conv_dim, dim)
            put(a + "in_proj_z.weight", vdim, dim)
            put(a + "in_proj_a.weight", nv, dim)
            put(a + "in_proj_b.weight", nv, dim)
            put(a + "conv1d.weight", conv_dim, 1, kern)
            put(a + "A_log", nv)
            put(a + "dt_bias", nv)
            put(a + "norm.weight", hv)
            put(a + "out_proj.weight", dim, vdim)
        else:
            a = lp + "self_attn."
            put(a + "q_norm.weight", head_dim)
            put(a + "k_norm.weight", head_dim)
            put(a + "q_proj.weight", 2 * q_dim, dim)
            put(a + "k_proj.weight", kv_dim, dim)
            put(a + "v_proj.weight", kv_dim, dim)
            put(a + "o_proj.weight", dim, q_dim)
        put(lp + "post_attention_layernorm.weight", dim)
        m = lp + "mlp."
        if moe:
            put(m + "gate.weight", 4, dim)
            put(m + "experts.gate_up_proj", 4, 2 * hidden, dim)
            put(m + "experts.down_proj", 4, dim, hidden)
            put(m + "shared_expert.gate_proj.weight", 24, dim)
            put(m + "shared_expert.up_proj.weight", 24, dim)
            put(m + "shared_expert.down_proj.weight", dim, 24)
            put(m + "shared_expert_gate.weight", 1, dim)
        else:
            put(m + "gate_proj.weight", hidden, dim)
            put(m + "down_proj.weight", dim, hidden)
            put(m + "up_proj.weight", hidden, dim)
    put("norm.weight", dim)
    if moe:
        put("lm_head.weight", vocab, dim)
        w["lm_head.weight"] = w.pop(PREFIX + "lm_head.weight")
    return cfg, w


def _write_shards(dirpath, w):
    """Write `w` as two bf16 safetensors shards plus an index.

    Two rather than one, and an index rather than a bare file, because that is
    the shape every Qwen3.5 release ships and the shape the old reader could
    not open. Writing it here means the selftest covers the reader as well as
    the writer, instead of handing convert_hybrid a dict that no checkpoint
    ever looks like.
    """
    names = sorted(w)
    halves = [names[:len(names) // 2], names[len(names) // 2:]]
    weight_map = {}
    for i, half in enumerate(halves, 1):
        fn = f"model-{i:05d}-of-{len(halves):05d}.safetensors"
        header, blob, off = {}, [], 0
        for n in half:
            raw = (w[n].view(np.uint32) >> 16).astype("<u2").tobytes()
            header[n] = {"dtype": "BF16", "shape": list(w[n].shape),
                         "data_offsets": [off, off + len(raw)]}
            blob.append(raw)
            off += len(raw)
            weight_map[n] = fn
        js = json.dumps(header).encode()
        with open(dirpath / fn, "wb") as f:
            f.write(struct.pack("<Q", len(js)))
            f.write(js)
            for b in blob:
                f.write(b)
    (dirpath / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))


def selftest():
    """Write a synthetic checkpoint of each kind and read it straight back."""
    import io
    import contextlib
    import tempfile
    from pathlib import Path

    import convert

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for moe in (False, True):
            kind = "qwen3_5_moe" if moe else "qwen3_5"
            for quant in (False, True):
                cfg, w = _synthetic(moe)
                src = Path(tmp) / f"{kind}-src"
                src.mkdir(exist_ok=True)
                _write_shards(src, w)
                dst = Path(tmp) / f"{kind}-{'i8' if quant else 'f32'}.bin"
                with contextlib.redirect_stdout(io.StringIO()):
                    convert.convert_hybrid(cfg, convert.read_sharded(src),
                                           dst, quant, 128, kind)
                # load() raises if the walk does not land on the last byte,
                # which is the assertion this whole exercise is for.
                got, _ = load(dst)

                if set(got) != set(w):
                    for n in sorted(set(w) ^ set(got)):
                        print(f"  {kind}: name mismatch {n}")
                    ok = False
                    continue
                worst, worst_n = 0.0, None
                for n, want in w.items():
                    g = got[n]
                    if g.shape != want.shape:
                        print(f"  {kind}: {n} came back {g.shape}, wrote {want.shape}")
                        ok = False
                        continue
                    rel = float(np.abs(g - want).max()) / max(float(np.abs(want).max()), 1e-9)
                    if rel > worst:
                        worst, worst_n = rel, n
                # int8 with a per-row scale cannot be worse than half a step,
                # and a row's peak sets the step, so 1/254 bounds it.
                limit = 1.0 / 254 if quant else 0.0
                bad = worst > limit + 1e-6
                ok = ok and not bad
                where = f"at {worst_n}" if worst_n else "bit-exact"
                print(f"  {kind:12s} {'int8' if quant else 'f32 ':4s} "
                      f"{len(w):3d} tensors  worst {worst:.3e} {where}"
                      f"{'   <-- OVER ' + format(limit, '.3e') if bad else ''}")
        # A guard nothing exercises is a guard that might not work. The
        # accounting in convert_hybrid is the only thing standing between a
        # tensor this writer forgot and a model with a layer of noise in it,
        # so make it fire on purpose.
        cfg, w = _synthetic(False)
        w[PREFIX + "layers.0.linear_attn.something_new"] = np.zeros((4, 4), np.float32)
        src = Path(tmp) / "negative"
        src.mkdir(exist_ok=True)
        _write_shards(src, w)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                convert.convert_hybrid(cfg, convert.read_sharded(src),
                                       Path(tmp) / "neg.bin", True, 128, "qwen3_5")
            print("  unaccounted tensor was NOT caught")
            ok = False
        except SystemExit as e:
            caught = "neither written nor skipped" in str(e)
            print(f"  unaccounted tensor {'caught' if caught else 'raised the wrong error: ' + str(e)}")
            ok = ok and caught

    print("[v4] round-trip ok" if ok else "[v4] round-trip FAILED")
    return 0 if ok else 1


def main():
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) != 2:
        raise SystemExit("usage: v4.py <model.bin> | v4.py --selftest")
    t, cfg = load(sys.argv[1])
    print(json.dumps({k: v for k, v in cfg.items() if k != "layer_types"},
                     indent=2))
    print("schedule " + "".join("F" if k == "full_attention" else "L"
                                for k in cfg["layer_types"]))
    print(f"{len(t)} tensors, {sum(v.nbytes for v in t.values()) / 2**20:.0f} MiB as f32")


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)

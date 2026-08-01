#!/usr/bin/env python3
"""Compute reference logits from a converted GLaDOS model, in numpy.

The oracle for the kernel's int8 forward pass. This reads the *converted* file
rather than the original safetensors, so it exercises the same layout,
quantisation, head geometry and rope_theta the kernel will -- a bug in
convert.py shows up here as well as there, and only a bug in the Rust shows up
as a mismatch between the two.

Feed the same token ids to `logits <ids>` in GLaDOS and compare. A stride
miscomputed, sign extension dropped in the AVX2 widening, QK-Norm applied on
the wrong side of RoPE, or the wrong RoPE base all produce fluent-looking
nonsense rather than an error, so a numeric comparison is the only thing that
actually settles it.

`--generate` is the cheap end of the same idea. An 0.6B instruction-tuned model
whose attention path is wired correctly writes coherent English; one with the
wrong head width or a missing QK-Norm writes confident gibberish. That is a
weaker check than comparing logits, but it needs no second implementation and
it catches everything structural.

Weights stay int8 in memory and are dequantised a block of rows at a time.
Materialising the whole model as f32 would be 2.4 GB for Qwen3-0.6B, most of it
the 151936-row classifier, to no benefit -- this is the reference and it is
allowed to be slow, but it is not allowed to need more RAM than the machine has.
"""

import struct
import sys

import numpy as np

HEADER = 64
FLAG_QK_NORM = 1 << 0
FLAG_ROPE_INTERLEAVED = 1 << 1
# Rows dequantised at once. Bounds the temporary at a few tens of MB even for
# the classifier.
BLOCK = 4096


def load(path):
    blob = np.fromfile(path, dtype=np.uint8)
    magic = blob[:8].tobytes()
    if magic != b"GLADOSM2":
        raise SystemExit(f"not a GLaDOS model file: {magic!r}")
    head = blob[:HEADER].tobytes()
    (version, dim, hidden, layers, heads, kv_heads, raw_vocab, seq, theta, quant) = (
        struct.unpack_from("<Iiiiiiii f I", head, 8)
    )
    if quant != 1:
        raise SystemExit("only int8 files are supported here")
    if version not in (2, 3):
        raise SystemExit(f"unknown version {version}")

    # v2 described a Llama and had no room for any of this; its defaults are
    # exactly the Llama ones.
    if version >= 3:
        head_dim, norm_eps, flags = struct.unpack_from("<i f I", head, 48)
    else:
        head_dim, norm_eps, flags = dim // heads, 1e-5, 0

    cfg = dict(
        version=version, dim=dim, hidden=hidden, layers=layers,
        heads=heads, kv_heads=kv_heads, head_dim=head_dim,
        vocab=abs(raw_vocab), seq=seq, theta=theta, tied=raw_vocab > 0,
        eps=norm_eps, qk_norm=bool(flags & FLAG_QK_NORM),
        rope_style="interleaved" if flags & FLAG_ROPE_INTERLEAVED else "half",
    )
    cfg["q_dim"] = heads * head_dim
    cfg["kv_dim"] = kv_heads * head_dim

    pos = HEADER
    raw = blob

    def q8(rows, cols):
        """One int8 tensor, kept quantised: (values, per-row scales)."""
        nonlocal pos
        scales = raw[pos:pos + rows * 4].view(np.float32)
        pos += rows * 4
        data = raw[pos:pos + rows * cols].view(np.int8).reshape(rows, cols)
        pos += rows * cols
        return data, scales

    def f32(n):
        nonlocal pos
        out = raw[pos:pos + n * 4].view(np.float32).copy()
        pos += n * 4
        return out

    d, h, l = dim, hidden, layers
    q_dim, kv = cfg["q_dim"], cfg["kv_dim"]

    w = {}
    w["embed"] = q8(cfg["vocab"], d)
    w["rms_att"] = [f32(d) for _ in range(l)]
    if cfg["qk_norm"]:
        w["q_norm"] = [f32(head_dim) for _ in range(l)]
        w["k_norm"] = [f32(head_dim) for _ in range(l)]
    w["wq"] = [q8(q_dim, d) for _ in range(l)]
    w["wk"] = [q8(kv, d) for _ in range(l)]
    w["wv"] = [q8(kv, d) for _ in range(l)]
    w["wo"] = [q8(d, q_dim) for _ in range(l)]
    w["rms_ffn"] = [f32(d) for _ in range(l)]
    w["w1"] = [q8(h, d) for _ in range(l)]
    w["w2"] = [q8(d, h) for _ in range(l)]
    w["w3"] = [q8(h, d) for _ in range(l)]
    w["rms_final"] = f32(d)
    w["wcls"] = w["embed"] if cfg["tied"] else q8(cfg["vocab"], d)

    # The layout is walked, not seeked. Landing anywhere but the end means the
    # reader and the writer disagree about a shape, which is the failure this
    # whole file exists to catch.
    if pos != len(raw):
        raise SystemExit(f"layout walked to {pos} but the file is {len(raw)}")
    return cfg, w


def mv(mat, x):
    """out = mat @ x, dequantising a block of rows at a time."""
    data, scales = mat
    rows = data.shape[0]
    out = np.empty(rows, dtype=np.float32)
    for i in range(0, rows, BLOCK):
        j = min(i + BLOCK, rows)
        out[i:j] = (data[i:j].astype(np.float32) @ x) * scales[i:j]
    return out


def row(mat, i):
    data, scales = mat
    return data[i].astype(np.float32) * scales[i]


def rmsnorm(x, weight, eps):
    return x / np.sqrt((x * x).mean() + eps) * weight


def rope(cfg, vec, pos, hs):
    """Rotary embedding, in place, over every head packed into `vec`.

    Two conventions exist and they are not compatible:

      * "interleaved" pairs dimension 2i with 2i+1. This is what llama2.c does
        and what the original GPT-NeoX paper describes.
      * "half" pairs dimension i with i + head_size/2. This is what
        `rotate_half` in HuggingFace's modeling code does, and therefore what
        every checkpoint trained through transformers expects -- Llama, Qwen,
        SmolLM2. SmolLM2's config even says so: `rope_interleaved: false`.

    Applying the wrong one does not fail. Both are norm-preserving rotations by
    the same set of angles, so activations stay in range and the model stays
    fluent -- it just attends by a scrambled notion of distance, which reads as
    a model that is topically aware and factually useless.
    """
    theta = cfg["theta"]
    half = hs // 2
    idx = np.arange(half, dtype=np.float32)
    inv = 1.0 / (theta ** (2.0 * idx / hs))
    ang = pos * inv
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    for base in range(0, len(vec), hs):
        head = vec[base:base + hs]
        if cfg["rope_style"] == "interleaved":
            a = head[0::2].copy()
            b = head[1::2].copy()
            head[0::2] = a * c - b * s
            head[1::2] = a * s + b * c
        else:
            a = head[:half].copy()
            b = head[half:].copy()
            head[:half] = a * c - b * s
            head[half:] = b * c + a * s


def forward(cfg, w, ids, cache=None, start=0):
    """Run `ids` starting at position `start`, returning the final logits.

    `cache` is a preallocated (kcache, vcache) pair. Passing one lets a caller
    decode incrementally instead of replaying the whole prefix per token, which
    is the difference between seconds and minutes at this size.
    """
    d, h, l = cfg["dim"], cfg["hidden"], cfg["layers"]
    heads, kv_heads = cfg["heads"], cfg["kv_heads"]
    hs, q_dim, kv = cfg["head_dim"], cfg["q_dim"], cfg["kv_dim"]
    kv_mul = heads // kv_heads
    eps = cfg["eps"]
    n = start + len(ids)

    if cache is None:
        kcache = np.zeros((l, n, kv), dtype=np.float32)
        vcache = np.zeros((l, n, kv), dtype=np.float32)
    else:
        kcache, vcache = cache
    logits = None

    for step, tok in enumerate(ids):
        pos = start + step
        x = row(w["embed"], tok)
        for li in range(l):
            xb = rmsnorm(x, w["rms_att"][li], eps)
            q = mv(w["wq"][li], xb)
            k = mv(w["wk"][li], xb)
            v = mv(w["wv"][li], xb)

            # QK-Norm, per head, before RoPE. Applied after it, the rotation
            # would be rescaled and position would stop meaning what it means.
            if cfg["qk_norm"]:
                for hi in range(heads):
                    o = hi * hs
                    q[o:o + hs] = rmsnorm(q[o:o + hs], w["q_norm"][li], eps)
                for hi in range(kv_heads):
                    o = hi * hs
                    k[o:o + hs] = rmsnorm(k[o:o + hs], w["k_norm"][li], eps)

            rope(cfg, q, pos, hs)
            rope(cfg, k, pos, hs)

            kcache[li, pos] = k
            vcache[li, pos] = v

            att_out = np.zeros(q_dim, dtype=np.float32)
            scale = 1.0 / np.sqrt(hs)
            for hi in range(heads):
                qo = hi * hs
                ko = (hi // kv_mul) * hs
                ks = kcache[li, :pos + 1, ko:ko + hs]
                scores = (ks @ q[qo:qo + hs]) * scale
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                att_out[qo:qo + hs] = scores @ vcache[li, :pos + 1, ko:ko + hs]

            x = x + mv(w["wo"][li], att_out)

            xb = rmsnorm(x, w["rms_ffn"][li], eps)
            hb = mv(w["w1"][li], xb)
            hb2 = mv(w["w3"][li], xb)
            hb = hb / (1.0 + np.exp(-hb)) * hb2  # SwiGLU
            x = x + mv(w["w2"][li], hb)

        if pos == n - 1:
            xb = rmsnorm(x, w["rms_final"], eps)
            logits = mv(w["wcls"], xb)
    return logits


def new_cache(cfg, capacity):
    l, kv = cfg["layers"], cfg["kv_dim"]
    return (
        np.zeros((l, capacity, kv), dtype=np.float32),
        np.zeros((l, capacity, kv), dtype=np.float32),
    )


def generate(cfg, w, ids, steps, tok=None, temperature=0.0):
    """Greedy (or low-temperature) decoding, printing as it goes.

    Keeps one KV cache across steps, so the prompt is run once and each new
    token costs a single position rather than a replay of everything before it.
    """
    rng = np.random.default_rng(0)
    out = list(ids)
    cache = new_cache(cfg, len(ids) + steps + 1)
    logits = forward(cfg, w, ids, cache, 0)
    for _ in range(steps):
        if temperature <= 0:
            nxt = int(np.argmax(logits))
        else:
            p = logits / temperature
            p = np.exp(p - p.max())
            p /= p.sum()
            nxt = int(rng.choice(len(p), p=p))
        out.append(nxt)
        if tok is not None:
            sys.stdout.write(tok.decode([nxt]))
            sys.stdout.flush()
        if cfg.get("eos") is not None and nxt == cfg["eos"]:
            break
        logits = forward(cfg, w, [nxt], cache, len(out) - 1)
    return out


def main():
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(
            "usage: reference.py <model.bin> [--tokenizer t.json]\n"
            "         [<id> ...] [--generate N --prompt TEXT] [--temp T]"
        )
    path = argv.pop(0)

    def opt(name, default=None):
        if name in argv:
            i = argv.index(name)
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        return default

    tok_path = opt("--tokenizer")
    steps = int(opt("--generate", "0"))
    prompt = opt("--prompt")
    temp = float(opt("--temp", "0"))

    cfg, w = load(path)
    cfg["eos"] = None
    print("  " + "  ".join(
        f"{k} {cfg[k]}" for k in
        ("version", "dim", "layers", "heads", "kv_heads", "head_dim", "vocab")
    ))
    print(f"  q_dim {cfg['q_dim']}  kv_dim {cfg['kv_dim']}  "
          f"qk_norm {cfg['qk_norm']}  eps {cfg['eps']:g}  theta {cfg['theta']:g}  "
          f"rope {cfg['rope_style']}")

    tok = None
    if tok_path:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tok_path)
        cfg["eos"] = tok.token_to_id("<|im_end|>")
        print(f"  tokenizer {tok.get_vocab_size()} tokens, eos {cfg['eos']}")

    if prompt is not None:
        if tok is None:
            raise SystemExit("--prompt needs --tokenizer")
        ids = tok.encode(prompt, add_special_tokens=False).ids
    else:
        ids = [int(a) for a in argv]
    if not ids:
        raise SystemExit("no input: pass --prompt or token ids")

    if steps:
        print(f"\n  prompt {len(ids)} tokens, generating {steps}\n")
        sys.stdout.write(prompt or "")
        generate(cfg, w, ids, steps, tok, temp)
        print()
    else:
        logits = forward(cfg, w, ids)
        order = np.argsort(-logits)[:10]
        print(f"\n  ids {ids}\n  top 10:")
        for rank, i in enumerate(order):
            s = repr(tok.decode([int(i)])) if tok else ""
            print(f"  {rank + 1:2}. id {int(i):7}  logit {float(logits[i]):9.4f}  {s}")
        print(f"\n  first 8 logits: {' '.join(f'{v:.4f}' for v in logits[:8])}")


if __name__ == "__main__":
    main()

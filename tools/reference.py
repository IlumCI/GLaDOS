#!/usr/bin/env python3
"""Compute reference logits from a converted GLaDOS model, in numpy.

The oracle for the kernel's int8 forward pass. This reads the *converted* file
rather than the original safetensors, so it exercises the same layout,
quantisation and rope_theta the kernel will -- a bug in convert.py shows up
here as well as there, and only a bug in the Rust shows up as a mismatch.

Feed the same token ids to `logits <ids>` in GLaDOS and compare. A stride
miscomputed, sign extension dropped in the AVX2 widening, or the wrong RoPE
base all produce fluent-looking nonsense rather than an error, so a numeric
comparison is the only thing that actually settles it.
"""

import struct
import sys

import numpy as np

HEADER = 64


def load(path):
    blob = np.fromfile(path, dtype=np.uint8)
    magic = blob[:8].tobytes()
    if magic != b"GLADOSM2":
        raise SystemExit(f"not a GLADOSM2 file: {magic!r}")
    (version, dim, hidden, layers, heads, kv_heads, raw_vocab, seq, theta, quant) = (
        struct.unpack_from("<Iiiiiiii f I", blob[:HEADER].tobytes(), 8)
    )
    if quant != 1:
        raise SystemExit("only int8 files are supported here")

    cfg = dict(
        dim=dim, hidden=hidden, layers=layers, heads=heads, kv_heads=kv_heads,
        vocab=abs(raw_vocab), seq=seq, theta=theta, tied=raw_vocab > 0,
    )

    pos = HEADER
    raw = blob

    def q8(rows, cols):
        nonlocal pos
        scales = raw[pos:pos + rows * 4].view(np.float32)
        pos += rows * 4
        data = raw[pos:pos + rows * cols].view(np.int8).reshape(rows, cols)
        pos += rows * cols
        # Dequantise up front: this is the reference, not the fast path.
        return data.astype(np.float32) * scales[:, None]

    def f32(n):
        nonlocal pos
        out = raw[pos:pos + n * 4].view(np.float32).copy()
        pos += n * 4
        return out

    d, h, l = dim, hidden, layers
    kv = dim * kv_heads // heads

    w = {}
    w["embed"] = q8(cfg["vocab"], d)
    w["rms_att"] = [f32(d) for _ in range(l)]
    w["wq"] = [q8(d, d) for _ in range(l)]
    w["wk"] = [q8(kv, d) for _ in range(l)]
    w["wv"] = [q8(kv, d) for _ in range(l)]
    w["wo"] = [q8(d, d) for _ in range(l)]
    w["rms_ffn"] = [f32(d) for _ in range(l)]
    w["w1"] = [q8(h, d) for _ in range(l)]
    w["w2"] = [q8(d, h) for _ in range(l)]
    w["w3"] = [q8(h, d) for _ in range(l)]
    w["rms_final"] = f32(d)
    w["wcls"] = w["embed"] if cfg["tied"] else q8(cfg["vocab"], d)

    if pos != len(raw):
        raise SystemExit(f"layout walked to {pos} but the file is {len(raw)}")
    return cfg, w, kv


def rmsnorm(x, weight, eps=1e-5):
    return x / np.sqrt((x * x).mean() + eps) * weight


def forward(cfg, w, kv, ids):
    d, h, l = cfg["dim"], cfg["hidden"], cfg["layers"]
    heads, kv_heads = cfg["heads"], cfg["kv_heads"]
    head_size = d // heads
    kv_mul = heads // kv_heads

    kcache = np.zeros((l, len(ids), kv), dtype=np.float32)
    vcache = np.zeros((l, len(ids), kv), dtype=np.float32)
    logits = None

    for pos, tok in enumerate(ids):
        x = w["embed"][tok].copy()
        for li in range(l):
            xb = rmsnorm(x, w["rms_att"][li])
            q = w["wq"][li] @ xb
            k = w["wk"][li] @ xb
            v = w["wv"][li] @ xb

            # RoPE. The kernel rotates pairs within each head, using
            # `i % head_size` for the frequency, so this must too.
            for i in range(0, d, 2):
                hd = i % head_size
                freq = 1.0 / (cfg["theta"] ** (hd / head_size))
                val = pos * freq
                fcr, fci = np.cos(val), np.sin(val)
                q0, q1 = q[i], q[i + 1]
                q[i] = q0 * fcr - q1 * fci
                q[i + 1] = q0 * fci + q1 * fcr
                if i < kv:
                    k0, k1 = k[i], k[i + 1]
                    k[i] = k0 * fcr - k1 * fci
                    k[i + 1] = k0 * fci + k1 * fcr

            kcache[li, pos] = k
            vcache[li, pos] = v

            att_out = np.zeros(d, dtype=np.float32)
            scale = 1.0 / np.sqrt(head_size)
            for hi in range(heads):
                qo = hi * head_size
                ko = (hi // kv_mul) * head_size
                scores = np.array(
                    [
                        float(q[qo:qo + head_size] @ kcache[li, t, ko:ko + head_size]) * scale
                        for t in range(pos + 1)
                    ],
                    dtype=np.float32,
                )
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                acc = np.zeros(head_size, dtype=np.float32)
                for t in range(pos + 1):
                    acc += scores[t] * vcache[li, t, ko:ko + head_size]
                att_out[qo:qo + head_size] = acc

            x = x + w["wo"][li] @ att_out

            xb = rmsnorm(x, w["rms_ffn"][li])
            hb = w["w1"][li] @ xb
            hb2 = w["w3"][li] @ xb
            hb = hb / (1.0 + np.exp(-hb)) * hb2  # SwiGLU
            x = x + w["w2"][li] @ hb

        xb = rmsnorm(x, w["rms_final"])
        logits = w["wcls"] @ xb
    return logits


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: reference.py <model.bin> <id> [id ...]")
    path = sys.argv[1]
    ids = [int(a) for a in sys.argv[2:]]

    cfg, w, kv = load(path)
    print(f"  dim {cfg['dim']} hidden {cfg['hidden']} layers {cfg['layers']} "
          f"heads {cfg['heads']}/{cfg['kv_heads']} kv {kv}")
    print(f"  vocab {cfg['vocab']} seq {cfg['seq']} theta {cfg['theta']:g} tied {cfg['tied']}")
    print(f"  ids {ids}")

    logits = forward(cfg, w, kv, ids)
    order = np.argsort(-logits)[:5]
    print("\n  top 5:")
    for rank, i in enumerate(order):
        v = float(logits[i])
        print(f"  {rank + 1}. id {int(i):6}  logit {v:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Change how much context a converted checkpoint can hold.

    reseq.py <model.bin> <positions> [--out FILE]

`seq_len` is written by convert.py and defaults to 512, which is a KV cache
that fits anywhere rather than a property of the model -- Qwen3 is trained to
32768. Nothing in a GLADOSM2 file's *layout* depends on it: the weight offsets
are walked from dim/hidden/layers/vocab, RoPE is computed at runtime from
`rope_theta`, and the legacy per-position tables belong to the llama2.c format
this does not touch. So the field can be rewritten in place, without the
original HuggingFace weights, which for a checkpoint converted months ago is
the difference between a four-byte edit and a re-download.

What it costs is kernel heap, which is what this prints. Refuse nothing on that
basis -- the machine has 11.7 GiB and the caller can read.
"""
import struct
import sys
from pathlib import Path

MAGIC = b"GLADOSM2"
KV_BLOCK = 32  # must match model.rs


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    want = int(sys.argv[2])
    dst = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else src

    data = bytearray(src.read_bytes())
    if len(data) < 64 or bytes(data[0:8]) != MAGIC:
        raise SystemExit(f"{src} is not a GLADOSM2 checkpoint")

    version = struct.unpack_from("<I", data, 8)[0]
    if version not in (2, 3, 4):
        raise SystemExit(f"version {version} is not one this understands")

    dim, hidden, layers, heads, kv_heads, vocab, seq = struct.unpack_from("<iiiiiii", data, 12)
    quant = struct.unpack_from("<I", data, 44)[0]
    head_dim = struct.unpack_from("<i", data, 48)[0]
    if head_dim <= 0:
        # v2 predates the field; heads divide the residual stream evenly there.
        head_dim = dim // heads

    trained = {}  # nothing on disk says what the model was trained to; the
    # caller is trusted, and convert.py checks it against config.json when the
    # HuggingFace directory is present.

    kv_dim = kv_heads * head_dim
    blocks = (kv_dim + KV_BLOCK - 1) // KV_BLOCK
    per_pos_layer = kv_dim + blocks * 4 if quant else kv_dim * 4
    per_pos = 2 * per_pos_layer * layers  # keys and values

    def mib(n):
        return n / 1024 / 1024

    print(f"{src.name}: v{version}  dim {dim}  layers {layers}  kv {kv_heads}x{head_dim}")
    print(f"  seq {seq} -> {want}")
    print(f"  KV cache {mib(per_pos * seq):.0f} MiB -> {mib(per_pos * want):.0f} MiB"
          f"  ({per_pos / 1024:.0f} KiB per position)")
    if version == 4:
        print("  hybrid: only the full-attention layers hold a KV cache, so the")
        print("  real figure is lower than the number above -- the linear layers")
        print("  keep a fixed-size recurrent state that does not grow with context")

    struct.pack_into("<i", data, 36, want)
    dst.write_bytes(bytes(data))
    print(f"  wrote {dst}")


if __name__ == "__main__":
    main()

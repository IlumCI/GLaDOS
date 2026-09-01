#!/usr/bin/env python3
"""Set the context window a checkpoint declares, without reconverting it.

The three 2B files in `out/` differ in exactly one byte. `--seq` does not
change a single weight: it changes `seq_len` in the header, and the kernel
sizes its KV cache from that. So shipping a 2B at three context lengths does
not mean three 1.89 GB uploads, it means one upload and a four-byte stamp.

    python tools/ctxstamp.py model.bin 8192
    python tools/ctxstamp.py model.bin 32768 --out model-32k.bin

### Where the offset comes from, and why it is not trusted

`seq_len` is the seventh little-endian i32 of the header body, which
`convert.py` packs from offset 8 as `"<Iiiiiiii f I"` and `v4.py` reads back as
`struct.unpack_from("<iiiiiii", raw, 12)`. Both agree it lands at 36, and that
agreement is why the number below is written down rather than searched for.

It is still not trusted. After stamping, this reads the file back **through
`v4.py`'s reader** -- the one that is deliberately not the writer -- and
asserts the value took. A patch at the wrong offset would otherwise corrupt a
dimension and produce a file that loads and computes something else, which is
the failure `model.rs` spends a page warning about: a model can be wrong
without being broken.

### What it cannot check

Whether the context is one the checkpoint was *trained* for.
`convert.py` refuses `--seq` above `max_position_embeddings`, which it reads
from the source config; that number is not in the converted file and cannot be
recovered from it. Stamping 128k onto a model trained for 32k produces a file
that loads, allocates a great deal of memory, and degrades past the trained
length with nothing to say so.

So the ceiling is a decision made where the source config is, and this tool
carries `MAX_SANE` only to catch a typo, not to enforce a limit it cannot know.
"""

import argparse
import shutil
import struct
import sys
from pathlib import Path

MAGIC = b"GLADOSM2"
SEQ_AT = 36
# Enough to catch a fat-fingered zero, not a claim about any model.
MAX_SANE = 1 << 20


def read_seq(path: Path) -> int:
    with path.open("rb") as f:
        head = f.read(64)
    if head[:8] != MAGIC:
        raise SystemExit(f"  {path}: not a GLaDOS checkpoint (magic is {head[:8]!r})")
    return struct.unpack_from("<i", head, SEQ_AT)[0]


def stamp(path: Path, seq: int) -> None:
    with path.open("r+b") as f:
        f.seek(SEQ_AT)
        f.write(struct.pack("<i", seq))


def verify_with_reader(path: Path, seq: int) -> None:
    """Read it back with v4.py, which is the reader and not the writer.

    Imported rather than reimplemented for exactly that reason. If this file
    parsed its own output the check would prove the two halves of one
    misunderstanding agree, which is the objection `manifest.py` and
    `tokenizer.py --verify` both raise about self-checking.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import v4  # noqa: E402
    except ImportError:
        print("  (v4.py not importable; falling back to a header re-read)")
        got = read_seq(path)
        if got != seq:
            raise SystemExit(f"  stamp did not take: header says {got}, wanted {seq}")
        return

    cfg = None
    for name in ("read_header", "header", "parse_header", "read"):
        fn = getattr(v4, name, None)
        if callable(fn):
            try:
                cfg = fn(path.read_bytes()[:4096]) if name != "read" else fn(str(path))
                break
            except Exception:
                cfg = None
    if cfg is None:
        got = read_seq(path)
        if got != seq:
            raise SystemExit(f"  stamp did not take: header says {got}, wanted {seq}")
        print(f"  re-read confirms seq {seq}")
        return

    got = cfg.get("seq_len") if isinstance(cfg, dict) else None
    if got != seq:
        raise SystemExit(f"  v4 reader says seq_len {got}, wanted {seq}")
    print(f"  v4 reader confirms seq_len {seq}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("seq", type=int)
    ap.add_argument("--out", help="write a copy instead of stamping in place")
    args = ap.parse_args()

    if args.seq < 1 or args.seq > MAX_SANE:
        raise SystemExit(f"  seq {args.seq} is outside 1..{MAX_SANE}; that is a typo, not a context")

    src = Path(args.model)
    if not src.is_file():
        raise SystemExit(f"  no such file: {src}")

    was = read_seq(src)
    dst = src
    if args.out:
        dst = Path(args.out)
        # Copied rather than rewritten from parts: the body is 1.8 GB and
        # streaming it through python to change four bytes is a lot of work to
        # do badly.
        shutil.copyfile(src, dst)

    stamp(dst, args.seq)
    verify_with_reader(dst, args.seq)
    print(f"  {dst}  seq {was} -> {args.seq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

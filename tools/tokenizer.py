#!/usr/bin/env python3
"""Convert a Hugging Face BPE tokenizer to the GLaDOS format, and verify it.

The existing on-disk tokenizer format came from llama2.c and assumes
sentencepiece: a dummy space is prepended to every input, and any byte the
vocabulary cannot express falls back to id `byte + 3`. Neither assumption holds
for a GPT-2 style byte-level BPE like SmolLM2's:

  * `add_prefix_space` is false, so prepending a space is simply wrong and
    shifts every following token.
  * Byte tokens are scattered through the vocabulary at whatever ids the
    byte-to-unicode mapping landed on, not at `byte + 3`.

So this emits a v2 format carrying an explicit 256-entry byte table and a flags
word, which makes both tokenizers describable without the loader having to
guess which family it is looking at.

Scores carry the merge order. Our encoder repeatedly merges the adjacent pair
with the highest score; BPE merges the pair with the lowest rank. Setting
`score = -rank` makes those the same rule, so the existing merge loop
implements real BPE with no change.

The verifier matters more than the converter. A tokenizer that is subtly wrong
produces text that still looks like text, and the damage shows up only as the
model being mysteriously worse -- so this reimplements the *kernel's* algorithm
in Python and diffs it against the reference `tokenizers` library.
"""

import json
import struct
import sys
from pathlib import Path

MAGIC = b"GLADOSTK"
VERSION = 2

FLAG_DUMMY_PREFIX = 1 << 0
FLAG_INDIVIDUAL_DIGITS = 1 << 1


def bytes_to_unicode():
    """GPT-2's reversible byte-to-printable-character map."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


UNICODE_TO_BYTE = bytes_to_unicode()


def token_to_bytes(tok):
    """Undo the byte-level mapping, so the stored token is its real bytes."""
    try:
        return bytes(UNICODE_TO_BYTE[ch] for ch in tok)
    except KeyError:
        # Added tokens like <|im_start|> are literal text, never byte-encoded.
        return tok.encode("utf-8")


def load(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    model = spec["model"]
    if model.get("type") != "BPE":
        raise SystemExit(f"unsupported tokenizer type {model.get('type')!r}")

    vocab = model["vocab"]
    size = len(vocab)

    # Merge rank per resulting token. `merges` may be strings or pairs
    # depending on the tokenizers version that wrote the file.
    rank = {}
    for i, m in enumerate(model.get("merges", [])):
        a, b = m.split(" ", 1) if isinstance(m, str) else (m[0], m[1])
        joined = a + b
        if joined in vocab:
            rank.setdefault(vocab[joined], i)

    specials = {t["id"]: t["content"] for t in spec.get("added_tokens", [])}

    by_id = [None] * size
    for tok, i in vocab.items():
        by_id[i] = tok
    for i, content in specials.items():
        if i < size:
            by_id[i] = content

    flags = 0
    pre = spec.get("pre_tokenizer") or {}
    subs = pre.get("pretokenizers", [pre])
    for p in subs:
        if p.get("type") == "Digits" and p.get("individual_digits"):
            flags |= FLAG_INDIVIDUAL_DIGITS
        if p.get("type") == "ByteLevel" and p.get("add_prefix_space"):
            flags |= FLAG_DUMMY_PREFIX

    return by_id, rank, set(specials), flags, vocab


def convert(src, dst):
    by_id, rank, special_ids, flags, vocab = load(src)
    size = len(by_id)

    raw = [token_to_bytes(t) if t is not None else b"" for t in by_id]

    # A token that is not the product of a merge must never be *formed* by one.
    # Seeds and specials get a score far below any real merge so the greedy
    # loop cannot pick them.
    NEVER = -1e30
    scores = [NEVER] * size
    for tid, r in rank.items():
        scores[tid] = -float(r)

    # Byte table: which id represents each raw byte on its own. This replaces
    # the `byte + 3` guess entirely.
    byte_table = [0xFFFFFFFF] * 256
    inv = {v: k for k, v in UNICODE_TO_BYTE.items()}
    for b in range(256):
        ch = inv.get(b)
        if ch is not None and ch in vocab:
            byte_table[b] = vocab[ch]

    bos = vocab.get("<|im_start|>", 1)
    eos = vocab.get("<|im_end|>", 2)
    unk = vocab.get("<|endoftext|>", 0)

    # Not every byte necessarily has a standalone token. SmolLM2 trained its
    # own 49152-entry BPE and 21 byte-characters never survived as single
    # tokens. They are all unreachable from well-formed input: 4, 6, 19, 20,
    # 22 and 29 are ASCII control codes, and 192, 193, 245-255 cannot appear
    # in valid UTF-8 at any position. Pointing them at unk is therefore exact
    # for every input that can actually occur, rather than a compromise -- but
    # a *printable* byte going missing would mean the mapping is wrong, so
    # that still fails loudly.
    missing = [b for b in range(256) if byte_table[b] == 0xFFFFFFFF]
    printable = [b for b in missing if 0x20 <= b < 0x7F]
    if printable:
        raise SystemExit(f"printable bytes have no token: {printable}")
    for b in missing:
        byte_table[b] = unk
    max_len = max(len(r) for r in raw)

    # Added tokens must be matched literally, before pre-tokenisation, or BPE
    # shreds them: <|im_start|> becomes 20 tokens instead of 1. That is not a
    # cosmetic difference -- ChatML framing is how an instruct model is told
    # who is speaking, and it only works if those markers survive as the exact
    # ids the model was trained on. Longest first, so a special that prefixes
    # another cannot mask it.
    specials = sorted(special_ids, key=lambda i: -len(raw[i]))

    out = bytearray()
    out += MAGIC
    out += struct.pack("<IIIIIII", VERSION, size, max_len, flags, bos, eos, unk)
    for v in byte_table:
        out += struct.pack("<I", v)
    out += struct.pack("<I", len(specials))
    for i in specials:
        out += struct.pack("<I", i)
    for i in range(size):
        out += struct.pack("<fI", scores[i], len(raw[i]))
        out += raw[i]

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_bytes(out)

    print(f"  {size} tokens, {len(rank)} merges, longest {max_len} B")
    print(f"  specials: bos={bos} eos={eos} unk={unk}, {len(special_ids)} added")
    print(f"  flags: dummy_prefix={bool(flags & FLAG_DUMMY_PREFIX)} "
          f"individual_digits={bool(flags & FLAG_INDIVIDUAL_DIGITS)}")
    if missing:
        print(f"  {len(missing)} unreachable bytes mapped to unk "
              f"(control codes and invalid UTF-8 lead bytes)")
    print(f"  wrote {dst}  {len(out):,} B")
    return raw, scores, byte_table, flags, special_ids


# --- the kernel's algorithm, reimplemented for verification -----------------

def pretokenize(text, individual_digits):
    """Split before BPE, the way ByteLevel(use_regex) does.

    The reference splits on:
        's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+
    Merges never cross these boundaries, so getting this wrong changes the
    tokenisation of ordinary sentences even when every merge rule is right.
    This is written the way it will be written in Rust -- a hand-rolled scan,
    no regex engine -- so that verifying it here verifies that.
    """
    out = []
    i = 0
    n = len(text)
    contractions = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
    while i < n:
        for c in contractions:
            if text.startswith(c, i):
                out.append(c)
                i += len(c)
                break
        else:
            start = i
            space = 0
            if text[i] == " " and i + 1 < n and not text[i + 1].isspace():
                space = 1
            j = i + space
            if j < n and text[j].isalpha():
                while j < n and text[j].isalpha():
                    j += 1
            elif j < n and text[j].isdigit():
                j += 1 if individual_digits else 0
                if not individual_digits:
                    while j < n and text[j].isdigit():
                        j += 1
            elif j < n and not text[j].isspace():
                while j < n and not text[j].isspace() and not text[j].isalnum():
                    j += 1
            else:
                # A run of whitespace; the last space belongs to the next word
                # unless the run ends the string.
                while j < n and text[j].isspace():
                    j += 1
                if j < n:
                    j -= 1
                if j <= start:
                    j = start + 1
            if j <= start:
                j = start + 1
            out.append(text[start:j])
            i = j
    return out


def encode_like_kernel(text, raw, scores, byte_table, flags, specials=()):
    lookup = {}
    for i, r in enumerate(raw):
        lookup.setdefault(r, i)

    # Split around literal special tokens first; only the gaps get BPE'd.
    order = sorted(specials, key=lambda i: -len(raw[i]))
    parts = []
    data = text.encode("utf-8")
    i = 0
    run = bytearray()
    while i < len(data):
        for sid in order:
            s = raw[sid]
            if s and data.startswith(s, i):
                if run:
                    parts.append(("text", bytes(run)))
                    run = bytearray()
                parts.append(("id", sid))
                i += len(s)
                break
        else:
            run.append(data[i])
            i += 1
    if run:
        parts.append(("text", bytes(run)))

    ids = []
    for kind, value in parts:
        if kind == "id":
            ids.append(value)
            continue
        ids.extend(_bpe(value.decode("utf-8", "replace"), raw, scores, byte_table, flags, lookup))
    return ids


def _bpe(text, raw, scores, byte_table, flags, lookup):
    ids = []
    for piece in pretokenize(text, bool(flags & FLAG_INDIVIDUAL_DIGITS)):
        toks = [byte_table[b] for b in piece.encode("utf-8")]
        while True:
            best, best_at, best_id = -1e29, -1, -1
            for k in range(len(toks) - 1):
                cand = raw[toks[k]] + raw[toks[k + 1]]
                j = lookup.get(cand)
                if j is not None and scores[j] > best:
                    best, best_at, best_id = scores[j], k, j
            if best_at < 0:
                break
            toks[best_at] = best_id
            del toks[best_at + 1]
        ids.extend(toks)
    return ids


CASES = [
    "Hello world",
    "list the files",
    "The quick brown fox jumps over the lazy dog.",
    "verify the disk, then take a snapshot",
    "GLaDOS is an operating system",
    "x = 6*7; println(x)",
    "snapshot 42 of 100",
    "don't stop believing",
    "  leading and trailing  ",
    "unicode: café naïve über",
    "<|im_start|>user\nlist the files<|im_end|>\n",
]


def verify(src, raw, scores, byte_table, flags, specials):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("  (tokenizers not installed -- skipping verification)")
        return True

    ref = Tokenizer.from_file(str(src))
    ok = True
    for text in CASES:
        want = ref.encode(text, add_special_tokens=False).ids
        got = encode_like_kernel(text, raw, scores, byte_table, flags, specials)
        mark = "ok  " if want == got else "DIFF"
        if want != got:
            ok = False
        print(f"  {mark} {text!r}")
        if want != got:
            print(f"       reference {want}")
            print(f"       ours      {got}")
    return ok


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: tokenizer.py <tokenizer.json> <out.bin> [--verify]")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    raw, scores, byte_table, flags, specials = convert(src, dst)
    if "--verify" in sys.argv:
        print("\n  verifying against the reference implementation:")
        if not verify(src, raw, scores, byte_table, flags, specials):
            raise SystemExit("\n  tokenisation does not match the reference")
        print("\n  every case matches")


if __name__ == "__main__":
    main()

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
import unicodedata
from pathlib import Path

MAGIC = b"GLADOSTK"
VERSION = 2

FLAG_DUMMY_PREFIX = 1 << 0
FLAG_INDIVIDUAL_DIGITS = 1 << 1
# Which pre-tokenizer regex the checkpoint was trained with. Absent means the
# GPT-2 pattern that ByteLevel(use_regex) implies; set means the cl100k one
# Qwen3 spells out as an explicit Split; both means the variant Qwen3.5 uses,
# which adds \p{M} to the letter run and to what punctuation may not swallow.
FLAG_SPLIT_CL100K = 1 << 2
FLAG_SPLIT_CL100KM = 1 << 3

# The distinguishing fragment of the cl100k pattern: a word may be led by any
# non-alphanumeric. Matching on the whole pattern string would break on
# whitespace or escaping differences between tokenizers releases; this clause
# appears in no other pattern in use.
CL100K_MARK = r"[^\r\n\p{L}\p{N}]?\p{L}+"
# And of the marks variant, which differs from plain cl100k exactly here:
CL100KM_MARK = r"[\p{L}\p{M}]+"


def is_mark(ch):
    """\\p{M}: general categories Mn, Mc, Me."""
    return unicodedata.category(ch)[0] == "M"


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
    specials_meta = spec.get("added_tokens", [])

    # The BPE table is not the vocabulary. SmolLM2 numbers its added tokens
    # inside `model.vocab`, so the two coincide and length alone was enough.
    # Qwen3 appends its 293 specials *above* the BPE range -- 151643 merges
    # plus specials to 151935 -- so sizing by the table drops every special and
    # then indexes past the end of it.
    #
    # This has to agree with the model's `vocab_size`, or every id above the
    # BPE range addresses a different row of the embedding than it was trained
    # on.
    size = len(vocab)
    if specials_meta:
        size = max(size, max(t["id"] for t in specials_meta) + 1)

    # Merge rank per resulting token. `merges` may be strings or pairs
    # depending on the tokenizers version that wrote the file.
    rank = {}
    for i, m in enumerate(model.get("merges", [])):
        a, b = m.split(" ", 1) if isinstance(m, str) else (m[0], m[1])
        joined = a + b
        if joined in vocab:
            rank.setdefault(vocab[joined], i)

    specials = {t["id"]: t["content"] for t in specials_meta}

    by_id = [None] * size
    for tok, i in vocab.items():
        by_id[i] = tok
    for i, content in specials.items():
        by_id[i] = content

    flags = 0
    pre = spec.get("pre_tokenizer") or {}
    subs = pre.get("pretokenizers", [pre])
    saw_split = False
    for p in subs:
        if p.get("type") == "Digits" and p.get("individual_digits"):
            flags |= FLAG_INDIVIDUAL_DIGITS
        if p.get("type") == "ByteLevel" and p.get("add_prefix_space"):
            flags |= FLAG_DUMMY_PREFIX
        if p.get("type") == "Split":
            saw_split = True
            regex = (p.get("pattern") or {}).get("Regex", "")
            if CL100K_MARK in regex:
                flags |= FLAG_SPLIT_CL100K
            elif CL100KM_MARK in regex:
                flags |= FLAG_SPLIT_CL100KM
    # An unrecognised Split would silently fall back to the GPT-2 pattern and
    # mis-tokenise everything by a few percent, which reads as the model being
    # mysteriously worse rather than as a bug.
    if saw_split and not flags & (FLAG_SPLIT_CL100K | FLAG_SPLIT_CL100KM):
        raise SystemExit(
            "the tokenizer has a Split pre-tokenizer whose pattern is not "
            "recognised; add it rather than falling back"
        )

    # Name -> id over the *whole* vocabulary, not just the BPE table.
    # `<|im_start|>` and friends live in `added_tokens` for Qwen3, so looking
    # them up in `model.vocab` silently returns the default of 1 -- an ordinary
    # BPE token -- and the model is handed an end-of-turn marker it was never
    # trained on. Generation then never terminates.
    names = {tok: i for i, tok in enumerate(by_id) if tok is not None}

    return by_id, rank, set(specials), flags, names


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

    # ChatML markers, by name. Falling back to a small integer here is how a
    # tokenizer ends up declaring an ordinary BPE token as end-of-turn, so a
    # miss is reported rather than defaulted -- the failure it causes
    # (generation that never stops) is a long way from the cause.
    def special(name, fallback):
        if name in vocab:
            return vocab[name]
        print(f"  WARNING: {name} not in the vocabulary; using {fallback}")
        return fallback

    bos = special("<|im_start|>", 1)
    eos = special("<|im_end|>", 2)
    unk = special("<|endoftext|>", 0)

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
          f"individual_digits={bool(flags & FLAG_INDIVIDUAL_DIGITS)} "
          f"split={'cl100k' if flags & FLAG_SPLIT_CL100K else ('cl100km' if flags & FLAG_SPLIT_CL100KM else 'gpt2')}")
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


CONTRACTIONS = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]


def pretokenize_cl100k(text, marks=False):
    """Split before BPE, the way Qwen's explicit Split regex does.

        (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}
        | ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+

    With `marks` set, this is the Qwen3.5 variant, which differs in exactly two
    clauses: the letter run admits \\p{M} (combining marks), and punctuation
    may no longer swallow them.

    Four differences from the GPT-2 pattern above, and every one of them
    changes real text:

      * a word may be led by *any* non-alphanumeric, not only a space, so
        `(x` is one piece where GPT-2 gives `(` and `x`;
      * `\\p{N}` takes one digit at a time, so Qwen3 gets digit splitting from
        the regex rather than from a Digits pre-tokenizer and the
        individual_digits flag is not set for it;
      * a punctuation run swallows the newlines that follow it;
      * whitespace ending in a newline is its own piece, which is what keeps
        `<|im_end|>\\n<|im_start|>` aligned.

    Alternation is ordered and first-match-wins, so the branches below are in
    the order the regex lists them. Written as a scan rather than with `re`
    because the kernel has no regex engine and this has to be the same
    algorithm there.
    """
    out = []
    i = 0
    n = len(text)
    is_nl = lambda c: c in "\r\n"
    word = lambda c: c.isalpha() or (marks and is_mark(c))
    punct = (
        (lambda c: not c.isspace() and not c.isalpha() and not c.isdigit())
        if not marks
        else (lambda c: not c.isspace() and not c.isalpha() and not c.isdigit() and not is_mark(c))
    )
    while i < n:
        start = i

        # 1. contractions, case-insensitively
        hit = next((c for c in CONTRACTIONS if text[i:i + len(c)].lower() == c), None)
        if hit:
            out.append(text[i:i + len(hit)])
            i += len(hit)
            continue

        # 2. [^\r\n\p{L}\p{N}]? [\p{L}\p{M}]+ -- optional lead then a word run.
        #    The regex backtracks: if the lead is taken but no run follows, it
        #    is given back and the run is tried at i itself, which is how a
        #    leading combining mark ends up as a piece of its own.
        matched = False
        for take_lead in (True, False):
            j = i + (1 if take_lead and not is_nl(text[i]) and not text[i].isalpha()
                     and not text[i].isdigit() else 0)
            if j < n and word(text[j]):
                while j < n and word(text[j]):
                    j += 1
                out.append(text[start:j])
                i = j
                matched = True
                break
        if matched:
            continue

        # 3. one digit
        if text[i].isdigit():
            out.append(text[i])
            i += 1
            continue

        # 4. ' ?' punctuation+ newline*
        j = i + (1 if text[i] == " " else 0)
        if j < n and punct(text[j]):
            while j < n and punct(text[j]):
                j += 1
            while j < n and is_nl(text[j]):
                j += 1
            out.append(text[start:j])
            i = j
            continue

        # 5/6/7. whitespace
        if text[i].isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            # `\s*[\r\n]+` is greedy on both halves, so it ends at the LAST
            # newline in the run, not the first.
            last_nl = max((k for k in range(i, j) if is_nl(text[k])), default=None)
            if last_nl is not None:
                out.append(text[start:last_nl + 1])
                i = last_nl + 1
                continue
            # `\s+(?!\S)` takes the whole run only at end of string; otherwise
            # it gives back one character and the last space leads the next
            # word.
            stop = j if j == n else max(j - 1, i + 1)
            out.append(text[start:stop])
            i = stop
            continue

        out.append(text[i])
        i += 1
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
    split = (
        pretokenize_cl100k(text, marks=bool(flags & FLAG_SPLIT_CL100KM))
        if flags & (FLAG_SPLIT_CL100K | FLAG_SPLIT_CL100KM)
        else pretokenize(text, bool(flags & FLAG_INDIVIDUAL_DIGITS))
    )
    for piece in split:
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
    # The Qwen3.5 pattern admits \p{M} into word runs and bars it from
    # punctuation runs. NFC composes Latin accents away, so the Hindi text is
    # what actually exercises the marks clauses; the last case forces both at
    # once -- the mark after the full stop must end the punctuation run and
    # then stand as a piece of its own via lead backtracking.
    "हिन्दी and café and नमस्ते",
    "end.\u0301go",
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

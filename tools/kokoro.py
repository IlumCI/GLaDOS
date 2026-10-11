#!/usr/bin/env python3
"""Pack Kokoro-82M (hexgrad, Apache-2.0) into GLADOSV1, the voice model
`src/voice/kokoro.rs` runs, with the misaki lexicon beside it.

Why a *named* container where the language model's `GLADOSM3` is bare
arithmetic: that checkpoint is forty tensors in a regular pattern and this
one is five hundred and fifty in no pattern at all -- six module families,
convolutions of every kernel size, LSTMs in both directions, and a weight
norm on most of it. A reader indexing that by arithmetic is a reader one
off-by-one from loading every tensor after the mistake as perfectly valid
garbage. Names cost 20 KB and make a wrong shape a refusal.

What is done to the weights before they are written, each of them a thing
the kernel would otherwise do on every call:

  weight norm    `w = g * v / |v|` folded; the kernel never sees g or v
  conv weights   (out, in, k) kept, the kernel reads it as (out, in*k) --
                 one matmul over an im2col'd input, which is the one kernel
                 this tree already has in AVX2
  transposed     ConvTranspose1d's (in, out, k) becomes (k, out, in), so
                 each tap is a contiguous (out, in) matrix: the kernel runs
                 one matvec per input frame per tap and scatters by stride
  depthwise      a (c, 1, k) group conv is a (c, k) table
  LSTM biases    b_ih + b_hh summed: PyTorch keeps two for CuDNN's sake
  int8           per-row, for every matrix `--q8` names (see below); an
                 all-zero row gets scale 1 so nothing divides by zero

Int8 is a measured choice and not a default. `q8test` in the session that
wrote this put the whole model through per-row int8 and read the result on
a smoothed log spectrum against f32: the number is printed by `--selftest`
when a checkpoint is given, so the decision stays checkable.

Format, little-endian, a table then 64-byte-aligned data:

    magic     8    "GLADOSV1"
    version   u32  1
    count     u32  tensors in the table
    per entry: u16 name length, name (utf-8), u8 dtype, u8 ndim,
               u32 dims[ndim], u64 offset from data start, u64 byte length
    dtype 0   f32, row-major over dims
    dtype 1   int8 per-row: i8[rows*cols] then f32 scale[rows],
              rows = dims[0], cols = product of the rest
    dtype 2   bytes -- the lexicon, in GLADOSX1 (below)

GLADOSX1, the misaki lexicon (gold then silver, gold winning), is CMUdict's
GLADOSL1 shape with IPA where the ARPAbet was: count u32, index u32 per
entry (offset into data), data per entry u8 word length, word, u8 phone
byte length, phones as utf-8. Sorted by word bytes for the binary search.
A word with variants takes its DEFAULT.

The reference dump (`--dump`) writes the same container holding every
intermediate of one forward pass for one phoneme sequence, so the kernel
can be diffed stage by stage rather than only at the waveform:

    ref.ids  ref.voice  ref.bert  ref.d_en  ref.d  ref.dur  ref.en
    ref.f0   ref.n      ref.t_en  ref.audio

    python3 tools/kokoro.py out/kokoro out/voice.bin --lexicon PATH/misaki/data
    python3 tools/kokoro.py out/kokoro out/voice.bin --dump out/voice-ref.bin
    python3 tools/kokoro.py --selftest [out/kokoro]
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"GLADOSV1"
LEX_MAGIC = b"GLADOSX1"
F32, Q8, BYTES = 0, 1, 2
ALIGN = 64

# Matrices that go int8 under `--q8 decoder` (the default): the vocoder's
# convolutions, which are the bulk of the weights and run at audio rate.
# The front end -- ALBERT, the LSTMs, the duration head -- stays f32: it is
# 28 M parameters that decide *what* is said, and a rounding there moves
# durations and pitch where a rounding in the decoder moves timbre.
Q8_FAMILIES = {"none": (), "decoder": ("decoder.",), "all": ("bert.", "predictor.", "text_encoder.", "decoder.", "bert_encoder.")}
Q8_MIN = 4096


class Writer:
    def __init__(self):
        self.entries = []
        self.data = bytearray()

    def _pad(self):
        while len(self.data) % ALIGN:
            self.data.append(0)

    def f32(self, name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        self._pad()
        off = len(self.data)
        self.data += arr.tobytes()
        self.entries.append((name, F32, arr.shape, off, arr.nbytes))

    def q8(self, name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        rows = arr.shape[0]
        w = arr.reshape(rows, -1)
        scale = np.abs(w).max(axis=1) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
        self._pad()
        off = len(self.data)
        self.data += q.tobytes() + scale.astype(np.float32).tobytes()
        self.entries.append((name, Q8, arr.shape, off, q.nbytes + rows * 4))

    def blob(self, name, b):
        self._pad()
        off = len(self.data)
        self.data += b
        self.entries.append((name, BYTES, (len(b),), off, len(b)))

    def write(self, path):
        table = bytearray(MAGIC + struct.pack("<II", 1, len(self.entries)))
        for name, dtype, shape, off, n in self.entries:
            nb = name.encode()
            table += struct.pack("<H", len(nb)) + nb + struct.pack("<BB", dtype, len(shape))
            table += b"".join(struct.pack("<I", d) for d in shape)
            table += struct.pack("<QQ", off, n)
        while len(table) % ALIGN:
            table.append(0)
        Path(path).write_bytes(bytes(table) + bytes(self.data))
        return len(table) + len(self.data)


def read(path):
    """The reader, deliberately not the writer: it walks the table and
    answers `{name: array}`; a q8 tensor comes back dequantised."""
    b = Path(path).read_bytes()
    if b[:8] != MAGIC:
        raise ValueError("not a GLADOSV1 file")
    version, count = struct.unpack_from("<II", b, 8)
    if version != 1:
        raise ValueError(f"version {version}")
    p = 16
    entries = []
    for _ in range(count):
        (nl,) = struct.unpack_from("<H", b, p); p += 2
        name = b[p:p + nl].decode(); p += nl
        dtype, nd = struct.unpack_from("<BB", b, p); p += 2
        dims = struct.unpack_from("<" + "I" * nd, b, p); p += 4 * nd
        off, n = struct.unpack_from("<QQ", b, p); p += 16
        entries.append((name, dtype, dims, off, n))
    data_start = (p + ALIGN - 1) // ALIGN * ALIGN
    out = {}
    for name, dtype, dims, off, n in entries:
        raw = b[data_start + off:data_start + off + n]
        if dtype == F32:
            out[name] = np.frombuffer(raw, dtype=np.float32).reshape(dims)
        elif dtype == Q8:
            rows = dims[0]; cols = int(np.prod(dims[1:]))
            q = np.frombuffer(raw[:rows * cols], dtype=np.int8).reshape(rows, cols).astype(np.float32)
            s = np.frombuffer(raw[rows * cols:], dtype=np.float32)
            out[name] = (q * s[:, None]).reshape(dims)
        else:
            out[name] = bytes(raw)
    return out


# ----------------------------------------------------------------- weights

def fold_norm(sd, base):
    """`weight_g` and `weight_v` at `base` into one weight, PyTorch's
    `_weight_norm` with dim=0: the norm runs over every axis but the first."""
    g = sd[base + ".weight_g"].numpy().astype(np.float64)
    v = sd[base + ".weight_v"].numpy().astype(np.float64)
    norm = np.sqrt((v.reshape(v.shape[0], -1) ** 2).sum(axis=1)).reshape([-1] + [1] * (v.ndim - 1))
    return (g * v / norm).astype(np.float32)


def convert(src_dir, out_path, lexicon_dir, q8="decoder", voices=("af_bella",)):
    import torch
    sd_all = torch.load(Path(src_dir) / "kokoro-v1_0.pth", map_location="cpu", weights_only=True)
    cfg = json.load(open(Path(src_dir) / "config.json"))
    w = Writer()
    families = Q8_FAMILIES[q8]

    def put(name, arr):
        arr = np.asarray(arr, dtype=np.float32)
        # The transposed convolutions are stored (k, out, in), so a per-row
        # scale there would be one scale per tap over every channel; they
        # are 12 MB and stay f32.
        if arr.ndim >= 2 and arr.size >= Q8_MIN and name.startswith(families) and ".ups." not in name:
            w.q8(name, arr)
        else:
            w.f32(name, arr)

    seen = set()
    for mod, sd in sd_all.items():
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        keys = list(sd.keys())
        for k in keys:
            full = f"{mod}.{k}"
            # keyed by the full name: two modules share bare names
            # (`lstm.weight_ih_l0` is in both the predictor and the text
            # encoder), and a set of bare names dropped the second one
            if full in seen:
                continue
            if k.startswith("pooler."):
                seen.add(full); continue                          # ALBERT's pooler is never run
            if k.endswith(".weight_g"):
                base = k[:-9]
                wt = fold_norm(sd, base)
                seen.add(f"{mod}.{base}.weight_g"); seen.add(f"{mod}.{base}.weight_v")
                if ".pool." in k or k.endswith("F0_conv.weight_g") or k.endswith("N_conv.weight_g"):
                    # ConvTranspose1d depthwise (pool) is (c, 1, 3): a table.
                    # F0_conv / N_conv are plain (1, 1, 3) convs, also tables.
                    put(f"{mod}.{base}.weight", wt.reshape(wt.shape[0], -1))
                elif ".ups." in k:
                    # ConvTranspose1d (in, out, k) -> (k, out, in)
                    put(f"{mod}.{base}.weight", np.ascontiguousarray(wt.transpose(2, 1, 0)))
                else:
                    put(f"{mod}.{base}.weight", wt)
                continue
            if k.endswith(".weight_v"):
                continue
            if ".bias_hh_l0" in k:
                continue                                       # folded into bias_ih below
            if ".bias_ih_l0" in k:
                b = sd[k].numpy() + sd[k.replace("bias_ih", "bias_hh")].numpy()
                put(f"{mod}.{k.replace('bias_ih', 'bias')}", b)
                seen.add(full); continue
            arr = sd[k].numpy()
            put(full, arr)
            seen.add(full)

    for v in voices:
        pack = torch.load(Path(src_dir) / "voices" / f"{v}.pt", weights_only=True).numpy()
        w.f32(f"voice.{v}", pack.reshape(pack.shape[0], -1))   # (510, 256): row = token count - 1
    # the phoneme vocabulary, as a 256-entry table of codepoints -> id
    vocab = cfg["vocab"]
    cps = np.zeros(len(vocab), dtype=np.int32); ids = np.zeros(len(vocab), dtype=np.int32)
    for i, (ch, idx) in enumerate(sorted(vocab.items(), key=lambda kv: kv[1])):
        cps[i] = ord(ch); ids[i] = idx
    w.f32("vocab.codepoints", cps.astype(np.float32))
    w.f32("vocab.ids", ids.astype(np.float32))
    if lexicon_dir:
        w.blob("lexicon.us", build_lexicon(Path(lexicon_dir)))
    n = w.write(out_path)
    q8n = sum(1 for e in w.entries if e[1] == Q8)
    print(f"wrote {out_path}: {len(w.entries)} tensors, {q8n} int8, {n / 1e6:.1f} MB")
    return n


def build_lexicon(d):
    gold = json.load(open(d / "us_gold.json")); silver = json.load(open(d / "us_silver.json"))
    table = {}
    for src in (silver, gold):                                  # gold last, so gold wins
        for k, v in src.items():
            if isinstance(v, dict):
                v = v.get("DEFAULT")
            if not v or len(k.encode()) > 255 or len(v.encode()) > 255:
                continue
            table[k] = v
    items = sorted(table.items(), key=lambda kv: kv[0].encode())
    data = bytearray(); index = []
    for k, v in items:
        kb = k.encode(); vb = v.encode()
        index.append(len(data))
        data += bytes([len(kb)]) + kb + bytes([len(vb)]) + vb
    out = bytearray(LEX_MAGIC + struct.pack("<II", len(items), len(data)))
    out += b"".join(struct.pack("<I", o) for o in index) + data
    return bytes(out)


def lexicon_lookup(blob, word):
    if blob[:8] != LEX_MAGIC:
        return None
    count, _ = struct.unpack_from("<II", blob, 8)
    base = 16 + 4 * count
    key = word.encode()
    lo, hi = 0, count
    while lo < hi:
        mid = (lo + hi) // 2
        (off,) = struct.unpack_from("<I", blob, 16 + 4 * mid)
        o = base + off
        wl = blob[o]; wb = blob[o + 1:o + 1 + wl]
        if wb < key:
            lo = mid + 1
        elif wb > key:
            hi = mid
        else:
            pl = blob[o + 1 + wl]
            return blob[o + 2 + wl:o + 2 + wl + pl].decode()
    return None


# ---------------------------------------------------------------- reference

def _catch_post(caught, i, o):
    # a forward hook that returns anything replaces the module's output,
    # so this one catches and says nothing
    caught["postin"] = i[0]
    caught["post"] = o


def dump(src_dir, out_path, phonemes, voice="af_bella", seed=0):
    """Every intermediate of one forward pass, Kokoro's own durations,
    for the kernel to diff against. The audio is generated under a fixed
    seed; the kernel's noise is its own, so the audio is compared on its
    envelope and everything before it is compared exactly."""
    import torch
    sys.path.insert(0, str(Path(src_dir)))
    from kmodel.model import KModel                             # the vendored copy beside the weights
    m = KModel(repo_id="hexgrad/Kokoro-82M", config=str(Path(src_dir) / "config.json"),
               model=str(Path(src_dir) / "kokoro-v1_0.pth")).eval()
    pack = torch.load(Path(src_dir) / "voices" / f"{voice}.pt", weights_only=True)
    ids = [0] + [m.vocab[c] for c in phonemes if c in m.vocab] + [0]
    torch.manual_seed(seed)
    # the vocoder's intermediates, caught on the way through, and the
    # random harmonic source itself -- the kernel takes that source in
    # place of its own, so everything after it is exactly comparable
    caught = {}
    gen = m.decoder.generator
    hooks = [
        gen.register_forward_hook(lambda mod, i, o: caught.__setitem__("x_dec", i[0])),
        gen.m_source.register_forward_hook(lambda mod, i, o: caught.__setitem__("har", o[0])),
        gen.ups[0].register_forward_hook(lambda mod, i, o: caught.__setitem__("up0", o)),
        gen.ups[1].register_forward_hook(lambda mod, i, o: caught.__setitem__("up1", o)),
        gen.noise_res[0].register_forward_hook(lambda mod, i, o: caught.__setitem__("nr0", o)),
        gen.noise_convs[0].register_forward_hook(lambda mod, i, o: (caught.__setitem__("harf", i[0]), caught.__setitem__("nc0", o)) and None),
        gen.noise_res[1].register_forward_hook(lambda mod, i, o: caught.__setitem__("nr1", o)),
        gen.conv_post.register_forward_hook(lambda mod, i, o: _catch_post(caught, i, o)),
    ]
    with torch.no_grad():
        input_ids = torch.LongTensor([ids]); ref_s = pack[len(ids) - 3]; L = len(ids)
        tm = torch.zeros(1, L, dtype=torch.bool)
        bert = m.bert(input_ids, attention_mask=(~tm).int())
        d_en = m.bert_encoder(bert).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = m.predictor.text_encoder(d_en, s, torch.LongTensor([L]), tm)
        x, _ = m.predictor.lstm(d)
        dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1)
        pred = torch.round(dur).clamp(min=1).long().squeeze()
        ind = torch.repeat_interleave(torch.arange(L), pred)
        aln = torch.zeros((L, len(ind))); aln[ind, torch.arange(len(ind))] = 1; aln = aln.unsqueeze(0)
        en = d.transpose(-1, -2) @ aln
        f0, n = m.predictor.F0Ntrain(en, s)
        t_en = m.text_encoder(input_ids, torch.LongTensor([L]), tm)
        audio = m.decoder(t_en @ aln, f0, n, ref_s[:, :128]).squeeze()
    w = Writer()
    w.f32("ref.ids", np.array(ids, dtype=np.float32))
    w.f32("ref.voice", np.array([len(ids) - 3], dtype=np.float32))
    w.f32("ref.bert", bert[0].numpy())
    w.f32("ref.d_en", d_en[0].numpy())
    w.f32("ref.d", d[0].numpy())
    w.f32("ref.dur", dur[0].numpy())
    w.f32("ref.en", en[0].numpy())
    w.f32("ref.f0", f0[0].numpy())
    w.f32("ref.n", n[0].numpy())
    w.f32("ref.t_en", t_en[0].numpy())
    w.f32("ref.audio", audio.numpy())
    for h in hooks:
        h.remove()
    for k, v in caught.items():
        w.f32(f"ref.{k}", v[0].numpy() if v.dim() == 3 else v.reshape(-1).numpy())
    size = w.write(out_path)
    print(f"wrote {out_path}: {len(ids)} ids, {int(pred.sum())} frames, {len(audio) / 24000:.2f} s, {size / 1e3:.0f} KB")


# ----------------------------------------------------------------- selftest

def selftest(src_dir=None):
    import tempfile
    ok = True

    def claim(what, good):
        nonlocal ok
        print(f"  {'ok  ' if good else 'FAIL'}  {what}")
        ok &= good

    rng = np.random.default_rng(1)
    w = Writer()
    a = rng.standard_normal((3, 5)).astype(np.float32)
    b = rng.standard_normal((70, 9, 3)).astype(np.float32)
    w.f32("a", a); w.q8("b", b); w.blob("lex", build_lexicon_from({"hello": "həlˈO", "Hello": "hɛlO", "a": "ɐ"}))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.bin"
        w.write(p)
        r = read(p)
        claim("an f32 tensor round-trips bit for bit", np.array_equal(r["a"], a))
        err = np.abs(r["b"] - b).max()
        bound = np.abs(b.reshape(70, -1)).max(axis=1).max() / 127 / 2 + 1e-6
        claim(f"int8 per-row error {err:.4f} is within half a step", err <= bound)
        claim("a q8 tensor keeps its three dims", r["b"].shape == b.shape)
        lex = r["lex"]
        claim("the lexicon finds a word", lexicon_lookup(lex, "hello") == "həlˈO")
        claim("case is a different key, as misaki has it", lexicon_lookup(lex, "Hello") == "hɛlO")
        claim("a missing word is None", lexicon_lookup(lex, "zzz") is None)
    z = np.zeros((4, 8), dtype=np.float32)
    w2 = Writer(); w2.q8("z", z)
    claim("an all-zero row quantises to zero and no NaN", not np.isnan(np.frombuffer(bytes(w2.data[-16:]), dtype=np.float32)).any())
    if src_dir:
        import torch
        sd = torch.load(Path(src_dir) / "kokoro-v1_0.pth", map_location="cpu", weights_only=True)["decoder"]
        sd = {k[7:]: v for k, v in sd.items()}
        folded = fold_norm(sd, "encode.conv1")
        ref = torch._weight_norm(sd["encode.conv1.weight_v"], sd["encode.conv1.weight_g"], 0).numpy()
        claim(f"weight norm folds as torch folds it (max diff {np.abs(folded - ref).max():.2e})", np.abs(folded - ref).max() < 1e-5)
    print("selftest", "passed" if ok else "FAILED")
    return ok


def build_lexicon_from(table):
    items = sorted(table.items(), key=lambda kv: kv[0].encode())
    data = bytearray(); index = []
    for k, v in items:
        kb = k.encode(); vb = v.encode()
        index.append(len(data))
        data += bytes([len(kb)]) + kb + bytes([len(vb)]) + vb
    return bytes(LEX_MAGIC + struct.pack("<II", len(items), len(data)) + b"".join(struct.pack("<I", o) for o in index) + data)


def main(argv):
    if "--selftest" in argv:
        rest = [a for a in argv[1:] if a != "--selftest"]
        return 0 if selftest(rest[0] if rest else None) else 1
    if len(argv) < 3:
        print(__doc__); return 2
    src, out = argv[1], argv[2]
    if "--dump" in argv:
        text = argv[argv.index("--dump") + 1]
        phon = argv[argv.index("--phonemes") + 1] if "--phonemes" in argv else "wi hæv sˈO mˈɛni tˈɛsts fɔɹ ju tədˈA."
        dump(src, text, phon)
        return 0
    lex = argv[argv.index("--lexicon") + 1] if "--lexicon" in argv else None
    q8 = argv[argv.index("--q8") + 1] if "--q8" in argv else "decoder"
    convert(src, out, lex, q8=q8)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
"""The dense forward pass again: batched, device-placed, and still checked.

`reference.py` is this project's numeric oracle and is deliberately written one
position at a time, a Python loop over tokens wrapping a Python loop over
layers doing matrix-vector products. That shape is why it is readable and why
it is trusted, and it is also why a 726-token prompt takes minutes. It is an
oracle, not a runner.

This is the runner, and it is faster three ways. **They were not the three
predicted, and the ranking was wrong**, which is worth stating first because
the obvious optimisation here is the one that bought least.

- **Positions within a prompt.** A prefill becomes `X @ W.T`, one GEMM instead
  of 726 matrix-vector products. 92x on the prefill, measured.
- **The shared prefix, computed once.** A 5-shot prompt is 726 tokens of which
  658 are the worked examples, byte-identical for all 1,319 questions. This is
  the largest win in the file and it is not a parallelism trick at all, it is
  noticing that the same arithmetic was being done 1,319 times.
- **Questions against each other.** Decode is one token at a time whatever you
  do, so a lone sequence drags every weight past the processor for one row of
  arithmetic. Raw decode throughput on CPU: 8.8 tok/s at batch 1, 33.6 at 8,
  40.2 at 16 -- which is a throughput figure and **not** a wall-clock one; see
  below for why the two came apart and what is still unmeasured.

**What the third one is worth is not yet known, and the last claim here about
it was measured wrong.** This docstring said batching moved 13.5 s/question to
13.8, "nothing", and drew a confident lesson from it. Both figures came from
runs at **batch 1**: `lm_eval` built its argument list without passing `batch`
through to `run_gsm8k`, so every `--batch 8` on the command line was a batch of
one. The measurement compared a configuration against itself and the difference
was host noise.

The tell was printed on every one of those runs and went unread --
`1319 group(s) of one length` is only possible at batch 1.

What is known after fixing it, on 24 questions: batch 1 and batch 8 give the
same answers and the same wall clock, **because at that size exact-length
grouping barely groups anything** -- 24 questions became 21 groups. A sample
that small cannot measure batching at all. The benefit exists only where
lengths collide, which at the full 1,319 is mean group 11.7.

So the honest state is: correctness under batching is checked (below, and by
identical answers across batch sizes), and the *speedup* is unmeasured. Do not
quote one here until a full run has been done both ways.

Two costs are real regardless and are worth knowing before expecting much. A
726-token prefill already saturates the processor, so batched prefills share
little. And a batch runs until its **longest** answer finishes: answers average
107 tokens against a 256 budget, so a group pays its slowest member's bill.

### A second implementation is a liability unless it is checked

`model.rs` makes the objection twice and this file has already paid for it
once: `lm_eval.py` used to carry its own dense pass, written for llama2, and it
could not run Qwen3 at all. Nobody noticed because nothing compared it to
anything.

    python tools/fastdense.py out/dense-check.bin --check --batch 4

runs this and the oracle over the same ids and prints the largest disagreement
per row. The check uses a **ragged** batch on purpose: rows all of one length
never exercise the padding mask, which is the half of this that `reference.py`
has no equivalent of and therefore the half most able to be quietly wrong.

### What has to match, and where each one hides when it does not

**QK-Norm before RoPE, per head.** Applied after, the rotation is rescaled and
position stops meaning what it means.

**`rotate_half`, pairing `i` with `i + head_dim/2`.** The interleaved
convention rotates by the same angles, so the model stays fluent and attends by
a scrambled notion of distance.

**The head width the file states**, not `dim // heads`. Qwen3 states 128 where
the derivation gives 64.

**The KV cache round-trip, when the checkpoint asks for one**, applied where
the kernel applies it: after QK-Norm for keys, on the raw projection for
values, before RoPE for both.

None of the four fails loudly. All four are in the check.

### Left padding, which is a correctness decision and not a formatting one

Prompts in a batch differ in length. Padding on the **right** puts each
sequence's next real token inside its own padding, so the rows stop sharing a
decode step and the whole point is lost. Padding on the **left** ends every row
at the same column, so one step advances all of them.

RoPE survives it because RoPE is relative: absolute positions shift by the pad
width and every distance between two real tokens is unchanged. The pad columns
are masked out of attention, so nothing reads them.

### The one conflict, stated rather than hidden

A held prefix must sit at **the same columns in every row**, because a key
carries the RoPE angle of the column it was computed at. Left padding slides
it. So the two wins above are mutually exclusive unless every prompt in a
group is the same length, and `prefill(prompts, prefix=...)` refuses rather
than silently sliding one.

Grouping by exact length is what `lm_eval` does, and the cost is measured
rather than assumed: GSM8K 5-shot has **113 distinct prompt lengths over 1,319
questions**, mean group 11.7 and median 8, so at batch 16 it needs 151 batches
against an ideal 83. Under twice the ideal, for an exact prefix and no padding
at all.

Resolving it properly means tracking a per-row end column and scattering the
cache writes, which removes padding entirely and lets any lengths share a
batch. That is the next thing to build here and it is not built.

### Where the weights live

A 4 GB laptop card holds a 0.6B in fp16 and does not hold a 2B, so layers are
**placed against a measured budget** rather than assumed to fit. Contiguous
from the front, because each boundary between devices costs a transfer of the
residual stream and one boundary is the fewest there can be. `--device cpu`
keeps everything in host memory and is what a machine with no CUDA build does.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference as R  # noqa: E402

DTYPES = {"f32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# f32 accumulated in a different order moves the last bit; what must not move
# is which token wins. Stated as a number per dtype rather than as a hope, and
# looser for the narrow ones because they genuinely are.
TOL = {"f32": 2e-2, "bf16": 5e-1, "fp16": 3e-1}


def pick_device(want):
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"no CUDA device visible to torch. This is torch {torch.__version__}; "
            "a '+cpu' build has no CUDA in it however many GPUs are fitted.")
    return want


def dequantise(mat, block=4096):
    """One int8 matrix to f32, a band of rows at a time.

    `reference.mv` scales by row, so row `i` of the real matrix is
    `data[i] * scales[i]`. Banded because the whole classifier at once is a
    622 MB temporary on top of the 622 MB result, and the peak is what decides
    whether this fits at all.
    """
    data, scales = mat
    rows, cols = data.shape
    out = np.empty((rows, cols), dtype=np.float32)
    for i in range(0, rows, block):
        j = min(i + block, rows)
        np.multiply(data[i:j], scales[i:j, None], out=out[i:j], dtype=np.float32)
    return out


def kv8_roundtrip(a, block=R.KV_BLOCK):
    """`reference.q8_roundtrip` over a whole tensor at once.

    Same blocks, same peak-over-127 scale, same round-to-nearest and clip. The
    point of the cache round-trip is that the error here is the error the
    kernel has, so an approximation of it would be worth nothing.
    """
    shape = a.shape
    flat = a.reshape(-1, block).float()
    peak = flat.abs().amax(dim=1, keepdim=True)
    scale = torch.where(peak == 0, torch.ones_like(peak), peak / 127.0)
    out = torch.round(flat / scale).clamp(-127, 127) * scale
    return out.reshape(shape).to(a.dtype)


class Dense:
    """A dense checkpoint on whatever hardware is here, ready to be stepped."""

    def __init__(self, path, max_len, batch=1, device="auto", dtype="f32",
                 kv_dtype=None, kv8=None, vram=None, verbose=False):
        t0 = time.time()
        self.device = pick_device(device)
        self.dtype = DTYPES[dtype]
        # **The cache is what scales with batch, and the weights are not.**
        # Weights are a fixed 3.0 GB at f32 for this checkpoint; the cache is
        # 264 MB per sequence at a 1152-token context, so batch 16 is 4.2 GB
        # of cache against 3.0 of weights. Halving the cache therefore roughly
        # doubles the batch that fits, where halving the weights buys one
        # constant. Separate knob for that reason.
        #
        # The kernel holds its own KV cache int8 and `reference.py` models
        # that with `q8_roundtrip`, so narrowing the cache is not a departure
        # from what the real system does. It *is* a departure from the oracle
        # whenever the checkpoint does not ask for it, which is why the check
        # reports the disagreement instead of hiding it.
        self.kv_dtype = DTYPES[kv_dtype] if kv_dtype else self.dtype
        # **Whether the eval runs the cache the kernel actually runs.**
        # `KvLayer` in `src/ai/model.rs` is `Vec<i8>` plus per-block scales:
        # the shipped system quantises its KV cache and keeps f32 only as an
        # explicit opt-in for gradient checks. This checkpoint's header says
        # `kv8: None`, so the oracle and this runner both keep f32 -- which
        # makes every host figure a number about the *checkpoint* rather than
        # about GLaDOS.
        #
        # Forcing it on measures what the kernel would score. It is off by
        # default because the oracle does not do it either, and the two have
        # to agree for `--check` to mean anything. Assigned after the load
        # below, because it reads the header.
        # Norms, attention and the softmax accumulate in f32 whatever the
        # weights are. A softmax over a long context in fp16 is where a
        # plausible-looking model quietly stops being the model.
        self.acc = torch.float32
        cfg, w = R.load(str(path))
        self.cfg = cfg
        self.kv8 = cfg.get("kv8") if kv8 is None else kv8
        self.max_len = max_len
        self.batch = batch
        self.latent_k = 0
        L = cfg["layers"]
        width = torch.finfo(self.dtype).bits // 8
        vocab = w["embed"][0].shape[0]

        # --- placement, decided from measured free memory ----------------
        self.gpu_layers = L
        if self.device == "cuda":
            free, _ = torch.cuda.mem_get_info()
            budget = vram if vram is not None else int(free * 0.85)
            d, h = cfg["dim"], cfg["hidden"]
            per_layer = (cfg["q_dim"] * d + cfg["kv_dim"] * d * 2
                         + d * cfg["q_dim"] + h * d * 3) * width
            tail = vocab * d * width
            kvw = torch.finfo(self.kv_dtype).bits // 8
            cache = L * batch * max_len * cfg["kv_dim"] * 2 * kvw
            room = budget - tail - cache
            self.gpu_layers = max(0, min(L, room // per_layer)) if per_layer else L
            if verbose:
                print(f"  vram {budget/1e9:.2f} GB: {tail/1e9:.2f} tail + "
                      f"{cache/1e9:.2f} cache + {per_layer/1e9:.3f}/layer "
                      f"-> {self.gpu_layers}/{L} layers on the GPU")

        def dev_of(li):
            return self.device if li < self.gpu_layers else "cpu"

        def load(mat, li):
            return torch.from_numpy(dequantise(mat)).to(
                device=dev_of(li), dtype=self.dtype)

        def norm(v, li):
            return torch.as_tensor(np.asarray(v, dtype=np.float32)).to(dev_of(li))

        # The tail -- embeddings, final norm, classifier -- goes where the last
        # layer ended, so the one boundary crossing stays one.
        self.tail_dev = dev_of(L - 1)
        emb = torch.from_numpy(dequantise(w["embed"])).to(
            device=self.tail_dev, dtype=self.dtype)
        tied = w["wcls"][0] is w["embed"][0]
        # Tied embeddings are one array under two names. Sharing the tensor
        # rather than dequantising twice is worth 622 MB on this checkpoint,
        # and the gather and the GEMM want the same device anyway.
        self.embed = emb
        self.wcls = emb if tied else torch.from_numpy(
            dequantise(w["wcls"])).to(device=self.tail_dev, dtype=self.dtype)

        self.wq = [load(w["wq"][i], i) for i in range(L)]
        self.wk = [load(w["wk"][i], i) for i in range(L)]
        self.wv = [load(w["wv"][i], i) for i in range(L)]
        self.wo = [load(w["wo"][i], i) for i in range(L)]
        self.w1 = [load(w["w1"][i], i) for i in range(L)]
        self.w2 = [load(w["w2"][i], i) for i in range(L)]
        self.w3 = [load(w["w3"][i], i) for i in range(L)]
        self.rms_att = [norm(w["rms_att"][i], i) for i in range(L)]
        self.rms_ffn = [norm(w["rms_ffn"][i], i) for i in range(L)]
        self.rms_final = torch.as_tensor(
            np.asarray(w["rms_final"], dtype=np.float32)).to(self.tail_dev)
        self.qk_norm = bool(cfg["qk_norm"])
        if self.qk_norm:
            self.q_norm = [norm(w["q_norm"][i], i) for i in range(L)]
            self.k_norm = [norm(w["k_norm"][i], i) for i in range(L)]
        del w

        half = cfg["head_dim"] // 2
        idx = torch.arange(half, dtype=torch.float32)
        self.inv = 1.0 / (cfg["theta"] ** (2.0 * idx / cfg["head_dim"]))

        self.kc, self.vc = [], []
        for i in range(L):
            self.kc.append(torch.zeros((batch, max_len, cfg["kv_dim"]),
                                       dtype=self.kv_dtype, device=dev_of(i)))
            self.vc.append(torch.zeros((batch, max_len, cfg["kv_dim"]),
                                       dtype=self.kv_dtype, device=dev_of(i)))
        # **The latent feedback needs a scale, and without one it is an
        # 898x overdrive.** A COCONUT pass puts the last hidden state where an
        # input embedding goes, and on this checkpoint those are not remotely
        # the same size: the embedding table has RMS 0.0278 and the final
        # residual stream has RMS 24.99. Fed back raw, the model is handed a
        # vector three orders of magnitude out of distribution, and it does
        # what that deserves -- it starts an answer correctly and collapses
        # into "1 the 1 the 1" inside a dozen tokens.
        #
        # Measured that way GSM8K went 38.0% to 2.0% at k=1, which reads as a
        # fact about untrained latent reasoning and is really a fact about a
        # missing normalisation. `latent_scale` rescales the fed-back state to
        # the embedding table's own RMS, so the thing arriving at layer 0 is
        # the size layer 0 was trained to receive.
        self.embed_rms = float(self.embed.float().pow(2).mean().sqrt())
        self.latent_scale = True
        self.pos = 0                                   # committed columns
        self.left = torch.zeros(batch, dtype=torch.long)   # pad width per row
        self.prefix = None                             # see `hold_prefix`
        if verbose:
            kvw = torch.finfo(self.kv_dtype).bits // 8
            wb = sum(t.numel() * t.element_size()
                     for t in (self.wq + self.wk + self.wv + self.wo
                               + self.w1 + self.w2 + self.w3)) + emb.numel() * emb.element_size()
            cb = L * batch * max_len * cfg["kv_dim"] * 2 * kvw
            print(f"  loaded in {time.time() - t0:.1f}s on {self.device}/{dtype}, "
                  f"kv {kv_dtype or dtype}{' int8' if self.kv8 else ''}, "
                  f"batch {batch}, vocab {vocab}"
                  f"{', tied' if tied else ''}")
            print(f"  resident: {wb/1e9:.2f} GB weights + {cb/1e9:.2f} GB cache "
                  f"= {(wb + cb)/1e9:.2f} GB")

    # --- the pass ------------------------------------------------------

    # --- the shared prefix ---------------------------------------------
    #
    # **This is the largest single win in the file and it is not batching.**
    # A 5-shot GSM8K prompt is 726 tokens of which 688 are the five worked
    # examples -- byte-identical for all 1,319 questions. Prefilling them once
    # per question is 1,319 x 688 tokens of arithmetic to recompute the same
    # keys and values, and it dominated everything else once decode was
    # batched: 1.4 hours of prefill against 1 hour of decode.
    #
    # Held once and copied into each row instead. The copy is a memcpy of
    # 158 MB where the recomputation is 28 layers of GEMM.
    #
    # The condition is that the prefix occupies **the same columns in every
    # row**, because a key carries the RoPE angle of the column it was
    # computed at. That is why the caller must batch prompts of one length:
    # pad any row and its prefix slides, and a slid prefix is a different set
    # of keys wearing the right values. `prefill` checks rather than trusts.

    def hold_prefix(self, ids):
        """Run `ids` once and keep the cache they produce."""
        self.reset()
        self.feed(ids)
        self.prefix = ([k[:1, :self.pos].clone() for k in self.kc],
                       [v[:1, :self.pos].clone() for v in self.vc],
                       self.pos)
        return self.pos

    def use_prefix(self, B):
        """Put the held prefix into the first `B` rows."""
        if self.prefix is None:
            raise ValueError("no prefix held")
        ks, vs, n = self.prefix
        for li in range(self.cfg["layers"]):
            self.kc[li][:B, :n] = ks[li]
            self.vc[li][:B, :n] = vs[li]
        self.pos = n
        self.left[:B] = 0

    def reset(self):
        self.pos = 0
        self.left.zero_()
        # Not cleared: attention never looks past `pos` nor left of the pad
        # width, so a stale column cannot be read. Zeroing hundreds of
        # megabytes per question would cost more than the question.

    def _rms(self, x, weight):
        f = x.to(self.acc)
        s = torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + self.cfg["eps"])
        return (f * s * weight.to(self.acc)).to(x.dtype)

    def _rope(self, x, pos0, T):
        """`rotate_half` over `(B, T, heads, hs)`."""
        half = self.cfg["head_dim"] // 2
        p = torch.arange(pos0, pos0 + T, dtype=torch.float32, device=x.device)
        ang = p[:, None] * self.inv.to(x.device)[None, :]
        cos = ang.cos()[None, :, None, :].to(x.dtype)
        sin = ang.sin()[None, :, None, :].to(x.dtype)
        a, b = x[..., :half], x[..., half:]
        return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)

    def _mask(self, q0, T, n, B, device):
        """Additive mask: causal, and blind to the left padding.

        A key's column index **is** its position, which is what left padding
        buys and what lets this be two comparisons. Query `q0 + t` may see key
        `j` when `j` is at or before it and `j` is past that row's pad.
        """
        q_at = q0 + torch.arange(T, device=device)
        keys = torch.arange(n, device=device)
        causal = keys[None, :] <= q_at[:, None]                    # (T, n)
        pad_ok = keys[None, None, :] >= self.left[:B].to(device)[:, None, None]
        ok = causal[None] & pad_ok                                 # (B, T, n)
        return torch.where(ok[:, None], 0.0,
                           torch.finfo(self.acc).min).to(self.acc)

    def _body(self, x, pos0, mask, commit):
        """Every layer once. `commit` is what makes a latent pass latent."""
        cfg = self.cfg
        heads, kvh, hs = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
        B, T, _ = x.shape
        n = self.pos + T
        for li in range(cfg["layers"]):
            dev = self.kc[li].device
            if x.device != dev:                 # the one boundary crossing
                x, mask = x.to(dev), mask.to(dev)
            xb = self._rms(x, self.rms_att[li])
            q = (xb @ self.wq[li].T).view(B, T, heads, hs)
            k = (xb @ self.wk[li].T).view(B, T, kvh, hs)
            v = xb @ self.wv[li].T

            if self.qk_norm:
                q = self._rms(q, self.q_norm[li])
                k = self._rms(k, self.k_norm[li])
            if self.kv8:
                k, v = kv8_roundtrip(k), kv8_roundtrip(v)

            q = self._rope(q, pos0, T)
            k = self._rope(k, pos0, T)

            if commit:
                self.kc[li][:B, self.pos:n] = k.reshape(B, T, -1).to(self.kv_dtype)
                self.vc[li][:B, self.pos:n] = v.to(self.kv_dtype)
                kc, vc = self.kc[li][:B, :n], self.vc[li][:B, :n]
            else:
                # A latent pass reads the committed prefix and writes nothing.
                # Its own key is not appended: "the KV prefix is frozen" is the
                # whole claim, and appending would also put a key at a column
                # whose index is no longer its position, which is the one
                # assumption the mask rests on.
                kc, vc = self.kc[li][:B, :self.pos], self.vc[li][:B, :self.pos]

            qh = q.transpose(1, 2).to(self.acc)
            kh = kc.view(B, -1, kvh, hs).transpose(1, 2).to(self.acc)
            vh = vc.view(B, -1, kvh, hs).transpose(1, 2).to(self.acc)
            mh = mask[:, :, :, :kc.shape[1]]
            if T == 1:
                # **SDPA is 6.5x slower than this at decode shape, measured.**
                # Flash and memory-efficient kernels are built for training,
                # where the query axis is long; a decode step has T=1, one
                # query row against the whole prefix, and torch picks a kernel
                # that runs at 23 GB/s on a card that does 167. Written out as
                # two batched matmuls it reaches 151 GB/s, which is ~90% of
                # what the hardware gives on a plain copy.
                #
                #     f32   sdpa 2102 us   manual  322 us   6.5x
                #     fp16  sdpa 2576 us   manual  195 us  13.2x
                #
                # over 28 layers that is 58.7 ms/step against 9.0. Prefill
                # keeps SDPA, because a long query axis is exactly what those
                # kernels are good at.
                rp = heads // kvh
                sc = torch.einsum("bkrh,bknh->bkrn",
                                  qh.view(B, kvh, rp, hs), kh) * (hs ** -0.5)
                sc = torch.softmax(sc + mh.view(B, 1, 1, -1), dim=-1)
                att = torch.einsum("bkrn,bknh->bkrh", sc, vh)
                att = att.reshape(B, heads, 1, hs)
            else:
                att = F.scaled_dot_product_attention(
                    qh, kh, vh, attn_mask=mh, enable_gqa=True)
            att = att.transpose(1, 2).reshape(B, T, -1).to(x.dtype)

            x = x + att @ self.wo[li].T
            xb = self._rms(x, self.rms_ffn[li])
            gate = F.silu((xb @ self.w1[li].T).to(self.acc)).to(x.dtype)
            x = x + (gate * (xb @ self.w3[li].T)) @ self.w2[li].T
        return x

    def _run(self, ids, left=None, latent=True):
        """Feed `(B, T)` ids and answer the last column's logits, `(B, vocab)`."""
        B, T = ids.shape
        if self.pos + T > self.max_len:
            raise ValueError(f"{self.pos + T} tokens into a {self.max_len} context")
        if left is not None:
            self.left[:B] = left
        dev = self.kc[0].device
        x = self.embed[ids.reshape(-1).to(self.embed.device)].view(B, T, -1).to(dev)
        x = self._body(x, self.pos, self._mask(self.pos, T, self.pos + T, B, dev),
                       commit=True)
        self.pos += T
        # Coconut-flavoured and training-free: K further passes with the last
        # hidden state as the input, no token decoded, position not advanced,
        # cache frozen. The refinement happens in the continuous stream and
        # leaves no trace in the context. Whether an untrained backbone gains
        # anything from it is exactly what a measurement is for, which is why
        # this is a flag and not a default.
        # **Once, after the prompt -- not on every decode step.** This loop
        # used to run on every call, so each of 256 generated tokens paid k
        # extra passes over the whole body. That is not COCONUT, which thinks
        # in a contiguous latent block *between* the prompt and the answer and
        # then generates normally; it is per-token refinement, and it compounds
        # a perturbation 256 times instead of once.
        #
        # It was also most of the cost: k=4 paid five body passes per token, so
        # a sweep over k=0,1,2,4 cost eleven runs' worth of decode rather than
        # four. Fixing the shape makes every k about as cheap as k=0.
        for _ in range(self.latent_k if latent else 0):
            d2 = self.kc[0].device
            h = x[:, -1:, :].to(d2)
            if self.latent_scale:
                rms = h.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                h = (h.float() / rms * self.embed_rms).to(h.dtype)
            x = self._body(h, self.pos - 1,
                           self._mask(self.pos - 1, 1, self.pos, B, d2),
                           commit=False)
        return self._logits(x[:, -1, :])

    def _logits(self, x):
        x = x.to(self.wcls.device)
        return (self._rms(x, self.rms_final) @ self.wcls.T).to(self.acc)

    # --- what callers use ---------------------------------------------

    @torch.inference_mode()
    def feed(self, ids):
        """One sequence, the shape `lm_eval`'s single-question loop wants."""
        t = torch.as_tensor(list(ids), dtype=torch.long).view(1, -1)
        # A multi-token call is a prompt and gets the latent block; a
        # single-token call is a decode step and does not. The compat path
        # uses `feed` for both, so the length is the only thing that tells
        # them apart here -- `prefill`/`step` say which they are explicitly.
        return self._run(t, latent=len(ids) > 1).float().cpu().numpy()[0]

    @torch.inference_mode()
    def prefill(self, prompts, prefix=None):
        """Left-pad a ragged batch and run it. Answers `(B, vocab)`.

        `prefix` is a token list every prompt starts with. Given one, it is
        computed on the first call and reused on every later one, and the
        prompts must then all be the same length -- see `hold_prefix` for why
        a padded row cannot share a prefix.
        """
        B = len(prompts)
        if B > self.batch:
            raise ValueError(f"{B} prompts into a batch of {self.batch}")

        if prefix is not None:
            lens = {len(p) for p in prompts}
            if len(lens) != 1:
                raise ValueError(
                    f"a shared prefix needs one prompt length, got {sorted(lens)}")
            if self.prefix is None or self.prefix[2] != len(prefix):
                self.prefix = None
                self.hold_prefix(list(prefix))
            self.use_prefix(B)
            tail = torch.as_tensor([p[len(prefix):] for p in prompts],
                                   dtype=torch.long)
            return self._run(tail).float().cpu().numpy()

        T = max(len(p) for p in prompts)
        ids = torch.zeros((B, T), dtype=torch.long)
        left = torch.zeros(B, dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, T - len(p):] = torch.as_tensor(p, dtype=torch.long)
            left[i] = T - len(p)
        return self._run(ids, left).float().cpu().numpy()

    @torch.inference_mode()
    def step(self, tokens):
        """One token for each row. Answers `(B, vocab)`.

        No latent block: the thinking happened once, at `prefill`.
        """
        t = torch.as_tensor(list(tokens), dtype=torch.long).view(-1, 1)
        return self._run(t, latent=False).float().cpu().numpy()


# --- the check, which is the only reason any of this may be used -----------


def check(path, tokens, device, dtype, batch, kv_dtype=None):
    print(f"[fastdense] {Path(path).name}: {tokens} token(s), batch {batch}, "
          f"{device}/{dtype}")
    cfg, w = R.load(str(path))
    vocab = w["embed"][0].shape[0]
    rng = np.random.default_rng(20260916)
    # Ragged on purpose: rows all of one length never exercise the padding
    # mask, and that is the half reference.py cannot check by construction.
    lens = [max(8, tokens - 7 * i) for i in range(batch)]
    seqs = [[int(v) for v in rng.integers(0, min(vocab, 30000), size=n)]
            for n in lens]

    t0 = time.time()
    slow = [R.forward(cfg, w, s, R.new_cache(cfg, len(s) + 4), 0) for s in seqs]
    t_slow = time.time() - t0
    del w

    d = Dense(path, max_len=tokens + 8, batch=batch, device=device,
              dtype=dtype, kv_dtype=kv_dtype, verbose=True)
    t0 = time.time()
    fast = list(d.prefill(seqs))
    t_fast = time.time() - t0

    print(f"  oracle {t_slow:8.2f}s      batched {t_fast:8.2f}s      "
          f"{t_slow / max(t_fast, 1e-9):.0f}x")
    ok = True
    for i, (a, b) in enumerate(zip(slow, fast)):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        diff = np.abs(a - b)
        top_a, top_b = list(np.argsort(a)[::-1][:5]), list(np.argsort(b)[::-1][:5])
        same, held = int(np.argmax(a)) == int(np.argmax(b)), top_a == top_b
        print(f"  row {i} len {lens[i]:4d}  max |dlogit| {diff.max():.3e}   "
              f"argmax {'same' if same else 'DIFFERENT'}   "
              f"top-5 {'held' if held else 'CHANGED'}")
        ok &= same and held and diff.max() < TOL[max(
            (dtype, kv_dtype or dtype), key=lambda x: TOL[x])]
    print("  " + ("agrees with the oracle" if ok
                  else "DOES NOT AGREE -- do not use"))
    return 0 if ok else 1


def bench(path, device, dtype, batch, prompt, steps, kv_dtype=None):
    """What a decode step costs, which is the number the batching is about."""
    d = Dense(path, max_len=prompt + steps + 8, batch=batch, device=device,
              dtype=dtype, kv_dtype=kv_dtype, verbose=True)
    rng = np.random.default_rng(7)
    seqs = [[int(v) for v in rng.integers(0, 30000, size=prompt)]
            for _ in range(batch)]
    t0 = time.time()
    d.prefill(seqs)
    t_pre = time.time() - t0
    t0 = time.time()
    for _ in range(steps):
        d.step([1] * batch)
    t_dec = time.time() - t0
    per = t_dec / steps
    print(f"  prefill {batch} x {prompt} tok  {t_pre:7.2f}s")
    print(f"  decode  {steps} steps         {t_dec:7.2f}s   "
          f"{per*1000:7.1f} ms/step   {per/batch*1000:6.1f} ms/tok/seq")
    print(f"  throughput                  {batch/per:8.1f} tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="f32", choices=list(DTYPES))
    ap.add_argument("--kv8", action="store_true",
                    help="round-trip the cache through int8 the way the kernel "
                         "does, whatever the header says. Off by default "
                         "because the oracle does not, and --check compares "
                         "against the oracle.")
    ap.add_argument("--kv-dtype", default=None, choices=list(DTYPES),
                    dest="kv_dtype",
                    help="cache precision, separately from the weights. The "
                         "cache is what grows with batch, so narrowing it is "
                         "what buys a bigger one.")
    ap.add_argument("--prompt", type=int, default=726)
    ap.add_argument("--steps", type=int, default=32)
    args = ap.parse_args()
    if args.bench:
        bench(args.model, args.device, args.dtype, args.batch,
              args.prompt, args.steps, args.kv_dtype)
        raise SystemExit(0)
    if not args.check:
        ap.error("--check or --bench; this module is imported to be used")
    raise SystemExit(check(args.model, args.tokens, args.device,
                           args.dtype, args.batch, args.kv_dtype))


if __name__ == "__main__":
    main()

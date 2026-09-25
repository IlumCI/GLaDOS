# The host-side GPU work

`sha256d.cu` is the SHA-256d kernel the XPU plan calls Stage 1. It is
host-side and has nothing to do with the kernel in `src/` -- it is built with
NVIDIA's toolchain on Windows, because `ptxas` is closed source and nobody
outside NVIDIA has replaced it. GLaDOS would eventually ship a precompiled
cubin; this is where that cubin comes from.

## Building

Needs the CUDA toolkit. The recipe below was written when this was a Windows
machine and wanted an MSVC host compiler with it:

```
cmd /c "vcvarsall.bat x64 && nvcc -O3 -arch=sm_86 sha256d.cu -o sha256d.exe"
```

**The development host is Linux now and needs neither the shell nor the host
compiler dance**, which is worth recording because this file spent a paragraph
on the fact that the toolchain was hard to get:

```
nvcc -O3 -arch=sm_86 sha256d.cu -o sha256d
```

Measured here: `nvcc` 13.4, driver 615.71.09, and the card is the RTX 3050
Laptop with 4096 MiB -- `sm_86`, so the architecture flag above is right rather
than inherited. `nvidia-smi --query-gpu=name,memory.total,driver_version
--format=csv,noheader` is the one line that settles all three.

## BLAKE2b, which is for Equihash and not for anything here yet

`blake2b.cuh` is the first piece of `design/equihash.md`'s build, and it is the
one the brief calls mechanical. It is **not** `blake2s.cuh` with wider words:
64-bit state, rotations 32/24/16/63, twelve rounds, a 128-byte block. Nothing
was reusable.

It carries the parameter block, which an ordinary BLAKE2b does not bother with,
because Equihash needs two fields out of it -- a digest length that is 48 rather
than 64 and mixes into the state *before* the message, and a 16-byte
personalisation that differs per coin. `design/equihash.md` lists getting the
second wrong as the hazard that bites first, and what it looks like is every
share rejected on some chains and accepted on others with no diagnostic
anywhere.

```
nvcc -O3 -arch=sm_86 blake2btest.cu -o blake2btest && ./blake2btest
python3 ../tools/blake2b.py
```

Four vectors, each failing differently: RFC 7693's own published digest, the
empty message (where the finalisation flag is the whole test), 200 bytes (which
crosses the block boundary a single-block implementation gets right by
accident), and 48 bytes out under `ZcashPoW`/192/7, which exercises both
parameter fields at once. The two commands print the same four lines or this is
not finished. They do, on the card above, first run.

The oracle is `hashlib.blake2b`, which is CPython's own C implementation and
takes `digest_size` and `person` natively -- so it is not a second copy of this
file, which is the whole of what `algo.cuh` argues for.

## Correctness before speed

```
sha256d.exe                       # digest for block 125552's nonce
```

must print

```
1dbd981fe6985776b644b173a4d0385ddc1aa2a829688d1e0000000000000000
```

which is `sha256(sha256(header))` for a real block, checked against `hashlib`
rather than against this program. Reversed, that is the block hash
`00000000000000001e8d6829a8a21adc5d38d0a473b144b6765798e61f98bd1d`. Nothing
here is timed until that matches.

## Measuring

```
sha256d.exe --bench --npt 2 --blocks 2048 --iters 3000
```

**Run it for seconds, not milliseconds.** A short run measures a GPU that never
left its idle clock, and the first sweep taken that way was wrong by 42%. See
`design/xpu.md` for what that cost and what the sustained figures are.

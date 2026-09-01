# The host-side GPU work

`sha256d.cu` is the SHA-256d kernel the QuantumGPU plan calls Stage 1. It is
host-side and has nothing to do with the kernel in `src/` -- it is built with
NVIDIA's toolchain on Windows, because `ptxas` is closed source and nobody
outside NVIDIA has replaced it. GLaDOS would eventually ship a precompiled
cubin; this is where that cubin comes from.

## Building

Needs the CUDA toolkit and an MSVC host compiler. Both are on this machine
now, which is worth saying because `.cargo/config.toml` still records that they
were not:

```
cmd /c "vcvarsall.bat x64 && nvcc -O3 -arch=sm_86 sha256d.cu -o sha256d.exe"
```

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
`design/quantumgpu.md` for what that cost and what the sustained figures are.

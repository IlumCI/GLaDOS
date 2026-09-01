// The miner harness: one algorithm at a time, verified before it is timed.
//
//   miner.exe --algo sha256d --verify
//   miner.exe --algo blake2s --bench --npt 2 --blocks 2048 --iters 3000
//   miner.exe --all --verify
//
// Adding an algorithm is a header, a row in the enum, two specialisations and
// a row in ALGOS. Nothing else moves.
//
// **Verify before bench, and the harness enforces it.** `--bench` refuses to
// run if the digest does not match what `tools/algocheck.py` computed with
// hashlib. A wrong hash at 0.7 GH/s looks exactly like a right one, and the
// only thing that tells them apart is an implementation nobody here wrote.
//
// **Run for seconds, not milliseconds.** A short run measures a GPU that never
// left its idle clock: 70 ms of SHA-256d reads 0.448 GH/s and the same kernel
// over five seconds reads 0.645. See design/quantumgpu.md.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "algo.cuh"
#include "sha256d.cuh"
#include "blake2s.cuh"

enum Algo { ALGO_SHA256D = 0, ALGO_BLAKE2S = 1, ALGO_COUNT = 2 };

template <Algo A>
__device__ __forceinline__ void algo_hash(uint32_t n, uint32_t out[8]);

template <>
__device__ __forceinline__ void algo_hash<ALGO_SHA256D>(uint32_t n, uint32_t out[8]) {
    sha256d_hash(n, out);
}
template <>
__device__ __forceinline__ void algo_hash<ALGO_BLAKE2S>(uint32_t n, uint32_t out[8]) {
    blake2s_hash(n, out);
}

__constant__ uint32_t cTarget[8];

template <Algo A>
__global__ void verify_kernel(uint32_t nonce, uint32_t *out) {
    uint32_t h[8];
    algo_hash<A>(nonce, h);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = h[i];
}

// NPT independent nonces per thread. Measured flat from 1 to 4 on SHA-256d and
// negative at 8, because that kernel is already near full occupancy at 40
// registers -- ILP substitutes for occupancy only where occupancy is the
// constraint. Kept as a knob because a heavier algorithm may not be.
template <Algo A, int NPT>
__global__ void mine_kernel(uint32_t base, uint32_t *found) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n0 = base + idx * NPT;
    uint32_t h[NPT][8];
#pragma unroll
    for (int j = 0; j < NPT; ++j) algo_hash<A>(n0 + j, h[j]);
#pragma unroll
    for (int j = 0; j < NPT; ++j)
        if (below_target_be(h[j], cTarget)) atomicCAS(found, 0xffffffffu, n0 + j);
}

struct AlgoDesc {
    const char *name;
    void (*upload)(void);
    const char *(*expect)(void);
    uint32_t verify_nonce;
    bool little_endian_digest;  // BLAKE2s writes its state little-endian
    const char *note;
};

static const AlgoDesc ALGOS[ALGO_COUNT] = {
    { "sha256d", sha256d_upload, sha256d_expect, 0, false,
      "Bitcoin, two compressions a nonce, block 125552 as the vector" },
    { "blake2s", blake2s_upload, blake2s_expect, 0x12345678u, true,
      "RFC 7693, one 64-byte block, ten ARX rounds" },
};

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda: %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
    return 1; } } while (0)

static void hex_of(const uint32_t h[8], bool le, char *out) {
    for (int i = 0; i < 8; ++i)
        for (int b = 0; b < 4; ++b) {
            int shift = le ? (8 * b) : (8 * (3 - b));
            sprintf(out + 2 * (4 * i + b), "%02x", (h[i] >> shift) & 0xff);
        }
    out[64] = 0;
}

static int run_verify(int a, char *got) {
    uint32_t nonce = ALGOS[a].verify_nonce;
    if (a == ALGO_SHA256D) nonce = sha256d_nonce();
    uint32_t *d_out;
    CK(cudaMalloc(&d_out, 32));
    switch (a) {
        case ALGO_SHA256D: verify_kernel<ALGO_SHA256D><<<1,1>>>(nonce, d_out); break;
        case ALGO_BLAKE2S: verify_kernel<ALGO_BLAKE2S><<<1,1>>>(nonce, d_out); break;
    }
    CK(cudaDeviceSynchronize());
    uint32_t h[8];
    CK(cudaMemcpy(h, d_out, 32, cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    hex_of(h, ALGOS[a].little_endian_digest, got);
    return 0;
}

static int launch(int a, int npt, int blocks, int threads, uint32_t base, uint32_t *found) {
#define L(AA) switch (npt) { \
        case 1: mine_kernel<AA,1><<<blocks,threads>>>(base, found); break; \
        case 2: mine_kernel<AA,2><<<blocks,threads>>>(base, found); break; \
        case 4: mine_kernel<AA,4><<<blocks,threads>>>(base, found); break; \
        case 8: mine_kernel<AA,8><<<blocks,threads>>>(base, found); break; \
        default: fprintf(stderr, "npt must be 1, 2, 4 or 8\n"); return 1; }
    if (a == ALGO_SHA256D) { L(ALGO_SHA256D) } else { L(ALGO_BLAKE2S) }
#undef L
    return 0;
}

int main(int argc, char **argv) {
    const char *want = "sha256d";
    bool bench = false, all = false;
    int npt = 2, blocks = 2048, threads = 256, iters = 3000;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--algo") && i + 1 < argc) want = argv[++i];
        else if (!strcmp(argv[i], "--bench")) bench = true;
        else if (!strcmp(argv[i], "--all")) all = true;
        else if (!strcmp(argv[i], "--verify")) bench = false;
        else if (!strcmp(argv[i], "--npt") && i + 1 < argc) npt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    }

    uint32_t target[8];
    for (int i = 0; i < 8; ++i) target[i] = 0xffffffffu;
    CK(cudaMemcpyToSymbol(cTarget, target, sizeof target));

    for (int a = 0; a < ALGO_COUNT; ++a) {
        if (!all && strcmp(ALGOS[a].name, want)) continue;
        ALGOS[a].upload();
        CK(cudaDeviceSynchronize());

        char got[65];
        if (run_verify(a, got)) return 1;
        const char *exp = ALGOS[a].expect();
        bool ok = !strcmp(got, exp);
        printf("%-9s %s\n", ALGOS[a].name, ALGOS[a].note);
        printf("  digest   %s  %s\n", got, ok ? "ok" : "MISMATCH");
        if (!ok) {
            printf("  expected %s\n", exp);
            printf("  refusing to benchmark a hash that does not match hashlib\n");
            continue;
        }
        if (!bench) continue;

        uint32_t *d_found;
        CK(cudaMalloc(&d_found, 4));
        if (launch(a, npt, blocks, threads, 0, d_found)) return 1;  // warm-up
        CK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
        CK(cudaEventRecord(t0));
        for (int it = 0; it < iters; ++it) {
            uint32_t base = (uint32_t)it * (uint32_t)blocks * threads * npt;
            if (launch(a, npt, blocks, threads, base, d_found)) return 1;
        }
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        double hashes = (double)blocks * threads * npt * iters;
        printf("  %.3f GH/s   npt %d, %.1f ms, %.2fG hashes\n",
               hashes / (ms * 1e6), npt, ms, hashes / 1e9);
        cudaFree(d_found);
    }
    return 0;
}

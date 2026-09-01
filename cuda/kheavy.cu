// Does the tensor core help the heavy step of kHeavyHash?
//
// The step is a 64x64 matrix over GF(16) against the 64 nibbles of a 32-byte
// hash: product[i] = (sum_j M[i][j] * v[j]) >> 10, packed back into bytes and
// XORed with the input. Transcribed from rusty-kaspa's `heavy_hash`.
//
// **This is the step, not the algorithm.** Full kHeavyHash wraps it in two
// cSHAKE256 passes, and cSHAKE is not in hashlib -- shake_256 uses different
// domain separation, so it cannot be built from what is available here.
// Shipping a hash with no independent oracle would be asserting its own
// correctness, which `algo.cuh` exists to forbid. What is here is the only
// part where a tensor core could possibly matter, and it is checkable.
//
// Three ways, all three checked against tools/algocheck.py before any timing:
//
//   scalar  4096 int32 multiply-accumulates a nonce
//   dp4a    1024 four-way int8 dot products a nonce, on the INT pipe
//   wmma    16 nonces a warp through int8 tensor cores
//
// The last one does not fit the per-thread contract in algo.cuh and that is
// the point of running it: a tensor core is a warp-level instruction, so using
// one forces a batched shape that the other two do not need.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define N 64

__constant__ int8_t  cM[N][N];      // the matrix, nibbles in int8
__constant__ int32_t cMp[N][N / 4]; // the same rows packed 4 nibbles a word
__constant__ uint8_t cInput[32];    // the fixed verification input

// The same matrix again, in ordinary device memory. wmma::load_matrix_sync
// refuses __constant__ -- nvcc says so outright -- so the tensor path needs
// its own copy somewhere it is allowed to read from.
__device__ int8_t gM[N][N];

// Matches khh_matrix_ref() in tools/algocheck.py.
static int host_m(int i, int j) { return (i * 7 + j * 13 + ((i * j) >> 2)) & 0x0F; }

// v[i] is the high nibble for even i. Getting this backwards is the most
// likely error in the whole step and the oracle's zero-matrix claim is what
// would catch it.
__device__ __forceinline__ void nibbles(const uint8_t h[32], int8_t v[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = (h[i >> 1] >> (4 * (1 - (i & 1)))) & 0x0F;
}

__device__ __forceinline__ void pack_out(const uint8_t h[32], const int p[N], uint8_t out[32]) {
#pragma unroll
    for (int k = 0; k < 32; ++k)
        out[k] = h[k] ^ (uint8_t)(((p[2 * k] << 4) | p[2 * k + 1]) & 0xFF);
}

__device__ __forceinline__ void khh_scalar(const uint8_t h[32], uint8_t out[32]) {
    int8_t v[N];
    nibbles(h, v);
    int p[N];
#pragma unroll 1
    for (int i = 0; i < N; ++i) {
        int acc = 0;
#pragma unroll
        for (int j = 0; j < N; ++j) acc += (int)cM[i][j] * (int)v[j];
        p[i] = acc >> 10;
    }
    pack_out(h, p, out);
}

// Everything the step does except the multiply: the input, the nibbles, the
// shift, the packing and the xor. Subtracting this from the others is what
// turns "the matmul is probably not the bottleneck" into a number.
__device__ __forceinline__ void khh_null(const uint8_t h[32], uint8_t out[32]) {
    int8_t v[N];
    nibbles(h, v);
    int p[N];
#pragma unroll
    for (int i = 0; i < N; ++i) p[i] = v[i] & 0x0F;
    pack_out(h, p, out);
}

__device__ __forceinline__ void khh_dp4a(const uint8_t h[32], uint8_t out[32]) {
    int8_t v[N];
    nibbles(h, v);
    int32_t vp[N / 4];
#pragma unroll
    for (int j = 0; j < N / 4; ++j)
        vp[j] = (v[4*j] & 0xFF) | ((v[4*j+1] & 0xFF) << 8) |
                ((v[4*j+2] & 0xFF) << 16) | ((v[4*j+3] & 0xFF) << 24);
    int p[N];
#pragma unroll 1
    for (int i = 0; i < N; ++i) {
        int acc = 0;
#pragma unroll
        for (int j = 0; j < N / 4; ++j) acc = __dp4a(cMp[i][j], vp[j], acc);
        p[i] = acc >> 10;
    }
    pack_out(h, p, out);
}

// A cheap deterministic stand-in for the Keccak that would produce this hash.
// The benchmark measures the matrix step, so what feeds it only has to vary.
__device__ __forceinline__ void fake_input(uint32_t nonce, uint8_t h[32]) {
    uint32_t x = nonce * 2654435761u + 1013904223u;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        h[k] = (uint8_t)(x & 0xFF);
    }
}

__global__ void verify_wmma(uint8_t *o) {
    __shared__ int8_t sB[16 * N];
    __shared__ int32_t sC[N * 16];
    __shared__ uint8_t sH[16][32];
    int lane = threadIdx.x;
    for (int t = lane; t < 16; t += 32) {
        for (int k = 0; k < 32; ++k) sH[t][k] = cInput[k];
        int8_t v[N];
        nibbles(sH[t], v);
        for (int k = 0; k < N; ++k) sB[t * N + k] = v[k];
    }
    __syncwarp();
    for (int mt = 0; mt < 4; ++mt) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> acc;
        wmma::fill_fragment(acc, 0);
        for (int kt = 0; kt < 4; ++kt) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, &gM[mt * 16][kt * 16], N);
            wmma::load_matrix_sync(fb, sB + kt * 16, N);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(sC + mt * 16 * 16, acc, 16, wmma::mem_row_major);
    }
    __syncwarp();
    if (lane == 0) {
        int p[N];
        for (int mt = 0; mt < 4; ++mt)
            for (int r = 0; r < 16; ++r) p[mt * 16 + r] = sC[mt * 256 + r * 16 + 0] >> 10;
        pack_out(sH[0], p, o);
    }
}

__global__ void verify_scalar(uint8_t *o) { uint8_t h[32]; for (int i=0;i<32;++i) h[i]=cInput[i]; khh_scalar(h, o); }
__global__ void verify_dp4a(uint8_t *o)   { uint8_t h[32]; for (int i=0;i<32;++i) h[i]=cInput[i]; khh_dp4a(h, o); }

template <int MODE>
__global__ void bench_kernel(uint32_t base, uint32_t *sink) {
    uint32_t n = base + blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t h[32], o[32];
    fake_input(n, h);
    if (MODE == 0) khh_scalar(h, o); else if (MODE == 1) khh_dp4a(h, o); else khh_null(h, o);
    // Fold the result so nothing is dead code. A benchmark whose output is
    // unused measures an empty kernel.
    uint32_t acc = 0;
#pragma unroll
    for (int k = 0; k < 32; ++k) acc = acc * 31u + o[k];
    if (acc == 0xffffffffu) atomicExch(sink, acc);
}

// ---- the tensor core version -------------------------------------------
//
// One warp, sixteen nonces. A is the 64x64 matrix, B is 64x16 of nibbles, C is
// 64x16 of int32. Tiles are 16x16x16, so four along m and four along k: 16
// mma ops for what the scalar path spends 65,536 multiply-accumulates on.
//
// The batching is forced rather than chosen. wmma::mma_sync is a warp-wide
// instruction, so there is no way to use it for one nonce -- which is exactly
// the tension worth measuring, because a miner does not otherwise need its
// nonces batched.
__global__ void bench_wmma(uint32_t base, uint32_t *sink) {
    __shared__ int8_t sB[16 * N];      // 16 nonces, column-major, ld = 64
    __shared__ int32_t sC[N * 16];     // 64 x 16 accumulator
    __shared__ uint8_t sH[16][32];

    int lane = threadIdx.x;            // one warp per block
    uint32_t n0 = base + blockIdx.x * 16;

    for (int t = lane; t < 16; t += 32) {
        fake_input(n0 + t, sH[t]);
        int8_t v[N];
        nibbles(sH[t], v);
        for (int k = 0; k < N; ++k) sB[t * N + k] = v[k];
    }
    __syncwarp();

    for (int mt = 0; mt < 4; ++mt) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> acc;
        wmma::fill_fragment(acc, 0);
        for (int kt = 0; kt < 4; ++kt) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, &gM[mt * 16][kt * 16], N);
            wmma::load_matrix_sync(fb, sB + kt * 16, N);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(sC + mt * 16 * 16, acc, 16, wmma::mem_row_major);
    }
    __syncwarp();

    uint32_t a = 0;
    for (int t = lane; t < 16; t += 32) {
        int p[N];
        for (int mt = 0; mt < 4; ++mt)
            for (int r = 0; r < 16; ++r) p[mt * 16 + r] = sC[mt * 256 + r * 16 + t] >> 10;
        uint8_t o[32];
        pack_out(sH[t], p, o);
        for (int k = 0; k < 32; ++k) a = a * 31u + o[k];
    }
    if (a == 0xffffffffu) atomicExch(sink, a);
}

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    fprintf(stderr,"cuda: %s line %d\n",cudaGetErrorString(e_),__LINE__); return 1;} } while(0)

static const char *EXPECT = "865d6b59ed37292a4cde38de0f728eec9092087992211e7de697d2acd55e6ff6";

int main(int argc, char **argv) {
    bool bench = false;
    int blocks = 8192, iters = 200;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--bench")) bench = true;
        else if (!strcmp(argv[i], "--blocks") && i+1 < argc) blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc) iters = atoi(argv[++i]);
    }

    int8_t M[N][N];
    int32_t Mp[N][N/4];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) M[i][j] = (int8_t)host_m(i, j);
        for (int j = 0; j < N/4; ++j)
            Mp[i][j] = (M[i][4*j] & 0xFF) | ((M[i][4*j+1] & 0xFF) << 8) |
                       ((M[i][4*j+2] & 0xFF) << 16) | ((M[i][4*j+3] & 0xFF) << 24);
    }
    // sha256("glados kheavyhash matrix vector"), from algocheck.py.
    static const uint8_t IN[32] = {
        0xb2,0x6e,0x58,0x1a,0xde,0x74,0x0a,0x19,0x7f,0xed,0x0c,0xed,0x3c,0x46,0xad,0xdf,
        0xa3,0xa1,0x3b,0x3a,0xa1,0x12,0x5d,0x4e,0xd5,0xd4,0xe1,0x9f,0xe6,0x1d,0x4b,0xc5};
    CK(cudaMemcpyToSymbol(cM, M, sizeof M));
    CK(cudaMemcpyToSymbol(gM, M, sizeof M));
    CK(cudaMemcpyToSymbol(cMp, Mp, sizeof Mp));
    CK(cudaMemcpyToSymbol(cInput, IN, sizeof IN));

    uint8_t *d_o; CK(cudaMalloc(&d_o, 32));
    uint8_t o[32]; char hex[65];
    const char *names[3] = { "scalar", "dp4a", "wmma" };
    int bad = 0;
    for (int m = 0; m < 3; ++m) {
        if (m == 0) verify_scalar<<<1,1>>>(d_o);
        else if (m == 1) verify_dp4a<<<1,1>>>(d_o);
        else verify_wmma<<<1,32>>>(d_o);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(o, d_o, 32, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 32; ++i) sprintf(hex + 2*i, "%02x", o[i]);
        hex[64] = 0;
        int ok = !strcmp(hex, EXPECT);
        if (!ok) bad++;
        printf("%-7s %s  %s\n", names[m], hex, ok ? "ok" : "MISMATCH");
    }
    printf("expect  %s\n", EXPECT);
    if (bad) {
        // The point of the whole file. A tensor path that is fast and wrong
        // looks exactly like one that is fast and right, and only the oracle
        // in tools/algocheck.py can tell them apart.
        printf("refusing to benchmark: %d path(s) disagree with the oracle\n", bad);
        return 1;
    }
    if (!bench) return 0;

    uint32_t *sink; CK(cudaMalloc(&sink, 4));
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    const char *bn[4] = { "scalar", "dp4a", "wmma", "no-matmul" };
    for (int m = 0; m < 4; ++m) {
        // Warm-up, discarded: a cold clock reads about a third low.
        for (int w = 0; w < 20; ++w) {
            if (m == 0) bench_kernel<0><<<blocks,256>>>(0, sink);
            else if (m == 1) bench_kernel<1><<<blocks,256>>>(0, sink);
            else if (m == 3) bench_kernel<2><<<blocks,256>>>(0, sink);
            else bench_wmma<<<blocks * 16, 32>>>(0, sink);
        }
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(t0));
        for (int it = 0; it < iters; ++it) {
            uint32_t base = (uint32_t)it * blocks * 256;
            if (m == 0) bench_kernel<0><<<blocks,256>>>(base, sink);
            else if (m == 1) bench_kernel<1><<<blocks,256>>>(base, sink);
            else if (m == 3) bench_kernel<2><<<blocks,256>>>(base, sink);
            else bench_wmma<<<blocks * 16, 32>>>(base, sink);
        }
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
        double steps = (double)blocks * 256 * iters;
        printf("%-7s %8.1f ms  %7.2f Mstep/s\n", bn[m], ms, steps / (ms * 1e3));
    }
    return 0;
}

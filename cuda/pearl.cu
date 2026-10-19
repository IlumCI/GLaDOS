// Pearl on an NVIDIA GPU, as glados-miner's helper: the whole mining round on
// the device and a proof for every tile under the share bound.
//
// The same round as the web miner's (glados-pool mine/js/pearl-gpu.js, checked
// there against a JavaScript reference whose proofs Pearl's own verifier
// accepts): draw A and B^t from a seed, commit each as Pearl's keyed BLAKE3
// Merkle tree, derive the noise seeds from the two roots, noise every row,
// then the jackpot of every 16 x 16 tile -- the int8 GEMM, folded into 16
// words per tile every `rank` steps of k, hashed with keyed BLAKE3. A tile
// whose hash, read little-endian, is under 2^(256 - bits) is a share; its
// proof is Pearl's PlainProof, bincode-encoded, as the pool's `plain_proof`.
//
// Unlike the browser there is no round trip mid-round: the few small hashes
// between the trees and the noise (the roots, the seeds, the permutations)
// run here on the host.
//
// ### Driven over a pipe, like xpu.cu
//
//     <- ready <m> <n> <k> <rank>
//     -> job <id> <header: 152 hex> <bits>
//     -> round
//     <- share <id> <plain_proof base64>      (zero or more)
//     <- done <tiles> <milliseconds>
//     -> check <m> <n> <s0> <s1> <header>     (tests: every tile's hash, two proofs)
//     -> quit
//
// The protocol to the pool stays in Rust (miner/), as it does for xpu.cu.
//
// Build: nvcc -O3 -arch=sm_86 pearl.cu -o pearl

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <cuda_runtime.h>

static const int K = 2048, RANK = 128, TILE = 16, CHUNK = 1024, PROOFS_PER_ROUND = 4, HIT_CAP = 1024;

struct W8 { uint32_t w[8]; };

// ------------------------------------------------------------------ BLAKE3 --
__host__ __device__ inline uint32_t rotr32(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
__host__ __device__ inline void g_mix(uint32_t *v, int a, int b, int c, int d, uint32_t x, uint32_t y) {
    v[a] = v[a] + v[b] + x; v[d] = rotr32(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];     v[b] = rotr32(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + y; v[d] = rotr32(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];     v[b] = rotr32(v[b] ^ v[c], 7);
}
enum { CHUNK_START = 1, CHUNK_END = 2, PARENT = 4, ROOT = 8, KEYED = 16 };

__host__ __device__ inline void compress(const uint32_t cv[8], const uint32_t msg[16], uint32_t counter, uint32_t blen, uint32_t flags, uint32_t out[8]) {
    const uint32_t IV[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    const int P[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
    uint32_t v[16], m[16], t[16];
    for (int i = 0; i < 8; i++) v[i] = cv[i];
    for (int i = 0; i < 4; i++) v[8 + i] = IV[i];
    v[12] = counter; v[13] = 0; v[14] = blen; v[15] = flags;
    for (int i = 0; i < 16; i++) m[i] = msg[i];
    for (int r = 0; r < 7; r++) {
        g_mix(v, 0, 4, 8, 12, m[0], m[1]);   g_mix(v, 1, 5, 9, 13, m[2], m[3]);
        g_mix(v, 2, 6, 10, 14, m[4], m[5]);  g_mix(v, 3, 7, 11, 15, m[6], m[7]);
        g_mix(v, 0, 5, 10, 15, m[8], m[9]);  g_mix(v, 1, 6, 11, 12, m[10], m[11]);
        g_mix(v, 2, 7, 8, 13, m[12], m[13]); g_mix(v, 3, 4, 9, 14, m[14], m[15]);
        for (int i = 0; i < 16; i++) t[i] = m[P[i]];
        for (int i = 0; i < 16; i++) m[i] = t[i];
    }
    for (int i = 0; i < 8; i++) out[i] = v[i] ^ v[i + 8];
}

static void le_words(const uint8_t *b, uint32_t *w, int n) { for (int i = 0; i < n; i++) w[i] = b[4*i] | (b[4*i+1] << 8) | (b[4*i+2] << 16) | ((uint32_t)b[4*i+3] << 24); }
static void words_le(const uint32_t *w, uint8_t *b, int n) { for (int i = 0; i < n; i++) { b[4*i] = w[i]; b[4*i+1] = w[i] >> 8; b[4*i+2] = w[i] >> 16; b[4*i+3] = w[i] >> 24; } }

// BLAKE3 of up to one chunk (1024 bytes), keyed when `key` is given. Every
// hash between the trees and the noise is one of these.
static void blake3_small(const uint8_t *key, const uint8_t *data, int len, uint8_t out[32]) {
    uint32_t cv[8];
    if (key) le_words(key, cv, 8);
    else { const uint32_t IV[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u}; memcpy(cv, IV, 32); }
    int blocks = len == 0 ? 1 : (len + 63) / 64;
    for (int b = 0; b < blocks; b++) {
        uint8_t blk[64] = {0};
        int n = std::min(64, len - b * 64); if (n < 0) n = 0;
        memcpy(blk, data + b * 64, n);
        uint32_t m[16]; le_words(blk, m, 16);
        uint32_t f = (key ? KEYED : 0) | (b == 0 ? CHUNK_START : 0) | (b == blocks - 1 ? (CHUNK_END | ROOT) : 0);
        compress(cv, m, 0, n, f, cv);
    }
    words_le(cv, out, 8);
}

// ---------------------------------------------------------------- constants --
static uint8_t LABEL_A[32], LABEL_B[32];
static const uint8_t SALT_A[32] = {0x82, 0x49, 0x40, 0x6c, 0xa0, 0xed, 0x15, 0x16, 0x96, 0x16, 0xf6, 0x92, 0xfc, 0xf0, 0x76, 0xf8, 0x92, 0xdb, 0xdb, 0x2a, 0x70, 0x23, 0xb8, 0x52, 0xf0, 0xd4, 0x77, 0x19, 0xc3, 0x90, 0x01, 0x7b};
static const uint8_t SALT_B[32] = {0x11, 0x30, 0x06, 0x32, 0xec, 0x63, 0x01, 0xca, 0x2b, 0xe2, 0xaf, 0x71, 0x8b, 0x3f, 0x4d, 0x4f, 0x1a, 0xe9, 0xc6, 0x39, 0x88, 0xe8, 0xcc, 0x04, 0x48, 0x44, 0x30, 0x1d, 0x71, 0xb8, 0x9a, 0xa9};

// MiningConfiguration::to_bytes for a 16 x 16 contiguous tile.
static void config_bytes(int k, int rank, uint8_t out[52]) {
    memset(out, 0, 52);
    out[0] = k; out[1] = k >> 8; out[2] = k >> 16; out[3] = k >> 24;
    out[4] = rank; out[5] = rank >> 8;
    out[9] = TILE - 1; out[15] = TILE - 1;
}

// ------------------------------------------------------------------ kernels --
__device__ inline uint32_t mix(uint32_t x) {
    uint32_t s = x * 747796405u + 2891336453u;
    uint32_t w = ((s >> ((s >> 28) + 4)) ^ s) * 277803737u;
    return (w >> 22) ^ w;
}
__device__ inline uint32_t element(uint32_t j, uint32_t s0, uint32_t s1) { return ((mix(mix(j ^ s0) + s1) % 129u) + 192u) & 255u; }

__global__ void k_draw(uint32_t *out, uint32_t n, uint32_t s0, uint32_t s1) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t j = i * 4;
    out[i] = element(j, s0, s1) | (element(j + 1, s0, s1) << 8) | (element(j + 2, s0, s1) << 16) | (element(j + 3, s0, s1) << 24);
}

__global__ void k_leaves(const uint32_t *mat, uint32_t *tree, uint32_t n, W8 key) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) cv[i] = key.w[i];
    for (int b = 0; b < 16; b++) {
        uint32_t m[16];
        for (int i = 0; i < 16; i++) m[i] = mat[c * 256 + b * 16 + i];
        compress(cv, m, c, 64, KEYED | (b == 0 ? CHUNK_START : 0) | (b == 15 ? CHUNK_END : 0), cv);
    }
    for (int i = 0; i < 8; i++) tree[c * 8 + i] = cv[i];
}

// One level of Pearl's tree: pairs, and a lone last node promoted unhashed.
__global__ void k_parents(uint32_t *tree, uint32_t in_off, uint32_t in_len, uint32_t out_off, uint32_t out_len, W8 key) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_len) return;
    uint32_t l = in_off + 2 * i, o[8];
    if (2 * i + 1 < in_len) {
        uint32_t m[16];
        for (int j = 0; j < 16; j++) m[j] = tree[l * 8 + j];
        compress(key.w, m, 0, 64, KEYED | PARENT, o);
    } else {
        for (int j = 0; j < 8; j++) o[j] = tree[l * 8 + j];
    }
    for (int j = 0; j < 8; j++) tree[(out_off + i) * 8 + j] = o[j];
}

// The uniform noise: block idx -> one keyed BLAKE3 of (1 + idx | label), each
// output byte to (b & 63) - 32.
__global__ void k_noise_rows(uint32_t *E, uint32_t n, W8 key, W8 label) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t m[16] = {0}, h[8];
    m[0] = 1 + idx;
    for (int i = 0; i < 8; i++) m[8 + i] = label.w[i];
    compress(key.w, m, 0, 64, CHUNK_START | CHUNK_END | ROOT | KEYED, h);
    for (int q = 0; q < 8; q++) {
        uint32_t o = 0;
        for (int e = 0; e < 4; e++) o |= ((((h[q] >> (8 * e)) & 63u) + 224u) & 255u) << (8 * e);
        E[idx * 8 + q] = o;
    }
}

__device__ inline int sbyte(uint32_t w, int e) { return (int)(w << ((3 - e) * 8)) >> 24; }

__global__ void k_noised(const uint32_t *raw, uint32_t *out, const uint32_t *perm, const uint32_t *E, uint32_t n, uint32_t kw, uint32_t r) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t base = (i / kw) * r, src = raw[i], o = 0;
    for (int e = 0; e < 4; e++) {
        uint32_t p = perm[(i % kw) * 4 + e];
        uint32_t a = base + (p & 0xffff), b = base + (p >> 16);
        int noise = sbyte(E[a / 4], a % 4) - sbyte(E[b / 4], b % 4);
        o |= ((uint32_t)(sbyte(src, e) + noise) & 255u) << (8 * e);
    }
    out[i] = o;
}

// The jackpot, jackpot.wgsl's port: a block of BM x BN outputs per 256
// threads, TM x TN per thread (always inside one 16 x 16 tile), k staged
// through shared memory 32 values at a time, __dp4a for the int8 MACs. Every
// `rank` values of k a tile's accumulators are XOR-ed into its jackpot word.
template <int TM, int TN>
__global__ void __launch_bounds__(256) k_jackpot(const uint32_t *__restrict__ A, const uint32_t *__restrict__ B, int n, int kw, int rw,
                                                 W8 key, W8 bound, uint32_t *hits, uint32_t *dbg) {
    constexpr int BM = 16 * TM, BN = 16 * TN, KCW = 8, KS = KCW + 1, TPR = BN / 16, NT = (BM / 16) * TPR;
    __shared__ uint32_t As[BM * KS], Bs[BN * KS], fold[NT], jack[NT * 16];
    const int li = threadIdx.x, tx = li % 16, ty = li / 16;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const int mine = (ty * TM / 16) * TPR + (tx * TN / 16);
    for (int i = li; i < NT * 16; i += 256) jack[i] = 0;
    if (li < NT) fold[li] = 0;
    int acc[TM * TN];
    for (int e = 0; e < TM * TN; e++) acc[e] = 0;
    const int steps = kw / KCW, every = rw / KCW;
    __syncthreads();
    for (int s = 0; s < steps; s++) {
        const int k0 = s * KCW;
        for (int i = li; i < BM * KCW; i += 256) As[(i / KCW) * KS + i % KCW] = A[(size_t)(row0 + i / KCW) * kw + k0 + i % KCW];
        for (int i = li; i < BN * KCW; i += 256) Bs[(i / KCW) * KS + i % KCW] = B[(size_t)(col0 + i / KCW) * kw + k0 + i % KCW];
        __syncthreads();
#pragma unroll
        for (int c = 0; c < KCW; c++) {
            int a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) a[i] = (int)As[(ty * TM + i) * KS + c];
#pragma unroll
            for (int j = 0; j < TN; j++) b[j] = (int)Bs[(tx * TN + j) * KS + c];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) acc[i * TN + j] = __dp4a(a[i], b[j], acc[i * TN + j]);
        }
        __syncthreads();
        if ((s + 1) % every == 0) {
            uint32_t x = 0;
            for (int e = 0; e < TM * TN; e++) x ^= (uint32_t)acc[e];
            atomicXor(&fold[mine], x);
            __syncthreads();
            if (li < NT) {
                const int tid = ((s + 1) / every - 1) % 16;
                const uint32_t f = fold[li], j0 = jack[li * 16 + tid];
                fold[li] = 0;
                jack[li * 16 + tid] = ((j0 << 13) | (j0 >> 19)) ^ f;
            }
            __syncthreads();
        }
    }
    if (li < NT) {
        uint32_t h[8];
        compress(key.w, &jack[li * 16], 0, 64, CHUNK_START | CHUNK_END | ROOT | KEYED, h);
        const uint32_t tr = blockIdx.y * (BM / 16) + li / TPR, tc = blockIdx.x * TPR + li % TPR;
        if (dbg) for (int i = 0; i < 8; i++) dbg[(tr * (n / 16) + tc) * 8 + i] = h[i];
        bool under = true;
        for (int i = 7; i >= 0; i--) {
            if (h[i] < bound.w[i]) { under = true; break; }
            if (h[i] > bound.w[i]) { under = false; break; }
        }
        if (under) {
            uint32_t at = atomicAdd(&hits[0], 1u);
            if (at < HIT_CAP) { hits[2 + 2 * at] = tr; hits[3 + 2 * at] = tc; }
        }
    }
}

// The jackpot on tensor cores (sm_80 and later): the same sums, through
// `mma.sync.m16n8k32` on int8, which is where an Ampere card's int8 throughput
// is -- DP4A above uses the ordinary cores and reaches a fraction of it.
//
// A block of 256 threads owns 128 x 128 outputs, 64 jackpot tiles. Each of its
// 8 warps owns 32 x 64 of them -- 2 x 4 whole tiles -- so a tile's sums all live
// in one warp and its XOR fold is a warp reduction, no atomics. k is staged
// through shared memory 128 values at a time, which is the rank: one fold per
// stage. Rows are padded to 144 bytes so the fragment loads of the eight
// row-groups land in different banks.
//
// Fragment layout for m16n8k32 .s8 (PTX ISA, "mma.m16n8k32"): with g = lane / 4
// and t = lane % 4, A's four registers hold rows g and g + 8 at k = 4t and 4t + 16;
// B's two hold column g at k = 4t and 4t + 16; C's four are rows g and g + 8.
// Which element sits where does not matter to the fold -- it XORs a tile's
// 256 sums whatever their order -- only which tile each belongs to.
__device__ inline void mma_s8(int c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    // Only where the instruction exists. The release builds Pascal and Turing
    // code too; there this compiles to nothing and the host never launches it.
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__global__ void __launch_bounds__(256) k_jackpot_tc(const uint32_t *__restrict__ A, const uint32_t *__restrict__ B, int n, int kw,
                                                    W8 key, W8 bound, uint32_t *hits, uint32_t *dbg) {
    constexpr int BM = 128, BN = 128, KC = 32, RS = 36; // KC words = 128 values = the rank; RS = padded row stride in words
    __shared__ __align__(16) uint32_t sA[BM * RS], sB[BN * RS];
    __shared__ uint32_t jack[64 * 16];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int wm = warp >> 1, wn = warp & 1;           // the warp's 32 rows and 64 columns
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    for (int i = threadIdx.x; i < 64 * 16; i += 256) jack[i] = 0;
    int c[2][8][4];
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < 8; b++) c[a][b][0] = c[a][b][1] = c[a][b][2] = c[a][b][3] = 0;
    const int chunks = kw / KC;
    for (int ch = 0; ch < chunks; ch++) {
        __syncthreads();
        // Stage 128 rows x 128 bytes of each, four 16-byte loads a thread a matrix.
        for (int i = threadIdx.x; i < BM * (KC / 4); i += 256) {
            const int r = i / (KC / 4), q = (i % (KC / 4)) * 4;
            *(uint4 *)&sA[r * RS + q] = *(const uint4 *)&A[(size_t)(row0 + r) * kw + ch * KC + q];
            *(uint4 *)&sB[r * RS + q] = *(const uint4 *)&B[(size_t)(col0 + r) * kw + ch * KC + q];
        }
        __syncthreads();
#pragma unroll
        for (int s = 0; s < KC / 8; s++) { // four k32 steps
            const int kq = s * 8 + t;     // word of k = 32s + 4t
            uint32_t a[2][4], b[8][2];
#pragma unroll
            for (int mb = 0; mb < 2; mb++) {
                const int r = wm * 32 + mb * 16 + g;
                a[mb][0] = sA[r * RS + kq]; a[mb][1] = sA[(r + 8) * RS + kq];
                a[mb][2] = sA[r * RS + kq + 4]; a[mb][3] = sA[(r + 8) * RS + kq + 4];
            }
#pragma unroll
            for (int nb = 0; nb < 8; nb++) {
                const int col = wn * 64 + nb * 8 + g;
                b[nb][0] = sB[col * RS + kq]; b[nb][1] = sB[col * RS + kq + 4];
            }
#pragma unroll
            for (int mb = 0; mb < 2; mb++)
#pragma unroll
                for (int nb = 0; nb < 8; nb++) mma_s8(c[mb][nb], a[mb][0], a[mb][1], a[mb][2], a[mb][3], b[nb][0], b[nb][1]);
        }
        // The fold: each of the warp's 8 tiles, its 256 sums XOR-ed together.
        const int tid = ch % 16;
#pragma unroll
        for (int mb = 0; mb < 2; mb++)
#pragma unroll
            for (int tc = 0; tc < 4; tc++) {
                uint32_t x = 0;
#pragma unroll
                for (int e = 0; e < 4; e++) x ^= (uint32_t)c[mb][2 * tc][e] ^ (uint32_t)c[mb][2 * tc + 1][e];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, o);
                if (lane == 0) {
                    const int tile = (wm * 2 + mb) * 8 + (wn * 4 + tc);
                    const uint32_t j0 = jack[tile * 16 + tid];
                    jack[tile * 16 + tid] = ((j0 << 13) | (j0 >> 19)) ^ x;
                }
            }
    }
    __syncthreads();
    if (threadIdx.x < 64) {
        const int tile = threadIdx.x;
        uint32_t h[8];
        compress(key.w, &jack[tile * 16], 0, 64, CHUNK_START | CHUNK_END | ROOT | KEYED, h);
        const uint32_t tr = blockIdx.y * 8 + tile / 8, tc = blockIdx.x * 8 + tile % 8;
        if (dbg) for (int i = 0; i < 8; i++) dbg[(tr * (n / 16) + tc) * 8 + i] = h[i];
        bool under = true;
        for (int i = 7; i >= 0; i--) {
            if (h[i] < bound.w[i]) { under = true; break; }
            if (h[i] > bound.w[i]) { under = false; break; }
        }
        if (under) {
            uint32_t at = atomicAdd(&hits[0], 1u);
            if (at < HIT_CAP) { hits[2 + 2 * at] = tr; hits[3 + 2 * at] = tc; }
        }
    }
}

// --------------------------------------------------------------------- host --
// Tensor cores where there are int8 mma's of this shape (sm_80+); DP4A
// everywhere else. PEARL_KERNEL=dp4a forces the fallback, for comparing them.
static bool use_tc = false;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); fflush(stdout); exit(2); } } while (0)

struct Level { uint32_t off, len; };
struct Tree {
    std::vector<Level> lv; uint32_t nodes = 0;
    void plan(uint32_t leaves) { lv.clear(); lv.push_back({0, leaves}); while (lv.back().len > 2) { Level p = lv.back(); lv.push_back({p.off + p.len, (p.len + 1) / 2}); } nodes = 0; for (auto &l : lv) nodes += l.len; }
};

struct Ctx {
    int m = 0, n = 0, k = K, rank = RANK, kw = K / 4;
    uint32_t *Araw = nullptr, *Braw = nullptr, *Anz = nullptr, *Bnz = nullptr, *treeA = nullptr, *treeB = nullptr;
    uint32_t *permA = nullptr, *permB = nullptr, *EA = nullptr, *EB = nullptr, *hits = nullptr, *dbg = nullptr;
    Tree tA, tB;
    uint8_t jobKey[32], rootA[32], rootB[32];

    bool alloc(int m_, int n_, bool debug) {
        m = m_; n = n_;
        tA.plan((uint32_t)((size_t)m * k / CHUNK)); tB.plan((uint32_t)((size_t)n * k / CHUNK));
        size_t a = (size_t)m * k, b = (size_t)n * k;
        if (cudaMalloc(&Araw, a) || cudaMalloc(&Braw, b) || cudaMalloc(&Anz, a) || cudaMalloc(&Bnz, b) ||
            cudaMalloc(&treeA, (size_t)tA.nodes * 32) || cudaMalloc(&treeB, (size_t)tB.nodes * 32) ||
            cudaMalloc(&permA, k * 4) || cudaMalloc(&permB, k * 4) || cudaMalloc(&EA, (size_t)m * rank) || cudaMalloc(&EB, (size_t)n * rank) ||
            cudaMalloc(&hits, (2 + 2 * HIT_CAP) * 4)) return false;
        if (debug && cudaMalloc(&dbg, (size_t)(m / 16) * (n / 16) * 32)) return false;
        return true;
    }
    void release() {
        for (uint32_t *p : {Araw, Braw, Anz, Bnz, treeA, treeB, permA, permB, EA, EB, hits, dbg}) if (p) cudaFree(p);
        Araw = Braw = Anz = Bnz = treeA = treeB = permA = permB = EA = EB = hits = dbg = nullptr;
    }

    void commit(uint32_t *raw, uint32_t *tree, Tree &t, W8 key, uint8_t root[32]) {
        k_leaves<<<(t.lv[0].len + 63) / 64, 64>>>(raw, tree, t.lv[0].len, key);
        for (size_t i = 1; i < t.lv.size(); i++)
            k_parents<<<(t.lv[i].len + 63) / 64, 64>>>(tree, t.lv[i - 1].off, t.lv[i - 1].len, t.lv[i].off, t.lv[i].len, key);
        Level top = t.lv.back();
        uint32_t nodes[16];
        CK(cudaMemcpy(nodes, tree + (size_t)top.off * 8, top.len * 32, cudaMemcpyDeviceToHost));
        if (top.len == 2) { uint32_t o[8]; compress(key.w, nodes, 0, 64, KEYED | PARENT | ROOT, o); words_le(o, root, 8); }
        else words_le(nodes, root, 8);
    }

    static void perms(const uint8_t *label, const uint8_t *seed, int k, int r, std::vector<uint32_t> &out) {
        out.resize(k);
        for (int i = 0; i < (k + 7) / 8; i++) {
            uint8_t msg[64] = {0}, h[32];
            uint32_t pv = 1 + i;
            msg[4] = pv; msg[5] = pv >> 8; msg[6] = pv >> 16; msg[7] = pv >> 24; // prepended at word 1
            memcpy(msg + 32, label, 32);
            blake3_small(seed, msg, 64, h);
            for (int j = 0; j < 8 && i * 8 + j < k; j++) {
                uint32_t rr = h[4*j] | (h[4*j+1] << 8) | (h[4*j+2] << 16) | ((uint32_t)h[4*j+3] << 24);
                uint32_t first = rr & (r - 1);
                uint32_t second = first ^ (1 + (uint32_t)(((uint64_t)(r - 1) * rr) >> 32));
                out[i * 8 + j] = first | (second << 16);
            }
        }
    }

    // One round: header (76 bytes), seeds, share bound bits. Hits land in `hits`.
    uint32_t round(const uint8_t header[76], uint32_t s0, uint32_t s1, int bits) {
        uint8_t hc[128]; memcpy(hc, header, 76); config_bytes(k, rank, hc + 76);
        blake3_small(nullptr, hc, 128, jobKey);
        W8 jk; le_words(jobKey, jk.w, 8);
        uint32_t na = (uint32_t)((size_t)m * kw), nb = (uint32_t)((size_t)n * kw);
        k_draw<<<(na + 255) / 256, 256>>>(Araw, na, s0, s1);
        k_draw<<<(nb + 255) / 256, 256>>>(Braw, nb, s0 ^ 0x9e3779b9u, s1 ^ 0x7f4a7c15u);
        commit(Araw, treeA, tA, jk, rootA);
        commit(Braw, treeB, tB, jk, rootB);
        // The v3 seed chain: each root bound to its dimension, then chained.
        auto bind = [](const uint8_t root[32], uint32_t dim, const uint8_t salt[32], uint8_t out[32]) {
            uint8_t b[64] = {0}; memcpy(b, root, 32); b[32] = dim; b[33] = dim >> 8; b[34] = dim >> 16; b[35] = dim >> 24;
            blake3_small(salt, b, 64, out);
        };
        uint8_t hA[32], hB[32], bSeed[32], aSeed[32], cat[64];
        bind(rootA, m, SALT_A, hA); bind(rootB, n, SALT_B, hB);
        memcpy(cat, jobKey, 32); memcpy(cat + 32, hB, 32); blake3_small(nullptr, cat, 64, bSeed);
        memcpy(cat, bSeed, 32); memcpy(cat + 32, hA, 32); blake3_small(nullptr, cat, 64, aSeed);
        std::vector<uint32_t> pa, pb;
        perms(LABEL_A, aSeed, k, rank, pa); perms(LABEL_B, bSeed, k, rank, pb);
        CK(cudaMemcpy(permA, pa.data(), k * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(permB, pb.data(), k * 4, cudaMemcpyHostToDevice));
        W8 ka, kb, la, lb; le_words(aSeed, ka.w, 8); le_words(bSeed, kb.w, 8); le_words(LABEL_A, la.w, 8); le_words(LABEL_B, lb.w, 8);
        k_noise_rows<<<(m * rank / 32 + 63) / 64, 64>>>(EA, m * rank / 32, ka, la);
        k_noise_rows<<<(n * rank / 32 + 63) / 64, 64>>>(EB, n * rank / 32, kb, lb);
        k_noised<<<(na + 255) / 256, 256>>>(Araw, Anz, permA, EA, na, kw, rank);
        k_noised<<<(nb + 255) / 256, 256>>>(Braw, Bnz, permB, EB, nb, kw, rank);
        W8 bound; for (int i = 0; i < 8; i++) bound.w[i] = 0xffffffffu;
        for (int b = 0; b < bits; b++) { int bit = 255 - b; bound.w[bit >> 5] &= ~(1u << (bit & 31)); }
        CK(cudaMemset(hits, 0, 8));
        if (use_tc && m % 128 == 0 && n % 128 == 0 && rank == 128)
            k_jackpot_tc<<<dim3(n / 128, m / 128), 256>>>(Anz, Bnz, n, kw, ka, bound, hits, dbg);
        else
            k_jackpot<4, 4><<<dim3(n / 64, m / 64), 256>>>(Anz, Bnz, n, kw, rank / 4, ka, bound, hits, dbg);
        CK(cudaGetLastError());
        uint32_t count = 0;
        CK(cudaMemcpy(&count, hits, 4, cudaMemcpyDeviceToHost));
        return count;
    }

    // Pearl's MerkleProof siblings for `leaves`, in compute_root's order.
    static void siblings(const std::vector<uint8_t> &flat, const Tree &t, const std::vector<uint32_t> &leaves, std::vector<uint8_t> &out) {
        std::set<uint32_t> cur(leaves.begin(), leaves.end());
        uint32_t len = t.lv[0].len; size_t lv = 0;
        auto node = [&](size_t level, uint32_t i) { const uint8_t *p = &flat[((size_t)t.lv[level].off + i) * 32]; out.insert(out.end(), p, p + 32); };
        if (len == 1) return;
        while (len > 2) {
            std::set<uint32_t> next;
            for (uint32_t i : cur) {
                if (i % 2 == 0) { if (!cur.count(i + 1) && i + 1 < len) node(lv, i + 1); next.insert(i / 2); }
                else if (!cur.count(i - 1)) { node(lv, i - 1); next.insert((i - 1) / 2); }
            }
            cur = next; len = (len + 1) / 2; lv++;
        }
        if (!cur.count(0)) node(lv, 0);
        if (!cur.count(1)) node(lv, 1);
    }

    // The PlainProof for tile (tr, tc) of the last round, bincode-encoded.
    std::vector<uint8_t> proof(uint32_t tr, uint32_t tc) {
        std::vector<uint8_t> o;
        auto u64 = [&](uint64_t x) { for (int i = 0; i < 8; i++) o.push_back((uint8_t)(x >> (8 * i))); };
        u64(m); u64(n); u64(k); u64(rank);
        auto side = [&](uint32_t *raw, uint32_t *tree, Tree &t, const uint8_t root[32], uint32_t first) {
            std::vector<uint8_t> rows((size_t)TILE * k), flat((size_t)t.nodes * 32), sib;
            CK(cudaMemcpy(rows.data(), (uint8_t *)raw + (size_t)first * k, rows.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(flat.data(), tree, flat.size(), cudaMemcpyDeviceToHost));
            uint32_t count = TILE * k / CHUNK, lead = first * k / CHUNK;
            std::vector<uint32_t> idx;
            for (uint32_t i = 0; i < count; i++) idx.push_back(lead + i);
            u64(count);
            for (uint32_t i = 0; i < count; i++) { u64(CHUNK); o.insert(o.end(), rows.begin() + (size_t)i * CHUNK, rows.begin() + (size_t)(i + 1) * CHUNK); }
            u64(count); for (uint32_t i : idx) u64(i);
            u64(t.lv[0].len);
            o.insert(o.end(), root, root + 32);
            siblings(flat, t, idx, sib);
            u64(sib.size() / 32); o.insert(o.end(), sib.begin(), sib.end());
            u64(TILE); for (int i = 0; i < TILE; i++) u64(first + i);
        };
        side(Araw, treeA, tA, rootA, tr * TILE);
        side(Braw, treeB, tB, rootB, tc * TILE);
        o.push_back(0); // moe: None
        return o;
    }
};

static std::string b64(const std::vector<uint8_t> &d) {
    static const char *T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string s; s.reserve((d.size() + 2) / 3 * 4);
    for (size_t i = 0; i < d.size(); i += 3) {
        uint32_t v = d[i] << 16 | (i + 1 < d.size() ? d[i + 1] << 8 : 0) | (i + 2 < d.size() ? d[i + 2] : 0);
        s += T[(v >> 18) & 63]; s += T[(v >> 12) & 63];
        s += i + 1 < d.size() ? T[(v >> 6) & 63] : '='; s += i + 2 < d.size() ? T[v & 63] : '=';
    }
    return s;
}
static bool unhex(const char *h, uint8_t *out, int n) {
    if ((int)strlen(h) != 2 * n) return false;
    for (int i = 0; i < n; i++) { unsigned v; if (sscanf(h + 2 * i, "%2x", &v) != 1) return false; out[i] = (uint8_t)v; }
    return true;
}
static std::string hex(const uint8_t *b, int n) { static const char *H = "0123456789abcdef"; std::string s; for (int i = 0; i < n; i++) { s += H[b[i] >> 4]; s += H[b[i] & 15]; } return s; }

int main() {
    setvbuf(stdout, NULL, _IOFBF, 1 << 20);
    memcpy(LABEL_A, "A_tensor", 8); memcpy(LABEL_B, "B_tensor", 8);
    // The largest square that fits comfortably: four matrices of size^2 x 2048 bytes.
    size_t freeb = 0, total = 0;
    if (cudaMemGetInfo(&freeb, &total) != cudaSuccess) { printf("error no CUDA device\n"); fflush(stdout); return 1; }
    int size = freeb > ((size_t)400 << 20) ? 8192 : freeb > ((size_t)120 << 20) ? 4096 : 2048;
    Ctx ctx;
    if (!ctx.alloc(size, size, false)) { printf("error could not allocate %d x %d\n", size, size); fflush(stdout); return 1; }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    const char *force = getenv("PEARL_KERNEL");
    use_tc = prop.major >= 8 && !(force && !strcmp(force, "dp4a"));
    for (char *p = prop.name; *p; p++) if (*p == ' ') *p = '_'; // one word on the ready line
    printf("ready %d %d %d %d %s %s sm_%d%d\n", ctx.m, ctx.n, K, RANK, use_tc ? "tensor" : "dp4a", prop.name, prop.major, prop.minor);
    fflush(stdout);

    std::random_device rd;
    std::mt19937_64 rng(((uint64_t)rd() << 32) ^ rd() ^ (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count());
    std::string jobId; uint8_t header[76]; int bits = 0; bool haveJob = false;
    static char line[1 << 16];
    while (fgets(line, sizeof line, stdin)) {
        char cmd[16] = {0}; sscanf(line, "%15s", cmd);
        if (!strcmp(cmd, "quit")) break;
        if (!strcmp(cmd, "job")) {
            char id[256], hh[200]; int b;
            if (sscanf(line, "job %255s %199s %d", id, hh, &b) == 3 && unhex(hh, header, 76) && b >= 0 && b < 256) { jobId = id; bits = b; haveJob = true; }
            else { printf("error bad job\n"); fflush(stdout); }
        } else if (!strcmp(cmd, "round")) {
            if (!haveJob) { printf("done 0 0\n"); fflush(stdout); continue; }
            auto t0 = std::chrono::steady_clock::now();
            uint32_t s0 = (uint32_t)rng(), s1 = (uint32_t)rng();
            uint32_t count = ctx.round(header, s0, s1, bits);
            std::vector<uint32_t> h(2 + 2 * std::min<uint32_t>(count, PROOFS_PER_ROUND));
            if (count) CK(cudaMemcpy(h.data(), ctx.hits, h.size() * 4, cudaMemcpyDeviceToHost));
            for (uint32_t i = 0; i < std::min<uint32_t>(count, PROOFS_PER_ROUND); i++)
                printf("share %s %s\n", jobId.c_str(), b64(ctx.proof(h[2 + 2 * i], h[3 + 2 * i])).c_str());
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            printf("done %u %.1f\n", (ctx.m / 16) * (ctx.n / 16), ms); fflush(stdout);
        } else if (!strcmp(cmd, "check")) {
            int cm, cn; unsigned s0, s1; char hh[200]; uint8_t hd[76];
            if (sscanf(line, "check %d %d %u %u %199s", &cm, &cn, &s0, &s1, hh) != 5 || !unhex(hh, hd, 76) || cm % 64 || cn % 64) { printf("error bad check\n"); fflush(stdout); continue; }
            Ctx c;
            if (!c.alloc(cm, cn, true)) { printf("error alloc\n"); fflush(stdout); continue; }
            c.round(hd, s0, s1, 0);
            printf("roots %s %s\n", hex(c.rootA, 32).c_str(), hex(c.rootB, 32).c_str());
            std::vector<uint32_t> d((size_t)(cm / 16) * (cn / 16) * 8);
            CK(cudaMemcpy(d.data(), c.dbg, d.size() * 4, cudaMemcpyDeviceToHost));
            for (int tr = 0; tr < cm / 16; tr++) for (int tc = 0; tc < cn / 16; tc++) {
                uint8_t b[32]; words_le(&d[((size_t)tr * (cn / 16) + tc) * 8], b, 8);
                printf("tile %d %d %s\n", tr, tc, hex(b, 32).c_str());
            }
            printf("proof 0 0 %s\n", b64(c.proof(0, 0)).c_str());
            printf("proof %d %d %s\n", cm / 16 - 1, cn / 16 - 1, b64(c.proof(cm / 16 - 1, cn / 16 - 1)).c_str());
            printf("checked\n"); fflush(stdout);
            c.release();
        }
    }
    ctx.release();
    return 0;
}

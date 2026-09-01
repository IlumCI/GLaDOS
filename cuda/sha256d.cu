// SHA-256d over a Bitcoin block header, on the GPU.
//
// Two modes, and the order matters: `--verify` prints the digest for a given
// nonce so it can be diffed against hashlib, and `--bench` measures. Nothing
// here is timed until the first mode agrees with an independent
// implementation, because a fast wrong hash is worth nothing and is very easy
// to write.
//
// The experiment this exists for is ILP width. NPT is nonces per thread: each
// thread carries NPT independent hash states and interleaves them, so the
// warp always has an eligible instruction instead of stalling on the message
// schedule's dependency chain. Published profiling of this workload class
// puts 47.6% of warp stalls in "no eligible" at 33% occupancy while the FMA
// pipe sits 88% idle, so the width sweep is the measurement that decides
// whether that headroom is reachable.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define Ch(e, f, g) (((e) & (f)) ^ (~(e) & (g)))
#define Maj(a, b, c) (((a) & (b)) ^ ((a) & (c)) ^ ((b) & (c)))
#define BS0(a) (ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22))
#define BS1(e) (ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25))
#define SS0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SS1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

__constant__ uint32_t cK[64];
__constant__ uint32_t cMid[8];    // state after the first 64 header bytes
__constant__ uint32_t cTail[3];   // header words 16,17,18 (bytes 64..76)
__constant__ uint32_t cTarget[8]; // big-endian word order, most significant first
__constant__ uint32_t cIV[8];     // the SHA-256 initial state, for the second hash

static const uint32_t hK[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};

static const uint32_t hIV[8] = {
    0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
    0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};

// One compression over a 16-word message, with a rolling schedule.
//
// A rolling window rather than a 64-word array, deliberately. The array is
// the textbook form and costs 64 live registers, which is exactly the
// pressure that caps occupancy at 33% -- and occupancy is what the ILP
// experiment is trying to buy back. This keeps 16.
__device__ __forceinline__ void compress(uint32_t st[8], uint32_t w[16]) {
    uint32_t a = st[0], b = st[1], c = st[2], d = st[3];
    uint32_t e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t wi;
        if (i < 16) {
            wi = w[i];
        } else {
            uint32_t s0 = SS0(w[(i + 1) & 15]);
            uint32_t s1 = SS1(w[(i + 14) & 15]);
            wi = w[i & 15] = w[i & 15] + s0 + w[(i + 9) & 15] + s1;
        }
        uint32_t t1 = h + BS1(e) + Ch(e, f, g) + cK[i] + wi;
        uint32_t t2 = BS0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d;
    st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

// The nonce-dependent half: finish the first hash, then hash its digest.
//
// Two compressions per nonce. The first 64 header bytes never change, so
// their compression is the host's midstate and never runs here -- that is the
// whole of the midstate optimisation and it halves the work.
__device__ __forceinline__ void sha256d_tail(uint32_t nonce, uint32_t out[8]) {
    uint32_t st[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) st[i] = cMid[i];

    uint32_t w[16];
    w[0] = cTail[0]; w[1] = cTail[1]; w[2] = cTail[2]; w[3] = nonce;
    w[4] = 0x80000000u;                     // padding: one bit, then zeros
#pragma unroll
    for (int i = 5; i < 15; ++i) w[i] = 0u;
    w[15] = 640u;                           // 80 bytes = 640 bits
    compress(st, w);

    // Second hash, over the 32-byte digest.
    uint32_t h2[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) h2[i] = cIV[i];
    uint32_t w2[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) w2[i] = st[i];
    w2[8] = 0x80000000u;
#pragma unroll
    for (int i = 9; i < 15; ++i) w2[i] = 0u;
    w2[15] = 256u;                          // 32 bytes = 256 bits
    compress(h2, w2);

#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = h2[i];
}

// A real 256-bit comparison, not a leading-zero proxy.
//
// Bitcoin reads the digest as a little-endian 256-bit integer, so the last
// word of the digest is the most significant. `out[7]` byte-swapped is
// therefore what to compare first. A leading-zero count is the usual stand-in
// and it answers a different question.
__device__ __forceinline__ bool below_target(const uint32_t h[8]) {
#pragma unroll
    for (int i = 7; i >= 0; --i) {
        uint32_t v = __byte_perm(h[i], 0, 0x0123);  // to big-endian order
        if (v < cTarget[7 - i]) return true;
        if (v > cTarget[7 - i]) return false;
    }
    return true;  // exactly equal counts as below
}

__global__ void verify_kernel(uint32_t nonce, uint32_t *out) {
    uint32_t h[8];
    sha256d_tail(nonce, h);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = h[i];
}

// NPT independent nonces per thread, interleaved.
//
// The states are separate arrays rather than a loop over one state, so the
// compiler can schedule NPT independent dependency chains against each other.
// A loop would serialise them and measure nothing.
template <int NPT>
__global__ void mine_kernel(uint32_t base, uint32_t *found) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n0 = base + idx * NPT;
    uint32_t h[NPT][8];
#pragma unroll
    for (int j = 0; j < NPT; ++j) sha256d_tail(n0 + j, h[j]);
#pragma unroll
    for (int j = 0; j < NPT; ++j) {
        if (below_target(h[j])) {
            atomicCAS(found, 0xffffffffu, n0 + j);
        }
    }
}

// ---- host ---------------------------------------------------------------

static uint32_t be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

// Host compression, used only to build the midstate. Same algorithm, written
// separately on purpose: if the device version and this one disagree the
// verification catches it, which a shared implementation could not.
static void host_compress(uint32_t st[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = be32(block + 4 * i);
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = ROTR(w[i-15],7) ^ ROTR(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROTR(w[i-2],17) ^ ROTR(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + BS1(e) + Ch(e,f,g) + hK[i] + w[i];
        uint32_t t2 = BS0(a) + Maj(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d;
    st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda: %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

// Bitcoin block 125552, the canonical mining test vector. Fields are stored
// little-endian in the header, which is why the nonce word the kernel is
// handed is not the nonce as usually quoted -- see the note in main.
static const uint8_t HEADER[80] = {
    0x01,0x00,0x00,0x00,
    0x81,0xcd,0x02,0xab,0x7e,0x56,0x9e,0x8b,0xcd,0x93,0x17,0xe2,
    0xfe,0x99,0xf2,0xde,0x44,0xd4,0x9a,0xb2,0xb8,0x85,0x1b,0xa4,
    0xa3,0x08,0x00,0x00,0x00,0x00,0x00,0x00,
    0xe3,0x20,0xb6,0xc2,0xff,0xfc,0x8d,0x75,0x04,0x23,0xdb,0x8b,
    0x1e,0xb9,0x42,0xae,0x71,0x0e,0x95,0x1e,0xd7,0x97,0xf7,0xaf,
    0xfc,0x88,0x92,0xb0,0xf1,0xfc,0x12,0x2b,
    0xc7,0xf5,0xd7,0x4d,
    0xf2,0xb9,0x44,0x1a,
    0x42,0xa1,0x46,0x95};

int main(int argc, char **argv) {
    bool bench = false;
    int npt = 1, blocks = 4096, threads = 256, iters = 20;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--bench")) bench = true;
        else if (!strcmp(argv[i], "--npt") && i + 1 < argc) npt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    }

    uint32_t mid[8];
    memcpy(mid, hIV, sizeof mid);
    host_compress(mid, HEADER);                 // first 64 bytes, nonce-free

    uint32_t tail[3] = { be32(HEADER + 64), be32(HEADER + 68), be32(HEADER + 72) };
    uint32_t nonce_word = be32(HEADER + 76);

    // Everything passes, so the bench measures the hash rate rather than the
    // difficulty. A real target goes here when this is wired to a pool.
    uint32_t target[8]; for (int i = 0; i < 8; ++i) target[i] = 0xffffffffu;

    CK(cudaMemcpyToSymbol(cK, hK, sizeof hK));
    CK(cudaMemcpyToSymbol(cIV, hIV, sizeof hIV));
    CK(cudaMemcpyToSymbol(cMid, mid, sizeof mid));
    CK(cudaMemcpyToSymbol(cTail, tail, sizeof tail));
    CK(cudaMemcpyToSymbol(cTarget, target, sizeof target));

    if (!bench) {
        uint32_t *d_out; CK(cudaMalloc(&d_out, 32));
        verify_kernel<<<1,1>>>(nonce_word, d_out);
        CK(cudaDeviceSynchronize());
        uint32_t h[8]; CK(cudaMemcpy(h, d_out, 32, cudaMemcpyDeviceToHost));
        printf("nonce word 0x%08x\ndigest ", nonce_word);
        for (int i = 0; i < 8; ++i)
            for (int b = 3; b >= 0; --b) printf("%02x", (h[i] >> (8*b)) & 0xff);
        printf("\n");
        cudaFree(d_out);
        return 0;
    }

    uint32_t *d_found; CK(cudaMalloc(&d_found, 4));
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

    // One warm-up launch outside the timing. The first launch of a kernel
    // pays for module load and JIT, which is not what is being measured.
    switch (npt) {
        case 1: mine_kernel<1><<<blocks,threads>>>(0, d_found); break;
        case 2: mine_kernel<2><<<blocks,threads>>>(0, d_found); break;
        case 4: mine_kernel<4><<<blocks,threads>>>(0, d_found); break;
        case 8: mine_kernel<8><<<blocks,threads>>>(0, d_found); break;
        default: fprintf(stderr, "npt must be 1, 2, 4 or 8\n"); return 1;
    }
    CK(cudaDeviceSynchronize());

    CK(cudaEventRecord(t0));
    for (int it = 0; it < iters; ++it) {
        uint32_t base = (uint32_t)it * (uint32_t)blocks * threads * npt;
        switch (npt) {
            case 1: mine_kernel<1><<<blocks,threads>>>(base, d_found); break;
            case 2: mine_kernel<2><<<blocks,threads>>>(base, d_found); break;
            case 4: mine_kernel<4><<<blocks,threads>>>(base, d_found); break;
            case 8: mine_kernel<8><<<blocks,threads>>>(base, d_found); break;
        }
    }
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));

    double hashes = (double)blocks * threads * npt * iters;
    printf("npt %d  blocks %d  threads %d  iters %d\n", npt, blocks, threads, iters);
    printf("  %.0f hashes in %.1f ms = %.3f GH/s\n", hashes, ms, hashes / (ms * 1e6));
    cudaFree(d_found);
    return 0;
}

// HeavyHash as Optical Bitcoin defines it, for `xpu.cu`.
//
// `tools/heavyhash.py` is the oracle and names where each step comes from in
// PoWx-Org/obtc-core; `src/mine/heavyhash.rs` is the kernel's and the pool's.
// This is the third implementation and it is checked the way the other two
// are: OBTC's mainnet genesis block, whose nonce a scan must find.
//
//     h1  = SHA3-256(header)
//     p   = (M * nibbles(h1)) >> 10          64 rows, each < 16
//     pow = SHA3-256(h1 XOR pack(p))
//
// **The matrix step is `kheavy.cu`'s dp4a path**, which measured 371 Mstep/s
// against 383 for tensor cores and needs no warp-wide batching. Each row is
// sixteen words of four int8 nibbles, and so is the vector, so a row is
// sixteen `__dp4a`s. The matrix belongs to the previous block and not to the
// nonce, so it is built once per job on the host and lives in constant memory.
//
// Keccak-f[1600] is written once and compiled for both sides: the host needs
// SHA3-256 to seed the matrix, and a second copy for the host is a second
// place for a rotation constant to be wrong.
#pragma once
#include <stdint.h>

#define HH_HD __host__ __device__ __forceinline__

__constant__ uint64_t hhRC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
static const uint64_t hhRC_host[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};

// **Funnel shifts on the device.** The GPU is a 32-bit machine, so a 64-bit
// rotate written as two shifts and an OR is four shifts and two ORs over the
// halves; `__funnelshift_l` does a half in one instruction. Keccak is all
// rotates and XORs, and it measured as the whole cost here -- one permutation
// alone ran at 358 MH/s, two at 180, and the matrix step was 12%.
HH_HD uint64_t hh_rotl(uint64_t x, int n) {
#ifdef __CUDA_ARCH__
    const uint32_t lo = (uint32_t)x, hi = (uint32_t)(x >> 32);
    uint32_t rlo, rhi;
    if (n < 32) {
        rhi = __funnelshift_l(lo, hi, n);
        rlo = __funnelshift_l(hi, lo, n);
    } else {
        rhi = __funnelshift_l(hi, lo, n - 32);
        rlo = __funnelshift_l(lo, hi, n - 32);
    }
    return ((uint64_t)rhi << 32) | rlo;
#else
    return (x << n) | (x >> (64 - n));
#endif
}

HH_HD void hh_keccakf(uint64_t a[25]) {
    // Generated: theta, rho and pi, chi and iota with every index a literal, so
    // the state lives in registers. A version walking rho/pi through small
    // lookup tables measured slower -- nvcc does not fold them, and a
    // runtime-indexed state array goes to local memory.
#pragma unroll
    for (int r = 0; r < 24; ++r) {
        uint64_t c0 = a[0]^a[5]^a[10]^a[15]^a[20], c1 = a[1]^a[6]^a[11]^a[16]^a[21],
                 c2 = a[2]^a[7]^a[12]^a[17]^a[22], c3 = a[3]^a[8]^a[13]^a[18]^a[23],
                 c4 = a[4]^a[9]^a[14]^a[19]^a[24];
        uint64_t d0 = c4 ^ hh_rotl(c1, 1), d1 = c0 ^ hh_rotl(c2, 1), d2 = c1 ^ hh_rotl(c3, 1),
                 d3 = c2 ^ hh_rotl(c4, 1), d4 = c3 ^ hh_rotl(c0, 1);
        a[0] ^= d0;
        a[1] ^= d1;
        a[2] ^= d2;
        a[3] ^= d3;
        a[4] ^= d4;
        a[5] ^= d0;
        a[6] ^= d1;
        a[7] ^= d2;
        a[8] ^= d3;
        a[9] ^= d4;
        a[10] ^= d0;
        a[11] ^= d1;
        a[12] ^= d2;
        a[13] ^= d3;
        a[14] ^= d4;
        a[15] ^= d0;
        a[16] ^= d1;
        a[17] ^= d2;
        a[18] ^= d3;
        a[19] ^= d4;
        a[20] ^= d0;
        a[21] ^= d1;
        a[22] ^= d2;
        a[23] ^= d3;
        a[24] ^= d4;
        uint64_t t = a[1], u;
        u = a[10]; a[10] = hh_rotl(t, 1); t = u;
        u = a[7]; a[7] = hh_rotl(t, 3); t = u;
        u = a[11]; a[11] = hh_rotl(t, 6); t = u;
        u = a[17]; a[17] = hh_rotl(t, 10); t = u;
        u = a[18]; a[18] = hh_rotl(t, 15); t = u;
        u = a[3]; a[3] = hh_rotl(t, 21); t = u;
        u = a[5]; a[5] = hh_rotl(t, 28); t = u;
        u = a[16]; a[16] = hh_rotl(t, 36); t = u;
        u = a[8]; a[8] = hh_rotl(t, 45); t = u;
        u = a[21]; a[21] = hh_rotl(t, 55); t = u;
        u = a[24]; a[24] = hh_rotl(t, 2); t = u;
        u = a[4]; a[4] = hh_rotl(t, 14); t = u;
        u = a[15]; a[15] = hh_rotl(t, 27); t = u;
        u = a[23]; a[23] = hh_rotl(t, 41); t = u;
        u = a[19]; a[19] = hh_rotl(t, 56); t = u;
        u = a[13]; a[13] = hh_rotl(t, 8); t = u;
        u = a[12]; a[12] = hh_rotl(t, 25); t = u;
        u = a[2]; a[2] = hh_rotl(t, 43); t = u;
        u = a[20]; a[20] = hh_rotl(t, 62); t = u;
        u = a[14]; a[14] = hh_rotl(t, 18); t = u;
        u = a[22]; a[22] = hh_rotl(t, 39); t = u;
        u = a[9]; a[9] = hh_rotl(t, 61); t = u;
        u = a[6]; a[6] = hh_rotl(t, 20); t = u;
        u = a[1]; a[1] = hh_rotl(t, 44); t = u;
        { uint64_t b0=a[0],b1=a[1],b2=a[2],b3=a[3],b4=a[4];
          a[0]=b0^(~b1&b2); a[1]=b1^(~b2&b3); a[2]=b2^(~b3&b4); a[3]=b3^(~b4&b0); a[4]=b4^(~b0&b1); }
        { uint64_t b0=a[5],b1=a[6],b2=a[7],b3=a[8],b4=a[9];
          a[5]=b0^(~b1&b2); a[6]=b1^(~b2&b3); a[7]=b2^(~b3&b4); a[8]=b3^(~b4&b0); a[9]=b4^(~b0&b1); }
        { uint64_t b0=a[10],b1=a[11],b2=a[12],b3=a[13],b4=a[14];
          a[10]=b0^(~b1&b2); a[11]=b1^(~b2&b3); a[12]=b2^(~b3&b4); a[13]=b3^(~b4&b0); a[14]=b4^(~b0&b1); }
        { uint64_t b0=a[15],b1=a[16],b2=a[17],b3=a[18],b4=a[19];
          a[15]=b0^(~b1&b2); a[16]=b1^(~b2&b3); a[17]=b2^(~b3&b4); a[18]=b3^(~b4&b0); a[19]=b4^(~b0&b1); }
        { uint64_t b0=a[20],b1=a[21],b2=a[22],b3=a[23],b4=a[24];
          a[20]=b0^(~b1&b2); a[21]=b1^(~b2&b3); a[22]=b2^(~b3&b4); a[23]=b3^(~b4&b0); a[24]=b4^(~b0&b1); }
#ifdef __CUDA_ARCH__
        a[0] ^= hhRC[r];
#else
        a[0] ^= hhRC_host[r];
#endif
    }
}

// SHA3-256 of a message shorter than one rate (136 bytes), which every input
// here is: 80 bytes of header, 32 of digest.
HH_HD void hh_sha3_short(const uint8_t *msg, int len, uint64_t out[4]) {
    uint64_t a[25];
#pragma unroll
    for (int i = 0; i < 25; ++i) a[i] = 0;
    for (int i = 0; i < len; ++i) a[i >> 3] ^= (uint64_t)msg[i] << (8 * (i & 7));
    a[len >> 3] ^= (uint64_t)0x06 << (8 * (len & 7));
    a[16] ^= 0x8000000000000000ULL;  // byte 135, the end of the rate
    hh_keccakf(a);
    out[0] = a[0]; out[1] = a[1]; out[2] = a[2]; out[3] = a[3];
}

// --- host: the matrix ----------------------------------------------------------

static uint64_t hh_xoshiro(uint64_t s[4]) {
    const uint64_t result = hh_rotl(s[0] + s[3], 23) + s[0];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = hh_rotl(s[3], 45);
    return result;
}

// Full rank modulo a prime: rank mod p can only be lower than over Q, so a yes
// here is a yes. `heavyhash.rs` argues the two-prime fallback.
static bool hh_full_rank_mod(const uint8_t m[64][64], uint64_t p) {
    static uint64_t a[64][64];
    for (int i = 0; i < 64; ++i)
        for (int j = 0; j < 64; ++j) a[i][j] = m[i][j] % p;
    for (int c = 0; c < 64; ++c) {
        int piv = -1;
        for (int r = c; r < 64; ++r) if (a[r][c]) { piv = r; break; }
        if (piv < 0) return false;
        for (int k = 0; k < 64; ++k) { uint64_t t = a[c][k]; a[c][k] = a[piv][k]; a[piv][k] = t; }
        uint64_t inv = 1, b = a[c][c], e = p - 2;
        while (e) { if (e & 1) inv = inv * b % p; b = b * b % p; e >>= 1; }
        for (int r = c + 1; r < 64; ++r) {
            if (!a[r][c]) continue;
            const uint64_t f = a[r][c] * inv % p;
            for (int k = c; k < 64; ++k) a[r][k] = (a[r][k] + p - f * a[c][k] % p) % p;
        }
    }
    return true;
}

// The matrix for `prev` (header bytes 4..36), as in `GenerateHeavyHashMatrix`.
static void hh_matrix(const uint8_t prev[32], uint8_t m[64][64]) {
    uint64_t seed[4];
    hh_sha3_short(prev, 32, seed);
    uint64_t s[4] = {seed[0], seed[1], seed[2], seed[3]};
    do {
        for (int i = 0; i < 64; ++i)
            for (int j = 0; j < 64; j += 16) {
                const uint64_t v = hh_xoshiro(s);
                for (int k = 0; k < 16; ++k) m[i][j + k] = (uint8_t)((v >> (4 * k)) & 0xF);
            }
    } while (!hh_full_rank_mod(m, 2147483647ULL) && !hh_full_rank_mod(m, 2147483629ULL));
}

// --- device: one nonce -----------------------------------------------------------
//
// **Lanes, never bytes.** The first version held the header, the digest and
// the XOR in `uint8_t` arrays and indexed them in loops; nvcc keeps such arrays
// in local memory, which is DRAM behind a cache, and it measured 174 MH/s. Every
// quantity here is instead a 64-bit Keccak lane or a 32-bit dp4a word with
// constant indices, so all of it stays in registers.
//
// The header is constant except for the nonce, so its sponge input is too:
// lanes 0..8 are header bytes 0..72, lane 9 is bytes 72..80 -- `nBits` low and
// the nonce high -- and the SHA3 padding is two constant bits. The host builds
// all 25 lanes once a job; a nonce only rewrites lane 9's high half.

__constant__ int32_t hhM[64][16];   // each row, four int8 nibbles a word
__constant__ uint64_t hhLanes[25];  // the first sponge's input, nonce zero

__device__ __forceinline__ uint32_t hh_vec(uint64_t lane, int shift) {
    // Two digest bytes -> four nibbles, high nibble of each byte first, as dp4a
    // bytes in the order the matrix rows are packed.
    const uint32_t x = (uint32_t)(lane >> shift) & 0xFFFFu;
    return ((x >> 4) & 0xFu) | ((x & 0xFu) << 8) | (((x >> 12) & 0xFu) << 16) | (((x >> 8) & 0xFu) << 24);
}

__device__ __forceinline__ void heavyhash_nonce(uint32_t nonce, uint32_t out[8]) {
    uint64_t a[25];
#pragma unroll
    for (int i = 0; i < 25; ++i) a[i] = hhLanes[i];
    a[9] |= (uint64_t)nonce << 32;
    hh_keccakf(a);
    const uint64_t d0 = a[0], d1 = a[1], d2 = a[2], d3 = a[3];

    int32_t v[16];
#pragma unroll
    for (int w = 0; w < 16; ++w) {
        const uint64_t lane = (w < 4) ? d0 : (w < 8) ? d1 : (w < 12) ? d2 : d3;
        v[w] = (int32_t)hh_vec(lane, 16 * (w & 3));
    }
    uint64_t x[4] = {0, 0, 0, 0};
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        int p0 = 0, p1 = 0;
#ifndef HH_PROBE_NOMAT
#pragma unroll
        for (int w = 0; w < 16; ++w) {
            p0 = __dp4a(hhM[2 * k][w], v[w], p0);
            p1 = __dp4a(hhM[2 * k + 1][w], v[w], p1);
        }
#else
        p0 = v[k & 15]; p1 = v[(k + 1) & 15];
#endif
        x[k >> 3] |= (uint64_t)(((p0 >> 10) << 4) | (p1 >> 10)) << (8 * (k & 7));
    }

#pragma unroll
    for (int i = 0; i < 25; ++i) a[i] = 0;
    a[0] = x[0] ^ d0; a[1] = x[1] ^ d1; a[2] = x[2] ^ d2; a[3] = x[3] ^ d3;
    a[4] = 0x06;                     // byte 32: the SHA3 domain bits
    a[16] = 0x8000000000000000ULL;   // byte 135: the end of the rate
#ifndef HH_PROBE_NOKECCAK2
    hh_keccakf(a);
#endif
    // Little-endian words, which is what `below_target_le` compares.
#pragma unroll
    for (int i = 0; i < 4; ++i) { out[2 * i] = (uint32_t)a[i]; out[2 * i + 1] = (uint32_t)(a[i] >> 32); }
}

// Upload a job: the sponge's constant input, and the matrix for the previous block.
static int heavyhash_upload(const uint8_t header[80]) {
    static uint8_t m[64][64];
    hh_matrix(header + 4, m);
    int32_t packed[64][16];
    for (int i = 0; i < 64; ++i)
        for (int w = 0; w < 16; ++w)
            packed[i][w] = (int32_t)((uint32_t)m[i][4 * w] | ((uint32_t)m[i][4 * w + 1] << 8) |
                                     ((uint32_t)m[i][4 * w + 2] << 16) | ((uint32_t)m[i][4 * w + 3] << 24));
    uint64_t lanes[25] = {0};
    for (int i = 0; i < 76; ++i) lanes[i >> 3] |= (uint64_t)header[i] << (8 * (i & 7));
    lanes[10] = 0x06;                  // byte 80
    lanes[16] = 0x8000000000000000ULL; // byte 135
    if (cudaMemcpyToSymbol(hhM, packed, sizeof packed) != cudaSuccess) return -1;
    if (cudaMemcpyToSymbol(hhLanes, lanes, sizeof lanes) != cudaSuccess) return -1;
    return 0;
}

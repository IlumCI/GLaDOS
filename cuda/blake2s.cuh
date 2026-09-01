// BLAKE2s-256 over a 64-byte input, RFC 7693.
//
// The second algorithm, and it exists to prove the abstraction is real rather
// than a shape drawn around one implementation. It is also a genuine data
// point for the survey the plan calls Stage 1b: BLAKE2s is ten rounds of pure
// add-rotate-xor against SHA-256's sixty-four, so if instruction count is what
// bounds this hardware it should be several times faster, and if it is not
// then something other than instruction count is in the way.
//
// Chosen over BLAKE-256r14, which several coins actually use, for one reason:
// `hashlib` has BLAKE2s and does not have BLAKE-256. An algorithm with no
// independent oracle cannot be verified, and this file would then be asserting
// its own correctness -- which is the arrangement `algo.cuh` exists to forbid.
//
// One block, final, unkeyed, 32-byte digest. A real miner hashes an 80-byte
// header and needs two blocks; this hashes 64 bytes with the nonce at offset
// 60 so it stays a *checkable* BLAKE2s. The throughput question does not care,
// because the second block is the same compression again.
//
// **BLAKE2s reads its message little-endian.** SHA-256 reads big-endian. Get
// this backwards and the digest is still 32 plausible bytes at full speed.

#pragma once
#include "algo.cuh"

__constant__ uint32_t b2s_h0[8];   // IV with the parameter block folded in
__constant__ uint32_t b2s_msg[16]; // the block, nonce word overwritten per hash

__device__ __constant__ uint8_t b2s_sigma[10][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
    {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4},
    { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8},
    { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13},
    { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9},
    {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11},
    {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10},
    { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0}};

__device__ __constant__ uint32_t b2s_iv[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

#define B2S_G(a, b, c, d, x, y)          \
    a = a + b + (x);                     \
    d = ROTR32(d ^ a, 16);               \
    c = c + d;                           \
    b = ROTR32(b ^ c, 12);               \
    a = a + b + (y);                     \
    d = ROTR32(d ^ a, 8);                \
    c = c + d;                           \
    b = ROTR32(b ^ c, 7);

__device__ __forceinline__ void blake2s_hash(uint32_t nonce, uint32_t out[8]) {
    uint32_t m[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = b2s_msg[i];
    m[15] = nonce;   // little-endian word 15 is bytes 60..64

    uint32_t v[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = b2s_h0[i];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[8 + i] = b2s_iv[i];
    v[12] ^= 64u;          // counter: 64 bytes consumed
    v[14] ^= 0xffffffffu;  // final block

#pragma unroll
    for (int r = 0; r < 10; ++r) {
        const uint8_t *s = b2s_sigma[r];
        B2S_G(v[0], v[4], v[ 8], v[12], m[s[ 0]], m[s[ 1]]);
        B2S_G(v[1], v[5], v[ 9], v[13], m[s[ 2]], m[s[ 3]]);
        B2S_G(v[2], v[6], v[10], v[14], m[s[ 4]], m[s[ 5]]);
        B2S_G(v[3], v[7], v[11], v[15], m[s[ 6]], m[s[ 7]]);
        B2S_G(v[0], v[5], v[10], v[15], m[s[ 8]], m[s[ 9]]);
        B2S_G(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        B2S_G(v[2], v[7], v[ 8], v[13], m[s[12]], m[s[13]]);
        B2S_G(v[3], v[4], v[ 9], v[14], m[s[14]], m[s[15]]);
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = b2s_h0[i] ^ v[i] ^ v[8 + i];
}

// ---- host side ----------------------------------------------------------

// A fixed 64-byte input whose last word the nonce replaces. Bytes 0..60 are
// "GLaDOS blake2s test vector padding.........." so the expected digest below
// is checkable by anybody with Python.
static void blake2s_upload(void) {
    uint8_t blk[64];
    static const char *seed = "GLaDOS blake2s vector 000000000000000000000000000000000000000000000000000000000000";
    for (int i = 0; i < 64; ++i) blk[i] = (uint8_t)seed[i];

    uint32_t msg[16];
    for (int i = 0; i < 16; ++i)      // little-endian, unlike SHA-256
        msg[i] = (uint32_t)blk[4*i] | ((uint32_t)blk[4*i+1] << 8) |
                 ((uint32_t)blk[4*i+2] << 16) | ((uint32_t)blk[4*i+3] << 24);

    // The parameter block: 0x01010000 ^ (keylen << 8) ^ digest_len, unkeyed,
    // 32 bytes out.
    uint32_t h0[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                      0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    h0[0] ^= 0x01010020u;

    cudaMemcpyToSymbol(b2s_h0, h0, sizeof h0);
    cudaMemcpyToSymbol(b2s_msg, msg, sizeof msg);
}

// Filled in from tools/algocheck.py, which computes it with hashlib.
static const char *blake2s_expect(void) { return "3e0658bef42701b1ad03fd5805c0e04d3bd53317312e850675742b1611804d83"; }

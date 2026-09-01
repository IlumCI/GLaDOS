// SHA-256d over a Bitcoin block header.
//
// Two compressions per nonce. The first 64 header bytes never change, so
// their compression is the host's midstate and never runs on the device --
// that is the whole midstate optimisation and it halves the work.
//
// SHA-256 reads its message as big-endian 32-bit words. BLAKE2s does not.

#pragma once
#include "algo.cuh"

#define Ch(e, f, g) (((e) & (f)) ^ (~(e) & (g)))
#define Maj(a, b, c) (((a) & (b)) ^ ((a) & (c)) ^ ((b) & (c)))
#define BS0(a) (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22))
#define BS1(e) (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25))
#define SS0(x) (ROTR32(x, 7) ^ ROTR32(x, 18) ^ ((x) >> 3))
#define SS1(x) (ROTR32(x, 17) ^ ROTR32(x, 19) ^ ((x) >> 10))

__constant__ uint32_t cK[64];
__constant__ uint32_t cMid[8];
__constant__ uint32_t cTail[3];
__constant__ uint32_t cIV[8];

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

// A rolling 16-word schedule rather than the textbook 64-word array. The
// array costs 64 live registers; this costs 16, and the register count is
// what decides occupancy -- measured at 40 registers and ~100% occupancy,
// which is why the ILP experiment found nothing left to buy.
__device__ __forceinline__ void sha_compress(uint32_t st[8], uint32_t w[16]) {
    uint32_t a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
#pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t wi;
        if (i < 16) wi = w[i];
        else {
            uint32_t s0 = SS0(w[(i+1)&15]), s1 = SS1(w[(i+14)&15]);
            wi = w[i&15] = w[i&15] + s0 + w[(i+9)&15] + s1;
        }
        uint32_t t1 = h + BS1(e) + Ch(e,f,g) + cK[i] + wi;
        uint32_t t2 = BS0(a) + Maj(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d;
    st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}

__device__ __forceinline__ void sha256d_hash(uint32_t nonce, uint32_t out[8]) {
    uint32_t st[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) st[i] = cMid[i];
    uint32_t w[16];
    w[0]=cTail[0]; w[1]=cTail[1]; w[2]=cTail[2]; w[3]=nonce;
    w[4]=0x80000000u;
#pragma unroll
    for (int i = 5; i < 15; ++i) w[i] = 0u;
    w[15] = 640u;                       // 80 bytes
    sha_compress(st, w);

    uint32_t h2[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) h2[i] = cIV[i];
    uint32_t w2[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) w2[i] = st[i];
    w2[8]=0x80000000u;
#pragma unroll
    for (int i = 9; i < 15; ++i) w2[i] = 0u;
    w2[15] = 256u;                      // 32 bytes
    sha_compress(h2, w2);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = h2[i];
}

// ---- host side ----------------------------------------------------------

// Bitcoin block 125552, the canonical mining test vector.
static const uint8_t SHA_HEADER[80] = {
    0x01,0x00,0x00,0x00,
    0x81,0xcd,0x02,0xab,0x7e,0x56,0x9e,0x8b,0xcd,0x93,0x17,0xe2,
    0xfe,0x99,0xf2,0xde,0x44,0xd4,0x9a,0xb2,0xb8,0x85,0x1b,0xa4,
    0xa3,0x08,0x00,0x00,0x00,0x00,0x00,0x00,
    0xe3,0x20,0xb6,0xc2,0xff,0xfc,0x8d,0x75,0x04,0x23,0xdb,0x8b,
    0x1e,0xb9,0x42,0xae,0x71,0x0e,0x95,0x1e,0xd7,0x97,0xf7,0xaf,
    0xfc,0x88,0x92,0xb0,0xf1,0xfc,0x12,0x2b,
    0xc7,0xf5,0xd7,0x4d, 0xf2,0xb9,0x44,0x1a, 0x42,0xa1,0x46,0x95};

static uint32_t sha_be32(const uint8_t *p) {
    return ((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|(uint32_t)p[3];
}

// Written separately from the device version on purpose. If the two disagree
// the verification catches it, which a shared implementation could not.
static void sha_host_compress(uint32_t st[8], const uint8_t blk[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = sha_be32(blk + 4*i);
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = ROTR32(w[i-15],7)^ROTR32(w[i-15],18)^(w[i-15]>>3);
        uint32_t s1 = ROTR32(w[i-2],17)^ROTR32(w[i-2],19)^(w[i-2]>>10);
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

static uint32_t sha256d_nonce(void) { return sha_be32(SHA_HEADER + 76); }

static void sha256d_upload(void) {
    uint32_t mid[8];
    memcpy(mid, hIV, sizeof mid);
    sha_host_compress(mid, SHA_HEADER);
    uint32_t tail[3] = { sha_be32(SHA_HEADER+64), sha_be32(SHA_HEADER+68), sha_be32(SHA_HEADER+72) };
    cudaMemcpyToSymbol(cK, hK, sizeof hK);
    cudaMemcpyToSymbol(cIV, hIV, sizeof hIV);
    cudaMemcpyToSymbol(cMid, mid, sizeof mid);
    cudaMemcpyToSymbol(cTail, tail, sizeof tail);
}

static const char *sha256d_expect(void) {
    return "1dbd981fe6985776b644b173a4d0385ddc1aa2a829688d1e0000000000000000";
}

// BLAKE2b-512, on the device, with the parameter block Equihash needs.
//
// **This is not `blake2s.cuh` with wider words.** The two share a shape and
// nothing else: 64-bit state against 32-bit, rotations 32/24/16/63 against
// 16/12/8/7, twelve rounds against ten, a 128-byte block against 64, and
// roughly double the register pressure. `design/equihash.md` says so as hazard
// 6 and it is right -- nothing was reusable, so nothing was reused.
//
// ### Why the parameter block is here rather than hardcoded
//
// An ordinary BLAKE2b implementation folds `digest_length | key<<8 |
// fanout<<16 | depth<<24` into `h[0]` and stops, because nobody uses the rest.
// Equihash uses two more fields and both are load-bearing:
//
//   - **digest length is not 64.** Equihash(n,k) hashes to `(512/n)*(n/8)`
//     bytes, which for 192,7 is 48, and the length is mixed into `h[0]` before
//     a single byte of message is absorbed. Producing 48 bytes by truncating a
//     64-byte digest gives a different answer, silently.
//
//   - **personalisation is per coin.** The 16 bytes at offset 48 of the
//     parameter block are XORed into `h[6]` and `h[7]`. Zcash uses
//     `"ZcashPoW" || le32(n) || le32(k)`; other chains on the same algorithm
//     use their own string. `design/equihash.md` lists getting this wrong as
//     the hazard that bites first, and what it looks like is every share
//     rejected on some coins and accepted on others, with no diagnostic
//     anywhere -- the solver is correct, the solutions are structurally
//     perfect, and they are solutions to a different puzzle.
//
// So the personalisation is an argument, never a compile-time constant.
//
// ### Checked against something nobody here wrote
//
// `algo.cuh` makes the argument at length and it applies with more force to a
// hash whose output is 48 bytes and whose parameters are per coin: a fast
// wrong hash is indistinguishable from a fast right one from the inside.
// `tools/blake2b.py` holds the same vectors under `hashlib.blake2b`, which
// supports `digest_size` and `person` natively, so the oracle is CPython's C
// implementation rather than a second copy of this.
#pragma once
#include <stdint.h>

#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))

__device__ __constant__ static const uint64_t B2B_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
};

// Twelve rows. Rounds 10 and 11 repeat rows 0 and 1, which is BLAKE2b's own
// schedule and not a truncation of BLAKE2s's ten.
__device__ __constant__ static const uint8_t B2B_SIGMA[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
    {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4},
    { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8},
    { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13},
    { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9},
    {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11},
    {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10},
    { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0},
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
};

struct Blake2b {
    uint64_t h[8];
    uint64_t t;          // bytes absorbed, counting the buffer's contents
    uint8_t buf[128];
    uint32_t fill;       // bytes currently in buf
    uint32_t outlen;
};

#define B2B_G(a, b, c, d, x, y)                                                \
    do {                                                                       \
        a = a + b + (x);                                                       \
        d = ROTR64(d ^ a, 32);                                                 \
        c = c + d;                                                             \
        b = ROTR64(b ^ c, 24);                                                 \
        a = a + b + (y);                                                       \
        d = ROTR64(d ^ a, 16);                                                 \
        c = c + d;                                                             \
        b = ROTR64(b ^ c, 63);                                                 \
    } while (0)

/// Compress one 128-byte block. `last` sets the finalisation flag, which is
/// what stops a message being extendable and is the one bit whose absence
/// produces a digest that looks perfectly good.
__device__ static void b2b_compress(Blake2b *s, const uint8_t *block, bool last)
{
    uint64_t v[16], m[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        // Little-endian, explicitly. The trap `algo.cuh` names for BLAKE2s is
        // the same trap one word wider: a byte order read off the host's
        // preference produces a plausible digest of the right length.
        uint64_t w = 0;
#pragma unroll
        for (int b = 0; b < 8; b++) {
            w |= (uint64_t)block[i * 8 + b] << (8 * b);
        }
        m[i] = w;
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        v[i] = s->h[i];
        v[i + 8] = B2B_IV[i];
    }
    v[12] ^= s->t;
    // v[13] ^= high 64 bits of the counter, which no input here reaches.
    if (last) {
        v[14] = ~v[14];
    }
#pragma unroll
    for (int r = 0; r < 12; r++) {
        const uint8_t *sg = B2B_SIGMA[r];
        B2B_G(v[0], v[4], v[8], v[12], m[sg[0]], m[sg[1]]);
        B2B_G(v[1], v[5], v[9], v[13], m[sg[2]], m[sg[3]]);
        B2B_G(v[2], v[6], v[10], v[14], m[sg[4]], m[sg[5]]);
        B2B_G(v[3], v[7], v[11], v[15], m[sg[6]], m[sg[7]]);
        B2B_G(v[0], v[5], v[10], v[15], m[sg[8]], m[sg[9]]);
        B2B_G(v[1], v[6], v[11], v[12], m[sg[10]], m[sg[11]]);
        B2B_G(v[2], v[7], v[8], v[13], m[sg[12]], m[sg[13]]);
        B2B_G(v[3], v[4], v[9], v[14], m[sg[14]], m[sg[15]]);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        s->h[i] ^= v[i] ^ v[i + 8];
    }
}

/// Start a hash. `outlen` is in bytes and may be anything from 1 to 64;
/// `person` is 16 bytes or null.
///
/// **`outlen` enters the state before the message does.** That is what makes a
/// 48-byte BLAKE2b a different function from a truncated 64-byte one, and it
/// is why Equihash cannot be served by a stock BLAKE2b-512 with a slice taken
/// off the end.
__device__ static void b2b_init(Blake2b *s, uint32_t outlen, const uint8_t *person)
{
#pragma unroll
    for (int i = 0; i < 8; i++) {
        s->h[i] = B2B_IV[i];
    }
    // Parameter block word 0: digest_length | key_length<<8 | fanout<<16 |
    // depth<<24. Key length is zero here and fanout and depth are 1, which is
    // the sequential-mode setting every caller in this repository wants.
    s->h[0] ^= (uint64_t)(outlen & 0xff) | (0ULL << 8) | (1ULL << 16) | (1ULL << 24);
    if (person) {
        // Bytes 48..63 of the parameter block, which are words 6 and 7.
        uint64_t p0 = 0, p1 = 0;
#pragma unroll
        for (int b = 0; b < 8; b++) {
            p0 |= (uint64_t)person[b] << (8 * b);
            p1 |= (uint64_t)person[8 + b] << (8 * b);
        }
        s->h[6] ^= p0;
        s->h[7] ^= p1;
    }
    s->t = 0;
    s->fill = 0;
    s->outlen = outlen;
}

__device__ static void b2b_update(Blake2b *s, const uint8_t *in, uint32_t len)
{
    for (uint32_t i = 0; i < len; i++) {
        // A full buffer is compressed only when another byte arrives, never on
        // the byte that fills it. BLAKE2 marks the *last* block and a buffer
        // flushed eagerly would mark the wrong one -- the digest is right for
        // every message that is not a multiple of the block size, which is the
        // shape of bug that passes every casual test.
        if (s->fill == 128) {
            s->t += 128;
            b2b_compress(s, s->buf, false);
            s->fill = 0;
        }
        s->buf[s->fill++] = in[i];
    }
}

__device__ static void b2b_final(Blake2b *s, uint8_t *out)
{
    s->t += s->fill;
    for (uint32_t i = s->fill; i < 128; i++) {
        s->buf[i] = 0;
    }
    b2b_compress(s, s->buf, true);
    for (uint32_t i = 0; i < s->outlen; i++) {
        out[i] = (uint8_t)(s->h[i >> 3] >> (8 * (i & 7)));
    }
}

// The contract an algorithm implements.
//
// Adding one means writing a header that provides the three things below and
// adding a row to ALGOS in miner.cu. Nothing else in the harness changes.
//
//   __device__ void NAME_hash(uint32_t nonce, uint32_t out[8])
//       The nonce-dependent half. Whatever is invariant across nonces belongs
//       in __constant__ memory, computed once on the host -- that is the
//       midstate optimisation and it is the difference between a miner and a
//       hash benchmark.
//
//   void NAME_upload(void)
//       Push this algorithm's constants. Called once before anything runs.
//
//   const char *NAME_expect(void)
//       The digest of a known input, as lowercase hex. The harness prints the
//       device's answer beside this and refuses to benchmark on a mismatch.
//
// **The expected digest is not optional and it is not a formality.** Every
// algorithm here is checked against an implementation nobody in this
// repository wrote -- hashlib for both of the current two -- because a fast
// wrong hash is indistinguishable from a fast right one from the inside, and
// this project has already lost weeks to exactly that class of bug in its
// transformer. `tools/algocheck.py` is the other half of that arrangement.
//
// Word order is the trap. SHA-256 reads its message as big-endian 32-bit
// words; BLAKE2s reads little-endian. An algorithm that gets this wrong still
// produces a plausible digest of the right length, at the right speed, and it
// is wrong. The `expect` string is what catches it.

#pragma once
#include <stdint.h>

#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

// Compare a digest against a 256-bit target, most significant word first.
//
// A real comparison rather than a leading-zero count. Bitcoin reads the digest
// as a little-endian 256-bit integer, so the *last* word is the most
// significant; a leading-zero proxy answers a different question and cannot
// express a target that is not a power of two.
__device__ __forceinline__ bool below_target_be(const uint32_t h[8],
                                                const uint32_t target[8]) {
#pragma unroll
    for (int i = 7; i >= 0; --i) {
        uint32_t v = __byte_perm(h[i], 0, 0x0123);
        if (v < target[7 - i]) return true;
        if (v > target[7 - i]) return false;
    }
    return true;
}

// The same, for algorithms whose digest is already in host word order.
__device__ __forceinline__ bool below_target_le(const uint32_t h[8],
                                                const uint32_t target[8]) {
#pragma unroll
    for (int i = 7; i >= 0; --i) {
        if (h[i] < target[7 - i]) return true;
        if (h[i] > target[7 - i]) return false;
    }
    return true;
}

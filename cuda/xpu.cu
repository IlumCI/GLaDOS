// The GPU as a pool worker: take a job, scan nonces, report what was found.
//
// `miner.cu` beside this is a *benchmark* harness -- it hashes one fixed
// vector as fast as it can and prints a rate. This takes work from somewhere
// else, which is a different program even though the kernels are the same.
//
// ### It is driven over a pipe, and does not speak the pool's protocol
//
// The obvious thing would be to put a socket and a JSON parser in here. That
// would be the *fourth* implementation of `design/pool.md`'s protocol -- the
// kernel's `src/mine/proto.rs`, the pool which includes that same file, and
// `tools/poolclient.py` are the other three -- and a protocol is exactly where
// a second implementation is most expensive and hardest to see wrong.
//
// So the protocol stays in Rust, in `miner/`. This process speaks four words
// on stdin and two on stdout:
//
//     <- ready
//     -> job <sha256d|blake2s> <header-160-hex> <target-64-hex>
//     <- ok                          or  err <why>
//     -> scan <base-8-hex> <count-decimal>
//     <- found <nonce-8-hex>         or  none <hashes-decimal>
//     -> quit
//
// ### The nonce on this pipe is always the protocol's, and that is deliberate
//
// A header stores its nonce little-endian at offset 76. SHA-256 reads the
// message as big-endian words, so the word its kernel indexes by is the
// *byte-swapped* view of those four bytes. BLAKE2s reads little-endian, so its
// word is the nonce itself. Two algorithms, two conventions, one pipe.
//
// The swap therefore lives here, beside the algorithm that needs it, rather
// than in the caller. A parent that had to know which algorithms swap is a
// parent that gets it wrong for the third one -- and getting it wrong is not
// an error, it is a share the pool rejects with nothing to say why.
//
// Pinned by the strongest vector there is: given block 125552's header and its
// own target, `scan` answers `9546a142`, which is that block's nonce.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "algo.cuh"
#include "sha256d.cuh"
#include "blake2s.cuh"
#include "neoscrypt.cuh"
#include "heavyhash.cuh"

// The target, most significant word first. Its own symbol rather than the
// headers' own, because those ship fixed benchmark targets and this one
// changes with every job.
__constant__ uint32_t xTarget[8];

// BLAKE2s over an *80-byte header*, which is two blocks.
//
// `blake2s.cuh`'s own `blake2s_hash` is one block and is not this: it is a
// benchmark over a synthetic 64-byte input, correct for what it computes and
// not what any chain computes. Its verify digest matches neither the header
// hash nor the header's first block, which is how the difference surfaced --
// and shipping it as a mining algorithm would have produced wrong shares at
// half a gigahash a second.
//
// Block 0 is the first 64 header bytes and never changes across a job, so it
// is compressed once on the host into `xB2Mid` -- the same midstate bargain
// SHA-256d makes, and the reason this is a miner rather than a hash loop.
// Block 1 is the remaining sixteen bytes zero-padded, with the counter at 80
// because the counter is the *message length* and not the block index.
__constant__ uint32_t xB2Mid[8];   // state after header[0..64], t=64, not last
__constant__ uint32_t xB2Tail[16]; // block 1's words; word 3 is the nonce

__device__ __forceinline__ void blake2s_header_hash(uint32_t nonce, uint32_t out[8]) {
    uint32_t m[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = xB2Tail[i];
    // Header bytes 76..80 are block 1's bytes 12..16, which is word 3. Read
    // little-endian, so this is the nonce itself with no swap.
    m[3] = nonce;

    uint32_t v[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = xB2Mid[i];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[8 + i] = b2s_iv[i];
    v[12] ^= 80u;          // eighty bytes consumed, not sixty-four
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
    for (int i = 0; i < 8; ++i) out[i] = xB2Mid[i] ^ v[i] ^ v[8 + i];
}

enum XAlgo { X_SHA256D = 0, X_BLAKE2S = 1, X_NEOSCRYPT = 2, X_HEAVYHASH = 3 };

// Nonces per thread. Two, which is what `design/xpu.md` measured: widening the
// instruction-level parallelism past that bought nothing on this part.
#define NPT 2

// One kernel, specialised per algorithm.
//
// `base` and the answer are both **protocol** nonces; the SHA-256d swap
// happens here, which keeps both conventions in one function rather than
// spread across a pipe.
template <int A>
__global__ void xpu_scan_kernel(uint32_t base, uint32_t *found) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n0 = base + idx * NPT;
    uint32_t h[NPT][8];
#pragma unroll
    for (int j = 0; j < NPT; ++j) {
        uint32_t n = n0 + j;
        if (A == X_SHA256D)        sha256d_hash(__byte_perm(n, 0, 0x0123), h[j]);
        else if (A == X_HEAVYHASH) heavyhash_nonce(n, h[j]);
        else                       blake2s_header_hash(n, h[j]);
    }
#pragma unroll
    for (int j = 0; j < NPT; ++j) {
        // SHA-256d's digest words are big-endian; BLAKE2s's are already in
        // host order. Comparing with the wrong one accepts and rejects an
        // unrelated set of shares.
        bool ok = (A == X_SHA256D) ? below_target_be(h[j], xTarget)
                                   : below_target_le(h[j], xTarget);
        if (ok) {
            // The *lowest* nonce wins, so a scan is a function of its range
            // rather than of which warp retired first. Two runs over one range
            // then agree, which is what makes a disagreement worth chasing.
            atomicMin(found, n0 + j);
        }
    }
}

// NeoScrypt does not fit the kernel above and cannot be made to.
//
// The template scans `NPT` nonces per thread out of registers alone, which is
// exactly right for a hash whose whole state is a digest. NeoScrypt needs 32
// KiB of scratchpad *per concurrent hash*, so a thread does one nonce and the
// launch is bounded by memory rather than by the range asked for.
//
// The scratchpad is allocated once, on the first NeoScrypt job, and never
// freed. Lazily because a device mining only sha256d should not be holding a
// gigabyte it never touches; once, because allocating per scan would put a
// `cudaMalloc` inside the hot loop.
#define NS_MAX_THREADS 32768u

__global__ void xpu_neoscrypt_kernel(uint32_t base, uint32_t *found,
                                     uint32_t *v, uint8_t *kdf, uint32_t live) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= live) return;
    uint32_t h[8];
    neoscrypt_hash_at(base + idx, h, v + (size_t)idx * NS_SCRATCH_WORDS,
                      kdf + (size_t)idx * NS_KDF_BYTES);
    // NeoScrypt's digest is already in host word order, like BLAKE2s and
    // unlike SHA-256d.
    if (below_target_le(h, xTarget)) atomicMin(found, base + idx);
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda: %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
    return -1; } } while (0)

static int unhex(const char *s, uint8_t *out, int want) {
    for (int n = 0; n < want; ++n) {
        int hi = -1, lo = -1;
        char a = s[2*n], b = s[2*n+1];
        if (a >= '0' && a <= '9') hi = a - '0';
        else if (a >= 'a' && a <= 'f') hi = a - 'a' + 10;
        else if (a >= 'A' && a <= 'F') hi = a - 'A' + 10;
        if (b >= '0' && b <= '9') lo = b - '0';
        else if (b >= 'a' && b <= 'f') lo = b - 'a' + 10;
        else if (b >= 'A' && b <= 'F') lo = b - 'A' + 10;
        if (hi < 0 || lo < 0) return -1;
        out[n] = (uint8_t)((hi << 4) | lo);
    }
    return 0;
}

static uint32_t le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Host BLAKE2s compression, written separately from the device version on
// purpose. If the two disagree the vector check catches it, which a shared
// implementation could not -- the same bargain `sha_host_compress` makes.
static void b2_host_compress(uint32_t h[8], const uint8_t blk[64],
                             uint64_t t, int last) {
    static const uint32_t iv[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    static const uint8_t sig[10][16] = {
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
    uint32_t m[16], v[16];
    for (int i = 0; i < 16; ++i) m[i] = le32(blk + 4 * i);
    for (int i = 0; i < 8; ++i) v[i] = h[i];
    for (int i = 0; i < 8; ++i) v[8 + i] = iv[i];
    v[12] ^= (uint32_t)t;
    v[13] ^= (uint32_t)(t >> 32);
    if (last) v[14] = ~v[14];
    for (int r = 0; r < 10; ++r) {
        const uint8_t *s = sig[r];
        B2S_G(v[0], v[4], v[ 8], v[12], m[s[ 0]], m[s[ 1]]);
        B2S_G(v[1], v[5], v[ 9], v[13], m[s[ 2]], m[s[ 3]]);
        B2S_G(v[2], v[6], v[10], v[14], m[s[ 4]], m[s[ 5]]);
        B2S_G(v[3], v[7], v[11], v[15], m[s[ 6]], m[s[ 7]]);
        B2S_G(v[0], v[5], v[10], v[15], m[s[ 8]], m[s[ 9]]);
        B2S_G(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        B2S_G(v[2], v[7], v[ 8], v[13], m[s[12]], m[s[13]]);
        B2S_G(v[3], v[4], v[ 9], v[14], m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[8 + i];
}

// Take the NeoScrypt scratchpad, once. Answers non-zero when the card will not
// give it, which is a refusal the parent can act on rather than a crash.
static int ns_setup(uint32_t **v, uint8_t **kdf) {
    if (*v) return 0;
    const size_t vbytes = (size_t)NS_MAX_THREADS * NS_SCRATCH_WORDS * sizeof(uint32_t);
    const size_t kbytes = (size_t)NS_MAX_THREADS * NS_KDF_BYTES;
    if (cudaMalloc(v, vbytes) != cudaSuccess) { *v = NULL; return -1; }
    if (cudaMalloc(kdf, kbytes) != cudaSuccess) {
        cudaFree(*v);
        *v = NULL;
        return -1;
    }
    fprintf(stderr, "[xpu] neoscrypt scratchpad %.0f MiB for %u threads\n",
            vbytes / 1048576.0, NS_MAX_THREADS);
    return 0;
}

static int set_job(int algo, const uint8_t header[80], const uint8_t target_be[32]) {
    uint32_t tgt[8];
    for (int i = 0; i < 8; ++i) tgt[i] = sha_be32(target_be + 4 * i);
    CK(cudaMemcpyToSymbol(xTarget, tgt, sizeof tgt));

    if (algo == X_SHA256D) {
        uint32_t mid[8];
        memcpy(mid, hIV, sizeof mid);
        sha_host_compress(mid, header);
        uint32_t tail[3] = {
            sha_be32(header + 64), sha_be32(header + 68), sha_be32(header + 72)
        };
        CK(cudaMemcpyToSymbol(cK, hK, sizeof hK));
        CK(cudaMemcpyToSymbol(cIV, hIV, sizeof hIV));
        CK(cudaMemcpyToSymbol(cMid, mid, sizeof mid));
        CK(cudaMemcpyToSymbol(cTail, tail, sizeof tail));
        return 0;
    }

    if (algo == X_HEAVYHASH) {
        // The matrix is the previous block's, so it is built here once a job
        // and never per nonce. See `heavyhash.cuh`.
        return heavyhash_upload(header);
    }

    if (algo == X_NEOSCRYPT) {
        // Nothing to precompute. FastKDF tiles all eighty bytes across its
        // buffer before the first PRF call, so there is no invariant prefix to
        // absorb -- the same property yespower has and the reason neither gets
        // a midstate.
        neoscrypt_upload(header);
        return 0;
    }

    // BLAKE2s. `0x01010020` is the unkeyed 32-byte-digest parameter block:
    // depth 1, fanout 1, key length 0, digest length 32. A wrong digest length
    // here produces a perfectly well-formed hash that no other implementation
    // agrees with.
    uint32_t h[8] = {
        0x6a09e667u ^ 0x01010020u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    b2_host_compress(h, header, 64, 0);

    uint8_t blk1[64];
    memset(blk1, 0, sizeof blk1);
    memcpy(blk1, header + 64, 16);
    uint32_t tail[16];
    for (int i = 0; i < 16; ++i) tail[i] = le32(blk1 + 4 * i);

    CK(cudaMemcpyToSymbol(xB2Mid, h, sizeof h));
    CK(cudaMemcpyToSymbol(xB2Tail, tail, sizeof tail));
    return 0;
}

// Search `count` nonces from `base`, in as many launches as that takes.
//
// **The whole count, and that is the contract.** This rounded down to whole
// blocks and reported what it had actually done, which the parent did not
// read -- so a scan of 1,000 nonces searched 512, the parent advanced its
// cursor by 1,000, and the difference was never searched by anybody. Silent,
// and it grew teeth with NeoScrypt, whose launch is capped by a gigabyte of
// scratchpad rather than by the range asked for: a request for 47,000 nonces
// can only put 32,768 in flight.
//
// Looping here rather than reporting a short scan keeps the pipe's meaning
// simple -- `scan N` searches N nonces -- and the parent already sizes N to
// about a quarter second from a measured rate, so no single call runs long.
//
// One function, called by the pipe and by `--selftest`. A self-test driving a
// second copy of this loop would be checking the copy.
static uint32_t scan_range(int algo, uint32_t base, uint32_t count,
                           uint32_t *d_found, uint32_t **ns_v, uint8_t **ns_kdf,
                           uint32_t *scanned) {
    const uint32_t none = 0xffffffffu;
    if (cudaMemcpy(d_found, &none, 4, cudaMemcpyHostToDevice) != cudaSuccess) {
        if (scanned) *scanned = 0;
        return none;
    }
    uint32_t done = 0;
    uint32_t got = none;
    while (done < count) {
        const uint32_t want = count - done;
        uint32_t here;
        if (algo == X_NEOSCRYPT) {
            if (ns_setup(ns_v, ns_kdf)) {
                fprintf(stderr, "[xpu] no room for a neoscrypt scratchpad\n");
                break;
            }
            const int threads = 64;
            here = want < NS_MAX_THREADS ? want : NS_MAX_THREADS;
            const uint32_t blocks = (here + threads - 1) / threads;
            xpu_neoscrypt_kernel<<<blocks, threads>>>(base + done, d_found, *ns_v,
                                                     *ns_kdf, here);
        } else {
            // Rounded *up*, so the range asked for is covered. The last threads
            // then work a few nonces past it, which is harmless: a hit above
            // the range is still a hit on this header, and the parent resumes
            // from the nonce it is told rather than from where it expected.
            const int threads = 256;
            const uint32_t per_block = (uint32_t)threads * NPT;
            const uint32_t blocks = (want + per_block - 1) / per_block;
            here = blocks * per_block;
            if (algo == X_SHA256D)
                xpu_scan_kernel<X_SHA256D><<<blocks, threads>>>(base + done, d_found);
            else if (algo == X_HEAVYHASH)
                xpu_scan_kernel<X_HEAVYHASH><<<blocks, threads>>>(base + done, d_found);
            else
                xpu_scan_kernel<X_BLAKE2S><<<blocks, threads>>>(base + done, d_found);
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            fprintf(stderr, "[xpu] launch: %s\n", cudaGetErrorString(cudaGetLastError()));
            got = none;
            done = count;
            break;
        }
        done += here;
        if (cudaMemcpy(&got, d_found, 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "[xpu] readback failed\n");
            got = none;
            break;
        }
        // Stop at the first hit: the parent resumes from that nonce, so
        // searching past it would be work it is about to ask for again.
        if (got != none) break;
    }
    if (scanned) *scanned = done;
    return got;
}

// Every algorithm this device has, against a nonce somebody else established,
// through the same `set_job` and scan path the pipe uses.
//
// **The odd range is the point of the second half.** The scan loop used to
// round the requested count down to whole blocks and report what it had
// actually done, which the parent did not read -- so a scan of 1,000 nonces
// searched 512, the parent advanced its cursor by 1,000, and the difference
// was never looked at by anybody. Silent, and it grew teeth with NeoScrypt,
// whose launch is capped by a gigabyte of scratchpad rather than by the range
// asked for. A count that is deliberately not a multiple of any block size is
// what catches that coming back.
struct Selftest {
    const char *algo;
    const char *header;
    const char *target;
    uint32_t base;
    uint32_t count;
    uint32_t want;
};

static int selftest(uint32_t *d_found, uint32_t **ns_v, uint8_t **ns_kdf) {
    // Block 125552, whose nonce every explorer in the world agrees on, and the
    // two Feathercoin blocks `tools/neoscrypt.py` derived from the live chain.
    static const Selftest T[] = {
        {"sha256d",
         "0100000081cd02ab7e569e8bcd9317e2fe99f2de44d49ab2b8851ba4a3080000000000"
         "00e320b6c2fffc8d750423db8b1eb942ae710e951ed797f7affc8892b0f1fc122bc7f5"
         "d74df2b9441a42a14695",
         "00000000000404cb000000000000000000000000000000000000000000000000",
         0x95460000u, 262144u, 0x9546a142u},
        {"neoscrypt",
         "040000200a9245b1198825ab30d6dbae2b185e31f342d1058cc36851629bf435fc682b"
         "554d100771e4a23d210cd3a1503e3b1ca524e8482913ee5b4036edf4bd20db6a2993c0"
         "a16adc09011d002c5f48",
         "0000000109dc0000000000000000000000000000000000000000000000000000",
         0x485f2c00u, 64u, 0x485f2c00u},
        // BLAKE2s over an 80-byte header, which no chain reachable from here
        // publishes a block for -- so the vector comes from `hashlib`, which
        // is the same oracle `cuda/blake2s.cuh` uses and is not ours. The
        // nonce is the *lowest* in the range meeting a 16-leading-zero-bit
        // target, so this also checks that `atomicMin` makes a scan a function
        // of its range rather than of which warp retired first.
        {"blake2s",
         "040000200a9245b1198825ab30d6dbae2b185e31f342d1058cc36851629bf435fc682b"
         "554d100771e4a23d210cd3a1503e3b1ca524e8482913ee5b4036edf4bd20db6a2993c0"
         "a16adc09011d002c5f48",
         "0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
         0x00000000u, 70000u, 0x0000438cu},
        // Optical Bitcoin's mainnet genesis, whose hash its node asserts at
        // startup. 3,001 nonces ending at the real one, and `tools/heavyhash.py`
        // confirms it is the only nonce in the range under this target, so the
        // answer is the genesis nonce or the implementation is wrong.
        {"heavyhash",
         "0100000000000000000000000000000000000000000000000000000000000000000000"
         "00f5febbad19864a6900b6ce84287511e6e746229ce6239ff3df7481654778a4c4d3e1"
         "5d60ffff001c0747d042",
         "0000000000120000000000000000000000000000000000000000000000000000",
         0x42d03b4fu, 3001u, 0x42d04707u},
        // The same sha256d block, over a range that is a multiple of nothing.
        // 9546a142 - 95460000 = 41,282, so 41,283 nonces from the base is the
        // smallest range containing it, and every block size here divides
        // neither it nor the count.
        {"sha256d",
         "0100000081cd02ab7e569e8bcd9317e2fe99f2de44d49ab2b8851ba4a3080000000000"
         "00e320b6c2fffc8d750423db8b1eb942ae710e951ed797f7affc8892b0f1fc122bc7f5"
         "d74df2b9441a42a14695",
         "00000000000404cb000000000000000000000000000000000000000000000000",
         0x95460000u, 41283u, 0x9546a142u},
    };

    int bad = 0;
    for (size_t i = 0; i < sizeof T / sizeof T[0]; ++i) {
        uint8_t header[80], target[32];
        if (strlen(T[i].header) != 160 || strlen(T[i].target) != 64 ||
            unhex(T[i].header, header, 80) || unhex(T[i].target, target, 32)) {
            printf("FAIL  %s: the vector in this file is malformed\n", T[i].algo);
            bad++;
            continue;
        }
        int a = !strcmp(T[i].algo, "sha256d") ? X_SHA256D
              : !strcmp(T[i].algo, "blake2s") ? X_BLAKE2S
              : !strcmp(T[i].algo, "heavyhash") ? X_HEAVYHASH : X_NEOSCRYPT;
        if (set_job(a, header, target)) {
            printf("FAIL  %s: upload\n", T[i].algo);
            bad++;
            continue;
        }
        const uint32_t got = scan_range(a, T[i].base, T[i].count, d_found,
                                        ns_v, ns_kdf, NULL);
        const bool ok = got == T[i].want;
        printf("%-4s  %-9s over %6u nonces from %08x -> %08x\n",
               ok ? "ok" : "FAIL", T[i].algo, T[i].count, T[i].base, got);
        if (!ok) {
            printf("      wanted %08x\n", T[i].want);
            bad++;
        }
    }
    return bad;
}

int main(int argc, char **argv) {
    // **Unbuffered, and _IOLBF is not good enough here.** Windows' CRT
    // documents line buffering as behaving like *full* buffering, so a reply
    // written to a pipe sits in a buffer while the parent blocks waiting for
    // it. That is a deadlock presenting as a GPU that mines nothing, which is
    // the least informative symptom available.
    //
    // Found by running it rather than by reading it: with _IOLBF this printed
    // nothing at all down a pipe, not even its greeting.
    setvbuf(stdout, NULL, _IONBF, 0);

    uint32_t *d_ns_v = NULL;
    uint8_t *d_ns_kdf = NULL;
    uint32_t *d_found = NULL;
    if (cudaMalloc(&d_found, sizeof(uint32_t)) != cudaSuccess) {
        fprintf(stderr, "cuda: no device\n");
        return 1;
    }
    // Touch the device before saying ready, so the parent's first `scan` is not
    // also paying for context creation -- about two hundred milliseconds that
    // would otherwise land inside a measured interval.
    cudaFree(0);

    if (argc > 1 && !strcmp(argv[1], "--selftest")) {
        const int bad = selftest(d_found, &d_ns_v, &d_ns_kdf);
        cudaFree(d_found);
        cudaFree(d_ns_v);
        cudaFree(d_ns_kdf);
        return bad ? 1 : 0;
    }

    printf("ready\n");

    char line[512];
    int algo = -1;
    while (fgets(line, sizeof line, stdin)) {
        if (!strncmp(line, "quit", 4)) break;

        if (!strncmp(line, "job ", 4)) {
            char an[32], hh[200], th[80];
            if (sscanf(line + 4, "%31s %199s %79s", an, hh, th) != 3 ||
                strlen(hh) != 160 || strlen(th) != 64) {
                printf("err malformed job\n");
                continue;
            }
            int a;
            if (!strcmp(an, "sha256d")) a = X_SHA256D;
            else if (!strcmp(an, "blake2s")) a = X_BLAKE2S;
            else if (!strcmp(an, "neoscrypt")) a = X_NEOSCRYPT;
            else if (!strcmp(an, "heavyhash")) a = X_HEAVYHASH;
            else { printf("err this device cannot compute %s\n", an); continue; }

            uint8_t header[80], target[32];
            if (unhex(hh, header, 80) || unhex(th, target, 32)) {
                printf("err job is not hex\n");
                continue;
            }
            if (set_job(a, header, target)) { printf("err upload failed\n"); continue; }
            algo = a;
            printf("ok\n");
            continue;
        }

        if (!strncmp(line, "scan ", 5)) {
            unsigned base = 0, count = 0;
            if (sscanf(line + 5, "%x %u", &base, &count) != 2) {
                printf("err malformed scan\n"); continue;
            }
            if (algo < 0) { printf("err no job\n"); continue; }

            const uint32_t none = 0xffffffffu;
            // **The whole count, in as many launches as it takes.**
            //
            // This rounded down to whole blocks and reported what it had
            // actually done, which the parent did not read -- so a scan of
            // 1,000 nonces did 512 and the miner advanced its cursor by 1,000,
            // skipping the difference. Silent, and it grew teeth with
            // NeoScrypt, whose launch is capped by a gigabyte of scratchpad
            // rather than by the range: a request for 47,000 nonces can only
            // put 32,768 in flight, and the other 14,000 would never be
            // searched at all.
            //
            // Looping here rather than reporting a short scan keeps the pipe's
            // meaning simple -- `scan N` searches N nonces -- and the parent
            // already sizes N to about a quarter second from a measured rate,
            // so no single call runs long.
            uint32_t done = 0;
            const uint32_t got = scan_range(algo, base, count, d_found,
                                            &d_ns_v, &d_ns_kdf, &done);
            if (got != none) printf("found %08x\n", got);
            else printf("none %u\n", done);
            continue;
        }

        printf("err unknown\n");
    }
    cudaFree(d_found);
    cudaFree(d_ns_v);
    cudaFree(d_ns_kdf);
    return 0;
}

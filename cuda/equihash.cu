// Equihash list generation on the device, and nothing else yet.
//
// `design/equihash.md` breaks the 192,7 solver into six parts and this is the
// second-cheapest of them: "list generation kernel, 100-150 lines, low". It is
// worth doing before the collision rounds for one reason -- it is the only part
// whose output can be compared, entry for entry, against something that is not
// this program. `tools/equihash.py --emit` prints the same list from
// `hashlib.blake2b`, and the two either agree on every byte or the rounds above
// would be sorting garbage.
//
// ### One thread per hash, not per entry
//
// A BLAKE2b output covers `per_output` indices -- two of them at 192,7, because
// 512/192 is 2 -- so a thread per *index* would compute every digest twice. The
// counter fed to the hash is `index / per_output` and the slice taken is
// `index % per_output`, which is the spec's own arithmetic and the reason the
// launch is over 2^24 threads to fill a 2^25-entry list.
//
// ### What it costs, which is the number that decides the rest
//
// 2^25 entries of 24 bytes is 768 MiB. `design/equihash.md` budgets "~2048 MiB"
// for a miniZ-class solver against the 3836 MiB this card has, and says the
// reference wants 3336 -- so the list alone is a fifth of the budget and the
// sort buffers above it are what make the difference between fitting and not.
// This program prints the figure rather than asserting it.
//
//   nvcc -O3 -arch=sm_86 equihash.cu -o equihash
//   ./equihash --emit 8            | diff against tools/equihash.py --emit
//   ./equihash --full              | time the whole list and report VRAM
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "blake2b.cuh"

struct Params {
    uint32_t n, k, collision, per_output, hash_len, slice_len, index_bits;
    uint64_t entries;
};

// Derived, never tabled, for the reason `tools/equihash.py` gives: a table is a
// second place for 192,7 to be written down.
static Params derive(uint32_t n, uint32_t k)
{
    Params p;
    p.n = n;
    p.k = k;
    p.collision = n / (k + 1);
    p.per_output = 512 / n;
    p.slice_len = n / 8;
    p.hash_len = p.per_output * p.slice_len;
    p.index_bits = p.collision + 1;
    p.entries = 1ULL << p.index_bits;
    return p;
}

__global__ void gen(const uint8_t *header, uint32_t header_len,
                    const uint8_t *pers, uint32_t hash_len,
                    uint32_t per_output, uint32_t slice_len,
                    uint64_t hashes, uint64_t entries, uint8_t *out)
{
    uint64_t h = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hashes) {
        return;
    }
    Blake2b s;
    b2b_init(&s, hash_len, pers);
    b2b_update(&s, header, header_len);
    // The counter is little-endian four bytes, which is the spec's and not a
    // choice. Big-endian here produces a perfectly plausible list that collides
    // at the right rate and solves a different puzzle.
    uint8_t ctr[4] = {
        (uint8_t)(h & 0xff), (uint8_t)((h >> 8) & 0xff),
        (uint8_t)((h >> 16) & 0xff), (uint8_t)((h >> 24) & 0xff),
    };
    b2b_update(&s, ctr, 4);
    uint8_t d[64];
    b2b_final(&s, d);

    for (uint32_t j = 0; j < per_output; j++) {
        uint64_t idx = h * per_output + j;
        if (idx >= entries) {
            return;
        }
        uint8_t *dst = out + idx * slice_len;
        for (uint32_t b = 0; b < slice_len; b++) {
            dst[b] = d[j * slice_len + b];
        }
    }
}

static void personal(uint32_t n, uint32_t k, uint8_t out[16])
{
    memcpy(out, "ZcashPoW", 8);
    out[8]  = (uint8_t)(n & 0xff);
    out[9]  = (uint8_t)((n >> 8) & 0xff);
    out[10] = (uint8_t)((n >> 16) & 0xff);
    out[11] = (uint8_t)((n >> 24) & 0xff);
    out[12] = (uint8_t)(k & 0xff);
    out[13] = (uint8_t)((k >> 8) & 0xff);
    out[14] = (uint8_t)((k >> 16) & 0xff);
    out[15] = (uint8_t)((k >> 24) & 0xff);
}

int main(int argc, char **argv)
{
    uint32_t n = 192, k = 7, emit = 0;
    bool full = false;
    const char *hdr = "equihash";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc)        n = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i + 1 < argc)   k = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--emit") && i + 1 < argc) emit = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--header") && i + 1 < argc) hdr = argv[++i];
        else if (!strcmp(argv[i], "--full"))                full = true;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    Params p = derive(n, k);
    if (p.hash_len > 64) {
        fprintf(stderr, "n=%u wants %u bytes of digest; BLAKE2b gives 64\n", n, p.hash_len);
        return 2;
    }

    // With `--emit` only the first few entries are wanted, so only the hashes
    // covering them are computed. A run that allocated the whole list to print
    // eight lines would refuse on a card with something else already on it.
    uint64_t entries = full ? p.entries : (emit ? emit : 8);
    if (entries > p.entries) entries = p.entries;
    uint64_t hashes = (entries + p.per_output - 1) / p.per_output;
    size_t bytes = (size_t)entries * p.slice_len;

    uint8_t pers[16];
    personal(n, k, pers);

    uint8_t *d_hdr = nullptr, *d_pers = nullptr, *d_out = nullptr;
    size_t hlen = strlen(hdr);
    if (cudaMalloc(&d_hdr, hlen ? hlen : 1) != cudaSuccess ||
        cudaMalloc(&d_pers, 16) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for the small buffers\n");
        return 1;
    }
    cudaError_t e = cudaMalloc(&d_out, bytes);
    if (e != cudaSuccess) {
        fprintf(stderr, "cudaMalloc of %.1f MiB failed: %s\n",
                bytes / 1048576.0, cudaGetErrorString(e));
        return 1;
    }
    cudaMemcpy(d_hdr, hdr, hlen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pers, pers, 16, cudaMemcpyHostToDevice);

    const int TPB = 256;
    uint64_t blocks = (hashes + TPB - 1) / TPB;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    gen<<<(unsigned)blocks, TPB>>>(d_hdr, (uint32_t)hlen, d_pers, p.hash_len,
                                   p.per_output, p.slice_len, hashes, entries, d_out);
    cudaEventRecord(t1);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(e));
        return 1;
    }
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);

    if (full) {
        size_t freeb = 0, totalb = 0;
        cudaMemGetInfo(&freeb, &totalb);
        printf("%u,%u  %llu entries x %u B = %.1f MiB in %.1f ms\n",
               n, k, (unsigned long long)entries, p.slice_len, bytes / 1048576.0, ms);
        printf("  %.1f Mentry/s, and the card reports %.0f of %.0f MiB free\n",
               entries / (ms * 1000.0), freeb / 1048576.0, totalb / 1048576.0);
        // The figure the rest of the solver has to fit inside, stated rather
        // than left to be discovered by a failing cudaMalloc at round three.
        printf("  the list is %.0f%% of this card\n", 100.0 * bytes / (double)totalb);
    } else {
        uint8_t *host = (uint8_t *)malloc(bytes);
        cudaMemcpy(host, d_out, bytes, cudaMemcpyDeviceToHost);
        for (uint64_t i = 0; i < entries; i++) {
            printf("%llu ", (unsigned long long)i);
            for (uint32_t b = 0; b < p.slice_len; b++) {
                printf("%02x", host[i * p.slice_len + b]);
            }
            printf("\n");
        }
        free(host);
    }
    cudaFree(d_hdr);
    cudaFree(d_pers);
    cudaFree(d_out);
    return 0;
}

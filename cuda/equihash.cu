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

static const int TPB = 256;

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

/// Which bucket an entry falls in: the top `bits` of its first collision block.
///
/// Bucketing on a *prefix* rather than the whole collision block is what makes
/// the sort affordable -- 24 bits is 16.7M buckets for 33.5M entries, which is a
/// table larger than the data it indexes. A prefix of 16 gives 65,536 buckets
/// and leaves the remaining 8 bits to be compared within one.
__device__ __forceinline__ uint32_t bucket_of(const uint8_t *e, uint32_t bits)
{
    // The first `bits` bits, most significant first, read a byte at a time so
    // the function does not care whether `bits` lands on a boundary.
    uint32_t v = 0;
    for (uint32_t i = 0; i < (bits + 7) / 8; i++) {
        v = (v << 8) | e[i];
    }
    uint32_t got = ((bits + 7) / 8) * 8;
    return v >> (got - bits);
}

/// Occupancy only: how many entries land in each bucket, and nothing stored.
///
/// **This exists because `design/equihash.md`'s hazard 3 cannot be debugged
/// later.** "When a bucket exceeds NSLOTS, entries are dropped. It does not
/// crash and does not corrupt: it *lowers the solution rate*, which is
/// indistinguishable from bad luck." So `NSLOTS` is chosen from a measured
/// distribution rather than from the average, and the tail is what decides it --
/// a bucket table sized at the mean drops entries from half its buckets.
///
/// Counters only, so this runs before any layout is committed to and costs
/// 4 bytes a bucket instead of `NSLOTS * entry`.
__global__ void histo(const uint8_t *list, uint64_t entries, uint32_t slice_len,
                      uint32_t bits, uint32_t *counts)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= entries) {
        return;
    }
    atomicAdd(&counts[bucket_of(list + i * slice_len, bits)], 1u);
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
    uint32_t n = 192, k = 7, emit = 0, bucket_bits = 0;
    bool full = false;
    const char *hdr = "equihash";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc)        n = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i + 1 < argc)   k = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--emit") && i + 1 < argc) emit = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--header") && i + 1 < argc) hdr = argv[++i];
        else if (!strcmp(argv[i], "--full"))                full = true;
        else if (!strcmp(argv[i], "--buckets") && i + 1 < argc) {
            bucket_bits = (uint32_t)atoi(argv[++i]);
            full = true;   // the distribution is only meaningful over the whole list
        }
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

    if (bucket_bits) {
        if (bucket_bits < 1 || bucket_bits > 24 || bucket_bits > p.collision) {
            fprintf(stderr, "--buckets wants 1..%u bits for these parameters\n", p.collision);
            return 2;
        }
        uint64_t nb = 1ULL << bucket_bits;
        uint32_t *d_counts = nullptr;
        if (cudaMalloc(&d_counts, nb * sizeof(uint32_t)) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc of %llu counters failed\n", (unsigned long long)nb);
            return 1;
        }
        cudaMemset(d_counts, 0, nb * sizeof(uint32_t));
        uint64_t hb = (entries + TPB - 1) / TPB;
        histo<<<(unsigned)hb, TPB>>>(d_out, entries, p.slice_len, bucket_bits, d_counts);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            fprintf(stderr, "histogram kernel failed\n");
            return 1;
        }
        uint32_t *counts = (uint32_t *)malloc(nb * sizeof(uint32_t));
        cudaMemcpy(counts, d_counts, nb * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        uint64_t sum = 0, empty = 0;
        uint32_t peak = 0;
        for (uint64_t b = 0; b < nb; b++) {
            uint32_t c = counts[b];
            sum += c;
            if (!c) empty++;
            if (c > peak) peak = c;
        }
        // **A histogram of the histogram, sized from the peak in a second pass
        // rather than capped at a constant.** Written with a fixed cap of 4096
        // first, which is fine while the mean is 512 and silently wrong the
        // moment it is 8192: every bucket above the cap piles into the last bin
        // and the quantiles come back *equal to the cap*, which reads like a
        // suspiciously round distribution rather than like a broken report. At
        // 2^12 buckets it printed p99.9 = 4096 against a real peak of 8591.
        //
        // Two passes over the counters is the fix and it costs nothing: this is
        // host memory and 16M counters is 64 MiB.
        uint64_t bins = (uint64_t)peak + 1;
        uint64_t *of = (uint64_t *)calloc(bins, sizeof(uint64_t));
        if (!of) {
            fprintf(stderr, "could not allocate %llu bins\n", (unsigned long long)bins);
            return 1;
        }
        for (uint64_t b = 0; b < nb; b++) {
            of[counts[b]]++;
        }
        double mean = (double)sum / (double)nb;
        // The quantiles that decide NSLOTS. The mean is the number that looks
        // like the answer and is not: sizing at it drops entries from about half
        // the buckets, silently, forever.
        uint64_t want999 = (uint64_t)(0.999 * (double)nb);
        uint64_t want9999 = (uint64_t)(0.9999 * (double)nb);
        uint32_t p999 = 0, p9999 = 0;
        uint64_t run = 0;
        for (uint64_t c = 0; c < bins; c++) {
            run += of[c];
            if (!p999 && run >= want999) p999 = (uint32_t)c;
            if (!p9999 && run >= want9999) p9999 = (uint32_t)c;
        }
        printf("%u,%u  %llu entries into 2^%u = %llu buckets\n",
               n, k, (unsigned long long)entries, bucket_bits, (unsigned long long)nb);
        printf("  mean %.1f   p99.9 %u   p99.99 %u   peak %u   empty %llu\n",
               mean, p999, p9999, peak, (unsigned long long)empty);
        printf("  entries counted %llu of %llu%s\n", (unsigned long long)sum,
               (unsigned long long)entries,
               sum == entries ? "  -- none lost" : "  -- LOST SOME");
        // What the layout would cost at each candidate, which is the trade the
        // brief says has to be tunable rather than guessed.
        for (uint32_t slots : {p999, p9999, peak}) {
            double mib = (double)nb * slots * (p.slice_len + 4) / 1048576.0;
            printf("  NSLOTS %-5u would cost %8.1f MiB of bucket table%s\n",
                   slots, mib, slots == peak ? "  (never drops an entry)" : "");
        }
        free(counts);
        free(of);
        cudaFree(d_counts);
        return 0;
    }

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

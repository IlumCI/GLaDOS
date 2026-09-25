// Does the device's BLAKE2b agree with somebody else's?
//
// Four vectors, chosen so that each one fails differently:
//
//   1. "abc" at 64 bytes -- RFC 7693's own published digest, so the first
//      check is against the standard rather than against a library.
//   2. the empty message -- the only input where the finalisation flag is the
//      whole of what is being tested.
//   3. 200 bytes -- crosses the 128-byte block boundary, which is the one
//      thing a single-block implementation gets right by accident. `b2b_update`
//      compresses a full buffer only when the next byte arrives, and this is
//      the vector that says whether that is working or merely written down.
//   4. 48 bytes out with Equihash's personalisation -- both of the parameter
//      block fields at once, which is the shape every Equihash hash has.
//
// Built and run from Linux with nvcc; `cuda/README.md`'s recipe is the Windows
// one and the toolkit is on this machine either way.
//
//   nvcc -O3 -arch=sm_86 blake2btest.cu -o blake2btest && ./blake2btest
//   python3 ../tools/blake2b.py
//
// The two print the same four lines or this is not finished.
#include <stdio.h>
#include <string.h>
#include "blake2b.cuh"

__global__ void run(const uint8_t *msg, uint32_t len, uint32_t outlen,
                    const uint8_t *person, int has_person, uint8_t *out)
{
    Blake2b s;
    b2b_init(&s, outlen, has_person ? person : nullptr);
    b2b_update(&s, msg, len);
    b2b_final(&s, out);
}

static void one(const char *label, const uint8_t *msg, uint32_t len,
                uint32_t outlen, const uint8_t *person)
{
    uint8_t *d_msg = nullptr, *d_out = nullptr, *d_p = nullptr;
    cudaMalloc(&d_msg, len ? len : 1);
    cudaMalloc(&d_out, outlen);
    cudaMalloc(&d_p, 16);
    if (len) cudaMemcpy(d_msg, msg, len, cudaMemcpyHostToDevice);
    if (person) cudaMemcpy(d_p, person, 16, cudaMemcpyHostToDevice);

    run<<<1, 1>>>(d_msg, len, outlen, d_p, person != nullptr, d_out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("%-28s CUDA ERROR %s\n", label, cudaGetErrorString(e));
        return;
    }
    uint8_t out[64];
    cudaMemcpy(out, d_out, outlen, cudaMemcpyDeviceToHost);
    printf("%-28s ", label);
    for (uint32_t i = 0; i < outlen; i++) printf("%02x", out[i]);
    printf("\n");
    cudaFree(d_msg); cudaFree(d_out); cudaFree(d_p);
}

int main(void)
{
    // "ZcashPoW" || le32(n) || le32(k), n=192 k=7. The bytes are written out
    // rather than built, because this is the constant `design/equihash.md`
    // names as the first hazard and a reader should be able to see it.
    const uint8_t person[16] = {
        'Z','c','a','s','h','P','o','W',
        192, 0, 0, 0,
        7,   0, 0, 0,
    };
    uint8_t abc[3] = {'a','b','c'};
    uint8_t big[200];
    for (int i = 0; i < 200; i++) big[i] = (uint8_t)i;

    one("abc/64", abc, 3, 64, nullptr);
    one("empty/64", nullptr, 0, 64, nullptr);
    one("200bytes/64", big, 200, 64, nullptr);
    one("abc/48/ZcashPoW-192-7", abc, 3, 48, person);
    return 0;
}

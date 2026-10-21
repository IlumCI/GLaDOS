/* membw -- memory bandwidth and a compute loop, timed, from inside a guest.
 * Compared against the same binary on the host, it says whether a guest's
 * memory is as fast as it should be (cache attributes, page size) and
 * whether its code is (CPU features, preemption).
 *   gcc -m32 -static -O2 -o out/membw32 tools/guest/membw.c
 *   gcc -static -O2 -o out/membw tools/guest/membw.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    size_t mb = argc > 1 ? (size_t)atoi(argv[1]) : 128;
    size_t n = mb << 20;
    char *a = mmap(0, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char *b = mmap(0, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (a == MAP_FAILED || b == MAP_FAILED) { puts("mmap failed"); return 1; }
    memset(a, 1, n); memset(b, 2, n);
    double t = now();
    for (int i = 0; i < 4; i++) memset(a, i, n);
    double set = 4.0 * mb / (now() - t);
    t = now();
    for (int i = 0; i < 4; i++) memcpy(b, a, n);
    double cpy = 4.0 * mb / (now() - t);
    /* Random reads across the buffer: what TLB reach costs. */
    unsigned x = 12345, sum = 0;
    t = now();
    for (int i = 0; i < 20000000; i++) { x = x * 1103515245 + 12345; sum += a[(x >> 3) % n]; }
    double rnd = 20.0 / (now() - t);
    /* Pure arithmetic, no memory. */
    double f = 1.0;
    t = now();
    for (long i = 0; i < 200000000; i++) f = f * 1.0000001 + 1e-9;
    double alu = 0.2 / (now() - t);
    printf("memset %.0f MB/s  memcpy %.0f MB/s  random %.1f M/s  alu %.2f G/s  (%u %g)\n", set, cpy, rnd, alu, sum, f);
    return 0;
}

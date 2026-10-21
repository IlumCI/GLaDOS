/* mtbw -- how guest threads scale: N threads of random reads over their own
 * buffer, then N threads of arithmetic, each timed against one thread alone.
 * A machine where four threads do less than four times the work of one is
 * losing it somewhere: TLB shootdowns, page walks, scheduling.
 *   gcc -m32 -static-pie -fPIE -O2 -pthread -o out/mtbw32 tools/guest/mtbw.c */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

static size_t len = 64 << 20;
static volatile unsigned sink;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void *reads(void *arg) {
    char *a = mmap(0, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memset(a, 1, len);
    unsigned x = (unsigned)(size_t)arg * 7919 + 1, s = 0;
    for (int i = 0; i < 30000000; i++) { x = x * 1103515245 + 12345; s += a[(x >> 3) % len]; }
    sink += s;
    munmap(a, len);
    return 0;
}

static void *alu(void *arg) {
    double f = 1.0 + (size_t)arg;
    for (long i = 0; i < 300000000; i++) f = f * 1.0000001 + 1e-9;
    sink += (unsigned)f;
    return 0;
}

static double run(void *(*fn)(void *), int n) {
    pthread_t t[16];
    double s = now();
    for (int i = 0; i < n; i++) pthread_create(&t[i], 0, fn, (void *)(size_t)i);
    for (int i = 0; i < n; i++) pthread_join(t[i], 0);
    return now() - s;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 4;
    double r1 = run(reads, 1), rn = run(reads, n), a1 = run(alu, 1), an = run(alu, n);
    printf("reads: 1 thread %.2fs, %d threads %.2fs -> %.2fx the work per second\n", r1, n, rn, n * r1 / rn);
    printf("alu:   1 thread %.2fs, %d threads %.2fs -> %.2fx the work per second\n", a1, n, an, n * a1 / an);
    return 0;
}

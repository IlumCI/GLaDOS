/* The two custom signals, driven by a real glibc program.
 *
 * SIGRANDOM (69420) rolls a real signal and sends that; SIGQUARANTINE (2020)
 * seals a process family off from the rest of the machine and terminates the
 * whole field at once. Both are sent through the raw syscall rather than the
 * kill() wrapper, since their numbers are above anything libc expects.
 *
 * Built with the host's gcc, run under the host's glibc; see tools/sky.py for
 * the closure.
 *   sigtest random       the roulette: a child dies of whatever the wheel rolls
 *   sigtest quarantine   a self-replicating tree, sealed and purged en masse
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <sys/wait.h>
#include <sys/syscall.h>

#define SIGRANDOM     69420
#define SIGQUARANTINE 2020

static long raw_kill(pid_t pid, long sig) { return syscall(SYS_kill, pid, sig); }

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "random";

    if (!strcmp(mode, "random")) {
        /* A child that keeps making a syscall, so a non-fatal roll is actually
         * delivered (delivery happens on the way out of a syscall). */
        pid_t c = fork();
        if (c == 0) { for (;;) sched_yield(); }
        raw_kill(c, SIGRANDOM);
        int st = 0;
        pid_t w = waitpid(c, &st, 0);
        /* Whatever the wheel landed on, the child had no handler, so a real
         * signal ends it. The one guarantee is it died of one, not of an exit. */
        int ok = w == c && WIFSIGNALED(st);
        printf("sigtest: random child %d %s (signal %d) %s\n", c,
               WIFSIGNALED(st) ? "died of" : "exited", WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st),
               ok ? "ok" : "WRONG");
        printf("sigtest: %s\n", ok ? "all ok" : "FAILED");
        return ok ? 0 : 1;
    }

    if (!strcmp(mode, "quarantine")) {
        /* A stand-in for a self-replicating tree: several children that spin
         * forever and would never stop on their own. */
        for (int i = 0; i < 3; i++) {
            if (fork() == 0) { for (;;) {} }
        }
        printf("sigtest: a family of 4 is running; sealing and purging it\n");
        fflush(stdout);
        /* Sent to the session itself. The family is the whole connected tree,
         * so this process goes into the field with its children -- there is no
         * line after this one, and the machine surviving is the result. */
        raw_kill(getpid(), SIGQUARANTINE);
        for (;;) {}
    }

    printf("sigtest: unknown mode %s\n", mode);
    return 2;
}

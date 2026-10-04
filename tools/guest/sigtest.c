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

#define SIGWINCH 28
#define SIGTSTP  20
#define SIGTERM  15

static long raw_kill(pid_t pid, long sig) { return syscall(SYS_kill, pid, sig); }

/* Fork a child that raises `sig` on itself and then exits 42 if it is still
 * alive, and report how it ended: 42 if it reached the exit, otherwise the
 * negative of the signal that killed it.
 *
 * The child signals itself rather than the parent signalling it, which removes
 * both hazards the earlier versions hit: no parent/child race over whether the
 * signal is pending yet, and no long loop or sleep -- the signal is pending the
 * instant raw_kill is called and is delivered on the way out of that very
 * syscall, so a run of a few cheap syscalls is all it takes and the 30-second
 * deadline is never in play. If the signal is ignored or stops-but-cannot-stop
 * here, the child reaches _exit(42); if it is fatal, it dies of it first. */
static int outcome(long sig) {
    pid_t c = fork();
    if (c == 0) {
        raw_kill(getpid(), sig);
        for (int i = 0; i < 8; i++) sched_yield();
        _exit(42);
    }
    int st = -1;
    pid_t w = waitpid(c, &st, 0);
    if (w != c) return -1000;
    if (WIFEXITED(st)) return WEXITSTATUS(st);
    if (WIFSIGNALED(st)) return -WTERMSIG(st);
    return -999;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "random";

    if (!strcmp(mode, "defaults")) {
        int bad = 0;
        /* Default-ignore: a window resize must not kill a program. */
        int w = outcome(SIGWINCH);
        if (w != 42) { printf("sigtest: SIGWINCH ended it with %d, expected exit 42 WRONG\n", w); bad |= 1; }
        else printf("sigtest: SIGWINCH ignored, survived ok\n");
        /* Default-stop: no job control, so a no-op -- but never a kill. */
        int s = outcome(SIGTSTP);
        if (s != 42) { printf("sigtest: SIGTSTP ended it with %d, expected exit 42 WRONG\n", s); bad |= 2; }
        else printf("sigtest: SIGTSTP did not kill it ok\n");
        /* Default-terminate still terminates: outcome is -15. */
        int t = outcome(SIGTERM);
        if (t != -15) { printf("sigtest: SIGTERM ended it with %d, expected signal 15 WRONG\n", t); bad |= 4; }
        else printf("sigtest: SIGTERM terminated ok\n");
        printf("sigtest: %s\n", bad ? "FAILED" : "all ok");
        return bad;
    }

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

/* glibc's two ways of making a process, as a real program uses them.
 *
 * glibc never calls fork(2): fork() is clone(CLONE_CHILD_SETTID |
 * CLONE_CHILD_CLEARTID | SIGCHLD), and posix_spawn() -- which system() uses --
 * is clone(CLONE_VM | CLONE_VFORK | SIGCHLD) on a stack of its own. Each case
 * prints what it saw, and the exit code is a mask of what went wrong, zero
 * meaning nothing did.
 *
 * Built with the host's gcc and run under the host's glibc; see tools/sky.py
 * for the closure. Usage inside GLaDOS: forktest            (the parent)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <signal.h>

extern char **environ;

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "spawned")) {
        printf("forktest: spawned child %d running\n", getpid());
        return 9;
    }
    int bad = 0;
    pid_t parent = getpid();

    /* fork(): the child sees its own pid, and glibc's cached tid is right. */
    pid_t c = fork();
    if (c == 0) {
        pid_t me = getpid();
        pid_t tid = (pid_t)syscall(SYS_gettid);
        int ok = me != parent && tid == me;
        printf("forktest: fork child pid %d tid %d %s\n", me, tid, ok ? "ok" : "WRONG");
        fflush(stdout);
        _exit(ok ? 7 : 8);
    }
    if (c < 0) { printf("forktest: fork failed\n"); bad |= 1; }
    else {
        int st = 0;
        pid_t w = waitpid(c, &st, 0);
        int ok = w == c && WIFEXITED(st) && WEXITSTATUS(st) == 7;
        printf("forktest: fork parent saw child %d exit %d %s\n", c, WEXITSTATUS(st), ok ? "ok" : "WRONG");
        if (!ok) bad |= 2;
    }

    /* posix_spawn(): itself, with an argument saying it is the child. */
    pid_t s = 0;
    char *args[] = { argv[0], "spawned", NULL };
    int r = posix_spawn(&s, argv[0], NULL, NULL, args, environ);
    if (r != 0) { printf("forktest: posix_spawn failed %d\n", r); bad |= 4; }
    else {
        int st = 0;
        pid_t w = waitpid(s, &st, 0);
        int ok = w == s && WIFEXITED(st) && WEXITSTATUS(st) == 9;
        printf("forktest: spawn parent saw child %d exit %d %s\n", s, WEXITSTATUS(st), ok ? "ok" : "WRONG");
        if (!ok) bad |= 8;
    }
    /* A child that crashes ends itself and nothing else: the parent is told it
     * died of SIGSEGV and carries on. Twice, so a fault that leaves the kernel
     * in a state the second one trips over is caught too. */
    for (int round = 0; round < 2; round++) {
        pid_t k = fork();
        if (k == 0) {
            volatile int *nowhere = (int *)0x10;
            *nowhere = 1;
            _exit(0);
        }
        int st = 0;
        pid_t w = waitpid(k, &st, 0);
        int ok = w == k && WIFSIGNALED(st) && WTERMSIG(st) == 11;
        printf("forktest: crashing child %d %s signal %d %s\n", k,
               WIFSIGNALED(st) ? "died of" : "exited, not", WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st),
               ok ? "ok" : "WRONG");
        if (!ok) bad |= 16;
    }
    /* A child spinning in a loop that makes no syscall can only be stopped by
     * SIGKILL, which therefore cannot wait for a syscall to be delivered. */
    {
        pid_t k = fork();
        if (k == 0) { for (;;) {} }
        kill(k, SIGKILL);
        int st = 0;
        pid_t w = waitpid(k, &st, 0);
        int ok = w == k && WIFSIGNALED(st) && WTERMSIG(st) == 9;
        printf("forktest: spinning child %d %s\n", k, ok ? "killed by SIGKILL ok" : "WRONG");
        if (!ok) bad |= 32;
    }

    /* Many in a row, so a child that leaks its memory copy runs the heap out
     * rather than going unnoticed. */
    {
        int good = 0;
        for (int i = 0; i < 20; i++) {
            pid_t k = fork();
            if (k == 0) _exit(i);
            int st = 0;
            if (waitpid(k, &st, 0) == k && WIFEXITED(st) && WEXITSTATUS(st) == i) good++;
        }
        printf("forktest: %d of 20 back-to-back children %s\n", good, good == 20 ? "ok" : "WRONG");
        if (good != 20) bad |= 64;
    }

    /* An orphan: left spinning when the parent exits, for the session's end to
     * sweep up. The shell coming back is the check. */
    if (argc > 1 && !strcmp(argv[1], "orphan")) {
        if (fork() == 0) { for (;;) {} }
        printf("forktest: leaving a spinning orphan behind\n");
    }
    printf("forktest: %s\n", bad ? "FAILED" : "all ok");
    return bad;
}

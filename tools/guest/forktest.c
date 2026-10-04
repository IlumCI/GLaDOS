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
    printf("forktest: %s\n", bad ? "FAILED" : "all ok");
    return bad;
}

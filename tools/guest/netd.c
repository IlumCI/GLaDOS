/*
 * netd -- the network manager, at ring 3.
 *
 * The kernel keeps the mechanism: drivers, the 802.11 station, WPA2, DHCP.
 * This keeps the policy: which network, when to rejoin, when to ask for an
 * address. It talks to the kernel over /run/glados/netctl, one line at a
 * time (see src/linux/netctl.rs for the whole vocabulary), and says
 * "manage on" first so the kernel's own auto-join stands down while it runs.
 *
 *   netd                       run as the manager (what /etc/services starts)
 *   netd <request...>          one request, printed: netd status,
 *                              netd join home secret, netd remember home secret
 *
 * Built static, so it needs nothing staged beside it:
 *   gcc -static -O2 -o out/netd tools/guest/netd.c
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define PATH "/run/glados/netctl"

static int fd = -1;

static int dial(void) {
    struct sockaddr_un a;
    int s = socket(AF_UNIX, SOCK_STREAM, 0);
    if (s < 0) return -1;
    memset(&a, 0, sizeof a);
    a.sun_family = AF_UNIX;
    strncpy(a.sun_path, PATH, sizeof a.sun_path - 1);
    if (connect(s, (struct sockaddr *)&a, sizeof a) < 0) {
        close(s);
        return -1;
    }
    return s;
}

/* Send one request and collect the reply up to its final ok/err line.
 * Answers 0 for ok, 1 for err, -1 for a broken connection. */
static int ask(const char *req, char *out, size_t cap) {
    size_t n = strlen(req), got = 0;
    if (write(fd, req, n) != (ssize_t)n || write(fd, "\n", 1) != 1) return -1;
    out[0] = 0;
    for (;;) {
        ssize_t r = read(fd, out + got, cap - 1 - got);
        if (r <= 0) return -1;
        got += r;
        out[got] = 0;
        /* The reply ends at a line that is "ok" or begins "err ". */
        char *last = out + got;
        if (got >= 1 && out[got - 1] == '\n') {
            char *p = last - 1;
            while (p > out && p[-1] != '\n') p--;
            if (!strncmp(p, "ok\n", 3)) return 0;
            if (!strncmp(p, "err ", 4)) return 1;
        }
        if (got >= cap - 1) return 1;
    }
}

static long now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec;
}

static void say(const char *what) {
    printf("[netd] %s\n", what);
    fflush(stdout);
}

/* The field after "key=" in a line, copied into out. */
static void field(const char *line, const char *key, char *out, size_t cap) {
    const char *p = strstr(line, key);
    out[0] = 0;
    if (!p) return;
    p += strlen(key);
    size_t i = 0;
    while (*p && *p != ' ' && *p != '\n' && i + 1 < cap) out[i++] = *p++;
    out[i] = 0;
}

static int manage(void) {
    char buf[8192], last[256] = "", line[256];
    long asked_eth = -100, asked_wlan = -100, scanned = -100, joined = -100;
    /* DHCP backs off while nobody answers: 15 s, doubling, to five minutes. */
    long wait_eth = 15, wait_wlan = 15;
    if (ask("manage on", buf, sizeof buf) != 0) return 1;
    say("managing the network");
    for (;;) {
        if (ask("status", buf, sizeof buf) < 0) return 1;
        long t = now_s();
        char state[64] = "none", ssid[128] = "-", ip[32];
        int have_wifi = 0;
        for (char *l = strtok(buf, "\n"); l; l = strtok(NULL, "\n")) {
            if (!strncmp(l, "iface eth0", 10)) {
                field(l, "ip=", ip, sizeof ip);
                if (strstr(l, "link=1") && !strcmp(ip, "0.0.0.0") && t - asked_eth >= wait_eth) {
                    if (asked_eth > 0) wait_eth = wait_eth * 2 > 300 ? 300 : wait_eth * 2;
                    say("eth0 has a link and no address: asking DHCP");
                    asked_eth = t;
                    char r[256];
                    ask("dhcp eth0", r, sizeof r);
                }
            } else if (!strncmp(l, "iface wlan0", 11)) {
                field(l, "ip=", ip, sizeof ip);
                if (strstr(l, "link=1") && !strcmp(ip, "0.0.0.0") && t - asked_wlan >= wait_wlan) {
                    if (asked_wlan > 0) wait_wlan = wait_wlan * 2 > 300 ? 300 : wait_wlan * 2;
                    say("wlan0 has a link and no address: asking DHCP");
                    asked_wlan = t;
                    char r[256];
                    ask("dhcp wlan0", r, sizeof r);
                }
            } else if (!strncmp(l, "wifi ", 5) && strncmp(l, "wifi none", 9)) {
                have_wifi = 1;
                sscanf(l + 5, "%63s", state);
                field(l, "ssid=", ssid, sizeof ssid);
            }
        }
        if (strcmp(ip, "0.0.0.0")) wait_wlan = 15;
        snprintf(line, sizeof line, "wifi %s, network %s", state, ssid);
        if (have_wifi && strcmp(line, last)) {
            say(line);
            strcpy(last, line);
        }
        int busy = !strcmp(state, "scanning") || !strcmp(state, "authenticating") ||
                   !strcmp(state, "associating") || !strcmp(state, "handshaking") ||
                   !strcmp(state, "running");
        if (have_wifi && !busy && t - joined >= 10) {
            /* Strongest network heard that is also known. */
            char known[4096], nets[8192], best[128] = "";
            int best_bars = -1;
            if (ask("known", known, sizeof known) == 0 && ask("networks", nets, sizeof nets) == 0) {
                for (char *l = strtok(nets, "\n"); l; l = strtok(NULL, "\n")) {
                    int bars;
                    char kind[16], name[128];
                    if (sscanf(l, "net %d %15s %127[^\n]", &bars, kind, name) != 3) continue;
                    char want[160];
                    snprintf(want, sizeof want, "known %s\n", name);
                    if (strstr(known, want) && bars > best_bars) {
                        best_bars = bars;
                        strcpy(best, name);
                    }
                }
            }
            if (best[0]) {
                char req[200], r[256];
                snprintf(req, sizeof req, "join %s", best);
                snprintf(line, sizeof line, "joining %s (%d bars)", best, best_bars);
                say(line);
                ask(req, r, sizeof r);
                joined = t;
            } else if (t - scanned >= 30) {
                char r[256];
                ask("scan", r, sizeof r);
                scanned = t;
            }
        }
        sleep(2);
    }
}

int main(int argc, char **argv) {
    /* The endpoint exists once the kernel has served a guest; give it a
     * moment rather than failing a service that started first thing. */
    for (int i = 0; (fd = dial()) < 0; i++) {
        if (argc > 1 || i >= 30) {
            fprintf(stderr, "netd: cannot reach %s: %s\n", PATH, strerror(errno));
            return 2;
        }
        sleep(1);
    }
    if (argc == 1) return manage();
    char req[1024] = "", buf[16384];
    for (int i = 1; i < argc; i++) {
        if (i > 1) strncat(req, " ", sizeof req - strlen(req) - 1);
        strncat(req, argv[i], sizeof req - strlen(req) - 1);
    }
    int r = ask(req, buf, sizeof buf);
    fputs(buf, stdout);
    return r == 0 ? 0 : 1;
}

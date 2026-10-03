/* A Wayland client for Skywalker, built against the real libwayland-client.
 *
 * It does what every shared-memory client does and nothing else: binds
 * wl_compositor, wl_shm and xdg_wm_base, makes a toplevel, waits for its first
 * configure, and draws frames into a memfd buffer, paced by frame callbacks.
 * Each step is printed, so a transcript says how far a run got.
 *
 * Built by tools/sky.py, which generates the xdg-shell glue with
 * wayland-scanner. Usage inside GLaDOS: skytest [frames]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <wayland-client.h>
#include "xdg-shell.h"

#define W 320
#define H 200

static struct wl_compositor *compositor;
static struct wl_shm *shm;
static struct xdg_wm_base *wm;
static int configured, closed, frames, want = 90, done_frame = 1;
static uint32_t *pixels;
static struct wl_buffer *buffer;
static struct wl_surface *surface;

static void global(void *d, struct wl_registry *r, uint32_t name, const char *iface, uint32_t ver) {
    printf("skytest: global %u %s v%u\n", name, iface, ver);
    if (!strcmp(iface, "wl_compositor"))
        compositor = wl_registry_bind(r, name, &wl_compositor_interface, 4);
    else if (!strcmp(iface, "wl_shm"))
        shm = wl_registry_bind(r, name, &wl_shm_interface, 1);
    else if (!strcmp(iface, "xdg_wm_base"))
        wm = wl_registry_bind(r, name, &xdg_wm_base_interface, 1);
}
static void global_remove(void *d, struct wl_registry *r, uint32_t name) {}
static const struct wl_registry_listener reg_l = { global, global_remove };

static void ping(void *d, struct xdg_wm_base *b, uint32_t serial) { xdg_wm_base_pong(b, serial); }
static const struct xdg_wm_base_listener wm_l = { ping };

static void xs_configure(void *d, struct xdg_surface *s, uint32_t serial) {
    xdg_surface_ack_configure(s, serial);
    configured = 1;
}
static const struct xdg_surface_listener xs_l = { xs_configure };

static void tl_configure(void *d, struct xdg_toplevel *t, int32_t w, int32_t h, struct wl_array *st) {}
static void tl_close(void *d, struct xdg_toplevel *t) { closed = 1; }
static const struct xdg_toplevel_listener tl_l = { tl_configure, tl_close };

static void draw(int n) {
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            uint32_t r = (x + n * 2) & 0xFF, g = (y + n) & 0xFF, b = 0x80;
            /* A square that walks across, so motion is visible in a screenshot. */
            if (x > (n * 3) % (W - 40) && x < (n * 3) % (W - 40) + 40 && y > 80 && y < 120)
                r = g = b = 0xFF;
            pixels[y * W + x] = 0xFF000000u | (r << 16) | (g << 8) | b;
        }
}

static void frame_done(void *d, struct wl_callback *cb, uint32_t t);
static const struct wl_callback_listener frame_l = { frame_done };

static void submit(void) {
    draw(frames);
    wl_surface_attach(surface, buffer, 0, 0);
    wl_surface_damage_buffer(surface, 0, 0, W, H);
    struct wl_callback *cb = wl_surface_frame(surface);
    wl_callback_add_listener(cb, &frame_l, NULL);
    wl_surface_commit(surface);
    done_frame = 0;
}

static void frame_done(void *d, struct wl_callback *cb, uint32_t t) {
    wl_callback_destroy(cb);
    frames++;
    done_frame = 1;
    if (frames % 30 == 0)
        printf("skytest: frame %d at %u ms\n", frames, t);
}

int main(int argc, char **argv) {
    if (argc > 1) want = atoi(argv[1]);
    struct wl_display *dpy = wl_display_connect(NULL);
    if (!dpy) { printf("skytest: no display\n"); return 1; }
    printf("skytest: connected\n");
    struct wl_registry *reg = wl_display_get_registry(dpy);
    wl_registry_add_listener(reg, &reg_l, NULL);
    wl_display_roundtrip(dpy);
    if (!compositor || !shm || !wm) { printf("skytest: missing a global\n"); return 2; }
    xdg_wm_base_add_listener(wm, &wm_l, NULL);

    int fd = memfd_create("skytest", MFD_CLOEXEC);
    if (fd < 0 || ftruncate(fd, W * H * 4) < 0) { printf("skytest: no memfd\n"); return 3; }
    pixels = mmap(NULL, W * H * 4, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (pixels == MAP_FAILED) { printf("skytest: no mapping\n"); return 4; }
    struct wl_shm_pool *pool = wl_shm_create_pool(shm, fd, W * H * 4);
    buffer = wl_shm_pool_create_buffer(pool, 0, W, H, W * 4, WL_SHM_FORMAT_XRGB8888);
    printf("skytest: buffer %dx%d\n", W, H);

    surface = wl_compositor_create_surface(compositor);
    struct xdg_surface *xs = xdg_wm_base_get_xdg_surface(wm, surface);
    xdg_surface_add_listener(xs, &xs_l, NULL);
    struct xdg_toplevel *tl = xdg_surface_get_toplevel(xs);
    xdg_toplevel_add_listener(tl, &tl_l, NULL);
    xdg_toplevel_set_title(tl, "Skywalker");
    wl_surface_commit(surface);
    while (!configured && wl_display_dispatch(dpy) != -1) {}
    printf("skytest: configured\n");

    while (frames < want && !closed) {
        if (done_frame) submit();
        if (wl_display_dispatch(dpy) == -1) { printf("skytest: connection lost\n"); return 5; }
    }
    printf("skytest: %d frame(s)%s\n", frames, closed ? ", closed by the desktop" : "");
    xdg_toplevel_destroy(tl);
    xdg_surface_destroy(xs);
    wl_surface_destroy(surface);
    wl_display_roundtrip(dpy);
    wl_display_disconnect(dpy);
    return 0;
}

/* A GL 1.1 program for GLaDOS: immediate mode, per-vertex colour, a depth
 * buffer, and the result written straight to the framebuffer.
 *
 * This exists because a compiler arrived. The hand-assembled fixture that
 * proved Mesa runs could clear the screen and nothing else -- `glBegin` and
 * `glVertex3f` in a loop is not something you write with `mov_imm`. Two
 * hundred and fifty lines of x86 became sixty lines of C, and the C can
 * actually draw.
 *
 * No headers are included for GL. `GL/osmesa.h` is not on this host and the
 * dozen prototypes below are the whole of what is used, so declaring them is
 * shorter than fetching a package and cannot drift from what is linked --
 * the linker checks these against the real `libOSMesa` at build time.
 *
 * Build:
 *   zig cc -target x86_64-linux-gnu -O2 tools/guest/gldemo.c \
 *       out/gl/gl/usr/lib/x86_64-linux-gnu/libOSMesa.so.8 -o out/guest/gldemo
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

/* linux/fb.h spells these as bare constants rather than through the _IOC
 * macros, which cost a run to find out: the encoded form is answered with
 * ENOTTY and the geometry stays zero. */
#define FBIOGET_VSCREENINFO 0x4600

/* The fields this program reads, at the offsets the ABI fixes them at. Read
 * as a byte array rather than as a struct, because `fb_var_screeninfo` is 160
 * bytes of which four are wanted and a partial declaration would be a
 * different structure. */
#define VAR_LEN 160
#define VAR_YRES 4
#define VAR_XRES_VIRTUAL 8 /* the stride in pixels, not the visible width */

/* OSMesa, and the one argument that matters is the first. Asking for BGRA
 * makes Mesa's byte order the display's own, so the blit is a copy rather
 * than a conversion. */
#define OSMESA_BGRA 0x1
#define OSMESA_Y_UP 0x11
#define GL_UNSIGNED_BYTE 0x1401
#define GL_COLOR_BUFFER_BIT 0x00004000
#define GL_DEPTH_BUFFER_BIT 0x00000100
#define GL_DEPTH_TEST 0x0B71
#define GL_TRIANGLES 0x0004
#define GL_PROJECTION 0x1701
#define GL_MODELVIEW 0x1700

typedef struct osmesa_context *OSMesaContext;
extern OSMesaContext OSMesaCreateContext(unsigned f, OSMesaContext share);
extern int OSMesaMakeCurrent(OSMesaContext c, void *buf, unsigned type,
                             int w, int h);
extern void OSMesaPixelStore(int name, int value);
extern void OSMesaDestroyContext(OSMesaContext c);
extern const unsigned char *glGetString(unsigned name);
extern void glViewport(int x, int y, int w, int h);
extern void glClearColor(float r, float g, float b, float a);
extern void glClear(unsigned mask);
extern void glEnable(unsigned cap);
extern void glMatrixMode(unsigned mode);
extern void glLoadIdentity(void);
extern void glOrtho(double l, double r, double b, double t, double n, double f);
extern void glRotatef(float a, float x, float y, float z);
extern void glBegin(unsigned mode);
extern void glEnd(void);
extern void glColor3f(float r, float g, float b);
extern void glVertex3f(float x, float y, float z);
extern void glFinish(void);

#define GL_VENDOR 0x1F00
#define GL_RENDERER 0x1F01
#define GL_VERSION 0x1F02

static unsigned rd32(const unsigned char *b, int off) {
    unsigned v;
    memcpy(&v, b + off, 4);
    return v;
}

int main(int argc, char **argv) {
    unsigned char var[VAR_LEN];
    int fb = open("/dev/fb0", O_RDWR);
    if (fb < 0) {
        printf("no /dev/fb0\n");
        return 1;
    }
    if (ioctl(fb, FBIOGET_VSCREENINFO, var) != 0) {
        printf("no geometry\n");
        return 2;
    }
    int w = (int)rd32(var, VAR_XRES_VIRTUAL);
    int h = (int)rd32(var, VAR_YRES);
    size_t bytes = (size_t)w * h * 4;
    printf("%dx%d, %zu bytes\n", w, h, bytes);

    void *buf = mmap(0, bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) {
        printf("no buffer\n");
        return 3;
    }

    OSMesaContext ctx = OSMesaCreateContext(OSMESA_BGRA, 0);
    if (!ctx || !OSMesaMakeCurrent(ctx, buf, GL_UNSIGNED_BYTE, w, h)) {
        printf("no context\n");
        return 4;
    }
    /* GL's origin is bottom-left and a framebuffer's is top-left, so without
     * this the picture is upside down -- which on a symmetric scene is a bug
     * nobody notices until the text is backwards. */
    OSMesaPixelStore(OSMESA_Y_UP, 0);

    printf("%s | %s | %s\n", glGetString(GL_VENDOR), glGetString(GL_RENDERER),
           glGetString(GL_VERSION));

    glViewport(0, 0, w, h);
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double aspect = (double)w / (double)h;
    glOrtho(-aspect, aspect, -1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glClearColor(0.04f, 0.06f, 0.10f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /* Three triangles at three angles, each with per-vertex colour, so the
     * interpolation across a face is visible. A single flat triangle would
     * prove the rasteriser fills and nothing about whether it shades. */
    for (int i = 0; i < 3; i++) {
        glLoadIdentity();
        glRotatef(120.0f * i, 0.0f, 0.0f, 1.0f);
        glBegin(GL_TRIANGLES);
        glColor3f(1.0f, 0.35f, 0.0f);
        glVertex3f(0.0f, 0.85f, 0.0f);
        glColor3f(0.0f, 0.75f, 0.95f);
        glVertex3f(-0.75f, -0.5f, 0.0f);
        glColor3f(0.95f, 0.95f, 0.95f);
        glVertex3f(0.75f, -0.5f, 0.0f);
        glEnd();
    }
    glFinish();

    ssize_t n = write(fb, buf, bytes);
    printf("wrote %zd of %zu\n", n, bytes);

    /* Held only when asked, because the harness photographs what is on screen
     * after the guest has gone and teardown puts the desktop back. The exit
     * path for a held run is the timeout. */
    if (argc > 1) {
        struct timespec t = {600, 0};
        nanosleep(&t, 0);
    }
    OSMesaDestroyContext(ctx);
    return n == (ssize_t)bytes ? 0 : 5;
}

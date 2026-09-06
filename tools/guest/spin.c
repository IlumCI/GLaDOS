/* Everything at once: SDL2 gets the window, Mesa draws into it, the `glados`
 * video driver puts it on the framebuffer.
 *
 * **A solid under perspective, and that is the point of the shape.** The first
 * version of this turned three flat triangles about Z, which proves the
 * rasteriser fills and interpolates and says nothing whatever about a third
 * dimension: an orthographic Z rotation is a 2D rotation with extra words, and
 * it would look identical on an engine with no depth buffer, no frustum and a
 * matrix stack that only ever multiplied two of three axes.
 *
 * A square pyramid tilted and spun about its own Y needs all of it, and each
 * piece fails visibly rather than subtly:
 *
 *   - **`glFrustum`**, so the near base corner is genuinely larger than the far
 *     one. Under `glOrtho` a spinning pyramid reads as a flat quadrilateral
 *     changing shape, which is what the old demo could not have told you.
 *   - **The depth buffer.** Four faces meet at the apex and two of them face
 *     away at any moment. Without depth testing the last one drawn wins, so
 *     the far face paints over the near one for half of every turn -- a
 *     flicker, not an error.
 *   - **The full modelview stack**, since translate-then-tilt-then-spin is
 *     three matrices composed in an order that is wrong in an obvious way if
 *     composed backwards: the pyramid orbits the camera instead of turning in
 *     place.
 *
 * Culling is deliberately off. The solid is closed and convex, so back faces
 * are invisible either way and enabling it would change no pixel while adding
 * a winding convention that can be quietly wrong.
 *
 * Mesa renders straight into the SDL window's own surface. They agree about
 * the format by construction -- `SDL_PIXELFORMAT_RGB888` is `0xXXRRGGBB`,
 * which in memory is B, G, R, unused, and `OSMESA_BGRA` is the same bytes --
 * so there is no conversion anywhere between `glVertex3f` and the display.
 *
 * Build:
 *   zig cc -target x86_64-linux-gnu -O2 -fPIE -pie tools/guest/spin.c \
 *     -Iout/sdl/SDL2-2.30.12/include out/guest/libSDL2-2.0.so.0 \
 *     out/gl/gl/usr/lib/x86_64-linux-gnu/libOSMesa.so.8 -o out/guest/spin
 */

#include <stdio.h>
#include "SDL.h"

#define OSMESA_BGRA 0x1
#define OSMESA_Y_UP 0x11
#define GL_UNSIGNED_BYTE 0x1401
#define GL_COLOR_BUFFER_BIT 0x00004000
#define GL_DEPTH_BUFFER_BIT 0x00000100
#define GL_DEPTH_TEST 0x0B71
#define GL_TRIANGLES 0x0004
#define GL_PROJECTION 0x1701
#define GL_MODELVIEW 0x1700
#define GL_SMOOTH 0x1D01
#define GL_VERSION 0x1F02
#define GL_RENDERER 0x1F01

typedef struct osmesa_context *OSMesaContext;
extern OSMesaContext OSMesaCreateContext(unsigned f, OSMesaContext share);
/* The Ext form because the depth buffer is half the demonstration and the
 * plain one leaves its size to Mesa. Asking for it is how a context with no
 * depth bits becomes a refusal here rather than a pyramid that renders
 * inside out. */
extern OSMesaContext OSMesaCreateContextExt(unsigned f, int depth, int stencil,
                                            int accum, OSMesaContext share);
extern int OSMesaMakeCurrent(OSMesaContext c, void *b, unsigned t, int w, int h);
extern void OSMesaPixelStore(int name, int value);
extern const unsigned char *glGetString(unsigned name);
extern void glViewport(int x, int y, int w, int h);
extern void glClearColor(float r, float g, float b, float a);
extern void glClear(unsigned mask);
extern void glEnable(unsigned cap);
extern void glShadeModel(unsigned mode);
extern void glMatrixMode(unsigned mode);
extern void glLoadIdentity(void);
extern void glFrustum(double l, double r, double b, double t, double n, double f);
extern void glTranslatef(float x, float y, float z);
extern void glRotatef(float a, float x, float y, float z);
extern void glBegin(unsigned mode);
extern void glEnd(void);
extern void glColor3f(float r, float g, float b);
extern void glVertex3f(float x, float y, float z);
extern void glFinish(void);

/* Apex plus four base corners. The apex is warm and each corner is its own
 * hue, so every one of the four side faces is a *different* gradient -- which
 * is what makes the rotation legible frame to frame, and what would show a
 * matrix stack rotating the geometry while leaving the colours behind. */
static const float APEX[3] = {0.00f, 0.95f, 0.00f};
static const float BASE[4][3] = {
    {-0.85f, -0.55f, -0.85f},
    { 0.85f, -0.55f, -0.85f},
    { 0.85f, -0.55f,  0.85f},
    {-0.85f, -0.55f,  0.85f},
};
static const float APEX_C[3] = {1.00f, 0.72f, 0.30f};
static const float BASE_C[4][3] = {
    {0.05f, 0.55f, 0.95f},
    {0.95f, 0.30f, 0.15f},
    {0.15f, 0.80f, 0.45f},
    {0.75f, 0.35f, 0.90f},
};

static void corner(int i)
{
    glColor3f(BASE_C[i][0], BASE_C[i][1], BASE_C[i][2]);
    glVertex3f(BASE[i][0], BASE[i][1], BASE[i][2]);
}

static void pyramid(void)
{
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < 4; i++) {
        glColor3f(APEX_C[0], APEX_C[1], APEX_C[2]);
        glVertex3f(APEX[0], APEX[1], APEX[2]);
        corner(i);
        corner((i + 1) % 4);
    }
    /* The underside, at a quarter brightness. It is only ever visible when the
     * tilt carries it into view, so a base drawn as bright as the sides would
     * make "we are looking at the bottom" indistinguishable from "we are
     * looking at a face" -- and that distinction is exactly what the depth
     * buffer is deciding. */
    for (int t = 0; t < 2; t++) {
        int idx[3] = {0, t + 1, t + 2};
        for (int k = 0; k < 3; k++) {
            int i = idx[k];
            glColor3f(BASE_C[i][0] * 0.25f, BASE_C[i][1] * 0.25f,
                      BASE_C[i][2] * 0.25f);
            glVertex3f(BASE[i][0], BASE[i][1], BASE[i][2]);
        }
    }
    glEnd();
}

int main(int argc, char **argv)
{
    int seconds = argc > 1 ? SDL_atoi(argv[1]) : 30;

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
    SDL_DisplayMode mode;
    SDL_GetCurrentDisplayMode(0, &mode);
    SDL_Window *win = SDL_CreateWindow("spin", 0, 0, mode.w, mode.h,
                                       SDL_WINDOW_FULLSCREEN);
    SDL_Surface *s = SDL_GetWindowSurface(win);
    if (!win || !s) {
        printf("no window: %s\n", SDL_GetError());
        return 2;
    }

    /* 16 bits of depth, no stencil, no accumulation. */
    OSMesaContext ctx = OSMesaCreateContextExt(OSMESA_BGRA, 16, 0, 0, 0);
    /* Straight into SDL's surface. No intermediate buffer, because the two
     * agree about the byte order and the pitch. */
    int vw = s->pitch / 4, vh = s->h;
    if (!ctx || !OSMesaMakeCurrent(ctx, s->pixels, GL_UNSIGNED_BYTE, vw, vh)) {
        printf("no context\n");
        return 3;
    }
    /* GL's origin is bottom-left, a framebuffer's is top-left. On a scene
     * with an apex at the top this is the difference between right and
     * upside down. */
    OSMesaPixelStore(OSMESA_Y_UP, 0);

    printf("%s | %s | SDL %s\n", glGetString(GL_RENDERER),
           glGetString(GL_VERSION), SDL_GetCurrentVideoDriver());
    printf("%dx%d, %d seconds\n", s->w, s->h, seconds);

    glViewport(0, 0, vw, vh);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);

    /* A 50-degree vertical field of view, written out rather than computed:
     * tan(25 degrees) is 0.4663, and hardcoding it costs one comment and saves
     * depending on libm resolving inside the guest. */
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double top = 0.4663, near = 1.0;
    double right = top * ((double)vw / (double)vh);
    glFrustum(-right, right, -top, top, near, 20.0);
    glMatrixMode(GL_MODELVIEW);

    Uint32 start = SDL_GetTicks();
    Uint32 frames = 0;
    for (;;) {
        Uint32 now = SDL_GetTicks();
        if ((now - start) > (Uint32)seconds * 1000) {
            break;
        }
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                goto done;
            }
        }

        float angle = (float)(now - start) * 0.05f;
        glClearColor(0.03f, 0.05f, 0.09f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /* Read bottom-up: spin about its own Y, tilt the result, push it away
         * from the eye. Composed the other way round the pyramid orbits the
         * camera, which is the loud failure this ordering has. */
        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -3.2f);
        glRotatef(22.0f, 1.0f, 0.0f, 0.0f);
        glRotatef(angle, 0.0f, 1.0f, 0.0f);
        pyramid();

        glFinish();
        SDL_UpdateWindowSurface(win);
        frames++;
    }
done:
    {
        Uint32 ms = SDL_GetTicks() - start;
        printf("%u frames in %u ms", frames, ms);
        if (ms) {
            printf(", %u.%02u fps", frames * 1000 / ms,
                   (frames * 100000 / ms) % 100);
        }
        printf("\n");
    }
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}

/* Everything at once: SDL2 gets the window, Mesa draws into it, the `glados`
 * video driver puts it on the framebuffer.
 *
 * The point of this one is that it *moves*. Every still picture so far proves
 * a frame reached the display and says nothing about whether a second one
 * can: a program that renders once and a program that renders sixty times
 * look identical in a screenshot. A spinning triangle is the smallest thing
 * that cannot be faked by a single blit.
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
#define GL_VERSION 0x1F02
#define GL_RENDERER 0x1F01

typedef struct osmesa_context *OSMesaContext;
extern OSMesaContext OSMesaCreateContext(unsigned f, OSMesaContext share);
extern int OSMesaMakeCurrent(OSMesaContext c, void *b, unsigned t, int w, int h);
extern void OSMesaPixelStore(int name, int value);
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

    OSMesaContext ctx = OSMesaCreateContext(OSMESA_BGRA, 0);
    /* Straight into SDL's surface. No intermediate buffer, because the two
     * agree about the byte order and the pitch. */
    if (!ctx || !OSMesaMakeCurrent(ctx, s->pixels, GL_UNSIGNED_BYTE,
                                   s->pitch / 4, s->h)) {
        printf("no context\n");
        return 3;
    }
    /* GL's origin is bottom-left, a framebuffer's is top-left. On a scene
     * with a triangle at the top this is the difference between right and
     * upside down. */
    OSMesaPixelStore(OSMESA_Y_UP, 0);

    printf("%s | %s | SDL %s\n", glGetString(GL_RENDERER),
           glGetString(GL_VERSION), SDL_GetCurrentVideoDriver());
    printf("%dx%d, %d seconds\n", s->w, s->h, seconds);

    glViewport(0, 0, s->pitch / 4, s->h);
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double aspect = (double)s->w / (double)s->h;
    glOrtho(-aspect, aspect, -1.0, 1.0, -1.0, 1.0);
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

        float angle = (float)(now - start) * 0.09f;
        glClearColor(0.03f, 0.05f, 0.09f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /* Three of them, a third of a turn apart, each gouraud shaded. One
         * flat triangle would show the rasteriser fills; this shows it
         * interpolates, and that the matrix stack works. */
        for (int i = 0; i < 3; i++) {
            glLoadIdentity();
            glRotatef(angle + 120.0f * i, 0.0f, 0.0f, 1.0f);
            glBegin(GL_TRIANGLES);
            glColor3f(1.0f, 0.42f, 0.05f);
            glVertex3f(0.0f, 0.80f, 0.0f);
            glColor3f(0.05f, 0.65f, 0.95f);
            glVertex3f(-0.69f, -0.40f, 0.0f);
            glColor3f(0.97f, 0.97f, 0.97f);
            glVertex3f(0.69f, -0.40f, 0.0f);
            glEnd();
        }
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

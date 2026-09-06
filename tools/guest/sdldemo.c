/* An unmodified SDL2 program, on a machine with no display server.
 *
 * Everything here is what you would write on any Linux desktop: `SDL_Init`,
 * `SDL_CreateWindow`, `SDL_GetWindowSurface`, draw, `SDL_UpdateWindowSurface`.
 * Nothing in it knows about `/dev/fb0`, and that is the claim -- the `glados`
 * video driver is what turns the last of those into a write to the display,
 * and the program is not aware there is anything unusual underneath.
 *
 * The picture is deliberately not a solid fill. A single colour proves the
 * surface reached the screen and nothing about whether the *geometry* did: a
 * driver that got the stride wrong shows a solid colour perfectly and shears
 * everything else. Bars plus a diagonal make a stride error obvious on sight.
 *
 * Build:
 *   zig cc -target x86_64-linux-gnu -O2 -fPIE -pie tools/guest/sdldemo.c \
 *       -Iout/sdl/SDL2-2.30.12/include out/guest/libSDL2-2.0.so.0 -o out/guest/sdldemo
 */

#include <stdio.h>
#include "SDL.h"

int main(int argc, char **argv)
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
    printf("video driver: %s\n", SDL_GetCurrentVideoDriver());

    SDL_DisplayMode mode;
    if (SDL_GetCurrentDisplayMode(0, &mode) != 0) {
        printf("no display mode: %s\n", SDL_GetError());
        return 2;
    }
    printf("display: %dx%d, format %s\n", mode.w, mode.h,
           SDL_GetPixelFormatName(mode.format));

    SDL_Window *win = SDL_CreateWindow("glados", SDL_WINDOWPOS_UNDEFINED,
                                       SDL_WINDOWPOS_UNDEFINED, mode.w, mode.h,
                                       SDL_WINDOW_FULLSCREEN);
    if (!win) {
        printf("no window: %s\n", SDL_GetError());
        return 3;
    }
    SDL_Surface *s = SDL_GetWindowSurface(win);
    if (!s) {
        printf("no surface: %s\n", SDL_GetError());
        return 4;
    }
    printf("surface: %dx%d pitch %d\n", s->w, s->h, s->pitch);

    /* Vertical bars, then a diagonal across them. The bars alone would look
     * right under a wrong stride as long as it were a multiple of the bar
     * width; the diagonal cannot, because a stride error turns a straight
     * line into a staircase that leans. */
    const int bars = 8;
    for (int i = 0; i < bars; i++) {
        SDL_Rect r = { i * s->w / bars, 0, s->w / bars, s->h };
        Uint8 v = (Uint8)(255 * i / (bars - 1));
        SDL_FillRect(s, &r, SDL_MapRGB(s->format, v, (Uint8)(255 - v), 128));
    }
    for (int x = 0; x < s->w; x++) {
        int y = x * s->h / s->w;
        SDL_Rect d = { x, y, 1, 3 };
        SDL_FillRect(s, &d, SDL_MapRGB(s->format, 255, 255, 255));
    }

    if (SDL_UpdateWindowSurface(win) != 0) {
        printf("no update: %s\n", SDL_GetError());
        return 5;
    }
    printf("frame on screen\n");

    if (argc > 1) {
        SDL_Delay(600000);
    }
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}

/* Where an SDL window's pixels actually go.
 *
 * SDL's dummy driver allocates a surface the size of the window and then
 * throws every frame away; this is that, with the throwing away replaced by a
 * write to the display. Everything above -- `SDL_CreateRenderer` on the
 * software backend, `SDL_UpdateWindowSurface`, blits, the whole 2D stack -- is
 * SDL's own and does not know the difference.
 */

#include "../../SDL_internal.h"

#ifdef SDL_VIDEO_DRIVER_GLADOS

#include "../SDL_sysvideo.h"
#include "SDL_gladosvideo.h"
#include "SDL_gladosframebuffer_c.h"

#include <unistd.h>

#define GLADOS_SURFACE "_SDL_GLaDOSSurface"

int SDL_GLADOS_CreateWindowFramebuffer(_THIS, SDL_Window *window, Uint32 *format,
                                       void **pixels, int *pitch)
{
    SDL_Surface *surface;
    /* Named by the value read as a word, so this is 0xXXRRGGBB, which in
     * memory is B, G, R, unused -- the `Bgrx` the display reports. Naming the
     * byte order instead would swap red and blue with no error anywhere. */
    const Uint32 surface_format = SDL_PIXELFORMAT_RGB888;
    int w, h;

    SDL_GLADOS_DestroyWindowFramebuffer(_this, window);

    SDL_GetWindowSizeInPixels(window, &w, &h);
    surface = SDL_CreateRGBSurfaceWithFormat(0, w, h, 0, surface_format);
    if (!surface) {
        return -1;
    }

    SDL_SetWindowData(window, GLADOS_SURFACE, surface);
    *format = surface_format;
    *pixels = surface->pixels;
    *pitch = surface->pitch;
    return 0;
}

int SDL_GLADOS_UpdateWindowFramebuffer(_THIS, SDL_Window *window,
                                       const SDL_Rect *rects, int numrects)
{
    SDL_GLADOS_Data *data = (SDL_GLADOS_Data *)_this->driverdata;
    SDL_Surface *surface;
    const Uint8 *src;
    Uint32 rows, copy, line;

    (void)rects;
    (void)numrects;

    surface = (SDL_Surface *)SDL_GetWindowData(window, GLADOS_SURFACE);
    if (!surface) {
        return SDL_SetError("no surface for this window");
    }
    if (!data || data->fd < 0) {
        return SDL_SetError("no display");
    }

    src = (const Uint8 *)surface->pixels;
    rows = (Uint32)surface->h < data->h ? (Uint32)surface->h : data->h;
    line = data->stride * 4;                       /* bytes per scan line */
    copy = (Uint32)surface->pitch < line ? (Uint32)surface->pitch : line;

    /* **The whole surface in one call when the shapes agree**, which is the
     * ordinary case: a full-screen window's pitch is the display's stride and
     * one `write` of four megabytes beats eight hundred of five kilobytes.
     * The row loop is for a window narrower than the screen, where the two
     * strides genuinely differ and a single write would shear the picture. */
    if ((Uint32)surface->pitch == line && rows == data->h) {
        ssize_t want = (ssize_t)line * (ssize_t)rows;
        if (lseek(data->fd, 0, SEEK_SET) < 0) {
            return SDL_SetError("cannot rewind the display");
        }
        if (write(data->fd, src, (size_t)want) != want) {
            return SDL_SetError("short write to the display");
        }
        return 0;
    }

    for (Uint32 y = 0; y < rows; y++) {
        if (lseek(data->fd, (off_t)y * (off_t)line, SEEK_SET) < 0) {
            return SDL_SetError("cannot seek the display");
        }
        if (write(data->fd, src + (size_t)y * (size_t)surface->pitch, copy) != (ssize_t)copy) {
            return SDL_SetError("short write to the display");
        }
    }
    return 0;
}

void SDL_GLADOS_DestroyWindowFramebuffer(_THIS, SDL_Window *window)
{
    SDL_Surface *surface;

    surface = (SDL_Surface *)SDL_SetWindowData(window, GLADOS_SURFACE, NULL);
    SDL_FreeSurface(surface);
}

#endif /* SDL_VIDEO_DRIVER_GLADOS */

/* vi: set ts=4 sw=4 expandtab: */

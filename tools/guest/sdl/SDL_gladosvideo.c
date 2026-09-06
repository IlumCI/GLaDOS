/* An SDL2 video driver for GLaDOS: one framebuffer, no window manager.
 *
 * **SDL2 has no framebuffer backend and that is the whole reason this file
 * exists.** Its video drivers are X11, Wayland, KMSDRM, offscreen and dummy;
 * fbdev was an SDL 1.2 thing and did not survive the rewrite. So a machine
 * with a framebuffer and no display server cannot run an unmodified SDL2
 * program, however good its `/dev/fb0` is.
 *
 * This is SDL's own dummy driver with the one function that matters filled
 * in. `SDL_nullframebuffer.c` already allocates a surface of the window's
 * size and hands back its pixels; `UpdateWindowFramebuffer` there does
 * nothing at all, and here it writes to the display. Everything else --
 * window creation, the surface, the software renderer above it -- is SDL's
 * and is untouched.
 *
 * Two details decide whether the picture is right:
 *
 * - **`SDL_PIXELFORMAT_RGB888` is what this display wants.** SDL names pixel
 *   formats by their value read as a 32-bit word, so `RGB888` is `0xXXRRGGBB`,
 *   which in memory on a little-endian machine is B, G, R, unused -- exactly
 *   the `Bgrx` the framebuffer reports. Naming the byte order instead
 *   (`BGR888`) would swap red and blue, with no error anywhere.
 * - **The window's pitch is not the display's stride.** SDL sizes a surface
 *   to the window; the framebuffer has a `line_length` that may be wider. The
 *   blit goes a row at a time when they differ, and in one `write` when they
 *   do not, which is the ordinary case and the fast one.
 */

#include "../../SDL_internal.h"

#ifdef SDL_VIDEO_DRIVER_GLADOS

#include "SDL_video.h"
#include "SDL_mouse.h"
#include "../SDL_sysvideo.h"
#include "../SDL_pixels_c.h"
#include "../../events/SDL_events_c.h"

#include "SDL_gladosvideo.h"
#include "SDL_gladosframebuffer_c.h"
#include "../dummy/SDL_nullevents_c.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define GLADOSVID_DRIVER_NAME "glados"

/* `linux/fb.h` spells these as bare constants rather than through the `_IOC`
 * macros, which is unusual enough that the encoded form is the natural
 * mistake -- and it is answered with ENOTTY, leaving the geometry zero. */
#define FBIOGET_VSCREENINFO 0x4600
#define VAR_XRES 0
#define VAR_YRES 4
#define VAR_XRES_VIRTUAL 8
#define VAR_LEN 160

static int GLADOS_VideoInit(_THIS);
static void GLADOS_VideoQuit(_THIS);

static void GLADOS_DeleteDevice(SDL_VideoDevice *device)
{
    SDL_GLADOS_Data *data = (SDL_GLADOS_Data *)device->driverdata;
    if (data) {
        if (data->fd >= 0) {
            close(data->fd);
        }
        SDL_free(data);
    }
    SDL_free(device);
}

static SDL_VideoDevice *GLADOS_CreateDevice(void)
{
    SDL_VideoDevice *device;
    SDL_GLADOS_Data *data;

    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        SDL_OutOfMemory();
        return 0;
    }
    data = (SDL_GLADOS_Data *)SDL_calloc(1, sizeof(SDL_GLADOS_Data));
    if (!data) {
        SDL_free(device);
        SDL_OutOfMemory();
        return 0;
    }
    data->fd = -1;
    device->driverdata = data;

    device->VideoInit = GLADOS_VideoInit;
    device->VideoQuit = GLADOS_VideoQuit;
    device->PumpEvents = DUMMY_PumpEvents;
    device->CreateWindowFramebuffer = SDL_GLADOS_CreateWindowFramebuffer;
    device->UpdateWindowFramebuffer = SDL_GLADOS_UpdateWindowFramebuffer;
    device->DestroyWindowFramebuffer = SDL_GLADOS_DestroyWindowFramebuffer;
    device->free = GLADOS_DeleteDevice;

    return device;
}

/* Available only when the display really is there. A driver that reported
 * itself present and then failed at the first frame would be chosen ahead of
 * the dummy, which is the one thing worse than not existing. */
static SDL_bool GLADOS_Available(void)
{
    int fd = open("/dev/fb0", O_RDWR);
    if (fd < 0) {
        return SDL_FALSE;
    }
    close(fd);
    return SDL_TRUE;
}

VideoBootStrap GLADOS_bootstrap = {
    GLADOSVID_DRIVER_NAME, "GLaDOS framebuffer",
    GLADOS_CreateDevice,
    NULL /* no ShowMessageBox */
};

static int GLADOS_VideoInit(_THIS)
{
    SDL_GLADOS_Data *data = (SDL_GLADOS_Data *)_this->driverdata;
    Uint8 var[VAR_LEN];
    SDL_DisplayMode mode;

    data->fd = open("/dev/fb0", O_RDWR);
    if (data->fd < 0) {
        return SDL_SetError("no /dev/fb0");
    }
    if (ioctl(data->fd, FBIOGET_VSCREENINFO, var) != 0) {
        close(data->fd);
        data->fd = -1;
        return SDL_SetError("/dev/fb0 will not describe itself");
    }
    SDL_memcpy(&data->w, var + VAR_XRES, 4);
    SDL_memcpy(&data->h, var + VAR_YRES, 4);
    SDL_memcpy(&data->stride, var + VAR_XRES_VIRTUAL, 4);

    SDL_zero(mode);
    /* Read as a word rather than as bytes: see the note at the top. */
    mode.format = SDL_PIXELFORMAT_RGB888;
    mode.w = (int)data->w;
    mode.h = (int)data->h;
    mode.refresh_rate = 0;
    if (SDL_AddBasicVideoDisplay(&mode) < 0) {
        return -1;
    }
    return 0;
}

static void GLADOS_VideoQuit(_THIS)
{
    SDL_GLADOS_Data *data = (SDL_GLADOS_Data *)_this->driverdata;
    if (data && data->fd >= 0) {
        close(data->fd);
        data->fd = -1;
    }
}

#endif /* SDL_VIDEO_DRIVER_GLADOS */

/* vi: set ts=4 sw=4 expandtab: */

/* Shared state for the GLaDOS video driver: the display and its shape. */
#ifndef SDL_gladosvideo_h_
#define SDL_gladosvideo_h_

#include "../../SDL_internal.h"
#include "../SDL_sysvideo.h"

typedef struct
{
    int fd;
    /* Visible size, and the stride in *pixels*, which is not the same number.
     * Using the width where the stride belongs gives a picture that shears
     * one row further left on every line, which `gfx` records having paid for
     * from the other side of the same seam. */
    Uint32 w, h, stride;
} SDL_GLADOS_Data;

extern VideoBootStrap GLADOS_bootstrap;

#endif /* SDL_gladosvideo_h_ */

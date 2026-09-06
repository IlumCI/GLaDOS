#include "../../SDL_internal.h"

extern int SDL_GLADOS_CreateWindowFramebuffer(_THIS, SDL_Window *window,
                                              Uint32 *format, void **pixels,
                                              int *pitch);
extern int SDL_GLADOS_UpdateWindowFramebuffer(_THIS, SDL_Window *window,
                                              const SDL_Rect *rects,
                                              int numrects);
extern void SDL_GLADOS_DestroyWindowFramebuffer(_THIS, SDL_Window *window);

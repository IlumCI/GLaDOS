/*
 * glinfo -- which GL a program gets, and how fast it draws.
 *
 * Opens an SDL2 GL window the way Source does, prints the four strings that
 * say which driver answered, then draws a few thousand textured-free
 * triangles a frame for a fixed time and reports frames per second. Small
 * enough to tell "zink is not being picked" from "the game is slow" without
 * waiting five minutes for a map to load.
 *
 *   glinfo [seconds] [width height]
 *
 * Built 32-bit, against the SDL2 the game disk carries:
 *   gcc -m32 -O2 -I/usr/include/SDL2 -o out/glinfo32 tools/guest/glinfo.c \
 *       <dir with libSDL2-2.0.so.0> -lGL -ldl
 */
#include <SDL.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    // Unbuffered: a guest's stdout is a pipe to the console, so a buffered
    // line would appear only when the program exits -- after the frames it
    // is reporting on.
    setvbuf(stdout, NULL, _IONBF, 0);
    double secs = argc > 1 ? atof(argv[1]) : 5;
    int w = argc > 3 ? atoi(argv[2]) : 1280, h = argc > 3 ? atoi(argv[3]) : 800;
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("SDL_Init: %s\n", SDL_GetError());
        return 2;
    }
    printf("video driver: %s\n", SDL_GetCurrentVideoDriver());
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_Window *win = SDL_CreateWindow("glinfo", 0, 0, w, h, SDL_WINDOW_OPENGL);
    if (!win) {
        printf("CreateWindow: %s\n", SDL_GetError());
        return 2;
    }
    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    if (!ctx) {
        printf("CreateContext: %s\n", SDL_GetError());
        return 2;
    }
    printf("GL_VENDOR    %s\n", glGetString(GL_VENDOR));
    printf("GL_RENDERER  %s\n", glGetString(GL_RENDERER));
    printf("GL_VERSION   %s\n", glGetString(GL_VERSION));
    SDL_GL_SetSwapInterval(0);

    glEnable(GL_DEPTH_TEST);
    Uint64 f = SDL_GetPerformanceFrequency(), t0 = SDL_GetPerformanceCounter();
    int frames = 0;
    for (;;) {
        double t = (double)(SDL_GetPerformanceCounter() - t0) / f;
        if (t >= secs) break;
        SDL_Event e;
        while (SDL_PollEvent(&e))
            if (e.type == SDL_QUIT) secs = 0;
        glViewport(0, 0, w, h);
        glClearColor(0.1f, 0.12f, 0.2f, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glRotatef((float)(t * 60), 0, 0, 1);
        glBegin(GL_TRIANGLES);
        /* Overdraw on purpose: a few thousand big triangles, which is fill
         * rate, which is what a software rasteriser runs out of first. */
        for (int i = 0; i < 2000; i++) {
            float a = i * 0.0031f, s = 0.9f - i * 0.0004f;
            glColor3f(fmodf(i * 0.37f, 1), fmodf(i * 0.61f, 1), fmodf(i * 0.13f, 1));
            glVertex3f(s * cosf(a), s * sinf(a), -i * 0.0004f);
            glVertex3f(s * cosf(a + 2.1f), s * sinf(a + 2.1f), -i * 0.0004f);
            glVertex3f(s * cosf(a + 4.2f), s * sinf(a + 4.2f), -i * 0.0004f);
        }
        glEnd();
        SDL_GL_SwapWindow(win);
        frames++;
    }
    double t = (double)(SDL_GetPerformanceCounter() - t0) / f;
    printf("%d frames in %.2f s, %.2f fps\n", frames, t, frames / t);
    SDL_GL_DeleteContext(ctx);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}

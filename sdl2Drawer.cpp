#include "SDL3/SDL.h"
#include <iostream>
#include "sdl2Drawer.h"
#include <vector>

SDL_Window* win = nullptr;
SDL_Renderer* winRenderer = nullptr;

void initSDL() {
    if (SDL_Init(SDL_INIT_EVENTS) != true) {
        std::cout << "SDL_Init Events Error!" << std::endl;
    }
    if (SDL_Init(SDL_INIT_VIDEO) != true) {
        std::cout << "SDL_Init Video Error!" << std::endl;
    }

    win = SDL_CreateWindow("SDL3 Window", 512, 512, SDL_WINDOW_RESIZABLE);
    if (win == nullptr) {
        std::cerr << "createWindow error: " << SDL_GetError() << std::endl;
        SDL_Quit();
    }

    winRenderer = SDL_CreateRenderer(win, NULL);

    if (winRenderer == nullptr) {
        std::cerr << "createRenderer error: " << SDL_GetError() << std::endl;
    }
}

void endSDL() {
    SDL_DestroyWindow(win);
    SDL_DestroyRenderer(winRenderer);
    SDL_Quit();
}

void drawFilledCircle(float centerX, float centerY, float radius) {
    SDL_SetRenderDrawColor(winRenderer, 255, 0, 0, 100);
    for (int w = 0; w < radius * 2; w++) {
        for (int h = 0; h < radius * 2; h++) {
            int dx = radius - w; // horizontal offset
            int dy = radius - h; // vertical offset
            if ((dx * dx + dy * dy) <= (radius * radius+1)) {
                SDL_RenderPoint(winRenderer, centerX + dx, centerY + dy);
            }
        }
    }
}

SDL_Event event;
bool isRunning = true;

void clearRenderer() {
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_EVENT_QUIT) {
            isRunning = false;
        }
    }
    SDL_SetRenderDrawColor(winRenderer, 0, 0, 0, 100);
    SDL_RenderClear(winRenderer);
}

bool SDLGetRunning() {
    return isRunning;
}

void drawParticle(const float* particle, float sz) {
    const float xProjected = 256-particle[0] / particle[2];
    const float yProjected = 256-particle[1] / particle[2];
    const float radius = sz / particle[2];

    drawFilledCircle(xProjected, yProjected, radius);
}

void drawParticlePoint(const float* particle, float sz) {
    SDL_SetRenderDrawColor(winRenderer, 255, 0, 0, 100);
    const float xProjected = 256 - particle[0] / particle[2];
    const float yProjected = 256 - particle[1] / particle[2];

    SDL_RenderPoint(winRenderer, xProjected, yProjected);
}

void presentRenderer() {
    SDL_RenderPresent(winRenderer);
}
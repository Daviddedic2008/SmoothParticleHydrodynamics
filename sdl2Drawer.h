#pragma once
void initSDL();

void endSDL();

void drawFilledCircle(float centerX, float centerY, float radius);

void presentRenderer();

void clearRenderer();

bool SDLGetRunning();

void drawParticle(const float* particles, float sz);
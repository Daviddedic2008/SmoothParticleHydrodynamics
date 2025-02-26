#include <iostream>
#include "sdl2Drawer.h"
#include "dataTransfer.cuh"
#include "particleKernel.cuh"

void particleDrawLoop() {
    copyParticlesFromGPU();
    clearRenderer();
    for (int pi = 0; pi < numParticles; pi++) {
        const particlePlaceholder p = getParticle(pi);
        drawParticle(p.pos, 10.0f);
    }
    presentRenderer();
}

int main() {
    initSDL();

    addParticleToDeviceArray(100.0f, 100.0f, 1.0f);
    while (SDLGetRunning()) {
        testGravityKernel();
        particleDrawLoop();
    }
    return 0;
}
#include <iostream>
#include "sdl2Drawer.h"
#include "dataTransfer.cuh"
#include "particleKernel.cuh"
#include "rng.cuh"

void particleDrawLoop() {
    clearRenderer();
    for (int pi = 0; pi < numParticles; pi++) {
        const particlePlaceholder p = getParticle(pi);
        drawParticlePoint(p.pos, 2.0f);
    }
    presentRenderer();
}

int main() {
    initSDL();
    unsigned int seed = 0x23409204;
    for (int p = 0; p < numParticles; p++) {
        addParticleToDeviceArray(xorShiftf(seed) * 512 - 256, xorShiftf(seed) * 512 - 256, 1.0f);
    }
    initBoundingVolumes();
        
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    while (SDLGetRunning()) {
        cudaEventRecord(start);
        updateLoop();
        particleDrawLoop();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Elapsed time: %f ms\n", milliseconds);

    }
    return 0;
}
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
        //printf("%d\n", p.id);
    }
    presentRenderer();
}

int main() {

    printf("%d particles \n", numParticles);
    printf("%d bounding boxes\n", numCellsX * numCellsY * numCellsZ);
    printf("initializing...\n");
    initSDL();
    unsigned int seed = 0x23409204;
    for (int p = 0; p < numParticles; p++) {
        addParticleToDeviceArray(xorShiftf(seed) * 512 - 256, xorShiftf(seed) * 512 - 256, 1.0f);
    }
    initBoundingVolumes();
     
    printf("done!\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int frame = 0;
    while (SDLGetRunning()) {
        cudaEventRecord(start, 0);
        updateLoop();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaEventRecord(start, 0);
        particleDrawLoop();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds2 = 0;
        cudaEventElapsedTime(&milliseconds2, start, stop);

        frame++;
        if (frame % 100 == 0) {
            printf("\rcomputation: %fms    drawing: %fms", milliseconds, milliseconds2);
        }
        

    }

    printf("\nexit\n");
    return 0;
}
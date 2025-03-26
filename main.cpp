#include <iostream>
#include "sdl2Drawer.h"
#include "dataTransfer.cuh"
#include "particleKernel.cuh"
#include "rng.cuh"

void particleDrawLoop() {
    clearRenderer();
    for (int pi = 0; pi < numParticles; pi++) {
        const particlePlaceholder p = getParticle(pi);
        drawParticlePoint(p.pos, 2.0f, p.density*12);
        //printf("%d\n", p.id);
    }
    presentRenderer();
}

int main() {

    printf("%d particles \n", numParticles);
    printf("%d bounding boxes\n", numCellsX * numCellsY * numCellsZ);
    printf("initializing...\n");
    initSDL();
    unsigned int seed = 0x1;
    for (int p = 0; p < numParticles; p++) {
        addParticleToDeviceArray(xorShiftf(seed) * 512 - 256, xorShiftf(seed) * 512 - 256, 1.0f);
    }
    initBoundingVolumes();
     
    printf("done!\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int frame = 0;
    float ct = 0.0f;
    while (SDLGetRunning()) {
        float milliseconds = 0, milliseconds2 = 0;
        if (frame % 100 == 0) {
            cudaEventRecord(start, 0);
        }
        
        updateLoop();
        if (frame % 100 == 0) {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            cudaEventRecord(start, 0);
        }
        particleDrawLoop();
        if (frame % 100 == 0) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds2, start, stop);
        }

        
        
        if (frame % 100 == 0) {
            printf("\rcomputation: %fms    drawing: %fms", milliseconds, milliseconds2);
            ct += milliseconds + milliseconds2;
        }
        
        frame++;
    }

    printf("\n\nclosed\n");
    frame /= 100;
    printf("\navg. frametime: %fms\n", ct / frame);
    return 0;
}
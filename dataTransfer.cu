#include "dataTransfer.cuh"

particlePlaceholder cpuParticleArr[numParticles];

void copyParticlesFromGPU() {
	cudaMemcpyFromSymbol(cpuParticleArr, particles, sizeof(cpuParticleArr));
}

__device__ int currentParticleIndex = 0;

__global__ void addParticleToParticleArr(const float x, const float y, const float z) {
	particlePlaceholder p;
	p.pos[0] = x;
	p.pos[1] = y;
	p.pos[2] = z;

	p.vel[0] = 0.0f;
	p.vel[1] = 0.0f;
	p.vel[2] = 0.0f;

	particles[currentParticleIndex] = p;
	currentParticleIndex++;
}

void addParticleToDeviceArray(const float x, const float y, const float z) {
	addParticleToParticleArr << <1, 1 >> > (x, y, z);
}

particlePlaceholder getParticle(const int index) {
	return cpuParticleArr[index];
}
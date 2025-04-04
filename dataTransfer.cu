#include "dataTransfer.cuh"

inline __host__ __device__ unsigned int xorShift(unsigned int state) {
	state ^= state << 13;
	state ^= state >> 17;
	state ^= state << 5;

	return state;
}

inline __host__ __device__ float xorShiftf(unsigned int& state) {
	return ((state = xorShift(state)) % 1001) / 1000.0f;
}


particlePlaceholder cpuParticleArr[numParticles];

void copyParticlesFromGPU() {
	cudaMemcpyFromSymbol(cpuParticleArr, particles, sizeof(cpuParticleArr));
}

void sendParticlesToGPU() {
	cudaMemcpyToSymbol(particles, cpuParticleArr, sizeof(cpuParticleArr));
}

__device__ int currentParticleIndex = 0;

__global__ void addParticleToParticleArr(const float x, const float y, const float z) {
	particlePlaceholder p;
	p.pos[0] = x;
	p.pos[1] = y;
	p.pos[2] = z;
	
	unsigned int tmpseed = (int)x * (int)y;
	float velx = xorShiftf(tmpseed);
	p.vel[0] = 0;
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
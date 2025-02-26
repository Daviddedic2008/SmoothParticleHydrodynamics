#include "particleKernel.cuh"
#include "stdinclude.cuh"

#include "particleStructs.cuh"

inline __device__ void applyGravity(particle& p) {
	p.velocity.y += gravityConst;
}

__global__ void applyForcesAllParticles() {
	const int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= numParticles) { return; }
	applyGravity(((particle*)particles)[id]);
}

__global__ void addForcesToPos() {
	const int id = threadIdx.x + blockIdx.x * blockDim.x;
	((particle*)particles)[id].pos.y += ((particle*)particles)[id].velocity.y;
}

void testGravityKernel() {
	applyForcesAllParticles << <512, numParticles / 512 + 1 >> > ();
	addForcesToPos << <512, numParticles / 512 + 1 >> > ();
}
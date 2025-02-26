#include "particleKernel.cuh"
#include "stdinclude.cuh"

#include "particleStructs.cuh"

inline __device__ void applyGravity(particle& p) {
	p.velocity.y += gravityConst;
}

inline __device__ void applyForcesParticle(const int id) {
	if (id >= numParticles) { return; }
	applyGravity(((particle*)particles)[id]);
}

inline __device__ void addForceToPos(const int id) {
	((particle*)particles)[id].pos.y += ((particle*)particles)[id].velocity.y;
}

inline __device__ void recalculateBoundingBox(const int id) {
	const particle p = ((particle*)particles)[id];
	((particle*)particles)[id].removeParticleFromCurrentBox();
	((particle*)particles)[id].addParticleToBox(p.pos.x, p.pos.y, p.pos.z);
}

__global__ void updateParticleKernel() {
	const int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= numParticles) { return; }
	applyForcesParticle(id);
	addForceToPos(id);
	recalculateBoundingBox(id);
}

void testGravityKernel() {
	updateParticleKernel << <512, numParticles / 512 + 1 >> > ();
}
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
	particle p = ((particle*)particles)[id];
	const vec3 newPos = p.pos + p.velocity;

	const unsigned char inx = inXBounds(newPos);
	const unsigned char iny = inYBounds(newPos);
	const unsigned char inz = inZBounds(newPos);
	const unsigned char inv = inx & iny & inz;

	p.velocity.x = p.velocity.x * (-2 * !inx + 1) * (!inx * dampingFactor + inx);
	p.velocity.y = p.velocity.y * (-2 * !iny + 1) * (!iny * dampingFactor + iny);
	p.velocity.z = p.velocity.z * (-2 * !inz + 1) * (!inz * dampingFactor + inz);

	p.pos = newPos * inv + p.pos * !inv;

	((particle*)particles)[id] = p;
}

inline __device__ void recalculateBoundingBox(const int id) {
	const particle p = ((particle*)particles)[id];
	((particle*)particles)[id].recalcBox();
}

__global__ void updateParticleKernel() {
	const int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= numParticles) { return; }

	// add to velocity vector
	applyForcesParticle(id);

	// add velocity vector to pos vector
	addForceToPos(id);

	// move particle to new bounding box if needed
	//recalculateBoundingBox(id);
}

void initBoundingVolumes() {
	initBoundingBoxes << <512, numCellsX* numCellsY* numCellsZ / 512 + 1 >> > ();
}

void testGravityKernel() {
	updateParticleKernel << <512, numParticles / 512 + 1 >> > ();
}
#include "particleKernel.cuh"
#include "stdinclude.cuh"
#include "sorts.cuh"
#include "dataTransfer.cuh"

#include "particleStructs.cuh"

inline __device__ float approximatePointDensity(const particle p) {
	const int xo = p.boxID % numCellsY;
	const int yo = p.boxID / numCellsY;
	float density = 0.0f;
	// once found, must divide by volume under smoothing function(to make smoothing radius irrelevant)

	for (char i = 0, x = -1, y = -1; i < 9; i++, x++, y += (i % 3) == 0, x -= (x == 1) * 2) {
		if ((xo + x) >= 0 && (xo + x) < numCellsX && (yo + y) >= 0 && (yo + y) < numCellsY) {
			const int id = xo + x + (yo + y) * numCellsX;

			#pragma unroll
			for (int pi = (id > 0) ? frozenCountArr[id - 1] : 0; pi < frozenCountArr[id]; pi++) {
				const vec3 posDiff = p.pos - ((particle*)particles)[pi].pos;

				const float tmp = smoothingFunction(magnitude(posDiff));

				density += (tmp > 0) * tmp;
			}
		}
	}

	return density / lookupVolume; // volume under smoothing function
}

inline __global__ void applyPressureForce() {
	const int threadId = threadIdx.x + blockIdx.x * blockDim.x;

	particle& p = ((particle*)particles)[threadId];

	const int xo = p.boxID % numCellsY;
	const int yo = p.boxID / numCellsY;
	// once found, must divide by volume under smoothing function(to make smoothing radius irrelevant)

	

	for (char i = 0, x = -1, y = -1; i < 9; i++, x++, y += (i % 3) == 0, x -= (x == 1) * 2) {
		if ((xo + x) >= 0 && (xo + x) < numCellsX && (yo + y) >= 0 && (yo + y) < numCellsY) {
			const int id = xo + x + (yo + y) * numCellsX;

#pragma unroll

			for (int pi = (id > 0) ? frozenCountArr[id - 1] : 0; pi < frozenCountArr[id]; pi++) {
				particle& p2 = ((particle*)particles)[pi];

				

				//printf("%d\n", skp);

				vec3 distDiff = p2.pos - p.pos;


				const float dist = magnitude(distDiff);

				const bool skp = dist < 1e-5; // avoid branching

				distDiff = distDiff * (1 / dist);

				vec3 tmpForce = distDiff * ((p.density-targetDensity + p2.density-targetDensity) / (p2.density * 2e3) * smoothingFunctionDerivative(dist));

				tmpForce = skp ? vec3(0, 0, 0) : tmpForce;

				atomicAdd(&(p.velocity.x), tmpForce.x);
				atomicAdd(&(p.velocity.y), tmpForce.y);
				atomicAdd(&(p.velocity.z), tmpForce.z);

				atomicAdd(&(p2.velocity.x), -tmpForce.x);
				atomicAdd(&(p2.velocity.y), -tmpForce.y);
				atomicAdd(&(p2.velocity.z), -tmpForce.z);
			}
		}
	}
}

inline __device__ vec3 getDensityGradient(const particle p) {
	// returns scaled vector!
	// to compute gradient, it deosnt sample density directly with the function
	// it takes n random particles from the left, right, up, and down bounding boxes
	// it uses the averages of each box to save computation time and retain most accuracy
	// works best with smaller boxes
	// particles in boxes can be assumed to have random order(thread-dependent order)
	
}

inline __device__ void applyForcesParticle(const int id) {
	if (id >= numParticles) { return; }
	particle& p = ((particle*)particles)[id];
	p.velocity.y += gravityConst;
	p.density = approximatePointDensity(p);
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
	recalculateBoundingBox(id);
}

void initBoundingVolumes() {
	initBoundingBoxes << <512, numCellsX* numCellsY* numCellsZ / 512 + 1 >> > ();
}

void updateLoop() {
	updateParticleKernel << <512, numParticles / 512 + 1 >> > ();

	applyPressureForce << <512, numParticles / 512 + 1 >> > ();
	
	//sortEvenOdd();
	radix();

	copyParticlesFromGPU();
	//quicksort(cpuParticleArr, 0 ,numParticles-1);
	//sendParticlesToGPU();
}
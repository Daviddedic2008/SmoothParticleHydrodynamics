#include "stdinclude.cuh"


// radix sort based on bytes instead of digits from least significant to most significant
inline __device__ uint16_t getByte(const int inputNum, const unsigned char bytePlace) {
	return ((uint16_t*)&inputNum)[bytePlace];
}

__device__ int countArr[65536];

__device__ int frozenCountArr[65536];

__global__ void initArr() {
	const int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < 256) {
		countArr[id] = 0;
		frozenCountArr[id] = 0;
	}
}

__global__ void fillCountArr(const unsigned char bytePlace) {
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= numParticles) {
		return;
	}

	atomicAdd(&countArr[getByte(particles[tid].id, bytePlace)], 1);

	atomicAdd(&frozenCountArr[getByte(particles[tid].id, bytePlace)], 1);

}

__global__ void cumAddCountArr() {
	for (int i = 1; i < 256; i++) {
		countArr[i] += countArr[i - 1];
		frozenCountArr[i] += frozenCountArr[i - 1];
	}
}

__global__ void sortUsingCountArr(const unsigned char bytePlace) {
	// sorting particles

	const int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= numParticles) {
		return;
	}

	const particlePlaceholder p = particles[id];

	int index = atomicSub(&countArr[getByte(p.id, bytePlace)], 1);
	particlesBuffer[index-1] = p;

}

__device__ int numSwaps[2];

#define pairsPerThread 2

__global__ void sortOdd() {
	int index = pairsPerThread * (threadIdx.x + blockIdx.x * blockDim.x) + 1;

	for (int i = 0; i < (pairsPerThread/2) + (!(pairsPerThread / 2)); ++i, index += 2) {

		const particlePlaceholder p1 = particles[index];
		const particlePlaceholder p2 = particles[index + 1];
		if (p1.id > p2.id) {
			particles[index] = p2;
			particles[index + 1] = p1;
			numSwaps[0]++;
		}
	}
}


__global__ void sortEven() {
	int index = pairsPerThread * (threadIdx.x + blockIdx.x * blockDim.x);

	if (index == 0) {
		numSwaps[0] = 0; numSwaps[1] = 0;
	}

	for (int i = 0; i < (pairsPerThread / 2) + (!(pairsPerThread / 2)); ++i, index += 2) {

		const particlePlaceholder p1 = particles[index];
		const particlePlaceholder p2 = particles[index + 1];
		if (p1.id > p2.id) {
			particles[index] = p2;
			particles[index + 1] = p1;
			numSwaps[1]++;
		}
	}
}

__global__ void sortMergeOdd(const unsigned char iteration) {
	const int threadId = threadIdx.x + blockIdx.x * blockDim.x;


}


bool isSortedEvenOdd() {
	int ret[2];
	cudaMemcpyFromSymbol(ret, numSwaps, sizeof(int)*2);
	//printf("%d %d\n", ret[0], ret[1]);
	return ret[0] == 0 && ret[1] == 0;
}

void sortEvenOdd() {
	while(true) {
		sortEven << <512, numParticles / (512*(pairsPerThread)) + 1 >> > ();
		cudaDeviceSynchronize();
		sortOdd << <512, numParticles / (512 * (pairsPerThread)) + 1 >> > ();
		cudaDeviceSynchronize();
		if (isSortedEvenOdd()) { break; }
	}
}

__global__ void copyFromParticleBuffer() {
	const int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= numParticles) {
		return;
	}

	particles[id] = particlesBuffer[id];
}

void radix() {
	initArr << <256, 256 >> > ();
	fillCountArr << <512, numParticles / 512 + 1 >> > (0);
	cumAddCountArr << <1, 1 >> > ();
	sortUsingCountArr << <512, numParticles / 512 + 1 >> > (0);
	copyFromParticleBuffer << <512, numParticles / 512 + 1 >> > ();
}

int medianOfThree(particlePlaceholder arr[], int low, int high) {
	int mid = low + (high - low) / 2;
	if (arr[low].id > arr[mid].id) std::swap(arr[low], arr[mid]);
	if (arr[low].id > arr[high].id) std::swap(arr[low], arr[high]);
	if (arr[mid].id > arr[high].id) std::swap(arr[mid], arr[high]);
	return mid; 
}


int partition(particlePlaceholder arr[], int low, int high) {
	int pivot = medianOfThree(arr, low, high);
	int i = low - 1;

	for (int j = low; j < high; ++j) {
		if (arr[j].id <= pivot) {
			++i;
			std::swap(arr[i], arr[j]);
		}
	}
	std::swap(arr[i + 1], arr[high]);
	return i + 1;
}

// Quicksort function
void quicksort(particlePlaceholder arr[], int low, int high) {
	while (low < high) {
		int pivotIndex = partition(arr, low, high);

		
		if (pivotIndex - low < high - pivotIndex) {
			quicksort(arr, low, pivotIndex - 1);
			low = pivotIndex + 1;              
		}
		else {
			quicksort(arr, pivotIndex + 1, high);
			high = pivotIndex - 1;            
		}
	}
}

typedef struct {
	int startIndex, endIndex;
}startEnd;

/*startEnd getParticlesInBox(const int boxId) {
	;
}*/
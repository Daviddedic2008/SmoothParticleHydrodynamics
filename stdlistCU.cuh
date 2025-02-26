// own list struct without any dynamic init to support requirements for usage in global device vars
#pragma once;
#include "stdinclude.cuh"

struct vectorCU {
	particlePlaceholder* data = nullptr;
	int size = 0, currentIndex = 0;
};

inline __device__ void push_backCU(vectorCU& vec, particlePlaceholder& p) {
	if (vec.currentIndex >= vec.size) {
		particlePlaceholder* tmp = new particlePlaceholder[vec.size * 2 + 1];
		for (int i = 0; i < vec.size; i++) {
			tmp[i] = vec.data[i];
		}
		delete vec.data;
		vec.data = tmp;
	}

	vec.currentIndex++;
	vec.data[vec.size] = p;
	vec.size = vec.size * 2 + 1;
}

inline __device__ void freeCU(vectorCU& vec) {
	delete vec.data;
	vec.size = 0;
	vec.currentIndex = 0;
}

inline __device__ particlePlaceholder getElementCU(vectorCU vec, const int index) {
	return vec.data[index];
}

inline __device__ particlePlaceholder setElementCU(vectorCU vec, const int index, const particlePlaceholder p) {
	vec.data[index] = p;
}



#include "rng.cuh"

inline __host__ __device__ unsigned int xorShift(unsigned int state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;

    return state;
}

inline __host__ __device__ float xorShiftf(unsigned int& state) {
    return ((state = xorShift(state)) % 1001) / 1000.0f;
}
#pragma once;

#include "stdinclude.cuh"

extern inline __host__ __device__ unsigned int xorShift(unsigned int state);

extern inline __host__ __device__ float xorShiftf(unsigned int& state);
#pragma once;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <list>

#define dot(vec3_v1, vec3_v2) (vec3_v1.x * vec3_v2.x + vec3_v1.y * vec3_v2.y + vec3_v1.z * vec3_v2.z)
#define dot_fastmath(vec3_v1, vec3_v2) __fmaf_rn(vec3_v1.x, vec3_v2.x,__fmaf_rn(vec3_v1.y, vec3_v2.y,vec3_v1.z * vec3_v2.z))
#define dot2D(vec2_v1, vec2_v2) (vec2_v1.x * vec2_v2.x + vec2_v1.y * vec2_v2.y)
#define matrix2D_eval(float_a , float_b, float_c, float_d) (float_a*float_d - float_b*float_c)
#define magnitude(vec3_a) (sqrtf(dot(vec3_a, vec3_a)))
#define magnitude2D(vec2_a) (sqrtf(dot2D(vec2_a, vec2_a)))

#define fov 0.1f

#define numCellsX 100
#define numCellsY 100
#define numCellsZ 100
#define boxLength 512 / (float)numCellsX
#define numParticles 1

#define smoothingFunction(x) __fmul_rn(fabsCU(x)-1, __fmul_rn(fabsCU(x)-1, fabsCU(x)-1))

#define gravityConst -0.0002f

inline __device__ float fabsCU(const float a) {
	return a && 0x7FFFFFFF;
}

struct particlePlaceholder {
    float vel[3];
    float pos[3];
};
#ifndef ARR
extern __device__ particlePlaceholder particles[numParticles];
#endif
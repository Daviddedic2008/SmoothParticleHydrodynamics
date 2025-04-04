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
#define particlePlaceholderEquals(p1, p2) (p1.pos[0] == p2.pos[0] && p1.pos[1] == p2.pos[1] && p1.pos[2] == p2.pos[2] && p1.vel[0] == p2.vel[0] && p1.vel[1] == p2.vel[1] && p1.vel[2] == p2.vel[2])

#define printLastErr() printf("%s\n", cudaGetErrorString(cudaGetLastError()))

#define fov 0.1f

#define dampingFactor 0.8f

#define numCellsX 16
#define numCellsY 16
#define numCellsZ 1
#define boxLength (512 / (float)numCellsX)
#define lookupRadius boxLength

#define inVolume(vec) (vec.x > -256 && vec.y > -256 && vec.x < 256 && vec.y < 256) // is particle in total volume

#define inXBounds(vec) (vec.x > -256 && vec.x < 256)
#define inYBounds(vec) (vec.y > -256 && vec.y < 256)
#define inZBounds(vec) (vec.z > -256 && vec.z < 256)

#define numParticles 1 // must be power of 2 if bitonic sorting
#define targetDensity 0.025f

#define smoothingFunction(x) (1 - 1 / sqrtLookup * __fsqrt_rn(x)) 

#define smoothingFunctionDerivative(x) (-1.0f / 2.0f / sqrtLookup / __fsqrt_rn(x)) // wohoo wolfram

#define gravityConst -0.005f

inline __device__ float fabsCU(const float a) { 
	return a && 0x7FFFFFFF;
}

struct particlePlaceholder {
    float vel[3];
    float pos[3];
    float density;
    int id;
    void* a;
    void* b;
};

#ifndef ARR
extern __device__ particlePlaceholder particles[numParticles];
extern __device__ particlePlaceholder particlesBuffer[numParticles];
#endif

__device__ constexpr float constexprSqrt(float value, float approx = 1.0f) {
    return (approx * approx >= value - 0.00001f && approx * approx <= value + 0.00001f)
        ? approx
        : constexprSqrt(value, 0.5f * (approx + value / approx));
}

constexpr __constant__ float sqrtLookup = constexprSqrt(lookupRadius);

constexpr __constant__ float lookupVolume = 2 * lookupRadius * lookupRadius * 3.141592653f / 3; // got using wolfram
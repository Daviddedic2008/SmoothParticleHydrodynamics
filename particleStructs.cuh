#pragma once;

#include "stdinclude.cuh"
#include "stdlistCU.cuh"

struct vec2 {
    float x, y;

    __host__ __device__ vec2() : x(0.0f), y(0.0f) {}
    __host__ __device__ vec2(float X, float Y) : x(X), y(Y) {}

    inline __host__ __device__ vec2 operator+(const vec2& f) const {
        return vec2(x + f.x, y + f.y);
    }

    inline __host__ __device__ vec2 operator-(const vec2& f) const {
        return vec2(x - f.x, y - f.y);
    }

    inline __host__ __device__ vec2 operator*(const float scalar) const {
        return vec2(x * scalar, y * scalar);
    }

    inline __host__ __device__ vec2 normalize() {
        const float scl = magnitude2D((*this));
        return vec2(x / scl, y / scl);
    }

    inline __device__ float dist_from_vec(vec2& v) {
        const float addx = (x + v.x);
        const float addy = (y + v.y);
        return __fsqrt_rn(__fmaf_rn(addx, addx, addy * addy));
    }
};

// Define the vec3 struct
struct vec3 {
    float x, y, z;

    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    inline __host__ __device__ vec3 operator+(const vec3& f) const {
        return vec3(x + f.x, y + f.y, z + f.z);
    }

    inline __host__ __device__ void operator+=(const vec3& f) {
        x += f.x;
        y += f.y;
        z += f.z;
    }

    inline __host__ __device__ vec3 operator-(const vec3& f) const {
        return vec3(x - f.x, y - f.y, z - f.z);
    }

    inline __host__ __device__ vec3 operator*(const float scalar) const {
        return vec3(x * scalar, y * scalar, z * scalar);
    }

    inline __host__ __device__ vec3 normalize() {
        const float scl = 1 / magnitude((*this));
        return vec3(x * scl, y * scl, z * scl);
    }

    inline __host__ __device__ bool operator==(const vec3& f) const {
        return fabs(x - f.x) < 0.01f && fabs(y - f.y) < 0.01f && fabs(z - f.z) < 0.01f;
    }

    inline __host__ __device__ vec2 convert_vec2() const {
        return vec2(x / (z * fov), y / (z * fov));
    }
};

// cross is more logical as its own function

inline __host__ __device__ vec3 cross(const vec3 v1, const vec3 v2) {
    vec3 ret;
    ret.x = matrix2D_eval(v1.y, v1.z, v2.y, v2.z);
    ret.y = matrix2D_eval(v1.x, v1.z, v2.x, v2.z);
    ret.z = matrix2D_eval(v1.x, v1.y, v2.x, v2.y);
    return ret;
}

__device__ linkedListCU boundingBoxes[numCellsX * numCellsY * numCellsZ];

__global__ void initBoundingBoxes() {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= numCellsX * numCellsY * numCellsZ) {
        return;
    }

    initList(boundingBoxes[id]);
}

struct particle {
    vec3 velocity;
    vec3 pos;
    int boxID = -1;
    iteratorCU boxPos;

    __device__ particle(const vec3 pos) : pos(pos), velocity(vec3(0.0f, 0.0f, 0.0f)) {}

    __device__ particle() {}

    inline __device__ void addParticleToBox(const int x, const int y, const int z) {
        const int tmpID = (int)((pos.x + 256) / lookupRadius) + (int)((pos.y + 256) / lookupRadius) * numCellsX + (int)(pos.z / lookupRadius) * numCellsX * numCellsY;

        if (boxID == tmpID) {
            return;
        }

        boxID = tmpID;
        boxPos = push_backLinkedCU(boundingBoxes[boxID], *(particlePlaceholder*)this);
    }

    inline __device__ void recalcBox() {
        const int tmpID = (int)((pos.x + 256)/lookupRadius) + (int)((pos.y + 256) / lookupRadius) * numCellsX + (int)(pos.z / lookupRadius) * numCellsX * numCellsY;
        if (tmpID != boxID) {
            boxID = tmpID % 256;
        }
    }
};
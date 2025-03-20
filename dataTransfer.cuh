#pragma once;

#include "stdinclude.cuh"

void copyParticlesFromGPU();

void sendParticlesToGPU();

void addParticleToDeviceArray(const float x, const float y, const float z);

particlePlaceholder getParticle(const int index);

extern particlePlaceholder cpuParticleArr[];
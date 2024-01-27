#pragma once
#include <curand_kernel.h>
__device__ double random_double() {
    // Returns a random real in [0,1).
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = tid_y * gridDim.x * blockDim.x + tid_x;
    return curand_uniform_double(&threadRandomStates[tid]);
}

__device__ double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_double();
}
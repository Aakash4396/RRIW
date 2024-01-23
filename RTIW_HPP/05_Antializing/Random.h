#pragma once
#include <curand_kernel.h>
__device__ double random_double(curandState* state) {
    // Returns a random real in [0,1).
    return curand_uniform_double(state);
}

__device__ double random_double(double min, double max, curandState* state) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_double(state);
}
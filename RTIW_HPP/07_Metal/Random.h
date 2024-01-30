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

__device__ vec3 random_in_unit_sphere() {
    while (true) {
        auto p = vec3(random_double(-1,1), random_double(-1,1), random_double(-1,1));
        if (p.length_squared() < 1) {
            return p;
        }
    }
}

__device__ vec3 random_unit_vector() {
    return unit_vector(random_in_unit_sphere());
}

__device__ vec3 random_on_hemisphere(const vec3& normal) {
    vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}


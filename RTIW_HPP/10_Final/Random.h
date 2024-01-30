#pragma once
#include <curand_kernel.h>

const double pi = 3.1415926535897932385;

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

__device__ vec3 random_vector(double min, double max) {
    return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
}

__device__ vec3 random_vector() {
    return vec3(random_double(), random_double(), random_double());
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

__device__ vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ double reflectance(double cosine, double ref_idx) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

__device__ vec3 random_in_unit_disk() {
    while (true) {
        auto p = vec3(random_double(-1, 1), random_double(-1, 1), 0);
        if (p.length_squared() < 1)
            return p;
    }
}


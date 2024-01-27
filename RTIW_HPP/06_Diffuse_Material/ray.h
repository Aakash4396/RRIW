#pragma once
#ifndef RAY_H
#define RAY_H

#include "gVec3.h"
#include "point.h"

class Ray {
private:
    point orig;
    vec3 dir;
public:
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const point& origin, const vec3& direction) : orig(origin), dir(direction) {}
    __host__ __device__ point origin() const { return orig; }
    __host__ __device__ vec3 direction() const { return dir; }
    template<typename T>
    __host__ __device__ point at(T t) const { return orig + (t * dir); }
};

#endif // !RAY_H

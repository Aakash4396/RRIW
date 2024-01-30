#pragma once
#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
class material;
struct hit_record {
    point p;
    vec3 normal;
    double t;
    material * mat;
    bool front_face;

    __host__ __device__ void set_face_normal(const Ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    virtual ~hittable() = default;

    __device__ virtual bool hit(const Ray& r, interval ray_t, hit_record& rec) const = 0;
};

#endif // !HITTABLE_H

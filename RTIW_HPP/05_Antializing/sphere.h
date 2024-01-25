#pragma once
#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

class sphere {
private:
    point center;
    double radius;
public:
    sphere() {}
    sphere(point _center, double _radius) : center(_center), radius(_radius) {}

    __host__ __device__ bool hit(const Ray& r, interval ray_t, hit_record& rec) {
        vec3 oc = r.origin() - center;
        auto a = r.direction().length_squared();
        auto half_b = dot(oc, r.direction());
        auto c = oc.length_squared() - radius * radius;

        auto discriminant = half_b * half_b - a * c;
        if (discriminant < 0) return false;
        auto sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (-half_b - sqrtd) / a;
        if (root <= ray_t.imin || ray_t.imax <= root) {
            root = (-half_b + sqrtd) / a;
            if (root <= ray_t.imin || ray_t.imax <= root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        return true;
    }
};
#endif // !SPHERE_H

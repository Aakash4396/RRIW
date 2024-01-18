#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H

#include "rtiw.h"

class material {
public:
    virtual ~material() = default;
    virtual bool scatter(const Ray& in, const hit_record& rec, color& attenuation, Ray& scattered) const = 0;
};


class lambertian : public material {
private :
    color albedo;
public:
    lambertian(const color& a) : albedo(a) {}

    bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered) const override {
        auto scatter_direction = rec.normal + random_unit_vector();

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = Ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

class metal : public material {
private:
    color albedo;
    double fuzz;

public:
    metal(const color& _albedo, double f) : albedo(_albedo), fuzz(f < 1 ? f : 1) {}

    bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz * random_unit_vector());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
};

#endif // !MATERIAL_H

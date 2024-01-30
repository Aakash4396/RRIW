#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H


class material {
public:
    __device__ virtual bool scatter(const Ray& in, const hit_record& rec, color& attenuation, Ray& scattered) const = 0;
};


class lambertian : public material {
private :
    color albedo;
public:
    __device__ lambertian(const color& a) : albedo(a) {}

    __device__ bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered) const override {
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
    __device__ metal(const color& _albedo, double f) : albedo(_albedo), fuzz(f < 1 ? f : 1) {}

    __device__ bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz * random_unit_vector());
        attenuation = albedo;
        return true;
    }
};

#endif // !MATERIAL_H

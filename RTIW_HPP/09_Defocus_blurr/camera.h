#pragma once
#ifndef CAMERA_H
#define CAMERA_H
#include "hittable.h"

#include "color.h"

#include <iostream>

class camera {

public:
    int width;
    int height;
    point center;
    point pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    int samples_per_pixel = 100;
    int max_depth = 50;

    vec3   u, v, w;
    vec3   defocus_disk_u;  // Defocus disk horizontal radius
    vec3   defocus_disk_v;  // Defocus disk vertical radius
    
    double vfov = 90;     // Vertical view angle (field of view)
    point lookfrom = point(0.0f, 0.0f, -1.0f);  // Point camera is looking from
    point lookat = point(0.0f, 0.0f, 0.0f);   // Point camera is looking at
    vec3 vup = vec3(0.0f, 1.0f, 0.0f);     // Camera-relative "up" direction

    double defocus_angle = 0;  // Variation angle of rays through each pixel
    double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus


    camera(int _width, int _height) : width(_width), height(_height) {
    }
    ~camera() {
    }

    void setup() {

        // Camera
        center = lookfrom;
        //auto focal_length = (lookfrom - lookat).length();;
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2 * h * focus_dist;
        double viewport_width = viewport_height * ((double)width / (double)height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        
        // Calculate vectors across viewport horizontal and viewport downword directions
        auto viewport_u = viewport_width * u;   // Vector across viewport horizontal edge
        auto viewport_v = viewport_height * -v; // Vector down viewport vertical edge
        
        // calculate delta vectors across X and -Y direction
        pixel_delta_u = viewport_u / width;
        pixel_delta_v = viewport_v / height;
        
        // location of upper left position
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + (0.5f) * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;

    }

    __device__ Ray get_ray(int i, int j) const {
        // Get a randomly-sampled camera ray for the pixel at location i,j, originating from
        // the camera defocus disk.

        auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        auto pixel_sample = pixel_center + pixel_sample_square();

        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
        auto ray_direction = pixel_sample - ray_origin;

        return Ray(ray_origin, ray_direction);
    }

    __device__ vec3 pixel_sample_square() const {
        // Returns a random point in the square surrounding a pixel at the origin.
        auto px = -0.5 + random_double();
        auto py = -0.5 + random_double();
        return (px * pixel_delta_u) + (py * pixel_delta_v);
    }

    __device__ point defocus_disk_sample() const {
        // Returns a random point in the camera defocus disk.
        auto p = random_in_unit_disk();
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    __device__ color ray_color(const Ray& r, int max_depth, hittable_list** world) {
        color result(1.0f, 1.0f, 1.0f);
        Ray current_ray = r;

        for (int depth = max_depth; depth > 0; --depth) {
            hit_record rec;

            if ((*world)->hit(current_ray, interval(0.001, FLT_MAX), rec)) {
                Ray scattered;
                color attenuation;
                if (rec.mat->scatter(current_ray, rec, attenuation, scattered)) {
                    // Update the ray and accumulate attenuation
                    current_ray = scattered;
                    result *= attenuation;
                } else {
                    // If no scattering, return black
                    return color(0, 0, 0);
                }
            } else {
                vec3 unit_direction = unit_vector(current_ray.direction());
                auto a = 0.5f * (unit_direction.y() + 1.0f);
                result *= (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
                break;  // Break out of the loop since there's no more intersection.
            }
        }
        return result;
    }
};

#endif
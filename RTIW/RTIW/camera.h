#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "rtiw.h"
#include "material.h"

#include <iostream>

class camera {

private:
    int width;
    int height;
    point center;
    point pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3   u, v, w;
    unsigned char* image;

public:

    // public variables
    int samples_per_pixel = 10; // Count of random samples for each pixel
    int max_depth = 10;   // Maximum number of ray bounces into scene

    double vfov = 90;     // Vertical view angle (field of view)
    point lookfrom = point(0.0f, 0.0f, -1.0f);  // Point camera is looking from
    point lookat = point(0.0f, 0.0f, 0.0f);   // Point camera is looking at
    vec3   vup = vec3(0.0f, 1.0f, 0.0f);     // Camera-relative "up" direction

    camera(int _width, int _height) : width(_width), height(_height) {
        image = new unsigned char[width*height*4];
    }
    ~camera() {
        delete[] image;
        image = nullptr;
    }

    unsigned char* createImage(const hittable& world) {
        initialize();

        // fout << "P3\n" << width << " " << height << "\n255\n";
        
        for (int j = 0; j < height; ++j) {
            std::clog << "\rScanlines remaining: " << (height - j) << ' ' << std::flush;
            for (int i = 0; i < width; ++i) {
                color pixel_color(0.0f, 0.0f, 0.0f);
                for (int sample = 0; sample < samples_per_pixel; ++sample) {
                    Ray r = get_ray(i, j);
                    pixel_color += ray_color(r, max_depth, world);
                }
                write_color(image, j, i, width, pixel_color, samples_per_pixel);
            }
        }
        
        std::clog << "\rDone.                 \n";
        return image;
    }

private:

    void initialize() {

        // Camera

        center = lookfrom;
        auto focal_length = (lookfrom - lookat).length();;
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2 * h * focal_length;
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
        auto viewport_upper_left = center - (focal_length * w) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + (0.5f) * (pixel_delta_u + pixel_delta_v);
    }

    Ray get_ray(int i, int j) const {
        // Get a randomly sampled camera ray for the pixel at location i,j.

        auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        auto pixel_sample = pixel_center + pixel_sample_square();

        auto ray_origin = center;
        auto ray_direction = pixel_sample - ray_origin;

        return Ray(ray_origin, ray_direction);
    }

    vec3 pixel_sample_square() const {
        // Returns a random point in the square surrounding a pixel at the origin.
        auto px = -0.5 + random_double();
        auto py = -0.5 + random_double();
        return (px * pixel_delta_u) + (py * pixel_delta_v);
    }

    color ray_color(const Ray& r, int depth, const hittable& world) {
        hit_record rec;

        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return color(0, 0, 0);


        if (world.hit(r, interval(0.001, infinity), rec)) {
            Ray scattered;
            color attenuation;
            if (rec.mat->scatter(r, rec, attenuation, scattered)) {
                return attenuation * ray_color(scattered, depth - 1, world);
            }
            return color(0.0f, 0.0f, 0.0f);
        }

        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
    }
};

#endif

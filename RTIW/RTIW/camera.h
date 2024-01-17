#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "rtiw.h"

#include "color.h"
#include "hittable.h"

#include <iostream>

class camera {

public:
    camera(int _width, int _height) : width(_width), height(_height) {
        image = new unsigned char[width*height*4];
    }
    ~camera() {
        delete[] image;
        image = nullptr;
    }

    unsigned char* createImage(const hittable& world) {
        initialize();

        //fout << "P3\n" << width << " " << height << "\n255\n";
        
        for (int j = 0; j < height; ++j) {
            std::clog << "\rScanlines remaining: " << (height - j) << ' ' << std::flush;
            for (int i = 0; i < width; ++i) {
                auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
                auto ray_direction = pixel_center - center;
                Ray r(center, ray_direction);
                color pixel_color = ray_color(r, world);
                //fout << int(255.99 * pixel_color.x()) << " " << int(255.99 * pixel_color.y()) << " " << int(255.99 * pixel_color.z()) << fflush;
                write_color(image, j, i, width, pixel_color);
            }
        }
        
        std::clog << "\rDone.                 \n";
        return image;
    }

private:
    int width;
    int height;
    point center;
    point pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    unsigned char* image;

    void initialize() {

        // Camera

        center = point(0.0f, 0.0f, 0.0f);
        auto focal_length = 1.0f;
        float viewport_height = 2.0f;
        float viewport_width = viewport_height * ((float)width / (float)height);
        
        // Calculate vectors across viewport horizontal and viewport downword directions
        auto viewport_u = vec3(viewport_width, 0.0f, 0.0f);
        auto viewport_v = vec3(0.0f, -viewport_height, 0.0f);
        
        // calculate delta vectors across X and -Y direction
        pixel_delta_u = viewport_u / width;
        pixel_delta_v = viewport_v / height;
        
        // location of upper left position
        auto viewport_upper_left = center - vec3(0.0f, 0.0f, focal_length) - viewport_u / 2.0f - viewport_v / 2.0f;
        pixel00_loc = viewport_upper_left + (0.5f) * (pixel_delta_u + pixel_delta_v);
    }

    color ray_color(const Ray& r, const hittable& world) {
        hit_record rec;
        if (world.hit(r, interval(0, infinity), rec)) {
            return 0.5f * (rec.normal + color(1, 1, 1));
        }

        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
    }
};

#endif

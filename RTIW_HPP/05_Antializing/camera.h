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
    int samples_per_pixel = 10;
    camera(int _width, int _height) : width(_width), height(_height) {
    }
    ~camera() {
    }

    void createImage(unsigned char* image, hittable_list* world) {
        initialize();

        unsigned char * d_image;
        gpuErrchk(cudaMalloc((void**)&d_image, width*height*4*sizeof(char)), "Failed to allocate memory to image.");

        camera* d_cam;
        gpuErrchk(cudaMalloc((void**)&d_cam, sizeof(camera)), "Failed to allocate memory to cam.");
        gpuErrchk(cudaMemcpy(d_cam, this, sizeof(camera), cudaMemcpyHostToDevice), "Failed to copy camera data from host to device.");


        // Define thread block size (e.g., 16x16)
        dim3 blockDim = dim3(16, 16);

        // Define grid size
        dim3 gridDim = dim3((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        CudaWrapper::cudaKernel(gridDim, blockDim, d_image, width, height, d_cam, world, samples_per_pixel);

        gpuErrchk( cudaPeekAtLastError(), "Error while launching kernel.");
        gpuErrchk( cudaDeviceSynchronize(), "Device Synchronization.");

        gpuErrchk(cudaMemcpy(image, d_image, width*height*4*sizeof(char), cudaMemcpyDeviceToHost), "Failed to get image data from device to host.");

        gpuErrchk(cudaFree(d_image), "Failed to free image data on device.");
        gpuErrchk(cudaFree(d_cam), "Failed to free camera data on device.");
        
    }

    __host__ __device__ void initialize() {

        // Camera

        center = point(0.0f, 0.0f, 0.0f);
        auto focal_length = 1.0f;
        double viewport_height = 2.0f;
        double viewport_width = viewport_height * ((double)width / (double)height);
        
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

    __device__ Ray get_ray(int i, int j) const {
        // Get a randomly sampled camera ray for the pixel at location i,j.

        auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        auto pixel_sample = pixel_center + pixel_sample_square();

        auto ray_origin = center;
        auto ray_direction = pixel_sample - ray_origin;

        return Ray(ray_origin, ray_direction);
    }

    __device__ vec3 pixel_sample_square() const {
        // Returns a random point in the square surrounding a pixel at the origin.
        auto px = -0.5 + random_double();
        auto py = -0.5 + random_double();
        return (px * pixel_delta_u) + (py * pixel_delta_v);
    }

    __host__ __device__ color ray_color(const Ray& r, hittable_list* world) {
        hit_record rec;
        if (world->hit(r, interval(0, 10000000.0), rec)) {
            return 0.5f * (rec.normal + color(1, 1, 1));
        }

        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
    }
};

#endif

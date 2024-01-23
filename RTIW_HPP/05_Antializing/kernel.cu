#include "kernel.cuh"
#include "Random.h"
#include "interval.h"
#include "hittable_list.h"
#include "point.h"
#include "color.h"
#include "ray.h"
#include "camera.h"
#include "sphere.h"



__global__ void createImage(unsigned char* image, int width, int height, camera* cam, hittable& world, int samples_per_pixel) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init(1234, i+j, 0, &state);

    if (j < height && i < width) {
        color pixel_color(0.0f, 0.0f, 0.0f);
        for (int sample = 0; sample < samples_per_pixel; ++sample) {
            Ray r = cam->get_ray(i, j, &state);
            pixel_color += cam->ray_color(r, world);
        }
        write_color(image, j, i, width, pixel_color, samples_per_pixel);
    }
}



void CudaWrapper::cudaKernel(dim3 gridDim, dim3 blockDim, unsigned char* image, int width, int height, camera* cam, hittable& world, int samples_per_pixel) {
    createImage<<<gridDim, blockDim>>>(image, width, height, cam, world, samples_per_pixel);
}


void CudaWrapper::cudaMain(unsigned char* image, int width, int height) {

    hittable_list world;

    world.add(new sphere(point(0.0f, 0.0f, -1.0f), 0.5f));
    world.add(new sphere(point(0.0f, -100.5f, -1.0f), 100.0f));

    camera cam(width, height);

    cam.samples_per_pixel = 100;

    cam.createImage(image, world);

}




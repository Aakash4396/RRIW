#include "kernel.cuh"
#include "interval.h"
#include "hittable_list.h"
#include "point.h"
#include "color.h"
#include "ray.h"
#include "Random.h"
#include "camera.h"
#include "sphere.h"


// Setup kernel to initialize curandState for each thread
__global__ void setupRandomStates(unsigned int seed) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = tid_y * gridDim.x * blockDim.x + tid_x;
    curand_init(seed + tid, 0, 0, &threadRandomStates[tid]);
}


__global__ void createImage(unsigned char* image, int width, int height, camera* cam, hittable_list* world, int max_depth, int samples_per_pixel) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < height && i < width) {
        color pixel_color(0.0f, 0.0f, 0.0f);
        for (int sample = 0; sample < samples_per_pixel; ++sample) {
            Ray r = cam->get_ray(i, j);
            pixel_color += cam->ray_color(r, max_depth, world);
        }
        write_color(image, j, i, width, pixel_color, samples_per_pixel);
    }
}



void CudaWrapper::cudaKernel(dim3 gridDim, dim3 blockDim, unsigned char* image, int width, int height, camera* cam, hittable_list* world, int max_depth, int samples_per_pixel) {
    createImage<<<gridDim, blockDim>>>(image, width, height, cam, world, max_depth, samples_per_pixel);
}


void CudaWrapper::cudaMain(unsigned char* image, int width, int height) {

    hittable_list h_world;
    h_world.add(sphere(point(0.0f, 0.0f, -1.0f), 0.5f));
    h_world.add(sphere(point(0.0f, -100.5f, -1.0f), 100.0f));
    
    hittable_list* d_world;
    gpuErrchk(cudaMalloc((void**)&d_world, sizeof(hittable_list)), "Failed to allocate memory to world.");


    gpuErrchk(cudaMemcpy(d_world, &h_world, sizeof(hittable_list), cudaMemcpyHostToDevice), "Failed to copy h_world data from host to device.");

    unsigned int seed = static_cast<unsigned int>(std::time(0));
    
    setupRandomStates<<<GRIDDIM, BLOCKDIM>>>(seed);

    camera cam(width, height);

    cam.samples_per_pixel = 100;

    cam.max_depth = 50;

    cam.createImage(image, d_world);
    
}




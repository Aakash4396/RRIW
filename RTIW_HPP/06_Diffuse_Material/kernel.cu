#include "kernel.cuh"
#include "interval.h"
#include "hittable_list.h"
#include "point.h"
#include "color.h"
#include "ray.h"
#include "Random.h"
#include "camera.h"
#include "sphere.h"

__global__ void init_world(hittable_list** d_world) {

    *d_world = new hittable_list();
    (*d_world)->add(new sphere(point(0.0f, -100.5f, -1.0f), 100.0f));
    (*d_world)->add(new sphere(point(0.0f, 0.0f, -1.0f), 0.5f));
}


// Setup kernel to initialize curandState for each thread
__global__ void setupRandomStates(unsigned int seed) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = tid_y * gridDim.x * blockDim.x + tid_x;
    curand_init(seed + tid, 0, 0, &threadRandomStates[tid]);
}


__global__ void createImage(unsigned char* image, int width, int height, camera* cam, hittable_list** world, int max_depth, int samples_per_pixel) {
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


void CudaWrapper::cudaMain(unsigned char* image, int width, int height) {

    hittable_list** d_world = NULL;
    gpuErrchk(cudaMalloc((void**)&d_world, sizeof(hittable_list*)), "Failed to allocate memory to d_world.");

    init_world<<<1, 1>>>(d_world);
    gpuErrchk( cudaPeekAtLastError(), "Error while launching kernel.");
    gpuErrchk( cudaDeviceSynchronize(), "Device Synchronization init_world.");
    
    unsigned int seed = static_cast<unsigned int>(std::time(0));
    
    setupRandomStates<<<GRIDDIM, BLOCKDIM>>>(seed);

    unsigned char * d_image;
    gpuErrchk(cudaMalloc((void**)&d_image, width*height*4*sizeof(char)), "Failed to allocate memory to image.");


    camera cam(width, height);
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;
    cam.setup();

    camera* d_cam;
    gpuErrchk(cudaMalloc((void**)&d_cam, sizeof(camera)), "Failed to allocate memory to cam.");
    gpuErrchk(cudaMemcpy(d_cam, &cam, sizeof(camera), cudaMemcpyHostToDevice), "Failed to copy camera data from host to device.");


    createImage<<<GRIDDIM, BLOCKDIM>>>(d_image, width, height, d_cam, d_world, cam.max_depth, cam.samples_per_pixel);

    gpuErrchk( cudaPeekAtLastError(), "Error while launching kernel.");
    gpuErrchk( cudaDeviceSynchronize(), "Device Synchronization.");

    gpuErrchk(cudaMemcpy(image, d_image, width*height*4*sizeof(char), cudaMemcpyDeviceToHost), "Failed to get image data from device to host.");

    gpuErrchk(cudaFree(d_world), "Failed to free world data on device.");
    gpuErrchk(cudaFree(d_image), "Failed to free image data on device.");
    gpuErrchk(cudaFree(d_cam), "Failed to free camera data on device.");
    
}





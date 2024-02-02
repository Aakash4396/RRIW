#include "kernel.cuh"
#include "interval.h"
#include "point.h"
#include "color.h"
#include "ray.h"
#include "Random.h"
#include "hittable_list.h"
#include "hittable.h"
#include "material.h"
#include "camera.h"
#include "sphere.h"
#include "helper_timer.h"

#define RND (curand_uniform_double(&local_rand_state))

float timeOnGPU = 0.0f;

__global__ void init_world(hittable_list** d_world, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        curandState local_rand_state = *rand_state;
        *d_world = new hittable_list();
        // Ground sphere
        (*d_world)->add(new sphere(point(0, -1000, 0), 1000.0f, new lambertian(color(0.5, 0.5, 0.5))));
        

        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                auto choose_mat = RND;
                point center(a + 0.9 * RND, 0.2, b + 0.9 * RND);

                if ((center - point(4, 0.2, 0)).length() > 0.9) {

                    material* sphere_material;
                    if (choose_mat < 0.8) {
                        // diffuse
                        auto albedo = vec3(RND*RND, RND*RND, RND*RND);
                        sphere_material = new lambertian(albedo);
                        (*d_world)->add(new sphere(center, 0.2, sphere_material));
                    }
                    else if (choose_mat < 0.95) {
                        // metal
                        auto albedo = vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND));
                        auto fuzz = 0.5f*RND;
                        sphere_material = new metal(albedo, fuzz);
                        (*d_world)->add(new sphere(center, 0.2, sphere_material));
                    }
                    else {
                        // glass
                        sphere_material = new dielectric(1.5);
                        (*d_world)->add(new sphere(center, 0.2, sphere_material));
                    }
                }
            }    
        }
        
        (*d_world)->add(new sphere(point(0, 1, 0), 1.0, new dielectric(1.5)));
        (*d_world)->add(new sphere(point(-4, 1, 0), 1.0, new lambertian(color(0.4, 0.2, 0.1))));
        (*d_world)->add(new sphere(point(4, 1, 0), 1.0, new metal(color(0.7, 0.6, 0.5), 0.0)));
    }
}

__global__ void rand_init(curandState *rand_state, unsigned int seed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(seed, 0, 0, rand_state);
    }
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

    curandState *d_rand_state2;
    gpuErrchk(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)), "Failed to allocate memory to rand_state variable.");

    unsigned int seed = static_cast<unsigned int>(std::time(0));
    

    rand_init<<<1,1>>>(d_rand_state2, seed);
    gpuErrchk( cudaPeekAtLastError(), "Error while initialting random numbers.");
    gpuErrchk( cudaDeviceSynchronize(), "Device Synchronization rand_init.");

    init_world<<<1, 1>>>(d_world, d_rand_state2);
    gpuErrchk( cudaPeekAtLastError(), "Error while initiating world object.");
    gpuErrchk( cudaDeviceSynchronize(), "Device Synchronization init_world.");
    
    seed = static_cast<unsigned int>(std::time(0));
    
    setupRandomStates<<<GRIDDIM, BLOCKDIM>>>(seed);
    gpuErrchk( cudaPeekAtLastError(), "Error while setting random states for all threads.");
    gpuErrchk( cudaDeviceSynchronize(), "Device Synchronization setting random states.");

    unsigned char * d_image;
    gpuErrchk(cudaMalloc((void**)&d_image, width*height*4*sizeof(char)), "Failed to allocate memory to image.");


    camera cam(width, height);
    cam.samples_per_pixel = 500;
    cam.max_depth = 50;
    cam.vfov = 30;
    cam.lookfrom = point(3, 32, 3);
    cam.lookat = point(0, 0, 0);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = 0.6;
    cam.focus_dist = 30.0;
    cam.setup();

    camera* d_cam;
    gpuErrchk(cudaMalloc((void**)&d_cam, sizeof(camera)), "Failed to allocate memory to cam.");
    gpuErrchk(cudaMemcpy(d_cam, &cam, sizeof(camera), cudaMemcpyHostToDevice), "Failed to copy camera data from host to device.");

    StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

    createImage<<<GRIDDIM, BLOCKDIM>>>(d_image, width, height, d_cam, d_world, cam.max_depth, cam.samples_per_pixel);

    gpuErrchk( cudaPeekAtLastError(), "Error while launching kernel.");
    gpuErrchk( cudaDeviceSynchronize(), "Device Synchronization create image.");
    
    sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;
    
    std::cerr << "Took " << timeOnGPU / CLOCKS_PER_SEC << " seconds to create image.\n";

    gpuErrchk(cudaMemcpy(image, d_image, width*height*4*sizeof(char), cudaMemcpyDeviceToHost), "Failed to get image data from device to host.");

    gpuErrchk(cudaFree(d_world), "Failed to free world data on device.");
    gpuErrchk(cudaFree(d_image), "Failed to free image data on device.");
    gpuErrchk(cudaFree(d_cam), "Failed to free camera data on device.");
    
}





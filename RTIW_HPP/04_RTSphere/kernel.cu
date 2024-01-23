#include "kernel.cuh"
#include "point.h"
#include "color.h"
#include "ray.h"

#define gpuErrchk(ans, msg) { gpuAssert((ans), __FILE__, __LINE__, (msg)); }
inline void gpuAssert(cudaError_t code, const char *file, int line, const char *msg, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"%s\nGPUassert: %s %s %d\n\n", msg, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__ __device__ bool hit_sphere(const point& center, float radius, const Ray& r) {
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = half_b * half_b - a * c;
    return (discriminant >= 0);        
}


__host__ __device__ color ray_color(const Ray& r) {
    if(hit_sphere(point(0.0f, 0.0f, -1.0f), 0.5f, r)) {
        return color(1.0f, 0.0f, 0.0f);
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
}


__global__ void createImage(unsigned char* image, int width, int height, vec3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v, vec3 camera_center) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < height && i < width) {
        auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        auto ray_direction = pixel_center - camera_center;
        Ray r(camera_center, ray_direction);
        color pixel_color = ray_color(r);
        write_color(image, j, i, width, pixel_color);
    }
}



void CudaWrapper::cudaKernel(dim3 gridDim, dim3 blockDim, unsigned char* image, int width, int height, vec3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v, vec3 camera_center) {
    createImage<<<gridDim, blockDim>>>(image, width, height, pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center);
}


void CudaWrapper::cudaMain(unsigned char* image, int width, int height) {

    vec3 camera_center = point(0.0f, 0.0f, 0.0f);

    auto focal_length = 1.0f;
    float viewport_height = 2.0f;
    float viewport_width = viewport_height * ((float)width/ (float)height);

    auto viewport_u = vec3(viewport_width, 0.0f, 0.0f);
    auto viewport_v = vec3(0.0f, -viewport_height, 0.0f);
    
    auto pixel_delta_u = viewport_u / width;
    auto pixel_delta_v = viewport_v / height;

    auto viewport_upper_left = camera_center - vec3(0.0f, 0.0f, focal_length) - viewport_u / 2.0f - viewport_v / 2.0f;

    auto pixel00_loc = viewport_upper_left + (0.5f) * (pixel_delta_u + pixel_delta_v);

    unsigned char* d_image;

    gpuErrchk(cudaMalloc((void**)&d_image, width*height*4*sizeof(char)), "Failed to allocate memory to image.");

    // Define thread block size (e.g., 16x16)
    dim3 blockDim = dim3(8, 8);

    // Define grid size
    dim3 gridDim = dim3((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    CudaWrapper::cudaKernel(gridDim, blockDim, d_image, width, height, pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center);

    gpuErrchk( cudaPeekAtLastError(), "Error while launching kernel.");
    gpuErrchk( cudaDeviceSynchronize(), "Device Synchronization.");

    gpuErrchk(cudaMemcpy(image, d_image, width*height*4*sizeof(char), cudaMemcpyDeviceToHost), "Failed to get image data from device to host.");

    gpuErrchk(cudaFree(d_image), "Failed to free image data on device.");
}




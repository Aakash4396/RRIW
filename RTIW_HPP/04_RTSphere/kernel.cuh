#include "cuda_runtime.h"
#include "device_launch_parameters.h"
class vec3;
class camera;
namespace CudaWrapper {
    void cudaKernel(dim3, dim3, unsigned char*, int, int, vec3, vec3, vec3, vec3);
    void cudaMain(unsigned char*, int, int);
}



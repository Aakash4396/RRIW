#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
class vec3;
class camera;
class hittable;
namespace CudaWrapper {
    void cudaKernel(dim3, dim3, unsigned char*, int, int, camera*, hittable&, int);
    void cudaMain(unsigned char*, int, int);
}

#define gpuErrchk(ans, msg) { gpuAssert((ans), __FILE__, __LINE__, (msg)); }
inline void gpuAssert(cudaError_t code, const char *file, int line, const char *msg, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"%s\nGPUassert: %s %s %d\n\n", msg, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


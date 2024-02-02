#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#define IMG_WIDTH 1200
#define IMG_HEIGHT (IMG_WIDTH * 9 / 16)
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16
#define BLOCKDIM dim3(BLOCKDIM_X, BLOCKDIM_Y)
#define GRIDDIM_X ((IMG_WIDTH + BLOCKDIM_X - 1) / BLOCKDIM_X) 
#define GRIDDIM_Y ((IMG_HEIGHT + BLOCKDIM_Y - 1) / BLOCKDIM_Y)
#define GRIDDIM dim3(GRIDDIM_X, GRIDDIM_Y)
#define TOTAL_THREADS (GRIDDIM_X * BLOCKDIM_X * GRIDDIM_Y * BLOCKDIM_Y)

__device__ curandState threadRandomStates[TOTAL_THREADS];

namespace CudaWrapper {
   void cudaMain(unsigned char*, int, int);
}

#define gpuErrchk(ans, msg) { gpuAssert((ans), __FILE__, __LINE__, (msg)); }
inline void gpuAssert(cudaError_t code, const char *file, int line, const char *msg, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"%s\nGPUassert: %s %s %d\n\n", msg, cudaGetErrorString(code), file, line);
      if (abort) {
         cudaDeviceReset();
         exit(code);
      }
   }
}


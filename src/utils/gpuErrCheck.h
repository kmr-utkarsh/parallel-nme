#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#define gpuErrCheck(ans) { gpuAssert((ans), __LINE__); }
inline void gpuAssert(cudaError_t, int);

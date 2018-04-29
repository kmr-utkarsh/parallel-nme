#include "gpuErrCheck.h"

inline void gpuAssert(cudaError_t code, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUAssert: %s %s %dn", cudaGetErrorString(code), line);
      		exit(code);
	}
}

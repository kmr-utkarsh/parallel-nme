/*
    *
    *   @title
    *       Compute GTOM Matrix from a Gene Correlation Matrix.
    *
    *   @description
    *       This program reads the gene-correlation-matrix from a
    *       CSV file, stores the matrix into an array, then spawns
    *       a kernel in GPU to compute the GTOM matrix. The GTOM matrix
    *       is computed in a parallel fashion on the GPU.
    *
    *   @authors
    *       Kumar Utkarsh <kumarutkarsh.ingen@gmail.com>
    *       Bikash Jaiswal <bjjaiswal@gmail.com>
    *
    *	@date
    *	    Apr 10 2017
    *
    *
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include "utils/readDataset.h"
#include "utils/gpuErrCheck.h"

#define gpuErrCheck(ans) { gpuAssert((ans), __LINE__); }
inline void gpuAssert(cudaError_t code, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), line);
		exit(code);
	}
}

#define MAX_LINE_LENGTH 100000

int main (int argc, char *argv[]) {

	if(argc < 4) {
		fprintf(stderr, "Usage: %s <input file> <output file> <m>\n", argv[0]);
		exit(1);
	}

	int val_m = 0;
	int m = atoi(argv[3]);
	if(m <= 0) {
		fprintf(stderr, "Invalid Value for M\n");
		exit(1);
	}

	val_m = m + 1;

	int width, height, i;
	float *matrix = NULL;
	float *output[val_m];

	/*
	--------------------------------------------------------------------
		Import Dataset
	--------------------------------------------------------------------
	The readDataset() function imports the correlation matrix from a .csv
	file and returns a pointer to the allocated memory in heap.
	*/

	matrix = readDataset(&height, &width, argv[1]);

	/*
	--------------------------------------------------------------------
		Generate Structure for Storing Intermediate Matrices
	--------------------------------------------------------------------
	The output[] is an array of pointers to 2D matrices of order equal to
	that of the input matrix. Each 'ith' matrix stores A^(i-1), A being
	the correlation matrix (input) itself.

	*/
	for(i = 0; i < val_m; i++) {
		output[i] = (float *)malloc(sizeof(float) * width * height);
		memset(output[i], 0x0, sizeof(float) * width * height);
	}

	// Copy the input matrix at output[0]
	memcpy(output[0], matrix, sizeof(float) * width * height);

	// Transfer the intermediate structure to GPU memory
	float *d_output[val_m];


	for(i = 0; i < val_m; i++) {
		gpuErrCheck(cudaMalloc(&d_output[i], sizeof(float) * width * height));
		gpuErrCheck(cudaMemcpy(d_output[i], output[i], sizeof(float) * height * width, cudaMemcpyHostToDevice));
	}

	clock_t start, end;
	double ct_time;

	/*
	-----------------------------------------------------------------------
		Multiply Matrices
	-----------------------------------------------------------------------
	cublasSgemm() calculates A^2, A^3, A^4, ... A^m, where m is val_m
	*/
	const float alp = 1.0;
	const float bta = 0.0;
	const float *alpha = &alp;
	const float *beta = &bta;

	cublasHandle_t handle;
	cublasCreate(&handle);
	start = clock();
	for(i = 1; i < val_m; i++) {
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, height, height, height, alpha, d_output[0], height, d_output[i-1], height, beta, d_output[i], height);
	}

	/*

	for(i = 0; i < val_m; i++) {
		gpuErrCheck(cudaMemcpy(output[i], d_output[i], sizeof(float) * height * width, cudaMemcpyDeviceToHost));
	}

	printf("Original:\n");
	printDataset(output[0], height, width);
	for(i = 1; i < val_m; i++) {
		printf("Power %d:\n", i+1);
		printDataset(output[i], height, width);
	}

	*/

	/*
	---------------------------------------------------------------------
		Addition of Intermediate Matrices
	---------------------------------------------------------------------
	Adding A, A^2, A^3, ..., A^m using parallell routines

	*/
	float *h_gtom, *d_gtom;

	h_gtom = (float *) malloc(sizeof(float) * height * width);
	memset(h_gtom, 0x0, sizeof(float) * height * width);
	gpuErrCheck(cudaMalloc((void **)&d_gtom, sizeof(float) * width * height));
	gpuErrCheck(cudaMemcpy(d_gtom, h_gtom, sizeof(float) * height * width, cudaMemcpyHostToDevice));

	const float add_alp = 1.0;
	const float add_bta = 1.0;
	const float *add_alpha = &add_alp;
	const float *add_beta = &add_bta;

	for(i = 0; i < val_m; i++) {
		cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, height, add_alpha, d_gtom, height, add_beta, d_output[i], height, d_gtom, height);
	}

	end = clock();
	ct_time = ((double) (end - start)) / CLOCKS_PER_SEC;

	gpuErrCheck(cudaMemcpy(h_gtom, d_gtom, sizeof(float) * width * height, cudaMemcpyDeviceToHost));

	writeDataset(argv[2], h_gtom, height, width);

	cublasDestroy(handle);
	cudaFree(d_gtom);
	free(h_gtom);
	cudaFree(d_output);
	free(matrix);
	printf("Time taken: %f seconds\n", ct_time);
	return 0;

}



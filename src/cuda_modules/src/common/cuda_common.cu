#include "pch.h"

__global__ void add(int n, float* y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	y[index] = 3.0f;
}

bool CMCudaTest(void) {
	int N = 1 << 16;
	float *y;

	// Allocate Unified Memory ¨C accessible from CPU or GPU
	cudaMallocManaged(&y, N * sizeof(float));

	CUDA_KERNAL_CALL(add, N)(N, y);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	// Check for errors
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i] - 3.0f));

	// Free memory
	cudaFree(y);

	return fabs(maxError) < 1e-5f;
}

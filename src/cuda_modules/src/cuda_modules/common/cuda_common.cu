#include "cuda_common.h"

#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void add(int n, float* y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	y[index] = 3.0f;
}

bool CMCudaTest(void) {
	int N = 1 << 16;
	float *y;

	// Allocate Unified Memory ¨C accessible from CPU or GPU
	cudaMallocManaged(&y, N * sizeof(float));

	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	CUDA_KERNAL_CALL(add, numBlocks, blockSize)(N, y);

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

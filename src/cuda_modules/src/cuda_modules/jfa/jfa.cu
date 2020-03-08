#include "pch.h"

#include "src/cuda_modules/common/cuda_common.h"

#include "jfa.h"

__global__ void CMJFAIterate(float *dfx, float *dfy, float *dfx_tgt, float *dfy_tgt, int step, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	dfx_tgt[index] = dfx[index];
	dfy_tgt[index] = dfy[index];

	float crtDist = hypot(dfx[index], dfy[index]);

	int row = index / size, column = index % size;


	// first row
	int i = index - step * (size + 1), r = row - step, c = column - step;
	if (r >= 0 && r < size && c >= 0 && c < size) {
		float d = hypot(dfx[i] - step, dfy[i] - step);
		if (d < crtDist) {
			crtDist = d;
			dfx_tgt[index] = dfx[i] - step;
			dfy_tgt[index] = dfy[i] - step;
		}
	}

	i += step;
	c += step;
	if (r >= 0 && r < size && c >= 0 && c < size) {
		float d = hypot(dfx[i] - step, dfy[i]);
		if (d < crtDist) {
			crtDist = d;
			dfx_tgt[index] = dfx[i] - step;
			dfy_tgt[index] = dfy[i];
		}
	}

	i += step;
	c += step;
	if (r >= 0 && r < size && c >= 0 && c < size) {
		float d = hypot(dfx[i] - step, dfy[i] + step);
		if (d < crtDist) {
			crtDist = d;
			dfx_tgt[index] = dfx[i] - step;
			dfy_tgt[index] = dfy[i] + step;
		}
	}

	// second row
	i = index - step, r = row, c = column - step;
	if (r >= 0 && r < size && c >= 0 && c < size) {
		float d = hypot(dfx[i], dfy[i] - step);
		if (d < crtDist) {
			crtDist = d;
			dfx_tgt[index] = dfx[i];
			dfy_tgt[index] = dfy[i] - step;
		}
	}

	i += step;
	c += step;
	// skip the pixel itself

	i += step;
	c += step;
	if (r >= 0 && r < size && c >= 0 && c < size) {
		float d = hypot(dfx[i], dfy[i] + step);
		if (d < crtDist) {
			crtDist = d;
			dfx_tgt[index] = dfx[i];
			dfy_tgt[index] = dfy[i] + step;
		}
	}

	// third row
	i = index + step * (size - 1), r = row + step, c = column - step;
	if (r >= 0 && r < size && c >= 0 && c < size) {
		float d = hypot(dfx[i] + step, dfy[i] - step);
		if (d < crtDist) {
			crtDist = d;
			dfx_tgt[index] = dfx[i] + step;
			dfy_tgt[index] = dfy[i] - step;
		}
	}

	i += step;
	c += step;
	if (r >= 0 && r < size && c >= 0 && c < size) {
		float d = hypot(dfx[i] + step, dfy[i]);
		if (d < crtDist) {
			crtDist = d;
			dfx_tgt[index] = dfx[i] + step;
			dfy_tgt[index] = dfy[i];
		}
	}

	i += step;
	c += step;
	if (r >= 0 && r < size && c >= 0 && c < size) {
		float d = hypot(dfx[i] + step, dfy[i] + step);
		if (d < crtDist) {
			crtDist = d;
			dfx_tgt[index] = dfx[i] + step;
			dfy_tgt[index] = dfy[i] + step;
		}
	}
}

static CMJFAStat CMJFAStatInstance;

DLLEXPORT void CMJFACalculate(float* dfx, float* dfy, int size) {
	int step = 1 << int(std::ceil(std::log2(size))) - 1;
#ifdef _DEBUG
	std::cout << "First step length: " << step << std::endl;
#endif

	int memSize = size * size * sizeof(float);
	float *dfx_gpu, *dfy_gpu, *dfx_gpu_tgt, *dfy_gpu_tgt;
	cudaMallocManaged(&dfx_gpu, memSize);
	cudaMallocManaged(&dfy_gpu, memSize);
	cudaMallocManaged(&dfx_gpu_tgt, memSize);
	cudaMallocManaged(&dfy_gpu_tgt, memSize);
	memcpy(dfx_gpu, dfx, memSize);
	memcpy(dfy_gpu, dfy, memSize);

	int cellSize = size * size;
	while (step > 0) {
		CUDA_KERNAL_CALL(CMJFAIterate, cellSize)(dfx_gpu, dfy_gpu, dfx_gpu_tgt, dfy_gpu_tgt, step, size);

		cudaDeviceSynchronize();

		std::swap(dfx_gpu, dfx_gpu_tgt);
		std::swap(dfy_gpu, dfy_gpu_tgt);

		step >>= 1;
	}

	memcpy(dfx, dfx_gpu, memSize);
	memcpy(dfy, dfy_gpu, memSize);

	CMJFAStatInstance.maxDist = hypotf(float(size), float(size)) / 2;
}

DLLEXPORT CMJFAStat CMJFAGetStat() {
	return CMJFAStatInstance;
}

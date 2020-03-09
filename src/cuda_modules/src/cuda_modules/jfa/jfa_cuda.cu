#include "pch.h"

#include "thrust/extrema.h"
#include "thrust/device_ptr.h"

#include "jfa.h"

#include "jfa_cuda.h"

__global__ void CMJFAInitDFKernel1(float *dfx, float *dfy, float large, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= size * size) {
		return;
	}

	dfx[index] = dfy[index] = large;
}

__global__ void CMJFAInitDFKernel2(float *dfx, float *dfy, CMJFAInitPoint *points, int size, int pointNum) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= pointNum) {
		return;
	}

	int dfBase = points[index].x * size + points[index].y;
	dfx[dfBase] = points[index].vx;
	dfy[dfBase] = points[index].vy;
}

void CMJFAInitDF(float *dfx, float *dfy, CMJFAInitPoint *points, int pointNum, int size) {
	int cellSize = size * size;
	CUDA_KERNAL_CALL(CMJFAInitDFKernel1, cellSize)(dfx, dfy, cellSize, size);
	cudaDeviceSynchronize();
	CUDA_KERNAL_CALL(CMJFAInitDFKernel2, pointNum)(dfx, dfy, points, size, pointNum);
	cudaDeviceSynchronize();
}

__global__ void CMJFAIterateKernel(float *dfx, float *dfy, float *dfx_tgt, float *dfy_tgt, float *df, int step, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= size * size) {
		return;
	}

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

	df[index] = crtDist;
}

void CMJFAIterate(float *dfx, float *dfy, float *dfx_tgt, float *dfy_tgt, float *df, int step, int size) {
	int cellSize = size * size;
	CUDA_KERNAL_CALL(CMJFAIterateKernel, cellSize)(dfx, dfy, dfx_tgt, dfy_tgt, df, step, size);
	cudaDeviceSynchronize();
}

float CMJFACalcMax(float *df, int size) {
	int cellSize = size * size;
	thrust::device_ptr<float> dfdp = thrust::device_pointer_cast(df);
	auto m_gpu = thrust::max_element(dfdp, dfdp + cellSize);
	float m;
	cudaMemcpy(&m, m_gpu.get(), sizeof(float), cudaMemcpyDeviceToHost);
#ifdef _DEBUG
	std::cout << "Max Dist: " << m << std::endl;
#endif
	return m;
}

__global__ void CMJFACalcTextureRGBAKernel(float *df, unsigned char *rgba, float m, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= size * size) {
		return;
	}

	int ri = index * 4;

	float rd = df[index] / m;
	unsigned char c = lround(rd * 255.f);
	rgba[ri] = rgba[ri + 1] = rgba[ri + 2] = c;
	rgba[ri + 3] = 255;
}

void CMJFACalcTextureRGBA(float *df, unsigned char *rgba, int size, float m) {
	int cellSize = size * size;
	CUDA_KERNAL_CALL(CMJFACalcTextureRGBAKernel, cellSize)(df, rgba, m, size);
	cudaDeviceSynchronize();
}
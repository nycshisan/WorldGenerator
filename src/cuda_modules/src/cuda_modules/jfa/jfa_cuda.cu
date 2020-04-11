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

__device__ void CMJFAInitDFKernel2Point(float *dfx, float *dfy, int ix, int iy, float dx, float dy, int size) {
	int pid = iy * size + ix;
	float nd = hypot(dx, dy), od = hypot(dfx[pid], dfy[pid]);
	if (nd < od) {
		dfx[pid] = dx;
		dfy[pid] = dy;
	}
}

__global__ void CMJFAInitDFKernel2(float *dfx, float *dfy, CMJFAInitPoint *points, int size, int pointNum) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= pointNum) {
		return;
	}

	float x = points[index].px, y = points[index].py;
	int left = floor(x), right = left + 1;
	int top = floor(y), down = top + 1;

	CMJFAInitDFKernel2Point(dfx, dfy, left, top, x - left, y - top, size);
	CMJFAInitDFKernel2Point(dfx, dfy, right, top, right - x, y - top, size);
	CMJFAInitDFKernel2Point(dfx, dfy, left, down, x - left, down - y, size);
	CMJFAInitDFKernel2Point(dfx, dfy, right, down, right - x, down - y, size);
}

void CMJFAInitDF(float *dfx, float *dfy, CMJFAInitPoint *points, int pointNum, int size) {
	int cellSize = size * size;
	CUDA_KERNAL_CALL(CMJFAInitDFKernel1, cellSize)(dfx, dfy, float(cellSize), size);
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

__global__ void CMJFARemoveSignKernel(float *dfx, float *dfy, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= size * size) {
		return;
	}

	dfx[index] = abs(dfx[index]);
	dfy[index] = abs(dfy[index]);
}

void CMJFARemoveSign(float *dfx, float *dfy, int size) {
	int cellSize = size * size;
	CUDA_KERNAL_CALL(CMJFARemoveSignKernel, cellSize)(dfx, dfy, size);
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

	float rd = 1.f - df[index] / m;
	unsigned char c = lround(rd * 255.f);
	rgba[ri] = rgba[ri + 1] = rgba[ri + 2] = c;
	rgba[ri + 3] = 255;
}

void CMJFACalcTextureRGBA(float *df, unsigned char *rgba, int size, float m) {
	int cellSize = size * size;
	CUDA_KERNAL_CALL(CMJFACalcTextureRGBAKernel, cellSize)(df, rgba, m, size);
	cudaDeviceSynchronize();
}
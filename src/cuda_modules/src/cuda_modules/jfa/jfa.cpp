#include "pch.h"

#include "jfa.h"

#include "jfa_cuda.h"

//#define CMJFA_PROFILE_MEM
//#define CMJFA_PROFILE_INIT
//#define CMJFA_PROFILE_ITERATION
//#define CMJFA_PROFILE_STAT
//#define CMJFA_PROFILE_GENTEX
//#define CMJFA_DUMP

DLLEXPORT CMJFAHandle *CMJFAHandleAlloc(int size) {
	auto handle = new CMJFAHandle;

	handle->size = size;
	int memSize = size * size * sizeof(float);

#ifdef CMJFA_PROFILE_MEM
	CMGetClockSecond();
#endif
	cudaMalloc(&handle->dfx_gpu, memSize);
	cudaMalloc(&handle->dfy_gpu, memSize);
	cudaMalloc(&handle->dfx_gpu_tgt, memSize);
	cudaMalloc(&handle->dfy_gpu_tgt, memSize);
	cudaMalloc(&handle->df_gpu, memSize);
#ifdef CMJFA_PROFILE_MEM
	std::cout << "Malloc df time cost: " << CMGetClockSecond() << std::endl;
#endif

	return handle;
}

DLLEXPORT void CMJFASetInitPoint(CMJFAHandle *handle, float px, float py) {
	handle->initializer.addPoint(px, py);
}

DLLEXPORT void CMJFACalculate(CMJFAHandle *handle, float *dfx_tgt, float *dfy_tgt) {
	handle->init();

	int size = handle->size, step = 1 << (int(std::ceil(std::log2(size))) - 1);
	
#ifdef _DEBUG
	std::cout << "First step length: " << step << std::endl;
#endif

#ifdef CMJFA_DUMP
	std::cout << "Dumped initial results." << std::endl;
	CMJFADumpData(handle);
#endif

	int iterNum = 0;
	while (step > 0) {
		float *dfx_gpu = handle->dfx_gpu,
			*dfy_gpu = handle->dfy_gpu,
			*dfx_gpu_tgt = handle->dfx_gpu_tgt,
			*dfy_gpu_tgt = handle->dfy_gpu_tgt,
			*df_gpu = handle->df_gpu;

		iterNum++;

#ifdef CMJFA_PROFILE_ITERATION
		CMGetClockSecond();
#endif
		CMJFAIterate(dfx_gpu, dfy_gpu, dfx_gpu_tgt, dfy_gpu_tgt, df_gpu, step, size);
#ifdef CMJFA_PROFILE_ITERATION
		std::cout << "The " << iterNum << "th iteration time cost: " << CMGetClockSecond() << std::endl;
#endif

		std::swap(handle->dfx_gpu, handle->dfx_gpu_tgt);
		std::swap(handle->dfy_gpu, handle->dfy_gpu_tgt);

#ifdef CMJFA_DUMP
		std::cout << "Dumped " << iterNum << "th iteration results." << std::endl;
		CMJFADumpData(handle);
#endif

		step >>= 1;
	}
	CMJFARemoveSign(handle->dfx_gpu, handle->dfy_gpu, size);

#ifdef CMJFA_PROFILE_MEM
	CMGetClockSecond();
#endif
	int memSize = size * size * sizeof(float);
	cudaMemcpy(dfx_tgt, handle->dfx_gpu, memSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(dfy_tgt, handle->dfy_gpu, memSize, cudaMemcpyDeviceToHost);
#ifdef CMJFA_PROFILE_MEM
	std::cout << "Memcpy output time cost: " << CMGetClockSecond() << std::endl;
#endif

	// make statistic
#ifdef CMJFA_PROFILE_STAT
	CMGetClockSecond();
#endif
	handle->stat.maxDist = CMJFACalcMax(handle->df_gpu, size);
#ifdef CMJFA_PROFILE_STAT
	std::cout << "Stat time cost: " << CMGetClockSecond() << std::endl;
#endif
}

DLLEXPORT void CMJFAGenerateTexture(CMJFAHandle *handle, unsigned char *rgba) {
#ifdef CMJFA_PROFILE_GENTEX
	CMGetClockSecond();
#endif

	int size = handle->size;
	int memSize = size * size * 4 * sizeof(unsigned char);

	unsigned char *rgba_gpu;
	cudaMalloc(&rgba_gpu, memSize);

	CMJFACalcTextureRGBA(handle->df_gpu, rgba_gpu, size, handle->stat.maxDist);
	cudaMemcpy(rgba, rgba_gpu, memSize, cudaMemcpyDeviceToHost);

	cudaFree(rgba_gpu);

#ifdef CMJFA_PROFILE_GENTEX
	std::cout << "Texture generation time cost: " << CMGetClockSecond() << std::endl;
#endif
}

DLLEXPORT void CMJFAHandleFree(CMJFAHandle *handle) {
	cudaFree(handle->dfx_gpu);
	cudaFree(handle->dfy_gpu);
	cudaFree(handle->dfx_gpu_tgt);
	cudaFree(handle->dfy_gpu_tgt);
	cudaFree(handle->df_gpu);

	delete handle;
}

float CMJFAGetStat(CMJFAHandle *handle, CMJFAStatType type) {
	switch (type) {
	case CMJFAStatType::MaxDist:
		return handle->stat.maxDist;
		break;
	default:
		std::cout << "Invalid statistic type!" << std::endl;
		return 0.f;
	}
}

CMJFADump CMJFADumpData(CMJFAHandle *handle) {
	CMJFADump dump;

	int size = handle->size;

	int memSize = size * size * sizeof(float);

	dump.dfx = new float[memSize];
	dump.dfy = new float[memSize];
	dump.df = new float[memSize];

	cudaMemcpy(dump.dfx, handle->dfx_gpu, memSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(dump.dfy, handle->dfy_gpu, memSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(dump.df, handle->df_gpu, memSize, cudaMemcpyDeviceToHost);

	std::cout << "dfx:" << std::endl;
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			std::cout << dump.dfx[i * size + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << "dfy:" << std::endl;
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			std::cout << dump.dfy[i * size + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << "df:" << std::endl;
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			std::cout << dump.df[i * size + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	return dump;
}

void CMJFAInitializer::exportToGPU(CMJFAInitPoint *points_gpu) {
	CMJFAInitPoint *points_data = points.data();


	cudaMemcpy(points_gpu, points_data, sizeof(CMJFAInitPoint) * pointNum, cudaMemcpyHostToDevice);
}

void CMJFAInitializer::addPoint(float px, float py) {
	pointNum++;
	points.emplace_back(px, py);
}

void CMJFAHandle::init() {
#ifdef CMJFA_PROFILE_INIT
	CMGetClockSecond();
#endif
	CMJFAInitPoint *points_gpu;
	cudaMalloc(&points_gpu, initializer.pointNum * sizeof(CMJFAInitPoint));
	initializer.exportToGPU(points_gpu);
#ifdef CMJFA_PROFILE_INIT
	std::cout << "Prepare initial value time cost: " << CMGetClockSecond() << std::endl;
#endif

#ifdef CMJFA_PROFILE_INIT
	CMGetClockSecond();
#endif
	CMJFAInitDF(dfx_gpu, dfy_gpu, points_gpu, initializer.pointNum, size);
#ifdef CMJFA_PROFILE_INIT
	std::cout << "Initialization time cost: " << CMGetClockSecond() << std::endl;
#endif
	cudaFree(points_gpu);
}

inline CMJFAInitPoint::CMJFAInitPoint(float x, float y) : px(x), py(y) {}

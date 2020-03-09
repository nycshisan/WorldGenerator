#pragma once

#include "common/common.h"

struct CMJFAStat {
	float maxDist = 0;
};

struct CMJFAHandle {
	float *dfx_gpu, *dfy_gpu, *dfx_gpu_tgt, *dfy_gpu_tgt, *df_gpu;
	int size;

	CMJFAStat stat;
};

struct CMJFAInitPoint {
	int x, y;
	float vx, vy;
};

DLLEXPORT CMJFAHandle *CMJFAHandleAlloc(int size);

DLLEXPORT void CMJFAInit(CMJFAHandle *handle, CMJFAInitPoint *points, int pointNum);

DLLEXPORT void CMJFACalculate(CMJFAHandle *handle, float *dfx_tgt, float *dfy_tgt);

DLLEXPORT void CMJFAGenerateTexture(CMJFAHandle *handle, unsigned char *rgba);

DLLEXPORT void CMJFAHandleFree(CMJFAHandle *handle);

struct CMJFADump {
	float *dfx, *dfy, *df;
};

DLLEXPORT CMJFADump CMJFADumpData(CMJFAHandle *handle);
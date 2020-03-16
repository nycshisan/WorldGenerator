#pragma once

#include "common/common.h"

struct CMJFAInitPoint {
	float px, py;

	CMJFAInitPoint(float x, float y);
};

struct CMJFAInitializer {
	int pointNum = 0;
	std::vector<CMJFAInitPoint> points;

	void exportToGPU(CMJFAInitPoint *points_gpu);
	void addPoint(float px, float py);
};

struct CMJFAStat {
	float maxDist = 0;
};

struct CMJFAHandle {
	float *dfx_gpu = nullptr, *dfy_gpu = nullptr,
		*dfx_gpu_tgt = nullptr, *dfy_gpu_tgt = nullptr, *df_gpu = nullptr;
	int size = -1;

	CMJFAInitializer initializer;
	CMJFAStat stat;

	void init();
};

DLLEXPORT CMJFAHandle *CMJFAHandleAlloc(int size);

DLLEXPORT void CMJFASetInitPoint(CMJFAHandle *handle, float px, float py);

DLLEXPORT void CMJFACalculate(CMJFAHandle *handle, float *dfx_tgt, float *dfy_tgt);

DLLEXPORT void CMJFAGenerateTexture(CMJFAHandle *handle, unsigned char *rgba);

DLLEXPORT void CMJFAHandleFree(CMJFAHandle *handle);

enum class CMJFAStatType : int {
	MaxDist,
};

DLLEXPORT float CMJFAGetStat(CMJFAHandle *handle, CMJFAStatType type);

struct CMJFADump {
	float *dfx, *dfy, *df;
};

DLLEXPORT CMJFADump CMJFADumpData(CMJFAHandle *handle);
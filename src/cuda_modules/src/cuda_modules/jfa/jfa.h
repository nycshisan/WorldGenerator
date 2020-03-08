#pragma once

#include "../common/common.h"

DLLEXPORT void CMJFACalculate(float *dfx, float *dfy, int size);

typedef struct {
	float maxDist = 0;
} CMJFAStat;

DLLEXPORT CMJFAStat CMJFAGetStat();
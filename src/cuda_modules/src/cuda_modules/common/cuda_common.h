#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define BLOCK_SIZE 256
#define NUM_BLOCKS(N) (N + BLOCK_SIZE - 1) / BLOCK_SIZE

#define CUDA_KERNAL_CALL(NAME, THREAD_NUM) NAME<<<NUM_BLOCKS(THREAD_NUM), BLOCK_SIZE>>>

bool CMCudaTest();
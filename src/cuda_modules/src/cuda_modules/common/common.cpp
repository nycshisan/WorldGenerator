#include "pch.h"

#include "cuda_common.h"

DLLEXPORT bool CMIsOK() {
	if (CMCudaTest()) {
#ifdef _DEBUG
		std::cout << "Cuda Modules DLL (Debug) is OK!" << std::endl;
#endif
		return true;
	} else {
		return false;
	}
}

DLLEXPORT void CMCheckCallArgumentOrder(char a, char b) {
	std::cout << "First: " << a << std::endl;
	std::cout << "Second: " << b << std::endl;
}

int* CMCheckMemory(int* src, int size) {
	auto r = new int[size];
	for (int i = 0; i < size; ++i) {
		r[i] = src[i];
	}
	return r;
}

void CMFreeArray(void* ptr) {
	delete[] ptr;
}

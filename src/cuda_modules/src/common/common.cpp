#include "pch.h"

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

DLLEXPORT int* CMCheckMemory(int* src, int size) {
	auto r = new int[size];
	for (int i = 0; i < size; ++i) {
		r[i] = src[i];
	}
	return r;
}

DLLEXPORT void CMFreeArray(void* ptr) {
	delete[] ptr;
}

DLLEXPORT double CMGetClockSecond() {
	static clock_t lastClock = 0;
	auto crt = clock();
	auto sec = double(crt - lastClock) / CLOCKS_PER_SEC;
	lastClock = crt;
	return sec;
}
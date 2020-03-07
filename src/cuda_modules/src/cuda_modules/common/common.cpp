#include "pch.h"

#include "cuda_common.h"

DLLEXPORT bool CMIsOK() {
	if (CMCudaTest()) {
		std::cout << "Cuda Modules DLL is OK!" << std::endl;
		return true;
	} else {
		return false;
	}
}

DLLEXPORT void CMCheckCallArgumentOrder(char a, char b) {
	std::cout << "First: " << a << std::endl;
	std::cout << "Second: " << b << std::endl;
}

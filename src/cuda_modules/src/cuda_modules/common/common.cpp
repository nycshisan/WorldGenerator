#include "pch.h"

DLLEXPORT bool CMIsOK() {
	std::cout << "Cuda Modules DLL is OK!" << std::endl;
	return true;
}

DLLEXPORT void CMCheckCallArgumentOrder(char a, char b) {
	std::cout << "First: " << a << std::endl;
	std::cout << "Second: " << b << std::endl;
}

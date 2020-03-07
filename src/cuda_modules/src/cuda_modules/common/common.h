#pragma once

#ifdef EXPORTING_DLL
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C" __declspec(dllimport)
#endif

DLLEXPORT bool CMIsOK();

DLLEXPORT void CMCheckCallArgumentOrder(char a, char b);
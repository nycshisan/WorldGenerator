#pragma once

void CMJFAInitDF(float *dfx, float *dfy, CMJFAInitPoint *points, int pointNum, int size);

void CMJFAIterate(float *dfx, float *dfy, float *dfx_tgt, float *dfy_tgt, float *df, int step, int size);

float CMJFACalcMax(float *df, int size);

void CMJFACalcTextureRGBA(float *df, unsigned char *rgba, int size, float m);
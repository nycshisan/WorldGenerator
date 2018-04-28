//
// Created by Nycshisan on 2018/4/26.
//

#ifndef WORLDGENERATOR_NOISE_H
#define WORLDGENERATOR_NOISE_H

class NoiseGenerator {
    static const unsigned permutation[];

public:
    static float PerlinNoise(float x, float y);
};

#endif //WORLDGENERATOR_NOISE_H

//
// Created by nycsh on 2020/2/23.
//

#ifndef WORLDGENERATOR_RANDOM_H
#define WORLDGENERATOR_RANDOM_H

#include <random>

namespace wg {

    class Random {
        std::mt19937 *_gen = nullptr;
        int _randomSeed;

        Random();
        static Random &SharedInstance();

    public:
        static void ResetRandomEngine();
        static int RandInt(int x, int y);
        static float RandFloat(float fx, float fy);
        static bool RandBinary();
    };

}

#endif //WORLDGENERATOR_RANDOM_H

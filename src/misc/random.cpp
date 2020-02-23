//
// Created by nycsh on 2020/2/23.
//

#include "random.h"

#include "../conf/conf.h"

wg::Random::Random() {
    _randomSeed = CONF.getMapRandomSeed();
    _gen = new std::mt19937(_randomSeed);
}

wg::Random &wg::Random::SharedInstance() {
    static Random instance;
    return instance;
}

int wg::Random::RandInt(int x, int y) {
    std::uniform_int_distribution<int> dist(x, y);
    return dist(*SharedInstance()._gen);
}

float wg::Random::RandFloat(float fx, float fy) {
    std::uniform_real_distribution<float> dist(fx, fy);
    return dist(*SharedInstance()._gen);
}

void wg::Random::ResetRandomEngine() {
    delete SharedInstance()._gen;
    SharedInstance()._gen = new std::mt19937(SharedInstance()._randomSeed);
}

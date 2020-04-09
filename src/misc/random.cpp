//
// Created by nycsh on 2020/2/23.
//

#include "random.h"

#include "../conf/conf.h"

namespace wg {
    
    Random::Random() {
        _randomSeed = CONF.getMapRandomSeed();
        _gen = new std::mt19937(_randomSeed);
    }
    
    Random& Random::SharedInstance() {
        static Random instance;
        return instance;
    }
    
    int Random::RandInt(int x, int y) {
        std::uniform_int_distribution<int> dist(x, y);
        return dist(*SharedInstance()._gen);
    }
    
    int Random::RandInt(const std::pair<int, int> &pair) {
        return Random::RandInt(pair.first, pair.second);
    }
    
    float Random::RandFloat(float fx, float fy) {
        std::uniform_real_distribution<float> dist(fx, fy);
        return dist(*SharedInstance()._gen);
    }
    
    float Random::RandFloat(const std::pair<float, float> &pair) {
        return Random::RandFloat(pair.first, pair.second);
    }
    
    bool Random::RandBinary() {
        std::uniform_int_distribution<int> dist(0, 1);
        return bool(dist(*SharedInstance()._gen));
    }
    
    void Random::ResetRandomEngine() {
        delete SharedInstance()._gen;
        SharedInstance()._gen = new std::mt19937(SharedInstance()._randomSeed);
    }
    
}
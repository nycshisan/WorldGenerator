//
// Created by Nycshisan on 2018/4/15.
//

#ifndef WORLDGENERATOR_COAST_H
#define WORLDGENERATOR_COAST_H

#include "blocks.h"

#include <random>

class Coast {
public:
    typedef Blocks::Output Input;
private:
    Input _blockInfos;

    int _k; // Ocean blocks number
    int _randomSeed;

    std::mt19937 _gen;

    void _findOceanBlocks(std::vector<int> &indices, int begin, int size, int k);

public:
    void input(Input input);
    void generate();
    void draw(Drawer &drawer);
};

#endif //WORLDGENERATOR_COAST_H

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

public:
    void input(Input input);
    void generate();
    void draw(Drawer &drawer);
};

#endif //WORLDGENERATOR_COAST_H

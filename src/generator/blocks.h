//
// Created by Nycshisan on 2018/4/15.
//

#ifndef WORLDGENERATOR_BLOCKS_H
#define WORLDGENERATOR_BLOCKS_H

#include "lloyd.h"

#include <vector>
#include <memory>

#include "../data/blockInfo.h"

class Blocks {
public:
    typedef LloydRelaxation::Output Input;
    typedef std::vector<std::shared_ptr<BlockInfo>> Output;

private:
    Input _relaxedVd;
    Output _blockInfos;

public:
    void input(Input relaxedVd);
    void generate();
    Output output();
    void draw(Drawer &drawer);
};

#endif //WORLDGENERATOR_BLOCKS_H

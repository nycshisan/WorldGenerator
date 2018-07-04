//
// Created by Nycshisan on 2018/4/15.
//

#ifndef WORLDGENERATOR_BLOCKS_H
#define WORLDGENERATOR_BLOCKS_H

#include "lloyd.h"

#include <vector>
#include <memory>

#include "../data/blockInfo.h"

namespace wg {

    class BlocksDrawable {
    protected:
        void _prepareBlockVertexes(Drawer &drawer, const std::shared_ptr<BlockInfo> &blockInfo, const sf::Color &color);
    };

    class Blocks : protected BlocksDrawable {
    public:
        typedef LloydRelaxation::Output Input;
        typedef std::vector<std::shared_ptr<BlockInfo>> Output;

    private:
        Input _relaxedVd;
        Output _blockInfos;

    public:
        void input(const Input &relaxedVd);

        void generate();

        Output output();

        void prepareVertexes(Drawer &drawer);
    };

}

#endif //WORLDGENERATOR_BLOCKS_H

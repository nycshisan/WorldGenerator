//
// Created by Nycshisan on 2018/4/15.
//

#ifndef WORLDGENERATOR_BLOCKS_H
#define WORLDGENERATOR_BLOCKS_H

#include "../impl.h"

#include "../../data/blockInfo.h"

namespace wg {
    class Blocks : public GeneratorImpl {
    public:
        typedef std::vector<std::shared_ptr<BlockInfo>> Output;

    private:
        Output _blockInfos;

    public:
        std::string getHintLabelText() override;

        void generate() override;

        void prepareVertexes(Drawer &drawer) override;

        std::string save() override;

        std::string load() override;
    };

}

#endif //WORLDGENERATOR_BLOCKS_H

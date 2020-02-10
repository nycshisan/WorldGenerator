//
// Created by Nycshisan on 2018/4/15.
//

#ifndef WORLDGENERATOR_BLOCKS_H
#define WORLDGENERATOR_BLOCKS_H

#include "lloyd.h"

#include <vector>
#include <memory>

#include "../../data/blockInfo.h"
#include "../../conf/conf.h"

namespace wg {
    class Blocks : public GeneratorImpl {
    public:
        typedef LloydRelaxation::Output Input;
        typedef std::vector<std::shared_ptr<BlockInfo>> Output;

    private:
        Input _relaxedVd;
        Output _blockInfos;

    public:
        std::string getHintLabelText() override;

        void input(void* inputData) override;

        void generate() override;

        void* output() override;

        void prepareVertexes(Drawer &drawer) override;

        std::string save() override;

        std::string load() override;
    };

}

#endif //WORLDGENERATOR_BLOCKS_H

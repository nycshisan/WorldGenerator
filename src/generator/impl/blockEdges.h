//
// Created by nycsh on 2020/2/14.
//

#ifndef WORLDGENERATOR_BLOCKEDGES_H
#define WORLDGENERATOR_BLOCKEDGES_H

#include "../impl.h"

namespace wg {

    class BlockEdges : public GeneratorImpl {
    public:
        typedef std::vector<std::shared_ptr<BlockInfo>> Output;

    private:
        Output _blockInfos;

        std::set<std::shared_ptr<EdgeInfo>> _edges;

    public:
        std::string getHintLabelText() override;

        void generate() override;

        void prepareVertexes(Drawer &drawer) override;
    };

}

#endif //WORLDGENERATOR_BLOCKEDGES_H

//
// Created by nycsh on 2020/3/7.
//

#ifndef WORLDGENERATOR_DISTFIELD_H
#define WORLDGENERATOR_DISTFIELD_H

#include "../impl.h"

namespace wg {

    class DistField : public GeneratorImpl {
    public:
        typedef std::vector<std::shared_ptr<BlockInfo>> Output;

    private:
        Output _blockInfos;

        float *_dfx = nullptr, *_dfy = nullptr;

        sf::Sprite _s;
        sf::Texture _t;

    public:
        std::string getHintLabelText() override;

        void generate() override;

        void prepareVertexes(Drawer &drawer) override;
    };

}

#endif //WORLDGENERATOR_DISTFIELD_H

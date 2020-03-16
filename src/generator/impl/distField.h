//
// Created by nycsh on 2020/3/7.
//

#ifndef WORLDGENERATOR_DISTFIELD_H
#define WORLDGENERATOR_DISTFIELD_H

#include "../impl.h"

#include "cuda_modules/jfa/jfa.h"

namespace wg {

    class DistField : public GeneratorImpl {
    public:
        typedef std::vector<std::shared_ptr<BlockInfo>> Output;

    private:
        Output _blockInfos;

        float *_dfx = nullptr, *_dfy = nullptr;
        float _maxDist;

        sf::Sprite _s;
        sf::Texture _t;

        void _initJFA(CMJFAHandle *handle);

    public:
        std::string getHintLabelText() override;

        void generate() override;

        void prepareVertexes(Drawer &drawer) override;

        std::string save() override;

        std::string load() override;
    };

}

#endif //WORLDGENERATOR_DISTFIELD_H

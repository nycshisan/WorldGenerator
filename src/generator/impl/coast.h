//
// Created by Nycshisan on 2018/4/15.
//

#ifndef WORLDGENERATOR_COAST_H
#define WORLDGENERATOR_COAST_H

#include "blocks.h"

#include <random>

#include "../config.h"

namespace wg {

    class Coast : public GeneratorImpl {
    public:
        typedef Blocks::Output Input;
    private:
        Input _blockInfos;

    public:
        std::string getHintLabelText() override;

        void input(void* inputData) override;

        void generate() override;

        void* output() override;

        void prepareVertexes(Drawer &drawer) override;

        void getConfigs(Generator &generator) override;

        std::string save() override;

        std::string load() override;
    };

}

#endif //WORLDGENERATOR_COAST_H

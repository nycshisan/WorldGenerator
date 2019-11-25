//
// Created by nycsh on 2019/11/25.
//

#ifndef WORLDGENERATOR_IMPL_H
#define WORLDGENERATOR_IMPL_H

#include "../graphics/drawer.h"

namespace wg {

    class Generator;

    class GeneratorImpl {
    public:
        virtual std::string getHintLabelText() = 0;

        virtual void input(void* inputData) = 0;

        virtual void generate() = 0;

        virtual void* output() = 0;

        virtual void prepareVertexes(Drawer &drawer) = 0;

        virtual std::string save();

        virtual std::string load();

        bool hasConfigs = false;
        virtual void getConfigs(Generator &generator);
    };
}

#endif //WORLDGENERATOR_IMPL_H

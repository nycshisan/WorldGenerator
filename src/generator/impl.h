//
// Created by nycsh on 2019/11/25.
//

#ifndef WORLDGENERATOR_IMPL_H
#define WORLDGENERATOR_IMPL_H

#include "../graphics/graphics.h"
#include "../misc/misc.h"
#include "../conf/conf.h"
#include "../data/data.h"

namespace wg {

    class Generator;

    class GeneratorImpl {
    protected:
        void *_inputData = nullptr, *_outputData = nullptr;

    public:
        virtual std::string getHintLabelText() = 0;

        void input(void* inputData);

        virtual void generate() = 0;

        void* output();

        virtual void prepareVertexes(Drawer &drawer) = 0;

        virtual std::string save();

        virtual std::string load();

        bool hasConfigs = false;
        virtual void getConfigs(Generator &generator);
    };
}

#endif //WORLDGENERATOR_IMPL_H

//
// Created by nycsh on 2020/2/2.
//

#ifndef WORLDGENERATOR_FINISH_H
#define WORLDGENERATOR_FINISH_H

#include "../impl.h"

namespace wg {

    class Finish : public GeneratorImpl {
        bool _installFile(const std::string &fp);

    public:
        std::string getHintLabelText() override;

        void input(void* inputData) override;

        void generate() override;

        void* output() override;

        void prepareVertexes(Drawer &drawer) override;
    };

}

#endif //WORLDGENERATOR_FINISH_H

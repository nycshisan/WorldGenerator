//
// Created by Nycshisan on 2018/4/15.
//

#ifndef WORLDGENERATOR_COAST_H
#define WORLDGENERATOR_COAST_H

#include "blocks.h"

#include <random>

#include "config.h"

namespace wg {

    class Coast : protected BlocksDrawable {
    public:
        typedef Blocks::Output Input;
    private:
        Input _blockInfos;

    public:
        void input(const Input &input);

        void generate();

        void prepareVertexes(Drawer &drawer);

        void getConfigs(std::vector<std::shared_ptr<GeneratorConfig>> &configs);
    };

}

#endif //WORLDGENERATOR_COAST_H

//
// Created by Nycshisan on 2018/4/15.
//

#ifndef WORLDGENERATOR_COAST_H
#define WORLDGENERATOR_COAST_H

#include "blocks.h"

#include <random>

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
    };

}

#endif //WORLDGENERATOR_COAST_H

//
// Created by Nycshisan on 2018/3/18.
//

#ifndef WORLDGENERATOR_CENTERS_H
#define WORLDGENERATOR_CENTERS_H

#include <vector>

#include "SFML/Graphics.hpp"

#include "../../misc/misc.h"
#include "../impl.h"

namespace wg {

    class Drawer;

    class Centers : public GeneratorImpl {
    public:
        typedef std::vector<Point> Output;
    private:
        Output _centers;

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

#endif //WORLDGENERATOR_CENTERS_H

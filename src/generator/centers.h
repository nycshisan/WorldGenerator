//
// Created by Nycshisan on 2018/3/18.
//

#ifndef WORLDGENERATOR_CENTERS_H
#define WORLDGENERATOR_CENTERS_H

#include <vector>

#include "SFML/Graphics.hpp"

#include "../misc/misc.h"

namespace wg {

    class Drawer;

    class Centers {
    public:
        typedef std::vector<Point> Output;
    private:
        Output _centers;

    public:
        void input();

        void generate();

        Output output();

        void prepareVertexes(Drawer &drawer);

        bool save();

        bool load();
    };

}

#endif //WORLDGENERATOR_CENTERS_H

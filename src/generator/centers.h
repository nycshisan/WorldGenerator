//
// Created by Nycshisan on 2018/3/18.
//

#ifndef WORLDGENERATOR_CENTERS_H
#define WORLDGENERATOR_CENTERS_H

#include <vector>

#include "SFML/Graphics.hpp"

#include "../graphics/window.h"
#include "../graphics/drawer.h"
#include "../misc/misc.h"

class BlockCenters {
public:
    typedef std::vector<Point> Output;
private:
    Output _centers;

    int _width, _height;
    int _span;

public:
    void input();
    void generate();
    Output output();
    void draw(Drawer &drawer);
    void save();
    void load();
};

#endif //WORLDGENERATOR_CENTERS_H

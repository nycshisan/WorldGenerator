//
// Created by Nycshisan on 2018/3/18.
//

#ifndef WORLDGENERATOR_CENTERS_H
#define WORLDGENERATOR_CENTERS_H

#include <vector>

#include "SFML/Graphics.hpp"

#include "../graphics/window.h"
#include "../misc/misc.h"

class BlockCenters {
public:
    typedef std::vector<Point> Output;
private:
    Output _centers;

    unsigned int _width, _height;
    sf::CircleShape _pointShape;

public:
    void input(unsigned int width, unsigned int height);
    void generate();
    Output output();
    void draw(Window &window);
    void save();
    void load();
};

#endif //WORLDGENERATOR_CENTERS_H

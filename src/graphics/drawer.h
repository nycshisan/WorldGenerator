//
// Created by Nycshisan on 2018/4/16.
//

#ifndef WORLDGENERATOR_DRAWER_H
#define WORLDGENERATOR_DRAWER_H

#include "window.h"
#include "../data/blockInfo.h"

namespace wg {

    class Drawer {
        Window *_window;

        sf::CircleShape _pointShape;

        sf::VertexArray _pointsBuf, _linesBuf, _trisBuf;

    public:
        explicit Drawer();

        void setWindow(Window *window);

        void clearVertexes();
        void appendVertex(sf::PrimitiveType type, const sf::Vertex &vertex);
        void commit();
    };

}

#endif //WORLDGENERATOR_DRAWER_H

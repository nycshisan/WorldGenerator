//
// Created by Nycshisan on 2018/4/16.
//

#ifndef WORLDGENERATOR_DRAWER_H
#define WORLDGENERATOR_DRAWER_H

#include <SFML/Graphics.hpp>

#include "../data/blockInfo.h"

namespace wg {

    class MainWindow;

    class Drawer {
        static constexpr float _BasePointRadius = 1.0;

        MainWindow *_window = nullptr;

        sf::CircleShape _pointShape;

        sf::VertexArray _pointsBuf, _linesBuf, _trisBuf;

    public:
        explicit Drawer();

        void setWindow(MainWindow *window);

        void clearVertexes();
        void appendVertex(sf::PrimitiveType type, const sf::Vertex &vertex);
        void commit();
    };

}

#endif //WORLDGENERATOR_DRAWER_H

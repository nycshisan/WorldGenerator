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
        std::vector<sf::CircleShape> _coloredPointShapes;

        sf::VertexArray _pointsBuf, _linesBuf, _trisBuf, _pointShapeBuf;

        std::vector<const sf::Sprite*> _sprites;

    public:
        explicit Drawer();

        void setWindow(MainWindow *window);

        void clear();
        void appendVertex(sf::PrimitiveType type, const sf::Vertex &vertex);
        void appendPointShape(const sf::Vertex &vertex);
        void appendCustomPointShape(const sf::Vertex &vertex, const sf::Color &color, float size = -1);
        void drawThickLine(const std::shared_ptr<EdgeInfo> &edgeInfo, float thickness);
        void addSprite(const sf::Sprite &s);
        void commit();
    };

}

#endif //WORLDGENERATOR_DRAWER_H

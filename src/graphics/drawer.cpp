//
// Created by Nycshisan on 2018/4/16.
//

#include "drawer.h"
#include "../conf/conf.h"

Drawer::Drawer(Window *window) {
    _window = window;
    _pointShape.setRadius(CONF.getUIPointRadius());
}

void Drawer::draw(const Point &point) {
    _pointShape.setPosition(point);
    _window->draw(_pointShape);
}

void Drawer::draw(const Point &point1, const Point &point2) {
    sf::Vertex vertexes[2] = { sf::Vertex(point1), sf::Vertex(point2) };
    _window->draw(vertexes, 2, sf::Lines);
}

void Drawer::draw(const Triangle &tri) {
    sf::Vertex vertexes[4] = { sf::Vertex(tri.points[0]), sf::Vertex(tri.points[1]), sf::Vertex(tri.points[2]), sf::Vertex(tri.points[0]) };
    _window->draw(vertexes, 4, sf::LineStrip);
}

void Drawer::draw(const BlockInfo &blockInfo) {
        sf::Uint8 r = 0, g = 0, b = 0;
        r += sf::Uint8(blockInfo.id);
        for (auto &edgeInfo: blockInfo.edges) {
            g += edgeInfo->id;
        }
        for (auto &vertexInfo: blockInfo.vertexes) {
            b += vertexInfo.lock()->id;
        }
        auto color = sf::Color(r, g, b);
        draw(blockInfo, color);
}

void Drawer::draw(const BlockInfo &blockInfo, const sf::Color &color) {
    unsigned long vertexesNumber = 1 + blockInfo.edges.size() * 2;
    int crtVertexesIndex = 1;
    sf::Vertex vertexes[vertexesNumber];
    vertexes[0] = sf::Vertex(blockInfo.center, color);
    for (auto &edgeInfo: blockInfo.edges) {
        for (auto &vertexInfo: edgeInfo->vertexes) {
            vertexes[crtVertexesIndex++] = sf::Vertex(vertexInfo->pos, color);
        }
    }
    _window->draw(vertexes, vertexesNumber, sf::TriangleFan);
}

void Drawer::draw(const EdgeInfo &edgeInfo) {
    sf::Vertex vertexes[2] = { sf::Vertex((*edgeInfo.vertexes.begin())->pos), sf::Vertex((*edgeInfo.vertexes.rbegin())->pos) };
    _window->draw(vertexes, 2, sf::Lines);
}

void Drawer::commit() {

}

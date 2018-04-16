//
// Created by Nycshisan on 2018/4/16.
//

#include "drawer.h"
#include "../conf/conf.h"

Drawer::Drawer(Window *window) {
    _window = window;
    _pointShape.setRadius(CONF.getUIPointRadius());
    int width = CONF.getMapWidth(), height = CONF.getMapHeight();
    _box = Rectangle(0, width, height, 0);
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
    std::vector<std::pair<Point, Line>> edgePoints;
    for (auto &edgeInfo: blockInfo.edges) {
        for (auto &vertexInfo: edgeInfo->vertexes) {
            Point &p = vertexInfo->pos;
            Line l;
            if (_box.onEdge(p, l)) {
                edgePoints.emplace_back(std::pair<Point, Line>(p, l));
            };
            vertexes[crtVertexesIndex++] = sf::Vertex(p, color);
        }
    }
    _window->draw(vertexes, vertexesNumber, sf::TriangleFan);
    if (!edgePoints.empty()) {
        auto &pa = edgePoints[0].first, &pb = edgePoints[1].first;
        auto &la = edgePoints[0].second, &lb = edgePoints[1].second;
        if ((la.vertical && lb.vertical) || (la.horizontal && lb.horizontal)) {
            sf::Vertex edgeVertexes[3] = { sf::Vertex(blockInfo.center, color), sf::Vertex(pa, color), sf::Vertex(pb, color) };
            _window->draw(edgeVertexes, 3, sf::Triangles);
        } else {
            if (la.horizontal) {
                std::swap(la, lb);
            }
            Point corner(la.verticalX, lb.horizontalY);
            sf::Vertex edgeVertexes[5] = { sf::Vertex(blockInfo.center, color),
                                           sf::Vertex(pa, color), sf::Vertex(corner, color),
                                           sf::Vertex(pb, color), sf::Vertex(corner, color) };
            _window->draw(edgeVertexes, 5, sf::TriangleFan);
        }
    }
}

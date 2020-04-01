//
// Created by Nycshisan on 2018/4/16.
//

#include "drawer.h"

#include <cmath>

#include "window.h"
#include "../conf/conf.h"

namespace wg {

    Drawer::Drawer() {
        _pointShape.setRadius(_BasePointRadius * CONF.getUIScale());
        _pointsBuf.setPrimitiveType(sf::Points);
        _linesBuf.setPrimitiveType(sf::Lines);
        _trisBuf.setPrimitiveType(sf::Triangles);
    }

    void Drawer::setWindow(MainWindow *window) {
        this->_window = window;
    }

    void Drawer::commit() {
        for (const auto spritePtr : _sprites) {
            _window->draw(*spritePtr);
        }
        _window->draw(_trisBuf);
        _window->draw(_linesBuf);
        _window->draw(_pointsBuf);
        for (size_t i = 0; i < _pointShapeBuf.getVertexCount(); ++i) {
            const sf::Vertex &vertex = _pointShapeBuf[i];
            _pointShape.setPosition(vertex.position);
            _window->draw(_pointShape);
        }
        for (const auto &shape: _coloredPointShapes) {
            _window->draw(shape);
        }
    }

    void Drawer::clear() {
        _pointsBuf.clear();
        _linesBuf.clear();
        _trisBuf.clear();
        _sprites.clear();
        _pointShapeBuf.clear();
        _coloredPointShapes.clear();
    }

    void Drawer::appendVertex(sf::PrimitiveType type, const sf::Vertex &vertex) {
        switch (type) {
            case sf::Points:
                _pointsBuf.append(vertex);
                break;
            case sf::Lines:
                _linesBuf.append(vertex);
                break;
            case sf::Triangles:
                _trisBuf.append(vertex);
                break;
            default:
                LOG("Error primitive type in drawer.");
        }
    }

    void Drawer::appendPointShape(const sf::Vertex &vertex) {
        _pointShapeBuf.append(vertex);
    }

    void Drawer::drawThickLine(const std::shared_ptr<EdgeInfo> &edgeInfo, float thickness) {
        thickness *= CONF.getUIScale();
        Point pointA = (*edgeInfo->vertexes.begin())->point, pointB = (*edgeInfo->vertexes.rbegin())->point;
        float length = std::sqrt(pointA.squareDistance(pointB));
        float sin = (pointA.y - pointB.y) / length, cos = (pointA.x - pointB.x) / length;
        Point ltp(pointB.x - thickness * sin, pointB.y + thickness * cos),
                rtp(pointB.x + thickness * sin, pointB.y - thickness * cos),
                lbp(pointA.x - thickness * sin, pointA.y + thickness * cos),
                rbp(pointA.x + thickness * sin, pointA.y - thickness * cos);
        appendVertex(sf::Triangles, ltp.vertexUI);
        appendVertex(sf::Triangles, lbp.vertexUI);
        appendVertex(sf::Triangles, rtp.vertexUI);
        appendVertex(sf::Triangles, rtp.vertexUI);
        appendVertex(sf::Triangles, lbp.vertexUI);
        appendVertex(sf::Triangles, rbp.vertexUI);
    }

    void Drawer::addSprite(const sf::Sprite &s) {
        _sprites.emplace_back(&s);
    }

    void Drawer::appendCustomPointShape(const sf::Vertex &vertex, const sf::Color &color, float size) {
        auto shape = _pointShape;
        shape.setPosition(vertex.position);
        shape.setFillColor(color);
        if (size > 0) {
            shape.setRadius(size);
        }
        _coloredPointShapes.emplace_back(std::move(shape));
    }
}

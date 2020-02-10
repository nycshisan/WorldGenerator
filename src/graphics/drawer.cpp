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
        _window->draw(_trisBuf);
        _window->draw(_linesBuf);
        for (size_t i = 0; i < _pointsBuf.getVertexCount(); ++i) {
            const sf::Vertex &vertex = _pointsBuf[i];
            _pointShape.setPosition(vertex.position);
            _window->draw(_pointShape);
        }
    }

    void Drawer::clearVertexes() {
        _pointsBuf.clear();
        _linesBuf.clear();
        _trisBuf.clear();
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

    void Drawer::drawThickLine(const std::shared_ptr<EdgeInfo> &edgeInfo, float thickness) {
        Point pointA = (*edgeInfo->vertexes.begin())->point, pointB = (*edgeInfo->vertexes.rbegin())->point;
        float length = std::sqrt(pointA.squareDistance(pointB));
        float sin = (pointA.y - pointB.y) / length, cos = (pointA.x - pointB.x) / length;
        Point ltp(pointB.x - thickness * sin, pointB.y + thickness * cos),
                rtp(pointB.x + thickness * sin, pointB.y - thickness * cos),
                lbp(pointA.x - thickness * sin, pointA.y + thickness * cos),
                rbp(pointA.x + thickness * sin, pointA.y - thickness * cos);
        appendVertex(sf::Triangles, ltp.vertex);
        appendVertex(sf::Triangles, lbp.vertex);
        appendVertex(sf::Triangles, rtp.vertex);
        appendVertex(sf::Triangles, rtp.vertex);
        appendVertex(sf::Triangles, lbp.vertex);
        appendVertex(sf::Triangles, rbp.vertex);
    }

}

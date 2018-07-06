//
// Created by Nycshisan on 2018/4/16.
//

#include "window.h"
#include "drawer.h"
#include "../conf/conf.h"

namespace wg {

    Drawer::Drawer() {
        _pointShape.setRadius(int(_BasePointRadius * CONF.getUIScale()));
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
                LOGOUT("Error primitive type in drawer.");
        }
    }

}

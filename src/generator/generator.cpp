//
// Created by Nycshisan on 2018/3/8.
//

#include "generator.h"

#include "../misc/log.h"

void NextButtonResponder(Window &window) {
    Generator &generator = Generator::SharedInstance();
    switch (generator._state) {
        case Ready:
            window.setHintLabel("Generated block centers.");
            generator.vd.init(window.getMapSize().x, window.getMapSize().y);
            generator.vd.generateCenters();
            break;
        case BlockCenters:
            window.setHintLabel("Generated Delaunay triangles.");
            generator.vd.generateDelaunayTriangles();
            break;
        case DelaunayTriangles:
            window.setHintLabel("Generated Voronoi diagram.");
            break;
        default:
            LOGERR("Invalid generator state!");
    }
    generator._nextState();
}

void RedoButtonResponder(Window &window) {
    Generator &generator = Generator::SharedInstance();
    switch (generator._state) {
        case Ready: break;
        case BlockCenters:
            generator.vd.generateCenters(); break;
        case DelaunayTriangles:
            generator.vd.generateDelaunayTriangles(); break;
        default:
            LOGERR("Invalid generator state!");
    }
}

void SaveButtonResponder(Window &window) {}

Generator &Generator::SharedInstance() {
    static Generator instance;
    return instance;
}

void Generator::_nextState() {
    switch (_state) {
        case Ready:
            _state = BlockCenters; break;
        case BlockCenters:
            _state = DelaunayTriangles; break;
        case DelaunayTriangles:
            _state = VoronoiDiagram; break;
        default:
            break;
    }
}

void Generator::display(Window &window) {
    switch (_state) {
        case BlockCenters:
            vd.drawPointsToWindow(window); break;
        case DelaunayTriangles:
            vd.drawPointsToWindow(window);
            vd.drawTrianglesToWindow(window); break;
        default:
            break;
    }
}

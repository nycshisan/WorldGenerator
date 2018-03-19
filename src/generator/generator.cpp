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
            generator._blockCenters.init(window.getMapSize().x, window.getMapSize().y);
            generator._blockCenters.generate();
            generator._centers = generator._blockCenters.output();
            break;
        case BlockCenters:
            window.setHintLabel("Generated Delaunay triangles.");
            generator._delaunayTriangles.init(generator._centers);
            generator._delaunayTriangles.generate();
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
            generator._blockCenters.generate(); break;
        case DelaunayTriangles:
            generator._delaunayTriangles.generate(); break;
        default:
            LOGERR("Invalid generator state!");
    }
}
void UndoButtonResponder(Window &window) {

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
            _blockCenters.draw(window); break;
        case DelaunayTriangles:
            _delaunayTriangles.draw(window); break;
        default:
            break;
    }
}

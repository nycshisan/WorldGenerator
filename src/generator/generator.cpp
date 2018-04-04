//
// Created by Nycshisan on 2018/3/8.
//

#include "generator.h"

#include "../misc/log.h"

void Generator::NextButtonResponder(Window &window) {
    Generator &generator = Generator::SharedInstance();
    switch (generator._state) {
        case Ready:
            window.setHintLabel("Generated block centers.");
            generator._blockCenters.input(window.getMapSize().x, window.getMapSize().y);
            generator._blockCenters.generate();
            break;
        case BlockCenters:
            window.setHintLabel("Generated Delaunay triangles.");
            generator._centers = generator._blockCenters.output();
            generator._delaunayTriangles.input(generator._centers);
            generator._delaunayTriangles.generate();
            break;
        case DelaunayTriangles:
            window.setHintLabel("Generated Voronoi diagram.");
            generator._tris = generator._delaunayTriangles.output();
            generator._voronoiDiagram.input(generator._centers, generator._tris);
            generator._voronoiDiagram.generate();
            break;
        default:
            LOGERR("Invalid generator state!");
    }
    generator._nextState();
}

void Generator::RedoButtonResponder(Window &window) {
    Generator &generator = Generator::SharedInstance();
    switch (generator._state) {
        case Ready: break;
        case BlockCenters:
            generator._blockCenters.generate(); break;
        case DelaunayTriangles:
            generator._delaunayTriangles.generate(); break;
        case VoronoiDiagram:
            generator._voronoiDiagram.generate(); break;
//        case LloydRelaxation:
//            generator._lloydRelaxation.generate(); break;
        default:
            LOGERR("Invalid generator state!");
    }
}
void Generator::UndoButtonResponder(Window &window) {
    Generator &generator = Generator::SharedInstance();
    generator._lastState();
}

void Generator::SaveButtonResponder(Window &window) {
    Generator &generator = Generator::SharedInstance();
    switch (generator._state) {
        case BlockCenters:
            generator._blockCenters.save();
            window.setHintLabel("Centers saved.");
            break;
        default:
            window.setHintLabel("Can't save.");
            break;
    }
}

void Generator::LoadButtonResponder(Window &window) {
    Generator &generator = Generator::SharedInstance();
    switch (generator._state) {
        case BlockCenters:
            generator._blockCenters.load();
            window.setHintLabel("Centers loaded.");
            break;
        default:
            window.setHintLabel("Can't load.");
            break;
    }
}


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
        case VoronoiDiagram:
            _state = LloydRelaxation; break;
        default:
            break;
    }
}

void Generator::_lastState() {
    switch (_state) {
        case Ready:
            break;
        case BlockCenters:
            _state = Ready; break;
        case DelaunayTriangles:
            _state = BlockCenters; break;
        case VoronoiDiagram:
            _state = DelaunayTriangles; break;
        case LloydRelaxation:
            _state = LloydRelaxation; break;
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
        case VoronoiDiagram:
            _voronoiDiagram.draw(window); break;
//        case LloydRelaxation:
//            _lloydRelaxation.draw(window); break;
        default:
            break;
    }
}

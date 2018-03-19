//
// Created by Nycshisan on 2018/3/8.
//

#include "generator.h"

#include "../misc/log.h"

#include <fstream>
void saveCentersForDebug(std::vector<Point> points) {
    std::ofstream outfile("logs/centers.txt");
    for (auto &point: points) {
        outfile << (int)point.x << " " << (int)point.y << std::endl;
    }
}

std::vector<Point> loadCentersForDebug() {
    std::ifstream infile("logs/centers.txt");
    std::vector<Point> centers;
    while (!infile.eof()) {
        int x, y;
        infile >> x >> y;
        centers.emplace_back(Point(x, y));
    }
    centers.pop_back();
    return centers;
}

void Generator::NextButtonResponder(Window &window) {
    Generator &generator = Generator::SharedInstance();
    switch (generator._state) {
        case Ready:
            window.setHintLabel("Generated block centers.");
            generator._blockCenters.init(window.getMapSize().x, window.getMapSize().y);
            generator._blockCenters.generate();
            break;
        case BlockCenters:
            window.setHintLabel("Generated Delaunay triangles.");
            generator._centers = generator._blockCenters.output();
            generator._delaunayTriangles.init(generator._centers);
            generator._delaunayTriangles.generate();
            break;
        case DelaunayTriangles:
            window.setHintLabel("Generated Voronoi diagram.");
            generator._tris = generator._delaunayTriangles.output();
            generator._voronoiDiagram.init(generator._centers, generator._tris);
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
        default:
            LOGERR("Invalid generator state!");
    }
}
void Generator::UndoButtonResponder(Window &window) {
    Generator &generator = Generator::SharedInstance();
    generator._lastState();
}

void Generator::SaveButtonResponder(Window &window) {}


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
        default:
            break;
    }
}

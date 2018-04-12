//
// Created by Nycshisan on 2018/4/4.
//

#include "lloyd.h"

#include "../conf/conf.h"

void LloydRelaxation::input(LloydRelaxation::Input vd) {
    _inputVd = std::move(vd);
    _factor = CONF.getLloydFactor();
    _iteration = CONF.getLloydIteration();
    _pointShape.setRadius(CONF.getUIPointRadius());
    _box = Rectangle(0, CONF.getMapWidth(), CONF.getMapHeight(), 0);
}

void LloydRelaxation::generate() {
    _relaxedVd = _inputVd;

    auto &vertexMap = _relaxedVd.first;
    auto &edgeMap = _relaxedVd.second;

    for (int i = 0; i < _iteration; ++i) {
        DelaunayTriangles::Input centers;

        for (auto &pair: vertexMap) {
            auto &vertex = pair.second;
            Point pos = vertex.pos;
            Point centerVec(0, 0);
            for (auto edgeId: vertex.edgeIds) {
                auto &edge = edgeMap[edgeId];
                for (auto &v : edge.vertex) {
                    centerVec += v.position;
                }
            }

            centerVec /= float(vertex.edgeIds.size() * 2);
            pos *= (1 - _factor);
            pos += centerVec * _factor;
            assertWithSave(_box.contains(pos));
            centers.emplace_back(pos);
        }

        DelaunayTriangles delaunayTriangles;
        delaunayTriangles.input(centers);
        delaunayTriangles.generate();
        auto tris = delaunayTriangles.output();

        VoronoiDiagram voronoiDiagram;
        voronoiDiagram.input(centers, tris);
        voronoiDiagram.generate();
        _relaxedVd = voronoiDiagram.output();
    }
}

LloydRelaxation::Output LloydRelaxation::output() {
    return LloydRelaxation::Output();
}

void LloydRelaxation::draw(Window &window) {
    for (auto &vertex: _relaxedVd.first) {
        _pointShape.setPosition(vertex.second.pos);
        window.draw(_pointShape);
    }

    for (auto &edge: _relaxedVd.second) {
        window.draw(edge.second.vertex, 2, sf::Lines);
    }
}

//
// Created by Nycshisan on 2018/4/4.
//

#include "lloyd.h"

#include "../conf/conf.h"

void LloydRelaxation::input(LloydRelaxation::Input vd) {
    _inputVd = std::move(vd);
    _factor = CONF.getLloydFactor();
    _iteration = CONF.getLloydIteration();
    _box = Rectangle(0, CONF.getMapWidth(), CONF.getMapHeight(), 0);
}

void LloydRelaxation::generate() {
    _relaxedVd = _inputVd;

    auto &centerMap = _relaxedVd.first;
    auto &edgeMap = _relaxedVd.second;

    for (int i = 0; i < _iteration; ++i) {
        DelaunayTriangles::Input centers;

        for (auto &pair: centerMap) {
            auto &center = pair.second;
            Point pos = center.pos;
            Point centerVec(0, 0);
            for (auto edgeId: center.edgeIds) {
                auto &edge = edgeMap[edgeId];
                for (auto &v : edge.vertex) {
                    centerVec += v;
                }
            }

            centerVec /= float(center.edgeIds.size() * 2);
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
    return _relaxedVd;
}

void LloydRelaxation::draw(Drawer &drawer) {
    for (auto &pair: _relaxedVd.first) {
        drawer.draw(pair.second.pos);
    }

    for (auto &pair: _relaxedVd.second) {
        drawer.draw(pair.second.vertex[0], pair.second.vertex[1]);
    }
}

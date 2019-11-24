//
// Created by Nycshisan on 2018/4/4.
//

#include "lloyd.h"

#include "../conf/conf.h"
#include "../graphics/drawer.h"

namespace wg {

    void LloydRelaxation::input(const Input &vd) {
        _inputVd = vd;
    }

    void LloydRelaxation::generate() {
        float factor = CONF.getLloydFactor();
        int iteration = CONF.getLloydIteration();
        Rectangle box = Rectangle(0, CONF.getMapWidth(), CONF.getMapHeight(), 0);

        _relaxedVd = _inputVd;

        auto &centerMap = _relaxedVd.first;
        auto &edgeMap = _relaxedVd.second;

        for (int i = 0; i < iteration; ++i) {
            DelaunayTriangles::Input centers;

            for (auto &pair: centerMap) {
                auto &center = pair.second;
                Point pos = center.point;
                Point centerVec(0, 0);
                for (auto edgeId: center.edgeIds) {
                    auto &edge = edgeMap[edgeId];
                    for (auto &v : edge.point) {
                        centerVec += v;
                    }
                }

                centerVec /= float(center.edgeIds.size() * 2);
                pos *= (1 - factor);
                pos += centerVec * factor;
                assertWithSave(box.contains(pos));
                centers.emplace_back(pos);
            }

            class DelaunayTriangles delaunayTriangles;
            delaunayTriangles.input(centers);
            delaunayTriangles.generate();
            auto tris = delaunayTriangles.output();

            class VoronoiDiagram voronoiDiagram;
            voronoiDiagram.input(centers, tris);
            voronoiDiagram.generate();
            _relaxedVd = voronoiDiagram.output();
        }
    }

    LloydRelaxation::Output LloydRelaxation::output() {
        return _relaxedVd;
    }

    void LloydRelaxation::prepareVertexes(Drawer &drawer) {
        for (auto &pair: _relaxedVd.first) {
            drawer.appendVertex(sf::Points, pair.second.point.vertex);
        }

        for (auto &pair: _relaxedVd.second) {
            drawer.appendVertex(sf::Lines, pair.second.point[0].vertex);
            drawer.appendVertex(sf::Lines, pair.second.point[1].vertex);
        }
    }

}
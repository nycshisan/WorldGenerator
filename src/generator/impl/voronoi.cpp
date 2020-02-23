//
// Created by Nycshisan on 2018/3/15.
//

#include "voronoi.h"

#include "delaunay.h"

namespace wg {

    bool VoronoiDiagram::_existsEdge(int paId, int pbId) {
        bool exist = _existsEdges.count(pbId) > 0 && _existsEdges[pbId].count(paId) > 0;
        _existsEdges[paId][pbId] = true;
        return exist;
    }

    void VoronoiDiagram::generate() {
        auto centersTris = *(DelaunayTriangles::Output*)_inputData;

        int width = CONF.getMapWidth(), height = CONF.getMapHeight();
        Rectangle box = Rectangle(0, float(width), float(height), 0);

        auto &centers = centersTris.first;
        auto &tris = centersTris.second;

        auto &centerMap = _diagram.first;
        auto &edgeMap = _diagram.second;

        int newEdgeId = 0;
        centerMap.clear();
        edgeMap.clear();
        _existsEdges.clear();

        for (int i = 0; i < centers.size(); ++i) {
            centerMap[i] = CenterNode(centers[i]);
        }

        int newVertexId = 0;
        for (auto &tri: tris) {
            if (tri->id > newVertexId)
                newVertexId = tri->id;
        }
        ++newVertexId;

        for (auto &tri: tris) {
            for (auto &edge: tri->edges) {
                if (edge.nextTri == nullptr) {
                    continue;
                }
                Point pa = tri->exCenter, pb = edge.nextTri->exCenter;
                int paId = tri->id, pbId = edge.nextTri->id;
                bool paInBox = box.contains(pa), pbInBox = box.contains(pb);
                if (!paInBox && !pbInBox) {
                    Point intersections[2];
                    int intersectionNumber = box.intersectSegment(pa, pb, intersections);
                    if (intersectionNumber == 0) {
                        continue;
                    }
                    pa = intersections[0];
                    pb = intersections[1];
                    paId = newVertexId++;
                    pbId = newVertexId++;
                } else if (!paInBox || !pbInBox) {
                    if (pbInBox) {
                        std::swap(pa, pb);
                        std::swap(paId, pbId);
                    }
                    pb = box.intersectRay(pa, pb);
                    pbId = newVertexId++;
                }

                if (_existsEdge(edge.pid[0], edge.pid[1])) {
                    continue;
                }

                edgeMap[newEdgeId] = EdgeNode(pa, paId, pb, pbId);
                for (int j = 0; j < 2; ++j) {
                    int pointId = edge.pid[j];
                    edgeMap[newEdgeId].relatedCenterIds[j] = pointId;
                    centerMap[pointId].edgeIds.emplace_back(newEdgeId);
                }

                newEdgeId++;
            }
        }

        for (auto &pair: centerMap) {
            assertWithSave(!pair.second.edgeIds.empty());
        }

        _outputData = (void*)&_diagram;
    }

    void VoronoiDiagram::prepareVertexes(Drawer &drawer) {
        for (auto &pair: _diagram.first) {
            drawer.appendPointShape(pair.second.point.vertex);
        }

        for (auto &pair: _diagram.second) {
            drawer.appendVertex(sf::Lines, pair.second.point[0].vertex);
            drawer.appendVertex(sf::Lines, pair.second.point[1].vertex);
        }
    }

    std::string VoronoiDiagram::getHintLabelText() {
        return "Generated Voronoi diagram.";
    }

    VoronoiDiagram::CenterNode::CenterNode(const Point &p) {
        point = p;
    }

    VoronoiDiagram::EdgeNode::EdgeNode(const Point &pa, int paId, const Point &pb, int pbId) {
        point[0] = pa;
        point[1] = pb;
        vertexIds[0] = paId;
        vertexIds[1] = pbId;
    }

}
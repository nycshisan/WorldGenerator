//
// Created by Nycshisan on 2018/3/15.
//

#include "voronoi.h"

#include "../conf/conf.h"

bool VoronoiDiagram::_existsEdge(int paId, int pbId) {
    bool exist = _existsEdges.count(pbId) > 0 && _existsEdges[pbId].count(paId) > 0;
    _existsEdges[paId][pbId] = true;
    return exist;
}

void VoronoiDiagram::input(VoronoiDiagram::InputCenters centers, VoronoiDiagram::InputTris tris) {
    _centers = std::move(centers);
    _tris = std::move(tris);
}

void VoronoiDiagram::generate() {
    int width = CONF.getMapWidth(), height = CONF.getMapHeight();
    Rectangle box = Rectangle(0, width, height, 0);

    std::map<int, CenterNode> &centerMap = _diagram.first;
    std::map<int, EdgeNode> &edgeMap = _diagram.second;

    int newEdgeId = 0;
    centerMap.clear();
    edgeMap.clear();
    _existsEdges.clear();

    for (int i = 0; i < _centers.size(); ++i) {
        centerMap[i] = CenterNode(_centers[i]);
    }

    int newVertexId = 0;
    for (auto &tri: _tris) {
        if (tri->id > newVertexId)
            newVertexId = tri->id;
    }
    ++newVertexId;

    for (auto &tri: _tris) {
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
                pa = intersections[0]; pb = intersections[1];
                paId = newVertexId++; pbId = newVertexId++;
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
}

VoronoiDiagram::Output VoronoiDiagram::output() {
    return _diagram;
}

void VoronoiDiagram::draw(Drawer &drawer) {
    for (auto &pair: _diagram.first) {
        drawer.draw(pair.second.pos);
    }

    for (auto &pair: _diagram.second) {
        drawer.draw(pair.second.vertex[0], pair.second.vertex[1]);
    }
}

VoronoiDiagram::CenterNode::CenterNode(const Point &p) {
    pos = p;
}

VoronoiDiagram::EdgeNode::EdgeNode(const Point &pa, int paId, const Point &pb, int pbId) {
    vertex[0] = pa; vertex[1] = pb;
    vertexIds[0] = paId; vertexIds[1] = pbId;
}
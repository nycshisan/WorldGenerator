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
    _pointShape.setRadius(CONF.getUIPointRadius());
    _width = CONF.getMapWidth(), _height = CONF.getMapHeight();
    _box = Rectangle(0, _width, _height, 0);
}

void VoronoiDiagram::generate() {
    std::map<int, VertexNode> &vertexMap = _diagram.first;
    std::map<int, EdgeNode> &edgeMap = _diagram.second;

    _newEdgeId = 0;
    vertexMap.clear();
    edgeMap.clear();
    _existsEdges.clear();

    for (int i = 0; i < _centers.size(); ++i) {
        vertexMap[i] = VertexNode(_centers[i]);
    }

    for (auto &tri: _tris) {
        for (auto &edge : tri->edges) {
            if (edge.nextTri == nullptr) {
                continue;
            }
            Point pa = tri->exCenter, pb = edge.nextTri->exCenter;
            bool paInBox = _box.contains(pa), pbInBox = _box.contains(pb);
            if (!paInBox && !pbInBox) {
                Point intersections[2];
                int intersectionNumber = _box.intersectSegment(pa, pb, intersections);
                if (intersectionNumber == 0) {
                    continue;
                }
                pa = intersections[0]; pb = intersections[1];
            } else if (!paInBox || !pbInBox) {
                if (pbInBox) {
                    std::swap(pa, pb);
                }
                pb = _box.intersectRay(pa, pb);
            }

            if (_existsEdge(edge.pid[0], edge.pid[1])) {
                continue;
            }

            edgeMap[_newEdgeId] = EdgeNode(pa, pb);
            for (int j = 0; j < 2; ++j) {
                int pointId = edge.pid[j];
                edgeMap[_newEdgeId].relatedCenterIds[j] = pointId;
                vertexMap[pointId].edgeIds.emplace_back(_newEdgeId);
            }

            _newEdgeId++;
        }
    }

    for (auto &pair: vertexMap) {
        assertWithSave(!pair.second.edgeIds.empty());
    }
}

VoronoiDiagram::Output VoronoiDiagram::output() {
    return _diagram;
}

void VoronoiDiagram::draw(Window &window) {
    for (auto &vertex: _diagram.first) {
        _pointShape.setPosition(vertex.second.pos);
        window.draw(_pointShape);
    }

    for (auto &edge: _diagram.second) {
        window.draw(edge.second.vertex, 2, sf::Lines);
    }
}

VoronoiDiagram::VertexNode::VertexNode(const Point &p) {
    pos = p;
}

VoronoiDiagram::EdgeNode::EdgeNode(const Point &pa, const Point &pb) {
    vertex[0] = sf::Vertex(pa); vertex[1] = sf::Vertex(pb);
}
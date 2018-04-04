//
// Created by Nycshisan on 2018/3/15.
//

#include "voronoi.h"

#include "../conf/conf.h"

void VoronoiDiagram::input(VoronoiDiagram::InputCenters centers, VoronoiDiagram::InputTris tris) {
    _centers = std::move(centers);
    _tris = std::move(tris);
    _pointShape.setRadius(CONF.getUIPointRadius());
    _width = CONF.getMapWidth(), _height = CONF.getMapHeight();
    Point boxVertexes[4] = { Point(0, 0), Point(0, _width), Point(_height, _width), Point(_height, 0) };
    _boxEdges[0] = Line(boxVertexes[0], boxVertexes[1]);
    _boxEdges[1] = Line(boxVertexes[1], boxVertexes[2]);
    _boxEdges[2] = Line(boxVertexes[2], boxVertexes[3]);
    _boxEdges[3] = Line(boxVertexes[3], boxVertexes[0]);
}

void VoronoiDiagram::generate() {
    _newEdgeId = 0;
    _diagram.first.clear();
    _diagram.second.clear();

    for (auto tri: _tris) {
        for (auto &edge : tri->edges) {
            if (edge.nextTri == nullptr) {
                continue;
            }
            Point pa = tri->exCenter, pb = edge.nextTri->exCenter;
            if (edge.nextTri->isBoundTriangle()) {
                Line lab(pa, pb);
                sf::Vector2f vab = pa - pb;
                bool debugFlagAssertPBFound = false;
                for (auto boxEdge: _boxEdges) {
                    Point intersection = lab.intersect(boxEdge);
                    if (intersection.x >= 0 && intersection.x <= _width && intersection.y >= 0 && intersection.y <= _height) {
                        sf::Vector2f vai = pa - intersection;
                        float dot = vab.x * vai.x + vab.y * vai.y;
                        if (dot >= 0) {
                            pb = intersection;
                            debugFlagAssertPBFound = true;
                            break;
                        }
                    }
                }
                assert(debugFlagAssertPBFound);
            }

            std::map<int, VertexNode> &vertexMap = _diagram.first;
            std::map<int, EdgeNode> &edgeMap = _diagram.second;
            edgeMap[_newEdgeId] = EdgeNode(pa, pb);
            for (int j = 0; j < 2; ++j) {
                int pointId = edge.pid[j];
                if (vertexMap.find(pointId) == vertexMap.end()) {
                    vertexMap[pointId] = VertexNode(_centers[pointId]);
                }
                edgeMap[_newEdgeId].relatedCenterIds[j] = pointId;
                vertexMap[pointId].edgeIds.emplace_back(_newEdgeId);
            }

            _newEdgeId++;
        }
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
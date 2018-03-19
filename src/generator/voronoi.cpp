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
    _boxEdge[0] = Line(boxVertexes[0], boxVertexes[1]);
    _boxEdge[1] = Line(boxVertexes[1], boxVertexes[2]);
    _boxEdge[2] = Line(boxVertexes[2], boxVertexes[3]);
    _boxEdge[3] = Line(boxVertexes[3], boxVertexes[0]);
}

void VoronoiDiagram::generate() {
    _newEdgeId = 0;
    _diagram.first.clear();
    _diagram.second.clear();

    for (auto tri: _tris) {
        bool triContainExCenter = true;
        bool isLongestEdge[3] = { false, false, false };
//        if (tri->edges[0].nextTri == nullptr || tri->edges[1].nextTri == nullptr || tri->edges[2].nextTri == nullptr) {
//            triContainExCenter = triangleContains(tri->points[0], tri->points[1], tri->points[2], tri->exCenter);
//            int longestEdgeIndex = 0;
//            float longestEdgeLength = 0;
//            for (int i = 0; i < 3; ++i) {
//                auto &edge = tri->edges[i];
//                Point epa = _centers[edge.pid[0]], epb = _centers[edge.pid[1]];
//                float dx = epa.x - epb.x, dy = epa.y - epb.y;
//                float edgeLength = dx * dx + dy * dy;
//                if (edgeLength > longestEdgeLength) {
//                    longestEdgeIndex = i;
//                    longestEdgeLength = edgeLength;
//                }
//            }
//            isLongestEdge[longestEdgeIndex] = true;
//        }

        for (int i = 0; i < 3; ++i) {
            auto &edge = tri->edges[i];
            Point pa = tri->exCenter, pb;
            if (edge.nextTri != nullptr) {
                pb = edge.nextTri->exCenter;
            } else {
                continue;
                bool outTriExCenterAndLongestEdgeFlag = triContainExCenter * !isLongestEdge[i];
                Point epa = _centers[edge.pid[0]], epb = _centers[edge.pid[1]];
                float midX = (epa.x + epb.x) / 2.0f, midY = (epa.y + epb.y) / 2.0f;
                Segment seg(epa, epb);
                Line midPerpendicular = seg.midPerpendicular();
                for (const auto &boxEdge : _boxEdge) {
                    Point intersectPoint = midPerpendicular.intersect(boxEdge);
                    if (intersectPoint.x < 0 || intersectPoint.y < 0 || intersectPoint.x > _width || intersectPoint.y > _height)
                        continue;
                    if ((intersectPoint.x - midX) * (midX - tri->exCenter.x) * outTriExCenterAndLongestEdgeFlag < 0 ||
                        (intersectPoint.y - midY) * (midY - tri->exCenter.y) * outTriExCenterAndLongestEdgeFlag < 0)
                        continue;
                    pb = intersectPoint;
                }
            }

            std::map<int, VertexNode> &vertexMap = _diagram.first;
            std::map<int, EdgeNode> &edgeMap = _diagram.second;
            edgeMap[_newEdgeId] = EdgeNode(pa, pb);
            for (int i = 0; i < 2; ++i) {
                int pointId = edge.pid[i];
                if (vertexMap.find(pointId) == vertexMap.end()) {
                    vertexMap[pointId] = VertexNode(_centers[pointId]);
                }
                edgeMap[_newEdgeId].relatedCenterIds[i] = pointId;
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
    endPoints[0] = pa; endPoints[1] = pb;
    vertex[0] = sf::Vertex(pa); vertex[1] = sf::Vertex(pb);
}
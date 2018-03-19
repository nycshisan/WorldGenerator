//
// Created by Nycshisan on 2018/3/19.
//

#include "delaunay.h"

#include "../conf/conf.h"

void DelaunayTriangles::init(const DelaunayTriangles::Input &input) {
    _centers = input;
}

void DelaunayTriangles::generate() {
    _deleteOldNodes();
    // Bowyer-Watson algorithm
    auto n = (int)_centers.size();
    _centers.emplace_back(0, 0);
    _centers.emplace_back(2 * CONF.getMapWidth(), 0);
    _centers.emplace_back(0, 2 * CONF.getMapHeight());
    triNetHead = new NetNode(n, n + 1, n + 2, _centers, n);
    _allocatedNodes.emplace_back(triNetHead);

    for (int i = 0; i < n; ++i) {
        auto center = _centers[i];
        auto containingTriangle = triNetHead->findContainingTriangle(center);
        std::set<NetNode*> influencedTris;
        std::vector<Edge*> influencedEdges;
        containingTriangle->findInfluenced(center, influencedTris, influencedEdges);
        std::vector<NetNode*> newTris;
        for (auto &edge: influencedEdges) {
            auto *newTri = new NetNode(edge->pid[0], edge->pid[1], i, _centers, n);
            _allocatedNodes.emplace_back(newTri);
            newTris.emplace_back(newTri);
            auto &newEdge = newTri->edges[0];
            newEdge.nextTri = edge->nextTri;
            newEdge.nextTriEdgeId = edge->nextTriEdgeId;
            edge->nextTri = nullptr; edge->nextTriEdgeId = -1;
            if (newEdge.nextTri != nullptr) {
                newEdge.nextTri->edges[newEdge.nextTriEdgeId].nextTri = newTri;
                newEdge.nextTri->edges[newEdge.nextTriEdgeId].nextTriEdgeId = 0;
            }
        }

        for (int j = 0; j < newTris.size(); ++j) {
            NetNode *tri = newTris[j], *lastTri, *nextTri;
            if (j == 0)
                lastTri = newTris[newTris.size() - 1];
            else
                lastTri = newTris[j - 1];
            if (j == newTris.size() - 1)
                nextTri = newTris[0];
            else
                nextTri = newTris[j + 1];

            tri->linkAnoTri(lastTri, 2, 1);
            tri->linkAnoTri(nextTri, 1, 2);
        }

        for (auto tri: influencedTris) {
            if (triNetHead == tri) {
                triNetHead = newTris[0];
                break;
            }
        }
    }

    triNetHead = triNetHead->removeBoundingTriangle();

    _centers.pop_back(); _centers.pop_back(); _centers.pop_back();
}

DelaunayTriangles::Output DelaunayTriangles::output() {
    return triNetHead;
}

void DelaunayTriangles::draw(Window &window) {
    triNetHead->draw(window);
}

void DelaunayTriangles::_deleteOldNodes() {
    if (!_allocatedNodes.empty()) {
        for (auto ptr: _allocatedNodes)
            delete ptr;
        _allocatedNodes.clear();
    }
    triNetHead = nullptr;
}

DelaunayTriangles::~DelaunayTriangles() {
    _deleteOldNodes();
}

DelaunayTriangles::Edge::Edge(int pointIdA, int pointIdB) {
    pid[0] = pointIdA;
    pid[1] = pointIdB;
}

void DelaunayTriangles::NetNode::_clearVisitFlag() {
    if (!_visited)
        return;
    _visited = false;
    for (auto &edge: edges) {
        if (edge.nextTri != nullptr)
            edge.nextTri->_clearVisitFlag();
    }
}

DelaunayTriangles::NetNode *DelaunayTriangles::NetNode::_findContainingTriangle(const Point &point) {
    if (_visited)
        return nullptr;
    _visited = true;
    if (triangleContains(points[0], points[1], points[2], point))
        return this;
    for (auto &edge: edges) {
        if (edge.nextTri == nullptr)
            continue;
        auto r = edge.nextTri->_findContainingTriangle(point);
        if (r != nullptr)
            return r;
    }
    return nullptr;
}

void DelaunayTriangles::NetNode::_findInfluenced(const Point &point, int beginEdgeId, std::set<NetNode *> &tris, std::vector<Edge *> &edges) {
    _visited = true;
    for (int i = 0; i < 3; ++i) {
        auto &edge = this->edges[(i + beginEdgeId) % 3];
        auto nextTri = edge.nextTri;
        if (nextTri == nullptr) {
            tris.insert(this);
            edges.emplace_back(&edge);
        } else if (!nextTri->_visited) {
            if (pointDistance(point, nextTri->exCenter) > nextTri->exRadius) {
                tris.insert(this);
                edges.emplace_back(&edge);
            } else {
                nextTri->_findInfluenced(point, edge.nextTriEdgeId, tris, edges);
            }
        }
    }
}

DelaunayTriangles::NetNode *DelaunayTriangles::NetNode::_removeBoundingTriangle() {
    if (_visited)
        return nullptr;
    _visited = true;
    NetNode *newHead = _isBoundTriangle ? nullptr : this;
    for (auto &edge: edges) {
        auto nextTri = edge.nextTri;
        if (nextTri != nullptr) {
            if (_isBoundTriangle) {
                nextTri->edges[edge.nextTriEdgeId].nextTri = nullptr;
                nextTri->edges[edge.nextTriEdgeId].nextTriEdgeId = -1;
                edge.nextTri = nullptr;
                edge.nextTriEdgeId = -1;
            }
            auto r = nextTri->_removeBoundingTriangle();
            newHead = r ? r : newHead;
        }
    }
    return newHead;
}

void DelaunayTriangles::NetNode::_draw(Window &window) {
    if (_visited)
        return;
    _visited = true;
    for (auto &edge: edges) {
        window.draw(_vertices, 2, sf::Lines);
        window.draw(_vertices + 1, 2, sf::Lines);
        window.draw(_vertices + 2, 2, sf::Lines);

        if (edge.nextTri != nullptr) {
            edge.nextTri->_draw(window);
        }
    }
}

DelaunayTriangles::NetNode::NetNode(int pointIdA, int pointIdB, int pointIdC, const std::vector<Point> &centers, int n) {
    points[0] = centers[pointIdA];
    points[1] = centers[pointIdB];
    points[2] = centers[pointIdC];
    _vertices[0] = sf::Vertex(points[0]);
    _vertices[1] = sf::Vertex(points[1]);
    _vertices[2] = sf::Vertex(points[2]);
    _vertices[3] = sf::Vertex(points[0]);

    edges[0] = Edge(pointIdA, pointIdB);
    edges[1] = Edge(pointIdB, pointIdC);
    edges[2] = Edge(pointIdC, pointIdA);
    exCenter = triangleExCenter(points[0], points[1], points[2]);
    exRadius = pointDistance(exCenter, points[0]);

    _isBoundTriangle = pointIdA >= n || pointIdB >= n || pointIdC >= n;
}

DelaunayTriangles::NetNode *DelaunayTriangles::NetNode::findContainingTriangle(const Point &point) {
    auto result = _findContainingTriangle(point);
    _clearVisitFlag();
    return result;
}

void DelaunayTriangles::NetNode::findInfluenced(const Point &point, std::set<NetNode *> &tris, std::vector<Edge *> &edges) {
    _findInfluenced(point, 0, tris, edges);
    _clearVisitFlag();
}

void DelaunayTriangles::NetNode::linkAnoTri(DelaunayTriangles::NetNode *anoTri, int edgeId, int anoEdgeId) {
    edges[edgeId].nextTri = anoTri;
    edges[edgeId].nextTriEdgeId = anoEdgeId;
    anoTri->edges[anoEdgeId].nextTri = this;
    anoTri->edges[anoEdgeId].nextTriEdgeId = edgeId;
}

DelaunayTriangles::NetNode *DelaunayTriangles::NetNode::removeBoundingTriangle() {
    auto r = _removeBoundingTriangle();
    _clearVisitFlag();
    return r;
}

void DelaunayTriangles::NetNode::draw(Window &window) {
    _draw(window);
    _clearVisitFlag();
}

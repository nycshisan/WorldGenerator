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
        std::vector<NetNode::Edge*> influencedEdges;
        containingTriangle->findAdjcInfluencedEdges(center, influencedTris, influencedEdges);
        std::vector<NetNode*> newTris;
        for (auto &edge: influencedEdges) {
            auto *newTri = new NetNode(edge->endpointId[0], edge->endpointId[1], i, _centers, n);
            _allocatedNodes.emplace_back(newTri);
            newTris.emplace_back(newTri);
            auto &newEdge = newTri->edges[0];
            newEdge.adjcTri = edge->adjcTri;
            newEdge.adjcTriEdgeId = edge->adjcTriEdgeId;
            edge->adjcTri = nullptr; edge->adjcTriEdgeId = -1;
            if (newEdge.adjcTri != nullptr) {
                newEdge.adjcTri->edges[newEdge.adjcTriEdgeId].adjcTri = newTri;
                newEdge.adjcTri->edges[newEdge.adjcTriEdgeId].adjcTriEdgeId = 0;
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

            tri->link(lastTri, 2, 1);
            tri->link(nextTri, 1, 2);
        }

        for (auto tri: influencedTris) {
            if (triNetHead == tri) {
                triNetHead = newTris[0];
                break;
            }
        }
    }

    triNetHead = triNetHead->removeBoundTriangle();

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

DelaunayTriangles::NetNode::Edge::Edge(int pointIdA, int pointIdB) {
    endpointId[0] = pointIdA;
    endpointId[1] = pointIdB;
}

void DelaunayTriangles::NetNode::_clearVisitFlag() {
    if (!_visited)
        return;
    _visited = false;
    for (auto &edge: edges) {
        if (edge.adjcTri != nullptr)
            edge.adjcTri->_clearVisitFlag();
    }
}

DelaunayTriangles::NetNode *DelaunayTriangles::NetNode::_findContainingTriangle(const Point &point) {
    if (_visited)
        return nullptr;
    _visited = true;
    if (triangleContains(points[0], points[1], points[2], point))
        return this;
    for (auto &edge: edges) {
        if (edge.adjcTri == nullptr)
            continue;
        auto r = edge.adjcTri->_findContainingTriangle(point);
        if (r != nullptr)
            return r;
    }
    return nullptr;
}

bool DelaunayTriangles::NetNode::_exCircleContain(const Point &point) {
    float dist = pointDistance(point, exCenter);
    return dist < exRadius;
}

void
DelaunayTriangles::NetNode::_findAdjcInfluencedEdges(const Point &point, int beginEdgeId, std::set<NetNode *> &tris,
                                                     std::vector<NetNode::Edge *> &edges) {
    _visited = true;
    for (int i = 0; i < 3; ++i) {
        auto &edge = this->edges[(i + beginEdgeId) % 3];
        auto adjcTri = edge.adjcTri;
        if (adjcTri == nullptr) {
            tris.insert(this);
            edges.emplace_back(&edge);
        } else if (!adjcTri->_visited) {
            if (!adjcTri->_exCircleContain(point)) {
                tris.insert(this);
                edges.emplace_back(&edge);
            } else {
                adjcTri->_findAdjcInfluencedEdges(point, edge.adjcTriEdgeId, tris, edges);
            }
        }
    }
}

DelaunayTriangles::NetNode *DelaunayTriangles::NetNode::_removeBoundTriangle() {
    if (_visited)
        return nullptr;
    _visited = true;
    NetNode *newHead = _isBoundTriangle ? nullptr : this;
    for (auto &edge: edges) {
        auto adjcTri = edge.adjcTri;
        if (adjcTri != nullptr) {
            if (_isBoundTriangle) {
                adjcTri->edges[edge.adjcTriEdgeId].adjcTri = nullptr;
                adjcTri->edges[edge.adjcTriEdgeId].adjcTriEdgeId = -1;
                edge.adjcTri = nullptr;
                edge.adjcTriEdgeId = -1;
            }
            auto r = adjcTri->_removeBoundTriangle();
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
        window.draw(vertices, 2, sf::Lines);
        window.draw(vertices + 1, 2, sf::Lines);
        window.draw(vertices + 2, 2, sf::Lines);

        if (edge.adjcTri != nullptr) {
            edge.adjcTri->_draw(window);
        }
    }
}

DelaunayTriangles::NetNode::NetNode(int pointIdA, int pointIdB, int pointIdC, const std::vector<Point> &centers, int n) {
    points[0] = centers[pointIdA];
    points[1] = centers[pointIdB];
    points[2] = centers[pointIdC];
    vertices[0] = sf::Vertex(points[0]);
    vertices[1] = sf::Vertex(points[1]);
    vertices[2] = sf::Vertex(points[2]);
    vertices[3] = sf::Vertex(points[0]);

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

void DelaunayTriangles::NetNode::findAdjcInfluencedEdges(const Point &point, std::set<NetNode *> &tris,
                                                         std::vector<NetNode::Edge *> &edges) {
    _findAdjcInfluencedEdges(point, 0, tris, edges);
    _clearVisitFlag();
}

void DelaunayTriangles::NetNode::link(DelaunayTriangles::NetNode *anoTri, int edgeId, int anoEdgeId) {
    edges[edgeId].adjcTri = anoTri;
    edges[edgeId].adjcTriEdgeId = anoEdgeId;
    anoTri->edges[anoEdgeId].adjcTri = this;
    anoTri->edges[anoEdgeId].adjcTriEdgeId = edgeId;
}

DelaunayTriangles::NetNode *DelaunayTriangles::NetNode::removeBoundTriangle() {
    auto r = _removeBoundTriangle();
    _clearVisitFlag();
    return r;
}

void DelaunayTriangles::NetNode::draw(Window &window) {
    _draw(window);
    _clearVisitFlag();
}

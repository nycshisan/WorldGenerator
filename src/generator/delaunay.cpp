//
// Created by Nycshisan on 2018/3/19.
//

#include "delaunay.h"

#include "../conf/conf.h"
#include "../misc/misc.h"

void DelaunayTriangles::input(const DelaunayTriangles::Input &input) {
    _centers = input;
}

void DelaunayTriangles::generate() {
    _deleteOldNodes();
    // Bowyer-Watson algorithm
    auto n = (int)_centers.size();
    int width = CONF.getMapWidth(), height = CONF.getMapHeight();
    _centers.emplace_back(-width, -height);
    _centers.emplace_back(3 * CONF.getMapWidth(), 0);
    _centers.emplace_back(0, 3 * CONF.getMapHeight());
    _triNetHead = new NetNode(n, n + 1, n + 2, _centers, n);
    _allocatedNodes.insert(_triNetHead);

    for (int i = 0; i < n; ++i) {
        auto center = _centers[i];
        NetNode *containingTriangle = nullptr;
        for (auto tri: _allocatedNodes) {
            if (triangleContains(tri->points[0], tri->points[1], tri->points[2], center)) {
                containingTriangle = tri;
            }
        }
        // `Influenced` triangles & edges means the triangles and edges constituting the cavity made by putting the newest center
        std::set<NetNode*> influencedTris;
        std::vector<Edge*> influencedEdges;
        containingTriangle->findInfluenced(center, influencedTris, influencedEdges);
        std::vector<NetNode*> newTris;
        for (auto &edge: influencedEdges) {
            auto *newTri = new NetNode(edge->pid[0], edge->pid[1], i, _centers, n);
            _allocatedNodes.insert(newTri);
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
            _allocatedNodes.erase(_allocatedNodes.find(tri));
            delete tri;
        }
        _triNetHead = *_allocatedNodes.begin();
    }

    _centers.pop_back(); _centers.pop_back(); _centers.pop_back();
}

DelaunayTriangles::Output DelaunayTriangles::output() {
    return _allocatedNodes;
}

void DelaunayTriangles::draw(Window &window) {
    for (auto &tri: _allocatedNodes) {
        if (!tri->_isBoundTriangle)
            window.draw(tri->_vertices, 4, sf::LineStrip);
    }
}

void DelaunayTriangles::_deleteOldNodes() {
    if (!_allocatedNodes.empty()) {
        for (auto ptr: _allocatedNodes)
            delete ptr;
        _allocatedNodes.clear();
    }
    _triNetHead = nullptr;
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

void DelaunayTriangles::NetNode::_findInfluenced(const Point &point, int beginEdgeId, std::set<NetNode*> &tris, std::vector<Edge*> &edges) {
    _visited = true;
    tris.insert(this); // Any triangle executing this function should be influenced due to the recursion condition.
    for (int i = 0; i < 3; ++i) {
        auto &edge = this->edges[(i + beginEdgeId) % 3];
        auto nextTri = edge.nextTri;
        if (nextTri == nullptr) {
            edges.emplace_back(&edge); // The outermost edge must be influenced.
        } else if (!nextTri->_visited) {
            if (pointDistance(point, nextTri->exCenter) > nextTri->exRadius) {
                edges.emplace_back(&edge); // Next triangle is not influenced, so the edge is influenced.
            } else {
                nextTri->_findInfluenced(point, edge.nextTriEdgeId, tris, edges);
            }
        }
    }
}

void DelaunayTriangles::NetNode::findInfluenced(const Point &point, std::set<NetNode*> &tris, std::vector<Edge*> &edges) {
    _findInfluenced(point, 0, tris, edges);
    _clearVisitFlag();
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

void DelaunayTriangles::NetNode::linkAnoTri(DelaunayTriangles::NetNode *anoTri, int edgeId, int anoEdgeId) {
    edges[edgeId].nextTri = anoTri;
    edges[edgeId].nextTriEdgeId = anoEdgeId;
    anoTri->edges[anoEdgeId].nextTri = this;
    anoTri->edges[anoEdgeId].nextTriEdgeId = edgeId;
}

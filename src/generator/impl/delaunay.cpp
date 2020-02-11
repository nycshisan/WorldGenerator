//
// Created by Nycshisan on 2018/3/19.
//

#include "delaunay.h"

#include "../../graphics/graphics.h"

namespace wg {

    void DelaunayTriangles::input(void* inputData) {
        _centers = *(Input*)inputData;
    }

    void DelaunayTriangles::generate() {
        // For reasons which are unknown for now (maybe because of the computation accuracy of float numbers), each time
        // we call this function will generate the triangles with different order but in almost same position.

        _deleteOldNodes();
        int newNetNodeId = 0;

        auto &allocatedNodes = _centersTris.second;

        // Bowyer-Watson algorithm
        auto n = (int) _centers.size();
        int width = CONF.getMapWidth(), height = CONF.getMapHeight();
        _centers.emplace_back(-width, -height);
        _centers.emplace_back(3 * CONF.getMapWidth(), 0);
        _centers.emplace_back(0, 3 * CONF.getMapHeight());
        _triNetHead = new NetNode(newNetNodeId++, n, n + 1, n + 2, _centers, n);
        allocatedNodes.insert(_triNetHead);

        for (int i = 0; i < n; ++i) {
            auto center = _centers[i];
            NetNode *containingTriangle = nullptr;
            for (auto tri: allocatedNodes) {
                if (center.x < tri->_minX || center.x > tri->_maxX || center.y < tri->_minY || center.y > tri->_maxY)
                    continue;
                if (tri->contains(center)) {
                    containingTriangle = tri;
                    break;
                }
            }
            assertWithSave(containingTriangle != nullptr);
            // `Influenced` triangles & edges means the triangles and edges constituting the cavity made by putting the newest center
            std::set<NetNode *> influencedTris;
            std::vector<Edge *> influencedEdges;
            containingTriangle->findInfluenced(center, influencedTris, influencedEdges);
            std::vector<NetNode *> newTris;
            for (auto &edge: influencedEdges) {
                auto *newTri = new NetNode(newNetNodeId++, edge->pid[0], edge->pid[1], i, _centers, n);
                allocatedNodes.insert(newTri);
                newTris.emplace_back(newTri);
                auto &newEdge = newTri->edges[0];
                newEdge.nextTri = edge->nextTri;
                newEdge.nextTriEdgeId = edge->nextTriEdgeId;
                edge->nextTri = nullptr;
                edge->nextTriEdgeId = -1;
                if (newEdge.nextTri != nullptr) {
                    newEdge.nextTri->edges[newEdge.nextTriEdgeId].nextTri = newTri;
                    newEdge.nextTri->edges[newEdge.nextTriEdgeId].nextTriEdgeId = 0;
                }
            }

            for (size_t j = 0; j < newTris.size(); ++j) {
                NetNode *tri = newTris[j], *lastTri, *nextTri;
                if (j == 0)
                    lastTri = newTris[newTris.size() - 1];
                else
                    lastTri = newTris[j - 1];
                if (j == newTris.size() - 1)
                    nextTri = newTris[0];
                else
                    nextTri = newTris[j + 1];

                tri->_linkAnoTri(lastTri, 2, 1);
                tri->_linkAnoTri(nextTri, 1, 2);
            }

            for (auto tri: influencedTris) {
                allocatedNodes.erase(allocatedNodes.find(tri));
                delete tri;
            }
            _triNetHead = *allocatedNodes.begin();
        }

        _centers.pop_back();
        _centers.pop_back();
        _centers.pop_back();


        _centersTris.first = _centers;
    }

    void* DelaunayTriangles::output() {
        return (void*)&_centersTris;
    }

    void DelaunayTriangles::prepareVertexes(Drawer &drawer) {
        auto &allocatedNodes = _centersTris.second;
        bool showBoundingTriangles = CONF.getDelaunayShowBoundingTriangles();
        for (auto &tri: allocatedNodes) {
            if (!tri->_isBoundingTriangle || showBoundingTriangles) {
                drawer.appendVertex(sf::Lines, tri->points[0].vertex);
                drawer.appendVertex(sf::Lines, tri->points[1].vertex);
                drawer.appendVertex(sf::Lines, tri->points[1].vertex);
                drawer.appendVertex(sf::Lines, tri->points[2].vertex);
                drawer.appendVertex(sf::Lines, tri->points[2].vertex);
                drawer.appendVertex(sf::Lines, tri->points[0].vertex);
            }
        }
    }

    void DelaunayTriangles::_deleteOldNodes() {
        auto &allocatedNodes = _centersTris.second;
        if (!allocatedNodes.empty()) {
            for (auto ptr: allocatedNodes)
                delete ptr;
            allocatedNodes.clear();
        }
        _triNetHead = nullptr;
    }

    DelaunayTriangles::~DelaunayTriangles() {
        _deleteOldNodes();
    }

    std::string DelaunayTriangles::getHintLabelText() {
        return "Generated Delaunay triangles.";
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

    void DelaunayTriangles::NetNode::_findInfluenced(const Point &point, int beginEdgeId, std::set<NetNode *> &tris,
                                                     std::vector<Edge *> &influencedEdges) {
        _visited = true;
        tris.insert(this); // Any triangle executing this function should be influenced due to the recursion condition.
        for (int i = 0; i < 3; ++i) {
            auto &edge = this->edges[(i + beginEdgeId) % 3];
            auto nextTri = edge.nextTri;
            if (nextTri == nullptr) {
                influencedEdges.emplace_back(&edge); // The outermost edge must be influenced.
            } else if (!nextTri->_visited) {
                if (point.squareDistance(nextTri->exCenter) > nextTri->exRadius) {
                    influencedEdges.emplace_back(&edge); // Next triangle is not influenced, so the edge is influenced.
                } else {
                    nextTri->_findInfluenced(point, edge.nextTriEdgeId, tris, influencedEdges);
                }
            }
        }
    }

    void DelaunayTriangles::NetNode::findInfluenced(const Point &point, std::set<NetNode *> &tris,
                                                    std::vector<Edge *> &influencedEdges) {
        _findInfluenced(point, 0, tris, influencedEdges);
        _clearVisitFlag();
    }

    DelaunayTriangles::NetNode::NetNode(int id, int pointIdA, int pointIdB, int pointIdC,
                                        const std::vector<Point> &centers, int n) : Triangle(centers[pointIdA],
                                                                                             centers[pointIdB],
                                                                                             centers[pointIdC]) {
        this->id = id;

        edges[0] = Edge(pointIdA, pointIdB);
        edges[1] = Edge(pointIdB, pointIdC);
        edges[2] = Edge(pointIdC, pointIdA);
        exCenter = getExCenter();
        exRadius = exCenter.squareDistance(points[0]);

        _isBoundingTriangle = pointIdA >= n || pointIdB >= n || pointIdC >= n;
        auto minmaxX = std::minmax({centers[pointIdA].x, centers[pointIdB].x, centers[pointIdC].x});
        auto minmaxY = std::minmax({centers[pointIdA].y, centers[pointIdB].y, centers[pointIdC].y});
        _minX = minmaxX.first;
        _maxX = minmaxX.second;
        _minY = minmaxY.first;
        _maxY = minmaxY.second;
    }

    void DelaunayTriangles::NetNode::_linkAnoTri(DelaunayTriangles::NetNode *anoTri, int edgeId, int anoEdgeId) {
        edges[edgeId].nextTri = anoTri;
        edges[edgeId].nextTriEdgeId = anoEdgeId;
        anoTri->edges[anoEdgeId].nextTri = this;
        anoTri->edges[anoEdgeId].nextTriEdgeId = edgeId;
    }

}


//
// Created by Nycshisan on 2018/3/15.
//

#include "voronoi.h"

#include <random>
#include <set>

#include "../conf/conf.h"

void VoronoiDiagram::init(unsigned int width, unsigned int height) {
    _width = width; _height = height;
    _pointShape.setRadius(CONF["ui"]["pointRadius"].GetFloat());
    _pointShape.setPointCount(5);

}

void VoronoiDiagram::generateCenters() {
    int n = CONF["voronoi"]["blockNumber"].GetInt();
    int padding = CONF["voronoi"]["padding"].GetInt();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> disX(padding, _width - padding), disY(padding, _height - padding);

    _centers.clear();
    for (int i = 0; i < n; ++i) {
        int x = disX(gen);
        int y = disY(gen);
        _centers.emplace_back(Point(x, y));
    }
}

struct VDDelaunayTriangleNetNode {
    struct Edge {
        int endpointId[2] = {-1, -1};
        VDDelaunayTriangleNetNode *adjcTri = nullptr;
        int adjcTriEdgeId = -1;

        Edge() = default;

        Edge(int pointIdA, int pointIdB) {
            endpointId[0] = pointIdA;
            endpointId[1] = pointIdB;
        }
    };

private:
    ~VDDelaunayTriangleNetNode() {
        for (auto &edge : edges) {
            delete edge.adjcTri;
            edge.adjcTri = nullptr;
            edge.adjcTriEdgeId = -1;
        }
    };

    bool _visited = false;
    void _clearVisitFlag() {
        if (!_visited)
            return;
        _visited = false;
        for (auto &edge: edges) {
            if (edge.adjcTri != nullptr)
                edge.adjcTri->_clearVisitFlag();
        }
    }

    VDDelaunayTriangleNetNode *_findContainingTriangle(const Point &point) {
        if (_visited)
            return nullptr;
        _visited = true;
        if (triangleContains(points[0], points[1], points[2], point))
            return this;
        for (auto &edge: edges) {
            auto r = edge.adjcTri->_findContainingTriangle(point);
            if (r != nullptr)
                return r;
        }
        return nullptr;
    }

    bool _exCircleContain(const Point &point) {
        float dist = pointDistance(point, exCenter);
        return dist < exRadius;
    }

    void _findAdjcInfluencedEdges(std::vector<std::pair<VDDelaunayTriangleNetNode*, Edge>> &result, const Point &point, int beginEdgeId) {
        _visited = true;
        for (int i = 0; i < 3; ++i) {
            auto &edge = edges[(i + beginEdgeId) % 3];
            auto adjcTri = edge.adjcTri;
            if (adjcTri == nullptr) {
                result.emplace_back(std::pair<VDDelaunayTriangleNetNode*, Edge>(this, edge));
            } else if (!adjcTri->_visited) {
                if (!adjcTri->_exCircleContain(point)) {
                    result.emplace_back(std::pair<VDDelaunayTriangleNetNode*, Edge>(this, edge));
                } else {
                    adjcTri->_findAdjcInfluencedEdges(result, point, edge.adjcTriEdgeId);
                }
            }
        }
    }


    void _draw(Window &window) {
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

public:
    Edge edges[3];
    Point points[3];
    sf::Vertex vertices[4];

    Point exCenter;
    float exRadius = 0;

    VDDelaunayTriangleNetNode(int pointIdA, int pointIdB, int pointIdC, const std::vector<Point> &centers) {
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
    }

    VDDelaunayTriangleNetNode *findContainingTriangle(const Point &point) {
        auto result = _findContainingTriangle(point);
        _clearVisitFlag();
        return result;
    }

    std::vector<std::pair<VDDelaunayTriangleNetNode*, Edge>> findAdjcInfluencedEdges(const Point &point) {
        std::vector<std::pair<VDDelaunayTriangleNetNode*, Edge>> result;
        _findAdjcInfluencedEdges(result, point, 0);
        _clearVisitFlag();
        return result;
    }

    void link(VDDelaunayTriangleNetNode *anoTri, int edgeId, int anoEdgeId) {
        edges[edgeId].adjcTri = anoTri;
        edges[edgeId].adjcTriEdgeId = anoEdgeId;
        anoTri->edges[anoEdgeId].adjcTri = this;
        anoTri->edges[anoEdgeId].adjcTriEdgeId = edgeId;
    }

    void destroyNet() {
        delete this;
    }

    void destroySelf() {
        for (auto &edge : edges) {
            if (edge.adjcTri != nullptr) {
                edge.adjcTri->edges[edge.adjcTriEdgeId].adjcTri = nullptr;
                edge.adjcTri->edges[edge.adjcTriEdgeId].adjcTriEdgeId = -1;
            }
            edge.adjcTri = nullptr;
            edge.adjcTriEdgeId = -1;
        }
        destroyNet();
    }

    void draw(Window &window) {
        _draw(window);
        _clearVisitFlag();
    }
};

void VoronoiDiagram::generateDelaunayTriangles() {
    // Bowyer-Watson algorithm
    auto n = (int)_centers.size();
    _centers.emplace_back(0, 0);
    _centers.emplace_back(2 * _width, 0);
    _centers.emplace_back(0, 2 * _height);
    triNetHead = new VDDelaunayTriangleNetNode(n, n + 1, n + 2, _centers);

    for (int i = 0; i < n; ++i) {
        auto center = _centers[i];
        auto containingTriangle = triNetHead->findContainingTriangle(center);
        auto influencedEdges = containingTriangle->findAdjcInfluencedEdges(center);
        std::vector<VDDelaunayTriangleNetNode*> newTris;
        for (auto &pair: influencedEdges) {
            auto &edge = pair.second;
            auto *newTri = new VDDelaunayTriangleNetNode(edge.endpointId[0], edge.endpointId[1], i, _centers);
            auto &newEdge = newTri->edges[0];
            newEdge.adjcTri = edge.adjcTri;
            newEdge.adjcTriEdgeId = edge.adjcTriEdgeId;
            edge.adjcTri = nullptr; edge.adjcTriEdgeId = -1;
            if (newEdge.adjcTri != nullptr) {
                newEdge.adjcTri->edges[newEdge.adjcTriEdgeId].adjcTri = newTri;
                newEdge.adjcTri->edges[newEdge.adjcTriEdgeId].adjcTriEdgeId = 0;
            }
        }

        for (int j = 0; j < newTris.size(); ++j) {
            VDDelaunayTriangleNetNode *tri = newTris[j], *lastTri, *nextTri;
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

        std::set<VDDelaunayTriangleNetNode*> influencedTris;
        for (auto &pair: influencedEdges) {
            influencedTris.insert(pair.first);
        }
        for (auto tri: influencedTris) {
            tri->destroySelf();
        }
    }

    //triNetHead->destroyNet();
}

void VoronoiDiagram::drawPointsToWindow(Window &window) {
    for (auto point : _centers) {
        _pointShape.setPosition(point);
        window.draw(_pointShape);
    }
}

void VoronoiDiagram::drawTrianglesToWindow(Window &window) {
    triNetHead->draw(window);
}

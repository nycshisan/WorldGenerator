//
// Created by Nycshisan on 2018/3/19.
//

#ifndef WORLDGENERATOR_DELAUNAY_H
#define WORLDGENERATOR_DELAUNAY_H

#include <set>

#include "../graphics/window.h"
#include "centers.h"

class DelaunayTriangles {
public:
    class NetNode;
    struct Edge {
        int pid[2] = {-1, -1};
        NetNode *nextTri = nullptr;
        int nextTriEdgeId = -1;

        Edge() = default;
        Edge(int pointIdA, int pointIdB);
    };

    class NetNode {
        friend class DelaunayTriangles;

        bool _visited = false;
        void _clearVisitFlag();

        NetNode *_findContainingTriangle(const Point &point);
        NetNode* findContainingTriangle(const Point &point);

        void _findInfluenced(const Point &point, int beginEdgeId, std::set<NetNode *> &tris, std::vector<Edge *> &edges);
        void findInfluenced(const Point &point, std::set<NetNode *> &tris, std::vector<Edge *> &edges);

        bool _isBoundTriangle;
        NetNode *_removeBoundingTriangle();
        NetNode *removeBoundingTriangle();

        sf::Vertex _vertices[4];
        void _draw(Window &window);
        void draw(Window &window);

        void linkAnoTri(NetNode *anoTri, int edgeId, int anoEdgeId);

        NetNode(int pointIdA, int pointIdB, int pointIdC, const std::vector<Point> &centers, int n);

    public:
        Edge edges[3];
        Point points[3];

        Point exCenter;
        float exRadius = 0;
    };

    typedef BlockCenters::Output Input;
    typedef NetNode* Output;
private:
    Input _centers;
    Output triNetHead = nullptr;

    std::vector<NetNode*> _allocatedNodes;
    void _deleteOldNodes();

public:
    void init(const Input &input);
    void generate();
    Output output();
    void draw(Window &window);

    ~DelaunayTriangles();
};

#endif //WORLDGENERATOR_DELAUNAY_H

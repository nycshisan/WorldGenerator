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
    struct NetNode {
        struct Edge {
            int endpointId[2] = {-1, -1};
            NetNode *adjcTri = nullptr;
            int adjcTriEdgeId = -1;

            Edge() = default;

            Edge(int pointIdA, int pointIdB);
        };

    private:
        bool _visited = false;
        void _clearVisitFlag();

        NetNode *_findContainingTriangle(const Point &point);

        bool _exCircleContain(const Point &point);

        void _findAdjcInfluencedEdges(const Point &point, int beginEdgeId, std::set<NetNode*> &tris, std::vector<NetNode::Edge*> &edges);

        bool _isBoundTriangle;

        NetNode* _removeBoundTriangle();

        void _draw(Window &window);

    public:
        Edge edges[3];
        Point points[3];
        sf::Vertex vertices[4];

        Point exCenter;
        float exRadius = 0;

        NetNode(int pointIdA, int pointIdB, int pointIdC, const std::vector<Point> &centers, int n);

        NetNode* findContainingTriangle(const Point &point);

        void findAdjcInfluencedEdges(const Point &point, std::set<NetNode*> &tris, std::vector<NetNode::Edge*> &edges);

        void link(NetNode *anoTri, int edgeId, int anoEdgeId);

        NetNode* removeBoundTriangle();

        void draw(Window &window);
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
